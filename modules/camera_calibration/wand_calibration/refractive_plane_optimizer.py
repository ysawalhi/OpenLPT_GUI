"""
Refractive Plane Optimizer (P1)

This module implements Phase P1 optimization for refractive window planes.
All camera parameters (intrinsics, extrinsics) remain FIXED.

Key Design Decisions:
1. Outer-Loop Lambda Adaptation: Lambda is constant within each least_squares call
   to ensure stable Jacobian computation. Lambda is updated only between solver
   calls to target an energy ratio of ~1.0 between ray and length residuals.

2. Joint Triangulation: For each frame and endpoint, a single global X is
   triangulated using rays from ALL active cameras (each using its own window
   plane). This ensures proper constraints even for single-camera windows.

3. Staged Optimization:
   - P1.1: 1D optimization of d per window
   - P1.2: 3D optimization of (d, alpha, beta) per window
   - P1.3: Joint optimization of all windows simultaneously
"""

import numpy as np
import time
import cv2
from scipy.optimize import least_squares
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import pyopenlpt as lpt

from .refractive_geometry import (
    Ray, normalize, build_pinplate_ray_cpp, triangulate_point, point_to_ray_dist,
    update_normal_tangent, rodrigues_to_R, camera_center, angle_between_vectors,
    optical_axis_world, update_cpp_camera_state,
    enable_ray_tracking, reset_ray_stats, print_ray_stats_report
)
from .refractive_constraints import compute_soft_barrier_penalty


@dataclass
class OptimizationConfig:
    """Configuration for P1 optimization."""
    # Regularization
    lambda_reg: float = 10.0  # Normal drift penalty weight
    
    # Unit Normalization
    px_target: float = 0.5            # 投影误差目标 (px)
    wand_tol_pct: float = 0.02        # 棒长容忍度 (2% 即 0.2mm)

    # Bounds
    alpha_beta_bound: float = 0.5  # ~28.6 degrees in radians
    margin_side_mm: float = 0.5  # Soft barrier margin
    
    # Subset
    max_frames: int = 50000  # Default to use all frames (high limit)
    
    # Logging
    verbosity: int = 1  # 0=clean, 1=summary, 2=full audit


class RefractivePlaneOptimizer:
    """
    Optimizer for refractive window plane parameters.
    
    Optimizes plane distance (d) and normal (n) for each window while keeping
    camera parameters fixed. Uses joint triangulation across all cameras.
    """
    
    def __init__(self, 
                 dataset: Dict,
                 cam_params: Dict[int, np.ndarray],
                 cams_cpp: Dict[int, 'lpt.Camera'],
                 cam_to_window: Dict[int, int],
                 window_media: Dict[int, Dict],
                 window_planes: Dict[int, Dict],
                 wand_length: float,
                 config: Optional[OptimizationConfig] = None,
                 progress_callback=None):
        """
        Initialize the optimizer.
        
        Args:
            dataset: Observation data with 'obsA', 'obsB', 'frames' keys
            cam_params: Dict mapping cam_id to parameter array [rvec(3), tvec(3), ...]
            cams_cpp: Dict mapping cam_id to pyopenlpt.Camera objects
            cam_to_window: Dict mapping cam_id to window_id
            window_media: Dict mapping window_id to media properties
            window_planes: Dict mapping window_id to {'plane_pt', 'plane_n'}
            wand_length: Target wand length in mm
            config: Optimization configuration
            progress_callback: Optional callback(phase, ray, len, cost)
        """
        self.dataset = dataset
        self.cam_params = cam_params
        self.cams_cpp = cams_cpp
        self.cam_to_window = cam_to_window
        self.window_media = window_media
        self.window_planes = {wid: {
            'plane_pt': pl['plane_pt'].copy(),
            'plane_n': pl['plane_n'].copy()
        } for wid, pl in window_planes.items()}
        self.wand_length = wand_length
        self.config = config if config else OptimizationConfig()
        self.progress_callback = progress_callback
        
        self.sigma_ray_global = 0.04  # Default, will be recalculated
        self.sigma_wand = 0.1        # Default, will be recalculated
        
        # Derived data
        self.active_cam_ids = list(cam_params.keys())
        self.window_ids = sorted(set(self.window_planes.keys()))
        self.n_windows = len(self.window_ids)
        
        # Build observation cache: {frame_id: {cam_id: (uvA, uvB)}}
        self._build_obs_cache()
        
        # Build window -> cameras mapping
        self.win_cams = {wid: [] for wid in self.window_ids}
        for cid in self.active_cam_ids:
            wid = cam_to_window.get(cid, 0)
            if wid in self.win_cams:
                self.win_cams[wid].append(cid)
        
        # [USER REQUEST] Re-sort window_ids by camera count (descending) 
        # to prioritize windows with more observations during per-window optimization.
        self.window_ids = sorted(self.window_ids, key=lambda wid: len(self.win_cams[wid]), reverse=True)
        
        # Compute anchors (mean camera center per window)
        self.win_anchors = {}
        for wid in self.window_ids:
            centers = []
            for cid in self.win_cams[wid]:
                p = cam_params[cid]
                R = rodrigues_to_R(p[0:3])
                C = camera_center(R, p[3:6])
                centers.append(C)
            if centers:
                self.win_anchors[wid] = np.mean(centers, axis=0)
            else:
                self.win_anchors[wid] = np.zeros(3)
        
        # Store initial normals for regularization
        self.initial_normals = {wid: pl['plane_n'].copy() 
                                for wid, pl in self.window_planes.items()}
        
        # Select frame subset
        import random
        rng = random.Random(42)
        all_frames = list(dataset.get('frames', range(len(dataset.get('obsA', [])))))
        subset_size = min(len(all_frames), self.config.max_frames)
        self.fids_optim = sorted(rng.sample(all_frames, subset_size))
        
    def _build_obs_cache(self):
        """Build observation cache from dataset."""
        self.obs_cache = {}
        obsA_raw = self.dataset.get('obsA', {})
        obsB_raw = self.dataset.get('obsB', {})
        
        # obsA and obsB are dicts: {fid: {cid: (u, v, ...)}}
        all_fids = sorted(list(set(obsA_raw.keys()) | set(obsB_raw.keys())))
        
        for fid in all_fids:
            self.obs_cache[fid] = {}
            for cid in self.active_cam_ids:
                uvA = None
                uvB = None
                if fid in obsA_raw and cid in obsA_raw[fid]:
                    pt = obsA_raw[fid][cid]
                    if pt is not None and len(pt) >= 2 and np.all(np.isfinite(pt[:2])):
                        uvA = pt[:2]
                if fid in obsB_raw and cid in obsB_raw[fid]:
                    pt = obsB_raw[fid][cid]
                    if pt is not None and len(pt) >= 2 and np.all(np.isfinite(pt[:2])):
                        uvB = pt[:2]
                if uvA is not None or uvB is not None:
                    self.obs_cache[fid][cid] = (uvA, uvB)
        
        self.fids_all = sorted(list(self.obs_cache.keys()))
        # self.fids_optim will be set later based on config.max_frames
        
    def _compute_physical_sigmas(self):
        """
        Estimate global sigma_ray and sigma_wand based on camera parameters and targets.
        Calculated once before each optimization stage.
        """
        cfg = self.config
        
        # 1. Calculate Average Focal Length (fx)
        all_f = []
        for cid in self.cam_params:
            all_f.append(self.cam_params[cid][6]) # Assuming index 6 is fx
        avg_f = np.mean(all_f) if all_f else 1000.0
        
        # 2. Calculate Average Object Distance Z
        sample_dists = []
        # Use first few frames of obs_cache to find 3D points
        fids_to_sample = self.fids_optim[::max(1, len(self.fids_optim) // 100)] if len(self.fids_optim) > 0 else []
        for fid in fids_to_sample:
            rays_A, rays_B = self._build_rays_frame(fid)
            for rays in [rays_A, rays_B]:
                valid_rays = [r for r in rays if r.valid]
                if len(valid_rays) >= 2:
                    X, _, ok, _ = triangulate_point(valid_rays)
                    if ok:
                        for r in valid_rays:
                            cid = r.cam_id
                            rv = self.cam_params[cid][0:3]
                            tv = self.cam_params[cid][3:6]
                            R = cv2.Rodrigues(rv)[0]
                            C = -R.T @ tv
                            sample_dists.append(np.linalg.norm(X - C))
                            
        avg_dist_z = np.mean(sample_dists) if sample_dists else 600.0
        
        # 3. Final Sigmas
        self.sigma_ray_global = cfg.px_target * (avg_dist_z / avg_f)
        self.sigma_wand = self.wand_length * cfg.wand_tol_pct
        
        if cfg.verbosity >= 1:
            print(f"  [P1] Unit Normalization: sigma_ray={self.sigma_ray_global:.4f}mm ({cfg.px_target}px at Z={avg_dist_z:.1f}mm), sigma_wand={self.sigma_wand:.4f}mm ({cfg.wand_tol_pct*100}%)")

    def _print_plane_diagnostics(self, stage_name: str):
        """Print current plane normals and angles between them."""
        print(f"\n  [{stage_name}] Plane Diagnostics:")
        wids = sorted(self.window_planes.keys())
        normals = []
        for wid in wids:
            n = self.window_planes[wid]['plane_n']
            pt = self.window_planes[wid]['plane_pt']
            normals.append(n)
            print(f"    Win {wid}: n=[{n[0]:.4f}, {n[1]:.4f}, {n[2]:.4f}], pt=[{pt[0]:.2f}, {pt[1]:.2f}, {pt[2]:.2f}]")
        
        if len(normals) == 2:
            ang = angle_between_vectors(normals[0], normals[1])
            print(f"    Angle between Win 0 and Win 1: {ang:.2f}°")
    
    def _apply_planes_to_cpp(self, planes: Dict[int, Dict]):
        """Apply plane parameters to all C++ camera objects."""
        for wid, pl in planes.items():
            pt_list = pl['plane_pt'].tolist()
            n_list = pl['plane_n'].tolist()
            for cid in self.win_cams.get(wid, []):
                if cid in self.cams_cpp:
                    update_cpp_camera_state(self.cams_cpp[cid], plane_geom={'pt': pt_list, 'n': n_list})
    


    def _build_rays_frame(self, fid: int) -> Tuple[List[Ray], List[Ray]]:
        """
        Build rays for a frame using JOINT triangulation approach.
        Returns (rays_A, rays_B) where each list contains rays from ALL cameras.
        """
        rays_A = []
        rays_B = []
        
        if fid not in self.obs_cache:
            return rays_A, rays_B
        
        # Use sorted CID to ensure stable ordering in residual vector
        for cid in sorted(self.obs_cache[fid].keys()):
            if cid not in self.cams_cpp:
                continue
            cam_ref = self.cams_cpp[cid]
            wid = self.cam_to_window.get(cid, -1)
            
            uvA, uvB = self.obs_cache[fid][cid]
            
            # NOTE: uvA/uvB should never be None if in cache, but we check for safety
            rA = build_pinplate_ray_cpp(cam_ref, uvA if uvA is not None else np.zeros(2), 
                                        cam_id=cid, window_id=wid, frame_id=fid, endpoint="A")
            if uvA is None: rA.valid = False
            
            rB = build_pinplate_ray_cpp(cam_ref, uvB if uvB is not None else np.zeros(2), 
                                        cam_id=cid, window_id=wid, frame_id=fid, endpoint="B")
            if uvB is None: rB.valid = False
            
            rays_A.append(rA)
            rays_B.append(rB)
        
        return rays_A, rays_B
    
    def evaluate_residuals(self, planes: Dict[int, Dict], lambda_eff: float,
                           include_reg: bool = True, mode: str = 'joint',
                           status_desc: Optional[str] = None
                           ) -> Tuple[np.ndarray, float, float, int, int, List[dict]]:
        """
        Evaluate residuals for given plane parameters.
        
        Returns:
            (residuals, S_ray, S_len_unweighted, N_ray, N_len, valid_points)
        """
        # Apply planes to C++ objects
        # print(f"[DEBUG] Entering evaluate_residuals (mode={mode})")
        self._apply_planes_to_cpp(planes)
        # print(f"[DEBUG] _apply_planes_to_cpp done")
        
        res_ray = []
        res_len = []
        valid_points = []
        
        radius_A = self.dataset.get('est_radius_small_mm', 0.0)
        radius_B = self.dataset.get('est_radius_large_mm', 0.0)
        
        # [CRITICAL] Penalties for invalid states to maintain FIXED length residual vector
        PENALTY_RAY = 100.0   # mm
        PENALTY_LEN = self.wand_length
        
        for fid in self.fids_optim:
            # Pump events deep inside loop (critical for responsiveness)
            if hasattr(self, 'progress_callback') and self.progress_callback:
                now = time.time()
                last_pump = getattr(self, '_last_pump_time', 0.0)
                if now - last_pump > 0.1:
                    self._last_pump_time = now
                    # Yield GIL to let Main Thread process GUI events
                    time.sleep(0)
                    try:
                        # Use provided status description or default
                        desc = status_desc if status_desc else "Calculating Errors..."
                        self.progress_callback(desc, -1.0, -1.0, -1.0)
                    except:
                        pass
            
            rays_A_all, rays_B_all = self._build_rays_frame(fid)
            
            # Triangulate A
            validA = False
            XA = None
            rays_A_valid = [r for r in rays_A_all if r.valid]
            if len(rays_A_valid) >= 2:
                XA, _, validA, _ = triangulate_point(rays_A_valid)
            
            if validA:
                for r in rays_A_all:
                    if r.valid:
                        d = point_to_ray_dist(XA, r.o, r.d)
                        res_ray.append(d / self.sigma_ray_global)
                    else:
                        res_ray.append(PENALTY_RAY / self.sigma_ray_global)
                wids = set(r.window_id for r in rays_A_all if r.window_id != -1)
                valid_points.append({'fid': fid, 'X': XA, 'wids': wids, 'ep': 'A'})
            else:
                for _ in rays_A_all:
                    res_ray.append(PENALTY_RAY / self.sigma_ray_global)

            # Triangulate B
            validB = False
            XB = None
            rays_B_valid = [r for r in rays_B_all if r.valid]
            if len(rays_B_valid) >= 2:
                XB, _, validB, _ = triangulate_point(rays_B_valid)

            if validB:
                for r in rays_B_all:
                    if r.valid:
                        d = point_to_ray_dist(XB, r.o, r.d)
                        res_ray.append(d / self.sigma_ray_global)
                    else:
                        res_ray.append(PENALTY_RAY / self.sigma_ray_global)
                wids = set(r.window_id for r in rays_B_all if r.window_id != -1)
                valid_points.append({'fid': fid, 'X': XB, 'wids': wids, 'ep': 'B'})
            else:
                for _ in rays_B_all:
                    res_ray.append(PENALTY_RAY / self.sigma_ray_global)
            
            # Wand Length
            if validA and validB:
                dist = np.linalg.norm(XA - XB)
                err = dist - self.wand_length
                res_len.append(err / self.sigma_wand)
            else:
                res_len.append(PENALTY_LEN / self.sigma_wand)
        
        # Compute statistics (Only on VALID entries for logging)
        # However, for Scipy, we return everything.
        arr_ray = np.array(res_ray)
        arr_len = np.array(res_len)
        
        # For stats reporting, we filter out penalty values? 
        # Actually, evaluate_residuals is also used for printing RMSE.
        # We should compute "Real" RMSE separate from "Solver" residual.
        real_res_ray = arr_ray[arr_ray < PENALTY_RAY * 0.9 / self.sigma_ray_global]
        real_res_len = arr_len[arr_len < PENALTY_LEN * 0.9 / self.sigma_wand]
        
        S_ray = np.sum(real_res_ray**2) if len(real_res_ray) > 0 else (PENALTY_RAY**2) * len(arr_ray)
        S_len = np.sum(real_res_len**2) if len(real_res_len) > 0 else (PENALTY_LEN**2) * len(arr_len)
        N_ray = len(real_res_ray)
        N_len = len(real_res_len)
        
        # Build weighted residual vector
        residuals = arr_ray.copy()
        
        if len(res_len) > 0:
            weighted_len = np.sqrt(lambda_eff) * arr_len
            residuals = np.concatenate([residuals, weighted_len])
        
        # [USER REQUEST] Hard Barrier Penalties (Always-On, like PR5 final round)
        # Weight = alpha_side * J_data / num_valid, where alpha_side = 10.0 (same as PR5)
        J_data = S_ray + lambda_eff * S_len
        
        # To keep length constant, we pre-calculate number of samples for weight
        # But for the residual vector itself, we must iterate frames and windows.
        num_valid = len(valid_points)
        alpha_side = 10.0  # Hard penalty coefficient (same as PR5 alpha_side_gate)
        w_side = alpha_side * J_data / max(1, num_valid)
        
        res_barrier = []
        margin_mm = self.config.margin_side_mm
        
        # Build map: (fid, ep) -> (X, wids)
        vp_map = {(d['fid'], d['ep']): (d['X'], d['wids']) for d in valid_points}
        
        # CRITICAL: We iterate over ALL fids and ALL endpoints to keep length CONSTANT
        for fid in self.fids_optim:
            for ep in ['A', 'B']:
                X, wids = vp_map.get((fid, ep), (None, None))
                # For each window, we must have a slot
                for wid in self.window_ids:
                    # [USER REQUEST] Global Barrier: Check against ALL windows, not just observers
                    if X is not None:
                        pl = planes[wid]
                        r_val = radius_A if ep == 'A' else radius_B
                        
                        # Soft Barrier Check
                        p_side, _ = compute_soft_barrier_penalty(
                            X, pl['plane_pt'], pl['plane_n'], 
                            w_side=w_side, 
                            sigma=0.01,
                            R_mm=r_val,
                            margin_mm=margin_mm
                        )
                        res_barrier.append(p_side)
                    else:
                        res_barrier.append(0.0)
        
        if res_barrier:
            arr_barrier = np.array(res_barrier)
            residuals = np.concatenate([residuals, arr_barrier])
        
        return residuals, S_ray, S_len, N_ray, N_len, valid_points
    
    def _joint_residual_func(self, x: np.ndarray, lambda_eff: float) -> np.ndarray:
        """
        Residual function for joint optimization.
        x = [d0, a0, b0, d1, a1, b1, ...] for all windows
        """
        # Unpack parameters
        planes = {}
        for i, wid in enumerate(self.window_ids):
            d_val = x[3*i]
            alpha = x[3*i + 1]
            beta = x[3*i + 2]
            
            n_base = self.initial_normals[wid]
            n_new = update_normal_tangent(n_base, alpha, beta)
            anchor = self.win_anchors[wid]
            plane_pt = anchor + n_new * d_val
            
            planes[wid] = {'plane_pt': plane_pt, 'plane_n': n_new}
        
        residuals, S_ray, S_len, N_ray, N_len, _ = self.evaluate_residuals(planes, lambda_eff, 
                                                         include_reg=True, mode='joint',
                                                         status_desc="Optimizing All Window parameters...")
        
        # Pump events and report progress
        if hasattr(self, 'progress_callback') and self.progress_callback:
            now = time.time()
            last_time = getattr(self, '_last_update_time', 0.0)
            if now - last_time > 0.1:
                self._last_update_time = now
                try:
                    # Compute metrics
                    rmse_ray = np.sqrt(S_ray / max(1, N_ray)) * self.sigma_ray_global
                    rmse_len = np.sqrt(S_len / max(1, N_len)) * self.sigma_wand if N_len > 0 else 0.0
                    cost = 0.5 * (S_ray + lambda_eff * S_len)
                    
                    self.progress_callback(f"P1.3: Optimizing All Window parameters...", rmse_ray, rmse_len, cost)
                except Exception as e:
                    print(f"[CRITICAL] Callback Error (Joint): {e}", flush=True)
                
        return residuals

    
    def _single_window_residual_func(self, x: np.ndarray, target_wid: int, 
                                      mode: str, lambda_eff: float) -> np.ndarray:
        """
        Residual function for single-window optimization.
        x = [d] for 1D mode, [d, alpha, beta] for 3D mode
        """
        d_val = x[0]
        anchor = self.win_anchors[target_wid]
        n_base = self.initial_normals[target_wid]
        
        if mode == '1D_d':
            n_new = n_base
        else:  # 3D_full
            alpha, beta = x[1], x[2]
            n_new = update_normal_tangent(n_base, alpha, beta)
        
        plane_pt = anchor + n_new * d_val
        
        # Create temporary planes dict (update only target window)
        planes = {wid: {'plane_pt': pl['plane_pt'].copy(), 'plane_n': pl['plane_n'].copy()}
                  for wid, pl in self.window_planes.items()}
        planes[target_wid] = {'plane_pt': plane_pt, 'plane_n': n_new}
        
        # Include regularization only for 3D mode
        include_reg = (mode == '3D_full')
        
        # Prepare status string
        if mode == '1D_d':
            status = f"Adjusting Window {target_wid} Distance..."
        else:
            status = f"Adjusting Window {target_wid} Angle..."

        residuals, S_ray, S_len, N_ray, N_len, _ = self.evaluate_residuals(planes, lambda_eff, 
                                                         include_reg=include_reg, mode=mode,
                                                         status_desc=status)
        
        # Pump events and report progress
        if hasattr(self, 'progress_callback') and self.progress_callback:
            now = time.time()
            last_time = getattr(self, '_last_update_time', 0.0)
            if now - last_time > 0.1:
                self._last_update_time = now
                try:
                    # Compute metrics
                    rmse_ray = np.sqrt(S_ray / max(1, N_ray)) * self.sigma_ray_global
                    rmse_len = np.sqrt(S_len / max(1, N_len)) * self.sigma_wand if N_len > 0 else 0.0
                    cost = 0.5 * (S_ray + lambda_eff * S_len)
                    
                    if mode == '1D_d':
                        self.progress_callback(f"Adjusting Window {target_wid} Distance...", rmse_ray, rmse_len, cost)
                    else:
                        self.progress_callback(f"Adjusting Window {target_wid} Angle...", rmse_ray, rmse_len, cost)
                except Exception as e:
                    print(f"[CRITICAL] Callback Error (Single): {e}", flush=True)

        return residuals

    
    def optimize_per_window(self) -> Dict[int, Dict]:
        """
        P1.1 and P1.2: Per-window optimization with FIXED weighting.
        
        For each window:
        - P1.1: 1D optimization of d only
        - P1.2: 3D optimization of (d, alpha, beta)
        
        Uses fixed lambda = 2.0 * N_cams.
        """
        self._compute_physical_sigmas()
        enable_ray_tracking(True, reset=True)
        print("\n" + "="*50)
        print("[Refractive][P1] Starting Per-Window Plane Optimization")
        
        cfg = self.config
        
        # [USER REQUEST] Fixed Weighting Strategy
        # Lambda = 2.0 * N_active_cams
        # This balances ~2N ray residuals vs 1 length residual
        n_cams = max(1, len(self.active_cam_ids))
        lambda_fixed = 2.0 * n_cams
        
        print(f"    Global Fixed Weighting: lambda={lambda_fixed:.1f} (N_cams={n_cams})")
        print(f"    Normal Constraint: alpha_beta_bound={cfg.alpha_beta_bound:.6f} rad ({np.degrees(cfg.alpha_beta_bound):.2f}°)")

        # Get wand radii for sanity checks
        radii = {
            'A': self.dataset.get('est_radius_small_mm', 0.0),
            'B': self.dataset.get('est_radius_large_mm', 0.0)
        }
        
        for wid in self.window_ids:
            if not self.win_cams.get(wid):
                print(f"\n    Window {wid}: [SKIP] No cameras")
                continue
            
            print(f"\n    === Window {wid} ===")
            
            # [USER REQUEST] Get contemporary 3D points for d_max
            _, _, _, _, _, valid_points_curr = self.evaluate_residuals(
                self.window_planes, lambda_fixed, include_reg=False, status_desc=f"Updating points for Window {wid}...")
            
            # Get initial parameters
            pl = self.window_planes[wid]
            n_init = pl['plane_n']
            anchor = self.win_anchors[wid]
            d_init = np.dot(n_init, pl['plane_pt'] - anchor)
            
            # Bounds
            thick = self.window_media.get(wid, {}).get('thickness', 10.0)
            d_min = max(1.0 * thick, 20.0)
            
            # [USER REQUEST] d_max = min(dist(anchor, X_i) - radius_j) - 0.05
            # We use the LATEST refracted 3D points instead of bootstrap pinhole points
            d_max = 2500.0
            rA = radii.get('A', 0.0)
            rB = radii.get('B', 0.0)
            margin = 0.05
            
            d_adj_list = []
            for vp in valid_points_curr:
                X = vp['X']
                ep = vp['ep']
                # distance to anchor of CURRENT window
                dist = np.linalg.norm(X - anchor)
                r_val = rA if ep == 'A' else rB
                d_adj_list.append(dist - r_val - margin)
            
            if d_adj_list:
                d_max_calc = min(d_adj_list)
                d_max = min(2500.0, d_max_calc)
                print(f"    d_max_calc: {d_max_calc:.1f} mm (min adjustment from {len(d_adj_list)} current pts)")
                if d_max < d_min + 5.0:
                    d_max = d_min + 50.0 # Safety fallback if points are somehow behind plane
                    print(f"    [WARNING] d_max too small, using fallback: {d_max:.1f}")
            else:
                # Fallback to bootstrap if triangulation fails (unlikely)
                print(f"    [WARNING] No current 3D points found. Falling back to bootstrap for d_max.")
                xa_boot = self.dataset.get('X_A_bootstrap', {})
                xb_boot = self.dataset.get('X_B_bootstrap', {})
                # ... (keep old logic as fallback if needed, but usually 3D points exist)
                d_max = 2500.0 # simplified fallback
            
            print(f"    Bounds: [d_min={d_min:.1f}, d_max={d_max:.1f}] (d_init={d_init:.1f})")
            
            # ===== P1.1: 1D Optimization (d only) =====
            print(f"    [P1.1] 1D Optimize d (init={d_init:.1f})")
            
            x_1d = np.array([d_init])
            x_1d = np.clip(x_1d, d_min, d_max)

            # Single Pass Optimization
            if self.progress_callback:
                try:
                    self.progress_callback(f"Adjusting Window {wid} Distance...", 0, 0, 0)
                except:
                    pass

            result = least_squares(
                lambda x: self._single_window_residual_func(x, wid, '1D_d', lambda_fixed),
                x_1d,
                bounds=(np.array([d_min]), np.array([d_max])),
                loss='huber', f_scale=1.0,
                verbose=0,
                max_nfev=50,
                ftol=1e-6, xtol=1e-6, gtol=1e-6
            )
            d_opt_1d = result.x[0]
            print(f"      -> d_opt: {d_opt_1d:.2f} mm (cost: {result.cost:.4f})")
            
            # Update planes with 1D result
            old_pt = self.window_planes[wid]['plane_pt'].copy()
            old_n = self.window_planes[wid]['plane_n'].copy()
            self.window_planes[wid]['plane_pt'] = anchor + n_init * d_opt_1d
            
            
            # ===== P1.2: 3D Optimization [d, alpha, beta] =====
            print(f"    [P1.2] 3D Optimize [d, alpha, beta]")
            
            lb = np.array([d_min, -cfg.alpha_beta_bound, -cfg.alpha_beta_bound])
            ub = np.array([d_max, cfg.alpha_beta_bound, cfg.alpha_beta_bound])
            x_3d = np.array([d_opt_1d, 0.0, 0.0])
            x_3d = np.clip(x_3d, lb, ub)
            
            # Single Pass Optimization
            if self.progress_callback:
                try:
                    self.progress_callback(f"Adjusting Window {wid} Angle...", 0, 0, 0)
                except:
                    pass
            
            result = least_squares(
                lambda x: self._single_window_residual_func(x, wid, '3D_full', lambda_fixed),
                x_3d,
                bounds=(lb, ub),
                loss='huber', f_scale=1.0,
                verbose=0,
                max_nfev=50,
                ftol=1e-6, xtol=1e-6, gtol=1e-6
            )

            d_final, a_final, b_final = result.x
            n_final = update_normal_tangent(n_init, a_final, b_final)
            pt_final = anchor + n_final * d_final
            
            # Report
            angle = angle_between_vectors(n_final, n_init)
            print(f"      -> d={d_final:.2f}, alpha={a_final:.4f}, beta={b_final:.4f} (cost: {result.cost:.4f})")
            print(f"      -> n_new: {n_final.round(4)}, angle_change: {angle:.2f}°")
            
            # Commit
            old_pt = self.window_planes[wid]['plane_pt'].copy()
            old_n = self.window_planes[wid]['plane_n'].copy()
            self.window_planes[wid] = {'plane_pt': pt_final, 'plane_n': n_final}
            
            
            # Update initial_normals for P1.3 regularization baseline
            self.initial_normals[wid] = n_final.copy()
            
        print("[Refractive][P1] Per-window optimization complete.")
        self._print_plane_diagnostics("P1.2 Per-Window End")
        print_ray_stats_report("P1.1/P1.2 Per-Window")
        enable_ray_tracking(False)
        return self.window_planes
    
    def optimize_joint(self) -> Dict[int, Dict]:
        """
        P1.3: Joint optimization of all windows with FIXED weighting.
        """
        self._compute_physical_sigmas()
        enable_ray_tracking(True, reset=True)
        print("\n" + "="*50)
        print("[Refractive][P1] Starting Joint Window Optimization (P1.3)")
        print(f"\n  [P1.3] Joint Optimization (All {self.n_windows} Windows)")
        
        cfg = self.config
        
        # Get wand radii for sanity checks
        radii = {
            'A': self.dataset.get('est_radius_small_mm', 0.0),
            'B': self.dataset.get('est_radius_large_mm', 0.0)
        }
        
        if self.progress_callback:
             try:
                 self.progress_callback("P1.3: Optimizing All Window parameters...", 0, 0, 0)
             except:
                 pass

        # [USER REQUEST] Fixed Weighting Strategy
        # Lambda = 2.0 * N_active_cams
        n_cams = max(1, len(self.active_cam_ids))
        lambda_fixed = 2.0 * n_cams
        
        # Initial evaluation just for logging / valid points
        _, S_ray_init, S_len_init, N_ray, N_len, valid_points_init = self.evaluate_residuals(
            self.window_planes, lambda_fixed, include_reg=False, status_desc="Initializing Joint Opt...")
        
        print(f"    Global Fixed Weighting: lambda={lambda_fixed:.1f} (N_cams={n_cams})")
        print(f"    Initial: S_ray={S_ray_init:.4f}, S_len={S_len_init:.4f}")
        
        # [USER REQUEST] Contemporary 3D points for Joint d_max bounds
        x0 = []
        lb = []
        ub = []
        for wid in self.window_ids:
            pl = self.window_planes[wid]
            n_init = pl['plane_n']
            anchor = self.win_anchors[wid]
            d_init = np.dot(n_init, pl['plane_pt'] - anchor)
            
            thick = self.window_media.get(wid, {}).get('thickness', 10.0)
            d_min = max(1.0 * thick, 20.0)
            
            d_max = 2500.0
            rA = radii.get('A', 0.0)
            rB = radii.get('B', 0.0)
            margin = 0.05
            
            d_adj_list = [np.linalg.norm(vp['X'] - anchor) - (rA if vp['ep'] == 'A' else rB) - margin 
                          for vp in valid_points_init]
            if d_adj_list:
                d_max = min(2500.0, min(d_adj_list))
                if d_max < d_min + 5.0: d_max = d_min + 50.0
                
            x0.extend([d_init, 0.0, 0.0])
            lb.extend([d_min, -cfg.alpha_beta_bound, -cfg.alpha_beta_bound])
            ub.extend([d_max, cfg.alpha_beta_bound, cfg.alpha_beta_bound])
        
        x0 = np.array(x0)
        bounds = (np.array(lb), np.array(ub))
        
        # Single Pass Optimization
        x_current = x0.copy()
        
        if self.progress_callback:
            try:
                self.progress_callback(f"Optimizing All Window parameters...", 0, 0, 0)
            except:
                pass

        result = least_squares(
            lambda x: self._joint_residual_func(x, lambda_fixed),
            x_current,
            bounds=bounds,
            loss='huber', f_scale=1.0,
            verbose=0,
            max_nfev=50,
            ftol=1e-6, xtol=1e-6, gtol=1e-6
        )
        x_current = result.x.copy()
        
        # Reconstruct planes from solution
        planes = {}
        for i, wid in enumerate(self.window_ids):
            d_val = x_current[3*i]
            alpha = x_current[3*i + 1]
            beta = x_current[3*i + 2]
            
            n_base = self.initial_normals[wid]
            n_new = update_normal_tangent(n_base, alpha, beta)
            anchor = self.win_anchors[wid]
            plane_pt = anchor + n_new * d_val
            
            planes[wid] = {'plane_pt': plane_pt, 'plane_n': n_new}
        
        # Final Report
        _, S_ray, S_len, _, _, _ = self.evaluate_residuals(planes, lambda_fixed, include_reg=False,
                                                      status_desc="Optimizing All Window parameters...")
        if self.progress_callback:
            rmse_ray_audit = np.sqrt(S_ray / max(1, N_ray))
            rmse_len_audit = np.sqrt(S_len / max(1, N_len)) if N_len > 0 else 0.0
            self.progress_callback(f"Optimizing All Window parameters...", rmse_ray_audit, rmse_len_audit, result.cost)
        
        # Final result
        final_planes = {}
        for i, wid in enumerate(self.window_ids):
            d_val = x_current[3*i]
            alpha = x_current[3*i + 1]
            beta = x_current[3*i + 2]
            
            n_base = self.initial_normals[wid]
            n_new = update_normal_tangent(n_base, alpha, beta)
            anchor = self.win_anchors[wid]
            plane_pt = anchor + n_new * d_val
            
            final_planes[wid] = {'plane_pt': plane_pt, 'plane_n': n_new}
            
            # Report per-window results
            angle = angle_between_vectors(n_new, self.initial_normals[wid])
            print(f"\n    Window {wid}: d={d_val:.2f}mm, alpha={alpha:.4f}, beta={beta:.4f}")
            print(f"      Normal change: {angle:.2f}°")
            print(f"      n_new: {n_new.round(4)}")
        
        # Update internal state
        old_planes = {w: {'plane_pt': self.window_planes[w]['plane_pt'].copy(), 
                          'plane_n': self.window_planes[w]['plane_n'].copy()} 
                      for w in self.window_ids}
                      
        self.window_planes = final_planes
        
        print("[Refractive][P1] P1.3 Joint optimization complete.")
        self._print_plane_diagnostics("P1.3 Joint End")
        print_ray_stats_report("P1.3 Joint")
        enable_ray_tracking(False)
        
        # Sanity Check
        radii = {
            'A': self.dataset.get('est_radius_small_mm', 0.0),
            'B': self.dataset.get('est_radius_large_mm', 0.0)
        }

        
        return final_planes
    
    def print_diagnostics(self):
        """Print comprehensive diagnostics after optimization."""
        print("\n  === P1 Optimization Diagnostics ===")
        
        # Apply final planes
        self._apply_planes_to_cpp(self.window_planes)
        
        # Collect per-camera and per-window residuals
        cam_residuals_A = {cid: [] for cid in self.active_cam_ids}
        cam_residuals_B = {cid: [] for cid in self.active_cam_ids}
        wand_lengths = []
        
        for fid in self.fids_optim:
            rays_A, rays_B = self._build_rays_frame(fid)
            
            validA, validB = False, False
            XA, XB = None, None
            
            if len(rays_A) >= 2:
                XA, _, validA, _ = triangulate_point(rays_A)
                if validA:
                    for r in rays_A:
                        d = point_to_ray_dist(XA, r.o, r.d)
                        cam_residuals_A[r.cam_id].append(d)
            
            if len(rays_B) >= 2:
                XB, _, validB, _ = triangulate_point(rays_B)
                if validB:
                    for r in rays_B:
                        d = point_to_ray_dist(XB, r.o, r.d)
                        cam_residuals_B[r.cam_id].append(d)
            
            if validA and validB:
                wand_lengths.append(np.linalg.norm(XA - XB))
        
        # Per-camera statistics
        print("\n  Per-Camera Ray Residuals (mm):")
        for cid in sorted(self.active_cam_ids):
            resA = cam_residuals_A[cid]
            resB = cam_residuals_B[cid]
            wid = self.cam_to_window.get(cid, 0)
            
            if resA:
                med_A = np.median(resA)
                mean_A = np.mean(resA)
                rmse_A = np.sqrt(np.mean(np.array(resA)**2))
            else:
                med_A = mean_A = rmse_A = 0.0
            
            if resB:
                med_B = np.median(resB)
                mean_B = np.mean(resB)
                rmse_B = np.sqrt(np.mean(np.array(resB)**2))
            else:
                med_B = mean_B = rmse_B = 0.0
            
            print(f"    Cam {cid} (Win {wid}): A[med={med_A:.3f}, rmse={rmse_A:.3f}] "
                  f"B[med={med_B:.3f}, rmse={rmse_B:.3f}]")
        
        # Per-window aggregate
        print("\n  Per-Window Ray Residuals (mm):")
        for wid in self.window_ids:
            win_res = []
            for cid in self.win_cams.get(wid, []):
                win_res.extend(cam_residuals_A.get(cid, []))
                win_res.extend(cam_residuals_B.get(cid, []))
            if win_res:
                med = np.median(win_res)
                p90 = np.percentile(win_res, 90)
                rmse = np.sqrt(np.mean(np.array(win_res)**2))
                print(f"    Window {wid}: median={med:.3f}, p90={p90:.3f}, rmse={rmse:.3f}")
        
        # Wand length statistics
        if wand_lengths:
            arr = np.array(wand_lengths)
            err = arr - self.wand_length
            print(f"\n  Wand Length Statistics (mm):")
            print(f"    Mean:   {np.mean(arr):.3f} (target: {self.wand_length:.2f})")
            print(f"    Median: {np.median(arr):.3f}")
            print(f"    RMSE:   {np.sqrt(np.mean(err**2)):.3f}")
            print(f"    Bias:   {np.mean(err):.3f}")
            print(f"    p90 err: {np.percentile(np.abs(err), 90):.3f}")
        
        # Angle diagnostics
        print("\n  Angle Diagnostics:")
        for wid in self.window_ids:
            pl = self.window_planes[wid]
            n_w = pl['plane_n']
            plane_pt = pl['plane_pt']
            
            print(f"    Window {wid}:")
            for cid in self.win_cams.get(wid, []):
                p = self.cam_params[cid]
                R = rodrigues_to_R(p[0:3])
                C = camera_center(R, p[3:6])
                
                # View direction to plane
                v_to_plane = normalize(plane_pt - C)
                angle_view = angle_between_vectors(n_w, v_to_plane)
                
                # Optical axis angle
                z_world = optical_axis_world(R)
                angle_optical = angle_between_vectors(n_w, z_world)
                
                print(f"      Cam {cid}: n vs view_to_plane={angle_view:.1f}°, "
                      f"n vs optical_axis={angle_optical:.1f}°")
