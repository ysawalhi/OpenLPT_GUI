"""
Refractive PR4 Bundle Adjustment Optimizer

This module implements Phase PR4 optimization: Selective Bundle Adjustment
that refines both window planes AND selected camera extrinsics (when observable).

Key Design Principles:
- Observability-based freezing: N_cam, baseline, view-angle diversity
- Staged optimization: PR4.1 → PR4.2 → PR4.3
- Intrinsics ALWAYS fixed
- Safe defaults: freeze rvec unless geometry is strong

Freeze Semantics (OPTIMIZE = move, FREEZE = fixed):
- N_cam = 1: Plane OPTIMIZE (strong reg), tvec FREEZE, rvec FREEZE
- N_cam ≥ 2, weak: Plane OPTIMIZE, tvec OPTIMIZE (small TR), rvec FREEZE
- N_cam ≥ 2, good: Plane OPTIMIZE, tvec OPTIMIZE, rvec OPTIMIZE

Rotation Thresholds:
- theta_enable_rot = 20°: Below → FREEZE rvec
- theta_strong_rot = 35°: Above → full rvec, weaker reg
- Baseline guard: If baseline < 10mm, keep rvec heavily damped
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set
from enum import Enum
import cv2
from scipy.optimize import least_squares
import json
import os
from pathlib import Path

from .refractive_constraints import (
    PlaneOrderConfig, 
    PointSideConfig,
    compute_plane_order_penalties,
    compute_point_side_penalty,
    compute_camera_side_penalty,
    compute_soft_barrier_penalty,
    clamp_ray_parameter,
    print_plane_side_verification
)


try:
    import pyopenlpt as lpt
except ImportError:
    lpt = None

from .refractive_geometry import (
    Ray, normalize, build_pinplate_ray_cpp, triangulate_point, point_to_ray_dist,
    point_to_ray_dist_vec,  # Vectorized version for performance
    update_normal_tangent, rodrigues_to_R, camera_center, angle_between_vectors,

    optical_axis_world
)
from datetime import datetime
from pathlib import Path
import os

def evaluate_plane_side_constraints(dataset, window_planes, cam_params, cams_cpp, cam_to_window, R_small_mm, R_large_mm, margin_mm):
    """
    Unified feasibility evaluator for PR5 / Polish.
    Returns: min_sX, violations_list, worst_viol, frame_details, min_dist
    """
    # Requires re-tracing logic. Can we reuse a simplified trace loop?
    # Yes, we need a simplified version of _trace_rays_all but standalone.
    # Or we can make this a method of a base class?
    # Standalone is cleaner but duplicates trace logic.
    # Let's rely on cams_cpp being UP-TO-DATE with cam_params/planes.
    # CALLER MUST ENSURE cams_cpp IS SYNCED.
    
    min_sX = 999.0
    min_dist = 999.0
    violations = []
    worst_viol = None
    frame_details = {} # fid -> {max_gap, min_g, best_g?}
    
    # Pre-fetch plane data
    w_data = {}
    for wid, pl in window_planes.items():
        w_data[wid] = {
            'n': np.array(pl['plane_n']),
            'pt': np.array(pl['plane_pt'])
        }
        
    for fid in dataset['frames']:
        # For efficiency, we can group rays? No, just iter.
        # k=0 (A), k=1 (B)
        for k, (key, radius_base) in enumerate([('obsA', R_small_mm), ('obsB', R_large_mm)]):
             if fid not in dataset[key]: continue
             
             limit = radius_base + margin_mm
             
             # Group observations by wid to triangulate?
             # No, standard triangulation needs mult-cam.
             # We need to collect rays for this point (fid, k).
             
             rays = []
             obs_map = dataset[key][fid]
             
             for cid, uv in obs_map.items():
                 if cid not in cam_to_window: continue
                 wid = cam_to_window[cid]
                 cam = cams_cpp[cid]
                 
                 # Trace
                 label = "A" if k == 0 else "B"
                 # Ensure build_pinplate_ray_cpp is available
                 r = build_pinplate_ray_cpp(
                     cam, uv, 
                     cam_id=cid, window_id=wid, frame_id=fid, endpoint=label
                 )
                 if r.valid:
                     rays.append(r)
             
             if len(rays) < 2: continue
             
             # Triangulate
             X, _, success, _ = triangulate_point(rays)
             if not success: continue
             
             # Check constraints for ALL involved windows
             involved_wids = set(r.window_id for r in rays)
             
             for wid in involved_wids:
                 if wid not in w_data: continue
                 pl = w_data[wid]
                 sX = np.dot(pl['n'], X - pl['pt'])
                 
                 gap = limit - sX # positive means violation
                 # User defined: g = sX - limit. g < 0 is violation.
                 # Let's stick to g = sX - limit.
                 g = sX - limit
                 
                 if sX < min_dist: min_dist = sX
                 if g < min_sX: min_sX = g
                 
                 if g < -0.001: # Violation threshold (hardcoded small epsilon for reporting equality)
                     v_entry = {
                         'frame': fid,
                         'cam': -1, # Point-based
                         'window': wid,
                         'endpoint': 'A' if k==0 else 'B',
                         'sX': sX,
                         'limit': limit,
                         'g': g,
                         'gap': -g
                     }
                     violations.append(v_entry)
                     if worst_viol is None or g < worst_viol['g']:
                         worst_viol = v_entry
                          
                 # Collect per-frame stats for injection
                 gap_val = gap # limit - sX
                 g_val = g     # sX - limit
                 
                 if fid not in frame_details:
                     frame_details[fid] = {'frame': fid, 'max_gap': -999.0, 'min_g': 999.0}
                 
                 if gap_val > frame_details[fid]['max_gap']:
                     frame_details[fid]['max_gap'] = gap_val
                     frame_details[fid]['worst_info'] = f"W{wid} {label}"
                     
                 if g_val < frame_details[fid]['min_g']:
                     frame_details[fid]['min_g'] = g_val

    return min_sX, violations, worst_viol, frame_details, min_dist

def analyze_worst_frames(frame_details):
    """
    Process frame_details to get sorted worst and near frames.
    """
    if not frame_details:
        return [], []
        
    # Sort by gap (descending) -> Most violating first
    # gap = limit - sX. larger gap = worse. 
    # frame_details is dict: fid -> {max_gap, min_g, ...}
    
    worst_list = [] # Frames with gap > 0 (violation)
    near_list = []  # All frames, sorted by g (closest to violating)
    
    for fid, stats in frame_details.items():
        if stats['max_gap'] > 0.001:
             worst_list.append(stats)
        
        near_list.append(stats)
        
    worst_sorted = sorted(worst_list, key=lambda x: x['max_gap'], reverse=True)
    # Near: Sort by g ascending. Smallest g is most negative (worst violation) or smallest positive (closest safety).
    near_sorted = sorted(near_list, key=lambda x: x['min_g'])
    
    return worst_sorted, near_sorted


class FreezeStatus(Enum):
    """Status for each parameter DOF."""
    OPTIMIZE = "OPTIMIZE"
    FREEZE = "FREEZE"
    OPTIMIZE_REGULARIZED = "OPTIMIZE_REG"
    OPTIMIZE_STRONG = "OPT_STRONG"


@dataclass
class ObservabilityInfo:
    """Per-window observability analysis."""
    window_id: int
    n_cam: int  # Number of cameras viewing this window
    camera_ids: List[int] = field(default_factory=list)
    baseline_max_mm: float = 0.0  # Max baseline between camera pairs
    baseline_median_mm: float = 0.0
    angle_diversity_p50: float = 0.0  # degrees, median pairwise view angle
    angle_diversity_p90: float = 0.0  # degrees, 90th percentile
    
    # Freeze decisions
    plane_status: FreezeStatus = FreezeStatus.OPTIMIZE
    tvec_status: FreezeStatus = FreezeStatus.FREEZE
    rvec_status: FreezeStatus = FreezeStatus.FREEZE
    freeze_reason: str = ""


@dataclass
class PR4Config:
    """Configuration for PR4 Bundle Adjustment."""
    # Lambda adaptation (same as P1)
    lambda0_init: float = 200.0
    lambda_min: float = 10.0
    lambda_max: float = 5000.0
    target_ratio: float = 1.0
    adaptation_eta: float = 0.30
    deadband_low: float = 0.7
    deadband_high: float = 1.5
    outer_rounds: int = 3
    
    # Regularization
    lambda_reg_plane: float = 10.0  # Normal drift penalty
    lambda_reg_tvec: float = 1.0    # Translation drift penalty
    lambda_reg_rvec: float = 50.0   # Rotation drift penalty (standard)
    
    # Strong rvec constraints (15-20 deg)
    allow_weak_rvec: bool = True    # If True, allow OPT_STRONG for 15-20 deg
    prior_lambda_rot_base: float = 1.0
    prior_p_rot: float = 2.0
    step_cap_rot_deg_strong: float = 0.1
    step_cap_rot_deg_weak: float = 0.5
    step_cap_rot_deg_normal: float = 2.0
    
    # Tvec step caps (mm per round)
    step_cap_tvec_weak_mm: float = 0.5   # For STRONG/REGULARIZED
    step_cap_tvec_normal_mm: float = 1.0 # For normal OPTIMIZE

    
    # Observability thresholds
    theta_freeze: float = 15.0        # Hard freeze below this
    theta_enable_rot: float = 20.0    # Normal optimization above this
    theta_strong_rot: float = 35.0    # Very strong diversity
    baseline_guard_mm: float = 10.0   # Below this, keep rvec heavily damped
    
    # Bounds
    alpha_beta_bound: float = 0.5  # radians
    tvec_bound: float = 50.0       # mm
    rvec_bound: float = 0.1        # radians (~5.7 degrees)
    
    # Sampling
    max_frames: int = 300
    random_seed: int = 42
    
    # Stage control
    skip_pr4: bool = False
    pr4_stage: int = 3
    verbosity: int = 1
    margin_side_mm: float = 0.05    # Margin for soft barrier (mm)
    
    
@dataclass
class PR5Config:
    """
    Configuration for PR5 Robust Bundle Adjustment.
    Geometric-only optimization (Ray + Wand) with Strong Priors.
    """
    pr5_stage: int = 2  # 1=BA, 2=Final Joint
    verbosity: int = 1
    
    # Robust Loss (Huber)
    delta_ray: float = 0.30
    delta_len: float = 0.15
    
    # Lambda Weights
    lambda_ray: float = 1.0     # Base weight for ray residuals
    lambda_len_init: float = 200.0  # Initial weight for length residuals
    lambda_len_min: float = 50.0
    lambda_len_max: float = 2000.0
    
    # Side Gate (Hysteresis-Based Feasibility Constraint)
    # Stable round-level gate with zero cost when feasible
    # Side Gate (Hysteresis-Based Feasibility Constraint)
    # Stable round-level gate with zero cost when feasible
    margin_side_mm: float = 0.05     # Margin for side constraint (mm); sX < margin is violation
    v_on_side_gate: float = 0.010    # Turn gate ON if v > 0.010
    v_off_side_gate: float = 0.0     # Turn gate OFF if v <= 0.0 (strict feasibility)
    alpha_side_gate: float = 10.0    # Gate magnitude: C_gate = alpha * J_ref
    beta_side_dir: float = 1e4       # Directional weight when gate ON
    beta_side_soft: float = 100.0    # Soft floor weight when gate OFF
    scale_len_gate_active: float = 0.1 # Multiply lambda_len by this when gate is ON (reduce dominance)
    
    # Bounds (Percentage)
    bounds_thick_pct: float = 0.05  # +/- 5%
    bounds_f_pct: float = 0.02      # +/- 2%
    bounds_alpha_beta_deg: float = 5.0
    bounds_d_delta_mm: float = 20.0
    
    # Sampling
    max_frames: int = 50000  # Default to all (high limit)
    random_seed: int = 42
    
    # Prior Weights (Lambda)
    # Note: Lambda usually means weight in least squares sum(w * r^2).
    # Here sigma is provided. weight = (1/sigma)^2.
    # Residual = (val - init) / sigma.
    # We will compute sigmas dynamically based on Observability.
    
    # Configuration for Priors (Sigmas)
    sigma_plane_ang_single: float = 0.0035  # ~0.2 deg
    sigma_plane_ang_weak: float = 0.006     # ~0.35 deg (<15 deg)
    sigma_plane_ang_mid: float = 0.010      # ~0.6 deg (<25 deg)
    sigma_plane_ang_strong: float = 0.026   # ~1.5 deg (>=25 deg)
    
    sigma_d_single: float = 1.0
    sigma_d_weak: float = 2.0  # <25 deg
    sigma_d_strong: float = 5.0 # >= 25 deg
    
    sigma_rvec_very_weak: float = 0.0087 # ~0.5 deg (<15)
    sigma_rvec_weak: float = 0.0175      # ~1.0 deg (<25)
    sigma_rvec_normal: float = 0.035     # ~2.0 deg (>=25)
    
    sigma_tvec_weak: float = 2.0    # N=2 & <25
    sigma_tvec_normal: float = 5.0
    
    # Step Caps
    step_cap_rvec_very_weak_deg: float = 0.3
    step_cap_rvec_weak_deg: float = 0.3
    step_cap_rvec_normal_deg: float = 0.6
    
    step_cap_tvec_weak_mm: float = 0.5
    step_cap_tvec_normal_mm: float = 1.0
    
    allow_weak_rvec: bool = True
    pr4_stage: int = 3
    
    # Logging
    verbosity: int = 1  # 0=clean, 1=summary+tables, 2=full audit  # Max stage to run (1, 2, or 3)


class RefractiveBAOptimizerPR4:
    """
    PR4 Bundle Adjustment Optimizer.
    
    Refines window planes AND selected camera extrinsics based on observability.
    """
    
    def __init__(self,
                 dataset: Dict,
                 cam_params: Dict[int, np.ndarray],
                 cams_cpp: Dict,
                 cam_to_window: Dict[int, int],
                 window_media: Dict[int, Dict],
                 window_planes: Dict[int, Dict],
                 wand_length: float,
                 config: Optional[PR4Config] = None,
                 progress_callback: Optional[callable] = None):
        """
        Initialize PR4 optimizer.
        
        Args:
            dataset: Observation data with 'obsA', 'obsB', 'frames' keys
            cam_params: Dict mapping cam_id to parameter array [rvec(3), tvec(3), ...]
            cams_cpp: Dict mapping cam_id to C++ Camera objects
            cam_to_window: Dict mapping cam_id to window_id
            window_media: Dict with window properties (thickness, n_obj, etc.)
            window_planes: Dict with plane parameters (plane_n, plane_pt)
            wand_length: Target wand length in mm
            config: Optimization configuration
            progress_callback: Optional callback(phase, ray_rmse, len_rmse, cost) for UI updates
        """
        self.dataset = dataset
        self.cam_params = {int(k): np.array(v, dtype=np.float64) for k, v in cam_params.items()}
        self.cams_cpp = cams_cpp
        self.cam_to_window = {int(k): int(v) for k, v in cam_to_window.items()}
        self.window_media = window_media
        self.wand_length = wand_length
        self.config = config or PR4Config()
        self.progress_callback = progress_callback  # For UI progress updates

        
        # Deep copy window_planes for modification
        self.window_planes = {}
        for wid, pl in window_planes.items():
            self.window_planes[int(wid)] = {
                'plane_pt': np.array(pl['plane_pt'], dtype=np.float64),
                'plane_n': np.array(pl['plane_n'], dtype=np.float64)
            }
        
        # Store initial values for regularization
        self.initial_planes = {wid: {
            k: (v.copy() if hasattr(v, 'copy') else v) for k, v in pl.items()
        } for wid, pl in self.window_planes.items()}
        
        self.initial_cam_params = {cid: p.copy() for cid, p in self.cam_params.items()}
        
        # Derived data
        self.window_ids = sorted(self.window_planes.keys())
        self.active_cam_ids = sorted(self.cam_params.keys())
        
        # Build per-window camera lists
        self.window_to_cams = {wid: [] for wid in self.window_ids}
        for cid, wid in self.cam_to_window.items():
            if cid in self.active_cam_ids and wid in self.window_to_cams:
                self.window_to_cams[wid].append(cid)
        
        # Build observation cache
        self._build_obs_cache()
        
        # Frame sampling
        all_frames = sorted(self.dataset.get('frames', []))
        import random
        rng = random.Random(self.config.random_seed)
        subset_size = min(len(all_frames), self.config.max_frames)
        self.fids_optim = sorted(rng.sample(all_frames, subset_size))
        
        # Observability analysis
        self.observability: Dict[int, ObservabilityInfo] = {}
        self._compute_observability()
        
        # Freeze table (computed from observability)
        self.freeze_table: Dict = {}
        self._build_freeze_table()
        
        # Active freeze table pointer (for staged optimization context switching)
        self._freeze_table_active = self.freeze_table
        
        self._last_ray_rmse = -1.0
        self._last_len_rmse = -1.0
    
    def _rvec_step_cap_deg(self, status) -> float:
        """Get step-cap in degrees for a given rvec freeze status.
        
        STRONG/REGULARIZED: need more constraint (smaller step)
        OPTIMIZE: normal constraint (larger step)
        FREEZE: return 0.0 (should not be called for frozen params)
        
        Also applies _step_cap_multiplier if set (from conditional accept).
        """
        cfg = self.config
        if status == FreezeStatus.OPTIMIZE_STRONG:
            # STRONG = needs stronger constraint, smaller step
            base = cfg.step_cap_rot_deg_weak
        elif status == FreezeStatus.OPTIMIZE_REGULARIZED:
            # REGULARIZED = needs constraint, smaller step
            base = cfg.step_cap_rot_deg_weak
        elif status == FreezeStatus.OPTIMIZE:
            # Normal = well-conditioned, larger step OK
            base = cfg.step_cap_rot_deg_normal
        elif status == FreezeStatus.FREEZE:
            return 0.0  # Should not happen if logic is correct
        else:
            base = cfg.step_cap_rot_deg_normal  # Fallback
        
        # Apply multiplier if set (from conditional accept in rollback logic)
        multiplier = getattr(self, '_step_cap_multiplier', 1.0)
        return base * multiplier


    
    def _tvec_step_cap_mm(self, status) -> float:
        """Get step-cap in mm for a given tvec freeze status.
        
        STRONG/REGULARIZED: need more constraint (smaller step)
        OPTIMIZE: normal constraint (larger step)
        """
        cfg = self.config
        if status in [FreezeStatus.OPTIMIZE_STRONG, FreezeStatus.OPTIMIZE_REGULARIZED]:
            return cfg.step_cap_tvec_weak_mm
        elif status == FreezeStatus.OPTIMIZE:
            return cfg.step_cap_tvec_normal_mm
        elif status == FreezeStatus.FREEZE:
            return 0.0
        else:
            return cfg.step_cap_tvec_normal_mm  # Fallback
    
    def _build_obs_cache(self):
        """Build observation cache from dataset."""
        self.obs_cache = {}
        obsA = self.dataset.get('obsA', {})
        obsB = self.dataset.get('obsB', {})
        
        all_fids = set(obsA.keys()) | set(obsB.keys())
        
        for fid in all_fids:
            self.obs_cache[fid] = {}
            for cid in self.active_cam_ids:
                uvA = None
                uvB = None
                if fid in obsA and cid in obsA[fid]:
                    pt = obsA[fid][cid]
                    if pt is not None and len(pt) >= 2 and np.all(np.isfinite(pt[:2])):
                        uvA = pt[:2]
                if fid in obsB and cid in obsB[fid]:
                    pt = obsB[fid][cid]
                    if pt is not None and len(pt) >= 2 and np.all(np.isfinite(pt[:2])):
                        uvB = pt[:2]
                if uvA is not None or uvB is not None:
                    self.obs_cache[fid][cid] = (uvA, uvB)
    
    def _compute_observability(self):
        """Compute observability metrics for each window."""
        cfg = self.config
        
        for wid in self.window_ids:
            cams = self.window_to_cams.get(wid, [])
            n_cam = len(cams)
            
            info = ObservabilityInfo(
                window_id=wid,
                n_cam=n_cam,
                camera_ids=cams.copy()
            )
            
            if n_cam < 2:
                info.freeze_reason = f"N_cam={n_cam} (single camera, insufficient baseline)"
                self.observability[wid] = info
                continue
            
            # Compute baselines with explicit constraints
            baselines = []
            baseline_pairs_str = []
            camera_centers = {}
            
            # Helper to log centers
            # Only print if requested (verbose debug)
            # print(f"  [DEBUG] Window {wid} Camera Geometries:")
            
            for cid in cams:
                if cid in self.cam_params:
                    # Extract R, t from params
                    p = self.cam_params[cid]
                    rvec = p[0:3]
                    tvec = p[3:6]
                    R, _ = cv2.Rodrigues(rvec)
                    C = camera_center(R, tvec)
                    camera_centers[cid] = C
                    # print(f"    Cam {cid}: tvec={tvec}, Center={C}")

            for i, cid1 in enumerate(cams):
                for cid2 in cams[i+1:]:
                    if cid1 in camera_centers and cid2 in camera_centers:
                        C1 = camera_centers[cid1]
                        C2 = camera_centers[cid2]
                        b = np.linalg.norm(C1 - C2)
                        
                        # Constraints
                        if b < 0: 
                            b = 0.0
                        
                        baselines.append(b)
                        baseline_pairs_str.append(f"({cid1}-{cid2}: {b:.2f}mm)")
            
            if baselines:
                info.baseline_max_mm = max(baselines)
                info.baseline_median_mm = np.median(baselines)
                # Verbosity 2: Audit baselines
                if cfg.verbosity >= 2:
                    print(f"  [DEBUG] Window {wid} Cams: {cams}")
                    print(f"  [DEBUG] Window {wid} Baselines: {', '.join(baseline_pairs_str)}")
                    print(f"  [DEBUG] Window {wid} Stats: max={info.baseline_max_mm:.2f}mm, median={info.baseline_median_mm:.2f}mm")
            
            # Compute view-angle diversity
            # Mean optical axis direction per camera, then pairwise angles
            plane_n = self.window_planes[wid]['plane_n']
            view_dirs = []
            
            # Keep track of IDs to map back to pairs
            dir_cam_ids = []
            
            for cid in cams:
                if cid in self.cam_params:
                    # Optical axis in world frame (Z-axis)
                    p = self.cam_params[cid]
                    rvec = p[0:3]
                    R, _ = cv2.Rodrigues(rvec)
                    axis = optical_axis_world(R)
                    view_dirs.append(axis)
                    dir_cam_ids.append(cid)
            
            if len(view_dirs) >= 2:
                angles = []
                angle_pairs_str = []
                
                for i, d1 in enumerate(view_dirs):
                    for j, d2 in enumerate(view_dirs[i+1:]):
                        real_j = i + 1 + j
                        cid1 = dir_cam_ids[i]
                        cid2 = dir_cam_ids[real_j]
                        
                        # angle_between_vectors returns DEGREES
                        ang = angle_between_vectors(d1, d2)
                        
                        # Sanity check
                        ang = np.clip(ang, 0.0, 180.0)
                            
                        angles.append(ang)
                        angle_pairs_str.append(f"({cid1}-{cid2}: {ang:.1f}deg)")
                
                # Verbosity 2: Audit angles
                if cfg.verbosity >= 2:
                    print(f"  [DEBUG] Window {wid} Angles: {', '.join(angle_pairs_str)}")
                
                if angles:
                    info.angle_diversity_p50 = np.percentile(angles, 50)
                    info.angle_diversity_p90 = np.percentile(angles, 90)
            
            # Compute view-angle diversity
            # Mean optical axis direction per camera, then pairwise angles
            plane_n = self.window_planes[wid]['plane_n']
            view_dirs = []
            
            # Keep track of IDs to map back to pairs
            dir_cam_ids = []
            
            for cid in cams:
                if cid in self.cam_params:
                    # Optical axis in world frame
                    p = self.cam_params[cid]
                    rvec = p[0:3]
                    R, _ = cv2.Rodrigues(rvec)
                    axis = optical_axis_world(R)
                    view_dirs.append(axis)
                    dir_cam_ids.append(cid)
            
            if len(view_dirs) >= 2:
                angles = []
                angle_pairs_str = []
                
                for i, d1 in enumerate(view_dirs):
                    for j, d2 in enumerate(view_dirs[i+1:]):
                        real_j = i + 1 + j
                        cid1 = dir_cam_ids[i]
                        cid2 = dir_cam_ids[real_j]
                        
                        # angle_between_vectors returns DEGREES
                        ang = angle_between_vectors(d1, d2)
                        
                        # Sanity check
                        if ang < 0 or ang > 180.1:
                            print(f"  [Error] Impossible angle between Cam {cid1} and {cid2}: {ang:.2f}")
                            ang = np.clip(ang, 0.0, 180.0)
                            
                        angles.append(ang)
                        angle_pairs_str.append(f"({cid1}-{cid2}: {ang:.1f}deg)")
                
                print(f"  [DEBUG] Window {wid} Angles: {', '.join(angle_pairs_str)}")
                
                if angles:
                    info.angle_diversity_p50 = np.percentile(angles, 50)
                    info.angle_diversity_p90 = np.percentile(angles, 90)
            
            # Determine freeze status based on observability
            # Default: plane OPTIMIZE, tvec FREEZE, rvec FREEZE
            info.plane_status = FreezeStatus.OPTIMIZE
            
            # tvec enabled if N_cam >= 2
            if n_cam >= 2:
                info.tvec_status = FreezeStatus.OPTIMIZE
                
                # Check baseline guard for rvec
                if info.baseline_median_mm < cfg.baseline_guard_mm:
                    info.rvec_status = FreezeStatus.FREEZE
                    info.freeze_reason = f"Baseline too small ({info.baseline_median_mm:.1f}mm < {cfg.baseline_guard_mm}mm)"
                elif info.angle_diversity_p50 < cfg.theta_freeze:
                    info.rvec_status = FreezeStatus.FREEZE
                    info.freeze_reason = f"Angle diversity too low ({info.angle_diversity_p50:.1f}° < {cfg.theta_freeze}°)"
                elif info.angle_diversity_p50 < cfg.theta_enable_rot:
                    # Weak observability (15-20 deg)
                    if cfg.allow_weak_rvec:
                        info.rvec_status = FreezeStatus.OPTIMIZE_STRONG
                        info.freeze_reason = f"Weak diversity ({info.angle_diversity_p50:.1f}°), Strong Constraints"
                    else:
                        info.rvec_status = FreezeStatus.FREEZE
                        info.freeze_reason = f"Weak diversity ({info.angle_diversity_p50:.1f}°), Strong mode disabled"
                elif info.angle_diversity_p50 < cfg.theta_strong_rot:
                    info.rvec_status = FreezeStatus.OPTIMIZE_REGULARIZED
                    info.freeze_reason = f"Moderate diversity ({info.angle_diversity_p50:.1f}°), regularized rvec"
                else:
                    info.rvec_status = FreezeStatus.OPTIMIZE
                    info.freeze_reason = f"Good diversity ({info.angle_diversity_p50:.1f}°, baseline {info.baseline_median_mm:.1f}mm)"
            else:
                info.freeze_reason = f"N_cam={n_cam}, extrinsics frozen"
            
            self.observability[wid] = info
    
        # Verbosity 2: Summary after computing all
        if cfg.verbosity >= 2:
            print("\n  [PR4] Observability Analysis Complete.")

    def _build_freeze_table(self):
        """Build freeze table from observability analysis."""
        self.freeze_table = {}
        
        for wid, info in self.observability.items():
            self.freeze_table[wid] = {
                'plane': info.plane_status,
                'cameras': {}
            }
            
            for cid in info.camera_ids:
                self.freeze_table[wid]['cameras'][cid] = {
                    'tvec': info.tvec_status,
                    'rvec': info.rvec_status
                }
    
    def print_freeze_table(self):
        """Print summary of freeze/optimize decisions."""
        cfg = self.config
        
        if cfg.verbosity >= 1:
            print("\n  [PR4] Optimization Freeze Table:")
            print(f"    {'WinID':<6} {'N_cam':<6} {'Base(mm)':<10} {'Ang(deg)':<10} {'Region Pl':<12} {'Region Tv':<12} {'Region Rv':<12} {'Reason'}")
            print("    " + "-"*96)
            
            for wid, info in self.observability.items():
                print(f"    {wid:<6} {info.n_cam:<6} {info.baseline_max_mm:<10.1f} {info.angle_diversity_p90:<10.1f} "
                      f"{info.plane_status.value:<12} {info.tvec_status.value:<12} {info.rvec_status.value:<12} {info.freeze_reason}")
            print("    " + "-"*96 + "\n")
        else:
             print("  [PR4] Computed Freeze Table (details hidden, verbosity=0)")

    def print_diagnostics(self, current_planes: Dict, current_cam_params: Dict):
        """Print final comparison of parameters."""
        cfg = self.config
        
        if cfg.verbosity >= 1:
             print("\n  [PR4] Final Parameter Diagnostics (Detailed Diff skipped for brevity)")
        
        print("\n  [PR4] Final Window State (Delta Invariant d_key):")
        for wid, pl_new in current_planes.items():
            pl_init = self.initial_planes[wid]
            
            # New normal/pt
            n_new = pl_new['plane_n']
            pt_new = pl_new['plane_pt']
            
            # Compute d_key: dot(n, pt - C_mean)
            # Need to re-compute C_mean from *optimized* cams
            cams = self.window_to_cams.get(wid, [])
            centers = []
            for cid in cams:
                if cid in current_cam_params:
                    p = current_cam_params[cid]
                    R, _ = cv2.Rodrigues(p[0:3])
                    C = camera_center(R, p[3:6])
                    centers.append(C)
            
            d_key = 0.0
            if centers:
                C_mean = np.mean(centers, axis=0)
                d_key = np.dot(n_new, pt_new - C_mean)
                
            delta_n_deg = np.degrees(angle_between_vectors(pl_init['plane_n'], n_new))
            
            print(f"    Win {wid}: d_key={d_key:.2f} mm (from opt C_mean), normal_shift={delta_n_deg:.2f} deg")

    def _update_cpp_camera_extrinsics(self, cam_obj, rvec: np.ndarray, tvec: np.ndarray):
        """Update C++ camera's extrinsics."""
        try:
            pp = cam_obj._pinplate_param
            
            # Update R
            R_mat, _ = cv2.Rodrigues(rvec)
            # [Fix] Element-wise assignment for pybind11 compatibility
            for i in range(3):
                for j in range(3):
                    pp.r_mtx[i, j] = float(R_mat[i, j])
            
            # Update T
            for i in range(3):
                pp.t_vec[i] = float(tvec[i])
            
            # Recalculate inverses
            R_inv = R_mat.T
            t_inv = -R_inv @ tvec
            
            for i in range(3):
                for j in range(3):
                    pp.r_mtx_inv[i, j] = float(R_inv[i, j])
                    
            for i in range(3):
                pp.t_vec_inv[i] = float(t_inv[i])
            
            cam_obj._pinplate_param = pp
            
        except Exception as e:
            print(f"[Warning] Extrinsics update failed for cam {cam_obj}: {e}")

    def _update_cpp_camera(self, cam_obj, plane_pt: List[float], plane_n: List[float]):
        """Update C++ camera's plane parameters using daisy-chain assignment."""
        try:
            pp = cam_obj._pinplate_param
            
            # [CRITICAL] Coordinate Alignment: PINPLATE model in C++ expects Farthest Interface
            # Python optimization uses Closest Interface as base.
            # Shift point along normal by thickness.
            thick_mm = pp.w_array[0] if pp.w_array else 0.0
            p_n = np.array(plane_n)
            p_pt = np.array(plane_pt)
            p_farthest = p_pt + p_n * thick_mm
            
            pl = pp.plane
            pl.pt = lpt.Pt3D(float(p_farthest[0]), float(p_farthest[1]), float(p_farthest[2]))
            pl.norm_vector = lpt.Pt3D(float(p_n[0]), float(p_n[1]), float(p_n[2]))
            pp.plane = pl
            cam_obj._pinplate_param = pp
            cam_obj.updatePt3dClosest() # Refresh internal geometric state
        except Exception as e:
            print(f"[Warning] Plane update failed for cam {cam_obj}: {e}")

    def _apply_planes_to_cpp(self, planes: Dict[int, Dict]):
        """Apply plane parameters to all C++ camera objects."""
        for wid, pl in planes.items():
            pt_list = pl['plane_pt'].tolist()
            n_list = pl['plane_n'].tolist()
            for cid in self.window_to_cams.get(wid, []):
                if cid in self.cams_cpp:
                    self._update_cpp_camera(self.cams_cpp[cid], pt_list, n_list)

    def _build_rays_frame(self, fid: int) -> Tuple[List[Ray], List[Ray]]:
        """Build rays for a frame using current C++ camera state."""
        rays_A, rays_B = [], []
        
        if fid not in self.obs_cache:
            return rays_A, rays_B
        
        for cid, (uvA, uvB) in self.obs_cache[fid].items():
            if cid not in self.cams_cpp:
                continue
            
            wid = self.cam_to_window.get(cid, -1)
            cam_obj = self.cams_cpp[cid]
            
            if uvA is not None:
                r = build_pinplate_ray_cpp(cam_obj, uvA, cam_id=cid, 
                                           window_id=wid, frame_id=fid, endpoint="A")
                if r.valid:
                    rays_A.append(r)
            
            if uvB is not None:
                r = build_pinplate_ray_cpp(cam_obj, uvB, cam_id=cid,
                                           window_id=wid, frame_id=fid, endpoint="B")
                if r.valid:
                    rays_B.append(r)
        
        return rays_A, rays_B

    def evaluate_residuals(self, planes: Dict[int, Dict], cam_params: Dict[int, np.ndarray],
                           lambda_eff: float) -> Tuple[np.ndarray, float, float, int, int]:
        """
        Evaluate residuals with fixed-size padding for Scipy least_squares compatibility.
        """
        # Apply planes and extrinsics
        self._apply_planes_to_cpp(planes)
        self._apply_extrinsics_to_cpp(cam_params)
        
        # 1. Pre-calculate total possible counts for FIXED size
        # Ray slots: sum of cameras per frame * 2
        # Length slots: 1 per frame
        # Barrier slots: number of valid points * windows per point (worst-case all points)
        
        # To keep it simple and truly fixed, use the observations present in fids_optim
        total_ray_slots = 0
        for fid in self.fids_optim:
            # We know exactly how many cameras see each endpoint from setup
            for key in ['obsA', 'obsB']:
                if fid in self.dataset[key]:
                    total_ray_slots += len(self.dataset[key][fid])
        
        total_len_slots = len(self.fids_optim)
        
        # Barrier slots: each point (max 2 per frame) can collide with its window's planes
        # Since windows are fixed, we can count slots
        total_barrier_slots = 0
        for fid in self.fids_optim:
            for key in ['obsA', 'obsB']:
                if fid in self.dataset[key]:
                    # Point sees these windows
                    wids = set(self.cam_to_window.get(cid) for cid in self.dataset[key][fid])
                    total_barrier_slots += len(wids)
                    
        # Pre-allocate
        res_ray_fixed = np.zeros(total_ray_slots)
        res_len_fixed = np.zeros(total_len_slots)
        res_barrier_fixed = np.zeros(total_barrier_slots)
        
        idx_ray = 0
        idx_len = 0
        idx_barrier = 0
        
        S_ray = 0.0
        S_len = 0.0
        N_ray_actual = 0
        N_len_actual = 0
        num_triangulated_points = 0
        valid_points_data = [] # For barrier computation
        
        radius_A = self.dataset.get('est_radius_small_mm', 0.0)
        radius_B = self.dataset.get('est_radius_large_mm', 0.0)
        
        # --- Computation Loop ---
        for fid in self.fids_optim:
            rays_A, rays_B = self._build_rays_frame(fid)
            
            # Get FIXED slot counts from dataset (not from rays which can vary)
            obs_A = self.dataset['obsA'].get(fid, {})
            obs_B = self.dataset['obsB'].get(fid, {})
            n_slots_A = len(obs_A)
            n_slots_B = len(obs_B)
            wids_A = set(self.cam_to_window.get(cid) for cid in obs_A)
            wids_B = set(self.cam_to_window.get(cid) for cid in obs_B)
            n_barrier_A = len(wids_A)
            n_barrier_B = len(wids_B)
            
            # Endpoint A
            validA = False
            XA = None
            if len(rays_A) >= 2:
                XA, _, validA, _ = triangulate_point(rays_A)
                if validA:
                    num_triangulated_points += 1
                    wids = set(r.window_id for r in rays_A if r.window_id is not None)
                    valid_points_data.append((XA, wids, 'A', idx_barrier))
                    # Fill Ray residuals for valid rays only
                    for r in rays_A:
                        d = point_to_ray_dist(XA, r.o, r.d)
                        res_ray_fixed[idx_ray] = d
                        S_ray += d**2
                        idx_ray += 1
                        N_ray_actual += 1
                    # Skip remaining slots if rays_A < n_slots_A
                    idx_ray += (n_slots_A - len(rays_A))
                else:
                    # Point invalid, skip all ray slots (keep at 0.0)
                    idx_ray += n_slots_A
            else:
                # Insufficient rays, skip all ray slots
                idx_ray += n_slots_A
            # Always advance barrier index by fixed count
            idx_barrier += n_barrier_A

            # Endpoint B
            validB = False
            XB = None
            if len(rays_B) >= 2:
                XB, _, validB, _ = triangulate_point(rays_B)
                if validB:
                    num_triangulated_points += 1
                    wids = set(r.window_id for r in rays_B if r.window_id is not None)
                    valid_points_data.append((XB, wids, 'B', idx_barrier))
                    for r in rays_B:
                        d = point_to_ray_dist(XB, r.o, r.d)
                        res_ray_fixed[idx_ray] = d
                        S_ray += d**2
                        idx_ray += 1
                        N_ray_actual += 1
                    idx_ray += (n_slots_B - len(rays_B))
                else:
                    idx_ray += n_slots_B
            else:
                idx_ray += n_slots_B
            idx_barrier += n_barrier_B

            # Length
            if validA and validB:
                wand_len = np.linalg.norm(XA - XB)
                err = wand_len - self.wand_length
                res_len_fixed[idx_len] = err
                S_len += err**2
                N_len_actual += 1
            idx_len += 1

        # 2. Side Barrier (Adaptive)
        # J_data = S_ray + lambda * S_len
        J_data = S_ray + lambda_eff * S_len
        N_samples = max(1, num_triangulated_points)
        w_side = 0.02 * J_data / N_samples
        
        margin_mm = self.config.margin_side_mm
        sX_vals = []
        
        for (X, wids, endpoint, b_start_idx) in valid_points_data:
            r_val = radius_A if endpoint == 'A' else radius_B
            curr_b_idx = b_start_idx
            for wid in sorted(list(wids)): # Sorted for deterministic indexing
                if wid not in planes: 
                    curr_b_idx += 1
                    continue
                pl = planes[wid]
                p_side, sX = compute_soft_barrier_penalty(
                    X, pl['plane_pt'], pl['plane_n'], 
                    w_side=w_side, sigma=0.01, R_mm=r_val, margin_mm=margin_mm
                )
                res_barrier_fixed[curr_b_idx] = p_side
                sX_vals.append(sX)
                curr_b_idx += 1

        # Diagnostics
        if sX_vals:
            sX_arr = np.array(sX_vals)
            self._last_barrier_stats = {
                'min_sX': np.min(sX_arr),
                'pct_near': np.mean(sX_arr < margin_mm) * 100,
                'ratio': np.sum(res_barrier_fixed**2) / max(1e-9, J_data),
                'w_side': w_side
            }
        else:
            self._last_barrier_stats = {}

        # 3. Combine
        weighted_len = np.sqrt(lambda_eff) * res_len_fixed
        residuals = np.concatenate([res_ray_fixed, weighted_len, res_barrier_fixed])
            
        return residuals, S_ray, S_len, N_ray_actual, N_len_actual


    def _adapt_lambda(self, lambda_old: float, S_ray: float, S_len: float, N_ray: int, N_len: int) -> float:
        """Adapt lambda based on RMSE comparison.
        
        Strategy:
        - If rmse_len <= rmse_ray: physics is good, keep lambda
        - If rmse_len > rmse_ray: 
            - First time: jump to 100
            - Subsequent: double lambda
        """
        cfg = self.config
        eps = 1e-12
        
        if S_len < eps or N_len == 0:
            return lambda_old
            
        rmse_ray = np.sqrt(S_ray / max(1, N_ray))
        rmse_len = np.sqrt(S_len / max(1, N_len))
        
        if rmse_len <= rmse_ray:
            # Physics constraint is satisfied, no need to increase lambda
            return lambda_old
        
        # rmse_len > rmse_ray: need more emphasis on wand length
        if lambda_old < 100.0:
            lambda_new = 100.0  # First time: jump to 100
        else:
            lambda_new = lambda_old * 2.0  # Subsequently: double
        
        # Global clamp
        lambda_new = min(lambda_new, cfg.lambda_max)
        
        return lambda_new

    def _apply_extrinsics_to_cpp(self, cam_params: Dict[int, np.ndarray]):
        """Apply extrinsic parameters to C++ cameras."""
        for cid, p in cam_params.items():
            if cid in self.cams_cpp:
                rvec = p[0:3]
                tvec = p[3:6]
                self._update_cpp_camera_extrinsics(self.cams_cpp[cid], rvec, tvec)

    def _get_param_layout(self, mode: str) -> List[Tuple]:
        """
        Get layout of parameter vector x.
        Returns list of (type, id, subparam_idx).
        """
        layout = []
        
        # 1. Planes (always optimized unless frozen?)
        # Actually P1.1 was d only, P1.2 was d,a,b.
        # PR4 implies full d,a,b optimization for planes
        for wid in self.window_ids:
            ft = self._freeze_table_active.get(wid, {})
            if ft.get('plane') != FreezeStatus.FREEZE:
                layout.append(('plane_d', wid, 0))
                layout.append(('plane_a', wid, 0))
                layout.append(('plane_b', wid, 0))
        
        # 2. Cameras
        if mode != 'planes_only':
            for wid in self.window_ids:
                ft = self._freeze_table_active.get(wid, {})
                for cid in self.window_to_cams.get(wid, []):
                    cft = ft.get('cameras', {}).get(cid, {})
                    
                    if cft.get('tvec') != FreezeStatus.FREEZE:
                        layout.append(('cam_t', cid, 0)) # tx
                        layout.append(('cam_t', cid, 1)) # ty
                        layout.append(('cam_t', cid, 2)) # tz
                    
                    # rvec (only if allowed by active freeze table)
                    r_status = cft.get('rvec')
                    if r_status is not None and r_status != FreezeStatus.FREEZE:
                        layout.append(('cam_r', cid, 0)) # rx
                        layout.append(('cam_r', cid, 1)) # ry
                        layout.append(('cam_r', cid, 2)) # rz
        
        # Debug: Log rvec layout for verification (verbosity >= 2)
        if self.config.verbosity >= 2:
            rvec_cids = sorted(set(pid for (ptype, pid, _) in layout if ptype == 'cam_r'))
            if rvec_cids:
                print(f"    [DEBUG] _get_param_layout({mode}): rvec enabled for cids={rvec_cids}")
        
        return layout
    
    def _rvec_step_cap_deg(self, status: int) -> float:
        """Get rotation step cap in degrees based on optimization status."""
        config = self.config
        if status == FreezeStatus.OPTIMIZE_STRONG:
            return config.step_cap_rot_deg_strong
        elif status == FreezeStatus.OPTIMIZE_REGULARIZED:
            return config.step_cap_rot_deg_weak
        else: # Normal OPTIMIZE
            return config.step_cap_rot_deg_normal
    
    def _rvec_step_cap_deg(self, status: int) -> float:
        """Get rotation step cap in degrees based on optimization status."""
        config = self.config
        if status == FreezeStatus.OPTIMIZE_STRONG:
            return config.step_cap_rot_deg_strong
        elif status == FreezeStatus.OPTIMIZE_REGULARIZED:
            return config.step_cap_rot_deg_weak
        else: # Normal OPTIMIZE
            return config.step_cap_rot_deg_normal


    def _unpack_params_delta(self, x: np.ndarray, layout: List[Tuple]) -> Tuple[Dict, Dict]:
        """
        Unpack x (deltas) into updated planes and cam_params.
        
        Returns:
            (new_planes, new_cam_params)
        """
        # Start from INITIAL state
        current_planes = {}
        for wid, pl in self.initial_planes.items():
            current_planes[wid] = {
                'plane_n': pl['plane_n'].copy(),
                'plane_pt': pl['plane_pt'].copy(),
                'initialized': pl.get('initialized', True)
            }
        
        current_cam_params = {cid: p.copy() for cid, p in self.initial_cam_params.items()}
        
        idx = 0
        for (ptype, pid, subidx) in layout:
            val = x[idx]
            idx += 1
            
            if ptype.startswith('plane'):
                # Plane update logic
                # We need to collect d, a, b for each window
                # This unpacking is slightly inefficient (repeatedly accessing), but safe
                pass 
        
        # Better approach: Group by ID first
        # But 'layout' defines the order in 'x'.
        # Let's iterate 'x' and accumulate updates
        
        plane_deltas = {wid: {'d': 0.0, 'a': 0.0, 'b': 0.0} for wid in self.window_ids}
        cam_deltas = {cid: {'t': np.zeros(3), 'r': np.zeros(3)} for cid in self.active_cam_ids}
        
        idx = 0
        for (ptype, pid, subidx) in layout:
            val = x[idx]
            idx += 1
            
            if ptype == 'plane_d':
                plane_deltas[pid]['d'] = val
            elif ptype == 'plane_a':
                plane_deltas[pid]['a'] = val
            elif ptype == 'plane_b':
                plane_deltas[pid]['b'] = val
            elif ptype == 'cam_t':
                cam_deltas[pid]['t'][subidx] = val
            elif ptype == 'cam_r':
                cam_deltas[pid]['r'][subidx] = val
        
        # Apply Plane Deltas
        for wid, deltas in plane_deltas.items():
            if wid not in current_planes: continue
            
            n0 = self.initial_planes[wid]['plane_n']
            pt0 = self.initial_planes[wid]['plane_pt']
            
            # 1. Update distance (d)
            # d_new = d_old + delta_d
            # pt_new = pt_old + delta_d * n0  (approximation of shift along normal)
            d_shift = deltas['d']
            pt_shifted = pt0 + d_shift * n0
            
            # 2. Update normal (alpha, beta) using tangent space
            alpha, beta = deltas['a'], deltas['b']
            n_new = update_normal_tangent(n0, alpha, beta)
            
            current_planes[wid]['plane_n'] = n_new
            current_planes[wid]['plane_pt'] = pt_shifted
            current_planes[wid]['initialized'] = True
            
        # Apply Camera Deltas
        for cid, deltas in cam_deltas.items():
            if cid not in current_cam_params: continue
            
            # Apply tvec delta
            current_cam_params[cid][3:6] += deltas['t']
            
            # Apply rvec delta
            # R_new = R_delta * R_old  (global perturbation? or local?)
            # Usually optimization finds a delta-rvec.
            # Local perturbation: R_new = R(delta) * R_old
            # Global perturbation: R_new = R_old * R(delta)
            # Let's use Local (perturbation intrisic to camera frame?)
            # Actually standard bundle adjustment often uses: R_new = exp(w) * R_old
            # where w is rotation vector update.
            
            r_old = current_cam_params[cid][0:3]
            dr = deltas['r']
            
            if np.linalg.norm(dr) > 1e-8:
                R_old, _ = cv2.Rodrigues(r_old)
                dR_mat, _ = cv2.Rodrigues(dr)
                # Left multiplication (global) vs Right multiplication (local)
                # Camera projects: X_c = R * X_w + T
                # If we perturb R: (I + [w]x) * R * X_w
                # This corresponds to left multiplication.
                R_new = dR_mat @ R_old
                r_new, _ = cv2.Rodrigues(R_new)
                current_cam_params[cid][0:3] = r_new.flatten()
            
        return current_planes, current_cam_params

    def _residuals_pr4(self, x: np.ndarray, layout: List[Tuple], mode: str, lambda_eff: float) -> np.ndarray:
        """Residual function for generic PR4 optimization."""
        # Unpack
        curr_planes, curr_cams = self._unpack_params_delta(x, layout)
        
        # Data Residuals
        # Note: evaluate_residuals handles applying to CPP internally
        residuals, S_ray, S_len, N_ray, N_len = self.evaluate_residuals(curr_planes, curr_cams, lambda_eff)
        
        # [Fix] Update live stats for logging
        self._last_ray_rmse = np.sqrt(S_ray / max(N_ray, 1))
        self._last_len_rmse = np.sqrt(S_len / max(N_len, 1)) if N_len > 0 else 0.0
        
        # Regularization
        reg_residuals = []
        cfg = self.config
        
        idx = 0
        for (ptype, pid, subidx) in layout:
            val = x[idx]
            idx += 1
            
            if ptype.startswith('plane'):
                # Plane regularization
                # Penalty on deviation from initial (d, alpha, beta)
                reg_residuals.append(val * np.sqrt(cfg.lambda_reg_plane))
            elif ptype == 'cam_t':
                reg_residuals.append(val * np.sqrt(cfg.lambda_reg_tvec))
            elif ptype == 'cam_r':
                # Check weak vs strong regularization
                wid = self.cam_to_window.get(pid)
                ft = self.freeze_table.get(wid, {})
                cft = ft.get('cameras', {}).get(pid, {})
                status = cft.get('rvec')
                
                weight = cfg.lambda_reg_rvec
                if status == FreezeStatus.OPTIMIZE_REGULARIZED:
                    weight *= 2.0  # Double regularization for weak geometry
                elif status == FreezeStatus.OPTIMIZE_STRONG:
                    # Dynamic Strong Prior: lambda = base * (20 / angle)^p
                    obs = self.observability.get(wid)
                    angle = obs.angle_diversity_p50 if obs else 20.0
                    angle = max(angle, 1e-6)
                    s = (cfg.theta_enable_rot / angle) ** cfg.prior_p_rot
                    weight = cfg.prior_lambda_rot_base * s
                
                reg_residuals.append(val * np.sqrt(weight))
        
        if len(reg_residuals) > 0:
            return np.concatenate([residuals, np.array(reg_residuals)])
        return np.array(residuals)

    def _optimize_generic(self, mode: str, description: str):
        """Generic optimization loop with step cap and rollback."""
        layout = self._get_param_layout(mode)
        if not layout:
            print(f"  [{description}] No parameters to optimize.")
            return

        x0 = np.zeros(len(layout), dtype=np.float64)
        # lambda_eff initialized later
        cfg = self.config
        
        print(f"  [{description}] optimizing {len(x0)} parameters ({len(layout)//3} blocks)...")
        # Calc initial RMSE for rollback reference
        planes0, cams0 = self._unpack_params_delta(x0, layout)
        # Use initial dummy lambda to get S components
        _, S_ray0, S_len0, N_ray, N_len = self.evaluate_residuals(planes0, cams0, cfg.lambda0_init)
        
        rmse_ray0 = np.sqrt(S_ray0 / max(N_ray, 1))
        rmse_len0 = np.sqrt(S_len0 / max(N_len, 1)) if N_len > 0 else 0.0

        # [USER REQUEST] Dynamic Initial Lambda
        # Set lambda such that lambda * S_len ~ S_ray (Ratio ~ 1.0)
        if S_len0 > 1e-12:
            lambda_eff = S_ray0 / S_len0
            # Clamp to reasonable range
            # [USER REQUEST] Cap initial lambda to prevent explosion (e.g. 500.0)
            initial_cap = 500.0
            lambda_eff = float(np.clip(lambda_eff, cfg.lambda_min, initial_cap))
        else:
            lambda_eff = cfg.lambda0_init
            
        # [USER REQUEST] Initial Physics Check
        # If physics is already good (Len < Ray), prevent initial explosion
        if rmse_len0 < rmse_ray0:
            lambda_cap = 10.0  # Conservative starting point
            if lambda_eff > lambda_cap:
                print(f"  [Init] Physics Good (Len={rmse_len0:.4f} < Ray={rmse_ray0:.4f}). Capping Lambda {lambda_eff:.1f}->{lambda_cap:.1f}")
                lambda_eff = lambda_cap
            
        print(f"  [Init] Dynamic Lambda: {lambda_eff:.2f} (S_ray={S_ray0:.2f}, S_len={S_len0:.2f})")
        
        # Build bounds for each parameter based on layout
        # We align these with the intended Step Caps to ensure symmetric constraints during optimization.
        lb = []
        ub = []
        for (ptype, pid, subidx) in layout:
            if ptype == 'plane_d':
                # Plane distance: +/- 20mm (Matching PR5 bounds)
                lb.append(-20.0)
                ub.append(20.0)
            elif ptype == 'plane_a' or ptype == 'plane_b':
                # Plane angles: +/- 10 degrees (allowing more flexibility in PR4)
                limit_rad = np.radians(10.0)
                lb.append(-limit_rad)
                ub.append(limit_rad)
            elif ptype == 'cam_r':
                # Camera rotation: Match Step-Cap based on status
                wid = self.cam_to_window.get(pid)
                status = self.freeze_table.get(wid, {}).get('cameras', {}).get(pid, {}).get('rvec')
                limit_deg = self._rvec_step_cap_deg(status)
                limit_rad = np.radians(limit_deg)
                lb.append(-limit_rad)
                ub.append(limit_rad)
            elif ptype == 'cam_t':
                # Camera translation: Match Step-Cap based on status
                wid = self.cam_to_window.get(pid)
                status = self.freeze_table.get(wid, {}).get('cameras', {}).get(pid, {}).get('tvec')
                limit_mm = self._tvec_step_cap_mm(status)
                lb.append(-limit_mm)
                ub.append(limit_mm)
            else:
                lb.append(-10.0)
                ub.append(10.0)
        
        bounds = (np.array(lb), np.array(ub))
        
        # Residual wrapper for event pumping
        self._res_call_count = 0
        def residuals_wrapper(x, *args, **kwargs):
            res = self._residuals_pr4(x, *args, **kwargs)
            self._res_call_count += 1
            if self.progress_callback and self._res_call_count % 30 == 0:
                try:
                    c_approx = 0.5 * np.sum(res**2)
                    # Log to console (disabled - too verbose)
                    # if self.config.verbosity >= 2:
                    #     print(f"    [Iter {self._res_call_count}] Cost={c_approx:.2f}, RayRMSE={self._last_ray_rmse:.4f}, LenRMSE={self._last_len_rmse:.4f}")
                        
                    # Pass -1.0 for RMSEs to show N.A.
                    if self.progress_callback:
                        self.progress_callback(f"Refining Camera & Window Parameters (Round {round_idx+1})...", self._last_ray_rmse, self._last_len_rmse, c_approx)
                except Exception as e:
                    print(f"[Warning] Progress callback failed: {e}")
            return res


        for round_idx in range(self.config.outer_rounds):
            # [DEBUG] Log Lambda for User Analysis
            print(f"    [PR4][ROUND {round_idx+1}] Lambda = {lambda_eff:.2f}")

            # Solve with bounds
            res = least_squares(
                residuals_wrapper, 
                x0, 
                args=(layout, mode, lambda_eff),
                method='trf', 
                bounds=bounds,
                verbose=0,
                x_scale='jac',
                ftol=1e-6,
                xtol=1e-6,
                gtol=1e-6,
                max_nfev=50
            )

            # Print Barrier Stats (PR4 Part 5)
            if cfg.verbosity >= 1 and hasattr(self, '_last_barrier_stats') and self._last_barrier_stats:
                s = self._last_barrier_stats
                print(f"    [PR4][SIDE-BARRIER][ROUND {round_idx+1}] min(sX)={s['min_sX']:.4f}mm, near(<20um)={s['pct_near']:.1f}%, cost/J={s['ratio']:.1e}")

            
            x_candidate = res.x
            
            # Step-capping now handled natively via least_squares bounds
            x_step_capped = res.x
            clamped_rvec_count = 0 
            clamped_tvec_count = 0
            num_rvec_blocks = 1
            
            # --- 2. Rollback Check ---
            # Evaluate new cost
            curr_planes, curr_cams = self._unpack_params_delta(x_step_capped, layout)
            _, S_ray, S_len, N_ray, N_len = self.evaluate_residuals(curr_planes, curr_cams, lambda_eff)
            
            rmse_ray = np.sqrt(S_ray / max(N_ray, 1))
            rmse_len = np.sqrt(S_len / max(N_len, 1)) if N_len > 0 else 0.0
            
            # [FIX] MSE-based normalized cost
            # Normalized Cost = S_ray/N_ray + lambda * S_len/N_len
            cost_new = (rmse_ray**2) + lambda_eff * (rmse_len**2)
            
            # Baseline J0 is established from the accepted state
            if round_idx == 0:
                # If round 0 (first attempt), J0 is from initial residuals (normalized)
                self._current_J0 = (rmse_ray0**2) + lambda_eff * (rmse_len0**2)
            
            J0 = self._current_J0
            
            # Use tracked J0 for rollback comparison
            # Ensure we have a valid baseline J0
            if not hasattr(self, '_current_J0'):
                 self._current_J0 = (rmse_ray0**2 * max(N_ray,1)) + lambda_eff * (rmse_len0**2 * max(N_len,1))

            # === NEW Tiered Rollback Logic (PR4.2/4.3) ===
            # Thresholds: 1% and 5%
            J0 = self._current_J0
            threshold_accept = 1.01  # ≤1% increase: direct accept
            threshold_soft = 1.05   # 1-5% increase: conditional accept
            
            # Calculate clamped ratio for conditional logic
            # layout is a list of (ptype, pid, subidx) tuples
            # Count unique camera IDs with 'cam_r' parameter type
            num_rvec_blocks = len(set(pid for ptype, pid, _ in layout if ptype == 'cam_r'))
            num_rvec_blocks = max(num_rvec_blocks, 1)  # Avoid division by zero
            clamped_ratio = clamped_rvec_count / num_rvec_blocks

            # Determine action
            if cost_new <= J0 * threshold_accept:
                # DIRECT ACCEPT: Cost improved or increased ≤1%
                accept = True
                accept_action = "ACCEPT"
            elif cost_new < J0 * threshold_soft:
                # CONDITIONAL ACCEPT (1-5% increase)
                accept = True
                if clamped_ratio >= 0.3:
                    # Many rvecs clamped -> reduce step size
                    self._step_cap_multiplier = getattr(self, '_step_cap_multiplier', 1.0) * 0.5
                    accept_action = "ACCEPT (clamped>30%, step_cap*=0.5)"
                else:
                    # Few rvecs clamped -> increase lambda
                    lambda_eff *= 2.0
                    accept_action = "ACCEPT (clamped<30%, lambda*=2)"
            else:
                # REJECT: Cost increased >5%
                accept = False
                accept_action = "REJECT"

            # Convergence Check: If improvement is negligible (< 1e-6), stop optimization rounds early.
            # Compare J_new vs J0
            diff_rel = (J0 - cost_new) / max(J0, 1e-12)
            if accept and (abs(diff_rel) < 1e-6) and round_idx > 0:
                if cfg.verbosity >= 1:
                    print(f"    Round {round_idx+1}: CONVERGED (Cost stable).")
                x0 = x_step_capped
                rmse_ray0, rmse_len0 = rmse_ray, rmse_len
                break
            
            if accept:
                x_opt = x_step_capped
                # Update references
                rmse_ray0 = rmse_ray
                rmse_len0 = rmse_len

                lambda_new = self._adapt_lambda(lambda_eff, S_ray, S_len, N_ray, N_len)
                
                if cfg.verbosity >= 2:
                    print(f"    Round {round_idx+1}: {accept_action} (J={cost_new:.4f}, J0={J0:.4f}, caps r={clamped_rvec_count}/{num_rvec_blocks}). lam={lambda_eff:.1f}->{lambda_new:.1f}")
 
                lambda_eff = lambda_new
                x0 = x_opt
                
                # Update J0 Baseline with NEW lambda for next comparison
                # Normalized Cost J0 = MSE_ray + lambda_new * MSE_len
                self._current_J0 = (rmse_ray0**2) + lambda_eff * (rmse_len0**2)
                
            else:
                # REJECT: Cost increased >5%
                if cfg.verbosity >= 1:
                     print(f"    Round {round_idx+1}: REJECT (J={cost_new:.4f} > J0*1.05={J0*1.05:.4f}). Rollback.")
                     print(f"       Caps: r={clamped_rvec_count}, t={clamped_tvec_count}")
                     print(f"       RMSE Ray: {rmse_ray0:.4f}->{rmse_ray:.4f}, Len: {rmse_len0:.4f}->{rmse_len:.4f}")
                
                # Increase lambda to constrain next attempt
                lambda_eff *= 2.0
                
                # Update J0 Baseline with NEW lambda for next comparison
                self._current_J0 = (rmse_ray0**2) + lambda_eff * (rmse_len0**2)
                
                # x0 remains same

            
            # Progress callback for UI updates (called every round, regardless of accept/reject)
            if self.progress_callback is not None:
                try:
                    # Report current best RMSE values
                    self.progress_callback(description, rmse_ray0, rmse_len0, cost_new)
                    self._last_ray_rmse = rmse_ray0
                    self._last_len_rmse = rmse_len0
                except Exception:
                    pass  # Silently ignore callback errors to not disrupt optimization

        
        # Commit Final Results
        self.window_planes, self.cam_params = self._unpack_params_delta(x0, layout)
        
        # Update Initial State for next stage
        self.initial_planes = {wid: {k: (v.copy() if hasattr(v, 'copy') else v) for k,v in pl.items()} for wid, pl in self.window_planes.items()}
        self.initial_cam_params = {cid: p.copy() for cid, p in self.cam_params.items()}


    def optimize(self, skip_pr4: bool = False, pr4_stage: int = 3) -> Tuple[Dict[int, Dict], Dict[int, np.ndarray]]:
        """
        Main PR4 optimization entry point.
        """
        if skip_pr4 or self.config.skip_pr4:
            print("\n[PR4] Skipped (skip_pr4=True)")
            return self.window_planes, self.cam_params
        
        print("\n" + "=" * 50)
        print("[Refractive] Phase PR4: Bundle Adjustment (Selective BA)")
        print("=" * 50)
        
        # Calculate initial RMSE for UI continuity
        try:
             _, S_ray, S_len, N_ray, N_len = self.evaluate_residuals(self.window_planes, self.cam_params, self.config.lambda0_init)
             self._last_ray_rmse = np.sqrt(S_ray / max(N_ray, 1))
             self._last_len_rmse = np.sqrt(S_len / max(N_len, 1)) if N_len > 0 else 0.0
        except:
             pass
        
        # Print freeze table
        self.print_freeze_table()
        
        # PR4.1: Planes + selected tvec
        if pr4_stage >= 1:
            # Dynamically set rvec to FREEZE for this stage
            # We can override the table temporarily?
            # Or just filter in _get_param_layout using 'mode'.
            # Current _get_param_layout respects freeze_table strictly.
            # So we should modify freeze table temporarily or have mode-switch?
            # Easier: Pass 'planes_and_tvec' mode to _optimize_generic 
            # and generic uses it to filter on top of freeze table?
            # Actually _get_param_layout already handles 'planes_only'.
            # Let's verify _get_param_layout behavior.
            # It enables cameras if mode != 'planes_only'.
            # I need finer control.
            
            # Temporary override of freeze table for PR4.1
            saved_ftp = {} # save rvec status
            for wid in self.freeze_table:
                for cid in self.freeze_table[wid]['cameras']:
                    saved_ftp[(wid, cid)] = self.freeze_table[wid]['cameras'][cid]['rvec']
                    self.freeze_table[wid]['cameras'][cid]['rvec'] = FreezeStatus.FREEZE
            
            print("\n[PR4.1] Optimizing planes + selected tvec...")
            self._optimize_generic("pr4_1", "Optimizing camera extrinsic parameter and window parameters...")
            
            # Restore rvec status
            for (wid, cid), status in saved_ftp.items():
                self.freeze_table[wid]['cameras'][cid]['rvec'] = status

        # PR4.2: Enable only STRONG rvec candidates
        if pr4_stage >= 2:
            # Build masked table: STRONG rvec enabled, others (OPTIMIZE/REG) temporarily FREEZE
            import copy
            masked_table_pr42 = copy.deepcopy(self.freeze_table)
            
            n_strong = 0
            strong_cids = []
            for wid in masked_table_pr42:
                cams = masked_table_pr42[wid].get('cameras', {})
                for cid in cams:
                    original_status = self.freeze_table[wid]['cameras'][cid].get('rvec', FreezeStatus.FREEZE)
                    if original_status == FreezeStatus.OPTIMIZE_STRONG:
                        # Keep STRONG enabled
                        n_strong += 1
                        strong_cids.append(cid)
                    elif original_status in [FreezeStatus.OPTIMIZE, FreezeStatus.OPTIMIZE_REGULARIZED]:
                        # Temporarily freeze non-STRONG rvec for PR4.2
                        masked_table_pr42[wid]['cameras'][cid]['rvec'] = FreezeStatus.FREEZE
            
            if n_strong > 0:
                print(f"\n[PR4.2] Optimization Phase 2 (Adding {n_strong} STRONG rvec candidates: {sorted(set(strong_cids))})...")
                # Context switch to masked table with exception safety
                old_table = self._freeze_table_active
                try:
                    self._freeze_table_active = masked_table_pr42
                    self._optimize_generic("pr4_2", "Optimizing camera extrinsic parameter and window parameters...")
                finally:
                    self._freeze_table_active = old_table
            else:
                print("\n[PR4.2] No STRONG rvec candidates, skipping.")
            
        # PR4.3: Enable STRONG + REGULARIZED + OPTIMIZE rvecs (full table)
        if pr4_stage >= 3:
            # Count all eligible rvec candidates
            n_total = 0
            all_cids = []
            for wid in self.freeze_table:
                cams = self.freeze_table[wid].get('cameras', {})
                for cid in cams:
                    status = self.freeze_table[wid]['cameras'][cid].get('rvec', FreezeStatus.FREEZE)
                    if status in [FreezeStatus.OPTIMIZE, FreezeStatus.OPTIMIZE_STRONG, FreezeStatus.OPTIMIZE_REGULARIZED]:
                        n_total += 1
                        all_cids.append(cid)
            
            if n_total > 0:
                print(f"\n[PR4.3] Joint bundle adjustment (Adding {n_total} rvec candidates: {sorted(set(all_cids))})...")
                # Use full original table (contains all statuses)
                self._freeze_table_active = self.freeze_table
                self._optimize_generic("pr4_3", "Optimizing camera extrinsic parameter and window parameters...")
            else:
                print("\n[PR4.3] No rvec candidates, skipping.")

            
        
        # [CRITICAL] Final Sync to C++ objects (Ensures next stage starts correctly)
        self.evaluate_residuals(self.window_planes, self.cam_params, self.config.lambda0_init)

        # Print diagnostics
        self.print_diagnostics()
        
        print("\n[PR4] Optimization Complete.")
        return self.window_planes, self.cam_params

    # Helper methods placeholders removed as they are integrated into optimize()
    def _optimize_pr4_1(self): pass
    def _optimize_pr4_2(self): pass
    def _optimize_pr4_3(self): pass
    
    def print_diagnostics(self):
        """Print comprehensive diagnostics after optimization."""
        print("\n[PR4] Final Diagnostics:")
        print("-" * 40)
        
        # Evaluate final residuals
        lambda_eff = self.config.lambda0_init
        residuals, S_ray, S_len, N_ray, N_len = self.evaluate_residuals(
            self.window_planes, self.cam_params, lambda_eff
        )
        
        # Ray stats
        if N_ray > 0:
            ray_rmse = np.sqrt(S_ray / N_ray)
            print(f"  Ray Distance RMSE: {ray_rmse:.4f} mm ({N_ray} rays)")
        
        # Wand stats
        if N_len > 0:
            wand_rmse = np.sqrt(S_len / N_len)
            print(f"  Wand Length RMSE: {wand_rmse:.4f} mm ({N_len} pairs)")
            print(f"  Wand Length Target: {self.wand_length:.2f} mm")
        
        # Per-window summary
        print("\n  Per-Window Summary (d_internal=dot(n,pt), d_key_phys=dot(n,pt-Cmean)):")
        for wid in self.window_ids:
            pl = self.window_planes[wid]
            n = pl['plane_n']
            pt = pl['plane_pt']
            
            # d_internal: legacy/optimization-internal value
            d_internal = np.dot(n, pt)
            
            # d_key_phys: canonical distance from cameras (P1 invariant)
            cams = self.window_to_cams.get(wid, [])
            centers = []
            for cid in cams:
                if cid in self.cam_params:
                    p = self.cam_params[cid]
                    R, _ = cv2.Rodrigues(p[0:3])
                    C = camera_center(R, p[3:6])
                    centers.append(C)
            
            d_key_phys = 0.0
            if centers:
                C_mean = np.mean(centers, axis=0)
                d_key_phys = np.dot(n, pt - C_mean)
            
            print(f"    Window {wid}: d_internal={d_internal:.2f}mm, d_key_phys={d_key_phys:.2f}mm, n=[{n[0]:.4f}, {n[1]:.4f}, {n[2]:.4f}]")
        
        print("-" * 40)

    def _get_cache_path(self, dataset_path: str) -> str:
        """Get path to PR4 cache file."""
        return str(Path(dataset_path).parent / "pr4_bundle_cache.json")

    def try_load_cache(self, out_path: str) -> bool:
        """
        Try to load PR4 results from cache.
        Returns True if loaded successfully.
        """
        cache_path = self._get_cache_path(out_path)
        if not Path(cache_path).exists():
            return False
            
        try:
            with open(cache_path, 'r') as f:
                data = json.load(f)
                
            # Verify version matching or simple existance
            # PR4 cache structure is simple: planes, cam_params
            
            # Load Params (Params / Windows) - NO DATASET
            cached_cams = data.get('cam_params', {})
            for cid_str, p_list in cached_cams.items():
                cid = int(cid_str)
                if cid in self.cam_params:
                    self.cam_params[cid] = np.array(p_list)
                    
            # Load Planes
            planes_data = data.get('planes', {}) # Current PR4 format uses 'planes'
            for wid_str, pl in planes_data.items():
                wid = int(wid_str)
                if wid in self.window_planes:
                     self.window_planes[wid]['plane_pt'] = np.array(pl['plane_pt'])
                     self.window_planes[wid]['plane_n'] = np.array(pl['plane_n'])
            
            # Apply to C++ objects
            self._apply_planes_to_cpp(self.window_planes)
            for cid in self.active_cam_ids:
                cam = self.cams_cpp[cid]
                # Update Extrinsics
                p = self.cam_params[cid]
                self._update_cpp_camera_extrinsics(cam, p[0:3], p[3:6])
                
            
            print(f"[PR4][CACHE] Loaded parameters successfully from {cache_path}")
            print(f"  Note: Using cached parameters with FRESH dataset observations.")
            return True
        except Exception as e:
            print(f"[PR4][CACHE] Load failed (ignored): {e}")
            return False

    def save_cache(self, out_path: str):
        """Save PR4 results to cache."""
        try:
            cache_path = self._get_cache_path(out_path)
            
            data = {
                'timestamp': str(datetime.now()),
                'cam_ids': self.active_cam_ids,
                'window_ids': self.window_ids,
                'planes': {
                    str(w): {
                        'plane_pt': np.asarray(pl['plane_pt']).tolist(),
                        'plane_n': np.asarray(pl['plane_n']).tolist()
                    } for w, pl in self.window_planes.items()
                },
                'cam_params': {
                    str(c): np.asarray(p).tolist() for c, p in self.cam_params.items()
                }
            }
            
            with open(cache_path, 'w') as f:
                json.dump(data, f, indent=2)
                
            print(f"[PR4][CACHE] Saved results to {cache_path}")
            
        except Exception as e:
            print(f"[PR4][CACHE] Save failed: {e}")


class RefractiveBAOptimizerPR5:
    """
    PR5: Robust Bundle Adjustment for Refractive Calibration.
    Optimizes:
    - Window Planes (d, alpha, beta)
    - Window Thickness (t)
    - Intrinsics (f)
    - Extrinsics (rvec, tvec)
    
    Uses purely geometric residuals (Ray Distance + Wand Length) with Strong Priors.
    """
    
    def __init__(self,
                 dataset: Dict,
                 cam_params: Dict[int, np.ndarray],
                 cams_cpp: Dict[int, 'lpt.Camera'],
                 cam_to_window: Dict[int, int],
                 window_media: Dict[int, Dict],
                 window_planes: Dict[int, Dict],
                 wand_length: float,
                 config: PR5Config,
                 progress_callback: Optional[callable] = None,
                 freeze_table: Dict = None):
        
        self.dataset = dataset
        self.cam_params = {int(cid): np.array(p, dtype=np.float64) for cid, p in cam_params.items()}
        self.freeze_table = freeze_table
        self.cams_cpp = cams_cpp
        self.cam_to_window = cam_to_window
        
        # Deep copy window data to avoid polluting previous stages until commit
        self.window_media = {int(w): m.copy() for w, m in window_media.items()}
        self.window_planes = {}
        for wid, pl in window_planes.items():
            self.window_planes[int(wid)] = {
                'plane_pt': np.array(pl['plane_pt'], dtype=np.float64),
                'plane_n': np.array(pl['plane_n'], dtype=np.float64),
                **{k: v for k, v in pl.items() if k not in ['plane_pt', 'plane_n']}
            }
        
        self.wand_length = wand_length
        self.config = config
        self.progress_callback = progress_callback  # For UI progress updates
        
        self._last_ray_rmse = -1.0
        self._last_len_rmse = -1.0
        self._last_s_ray = 0.0
        self._last_n_ray = 0
        self._last_s_len = 0.0
        self._last_n_len = 0

        
        self.active_cam_ids = sorted(list(self.cams_cpp.keys()))
        self.window_ids = sorted(list(self.window_planes.keys()))
        
        # Initial State storage (for priors)
        self.initial_cam_params = {cid: p.copy() for cid, p in self.cam_params.items()}
        self.initial_planes = {w: {k: (v.copy() if hasattr(v, 'copy') else v) for k, v in p.items()} for w, p in self.window_planes.items()}
        self.initial_media = {w: (m.copy() if hasattr(m, 'copy') else m) for w, m in self.window_media.items()}
        
        # Extract initial intrinsics (f)
        self.initial_f = {}
        for cid in self.active_cam_ids:
            # Assuming pinplate param structure
            pp = self.cams_cpp[cid]._pinplate_param
            # f = (fx + fy) / 2
            fx = pp.cam_mtx[0, 0]
            fy = pp.cam_mtx[1, 1]
            self.initial_f[cid] = (fx + fy) / 2.0

        # Build Observation Cache (same as PR4)
        self._build_obs_cache()
        
        # Build Camera->Window reverse map logic
        self.window_to_cams = {}
        for cid in self.active_cam_ids:
            wid = self.cam_to_window.get(cid)
            if wid not in self.window_to_cams: self.window_to_cams[wid] = []
            self.window_to_cams[wid].append(cid)
            
        # Observability Analysis
        self.observability = {}
        self.freeze_table = {} # {wid: {'plane':..., 'thick':..., 'cameras':{cid: {'tvec':..., 'rvec':..., 'f':...}}}}
        self._compute_observability_and_freeze()
        
    def _build_obs_cache(self, frames: Optional[List[int]] = None):
        """Build observation cache from dataset (optionally subset of frames)."""
        self.obs_cache = {} # {fid: {cid: (uvA, uvB)}}
        obsA = self.dataset.get('obsA', {})
        obsB = self.dataset.get('obsB', {})
        
        target_frames = frames
        if target_frames is None:
            # Default to all frames if not specified
            target_frames = sorted(list(set(obsA.keys()) | set(obsB.keys())))
        
        for fid in target_frames:
            self.obs_cache[fid] = {}
            for cid in self.active_cam_ids:
                uvA = None
                uvB = None
                
                if cid in obsA.get(fid, {}):
                    pt = obsA[fid][cid]
                    if pt is not None and len(pt) >= 2 and np.all(np.isfinite(pt[:2])):
                        uvA = pt[:2]
                
                if cid in obsB.get(fid, {}):
                    pt = obsB[fid][cid]
                    if pt is not None and len(pt) >= 2 and np.all(np.isfinite(pt[:2])):
                        uvB = pt[:2]
                        
                if uvA is not None or uvB is not None:
                    self.obs_cache[fid][cid] = (uvA, uvB)


    def build_pr5_frame_subset(self, all_frame_ids, max_frames, injected_frames_prev=None, rng=None):
        """
        Returns a list of frame_ids to use for this PR5 round.
        Strategy:
          - K_cap = ceil(0.10 * max_frames)
          - injected = first min(len(injected_frames_prev), K_cap) frames
          - remaining = max_frames - len(injected)
          - random sample remaining frames from (all_frame_ids - injected)
        """
        if injected_frames_prev is None:
            injected_frames_prev = []
            
        K_cap = int(np.ceil(0.10 * max_frames))
        
        # 1. Select Injected (Preserve order: worst first)
        injected = []
        for fid in injected_frames_prev:
            if len(injected) >= K_cap:
                break
            if fid in all_frame_ids:
                injected.append(fid)
                
        # 2. Random Sample Remaining
        needed = max_frames - len(injected)
        pool = sorted(list(set(all_frame_ids) - set(injected)))
        
        if rng is None:
            import random
            rng = random.Random(self.config.random_seed if hasattr(self.config, 'random_seed') else 42)
            
        if len(pool) <= needed:
            subset = injected + pool
        else:
            selected_random = sorted(rng.sample(pool, needed))
            subset = injected + selected_random
            
        # Log sampling
        if self.config.verbosity >= 1:
            print(f"[PR5][SAMPLE] max={max_frames}, K_cap={K_cap}, injected={len(injected)}, random={len(subset)-len(injected)}")
                
        return sorted(subset)

    def _compute_observability_and_freeze(self):
        """Compute observability metrics and build Freeze Table according to PR5 rules."""
        # Calculate N_cam, Baseline, Angle Diversity
        for wid in self.window_ids:
            info = ObservabilityInfo(wid, n_cam=0)
            cams = self.window_to_cams.get(wid, [])
            info.n_cam = len(cams)
            info.camera_ids = cams
            
            # 1. Baseline
            if info.n_cam >= 2:
                # Calculate baselines... reusing code from PR4 ideally, but rewriting for independence
                max_base = 0.0
                centers = []
                for cid in cams:
                    p = self.cam_params[cid]
                    R = rodrigues_to_R(p[0:3])
                    C = -R.T @ p[3:6]
                    centers.append(C)
                
                for i in range(len(centers)):
                    for j in range(i+1, len(centers)):
                        dist = np.linalg.norm(centers[i] - centers[j])
                        max_base = max(max_base, dist)
                info.baseline_max_mm = max_base
            
            # 2. Angle Diversity
            angles = []
            if info.n_cam >= 2:
                 for i in range(len(cams)):
                    for j in range(i+1, len(cams)):
                        p1 = self.cam_params[cams[i]]
                        p2 = self.cam_params[cams[j]]
                        # Angle between optical axes (Z)
                        z1 = rodrigues_to_R(p1[0:3])[:, 2]
                        z2 = rodrigues_to_R(p2[0:3])[:, 2]
                        ang = np.degrees(np.arccos(np.clip(np.dot(z1, z2), -1.0, 1.0)))
                        angles.append(ang)
            
            if angles:
                info.angle_diversity_p50 = np.median(angles)
                info.angle_diversity_p90 = np.percentile(angles, 90) if len(angles) >= 2 else angles[0]
            else:
                info.angle_diversity_p50 = 0.0
                info.angle_diversity_p90 = 0.0
                
            self.observability[wid] = info

            
            # 3. Rules Application
            ft = {
                'plane': FreezeStatus.OPTIMIZE,
                'thickness': FreezeStatus.OPTIMIZE_STRONG, # Always strong
                'cameras': {}
            }
            
            for cid in cams:
                cft = {'f': FreezeStatus.OPTIMIZE_STRONG} # F always strong
                
                # Extrinsics Logic
                if info.n_cam == 1:
                    # Single cam: Freeze all extrinsics to hold gauge
                    cft['tvec'] = FreezeStatus.FREEZE
                    cft['rvec'] = FreezeStatus.FREEZE
                elif info.n_cam == 2:
                    if info.angle_diversity_p50 < 15.0:
                         cft['tvec'] = FreezeStatus.OPTIMIZE 
                         # Weak/Strong logic per user spec
                         cft['rvec'] = FreezeStatus.OPTIMIZE_STRONG if self.config.allow_weak_rvec else FreezeStatus.FREEZE
                    elif info.angle_diversity_p50 < 25.0:
                         cft['tvec'] = FreezeStatus.OPTIMIZE
                         cft['rvec'] = FreezeStatus.OPTIMIZE_STRONG
                    else: # >= 25
                         cft['tvec'] = FreezeStatus.OPTIMIZE
                         cft['rvec'] = FreezeStatus.OPTIMIZE
                else: # N >= 3
                    cft['tvec'] = FreezeStatus.OPTIMIZE
                    cft['rvec'] = FreezeStatus.OPTIMIZE
                
                ft['cameras'][cid] = cft
            
            self.freeze_table[wid] = ft
    
    def _print_freeze_table(self):
        """Print PR5 observability and freeze table."""
        if self.config.verbosity >= 1:
            print(f"\n  [PR5] Optimization Freeze & Observability Table:")
            print(f"    {'WinID':<6} {'N_cam':<6} {'Base':<8} {'Ang(deg)':<8} {'Plane':<8} {'Thick':<8} {'Tv':<8} {'Rv':<8} {'F':<8}")
            print("    " + "-"*80)
            
            for wid in self.window_ids:
                info = self.observability[wid]
                ft = self.freeze_table[wid]
                
                # Summary for cams
                tv_stats = set()
                rv_stats = set()
                f_stats = set()
                
                for cid in info.camera_ids:
                     cft = ft['cameras'][cid]
                     tv_stats.add(cft['tvec'].value)
                     rv_stats.add(cft['rvec'].value)
                     f_stats.add(cft['f'].value)
                
                tv_str = "/".join(sorted(list(tv_stats))) if tv_stats else "N/A"
                rv_str = "/".join(sorted(list(rv_stats))) if rv_stats else "N/A"
                f_str = "/".join(sorted(list(f_stats))) if f_stats else "N/A"
                
                # Truncate if long
                if len(tv_str) > 8: tv_str = tv_str[:7]+".."
                if len(rv_str) > 8: rv_str = rv_str[:7]+".."
                
                print(f"    {wid:<6} {info.n_cam:<6} {info.baseline_max_mm:<8.1f} {info.angle_diversity_p90:<8.1f} "
                      f"{ft['plane'].value:<8} {ft['thickness'].value:<8} "
                      f"{tv_str:<8} {rv_str:<8} {f_str:<8}")
            print("    " + "-"*80 + "\n")

    def optimize(self) -> Tuple[Dict, Dict, Dict]:
        """Run PR5 optimization."""
        print("\n" + "="*60)
        print("PR5: ROBUST BUNDLE ADJUSTMENT (Geometric Only)")
        print("="*60)
        
        self._print_freeze_table()
        
        # [POLISH PREP] Populate self.frozen_cams based on config
        self.frozen_cams = []
        if self.freeze_table:
            # Use passed freeze table
            for wid, ft in self.freeze_table.items():
                if 'cameras' in ft:
                    for cid, c_status in ft['cameras'].items():
                        # If rvec is frozen (or any other logic), add to candidates
                        if c_status.get('rvec') == FreezeStatus.FREEZE:
                            self.frozen_cams.append(cid)
        else:
             # Fallback if no table (e.g. loaded from cache without it?)
             # Polish might be skipped or just target all?
             # For safety given current panic, if empty, we skip polish (warned).
             pass
             
        self.frozen_cams = sorted(list(set(self.frozen_cams)))
        
        # Calculate initial RMSE to avoid -1.0 display at start
        try:
             x0, _, _ = self._pack_parameters()
             if len(x0) > 0:
                 res = self._residuals_pr5(x0, self.config.lambda_len_init)
                 # Values are now set in self._last_s_ray/len 
                 self._last_ray_rmse = np.sqrt(self._last_s_ray / max(self._last_n_ray, 1))
                 self._last_len_rmse = np.sqrt(self._last_s_len / max(self._last_n_len, 1)) if self._last_n_len > 0 else 0.0
        except:
             pass
        
        # PR5.1 Main BA
        if self.config.pr5_stage >= 1:
             self._optimize_stage(stage_name="Optimizing camera intrinsic and extrinsic parameters and window parameters...", max_rounds=5, tight=False)
        
        # [DISABLED] PR5.2 Final Joint
        # if self.config.pr5_stage >= 2:
        #    self._optimize_stage(stage_name="Optimizing camera intrinsic and extrinsic parameters and window parameters...", max_rounds=3, tight=True)
            
        return self.window_planes, self.cam_params, self.window_media

    def _compute_side_violation(self, x):
        """
        Compute side violation v = max(0, margin - min(sX)) and worst point info.
        
        Returns:
            v: float, violation gap (mm)
            worst_info: tuple with worst violating point info
            all_sX_info: list of (sX, fid, wid, endpoint) tuples
            all_sC_info: list of (sC, cid, wid) tuples
        """
        cfg = self.config
        
        # Unpack and apply parameters
        planes_scalars, thick_map, f_map, rvec_map, tvec_map = self._unpack_parameters(x)
        planes_up = self._apply_params_to_cpp(planes_scalars, thick_map, f_map, rvec_map, tvec_map)
        
        all_sX_info = []
        all_sC_info = []
        
        # [STEP 2] Get estimated radii for PR5
        radius_A = self.dataset.get('est_radius_small_mm', 0.0)
        radius_B = self.dataset.get('est_radius_large_mm', 0.0)
        
        # Collect signed distances from obs_cache (matching _residuals_pr5)
        for fid in self.obs_cache:
            rays_A = []
            rays_B = []
            
            # Build rays
            for cid, (uvA, uvB) in self.obs_cache[fid].items():
                cam = self.cams_cpp[cid]
                wid = self.cam_to_window.get(cid)
                
                if uvA is not None:
                     rA = build_pinplate_ray_cpp(cam, uvA, cam_id=cid, window_id=wid, frame_id=fid, endpoint="A")
                     if rA.valid: rays_A.append(rA)
                if uvB is not None:
                     rB = build_pinplate_ray_cpp(cam, uvB, cam_id=cid, window_id=wid, frame_id=fid, endpoint="B")
                     if rB.valid: rays_B.append(rB)
            
            XA, _, vA, _ = triangulate_point(rays_A)
            XB, _, vB, _ = triangulate_point(rays_B)
            
            wids_A = set(r.window_id for r in rays_A if r.window_id is not None)
            wids_B = set(r.window_id for r in rays_B if r.window_id is not None)
            
            if vA:
                for wid in wids_A:
                    if wid not in planes_up:
                        continue
                    pl = planes_up[wid]
                    P_plane = np.array(pl['plane_pt'])
                    n = np.array(pl['plane_n'])
                    n = n / (np.linalg.norm(n) + 1e-12)
                    sX = float(np.dot(n, XA - P_plane))
                    all_sX_info.append((sX, fid, wid, 'A'))
            
            if vB:
                for wid in wids_B:
                    if wid not in planes_up:
                        continue
                    pl = planes_up[wid]
                    P_plane = np.array(pl['plane_pt'])
                    n = np.array(pl['plane_n'])
                    n = n / (np.linalg.norm(n) + 1e-12)
                    sX = float(np.dot(n, XB - P_plane))
                    all_sX_info.append((sX, fid, wid, 'B'))
        
        # Camera-side
        for cid in self.active_cam_ids:
            wid = self.cam_to_window.get(cid)
            if wid is None or wid not in planes_up:
                continue
            
            cp = self.cam_params[cid]
            rvec = cp[0:3]
            tvec = cp[3:6]
            R_cam = cv2.Rodrigues(rvec.reshape(3, 1))[0]
            C = -R_cam.T @ tvec
            
            pl = planes_up[wid]
            P_plane = np.array(pl['plane_pt'])
            n = np.array(pl['plane_n'])
            n = n / (np.linalg.norm(n) + 1e-12)
            
            sC = float(np.dot(n, C - P_plane))
            all_sC_info.append((sC, cid, wid))
        
        # Compute gaps
        gaps_with_info = []
        for (sX, fid, wid, endpoint) in all_sX_info:
            # [STEP 2] Check against radius
            # Limit is sX >= margin + R
            # Violation if sX < margin + R
            # Gap = (margin + R) - sX
            
            r_val = radius_A if endpoint == 'A' else radius_B
            limit = cfg.margin_side_mm + r_val
            gap = limit - sX
            
            gaps_with_info.append((gap, 'point', fid, wid, endpoint, sX))
        for (sC, cid, wid) in all_sC_info:
            gap = sC + cfg.margin_side_mm
            gaps_with_info.append((gap, 'cam', cid, wid, None, sC))
        
        if gaps_with_info:
            gaps_with_info.sort(key=lambda x: -x[0])
            worst = gaps_with_info[0]
            v = max(0.0, worst[0])
        else:
            worst = None
            v = 0.0
        
        return v, worst, all_sX_info, all_sC_info

    def _compute_j_data(self, x, lambda_len):
        """Compute J_data = S_ray + lambda_len * S_len for given parameters."""
        # Quick evaluation without full residual computation
        planes_up, thick_map, f_map, rvec_map, tvec_map = self._unpack_parameters(x)
        self._apply_params_to_cpp(planes_up, thick_map, f_map, rvec_map, tvec_map)
        
        s_ray = 0.0
        s_len = 0.0
        
        for fid in self.obs_cache:
            rays_A = []
            rays_B = []
            
            # Build rays
            for cid, (uvA, uvB) in self.obs_cache[fid].items():
                cam = self.cams_cpp[cid]
                wid = self.cam_to_window.get(cid)
                
                if uvA is not None:
                     rA = build_pinplate_ray_cpp(cam, uvA, cam_id=cid, window_id=wid, frame_id=fid, endpoint="A")
                     if rA.valid: rays_A.append(rA)
                if uvB is not None:
                     rB = build_pinplate_ray_cpp(cam, uvB, cam_id=cid, window_id=wid, frame_id=fid, endpoint="B")
                     if rB.valid: rays_B.append(rB)
            
            XA, _, vA, _ = triangulate_point(rays_A)
            XB, _, vB, _ = triangulate_point(rays_B)
            
            if vA and len(rays_A) > 0:
                O_A = np.vstack([r.o for r in rays_A])
                D_A = np.vstack([r.d for r in rays_A])
                dists_A = point_to_ray_dist_vec(XA, O_A, D_A)
                s_ray += np.sum(dists_A ** 2)
            
            if vB and len(rays_B) > 0:
                O_B = np.vstack([r.o for r in rays_B])
                D_B = np.vstack([r.d for r in rays_B])
                dists_B = point_to_ray_dist_vec(XB, O_B, D_B)
                s_ray += np.sum(dists_B ** 2)
            
            if vA and vB:
                L = np.linalg.norm(XA - XB)
                err = L - self.wand_length
                s_len += err ** 2
        
        return s_ray + lambda_len * s_len

    def _optimize_stage(self, stage_name: str, max_rounds: int, tight: bool = False):
        """Generic PR5 stage optimization."""
        print(f"\n  [{stage_name}] Optimization ({max_rounds} rounds)...")
        
        cfg = self.config
        lambda_len = cfg.lambda_len_init
        
        # [DEBUG] Start F
        if len(self.active_cam_ids) > 0:
             cid0 = self.active_cam_ids[0]
             print(f"    [DEBUG] Stage Start: Cam {cid0} F={self.cam_params[cid0][6]:.4f}")
        
        # Initial pack
        x0, bounds, names = self._pack_parameters()
        
        if len(x0) == 0:
            print("    No parameters to optimize.")
            return

        print(f"    Num Params: {len(x0)}")
        
        # Initialize persistent gate state
        if not hasattr(self, '_gate_enabled'):
            self._gate_enabled = False
            
        # Initialize Injection State for this stage
        injected_frames = []
        
        # Robustly gather ALL frames from observations + definition
        # (Fixes issue where dataset['frames'] might be incomplete subset)
        f_def = set(self.dataset.get('frames', []))
        f_obsA = set(self.dataset.get('obsA', {}).keys())
        f_obsB = set(self.dataset.get('obsB', {}).keys())
        all_frames = sorted(list(f_def | f_obsA | f_obsB))
            
        max_frames_cfg = getattr(cfg, 'max_frames', 300)

        # Outer Loop
        for round in range(max_rounds):
             # -------------------------------------------------------------------------
             # 1. SAMPLING (Worst-K Injection)
             # -------------------------------------------------------------------------
             subset = self.build_pr5_frame_subset(all_frames, max_frames_cfg, injected_frames_prev=injected_frames)
             
             # Rebuild cache for this subset (Optimization uses this)
             self._build_obs_cache(subset)
             
             # -------------------------------------------------------------------------
             # 2. ROUND SETUP (Subsampled)
             # -------------------------------------------------------------------------
             # Must unpack current x0 to compute initial state for this round
             
             # Evaluate current v using initial x0 for this round (SUBSET ONLY)
             # This sets the gate based on what the optimizer "sees".
             v_current, worst_info, all_sX_info, all_sC_info = self._compute_side_violation(x0)
             
             # Hysteresis update
             if self._gate_enabled:
                 if v_current < cfg.v_off_side_gate:
                     self._gate_enabled = False
             else:
                 if v_current > cfg.v_on_side_gate:
                     self._gate_enabled = True
             
             # Compute J_ref at round start (using initial parameters, SUBSET)
             self._j_ref_for_round = self._compute_j_data(x0, lambda_len)
             
             # Store round-level gate state
             self._round_gate_enabled = self._gate_enabled
             
             # [USER REQUEST] Force gate ON for the final round to ensure side constraints 
             if round == max_rounds - 1:
                 if not self._round_gate_enabled:
                     print(f"    [PR5] Final Round: Forcing side gate ON.")
                 self._round_gate_enabled = True
             
             self._round_v_current = v_current
             
             # Log Round Header (Explicitly SUBSET)
             if cfg.verbosity >= 1:
                 # Helper to count violations in S_info
                 def count_viols(sX_info, sC_info):
                     n_X = 0
                     n_C = 0
                     radius_A = self.dataset.get('est_radius_small_mm', 0.0)
                     radius_B = self.dataset.get('est_radius_large_mm', 0.0)
                     for (sX, _, _, ep) in sX_info:
                         r_val = radius_A if ep == 'A' else radius_B
                         if sX < cfg.margin_side_mm + r_val: n_X += 1
                     for (sC, _, _) in sC_info:
                         if sC > -cfg.margin_side_mm: n_C += 1
                     return n_X, n_C

                 n_viol_X, n_viol_C = count_viols(all_sX_info, all_sC_info)
                 
                 print(f"\n[PR5][ROUND {round+1}][SUBSET] ({len(subset)} frames)")
                 print(f"  margin_mm      = {cfg.margin_side_mm:.4f}")
                 print(f"  v (subset)     = {v_current:.4f} mm")
                 print(f"  gate_enabled   = {self._round_gate_enabled}")
                 print(f"  violations     = {n_viol_X} points + {n_viol_C} cameras (in subset)")

             # Update lambda_len (Annealing?)
             if round > 0:
                 lambda_len = max(cfg.lambda_len_min, lambda_len * 0.8)
             
             # Scale lambda_len if gate is active
             lambda_len_eff = lambda_len
             if self._round_gate_enabled:
                 lambda_len_eff *= cfg.scale_len_gate_active

             # Residual wrapper
             self._res_call_count_pr5 = 0
             def residuals_wrapper_pr5(x):
                 res = self._residuals_pr5(x, lambda_len_eff)
                 self._res_call_count_pr5 += 1
                 if self.progress_callback and self._res_call_count_pr5 % 30 == 0:
                     try:
                         # Approx cost
                         c_approx = 0.5 * np.sum(res**2)
                         # Report actual tracked RMSEs (Subsampled)
                         r_rmse = np.sqrt(self._last_s_ray / max(self._last_n_ray, 1))
                         l_rmse = np.sqrt(self._last_s_len / max(self._last_n_len, 1)) if self._last_n_len > 0 else 0.0
                         self.progress_callback(stage_name, r_rmse, l_rmse, c_approx)
                     except:
                         pass
                 return res

             # -------------------------------------------------------------------------
             # 3. OPTIMIZE (Subsampled)
             # -------------------------------------------------------------------------
             # [USER REQUEST] Use tighter tolerance for final round only
             is_final_round = (round == max_rounds - 1)
             tol = 1e-8 if is_final_round else 1e-6
             
             # [FINAL ROUND POLISH] Unfreeze all 'FREEZE' params to 'OPTIMIZE_STRONG'
             if is_final_round:
                 if cfg.verbosity >= 1:
                     print("    [PR5] Final Round: Unfreezing 'FREEZE' parameters to 'OPTIMIZE_STRONG'.")
                 
                 # 1. Sync current x0 to internal objects so _pack_parameters has latest state
                 planes_scalars, thick_map, f_map, rvec_map, tvec_map = self._unpack_parameters(x0)
                 # We don't need full C++ sync here, just python dict update for packing
                 # Update window planes
                 planes_up = self._apply_params_to_cpp(planes_scalars, thick_map, f_map, rvec_map, tvec_map)
                 self.window_planes = planes_up # Temporary update for packing
                 for wid, pl in planes_up.items():
                    self.window_media[wid]['thickness'] = thick_map[wid]
                 
                 # Update cameras
                 for cid in self.active_cam_ids:
                     if cid in f_map: self.cam_params[cid][6] = f_map[cid]
                     if cid in rvec_map: self.cam_params[cid][0:3] = rvec_map[cid]
                     if cid in tvec_map: self.cam_params[cid][3:6] = tvec_map[cid]
                 
                 # 2. Update Freeze Table
                 # Iterate ALL windows/cameras in freeze table
                 for wid in self.freeze_table:
                     ft = self.freeze_table[wid]
                     if ft.get('plane') == FreezeStatus.FREEZE: ft['plane'] = FreezeStatus.OPTIMIZE_STRONG
                     if ft.get('thick') == FreezeStatus.FREEZE: ft['thick'] = FreezeStatus.OPTIMIZE_STRONG
                     
                     for cid in ft.get('cameras', {}):
                         cft = ft['cameras'][cid]
                         if cft.get('f') == FreezeStatus.FREEZE: cft['f'] = FreezeStatus.OPTIMIZE_STRONG
                         if cft.get('rvec') == FreezeStatus.FREEZE: cft['rvec'] = FreezeStatus.OPTIMIZE_STRONG
                         if cft.get('tvec') == FreezeStatus.FREEZE: cft['tvec'] = FreezeStatus.OPTIMIZE_STRONG
                 
                 # 3. Re-pack with expanded parameter set
                 x0, bounds, names = self._pack_parameters()
                 print(f"    [PR5] Expanded Param Set: {len(x0)} params")
             
             res = least_squares(
                 residuals_wrapper_pr5,
                 x0,
                 bounds=bounds,
                 loss='linear',
                 ftol=tol,
                 xtol=tol,
                 gtol=tol,
                 verbose=0,
                 max_nfev=50 if not tight else 20
             )
             x0 = res.x
             
             # Log Optimize Params (SUBSET)
             if cfg.verbosity >= 1:
                 # J_data = S_ray + lambda_len * S_len
                 J_data = self._last_s_ray + lambda_len_eff * self._last_s_len
                 print(f"    [SUBSET] Res: J_data={J_data:.4f}, nfev={res.nfev}, msg={res.message}")

             # -------------------------------------------------------------------------
             # 4. FULL AUDIT (Full Dataset)
             # -------------------------------------------------------------------------
             # Sync params to C++ for full audit
             planes_scalars, thick_map, f_map, rvec_map, tvec_map = self._unpack_parameters(x0)
             planes_up = self._apply_params_to_cpp(planes_scalars, thick_map, f_map, rvec_map, tvec_map)
             
             # PERSIST optimized planes to self.window_planes so they are returned correctly
             self.window_planes = planes_up
             
             # [CRITICAL FIX] PERSIST optimized camera params to self.cam_params
             # _apply_params_to_cpp updates C++ but NOT the python dicts we return/save!
             for cid in self.active_cam_ids:
                 # Update f
                 if cid in f_map:
                     self.cam_params[cid][6] = f_map[cid]
                 # Update rvec
                 if cid in rvec_map:
                     self.cam_params[cid][0:3] = rvec_map[cid]
                 # Update tvec
                 if cid in tvec_map:
                     self.cam_params[cid][3:6] = tvec_map[cid]
             
             Rs = self.dataset.get('est_radius_small_mm', 0.0) # Correct key from code context?
             # Wait, prev code used self.dataset.get('est_radius_small_mm', 0.0) in _compute_side_violation
             Rl = self.dataset.get('est_radius_large_mm', 0.0)
             
             # Call module-level full audit
             min_sX_full, viols_full, worst_full, frame_details, min_dist_full = evaluate_plane_side_constraints(
                self.dataset, planes_up, self.cam_params, self.cams_cpp,
                self.cam_to_window, Rs, Rl, cfg.margin_side_mm
             )
             
             # Log Full Audit
             if cfg.verbosity >= 1:
                 print(f"  [PR5][FULL_AUDIT] min_sX(raw)={min_dist_full:.5f}mm, min_g={min_sX_full:.4f}mm, violations={len(viols_full)}")
                 if worst_full:
                     print(f"    Top: {worst_full['gap']:.4f}mm gap at F{worst_full['frame']} W{worst_full['window']} {worst_full['endpoint']}")
                     
             # -------------------------------------------------------------------------
             # 5. PREPARE NEXT ROUND (Injection)
             # -------------------------------------------------------------------------
             # Analyze audit for injection
             worst_sorted, near_sorted = analyze_worst_frames(frame_details)
             
             # Strategy: Inject violating frames first, then near-boundary
             # K_cap is handled in build_pr5_frame_subset, but we define the candidate list here.
             # We construct 'injected_frames' list for next round
             
             candidates = []
             # 1. Violations (Worst first)
             for item in worst_sorted:
                 candidates.append(item['frame'])
                 
             # 2. Near Boundary (Worst first - closest to violation)
             for item in near_sorted:
                 if item['frame'] not in candidates:
                     candidates.append(item['frame'])
                     
             # Pass ALL candidates to build_pr5_frame_subset?
             # No, build_pr5_frame_subset takes 'injected_frames_prev' and truncates to K_cap.
             # So we just pass the sorted candidate list.
             injected_frames = candidates
             
             # Progress callback
             if self.progress_callback is not None:
                 try:
                     # Report actual component RMSEs
                     ray_rmse = np.sqrt(self._last_s_ray / max(self._last_n_ray, 1))
                     len_rmse = np.sqrt(self._last_s_len / max(self._last_n_len, 1)) if self._last_n_len > 0 else 0.0
                     self.progress_callback(stage_name, ray_rmse, len_rmse, res.cost)
                     self._last_ray_rmse = ray_rmse
                     self._last_len_rmse = len_rmse
                 except Exception:
                     pass  # Silently ignore callback errors

             
        # Commit Final State
        planes_up, thick_map, f_map, rvec_map, tvec_map = self._unpack_parameters(x0)
        
        # 1. Update C++ and get final planes
        final_planes = self._apply_params_to_cpp(planes_up, thick_map, f_map, rvec_map, tvec_map)
        
        # 2. Update Python state
        for wid, pl in final_planes.items():
            self.window_planes[wid] = pl
            self.window_media[wid]['thickness'] = thick_map[wid]
            
        for cid in self.active_cam_ids:
            if cid in rvec_map:
                self.cam_params[cid][0:3] = rvec_map[cid]
            if cid in tvec_map:
                self.cam_params[cid][3:6] = tvec_map[cid]
            if cid in f_map:
                self.cam_params[cid][6] = f_map[cid]
        
        # 3. Stage Report
        print(f"  [{stage_name}] Complete.")
        if cfg.verbosity >= 1:
            mean_f = np.mean([f_map[c] for c in self.active_cam_ids])
            mean_t = np.mean([thick_map[w] for w in self.window_ids])
            f_delta = mean_f - np.mean(list(self.initial_f.values()))
            t_delta = mean_t - np.mean([self.initial_media[w]['thickness'] for w in self.window_ids])
            print(f"    Avg F: {mean_f:.2f} (diff: {f_delta:+.2f})")
            print(f"    Avg Thick: {mean_t:.4f} (diff: {t_delta:+.4f} mm)")

        # [DEBUG] Verify F persistence
        if len(self.active_cam_ids) > 0:
             cid0 = self.active_cam_ids[0]
             print(f"    [DEBUG] Stage End: Cam {cid0} F={self.cam_params[cid0][6]:.4f}")

    def _get_cache_path(self, dataset_path: str) -> str:
        """Get path to PR5 cache file."""
        return str(Path(dataset_path).parent / "pr5_bundle_cache.json")

    def try_load_cache(self, out_path: str) -> bool:
        """
        Try to load PR5 results from cache.
        Returns True if loaded successfully.
        """
        cache_path = self._get_cache_path(out_path)
        if not os.path.exists(cache_path):
            return False
            
        try:
            with open(cache_path, 'r') as f:
                data = json.load(f)
                
            # Validation
            # Check camera IDs match
            cached_cams = set(data.get('cam_ids', []))
            current_cams = set(self.active_cam_ids)
            if cached_cams != current_cams:
                print(f"[PR5][CACHE] Mismatch cam IDs: {cached_cams} vs {current_cams}")
                # We can accept mismatch if dataset is fresh? Maybe risky for now.
                return False
                
            # Check Window IDs
            cached_wins = set(map(int, data.get('window_ids', [])))
            current_wins = set(self.window_ids)
            if cached_wins != current_wins:
                print(f"[PR5][CACHE] Mismatch Window IDs")
                return False
                
            # Load Data (Parameters Only)
            print(f"[PR5][CACHE] Loading PR5 params from {cache_path}")
            
            # 1. Window Planes
            # Check format ('window_planes' vs 'planes')
            # The save_cache code uses 'planes' in PR4 but let's check PR5 save code.
            # Assuming typical JSON structure.
            
            # Robust load for planes
            planes_src = data.get('window_planes', data.get('planes', {}))
            for wid_str, pl in planes_src.items():
                wid = int(wid_str)
                if wid in self.window_planes:
                    self.window_planes[wid]['plane_pt'] = np.array(pl['plane_pt'])
                    self.window_planes[wid]['plane_n'] = np.array(pl['plane_n'])
                    if 'd' in pl: # Also support d/alpha/beta if stored
                        pass 
            
            # 2. Window Media
            media_src = data.get('window_media', data.get('media', {}))
            for wid_str, m in media_src.items():
                wid = int(wid_str)
                if wid in self.window_media:
                    # Update thickness
                    if 'thickness' in m:
                        self.window_media[wid]['thickness'] = float(m['thickness'])
                
            # 3. Camera Params
            cam_src = data.get('cam_params', {})
            for cid_str, p in cam_src.items():
                cid = int(cid_str)
                if cid in self.cam_params:
                    self.cam_params[cid] = np.array(p)
                
            # 4. Update C++ objects
            # Use internal helper to push everything to C++
            self._apply_all_to_cpp()
            
            # 5. RE-BUILD OBSERVATION CACHE (Important!)
            # Ensure obs_cache matches the FRESH dataset, not whatever was implicitly loaded
            # or left over. This ensures 'frames' list and 'obsA/B' coincide.
            # self.dataset was UNTOUCHED by this load function.
            self._build_obs_cache()
            
            print(f"[PR5][CACHE] Success. Params loaded, used FREH dataset for obs.")
            return True
            
        except Exception as e:
            print(f"[PR5][CACHE] Load failed: {e}")
            return False

        except Exception as e:
            print(f"[PR5][CACHE] Save failed: {e}")

    def _update_cpp_camera(self, cid):
        """Update a single C++ camera from self.cam_params."""
        if cid not in self.cams_cpp: return
        cam = self.cams_cpp[cid]
        cp = self.cam_params[cid]
        
        # Unpack params
        rvec = cp[0:3]
        tvec = cp[3:6]
        f = cp[6]
        dist = cp[7:] 
        
        try:
            # [CPP_PROTOCOL] Manual Update via _pinplate_param
            pp = cam._pinplate_param
            
            # 1. Extrinsics
            R_lpt = lpt.MatrixDouble(3, 3, 0.0)
            R_np, _ = cv2.Rodrigues(rvec)
            for i in range(3):
                for j in range(3): R_lpt[i,j] = float(R_np[i,j])
            pp.r_mtx = R_lpt
            
            t_pt = lpt.Pt3D(float(tvec[0]), float(tvec[1]), float(tvec[2]))
            pp.t_vec = t_pt
            
            # Inverse Extrinsics
            R_inv = R_np.T
            t_inv = -R_inv @ tvec
            R_inv_lpt = lpt.MatrixDouble(3, 3, 0.0)
            for i in range(3):
                for j in range(3): R_inv_lpt[i,j] = float(R_inv[i,j])
            pp.r_mtx_inv = R_inv_lpt
            pp.t_vec_inv = lpt.Pt3D(float(t_inv[0]), float(t_inv[1]), float(t_inv[2]))
            
            # 2. Intrinsics (f only, preserve cx,cy)
            # Use fresh MatrixDouble to be safe (match _apply_params_to_cpp pattern)
            K = lpt.MatrixDouble(3, 3, 0.0)
            K[0, 0] = float(f)
            K[1, 1] = float(f)
            K[0, 2] = float(pp.cam_mtx[0, 2]) # Preserve cx
            K[1, 2] = float(pp.cam_mtx[1, 2]) # Preserve cy
            K[2, 2] = 1.0
            pp.cam_mtx = K
            
            # 3. Distortion
            if len(dist) > 0:
                pp.dist_coeff = [float(x) for x in dist]
                pp.n_dist_coeff = len(dist)
            
            cam._pinplate_param = pp
            cam.updatePt3dClosest()
            
        except Exception as e:
             print(f"[Warning] Camera update failed for cid {cid}: {e}")

    def _apply_all_to_cpp(self):
        """Apply current Python state (self.cam_params, self.window_planes, self.window_media) to C++."""
        # 1. Update Camera Intrinsics/Extrinsics
        for cid in self.active_cam_ids:
            self._update_cpp_camera(cid)
            
        # 2. Update Window Planes (Geometry)
        for wid, pl in self.window_planes.items():
            if wid not in self.window_media: continue
            media = self.window_media[wid]
            thick = media.get('thickness', 10.0)
            
            n = np.array(pl['plane_n'])
            pt = np.array(pl['plane_pt'])
            
            # Apply to all cameras sharing this window
            cams = self.window_to_cams.get(wid, [])
            for cid in cams:
                if cid not in self.cams_cpp: continue
                cam = self.cams_cpp[cid]
                
                try:
                    pp = cam._pinplate_param
                    
                    # Refractive Props
                    pp.n1 = media['n1']
                    pp.n2 = media['n2']
                    pp.n3 = media['n3']
                    pp.w_array = [float(thick)]
                    
                    # Plane Geometry
                    plane_obj = pp.plane
                    plane_obj.pt = lpt.Pt3D(float(pt[0]), float(pt[1]), float(pt[2]))
                    plane_obj.norm_vector = lpt.Pt3D(float(n[0]), float(n[1]), float(n[2]))
                    pp.plane = plane_obj
                    
                    cam._pinplate_param = pp
                    cam.updatePt3dClosest() 
                    
                except Exception as e:
                    print(f"[Warning] Plane update failed for cid {cid} (win {wid}): {e}")

    def _pack_parameters(self) -> Tuple[np.ndarray, List[Tuple[float, float]], List[str]]:
        """
        Pack optimization parameters into vector x.
        Order:
        - For each window: [d, alpha, beta, thick]
        - For each camera: [f, rvec(3), tvec(3)]
        """
        x = []
        bounds_lower = []
        bounds_upper = []
        names = []
        
        cfg = self.config
        
        # 1. Windows
        for wid in self.window_ids:
            pl = self.window_planes[wid]
            media = self.window_media[wid]
            ft = self.freeze_table.get(wid, {})
            
            # Plane (d, alpha, beta) - only if not frozen
            if ft.get('plane') != FreezeStatus.FREEZE:
                # d (delta)
                x.append(0.0)
                names.append(f"w{wid}_d")
                d_bound = cfg.bounds_d_delta_mm
                bounds_lower.append(-d_bound)
                bounds_upper.append(d_bound)
                
                # alpha, beta
                x.extend([0.0, 0.0])
                names.append(f"w{wid}_a"); names.append(f"w{wid}_b")
                ab_bound = np.radians(cfg.bounds_alpha_beta_deg)
                bounds_lower.extend([-ab_bound, -ab_bound])
                bounds_upper.extend([ab_bound, ab_bound])
            
            # Thickness - only if not frozen
            if ft.get('thick') != FreezeStatus.FREEZE:
                t_curr = media['thickness']
                x.append(t_curr)
                names.append(f"w{wid}_t")
                # Bounds +/- 5%
                t_min = t_curr * (1.0 - cfg.bounds_thick_pct)
                t_max = t_curr * (1.0 + cfg.bounds_thick_pct)
                bounds_lower.append(t_min)
                bounds_upper.append(t_max)

            
        # 2. Cameras
        for cid in self.active_cam_ids:
            # Find which window controls this camera to get freeze status
            wid = self.cam_to_window.get(cid)
            if wid not in self.freeze_table:
                # Missing window in freeze table (consistency error or skipped window)
                continue
            cft = self.freeze_table[wid]['cameras'][cid]
            
            # Focal (f)
            if cft['f'] != FreezeStatus.FREEZE:
                f_val = self.cam_params[cid][6] # Use current state, not absolute initial
                x.append(f_val)
                names.append(f"c{cid}_f")
                # Bounds +/- 2% from absolute initial
                f0 = self.initial_f[cid]
                f_min = f0 * (1.0 - cfg.bounds_f_pct)
                f_max = f0 * (1.0 + cfg.bounds_f_pct)
                bounds_lower.append(f_min)
                bounds_upper.append(f_max)
            
            # Rvec
            r_curr = self.cam_params[cid][0:3]
            status_r = cft['rvec']
            if status_r != FreezeStatus.FREEZE:
                x.extend(r_curr)
                names.extend([f"c{cid}_r0", f"c{cid}_r1", f"c{cid}_r2"])
                # Bounds? Rvec usually unbound or large bounds.
                # Prior will constrain it.
                inf = np.inf
                bounds_lower.extend([-inf, -inf, -inf])
                bounds_upper.extend([inf, inf, inf])
            
            # Tvec
            t_curr = self.cam_params[cid][3:6]
            status_t = cft['tvec']
            if status_t != FreezeStatus.FREEZE:
                x.extend(t_curr)
                names.extend([f"c{cid}_tx", f"c{cid}_ty", f"c{cid}_tz"])
                inf = np.inf
                bounds_lower.extend([-inf, -inf, -inf])
                bounds_upper.extend([inf, inf, inf])
            
        return np.array(x), (bounds_lower, bounds_upper), names

    def _unpack_parameters(self, x: np.ndarray) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
        """
        Unpack x into readable dicts.
        Returns (dt_planes, dict_thick, dict_f, dict_rvec, dict_tvec)
        """
        idx = 0
        
        # Temp storage
        planes_update = {} # {wid: (d_delta, alpha, beta)}
        thick_new = {}
        f_new = {}
        rvec_new = {}
        tvec_new = {}
        
        # 1. Windows (must match pack order exactly)
        for wid in self.window_ids:
            ft = self.freeze_table.get(wid, {})
            
            # Plane - only if not frozen
            if ft.get('plane') != FreezeStatus.FREEZE:
                d_delta = x[idx]; idx += 1
                alpha = x[idx]; idx += 1
                beta = x[idx]; idx += 1
                planes_update[wid] = (d_delta, alpha, beta)
            else:
                # Use identity (no change)
                planes_update[wid] = (0.0, 0.0, 0.0)
            
            # Thickness - only if not frozen
            if ft.get('thick') != FreezeStatus.FREEZE:
                thick_new[wid] = x[idx]; idx += 1
            else:
                thick_new[wid] = self.window_media[wid]['thickness']
            
        # 2. Cameras
        for cid in self.active_cam_ids:
            wid = self.cam_to_window.get(cid)
            cft = self.freeze_table[wid]['cameras'][cid]
            
            # F
            if cft['f'] != FreezeStatus.FREEZE:
                f_new[cid] = x[idx]; idx += 1
            else:
                f_new[cid] = self.cam_params[cid][6] # Stay at current
                
            # Rvec
            if cft['rvec'] != FreezeStatus.FREEZE:
                rvec_new[cid] = x[idx:idx+3]; idx += 3
            else:
                rvec_new[cid] = self.cam_params[cid][0:3]
                
            # Tvec
            if cft['tvec'] != FreezeStatus.FREEZE:
                tvec_new[cid] = x[idx:idx+3]; idx += 3
            else:
                tvec_new[cid] = self.cam_params[cid][3:6]
        
        return planes_update, thick_new, f_new, rvec_new, tvec_new


    def _apply_params_to_cpp(self, planes_update, thick_map, f_map, rvec_map, tvec_map):
        """
        Update C++ objects with temporary parameters.
        Returns: updated_planes_dict (for residual calc)
        """
        updated_planes = {}
        
        # Update Windows
        for wid, (d_delta, alpha, beta) in planes_update.items():
            # Update thickness in window_media (not CPP directly? CPP Camera holds thickness in PinPlate)
            # Actually C++ Camera has _pinplate_param.thickness and .plane
            
            # Compute new plane
            pl_init = self.window_planes[wid] # Current base
            n_init = pl_init['plane_n']
            pt_init = pl_init['plane_pt']
            
            # Alpha/Beta update logic
            # Use 'initial_planes' as basis for tangent space or current?
            # PR5 spec: "local incremental parameterization around CURRENT normal"
            # But here x[alpha] is optimized from 0. So it's relative to the start of 'optimize' call (or step).
            # If we call this inside residuals, and x is from solver, it's relative to self.window_planes.
            
            n_new = update_normal_tangent(n_init, alpha, beta)
            
            # d update: pt_new = pt_init + n_new * d_delta
            # PR5 spec: d_delta along n_new
            pt_new = pt_init + n_new * d_delta
            
            # Thickness from thick_map
            t_val = thick_map[wid]
            
            updated_planes[wid] = {
                'plane_pt': pt_new, 
                'plane_n': n_new,
                'initialized': True,
                'thick_mm': t_val
            }
            
            # Update all cameras attached to this window
            for cid in self.window_to_cams.get(wid, []):
                cam = self.cams_cpp[cid]
                
                # Extrinsics & F Update
                rv = rvec_map[cid]
                tv = tvec_map[cid]
                f = f_map[cid]
                
                # Update Intrinsics (Keep cx, cy fixed from initial state)
                pp = cam._pinplate_param
                
                # Standardize MatrixDouble population via indexing (bypass initializer_list issues)
                K = lpt.MatrixDouble(3, 3, 0.0)
                K[0, 0] = float(f)
                K[1, 1] = float(f)
                K[0, 2] = float(pp.cam_mtx[0, 2])
                K[1, 2] = float(pp.cam_mtx[1, 2])
                K[2, 2] = 1.0
                pp.cam_mtx = K
                
                # Update Extrinsics (Forward)
                R_np = rodrigues_to_R(rv)
                R_lpt = lpt.MatrixDouble(3, 3, 0.0)
                for i in range(3):
                    for j in range(3):
                        R_lpt[i, j] = float(R_np[i, j])
                pp.r_mtx = R_lpt
                pp.t_vec = lpt.Pt3D(float(tv[0]), float(tv[1]), float(tv[2]))
                
                # Update Extrinsics (Inverse) - Crucial for pinplateLine
                R_inv_np = R_np.T
                t_inv_np = -R_inv_np @ tv
                R_inv_lpt = lpt.MatrixDouble(3, 3, 0.0)
                for i in range(3):
                    for j in range(3):
                        R_inv_lpt[i, j] = float(R_inv_np[i, j])
                pp.r_mtx_inv = R_inv_lpt
                pp.t_vec_inv = lpt.Pt3D(float(t_inv_np[0]), float(t_inv_np[1]), float(t_inv_np[2]))
                
                # Update Plane & Thickness
                pp.w_array = [float(t_val)]
                
                # Re-apply updated Plane struct
                pl_internal = pp.plane
                # [CRITICAL] Coordinate Alignment: Shift to Farthest Interface for C++ engine
                p_farthest = pt_new + n_new * t_val
                pl_internal.pt = lpt.Pt3D(float(p_farthest[0]), float(p_farthest[1]), float(p_farthest[2]))
                pl_internal.norm_vector = lpt.Pt3D(float(n_new[0]), float(n_new[1]), float(n_new[2]))
                pp.plane = pl_internal
                
                # Commit everything back to Camera object
                cam._pinplate_param = pp
                cam.updatePt3dClosest() # Refresh internal geometric state
                    
        return updated_planes

    def _residuals_pr5(self, x: np.ndarray, lambda_len: float) -> np.ndarray:
        """
        PR5 Cost Function:
        Ray Dist + Wand Length + Strong Priors
        """
        # Unpack
        # Unpack
        planes_scalars, thick_map, f_map, rvec_map, tvec_map = self._unpack_parameters(x)
        
        # Apply to C++ and get updated plane geometry
        planes_up = self._apply_params_to_cpp(planes_scalars, thick_map, f_map, rvec_map, tvec_map)
        
        res_ray = []
        res_len = []
        priors = []

        cfg = self.config
        
        # Track side violations for logging
        all_sX = []
        
        # 1. Observations (Ray + Wand)
        for fid in self.obs_cache:

            rays_A = []
            rays_B = []
            
            # Build rays
            for cid, (uvA, uvB) in self.obs_cache[fid].items():
                cam = self.cams_cpp[cid]
                wid = self.cam_to_window.get(cid)
                
                if uvA is not None:
                     rA = build_pinplate_ray_cpp(cam, uvA, cam_id=cid, window_id=wid, frame_id=fid, endpoint="A")
                     if rA.valid: rays_A.append(rA)
                if uvB is not None:
                     rB = build_pinplate_ray_cpp(cam, uvB, cam_id=cid, window_id=wid, frame_id=fid, endpoint="B")
                     if rB.valid: rays_B.append(rB)
            
            # Triangulate
            XA, _, vA, _ = triangulate_point(rays_A)
            XB, _, vB, _ = triangulate_point(rays_B)
            
            # Ray Residuals (VECTORIZED for performance)
            if vA and len(rays_A) > 0:
                # Stack ray data into numpy arrays
                O_A = np.vstack([r.o for r in rays_A])  # (N, 3)
                D_A = np.vstack([r.d for r in rays_A])  # (N, 3)
                # Vectorized half-line distance
                dists_A = point_to_ray_dist_vec(XA, O_A, D_A)
                res_ray.extend(dists_A.tolist())
            
            if vB and len(rays_B) > 0:
                O_B = np.vstack([r.o for r in rays_B])
                D_B = np.vstack([r.d for r in rays_B])
                dists_B = point_to_ray_dist_vec(XB, O_B, D_B)
                res_ray.extend(dists_B.tolist())

            
            # Wand Length
            if vA and vB:
                L = np.linalg.norm(XA - XB)
                err = L - self.wand_length
                res_len.append(err)
            
            # Side Gate: Collect signed distances with source info for debug
            # sX > margin_side_mm is OK, sX < margin_side_mm is violation
            # Store (sX, fid, wid, endpoint) tuples for worst-point identification
            
            # Get unique window IDs from rays used in triangulation
            wids_A = set(r.window_id for r in rays_A if r.window_id is not None)
            wids_B = set(r.window_id for r in rays_B if r.window_id is not None)

            
            # XA signed distances (Differentiable)
            if vA:
                for wid in wids_A:
                    if wid not in planes_up: continue
                    pl = planes_up[wid]
                    P_plane = pl['plane_pt']
                    n = pl['plane_n']
                    sX = np.dot(n, XA - P_plane)
                    all_sX.append((sX, fid, wid, 'A'))
            
            # XB signed distances (Differentiable)
            if vB:
                for wid in wids_B:
                    if wid not in planes_up: continue
                    pl = planes_up[wid]
                    P_plane = pl['plane_pt']
                    n = pl['plane_n']
                    sX = np.dot(n, XB - P_plane)
                    all_sX.append((sX, fid, wid, 'B'))
        
        # Camera-side: Collect signed distances with source info
        # Camera should be on camera-side: s < 0 (i.e. s + margin < 0 is OK)
        all_sC = []  # (sC, cid, wid)
        for cid in self.active_cam_ids:
            wid = self.cam_to_window.get(cid)
            if wid not in planes_up: continue
            
            # Reconstruct C from C++ object (already updated by _apply_params_to_cpp)
            cam = self.cams_cpp[cid]
            pp = cam._pinplate_param
            
            # Extract R, t
            R_lpt = pp.r_mtx
            t_lpt = pp.t_vec
            
            # Manual conversion to numpy (fast enough for small N_cam)
            R_np = np.zeros((3, 3))
            for i in range(3):
                for j in range(3):
                    R_np[i, j] = R_lpt[i, j]
            t_np = np.array([t_lpt[0], t_lpt[1], t_lpt[2]])
            
            C = -R_np.T @ t_np
            
            pl = planes_up[wid]
            P_plane = pl['plane_pt']
            n = pl['plane_n']
            
            sC = np.dot(n, C - P_plane)
            all_sC.append((sC, cid, wid))
        
        # Debug Output for Convergence Check (Sampled)
        if self._res_call_count_pr5 % 20 == 0 and cfg.verbosity >= 1:
            # Check min sX to ensure it is moving
            sX_vals = [val[0] for val in all_sX]
            min_sx = min(sX_vals) if sX_vals else 0.0
            print(f"      [Step {self._res_call_count_pr5}] min(sX)={min_sx:.5f} mm")
        
        # 2. Priors


        # Plane update priors
        # Plane update priors
        for wid, (d_delta, alpha, beta) in planes_scalars.items():
            info = self.observability[wid]
            
            # Angle Sigma
            if info.n_cam == 1: sigma_ang = cfg.sigma_plane_ang_single
            elif info.angle_diversity_p50 < 15: sigma_ang = cfg.sigma_plane_ang_weak
            elif info.angle_diversity_p50 < 25: sigma_ang = cfg.sigma_plane_ang_mid
            else: sigma_ang = cfg.sigma_plane_ang_strong
            
            # D Sigma
            if info.n_cam == 1: sigma_d = cfg.sigma_d_single
            elif info.angle_diversity_p50 < 25: sigma_d = cfg.sigma_d_weak
            else: sigma_d = cfg.sigma_d_strong
            
            priors.append(alpha / sigma_ang)
            priors.append(beta / sigma_ang)
            priors.append(d_delta / sigma_d)
            
            # Thickness Prior (Strong: bounds +/- 5%, sigma ~2%)
            # t_init = self.initial_media[wid]['thickness'] # Base value from UI?
            # Yes, "strong prior to UI-given thickness"
            t_ui = self.window_media[wid]['thickness'] # Careful, this might have been updated by Packing?
            # Actually self.window_media is state at start of PR5.
            # Ideally use 'initial_media' which is copy at start.
            t_ui = self.initial_media[wid]['thickness']
            t_curr = thick_map[wid]
            sigma_t = (cfg.bounds_thick_pct * t_ui) / 3.0
            priors.append((t_curr - t_ui) / sigma_t)
            
        # Camera Priors
        for cid in self.active_cam_ids:
            wid = self.cam_to_window.get(cid)
            info = self.observability[wid]
            
            # F Prior
            f_curr = f_map[cid]
            f_init = self.initial_f[cid]
            sigma_f = (cfg.bounds_f_pct * f_init) / 3.0
            priors.append((f_curr - f_init) / sigma_f)
            
            # Rvec Prior
            if cid in rvec_map:
                r_curr = rvec_map[cid]
                r_init = self.initial_cam_params[cid][0:3]
                
                # Sigma Select
                if info.angle_diversity_p50 < 15: sigma_r = cfg.sigma_rvec_very_weak
                elif info.angle_diversity_p50 < 25: sigma_r = cfg.sigma_rvec_weak
                else: sigma_r = cfg.sigma_rvec_normal
                
                diff = r_curr - r_init
                priors.extend(diff / sigma_r)
            
            # Tvec Prior
            if cid in tvec_map:
                t_curr = tvec_map[cid]
                t_init = self.initial_cam_params[cid][3:6]
                
                if info.n_cam == 2 and info.angle_diversity_p50 < 25:
                    sigma_t = cfg.sigma_tvec_weak
                else:
                    sigma_t = cfg.sigma_tvec_normal
                
                diff = t_curr - t_init
                priors.extend(diff / sigma_t)
                
        # Aggregate
        # Manual Huber Weighting for Ray/Len residuals
        # Priors remain pure quadratic (no robust loss)
        # Huber: if |u| <= 1: w = 1; else w = 1/|u|
        # Return sqrt(w) * u so that LM/TRF does weighted least squares
        
        def apply_huber(r_array, delta):
            """Apply huber weighting: u = r/delta, return sqrt(w)*u."""
            if len(r_array) == 0:
                return np.array([])
            r = np.asarray(r_array)
            u = r / delta  # Normalized residual
            abs_u = np.abs(u)
            w = np.where(abs_u <= 1.0, 1.0, 1.0 / abs_u)  # Huber weight
            # Return physical units (sqrt(w) * r) to match PR4 cost magnitude
            # Inliers: w=1 -> returns r (squared: r^2)
            # Outliers: w=delta/|r| -> returns sqrt(delta*|r|)*sgn(r) (squared: delta*|r|)
            return np.sqrt(w) * r
        
        arr_ray = apply_huber(res_ray, cfg.delta_ray) * np.sqrt(cfg.lambda_ray)
        arr_len = apply_huber(res_len, cfg.delta_len) * np.sqrt(lambda_len)
        arr_pri = np.array(priors)  # Priors: pure quadratic, no huber
        
        # Store components for display (Unweighted physical residuals)
        self._last_s_ray = np.sum(np.array(res_ray)**2)
        self._last_n_ray = len(res_ray)
        self._last_s_len = np.sum(np.array(res_len)**2)
        self._last_n_len = len(res_len)
        
        # === SIDE GATE: Hysteresis-Based Feasibility Constraint ===
        # Uses round-level gate state (computed once at round start, constant here)
        # Gate enabled: strong enforcement (r_gate = sqrt(2*C_gate), r_dir = sqrt(2*beta)*v)
        # Gate disabled: weak floor only (r_gate = 0, r_dir = sqrt(2*beta_soft)*v)
        
        
        cfg = self.config
        
        # Use round-level gate state (set at round start, unchanged during round)
        gate_enabled = getattr(self, '_round_gate_enabled', False)
        J_ref = getattr(self, '_j_ref_for_round', 1.0)
        
        # Per-violation residuals (Fixed Order, Fixed Length)
        # We must generate residuals for ALL points/cameras to maintain vector size for least_squares
        res_side = []
        
        # Pre-compute constants
        C_gate = cfg.alpha_side_gate * J_ref
        r_fix_const = np.sqrt(2.0 * C_gate)
        r_grad_const = np.sqrt(2.0 * cfg.beta_side_dir)
        r_soft_const = np.sqrt(2.0 * cfg.beta_side_soft)
        tau = 0.01
        
        radius_A = self.dataset.get('est_radius_small_mm', 0.0)
        radius_B = self.dataset.get('est_radius_large_mm', 0.0)

        # 1. Points
        for (sX, _, _, endpoint) in all_sX:
            r_val = radius_A if endpoint == 'A' else radius_B
            gap = (cfg.margin_side_mm + r_val) - sX
            if gap > 0:
                if gate_enabled:
                    # Gate ON: Smooth Step + Gradient
                    term_step = r_fix_const * (1.0 - np.exp(-gap / tau))
                    res_side.append(term_step)
                    res_side.append(r_grad_const * gap)
                else:
                    # Gate OFF: Weak Floor
                    res_side.append(r_soft_const * gap)
            else:
                # Feasible
                if gate_enabled:
                    res_side.append(0.0)
                    res_side.append(0.0)
                else:
                    res_side.append(0.0)

        # 2. Cameras
        for (sC, _, _) in all_sC:
            gap = sC + cfg.margin_side_mm
            if gap > 0:
                if gate_enabled:
                    term_step = r_fix_const * (1.0 - np.exp(-gap / tau))
                    res_side.append(term_step)
                    res_side.append(r_grad_const * gap)
                else:
                    res_side.append(r_soft_const * gap)
            else:
                if gate_enabled:
                    res_side.append(0.0)
                    res_side.append(0.0)
                else:
                    res_side.append(0.0)
                    
        arr_side = np.array(res_side)
        
        # Debug: Check parameter movement (first few calls)
        if self._res_call_count_pr5 < 5 and cfg.verbosity >= 2:
            planes_up, _, _, _, _ = self._unpack_parameters(x)
            # Just print one plane centroid to see if it moves
            if 1 in planes_up:
                p1 = planes_up[1]['plane_pt']
                print(f"      [Step {self._res_call_count_pr5}] Plane1 Pt: {p1[0]:.4f}, {p1[1]:.4f}, {p1[2]:.4f}")
        
        arr_side = np.array(res_side)
        
        # Store current J_data
        J_data_current = self._last_s_ray + lambda_len * self._last_s_len
        self._last_j_data = J_data_current

        return np.concatenate([arr_ray, arr_len, arr_pri, arr_side])

    def save_cache(self, out_path: str, points_3d: list = None):
        """Save PR5 results to cache (with Feasibility Audit)."""
        try:
            # Audit constraints before saving
            # Sync internal C++ state first
            for cid in self.cam_params:
                self._update_cpp_camera(cid)
                
            Rs = self.dataset.get('est_radius_small_mm', 2.0)
            Rl = self.dataset.get('est_radius_large_mm', 5.0)
            
            min_sX, viols, worst, _, _ = evaluate_plane_side_constraints(
                self.dataset, self.window_planes, self.cam_params, self.cams_cpp,
                self.cam_to_window, Rs, Rl, self.config.margin_side_mm
            )
            
            is_feasible = (len(viols) == 0)
            if not is_feasible:
                print(f"[PR5][WARNING] Final solution is NOT feasible: violations={len(viols)}")
                if worst:
                    print(f"  Worst: {worst['gap']:.4f}mm violation at W{worst['window']} F{worst['frame']}")

            cache_path = str(Path(out_path).parent / "pr5_bundle_cache.json")
            
            data = {
                'timestamp': str(datetime.now()),
                'feasible': is_feasible,
                'min_sX_gap': min_sX,
                'violations_count': len(viols),
                'cam_ids': self.active_cam_ids,
                'window_ids': self.window_ids,
                'planes': {
                    str(w): {
                        'plane_pt': np.asarray(pl['plane_pt']).tolist(),
                        'plane_n': np.asarray(pl['plane_n']).tolist(),
                        'd_delta': pl.get('d_delta', 0.0),
                        'alpha': pl.get('alpha', 0.0),
                        'beta': pl.get('beta', 0.0)
                    } for w, pl in self.window_planes.items()
                },
                'cam_params': {
                    str(c): np.asarray(p).tolist() for c, p in self.cam_params.items()
                },
                'window_media': {
                   str(w): m for w, m in self.window_media.items()
                }
            }
            
            # Add 3D points if provided
            if points_3d is not None:
                data['points_3d'] = points_3d
            
            with open(cache_path, 'w') as f:
                json.dump(data, f, indent=2)
                
            print(f"[PR5][CACHE] Saved results to {cache_path} (feasible={is_feasible})")
            
        except Exception as e:
            print(f"[PR5][CACHE] Save failed: {e}")


# --------------------------------------------------------------------------------------
# PR5_POLISH: REMOVED
# --------------------------------------------------------------------------------------
# RefractivePolishOptimizer and PolishConfig were removed to resolve C++ 
# synchronization issues (non-existent set_rvec/set_tvec methods).
# See implementation_plan.md for details.
