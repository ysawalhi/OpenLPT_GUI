# DEBUG COPY — DO NOT USE IN PRODUCTION
# This is a debug copy of refractive_bootstrap.py with experimental fixes.
# Modification: Replaced cv2.recoverPose() with manual chirality-only selection (fix for anchor frame sensitivity).
# pyright: reportMissingImports=false, reportAttributeAccessIssue=false, reportMissingTypeArgument=false, reportUninitializedInstanceVariable=false
"""
Refractive Bootstrap (P0 Stage)
===============================
Frozen-Intrinsics Pinhole Bootstrap for Refractive Calibration.

This provides a physically reasonable extrinsic initialization for later stages.
It is NOT calibration - it ONLY initializes extrinsics.

KEY RULES:
- Intrinsics (fx, fy, cx, cy) are FROZEN to UI values
- NO distortion parameters
- NO camFile output (in-memory only)
- Uses 8-Point Algorithm for initialization (same as pinhole Phase 1)
- Only optimizes extrinsics (Phase 1 BA with frozen intrinsics)
- Wand length is the ONLY scale constraint
- Pair selection is handled externally via precalibrate
"""

import logging
import numpy as np
from scipy.optimize import least_squares
import cv2
from itertools import combinations
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import re

logger = logging.getLogger(__name__)


def select_best_pair_via_precalib(base_calibrator, wand_len_mm: float, initial_focal_px: float) -> Optional[Tuple[int, int]]:
    """
    Run pinhole precalibration to determine the best camera pair based on
    reprojection errors. Useful baseline even for refractive setups.
    
    Args:
        base_calibrator: WandCalibrator instance
        wand_len_mm: Target wand length in mm
        initial_focal_px: Initial focal length in pixels
        
    Returns:
        (cam_i, cam_j) tuple of best camera pair, or None if failed
    """
    print("\n[BOOT] Running Precalibration Check to select best pair...")
    
    try:
        if not hasattr(base_calibrator, 'run_precalibration_check'):
            raise AttributeError("run_precalibration_check is unavailable on base calibrator")
        ret, msg, precalib_result = base_calibrator.run_precalibration_check(
            wand_length=wand_len_mm,
            init_focal_length=initial_focal_px
        )
    except Exception as e:
        print(f"  [WARN] Precalibration failed: {e}. Falling back to shared count.")
        
        # Fallback: Select pair with most common frames
        counts = {}
        points = getattr(base_calibrator, 'wand_points_filtered', None) or getattr(base_calibrator, 'wand_points', {})
        if not points:
             return None
        
        all_cams = set()
        for fid, cams in points.items():
            cam_list = list(cams.keys())
            for i in range(len(cam_list)):
                for j in range(i+1, len(cam_list)):
                    c1, c2 = sorted((cam_list[i], cam_list[j]))
                    counts[(c1, c2)] = counts.get((c1, c2), 0) + 1
                    all_cams.add(c1)
                    all_cams.add(c2)
        
        if not counts:
             # Just pick first two cams
             cams = sorted(list(all_cams))
             if len(cams) >= 2:
                 return (cams[0], cams[1])
             return None

        # Sort by shared count desc; tie-break by median pixel disparity (larger is better).
        sorted_pairs = sorted(counts.items(), key=lambda x: x[1], reverse=True)

        def _extract_uv(pts):
            if pts is None:
                return None
            arr = np.asarray(pts)
            if arr.ndim == 1 and arr.shape[0] >= 2:
                return np.array(arr[:2], dtype=np.float64)
            if arr.ndim >= 2 and arr.shape[0] >= 1 and arr.shape[1] >= 2:
                # Use centroid of first two detected points (small/large) when available.
                n = min(2, arr.shape[0])
                return np.mean(arr[:n, :2].astype(np.float64), axis=0)
            return None

        def _pair_disparity(pair):
            c1, c2 = pair
            d = []
            for _, cams in points.items():
                if c1 not in cams or c2 not in cams:
                    continue
                uv1 = _extract_uv(cams[c1])
                uv2 = _extract_uv(cams[c2])
                if uv1 is None or uv2 is None:
                    continue
                disp = np.linalg.norm(uv1 - uv2)
                if np.isfinite(disp):
                    d.append(float(disp))
            if not d:
                return -1.0
            return float(np.median(d))

        top_count = sorted_pairs[0][1]
        top_pairs = [p for p, cnt in sorted_pairs if cnt == top_count]
        best_pair = max(top_pairs, key=_pair_disparity)
        print(
            f"[BOOT] Fallback: Selected pair {best_pair} with {top_count} shared frames "
            f"(median disparity={_pair_disparity(best_pair):.2f}px)."
        )
        return best_pair

    if not ret:
        print(f"  [WARN] Precalibration returned False: {msg}")
        # Try to parse errors anyway
        pass

    # Extract errors
    wand_data = base_calibrator.wand_points_filtered or base_calibrator.wand_points
    all_cam_ids = sorted(list(set(cid for f in wand_data.values() for cid in f)))
    
    per_cam_error = {}
    # Try to get from internal state first
    if hasattr(base_calibrator, 'per_frame_errors') and base_calibrator.per_frame_errors:
        cam_errors_list = {cid: [] for cid in all_cam_ids}
        for fid, frame_data in base_calibrator.per_frame_errors.items():
            if 'cam_errors' in frame_data:
                for cid, err in frame_data['cam_errors'].items():
                    if cid in cam_errors_list: 
                        cam_errors_list[cid].append(err)
        for cid in all_cam_ids:
            if cam_errors_list[cid]:
                per_cam_error[cid] = np.sqrt(np.mean(np.array(cam_errors_list[cid])**2))
    
    # Fallback to parsing message
    if not per_cam_error:
        for line in msg.split('\n'):
            match = re.search(r'Cam\s*(\d+):\s*([\d.]+)\s*px', line)
            if match:
                per_cam_error[int(match.group(1))] = float(match.group(2))
    
    if not per_cam_error:
        print("  [WARN] Could not determine per-camera errors.")
        return None
        
    print("\n[BOOT] Per-camera reprojection errors (Pinhole approx):")
    for cid in sorted(per_cam_error.keys()):
        print(f"  Cam {cid}: {per_cam_error[cid]:.2f}px")
        
    sorted_cams = sorted(per_cam_error.keys(), key=lambda c: per_cam_error[c])
    if len(sorted_cams) < 2:
        return None
        
    best_pair = (min(sorted_cams[0], sorted_cams[1]), max(sorted_cams[0], sorted_cams[1]))
    print(f"[BOOT] Selected best pair: {best_pair}")
    return best_pair


@dataclass
class PinholeBootstrapP0Config:
    """Configuration for P0 bootstrap."""
    wand_length_mm: float = 47.2
    ui_focal_px: float = 9000.0  # UI-provided focal length (FROZEN)
    ftol: float = 1e-6
    xtol: float = 1e-6
    
    def __post_init__(self):
        """Validate configuration fields."""
        if self.wand_length_mm <= 0:
            raise ValueError(f"wand_length_mm must be > 0, got {self.wand_length_mm}")
        if self.ui_focal_px <= 0:
            raise ValueError(f"ui_focal_px must be > 0, got {self.ui_focal_px}")
        if self.ftol <= 0:
            raise ValueError(f"ftol must be > 0, got {self.ftol}")
        if self.xtol <= 0:
            raise ValueError(f"xtol must be > 0, got {self.xtol}")
        
        # Check for NaN/inf in all float fields
        if not np.isfinite(self.wand_length_mm):
            raise ValueError(f"wand_length_mm must be finite, got {self.wand_length_mm}")
        if not np.isfinite(self.ui_focal_px):
            raise ValueError(f"ui_focal_px must be finite, got {self.ui_focal_px}")
        if not np.isfinite(self.ftol):
            raise ValueError(f"ftol must be finite, got {self.ftol}")
        if not np.isfinite(self.xtol):
            raise ValueError(f"xtol must be finite, got {self.xtol}")


class PinholeBootstrapP0:
    """
    Stage P0: Two-camera pinhole initialization with frozen intrinsics.
    
    Uses 8-Point Algorithm (same as original pinhole Phase 1) but with frozen intrinsics.
    Optimizes only extrinsics.
    
    Pair selection is handled externally (via precalibrate).
    """
    
    def __init__(self, config: PinholeBootstrapP0Config):
        self.config = config

    @staticmethod
    def _get_camera_intrinsics(cam_id: int, camera_settings: Dict[int, dict]) -> Tuple[np.ndarray, float, float, float]:
        if cam_id not in camera_settings:
            raise ValueError(f"[P0] Missing camera_settings for cam {cam_id}")
        cfg = camera_settings[cam_id]
        f = float(cfg.get('focal', 0.0))
        w = float(cfg.get('width', 0.0))
        h = float(cfg.get('height', 0.0))
        if f <= 0 or w <= 0 or h <= 0:
            raise ValueError(
                f"[P0] Invalid camera_settings for cam {cam_id}: focal={f}, width={w}, height={h}"
            )
        cx, cy = w / 2.0, h / 2.0
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
        return K, f, cx, cy

    @staticmethod
    def _compute_phase3_wand_rms(final_res: np.ndarray, n_frames: int, n_cameras: int) -> float:
        """Compute Phase 3 wand RMS from interleaved per-frame residual layout.

        Residual blocks are laid out per frame as:
            [wand_k, reproj_k_cam0_uvA(2), reproj_k_cam0_uvB(2), ...]
        so wand residuals are every ``(1 + 2*n_cameras)`` elements.
        """
        residuals_per_frame = 1 + 2 * n_cameras
        wand_indices = np.arange(n_frames, dtype=np.int64) * residuals_per_frame
        wand_residuals = final_res[wand_indices]
        return float(np.sqrt(np.mean(wand_residuals**2)))

    def _compute_pair_metric_feasibility(
        self,
        cam_i: int,
        cam_j: int,
        observations: Dict[int, Dict[int, Tuple[np.ndarray, np.ndarray]]],
        camera_settings: Dict[int, dict],
        wand_length_mm: float,
        min_required_frames: int = 50,
    ) -> Optional[dict]:
        """Evaluate Phase-1 metric feasibility for a camera pair via E+chirality+triangulation."""
        valid_frames = self._collect_valid_frames(observations, cam_i, cam_j)
        n_valid_raw = len(valid_frames)
        if n_valid_raw < min_required_frames:
            return None

        K_i, _, _, _ = self._get_camera_intrinsics(cam_i, camera_settings)
        K_j, _, _, _ = self._get_camera_intrinsics(cam_j, camera_settings)

        pts_i = []
        pts_j = []
        for fid in valid_frames:
            uvA_i, uvB_i = observations[fid][cam_i]
            uvA_j, uvB_j = observations[fid][cam_j]
            pts_i.append(uvA_i)
            pts_i.append(uvB_i)
            pts_j.append(uvA_j)
            pts_j.append(uvB_j)

        pts_i = np.array(pts_i, dtype=np.float64)
        pts_j = np.array(pts_j, dtype=np.float64)

        pts_i_norm = cv2.undistortPoints(pts_i.reshape(-1, 1, 2), K_i, None).reshape(-1, 2)
        pts_j_norm = cv2.undistortPoints(pts_j.reshape(-1, 1, 2), K_j, None).reshape(-1, 2)

        E, mask_ess = cv2.findEssentialMat(
            pts_i_norm,
            pts_j_norm,
            focal=1.0,
            pp=(0.0, 0.0),
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1e-3,
        )
        if E is None or E.shape != (3, 3) or mask_ess is None:
            raise RuntimeError("Essential matrix estimation failed")

        mask_ess_flat = np.asarray(mask_ess).reshape(-1) > 0
        if mask_ess_flat.shape[0] != len(pts_i_norm):
            raise RuntimeError("Essential mask shape mismatch")

        inlier_idx_ess = np.where(mask_ess_flat)[0]
        if len(inlier_idx_ess) < 8:
            raise RuntimeError(f"Insufficient E inliers: {len(inlier_idx_ess)}")

        pts_i_ess = pts_i_norm[inlier_idx_ess]
        pts_j_ess = pts_j_norm[inlier_idx_ess]

        R1_cand, R2_cand, t_unit = cv2.decomposeEssentialMat(E)

        best_R_rel = None
        best_t_rel = None
        best_chirality_mask = None
        best_chirality_count = -1

        for R_cand in (R1_cand, R2_cand):
            for t_sign in (+1.0, -1.0):
                t_cand = t_sign * t_unit.reshape(3, 1)
                P_cand_i = np.hstack([np.eye(3), np.zeros((3, 1))])
                P_cand_j = np.hstack([R_cand, t_cand])

                pts_4d_cand = cv2.triangulatePoints(
                    P_cand_i, P_cand_j,
                    pts_i_ess.T, pts_j_ess.T
                )

                w_cand = pts_4d_cand[3]
                valid_w = np.abs(w_cand) > 1e-8

                z_i = np.full(pts_4d_cand.shape[1], np.nan)
                z_i[valid_w] = pts_4d_cand[2, valid_w] / w_cand[valid_w]

                pts_3d_cand = np.full((3, pts_4d_cand.shape[1]), np.nan)
                if np.any(valid_w):
                    pts_3d_cand[:, valid_w] = pts_4d_cand[:3, valid_w] / w_cand[valid_w]

                z_j = np.full(pts_4d_cand.shape[1], np.nan)
                if np.any(valid_w):
                    pts_j_space = R_cand @ pts_3d_cand[:, valid_w] + t_cand
                    z_j[valid_w] = pts_j_space[2]

                chirality_mask = (z_i > 0) & (z_j > 0) & valid_w
                chirality_count = int(np.sum(chirality_mask))

                if chirality_count > best_chirality_count:
                    best_chirality_count = chirality_count
                    best_R_rel = R_cand.copy()
                    best_t_rel = t_cand.copy()
                    best_chirality_mask = chirality_mask.copy()

        if best_R_rel is None or best_t_rel is None or best_chirality_mask is None or best_chirality_count < 8:
            raise RuntimeError(f"Chirality-only pose selection failed: best_count={best_chirality_count}")

        combined_mask = np.zeros(len(pts_i_norm), dtype=bool)
        combined_mask[inlier_idx_ess] = np.asarray(best_chirality_mask, dtype=bool)

        frame_inlier_flags = []
        for frame_idx in range(len(valid_frames)):
            corr_start = frame_idx * 2
            frame_inliers = int(np.sum(combined_mask[corr_start:corr_start + 2]))
            frame_inlier_flags.append(frame_inliers == 2)

        inlier_to_original_idx = []
        for idx, keep in enumerate(frame_inlier_flags):
            if keep:
                inlier_to_original_idx.extend([idx * 2, idx * 2 + 1])
        inlier_to_original_idx = np.array(inlier_to_original_idx, dtype=np.int64)
        if inlier_to_original_idx.size < 10:
            raise RuntimeError("Insufficient fully-inlier frames for triangulation")

        P_i = np.hstack([np.eye(3), np.zeros((3, 1))])
        P_j = np.hstack([best_R_rel, best_t_rel])
        pts_i_tri = pts_i_norm[inlier_to_original_idx]
        pts_j_tri = pts_j_norm[inlier_to_original_idx]
        pts_4d_hom = cv2.triangulatePoints(P_i, P_j, pts_i_tri.T, pts_j_tri.T)

        w_hom = pts_4d_hom[3]
        pts_3d = np.full((pts_4d_hom.shape[1], 3), np.nan, dtype=np.float64)
        valid_w = np.abs(w_hom) > 1e-8
        if np.any(valid_w):
            pts_3d[valid_w] = (pts_4d_hom[:3, valid_w] / w_hom[valid_w]).T

        wand_lengths = []
        for i in range(0, len(pts_3d), 2):
            ptA = pts_3d[i]
            ptB = pts_3d[i + 1]
            wand_lengths.append(np.linalg.norm(ptB - ptA))

        wand_lengths = np.asarray(wand_lengths, dtype=np.float64)
        valid_len_mask = np.isfinite(wand_lengths) & (wand_lengths > 1e-3) & (wand_lengths < 1e3)
        valid_lengths = wand_lengths[valid_len_mask]
        if valid_lengths.size == 0:
            raise RuntimeError("No valid triangulated wand lengths")

        min_wand = float(np.min(valid_lengths))
        sum_l = float(np.sum(valid_lengths))
        sum_l2 = float(np.sum(valid_lengths ** 2))
        s_max = float(wand_length_mm / min_wand)
        s_ls = float((wand_length_mm * sum_l / sum_l2) if sum_l2 > 1e-12 else 0.0)

        return {
            'pair': (cam_i, cam_j),
            'n_valid_frames_raw': int(n_valid_raw),
            'n_valid_frames': int(valid_lengths.size),
            'min_wand': min_wand,
            's_max': s_max,
            's_ls': s_ls,
        }

    def _select_metric_feasible_pair(
        self,
        observations: Dict[int, Dict[int, Tuple[np.ndarray, np.ndarray]]],
        camera_settings: Dict[int, dict],
        wand_length_mm: float,
        all_cam_ids: List[int],
        target_scale: float = 34.37,
    ) -> tuple:
        """Sweep all camera pairs and return the pair with highest s_max metric feasibility."""
        pair_results = []
        sorted_cam_ids = sorted(set(all_cam_ids))
        for i in range(len(sorted_cam_ids)):
            for j in range(i + 1, len(sorted_cam_ids)):
                ci, cj = sorted_cam_ids[i], sorted_cam_ids[j]
                try:
                    stats = self._compute_pair_metric_feasibility(
                        ci,
                        cj,
                        observations,
                        camera_settings,
                        wand_length_mm,
                        min_required_frames=50,
                    )
                    if stats is None:
                        n_valid_raw = len(self._collect_valid_frames(observations, ci, cj))
                        print(
                            f"[P0 PAIR SWEEP] pair ({ci},{cj}): n_valid={n_valid_raw}, "
                            f"min_wand=nan, s_max=0.00, s_ls=0.00 [SKIP <50]"
                        )
                        continue

                    pair_results.append(stats)
                    print(
                        f"[P0 PAIR SWEEP] pair ({ci},{cj}): n_valid={stats['n_valid_frames']}, "
                        f"min_wand={stats['min_wand']:.4f}, s_max={stats['s_max']:.2f}, s_ls={stats['s_ls']:.2f}"
                    )
                except Exception as e:
                    print(
                        f"[P0 PAIR SWEEP] pair ({ci},{cj}): FAILED ({e})"
                    )

        if not pair_results:
            print("[P0 PAIR SWEEP] No feasible pair found.")
            return None, None, []

        pair_results.sort(key=lambda x: x['s_max'], reverse=True)
        best = pair_results[0]
        best_pair = best['pair']
        print(
            f"[P0 PAIR SWEEP] Best pair: ({best_pair[0]},{best_pair[1]}) "
            f"with s_max={best['s_max']:.2f} (target={target_scale:.2f})"
        )
        return best_pair, best, pair_results
        
    def run(
        self,
        cam_i: int,
        cam_j: int,
        observations: Dict[int, Dict[int, Tuple[np.ndarray, np.ndarray]]],
        camera_settings: Dict[int, dict],
        progress_callback=None,
        all_cam_ids: Optional[List[int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Run P0 pinhole bootstrap for camera pair using 8-Point Algorithm.
        
        This is identical to pinhole Phase 1, but with frozen intrinsics.
        
        Args:
            cam_i, cam_j: Camera IDs (cam_i fixed at origin)
            observations: {fid: {cid: (uvA, uvB)}}
            camera_settings: per-camera intrinsics source from UI table
            progress_callback: Optional callback(phase, ray, len, cost)
            
        Returns:
            params_i: [rvec(3), tvec(3)] for cam_i (zeros)
            params_j: [rvec(3), tvec(3)] for cam_j (from 8-Point + refinement)
            report: diagnostics dict
        """
        print(f"\n{'='*60}")
        print(f"[P0] Pinhole Bootstrap - Frozen Intrinsics (8-Point)")
        print(f"{'='*60}")
        print(f"  Camera pair: ({cam_i}, {cam_j})")

        def _norm_pt(K_inv, uv):
            """Normalize a pixel coordinate using pre-inverted camera intrinsics K_inv."""
            x = K_inv @ np.array([uv[0], uv[1], 1.0], dtype=np.float64)
            return x[:2] / x[2]

        def _frame_proxy_score(obs_i, obs_j, K_inv_i_local, K_inv_j_local):
            """Per-frame proxy: min(normalized wand length in cam_i, normalized wand length in cam_j).
            Larger = more high-parallax/closer wand = better metric conditioning."""
            uvA_i, uvB_i = obs_i
            uvA_j, uvB_j = obs_j
            w_i = np.linalg.norm(_norm_pt(K_inv_i_local, uvA_i) - _norm_pt(K_inv_i_local, uvB_i))
            w_j = np.linalg.norm(_norm_pt(K_inv_j_local, uvA_j) - _norm_pt(K_inv_j_local, uvB_j))
            return min(w_i, w_j)

        def _estimate_E_8point(x1, x2):
            """Non-RANSAC 8-point essential matrix estimation (no findEssentialMat)."""
            A = np.column_stack([
                x2[:, 0] * x1[:, 0], x2[:, 0] * x1[:, 1], x2[:, 0],
                x2[:, 1] * x1[:, 0], x2[:, 1] * x1[:, 1], x2[:, 1],
                x1[:, 0], x1[:, 1], np.ones(len(x1)),
            ])
            _, _, Vt = np.linalg.svd(A, full_matrices=False)
            e_mat = Vt[-1].reshape(3, 3)
            U, _, Vt2 = np.linalg.svd(e_mat)
            e_mat = U @ np.diag([1.0, 1.0, 0.0]) @ Vt2
            return e_mat / np.linalg.norm(e_mat)

        if all_cam_ids is None:
            all_cam_ids = sorted({cid for frame_obs in observations.values() for cid in frame_obs.keys()})

        pair_proxy_stats = []
        all_frame_proxies_global = []

        for ci, cj in combinations(all_cam_ids, 2):
            K_ci, *_ = self._get_camera_intrinsics(ci, camera_settings)
            K_cj, *_ = self._get_camera_intrinsics(cj, camera_settings)
            K_inv_ci = np.linalg.inv(K_ci)  # precomputed once per pair
            K_inv_cj = np.linalg.inv(K_cj)  # precomputed once per pair
            fids = self._collect_valid_frames(observations, ci, cj)

            proxies = []
            for fid in fids:
                try:
                    p = _frame_proxy_score(observations[fid][ci], observations[fid][cj], K_inv_ci, K_inv_cj)
                    proxies.append((fid, float(p)))
                    all_frame_proxies_global.append(float(p))
                except Exception:
                    pass
            pair_proxy_stats.append({"pair": (ci, cj), "frames": proxies})

        if len(all_frame_proxies_global) == 0:
            print("[P0 ITER12] Proxy scoring failed — keeping current pair")
        else:
            global_q75 = float(np.quantile(all_frame_proxies_global, 0.75))

            for s in pair_proxy_stats:
                vals = np.array([p for _, p in s["frames"]], dtype=np.float64)
                if len(vals) == 0:
                    s["strong_count"] = 0
                    s["q90"] = -np.inf
                else:
                    s["strong_count"] = int(np.sum(vals >= global_q75))
                    s["q90"] = float(np.quantile(vals, 0.90))

            pair_proxy_stats.sort(key=lambda s: (s["strong_count"], s["q90"]), reverse=True)

            print("[P0 ITER12] Proxy ranking (stage 1):")
            for s in pair_proxy_stats:
                print(f"  pair {s['pair']}: strong_count={s['strong_count']}, q90={s['q90']:.4f}")

            shortlist = pair_proxy_stats[:3]
            if not any(s["pair"] == (cam_i, cam_j) for s in shortlist):
                current_entry = next((s for s in pair_proxy_stats if s["pair"] == (cam_i, cam_j)), None)
                if current_entry is not None:
                    shortlist = [current_entry] + shortlist[:2]

            def _sampled_pair_score(pair_entry):
                """Run a quick 8-point + triangulation check on ~128 sampled frames of a pair."""
                ci_s, cj_s = pair_entry["pair"]
                K_ci_s, *_ = self._get_camera_intrinsics(ci_s, camera_settings)
                K_cj_s, *_ = self._get_camera_intrinsics(cj_s, camera_settings)
                K_inv_ci_s = np.linalg.inv(K_ci_s)
                K_inv_cj_s = np.linalg.inv(K_cj_s)

                ranked = sorted(pair_entry["frames"], key=lambda t: t[1], reverse=True)[:256]
                if len(ranked) < 16:
                    return None

                idx = np.linspace(0, len(ranked) - 1, min(128, len(ranked)), dtype=int)
                chosen = [ranked[k][0] for k in idx]

                pts1, pts2 = [], []
                for fid in chosen:
                    try:
                        uvA_ci, uvB_ci = observations[fid][ci_s]
                        uvA_cj, uvB_cj = observations[fid][cj_s]
                        pts1.append(_norm_pt(K_inv_ci_s, uvA_ci))
                        pts1.append(_norm_pt(K_inv_ci_s, uvB_ci))
                        pts2.append(_norm_pt(K_inv_cj_s, uvA_cj))
                        pts2.append(_norm_pt(K_inv_cj_s, uvB_cj))
                    except Exception:
                        pass

                if len(pts1) < 16:
                    return None

                pts1_arr = np.asarray(pts1, np.float64)
                pts2_arr = np.asarray(pts2, np.float64)

                try:
                    E_sampled = _estimate_E_8point(pts1_arr, pts2_arr)
                    _, R_s, t_s, _ = cv2.recoverPose(E_sampled, pts1_arr, pts2_arr)
                    t_s = t_s / max(np.linalg.norm(t_s), 1e-8)
                except Exception as e:
                    print(f"  [P0 ITER12] pair {pair_entry['pair']} sampled E/pose failed: {e}")
                    return None

                P1_s = np.hstack([np.eye(3), np.zeros((3, 1))])
                P2_s = np.hstack([R_s, t_s])

                wand_lens_s = []
                for fid in chosen:
                    try:
                        uvA_ci, uvB_ci = observations[fid][ci_s]
                        uvA_cj, uvB_cj = observations[fid][cj_s]
                        x1A = _norm_pt(K_inv_ci_s, uvA_ci)
                        x2A = _norm_pt(K_inv_cj_s, uvA_cj)
                        x1B = _norm_pt(K_inv_ci_s, uvB_ci)
                        x2B = _norm_pt(K_inv_cj_s, uvB_cj)
                        XA = cv2.triangulatePoints(P1_s, P2_s, x1A.reshape(2, 1), x2A.reshape(2, 1))
                        XB = cv2.triangulatePoints(P1_s, P2_s, x1B.reshape(2, 1), x2B.reshape(2, 1))
                        wA = float(XA[3])
                        wB = float(XB[3])
                        if abs(wA) < 1e-8 or abs(wB) < 1e-8:
                            continue
                        XA3 = (XA[:3] / wA).ravel()
                        XB3 = (XB[:3] / wB).ravel()
                        if not (np.all(np.isfinite(XA3)) and np.all(np.isfinite(XB3))):
                            continue
                        wand_lens_s.append(float(np.linalg.norm(XA3 - XB3)))
                    except Exception:
                        pass

                if len(wand_lens_s) < 8:
                    return None

                wl = np.asarray(wand_lens_s, np.float64)
                return {
                    "pair": pair_entry["pair"],
                    "small_count": int(np.sum(wl < 0.5)),
                    "q10": float(np.quantile(wl, 0.10)),
                    "q25": float(np.quantile(wl, 0.25)),
                    "n_sampled": len(wl),
                }

            geom_scores = []
            for p_entry in shortlist:
                sc = _sampled_pair_score(p_entry)
                if sc is not None:
                    geom_scores.append(sc)
                    print(
                        f"  [P0 ITER12] sampled geom pair {sc['pair']}: "
                        f"small_count={sc['small_count']}, q10={sc['q10']:.4f}, n={sc['n_sampled']}"
                    )
                else:
                    print(f"  [P0 ITER12] pair {p_entry['pair']}: sampled geom check failed")

            if geom_scores:
                geom_scores.sort(key=lambda s: (s["small_count"], -s["q10"]), reverse=True)
                winner = geom_scores[0]

                current_score = next((s for s in geom_scores if s["pair"] == (cam_i, cam_j)), None)
                if current_score is None:
                    current_entry = next((p for p in pair_proxy_stats if p["pair"] == (cam_i, cam_j)), None)
                    if current_entry is not None:
                        current_score = _sampled_pair_score(current_entry)

                if current_score is None:
                    if winner["pair"] != (cam_i, cam_j):
                        print(f"[P0 ITER12] Cannot compare current pair — switching to winner {winner['pair']}")
                        cam_i, cam_j = winner["pair"]
                elif winner["pair"] != (cam_i, cam_j) and \
                        winner["small_count"] > current_score["small_count"] and \
                        winner["q10"] < current_score["q10"]:
                    print(
                        f"[P0 ITER12] Switching pair ({cam_i},{cam_j}) -> {winner['pair']} "
                        f"(small_count {current_score['small_count']} -> {winner['small_count']}, "
                        f"q10 {current_score['q10']:.4f} -> {winner['q10']:.4f})"
                    )
                    cam_i, cam_j = winner["pair"]
                else:
                    print(
                        f"[P0 ITER12] Keeping pair ({cam_i},{cam_j}): "
                        f"winner={winner['pair']} did not materially beat current "
                        f"(small_count {current_score.get('small_count', '?')} vs {winner['small_count']}, "
                        f"q10 {current_score.get('q10', float('inf')):.4f} vs {winner['q10']:.4f})"
                    )
            else:
                print("[P0 ITER12] All sampled geometry checks failed — keeping current pair")

        K_i, f_i, cx_i, cy_i = self._get_camera_intrinsics(cam_i, camera_settings)
        K_j, f_j, cx_j, cy_j = self._get_camera_intrinsics(cam_j, camera_settings)
        
        if progress_callback:
            try:
                progress_callback("P0 Pair Init", -1, 0, 0, 0)
            except Exception as e:
                logger.debug("progress_callback error (P0 Pair Init): %s", e)
        
        # Collect valid frames and points
        valid_frames = self._collect_valid_frames(observations, cam_i, cam_j)
        print(f"  Valid frames: {len(valid_frames)}")
        
        if len(valid_frames) < 10:
            raise ValueError(f"[P0] Insufficient frames: {len(valid_frames)} < 10")
        
        # Collect point correspondences for 8-Point Algorithm
        pts_i = []  # Points in cam_i
        pts_j = []  # Points in cam_j
        
        for fid in valid_frames:
            uvA_i, uvB_i = observations[fid][cam_i]
            uvA_j, uvB_j = observations[fid][cam_j]
            
            pts_i.append(uvA_i)
            pts_i.append(uvB_i)
            pts_j.append(uvA_j)
            pts_j.append(uvB_j)
        
        pts_i = np.array(pts_i, dtype=np.float64)
        pts_j = np.array(pts_j, dtype=np.float64)
        
        print(f"\n[P0] Step 1: Essential Matrix (8-Point Algorithm)...")
        if progress_callback:
            try:
                progress_callback("Use PinHole model to initialize camera parameters...", -1, 0, 0, 0)
            except Exception as e:
                logger.debug("progress_callback error (Essential Matrix): %s", e)

        # Step 1: Essential Matrix (normalized rays, supports different intrinsics)
        pts_i_norm = cv2.undistortPoints(pts_i.reshape(-1, 1, 2), K_i, None).reshape(-1, 2)
        pts_j_norm = cv2.undistortPoints(pts_j.reshape(-1, 1, 2), K_j, None).reshape(-1, 2)
        E, mask_ess = cv2.findEssentialMat(
            pts_i_norm,
            pts_j_norm,
            focal=1.0,
            pp=(0.0, 0.0),
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1e-3,
        )
        
        if E is None or E.shape != (3, 3):
            raise RuntimeError("[P0 FAIL] Essential Matrix computation failed")

        if mask_ess is None:
            raise RuntimeError("[P0 FAIL] Essential Matrix did not return an inlier mask")

        mask_ess_flat = np.asarray(mask_ess).reshape(-1) > 0
        if mask_ess_flat.shape[0] != len(pts_i_norm):
            raise RuntimeError(
                f"[P0 FAIL] Essential mask shape mismatch: {mask_ess_flat.shape[0]} vs {len(pts_i_norm)}"
            )

        inlier_idx_ess = np.where(mask_ess_flat)[0]
        if len(inlier_idx_ess) < 8:
            raise RuntimeError(
                f"[P0 FAIL] Insufficient inliers from essential matrix: {len(inlier_idx_ess)} < 8"
            )

        pts_i_ess = pts_i_norm[inlier_idx_ess]
        pts_j_ess = pts_j_norm[inlier_idx_ess]
        
        # Step 2: Recover Pose (Manual Decomposition - Chirality Only)
        # CHANGE from original: replaced cv2.recoverPose() with manual 4-candidate
        # evaluation using chirality-only masking. This avoids recoverPose()'s internal
        # depth-bounded candidate filtering, which becomes overly harsh when high-parallax
        # anchor frames are absent from the dataset.
        R1_cand, R2_cand, t_unit = cv2.decomposeEssentialMat(E)

        best_R_rel = None
        best_t_rel = None
        best_chirality_mask = None
        best_chirality_count = -1

        for R_cand in (R1_cand, R2_cand):
            for t_sign in (+1.0, -1.0):
                t_cand = t_sign * t_unit.reshape(3, 1)
                P_cand_i = np.hstack([np.eye(3), np.zeros((3, 1))])
                P_cand_j = np.hstack([R_cand, t_cand])

                pts_4d_cand = cv2.triangulatePoints(
                    P_cand_i, P_cand_j,
                    pts_i_ess.T, pts_j_ess.T
                )

                # Guard near-zero homogeneous coordinates (w ~ 0 = point at infinity)
                w_cand = pts_4d_cand[3]
                valid_w = np.abs(w_cand) > 1e-8

                # Depth in cam_i: just z = pts_4d[2] / pts_4d[3]
                z_i = np.full(pts_4d_cand.shape[1], np.nan)
                z_i[valid_w] = pts_4d_cand[2, valid_w] / w_cand[valid_w]

                # Depth in cam_j: transform to cam_j space
                pts_3d_cand = np.full((3, pts_4d_cand.shape[1]), np.nan)
                if np.any(valid_w):
                    pts_3d_cand[:, valid_w] = pts_4d_cand[:3, valid_w] / w_cand[valid_w]

                z_j = np.full(pts_4d_cand.shape[1], np.nan)
                if np.any(valid_w):
                    pts_j_space = R_cand @ pts_3d_cand[:, valid_w] + t_cand
                    z_j[valid_w] = pts_j_space[2]

                # Chirality: both z_i > 0 AND z_j > 0 (positive depth in both cameras)
                chirality_mask = (z_i > 0) & (z_j > 0) & valid_w
                chirality_count = int(np.sum(chirality_mask))

                if chirality_count > best_chirality_count:
                    best_chirality_count = chirality_count
                    best_R_rel = R_cand.copy()
                    best_t_rel = t_cand.copy()
                    best_chirality_mask = chirality_mask.copy()

        if best_R_rel is None or best_chirality_count < 8:
            raise RuntimeError(
                f"[P0 FAIL] Chirality-only pose selection failed: best_count={best_chirality_count}"
            )

        if best_chirality_mask is None:
            raise RuntimeError("[P0 FAIL] Chirality-only pose selection returned no mask")

        R_rel = best_R_rel
        t_rel = best_t_rel  # shape (3, 1)
        mask_pose_flat = np.asarray(best_chirality_mask, dtype=bool)  # len = len(inlier_idx_ess)

        print(f"  Chirality-only pose: best candidate has {best_chirality_count}/{len(pts_i_ess)} inliers")

        if mask_pose_flat.shape[0] != len(inlier_idx_ess):
            raise RuntimeError(
                f"[P0 FAIL] Chirality mask shape mismatch: {mask_pose_flat.shape[0]} vs {len(inlier_idx_ess)}"
            )

        # Combined inlier mask in original correspondence indexing.
        combined_mask = np.zeros(len(pts_i_norm), dtype=bool)
        combined_mask[inlier_idx_ess] = mask_pose_flat
        n_inliers = int(np.sum(combined_mask))
        n_inliers_pose = best_chirality_count  # for report compatibility

        if n_inliers < 8:
            raise RuntimeError(
                f"[P0 FAIL] Insufficient inliers after chirality selection: {n_inliers} < 8"
            )
        
        print(f"  RANSAC inliers: {n_inliers} / {len(pts_i)} ({100.0 * n_inliers / max(1, len(pts_i)):.1f}%)")

        # Build per-frame inlier summary (2 correspondences per frame: A and B endpoint).
        frame_inlier_flags = []
        for frame_idx, fid in enumerate(valid_frames):
            corr_start = frame_idx * 2
            frame_inliers = int(np.sum(combined_mask[corr_start:corr_start + 2]))
            frame_inlier_flags.append(frame_inliers == 2)
            print(f"    Frame {fid}: {frame_inliers}/2 inliers ({50.0 * frame_inliers:.1f}%)")

        valid_frames_inlier = [fid for idx, fid in enumerate(valid_frames) if frame_inlier_flags[idx]]
        if len(valid_frames_inlier) < 5:
            raise RuntimeError(
                f"[P0 FAIL] Insufficient inlier frames after RANSAC filtering: {len(valid_frames_inlier)} < 5"
            )

        inlier_to_original_idx = []
        for idx, keep in enumerate(frame_inlier_flags):
            if keep:
                inlier_to_original_idx.extend([idx * 2, idx * 2 + 1])
        inlier_to_original_idx = np.array(inlier_to_original_idx, dtype=np.int64)
        
        # Step 3: Triangulate to compute scale
        print(f"\n[P0] Step 2: Triangulation & Scale Recovery...")
        if progress_callback:
            try:
                progress_callback("Use PinHole model to initialize camera parameters...", -1, 0, 0, 0)
            except Exception as e:
                logger.debug("progress_callback error (Triangulation): %s", e)
        
        # Projection matrices (cam_i at origin)
        P_i = np.hstack([np.eye(3), np.zeros((3, 1))])
        P_j = np.hstack([R_rel, t_rel])
        
        # Triangulate inlier-only correspondences
        pts_i_tri = pts_i_norm[inlier_to_original_idx]
        pts_j_tri = pts_j_norm[inlier_to_original_idx]
        pts_4d_hom = cv2.triangulatePoints(P_i, P_j, pts_i_tri.T, pts_j_tri.T)
        # Guard homogeneous division: near-zero |w| indicates point at infinity/unstable geometry.
        # NOTE: w can legitimately be negative (DLT sign convention) — guard on |w|, not w.
        w_hom = pts_4d_hom[3]
        pts_3d = np.full((pts_4d_hom.shape[1], 3), np.nan, dtype=np.float64)
        valid_w = np.abs(w_hom) > 1e-8
        if np.any(valid_w):
            pts_3d[valid_w] = (pts_4d_hom[:3, valid_w] / w_hom[valid_w]).T  # (N, 3)
        
        # Compute wand lengths
        wand_lengths = []
        for i in range(0, len(pts_3d), 2):
            ptA = pts_3d[i]
            ptB = pts_3d[i + 1]
            wand_lengths.append(np.linalg.norm(ptB - ptA))
        
        wand_lengths = np.asarray(wand_lengths, dtype=np.float64)
        valid_len_mask = np.isfinite(wand_lengths) & (wand_lengths > 1e-3) & (wand_lengths < 1e3)

        if np.sum(valid_len_mask) < 5:
            raise RuntimeError("[P0 FAIL] Triangulation failed to produce valid structure")

        # ── Iter 10 fix: metric-anchor subset for Phase 1 BA ─────────────────────────
        # The Phase 1 BA objective is dominated by low-parallax frames (large wand_len)
        # which pull scale to the wrong basin. Use only the smallest-wand-length frames
        # (highest parallax, most metric-informative) for Phase 1 BA.
        # Phase 2/3 still use the full valid_frames_inlier set.
        valid_lengths_all = wand_lengths[valid_len_mask]
        valid_frames_inlier_all = list(valid_frames_inlier)  # preserve for Phase 2/3

        # Build anchor subset: lowest 5% of valid wand lengths, floor 32, cap 150
        n_valid = int(np.sum(valid_len_mask))
        n_anchor_target = max(32, min(150, int(np.ceil(n_valid * 0.05))))

        # Map from valid_len_mask index → frame index in valid_frames_inlier
        valid_frame_indices = np.where(valid_len_mask)[0]  # indices into valid_frames_inlier

        # Sort by wand length ascending → most high-parallax first
        sort_order = np.argsort(valid_lengths_all)
        anchor_frame_indices = valid_frame_indices[sort_order[:n_anchor_target]]
        anchor_frame_indices_sorted = np.sort(anchor_frame_indices)

        # Build anchor subset frame list and matching pts_3d
        valid_frames_metric = [valid_frames_inlier[idx] for idx in anchor_frame_indices_sorted]
        anchor_len_vals = valid_lengths_all[sort_order[:n_anchor_target]]

        print(
            f"[P0 ITER10] Metric-anchor subset: {len(valid_frames_metric)}/{len(valid_frames_inlier)} frames"
            f" | wand_len range [{anchor_len_vals.min():.4f}, {anchor_len_vals.max():.4f}]"
            f" (full: min={valid_lengths_all.min():.4f}, median={np.median(valid_lengths_all):.4f})"
        )

        # Fallback: if anchor subset is too small or its wand lengths are all bad, keep all frames
        if len(valid_frames_metric) < 20:
            print("[P0 ITER10] Anchor subset too small — falling back to all inlier frames for BA")
            valid_frames_metric = list(valid_frames_inlier)
            anchor_frame_indices_sorted = np.arange(len(valid_frames_inlier))
            anchor_len_vals = valid_lengths_all

        # Rebuild pts_3d for the anchor subset only
        # Each frame contributes 2 points (ptA, ptB); pts_3d is indexed as frame_idx*2, frame_idx*2+1
        pts_3d_metric = np.vstack([pts_3d[idx * 2: idx * 2 + 2] for idx in anchor_frame_indices_sorted])
        # ── End Iter 10 anchor subset ────────────────────────────────────────────────

        # ── Iter 8 fix: multi-start Phase 1 BA over geometric scale seeds ──────────
        # Pre-BA scale statistics are unreliable when dataset lacks high-parallax
        # anchor frames (all wand_len >> 0.29 in normalized coords).
        # Solution: try several scale seeds and let wand-length + reproj BA pick the best.
        valid_lengths = anchor_len_vals
        p5_implied = self.config.wand_length_mm / np.percentile(valid_lengths, 5.0)
        seed_scales = np.geomspace(2.0, 80.0, 7).tolist()
        if np.isfinite(p5_implied) and p5_implied > 0:
            seed_scales.append(float(p5_implied))

        # Convert R to rvec (shared across all seeds)
        rvec_j, _ = cv2.Rodrigues(R_rel)

        print(f"\n[P0] Step 3: Extrinsic Refinement (frozen intrinsics)...")
        if progress_callback:
            try:
                progress_callback("Use PinHole model to initialize camera parameters...", -1, 0, 0, 0)
            except Exception as e:
                logger.debug("progress_callback error (Extrinsic Refinement): %s", e)
        
        # Shared point count for state vector [cam_j(6), pts_3d(N*3)]
        # Use ALL triangulated points for BA (anchor subset used only for scale seed)
        n_pts = len(pts_3d)
        
        from scipy.sparse import lil_matrix
        
        # Residuals: for each frame: wand(1) + reproj(8) = 9
        n_frames = len(valid_frames_inlier)
        n_res = n_frames * 9
        n_cams = 1
        n_cam_params = 6
        pt_start = n_cams * n_cam_params  # = 6
        n_params = pt_start + n_pts * 3
        
        A_sparsity = lil_matrix((n_res, n_params), dtype=int)
        
        for i, fid in enumerate(valid_frames_inlier):
            idx_ptA = pt_start + i * 6
            idx_ptB = pt_start + i * 6 + 3
            base_res = i * 9
            
            # Wand length
            A_sparsity[base_res, idx_ptA:idx_ptA+3] = 1
            A_sparsity[base_res, idx_ptB:idx_ptB+3] = 1
            
            # Reproj cam_i ptA/ptB — cam_i is frozen (no state cols)
            A_sparsity[base_res+1:base_res+3, idx_ptA:idx_ptA+3] = 1
            A_sparsity[base_res+3:base_res+5, idx_ptB:idx_ptB+3] = 1
            
            # Reproj cam_j ptA/ptB
            A_sparsity[base_res+5:base_res+7, 0:6] = 1
            A_sparsity[base_res+5:base_res+7, idx_ptA:idx_ptA+3] = 1
            A_sparsity[base_res+7:base_res+9, 0:6] = 1
            A_sparsity[base_res+7:base_res+9, idx_ptB:idx_ptB+3] = 1
        
        # Residuals function (frozen intrinsics)
        self._res_call_count = 0 
        def residuals_func(x):
            # cam_i is frozen at origin
            p_i = np.zeros(6)
            p_j = x[:6]
            pts = x[6:].reshape(-1, 3)
            
            R_i, _ = cv2.Rodrigues(p_i[:3])
            t_i = p_i[3:6].reshape(3, 1)
            R_j, _ = cv2.Rodrigues(p_j[:3])
            t_j = p_j[3:6].reshape(3, 1)
            
            res = []
            sq_err_len = 0.0
            n_len = 0
            sq_err_proj = 0.0
            n_proj = 0
            for idx, fid in enumerate(valid_frames_inlier):
                uvA_i, uvB_i = observations[fid][cam_i]
                uvA_j, uvB_j = observations[fid][cam_j]
                
                ptA = pts[idx * 2]
                ptB = pts[idx * 2 + 1]
                
                # Wand length
                wand_len = np.linalg.norm(ptB - ptA)
                d_len = wand_len - self.config.wand_length_mm
                res.append(d_len)
                sq_err_len += d_len * d_len
                n_len += 1
                
                # Reprojections with frozen K (cheirality: NaN → zero residual)
                proj_Ai = self._project(ptA, R_i, t_i, K_i)
                proj_Bi = self._project(ptB, R_i, t_i, K_i)
                diff_Ai = np.zeros(2) if np.isnan(proj_Ai[0]) else (proj_Ai - uvA_i)
                diff_Bi = np.zeros(2) if np.isnan(proj_Bi[0]) else (proj_Bi - uvB_i)
                res.extend(diff_Ai.tolist())
                res.extend(diff_Bi.tolist())
                sq_err_proj += float(np.sum(diff_Ai**2) + np.sum(diff_Bi**2))
                n_proj += 4

                proj_Aj = self._project(ptA, R_j, t_j, K_j)
                proj_Bj = self._project(ptB, R_j, t_j, K_j)
                diff_Aj = np.zeros(2) if np.isnan(proj_Aj[0]) else (proj_Aj - uvA_j)
                diff_Bj = np.zeros(2) if np.isnan(proj_Bj[0]) else (proj_Bj - uvB_j)
                res.extend(diff_Aj.tolist())
                res.extend(diff_Bj.tolist())
                sq_err_proj += float(np.sum(diff_Aj**2) + np.sum(diff_Bj**2))
                n_proj += 4

            # To numpy
            res_arr = np.array(res)

            # Report progress with metrics
            self._res_call_count += 1
            if progress_callback and self._res_call_count % 5 == 0:
                try:
                    rmse_len = np.sqrt(sq_err_len / max(1, n_len))
                    rmse_proj = np.sqrt(sq_err_proj / max(1, n_proj))
                    rmse_ray = -1.0
                    cost = 0.5 * np.sum(res_arr**2)

                    progress_callback(
                        "Use PinHole model to initialize camera parameters...",
                        rmse_ray,
                        rmse_len,
                        rmse_proj,
                        cost,
                    )
                except Exception as e:
                    logger.debug("progress_callback error (Phase 1 BA iter): %s", e)
            
            return res_arr

        # ── inner helper: build x0 and run BA for a given scale0, return (result, x0) ──
        def _run_ba_for_scale(scale0):
            t_scaled = t_rel * scale0
            pts_3d_scaled = pts_3d * scale0
            params_j = np.concatenate([rvec_j.flatten(), t_scaled.flatten()])
            x0 = np.concatenate([params_j, pts_3d_scaled.flatten()])

            result_local = least_squares(
                residuals_func, x0,
                jac_sparsity=A_sparsity,
                method='trf',
                x_scale='jac',
                f_scale=1.0,
                verbose=0,
                ftol=self.config.ftol,
                xtol=self.config.xtol,
                max_nfev=100,
            )
            return result_local, x0

        best_result = None
        best_cost = np.inf
        best_scale = None
        best_quality = -np.inf
        n_frames_ba = len(valid_frames_inlier)  # number of frames in BA (2 pts each)

        print(f"\n[P0] Step 3: Extrinsic Refinement — multi-start BA ({len(seed_scales)} seeds)...")
        for scale0 in seed_scales:
            try:
                r, x0_s = _run_ba_for_scale(scale0)
                # Extract optimized 3D points and compute wand lengths
                pts_opt = r.x[6:].reshape(-1, 3)
                wand_lengths_opt = np.array([
                    np.linalg.norm(pts_opt[i * 2 + 1] - pts_opt[i * 2])
                    for i in range(n_frames_ba)
                ])
                wand_median = np.median(wand_lengths_opt)
                error_ratio = wand_median / self.config.wand_length_mm
                # Quality: penalise wand-length deviation from reference, with small cost tiebreaker
                seed_quality = -(error_ratio - 1.0) ** 2 - 0.01 * r.cost
                print(f"  seed={scale0:.2f}: cost={r.cost:.3e}, wand_ratio={error_ratio:.3f}, quality={seed_quality:.3e}")
                if seed_quality > best_quality:
                    best_quality = seed_quality
                    best_cost = r.cost
                    best_result = r
                    best_scale = scale0
            except Exception as e:
                print(f"  seed={scale0:.2f}: FAILED ({e})")

        if best_result is None:
            raise RuntimeError("[P0 FAIL] All scale seeds failed in multi-start BA")

        result = best_result
        print(f"  Best seed: scale0={best_scale:.2f}, cost={result.cost:.3e}, quality={best_quality:.3e}")

        params_i_opt = np.zeros(6)  # frozen
        params_j_opt = result.x[:6]
        pts_3d_opt = result.x[6:].reshape(-1, 3)

        # Derive scale_factor from optimized tvec for reporting only
        t_opt_norm = np.linalg.norm(params_j_opt[3:6])
        t_dir_norm = np.linalg.norm(t_rel)
        scale_factor = t_opt_norm / max(t_dir_norm, 1e-8)
        
        print(f"  BA cost: {result.cost:.2e}")
        print(f"  cam_i: frozen at origin [0,0,0,0,0,0]")
        
        # Compute diagnostics
        report = self._compute_diagnostics(
            cam_i, cam_j, params_i_opt, params_j_opt,
            observations, valid_frames_inlier, K_i, K_j
        )
        report['scale_factor'] = scale_factor
        report['n_inliers'] = n_inliers
        report['n_inliers_pose'] = int(n_inliers_pose)
        report['n_inlier_frames'] = len(valid_frames_inlier)
        report['phase1_inlier_frames'] = valid_frames_inlier_all
        report['phase1_pair'] = (cam_i, cam_j)
        
        # Sanity checks
        self._validate(report)
        
        print(f"\n[P0] Phase 1 Complete:")
        print(f"  cam_{cam_i}: frozen at origin [0,0,0,0,0,0]")
        print(f"  cam_{cam_j} rvec: [{params_j_opt[0]:.4f}, {params_j_opt[1]:.4f}, {params_j_opt[2]:.4f}]")
        print(f"  cam_{cam_j} tvec: [{params_j_opt[3]:.2f}, {params_j_opt[4]:.2f}, {params_j_opt[5]:.2f}]")
        print(f"  Baseline: {report['baseline_mm']:.2f} mm")
        print(f"  Wand length: {report['wand_length_median']:.4f} mm")
        
        return params_i_opt, params_j_opt, report
    

    
    def _project(self, pt3d: np.ndarray, R: np.ndarray, t: np.ndarray, K: np.ndarray) -> np.ndarray:
        """Project 3D point to 2D using pinhole model."""
        pt_cam = R @ pt3d.reshape(3, 1) + t
        pt_cam = pt_cam.flatten()
        if pt_cam[2] <= 0:
            return np.array([np.nan, np.nan])  # Cheirality: behind camera
        pt_norm = pt_cam[:2] / pt_cam[2]
        pt_px = K[:2, :2] @ pt_norm + K[:2, 2]
        return pt_px


    def _collect_valid_frames(
        self,
        observations: Dict[int, Dict[int, Tuple[np.ndarray, np.ndarray]]],
        cam_i: int,
        cam_j: int
    ) -> List[int]:
        """Collect frames where both cameras see both A and B."""
        valid = []
        for fid, cam_obs in observations.items():
            if cam_i in cam_obs and cam_j in cam_obs:
                uvA_i, uvB_i = cam_obs[cam_i]
                uvA_j, uvB_j = cam_obs[cam_j]
                if all(x is not None for x in [uvA_i, uvB_i, uvA_j, uvB_j]):
                    valid.append(fid)
        return valid
    
    def _compute_diagnostics(
        self,
        cam_i: int,
        cam_j: int,
        params_i: np.ndarray,
        params_j: np.ndarray,
        observations: Dict[int, Dict[int, Tuple[np.ndarray, np.ndarray]]],
        valid_frames: List[int],
        K_i: np.ndarray,
        K_j: np.ndarray,
    ) -> dict:
        """Compute diagnostics after optimization."""
        # Build projection matrices
        R_i, _ = cv2.Rodrigues(params_i[:3])
        t_i = params_i[3:6].reshape(3, 1)
        R_j, _ = cv2.Rodrigues(params_j[:3])
        t_j = params_j[3:6].reshape(3, 1)
        
        P_i = K_i @ np.hstack([R_i, t_i])
        P_j = K_j @ np.hstack([R_j, t_j])
        
        wand_lengths = []
        reproj_errors = []
        
        DIAG_FRAME_CAP = 200
        frames_to_process = valid_frames[:DIAG_FRAME_CAP]
        if len(valid_frames) >= DIAG_FRAME_CAP:
            print(
                f"  [WARN] Diagnostics frame count hit {DIAG_FRAME_CAP}-frame cap "
                f"({len(valid_frames)} total frames truncated to {DIAG_FRAME_CAP})"
            )
        
        for fid in frames_to_process:
            uvA_i, uvB_i = observations[fid][cam_i]
            uvA_j, uvB_j = observations[fid][cam_j]
            
            # Triangulate
            pts_4d_A = cv2.triangulatePoints(P_i, P_j, 
                                             uvA_i.reshape(2, 1), uvA_j.reshape(2, 1))
            pts_4d_B = cv2.triangulatePoints(P_i, P_j, 
                                             uvB_i.reshape(2, 1), uvB_j.reshape(2, 1))
            
            # Guard homogeneous division: near-zero |w| indicates point at infinity/unstable geometry.
            # NOTE: w can legitimately be negative (DLT sign convention) — guard on |w|, not w.
            w_A = float(pts_4d_A[3, 0])
            w_B = float(pts_4d_B[3, 0])
            ptA = (pts_4d_A[:3] / w_A).flatten() if abs(w_A) > 1e-8 else np.full(3, np.nan, dtype=np.float64)
            ptB = (pts_4d_B[:3] / w_B).flatten() if abs(w_B) > 1e-8 else np.full(3, np.nan, dtype=np.float64)
            
            wand_lengths.append(np.linalg.norm(ptB - ptA))
            
            # Reprojection error for BOTH endpoints (skip behind-camera NaN projections)
            proj_Ai = self._project(ptA, R_i, t_i, K_i)
            proj_Aj = self._project(ptA, R_j, t_j, K_j)
            proj_Bi = self._project(ptB, R_i, t_i, K_i)
            proj_Bj = self._project(ptB, R_j, t_j, K_j)
            
            if not np.isnan(proj_Ai[0]):
                reproj_errors.append(np.linalg.norm(proj_Ai - uvA_i))
            if not np.isnan(proj_Aj[0]):
                reproj_errors.append(np.linalg.norm(proj_Aj - uvA_j))
            if not np.isnan(proj_Bi[0]):
                reproj_errors.append(np.linalg.norm(proj_Bi - uvB_i))
            if not np.isnan(proj_Bj[0]):
                reproj_errors.append(np.linalg.norm(proj_Bj - uvB_j))
        
        return {
            'baseline_mm': np.linalg.norm(params_j[3:6]),
            'wand_length_median': np.median(wand_lengths) if wand_lengths else 0,
            'wand_length_std': np.std(wand_lengths) if wand_lengths else 0,
            'wand_length_error': abs(np.median(wand_lengths) - self.config.wand_length_mm) if wand_lengths else float('inf'),
            'reproj_err_mean': np.mean(reproj_errors) if reproj_errors else 0,
            'reproj_err_max': np.max(reproj_errors) if reproj_errors else 0,
            'reproj_n_samples': len(reproj_errors),
            'valid_frames': len(valid_frames),
        }
    
    def _validate(self, report: dict):
        """Validate P0 results. FAIL if constraints violated."""
        print(f"\n{'-'*60}")
        print("[P0 VALIDATION]")
        print(f"{'-'*60}")
        
        b = report['baseline_mm']
        print(f"  Baseline: {b:.2f} mm (recommended min: 50 mm)")
        if b < 50.0:
            print(f"  [WARN] Baseline is below recommended minimum: {b:.2f} mm < 50 mm")
        
        reproj = report.get('reproj_err_mean', 0)
        print(f"  Reproj error mean: {reproj:.2f} px")
        if reproj > 50.0:
            raise RuntimeError(f"[P0 FAIL] Reprojection error too high: {reproj:.2f} px")
        
        wand_err = report.get('wand_length_error', float('inf'))
        print(f"  Wand length error: {wand_err:.4f} mm")
        
        print("[P0 VALIDATION] PASSED")
        print(f"{'-'*60}")
    
    # =========================================================================
    # Phase 2: Calibrate remaining cameras using 3D points from Phase 1
    # =========================================================================
    
    def run_phase2(
        self,
        cam_params: Dict[int, np.ndarray],
        observations: Dict[int, Dict[int, Tuple[np.ndarray, np.ndarray]]],
        points_3d: Dict[int, Tuple[np.ndarray, np.ndarray]],
        camera_settings: Dict[int, dict],
        all_cam_ids: List[int],
    ) -> Dict[int, np.ndarray]:
        """
        Phase 2: Calibrate remaining cameras using 3D points from Phase 1.
        
        For each camera not in cam_params:
        - Collect 2D-3D correspondences from points_3d
        - Solve PnP with frozen intrinsics
        """
        dist_coeffs = np.zeros(5)
        
        calibrated_cams = set(cam_params.keys())
        remaining_cams = [c for c in all_cam_ids if c not in calibrated_cams]
        
        if not remaining_cams:
            print("[P0 Phase 2] No remaining cameras to calibrate.")
            return cam_params
        
        print(f"\n{'='*60}")
        print(f"[P0 Phase 2] Calibrating {len(remaining_cams)} remaining cameras")
        print(f"{'='*60}")
        print(f"  Already calibrated: {sorted(calibrated_cams)}")
        print(f"  To calibrate: {remaining_cams}")
        
        for cid in remaining_cams:
            print(f"\n  --- Calibrating cam_{cid} ---")
            
            # Collect 2D-3D correspondences
            pts_2d = []
            pts_3d_list = []
            
            for fid, (XA, XB) in points_3d.items():
                if fid not in observations:
                    continue
                if cid not in observations[fid]:
                    continue
                    
                uvA, uvB = observations[fid][cid]
                if uvA is not None:
                    pts_2d.append(uvA)
                    pts_3d_list.append(XA)
                if uvB is not None:
                    pts_2d.append(uvB)
                    pts_3d_list.append(XB)
            
            if len(pts_2d) < 6:
                print(f"    [WARN] Insufficient correspondences: {len(pts_2d)} < 6. Skipping.")
                continue
            
            pts_2d = np.array(pts_2d, dtype=np.float64)
            pts_3d_arr = np.array(pts_3d_list, dtype=np.float64)
            
            print(f"    Correspondences: {len(pts_2d)}")
            
            K, _, _, _ = self._get_camera_intrinsics(cid, camera_settings)
            # Solve PnP with frozen intrinsics (EPNP + ITERATIVE, like original)
            success, rvec, tvec = cv2.solvePnP(
                pts_3d_arr, pts_2d, K, dist_coeffs,
                flags=cv2.SOLVEPNP_EPNP
            )
            
            if not success:
                print(f"    [WARN] PnP (EPNP) failed for cam_{cid}. Skipping.")
                continue
            
            # Refine with ITERATIVE
            success, rvec, tvec = cv2.solvePnP(
                pts_3d_arr, pts_2d, K, dist_coeffs,
                rvec, tvec, useExtrinsicGuess=True,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            
            rvec = rvec.flatten()
            tvec = tvec.flatten()
            
            # Compute initial reprojection error
            pts_reproj, _ = cv2.projectPoints(pts_3d_arr, rvec, tvec, K, dist_coeffs)
            pts_reproj = pts_reproj.reshape(-1, 2)
            reproj_err_init = np.sqrt(np.mean(np.sum((pts_2d - pts_reproj)**2, axis=1)))
            
            print(f"    PnP result: RMS = {reproj_err_init:.2f}px")
            
            # Per-camera extrinsic-only optimization (like original Phase 2, but frozen f)
            print(f"    Optimizing extrinsics (frozen intrinsics)...")
            
            x0_cam = np.concatenate([rvec, tvec])  # [rvec(3), tvec(3)]
            
            def residuals_cam(x):
                r = x[:3].reshape(3, 1)
                t = x[3:6].reshape(3, 1)
                pts_proj, _ = cv2.projectPoints(pts_3d_arr, r, t, K, dist_coeffs)
                pts_proj = pts_proj.reshape(-1, 2)
                return (pts_2d - pts_proj).flatten()
            
            result = least_squares(
                residuals_cam, x0_cam,
                method='lm',
                ftol=self.config.ftol,
                xtol=self.config.xtol,
                max_nfev=100,
            )
            
            rvec_opt = result.x[:3]
            tvec_opt = result.x[3:6]
            
            # Compute final reprojection error
            pts_reproj_opt, _ = cv2.projectPoints(pts_3d_arr, rvec_opt, tvec_opt, K, dist_coeffs)
            pts_reproj_opt = pts_reproj_opt.reshape(-1, 2)
            
            # Compute per-point residuals
            residuals = np.sqrt(np.sum((pts_2d - pts_reproj_opt)**2, axis=1))
            median_residual = np.median(residuals)
            threshold = 3.0 * median_residual
            
            # Filter outliers
            inlier_mask = residuals <= threshold
            n_outliers = np.sum(~inlier_mask)
            
            if n_outliers > 0:
                print(f"    Filtering {n_outliers}/{len(pts_2d)} outliers (threshold={threshold:.2f}px)")
                
                # Re-optimize with inliers only
                pts_2d_inliers = pts_2d[inlier_mask]
                pts_3d_inliers = pts_3d_arr[inlier_mask]
                
                if len(pts_2d_inliers) >= 6:
                    x0_cam_refined = np.concatenate([rvec_opt, tvec_opt])
                    
                    def residuals_cam_refined(x):
                        r = x[:3].reshape(3, 1)
                        t = x[3:6].reshape(3, 1)
                        pts_proj, _ = cv2.projectPoints(pts_3d_inliers, r, t, K, dist_coeffs)
                        pts_proj = pts_proj.reshape(-1, 2)
                        return (pts_2d_inliers - pts_proj).flatten()
                    
                    result_refined = least_squares(
                        residuals_cam_refined, x0_cam_refined,
                        method='lm',
                        ftol=self.config.ftol,
                        xtol=self.config.xtol,
                        max_nfev=100,
                    )
                    
                    rvec_opt = result_refined.x[:3]
                    tvec_opt = result_refined.x[3:6]
                    
                    # Compute final error on inliers
                    pts_reproj_final, _ = cv2.projectPoints(pts_3d_inliers, rvec_opt, tvec_opt, K, dist_coeffs)
                    pts_reproj_final = pts_reproj_final.reshape(-1, 2)
                    reproj_err_final = np.sqrt(np.mean(np.sum((pts_2d_inliers - pts_reproj_final)**2, axis=1)))
                else:
                    print(f"    [WARN] Too few inliers after filtering ({len(pts_2d_inliers)} < 6), using original solution")
                    reproj_err_final = np.sqrt(np.mean(residuals[inlier_mask]**2)) if np.any(inlier_mask) else np.sqrt(np.mean(residuals**2))
            else:
                reproj_err_final = np.sqrt(np.mean(residuals**2))
            
            print(f"    rvec: [{rvec_opt[0]:.4f}, {rvec_opt[1]:.4f}, {rvec_opt[2]:.4f}]")
            print(f"    tvec: [{tvec_opt[0]:.2f}, {tvec_opt[1]:.2f}, {tvec_opt[2]:.2f}]")
            print(f"    Reproj RMS: {reproj_err_init:.2f} -> {reproj_err_final:.2f}px")
            
            cam_params[cid] = np.concatenate([rvec_opt, tvec_opt])
        
        print(f"\n[P0 Phase 2] Calibrated {len(cam_params)} cameras total.")
        return cam_params
    
    def triangulate_all_points(
        self,
        cam_i: int,
        cam_j: int,
        params_i: np.ndarray,
        params_j: np.ndarray,
        observations: Dict[int, Dict[int, Tuple[np.ndarray, np.ndarray]]],
        camera_settings: Dict[int, dict],
    ) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """Triangulate all 3D wand points using Phase 1 cameras."""
        K_i, _, _, _ = self._get_camera_intrinsics(cam_i, camera_settings)
        K_j, _, _, _ = self._get_camera_intrinsics(cam_j, camera_settings)
        
        R_i, _ = cv2.Rodrigues(params_i[:3])
        t_i = params_i[3:6].reshape(3, 1)
        R_j, _ = cv2.Rodrigues(params_j[:3])
        t_j = params_j[3:6].reshape(3, 1)
        
        P_i = K_i @ np.hstack([R_i, t_i])
        P_j = K_j @ np.hstack([R_j, t_j])
        
        points_3d = {}
        valid_frames = self._collect_valid_frames(observations, cam_i, cam_j)
        
        for fid in valid_frames:
            uvA_i, uvB_i = observations[fid][cam_i]
            uvA_j, uvB_j = observations[fid][cam_j]
            
            pts_4d_A = cv2.triangulatePoints(P_i, P_j, 
                                             uvA_i.reshape(2, 1), uvA_j.reshape(2, 1))
            pts_4d_B = cv2.triangulatePoints(P_i, P_j, 
                                             uvB_i.reshape(2, 1), uvB_j.reshape(2, 1))
            
            # Guard homogeneous division: near-zero |w| indicates point at infinity/unstable geometry.
            # NOTE: w can legitimately be negative (DLT sign convention) — guard on |w|, not w.
            w_A = float(pts_4d_A[3, 0])
            w_B = float(pts_4d_B[3, 0])
            XA = (pts_4d_A[:3] / w_A).flatten() if abs(w_A) > 1e-8 else np.full(3, np.nan, dtype=np.float64)
            XB = (pts_4d_B[:3] / w_B).flatten() if abs(w_B) > 1e-8 else np.full(3, np.nan, dtype=np.float64)
            
            points_3d[fid] = (XA, XB)
        
        return points_3d
    
    def run_phase3(
        self,
        cam_params: Dict[int, np.ndarray],
        observations: Dict[int, Dict[int, Tuple[np.ndarray, np.ndarray]]],
        camera_settings: Dict[int, dict],
        cam_i: int,
        cam_j: int,
        progress_callback=None
    ) -> Dict[int, np.ndarray]:
        """
        Phase 3: Global BA with all cameras and frozen intrinsics.
        
        Joint optimization of all camera extrinsics + 3D points.
        """
        K_by_cam = {}
        for cid in cam_params.keys():
            K_by_cam[cid], _, _, _ = self._get_camera_intrinsics(cid, camera_settings)
        
        all_cam_ids = sorted(cam_params.keys())
        n_cams = len(all_cam_ids)
        fixed_cam_ids = (int(cam_i), int(cam_j))
        free_cam_ids = [cid for cid in all_cam_ids if cid not in fixed_cam_ids]
        n_free_cams = len(free_cam_ids)
        free_cam_id_to_idx = {cid: i for i, cid in enumerate(free_cam_ids)}

        def _camera_center_from_params(p: np.ndarray) -> np.ndarray:
            R, _ = cv2.Rodrigues(p[:3])
            t = p[3:6].reshape(3, 1)
            return (-R.T @ t).reshape(3)

        baseline_phase1 = float(
            np.linalg.norm(
                _camera_center_from_params(cam_params[cam_j])
                - _camera_center_from_params(cam_params[cam_i])
            )
        )
        
        print(f"\n{'='*60}")
        print(f"[P0 Phase 3] Global BA with frozen intrinsics")
        print(f"{'='*60}")
        print(f"  Cameras: {all_cam_ids}")
        print(f"  Fixed seed cameras: cam_{cam_i}, cam_{cam_j}")
        print(f"  Free cameras: {free_cam_ids}")
        print(f"  baseline_phase1 (cam_{cam_i}-cam_{cam_j}): {baseline_phase1:.3f} mm")
        print("  Frozen intrinsics: per-camera table values")
        
        # Collect valid frames (seen by at least 2 calibrated cameras)
        valid_frames = []
        for fid, cams in observations.items():
            calibrated_in_frame = [c for c in cams.keys() if c in cam_params]
            if len(calibrated_in_frame) >= 2:
                valid_frames.append(fid)
        
        print(f"  Valid frames: {len(valid_frames)}")
        
        if len(valid_frames) < 10:
            print("  [WARN] Not enough frames for Phase 3, skipping.")
            return cam_params
        
        # Triangulate initial 3D points using first available pair per frame
        print("  Triangulating initial points for global BA...")
        pts_3d_init = []
        frame_cams = []  # [(fid, [cams that see this frame])]
        
        for fid in valid_frames:
            cams_in_frame = [c for c in observations[fid].keys() if c in cam_params]
            if len(cams_in_frame) < 2:
                continue
                
            # Use first two cameras for triangulation
            c1, c2 = cams_in_frame[0], cams_in_frame[1]
            p1, p2 = cam_params[c1], cam_params[c2]
            
            R1, _ = cv2.Rodrigues(p1[:3])
            t1 = p1[3:6].reshape(3, 1)
            R2, _ = cv2.Rodrigues(p2[:3])
            t2 = p2[3:6].reshape(3, 1)
            
            P1 = K_by_cam[c1] @ np.hstack([R1, t1])
            P2 = K_by_cam[c2] @ np.hstack([R2, t2])
            
            uvA_1, uvB_1 = observations[fid][c1]
            uvA_2, uvB_2 = observations[fid][c2]
            
            pts_4d_A = cv2.triangulatePoints(P1, P2, uvA_1.reshape(2, 1), uvA_2.reshape(2, 1))
            pts_4d_B = cv2.triangulatePoints(P1, P2, uvB_1.reshape(2, 1), uvB_2.reshape(2, 1))
            
            # Guard homogeneous division: near-zero |w| indicates point at infinity/unstable geometry.
            # NOTE: w can legitimately be negative (DLT sign convention) — guard on |w|, not w.
            w_A = float(pts_4d_A[3, 0])
            w_B = float(pts_4d_B[3, 0])
            ptA = (pts_4d_A[:3] / w_A).flatten() if abs(w_A) > 1e-8 else np.full(3, np.nan, dtype=np.float64)
            ptB = (pts_4d_B[:3] / w_B).flatten() if abs(w_B) > 1e-8 else np.full(3, np.nan, dtype=np.float64)
            
            pts_3d_init.append(ptA)
            pts_3d_init.append(ptB)
            frame_cams.append((fid, cams_in_frame))
        
        pts_3d_init = np.array(pts_3d_init)
        n_pts = len(pts_3d_init)
        n_frames = len(frame_cams)
        
        print(f"  Initial points: {n_pts}")
        
        # Build state vector: [free_cams(n_free_cams*6), pts_3d(n_pts*3)]
        n_cam_params = 6  # Only extrinsics
        pt_start = n_free_cams * n_cam_params
        
        x0 = np.zeros(pt_start + n_pts * 3)
        
        for cid in free_cam_ids:
            idx = free_cam_id_to_idx[cid]
            params = cam_params[cid]
            x0[idx * n_cam_params:(idx + 1) * n_cam_params] = params[:6]
        
        x0[pt_start:] = pts_3d_init.flatten()
        
        # Build sparse Jacobian
        from scipy.sparse import lil_matrix
        
        # Count residuals
        n_res = 0
        for fid, cams_in_frame in frame_cams:
            n_res += 1  # wand length
            n_res += len(cams_in_frame) * 4  # 2 points × 2 coords per camera
        
        n_params = len(x0)
        A_sparsity = lil_matrix((n_res, n_params), dtype=int)
        
        ridx = 0
        for frame_idx, (fid, cams_in_frame) in enumerate(frame_cams):
            idx_ptA = pt_start + frame_idx * 6
            idx_ptB = pt_start + frame_idx * 6 + 3
            
            # Wand length
            A_sparsity[ridx, idx_ptA:idx_ptA+3] = 1
            A_sparsity[ridx, idx_ptB:idx_ptB+3] = 1
            ridx += 1
            
            # Reprojection for each camera
            for cid in cams_in_frame:
                if cid in fixed_cam_ids:
                    # Frozen camera contributes residual rows with point-only Jacobian terms.
                    A_sparsity[ridx:ridx+2, idx_ptA:idx_ptA+3] = 1
                    ridx += 2
                    A_sparsity[ridx:ridx+2, idx_ptB:idx_ptB+3] = 1
                    ridx += 2
                    continue

                cam_idx = free_cam_id_to_idx[cid]
                cam_start = cam_idx * n_cam_params
                
                # ptA projection (2 residuals)
                A_sparsity[ridx:ridx+2, cam_start:cam_start+6] = 1
                A_sparsity[ridx:ridx+2, idx_ptA:idx_ptA+3] = 1
                ridx += 2
                
                # ptB projection (2 residuals)
                A_sparsity[ridx:ridx+2, cam_start:cam_start+6] = 1
                A_sparsity[ridx:ridx+2, idx_ptB:idx_ptB+3] = 1
                ridx += 2
        
        print(f"  Residuals: {n_res}, Params: {n_params}")
        
        # Residuals function
        self._phase3_res_count = 0
        def residuals_phase3(x):
            # Extract camera params
            cams = {}
            for cid in fixed_cam_ids:
                p_fix = cam_params[cid]
                R_fix, _ = cv2.Rodrigues(p_fix[:3])
                t_fix = p_fix[3:6].reshape(3, 1)
                cams[cid] = (R_fix, t_fix)
            for cid in free_cam_ids:
                idx = free_cam_id_to_idx[cid]
                p = x[idx * n_cam_params:(idx + 1) * n_cam_params]
                R, _ = cv2.Rodrigues(p[:3])
                t = p[3:6].reshape(3, 1)
                cams[cid] = (R, t)
            
            pts = x[pt_start:].reshape(-1, 3)
            
            res = []
            
            # Track stats for progress reporting
            sq_err_len = 0.0
            n_len = 0
            sq_err_proj = 0.0
            n_proj = 0
            
            for frame_idx, (fid, cams_in_frame) in enumerate(frame_cams):
                ptA = pts[frame_idx * 2]
                ptB = pts[frame_idx * 2 + 1]
                
                # Wand length
                wand_len = np.linalg.norm(ptB - ptA)
                d_len = wand_len - self.config.wand_length_mm
                res.append(d_len)
                
                sq_err_len += d_len**2
                n_len += 1
                
                # Reprojection for each camera
                for cid in cams_in_frame:
                    R_cam, t_cam = cams[cid]
                    uvA, uvB = observations[fid][cid]
                    
                    # Project ptA (cheirality: NaN → large penalty)
                    proj_A = self._project(ptA, R_cam, t_cam, K_by_cam[cid])
                    diffA = np.array([100.0, 100.0]) if np.isnan(proj_A[0]) else (proj_A - uvA)
                    res.extend(diffA.tolist())
                    sq_err_proj += float(np.sum(diffA**2))
                    n_proj += 2
                    
                    # Project ptB (cheirality: NaN → large penalty)
                    proj_B = self._project(ptB, R_cam, t_cam, K_by_cam[cid])
                    diffB = np.array([100.0, 100.0]) if np.isnan(proj_B[0]) else (proj_B - uvB)
                    res.extend(diffB.tolist())
                    sq_err_proj += float(np.sum(diffB**2))
                    n_proj += 2

            res_arr = np.array(res)

            # Report progress
            self._phase3_res_count += 1
            if progress_callback and self._phase3_res_count % 5 == 0:
                try:
                    rmse_len = np.sqrt(sq_err_len / max(1, n_len))
                    rmse_ray = -1.0
                    rmse_proj = np.sqrt(sq_err_proj / max(1, n_proj))
                    cost = 0.5 * float(np.sum(res_arr**2))
                    progress_callback(
                        "Use PinHole model to initialize camera parameters...",
                        rmse_ray,
                        rmse_len,
                        rmse_proj,
                        cost,
                    )
                except Exception as e:
                    logger.debug("progress_callback error (Phase 3 BA iter): %s", e)
            
            return res_arr

        
        # Run global BA
        print("  Running global BA...")
        result = least_squares(
            residuals_phase3, x0,
            jac_sparsity=A_sparsity,
            method='trf',
            loss='linear',
            x_scale='jac',
            f_scale=1.0,
            verbose=1,
            ftol=self.config.ftol,
            xtol=self.config.xtol,
            max_nfev=100,
        )
        
        print(f"  Phase 3 cost: {result.cost:.2e}")
        
        # Extract optimized params
        cam_params_opt = {
            cid: cam_params[cid].copy() for cid in fixed_cam_ids
        }
        for cid in free_cam_ids:
            idx = free_cam_id_to_idx[cid]
            cam_params_opt[cid] = result.x[idx * n_cam_params:(idx + 1) * n_cam_params]

        baseline_phase3 = float(
            np.linalg.norm(
                _camera_center_from_params(cam_params_opt[cam_j])
                - _camera_center_from_params(cam_params_opt[cam_i])
            )
        )
        print(f"  baseline_phase3 (cam_{cam_i}-cam_{cam_j}): {baseline_phase3:.3f} mm")
        print(f"  baseline drift ratio (phase3/phase1): {baseline_phase3 / max(1e-12, baseline_phase1):.6f}")
        
        # Compute final RMS using wand residuals from interleaved frame blocks
        final_res = residuals_phase3(result.x)
        rms = self._compute_phase3_wand_rms(final_res, n_frames, n_cams)
        print(f"  Final RMS: {rms:.2f}px")
        
        return cam_params_opt
    
    def run_all(
        self,
        cam_i: int,
        cam_j: int,
        observations: Dict[int, Dict[int, Tuple[np.ndarray, np.ndarray]]],
        camera_settings: Dict[int, dict],
        all_cam_ids: List[int],
        progress_callback=None
    ) -> Tuple[Dict[int, np.ndarray], dict]:
        """
        Run full P0 bootstrap: Phase 1 (8-Point + BA) + Phase 2 (PnP) + Phase 3 (Global BA).
        
        All phases use frozen intrinsics (fixed focal length).
        """
        # Phase 1: Calibrate best pair
        params_i, params_j, report = self.run(
            cam_i,
            cam_j,
            observations,
            camera_settings,
            progress_callback=progress_callback,
            all_cam_ids=all_cam_ids,
        )

        phase1_pair = report.get('phase1_pair', (cam_i, cam_j))
        cam_i_run, cam_j_run = int(phase1_pair[0]), int(phase1_pair[1])
        
        cam_params = {
            cam_i_run: params_i,
            cam_j_run: params_j,
        }

        # Propagate Phase 1 inlier-validated frame set into Phase 2/3.
        inlier_frames_phase1 = report.get('phase1_inlier_frames', [])
        if inlier_frames_phase1:
            observations_phase23 = {
                fid: observations[fid]
                for fid in inlier_frames_phase1
                if fid in observations
            }
        else:
            observations_phase23 = observations

        valid_frames_phase23 = [
            fid for fid, cams in observations_phase23.items() if len(cams) >= 2
        ]
        print(f"  Phase 2: {len(valid_frames_phase23)} valid frames (≥2 cameras)")
        if len(valid_frames_phase23) < len(observations_phase23):
            print(
                "  [WARN] Some Phase 2/3 frames have <2 cameras and will be skipped "
                f"({len(observations_phase23) - len(valid_frames_phase23)} frames)."
            )
        
        # Triangulate 3D points
        print(f"\n[P0] Triangulating 3D points for Phase 2...")
        if progress_callback:
            try:
                progress_callback("Use PinHole model to initialize camera parameters...", -1, 0, 0, 0)
            except Exception as e:
                logger.debug("progress_callback error (Triangulate for Phase 2): %s", e)

        points_3d = self.triangulate_all_points(
            cam_i_run, cam_j_run, params_i, params_j, observations_phase23, camera_settings
        )
        report['points_3d'] = points_3d
        print(f"  Triangulated {len(points_3d)} frames")
        
        # Phase 2: Calibrate remaining cameras
        if progress_callback:
            try:
                progress_callback("Use PinHole model to initialize camera parameters...", -1, 0, 0, 0)
            except Exception as e:
                logger.debug("progress_callback error (Phase 2): %s", e)
        
        cam_params = self.run_phase2(
            cam_params, observations_phase23, points_3d, camera_settings, all_cam_ids
        )
        
        # Phase 3: Global BA with all cameras
        if progress_callback:
            try:
                progress_callback("Use PinHole model to initialize camera parameters...", -1, 0, 0, 0)
            except Exception as e:
                logger.debug("progress_callback error (Phase 3): %s", e)

        valid_frames_phase3 = [
            fid for fid, cams in observations_phase23.items() if len(cams) >= 2
        ]
        print(f"  Phase 3: {len(valid_frames_phase3)} valid frames (≥2 cameras)")

        cam_params = self.run_phase3(
            cam_params,
            observations_phase23,
            camera_settings,
            cam_i=cam_i_run,
            cam_j=cam_j_run,
            progress_callback=progress_callback,
        )

        
        report['all_cam_ids'] = list(cam_params.keys())
        report['phase23_frames'] = list(observations_phase23.keys())
        
        return cam_params, report

