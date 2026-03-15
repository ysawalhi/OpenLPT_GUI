#!/usr/bin/env python
"""
Experimental staged refractive wand calibration strategy test.

Run this under OpenLPT environment.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import least_squares
import pyopenlpt as lpt
import cv2

from modules.camera_calibration.wand_calibration.wand_calibrator import WandCalibrator
from modules.camera_calibration.wand_calibration.refraction_wand_calibrator import RefractiveWandCalibrator
from modules.camera_calibration.wand_calibration.refractive_bootstrap import PinholeBootstrapP0, PinholeBootstrapP0Config
from modules.camera_calibration.wand_calibration.refraction_calibration_BA import (
    RefractiveBAConfig,
    RefractiveBAOptimizer,
    CppSyncAdapter,
)
from modules.camera_calibration.wand_calibration.refractive_geometry import (
    build_pinplate_rays_cpp_batch,
    point_to_ray_dist,
    triangulate_point,
    update_normal_tangent,
)


class ProgressLogger:
    def __init__(self, every: int = 50):
        self.every = max(1, int(every))
        self.count = 0

    def callback(self, phase: str, ray_rmse: float, len_rmse: float, proj_rmse: float, cost: float):
        self.count += 1
        if self.count % self.every != 0:
            return
        print(
            f"[Progress#{self.count:05d}] {phase} | "
            f"ray={ray_rmse:.6f} mm, len={len_rmse:.6f} mm, proj={proj_rmse:.6f} px, cost={cost:.6e}"
        )


def _cam_center_axis_from_param(p: np.ndarray, axis_len: float = 80.0) -> Tuple[np.ndarray, np.ndarray]:
    rvec = np.asarray(p[0:3], dtype=np.float64).reshape(3, 1)
    tvec = np.asarray(p[3:6], dtype=np.float64).reshape(3, 1)
    R, _ = cv2.Rodrigues(rvec)
    C = (-R.T @ tvec).reshape(3)
    # World direction of camera +Z optical axis.
    z_world = (R.T @ np.array([0.0, 0.0, 1.0], dtype=np.float64)).reshape(3)
    z_world = z_world / (np.linalg.norm(z_world) + 1e-12)
    return C, C + z_world * float(axis_len)


def _plane_patch(plane_pt: np.ndarray, plane_n: np.ndarray, size: float = 200.0) -> np.ndarray:
    n = np.asarray(plane_n, dtype=np.float64).reshape(3)
    n = n / (np.linalg.norm(n) + 1e-12)
    a = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(a, n))) > 0.9:
        a = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    u = np.cross(n, a)
    u = u / (np.linalg.norm(u) + 1e-12)
    v = np.cross(n, u)
    p = np.asarray(plane_pt, dtype=np.float64).reshape(3)
    s = float(size)
    return np.array(
        [
            p + u * s + v * s,
            p - u * s + v * s,
            p - u * s - v * s,
            p + u * s - v * s,
            p + u * s + v * s,
        ],
        dtype=np.float64,
    )


def _plot_best_pair_diagnostics(
    out_png: Path,
    seed_wid: int,
    seed_pair: Tuple[int, int],
    seed_points: Dict[int, Dict[str, np.ndarray]],
    window_plane: Dict[str, np.ndarray],
    p0_cam_params: Dict[int, np.ndarray],
    best_pair_cam_params: Dict[int, np.ndarray],
    same_plane_cam_ids: List[int],
    show_plot: bool = False,
) -> None:
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    except Exception as e:
        print(f"[STAGED][PLOT][WARN] matplotlib unavailable, skip plot: {e}")
        return

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")

    # 1) 3D points (A/B merged)
    pts = []
    for fid in sorted(seed_points.keys()):
        pts.append(np.asarray(seed_points[fid]["A"], dtype=np.float64))
        pts.append(np.asarray(seed_points[fid]["B"], dtype=np.float64))
    if pts:
        arr = np.vstack(pts)
        ax.scatter(arr[:, 0], arr[:, 1], arr[:, 2], s=3, c="#7f8c8d", alpha=0.5, label="3D points (best-pair triangulated)")

    # 2) Plane
    plane_pt = np.asarray(window_plane["plane_pt"], dtype=np.float64)
    plane_n = np.asarray(window_plane["plane_n"], dtype=np.float64)
    patch = _plane_patch(plane_pt, plane_n, size=220.0)
    ax.plot(patch[:, 0], patch[:, 1], patch[:, 2], color="#1f77b4", linewidth=2.0, label=f"Plane {seed_wid}")
    ax.plot(
        [plane_pt[0], plane_pt[0] + plane_n[0] * 120.0],
        [plane_pt[1], plane_pt[1] + plane_n[1] * 120.0],
        [plane_pt[2], plane_pt[2] + plane_n[2] * 120.0],
        color="#1f77b4",
        linewidth=3.0,
        label=f"Plane {seed_wid} normal",
    )

    # 3) P0 cameras on this plane
    p0_color = "#2ca02c"
    sorted_plane_cams = sorted(same_plane_cam_ids)
    for i, cid in enumerate(sorted_plane_cams):
        if cid not in p0_cam_params:
            continue
        c0, c1 = _cam_center_axis_from_param(p0_cam_params[cid], axis_len=90.0)
        ax.scatter([c0[0]], [c0[1]], [c0[2]], c=p0_color, s=36, marker="o", label="P0 cams (same plane)" if i == 0 else None)
        ax.plot([c0[0], c1[0]], [c0[1], c1[1]], [c0[2], c1[2]], color=p0_color, linewidth=2)
        ax.text(c0[0], c0[1], c0[2], f"P0 Cam{cid}", color=p0_color, fontsize=8)

    # 4) Best-pair optimized cameras
    best_color = "#d62728"
    for i, cid in enumerate(seed_pair):
        if cid not in best_pair_cam_params:
            continue
        c0, c1 = _cam_center_axis_from_param(best_pair_cam_params[cid], axis_len=110.0)
        ax.scatter([c0[0]], [c0[1]], [c0[2]], c=best_color, s=55, marker="D", label="Best pair (optimized)" if i == 0 else None)
        ax.plot([c0[0], c1[0]], [c0[1], c1[1]], [c0[2], c1[2]], color=best_color, linewidth=3)
        ax.text(c0[0], c0[1], c0[2], f"BestPair Cam{cid}", color=best_color, fontsize=9)

    ax.set_title(f"Best Pair Diagnostic (Window {seed_wid}, Pair {seed_pair})")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.25)
    ax.set_box_aspect((1, 1, 1))

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(str(out_png), dpi=180)
    print(f"[STAGED][PLOT] Plot saved: {out_png}")
    if show_plot:
        plt.show()
    plt.close(fig)


def _camera_settings_default() -> Dict[int, Dict[str, float]]:
    return {
        1: {"focal": 5250.0, "width": 1280, "height": 800},
        2: {"focal": 5250.0, "width": 1280, "height": 800},
        3: {"focal": 5250.0, "width": 1280, "height": 800},
        4: {"focal": 5250.0, "width": 1280, "height": 800},
        5: {"focal": 9000.0, "width": 1024, "height": 976},
    }


def _cam_to_window_default() -> Dict[int, int]:
    return {1: 0, 2: 0, 3: 0, 4: 0, 5: 1}


def _window_media_default() -> Dict[int, Dict[str, float]]:
    return {
        0: {"n1": 1.0, "n2": 1.49, "n3": 1.333, "thickness": 31.75},
        1: {"n1": 1.0, "n2": 1.49, "n3": 1.333, "thickness": 31.75},
    }


def _deepcopy_dataset(dataset: Dict) -> Dict:
    return {
        "frames": list(dataset["frames"]),
        "cam_ids": list(dataset["cam_ids"]),
        "obsA": {fid: dict(v) for fid, v in dataset["obsA"].items()},
        "obsB": {fid: dict(v) for fid, v in dataset["obsB"].items()},
        "radii_small": {fid: dict(v) for fid, v in dataset["radii_small"].items()},
        "radii_large": {fid: dict(v) for fid, v in dataset["radii_large"].items()},
        "maskA": {fid: dict(v) for fid, v in dataset["maskA"].items()},
        "maskB": {fid: dict(v) for fid, v in dataset["maskB"].items()},
        "num_frames": int(dataset["num_frames"]),
        "num_cams": int(dataset["num_cams"]),
        "wand_length": float(dataset["wand_length"]),
        "dist_coeff_num": int(dataset.get("dist_coeff_num", 2)),
        "total_observations": int(dataset.get("total_observations", 0)),
        "est_radius_small_mm": float(dataset.get("est_radius_small_mm", 1.5)),
        "est_radius_large_mm": float(dataset.get("est_radius_large_mm", 2.0)),
    }


def _filter_dataset_by_cams(dataset: Dict, cams_keep: List[int]) -> Dict:
    cams_set = set(cams_keep)
    out = _deepcopy_dataset(dataset)
    out["cam_ids"] = sorted(cams_set)

    for fid in out["frames"]:
        out["obsA"][fid] = {cid: uv for cid, uv in out["obsA"][fid].items() if cid in cams_set}
        out["obsB"][fid] = {cid: uv for cid, uv in out["obsB"][fid].items() if cid in cams_set}
        out["radii_small"][fid] = {cid: r for cid, r in out["radii_small"][fid].items() if cid in cams_set}
        out["radii_large"][fid] = {cid: r for cid, r in out["radii_large"][fid].items() if cid in cams_set}
        out["maskA"][fid] = {cid: m for cid, m in out["maskA"][fid].items() if cid in cams_set}
        out["maskB"][fid] = {cid: m for cid, m in out["maskB"][fid].items() if cid in cams_set}

    out["num_cams"] = len(out["cam_ids"])
    out["total_observations"] = len(out["frames"]) * max(1, len(out["cam_ids"]))
    return out


def _rms(v: List[float]) -> float:
    arr = np.asarray(v, dtype=np.float64)
    if arr.size == 0:
        return float("inf")
    return float(np.sqrt(np.mean(arr * arr)))


def _calc_p0_cam_error(base: WandCalibrator) -> Dict[int, float]:
    per_cam: Dict[int, List[float]] = {}
    for _, frame_data in getattr(base, "per_frame_errors", {}).items():
        for cid, err in frame_data.get("cam_errors", {}).items():
            per_cam.setdefault(int(cid), []).append(float(err))
    return {cid: _rms(vals) for cid, vals in per_cam.items()}


def _select_seed_pair_same_window(per_cam_err: Dict[int, float], cam_to_window: Dict[int, int], active_cams: List[int]) -> Tuple[int, int, int]:
    by_window: Dict[int, List[int]] = {}
    for cid in active_cams:
        wid = int(cam_to_window[cid])
        by_window.setdefault(wid, []).append(cid)

    best = None
    best_score = float("inf")
    for wid, cams in sorted(by_window.items()):
        if len(cams) < 2:
            continue
        cams_sorted = sorted(cams, key=lambda c: per_cam_err.get(c, float("inf")))
        c1, c2 = cams_sorted[0], cams_sorted[1]
        score = 0.5 * (per_cam_err.get(c1, float("inf")) + per_cam_err.get(c2, float("inf")))
        if score < best_score:
            best_score = score
            best = (wid, c1, c2)

    if best is None:
        raise RuntimeError("No window has >=2 active cameras, cannot choose same-window seed pair.")
    return best


def _fallback_seed_pair_by_shared_frames(
    wand_points: Dict[int, Dict[int, np.ndarray]],
    cam_to_window: Dict[int, int],
    active_cams: List[int],
) -> Tuple[int, int, int]:
    by_window: Dict[int, List[int]] = {}
    for cid in active_cams:
        by_window.setdefault(int(cam_to_window[cid]), []).append(int(cid))

    best = None
    best_count = -1
    for wid, cams in sorted(by_window.items()):
        if len(cams) < 2:
            continue
        cams = sorted(cams)
        for i in range(len(cams)):
            for j in range(i + 1, len(cams)):
                c1, c2 = cams[i], cams[j]
                cnt = 0
                for _, frame in wand_points.items():
                    if c1 in frame and c2 in frame:
                        cnt += 1
                if cnt > best_count:
                    best_count = cnt
                    best = (wid, c1, c2)

    if best is None:
        raise RuntimeError("Fallback seed-pair selection failed: no same-window camera pair found.")
    return best


def _run_optimizer(dataset, cam_params, cams_cpp, cam_to_window, window_media, window_planes, wand_length, cfg, stage, progress_cb=None):
    opt = RefractiveBAOptimizer(
        dataset=dataset,
        cam_params=cam_params,
        cams_cpp=cams_cpp,
        cam_to_window=cam_to_window,
        window_media=window_media,
        window_planes=window_planes,
        wand_length=wand_length,
        config=cfg,
        progress_callback=progress_cb,
    )
    wp, cp = opt.optimize(stage=stage)
    return wp, cp, opt.window_media


def _run_final_refined_only(dataset, cam_params, cams_cpp, cam_to_window, window_media, window_planes, wand_length, cfg, progress_cb=None):
    opt = RefractiveBAOptimizer(
        dataset=dataset,
        cam_params=cam_params,
        cams_cpp=cams_cpp,
        cam_to_window=cam_to_window,
        window_media=window_media,
        window_planes=window_planes,
        wand_length=wand_length,
        config=cfg,
        progress_callback=progress_cb,
    )

    enable_k1 = cfg.dist_coeff_num >= 1
    enable_k2 = cfg.dist_coeff_num >= 2
    opt._optimize_generic(
        mode="final_refined",
        description="Global Final Refined (staged strategy)",
        enable_planes=True,
        enable_cam_t=True,
        enable_cam_r=True,
        enable_cam_f=True,
        enable_win_t=True,
        enable_cam_k1=enable_k1,
        enable_cam_k2=enable_k2,
        limit_rot_rad=np.radians(20.0),
        limit_trans_mm=50.0,
        limit_plane_d_mm=10.0,
        limit_plane_angle_rad=np.radians(5.0),
        ftol=1e-5,
        xtol=1e-5,
        gtol=1e-5,
        loss=cfg.loss_round4,
        max_nfev=100,
    )
    return opt.window_planes, opt.cam_params, opt.window_media


def _run_best_pair_fixed_plane(
    dataset: Dict,
    cam_params: Dict[int, np.ndarray],
    cams_cpp: Dict,
    cam_to_window: Dict[int, int],
    window_media: Dict[int, Dict[str, float]],
    window_planes: Dict[int, Dict],
    wand_length: float,
    cfg: RefractiveBAConfig,
    progress_cb=None,
) -> Tuple[Dict[int, Dict], Dict[int, np.ndarray], Dict[int, Dict[str, float]]]:
    """
    Optimize best-pair cameras with plane fixed (no plane variables).
    """
    opt = RefractiveBAOptimizer(
        dataset=dataset,
        cam_params=cam_params,
        cams_cpp=cams_cpp,
        cam_to_window=cam_to_window,
        window_media=window_media,
        window_planes=window_planes,
        wand_length=wand_length,
        config=cfg,
        progress_callback=progress_cb,
    )
    opt._optimize_generic(
        mode="best_pair_fixed_plane_cams",
        description="Best pair optimization (fixed plane, extrinsics only)",
        enable_planes=False,
        enable_cam_t=True,
        enable_cam_r=True,
        limit_rot_rad=np.deg2rad(180.0),
        limit_trans_mm=2000.0,
        limit_plane_d_mm=0.0,
        limit_plane_angle_rad=0.0,
        ftol=5e-4,
        xtol=1e-5,
        gtol=1e-5,
        loss=cfg.loss_cam,
        max_nfev=80,
    )
    return opt.window_planes, opt.cam_params, opt.window_media


def _triangulate_seed_points(dataset, cams_cpp, seed_pair):
    c1, c2 = seed_pair
    out = {}
    for fid in dataset["frames"]:
        obsA = dataset["obsA"].get(fid, {})
        obsB = dataset["obsB"].get(fid, {})
        if c1 not in obsA or c2 not in obsA or c1 not in obsB or c2 not in obsB:
            continue

        raysA = build_pinplate_rays_cpp_batch(
            cams_cpp[c1],
            [obsA[c1]],
            meta_list=[{"cam_id": c1, "frame_id": fid, "endpoint": "A"}],
        )
        raysA += build_pinplate_rays_cpp_batch(
            cams_cpp[c2],
            [obsA[c2]],
            meta_list=[{"cam_id": c2, "frame_id": fid, "endpoint": "A"}],
        )
        raysB = build_pinplate_rays_cpp_batch(
            cams_cpp[c1],
            [obsB[c1]],
            meta_list=[{"cam_id": c1, "frame_id": fid, "endpoint": "B"}],
        )
        raysB += build_pinplate_rays_cpp_batch(
            cams_cpp[c2],
            [obsB[c2]],
            meta_list=[{"cam_id": c2, "frame_id": fid, "endpoint": "B"}],
        )

        XA, _, okA, _ = triangulate_point(raysA)
        XB, _, okB, _ = triangulate_point(raysB)
        if okA and okB:
            out[fid] = {"A": XA, "B": XB}
    return out


def _select_evenly_spaced_frames(frames: List[int], max_frames: int) -> List[int]:
    frames_sorted = sorted(int(fid) for fid in frames)
    if max_frames <= 0 or len(frames_sorted) <= max_frames:
        return frames_sorted
    idx = np.linspace(0, len(frames_sorted) - 1, num=max_frames)
    picked = sorted(set(int(round(v)) for v in idx))
    return [frames_sorted[i] for i in picked]


def _estimate_plane_search_scale(
    cam_params: Dict[int, np.ndarray],
    cams_on_window: List[int],
    XA: Dict[int, np.ndarray],
    XB: Dict[int, np.ndarray],
) -> float:
    mids = []
    for fid, pa in XA.items():
        pb = XB.get(fid)
        if pb is None:
            continue
        mids.append(0.5 * (np.asarray(pa, dtype=np.float64) + np.asarray(pb, dtype=np.float64)))
    if not mids:
        return 500.0

    centers = []
    for cid in cams_on_window:
        if cid not in cam_params:
            continue
        p = np.asarray(cam_params[cid], dtype=np.float64)
        rvec = p[0:3].reshape(3, 1)
        tvec = p[3:6].reshape(3, 1)
        R, _ = cv2.Rodrigues(rvec)
        centers.append((-R.T @ tvec).reshape(3))
    if not centers:
        return 500.0

    mids_arr = np.vstack(mids)
    scales = []
    for C in centers:
        d = np.linalg.norm(mids_arr - C.reshape(1, 3), axis=1)
        if d.size > 0:
            scales.append(float(np.median(d)))
    if not scales:
        return 500.0
    return max(100.0, float(np.median(scales)))


def _score_plane_candidate_for_window(
    dataset: Dict,
    cam_params: Dict[int, np.ndarray],
    cams_on_window: List[int],
    cam_to_window: Dict[int, int],
    window_media: Dict[int, Dict[str, float]],
    window_plane: Dict[str, np.ndarray],
    target_wid: int,
    frame_ids: List[int],
    wand_length: float,
    refr: RefractiveWandCalibrator,
) -> Dict[str, float]:
    cp_eval = {cid: np.asarray(cam_params[cid], dtype=np.float64).copy() for cid in cams_on_window if cid in cam_params}
    ctw_eval = {cid: cam_to_window[cid] for cid in cams_on_window if cid in cam_to_window}
    wm_eval = {target_wid: copy.deepcopy(window_media[target_wid])}
    wp_eval = {
        target_wid: {
            "plane_pt": np.asarray(window_plane["plane_pt"], dtype=np.float64).copy(),
            "plane_n": np.asarray(window_plane["plane_n"], dtype=np.float64).copy(),
        }
    }
    cams_cpp = refr._init_cams_cpp_in_memory(cp_eval, wm_eval, ctw_eval, wp_eval)

    ray_errors = []
    len_errors = []
    invalid_ray_count = 0
    total_ray_count = 0
    failed_points = 0
    attempted_points = 0
    used_frames = 0

    for fid in frame_ids:
        obsA_all = dataset["obsA"].get(fid, {})
        obsB_all = dataset["obsB"].get(fid, {})
        obsA = {cid: obsA_all[cid] for cid in cams_on_window if cid in obsA_all}
        obsB = {cid: obsB_all[cid] for cid in cams_on_window if cid in obsB_all}
        if len(obsA) < 2 or len(obsB) < 2:
            continue

        endpoint_res = {}
        frame_ok = True
        for endpoint, obs in (("A", obsA), ("B", obsB)):
            attempted_points += 1
            cids = sorted(obs.keys())
            rays = build_pinplate_rays_cpp_batch(
                cams_cpp[cids[0]],
                [obs[cids[0]]],
                meta_list=[{"cam_id": cids[0], "frame_id": fid, "endpoint": endpoint}],
            )
            for cid in cids[1:]:
                rays += build_pinplate_rays_cpp_batch(
                    cams_cpp[cid],
                    [obs[cid]],
                    meta_list=[{"cam_id": cid, "frame_id": fid, "endpoint": endpoint}],
                )

            total_ray_count += len(rays)
            invalid_ray_count += sum(0 if getattr(r, "valid", False) else 1 for r in rays)
            rays_valid = [r for r in rays if getattr(r, "valid", False)]
            if len(rays_valid) < 2:
                failed_points += 1
                frame_ok = False
                continue

            X, _, ok, _ = triangulate_point(rays_valid)
            if not ok:
                failed_points += 1
                frame_ok = False
                continue

            dists = [point_to_ray_dist(np.asarray(X, dtype=np.float64), np.asarray(r.o, dtype=np.float64), np.asarray(r.d, dtype=np.float64)) for r in rays_valid]
            ray_errors.append(float(np.mean(dists)))
            endpoint_res[endpoint] = np.asarray(X, dtype=np.float64)

        if frame_ok and "A" in endpoint_res and "B" in endpoint_res:
            used_frames += 1
            len_errors.append(abs(float(np.linalg.norm(endpoint_res["A"] - endpoint_res["B"]) - wand_length)))

    ray_med = float(np.median(ray_errors)) if ray_errors else float("inf")
    len_med = float(np.median(len_errors)) if len_errors else float("inf")
    invalid_rate = float(invalid_ray_count / max(1, total_ray_count))
    fail_rate = float(failed_points / max(1, attempted_points))
    coverage_rate = float(used_frames / max(1, len(frame_ids)))
    score = ray_med + len_med + wand_length * (1.5 * fail_rate + 0.5 * invalid_rate + (1.0 - coverage_rate))
    if used_frames <= 0 or not np.isfinite(score):
        score = float("inf")
    return {
        "score": float(score),
        "ray_median_mm": ray_med,
        "len_median_mm": len_med,
        "invalid_ray_rate": invalid_rate,
        "failed_point_rate": fail_rate,
        "coverage_rate": coverage_rate,
        "used_frames": int(used_frames),
        "attempted_points": int(attempted_points),
        "total_ray_count": int(total_ray_count),
    }


def _search_seed_window_plane_initializer(
    dataset: Dict,
    cam_params: Dict[int, np.ndarray],
    cam_to_window: Dict[int, int],
    window_media: Dict[int, Dict[str, float]],
    base_plane: Dict[str, np.ndarray],
    seed_wid: int,
    same_plane_cams: List[int],
    XA: Dict[int, np.ndarray],
    XB: Dict[int, np.ndarray],
    wand_length: float,
    refr: RefractiveWandCalibrator,
    offset_frac: float,
    tilt_deg: float,
    offset_steps: int,
    tilt_steps: int,
    max_frames: int,
) -> Dict[str, object]:
    if offset_steps < 1 or tilt_steps < 1:
        raise ValueError("offset_steps and tilt_steps must be >= 1")

    base_pt = np.asarray(base_plane["plane_pt"], dtype=np.float64).copy()
    base_n = np.asarray(base_plane["plane_n"], dtype=np.float64).copy()
    base_n = base_n / (np.linalg.norm(base_n) + 1e-12)
    scale_mm = _estimate_plane_search_scale(cam_params, same_plane_cams, XA, XB)
    offset_extent = abs(float(offset_frac)) * scale_mm
    tilt_extent_rad = np.deg2rad(abs(float(tilt_deg)))
    offset_grid = np.linspace(-offset_extent, offset_extent, num=2 * int(offset_steps) + 1)
    tilt_grid = np.linspace(-tilt_extent_rad, tilt_extent_rad, num=2 * int(tilt_steps) + 1)
    frames = _select_evenly_spaced_frames(dataset["frames"], max_frames=max_frames)
    candidates = []
    best = None
    for d in offset_grid:
        for a in tilt_grid:
            for b in tilt_grid:
                plane_n = update_normal_tangent(base_n, float(a), float(b))
                plane_pt = base_pt + float(d) * base_n
                metrics = _score_plane_candidate_for_window(
                    dataset=dataset,
                    cam_params=cam_params,
                    cams_on_window=same_plane_cams,
                    cam_to_window=cam_to_window,
                    window_media=window_media,
                    window_plane={"plane_pt": plane_pt, "plane_n": plane_n},
                    target_wid=seed_wid,
                    frame_ids=frames,
                    wand_length=wand_length,
                    refr=refr,
                )
                rec = {
                    "offset_mm": float(d),
                    "tilt_a_deg": float(np.rad2deg(a)),
                    "tilt_b_deg": float(np.rad2deg(b)),
                    **metrics,
                }
                candidates.append(rec)
                if best is None or rec["score"] < best["score"]:
                    best = rec

    if best is None:
        raise RuntimeError("Plane initializer search failed: no candidates evaluated.")
    if (not np.isfinite(float(best["score"]))) or int(best.get("used_frames", 0)) <= 0:
        raise RuntimeError(
            "Plane initializer search failed: no candidate produced enough valid frames for a finite score. "
            f"Try widening the search or using baseline mode. seed_wid={seed_wid}"
        )

    hit_offset_boundary = abs(best["offset_mm"]) >= max(abs(float(offset_grid[0])), abs(float(offset_grid[-1]))) - 1e-9
    hit_tilt_boundary = max(abs(best["tilt_a_deg"]), abs(best["tilt_b_deg"])) >= abs(float(np.rad2deg(tilt_grid[-1]))) - 1e-9
    best_plane_n = update_normal_tangent(base_n, np.deg2rad(best["tilt_a_deg"]), np.deg2rad(best["tilt_b_deg"]))
    best_plane_pt = base_pt + best["offset_mm"] * base_n
    return {
        "scale_mm": float(scale_mm),
        "offset_extent_mm": float(offset_extent),
        "tilt_extent_deg": float(abs(float(tilt_deg))),
        "frames_evaluated": int(len(frames)),
        "candidate_count": int(len(candidates)),
        "boundary_hit": bool(hit_offset_boundary or hit_tilt_boundary),
        "boundary_hit_offset": bool(hit_offset_boundary),
        "boundary_hit_tilt": bool(hit_tilt_boundary),
        "best_candidate": best,
        "best_plane": {
            "plane_pt": np.asarray(best_plane_pt, dtype=np.float64),
            "plane_n": np.asarray(best_plane_n, dtype=np.float64),
        },
        "top_candidates": sorted(candidates, key=lambda x: x["score"])[:10],
    }


def _compute_final_metrics(
    dataset: Dict,
    cam_params: Dict[int, np.ndarray],
    cams_cpp: Dict,
    cam_to_window: Dict[int, int],
    window_media: Dict[int, Dict[str, float]],
    window_planes: Dict[int, Dict],
    wand_length: float,
    cfg: RefractiveBAConfig,
) -> Dict[str, float]:
    opt = RefractiveBAOptimizer(
        dataset=dataset,
        cam_params=cam_params,
        cams_cpp=cams_cpp,
        cam_to_window=cam_to_window,
        window_media=window_media,
        window_planes=window_planes,
        wand_length=wand_length,
        config=cfg,
        progress_callback=None,
    )
    opt._compute_physical_sigmas()
    lambda_fixed = opt._compute_lambda_fixed("final_refined")
    _, S_ray, S_len, N_ray, N_len, S_proj, N_proj = opt.evaluate_residuals(
        opt.window_planes,
        opt.cam_params,
        lambda_fixed,
        window_media=opt.window_media,
    )
    return {
        "ray_rmse_mm": float(np.sqrt(S_ray / max(1, N_ray))) if N_ray > 0 else float("inf"),
        "len_rmse_mm": float(np.sqrt(S_len / max(1, N_len))) if N_len > 0 else float("inf"),
        "proj_rmse_px": float(np.sqrt(S_proj / max(1, N_proj))) if N_proj > 0 else None,
        "n_ray": int(N_ray),
        "n_len": int(N_len),
        "n_proj": int(N_proj),
    }


def _try_align_cam_ids(cam_settings, cam_to_window, observed_cam_ids):
    obs = sorted(int(c) for c in observed_cam_ids)
    cfg = sorted(cam_settings.keys())
    if obs == cfg:
        return cam_settings, cam_to_window

    shifted = {k - 1: v for k, v in cam_settings.items()}
    shifted_map = {k - 1: v for k, v in cam_to_window.items()}
    if sorted(shifted.keys()) == obs:
        return shifted, shifted_map

    raise RuntimeError(f"Camera IDs mismatch. observed={obs}, config={cfg}, config_minus1={sorted(shifted.keys())}")


def _optimize_single_cam_extrinsic_with_fixed_points(
    target_cid: int,
    cam_params: Dict[int, np.ndarray],
    cams_cpp: Dict,
    cam_to_window: Dict[int, int],
    window_media: Dict[int, Dict[str, float]],
    window_planes: Dict[int, Dict],
    dataset: Dict,
    fixed_points: Dict[int, Dict[str, np.ndarray]],
    progress_logger: ProgressLogger | None = None,
    max_nfev: int = 80,
    optimize_plane: bool = False,
) -> bool:
    # Build fixed correspondences: (3D point, 2D obs, endpoint-tag)
    p3d_list = []
    uv_list = []
    tags = []
    for fid in sorted(fixed_points.keys()):
        if fid not in dataset["obsA"] or fid not in dataset["obsB"]:
            continue
        obsA = dataset["obsA"][fid]
        obsB = dataset["obsB"][fid]
        if target_cid not in obsA or target_cid not in obsB:
            continue
        p3d_list.append(np.asarray(fixed_points[fid]["A"], dtype=np.float64))
        uv_list.append(np.asarray(obsA[target_cid], dtype=np.float64))
        tags.append((fid, "A"))
        p3d_list.append(np.asarray(fixed_points[fid]["B"], dtype=np.float64))
        uv_list.append(np.asarray(obsB[target_cid], dtype=np.float64))
        tags.append((fid, "B"))

    if len(p3d_list) < 12:
        print(f"[STAGED][Step4][WARN] Cam {target_cid}: too few fixed-point correspondences ({len(p3d_list)}). Skip.")
        return False

    p0 = cam_params[target_cid].copy()
    x0 = p0[:6].copy()
    wid = int(cam_to_window[target_cid])
    plane_ref = window_planes[wid]
    pt0 = np.asarray(plane_ref["plane_pt"], dtype=np.float64).copy()
    n0 = np.asarray(plane_ref["plane_n"], dtype=np.float64).copy()

    # PnP initialization (pinhole approximation) for better starting pose.
    obj_pts = np.asarray(p3d_list, dtype=np.float64).reshape(-1, 1, 3)
    img_pts = np.asarray(uv_list, dtype=np.float64).reshape(-1, 1, 2)
    K = np.array(
        [
            [float(p0[6]), 0.0, float(p0[7])],
            [0.0, float(p0[6]), float(p0[8])],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    dist = np.array([float(p0[9]), float(p0[10]), 0.0, 0.0, 0.0], dtype=np.float64).reshape(-1, 1)

    x0_pnp = x0.copy()
    pnp_used = False
    try:
        ok_ransac, rvec_r, tvec_r, inliers = cv2.solvePnPRansac(
            objectPoints=obj_pts,
            imagePoints=img_pts,
            cameraMatrix=K,
            distCoeffs=dist,
            flags=cv2.SOLVEPNP_EPNP,
            reprojectionError=8.0,
            confidence=0.99,
            iterationsCount=200,
        )
        if ok_ransac:
            ok_refine, rvec_i, tvec_i = cv2.solvePnP(
                objectPoints=obj_pts,
                imagePoints=img_pts,
                cameraMatrix=K,
                distCoeffs=dist,
                rvec=rvec_r,
                tvec=tvec_r,
                useExtrinsicGuess=True,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if ok_refine:
                x0_pnp = np.concatenate(
                    [
                        np.asarray(rvec_i, dtype=np.float64).reshape(3),
                        np.asarray(tvec_i, dtype=np.float64).reshape(3),
                    ]
                )
                pnp_used = True
                inl = 0 if inliers is None else int(len(inliers))
                print(f"[STAGED][Step4] Cam {target_cid}: PnP init success (RANSAC inliers={inl}/{len(p3d_list)}).")
    except Exception as e:
        print(f"[STAGED][Step4][WARN] Cam {target_cid}: PnP init failed, fallback to current pose. err={e}")

    x0_cam = x0_pnp if pnp_used else x0
    if optimize_plane:
        x0_full = np.concatenate([x0_cam, np.zeros(3, dtype=np.float64)])  # [cam6, plane_d, plane_a, plane_b]
        b_lo = np.array([-np.inf] * 6 + [-150.0, -np.deg2rad(20.0), -np.deg2rad(20.0)], dtype=np.float64)
        b_hi = np.array([np.inf] * 6 + [150.0, np.deg2rad(20.0), np.deg2rad(20.0)], dtype=np.float64)
    else:
        x0_full = x0_cam

    pt3d_lpt = [lpt.Pt3D(float(P[0]), float(P[1]), float(P[2])) for P in p3d_list]
    n_calls = {"k": 0}

    def _decode_state(x: np.ndarray):
        cam_x = x[:6]
        if not optimize_plane:
            return cam_x, pt0, n0
        d = float(x[6])
        a = float(x[7])
        b = float(x[8])
        n_new = update_normal_tangent(n0, a, b)
        pt_new = pt0 + d * n0
        return cam_x, pt_new, n_new

    def _apply_state(x: np.ndarray):
        cam_x, pt_new, n_new = _decode_state(x)
        pp = p0.copy()
        pp[:6] = cam_x
        wp_eval = dict(window_planes)
        wp_eval[wid] = {"plane_pt": np.asarray(pt_new, dtype=np.float64), "plane_n": np.asarray(n_new, dtype=np.float64)}
        kwargs = CppSyncAdapter.build_update_kwargs(
            cam_params={target_cid: pp},
            window_planes=wp_eval,
            window_media=window_media,
            cam_to_window=cam_to_window,
            cam_id=target_cid,
        )
        CppSyncAdapter.apply(cams_cpp, target_cid, kwargs)

    def _residual(x: np.ndarray) -> np.ndarray:
        _apply_state(x)
        n_calls["k"] += 1
        proj = cams_cpp[target_cid].projectBatchStatus(pt3d_lpt, False)
        res = np.zeros(2 * len(uv_list), dtype=np.float64)
        PEN = 50.0
        sse = 0.0
        for i, (okp, uvp, _) in enumerate(proj):
            if not okp:
                res[2 * i] = PEN
                res[2 * i + 1] = PEN
                sse += 2.0 * PEN * PEN
                continue
            du = float(uvp[0] - uv_list[i][0])
            dv = float(uvp[1] - uv_list[i][1])
            res[2 * i] = du
            res[2 * i + 1] = dv
            sse += du * du + dv * dv

        if progress_logger is not None:
            rmse_px = np.sqrt(sse / max(1, 2 * len(uv_list)))
            progress_logger.callback(
                f"Step4 Cam{target_cid} fixed3D-extrinsic",
                -1.0,
                -1.0,
                float(rmse_px),
                0.5 * float(sse),
            )
        return res

    result = least_squares(
        _residual,
        x0_full,
        method="trf",
        bounds=(b_lo, b_hi) if optimize_plane else (-np.inf, np.inf),
        ftol=1e-6,
        xtol=1e-6,
        gtol=1e-6,
        max_nfev=max_nfev,
        verbose=0,
    )

    cam_x, pt_new, n_new = _decode_state(result.x)
    cam_params[target_cid][:6] = cam_x
    if optimize_plane:
        window_planes[wid]["plane_pt"] = np.asarray(pt_new, dtype=np.float64)
        window_planes[wid]["plane_n"] = np.asarray(n_new, dtype=np.float64)
    _apply_state(result.x)
    print(
        f"[STAGED][Step4] Cam {target_cid}: fixed-3D {'cam+plane' if optimize_plane else 'extrinsic'} done. "
        f"nfev={getattr(result, 'nfev', '?')}, cost={getattr(result, 'cost', float('nan')):.6e}, "
        f"calls={n_calls['k']}"
    )
    return True


def _pnp_init_cam_extrinsic_from_fixed_points(
    target_cid: int,
    cam_params: Dict[int, np.ndarray],
    dataset: Dict,
    fixed_points: Dict[int, Dict[str, np.ndarray]],
) -> bool:
    if target_cid not in cam_params:
        return False

    p = cam_params[target_cid].copy()
    p3d_list = []
    uv_list = []
    for fid in sorted(fixed_points.keys()):
        if fid not in dataset["obsA"] or fid not in dataset["obsB"]:
            continue
        obsA = dataset["obsA"][fid]
        obsB = dataset["obsB"][fid]
        if target_cid not in obsA or target_cid not in obsB:
            continue
        p3d_list.append(np.asarray(fixed_points[fid]["A"], dtype=np.float64))
        uv_list.append(np.asarray(obsA[target_cid], dtype=np.float64))
        p3d_list.append(np.asarray(fixed_points[fid]["B"], dtype=np.float64))
        uv_list.append(np.asarray(obsB[target_cid], dtype=np.float64))

    if len(p3d_list) < 12:
        print(f"[STAGED][Step6][WARN] Cam {target_cid}: insufficient fixed-point matches for PnP ({len(p3d_list)}).")
        return False

    obj_pts = np.asarray(p3d_list, dtype=np.float64).reshape(-1, 1, 3)
    img_pts = np.asarray(uv_list, dtype=np.float64).reshape(-1, 1, 2)
    K = np.array(
        [
            [float(p[6]), 0.0, float(p[7])],
            [0.0, float(p[6]), float(p[8])],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    dist = np.array([float(p[9]), float(p[10]), 0.0, 0.0, 0.0], dtype=np.float64).reshape(-1, 1)

    try:
        ok, rvec_r, tvec_r, inliers = cv2.solvePnPRansac(
            objectPoints=obj_pts,
            imagePoints=img_pts,
            cameraMatrix=K,
            distCoeffs=dist,
            flags=cv2.SOLVEPNP_EPNP,
            reprojectionError=8.0,
            confidence=0.99,
            iterationsCount=200,
        )
        if not ok:
            print(f"[STAGED][Step6][WARN] Cam {target_cid}: solvePnPRansac failed.")
            return False

        ok2, rvec_i, tvec_i = cv2.solvePnP(
            objectPoints=obj_pts,
            imagePoints=img_pts,
            cameraMatrix=K,
            distCoeffs=dist,
            rvec=rvec_r,
            tvec=tvec_r,
            useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok2:
            print(f"[STAGED][Step6][WARN] Cam {target_cid}: solvePnP refine failed.")
            return False

        cam_params[target_cid][:3] = np.asarray(rvec_i, dtype=np.float64).reshape(3)
        cam_params[target_cid][3:6] = np.asarray(tvec_i, dtype=np.float64).reshape(3)
        inl = 0 if inliers is None else int(len(inliers))
        print(f"[STAGED][Step6] Cam {target_cid}: PnP init success (inliers={inl}/{len(p3d_list)}).")
        return True
    except Exception as e:
        print(f"[STAGED][Step6][WARN] Cam {target_cid}: PnP init exception: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run staged refractive wand calibration strategy test")
    parser.add_argument("--csv", type=str, default=r"J:\Fish\T0\R2\wandpoints_filtered2.csv")
    parser.add_argument("--out", type=str, default=r"J:\Fish\T0\R2\StagedStrategyTest")
    parser.add_argument("--wand-length", type=float, default=10.0)
    parser.add_argument("--dist-num", type=int, default=2, choices=[0, 1, 2])
    parser.add_argument("--progress-every", type=int, default=50, help="Print progress every N callback hits")
    parser.add_argument("--plot-best-pair", action="store_true", help="Export 3D plot after best-pair calibration")
    parser.add_argument("--show-plot", action="store_true", help="Show matplotlib interactive window if backend supports GUI")
    parser.add_argument("--plane-init-mode", type=str, default="baseline", choices=["baseline", "candidate-search"], help="Initializer mode for the seed-window plane")
    parser.add_argument("--plane-search-offset-frac", type=float, default=0.20, help="Half-range for plane offset search as a fraction of seed-window scene scale")
    parser.add_argument("--plane-search-tilt-deg", type=float, default=3.0, help="Half-range for plane tilt search in degrees")
    parser.add_argument("--plane-search-offset-steps", type=int, default=2, help="Number of positive offset steps per side; total samples are 2*n+1")
    parser.add_argument("--plane-search-tilt-steps", type=int, default=1, help="Number of positive tilt steps per side; total samples are 2*n+1")
    parser.add_argument("--plane-search-max-frames", type=int, default=80, help="Maximum number of frames to score during candidate search")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    base = WandCalibrator()
    ok, msg = base.load_wand_data_from_csv(args.csv)
    if not ok:
        raise RuntimeError(f"Failed to load wand CSV: {msg}")

    base.wand_length = float(args.wand_length)
    base.dist_coeff_num = int(args.dist_num)
    cam_settings = _camera_settings_default()
    cam_to_window = _cam_to_window_default()
    window_media = _window_media_default()

    observed_cam_ids = sorted(list({cid for _, frame in base.wand_points.items() for cid in frame.keys()}))
    cam_settings, cam_to_window = _try_align_cam_ids(cam_settings, cam_to_window, observed_cam_ids)

    base.camera_settings = cam_settings
    # Must be dict-like entries because run_precalibration_check writes fields into cameras[cid].
    base.cams = {cid: {} for cid in cam_settings.keys()}
    base.cameras = base.cams
    base.active_cam_ids = sorted(cam_settings.keys())
    first_cid = base.active_cam_ids[0]
    base.image_size = (cam_settings[first_cid]["height"], cam_settings[first_cid]["width"])

    refr = RefractiveWandCalibrator(base)
    dataset = refr._collect_observations(cam_to_window)
    observations = refr._prepare_observations_for_bootstrap(cam_to_window)
    prog = ProgressLogger(every=args.progress_every)

    p0_cam_err: Dict[int, float] = {}
    try:
        base.run_precalibration_check(
            wand_length=args.wand_length,
            init_focal_length=cam_settings[first_cid]["focal"],
        )
        p0_cam_err = _calc_p0_cam_error(base)
    except Exception as e:
        print(f"[STAGED][WARN] run_precalibration_check failed, fallback to shared-frame seed selection: {e}")

    bootstrap = PinholeBootstrapP0(PinholeBootstrapP0Config(wand_length_mm=args.wand_length))
    cams_all = dataset["cam_ids"]
    if len(cams_all) < 2:
        raise RuntimeError("Need at least 2 cameras")
    p0_pair = (cams_all[0], cams_all[1])
    cam_params_p0, report = bootstrap.run_all(
        cam_i=p0_pair[0],
        cam_j=p0_pair[1],
        observations=observations,
        camera_settings=cam_settings,
        all_cam_ids=cams_all,
        progress_callback=None,
    )

    cam_params = {}
    for cid, p in cam_params_p0.items():
        f, cx, cy = refr._get_cam_intrinsics(cid)
        cam_params[cid] = np.concatenate([p, [f, cx, cy, 0.0, 0.0]])
    cam_params_p0_snapshot = {cid: v.copy() for cid, v in cam_params.items()}

    points_3d = report.get("points_3d", {})
    XA = {fid: ab[0] for fid, ab in points_3d.items()}
    XB = {fid: ab[1] for fid, ab in points_3d.items()}
    dataset["X_A_bootstrap"] = XA
    dataset["X_B_bootstrap"] = XB

    window_planes = refr._init_window_planes_from_cameras(
        cam_params=cam_params,
        cam_to_window=cam_to_window,
        window_media=window_media,
        err_px=p0_cam_err,
        X_A_list=XA,
        X_B_list=XB,
        active_cam_ids=cams_all,
    )

    if p0_cam_err:
        seed_wid, seed_c1, seed_c2 = _select_seed_pair_same_window(p0_cam_err, cam_to_window, cams_all)
    else:
        seed_wid, seed_c1, seed_c2 = _fallback_seed_pair_by_shared_frames(base.wand_points, cam_to_window, cams_all)
    print(f"[STAGED] Seed window={seed_wid}, seed pair=({seed_c1}, {seed_c2})")

    same_plane_cams = sorted([cid for cid, wid in cam_to_window.items() if wid == seed_wid and cid in cam_params])
    plane_init_debug = {
        "mode": str(args.plane_init_mode),
        "seed_window": int(seed_wid),
        "same_plane_cams": [int(cid) for cid in same_plane_cams],
        "baseline_plane": {
            "plane_pt": np.asarray(window_planes[seed_wid]["plane_pt"], dtype=np.float64).tolist(),
            "plane_n": np.asarray(window_planes[seed_wid]["plane_n"], dtype=np.float64).tolist(),
        },
    }
    if args.plane_init_mode == "candidate-search":
        search_result = _search_seed_window_plane_initializer(
            dataset=dataset,
            cam_params=cam_params,
            cam_to_window=cam_to_window,
            window_media=window_media,
            base_plane=window_planes[seed_wid],
            seed_wid=seed_wid,
            same_plane_cams=same_plane_cams,
            XA=XA,
            XB=XB,
            wand_length=args.wand_length,
            refr=refr,
            offset_frac=args.plane_search_offset_frac,
            tilt_deg=args.plane_search_tilt_deg,
            offset_steps=args.plane_search_offset_steps,
            tilt_steps=args.plane_search_tilt_steps,
            max_frames=args.plane_search_max_frames,
        )
        window_planes[seed_wid] = search_result["best_plane"]
        plane_init_debug.update(
            {
                "search": {
                    "scale_mm": float(search_result["scale_mm"]),
                    "offset_extent_mm": float(search_result["offset_extent_mm"]),
                    "tilt_extent_deg": float(search_result["tilt_extent_deg"]),
                    "frames_evaluated": int(search_result["frames_evaluated"]),
                    "candidate_count": int(search_result["candidate_count"]),
                    "boundary_hit": bool(search_result["boundary_hit"]),
                    "boundary_hit_offset": bool(search_result["boundary_hit_offset"]),
                    "boundary_hit_tilt": bool(search_result["boundary_hit_tilt"]),
                    "best_candidate": dict(search_result["best_candidate"]),
                    "top_candidates": [dict(rec) for rec in search_result["top_candidates"]],
                },
                "selected_plane": {
                    "plane_pt": np.asarray(window_planes[seed_wid]["plane_pt"], dtype=np.float64).tolist(),
                    "plane_n": np.asarray(window_planes[seed_wid]["plane_n"], dtype=np.float64).tolist(),
                },
            }
        )
        print(
            "[STAGED][INIT] Candidate-search selected seed-window plane "
            f"for window {seed_wid}: score={search_result['best_candidate']['score']:.6f}, "
            f"offset={search_result['best_candidate']['offset_mm']:.3f} mm, "
            f"tilt=({search_result['best_candidate']['tilt_a_deg']:.3f}, "
            f"{search_result['best_candidate']['tilt_b_deg']:.3f}) deg, "
            f"boundary_hit={search_result['boundary_hit']}"
        )

    # Step1.5: pre-optimize seed plane with all cameras on that plane (Loop + Joint).
    ds_pre = _filter_dataset_by_cams(dataset, same_plane_cams)
    cp_pre = {cid: cam_params[cid].copy() for cid in same_plane_cams}
    ctw_pre = {cid: cam_to_window[cid] for cid in same_plane_cams}
    wp_pre = {seed_wid: copy.deepcopy(window_planes[seed_wid])}
    wm_pre = {seed_wid: copy.deepcopy(window_media[seed_wid])}
    cams_cpp_pre = refr._init_cams_cpp_in_memory(cp_pre, wm_pre, ctw_pre, wp_pre)
    cfg_pre = RefractiveBAConfig(stage=3, dist_coeff_num=args.dist_num, verbosity=1)
    wp_pre_opt, cp_pre_opt, _ = _run_optimizer(
        ds_pre, cp_pre, cams_cpp_pre, ctw_pre, wm_pre, wp_pre, args.wand_length, cfg_pre, stage=3, progress_cb=prog.callback
    )
    for cid in same_plane_cams:
        cam_params[cid] = cp_pre_opt[cid]
    window_planes[seed_wid] = wp_pre_opt[seed_wid]
    print(f"[STAGED] Step1.5 done: Loop+Joint on plane {seed_wid} with cams={same_plane_cams}")

    seed_cams = [seed_c1, seed_c2]
    ds_seed = _filter_dataset_by_cams(dataset, seed_cams)
    cp_seed = {cid: cam_params[cid].copy() for cid in seed_cams}
    ctw_seed = {cid: cam_to_window[cid] for cid in seed_cams}
    wp_seed = {seed_wid: copy.deepcopy(window_planes[seed_wid])}
    wm_seed = {seed_wid: copy.deepcopy(window_media[seed_wid])}

    cams_cpp_seed = refr._init_cams_cpp_in_memory(cp_seed, wm_seed, ctw_seed, wp_seed)
    # Step 2: optimize best pair with plane fixed (extrinsics only).
    cfg_seed = RefractiveBAConfig(stage=1, dist_coeff_num=args.dist_num, verbosity=1)
    wp_seed_opt, cp_seed_opt, _ = _run_best_pair_fixed_plane(
        ds_seed, cp_seed, cams_cpp_seed, ctw_seed, wm_seed, wp_seed, args.wand_length, cfg_seed, progress_cb=prog.callback
    )
    for cid, p in cp_seed_opt.items():
        cam_params[cid] = p
    window_planes[seed_wid] = wp_seed_opt[seed_wid]

    ctw_global = dict(cam_to_window)
    cams_cpp_global = refr._init_cams_cpp_in_memory(cam_params, window_media, ctw_global, window_planes)
    seed_pts = _triangulate_seed_points(dataset, cams_cpp_global, (seed_c1, seed_c2))
    print(f"[STAGED] Seed triangulated frames: {len(seed_pts)}")
    if args.plot_best_pair:
        best_pair_state = {cid: cam_params[cid].copy() for cid in seed_cams}
        plot_path = out_dir / "best_pair_diagnostic_3d.png"
        _plot_best_pair_diagnostics(
            out_png=plot_path,
            seed_wid=seed_wid,
            seed_pair=(seed_c1, seed_c2),
            seed_points=seed_pts,
            window_plane=window_planes[seed_wid],
            p0_cam_params=cam_params_p0_snapshot,
            best_pair_cam_params=best_pair_state,
            same_plane_cam_ids=same_plane_cams,
            show_plot=bool(args.show_plot),
        )

    seed_window_cams = sorted([cid for cid, wid in cam_to_window.items() if wid == seed_wid and cid in cam_params])
    remain_seed_cams = [c for c in seed_window_cams if c not in seed_cams]
    for tcid in remain_seed_cams:
        # Step4: fixed 3D points + fixed plane; optimize only this camera extrinsic.
        _optimize_single_cam_extrinsic_with_fixed_points(
            target_cid=tcid,
            cam_params=cam_params,
            cams_cpp=cams_cpp_global,
            cam_to_window=ctw_global,
            window_media=window_media,
            window_planes=window_planes,
            dataset=dataset,
            fixed_points=seed_pts,
            progress_logger=prog,
            max_nfev=80,
        )
        print(f"[STAGED] Step5 done for cam {tcid}")

    other_wids = sorted([wid for wid in window_media.keys() if wid != seed_wid])
    for wid in other_wids:
        target_cams = sorted([cid for cid, w in cam_to_window.items() if w == wid and cid in cam_params])
        if not target_cams:
            continue

        # Degenerate case: one-camera plane cannot robustly run refractive wand BA Loop+Joint.
        # Fallback to plate-style single-camera extrinsic calibration with fixed plane/fixed 3D.
        if len(target_cams) == 1:
            tcid = target_cams[0]
            _optimize_single_cam_extrinsic_with_fixed_points(
                target_cid=tcid,
                cam_params=cam_params,
                cams_cpp=cams_cpp_global,
                cam_to_window=ctw_global,
                window_media=window_media,
                window_planes=window_planes,
                dataset=dataset,
                fixed_points=seed_pts,
                progress_logger=prog,
                max_nfev=100,
                optimize_plane=True,
            )
            print(f"[STAGED] Step6(single-cam plane) done for window {wid}, cam={tcid}")
            continue

        # Step6 strict mode: only target-plane cameras (no seed cameras mixed in).
        sub_cams = sorted(set(target_cams))
        ds_sub = _filter_dataset_by_cams(dataset, sub_cams)
        cp_sub = {cid: cam_params[cid].copy() for cid in sub_cams}
        ctw_sub = {cid: cam_to_window[cid] for cid in sub_cams}
        wp_sub = {wid: copy.deepcopy(window_planes[wid])}
        wm_sub = {wid: copy.deepcopy(window_media[wid])}

        # Step6 pre-init: update target-window cameras with pinhole PnP from fixed 3D points.
        for tcid in target_cams:
            _pnp_init_cam_extrinsic_from_fixed_points(
                target_cid=tcid,
                cam_params=cp_sub,
                dataset=dataset,
                fixed_points=seed_pts,
            )

        cams_cpp_sub = refr._init_cams_cpp_in_memory(cp_sub, wm_sub, ctw_sub, wp_sub)
        cfg_w = RefractiveBAConfig(stage=3, dist_coeff_num=args.dist_num, verbosity=1)
        wp_w, cp_w, _ = _run_optimizer(
            ds_sub, cp_sub, cams_cpp_sub, ctw_sub, wm_sub, wp_sub, args.wand_length, cfg_w, stage=3, progress_cb=prog.callback
        )
        for cid in target_cams:
            cam_params[cid] = cp_w[cid]
        window_planes[wid] = wp_w[wid]
        print(f"[STAGED] Step6 done for window {wid}, cams={target_cams}")

    ds_all = _filter_dataset_by_cams(dataset, sorted(cam_params.keys()))
    cams_cpp_all = refr._init_cams_cpp_in_memory(cam_params, window_media, ctw_global, window_planes)
    cfg_final = RefractiveBAConfig(stage=4, dist_coeff_num=args.dist_num, verbosity=1)
    wp_final, cp_final, wm_final = _run_final_refined_only(
        ds_all, cam_params, cams_cpp_all, ctw_global, window_media, window_planes, args.wand_length, cfg_final, progress_cb=prog.callback
    )
    cam_params = cp_final
    window_planes = wp_final
    window_media = wm_final
    final_metrics = _compute_final_metrics(
        dataset=ds_all,
        cam_params=cam_params,
        cams_cpp=refr._init_cams_cpp_in_memory(cam_params, window_media, ctw_global, window_planes),
        cam_to_window=ctw_global,
        window_media=window_media,
        window_planes=window_planes,
        wand_length=args.wand_length,
        cfg=cfg_final,
    )

    cam_dir = out_dir / "camFile_staged"
    cam_dir.mkdir(parents=True, exist_ok=True)
    refr.export_camfile_with_refraction(
        out_dir=str(cam_dir),
        cam_params=cam_params,
        window_media=window_media,
        cam_to_window=ctw_global,
        window_planes=window_planes,
    )

    summary = {
        "csv": str(args.csv),
        "seed_window": int(seed_wid),
        "seed_pair": [int(seed_c1), int(seed_c2)],
        "p0_cam_error_px": {str(k): float(v) for k, v in sorted(p0_cam_err.items())},
        "seed_triangulated_frames": int(len(seed_pts)),
        "plane_initializer": plane_init_debug,
        "final_metrics": final_metrics,
        "cam_ids": [int(c) for c in sorted(cam_params.keys())],
        "window_ids": [int(w) for w in sorted(window_planes.keys())],
        "cam_out_dir": str(cam_dir),
        "run_env": "OpenLPT",
    }
    summary_path = out_dir / "staged_strategy_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[STAGED] Done. Summary: {summary_path}")


if __name__ == "__main__":
    main()
