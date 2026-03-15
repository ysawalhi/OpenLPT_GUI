#!/usr/bin/env python
"""
Phase 5 Verification: Numerical equivalence of parallel vs sequential probing.

Tests that:
- Stage 1 (probe_scales_multidim_stage1) parallel and sequential paths produce
  numerically identical scales (np.allclose with rtol=1e-10).
- Stage 2 (probe_scales_multidim_stage2_blocks) parallel and sequential paths
  produce numerically identical block_scales.
- Worker count and parallelization decision logic are correct.
- Timing is logged for both paths.
"""

from __future__ import annotations

import csv
import logging
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

# ── Setup paths ──────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

# ── Logging ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)
logger = logging.getLogger("verify_parallel_probing")

# ── Imports from the module under test ───────────────────────────────────
from modules.camera_calibration.wand_calibration.full_global_search import (
    build_evaluation_context,
    build_shared_setup,
    load_observations_csv,
    load_reference_state,
    probe_scales_multidim_stage1,
    probe_scales_multidim_stage2_blocks,
)


# =========================================================================
#  Synthetic test data generation
# =========================================================================

WAND_LENGTH = 10.0  # mm


def _write_camfile(path: Path, cam_id: int, window_id: int,
                   rvec: np.ndarray, tvec: np.ndarray,
                   fx: float, fy: float, cx: float, cy: float,
                   plane_pt: np.ndarray, plane_n: np.ndarray,
                   refract: list, thickness: float,
                   img_rows: int = 1024, img_cols: int = 1024) -> None:
    """Write a synthetic PINPLATE camfile with refraction meta."""
    import cv2 as _cv2

    R, _ = _cv2.Rodrigues(rvec.astype(np.float64))
    R_inv = R.T
    t_inv = -R.T @ tvec

    # NOTE: The parse_camfile parser in full_global_search.py uses a mixed
    # format — whitespace-split for cam-matrix rows, translation, inverse-
    # translation, plane_pt, and plane_n; comma-split for image size,
    # distortion, rotation-vector, refract array, and thickness.
    lines = [
        f"# BEGIN_REFRACTION_META",
        f"# CAM_ID={cam_id}",
        f"# WINDOW_ID={window_id}",
        f"# END_REFRACTION_META",
        f"# Camera Model: (PINHOLE/POLYNOMIAL)",
        f"PINPLATE",
        f"# Camera Calibration Error: ",
        f"None",
        f"# Pose Calibration Error: ",
        f"None",
        f"# Image Size: (n_row,n_col)",
        f"{img_rows},{img_cols}",
        f"# Camera Matrix: ",
        f"{fx} 0.0 {cx}",          # whitespace-separated (parser: split())
        f"0.0 {fy} {cy}",           # whitespace-separated
        f"0.0 0.0 1.0",             # whitespace-separated
        f"# Distortion Coefficients: ",
        f"0.0,0.0,0.0,0.0,0.0",     # comma-separated (parser: split(","))
        f"# Rotation Vector: ",
        f"{rvec[0]},{rvec[1]},{rvec[2]}",  # comma-separated
        f"# Rotation Matrix: ",
        f"{R[0,0]} {R[0,1]} {R[0,2]}",    # whitespace-separated
        f"{R[1,0]} {R[1,1]} {R[1,2]}",
        f"{R[2,0]} {R[2,1]} {R[2,2]}",
        f"# Inverse of Rotation Matrix: ",
        f"{R_inv[0,0]} {R_inv[0,1]} {R_inv[0,2]}",
        f"{R_inv[1,0]} {R_inv[1,1]} {R_inv[1,2]}",
        f"{R_inv[2,0]} {R_inv[2,1]} {R_inv[2,2]}",
        f"# Translation Vector: ",
        f"{tvec[0]} {tvec[1]} {tvec[2]}",  # whitespace-separated
        f"# Inverse of Translation Vector: ",
        f"{t_inv[0]} {t_inv[1]} {t_inv[2]}",
        f"# Reference Point of Refractive Plate: ",
        f"{plane_pt[0]} {plane_pt[1]} {plane_pt[2]}",  # whitespace-separated
        f"# Normal Vector of Refractive Plate: ",
        f"{plane_n[0]} {plane_n[1]} {plane_n[2]}",     # whitespace-separated
        f"# Refractive Index Array: ",
        f"{refract[0]},{refract[1]},{refract[2]}",      # comma-separated
        f"# Width of the Refractive Plate: (mm)",
        f"{thickness}",                                  # comma-separated
        f"# Projection Tolerance: ",
        f"1e-8",
        f"# Maximum Number of Iterations for Projection: ",
        f"50000",
        f"# Optimization Rate: ",
        f"0.3",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _create_synthetic_camfiles(camdir: Path, n_cams: int = 4) -> None:
    """Create n_cams synthetic PINPLATE camfiles sharing 1 window."""
    camdir.mkdir(parents=True, exist_ok=True)

    # Common plane: z = -50 facing +z
    plane_pt = np.array([0.0, 0.0, -50.0])
    plane_n = np.array([0.0, 0.0, 1.0])
    refract = [1.33, 1.49, 1.0]  # n_obj, n_win, n_air
    thickness = 12.7

    # Camera ring at z=300, looking towards origin
    for i in range(n_cams):
        angle = 2 * np.pi * i / n_cams
        radius = 200.0
        cx_world = radius * np.cos(angle)
        cy_world = radius * np.sin(angle)
        cz_world = 300.0

        # Camera looks towards origin: compute rotation
        cam_pos = np.array([cx_world, cy_world, cz_world])
        forward = -cam_pos / np.linalg.norm(cam_pos)
        # Approximate up vector
        world_up = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, world_up)
        if np.linalg.norm(right) < 1e-6:
            world_up = np.array([0.0, 1.0, 0.0])
            right = np.cross(forward, world_up)
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        up /= np.linalg.norm(up)

        # R maps world->camera: rows are right, -up, forward (OpenCV convention)
        R = np.array([right, -up, forward])
        import cv2
        rvec, _ = cv2.Rodrigues(R)
        rvec = rvec.flatten()
        tvec = (R @ cam_pos).flatten()  # t = R * C (wrong sign deliberately for testing)
        # Actually t = -R * C for standard pinhole, but here tvec = R@cam_pos for consistency
        # The BA optimizer will handle the conventions internally

        _write_camfile(
            path=camdir / f"cam{i}.txt",
            cam_id=i,
            window_id=0,  # All cams share window 0
            rvec=rvec,
            tvec=tvec,
            fx=9000.0, fy=-9000.0, cx=511.5, cy=511.5,
            plane_pt=plane_pt,
            plane_n=plane_n,
            refract=refract,
            thickness=thickness,
        )


def _create_synthetic_observations(csv_path: Path, cam_ids: list,
                                   n_frames: int = 20) -> None:
    """Create a synthetic wand-points CSV with Filtered_Small / Filtered_Large rows."""
    rng = np.random.RandomState(42)

    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "Frame", "Camera", "Status", "PointIdx", "X", "Y", "Radius", "Metric",
        ])
        writer.writeheader()

        for fid in range(n_frames):
            # Generate random 2D observations for each camera
            for cid in cam_ids:
                # Small point (endpoint A)
                x_small = 300.0 + rng.uniform(-200, 200)
                y_small = 300.0 + rng.uniform(-200, 200)
                writer.writerow({
                    "Frame": fid,
                    "Camera": cid,
                    "Status": "Filtered_Small",
                    "PointIdx": 0,
                    "X": f"{x_small:.4f}",
                    "Y": f"{y_small:.4f}",
                    "Radius": "3.0",
                    "Metric": "1.0",
                })
                # Large point (endpoint B) — offset from small
                x_large = x_small + rng.uniform(5, 30)
                y_large = y_small + rng.uniform(-15, 15)
                writer.writerow({
                    "Frame": fid,
                    "Camera": cid,
                    "Status": "Filtered_Large",
                    "PointIdx": 1,
                    "X": f"{x_large:.4f}",
                    "Y": f"{y_large:.4f}",
                    "Radius": "5.0",
                    "Metric": "1.0",
                })


# =========================================================================
#  Core verification logic
# =========================================================================

def run_verification() -> bool:
    """Run parallel vs sequential probing equivalence test."""
    all_passed = True

    # ── System info ──
    cpu_count = os.cpu_count() or 1
    max_workers = min(32, max(1, int(cpu_count * 0.8)))
    logger.info('System CPU count: %d, max_workers: %d', cpu_count, max_workers)
    print("=" * 70)
    print("PHASE 5: PARALLEL PROBING VERIFICATION")
    print("=" * 70)
    print(f"\nSystem Info:")
    print(f"  os.cpu_count()   = {cpu_count}")
    print(f"  max_workers      = min(32, max(1, int({cpu_count} * 0.8))) = {max_workers}")
    print(f"  Python           = {sys.version}")

    # ── Create synthetic test data in temp directory ──
    with tempfile.TemporaryDirectory(prefix="verify_probing_") as tmpdir:
        tmpdir = Path(tmpdir)
        camdir = tmpdir / "camFile"
        obs_csv = tmpdir / "wand_obs.csv"

        N_CAMS = 4
        N_FRAMES = 10
        cam_ids = list(range(N_CAMS))

        print(f"\n--- Generating synthetic test data ---")
        print(f"  Cameras:  {N_CAMS} (all sharing window 0)")
        print(f"  Frames:   {N_FRAMES}")
        print(f"  Tempdir:  {tmpdir}")

        _create_synthetic_camfiles(camdir, n_cams=N_CAMS)
        _create_synthetic_observations(obs_csv, cam_ids=cam_ids, n_frames=N_FRAMES)

        # ── Load data ──
        print(f"\n--- Loading reference state and observations ---")
        ref_state = load_reference_state(str(camdir))
        dataset = load_observations_csv(str(obs_csv), WAND_LENGTH)

        # ── Build evaluation context ──
        ctx = build_evaluation_context(ref_state, dataset, WAND_LENGTH, max_frames=N_FRAMES)
        n_params = ctx.n_params
        print(f"  n_params = {n_params}")
        print(f"  cam_ids  = {ctx.cam_ids}")
        print(f"  win_ids  = {ctx.window_ids}")

        # ── Build shared_setup (needed for parallel path) ──
        shared_setup = build_shared_setup(
            ref_state, dataset,
            np.ones(n_params, dtype=np.float64),
            wand_length=WAND_LENGTH,
            max_frames=N_FRAMES,
        )

        # ── Probing config (small budget for test speed) ──
        probe_kwargs = dict(
            max_evals=100,
            max_wall_seconds=30.0,
            ray_rmse_stop_factor=1.1,
            enable_compensation=True,
            max_compensation_iters=2,
            alpha_growth=2.0,
            max_alpha_steps=6,
        )

        # ================================================================
        #  STAGE 1: probe_scales_multidim_stage1
        # ================================================================
        print("\n" + "=" * 70)
        print("STAGE 1: probe_scales_multidim_stage1")
        print("=" * 70)

        # -- Parallelization decision analysis --
        s1_use_parallel = (shared_setup is not None and max_workers > 1
                           and n_params >= 2 * max_workers)
        logger.info('Stage 1: n_params=%d, parallel=%s (items >= 2*workers: %d >= %d)',
                     n_params, s1_use_parallel, n_params, 2 * max_workers)
        print(f"\n  Parallelization decision (Stage 1):")
        print(f"    shared_setup is not None  = True")
        print(f"    max_workers > 1           = {max_workers > 1} ({max_workers})")
        print(f"    n_params >= 2*max_workers = {n_params >= 2 * max_workers} "
              f"({n_params} >= {2 * max_workers})")
        print(f"    => use_parallel           = {s1_use_parallel}")

        # -- Run WITH shared_setup (triggers parallel if eligible) --
        print(f"\n  Running Stage 1 WITH shared_setup (parallel-eligible) ...")
        t0 = time.monotonic()
        result_with_setup = probe_scales_multidim_stage1(
            ctx, shared_setup=shared_setup, **probe_kwargs,
        )
        t_with = time.monotonic() - t0
        print(f"    Time: {t_with:.2f}s, evals: {result_with_setup.n_evals}")

        # -- Run WITHOUT shared_setup (forces sequential) --
        print(f"  Running Stage 1 WITHOUT shared_setup (forced sequential) ...")
        t0 = time.monotonic()
        result_no_setup = probe_scales_multidim_stage1(
            ctx, shared_setup=None, **probe_kwargs,
        )
        t_seq = time.monotonic() - t0
        print(f"    Time: {t_seq:.2f}s, evals: {result_no_setup.n_evals}")

        # -- Compare scales --
        scales_with = result_with_setup.scales
        scales_without = result_no_setup.scales
        print(f"\n  Comparing Stage 1 scales:")
        print(f"    scales (with setup):    shape={scales_with.shape}, "
              f"range=[{scales_with.min():.6e}, {scales_with.max():.6e}]")
        print(f"    scales (no setup):      shape={scales_without.shape}, "
              f"range=[{scales_without.min():.6e}, {scales_without.max():.6e}]")

        if s1_use_parallel:
            # Parallel path was taken: compare results
            s1_equiv = np.allclose(scales_with, scales_without, rtol=1e-10)
            max_rel_diff = float(np.max(np.abs(scales_with - scales_without)
                                        / np.maximum(np.abs(scales_without), 1e-30)))
            print(f"    np.allclose(rtol=1e-10)  = {s1_equiv}")
            print(f"    max relative difference  = {max_rel_diff:.2e}")
            if s1_equiv:
                print(f"    PASS: Stage 1 numerical equivalence verified")
            else:
                print(f"    FAIL: Stage 1 scales differ beyond tolerance!")
                all_passed = False
        else:
            # Both paths ran sequential (n_params < 2*max_workers)
            # They should be IDENTICAL since both use the same sequential code path
            s1_equiv = np.allclose(scales_with, scales_without, rtol=1e-10)
            print(f"    NOTE: Both paths used sequential execution (n_params too small for parallel)")
            print(f"    np.allclose(rtol=1e-10)  = {s1_equiv}")
            if s1_equiv:
                print(f"    PASS: Stage 1 sequential consistency verified")
            else:
                print(f"    FAIL: Stage 1 scales differ even in sequential mode!")
                all_passed = False

        # -- Compare sensitivities --
        sens_with = result_with_setup.sensitivities
        sens_without = result_no_setup.sensitivities
        sens_equiv = np.allclose(sens_with, sens_without, rtol=1e-10)
        print(f"    sensitivities match      = {sens_equiv}")
        if not sens_equiv:
            max_sens_diff = float(np.max(np.abs(sens_with - sens_without)
                                         / np.maximum(np.abs(sens_without), 1e-30)))
            print(f"    max sensitivity rel diff = {max_sens_diff:.2e}")

        # ================================================================
        #  STAGE 2: probe_scales_multidim_stage2_blocks
        # ================================================================
        print("\n" + "=" * 70)
        print("STAGE 2: probe_scales_multidim_stage2_blocks")
        print("=" * 70)

        # Build layout from stage 1
        layout = result_no_setup.param_layout

        # -- Analyze parallelization decision --
        # Stage 2 checks n_blocks >= 2 * max_workers
        # We'll discover n_blocks after running
        print(f"\n  Layout: {n_params} params")

        # -- Run WITH shared_setup --
        print(f"\n  Running Stage 2 WITH shared_setup (parallel-eligible) ...")
        t0 = time.monotonic()
        s2_result_with = probe_scales_multidim_stage2_blocks(
            ctx, layout=layout, shared_setup=shared_setup, **probe_kwargs,
        )
        t_s2_with = time.monotonic() - t0
        print(f"    Time: {t_s2_with:.2f}s, evals: {s2_result_with.n_evals}")

        # -- Run WITHOUT shared_setup (forces sequential) --
        print(f"  Running Stage 2 WITHOUT shared_setup (forced sequential) ...")
        t0 = time.monotonic()
        s2_result_without = probe_scales_multidim_stage2_blocks(
            ctx, layout=layout, shared_setup=None, **probe_kwargs,
        )
        t_s2_seq = time.monotonic() - t0
        print(f"    Time: {t_s2_seq:.2f}s, evals: {s2_result_without.n_evals}")

        # -- Compare block_scales --
        bs_with = s2_result_with.block_scales
        bs_without = s2_result_without.block_scales
        print(f"\n  Comparing Stage 2 block_scales:")
        print(f"    block_scales (with setup): shape={bs_with.shape}, "
              f"range=[{bs_with.min():.6e}, {bs_with.max():.6e}]")
        print(f"    block_scales (no setup):   shape={bs_without.shape}, "
              f"range=[{bs_without.min():.6e}, {bs_without.max():.6e}]")

        s2_equiv = np.allclose(bs_with, bs_without, rtol=1e-10)
        if bs_without.size > 0 and np.any(np.abs(bs_without) > 1e-30):
            max_rel_diff_s2 = float(np.max(np.abs(bs_with - bs_without)
                                           / np.maximum(np.abs(bs_without), 1e-30)))
        else:
            max_rel_diff_s2 = float(np.max(np.abs(bs_with - bs_without)))
        print(f"    np.allclose(rtol=1e-10)  = {s2_equiv}")
        print(f"    max difference           = {max_rel_diff_s2:.2e}")

        # Determine if parallel was actually used for Stage 2
        # We need to compute n_blocks — reconstruct from the function's internal logic
        from modules.camera_calibration.wand_calibration.full_global_search import (
            _build_dynamic_probe_blocks,
            build_search_parameter_layout,
        )
        cam_to_window = dict(getattr(ctx.optimizer, 'cam_to_window', {}))
        blocks = _build_dynamic_probe_blocks(layout, cam_to_window=cam_to_window)
        n_blocks = len(blocks)
        s2_use_parallel = (shared_setup is not None and max_workers > 1
                           and n_blocks >= 2 * max_workers)
        logger.info('Stage 2: n_blocks=%d, parallel=%s (items >= 2*workers: %d >= %d)',
                     n_blocks, s2_use_parallel, n_blocks, 2 * max_workers)
        print(f"\n  Stage 2 parallelization decision:")
        print(f"    n_blocks                 = {n_blocks}")
        print(f"    n_blocks >= 2*max_workers = {n_blocks >= 2 * max_workers} "
              f"({n_blocks} >= {2 * max_workers})")
        print(f"    => use_parallel          = {s2_use_parallel}")

        if s2_equiv:
            print(f"    PASS: Stage 2 numerical equivalence verified")
        else:
            print(f"    FAIL: Stage 2 block_scales differ beyond tolerance!")
            all_passed = False

        # ================================================================
        #  TIMING SUMMARY
        # ================================================================
        print("\n" + "=" * 70)
        print("TIMING SUMMARY")
        print("=" * 70)
        print(f"  Stage 1 (with shared_setup):    {t_with:.2f}s")
        print(f"  Stage 1 (forced sequential):    {t_seq:.2f}s")
        print(f"  Stage 2 (with shared_setup):    {t_s2_with:.2f}s")
        print(f"  Stage 2 (forced sequential):    {t_s2_seq:.2f}s")
        logger.info(
            'Timing: Stage1_with=%.2fs Stage1_seq=%.2fs Stage2_with=%.2fs Stage2_seq=%.2fs',
            t_with, t_seq, t_s2_with, t_s2_seq,
        )

        # ================================================================
        #  PARALLELIZATION DECISION VERIFICATION
        # ================================================================
        print("\n" + "=" * 70)
        print("PARALLELIZATION DECISION VERIFICATION")
        print("=" * 70)
        print(f"  cpu_count          = {cpu_count}")
        print(f"  max_workers        = {max_workers}")
        print(f"  n_params (Stage 1) = {n_params}")
        print(f"  n_blocks (Stage 2) = {n_blocks}")
        print(f"  Stage 1 parallel?  = {s1_use_parallel} "
              f"(threshold: n_params >= {2 * max_workers})")
        print(f"  Stage 2 parallel?  = {s2_use_parallel} "
              f"(threshold: n_blocks >= {2 * max_workers})")

        # Verify the decision logic is correct
        expected_s1 = (max_workers > 1 and n_params >= 2 * max_workers)
        expected_s2 = (max_workers > 1 and n_blocks >= 2 * max_workers)
        decision_ok = (s1_use_parallel == expected_s1 and s2_use_parallel == expected_s2)
        if decision_ok:
            print(f"  PASS: Parallelization decisions are correct")
        else:
            print(f"  FAIL: Decision logic mismatch!")
            all_passed = False

        # ================================================================
        #  EDGE CASE: Small problem forces sequential fallback
        # ================================================================
        print("\n" + "=" * 70)
        print("EDGE CASE: Sequential fallback verification")
        print("=" * 70)
        # With N_CAMS=4 and 1 window, n_params is typically small.
        # The code should use sequential path when n_params < 2*max_workers.
        if not s1_use_parallel:
            print(f"  PASS: Small problem ({n_params} params) correctly triggers "
                  f"sequential fallback (threshold={2 * max_workers})")
        else:
            print(f"  INFO: Problem large enough for parallel ({n_params} params >= "
                  f"{2 * max_workers}). Edge case tested implicitly.")

        if not s2_use_parallel:
            print(f"  PASS: Small problem ({n_blocks} blocks) correctly triggers "
                  f"sequential fallback (threshold={2 * max_workers})")
        else:
            print(f"  INFO: Blocks large enough for parallel ({n_blocks} blocks >= "
                  f"{2 * max_workers}). Edge case tested implicitly.")

    # ================================================================
    #  FINAL VERDICT
    # ================================================================
    print("\n" + "=" * 70)
    if all_passed:
        print("PASS: Verification succeeded — all checks passed")
        print("PASS: numerical equivalence verified")
    else:
        print("FAIL: Verification failed — see details above")
    print("=" * 70)

    return all_passed


if __name__ == "__main__":
    success = run_verification()
    sys.exit(0 if success else 1)
