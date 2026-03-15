#!/usr/bin/env python
"""
Unit tests for wand calibrator fixes — BA optimizer infrastructure.

Tests cover:
- C++ mocking infrastructure (patching build_pinplate_rays_cpp_batch, update_cpp_camera_state)
- Synthetic data factory functions for reproducible unit tests
- Helper utilities: radius validation, optimization result checking,
  robust plane init, and fully-mocked optimizer construction.

Run:
    conda run -n OpenLPT python -m pytest modules/camera_calibration/wand_calibration/test_wand_calibrator_fixes.py -v
"""

from __future__ import annotations

import unittest
from unittest.mock import patch, MagicMock
from typing import Dict, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Patch targets (full dotted path where the names are looked up at runtime)
# ---------------------------------------------------------------------------
_PATCH_BUILD_RAYS = (
    "modules.camera_calibration.wand_calibration"
    ".refraction_calibration_BA.build_pinplate_rays_cpp_batch"
)
_PATCH_UPDATE_CPP = (
    "modules.camera_calibration.wand_calibration"
    ".refraction_calibration_BA.update_cpp_camera_state"
)


# ===================================================================
# Synthetic data factory functions
# ===================================================================

def make_minimal_dataset(n_frames: int = 5, cam_ids: Tuple[int, ...] = (0, 1)) -> dict:
    """Create a minimal dataset dict suitable for ``RefractiveBAOptimizer``.

    Returns a dict with keys: obsA, obsB, frames, est_radius_small_mm,
    est_radius_large_mm.  Observations are random pixel coords so the
    constructor's ``ObsCacheBuilder.build`` populates a non-empty cache.
    """
    rng = np.random.RandomState(42)
    frames = list(range(n_frames))
    obsA: Dict[int, Dict[int, np.ndarray]] = {}
    obsB: Dict[int, Dict[int, np.ndarray]] = {}
    for fid in frames:
        obsA[fid] = {}
        obsB[fid] = {}
        for cid in cam_ids:
            obsA[fid][cid] = rng.uniform(100, 900, size=(2,)).astype(np.float64)
            obsB[fid][cid] = rng.uniform(100, 900, size=(2,)).astype(np.float64)
    return {
        "obsA": obsA,
        "obsB": obsB,
        "frames": frames,
        "est_radius_small_mm": 1.5,
        "est_radius_large_mm": 2.0,
    }


def make_cam_params(cam_ids: Tuple[int, ...] = (0, 1), f: float = 1000.0) -> Dict[int, np.ndarray]:
    """Return cam_params: ``{cam_id: (11,) float64}`` with focal length at index 6.

    Layout: [rvec(3), tvec(3), f, cx, cy, k1, k2]
    """
    params: Dict[int, np.ndarray] = {}
    for cid in cam_ids:
        p = np.zeros(11, dtype=np.float64)
        p[6] = f       # focal length
        p[7] = 512.0   # cx
        p[8] = 512.0   # cy
        params[cid] = p
    return params


def make_window_planes() -> Dict[int, dict]:
    """Return window planes: ``{window_id: {'plane_pt': (3,), 'plane_n': (3,)}}``."""
    return {
        0: {
            "plane_pt": np.array([0.0, 0.0, 500.0], dtype=np.float64),
            "plane_n": np.array([0.0, 0.0, 1.0], dtype=np.float64),
        }
    }


def make_window_media() -> Dict[int, dict]:
    """Return window media: ``{window_id: {'n1': float, 'n2': float, 'n3': float, 'thickness': float}}``."""
    return {
        0: {
            "n1": 1.0,
            "n2": 1.5,
            "n3": 1.333,
            "thickness": 10.0,
        }
    }


# ===================================================================
# Helper utilities
# ===================================================================

def _validate_radii(radius_A: float, radius_B: float, label: str = "radii") -> None:
    """Validate that *radius_A* and *radius_B* are finite, positive, in
    [0.1, 50.0] mm, and properly ordered (B >= A).

    Raises ``AssertionError`` with a descriptive message on failure.
    Uses ``np.isfinite`` (NOT ``is not None``).
    """
    for name, val in [("radius_A", radius_A), ("radius_B", radius_B)]:
        if not np.isfinite(val):
            raise AssertionError(
                f"[{label}] {name}={val} is not finite"
            )
        if val <= 0:
            raise AssertionError(
                f"[{label}] {name}={val} must be positive"
            )
        if val < 0.1 or val > 50.0:
            raise AssertionError(
                f"[{label}] {name}={val} out of valid range [0.1, 50.0] mm"
            )
    if radius_B < radius_A:
        raise AssertionError(
            f"[{label}] radius_B ({radius_B}) must be >= radius_A ({radius_A})"
        )


def check_optimization_result(
    res, mode: str
) -> Tuple[bool, str, str]:
    """Interpret a ``scipy.optimize.OptimizeResult``-like object.

    Returns:
        (use_result, severity, msg) where
        - status >= 1  → (True, 'info', ...)
        - status == 0  → (True, 'warning', ...)    budget exhausted but partial progress valid
        - status == -1 → (False, 'error', ...)
    """
    status = getattr(res, "status", getattr(res, "get", lambda k, d: d)("status", -999))
    message = getattr(res, "message", getattr(res, "get", lambda k, d: d)("message", "unknown"))

    if status >= 1:
        return (True, "info", f"[{mode}] converged (status={status}): {message}")
    elif status == 0:
        return (True, "warning", f"[{mode}] budget exhausted (status=0): {message}")
    else:
        return (False, "error", f"[{mode}] failed (status={status}): {message}")


def build_optimizer_for_test() -> "RefractiveBAOptimizer":
    """Construct a fully-mocked ``RefractiveBAOptimizer`` instance.

    C++ camera objects are ``MagicMock`` stubs with
    ``projectBatchStatus.return_value = []``.  The two heavy C++ helpers
    (``build_pinplate_rays_cpp_batch`` and ``update_cpp_camera_state``) are
    patched out so no native library is needed.
    """
    from modules.camera_calibration.wand_calibration.refraction_calibration_BA import (
        RefractiveBAOptimizer,
        RefractiveBAConfig,
    )

    cam_ids = (0, 1)
    dataset = make_minimal_dataset(n_frames=5, cam_ids=cam_ids)
    cam_params = make_cam_params(cam_ids=cam_ids, f=1000.0)
    window_planes = make_window_planes()
    window_media = make_window_media()

    # Mock C++ camera objects
    cam0 = MagicMock(name="cam_cpp_0")
    cam0.projectBatchStatus.return_value = []
    cam1 = MagicMock(name="cam_cpp_1")
    cam1.projectBatchStatus.return_value = []
    cams_cpp = {0: cam0, 1: cam1}

    cam_to_window = {0: 0, 1: 0}
    wand_length = 10.0  # mm

    config = RefractiveBAConfig(verbosity=0)

    with patch(_PATCH_BUILD_RAYS, return_value=[]), \
         patch(_PATCH_UPDATE_CPP, return_value=None):
        optimizer = RefractiveBAOptimizer(
            dataset=dataset,
            cam_params=cam_params,
            cams_cpp=cams_cpp,
            cam_to_window=cam_to_window,
            window_media=window_media,
            window_planes=window_planes,
            wand_length=wand_length,
            config=config,
            progress_callback=None,
        )
    return optimizer


def robust_plane_point_init(midpoints: np.ndarray) -> np.ndarray:
    """Compute a robust initial plane point from a set of 3-D midpoints.

    Uses the coordinate-wise median (resistant to outliers) rather than the
    mean.

    Args:
        midpoints: ``(N, 3)`` array of 3-D points.

    Returns:
        ``(3,)`` float64 ndarray — the median-based plane point.
    """
    pts = np.asarray(midpoints, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)
    return np.median(pts, axis=0).astype(np.float64)


# ===================================================================
# Test class
# ===================================================================

@patch(_PATCH_UPDATE_CPP, return_value=None)
@patch(_PATCH_BUILD_RAYS, return_value=[])
class TestRefractiveBAOptimizerUnit(unittest.TestCase):
    """Unit tests for RefractiveBAOptimizer with C++ calls mocked out."""

    def test_constructor_initializes_state(
        self, mock_build_rays: MagicMock, mock_update_cpp: MagicMock
    ) -> None:
        """Verify the optimizer initializes all expected attributes."""
        from modules.camera_calibration.wand_calibration.refraction_calibration_BA import (
            RefractiveBAOptimizer,
            RefractiveBAConfig,
        )

        cam_ids = (0, 1)
        dataset = make_minimal_dataset(n_frames=5, cam_ids=cam_ids)
        cam_params = make_cam_params(cam_ids=cam_ids, f=1000.0)
        window_planes = make_window_planes()
        window_media = make_window_media()

        cam0 = MagicMock(name="cam_cpp_0")
        cam0.projectBatchStatus.return_value = []
        cam1 = MagicMock(name="cam_cpp_1")
        cam1.projectBatchStatus.return_value = []
        cams_cpp = {0: cam0, 1: cam1}
        cam_to_window = {0: 0, 1: 0}
        wand_length = 10.0

        config = RefractiveBAConfig(verbosity=0)

        opt = RefractiveBAOptimizer(
            dataset=dataset,
            cam_params=cam_params,
            cams_cpp=cams_cpp,
            cam_to_window=cam_to_window,
            window_media=window_media,
            window_planes=window_planes,
            wand_length=wand_length,
            config=config,
        )

        # Core state
        self.assertEqual(opt.wand_length, wand_length)
        self.assertEqual(sorted(opt.active_cam_ids), [0, 1])
        self.assertEqual(opt.window_ids, [0])
        self.assertIsInstance(opt.config, RefractiveBAConfig)

        # cam_params stored as float64
        for cid in cam_ids:
            self.assertEqual(opt.cam_params[cid].dtype, np.float64)
            self.assertEqual(opt.cam_params[cid].shape, (11,))
            self.assertAlmostEqual(float(opt.cam_params[cid][6]), 1000.0)

        # Window planes deep-copied
        self.assertIn(0, opt.window_planes)
        np.testing.assert_array_almost_equal(
            opt.window_planes[0]["plane_pt"], [0.0, 0.0, 500.0]
        )
        np.testing.assert_array_almost_equal(
            opt.window_planes[0]["plane_n"], [0.0, 0.0, 1.0]
        )

        # Observation cache populated
        self.assertGreater(len(opt.obs_cache), 0)

        # Initial copies exist
        self.assertIn(0, opt.initial_planes)
        self.assertIn(0, opt.initial_cam_params)


# ===================================================================
# Standalone helper tests (no class-level patches needed)
# ===================================================================

class TestFactoryFunctions(unittest.TestCase):
    """Validate factory functions produce correct shapes and types."""

    def test_make_minimal_dataset(self):
        ds = make_minimal_dataset(n_frames=3, cam_ids=(0, 1, 2))
        self.assertIn("obsA", ds)
        self.assertIn("obsB", ds)
        self.assertIn("frames", ds)
        self.assertEqual(len(ds["frames"]), 3)
        self.assertEqual(ds["est_radius_small_mm"], 1.5)
        self.assertEqual(ds["est_radius_large_mm"], 2.0)
        # Every frame has every cam
        for fid in ds["frames"]:
            for cid in (0, 1, 2):
                self.assertEqual(ds["obsA"][fid][cid].shape, (2,))
                self.assertEqual(ds["obsB"][fid][cid].shape, (2,))

    def test_make_cam_params(self):
        cp = make_cam_params(cam_ids=(0, 1), f=800.0)
        self.assertIn(0, cp)
        self.assertIn(1, cp)
        for cid in (0, 1):
            self.assertEqual(cp[cid].shape, (11,))
            self.assertEqual(cp[cid].dtype, np.float64)
            self.assertAlmostEqual(float(cp[cid][6]), 800.0)

    def test_make_window_planes(self):
        wp = make_window_planes()
        self.assertIn(0, wp)
        np.testing.assert_array_almost_equal(wp[0]["plane_pt"], [0, 0, 500.0])
        np.testing.assert_array_almost_equal(wp[0]["plane_n"], [0, 0, 1.0])

    def test_make_window_media(self):
        wm = make_window_media()
        self.assertIn(0, wm)
        self.assertAlmostEqual(wm[0]["n1"], 1.0)
        self.assertAlmostEqual(wm[0]["n2"], 1.5)
        self.assertAlmostEqual(wm[0]["n3"], 1.333)
        self.assertAlmostEqual(wm[0]["thickness"], 10.0)


class TestValidateRadii(unittest.TestCase):
    """Tests for ``_validate_radii``."""

    def test_valid_radii(self):
        # Should not raise
        _validate_radii(1.5, 2.0, label="test")

    def test_non_finite_raises(self):
        with self.assertRaises(AssertionError):
            _validate_radii(float("nan"), 2.0, label="nan-check")
        with self.assertRaises(AssertionError):
            _validate_radii(1.0, float("inf"), label="inf-check")

    def test_non_positive_raises(self):
        with self.assertRaises(AssertionError):
            _validate_radii(0.0, 2.0, label="zero-check")
        with self.assertRaises(AssertionError):
            _validate_radii(-1.0, 2.0, label="neg-check")

    def test_out_of_range_raises(self):
        with self.assertRaises(AssertionError):
            _validate_radii(0.05, 2.0, label="too-small")
        with self.assertRaises(AssertionError):
            _validate_radii(1.0, 55.0, label="too-large")

    def test_wrong_order_raises(self):
        with self.assertRaises(AssertionError):
            _validate_radii(3.0, 1.0, label="order-check")


class TestCheckOptimizationResult(unittest.TestCase):
    """Tests for ``check_optimization_result``."""

    def test_status_ge_1(self):
        res = MagicMock(status=1, message="converged")
        use, sev, msg = check_optimization_result(res, "joint")
        self.assertTrue(use)
        self.assertEqual(sev, "info")
        self.assertIn("converged", msg)

    def test_status_0(self):
        res = MagicMock(status=0, message="max iter")
        use, sev, msg = check_optimization_result(res, "joint")
        self.assertTrue(use)
        self.assertEqual(sev, "warning")
        self.assertIn("budget", msg.lower())

    def test_status_neg1(self):
        res = MagicMock(status=-1, message="failed")
        use, sev, msg = check_optimization_result(res, "final_refined")
        self.assertFalse(use)
        self.assertEqual(sev, "error")
        self.assertIn("failed", msg)


class TestBuildOptimizerForTest(unittest.TestCase):
    """Tests for ``build_optimizer_for_test``."""

    def test_returns_optimizer(self):
        from modules.camera_calibration.wand_calibration.refraction_calibration_BA import (
            RefractiveBAOptimizer,
        )
        opt = build_optimizer_for_test()
        self.assertIsInstance(opt, RefractiveBAOptimizer)
        self.assertEqual(sorted(opt.active_cam_ids), [0, 1])
        self.assertGreater(len(opt.obs_cache), 0)


class TestRobustPlanePointInit(unittest.TestCase):
    """Tests for ``robust_plane_point_init``."""

    def test_median_of_points(self):
        pts = np.array([
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0],
            [100.0, 200.0, 300.0],  # outlier
        ])
        result = robust_plane_point_init(pts)
        self.assertEqual(result.shape, (3,))
        self.assertEqual(result.dtype, np.float64)
        # Median should resist the outlier
        np.testing.assert_array_almost_equal(result, [1.0, 2.0, 3.0])

    def test_single_point(self):
        pt = np.array([5.0, 6.0, 7.0])
        result = robust_plane_point_init(pt)
        np.testing.assert_array_almost_equal(result, [5.0, 6.0, 7.0])


if __name__ == "__main__":
    unittest.main()
