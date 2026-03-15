# Wand Calibrator Fixes — Learnings

## 2026-03-15 Init: Plan startup

### Key Codebase Facts
- Target files (ONLY these 4 may be modified):
  1. `modules/camera_calibration/wand_calibration/refraction_wand_calibrator.py` — B5 (~line 2224), P2 (lines 486–495)
  2. `modules/vsc/refraction_optimizer.py` — B1 (lines 458–465)
  3. `modules/camera_calibration/wand_calibration/refraction_calibration_BA.py` — B3, P1, P3, P4
  4. `modules/camera_calibration/wand_calibration/test_wand_calibrator_fixes.py` (NEW — to be created)

- Test file location: `modules/camera_calibration/wand_calibration/test_wand_calibrator_fixes.py`
  - NOTE: Plan says `tests/` dir but actual target should match module structure; agent must verify which `tests/` vs `modules/.../` path
  
### Mock Targets
- `build_pinplate_rays_cpp_batch` and `update_cpp_camera_state` imported at lines 44–51 of BA file
- `@patch` at module import level: `modules.camera_calibration.wand_calibration.refraction_calibration_BA.<fn>`
- `cams_cpp` values: `MagicMock()` with `projectBatchStatus.return_value = []`

### Noise Model Location
- `sigma_ray = px_target * (avg_dist_z / avg_f)` at line 561 of refraction_calibration_BA.py
- NOT in refraction_wand_calibrator.py:400–450 (that's plane init)

### Radii Validation Rules
- Must use `np.isfinite()`, NOT `is not None`
- Valid range: [0.1, 50.0] mm, positive, finite, radius_B ≥ radius_A
- Trap: `dataset.get('est_radius_small_mm', 0.0)` default of 0.0 must be caught

### Status Code Handling
- status ≥ 1 → SUCCESS, always use res.x
- status == 0 → WARNING (budget exhausted), STILL use res.x (partial progress valid)
- status == -1 → ERROR, raise RuntimeError
- DO NOT raise on status=0 (breaks chunked optimization at lines 1432–1530)

### Conda Environment
- Use `conda run -n OpenLPT python ...` for all Python commands
- Test command: `conda run -n OpenLPT python -m pytest <file> -v`

### Branch
- Currently on `Refraction` branch (detached HEAD or branch, see git status)

## 2026-03-15 B5: rs_pr4/rl_pr4 fallback hardening

### Fix Pattern
- In `refraction_wand_calibrator.py` around BA config construction, keep explicit defaults before optional estimate path:
  - `rs_pr4, rl_pr4 = 1.5, 2.0  # Defaults`
- After estimate assignment block, guard both values with `np.isfinite(...)` + positive checks and reset both to defaults on failure.

### Regression Test Pattern
- Added `TestWandCalibratorB5Fixes.test_b5_defaults_fallback` in existing `test_wand_calibrator_fixes.py`.
- For this path-specific test, patch calibrator internals to bypass unrelated bootstrap/triangulation complexity and capture BA config radii.
- Minimal dummy base must include `dist_coeff_num` in addition to camera settings/cam_params.

## 2026-03-15 T1+T2: Test infrastructure file created

### Test File
- Created `modules/camera_calibration/wand_calibration/test_wand_calibrator_fixes.py` (430 lines, 16 tests)
- All 16 tests pass with both `unittest` and `pytest`

### Constructor Mocking Pattern
- `RefractiveBAOptimizer.__init__` calls `_refresh_plane_round_reference()` which uses `cv2.Rodrigues` on `cam_params`
- Zero rvec/tvec works fine with real cv2 — no need to mock cv2
- The two C++ functions that MUST be patched: `build_pinplate_rays_cpp_batch` (returns `[]`) and `update_cpp_camera_state` (returns `None`)
- Class-level `@patch` on `TestRefractiveBAOptimizerUnit` — mock args arrive as extra positional params to test methods

### Pytest Discovery
- `pytest` was NOT installed in OpenLPT conda env; installed via `pip install pytest`
- Alternative: `python -m unittest modules.camera_calibration.wand_calibration.test_wand_calibrator_fixes -v` always works
- Direct python path: `C:\Users\tan_s\miniconda3\envs\OpenLPT\python.exe` avoids conda run issues

### grep/LSP Quirks
- `grep` tool does NOT find patterns in the BA file (3281 lines, possibly encoding/size issue)
- Workaround: use `conda run -n OpenLPT python -c "import ast; ..."` to parse with AST and find class/function definitions
- LSP diagnostics show pre-existing numpy/cv2/scipy stub errors throughout the entire codebase — not actionable

### Key Architectural Decisions
- `ObsCacheBuilder.build()` at line 142 — static method, builds from `dataset['obsA']` and `dataset['obsB']`
- `RefractiveBAConfig` at line 173 — pure `@dataclass`, no need to mock (per MUST NOT rules)
- `cam_to_window` maps cam_id → window_id; `wand_length` is float in mm
- Constructor signature: `(dataset, cam_params, cams_cpp, cam_to_window, window_media, window_planes, wand_length, config=None, progress_callback=None)`
