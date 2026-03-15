# Wand Calibrator: Critical Bug Fixes & Precision Improvements

## TL;DR
> **Summary**: Fix 7 critical/high bugs in wand calibrator (`refraction_wand_calibrator.py` and `refraction_calibration_BA.py`), implement 5 priority precision improvements, and add comprehensive test coverage via TDD.
> **Deliverables**: 
> - 7 bug fixes (B1–B7) with atomic commits and test cases
> - 5 precision improvements (P1–P5) with test-driven validation
> - Full test suite for both modules
> - Evidence artifacts for all QA scenarios
> **Effort**: Large (50+ TODOs, 2-3 week executive capacity)
> **Parallel**: YES — 4 waves (dependencies tracked)
> **Critical Path**: B5 (unbound vars) → B1/B2 (stale normal/RMSE) → B3 (scipy status) → P1–P5 (precision tuning)

## Context

### Original Request
User asked Metis and Oracle to audit `refraction_wand_calibrator.py` for bugs, precision improvements, and robustness issues. After comprehensive analysis, 7 critical/high bugs and 12 precision/robustness improvements were identified. User requested a formal fix plan with atomic commits, TDD structure, and Metis review.

### Interview Summary
**Scope Confirmed**: Fix all 7 HIGH/CRITICAL bugs + implement 5 priority precision improvements (P1–P5).  
**Test Strategy**: TDD (RED-GREEN-REFACTOR) — write failing tests first, then fix code.  
**Commits**: Atomic (one bug/improvement per commit with message format `fix(wand_calibrator): {description}`).  
**Coverage**: Every fix must have ≥2 QA scenarios (happy path + failure/edge case).  
**No Decisions Pending**: User approved full scope in earlier session.

### Metis Review (gaps addressed)
[PENDING — will update after Metis consultation]

## Work Objectives

### Core Objective
Repair and harden the wand calibration pipeline by fixing undefined-state bugs, control-flow defects, and stale-variable issues that degrade calibration precision and robustness. Implement targeted precision improvements (barrier smoothing, percentile initialization) validated via test-driven development.

### Deliverables
1. ✅ Fixed `refraction_wand_calibrator.py` (7 bugs addressed, 0 regressions)
2. ✅ Fixed `refraction_calibration_BA.py` (5 bugs addressed, scipy validation)
3. ✅ Comprehensive test suite (`test_wand_calibrator_fixes.py`) with ≥30 test cases
4. ✅ Atomic commits (1 per bug/improvement, signed-off)
5. ✅ Evidence artifacts (screenshots/logs from QA scenarios)

### Definition of Done (verifiable via agents)
- [ ] All 7 bugs fixed + verified by test cases
- [ ] All 5 precision improvements implemented + validated
- [ ] Test suite runs cleanly: `conda run -n OpenLPT pytest modules/camera_calibration/wand_calibration/test_wand_calibrator_fixes.py -v`
- [ ] No regressions in `test_refractive_staged_strategy.py`
- [ ] All commits atomic, signed-off, with Fixes: tags
- [ ] Code passes linting: `pylint src/ --disable=<approved-exceptions>`

### Must Have
- Bug fixes ONLY in `refraction_wand_calibrator.py` and `refraction_calibration_BA.py`
- Test cases for every fix (RED→GREEN→REFACTOR cycle)
- Atomic commits (no god commits)
- No modifications to unrelated files
- Full QA scenarios with expected outputs

### Must NOT Have
- Refactoring unrelated to bugs (e.g., renaming vars, dead-code cleanup)
- Speculative improvements beyond P1–P5
- Changes to `refractive_bootstrap.py` or `refraction_plate_calibration.py` (reference only)
- Documentation-only commits

## Verification Strategy

**ZERO HUMAN INTERVENTION** — all verification is agent-executed.

- **Test Decision**: TDD (RED-GREEN-REFACTOR) using `pytest` + fixtures
- **QA Policy**: Every bug fix + improvement has ≥2 test scenarios (happy + edge case)
- **Evidence**: `.sisyphus/evidence/task-{N}-{slug}.{ext}` (test logs, screenshots)
- **Regression Prevention**: Run full test suite after each fix; no regressions allowed
- **Acceptance**: Agent passes all test scenarios and produces evidence artifacts

## Execution Strategy

### Parallel Execution Waves

**Wave 1 (Foundation / Test Infrastructure)**
- T1. Set up test module and fixtures (refraction_BA mock, window/wand mocks)
- T2. Create test helper functions (synthetic point clouds, ground-truth normals)

**Wave 2 (Critical Bug Fixes)**
- T3. Fix B5: Uncomment `rs_pr4`/`rl_pr4` defaults + add validation test
- T4. Fix B1: Replace stale `n0` with `n_new` in plane update logic
- T5. Fix B2: Correct RMSE denominator (2N → len(uv_list))
- T6. Fix B3: Add scipy status validation (status > 0) before accessing optim result

**Wave 3 (High-Priority Fixes & First Precision Improvement)**
- T7. Fix B4: Add early validation in `calibrate()` to catch undefined vars
- T8. Fix B6: Add window bounds validation in wand detection loop
- T9. Fix B7: Clarify/enforce plane normal semantics in docstrings
- T10. Implement P1: Replace hard C0 kink in barrier with softplus (smooth throughout)

**Wave 4 (Precision Improvements & Final Validation)**
- T11. Implement P2: Use percentile-based plane initialization (robust to outliers)
- T12. Implement P3: Adaptive sigma estimation from residuals
- T13. Implement P4: Convergence tolerance tuning for preconditioned optimization
- T14. Final regression test suite + integration smoke test

### Dependency Matrix

| Task | Depends On | Blocks |
|------|-----------|--------|
| T1 | — | T3–T14 (all) |
| T2 | T1 | T3–T14 (all) |
| T3 | T1, T2 | T4, T5, T6, T7 |
| T4 | T1, T2, T3 | T8, T10 |
| T5 | T1, T2, T3 | T8, T10 |
| T6 | T1, T2, T3 | T8, T10 |
| T7 | T3, T4, T5, T6 | T9 |
| T8 | T4, T5, T6, T7 | T10 |
| T9 | T8 | T10 |
| T10 | T8, T9 | T11 |
| T11 | T10 | T12 |
| T12 | T11 | T13 |
| T13 | T12 | T14 |
| T14 | T13 | — |

### Agent Dispatch Summary

| Wave | Tasks | Count | Categories |
|------|-------|-------|-----------|
| 1 | T1, T2 | 2 | `deep` (test infrastructure design) |
| 2 | T3–T6 | 4 | `quick` (targeted bug fixes) |
| 3 | T7–T10 | 4 | `quick` → `unspecified-high` (P1 requires barrier math) |
| 4 | T11–T14 | 4 | `unspecified-high` (precision tuning) + `deep` (regression suite) |

---

## TODOs

- [ ] T1. Set up test module and fixtures (refraction_BA mock, window/wand mocks)

  **What to do**: 
  - Create `modules/camera_calibration/wand_calibration/test_wand_calibrator_fixes.py` with pytest fixtures
  - Mock `RefractiveCalibrationBA` class with controllable outputs (optim result, residuals, jacobian)
  - Create window fixture (list of dicts with keys: `center`, `normal`, `boundaries`)
  - Create wand fixture (dict with `tip`, `shaft_dirs`, `sphere_center`, `sphere_radius`)
  - Create synthetic point cloud fixture (100 3D points + corresponding UV points with known ground truth)
  - Import necessary modules: `pytest`, `numpy`, `sys.path` injection for import `refraction_wand_calibrator`

  **Must NOT do**: 
  - Do not import real camera/image data
  - Do not modify any existing source files yet
  - Do not create fixtures for unrelated modules

  **Recommended Agent Profile**:
  - Category: `deep` — Reason: Requires understanding of test infrastructure, fixture design, and mock strategies
  - Skills: [] — Standard Python + pytest knowledge sufficient
  - Omitted: [`git-master`] — No commits yet; T1 creates test file only

  **Parallelization**: Can Parallel: NO | Wave 1 | Blocks: T2–T14 | Blocked By: —

  **References**:
  - Test pattern: `modules/camera_calibration/wand_calibration/test_refractive_staged_strategy.py:1–50` — pytest import and fixture structure
  - Mock pattern: `modules/camera_calibration/plate_calibration/test_refraction_plate_calibration.py` (if exists) or standard unittest.mock docs
  - RefractiveCalibrationBA class signature: `refraction_calibration_BA.py:1–150` (constructor, calibrate method, optim_result attribute)
  - Window/wand dict structures: `refraction_wand_calibrator.py:50–100` (docstrings and method signatures)

  **Acceptance Criteria**:
  - [ ] Test file created at correct path with no syntax errors
  - [ ] All fixtures import without runtime errors: `conda run -n OpenLPT python -m pytest modules/camera_calibration/wand_calibration/test_wand_calibrator_fixes.py::test_fixtures_load -v`
  - [ ] Mock `RefractiveCalibrationBA` accepts constructor args and returns expected attributes
  - [ ] Window/wand fixtures produce valid dicts (keys verified via assertion)
  - [ ] Synthetic point cloud has ≥100 points with matching UV/3D pairs

  **QA Scenarios**:
  ```
  Scenario: Fixtures load and produce expected types
    Tool: interactive_bash (conda run)
    Steps: 
      1. Run pytest with fixture test: conda run -n OpenLPT pytest modules/camera_calibration/wand_calibration/test_wand_calibrator_fixes.py -v -k "fixture" 2>&1 | tee .sisyphus/evidence/task-1-fixtures.log
      2. Check return code is 0
    Expected: All fixtures collected and validated; return code 0; no import errors
    Evidence: .sisyphus/evidence/task-1-fixtures.log

  Scenario: Mock RefractiveCalibrationBA responds correctly
    Tool: interactive_bash
    Steps:
      1. Add a test function that instantiates mock, calls calibrate(), checks optim_result
      2. Run: conda run -n OpenLPT pytest modules/camera_calibration/wand_calibration/test_wand_calibrator_fixes.py::test_mock_ba -v
    Expected: Mock returns predefined optim_result with status=1 (success); no exceptions
    Evidence: .sisyphus/evidence/task-1-mock.log
  ```

  **Commit**: YES | Message: `test(wand_calibrator): add test module and mock fixtures for BA/wand/window` | Files: `modules/camera_calibration/wand_calibration/test_wand_calibrator_fixes.py`

---

- [ ] T2. Create test helper functions (synthetic point clouds, ground-truth normals)

  **What to do**:
  - Add helper functions to test module:
    - `make_synthetic_point_cloud(n_points=100, normal_ref=None)` → returns 3D points + UV points + ground truth normal
    - `make_outlier_cloud(n_good=80, n_outliers=20)` → returns cloud with known outliers for robust initialization tests
    - `make_ill_conditioned_cloud(condition_num=1000)` → returns nearly-coplanar cloud for convergence tests
  - Each function must return dict: `{"points_3d", "points_uv", "normal_ref", "plane_origin"}`
  - Ensure clouds are deterministic (use fixed random seed)
  - Validate cloud geometry (e.g., points lie on reference plane ± noise)

  **Must NOT do**:
  - Do not modify existing test files
  - Do not add to source modules yet
  - Do not create new dependencies (use numpy only)

  **Recommended Agent Profile**:
  - Category: `quick` — Reason: Straightforward synthetic data generation, no architectural decisions
  - Skills: [] — numpy geometry calculations only
  - Omitted: [] — None

  **Parallelization**: Can Parallel: NO | Wave 1 | Blocks: T3–T14 | Blocked By: T1

  **References**:
  - Plane equation: `refraction_calibration_BA.py:1050–1100` (plane repr: origin + normal)
  - Point cloud generation pattern: `test_refractive_staged_strategy.py:100–150` (if similar test exists)
  - Noise model: `refraction_wand_calibrator.py:400–450` (observation noise assumptions)

  **Acceptance Criteria**:
  - [ ] `make_synthetic_point_cloud(100)` returns 100 points with matching UV/3D
  - [ ] Points lie on reference plane within ±0.001 mm (validate via dot product)
  - [ ] `make_outlier_cloud()` produces exactly 80 good + 20 outliers (verifiable by distance from plane)
  - [ ] All clouds use fixed seed; repeated calls return identical results
  - [ ] Helper functions are importable: `from test_wand_calibrator_fixes import make_synthetic_point_cloud`

  **QA Scenarios**:
  ```
  Scenario: Synthetic clouds have correct geometry
    Tool: interactive_bash
    Steps:
      1. Create test: def test_cloud_geometry(): cloud = make_synthetic_point_cloud(); assert np.allclose(np.dot(cloud["points_3d"] - cloud["plane_origin"], cloud["normal_ref"]), 0, atol=1e-3)
      2. Run: conda run -n OpenLPT pytest modules/camera_calibration/wand_calibration/test_wand_calibrator_fixes.py::test_cloud_geometry -v
    Expected: All points satisfy plane equation; test passes
    Evidence: .sisyphus/evidence/task-2-geometry.log

  Scenario: Outlier cloud has exact counts
    Tool: interactive_bash
    Steps:
      1. Test: cloud = make_outlier_cloud(); distances = np.linalg.norm(cloud["points_3d"] - ..., axis=1); assert (distances > 0.5).sum() == 20
      2. Run: conda run -n OpenLPT pytest test_wand_calibrator_fixes.py::test_outlier_count -v
    Expected: Exactly 20 outliers detected; test passes
    Evidence: .sisyphus/evidence/task-2-outliers.log
  ```

  **Commit**: YES | Message: `test(wand_calibrator): add synthetic point cloud generators and geometry validators` | Files: `modules/camera_calibration/wand_calibration/test_wand_calibrator_fixes.py`

