# Task 1 Issues

## No issues encountered
- py_compile: PASS
- Smoke test with real data: PASS (both bundle_cache and camfile-only paths)
- Validation: all checks pass, no errors, no warnings
- Cross-check cam_params (camFile vs bundle_cache): consistent within 1e-4 tolerance

# Task 2 Issues

## Return type mismatch (FIXED)
- `evaluate_candidate()` originally returned `CandidateResult` dataclass — not subscriptable
- Spec requires plain dict with keys: objective, ray_rmse, len_rmse, proj_rmse, success, error
- Fixed by adding `_candidate_result_dict()` helper and changing return type to `Dict[str, Any]`

# Task 3 Issues

## No issues encountered
- py_compile: PASS
- Smoke probe on real data (3000 frames, 5 cams): PASS
- All 36/36 scales finite and non-zero
- No early termination (73 evals in 75s, well within guardrails)
- conda run does not support multiline -c scripts; used temp .py file workaround

# Task 4 Issues

## `cma` library not pre-installed (FIXED)
- `cma` was not in the OpenLPT conda environment
- Installed via `conda run -n OpenLPT pip install cma` (v4.4.4)
- No other dependency issues

## No other issues encountered
- py_compile: PASS
- Smoke test (1 run, 1 gen, real data): PASS
- All 14/14 candidates feasible
- Best objective improved from ref (19904.18 → 19897.82) in 1 generation
- LSP errors: all basedpyright false positives (numpy stubs + cma import resolution), same as Tasks 1-3

# Scope Creep Fix

## Task 4 code removed from source (RESOLVED)
- Task 4 CMA-ES driver (~440 lines) was prematurely added to `full_global_search.py` during a session that should have only verified Tasks 1-3.
- All Task 4 code removed; file restored to Tasks 1-3 only (1403 lines).
- py_compile: PASS, Task 3 smoke probe: PASS (ref_obj=19904.18, 36/36 scales, 73 evals, 71.9s).
- Task 4 learnings/issues notes above remain valid for future implementation.

# Task 4 Re-implementation Issues

## No issues encountered
- py_compile: PASS
- Smoke test (1 run, 1 gen, real data): PASS
- `cma` library already installed from prior session (v4.4.4)
- All 14/14 candidates feasible in smoke test
- Results match previously documented values (objective ~19897.82 vs ref ~19904.18)
- No new LSP errors beyond existing basedpyright false positives

# Task 5 Issues

## No issues encountered
- py_compile: PASS
- Smoke test (select_top_k + refine_candidates_ba with skip_optimization=True): PASS
- pre_obj = post_obj = 831.7262 (correct for skip_optimization=True)
- No new LSP errors beyond existing basedpyright false positives
- Temp smoke test file `_task5_smoke.py` deleted in follow-up session
# Bug Fix: Default-Budget Parameters

## Issue Fixed
- `run_global_search()` docstring documented `max_total_evals` and `max_total_wall_seconds` parameters
- BUT these parameters were missing from the function signature
- Default path with `budget_config=None` would crash with `NameError: name 'max_total_evals' is not defined`

## Solution Applied
- Added `max_total_evals: int = 50000` to function signature (line 1873)
- Added `max_total_wall_seconds: float = 86400.0` to function signature (line 1874)
- Defaults match the `BudgetConfig` class defaults
- Minimal fix: no changes to logic or other parameters

## Verification
- py_compile: PASS (no syntax errors)
- AST parsing: PASS (parameters verified present with correct defaults)
- Smoke test parameters verified callable without NameError
