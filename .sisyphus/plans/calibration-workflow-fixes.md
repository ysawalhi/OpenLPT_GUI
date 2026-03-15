# Calibration Workflow Fixes: Stage-2 Parallelization, Probing Metrics, BrokenProcessPool Retry

## TL;DR
> **Summary**: Three targeted fixes to improve calibration stability and performance: (1) enable Stage-2 parallelization by removing threshold, (2) investigate and log correct compensated metric in probing stop condition, (3) add retry-with-backoff mechanism for BrokenProcessPool failures.
> **Deliverables**: Modified `full_global_search.py` with all three fixes applied and tested.
> **Effort**: Medium
> **Parallel**: NO (sequential fixes required; testing last)
> **Critical Path**: Fix 1 → Fix 2 → Fix 3 → Test

## Context

### Original Request
User discovered three issues via log analysis:
1. Stage-2 parallelization not triggering (n_blocks >= max_workers condition was too conservative)
2. Probing stop rule inconsistency (logged ray_now ~0.72-0.78 differs from stop trigger ~0.99)
3. BrokenProcessPool failures during parallel CMA-ES (currently falls back to serial only)

### Interview Summary
- **User decision**: Remove Stage-2 threshold entirely (not just relax it)
- **User decision**: Investigate root cause of metric mismatch (not just logging fix)
- **User decision**: Implement retry-with-backoff + diagnostics for BrokenProcessPool

### Discovery Findings
1. **Stage-2 code location**: Lines 2847-2856, 2863-3093 (sequential path), 3036-3093 (parallel path)
2. **Probing stop metric**: Uses `comp.ray_rmse` (compensated metric) at line 2737 and 2952
3. **Logging inconsistency**: Line 2749 and 2970 log `ray_now = float(comp.ray_rmse)` but value differs from stop threshold
4. **BrokenProcessPool handling**: 
   - Currently: Lines 4474-4516 in CMA-ES code only (catches and falls back to serial)
   - Missing: In Stage-2 parallel path (lines 3042-3093)
5. **Worker import**: Already fixed at line 2657 (`import time as _time`)

### Reference Code Locations
- **Stage-2 sequential evaluation**: Lines 2933-2941 and 2952-2957 (use `comp.ray_rmse`)
- **Stage-2 parallel evaluation**: Lines 2717-2738 (use `comp.ray_rmse`)
- **Stage-2 parallel executor**: Lines 3042-3093 (NO BrokenProcessPool handling)
- **CMA-ES BrokenProcessPool handler**: Lines 4474-4516 (working example)

## Work Objectives

### Objective 1: Enable Stage-2 Parallelization
Remove the `n_blocks >= max_workers` condition that was blocking parallel execution.

### Objective 2: Investigate Probing Metric Mismatch
Find why the logged `ray_now` differs from the value used in stop condition, then add instrumentation to capture both metrics for debugging.

### Objective 3: Add BrokenProcessPool Retry Mechanism
Implement exponential backoff retry in Stage-2 parallel path with diagnostics.

## TODOs

### Fix 1: Remove Stage-2 Parallelization Threshold

**What to do**: 
1. Locate line 2856 in `full_global_search.py`
2. Change condition from `use_parallel = (shared_setup is not None and max_workers > 1 and n_blocks >= max_workers)` 
3. To: `use_parallel = (shared_setup is not None and max_workers > 1)`
4. Update comment above (lines 2853-2855) to reflect the removal of the threshold

**Why**: User explicitly requested removing the condition entirely to maximize parallelism on multi-core systems.

**Must NOT do**: 
- Do not relax threshold to `n_blocks >= max_workers/2` or similar
- Do not make it conditional on a config flag

**References**:
- Current code: `full_global_search.py:2853-2856`
- Similar pattern (Stage-1): `full_global_search.py:2150-2157` (already relaxed to `n_params >= max_workers`)

**Acceptance Criteria**:
- [x] Line 2856 condition is simplified to check only `shared_setup is not None and max_workers > 1`
- [x] Comment clarifies threshold was removed
- [x] Code compiles without syntax errors
- [x] Logging output at line 2858-2861 will show `parallel=True` when n_blocks is small (e.g., 6 blocks, 19 workers)

---

### Fix 2: Investigate and Log Probing Metric Mismatch

**What to do**:
1. Add detailed logging at line 2737 (Stage-2 worker compensation step) to capture:
   - `alpha` (current step size)
   - `comp.ray_rmse` (compensated metric)
   - `ray_stop_threshold` (reference threshold)
   - Comparison result
2. Add same logging at line 2952 (sequential Stage-2) for consistency
3. Search codebase to understand if `evaluate_probe_step_with_compensation()` modifies the metrics between returned value and stop check
4. If metric difference found, document the semantic difference (e.g., "raw vs. adjusted", "pre-compensation vs. post-compensation")

**Why**: Log values (~0.72-0.78) differ from stop trigger values (~0.99), indicating either:
- A different metric is being compared in the stop condition
- The metric is being modified between logging and comparison
- The stop threshold is computed differently than documented

**Must NOT do**:
- Do not change the stopping logic itself (that's correct per design)
- Do not modify the compensation function
- Do not assume metric mismatch is a bug (it may be intentional design)

**References**:
- Worker function compensation call: `full_global_search.py:2718-2724` and `2934-2940`
- Stop check: `full_global_search.py:2736-2738` and `2952-2957`
- Logging location: `full_global_search.py:2742-2751` (worker) and `2961-2971` (sequential)

**Acceptance Criteria**:
- [x] Enhanced logging added at worker compensation (line 2737)
- [x] Enhanced logging added at sequential compensation (line 2952)
- [x] Log format includes: alpha, comp.ray_rmse, ray_stop_threshold, and comparison result
- [x] All debug.log entries match the actual values used in stopping condition
- [x] Documentation clarifies semantic of `ray_rmse` vs. `ray_stop_threshold` (if different)

---

### Fix 3: Add BrokenProcessPool Retry Mechanism to Stage-2 Parallel Path

**What to do**:
1. Import `BrokenProcessPool` at top of file (add to import at line 4413, move it earlier if needed): `from concurrent.futures.process import BrokenProcessPool`
2. Wrap the ProcessPoolExecutor context (lines 3042-3093) in a retry loop with exponential backoff
3. For each retry:
   - Log attempt number and backoff delay
   - Wait exponentially (1s, 2s, 4s, etc., max 3 attempts)
   - Re-create the ProcessPoolExecutor
   - Re-submit all unfinished blocks
4. After max retries exhausted (3), fall back to sequential execution of remaining blocks
5. Log diagnostics: worker IDs that crashed (if available), memory usage (if available), exception details

**Why**: 
- ProcessPoolExecutor can break due to worker crashes, pickling errors, or memory pressure
- Current behavior (CMA-ES only): log error and fall back to serial
- Missing in Stage-2: no safeguards, causing hard failure
- User requested retry mechanism for stability

**Must NOT do**:
- Do not retry more than 3 times (diminishing returns)
- Do not change the overall logic (still fall back to serial if all retries fail)
- Do not add retry to sequential path (not needed)
- Do not use threading for retries (keep multiprocessing)

**References**:
- CMA-ES implementation (working pattern): `full_global_search.py:4413-4559`
  - BrokenProcessPool import: line 4413
  - Catch and log: lines 4474-4484 and 4511-4516
  - Fallback to serial: lines 4519-4559
- Stage-2 parallel path to modify: `full_global_search.py:3036-3093`
  - Executor creation: lines 3042-3046
  - Result collection: lines 3068-3072
  - No error handling currently

**Acceptance Criteria**:
- [x] `BrokenProcessPool` is imported from `concurrent.futures.process`
- [x] Stage-2 parallel executor wrapped in retry loop with exponential backoff (1s, 2s, 4s)
- [x] Max 3 retry attempts before falling back to serial
- [x] Serial fallback path implemented (similar to CMA-ES fallback at lines 4519-4559)
- [x] All retry attempts logged with attempt number and backoff duration
- [x] Worker exception details logged for debugging
- [x] Code compiles without syntax errors
- [x] Maintains same result aggregation logic (np.maximum for block_scales)

**QA Scenarios**:
```
Scenario: Normal parallel execution (no pool failure)
  Tool: Bash
  Steps: Run calibration with 8+ blocks and verify "parallel=True" in Stage-2 log
  Expected: All blocks complete in parallel, no retries logged
  Evidence: .sisyphus/evidence/fix3-normal-execution.log

Scenario: Single worker crash with recovery (simulated BrokenProcessPool)
  Tool: Manual (optional) — can also test by checking exception handling
  Steps: Trigger ProcessPoolExecutor.shutdown(wait=False) during result collection
  Expected: Retry loop catches BrokenProcessPool, retries 1-3x, eventually completes via serial
  Evidence: .sisyphus/evidence/fix3-retry-recovery.log

Scenario: Max retries exhausted
  Tool: Manual inspection of code
  Steps: Verify serial fallback path aggregates results correctly after pool failure
  Expected: Log shows "falling back to sequential execution", all blocks processed serially
  Evidence: Code inspection at Stage-2 fallback logic
```

---

## Verification Strategy

**Test Decision**: Tests-after (manual verification + diagnostics logging)

**QA Policy**: 
- Fix 1: Verify log output changes from `parallel=False` to `parallel=True` with small n_blocks
- Fix 2: Verify enhanced logging captures all metrics consistently
- Fix 3: Verify code paths execute correctly (can't easily trigger real BrokenProcessPool in test)

**Evidence Collection**:
- `.sisyphus/evidence/fix1-parallelization.log` — Stage-2 output with parallel=True
- `.sisyphus/evidence/fix2-metrics.log` — Stage-2 debug output with enhanced logging
- `.sisyphus/evidence/fix3-retry-logic.log` — Code inspection or diagnostic run with fallback path

---

## Execution Strategy

### Single Sequential Fix Flow (no parallelism needed)

1. **Fix 1** (20 min): Modify line 2856 condition, update comment, verify syntax
2. **Fix 2** (30 min): Add logging at lines 2737 and 2952, document metric semantics, verify syntax  
3. **Fix 3** (45 min): Add import, wrap executor in retry loop, implement serial fallback, verify syntax
4. **Integration Test** (30 min): Run short diagnostic calibration to verify all fixes work together

### Agent Dispatch Summary

| Task | Agent Profile | Reason | Timing |
|------|---|---|---|
| Fix 1 | quick | Single-line condition change + comment update | Sequential |
| Fix 2 | quick | Add logging statements at 2 locations | Sequential (depends on Fix 1) |
| Fix 3 | unspecified-high | New retry logic + fallback path, complex control flow | Sequential (depends on Fix 2) |
| Integration Test | unspecified-high | Full workflow test to verify no regressions | Sequential (depends on Fix 3) |

---

## Commit Strategy

**Single commit** after all three fixes + testing:
- **Type**: `fix`
- **Scope**: `calibration:stage2-parallelization`
- **Message**: `fix(calibration:stage2): enable parallelization, log probe metrics, add BrokenProcessPool retry`
- **Files Modified**: 
  - `modules/camera_calibration/wand_calibration/full_global_search.py` (3 separate edits within single session, committed together)

---

## Success Criteria

1. **Fix 1 Verified**: 
   - `use_parallel` variable set to True when max_workers > 1, regardless of n_blocks
   - Log output shows `parallel=True` for Stage-2 with 6 blocks and 19 workers

2. **Fix 2 Verified**:
   - Enhanced logging captures `alpha`, `comp.ray_rmse`, `ray_stop_threshold` at every step
   - No mismatch between logged value and stop-trigger value
   - If metric difference exists, documented in log or code comment

3. **Fix 3 Verified**:
   - Import statement added
   - Retry loop with backoff implemented
   - Serial fallback path mirrors CMA-ES logic
   - All code paths execute without errors

4. **Integration Test Passed**:
   - Diagnostic calibration run completes successfully
   - All three fixes log their respective outputs
   - No regressions in existing functionality
   - CSV output (if enabled) still works correctly

---

## Notes for Executor

- **User Constraint**: Only modify `full_global_search.py` — no helper scripts or other files
- **Environment**: Use `conda run -n OpenLPT python` for any test execution
- **Test Data**: Use existing test calibration project or short diagnostic run
- **Commit**: Only after all tests pass; do NOT commit partial fixes
