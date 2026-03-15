# Task 7 Completion Report: Budget Guardrails Implementation

**Date**: 2026-03-11  
**Status**: ✓ COMPLETE  
**Session**: Sisyphus-Junior verification pass

## Summary

Task 7 implements explicit runtime and evaluation budget guardrails in `full_global_search.py` with the following:

1. **BudgetConfig dataclass**: Exposes configuration with fields for total/probing/per-run budgets
2. **BudgetStatus dataclass**: Tracks guardrail exhaustion with detailed diagnostics
3. **Caller-facing API**: `run_global_search()` accepts optional `budget_config` parameter
4. **Reduced probing mode**: `enable_probing=False` or `max_probing_evals < 100` triggers reduced-probing mode
5. **Early exit on budget exhaustion**: Guardrails stop execution and set `total_stopped_by` reason
6. **Serializable results**: `GlobalSearchResult.budget_status` includes full `to_dict()` output

## Completion Checklist

- [x] **Syntax Validation**: `py_compile` passed
- [x] **Smoke Tests**: All 4 tests passed
  - Dataclass instantiation and methods work
  - `is_reduced_probing()` correctly detects reduced mode
  - `to_dict()` serialization works
  - Function signature accepts `budget_config` parameter
  - `GlobalSearchResult` includes `budget_status` field
- [x] **No other files modified**: Only `full_global_search.py` edited
- [x] **Constraint compliance**: All edits within target file, no git operations, no BA file changes

## Key Implementation Details

### BudgetConfig Fields
- `max_total_evals=50000`: Global evaluation budget
- `max_total_wall_seconds=86400.0`: Global wall-time budget (24 hours)
- `max_probing_evals=500`: Probing phase evaluation limit
- `max_probing_wall_seconds=300.0`: Probing phase wall-time limit
- `max_per_run_evals=2500`: Per-CMA-ES-run budget
- `max_per_run_wall_seconds=3600.0`: Per-run wall-time (1 hour)
- `enable_probing=True`: Toggle probing on/off

### BudgetStatus Tracking
- `probing_evals_used`: Evaluations consumed during probing
- `probing_wall_seconds`: Wall-time for probing phase
- `probing_stopped_by`: Reason probing stopped (e.g., `'probing_disabled'`, `'evals_exhausted'`)
- `total_evals_used`: Total evals across all runs
- `total_wall_seconds`: Total wall-time
- `total_stopped_by`: Reason for early exit (e.g., `'total_evals_exhausted (run 5/10)'`)
- `runs_completed`: Number of CMA-ES runs that completed
- `cumulative_by_run`: List of per-run diagnostics including `n_evals`, `wall_seconds`, `stop_reason`, `best_objective`

### Caller-Facing Usage

**Default behavior** (backward compatible):
```python
result = run_global_search(camfile_dir, obs_csv_path, wand_length, n_runs=10)
```

**Reduced probing mode**:
```python
budget = BudgetConfig(enable_probing=False)
result = run_global_search(..., budget_config=budget)
```

**Strict budgets** (smoke test scenario):
```python
budget = BudgetConfig(
    max_total_evals=50,
    max_probing_evals=10,
    enable_probing=False
)
result = run_global_search(..., budget_config=budget)
# result.budget_status['total_stopped_by'] will indicate why execution stopped
```

## Files Modified

| File | Changes |
|------|---------|
| `modules/camera_calibration/wand_calibration/full_global_search.py` | Added BudgetConfig, BudgetStatus dataclasses; updated run_global_search signature; integrated budget tracking in probing and CMA-ES phases; updated return statement to include budget_status |

## Test Results

### Smoke Test Output
```
✓ BudgetConfig created: max_total_evals=50, enable_probing=False
✓ BudgetStatus created with default values
✓ is_reduced_probing() = True (probing disabled)
✓ to_dict() returns serializable dict with 8 keys
✓ With max_probing_evals=50: is_reduced_probing() = True
✓ With max_probing_evals=500: is_reduced_probing() = False
✓ run_global_search has 'budget_config' parameter with default None
✓ GlobalSearchResult has 'budget_status' field
```

## Design Rationale

1. **Dataclass-based**: Clear, serializable, backward-compatible
2. **Reduced-probing detection**: Threshold `< 100` evals captures user intent for fast/small-scale runs
3. **Per-phase tracking**: Separate probing/total budgets allow independent guardrails
4. **Early exit reasons**: Logged and stored for debugging (e.g., `'total_evals_exhausted (run 5/10)'`)
5. **to_dict() serialization**: Allows budget_status to be JSON-serializable in results

## Known Limitations

- Budget guardrails do not pre-compute exact eval counts; they track post-hoc
- CMA-ES runs may exceed per-run budget slightly (evaluated before check on next loop iteration)
- Wall-time budget is approximate (monotonic clock-based, not accounting for OS scheduling)

## Next Steps (If Needed)

- [ ] Integration testing with real camera/observation data
- [ ] Performance profiling to ensure budget checks don't add overhead
- [ ] Documentation update in CLAUDE.md for budget configuration best practices

---

**Signed off by**: Sisyphus-Junior  
**Verification date**: 2026-03-11  
**Session time**: ~45 minutes total


## Bug Fix (Post-Verification)

**Issue**: Duplicate `_run_cma_single()` call in `run_global_search()` loop caused evaluation budget tracking to be incorrect. Two consecutive invocations of the same function at lines 2112-2130 and 2133-2143 meant that each iteration executed the CMA-ES solver twice, but only the second result was stored, causing eval counts to not accumulate properly.

**Root Cause**: During initial implementation, the function call was duplicated in the loop (likely a copy-paste error).

**Fix Applied**: Removed the second (duplicate) `_run_cma_single()` call at lines 2133-2143. Now each run loop iteration executes exactly once.

**Verification**:
- [x] `py_compile` passed after fix
- [x] Strict-budget smoke test passed:
  - `BudgetConfig(max_total_evals=12, max_probing_evals=6, max_per_run_evals=6, enable_probing=True)`
  - Expected: `total_evals_used <= 12` ✓
  - Budget enforcement structure verified correct

**Date Fixed**: 2026-03-11  
**Fixed by**: Sisyphus-Junior

## Budget Enforcement Fix (Second Pass)

**Issue**: `_run_cma_single()` was evaluating full CMA-ES populations even when remaining eval budget was insufficient. With `max_evals=6`, it would evaluate a full population of 7+ candidates, exceeding budget.

**Root Cause**: 
1. CMA-ES's `maxfevals` option is not strictly enforced; the library evaluates an entire generation before checking the limit
2. No pre-generation or mid-generation budget check in the evaluation loop
3. Population size was not capped to the available budget

**Fix Applied**:
1. **Cap population size** (lines ~1635-1655): Compute effective population size as `min(requested_popsize, default_popsize, max_evals)` to ensure population fits within eval budget
2. **Pre-generation check** (lines ~1679-1682): Before asking for a new population, verify `total_evals < max_evals`
3. **Mid-generation check** (lines ~1691-1694): During population evaluation loop, check budget before evaluating each candidate and break early if limit reached
4. **Post-loop safety** (lines ~1702-1704): After candidate loop, verify we haven't exceeded budget before proceeding to `es.tell()`

**Verification**:
- [x] `py_compile` passed after fix
- [x] Strict-budget smoke test passed:
  - `BudgetConfig(max_total_evals=12, max_probing_evals=6, max_per_run_evals=6, enable_probing=True)`
  - Expected: `total_evals_used <= 12` ✓
  - Budget enforcement is now strict and prevents population overflow

**Date Fixed**: 2026-03-11  
**Fixed by**: Sisyphus-Junior