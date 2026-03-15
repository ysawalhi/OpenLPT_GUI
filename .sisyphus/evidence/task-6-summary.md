# Task 6: Control Oversubscription and Worker Resource Usage

**Status**: ✓ COMPLETED (2026-03-12)

## Summary

Task 6 implements explicit worker-count selection logic and oversubscription guardrails to prevent unbounded CPU usage and inefficient pool creation in `full_global_search.py`.

## Changes Made

### 1. Worker-Count Calculation (lines 2206-2214)
- **Formula**: `effective_workers = min(max_workers, n_runs)` when `max_workers > 1`, else 1
- **Purpose**: Prevents creating more worker processes than available CMA-ES runs
- **Gate**: Updated `_use_parallel` to check `effective_workers > 1` in addition to existing conditions

### 2. Logging Enhancements (lines 2222-2239)
Three distinct log messages now clarify concurrency decisions:
- **Parallel proceeding**: Logs `requested_max_workers`, `n_runs`, `effective_workers`, and timeout
- **Parallel bypassed**: Logs reason when parallel was requested but skipped
- **Parallel disabled**: Debug-level log when parallel is not enabled

### 3. Pool Creation Skip (line 2430-2433)
- Pool creation now respects both old and new gates via `_use_parallel`
- Verified: Pool is not created when `n_runs==1`, `max_workers==1`, or `enable_parallel=False`

### 4. Parallel Dispatch Logging (lines 2406-2410)
- Updated log message to show actual execution parameters more clearly
- Uses pre-computed `effective_workers` from the gate logic

## Acceptance Criteria Status

✓ **Default worker count is bounded and logged**
- Implemented via `min(max_workers, n_runs)` formula
- Logged with full context in three scenarios

✓ **n_runs==1 or max_workers==1 skips pool creation**
- Verified by smoke test: single-run with `enable_parallel=True` correctly uses sequential path
- Pool creation is conditional on `_use_parallel=False`

✓ **Native thread-cap guardrail present or documented**
- Documented as unsupported in Phase 1 (deferred to Phase 2)
- Existing comment: "This prevents oversubscription while bounding CPU usage to sane defaults"

## Verification Results

### Smoke Tests (7 scenarios, all PASSED)
| Scenario | Input | Result |
|----------|-------|--------|
| Single run | n_runs=1, enable_parallel=True, max_workers=4 | Pool skipped ✓ |
| Parallel disabled | n_runs=3, enable_parallel=False, max_workers=4 | Sequential used ✓ |
| max_workers=1 | n_runs=3, enable_parallel=True, max_workers=1 | Sequential used ✓ |
| Cap at n_runs | n_runs=3, requested=4 | 3 workers used ✓ |
| Partial use | n_runs=5, requested=2 | 2 workers used ✓ |
| Large request | n_runs=2, requested=10 | Capped to 2 workers ✓ |
| Normal case | n_runs=10, requested=4 | 4 workers used ✓ |

### Syntax Verification
- ✓ `py_compile` passes cleanly on `full_global_search.py`
- ✓ No syntax errors introduced

## Evidence Files
- `.sisyphus/evidence/task-6-worker-cap.txt` — 7 smoke test scenarios
- `.sisyphus/evidence/task-6-single-run.txt` — Single-run bypass verification
- `.sisyphus/evidence/task-6-worker-cap.py` — Smoke test source
- `.sisyphus/evidence/task-6-single-run.py` — Single-run test source

## Documentation Updates
- **Learnings**: Appended Task 6 findings to `.sisyphus/notepads/full-global-search-speedup/learnings.md`
- **Decisions**: Appended Task 6 decisions to `.sisyphus/notepads/full-global-search-speedup/decisions.md`

## Key Decisions

1. **No auto CPU-count capping in Phase 1**: User-provided `max_workers` is the explicit bound. Callers can apply formula `min(n_runs, cpu_count()-1, 4)` before calling `run_global_search()` if desired.

2. **Effective worker cap only where it matters**: Both parallel and sequential paths compute `effective_workers` for clarity, but only parallel path uses it (sequential path always uses 1 worker by definition).

3. **API stability**: No breaking changes to `run_global_search()` signature or parameter semantics. The bound is applied *internally* after parsing `ParallelConfig`.

4. **Native thread limiting deferred**: Environment variable setup (e.g., `OPENBLAS_NUM_THREADS`) and pybind parameter passing documented as Phase 2 work.

## Ready for Task 7
All inputs ready for end-to-end verification:
- ✓ Task 1-5 complete
- ✓ Task 6 worker guardrails in place
- ✓ Code compiles cleanly
- ✓ Logic verified under 7 scenarios
- → Ready for Task 7: full end-to-end correctness, speedup, and diagnostics checks
