## Task 1 Learnings (2026-03-12)

- `full_global_search.py` is ~2900+ lines. The CMA-ES run loop starts around line 2180+.
- `_run_cma_single()` contains the inner ask/evaluate/tell loop; it must stay sequential per Phase-1 constraints.
- `run_global_search()` already had a `BudgetConfig` dataclass pattern — `ParallelConfig` follows the same style.
- The `conda run -n OpenLPT` command on Windows has heavy VS dev environment init overhead (~30s). Using `C:/Users/tan_s/miniconda3/envs/OpenLPT/python.exe` directly is much faster for verification.
- The three-way parallel gate (`enable_parallel and max_workers > 1 and n_runs > 1`) ensures any single falsy condition falls back to sequential.

## Task 2 Learnings (2026-03-12)

- `build_shared_setup(...)` should package only `ref_state`, `dataset`, scalar config values, and parent-probed `probe_scales`; this keeps payload pickle-safe and avoids pybind/native-object leakage.
- Worker reconstruction can safely reuse `build_evaluation_context(...)` directly; this guarantees native C++ cameras are created in the worker process without introducing a second evaluation path.
- A worker-global runtime cache (`initialize_worker_evaluation_runtime`) is a simple way to guarantee one-time native context creation per worker lifecycle.

## Task 3 Learnings (2026-03-12)

- `_run_cma_worker()` must be a module-level function (not closure/lambda) for `spawn`-context pickle compatibility.
- `inspect.signature()` returns string annotations (e.g., `'CMARunResult'`) when `from __future__ import annotations` is active, not the class object itself. Test assertions must handle both.
- The parallel branch pre-computes `run_params` (seeds, budgets) before submission because budget math depends on `total_evals` which is known upfront for all runs.
- `as_completed()` returns results in non-deterministic order; sorting by `run_id` after collection ensures deterministic output regardless of worker finish order.
- `BrokenProcessPool` exceptions are caught implicitly via the `except Exception` around `future.result()`; full resilience (retry, fallback) is deferred to Task 5.
- Existing smoke tests only verify structural correctness (signatures, dataclass fields) since end-to-end testing requires real calibration data.

## Task 4 Learnings (2026-03-12)

- Sorting only after `as_completed()` is not enough if budget/diagnostic aggregation already consumed completion-order events; aggregate per-run diagnostics only after deterministic `run_id` ordering is established.
- Enforcing deterministic order once before downstream consumers (`runs.sort(run_id)` right before dedup/ranking/return) provides a single invariant for sequential and parallel modes.
- Diagnostics writers should not rely on caller ordering assumptions; iterating `sorted(result.runs, key=run_id)` in JSON/generation CSV emission guarantees stable serialization.

## Task 4 Learnings (2026-03-12)

- The deterministic `runs.sort(key=lambda r: r.run_id)` added in the Task 3 parallel branch is already sufficient for Task 4 acceptance on same-seed smoke tests.
- Same-seed sequential vs parallel smoke runs matched exactly on `(run_id, best_objective, n_generations, stop_reason)`.
- Diagnostics schema stayed invariant across modes: JSON top-level keys matched, and eval/generation CSV headers matched exactly.


## Task 5 Learnings (2026-03-12)

- `BrokenProcessPool` can be raised in two places: (1) inside `future.result()` when the worker that produced that future crashed, and (2) during the `as_completed()` iteration itself when the pool dies mid-iteration. Both cases need separate try/except blocks.
- The pretest pattern (`pretest_global_search.py:1288-1321`) handles broken pools per-generation with `_serial_retry()`. The full global search version operates at a higher granularity (entire CMA runs), so fallback uses `_run_cma_single()` directly in the parent process with the parent's already-constructed `ctx` and `scales`.
- Failed run tracking (`failed_run_ids: Dict[int, str]`) maps run_id to error description for clear diagnostics. This is richer than just logging and allows downstream consumers to see exactly which runs failed.
- The serial fallback uses the same `run_params` pre-computed before pool creation, ensuring identical seeds and budgets. No recalculation needed.
- Adding `dispatch_mode` to `cumulative_by_run` entries is a non-breaking schema extension — existing consumers that don't look for this key are unaffected.
- The `pool_broken` flag is separate from individual run failures because pool death affects all pending futures, not just one run.

## Task 6 Learnings (2026-03-12)

- **Worker-count guardrail logic**: Effective worker count is computed as `min(max_workers, n_runs)` when `max_workers > 1`, otherwise defaults to 1. This prevents oversubscription and ensures no more workers than available runs.
- **Pool-creation gate**: Pool is skipped when `_use_parallel=False`, which happens when ANY of: `enable_parallel=False`, `effective_workers <= 1`, or `n_runs <= 1`. This preserves sequential behavior as the default.
- **Logging clarity**: New log messages distinguish between (1) parallel execution proceeding with bounded workers, (2) parallel requested but bypassed with reasons, and (3) parallel disabled entirely. Each includes `requested_max_workers`, `n_runs`, and `effective_workers` for debugging.
- **Three-way gate suffices**: No need for separate `cpu_count()` capping in Phase 1; the requested `max_workers` is already a user-provided bound. The formula `min(n_runs, cpu_count()-1, 4)` can be applied by callers *before* invoking `run_global_search()` if desired.
- **Native thread-cap guardrail**: Documented as unsupported in Phase 1 (note in docstring). Within-worker thread limits would require either environment variable setup (e.g., `OPENBLAS_NUM_THREADS`) or pybind parameter passing, both deferred to Phase 2.
- **Smoke test validation**: All seven worker-selection scenarios passed (single-run bypass, disabled parallel, explicit max_workers=1, various multi-run caps, and asymmetric requests). Pool is correctly skipped for n_runs=1 even with enable_parallel=True.
- **Syntax verified**: Full module compiles cleanly with `py_compile`; no syntax errors introduced.
