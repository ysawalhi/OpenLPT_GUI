## Task 1 Decisions (2026-03-12)

- **ParallelConfig placed before BudgetConfig**: Follows existing dataclass ordering pattern in the file.
- **Three scalar params + ParallelConfig param**: `parallel_config` takes precedence when provided; otherwise individual `enable_parallel`, `max_workers`, `worker_timeout` are used. This gives callers flexibility.
- **`_use_parallel` bool gate**: Computed once at function start, used later by Task 3 for branching. Keeps decision logic centralized.
- **Default `max_workers=1`**: Even with `enable_parallel=True`, having `max_workers=1` forces sequential. Double-safety.
- **`worker_timeout_seconds=7200.0`** (2 hours): Generous default for long CMA-ES runs; can be tuned per-task.
- **Architecture comments in CMA loop**: Minimal, marking exact insertion points for Task 3 without adding any logic.

## Task 2 Decisions (2026-03-12)

- Added `WorkerEvaluationRuntime` plus `initialize_worker_evaluation_runtime(...)` / `get_worker_evaluation_runtime(...)` to enforce one-time worker-local context construction and reuse.
- Added `rebuild_worker_evaluation_runtime(shared_setup)` as the canonical worker-side reconstruction entrypoint; it validates that parent-provided probe scales match reconstructed `ctx.n_params`.
- Kept `run_global_search(...)` behavior unchanged except for preparing serializable `_shared_setup` after probing; no process-pool orchestration was introduced in this task.

## Task 4 Decisions (2026-03-12)

- No additional source changes were needed for Task 4 because Task 3 already sorts completed parallel run results by `run_id` before downstream ranking and diagnostics emission.
- Task 4 verification is satisfied by direct sequential-vs-parallel same-seed smoke comparison plus diagnostics schema comparison.

## Task 3 Decisions (2026-03-12)

- **`_run_cma_worker()` placed at module level after Task 2 helpers (line ~1046)**: Follows natural dependency order — Task 2 runtime helpers → worker entrypoint → `_run_cma_single()`.
- **One future per CMA run**: Each worker executes one complete `_run_cma_single()` call. No sub-parallelism within a run.
- **Pre-computed `run_params` list**: Seeds and per-run budgets calculated before submission to avoid race conditions on `total_evals` counter.
- **`as_completed()` collection with post-sort by `run_id`**: Results arrive in arbitrary order but are sorted before downstream processing for deterministic behavior.
- **Minimal error handling**: `except Exception` logs and continues (`continue`). Failed runs are simply skipped. Full resilience (retry, sequential fallback) deferred to Task 5.
- **Budget tracking in parallel path uses `runs.append` + `total_evals +=`**: Same pattern as sequential path but without inter-run budget guardrails (wall-time, eval budget checks between runs). This is acceptable because all runs are pre-committed at submission time.
- **Sequential path preserved unchanged**: No modifications to the original sequential loop. Gate logic (`_use_parallel`) cleanly separates the two paths.

## Task 4 Decisions (2026-03-12)

- **Parallel completion is staged, then aggregated in `run_id` order**: `as_completed()` results are first buffered by ID; `runs`, `total_evals`, and `budget_status.cumulative_by_run` are populated only from sorted IDs.
- **Single deterministic normalization point before downstream logic**: Added explicit `runs.sort(key=run_id)` and `budget_status.cumulative_by_run.sort(key=run_id)` immediately before dedup/ranking/return.
- **Diagnostics serialization made order-explicit**: `write_generation_csv(...)` and `write_diagnostics_json(...)` now iterate over `sorted(result.runs, key=run_id)` to lock stable output ordering.


## Task 5 Decisions (2026-03-12)

- **Two-level BrokenProcessPool catch**: Inner catch on `future.result()` (marks one run failed, sets `pool_broken=True`, breaks loop), outer catch on `as_completed()` iteration (handles pool death during iteration). Both set `pool_broken=True` to trigger fallback.
- **Serial fallback uses parent-process `ctx` and `scales`**: Since the parent already has a fully constructed `EvaluationContext` and probed scales, fallback runs use `_run_cma_single(ctx, scales, ...)` directly — no worker reconstruction needed.
- **Three dispatch modes in diagnostics**: `cumulative_by_run` entries now include `dispatch_mode` = `'parallel'` | `'serial_fallback'` | `'failed'`. Failed runs get synthetic entries with `n_evals=0`, `best_objective=inf`, and a descriptive `stop_reason`.
- **`budget_status.total_stopped_by` appends pool metadata**: If failures and pool_broken both occur, the string accumulates both pieces of info separated by `'; '`.
- **No retry logic for individual run failures**: If a single run raises `Exception` (not `BrokenProcessPool`), it is logged and skipped. No automatic retry — keeps complexity bounded for Phase 1.
- **Fallback run failures are also tracked**: If `_run_cma_single()` fails during serial fallback, the error is recorded in `failed_run_ids` with a `serial_fallback_` prefix to distinguish from pool-side failures.

## Task 6 Decisions (2026-03-12)

- **Effective worker-count formula**: `effective_workers = min(max_workers, n_runs)` when `max_workers > 1`, else 1. This is simple, deterministic, and reusable by both parallel and sequential paths (though only parallel needs it).
- **Default behavior preserved**: With default `ParallelConfig(enable_parallel=False, max_workers=1)`, `effective_workers` is 1 and `_use_parallel` is False. Sequential execution remains the safe default.
- **No CPU-count auto-capping in Phase 1**: The user-provided `max_workers` is treated as the explicit bound. Formula `min(n_runs, cpu_count()-1, 4)` can be applied by callers before passing to `run_global_search()` if CPU-count awareness is desired.
- **Pool creation skipped conditionally**: Changed from "always skip pool if only 1 run" to "skip pool if `_use_parallel=False`", which now includes the effective_workers check. This is semantically equivalent but more general.
- **Native thread-cap guardrail deferred**: Documented in a comment near the parallel-gate section that native thread limiting (e.g., via `OPENBLAS_NUM_THREADS` or pybind parameters) is not implemented in Phase 1. Will be addressed in Phase 2 if needed.
- **Logging clarity**: Added three distinct log branches — (1) parallel proceeding with bounded workers and full details, (2) parallel requested but bypassed with reasons, (3) parallel disabled with debug-level log. All include `requested_max_workers`, `n_runs`, `effective_workers` where relevant.
- **No breaking changes to API**: `max_workers` parameter retains exact semantics; only the *effective* worker count is now capped at `n_runs`. Existing callers are unaffected.
