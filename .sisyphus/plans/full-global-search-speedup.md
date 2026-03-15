# Full Global Search Speedup

## TL;DR
> **Summary**: Speed up `full_global_search.py` by parallelizing independent CMA-ES runs first, while preserving deterministic results, diagnostics structure, and native C++ camera isolation. Defer within-generation candidate parallelism until worker-local evaluation is proven safe and beneficial.
> **Deliverables**:
> - Add inter-run process parallelism for CMA-ES restarts in `modules/camera_calibration/wand_calibration/full_global_search.py`
> - Preserve current diagnostics JSON/CSV semantics and sequential-equivalent results
> - Add serial fallback, worker failure isolation, and concurrency guardrails
> - Add automated verification for correctness, speedup, and failure handling
> **Effort**: Medium
> **Parallel**: YES - 2 waves
> **Critical Path**: Task 1 -> Task 2 -> Task 3 -> Task 4 -> Task 7

## Context
### Original Request
- Increase the speed of full global search.
- Consider parallelizing runs and generation calculations.
- Ask Metis/Oracle to come up with a plan.

### Interview Summary
- The target is `modules/camera_calibration/wand_calibration/full_global_search.py`.
- Current CMA-ES execution is sequential: outer loop over runs in `run_global_search()`, inner loop over candidates in `_run_cma_single()`.
- The evaluation path touches native C++ camera state through `evaluate_candidate()` -> `opt.evaluate_residuals(...)`, so any concurrency plan must isolate native state per worker.
- Existing repo precedent for safe multiprocessing exists in `modules/camera_calibration/wand_calibration/pretest_global_search.py` using `ProcessPoolExecutor`, Windows `spawn`, worker init, and `BrokenProcessPool` fallback.
- Preserve current objective semantics, result ranking, and diagnostics artifacts.

### Metis Review (gaps addressed)
- Lock Phase 1 to inter-run parallelism only; do not attempt within-generation candidate parallelism yet.
- Fix worker lifecycle up front: worker-local context reconstruction, no pickling of C++ objects.
- Preserve deterministic seed schedule and sequential-equivalent results.
- Define fallback behavior for broken pools and one-run crashes.
- Require automated checks for speedup, correctness, and diagnostics invariance.

## Work Objectives
### Core Objective
Reduce wall time of `full_global_search.py` by parallelizing independent CMA-ES runs safely on Windows, without changing objective semantics, result ordering, or emitted diagnostics structure.

### Deliverables
- `full_global_search.py` supports inter-run parallel execution behind explicit config flags.
- Each worker process constructs its own evaluation context and native C++ camera objects locally.
- Parent process aggregates per-run results identically to current sequential logic.
- Serial fallback handles worker crashes or `BrokenProcessPool` without losing completed runs.
- Verification proves same-seed parallel results match sequential results.

### Definition of Done (verifiable conditions with commands)
- Parallel mode with `n_runs>=2` completes successfully using `conda run -n OpenLPT python -m modules.camera_calibration.wand_calibration.run_full_global_search` with speedup config.
- Sequential and parallel runs with identical seeds produce matching per-run best objectives and generation counts.
- One injected worker failure still yields surviving completed runs and final merged diagnostics.
- Output files remain `full_global_diagnostics.json`, `full_global_eval.csv`, and `full_global_generation.csv` with unchanged schema.

### Must Have
- Windows-safe `spawn` multiprocessing only.
- No pickling of `cams_cpp`, `RefractiveBAOptimizer`, or `EvaluationContext` objects.
- `enable_parallel` and `max_workers` configuration in `full_global_search.py`.
- `n_runs == 1` and `max_workers == 1` bypass pool and run direct sequential path.
- Deterministic seed schedule `seed_base + run_id` preserved.

### Must NOT Have
- No within-generation parallel candidate evaluation in Phase 1.
- No nested parallelism (outer runs + inner candidates simultaneously).
- No cross-process atomic total-eval budget accounting in Phase 1.
- No changes to objective formula, probe scales behavior, candidate ranking, or diagnostics schema.
- No new source-file changes outside `modules/camera_calibration/wand_calibration/full_global_search.py` unless strictly required by import/runtime support already in that file.

## Verification Strategy
> ZERO HUMAN INTERVENTION — all verification is agent-executed.
- Test decision: tests-after + targeted smoke/perf verification in the OpenLPT conda environment
- QA policy: Every task includes executable scenarios with explicit pass/fail checks
- Evidence: `.sisyphus/evidence/task-{N}-{slug}.{ext}`

## Execution Strategy
### Parallel Execution Waves
> Target: 5-8 tasks per wave. <3 per wave (except final) = under-splitting.
> Extract shared dependencies as Wave-1 tasks for max parallelism.

Wave 1: architecture extraction and worker-safe execution primitives
- config and API design
- worker setup serialization design
- worker entrypoints and pool orchestration
- diagnostics merge invariance

Wave 2: guardrails and verification
- serial fallback and failure isolation
- deterministic correctness checks
- speedup measurement harness

### Dependency Matrix (full, all tasks)
- Task 1 blocks Tasks 2, 3, 4, 5, 6, 7
- Task 2 blocks Tasks 3, 4, 5
- Task 3 blocks Tasks 4, 5, 6, 7
- Task 4 blocks Tasks 5, 6, 7
- Task 5 blocks Task 7
- Task 6 blocks Task 7

### Agent Dispatch Summary (wave -> task count -> categories)
- Wave 1 -> 4 tasks -> `unspecified-high`, `deep`
- Wave 2 -> 3 tasks -> `unspecified-high`, `quick`
- Final Verification -> 4 tasks -> `oracle`, `deep`, `unspecified-high`

## TODOs
> Implementation + Test = ONE task. Never separate.
> EVERY task MUST have: Agent Profile + Parallelization + QA Scenarios.

- [x] 1. Lock the Phase-1 parallelization architecture in `full_global_search.py`

  **What to do**: Introduce explicit Phase-1 design decisions in code shape before touching execution flow: inter-run process parallelism only, no inner candidate parallelism, no nested pool usage, and no pickling of native objects. Define new config knobs in `run_global_search()` / helper signatures for `enable_parallel`, `max_workers`, and worker timeout behavior. Ensure the sequential path remains the source-of-truth fallback.
  **Must NOT do**: Do not parallelize `probe_scales()`. Do not change `_run_cma_single()` CMA semantics. Do not add any implementation in other source files.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` — Reason: cross-cutting runtime control flow with native-state constraints
  - Skills: `[]` — no special skill required beyond code reasoning
  - Omitted: `git-master` — no git operation required

  **Parallelization**: Can Parallel: NO | Wave 1 | Blocks: 2,3,4,5,6,7 | Blocked By: none

  **References**:
  - Pattern: `modules/camera_calibration/wand_calibration/full_global_search.py:1862` — current `run_global_search()` orchestration
  - Pattern: `modules/camera_calibration/wand_calibration/full_global_search.py:1589` — current sequential `_run_cma_single()` contract
  - Pattern: `modules/camera_calibration/wand_calibration/pretest_global_search.py:1235` — Windows-safe `spawn` pool usage
  - Pattern: `modules/camera_calibration/wand_calibration/pretest_global_search.py:1288` — `BrokenProcessPool` handling precedent
  - API/Type: `modules/camera_calibration/wand_calibration/full_global_search.py:1439` — `CMARunResult`
  - API/Type: `modules/camera_calibration/wand_calibration/full_global_search.py:767` — `EvaluationContext`

  **Acceptance Criteria**:
  - [ ] `run_global_search()` exposes Phase-1 parallel config flags with defaults preserving current sequential behavior
  - [ ] No code path attempts inner candidate parallelism
  - [ ] No code path attempts to pickle native/C++ objects

  **QA Scenarios**:
  ```text
  Scenario: API preserves sequential default
    Tool: Bash
    Steps: Run `conda run -n OpenLPT python -c "import inspect; from modules.camera_calibration.wand_calibration.full_global_search import run_global_search; print(inspect.signature(run_global_search))"` and verify new flags exist with safe defaults.
    Expected: Signature includes parallel controls; default invocation remains valid.
    Evidence: .sisyphus/evidence/task-1-architecture.txt

  Scenario: No inner parallel pool added
    Tool: Grep
    Steps: Search `modules/camera_calibration/wand_calibration/full_global_search.py` for `ProcessPoolExecutor`, `ThreadPoolExecutor`, `executor.submit`, and confirm usage is limited to outer-run orchestration.
    Expected: No generation-member pool logic appears in `_run_cma_single()`.
    Evidence: .sisyphus/evidence/task-1-architecture-grep.txt
  ```

  **Commit**: NO | Message: `perf(full-global-search): define phase-1 parallel architecture` | Files: [`modules/camera_calibration/wand_calibration/full_global_search.py`]

- [x] 2. Build worker-safe serialized setup and local evaluation-context reconstruction

  **What to do**: Extract a serializable `shared_setup` payload from the parent process after probing and before run dispatch. Add worker-side helpers to rebuild an `EvaluationContext` and native C++ cameras locally inside each spawned worker. Reuse existing loader/build logic from `full_global_search.py` instead of inventing a second code path.
  **Must NOT do**: Do not serialize `EvaluationContext`, `RefractiveBAOptimizer`, `cams_cpp`, or any pybind object. Do not rebuild probe scales independently inside each worker.

  **Recommended Agent Profile**:
  - Category: `deep` — Reason: requires careful separation of serializable and non-serializable runtime state
  - Skills: `[]` — no external skill required
  - Omitted: `playwright` — no browser work

  **Parallelization**: Can Parallel: NO | Wave 1 | Blocks: 3,4,5,6,7 | Blocked By: 1

  **References**:
  - Pattern: `modules/camera_calibration/wand_calibration/full_global_search.py:814` — `build_evaluation_context()` creates native cameras
  - Pattern: `modules/camera_calibration/wand_calibration/full_global_search.py:968` — `evaluate_candidate()` uses shared optimizer/native state
  - Pattern: `modules/camera_calibration/wand_calibration/full_global_search.py:1980` — current parent-side preload flow
  - Pattern: `modules/camera_calibration/wand_calibration/pretest_global_search.py:45` — worker-shared init pattern
  - Caveat: `modules/camera_calibration/wand_calibration/full_global_search.py:1025` — `opt.evaluate_residuals(...)` mutates native state internally

  **Acceptance Criteria**:
  - [ ] Parent passes only plain Python/NumPy serializable data to workers
  - [ ] Each worker rebuilds its own evaluation context and native cameras exactly once
  - [ ] Probe scales from parent are reused unchanged in workers

  **QA Scenarios**:
  ```text
  Scenario: Worker setup is serializable
    Tool: Bash
    Steps: Run a small Python snippet that builds the parent `shared_setup`, pickles it, unpickles it, and reconstructs a worker-local context.
    Expected: Pickle/unpickle succeeds; worker context builds without native-object serialization errors.
    Evidence: .sisyphus/evidence/task-2-worker-setup.txt

  Scenario: Probe scales are reused, not recomputed
    Tool: Grep
    Steps: Search `full_global_search.py` for worker code paths calling `probe_scales(`.
    Expected: Worker path does not call `probe_scales()`.
    Evidence: .sisyphus/evidence/task-2-worker-setup-grep.txt
  ```

  **Commit**: NO | Message: `perf(full-global-search): add worker-local context reconstruction` | Files: [`modules/camera_calibration/wand_calibration/full_global_search.py`]

- [x] 3. Implement outer-run process parallelism with Windows `spawn`

  **What to do**: Replace the sequential run loop in `run_global_search()` with an outer `ProcessPoolExecutor` path using `mp.get_context("spawn")`, one future per CMA run, and `as_completed()` collection. Each worker should execute one full `_run_cma_single()` to completion. Keep the existing sequential loop available when `enable_parallel=False`, `n_runs==1`, or `max_workers==1`.
  **Must NOT do**: Do not run `es.ask()` / `es.tell()` across process boundaries. Do not parallelize inside `_run_cma_single()`.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` — Reason: multiprocessing orchestration with Windows constraints
  - Skills: `[]`
  - Omitted: `frontend-ui-ux` — irrelevant

  **Parallelization**: Can Parallel: NO | Wave 1 | Blocks: 4,5,6,7 | Blocked By: 1,2

  **References**:
  - Pattern: `modules/camera_calibration/wand_calibration/pretest_global_search.py:1257` — `ProcessPoolExecutor` + `spawn`
  - Pattern: `modules/camera_calibration/wand_calibration/full_global_search.py:2083` — current sequential run loop over `n_runs`
  - Pattern: `modules/camera_calibration/wand_calibration/full_global_search.py:1685` — generation stays sequential inside run
  - External: Oracle recommendation in session `ses_31d8506f4ffeqRS74viPhknd6j` — inter-run parallelism first, no inner parallelism

  **Acceptance Criteria**:
  - [ ] Parallel mode dispatches up to `max_workers` independent CMA runs concurrently
  - [ ] Sequential mode remains available and functionally identical to current behavior
  - [ ] On Windows, the implementation uses `spawn` context explicitly

  **QA Scenarios**:
  ```text
  Scenario: Parallel dispatch runs multiple seeds concurrently
    Tool: Bash
    Steps: Run a smoke test with `n_runs=3`, `enable_parallel=True`, `max_workers=2`, small budgets, and inspect logs/results.
    Expected: Runs start concurrently, all complete, and merged result contains three run records.
    Evidence: .sisyphus/evidence/task-3-outer-parallel.txt

  Scenario: Sequential fallback path still works
    Tool: Bash
    Steps: Run identical smoke config with `enable_parallel=False` and then with `max_workers=1`.
    Expected: Both complete successfully and use the direct sequential path.
    Evidence: .sisyphus/evidence/task-3-outer-parallel-fallback.txt
  ```

  **Commit**: NO | Message: `perf(full-global-search): parallelize cma restarts across processes` | Files: [`modules/camera_calibration/wand_calibration/full_global_search.py`]

- [x] 4. Preserve deterministic seeds, result ordering, and diagnostics aggregation

  **What to do**: Guarantee that parallel execution preserves per-run seed schedule (`seed_base + run_id`), stable run identifiers, and the same merge/dedup semantics currently used after sequential runs finish. Aggregate completed `CMARunResult` objects in deterministic `run_id` order before downstream ranking, dedup, and diagnostics emission.
  **Must NOT do**: Do not let `as_completed()` arrival order change output ordering. Do not change candidate dedup thresholds or ranking rules.

  **Recommended Agent Profile**:
  - Category: `deep` — Reason: deterministic behavior and result-equivalence requirements
  - Skills: `[]`
  - Omitted: `git-master`

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: 7 | Blocked By: 1,2,3

  **References**:
  - Pattern: `modules/camera_calibration/wand_calibration/full_global_search.py:2150` — current seed schedule per run
  - Pattern: `modules/camera_calibration/wand_calibration/full_global_search.py:1829` — top-candidate dedup depends on run results
  - Pattern: `modules/camera_calibration/wand_calibration/full_global_search.py:2211` — global best merge logic
  - Pattern: `modules/camera_calibration/wand_calibration/full_global_search.py:2585` — diagnostics writing expects stable run/candidate structures

  **Acceptance Criteria**:
  - [ ] Parallel and sequential execution with the same seeds produce matching per-run best objectives and generation counts
  - [ ] Output JSON/CSV schemas remain unchanged
  - [ ] Run records in diagnostics remain sorted by `run_id`

  **QA Scenarios**:
  ```text
  Scenario: Sequential vs parallel equivalence
    Tool: Bash
    Steps: Run one sequential smoke test and one parallel smoke test with identical seeds and budgets; compare per-run `best_objective`, `n_generations`, and `stop_reason`.
    Expected: Matching per-run results within floating-point tolerance.
    Evidence: .sisyphus/evidence/task-4-determinism.txt

  Scenario: Diagnostics schema invariance
    Tool: Bash
    Steps: Read emitted JSON/CSV from both modes and compare top-level keys and CSV headers.
    Expected: Same schema in both modes.
    Evidence: .sisyphus/evidence/task-4-diagnostics.txt
  ```

  **Commit**: NO | Message: `perf(full-global-search): preserve deterministic result aggregation` | Files: [`modules/camera_calibration/wand_calibration/full_global_search.py`]

- [x] 5. Add pool failure isolation and serial fallback

  **What to do**: Catch run-level exceptions, `BrokenProcessPool`, and worker timeouts without discarding successful completed runs. If the pool breaks, finish remaining pending runs sequentially in the parent process using the same seed schedule and budgets. Report failure metadata clearly in logs and diagnostics.
  **Must NOT do**: Do not abort the whole search because one run fails. Do not silently swallow exceptions without logging `run_id` and reason.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` — Reason: robust runtime recovery on Windows multiprocessing
  - Skills: `[]`
  - Omitted: `playwright`

  **Parallelization**: Can Parallel: YES | Wave 2 | Blocks: 7 | Blocked By: 3

  **References**:
  - Pattern: `modules/camera_calibration/wand_calibration/pretest_global_search.py:1288` — worker pool break handling
  - Pattern: `modules/camera_calibration/wand_calibration/pretest_global_search.py:1305` — propagate readable pool error text
  - Pattern: `modules/camera_calibration/wand_calibration/full_global_search.py:2196` — current run-result logging hook

  **Acceptance Criteria**:
  - [ ] One failed run still allows other runs to complete and be merged
  - [ ] `BrokenProcessPool` triggers serial fallback for unfinished runs
  - [ ] Logs and final diagnostics clearly indicate failed/fallback runs

  **QA Scenarios**:
  ```text
  Scenario: One run crashes, others survive
    Tool: Bash
    Steps: Add a temporary fault injection guard keyed by `run_id` in a local test branch of the execution path and run with `n_runs=3`.
    Expected: Two non-failing runs complete; diagnostics still emit merged output with failure noted.
    Evidence: .sisyphus/evidence/task-5-failure-isolation.txt

  Scenario: Pool break falls back to serial
    Tool: Bash
    Steps: Simulate `BrokenProcessPool` during smoke execution.
    Expected: Remaining runs execute sequentially and final artifacts are written.
    Evidence: .sisyphus/evidence/task-5-serial-fallback.txt
  ```

  **Commit**: NO | Message: `perf(full-global-search): add resilient parallel fallback handling` | Files: [`modules/camera_calibration/wand_calibration/full_global_search.py`]

- [x] 6. Control oversubscription and worker resource usage

  **What to do**: Add explicit worker-count selection logic (`min(n_runs, cpu_count()-1, 4)` by default) and set native thread limits inside each worker if the current stack uses threaded native libs. Ensure pool startup is skipped when only one run is requested. Log chosen concurrency and any downgraded worker count.
  **Must NOT do**: Do not allow unbounded `cpu_count()` workers by default. Do not combine outer-run process parallelism with inner native thread fan-out blindly.

  **Recommended Agent Profile**:
  - Category: `quick` — Reason: bounded runtime-config guardrails
  - Skills: `[]`
  - Omitted: `oracle`

  **Parallelization**: Can Parallel: YES | Wave 2 | Blocks: 7 | Blocked By: 3

  **References**:
  - Oracle recommendation in `ses_31d8506f4ffeqRS74viPhknd6j` — cap workers and native threads to avoid oversubscription
  - Pattern: `modules/camera_calibration/wand_calibration/full_global_search.py:1638` — current popsize budget capping logic can guide worker cap style

  **Acceptance Criteria**:
  - [ ] Default worker count is bounded and logged
  - [ ] `n_runs==1` or `max_workers==1` skips pool creation
  - [ ] Native thread-cap guardrail is present or explicitly documented/logged as unsupported

  **QA Scenarios**:
  ```text
  Scenario: Worker count auto-caps correctly
    Tool: Bash
    Steps: Run a small script that prints chosen `max_workers` under `n_runs=1`, `n_runs=3`, and oversized requested worker counts.
    Expected: Chosen worker count follows plan defaults and never exceeds `n_runs`.
    Evidence: .sisyphus/evidence/task-6-worker-cap.txt

  Scenario: Single-run bypass path
    Tool: Bash
    Steps: Run `n_runs=1`, `enable_parallel=True` smoke config.
    Expected: Code logs direct sequential path and does not create a pool.
    Evidence: .sisyphus/evidence/task-6-single-run.txt
  ```

  **Commit**: NO | Message: `perf(full-global-search): add concurrency guardrails` | Files: [`modules/camera_calibration/wand_calibration/full_global_search.py`]

- [x] 7. Verify correctness, speedup, and diagnostics invariance end-to-end

  **What to do**: Run targeted smoke/perf validation in the OpenLPT conda environment. Compare sequential vs parallel on identical seeds, verify wall-time improvement for `n_runs>=2`, check merged diagnostics invariance, and validate failure isolation behavior. Record evidence for each scenario.
  **Must NOT do**: Do not rely on manual inspection only. Do not declare success without sequential-vs-parallel equivalence checks.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` — Reason: multi-scenario runtime verification
  - Skills: `[]`
  - Omitted: `frontend-ui-ux`

  **Parallelization**: Can Parallel: NO | Wave 2 | Blocks: none | Blocked By: 4,5,6

  **References**:
  - Environment: `AGENTS.md` — must use `conda run -n OpenLPT python ...`
  - Pattern: `modules/camera_calibration/wand_calibration/run_full_global_search.py` — existing smoke-run harness
  - Output artifacts: `J:\Fish\T0\Full_Global_Search\full_global_diagnostics.json`, `J:\Fish\T0\Full_Global_Search\full_global_eval.csv`, `J:\Fish\T0\Full_Global_Search\full_global_generation.csv`

  **Acceptance Criteria**:
  - [ ] Parallel mode completes successfully with `n_runs>=2`
  - [ ] Sequential and parallel same-seed smoke runs match per-run best objective and generation count
  - [ ] Parallel wall time is at least 30% lower than sequential for a representative `n_runs>=2` smoke configuration
  - [ ] Diagnostics files are emitted with unchanged schema and expected row counts

  **QA Scenarios**:
  ```text
  Scenario: Sequential vs parallel correctness
    Tool: Bash
    Steps: Run two smoke tests with identical seeds and budgets, one sequential and one parallel; compare JSON summaries and per-run entries.
    Expected: Matching run outcomes within tolerance.
    Evidence: .sisyphus/evidence/task-7-correctness.txt

  Scenario: Measured wall-time speedup
    Tool: Bash
    Steps: Run representative `n_runs=3` smoke/perf configs in both modes and record elapsed wall time.
    Expected: Parallel wall time < 0.7 * sequential wall time.
    Evidence: .sisyphus/evidence/task-7-speedup.txt
  ```

  **Commit**: NO | Message: `test(full-global-search): verify inter-run parallel speedup and invariance` | Files: [`modules/camera_calibration/wand_calibration/full_global_search.py`]

## Final Verification Wave (4 parallel agents, ALL must APPROVE)
- [ ] F1. Plan Compliance Audit — oracle
- [ ] F2. Code Quality Review — unspecified-high
- [ ] F3. Real Manual QA — unspecified-high
- [ ] F4. Scope Fidelity Check — deep

## Commit Strategy
- No commits during planning.
- During implementation, keep changes in `modules/camera_calibration/wand_calibration/full_global_search.py` only unless runtime support proves impossible without touching the runner harness; if that exception is required, document it explicitly before implementation.
- Recommended implementation commit split:
  - `perf(full-global-search): parallelize cma restarts across processes`
  - `perf(full-global-search): add fallback and determinism guardrails`
  - `test(full-global-search): verify parallel correctness and speedup`

## Success Criteria
- Full global search uses safe process-level inter-run parallelism on Windows.
- Native camera/C++ state is isolated per worker process.
- Sequential and parallel modes produce equivalent same-seed results.
- Parallel mode reduces wall time materially for multi-run searches.
- Diagnostics and result-selection behavior remain unchanged.
