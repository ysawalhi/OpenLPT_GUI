# Parallelize Stage 1 and Stage 2 Multidimensional Probing

## TL;DR
> **Summary**: Add parallel execution to Stage 1 and Stage 2 multidimensional probing using auto-detected worker count (80% of CPU cores, capped at 32)
> **Deliverables**: Parallel parameter probing (Stage 1), parallel block probing (Stage 2), with automatic fallback to sequential for small problems
> **Effort**: Medium
> **Parallel**: NO (sequential dependencies between phases)
> **Critical Path**: Phase 0 → Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 5

## Context

### Original Request
User requested parallelization of multidimensional probing in camera calibration to reduce wall-time from ~9-15 hours to ~30-50 minutes. Specifically targeting Stage 1 (`probe_scales_multidim_stage1`) and Stage 2 (`probe_scales_multidim_stage2_blocks`) functions in `full_global_search.py`.

### Interview Summary
- **Scope**: Both Stage 1 and Stage 2 probing functions
- **Parallelization level**: Parameter/block level (NOT alpha steps - keep sequential)
- **Worker count**: Auto-detect using `os.cpu_count() * 0.8`, capped at 32, no CLI parameters
- **Memory strategy**: Start aggressive, let system handle it
- **Reference eval strategy**: Option B — compute `ray_rmse_ref` once in main process, pass to workers
- **Verified optimization**: Ray RMSE already uses batch processing (`build_pinplate_rays_cpp_batch` → C++ `lineOfSightBatchStatus`)

### Metis Review (Critical Issues Fixed)
Metis identified two critical issues that this revised plan addresses:

1. **Issue**: `_build_probe_shared_setup()` tried to access non-existent optimizer attributes
   - **Fix**: Build `shared_setup` at caller site in `run_global_search()` where `ref_state` and `dataset` are already in scope

2. **Issue**: Caller sites in `run_global_search()` were not modified (lines 3690, 3711)
   - **Fix**: Added Phase 0 to modify caller sites and pass `shared_setup` to probing functions

## Work Objectives

### Core Objective
Parallelize Stage 1 and Stage 2 multidimensional probing to achieve 10-15× speedup on typical multi-core systems while maintaining numerical equivalence with sequential execution.

### Deliverables
1. Auto-detection of worker count (80% of CPU cores, capped at 32)
2. Worker initializer function to suppress BLAS threading
3. Stage 1 worker function for single-parameter probing
4. Stage 2 worker function for single-block probing
5. Parallel execution in `probe_scales_multidim_stage1()`
6. Parallel execution in `probe_scales_multidim_stage2_blocks()`
7. Caller site modifications in `run_global_search()` to build and pass `shared_setup`
8. Verification tests confirming numerical equivalence

### Definition of Done
- [ ] Both probing functions detect worker count automatically
- [ ] Parallel execution matches sequential results (verified via `np.allclose` with `rtol=1e-10`)
- [ ] Worker count logged at start of probing
- [ ] Falls back to sequential when items < 2 × workers
- [ ] Memory usage stays within system limits on multi-core machines
- [ ] No changes to function signatures or CLI parameters
- [ ] Caller sites in `run_global_search()` properly pass `shared_setup`

### Must Have
- Auto-detection code using: `max_workers = min(32, max(1, int((os.cpu_count() or 1) * 0.8)))`
- Worker functions as module-level functions (required for `spawn` context)
- Per-worker context reconstruction via `build_shared_setup()` + `initialize_worker_evaluation_runtime()`
- BLAS thread suppression: `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`
- %-formatting in logger calls (NOT f-strings)
- Preserve `_restore_optimizer_native_reference_state()` in finally blocks
- `shared_setup` built at caller site and passed to probing functions

### Must NOT Have
- Hardcoded core counts (no assumptions about 24 cores)
- Threading (`ThreadPoolExecutor` or `joblib` threading)
- CLI parameters for worker count (auto-detect only)
- Function signature changes to public API
- Modifications to evaluation logic or compensation algorithm
- Removal of sequential code path

## Verification Strategy
> All verification is agent-executed. No human intervention required.

**Test Decision**: Tests-after (write verification scripts after implementation)
**QA Policy**: Every phase has agent-executed acceptance criteria with concrete commands

**Evidence Location**: `.sisyphus/evidence/parallelize-probing-*.{txt,log}`

## Execution Strategy

### Sequential Execution (NO parallel waves)
All phases depend on previous phases completing successfully. Implementation must proceed in order.

**Dependency Chain**:
Phase 0 (caller sites) → Phase 1 (worker helpers) → Phase 2 (Stage 1 worker) → Phase 3 (Stage 2 worker) → Phase 4 (parallel dispatch) → Phase 5 (verification)

### Agent Dispatch Summary
- Phase 0-3: `quick` category (code extraction and integration)
- Phase 4: `quick` category (parallel dispatch integration)
- Phase 5: `unspecified-high` category (verification and testing)

## TODOs

---

### Phase 0: Modify Caller Sites in `run_global_search()`

- [x] 0. Modify `run_global_search()` to build and pass `shared_setup`

  **What to do**:
  1. Locate Stage 1 call in `run_global_search()` at line ~3690:
     ```python
     result = probe_scales_multidim_stage1(...)
     ```
  2. Before this call, add code to build `shared_setup`:
     ```python
     import os
     shared_setup = build_shared_setup(
         ref_state,
         dataset,
         np.ones(ctx.n_params, dtype=np.float64),  # placeholder, ctx.n_params is available at caller site
         wand_length=wand_length,
         lambda_base_per_cam=lambda_base_per_cam,
         max_frames=max_frames,
         dist_coeff_num=dist_coeff_num,
     )
     ```
  3. Modify the Stage 1 call to pass `shared_setup`:
     ```python
     result = probe_scales_multidim_stage1(
         ...,  # keep existing params
         shared_setup=shared_setup,
     )
     ```
  4. Repeat steps 2-3 for Stage 2 call at line ~3711 (same `shared_setup` can be reused)
  5. Handle the case where `shared_setup` is `None` (fallback to sequential)
  6. Document third caller site (~line 2779) as intentional sequential-only wrapper

  **Must NOT do**:
  - Change function signatures of probing functions (add `shared_setup` as optional kwarg with default `None`)
  - Remove any existing parameters
  - Modify the logic of `run_global_search()`

  **Recommended Agent Profile**:
  - Category: `quick` — Simple parameter addition
  - Skills: [] — No special skills needed

  **Parallelization**: Can Parallel: NO | Wave 0 | Blocks: [1, 2, 3, 4, 5] | Blocked By: []

  **References**:
  - Pattern: `full_global_search.py:3845` — How CMA-ES builds `shared_setup`
  - Pattern: `full_global_search.py:3886-3920` — How CMA-ES passes context to workers
  - API: `full_global_search.py:934-975` (`build_shared_setup`) — Signature and usage
  - Location: `full_global_search.py:3690` and `3711` — Caller sites to modify

  **Acceptance Criteria**:
  - [ ] `shared_setup` is built before Stage 1 call at line ~3690
  - [ ] Call signature matches `build_shared_setup(ref_state, dataset, probe_scales, *, wand_length, ...)`
  - [ ] `probe_scales` uses `np.ones(ctx.n_params, dtype=np.float64)` (available at caller site)
  - [ ] Both probing function signatures accept `shared_setup=None` kwarg
  - [ ] Fallback to sequential works when `shared_setup is None`

  **QA Scenarios**:
  ```
  Scenario: shared_setup is built correctly
    Tool: Bash
    Steps:
      grep -n "shared_setup = build_shared_setup" modules/camera_calibration/wand_calibration/full_global_search.py | grep -B5 "probe_scales_multidim_stage1" && echo "PASS: shared_setup built"
    Expected: Output contains "PASS: shared_setup built"
    Evidence: .sisyphus/evidence/task-0-shared-setup-built.txt

  Scenario: Both probing functions receive shared_setup
    Tool: Bash
    Steps:
      grep -n "probe_scales_multidim_stage1.*shared_setup=" modules/camera_calibration/wand_calibration/full_global_search.py && grep -n "probe_scales_multidim_stage2_blocks.*shared_setup=" modules/camera_calibration/wand_calibration/full_global_search.py && echo "PASS: Both functions get shared_setup"
    Expected: Output contains "PASS: Both functions get shared_setup"
    Evidence: .sisyphus/evidence/task-0-both-get-setup.txt
  ```

  **Commit**: YES | Message: `refactor(calibration): prepare caller sites for parallel probing` | Files: [`modules/camera_calibration/wand_calibration/full_global_search.py`]

---

### Phase 1: Create Helper Functions

- [x] 1. Create BLAS thread suppression initializer

  **What to do**:
  1. Create module-level function `_init_probing_worker()` before any probing functions:
     ```python
     def _init_probing_worker():
         """Suppress BLAS multi-threading in worker processes."""
         import os
         os.environ['OMP_NUM_THREADS'] = '1'
         os.environ['MKL_NUM_THREADS'] = '1'
         os.environ['OPENBLAS_NUM_THREADS'] = '1'
     ```
  2. Place this function around line 900 (before probing functions)

  **Must NOT do**:
  - Modify global environment (only in worker processes)
  - Change any existing code

  **Recommended Agent Profile**:
  - Category: `quick` — Simple function addition
  - Skills: []

  **Parallelization**: Can Parallel: NO | Wave 1 | Blocks: [2, 3, 4, 5] | Blocked By: [0]

  **References**:
  - Pattern: `full_global_search.py:1046-1090` — Worker function structure
  - Reason: Prevent BLAS libraries from over-subscribing with threads

  **Acceptance Criteria**:
  - [ ] Function `_init_probing_worker` exists as module-level function
  - [ ] Function sets all three environment variables
  - [ ] Function is placed before probing worker functions

  **QA Scenarios**:
  ```
  Scenario: Initializer function exists
    Tool: Bash
    Steps:
      conda run -n OpenLPT python -c "from modules.camera_calibration.wand_calibration.full_global_search import _init_probing_worker; print('PASS: Initializer exists')"
    Expected: Output contains "PASS: Initializer exists"
    Evidence: .sisyphus/evidence/task-1-initializer-exists.txt

  Scenario: Initializer sets environment variables
    Tool: Bash
    Steps:
      grep -A7 "def _init_probing_worker" modules/camera_calibration/wand_calibration/full_global_search.py | grep -q "OMP_NUM_THREADS.*1" && echo "PASS: Sets env vars"
    Expected: Output contains "PASS: Sets env vars"
    Evidence: .sisyphus/evidence/task-1-env-vars.txt
  ```

  **Commit**: YES | Message: `perf(calibration): add BLAS thread suppression for probing workers` | Files: [`modules/camera_calibration/wand_calibration/full_global_search.py`]

---

### Phase 2: Create Stage 1 Worker Function

- [x] 2. Create `_probe_stage1_single_param()` worker function

  **What to do**:
  1. Create module-level function at line ~1900 (before `probe_scales_multidim_stage1`):
     ```python
     def _probe_stage1_single_param(shared_setup, param_idx, layout_entry, base_h, ray_rmse_ref, ray_stop_threshold, ...):
         """Worker function for single-parameter Stage 1 probing."""
         runtime = initialize_worker_evaluation_runtime(shared_setup)
         ctx = runtime.ctx
         # Extract the alpha-expansion loop body from probe_scales_multidim_stage1()
         # ... perform probing for this single parameter ...
         return {
             'param_idx': param_idx,
             'scale': scale_value,
             'sensitivity': sensitivity_value,
             'stop_reason': stop_reason_str,
             'n_evals': n_evals_int,
         }
     ```
  2. Extract inner loop logic from `probe_scales_multidim_stage1()` (lines ~2055-2114)
  3. Worker should:
     - Call `runtime = initialize_worker_evaluation_runtime(shared_setup)` first, then extract `ctx = runtime.ctx`
     - Use `ray_rmse_ref` and `ray_stop_threshold` passed as arguments (Option B strategy)
     - Perform alpha expansion for a single parameter
     - Return dict with computed values
     - Use try/finally to restore optimizer state

  **Must NOT do**:
  - Change evaluation logic
  - Use f-strings in logger calls
  - Modify compensation algorithm

  **Recommended Agent Profile**:
  - Category: `quick` — Code extraction from existing loop
  - Skills: []

  **Parallelization**: Can Parallel: NO | Wave 2 | Blocks: [3, 4, 5] | Blocked By: [1]

  **References**:
  - Pattern: `full_global_search.py:1046-1090` (`_run_cma_worker`) — Worker structure to follow
  - Pattern: `full_global_search.py:2055-2114` — Alpha loop to extract
  - API: `full_global_search.py:1017-1043` (`initialize_worker_evaluation_runtime`)

  **Acceptance Criteria**:
  - [ ] Function `_probe_stage1_single_param` exists as module-level function
  - [ ] Function signature includes `ray_rmse_ref` and `ray_stop_threshold` parameters (Option B pass-through)
  - [ ] Function returns dict with keys: `param_idx`, `scale`, `sensitivity`, `stop_reason`, `n_evals`
  - [ ] Pattern: `runtime = initialize_worker_evaluation_runtime(shared_setup); ctx = runtime.ctx` (NOT `ctx =` alone)
  - [ ] Uses %-formatting in logger calls

  **QA Scenarios**:
  ```
  Scenario: Worker function exists
    Tool: Bash
    Steps:
      conda run -n OpenLPT python -c "from modules.camera_calibration.wand_calibration.full_global_search import _probe_stage1_single_param; print('PASS: Function exists')"
    Expected: Output contains "PASS: Function exists"
    Evidence: .sisyphus/evidence/task-2-worker-exists.txt
  ```

  **Commit**: YES | Message: `feat(calibration): extract Stage 1 single-param worker function` | Files: [`modules/camera_calibration/wand_calibration/full_global_search.py`]

---

### Phase 3: Create Stage 2 Worker Function

- [x] 3. Create `_probe_stage2_single_block()` worker function

  **What to do**:
  1. Create module-level function (similar to Phase 2):
     ```python
     def _probe_stage2_single_block(shared_setup, block_idx, block, ray_rmse_ref, ray_stop_threshold, ...):
         """Worker function for single-block Stage 2 probing."""
         runtime = initialize_worker_evaluation_runtime(shared_setup)
         ctx = runtime.ctx
         # Extract block probe logic from probe_scales_multidim_stage2_blocks()
         # ... perform probing for this single block ...
         return {
             'block_idx': block_idx,
             'block_scales': block_scales_array,
             'stop_reason': stop_reason_str,
             'n_evals': n_evals_int,
         }
     ```
  2. Extract inner logic from `probe_scales_multidim_stage2_blocks()` (lines ~2438-2538)
  3. Same structure as Stage 1 worker (use Option B pass-through of `ray_rmse_ref`, `ray_stop_threshold`)

  **Must NOT do**:
  - Change evaluation logic
  - Remove sequential loop over directions/alpha steps
  - Use f-strings in logger calls

  **Recommended Agent Profile**:
  - Category: `quick` — Repeat Phase 2 pattern
  - Skills: []

  **Parallelization**: Can Parallel: NO | Wave 3 | Blocks: [4, 5] | Blocked By: [2]

  **References**:
  - Pattern: `_probe_stage1_single_param` — Worker structure (created in Phase 2)
  - Pattern: `full_global_search.py:2438-2538` — Block loop to extract
  - API: `numpy.maximum` — For merging block_scales from workers

  **Acceptance Criteria**:
  - [ ] Function `_probe_stage2_single_block` exists as module-level function
  - [ ] Function signature includes `ray_rmse_ref` and `ray_stop_threshold` parameters (Option B pass-through)
  - [ ] Function returns dict with keys: `block_idx`, `block_scales`, `stop_reason`, `n_evals`
  - [ ] Pattern: `runtime = initialize_worker_evaluation_runtime(shared_setup); ctx = runtime.ctx`
  - [ ] Uses %-formatting in logger calls
  **QA Scenarios**:
  ```
  Scenario: Stage 2 worker function exists
    Tool: Bash
    Steps:
      conda run -n OpenLPT python -c "from modules.camera_calibration.wand_calibration.full_global_search import _probe_stage2_single_block; print('PASS: Stage 2 worker exists')"
    Expected: Output contains "PASS: Stage 2 worker exists"
    Evidence: .sisyphus/evidence/task-3-stage2-worker-exists.txt
  ```

  **Commit**: YES | Message: `feat(calibration): extract Stage 2 single-block worker function` | Files: [`modules/camera_calibration/wand_calibration/full_global_search.py`]

---

### Phase 4: Add Parallel Dispatch

- [x] 4. Add parallel dispatch to `probe_scales_multidim_stage1()` and `probe_scales_multidim_stage2_blocks()`

   **What to do**:
  1. In `probe_scales_multidim_stage1()` (after line 1947):
     - Add function signature: accept `shared_setup=None` kwarg
     - Compute reference eval ONCE in main process before worker dispatch:
       ```python
       ray_rmse_ref = None
       ray_stop_threshold = None
       if shared_setup is not None:
           ref_result = evaluate_candidate(None, ctx)  # Evaluate reference
           ray_rmse_ref = float(ref_result['ray_rmse'])
           ray_stop_threshold = float(max(ray_rmse_stop_factor, 1.0) * ray_rmse_ref)
       ```
     - Add auto-detection at start:
       ```python
       import os
       _cpu = os.cpu_count() or 1
       max_workers = min(32, max(1, int(_cpu * 0.8)))
       n_params = len(layout.entries)
       use_parallel = (shared_setup is not None and
                       max_workers > 1 and
                       n_params >= 2 * max_workers)
       logger.info('probe_scales_multidim_stage1: %d params, cpu_count=%d, max_workers=%d, parallel=%s',
                   n_params, _cpu, max_workers, use_parallel)
       ```
     - Wrap existing sequential loop in `if not use_parallel:` block
     - Add parallel path using `ProcessPoolExecutor` with `spawn` context and `_init_probing_worker` initializer
     - When submitting workers, pass `ray_rmse_ref` and `ray_stop_threshold` (Option B):
       ```python
       if use_parallel:
           with ProcessPoolExecutor(max_workers=max_workers,
                                    mp_context=mp.get_context('spawn'),
                                    initializer=_init_probing_worker) as executor:
               futures = {executor.submit(_probe_stage1_single_param, shared_setup, i, layout_entry, base_h,
                                                                     ray_rmse_ref, ray_stop_threshold, ...): i
                          for i in range(n_params)}
               for future in as_completed(futures):
                   result = future.result()
                   # ... aggregate result ...
       else:
           # ... existing sequential loop ...
       ```
  2. Repeat for `probe_scales_multidim_stage2_blocks()` at line ~2355
     - Stage 2 results must be aggregated using `np.maximum` for block_scales merging:
       ```python
       block_scales = np.zeros(n, dtype=np.float64)  # Initialize accumulator
       for future in as_completed(futures):
           result = future.result()
           block_scales = np.maximum(block_scales, result['block_scales'])
       ```
  3. Note about third caller site (~line 2779): Sequential-only wrapper function. Leave unchanged — intentional.

  **Must NOT do**:
  - Remove sequential path
  - Change function signatures beyond adding `shared_setup=None`
  - Pass C++ objects to workers
  - Use ThreadPoolExecutor

  **Recommended Agent Profile**:
  - Category: `quick` — Following existing CMA-ES pattern
  - Skills: []

  **Parallelization**: Can Parallel: NO | Wave 4 | Blocks: [5] | Blocked By: [3]

  **References**:
  - Pattern: `full_global_search.py:3886-3940` — CMA-ES parallel dispatch to follow
  - Pattern: `full_global_search.py:3907-3935` — ProcessPoolExecutor usage
  - API: `concurrent.futures.as_completed()`

  **Acceptance Criteria**:
  - [ ] Both functions accept `shared_setup=None` kwarg
  - [ ] Reference eval computed once in main process (before worker dispatch)
  - [ ] `ray_rmse_ref` and `ray_stop_threshold` passed as arguments to workers (Option B)
  - [ ] Auto-detection code present in both functions
  - [ ] Threshold check: `use_parallel = (shared_setup is not None and max_workers > 1 and n_items >= 2 * max_workers)`
  - [ ] Worker count logged with %-formatting
  - [ ] Sequential and parallel paths both exist
  - [ ] ProcessPoolExecutor uses `spawn` context and `_init_probing_worker` initializer
  - [ ] Stage 2: Block scales merged using `np.maximum` aggregator

  **QA Scenarios**:
  ```
  Scenario: Reference eval computed once (Option B verification)
    Tool: Bash
    Steps:
      grep -n "ray_rmse_ref = float(ref_result\[" modules/camera_calibration/wand_calibration/full_global_search.py && echo "PASS: Reference eval computed once"
    Expected: Output contains "PASS: Reference eval computed once"
    Evidence: .sisyphus/evidence/task-4-ref-eval-once.txt

  Scenario: Auto-detection code exists in both functions
    Tool: Bash
    Steps:
      grep -c "os.cpu_count()" modules/camera_calibration/wand_calibration/full_global_search.py | grep -q "[2-9]" && echo "PASS: Auto-detection in both functions"
    Expected: Output contains "PASS: Auto-detection in both functions"
    Evidence: .sisyphus/evidence/task-4-auto-detect-both.txt

  Scenario: Stage 2 block merge logic using np.maximum
    Tool: Bash
    Steps:
      grep -n "np.maximum(block_scales," modules/camera_calibration/wand_calibration/full_global_search.py && echo "PASS: Stage 2 merge logic present"
    Expected: Output contains "PASS: Stage 2 merge logic present"
    Evidence: .sisyphus/evidence/task-4-stage2-merge.txt

  Scenario: Parallel and sequential paths both exist
    Tool: Bash
    Steps:
      grep -n "if not use_parallel:" modules/camera_calibration/wand_calibration/full_global_search.py && grep -n "ProcessPoolExecutor" modules/camera_calibration/wand_calibration/full_global_search.py && echo "PASS: Both paths present"
    Expected: Output contains "PASS: Both paths present"
    Evidence: .sisyphus/evidence/task-4-both-paths.txt
  ```

  **Commit**: YES | Message: `feat(calibration): add parallel dispatch to Stage 1 and Stage 2 probing` | Files: [`modules/camera_calibration/wand_calibration/full_global_search.py`]

---

### Phase 5: Verification and Testing

- [x] 5. Create verification script and run numerical equivalence tests

   **What to do**:
  1. Create `.sisyphus/evidence/verify_parallel_probing.py`:
     - Load a small test dataset
     - Run Stage 1 probing with parallelization enabled (if items ≥ 2 × workers)
     - Run Stage 1 probing with parallelization disabled (force sequential)
     - Compare results using `np.allclose(scales_parallel, scales_sequential, rtol=1e-10)`
     - Repeat for Stage 2
     - Log worker count and timing

  2. Run verification script with `conda run -n OpenLPT python .sisyphus/evidence/verify_parallel_probing.py`

  3. Verify all acceptance criteria pass

  **Must NOT do**:
  - Skip numerical equivalence verification
  - Accept results with tolerance > 1e-10 without investigation

  **Recommended Agent Profile**:
  - Category: `unspecified-high` — Test harness creation
  - Skills: []

  **Parallelization**: Can Parallel: NO | Wave 5 | Blocks: [] | Blocked By: [4]

  **References**:
  - API: `numpy.allclose`
  - Pattern: `test/` directory — Test structure

  **Acceptance Criteria**:
  - [ ] Verification script exists at `.sisyphus/evidence/verify_parallel_probing.py`
  - [ ] Script runs without errors
  - [ ] Numerical equivalence: `np.allclose(scales_parallel, scales_sequential, rtol=1e-10)`
  - [ ] Worker count logged and correct
  - [ ] Parallelization triggered/bypassed correctly based on items vs. workers

  **QA Scenarios**:
  ```
  Scenario: Verification script exists
    Tool: Bash
    Steps:
      test -f .sisyphus/evidence/verify_parallel_probing.py && echo "PASS: Script exists"
    Expected: Output contains "PASS: Script exists"
    Evidence: .sisyphus/evidence/task-5-script-exists.txt

  Scenario: Numerical equivalence verified
    Tool: Bash
    Steps:
      conda run -n OpenLPT python .sisyphus/evidence/verify_parallel_probing.py > .sisyphus/evidence/task-5-output.txt 2>&1
      grep -q "PASS.*equivalence" .sisyphus/evidence/task-5-output.txt && echo "PASS: Verification succeeded"
    Expected: Output contains "PASS: Verification succeeded"
    Evidence: .sisyphus/evidence/task-5-output.txt
  ```

  **Commit**: NO | Message: N/A | Files: N/A

---

## Final Verification Wave

- [x] F1. Plan Compliance Audit — oracle
- [x] F2. Code Quality Review — unspecified-high
- [x] F3. Real Manual QA — unspecified-high
- [x] F4. Scope Fidelity Check — deep

---

## Commit Strategy

Incremental commits per phase:
1. After Phase 0: `refactor(calibration): prepare caller sites for parallel probing`
2. After Phase 1: `perf(calibration): add BLAS thread suppression for probing workers`
3. After Phase 2: `feat(calibration): extract Stage 1 single-param worker function`
4. After Phase 3: `feat(calibration): extract Stage 2 single-block worker function`
5. After Phase 4: `feat(calibration): add parallel dispatch to Stage 1 and Stage 2 probing`
6. No commit for Phase 5 (evidence files)

---

## Success Criteria

1. **Performance**: Speedup ≈ 13-16× on 24-core system with 40 parameters
2. **Correctness**: `np.allclose(parallel, sequential, rtol=1e-10)`
3. **Robustness**: Sequential fallback when items < 2 × workers
4. **Scope**: No function signature changes to public API, no CLI parameters, no evaluation logic changes

---

## Dependencies

- `concurrent.futures.ProcessPoolExecutor` (stdlib)
- `multiprocessing.get_context` (stdlib)
- Existing functions: `build_shared_setup()`, `initialize_worker_evaluation_runtime()`
- NumPy for result aggregation

No new external dependencies required.
