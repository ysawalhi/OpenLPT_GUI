# Full Global Search Implementation

## TL;DR
> **Summary**: Build a BA-compatible global search runner that perturbs a BA-refined reference state (planes + camera extrinsics) and ranks candidates using BA residual logic, then hands top-K to existing BA refinement.
> **Deliverables**: reference-state reconstruction path; camFile→cam_params mapping; CMA-ES search driver; BA-compatible objective wrapper; top‑K export and BA refinement handoff; diagnostics.
> **Effort**: XL
> **Parallel**: YES - 3 waves
> **Critical Path**: reference snapshot -> camFile/cam_params mapping -> objective wrapper -> optimizer loop -> top-K BA refinement

## Context
### Original Request
- Implement and test the plan in `modules/camera_calibration/wand_calibration/full_global_search.md` using the same dataset as `modules/camera_calibration/wand_calibration/pretest_global_search.py`.
- Start from camera files that were produced by one BA round (not by running a new BA round at startup, and not from pinhole P0).
- Reference camera files: `J:\Fish\T0\Refraction\camFile`.
- LOS and Ray RMSE logic must match `modules/camera_calibration/wand_calibration/refraction_calibration_BA.py`.

### Interview Summary
- Global objective must align with BA residual logic (ray + wand-length + optional projection + barrier penalties), not pure Ray RMSE.
- Plane parameterization must match BA anchor/tangent representation.
- Reference state must be reconstructed from the BA-round output files: camera files from `J:\Fish\T0\Refraction\camFile` plus the matching plane/media state from the same BA-produced result, not by running a fresh BA round at startup.
- camFile → cam_params reconstruction must be explicit.
- Intrinsics remain fixed.
- This is perturbation-based basin exploration around a BA-refined state, not a P0 replacement.

### Metis Review (gaps addressed)
- Identified missing camFile→cam_params path and anchor-drift risk; plan now includes explicit mapping and fixed anchors during search.
- Warned about objective mismatch and penalty inconsistency; plan now reuses BA residual logic.
- Flagged runtime risk; plan includes timing guardrails and optional reduced probing.

## Work Objectives
### Core Objective
Build a first-version full global search that perturbs a BA-refined reference state in BA-compatible parameter space and ranks candidates with BA-compatible residual logic.

### Deliverables
- `full_global_search.py` runner module (CMA‑ES driver + objective wrapper).
- Reference-state reconstruction pipeline (BA snapshot + camFile mapping).
- Top‑K candidate export and BA refinement handoff script.
- Diagnostics: per-eval and per-generation logs consistent with BA metrics.

### Definition of Done
- Global search can load reference state from BA snapshot + camFile mapping and evaluate candidates without crashing.
- Candidate ranking uses BA residual logic (same components/penalty logic as `refraction_calibration_BA.py`).
- Top‑K candidates can be passed into existing BA refinement without parameter conversion errors.
- Validation commands complete with exit code 0 (see Acceptance Criteria per task).

### Must Have
- BA-compatible plane parameterization (anchor + tangent + depth).
- BA-compatible camera perturbations (left-multiplied rotation, translation deltas).
- Intrinsics fixed.
- Explicit camFile→cam_params mapping using the BA-produced camera files already on disk.
- Explicit matching source for plane/media state from the same BA-produced result on disk.
- Implementation changes must be confined to `modules/camera_calibration/wand_calibration/full_global_search.py` only.

### Must NOT Have
- New C++ bindings.
- P0 / pinhole bootstrap in the global stage.
- Alternative plane parameterization (raw point+normal in the search space).
- Pure Ray RMSE ranking detached from BA residual logic.
- Changes to any file other than `modules/camera_calibration/wand_calibration/full_global_search.py`.

## Verification Strategy
> ZERO HUMAN INTERVENTION — all verification is agent-executed.
- Test decision: tests-after (Python checks + small-run dry evaluation)
- QA policy: Every task includes agent-executed scenarios
- Evidence: .sisyphus/evidence/task-{N}-{slug}.txt

## Execution Strategy
### Parallel Execution Waves
Wave 1: Reference-state reconstruction + camFile mapping + objective wrapper
Wave 2: CMA‑ES runner + diagnostics + runtime guardrails
Wave 3: Top‑K export + BA handoff + validation scripts

### Dependency Matrix
- Task 1 blocks Tasks 2–5
- Task 2 blocks Tasks 3–5
- Task 3 blocks Tasks 4–5
- Task 4 blocks Task 5

### Agent Dispatch Summary
- Wave 1: 3 tasks (unspecified-high)
- Wave 2: 2 tasks (unspecified-high)
- Wave 3: 2 tasks (unspecified-high)

## TODOs
- [x] 1. Define reference-state reconstruction (BA snapshot)

  **What to do**: Identify/standardize the BA-produced on-disk reference state and implement a loader that produces `cam_params`, `window_planes`, `window_media`, `cam_to_window`, and dataset metadata compatible with BA. Camera parameters must come from the existing BA-produced cam files in `J:\Fish\T0\Refraction\camFile`, not from running a new BA round at startup. Define the conversion path from `lpt.Camera`/camFile text to BA `cam_params` arrays. Ensure planes/media are loaded from the matching BA-produced result on disk.
  **Must NOT do**: Mix camera parameters from camFile with planes/media from another round or initializer.
  **Must NOT do**: Run a new BA round at startup to create the reference state.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` — Reason: multi-file data-model alignment
  - Skills: []
  - Omitted: [`playwright`] — no UI

  **Parallelization**: Can Parallel: NO | Wave 1 | Blocks: [2,3,4,5,6,7] | Blocked By: []

  **References**:
  - Pattern: `modules/camera_calibration/wand_calibration/refraction_calibration_BA.py` — BA snapshot layout and parameterization
  - Pattern: `modules/camera_calibration/wand_calibration/refraction_wand_calibrator.py:1163` — camFile export/reload usage
  - Pattern: `modules/camera_calibration/wand_calibration/pretest_global_search.py:1600` — dataset build path
  - External: `J:\Fish\T0\Refraction\camFile` — reference camera files

  **Acceptance Criteria**:
  - [ ] A loader script can print the number of cameras and planes from the BA snapshot with exit code 0
  - [ ] `cam_params` arrays have shape `(N_cam, 11)` with finite values

  **QA Scenarios**:
  ```
  Scenario: Load reference snapshot
    Tool: Bash
    Steps: conda run -n OpenLPT python -c "from modules.camera_calibration.wand_calibration.full_global_search import load_reference_state; s=load_reference_state(...); print(len(s['cam_params']))"
    Expected: prints camera count and exits 0
    Evidence: .sisyphus/evidence/task-1-refstate.txt

  Scenario: Detect mixed-source state
    Tool: Bash
    Steps: conda run -n OpenLPT python -c "from modules.camera_calibration.wand_calibration.full_global_search import validate_reference_state; validate_reference_state(...)"
    Expected: exits 0 only if planes/cameras/media are consistent
    Evidence: .sisyphus/evidence/task-1-refstate-error.txt
  ```

  **Commit**: YES | Message: `feat(global-search): add reference state loader` | Files: [modules/camera_calibration/wand_calibration/full_global_search.py]

- [x] 2. Implement BA-compatible objective wrapper

  **What to do**: Build a wrapper that evaluates a candidate by reusing BA residual logic (`evaluate_residuals`) without running full BA. Ensure candidate parameterization matches BA (`_unpack_params_delta` logic). Handle NaN/Inf with BA-style coercion and return scalar objective plus diagnostics (ray rmse, len rmse, proj rmse, penalty terms).
  **Must NOT do**: Replace BA residual logic with a simplified Ray RMSE objective.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` — Reason: tight coupling to BA internals
  - Skills: []
  - Omitted: [`playwright`]

  **Parallelization**: Can Parallel: NO | Wave 1 | Blocks: [3,4,5,6,7] | Blocked By: [1]

  **References**:
  - API/Type: `modules/camera_calibration/wand_calibration/refraction_calibration_BA.py:1291` — evaluate_residuals usage
  - Pattern: `modules/camera_calibration/wand_calibration/refraction_calibration_BA.py:1230` — plane reconstruction
  - Pattern: `modules/camera_calibration/wand_calibration/refraction_calibration_BA.py:1271` — rotation update

  **Acceptance Criteria**:
  - [ ] Evaluator returns finite scalar for the BA reference state
  - [ ] Evaluator returns diagnostics with ray/len/proj rmses

  **QA Scenarios**:
  ```
  Scenario: Evaluate reference state
    Tool: Bash
    Steps: conda run -n OpenLPT python -c "from modules.camera_calibration.wand_calibration.full_global_search import evaluate_candidate; print(evaluate_candidate(...))"
    Expected: prints objective and metrics, exit 0
    Evidence: .sisyphus/evidence/task-2-eval.txt
  ```

  **Commit**: YES | Message: `feat(global-search): add BA-compatible evaluator` | Files: [modules/camera_calibration/wand_calibration/full_global_search.py]

- [x] 3. Define search parameter vector and scaling (1D probing)

  **What to do**: Define parameter layout: per-plane `[d,a,b]`, per-camera `[dr(3), dt(3)]`. Implement 1D probing around BA reference to estimate per-parameter scales using the BA-compatible evaluator. Enforce runtime guardrails (limit probing if too expensive).
  **Must NOT do**: Probe using pure Ray RMSE or use pinhole-derived scales.

  **Recommended Agent Profile**:
  - Category: `unspecified-high`
  - Skills: []
  - Omitted: [`playwright`]

  **Parallelization**: Can Parallel: NO | Wave 1 | Blocks: [4,5,6,7] | Blocked By: [1,2]

  **References**:
  - Pattern: `modules/camera_calibration/wand_calibration/full_global_search.md` — 1D probing spec
  - Pattern: `modules/camera_calibration/wand_calibration/refraction_calibration_BA.py` — diff step scales

  **Acceptance Criteria**:
  - [ ] Scales vector is finite and non-zero for all active parameters
  - [ ] Probing exits early if runtime guardrail exceeded

  **QA Scenarios**:
  ```
  Scenario: Probe scales
    Tool: Bash
    Steps: conda run -n OpenLPT python -c "from modules.camera_calibration.wand_calibration.full_global_search import probe_scales; print(probe_scales(...))"
    Expected: prints scale summary, exit 0
    Evidence: .sisyphus/evidence/task-3-probe.txt
  ```

  **Commit**: YES | Message: `feat(global-search): add scale probing` | Files: [modules/camera_calibration/wand_calibration/full_global_search.py]

- [x] 4. Implement CMA‑ES (or selected optimizer) driver

  **What to do**: Add CMA‑ES driver with bounds and sigma initialization based on probe scales. Ensure candidate decoding uses BA parameterization and evaluation uses BA-compatible objective wrapper. Use evaluation budget guardrails and record per-eval diagnostics.
  **Must NOT do**: Use full BA loop per candidate.

  **Recommended Agent Profile**:
  - Category: `unspecified-high`
  - Skills: []

  **Parallelization**: Can Parallel: NO | Wave 2 | Blocks: [5,6,7] | Blocked By: [1,2,3]

  **References**:
  - Pattern: `modules/camera_calibration/wand_calibration/pretest_global_search.py` — evaluation loop structure
  - External: CMA-ES library (choose and document)

  **Acceptance Criteria**:
  - [ ] Optimizer runs at least one generation and writes per-eval logs
  - [ ] Best objective improves or remains finite

  **QA Scenarios**:
  ```
  Scenario: Run single generation
    Tool: Bash
    Steps: conda run -n OpenLPT python -c "from modules.camera_calibration.wand_calibration.full_global_search import run_global_search; run_global_search(max_generations=1, ...)"
    Expected: per-eval log written, exit 0
    Evidence: .sisyphus/evidence/task-4-run.txt
  ```

  **Commit**: YES | Message: `feat(global-search): add CMA-ES driver` | Files: [modules/camera_calibration/wand_calibration/full_global_search.py]

- [x] 5. Top‑K selection and BA refinement handoff

  **What to do**: Rank candidates by objective, select top‑K, and provide a handoff function to run the existing BA refinement for each candidate. Ensure candidate parameters map back into BA initial state correctly and restore reference state between candidates.
  **Must NOT do**: Change BA optimizer logic.

  **Recommended Agent Profile**:
  - Category: `unspecified-high`

  **Parallelization**: Can Parallel: NO | Wave 3 | Blocks: [6,7] | Blocked By: [1,2,3,4]

  **References**:
  - Pattern: `modules/camera_calibration/wand_calibration/refraction_calibration_BA.py` — optimization entrypoints
  - Pattern: `modules/camera_calibration/wand_calibration/pretest_global_search.py` — results/summary writing

  **Acceptance Criteria**:
  - [ ] Top‑K candidates exported with parameters and scores
  - [ ] BA refinement runs for at least one candidate without exceptions

  **QA Scenarios**:
  ```
  Scenario: Run BA refinement on best candidate
    Tool: Bash
    Steps: conda run -n OpenLPT python -c "from modules.camera_calibration.wand_calibration.full_global_search import refine_topk; refine_topk(k=1, ...)"
    Expected: BA refinement completes, exit 0
    Evidence: .sisyphus/evidence/task-5-refine.txt
  ```

  **Commit**: YES | Message: `feat(global-search): add top-K BA handoff` | Files: [modules/camera_calibration/wand_calibration/full_global_search.py]

- [x] 6. Diagnostics and logging (per-eval + per-generation)

  **What to do**: Log per-eval objective components, invalid fractions, LOS reasons, geometry penalties, and per-generation summaries. Keep format compatible with existing JSON/CSV patterns in `pretest_global_search.py`.
  **Must NOT do**: Omit objective components that BA uses.

  **Recommended Agent Profile**:
  - Category: `unspecified-high`

  **Parallelization**: Can Parallel: YES | Wave 2 | Blocks: [7] | Blocked By: [2,4]

  **References**:
  - Pattern: `modules/camera_calibration/wand_calibration/pretest_global_search.py` — CSV/JSON writers

  **Acceptance Criteria**:
  - [ ] Per-eval logs contain objective components and failure reasons
  - [ ] Per-generation logs contain best/median/feasible fraction

  **QA Scenarios**:
  ```
  Scenario: Inspect logs
    Tool: Bash
    Steps: conda run -n OpenLPT python -c "from modules.camera_calibration.wand_calibration.full_global_search import write_logs; print('ok')"
    Expected: log files exist, exit 0
    Evidence: .sisyphus/evidence/task-6-logs.txt
  ```

  **Commit**: YES | Message: `feat(global-search): add diagnostics` | Files: [modules/camera_calibration/wand_calibration/full_global_search.py]

- [x] 7. Runtime guardrails and evaluation budget

  **What to do**: Measure and enforce per-eval wall-time and max-evals budget. Provide an optional reduced-probing mode to cap probing cost.
  **Must NOT do**: Run unbounded evaluation loops.

  **Recommended Agent Profile**:
  - Category: `unspecified-high`

  **Parallelization**: Can Parallel: YES | Wave 2 | Blocks: [] | Blocked By: [2,3,4]

  **References**:
  - Pattern: `modules/camera_calibration/wand_calibration/pretest_global_search.py` — evaluation counters

  **Acceptance Criteria**:
  - [ ] Evaluation budget respected (no more than configured max)
  - [ ] Probing stops if runtime cap exceeded

  **QA Scenarios**:
  ```
  Scenario: Enforce budget
    Tool: Bash
    Steps: conda run -n OpenLPT python -c "from modules.camera_calibration.wand_calibration.full_global_search import run_global_search; run_global_search(max_evals=10, ...)"
    Expected: exits after 10 evals
    Evidence: .sisyphus/evidence/task-7-budget.txt
  ```

  **Commit**: YES | Message: `feat(global-search): add guardrails` | Files: [modules/camera_calibration/wand_calibration/full_global_search.py]

## Final Verification Wave (4 parallel agents, ALL must APPROVE)
- [x] F1. Plan Compliance Audit — oracle
- [x] F2. Code Quality Review — unspecified-high
- [x] F3. Real Manual QA — unspecified-high
- [x] F4. Scope Fidelity Check — deep

## Commit Strategy
- Commit per major task group (reference loader, evaluator, CMA‑ES driver, handoff/logging) if requested.

## Success Criteria
- Reference state loads and matches BA snapshot without inconsistency.
- Global evaluator produces stable, BA-compatible objective values.
- CMA‑ES runs and outputs top‑K candidates.
- BA refinement succeeds on at least the best candidate.
