# Full Global Search Probing Adjustment

## TL;DR
> **Summary**: Replace the too-local 1-D curvature probing in `full_global_search.py` with a new multidimensional probing mode that uses Ray-RMSE-based stop criteria and coupled block directions, while preserving the old probing path as a fallback. Add per-run generation-detail CSV output that records best metrics and real-value population ranges for every parameter without polluting console logs.
> **Deliverables**:
> - New multidimensional probing mode in `modules/camera_calibration/wand_calibration/full_global_search.py`
> - Conservative CMA scale mapping using a shrink factor
> - Per-run generation detail CSV files with best real parameters and population real min/max
> - No regression to existing summary logs/CSV/JSON outputs
> **Effort**: Medium
> **Parallel**: YES - 2 waves
> **Critical Path**: Task 1 -> Task 2 -> Task 3 -> Task 5 -> Task 6

## Context
### Original Request
- Probing scales in `full_global_search` are too small.
- New approach is described in `modules/camera_calibration/wand_calibration/multidimension_probing.md`.
- Ask Metis and Oracle to analyze the approach and provide opinions.
- Also output per-generation parameter search ranges using real values, and include best ray/len RMSE, best parameters, sigma, etc.
- Do not rely on noisy console logs for all of this detail; CSV output is preferred.

### Interview Summary
- The current `probe_scales()` is 1-D central-difference curvature probing with `scale = 1/sqrt(curvature)` and no coupled-direction probing.
- The user selected the Ray-RMSE stop-rule path (`ray_rmse >= 1.1 * ray_rmse_ref`).
- “Search range” is defined as empirical real-value min/max across the sampled population for each generation.
- Generation detail should be written to CSV, not dumped to console.
- Parallel logging already interleaves across run workers, so detailed diagnostics should live in per-run files rather than shared stdout.

### Metis Review (gaps addressed)
- Add a new multidimensional probing mode instead of replacing the current 1-D path immediately.
- Use a shrink factor when mapping probed basin-width scales to CMA normalization scales.
- Keep existing summary generation CSV unchanged and add a new detail CSV path.
- Compensation optimization is required in the first implementation pass for multidimensional probing; omission would keep the scales biased too small for coupled parameters.
- Make dynamic block definitions from layout/camera topology, not hardcoded camera counts.

## Work Objectives
### Core Objective
Improve the realism of full-global-search parameter scales by capturing coupled basin directions, and make generation-level search behavior observable via per-run CSVs containing best metrics and real-value parameter ranges.

### Deliverables
- `full_global_search.py` supports at least two probing modes: existing 1-D and new multidimensional probing.
- New multidimensional probing uses Ray-RMSE-based stop criteria with progressive alpha expansion and dynamic block directions.
- CMA uses a conservative shrink factor on multidimensional scales before normalized search.
- A new per-run generation detail CSV is written with best ray/len RMSE, sigma, best real parameters, and per-parameter real min/max for each generation.
- Existing summary generation CSV, eval CSV, diagnostics JSON, and console logs remain stable.

### Definition of Done (verifiable conditions with commands)
- `conda run -n OpenLPT python -c "from modules.camera_calibration.wand_calibration.full_global_search import probe_scales, probe_scales_multidim; print('OK')"` succeeds.
- Same-run smoke test with `probing_mode='multidim'` produces scales that are not smaller than the fallback 1-D scales on coupled dimensions.
- A run with generation detail CSV enabled emits one per-run CSV file with required columns for every parameter.
- Existing `write_generation_csv()` output columns remain unchanged.

### Must Have
- Keep current 1-D probing path available.
- Add multidimensional probing as a separate mode.
- Use `ray_rmse >= 1.1 * ray_rmse_ref` as the probing stop rule.
- For each multidimensional probe step, run a small compensation optimization before measuring probing residuals.
- Lock the probed parameters during compensation and allow only non-block parameters to move.
- Use ray-residual-only optimization during compensation with a hard cap of 2-3 GN/LM iterations.
- Enforce geometry validity before and during compensation; invalid geometry terminates that probe direction.
- Use dynamic block definitions from `SearchParameterLayout` / camera topology.
- Apply a named shrink factor to multidimensional scales before CMA normalization.
- Write per-run generation detail CSVs with real-value ranges and best metrics.

### Must NOT Have
- No within-generation parallelism changes.
- No full BA polishing during probing; compensation is limited to 2-3 GN/LM iterations.
- No replacement/removal of the existing summary generation CSV.
- No noisy console dump of every parameter range every generation.
- No file changes outside `modules/camera_calibration/wand_calibration/full_global_search.py` unless a runner-only path is strictly required.

## Verification Strategy
> ZERO HUMAN INTERVENTION — all verification is agent-executed.
- Test decision: tests-after + OpenLPT smoke verification
- QA policy: every task includes executable checks; CSV schemas are verified programmatically
- Evidence: `.sisyphus/evidence/task-{N}-{slug}.{ext}`

## Execution Strategy
### Parallel Execution Waves
> Target: 5-8 tasks per wave. <3 per wave (except final) = under-splitting.

Wave 1: probing-mode implementation and scale computation
- mode/config plumbing
- compensation optimization helper
- dynamic block construction
- multidimensional probing core
- scale merge and shrink-factor mapping

Wave 2: generation-detail observability
- extend generation record data
- write per-run detail CSVs
- verify no regression in existing outputs

### Dependency Matrix (full, all tasks)
- Task 1 blocks Tasks 2, 3, 4, 5, 6, 7
- Task 2 blocks Tasks 3, 4, 5
- Task 3 blocks Tasks 4, 5
- Task 4 blocks Tasks 5, 6, 7
- Task 5 blocks Tasks 6, 7
- Task 6 blocks Task 7

### Agent Dispatch Summary (wave -> task count -> categories)
- Wave 1 -> 5 tasks -> `deep`, `unspecified-high`
- Wave 2 -> 2 tasks -> `unspecified-high`, `quick`
- Final Verification -> 4 tasks -> `oracle`, `deep`, `unspecified-high`

## TODOs
> Implementation + Test = ONE task. Never separate.
> EVERY task MUST have: Agent Profile + Parallelization + QA Scenarios.

- [x] 1. Add probing-mode and generation-detail configuration to `full_global_search.py`

  **What to do**: Add explicit configuration knobs for the new probing path and generation-detail CSV behavior: probing mode selection, shrink factor, block-probing enable/disable, compensation-optimization controls (`enable_compensation`, `max_compensation_iters`), and generation-detail output enable/pathing. Keep defaults backward compatible so existing callers still get current behavior unless they opt into multidimensional probing/detail output.
  **Must NOT do**: Do not change existing summary CSV filenames or default console log verbosity.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` — Reason: config/API plumbing across probing and output layers
  - Skills: `[]`
  - Omitted: `playwright` — no browser work

  **Parallelization**: Can Parallel: NO | Wave 1 | Blocks: 2,3,4,5,6,7 | Blocked By: none

  **References**:
  - Pattern: `modules/camera_calibration/wand_calibration/full_global_search.py:2091` — current `run_global_search()` argument pattern
  - Pattern: `modules/camera_calibration/wand_calibration/full_global_search.py:1401` — `ProbeResult` dataclass structure
  - Pattern: `modules/camera_calibration/wand_calibration/full_global_search.py:1585` — `GenerationLog` dataclass structure
  - External: `modules/camera_calibration/wand_calibration/multidimension_probing.md:47` — three-stage probing model

  **Acceptance Criteria**:
  - [ ] `run_global_search()` exposes probing/detail-output and compensation flags with backward-compatible defaults
  - [ ] New defaults preserve current behavior when multidimensional probing is not enabled

  **QA Scenarios**:
  ```text
  Scenario: New config flags exist and preserve default behavior
    Tool: Bash
    Steps: Run `conda run -n OpenLPT python -c "import inspect; from modules.camera_calibration.wand_calibration.full_global_search import run_global_search; print(inspect.signature(run_global_search))"`
    Expected: Signature contains new probing/detail/compensation flags; default call remains valid.
    Evidence: .sisyphus/evidence/task-1-probing-config.txt

  Scenario: Existing summary CSV names remain unchanged
    Tool: Grep
    Steps: Inspect `write_generation_csv`, `write_eval_csv`, and `emit_diagnostics` naming logic.
    Expected: Existing filenames/paths are unchanged.
    Evidence: .sisyphus/evidence/task-1-probing-config-grep.txt
  ```

  **Commit**: NO | Message: `feat(full-global-search): add probing and detail-output config` | Files: [`modules/camera_calibration/wand_calibration/full_global_search.py`]

- [x] 2. Add compensation-optimization helper for multidimensional probing

  **What to do**: Add a dedicated helper that evaluates a probe step with limited compensation optimization. Given a perturbed candidate state and a locked parameter set, the helper must lock all probe-block parameters, allow all non-block active parameters to adjust, run 2-3 GN/LM iterations minimizing ray residual only, enforce geometry validity before and during compensation, and return the compensated `ray_rmse`, validity status, and compensated state. It must restore/reference-manage optimizer/native state cleanly between probe evaluations.
  **Must NOT do**: Do not run a full BA. Do not allow locked parameters to drift. Do not mutate the stored reference state across probe evaluations.

  **Recommended Agent Profile**:
  - Category: `deep` — Reason: masked mini-optimization, geometry guards, and native-state restoration are the core risk in this change
  - Skills: `[]`
  - Omitted: `frontend-ui-ux`

  **Parallelization**: Can Parallel: NO | Wave 1 | Blocks: 3,4,5,6,7 | Blocked By: 1

  **References**:
  - Pattern: `modules/camera_calibration/wand_calibration/full_global_search.py:968` — `evaluate_candidate()` fixed-state evaluation path
  - Pattern: `modules/camera_calibration/wand_calibration/refraction_calibration_BA.py` — existing least-squares / LM infrastructure for refractive BA
  - External: `modules/camera_calibration/wand_calibration/multidimension_probing.md:143` — compensation optimization spec

  **Acceptance Criteria**:
  - [ ] Compensation runs no more than the configured 2-3 iterations
  - [ ] Locked probe parameters remain unchanged after compensation
  - [ ] Compensation optimizes ray residual only
  - [ ] Invalid geometry causes probe-step invalidation and direction stop

  **QA Scenarios**:
  ```text
  Scenario: Compensation helper respects locked parameters
    Tool: Bash
    Steps: Run a reduced probing snippet with a known locked parameter set and inspect pre/post locked values.
    Expected: Locked parameters are unchanged within tolerance after compensation.
    Evidence: .sisyphus/evidence/task-2-comp-locks.txt

  Scenario: Compensation helper respects iteration limit
    Tool: Bash
    Steps: Run the helper and capture optimizer iteration/nfev metadata.
    Expected: Compensation never exceeds the configured 2-3 GN/LM iterations.
    Evidence: .sisyphus/evidence/task-2-comp-iters.txt

  Scenario: Invalid geometry terminates probe step
    Tool: Bash
    Steps: Evaluate a deliberately invalid probe step through the helper.
    Expected: The helper marks the step invalid and the probe direction stops.
    Evidence: .sisyphus/evidence/task-2-comp-geometry.txt
  ```

  **Commit**: NO | Message: `feat(full-global-search): add probing compensation helper` | Files: [`modules/camera_calibration/wand_calibration/full_global_search.py`]

- [x] 3. Implement multidimensional probing Stage 1: Ray-RMSE-based 1-D alpha expansion

  **What to do**: Add a new probing function/path that performs 1-D probing with progressive alpha expansion using the existing parameter-type step sizes as the base unit. For each parameter, evaluate growing perturbations, then run the compensation helper with that single probed parameter locked, and stop when compensated `ray_rmse >= 1.1 * ray_rmse_ref` or geometry becomes invalid. Record the largest safe step as `scale_1d`.
  **Must NOT do**: Do not use the old curvature-based `1/sqrt(curvature)` formula inside this new mode. Do not bypass compensation in multidimensional probing mode.

  **Recommended Agent Profile**:
  - Category: `deep` — Reason: new probing algorithm with stop-rule semantics and guardrails
  - Skills: `[]`
  - Omitted: `frontend-ui-ux`

  **Parallelization**: Can Parallel: NO | Wave 1 | Blocks: 4,5,6,7 | Blocked By: 1,2

  **References**:
  - Pattern: `modules/camera_calibration/wand_calibration/full_global_search.py:1435` — current `probe_scales()` structure and guardrails
  - Pattern: `modules/camera_calibration/wand_calibration/full_global_search.py:1499` — reference-state evaluation path
  - External: `modules/camera_calibration/wand_calibration/multidimension_probing.md:55` — Stage 1 probing spec
  - External: `modules/camera_calibration/wand_calibration/multidimension_probing.md:73` — stop rule

  **Acceptance Criteria**:
  - [ ] New Stage-1 probing returns positive scales for all parameters with explicit early-stop reasons when needed
  - [ ] Stop rule uses compensated `ray_rmse`, not the weighted objective
  - [ ] Compensation is applied at each probe step with the probed parameter locked

  **QA Scenarios**:
  ```text
  Scenario: Stage-1 probing path is importable and executable
    Tool: Bash
    Steps: Run a reduced-budget smoke snippet that builds context and invokes the new multidimensional probe mode.
    Expected: Probe completes without syntax/runtime error and returns a scale vector with expected length.
    Evidence: .sisyphus/evidence/task-2-md-probe-stage1.txt

  Scenario: Ray-RMSE stop rule is active
    Tool: Read + Bash
    Steps: Inspect probe implementation and run a reduced-budget smoke case that logs/returns the stop reason.
    Expected: Stop logic references compensated `ray_rmse >= 1.1 * ray_rmse_ref` or invalid geometry, not weighted objective.
    Evidence: .sisyphus/evidence/task-2-md-probe-stage1-stop.txt
  ```

  **Commit**: NO | Message: `feat(full-global-search): add ray-rmse multidim probe stage1` | Files: [`modules/camera_calibration/wand_calibration/full_global_search.py`]

- [x] 4. Implement multidimensional probing Stage 2: dynamic block directional probing

  **What to do**: Generate block directions from actual layout/topology and probe them with the same alpha-expansion and compensated Ray-RMSE stop rule. At minimum, implement depth blocks (`plane_d` + all associated `cam_tz`) and tilt blocks (`plane_a` + all associated `cam_rx`, `plane_b` + all associated `cam_ry`). For each directional probe step, lock all parameters in the active block, run the compensation helper on all remaining non-block active parameters, then convert each safe directional scale into per-parameter `scale_block(i)` contributions.
  **Must NOT do**: Do not hardcode a fixed camera count or assume one window unless the actual layout says so. Do not allow any block parameter to move during compensation.

  **Recommended Agent Profile**:
  - Category: `deep` — Reason: coupled-direction probing requires layout-aware block construction
  - Skills: `[]`
  - Omitted: `git-master`

  **Parallelization**: Can Parallel: NO | Wave 1 | Blocks: 5,6,7 | Blocked By: 1,2,3

  **References**:
  - Pattern: `modules/camera_calibration/wand_calibration/full_global_search.py:1321` — `SearchParameterLayout` entity/group metadata
  - Pattern: `modules/camera_calibration/wand_calibration/full_global_search.py:480` — `cam_to_window` information in reference state
  - External: `modules/camera_calibration/wand_calibration/multidimension_probing.md:85` — block directional probing spec
  - External: `modules/camera_calibration/wand_calibration/multidimension_probing.md:163` — block-scale distribution rule

  **Acceptance Criteria**:
  - [ ] Block directions are created dynamically from the actual parameter layout
  - [ ] Coupled parameters receive `scale_block` contributions from safe directional probing
  - [ ] Block probe evaluations apply compensation with the entire active block locked

  **QA Scenarios**:
  ```text
  Scenario: Dynamic block construction matches current layout
    Tool: Bash
    Steps: Run a snippet that builds `SearchParameterLayout` and prints the derived block definitions for the test dataset.
    Expected: Depth and tilt blocks exist with labels consistent with current camera/window topology.
    Evidence: .sisyphus/evidence/task-3-md-probe-blocks.txt

  Scenario: Block probing produces non-zero coupled scales
    Tool: Bash
    Steps: Run reduced-budget block probing and inspect resulting per-parameter block-scale vector.
    Expected: Coupled parameters receive finite block contributions.
    Evidence: .sisyphus/evidence/task-3-md-probe-block-scales.txt
  ```

  **Commit**: NO | Message: `feat(full-global-search): add dynamic block probing` | Files: [`modules/camera_calibration/wand_calibration/full_global_search.py`]

- [x] 5. Merge scales and apply conservative CMA shrink-factor mapping

  **What to do**: Merge multidimensional scales using `effective_scale(i) = max(scale_1d(i), scale_block(i))`, then apply a named shrink factor before feeding scales into CMA normalization. Keep the existing 1-D path available and selectable. Add a post-probe sanity guardrail if needed to shrink scales globally when raw candidate feasibility/objective behavior is clearly unstable.
  **Must NOT do**: Do not silently replace the old probing mode or change the objective itself.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` — Reason: ties probing semantics into actual CMA search behavior
  - Skills: `[]`
  - Omitted: `playwright`

  **Parallelization**: Can Parallel: NO | Wave 1 | Blocks: 6,7 | Blocked By: 3,4

  **References**:
  - Pattern: `modules/camera_calibration/wand_calibration/full_global_search.py:1778` — normalized-to-physical mapping
  - Pattern: `modules/camera_calibration/wand_calibration/full_global_search.py:1866` — current CMA options/sigma0
  - Context: `modules/camera_calibration/wand_calibration/multidimension_probing.md:223` — larger coupled basin widths are expected and are the reason for this change
  - External: `modules/camera_calibration/wand_calibration/multidimension_probing.md:181` — final scale merge rule

  **Acceptance Criteria**:
  - [ ] Multidimensional mode maps basin-width scales to CMA scales through an explicit named shrink factor
  - [ ] Existing 1-D probing mode remains selectable
  - [ ] With-compensation multidimensional scales are not smaller than the uncompensated counterpart on representative coupled dimensions

  **QA Scenarios**:
  ```text
  Scenario: Effective multidim scales dominate coupled 1-D scales when appropriate
    Tool: Bash
    Steps: Run a smoke probe in both modes and compare scale vectors for coupled dimensions.
    Expected: Multidim mode with compensation yields same-or-larger pre-shrink basin-width scales on coupled dimensions than the uncompensated counterpart.
    Evidence: .sisyphus/evidence/task-4-scale-merge.txt

  Scenario: CMA scale mapping is explicit and bounded
    Tool: Read + Bash
    Steps: Inspect implementation and run a small sample to print final scales used by CMA.
    Expected: Final CMA scales equal `effective_scale * shrink_factor` (or documented equivalent) and remain finite.
    Evidence: .sisyphus/evidence/task-4-shrink-factor.txt
  ```

  **Commit**: NO | Message: `feat(full-global-search): map multidim basin scales to cma` | Files: [`modules/camera_calibration/wand_calibration/full_global_search.py`]

- [x] 6. Extend generation data with best metrics and real-value population ranges

  **What to do**: Extend the per-generation data captured in `_run_cma_single()` so each generation records best objective, best ray/len RMSE, sigma, best real parameter values, and empirical real-value min/max for every parameter across the sampled population. Preserve current console log brevity; keep the detailed values in structured data.
  **Must NOT do**: Do not replace the current summary `GenerationLog` fields with only detailed arrays; extend them compatibly.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` — Reason: needs careful per-generation data capture without disturbing current flow
  - Skills: `[]`
  - Omitted: `frontend-ui-ux`

  **Parallelization**: Can Parallel: YES | Wave 2 | Blocks: 7 | Blocked By: 1,5

  **References**:
  - Pattern: `modules/camera_calibration/wand_calibration/full_global_search.py:1901` — generation population `X = es.ask()`
  - Pattern: `modules/camera_calibration/wand_calibration/full_global_search.py:1944` — current `GenerationLog` append point
  - Pattern: `modules/camera_calibration/wand_calibration/full_global_search.py:1956` — concise generation console log

  **Acceptance Criteria**:
  - [ ] Generation records contain best ray/len RMSE and real-value min/max arrays for each parameter
  - [ ] Current concise console generation log remains readable and unchanged except for intentional small additions

  **QA Scenarios**:
  ```text
  Scenario: Generation record captures real-value population ranges
    Tool: Bash
    Steps: Run a small single-run smoke search and inspect the in-memory generation record lengths/keys.
    Expected: Each generation stores best metrics plus per-parameter best/real-min/real-max arrays of length `n_params`.
    Evidence: .sisyphus/evidence/task-5-generation-data.txt

  Scenario: Console logs remain concise
    Tool: Bash
    Steps: Run a smoke search and capture stdout.
    Expected: Console still prints summary generation lines, not dozens of per-parameter values.
    Evidence: .sisyphus/evidence/task-5-generation-log.txt
  ```

  **Commit**: NO | Message: `feat(full-global-search): capture generation detail metrics` | Files: [`modules/camera_calibration/wand_calibration/full_global_search.py`]

- [x] 7. Write per-run generation detail CSVs without regressing existing outputs

  **What to do**: Add a new writer path for generation-detail CSV files, one file per run, containing the detailed per-generation fields from Task 5. Keep the existing summary generation CSV untouched. Ensure per-run detail writing works cleanly under inter-run parallelism by using per-run files rather than shared incremental writes from multiple workers.
  **Must NOT do**: Do not change the schema of the existing summary generation CSV or eval CSV.

  **Recommended Agent Profile**:
  - Category: `quick` — Reason: output-layer extension on top of already-captured data
  - Skills: `[]`
  - Omitted: `oracle`

  **Parallelization**: Can Parallel: YES | Wave 2 | Blocks: none | Blocked By: 6

  **References**:
  - Pattern: `modules/camera_calibration/wand_calibration/full_global_search.py:3175` — existing summary generation CSV writer
  - Pattern: `modules/camera_calibration/wand_calibration/full_global_search.py:3329` — diagnostics emission orchestration
  - Requirement source: `modules/camera_calibration/wand_calibration/full_global_search.py:1901` — generation population exists here and can support per-run detail CSV output

  **Acceptance Criteria**:
  - [ ] Existing summary generation CSV columns remain unchanged
  - [ ] New per-run generation-detail CSVs are emitted with best metrics and real-value ranges
  - [ ] Parallel mode emits one detail CSV per run without file-write collisions

  **QA Scenarios**:
  ```text
  Scenario: Summary CSV remains unchanged
    Tool: Bash
    Steps: Emit diagnostics after a smoke run and compare summary generation CSV headers to the pre-change expected header list.
    Expected: Summary CSV header is unchanged.
    Evidence: .sisyphus/evidence/task-6-summary-csv.txt

  Scenario: Per-run detail CSVs exist and contain expected columns
    Tool: Bash
    Steps: Run a parallel smoke search with `n_runs=3`, emit diagnostics, and inspect generated detail CSV files.
    Expected: One detail CSV per run; each file contains `best_ray_rmse`, `best_len_rmse`, `sigma`, `best_real_*`, `real_min_*`, `real_max_*` columns.
    Evidence: .sisyphus/evidence/task-6-detail-csv.txt
  ```

  **Commit**: NO | Message: `feat(full-global-search): add per-run generation detail csv` | Files: [`modules/camera_calibration/wand_calibration/full_global_search.py`]

## Final Verification Wave (4 parallel agents, ALL must APPROVE)
- [ ] F1. Plan Compliance Audit — oracle
- [ ] F2. Code Quality Review — unspecified-high
- [ ] F3. Real Manual QA — unspecified-high
- [ ] F4. Scope Fidelity Check — deep

## Commit Strategy
- No commits during planning.
- During implementation, keep source changes in `modules/camera_calibration/wand_calibration/full_global_search.py` unless the existing runner script must be extended to enable the new mode/output path explicitly; if that becomes necessary, document it before implementation.
- Recommended implementation commit split:
  - `feat(full-global-search): add multidimensional probing mode`
  - `feat(full-global-search): add generation detail csv output`
  - `test(full-global-search): verify multidim probing and detail outputs`

## Success Criteria
- Full global search can use a multidimensional probing mode that yields larger, more realistic scales on coupled dimensions.
- Ray-RMSE-based probing semantics match the chosen design.
- CMA uses the new multidimensional scales conservatively via an explicit shrink factor.
- Each run emits a generation detail CSV with best metrics and real-value population ranges.
- Existing summary generation CSV, eval CSV, diagnostics JSON, and concise console logs continue to work.
