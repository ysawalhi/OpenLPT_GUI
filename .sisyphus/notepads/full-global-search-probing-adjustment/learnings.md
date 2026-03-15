# Learnings — Task 1: Config Plumbing

## File structure
- `full_global_search.py` is ~3500 lines. Dataclass configs (`BudgetConfig`, `ParallelConfig`) are defined before `run_global_search()`.
- Config resolution pattern: check `if config is None: config = ConfigClass()` then log it. Same pattern used for `budget_config` and `parallel_config`.
- Probing dispatch is inside `run_global_search()` at the `if budget_config.enable_probing:` block.

## Placement
- `ProbingConfig` and `GenerationDetailConfig` placed after `ParallelConfig` (line ~1676 and ~1735).
- New params added after `parallel_config` in signature — preserves position of all existing params.

## Backward compatibility
- All new fields have defaults that preserve existing behavior:
  - `ProbingConfig.probing_mode='1d'` → existing 1-D probing
  - `GenerationDetailConfig.enable=False` → no per-run detail CSV
- `run_global_search()` default `None` for both configs → auto-creates default instances.

## conda run quirk
- `conda run -n OpenLPT` on Windows produces very verbose VS2022 activation output but works correctly. Look for the actual Python output near the end.

## Task 2: Compensation helper learnings
- `evaluate_residuals()` updates native camera state each call even when Python-side reference state is untouched; compensation helper must explicitly restore zero-delta native state in `finally`.
- Ray-only compensation can reuse existing residual machinery by slicing the first `ray_slots` residual block (`optimizer._compute_slot_counts()[0]`) from the full residual vector.
- Geometry invalidation is safest as a two-layer guard: pre-check candidate geometry before optimization and in-loop checks per residual call; either guard invalidates the probe step.
- Locked-index invariance should be enforced twice: optimize only free dimensions and re-assign locked dimensions from `x_probe` after optimization before final evaluation.
- OpenLPT-env smoke with a dummy optimizer validated expected behavior: locked delta remained `0.0`, invalid geometry returned `is_valid=False`, and compensation `nfev` respected the configured cap (`2`).

## Task 3: Stage-1 multidim probing learnings
- Stage-1 can be implemented without touching block probing by keeping the existing per-parameter base step map (`_DEFAULT_PROBE_STEPS`) and applying progressive alpha growth (`h = base_step * alpha`) per axis.
- The stop rule must be driven by compensated `ray_rmse` only; using compensated objective values would silently violate the selected design.
- Reliable stop diagnostics are easiest as per-parameter stop-reason strings (e.g., `compensated_ray_rmse_stop`, `invalid_geometry_or_eval`) aggregated into `ProbeResult.early_stop_reason` for traceability.
- A safe positive fallback scale is required for every parameter even when the first expanded step fails; clipping to `[min_scale, max_scale]` preserves positivity and avoids zero-scale CMA dimensions.

## Task 4: Stage-2 dynamic block probing learnings
- Dynamic block construction is most robust when driven by `SearchParameterLayout` index metadata plus `optimizer.cam_to_window`, so only active/searchable parameters are included.
- Direction vectors should be normalized before alpha expansion; converting `safe_alpha` into `scale_block(i) = |v_i| * safe_alpha` then gives finite per-parameter contributions without extra heuristics.
- Stage-2 budgeting composes cleanly by running Stage-1 first, then allocating Stage-2 from the remaining probing eval/wall budget.
- Reduced-budget synthetic smoke in OpenLPT Python produced expected depth/tilt blocks and finite non-zero coupled block-scale contributions.

## Task 5: Scale merge and shrink-factor mapping learnings
- Scale consumption path: `probe_result.scales` → `build_shared_setup()` → `_run_cma_single()` → `_cma_objective_full()` where `x_delta = scales * x_norm` converts normalized CMA vectors to physical deltas.
- Edit insertion point is between extracting raw scales from `probe_result` (line ~3438) and passing them to `build_shared_setup()` (line ~3504); this is the only place where merged/shrunk scales can be injected without touching CMA internals.
- In 1-D mode (`probing_mode='1d'`), `block_scales` has size 0 (empty array), so the merge path naturally skips via the size-mismatch guard and raw scales pass through unchanged.
- `shrink_factor` must be strictly in (0, 1] — values ≤0 would zero out scales (CMA stalls) and >1 would amplify them (defeats the purpose). The guardrail clamps to 0.5 and logs a warning.
- The sanity guardrail ceiling of 1000.0 covers pathological cases where alpha expansion produces extremely large scales; the fallback value of 0.3 for non-finite scales was chosen as a conservative mid-range default.
- All 288 LSP diagnostics in `full_global_search.py` are pre-existing basedpyright numpy stub issues, none related to Task 5 changes.

## Task 6: Extended GenerationLog per-generation data
- `GenerationLog` is a `@dataclass` — new fields with `field(default=...)` maintain backward compatibility with existing code that constructs it without the new args.
- Reference values for real-param computation come from: `opt.initial_cam_params[cid][3+sub_idx]` (cam_t), `opt._plane_d0[entity_id]` (plane_d), 0.0 for angular params (cam_r, plane_a/b are delta-from-identity perturbations).
- Population real values are computed as `ref_values + scales * x_norm_i`, vectorizable over the population via broadcasting: `pop_real = ref_values[np.newaxis, :] + pop_deltas`.
- `gen_best_diag` from `_cma_objective_full` returns a dict with `ray_rmse`, `len_rmse`, `success` keys — these are the per-generation diagnostic metrics.
- Console log format can remain concise by reusing computed variables (`gen_best_ray_rmse`, `gen_best_len_rmse`) rather than adding new print statements.

## Task 7: Per-run generation-detail CSV output learnings
- The detail CSV writer is implemented as `write_generation_detail_csv()`, a dedicated function that takes a single `CMARunResult` and writes one CSV per run.
- Function location: inserted after `write_generation_csv()` (line ~4477) to maintain logical grouping with other CSV writers.
- Parameter names are inferred from array length in first generation record: `param_00`, `param_01`, ..., `param_NN`.
- CSV columns: base summary cols (`run_id`, `gen`, `best_objective`, etc.) then detail cols sorted lexicographically: `best_real_param_00`, `best_real_param_01`, ..., `real_min_param_00`, `real_min_param_01`, ...`
- The sort ensures parameter-grouped columns (all info for param_0, then param_1, etc.) rather than mixing best/min/max across parameters.
- `emit_diagnostics()` now writes per-run detail CSVs by looping over runs and calling `write_generation_detail_csv()` for each; output filenames follow pattern `{prefix}_run{run_id:03d}_detail.csv`.
- Detail CSV paths are collected in a dict keyed by `run_id` and added to the returned paths dict as `paths['detail_csvs']`.
- Backward compatibility: summary generation CSV schema remains unchanged; detail CSVs are only created during `emit_diagnostics()` and do not affect any existing output.
- py_compile check passed; no new LSP errors introduced (pre-existing numpy stub issues remain, ~306 total, all pre-existing per Task 5 learnings).
- Module import test confirmed: `write_generation_detail_csv`, `GenerationLog`, and `CMARunResult` all present and accessible.
