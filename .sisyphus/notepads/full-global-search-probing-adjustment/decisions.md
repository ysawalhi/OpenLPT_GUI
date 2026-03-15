# Decisions — Task 1: Config Plumbing

## D1: ProbingConfig defaults
- `probing_mode='1d'` — preserves existing curvature probing
- `shrink_factor=0.5` — matches plan recommendation
- `enable_block_probing=True` — default on so multidim mode gets full pipeline
- `enable_compensation=True`, `max_compensation_iters=3` — per multidimension_probing.md
- `ray_rmse_stop_factor=1.1` — per plan guidance

## D2: GenerationDetailConfig defaults
- `enable=False` — no per-run CSV by default (preserves current behavior)
- `output_dir=None` — will use diagnostics output_dir when implemented
- `prefix='full_global'` — matches existing `emit_diagnostics` prefix convention

## D3: Multidim fallback in Task 1
- When `probing_mode='multidim'` is requested but not yet implemented (Task 3+), we log a WARNING and fall back to 1-D probing. This lets the config plumbing be tested end-to-end without breaking anything.

## D4: Config resolution location
- Both configs resolved after `budget_config` resolution, before budget_status tracking. This keeps all config resolution in one logical block.

## D5: Logging level
- ProbingConfig logged at DEBUG (not noisy for normal use)
- GenerationDetailConfig logged at INFO only when enabled (relevant for user awareness)

## D6: Compensation optimizer strategy in Task 2
- Use `least_squares` with `method='lm'` for mini GN/LM-style compensation and `max_nfev=max_compensation_iters` to enforce the 2-3 iteration cap.
- Fallback to `method='trf'` only if LM cannot run (numerical/shape failure), while preserving the same `max_nfev` cap.

## D7: Objective semantics for compensation
- Compensation path optimizes ray residual only by returning only the ray residual block from `evaluate_residuals` inside the helper objective wrapper.
- Final reported metric for probe-stop logic is compensated `ray_rmse` from a full candidate evaluation, while optimization itself remains ray-only.

## D8: Geometry validity and state hygiene
- Geometry validity is enforced pre-compensation, during each compensation residual call, and post-compensation before final metric emission.
- Helper always restores optimizer native state to zero-delta reference in a `finally` block to avoid cross-probe contamination.

## D9: Stage-1 multidim probing semantics (Task 3)
- Add `probe_scales_multidim_stage1()` as a separate path; keep `probe_scales()` unchanged and still selectable via `probing_mode='1d'`.
- In `probing_mode='multidim'`, dispatch to Stage-1 alpha expansion with compensation at every probe step and the probed index locked.
- Stop per-parameter expansion when compensated `ray_rmse >= ray_rmse_stop_factor * ray_rmse_ref` or when compensation/geometry becomes invalid.
- Stage-1 implementation intentionally skips block probing even if `enable_block_probing=True` (logged as informational), matching Task 3 scope.

## D10: Stage-2 dynamic block model (Task 4)
- Build blocks from live `SearchParameterLayout` + `optimizer.cam_to_window` topology, not hardcoded camera counts.
- Require coupled blocks to include at least one plane parameter and one associated camera parameter before probing.
- Implement minimum required block families per window: depth (`plane_d` + `cam_tz`), tilt-a (`plane_a` + `cam_rx`), tilt-b (`plane_b` + `cam_ry`).
- Keep Stage-1 `scales` untouched in Task 4; expose Stage-2 outputs via `ProbeResult.block_scales` and `ProbeResult.block_probe_summary` only (Task 5 will merge scales).

## D11: Scale merge strategy (Task 5)
- Merge via `effective_scale(i) = max(scale_1d(i), scale_block(i))` — `max` ensures the CMA search radius on each axis is at least as wide as the more informative probe; `mean` or `min` could under-explore coupled dimensions.
- When block scales are absent or size-mismatched, fall back to Stage-1 scales only with an informational log (no error).

## D12: Shrink-factor application point (Task 5)
- Apply shrink_factor multiplicatively *after* merge but *before* passing to `build_shared_setup()`, so CMA sees conservative (smaller) step sizes while the raw probed scales remain available for diagnostics.
- Shrink_factor validated to (0, 1] with clamp-to-0.5 fallback; this prevents silent misconfiguration.

## D13: Post-probe sanity guardrail (Task 5)
- Ceiling of 1000.0 for any individual scale; non-finite values replaced with 0.3 (conservative mid-range). Floor of 1e-8 ensures CMA positivity.
- Guardrail is applied after shrink (not before) so that the shrink factor has already reduced most pathological values; the guardrail catches only edge cases.

## D14: 1-D passthrough semantics (Task 5)
- When `probing_mode='1d'`, raw Stage-1 scales are passed directly to CMA with no merge, no shrink, no guardrail — preserving exact backward compatibility with the pre-Task-5 pipeline.
- This is enforced by the `if probing_config.probing_mode == 'multidim':` branch; the `else` simply assigns `scales = raw_scales_1d`.

## D15: Extended GenerationLog field design (Task 6)
- Added 5 new fields to `GenerationLog` dataclass: `best_ray_rmse`, `best_len_rmse` (floats, default `NaN`), `best_real_params`, `pop_real_min`, `pop_real_max` (Optional[np.ndarray], default `None`).
- NaN/None defaults chosen over sentinel values so that backward-compatible construction (core fields only) produces distinguishable "not populated" state.
- `_extract_reference_values(ctx)` computes reference vector once per CMA run rather than per generation — reference values are constant across generations within a single run.
- ref=0 for angular params (cam_r, plane_a/b) is adequate for search-range diagnostics even though the actual parameter composition is non-additive (Rodrigues, tangent-space).
- Console log format unchanged: `gen_best_ray_rmse` and `gen_best_len_rmse` variables are reused in logger.info rather than duplicating the diagnostic extraction.
- `try/except` fallback to zeros for `ref_values` ensures robustness if optimizer state is incomplete during early debugging.

## D16: Per-run detail CSV writer design (Task 7)
- Dedicated function `write_generation_detail_csv()` takes a single `CMARunResult` (not the full result) to avoid multi-process write collisions.
- Each run writes one CSV file: `{prefix}_run{run_id:03d}_detail.csv` for per-run isolation under inter-run parallelism.
- Parameter names auto-generated from array length: `param_00`, `param_01`, etc. (no dependency on external parameter metadata).
- Column ordering: base summary cols first (run_id, gen, objective metrics, sigma, evals, wall-time), then detail cols sorted lexicographically.
- Detail cols group by parameter (all columns for param_0 before param_1) to aid post-processing and analysis.
- Empty generation logs produce a warning and skip CSV write (safe fallback); no error thrown.
- `emit_diagnostics()` updated to call `write_generation_detail_csv()` for each run and collect paths in dict; returned paths dict now includes `'detail_csvs': {run_id: Path}`.
- Backward compatibility: summary generation CSV schema (via `_flatten_generation_log()`) remains identical; Task 6 extended fields are NOT included in summary CSV, only in per-run detail CSVs.

