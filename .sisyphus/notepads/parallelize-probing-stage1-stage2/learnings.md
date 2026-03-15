# Learnings — Parallelize Probing Stage1/Stage2

## Phase 5: Verification Script

### Key Findings
- The verification script existed from a prior phase and was comprehensive
- Only needed minor additions: `logger.info` calls with `%`-formatting for worker metrics
- With 4 cams, 1 window, 10 frames: `n_params=27`, `n_blocks=3` — both below `2*max_workers=38`
- Both paths correctly fall back to sequential, producing identical results (`np.allclose=True`)
- Total execution: ~55s per run (Stage1 ~17s × 2 + Stage2 ~10.5s × 2)

### Parallelization Decision Threshold
- `max_workers = min(32, max(1, int(cpu_count * 0.8)))` → 19 on 24-core machine
- Parallel triggers when `items >= 2 * max_workers` → threshold of 38
- With 4 cameras + 1 window: 27 params (3 plane + 24 cam extrinsic) — below threshold
- With 4 cameras + 1 window: 3 blocks — well below threshold
- Would need ~10+ cameras sharing multiple windows to trigger parallel

### Timing Observations
- Sequential Stage 1: ~17s for 100 evals (27 params)
- Sequential Stage 2: ~10.5s for 73 evals (3 blocks)
- No speedup expected since both paths run identical sequential code

### Output Reliability
- `conda run -n OpenLPT` on Windows includes ~30s of Visual Studio environment setup
- The initial timeout (300s) was hit because conda init + VS setup + 4 probing runs > 300s
- Reducing budget to `max_evals=100, max_wall_seconds=30s` made it complete in ~55s of actual Python time
- Total wall time including conda init: ~90-120s
## F2 Code Quality Review Findings

### Logger Formatting
- All logger calls in new parallel functions use %%-formatting (PASS)
- f-strings in sequential paths are in local variables (reason, early_stop), not in logger calls — acceptable

### Type Hints
- _init_probing_worker: No type hints (no params, no return) — minor gap
- _probe_stage1_single_param: Full type hints, returns Dict[str, Any]
- _probe_stage2_single_block: Missing type hints on function signature, has docstring types
