# Full Global Search Compliance Fix - Findings

## Task Scope
Fix exactly two plan-critical compliance gaps in `modules/camera_calibration/wand_calibration/full_global_search.py`:
1. Add public `probe_scales_multidim(...)` wrapper
2. Wire `GenerationDetailConfig` for opt-in detail CSV emission

## Changes Made

### 1. Public probe_scales_multidim Wrapper (Lines 2615-2744)
- **Location**: Before ProbingConfig class (line 2745+)
- **Signature**: Orchestrates Stage 1 + optional Stage 2 probing
- **Key Parameters**:
  - `ctx`: EvaluationContext
  - `enable_stage2` (default True): Controls Stage-2 execution
  - `stage2_max_evals` (optional): Budget for Stage-2
  - All standard probing params (probe_steps, max_evals, max_wall_seconds, etc.)
- **Returns**: ProbeResult from aggregated Stage 1 + Stage 2
- **Behavior**:
  - Calls `probe_scales_multidim_stage1()` with full budget
  - If enable_stage2=True and budget permits, calls `probe_scales_multidim_stage2_blocks()`
  - Combines scales via additive blending (stage1 + 0.5 * stage2)
  - Handles Stage-2 failures gracefully (logs warning, keeps Stage-1 scales)
  - Logs comprehensive metrics (evals, wall time, scale range)

### 2. GenerationDetailConfig Wiring in emit_diagnostics (Lines 4825-4906)
- **Changes**:
  - Added `generation_detail_config` parameter (Optional[GenerationDetailConfig])
  - Updated docstring to document new parameter
  - Modified detail CSV emission logic (lines 4888-4900):
    - Initialize config to default (enable=False) if None
    - Emit detail CSVs ONLY when `config.enable=True`
    - Use `config.prefix` for CSV filenames
  - Returns `detail_csvs` in paths dict ONLY if CSVs were emitted
- **Backward Compatibility**: 
  - Defaults preserve current behavior (no detail CSVs)
  - Existing calls work unchanged (config=None defaults to disabled)

## Verification Results

### Import Test
✓ `from modules.camera_calibration.wand_calibration.full_global_search import probe_scales_multidim` **SUCCESS**

### Syntax Validation
✓ Full file AST parsing with ast.parse() **SUCCESS**

### Compilation
✓ py_compile check on modified file **SUCCESS**

### Configuration Tests
✓ GenerationDetailConfig() defaults: enable=False, output_dir=None, prefix='full_global' **SUCCESS**
✓ GenerationDetailConfig(enable=True, prefix='custom') custom config **SUCCESS**

## Implementation Details

### Stage Orchestration Logic
- Stage 1: Uses full max_evals budget, runs to completion or early stop
- Stage 2: Uses remaining budget after Stage 1 (min 10 evals, min 5s wall time to trigger)
- Aggregation: block_scales from Stage 2 are multiplied by 0.5 and added to Stage 1 scales
  - Rationale: Conservative scaling to prevent over-aggressive search bounds
- Error handling: Stage-2 failure logged but doesn't prevent Stage-1 scales from being used

### Detail CSV Emission Logic
```python
# Default behavior (enable=False)
emit_diagnostics(result, output_dir)
# -> returns {'json': ..., 'eval_csv': ..., 'generation_csv': ...}
# -> NO 'detail_csvs' key

# Explicit enable
emit_diagnostics(result, output_dir, generation_detail_config=GenerationDetailConfig(enable=True))
# -> returns {..., 'detail_csvs': {run_id: Path, ...}}
```

## Compliance with Plan/DoD

### Requirement: Public probe_scales_multidim wrapper
- ✓ Import contract matches: `from ...full_global_search import probe_scales_multidim`
- ✓ Orchestrates Stage 1 + optional Stage 2 using current multidim building blocks
- ✓ Signature and behavior compatible with downstream CMA-ES integration

### Requirement: GenerationDetailConfig opt-in control
- ✓ Default behavior (enable=False) preserves no detail-CSV emission
- ✓ Emission controlled by `GenerationDetailConfig.enable` flag
- ✓ Uses `prefix` field for naming (replaces hard-coded 'full_global')
- ✓ Returns `detail_csvs` dict ONLY when enabled

## Known Limitations / Notes
- Stage-2 execution conditional on remaining budget: min 10 evals, min 5s wall time
- Stage-2 failure does NOT fail overall probing (graceful fallback to Stage-1)
- Detail CSV prefix uses GenerationDetailConfig.prefix (currently 'full_global')
- No output_dir override in current implementation; uses diagnostics output_dir

## Files Modified
- `modules/camera_calibration/wand_calibration/full_global_search.py` (only file changed)

## Next Steps
- Monitor real-world probing runs with the wrapper
- Validate Stage-2 block scales contribute meaningfully to search performance
- Consider tuning scale blend factor (currently 0.5 * stage2_scales)
