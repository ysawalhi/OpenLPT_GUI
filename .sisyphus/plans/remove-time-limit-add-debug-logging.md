# Plan: Remove Time Limit + Add DEBUG Logging to Stage 1 & Stage 2 Probing

## TL;DR
> **Summary**: Change default wall-time limits from 300s to unlimited (`float('inf')`) and add detailed DEBUG-level logging to both Stage 1 and Stage 2 probing for monitoring progress.
> **Deliverables**: Modified `full_global_search.py` with 6 default changes, 1 docstring update, and 11 new/enhanced DEBUG logs (5 for Stage 1, 6 for Stage 2).
> **Effort**: Quick (single-file, known line locations)
> **Critical Path**: Edit defaults → Stage 1 logging → Stage 2 logging → Verify

## Context

### Original Request
User wants to:
1. Remove the 300-second time limit for multidimensional probing
2. Add detailed DEBUG-level logging to Stage 1 and Stage 2 to monitor progress

### Problem Analysis
- Stage 1 probing has only 1 DEBUG log (per-param result) - no start banner, no progress
- Stage 2 probing has minimal logging (only DEBUG after each direction)
- With default 300s limit, probing times out before completing all parameters
- Each probe step takes ~79 seconds, making time budget critical

### Key Findings
- `max_wall_seconds: float = 300.0` appears in 6 locations
- `BudgetConfig.max_probing_wall_seconds: float = 300.0` is the upstream default
- Stage 1 function `probe_scales_multidim_stage1()` starts at line 1947
- Stage 2 function `probe_scales_multidim_stage2_blocks()` starts at line 2317
- Logger already exists at line 28: `logger = logging.getLogger(__name__)`

## Work Objectives

### Core Objective
Enable unlimited probing time with visible progress logging for debugging.

### Deliverables
1. All 6 time-limit defaults changed from `300.0` to `float('inf')`
2. 5 DEBUG logging statements added/enhanced in Stage 1 function
3. 6 DEBUG logging statements added/enhanced in Stage 2 function
4. Updated docstring for the default change

### Definition of Done
- [ ] `py_compile` passes with exit code 0
- [ ] No `300.0` wall-time defaults remain
- [ ] At least 6 `float('inf')` defaults present
- [ ] All 11 DEBUG log strings present in file
- [ ] Import succeeds without errors

### Must Have
- Default `max_wall_seconds = float('inf')` in all functions
- DEBUG logs for Stage 1: starting, param progress, alpha_step, ray_rmse, stop_reason
- DEBUG logs for Stage 2: ref_eval, block start, direction start, alpha step, direction result, block complete

### Must NOT Have (Guardrails)
- NO changes to evaluation logic or compensation algorithm
- NO changes to INFO/WARNING level logs
- NO changes to functions outside the specified ones
- NO f-strings in log calls (use %-formatting)

## Execution Strategy

### Sequential Execution Order

**Step 1**: Change defaults (Task 1, Task 2) - affects line numbers
**Step 2**: Add Stage 1 logging (Task 3) - adds lines before Stage 2
**Step 3**: Add Stage 2 logging (Task 4) - after Stage 1 changes
**Step 4**: Verify (Task 5)

### Dependency Matrix
| Task | Depends On | Reason |
|------|------------|--------|
| Task 1 (defaults) | None | Standalone |
| Task 2 (docstring) | None | Standalone |
| Task 3 (Stage 1 logging) | Task 1, 2 | Line numbers shift |
| Task 4 (Stage 2 logging) | Task 3 | Line numbers shift |
| Task 5 (verify) | All | Must verify complete |

## TODOs

### Task 1: Change Time-Limit Defaults to float('inf')

**What to do**: Change `300.0` to `float('inf')` at the following locations in `modules/camera_calibration/wand_calibration/full_global_search.py`:

| Line | Function/Class | Change |
|------|----------------|--------|
| 1811 | `probe_scales()` signature | `max_wall_seconds: float = 300.0` → `float('inf')` |
| 1952 | `probe_scales_multidim_stage1()` signature | `max_wall_seconds: float = 300.0` → `float('inf')` |
| 2323 | `probe_scales_multidim_stage2_blocks()` signature | `max_wall_seconds: float = 300.0` → `float('inf')` |
| 2627 | `probe_scales_multidim()` signature | `max_wall_seconds: float = 300.0` → `float('inf')` |
| 2865 | `BudgetConfig` dataclass | `max_probing_wall_seconds: float = 300.0` → `float('inf')` |
| 3368 | `run_global_search()` signature | `probe_max_wall: float = 300.0` → `float('inf')` |

**References**:
- Line 2693 uses `max(max_wall_seconds, 1.0)` - safe with `inf`
- Line 2715 uses `remaining_wall_seconds > 5.0` - safe with `inf`

**Recommended Agent Profile**:
- Category: `quick` — Simple text substitution at known line numbers
- Skills: [] — No special skills needed

**Parallelization**: Can Parallel: YES | None blocked | None blocking

**Acceptance Criteria**:
```bash
# Verify no 300.0 defaults remain
conda run -n OpenLPT python -c "import re; text=open('modules/camera_calibration/wand_calibration/full_global_search.py').read(); matches=re.findall(r'(?:max_wall|max_probing_wall|probe_max_wall).*?300', text); assert len(matches)==0, f'Found: {matches}'; print('PASS')"
```

---

### Task 2: Update Docstring

**What to do**: Update docstring at line ~2654 from "default 300s" to "default: no limit"

```python
# FROM (line ~2654):
        Total wall-clock time budget across both stages (default 300s).
# TO:
        Total wall-clock time budget across both stages (default: no limit).
```

**Recommended Agent Profile**:
- Category: `quick`
- Skills: []

**Parallelization**: Can Parallel: YES | None blocked | None blocking

---

### Task 3: Add DEBUG Logging to Stage 1

**What to do**: Add 4 new `logger.debug(...)` calls and enhance 1 existing one in `probe_scales_multidim_stage1()` (starts at ~line 1947, will shift after Task 1).

#### Log 3a — Start Banner (after line ~2007, after ray_stop_threshold computed):

Insert after `ray_stop_threshold = float(max(ray_rmse_stop_factor, 1.0) * ray_rmse_ref)`:

```python
    logger.debug(
        'probe_scales_multidim_stage1: starting — n_params=%d max_evals=%d '
        'max_wall_seconds=%s ray_rmse_ref=%.6g ray_stop_threshold=%.6g',
        n,
        max_evals,
        'unlimited' if max_wall_seconds == float('inf') else f'{max_wall_seconds:.0f}s',
        ray_rmse_ref,
        ray_stop_threshold,
    )
```

#### Log 3b — Param Progress (at loop entry, after `for i, entry in enumerate(layout.entries):`):

Insert at start of param loop body, before time checks:

```python
    if i % 5 == 0 or i == n - 1:
        elapsed = _time.monotonic() - t0
        logger.debug(
            'probe_scales_multidim_stage1: param %d/%d (%s) elapsed=%.1fs evals=%d',
            i + 1,
            n,
            entry.label,
            elapsed,
            n_evals,
        )
```

#### Log 3c — Rename alpha loop variable:

Change line ~2035:
```python
# FROM:
        for _ in range(max_alpha_steps):
# TO:
        for alpha_step in range(max_alpha_steps):
```

#### Log 3d — Alpha Step (after `n_evals += 1` in alpha loop):

Insert after `n_evals += 1`:
```python
            logger.debug(
                'probe_scales_multidim_stage1: [%d/%d] %s alpha_step=%d/%d '
                'step=%.6g ray_rmse=%.6g (ref=%.6g) valid=%s',
                i + 1,
                n,
                entry.label,
                alpha_step + 1,
                max_alpha_steps,
                step,
                ray_now if comp.is_valid else float('nan'),
                ray_rmse_ref,
                comp.is_valid,
            )
```

#### Log 3e — Enhance Existing (replace lines ~2087-2092):

```python
# FROM:
        logger.debug(
            'probe_scales_multidim_stage1: %s scale=%.6g stop_reason=%s',
            entry.label,
            scales[i],
            reason,
        )
# TO:
        logger.debug(
            'probe_scales_multidim_stage1: [%d/%d] %s scale=%.6g '
            'ray_rmse=%.6g (ref=%.6g) stop_reason=%s',
            i + 1,
            n,
            entry.label,
            scales[i],
            last_safe_ray,
            ray_rmse_ref,
            reason,
        )
```

**References**:
- Per-param loop: line ~2013 (`for i, entry in enumerate(layout.entries):`)
- Alpha loop: line ~2035 (`for _ in range(max_alpha_steps):`)
- Existing DEBUG log: lines ~2087-2092
- `_time` imported locally, `np` available

**Recommended Agent Profile**:
- Category: `quick`
- Skills: []

**Parallelization**: Can Parallel: NO | Depends on Task 1, 2 (line numbers shift)

**Acceptance Criteria**:
```bash
# Verify Stage 1 log strings present
conda run -n OpenLPT python -c "text=open('modules/camera_calibration/wand_calibration/full_global_search.py').read(); checks=['stage1: starting', 'param ', 'alpha_step=', 'stop_reason=']; missing=[c for c in checks if c not in text]; assert not missing, f'Missing: {missing}'; print('PASS')"
```

---

### Task 4: Add DEBUG Logging to Stage 2

**What to do**: Add 5 new `logger.debug(...)` calls and enhance 1 existing one in `probe_scales_multidim_stage2_blocks()` (line numbers will shift after Stage 1 changes).

#### Log 4a — Ref Eval (after ray_stop_threshold computed):

Insert after `ray_stop_threshold = ...`:
```python
    logger.debug(
        'probe_scales_multidim_stage2: ref_eval ray_rmse=%.6g '
        'ray_stop_threshold=%.6g n_params=%d',
        ray_rmse_ref,
        ray_stop_threshold,
        n,
    )
```

#### Log 4b — Block Start (at block loop entry):

Insert after `for blk_idx, block in enumerate(blocks):`:
```python
        logger.debug(
            'probe_scales_multidim_stage2: starting block %d/%d name=%s '
            'n_directions=%d param_indices=%s',
            blk_idx + 1,
            len(blocks),
            block.name,
            len(block.directions),
            list(block.param_indices),
        )
```

#### Log 4c — Direction Start (at direction loop entry):

Insert after `for direction in block.directions:`:
```python
            logger.debug(
                'probe_scales_multidim_stage2: block=%s starting direction=%s',
                block.name,
                direction.name,
            )
```

#### Log 4d — Alpha Step (after `alpha *= alpha_growth`):

Insert after `alpha *= alpha_growth`:
```python
                logger.debug(
                    'probe_scales_multidim_stage2: block=%s dir=%s '
                    'alpha_step safe_alpha=%.6g next_alpha=%.6g '
                    'ray_now=%.6g n_evals=%d',
                    block.name,
                    direction.name,
                    safe_alpha,
                    alpha,
                    ray_now,
                    n_evals,
                )
```

#### Log 4e — Direction Result (enhance existing):

Replace existing log:
```python
            logger.debug(
                'probe_scales_multidim_stage2: block=%s dir=%s RESULT '
                'safe_alpha=%.6g reason=%s contrib_max=%.6g',
                block.name,
                direction.name,
                safe_alpha,
                reason,
                float(np.max(contrib)) if contrib.size else 0.0,
            )
```

#### Log 4f — Block Complete (before `if early_stop:`):

Insert before block-level early_stop check:
```python
        logger.debug(
            'probe_scales_multidim_stage2: block %d/%d name=%s completed '
            'n_evals_so_far=%d elapsed=%.1fs block_scale_max=%.6g',
            blk_idx + 1,
            len(blocks),
            block.name,
            n_evals,
            _time.monotonic() - t0,
            float(np.max(block_scales)) if block_scales.size else 0.0,
        )
```

**Recommended Agent Profile**:
- Category: `quick`
- Skills: []

**Parallelization**: Can Parallel: NO | Depends on Task 3 (line numbers shift)

**Acceptance Criteria**:
```bash
# Verify Stage 2 log strings present
conda run -n OpenLPT python -c "text=open('modules/camera_calibration/wand_calibration/full_global_search.py').read(); checks=['ref_eval ray_rmse', 'starting block', 'starting direction', 'alpha_step safe_alpha', 'RESULT', 'completed']; missing=[c for c in checks if c not in text]; assert not missing, f'Missing: {missing}'; print('PASS')"
```

---

### Task 5: Verify All Changes

**What to do**: Run all verification commands.

**Acceptance Criteria**:
```bash
# 1. Syntax check
conda run -n OpenLPT python -m py_compile modules/camera_calibration/wand_calibration/full_global_search.py

# 2. No 300.0 defaults
conda run -n OpenLPT python -c "import re; text=open('modules/camera_calibration/wand_calibration/full_global_search.py').read(); matches=re.findall(r'(?:max_wall|max_probing_wall|probe_max_wall).*?300', text); assert len(matches)==0, f'Found: {matches}'; print('PASS: no 300s defaults')"

# 3. float('inf') present (>=6)
conda run -n OpenLPT python -c "import re; text=open('modules/camera_calibration/wand_calibration/full_global_search.py').read(); matches=re.findall(r\"float\('inf'\)\", text); assert len(matches)>=6, f'Expected >=6, got {len(matches)}'; print('PASS: inf defaults present')"

# 4. Stage 1 DEBUG logs present
conda run -n OpenLPT python -c "text=open('modules/camera_calibration/wand_calibration/full_global_search.py').read(); checks=['stage1: starting', 'param ', 'alpha_step=', 'stop_reason=']; missing=[c for c in checks if c not in text]; assert not missing, f'Missing Stage 1: {missing}'; print('PASS: Stage 1 logs')"

# 5. Stage 2 DEBUG logs present
conda run -n OpenLPT python -c "text=open('modules/camera_calibration/wand_calibration/full_global_search.py').read(); checks=['ref_eval ray_rmse', 'starting block', 'starting direction', 'alpha_step safe_alpha', 'RESULT', 'completed']; missing=[c for c in checks if c not in text]; assert not missing, f'Missing Stage 2: {missing}'; print('PASS: Stage 2 logs')"

# 6. Import test
conda run -n OpenLPT python -c "from modules.camera_calibration.wand_calibration.full_global_search import probe_scales_multidim_stage1, probe_scales_multidim_stage2_blocks; print('PASS: import OK')"
```

**Recommended Agent Profile**:
- Category: `quick`
- Skills: []

**Parallelization**: Can Parallel: NO | Depends on all previous tasks

---

## Final Verification Wave

- [ ] F1. Plan Compliance Audit — oracle
- [ ] F2. Code Quality Review — unspecified-high
- [ ] F3. Verify all QA commands pass
- [ ] F4. Test import and basic functionality

## Commit Strategy

Single atomic commit after all changes:
```
Remove 300s wall-time limit defaults and add DEBUG logging to Stage 1 & Stage 2 probing
```

## How to Enable DEBUG Logging

After changes are complete, enable DEBUG level in your script:

```python
import logging
# Enable DEBUG for just this module
logging.getLogger('modules.camera_calibration.wand_calibration.full_global_search').setLevel(logging.DEBUG)

# Or enable DEBUG globally
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(name)s %(levelname)s %(message)s')
```

## Success Criteria

1. `py_compile` passes (exit code 0)
2. Zero occurrences of `300.0` as wall-time default
3. At least 6 occurrences of `float('inf')` as default
4. All 5 Stage 1 DEBUG log strings present in file
5. All 6 Stage 2 DEBUG log strings present in file
6. Import succeeds without runtime errors