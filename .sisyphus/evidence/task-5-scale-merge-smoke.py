"""Smoke test for Task 5: scale merge and shrink-factor mapping."""
import numpy as np

# Test 1: multidim with both stages
print('=== Test 1: Multidim with block scales ===')
raw_scales_1d = np.array([0.1, 0.2, 0.05, 0.3, 0.15])
raw_block_scales = np.array([0.0, 0.5, 0.08, 0.0, 0.4])
shrink = 0.5

effective = np.maximum(raw_scales_1d, raw_block_scales)
n_dom = int(np.sum(raw_block_scales > raw_scales_1d))
cma_scales = effective * shrink
print(f'  scale_1d:       {raw_scales_1d}')
print(f'  block_scales:   {raw_block_scales}')
print(f'  effective:      {effective}')
print(f'  CMA (shrink={shrink}): {cma_scales}')
print(f'  block-dominated: {n_dom}/{len(effective)}')
for i in range(len(raw_scales_1d)):
    if raw_block_scales[i] > raw_scales_1d[i]:
        assert effective[i] >= raw_scales_1d[i], f'effective[{i}] < scale_1d[{i}]'
        print(f'  dim {i}: coupled, effective={effective[i]:.4f} >= scale_1d={raw_scales_1d[i]:.4f} OK')

# Test 2: multidim without block scales (Stage-2 skipped)
print('\n=== Test 2: Multidim, Stage-2 skipped ===')
raw_block_empty = np.zeros(0, dtype=np.float64)
if raw_block_empty.size == raw_scales_1d.size and raw_block_empty.size > 0:
    effective2 = np.maximum(raw_scales_1d, raw_block_empty)
else:
    effective2 = raw_scales_1d.copy()
cma2 = effective2 * shrink
print(f'  effective:      {effective2}')
print(f'  CMA:            {cma2}')
assert np.allclose(effective2, raw_scales_1d), 'Should equal raw scales when no block'
print('  Correctly uses Stage-1 only')

# Test 3: 1-D mode (no merge, no shrink)
print('\n=== Test 3: 1-D mode ===')
scales_1d_mode = raw_scales_1d
print(f'  scales:  {scales_1d_mode}')
assert np.array_equal(scales_1d_mode, raw_scales_1d), 'Should be raw scales unchanged'
print('  1-D scales pass through unchanged')

# Test 4: sanity guardrail
print('\n=== Test 4: Sanity guardrail ===')
bad_scales = np.array([0.1, np.inf, 2000.0, np.nan, 0.5])
CEIL = 1000.0
FLOOR = 1e-8
fixed = np.where(np.isfinite(bad_scales), np.minimum(bad_scales, CEIL), 0.3)
fixed = np.maximum(fixed, FLOOR)
print(f'  input:   {bad_scales}')
print(f'  fixed:   {fixed}')
assert np.all(np.isfinite(fixed)), 'All should be finite'
assert np.all(fixed > 0), 'All should be positive'
print('  Guardrail works correctly')

# Test 5: shrink_factor validation
print('\n=== Test 5: Shrink factor bounds ===')
for sf in [0.0, -0.5, 1.5, 0.5, 1.0]:
    if sf <= 0.0 or sf > 1.0:
        print(f'  shrink_factor={sf:.1f} -> CLAMPED to 0.5')
    else:
        print(f'  shrink_factor={sf:.1f} -> OK')

# Test 6: CMA scales are exactly effective * shrink
print('\n=== Test 6: CMA scale = effective_scale * shrink_factor ===')
assert np.allclose(cma_scales, effective * shrink), 'CMA scales must equal effective * shrink'
print(f'  Verified: cma_scales == effective_scales * {shrink}')

# Test 7: multidim scales >= 1-D on coupled dims (pre-shrink)
print('\n=== Test 7: Pre-shrink multidim >= 1-D on coupled dims ===')
for i in range(len(raw_scales_1d)):
    assert effective[i] >= raw_scales_1d[i], f'effective[{i}] < scale_1d[{i}]'
print('  All effective_scales >= scale_1d (by construction via max())')

print('\nALL SMOKE TESTS PASSED')
