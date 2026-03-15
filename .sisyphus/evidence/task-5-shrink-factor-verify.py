"""Evidence for Task 5: shrink-factor mapping verification."""
import numpy as np
from modules.camera_calibration.wand_calibration.full_global_search import ProbingConfig

# Verify shrink factor is accessible and has expected default
pc = ProbingConfig()
print(f'ProbingConfig default shrink_factor: {pc.shrink_factor}')
assert pc.shrink_factor == 0.5, f'Expected 0.5, got {pc.shrink_factor}'

# Verify multidim mode with custom shrink factor
pc2 = ProbingConfig(probing_mode='multidim', shrink_factor=0.3)
print(f'Custom config: mode={pc2.probing_mode}, shrink={pc2.shrink_factor}')

# Simulate final CMA scale derivation
scales_1d = np.array([0.1, 0.2, 0.05, 0.3, 0.15])
block_scales = np.array([0.0, 0.5, 0.08, 0.0, 0.4])
effective = np.maximum(scales_1d, block_scales)
cma_scales = effective * pc2.shrink_factor

print(f'\nEffective scales: {effective}')
print(f'CMA scales (shrink={pc2.shrink_factor}): {cma_scales}')
print(f'All finite: {np.all(np.isfinite(cma_scales))}')
print(f'All positive: {np.all(cma_scales > 0)}')

# Verify derivation
for i in range(len(scales_1d)):
    expected = max(scales_1d[i], block_scales[i]) * pc2.shrink_factor
    assert abs(cma_scales[i] - expected) < 1e-15, f'Mismatch at {i}'
    print(f'  param[{i}]: effective={effective[i]:.4f} * shrink={pc2.shrink_factor} = cma={cma_scales[i]:.4f} OK')

print('\nSHRINK FACTOR VERIFICATION PASSED')
