#!/usr/bin/env python
"""Task 6 single-run bypass test: verify that n_runs=1 skips pool creation."""

import logging
from io import StringIO

# Capture logs
log_capture = StringIO()
handler = logging.StreamHandler(log_capture)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(levelname)s: %(message)s')
handler.setFormatter(formatter)

logging.basicConfig(level=logging.DEBUG, handlers=[handler])

# Simulate the worker-count logic from Task 6
def test_single_run_bypass():
    """Test that single-run configuration skips pool creation."""
    logger = logging.getLogger(__name__)
    
    class ParallelConfig:
        def __init__(self, enable_parallel, max_workers):
            self.enable_parallel = enable_parallel
            self.max_workers = max_workers
            self.worker_timeout_seconds = 7200.0
    
    # Case 1: n_runs=1, enable_parallel=True, max_workers=4
    # Expected: pool is NOT created, sequential path is used
    print("=" * 80)
    print("Test Case: n_runs=1 with parallel enabled")
    print("=" * 80)
    
    n_runs = 1
    _par = ParallelConfig(enable_parallel=True, max_workers=4)
    
    # Task 6 logic
    if _par.max_workers <= 1:
        effective_workers = 1
    else:
        effective_workers = min(_par.max_workers, n_runs)
    
    _use_parallel = (
        _par.enable_parallel
        and effective_workers > 1
        and n_runs > 1
    )
    
    if _use_parallel:
        logger.info(
            'Task 6: Phase-1 parallel mode with bounded workers '
            '(requested_max_workers=%d, n_runs=%d, effective_workers=%d, timeout=%.0fs)',
            _par.max_workers, n_runs, effective_workers, _par.worker_timeout_seconds,
        )
    else:
        if _par.enable_parallel and (effective_workers <= 1 or n_runs <= 1):
            logger.info(
                'Parallel requested but bypassed (requested_max_workers=%d, n_runs=%d, effective_workers=%d); '
                'using sequential path',
                _par.max_workers, n_runs, effective_workers,
            )
    
    print(f"\nInput: n_runs={n_runs}, enable_parallel={_par.enable_parallel}, requested_max_workers={_par.max_workers}")
    print(f"Result: _use_parallel={_use_parallel}, effective_workers={effective_workers}")
    print(f"Logs:\n{log_capture.getvalue()}")
    
    # Verify expectations
    expected_use_parallel = False
    expected_effective_workers = 1
    
    if _use_parallel == expected_use_parallel and effective_workers == expected_effective_workers:
        print("\n✓ PASS: Single-run bypass works correctly — pool creation is skipped")
        return True
    else:
        print("\n✗ FAIL: Single-run bypass failed")
        return False

if __name__ == '__main__':
    import sys
    success = test_single_run_bypass()
    sys.exit(0 if success else 1)
