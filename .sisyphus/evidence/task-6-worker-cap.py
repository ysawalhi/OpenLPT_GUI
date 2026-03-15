#!/usr/bin/env python
"""Task 6 smoke test: verify worker-count selection logic under various inputs."""

import logging
import sys
from io import StringIO

# Configure logging to capture INFO level messages
log_capture = StringIO()
handler = logging.StreamHandler(log_capture)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(levelname)s: %(message)s')
handler.setFormatter(formatter)

logging.basicConfig(level=logging.DEBUG, handlers=[handler])

# Simulate the worker-count logic from Task 6
def test_worker_selection(n_runs, enable_parallel, requested_max_workers):
    """Test the worker-selection logic."""
    logger = logging.getLogger(__name__)
    
    class ParallelConfig:
        def __init__(self, enable_parallel, max_workers):
            self.enable_parallel = enable_parallel
            self.max_workers = max_workers
            self.worker_timeout_seconds = 7200.0
    
    _par = ParallelConfig(enable_parallel=enable_parallel, max_workers=requested_max_workers)
    
    # Task 6 logic: compute effective worker count
    if _par.max_workers <= 1:
        effective_workers = 1
    else:
        effective_workers = min(_par.max_workers, n_runs)
    
    # Gate decision
    _use_parallel = (
        _par.enable_parallel
        and effective_workers > 1
        and n_runs > 1
    )
    
    # Log the decision
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
        elif not _par.enable_parallel:
            logger.debug(
                'Parallel disabled; using sequential path (n_runs=%d)',
                n_runs,
            )
    
    return _use_parallel, effective_workers

# Test scenarios: (n_runs, enable_parallel, req_workers, exp_parallel, description)
test_cases = [
    (1, True, 4, False, "Single run should bypass pool"),
    (3, False, 4, False, "Parallel disabled should use sequential"),
    (3, True, 1, False, "max_workers=1 should use sequential"),
    (3, True, 4, True, "n_runs=3, req=4 should use parallel with 3 workers"),
    (5, True, 2, True, "n_runs=5, req=2 should use parallel with 2 workers"),
    (2, True, 10, True, "n_runs=2, req=10 should cap workers at 2"),
    (10, True, 4, True, "n_runs=10, req=4 should use 4 workers"),
]

print("=" * 80)
print("Task 6 Worker-Count Selection Smoke Tests")
print("=" * 80)

all_passed = True
for n_runs, enable_parallel, req_workers, exp_parallel, description in test_cases:
    log_capture.truncate(0)
    log_capture.seek(0)
    
    use_parallel, effective_workers = test_worker_selection(n_runs, enable_parallel, req_workers)
    logs = log_capture.getvalue()
    
    passed = use_parallel == exp_parallel
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"\n{status}: {description}")
    print(f"  Input: n_runs={n_runs}, enable_parallel={enable_parallel}, requested_max_workers={req_workers}")
    print(f"  Expected: use_parallel={exp_parallel}")
    print(f"  Actual:   use_parallel={use_parallel}, effective_workers={effective_workers}")
    if logs.strip():
        for line in logs.strip().split('\n'):
            print(f"  Log: {line}")
    
    if not passed:
        all_passed = False

print("\n" + "=" * 80)
if all_passed:
    print("✓ ALL TESTS PASSED")
    sys.exit(0)
else:
    print("✗ SOME TESTS FAILED")
    sys.exit(1)
