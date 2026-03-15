## Task 1: RMSE Calculation Fix

### Issue
- **Location**: `refraction_plate_calibration.py:442` (was line 450 in evidence)
- **Problem**: RMSE calculated as `sqrt(proj_sq / proj_n)` where `proj_n = 2N` (counts both du and dv)
- **Result**: RMSE reported √2 too small (~0.707x actual value)

### Fix Applied
- **Changed**: Line 442 from `proj_sq / max(proj_n, 1)` to `proj_sq / max(proj_n // 2, 1)`
- **Updated**: Comments (lines 437-439) to reflect "FIXED" status
- **Formula**: Now correctly computes `sqrt(sum(du² + dv²) / N_observations)`

### Verification
- ✅ Python syntax validated with `py_compile`
- ✅ LSP diagnostics clean (no new errors)
- ✅ Grep confirms `proj_n // 2` present at line 441
- ✅ Comment updated to "FIXED: RMSE denominator now correctly divides by N observations"

### Impact
- RMSE values will now be √2 larger (~1.414x)
- Calibration convergence thresholds may need adjustment
- More accurate representation of projection error magnitude

## Task F2: Numerical Validation of Precision Fixes

### Softplus Stability (Line ~453)
- Verified `gap_smooth = tau * np.logaddexp(0.0, gap / tau)` remains finite for extreme arguments (`gap/tau = ±5e7` equivalent tested via ±1e6 with tau=0.02).
- Confirmed asymptotics: softplus(700) ≈ 700, softplus(-700) ≈ 0, and no overflow/NaN with `logaddexp`.
- Confirmed legacy `log1p(exp(x))` can overflow (`x=1e6` gives inf), validating fix rationale.
- Barrier terms remain finite: `r1` bounded in `[0, sqrt(2*alpha)]`, `r2` linear in `gap_smooth` and finite in tested range.

### Solver Scaling (Lines ~542-570)
- `diff_step` is created with `np.float64`, length `len(layout)`, initialized positive, and overwritten with positive per-parameter magnitudes.
- Verified SciPy accepts `x_scale='jac'` with array `diff_step` together in `least_squares(...)` and converges in a synthetic heterogeneous-scale problem.

### RMSE Denominator (Line ~470)
- Verified formula numerically: `sqrt(proj_sq / max(proj_n//2,1))` gives expected values for 0, 2-observation, and 25-observation cases.

### Error Diagnostics Schema
- `_summarize_errors` computes `{mean,std,median,p90,p95,max,count}` using float/int casts (JSON-safe).
- Edge cases validated numerically: single-value (`std=0`, percentiles=value), all-zero, mixed-sign arrays.
- Note: empty per-camera arrays are emitted as `{}` in detail maps (not `None`) due `... if vp else {}` in `_compute_error_stats`.
