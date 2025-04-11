"""
    detect_boundary_signal_fast(y_centered, n; which_end::Symbol)

Efficiently detects the appropriate boundary condition type for signal extrapolation
based on local behavior at either end of the signal.

This function analyzes a window of the signal to determine whether it exhibits
periodic behavior or can be well-approximated by quadratic extrapolation.

# Arguments
- `y`: Input signal vector
- `which_end::Symbol`: Specify which end of the signal to analyze (`:left` or `:right`)

# Returns
- `coefs`: Coefficients for extrapolation
    - `[3.0, -3.0, 1.0]` for quadratic extrapolation (default)
    - Result of `autocor_coefs(y_centered)` for periodic signals

# Details
The function uses a sign-change approach to detect periodicity in the signal's
first differences. If periodicity is detected, it returns linear prediction coefficients
mixed with quadratic extrapolation. Otherwise, it returns quadratic extrapolation
coefficients as the default.

This approach automatically adapts to different sampling rates. For coarsely sampled
signals with visible oscillations, it's more likely to detect periodicity and use
autocorrelation. For finely sampled signals where local behavior appears smooth, it
tends to use quadratic extrapolation.

# Examples
```julia
# For a left boundary condition on a signal
coefs = detect_boundary_signal_fast(signal, which_end=:left)

# For a right boundary
coefs = detect_boundary_signal_fast(signal, which_end=:right)
```
"""
function detect_boundary_signal_fast(y_centered, n, h; which_end::Symbol)
    
    # First, check for periodicity using signs of gradients
    local_window = min(5, n)
    if which_end == :left
        local_indices = 1:local_window
    else
        local_indices = max(1, n-local_window+1):n
    end
    local_y = view(y_centered, local_indices)

    # if gradients have sign changes above threshold
    if prod(extrema(diff(local_y)))/h^2 < -1/3
        # If periodic, use linear prediction mixed with quadratic extrapolation
        return periodic_boundary(y_centered)
    else
        # For non-periodic signals, default to quadratic extrapolation
        return [3.0, -3.0, 1.0]
    end
end