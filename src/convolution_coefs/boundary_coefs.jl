"""
    boundary_coefs(y, h::T) where T<:Number

Computes boundary coefficients for a given 1D signal slice.

# Arguments
- `y`: Vector representing the signal slice
- `h`: Scaling parameter (of type T<:Number)

# Returns
- Tuple containing:
  1. Vector of prediction coefficients
  2. Mean value of the signal (offset)
  3. Mean-centered signal values

# Details
This function analyzes the signal characteristics and selects the appropriate boundary method:
1. For highly linear signals (correlation > 0.95), uses linear prediction `[2.0, -1.0, 0.0]`
2. For highly quadratic signals (correlation > 0.95 and length > 5), uses quadratic prediction `[3.0, -3.0, 1.0]`
3. For other signals, computes autocorrelation-based coefficients assuming periodicity

The function decomposes the signal into trend (offset) and fluctuation components for more accurate boundary handling.
"""
function boundary_coefs(y, h::T) where T<:Number

    # check for linear signal
    n = length(y)
    y_mean = sum(y)/n
    x_uniform = 1:n
    x_mean = (n + 1) / 2
    x_centered = x_uniform .- x_mean
    y_centered = y .- y_mean    
    x_norm = sqrt(sum(x_centered.^2))
    y_norm = sqrt(sum(y_centered.^2))
    linear_score = abs(sum(x_centered.*y_centered) / (x_norm * y_norm))
    if linear_score > 0.9
        return [2.0, -1.0, 0.0], y_mean, y_centered
    end

    # check for quadratic signal
    x_squared_centered = x_centered.^2 .- sum(x_centered.^2)/n
    quadratic_score = abs(sum(x_squared_centered.*y_centered) / (sqrt(sum(x_squared_centered.^2)) * y_norm))
    if quadratic_score > 0.9 && n > 5
        return [3.0, -3.0, 1.0], y_mean, y_centered
    end

    # assume periodic signal
    return autocor_coefs(y_centered), y_mean, y_centered
end