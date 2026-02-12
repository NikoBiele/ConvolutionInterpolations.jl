"""
    autocor_coefs(signal)

Computes prediction coefficients from a signal using autocorrelation analysis.

# Arguments
- `signal`: Vector representing the input signal

# Returns
- Vector of coefficients for linear prediction

# Details
This function:
1. Computes the autocorrelation function of the input signal
2. Derives prediction coefficients from the autocorrelation values
3. Falls back to quadratic prediction `[3.0, -3.0, 1.0]` if numerical instability is detected
"""
function autocor_coefs(signal::Vector{T}) where T
    acf = autocorrelation(signal)
    coefs = linear_predict(acf)
    if occursin("NaN", string(coefs))
        return T[T(3), T(-3), T(1)] # fallback to quadratic
    end
    return coefs
end