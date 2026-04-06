"""
    periodic_boundary(y_centered::NTuple{NS,T}) where {NS,T}

Computes the coefficients for periodic boundary conditions based on a mix of:
- Autocorrelation-based coefficients for linear prediction
- Quadratic extrapolation coefficients.
This mix works well since the autocorrelation naturally decays, while quadratic extrapolation naturally grows.
Together these effects approximately balances out, creating a boundary condition suitable for periodic signals.

# Arguments
- `y_centered`: Vector representing the mean-centered signal values

# Returns
- Vector of coefficients for periodic boundary conditions
"""
function periodic_boundary(y_centered::NTuple{NS,T}) where {NS,T}
    autocorrelation_coefs = autocor_coefs(y_centered)
    length_coefs = length(autocorrelation_coefs)
    return ntuple(i -> i <= 3 ? (autocorrelation_coefs[i] + (i == 1 ? T(3) : i == 2 ? T(-3) : one(T))) / 2 : autocorrelation_coefs[i], length_coefs)
end