"""
    periodic_boundary(y_centered)

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
function periodic_boundary(y_centered::Vector{T}) where T
    autocorrelation_coefs = autocor_coefs(y_centered)
    quadratic_coefs = T[T(3), T(-3), one(T)]
    return T[i <= length(quadratic_coefs) ?
                (quadratic_coefs[i]+autocorrelation_coefs[i])/2 :
                autocorrelation_coefs[i] for i in eachindex(autocorrelation_coefs)]
end