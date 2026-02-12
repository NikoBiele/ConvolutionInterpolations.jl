"""
    horner(x::T, coeffs::Dict{Symbol,Vector{S}}, poly::Symbol) where {T,S}

Efficiently evaluates a polynomial at point `x` using Horner's method.

# Arguments
- `x::T`: The point at which to evaluate the polynomial
- `coeffs::Dict{Symbol,Vector{S}}`: A dictionary mapping polynomial identifiers to coefficient vectors
- `poly::Symbol`: The identifier of the polynomial to evaluate

# Returns
- The value of the polynomial at point `x`

# Details
This function implements Horner's method for polynomial evaluation, which is numerically
stable and computationally efficient. It evaluates a polynomial of the form:

``p(x) = c_1 + c_2 x + c_3 x^2 + ... + c_n x^{n-1}``

using the coefficients provided in `coeffs[poly]`.

The coefficients in the vector should be ordered from the constant term (power 0) 
to the highest-degree term. The function uses type promotion to ensure proper
numerical precision in the result.

This function is primarily used internally by the convolution kernel implementations
to evaluate the piecewise polynomial segments efficiently.

# Examples
```julia
# Define coefficients for different polynomial segments
coefs = Dict(
    :eq1 => [1.0, 0.0, -7/3, 4/3],  # 1 - 7/3*x^2 + 4/3*x^3
    :eq2 => [15/6, -59/12, 3.0, -7/12]
)

# Evaluate the first polynomial at x = 0.5
value = horner(0.5, coefs, :eq1)
```
"""

function horner(x::T, coef_dict, key, ::Type{T}, DO::P) where {T,P<:Integer}
    
    coef = coef_dict[key]
    result = T(coef[end])
    for i in length(coef)-1:-1:(1+(DO>0 ? DO : 0))
        result = result * x + T(coef[i])
    end
    return result
end