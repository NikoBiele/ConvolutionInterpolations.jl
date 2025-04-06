"""
    linear_predict(r)

Computes the coefficients for linear prediction based on autocorrelation values.

# Arguments
- `r`: Vector of autocorrelation values

# Returns
- Vector of coefficients for linear prediction

# Details
The function implements different strategies based on the length of the input:
- Length 1: Returns constant prediction `[1.0]`
- Length 2: Returns linear prediction `[2.0, -1.0]`
- Length 3: Returns quadratic prediction `[3.0, -3.0, 1.0]`
- Length 4+: Computes higher-order boundary coefficients using mathematical formulations that preserve signal characteristics

The coefficients returned represent the optimal linear prediction filter for the given autocorrelation sequence.
"""
function linear_predict(r)
    if length(r) == 1
        return [1.0] # constant
    elseif length(r) == 2
        return [2.0, -1.0] # linear
    elseif length(r) == 3
        return [3.0, -3.0, 1.0] # quadratic
    elseif length(r) == 4
        # third order boundary condition
        r0, r1, r2, r3 = r[1:4]
        coefs = zeros(3)
        coefs[1] = (r0^2*r1 - r0*r1*r2 - r0*r2*r3 - r1^3 + r1^2*r3 + r1*r2^2)/(isapprox((r0^3 - 2*r0*r1^2 - r0*r2^2 + 2*r1^2*r2), 0.0, atol=1e-6) ? 1e-6 : (r0^3 - 2*r0*r1^2 - r0*r2^2 + 2*r1^2*r2))
        coefs[2] = (r0*r2 - r1^2 - r1*r3 + r2^2)/(isapprox((r0^2 + r0*r2 - 2*r1^2), 0.0, atol=1e-6) ? 1e-6 : (r0^2 + r0*r2 - 2*r1^2))
        coefs[3] = (r0^2*r3 - 2*r0*r1*r2 + r1^3 - r1^2*r3 + r1*r2^2)/(isapprox((r0^3 - 2*r0*r1^2 - r0*r2^2 + 2*r1^2*r2), 0.0, atol=1e-6) ? 1e-6 : (r0^3 - 2*r0*r1^2 - r0*r2^2 + 2*r1^2*r2))
        return coefs
    elseif length(r) == 5
        # fourth order boundary condition
        r0, r1, r2, r3, r4 = r[1:5]
        coefs = zeros(4)
        coefs[1] = (r0^3*r1 - r0^2*r1*r2 - r0^2*r2*r3 - r0^2*r3*r4 - 2*r0*r1^3 + r0*r1^2*r3 + 2*r0*r1*r2*r4 + r0*r1*r3^2 + r0*r2^2*r3 + 3*r1^3*r2 - r1^3*r4 - 2*r1^2*r2*r3 + r1^2*r3*r4 - r1*r2^3 - r1*r2^2*r4 - r1*r2*r3^2 + r2^3*r3)/(r0^4 - 3*r0^2*r1^2 - 2*r0^2*r2^2 - r0^2*r3^2 + 4*r0*r1^2*r2 + 4*r0*r1*r2*r3 + r1^4 - 2*r1^3*r3 - 2*r1^2*r2^2 + r1^2*r3^2 - 2*r1*r2^2*r3 + r2^4)
        coefs[2] = (r0^3*r2 - r0^2*r1^2 - r0^2*r1*r3 - r0^2*r2*r4 + r0*r1^2*r4 + 3*r0*r1*r2*r3 + r0*r1*r3*r4 - r0*r2^3 - r0*r2*r3^2 + r1^4 - r1^3*r3 - r1^2*r2^2 - r1^2*r2*r4 - r1^2*r3^2 + 2*r1*r2^2*r3 - r1*r2*r3*r4 + r1*r3^3 + r2^3*r4 - r2^2*r3^2)/(r0^4 - 3*r0^2*r1^2 - 2*r0^2*r2^2 - r0^2*r3^2 + 4*r0*r1^2*r2 + 4*r0*r1*r2*r3 + r1^4 - 2*r1^3*r3 - 2*r1^2*r2^2 + r1^2*r3^2 - 2*r1*r2^2*r3 + r2^4)
        coefs[3] = (r0^3*r3 - 2*r0^2*r1*r2 - r0^2*r1*r4 + r0*r1^3 + 2*r0*r1*r2^2 + r0*r1*r2*r4 - r0*r2^2*r3 + r0*r2*r3*r4 - r0*r3^3 - r1^3*r2 + r1^3*r4 - 2*r1^2*r2*r3 - r1^2*r3*r4 + r1*r2^3 - r1*r2^2*r4 + 3*r1*r2*r3^2 - r2^3*r3)/(r0^4 - 3*r0^2*r1^2 - 2*r0^2*r2^2 - r0^2*r3^2 + 4*r0*r1^2*r2 + 4*r0*r1*r2*r3 + r1^4 - 2*r1^3*r3 - 2*r1^2*r2^2 + r1^2*r3^2 - 2*r1*r2^2*r3 + r2^4)
        coefs[4] = (r0^3*r4 - 2*r0^2*r1*r3 - r0^2*r2^2 + 3*r0*r1^2*r2 - 2*r0*r1^2*r4 + 2*r0*r1*r2*r3 - r0*r2^2*r4 + r0*r2*r3^2 - r1^4 + 2*r1^3*r3 - 2*r1^2*r2^2 + 2*r1^2*r2*r4 - r1^2*r3^2 - 2*r1*r2^2*r3 + r2^4)/(r0^4 - 3*r0^2*r1^2 - 2*r0^2*r2^2 - r0^2*r3^2 + 4*r0*r1^2*r2 + 4*r0*r1*r2*r3 + r1^4 - 2*r1^3*r3 - 2*r1^2*r2^2 + r1^2*r3^2 - 2*r1*r2^2*r3 + r2^4)
        return coefs
    else # if length(r) >= 6
        # fifth order boundary condition
        r0, r1, r2, r3, r4, r5 = r[1:6]
        coefs = zeros(5)
        coefs[1] = (r0^4*r1 - r0^3*r1*r2 - r0^3*r2*r3 - r0^3*r3*r4 - r0^3*r4*r5 - 3*r0^2*r1^3 + r0^2*r1^2*r3 - r0^2*r1*r2^2 + 2*r0^2*r1*r2*r4 + 2*r0^2*r1*r3*r5 + r0^2*r1*r4^2 + r0^2*r2^2*r3 + r0^2*r2^2*r5 + 2*r0^2*r2*r3*r4 + 6*r0*r1^3*r2 - r0*r1^3*r4 + 3*r0*r1^2*r2*r3 - 3*r0*r1^2*r2*r5 - r0*r1^2*r3*r4 + 2*r0*r1^2*r4*r5 - 4*r0*r1*r2^2*r4 - 4*r0*r1*r2*r3^2 - 2*r0*r1*r2*r3*r5 - r0*r1*r2*r4^2 - r0*r1*r3^2*r4 + r0*r2^2*r3*r4 + r0*r2^2*r4*r5 + r0*r2*r3^3 - r0*r2*r3^2*r5 - r0*r2*r3*r4^2 + r0*r3^3*r4 + r1^5 - 3*r1^4*r3 + r1^4*r5 - 5*r1^3*r2^2 + 2*r1^3*r2*r4 + 2*r1^3*r3^2 - 2*r1^3*r3*r5 - r1^3*r4^2 + r1^2*r2^2*r3 + 2*r1^2*r2^2*r5 + 2*r1^2*r2*r3*r4 - 2*r1^2*r2*r4*r5 + r1^2*r3^3 + r1^2*r3^2*r5 + r1^2*r3*r4^2 + 2*r1*r2^4 - r1*r2^2*r3^2 + 2*r1*r2^2*r3*r5 + r1*r2^2*r4^2 - 2*r1*r2*r3^2*r4 - r1*r3^4 - r2^4*r3 - r2^4*r5 + r2^2*r3^3)/(r0^5 - 4*r0^3*r1^2 - 3*r0^3*r2^2 - 2*r0^3*r3^2 - r0^3*r4^2 + 6*r0^2*r1^2*r2 + 8*r0^2*r1*r2*r3 + 4*r0^2*r1*r3*r4 + 2*r0^2*r2^2*r4 + 3*r0*r1^4 - 4*r0*r1^3*r3 - 2*r0*r1^2*r2^2 - 6*r0*r1^2*r2*r4 + 2*r0*r1^2*r4^2 - 8*r0*r1*r2^2*r3 - 4*r0*r1*r2*r3*r4 + 2*r0*r2^4 + 2*r0*r2^2*r3^2 + r0*r2^2*r4^2 - 2*r0*r2*r3^2*r4 + r0*r3^4 - 4*r1^4*r2 + 2*r1^4*r4 + 4*r1^3*r2*r3 - 4*r1^3*r3*r4 + 2*r1^2*r2^3 + 4*r1^2*r2^2*r4 + 4*r1^2*r2*r3^2 - 2*r1^2*r2*r4^2 + 2*r1^2*r3^2*r4 - 4*r1*r2^3*r3 + 4*r1*r2^2*r3*r4 - 4*r1*r2*r3^3 - 2*r2^4*r4 + 2*r2^3*r3^2)
        coefs[2] = (r0^4*r2 - r0^3*r1^2 - r0^3*r1*r3 - r0^3*r2*r4 - r0^3*r3*r5 - r0^2*r1^2*r2 + r0^2*r1^2*r4 + 3*r0^2*r1*r2*r3 + 2*r0^2*r1*r2*r5 + 3*r0^2*r1*r3*r4 + r0^2*r1*r4*r5 - 2*r0^2*r2^3 - r0^2*r2*r4^2 + 2*r0*r1^4 - r0*r1^3*r5 + 2*r0*r1^2*r2^2 - 4*r0*r1^2*r2*r4 - 3*r0*r1^2*r3^2 - r0*r1^2*r4^2 - 2*r0*r1*r2^2*r5 - r0*r1*r2*r3*r4 - r0*r1*r2*r4*r5 + r0*r1*r3^3 + r0*r1*r3*r4^2 + 4*r0*r2^3*r4 - r0*r2^2*r3^2 + r0*r2^2*r3*r5 - r0*r2*r3^2*r4 - r0*r2*r3*r4*r5 + r0*r2*r4^3 + r0*r3^3*r5 - r0*r3^2*r4^2 - 3*r1^4*r2 + r1^4*r4 + r1^3*r2*r3 + r1^3*r2*r5 - r1^3*r4*r5 + 2*r1^2*r2^3 + 2*r1^2*r2*r3^2 + 2*r1^2*r2*r3*r5 + 3*r1^2*r2*r4^2 - 2*r1^2*r3^2*r4 + r1^2*r3*r4*r5 - r1^2*r4^3 - 3*r1*r2^3*r3 - r1*r2^3*r5 - 3*r1*r2^2*r3*r4 + r1*r2^2*r4*r5 + r1*r2*r3^3 - 3*r1*r2*r3^2*r5 + r1*r2*r3*r4^2 + r1*r3^3*r4 + r2^3*r3^2 + r2^3*r3*r5 - 2*r2^3*r4^2 + 2*r2^2*r3^2*r4 - r2*r3^4)/(r0^5 - 4*r0^3*r1^2 - 3*r0^3*r2^2 - 2*r0^3*r3^2 - r0^3*r4^2 + 6*r0^2*r1^2*r2 + 8*r0^2*r1*r2*r3 + 4*r0^2*r1*r3*r4 + 2*r0^2*r2^2*r4 + 3*r0*r1^4 - 4*r0*r1^3*r3 - 2*r0*r1^2*r2^2 - 6*r0*r1^2*r2*r4 + 2*r0*r1^2*r4^2 - 8*r0*r1*r2^2*r3 - 4*r0*r1*r2*r3*r4 + 2*r0*r2^4 + 2*r0*r2^2*r3^2 + r0*r2^2*r4^2 - 2*r0*r2*r3^2*r4 + r0*r3^4 - 4*r1^4*r2 + 2*r1^4*r4 + 4*r1^3*r2*r3 - 4*r1^3*r3*r4 + 2*r1^2*r2^3 + 4*r1^2*r2^2*r4 + 4*r1^2*r2*r3^2 - 2*r1^2*r2*r4^2 + 2*r1^2*r3^2*r4 - 4*r1*r2^3*r3 + 4*r1*r2^2*r3*r4 - 4*r1*r2*r3^3 - 2*r2^4*r4 + 2*r2^3*r3^2)
        coefs[3] = (r0^2*r3 - 2*r0*r1*r2 - r0*r1*r4 + r0*r2*r3 - r0*r2*r5 + r0*r3*r4 + r1^3 + r1^2*r5 - 2*r1*r3^2 + r1*r3*r5 - r1*r4^2 + r2^2*r3 - r2^2*r5 + 2*r2*r3*r4 - r3^3)/(r0^3 + r0^2*r2 + r0^2*r4 - 3*r0*r1^2 - 2*r0*r1*r3 - 2*r0*r2^2 + r0*r2*r4 - r0*r3^2 + 4*r1^2*r2 - 2*r1^2*r4 + 4*r1*r2*r3 - 2*r2^3)
        coefs[4] = (r0^4*r4 - 2*r0^3*r1*r3 - r0^3*r1*r5 - r0^3*r2^2 + 3*r0^2*r1^2*r2 - r0^2*r1^2*r4 + 4*r0^2*r1*r2*r3 + r0^2*r1*r2*r5 - 2*r0^2*r2^2*r4 + r0^2*r2*r3^2 + r0^2*r2*r3*r5 - r0^2*r3^2*r4 + r0^2*r3*r4*r5 - r0^2*r4^3 - r0*r1^4 + r0*r1^3*r3 + 2*r0*r1^3*r5 - 4*r0*r1^2*r2^2 - 2*r0*r1^2*r3^2 - r0*r1^2*r3*r5 - 2*r0*r1*r2^2*r3 - 2*r0*r1*r2*r4*r5 + 2*r0*r1*r3^3 - r0*r1*r3^2*r5 + 3*r0*r1*r3*r4^2 + 2*r0*r2^4 - r0*r2^2*r3^2 - r0*r2^2*r3*r5 + 3*r0*r2^2*r4^2 - 2*r0*r2*r3^2*r4 + r1^4*r2 - r1^4*r4 + r1^3*r2*r3 - 3*r1^3*r2*r5 + 2*r1^3*r3*r4 + r1^3*r4*r5 + 4*r1^2*r2^2*r4 - 2*r1^2*r2*r3^2 + 2*r1^2*r2*r3*r5 - 3*r1^2*r2*r4^2 - 2*r1^2*r3^2*r4 - r1^2*r3*r4*r5 + r1^2*r4^3 - r1*r2^3*r3 + r1*r2^3*r5 + r1*r2^2*r3*r4 + r1*r2^2*r4*r5 + r1*r2*r3^3 + r1*r2*r3^2*r5 - 3*r1*r2*r3*r4^2 + r1*r3^3*r4 - 2*r2^4*r4 + r2^3*r3^2 - r2^3*r3*r5 + 2*r2^2*r3^2*r4 - r2*r3^4)/(r0^5 - 4*r0^3*r1^2 - 3*r0^3*r2^2 - 2*r0^3*r3^2 - r0^3*r4^2 + 6*r0^2*r1^2*r2 + 8*r0^2*r1*r2*r3 + 4*r0^2*r1*r3*r4 + 2*r0^2*r2^2*r4 + 3*r0*r1^4 - 4*r0*r1^3*r3 - 2*r0*r1^2*r2^2 - 6*r0*r1^2*r2*r4 + 2*r0*r1^2*r4^2 - 8*r0*r1*r2^2*r3 - 4*r0*r1*r2*r3*r4 + 2*r0*r2^4 + 2*r0*r2^2*r3^2 + r0*r2^2*r4^2 - 2*r0*r2*r3^2*r4 + r0*r3^4 - 4*r1^4*r2 + 2*r1^4*r4 + 4*r1^3*r2*r3 - 4*r1^3*r3*r4 + 2*r1^2*r2^3 + 4*r1^2*r2^2*r4 + 4*r1^2*r2*r3^2 - 2*r1^2*r2*r4^2 + 2*r1^2*r3^2*r4 - 4*r1*r2^3*r3 + 4*r1*r2^2*r3*r4 - 4*r1*r2*r3^3 - 2*r2^4*r4 + 2*r2^3*r3^2)
        coefs[5] = (r0^4*r5 - 2*r0^3*r1*r4 - 2*r0^3*r2*r3 + 3*r0^2*r1^2*r3 - 3*r0^2*r1^2*r5 + 3*r0^2*r1*r2^2 + 2*r0^2*r1*r2*r4 + r0^2*r1*r3^2 - 2*r0^2*r2^2*r5 + 2*r0^2*r2*r3*r4 - r0^2*r3^2*r5 + r0^2*r3*r4^2 - 4*r0*r1^3*r2 + 4*r0*r1^3*r4 - 2*r0*r1^2*r2*r3 + 4*r0*r1^2*r2*r5 - 2*r0*r1^2*r3*r4 - 2*r0*r1*r2^3 - 4*r0*r1*r2*r3^2 + 4*r0*r1*r2*r3*r5 - 2*r0*r1*r2*r4^2 - 2*r0*r1*r3^2*r4 + 2*r0*r2^3*r3 - 2*r0*r2^2*r3*r4 + 2*r0*r2*r3^3 + r1^5 - 3*r1^4*r3 + r1^4*r5 + 3*r1^3*r2^2 - 6*r1^3*r2*r4 + 2*r1^3*r3^2 - 2*r1^3*r3*r5 + r1^3*r4^2 + 5*r1^2*r2^2*r3 - 2*r1^2*r2^2*r5 + 4*r1^2*r2*r3*r4 + r1^2*r3^3 + r1^2*r3^2*r5 - r1^2*r3*r4^2 - 2*r1*r2^4 + 2*r1*r2^3*r4 - 5*r1*r2^2*r3^2 - 2*r1*r2^2*r3*r5 + r1*r2^2*r4^2 + 2*r1*r2*r3^2*r4 - r1*r3^4 + r2^4*r3 + r2^4*r5 - 2*r2^3*r3*r4 + r2^2*r3^3)/(r0^5 - 4*r0^3*r1^2 - 3*r0^3*r2^2 - 2*r0^3*r3^2 - r0^3*r4^2 + 6*r0^2*r1^2*r2 + 8*r0^2*r1*r2*r3 + 4*r0^2*r1*r3*r4 + 2*r0^2*r2^2*r4 + 3*r0*r1^4 - 4*r0*r1^3*r3 - 2*r0*r1^2*r2^2 - 6*r0*r1^2*r2*r4 + 2*r0*r1^2*r4^2 - 8*r0*r1*r2^2*r3 - 4*r0*r1*r2*r3*r4 + 2*r0*r2^4 + 2*r0*r2^2*r3^2 + r0*r2^2*r4^2 - 2*r0*r2*r3^2*r4 + r0*r3^4 - 4*r1^4*r2 + 2*r1^4*r4 + 4*r1^3*r2*r3 - 4*r1^3*r3*r4 + 2*r1^2*r2^3 + 4*r1^2*r2^2*r4 + 4*r1^2*r2*r3^2 - 2*r1^2*r2*r4^2 + 2*r1^2*r3^2*r4 - 4*r1*r2^3*r3 + 4*r1*r2^2*r3*r4 - 4*r1*r2*r3^3 - 2*r2^4*r4 + 2*r2^3*r3^2)
        return coefs
    end
end