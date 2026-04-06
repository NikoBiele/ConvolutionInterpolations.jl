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

The coefficients returned represent the linear prediction filter for the given autocorrelation sequence.
"""
function linear_predict(r::NTuple{NS,G}) where {NS,G}
    if length(r) == 1
        return (one(G), zero(G), zero(G), zero(G), zero(G)) # constant
    elseif length(r) == 2
        return (G(2), -one(G), zero(G), zero(G), zero(G)) # linear
    elseif length(r) == 3
        return (G(3), G(-3), one(G), zero(G), zero(G)) # quadratic
    elseif length(r) == 4
        # third order boundary condition
        r0, r1, r2, r3 = r[1:4]
        coefs = ((r0^2*r1 - r0*r1*r2 - r0*r2*r3 - r1^3 + r1^2*r3 + r1*r2^2)/(isapprox((r0^3 - 2*r0*r1^2 - r0*r2^2 + 2*r1^2*r2), zero(G), atol=1e-6) ? 1e-6 : (r0^3 - 2*r0*r1^2 - r0*r2^2 + 2*r1^2*r2)),
                 (r0*r2 - r1^2 - r1*r3 + r2^2)/(isapprox((r0^2 + r0*r2 - 2*r1^2), zero(G), atol=1e-6) ? 1e-6 : (r0^2 + r0*r2 - 2*r1^2)),
                 (r0^2*r3 - 2*r0*r1*r2 + r1^3 - r1^2*r3 + r1*r2^2)/(isapprox((r0^3 - 2*r0*r1^2 - r0*r2^2 + 2*r1^2*r2), zero(G), atol=1e-6) ? 1e-6 : (r0^3 - 2*r0*r1^2 - r0*r2^2 + 2*r1^2*r2)),
                 zero(G),
                 zero(G))
        return coefs
    elseif length(r) == 5
        # fourth order boundary condition
        r0, r1, r2, r3, r4 = r[1:5]
        coefs = ((r0^3*r1 - r0^2*r1*r2 - r0^2*r2*r3 - r0^2*r3*r4 - 2*r0*r1^3 + r0*r1^2*r3 + 2*r0*r1*r2*r4 + r0*r1*r3^2 + r0*r2^2*r3 + 3*r1^3*r2 - r1^3*r4 - 2*r1^2*r2*r3 + r1^2*r3*r4 - r1*r2^3 - r1*r2^2*r4 - r1*r2*r3^2 + r2^3*r3)/(r0^4 - 3*r0^2*r1^2 - 2*r0^2*r2^2 - r0^2*r3^2 + 4*r0*r1^2*r2 + 4*r0*r1*r2*r3 + r1^4 - 2*r1^3*r3 - 2*r1^2*r2^2 + r1^2*r3^2 - 2*r1*r2^2*r3 + r2^4),
                 (r0^3*r2 - r0^2*r1^2 - r0^2*r1*r3 - r0^2*r2*r4 + r0*r1^2*r4 + 3*r0*r1*r2*r3 + r0*r1*r3*r4 - r0*r2^3 - r0*r2*r3^2 + r1^4 - r1^3*r3 - r1^2*r2^2 - r1^2*r2*r4 - r1^2*r3^2 + 2*r1*r2^2*r3 - r1*r2*r3*r4 + r1*r3^3 + r2^3*r4 - r2^2*r3^2)/(r0^4 - 3*r0^2*r1^2 - 2*r0^2*r2^2 - r0^2*r3^2 + 4*r0*r1^2*r2 + 4*r0*r1*r2*r3 + r1^4 - 2*r1^3*r3 - 2*r1^2*r2^2 + r1^2*r3^2 - 2*r1*r2^2*r3 + r2^4),
                 (r0^3*r3 - 2*r0^2*r1*r2 - r0^2*r1*r4 + r0*r1^3 + 2*r0*r1*r2^2 + r0*r1*r2*r4 - r0*r2^2*r3 + r0*r2*r3*r4 - r0*r3^3 - r1^3*r2 + r1^3*r4 - 2*r1^2*r2*r3 - r1^2*r3*r4 + r1*r2^3 - r1*r2^2*r4 + 3*r1*r2*r3^2 - r2^3*r3)/(r0^4 - 3*r0^2*r1^2 - 2*r0^2*r2^2 - r0^2*r3^2 + 4*r0*r1^2*r2 + 4*r0*r1*r2*r3 + r1^4 - 2*r1^3*r3 - 2*r1^2*r2^2 + r1^2*r3^2 - 2*r1*r2^2*r3 + r2^4),
                 (r0^3*r4 - 2*r0^2*r1*r3 - r0^2*r2^2 + 3*r0*r1^2*r2 - 2*r0*r1^2*r4 + 2*r0*r1*r2*r3 - r0*r2^2*r4 + r0*r2*r3^2 - r1^4 + 2*r1^3*r3 - 2*r1^2*r2^2 + 2*r1^2*r2*r4 - r1^2*r3^2 - 2*r1*r2^2*r3 + r2^4)/(r0^4 - 3*r0^2*r1^2 - 2*r0^2*r2^2 - r0^2*r3^2 + 4*r0*r1^2*r2 + 4*r0*r1*r2*r3 + r1^4 - 2*r1^3*r3 - 2*r1^2*r2^2 + r1^2*r3^2 - 2*r1*r2^2*r3 + r2^4),
                 zero(G))
        return coefs
    else # if length(r) >= 6
        # fifth order boundary condition
        r0, r1, r2, r3, r4, r5 = r[1:6]
        coefs = ((r0^4*r1 - r0^3*r1*r2 - r0^3*r2*r3 - r0^3*r3*r4 - r0^3*r4*r5 - 3*r0^2*r1^3 + r0^2*r1^2*r3 - r0^2*r1*r2^2 + 2*r0^2*r1*r2*r4 + 2*r0^2*r1*r3*r5 + r0^2*r1*r4^2 + r0^2*r2^2*r3 + r0^2*r2^2*r5 + 2*r0^2*r2*r3*r4 + 6*r0*r1^3*r2 - r0*r1^3*r4 + 3*r0*r1^2*r2*r3 - 3*r0*r1^2*r2*r5 - r0*r1^2*r3*r4 + 2*r0*r1^2*r4*r5 - 4*r0*r1*r2^2*r4 - 4*r0*r1*r2*r3^2 - 2*r0*r1*r2*r3*r5 - r0*r1*r2*r4^2 - r0*r1*r3^2*r4 + r0*r2^2*r3*r4 + r0*r2^2*r4*r5 + r0*r2*r3^3 - r0*r2*r3^2*r5 - r0*r2*r3*r4^2 + r0*r3^3*r4 + r1^5 - 3*r1^4*r3 + r1^4*r5 - 5*r1^3*r2^2 + 2*r1^3*r2*r4 + 2*r1^3*r3^2 - 2*r1^3*r3*r5 - r1^3*r4^2 + r1^2*r2^2*r3 + 2*r1^2*r2^2*r5 + 2*r1^2*r2*r3*r4 - 2*r1^2*r2*r4*r5 + r1^2*r3^3 + r1^2*r3^2*r5 + r1^2*r3*r4^2 + 2*r1*r2^4 - r1*r2^2*r3^2 + 2*r1*r2^2*r3*r5 + r1*r2^2*r4^2 - 2*r1*r2*r3^2*r4 - r1*r3^4 - r2^4*r3 - r2^4*r5 + r2^2*r3^3)/(r0^5 - 4*r0^3*r1^2 - 3*r0^3*r2^2 - 2*r0^3*r3^2 - r0^3*r4^2 + 6*r0^2*r1^2*r2 + 8*r0^2*r1*r2*r3 + 4*r0^2*r1*r3*r4 + 2*r0^2*r2^2*r4 + 3*r0*r1^4 - 4*r0*r1^3*r3 - 2*r0*r1^2*r2^2 - 6*r0*r1^2*r2*r4 + 2*r0*r1^2*r4^2 - 8*r0*r1*r2^2*r3 - 4*r0*r1*r2*r3*r4 + 2*r0*r2^4 + 2*r0*r2^2*r3^2 + r0*r2^2*r4^2 - 2*r0*r2*r3^2*r4 + r0*r3^4 - 4*r1^4*r2 + 2*r1^4*r4 + 4*r1^3*r2*r3 - 4*r1^3*r3*r4 + 2*r1^2*r2^3 + 4*r1^2*r2^2*r4 + 4*r1^2*r2*r3^2 - 2*r1^2*r2*r4^2 + 2*r1^2*r3^2*r4 - 4*r1*r2^3*r3 + 4*r1*r2^2*r3*r4 - 4*r1*r2*r3^3 - 2*r2^4*r4 + 2*r2^3*r3^2),
                 (r0^4*r2 - r0^3*r1^2 - r0^3*r1*r3 - r0^3*r2*r4 - r0^3*r3*r5 - r0^2*r1^2*r2 + r0^2*r1^2*r4 + 3*r0^2*r1*r2*r3 + 2*r0^2*r1*r2*r5 + 3*r0^2*r1*r3*r4 + r0^2*r1*r4*r5 - 2*r0^2*r2^3 - r0^2*r2*r4^2 + 2*r0*r1^4 - r0*r1^3*r5 + 2*r0*r1^2*r2^2 - 4*r0*r1^2*r2*r4 - 3*r0*r1^2*r3^2 - r0*r1^2*r4^2 - 2*r0*r1*r2^2*r5 - r0*r1*r2*r3*r4 - r0*r1*r2*r4*r5 + r0*r1*r3^3 + r0*r1*r3*r4^2 + 4*r0*r2^3*r4 - r0*r2^2*r3^2 + r0*r2^2*r3*r5 - r0*r2*r3^2*r4 - r0*r2*r3*r4*r5 + r0*r2*r4^3 + r0*r3^3*r5 - r0*r3^2*r4^2 - 3*r1^4*r2 + r1^4*r4 + r1^3*r2*r3 + r1^3*r2*r5 - r1^3*r4*r5 + 2*r1^2*r2^3 + 2*r1^2*r2*r3^2 + 2*r1^2*r2*r3*r5 + 3*r1^2*r2*r4^2 - 2*r1^2*r3^2*r4 + r1^2*r3*r4*r5 - r1^2*r4^3 - 3*r1*r2^3*r3 - r1*r2^3*r5 - 3*r1*r2^2*r3*r4 + r1*r2^2*r4*r5 + r1*r2*r3^3 - 3*r1*r2*r3^2*r5 + r1*r2*r3*r4^2 + r1*r3^3*r4 + r2^3*r3^2 + r2^3*r3*r5 - 2*r2^3*r4^2 + 2*r2^2*r3^2*r4 - r2*r3^4)/(r0^5 - 4*r0^3*r1^2 - 3*r0^3*r2^2 - 2*r0^3*r3^2 - r0^3*r4^2 + 6*r0^2*r1^2*r2 + 8*r0^2*r1*r2*r3 + 4*r0^2*r1*r3*r4 + 2*r0^2*r2^2*r4 + 3*r0*r1^4 - 4*r0*r1^3*r3 - 2*r0*r1^2*r2^2 - 6*r0*r1^2*r2*r4 + 2*r0*r1^2*r4^2 - 8*r0*r1*r2^2*r3 - 4*r0*r1*r2*r3*r4 + 2*r0*r2^4 + 2*r0*r2^2*r3^2 + r0*r2^2*r4^2 - 2*r0*r2*r3^2*r4 + r0*r3^4 - 4*r1^4*r2 + 2*r1^4*r4 + 4*r1^3*r2*r3 - 4*r1^3*r3*r4 + 2*r1^2*r2^3 + 4*r1^2*r2^2*r4 + 4*r1^2*r2*r3^2 - 2*r1^2*r2*r4^2 + 2*r1^2*r3^2*r4 - 4*r1*r2^3*r3 + 4*r1*r2^2*r3*r4 - 4*r1*r2*r3^3 - 2*r2^4*r4 + 2*r2^3*r3^2),
                 (r0^2*r3 - 2*r0*r1*r2 - r0*r1*r4 + r0*r2*r3 - r0*r2*r5 + r0*r3*r4 + r1^3 + r1^2*r5 - 2*r1*r3^2 + r1*r3*r5 - r1*r4^2 + r2^2*r3 - r2^2*r5 + 2*r2*r3*r4 - r3^3)/(r0^3 + r0^2*r2 + r0^2*r4 - 3*r0*r1^2 - 2*r0*r1*r3 - 2*r0*r2^2 + r0*r2*r4 - r0*r3^2 + 4*r1^2*r2 - 2*r1^2*r4 + 4*r1*r2*r3 - 2*r2^3),
                 (r0^4*r4 - 2*r0^3*r1*r3 - r0^3*r1*r5 - r0^3*r2^2 + 3*r0^2*r1^2*r2 - r0^2*r1^2*r4 + 4*r0^2*r1*r2*r3 + r0^2*r1*r2*r5 - 2*r0^2*r2^2*r4 + r0^2*r2*r3^2 + r0^2*r2*r3*r5 - r0^2*r3^2*r4 + r0^2*r3*r4*r5 - r0^2*r4^3 - r0*r1^4 + r0*r1^3*r3 + 2*r0*r1^3*r5 - 4*r0*r1^2*r2^2 - 2*r0*r1^2*r3^2 - r0*r1^2*r3*r5 - 2*r0*r1*r2^2*r3 - 2*r0*r1*r2*r4*r5 + 2*r0*r1*r3^3 - r0*r1*r3^2*r5 + 3*r0*r1*r3*r4^2 + 2*r0*r2^4 - r0*r2^2*r3^2 - r0*r2^2*r3*r5 + 3*r0*r2^2*r4^2 - 2*r0*r2*r3^2*r4 + r1^4*r2 - r1^4*r4 + r1^3*r2*r3 - 3*r1^3*r2*r5 + 2*r1^3*r3*r4 + r1^3*r4*r5 + 4*r1^2*r2^2*r4 - 2*r1^2*r2*r3^2 + 2*r1^2*r2*r3*r5 - 3*r1^2*r2*r4^2 - 2*r1^2*r3^2*r4 - r1^2*r3*r4*r5 + r1^2*r4^3 - r1*r2^3*r3 + r1*r2^3*r5 + r1*r2^2*r3*r4 + r1*r2^2*r4*r5 + r1*r2*r3^3 + r1*r2*r3^2*r5 - 3*r1*r2*r3*r4^2 + r1*r3^3*r4 - 2*r2^4*r4 + r2^3*r3^2 - r2^3*r3*r5 + 2*r2^2*r3^2*r4 - r2*r3^4)/(r0^5 - 4*r0^3*r1^2 - 3*r0^3*r2^2 - 2*r0^3*r3^2 - r0^3*r4^2 + 6*r0^2*r1^2*r2 + 8*r0^2*r1*r2*r3 + 4*r0^2*r1*r3*r4 + 2*r0^2*r2^2*r4 + 3*r0*r1^4 - 4*r0*r1^3*r3 - 2*r0*r1^2*r2^2 - 6*r0*r1^2*r2*r4 + 2*r0*r1^2*r4^2 - 8*r0*r1*r2^2*r3 - 4*r0*r1*r2*r3*r4 + 2*r0*r2^4 + 2*r0*r2^2*r3^2 + r0*r2^2*r4^2 - 2*r0*r2*r3^2*r4 + r0*r3^4 - 4*r1^4*r2 + 2*r1^4*r4 + 4*r1^3*r2*r3 - 4*r1^3*r3*r4 + 2*r1^2*r2^3 + 4*r1^2*r2^2*r4 + 4*r1^2*r2*r3^2 - 2*r1^2*r2*r4^2 + 2*r1^2*r3^2*r4 - 4*r1*r2^3*r3 + 4*r1*r2^2*r3*r4 - 4*r1*r2*r3^3 - 2*r2^4*r4 + 2*r2^3*r3^2),
                 (r0^4*r5 - 2*r0^3*r1*r4 - 2*r0^3*r2*r3 + 3*r0^2*r1^2*r3 - 3*r0^2*r1^2*r5 + 3*r0^2*r1*r2^2 + 2*r0^2*r1*r2*r4 + r0^2*r1*r3^2 - 2*r0^2*r2^2*r5 + 2*r0^2*r2*r3*r4 - r0^2*r3^2*r5 + r0^2*r3*r4^2 - 4*r0*r1^3*r2 + 4*r0*r1^3*r4 - 2*r0*r1^2*r2*r3 + 4*r0*r1^2*r2*r5 - 2*r0*r1^2*r3*r4 - 2*r0*r1*r2^3 - 4*r0*r1*r2*r3^2 + 4*r0*r1*r2*r3*r5 - 2*r0*r1*r2*r4^2 - 2*r0*r1*r3^2*r4 + 2*r0*r2^3*r3 - 2*r0*r2^2*r3*r4 + 2*r0*r2*r3^3 + r1^5 - 3*r1^4*r3 + r1^4*r5 + 3*r1^3*r2^2 - 6*r1^3*r2*r4 + 2*r1^3*r3^2 - 2*r1^3*r3*r5 + r1^3*r4^2 + 5*r1^2*r2^2*r3 - 2*r1^2*r2^2*r5 + 4*r1^2*r2*r3*r4 + r1^2*r3^3 + r1^2*r3^2*r5 - r1^2*r3*r4^2 - 2*r1*r2^4 + 2*r1*r2^3*r4 - 5*r1*r2^2*r3^2 - 2*r1*r2^2*r3*r5 + r1*r2^2*r4^2 + 2*r1*r2*r3^2*r4 - r1*r3^4 + r2^4*r3 + r2^4*r5 - 2*r2^3*r3*r4 + r2^2*r3^3)/(r0^5 - 4*r0^3*r1^2 - 3*r0^3*r2^2 - 2*r0^3*r3^2 - r0^3*r4^2 + 6*r0^2*r1^2*r2 + 8*r0^2*r1*r2*r3 + 4*r0^2*r1*r3*r4 + 2*r0^2*r2^2*r4 + 3*r0*r1^4 - 4*r0*r1^3*r3 - 2*r0*r1^2*r2^2 - 6*r0*r1^2*r2*r4 + 2*r0*r1^2*r4^2 - 8*r0*r1*r2^2*r3 - 4*r0*r1*r2*r3*r4 + 2*r0*r2^4 + 2*r0*r2^2*r3^2 + r0*r2^2*r4^2 - 2*r0*r2*r3^2*r4 + r0*r3^4 - 4*r1^4*r2 + 2*r1^4*r4 + 4*r1^3*r2*r3 - 4*r1^3*r3*r4 + 2*r1^2*r2^3 + 4*r1^2*r2^2*r4 + 4*r1^2*r2*r3^2 - 2*r1^2*r2*r4^2 + 2*r1^2*r3^2*r4 - 4*r1*r2^3*r3 + 4*r1*r2^2*r3*r4 - 4*r1*r2*r3^3 - 2*r2^4*r4 + 2*r2^3*r3^2))
        return coefs
    end
end