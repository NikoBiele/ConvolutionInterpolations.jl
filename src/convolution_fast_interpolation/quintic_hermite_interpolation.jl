"""
    quintic_hermite(t, f0, f1, d0, d1, dd0, dd1, h)

Quintic Hermite interpolation using values, first and second derivatives at endpoints.

# Arguments
- `t`: Local coordinate in [0, 1]
- `f0, f1`: Function values at left and right endpoints
- `d0, d1`: First derivatives at left and right endpoints
- `dd0, dd1`: Second derivatives at left and right endpoints
- `h`: Spacing between endpoints (for scaling derivatives)

# Returns
Interpolated value with O(h⁶) error for C⁶ functions.
"""
@inline function quintic_hermite(t::T, f0::T, f1::T, d0::T, d1::T, dd0::T, dd1::T, h::T) where T
    t2 = t * t
    t3 = t2 * t
    t4 = t3 * t
    t5 = t4 * t
    
    # Basis for values
    h00 = one(T) - 10t3 + 15t4 - 6t5
    h01 = 10t3 - 15t4 + 6t5
    
    # Basis for first derivatives (scaled by h)
    h10 = t - 6t3 + 8t4 - 3t5
    h11 = -4t3 + 7t4 - 3t5
    
    # Basis for second derivatives (scaled by h²/2)
    h20 = t2 - 3t3 + 3t4 - t5
    h21 = t3 - 2t4 + t5
    
    h2 = h * h
    
    return h00*f0 + h01*f1 + h*(h10*d0 + h11*d1) + (h2/2)*(h20*dd0 + h21*dd1)
end