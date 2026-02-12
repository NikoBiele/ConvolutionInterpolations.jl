"""
    cubic_hermite(t, f0, f1, d0, d1, h)

Cubic Hermite interpolation using values and first derivatives at endpoints.

# Arguments
- `t`: Local coordinate in [0, 1]
- `f0, f1`: Function values at left and right endpoints
- `d0, d1`: First derivatives at left and right endpoints
- `h`: Spacing between endpoints (for scaling derivatives)

# Returns
Interpolated value with O(h⁴) error for C⁴ functions.
"""
@inline function cubic_hermite(t::T, f0::T, f1::T, d0::T, d1::T, h::T) where T
    t2 = t * t
    t3 = t2 * t
    
    # Basis for values
    h00 = one(T) + t2*(T(-3) + T(2)*t)    # 1 - 3t² + 2t³
    h01 = t2*(T(3) - T(2)*t)               # 3t² - 2t³
    
    # Basis for first derivatives (scaled by h)
    h10 = t + t2*(T(-2) + t)               # t - 2t² + t³
    h11 = t2*(t - one(T))                  # -t² + t³
    
    return h00*f0 + h01*f1 + h*(h10*d0 + h11*d1)
end