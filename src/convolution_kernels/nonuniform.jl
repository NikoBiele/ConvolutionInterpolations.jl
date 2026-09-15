# ============================================================
# Grid uniformity check
# ============================================================

"""
    is_uniform_grid(knots::AbstractVector; rtol=1e-10)

Check whether a 1D knot vector is uniformly spaced.

Returns `true` if all spacings are within relative tolerance `rtol`
of the first spacing.
"""
function is_uniform_grid(knots::AbstractVector{T}) where T
    tol = T <: Float32 ? 1e-5 : 1e-10
    knots isa AbstractRange && return true  # ranges are uniform by construction
    h0 = knots[2] - knots[1]
    isapprox(h0, 0, atol=tol) && return false
    for i in 2:length(knots)-1
        hi = knots[i+1] - knots[i]
        if abs(hi - h0) > tol * abs(h0)
            return false
        end
    end
    return true
end

"""
    n3_to_a3_on_uniform(kernels, knots)

On a fully uniform grid the nonuniform cubic `:n3` reduces exactly to the
uniform cubic `:a3`, which has a faster evaluator and a fast-path
implementation. Remap `:n3` → `:a3` in every dimension when all grids are
uniform; leave the kernels untouched otherwise (the nonuniform path handles
mixed uniform/nonuniform dimensions).
"""
function n3_to_a3_on_uniform(kernels::NTuple{N,Symbol}, knots::NTuple{N,AbstractVector}) where N
    any(==(:n3), kernels) || return kernels
    all(d -> is_uniform_grid(knots[d]), 1:N) || return kernels
    return ntuple(d -> kernels[d] == :n3 ? :a3 : kernels[d], N)
end

# ============================================================
# Nonuniform cubic weights (Keys-like, Catmull-Rom equivalent)
# ============================================================

"""
    nonuniform_cubic_weights(x_local, h_m1, h_0, h_1)

Compute 4 cubic interpolation weights for nonuniform grid spacing.

# Arguments
- `x_local`: position within interval [0, h_0] (distance from left node)
- `h_m1`: spacing to the left of the interval (x_0 - x_{-1})
- `h_0`: width of the current interval (x_1 - x_0)
- `h_1`: spacing to the right of the interval (x_2 - x_1)

# Returns
Tuple `(w_m1, w_0, w_1, w_2)` of weights for the 4-point stencil
{x_{-1}, x_0, x_1, x_2}.

# Properties
- Cardinal: w_j(0) = δ_{j,0} and w_j(h_0) = δ_{j,1}
- Partition of unity: Σ w_j = 1
- Exact for polynomials up to degree 2 (O(h³) convergence)
- Recovers Keys' kernel (a=-1/2) when h_m1 = h_0 = h_1
"""
@inline function nonuniform_cubic_weights(s::T, hm::T, h0::T, hp::T) where T
    # Precompute common terms
    s_minus_h0 = s - h0           # (x - h_0), negative in [0, h_0)
    s2 = s * s
    
    # w_{-1}: outermost left weight
    w_m1 = -s * s_minus_h0 * s_minus_h0 / (h0 * hm * (h0 + hm))
    
    # w_2: outermost right weight  
    w_2 = s2 * s_minus_h0 / (h0 * hp * (h0 + hp))
    
    # w_0: left interior weight
    numer_0 = -h0*h0*hm - h0*h0*s - h0*hp*hm - h0*hp*s + h0*s2 + hp*s2 + hm*s2
    w_0 = s_minus_h0 * numer_0 / (h0*h0 * hm * (h0 + hp))
    
    # w_1: right interior weight
    numer_1 = -h0*h0*s - h0*hp*hm - 2*h0*hp*s - h0*hm*s + h0*s2 + hp*s2 + hm*s2
    w_1 = -s * numer_1 / (h0*h0 * hp * (h0 + hm))
    
    return (w_m1, w_0, w_1, w_2)
end