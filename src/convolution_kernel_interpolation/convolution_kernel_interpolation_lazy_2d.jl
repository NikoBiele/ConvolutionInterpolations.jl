"""
    (itp::ConvolutionInterpolation{T,2,TCoefs,Axs,KA,Val{2},Val{DG},EQ})(x::Vararg{T,2}) where {T,TCoefs,Axs,KA,DG,EQ}

Evaluate a two-dimensional convolution interpolation at the given point.

# Arguments
- `itp`: The two-dimensional interpolation object
- `x`: The coordinates at which to evaluate the interpolation (x₁, x₂)

# Returns
- The interpolated value at the specified coordinates

# Details
This method implements specialized, efficient convolution interpolation for the two-dimensional case.
It finds the nearest knot points in each dimension, then computes a weighted sum of neighboring values
using the convolution kernel. The number of neighboring points used depends on the equation order (eqs).

The interpolation uses the formula:
```
result = ∑∑ coefs[i+l, j+m] * kernel((x₁ - knots₁[i+l])/h₁) * kernel((x₂ - knots₂[j+m])/h₂)
```

for `l, m = -(eqs-1):eqs`, where `i` and `j` are the indices of the nearest knot points less than
or equal to `x₁` and `x₂` respectively.
"""

# ═══════════════════════════════════════════════════════════════
# 2D eval with resolve pattern
# ═══════════════════════════════════════════════════════════════

@inline _eqs_d(eqs::Int, d) = eqs
@inline _eqs_d(eqs::Tuple, d) = eqs[d]

@inline _kernel_d(kernel, d) = kernel
@inline _kernel_d(kernel::Tuple, d) = kernel[d]

function (itp::ConvolutionInterpolation{T,2,0,TCoefs,Axs,KA,Val{2},
                    DG,EQ,KBC,DerivativeOrder{DO},FD,SD,Val{:not_used},NB,Val{true},Val{0}})(x::Vararg{Number,2}) where 
                    {T<:AbstractFloat,TCoefs<:AbstractArray{T,2},Axs<:Tuple{<:AbstractVector,<:AbstractVector},
                    KA<:Tuple{<:ConvolutionKernel,<:ConvolutionKernel},
                    DG,EQ<:Tuple{Int,Int},KBC<:Tuple{<:Tuple{Symbol,Symbol},<:Tuple{Symbol,Symbol}},
                    DO,FD,SD,NB<:Nothing}

    # === Uniform path ===

    i_float = (x[1] - itp.knots[1][1]) / itp.h[1] + one(T)
    i = clamp(floor(Int, i_float), 1, length(itp.knots[1]) - 1)
    
    j_float = (x[2] - itp.knots[2][1]) / itp.h[2] + one(T)
    j = clamp(floor(Int, j_float), 1, length(itp.knots[2]) - 1)
    
    result = zero(T)
    if LZ === Val{true} && DG !== Val{:a0} && DG !== Val{:a1} &&
                   (is_boundary_stencil(i, size(itp.coefs, 1), itp.eqs[1]) ||
                    is_boundary_stencil(j, size(itp.coefs, 2), itp.eqs[2]))
        ng = itp.eqs[1] - 1 # lazy uses same kernel in all directions
        ghost_matrix = get_polynomial_ghost_coeffs(_kernel_sym(itp.kernel_sym)[1])
        factor_nd = -one(T)
        sw = 2 * itp.eqs[1]
        n1 = size(itp.coefs, 1)
        n2 = size(itp.coefs, 2)
        res1 = ntuple(s -> _resolve_dim_uniform(n1, ng, i + ng + s - itp.eqs[1], ghost_matrix, factor_nd, 2), sw)
        res2 = ntuple(s -> _resolve_dim_uniform(n2, ng, j + ng + s - itp.eqs[2], ghost_matrix, factor_nd, 2), sw)
        s1_val = (x[1] - itp.knots[1][i]) / itp.h[1]
        s2_val = (x[2] - itp.knots[2][j]) / itp.h[2]
        @inbounds for (li, l) in enumerate(-(itp.eqs[1]-1):itp.eqs[1])
            kx = itp.kernel(s1_val - T(l))
            for (cw1, ri1) in res1[li]
                w1 = kx * cw1
                for (mi, m) in enumerate(-(itp.eqs[2]-1):itp.eqs[2])
                    ky = itp.kernel(s2_val - T(m))
                    for (cw2, ri2) in res2[mi]
                        result += w1 * ky * cw2 * itp.coefs[ri1, ri2]
                    end
                end
            end
        end
    else
        @inbounds for l = -(itp.eqs[1]-1):itp.eqs[1],
                      m = -(itp.eqs[2]-1):itp.eqs[2]
            result += itp.coefs[i+l, j+m] *
                      itp.kernel[1]((x[1] - itp.knots[1][i+l]) / itp.h[1]) *
                      itp.kernel[1]((x[2] - itp.knots[2][j+m]) / itp.h[2])
        end
    end
    scale = prod(d -> DO[d] >= 0 ? one(T) / itp.h[d]^DO[d] : itp.h[d]^(-DO[d]), 1:2)
    return @fastmath result * scale
end