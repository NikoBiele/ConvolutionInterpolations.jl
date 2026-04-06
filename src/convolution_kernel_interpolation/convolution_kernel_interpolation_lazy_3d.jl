"""
    (itp::ConvolutionInterpolation{T,3,TCoefs,Axs,KA,Val{3},Val{DG},EQ})(x::Vararg{T,3}) where {T,TCoefs,Axs,KA,DG,EQ}

Evaluate a three-dimensional convolution interpolation at the given point.

# Arguments
- `itp`: The three-dimensional interpolation object
- `x`: The coordinates at which to evaluate the interpolation (x₁, x₂, x₃)

# Returns
- The interpolated value at the specified coordinates

# Details
This method implements specialized, efficient convolution interpolation for the three-dimensional case.
It finds the nearest knot points in each dimension, then computes a weighted sum of neighboring values
using the convolution kernel. The number of neighboring points used depends on the equation order (eqs).

The interpolation uses the formula:
```
result = ∑∑∑ coefs[i+l, j+m, k+n] * kernel((x₁ - knots₁[i+l])/h₁) * 
                                      kernel((x₂ - knots₂[j+m])/h₂) * 
                                      kernel((x₃ - knots₃[k+n])/h₃)
```

for `l, m, n = -(eqs-1):eqs`, where `i`, `j`, and `k` are the indices of the nearest knot points
less than or equal to `x₁`, `x₂`, and `x₃` respectively.
"""

# ═══════════════════════════════════════════════════════════════
# 3D eval with resolve pattern
# ═══════════════════════════════════════════════════════════════

function (itp::ConvolutionInterpolation{T,3,0,TCoefs,Axs,KA,Val{3},
                    DG,EQ,KBC,DerivativeOrder{DO},FD,SD,Val{:not_used},NB,Val{true},Val{0}})(x::Vararg{Number,3}) where 
                    {T<:AbstractFloat,TCoefs<:AbstractArray{T,3},
                    Axs<:Tuple{<:AbstractVector,<:AbstractVector,<:AbstractVector},
                    KA<:Tuple{<:ConvolutionKernel,<:ConvolutionKernel,<:ConvolutionKernel},
                    DG,EQ<:Tuple{Int,Int,Int},KBC<:Tuple{<:Tuple{Symbol,Symbol},<:Tuple{Symbol,Symbol},<:Tuple{Symbol,Symbol}},
                    DO,FD,SD,NB<:Nothing}

    # === Uniform path ===

    i_float = (x[1] - itp.knots[1][1]) / itp.h[1] + one(T)
    i = clamp(floor(Int, i_float), 1, length(itp.knots[1]) - 1)
    
    j_float = (x[2] - itp.knots[2][1]) / itp.h[2] + one(T)
    j = clamp(floor(Int, j_float), 1, length(itp.knots[2]) - 1)
    
    k_float = (x[3] - itp.knots[3][1]) / itp.h[3] + one(T)
    k = clamp(floor(Int, k_float), 1, length(itp.knots[3]) - 1)
    
    result = zero(T)
    if LZ === Val{true} && DG !== Val{:a0} && DG !== Val{:a1} &&
                   (is_boundary_stencil(i, size(itp.coefs, 1), itp.eqs[1]) ||
                    is_boundary_stencil(j, size(itp.coefs, 2), itp.eqs[2]) ||
                    is_boundary_stencil(k, size(itp.coefs, 3), itp.eqs[3]))
        ng = itp.eqs[1] - 1 # lazy uses same kernel in all directions
        ghost_matrix = get_polynomial_ghost_coeffs(:not_used, _kernel_sym(itp.kernel_sym)[1])
        factor_nd = -one(T)
        sw = 2 * itp.eqs[1]
        n1 = size(itp.coefs, 1)
        n2 = size(itp.coefs, 2)
        n3 = size(itp.coefs, 3)
        res1 = ntuple(s -> _resolve_dim_uniform(n1, ng, i + ng + s - itp.eqs[1], ghost_matrix, factor_nd, 3), sw)
        res2 = ntuple(s -> _resolve_dim_uniform(n2, ng, j + ng + s - itp.eqs[2], ghost_matrix, factor_nd, 3), sw)
        res3 = ntuple(s -> _resolve_dim_uniform(n3, ng, k + ng + s - itp.eqs[3], ghost_matrix, factor_nd, 3), sw)
        s1_val = (x[1] - itp.knots[1][i]) / itp.h[1]
        s2_val = (x[2] - itp.knots[2][j]) / itp.h[2]
        s3_val = (x[3] - itp.knots[3][k]) / itp.h[3]
        @inbounds for (li, l) in enumerate(-(itp.eqs[1]-1):itp.eqs[1])
            kx = itp.kernel(s1_val - T(l))
            for (cw1, ri1) in res1[li]
                w1 = kx * cw1
                for (mi, m) in enumerate(-(itp.eqs[2]-1):itp.eqs[2])
                    ky = itp.kernel(s2_val - T(m))
                    for (cw2, ri2) in res2[mi]
                        w12 = w1 * ky * cw2
                        for (ni, n) in enumerate(-(itp.eqs[3]-1):itp.eqs[3])
                            kz = itp.kernel[1](s3_val - T(n)) # lazy has same kernels in all dimensions
                            for (cw3, ri3) in res3[ni]
                                result += w12 * kz * cw3 * itp.coefs[ri1, ri2, ri3]
                            end
                        end
                    end
                end
            end
        end
    else
        @inbounds for l = -(itp.eqs[1]-1):itp.eqs[1],
                      m = -(itp.eqs[2]-1):itp.eqs[2],
                      n = -(itp.eqs[3]-1):itp.eqs[3]
            result += itp.coefs[i+l, j+m, k+n] *
                      itp.kernel[1]((x[1] - itp.knots[1][i+l]) / itp.h[1]) *
                      itp.kernel[2]((x[2] - itp.knots[2][j+m]) / itp.h[2]) *
                      itp.kernel[3]((x[3] - itp.knots[3][k+n]) / itp.h[3])
        end
    end
    scale = prod(d -> DO[d] >= 0 ? one(T) / itp.h[d]^DO[d] : itp.h[d]^(-DO[d]), 1:3)
    return @fastmath result * scale
end