"""
    (itp::ConvolutionInterpolation{T,1,TCoefs,IT,Axs,KA,Val{1},Val{DG},EQ})(x::Vararg{T,1}) where {T,TCoefs,IT,Axs,KA,DG,EQ}

Evaluate a one-dimensional convolution interpolation at the given point.

# Arguments
- `itp`: The one-dimensional interpolation object
- `x`: The coordinate at which to evaluate the interpolation

# Returns
- The interpolated value at the specified coordinate

# Details
This method implements specialized, efficient convolution interpolation for the one-dimensional case.
It finds the nearest knot point, then computes a weighted sum of neighboring values using the
convolution kernel. The number of neighboring points used depends on the equation order (eqs).

The interpolation uses the formula:
```
result = ∑ coefs[i+l] * kernel((x - knots[i+l])/h) for l = -(eqs-1):eqs
```

where `i` is the index of the nearest knot point less than or equal to `x`.
"""

# ── Uniform resolve ───────────────────────────────────────────

"""
    _resolve_dim_uniform(n_d, ng, vid, ghost_matrix, factor_nd, N, T)

Resolve virtual index `vid` in one dimension for uniform lazy grids.

Virtual indexing: [1..ng] = left ghost, [ng+1..ng+n_d] = interior,
[ng+n_d+1..end] = right ghost.

Returns a Vector{Tuple{T,Int}} of (weight, raw_index) pairs.
Interior returns 1 entry. Ghost returns n_d + n_stencil entries
from the polynomial ghost matrix with mean-centering:

    ghost = y_mean + factor * Σ coef[m] * (stencil[m] - y_mean)
          = (1 - factor*Σcoef)/n_d * Σ interior[k]  +  factor * Σ coef[m] * stencil[m]
"""
@inline function _resolve_dim_uniform(n_d::Int, ng::Int, vid::Int,
                                       ghost_matrix::Matrix{Float64},
                                       factor_nd::T, N::Int) where T
    if vid > ng && vid <= ng + n_d
        return [(one(T), vid - ng)]
    end

    ghost_side = vid <= ng ? :left : :right
    ghost_layer = ghost_side === :left ? (ng + 1 - vid) : (vid - ng - n_d)
    factor = ghost_side === :right ? factor_nd : one(T)
    n_stencil = size(ghost_matrix, 2)
    ns = min(n_stencil, n_d)

    coef_sum = zero(T)
    for m in 1:ns
        coef_sum += T(ghost_matrix[ghost_layer, m])
    end
    mean_weight = (one(T) - factor * coef_sum) / n_d

    result = Vector{Tuple{T,Int}}(undef, n_d + ns)

    # Mean contribution to all interior points
    for k in 1:n_d
        result[k] = (mean_weight, k)
    end

    # Stencil contributions
    for m in 1:ns
        ri = if ghost_side === :left
            m
        elseif N == 1
            n_d + 1 - m   # 1D right: reversed (matches eager)
        else
            m              # ND right: natural (matches eager double-reversal)
        end
        result[n_d + m] = (factor * T(ghost_matrix[ghost_layer, m]), ri)
    end

    return result
end

# ═══════════════════════════════════════════════════════════════
# 1D eval with resolve pattern for both uniform and nonuniform
# ═══════════════════════════════════════════════════════════════

@inline function (itp::ConvolutionInterpolation{T,1,TCoefs,IT,Axs,KA,Val{1},
                    DG,EQ,KBC,DO,FD,SD,SG})(x::Vararg{Number,1}) where 
                    {T,TCoefs,IT,Axs,KA,DG,EQ,KBC,DO,FD,SD,SG}

    # Check if nonuniform lazy
    n_k = length(itp.knots[1])
    n_d = size(itp.coefs, 1)
    is_nonuniform = itp.lazy && (n_k != n_d)

    if is_nonuniform
        ng = (n_k - n_d) ÷ 2

        if DG === Val{:a0}
            i = clamp(searchsortedlast(itp.knots[1], x[1]), ng + 1, n_k - ng - 1)
            xi = (x[1] - itp.knots[1][i]) <= (itp.knots[1][i+1] - itp.knots[1][i]) / 2 ? i : i + 1
            return itp.coefs[xi - ng]

        elseif DG === Val{:a1}
            i = clamp(searchsortedlast(itp.knots[1], x[1]), ng + 1, n_k - ng - 1)
            t = (T(x[1]) - itp.knots[1][i]) / (itp.knots[1][i+1] - itp.knots[1][i])
            return (one(T) - t) * itp.coefs[i - ng] + t * itp.coefs[i + 1 - ng]

        else
            # n3 nonuniform lazy — resolve_dim pattern
            knots_orig_d = itp.knots[1][ng+1:end-ng]
            i_nu, w = _nonuniform_dim_ghost(itp.knots[1], x[1])

            res = ntuple(s -> _resolve_dim_nonuniform(knots_orig_d, i_nu + s - 2, n_d, T), 4)

            result = zero(T)
            @inbounds for s in 1:4
                for (cw, ri) in res[s]
                    result += w[s] * cw * itp.coefs[ri]
                end
            end
            return result
        end
    end

    # === Uniform path ===

    i_float = (x[1] - itp.knots[1][1]) / itp.h[1] + one(T)
    i = if itp.lazy
        clamp(floor(Int, i_float), 1, length(itp.knots[1]) - 1)
    else
        clamp(floor(Int, i_float), itp.eqs, length(itp.knots[1]) - itp.eqs)
    end

    result = zero(T)
    if itp.lazy && DG !== Val{:a0} && DG !== Val{:a1} && is_boundary_stencil(i, n_d, itp.eqs)
        # Uniform lazy boundary — resolve_dim pattern
        ng = itp.eqs - 1
        ghost_matrix = get_polynomial_ghost_coeffs(_kernel_sym(itp.deg))
        factor_nd = one(T)  # 1D
        stencil_width = 2 * itp.eqs
        res = ntuple(s -> _resolve_dim_uniform(n_d, ng, i + ng + s - itp.eqs, 
                          ghost_matrix, factor_nd, 1), stencil_width)

        s_val = (x[1] - itp.knots[1][i]) / itp.h[1]
        @inbounds for (li, l) in enumerate(-(itp.eqs-1):itp.eqs)
            kval = itp.kernel(s_val - T(l))
            for (cw, ri) in res[li]
                result += kval * cw * itp.coefs[ri]
            end
        end
    else
        @inbounds for l = -(itp.eqs-1):itp.eqs
            result += itp.coefs[i+l] * 
                    itp.kernel( (x[1] - itp.knots[1][i+l]) / itp.h[1] )
        end
    end
    return @fastmath result * one(T)/(itp.h[1])^DO.parameters[1]
end