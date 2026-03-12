"""
    (itp::ConvolutionInterpolation{T,2,TCoefs,IT,Axs,KA,Val{2},Val{DG},EQ})(x::Vararg{T,2}) where {T,TCoefs,IT,Axs,KA,DG,EQ}

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

@inline function (itp::ConvolutionInterpolation{T,2,TCoefs,IT,Axs,KA,Val{2},
                    DG,EQ,KBC,IntegralOrder,FD,SD,SG})(x::Vararg{Number,2}) where 
                    {T,TCoefs,IT,Axs,KA,DG,EQ,KBC,FD,SD,SG}

    result = zero(T)
    n1 = size(itp.coefs, 1)
    n2 = size(itp.coefs, 2)
    @inbounds for i1 in 1:n1
        xi = itp.knots[1][_eqs_d(itp.eqs,1)] + (i1 - _eqs_d(itp.eqs,1)) * itp.h[1]
        dx = _kernel_d(itp.kernel,1)((x[1]          - xi) / itp.h[1]) -
             _kernel_d(itp.kernel,1)((itp.anchor[1] - xi) / itp.h[1])
        for i2 in 1:n2
            yj = itp.knots[2][_eqs_d(itp.eqs,2)] + (i2 - _eqs_d(itp.eqs,2)) * itp.h[2]
            dy = _kernel_d(itp.kernel,2)((x[2]          - yj) / itp.h[2]) -
                 _kernel_d(itp.kernel,2)((itp.anchor[2] - yj) / itp.h[2])
            result += itp.coefs[i1, i2] * dx * dy
        end
    end
    return result * itp.h[1] * itp.h[2]
end

function (itp::ConvolutionInterpolation{T,2,TCoefs,IT,Axs,KA,Val{2},
                    DG,EQ,KBC,DerivativeOrder{DO},FD,SD,SG})(x::Vararg{Number,2}) where 
                    {T,TCoefs,IT,Axs,KA,DG,EQ<:Int,KBC,DO,FD,SD,SG}

    # Check if nonuniform lazy
    is_nonuniform = itp.lazy && any(d -> length(itp.knots[d]) != size(itp.coefs, d), 1:2)

    if is_nonuniform
        ngs = ntuple(d -> (length(itp.knots[d]) - size(itp.coefs, d)) ÷ 2, 2)
        n1 = size(itp.coefs, 1)
        n2 = size(itp.coefs, 2)

        if DG === Val{:a0}
            ids = ntuple(2) do d
                i = clamp(searchsortedlast(itp.knots[d], x[d]), ngs[d] + 1, length(itp.knots[d]) - ngs[d] - 1)
                xi = (x[d] - itp.knots[d][i]) <= (itp.knots[d][i+1] - itp.knots[d][i]) / 2 ? i : i + 1
                xi - ngs[d]
            end
            return @inbounds itp.coefs[ids...]

        elseif DG === Val{:a1}
            iw = ntuple(2) do d
                i = clamp(searchsortedlast(itp.knots[d], x[d]), ngs[d] + 1, length(itp.knots[d]) - ngs[d] - 1)
                t = (T(x[d]) - itp.knots[d][i]) / (itp.knots[d][i+1] - itp.knots[d][i])
                (i - ngs[d], t)
            end
            i1, t1 = iw[1]; i2, t2 = iw[2]
            @inbounds return (one(T)-t1)*(one(T)-t2)*itp.coefs[i1,i2] +
                             t1*(one(T)-t2)*itp.coefs[i1+1,i2] +
                             (one(T)-t1)*t2*itp.coefs[i1,i2+1] +
                             t1*t2*itp.coefs[i1+1,i2+1]

        else
            k1 = itp.knots[1][ngs[1]+1:end-ngs[1]]
            k2 = itp.knots[2][ngs[2]+1:end-ngs[2]]
            i, wi = _nonuniform_dim_ghost(itp.knots[1], x[1])
            j, wj = _nonuniform_dim_ghost(itp.knots[2], x[2])
            res1 = ntuple(s -> _resolve_dim_nonuniform(k1, i + s - 2, n1, T), 4)
            res2 = ntuple(s -> _resolve_dim_nonuniform(k2, j + s - 2, n2, T), 4)
            result = zero(T)
            @inbounds for s1 in 1:4
                for (cw1, ri1) in res1[s1]
                    w1 = wi[s1] * cw1
                    for s2 in 1:4
                        for (cw2, ri2) in res2[s2]
                            result += w1 * wj[s2] * cw2 * itp.coefs[ri1, ri2]
                        end
                    end
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

    j_float = (x[2] - itp.knots[2][1]) / itp.h[2] + one(T)
    j = if itp.lazy
        clamp(floor(Int, j_float), 1, length(itp.knots[2]) - 1)
    else
        clamp(floor(Int, j_float), itp.eqs, length(itp.knots[2]) - itp.eqs)
    end

    result = zero(T)
    if itp.lazy && DG !== Val{:a0} && DG !== Val{:a1} &&
                   (is_boundary_stencil(i, size(itp.coefs, 1), itp.eqs) ||
                    is_boundary_stencil(j, size(itp.coefs, 2), itp.eqs))
        ng = itp.eqs - 1
        ghost_matrix = get_polynomial_ghost_coeffs(_kernel_sym(itp.deg))
        factor_nd = -one(T)
        sw = 2 * itp.eqs
        n1 = size(itp.coefs, 1)
        n2 = size(itp.coefs, 2)
        res1 = ntuple(s -> _resolve_dim_uniform(n1, ng, i + ng + s - itp.eqs, ghost_matrix, factor_nd, 2), sw)
        res2 = ntuple(s -> _resolve_dim_uniform(n2, ng, j + ng + s - itp.eqs, ghost_matrix, factor_nd, 2), sw)
        s1_val = (x[1] - itp.knots[1][i]) / itp.h[1]
        s2_val = (x[2] - itp.knots[2][j]) / itp.h[2]
        @inbounds for (li, l) in enumerate(-(itp.eqs-1):itp.eqs)
            kx = itp.kernel(s1_val - T(l))
            for (cw1, ri1) in res1[li]
                w1 = kx * cw1
                for (mi, m) in enumerate(-(itp.eqs-1):itp.eqs)
                    ky = itp.kernel(s2_val - T(m))
                    for (cw2, ri2) in res2[mi]
                        result += w1 * ky * cw2 * itp.coefs[ri1, ri2]
                    end
                end
            end
        end
    else
        @inbounds for l = -(itp.eqs-1):itp.eqs,
                      m = -(itp.eqs-1):itp.eqs
            result += itp.coefs[i+l, j+m] *
                      itp.kernel((x[1] - itp.knots[1][i+l]) / itp.h[1]) *
                      itp.kernel((x[2] - itp.knots[2][j+m]) / itp.h[2])
        end
    end
    scale = DO >= 0 ? (one(T)/itp.h[1])^DO * (one(T)/itp.h[2])^DO :
                      itp.h[1]^(-DO) * itp.h[2]^(-DO)
    return @fastmath result * scale
end

@inline function (itp::ConvolutionInterpolation{T,2,TCoefs,IT,Axs,KA,Val{2},
                    DG,EQ,KBC,DerivativeOrder{DO},FD,SD,SG,NB})(x::Vararg{Number,2}) where 
                    {T,TCoefs,IT,Axs,KA<:Tuple{<:ConvolutionKernel,<:ConvolutionKernel},
                    DG,EQ<:Tuple{Int,Int},KBC,DO,FD,SD,SG,NB<:Nothing}

    i_float = (x[1] - itp.knots[1][1]) / itp.h[1] + one(T)
    i = clamp(floor(Int, i_float), itp.eqs[1], length(itp.knots[1]) - itp.eqs[1])

    j_float = (x[2] - itp.knots[2][1]) / itp.h[2] + one(T)
    j = clamp(floor(Int, j_float), itp.eqs[2], length(itp.knots[2]) - itp.eqs[2])

    result = zero(T)
    @inbounds for l = -(itp.eqs[1]-1):itp.eqs[1]
        kx = itp.kernel[1]((x[1] - itp.knots[1][i+l]) / itp.h[1])
        for m = -(itp.eqs[2]-1):itp.eqs[2]
            result += itp.coefs[i+l, j+m] * kx * itp.kernel[2]((x[2] - itp.knots[2][j+m]) / itp.h[2])
        end
    end
    scale = prod(d -> DO[d] >= 0 ? one(T) / itp.h[d]^DO[d] : itp.h[d]^(-DO[d]), 1:2)
    return @fastmath result * scale
end

@inline function (itp::ConvolutionInterpolation{T,2,TCoefs,IT,Axs,KA,Val{2},
                    DG,EQ,KBC,MixedIntegralOrder{DO},FD,SD,SG,NB})(x::Vararg{Number,2}) where
                    {T,TCoefs,IT,Axs,KA<:Tuple{<:ConvolutionKernel,<:ConvolutionKernel},
                    DG,EQ<:Tuple{Int,Int},KBC,DO,FD,SD,SG,NB<:Nothing}
 
    n1 = size(itp.coefs, 1)
    n2 = size(itp.coefs, 2)
 
    # For derivative dims, find the local stencil centre
    if DO[1] != -1
        i_float = (x[1] - itp.knots[1][1]) / itp.h[1] + one(T)
        i = clamp(floor(Int, i_float), itp.eqs[1], length(itp.knots[1]) - itp.eqs[1])
    end
    if DO[2] != -1
        j_float = (x[2] - itp.knots[2][1]) / itp.h[2] + one(T)
        j = clamp(floor(Int, j_float), itp.eqs[2], length(itp.knots[2]) - itp.eqs[2])
    end
 
    result = zero(T)
 
    # Outer loop: all coefs if integrating dim 1, stencil if differentiating
    i1_range = DO[1] == -1 ? (1:n1) : ((i - (itp.eqs[1]-1)):(i + itp.eqs[1]))
    i2_range = DO[2] == -1 ? (1:n2) : ((j - (itp.eqs[2]-1)):(j + itp.eqs[2]))
 
    @inbounds for i1 in i1_range
        if DO[1] == -1
            xi = itp.knots[1][itp.eqs[1]] + (i1 - itp.eqs[1]) * itp.h[1]
            kx = itp.kernel[1]((x[1]          - xi) / itp.h[1]) -
                 itp.kernel[1]((itp.anchor[1] - xi) / itp.h[1])
        else
            kx = itp.kernel[1]((x[1] - itp.knots[1][i1]) / itp.h[1])
        end
        for i2 in i2_range
            if DO[2] == -1
                yj = itp.knots[2][itp.eqs[2]] + (i2 - itp.eqs[2]) * itp.h[2]
                ky = itp.kernel[2]((x[2]          - yj) / itp.h[2]) -
                     itp.kernel[2]((itp.anchor[2] - yj) / itp.h[2])
            else
                ky = itp.kernel[2]((x[2] - itp.knots[2][i2]) / itp.h[2])
            end
            result += itp.coefs[i1, i2] * kx * ky
        end
    end
 
    scale = one(T)
    scale *= DO[1] == -1 ? itp.h[1] : one(T) / itp.h[1]^DO[1]
    scale *= DO[2] == -1 ? itp.h[2] : one(T) / itp.h[2]^DO[2]
    return @fastmath result * scale
end