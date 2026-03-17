"""
    (itp::ConvolutionInterpolation{T,3,TCoefs,IT,Axs,KA,Val{3},Val{DG},EQ})(x::Vararg{T,3}) where {T,TCoefs,IT,Axs,KA,DG,EQ}

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

@inline function (itp::ConvolutionInterpolation{T,3,TCoefs,IT,Axs,KA,Val{3},
                    DG,EQ,KBC,IntegralOrder,FD,SD,SG})(x::Vararg{Number,3}) where 
                    {T,TCoefs,IT,Axs,KA,DG,EQ,KBC,FD,SD,SG}

    result = zero(T)
    n1 = size(itp.coefs, 1)
    n2 = size(itp.coefs, 2)
    n3 = size(itp.coefs, 3)
    @inbounds for i1 in 1:n1
        xi = itp.knots[1][_eqs_d(itp.eqs,1)] + (i1 - _eqs_d(itp.eqs,1)) * itp.h[1]
        dx = _kernel_d(itp.kernel,1)((x[1]          - xi) / itp.h[1]) -
             _kernel_d(itp.kernel,1)((itp.anchor[1] - xi) / itp.h[1])
        for i2 in 1:n2
            yj = itp.knots[2][_eqs_d(itp.eqs,2)] + (i2 - _eqs_d(itp.eqs,2)) * itp.h[2]
            dy = _kernel_d(itp.kernel,2)((x[2]          - yj) / itp.h[2]) -
                 _kernel_d(itp.kernel,2)((itp.anchor[2] - yj) / itp.h[2])
            for i3 in 1:n3
                zk = itp.knots[3][_eqs_d(itp.eqs,3)] + (i3 - _eqs_d(itp.eqs,3)) * itp.h[3]
                dz = _kernel_d(itp.kernel,3)((x[3]          - zk) / itp.h[3]) -
                     _kernel_d(itp.kernel,3)((itp.anchor[3] - zk) / itp.h[3])
                result += itp.coefs[i1, i2, i3] * dx * dy * dz
            end
        end
    end
    return result * itp.h[1] * itp.h[2] * itp.h[3]
end

function (itp::ConvolutionInterpolation{T,3,TCoefs,IT,Axs,KA,Val{3},
                    DG,EQ,KBC,DerivativeOrder{DO},FD,SD,SG})(x::Vararg{Number,3}) where 
                    {T,TCoefs,IT,Axs,KA,DG,EQ<:Int,KBC,DO,FD,SD,SG}

    # Check if nonuniform lazy
    is_nonuniform = itp.lazy && any(d -> length(itp.knots[d]) != size(itp.coefs, d), 1:3)

    if is_nonuniform
        ngs = ntuple(d -> (length(itp.knots[d]) - size(itp.coefs, d)) ÷ 2, 3)
        n1 = size(itp.coefs, 1)
        n2 = size(itp.coefs, 2)
        n3 = size(itp.coefs, 3)

        if DG === Val{:a0}
            ids = ntuple(3) do d
                i = clamp(searchsortedlast(itp.knots[d], x[d]), ngs[d] + 1, length(itp.knots[d]) - ngs[d] - 1)
                xi = (x[d] - itp.knots[d][i]) <= (itp.knots[d][i+1] - itp.knots[d][i]) / 2 ? i : i + 1
                xi - ngs[d]
            end
            return @inbounds itp.coefs[ids...]

        elseif DG === Val{:a1}
            iw = ntuple(3) do d
                i = clamp(searchsortedlast(itp.knots[d], x[d]), ngs[d] + 1, length(itp.knots[d]) - ngs[d] - 1)
                t = (T(x[d]) - itp.knots[d][i]) / (itp.knots[d][i+1] - itp.knots[d][i])
                (i - ngs[d], t)
            end
            result = zero(T)
            @inbounds for c3 in 0:1, c2 in 0:1, c1 in 0:1
                w = (c1 == 0 ? one(T) - iw[1][2] : iw[1][2]) *
                    (c2 == 0 ? one(T) - iw[2][2] : iw[2][2]) *
                    (c3 == 0 ? one(T) - iw[3][2] : iw[3][2])
                result += w * itp.coefs[iw[1][1]+c1, iw[2][1]+c2, iw[3][1]+c3]
            end
            return result

        else
            k1 = itp.knots[1][ngs[1]+1:end-ngs[1]]
            k2 = itp.knots[2][ngs[2]+1:end-ngs[2]]
            k3 = itp.knots[3][ngs[3]+1:end-ngs[3]]
            ii, wi = _nonuniform_dim_ghost(itp.knots[1], x[1])
            jj, wj = _nonuniform_dim_ghost(itp.knots[2], x[2])
            kk, wk = _nonuniform_dim_ghost(itp.knots[3], x[3])
            res1 = ntuple(s -> _resolve_dim_nonuniform(k1, ii + s - 2, n1, T), 4)
            res2 = ntuple(s -> _resolve_dim_nonuniform(k2, jj + s - 2, n2, T), 4)
            res3 = ntuple(s -> _resolve_dim_nonuniform(k3, kk + s - 2, n3, T), 4)
            result = zero(T)
            @inbounds for s1 in 1:4
                for (cw1, ri1) in res1[s1]
                    w1 = wi[s1] * cw1
                    for s2 in 1:4
                        for (cw2, ri2) in res2[s2]
                            w12 = w1 * wj[s2] * cw2
                            for s3 in 1:4
                                for (cw3, ri3) in res3[s3]
                                    result += w12 * wk[s3] * cw3 * itp.coefs[ri1, ri2, ri3]
                                end
                            end
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

    k_float = (x[3] - itp.knots[3][1]) / itp.h[3] + one(T)
    k = if itp.lazy
        clamp(floor(Int, k_float), 1, length(itp.knots[3]) - 1)
    else
        clamp(floor(Int, k_float), itp.eqs, length(itp.knots[3]) - itp.eqs)
    end

    result = zero(T)
    if itp.lazy && DG !== Val{:a0} && DG !== Val{:a1} &&
                   (is_boundary_stencil(i, size(itp.coefs, 1), itp.eqs) ||
                    is_boundary_stencil(j, size(itp.coefs, 2), itp.eqs) ||
                    is_boundary_stencil(k, size(itp.coefs, 3), itp.eqs))
        ng = itp.eqs - 1
        ghost_matrix = get_polynomial_ghost_coeffs(_kernel_sym(itp.kernel_sym))
        factor_nd = -one(T)
        sw = 2 * itp.eqs
        n1 = size(itp.coefs, 1)
        n2 = size(itp.coefs, 2)
        n3 = size(itp.coefs, 3)
        res1 = ntuple(s -> _resolve_dim_uniform(n1, ng, i + ng + s - itp.eqs, ghost_matrix, factor_nd, 3), sw)
        res2 = ntuple(s -> _resolve_dim_uniform(n2, ng, j + ng + s - itp.eqs, ghost_matrix, factor_nd, 3), sw)
        res3 = ntuple(s -> _resolve_dim_uniform(n3, ng, k + ng + s - itp.eqs, ghost_matrix, factor_nd, 3), sw)
        s1_val = (x[1] - itp.knots[1][i]) / itp.h[1]
        s2_val = (x[2] - itp.knots[2][j]) / itp.h[2]
        s3_val = (x[3] - itp.knots[3][k]) / itp.h[3]
        @inbounds for (li, l) in enumerate(-(itp.eqs-1):itp.eqs)
            kx = itp.kernel(s1_val - T(l))
            for (cw1, ri1) in res1[li]
                w1 = kx * cw1
                for (mi, m) in enumerate(-(itp.eqs-1):itp.eqs)
                    ky = itp.kernel(s2_val - T(m))
                    for (cw2, ri2) in res2[mi]
                        w12 = w1 * ky * cw2
                        for (ni, n) in enumerate(-(itp.eqs-1):itp.eqs)
                            kz = itp.kernel(s3_val - T(n))
                            for (cw3, ri3) in res3[ni]
                                result += w12 * kz * cw3 * itp.coefs[ri1, ri2, ri3]
                            end
                        end
                    end
                end
            end
        end
    else
        @inbounds for l = -(itp.eqs-1):itp.eqs,
                      m = -(itp.eqs-1):itp.eqs,
                      n = -(itp.eqs-1):itp.eqs
            result += itp.coefs[i+l, j+m, k+n] *
                      itp.kernel((x[1] - itp.knots[1][i+l]) / itp.h[1]) *
                      itp.kernel((x[2] - itp.knots[2][j+m]) / itp.h[2]) *
                      itp.kernel((x[3] - itp.knots[3][k+n]) / itp.h[3])
        end
    end
    scale = DO >= 0 ? (one(T)/itp.h[1])^DO * (one(T)/itp.h[2])^DO * (one(T)/itp.h[3])^DO :
                      itp.h[1]^(-DO) * itp.h[2]^(-DO) * itp.h[3]^(-DO)
    return @fastmath result * scale
end

@inline function (itp::ConvolutionInterpolation{T,3,TCoefs,IT,Axs,KA,Val{3},
                    DG,EQ,KBC,DerivativeOrder{DO},FD,SD,SG})(x::Vararg{Number,3}) where 
                    {T,TCoefs,IT,Axs,KA,DG,EQ<:Tuple{Int,Int,Int},KBC,DO,FD,SD,SG}

    i_float = (x[1] - itp.knots[1][1]) / itp.h[1] + one(T)
    i = clamp(floor(Int, i_float), itp.eqs[1], length(itp.knots[1]) - itp.eqs[1])

    j_float = (x[2] - itp.knots[2][1]) / itp.h[2] + one(T)
    j = clamp(floor(Int, j_float), itp.eqs[2], length(itp.knots[2]) - itp.eqs[2])

    k_float = (x[3] - itp.knots[3][1]) / itp.h[3] + one(T)
    k = clamp(floor(Int, k_float), itp.eqs[3], length(itp.knots[3]) - itp.eqs[3])

    result = zero(T)
    @inbounds for l = -(itp.eqs[1]-1):itp.eqs[1],
                  m = -(itp.eqs[2]-1):itp.eqs[2],
                  n = -(itp.eqs[3]-1):itp.eqs[3]
        result += itp.coefs[i+l, j+m, k+n] *
                  itp.kernel[1]((x[1] - itp.knots[1][i+l]) / itp.h[1]) *
                  itp.kernel[2]((x[2] - itp.knots[2][j+m]) / itp.h[2]) *
                  itp.kernel[3]((x[3] - itp.knots[3][k+n]) / itp.h[3])
    end
    scale = prod(d -> DO[d] >= 0 ? one(T) / itp.h[d]^DO[d] : itp.h[d]^(-DO[d]), 1:3)
    return @fastmath result * scale
end

@inline function (itp::ConvolutionInterpolation{T,3,TCoefs,IT,Axs,KA,Val{3},
                    DG,EQ,KBC,MixedIntegralOrder{DO},FD,SD,SG})(x::Vararg{Number,3}) where
                    {T,TCoefs,IT,Axs,
                    KA<:Tuple{<:ConvolutionKernel,<:ConvolutionKernel,<:ConvolutionKernel},
                    DG,EQ<:Tuple{Int,Int,Int},KBC,DO,FD,SD,SG}
 
    ns = ntuple(d -> size(itp.coefs, d), 3)
 
    ids = ntuple(3) do d
        if DO[d] != -1
            clamp(floor(Int, (x[d] - itp.knots[d][1]) / itp.h[d] + one(T)),
                  itp.eqs[d], length(itp.knots[d]) - itp.eqs[d])
        else
            0  # unused for integral dims
        end
    end
 
    ranges = ntuple(3) do d
        if DO[d] == -1
            1:ns[d]
        else
            (ids[d] - (itp.eqs[d]-1)):(ids[d] + itp.eqs[d])
        end
    end
 
    result = zero(T)
    @inbounds for i1 in ranges[1]
        if DO[1] == -1
            xi = itp.knots[1][itp.eqs[1]] + (i1 - itp.eqs[1]) * itp.h[1]
            kx = itp.kernel[1]((x[1] - xi) / itp.h[1]) -
                 itp.kernel[1]((itp.anchor[1] - xi) / itp.h[1])
        else
            kx = itp.kernel[1]((x[1] - itp.knots[1][i1]) / itp.h[1])
        end
        for i2 in ranges[2]
            if DO[2] == -1
                yj = itp.knots[2][itp.eqs[2]] + (i2 - itp.eqs[2]) * itp.h[2]
                ky = itp.kernel[2]((x[2] - yj) / itp.h[2]) -
                     itp.kernel[2]((itp.anchor[2] - yj) / itp.h[2])
            else
                ky = itp.kernel[2]((x[2] - itp.knots[2][i2]) / itp.h[2])
            end
            for i3 in ranges[3]
                if DO[3] == -1
                    zk = itp.knots[3][itp.eqs[3]] + (i3 - itp.eqs[3]) * itp.h[3]
                    kz = itp.kernel[3]((x[3] - zk) / itp.h[3]) -
                         itp.kernel[3]((itp.anchor[3] - zk) / itp.h[3])
                else
                    kz = itp.kernel[3]((x[3] - itp.knots[3][i3]) / itp.h[3])
                end
                result += itp.coefs[i1, i2, i3] * kx * ky * kz
            end
        end
    end
 
    scale = one(T)
    for d in 1:3
        scale *= DO[d] == -1 ? itp.h[d] : one(T) / itp.h[d]^DO[d]
    end
    return @fastmath result * scale
end