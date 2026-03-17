"""
    (itp::ConvolutionInterpolation{T,N,TCoefs,IT,Axs,KA,HigherDimension{N},Val{DG},EQ})(x::Vararg{T,N}) where {T,N,TCoefs,IT,KA,Axs,DG,EQ}

Evaluate a higher-dimensional (N > 3) convolution interpolation at the given point.

# Arguments
- `itp`: The N-dimensional interpolation object
- `x`: The coordinates at which to evaluate the interpolation (x₁, x₂, ..., xₙ)

# Returns
- The interpolated value at the specified coordinates

# Details
This method implements a generalized approach for convolution interpolation in any number of dimensions.
It is optimized for higher-dimensional cases (N > 3) where specialized implementations would be impractical.

The interpolation finds the nearest knot points in each dimension, then computes a weighted sum of
neighboring values using the convolution kernel. The number of neighboring points used depends on
the equation order (eqs).

The interpolation uses a product of kernel evaluations across all dimensions:
```
result = ∑ coefs[pos_ids + offsets] * ∏ kernel((x[d] - knots[d][pos_ids[d] + offsets[d]])/h[d])
```

where the sum is over all possible offset combinations in the N-dimensional neighborhood, and the product
is across all dimensions. This generalizes to any number of dimensions efficiently.
"""

@inline function (itp::ConvolutionInterpolation{T,N,TCoefs,IT,Axs,KA,HigherDimension{N},
                    DG,EQ,KBC,IntegralOrder,FD,SD,SG})(x::Vararg{Number,N}) where 
                    {T,N,TCoefs,IT,KA,Axs,DG,EQ,KBC,FD,SD,SG}

    result = zero(T)
    ns = ntuple(d -> size(itp.coefs, d), N)
    @inbounds for idx in Iterators.product(ntuple(d -> 1:ns[d], N)...)
        dx = prod(d -> begin
            eqs_d = _eqs_d(itp.eqs, d)
            xd = itp.knots[d][eqs_d] + (idx[d] - eqs_d) * itp.h[d]
            _kernel_d(itp.kernel, d)((x[d]          - xd) / itp.h[d]) -
            _kernel_d(itp.kernel, d)((itp.anchor[d] - xd) / itp.h[d])
        end, 1:N)
        result += itp.coefs[idx...] * dx
    end
    return result * prod(itp.h[d] for d in 1:N)
end

function (itp::ConvolutionInterpolation{T,N,TCoefs,IT,Axs,KA,HigherDimension{N},
                    DG,EQ,KBC,DerivativeOrder{DO},FD,SD,SG})(x::Vararg{Number,N}) where 
                    {T,N,TCoefs,IT,KA,Axs,DG,EQ<:Int,KBC,DO,FD,SD,SG}

    # Check if nonuniform lazy
    is_nonuniform = itp.lazy && any(d -> length(itp.knots[d]) != size(itp.coefs, d), 1:N)

    if is_nonuniform
        ngs = ntuple(d -> (length(itp.knots[d]) - size(itp.coefs, d)) ÷ 2, N)

        if DG === Val{:a0}
            ids = ntuple(N) do d
                i = clamp(searchsortedlast(itp.knots[d], x[d]), ngs[d] + 1, length(itp.knots[d]) - ngs[d] - 1)
                xi = (x[d] - itp.knots[d][i]) <= (itp.knots[d][i+1] - itp.knots[d][i]) / 2 ? i : i + 1
                xi - ngs[d]
            end
            return @inbounds itp.coefs[ids...]

        elseif DG === Val{:a1}
            iw = ntuple(N) do d
                i = clamp(searchsortedlast(itp.knots[d], x[d]), ngs[d] + 1, length(itp.knots[d]) - ngs[d] - 1)
                t = (T(x[d]) - itp.knots[d][i]) / (itp.knots[d][i+1] - itp.knots[d][i])
                (i - ngs[d], t)
            end
            result = zero(T)
            @inbounds for offsets in Iterators.product(ntuple(_ -> 0:1, N)...)
                w = prod(d -> offsets[d] == 0 ? one(T) - iw[d][2] : iw[d][2], 1:N)
                idx = ntuple(d -> iw[d][1] + offsets[d], N)
                result += w * itp.coefs[idx...]
            end
            return result

        else
            knots_orig = ntuple(d -> itp.knots[d][ngs[d]+1:end-ngs[d]], N)
            iw = ntuple(d -> _nonuniform_dim_ghost(itp.knots[d], x[d]), N)
            indices = ntuple(d -> iw[d][1], N)
            weights = ntuple(d -> iw[d][2], N)
            result = zero(T)
            @inbounds for offsets in Iterators.product(ntuple(_ -> 1:4, N)...)
                w_prod = prod(weights[d][offsets[d]] for d in 1:N)
                idx = ntuple(d -> indices[d] + offsets[d] - 2, N)
                c = _nonuniform_lazy_coef(itp.coefs, knots_orig, idx)
                result += w_prod * c
            end
            return result
        end
    end

    # === Uniform path ===

    i_floats = ntuple(d -> (x[d] - itp.knots[d][1]) / itp.h[d] + one(T), N)
    pos_ids = if itp.lazy
        ntuple(d -> clamp(floor(Int, i_floats[d]), 1, length(itp.knots[d]) - 1), N)
    else
        ntuple(d -> clamp(floor(Int, i_floats[d]), itp.eqs, length(itp.knots[d]) - itp.eqs), N)
    end

    result = zero(T)
    if itp.lazy && DG !== Val{:a0} && DG !== Val{:a1} &&
                   any(d -> is_boundary_stencil(pos_ids[d], size(itp.coefs, d), itp.eqs), 1:N)
        kernel_type = _kernel_sym(itp.kernel_sym)
        ng = itp.eqs - 1
        s = ntuple(d -> (x[d] - itp.knots[d][pos_ids[d]]) / itp.h[d], N)
        @inbounds for offsets in Iterators.product(ntuple(_ -> -(itp.eqs-1):itp.eqs, N)...)
            vidx = ntuple(d -> pos_ids[d] + ng + offsets[d], N)
            coef = lazy_ghost_value(itp.coefs, vidx, itp.eqs, kernel_type)
            result += coef * prod(itp.kernel(s[d] - T(offsets[d])) for d in 1:N)
        end
    else
        @inbounds for offsets in Iterators.product(ntuple(_ -> -(itp.eqs-1):itp.eqs, N)...)
            result += itp.coefs[(pos_ids .+ offsets)...] *
                      prod(itp.kernel((x[d] - itp.knots[d][pos_ids[d] + offsets[d]]) / itp.h[d]) for d in 1:N)
        end
    end
    scale = DO >= 0 ? prod((one(T)/itp.h[d])^DO for d in 1:N) :
                      prod(itp.h[d]^(-DO) for d in 1:N)
    return @inbounds @fastmath result * scale
end

@inline function (itp::ConvolutionInterpolation{T,N,TCoefs,IT,Axs,KA,HigherDimension{N},
                    DG,EQ,KBC,DerivativeOrder{DO},FD,SD,SG})(x::Vararg{Number,N}) where 
                    {T,N,TCoefs,IT,KA,Axs,DG,EQ<:Tuple,KBC,DO,FD,SD,SG}

    i_floats = ntuple(d -> (x[d] - itp.knots[d][1]) / itp.h[d] + one(T), N)
    pos_ids = ntuple(d -> clamp(floor(Int, i_floats[d]), itp.eqs[d], length(itp.knots[d]) - itp.eqs[d]), N)

    result = zero(T)
    @inbounds for offsets in Iterators.product(ntuple(d -> -(itp.eqs[d]-1):itp.eqs[d], N)...)
        result += itp.coefs[(pos_ids .+ offsets)...] *
                  prod(itp.kernel[d]((x[d] - itp.knots[d][pos_ids[d] + offsets[d]]) / itp.h[d]) for d in 1:N)
    end
    scale = prod(d -> DO[d] >= 0 ? one(T) / itp.h[d]^DO[d] : itp.h[d]^(-DO[d]), 1:N)
    return @inbounds @fastmath result * scale
end

function (itp::ConvolutionInterpolation{T,N,TCoefs,IT,Axs,KA,HigherDimension{N},
                    DG,EQ,KBC,MixedIntegralOrder{DO},FD,SD,SG})(x::Vararg{Number,N}) where
                    {T,N,TCoefs,IT,Axs,KA<:Tuple,DG,EQ<:Tuple,KBC,DO,FD,SD,SG}
 
    ns = ntuple(d -> size(itp.coefs, d), N)
 
    ids = ntuple(N) do d
        DO[d] != -1 ? clamp(floor(Int, (x[d] - itp.knots[d][1]) / itp.h[d] + one(T)),
                            itp.eqs[d], length(itp.knots[d]) - itp.eqs[d]) : 0
    end
 
    ranges = ntuple(N) do d
        DO[d] == -1 ? (1:ns[d]) : ((ids[d] - (itp.eqs[d]-1)):(ids[d] + itp.eqs[d]))
    end
 
    result = zero(T)
    @inbounds for idx in Iterators.product(ranges...)
        kval = one(T)
        for d in 1:N
            i_d = idx[d]
            if DO[d] == -1
                xj = itp.knots[d][itp.eqs[d]] + (i_d - itp.eqs[d]) * itp.h[d]
                kval *= itp.kernel[d]((x[d]          - xj) / itp.h[d]) -
                        itp.kernel[d]((itp.anchor[d] - xj) / itp.h[d])
            else
                kval *= itp.kernel[d]((x[d] - itp.knots[d][i_d]) / itp.h[d])
            end
        end
        result += itp.coefs[idx...] * kval
    end
 
    scale = one(T)
    for d in 1:N
        scale *= DO[d] == -1 ? itp.h[d] : one(T) / itp.h[d]^DO[d]
    end
    return @fastmath result * scale
end