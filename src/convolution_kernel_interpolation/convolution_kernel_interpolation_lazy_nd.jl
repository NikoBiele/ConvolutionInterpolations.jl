"""
    (itp::ConvolutionInterpolation{T,N,TCoefs,Axs,KA,HigherDimension{N},Val{DG},EQ})(x::Vararg{T,N}) where {T,N,TCoefs,KA,Axs,DG,EQ}

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

function (itp::ConvolutionInterpolation{T,N,0,TCoefs,Axs,KA,HigherDimension{N},
                    DG,EQ,KBC,DerivativeOrder{DO},FD,SD,Val{:not_used},NB,Val{true},Val{0}})(x::Vararg{Number,N}) where 
                    {T<:AbstractFloat,N,TCoefs<:AbstractArray{T,N},
                    KA<:Tuple{Vararg{ConvolutionKernel}},Axs<:Tuple{Vararg{AbstractVector}},
                    DG,EQ<:Tuple{Vararg{Int}},KBC<:Tuple{Vararg{Tuple{Symbol,Symbol}}},DO,FD,SD,NB<:Nothing}

    # === Uniform path ===

    i_floats = ntuple(d -> (x[d] - itp.knots[d][1]) / itp.h[d] + one(T), N)
    pos_ids = ntuple(d -> clamp(floor(Int, i_floats[d]), 1, length(itp.knots[d]) - 1), N)
    
    result = zero(T)
    if itp.lazy && DG !== Val{:a0} && DG !== Val{:a1} &&
                   any(d -> is_boundary_stencil(pos_ids[d], size(itp.coefs, d), itp.eqs[d]), 1:N)
        kernel_type = _kernel_sym(itp.kernel_sym)[1]
        ng = itp.eqs[1] - 1
        s = ntuple(d -> (x[d] - itp.knots[d][pos_ids[d]]) / itp.h[d], N)
        @inbounds for offsets in Iterators.product(ntuple(dim -> -(itp.eqs[dim]-1):itp.eqs[dim], N)...)
            vidx = ntuple(d -> pos_ids[d] + ng + offsets[d], N)
            coef = lazy_ghost_value(itp.coefs, vidx, itp.eqs, kernel_type)
            result += coef * prod(itp.kernel(s[d] - T(offsets[d])) for d in 1:N)
        end
    else
        @inbounds for offsets in Iterators.product(ntuple(dim -> -(itp.eqs[dim]-1):itp.eqs[dim], N)...)
            result += itp.coefs[(pos_ids .+ offsets)...] *
                      prod(itp.kernel((x[d] - itp.knots[d][pos_ids[d] + offsets[d]]) / itp.h[d]) for d in 1:N)
        end
    end
    scale = prod(d -> DO[d] >= 0 ? one(T) / itp.h[d]^DO[d] : itp.h[d]^(-DO[d]), 1:N)
    return @inbounds @fastmath result * scale
end