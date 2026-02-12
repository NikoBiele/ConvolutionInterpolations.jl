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
function (itp::ConvolutionInterpolation{T,N,TCoefs,IT,Axs,KA,HigherDimension{N},DG,EQ,KBC,DO})(x::Vararg{Number,N}) where {T,N,TCoefs,IT,KA,Axs,DG,EQ,KBC,DO}

    # Compute i_float once per dimension
    i_floats = ntuple(d -> (x[d] - itp.knots[d][1]) / itp.h[d] + one(T), N)
    # Find knot indices for each dimension
    pos_ids = ntuple(d -> clamp(floor(Int, i_floats[d]), itp.eqs, length(itp.knots[d]) - itp.eqs), N)

    result = zero(T)
    @inbounds for offsets in Iterators.product(ntuple(_ -> -(itp.eqs-1):itp.eqs, N)...)
        result += itp.coefs[(pos_ids .+ offsets)...] * 
                  prod(itp.kernel((x[d] - itp.knots[d][pos_ids[d] + offsets[d]]) / itp.h[d]) for d in 1:N)
    end
    
    return @inbounds @fastmath result * prod((one(T)/itp.h[i])^(DO.parameters[1]) for i in 1:N)
end