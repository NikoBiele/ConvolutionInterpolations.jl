"""
    (itp::FastConvolutionInterpolation{T,N,TCoefs,IT,Axs,KA,HigherDimension{N},Val{DG},EQ,PR,KP})(x::Vararg{AbstractFloat,N}) where {T,N,TCoefs,IT,KA,Axs,DG,EQ,PR,KP}

Evaluate a higher-dimensional (N > 3) fast convolution interpolation at the given point.

# Arguments
- `itp`: The N-dimensional fast interpolation object
- `x`: The coordinates at which to evaluate the interpolation (x₁, x₂, ..., xₙ)

# Returns
- The interpolated value at the specified coordinates

# Details
This method implements a generalized approach for fast convolution interpolation in any number
of dimensions. It is designed for higher-dimensional cases (N > 3) where specialized implementations
would be impractical, while still maintaining high performance through precomputed kernel values.

The algorithm:
1. For each dimension:
   a. Find the nearest knot point less than or equal to the coordinate
   b. Compute the normalized distance between the coordinate and this knot point
   c. Find the closest precomputed kernel value for this distance
2. Iterate through all possible offset combinations in the N-dimensional neighborhood
3. Compute a weighted sum using precomputed kernel values

This generalized approach scales efficiently to any number of dimensions, with the precomputed
kernel values significantly reducing computational overhead compared to the standard method.
This optimization becomes increasingly important as dimensionality increases, since the number
of kernel evaluations grows exponentially with the number of dimensions.
"""
function (itp::FastConvolutionInterpolation{T,N,TCoefs,IT,Axs,KA,HigherDimension{N},Val{DG},EQ,PR,KP})(x::Vararg{AbstractFloat,N}) where {T,N,TCoefs,IT,KA,Axs,DG,EQ,PR,KP}

    pos_ids = ntuple(d -> limit_convolution_bounds(d, ntuple(d -> searchsortedlast(itp.knots[d], x[d]), N)[d], itp), N)
    diff_left = ntuple(d -> (x[d] - itp.knots[d][pos_ids[d]])/itp.h[d], N) # Normalized distances to left sample
    diff_right = ntuple(d -> 1 - diff_left[d], N) # Normalized distance to right sample
    index_nearest = ntuple(d -> searchsortednearest(itp.pre_range, diff_right[d]), N) # lookup index in precomputed matrix

    result = zero(T)
    for offsets in Iterators.product(ntuple(_ -> -(itp.eqs-1):itp.eqs, N)...)
        result += itp.coefs[(pos_ids .+ offsets)...] * prod(itp.kernel_pre[index_nearest[d], offsets[d]+itp.eqs] for d in 1:N)
    end
    
    return result
end