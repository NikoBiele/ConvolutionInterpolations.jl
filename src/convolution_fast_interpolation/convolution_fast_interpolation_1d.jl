"""
    (itp::FastConvolutionInterpolation{T,1,TCoefs,IT,Axs,KA,Val{1},Val{DG},EQ,PR,KP})(x::Vararg{AbstractFloat,1}) where {T,TCoefs,IT,Axs,KA,DG,EQ,PR,KP}

Evaluate a one-dimensional fast convolution interpolation at the given point.

# Arguments
- `itp`: The one-dimensional fast interpolation object
- `x`: The coordinate at which to evaluate the interpolation

# Returns
- The interpolated value at the specified coordinate

# Details
This method implements a highly optimized one-dimensional convolution interpolation
using precomputed kernel values for speed. Rather than computing kernel values on-the-fly,
it looks up the closest precomputed value for each offset.

The algorithm:
1. Finds the nearest knot point less than or equal to `x`
2. Computes the normalized distance between `x` and this knot point
3. Finds the closest precomputed kernel value for this distance
4. Applies a weighted sum using these precomputed values

This approach provides a significant speedup over the standard method by trading
memory for computational efficiency, with minimal loss of accuracy.
"""
function (itp::FastConvolutionInterpolation{T,1,TCoefs,IT,Axs,KA,Val{1},Val{DG},EQ,PR,KP})(x::Vararg{AbstractFloat,1}) where {T,TCoefs,IT,Axs,KA,DG,EQ,PR,KP}

    i = limit_convolution_bounds(1, searchsortedlast(itp.knots[1], x[1]), itp)
    x_diff_left = (x[1] - itp.knots[1][i])/itp.h[1] # Normalized distance to left sample
    x_diff_right = 1 - x_diff_left # Normalized distance to right sample
    index_nearest = searchsortednearest(itp.pre_range, x_diff_right) # lookup index in precomputed matrix
    result = sum(itp.coefs[i+j]*itp.kernel_pre[index_nearest, j+itp.eqs] for j in -(itp.eqs-1):itp.eqs)
    return result
end