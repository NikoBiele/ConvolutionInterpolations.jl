"""
    (itp::FastConvolutionInterpolation{T,2,TCoefs,IT,Axs,KA,Val{2},Val{DG},EQ,PR,KP})(x::Vararg{AbstractFloat,2}) where {T,TCoefs,IT,Axs,KA,DG,EQ,PR,KP}

Evaluate a two-dimensional fast convolution interpolation at the given point.

# Arguments
- `itp`: The two-dimensional fast interpolation object
- `x`: The coordinates at which to evaluate the interpolation (x₁, x₂)

# Returns
- The interpolated value at the specified coordinates

# Details
This method implements a highly optimized two-dimensional convolution interpolation
using precomputed kernel values for speed. Rather than computing kernel values on-the-fly,
it looks up the closest precomputed value for each offset in each dimension.

The algorithm:
1. For each dimension:
   a. Find the nearest knot point less than or equal to the coordinate
   b. Compute the normalized distance between the coordinate and this knot point
   c. Find the closest precomputed kernel value for this distance
2. Apply a weighted sum using these precomputed values across both dimensions

The precomputed kernel values significantly reduce computational overhead, providing
much faster interpolation with minimal loss of accuracy compared to the standard method.
"""
function (itp::FastConvolutionInterpolation{T,2,TCoefs,IT,Axs,KA,Val{2},Val{DG},EQ,PR,KP})(x::Vararg{AbstractFloat,2}) where {T,TCoefs,IT,Axs,KA,DG,EQ,PR,KP}

    # first dimension
    i = limit_convolution_bounds(1, searchsortedlast(itp.knots[1], x[1]), itp)
    x_diff_left = (x[1] - itp.knots[1][i])/itp.h[1] # Normalized distance to left sample
    x_diff_right = 1 - x_diff_left # Normalized distance to right sample
    index_nearest_x = searchsortednearest(itp.pre_range, x_diff_right) # lookup index in precomputed matrix
    # second dimension
    j = limit_convolution_bounds(2, searchsortedlast(itp.knots[2], x[2]), itp)
    y_diff_left = (x[2] - itp.knots[2][j])/itp.h[2] # Normalized distance to left sample
    y_diff_right = 1 - y_diff_left # Normalized distance to right sample
    index_nearest_y = searchsortednearest(itp.pre_range, y_diff_right) # lookup index in precomputed matrix

    # result
    result = sum(sum(itp.coefs[i+l, j+m]*itp.kernel_pre[index_nearest_x, l+itp.eqs]*itp.kernel_pre[index_nearest_y, m+itp.eqs] for l in -(itp.eqs-1):itp.eqs) for m in -(itp.eqs-1):itp.eqs)

    return result
end