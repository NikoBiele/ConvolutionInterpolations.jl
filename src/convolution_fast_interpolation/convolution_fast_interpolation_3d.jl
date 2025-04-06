"""
    (itp::FastConvolutionInterpolation{T,3,TCoefs,IT,Axs,KA,Val{3},Val{DG},EQ,PR,KP})(x::Vararg{AbstractFloat,3}) where {T,TCoefs,IT,Axs,KA,DG,EQ,PR,KP}

Evaluate a three-dimensional fast convolution interpolation at the given point.

# Arguments
- `itp`: The three-dimensional fast interpolation object
- `x`: The coordinates at which to evaluate the interpolation (x₁, x₂, x₃)

# Returns
- The interpolated value at the specified coordinates

# Details
This method implements a highly optimized three-dimensional convolution interpolation
using precomputed kernel values for speed. Rather than computing kernel values on-the-fly,
it looks up the closest precomputed value for each offset in each dimension.

The algorithm:
1. For each dimension:
   a. Find the nearest knot point less than or equal to the coordinate
   b. Compute the normalized distance between the coordinate and this knot point
   c. Find the closest precomputed kernel value for this distance
2. Apply a weighted sum using these precomputed values across all three dimensions

Despite the complexity of three-dimensional interpolation, the precomputed kernel values 
make this method much faster than the standard approach, with minimal loss of accuracy.
This optimization is particularly valuable in 3D where the number of kernel evaluations
grows cubically with the support size.
"""
function (itp::FastConvolutionInterpolation{T,3,TCoefs,IT,Axs,KA,Val{3},Val{DG},EQ,PR,KP})(x::Vararg{AbstractFloat,3}) where {T,TCoefs,IT,Axs,KA,DG,EQ,PR,KP}

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
    # third dimension
    k = limit_convolution_bounds(3, searchsortedlast(itp.knots[3], x[3]), itp)
    z_diff_left = (x[3] - itp.knots[3][k])/itp.h[3] # Normalized distance to left sample
    z_diff_right = 1 - z_diff_left # Normalized distance to right sample
    index_nearest_z = searchsortednearest(itp.pre_range, z_diff_right) # lookup index in precomputed matrix

    # result
    result = sum(sum(sum(itp.coefs[i+l, j+m, k+n]*itp.kernel_pre[index_nearest_x, l+itp.eqs]*
                                                itp.kernel_pre[index_nearest_y, m+itp.eqs]*
                                                itp.kernel_pre[index_nearest_z, n+itp.eqs]
                                                for l in -(itp.eqs-1):itp.eqs) for m in -(itp.eqs-1):itp.eqs) for n in -(itp.eqs-1):itp.eqs )

    return result
end