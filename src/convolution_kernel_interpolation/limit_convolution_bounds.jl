"""
    limit_convolution_bounds(dim::Int, index::Int, itp::ConvolutionInterpolation{T,N,TCoefs,IT,Axs,ConvolutionKernel{DG},Val{N},Val{DG},EQ}) where {T,N,TCoefs,IT,Axs,DG,EQ}

Ensure interpolation indices stay within valid bounds for Val{N}-typed dimensions.

# Arguments
- `dim::Int`: The dimension being processed
- `index::Int`: The candidate index in that dimension
- `itp`: The interpolation object

# Returns
- An adjusted index value that ensures the kernel support stays within the coefficient array bounds

# Details
This function ensures that the index used for interpolation is at least `eqs` positions away
from the edges of the coefficient array, which is necessary to ensure that all kernel evaluations
can access valid coefficient values. This implementation is specialized for dimensions represented
as value types (Val{N}).

If the supplied index is too close to the beginning or end of the array, it is adjusted to ensure
that the kernel's support region is fully contained within the valid part of the coefficient array.
"""
function limit_convolution_bounds(dim::Int, index::Int, itp::ConvolutionInterpolation{T,N,TCoefs,IT,Axs,ConvolutionKernel{DG},Val{N},Val{DG},EQ}) where {T,N,TCoefs,IT,Axs,DG,EQ}
    if index < itp.eqs
        index = itp.eqs
    elseif index > length(itp.knots[dim]) - itp.eqs
        index = length(itp.knots[dim]) - itp.eqs
    end
    return index
end

"""
    limit_convolution_bounds(dim::Int, index::Int, itp::ConvolutionInterpolation{T,N,TCoefs,IT,Axs,ConvolutionKernel{DG},HigherDimension{N},Val{DG},EQ}) where {T,N,TCoefs,IT,Axs,DG,EQ}

Ensure interpolation indices stay within valid bounds for HigherDimension{N}-typed dimensions.

# Arguments
- `dim::Int`: The dimension being processed
- `index::Int`: The candidate index in that dimension
- `itp`: The interpolation object

# Returns
- An adjusted index value that ensures the kernel support stays within the coefficient array bounds

# Details
This function ensures that the index used for interpolation is at least `eqs` positions away
from the edges of the coefficient array, which is necessary to ensure that all kernel evaluations
can access valid coefficient values. This implementation is specialized for dimensions represented
as HigherDimension{N} types (for N > 3).

If the supplied index is too close to the beginning or end of the array, it is adjusted to ensure
that the kernel's support region is fully contained within the valid part of the coefficient array.
This function is functionally identical to the Val{N} version but uses a different type dispatch
for higher dimensions.
"""
function limit_convolution_bounds(dim::Int, index::Int, itp::ConvolutionInterpolation{T,N,TCoefs,IT,Axs,ConvolutionKernel{DG},HigherDimension{N},Val{DG},EQ}) where {T,N,TCoefs,IT,Axs,DG,EQ}
    if index < itp.eqs
        index = itp.eqs
    elseif index > length(itp.knots[dim]) - itp.eqs
        index = length(itp.knots[dim]) - itp.eqs
    end
    return index
end