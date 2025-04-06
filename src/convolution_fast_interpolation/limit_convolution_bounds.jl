"""
    limit_convolution_bounds(dim::Int, index::Int, itp::AbstractConvolutionInterpolation{T,N,TCoefs,IT,Axs,ConvolutionKernel{DG},Val{N},Val{DG},EQ}) where {T,N,TCoefs,IT,Axs,DG,EQ}

Ensure interpolation indices stay within valid bounds for the abstract interpolation type with Val{N} dimensions.

# Arguments
- `dim::Int`: The dimension being processed
- `index::Int`: The candidate index in that dimension
- `itp`: The interpolation object (can be either standard or fast)

# Returns
- An adjusted index value that ensures the kernel support stays within the coefficient array bounds

# Details
This function ensures that the index used for interpolation is at least `eqs` positions away
from the edges of the coefficient array, which is necessary to ensure that all kernel evaluations
can access valid coefficient values. This implementation works for the abstract interpolation
type, allowing it to be used with both standard and fast interpolation objects when dimensions
are represented as value types (Val{N}).

If the supplied index is too close to the beginning or end of the array, it is adjusted to ensure
that the kernel's support region is fully contained within the valid part of the coefficient array.
"""
function limit_convolution_bounds(dim::Int, index::Int, itp::AbstractConvolutionInterpolation{T,N,TCoefs,IT,Axs,ConvolutionKernel{DG},Val{N},Val{DG},EQ}) where {T,N,TCoefs,IT,Axs,DG,EQ}
    if index < itp.eqs
        index = itp.eqs
    elseif index > length(itp.knots[dim]) - itp.eqs
        index = length(itp.knots[dim]) - itp.eqs
    end
    return index
end

"""
    limit_convolution_bounds(dim::Int, index::Int, itp::AbstractConvolutionInterpolation{T,N,TCoefs,IT,Axs,ConvolutionKernel{DG},HigherDimension{N},Val{DG},EQ}) where {T,N,TCoefs,IT,Axs,DG,EQ}

Ensure interpolation indices stay within valid bounds for the abstract interpolation type with HigherDimension{N}.

# Arguments
- `dim::Int`: The dimension being processed
- `index::Int`: The candidate index in that dimension
- `itp`: The interpolation object (can be either standard or fast)

# Returns
- An adjusted index value that ensures the kernel support stays within the coefficient array bounds

# Details
This function ensures that the index used for interpolation is at least `eqs` positions away
from the edges of the coefficient array, which is necessary to ensure that all kernel evaluations
can access valid coefficient values. This implementation works for the abstract interpolation
type, allowing it to be used with both standard and fast interpolation objects when dimensions
are represented as HigherDimension{N} types (for N > 3).

If the supplied index is too close to the beginning or end of the array, it is adjusted to ensure
that the kernel's support region is fully contained within the valid part of the coefficient array.
This function is functionally identical to the Val{N} version but uses a different type dispatch
for higher dimensions.
"""
function limit_convolution_bounds(dim::Int, index::Int, itp::AbstractConvolutionInterpolation{T,N,TCoefs,IT,Axs,ConvolutionKernel{DG},HigherDimension{N},Val{DG},EQ}) where {T,N,TCoefs,IT,Axs,DG,EQ}
    if index < itp.eqs
        index = itp.eqs
    elseif index > length(itp.knots[dim]) - itp.eqs
        index = length(itp.knots[dim]) - itp.eqs
    end
    return index
end

"""
    limit_convolution_bounds(dim::Int, index::Int, itp::AbstractConvolutionInterpolation{T,N,TCoefs,IT,Axs,GaussianConvolutionKernel{B},Val{N},Val{DG},EQ}) where {T,N,TCoefs,IT,Axs,DG,EQ,B}

Ensure interpolation indices stay within valid bounds for the Gaussian kernel with Val{N} dimensions.

# Arguments
- `dim::Int`: The dimension being processed
- `index::Int`: The candidate index in that dimension
- `itp`: The interpolation object with a Gaussian kernel

# Returns
- An adjusted index value that ensures the kernel support stays within the coefficient array bounds

# Details
This function ensures that the index used for interpolation is at least `eqs` positions away
from the edges of the coefficient array, which is necessary to ensure that all kernel evaluations
can access valid coefficient values. This implementation is specialized for interpolation objects
using a Gaussian convolution kernel with dimensions represented as value types (Val{N}).

The Gaussian kernel typically requires a larger support region (higher `eqs` value) than polynomial
kernels due to its infinite support, which is why a specialized implementation is provided.

If the supplied index is too close to the beginning or end of the array, it is adjusted to ensure
that the kernel's support region is fully contained within the valid part of the coefficient array.
"""
function limit_convolution_bounds(dim::Int, index::Int, itp::AbstractConvolutionInterpolation{T,N,TCoefs,IT,Axs,GaussianConvolutionKernel{B},Val{N},Val{DG},EQ}) where {T,N,TCoefs,IT,Axs,DG,EQ,B}
    if index < itp.eqs
        index = itp.eqs
    elseif index > length(itp.knots[dim]) - itp.eqs
        index = length(itp.knots[dim]) - itp.eqs
    end
    return index
end

"""
    limit_convolution_bounds(dim::Int, index::Int, itp::AbstractConvolutionInterpolation{T,N,TCoefs,IT,Axs,GaussianConvolutionKernel{B},HigherDimension{N},Val{DG},EQ}) where {T,N,TCoefs,IT,Axs,DG,EQ,B}

Ensure interpolation indices stay within valid bounds for the Gaussian kernel with HigherDimension{N}.

# Arguments
- `dim::Int`: The dimension being processed
- `index::Int`: The candidate index in that dimension
- `itp`: The interpolation object with a Gaussian kernel

# Returns
- An adjusted index value that ensures the kernel support stays within the coefficient array bounds

# Details
This function ensures that the index used for interpolation is at least `eqs` positions away
from the edges of the coefficient array, which is necessary to ensure that all kernel evaluations
can access valid coefficient values. This implementation is specialized for interpolation objects
using a Gaussian convolution kernel with dimensions represented as HigherDimension{N} types (for N > 3).

The Gaussian kernel typically requires a larger support region (higher `eqs` value) than polynomial
kernels due to its infinite support, which is why a specialized implementation is provided.

If the supplied index is too close to the beginning or end of the array, it is adjusted to ensure
that the kernel's support region is fully contained within the valid part of the coefficient array.
This function is functionally identical to the Val{N} version but uses a different type dispatch
for higher dimensions.
"""
function limit_convolution_bounds(dim::Int, index::Int, itp::AbstractConvolutionInterpolation{T,N,TCoefs,IT,Axs,GaussianConvolutionKernel{B},HigherDimension{N},Val{DG},EQ}) where {T,N,TCoefs,IT,Axs,DG,EQ,B}
    if index < itp.eqs
        index = itp.eqs
    elseif index > length(itp.knots[dim]) - itp.eqs
        index = length(itp.knots[dim]) - itp.eqs
    end
    return index
end
