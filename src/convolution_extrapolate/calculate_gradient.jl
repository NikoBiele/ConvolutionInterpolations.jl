"""
    calculate_gradient(etp::ConvolutionExtrapolation{T,N,ITPT,ET,O}, x::NTuple{N,Number}, dir::Vector{T}) where {T,N,ITPT,ET,O}

Calculate the gradient vector (all partial derivatives) of the interpolated function at a given point.

# Arguments
- `etp::ConvolutionExtrapolation`: The extrapolation object
- `x::NTuple{N,Number}`: The coordinates at which to evaluate the gradient
- `dir::Vector{T}`: Direction indicators for each dimension (-1 for left boundary, 0 for interior, 1 for right boundary)

# Returns
- `Vector{T}`: The gradient vector of length N, containing high-accuracy partial derivatives for each dimension

# Details
This function calculates the complete gradient vector by computing the partial derivative
in each dimension using `calculate_component_gradient`. The resulting gradient vector
contains the rate of change of the interpolated function with respect to each coordinate.

The gradient is used primarily for linear extrapolation beyond the boundaries, providing
a more accurate approximation than simple constant extrapolation. By calculating separate
derivatives for each dimension, the function properly handles directional changes in the
underlying data.

Each component of the gradient is computed using high-order finite difference methods
appropriate to its position (boundary or interior), ensuring accuracy even for nonlinear
functions.
"""
function calculate_gradient(etp::ConvolutionExtrapolation{T,N,ITPT,ET,O}, x::NTuple{N,Number}, dir::Vector{T}) where {T,N,ITPT,ET,O} 
    itp = etp.itp
    knots = getknots(itp)
    grad = zeros(T, N)

    for d in 1:N
        grad[d] = calculate_component_gradient(etp, x, d, dir)
    end

    return grad
end