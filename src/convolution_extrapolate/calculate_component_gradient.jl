"""
    calculate_component_gradient(etp::ConvolutionExtrapolation{T,N,ITPT,ET,O}, x::NTuple{N,Number}, d::Int, dir::Vector{T}) where {T,N,ITPT,ET,O}

Calculate the high-accuracy partial derivative along a specific dimension at a given point.

# Arguments
- `etp::ConvolutionExtrapolation`: The extrapolation object
- `x::NTuple{N,Number}`: The coordinates at which to evaluate the partial derivative
- `d::Int`: The dimension along which to calculate the partial derivative (1 to N)
- `dir::Vector{T}`: Direction indicators (-1 for left boundary, 0 for interior, 1 for right boundary)

# Returns
- `T`: The partial derivative value along the specified dimension

# Details
This function computes the partial derivative using high-order finite difference approximations
to maintain accuracy, especially for nonlinear functions. Different formulas are used based on
the position (boundary or interior):

- For left boundary points (dir[d] = -1): Uses a fourth-order forward difference scheme
  ```
  (-25f(x) + 48f(x+h) - 36f(x+2h) + 16f(x+3h) - 3f(x+4h)) / (12h)
  ```

- For interior points (dir[d] = 0): Uses a fourth-order central difference scheme
  ```
  (f(x-2h) - 8f(x-h) + 8f(x+h) - f(x+2h)) / (12h)
  ```

- For right boundary points (dir[d] = 1): Uses a fourth-order backward difference scheme
  ```
  (25f(x) - 48f(x-h) + 36f(x-2h) - 16f(x-3h) + 3f(x-4h)) / (12h)
  ```

These high-order formulas provide excellent accuracy for extrapolation, particularly when
dealing with nonlinear functions where the gradient may change rapidly near the boundaries.
"""
function calculate_component_gradient(etp::ConvolutionExtrapolation{T,N,ITPT,ET,O}, x::NTuple{N,Number}, d::Int, dir::Vector{T}) where {T,N,ITPT,ET,O}
    itp = etp.itp
    h = etp.itp.h[d]*0.01

    # Helper function to evaluate interpolator along dimension d
    function eval_along_dim(offset)
        return itp(ntuple(i -> i == d ? x[i] + offset : x[i], N)...)
    end

    if dir[d] == -1
        # Left boundary (forward difference, fourth-order)
        return (-25*eval_along_dim(0.0) + 48*eval_along_dim(h) - 36*eval_along_dim(2h) + 16*eval_along_dim(3h) - 3*eval_along_dim(4h)) / (12h)
        
    elseif dir[d] == 0
        # Interior points (central difference, fourth-order)
        return (eval_along_dim(-2h) - 8*eval_along_dim(-h) + 8*eval_along_dim(h) - eval_along_dim(2h)) / (12h)

    else # if dir[d] == 1
        # Right boundary (backward difference, fourth-order)
        return (25*eval_along_dim(0.0) - 48*eval_along_dim(-h) + 36*eval_along_dim(-2h) - 16*eval_along_dim(-3h) + 3*eval_along_dim(-4h)) / (12h)
    end
end
