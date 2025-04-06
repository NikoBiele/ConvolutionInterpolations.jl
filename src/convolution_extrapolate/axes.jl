"""
    Base.axes(etp::ConvolutionExtrapolation)

Get the axes of the coefficient array.

# Arguments
- `etp::ConvolutionExtrapolation`: The extrapolation object

# Returns
- The axes of the coefficient array from the underlying interpolation

# Details
This method forwards the request to the underlying interpolation object's
coefficient array, providing the index ranges for each dimension.
"""
Base.axes(etp::ConvolutionExtrapolation) = axes(etp.itp.coefs)