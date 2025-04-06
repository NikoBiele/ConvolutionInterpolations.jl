"""
    Base.size(etp::ConvolutionExtrapolation)

Get the size of the coefficient array.

# Arguments
- `etp::ConvolutionExtrapolation`: The extrapolation object

# Returns
- The size of the coefficient array from the underlying interpolation

# Details
This method forwards the request to the underlying interpolation object's
coefficient array, providing the dimensions of the data.
"""
Base.size(etp::ConvolutionExtrapolation) = size(etp.itp.coefs)