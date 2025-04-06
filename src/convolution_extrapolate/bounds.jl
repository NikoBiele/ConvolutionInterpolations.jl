"""
    bounds(etp::ConvolutionExtrapolation)

Get the bounds of the interpolation domain.

# Arguments
- `etp::ConvolutionExtrapolation`: The extrapolation object

# Returns
- The bounds of the underlying interpolation object

# Details
This method forwards the request to the underlying interpolation object,
returning the same bounds as the original interpolation domain.
"""
bounds(etp::ConvolutionExtrapolation) = bounds(etp.itp)
