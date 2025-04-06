"""
    getknots(etp::ConvolutionExtrapolation)

Get the knot points of the interpolation grid.

# Arguments
- `etp::ConvolutionExtrapolation`: The extrapolation object

# Returns
- The knots of the underlying interpolation object

# Details
This method forwards the request to the underlying interpolation object,
returning the same knots as used by the original interpolation.
"""
getknots(etp::ConvolutionExtrapolation) = getknots(etp.itp)
