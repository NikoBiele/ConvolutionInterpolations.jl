"""
    ConvolutionExtrapolation(itp::AbstractConvolutionInterpolation, et::AbstractExtrapolation)

Wrap an interpolant with an extrapolation rule, so that it can be evaluated outside its grid.

# Arguments
- `itp`: The interpolant to extend, a `ConvolutionInterpolation` or a `FastConvolutionInterpolation`
- `et`: The rule applied outside the grid: `Throw()`, `Flat()`, `Line()` or `Natural()`

# Returns
- `ConvolutionExtrapolation`: The interpolant with the rule; inside the grid it evaluates `itp`
  unchanged, and the domain bounds are stored with a tolerance of a few `eps` so that points on
  the boundary count as inside

`convolution_interpolation` builds this wrapper from its `extrap` keyword, which is the
recommended way to get one.
"""
function ConvolutionExtrapolation(itp::AbstractConvolutionInterpolation{T,N}, et::ET) where
                                    {T,N,ET<:AbstractExtrapolation}
    lo = ntuple(d -> T(_domain_bounds(itp, d)[1]), N)
    hi = ntuple(d -> T(_domain_bounds(itp, d)[2]), N)
    tol = ntuple(d -> 8 * eps(T) * max(abs(lo[d]), abs(hi[d]), one(T)), N)
    ConvolutionExtrapolation{T,N,typeof(itp),ET}(itp, et, lo, hi, tol)
end