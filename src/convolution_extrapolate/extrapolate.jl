"""
    ConvolutionExtrapolation(itp::AbstractConvolutionInterpolation{T,N,TCoefs,IT,Axs,KA,DT,DG,EQ}, et::ET) where {T,N,TCoefs,IT,Axs,KA,DT,DG,EQ,ET<:BoundaryCondition}

Construct a convolution extrapolation object for any interpolation type.

# Arguments
- `itp::AbstractConvolutionInterpolation`: The interpolation object to extend with extrapolation
  (works with both `ConvolutionInterpolation` and `FastConvolutionInterpolation`)
- `et::BoundaryCondition`: The boundary condition to apply outside the interpolation domain

# Returns
- `ConvolutionExtrapolation`: An object that extends the interpolation with the specified boundary condition

# Details
This constructor creates a `ConvolutionExtrapolation` object by wrapping any subtype of 
`AbstractConvolutionInterpolation` with a boundary condition. It properly handles the type
parameters to ensure compatibility with both standard and fast interpolation implementations.

# Examples
```julia
# Create an interpolation object
knots = range(0, 1, length=10)
values = sin.(2π .* knots)

# Regular interpolation with linear extrapolation
itp = ConvolutionInterpolation((knots,), values)
etp = ConvolutionExtrapolation(itp, Line())

# Fast interpolation with flat extrapolation
itp_fast = FastConvolutionInterpolation((knots,), values)
etp_fast = ConvolutionExtrapolation(itp_fast, Flat())
```
"""
function ConvolutionExtrapolation(itp::AbstractConvolutionInterpolation{T,N,TCoefs,IT,Axs,KA,DT,DG,EQ}, et::ET) where {T,N,TCoefs,IT,Axs,KA,DT,DG,EQ,ET<:BoundaryCondition}
    ConvolutionExtrapolation{T,N,typeof(itp),ET,DG}(itp, et)
end