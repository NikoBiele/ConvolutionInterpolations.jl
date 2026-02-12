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

# Recommendation
These constructors are not simple to use manually, so it is recommended to use the `convolution_interpolation` function.
"""
function ConvolutionExtrapolation(itp::AbstractConvolutionInterpolation{T,N,TCoefs,IT,Axs,KA,DT,DG,EQ,DO}, et::ET) where {T,N,TCoefs,IT,Axs,KA,DT,DG,EQ,ET<:BoundaryCondition,DO}
    ConvolutionExtrapolation{T,N,typeof(itp),DG,EQ,ET,DO}(itp, et)
end