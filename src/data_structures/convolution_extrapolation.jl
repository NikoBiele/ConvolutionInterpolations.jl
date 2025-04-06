"""
    ConvolutionExtrapolation{T,N,ITPT<:AbstractConvolutionInterpolation,ET<:BoundaryCondition,O}

A structure that combines a convolution interpolation with boundary condition extrapolation techniques.

# Type Parameters
- `T`: The data type of the interpolated values
- `N`: The dimensionality of the data (number of dimensions)
- `ITPT`: The specific type of convolution interpolation being used
- `ET`: The boundary condition extrapolation technique
- `O`: The order of the interpolation

# Fields
- `itp::ITPT`: The underlying interpolation object
- `et::ET`: The extrapolation technique (boundary condition)

# Details
This structure extends the functionality of convolution interpolation objects by defining 
behavior for points outside the domain of the original data through various extrapolation 
techniques.

When called as a function, the object automatically determines whether the requested point
is within the interpolation domain. If so, it uses direct interpolation; if not, it applies
the specified boundary condition to handle the extrapolation.

# Examples
```julia
# Create an interpolation object
knots = range(0, 1, length=10)
values = sin.(2π .* knots)
itp = ConvolutionInterpolation((knots,), values)

# Extend with linear extrapolation
etp = ConvolutionExtrapolation(itp, Line())

# Evaluate inside and outside the domain
etp(0.5)   # Uses interpolation
etp(1.2)   # Uses extrapolation
```
"""
struct ConvolutionExtrapolation{T,N,ITPT<:AbstractConvolutionInterpolation,ET<:BoundaryCondition,O}
    itp::ITPT
    et::ET
end