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

# This struct is not simple to use manually, so it is recommended to use the `convolution_interpolation` function.
"""
struct ConvolutionExtrapolation{T,N,ITPT<:AbstractConvolutionInterpolation,DG,EQ,ET<:BoundaryCondition,DO}
    itp::ITPT
    et::ET
end