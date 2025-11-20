module ConvolutionInterpolations

using Scratch
using Serialization
using LinearAlgebra

include("data_structures/data_structures.jl")
include("convolution/convolution.jl")
include("convolution_extrapolate/convolution_extrapolate.jl")
include("convolution_kernels/convolution_kernels.jl")
include("convolution_kernel_interpolation/convolution_kernel_interpolation.jl")
include("convolution_fast_interpolation/convolution_fast_interpolation.jl")
include("convolution_coefs/convolution_coefs.jl")

export 
    # Main convenience functions
    convolution_interpolation,

    # Core interpolation types for advanced users
    ConvolutionInterpolation,
    FastConvolutionInterpolation,
    
    # Core Extrapolation type for advanced users
    ConvolutionExtrapolation,
    
    # Boundary conditions
    BoundaryCondition, Line, Flat, Periodic, Reflect, Throw, Natural
end
