module ConvolutionInterpolations

using LinearAlgebra
using SparseArrays

include("data_structures/data_structures.jl")
include("convolution/convolution.jl")
include("convolution_extrapolate/convolution_extrapolate.jl")
include("convolution_kernels/convolution_kernels.jl")
include("convolution_interpolation/convolution_interpolation.jl")
include("convolution_coefs/convolution_coefs.jl")
include("convolution_fit_scattered/convolution_fit_scattered.jl")

export 
    # Main convenience functions
    convolution_interpolation,
    convolution_gaussian,
    convolution_smooth,
    convolution_resample,
    fit_scattered,

    # Core interpolation type for advanced users
    ConvolutionInterpolation,
    
    # Core Extrapolation type for advanced users
    ConvolutionExtrapolation,

    # Extrapolation methods
    Throw, Line, Flat, Natural

end