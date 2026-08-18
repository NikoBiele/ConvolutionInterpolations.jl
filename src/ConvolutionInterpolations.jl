module ConvolutionInterpolations

using Scratch
using Serialization
using LinearAlgebra
using NearestNeighbors
using SparseArrays

include("data_structures/data_structures.jl")
include("convolution/convolution.jl")
include("convolution_extrapolate/convolution_extrapolate.jl")
include("convolution_kernels/convolution_kernels.jl")
include("convolution_kernel_interpolation/convolution_kernel_interpolation.jl")
include("convolution_fast_interpolation/convolution_fast_interpolation.jl")
include("convolution_coefs/convolution_coefs.jl")
include("precomputed_kernels/precomputed_kernel_tables.jl")
include("scattered_to_grid/scattered_to_grid.jl")
include("convolution_fit_scattered/fit_scattered.jl")

export 
    # Main convenience functions
    convolution_interpolation,
    convolution_gaussian,
    convolution_smooth,
    convolution_resample,
    scattered_to_grid,
    fit_scattered,

    # Core interpolation types for advanced users
    ConvolutionInterpolation,
    FastConvolutionInterpolation,
    
    # Core Extrapolation type for advanced users
    ConvolutionExtrapolation,

    # Extrapolation methods
    Throw, Line, Flat, Natural

end