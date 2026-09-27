# eager
include("convolution_fast_interpolation_eager_1d.jl")
include("convolution_fast_interpolation_eager_2d.jl")
include("convolution_fast_interpolation_eager_3d.jl")
include("convolution_fast_interpolation_eager_nd.jl")
# lazy
include("convolution_fast_interpolation_lazy_1d.jl")
include("convolution_fast_interpolation_lazy_2d.jl")
include("convolution_fast_interpolation_lazy_3d.jl")
include("convolution_fast_interpolation_lazy_nd.jl")
# perdim
include("convolution_fast_interpolation_perdim_2d.jl")
include("convolution_fast_interpolation_perdim_3d.jl")
include("convolution_fast_interpolation_perdim_nd.jl")
# general integral order, general dimension, mixed integrals
include("convolution_fast_integration_general.jl")