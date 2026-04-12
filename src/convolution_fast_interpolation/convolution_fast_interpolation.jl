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
# full integrals
include("convolution_fast_integration_1d.jl")
include("convolution_fast_integration_2d.jl")
include("convolution_fast_integration_3d.jl")
include("convolution_fast_integration_nd.jl")
# mixed integrals
include("convolution_fast_mixed_1_integral_dim.jl")
include("convolution_fast_mixed_2_integral_dims.jl")
include("convolution_fast_mixed_3_integral_dims.jl")
include("convolution_fast_mixed_N_integral_dims.jl")
# helpers
include("precompute_kernel_and_range.jl")
include("get_precomputed_kernel_and_range.jl")
include("quintic_hermite_interpolation.jl")
include("cubic_hermite_interpolation.jl")
include("eval_subgrid_helpers.jl")