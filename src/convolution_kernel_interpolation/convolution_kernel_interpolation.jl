# eager
include("convolution_kernel_interpolation_eager_1d.jl")
include("convolution_kernel_interpolation_eager_2d.jl")
include("convolution_kernel_interpolation_eager_3d.jl")
include("convolution_kernel_interpolation_eager_nd.jl")
# lazy
include("convolution_kernel_interpolation_lazy_1d.jl")
include("convolution_kernel_interpolation_lazy_2d.jl")
include("convolution_kernel_interpolation_lazy_3d.jl")
include("convolution_kernel_interpolation_lazy_nd.jl")
# pure integration
include("convolution_kernel_integration_1d.jl")
include("convolution_kernel_integration_2d.jl")
include("convolution_kernel_integration_3d.jl")
include("convolution_kernel_integration_nd.jl")
# mixed integration
include("convolution_kernel_mixed_integral_2d.jl")
include("convolution_kernel_mixed_integral_3d.jl")
include("convolution_kernel_mixed_integral_nd.jl")
# nonuniform
include("convolution_nonuniform_b_kernels_eager.jl")
include("convolution_nonuniform_n3_kernel_eager.jl")
include("convolution_nonuniform_n3_kernel_lazy.jl")
include("convolution_nonuniform_a0_a1_kernels.jl")