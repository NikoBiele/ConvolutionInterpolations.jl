using ConvolutionInterpolations
using Test

@testset "ConvolutionInterpolations.jl" begin
    include("test_constructors.jl")
    include("test_uniform_interpolation.jl")
    include("test_uniform_derivatives.jl")
    include("test_uniform_convergence.jl")
    include("test_nonuniform_interpolation.jl")
    include("test_nonuniform_derivatives.jl")
    include("test_nonuniform_convergence.jl")
    include("test_nonuniform_perdim_derivatives.jl")
    include("test_lazy.jl")
    include("test_nonuniform_a0_a1.jl")
    include("test_antiderivative.jl")
    include("test_perdim_derivatives.jl")
    include("test_mixed_integral.jl")
end