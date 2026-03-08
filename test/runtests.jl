using ConvolutionInterpolations
using Test

# shared helper
function make_nonuniform_grid(n; a=0.0, b=1.0, strength=0.3)
    x = collect(range(a, b, length=n))
    h = (b - a) / (n - 1)
    for i in 2:n-1
        x[i] += strength * h * sin(3π * (x[i] - a) / (b - a))
    end
    sort!(x)
    return x
end

@testset "ConvolutionInterpolations.jl" begin
    include("test_constructors.jl")
    include("test_uniform_interpolation.jl")
    include("test_uniform_derivatives.jl")
    include("test_uniform_convergence.jl")
    include("test_nonuniform_interpolation.jl")
    include("test_nonuniform_derivatives.jl")
    include("test_nonuniform_convergence.jl")
    include("test_lazy.jl")
    include("test_nonuniform_a0_a1.jl")
    include("test_antiderivative.jl")
end
