###################################################################
### TEST CONTINUITY, POLYNOMIAL REPRODUCTION, CONVERGENCE ###
###################################################################

using ConvolutionInterpolations
using Printf

kernels = [:a0, :a1, :a3, :a4, :a5, :a7, :b5, :b7, :b9, :b11, :b13]

println("="^80)
println("POLYNOMIAL REPRODUCTION TEST")
println("="^80)
println()

N = 200
x = range(-1.0, 1.0, length=N)
x_test = range(-0.9, 0.9, length=500)

for kernel in kernels
    @printf("%-6s |  ", kernel)
    max_repro = -1
    for deg in 0:25
        y = x .^ deg
        try
            itp = convolution_interpolation(x, y; degree=kernel)
            maxerr = maximum(abs.(itp.(x_test) .- x_test .^ deg))
            if maxerr < 1e-10
                max_repro = deg
            end
            @printf("d%d:%.0e ", deg, maxerr)
        catch e
            @printf("d%d:ERR ", deg)
        end
    end
    @printf(" → repro≤%d\n", max_repro)
end

println()
println("="^80)
println("CONVERGENCE ORDER TEST (Runge function)")
println("="^80)
println()

f(x) = 1 / (1 + 25x^2)

for kernel in kernels
    @printf("%-6s |  ", kernel)
    prev_err = nothing
    for N_pts in [50, 100, 200, 400]
        xg = range(-1.0, 1.0, length=N_pts)
        yg = f.(xg)
        xt = range(-0.9, 0.9, length=1000)
        try
            itp = convolution_interpolation(xg, yg; degree=kernel)
            maxerr = maximum(abs.(itp.(xt) .- f.(xt)))
            if prev_err !== nothing && maxerr > 0 && prev_err > 0
                order = log2(prev_err / maxerr)
                @printf("N=%d:%.0e(%.1f) ", N_pts, maxerr, order)
            else
                @printf("N=%d:%.0e      ", N_pts, maxerr)
            end
            prev_err = maxerr
        catch e
            @printf("N=%d:ERR      ", N_pts)
            prev_err = nothing
        end
    end
    println()
end

println()
println("="^80)
println("DERIVATIVE SUPPORT TEST")
println("="^80)
println()

x_sin = range(0.0, 2π, length=200)
y_sin = sin.(x_sin)

for kernel in kernels
    @printf("%-6s |  ", kernel)
    max_deriv = 0
    for d in 1:8
        try
            itp = convolution_interpolation(x_sin, y_sin; degree=kernel, derivative=d)
            itp(1.0)  # just test it works
            max_deriv = d
            @printf("d%d:OK ", d)
        catch e
            @printf("d%d:no ", d)
            break
        end
    end
    @printf(" → max_deriv=%d\n", max_deriv)
end