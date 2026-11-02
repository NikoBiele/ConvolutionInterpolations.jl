println("\n" * "-"^60)
println("Testing derivative order limits on uniform, nonuniform and mixed grids...")
println("-"^60)

const CI_LIM = ConvolutionInterpolations

# a deterministic nonuniform grid on [a, b]: uniform knots with a ±30 % sine jitter, ends fixed
function jittered_knots(n; a = 0.0, b = 1.0)
    x = collect(range(a, b, length = n))
    h = (b - a) / (n - 1)
    for i in 2:n-1
        x[i] += 0.3 * h * sin(7.0 * i)
    end
    return x
end

@testset "uniform derivative limits" begin
    x = range(0.0, 1.0, length = 41)
    y = exp.(2 .* x)
    for (kernel, kmax) in CI_LIM._max_derivative_uniform
        println("    - uniform limit: ", kernel, " up to ", kmax)
        if kmax > 0
            itp = convolution_interpolation(x, y; kernel, derivative = kmax)
            @test isfinite(itp(0.5))
        end
        @test_throws ErrorException convolution_interpolation(x, y; kernel, derivative = kmax + 1)
        # the direct (non-fast) constructor enforces the same limit
        if kmax > 0
            @test isfinite(ConvolutionInterpolation(x, y; kernel, derivative = kmax)(0.5))
        end
        @test_throws ErrorException ConvolutionInterpolation(x, y; kernel, derivative = kmax + 1)
    end
end

@testset "nonuniform derivative limits" begin
    x = jittered_knots(41)
    y = exp.(2 .* x)
    for (kernel, kmax) in CI_LIM._max_derivative_nonuniform
        println("    - nonuniform limit: ", kernel, " up to ", kmax)
        if kmax > 0
            itp = convolution_interpolation(x, y; kernel, derivative = kmax)
            @test isfinite(itp(0.5))
        end
        @test_throws ErrorException convolution_interpolation(x, y; kernel, derivative = kmax + 1)
    end
    # :a3 evaluates as :n3 on a nonuniform grid and inherits its limit
    @test_throws ErrorException convolution_interpolation(x, y; kernel = :a3, derivative = 1)
end

@testset "mixed grid: the limit is per axis" begin
    println("    - mixed grid, :b13: uniform axis up to 5, nonuniform axis up to 11")
    xu = range(0.0, 1.0, length = 31)
    yn = jittered_knots(31)
    z = [exp(xi + yj) for xi in xu, yj in yn]
    itp = convolution_interpolation((xu, yn), z; kernel = :b13, derivative = (5, 11))
    @test isfinite(itp(0.5, 0.5))
    @test_throws ErrorException convolution_interpolation((xu, yn), z; kernel = :b13, derivative = (6, 11))
    @test_throws ErrorException convolution_interpolation((xu, yn), z; kernel = :b13, derivative = (5, 12))
end

@testset "high nonuniform derivatives converge (:b13, 7th derivative, BigFloat)" begin
    println("    - :b13 7th derivative on a jittered grid, BigFloat, 101 -> 201 knots")
    setprecision(BigFloat, 256) do
        xt = range(BigFloat(0.2), BigFloat(0.8), length = 101)
        errs = map((101, 201)) do n
            x = BigFloat.(jittered_knots(n))
            y = exp.(2 .* x)
            itp = convolution_interpolation(x, y; kernel = :b13, derivative = 7)
            maximum(abs(itp(t) - 2^7 * exp(2t)) for t in xt)
        end
        # asymptotic rate is 2^7 per doubling; ask for a comfortable fraction of it
        @test errs[1] / errs[2] > 2^4
    end
end