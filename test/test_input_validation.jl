println("\n" * "-"^60)
println("Testing input validation (uniform grids, resample, smooth)...")
println("-"^60)

@testset "Input validation" begin

    @testset "is_uniform_grid follows the rounding of the knot type" begin
        println("    - is_uniform_grid follows the rounding of the knot type...")
        is_uniform = ConvolutionInterpolations.is_uniform_grid
        # uniform grids in several representations
        @test is_uniform(range(0.0, 1.0, length=1001))
        @test is_uniform(collect(range(0.0, 1.0, length=1001)))
        @test is_uniform(Float32.(collect(range(0.0, 1.0, length=1001))))
        @test is_uniform(1.0e9 .+ 0.001 .* (0:999))
        @test is_uniform([0, 1, 2, 3])
        @test is_uniform([BigFloat(i) / 10 for i in 0:10])
        # a deviation above rounding is not uniform, however small against the spacing
        x = collect(range(0.0, 1.0, length=11))
        x[5] += 1e-6
        @test !is_uniform(x)
        x = collect(range(0.0, 1.0, length=11))
        x[5] += 1e-13
        @test !is_uniform(x)
        @test !is_uniform([0, 1, 3])
        # Float64 knots converted to BigFloat keep their Float64 rounding
        @test !is_uniform(BigFloat.(collect(range(0.0, 1.0, length=11))))
        # degenerate vectors
        @test !is_uniform([1.0])
        @test !is_uniform([1.0, 1.0, 1.0])
    end

    @testset "convolution_resample rejects invalid input" begin
        println("    - convolution_resample rejects invalid input...")
        x_in = range(0.0, 1.0, length=41)
        y = sin.(2.0 .* x_in)
        x_out = range(0.0, 1.0, length=81)
        # nonuniform input knots, with and without keywords
        x_bad = collect(x_in)
        x_bad[10] += 1e-3
        @test_throws ArgumentError convolution_resample(x_bad, x_out, y)
        @test_throws ArgumentError convolution_resample((x_bad,), (x_out,), y; kernel=:b5)
        # decreasing input knots
        @test_throws ArgumentError convolution_resample(range(1.0, 0.0, length=41), x_out, y)
        # knots that do not match the data
        @test_throws ArgumentError convolution_resample(range(0.0, 1.0, length=40), x_out, y)
        # output points outside the input range, on either side
        @test_throws ArgumentError convolution_resample(x_in, range(-0.1, 1.0, length=81), y)
        @test_throws ArgumentError convolution_resample(x_in, range(0.0, 1.1, length=81), y)
        # a nonuniform second axis in 2D
        y_in = range(0.0, 1.0, length=41)
        z = [sin(2.0 * xi) * cos(yi) for xi in x_in, yi in y_in]
        y_bad = collect(y_in)
        y_bad[10] += 1e-3
        @test_throws ArgumentError convolution_resample((x_in, y_bad), (x_out, x_out), z)
    end

    @testset "convolution_resample accepts nonuniform output points" begin
        println("    - convolution_resample accepts nonuniform output points...")
        x_in = range(0.0, 1.0, length=41)
        y = sin.(2.0 .* x_in)
        x_out = [0.0, 0.13, 0.5, 0.77, 1.0]
        result = convolution_resample(x_in, x_out, y)
        @test length(result) == length(x_out)
        @test maximum(abs.(result .- sin.(2.0 .* x_out))) < 1e-7
    end

    @testset "convolution_resample defaults to :b7" begin
        println("    - convolution_resample defaults to :b7...")
        x_in = range(0.0, 1.0, length=41)
        y = sin.(2.0 .* x_in)
        x_out = range(0.0, 1.0, length=81)
        @test convolution_resample(x_in, x_out, y) ==
              convolution_resample((x_in,), (x_out,), y; kernel=:b7)
    end

    @testset "convolution_smooth rejects invalid input" begin
        println("    - convolution_smooth rejects invalid input...")
        x = range(0.0, 1.0, length=200)
        y = sin.(2.0 .* x)
        # nonuniform knots
        x_bad = collect(x)
        x_bad[50] += 1e-3
        @test_throws ArgumentError convolution_smooth(x_bad, y, 0.05)
        # knots that do not match the data
        @test_throws ArgumentError convolution_smooth(range(0.0, 1.0, length=199), y, 0.05)
        # uniform knots given as a vector are accepted, in Float64 and in Float32
        @test length(convolution_smooth(collect(x), y, 0.05)) == length(y)
        x32 = Float32.(collect(x))
        y32 = sin.(2.0f0 .* x32)
        @test length(convolution_smooth(x32, y32, 0.05f0)) == length(y32)
    end
end