println("Testing per-dim nonuniform b-kernel combinations...")
@testset "Per-dim nonuniform b-kernel combinations" begin

    @testset "2D (:b5, :b7)" begin
        println("Testing 2D (:b5, :b7) interpolation")
        x = range(0.0, 2π, length=15) .+ 1e-4 * rand(15)
        y = range(0.0, 2π, length=15) .+ 1e-4 * rand(15)
        f(x, y) = sin(x) * cos(y)
        vs = [f(xi, yi) for xi in x, yi in y]
        itp = convolution_interpolation((x, y), vs; kernel=(:b5, :b7))
        @test itp(1.5, 1.0) ≈ f(1.5, 1.0) atol=1e-4
    end

    @testset "2D (:b7, :b5)" begin
        println("Testing 2D (:b7, :b5) interpolation")
        x = range(0.0, 2π, length=15) .+ 1e-4 * rand(15)
        y = range(0.0, 2π, length=15) .+ 1e-4 * rand(15)
        f(x, y) = sin(x) * cos(y)
        vs = [f(xi, yi) for xi in x, yi in y]
        itp = convolution_interpolation((x, y), vs; kernel=(:b7, :b5))
        @test itp(1.5, 1.0) ≈ f(1.5, 1.0) atol=1e-4
    end

    @testset "3D (:b5, :b7, :b5)" begin
        println("Testing 3D (:b5, :b7, :b5) interpolation")
        x = range(0.0, 2π, length=10) .+ 1e-4 * rand(10)
        y = range(0.0, 2π, length=10) .+ 1e-4 * rand(10)
        z = range(0.0, 2π, length=10) .+ 1e-4 * rand(10)
        f(x, y, z) = sin(x) * cos(y) * sin(z)
        vs = [f(xi, yi, zi) for xi in x, yi in y, zi in z]
        itp = convolution_interpolation((x, y, z), vs; kernel=(:b5, :b7, :b5))
        @test itp(1.5, 1.0, 1.0) ≈ f(1.5, 1.0, 1.0) atol=1e-3
    end

    @testset "3D (:b7, :b5, :b7)" begin
        println("Testing 3D (:b7, :b5, :b7) interpolation")
        x = range(0.0, 2π, length=10) .+ 1e-4 * rand(10)
        y = range(0.0, 2π, length=10) .+ 1e-4 * rand(10)
        z = range(0.0, 2π, length=10) .+ 1e-4 * rand(10)
        f(x, y, z) = sin(x) * cos(y) * sin(z)
        vs = [f(xi, yi, zi) for xi in x, yi in y, zi in z]
        itp = convolution_interpolation((x, y, z), vs; kernel=(:b7, :b5, :b7))
        @test itp(1.5, 1.0, 1.0) ≈ f(1.5, 1.0, 1.0) atol=1e-3
    end

end