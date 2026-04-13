#########################################################################
### TEST LAZY MODE — UNIFORM GRIDS ######################################
#########################################################################
println("Testing uniform lazy mode...")

@testset "Lazy mode — uniform" begin

    @testset "1D lazy boundary_fallback=false matches eager" begin
        println("  1D lazy boundary_fallback=false matches eager...")
        for kernel in (:a3, :b5)
            xs = range(0.0, 2π, length=50)
            vs = sin.(xs)
            itp_e = convolution_interpolation(xs, vs; kernel=kernel, lazy=false)
            itp_l = convolution_interpolation(xs, vs; kernel=kernel, lazy=true, boundary_fallback=false)
            for x in range(0.01, 2π-0.01, length=30)
                @test itp_e(x) ≈ itp_l(x) atol=1e-10
            end
        end
    end

    @testset "2D lazy boundary_fallback=false matches eager" begin
        println("  2D lazy boundary_fallback=false matches eager...")
        for kernel in (:a3, :b5)
            xs = range(0.0, 2π, length=40)
            vs = [sin(x)*cos(y) for x in xs, y in xs]
            itp_e = convolution_interpolation((xs, xs), vs; kernel=kernel, lazy=false)
            itp_l = convolution_interpolation((xs, xs), vs; kernel=kernel, lazy=true, boundary_fallback=false)
            for x in [0.02, 1.0, 2.5, 5.5, 2π-0.02]
                for y in [0.02, 1.5, 4.0, 2π-0.02]
                    @test itp_e(x, y) ≈ itp_l(x, y) atol=1e-10
                end
            end
        end
    end

    @testset "3D lazy boundary_fallback=false matches eager" begin
        println("  3D lazy boundary_fallback=false matches eager...")
        for kernel in (:a3, :b5)
            xs = range(0.0, 2π, length=30)
            vs = [sin(x)*cos(y)*sin(z) for x in xs, y in xs, z in xs]
            itp_e = convolution_interpolation((xs, xs, xs), vs; kernel=kernel, lazy=false)
            itp_l = convolution_interpolation((xs, xs, xs), vs; kernel=kernel, lazy=true, boundary_fallback=false)
            for (x, y, z) in [(0.02, 0.5, 1.0), (1.5, 2.0, 3.0), (5.5, 5.0, 4.5), (2π-0.2, 2π-0.3, 2π-0.04)]
                @test itp_e(x, y, z) ≈ itp_l(x, y, z) atol=1e-10
            end
        end
    end

    @testset "4D lazy boundary_fallback=true interior matches eager" begin
        println("  4D lazy boundary_fallback=true interior matches eager...")
        for kernel in (:a3, :b5)
            xs = range(0.0, 2π, length=30)
            vs = [sin(x)*cos(y)*sin(z)*cos(t) for x in xs, y in xs, z in xs, t in xs]
            itp_e = convolution_interpolation((xs, xs, xs, xs), vs;
                        kernel=kernel, lazy=false, boundary_fallback=false)
            itp_l = convolution_interpolation((xs, xs, xs, xs), vs;
                        kernel=kernel, lazy=true, boundary_fallback=true)
            # interior
            for (x, y, z, t) in [(1.0, 1.5, 2.0, 2.5), (2.0, 2.5, 3.0, 3.5), (3.0, 3.5, 4.0, 4.5)]
                @test itp_e(x, y, z, t) ≈ itp_l(x, y, z, t) atol=1e-10
            end
            # near boundary — linear fallback
            for (x, y, z, t) in [(0.1, 0.1, 0.1, 0.1), (2π-0.1, 2π-0.1, 2π-0.1, 2π-0.1)]
                @test itp_e(x, y, z, t) ≈ itp_l(x, y, z, t) atol=1e-2
            end
        end
    end

    @testset "Lazy construction is fast" begin
        println("  Lazy construction is fast...")
        xs_3d = range(0.0, 1.0, length=50)
        vs_3d = rand(50, 50, 50)
        convolution_interpolation((xs_3d, xs_3d, xs_3d), vs_3d; kernel=:b5, lazy=true)
        t = @elapsed convolution_interpolation((xs_3d, xs_3d, xs_3d), vs_3d; kernel=:b5, lazy=true)
        @test t < 1.0

        xs_4d = range(0.0, 1.0, length=20)
        vs_4d = rand(20, 20, 20, 20)
        convolution_interpolation((xs_4d, xs_4d, xs_4d, xs_4d), vs_4d;
                    kernel=:b5, lazy=true, boundary_fallback=true)
        t = @elapsed convolution_interpolation((xs_4d, xs_4d, xs_4d, xs_4d), vs_4d;
                    kernel=:b5, lazy=true, boundary_fallback=true)
        @test t < 1.0
    end

    @testset "Lazy stores raw values" begin
        println("  Lazy stores raw values...")
        xs = range(0.0, 2π, length=20)
        vs = sin.(xs)
        itp_l = convolution_interpolation(xs, vs; kernel=:b5, lazy=true)
        @test itp_l.itp.lazy == Val{true}()
        @test itp_l.itp.coefs === vs
    end

    @testset "Lazy with derivatives (boundary_fallback=false)" begin
        println("  Lazy with derivatives (boundary_fallback=false)...")
        xs = range(0.0, 2π, length=50)
        vs = sin.(xs)
        for d in (1, 2)
            itp_e = convolution_interpolation(xs, vs; kernel=:b5, derivative=d, lazy=false)
            itp_l = convolution_interpolation(xs, vs; kernel=:b5, derivative=d, lazy=true, boundary_fallback=false)
            for x in range(0.5, 2π-0.5, length=15)
                @test itp_e(x) ≈ itp_l(x) atol=1e-10
            end
        end
    end

    @testset "Slow lazy functor error guard" begin
        println("  Slow lazy functor error guard...")
        x_uniform = range(0.0, 2π, length=20)
        y = sin.(x_uniform)
        @test_throws ErrorException convolution_interpolation(x_uniform, y; lazy=true, fast=false)
    end
end