#########################################################################
### TEST LAZY MODE — NONUNIFORM GRIDS (n3) ##############################
#########################################################################
println("Testing nonuniform lazy mode...")

@testset "Lazy mode — nonuniform" begin

    @testset "1D n3 lazy matches eager" begin
        println("  1D n3 lazy matches eager...")
        knots = [0.0, 0.3, 0.7, 1.2, 1.5, 2.0]
        vs = sin.(knots)
        itp_e = convolution_interpolation(knots, vs; kernel=:a3, lazy=false)
        itp_l = convolution_interpolation(knots, vs; kernel=:a3, lazy=true)

        @test itp_l.itp.lazy == Val{true}()
        @test itp_l.itp.coefs === vs

        for x in [0.1, 0.5, 1.0, 1.3, 1.9]
            @test itp_e(x) ≈ itp_l(x) atol=1e-6
        end
    end

    @testset "2D n3 lazy matches eager" begin
        println("  2D n3 lazy matches eager...")
        kx = [0.0, 0.4, 0.9, 1.5, 2.0]
        ky = [0.0, 0.3, 0.8, 1.2, 1.7, 2.0]
        vs = Float64[sin(x + y) for x in kx, y in ky]
        itp_e = convolution_interpolation((kx, ky), vs; kernel=:a3, lazy=false)
        itp_l = convolution_interpolation((kx, ky), vs; kernel=:a3, lazy=true)

        @test itp_l.itp.lazy == Val{true}()
        @test itp_l.itp.coefs === vs

        for (x, y) in [(0.2, 0.5), (1.0, 1.0), (0.2, 1.8), (1.8, 0.2), (1.8, 1.8)]
            @test itp_e(x, y) ≈ itp_l(x, y) atol=1e-6
        end
    end

    @testset "3D n3 lazy matches eager" begin
        println("  3D n3 lazy matches eager...")
        kx = [0.0, 0.4, 0.9, 1.5, 2.0]
        ky = [0.0, 0.3, 0.8, 1.2, 1.7, 2.0]
        kz = [0.0, 0.5, 1.0, 1.6, 2.0]
        vs = Float64[sin(x + y + z) for x in kx, y in ky, z in kz]
        itp_e = convolution_interpolation((kx, ky, kz), vs; kernel=:a3, lazy=false)
        itp_l = convolution_interpolation((kx, ky, kz), vs; kernel=:a3, lazy=true)

        @test itp_l.itp.lazy == Val{true}()
        @test itp_l.itp.coefs === vs

        for (x, y, z) in [(0.5, 0.5, 0.5), (0.2, 1.5, 0.8), (1.8, 0.2, 1.5)]
            @test itp_e(x, y, z) ≈ itp_l(x, y, z) atol=1e-6
        end
    end

    @testset "1D n3 lazy boundary_fallback" begin
        println("  1D n3 lazy boundary_fallback...")
        knots = [0.0, 0.3, 0.7, 1.2, 1.5, 2.0]
        vs = sin.(knots)
        itp_bf = convolution_interpolation(knots, vs; kernel=:a3, lazy=true, boundary_fallback=true)
        itp_no = convolution_interpolation(knots, vs; kernel=:a3, lazy=true, boundary_fallback=false)

        # interior should match exactly
        for x in [0.5, 1.0, 1.3]
            @test itp_bf(x) ≈ itp_no(x) atol=1e-10
        end

        # near boundary — boundary_fallback uses linear, should still be close
        for x in [0.05, 1.95]
            @test itp_bf(x) ≈ itp_no(x) atol=0.1
        end
    end

    @testset "2D n3 lazy boundary_fallback" begin
        println("  2D n3 lazy boundary_fallback...")
        kx = [0.0, 0.4, 0.9, 1.5, 2.0]
        ky = [0.0, 0.3, 0.8, 1.2, 1.7, 2.0]
        vs = Float64[sin(x + y) for x in kx, y in ky]
        itp_bf = convolution_interpolation((kx, ky), vs; kernel=:a3, lazy=true, boundary_fallback=true)
        itp_no = convolution_interpolation((kx, ky), vs; kernel=:a3, lazy=true, boundary_fallback=false)

        # interior should match exactly
        for (x, y) in [(1.0, 1.0), (0.8, 1.5)]
            @test itp_bf(x, y) ≈ itp_no(x, y) atol=1e-10
        end

        # near boundary — fallback uses linear, should still be close
        for (x, y) in [(0.05, 0.5), (1.95, 1.95), (0.05, 1.95)]
            @test itp_bf(x, y) ≈ itp_no(x, y) atol=0.1
        end
    end

    @testset "3D n3 lazy boundary_fallback" begin
        println("  3D n3 lazy boundary_fallback...")
        kx = [0.0, 0.4, 0.9, 1.5, 2.0]
        ky = [0.0, 0.3, 0.8, 1.2, 1.7, 2.0]
        kz = [0.0, 0.5, 1.0, 1.6, 2.0]
        vs = Float64[sin(x + y + z) for x in kx, y in ky, z in kz]
        itp_bf = convolution_interpolation((kx, ky, kz), vs; kernel=:a3, lazy=true, boundary_fallback=true)
        itp_no = convolution_interpolation((kx, ky, kz), vs; kernel=:a3, lazy=true, boundary_fallback=false)

        # interior should match exactly
        for (x, y, z) in [(0.5, 0.5, 0.5), (1.0, 1.0, 1.0)]
            @test itp_bf(x, y, z) ≈ itp_no(x, y, z) atol=1e-10
        end

        # near boundary — both should agree reasonably
        for (x, y, z) in [(0.05, 0.5, 0.5), (1.95, 1.95, 1.95)]
            @test itp_bf(x, y, z) ≈ itp_no(x, y, z) atol=0.1
        end
    end

    @testset "Nonuniform n3 lazy stores raw values" begin
        println("  Nonuniform n3 lazy stores raw values...")
        xs = [0.0, 0.3, 0.7, 1.2, 1.5, 2.0]
        vs = sin.(xs)
        itp_l = convolution_interpolation(xs, vs; kernel=:a3, lazy=true)
        @test itp_l.itp.lazy == Val{true}()
        @test itp_l.itp.coefs === vs
    end

    @testset "Nonuniform b-kernels force eager despite lazy=true" begin
        println("  Nonuniform b-kernels force eager despite lazy=true...")
        xs = [0.0, 0.3, 0.7, 1.2, 1.5, 2.0]
        vs = sin.(xs)
        itp_b = convolution_interpolation(xs, vs; kernel=:b5, lazy=true)
        @test itp_b.itp.lazy == Val{false}()
    end

    @testset "Nonuniform n3 lazy works with :n3 kernel explicitly" begin
        println("  Nonuniform n3 lazy works with :n3 kernel explicitly...")
        x_nonuniform = sort([0.0; rand(18) .* 2π; 2π])
        y_nu = sin.(x_nonuniform)
        itp = convolution_interpolation(x_nonuniform, y_nu; kernel=:n3, lazy=true)
        @test itp(1.0) isa Float64
        @test isfinite(itp(1.0))
    end
end