#########################################################################
### TEST LAZY BOUNDARY CORRECTNESS ######################################
#########################################################################
println("Testing lazy boundary correctness...")
lazy_tolerance = 1e-6
@testset "Lazy boundary correctness" begin
    # Test that lazy matches eager for a few key cases
    for kernel in (:a3, :b5, :b7)
        # 1D
        vs_1d = Float64.(0:9)
        knots_1d = (range(0.0, 9.0, length=10),)
        itp_e = convolution_interpolation(knots_1d, vs_1d, kernel=kernel, lazy=false)
        itp_l = convolution_interpolation(knots_1d, vs_1d, kernel=kernel, lazy=true)
        for x in [0.5, 4.5, 8.5]
            @test itp_e(x) ≈ itp_l(x) atol=lazy_tolerance
        end

        # 2D
        vs_2d = Float64[i + j for i in 0:9, j in 0:9]
        knots_2d = (range(0.0, 9.0, length=10), range(0.0, 9.0, length=10))
        itp_e = convolution_interpolation(knots_2d, vs_2d, kernel=kernel, lazy=false)
        itp_l = convolution_interpolation(knots_2d, vs_2d, kernel=kernel, lazy=true)
        for (x, y) in [(0.5, 0.5), (4.5, 4.5), (8.5, 8.5), (0.5, 8.5)]
            @test itp_e(x, y) ≈ itp_l(x, y) atol=lazy_tolerance
        end
    end

    # Test that lazy construction is reasonably fast
    vs_big = rand(50, 50, 50)
    knots_big = ntuple(_ -> range(0.0, 1.0, length=50), 3)
    t = @elapsed convolution_interpolation(knots_big, vs_big, kernel=:b5, lazy=true)
    @test t < 2.0  # sub 2 seconds construction
end

#########################################################################
### TEST LAZY NONUNIFORM CORRECTNESS ####################################
#########################################################################
println("Testing lazy nonuniform correctness...")
@testset "Lazy nonuniform n3 correctness" begin
    # 1D: eager vs lazy match
    knots_1d = [0.0, 0.3, 0.7, 1.2, 1.5, 2.0]
    vs_1d = sin.(knots_1d)
    itp_e = convolution_interpolation(knots_1d, vs_1d, kernel=:a3, lazy=false)
    itp_l = convolution_interpolation(knots_1d, vs_1d, kernel=:a3, lazy=true)

    # Verify lazy stores raw values
    @test itp_l.itp.lazy == true
    @test itp_l.itp.coefs === vs_1d
    @test size(itp_l.itp.coefs) == size(vs_1d)

    # Interior, boundary, and near-boundary points
    for x in [0.1, 0.5, 1.0, 1.3, 1.9]
        @test itp_e(x) ≈ itp_l(x) atol=lazy_tolerance
    end

    # 2D: eager vs lazy match
    kx = [0.0, 0.4, 0.9, 1.5, 2.0]
    ky = [0.0, 0.3, 0.8, 1.2, 1.7, 2.0]
    vs_2d = Float64[sin(x + y) for x in kx, y in ky]
    itp_e = convolution_interpolation((kx, ky), vs_2d, kernel=:a3, lazy=false)
    itp_l = convolution_interpolation((kx, ky), vs_2d, kernel=:a3, lazy=true)

    @test itp_l.itp.lazy == true
    @test itp_l.itp.coefs === vs_2d

    for (x, y) in [(0.2, 0.5), (1.0, 1.0), (0.2, 1.8), (1.8, 0.2), (1.8, 1.8)]
        @test itp_e(x, y) ≈ itp_l(x, y) atol=lazy_tolerance
    end

    # 3D: eager vs lazy match
    kz = [0.0, 0.5, 1.0, 1.6, 2.0]
    vs_3d = Float64[sin(x + y + z) for x in kx, y in ky, z in kz]
    itp_e = convolution_interpolation((kx, ky, kz), vs_3d, kernel=:a3, lazy=false)
    itp_l = convolution_interpolation((kx, ky, kz), vs_3d, kernel=:a3, lazy=true)

    @test itp_l.itp.lazy == true
    @test itp_l.itp.coefs === vs_3d

    for (x, y, z) in [(0.5, 0.5, 0.5), (0.2, 1.5, 0.8), (1.8, 0.2, 1.5)]
        @test itp_e(x, y, z) ≈ itp_l(x, y, z) atol=lazy_tolerance
    end

    # Nonuniform b-kernels still force eager
    itp_b = convolution_interpolation(knots_1d, vs_1d, kernel=:b5, lazy=true)
    @test itp_b.itp.lazy == false
end

#################################################################
### TEST LAZY BOUNDARY CORRECTNESS ##############################
#################################################################
println("Testing lazy boundary correctness in 1D, 4D, 5D...")
@testset "Lazy boundary correctness" begin
    for kernel in (:a3, :b5, :b7)
        println("Testing kernel lazy boundary correctness: $kernel...")
        # 1D: boundary_fallback=false still works
        knots_1d = range(0.0, 2*pi, length=20)
        vs_1d = sin.(knots_1d)
        itp_1 = convolution_interpolation(knots_1d, vs_1d, kernel=kernel, lazy=true, boundary_fallback=false)
        @test itp_1(0.01) ≈ sin(0.01) atol=1e-3

        # 4D: lazy=true
        knots = (knots_1d, knots_1d, knots_1d, knots_1d)
        vs_4d = [sin(x)*sin(y)*sin(z)*sin(t) for x in knots_1d, y in knots_1d, z in knots_1d, t in knots_1d]
        itp_3 = convolution_interpolation(knots, vs_4d, kernel=kernel, lazy=true)
        val_4d = itp_3(0.01, 0.01, 0.01, 0.01)
        @test val_4d ≈ sin(0.01)^4 atol=lazy_tolerance

        # 5D lazy with boundary_fallback: returns extrapolated value, no error
        knots = (knots_1d, knots_1d, knots_1d, knots_1d, knots_1d)
        vs_5d = [sin(x)*sin(y)*sin(z)*sin(t)*sin(u) for x in knots_1d, y in knots_1d, z in knots_1d, t in knots_1d, u in knots_1d]
        itp_4 = convolution_interpolation(knots, vs_5d, kernel=kernel, lazy=true, boundary_fallback=true)
        val_5d = itp_4(0.01, 0.01, 0.01, 0.01, 0.01)
        @test isfinite(val_5d)
    end
end