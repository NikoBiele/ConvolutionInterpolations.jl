#########################################################################
### TEST NONUNIFORM :a0/:a1 INTERPOLATION ###############################
#########################################################################
println("Testing nonuniform :a0/:a1 interpolation...")
@testset "Nonuniform a0 nearest-neighbor" begin
    knots = [0.0, 0.3, 0.7, 1.2, 1.5, 2.0]
    vs = sin.(knots)
    itp = convolution_interpolation(knots, vs, kernel=:a0)

    # On-grid reproduction
    for (i, k) in enumerate(knots)
        @test itp(k) ≈ vs[i] atol=1e-6
    end

    # 2D on-grid
    kx = [0.0, 0.4, 0.9, 1.5, 2.0]
    ky = [0.0, 0.3, 0.8, 1.2, 2.0]
    vs_2d = Float64[sin(x + y) for x in kx, y in ky]
    itp_2d = convolution_interpolation((kx, ky), vs_2d, kernel=:a0)
    for (i, x) in enumerate(kx), (j, y) in enumerate(ky)
        @test itp_2d(x, y) ≈ vs_2d[i, j] atol=1e-6
    end
end

@testset "Nonuniform a1 linear interpolation" begin
    knots = [0.0, 0.3, 0.7, 1.2, 1.5, 2.0]
    vs = sin.(knots)
    itp = convolution_interpolation(knots, vs, kernel=:a1)

    # On-grid reproduction
    for (i, k) in enumerate(knots)
        @test itp(k) ≈ vs[i] atol=1e-4
    end

    # Midpoint = average of neighbors
    for i in 1:length(knots)-1
        xm = (knots[i] + knots[i+1]) / 2
        @test itp(xm) ≈ (vs[i] + vs[i+1]) / 2 atol=1e-6
    end

    # 1D linear data: exact everywhere
    vs_lin = Float64.(1:6)
    itp_lin = convolution_interpolation(knots, vs_lin, kernel=:a1)
    for x in [0.1, 0.5, 0.85, 1.35, 1.8]
        # Find interval and compute expected linear interp
        i = searchsortedlast(knots, x)
        i = clamp(i, 1, length(knots)-1)
        t = (x - knots[i]) / (knots[i+1] - knots[i])
        expected = (1 - t) * vs_lin[i] + t * vs_lin[i+1]
        @test itp_lin(x) ≈ expected atol=1e-6
    end

    # 2D bilinear: exact for linear data f(x,y) = x + y
    kx = [0.0, 0.4, 0.9, 1.5, 2.0]
    ky = [0.0, 0.3, 0.8, 1.2, 2.0]
    vs_2d = Float64[x + y for x in kx, y in ky]
    itp_2d = convolution_interpolation((kx, ky), vs_2d, kernel=:a1)

    for (i, x) in enumerate(kx), (j, y) in enumerate(ky)
        @test itp_2d(x, y) ≈ vs_2d[i, j] atol=1e-6
    end
    for (x, y) in [(0.2, 0.5), (1.0, 1.0), (0.5, 1.5), (1.8, 0.1)]
        @test itp_2d(x, y) ≈ x + y atol=1e-6
    end

    # 3D trilinear: exact for linear data f(x,y,z) = x + y + z
    kz = [0.0, 0.5, 1.0, 1.6, 2.0]
    vs_3d = Float64[x + y + z for x in kx, y in ky, z in kz]
    itp_3d = convolution_interpolation((kx, ky, kz), vs_3d, kernel=:a1)

    for (x, y, z) in [(0.2, 0.5, 0.3), (1.0, 1.0, 0.8), (1.8, 0.1, 1.5)]
        @test itp_3d(x, y, z) ≈ x + y + z atol=1e-6
    end
end

@testset "Nonuniform a0/a1 lazy fields" begin
    knots = [0.0, 0.3, 0.7, 1.2, 1.5, 2.0]
    vs = sin.(knots)

    itp_a0 = convolution_interpolation(knots, vs, kernel=:a0)
    @test itp_a0.itp.lazy == true
    @test itp_a0.itp.coefs === vs

    itp_a1 = convolution_interpolation(knots, vs, kernel=:a1)
    @test itp_a1.itp.lazy == true
    @test itp_a1.itp.coefs === vs
end
