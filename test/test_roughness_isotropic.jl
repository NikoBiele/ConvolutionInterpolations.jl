println("\n" * "-"^60)
println("Testing the isotropic roughness operator...")
println("-"^60)

const CI = ConvolutionInterpolations

@testset "isotropic roughness" begin

    @testset "multi-indices" begin
        println("    - Testing the multi-index enumeration")
        for q in 0:5, D in 1:4
            α = CI._multi_indices(q, D)
            @test length(α) == binomial(q + D - 1, D - 1)        # compositions of q into D parts
            @test all(a -> sum(a) == q, α)
            @test allunique(α)
        end
    end

    @testset "1D: the two seminorms coincide" begin
        println("    - Testing 1D agreement with the axis-wise operator")
        for q in 1:6
            n = (12 + 2 * 5,)
            @test CI._scattered_roughness_iso(n, q, Float64) == CI._scattered_roughness(n, (q,), Float64)
        end
        ax = ([0.0, 0.1, 0.25, 0.3, 0.55, 0.6, 0.9, 1.0],)
        for q in 1:4
            @test CI._scattered_roughness_iso(ax, (2,), q, Float64) == CI._scattered_roughness(ax, (2,), (q,), Float64)
        end
    end

    @testset "2D: null space and mixed weight" begin
        println("    - Testing the 2D null space and the mixed-difference weight")
        n  = (9, 11)
        xs = 0:n[1]-1
        ys = 0:n[2]-1
        grid(f) = vec([Float64(f(x, y)) for x in xs, y in ys])     # column-major, as the kron ordering
        R_iso  = CI._scattered_roughness_iso(n, 2, Float64)
        R_axis = CI._scattered_roughness(n, (2, 2), Float64)
        # both annihilate the polynomials of total degree < 2 (the isotropic operator's mixed
        # rows carry the irrational weight sqrt(2), so their cancellation is to rounding)
        for f in ((x, y) -> 1, (x, y) -> x, (x, y) -> y, (x, y) -> 3x - 2y + 1)
            @test maximum(abs, R_iso * grid(f)) < 1e-12
            @test maximum(abs, R_axis * grid(f)) == 0
        end
        # the isotropic seminorm sees x*y, the axis-wise one does not
        @test maximum(abs, R_axis * grid((x, y) -> x * y)) == 0
        @test maximum(abs, R_iso * grid((x, y) -> x * y)) > 0
        # the mixed block carries weight sqrt(2!/(1!1!)) = sqrt(2): Δx Δy (x*y) = 1 on every row
        @test sum(abs2, R_iso * grid((x, y) -> x * y)) ≈ 2 * (n[1] - 1) * (n[2] - 1)
        # the pure blocks are the axis-wise rows: Δxx (x²) = 2 on every row
        @test sum(abs2, R_iso * grid((x, y) -> x^2)) ≈ 4 * (n[1] - 2) * n[2]
        # row count: one block per multi-index
        @test size(R_iso, 1) == (n[1] - 2) * n[2] + (n[1] - 1) * (n[2] - 1) + n[1] * (n[2] - 2)
    end

    @testset "3D: null space of order 3" begin
        println("    - Testing the 3D null space")
        n  = (7, 8, 9)
        grid(f) = vec([Float64(f(x, y, z)) for x in 0:n[1]-1, y in 0:n[2]-1, z in 0:n[3]-1])
        R = CI._scattered_roughness_iso(n, 3, Float64)
        for f in ((x, y, z) -> 1, (x, y, z) -> x * y, (x, y, z) -> y * z + x^2, (x, y, z) -> x - z^2)
            @test maximum(abs, R * grid(f)) < 1e-12      # mixed rows weighted by sqrt(3), sqrt(6)
        end
        @test maximum(abs, R * grid((x, y, z) -> x * y * z)) > 0
        @test maximum(abs, R * grid((x, y, z) -> x^3)) > 0
    end

    @testset "nonuniform construction on uniform knots" begin
        println("    - Testing the nonuniform operator on uniform knots")
        ax = (collect(range(-1.0, 2.0, length = 10)), collect(range(0.0, 1.0, length = 8)))
        M  = (2, 1)
        n  = (length(ax[1]) + 2M[1], length(ax[2]) + 2M[2])
        for q in 2:4
            R_nu = CI._scattered_roughness_iso(ax, M, q, Float64)
            R_u  = CI._scattered_roughness_iso(n, q, Float64)
            @test size(R_nu) == size(R_u)
            @test maximum(abs, R_nu - R_u) < 1e-12
        end
    end

    @testset "isotropic order" begin
        println("    - Testing the single-order requirement")
        @test CI._isotropic_order((4, 4, 4)) == 4
        @test_throws ArgumentError CI._isotropic_order((4, 5))
    end

    @testset "default orders" begin
        println("    - Testing the default roughness orders")
        @test CI._default_roughness_order(:smooth, :b5, 6, false, 3) == 5
        @test CI._default_roughness_order(:exact, :b5, 6, false, 1) == 5
        @test CI._default_roughness_order(:exact, :b7, 8, false, 1) == 7
        @test CI._default_roughness_order(:exact, :a3, 4, false, 1) == 5
        @test CI._default_roughness_order(:exact, :b13, 14, true, 1) == 7
        @test CI._default_roughness_order(:exact, :a4, 5, false, 2) == 5
        @test CI._default_roughness_order(:exact, :b7, 8, false, 2) == 5
        @test CI._default_roughness_order(:exact, :a4, 5, false, 3) == 5
        @test CI._default_roughness_order(:exact, :a3, 4, false, 4) == 5
        @test CI._default_roughness_order(:exact, :n3, 3, true, 2) == 4
        @test CI._default_roughness_order(:exact, :b5, 6, true, 3) == 5
    end

    @testset "3D :exact fit with the isotropic seminorm" begin
        println("    - Testing a 3D :exact fit")
        rng = MersenneTwister(7)
        pts = 2 .* rand(rng, 3, 400) .- 1
        f   = [1 / (1 + 4 * (p[1]^2 + p[2]^2 + p[3]^2)) for p in eachcol(pts)]
        knots, vals, viol = fit_scattered(pts, f; mode = :exact)
        @test viol < 1e-9
        itp = convolution_interpolation(knots, vals; kernel = :a4, bc = :poly)
        @test maximum(abs(itp(p...) - v) for (p, v) in zip(eachcol(pts), f)) < 1e-8
        @test abs(itp(0.1, -0.2, 0.3) - 1 / (1 + 4 * 0.14)) < 0.05
    end
end