println("\n" * "-"^60)
println("Testing fitting scattered points to a grid...")
println("-"^60)

# analytic test functions with closed-form derivatives and integrals
g2(x, y)     = sin(3x) * cos(2y)
g2_dx(x, y)  = 3cos(3x) * cos(2y)
g2_dxx(x, y) = -9sin(3x) * cos(2y)
g2_int(a, b, c, d) = (cos(3a) - cos(3b)) / 3 * (sin(2d) - sin(2c)) / 2
g3(x, y, z)  = sin(3x) * cos(2y) * exp(-z)
g1(x)        = sin(6x) + 0.3cos(11x)

function jittered_pts(n_side, jitter, rng; dims = 2)
    pts = Matrix{Float64}(undef, dims, n_side^dims)
    for (k, ci) in enumerate(CartesianIndices(ntuple(_ -> n_side, dims)))
        for d in 1:dims
            pts[d, k] = clamp((ci[d] - 0.5) / n_side +
                              jitter / n_side * (2rand(rng) - 1), 0.0, 1.0)
        end
    end
    pts
end

@testset "fit_scattered" begin

    @testset "exactness at the data (1D/2D/3D)" begin
        println("    - Testing exactness at fitting scattered 1D/2D/3D")
        for (dims, n_side, tol) in ((1, 200, 1e-9), (2, 25, 1e-9), (3, 8, 1e-8))
            rng = MersenneTwister(7)
            pts = jittered_pts(n_side, 0.35, rng; dims)
            f = dims == 1 ? g1.(vec(pts)) :
                dims == 2 ? [g2(pts[1,i], pts[2,i]) for i in 1:size(pts,2)] :
                            [g3(pts[1,i], pts[2,i], pts[3,i]) for i in 1:size(pts,2)]
            itp = convolution_interpolation(pts, f)
            res = maximum(abs(itp(pts[:, i]...) - f[i]) for i in 1:size(pts, 2))
            @test res < tol
        end
    end

    @testset "perturbed grid ~ true grid (consistency)" begin
        println("    - Testing perturbed grid consistency")
        rng = MersenneTwister(11)
        n = 21
        xg = range(0.0, 1.0, length = n)
        itp_grid = FastConvolutionInterpolation((xg, xg),
                    [g2(x, y) for x in xg, y in xg]; kernel = :b5, bc = :poly)
        h = step(xg)
        pts = Matrix{Float64}(undef, 2, n^2)
        for (k, ci) in enumerate(CartesianIndices((n, n)))
            pts[1, k] = clamp(xg[ci[1]] + 1e-3h * (2rand(rng) - 1), 0.0, 1.0)
            pts[2, k] = clamp(xg[ci[2]] + 1e-3h * (2rand(rng) - 1), 0.0, 1.0)
        end
        vals = [g2(pts[1,i], pts[2,i]) for i in 1:size(pts, 2)]
        itp_sc = convolution_interpolation(pts, vals; kernel = :b5)
        ev = range(0.1, 0.9, length = 21)
        dmax = maximum(abs(itp_sc(x, y) - itp_grid(x, y)) for x in ev, y in ev)
        @test dmax < 1e-3          # O(perturbation * f') + both fit errors
    end

    @testset "convergence order spot-check (2D, :b5)" begin
        println("    - Testing scattered fit convergence order")
        # two sizes, loose bound: rate must exceed 5 (measured ~7)
        rmss = Float64[]
        for n_side in (20, 40)
            rng = MersenneTwister(3)
            pts = jittered_pts(n_side, 0.35, rng)
            vals = [g2(pts[1,i], pts[2,i]) for i in 1:size(pts, 2)]
            itp = convolution_interpolation(pts, vals; kernel = :b5)
            ev = range(0.1, 0.9, length = 31)
            push!(rmss, sqrt(mean((itp(x, y) - g2(x, y))^2 for x in ev, y in ev)))
        end
        rate = log(rmss[1] / rmss[2]) / log(2)
        @test rate > 5.0
    end

    @testset "derivatives vs analytic (2D, :b5)" begin
        println("    - Testing scattered fit derivatives")
        rng = MersenneTwister(3)
        pts = jittered_pts(40, 0.35, rng)
        vals = [g2(pts[1,i], pts[2,i]) for i in 1:size(pts, 2)]
        knots, gvals, _ = fit_scattered(pts, vals; kernel = :b5)
        ev = range(0.1, 0.9, length = 31)
        d1 = FastConvolutionInterpolation(knots, gvals; kernel = :b5, bc = :poly,
                                        derivative = (1, 0))
        @test sqrt(mean((d1(x, y) - g2_dx(x, y))^2 for x in ev, y in ev)) < 1e-4
        d2 = FastConvolutionInterpolation(knots, gvals; kernel = :b5, bc = :poly,
                                        derivative = (2, 0))
        @test sqrt(mean((d2(x, y) - g2_dxx(x, y))^2 for x in ev, y in ev)) < 1e-2
    end

    @testset "box integrals vs analytic (2D, :b5)" begin
        println("    - Testing scattered fit boxed integrals")
        rng = MersenneTwister(3)
        pts = jittered_pts(57, 0.35, rng)
        vals = [g2(pts[1,i], pts[2,i]) for i in 1:size(pts, 2)]
        knots, gvals, _ = fit_scattered(pts, vals; kernel = :b5)
        Ixy = FastConvolutionInterpolation(knots, gvals; kernel = :b5, bc = :poly,
                                        derivative = (-1, -1))
        for (a, b, c, d) in ((0.1, 0.9, 0.1, 0.9), (0.2, 0.5, 0.3, 0.8))
            num = Ixy(b, d) - Ixy(a, d) - Ixy(b, c) + Ixy(a, c)
            @test abs(num - g2_int(a, b, c, d)) < 1e-9
        end
    end

    @testset "solver agreement oracle (nullspace as source of truth)" begin
        println("    - Testing scattered solver agreement vs direct nullspace solve")
        rng = MersenneTwister(5)
        xs = sort(rand(rng, 60))
        pts = reshape(xs, 1, :)
        vals = g1.(xs)
        _, v_ns, _ = fit_scattered(pts, vals; solver = :nullspace, gridsize = 96)
        for s in (:cholesky, :qr)
            _, v_s, _ = fit_scattered(pts, vals; solver = s, gridsize = 96)
            @test maximum(abs, v_ns .- v_s) < 1e-4
        end
    end

    @testset ":smooth fits noisy data" begin
        println("    - Testing scattered fit with smoothing (:smooth)")
        rng = MersenneTwister(9)
        pts = jittered_pts(40, 0.35, rng)
        clean = [g2(pts[1,i], pts[2,i]) for i in 1:size(pts, 2)]
        sigma = 0.05
        noisy = clean .+ sigma .* randn(rng, length(clean))
        ev = range(0.1, 0.9, length = 31)

        # noise reduction and noise estimate
        knots, gvals, info = fit_scattered(pts, noisy; kernel = :b5, mode = :smooth)
        itp = FastConvolutionInterpolation(knots, gvals; kernel = :b5, bc = :poly)
        rms = sqrt(mean((itp(x, y) - g2(x, y))^2 for x in ev, y in ev))
        @test rms < 0.5 * sigma                      # must beat raw noise decisively
        @test keys(info) == (:lambda, :noise)
        @test info.lambda > 0
        @test abs(info.noise - sigma) < 0.2 * sigma  # estimated noise level

        # the returned lambda reproduces the fit on the same grid
        _, gvals2, info2 = fit_scattered(pts, noisy; kernel = :b5, mode = :smooth,
                                         lambda = info.lambda, gridsize = length(knots[1]))
        @test maximum(abs, gvals2 .- gvals) < 1e-10
        @test info2.lambda ≈ info.lambda

        # a given gridsize is used as is
        knots3, gvals3, _ = fit_scattered(pts, noisy; kernel = :b5, mode = :smooth, gridsize = 20)
        @test length.(knots3) == (20, 20)
        @test size(gvals3) == (20, 20)

        # noise-free data: the fit approaches interpolation
        knots4, gvals4, info4 = fit_scattered(pts, clean; kernel = :b5, mode = :smooth)
        itp4 = FastConvolutionInterpolation(knots4, gvals4; kernel = :b5, bc = :poly)
        rms4 = sqrt(mean((itp4(x, y) - g2(x, y))^2 for x in ev, y in ev))
        @test rms4 < 1e-3
        @test info4.noise < 1e-3

        # the convenience constructor uses the same fit
        itp5 = convolution_interpolation(pts, noisy; kernel = :b5, mode = :smooth)
        @test maximum(abs(itp5(x, y) - itp(x, y)) for x in ev, y in ev) < 1e-10

        # 1D: noise reduction with grid refinement
        xs = sort(rand(rng, 400))
        clean1 = g1.(xs)
        noisy1 = clean1 .+ sigma .* randn(rng, length(xs))
        knots1, gvals1, info1 = fit_scattered(reshape(xs, 1, :), noisy1; mode = :smooth)
        itp1 = FastConvolutionInterpolation(knots1, gvals1; kernel = :b5, bc = :poly)
        ev1 = range(0.1, 0.9, length = 201)
        rms1 = sqrt(mean((itp1(x) - g1(x))^2 for x in ev1))
        @test rms1 < 0.5 * sigma
        @test abs(info1.noise - sigma) < 0.2 * sigma

        # element type: Float32 data give a Float32 grid
        _, gvals32, _ = fit_scattered(Float32.(pts), Float32.(noisy); kernel = :b5, mode = :smooth)
        @test eltype(gvals32) == Float32

        # errors: the removed :lsq mode and invalid smoothing parameters
        @test_throws ArgumentError fit_scattered(pts, noisy; mode = :lsq)
        @test_throws ArgumentError fit_scattered(pts, noisy; mode = :smooth, lambda = -1.0)
        @test_throws ArgumentError fit_scattered(pts, noisy; mode = :smooth, lambda = 0)
        @test_throws ArgumentError fit_scattered(pts, noisy; mode = :smooth, lambda = :auto)
    end

    @testset "per-dimension kernels" begin
        println("    - Testing scattered fit per-dimension kernels")
        rng = MersenneTwister(13)
        pts = jittered_pts(25, 0.35, rng)
        vals = [g2(pts[1,i], pts[2,i]) for i in 1:size(pts, 2)]
        itp = convolution_interpolation(pts, vals; kernel = (:b7, :b5))
        res = maximum(abs(itp(pts[1,i], pts[2,i]) - vals[i]) for i in 1:size(pts, 2))
        @test res < 1e-9
    end

    @testset "eltypes" begin
        println("    - Testing scattered fit element types")
        rng = MersenneTwister(4)
        pts = jittered_pts(10, 0.35, rng)
        for T in (Float32, BigFloat)
            vals = T[g2(pts[1,i], pts[2,i]) for i in 1:size(pts, 2)]
            knots, gvals, viol = fit_scattered(T.(pts), vals; kernel = :b5)
            @test eltype(gvals) == T
            @test Float64(viol) < 1e-8
        end
    end

    @testset "non-unit domain" begin
        println("    - Testing scattered fit non-unit domain")
        rng = MersenneTwister(9)
        pts = jittered_pts(30, 0.35, rng)
        pts[1, :] .= 2 .* pts[1, :]              # [0, 2]
        pts[2, :] .= 2 .* pts[2, :] .- 1         # [-1, 1]
        gg(x, y) = sin(2x) * cos(1.5y)
        vals = [gg(pts[1,i], pts[2,i]) for i in 1:size(pts, 2)]
        itp = convolution_interpolation(pts, vals; kernel = :b5)
        res = maximum(abs(itp(pts[1,i], pts[2,i]) - vals[i]) for i in 1:size(pts, 2))
        @test res < 1e-9
    end

    @testset "guardrails" begin
        println("    - Testing scattered fit errors thrown correctly")
        rng = MersenneTwister(1)
        pts = rand(rng, 2, 100)
        vals = vec(sum(pts; dims = 1))
        # transposed input: fewer points than dimensions
        @test_throws ArgumentError fit_scattered(collect(pts'), vec(vals[1:2]))
        # length mismatch
        @test_throws ArgumentError fit_scattered(pts, vals[1:50])
        # unsupported kernel (no polynomial BC table)
        @test_throws ArgumentError fit_scattered(pts, vals; kernel = :gaussian)
        # bad mode / solver symbols
        @test_throws ArgumentError fit_scattered(pts, vals; mode = :magic)
        @test_throws ArgumentError fit_scattered(pts, vals; solver = :abacus)
        # nullspace size cap is a hard error, not a hang
        big = rand(rng, 2, 4000)
        bigv = vec(sum(big; dims = 1))
        @test_throws Exception fit_scattered(big, bigv; solver = :nullspace)
    end

    @testset "duplicates and warnings" begin
        println("    - Testing scattered fit duplicates and warnings")
        rng = MersenneTwister(2)
        pts = rand(rng, 2, 80)
        pts = hcat(pts, pts[:, 1])               # exact duplicate, same value
        vals = [g2(pts[1,i], pts[2,i]) for i in 1:size(pts, 2)]
        knots, gvals, viol = fit_scattered(pts, vals; kernel = :b5)
        @test viol < 1e-8                        # consistent duplicate is harmless
        # conflicting duplicate: provably infeasible, warning fires deterministically
        pts2 = hcat(pts, pts[:, 1])
        vals2 = vcat(vals, vals[1] + 0.1)
        @test_logs (:warn, r"constraint violation") match_mode = :any begin
            _, _, viol = fit_scattered(pts2, vals2; kernel = :b5)
            @test viol > 0.01            # ~|Δv|/2, the LSQ compromise
        end
    end

end # fit_scattered