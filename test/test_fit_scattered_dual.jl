println("\n" * "-"^60)
println("Testing the dual solver of the exact scattered fit...")
println("-"^60)

# test functions of any dimension, called with a point (a column of the points matrix)
dual_runge(p) = 1 / (1 + 25 * sum(abs2, p))
dual_wave(p)  = prod(sin(2 * x) for x in p)

# scattered points in [-1, 1]^D: Halton points, the same on every Julia version;
# `skip` selects a set that shares no point with the default one
dual_points(D, Np; skip = 0) = 2 .* halton_points(D, Np; skip = skip) .- 1

# the interpolant of a fitted grid, with the kernel the exact fit chose by default
function dual_interpolant(knots, vals)
    kernel = ConvolutionInterpolations._default_scattered_kernel(length(knots), :exact,
                                                                 false)
    return convolution_interpolation(knots, vals; kernel = kernel, bc = :poly)
end

# largest error of a fitted grid against `fun` at scattered test points of the data's box;
# `inner` restricts the test points to the central 80 % of every axis
function dual_error(knots, vals, fun; inner = false, n_test = 2000)
    D = length(knots)
    lo = [first(k) for k in knots]
    hi = [last(k) for k in knots]
    # apart from any data set
    tests = lo .+ (hi .- lo) .* halton_points(D, n_test; skip = 100_000)
    itp = dual_interpolant(knots, vals)
    worst = 0.0
    for p in eachcol(tests)
        if inner && !all(abs.(p .- (lo .+ hi) ./ 2) .<= 0.4 .* (hi .- lo))
            continue
        end
        worst = max(worst, abs(itp(p...) - fun(p)))
    end
    return worst
end

@testset "fit_scattered: dual solver" begin

    @testset "agrees with :cholesky in the interior (2D)" begin
        println("    - Testing the dual solver against :cholesky in the interior (2D)")
        pts = dual_points(2, 2000)
        f = [dual_runge(p) for p in eachcol(pts)]
        knots_c, vals_c, viol_c = fit_scattered(pts, f; mode = :exact, solver = :cholesky)
        # on this fine grid the eigenvalue shift is what keeps the data reproduced: no
        # warning
        knots_d, vals_d, viol_d = @test_logs fit_scattered(pts, f; mode = :exact,
                                                           solver = :dual)
        @test knots_d == knots_c                                # the same grid
        err_c = dual_error(knots_c, vals_c, dual_runge; inner = true)
        err_d = dual_error(knots_d, vals_d, dual_runge; inner = true)
        println("      interior error: :cholesky $(round(err_c, sigdigits = 2)), :dual " *
                "$(round(err_d, sigdigits = 2)); :dual misfit " *
                "$(round(viol_d, sigdigits = 2))")
        @test err_c < 1e-4
        @test err_d < 3 * err_c
        @test viol_d < 1e-9
    end

    @testset "reproduces the data (2D, 3D, 4D)" begin
        println("    - Testing that the dual solver reproduces the data (2D, 3D, 4D)")
        bound = 1e-9
        for (D, Np) in ((2, 300), (3, 500), (4, 400))
            pts = dual_points(D, Np)
            f = [dual_wave(p) for p in eachcol(pts)]
            knots, vals, viol = fit_scattered(pts, f; mode = :exact, solver = :dual)
            itp = dual_interpolant(knots, vals)
            at_data = maximum(abs(itp(p...) - f[i]) for (i, p) in enumerate(eachcol(pts)))
            println("      $(D)D, $Np points: misfit $(round(viol, sigdigits = 2)), " *
                    "interpolant at the data $(round(at_data, sigdigits = 2))")
            @test size(vals) == length.(knots)
            @test viol < bound
            @test at_data < 10 * bound
        end
    end

    @testset "default solver" begin
        println("    - Testing the default solver of the exact fit")
        rule = ConvolutionInterpolations._default_scattered_solver
        @test rule(1, false, Float64, 1000, 5) === :cholesky
        @test rule(2, false, Float64, 1000, 5) === :cholesky
        @test rule(3, false, Float64, 1000, 5) === :dual
        @test rule(4, false, Float64, 1000, 5) === :dual
        @test rule(5, false, Float64, 1000, 5) === :dual
        # 70 tail polynomials in 4D: one point too few
        @test rule(4, false, Float64, 70, 5) === :cholesky
        @test rule(4, false, Float64, 71, 5) === :dual
        @test rule(4, true, Float64, 1000, 5) === :cholesky     # nonuniform grid
        @test rule(4, false, BigFloat, 1000, 5) === :cholesky   # not Float64

        # 2D: the default is the Cholesky fit
        pts = dual_points(2, 300)
        f = [dual_wave(p) for p in eachcol(pts)]
        _, vals_auto, _ = fit_scattered(pts, f; mode = :exact)
        _, vals_c, _ = fit_scattered(pts, f; mode = :exact, solver = :cholesky)
        @test vals_auto ≈ vals_c rtol = 1e-10

        # 3D: the default is the dual fit, which finds the grid of the Cholesky fit
        pts = dual_points(3, 500)
        f = [dual_wave(p) for p in eachcol(pts)]
        _, vals_auto, _ = fit_scattered(pts, f; mode = :exact)
        _, vals_d, _ = fit_scattered(pts, f; mode = :exact, solver = :dual)
        _, vals_c, _ = fit_scattered(pts, f; mode = :exact, solver = :cholesky)
        @test vals_auto ≈ vals_d rtol = 1e-10
        @test vals_auto ≈ vals_c rtol = 1e-6

        # 4D: the default is the dual fit
        pts = dual_points(4, 400)
        f = [dual_wave(p) for p in eachcol(pts)]
        _, vals_auto, _ = fit_scattered(pts, f; mode = :exact)
        _, vals_d, _ = fit_scattered(pts, f; mode = :exact, solver = :dual)
        @test vals_auto ≈ vals_d rtol = 1e-10
    end

    @testset "one-call interpolant in 4D" begin
        println("    - Testing convolution_interpolation(points, values) in 4D")
        pts = dual_points(4, 400)
        f = [dual_runge(p) for p in eachcol(pts)]
        itp_direct = convolution_interpolation(pts, f; mode = :exact)
        knots, vals, _ = fit_scattered(pts, f; mode = :exact)
        itp_manual = dual_interpolant(knots, vals)
        probes = 0.8 .* dual_points(4, 20; skip = 10_000)       # well inside the data's box
        @test all(itp_direct(p...) ≈ itp_manual(p...) for p in eachcol(probes))
    end

    @testset "polynomials below degree q are reproduced (3D)" begin
        println("    - Testing that the dual solver reproduces a cubic polynomial (3D)")
        cubic(p) = 1 + p[1] - 2 * p[2] + 0.5 * p[3] + p[1] * p[2] - p[3]^2 + p[1]^2 * p[3] -
                   p[2]^3
        pts = dual_points(3, 300)
        f = [cubic(p) for p in eachcol(pts)]
        knots, vals, viol = fit_scattered(pts, f; mode = :exact, solver = :dual)
        err = dual_error(knots, vals, cubic; n_test = 500)
        println("      error away from the data $(round(err, sigdigits = 2)), " *
                "misfit $(round(viol, sigdigits = 2))")
        @test err < 1e-8
    end

    @testset "input the dual solver rejects" begin
        println("    - Testing the input the dual solver rejects")
        # fewer points than tail polynomials (70 in 4D)
        pts = dual_points(4, 50)
        f = [dual_runge(p) for p in eachcol(pts)]
        @test_throws ArgumentError fit_scattered(pts, f; mode = :exact, solver = :dual)
        # nonuniform grid
        pts = dual_points(2, 200)
        f = [dual_runge(p) for p in eachcol(pts)]
        @test_throws ArgumentError fit_scattered(pts, f; mode = :exact, solver = :dual,
                                                 grid = :adaptive)
        # not Float64
        @test_throws ArgumentError fit_scattered(BigFloat.(pts),
                                                 BigFloat.(f); mode = :exact,
                                                 solver = :dual)
        # an unknown solver
        @test_throws ArgumentError fit_scattered(pts, f; mode = :exact, solver = :abacus)
    end

    @testset "duplicated points are reported" begin
        println("    - Testing the dual solver on duplicated points with conflicting " *
                "values")
        pts = dual_points(2, 200)
        f = [dual_runge(p) for p in eachcol(pts)]
        pts[:, 2] .= pts[:, 1]                                  # the same point twice,
        f[2] = f[1] + 1.0                                       # with two different values
        knots, vals, viol =
            @test_logs (:warn, r"solver = :dual") match_mode = :any fit_scattered(
                pts, f; mode = :exact, solver = :dual)
        println("      reported misfit $(round(viol, sigdigits = 2))")
        @test viol > 1e-3
    end
end
