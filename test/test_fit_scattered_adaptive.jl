println("\n" * "-"^60)
println("Testing fit_scattered on adaptive and user-supplied grids...")
println("-"^60)

# clustered 1D samples and a boundary-layer function
layer(x) = exp(-20x)
clustered_1d(n, rng) = sort(rand(rng, n) .^ 3)                       # dense near 0
jittered_1d(n, rng)  = reshape(sort((collect(1:n) .- 0.5 .+ 0.35 .* (2 .* rand(rng, n) .- 1)) ./ n), 1, :)
gsmooth(x) = sin(6x) + 0.3cos(11x)                                   # smooth test signal

@testset "fit_scattered adaptive grids" begin

    @testset "axis builder" begin
        println("    - Testing the adaptive axis builder")
        rng = MersenneTwister(5)
        c = clustered_1d(300, rng)
        lo, hi = 0.0, 1.0
        for ratio in (1.5, 2.0, 4.0)
            k = ConvolutionInterpolations._fit_axis_adaptive(c, 50, lo, hi; ratio)
            h = diff(k)
            @test length(k) == 50
            @test k[1] == lo && k[end] == hi
            @test all(h .> 0)
            @test maximum(h[2:end] ./ h[1:end-1]) <= ratio * (1 + 1e-12)
            @test maximum(h[1:end-1] ./ h[2:end]) <= ratio * (1 + 1e-12)
        end
        # alpha = 0 is the uniform grid
        k0 = ConvolutionInterpolations._fit_axis_adaptive(c, 20, lo, hi; alpha = 0.0)
        @test k0 ≈ collect(range(lo, hi, length = 20))
        # the quantile grid is denser where the points are
        k1 = ConvolutionInterpolations._fit_axis_adaptive(c, 20, lo, hi; alpha = 1.0)
        @test diff(k1)[1] < diff(k1)[end]
        # duplicated coordinates do not break the builder
        kd = ConvolutionInterpolations._fit_axis_adaptive(vcat(fill(0.2, 50), c), 30, lo, hi)
        @test all(diff(kd) .> 0)
    end

    @testset "user axes equal to the uniform grid agree with grid = :uniform" begin
        println("    - Testing user axes against the uniform path")
        rng = MersenneTwister(7)
        pts = jittered_1d(200, rng)
        f = gsmooth.(vec(pts))
        ku, vu, viol_u = fit_scattered(pts, f; mode = :exact, kernel = :b5)
        ax = (collect(ku[1]),)                                          # the same knots as a Vector
        ka, va, viol_a = fit_scattered(pts, f; mode = :exact, kernel = :b5, grid = ax)
        @test viol_u < 1e-9
        @test viol_a < 1e-9
        @test ka[1] == ax[1]
        itp_u = convolution_interpolation(ku, vu; kernel = :b5, bc = :poly)
        itp_a = convolution_interpolation(ka, va; kernel = :b5)
        ev = range(0.2, 0.8, length = 61)                               # away from the ghost regions
        @test maximum(abs(itp_u(x) - itp_a(x)) for x in ev) < 1e-6
    end

    @testset ":exact interpolates clustered data (1D)" begin
        println("    - Testing :exact on an adaptive grid, 1D")
        rng = MersenneTwister(3)
        x = clustered_1d(60, rng)
        pts = reshape(x, 1, :)
        f = layer.(x)
        for kern in (:b5, :n3)                                          # :b5 is the adaptive :exact default
            ka, va, viol = fit_scattered(pts, f; mode = :exact, grid = :adaptive, kernel = kern)
            @test viol < 1e-9
            itp = convolution_interpolation(ka, va; kernel = kern)
            @test maximum(abs(itp(x[i]) - f[i]) for i in eachindex(x)) < 1e-9
            # cardinality: the fitted values are the interpolant's values at the knots
            @test maximum(abs.(itp.(ka[1]) .- va)) < 1e-10
        end
        # the one-call wrapper builds the same interpolant as the manual path
        ka, va, _ = fit_scattered(pts, f; mode = :exact, grid = :adaptive)
        itp = convolution_interpolation(ka, va; kernel = :b5)
        itp_w = convolution_interpolation(pts, f; mode = :exact, grid = :adaptive)
        ev = range(x[1], x[end], length = 201)
        @test maximum(abs(itp_w(t) - itp(t)) for t in ev) < 1e-12
    end

    @testset ":exact interpolates clustered data (2D)" begin
        println("    - Testing :exact on an adaptive grid, 2D")
        rng = MersenneTwister(11)
        n = 400
        xs = rand(rng, n) .^ 3
        u  = 2 .* rand(rng, n) .- 1
        ys = 0.5 .+ 0.5 .* sign.(u) .* abs.(u) .^ 3
        pts = permutedims(hcat(xs, ys))
        f = [layer(pts[1, i]) * (1 + tanh(40 * (pts[2, i] - 0.5))) for i in 1:n]
        # the 2D default oversample (2.5) separates random points that share a cell at 1.6
        ka, va, viol = fit_scattered(pts, f; mode = :exact, grid = :adaptive)
        @test viol < 1e-9
        itp = convolution_interpolation(ka, va; kernel = :b5)
        @test maximum(abs(itp(pts[1, i], pts[2, i]) - f[i]) for i in 1:n) < 1e-8
        @test maximum(abs.([itp(x, y) for x in ka[1], y in ka[2]] .- va)) < 1e-10
    end

    @testset ":smooth on an adaptive grid resolves a boundary layer" begin
        println("    - Testing :smooth adaptive vs uniform at a fixed knot budget")
        rng = MersenneTwister(4)
        x = clustered_1d(2000, rng)
        pts = reshape(x, 1, :)
        noisy = layer.(x) .+ 1e-3 .* randn(rng, 2000)
        ev = range(x[1], x[end], length = 1001)
        # uniform grid with its :smooth default kernel (:a4), the comparison the docs make
        ku, vu, iu = fit_scattered(pts, noisy; mode = :smooth, grid = :uniform, gridsize = 40)
        su = convolution_interpolation(ku, vu; kernel = :a4, bc = :poly)
        eu = maximum(abs(su(t) - layer(t)) for t in ev)
        for kern in (:n3, :b5)                                          # :n3 is the adaptive :smooth default
            ka, va, ia = fit_scattered(pts, noisy; mode = :smooth, grid = :adaptive, gridsize = 40, kernel = kern)
            sa = convolution_interpolation(ka, va; kernel = kern)
            ea = maximum(abs(sa(t) - layer(t)) for t in ev)
            @test ea < eu / 2                                           # measured ~2.5x for :n3
            @test 0.5e-3 < ia.noise < 2e-3                              # noise estimate in range
            @test maximum(abs.(sa.(ka[1]) .- va)) < 1e-10               # cardinality: ghosts match the constructor
        end
        # the default kernel on an adaptive grid is :n3 for :smooth
        itp = convolution_interpolation(pts, noisy; grid = :adaptive, gridsize = 40)
        @test occursin("n3", sprint(show, itp))
    end

    @testset "guardrails" begin
        println("    - Testing adaptive-grid argument errors")
        rng = MersenneTwister(9)
        pts = rand(rng, 2, 100)
        f = vec(sum(pts; dims = 1))
        @test_throws ArgumentError fit_scattered(pts, f; grid = :foo)
        @test_throws ArgumentError fit_scattered(pts, f; grid = :adaptive, kernel = :a4)
        @test_throws ArgumentError fit_scattered(pts, f; grid = :adaptive, kernel = :a3)   # uniform-only name, even though it equals :n3 there
        @test fit_scattered(pts, f; grid = :adaptive, kernel = :n3) isa Tuple                 # :n3 is accepted
        ax = (collect(range(0.0, 1.0, length = 12)), collect(range(0.0, 1.0, length = 12)))
        @test_throws ArgumentError fit_scattered(pts, f; grid = ax, gridsize = 12)
        @test_throws ArgumentError fit_scattered(pts, f; grid = (ax[1],))
        short = (collect(range(0.1, 0.9, length = 12)), ax[2])          # does not cover the data
        @test_throws ArgumentError fit_scattered(pts, f; grid = short)
        unsorted = (reverse(ax[1]), ax[2])
        @test_throws ArgumentError fit_scattered(pts, f; grid = unsorted)
        @test_throws ArgumentError fit_scattered(pts, f; grid = :adaptive, ratio = 0.5)
        @test_throws ArgumentError fit_scattered(pts, f; grid = :adaptive, alpha = 1.5)
    end
end