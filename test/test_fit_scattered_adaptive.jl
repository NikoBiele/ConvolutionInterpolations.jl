println("\n" * "-"^60)
println("Testing fit_scattered on adaptive and user-supplied grids...")
println("-"^60)

# Deterministic point sets: the MersenneTwister stream is only stable within a Julia minor
# version, and these tests assert interpolation to 1e-9, which a different draw with a closer
# pair of points turns into a least-squares fit. Low-discrepancy sequences give the same
# clustering with a guaranteed minimum separation on every Julia version.
layer(x) = exp(-20x)                                                 # boundary-layer test signal
gsmooth(x) = sin(6x) + 0.3cos(11x)                                   # smooth test signal
vdc(i, base) = (r = 0.0; f = 1.0 / base; k = i; while k > 0; r += f * (k % base); k ÷= base; f /= base; end; r)
clustered_1d(n) = sort([((i - 0.5) / n)^3 for i in 1:n])            # dense near 0, evenly spaced in the cube root
jittered_1d(n)  = reshape(sort([(i - 0.5 + 0.35 * (2vdc(i, 2) - 1)) / n for i in 1:n]), 1, :)
function clustered_2d(n)                                             # dense near x = 0 and y = 0.5
    xs = [vdc(i, 2)^3 for i in 1:n]
    ys = [(u = 2vdc(i, 3) - 1; 0.5 + 0.5 * sign(u) * abs(u)^3) for i in 1:n]
    return permutedims(hcat(xs, ys))
end

@testset "fit_scattered adaptive grids" begin

    @testset "axis builder" begin
        println("    - Testing the adaptive axis builder")
        c = clustered_1d(300)
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
        pts = jittered_1d(200)
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
        x = clustered_1d(60)
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
        n = 400
        pts = clustered_2d(n)
        f = [layer(pts[1, i]) * (1 + tanh(40 * (pts[2, i] - 0.5))) for i in 1:n]
        ka, va, viol = fit_scattered(pts, f; mode = :exact, grid = :adaptive)
        @test viol < 1e-9
        itp = convolution_interpolation(ka, va; kernel = :b5)
        @test maximum(abs(itp(pts[1, i], pts[2, i]) - f[i]) for i in 1:n) < 1e-8
        @test maximum(abs.([itp(x, y) for x in ka[1], y in ka[2]] .- va)) < 1e-10
    end

    @testset ":smooth on an adaptive grid resolves a boundary layer" begin
        println("    - Testing :smooth adaptive vs uniform at a fixed knot budget")
        rng = MersenneTwister(4)                                        # noise only; the points are deterministic
        x = clustered_1d(2000)
        pts = reshape(x, 1, :)
        sharp(t) = exp(-50t)                                            # a layer the uniform 40-knot grid cannot resolve
        noisy = sharp.(x) .+ 1e-3 .* randn(rng, 2000)
        ev = range(x[1], x[end], length = 1001)
        # uniform grid with its :smooth default kernel (:b5), the comparison the docs make
        ku, vu, iu = fit_scattered(pts, noisy; mode = :smooth, grid = :uniform, gridsize = 40)
        su = convolution_interpolation(ku, vu; kernel = :b5, bc = :poly)
        eu = maximum(abs(su(t) - sharp(t)) for t in ev)
        for kern in (:n3, :b5)                                          # :n3 is the adaptive :smooth default
            ka, va, ia = fit_scattered(pts, noisy; mode = :smooth, grid = :adaptive, gridsize = 40, kernel = kern)
            sa = convolution_interpolation(ka, va; kernel = kern)
            ea = maximum(abs(sa(t) - sharp(t)) for t in ev)
            @test ea < 0.7 * eu                                         # adaptive resolves the layer better; the noise draw moves the ratio
            @test 0.5e-3 < ia.noise < 2e-3                              # noise estimate in range
            @test maximum(abs.(sa.(ka[1]) .- va)) < 1e-10               # cardinality: ghosts match the constructor
        end
        # the default kernel on an adaptive grid is :b5 for :smooth in 1D and 2D
        itp = convolution_interpolation(pts, noisy; grid = :adaptive, gridsize = 40)
        @test occursin("b5", sprint(show, itp))
        # uniformly sampled points: the point density says nothing, the knots follow the
        # signal's curvature into the layer, with fewer knots than the uniform fit
        xu = collect(range(0.0, 1.0, length = 2000))
        pu = reshape(xu, 1, :)
        nu = layer.(xu) .+ 1e-3 .* randn(rng, 2000)
        kuu, vuu, _ = fit_scattered(pu, nu; mode = :smooth, grid = :uniform)
        suu = convolution_interpolation(kuu, vuu; kernel = :b5, bc = :poly)
        kau, vau, iau = fit_scattered(pu, nu; mode = :smooth, grid = :adaptive)
        sau = convolution_interpolation(kau, vau; kernel = :b5)
        euu = maximum(abs(suu(t) - layer(t)) for t in ev)
        eau = maximum(abs(sau(t) - layer(t)) for t in ev)
        @test eau < 2 * euu                                             # on this gently varying signal, within a factor of the uniform fit
        @test length(kau[1]) < length(kuu[1])                           # on about half the knots
        @test 0.5e-3 < iau.noise < 2e-3
        @test maximum(abs.(sau.(kau[1]) .- vau)) < 1e-10                # cardinality: ghosts match the constructor
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