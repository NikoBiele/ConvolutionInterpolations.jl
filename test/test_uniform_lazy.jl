println("\n" * "-"^60)
println("Testing uniform lazy mode...")
println("-"^60)

# Largest |lazy patch value − eager coefficient| over the stencil of the boundary cell holding the
# point P: the patch `_lazy_fill_patch` builds must hold eager's stored coefficients exactly, the
# ghosts as well as the copied data. Returns -1.0 for an interior cell, which builds no patch.
function lazy_patch_mismatch(eager, lazy, P)
    CI = ConvolutionInterpolations
    dims = length(P)                                      # number of dimensions
    eqs = lazy.eqs                                        # stencil half-widths
    # cell exactly as the lazy evaluator computes it
    pos = ntuple(d -> clamp(floor(Int, (P[d] - lazy.knots[d][1]) / lazy.h[d] + 1), 1,
                            length(lazy.knots[d]) - 1), dims)
    any(d -> CI.is_boundary_stencil(pos[d], lazy.domain_size[d], eqs[d]), 1:dims) || return -1.0
    buf, lo, stride = CI._lazy_fill_patch(lazy, pos)
    worst = 0.0
    for K in CartesianIndices(ntuple(d -> 1:2eqs[d], dims))
        J = ntuple(d -> pos[d] + K[d] - eqs[d], dims)             # data index of stencil entry K
        p = 1 + sum(ntuple(d -> (J[d] - lo[d]) * stride[d], dims)) # its position in the patch
        E = ntuple(d -> J[d] + eqs[d] - 1, dims)                  # its index in eager's array
        worst = max(worst, abs(buf[p] - eager.coefs[E...]))
    end
    return worst
end

@testset "Lazy mode — uniform" begin

    @testset "1D lazy matches eager" begin
        println("    - 1D lazy matches eager...")
        for kernel in (:a3, :b5)
            xs = range(0.0, 2π, length=50)
            vs = sin.(xs)
            itp_e = convolution_interpolation(xs, vs; kernel=kernel, lazy=false)
            itp_l = convolution_interpolation(xs, vs; kernel=kernel, lazy=true)
            for x in range(0.01, 2π-0.01, length=30)
                @test itp_e(x) ≈ itp_l(x) atol=1e-10
            end
        end
    end

    @testset "2D lazy matches eager" begin
        println("    - 2D lazy matches eager...")
        for kernel in (:a3, :b5)
            xs = range(0.0, 2π, length=40)
            vs = [sin(x)*cos(y) for x in xs, y in xs]
            itp_e = convolution_interpolation((xs, xs), vs; kernel=kernel, lazy=false)
            itp_l = convolution_interpolation((xs, xs), vs; kernel=kernel, lazy=true)
            for x in [0.02, 1.0, 2.5, 5.5, 2π-0.02]
                for y in [0.02, 1.5, 4.0, 2π-0.02]
                    @test itp_e(x, y) ≈ itp_l(x, y) atol=1e-10
                end
            end
        end
    end

    @testset "3D lazy matches eager" begin
        println("    - 3D lazy matches eager...")
        for kernel in (:a3, :b5)
            xs = range(0.0, 2π, length=30)
            vs = [sin(x)*cos(y)*sin(z) for x in xs, y in xs, z in xs]
            itp_e = convolution_interpolation((xs, xs, xs), vs; kernel=kernel, lazy=false)
            itp_l = convolution_interpolation((xs, xs, xs), vs; kernel=kernel, lazy=true)
            for (x, y, z) in [(0.02, 0.5, 1.0), (1.5, 2.0, 3.0), (5.5, 5.0, 4.5), (2π-0.2, 2π-0.3, 2π-0.04)]
                @test itp_e(x, y, z) ≈ itp_l(x, y, z) atol=1e-10
            end
        end
    end

    @testset "4D lazy matches eager" begin
        println("    - 4D lazy matches eager...")
        for kernel in (:a3, :b5)
            xs = range(0.0, 2π, length=30)
            vs = [sin(x)*cos(y)*sin(z)*cos(t) for x in xs, y in xs, z in xs, t in xs]
            itp_e = convolution_interpolation((xs, xs, xs, xs), vs; kernel=kernel, lazy=false)
            itp_l = convolution_interpolation((xs, xs, xs, xs), vs; kernel=kernel, lazy=true)
            # interior, and near the boundaries: lazy computes the same ghosts as eager
            for (x, y, z, t) in [(1.0, 1.5, 2.0, 2.5), (2.0, 2.5, 3.0, 3.5), (3.0, 3.5, 4.0, 4.5),
                                 (0.1, 0.1, 0.1, 0.1), (2π-0.1, 2π-0.1, 2π-0.1, 2π-0.1)]
                @test itp_e(x, y, z, t) ≈ itp_l(x, y, z, t) atol=1e-10
            end
        end
    end

    @testset "Lazy ghosts equal eager ghosts exactly" begin
        println("    - Lazy ghosts equal eager ghosts exactly...")
        rng = Xoshiro(7)                                  # reproducible roughness
        kx = range(0.0, 1.0, length=30)                   # regular axis
        ks = range(0.0, 1.0, length=9)                    # short axis: eager uses the linear rule
        smooth = [exp(x) * sin(3x + 1) * cos(2y) for x in kx, y in kx]
        rough = copy(smooth)                              # :detect rejects some lines, accepts others
        rough[1:4, 1:2:end] .+= 0.3 .* randn(rng, 4, 15)
        short = [exp(x) * sin(3x + 1) * cos(2y) for x in kx, y in ks]
        cases = [((kx, kx), smooth, :detect), ((kx, kx), rough, :detect),
                 ((kx, kx), smooth, :poly), ((kx, ks), short, :detect)]
        for (knots, data, bc) in cases, kernel in (:a3, :b7)
            eager = convolution_interpolation(knots, data; kernel=kernel, bc=bc, lazy=false).itp
            lazy  = convolution_interpolation(knots, data; kernel=kernel, bc=bc, lazy=true).itp
            pts = ntuple(d -> range(first(knots[d]), last(knots[d]), length=41), 2)
            worst = maximum(lazy_patch_mismatch(eager, lazy, P) for P in Iterators.product(pts...))
            @test worst == 0.0
        end
        # 3D: edges and corners, where ghosts are formed from the ghosts of lower axes
        k3 = range(0.0, 1.0, length=12)
        data3 = [exp(x) * sin(3x + 1) * cos(2y) * (1 + z^2) for x in k3, y in k3, z in k3]
        eager3 = convolution_interpolation((k3, k3, k3), data3; kernel=:a3, lazy=false).itp
        lazy3  = convolution_interpolation((k3, k3, k3), data3; kernel=:a3, lazy=true).itp
        p3 = range(0.0, 1.0, length=13)
        worst3 = maximum(lazy_patch_mismatch(eager3, lazy3, P) for P in Iterators.product(p3, p3, p3))
        @test worst3 == 0.0
    end

    @testset "Lazy default depends on eager's memory and construction cost" begin
        println("    - Lazy default depends on eager's memory and construction cost...")
        CI = ConvolutionInterpolations
        max_total = CI.LAZY_MAX_EAGER_COEFFICIENTS          # memory limit: eager's total coefficients
        ng = CI.get_equations_for_degree(:a3) - 1           # eager's ghosts per side for :a3
        # memory limit exactly (1D has only 2ng ghosts): at the limit eager, one more lazy
        @test CI._default_lazy((max_total - 2ng,), (:a3,), (0,)) == false
        @test CI._default_lazy((max_total - 2ng + 1,), (:a3,), (0,)) == true
        # memory limit alone: 3D :b7, ghosts below their limit on both sides
        @test CI._default_lazy((500, 500, 500), (:b7, :b7, :b7), (0, 0, 0)) == false
        @test CI._default_lazy((600, 600, 600), (:b7, :b7, :b7), (0, 0, 0)) == true
        # construction limit alone: 4D :a4, totals below the memory limit on both sides
        @test CI._default_lazy(ntuple(_ -> 99, 4), ntuple(_ -> :a4, 4), ntuple(_ -> 0, 4)) == false
        @test CI._default_lazy(ntuple(_ -> 100, 4), ntuple(_ -> :a4, 4), ntuple(_ -> 0, 4)) == true
        # 5D :a4: small eager, large lazy
        @test CI._default_lazy(ntuple(_ -> 12, 5), ntuple(_ -> :a4, 5), ntuple(_ -> 0, 5)) == false
        @test CI._default_lazy(ntuple(_ -> 30, 5), ntuple(_ -> :a4, 5), ntuple(_ -> 0, 5)) == true
        # antiderivatives always build eagerly, however large
        @test CI._default_lazy((2max_total,), (:a3,), (-1,)) == false
        # small data builds eagerly by default in any dimension, with and without keywords;
        # an explicit choice always wins
        k = range(0.0, 1.0, length=8)
        k5 = ntuple(_ -> k, 5)
        v5 = [sum(x) for x in Iterators.product(k5...)]
        @test convolution_interpolation(k5, v5).itp.lazy isa Val{false}
        @test convolution_interpolation(k5, v5; kernel=:a3).itp.lazy isa Val{false}
        @test convolution_interpolation(k5, v5; kernel=:a3, lazy=true).itp.lazy isa Val{true}
        @test FastConvolutionInterpolation(k5, v5; kernel=:a3).lazy isa Val{false}
        @test FastConvolutionInterpolation(k5, v5; kernel=:a3, lazy=true).lazy isa Val{true}
    end

    @testset "Lazy evaluation is thread-safe" begin
        println("    - Lazy evaluation is thread-safe...")
        k = range(0.0, 1.0, length=30)
        z = [exp(x) * sin(3x + 1) * cos(2y) for x in k, y in k]
        lazy = convolution_interpolation((k, k), z; kernel=:b7, lazy=true)
        rng = Xoshiro(3)
        px = 0.2 .* rand(rng, 20_000)                     # x in the first cells: boundary-heavy
        py = rand(rng, 20_000)
        serial = [lazy(px[m], py[m]) for m in eachindex(px)]
        threaded = similar(serial)
        Threads.@threads for m in eachindex(px)
            threaded[m] = lazy(px[m], py[m])
        end
        @test threaded == serial                          # meaningful with more than one thread
    end

    @testset "Lazy construction is fast" begin
        println("    - Lazy construction is fast...")
        xs_3d = range(0.0, 1.0, length=50)
        vs_3d = rand(50, 50, 50)
        convolution_interpolation((xs_3d, xs_3d, xs_3d), vs_3d; kernel=:b5, lazy=true)
        t = @elapsed convolution_interpolation((xs_3d, xs_3d, xs_3d), vs_3d; kernel=:b5, lazy=true)
        @test t < 1.0

        xs_4d = range(0.0, 1.0, length=20)
        vs_4d = rand(20, 20, 20, 20)
        convolution_interpolation((xs_4d, xs_4d, xs_4d, xs_4d), vs_4d; kernel=:b5, lazy=true)
        t = @elapsed convolution_interpolation((xs_4d, xs_4d, xs_4d, xs_4d), vs_4d; kernel=:b5, lazy=true)
        @test t < 1.0
    end

    @testset "Lazy stores raw values" begin
        println("    - Lazy stores raw values...")
        xs = range(0.0, 2π, length=20)
        vs = sin.(xs)
        itp_l = convolution_interpolation(xs, vs; kernel=:b5, lazy=true)
        @test itp_l.itp.lazy == Val{true}()
        @test itp_l.itp.coefs === vs
    end

    @testset "Lazy with derivatives" begin
        println("    - Lazy with derivatives...")
        xs = range(0.0, 2π, length=50)
        vs = sin.(xs)
        for d in (1, 2)
            itp_e = convolution_interpolation(xs, vs; kernel=:b5, derivative=d, lazy=false)
            itp_l = convolution_interpolation(xs, vs; kernel=:b5, derivative=d, lazy=true)
            # the whole domain, boundary cells included
            for x in range(0.02, 2π-0.02, length=25)
                @test itp_e(x) ≈ itp_l(x) atol=1e-10
            end
        end
    end

    @testset "Slow lazy functor error guard" begin
        println("    - Slow lazy functor error guard...")
        x_uniform = range(0.0, 2π, length=20)
        y = sin.(x_uniform)
        @test_throws ErrorException convolution_interpolation(x_uniform, y; lazy=true, fast=false)
    end
end