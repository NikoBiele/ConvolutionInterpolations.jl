println("\n" * "-"^60)
println("Testing smoothing kernels...")
println("-"^60)

println("    - Gaussian smoothing kernel")
@testset "Gaussian kernel" begin
    xs = range(0.0, 2π, length=50)
    vs = sin.(xs)

    for B in (0.3, 0.2, 0.1)
        itp = convolution_gaussian(xs, vs, B)
        # should be smooth and reasonably close to sin
        @test abs(itp(1.5) - sin(1.5)) < 0.1
    end

    # 2D
    xs2 = range(0.0, 2π, length=50)
    ys2 = range(0.0, 2π, length=50)
    vs2 = [sin(x)*cos(y) for x in xs2, y in ys2]
    B = 0.1
    itp2 = convolution_gaussian((xs2, ys2), vs2, B)
    @test maximum(abs.(itp2.(xs2, ys2') - vs2)) < 0.1
end

println("    - Convolution_smooth")
@testset "convolution_smooth" begin

    @testset "1D smooth recovers signal" begin
        xs = range(0.0, 2π, length=200)
        signal = sin.(xs)
        noisy  = signal .+ 0.1 .* randn(200)
        smoothed = convolution_smooth(xs, noisy, 0.05)
        # should recover sin well after smoothing
        @test maximum(abs.(smoothed .- signal)) < 0.15
        @test length(smoothed) == length(xs)
        @test eltype(smoothed) == Float64
    end

    @testset "1D smooth with clean signal" begin
        xs = range(0.0, 2π, length=200)
        signal = cos.(xs)
        smoothed = convolution_smooth(xs, signal, 0.05)
        # clean signal should be nearly unchanged for moderate B
        @test maximum(abs.(smoothed .- signal)) < 0.15
    end

    @testset "2D smooth recovers signal" begin
        xs = range(0.0, 2π, length=100)
        ys = range(0.0, 2π, length=100)
        signal   = [sin(x)*cos(y) for x in xs, y in ys]
        noisy    = signal .+ 0.1 .* randn(100, 100)
        smoothed = convolution_smooth((xs, ys), noisy, 0.05)
        @test size(smoothed) == (100, 100)
        @test eltype(smoothed) == Float64
        @test maximum(abs.(smoothed .- signal)) < 0.2
    end

    @testset "3D smooth recovers signal" begin
        xs = range(0.0, 2π, length=100)
        signal   = [sin(x)*cos(y)*sin(z) for x in xs, y in xs, z in xs]
        noisy    = signal .+ 0.1 .* randn(100, 100, 100)
        smoothed = convolution_smooth((xs, xs, xs), noisy, 0.05)
        @test size(smoothed) == (100, 100, 100)
        @test maximum(abs.(smoothed .- signal)) < 0.25
    end

    @testset "smooth preserves array size" begin
        for N in (1, 2, 3)
            n  = 20
            ks = ntuple(_ -> range(0.0, 2π, length=n), N)
            vs = randn(ntuple(_ -> n, N)...)
            sm = convolution_smooth(ks, vs, 0.5)
            @test size(sm) == size(vs)
            @test eltype(sm) == Float64
        end
    end

    @testset "stronger smoothing gives smoother result" begin
        xs = range(0.0, 2π, length=200)
        noisy = sin.(xs) .+ 0.3 .* randn(200)
        sm_weak   = convolution_smooth(xs, noisy, 0.5)
        sm_strong = convolution_smooth(xs, noisy, 0.05)
        # stronger smoothing (smaller B) should have smaller max second difference
        roughness(v) = maximum(abs.(diff(diff(v))))
        @test roughness(sm_strong) < roughness(sm_weak)
    end

end

@testset "constant preservation" begin
    # A smoother whose weights sum to one must reproduce a constant
    # exactly, at the boundary as well as the interior. Regression test
    # for an off-by-one that left the ghost slot adjacent to the data at
    # zero, dropping one kernel tap per boundary (16% of the weight at
    # B=0.1).
    # On-node evaluation is exact: theta(B) normalises the taps at integer
    # offsets by construction.
    xs = range(0.0, 2π, length=161)
    for B in (0.05, 0.1, 0.3), C in (1.0, 100.0)
        itp = convolution_gaussian(xs, fill(C, 161), B)
        @test maximum(abs.(itp.(range(0.0, 2π, length=401)) .- C)) < 1e-12 * C
    end
    @test_throws ErrorException convolution_gaussian(xs, fill(1.0, 161), 0.5)

    # Between nodes a sampled Gaussian is not a partition of unity. By
    # Poisson summation the taps sum to 1 + 2q*cos(2*pi*x) + O(q^2) with
    # q = exp(-pi^2/B), so the ripple is ~4q -- negligible for the
    # recommended B <= 0.1, but ~3% at B = 2.
    for B in (0.05, 0.1, 0.3), C in (1.0, 100.0)
        itp = convolution_gaussian(xs, fill(C, 161), B)
        tol = max(1e-12, 6 * exp(-pi^2 / B)) * C
        @test maximum(abs.(itp.(range(0.0, 2π, length=401)) .- C)) < tol
    end
end

@testset "linear reproduction" begin
    # The ghost fallback fits a line and extrapolates it, and the kernel
    # is symmetric with weights summing to one, so an exactly linear
    # input must come back unchanged everywhere. Fails if the ghost
    # count, the ghost values, or the mean offset are wrong.
    xs = range(0.0, 2π, length=161)
    for B in (0.05, 0.1, 0.5), (a, b) in ((0.0, 1.0), (5.0, -2.0))
        y  = a .+ b .* collect(xs)
        sm = convolution_smooth(xs, y, B)
        @test maximum(abs.(sm .- y)) < 1e-10 * maximum(abs, y)
    end

    xs2 = range(0.0, 2π, length=60)
    y2  = [1.0 + 2x - 3yv for x in xs2, yv in xs2]
    sm2 = convolution_smooth((xs2, xs2), y2, 0.1)
    @test maximum(abs.(sm2 .- y2)) < 1e-10 * maximum(abs, y2)
end

@testset "offset invariance" begin
    # Smoothing commutes with adding a constant. Fails if any ghost value
    # carries a stray mean term.
    xs = range(0.0, 2π, length=161)
    base = sin.(xs) .+ 0.3 .* sin.(4 .* xs)
    for B in (0.05, 0.1, 0.5), C in (1.0, 1e3, 1e6)
        s0 = convolution_smooth(xs, base, B)
        sC = convolution_smooth(xs, base .+ C, B)
        @test maximum(abs.(sC .- C .- s0)) < 1e-9 * max(1.0, C)
    end
end

@testset "boundary symmetry" begin
    # Left and right boundaries must be handled identically: reversing the
    # input must reverse the output.
    xs = range(0.0, 2π, length=101)
    y  = exp.(-collect(xs)) .+ 0.2 .* sin.(3 .* collect(xs))
    for B in (0.05, 0.1, 0.5)
        @test maximum(abs.(reverse(convolution_smooth(xs, y, B)) .-
                            convolution_smooth(xs, reverse(y), B))) < 1e-12
    end
end