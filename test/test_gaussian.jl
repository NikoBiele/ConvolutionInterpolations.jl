println("Testing Gaussian smoothing kernel...")
@testset "Gaussian kernel" begin
    xs = range(0.0, 2π, length=50)
    vs = sin.(xs)

    for B in (1.0, 0.75, 0.6)
        itp = convolution_gaussian(xs, vs, B)
        # should be smooth and reasonably close to sin
        @test abs(itp(1.5) - sin(1.5)) < 0.1
    end

    # 2D
    xs2 = range(0.0, 2π, length=20)
    ys2 = range(0.0, 2π, length=20)
    vs2 = [sin(x)*cos(y) for x in xs2, y in ys2]
    B = 0.6
    itp2 = convolution_gaussian((xs2, ys2), vs2, B)
    @test maximum(abs.(itp2.(xs2, ys2') - vs2)) < 0.1
end

println("Testing convolution_smooth...")
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