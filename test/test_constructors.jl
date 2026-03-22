#########################################################################
### TEST CONSTRUCTORS ###################################################
#########################################################################
@testset "Low-level constructors" begin
    println("Testing low-level constructors...")
    x = range(0.0, 2π, length=30)
    vs = sin.(x)

    for kernel in (:a0, :a1, :a3, :a4, :a5, :a7, :b5, :b7, :b9, :b11, :b13)
        println("Testing kernel: $kernel")

        # Direct constructor
        itp_direct = ConvolutionInterpolation(x, vs, kernel=kernel)
        @test itp_direct(1.0) ≈ sin(1.0) atol=0.1

        # Fast constructor
        itp_fast = FastConvolutionInterpolation(x, vs, kernel=kernel)
        @test itp_fast(1.0) ≈ sin(1.0) atol=0.1

        # Direct and fast agree
        @test itp_direct(1.0) ≈ itp_fast(1.0) atol=1e-6

        # Extrapolation wrapper
        etp = ConvolutionExtrapolation(itp_fast, :line)
        @test etp(1.0) ≈ sin(1.0) atol=0.1
        @test etp(-0.5) isa Float64  # doesn't throw

        etp_throw = ConvolutionExtrapolation(itp_fast, :throw)
        @test_throws ErrorException etp_throw(-0.5)

        # Lazy variants
        itp_direct_lazy = ConvolutionInterpolation(x, vs, kernel=kernel, lazy=true)
        itp_fast_lazy = FastConvolutionInterpolation(x, vs, kernel=kernel, lazy=true)
        @test itp_direct_lazy(1.0) ≈ itp_direct(1.0) atol=1e-6
        @test itp_fast_lazy(1.0) ≈ itp_fast(1.0) atol=1e-6

        # 2D
        y = range(0.0, 2π, length=30)
        vs_2d = [sin(xi) * cos(yi) for xi in x, yi in y]
        knots_2d = (x, y)

        itp_direct_2d = ConvolutionInterpolation(knots_2d, vs_2d, kernel=kernel)
        itp_fast_2d = FastConvolutionInterpolation(knots_2d, vs_2d, kernel=kernel)
        @test itp_direct_2d(1.0, 1.0) ≈ itp_fast_2d(1.0, 1.0) atol=1e-6
        @test itp_direct_2d(1.0, 1.0) ≈ sin(1.0) * cos(1.0) atol=0.1
    end
end

kernels = (:a0, :a1, :a3, :b5)
xs = range(0.0, 2π, length=30)
ys = range(0.0, 2π, length=30)
vs = [sin(x)*cos(y) for x in xs, y in ys]

@testset "2D per-dim kernel combinations" begin
    println("Testing 2D per-dim kernel combinations...")
    for k1 in kernels, k2 in kernels
        println("Testing kernel combination: ", k1, ", ", k2)
        itp = convolution_interpolation((xs, ys), vs; kernel=(k1, k2))
        @test isfinite(itp(1.5, 1.0))
    end
end

zs = range(0.0, 2π, length=30)
vs3 = [sin(x)*cos(y)*sin(z) for x in xs, y in ys, z in zs]

@testset "3D per-dim kernel combinations" begin
    println("Testing 3D per-dim kernel combinations...")
    for k1 in kernels, k2 in kernels, k3 in kernels
        println("Testing kernel combination: ", k1, ", ", k2, ", ", k3)
        itp = convolution_interpolation((xs, ys, zs), vs3; kernel=(k1, k2, k3))
        @test isfinite(itp(1.5, 1.0, 2.0))
    end
end