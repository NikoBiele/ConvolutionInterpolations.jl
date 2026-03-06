#########################################################################
### TEST CONSTRUCTORS ###################################################
#########################################################################
@testset "Low-level constructors" begin
    println("Testing low-level constructors...")
    x = range(0.0, 2π, length=30)
    vs = sin.(x)
    knots = (x,)

    for degree in (:a0, :a1, :a3, :a4, :a5, :a7, :b5, :b7, :b9, :b11, :b13)
        println("Testing kernel degree: $degree")

        # Direct constructor
        itp_direct = ConvolutionInterpolation(knots, vs, degree=degree)
        @test itp_direct(1.0) ≈ sin(1.0) atol=0.1

        # Fast constructor
        itp_fast = FastConvolutionInterpolation(knots, vs, degree=degree)
        @test itp_fast(1.0) ≈ sin(1.0) atol=0.1

        # Direct and fast agree
        @test itp_direct(1.0) ≈ itp_fast(1.0) atol=1e-3

        # Extrapolation wrapper
        etp = ConvolutionExtrapolation(itp_fast, Line())
        @test etp(1.0) ≈ sin(1.0) atol=0.1
        @test etp(-0.5) isa Float64  # doesn't throw

        etp_throw = ConvolutionExtrapolation(itp_fast, Throw())
        @test_throws ErrorException etp_throw(-0.5)

        # Lazy variants
        itp_direct_lazy = ConvolutionInterpolation(knots, vs, degree=degree, lazy=true)
        itp_fast_lazy = FastConvolutionInterpolation(knots, vs, degree=degree, lazy=true)
        @test itp_direct_lazy(1.0) ≈ itp_direct(1.0) atol=1e-3
        @test itp_fast_lazy(1.0) ≈ itp_fast(1.0) atol=1e-3

        # 2D
        y = range(0.0, 2π, length=30)
        vs_2d = [sin(xi) * cos(yi) for xi in x, yi in y]
        knots_2d = (x, y)

        itp_direct_2d = ConvolutionInterpolation(knots_2d, vs_2d, degree=degree)
        itp_fast_2d = FastConvolutionInterpolation(knots_2d, vs_2d, degree=degree)
        @test itp_direct_2d(1.0, 1.0) ≈ itp_fast_2d(1.0, 1.0) atol=1e-3
        @test itp_direct_2d(1.0, 1.0) ≈ sin(1.0) * cos(1.0) atol=0.1
    end
end