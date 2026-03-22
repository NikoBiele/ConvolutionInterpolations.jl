println("Testing Gaussian smoothing kernel...")
@testset "Gaussian kernel" begin
    xs = range(0.0, 2π, length=50)
    vs = sin.(xs)

    for B in (1.0, 2.0, 5.0)
        itp = convolution_interpolation(xs, vs; B=B)
        # should be smooth and reasonably close to sin
        @test abs(itp(1.5) - sin(1.5)) < 0.1
    end

    # 2D
    xs2 = range(0.0, 2π, length=20)
    ys2 = range(0.0, 2π, length=20)
    vs2 = [sin(x)*cos(y) for x in xs2, y in ys2]
    itp2 = convolution_interpolation((xs2, ys2), vs2; B=2.0)
    @test maximum(abs.(itp2.(xs2, ys2') - vs2)) < 0.1
end