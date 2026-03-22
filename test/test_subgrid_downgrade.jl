println("Testing subgrid downgrade at top smooth derivative...")
@testset "Subgrid downgrade at top smooth derivative" begin
    xs = range(0.0, 2π, length=30)
    ys = range(0.0, 2π, length=30)
    vs = [sin(x)*cos(y) for x in xs, y in ys]

    top_deriv = Dict(:a3 => 1, :b5 => 3)

    for (k, d) in top_deriv
        # only dim 1 at top derivative
        itp = convolution_interpolation((xs, ys), vs; kernel=k, derivative=(d, 0), subgrid=:cubic)
        @test itp.itp.subgrid == Val((:linear, :cubic))

        # both dims at top derivative
        itp2 = convolution_interpolation((xs, ys), vs; kernel=k, derivative=(d, d), subgrid=:cubic)
        @test itp2.itp.subgrid == Val((:linear, :linear))
    end
end