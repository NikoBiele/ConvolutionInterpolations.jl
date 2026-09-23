println("\n" * "-"^60)
println("Testing deprecated keywords (precompute, subgrid)...")
println("-"^60)

# `precompute` and `subgrid` no longer have any effect. Setting them must produce a warning
# and exactly the same result as not setting them; not setting them must produce no warnings.
@testset "Deprecated keywords" begin
    x = range(0.0, 2π, length=40)
    y = sin.(x)

    println("    - convolution_interpolation")
    @testset "convolution_interpolation" begin
        # No keywords: no warnings at all
        itp_ref = @test_logs convolution_interpolation(x, y; kernel=:b5)
        # Both deprecated keywords: one warning each, and an identical interpolant
        itp_dep = @test_logs (:warn, r"`precompute` no longer has any effect") (:warn, r"`subgrid` no longer has any effect") convolution_interpolation(x, y; kernel=:b5, precompute=10_000, subgrid=:quintic)
        @test itp_dep(1.3) == itp_ref(1.3)
        @test itp_dep(4.7) == itp_ref(4.7)
    end

    println("    - FastConvolutionInterpolation")
    @testset "FastConvolutionInterpolation" begin
        # No keywords: no warnings at all
        fi_ref = @test_logs FastConvolutionInterpolation(x, y; kernel=:b5)
        # Deprecated subgrid keyword: a warning, and an identical interpolant
        fi_dep = @test_logs (:warn, r"`subgrid` no longer has any effect") FastConvolutionInterpolation(x, y; kernel=:b5, subgrid=:linear)
        @test fi_dep(1.3) == fi_ref(1.3)
    end

    println("    - Scattered convolution_interpolation")
    @testset "scattered convolution_interpolation" begin
        # Scattered 2D points (one per column) and their values
        rng = Random.MersenneTwister(3)
        points = 2π .* rand(rng, 2, 40)
        vals = [sin(points[1, j]) * cos(points[2, j]) for j in 1:size(points, 2)]
        # An evaluation point at the centre of the scattered points
        xc = sum(points[1, :]) / size(points, 2)
        yc = sum(points[2, :]) / size(points, 2)
        # No keywords: the reference fit
        s_ref = convolution_interpolation(points, vals)
        # Deprecated precompute keyword: a warning, and an identical fit
        s_dep = @test_logs (:warn, r"`precompute` no longer has any effect") convolution_interpolation(points, vals; precompute=101)
        @test s_dep(xc, yc) == s_ref(xc, yc)
    end
end