println("\n" * "-"^60)
println("Testing deprecations (precompute, subgrid, boundary_fallback, scattered_to_grid)...")
println("-"^60)

# `precompute`, `subgrid` and `boundary_fallback` no longer have any effect. Setting them must
# produce a warning and exactly the same result as not setting them; not setting them must
# produce no warnings.
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
        # The reference fit (noise-free data: exact interpolation)
        s_ref = convolution_interpolation(points, vals; mode = :exact)
        # Deprecated precompute keyword: a warning, and an identical fit
        s_dep = @test_logs (:warn, r"`precompute` no longer has any effect") convolution_interpolation(points, vals; mode = :exact, precompute=101)
        @test s_dep(xc, yc) == s_ref(xc, yc)
    end

    println("    - boundary_fallback")
    @testset "boundary_fallback" begin
        warning = (:warn, r"`boundary_fallback` no longer has any effect")
        # uniform lazy 2D data: the keyword used to switch boundary cells to bilinear interpolation
        xs = range(0.0, 2π, length=30)
        z = [sin(a) * cos(b) for a in xs, b in xs]
        ref = @test_logs convolution_interpolation((xs, xs), z; kernel=:b5, lazy=true)
        for bf in (true, false)
            dep = @test_logs warning convolution_interpolation((xs, xs), z; kernel=:b5, lazy=true,
                                                               boundary_fallback=bf)
            @test dep(0.05, 0.05) == ref(0.05, 0.05)          # a boundary cell
            @test dep(3.0, 2.0) == ref(3.0, 2.0)              # an interior cell
        end
        # the fast constructor directly
        fi_ref = FastConvolutionInterpolation((xs, xs), z; kernel=:b5, lazy=true)
        fi_dep = @test_logs warning FastConvolutionInterpolation((xs, xs), z; kernel=:b5, lazy=true,
                                                                 boundary_fallback=true)
        @test fi_dep(0.05, 0.05) == fi_ref(0.05, 0.05)
        # the direct constructor, on a nonuniform grid (the :n3 lazy path)
        xn = [0.0, 0.3, 0.7, 1.2, 1.5, 2.0]
        cn_ref = ConvolutionInterpolation(xn, sin.(xn); kernel=:n3, lazy=true)
        cn_dep = @test_logs warning ConvolutionInterpolation(xn, sin.(xn); kernel=:n3, lazy=true,
                                                             boundary_fallback=true)
        @test cn_dep(0.05) == cn_ref(0.05)
    end

    println("    - scattered_to_grid")
    @testset "scattered_to_grid" begin
        # Scattered 2D points (one per column), their values, and a target grid
        rng = Random.MersenneTwister(4)
        points = rand(rng, 2, 50)
        vals = vec(sum(points; dims = 1))
        xs = range(0.0, 1.0, length = 11)
        # The deprecated function warns and still grids the data
        grid = @test_logs (:warn, r"`scattered_to_grid` is deprecated") scattered_to_grid(points, vals, (xs, xs))
        @test size(grid) == (11, 11)
        @test all(isfinite, grid)
    end
end