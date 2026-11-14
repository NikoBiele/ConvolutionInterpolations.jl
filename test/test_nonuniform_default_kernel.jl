println("\n" * "-"^60)
println("Testing the default kernel on nonuniform grids...")
println("-"^60)

@testset "Default kernel — nonuniform grids" begin

    @testset "the rule by dimension and grid" begin
        println("    - the rule by dimension and grid...")
        default_kernel = ConvolutionInterpolations._default_kernel
        # 1D to 3D: the same kernel on any grid
        for nonuniform in (false, true)
            @test default_kernel(1, nonuniform) === :b7
            @test default_kernel(2, nonuniform) === :b7
            @test default_kernel(3, nonuniform) === :b5
        end
        # from 4D: the uniform-only :a4 on uniform grids, :n3 as soon as an axis is nonuniform
        @test default_kernel(4, false) === :a4
        @test default_kernel(5, false) === :a4
        @test default_kernel(6, false) === :a3
        @test default_kernel(4, true) === :n3
        @test default_kernel(5, true) === :n3
        @test default_kernel(6, true) === :n3
        # the one-argument form means a uniform grid
        @test default_kernel(4) === :a4
    end

    # a linear function, which :n3 reproduces, so the interpolant can be checked between the knots
    f4(x, y, z, w) = 1.0 + x + 2.0 * y - z + 0.5 * w
    points = [(0.5, 0.6, 0.9, 1.1), (0.2, 1.4, 0.35, 0.8), (1.7, 0.1, 1.2, 1.9)]

    @testset "4D, every axis nonuniform, no kernel given" begin
        println("    - 4D, every axis nonuniform, no kernel given...")
        kx = [0.0, 0.3, 0.7, 1.2, 1.5, 2.0]
        ky = [0.0, 0.4, 0.9, 1.5, 2.0]
        kz = [0.0, 0.2, 0.8, 1.1, 1.6, 2.0]
        kw = [0.0, 0.5, 0.9, 1.4, 2.0]
        vs = Float64[f4(x, y, z, w) for x in kx, y in ky, z in kz, w in kw]
        itp = convolution_interpolation((kx, ky, kz, kw), vs)
        itp_n3 = convolution_interpolation((kx, ky, kz, kw), vs; kernel=:n3)
        for p in points
            @test itp(p...) == itp_n3(p...)
            @test isapprox(itp(p...), f4(p...); atol=1e-12)
        end
        @test isapprox(itp(kx[3], ky[2], kz[4], kw[3]), vs[3, 2, 4, 3]; atol=1e-12)
    end

    @testset "4D, one nonuniform axis among ranges, no kernel given" begin
        println("    - 4D, one nonuniform axis among ranges, no kernel given...")
        kx = [0.0, 0.3, 0.7, 1.2, 1.5, 2.0]
        ky = range(0.0, 2.0, length=5)
        kz = range(0.0, 2.0, length=6)
        kw = range(0.0, 2.0, length=5)
        vs = Float64[f4(x, y, z, w) for x in kx, y in ky, z in kz, w in kw]
        itp = convolution_interpolation((kx, ky, kz, kw), vs)
        itp_n3 = convolution_interpolation((kx, ky, kz, kw), vs; kernel=:n3)
        for p in points
            @test itp(p...) == itp_n3(p...)
            @test isapprox(itp(p...), f4(p...); atol=1e-12)
        end
        @test isapprox(itp(kx[3], ky[2], kz[4], kw[3]), vs[3, 2, 4, 3]; atol=1e-12)
    end

    @testset "4D uniform grid keeps its default" begin
        println("    - 4D uniform grid keeps its default...")
        k = range(0.0, 2.0, length=8)
        vs = Float64[f4(x, y, z, w) for x in k, y in k, z in k, w in k]
        itp = convolution_interpolation((k, k, k, k), vs)
        itp_a4 = convolution_interpolation((k, k, k, k), vs; kernel=:a4)
        for p in points
            @test itp(p...) == itp_a4(p...)
        end
    end

    @testset ":n3 on a 2D grid of a range and a vector" begin
        println("    - :n3 on a 2D grid of a range and a vector...")
        kx = range(0.0, 2.0, length=7)
        ky = [0.0, 0.3, 0.8, 1.2, 1.7, 2.0]
        f2(x, y) = 2.0 - x + 3.0 * y
        vs = Float64[f2(x, y) for x in kx, y in ky]
        itp = convolution_interpolation((kx, ky), vs; kernel=:n3)
        for p in [(0.5, 0.6), (1.3, 1.9), (0.1, 0.2)]
            @test isapprox(itp(p...), f2(p...); atol=1e-12)
        end
    end
end