println("\n" * "-"^60)
println("Testing lazy mode on nonuniform grids...")
println("-"^60)

@testset "Lazy mode — nonuniform" begin

    @testset "lazy=true is an error on nonuniform grids" begin
        println("    - lazy=true is an error on nonuniform grids...")
        knots = [0.0, 0.3, 0.7, 1.2, 1.5, 2.0]
        vs = sin.(knots)
        @test_throws ErrorException convolution_interpolation(knots, vs; kernel=:a3, lazy=true)
        @test_throws ErrorException convolution_interpolation(knots, vs; kernel=:n3, lazy=true)
        @test_throws ErrorException convolution_interpolation(knots, vs; kernel=:b5, lazy=true)
        kx = [0.0, 0.4, 0.9, 1.5, 2.0]
        ky = [0.0, 0.3, 0.8, 1.2, 1.7, 2.0]
        vs2 = Float64[sin(x + y) for x in kx, y in ky]
        @test_throws ErrorException convolution_interpolation((kx, ky), vs2; kernel=:a3, lazy=true)
    end

    @testset "the default resolves to eager on nonuniform grids" begin
        println("    - the default resolves to eager on nonuniform grids...")
        knots = [0.0, 0.3, 0.7, 1.2, 1.5, 2.0]
        vs = sin.(knots)
        itp_default = convolution_interpolation(knots, vs; kernel=:a3)
        itp_eager = convolution_interpolation(knots, vs; kernel=:a3, lazy=false)
        @test itp_default.itp.lazy == Val{false}()
        @test itp_default.itp.coefs == itp_eager.itp.coefs
        for x in [0.1, 0.5, 1.0, 1.3, 1.9]
            @test itp_default(x) == itp_eager(x)
        end
        itp_b = convolution_interpolation(knots, vs; kernel=:b5)
        @test itp_b.itp.lazy == Val{false}()
    end
end