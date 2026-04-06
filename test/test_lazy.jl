#########################################################################
### TEST LAZY MODE CORRECTNESS ##########################################
#########################################################################
println("Testing lazy mode...")

@testset "Lazy mode" begin

    @testset "1D-3D lazy=true boundary_fallback=false matches eager" begin
        println("Testing 1D-3D lazy=true boundary_fallback=false matches eager...")
        # For N<=3 with boundary_fallback=false, lazy should match eager to high precision
        # both in interior AND at boundaries
        for kernel in (:a3, :b5)
            xs = range(0.0, 2π, length=50)
            vs = sin.(xs)
            itp_e = convolution_interpolation(xs, vs; kernel=kernel, lazy=false)
            itp_l = convolution_interpolation(xs, vs; kernel=kernel, lazy=true, boundary_fallback=false)
            # interior
            for x in range(0.5, 2π-0.5, length=20)
                @test itp_e(x) ≈ itp_l(x) atol=1e-10
            end
            # near boundaries
            for x in [0.01, 0.03, 2π-0.03, 2π-0.01]
                @test itp_e(x) ≈ itp_l(x) atol=1e-10
            end
        end

        for kernel in (:a3, :b5)
            xs = range(0.0, 2π, length=40)
            vs_2d = [sin(x)*cos(y) for x in xs, y in xs]
            itp_e = convolution_interpolation((xs,xs), vs_2d; kernel=kernel, lazy=false)
            itp_l = convolution_interpolation((xs,xs), vs_2d; kernel=kernel, lazy=true, boundary_fallback=false)
            for x in [0.02, 1.0, 2.5, 5.5, 2π-0.02]
                for y in [0.02, 1.5, 4.0, 2π-0.02]
                    @test itp_e(x, y) ≈ itp_l(x, y) atol=1e-10
                end
            end
        end

        for kernel in (:a3, :b5)
            xs = range(0.0, 2π, length=30)
            vs_3d = [sin(x)*cos(y)*sin(z) for x in xs, y in xs, z in xs]
            itp_e = convolution_interpolation((xs,xs,xs), vs_3d; kernel=kernel, lazy=false)
            itp_l = convolution_interpolation((xs,xs,xs), vs_3d; kernel=kernel, lazy=true, boundary_fallback=false)
            for (x,y,z) in [(0.02,0.5,1.0), (1.5,2.0,3.0), (5.5,5.0,4.5), (2π-0.2,2π-0.3,2π-0.04)]
                @test itp_e(x, y, z) ≈ itp_l(x, y, z) atol=1e-10
            end
        end
    end

    @testset "N>=4 lazy=true boundary_fallback=true interior matches eager" begin
        println("Testing N>=4 lazy=true boundary_fallback=true interior matches eager...")
        # Interior points should match eager to high precision
        # Near-boundary points use linear fallback — still correct but lower order
        for kernel in (:a3, :b5)
            xs = range(0.0, 2π, length=30)
            vs_4d = [sin(x)*cos(y)*sin(z)*cos(t)
                     for x in xs, y in xs, z in xs, t in xs]
            itp_e = convolution_interpolation((xs,xs,xs,xs), vs_4d;
                        kernel=kernel, lazy=false, boundary_fallback=true)
            itp_l = convolution_interpolation((xs,xs,xs,xs), vs_4d;
                        kernel=kernel, lazy=true, boundary_fallback=true)

            # interior — should agree to high precision
            for (x,y,z,t) in [(1.0,1.5,2.0,2.5), (2.0,2.5,3.0,3.5), (3.0,3.5,4.0,4.5)]
                @test itp_e(x, y, z, t) ≈ itp_l(x, y, z, t) atol=1e-10
            end

            # near boundary — linear fallback, still reasonable accuracy
            for (x,y,z,t) in [(0.1,0.1,0.1,0.1), (2π-0.1,2π-0.1,2π-0.1,2π-0.1)]
                @test itp_e(x, y, z, t) ≈ itp_l(x, y, z, t) atol=1e-2
            end
        end
    end

    @testset "lazy construction is fast" begin
        println("Testing lazy construction is fast...")
        # lazy construction should be fast regardless of grid size
        xs_big = range(0.0, 1.0, length=50)
        vs_3d  = rand(50, 50, 50)
        convolution_interpolation((xs_big,xs_big,xs_big), vs_3d;
                        kernel=:b5, lazy=true) # warm up
        t = @elapsed convolution_interpolation((xs_big,xs_big,xs_big), vs_3d;
                        kernel=:b5, lazy=true)
        @test t < 1.0

        xs_4d = range(0.0, 1.0, length=20)
        vs_4d = rand(20, 20, 20, 20)
        convolution_interpolation((xs_4d,xs_4d,xs_4d,xs_4d), vs_4d;
                        kernel=:b5, lazy=true, boundary_fallback=true) # warm up
        t = @elapsed convolution_interpolation((xs_4d,xs_4d,xs_4d,xs_4d), vs_4d;
                        kernel=:b5, lazy=true, boundary_fallback=true)
        @test t < 1.0
    end

    @testset "lazy stores raw values for uniform grids" begin
        println("Testing lazy stores raw values for uniform grids...")
        xs = range(0.0, 2π, length=20)
        vs = sin.(xs)
        itp_l = convolution_interpolation(xs, vs; kernel=:b5, lazy=true)
        @test itp_l.itp.lazy == Val{true}()
        @test itp_l.itp.coefs === vs
    end

    @testset "lazy nonuniform n3 stores raw values" begin
        println("Testing lazy nonuniform n3 stores raw values...")
        xs = [0.0, 0.3, 0.7, 1.2, 1.5, 2.0]
        vs = sin.(xs)
        itp_l = convolution_interpolation(xs, vs; kernel=:a3, lazy=true)
        @test itp_l.itp.lazy == Val{true}()
        @test itp_l.itp.coefs === vs
    end

    @testset "nonuniform b-kernels force eager despite lazy=true" begin
        println("Testing nonuniform b-kernels force eager despite lazy=true...")
        xs = [0.0, 0.3, 0.7, 1.2, 1.5, 2.0]
        vs = sin.(xs)
        itp_b = convolution_interpolation(xs, vs; kernel=:b5, lazy=true)
        @test itp_b.itp.lazy == Val{false}()
    end

    @testset "Lazy nonuniform n3 correctness in 1D, 2D, 3D" begin
        println("Testing lazy nonuniform n3 correctness in 1D, 2D, 3D...")
        # 1D: eager vs lazy match
        knots_1d = [0.0, 0.3, 0.7, 1.2, 1.5, 2.0]
        vs_1d = sin.(knots_1d)
        itp_e = convolution_interpolation(knots_1d, vs_1d, kernel=:a3, lazy=false)
        itp_l = convolution_interpolation(knots_1d, vs_1d, kernel=:a3, lazy=true)

        # Verify lazy stores raw values
        @test itp_l.itp.lazy == Val{true}()
        @test itp_l.itp.coefs === vs_1d
        @test size(itp_l.itp.coefs) == size(vs_1d)

        # Interior, boundary, and near-boundary points
        for x in [0.1, 0.5, 1.0, 1.3, 1.9]
            @test itp_e(x) ≈ itp_l(x) atol=1e-6
        end

        # 2D: eager vs lazy match
        kx = [0.0, 0.4, 0.9, 1.5, 2.0]
        ky = [0.0, 0.3, 0.8, 1.2, 1.7, 2.0]
        vs_2d = Float64[sin(x + y) for x in kx, y in ky]
        itp_e = convolution_interpolation((kx, ky), vs_2d, kernel=:a3, lazy=false)
        itp_l = convolution_interpolation((kx, ky), vs_2d, kernel=:a3, lazy=true)

        @test itp_l.itp.lazy == Val{true}()
        @test itp_l.itp.coefs === vs_2d

        for (x, y) in [(0.2, 0.5), (1.0, 1.0), (0.2, 1.8), (1.8, 0.2), (1.8, 1.8)]
            @test itp_e(x, y) ≈ itp_l(x, y) atol=1e-6
        end

        # 3D: eager vs lazy match
        kz = [0.0, 0.5, 1.0, 1.6, 2.0]
        vs_3d = Float64[sin(x + y + z) for x in kx, y in ky, z in kz]
        itp_e = convolution_interpolation((kx, ky, kz), vs_3d, kernel=:a3, lazy=false)
        itp_l = convolution_interpolation((kx, ky, kz), vs_3d, kernel=:a3, lazy=true)

        @test itp_l.itp.lazy == Val{true}()
        @test itp_l.itp.coefs === vs_3d

        for (x, y, z) in [(0.5, 0.5, 0.5), (0.2, 1.5, 0.8), (1.8, 0.2, 1.5)]
            @test itp_e(x, y, z) ≈ itp_l(x, y, z) atol=1e-6
        end

        # Nonuniform b-kernels still force eager
        itp_b = convolution_interpolation(knots_1d, vs_1d, kernel=:b5, lazy=true)
        @test itp_b.itp.lazy == Val{false}()
    end

end