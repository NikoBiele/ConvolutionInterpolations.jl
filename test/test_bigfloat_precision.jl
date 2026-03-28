@testset "BigFloat precision preservation" begin
    setprecision(256) do
        x_test = BigFloat("1.234567890123456789")
        x1, x2 = BigFloat("1.0"), BigFloat("2.0")

        # ── type preservation only (coarse grid, all kernels) ─────────────
        @testset "type preservation kernel $k" for k in [:b5, :b7, :b9, :b11, :b13, :a3, :a4, :a5, :a7]
            println("Testing 1D BigFloat type preservation for kernel $k")
            xs = LinRange(BigFloat("0.0"), BigFloat(2) * BigFloat(π), 50)
            vs = sin.(xs)

            itp = convolution_interpolation(xs, vs, kernel=k)
            result = itp(x_test)
            @test result isa BigFloat
            @test abs(result - sin(x_test)) < BigFloat("1e-4")

            itp_d = convolution_interpolation(xs, vs, kernel=k, derivative=1)
            result_d = itp_d(x_test)
            @test result_d isa BigFloat
            @test abs(result_d - cos(x_test)) < BigFloat("1e-3")

            itp_int = convolution_interpolation(xs, vs, kernel=k, derivative=-1)
            result_int = itp_int(x2) - itp_int(x1)
            @test result_int isa BigFloat
            @test abs(result_int - (-cos(x2) + cos(x1))) < BigFloat("1e-3")
        end

        # ── machine precision on dense grid (b-kernels only) ─────────────
        @testset "machine precision kernel $k" for k in [:b5, :b7, :b9, :b11, :b13]
            println("Testing 1D BigFloat machine precision for kernel $k")
            xs = LinRange(BigFloat("0.0"), BigFloat(2) * BigFloat(π), 2000)
            vs = sin.(xs)

            itp = convolution_interpolation(xs, vs, kernel=k)
            result = itp(x_test)
            @test result isa BigFloat
            @test abs(result - sin(x_test)) < BigFloat("1e-16")

            itp_d = convolution_interpolation(xs, vs, kernel=k, derivative=1)
            result_d = itp_d(x_test)
            @test result_d isa BigFloat
            @test abs(result_d - cos(x_test)) < BigFloat("1e-14")

            itp_int = convolution_interpolation(xs, vs, kernel=k, derivative=-1)
            result_int = itp_int(x2) - itp_int(x1)
            @test result_int isa BigFloat
            @test abs(result_int - (-cos(x2) + cos(x1))) < BigFloat("1e-14")
        end

        # ── 2D type preservation ──────────────────────────────────────────
        @testset "2D BigFloat type preservation" begin
            println("Testing 2D BigFloat type preservation")
            xs = LinRange(BigFloat("0.0"), BigFloat(2) * BigFloat(π), 30)
            ys = LinRange(BigFloat("0.0"), BigFloat(2) * BigFloat(π), 30)
            vs2 = [sin(x) * cos(y) for x in xs, y in ys]
            itp2 = convolution_interpolation((xs, ys), vs2)
            result2 = itp2(x_test, BigFloat("0.8"))
            @test result2 isa BigFloat
            @test abs(result2 - sin(x_test) * cos(BigFloat("0.8"))) < BigFloat("1e-4")
        end

    end
end