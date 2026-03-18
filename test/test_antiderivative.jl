#######################################################################
### TEST ANTIDERIVATIVE (derivative = -1) #############################
#######################################################################

# Kernels to test for antiderivative
antideriv_kernels_a = [:a3, :a4, :a5, :a7]
antideriv_kernels_b = [:b5, :b7, :b9] #, :b11] #, :b13]
antideriv_kernels_all = vcat(antideriv_kernels_a, antideriv_kernels_b)

# Expected convergence order for antiderivative (one order higher than interpolation)
# a-series: reproduces degree-3 polynomials → antiderivative converges at order 4
# b-series: high order → converges at 6+ (conservative threshold)
expected_order_antideriv_fast = Dict(
    :a3 => 3, :a4 => 3, :a5 => 3, :a7 => 3,
    :b5 => 5, :b7 => 5, :b9 => 5 #, :b11 => 5 #, :b13 => 5
)
expected_order_antideriv_direct = Dict(
    :a3 => 3, :a4 => 3, :a5 => 3, :a7 => 3,
    :b5 => 5, :b7 => 5, :b9 => 5 #, :b11 => 5 #, :b13 => 5
)

# Test points well away from boundaries (at least eqs points in)
# eqs ranges from 2 (a3) to 9 (b13), so index 10 from each end is safe for all kernels
interior_frac_1d(r) = range(r[10], r[end-9], length=50)
interior_pts_2d(r1, r2) = [(x, y) for x in range(r1[8], r1[end-7], length=8),
                                    y in range(r2[8], r2[end-7], length=8)]
interior_pts_3d(r1, r2, r3) = [(x, y, z) for x in range(r1[6], r1[end-5], length=5),
                                            y in range(r2[6], r2[end-5], length=5),
                                            z in range(r3[6], r3[end-5], length=5)]

println("Testing antiderivative (derivative=-1) convergence in 1D, 2D, 3D...")

# ── :a0 antiderivative ────────────────────────────────────────────────

@testset "1D antiderivative :a0 kernel" begin
    n = 1001
    x = range(0.0, 2π, length=n)
    h = x[2] - x[1]

    # constant function: ∫c dx = c*(x - anchor)
    y = fill(2.0, length(x))
    for fast in [true, false]
        itp = convolution_interpolation(x, y; kernel=:a0, derivative=-1, fast=fast)
        anchor = itp.itp.anchor[1]
        @test abs(itp(anchor)) < 1e-10
        @test abs(itp(1.0) - 2.0 * (1.0 - anchor)) < 1e-6
    end

    # delta spike: ∫δ dx = Heaviside
    y2 = [abs(xi - π) ≈ 0.0 ? 1.0/h : 0.0 for xi in x]
    for fast in [true, false]
        itp = convolution_interpolation(x, y2; kernel=:a0, derivative=-1, fast=fast)
        @test abs(itp(π + 0.1) - 1.0) < 1e-6
        @test abs(itp(π - 0.1)) < 1e-6
    end

    # fast vs direct agreement
    itp_f = convolution_interpolation(x, y2; kernel=:a0, derivative=-1, fast=true)
    itp_s = convolution_interpolation(x, y2; kernel=:a0, derivative=-1, fast=false)
    @test abs(itp_f(π + 0.5) - itp_s(π + 0.5)) < 1e-6
end

# ── 1D fast ───────────────────────────────────────────────────────────

@testset "1D antiderivative convergence fast" begin
    for kernel in antideriv_kernels_all
        println("Testing 1D antiderivative fast: ", kernel)
        errs = Float64[]
        for n in [24, 48, 96]
            r = range(0.0, 2π, length=n)
            vals = sin.(collect(r))
            itp = convolution_interpolation(r, vals; kernel=kernel, derivative=-1,
                                            fast=true, bc=:poly)
            anchor = itp.itp.anchor[1]
            exact(x) = -cos(x) + cos(anchor)
            test_pts = interior_frac_1d(collect(r))
            err = maximum(abs(itp(x) - exact(x)) for x in test_pts)
            push!(errs, err)
        end
        min_ratio = 2.0^(expected_order_antideriv_fast[kernel] - 1)
        @test errs[1] / errs[2] > min_ratio
        @test errs[2] / errs[3] > min_ratio
    end
end

# ── 1D direct ─────────────────────────────────────────────────────────

@testset "1D antiderivative convergence direct" begin
    for kernel in antideriv_kernels_all
        println("Testing 1D antiderivative direct: ", kernel)
        errs = Float64[]
        for n in [24, 48, 96]
            r = range(0.0, 2π, length=n)
            vals = sin.(collect(r))
            itp = convolution_interpolation(r, vals; kernel=kernel, derivative=-1,
                                            fast=false, bc=:poly)
            anchor = itp.itp.anchor[1]
            exact(x) = -cos(x) + cos(anchor)
            test_pts = interior_frac_1d(collect(r))
            err = maximum(abs(itp(x) - exact(x)) for x in test_pts)
            push!(errs, err)
        end
        min_ratio = 2.0^(expected_order_antideriv_direct[kernel] - 1)
        @test errs[1] / errs[2] > min_ratio
        @test errs[2] / errs[3] > min_ratio
    end
end

# ── 1D fast vs direct agreement ───────────────────────────────────────

@testset "1D antiderivative fast vs direct agreement" begin
    for kernel in antideriv_kernels_all
        println("Testing 1D antiderivative fast vs direct: ", kernel)
        n = 30
        r = range(0.0, 2π, length=n)
        vals = sin.(collect(r))
        itp_fast   = convolution_interpolation(r, vals; kernel=kernel, derivative=-1,
                                               fast=true,  bc=:poly)
        itp_direct = convolution_interpolation(r, vals; kernel=kernel, derivative=-1,
                                               fast=false, bc=:poly)
        test_pts = interior_frac_1d(collect(r))
        err = maximum(abs(itp_fast(x) - itp_direct(x)) for x in test_pts)
        @test err < 1e-6
    end
end

# ── 1D anchor is zero ─────────────────────────────────────────────────

@testset "1D antiderivative anchor is zero" begin
    for kernel in antideriv_kernels_all
        r = range(0.0, 2π, length=20)
        vals = sin.(collect(r))
        for fast in [true, false]
            itp = convolution_interpolation(r, vals; kernel=kernel, derivative=-1,
                                            fast=fast, bc=:poly)
            anchor = itp.itp.anchor[1]
            @test abs(itp(anchor)) < 1e-10
        end
    end
end

# ── 2D fast ───────────────────────────────────────────────────────────

@testset "2D antiderivative convergence fast" begin
    for kernel in [:a3, :b5, :b7, :b9]
        println("Testing 2D antiderivative fast: ", kernel)
        errs = Float64[]
        for n in [20, 40, 80]
            xs = range(0.0, 2π, length=n)
            ys = range(0.0, 2π, length=n)
            zs = [sin(x)*cos(y) for x in xs, y in ys]
            itp = convolution_interpolation((xs, ys), zs; kernel=kernel, derivative=-1,
                                            fast=true, bc=:poly)
            ax, ay = itp.itp.anchor[1], itp.itp.anchor[2]
            exact(x, y) = (-cos(x) + cos(ax)) * (sin(y) - sin(ay))
            test_pts = interior_pts_2d(collect(xs), collect(ys))
            err = maximum(abs(itp(p...) - exact(p...)) for p in test_pts)
            push!(errs, err)
        end
        min_ratio = 2.0^(expected_order_antideriv_fast[kernel] - 1)
        @test errs[1] / errs[2] > min_ratio
        @test errs[2] / errs[3] > min_ratio
    end
end

# ── 2D direct ─────────────────────────────────────────────────────────

@testset "2D antiderivative convergence direct" begin
    for kernel in [:a3, :b5, :b7, :b9]
        println("Testing 2D antiderivative direct: ", kernel)
        errs = Float64[]
        for n in [20, 40, 80]
            xs = range(0.0, 2π, length=n)
            ys = range(0.0, 2π, length=n)
            zs = [sin(x)*cos(y) for x in xs, y in ys]
            itp = convolution_interpolation((xs, ys), zs; kernel=kernel, derivative=-1,
                                            fast=false, bc=:poly)
            ax, ay = itp.itp.anchor[1], itp.itp.anchor[2]
            exact(x, y) = (-cos(x) + cos(ax)) * (sin(y) - sin(ay))
            test_pts = interior_pts_2d(collect(xs), collect(ys))
            err = maximum(abs(itp(p...) - exact(p...)) for p in test_pts)
            push!(errs, err)
        end
        min_ratio = 2.0^(expected_order_antideriv_direct[kernel] - 1)
        @test errs[1] / errs[2] > min_ratio
        @test errs[2] / errs[3] > min_ratio
    end
end

# ── 2D anchor is zero ─────────────────────────────────────────────────

@testset "2D antiderivative anchor is zero" begin
    for kernel in [:a3, :b5, :b7]
        xs = range(0.0, 2π, length=20)
        ys = range(0.0, 2π, length=20)
        zs = [sin(x)*cos(y) for x in xs, y in ys]
        for fast in [true, false]
            itp = convolution_interpolation((xs, ys), zs; kernel=kernel, derivative=-1,
                                            fast=fast, bc=:poly)
            ax, ay = itp.itp.anchor[1], itp.itp.anchor[2]
            @test abs(itp(ax, ay)) < 1e-10
        end
    end
end

# ── 2D fast/direct even/odd functions ─────────────────────────────────

@testset "2D antiderivative fast/direct even/odd functions" begin
    for (fname, f, F) in [
        ("sincos", (x,y)->sin(x)*cos(y), (x,y)->(1-cos(x))*sin(y)),
        ("cossin", (x,y)->cos(x)*sin(y), (x,y)->sin(x)*(1-cos(y))),
        ("sinsin", (x,y)->sin(x)*sin(y), (x,y)->(1-cos(x))*(1-cos(y))),
        ("coscos", (x,y)->cos(x)*cos(y), (x,y)->sin(x)*sin(y)),
    ]
        xs = range(0.0, 2π, length=40)
        ys = range(0.0, 2π, length=40)
        vs = [f(x,y) for x in xs, y in ys]
        itp_f = convolution_interpolation((xs,ys), vs; kernel=:b5, derivative=-1, fast=true)
        itp_s = convolution_interpolation((xs,ys), vs; kernel=:b5, derivative=-1, fast=false)
        pts = interior_pts_2d(collect(xs), collect(ys))
        err_f = maximum(abs(itp_f(p...) - F(p...)) for p in pts)
        err_s = maximum(abs(itp_s(p...) - F(p...)) for p in pts)
        @test err_f < 1e-6
        @test err_f < 10 * err_s  # fast should be close to slow
    end
end

# ── 3D fast ───────────────────────────────────────────────────────────

@testset "3D antiderivative convergence fast" begin
    for kernel in [:a3, :b5, :b7]
        println("Testing 3D antiderivative fast: ", kernel)
        errs = Float64[]
        for n in [12, 24, 48]
            xs = range(0.0, 2π, length=n)
            ys = range(0.0, 2π, length=n)
            zs = range(0.0, 2π, length=n)
            vs = [sin(x)*cos(y)*sin(z) for x in xs, y in ys, z in zs]
            itp = convolution_interpolation((xs, ys, zs), vs; kernel=kernel, derivative=-1,
                                            fast=true, bc=:poly)
            ax, ay, az = itp.itp.anchor[1], itp.itp.anchor[2], itp.itp.anchor[3]
            exact(x, y, z) = (-cos(x) + cos(ax)) * (sin(y) - sin(ay)) * (-cos(z) + cos(az))
            test_pts = interior_pts_3d(collect(xs), collect(ys), collect(zs))
            err = maximum(abs(itp(p...) - exact(p...)) for p in test_pts)
            push!(errs, err)
        end
        min_ratio = 2.0^(expected_order_antideriv_fast[kernel] - 1)
        @test errs[1] / errs[2] > min_ratio
        @test errs[2] / errs[3] > min_ratio
    end
end

# ── 3D direct ─────────────────────────────────────────────────────────

@testset "3D antiderivative convergence direct" begin
    for kernel in [:a3, :b5, :b7]
        println("Testing 3D antiderivative direct: ", kernel)
        errs = Float64[]
        for n in [12, 24, 48]
            xs = range(0.0, 2π, length=n)
            ys = range(0.0, 2π, length=n)
            zs = range(0.0, 2π, length=n)
            vs = [sin(x)*cos(y)*sin(z) for x in xs, y in ys, z in zs]
            itp = convolution_interpolation((xs, ys, zs), vs; kernel=kernel, derivative=-1,
                                            fast=false, bc=:poly)
            ax, ay, az = itp.itp.anchor[1], itp.itp.anchor[2], itp.itp.anchor[3]
            exact(x, y, z) = (-cos(x) + cos(ax)) * (sin(y) - sin(ay)) * (-cos(z) + cos(az))
            test_pts = interior_pts_3d(collect(xs), collect(ys), collect(zs))
            err = maximum(abs(itp(p...) - exact(p...)) for p in test_pts)
            push!(errs, err)
        end
        min_ratio = 2.0^(expected_order_antideriv_direct[kernel] - 1)
        @test errs[1] / errs[2] > min_ratio
        @test errs[2] / errs[3] > min_ratio
    end
end