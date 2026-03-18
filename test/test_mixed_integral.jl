#######################################################################
### TEST MIXED INTEGRAL ORDER (derivative = (-1,...) per dimension) ###
#######################################################################

# f(x,y) = sin(x)*cos(y)
# ∫f dx   = -cos(x)*cos(y) + C,  anchor ax: C = cos(ax)*cos(y)  → exact(x,y) = (-cos(x) + cos(ax))*cos(y)
# ∫f dy   = sin(x)*sin(y) + C,   anchor ay: C = -sin(x)*sin(ay) → exact(x,y) = sin(x)*(sin(y) - sin(ay))
# ∂f/∂x   = cos(x)*cos(y)
# ∂f/∂y   = -sin(x)*sin(y)
# ∫(∂f/∂y)dx = cos(x)*sin(y) + C, anchor ax: C = -cos(ax)*sin(y) → exact = (cos(x) - cos(ax))*sin(y)
# ∂(∫f dx)/∂y = (-cos(x)+cos(ax))*(-sin(y))

mixed_kernels = [:b5, :b7, :b9]

expected_order_mixed = Dict(:b5 => 5, :b7 => 5, :b9 => 5)

interior_pts_2d_integral(r1, r2) = [(x, y) for x in range(r1[8], r1[end-7], length=8),
                                    y in range(r2[8], r2[end-7], length=8)]

println("Testing mixed integral/derivative order in 2D (slow and fast paths)...")

# ── helpers ───────────────────────────────────────────────────────────

function mixed_exact_int_x(ax, x, y)
    (-cos(x) + cos(ax)) * cos(y)
end

function mixed_exact_int_y(ay, x, y)
    sin(x) * (sin(y) - sin(ay))
end

function mixed_exact_intx_dy(ax, x, y)
    (cos(x) - cos(ax)) * sin(y)
end

function mixed_exact_dx_inty(ay, x, y)
    cos(x) * (sin(y) - sin(ay))
end

# ── 2D slow: (-1, 0) ─────────────────────────────────────────────────

@testset "2D mixed (-1,0) convergence direct" begin
    for kernel in mixed_kernels
        println("Testing 2D mixed (-1,0) direct: ", kernel)
        errs = Float64[]
        for n in [20, 40, 80]
            xs = range(0.0, 2π, length=n); ys = range(0.0, 2π, length=n)
            zs = [sin(x)*cos(y) for x in xs, y in ys]
            itp = convolution_interpolation((xs, ys), zs; kernel=kernel,
                      derivative=(-1, 0), fast=false, bc=:poly)
            ax = itp.itp.anchor[1]
            test_pts = interior_pts_2d_integral(collect(xs), collect(ys))
            err = maximum(abs(itp(p...) - mixed_exact_int_x(ax, p...)) for p in test_pts)
            push!(errs, err)
        end
        min_ratio = 2.0^(expected_order_mixed[kernel] - 1)
        @test errs[1] / errs[2] > min_ratio
        @test errs[2] / errs[3] > min_ratio
    end
end

# ── 2D slow: (0, -1) ─────────────────────────────────────────────────

@testset "2D mixed (0,-1) convergence direct" begin
    for kernel in mixed_kernels
        println("Testing 2D mixed (0,-1) direct: ", kernel)
        errs = Float64[]
        for n in [20, 40, 80]
            xs = range(0.0, 2π, length=n); ys = range(0.0, 2π, length=n)
            zs = [sin(x)*cos(y) for x in xs, y in ys]
            itp = convolution_interpolation((xs, ys), zs; kernel=kernel,
                      derivative=(0, -1), fast=false, bc=:poly)
            ay = itp.itp.anchor[2]
            test_pts = interior_pts_2d_integral(collect(xs), collect(ys))
            err = maximum(abs(itp(p...) - mixed_exact_int_y(ay, p...)) for p in test_pts)
            push!(errs, err)
        end
        min_ratio = 2.0^(expected_order_mixed[kernel] - 1)
        @test errs[1] / errs[2] > min_ratio
        @test errs[2] / errs[3] > min_ratio
    end
end

# ── 2D slow: (-1, 1) ─────────────────────────────────────────────────

@testset "2D mixed (-1,1) convergence direct" begin
    for kernel in mixed_kernels
        println("Testing 2D mixed (-1,1) direct: ", kernel)
        errs = Float64[]
        for n in [20, 40, 80]
            xs = range(0.0, 2π, length=n); ys = range(0.0, 2π, length=n)
            zs = [sin(x)*cos(y) for x in xs, y in ys]
            itp = convolution_interpolation((xs, ys), zs; kernel=kernel,
                      derivative=(-1, 1), fast=false, bc=:poly)
            ax = itp.itp.anchor[1]
            test_pts = interior_pts_2d_integral(collect(xs), collect(ys))
            err = maximum(abs(itp(p...) - mixed_exact_intx_dy(ax, p...)) for p in test_pts)
            push!(errs, err)
        end
        min_ratio = 2.0^(expected_order_mixed[kernel] - 1)
        @test errs[1] / errs[2] > min_ratio
        @test errs[2] / errs[3] > min_ratio
    end
end

# ── 2D slow: (1, -1) ─────────────────────────────────────────────────

@testset "2D mixed (1,-1) convergence direct" begin
    for kernel in mixed_kernels
        println("Testing 2D mixed (1,-1) direct: ", kernel)
        errs = Float64[]
        for n in [20, 40, 80]
            xs = range(0.0, 2π, length=n); ys = range(0.0, 2π, length=n)
            zs = [sin(x)*cos(y) for x in xs, y in ys]
            itp = convolution_interpolation((xs, ys), zs; kernel=kernel,
                      derivative=(1, -1), fast=false, bc=:poly)
            ay = itp.itp.anchor[2]
            test_pts = interior_pts_2d_integral(collect(xs), collect(ys))
            err = maximum(abs(itp(p...) - mixed_exact_dx_inty(ay, p...)) for p in test_pts)
            push!(errs, err)
        end
        min_ratio = 2.0^(expected_order_mixed[kernel] - 1)
        @test errs[1] / errs[2] > min_ratio
        @test errs[2] / errs[3] > min_ratio
    end
end

# ── 2D slow: anchor is zero ───────────────────────────────────────────

@testset "2D mixed anchor is zero direct" begin
    for kernel in mixed_kernels
        xs = range(0.0, 2π, length=20); ys = range(0.0, 2π, length=20)
        zs = [sin(x)*cos(y) for x in xs, y in ys]
        for deriv in [(-1,0), (0,-1), (-1,1), (1,-1)]
            itp = convolution_interpolation((xs, ys), zs; kernel=kernel,
                      derivative=deriv, fast=false, bc=:poly)
            ax, ay = itp.itp.anchor
            # evaluate at anchor point: integral dim should be zero there
            @test abs(itp(ax, ay)) < 1e-10
        end
    end
end

# ── 2D fast: (-1, 0) ─────────────────────────────────────────────────

@testset "2D mixed (-1,0) convergence fast" begin
    for kernel in mixed_kernels
        println("Testing 2D mixed (-1,0) fast: ", kernel)
        errs = Float64[]
        for n in [20, 40, 80]
            xs = range(0.0, 2π, length=n); ys = range(0.0, 2π, length=n)
            zs = [sin(x)*cos(y) for x in xs, y in ys]
            itp = convolution_interpolation((xs, ys), zs; kernel=kernel,
                      derivative=(-1, 0), fast=true, bc=:poly)
            ax = itp.itp.anchor[1]
            test_pts = interior_pts_2d_integral(collect(xs), collect(ys))
            err = maximum(abs(itp(p...) - mixed_exact_int_x(ax, p...)) for p in test_pts)
            push!(errs, err)
        end
        min_ratio = 2.0^(expected_order_mixed[kernel] - 1)
        @test errs[1] / errs[2] > min_ratio
        @test errs[2] / errs[3] > min_ratio
    end
end

# ── 2D fast: (0, -1) ─────────────────────────────────────────────────

@testset "2D mixed (0,-1) convergence fast" begin
    for kernel in mixed_kernels
        println("Testing 2D mixed (0,-1) fast: ", kernel)
        errs = Float64[]
        for n in [20, 40, 80]
            xs = range(0.0, 2π, length=n); ys = range(0.0, 2π, length=n)
            zs = [sin(x)*cos(y) for x in xs, y in ys]
            itp = convolution_interpolation((xs, ys), zs; kernel=kernel,
                      derivative=(0, -1), fast=true, bc=:poly)
            ay = itp.itp.anchor[2]
            test_pts = interior_pts_2d_integral(collect(xs), collect(ys))
            err = maximum(abs(itp(p...) - mixed_exact_int_y(ay, p...)) for p in test_pts)
            push!(errs, err)
        end
        min_ratio = 2.0^(expected_order_mixed[kernel] - 1)
        @test errs[1] / errs[2] > min_ratio
        @test errs[2] / errs[3] > min_ratio
    end
end

# ── 2D fast: (-1, 1) ─────────────────────────────────────────────────

@testset "2D mixed (-1,1) convergence fast" begin
    for kernel in mixed_kernels
        println("Testing 2D mixed (-1,1) fast: ", kernel)
        errs = Float64[]
        for n in [20, 40, 80]
            xs = range(0.0, 2π, length=n); ys = range(0.0, 2π, length=n)
            zs = [sin(x)*cos(y) for x in xs, y in ys]
            itp = convolution_interpolation((xs, ys), zs; kernel=kernel,
                      derivative=(-1, 1), fast=true, bc=:poly)
            ax = itp.itp.anchor[1]
            test_pts = interior_pts_2d_integral(collect(xs), collect(ys))
            err = maximum(abs(itp(p...) - mixed_exact_intx_dy(ax, p...)) for p in test_pts)
            push!(errs, err)
        end
        min_ratio = 2.0^(expected_order_mixed[kernel] - 1)
        @test errs[1] / errs[2] > min_ratio
        @test errs[2] / errs[3] > min_ratio
    end
end

# ── 2D fast: (1, -1) ─────────────────────────────────────────────────

@testset "2D mixed (1,-1) convergence fast" begin
    for kernel in mixed_kernels
        println("Testing 2D mixed (1,-1) fast: ", kernel)
        errs = Float64[]
        for n in [20, 40, 80]
            xs = range(0.0, 2π, length=n); ys = range(0.0, 2π, length=n)
            zs = [sin(x)*cos(y) for x in xs, y in ys]
            itp = convolution_interpolation((xs, ys), zs; kernel=kernel,
                      derivative=(1, -1), fast=true, bc=:poly)
            ay = itp.itp.anchor[2]
            test_pts = interior_pts_2d_integral(collect(xs), collect(ys))
            err = maximum(abs(itp(p...) - mixed_exact_dx_inty(ay, p...)) for p in test_pts)
            push!(errs, err)
        end
        min_ratio = 2.0^(expected_order_mixed[kernel] - 1)
        @test errs[1] / errs[2] > min_ratio
        @test errs[2] / errs[3] > min_ratio
    end
end

# ── 2D fast: anchor is zero ───────────────────────────────────────────

@testset "2D mixed anchor is zero fast" begin
    for kernel in mixed_kernels
        xs = range(0.0, 2π, length=20); ys = range(0.0, 2π, length=20)
        zs = [sin(x)*cos(y) for x in xs, y in ys]
        for deriv in [(-1,0), (0,-1), (-1,1), (1,-1)]
            itp = convolution_interpolation((xs, ys), zs; kernel=kernel,
                      derivative=deriv, fast=true, bc=:poly)
            ax, ay = itp.itp.anchor
            @test abs(itp(ax, ay)) < 1e-10
        end
    end
end

# ── 2D fast vs direct agreement ──────────────────────────────────────

@testset "2D mixed fast vs direct agreement" begin
    for kernel in mixed_kernels
        println("Testing 2D mixed fast vs direct: ", kernel)
        n = 30
        xs = range(0.0, 2π, length=n); ys = range(0.0, 2π, length=n)
        zs = [sin(x)*cos(y) for x in xs, y in ys]
        test_pts = interior_pts_2d_integral(collect(xs), collect(ys))
        for deriv in [(-1,0), (0,-1), (-1,1), (1,-1)]
            itp_fast   = convolution_interpolation((xs, ys), zs; kernel=kernel,
                             derivative=deriv, fast=true,  bc=:poly)
            itp_direct = convolution_interpolation((xs, ys), zs; kernel=kernel,
                             derivative=deriv, fast=false, bc=:poly)
            err = maximum(abs(itp_fast(p...) - itp_direct(p...)) for p in test_pts)
            @test err < 1e-6
        end
    end
end