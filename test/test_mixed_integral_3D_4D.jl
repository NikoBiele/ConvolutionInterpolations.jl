#######################################################################
### TEST MIXED INTEGRAL ORDER 3D/4D (fast path only)               ###
#######################################################################

# 3D test function: f(x,y,z) = sin(x)*cos(y)*exp(z/4)
# Construct on grids (20, 40) with lazy=true for fast construction.
# Evaluate well inside domain (away from boundaries) to avoid
# boundary ghost point computation. b5 has 5+5=10 point stencil,
# so we stay at least 5 grid points from each boundary.
# A few eager (lazy=false) tests verify near-boundary behavior.

mixed_kernels_3d = [:b5]

# safe evaluation range — middle of domain, away from boundaries
# valid for n=20 (spacing ~0.33, 5 points = ~1.65 from boundary)
xs_eval = range(5*2π/19, 15*2π/19, length=21)
ys_eval = range(5*2π/19, 15*2π/19, length=21)
zs_eval = range(5*2π/19, 15*2π/19, length=21)
ws_eval = range(5*2π/19, 15*2π/19, length=21)

println("Testing mixed integral/derivative order in 3D and 4D (fast path)...")

# ── analytical solutions via differences ─────────────────────────────

exact_intx_3d(x1,x2,y,z)        = (-cos(x2)+cos(x1)) * cos(y) * exp(z/4)
exact_inty_3d(x,y1,y2,z)        = sin(x) * (sin(y2)-sin(y1)) * exp(z/4)
exact_intz_3d(x,y,z1,z2)        = sin(x) * cos(y) * 4*(exp(z2/4)-exp(z1/4))
exact_intxy(x1,x2,y1,y2,z)      = (-cos(x2)+cos(x1)) * (sin(y2)-sin(y1)) * exp(z/4)
exact_intxz(x1,x2,y,z1,z2)      = (-cos(x2)+cos(x1)) * cos(y) * 4*(exp(z2/4)-exp(z1/4))
exact_intyz(x,y1,y2,z1,z2)      = sin(x) * (sin(y2)-sin(y1)) * 4*(exp(z2/4)-exp(z1/4))
exact_intx_dy(x1,x2,y,z)        = (-cos(x2)+cos(x1)) * (-sin(y)) * exp(z/4)
exact_intx_dz(x1,x2,y,z)        = (-cos(x2)+cos(x1)) * cos(y) * exp(z/4)/4
exact_dx_inty(x,y1,y2,z)        = cos(x) * (sin(y2)-sin(y1)) * exp(z/4)
exact_intxyz(x1,x2,y1,y2,z1,z2) = (-cos(x2)+cos(x1)) * (sin(y2)-sin(y1)) * 4*(exp(z2/4)-exp(z1/4))

# fixed difference endpoints — well inside safe zone
x1d, x2d = 2.0, 3.5
y1d, y2d = 2.1, 3.4
z1d, z2d = 2.2, 3.3

# ── 3D: 1 integral dim ───────────────────────────────────────────────

@testset "3D mixed (-1,0,0) convergence fast" begin
    for kernel in mixed_kernels_3d
        println("Testing 3D mixed (-1,0,0) fast: ", kernel)
        errs = Float64[]
        for n in [20, 40]
            xs = range(0.0, 2π, length=n)
            ys = range(0.0, 2π, length=n)
            zs = range(0.0, 2π, length=n)
            vs = [sin(x)*cos(y)*exp(z/4) for x in xs, y in ys, z in zs]
            itp = convolution_interpolation((xs,ys,zs), vs; kernel=kernel,
                      derivative=(-1,0,0), fast=true, bc=:poly, lazy=true)
            err = maximum(abs(itp(x2d,y,z) - itp(x1d,y,z)
                - exact_intx_3d(x1d,x2d,y,z))
                for y in ys_eval, z in zs_eval)
            push!(errs, err)
        end
        @test errs[1] / errs[2] > 2.0^4
    end
end

@testset "3D mixed (0,-1,0) convergence fast" begin
    for kernel in mixed_kernels_3d
        println("Testing 3D mixed (0,-1,0) fast: ", kernel)
        errs = Float64[]
        for n in [20, 40]
            xs = range(0.0, 2π, length=n)
            ys = range(0.0, 2π, length=n)
            zs = range(0.0, 2π, length=n)
            vs = [sin(x)*cos(y)*exp(z/4) for x in xs, y in ys, z in zs]
            itp = convolution_interpolation((xs,ys,zs), vs; kernel=kernel,
                      derivative=(0,-1,0), fast=true, bc=:poly, lazy=true)
            err = maximum(abs(itp(x,y2d,z) - itp(x,y1d,z)
                - exact_inty_3d(x,y1d,y2d,z))
                for x in xs_eval, z in zs_eval)
            push!(errs, err)
        end
        @test errs[1] / errs[2] > 2.0^4
    end
end

@testset "3D mixed (0,0,-1) convergence fast" begin
    for kernel in mixed_kernels_3d
        println("Testing 3D mixed (0,0,-1) fast: ", kernel)
        errs = Float64[]
        for n in [20, 40]
            xs = range(0.0, 2π, length=n)
            ys = range(0.0, 2π, length=n)
            zs = range(0.0, 2π, length=n)
            vs = [sin(x)*cos(y)*exp(z/4) for x in xs, y in ys, z in zs]
            itp = convolution_interpolation((xs,ys,zs), vs; kernel=kernel,
                      derivative=(0,0,-1), fast=true, bc=:poly, lazy=true)
            err = maximum(abs(itp(x,y,z2d) - itp(x,y,z1d)
                - exact_intz_3d(x,y,z1d,z2d))
                for x in xs_eval, y in ys_eval)
            push!(errs, err)
        end
        @test errs[1] / errs[2] > 2.0^4
    end
end

@testset "3D mixed (-1,1,0) convergence fast" begin
    for kernel in mixed_kernels_3d
        println("Testing 3D mixed (-1,1,0) fast: ", kernel)
        errs = Float64[]
        for n in [20, 40]
            xs = range(0.0, 2π, length=n)
            ys = range(0.0, 2π, length=n)
            zs = range(0.0, 2π, length=n)
            vs = [sin(x)*cos(y)*exp(z/4) for x in xs, y in ys, z in zs]
            itp = convolution_interpolation((xs,ys,zs), vs; kernel=kernel,
                      derivative=(-1,1,0), fast=true, bc=:poly, lazy=true)
            err = maximum(abs(itp(x2d,y,z) - itp(x1d,y,z)
                - exact_intx_dy(x1d,x2d,y,z))
                for y in ys_eval, z in zs_eval)
            push!(errs, err)
        end
        @test errs[1] / errs[2] > 2.0^4
    end
end

@testset "3D mixed (-1,0,1) convergence fast" begin
    for kernel in mixed_kernels_3d
        println("Testing 3D mixed (-1,0,1) fast: ", kernel)
        errs = Float64[]
        for n in [20, 40]
            xs = range(0.0, 2π, length=n)
            ys = range(0.0, 2π, length=n)
            zs = range(0.0, 2π, length=n)
            vs = [sin(x)*cos(y)*exp(z/4) for x in xs, y in ys, z in zs]
            itp = convolution_interpolation((xs,ys,zs), vs; kernel=kernel,
                      derivative=(-1,0,1), fast=true, bc=:poly, lazy=true)
            err = maximum(abs(itp(x2d,y,z) - itp(x1d,y,z)
                - exact_intx_dz(x1d,x2d,y,z))
                for y in ys_eval, z in zs_eval)
            push!(errs, err)
        end
        @test errs[1] / errs[2] > 2.0^4
    end
end

@testset "3D mixed (1,-1,0) convergence fast" begin
    for kernel in mixed_kernels_3d
        println("Testing 3D mixed (1,-1,0) fast: ", kernel)
        errs = Float64[]
        for n in [20, 40]
            xs = range(0.0, 2π, length=n)
            ys = range(0.0, 2π, length=n)
            zs = range(0.0, 2π, length=n)
            vs = [sin(x)*cos(y)*exp(z/4) for x in xs, y in ys, z in zs]
            itp = convolution_interpolation((xs,ys,zs), vs; kernel=kernel,
                      derivative=(1,-1,0), fast=true, bc=:poly, lazy=true)
            err = maximum(abs(itp(x,y2d,z) - itp(x,y1d,z)
                - exact_dx_inty(x,y1d,y2d,z))
                for x in xs_eval, z in zs_eval)
            push!(errs, err)
        end
        @test errs[1] / errs[2] > 2.0^4
    end
end

# ── 3D: 2 integral dims ───────────────────────────────────────────────

@testset "3D mixed (-1,-1,0) convergence fast" begin
    for kernel in mixed_kernels_3d
        println("Testing 3D mixed (-1,-1,0) fast: ", kernel)
        errs = Float64[]
        for n in [20, 40]
            xs = range(0.0, 2π, length=n)
            ys = range(0.0, 2π, length=n)
            zs = range(0.0, 2π, length=n)
            vs = [sin(x)*cos(y)*exp(z/4) for x in xs, y in ys, z in zs]
            itp = convolution_interpolation((xs,ys,zs), vs; kernel=kernel,
                      derivative=(-1,-1,0), fast=true, bc=:poly, lazy=true)
            err = maximum(abs(
                itp(x2d,y2d,z) - itp(x1d,y2d,z) - itp(x2d,y1d,z) + itp(x1d,y1d,z)
                - exact_intxy(x1d,x2d,y1d,y2d,z))
                for z in zs_eval)
            push!(errs, err)
        end
        @test errs[1] / errs[2] > 2.0^4
    end
end

@testset "3D mixed (-1,0,-1) convergence fast" begin
    for kernel in mixed_kernels_3d
        println("Testing 3D mixed (-1,0,-1) fast: ", kernel)
        errs = Float64[]
        for n in [20, 40]
            xs = range(0.0, 2π, length=n)
            ys = range(0.0, 2π, length=n)
            zs = range(0.0, 2π, length=n)
            vs = [sin(x)*cos(y)*exp(z/4) for x in xs, y in ys, z in zs]
            itp = convolution_interpolation((xs,ys,zs), vs; kernel=kernel,
                      derivative=(-1,0,-1), fast=true, bc=:poly, lazy=true)
            err = maximum(abs(
                itp(x2d,y,z2d) - itp(x1d,y,z2d) - itp(x2d,y,z1d) + itp(x1d,y,z1d)
                - exact_intxz(x1d,x2d,y,z1d,z2d))
                for y in ys_eval)
            push!(errs, err)
        end
        @test errs[1] / errs[2] > 2.0^4
    end
end

@testset "3D mixed (0,-1,-1) convergence fast" begin
    for kernel in mixed_kernels_3d
        println("Testing 3D mixed (0,-1,-1) fast: ", kernel)
        errs = Float64[]
        for n in [20, 40]
            xs = range(0.0, 2π, length=n)
            ys = range(0.0, 2π, length=n)
            zs = range(0.0, 2π, length=n)
            vs = [sin(x)*cos(y)*exp(z/4) for x in xs, y in ys, z in zs]
            itp = convolution_interpolation((xs,ys,zs), vs; kernel=kernel,
                      derivative=(0,-1,-1), fast=true, bc=:poly, lazy=true)
            err = maximum(abs(
                itp(x,y2d,z2d) - itp(x,y1d,z2d) - itp(x,y2d,z1d) + itp(x,y1d,z1d)
                - exact_intyz(x,y1d,y2d,z1d,z2d))
                for x in xs_eval)
            push!(errs, err)
        end
        @test errs[1] / errs[2] > 2.0^4
    end
end

# ── 3D: 3 integral dims ───────────────────────────────────────────────

@testset "3D mixed (-1,-1,-1) convergence fast" begin
    for kernel in mixed_kernels_3d
        println("Testing 3D mixed (-1,-1,-1) fast: ", kernel)
        errs = Float64[]
        for n in [20, 40]
            xs = range(0.0, 2π, length=n)
            ys = range(0.0, 2π, length=n)
            zs = range(0.0, 2π, length=n)
            vs = [sin(x)*cos(y)*exp(z/4) for x in xs, y in ys, z in zs]
            itp = convolution_interpolation((xs,ys,zs), vs; kernel=kernel,
                      derivative=(-1,-1,-1), fast=true, bc=:poly, lazy=true)
            err = abs(
                itp(x2d,y2d,z2d) - itp(x1d,y2d,z2d) - itp(x2d,y1d,z2d) + itp(x1d,y1d,z2d)
              - itp(x2d,y2d,z1d) + itp(x1d,y2d,z1d) + itp(x2d,y1d,z1d) - itp(x1d,y1d,z1d)
              - exact_intxyz(x1d,x2d,y1d,y2d,z1d,z2d))
            push!(errs, err)
        end
        @test errs[1] / errs[2] > 2.0^4
    end
end

# ── 3D: eager boundary tests ─────────────────────────────────────────

@testset "3D mixed eager near boundary" begin
    for kernel in mixed_kernels_3d
        n = 20
        xs = range(0.0, 2π, length=n)
        ys = range(0.0, 2π, length=n)
        zs = range(0.0, 2π, length=n)
        vs = [sin(x)*cos(y)*exp(z/4) for x in xs, y in ys, z in zs]
        for deriv in [(-1,0,0), (0,-1,0), (0,0,-1), (-1,-1,0), (-1,-1,-1)]
            itp = convolution_interpolation((xs,ys,zs), vs; kernel=kernel,
                      derivative=deriv, fast=true, bc=:poly, lazy=false)
            # evaluate near but not at boundary — 2 grid spacings in
            h = xs[2] - xs[1]
            result = itp(xs[1]+2h, ys[1]+2h, zs[1]+2h)
            @test result isa Float64
            @test isfinite(result)
        end
    end
end

# ── 3D anchor is zero ─────────────────────────────────────────────────

@testset "3D mixed anchor is zero fast" begin
    for kernel in mixed_kernels_3d
        xs = range(0.0, 2π, length=20)
        ys = range(0.0, 2π, length=20)
        zs = range(0.0, 2π, length=20)
        vs = [sin(x)*cos(y)*exp(z/4) for x in xs, y in ys, z in zs]
        for deriv in [(-1,0,0), (0,-1,0), (0,0,-1),
                      (-1,-1,0), (-1,0,-1), (0,-1,-1),
                      (-1,1,0), (-1,0,1), (1,-1,0),
                      (-1,-1,-1)]
            itp = convolution_interpolation((xs,ys,zs), vs; kernel=kernel,
                      derivative=deriv, fast=true, bc=:poly, lazy=true)
            ax, ay, az = itp.itp.anchor
            @test abs(itp(ax, ay, az)) < 1e-10
        end
    end
end

# ── 4D ───────────────────────────────────────────────────────────────
# f(x,y,z,w) = sin(x)*cos(y)*exp(z/4)*sin(w)

println("Testing 4D mixed integral/derivative (fast path)...")

@testset "4D mixed construction and callability" begin
    n = 20
    xs = range(0.0, 2π, length=n)
    ys = range(0.0, 2π, length=n)
    zs = range(0.0, 2π, length=n)
    ws = range(0.0, 2π, length=n)
    vs = [sin(x)*cos(y)*exp(z/4)*sin(w) for x in xs, y in ys, z in zs, w in ws]
    for deriv in [(-1,0,0,0), (0,-1,0,0), (0,0,-1,0), (0,0,0,-1),
                  (-1,-1,0,0), (-1,0,-1,0), (-1,0,0,1),
                  (-1,-1,-1,0), (-1,-1,0,1),
                  (-1,-1,-1,-1)]
        itp = convolution_interpolation((xs,ys,zs,ws), vs;
                  kernel=:b5, derivative=deriv, fast=true, bc=:poly, lazy=true)
        @test itp isa ConvolutionExtrapolation
        result = itp(x1d, y1d, z1d, 2.0)
        @test result isa Float64
        @test isfinite(result)
    end
end

@testset "4D mixed (-1,-1,-1,0) numerical fast" begin
    n = 20
    xs = range(0.0, 2π, length=n)
    ys = range(0.0, 2π, length=n)
    zs = range(0.0, 2π, length=n)
    ws = range(0.0, 2π, length=n)
    vs = [sin(x)*cos(y)*exp(z/4)*sin(w) for x in xs, y in ys, z in zs, w in ws]
    itp = convolution_interpolation((xs,ys,zs,ws), vs;
              kernel=:b5, derivative=(-1,-1,-1,0), fast=true, bc=:poly, lazy=true)
    err = maximum(abs(
        itp(x2d,y2d,z2d,w) - itp(x1d,y2d,z2d,w) - itp(x2d,y1d,z2d,w) + itp(x1d,y1d,z2d,w)
      - itp(x2d,y2d,z1d,w) + itp(x1d,y2d,z1d,w) + itp(x2d,y1d,z1d,w) - itp(x1d,y1d,z1d,w)
      - exact_intxyz(x1d,x2d,y1d,y2d,z1d,z2d) * sin(w))
        for w in ws_eval)
    @test err < 1e-3
end

@testset "4D mixed (-1,0,0,1) numerical fast" begin
    n = 20
    xs = range(0.0, 2π, length=n)
    ys = range(0.0, 2π, length=n)
    zs = range(0.0, 2π, length=n)
    ws = range(0.0, 2π, length=n)
    vs = [sin(x)*cos(y)*exp(z/4)*sin(w) for x in xs, y in ys, z in zs, w in ws]
    itp = convolution_interpolation((xs,ys,zs,ws), vs;
              kernel=:b5, derivative=(-1,0,0,1), fast=true, bc=:poly, lazy=true)
    for (y,z,w) in [(2.1, 2.2, 2.3), (2.5, 2.7, 2.9), (3.0, 3.1, 3.2)]
        numerical  = itp(x2d,y,z,w) - itp(x1d,y,z,w)
        analytical = (-cos(x2d)+cos(x1d)) * cos(y) * exp(z/4) * cos(w)
        @test abs(numerical - analytical) < 1e-3
    end
end

#################

# ── 3D: direct path convergence ──────────────────────────────────────

@testset "3D mixed (-1,0,0) convergence direct" begin
    for kernel in mixed_kernels_3d
        println("Testing 3D mixed (-1,0,0) direct: ", kernel)
        errs = Float64[]
        for n in [20, 40]
            xs = range(0.0, 2π, length=n)
            ys = range(0.0, 2π, length=n)
            zs = range(0.0, 2π, length=n)
            vs = [sin(x)*cos(y)*exp(z/4) for x in xs, y in ys, z in zs]
            itp = convolution_interpolation((xs,ys,zs), vs; kernel=kernel,
                      derivative=(-1,0,0), fast=false, bc=:poly, lazy=true)
            err = maximum(abs(itp(x2d,y,z) - itp(x1d,y,z)
                - exact_intx_3d(x1d,x2d,y,z))
                for y in ys_eval, z in zs_eval)
            push!(errs, err)
        end
        @test errs[1] / errs[2] > 2.0^4
    end
end

@testset "3D mixed (0,-1,0) convergence direct" begin
    for kernel in mixed_kernels_3d
        println("Testing 3D mixed (0,-1,0) direct: ", kernel)
        errs = Float64[]
        for n in [20, 40]
            xs = range(0.0, 2π, length=n)
            ys = range(0.0, 2π, length=n)
            zs = range(0.0, 2π, length=n)
            vs = [sin(x)*cos(y)*exp(z/4) for x in xs, y in ys, z in zs]
            itp = convolution_interpolation((xs,ys,zs), vs; kernel=kernel,
                      derivative=(0,-1,0), fast=false, bc=:poly, lazy=true)
            err = maximum(abs(itp(x,y2d,z) - itp(x,y1d,z)
                - exact_inty_3d(x,y1d,y2d,z))
                for x in xs_eval, z in zs_eval)
            push!(errs, err)
        end
        @test errs[1] / errs[2] > 2.0^4
    end
end

@testset "3D mixed (0,0,-1) convergence direct" begin
    for kernel in mixed_kernels_3d
        println("Testing 3D mixed (0,0,-1) direct: ", kernel)
        errs = Float64[]
        for n in [20, 40]
            xs = range(0.0, 2π, length=n)
            ys = range(0.0, 2π, length=n)
            zs = range(0.0, 2π, length=n)
            vs = [sin(x)*cos(y)*exp(z/4) for x in xs, y in ys, z in zs]
            itp = convolution_interpolation((xs,ys,zs), vs; kernel=kernel,
                      derivative=(0,0,-1), fast=false, bc=:poly, lazy=true)
            err = maximum(abs(itp(x,y,z2d) - itp(x,y,z1d)
                - exact_intz_3d(x,y,z1d,z2d))
                for x in xs_eval, y in ys_eval)
            push!(errs, err)
        end
        @test errs[1] / errs[2] > 2.0^4
    end
end

@testset "3D mixed (-1,1,0) convergence direct" begin
    for kernel in mixed_kernels_3d
        println("Testing 3D mixed (-1,1,0) direct: ", kernel)
        errs = Float64[]
        for n in [20, 40]
            xs = range(0.0, 2π, length=n)
            ys = range(0.0, 2π, length=n)
            zs = range(0.0, 2π, length=n)
            vs = [sin(x)*cos(y)*exp(z/4) for x in xs, y in ys, z in zs]
            itp = convolution_interpolation((xs,ys,zs), vs; kernel=kernel,
                      derivative=(-1,1,0), fast=false, bc=:poly, lazy=true)
            err = maximum(abs(itp(x2d,y,z) - itp(x1d,y,z)
                - exact_intx_dy(x1d,x2d,y,z))
                for y in ys_eval, z in zs_eval)
            push!(errs, err)
        end
        @test errs[1] / errs[2] > 2.0^4
    end
end

@testset "3D mixed (-1,0,1) convergence direct" begin
    for kernel in mixed_kernels_3d
        println("Testing 3D mixed (-1,0,1) direct: ", kernel)
        errs = Float64[]
        for n in [20, 40]
            xs = range(0.0, 2π, length=n)
            ys = range(0.0, 2π, length=n)
            zs = range(0.0, 2π, length=n)
            vs = [sin(x)*cos(y)*exp(z/4) for x in xs, y in ys, z in zs]
            itp = convolution_interpolation((xs,ys,zs), vs; kernel=kernel,
                      derivative=(-1,0,1), fast=false, bc=:poly, lazy=true)
            err = maximum(abs(itp(x2d,y,z) - itp(x1d,y,z)
                - exact_intx_dz(x1d,x2d,y,z))
                for y in ys_eval, z in zs_eval)
            push!(errs, err)
        end
        @test errs[1] / errs[2] > 2.0^4
    end
end

@testset "3D mixed (1,-1,0) convergence direct" begin
    for kernel in mixed_kernels_3d
        println("Testing 3D mixed (1,-1,0) direct: ", kernel)
        errs = Float64[]
        for n in [20, 40]
            xs = range(0.0, 2π, length=n)
            ys = range(0.0, 2π, length=n)
            zs = range(0.0, 2π, length=n)
            vs = [sin(x)*cos(y)*exp(z/4) for x in xs, y in ys, z in zs]
            itp = convolution_interpolation((xs,ys,zs), vs; kernel=kernel,
                      derivative=(1,-1,0), fast=false, bc=:poly, lazy=true)
            err = maximum(abs(itp(x,y2d,z) - itp(x,y1d,z)
                - exact_dx_inty(x,y1d,y2d,z))
                for x in xs_eval, z in zs_eval)
            push!(errs, err)
        end
        @test errs[1] / errs[2] > 2.0^4
    end
end

@testset "3D mixed (-1,-1,0) convergence direct" begin
    for kernel in mixed_kernels_3d
        println("Testing 3D mixed (-1,-1,0) direct: ", kernel)
        errs = Float64[]
        for n in [20, 40]
            xs = range(0.0, 2π, length=n)
            ys = range(0.0, 2π, length=n)
            zs = range(0.0, 2π, length=n)
            vs = [sin(x)*cos(y)*exp(z/4) for x in xs, y in ys, z in zs]
            itp = convolution_interpolation((xs,ys,zs), vs; kernel=kernel,
                      derivative=(-1,-1,0), fast=false, bc=:poly, lazy=true)
            err = maximum(abs(
                itp(x2d,y2d,z) - itp(x1d,y2d,z) - itp(x2d,y1d,z) + itp(x1d,y1d,z)
                - exact_intxy(x1d,x2d,y1d,y2d,z))
                for z in zs_eval)
            push!(errs, err)
        end
        @test errs[1] / errs[2] > 2.0^4
    end
end

@testset "3D mixed (-1,0,-1) convergence direct" begin
    for kernel in mixed_kernels_3d
        println("Testing 3D mixed (-1,0,-1) direct: ", kernel)
        errs = Float64[]
        for n in [20, 40]
            xs = range(0.0, 2π, length=n)
            ys = range(0.0, 2π, length=n)
            zs = range(0.0, 2π, length=n)
            vs = [sin(x)*cos(y)*exp(z/4) for x in xs, y in ys, z in zs]
            itp = convolution_interpolation((xs,ys,zs), vs; kernel=kernel,
                      derivative=(-1,0,-1), fast=false, bc=:poly, lazy=true)
            err = maximum(abs(
                itp(x2d,y,z2d) - itp(x1d,y,z2d) - itp(x2d,y,z1d) + itp(x1d,y,z1d)
                - exact_intxz(x1d,x2d,y,z1d,z2d))
                for y in ys_eval)
            push!(errs, err)
        end
        @test errs[1] / errs[2] > 2.0^4
    end
end

@testset "3D mixed (0,-1,-1) convergence direct" begin
    for kernel in mixed_kernels_3d
        println("Testing 3D mixed (0,-1,-1) direct: ", kernel)
        errs = Float64[]
        for n in [20, 40]
            xs = range(0.0, 2π, length=n)
            ys = range(0.0, 2π, length=n)
            zs = range(0.0, 2π, length=n)
            vs = [sin(x)*cos(y)*exp(z/4) for x in xs, y in ys, z in zs]
            itp = convolution_interpolation((xs,ys,zs), vs; kernel=kernel,
                      derivative=(0,-1,-1), fast=false, bc=:poly, lazy=true)
            err = maximum(abs(
                itp(x,y2d,z2d) - itp(x,y1d,z2d) - itp(x,y2d,z1d) + itp(x,y1d,z1d)
                - exact_intyz(x,y1d,y2d,z1d,z2d))
                for x in xs_eval)
            push!(errs, err)
        end
        @test errs[1] / errs[2] > 2.0^4
    end
end

@testset "3D mixed (-1,-1,-1) convergence direct" begin
    for kernel in mixed_kernels_3d
        println("Testing 3D mixed (-1,-1,-1) direct: ", kernel)
        errs = Float64[]
        for n in [20, 40]
            xs = range(0.0, 2π, length=n)
            ys = range(0.0, 2π, length=n)
            zs = range(0.0, 2π, length=n)
            vs = [sin(x)*cos(y)*exp(z/4) for x in xs, y in ys, z in zs]
            itp = convolution_interpolation((xs,ys,zs), vs; kernel=kernel,
                      derivative=(-1,-1,-1), fast=false, bc=:poly, lazy=true)
            err = abs(
                itp(x2d,y2d,z2d) - itp(x1d,y2d,z2d) - itp(x2d,y1d,z2d) + itp(x1d,y1d,z2d)
              - itp(x2d,y2d,z1d) + itp(x1d,y2d,z1d) + itp(x2d,y1d,z1d) - itp(x1d,y1d,z1d)
              - exact_intxyz(x1d,x2d,y1d,y2d,z1d,z2d))
            push!(errs, err)
        end
        @test errs[1] / errs[2] > 2.0^4
    end
end

# ── 3D: fast vs direct agreement ─────────────────────────────────────

@testset "3D mixed fast vs direct agreement" begin
    for kernel in mixed_kernels_3d
        println("Testing 3D mixed fast vs direct: ", kernel)
        n = 20
        xs = range(0.0, 2π, length=n)
        ys = range(0.0, 2π, length=n)
        zs = range(0.0, 2π, length=n)
        vs = [sin(x)*cos(y)*exp(z/4) for x in xs, y in ys, z in zs]
        for deriv in [(-1,0,0), (0,-1,0), (0,0,-1),
                      (-1,-1,0), (-1,0,-1), (0,-1,-1),
                      (-1,1,0), (-1,0,1), (1,-1,0),
                      (-1,-1,-1)]
            itp_fast   = convolution_interpolation((xs,ys,zs), vs; kernel=kernel,
                             derivative=deriv, fast=true,  bc=:poly, lazy=true)
            itp_direct = convolution_interpolation((xs,ys,zs), vs; kernel=kernel,
                             derivative=deriv, fast=false, bc=:poly, lazy=true)
            err = maximum(abs(itp_fast(x,y,z) - itp_direct(x,y,z))
                for x in xs_eval[5:5:15], y in ys_eval[5:5:15], z in zs_eval[5:5:15])
            @test err < 1e-6
        end
    end
end

# ── 4D: fast vs direct agreement ─────────────────────────────────────

@testset "4D mixed fast vs direct agreement" begin
    for kernel in mixed_kernels_3d
        println("Testing 4D mixed fast vs direct: ", kernel)
        n = 15
        xs = range(0.0, 2π, length=n)
        ys = range(0.0, 2π, length=n)
        zs = range(0.0, 2π, length=n)
        ws = range(0.0, 2π, length=n)
        vs = [sin(x)*cos(y)*exp(z/4)*sin(w) for x in xs, y in ys, z in zs, w in ws]
        for deriv in [(-1,0,0,0), (0,-1,0,0), (-1,-1,0,0),
                      (-1,-1,-1,0), (-1,-1,-1,-1)]
            itp_fast   = convolution_interpolation((xs,ys,zs,ws), vs; kernel=kernel,
                             derivative=deriv, fast=true,  bc=:poly, lazy=true)
            itp_direct = convolution_interpolation((xs,ys,zs,ws), vs; kernel=kernel,
                             derivative=deriv, fast=false, bc=:poly, lazy=true)
            for (x,y,z,w) in [(x1d,y1d,z1d,2.0), (x2d,y2d,z2d,2.5), (2.8,2.9,3.0,3.1)]
                @test abs(itp_fast(x,y,z,w) - itp_direct(x,y,z,w)) < 1e-6
            end
        end
    end
end