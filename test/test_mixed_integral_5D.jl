# ── 5D mixed: exercises the N-integral-dims functor ──────────────────
# f(x,y,z,w,v) = sin(x)*cos(y)*exp(z/4)*sin(w)*cos(v)
# With derivative=(-1,-1,-1,-1,0), the 5th dim is interpolation,
# first 4 are integrals → hits the generic N-integral functor.

println("Testing 5D mixed integral/derivative (fast path, N-integral functor)...")

@testset "5D mixed construction and callability" begin
    n = 10
    xs = range(0.0, 2π, length=n)
    ys = range(0.0, 2π, length=n)
    zs = range(0.0, 2π, length=n)
    ws = range(0.0, 2π, length=n)
    us = range(0.0, 2π, length=n)
    vs = [sin(x)*cos(y)*exp(z/4)*sin(w)*cos(u) for x in xs, y in ys, z in zs, w in ws, u in us]
    for deriv in [(-1,-1,-1,-1,0), (-1,-1,-1,0,-1), (-1,-1,0,-1,-1)]
        itp = convolution_interpolation((xs,ys,zs,ws,us), vs;
                  kernel=:b5, derivative=deriv, fast=true, bc=:poly, lazy=false)
        @test itp isa ConvolutionExtrapolation
        result = itp(2.0, 2.1, 2.2, 2.3, 2.4)
        @test result isa Float64
        @test isfinite(result)
    end
end

@testset "5D mixed (-1,-1,-1,-1,0) numerical fast" begin
    n = 10
    xs = range(0.0, 2π, length=n)
    ys = range(0.0, 2π, length=n)
    zs = range(0.0, 2π, length=n)
    ws = range(0.0, 2π, length=n)
    us = range(0.0, 2π, length=n)
    vs = [sin(x)*cos(y)*exp(z/4)*sin(w)*cos(u) for x in xs, y in ys, z in zs, w in ws, u in us]
    itp = convolution_interpolation((xs,ys,zs,ws,us), vs;
              kernel=:b5, derivative=(-1,-1,-1,-1,0), fast=true, bc=:poly, lazy=false)
    # Use differences to cancel anchors:
    # ∫∫∫∫ f dx dy dz dw = (-cos(x)+C1)*(sin(y)+C2)*4*(exp(z/4)+C3)*(-cos(w)+C4)*cos(u)
    # 4-fold difference eliminates all constants
    x1, x2 = 2.0, 3.5
    y1, y2 = 2.1, 3.4
    z1, z2 = 2.2, 3.3
    w1, w2 = 2.3, 3.2
    for u in [2.0, 3.0, 4.0]
        numerical = (
            itp(x2,y2,z2,w2,u) - itp(x1,y2,z2,w2,u)
          - itp(x2,y1,z2,w2,u) + itp(x1,y1,z2,w2,u)
          - itp(x2,y2,z1,w2,u) + itp(x1,y2,z1,w2,u)
          + itp(x2,y1,z1,w2,u) - itp(x1,y1,z1,w2,u)
          - itp(x2,y2,z2,w1,u) + itp(x1,y2,z2,w1,u)
          + itp(x2,y1,z2,w1,u) - itp(x1,y1,z2,w1,u)
          + itp(x2,y2,z1,w1,u) - itp(x1,y2,z1,w1,u)
          - itp(x2,y1,z1,w1,u) + itp(x1,y1,z1,w1,u)
        )
        analytical = (-cos(x2)+cos(x1)) * (sin(y2)-sin(y1)) *
                     4*(exp(z2/4)-exp(z1/4)) * (-cos(w2)+cos(w1)) * cos(u)
        println("error $(abs(numerical - analytical))")
        @test abs(numerical - analytical) < 1e-3  # coarse grid, relaxed tolerance
    end
end

@testset "5D mixed (-1,-1,-1,-1,0) fast vs direct agreement" begin
    n = 10
    xs = range(0.0, 2π, length=n)
    ys = range(0.0, 2π, length=n)
    zs = range(0.0, 2π, length=n)
    ws = range(0.0, 2π, length=n)
    us = range(0.0, 2π, length=n)
    vs = [sin(x)*cos(y)*exp(z/4)*sin(w)*cos(u) for x in xs, y in ys, z in zs, w in ws, u in us]
    itp_fast   = convolution_interpolation((xs,ys,zs,ws,us), vs;
                     kernel=:b5, derivative=(-1,-1,-1,-1,0), fast=true, bc=:poly, lazy=false)
    itp_direct = convolution_interpolation((xs,ys,zs,ws,us), vs;
                     kernel=:b5, derivative=(-1,-1,-1,-1,0), fast=false, bc=:poly, lazy=false)
    for (x,y,z,w,u) in [(2.0,2.1,2.2,2.3,2.4), (3.0,3.1,3.2,2.8,2.5)]
        println("error $(abs(itp_fast(x,y,z,w,u) - itp_direct(x,y,z,w,u)))")
        @test abs(itp_fast(x,y,z,w,u) - itp_direct(x,y,z,w,u)) < 1e-6
    end
end

@testset "5D mixed anchor is zero fast" begin
    n = 10
    xs = range(0.0, 2π, length=n)
    ys = range(0.0, 2π, length=n)
    zs = range(0.0, 2π, length=n)
    ws = range(0.0, 2π, length=n)
    us = range(0.0, 2π, length=n)
    vs = [sin(x)*cos(y)*exp(z/4)*sin(w)*cos(u) for x in xs, y in ys, z in zs, w in ws, u in us]
    itp = convolution_interpolation((xs,ys,zs,ws,us), vs;
              kernel=:b5, derivative=(-1,-1,-1,-1,0), fast=true, bc=:poly, lazy=false)
    anchor = itp.itp.anchor
    @test abs(itp(anchor...)) < 1e-10
end