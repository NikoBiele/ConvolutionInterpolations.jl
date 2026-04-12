#######################################################################
### TEST ZERO ALLOCATIONS                                            ###
#######################################################################

println("Testing zero allocations across all functors...")

xs = range(0.0, 2π, length=10)
ys = range(0.0, 2π, length=10)
zs = range(0.0, 2π, length=10)
ws = range(0.0, 2π, length=10)

x0, y0, z0, w0 = 1.5, 1.6, 1.7, 1.8

vs1 = sin.(xs)
vs2 = [sin(x)*cos(y) for x in xs, y in ys]
vs3 = [sin(x)*cos(y)*exp(z/4) for x in xs, y in ys, z in zs]
vs4 = [sin(x)*cos(y)*exp(z/4)*sin(w) for x in xs, y in ys, z in zs, w in ws]

println("Testing 1D kernels")
@testset "Zero allocations — 1D kernels" begin
    for kernel in [:a0, :a1, :a3, :b5]
        for fast in [true]
            for lazy in [false, true]
                itp = convolution_interpolation(xs, vs1; kernel=kernel,
                            fast=fast, lazy=lazy)
                @test @allocated(itp(x0)) == 0
            end
        end
    end
end

println("Testing 1D derivatives")
@testset "Zero allocations — 1D derivatives" begin
    for deriv in [0, 1, -1]
        for fast in [true]
            for lazy in [false, true]
                itp = convolution_interpolation(xs, vs1; kernel=:b5, derivative=deriv,
                            fast=fast, lazy=deriv == -1 ? false : lazy)
                @test @allocated(itp(x0)) == 0
            end
        end
    end
end

println("Testing 2D kernels")
@testset "Zero allocations — 2D kernels" begin
    for kernel in [:a0, :a1, :a3, :b5]
        for fast in [true]
            for lazy in [false, true]
                itp = convolution_interpolation((xs,ys), vs2; kernel=kernel,
                            fast=fast, lazy=lazy)
                @test @allocated(itp(x0,y0)) == 0
            end
        end
    end
end

println("Testing 2D per-dim kernels")
@testset "Zero allocations — 2D per-dim kernels" begin
    for (k1,k2) in [(:a3,:b5), (:b5,:a3), (:a0,:a1), (:a1,:a0)]
        for fast in [true]
            for lazy in [false, true]
                itp = convolution_interpolation((xs,ys), vs2; kernel=(k1,k2),
                            fast=fast, lazy=lazy)
                @test @allocated(itp(x0,y0)) == 0
            end
        end
    end
end

println("Testing 2D per-dim derivatives")
@testset "Zero allocations — 2D per-dim derivatives" begin
    for deriv in [(1,0), (0,1), (1,1), (2,0), (0,2), (-1,0), (0,-1), (-1,1), (1,-1), (-1,-1)]
        for fast in [true]
            for lazy in [false, true]
                itp = convolution_interpolation((xs,ys), vs2; kernel=:b5,
                            fast=fast, lazy=-1 in deriv ? false : lazy, derivative=deriv)
                @test @allocated(itp(x0,y0)) == 0
            end
        end
    end
end

println("Testing 3D kernels")
@testset "Zero allocations — 3D kernels" begin
    for kernel in [:a0, :a1, :a3, :b5]
        for fast in [true]
            for lazy in [false, true]
                itp = convolution_interpolation((xs,ys,zs), vs3; kernel=kernel,
                            fast=fast, lazy=lazy)
                @test @allocated(itp(x0,y0,z0)) == 0
            end
        end
    end
end

println("Testing 3D per-dim derivatives")
@testset "Zero allocations — 3D per-dim derivatives" begin
    for deriv in [(1,0,0), (0,1,0), (0,0,1), (1,1,0), (-1,0,0), (0,-1,0),
                  (-1,-1,0), (-1,0,-1), (0,-1,-1), (-1,1,0), (-1,0,1),
                  (-1,-1,-1)]
        for fast in [true]
            for lazy in [false, true]
                itp = convolution_interpolation((xs,ys,zs), vs3; kernel=:b5,
                        derivative=deriv, lazy=-1 in deriv ? false : lazy, fast=fast)
                @test @allocated(itp(x0,y0,z0)) == 0
            end
        end
    end
end

println("Testing 4D")
@testset "Zero allocations — 4D" begin
    for deriv in [(0,0,0,0), (1,0,0,0), (-1,0,0,0),
                  (-1,-1,0,0), (-1,0,-1,0), (-1,0,0,1),
                  (-1,-1,-1,0), (-1,-1,0,1),
                  (-1,-1,-1,-1)]
        itp = convolution_interpolation((xs,ys,zs,ws), vs4; kernel=:b5,
                  derivative=deriv, lazy=false)
        @test @allocated(itp(x0,y0,z0,w0)) == 0
    end
end

println("Testing extrapolation types")
@testset "Zero allocations — extrapolation types" begin
    for et in [Throw(), Line(), Flat()]
        itp = convolution_interpolation(xs, vs1; kernel=:b5, extrap=et)
        @test @allocated(itp(x0)) == 0
    end
end