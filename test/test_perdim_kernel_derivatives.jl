kernels_higher = (:a3, :b5)  # only kernels that support derivatives
xs = range(0.0, 2π, length=30)
ys = range(0.0, 2π, length=30)
vs = [sin(x)*cos(y) for x in xs, y in ys]

println("Testing 2D per-dim kernel derivative combinations...")
@testset "2D per-dim kernel derivative combinations" begin
    for k1 in kernels_higher, k2 in kernels_higher
        println("Testing kernel combination (fast/direct): ", k1, ", ", k2)
        itp_dx  = convolution_interpolation((xs, ys), vs; kernel=(k1, k2), derivative=(1, 0))
        itp_dy  = convolution_interpolation((xs, ys), vs; kernel=(k1, k2), derivative=(0, 1))
        itp_dxy = convolution_interpolation((xs, ys), vs; kernel=(k1, k2), derivative=(1, 1))
        @test isfinite(itp_dx(1.5, 1.0))
        @test isfinite(itp_dy(1.5, 1.0))
        @test isfinite(itp_dxy(1.5, 1.0))
        itp_dx  = convolution_interpolation((xs, ys), vs; kernel=(k1, k2), derivative=(1, 0), fast=false)
        itp_dy  = convolution_interpolation((xs, ys), vs; kernel=(k1, k2), derivative=(0, 1), fast=false)
        itp_dxy = convolution_interpolation((xs, ys), vs; kernel=(k1, k2), derivative=(1, 1), fast=false)
        @test isfinite(itp_dx(1.5, 1.0))
        @test isfinite(itp_dy(1.5, 1.0))
        @test isfinite(itp_dxy(1.5, 1.0))
    end
end

zs = range(0.0, 2π, length=30)
vs3 = [sin(x)*cos(y)*sin(z) for x in xs, y in ys, z in zs]

println("Testing 3D per-dim kernel derivative combinations...")
@testset "3D per-dim kernel derivative combinations" begin
    for k1 in kernels_higher, k2 in kernels_higher, k3 in kernels_higher
        println("Testing kernel combination (fast/direct): ", k1, ", ", k2, ", ", k3)
        itp_dx   = convolution_interpolation((xs, ys, zs), vs3; kernel=(k1, k2, k3), derivative=(1, 0, 0))
        itp_dy   = convolution_interpolation((xs, ys, zs), vs3; kernel=(k1, k2, k3), derivative=(0, 1, 0))
        itp_dz   = convolution_interpolation((xs, ys, zs), vs3; kernel=(k1, k2, k3), derivative=(0, 0, 1))
        itp_dxyz = convolution_interpolation((xs, ys, zs), vs3; kernel=(k1, k2, k3), derivative=(1, 1, 1))
        @test isfinite(itp_dx(1.5, 1.0, 2.0))
        @test isfinite(itp_dy(1.5, 1.0, 2.0))
        @test isfinite(itp_dz(1.5, 1.0, 2.0))
        @test isfinite(itp_dxyz(1.5, 1.0, 2.0))
        itp_dx   = convolution_interpolation((xs, ys, zs), vs3; kernel=(k1, k2, k3), derivative=(1, 0, 0), fast=false)
        itp_dy   = convolution_interpolation((xs, ys, zs), vs3; kernel=(k1, k2, k3), derivative=(0, 1, 0), fast=false)
        itp_dz   = convolution_interpolation((xs, ys, zs), vs3; kernel=(k1, k2, k3), derivative=(0, 0, 1), fast=false)
        itp_dxyz = convolution_interpolation((xs, ys, zs), vs3; kernel=(k1, k2, k3), derivative=(1, 1, 1), fast=false)
        @test isfinite(itp_dx(1.5, 1.0, 2.0))
        @test isfinite(itp_dy(1.5, 1.0, 2.0))
        @test isfinite(itp_dz(1.5, 1.0, 2.0))
        @test isfinite(itp_dxyz(1.5, 1.0, 2.0))
    end
end