###############################################################################
### TEST UNIFORM GRID EXTRAPOLATIONS IN 1D, 2D, 3D, 4D ########################
###############################################################################

N = 4 # number of samples in each dimension
tolerance = 1e-6 # tight tolerance
# test all kernels
kernels = [:a1, :a3, :b5] #, :a4, :a5, :a7, :b7, :b9, :b11 :b13] # only test kernels with separate dispatch
bc = :linear # control kernel boundary conditions

println("Testing uniform grid extrapolations for direct and fast kernels for 1D, 2D, 3D, 4D...")
### 1D
@testset "1D direct kernels" begin
    for kernel in kernels
        println("Testing 1D direct kernel: ", kernel)
        range_1d = range(0.0, stop=1.0, length=N)
        vals_1d_linear = range(0.0, stop=1.0, length=N)
        
        # extrapolation
        etp_1d_linear = convolution_interpolation(range_1d, vals_1d_linear; kernel=kernel, fast=false, bc=bc, extrap=:line)
        @test etp_1d_linear.([-0.2, -0.1, 1.1, 1.2]) ≈ [-0.2, -0.1, 1.1, 1.2]  atol=tolerance
        etp_1d_flat = convolution_interpolation(range_1d, vals_1d_linear; kernel=kernel, fast=false, bc=bc, extrap=:flat)
        @test etp_1d_flat.([-0.2, -0.1, 1.1, 1.2]) ≈ [0.0, 0.0, 1.0, 1.0]  atol=tolerance
    end
end
@testset "1D fast kernels" begin
    for kernel in kernels
        println("Testing 1D fast kernel: ", kernel)
        range_1d = range(0.0, stop=1.0, length=N)
        vals_1d_linear = range(0.0, stop=1.0, length=N)

        # extrapolation
        etp_1d_linear = convolution_interpolation(range_1d, vals_1d_linear; kernel=kernel, fast=true, bc=bc, extrap=:line)
        @test etp_1d_linear.([-0.2, -0.1, 1.1, 1.2]) ≈ [-0.2, -0.1, 1.1, 1.2]  atol=tolerance
        etp_1d_flat = convolution_interpolation(range_1d, vals_1d_linear; kernel=kernel, fast=true, bc=bc, extrap=:flat)
        @test etp_1d_flat.([-0.2, -0.1, 1.1, 1.2]) ≈ [0.0, 0.0, 1.0, 1.0]  atol=tolerance
    end
end

### 2D
@testset "2D direct kernels" begin
    for kernel in kernels
        println("Testing 2D direct kernel: ", kernel)
        range_2d = range(0.0, stop=1.0, length=N)
        f_2d_linear(i,j) = (i-1)/(N-1)+(j-1)/(N-1)
        vals_2d_linear = [f_2d_linear(i,j) for i in 1:N, j in 1:N]
            
        # extrapolation
        etp_2d_linear = convolution_interpolation((range_2d, range_2d), vals_2d_linear; kernel=kernel, fast=false, bc=bc, extrap=:line)
        @test [etp_2d_linear(i, i) for i in [-0.2, -0.1, 1.1, 1.2]] ≈ [-0.4, -0.2, 2.2, 2.4]  atol=tolerance
        etp_2d_flat = convolution_interpolation((range_2d, range_2d,), vals_2d_linear; kernel=kernel, fast=false, bc=bc, extrap=:flat)
        @test [etp_2d_flat(i, i) for i in [-0.2, -0.1, 1.1, 1.2]] ≈ [0.0, 0.0, 2.0, 2.0]  atol=tolerance
    end
end
@testset "2D fast kernels" begin
    for kernel in kernels
        println("Testing 2D fast kernel: ", kernel)
        range_2d = range(0.0, stop=1.0, length=N)
        f_2d_linear(i,j) = (i-1)/(N-1)+(j-1)/(N-1)
        vals_2d_linear = [f_2d_linear(i,j) for i in 1:N, j in 1:N]
            
        # extrapolation
        etp_2d_linear = convolution_interpolation((range_2d, range_2d), vals_2d_linear; kernel=kernel, fast=true, bc=bc, extrap=:line)
        @test [etp_2d_linear(i, i) for i in [-0.2, -0.1, 1.1, 1.2]] ≈ [-0.4, -0.2, 2.2, 2.4]  atol=tolerance
        etp_2d_flat = convolution_interpolation((range_2d, range_2d,), vals_2d_linear; kernel=kernel, fast=true, bc=bc, extrap=:flat)
        @test [etp_2d_flat(i, i) for i in [-0.2, -0.1, 1.1, 1.2]] ≈ [0.0, 0.0, 2.0, 2.0]  atol=tolerance
    end
end

### 3D
@testset "3D direct kernels" begin
    for kernel in kernels
        println("Testing 3D direct kernel: ", kernel)
        range_3d = range(0.0, stop=1.0, length=N)
        f_3d_linear(i,j,k) = (i-1)/(N-1)+(j-1)/(N-1)+(k-1)/(N-1)
        vals_3d_linear = Array{Float64}(undef, N, N, N)
        for i in 1:N, j in 1:N, k in 1:N
            vals_3d_linear[i,j,k] = f_3d_linear(i,j,k)
        end

        # extrapolation
        etp_3d_linear = convolution_interpolation((range_3d, range_3d, range_3d), vals_3d_linear; kernel=kernel, fast=false, bc=bc, extrap=:line)
        @test isapprox([etp_3d_linear(i, i, i) for i in [-0.2, -0.1, 1.1, 1.2]], [-0.6, -0.3, 3.3, 3.6], atol=1e-3)  atol=tolerance
        etp_3d_flat = convolution_interpolation((range_3d, range_3d, range_3d), vals_3d_linear; kernel=kernel, fast=false, bc=bc, extrap=:flat)
        @test [etp_3d_flat(i, i, i) for i in [-0.2, -0.1, 1.1, 1.2]] ≈ [0.0, 0.0, 3.0, 3.0]  atol=tolerance
    end
end
@testset "3D fast kernels" begin
    for kernel in kernels
        println("Testing 3D fast kernel: ", kernel)
        range_3d = range(0.0, stop=1.0, length=N)
        f_3d_linear(i,j,k) = (i-1)/(N-1)+(j-1)/(N-1)+(k-1)/(N-1)
        vals_3d_linear = Array{Float64}(undef, N, N, N)
        for i in 1:N, j in 1:N, k in 1:N
            vals_3d_linear[i,j,k] = f_3d_linear(i,j,k)
        end

        # extrapolation
        etp_3d_linear = convolution_interpolation((range_3d, range_3d, range_3d), vals_3d_linear; kernel=kernel, fast=true, bc=bc, extrap=:line)
        @test isapprox([etp_3d_linear(i, i, i) for i in [-0.2, -0.1, 1.1, 1.2]], [-0.6, -0.3, 3.3, 3.6], atol=1e-3)  atol=tolerance
        etp_3d_flat = convolution_interpolation((range_3d, range_3d, range_3d), vals_3d_linear; kernel=kernel, fast=true, bc=bc, extrap=:flat)
        @test [etp_3d_flat(i, i, i) for i in [-0.2, -0.1, 1.1, 1.2]] ≈ [0.0, 0.0, 3.0, 3.0]  atol=tolerance
    end
end

### 4D
@testset "4D direct kernels" begin
    for kernel in kernels
        println("Testing 4D direct kernel: ", kernel)
        range_4d = range(0.0, stop=1.0, length=N)
        f_4d_linear(i,j,k,l) = (i-1)/(N-1)+(j-1)/(N-1)+(k-1)/(N-1)+(l-1)/(N-1)
        vals_4d_linear = Array{Float64}(undef, N, N, N, N)
        for i in 1:N, j in 1:N, k in 1:N, l in 1:N
            vals_4d_linear[i,j,k,l] = f_4d_linear(i,j,k,l)
        end

        # test extrapolation bc
        etp_4d_linear = convolution_interpolation((range_4d, range_4d, range_4d, range_4d), vals_4d_linear; kernel=kernel, fast=false, bc=bc, extrap=:line)
        @test [etp_4d_linear( i, i, i, i) for i in [-0.2, -0.1, 1.1, 1.2]] ≈ [-0.8, -0.4, 4.4, 4.8]  atol=tolerance
        etp_4d_flat = convolution_interpolation((range_4d, range_4d, range_4d, range_4d), vals_4d_linear; kernel=kernel, fast=false, bc=bc, extrap=:flat)
        @test [etp_4d_flat( i, i, i, i) for i in [-0.2, -0.1, 1.1, 1.2]] ≈ [0.0, 0.0, 4.0, 4.0]  atol=tolerance
    end
end
@testset "4D fast kernels" begin
    for kernel in kernels
        println("Testing 4D fast kernel: ", kernel)
        range_4d = range(0.0, stop=1.0, length=N)
        f_4d_linear(i,j,k,l) = (i-1)/(N-1)+(j-1)/(N-1)+(k-1)/(N-1)+(l-1)/(N-1)
        vals_4d_linear = Array{Float64}(undef, N, N, N, N)
        for i in 1:N, j in 1:N, k in 1:N, l in 1:N
            vals_4d_linear[i,j,k,l] = f_4d_linear(i,j,k,l)
        end
            
        # test extrapolation bc
        etp_4d_linear = convolution_interpolation((range_4d, range_4d, range_4d, range_4d), vals_4d_linear; kernel=kernel, fast=true, bc=bc, extrap=:line)
        @test [etp_4d_linear( i, i, i, i) for i in [-0.2, -0.1, 1.1, 1.2]] ≈ [-0.8, -0.4, 4.4, 4.8]  atol=tolerance
        etp_4d_flat = convolution_interpolation((range_4d, range_4d, range_4d, range_4d), vals_4d_linear; kernel=kernel, fast=true, bc=bc, extrap=:flat)
        @test [etp_4d_flat( i, i, i, i) for i in [-0.2, -0.1, 1.1, 1.2]] ≈ [0.0, 0.0, 4.0, 4.0]  atol=tolerance
    end
end