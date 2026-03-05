#################################################################
### TEST UNIFORM GRID DERIVATIVES ###############################
#################################################################

### Derivative test settings
deriv_kernels = [:b5, :b7, :b9, :b11] #, :b13] # :a3, :a4, :a5, :a7, # drop lower order kernels (slow convergence)
N_deriv = 10 # number of test points
kernel_bc_deriv = :polynomial # control kernel boundary conditions for derivatives
tolerance_deriv = 0.01 # tolerance for derivative tests

println("Testing uniform grid derivatives for 1D, 2D, 3D, 4D...")
@testset "1D derivative direct" begin
    for kernel in deriv_kernels
        println("Testing 1D derivative direct: ", kernel)
        range_1d = range(0.0, stop=2π, length=N_deriv)
        vals_1d = sin.(range_1d)
        itp_d1 = convolution_interpolation(range_1d, vals_1d; degree=kernel, fast=false,
                    kernel_bc=kernel_bc_deriv, derivative=1)
        
        test_points = [(range_1d[i] + range_1d[i+1]) / 2 for i in 2:N_deriv-2]
        analytical = cos.(test_points)
        interpolated = Float64[itp_d1(x) for x in test_points]
        @test interpolated ≈ analytical atol=tolerance_deriv
    end
end

@testset "1D derivative fast" begin
    for kernel in deriv_kernels
        println("Testing 1D derivative fast: ", kernel)
        range_1d = range(0.0, stop=2π, length=N_deriv)
        vals_1d = sin.(range_1d)
        itp_d1 = convolution_interpolation(range_1d, vals_1d; degree=kernel, fast=true,
                    kernel_bc=kernel_bc_deriv, derivative=1)
        
        test_points = [(range_1d[i] + range_1d[i+1]) / 2 for i in 2:N_deriv-2]
        analytical = cos.(test_points)
        interpolated = Float64[itp_d1(x) for x in test_points]
        @test interpolated ≈ analytical atol=tolerance_deriv
    end
end

@testset "2D derivative direct" begin
    for kernel in deriv_kernels
        println("Testing 2D derivative direct: ", kernel)
        range_2d = range(0.0, stop=2π, length=N_deriv)
        vals_2d = [sin(x) * sin(y) for x in range_2d, y in range_2d]
        itp_d1 = convolution_interpolation((range_2d, range_2d), vals_2d; degree=kernel, fast=false,
                    kernel_bc=kernel_bc_deriv, derivative=1)
        
        test_points = [(range_2d[i] + range_2d[i+1]) / 2 for i in 3:N_deriv-3]
        analytical = [cos(x) * cos(y) for x in test_points, y in test_points]
        interpolated = Float64[itp_d1(x, y) for x in test_points, y in test_points]
        @test interpolated ≈ analytical atol=tolerance_deriv
    end
end

@testset "2D derivative fast" begin
    for kernel in deriv_kernels
        println("Testing 2D derivative fast: ", kernel)
        range_2d = range(0.0, stop=2π, length=N_deriv)
        vals_2d = [sin(x) * sin(y) for x in range_2d, y in range_2d]
        itp_d1 = convolution_interpolation((range_2d, range_2d), vals_2d; degree=kernel, fast=true,
                    kernel_bc=kernel_bc_deriv, derivative=1)
        
        test_points = [(range_2d[i] + range_2d[i+1]) / 2 for i in 3:N_deriv-3]
        analytical = [cos(x) * cos(y) for x in test_points, y in test_points]
        interpolated = Float64[itp_d1(x, y) for x in test_points, y in test_points]
        @test interpolated ≈ analytical atol=tolerance_deriv
    end
end

@testset "3D derivative direct" begin
    for kernel in deriv_kernels
        println("Testing 3D derivative direct: ", kernel)
        range_3d = range(0.0, stop=2π, length=N_deriv)
        vals_3d = [sin(x) * sin(y) * sin(z) for x in range_3d, y in range_3d, z in range_3d]
        itp_d1 = convolution_interpolation((range_3d, range_3d, range_3d), vals_3d; degree=kernel, fast=false,
                    kernel_bc=kernel_bc_deriv, derivative=1)
        
        test_points = [(range_3d[i] + range_3d[i+1]) / 2 for i in 3:N_deriv-3]
        analytical = [cos(x) * cos(y) * cos(z) for x in test_points, y in test_points, z in test_points]
        interpolated = [itp_d1(x, y, z) for x in test_points, y in test_points, z in test_points]
        @test interpolated ≈ analytical atol=tolerance_deriv
    end
end

@testset "3D derivative fast" begin
    for kernel in deriv_kernels
        println("Testing 3D derivative fast: ", kernel)
        range_3d = range(0.0, stop=2π, length=N_deriv)
        vals_3d = [sin(x) * sin(y) * sin(z) for x in range_3d, y in range_3d, z in range_3d]
        itp_d1 = convolution_interpolation((range_3d, range_3d, range_3d), vals_3d; degree=kernel, fast=true,
                    kernel_bc=kernel_bc_deriv, derivative=1)
        
        test_points = [(range_3d[i] + range_3d[i+1]) / 2 for i in 3:N_deriv-3]
        analytical = [cos(x) * cos(y) * cos(z) for x in test_points, y in test_points, z in test_points]
        interpolated = [itp_d1(x, y, z) for x in test_points, y in test_points, z in test_points]
        @test interpolated ≈ analytical atol=tolerance_deriv
    end
end

@testset "4D derivative fast" begin
    for kernel in deriv_kernels
        println("Testing 4D derivative fast: ", kernel)
        range_4d = range(0.0, stop=2π, length=N_deriv)
        vals_4d = [sin(x) * sin(y) * sin(z) * sin(w) for x in range_4d, y in range_4d, z in range_4d, w in range_4d]
        itp_d1 = convolution_interpolation((range_4d, range_4d, range_4d, range_4d), vals_4d; degree=kernel, fast=true,
                    kernel_bc=kernel_bc_deriv, derivative=1)
        
        test_points = [(range_4d[i] + range_4d[i+1]) / 2 for i in 4:N_deriv-4]
        analytical = [cos(x) * cos(y) * cos(z) * cos(w) for x in test_points, y in test_points, z in test_points, w in test_points]
        interpolated = [itp_d1(x, y, z, w) for x in test_points, y in test_points, z in test_points, w in test_points]
        @test interpolated ≈ analytical atol=tolerance_deriv
    end
end
