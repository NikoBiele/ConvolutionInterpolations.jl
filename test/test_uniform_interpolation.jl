###############################################################################
### TEST UNIFORM GRID INTERPOLATIONS IN 1D, 2D, 3D, 4D ########################
###############################################################################

N = 4 # number of samples in each dimension
tolerance = 1e-6 # tight tolerance
# test all kernels
kernels = [:a3, :a4, :a5, :a7, :b5, :b7, :b9, :b11] #, :b13] :a0, :a1, # only test kernels with cubic subgrid
kernel_bc = :linear # control kernel boundary conditions

println("Testing uniform grid interpolations for direct and fast kernels for 1D, 2D, 3D, 4D...")
### 1D
@testset "1D direct kernels" begin
    for kernel in kernels
        println("Testing 1D direct kernel: ", kernel)
        # random uniformly spaced 1D data
        range_1d = range(0.0, stop=1.0, length=N)
        vals_1d_rand = rand(N)
        itp_1d_rand = convolution_interpolation(range_1d, vals_1d_rand; degree=kernel, fast=false, kernel_bc=kernel_bc)
        @test itp_1d_rand.itp.(range_1d) ≈ vals_1d_rand atol=tolerance
        
        if kernel != :a0
            # match mean of linear uniformly spaced 1D data
            vals_1d_linear = range(0.0, stop=1.0, length=N)
            vals_1d_linear_mean =[(vals_1d_linear[i]+vals_1d_linear[i+1])/2 for i in 1:N-1]
            itp_1d_linear = convolution_interpolation(range_1d, vals_1d_linear; degree=kernel, fast=false, kernel_bc=kernel_bc, extrapolation_bc=Line())
            @test itp_1d_linear.itp.(range(vals_1d_linear_mean[1], stop=vals_1d_linear_mean[end], length=N-1)) ≈ vals_1d_linear_mean  atol=tolerance

            # extrapolation
            etp_1d_linear = convolution_interpolation(range_1d, vals_1d_linear; degree=kernel, fast=false, kernel_bc=kernel_bc, extrapolation_bc=Line())
            @test etp_1d_linear.([-0.2, -0.1, 1.1, 1.2]) ≈ [-0.2, -0.1, 1.1, 1.2]  atol=tolerance
            etp_1d_flat = convolution_interpolation(range_1d, vals_1d_linear; degree=kernel, fast=false, kernel_bc=kernel_bc, extrapolation_bc=Flat())
            @test etp_1d_flat.([-0.2, -0.1, 1.1, 1.2]) ≈ [0.0, 0.0, 1.0, 1.0]  atol=tolerance
        end
    end
end
@testset "1D fast kernels" begin
    for kernel in kernels
        println("Testing 1D fast kernel: ", kernel)
        # random uniformly spaced 1D data
        range_1d = range(0.0, stop=1.0, length=N)
        vals_1d_rand = rand(N)
        itp_1d_rand = convolution_interpolation(range_1d, vals_1d_rand; degree=kernel, fast=true, kernel_bc=kernel_bc)
        @test itp_1d_rand.itp.(range_1d) ≈ vals_1d_rand atol=tolerance
        itp_1d_rand.itp.coefs

        if kernel != :a0
            # match mean of linear uniformly spaced 1D data
            vals_1d_linear = range(0.0, stop=1.0, length=N)
            vals_1d_linear_mean =[(vals_1d_linear[i]+vals_1d_linear[i+1])/2 for i in 1:N-1]
            itp_1d_linear = convolution_interpolation(range_1d, vals_1d_linear; degree=kernel, fast=true, kernel_bc=kernel_bc, extrapolation_bc=Line())
            @test itp_1d_linear.itp.(range(vals_1d_linear_mean[1], stop=vals_1d_linear_mean[end], length=N-1)) ≈ vals_1d_linear_mean  atol=tolerance

            # extrapolation
            etp_1d_linear = convolution_interpolation(range_1d, vals_1d_linear; degree=kernel, fast=true, kernel_bc=kernel_bc, extrapolation_bc=Line())
            @test etp_1d_linear.([-0.2, -0.1, 1.1, 1.2]) ≈ [-0.2, -0.1, 1.1, 1.2]  atol=tolerance
            etp_1d_flat = convolution_interpolation(range_1d, vals_1d_linear; degree=kernel, fast=true, kernel_bc=kernel_bc, extrapolation_bc=Flat())
            @test etp_1d_flat.([-0.2, -0.1, 1.1, 1.2]) ≈ [0.0, 0.0, 1.0, 1.0]  atol=tolerance
        end
    end
end

### 2D
@testset "2D direct kernels" begin
    for kernel in kernels
        println("Testing 2D direct kernel: ", kernel)
        # random uniformly spaced 2D data
        range_2d = range(0.0, stop=1.0, length=N)
        vals_2d_rand = rand(N,N)
        itp_2d_rand = convolution_interpolation((range_2d, range_2d), vals_2d_rand; degree=kernel, fast=false, kernel_bc=kernel_bc)
        @test [itp_2d_rand.itp(range_2d[i], range_2d[j]) for i in 1:N for j in 1:N] ≈ [vals_2d_rand[i,j] for i in 1:N for j in 1:N] atol=tolerance

        if kernel != :a0
            # mean of linear uniformly spaced 2D data
            f_2d_linear(i,j) = (i-1)/(N-1)+(j-1)/(N-1)
            vals_2d_linear = [f_2d_linear(i,j) for i in 1:N, j in 1:N]
            vals_2d_linear_mean = [(vals_2d_linear[i,j]+vals_2d_linear[i+1,j]+vals_2d_linear[i,j+1]+vals_2d_linear[i+1,j+1])/4 for i in 1:N-1, j in 1:N-1]
            itp_2d_linear = convolution_interpolation((range_2d, range_2d), vals_2d_linear; degree=kernel, fast=false, kernel_bc=kernel_bc)
            itp_2d_linear_mean =  [itp_2d_linear.itp((i-0.5)/(N-1),(j-0.5)/(N-1)) for i in 1:N-1, j in 1:N-1] 
            @test itp_2d_linear_mean ≈ vals_2d_linear_mean atol=tolerance  atol=tolerance
            
            # extrapolation
            etp_2d_linear = convolution_interpolation((range_2d, range_2d), vals_2d_linear; degree=kernel, fast=false, kernel_bc=kernel_bc, extrapolation_bc=Line())
            @test [etp_2d_linear(i, i) for i in [-0.2, -0.1, 1.1, 1.2]] ≈ [-0.4, -0.2, 2.2, 2.4]  atol=tolerance
            etp_2d_flat = convolution_interpolation((range_2d, range_2d,), vals_2d_linear; degree=kernel, fast=false, kernel_bc=kernel_bc, extrapolation_bc=Flat())
            @test [etp_2d_flat(i, i) for i in [-0.2, -0.1, 1.1, 1.2]] ≈ [0.0, 0.0, 2.0, 2.0]  atol=tolerance
        end
    end
end
@testset "2D fast kernels" begin
    for kernel in kernels
        println("Testing 2D fast kernel: ", kernel)
        # random uniformly spaced 2D data
        range_2d = range(0.0, stop=1.0, length=N)
        vals_2d_rand = rand(N,N)
        itp_2d_rand = convolution_interpolation((range_2d, range_2d), vals_2d_rand; degree=kernel, fast=true, kernel_bc=kernel_bc)
        @test [itp_2d_rand.itp(range_2d[i], range_2d[j]) for i in 1:N for j in 1:N] ≈ [vals_2d_rand[i,j] for i in 1:N for j in 1:N] atol=tolerance

        if kernel != :a0
            # mean of linear uniformly spaced 2D data
            f_2d_linear(i,j) = (i-1)/(N-1)+(j-1)/(N-1)
            vals_2d_linear = [f_2d_linear(i,j) for i in 1:N, j in 1:N]
            vals_2d_linear_mean = [(vals_2d_linear[i,j]+vals_2d_linear[i+1,j]+vals_2d_linear[i,j+1]+vals_2d_linear[i+1,j+1])/4 for i in 1:N-1, j in 1:N-1]
            itp_2d_linear = convolution_interpolation((range_2d, range_2d), vals_2d_linear; degree=kernel, fast=true, kernel_bc=kernel_bc)
            itp_2d_linear_mean =  [itp_2d_linear.itp((i-0.5)/(N-1),(j-0.5)/(N-1)) for i in 1:N-1, j in 1:N-1] 
            @test itp_2d_linear_mean ≈ vals_2d_linear_mean atol=tolerance  atol=tolerance
            
            # extrapolation
            etp_2d_linear = convolution_interpolation((range_2d, range_2d), vals_2d_linear; degree=kernel, fast=true, kernel_bc=kernel_bc, extrapolation_bc=Line())
            @test [etp_2d_linear(i, i) for i in [-0.2, -0.1, 1.1, 1.2]] ≈ [-0.4, -0.2, 2.2, 2.4]  atol=tolerance
            etp_2d_flat = convolution_interpolation((range_2d, range_2d,), vals_2d_linear; degree=kernel, fast=true, kernel_bc=kernel_bc, extrapolation_bc=Flat())
            @test [etp_2d_flat(i, i) for i in [-0.2, -0.1, 1.1, 1.2]] ≈ [0.0, 0.0, 2.0, 2.0]  atol=tolerance
        end
    end
end

### 3D
@testset "3D direct kernels" begin
    for kernel in kernels
        println("Testing 3D direct kernel: ", kernel)
        # random uniformly spaced 3D data
        range_3d = range(0.0, stop=1.0, length=N)
        vals_3d_rand = rand(N,N,N)
        itp_3d_rand = convolution_interpolation((range_3d, range_3d, range_3d), vals_3d_rand; degree=kernel, fast=false, kernel_bc=kernel_bc)
        interpolated_vals_3d_rand = [itp_3d_rand.itp(range_3d[i], range_3d[j], range_3d[k]) for i in 1:N for j in 1:N for k in 1:N]
        true_vals_3d_rand = [vals_3d_rand[i,j,k] for i in 1:N for j in 1:N for k in 1:N]
        @test interpolated_vals_3d_rand ≈ true_vals_3d_rand atol=tolerance

        if kernel != :a0
            # mean of linear uniformly spaced 3D data
            f_3d_linear(i,j,k) = (i-1)/(N-1)+(j-1)/(N-1)+(k-1)/(N-1)
            vals_3d_linear = Array{Float64}(undef, N, N, N)
            [vals_3d_linear[i,j,k] = f_3d_linear(i,j,k) for i in 1:N for j in 1:N for k in 1:N]
            vals_3d_linear_mean =[(f_3d_linear(i,j,k)+f_3d_linear(i+1,j,k)+f_3d_linear(i,j+1,k)+f_3d_linear(i,j,k+1)+
                                    f_3d_linear(i+1,j+1,k+1)+f_3d_linear(i,j+1,k+1)+f_3d_linear(i+1,j,k+1)+f_3d_linear(i+1,j+1,k))/8 
                                    for i in 1:N-1 for j in 1:N-1 for k in 1:N-1]
            itp_3d_linear = convolution_interpolation((range_3d, range_3d, range_3d), vals_3d_linear; degree=kernel, fast=false, kernel_bc=kernel_bc)
            @test [itp_3d_linear.itp((i-1/2)/(N-1),(j-1/2)/(N-1),(k-1/2)/(N-1)) for i in 1:N-1 for j in 1:N-1 for k in 1:N-1][:] ≈ vals_3d_linear_mean  atol=tolerance

            # extrapolation
            etp_3d_linear = convolution_interpolation((range_3d, range_3d, range_3d), vals_3d_linear; degree=kernel, fast=false, kernel_bc=kernel_bc, extrapolation_bc=Line())
            @test isapprox([etp_3d_linear(i, i, i) for i in [-0.2, -0.1, 1.1, 1.2]], [-0.6, -0.3, 3.3, 3.6], atol=1e-3)  atol=tolerance
            etp_3d_flat = convolution_interpolation((range_3d, range_3d, range_3d), vals_3d_linear; degree=kernel, fast=false, kernel_bc=kernel_bc, extrapolation_bc=Flat())
            @test [etp_3d_flat(i, i, i) for i in [-0.2, -0.1, 1.1, 1.2]] ≈ [0.0, 0.0, 3.0, 3.0]  atol=tolerance
        end
    end
end
@testset "3D fast kernels" begin
    for kernel in kernels
        println("Testing 3D fast kernel: ", kernel)
        # random uniformly spaced 3D data
        range_3d = range(0.0, stop=1.0, length=N)
        vals_3d_rand = rand(N,N,N)
        itp_3d_rand = convolution_interpolation((range_3d, range_3d, range_3d), vals_3d_rand; degree=kernel, fast=true, kernel_bc=kernel_bc)
        interpolated_vals_3d_rand = [itp_3d_rand.itp(range_3d[i], range_3d[j], range_3d[k]) for i in 1:N for j in 1:N for k in 1:N]
        true_vals_3d_rand = [vals_3d_rand[i,j,k] for i in 1:N for j in 1:N for k in 1:N]
        @test interpolated_vals_3d_rand ≈ true_vals_3d_rand atol=tolerance

        if kernel != :a0
            # mean of linear uniformly spaced 3D data
            f_3d_linear(i,j,k) = (i-1)/(N-1)+(j-1)/(N-1)+(k-1)/(N-1)
            vals_3d_linear = Array{Float64}(undef, N, N, N)
            [vals_3d_linear[i,j,k] = f_3d_linear(i,j,k) for i in 1:N for j in 1:N for k in 1:N]
            vals_3d_linear_mean =[(f_3d_linear(i,j,k)+f_3d_linear(i+1,j,k)+f_3d_linear(i,j+1,k)+f_3d_linear(i,j,k+1)+
                                    f_3d_linear(i+1,j+1,k+1)+f_3d_linear(i,j+1,k+1)+f_3d_linear(i+1,j,k+1)+f_3d_linear(i+1,j+1,k))/8 
                                    for i in 1:N-1 for j in 1:N-1 for k in 1:N-1]
            itp_3d_linear = convolution_interpolation((range_3d, range_3d, range_3d), vals_3d_linear; degree=kernel, fast=true, kernel_bc=kernel_bc)
            @test [itp_3d_linear.itp((i-1/2)/(N-1),(j-1/2)/(N-1),(k-1/2)/(N-1)) for i in 1:N-1 for j in 1:N-1 for k in 1:N-1][:] ≈ vals_3d_linear_mean  atol=tolerance

            # extrapolation
            etp_3d_linear = convolution_interpolation((range_3d, range_3d, range_3d), vals_3d_linear; degree=kernel, fast=true, kernel_bc=kernel_bc, extrapolation_bc=Line())
            @test isapprox([etp_3d_linear(i, i, i) for i in [-0.2, -0.1, 1.1, 1.2]], [-0.6, -0.3, 3.3, 3.6], atol=1e-3)  atol=tolerance
            etp_3d_flat = convolution_interpolation((range_3d, range_3d, range_3d), vals_3d_linear; degree=kernel, fast=true, kernel_bc=kernel_bc, extrapolation_bc=Flat())
            @test [etp_3d_flat(i, i, i) for i in [-0.2, -0.1, 1.1, 1.2]] ≈ [0.0, 0.0, 3.0, 3.0]  atol=tolerance
        end
    end
end

### 4D
@testset "4D direct kernels" begin
    for kernel in kernels
        println("Testing 4D direct kernel: ", kernel)
        # random uniformly spaced 4D data
        range_4d = range(0.0, stop=1.0, length=N)
        vals_4d_rand = rand(N,N,N,N)
        itp_4d_rand = convolution_interpolation((range_4d, range_4d, range_4d, range_4d), vals_4d_rand; degree=kernel, fast=false, kernel_bc=kernel_bc)
        interpolated_vals_4d_rand = [itp_4d_rand.itp(range_4d[i], range_4d[j], range_4d[k], range_4d[l]) for i in 1:N for j in 1:N for k in 1:N for l in 1:N]
        true_vals_4d_rand = [vals_4d_rand[i,j,k,l] for i in 1:N for j in 1:N for k in 1:N for l in 1:N]
        @test interpolated_vals_4d_rand ≈ true_vals_4d_rand  atol=tolerance

        if kernel != :a0
            # mean of linear uniformly spaced 4D data
            f_4d_linear(i,j,k,l) = (i-1)/(N-1)+(j-1)/(N-1)+(k-1)/(N-1)+(l-1)/(N-1)
            vals_4d_linear = Array{Float64}(undef, N, N, N, N)
            [vals_4d_linear[i,j,k,l] = f_4d_linear(i,j,k,l) for i in 1:N for j in 1:N for k in 1:N for l in 1:N]
            function hypercube_mean(f_4d_linear::Function, N::Int64)
                [(
                    f_4d_linear(i, j, k, l) +
                    f_4d_linear(i+1, j, k, l) +
                    f_4d_linear(i, j+1, k, l) +
                    f_4d_linear(i, j, k+1, l) +
                    f_4d_linear(i, j, k, l+1) +
                    f_4d_linear(i+1, j+1, k, l) +
                    f_4d_linear(i+1, j, k+1, l) +
                    f_4d_linear(i+1, j, k, l+1) +
                    f_4d_linear(i, j+1, k+1, l) +
                    f_4d_linear(i, j+1, k, l+1) +
                    f_4d_linear(i, j, k+1, l+1) +
                    f_4d_linear(i+1, j+1, k+1, l) +
                    f_4d_linear(i+1, j+1, k, l+1) +
                    f_4d_linear(i+1, j, k+1, l+1) +
                    f_4d_linear(i, j+1, k+1, l+1) +
                    f_4d_linear(i+1, j+1, k+1, l+1)
                ) / 16 for i in 1:N-1, j in 1:N-1, k in 1:N-1, l in 1:N-1]
            end
            vals_4d_linear_mean = hypercube_mean(f_4d_linear, N)
            itp_4d_linear = convolution_interpolation((range_4d, range_4d, range_4d, range_4d), vals_4d_linear; degree=kernel, fast=false, kernel_bc=kernel_bc)
            interpolated_vals_4d_linear_mean = [itp_4d_linear.itp((i-1/2)/(N-1),(j-1/2)/(N-1),(k-1/2)/(N-1),(l-1/2)/(N-1)) for i in 1:N-1 
                                                for j in 1:N-1 for k in 1:N-1 for l in 1:N-1]
            @test interpolated_vals_4d_linear_mean ≈ vals_4d_linear_mean[:]  atol=tolerance

            # test extrapolation bc
            etp_4d_linear = convolution_interpolation((range_4d, range_4d, range_4d, range_4d), vals_4d_linear; degree=kernel, fast=false, kernel_bc=kernel_bc, extrapolation_bc=Line())
            @test [etp_4d_linear( i, i, i, i) for i in [-0.2, -0.1, 1.1, 1.2]] ≈ [-0.8, -0.4, 4.4, 4.8]  atol=tolerance
            etp_4d_flat = convolution_interpolation((range_4d, range_4d, range_4d, range_4d), vals_4d_linear; degree=kernel, fast=false, kernel_bc=kernel_bc, extrapolation_bc=Flat())
            @test [etp_4d_flat( i, i, i, i) for i in [-0.2, -0.1, 1.1, 1.2]] ≈ [0.0, 0.0, 4.0, 4.0]  atol=tolerance
        end
    end
end
@testset "4D fast kernels" begin
    for kernel in kernels
        println("Testing 4D fast kernel: ", kernel)
        # random uniformly spaced 4D data
        range_4d = range(0.0, stop=1.0, length=N)
        vals_4d_rand = rand(N,N,N,N)
        itp_4d_rand = convolution_interpolation((range_4d, range_4d, range_4d, range_4d), vals_4d_rand; degree=kernel, fast=true, kernel_bc=kernel_bc)
        interpolated_vals_4d_rand = [itp_4d_rand.itp(range_4d[i], range_4d[j], range_4d[k], range_4d[l]) for i in 1:N for j in 1:N for k in 1:N for l in 1:N]
        true_vals_4d_rand = [vals_4d_rand[i,j,k,l] for i in 1:N for j in 1:N for k in 1:N for l in 1:N]
        @test interpolated_vals_4d_rand ≈ true_vals_4d_rand  atol=tolerance

        if kernel != :a0
            # mean of linear uniformly spaced 4D data
            f_4d_linear(i,j,k,l) = (i-1)/(N-1)+(j-1)/(N-1)+(k-1)/(N-1)+(l-1)/(N-1)
            vals_4d_linear = Array{Float64}(undef, N, N, N, N)
            [vals_4d_linear[i,j,k,l] = f_4d_linear(i,j,k,l) for i in 1:N for j in 1:N for k in 1:N for l in 1:N]
            function hypercube_mean(f_4d_linear::Function, N::Int64)
                [(
                    f_4d_linear(i, j, k, l) +
                    f_4d_linear(i+1, j, k, l) +
                    f_4d_linear(i, j+1, k, l) +
                    f_4d_linear(i, j, k+1, l) +
                    f_4d_linear(i, j, k, l+1) +
                    f_4d_linear(i+1, j+1, k, l) +
                    f_4d_linear(i+1, j, k+1, l) +
                    f_4d_linear(i+1, j, k, l+1) +
                    f_4d_linear(i, j+1, k+1, l) +
                    f_4d_linear(i, j+1, k, l+1) +
                    f_4d_linear(i, j, k+1, l+1) +
                    f_4d_linear(i+1, j+1, k+1, l) +
                    f_4d_linear(i+1, j+1, k, l+1) +
                    f_4d_linear(i+1, j, k+1, l+1) +
                    f_4d_linear(i, j+1, k+1, l+1) +
                    f_4d_linear(i+1, j+1, k+1, l+1)
                ) / 16 for i in 1:N-1, j in 1:N-1, k in 1:N-1, l in 1:N-1]
            end
            vals_4d_linear_mean = hypercube_mean(f_4d_linear, N)
            itp_4d_linear = convolution_interpolation((range_4d, range_4d, range_4d, range_4d), vals_4d_linear; degree=kernel, fast=true, kernel_bc=kernel_bc)
            interpolated_vals_4d_linear_mean = [itp_4d_linear.itp((i-1/2)/(N-1),(j-1/2)/(N-1),(k-1/2)/(N-1),(l-1/2)/(N-1)) for i in 1:N-1 
                                                for j in 1:N-1 for k in 1:N-1 for l in 1:N-1]
            @test interpolated_vals_4d_linear_mean ≈ vals_4d_linear_mean[:]  atol=tolerance

            # test extrapolation bc
            etp_4d_linear = convolution_interpolation((range_4d, range_4d, range_4d, range_4d), vals_4d_linear; degree=kernel, fast=true, kernel_bc=kernel_bc, extrapolation_bc=Line())
            @test [etp_4d_linear( i, i, i, i) for i in [-0.2, -0.1, 1.1, 1.2]] ≈ [-0.8, -0.4, 4.4, 4.8]  atol=tolerance
            etp_4d_flat = convolution_interpolation((range_4d, range_4d, range_4d, range_4d), vals_4d_linear; degree=kernel, fast=true, kernel_bc=kernel_bc, extrapolation_bc=Flat())
            @test [etp_4d_flat( i, i, i, i) for i in [-0.2, -0.1, 1.1, 1.2]] ≈ [0.0, 0.0, 4.0, 4.0]  atol=tolerance
        end
    end
end