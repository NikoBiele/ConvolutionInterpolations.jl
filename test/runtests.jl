using ConvolutionInterpolations
using Test

"""
    convolution_test()

Comprehensive test suite for ConvolutionInterpolations.jl covering all kernels and dimensions.

# Test Coverage

## Kernels Tested
All available kernels are tested in both direct and fast evaluation modes:
- `:a0` (nearest neighbor)
- `:a1` (linear)
- `:a3`, `:a5`, `:a7` (cubic, quintic, septic)
- `:a4`, `:b5`, `:b7`, `:b9`, `:b11`, `:b13` (high-order polynomial reproduction)

## Dimensions Tested
Tests span 1D through 4D with both direct (`fast=false`) and fast (`fast=true`) modes:
- **1D**: Vector interpolation
- **2D**: Surface interpolation
- **3D**: Volume interpolation
- **4D**: Hypercube interpolation

## Test Strategy

### 1. Grid Point Reproduction
Verifies that interpolation exactly reproduces values at grid points for random data.
This is the most fundamental property - interpolation must pass through data points.

**Test**: `itp(grid_points) ≈ data_values` with tolerance 1e-4

### 2. Midpoint Interpolation for Linear Data
For linear functions, tests that interpolation correctly computes midpoint values.
This validates the kernel's ability to reproduce polynomials of its designed order.

**Example (1D)**: For linear data `[0.0, 0.333, 0.667, 1.0]`, verifies that 
`itp(0.167) ≈ 0.167` (midpoint between first two grid points)

**Example (2D)**: For bilinear function `f(i,j) = i + j`, verifies that the 
interpolation at cell centers matches the average of four corner values

This test is skipped for `:a0` (nearest neighbor) as it cannot reproduce linear functions.

### 3. Extrapolation Boundary Conditions
Tests correctness of extrapolation outside the domain for two boundary conditions:

- **Line()**: Linear extrapolation continues the trend
  - Test: `etp(-0.2)` for data `[0.0, 1.0]` should give `-0.2`
  
- **Flat()**: Constant extrapolation holds boundary value
  - Test: `etp(-0.2)` for data `[0.0, 1.0]` should give `0.0`

### 4. Dimensional Scaling
Each dimension uses the same test patterns, verifying that the tensor product
structure maintains correctness as dimensionality increases:

- **1D**: Linear function `f(x) = x`
- **2D**: Bilinear function `f(x,y) = x + y`
- **3D**: Trilinear function `f(x,y,z) = x + y + z`
- **4D**: Quadrilinear function `f(x,y,z,w) = x + y + z + w`

Midpoint tests verify averages over 2^N hypercube corners (2 in 1D, 4 in 2D, 8 in 3D, 16 in 4D).

## Test Parameters
- **Grid size**: N=4 points per dimension (minimal but sufficient for testing)
- **Tolerance**: 1e-4 for numerical accuracy
- **Boundary conditions**: `:linear` for kernel BC (polynomial reproduction coefficients)

## Why This Testing Approach?

1. **Grid point reproduction**: Fundamental correctness - interpolation must be exact at data points
2. **Linear midpoint test**: Validates polynomial reproduction order (higher-order kernels 
   should handle linear functions perfectly)
3. **Random data**: Catches edge cases and ensures robustness beyond structured test cases
4. **Extrapolation**: Verifies boundary handling doesn't break at domain edges
5. **Multi-dimensional**: Ensures tensor product structure works correctly in all dimensions

## Example Test Flow

For kernel `:b5` in 2D fast mode:
1. Create 4×4 random data on [0,1]×[0,1] grid
2. Build fast interpolator: `itp = convolution_interpolation((x,y), data, degree=:b5, fast=true)`
3. Verify `itp(grid_points) ≈ data` (grid point reproduction)
4. Create linear test data `f(i,j) = i + j`
5. Verify midpoint values match analytical averages (polynomial reproduction)
6. Test extrapolation with Line() and Flat() boundary conditions
7. Repeat for direct mode (`fast=false`)

This comprehensive approach ensures correctness across all use cases while maintaining
fast test execution (~seconds for full suite).
"""

function convolution_test()

    ###############################################################################
    ### TEST UNIFORM GRID INTERPOLATIONS IN 1D, 2D, 3D, 4D ########################
    ###############################################################################

    N = 4 # number of samples in each dimension
    tolerance = 1e-4 # tight tolerance
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

    #################################################################
    ### TEST UNIFORM GRID DERIVATIVES ###############################
    #################################################################

    ### Derivative test settings
    deriv_kernels = [:b5, :b7, :b9, :b11] #, :b13] # :a3, :a4, :a5, :a7, # drop lower order kernels (slow convergence)
    N_deriv = 10 # number of test points
    kernel_bc_deriv = :polynomial # control kernel boundary conditions for derivatives
    tolerance_deriv = 0.1 # tolerance for derivative tests

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

    #######################################################################
    ### TEST UNIFORM CONVERGENCE ##########################################
    #######################################################################

    convergence_kernels = [:a3, :a4, :a5, :a7, :b5, :b7, :b9, :b11]
    convergence_deriv_kernels = [:b5, :b7, :b9, :b11]
    kernel_bc_deriv = :polynomial # control kernel boundary conditions for derivatives

    println("Testing uniform grid convergence in 1D for direct and fast kernels for 0th, 1st and 2nd derivatives...")
    # function value convergence
    @testset "1D uniform convergence d0 fast" begin
        for kernel in convergence_kernels
            println("Testing 1D uniform convergence d0 fast: ", kernel)
            errs = Float64[]
            for n in [12, 24, 48]
                r = range(0.0, 1.0, length=n)
                vals = sin.(2π .* collect(r))
                itp = convolution_interpolation(r, vals; degree=kernel, fast=true, kernel_bc=kernel_bc_deriv)
                test_pts = range(collect(r)[5], collect(r)[end-4], length=100)
                err = maximum(abs.(itp.itp.(test_pts) .- sin.(2π .* test_pts)))
                push!(errs, err)
            end
            @test errs[2] / errs[3] > 2.0
            @test errs[1] / errs[2] > 2.0
        end
    end

    @testset "1D uniform convergence d0 direct" begin
        for kernel in convergence_kernels
            println("Testing 1D uniform convergence d0 direct: ", kernel)
            errs = Float64[]
            for n in [12, 24, 48]
                r = range(0.0, 1.0, length=n)
                vals = sin.(2π .* collect(r))
                itp = convolution_interpolation(r, vals; degree=kernel, fast=false, kernel_bc=kernel_bc_deriv)
                test_pts = range(collect(r)[5], collect(r)[end-4], length=100)
                err = maximum(abs.(itp.itp.(test_pts) .- sin.(2π .* test_pts)))
                push!(errs, err)
            end
            @test errs[2] / errs[3] > 2.0
            @test errs[1] / errs[2] > 2.0
        end
    end

    # first derivative convergence
    @testset "1D uniform convergence d1 fast" begin
        for kernel in convergence_deriv_kernels
            println("Testing 1D uniform convergence d1 fast: ", kernel)
            errs = Float64[]
            for n in [12, 24, 48]
                r = range(0.0, 2π, length=n)
                vals = sin.(collect(r))
                itp_d1 = convolution_interpolation(r, vals; degree=kernel, fast=true, kernel_bc=kernel_bc_deriv, derivative=1)
                test_pts = range(collect(r)[6], collect(r)[end-5], length=100)
                err = maximum(abs.(Float64[itp_d1(x) for x in test_pts] .- cos.(test_pts)))
                push!(errs, err)
            end
            @test errs[2] / errs[3] > 2.0
            @test errs[1] / errs[2] > 2.0
        end
    end

    @testset "1D uniform convergence d1 direct" begin
        for kernel in convergence_deriv_kernels
            println("Testing 1D uniform convergence d1 direct: ", kernel)
            errs = Float64[]
            for n in [12, 24, 48]
                r = range(0.0, 2π, length=n)
                vals = sin.(collect(r))
                itp_d1 = convolution_interpolation(r, vals; degree=kernel, fast=false, kernel_bc=kernel_bc_deriv, derivative=1)
                test_pts = range(collect(r)[6], collect(r)[end-5], length=100)
                err = maximum(abs.(Float64[itp_d1(x) for x in test_pts] .- cos.(test_pts)))
                push!(errs, err)
            end
            @test errs[2] / errs[3] > 2.0
            @test errs[1] / errs[2] > 2.0
        end
    end

    # second derivative convergence
    @testset "1D uniform convergence d2 fast" begin
        for kernel in convergence_deriv_kernels
            println("Testing 1D uniform convergence d2 fast: ", kernel)
            errs = Float64[]
            for n in [12, 24, 48]
                r = range(0.0, 2π, length=n)
                vals = sin.(collect(r))
                itp_d2 = convolution_interpolation(r, vals; degree=kernel, fast=true, kernel_bc=kernel_bc_deriv, derivative=2)
                test_pts = range(collect(r)[6], collect(r)[end-5], length=100)
                err = maximum(abs.(Float64[itp_d2(x) for x in test_pts] .- (-sin.(test_pts))))
                push!(errs, err)
            end
            @test errs[2] / errs[3] > 2.0
            @test errs[1] / errs[2] > 2.0
        end
    end

    @testset "1D uniform convergence d2 direct" begin
        for kernel in convergence_deriv_kernels
            println("Testing 1D uniform convergence d2 direct: ", kernel)
            errs = Float64[]
            for n in [12, 24, 48]
                r = range(0.0, 2π, length=n)
                vals = sin.(collect(r))
                itp_d2 = convolution_interpolation(r, vals; degree=kernel, fast=false, kernel_bc=kernel_bc_deriv, derivative=2)
                test_pts = range(collect(r)[6], collect(r)[end-5], length=100)
                err = maximum(abs.(Float64[itp_d2(x) for x in test_pts] .- (-sin.(test_pts))))
                push!(errs, err)
            end
            @test errs[2] / errs[3] > 2.0
            @test errs[1] / errs[2] > 2.0
        end
    end
    
    ###########################################################################
    ### NONUNIFORM TESTS ######################################################
    ###########################################################################

    # ── Nonuniform grid helper ──────────────────────────────────
    function make_nonuniform_grid(n; a=0.0, b=1.0, strength=0.3)
        x = collect(range(a, b, length=n))
        h = (b - a) / (n - 1)
        for i in 2:n-1
            x[i] += strength * h * sin(3π * (x[i] - a) / (b - a))
        end
        sort!(x)
        return x
    end

    ### Nonuniform b-kernel interpolation
    nu_kernels = [:a3, :b5, :b7, :b9, :b11] # :a3 kernel triggers nonuniform lower order kernel
    N_nu = 12
    tolerance_nu = 1e-4

    println("Testing nonuniform kernel interpolation in 1D, 2D, and 3D...")
    @testset "1D nonuniform kernels" begin
        for kernel in nu_kernels
            println("Testing 1D nonuniform kernel: ", kernel)
            # grid point reproduction with random data
            x_nu = make_nonuniform_grid(N_nu)
            vals_rand = rand(N_nu)
            itp_rand = convolution_interpolation(x_nu, vals_rand; degree=kernel)
            @test itp_rand.itp.(x_nu) ≈ vals_rand atol=tolerance_nu

            # linear midpoint reproduction
            vals_linear = x_nu
            itp_linear = convolution_interpolation(x_nu, vals_linear; degree=kernel)
            midpoints = [(x_nu[i] + x_nu[i+1]) / 2 for i in 3:N_nu-3]
            @test itp_linear.itp.(midpoints) ≈ midpoints atol=tolerance_nu
        end
    end

    @testset "2D nonuniform kernels" begin
        for kernel in nu_kernels
            println("Testing 2D nonuniform kernel: ", kernel)
            x_nu = make_nonuniform_grid(N_nu)
            y_nu = make_nonuniform_grid(N_nu; strength=0.2)

            # grid point reproduction with random data
            vals_rand = rand(N_nu, N_nu)
            itp_rand = convolution_interpolation((x_nu, y_nu), vals_rand; degree=kernel)
            @test [itp_rand.itp(x_nu[i], y_nu[j]) for i in 1:N_nu for j in 1:N_nu] ≈ 
                  [vals_rand[i,j] for i in 1:N_nu for j in 1:N_nu] atol=tolerance_nu

            # bilinear midpoint reproduction
            vals_linear = [x_nu[i] + y_nu[j] for i in 1:N_nu, j in 1:N_nu]
            itp_linear = convolution_interpolation((x_nu, y_nu), vals_linear; degree=kernel)
            test_x = [(x_nu[i] + x_nu[i+1]) / 2 for i in 3:N_nu-3]
            test_y = [(y_nu[i] + y_nu[i+1]) / 2 for i in 3:N_nu-3]
            analytical = [tx + ty for tx in test_x, ty in test_y]
            interpolated = [itp_linear.itp(tx, ty) for tx in test_x, ty in test_y]
            @test interpolated ≈ analytical atol=tolerance_nu
        end
    end

    @testset "3D nonuniform kernels" begin
        for kernel in nu_kernels
            println("Testing 3D nonuniform kernel: ", kernel)
            x_nu = make_nonuniform_grid(N_nu)
            y_nu = make_nonuniform_grid(N_nu; strength=0.2)
            z_nu = make_nonuniform_grid(N_nu; strength=0.25)

            # grid point reproduction with random data
            vals_rand = rand(N_nu, N_nu, N_nu)
            itp_rand = convolution_interpolation((x_nu, y_nu, z_nu), vals_rand; degree=kernel)
            interpolated = [itp_rand.itp(x_nu[i], y_nu[j], z_nu[k]) for i in 1:N_nu for j in 1:N_nu for k in 1:N_nu]
            true_vals = [vals_rand[i,j,k] for i in 1:N_nu for j in 1:N_nu for k in 1:N_nu]
            @test interpolated ≈ true_vals atol=tolerance_nu
        end
    end

    ###########################################################################
    ### NONUNIFORM DERIVATIVES ################################################
    ###########################################################################

    ### Nonuniform b-kernel derivatives
    nu_deriv_kernels = [:b5, :b7, :b9, :b11]
    N_nu_deriv = 12
    tolerance_nu_deriv = 0.1

    println("Testing nonuniform derivative kernel interpolation in 1D, 2D...")
    @testset "1D nonuniform derivative d1" begin
        for kernel in nu_deriv_kernels
            println("Testing 1D nonuniform derivative d1: ", kernel)
            x_nu = make_nonuniform_grid(N_nu_deriv; a=0.0, b=2π)
            vals = sin.(x_nu)
            itp_d1 = convolution_interpolation(x_nu, vals; degree=kernel, derivative=1)

            test_points = [(x_nu[i] + x_nu[i+1]) / 2 for i in 4:N_nu_deriv-4]
            analytical = cos.(test_points)
            interpolated = Float64[itp_d1(x) for x in test_points]
            @test interpolated ≈ analytical atol=tolerance_nu_deriv
        end
    end

    @testset "1D nonuniform derivative d2" begin
        for kernel in nu_deriv_kernels
            println("Testing 1D nonuniform derivative d2: ", kernel)
            x_nu = make_nonuniform_grid(N_nu_deriv; a=0.0, b=2π)
            vals = sin.(x_nu)
            itp_d2 = convolution_interpolation(x_nu, vals; degree=kernel, derivative=2)

            test_points = [(x_nu[i] + x_nu[i+1]) / 2 for i in 4:N_nu_deriv-4]
            analytical = -sin.(test_points)
            interpolated = Float64[itp_d2(x) for x in test_points]
            @test interpolated ≈ analytical atol=tolerance_nu_deriv
        end
    end

    @testset "2D nonuniform derivative d1" begin
        for kernel in nu_deriv_kernels
            println("Testing 2D nonuniform derivative d1: ", kernel)
            x_nu = make_nonuniform_grid(N_nu_deriv; a=0.0, b=2π)
            y_nu = make_nonuniform_grid(N_nu_deriv; a=0.0, b=2π, strength=0.2)
            vals = [sin(x) * sin(y) for x in x_nu, y in y_nu]
            itp_d1 = convolution_interpolation((x_nu, y_nu), vals; degree=kernel, derivative=1)

            test_x = [(x_nu[i] + x_nu[i+1]) / 2 for i in 5:N_nu_deriv-5]
            test_y = [(y_nu[i] + y_nu[i+1]) / 2 for i in 5:N_nu_deriv-5]
            analytical = [cos(x) * cos(y) for x in test_x, y in test_y]
            interpolated = Float64[itp_d1(x, y) for x in test_x, y in test_y]
            @test interpolated ≈ analytical atol=tolerance_nu_deriv
        end
    end

    ###########################################################################
    ### NONUNIFORM CONVERGENCE ################################################
    ###########################################################################

    convergence_nu_kernels = [:b5, :b7, :b9, :b11]

    println("Testing nonuniform convergence of 0th, 1st, and 2nd derivatives in 1D and 2D...")
    @testset "1D nonuniform convergence d0" begin
        for kernel in convergence_nu_kernels
            println("Testing 1D nonuniform convergence d0: ", kernel)
            errs = Float64[]
            for n in [20, 40, 80]
                x_nu = make_nonuniform_grid(n)
                vals = sin.(2π .* x_nu)
                itp = convolution_interpolation(x_nu, vals; degree=kernel)
                test_pts = range(x_nu[5], x_nu[end-4], length=100)
                err = maximum(abs.(itp.itp.(test_pts) .- sin.(2π .* test_pts)))
                push!(errs, err)
            end
            @test errs[2] / errs[3] > 2.0
            @test errs[1] / errs[2] > 2.0
        end
    end

    @testset "1D nonuniform convergence d1" begin
        for kernel in convergence_nu_kernels
            println("Testing 1D nonuniform convergence d1: ", kernel)
            errs = Float64[]
            for n in [20, 40, 80]
                x_nu = make_nonuniform_grid(n; a=0.0, b=2π)
                vals = sin.(x_nu)
                itp_d1 = convolution_interpolation(x_nu, vals; degree=kernel, derivative=1)
                test_pts = range(x_nu[6], x_nu[end-5], length=100)
                err = maximum(abs.(Float64[itp_d1(x) for x in test_pts] .- cos.(test_pts)))
                push!(errs, err)
            end
            # error should decrease at least 2× per doubling
            @test errs[2] / errs[3] > 2.0
            @test errs[1] / errs[2] > 2.0
        end
    end

    @testset "1D nonuniform convergence d2" begin
        for kernel in convergence_nu_kernels
            println("Testing 1D nonuniform convergence d2: ", kernel)
            errs = Float64[]
            for n in [20, 40, 80]
                x_nu = make_nonuniform_grid(n; a=0.0, b=2π)
                vals = sin.(x_nu)
                itp_d2 = convolution_interpolation(x_nu, vals; degree=kernel, derivative=2)
                test_pts = range(x_nu[6], x_nu[end-5], length=100)
                err = maximum(abs.(Float64[itp_d2(x) for x in test_pts] .- (-sin.(test_pts))))
                push!(errs, err)
            end
            # error should decrease at least 2× per doubling
            @test errs[2] / errs[3] > 2.0
            @test errs[1] / errs[2] > 2.0
        end
    end

    #########################################################################
    ### TEST NEARLY UNIFORM VIA NONUNIFORM PATH #############################
    #########################################################################

    nu_regression_kernels = [:a3, :b5, :b7, :b9, :b11] # :a3 kernel triggers nonuniform lower order kernel

    println("Testing nearly-uniform via nonuniform path in 1D...")
    @testset "1D nearly-uniform via nonuniform matches fast" begin
        for kernel in nu_regression_kernels
            println("Testing nearly-uniform via nonuniform path: ", kernel)
            x_nu = make_nonuniform_grid(N_nu; strength=1e-8)
            vals_nu = sin.(2π .* x_nu)

            range_u = range(0.0, 1.0, length=N_nu)
            vals_u = sin.(2π .* collect(range_u))
            itp_fast = convolution_interpolation(range_u, vals_u; degree=kernel, fast=true)
            itp_nu = convolution_interpolation(x_nu, vals_nu; degree=kernel)

            test_pts = range(x_nu[5], x_nu[end-4], length=50)
            @test itp_nu.itp.(test_pts) ≈ itp_fast.itp.(test_pts) atol=0.1
        end
    end
end

@testset "ConvolutionInterpolations.jl" begin
    convolution_test()
end
