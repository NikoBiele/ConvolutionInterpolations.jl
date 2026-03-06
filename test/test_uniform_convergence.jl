#######################################################################
### TEST UNIFORM CONVERGENCE ##########################################
#######################################################################

convergence_kernels = [:a1, :a3, :a4, :a5, :a7, :b5, :b7, :b9, :b11]
convergence_deriv_kernels = [:b5, :b7, :b9, :b11]
expected_order_fast = Dict(
    :a1 => 1, :a3 => 3, :a4 => 4, :a5 => 3, :a7 => 3,
    :b5 => 7, :b7 => 7, :b9 => 7, :b11 => 7 # no accumulated floating point error (precomputed kernel)
)
expected_order_direct = Dict(
    :a1 => 1, :a3 => 3, :a4 => 4, :a5 => 3, :a7 => 3,
    :b5 => 7, :b7 => 6, :b9 => 5, :b11 => 4 # accumulated floating point error for higher orders
)
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
        min_ratio = kernel == :a1 ? 1.5 : 2.0^(expected_order_fast[kernel] - 1)
        @test errs[2] / errs[3] > min_ratio
        @test errs[1] / errs[2] > min_ratio
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
        min_ratio = kernel == :a1 ? 1.5 : 2.0^(expected_order_direct[kernel] - 1)
        @test errs[2] / errs[3] > min_ratio
        @test errs[1] / errs[2] > min_ratio
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
        min_ratio = 2.0^(expected_order_fast[kernel] - 2)
        @test errs[2] / errs[3] > min_ratio
        @test errs[1] / errs[2] > min_ratio
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
        min_ratio = 2.0^(expected_order_direct[kernel] - 2)
        @test errs[2] / errs[3] > min_ratio
        @test errs[1] / errs[2] > min_ratio
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
        min_ratio = 2.0^(expected_order_fast[kernel] - 3)
        @test errs[2] / errs[3] > min_ratio
        @test errs[1] / errs[2] > min_ratio
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
        min_ratio = 2.0^(expected_order_direct[kernel] - 3)
        @test errs[2] / errs[3] > min_ratio
        @test errs[1] / errs[2] > min_ratio
    end
end
