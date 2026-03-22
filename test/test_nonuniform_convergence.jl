###########################################################################
### NONUNIFORM CONVERGENCE ################################################
###########################################################################

convergence_nu_kernels = [:b5] #, :b7, :b9, :b11]
expected_order_nonuniform = Dict(
    :b5 => 7 #, :b7 => 7, :b9 => 7, :b11 => 6 # no accumulated floating point error (precomputed kernel)
)

# shared helper
function make_nonuniform_grid_convergence(n; a=0.0, b=1.0, strength=0.3)
    x = collect(range(a, b, length=n))
    h = (b - a) / (n - 1)
    for i in 2:n-1
        x[i] += strength * h * sin(3π * (x[i] - a) / (b - a))
    end
    sort!(x)
    return x
end

println("Testing nonuniform convergence of 0th, 1st, and 2nd derivatives in 1D and 2D...")
@testset "1D nonuniform convergence d0" begin
    for kernel in convergence_nu_kernels
        println("Testing 1D nonuniform convergence d0: ", kernel)
        errs = Float64[]
        for n in [14, 28, 56]
            x_nu = make_nonuniform_grid_convergence(n)
            vals = sin.(2π .* x_nu)
            itp = convolution_interpolation(x_nu, vals; kernel=kernel)
            test_pts = range(x_nu[1], x_nu[end], length=100)
            err = maximum(abs.(itp.itp.(test_pts) .- sin.(2π .* test_pts)))
            push!(errs, err)
        end
        min_ratio = 2.0^(expected_order_nonuniform[kernel] - 1)
        @test errs[2] / errs[3] > min_ratio
        @test errs[1] / errs[2] > min_ratio
    end
end

@testset "1D nonuniform convergence d1" begin
    for kernel in convergence_nu_kernels
        println("Testing 1D nonuniform convergence d1: ", kernel)
        errs = Float64[]
        for n in [14, 28, 56]
            x_nu = make_nonuniform_grid_convergence(n; a=0.0, b=2π)
            vals = sin.(x_nu)
            itp_d1 = convolution_interpolation(x_nu, vals; kernel=kernel, derivative=1)
            test_pts = range(x_nu[1], x_nu[end], length=100)
            err = maximum(abs.(Float64[itp_d1(x) for x in test_pts] .- cos.(test_pts)))
            push!(errs, err)
        end
        min_ratio = 2.0^(expected_order_nonuniform[kernel] - 2)
        @test errs[2] / errs[3] > min_ratio
        @test errs[1] / errs[2] > min_ratio
    end
end

@testset "1D nonuniform convergence d2" begin
    for kernel in convergence_nu_kernels
        println("Testing 1D nonuniform convergence d2: ", kernel)
        errs = Float64[]
        for n in [14, 28, 56]
            x_nu = make_nonuniform_grid_convergence(n; a=0.0, b=2π)
            vals = sin.(x_nu)
            itp_d2 = convolution_interpolation(x_nu, vals; kernel=kernel, derivative=2)
            test_pts = range(x_nu[1], x_nu[end], length=100)
            err = maximum(abs.(Float64[itp_d2(x) for x in test_pts] .- (-sin.(test_pts))))
            push!(errs, err)
        end
        min_ratio = 2.0^(expected_order_nonuniform[kernel] - 3)
        @test errs[2] / errs[3] > min_ratio
        @test errs[1] / errs[2] > min_ratio
    end
end

#########################################################################
### TEST NEARLY UNIFORM VIA NONUNIFORM PATH #############################
#########################################################################

nu_regression_kernels = [:a0, :a1, :a3, :b5] # :a3 kernel triggers nonuniform lower order kernel

println("Testing nearly-uniform via nonuniform path in 1D...")
@testset "1D nearly-uniform via nonuniform matches fast" begin
    for kernel in nu_regression_kernels
        println("Testing nearly-uniform via nonuniform path: ", kernel)
        N_nu = 40
        x_nu = make_nonuniform_grid_convergence(N_nu; strength=1e-8)
        vals_nu = sin.(2π .* x_nu)

        range_u = range(0.0, 1.0, length=N_nu)
        vals_u = sin.(2π .* collect(range_u))
        itp_fast = convolution_interpolation(range_u, vals_u; kernel=kernel, fast=true)
        itp_nu = convolution_interpolation(x_nu, vals_nu; kernel=kernel)

        test_pts = range(x_nu[1], x_nu[end], length=50)
        @test itp_nu.itp.(test_pts) ≈ itp_fast.itp.(test_pts) atol=1e-4
    end
end
