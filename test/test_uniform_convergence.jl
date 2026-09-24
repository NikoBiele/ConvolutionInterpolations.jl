println("\n" * "-"^60)
println("Testing uniform grid convergence in 1D for direct and fast kernels for 0th, 1st and 2nd derivatives...")
println("-"^60)

convergence_kernels = [:a0, :a1, :a3, :b5] # :a4, :a5, :a7, :b7, :b9, :b11 
convergence_deriv_kernels = [:b5] #, :b7, :b9, :b11]
expected_order_fast = Dict(
    :a0 => 1, :a1 => 2, :a3 => 3, #:a4 => 4, :a5 => 3, :a7 => 3,
    :b5 => 7 #, :b7 => 7, :b9 => 7, :b11 => 7 # no accumulated floating point error (exact kernel coefficients)
)
expected_order_direct = Dict(
    :a0 => 1, :a1 => 2, :a3 => 3, # :a4 => 4, :a5 => 3, :a7 => 3,
    :b5 => 7 #, :b7 => 6, :b9 => 5, :b11 => 4 # accumulated floating point error for higher orders
)
bc_deriv = :poly # control kernel boundary conditions for derivatives

# function value convergence
@testset "1D uniform convergence d0 fast" begin
    for kernel in convergence_kernels
        println("    - 1D uniform convergence d0 fast: ", kernel)
        errs = Float64[]
        for n in [12, 24, 48]
            r = range(0.0, 1.0, length=n)
            vals = sin.(2π .* collect(r))
            itp = convolution_interpolation(r, vals; kernel=kernel, fast=true, bc=bc_deriv)
            test_pts = range(collect(r)[1], collect(r)[end], length=100)
            err = maximum(abs.(itp.itp.(test_pts) .- sin.(2π .* test_pts)))
            push!(errs, err)
        end
        min_ratio = kernel == :a0 || kernel == :a1  ? 1.5 : 2.0^(expected_order_fast[kernel] - 1)
        @test errs[2] / errs[3] > min_ratio
        @test errs[1] / errs[2] > min_ratio
    end
end

@testset "1D uniform convergence d0 direct" begin
    for kernel in convergence_kernels
        println("    - 1D uniform convergence d0 direct: ", kernel)
        errs = Float64[]
        for n in [12, 24, 48]
            r = range(0.0, 1.0, length=n)
            vals = sin.(2π .* collect(r))
            itp = convolution_interpolation(r, vals; kernel=kernel, fast=false, bc=bc_deriv)
            test_pts = range(collect(r)[1], collect(r)[end], length=100)
            err = maximum(abs.(itp.itp.(test_pts) .- sin.(2π .* test_pts)))
            push!(errs, err)
        end
        min_ratio = kernel == :a0 || kernel == :a1  ? 1.5 : 2.0^(expected_order_direct[kernel] - 1)
        @test errs[2] / errs[3] > min_ratio
        @test errs[1] / errs[2] > min_ratio
    end
end

# first derivative convergence
@testset "1D uniform convergence d1 fast" begin
    for kernel in convergence_deriv_kernels
        println("    - 1D uniform convergence d1 fast: ", kernel)
        errs = Float64[]
        for n in [12, 24, 48]
            r = range(0.0, 2π, length=n)
            vals = sin.(collect(r))
            itp_d1 = convolution_interpolation(r, vals; kernel=kernel, fast=true, bc=bc_deriv, derivative=1)
            test_pts = range(collect(r)[1], collect(r)[end], length=100)
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
        println("    - 1D uniform convergence d1 direct: ", kernel)
        errs = Float64[]
        for n in [12, 24, 48]
            r = range(0.0, 2π, length=n)
            vals = sin.(collect(r))
            itp_d1 = convolution_interpolation(r, vals; kernel=kernel, fast=false, bc=bc_deriv, derivative=1)
            test_pts = range(collect(r)[1], collect(r)[end], length=100)
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
        println("    - 1D uniform convergence d2 fast: ", kernel)
        errs = Float64[]
        for n in [12, 24, 48]
            r = range(0.0, 2π, length=n)
            vals = sin.(collect(r))
            itp_d2 = convolution_interpolation(r, vals; kernel=kernel, fast=true, bc=bc_deriv, derivative=2)
            test_pts = range(collect(r)[1], collect(r)[end], length=100)
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
        println("    - 1D uniform convergence d2 direct: ", kernel)
        errs = Float64[]
        for n in [12, 24, 48]
            r = range(0.0, 2π, length=n)
            vals = sin.(collect(r))
            itp_d2 = convolution_interpolation(r, vals; kernel=kernel, fast=false, bc=bc_deriv, derivative=2)
            test_pts = range(collect(r)[1], collect(r)[end], length=100)
            err = maximum(abs.(Float64[itp_d2(x) for x in test_pts] .- (-sin.(test_pts))))
            push!(errs, err)
        end
        min_ratio = 2.0^(expected_order_direct[kernel] - 3)
        @test errs[2] / errs[3] > min_ratio
        @test errs[1] / errs[2] > min_ratio
    end
end

# Densely sampled regime
@testset "1D dense-grid rounding floor" begin
    n = 10_000
    r = range(-1.0, 1.0, length=n)
    # Evaluation points across the grid, plus points just inside its far end
    test_pts = vcat(collect(range(-1.0, 1.0, length=1001)), [1.0 - k * step(r) / 7 for k in 1:20])
    f(x) = 3x - 1                       # the data: linear
    F(x) = 1.5 * (x^2 - 1) - (x + 1)    # its antiderivative, anchored at the first knot x = -1
    for kernel in (:a1, :a3, :b5, :b13)
        println("    - 1D dense-grid rounding floor: ", kernel)
        # Values: absolute error, |f| <= 4
        itp = convolution_interpolation(r, f.(r); kernel=kernel, bc=:poly)
        @test maximum(abs(itp(x) - f(x)) for x in test_pts) < 1e-13
        # Antiderivative: absolute error, |F| <= 4.5
        itp_int = convolution_interpolation(r, f.(r); kernel=kernel, bc=:poly, derivative=-1)
        @test maximum(abs(itp_int(x) - F(x)) for x in test_pts) < 5e-13
        # First derivative (exactly 3): relative error, since rounding is amplified by 1/h
        if kernel != :a1
            itp_d1 = convolution_interpolation(r, f.(r); kernel=kernel, bc=:poly, derivative=1)
            @test maximum(abs(itp_d1(x) - 3) for x in test_pts) / 3 < 1e-10
        end
    end

    # Resampling onto a different dense grid, across the full range
    r_out = range(-1.0, 1.0, length=7001)
    resampled = convolution_resample((r,), (r_out,), f.(r); kernel=:b5, bc=:poly)
    @test maximum(abs.(resampled .- f.(r_out))) < 1e-13
end