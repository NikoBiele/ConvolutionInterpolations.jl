println("\n" * "-"^60)
println("Testing 1D antiderivatives of order 2 and higher...")
println("-"^60)

using LinearAlgebra

# Gauss–Legendre nodes and weights on [-1, 1] (Golub–Welsch)
function gauss_legendre(n)
    β = [k / sqrt(4k^2 - 1) for k in 1:n-1]
    E = eigen(SymTridiagonal(zeros(n), β))
    return E.values, 2 .* E.vectors[1, :] .^ 2
end

# ∫ₐˣ g(t) dt, split at every half grid step (g is a polynomial on each piece), 12 nodes each
function integrate_piecewise(g, a, x, h)
    nodes, weights = gauss_legendre(12)
    step = h / 2
    n_pieces = ceil(Int, (x - a) / step - 1e-12)
    total = 0.0
    for p in 1:n_pieces
        lo = a + (p - 1) * step
        hi = min(a + p * step, x)
        mid, half = (lo + hi) / 2, (hi - lo) / 2
        total += half * sum(w * g(mid + half * τ) for (τ, w) in zip(nodes, weights))
    end
    return total
end

# Highest integral order per kernel (m − 1 must not exceed the kernel's vanishing moments)
const max_orders = Dict(:a0 => 2, :a1 => 2, :a3 => 4, :a4 => 4, :a5 => 4, :a7 => 4,
                        :b5 => 6, :b7 => 8, :b9 => 8, :b11 => 8, :b13 => 8)

println("    - exactness on reproduced data, all kernels and orders")
@testset "Higher integrals: exact on reproduced data" begin
    x = range(-1.0, 1.0, length=60)
    pts = range(-1.0, 1.0, length=301)
    for (kernel, max_order) in max_orders
        # :a0 reproduces constants, every other kernel reproduces linear functions
        slope = kernel == :a0 ? 0.0 : 3.0
        f(t) = slope * t - 1
        # exact M-fold integral anchored at a = -1: f(a)(t−a)^M/M! + slope(t−a)^(M+1)/(M+1)!
        F(M, t) = f(-1.0) * (t + 1)^M / factorial(M) + slope * (t + 1)^(M + 1) / factorial(M + 1)
        for M in 2:max_order
            itp = convolution_interpolation(x, f.(x); kernel=kernel, bc=:poly, derivative=-M)
            @test maximum(abs(itp(p) - F(M, p)) for p in pts) < 1e-13
        end
    end
end

println("    - consistency between consecutive orders (quadrature)")
@testset "Higher integrals: F_M equals the integral of F_(M-1)" begin
    x = range(0.0, 2π, length=40)
    h = step(x)
    y = sin.(x) .+ 0.3 .* cos.(3 .* x)
    pts = range(x[1] + 0.3h, x[end], length=25)
    for (kernel, max_order) in ((:a0, 2), (:a1, 2), (:a3, 4), (:b5, 6), (:b13, 8))
        for M in 2:max_order
            lower = convolution_interpolation(x, y; kernel=kernel, bc=:poly, derivative=-(M - 1))
            upper = convolution_interpolation(x, y; kernel=kernel, bc=:poly, derivative=-M)
            # both anchored at x[1]: the M-fold integral is the integral of the (M−1)-fold one
            err = maximum(abs(upper(p) - integrate_piecewise(lower, x[1], p, h)) for p in pts)
            @test err < 1e-12
        end
    end
end

println("    - convergence on smooth data")
@testset "Higher integrals: convergence" begin
    # exact M-fold integral of sin anchored at a: sin(t − Mπ/2) minus its Taylor polynomial at a
    function sin_integral(M, t, a)
        value = sin(t - M * π / 2)
        for k in 0:M-1
            value -= sin(a - M * π / 2 + k * π / 2) * (t - a)^k / factorial(k)
        end
        return value
    end
    for (kernel, order, max_order) in ((:a3, 3, 4), (:b5, 7, 6))
        for M in 2:max_order
            errs = Float64[]
            for n in (20, 40)
                x = range(0.0, 2π, length=n)
                itp = convolution_interpolation(x, sin.(x); kernel=kernel, bc=:poly, derivative=-M)
                pts = range(0.0, 2π, length=200)
                push!(errs, maximum(abs(itp(p) - sin_integral(M, p, 0.0)) for p in pts))
            end
            # halving h must reduce the error at the kernel's order (with a margin of one order)
            @test errs[1] / errs[2] > 2.0^(order - 1)
        end
    end
end

println("    - Float32 and BigFloat")
@testset "Higher integrals: precision" begin
    # Float32 in, Float32 out
    x32 = range(0.0f0, 1.0f0, length=30)
    itp32 = convolution_interpolation(x32, 3 .* x32 .- 1; kernel=:b5, bc=:poly, derivative=-2)
    @test itp32(0.37f0) isa Float32
    # BigFloat: exact on linear data far beyond Float64 precision
    setprecision(BigFloat, 256) do
        xb = range(big(-1.0), big(1.0), length=30)
        itpb = convolution_interpolation(xb, 3 .* xb .- 1; kernel=:b5, bc=:poly, derivative=-3)
        t = big(37) / 100
        exact = (-4) * (t + 1)^3 / 6 + 3 * (t + 1)^4 / 24
        @test itpb(t) isa BigFloat
        @test abs(itpb(t) - exact) < big(10.0)^-60
    end
end

println("    - errors for unsupported combinations")
@testset "Higher integrals: errors" begin
    x = range(0.0, 1.0, length=30)
    y = sin.(x)
    # beyond the kernel's highest order
    @test_throws ErrorException convolution_interpolation(x, y; kernel=:a3, derivative=-5)
    # more than one dimension works: separable data gives the product of the 1D results
    itp2 = convolution_interpolation((x, x), [sin(a) * cos(b) for a in x, b in x];
                                     kernel=:b5, bc=:poly, derivative=(-2, 0))
    ia = convolution_interpolation(x, sin.(x); kernel=:b5, bc=:poly, derivative=-2)
    ib = convolution_interpolation(x, cos.(x); kernel=:b5, bc=:poly)
    @test itp2(0.63, 0.41) ≈ ia(0.63) * ib(0.41) rtol=1e-12
    # lazy mode, via the constructor directly (order 1 and higher)
    @test_throws ErrorException FastConvolutionInterpolation(x, y; kernel=:b5, derivative=-1, lazy=true)
    @test_throws ErrorException FastConvolutionInterpolation(x, y; kernel=:b5, derivative=-2, lazy=true)
    # the direct path
    @test_throws ErrorException convolution_interpolation(x, y; kernel=:b5, derivative=-2, fast=false)
    # nonuniform grids
    xn = sort(vcat(0.0, 0.05:0.1:0.95, 1.0))
    @test_throws ErrorException convolution_interpolation(xn, sin.(xn); kernel=:b5, derivative=-2)
end