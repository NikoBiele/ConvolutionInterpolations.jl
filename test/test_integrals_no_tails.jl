# Antiderivatives of any order without stored tails (fast = false): agreement with the stored
# tails, exact integrals of reproduced polynomials, mixed derivative/integral axes, and the
# order caps.

const CIdi = ConvolutionInterpolations

println("\n" * "-"^60)
println("Testing antiderivatives without stored tails (fast = false)...")
println("-"^60)

# all kernels with their highest integral orders
const integral_kernels = (:a0, :a1, :a3, :a4, :a5, :a7, :b5, :b7, :b9, :b11, :b13)

@testset "fast = false vs fast = true, 1D, all kernels and orders" begin
    x = range(0, 2, length = 41)
    f = @. exp(-x^2) * cos(3x)
    ts = range(0, 2, length = 97)
    for kernel in integral_kernels
        max_order = CIdi._max_integral_order[kernel]
        for m in 1:max_order
            notails = convolution_interpolation(x, f; kernel = kernel, derivative = -m, fast = false)
            tails   = convolution_interpolation(x, f; kernel = kernel, derivative = -m, fast = true)
            scale = maximum(abs(tails(t)) for t in ts) + 1
            err = maximum(abs(notails(t) - tails(t)) for t in ts) / scale
            @test err < 1e-12
        end
        println("    - kernel $kernel: orders 1 to $max_order agree with the stored tails")
    end
end

@testset "Exact integrals of a reproduced polynomial, 1D" begin
    # the b kernels reproduce cubics on uniform grids; the m-fold antiderivative of
    # p(x) = x^3 anchored at a (vanishing with its first m − 1 derivatives at a) is
    # I_m(x) = Σ_{k=0}^{3} p^(k)(a)/k! · (x − a)^(k+m) · k! / (k+m)!  = Σ_k p^(k)(a) (x − a)^(k+m) / (k+m)!
    x = range(-1, 1, length = 31)
    a = x[1]                                   # the anchor is the leftmost interior knot
    p(t) = t^3
    dp = (t -> t^3, t -> 3t^2, t -> 6t, t -> 6.0)
    exact(m, t) = sum(dp[k+1](a) * (t - a)^(k + m) / factorial(k + m) for k in 0:3)
    f = p.(x)
    ts = range(-1, 1, length = 61)
    for kernel in (:b5, :b7, :b9, :b11, :b13)
        for m in 1:CIdi._max_integral_order[kernel]
            itp = convolution_interpolation(x, f; kernel = kernel, derivative = -m, fast = false)
            err = maximum(abs(itp(t) - exact(m, t)) for t in ts)
            @test err < 1e-12
        end
        println("    - kernel $kernel: cubic integrated exactly at every order")
    end
end

@testset "fast = false vs fast = true, 2D–4D, pure and mixed orders" begin
    x = range(0, 2, length = 21)
    y = range(-1, 1, length = 17)
    z = range(0, 1, length = 13)
    w = range(-1, 0, length = 9)
    g2 = [exp(-(a^2 + b^2)) for a in x, b in y]
    g3 = [exp(-(a^2 + b^2 + c^2)) for a in x, b in y, c in z]
    g4 = [exp(-(a^2 + b^2 + c^2 + d^2)) for a in x, b in y, c in z, d in w]
    ts2 = Iterators.product(range(0, 2, length = 13), range(-1, 1, length = 11))
    ts3 = Iterators.product(range(0, 2, length = 7), range(-1, 1, length = 7), range(0, 1, length = 5))
    ts4 = Iterators.product(range(0, 2, length = 5), range(-1, 1, length = 5),
                            range(0, 1, length = 4), range(-1, 0, length = 4))
    cases = (
        ((x, y), g2, ts2, (-2, -2)),
        ((x, y), g2, ts2, (-3, 0)),
        ((x, y), g2, ts2, (-2, 1)),
        ((x, y, z), g3, ts3, (-2, -2, -2)),
        ((x, y, z), g3, ts3, (-3, 1, -2)),
        ((x, y, z, w), g4, ts4, (-2, -2, -2, -2)),
        ((x, y, z, w), g4, ts4, (-2, 0, 1, -3)),
    )
    for (knots, g, ts, orders) in cases
        notails = convolution_interpolation(knots, g; kernel = :b5, derivative = orders, fast = false)
        tails   = convolution_interpolation(knots, g; kernel = :b5, derivative = orders, fast = true)
        scale = maximum(abs(tails(t...)) for t in ts) + 1
        err = maximum(abs(notails(t...) - tails(t...)) for t in ts) / scale
        @test err < 1e-10
        println("    - $(length(knots))D, derivative = $orders: agrees with the stored tails")
    end
end

@testset "Order cap and antiderivative kernels" begin
    x = range(0, 1, length = 11)
    f = x .^ 2
    for kernel in integral_kernels
        max_order = CIdi._max_integral_order[kernel]
        @test_throws ErrorException convolution_interpolation(x, f; kernel = kernel,
                                        derivative = -(max_order + 1), fast = false)
    end
    println("    - orders above each kernel's cap are refused")
    # the kernel functor at a negative order is the exact antiderivative at the integers
    for kernel in integral_kernels, m in 1:CIdi._max_integral_order[kernel]
        K = CIdi.ConvolutionKernel(Val(kernel), Val(-m))
        for σ in -10:10
            @test K(Float64(σ)) ≈ Float64(CIdi._kernel_value_exact(kernel, m, σ)) atol = 1e-14
        end
    end
    println("    - antiderivative kernels match the exact values at the integers")
end