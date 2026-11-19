println("\n" * "-"^60)
println("Testing the refinement of nonuniform interpolants...")
println("-"^60)

# the test function, and its values on a set of knots
refine_runge(x) = 1 / (1 + 25 * x^2)

# knots on [-1, 1]: uniform spacing with every interior knot moved by up to 15 % of it,
# the same on every Julia version
function refine_knots(n::Int)
    h = 2 / (n - 1)
    x = [-1 + (j - 1) * h for j in 1:n]
    for j in 2:n-1
        x[j] += 0.3 * h * (vdc(j, 2) - 0.5)
    end
    return x
end

# knots dense at the centre and coarse at the ends, where the wide kernels oscillate
refine_graded_knots(n::Int) = [sinh(3 * u) / sinh(3) for u in range(-1, 1, length = n)]

@testset "convolution_refine" begin

    @testset "refining equals building on the merged knots" begin
        println("    - refining against a full build on the merged knots...")
        additions = ([0.013],                          # one knot in the interior
                     # several, two of them close together, two near the ends
                     [-0.97, 0.4, 0.41, 0.999],
                     [-1.2, 1.3],                      # outside the old grid on both sides
                     [-0.5])
        for kernel in (:b5, :b7, :b13, :n3, :a1), derivative in (0, 1), x_new in additions
            if kernel in (:n3, :a1) && derivative == 1
                # no derivatives of these on nonuniform grids
                continue
            end
            x = refine_knots(40)
            itp = convolution_interpolation(x, refine_runge.(x); kernel = kernel,
                                            derivative = derivative)
            refined = convolution_refine(itp, x_new, refine_runge.(x_new))
            x_all = sort(vcat(x, x_new))
            full = convolution_interpolation(x_all, refine_runge.(x_all); kernel = kernel,
                                             derivative = derivative)
            @test typeof(refined) == typeof(full)
            @test refined.itp.knots == full.itp.knots
            @test refined.itp.coefs == full.itp.coefs
            @test refined.itp.nonuniform == full.itp.nonuniform
            @test all(refined(t) == full(t)
                      for t in range(minimum(x_all), maximum(x_all), length = 301))
        end
    end

    @testset "only the intervals near a new knot are computed" begin
        println("    - the intervals computed for one new knot...")
        CI = ConvolutionInterpolations
        x = refine_knots(100)
        x_all = sort(vcat(x, 0.0131))
        # 2M intervals, M ghosts per side
        for (kernel, expected) in ((:b5, 10), (:b7, 12), (:b13, 18))
            itp = convolution_interpolation(x, refine_runge.(x); kernel = kernel)
            _, _, computed = CI._refine_b_weights(x_all, kernel, 0, itp.itp.nonuniform[1],
                                                  itp.itp.knots[1])
            @test computed == expected
            refined = convolution_refine(itp, 0.0131, refine_runge(0.0131))
            shared = count(k -> any(refined.itp.nonuniform[1][k] === w
                                    for w in itp.itp.nonuniform[1]), 1:100)
            # the other tables are the old ones, not copies
            @test shared == 100 - expected
            @test itp.itp.domain_size == (100,)     # the interpolant passed in is unchanged
        end
    end

    @testset "element type and extrapolation carry over" begin
        println("    - BigFloat interpolants and the extrapolation setting...")
        x = BigFloat.(refine_knots(40))
        itp = convolution_interpolation(x, refine_runge.(x); kernel = :b7)
        x_new = big"0.0131"
        refined = convolution_refine(itp, [x_new], [refine_runge(x_new)])
        x_all = sort(vcat(x, x_new))
        full = convolution_interpolation(x_all, refine_runge.(x_all); kernel = :b7)
        @test eltype(refined.itp.coefs) == BigFloat
        @test refined.itp.coefs == full.itp.coefs
        @test refined.itp.nonuniform == full.itp.nonuniform

        xf = refine_knots(40)
        line = convolution_interpolation(xf, refine_runge.(xf); kernel = :b5,
                                         extrap = Line())
        refined_line = convolution_refine(line, 0.0131, refine_runge(0.0131))
        @test typeof(refined_line.et) == typeof(line.et)
        @test isfinite(refined_line(1.5))              # still extrapolates outside the grid
    end

    @testset "input that is rejected" begin
        println("    - input that is rejected...")
        x = refine_knots(40)
        itp = convolution_interpolation(x, refine_runge.(x); kernel = :b5)
        # already a knot
        @test_throws ArgumentError convolution_refine(itp, [x[10]], [1.0])
        # given twice
        @test_throws ArgumentError convolution_refine(itp, [0.3, 0.3], [1.0, 1.0])
        # lengths differ
        @test_throws ArgumentError convolution_refine(itp, [0.3], [1.0, 2.0])
        @test_throws ArgumentError convolution_refine(itp, [NaN], [1.0])
        @test_throws ArgumentError convolution_refine(itp, [0.3], [Inf])

        # a uniform grid
        xu = range(-1, 1, length = 30)
        uniform = convolution_interpolation(xu, refine_runge.(xu))
        @test_throws ArgumentError convolution_refine(uniform, [0.31], [1.0])

        # two dimensions
        x2 = refine_knots(12)
        plane = convolution_interpolation((x2, x2),
                                          [a + b for a in x2, b in x2]; kernel = :b5)
        @test_throws ArgumentError convolution_refine(plane, [0.31], [1.0])
    end
end

@testset "convolution_suggest" begin

    # largest error of an interpolant of `f` on [-1, 1]
    suggest_error(itp, f) = maximum(abs(itp(t) - f(t)) for t in range(-1, 1, length = 2001))

    # suggest and refine until nothing is suggested; returns the interpolant and the passes
    # used
    function suggest_loop(itp, f, tol)
        passes = 0
        x_new = convolution_suggest(itp; tol = tol)
        while !isempty(x_new) && passes < 20
            itp = convolution_refine(itp, x_new, f.(x_new))
            passes += 1
            x_new = convolution_suggest(itp; tol = tol)
        end
        return itp, passes
    end

    @testset "refining to a tolerance" begin
        println("    - suggested knots bring the error to the tolerance...")
        x = refine_knots(25)
        steep(t) = tanh(20 * t)
        for kernel in (:b5, :b7, :b13), (f, range_f) in ((refine_runge, 1.0),
                                                         (steep, 2.0)), tol in (1e-4, 1e-7)
            start = convolution_interpolation(x, f.(x); kernel = kernel)
            itp, passes = suggest_loop(start, f, tol)
            @test isempty(convolution_suggest(itp; tol = tol))
            @test passes <= 8
            # the tolerance is a fraction of the data range
            @test suggest_error(itp, f) < 2 * tol * range_f
            @test suggest_error(itp, f) < suggest_error(start, f)
        end
    end

    @testset "knots go where the data need them" begin
        println("    - knots gather at the feature...")
        x = refine_knots(25)
        steep(t) = tanh(20 * t)
        start = convolution_interpolation(x, steep.(x); kernel = :b7)
        itp, _ = suggest_loop(start, steep, 1e-7)
        # the interpolant inside the extrapolation wrapper
        knots, _ = ConvolutionInterpolations._refine_interior(itp.itp)
        inner = count(t -> abs(t) < 0.25, knots)
        # more knots in the quarter around the step than outside it
        @test inner > length(knots) - inner
        n = length(knots)
        even = [-1 + 2 * (j - 1) / (n - 1) + 1e-4 * (vdc(j, 2) - 0.5) * (1 < j < n)
                for j in 1:n]
        spread = convolution_interpolation(even, steep.(even); kernel = :b7)
        # against the same number spread evenly
        @test suggest_error(itp, steep) < 1e-2 * suggest_error(spread, steep)
    end

    @testset "oscillation on a grid with coarse ends" begin
        println("    - the oscillation of a wide kernel on a grid with coarse ends...")
        x = refine_graded_knots(21)
        start = convolution_interpolation(x, refine_runge.(x); kernel = :b13)
        itp, passes = suggest_loop(start, refine_runge, 1e-3)
        error_before = suggest_error(start, refine_runge)
        error_after = suggest_error(itp, refine_runge)
        println("      error $(round(error_before, sigdigits = 2)) -> " *
                "$(round(error_after, sigdigits = 2)) in $passes passes, " *
                "$(itp.itp.domain_size[1]) knots")
        @test error_before > 1                    # the interpolant oscillates near the ends
        @test passes <= 5
        @test error_after < 1e-3
    end

    @testset "what needs no knots" begin
        println("    - polynomials of the kernel's degree, constant data and the " *
                "tolerance...")
        x = refine_knots(30)
        quintic = convolution_interpolation(x, [3 * t^5 - t^2 + 1 for t in x]; kernel = :b5)
        # reproduced, so nothing to estimate
        @test isempty(convolution_suggest(quintic; tol = 1e-9))
        flat = convolution_interpolation(x, fill(2.0, 30); kernel = :b7)
        @test isempty(convolution_suggest(flat))
        itp = convolution_interpolation(x, refine_runge.(x); kernel = :b5)
        # a tolerance nothing exceeds
        @test isempty(convolution_suggest(itp; tol = 1e6))
        @test length(convolution_suggest(itp; tol = 1e-8)) >=
              length(convolution_suggest(itp; tol = 1e-4)) > 0
        suggested = convolution_suggest(itp; tol = 1e-4)
        # midpoints
        @test all(any(x[k] < m < x[k+1] && m == (x[k] + x[k+1]) / 2 for k in 1:29)
                  for m in suggested)
    end

    @testset "other number types and extrapolation" begin
        println("    - Float32, BigFloat and an extrapolating interpolant...")
        x = refine_knots(30)
        single = convolution_interpolation(Float32.(x),
                                           Float32.(refine_runge.(x)); kernel = :b5)
        @test convolution_suggest(single; tol = 1e-3) isa Vector{Float32}
        wide = convolution_interpolation(big.(x), refine_runge.(big.(x)); kernel = :b5)
        @test convolution_suggest(wide; tol = 1e-4) isa Vector{BigFloat}
        @test Float64.(convolution_suggest(wide; tol = 1e-4)) ≈
              convolution_suggest(
                  convolution_interpolation(x, refine_runge.(x); kernel = :b5); tol = 1e-4)
        inside = convolution_interpolation(x, refine_runge.(x); kernel = :b5)
        beyond = convolution_interpolation(x, refine_runge.(x); kernel = :b5,
                                           extrap = :line)
        @test convolution_suggest(beyond; tol = 1e-4) ==
              convolution_suggest(inside; tol = 1e-4)
    end

    @testset "input that is rejected" begin
        println("    - input that is rejected...")
        x = refine_knots(40)
        itp = convolution_interpolation(x, refine_runge.(x); kernel = :b5)
        @test_throws ArgumentError convolution_suggest(itp; tol = 0)
        @test_throws ArgumentError convolution_suggest(itp; samples = 0)
        few = refine_knots(6)                        # fewer knots than the difference needs
        @test_throws ArgumentError convolution_suggest(
            convolution_interpolation(few, refine_runge.(few); kernel = :b5))
        @test_throws ArgumentError convolution_suggest(
            convolution_interpolation(x, refine_runge.(x); kernel = :a1))
        @test_throws ArgumentError convolution_suggest(
            convolution_interpolation(x, refine_runge.(x); kernel = :b5, derivative = 1))
        xu = range(-1, 1, length = 30)               # a uniform grid
        @test_throws ArgumentError convolution_suggest(
            convolution_interpolation(xu, refine_runge.(xu)))
        x2 = refine_knots(12)                        # two dimensions
        plane = convolution_interpolation((x2, x2),
                                          [a + b for a in x2, b in x2]; kernel = :b5)
        @test_throws ArgumentError convolution_suggest(plane)
    end
end
