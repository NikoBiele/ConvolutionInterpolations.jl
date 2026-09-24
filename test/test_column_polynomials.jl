# The exact column polynomials must equal the kernel itself exactly: on every column c
# (offset o = c − 1 − eqs), the column polynomial at τ equals K⁽ᵈ⁾(o + τ). Both sides are
# evaluated in exact rational arithmetic, for every kernel and every derivative order from −1
# (antiderivative) up, at points spread over the column including both ends.
@testset "Column polynomials equal the kernels exactly" begin
    CI = ConvolutionInterpolations
    # Exact positions within a column, including both ends
    τs = [big(0)//1, big(1)//7, big(1)//3, big(1)//2, big(5)//8, big(9)//10, big(1)//1]
    # Every derivative order each kernel supports (−1 = antiderivative)
    derivatives = Dict(:a1 => -1:0,
                       :a3 => -1:1, :a4 => -1:1, :a5 => -1:1, :a7 => -1:1,
                       :b5 => -1:3, :b7 => -1:5, :b9 => -1:6, :b11 => -1:7, :b13 => -1:7)
    for (kernel, ds) in derivatives, d in ds
        eqs = CI.get_equations_for_degree(kernel)
        # The kernel (or its derivative / antiderivative), evaluated exactly
        K = CI.ConvolutionKernel(Val(kernel), Val(d))
        # The exact column polynomials of the same kernel and order
        columns = CI._column_polynomials_exact(kernel, d)
        @test all(CI._poly_eval(columns[c], τ) == K(big(c - 1 - eqs) // 1 + τ)
                  for c in 1:2eqs, τ in τs)
    end
end

# Exact integral kernel values at integer offsets, for every order up to each kernel's cap.
# Order 1 must equal the antiderivative kernel itself; inside the support every order must equal
# the column polynomials; and at the support boundary the far-field formula must continue the
# column polynomials exactly, which confirms the per-kernel order caps.
@testset "Exact integral kernel values at integer offsets" begin
    CI = ConvolutionInterpolations
    for (kernel, max_order) in CI._max_integral_order
        kernel == :a0 && continue                   # closed form, checked below
        eqs = CI.get_equations_for_degree(kernel)
        # order 1 against the antiderivative kernel, inside and outside the support
        K1 = CI.ConvolutionKernel(Val(kernel), Val(-1))
        @test all(CI._kernel_value_exact(kernel, 1, σ) == K1(big(σ) // 1) for σ in -eqs-2:eqs+2)
        for q in 1:max_order
            columns = CI._column_polynomials_exact(kernel, -q)
            # inside the support, and at σ = −eqs: column c = σ + eqs + 1 at τ = 0
            @test all(CI._kernel_value_exact(kernel, q, σ) == CI._poly_eval(columns[σ + eqs + 1], big(0) // 1)
                      for σ in -eqs:eqs-1)
            # at σ = eqs the far field must equal the last column at τ = 1
            @test CI._kernel_value_exact(kernel, q, eqs) == CI._poly_eval(columns[2eqs], big(1) // 1)
        end
    end
    # :a0 in closed form: K₁ = clamp(s, −½, ½); K₂ = s²/2 + 1/8 inside, |s|/2 outside
    @test [CI._kernel_value_exact(:a0, 1, σ) for σ in -2:2] == [-1//2, -1//2, 0, 1//2, 1//2]
    @test [CI._kernel_value_exact(:a0, 2, σ) for σ in -2:2] == [1, 1//2, 1//8, 1//2, 1]
end

# For order 1, the polynomial left tail must reproduce the existing order-1 prefix sums, along
# each dimension of a 2D coefficient array. (Differences can only be in the last bit: the new
# builder rounds ½ − K̃ once from the exact value.)
@testset "Polynomial left tail reproduces order-1 tails" begin
    CI = ConvolutionInterpolations
    rng = Random.MersenneTwister(5)
    for kernel in (:a0, :a1, :a3, :b5, :b13)
        eqs = CI.get_equations_for_degree(kernel)
        coefs = randn(rng, 30, 27)
        for d in 1:2
            # today's order-1 tail: prefix sum of c·(½ − left value) along dimension d
            lv = CI._compute_left_values(Float64, kernel, eqs, size(coefs, d))
            expected = cumsum(coefs .* CI._left_weights(lv, d, Val(2)), dims=d)
            @test CI._left_tail_polynomial(coefs, kernel, 1, eqs, d)[1] ≈ expected rtol=1e-13
        end
    end
end