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