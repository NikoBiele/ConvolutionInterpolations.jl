"""
    precompute_kernel_and_range(degree::Symbol; precompute::Int=101,
        F::Type{<:AbstractFloat}=Float64, derivative::Int=0)

Generate precomputed kernel values on a uniform grid for fast interpolation.

This is the computation backend called by `_get_cached_kernel` when pre-shipped tables
are not available (e.g. for `:linear` subgrid with high resolution, BigFloat precision,
or top-derivative orders). For the default path (`precompute=101`, cubic/quintic subgrid,
Float64), pre-shipped constant tables are used instead and this function is not called.

# Arguments
- `degree::Symbol`: Kernel type (`:a0`, `:a1`, `:a3`, `:b5`, etc.)

# Keyword Arguments
- `precompute::Int=101`: Number of uniformly spaced points at which to evaluate the kernel
- `F::Type{<:AbstractFloat}=Float64`: Floating-point type for computations and storage
- `derivative::Int=0`: Output derivative order

# Returns
Tuple of `(pre_range, kernel_pre, kernel_d1_pre, kernel_d2_pre)`:
- `pre_range`: Vector of length `precompute` with uniformly spaced values from 0 to 1
- `kernel_pre`: Matrix of size `(precompute, 2*eqs)` with kernel values for output derivative
- `kernel_d1_pre`: Matrix with kernel's next derivative (for cubic Hermite subgrid), or `nothing`
- `kernel_d2_pre`: Matrix with kernel's second-next derivative (for quintic Hermite), or `nothing`

# Details
Evaluates the convolution kernel at all relevant offsets using exact `Rational{BigInt}`
arithmetic, then converts to the requested float type. The kernel extends from `-(eqs-1)`
to `eqs` (total support: `2*eqs` columns).

Special cases: `:a0` and `:a1` always use `precompute=2` (just endpoints).

Results are cached to disk by `_get_cached_kernel` via Scratch.jl, so this function
typically runs only once per (degree, precompute, precision, derivative) combination.

See also: `get_precomputed_kernel_and_range`, `get_shipped_kernel_tables`.
"""

function precompute_kernel_and_range(degree::Symbol;
          precompute::P, F::Type{T}, derivative::Int=0) where {P,T}
    eqs = get_equations_for_degree(degree)
    pre_range_exact = [big(0//1) + big(i//1-1//1)/big(precompute//1-1//1) for i in 1:precompute]
    pre_range = F.(pre_range_exact)
    if degree in [:a0, :a1]
      kernel_pre = precompute_kernel(F, degree, derivative, eqs, precompute, pre_range_exact)
      return pre_range, kernel_pre, [zero(F), zero(F)], [zero(F), zero(F)]
    elseif degree in [:a3, :a4, :a5, :a7]
      if derivative == 0
        kernel_pre = precompute_kernel(F, degree, derivative, eqs, precompute, pre_range_exact)
        kernel_d1_pre = precompute_kernel(F, degree, derivative+1, eqs, precompute, pre_range_exact)
        return pre_range, kernel_pre, kernel_d1_pre, [zero(F), zero(F)]
      elseif derivative == 1
        kernel_pre = precompute_kernel(F, degree, derivative, eqs, precompute, pre_range_exact)
        return pre_range, kernel_pre, [zero(F), zero(F)], [zero(F), zero(F)]
      end
    elseif degree == :b5
      # b5 kernel
      if derivative in [0, 1]
        kernel_pre = precompute_kernel(F, degree, derivative, eqs, precompute, pre_range_exact)
        kernel_d1_pre = precompute_kernel(F, degree, derivative+1, eqs, precompute, pre_range_exact)
        kernel_d2_pre = precompute_kernel(F, degree, derivative+2, eqs, precompute, pre_range_exact)
        return pre_range, kernel_pre, kernel_d1_pre, kernel_d2_pre
      elseif derivative == 2
        kernel_pre = precompute_kernel(F, degree, derivative, eqs, precompute, pre_range_exact)
        kernel_d1_pre = precompute_kernel(F, degree, derivative+1, eqs, precompute, pre_range_exact)
        return pre_range, kernel_pre, kernel_d1_pre, [zero(F), zero(F)]
      else # derivative == 3 && degree == :b5
        kernel_pre = precompute_kernel(F, degree, derivative, eqs, precompute, pre_range_exact)
        return pre_range, kernel_pre, [zero(F), zero(F)], [zero(F), zero(F)]
      end
    elseif degree == :b7
      # b7 kernel
      if derivative in [0, 1, 2]
        kernel_pre = precompute_kernel(F, degree, derivative, eqs, precompute, pre_range_exact)
        kernel_d1_pre = precompute_kernel(F, degree, derivative+1, eqs, precompute, pre_range_exact)
        kernel_d2_pre = precompute_kernel(F, degree, derivative+2, eqs, precompute, pre_range_exact)
        return pre_range, kernel_pre, kernel_d1_pre, kernel_d2_pre
      elseif derivative == 3
        kernel_pre = precompute_kernel(F, degree, derivative, eqs, precompute, pre_range_exact)
        kernel_d1_pre = precompute_kernel(F, degree, derivative+1, eqs, precompute, pre_range_exact)
        return pre_range, kernel_pre, kernel_d1_pre, [zero(F), zero(F)]
      else # derivative == 4
        kernel_pre = precompute_kernel(F, degree, derivative, eqs, precompute, pre_range_exact)
        return pre_range, kernel_pre, [zero(F), zero(F)], [zero(F), zero(F)]
      end
    elseif degree == :b9
      # b9 kernel
      if derivative in [0, 1, 2, 3]
        kernel_pre = precompute_kernel(F, degree, derivative, eqs, precompute, pre_range_exact)
        kernel_d1_pre = precompute_kernel(F, degree, derivative+1, eqs, precompute, pre_range_exact)
        kernel_d2_pre = precompute_kernel(F, degree, derivative+2, eqs, precompute, pre_range_exact)
        return pre_range, kernel_pre, kernel_d1_pre, kernel_d2_pre
      elseif derivative == 4
        kernel_pre = precompute_kernel(F, degree, derivative, eqs, precompute, pre_range_exact)
        kernel_d1_pre = precompute_kernel(F, degree, derivative+1, eqs, precompute, pre_range_exact)
        return pre_range, kernel_pre, kernel_d1_pre, [zero(F), zero(F)]
      else # derivative == 5 && degree == :b9
        kernel_pre = precompute_kernel(F, degree, derivative, eqs, precompute, pre_range_exact)
        return pre_range, kernel_pre, [zero(F), zero(F)], [zero(F), zero(F)]
      end
    elseif degree == :b11
      # b11 kernel
      if derivative in [0, 1, 2, 3, 4]
        kernel_pre = precompute_kernel(F, degree, derivative, eqs, precompute, pre_range_exact)
        kernel_d1_pre = precompute_kernel(F, degree, derivative+1, eqs, precompute, pre_range_exact)
        kernel_d2_pre = precompute_kernel(F, degree, derivative+2, eqs, precompute, pre_range_exact)
        return pre_range, kernel_pre, kernel_d1_pre, kernel_d2_pre
      elseif derivative == 5
        kernel_pre = precompute_kernel(F, degree, derivative, eqs, precompute, pre_range_exact)        
        kernel_d1_pre = precompute_kernel(F, degree, derivative+1, eqs, precompute, pre_range_exact)
        return pre_range, kernel_pre, kernel_d1_pre, [zero(F), zero(F)]
      else # derivative == 6 && degree == :b11
        kernel_pre = precompute_kernel(F, degree, derivative, eqs, precompute, pre_range_exact)
        return pre_range, kernel_pre, [zero(F), zero(F)], [zero(F), zero(F)]
      end
    elseif degree == :b13
      # b13 kernel
      if derivative in [0, 1, 2, 3, 4]
        kernel_pre = precompute_kernel(F, degree, derivative, eqs, precompute, pre_range_exact)
        kernel_d1_pre = precompute_kernel(F, degree, derivative+1, eqs, precompute, pre_range_exact)
        kernel_d2_pre = precompute_kernel(F, degree, derivative+2, eqs, precompute, pre_range_exact)
        return pre_range, kernel_pre, kernel_d1_pre, kernel_d2_pre
      elseif derivative == 5
        kernel_pre = precompute_kernel(F, degree, derivative, eqs, precompute, pre_range_exact)        
        kernel_d1_pre = precompute_kernel(F, degree, derivative+1, eqs, precompute, pre_range_exact)
        return pre_range, kernel_pre, kernel_d1_pre, [zero(F), zero(F)] 
      else # derivative == 6 && degree == :b13
        kernel_pre = precompute_kernel(F, degree, derivative, eqs, precompute, pre_range_exact)
        return pre_range, kernel_pre, [zero(F), zero(F)], [zero(F), zero(F)]
      end
  end
end

function precompute_kernel(F, degree, derivative, eqs, precompute, pre_range_exact)
    kernel = ConvolutionKernel(Val(degree), Val(derivative))
    kernel_pre = zeros(F, Int64(precompute), 2*eqs)
    for i = 1:2*eqs
        kernel_pre[:,i] .= F.(kernel.(pre_range_exact .- big(eqs//1) .+ big(i//1) .- big(1//1)))
    end
    return kernel_pre
end