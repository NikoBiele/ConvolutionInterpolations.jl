"""
    precompute_kernel_and_range(degree::Symbol; precompute::Int=100_000, F::Type{<:AbstractFloat}=Float64)

Generate precomputed kernel values on a uniform grid for fast interpolation.

# Arguments
- `degree::Symbol`: Kernel type (`:a0`, `:a1`, `:a3`, `:b5`, etc.)

# Keyword Arguments
- `precompute::Int=100_000`: Number of uniformly spaced points at which to evaluate
  the kernel. Default value provides maximum accuracy (kernel interpolation error
  becomes negligible). Values below ~10,000 may degrade interpolation accuracy
- `F::Type{<:AbstractFloat}=Float64`: Floating-point type for computations and storage

# Returns
Tuple of `(pre_range, kernel_pre)`:
- `pre_range`: Vector of length `precompute` with uniformly spaced values from 0 to 1
- `kernel_pre`: Matrix of size `(precompute, 2*eqs)` where:
  - `eqs` is the kernel support radius (number of equations)
  - Column `i` contains kernel values for offset `i - eqs` from evaluation point
  - Row `j` contains kernel values at position `pre_range[j]`

# Details
This function evaluates the convolution kernel at uniformly spaced positions and
organizes the results for efficient lookup during interpolation. The kernel is
evaluated at all relevant offsets from the evaluation point.

**Special Cases**:
- `:a0`, `:a1`: Returns minimal `pre_range = [0.0, 1.0]` (no precomputation needed)
- Higher-order kernels: Full uniform grid from 0 to 1

**Kernel Support**: For a kernel with `eqs` equations (support radius), the kernel
extends from `-(eqs-1)` to `eqs` (total support: `2*eqs-1` points). The precomputed
matrix stores values for offsets from `-(eqs-1)` to `eqs`, allowing efficient
convolution evaluation.

# Algorithm
1. Create kernel function object for the specified degree
2. Determine kernel support size (`eqs`)
3. Generate uniform grid `pre_range` from 0 to 1
4. For each offset position `i` in `[-(eqs-1), eqs]`:
   - Evaluate kernel at `pre_range .- eqs .+ i .- 1`
   - Store in column `i + eqs` of `kernel_pre`

# Performance
Generation time depends on kernel order and precompute size:
- `:a3` with 100k points: ~50ms
- `:b5` with 100k points: ~200ms
- `:b13` with 100k points: ~1s

Results are cached by `get_precomputed_kernel_and_range`, so this function
typically runs only once per (degree, precompute, precision) combination.

# Examples
```julia
# Standard precomputation for b5 kernel
pre_range, kernel_pre = precompute_kernel_and_range(:b5)
# pre_range: 100,000 points from 0 to 1
# kernel_pre: (100000, 10) matrix for eqs=5

# Lower resolution (faster but less accurate)
pre_range, kernel_pre = precompute_kernel_and_range(:b5, precompute=10_000)

# High-precision precomputation
setprecision(256)
pre_range, kernel_pre = precompute_kernel_and_range(:b5, F=BigFloat)
```

See also: `get_precomputed_kernel_and_range`, `ConvolutionKernel`, `FastConvolutionInterpolation`.
"""
function precompute_kernel_and_range(degree::Symbol; precompute::P=100_000, F::Type{<:AbstractFloat}=Float64) where {P}
    kernel = ConvolutionKernel(Val(degree))
    eqs = get_equations_for_degree(degree)
    pre_range = degree == :a0 || degree == :a1 ? F[zero(F), one(F)] : F[zero(F) + F(i-1)/F(precompute-1) for i in 1:precompute]
    kernel_pre = zeros(F, precompute, 2*eqs)
    for i = 1:2*eqs
        kernel_pre[:,i] .= kernel.(pre_range .- eqs .+ i .- 1)
    end

    return pre_range, kernel_pre
end