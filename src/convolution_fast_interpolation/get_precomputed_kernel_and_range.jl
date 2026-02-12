"""
    get_precomputed_kernel_and_range(degree::Symbol; precompute::Int=100_000, float_type::Type{T}=Float64) where T

Retrieve or generate precomputed kernel values for fast convolution interpolation.

# Arguments
- `degree::Symbol`: Kernel type (`:a0`, `:a1`, `:a3`, `:b5`, etc.)

# Keyword Arguments
- `precompute::Int=100_000`: Number of kernel values to precompute. Default provides
  maximum accuracy. Ignored for `:a0` and `:a1` (always uses 2 points)
- `float_type::Type{T}=Float64`: Floating-point precision for kernel values

# Returns
Tuple of `(pre_range, kernel_pre)`:
- `pre_range`: Vector of positions where kernel is precomputed (typically 0 to 1)
- `kernel_pre`: Matrix where `kernel_pre[i, j]` is the kernel value at position `i`
  for offset `j` from the evaluation point

# Details
This function manages a disk-based cache of precomputed kernel values using
Scratch.jl. On first use for a given (degree, precompute, float_type) combination,
it generates and caches the kernel values. Subsequent calls load from the cache,
making initialization nearly instantaneous.

**Cache Location**: Stored in a scratch space directory managed by Scratch.jl

**Precision Handling**: 
- For standard types (`Float64`, `Float32`), cache files include the type name
- For `BigFloat`, cache files include the precision (e.g., `BigFloat256`)
- This ensures correct precision across different Julia sessions

**Special Cases**:
- `:a0` and `:a1`: Always use `precompute=2` (no kernel precomputation needed)
- Higher-order kernels: Use full precompute resolution for accuracy

# Caching Strategy
The cache is persistent across Julia sessions and shared across all uses of the
package. This means:
- First time: ~0.1-10 seconds to generate (depends on kernel order and precompute size)
- Subsequent times: <1ms to load from disk
- Different precisions maintain separate caches

# Examples
```julia
# Standard usage with default settings
pre_range, kernel_pre = get_precomputed_kernel_and_range(:b5)
# First call: generates and caches, takes ~1 second
# Subsequent calls: loads from cache, takes ~1ms

# Custom precompute resolution (lower accuracy, faster initialization)
pre_range, kernel_pre = get_precomputed_kernel_and_range(:b5, precompute=10_000)

# High-precision computation
pre_range, kernel_pre = get_precomputed_kernel_and_range(:b5, float_type=BigFloat)
```

See also: `precompute_kernel_and_range`, `FastConvolutionInterpolation`.
"""
function get_precomputed_kernel_and_range(degree::Symbol;
                precompute::Union{Int,Rational{P}}=1//100, float_type::Type{T}, derivative::Int=0) where {T,P}
    cache_dir = @get_scratch!("precomputed_kernels")
    degree_str = string(degree)
    precompute = degree == :a0 || degree == :a1 ? 2 : precompute

    # Add precision info for BigFloat
    if float_type <: BigFloat
        prec = precision(BigFloat)
        cache_file_range = joinpath(cache_dir, "$(degree_str)_$(Int64(precompute))_BigFloat$(prec)_range.jls")
        cache_file_d0 = joinpath(cache_dir, "$(degree_str)_$(Int64(precompute))_BigFloat$(prec)_derivative_$(derivative)_kernel.jls")
        cache_file_d1 = joinpath(cache_dir, "$(degree_str)_$(Int64(precompute))_BigFloat$(prec)_derivative_$(derivative+1)_kernel.jls")
        cache_file_d2 = joinpath(cache_dir, "$(degree_str)_$(Int64(precompute))_BigFloat$(prec)_derivative_$(derivative+2)_kernel.jls")
    else
        type_str = string(float_type)  # "Float64", "Float32", etc.
        cache_file_range = joinpath(cache_dir, "$(degree_str)_$(Int64(precompute))_$(type_str)_range.jls")
        cache_file_d0 = joinpath(cache_dir, "$(degree_str)_$(Int64(precompute))_$(type_str)_derivative_$(derivative)_kernel.jls")
        cache_file_d1 = joinpath(cache_dir, "$(degree_str)_$(Int64(precompute))_$(type_str)_derivative_$(derivative+1)_kernel.jls")
        cache_file_d2 = joinpath(cache_dir, "$(degree_str)_$(Int64(precompute))_$(type_str)_derivative_$(derivative+2)_kernel.jls")
    end
    
    if isfile(cache_file_range) && isfile(cache_file_d0) && isfile(cache_file_d1) && isfile(cache_file_d2)
        return deserialize(cache_file_range), deserialize(cache_file_d0), deserialize(cache_file_d1), deserialize(cache_file_d2)
    else
        @info "Generating precomputed kernel of length $precompute for:\n
                    - fast convolution kernel $degree\n
                    - with $(float_type <: BigFloat ? "BigFloat"*"$(precision(BigFloat))" : float_type) precision\n
                    - derivative order $derivative\n
                    - (first time only)..."
        range_data, kernel_data_d0, kernel_data_d1, kernel_data_d2 = 
                            precompute_kernel_and_range(degree; precompute=precompute, F=float_type, 
                            derivative=derivative)
        serialize(cache_file_range, range_data)
        serialize(cache_file_d0, kernel_data_d0)
        serialize(cache_file_d1, kernel_data_d1)
        serialize(cache_file_d2, kernel_data_d2)

        return range_data, kernel_data_d0, kernel_data_d1, kernel_data_d2
    end
end