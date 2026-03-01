"""
    get_precomputed_kernel_and_range(degree::Symbol; precompute, float_type::Type{T},
                                     derivative::Int=0, subgrid::Symbol=:cubic) where T

Retrieve precomputed kernel values for fast convolution interpolation.

For `:cubic` and `:quintic` subgrid modes (default), uses pre-shipped kernel tables
computed at exact rational precision. No disk I/O or caching required.

For `:linear`/`:nearest` subgrid, the top derivative of each kernel, or BigFloat
precision, computes high-resolution tables on demand and caches them to disk.

# Returns
Tuple of `(pre_range, kernel_pre, kernel_d1_pre, kernel_d2_pre)`.
"""

function get_precomputed_kernel_and_range(degree::Symbol;
            precompute::Int=101, float_type::Type{T},
            derivative::Int=0, subgrid::Symbol=:cubic) where {T}

    # Use cache path for: linear/nearest subgrid, BigFloat, or top derivative
    if subgrid in (:linear, :nearest) || float_type <: BigFloat || 
        _is_top_derivative(degree, derivative) || precompute != 101
        return _get_cached_kernel(degree; precompute=precompute,
                    float_type=float_type, derivative=derivative, subgrid=subgrid)
    end

    # For :cubic and :quintic subgrid at Float64/Float32, use shipped tables
    return get_shipped_kernel_tables(degree, derivative, float_type)
end

# Top derivative: only :linear subgrid is possible, not shipped
const _max_shipped_derivative = Dict(
    :a0 => 0, :a1 => 0,
    :a3 => 0, :a4 => 0, :a5 => 0, :a7 => 0,
    :b5 => 2, :b7 => 3, :b9 => 4, :b11 => 5, :b13 => 5,
)

_is_top_derivative(degree::Symbol, derivative::Int) =
    derivative > get(_max_shipped_derivative, degree, -1)

function _get_cached_kernel(degree::Symbol;
            precompute::Int, float_type::Type{T},
            derivative::Int=0, subgrid::Symbol=:linear) where {T}

    cache_dir = @get_scratch!("precomputed_kernels")
    degree_str = string(degree)
    precompute = degree == :a0 || degree == :a1 ? 2 : precompute

    # Build cache file paths
    if float_type <: BigFloat
        prec = precision(BigFloat)
        type_tag = "BigFloat$(prec)"
    else
        type_tag = string(float_type)
    end

    cache_file_range = joinpath(cache_dir, "$(degree_str)_$(Int64(precompute))_$(type_tag)_range.jls")
    cache_file_d0 = joinpath(cache_dir, "$(degree_str)_$(Int64(precompute))_$(type_tag)_derivative_$(derivative)_kernel.jls")
    cache_file_d1 = joinpath(cache_dir, "$(degree_str)_$(Int64(precompute))_$(type_tag)_derivative_$(derivative+1)_kernel.jls")
    cache_file_d2 = joinpath(cache_dir, "$(degree_str)_$(Int64(precompute))_$(type_tag)_derivative_$(derivative+2)_kernel.jls")

    if (degree == :a0 || degree == :a1) && isfile(cache_file_range) && isfile(cache_file_d0)
        return deserialize(cache_file_range), deserialize(cache_file_d0), nothing, nothing
    elseif subgrid == :linear && isfile(cache_file_range) && isfile(cache_file_d0)
        return deserialize(cache_file_range), deserialize(cache_file_d0), nothing, nothing
    elseif subgrid == :cubic && isfile(cache_file_range) && isfile(cache_file_d0) && isfile(cache_file_d1)
        return deserialize(cache_file_range), deserialize(cache_file_d0), deserialize(cache_file_d1), nothing
    elseif subgrid == :quintic && isfile(cache_file_range) && isfile(cache_file_d0) && isfile(cache_file_d1) && isfile(cache_file_d2)
        return deserialize(cache_file_range), deserialize(cache_file_d0), deserialize(cache_file_d1), deserialize(cache_file_d2)
    else
        @info "Generating precomputed kernel" degree float_type precompute derivative subgrid
        range_data, kernel_data_d0, kernel_data_d1, kernel_data_d2 =
                    precompute_kernel_and_range(degree; precompute=precompute,
                    F=float_type, derivative=derivative)
        serialize(cache_file_range, range_data)
        serialize(cache_file_d0, kernel_data_d0)
        if kernel_data_d1 isa AbstractMatrix
            serialize(cache_file_d1, kernel_data_d1)
        end
        if kernel_data_d2 isa AbstractMatrix
            serialize(cache_file_d2, kernel_data_d2)
        end
        return range_data, kernel_data_d0, kernel_data_d1, kernel_data_d2
    end
end