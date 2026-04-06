"""
    convolution_resample(knots_in, knots_out, values)
    convolution_resample(knots_in, knots_out, values; kernel, derivative, bc, subgrid, precompute)

Resample `values` from the uniform grid `knots_in` to the uniform grid `knots_out` using
high-order separable convolution interpolation.

# Arguments
- `knots_in`: input grid — an `AbstractVector` (1D) or `NTuple{N,AbstractVector}` (ND)
- `knots_out`: output grid — same format as `knots_in`
- `values`: array of values on `knots_in`, must have `size(values, d) == length(knots_in[d])`

# Keyword Arguments
- `kernel`: convolution kernel, default `:b9`. A `Symbol` applies to all dimensions;
  an `NTuple{N,Symbol}` specifies per-dimension kernels. See [Kernel Reference] for options.
- `derivative`: derivative order, default `0`. An `Int` applies to all dimensions;
  an `NTuple{N,Int}` specifies per-dimension orders. Negative values compute antiderivatives.
- `bc`: boundary condition, default `:detect`. See [Boundary Conditions] for options.
- `subgrid`: subgrid interpolation method, default `:cubic`. Options: `:linear`, `:cubic`, `:quintic`.
- `precompute`: number of precomputed kernel points, default `101`.

# Default kernels by dimension
The convenience wrappers (called without keywords) use dimension-appropriate defaults:
- 1D, 2D: `:b9` with `:cubic` subgrid
- 3D–10D: `:b9` with `:linear` subgrid

# Returns
Array of resampled values with `size(result, d) == length(knots_out[d])`.

# Performance
Resampling is separable — each dimension is processed independently in a single pass.
Allocation count is independent of grid size, and the function is fully type stable
when called via the convenience wrappers. For grid-to-grid operations this is
significantly faster than constructing a full interpolant and evaluating at each point:

| Grid | `convolution_resample` | construct + eval | Speedup |
|------|----------------------|------------------|---------|
| 1D 500→1000 | 22 μs | 29 μs | 1.3× |
| 2D 100²→200² | 1.9 ms | 7.0 ms | 3.6× |
| 3D 30³→60³ | 8.9 ms | 127 ms | 14× |

# Examples
```julia
# 1D resampling
x_coarse = range(0.0, 2π, length=20)
x_fine   = range(0.0, 2π, length=200)
y_coarse = sin.(x_coarse)
y_fine   = convolution_resample(x_coarse, x_fine, y_coarse)

# 2D resampling
xs = range(0.0, 2π, length=20)
ys = range(0.0, 2π, length=20)
z  = [sin(x)*cos(y) for x in xs, y in ys]
xs_fine = range(0.0, 2π, length=100)
ys_fine = range(0.0, 2π, length=100)
z_fine  = convolution_resample((xs, ys), (xs_fine, ys_fine), z)

# Resample and differentiate in one pass
z_dx = convolution_resample((xs, ys), (xs_fine, ys_fine), z; derivative=(1, 0))
```

See also: [`convolution_interpolation`](@ref), [`convolution_smooth`](@ref)
"""

# 1D convenience wrapper
function convolution_resample(knots_in::AbstractVector, knots_out::AbstractVector, values::AbstractVector{T}) where {T}
    return convolution_resample_internal((knots_in,), (knots_out,), values,
                                        101,
                                        (:b9,),
                                        (0,),
                                        ((:detect,:detect),),
                                        (:cubic,))
end

# 2D convenience wrapper
function convolution_resample(knots_in::NTuple{2,AbstractVector}, knots_out::NTuple{2,AbstractVector},
                                values::AbstractArray{T,2}) where {T}
    return convolution_resample_internal(knots_in, knots_out, values,
                                        101,
                                        (:b9,:b9),
                                        (0,0),
                                        ((:detect,:detect),(:detect,:detect)),
                                        (:cubic,:cubic))
end

# 3D convenience wrapper
function convolution_resample(knots_in::NTuple{3,AbstractVector}, knots_out::NTuple{3,AbstractVector},
                                values::AbstractArray{T,3}) where {T}
    return convolution_resample_internal(knots_in, knots_out, values,
                                        101,
                                        (:b9,:b9,:b9),
                                        (0,0,0),
                                        ((:detect,:detect),(:detect,:detect),(:detect,:detect)),
                                        (:linear,:linear,:linear))
end

# 4D convenience wrapper
function convolution_resample(knots_in::NTuple{4,AbstractVector}, knots_out::NTuple{4,AbstractVector},
                                values::AbstractArray{T,4}) where {T}
    return convolution_resample_internal(knots_in, knots_out, values,
                                        101,
                                        (:b9,:b9,:b9,:b9),
                                        (0,0,0,0),
                                        ((:detect,:detect),(:detect,:detect),(:detect,:detect),(:detect,:detect)),
                                        (:linear,:linear,:linear,:linear))
end

# 5D convenience wrapper
function convolution_resample(knots_in::NTuple{5,AbstractVector}, knots_out::NTuple{5,AbstractVector},
                                values::AbstractArray{T,5}) where {T}
    return convolution_resample_internal(knots_in, knots_out, values,
                                        101,
                                        (:b9,:b9,:b9,:b9,:b9),
                                        (0,0,0,0,0),
                                        ((:detect,:detect),(:detect,:detect),(:detect,:detect),(:detect,:detect),(:detect,:detect)),
                                        (:linear,:linear,:linear,:linear,:linear))
end

# 6D convenience wrapper
function convolution_resample(knots_in::NTuple{6,AbstractVector}, knots_out::NTuple{6,AbstractVector},
                                values::AbstractArray{T,6}) where {T}
    return convolution_resample_internal(knots_in, knots_out, values,
                                        101,
                                        (:b9,:b9,:b9,:b9,:b9,:b9),
                                        (0,0,0,0,0,0),
                                        ((:detect,:detect),(:detect,:detect),(:detect,:detect),(:detect,:detect),(:detect,:detect),(:detect,:detect)),
                                        (:linear,:linear,:linear,:linear,:linear,:linear))
end

# 7D convenience wrapper
function convolution_resample(knots_in::NTuple{7,AbstractVector}, knots_out::NTuple{7,AbstractVector},
                                values::AbstractArray{T,7}) where {T}
    return convolution_resample_internal(knots_in, knots_out, values,
                                        101,
                                        (:b9,:b9,:b9,:b9,:b9,:b9,:b9),
                                        (0,0,0,0,0,0,0),
                                        ((:detect,:detect),(:detect,:detect),(:detect,:detect),(:detect,:detect),(:detect,:detect),(:detect,:detect),(:detect,:detect)),
                                        (:linear,:linear,:linear,:linear,:linear,:linear,:linear))
end

# 8D convenience wrapper
function convolution_resample(knots_in::NTuple{8,AbstractVector}, knots_out::NTuple{8,AbstractVector},
                                values::AbstractArray{T,8}) where {T}
    return convolution_resample_internal(knots_in, knots_out, values,
                                        101,
                                        (:b9,:b9,:b9,:b9,:b9,:b9,:b9,:b9),
                                        (0,0,0,0,0,0,0,0),
                                        ((:detect,:detect),(:detect,:detect),(:detect,:detect),(:detect,:detect),(:detect,:detect),(:detect,:detect),(:detect,:detect),(:detect,:detect)),
                                        (:linear,:linear,:linear,:linear,:linear,:linear,:linear,:linear))
end


# 9D convenience wrapper
function convolution_resample(knots_in::NTuple{9,AbstractVector}, knots_out::NTuple{9,AbstractVector},
                                values::AbstractArray{T,9}) where {T}
    return convolution_resample_internal(knots_in, knots_out, values,
                                        101,
                                        (:b9,:b9,:b9,:b9,:b9,:b9,:b9,:b9,:b9),
                                        (0,0,0,0,0,0,0,0,0),
                                        ((:detect,:detect),(:detect,:detect),(:detect,:detect),(:detect,:detect),(:detect,:detect),(:detect,:detect),(:detect,:detect),(:detect,:detect),(:detect,:detect)),
                                        (:linear,:linear,:linear,:linear,:linear,:linear,:linear,:linear,:linear))
end

# 10D convenience wrapper
function convolution_resample(knots_in::NTuple{10,AbstractVector}, knots_out::NTuple{10,AbstractVector},
                                values::AbstractArray{T,10}) where {T}
    return convolution_resample_internal(knots_in, knots_out, values,
                                        101,
                                        (:b9,:b9,:b9,:b9,:b9,:b9,:b9,:b9,:b9,:b9),
                                        (0,0,0,0,0,0,0,0,0,0),
                                        ((:detect,:detect),(:detect,:detect),(:detect,:detect),(:detect,:detect),(:detect,:detect),(:detect,:detect),(:detect,:detect),(:detect,:detect),(:detect,:detect),(:detect,:detect)),
                                        (:linear,:linear,:linear,:linear,:linear,:linear,:linear,:linear,:linear,:linear))
end

function convolution_resample(knots_in::NTuple{N,AbstractVector},
                               knots_out::NTuple{N,AbstractVector},
                               values::AbstractArray{T,N};
                               precompute::Int=101,
                               kernel::Union{Symbol,NTuple{N,Symbol}}=:b9,
                               derivative::Union{Int,NTuple{N,Int}}=0,
                               bc::Union{Symbol,Tuple{Symbol,Symbol},NTuple{N,Tuple{Symbol,Symbol}}}=:detect,
                               subgrid::Union{Symbol,NTuple{N,Symbol}}=:cubic) where {T,N}

    # check and normalize inputs
    knots_in_tuple = knots_in isa AbstractVector ? (T.(knots_in),) : 
                  knots_in isa NTuple{N,AbstractVector} ? ntuple(d -> T.(knots_in[d]), N) :
                    error("Invalid knots_in specification: $knots_in.")
    knots_out_tuple = knots_out isa AbstractVector ? (T.(knots_out),) : 
                  knots_out isa NTuple{N,AbstractVector} ? ntuple(d -> T.(knots_out[d]), N) :
                    error("Invalid knots_out specification: $knots_out.")
    kernels_tuple = kernel isa NTuple{N,Symbol} ? kernel :
                    kernel isa Symbol ? ntuple(_ -> kernel, N) :
                    kernel isa NTuple{1,Symbol} ? ntuple(_ -> kernel[1], N) :
                    error("Invalid kernel specification: $kernel.")
    derivatives_tuple = derivative isa NTuple{N,Int} ? derivative :
                    derivative isa Int ? ntuple(_ -> derivative, N) :
                    derivative isa NTuple{1,Int} ? ntuple(_ -> derivative[1], N) : 
                    error("Invalid derivative specification: $derivative.")
    subgrids_tuple = subgrid isa NTuple{N,Symbol} ? subgrid :
                    subgrid isa Symbol ? ntuple(_ -> subgrid, N) : 
                    subgrid isa NTuple{1,Symbol} ? ntuple(_ -> subgrid[1], N) :
                    error("Invalid subgrid specification: $subgrid.")
    bcs_tuple = bc isa NTuple{N,Tuple{Symbol,Symbol}} ? bc :
                    bc isa Tuple{Symbol,Symbol} ? ntuple(_ -> bc, N) :
                    bc isa NTuple{1,Tuple{Symbol,Symbol}} ? ntuple(_ -> bc[1], N) :
                    bc isa Symbol ? ntuple(_ -> (bc, bc), N) :
                    error("Invalid bc specification: $bc.")

    # validate subgrid/kernel combinations
    for d in 1:N
        if subgrids_tuple[d] != :linear && kernels_tuple[d] in (:a0, :a1)
            error("Cannot use $(subgrids_tuple[d]) subgrid with $(kernels_tuple[d]) kernel. Use :linear instead.")
        end
    end

    for d in 1:N
        if derivatives_tuple[d] < 0
            error("convolution_resample does not support antiderivatives (derivative=-1). "*
                "Use convolution_interpolation with derivative=-1 instead.")
        end
    end

    return convolution_resample_internal(knots_in_tuple, knots_out_tuple, values,
                                        precompute,
                                        kernels_tuple,
                                        derivatives_tuple,
                                        bcs_tuple,
                                        subgrids_tuple)

end

function convolution_resample_internal(knots_in::NTuple{N,AbstractVector},
                               knots_out::NTuple{N,AbstractVector},
                               values::AbstractArray{T,N},
                               precompute::Int,
                               kernels::NTuple{N,Symbol},
                               derivatives::NTuple{N,Int},
                               bcs::NTuple{N,Tuple{Symbol,Symbol}},
                               subgrids::NTuple{N,Symbol}) where {T,N}

    # lookup tables once if all dims use same kernel/derivative/subgrid
    same_all = allequal(kernels) && allequal(derivatives) && allequal(subgrids)
    if same_all
        pre_range, kernel_pre, kd1_pre, kd2_pre = get_precomputed_kernel_and_range(
            kernels[1], precompute, T, derivatives[1], subgrids[1])
        eqs_same = DEGREE_TO_EQUATIONS[kernels[1]]
        l_range  = -(eqs_same-1):eqs_same
    end

    # pre-allocate a concrete buffer
    buf_in  = Array{T,N}(undef, size(values))
    copyto!(buf_in, values)

    # use Ref to avoid boxing of current
    current_ref = Ref{Array{T,N}}(buf_in)

    for d in 1:N
        current = current_ref[]
        n_in   = size(current, d)
        n_out  = length(knots_out[d])
        eqs_d  = DEGREE_TO_EQUATIONS[kernels[d]]
        h_d    = T(knots_in[d][2] - knots_in[d][1])
        n_coef = n_in + 2*(eqs_d - 1)
        sg     = subgrids[d]

        # allocate once per dimension pass
        coef_buf  = zeros(T, n_coef)
        out_buf   = zeros(T, n_out)
        workspace = BoundaryWorkspace(T, Val(1), eqs_d, n_in)
        if !same_all
            pre_range, kernel_pre, kd1_pre, kd2_pre = get_precomputed_kernel_and_range(
                kernels[d], precompute, T, derivatives[d], sg)
            l_range = -(eqs_d-1):eqs_d
        end
        n_pre        = length(pre_range)
        h_pre        = one(T) / T(n_pre - 1)
        x_expanded_1 = knots_in[d][1] - (eqs_d - 1) * h_d

        next_size    = ntuple(i -> i == d ? n_out : size(current, i), Val(N))
        next         = zeros(T, next_size)
        strides_in   = strides(current)
        strides_out  = strides(next)
        other_dims   = ntuple(i -> i < d ? i : i + 1, Val(N-1))
        other_ranges = ntuple(i -> 1:size(current, other_dims[i]), Val(N-1))

        # view into coef_buf interior — allocated once per pass
        vs_1d = view(coef_buf, eqs_d:(eqs_d + n_in - 1))

        for outer_idx in CartesianIndices(other_ranges)

            # compute base offsets using strides
            offset_in  = 0
            offset_out = 0
            for j in 1:N-1
                dim_j       = other_dims[j]
                offset_in  += (outer_idx[j] - 1) * strides_in[dim_j]
                offset_out += (outer_idx[j] - 1) * strides_out[dim_j]
            end

            # extract slice into interior of coef_buf
            fill!(coef_buf, zero(T))
            for k in 1:n_in
                coef_buf[k + eqs_d - 1] = current[offset_in + (k-1)*strides_in[d] + 1]
            end

            # apply BCs in-place
            apply_boundary_conditions_for_dim!(coef_buf, vs_1d, 1, (h_d,), eqs_d,
                                               (bcs[d],), kernels[d],
                                               workspace, (n_in,), Val(false))

            # evaluate at output knots into out_buf
            @inbounds for k in 1:n_out
                xq           = knots_out[d][k]
                i_float      = (xq - x_expanded_1) / h_d + one(T)
                i            = clamp(floor(Int, i_float), eqs_d, n_coef - eqs_d)
                knot_i       = x_expanded_1 + (i - 1) * h_d
                x_diff_left  = (xq - knot_i) / h_d
                x_diff_right = one(T) - x_diff_left

                continuous_idx = x_diff_right * T(n_pre - 1) + one(T)
                idx      = clamp(floor(Int, continuous_idx), 1, n_pre - 1)
                idx_next = idx + 1
                t        = continuous_idx - T(idx)

                if sg == :quintic
                    sum_f0  = zero(T); sum_f1  = zero(T)
                    sum_d0  = zero(T); sum_d1  = zero(T)
                    sum_dd0 = zero(T); sum_dd1 = zero(T)
                    for l in l_range
                        coef    = coef_buf[i + l]
                        col     = l + eqs_d
                        sum_f0  += coef * kernel_pre[idx,      col]
                        sum_f1  += coef * kernel_pre[idx_next, col]
                        sum_d0  += coef * kd1_pre[idx,      col]
                        sum_d1  += coef * kd1_pre[idx_next, col]
                        sum_dd0 += coef * kd2_pre[idx,      col]
                        sum_dd1 += coef * kd2_pre[idx_next, col]
                    end
                    out_buf[k] = quintic_hermite(t, sum_f0, sum_f1, sum_d0, sum_d1,
                                                 sum_dd0, sum_dd1, h_pre)

                elseif sg == :cubic
                    sum_f0 = zero(T); sum_f1 = zero(T)
                    sum_d0 = zero(T); sum_d1 = zero(T)
                    for l in l_range
                        coef   = coef_buf[i + l]
                        col    = l + eqs_d
                        sum_f0 += coef * kernel_pre[idx,      col]
                        sum_f1 += coef * kernel_pre[idx_next, col]
                        sum_d0 += coef * kd1_pre[idx,      col]
                        sum_d1 += coef * kd1_pre[idx_next, col]
                    end
                    out_buf[k] = cubic_hermite(t, sum_f0, sum_f1, sum_d0, sum_d1, h_pre)

                else # :linear
                    result_lower = zero(T)
                    result_upper = zero(T)
                    for l in l_range
                        coef         = coef_buf[i + l]
                        col          = l + eqs_d
                        result_lower += coef * kernel_pre[idx,      col]
                        result_upper += coef * kernel_pre[idx_next, col]
                    end
                    out_buf[k] = (one(T) - t) * result_lower + t * result_upper

                end
            end

            # write out_buf into next
            for k in 1:n_out
                next[offset_out + (k-1)*strides_out[d] + 1] = out_buf[k] * (-one(T)/h_d)^derivatives[d]
            end
        end

        current_ref[] = next
    end
    return current_ref[]
end