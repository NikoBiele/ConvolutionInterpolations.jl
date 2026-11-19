"""
    convolution_resample(knots_in, knots_out, values)
    convolution_resample(knots_in, knots_out, values; kernel, derivative, bc)

Resample `values` from the uniform grid `knots_in` to the points `knots_out` by high-order
convolution interpolation, one axis at a time. For grid-to-grid work this avoids building an
interpolant and evaluating it point by point.

# Arguments
- `knots_in`: the input grid, an `AbstractVector` in 1D or a tuple with one vector per axis.
  Every axis must be increasing and uniformly spaced.
- `knots_out`: the output points of each axis, in the same format. They need not be
  uniformly spaced, but along every axis they must lie within the range of `knots_in`:
  resampling does not extrapolate.
- `values`: the data on `knots_in`, with `size(values, d) == length(knots_in[d])`.

An `ArgumentError` is thrown when the input knots do not match the data, are not uniform, or
when an output point lies outside their range. For data on a nonuniform grid, or to
extrapolate, build an interpolant with `convolution_interpolation` instead.

# Keyword Arguments
With keywords, `knots_in` and `knots_out` are tuples in every dimension, so `(x,)` and
`(x_new,)` in 1D.
- `kernel=:b7`: the convolution kernel, a `Symbol` for all axes or a tuple with one per
  axis, from the kernels of uniform grids (see `convolution_interpolation`).
- `derivative=0`: the derivative order, an `Int` for all axes or a tuple with one per axis,
  within the range of the kernel (see `convolution_interpolation`). Antiderivatives are not
  supported here; use `convolution_interpolation` with a negative `derivative`.
- `bc=:detect`: the boundary condition, as in `convolution_interpolation` on a uniform grid:
  `:detect`, `:poly`, `:linear` or `:quadratic`.

# Axes with few points
The high-order boundary values need a minimum number of points along an axis: with the
default `bc = :detect`, 5 for `:a3` to `:a7`, 7 for `:b5` and 9 for `:b7` to `:b13`, and one
fewer with `bc = :poly`. An axis with fewer points is resampled without an error, with
linear boundary values at its ends: exact for a straight line, less accurate otherwise.

# Returns
The array of resampled values, with `size(result, d) == length(knots_out[d])`.

# Examples
```julia
# 1D resampling
x_coarse = range(0.0, 2π, length=20)
x_fine   = range(0.0, 2π, length=200)
y_coarse = sin.(x_coarse)
y_fine   = convolution_resample(x_coarse, x_fine, y_coarse)

# 1D with a keyword: the knots as one-element tuples
y_fine_b5 = convolution_resample((x_coarse,), (x_fine,), y_coarse; kernel=:b5)

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
function convolution_resample(knots_in::AbstractVector, knots_out::AbstractVector,
                              values::AbstractVector{T}) where {T}
    return convolution_resample_internal((knots_in,), (knots_out,), values,
                                        (:b7,),
                                        (0,),
                                        ((:detect,:detect),))
end

# 2D convenience wrapper
function convolution_resample(knots_in::NTuple{2,AbstractVector},
                              knots_out::NTuple{2,AbstractVector},
                              values::AbstractArray{T,2}) where {T}
    return convolution_resample_internal(knots_in, knots_out, values,
                                        (:b7,:b7),
                                        (0,0),
                                        ((:detect,:detect),(:detect,:detect)))
end

# 3D convenience wrapper
function convolution_resample(knots_in::NTuple{3,AbstractVector},
                              knots_out::NTuple{3,AbstractVector},
                              values::AbstractArray{T,3}) where {T}
    return convolution_resample_internal(knots_in, knots_out, values,
                                        (:b7,:b7,:b7),
                                        (0,0,0),
                                        ((:detect,:detect),(:detect,:detect),
                                         (:detect,:detect)))
end

# 4D convenience wrapper
function convolution_resample(knots_in::NTuple{4,AbstractVector},
                              knots_out::NTuple{4,AbstractVector},
                              values::AbstractArray{T,4}) where {T}
    return convolution_resample_internal(knots_in, knots_out, values,
                                        (:b7,:b7,:b7,:b7),
                                        (0,0,0,0),
                                        ((:detect,:detect),(:detect,:detect),
                                         (:detect,:detect),(:detect,:detect)))
end

# 5D convenience wrapper
function convolution_resample(knots_in::NTuple{5,AbstractVector},
                              knots_out::NTuple{5,AbstractVector},
                              values::AbstractArray{T,5}) where {T}
    return convolution_resample_internal(knots_in, knots_out, values,
                                        (:b7,:b7,:b7,:b7,:b7),
                                        (0,0,0,0,0),
                                        ((:detect,:detect),(:detect,:detect),
                                         (:detect,:detect),(:detect,:detect),
                                         (:detect,:detect)))
end

# 6D convenience wrapper
function convolution_resample(knots_in::NTuple{6,AbstractVector},
                              knots_out::NTuple{6,AbstractVector},
                              values::AbstractArray{T,6}) where {T}
    return convolution_resample_internal(knots_in, knots_out, values,
                                        (:b7,:b7,:b7,:b7,:b7,:b7),
                                        (0,0,0,0,0,0),
                                        ((:detect,:detect),(:detect,:detect),
                                         (:detect,:detect),(:detect,:detect),
                                         (:detect,:detect),(:detect,:detect)))
end

# 7D convenience wrapper
function convolution_resample(knots_in::NTuple{7,AbstractVector},
                              knots_out::NTuple{7,AbstractVector},
                              values::AbstractArray{T,7}) where {T}
    return convolution_resample_internal(knots_in, knots_out, values,
                                        (:b7,:b7,:b7,:b7,:b7,:b7,:b7),
                                        (0,0,0,0,0,0,0),
                                        ((:detect,:detect),(:detect,:detect),
                                         (:detect,:detect),(:detect,:detect),
                                         (:detect,:detect),(:detect,:detect),
                                         (:detect,:detect)))
end

# 8D convenience wrapper
function convolution_resample(knots_in::NTuple{8,AbstractVector},
                              knots_out::NTuple{8,AbstractVector},
                              values::AbstractArray{T,8}) where {T}
    return convolution_resample_internal(knots_in, knots_out, values,
                                        (:b7,:b7,:b7,:b7,:b7,:b7,:b7,:b7),
                                        (0,0,0,0,0,0,0,0),
                                        ((:detect,:detect),(:detect,:detect),
                                         (:detect,:detect),(:detect,:detect),
                                         (:detect,:detect),(:detect,:detect),
                                         (:detect,:detect),(:detect,:detect)))
end

# 9D convenience wrapper
function convolution_resample(knots_in::NTuple{9,AbstractVector},
                              knots_out::NTuple{9,AbstractVector},
                              values::AbstractArray{T,9}) where {T}
    return convolution_resample_internal(knots_in, knots_out, values,
                                        (:b7,:b7,:b7,:b7,:b7,:b7,:b7,:b7,:b7),
                                        (0,0,0,0,0,0,0,0,0),
                                        ((:detect,:detect),(:detect,:detect),
                                         (:detect,:detect),(:detect,:detect),
                                         (:detect,:detect),(:detect,:detect),
                                         (:detect,:detect),(:detect,:detect),
                                         (:detect,:detect)))
end

# 10D convenience wrapper
function convolution_resample(knots_in::NTuple{10,AbstractVector},
                              knots_out::NTuple{10,AbstractVector},
                              values::AbstractArray{T,10}) where {T}
    return convolution_resample_internal(knots_in, knots_out, values,
                                        (:b7,:b7,:b7,:b7,:b7,:b7,:b7,:b7,:b7,:b7),
                                        (0,0,0,0,0,0,0,0,0,0),
                                        ((:detect,:detect),(:detect,:detect),
                                         (:detect,:detect),(:detect,:detect),
                                         (:detect,:detect),(:detect,:detect),
                                         (:detect,:detect),(:detect,:detect),
                                         (:detect,:detect),(:detect,:detect)))
end

function convolution_resample(knots_in::NTuple{N,AbstractVector},
                              knots_out::NTuple{N,AbstractVector},
                              values::AbstractArray{T,N};
                              kernel::Union{Symbol,NTuple{N,Symbol}}=:b7,
                              derivative::Union{Int,NTuple{N,Int}}=0,
                              bc::Union{Symbol,Tuple{Symbol,Symbol},
                                        NTuple{N,Tuple{Symbol,Symbol}}}=:detect
                              ) where {T,N}

    # check and normalize inputs
    knots_in_tuple = knots_in isa AbstractVector ? (T.(knots_in),) : 
                  knots_in isa NTuple{N,AbstractVector} ? ntuple(d -> T.(knots_in[d]), N) :
                    throw(ArgumentError("Invalid knots_in specification: $knots_in."))
    knots_out_tuple = knots_out isa AbstractVector ? (T.(knots_out),) : 
                  knots_out isa NTuple{N,AbstractVector} ?
                  ntuple(d -> T.(knots_out[d]), N) :
                    throw(ArgumentError("Invalid knots_out specification: $knots_out."))
    kernels_tuple = kernel isa NTuple{N,Symbol} ? kernel :
                    kernel isa Symbol ? ntuple(_ -> kernel, N) :
                    kernel isa NTuple{1,Symbol} ? ntuple(_ -> kernel[1], N) :
                    throw(ArgumentError("Invalid kernel specification: $kernel."))
    derivatives_tuple = derivative isa NTuple{N,Int} ? derivative :
                    derivative isa Int ? ntuple(_ -> derivative, N) :
                    derivative isa NTuple{1,Int} ? ntuple(_ -> derivative[1], N) : 
                    throw(ArgumentError("Invalid derivative specification: $derivative."))
    bcs_tuple = bc isa NTuple{N,Tuple{Symbol,Symbol}} ? bc :
                    bc isa Tuple{Symbol,Symbol} ? ntuple(_ -> bc, N) :
                    bc isa NTuple{1,Tuple{Symbol,Symbol}} ? ntuple(_ -> bc[1], N) :
                    bc isa Symbol ? ntuple(_ -> (bc, bc), N) :
                    throw(ArgumentError("Invalid bc specification: $bc."))

    for d in 1:N
        if derivatives_tuple[d] < 0
            throw(ArgumentError("convolution_resample does not support antiderivatives " *
                                "(derivative=-1). Use convolution_interpolation with " *
                                "derivative=-1 instead."))
        end
    end

    return convolution_resample_internal(knots_in_tuple, knots_out_tuple, values,
                                        kernels_tuple,
                                        derivatives_tuple,
                                        bcs_tuple)

end

"""
    _check_resample_axis(knots_in, knots_out, n_values, d)

Validate axis `d` of a resampling call, throwing an `ArgumentError` otherwise: the input
knots match the data (`n_values` values along this axis), are increasing and uniformly
spaced, and every output point lies within their range. The resampling passes use one
spacing per axis and clamp to the boundary cells, so a call that fails any of these would
return wrong values without an error. The range test allows for rounding at the two ends.
"""
function _check_resample_axis(knots_in::AbstractVector, knots_out::AbstractVector,
                              n_values::Int, d::Int)
    n_in = length(knots_in)
    if n_in != n_values
        throw(ArgumentError("convolution_resample: knots_in has $n_in knots along " *
                            "dimension $d, but values has $n_values entries there."))
    end
    if n_in < 2
        throw(ArgumentError("convolution_resample: knots_in needs at least 2 knots along " *
                            "dimension $d, got $n_in."))
    end
    lo = first(knots_in)
    hi = last(knots_in)
    if !(hi > lo) || !is_uniform_grid(knots_in)
        throw(ArgumentError("convolution_resample: knots_in must be increasing and " *
                            "uniformly spaced along dimension $d. For data on a " *
                            "nonuniform grid, build an interpolant with " *
                            "convolution_interpolation and evaluate it at the new points."))
    end
    if isempty(knots_out)
        throw(ArgumentError("convolution_resample: knots_out is empty along dimension $d."))
    end
    spacing = (hi - lo) / (n_in - 1)
    tol = sqrt(eps(float(typeof(spacing)))) * spacing  # rounding allowance at the two ends
    if minimum(knots_out) < lo - tol || maximum(knots_out) > hi + tol
        throw(ArgumentError("convolution_resample: knots_out reaches outside the range " *
                            "of knots_in along dimension $d ([$lo, $hi]). Resampling " *
                            "does not extrapolate; build an interpolant with " *
                            "convolution_interpolation and its extrap keyword instead."))
    end
    return nothing
end

function convolution_resample_internal(knots_in::NTuple{N,AbstractVector},
                               knots_out::NTuple{N,AbstractVector},
                               values::AbstractArray{T,N},
                               kernels::NTuple{N,Symbol},
                               derivatives::NTuple{N,Int},
                               bcs::NTuple{N,Tuple{Symbol,Symbol}}) where {T,N}

    # reject calls the passes below would answer wrongly: mismatched or nonuniform input
    # knots, and output points outside the input range
    for d in 1:N
        _check_resample_axis(knots_in[d], knots_out[d], size(values, d), d)
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
        h_d    = T((last(knots_in[d]) - first(knots_in[d]))/(length(knots_in[d]) - 1))

        n_coef = n_in + 2*(eqs_d - 1)

        # kernel and derivative order of this dimension, passed to the slice evaluator
        kernel_val     = Val(kernels[d])
        derivative_val = Val(derivatives[d])

        # allocate once per dimension pass
        coef_buf  = zeros(T, n_coef)
        out_buf   = zeros(T, n_out)
        workspace = BoundaryWorkspace(T, Val(1), eqs_d, n_in)
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
            _resample_slice!(out_buf, coef_buf, knots_out[d], x_expanded_1, h_d, eqs_d,
                             n_coef, kernel_val, derivative_val)

            # write out_buf into next
            for k in 1:n_out
                next[offset_out + (k-1)*strides_out[d] + 1] =
                    out_buf[k] * (-one(T)/h_d)^derivatives[d]
            end
        end

        current_ref[] = next
    end
    return current_ref[]
end

# Evaluate one slice (its coefficients in coef_buf) at all output knots of the dimension,
# with the exact column polynomials of kernel K at derivative order D. Specialized per
# (K, D), so the column polynomials are constants inside the loop.
function _resample_slice!(out_buf::Vector{T}, coef_buf::Vector{T},
                          knots_out_d::AbstractVector,
                          x_expanded_1::T, h_d::T, eqs_d::Int, n_coef::Int,
                          ::Val{K}, ::Val{D}) where {T,K,D}
    @inbounds for k in eachindex(out_buf)
        xq           = knots_out_d[k]
        i_float      = (xq - x_expanded_1) / h_d + one(T)
        i            = clamp(floor(Int, i_float), eqs_d, n_coef - eqs_d)
        knot_i       = x_expanded_1 + (i - 1) * h_d
        x_diff_right = one(T) - (xq - knot_i) / h_d

        # kernel weights of all 2·eqs columns (column c ↔ coef_buf[i − eqs + c])
        w = _kernel_weights(Val(K), Val(D), x_diff_right)
        s = zero(T)
        for c in 1:length(w)
            s += coef_buf[i - eqs_d + c] * w[c]
        end
        out_buf[k] = s
    end
    return out_buf
end