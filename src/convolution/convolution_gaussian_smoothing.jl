"""
    convolution_smooth(knots, values, B)

Apply separable Gaussian smoothing to `values` on the uniform grid `knots`.
Returns an array of the same size as `values` with noise reduced.

# Arguments
- `knots`: grid — an `AbstractVector` (1D) or `NTuple{N,AbstractVector}` (ND)
- `values`: array of values to smooth, must have `size(values, d) == length(knots[d])`
- `B`: Gaussian width parameter. Larger `B` means a narrower kernel (less smoothing),
  smaller `B` means a wider kernel (more smoothing). **B ≤ 0.1 recommended** for
  effective noise reduction.

# Returns
Array of smoothed values, same size and element type as `values`.

# Performance
Smoothing is separable and allocation-free in the inner loop. The Gaussian kernel
weights are precomputed once per call — `exp` is evaluated only `2*eqs+1` times
regardless of grid size or dimension:

| Grid | B=0.02 | B=0.05 | B=0.1 |
|------|--------|--------|-------|
| 1D n=1000 | 22 μs | 13 μs | 9.6 μs |
| 2D 200² | 1.7 ms | 0.99 ms | 0.72 ms |
| 3D 50³ | 9.3 ms | 5.7 ms | 4.4 ms |
| 4D 40⁴ | 273 ms | 206 ms | 169 ms |

# Examples
```julia
# 1D smoothing
x = range(0.0, 2π, length=500)
y_noisy = sin.(x) .+ 0.1.*randn(500)
y_smooth = convolution_smooth(x, y_noisy, 0.05)

# 2D smoothing
x = range(0.0, 2π, length=200)
y = range(0.0, 2π, length=200)
z_noisy  = [cos(xi)*sin(yi) for xi in x, yi in y] .+ 0.1.*randn(200,200)
z_smooth = convolution_smooth((x,y), z_noisy, 0.02)
```

See also: [`convolution_gaussian`](@ref), [`convolution_interpolation`](@ref)
"""
function convolution_smooth(knots::NTuple{N,AbstractVector}, values::AbstractArray{T,N}, B::Float64) where {T,N}
    eqs_d = ceil(Int, sqrt(-log(1e-12/2) / B))
    NT    = eqs_d

    # check that stencil fits within domain
    for d in 1:N
        n = length(knots[d])
        if eqs_d >= n
            error("Gaussian stencil half-width ($eqs_d points) exceeds grid size (n=$n) in dimension $d. "*
                  "Use a larger B (currently B=$B) or more grid points (need n > $eqs_d).")
        end
    end

    # precompute normalized kernel weights once — reused for all dims, all slices, all points
    norm  = θ(B, NT)
    kw    = zeros(T, 2*eqs_d+1)
    for l in -eqs_d:eqs_d
        kw[l + eqs_d + 1] = exp(-B * T(l)^2) / norm
    end
    l_range = -eqs_d:eqs_d

    buf_in  = Array{T,N}(undef, size(values))
    copyto!(buf_in, values)
    current_ref = Ref{Array{T,N}}(buf_in)

    for d in 1:N
        current = current_ref[]
        n       = size(current, d)
        h_d     = T(knots[d][2] - knots[d][1])
        n_coef  = n + 2*eqs_d

        coef_buf  = zeros(T, n_coef)
        out_buf   = zeros(T, n)
        workspace = BoundaryWorkspace(T, Val(1), eqs_d, n)
        vs_1d     = view(coef_buf, (eqs_d+1):(eqs_d + n))

        next         = zeros(T, size(current))
        strides_in   = strides(current)
        strides_out  = strides(next)
        other_dims   = ntuple(i -> i < d ? i : i + 1, Val(N-1))
        other_ranges = ntuple(i -> 1:size(current, other_dims[i]), Val(N-1))

        for outer_idx in CartesianIndices(other_ranges)
            offset_in  = 0
            offset_out = 0
            for j in 1:N-1
                dim_j       = other_dims[j]
                offset_in  += (outer_idx[j] - 1) * strides_in[dim_j]
                offset_out += (outer_idx[j] - 1) * strides_out[dim_j]
            end

            # extract slice
            fill!(coef_buf, zero(T))
            for k in 1:n
                coef_buf[k + eqs_d] = current[offset_in + (k-1)*strides_in[d] + 1]
            end

            # apply BCs in-place
            apply_boundary_conditions_for_dim!(coef_buf, vs_1d, 1, (h_d,), eqs_d,
                                               ((:detect,:detect),), :gauss,
                                               workspace, (n,), Val(true))

            # evaluate — kernel weights precomputed, just multiply-accumulate
            @inbounds for k in 1:n
                result = zero(T)
                i_base = k + eqs_d
                for l in l_range
                    result += coef_buf[i_base + l] * kw[l + eqs_d + 1]
                end
                out_buf[k] = result
            end

            # write back
            for k in 1:n
                next[offset_out + (k-1)*strides_out[d] + 1] = out_buf[k]
            end
        end

        current_ref[] = next
    end
    return current_ref[]
end

# 1D convenience wrapper
function convolution_smooth(knots::AbstractVector, values::AbstractArray{T,1}, B::Float64) where {T}
    return convolution_smooth((knots,), values, B)
end

"""
    convolution_gaussian(knots, values, B)
    convolution_gaussian(knots, values, B; extrap)

Construct a Gaussian smoothing interpolant from `values` on the uniform grid `knots`.
Unlike `convolution_smooth` which returns an array, this returns an interpolant that
can be evaluated at arbitrary locations.

Unlike polynomial convolution kernels which interpolate exactly through data points,
the Gaussian kernel smooths: it produces a weighted average of nearby values controlled
by `B`. Useful for noisy data where exact interpolation is undesirable.

# Arguments
- `knots`: grid — an `AbstractVector` (1D) or `NTuple{N,AbstractVector}` (ND)
- `values`: array of values, must have `size(values, d) == length(knots[d])`
- `B`: Gaussian width parameter. Larger `B` means narrower kernel (less smoothing),
  smaller `B` means wider kernel (more smoothing). **B ≤ 0.1 recommended.**

# Keyword Arguments
- `extrap`: extrapolation behavior, default `Throw()`. See [`convolution_interpolation`](@ref).

# Returns
A `ConvolutionExtrapolation` interpolant supporting evaluation at arbitrary locations.

# Examples
```julia
x = range(0.0, 2π, length=100)
y_noisy = sin.(x) .+ 0.1.*randn(100)

itp = convolution_gaussian(x, y_noisy, 0.05)
itp(1.5)         # evaluate at arbitrary point
itp.(x)          # evaluate on original grid — smoothed values

# 2D
xs = range(0.0, 2π, length=50)
ys = range(0.0, 2π, length=50)
z_noisy = [sin(x)*cos(y) for x in xs, y in ys] .+ 0.1.*randn(50,50)
itp2 = convolution_gaussian((xs,ys), z_noisy, 0.05)
itp2(1.0, 2.0)   # evaluate at arbitrary 2D point
```

See also: [`convolution_smooth`](@ref), [`convolution_interpolation`](@ref)
"""
# default 1D call, no extrapolation
function convolution_gaussian(knots::AbstractVector, values::AbstractArray{T,1},
                                B::Float64) where {T}
    itp = convolution_gaussian_itp((knots,), values, B)
    return ConvolutionExtrapolation(itp, Throw())
end

# extrapolation keyword option
function convolution_gaussian(knots::NTuple{N,AbstractVector}, values::AbstractArray{T,N},
                                B::Float64; extrap::EXT=Throw()) where {T,N,EXT<:Union{Symbol,AbstractExtrapolation}}

    itp = convolution_gaussian_itp(knots, values, B)
    return ConvolutionExtrapolation(itp, _extrap_type(extrap))
end

# internal function
function convolution_gaussian_itp(knots::NTuple{N,AbstractVector}, values::AbstractArray{T,N},
                                B::Float64) where {T,N}
        uniform_dims = ntuple(d -> is_uniform_grid(knots[d]), N)
    if !all(uniform_dims)
        error("Gaussian kernel not supported for non-uniform grids.")
    end
    eqs_for_gaussian = ceil(Int, sqrt(-log(1e-12/2) / B))
    if eqs_for_gaussian > 50
        @warn "Gaussian kernel stencil-width with B=$B yielded $(2*eqs_for_gaussian) points. Consider using a higher B."
    end

    h = ntuple(d -> knots[d][2] - knots[d][1], N)
    domain_size = ntuple(d -> size(values, d), N)
    eqs = ntuple(_ -> eqs_for_gaussian, N)
    bc = ntuple(_ -> (:linear, :linear), N)
    DV = ntuple(_ -> 0, N)
    KS = ntuple(_ -> :gauss, N)
    coefs, knots_new = _build_eager_coefs(knots, values, eqs, bc, KS, h, Val(true))
    kernel_type = ntuple(_ -> GaussianConvolutionKernel(Val(B), Val(eqs_for_gaussian)), N)
    dimension = N <= 3 ? Val(N) : HigherDimension(Val(N))
    do_type = _build_do_type(Val{DV}())
    n_integral = _count_integrals(Val{DV}())
    kernel_d1_pre, kernel_d2_pre, subgrid = (nothing, nothing, (:not_used))
    nb_wc = nothing
    anchor = ntuple(d -> zero(T), N)
    integral_dimension = n_integral <= 3 ? Val(n_integral) : HigherDimension(Val(n_integral))

    return ConvolutionInterpolation{T,N,n_integral,typeof(coefs),typeof(knots_new),typeof(kernel_type),
                            typeof(dimension),typeof(Val{KS}()),typeof(eqs),typeof(bc),
                            typeof(do_type),typeof(kernel_d1_pre),typeof(kernel_d2_pre),
                            typeof(Val(subgrid)),typeof(nb_wc),typeof(Val{false}()),
                            typeof(integral_dimension),typeof(domain_size)}(
        coefs, domain_size, knots_new, h, kernel_type, dimension, Val{KS}(), eqs, bc, do_type,
        kernel_d1_pre, kernel_d2_pre, Val(subgrid), nb_wc, Val{false}(), false,
        anchor, integral_dimension
    )
end