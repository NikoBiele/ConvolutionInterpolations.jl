# =============================================================================
# refinement of a nonuniform interpolant: new knots, and where to put them
# =============================================================================
#
# The weights of a nonuniform b kernel are built per interval from the 2M knots around it (M
# ghosts per side: 5 for :b5 up to 9 for :b13), each by a projection in extended precision,
# which is what makes these interpolants slow to construct. A new knot therefore changes the
# weights of the intervals whose 2M knots include it and of no others. `convolution_refine`
# adds knots to an interpolant and recomputes only those intervals; `convolution_suggest`
# estimates the error of an interpolant and returns where knots are needed.

"""Kernel symbols of a nonuniform interpolant, one per axis; an error for a uniform one."""
_refine_kernels(::NonUniformMixedOrderKernel{K}) where {K} = K
function _refine_kernels(::Any)
    throw(ArgumentError("this interpolant is on a uniform grid; only an interpolant on a " *
                        "nonuniform grid can be refined."))
end

"""Derivative orders of an interpolant, one per axis."""
_refine_derivatives(::DerivativeOrder{DO}) where {DO} = DO

"""Interior knots and values of a 1D nonuniform interpolant, without its ghosts."""
function _refine_interior(itp::ConvolutionInterpolation{T,1}) where {T}
    n = itp.domain_size[1]
    knot_ghosts = div(length(itp.knots[1]) - n, 2)
    coef_ghosts = div(length(itp.coefs) - n, 2)
    knots = collect(T, view(itp.knots[1], knot_ghosts+1:knot_ghosts+n))
    values = collect(T, view(itp.coefs, coef_ghosts+1:coef_ghosts+n))
    return knots, values
end

"""Weight tables of a nonuniform b kernel on `knots`, reusing those of `old_weights` (the
tables on `old_expanded`, the old knots with their ghosts): an interval keeps its table when
the 2M knots around it are the old ones, and gets a new one otherwise. Returns the tables,
the expanded knots and the number of intervals computed."""
function _refine_b_weights(knots::Vector{T}, kernel::Symbol, derivative::Int,
                           old_weights::Vector{Matrix{T}},
                           old_expanded::AbstractVector{T}) where {T}
    M, p_deg = nonuniform_b_params(kernel)
    n = length(knots)
    n_old = length(old_expanded) - 2 * M

    knots_r = Rational{BigInt}.(knots)    # the expanded knots, as the full build makes them
    for _ in 1:M
        pushfirst!(knots_r, knots_r[1] - (knots_r[2] - knots_r[1]))
        push!(knots_r, knots_r[end] + (knots_r[end] - knots_r[end-1]))
    end
    expanded = T.(knots_r)

    # interval k lies between knots k and k + 1 and is built from the expanded knots k+1 :
    # k+2M; `source[k]` is the old interval with the same ones, or zero
    source = zeros(Int, n - 1)
    j = 1
    for k in 1:n-1
        while j < n_old && old_expanded[M + j] < knots[k]
            j += 1
        end
        if j < n_old && old_expanded[M + j] == knots[k] &&
           view(expanded, k+1:k+2*M) == view(old_expanded, j+1:j+2*M)
            source[k] = j
        end
    end

    weights = Vector{Matrix{T}}(undef, n - 1)
    fresh = Int[]
    for k in 1:n-1
        if source[k] > 0
            weights[k] = old_weights[source[k]]
        else
            push!(fresh, k)
        end
    end
    kernel_coefs = _get_kernel_rational_coefs(kernel, derivative)
    Threads.@threads for i in eachindex(fresh)
        k = fresh[i]
        weights[k] = precompute_nonuniform_b_interval(T, knots_r, k + M, kernel_coefs, M,
                                                      p_deg, derivative)
    end
    return weights, expanded, length(fresh)
end

"""
    convolution_refine(itp, x_new, values_new)

Add knots to a one-dimensional interpolant on a nonuniform grid and return the new
interpolant.

`x_new` holds the positions of the new knots and `values_new` the data values there (two
numbers for a single knot). The result is the interpolant that `convolution_interpolation`
builds on the merged knots and values, with the same kernel, derivative order and
extrapolation, but only the weights near the new knots are computed: with a `:b` kernel an
interval keeps its weights unless one of the knots it is built from is new. Those are the M
knots on either side of it, with M = 5 for `:b5` up to 9 for `:b13`, so one new knot costs
2M intervals however many knots the grid has. The other kernels of nonuniform grids (`:n3`,
`:a0`, `:a1`) have closed-form weights and are simply rebuilt.

New knots may lie anywhere, also outside the old grid, but not on an existing knot. The
interpolant passed in is left unchanged.

Use [`convolution_suggest`](@ref) to find where an interpolant needs knots.

```julia
# nonuniform knots
x = [0.0, 0.1, 0.25, 0.3, 0.5, 0.55, 0.7, 0.9, 1.0]
# interpolant on them
itp = convolution_interpolation(x, sin.(x); kernel = :b5)
# where knots are to be added
x_new = [0.4, 0.8]
# the interpolant on all eleven knots
itp2 = convolution_refine(itp, x_new, sin.(x_new))
```

See also: [`convolution_interpolation`](@ref), [`convolution_suggest`](@ref).
"""
function convolution_refine(itp::ConvolutionInterpolation{T,1},
                            x_new::AbstractVector{<:Real},
                            values_new::AbstractVector{<:Real}) where {T}
    kernel = _refine_kernels(itp.kernel_sym)
    derivative = _refine_derivatives(itp.derivative_order)
    length(x_new) == length(values_new) ||
        throw(ArgumentError("x_new has $(length(x_new)) knots and values_new " *
                            "$(length(values_new)) values."))
    all(isfinite, x_new) && all(isfinite, values_new) ||
        throw(ArgumentError("the new knots and values must be finite."))

    old_knots, old_values = _refine_interior(itp)
    order = sortperm(vcat(old_knots, T.(x_new)))
    knots = vcat(old_knots, T.(x_new))[order]
    values = vcat(old_values, T.(values_new))[order]
    for k in 1:length(knots)-1
        knots[k] < knots[k+1] ||
            throw(ArgumentError("the new knot $(knots[k]) is already a knot of the " *
                                "interpolant, or is given twice."))
    end

    if kernel[1] in (:b5, :b7, :b9, :b11, :b13)
        weights, expanded, _ = _refine_b_weights(knots, kernel[1], derivative[1],
                                                 itp.nonuniform[1], itp.knots[1])
        return _build_nonuniform_convolution((knots,), values, itp.bc, (false,),
                                             Val(kernel), Val(derivative);
                                             weights = ((weights, expanded),))
    else
        return _build_nonuniform_convolution((knots,), values, itp.bc, (false,),
                                             Val(kernel), Val(derivative))
    end
end

function convolution_refine(itp::ConvolutionInterpolation{T,1}, x_new::Real,
                            value_new::Real) where {T}
    return convolution_refine(itp, [x_new], [value_new])
end

function convolution_refine(itp::ConvolutionInterpolation, x_new, values_new)
    throw(ArgumentError("convolution_refine takes a one-dimensional interpolant and the " *
                        "new knots and their values as two numbers or two vectors of " *
                        "numbers; got a $(ndims(itp.coefs))D interpolant, " *
                        "$(typeof(x_new)) and $(typeof(values_new))."))
end

function convolution_refine(etp::ConvolutionExtrapolation, x_new, values_new)
    return ConvolutionExtrapolation(convolution_refine(etp.itp, x_new, values_new), etp.et)
end

"""Divided difference of `values` over all of `knots`, computed in place in `work`."""
function _suggest_divided_difference!(work::Vector{T}, knots::AbstractVector{T},
                                      values::AbstractVector{T}) where {T}
    m = length(knots)
    copyto!(work, values)
    for order in 1:m-1
        for j in m:-1:order+1
            work[j] = (work[j] - work[j-1]) / (knots[j] - knots[j-order])
        end
    end
    return work[m]
end

"""
    convolution_suggest(itp; tol = 1e-6, samples = 8)

Where a one-dimensional interpolant on a nonuniform grid needs knots to be accurate: the
midpoints of the intervals in which its estimated error exceeds `tol`.

A `b` kernel of degree `p` reproduces polynomials up to that degree on a nonuniform grid, so
the leading part of its error comes from the term of degree `p + 1` in the data. The
estimate is the product of two factors. The first is the size of that term: the divided
difference of order `p + 1` of the data over the `p + 2` knots nearest the interval, taken
as the largest of the interval and its two neighbours. The second is the error the kernel
itself makes on the monomial of degree `p + 1` on these knots, measured by interpolating it.
This is the test that `bc = :detect` applies at the boundary of a uniform grid, carried to
every interval and scaled to an error in the units of the data.

The estimate is also large where a wide kernel oscillates on a grid that is too coarse for
it, most often near the ends of the grid, so the same knots cure that.

Evaluate the data at the returned points and pass both to [`convolution_refine`](@ref).
Repeat until nothing is returned, with a limit on the number of passes: where the data have
a kink or a jump the estimate does not fall and the interval is returned every time.

The estimate uses only the knots and their values. A feature of the data that lies entirely
between two knots cannot be seen, and at rounding level the differences are noise, so a
tolerance near the precision of the data flags intervals without end.

# Keyword Arguments
- `tol::Real=1e-6`: the estimated error that calls for a knot, as a fraction of the range of
  all the data.
- `samples::Int=8`: the number of points inside each interval at which the error on the
  monomial is measured.

```julia
# the data
f(x) = 1 / (1 + 25x^2)
# nonuniform knots
x = [-1.0, -0.8, -0.55, -0.3, -0.1, 0.0, 0.15, 0.3, 0.5, 0.75, 0.9, 1.0]
# a coarse interpolant
itp = convolution_interpolation(x, f.(x); kernel = :b5)
# refine, with a limit on the passes
for pass in 1:20
    # midpoints of the intervals that need a knot
    x_new = convolution_suggest(itp; tol = 1e-6)
    # the estimate is below tol everywhere
    isempty(x_new) && break
    # the interpolant with knots added there
    itp = convolution_refine(itp, x_new, f.(x_new))
end
```

See also: [`convolution_refine`](@ref).
"""
function convolution_suggest(itp::ConvolutionInterpolation{T,1}; tol::Real = 1e-6,
                             samples::Int = 8) where {T}
    kernel = _refine_kernels(itp.kernel_sym)        # a nonuniform interpolant, or an error
    kernel[1] in (:b5, :b7, :b9, :b11, :b13) ||
        throw(ArgumentError("convolution_suggest estimates the error of the b kernels; " *
                            "got :$(kernel[1])."))
    all(==(0), _refine_derivatives(itp.derivative_order)) ||
        throw(ArgumentError("convolution_suggest estimates the error of the interpolant " *
                            "itself; it needs derivative = 0."))
    tol > 0 || throw(ArgumentError("tol must be positive, got $tol."))
    samples >= 1 || throw(ArgumentError("samples must be at least 1, got $samples."))

    knots, values = _refine_interior(itp)
    n = length(knots)
    _, degree = nonuniform_b_params(kernel[1])
    width = degree + 2                         # knots in a difference of order degree + 1
    n >= width ||
        throw(ArgumentError("convolution_suggest needs at least $width knots for " *
                            ":$(kernel[1]), got $n."))
    lowest, highest = extrema(values)
    scale = highest - lowest
    if scale == 0                              # constant data: measure against their size
        scale = max(abs(highest), one(T))
    end

    # the knots mapped to [-1, 1], which keeps the monomial and the differences well scaled
    center = (knots[1] + knots[n]) / 2
    half = (knots[n] - knots[1]) / 2
    scaled = (knots .- center) ./ half

    # the kernel's own error on the first monomial it does not reproduce, on these knots;
    # the interpolant of the monomial takes the weights of `itp`, so nothing is projected
    # again
    monomial = scaled .^ (degree + 1)
    probe = _build_nonuniform_convolution((knots,), monomial, itp.bc, (false,),
                                          Val(kernel), Val((0,));
                                          weights = ((itp.nonuniform[1], itp.knots[1]),))
    kernel_error = zeros(T, n - 1)
    for k in 1:n-1
        for s in 1:samples
            x = knots[k] + (knots[k+1] - knots[k]) * T(s) / T(samples + 1)
            exact = ((x - center) / half)^(degree + 1)
            kernel_error[k] = max(kernel_error[k], abs(probe(x) - exact))
        end
    end

    # the divided difference of order degree + 1 over the knots nearest each interval
    difference = zeros(T, n - 1)
    work = Vector{T}(undef, width)
    for k in 1:n-1
        first_knot = clamp(k - div(width, 2) + 1, 1, n - width + 1)
        stencil = first_knot:first_knot+width-1
        difference[k] = abs(_suggest_divided_difference!(work, view(scaled, stencil),
                                                         view(values, stencil)))
    end

    midpoints = T[]
    for k in 1:n-1
        largest = maximum(view(difference, max(1, k - 1):min(n - 1, k + 1)))
        if largest * kernel_error[k] > T(tol) * scale
            push!(midpoints, (knots[k] + knots[k+1]) / 2)
        end
    end
    return midpoints
end

function convolution_suggest(itp::ConvolutionInterpolation; kwargs...)
    throw(ArgumentError("convolution_suggest checks one-dimensional interpolants; got " *
                        "$(ndims(itp.coefs))D."))
end

convolution_suggest(etp::ConvolutionExtrapolation; kwargs...) =
    convolution_suggest(etp.itp; kwargs...)
