"""
    _fit_axis_adaptive(coords, g, lo, hi; alpha = 1.0, ratio = 2.0)

Knot vector of `g` knots on `[lo, hi]` that follows the density of the 1D sample `coords`.

1. Quantile knots: the sorted sample at `g` evenly spaced quantiles, with the endpoints
   fixed to `lo` and `hi`.
2. Blend: `(1 - alpha) * uniform + alpha * quantile`.
3. Ratio cap: adjacent spacings are forced to satisfy `1/ratio <= h[k+1]/h[k] <= ratio` by
   clamping the log-spacings in a forward and then a backward sweep (the backward sweep
   alone guarantees every pair, the forward one removes the directional bias), after which
   the spacings are rescaled so the knots span exactly `[lo, hi]`.

Duplicated coordinates can make quantile spacings zero; those are lifted to a small
fraction of the mean spacing before the cap is applied, and the cap then spreads the
neighbouring spacings smoothly.
"""
function _fit_axis_adaptive(coords::AbstractVector{T}, g::Int, lo::T, hi::T;
                            alpha::Real = 1.0, ratio::Real = 2.0) where {T<:Real}
    g >= 2 || throw(ArgumentError("an adaptive axis needs at least 2 knots, got $g."))
    0 <= alpha <= 1 || throw(ArgumentError("alpha must be in [0, 1], got $alpha."))
    ratio >= 1 || throw(ArgumentError("ratio must be >= 1, got $ratio."))
    hi > lo || throw(ArgumentError("axis has zero extent: lo = $lo, hi = $hi."))

    c  = sort(T.(coords))
    Np = length(c)
    a  = T(alpha)

    # ---- 1. quantile knots, 2. blend with the uniform knots ---------------------------
    knots = Vector{T}(undef, g)
    for k in 1:g
        p   = T(k - 1) / T(g - 1)                                  # quantile level in [0, 1]
        pos = one(T) + p * T(Np - 1)                               # fractional index into c
        i   = clamp(floor(Int, pos), 1, Np - 1)
        w   = pos - T(i)
        quantile_knot = (one(T) - w) * c[i] + w * c[i + 1]        # linear interpolation of the sorted sample
        uniform_knot  = lo + p * (hi - lo)
        knots[k] = (one(T) - a) * uniform_knot + a * quantile_knot
    end
    knots[1]   = lo                                                # endpoints are the bounding box
    knots[end] = hi

    # ---- 3. ratio cap on adjacent spacings, enforced by raising spacings only ---------
    h     = diff(knots)
    hmean = (hi - lo) / T(g - 1)
    hmin  = hmean * T(1e-3)                                        # floor for zero spacings (duplicated coordinates)
    for k in 1:(g - 1)
        h[k] = max(h[k], hmin)
    end
    r = T(ratio)
    for k in 2:(g - 1)                                             # forward: no spacing below 1/r of the previous
        h[k] = max(h[k], h[k - 1] / r)
    end
    for k in (g - 2):-1:1                                          # backward: no spacing below 1/r of the next
        h[k] = max(h[k], h[k + 1] / r)
    end
    h .*= (hi - lo) / sum(h)                                       # rescale to span [lo, hi]
    
    knots[1] = lo
    for k in 2:g
        knots[k] = knots[k - 1] + h[k - 1]
    end
    knots[end] = hi                                                # exact, not the rounded cumulative sum
    return knots
end

"""
    _fit_axes(pts, n_int, lo, hi, grid; alpha = 1.0, ratio = 2.0)

The knot axes of the fit as an `NTuple{D}`: ranges for `grid = :uniform`, adaptive knot
vectors for `grid = :adaptive` (from the marginal distribution of each coordinate of `pts`,
`D x Np`), or the user's own knot vectors for `grid = (xs, ys, …)`. `n_int` is the number of
knots per axis for the two symbolic choices and is ignored for user axes. User axes must be
strictly increasing and cover the bounding box `[lo, hi]`, since the interpolant is only
defined on its grid.
"""
function _fit_axes(pts::AbstractMatrix{T}, n_int::NTuple{D,Int}, lo::NTuple{D,T},
                   hi::NTuple{D,T}, grid; alpha::Real = 1.0, ratio::Real = 2.0) where {T<:Real,D}
    if grid === :uniform
        return ntuple(d -> range(lo[d], hi[d], length = n_int[d]), D)
    elseif grid === :adaptive
        return ntuple(d -> _fit_axis_adaptive(view(pts, d, :), n_int[d], lo[d], hi[d];
                                              alpha, ratio), D)
    elseif grid isa Tuple
        length(grid) == D ||
            throw(ArgumentError("grid must have $D knot vectors, got $(length(grid))."))
        return ntuple(D) do d
            ax = grid[d]
            ax isa AbstractVector{<:Real} ||
                throw(ArgumentError("grid[$d] must be a vector of knots."))
            length(ax) >= 2 || throw(ArgumentError("grid[$d] needs at least 2 knots."))
            issorted(ax) && allunique(ax) ||
                throw(ArgumentError("grid[$d] must be strictly increasing."))
            first(ax) <= lo[d] && last(ax) >= hi[d] ||
                throw(ArgumentError("grid[$d] spans [$(first(ax)), $(last(ax))] but the data " *
                                    "span [$(lo[d]), $(hi[d])]; the knots must cover the data."))
            T.(ax)
        end
    else
        throw(ArgumentError("grid must be :uniform, :adaptive or a tuple of knot vectors, got $grid."))
    end
end

# number of fine-grid points per axis on which the curvature monitor is sampled
_signal_fine(D::Int) = D == 1 ? 4001 : D == 2 ? 401 : D == 3 ? 101 : 31
const _SIGNAL_FLOOR  = 0.003                               # monitor floor, relative to the largest curvature
const _SIGNAL_BUDGET = 0.5                                 # adaptive knots per axis, as a fraction of the uniform fit's
const _SIGNAL_PASSES = 2                                   # refits on curvature-placed knots

"""
    _signal_axes(s, g, lo, hi; alpha, ratio)

Knot axes that follow the signal: on each axis, the knots equidistribute `√(|∂²s/∂x_d²| + ε)`,
the curvature of the fitted interpolant `s` averaged over the other axes, with the floor `ε`
a fraction `_SIGNAL_FLOOR` of the largest curvature so that flat regions keep knots. The
curvature is sampled on a fine tensor grid by second differences; a sample with the
monitor's density is drawn by inverting its cumulative integral, and `_fit_axis_adaptive`
places `g[d]` knots at that sample's quantiles, blended with uniform knots by `alpha` and with
the spacing ratio capped at `ratio`.
"""
function _signal_axes(s, g::NTuple{D,Int}, lo::NTuple{D,T}, hi::NTuple{D,T};
                      alpha::Real = 1.0, ratio::Real = 2.0) where {D,T<:Real}
    nf   = _signal_fine(D)
    fine = ntuple(d -> range(lo[d], hi[d], length = nf), D)
    V    = [s(c...) for c in Iterators.product(fine...)]      # the fit on the fine tensor grid
    return ntuple(D) do d
        h   = step(fine[d])
        S2  = abs.(diff(diff(V; dims = d); dims = d)) ./ h^2  # |∂²s/∂x_d²| by second differences
        m   = vec(maximum(S2; dims = Tuple(k for k in 1:D if k != d)))   # the largest over the other axes: an isolated feature needs its knot lines too
        m   = vcat(m[1], m, m[end])                           # padded to the fine axis
        m   = sqrt.(m .+ _SIGNAL_FLOOR * maximum(m))          # the monitor, floored
        C   = cumsum(m) .- m[1]                               # its cumulative integral, normalised
        C ./= C[end]
        coords = [fine[d][searchsortedfirst(C, p)] for p in range(0, 1, length = nf)]   # a sample with density m
        _fit_axis_adaptive(coords, g[d], lo[d], hi[d]; alpha, ratio)
    end
end