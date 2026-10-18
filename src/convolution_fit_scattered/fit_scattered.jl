"""
    fit_scattered(points, values; kwargs...)

Fit scattered data with a convolution interpolant on a grid over the data's bounding box.

The tensor-product kernel basis lives on a grid covering the data's bounding
box: uniform by default, or, with `grid = :adaptive`, with per-axis knots that
follow the density of the points. Ghost coefficients are eliminated before the
solve using the package's polynomial boundary condition (the Vandermonde ghost
extrapolation of the nonuniform kernels on nonuniform grids), so the solution
is a completed grid consistent with `bc = :poly`. Because the kernels are
cardinal on the grid, the fitted interior coefficients are the interpolant's
values at the interior knots: `fit_scattered` returns exactly the
`(knots, values)` pair the gridded constructors consume.

Two modes:
- `:smooth` (default) smooths: it minimizes `(1/n)·‖A c − f‖² + lambda·Σ_d ∫|∂_d^q s|²`,
  the mean squared misfit plus `lambda` times the roughness (in coordinates
  normalized to the bounding box, so `lambda` does not depend on the grid or the
  units of each axis). By default `lambda` is chosen by generalized
  cross-validation, so no smoothing parameter needs to be tuned. For measured
  data, which usually carry some noise; on noise-free data it approaches
  interpolation, less accurately than `:exact`. Needs more data points than the
  penalty's null space has dimensions: `q^D`, i.e. 125 in 3D with the default
  `q = 5`.
- `:exact` interpolates: the fit passes through every data point to machine
  precision, and among the admissible grids the one minimizing a q-th order
  roughness seminorm is selected. For trusted, noise-free data only: noise is
  reproduced exactly and amplified between the points, most of all in
  derivatives.

# Arguments
- `points::AbstractMatrix`: `D x Np` matrix; each column is one data point.
- `values::AbstractVector`: `Np` data values.

# Keyword Arguments
- `kernel::Union{Symbol,NTuple{D,Symbol}}=:auto`: on a uniform grid any kernel
  with a polynomial boundary-condition table (`:a3`, `:a4`, `:a5`, `:a7`,
  `:b5`...`:b13`); on a nonuniform grid (`grid = :adaptive` or user axes) `:n3`
  or a `:b` kernel, the kernels that support nonuniform grids. `:auto` uses the
  scattered-path default. Uniform grid, `:smooth`: `:a4` in every dimension.
  With noisy data the accuracy is set by the noise and the roughness order
  rather than the kernel, and `:a4`'s narrow stencil makes the fit 2-3x faster
  than `:b5` in 2D and 3-9x in 3D at comparable accuracy; at a knot budget too
  small to resolve the signal, `:b5` is the more accurate choice. Uniform grid,
  `:exact`: `:b5` for 1-3D (measured to be both the fastest and the
  deepest-converging choice with the default solver), `:a4` for 4-5D, `:a3`
  above. Note this differs from the gridded default (`:b7` for 1-2D): for
  scattered data the boundary-condition tables cap all b-kernels at degree-7
  reproduction, and `:b5`'s shallower ghost extrapolation gives it the lowest
  roundoff floor under `:cholesky`. Nonuniform grid: `:n3` for `:smooth` (the
  counterpart of `:a4`: closed-form weights, a 4-wide stencil and no
  per-interval precompute) and `:b5` for `:exact`. Interpolate the fitted grid
  with the fitting kernel.
- `mode::Symbol=:smooth`: `:smooth` or `:exact` (see above).
- `grid::Union{Symbol,Tuple}=:uniform`: the knot axes. `:uniform` is the uniform
  grid over the bounding box. `:adaptive` places the knots of each axis at the
  quantiles of the points' coordinates along that axis, blended with the uniform
  knots by `alpha` and with the ratio of adjacent spacings capped at `ratio`, so
  a dense band of points gets a dense band of knots. This is what lets `:exact`
  interpolate clustered points at all, and lets `:smooth` resolve a boundary
  layer, a step or a log-scaled axis at a knot count a uniform grid cannot. The
  grid is still a tensor product, so it helps when the point density is roughly
  separable along the axes, not for isolated clusters; in two or more dimensions
  points can still share a cell where the clusters of several axes overlap, and
  `oversample` is the remedy. A tuple of strictly increasing knot vectors, one
  per axis, is used as given; the vectors must cover the data. Nonuniform grids
  give a nonuniform interpolant: no antiderivatives (`derivative = -1`) and
  `lazy` only with `:n3`. With the `:b` kernels construction costs one
  exact-arithmetic projection per interval, so keep their knot counts modest;
  `:n3` has no such cost.
- `alpha::Real=1.0`, `ratio::Real=2.0` (`grid = :adaptive` only): the blend
  between uniform (`0`) and quantile (`1`) knots, and the largest allowed ratio
  of adjacent knot spacings. The cap keeps the nonuniform kernels and their ghost
  extrapolation well conditioned; raise it for strongly graded data.
- `gridsize::Union{Symbol,Int,NTuple{D,Int}}=:auto`: interior knots per
  dimension. An `Int` or tuple is used as given in both modes. With `:auto`,
  both modes size the grid from `oversample` and the data count. `:smooth`
  uses that grid directly rather than a coarser one: a grid that cannot
  represent part of the signal absorbs it into the residual, where nothing in
  the fit reveals it, while the smoothing itself is set by `lambda`, not by
  the knot count. With user axes the knot counts are those of the axes, and
  `gridsize` must be `:auto`.
- `oversample::Real=0.0`: grid knots per data spacing when `gridsize=:auto`.
  `0.0` selects the mode default: 0.8 for `:smooth`; for `:exact` 1.6 in 1D,
  2.5 in 2D and 2.0 in 3D and above, since random points in two or more
  dimensions share a grid cell often enough at the 1D density that the fit
  goes least-squares. Points closer together than the local knot spacing
  cannot be interpolated separately on any grid; if `:exact` warns, raise
  `oversample`. `:smooth` warns when the fitted smoothing would need a finer
  grid than this allows (about eight knots per cycle of its cutoff): the fit
  may then be under-resolved, and a larger `oversample` helps.
- `q::Union{Int,Symbol}=:auto`: roughness (difference) order of the seminorm.
  `:auto` uses, for `:smooth`, 5 for every kernel (higher orders make the
  smoothing system ill-conditioned, lower orders are less accurate); for
  `:exact`, the ghost extrapolation's stencil width minus one (5 for `:b5`, 7
  for `:b7`-class tables), which preserves the kernel's convergence order in
  data gaps. On nonuniform grids the roughness is discretised by divided
  differences scaled to plain differences on the local spacing, with the same
  defaults except that `:n3`'s `:exact` default is 2, matching its three-knot
  ghost extrapolation. For data that oscillate everywhere (a chirp) no `q`
  smooths without biasing the signal, and GCV then leaves `lambda` near zero:
  the knot count is the regulariser in that case.
- `lambda::Union{Symbol,Real}=:gcv` (`:smooth` only): the smoothing parameter,
  or `:gcv` to choose it by generalized cross-validation. The `lambda` returned
  by a fit reproduces that fit when passed back.
- `solver::Symbol=:cholesky` (`:exact` only): `:cholesky` (default) factors the
  Jacobi-equilibrated SPD normal system with CHOLMOD: fast, and with the
  default `:b5` kernel it is truncation-limited (no visible roundoff floor
  down to ~3e-11 in 2D benchmarks). `:qr` solves each multiplier iteration
  as a sparse least-squares problem via SPQR without forming normal
  equations; it is several times slower but lowers the roundoff floor of the
  wide kernels (measured 2D floors: :b7 ~5e-12 and still converging,
  :b9 ~1e-11, :b11 ~6e-10, :b13 ~3e-9). Use `solver=:qr, kernel=:b7` when
  chasing the deepest accuracy. The `:nullspace` solver solves for the
  coefficients exactly, but is only feasible on small meshes (few points).
- `W::Real=1e6` (`:exact` only): augmented-Lagrangian penalty weight.
- `iters::Int=20`, `tol::Real=1e-11` (`:exact` only): multiplier iteration
  controls; convergence typically takes 1-2 iterations.
- `verbose::Bool=false`: `:exact` prints the constraint violation per
  iteration; `:smooth` prints the grid, `lambda`, cutoff, effective degrees of
  freedom and estimated noise of the fit.

# Returns
`(knots, vals, third)`: the tuple of interior knot axes (ranges for a uniform
grid, vectors otherwise), the array of fitted interior grid values, and
- for `:smooth`, `(lambda = λ, noise = σ)`: the smoothing parameter used and
  the estimated standard deviation of the data about the fit,
  `√(‖A c − f‖² / (n − edf))` with `edf` the fit's effective degrees of freedom;
- for `:exact`, `viol`: the final maximum constraint violation.

Construct an interpolant with `convolution_interpolation(knots, vals; kernel, bc=:poly, ...)`,
or use `convolution_interpolation(points, values)` which does this directly.

Notes: `:exact` solves a sparse SPD system (CHOLMOD for `Float64`, dense
Cholesky otherwise). Exact interpolation requires distinct data points with
consistent values; duplicated points with conflicting values are resolved in
the least-squares sense and reported through `viol`. `:smooth` always solves in
`Float64` (its accuracy is limited by the noise) and casts the fitted grid back
to the element type of the data. On nonuniform grids the kernel weights are
Float64, as in the nonuniform interpolant itself.
"""
function fit_scattered(points::AbstractMatrix, values::AbstractVector;
                       kernel::Union{Symbol,Tuple{Vararg{Symbol}}}=:auto,
                       mode::Symbol=:smooth,
                       grid::Union{Symbol,Tuple}=:uniform,
                       alpha::Real=1.0, ratio::Real=2.0,
                       gridsize::Union{Symbol,Int,Tuple{Vararg{Int}}}=:auto,
                       oversample::Real=0.0,
                       q::Union{Int,Symbol}=:auto,
                       lambda::Union{Symbol,Real}=:gcv,
                       solver::Symbol=:cholesky,
                       W::Real=1e6, iters::Int=20, tol::Real=1e-11,
                       verbose::Bool=false)

    solver in (:nullspace, :qr, :cholesky) ||
        throw(ArgumentError("solver must be :nullspace, :qr or :cholesky, got $solver."))
    D  = size(points, 1)
    Np = size(points, 2)
    Np == length(values) ||
        throw(ArgumentError("points has $Np columns but values has $(length(values)) entries."))
    Np >= 2 || throw(ArgumentError("need at least 2 data points."))
    mode === :lsq &&
        throw(ArgumentError("mode = :lsq has been replaced by mode = :smooth, which fits " *
                            "noisy data with a smoothing parameter chosen automatically."))
    mode in (:exact, :smooth) ||
        throw(ArgumentError("mode must be :exact or :smooth, got $mode."))
    (grid === :uniform || grid === :adaptive || grid isa Tuple) ||
        throw(ArgumentError("grid must be :uniform, :adaptive or a tuple of knot vectors, got $grid."))
    (grid isa Tuple && gridsize !== :auto) &&
        throw(ArgumentError("gridsize cannot be combined with user-supplied grid axes; " *
                            "the knot counts are those of the axes."))
    (lambda === :gcv || (lambda isa Real && lambda > 0)) ||
        throw(ArgumentError("lambda must be :gcv or a positive number, got $lambda."))
    Np >= D ||
        throw(ArgumentError("points is $(D)x$(Np): fewer points than dimensions. " *
            "Points are COLUMNS (D x Np); for 1D scattered data use " *
            "reshape(x, 1, :). For gridded/nonuniform data at known knots, " *
            "pass knots as a Vector, not a Matrix."))
    nonuniform = grid !== :uniform

    F = float(promote_type(eltype(points), eltype(values)))
    # Solve eltype: Float16/Float32 are promoted to Float64 internally: the
    # penalty-weighted system spans ~W in scale, which single precision cannot
    # factor reliably (and the result's accuracy is data-limited anyway).
    # The fitted grid is cast back to F. BigFloat and friends solve natively
    # via the dense generic path (:exact; :smooth always solves in Float64).
    Fs = F <: Union{Float16,Float32} ? Float64 : F
    pts = Fs.(points)
    f   = Fs.(values)

    # ---- kernels, ghost stencils, roughness orders ----------------------------
    kernel = kernel === :auto ? _default_scattered_kernel(D, mode, nonuniform) : kernel
    kernels = kernel isa Symbol ? ntuple(_ -> kernel, D) :
              length(kernel) == D ? NTuple{D,Symbol}(kernel) :
              throw(ArgumentError("kernel must be a Symbol or an NTuple{$D,Symbol}."))
    if nonuniform
        for k in kernels
            k in (:n3, :b5, :b7, :b9, :b11, :b13) ||
                throw(ArgumentError("nonuniform grids (grid = :adaptive or user axes) require " *
                                    ":n3 or a :b kernel (:b5 … :b13) on every axis; :$k is not supported."))
        end
        pdeg = ntuple(d -> kernels[d] === :n3 ? 2 : nonuniform_b_params(kernels[d])[2], D)   # ghost extrapolation degree
        ns   = ntuple(d -> pdeg[d] + 1, D)                           # knots per ghost
        qv   = ntuple(d -> q !== :auto ? q::Int :                    # roughness orders
                          mode === :smooth ? 5 : min(pdeg[d], 7), D)
    else
        for k in kernels
            haskey(POLYNOMIAL_GHOST_COEFFS, k) && k !== :n3 ||
                throw(ArgumentError("fit_scattered requires a kernel with a polynomial " *
                                    "boundary-condition table; :$k is not supported."))
        end
        G  = ntuple(d -> get_polynomial_ghost_coeffs(:poly, kernels[d]), D)
        ns = ntuple(d -> size(G[d], 2), D)                           # BC stencil width
        qv = ntuple(d -> q !== :auto ? q::Int :                      # roughness orders
                        mode === :smooth ? 5 : ns[d] - 1, D)
    end

    # ---- bounding box ---------------------------------------------------------
    lo = ntuple(d -> minimum(view(pts, d, :)), D)
    hi = ntuple(d -> maximum(view(pts, d, :)), D)
    for d in 1:D
        hi[d] > lo[d] ||
            throw(ArgumentError("data has zero extent in dimension $d."))
    end

    # ---- a given grid (both modes), validated against the stencil width ------
    n_given = gridsize isa Int ? ntuple(_ -> gridsize, D) :
              gridsize isa Tuple ? (length(gridsize) == D ? NTuple{D,Int}(gridsize) :
                  throw(ArgumentError("gridsize must have $D entries."))) :
              grid isa Tuple ? (length(grid) == D ? ntuple(d -> length(grid[d]), D) :
                  throw(ArgumentError("grid must have $D knot vectors, got $(length(grid))."))) :
              nothing
    if n_given !== nothing
        for d in 1:D
            n_given[d] >= ns[d] ||
                throw(ArgumentError("gridsize[$d] = $(n_given[d]) is below the boundary-" *
                                    "condition stencil width $(ns[d]) for :$(kernels[d])."))
        end
    end

    # ---- :smooth: penalised fit with GCV-chosen smoothing (fit_smooth.jl) ----
    if mode === :smooth
        n_null = prod(qv)                                  # polynomials the penalty does not see
        Np > n_null ||
            throw(ArgumentError("mode = :smooth needs more than $n_null data points in $(D)D " *
                                "with q = $(join(unique(qv), "/")) (the penalty's null space); " *
                                "got $Np. Use mode = :exact, or a lower q."))
        os    = oversample > 0 ? Float64(oversample) : 0.8
        g_cap = ntuple(d -> max(ns[d], ceil(Int, os * Np^(1 / D)) + 1), D)   # largest :auto grid
        pts64 = Float64.(pts)
        lo64  = ntuple(d -> Float64(lo[d]), D)
        hi64  = ntuple(d -> Float64(hi[d]), D)
        make_axes = n_int -> _fit_axes(pts64, n_int, lo64, hi64, grid; alpha, ratio)
        axes, c_int, info = _fit_smooth(pts64, Float64.(f), kernels, qv, make_axes,
                                        n_given, g_cap, lambda, verbose)
        n_int = ntuple(d -> length(axes[d]), D)
        knots = nonuniform ? ntuple(d -> F.(axes[d]), D) :
                             ntuple(d -> range(lo[d], hi[d], length = n_int[d]), D)
        return knots, F.(reshape(c_int, n_int)), info
    end

    # ---- :exact: grid ---------------------------------------------------------
    # default knot density per data spacing: 1.6 in 1D, where quantile-like spacing already
    # separates the points; higher in 2D and 3D, where random points share a cell often
    # enough at 1.6 that the fit goes least-squares; moderated in 3D and above, where the
    # unknowns grow as os^D and large uniform systems with few data rows lose positive
    # definiteness numerically
    os = oversample > 0 ? Fs(oversample) : Fs(D == 1 ? 1.6 : D == 2 ? 2.5 : 2.0)
    n_int = n_given !== nothing ? n_given :
            ntuple(d -> max(ns[d], ceil(Int, os * Np^(1 / D)) + 1), D)
    axes = _fit_axes(pts, n_int, lo, hi, grid; alpha, ratio)

    # ---- sparse system on the interior unknowns -----------------------------
    A_eff, R_eff = _scattered_operators(pts, kernels, axes, qv)

    c_int, viol = _scattered_auglag(A_eff, f, R_eff, Fs(W), iters, Fs(tol),
                                    verbose, Val(solver))

    if viol > max(Fs(tol), sqrt(eps(Fs))) * max(one(Fs), maximum(abs, f))
        hint = nonuniform ? "" :
               " If the points are clustered, `grid = :adaptive` places the knots where " *
               "the points are, so that each point gets its own knot spacing."
        @warn "fit_scattered (:exact): constraint violation $viol did not reach " *
              "machine precision. The system may be overdetermined for this grid " *
              "(try a larger gridsize/oversample) or the data may contain " *
              "duplicated points with conflicting values. Result is the " *
              "least-squares fit." * hint
    end

    vals_grid = F === Fs ? reshape(c_int, n_int) : F.(reshape(c_int, n_int))
    knots = F === Fs ? axes : ntuple(d -> F.(axes[d]), D)
    return knots, vals_grid, F(viol)
end

"""
    convolution_interpolation(points::AbstractMatrix, values::AbstractVector; kwargs...)

Scattered-data convolution interpolation. Each column of `points` is one data
point (`D x Np`); `values` holds the corresponding data values.

Fits the data with `fit_scattered` and wraps the fitted grid in the standard
interpolant, so evaluation, derivatives, and integrals work exactly as for
gridded data. The default `mode = :smooth` smooths the data with the smoothing
chosen automatically, which suits measured, noisy data (to see the chosen
`lambda` and the estimated noise, call `fit_scattered` directly); for trusted,
noise-free data use `mode = :exact`, whose interpolant passes through the data.
Unlike the gridded constructors, which always interpolate, the scattered path
smooths by default, because scattered data are usually measurements.
All `fit_scattered` keyword arguments are accepted, including `grid`, `alpha`
and `ratio` for adaptive or user-supplied knot axes, plus `derivative`, which
is forwarded to the interpolant, and `extrap`. The deprecated `precompute` and
`subgrid` keywords are still accepted and forwarded, but have no effect. On a
uniform grid the boundary condition is fixed to `:poly` for consistency with
the fit; on a nonuniform grid the interpolant's own ghost extrapolation is the
one the fit used. A nonuniform grid gives a nonuniform interpolant (`:n3` by
default for `:smooth`, `:b5` for `:exact`): no antiderivatives
(`derivative = -1`), and `lazy` only with `:n3`.

```julia
pts = rand(2, 500)                                        # 500 scattered points in 2D, one per column
vals = [sin(3p[1]) * cos(2p[2]) for p in eachcol(pts)]    # the signal at the points
noisy = vals .+ 0.05 .* randn(500)                        # measured with noise
itp = convolution_interpolation(pts, noisy)               # smoothing chosen automatically
itp(0.4, 0.6)                                             # close to the noise-free value

itp_x = convolution_interpolation(pts, vals; mode = :exact)   # noise-free data: interpolate exactly
itp_x(0.4, 0.6)                                           # evaluate anywhere inside the data's box

itp_a = convolution_interpolation(pts, vals; mode = :exact, grid = :adaptive)   # knots follow the point density
```
"""
function convolution_interpolation(points::AbstractMatrix, values::AbstractVector;
                                   kernel::Union{Symbol,Tuple{Vararg{Symbol}}}=:auto,
                                   mode::Symbol=:smooth,
                                   grid::Union{Symbol,Tuple}=:uniform,
                                   alpha::Real=1.0, ratio::Real=2.0,
                                   gridsize::Union{Symbol,Int,Tuple{Vararg{Int}}}=:auto,
                                   oversample::Real=0.0,
                                   q::Union{Int,Symbol}=:auto,
                                   lambda::Union{Symbol,Real}=:gcv,
                                   solver::Symbol=:cholesky,
                                   W::Real=1e6, iters::Int=20, tol::Real=1e-11,
                                   verbose::Bool=false,
                                   precompute=nothing,
                                   derivative::Union{Int,Tuple{Vararg{Int}}}=0,
                                   subgrid=nothing,
                                   extrap::Union{Symbol,AbstractExtrapolation}=Throw())
    D = size(points, 1)
    nonuniform = grid !== :uniform
    kernel_res = kernel === :auto ? _default_scattered_kernel(D, mode, nonuniform) : kernel
    knots, vals, _ = fit_scattered(points, values;
                                   kernel = kernel_res, mode, grid, alpha, ratio, gridsize,
                                   oversample, q, lambda, solver, W, iters, tol, verbose)
    if nonuniform
        # the gridded constructor detects the nonuniform knots and builds the nonuniform
        # interpolant, whose Vandermonde ghosts are exactly the ones the fit eliminated
        return convolution_interpolation(knots, vals; kernel = kernel_res, derivative, extrap)
    end
    itp = FastConvolutionInterpolation(knots, vals;
                                       kernel = kernel_res, bc = :poly,
                                       precompute, derivative, subgrid)
    return ConvolutionExtrapolation(itp, _extrap_type(extrap))
end

# =============================================================================
# internals
# =============================================================================

"""Scattered-path default kernel. Uniform grid — `:smooth`: `:a4` in every dimension
(accuracy is noise-limited, and its narrow stencil keeps the fit fast); `:exact`: `:b5`
for 1-3D (fastest and, with the default `:cholesky` solver, also the deepest-converging
in benchmarks), `:a4` for 4-5D, `:a3` above. Nonuniform grid — `:smooth`: `:n3`, the
nonuniform counterpart of `:a4`'s role (closed-form weights, no per-interval
precompute); `:exact`: `:b5`."""
_default_scattered_kernel(D::Int, mode::Symbol = :smooth, nonuniform::Bool = false) =
    nonuniform ? (mode === :smooth ? :n3 : :b5) :
        mode === :smooth ? :a4 : D <= 3 ? :b5 : D <= 5 ? :a4 : :a3

"""Collocation and roughness operators of the fit on the interior unknowns, `(A*S, R*S)`,
for the given axes. Ranges mean a uniform grid: the uniform kernels with the `:poly` ghost
tables, as the uniform path has always built them. Vectors mean a nonuniform grid: the
nonuniform kernels (`:n3` or the b-kernels) through the per-axis bases of `_nu_axis`, with
divided-difference roughness in plain-difference units; `_smooth_system` applies the same
`rowscale` to both grid types."""
function _scattered_operators(pts::AbstractMatrix{F}, kernels::NTuple{D,Symbol},
                              axes::NTuple{D,<:AbstractRange}, qv::NTuple{D,Int}) where {F,D}
    eqs = ntuple(d -> get_equations_for_degree(kernels[d]), D)
    ng  = ntuple(d -> eqs[d] - 1, D)                       # ghosts per side
    G   = ntuple(d -> get_polynomial_ghost_coeffs(:poly, kernels[d]), D)
    ns  = ntuple(d -> size(G[d], 2), D)                    # BC stencil width
    n_int   = ntuple(d -> length(axes[d]), D)
    h       = ntuple(d -> F(step(axes[d])), D)
    n_full  = ntuple(d -> n_int[d] + 2ng[d], D)
    x0_full = ntuple(d -> F(first(axes[d])) - ng[d] * h[d], D)

    A = _scattered_collocation(pts, kernels, eqs, n_full, x0_full, h)
    S = _scattered_extension(n_full, n_int, ng, ns, G, F)
    R = _scattered_roughness(n_full, qv, F)
    return A * S, R * S
end

function _scattered_operators(pts::AbstractMatrix{F}, kernels::NTuple{D,Symbol},
                              axes::NTuple{D,<:AbstractVector}, qv::NTuple{D,Int}) where {F,D}
    bases = ntuple(d -> _nu_axis(kernels[d], axes[d], F), D)   # per-axis nonuniform basis
    M     = ntuple(d -> bases[d].M, D)                         # ghosts per side

    A = _scattered_collocation(pts, axes, bases)
    S = _scattered_extension(axes, bases, F)
    R = _scattered_roughness(axes, M, qv, F)
    return A * S, R * S
end

"""Nonuniform basis of one axis for the scattered fit: everything the collocation, extension
and roughness operators need, built once from the kernel and the knots. Fields: `M` ghosts per
side; `ns`, `Gl`, `Gr` the ghost extrapolation (each ghost is a combination of the `ns`
nearest interior knots, `Gl[g, :]` for the g-th left ghost from knots `1:ns`, `Gr[g, :]` for the
g-th right ghost from knots `n-ns+1:n`), exactly as the interpolant constructor fills them;
`stencil` the number of weights per point; and `eval(x)`, returning `(col0, w)` such that point
`x` has weight `w[j]` at full-grid column `col0 + j`, `j = 1:stencil`."""
function _nu_axis(kernel::Symbol, ax::AbstractVector, ::Type{F}) where {F}
    n_int = length(ax)
    if kernel === :n3
        # nonuniform Catmull-Rom: one ghost per side from a three-knot quadratic extrapolation,
        # four closed-form weights per point from the local spacings (`_nonuniform_dim_ghost`)
        kexp = vcat(ax[1] - (ax[2] - ax[1]), ax, ax[end] + (ax[end] - ax[end - 1]))
        Gl = F.(nonuniform_ghost_coefficients(ax, 1, 2, :left))
        Gr = F.(nonuniform_ghost_coefficients(ax, 1, 2, :right))
        eval = function (x)
            i, w = _nonuniform_dim_ghost(kexp, x)          # expanded index i, weights at i-1 … i+2
            return i - 2, F[w[1], w[2], w[3], w[4]]
        end
        return (M = 1, ns = 3, Gl = Gl, Gr = Gr, stencil = 4, eval = eval)
    else
        # b-kernels: M ghosts per side from a degree-p Vandermonde extrapolation, 2M weight
        # polynomials per interval from the exact per-interval projection
        M, p = nonuniform_b_params(kernel)
        weights = precompute_nonuniform_b_all(ax, kernel, 0)[1]
        Gl = F.(nonuniform_b_ghost_coefficients(ax, M, p, :left))
        Gr = F.(nonuniform_b_ghost_coefficients(ax, M, p, :right))
        eval = function (x)
            k  = clamp(searchsortedlast(ax, x), 1, n_int - 1)    # interval in the interior axis
            s  = (x - ax[k]) / (ax[k + 1] - ax[k])
            coeffs = weights[k]
            n_poly = size(coeffs, 2)
            w = Vector{F}(undef, 2M)
            for j in 1:2M
                val = F(coeffs[j, n_poly])
                for q in (n_poly - 1):-1:1
                    val = val * s + F(coeffs[j, q])
                end
                w[j] = val
            end
            return k, w                                           # columns k+1 … k+2M, as in `_nb_dim`
        end
        return (M = M, ns = p + 1, Gl = Gl, Gr = Gr, stencil = 2M, eval = eval)
    end
end

"""Sparse collocation matrix over the full (ghost-extended) grid: row i holds
the tensor-product kernel weights of data point i at the grid knots."""
function _scattered_collocation(pts::AbstractMatrix{F}, kernels::NTuple{D,Symbol},
                                eqs::NTuple{D,Int}, n_full::NTuple{D,Int},
                                x0_full::NTuple{D,F}, h::NTuple{D,F}) where {F,D}
    Np = size(pts, 2)
    lin = LinearIndices(n_full)
    stencil_max = prod(d -> 2eqs[d], 1:D)
    I_ = Vector{Int}(undef, 0);  sizehint!(I_, Np * stencil_max)
    J_ = Vector{Int}(undef, 0);  sizehint!(J_, Np * stencil_max)
    V_ = Vector{F}(undef, 0);    sizehint!(V_, Np * stencil_max)

    kers = ntuple(d -> ConvolutionKernel(Val(kernels[d]), Val(0)), D)
    wbuf = ntuple(d -> Vector{F}(undef, 2eqs[d]), D)

    for i in 1:Np
        pos    = ntuple(d -> (pts[d, i] - x0_full[d]) / h[d] + 1, D)   # 1-based
        base   = ntuple(d -> floor(Int, pos[d]), D)
        jlo    = ntuple(d -> max(1, base[d] - eqs[d] + 1), D)
        jhi    = ntuple(d -> min(n_full[d], base[d] + eqs[d]), D)
        for d in 1:D
            for (m, j) in enumerate(jlo[d]:jhi[d])
                wbuf[d][m] = kers[d](pos[d] - j)
            end
        end
        for ci in CartesianIndices(ntuple(d -> jlo[d]:jhi[d], D))
            w = one(F)
            for d in 1:D
                w *= wbuf[d][ci[d] - jlo[d] + 1]
            end
            if w != zero(F)
                push!(I_, i); push!(J_, lin[ci]); push!(V_, w)
            end
        end
    end
    return sparse(I_, J_, V_, Np, prod(n_full))
end

"""Sparse collocation matrix over the full (ghost-extended) nonuniform grid: row i holds the
tensor-product weights of data point i at the grid knots, from the per-axis bases of
`_nu_axis` (`basis.eval(x)` gives the first column minus one and the weights of a point along
that axis)."""
function _scattered_collocation(pts::AbstractMatrix{F}, axes::NTuple{D,<:AbstractVector},
                                bases::NTuple{D,<:NamedTuple}) where {F,D}
    Np      = size(pts, 2)
    n_full  = ntuple(d -> length(axes[d]) + 2bases[d].M, D)
    stencil = ntuple(d -> bases[d].stencil, D)
    lin     = LinearIndices(n_full)
    stencil_max = prod(stencil)
    I_ = Vector{Int}(undef, 0);  sizehint!(I_, Np * stencil_max)
    J_ = Vector{Int}(undef, 0);  sizehint!(J_, Np * stencil_max)
    V_ = Vector{F}(undef, 0);    sizehint!(V_, Np * stencil_max)

    for i in 1:Np
        cw   = ntuple(d -> bases[d].eval(pts[d, i]), D)          # (col0, weights) per axis
        col0 = ntuple(d -> cw[d][1], D)
        for ci in CartesianIndices(stencil)
            w = one(F)
            for d in 1:D
                w *= cw[d][2][ci[d]]
            end
            if w != zero(F)
                push!(I_, i)
                push!(J_, lin[ntuple(d -> col0[d] + ci[d], D)...])
                push!(V_, w)
            end
        end
    end
    return sparse(I_, J_, V_, Np, prod(n_full))
end

"""Per-dimension extension operators (interior -> full grid via the polynomial
BC), combined column-major: S = kron(E_D, ..., E_1)."""
function _scattered_extension(n_full::NTuple{D,Int}, n_int::NTuple{D,Int},
                              ng::NTuple{D,Int}, ns::NTuple{D,Int},
                              G::NTuple{D,Matrix{Float64}}, ::Type{F}) where {F,D}
    Es = ntuple(D) do d
        I_ = Int[]; J_ = Int[]; V_ = F[]
        for j in 1:n_int[d]                          # interior identity
            push!(I_, ng[d] + j); push!(J_, j); push!(V_, one(F))
        end
        for g in 1:ng[d], j in 1:ns[d]               # left ghosts, nearest-first
            push!(I_, ng[d] + 1 - g); push!(J_, j); push!(V_, F(G[d][g, j]))
        end
        for g in 1:ng[d], j in 1:ns[d]               # right ghosts, mirrored
            push!(I_, n_full[d] - ng[d] + g)
            push!(J_, n_int[d] - ns[d] + j)
            push!(V_, F(G[d][g, ns[d] + 1 - j]))
        end
        sparse(I_, J_, V_, n_full[d], n_int[d])
    end
    return reduce(kron, reverse(Es))
end

"""Per-dimension extension operators on a nonuniform grid (interior -> full grid via the ghost
extrapolation of the kernel, from the per-axis bases of `_nu_axis`), combined column-major:
S = kron(E_D, ..., E_1). Ghost g (g = 1 nearest the boundary) sits at full index M+1-g on the
left and M+n+g on the right, and combines the `ns` nearest interior knots, exactly as the
interpolant constructor fills them."""
function _scattered_extension(axes::NTuple{D,<:AbstractVector}, bases::NTuple{D,<:NamedTuple},
                              ::Type{F}) where {F,D}
    Es = ntuple(D) do d
        n_int  = length(axes[d])
        M, ns  = bases[d].M, bases[d].ns
        Gl, Gr = bases[d].Gl, bases[d].Gr
        n_full = n_int + 2M
        I_ = Int[]; J_ = Int[]; V_ = F[]
        for j in 1:n_int                                         # interior identity
            push!(I_, M + j); push!(J_, j); push!(V_, one(F))
        end
        for g in 1:M, k in 1:ns                                  # left ghosts from knots 1 … ns
            push!(I_, M + 1 - g); push!(J_, k); push!(V_, F(Gl[g, k]))
        end
        for g in 1:M, k in 1:ns                                  # right ghosts from knots n_int-ns+1 … n_int
            push!(I_, M + n_int + g)
            push!(J_, n_int - ns + k)
            push!(V_, F(Gr[g, k]))
        end
        sparse(I_, J_, V_, n_full, n_int)
    end
    return reduce(kron, reverse(Es))
end

"""Roughness operator on the full grid: q-th differences along each axis,
stacked. Column-major kron ordering to match `_scattered_extension`."""
function _scattered_roughness(n_full::NTuple{D,Int}, q::NTuple{D,Int},
                              ::Type{F}) where {F,D}
    function dq(n, qq)
        I_ = Int[]; J_ = Int[]; V_ = F[]
        for i in 1:(n - qq), k in 0:qq
            push!(I_, i); push!(J_, i + k)
            push!(V_, F((-1)^(qq - k) * binomial(qq, k)))
        end
        sparse(I_, J_, V_, n - qq, n)
    end
    blocks = map(1:D) do dd
        ops = ntuple(d -> d == dd ? dq(n_full[d], q[d]) :
                                    sparse(one(F) * I, n_full[d], n_full[d]), D)
        reduce(kron, reverse(ops))
    end
    return reduce(vcat, blocks)
end

"""Roughness operator on the full nonuniform grid: q-th divided differences along each axis,
each row scaled by its support length to the q-th power, so every row is a plain q-th
difference on the local scale, O(1) on any grid and equal to the uniform operator on a uniform
grid; identity along the transverse axes; stacked over the axes. `_smooth_system` applies the
same mean-grid `rowscale` as on uniform grids, so on a nonuniform grid the `:smooth` penalty
is the sum of squared local differences in mean-grid units rather than exactly ∫|∂^q s|²: λ's
cutoff interpretation is then approximate, while GCV still chooses λ from the data. The ghost
knots (`M[d]` per side) mirror the end spacings, as the interpolant constructor places them.
Column-major kron ordering to match `_scattered_extension`."""
function _scattered_roughness(axes::NTuple{D,<:AbstractVector}, M::NTuple{D,Int},
                              q::NTuple{D,Int}, ::Type{F}) where {F,D}
    # full knot vectors, normalised to [0, 1] over the interior, with mirrored ghosts
    xs = ntuple(D) do d
        ax   = axes[d]
        lo   = F(first(ax))
        span = F(last(ax)) - lo
        x    = [(F(a) - lo) / span for a in ax]
        h1   = x[2] - x[1]
        hn   = x[end] - x[end - 1]
        vcat([x[1] - g * h1 for g in M[d]:-1:1], x, [x[end] + g * hn for g in 1:M[d]])
    end
    # q-th divided differences along one axis, each row times ℓ^q with ℓ its support length
    # over q: a plain q-th difference on the local scale, O(1) on any grid and equal to the
    # uniform operator's binomial differences on a uniform grid
    function dq(x::Vector{F}, qq::Int)
        n = length(x)
        I_ = Int[]; J_ = Int[]; V_ = F[]
        fac = F(factorial(qq))
        for i in 1:(n - qq)
            ell = (x[i + qq] - x[i]) / qq
            for k in 0:qq
                denom = one(F)
                for m in 0:qq
                    m == k && continue
                    denom *= x[i + k] - x[i + m]
                end
                push!(I_, i); push!(J_, i + k); push!(V_, ell^qq * fac / denom)
            end
        end
        return sparse(I_, J_, V_, n - qq, n)
    end

    blocks = map(1:D) do dd
        ops = ntuple(d -> d == dd ? dq(xs[d], q[d]) :
                                    sparse(one(F) * I, length(xs[d]), length(xs[d])), D)
        reduce(kron, reverse(ops))
    end
    return reduce(vcat, blocks)
end

"""Sparse least-squares factorization: SPQR for Float64, dense QR otherwise."""
_scattered_lsq_factorize(K::SparseMatrixCSC{Float64}) = qr(K)
_scattered_lsq_factorize(K::SparseMatrixCSC{F}) where {F} = qr(Matrix(K))

"""SPD factorization helper: CHOLMOD for Float64 sparse, dense Cholesky
otherwise (BigFloat and friends)."""
_scattered_spd_factorize(H::SparseMatrixCSC{Float64}) = cholesky(Symmetric(H))
function _scattered_spd_factorize(H::SparseMatrixCSC{F}) where {F}
    # generic dense path (BigFloat etc.): tiny relative shift guards against
    # roundoff-indefiniteness in the penalty-scaled system
    Hd = Matrix(H)
    shift = sqrt(eps(F)) * F(1e-3) * maximum(abs, view(Hd, diagind(Hd)))
    @inbounds for i in diagind(Hd)
        Hd[i] += shift
    end
    return cholesky!(Symmetric(Hd))
end

"""Augmented Lagrangian for  min ||R c||^2  s.t.  A c = f  (or its
least-squares limit when the constraints are infeasible). One factorization,
few multiplier iterations; early exit when the violation stops contracting.
QR-based multiplier iteration: each step solves the least-squares problem
min ||sqrt(W)(A c - g)||^2 + ||R c||^2 with g = f - lam/W, factored once as
qr([sqrt(W) A; R]). No normal equations are formed, so the roundoff floor
scales with cond(K) rather than cond(K)^2."""
function _scattered_auglag(A::SparseMatrixCSC{F}, f::Vector{F},
                           R::SparseMatrixCSC{F}, W::F, iters::Int, tol::F,
                           verbose::Bool, ::Val{:qr}) where {F}
    sqW = sqrt(W)
    fac = _scattered_lsq_factorize([sqW * A; R])
    mR  = size(R, 1)
    lam = zeros(F, size(A, 1))
    c   = zeros(F, size(A, 2))
    viol = F(Inf); prev = F(Inf)
    for it in 1:iters
        rhs = [sqW .* (f .- lam ./ W); zeros(F, mR)]
        c = fac \ rhs
        r = A * c - f
        viol = maximum(abs, r)
        verbose && println("  fit_scattered iter $it: constraint violation $viol")
        (viol < tol || viol > F(0.5) * prev) && break
        prev = viol
        lam .+= W .* r
    end
    return c, viol
end

function _scattered_auglag(A::SparseMatrixCSC{F}, f::Vector{F},
                           R::SparseMatrixCSC{F}, W::F, iters::Int, tol::F,
                           verbose::Bool, ::Val{:cholesky}) where {F}
    H = W * (A' * A) + R' * R
    # Jacobi equilibration: boundary-adjacent columns carry the (large)
    # polynomial-extrapolation weights of the deep ghosts, so column norms
    # span many orders of magnitude for wide kernels (b9-b13). Symmetric
    # diagonal scaling D H D with D = diag(1 ./ sqrt.(diag(H))) restores
    # balanced scaling, preserves SPD, and lowers the roundoff error floor.
    d = [one(F) / sqrt(max(H[i, i], eps(F))) for i in 1:size(H, 1)]
    Dm = spdiagm(0 => d)
    Hs = Dm * H * Dm
    fac = try
        _scattered_spd_factorize(Hs)
    catch err
        err isa PosDefException || rethrow()
        # the normal matrix lost positive definiteness numerically (large grids with few data
        # rows): solve the stacked least-squares system by sparse QR instead, which does not
        # square the condition number
        @warn "fit_scattered (:exact): the normal equations are not numerically positive " *
              "definite on this grid ($(size(A, 2)) unknowns for $(size(A, 1)) points); " *
              "falling back to solver = :qr. A smaller gridsize/oversample avoids this."
        return _scattered_auglag(A, f, R, W, iters, tol, verbose, Val(:qr))
    end
    lam = zeros(F, size(A, 1))
    c   = zeros(F, size(A, 2))
    viol = F(Inf); prev = F(Inf)
    for it in 1:iters
        rhs = A' * (W .* f .- lam)
        c = d .* (fac \ Vector(d .* rhs))
        r = A * c - f
        viol = maximum(abs, r)
        verbose && println("  fit_scattered iter $it: constraint violation $viol")
        (viol < tol || viol > F(0.5) * prev) && break
        prev = viol
        lam .+= W .* r
    end
    return c, viol
end

function _scattered_auglag(A::SparseMatrixCSC{F}, f::Vector{F},
                           R::SparseMatrixCSC{F}, W::F, iters::Int, tol::F,
                           verbose::Bool, ::Val{:nullspace}) where {F}
    N, M = size(A)
    M <= 5_000 ||
        error("solver = :nullspace is dense (O(M^2) memory in M = $M unknowns); " *
              "use :cholesky or :qr for systems this large.")
 
    Ad = Matrix(A)
    Rd = Matrix(R)
 
    # particular solution: pivoted-QR least squares (exact when feasible)
    c0 = Ad \ f
 
    # nullspace basis of A from the full Q of a pivoted QR of A' (M x N):
    # A' = Q S P'  =>  the columns of Q beyond rank(A) span null(A).
    qrf = qr(Matrix(Ad'), ColumnNorm())
    dS = abs.(diag(qrf.R))
    rtol = maximum(dS; init = zero(F)) * max(N, M) * eps(F)
    r = count(>(rtol), dS)
    Qfull = qrf.Q * Matrix{F}(I, M, M)
    Z = Qfull[:, r+1:M]
 
    # reduced problem in the nullspace (pivoted-QR least squares again)
    c = if size(Z, 2) == 0
        c0                                   # constraints determine everything
    else
        z = (Rd * Z) \ (-(Rd * c0))
        c0 + Z * z
    end
 
    viol = maximum(abs, A * c - f)
    verbose && println("  fit_scattered nullspace: rank $r of $N constraints, " *
                       "$(size(Z, 2)) free directions, violation $viol")
    return c, viol
end