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
  or a `:b` kernel, the kernels that support nonuniform grids. `:auto` depends
  on the mode and the dimension. For `:smooth`, and for `:exact` in 1D, it is
  `:b5` in 1-3D: the accuracy of a `:b` kernel at a cost that is small once a
  `:smooth` fit has coarsened to the grid its smoothing needs, and in 1D the
  fastest and most accurate `:exact` choice. Note this differs from the gridded
  default (`:b7` for 1-2D): for scattered data the boundary-condition tables
  cap all b-kernels at degree-7 reproduction, and `:b5`'s shallower ghost
  extrapolation gives it the lowest roundoff floor under `:cholesky`. For
  `:exact` on a uniform grid from 2D up the narrow kernels take over, `:a4` in
  2-3D and `:a3` from 4D: the grid is then mostly gaps between the data points,
  where the roughness order sets the accuracy and the kernel order hardly
  matters, so the narrow kernels match the `:b` kernels at two to ten times
  less factorization time (and `:b5` falls behind them in 3D). For `:smooth` on
  a uniform grid the cheap kernels take over from 4D: `:a4` for 4-5D and `:a3`
  above. On nonuniform grids both modes use `:n3` from 4D (closed-form weights,
  a 4-wide stencil and no per-interval precompute). Interpolate the fitted grid
  with the fitting kernel.
- `mode::Symbol=:smooth`: `:smooth` or `:exact` (see above).
- `grid::Union{Symbol,Tuple}=:uniform`: the knot axes. `:uniform` is the uniform
  grid over the bounding box. `:adaptive` places the knots of each axis where
  they are needed, blended with the uniform knots by `alpha` and with the ratio
  of adjacent spacings capped at `ratio`. For `:exact` the need is the points: the
  knots sit at the quantiles of the coordinates along each axis, so a dense band
  of points gets a dense band of knots, which is what lets `:exact` interpolate
  clustered points at all. For `:smooth` the need is the signal: the uniform fit
  is made first, and its curvature along each axis places the knots — dense in a
  boundary layer, a step or a fast oscillation, sparse where the fit is flat — and
  the fit is repeated on them, with half the uniform fit's knot count unless
  `gridsize` says otherwise. The grid is still a tensor product, so it helps when
  the need is roughly separable along the axes, not for isolated features; in two
  or more dimensions points can still share a cell of an `:exact` grid where the
  clusters of several axes overlap, and `oversample` is the remedy. A tuple of
  strictly increasing knot vectors, one per axis, is used as given; the vectors
  must cover the data. Nonuniform grids
  give a nonuniform interpolant: no antiderivatives (`derivative = -1`) and
  no lazy construction. With the `:b` kernels construction costs one
  exact-arithmetic projection per interval, so keep their knot counts modest;
  `:n3` has no such cost.
- `alpha::Real=1.0`, `ratio::Real=2.0` (`grid = :adaptive` only): the blend
  between uniform (`0`) and quantile (`1`) knots, and the largest allowed ratio
  of adjacent knot spacings. The cap keeps the nonuniform kernels and their ghost
  extrapolation well conditioned; raise it for strongly graded data.
- `gridsize::Union{Symbol,Int,NTuple{D,Int}}=:auto`: interior knots per
  dimension. An `Int` or tuple is used as given in both modes. With `:auto`,
  both modes size the grid from `oversample` and the data count. `:smooth`
  fits on that grid first — a coarser grid could absorb part of the signal
  into the residual, where nothing in the fit reveals it — and then, when the
  fitted smoothing needs far fewer knots, refits on the grid that resolves its
  cutoff (about eight knots per cycle), warm-started from `lambda`. The
  returned grid is therefore as coarse as the fit allows; a grid much finer
  than the fit is ill-conditioned and would hold the smoothing short of its
  optimum. With user axes the knot counts are those of the axes, and
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
- `q::Union{Int,Symbol}=:auto`: order of the roughness seminorm. `:smooth`
  penalises the q-th differences along each axis; `:exact` the full isotropic
  `H^q` seminorm from 2D up, every mixed difference of total order `q` included
  (in 1D the two coincide). The isotropic seminorm is the one the polyharmonic
  splines minimise, and it decides what the completion does in the gaps between
  the data points, where from 2D up most of the grid lies: the axis-wise
  differences leave the mixed modes to the data alone, which costs accuracy
  from 3D up, most of all away from the data's centre. It favours features
  without a preferred direction; for a separable signal (a product of
  one-dimensional factors) the axis-wise differences would be the natural
  prior, and the fit is somewhat less accurate there. `:auto` uses, for
  `:smooth`, 5 for every kernel (higher orders make the smoothing system
  ill-conditioned, lower orders are less accurate); for `:exact` in 1D the
  ghost extrapolation's stencil width minus one, at least 5 on uniform grids
  (so 5 for `:b5`, 7 for the `:b7`-class tables); from 2D one order for all
  axes, independent of the kernel: 5, the order that completed the gaps best
  over a range of test functions in two to four dimensions, with 6 a little
  better in 3D and higher orders showing in the constraint violation. On
  nonuniform grids the differences are divided differences scaled to plain
  differences on the local spacing, and `:n3` keeps `q = 4`, the highest order
  whose completion stays bounded where the data thin out towards the bounds
  (the `:b` kernels' full-degree ghost extrapolation oscillates there at any
  `q`, so on such data prefer `:n3`). For data that oscillate everywhere (a
  chirp) no `q` smooths without biasing the signal, and GCV then leaves
  `lambda` near zero: the knot count is the regulariser in that case.
- `lambda::Union{Symbol,Real}=:gcv` (`:smooth` only): the smoothing parameter,
  or `:gcv` to choose it by generalized cross-validation. The `lambda` returned
  by a fit reproduces that fit when passed back.
- `bc::Union{Symbol,Tuple}=:poly` (`:smooth` on uniform grids only): the ghost
  rule at the ends of the grid, `:poly` (the kernel's full-degree extrapolation,
  as in `:exact`), `:quadratic` or `:linear`, as a Symbol for every end, a
  `(left, right)` pair for every axis, or one pair per axis, as in
  `convolution_interpolation`. A lower degree amplifies the noise less at an
  end but biases the fit where the signal is steep there; the right choice
  depends on the data, so `:poly` is the default and a fit that is noisy at a
  flat end is worth repeating with `:linear` there. Interpolate the fitted
  grid with the same `bc`. On nonuniform grids the ghosts are the basis's own.
- `solver::Symbol=:auto` (`:exact` only): how the exact fit is solved. `:auto`
  uses `:cholesky` up to 3D and `:dual` from 4D, where `:dual` applies. It
  also switches from `:cholesky` to `:dual`, with a warning, when the
  factorization is predicted to need more memory than the machine has. A
  solver given explicitly is used as given.
  - `:cholesky` factors the Jacobi-equilibrated SPD normal system in the grid
    unknowns with CHOLMOD. It is the most accurate choice up to 3D, and with
    the narrow kernels its error is set by the fit, not by roundoff. The
    memory of the factorization grows steeply with the grid from 3D on. If
    the system loses positive definiteness numerically it falls back to `:qr`
    while that is affordable, and raises an error otherwise.
  - `:qr` solves each multiplier iteration as a sparse least-squares problem
    via SPQR without forming normal equations; it is several times slower but
    lowers the roundoff floor of the wide kernels, which rises with the
    kernel's width. Use `solver=:qr, kernel=:b7` when chasing the deepest
    accuracy. Its memory grows as the rows times the unknowns of the
    least-squares system, which limits it to small grids from 3D up.
  - `:dual` solves a dense system with one unknown per data point: the same
    roughness order, measured with reflecting boundaries so that it can be
    inverted by a cosine transform of the grid. Its cost follows the point
    count and not the grid (memory: 8 bytes times the square of the point
    count), which makes exact fits practical in 4D and above, and makes it
    the faster and lighter choice in 3D. It reproduces the data to rounding.
    Its accuracy is comparable to `:cholesky` in the interior of the box;
    near the faces it is less accurate once the data are dense enough to
    resolve the function there. In 3D with thousands of points that loss
    reaches orders of magnitude for smooth functions, most of all for one
    that is steep at a face. It can be chosen in any dimension, and needs a
    uniform grid, `Float64` data, and more data points than there are
    polynomials of total degree below `q` in `D` variables (with `q = 5`: 35
    in 3D, 70 in 4D, 126 in 5D). It runs on the available threads.
  - `:nullspace` is deprecated and will be removed in the next breaking
    release: a dense direct solve through a null-space basis of the
    constraints, limited to small problems and less accurate than the others.
- `W::Real=1e6` (`:exact` with `:cholesky` or `:qr`): augmented-Lagrangian
  penalty weight.
- `iters::Int=20`, `tol::Real=1e-11` (`:exact` with `:cholesky` or `:qr`):
  multiplier iteration controls; convergence typically takes 1-2 iterations.
  `:dual` solves directly and does not iterate.
- `verbose::Bool=false`: `:exact` prints the constraint violation per
  iteration, or per solve with `:dual`; `:smooth` prints the grid, `lambda`,
  cutoff, effective degrees of freedom and estimated noise of the fit.

# Returns
`(knots, vals, third)`: the tuple of interior knot axes (ranges for a uniform
grid, vectors otherwise), the array of fitted interior grid values, and
- for `:smooth`, `(lambda = λ, noise = σ)`: the smoothing parameter used and
  the estimated standard deviation of the data about the fit,
  `√(‖A c − f‖² / (n − edf))` with `edf` the fit's effective degrees of freedom;
- for `:exact`, `viol`: the final maximum constraint violation.

Construct an interpolant with `convolution_interpolation(knots, vals; kernel, bc=:poly, ...)`,
or use `convolution_interpolation(points, values)` which does this directly.

Notes: by default `:exact` solves a sparse SPD system up to 3D (CHOLMOD for
`Float64`, dense Cholesky otherwise) and a dense system in the data points
from 4D; see `solver`. Exact interpolation requires distinct data points with
consistent values. With `:cholesky` and `:qr`, duplicated points with
conflicting values are resolved in the least-squares sense and reported
through `viol`. `:dual` cannot solve data with duplicated points: it warns,
and `viol` shows that the data were not reproduced. `:smooth` always solves in
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
                       bc::Union{Symbol,Tuple}=:poly,
                       solver::Symbol=:auto,
                       W::Real=1e6, iters::Int=20, tol::Real=1e-11,
                       verbose::Bool=false)

    solver in (:auto, :cholesky, :qr, :dual, :nullspace) ||
        throw(ArgumentError("solver must be :auto, :cholesky, :qr or :dual, got $solver."))
    if solver === :nullspace
        @warn "solver = :nullspace is deprecated and will be removed in the next breaking " *
              "release: it is less accurate than :cholesky and :qr. Use one of those instead."
    end
    # the ghost rule per axis and end, normalised to ((left, right), …): a Symbol applies
    # everywhere, a pair of Symbols to both ends of every axis, a tuple of pairs per axis
    D_bc = size(points, 1)
    bcs = bc isa Symbol ? ntuple(_ -> (bc, bc), D_bc) :
          bc isa Tuple{Symbol,Symbol} ? ntuple(_ -> bc, D_bc) :
          length(bc) == D_bc && all(b -> b isa Tuple{Symbol,Symbol}, bc) ? NTuple{D_bc,Tuple{Symbol,Symbol}}(bc) :
          throw(ArgumentError("bc must be a Symbol, a (left, right) pair of Symbols, or one pair per axis; got $bc."))
    for (l, r) in bcs, b in (l, r)
        b in (:poly, :quadratic, :linear) ||
            throw(ArgumentError("bc must be :poly, :quadratic or :linear, got :$b."))
    end
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

    # ---- :smooth on an adaptive grid: the knots follow the signal ---------------
    # The uniform fit first (coarsened by the package to the grid its smoothing needs);
    # its curvature places the knots, and the fit is repeated on them. The uniform pass
    # uses the uniform default kernel; the adaptive passes use `kernel` (default :b5).
    if mode === :smooth && grid === :adaptive
        ku, vu, _ = fit_scattered(points, values; mode = :smooth, grid = :uniform,
                                  gridsize, oversample, q, lambda, bc, verbose)
        s  = convolution_interpolation(ku, vu; kernel = _default_scattered_kernel(D, :smooth, false), bc = bc)
        # the adaptive knot count per axis: a fraction of the uniform fit's, but at least the
        # 1D floor where the uniform fit has that many, and never more than the uniform fit
        g  = gridsize isa Int ? ntuple(_ -> gridsize, D) :
             gridsize isa Tuple ? NTuple{D,Int}(gridsize) :
             ntuple(d -> max(min(_SMOOTH_MIN_KNOTS, length(ku[d])), round(Int, _SIGNAL_BUDGET * length(ku[d]))), D)
        lo = ntuple(d -> minimum(view(points, d, :)), D)
        hi = ntuple(d -> maximum(view(points, d, :)), D)
        local knots, vals, info
        for pass in 1:_SIGNAL_PASSES
            axes = _signal_axes(s, g, lo, hi; alpha, ratio)
            knots, vals, info = fit_scattered(points, values; kernel, mode = :smooth, grid = axes,
                                              q, lambda, verbose)
            kern = kernel === :auto ? _default_scattered_kernel(D, :smooth, true) : kernel
            s = convolution_interpolation(knots, vals; kernel = kern)
        end
        return knots, vals, info
    end
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
                          _default_roughness_order(mode, kernels[d], ns[d], true, D), D)
    else
        for k in kernels
            haskey(POLYNOMIAL_GHOST_COEFFS, k) && k !== :n3 ||
                throw(ArgumentError("fit_scattered requires a kernel with a polynomial " *
                                    "boundary-condition table; :$k is not supported."))
        end
        G  = ntuple(d -> get_polynomial_ghost_coeffs(:poly, kernels[d]), D)
        ns = ntuple(d -> size(G[d], 2), D)                           # BC stencil width (:poly is the widest rule)
        qv = ntuple(d -> q !== :auto ? q::Int :                      # roughness orders
                        _default_roughness_order(mode, kernels[d], ns[d], false, D), D)
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
                                        n_given, g_cap, lambda, verbose; bc = bcs)
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

    # ---- solver ---------------------------------------------------------------
    q_iso = _isotropic_order(qv)
    if solver === :auto
        solver_used = _default_scattered_solver(D, nonuniform, Fs, Np, q_iso)
    else
        solver_used = solver
    end

    sparse_fit = nothing
    if solver_used !== :dual
        # sparse system on the interior unknowns (fit_exact.jl), with
        # the isotropic H^q seminorm: in 1D the axis-wise differences, from 2D up every mixed
        # difference of total order q as well, which the scattered data alone pin too loosely
        A_eff, R_eff = _scattered_operators(pts, kernels, axes, qv; isotropic = true)
        if solver === :auto
            # the default must not exhaust the machine's memory: :cholesky declines a
            # factorization predicted to need more than there is and returns nothing
            sparse_fit = _scattered_auglag(A_eff, f, R_eff, Fs(W), iters, Fs(tol), verbose,
                                           Val(:cholesky); memory_limit = Sys.total_memory())
        else
            # a solver chosen explicitly is honoured without that check
            sparse_fit = _scattered_auglag(A_eff, f, R_eff, Fs(W), iters, Fs(tol), verbose,
                                           Val(solver_used))
        end
        if sparse_fit === nothing
            _scattered_dual_applies(D, nonuniform, Fs, Np, q_iso) ||
                error("fit_scattered (:exact): the Cholesky factorization of this grid " *
                      "($(prod(n_int)) unknowns) is predicted to need more memory than this " *
                      "machine has, and solver = :dual does not apply here (it needs a " *
                      "uniform grid, Float64 data and more data points). Use a smaller " *
                      "gridsize/oversample; solver = :cholesky forces the factorization.")
            @warn "fit_scattered (:exact): the Cholesky factorization of this grid " *
                  "($(prod(n_int)) unknowns) is predicted to need more memory than this " *
                  "machine has; solving with solver = :dual instead, which is less accurate " *
                  "near the faces of the box. A smaller gridsize/oversample keeps " *
                  ":cholesky; solver = :cholesky forces the factorization."
            solver_used = :dual
        end
    end

    if solver_used === :dual
        # dense system in the data points (fit_dual.jl): the same roughness order, measured
        # with reflecting boundaries, inverted by a cosine transform of the uniform grid
        nonuniform &&
            throw(ArgumentError("solver = :dual needs a uniform grid; use :cholesky or :qr " *
                                "with grid = :adaptive or user-supplied axes."))
        Fs === Float64 ||
            throw(ArgumentError("solver = :dual solves in Float64; use :cholesky for $Fs data."))
        c_int, viol = _scattered_dual(pts, f, kernels, axes, q_iso, verbose)
    else
        c_int, viol = sparse_fit
    end

    if viol > max(Fs(tol), sqrt(eps(Fs))) * max(one(Fs), maximum(abs, f))
        if solver_used === :dual
            @warn "fit_scattered (:exact, solver = :dual): constraint violation $viol did " *
                  "not reach machine precision, so the result does not reproduce the data. " *
                  "The dense system in the data points is singular or ill-conditioned: the " *
                  "data may contain duplicated or nearly coincident points, or the grid may " *
                  "be too fine for this solver (try a smaller gridsize/oversample, or " *
                  "solver = :cholesky)."
        else
            hint = nonuniform ? "" :
                   " If the points are clustered, `grid = :adaptive` places the knots where " *
                   "the points are, so that each point gets its own knot spacing."
            @warn "fit_scattered (:exact): constraint violation $viol did not reach " *
                  "machine precision. The system may be overdetermined for this grid " *
                  "(try a larger gridsize/oversample) or the data may contain " *
                  "duplicated points with conflicting values. Result is the " *
                  "least-squares fit." * hint
        end
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
is forwarded to the interpolant, and `extrap`. On a uniform grid the boundary
condition is fixed to `:poly` for consistency with the fit; on a nonuniform
grid the interpolant's own ghost extrapolation is the one the fit used.
A nonuniform grid gives a nonuniform interpolant (`:b5` by default for
`:smooth` in 1-3D and for `:exact` in 1D, `:n3` otherwise; see
`fit_scattered`): no antiderivatives (`derivative = -1`) and no lazy ghosts.

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
                                   solver::Symbol=:auto,
                                   W::Real=1e6, iters::Int=20, tol::Real=1e-11,
                                   verbose::Bool=false,
                                   derivative::Union{Int,Tuple{Vararg{Int}}}=0,
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
    itp = ConvolutionInterpolation(knots, vals;
                                       kernel = kernel_res, bc = :poly, derivative)
    return ConvolutionExtrapolation(itp, _extrap_type(extrap))
end

# =============================================================================
# internals
# =============================================================================

"""Default roughness order of the fit on one axis. `:smooth` uses 5 for every kernel:
higher orders make the smoothing system ill-conditioned, lower orders are less accurate.
`:exact` fills the gaps between the data points with the minimum-roughness completion. In
1D the order is the ghost extrapolation's stencil width minus one (5 for `:b5`, 7 for the
`:b7`-class tables), at least 5 on uniform grids and at most 7 on nonuniform ones. From 2D
up the penalty is the isotropic `H^q` seminorm, whose order is one for all axes and does
not depend on the kernel: 5, the order that completed the gaps best over a range of test
functions in two to four dimensions; `:n3` on nonuniform grids keeps 4, the highest order
whose completion stays bounded where the data thin out towards the bounds."""
function _default_roughness_order(mode::Symbol, kernel::Symbol, ns::Int, nonuniform::Bool,
                                  D::Int)
    mode === :smooth && return 5
    if D == 1
        if nonuniform
            return min(ns - 1, 7)
        end
        return max(ns - 1, 5)
    end
    if nonuniform && kernel === :n3
        return 4
    end
    return 5
end

"""Scattered-path default kernel. `:smooth`: `:b5` for 1-3D on both grid types (the fit
coarsens to the grid its smoothing needs, so the wide kernel stays cheap), then `:a4` for
4-5D and `:a3` above on uniform grids, `:n3` on nonuniform grids (closed-form weights, no
per-interval precompute). `:exact`: `:b5` in 1D on both grid types, where it is both the
fastest and the most accurate choice; from 2D `:a4` (2-3D) and `:a3` (4D+) on uniform
grids, `:n3` on nonuniform grids. From 2D up the grid is mostly gaps between the data
points, the roughness order rather than the kernel order sets the accuracy, and the narrow
kernels match the `:b` kernels at a fraction of the factorization cost; on nonuniform
grids the `:b` kernels also oscillate where the data thin out towards the bounds, which
`:n3`'s quadratic ghosts do not. Measured on a Runge bump and a smooth wave, uniform and
clustered points, 1-4D."""
function _default_scattered_kernel(D::Int, mode::Symbol = :smooth, nonuniform::Bool = false)
    if mode === :smooth
        D <= 3 && return :b5                        # the fit coarsens its grid, so the wide kernel stays cheap
        nonuniform && return :n3                    # 4D+: the cheap kernels
        return D <= 5 ? :a4 : :a3
    else                                            # :exact
        D == 1 && return :b5                        # fastest and most accurate in 1D
        nonuniform && return :n3                    # 2D+: the grid is mostly gaps, the narrow kernels win
        return D <= 3 ? :a4 : :a3
    end
end

"""Whether `solver = :dual` can solve this fit: it needs a uniform grid, Float64, and more
data points than there are polynomials of total degree below `q` in `D` variables, which its
penalty does not see and which it determines from the data."""
function _scattered_dual_applies(D::Int, nonuniform::Bool, ::Type{F}, Np::Int,
                                 q::Int) where {F}
    return !nonuniform && F === Float64 && Np > binomial(q - 1 + D, D)
end

"""Default solver of the `:exact` fit. `:cholesky` up to 3D, where the sparse factorization
is the more accurate choice near the faces of the box. From 4D `:dual` where it applies
(`_scattered_dual_applies`): there the fill of the factorization makes `:cholesky` slow and
heavy on memory, while the dual system has only one unknown per data point."""
function _default_scattered_solver(D::Int, nonuniform::Bool, ::Type{F}, Np::Int,
                                   q::Int) where {F}
    if D >= 4 && _scattered_dual_applies(D, nonuniform, F, Np, q)
        return :dual
    end
    return :cholesky
end
