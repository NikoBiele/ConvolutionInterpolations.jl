# =============================================================================
# mode = :smooth: penalised least squares with a GCV-chosen smoothing parameter
# =============================================================================
#
# The fit minimises
#
#     (1/n)·‖A c − f‖² + λ·Σ_d ∫|∂_d^q s|²
#
# over the interior grid coefficients c. A is the collocation of the kernel basis at the data
# points with the ghost coefficients eliminated by the polynomial boundary condition (as in
# :exact). The integral is discretised by q-th differences in coordinates normalised to the
# data's bounding box: the differences along axis d are weighted by √(Π_k ĥ_k / ĥ_d^(2q_d)),
# with ĥ_d = 1/(g_d − 1), so λ does not depend on the grid or on the units of each axis. With
# W = 1/(n·λ) the normal equations are (W·AᵀA + R̃ᵀR̃)·c = W·Aᵀf.
#
# λ = :gcv chooses λ by generalised cross-validation,
#
#     V(λ) = n·‖A c − f‖² / (n − tr H)²,   H = W·A·(W·AᵀA + R̃ᵀR̃)⁻¹·Aᵀ,
#
# with tr H estimated by Hutchinson's method from fixed ±1 probes. The search runs over
# t = log10(λ/λ_ref), where λ_ref balances the traces of the two terms: a coarse scan whose
# bracket is extended at the end that holds the minimum, then golden-section refinement.
#
# With gridsize = :auto the grid starts small and is refined until it resolves the fitted cutoff
# k_c = λ^(−1/(2q))/(2π) (cycles per bounding box) with about eight knots per cycle, capped at
# the density set by oversample. Later passes warm-start the search from the previous λ, which
# carries over between grids because the penalty is scaled in normalised coordinates. Heavy
# smoothing therefore runs on a small, well-conditioned grid.
#
# The normal equations are solved by Cholesky (Jacobi-equilibrated, one symbolic analysis per
# grid). A λ whose factorisation fails is solved by sparse QR of the stacked system
# [√W·A; R̃] instead, with tr H from the triangular factor. Everything is solved in Float64.

const _SMOOTH_T_START    = (-4.0, 12.0)    # initial coarse scan of t = log10(λ/λ_ref)
const _SMOOTH_T_LIMIT    = (-40.0, 40.0)   # the bracket is never extended beyond these
const _SMOOTH_T_COARSE   = 2.0             # coarse scan and extension step (decades)
const _SMOOTH_T_WARM     = 1.0             # scan step around a warm-start λ (decades)
const _SMOOTH_T_TOL      = 0.1             # golden-section tolerance (decades)
const _SMOOTH_FLAT_TOL   = 1e-6            # relative change of V that ends an extension
const _SMOOTH_RESID_TOL  = 1e-3            # relative residual above which a Cholesky solve is rejected
const _SMOOTH_GOLDEN     = (sqrt(5) - 1) / 2
const _SMOOTH_PROBES     = 10              # Hutchinson probe vectors
const _SMOOTH_GAMMA      = 8               # knots per cycle of the cutoff
const _SMOOTH_REFINE     = 1.25            # a grid within this factor of the needed one is kept
const _SMOOTH_MIN_KNOTS  = 32              # the coarsest grid the fit coarsens to (knots per axis)
const _SMOOTH_MAX_PASSES = 5               # fits on successively coarser grids, at most

"""SplitMix64 bit mixer: deterministic probe signs without a random-number dependency."""
function _splitmix64(x::UInt64)
    x += 0x9e3779b97f4a7c15                                    # golden-ratio increment (wraps)
    z = (x ⊻ (x >> 30)) * 0xbf58476d1ce4e5b9                   # mix
    z = (z ⊻ (z >> 27)) * 0x94d049bb133111eb                   # mix
    return z ⊻ (z >> 31)                                       # final avalanche
end

"""`n × K` matrix of ±1 Hutchinson probes, identical for identical `n` and `K`."""
_smooth_probes(n::Int, K::Int) =
    [isodd(_splitmix64(UInt64((k - 1) * n + i)) >> 63) ? 1.0 : -1.0 for i in 1:n, k in 1:K]

"""Cutoff (cycles per bounding box) of the smoothing with parameter `λ` and roughness order `q`."""
_smooth_cutoff(λ::Real, q::Int) = Float64(λ)^(-1 / (2q)) / (2π)

"""Operators of the penalised fit on the given axes: the collocation `Ae` and scaled roughness
`Re` on the interior unknowns, their normal matrices, `Aᵀf`, the reference `λref` and the probe
images `U = Aᵀ·Z`. `_scattered_operators` returns the roughness in plain-difference units on
both grid types (q-th differences on ranges, divided differences rescaled to match them on
vectors); the physical scaling into normalised coordinates is applied here to both."""
function _smooth_system(pts::Matrix{Float64}, f::Vector{Float64}, kernels::NTuple{D,Symbol},
                        qv::NTuple{D,Int}, axes::NTuple{D,<:AbstractVector}) where {D}
    n     = size(pts, 2)
    n_int = ntuple(d -> length(axes[d]), D)
    ng    = axes isa NTuple{D,<:AbstractRange} ?
            ntuple(d -> get_equations_for_degree(kernels[d]) - 1, D) :   # uniform: poly-table ghosts
            ntuple(d -> kernels[d] === :n3 ? 1 : nonuniform_b_params(kernels[d])[1], D)   # nonuniform: ghosts per side
    n_full = ntuple(d -> n_int[d] + 2ng[d], D)

    Ae, Re = _scattered_operators(pts, kernels, axes, qv)       # interior unknowns

    # physical scaling of each axis's block of q-th differences, in normalised coordinates
    ĥ   = ntuple(d -> 1.0 / (n_int[d] - 1), D)
    vol = prod(ĥ)
    rowscale = reduce(vcat, [fill(sqrt(vol / ĥ[dd]^(2qv[dd])),
                                  (n_full[dd] - qv[dd]) * prod((n_full[d] for d in 1:D if d != dd); init = 1))
                             for dd in 1:D])
    length(rowscale) == size(Re, 1) || error("roughness block sizes do not match the operator.")
    Re = spdiagm(0 => rowscale) * Re

    AtA = Ae' * Ae
    RtR = Re' * Re
    λref = sum(diag(AtA)) / (n * sum(diag(RtR)))                # balances the two traces
    U   = Matrix(Ae' * _smooth_probes(n, _SMOOTH_PROBES))       # probe images Aᵀz
    return (Ae = Ae, Re = Re, AtA = AtA, RtR = RtR, Atf = Ae' * f, λref = λref, U = U)
end

"""Penalised fit on one grid: at the given `λ`, or at the GCV minimum (`λ = :gcv`), with the
search warm-started around `λ_start` if given. Returns the coefficients, λ, tr H, the residual
sum of squares and the numbers of Cholesky and QR factorisations."""
function _smooth_gcv(sys, f::Vector{Float64}, λ, λ_start)
    n    = length(f)
    nu   = size(sys.Ae, 2)
    K    = size(sys.U, 2)
    Fref = Ref{Any}(nothing)                                    # CHOLMOD factor, reused per grid
    nchol = Ref(0)
    nqr   = Ref(0)
    failed(t) = (t = t, V = Inf, c = Float64[], edf = NaN, rss = NaN)

    # Cholesky of the equilibrated normal equations; nothing if it is not positive definite
    function by_cholesky(t)
        W  = 1 / (n * sys.λref * 10.0^t)                        # data weight for this λ
        H  = W * sys.AtA + sys.RtR
        dv = [1 / sqrt(max(x, eps())) for x in diag(H)]         # Jacobi scaling
        Dm = spdiagm(0 => dv)
        Hs = Symmetric(Dm * H * Dm)
        nchol[] += 1
        try
            if Fref[] === nothing
                Fref[] = cholesky(Hs)                           # symbolic + numeric
            else
                cholesky!(Fref[], Hs)                           # numeric only
            end
        catch err
            err isa PosDefException || rethrow()
            return nothing
        end
        F   = Fref[]
        bs  = dv .* (W .* sys.Atf)                              # the equilibrated right-hand side
        cs  = F \ bs
        # a factorisation can complete and still be wrong: when W·AᵀA is far below R̃ᵀR̃ the
        # penalty's nullspace (low-degree polynomials) is resolved from rounding noise, and the
        # solve is inaccurate before Cholesky fails outright. The residual of the equilibrated
        # system catches it; such a λ goes to the QR fallback like a failed factorisation
        norm(Hs * cs .- bs) <= _SMOOTH_RESID_TOL * norm(bs) || return nothing
        c   = dv .* cs
        rss = sum(abs2, sys.Ae * c .- f)
        edf = W * sum(sys.U .* (dv .* (F \ (dv .* sys.U)))) / K # Hutchinson tr H
        V   = edf < n ? n * rss / (n - edf)^2 : Inf
        return (t = t, V = V, c = c, edf = edf, rss = rss)
    end

    # sparse QR of the stacked system [√W·A; R̃]: K[prow, pcol] = Q·R, so that
    # uᵀ(KᵀK)⁻¹u = ‖R⁻ᵀ·u[pcol]‖²
    # the columns are scaled by the same Jacobi factors as the normal equations, so that the
    # stacked matrix has unit column norms and the data rows are not dropped by SPQR's rank
    # tolerance when W is small: K·D = Q·R, c = D·(K·D)⁻¹·b, and uᵀ(KᵀK)⁻¹u = ‖R⁻ᵀ·(D·u)[pcol]‖²
    function by_qr(t)
        W  = 1 / (n * sys.λref * 10.0^t)
        dv = [1 / sqrt(max(x, eps())) for x in W .* diag(sys.AtA) .+ diag(sys.RtR)]   # Jacobi scaling
        Fq = qr(vcat(sqrt(W) .* sys.Ae, sys.Re) * spdiagm(0 => dv))
        nqr[] += 1
        size(Fq.R) == (nu, nu) || return failed(t)              # rank deficient
        any(iszero, diag(Fq.R)) && return failed(t)             # rank truncated by SPQR
        c  = dv .* (Fq \ vcat(sqrt(W) .* f, zeros(size(sys.Re, 1))))
        Lt = LowerTriangular(sparse(transpose(Fq.R)))
        Y  = try
            Lt \ (dv .* sys.U)[Fq.pcol, :]
        catch err
            err isa SingularException || rethrow()
            return failed(t)
        end
        edf = W * sum(abs2, Y) / K
        (all(isfinite, c) && isfinite(edf)) || return failed(t)
        rss = sum(abs2, sys.Ae * c .- f)
        V   = edf < n ? n * rss / (n - edf)^2 : Inf
        return (t = t, V = V, c = c, edf = edf, rss = rss)
    end

    function evaluate(t)
        e = by_cholesky(t)
        return e === nothing ? by_qr(t) : e
    end

    best = Ref{Any}(nothing)
    function visit(t)
        e = evaluate(t)
        (best[] === nothing || e.V < best[].V) && (best[] = e)
        return e.V
    end

    if λ === :gcv
        # coarse scan (full range, or one decade either side of a warm start)
        tstep = λ_start === nothing ? _SMOOTH_T_COARSE : _SMOOTH_T_WARM
        ts = λ_start === nothing ? collect(_SMOOTH_T_START[1]:_SMOOTH_T_COARSE:_SMOOTH_T_START[2]) :
             (t0 = log10(Float64(λ_start) / sys.λref); [t0 - tstep, t0, t0 + tstep])
        Vs = [visit(t) for t in ts]
        # extend the end that holds the minimum until V rises, flattens or fails
        while true
            k = argmin(Vs)
            if k == length(ts) && ts[end] + tstep <= _SMOOTH_T_LIMIT[2]
                t_new = ts[end] + tstep
                V_new = visit(t_new)
                flat  = abs(V_new - Vs[end]) <= _SMOOTH_FLAT_TOL * Vs[end]
                push!(ts, t_new)
                push!(Vs, V_new)
                flat && break
            elseif k == 1 && ts[1] - tstep >= _SMOOTH_T_LIMIT[1]
                t_new = ts[1] - tstep
                V_new = visit(t_new)
                flat  = abs(V_new - Vs[1]) <= _SMOOTH_FLAT_TOL * Vs[1]
                pushfirst!(ts, t_new)
                pushfirst!(Vs, V_new)
                flat && break
            else
                break
            end
        end
        # golden-section refinement between the neighbours of the best scan point
        k = argmin(Vs)
        a, b = ts[max(k - 1, 1)], ts[min(k + 1, length(ts))]
        x1, x2 = b - _SMOOTH_GOLDEN * (b - a), a + _SMOOTH_GOLDEN * (b - a)
        f1, f2 = visit(x1), visit(x2)
        while b - a > _SMOOTH_T_TOL
            if f1 <= f2
                b, x2, f2 = x2, x1, f1
                x1 = b - _SMOOTH_GOLDEN * (b - a)
                f1 = visit(x1)
            else
                a, x1, f1 = x1, x2, f2
                x2 = a + _SMOOTH_GOLDEN * (b - a)
                f2 = visit(x2)
            end
        end
    else
        visit(log10(Float64(λ) / sys.λref))                     # the given λ
    end

    e = best[]
    isempty(e.c) && error("fit_scattered (:smooth): no smoothing parameter could be solved on " *
                          "this grid; try a smaller gridsize.")
    return (c = e.c, λ = sys.λref * 10.0^e.t, edf = e.edf, rss = e.rss,
            nchol = nchol[], nqr = nqr[])
end

"""
    _fit_smooth(pts, f, kernels, qv, make_axes, n_fixed, g_cap, λ, verbose)

Penalised fit of `mode = :smooth`. `make_axes(n_int)` builds the knot axes for `n_int` knots
per axis (uniform ranges, adaptive vectors or the user's own axes, see `_fit_axes`). With
`n_fixed === nothing` (gridsize = :auto) the first fit is on the largest grid `oversample`
allows, `g_cap`, so that nothing the points resolve can alias into the residual; the fitted
cutoff then says how many knots the fit needs, and when the grid is much finer than that the
fit is repeated on the coarser grid, warm-started from `λ`, until the grid matches the cutoff
(at most `_SMOOTH_MAX_PASSES` fits, never below `_SMOOTH_MIN_KNOTS` knots per axis). A grid
much finer than the fit is ill-conditioned — the q-th difference penalty spans (π/g)^(2q) in
eigenvalue — which stops the λ search short of heavy smoothing; the coarse grid frees it. With
a given `λ` the grid is chosen from its cutoff directly. If the cap limits the final grid, a
warning says the fit may be under-resolved. Otherwise the grid `n_fixed` is used as given.
Returns the axes of the final grid, the fitted interior coefficients (Float64, column-major
over the grid) and `(lambda, noise)`, with `noise` the estimated standard deviation of the
data about the fit, √(‖A c − f‖²/(n − tr H)). On adaptive grids the cutoff criterion is
applied to the knot count, i.e. to the grid's average resolution.
"""
function _fit_smooth(pts::Matrix{Float64}, f::Vector{Float64}, kernels::NTuple{D,Symbol},
                     qv::NTuple{D,Int}, make_axes, n_fixed::Union{Nothing,NTuple{D,Int}},
                     g_cap::NTuple{D,Int}, λ, verbose::Bool) where {D}
    n = size(pts, 2)
    # knots per axis that resolve the cutoff of the smoothing λ, within the cap and the floor
    wanted(λv) = ntuple(d -> clamp(ceil(Int, min(_SMOOTH_GAMMA * _smooth_cutoff(λv, qv[d]) + 1, 1e9)),
                                   min(_SMOOTH_MIN_KNOTS, g_cap[d]), g_cap[d]), D)
    # the first grid: as given; the cap for a GCV search; the cutoff's own grid for a given λ
    n_int = n_fixed !== nothing ? n_fixed :
            λ === :gcv          ? g_cap : wanted(λ)
    λ_start = nothing
    local axes, fit
    for pass in 1:_SMOOTH_MAX_PASSES
        axes = make_axes(n_int)
        sys  = _smooth_system(pts, f, kernels, qv, axes)
        fit  = _smooth_gcv(sys, f, λ, λ_start)
        if verbose
            k_c = join((round(_smooth_cutoff(fit.λ, qv[d]); sigdigits = 3) for d in 1:D), " × ")
            σ̂_p = fit.edf < n ? sqrt(fit.rss / (n - fit.edf)) : NaN
            println("  fit_scattered (:smooth) pass $pass, grid $(join(n_int, "×")): lambda = $(round(fit.λ; sigdigits = 4)), " *
                    "cutoff ≈ $k_c cycles per box, edf ≈ $(round(fit.edf; sigdigits = 4)), " *
                    "noise ≈ $(round(σ̂_p; sigdigits = 3)), $(fit.nchol) Cholesky / $(fit.nqr) QR")
        end
        # coarsening is for the GCV search on an automatic grid; a given grid or λ is used as is
        (n_fixed === nothing && λ === :gcv) || break
        want = wanted(fit.λ)
        all(n_int[d] <= _SMOOTH_REFINE * want[d] for d in 1:D) && break   # the grid matches the cutoff
        n_int   = ntuple(d -> min(n_int[d], want[d]), D)                  # never finer than it is
        λ_start = fit.λ
    end
    σ̂ = fit.edf < n ? sqrt(fit.rss / (n - fit.edf)) : NaN
    # knots per axis that would resolve the fitted cutoff; more than the grid has means the
    # fit is limited by the grid, not by the data
    want  = ntuple(d -> min(_SMOOTH_GAMMA * _smooth_cutoff(fit.λ, qv[d]) + 1, 1e9), D)
    short = ntuple(d -> want[d] >= _SMOOTH_REFINE * n_int[d], D)
    if any(short)
        need  = join((short[d] ? ceil(Int, want[d]) : n_int[d] for d in 1:D), "×")
        limit = n_fixed === nothing ? "`oversample`" : "`gridsize`"
        hint  = axes isa NTuple{D,<:AbstractRange} ?
                " If the points are clustered where the data vary, `grid = :adaptive` " *
                "concentrates the knots there and may resolve the fit at this knot count." : ""
        @warn "fit_scattered (:smooth): the grid ($(join(n_int, "×")) knots) is limited by " *
              "$limit but the fitted smoothing needs about $need knots; the fit may be " *
              "under-resolved, which also inflates the noise estimate. Increase $limit; " *
              "if the data are noise-free, use `mode = :exact`." * hint
    end
    return axes, fit.c, (lambda = fit.λ, noise = σ̂)
end