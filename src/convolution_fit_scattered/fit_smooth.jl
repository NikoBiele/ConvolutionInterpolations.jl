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
const _SMOOTH_GOLDEN     = (sqrt(5) - 1) / 2
const _SMOOTH_PROBES     = 10              # Hutchinson probe vectors
const _SMOOTH_G0         = 32              # first grid (knots per axis), at most the cap
const _SMOOTH_GAMMA      = 8               # knots per cycle of the cutoff
const _SMOOTH_REFINE     = 1.25            # refine only when the needed grid is this much larger
const _SMOOTH_MAX_PASSES = 4               # at most this many grids

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

"""Operators of the penalised fit on the grid with `n_int` interior knots per axis: the
collocation `Ae` and scaled roughness `Re` on the interior unknowns, their normal matrices,
`Aᵀf`, the reference `λref` and the probe images `U = Aᵀ·Z`."""
function _smooth_system(pts::Matrix{Float64}, f::Vector{Float64}, kernels::NTuple{D,Symbol},
                        eqs::NTuple{D,Int}, ng::NTuple{D,Int}, G::NTuple{D,Matrix{Float64}},
                        ns::NTuple{D,Int}, qv::NTuple{D,Int}, lo::NTuple{D,Float64},
                        hi::NTuple{D,Float64}, n_int::NTuple{D,Int}) where {D}
    n       = size(pts, 2)
    knots   = ntuple(d -> range(lo[d], hi[d], length = n_int[d]), D)
    h       = ntuple(d -> Float64(step(knots[d])), D)
    n_full  = ntuple(d -> n_int[d] + 2ng[d], D)
    x0_full = ntuple(d -> lo[d] - ng[d] * h[d], D)

    A = _scattered_collocation(pts, kernels, eqs, n_full, x0_full, h)
    S = _scattered_extension(n_full, n_int, ng, ns, G, Float64)
    R = _scattered_roughness(n_full, qv, Float64)

    # physical scaling of each axis's block of q-th differences, in normalised coordinates
    ĥ   = ntuple(d -> 1.0 / (n_int[d] - 1), D)
    vol = prod(ĥ)
    rowscale = reduce(vcat, [fill(sqrt(vol / ĥ[dd]^(2qv[dd])),
                                  (n_full[dd] - qv[dd]) * prod((n_full[d] for d in 1:D if d != dd); init = 1))
                             for dd in 1:D])
    length(rowscale) == size(R, 1) || error("roughness block sizes do not match the operator.")

    Ae  = A * S                                                 # collocation, interior unknowns
    Re  = spdiagm(0 => rowscale) * (R * S)                      # scaled roughness, interior unknowns
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
        c   = dv .* (F \ (dv .* (W .* sys.Atf)))
        rss = sum(abs2, sys.Ae * c .- f)
        edf = W * sum(sys.U .* (dv .* (F \ (dv .* sys.U)))) / K # Hutchinson tr H
        V   = edf < n ? n * rss / (n - edf)^2 : Inf
        return (t = t, V = V, c = c, edf = edf, rss = rss)
    end

    # sparse QR of the stacked system [√W·A; R̃]: K[prow, pcol] = Q·R, so that
    # uᵀ(KᵀK)⁻¹u = ‖R⁻ᵀ·u[pcol]‖²
    function by_qr(t)
        W  = 1 / (n * sys.λref * 10.0^t)
        Fq = qr(vcat(sqrt(W) .* sys.Ae, sys.Re))
        nqr[] += 1
        size(Fq.R) == (nu, nu) || return failed(t)              # rank deficient
        any(iszero, diag(Fq.R)) && return failed(t)             # rank truncated by SPQR
        c  = Fq \ vcat(sqrt(W) .* f, zeros(size(sys.Re, 1)))
        Lt = LowerTriangular(sparse(transpose(Fq.R)))
        Y  = try
            Lt \ sys.U[Fq.pcol, :]
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
    _fit_smooth(pts, f, kernels, eqs, ng, G, ns, qv, lo, hi, n_fixed, g_cap, λ, verbose)

Penalised fit of `mode = :smooth`. With `n_fixed === nothing` (gridsize = :auto) the grid
starts at `min(32, g_cap)` knots per axis and is refined, up to `g_cap`, until it resolves the
fitted cutoff; otherwise the grid `n_fixed` is used as given. Returns the interior knot counts,
the fitted interior coefficients (Float64, column-major over the grid) and `(lambda, noise)`,
with `noise` the estimated standard deviation of the data about the fit, √(‖A c − f‖²/(n − tr H)).
"""
function _fit_smooth(pts::Matrix{Float64}, f::Vector{Float64}, kernels::NTuple{D,Symbol},
                     eqs::NTuple{D,Int}, ng::NTuple{D,Int}, G::NTuple{D,Matrix{Float64}},
                     ns::NTuple{D,Int}, qv::NTuple{D,Int}, lo::NTuple{D,Float64},
                     hi::NTuple{D,Float64}, n_fixed::Union{Nothing,NTuple{D,Int}},
                     g_cap::NTuple{D,Int}, λ, verbose::Bool) where {D}
    n      = size(pts, 2)
    n_int  = n_fixed === nothing ? ntuple(d -> min(_SMOOTH_G0, g_cap[d]), D) : n_fixed
    λ_prev = nothing
    local fit
    for pass in 1:_SMOOTH_MAX_PASSES
        sys = _smooth_system(pts, f, kernels, eqs, ng, G, ns, qv, lo, hi, n_int)
        fit = _smooth_gcv(sys, f, λ, λ_prev)
        if verbose
            k_c = join((round(_smooth_cutoff(fit.λ, qv[d]); sigdigits = 3) for d in 1:D), " × ")
            σ̂   = fit.edf < n ? sqrt(fit.rss / (n - fit.edf)) : NaN
            println("  fit_scattered (:smooth) grid $(join(n_int, "×")): lambda = $(round(fit.λ; sigdigits = 4)), " *
                    "cutoff ≈ $k_c cycles per box, edf ≈ $(round(fit.edf; sigdigits = 4)), " *
                    "noise ≈ $(round(σ̂; sigdigits = 3)), $(fit.nchol) Cholesky / $(fit.nqr) QR")
        end
        n_fixed === nothing || break                            # a given grid is used as is
        # refine the axes whose grid is clearly too coarse for the fitted cutoff
        g_need = ntuple(d -> ceil(Int, min(_SMOOTH_GAMMA * _smooth_cutoff(fit.λ, qv[d]),
                                           Float64(g_cap[d]))) + 1, D)
        grow   = ntuple(d -> n_int[d] < g_cap[d] && g_need[d] >= _SMOOTH_REFINE * n_int[d], D)
        any(grow) || break
        n_int  = ntuple(d -> grow[d] ? min(g_cap[d], g_need[d]) : n_int[d], D)
        λ_prev = fit.λ                                          # warm start on the next grid
    end
    σ̂ = fit.edf < n ? sqrt(fit.rss / (n - fit.edf)) : NaN
    return n_int, fit.c, (lambda = fit.λ, noise = σ̂)
end