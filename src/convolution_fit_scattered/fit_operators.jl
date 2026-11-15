# =============================================================================
# collocation, ghost extension and roughness operators of the scattered fit
# =============================================================================

"""Collocation and roughness operators of the fit on the interior unknowns, `(A*S, R*S)`,
for the given axes. Ranges mean a uniform grid: the uniform kernels with the `:poly` ghost
tables, as the uniform path has always built them. Vectors mean a nonuniform grid: the
nonuniform kernels (`:n3` or the b-kernels) through the per-axis bases of `_nu_axis`, with
divided-difference roughness in plain-difference units; `_smooth_system` applies the same
`rowscale` to both grid types. `isotropic = true` selects the isotropic `H^q` seminorm
(`_scattered_roughness_iso`, one order for all axes) instead of the axis-wise differences."""
function _scattered_operators(pts::AbstractMatrix{F}, kernels::NTuple{D,Symbol},
                              axes::NTuple{D,<:AbstractRange}, qv::NTuple{D,Int};
                              bc::NTuple{D,Tuple{Symbol,Symbol}} = ntuple(_ -> (:poly, :poly), D),
                              isotropic::Bool = false) where {F,D}
    A, S, n_full = _scattered_uniform_parts(pts, kernels, axes; bc)
    if isotropic
        R = _scattered_roughness_iso(n_full, _isotropic_order(qv), F)
    else
        R = _scattered_roughness(n_full, qv, F)
    end
    return A * S, R * S
end

"""Collocation matrix `A` on the full (ghost-extended) uniform grid, the extension `S` from the
interior unknowns to the full grid, and the full grid size. `A * S` is the collocation matrix
of the fit on the interior unknowns; `_scattered_operators` adds the roughness operator, which
the `:dual` solver does not need."""
function _scattered_uniform_parts(pts::AbstractMatrix{F}, kernels::NTuple{D,Symbol},
                                  axes::NTuple{D,<:AbstractRange};
                                  bc::NTuple{D,Tuple{Symbol,Symbol}} = ntuple(_ -> (:poly, :poly), D)) where {F,D}
    eqs = ntuple(d -> get_equations_for_degree(kernels[d]), D)
    ng  = ntuple(d -> eqs[d] - 1, D)                       # ghosts per side
    Gl  = ntuple(d -> get_polynomial_ghost_coeffs(bc[d][1], kernels[d]), D)   # left ghost rule per axis
    Gr  = ntuple(d -> get_polynomial_ghost_coeffs(bc[d][2], kernels[d]), D)   # right ghost rule per axis
    n_int   = ntuple(d -> length(axes[d]), D)
    h       = ntuple(d -> F(step(axes[d])), D)
    n_full  = ntuple(d -> n_int[d] + 2ng[d], D)
    x0_full = ntuple(d -> F(first(axes[d])) - ng[d] * h[d], D)

    A = _scattered_collocation(pts, kernels, eqs, n_full, x0_full, h)
    S = _scattered_extension(n_full, n_int, ng, Gl, Gr, F)
    return A, S, n_full
end

function _scattered_operators(pts::AbstractMatrix{F}, kernels::NTuple{D,Symbol},
                              axes::NTuple{D,<:AbstractVector}, qv::NTuple{D,Int};
                              isotropic::Bool = false) where {F,D}
    bases = ntuple(d -> _nu_axis(kernels[d], axes[d], F), D)   # per-axis nonuniform basis
    M     = ntuple(d -> bases[d].M, D)                         # ghosts per side

    A = _scattered_collocation(pts, axes, bases)
    S = _scattered_extension(axes, bases, F)
    if isotropic
        R = _scattered_roughness_iso(axes, M, _isotropic_order(qv), F)
    else
        R = _scattered_roughness(axes, M, qv, F)
    end
    return A * S, R * S
end

"""The single roughness order of an isotropic seminorm: the per-axis orders must agree."""
function _isotropic_order(qv::NTuple{D,Int}) where {D}
    all(==(qv[1]), qv) ||
        throw(ArgumentError("the isotropic roughness seminorm needs one order on every axis, got $qv."))
    return qv[1]
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
                              ng::NTuple{D,Int}, Gl::NTuple{D,Matrix{Int}},
                              Gr::NTuple{D,Matrix{Int}}, ::Type{F}) where {F,D}
    Es = ntuple(D) do d
        nl, nr = size(Gl[d], 2), size(Gr[d], 2)      # stencil widths of the two ends' ghost rules
        I_ = Int[]; J_ = Int[]; V_ = F[]
        for j in 1:n_int[d]                          # interior identity
            push!(I_, ng[d] + j); push!(J_, j); push!(V_, one(F))
        end
        for g in 1:ng[d], j in 1:nl                  # left ghosts, nearest-first
            push!(I_, ng[d] + 1 - g); push!(J_, j); push!(V_, F(Gl[d][g, j]))
        end
        for g in 1:ng[d], j in 1:nr                  # right ghosts, mirrored
            push!(I_, n_full[d] - ng[d] + g)
            push!(J_, n_int[d] - nr + j)
            push!(V_, F(Gr[d][g, nr + 1 - j]))
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

"""q-th differences along one uniform axis of `n` knots: `n - q` rows with the binomial
weights `(-1)^(q-k) binomial(q, k)` at columns `i … i+q`."""
function _difference_rows(n::Int, q::Int, ::Type{F}) where {F}
    I_ = Int[]; J_ = Int[]; V_ = F[]
    for i in 1:(n - q), k in 0:q
        push!(I_, i); push!(J_, i + k)
        push!(V_, F((-1)^(q - k) * binomial(q, k)))
    end
    return sparse(I_, J_, V_, n - q, n)
end

"""All multi-indices `α` of `D` nonnegative integers with `sum(α) == q`: the orders of the
mixed differences of total order `q`."""
function _multi_indices(q::Int, D::Int)
    D == 1 && return [(q,)]
    out = NTuple{D,Int}[]
    for a in 0:q, rest in _multi_indices(q - a, D - 1)
        push!(out, (a, rest...))
    end
    return out
end

"""Roughness operator on the full grid: q-th differences along each axis,
stacked. Column-major kron ordering to match `_scattered_extension`."""
function _scattered_roughness(n_full::NTuple{D,Int}, q::NTuple{D,Int},
                              ::Type{F}) where {F,D}
    blocks = map(1:D) do dd
        ops = ntuple(d -> d == dd ? _difference_rows(n_full[d], q[d], F) :
                                    sparse(one(F) * I, n_full[d], n_full[d]), D)
        reduce(kron, reverse(ops))
    end
    return reduce(vcat, blocks)
end

"""Isotropic roughness operator on the full grid: the discrete `H^q` seminorm, every mixed
difference of total order `q` weighted by `sqrt(q!/α!)`, so that `‖R c‖²` discretises
`∫ Σ_{|α|=q} (q!/α!) |∂^α s|²`, the seminorm the polyharmonic splines minimise. Unlike the
axis-wise operator it also penalises the mixed modes, which from three dimensions up the
scattered data pin too loosely on their own; in 1D the two operators coincide.
Column-major kron ordering to match `_scattered_extension`."""
function _scattered_roughness_iso(n_full::NTuple{D,Int}, q::Int, ::Type{F}) where {F,D}
    blocks = map(_multi_indices(q, D)) do α
        w = sqrt(F(factorial(q)) / F(prod(factorial, α)))
        ops = ntuple(d -> α[d] == 0 ? sparse(one(F) * I, n_full[d], n_full[d]) :
                                      _difference_rows(n_full[d], α[d], F), D)
        w * reduce(kron, reverse(ops))
    end
    return reduce(vcat, blocks)
end

"""Full knot vectors of a nonuniform grid for the roughness operators: each axis normalised to
[0, 1] over the interior, with `M[d]` ghost knots per side mirroring the end spacings, as the
interpolant constructor places them."""
function _roughness_knots(axes::NTuple{D,<:AbstractVector}, M::NTuple{D,Int},
                          ::Type{F}) where {F,D}
    return ntuple(D) do d
        ax   = axes[d]
        lo   = F(first(ax))
        span = F(last(ax)) - lo
        x    = [(F(a) - lo) / span for a in ax]
        h1   = x[2] - x[1]
        hn   = x[end] - x[end - 1]
        vcat([x[1] - g * h1 for g in M[d]:-1:1], x, [x[end] + g * hn for g in 1:M[d]])
    end
end

"""q-th divided differences along one nonuniform axis with knots `x`, each row times ℓ^q with
ℓ its support length over q: a plain q-th difference on the local scale, O(1) on any grid and
equal to the uniform operator's binomial differences on a uniform grid."""
function _divided_difference_rows(x::Vector{F}, q::Int) where {F}
    n = length(x)
    I_ = Int[]; J_ = Int[]; V_ = F[]
    fac = F(factorial(q))
    for i in 1:(n - q)
        ell = (x[i + q] - x[i]) / q
        for k in 0:q
            denom = one(F)
            for m in 0:q
                m == k && continue
                denom *= x[i + k] - x[i + m]
            end
            push!(I_, i); push!(J_, i + k); push!(V_, ell^q * fac / denom)
        end
    end
    return sparse(I_, J_, V_, n - q, n)
end

"""Roughness operator on the full nonuniform grid: q-th divided differences along each axis
in plain-difference units (`_divided_difference_rows`), identity along the transverse axes,
stacked over the axes. `_smooth_system` applies the same mean-grid `rowscale` as on uniform
grids, so on a nonuniform grid the `:smooth` penalty is the sum of squared local differences
in mean-grid units rather than exactly ∫|∂^q s|²: λ's cutoff interpretation is then
approximate, while GCV still chooses λ from the data. Column-major kron ordering to match
`_scattered_extension`."""
function _scattered_roughness(axes::NTuple{D,<:AbstractVector}, M::NTuple{D,Int},
                              q::NTuple{D,Int}, ::Type{F}) where {F,D}
    xs = _roughness_knots(axes, M, F)
    blocks = map(1:D) do dd
        ops = ntuple(d -> d == dd ? _divided_difference_rows(xs[d], q[d]) :
                                    sparse(one(F) * I, length(xs[d]), length(xs[d])), D)
        reduce(kron, reverse(ops))
    end
    return reduce(vcat, blocks)
end

"""Isotropic roughness operator on the full nonuniform grid: the discrete `H^q` seminorm of
the uniform `_scattered_roughness_iso`, with the divided differences of
`_divided_difference_rows` along each axis in place of the binomial differences. Column-major
kron ordering to match `_scattered_extension`."""
function _scattered_roughness_iso(axes::NTuple{D,<:AbstractVector}, M::NTuple{D,Int},
                                  q::Int, ::Type{F}) where {F,D}
    xs = _roughness_knots(axes, M, F)
    blocks = map(_multi_indices(q, D)) do α
        w = sqrt(F(factorial(q)) / F(prod(factorial, α)))
        ops = ntuple(d -> α[d] == 0 ? sparse(one(F) * I, length(xs[d]), length(xs[d])) :
                                      _divided_difference_rows(xs[d], α[d]), D)
        w * reduce(kron, reverse(ops))
    end
    return reduce(vcat, blocks)
end
