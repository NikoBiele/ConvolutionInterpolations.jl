# =============================================================================
# mode = :exact, solver = :dual: one unknown per data point
# =============================================================================
#
# The sparse solvers of fit_exact.jl factorize a matrix in the grid unknowns. Its fill grows
# quickly with the dimension, and from 3D the factorization needs more memory than the grid
# itself by orders of magnitude. This solver finds the same grid from a dense system in the
# data points instead. It minimises
#
#     c' P c   subject to   A c = f
#
# over the interior grid values c, where A is the collocation matrix on the interior
# unknowns and P is the isotropic H^q seminorm of the sparse solvers on the ghost-extended
# grid.
#
# P is a sum of tensor products of small matrices, one per axis and difference order:
#
#     P = sum over |alpha| = q of (q! / alpha!) G[alpha_1] x ... x G[alpha_D],
#
# with G[k] the Gram matrix of the k-th difference of one ghost-extended axis in its
# interior unknowns (G[0]: of the extension alone). It is applied term by term along the
# axes and never stored. Its inverse has no such form, because the G[k] of an axis do not
# share their eigenvectors. The solver therefore works in two steps.
#
# 1. A surrogate penalty M that can be inverted: on each axis the modes V diagonalise G[q]
#    and G[0] together, and M keeps of P only its diagonal in the tensor product of those
#    modes. M has the null space of P, the polynomials that the q-th differences do not see
#    (the tail T), and the same behaviour at the faces of the box. With Z an orthonormal
#    basis of the data vectors orthogonal to the tail at the data points, the grid of least
#    M that reproduces the data is
#
#        (Z' A M⁺ A' Z) y = Z' f,    c1 = M⁺ A' Z y,
#        c = c1 + T b,    b from A T b = f - A c1.
#
#    The dense system has one unknown per data point (less the tail). It is assembled in
#    blocks of columns and factorized once. Nothing of size unknowns x points is stored: the
#    memory is the dense system, 8 bytes times the square of the point count, plus work
#    arrays bounded by _DUAL_WORK_BYTES.
#
# 2. Conjugate gradients on P itself, over the grids that reproduce the data, preconditioned
#    by M: every iteration applies P once and solves once more with the factorized dense
#    system. M and P are close enough that a few tens of iterations reach the grid the
#    sparse solvers find. In 1D M is P and there is nothing to iterate.
#
# The weight of a mode in M⁺ falls with its roughness to the power q, so on a grid with many
# knots per axis the smoothest modes outweigh the roughest by more than double precision can
# carry. From 2D the weights are therefore limited to a ratio of _DUAL_WEIGHT_RATIO. That
# alters M, the preconditioner, on a few smooth modes; the iteration still converges to the
# minimum of P.
#
# The result reproduces the data to rounding and agrees with the sparse solvers.

# memory of the grid-sized work arrays of the assembly, all tasks together
const _DUAL_WORK_BYTES   = 2^30
# largest ratio between two weights of the inverse surrogate penalty (2D and up)
const _DUAL_WEIGHT_RATIO = 1e10
const _DUAL_CG_ITERS     = 60     # most conjugate-gradient iterations on the true penalty
const _DUAL_CG_TOL       = 1e-6   # relative residual at which the iteration stops
const _DUAL_CG_LOOSE     = 1e-3   # residual above which an unfinished iteration is reported
# multiple of the rounding bound below which the penalty of a grid counts as zero
const _DUAL_ROUNDING     = 10

"""The penalty of one axis for the dual solver: `G[k + 1]` is the Gram matrix of the k-th
difference of the ghost-extended axis in its `n` interior unknowns, `k = 0:q` (`G[1]`: of
the extension alone). The columns of `V` are modes that diagonalise `G[q + 1]` and `G[1]`
together: `V' G[1] V` is the identity and `V' G[q + 1] V` is diagonal. The first modes are
the polynomials that the q-th difference does not see, in order of degree. `g[k + 1, j]` is
the diagonal entry of `G[k + 1]` for mode `j`, exactly zero where a polynomial has no k-th
difference."""
function _dual_axis_modes(kernel::Symbol, n::Int, q::Int, ::Type{F}) where {F}
    ng = get_equations_for_degree(kernel) - 1                   # ghosts per side
    ghost = get_polynomial_ghost_coeffs(:poly, kernel)
    n_full = n + 2 * ng
    E = Matrix(_scattered_extension((n_full,), (n,), (ng,), (ghost,), (ghost,), F))
    G = Vector{Matrix{F}}(undef, q + 1)
    G[1] = E' * E
    DE = E
    for k in 1:q
        # k-th difference of the extended axis
        DE = Matrix(_difference_rows(n_full, k, F)) * E
        G[k + 1] = DE' * DE
    end

    # the polynomials without a q-th difference, orthonormal in the G[1] inner product
    t = [F(2 * j - n - 1) / F(n - 1) for j in 1:n]              # knot coordinate in [-1, 1]
    seen = sqrt(eps(F)) * maximum(abs, DE)
    polys = Matrix{F}(undef, n, 0)
    for degree in 0:q-1
        v = t .^ degree
        # the ghost rule does not extend this degree exactly
        if maximum(abs, DE * v) > seen
            break
        end
        for pass in 1:2, j in 1:size(polys, 2)
            v .-= dot(view(polys, :, j), G[1] * v) .* view(polys, :, j)
        end
        v ./= sqrt(dot(v, G[1] * v))
        polys = hcat(polys, v)
    end
    n_poly = size(polys, 2)

    # the other modes: eigenvectors of G[q + 1] against G[1] in the complement of the
    # polynomials
    C = nullspace(polys' * G[1])
    modes = eigen(Symmetric(C' * G[q + 1] * C), Symmetric(C' * G[1] * C))
    V = hcat(polys, C * modes.vectors)

    g = Matrix{F}(undef, q + 1, n)
    for k in 0:q, j in 1:n
        if j <= n_poly && j - 1 < k
            g[k + 1, j] = zero(F)      # a polynomial of degree j - 1 has no k-th difference
        else
            g[k + 1, j] = dot(view(V, :, j), G[k + 1] * view(V, :, j))
        end
    end
    return (V = V, g = g, G = G)
end

"""The terms of the isotropic seminorm of order `q` in `D` dimensions: the difference orders
per axis and the weight `q! / alpha!` of each."""
function _dual_penalty_terms(q::Int, D::Int, ::Type{F}) where {F}
    return [(alpha, F(factorial(q)) / F(prod(factorial, alpha)))
            for alpha in _multi_indices(q, D)]
end

"""Diagonal of the true penalty in the tensor product of the axis modes: for the mode with
index `J`, the sum over the terms of the weight times the entries `g[alpha_d + 1, J_d]` of
the axes. Exactly zero for the polynomials the penalty does not see."""
function _dual_mode_weights(modes::NTuple{D,<:NamedTuple}, terms::Vector,
                            n_int::NTuple{D,Int},
                            ::Type{F}) where {F,D}
    w = zeros(F, n_int)
    for (alpha, coef) in terms, J in CartesianIndices(n_int)
        value = coef
        for d in 1:D
            value *= modes[d].g[alpha[d] + 1, J[d]]
        end
        w[J] += value
    end
    return w
end

"""The polynomial tail on the grid (unknowns x polynomials): the tensor modes whose weight
in `w` is zero, which span the null space of the penalty."""
function _dual_null_polynomials(modes::NTuple{D,<:NamedTuple}, w::Array{F,D}) where {F,D}
    n_int = size(w)
    null_modes = findall(iszero, w)
    tail = Matrix{F}(undef, prod(n_int), length(null_modes))
    for (col, J) in enumerate(null_modes), (row, I) in enumerate(CartesianIndices(n_int))
        value = one(F)
        for d in 1:D
            value *= modes[d].V[I[d], J[d]]
        end
        tail[row, col] = value
    end
    return tail
end

"""Weights of the inverse surrogate penalty per tensor mode: `1 / w`, and zero for the
polynomials of the tail, which the tail carries. No weight exceeds `ratio` times the
smallest one: on a grid with many knots per axis the smoothest modes would otherwise
outweigh the roughest by more than double precision can carry. The limit changes the
preconditioner of the iteration, not the grid it converges to."""
function _dual_inverse_weights(w::Array{F}, ratio::Real) where {F}
    floor_w = maximum(w) / F(ratio)
    w_inv = Vector{F}(undef, length(w))
    for i in eachindex(w_inv)
        if w[i] > 0
            w_inv[i] = one(F) / max(w[i], floor_w)
        else
            w_inv[i] = zero(F)
        end
    end
    return w_inv
end

"""Multiply each column of `X` (unknowns x columns), viewed on the grid `n_int`, along every
grid axis `d` by `mats[d]` from the right: entry `i` along the axis becomes
`sum(x[j] * mats[d][j, i] for j)`. `X` and `work` (same size) are both overwritten; the
result is returned and is one of the two."""
function _dual_mul_axes!(X::Matrix{F}, work::Matrix{F}, mats::NTuple{D,Matrix{F}},
                         n_int::NTuple{D,Int}) where {F,D}
    src = X
    dst = work
    for d in 1:D
        left = 1                                    # entries per slice ahead of axis d
        for k in 1:d-1
            left *= n_int[k]
        end
        right = div(length(src), left * n_int[d])   # slices: the later axes and the columns
        # the first axis: one product for all slices
        if left == 1
            mul!(reshape(dst, n_int[d], right), transpose(mats[d]),
                 reshape(src, n_int[d], right))
        else
            S = reshape(src, left, n_int[d], right)
            T = reshape(dst, left, n_int[d], right)
            for r in 1:right
                mul!(view(T, :, :, r), view(S, :, :, r), mats[d])
            end
        end
        src, dst = dst, src
    end
    return src
end

"""Inverse of the surrogate penalty applied to each column of `X` (unknowns x columns): to
the modes with `V`, times the weights `w`, back to the grid with `Vt`. `X` and `work` are
both overwritten; the result is returned and is one of the two."""
function _dual_inverse_penalty!(X::Matrix{F}, work::Matrix{F}, w::Vector{F},
                                V::NTuple{D,Matrix{F}}, Vt::NTuple{D,Matrix{F}},
                                n_int::NTuple{D,Int}) where {F,D}
    modal = _dual_mul_axes!(X, work, V, n_int)
    modal .*= w
    if modal === X
        return _dual_mul_axes!(modal, work, Vt, n_int)
    else
        return _dual_mul_axes!(modal, X, Vt, n_int)
    end
end

"""Columns `cols` of the dense system `Z' A M⁺ A' Z` into `K`, `block` columns at a time.
`sys` holds the operators of `_scattered_dual`."""
function _dual_assemble_columns!(K::Matrix{F}, cols::UnitRange{Int}, block::Int,
                                 sys) where {F}
    Np, n = size(sys.A)
    X = Matrix{F}(undef, n, 0)
    work = Matrix{F}(undef, n, 0)
    j0 = first(cols)
    while j0 <= last(cols)
        J = j0:min(j0 + block - 1, last(cols))
        if size(X, 2) != length(J)
            X = Matrix{F}(undef, n, length(J))
            work = Matrix{F}(undef, n, length(J))
        end
        E = zeros(F, Np, length(J))          # unit vectors that select the columns J of Z
        for (k, j) in enumerate(J)
            E[sys.n_tail + j, k] = one(F)
        end
        mul!(X, sys.At, sys.tail_qr.Q * E)   # A' Z on the grid, for these columns
        Y = _dual_inverse_penalty!(X, work, sys.w_inv, sys.V, sys.Vt, sys.n_int)
        # Z' A applied to the result
        K[:, J] = view(sys.tail_qr.Q' * (sys.A * Y), sys.n_tail+1:Np, :)
        j0 += block
    end
    return K
end

"""The factorized dense system `Z' A M⁺ A' Z` of the dual solver, assembled over the
threads; `nothing` if it is singular."""
function _dual_factorize(sys, ::Type{F}) where {F}
    Np, n = size(sys.A)
    n_free = Np - sys.n_tail                 # unknowns of the dense system
    K = Matrix{F}(undef, n_free, n_free)
    ntasks = min(Threads.nthreads(), n_free)
    block = max(1,
                min(cld(n_free, ntasks), div(_DUAL_WORK_BYTES, 2 * sizeof(F) * n * ntasks)))
    if ntasks == 1
        _dual_assemble_columns!(K, 1:n_free, block, sys)
    else
        per_task = cld(n_free, ntasks)
        blas_threads = BLAS.get_num_threads()
        BLAS.set_num_threads(1)              # the tasks use the cores, not BLAS inside them
        try
            Threads.@threads for t in 1:ntasks
                cols = ((t - 1) * per_task + 1):min(t * per_task, n_free)
                _dual_assemble_columns!(K, cols, block, sys)
            end
        finally
            BLAS.set_num_threads(blas_threads)
        end
    end
    for j in 1:n_free, i in 1:j-1            # symmetric up to rounding: make it exactly so
        mean_ij = (K[i, j] + K[j, i]) / 2
        K[i, j] = mean_ij
        K[j, i] = mean_ij
    end
    K_fac = lu!(K; check = false)
    if !issuccess(K_fac)
        return nothing
    end
    return K_fac
end

"""Inverse of the surrogate penalty applied to the grid vector `x`; a new vector."""
function _dual_inverse_surrogate(sys, x::Vector{F}) where {F}
    X = reshape(copy(x), length(x), 1)
    work = similar(X)
    return vec(_dual_inverse_penalty!(X, work, sys.w_inv, sys.V, sys.Vt, sys.n_int))
end

"""One multiplier per data point for the data vector `rhs`: the solution of the dense
system, which lies in the data vectors orthogonal to the tail."""
function _dual_multipliers(sys, K_fac, rhs::Vector{F}) where {F}
    Np = size(sys.A, 1)
    y = K_fac \ (sys.tail_qr.Q' * rhs)[sys.n_tail+1:Np]
    return sys.tail_qr.Q * vcat(zeros(F, sys.n_tail), y)
end

"""The grid closest to `x` in the surrogate penalty that reproduces the data `f`."""
function _dual_onto_data(sys, K_fac, x::Vector{F}, f::Vector{F}) where {F}
    step = _dual_inverse_surrogate(sys,
                                   sys.At * _dual_multipliers(sys, K_fac, f .- sys.A * x))
    moved = x .+ step
    return moved .+ sys.tail * (sys.tail_qr \ (f .- sys.A * moved))
end

"""The gradient `r` preconditioned by the surrogate penalty and moved into the grids that
vanish at the data points: the search direction of the conjugate gradients."""
function _dual_project(sys, K_fac, r::Vector{F}) where {F}
    u = _dual_inverse_surrogate(sys, r)
    g = u .- _dual_inverse_surrogate(sys, sys.At * _dual_multipliers(sys, K_fac, sys.A * u))
    return g .- sys.tail * (sys.tail_qr \ (sys.A * g))
end

"""The true penalty applied to the grid vector `x`, term by term along the axes and over the
threads; a new vector. `grams[d][k + 1]` is the Gram matrix of the k-th difference of axis
`d`."""
function _dual_true_penalty(x::Vector{F}, terms::Vector, grams::NTuple{D,Vector{Matrix{F}}},
                            n_int::NTuple{D,Int}) where {F,D}
    n = length(x)
    ntasks = min(Threads.nthreads(), length(terms))
    per_task = cld(length(terms), ntasks)
    partial = zeros(F, n, ntasks)                          # one sum per task
    Threads.@threads for t in 1:ntasks
        X = Matrix{F}(undef, n, 1)
        work = Matrix{F}(undef, n, 1)
        for i in ((t - 1) * per_task + 1):min(t * per_task, length(terms))
            alpha, coef = terms[i]
            copyto!(X, x)
            mats = ntuple(d -> grams[d][alpha[d] + 1], D)
            Y = _dual_mul_axes!(X, work, mats, n_int)
            for row in 1:n
                partial[row, t] += coef * Y[row, 1]
            end
        end
    end
    return vec(sum(partial; dims = 2))
end

"""Conjugate gradients on the true penalty over the grids that reproduce the data, started
from the surrogate fit `x` and preconditioned by the surrogate penalty. Stops at the
tolerance, at the iteration limit, or when the residual turns upwards, which is where
rounding takes over. Returns the iterate of the smallest residual, the iterations used and
that residual, relative to the first. Where the penalty of `x` is rounding alone, as for
data that are a polynomial the penalty does not see, there is nothing to iterate on: `x` is
returned with a residual of zero."""
function _dual_refine(sys, K_fac, x::Vector{F}, terms::Vector,
                      grams::NTuple{D,Vector{Matrix{F}}},
                      verbose::Bool) where {F,D}
    r = _dual_true_penalty(x, terms, grams, sys.n_int)
    x_best = copy(x)
    resid_best = one(F)
    iterations = 0
    # the rounding of r is bounded by the penalty of absolute values applied to |x|
    grams_abs = ntuple(d -> [abs.(G) for G in grams[d]], D)
    rounding = eps(F) * maximum(_dual_true_penalty(abs.(x), terms, grams_abs, sys.n_int))
    if maximum(abs, r) <= _DUAL_ROUNDING * rounding
        return x_best, iterations, zero(F)
    end
    g = _dual_project(sys, K_fac, r)
    rg = dot(r, g)
    rg_first = rg
    if !(rg_first > 0)                            # the surrogate fit is already the minimum
        return x_best, iterations, zero(F)
    end
    p = -g
    for it in 1:_DUAL_CG_ITERS
        Pp = _dual_true_penalty(p, terms, grams, sys.n_int)
        step = rg / dot(p, Pp)
        x .+= step .* p
        r .+= step .* Pp
        g = _dual_project(sys, K_fac, r)
        rg_new = dot(r, g)
        resid = sqrt(abs(rg_new) / rg_first)
        iterations = it
        verbose && println("  fit_scattered dual iter $it: relative residual $resid")
        if resid < resid_best
            resid_best = resid
            x_best .= x
        end
        if resid <= F(_DUAL_CG_TOL) || resid > 10 * resid_best
            break
        end
        p = -g .+ (rg_new / rg) .* p
        rg = rg_new
    end
    return x_best, iterations, resid_best
end

"""Exact fit on a uniform grid by the dual system: the interior grid values that reproduce
`f` at `pts` with the least roughness of order `q`, the grid the sparse solvers find, and
the largest data misfit (`Inf` if the dense system is singular)."""
function _scattered_dual(pts::AbstractMatrix{F}, f::Vector{F}, kernels::NTuple{D,Symbol},
                         axes::NTuple{D,<:AbstractRange}, q::Int, verbose::Bool) where {F,D}
    A_full, S, _ = _scattered_uniform_parts(pts, kernels, axes)
    A = A_full * S                                    # collocation on the interior unknowns
    Np, n = size(A)
    n_int = ntuple(d -> length(axes[d]), D)

    modes = ntuple(d -> _dual_axis_modes(kernels[d], n_int[d], q, F), D)
    terms = _dual_penalty_terms(q, D, F)
    w = _dual_mode_weights(modes, terms, n_int, F)
    tail = _dual_null_polynomials(modes, w)
    n_tail = size(tail, 2)
    Np > n_tail ||
        throw(ArgumentError("solver = :dual needs more than $n_tail data points in $(D)D " *
                            "with q = $q (the polynomials the roughness measure does not " *
                            "see); got $Np. Use solver = :cholesky, or a lower q."))

    if D == 1
        # no iteration follows in 1D: the weights stay exact
        ratio = Inf
    else
        ratio = _DUAL_WEIGHT_RATIO
    end
    sys = (A = A, At = sparse(A'), tail = tail, n_tail = n_tail, tail_qr = qr(A * tail),
           V = ntuple(d -> modes[d].V, D),                # grid to modes, as a right factor
           # modes to grid, as a right factor
           Vt = ntuple(d -> Matrix(transpose(modes[d].V)), D),
           w_inv = _dual_inverse_weights(w, ratio), n_int = n_int)
    K_fac = _dual_factorize(sys, F)
    if K_fac === nothing
        return zeros(F, n), F(Inf)
    end

    c = _dual_onto_data(sys, K_fac, zeros(F, n), f)       # the surrogate fit
    iterations = 0
    resid = zero(F)
    # in 1D the surrogate is the true penalty
    if D > 1
        grams = ntuple(d -> modes[d].G, D)
        c, iterations, resid = _dual_refine(sys, K_fac, c, terms, grams, verbose)
        c = _dual_onto_data(sys, K_fac, c, f)             # back on the data to rounding
    end
    viol = maximum(abs, A * c .- f)
    if !isfinite(viol)
        viol = F(Inf)
    end
    verbose && println("  fit_scattered dual: $(Np - n_tail) dual unknowns for $n grid " *
                       "unknowns, $iterations iterations, relative residual $resid, " *
                       "constraint violation $viol")
    if resid > F(_DUAL_CG_LOOSE)
        @warn "fit_scattered (:exact, solver = :dual): the iteration on the roughness " *
              "measure stopped at a relative residual of $resid. The result reproduces " *
              "the data, but it is not the smoothest such grid; solver = :cholesky finds " *
              "that one."
    end
    return c, viol
end
