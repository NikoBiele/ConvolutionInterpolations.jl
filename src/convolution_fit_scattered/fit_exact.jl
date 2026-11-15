# =============================================================================
# mode = :exact: the sparse solvers (:cholesky, :qr and the deprecated :nullspace)
# =============================================================================

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

const _EXACT_QR_FALLBACK_BYTES = 2^31   # largest sparse-QR fallback of :cholesky, estimated as rows x unknowns x 8 bytes

"""Whether the CHOLMOD factorization of `Hs` is predicted to need more than `memory_limit`
bytes. The prediction is twice the size of the factor, which the symbolic analysis gives
before any numbers are computed; the other half is the matrix, CHOLMOD's copy of it and its
working space. A factor is never larger than a dense triangle, so systems for which even
that fits are not analysed. The factor size is read from CHOLMOD's own structures; where a
Julia version does not provide them, the answer is `false` and the fit proceeds unchecked."""
function _scattered_cholesky_exceeds(Hs::SparseMatrixCSC{Float64}, memory_limit::Real)
    n = size(Hs, 1)
    if 8.0 * n * n <= memory_limit
        return false
    end
    try
        analysis = SparseArrays.CHOLMOD.symbolic(SparseArrays.CHOLMOD.Sparse(Symmetric(Hs)))
        factor_bytes = 8.0 * unsafe_load(pointer(analysis)).xsize
        return 2 * factor_bytes > memory_limit
    catch err
        err isa InterruptException && rethrow()
        return false
    end
end
_scattered_cholesky_exceeds(Hs::SparseMatrixCSC, memory_limit::Real) = false

"""Cholesky-based multiplier iteration on the normal equations `W A'A + R'R`. Returns
`(c, viol)`, or `nothing` when the factorization is predicted to need more than
`memory_limit` bytes (`_scattered_cholesky_exceeds`); the default sets no limit."""
function _scattered_auglag(A::SparseMatrixCSC{F}, f::Vector{F},
                           R::SparseMatrixCSC{F}, W::F, iters::Int, tol::F,
                           verbose::Bool, ::Val{:cholesky};
                           memory_limit::Real = Inf) where {F}
    H = W * (A' * A) + R' * R
    # Jacobi equilibration: boundary-adjacent columns carry the (large)
    # polynomial-extrapolation weights of the deep ghosts, so column norms
    # span many orders of magnitude for wide kernels (b9-b13). Symmetric
    # diagonal scaling D H D with D = diag(1 ./ sqrt.(diag(H))) restores
    # balanced scaling, preserves SPD, and lowers the roundoff error floor.
    d = [one(F) / sqrt(max(H[i, i], eps(F))) for i in 1:size(H, 1)]
    Dm = spdiagm(0 => d)
    Hs = Dm * H * Dm
    if _scattered_cholesky_exceeds(Hs, memory_limit)
        return nothing
    end
    fac = try
        _scattered_spd_factorize(Hs)
    catch err
        err isa PosDefException || rethrow()
        # the normal matrix lost positive definiteness numerically (large grids with few data
        # rows). Sparse QR of the stacked least-squares system does not square the condition
        # number, but its memory grows as rows x unknowns: it is the fallback only while that
        # stays small, and an error otherwise
        qr_bytes = (size(A, 1) + size(R, 1)) * size(A, 2) * sizeof(F)
        if qr_bytes > _EXACT_QR_FALLBACK_BYTES
            error("fit_scattered (:exact): the normal equations are not numerically positive " *
                  "definite on this grid ($(size(A, 2)) unknowns for $(size(A, 1)) points), and " *
                  "solving by sparse QR instead would need about " *
                  "$(round(qr_bytes / 2^30; digits = 1)) GB. Use solver = :dual on a uniform " *
                  "grid, or a smaller gridsize/oversample; solver = :qr forces the sparse QR.")
        end
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
