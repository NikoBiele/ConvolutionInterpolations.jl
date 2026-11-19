# =============================================================================
# mode = :exact: the sparse solvers (:cholesky and :qr)
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

# largest sparse-QR fallback of :cholesky, estimated as rows x unknowns x 8 bytes
const _EXACT_QR_FALLBACK_BYTES = 2^31

"""Cholesky-based multiplier iteration on the normal equations `W A'A + R'R`. Returns
`(c, viol)`."""
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
        # the normal matrix lost positive definiteness numerically (large grids with few
        # data rows). Sparse QR of the stacked least-squares system does not square the
        # condition number, but its memory grows as rows x unknowns: it is the fallback only
        # while that stays small, and an error otherwise
        qr_bytes = (size(A, 1) + size(R, 1)) * size(A, 2) * sizeof(F)
        if qr_bytes > _EXACT_QR_FALLBACK_BYTES
            error("fit_scattered (:exact): the normal equations are not numerically " *
                  "positive definite on this grid ($(size(A, 2)) unknowns for " *
                  "$(size(A, 1)) points), and solving by sparse QR instead would need " *
                  "about $(round(qr_bytes / 2^30; digits = 1)) GB. Use solver = :dual on " *
                  "a uniform grid, or a smaller gridsize/oversample; solver = :qr forces " *
                  "the sparse QR.")
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
