# =============================================================================
# mode = :exact, solver = :dual: one unknown per data point
# =============================================================================
#
# The sparse solvers of fit_exact.jl factorize a matrix in the grid unknowns. Its fill grows
# quickly with the dimension, and from 4D the factorization needs more memory than the grid
# itself by orders of magnitude. This solver works in the data points instead. It minimises
#
#     c1' M c1   subject to   A (c1 + T b) = f
#
# over the interior grid values c = c1 + T b, where A is the collocation matrix on the interior
# unknowns, M is the q-th power of the reflecting (Neumann) grid Laplacian, and the columns of
# T are the polynomials of total degree below q on the grid (the tail). M is the isotropic H^q
# seminorm of the sparse solvers with its boundary rows replaced by reflection; the tail gives
# it back the seminorm's null space, which the reflecting Laplacian alone does not have.
#
# M is diagonal in the cosine basis of the grid, a tensor product of one small matrix per axis,
# so its inverse (pseudo-inverse: the constant mode is left to the tail) is applied by
# multiplications along the axes. With Z an orthonormal basis of the data vectors orthogonal
# to the tail at the data points, the solution is
#
#     (Z' A M⁻¹ A' Z) y = Z' f,    c1 = M⁻¹ A' Z y,    b from A T b = f - A c1.
#
# The dense system has one unknown per data point (less the tail). It is assembled in blocks
# of columns, each projected onto Z before the penalty weights are applied, which keeps the
# large weights of the smooth modes away from the subtraction that the projection performs.
# Nothing of size unknowns x points is stored: the memory is the dense system, 8 bytes times
# the square of the point count, plus work arrays bounded by _DUAL_WORK_BYTES.
#
# The weight of a cosine mode in M⁻¹ is its Laplacian eigenvalue to the power -q, so on a grid
# with many knots per axis the smoothest modes outweigh the roughest by more than double
# precision can carry, and the data would be missed. The eigenvalues are therefore shifted by
# the smallest amount that keeps the ratio of any two weights below _DUAL_WEIGHT_RATIO
# (_dual_shift): nothing on the coarse grids of four and more dimensions, a few hundredths on
# fine grids, where it changes the penalty only on modes far smoother than the data resolve.
#
# The result reproduces the data to rounding, as the sparse solvers do. Away from the faces of
# the box it agrees with them; near the faces the reflecting boundary makes it less accurate
# once the data are dense enough to resolve the function there.

const _DUAL_WORK_BYTES   = 2^30   # memory of the grid-sized work arrays of the assembly, all tasks together
const _DUAL_WEIGHT_RATIO = 1e13   # largest ratio between the weights of two cosine modes in the inverse penalty

"""Orthonormal cosine basis of the reflecting Laplacian on `n` knots (columns of `V`) and its
eigenvalues `lam`, in units of the knot spacing; `lam[1] == 0` belongs to the constant."""
function _dual_cosine_basis(n::Int, ::Type{F}) where {F}
    V = Matrix{F}(undef, n, n)
    lam = Vector{F}(undef, n)
    for k in 0:n-1
        if k == 0
            scale = sqrt(one(F) / n)
        else
            scale = sqrt(F(2) / n)
        end
        for j in 1:n
            V[j, k + 1] = scale * cos(F(pi) * k * (2 * j - 1) / (2 * n))
        end
        lam[k + 1] = 2 - 2 * cos(F(pi) * k / n)
    end
    return V, lam
end

"""Multiply each column of `X` (unknowns x columns), viewed on the grid `n_int`, along every
grid axis `d` by `mats[d]` from the right: entry `i` along the axis becomes
`sum(x[j] * mats[d][j, i] for j)`. `X` and `work` (same size) are both overwritten; the result
is returned and is one of the two."""
function _dual_mul_axes!(X::Matrix{F}, work::Matrix{F}, mats::NTuple{D,Matrix{F}},
                         n_int::NTuple{D,Int}) where {F,D}
    src = X
    dst = work
    for d in 1:D
        left = 1                                           # entries per slice ahead of axis d
        for k in 1:d-1
            left *= n_int[k]
        end
        right = div(length(src), left * n_int[d])          # slices: the later axes and the columns
        S = reshape(src, left, n_int[d], right)
        T = reshape(dst, left, n_int[d], right)
        for r in 1:right
            mul!(view(T, :, :, r), view(S, :, :, r), mats[d])
        end
        src, dst = dst, src
    end
    return src
end

"""Inverse of the penalty applied to each column of `X` (unknowns x columns): to the cosine
basis with `V`, times the weights `w`, back to the grid with `Vt`. `X` and `work` are both
overwritten; the result is returned and is one of the two."""
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

"""Shift of the Laplacian eigenvalues in the inverse penalty: the smallest one for which the
weights `1 / (lam + shift)^q` of the modes between `lam_min` (the smallest nonzero eigenvalue)
and `lam_max` differ by no more than the factor `_DUAL_WEIGHT_RATIO`; zero where they already
do."""
function _dual_shift(lam_min::F, lam_max::F, q::Int) where {F}
    eigenvalue_ratio = F(_DUAL_WEIGHT_RATIO)^(one(F) / q)
    return max(zero(F), lam_max / eigenvalue_ratio - lam_min)
end

"""Weights of the inverse penalty per cosine mode: `1 / (lam + shift)^q`. The constant mode
(`lam == 0`) gets weight zero: the polynomial tail carries the constants."""
function _dual_weights(lam::Vector{F}, q::Int, shift::F) where {F}
    w = similar(lam)
    for i in eachindex(lam)
        if lam[i] > 0
            w[i] = one(F) / (lam[i] + shift)^q
        else
            w[i] = zero(F)
        end
    end
    return w
end

"""The polynomial tail on the grid `n_int` (unknowns x polynomials): all monomials of total
degree below `q`, in knot coordinates scaled to [-1, 1] per axis to keep the columns well
conditioned."""
function _dual_polynomials(n_int::NTuple{D,Int}, q::Int, ::Type{F}) where {F,D}
    t = ntuple(d -> [F(2 * j - n_int[d] - 1) / F(n_int[d] - 1) for j in 1:n_int[d]], D)
    exponents = NTuple{D,Int}[]
    for degree in 0:q-1
        append!(exponents, _multi_indices(degree, D))
    end
    tail = Matrix{F}(undef, prod(n_int), length(exponents))
    for (col, e) in enumerate(exponents)
        for (row, J) in enumerate(CartesianIndices(n_int))
            value = one(F)
            for d in 1:D
                value *= t[d][J[d]]^e[d]
            end
            tail[row, col] = value
        end
    end
    return tail
end

"""Columns `cols` of the dense system `Z' A M⁻¹ A' Z` into `K`, `block` columns at a time.
`sys` holds the operators of `_scattered_dual`; `w` are the weights of the inverse penalty."""
function _dual_assemble_columns!(K::Matrix{F}, cols::UnitRange{Int}, block::Int, sys,
                                 w::Vector{F}) where {F}
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
        E = zeros(F, Np, length(J))                        # unit vectors that select the columns J of Z
        for (k, j) in enumerate(J)
            E[sys.n_tail + j, k] = one(F)
        end
        mul!(X, sys.At, sys.tail_qr.Q * E)                 # A' Z on the grid, for these columns
        Y = _dual_inverse_penalty!(X, work, w, sys.V, sys.Vt, sys.n_int)
        K[:, J] = view(sys.tail_qr.Q' * (sys.A * Y), sys.n_tail+1:Np, :)   # Z' A applied to the result
        j0 += block
    end
    return K
end

"""Solve the dual system for the weights `w` of the inverse penalty: the interior grid values
and the largest data misfit, `Inf` if the dense system is singular."""
function _dual_solve(sys, f::Vector{F}, w::Vector{F}) where {F}
    Np, n = size(sys.A)
    n_free = Np - sys.n_tail                               # unknowns of the dense system
    K = Matrix{F}(undef, n_free, n_free)
    ntasks = min(Threads.nthreads(), n_free)
    block = max(1, min(cld(n_free, ntasks), div(_DUAL_WORK_BYTES, 2 * sizeof(F) * n * ntasks)))
    if ntasks == 1
        _dual_assemble_columns!(K, 1:n_free, block, sys, w)
    else
        per_task = cld(n_free, ntasks)
        blas_threads = BLAS.get_num_threads()
        BLAS.set_num_threads(1)                            # the tasks use the cores, not BLAS inside them
        try
            Threads.@threads for t in 1:ntasks
                cols = ((t - 1) * per_task + 1):min(t * per_task, n_free)
                _dual_assemble_columns!(K, cols, block, sys, w)
            end
        finally
            BLAS.set_num_threads(blas_threads)
        end
    end
    for j in 1:n_free, i in 1:j-1                          # symmetric up to rounding: make it exactly so
        mean_ij = (K[i, j] + K[j, i]) / 2
        K[i, j] = mean_ij
        K[j, i] = mean_ij
    end
    K_fac = lu!(K; check = false)
    if !issuccess(K_fac)
        return zeros(F, n), F(Inf)
    end

    y = K_fac \ (sys.tail_qr.Q' * f)[sys.n_tail+1:Np]
    dual = sys.tail_qr.Q * vcat(zeros(F, sys.n_tail), y)   # Z y: one multiplier per data point
    X = reshape(sys.At * dual, n, 1)
    c1 = vec(_dual_inverse_penalty!(X, similar(X), w, sys.V, sys.Vt, sys.n_int))
    b = sys.tail_qr \ (f .- sys.A * c1)                    # the tail carries what c1 leaves of the data
    c = c1 .+ sys.tail * b
    viol = maximum(abs, sys.A * c .- f)
    if !isfinite(viol)
        viol = F(Inf)
    end
    return c, viol
end

"""Exact fit on a uniform grid by the dual system: the interior grid values that reproduce
`f` at `pts` with the least roughness of order `q`, measured with reflecting boundaries, and
the largest data misfit (`Inf` if the dense system is singular)."""
function _scattered_dual(pts::AbstractMatrix{F}, f::Vector{F}, kernels::NTuple{D,Symbol},
                         axes::NTuple{D,<:AbstractRange}, q::Int, verbose::Bool) where {F,D}
    A_full, S, _ = _scattered_uniform_parts(pts, kernels, axes)
    A = A_full * S                                         # collocation on the interior unknowns
    Np = size(A, 1)
    n_int = ntuple(d -> length(axes[d]), D)

    tail = _dual_polynomials(n_int, q, F)
    n_tail = size(tail, 2)
    Np > n_tail ||
        throw(ArgumentError("solver = :dual needs more than $n_tail data points in $(D)D with " *
                            "q = $q (the polynomials the roughness measure does not see); " *
                            "got $Np. Use solver = :cholesky, or a lower q."))

    bases = ntuple(d -> _dual_cosine_basis(n_int[d], F), D)
    V  = ntuple(d -> bases[d][1], D)                       # grid to cosine modes, as a right factor
    Vt = ntuple(d -> Matrix(transpose(bases[d][1])), D)    # cosine modes to grid, as a right factor
    lam = vec([sum(bases[d][2][J[d]] for d in 1:D) for J in CartesianIndices(n_int)])
    lam_min = minimum(bases[d][2][2] for d in 1:D)         # smallest nonzero eigenvalue of the grid
    lam_max = sum(bases[d][2][end] for d in 1:D)           # largest eigenvalue of the grid
    shift = _dual_shift(lam_min, lam_max, q)

    sys = (A = A, At = sparse(A'), tail = tail, n_tail = n_tail, tail_qr = qr(A * tail),
           V = V, Vt = Vt, n_int = n_int)
    c, viol = _dual_solve(sys, f, _dual_weights(lam, q, shift))
    verbose && println("  fit_scattered dual: $(Np - n_tail) dual unknowns for $(size(A, 2)) grid " *
                       "unknowns, eigenvalue shift $shift, constraint violation $viol")
    return c, viol
end
