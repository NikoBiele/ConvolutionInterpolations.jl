"""
    fit_scattered(points, values; kwargs...)

Fit scattered data with a grid-constrained convolution interpolant.

The tensor-product kernel basis lives on a uniform grid covering the data's
bounding box; the grid coefficients are determined by requiring the interpolant
to pass through the scattered data (exactly, or in the least-squares sense),
selecting among admissible solutions by minimizing a q-th order roughness
seminorm. Ghost coefficients are eliminated before the solve using the
package's polynomial boundary condition, so the solution is a completed
grid consistent with `bc = :poly`.

Because the kernels are cardinal on the grid, the fitted interior coefficients
are the interpolant's values at the interior knots. `fit_scattered` therefore
returns exactly the `(knots, values)` pair the gridded constructors consume.

# Arguments
- `points::AbstractMatrix`: `D x Np` matrix; each column is one data point.
- `values::AbstractVector`: `Np` data values.

# Keyword Arguments
- `kernel::Union{Symbol,NTuple{D,Symbol}}=:auto`: any kernel with a polynomial
  boundary-condition table (`:a3`, `:a4`, `:a5`, `:a7`, `:b5`...`:b13`).
  `:auto` uses the scattered-path default: `:b5` for 1-3D (measured to be
  both the fastest and the deepest-converging choice with the default
  solver), `:a4` for 4-5D, `:a3` above. Note this differs from the gridded
  default (`:b7` for 1-2D): for scattered data the boundary-condition tables
  cap all b-kernels at degree-7 reproduction, and `:b5`'s shallower ghost
  extrapolation gives it the lowest roundoff floor under `:cholesky`.
- `mode::Symbol=:exact`: `:exact` sizes the grid finer than the data so the
  interpolation constraints are feasible and enforced to machine precision.
  `:lsq` sizes the grid at/below the data density; the constraints are then
  enforced in the least-squares sense, with residuals at the grid's
  representability level.
- `gridsize::Union{Symbol,Int,NTuple{D,Int}}=:auto`: interior knots per
  dimension. `:auto` derives it from `mode`, `oversample`, and the data count.
- `oversample::Real=0.0`: grid knots per data spacing when `gridsize=:auto`.
  `0.0` selects the mode default (1.6 for `:exact`, 0.8 for `:lsq`).
- `q::Union{Int,Symbol}=:auto`: roughness (difference) order of the selection
  seminorm. `:auto` uses the boundary-condition stencil width minus one
  (5 for `:b5`, 7 for `:b7`-class tables), which preserves the kernel's
  convergence order in data gaps.
- `solver::Symbol=:cholesky`: `:cholesky` (default) factors the
  Jacobi-equilibrated SPD normal system with CHOLMOD: fast, and with the
  default `:b5` kernel it is truncation-limited (no visible roundoff floor
  down to ~3e-11 in 2D benchmarks). `:qr` solves each multiplier iteration
  as a sparse least-squares problem via SPQR without forming normal
  equations; it is several times slower but lowers the roundoff floor of the
  wide kernels (measured 2D floors: :b7 ~5e-12 and still converging,
  :b9 ~1e-11, :b11 ~6e-10, :b13 ~3e-9). Use `solver=:qr, kernel=:b7` when
  chasing the deepest accuracy. The `:nullspace` solver solves for the
  coefficients exactly, but is only feasible on small meshes (few points).
- `W::Real=1e6`: augmented-Lagrangian penalty weight.
- `iters::Int=20`, `tol::Real=1e-11`: multiplier iteration controls. In the
  `:exact` regime convergence typically takes 1-2 iterations.
- `verbose::Bool=false`: print constraint violation per iteration.

# Returns
`(knots, vals, viol)`: the tuple of interior knot ranges, the array of fitted
interior grid values, and the final maximum constraint violation. Construct an
interpolant with `FastConvolutionInterpolation(knots, vals; bc=:poly, ...)`,
or use `convolution_interpolation(points, values)` which does this directly.

Notes: the solve runs on a sparse SPD system (CHOLMOD for `Float64`, dense
Cholesky otherwise). Exact interpolation requires distinct data points with
consistent values; duplicated points with conflicting values are resolved in
the least-squares sense and reported through `viol`.
"""
function fit_scattered(points::AbstractMatrix, values::AbstractVector;
                       kernel::Union{Symbol,Tuple{Vararg{Symbol}}}=:auto,
                       mode::Symbol=:exact,
                       gridsize::Union{Symbol,Int,Tuple{Vararg{Int}}}=:auto,
                       oversample::Real=0.0,
                       q::Union{Int,Symbol}=:auto,
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
    mode in (:exact, :lsq) ||
        throw(ArgumentError("mode must be :exact or :lsq, got $mode."))
    Np >= D ||
        throw(ArgumentError("points is $(D)x$(Np): fewer points than dimensions. " *
            "Points are COLUMNS (D x Np); for 1D scattered data use " *
            "reshape(x, 1, :). For gridded/nonuniform data at known knots, " *
            "pass knots as a Vector, not a Matrix."))

    F = float(promote_type(eltype(points), eltype(values)))
    # Solve eltype: Float16/Float32 are promoted to Float64 internally: the
    # penalty-weighted system spans ~W in scale, which single precision cannot
    # factor reliably (and the result's accuracy is data-limited anyway).
    # The fitted grid is cast back to F. BigFloat and friends solve natively
    # via the dense generic path.
    Fs = F <: Union{Float16,Float32} ? Float64 : F
    pts = Fs.(points)
    f   = Fs.(values)

    # ---- kernels, boundary-condition tables, ghost counts -------------------
    kernel = kernel === :auto ? _default_scattered_kernel(D) : kernel
    kernels = kernel isa Symbol ? ntuple(_ -> kernel, D) :
              length(kernel) == D ? NTuple{D,Symbol}(kernel) :
              throw(ArgumentError("kernel must be a Symbol or an NTuple{$D,Symbol}."))
    for k in kernels
        haskey(POLYNOMIAL_GHOST_COEFFS, k) && k !== :n3 ||
            throw(ArgumentError("fit_scattered requires a kernel with a polynomial " *
                                "boundary-condition table; :$k is not supported."))
    end
    eqs = ntuple(d -> get_equations_for_degree(kernels[d]), D)
    ng  = ntuple(d -> eqs[d] - 1, D)                       # ghosts per side
    G   = ntuple(d -> get_polynomial_ghost_coeffs(:poly, kernels[d]), D)
    ns  = ntuple(d -> size(G[d], 2), D)                    # BC stencil width

    # ---- grid over the bounding box -----------------------------------------
    lo = ntuple(d -> minimum(view(pts, d, :)), D)
    hi = ntuple(d -> maximum(view(pts, d, :)), D)
    for d in 1:D
        hi[d] > lo[d] ||
            throw(ArgumentError("data has zero extent in dimension $d."))
    end

    os = oversample > 0 ? Fs(oversample) : (mode === :exact ? Fs(1.6) : Fs(0.8))
    n_int = gridsize isa Int ? ntuple(_ -> gridsize, D) :
            gridsize isa Tuple ? (length(gridsize) == D ? NTuple{D,Int}(gridsize) :
                throw(ArgumentError("gridsize must have $D entries."))) :
            ntuple(d -> max(ns[d], ceil(Int, os * Np^(1 / D)) + 1), D)
    for d in 1:D
        n_int[d] >= ns[d] ||
            throw(ArgumentError("gridsize[$d] = $(n_int[d]) is below the boundary-" *
                                "condition stencil width $(ns[d]) for :$(kernels[d])."))
    end
    knots = ntuple(d -> range(lo[d], hi[d], length = n_int[d]), D)
    h     = ntuple(d -> Fs(step(knots[d])), D)
    n_full  = ntuple(d -> n_int[d] + 2ng[d], D)
    x0_full = ntuple(d -> Fs(lo[d]) - ng[d] * h[d], D)

    # ---- sparse system -------------------------------------------------------
    A = _scattered_collocation(pts, kernels, eqs, n_full, x0_full, h)
    S = _scattered_extension(n_full, n_int, ng, ns, G, Fs)
    qv = ntuple(d -> q === :auto ? ns[d] - 1 : q::Int, D)
    R = _scattered_roughness(n_full, qv, Fs)

    A_eff = A * S
    R_eff = R * S

    c_int, viol = _scattered_auglag(A_eff, f, R_eff, Fs(W), iters, Fs(tol),
                                    verbose, Val(solver))

    if mode === :exact && viol > max(Fs(tol), sqrt(eps(Fs))) * max(one(Fs), maximum(abs, f))
        @warn "fit_scattered (:exact): constraint violation $viol did not reach " *
              "machine precision. The system may be overdetermined for this grid " *
              "(try a larger gridsize/oversample) or the data may contain " *
              "duplicated points with conflicting values. Result is the " *
              "least-squares fit."
    end

    vals_grid = F === Fs ? reshape(c_int, n_int) : F.(reshape(c_int, n_int))
    return knots, vals_grid, F(viol)
end

"""
    convolution_interpolation(points::AbstractMatrix, values::AbstractVector; kwargs...)

Scattered-data convolution interpolation. Each column of `points` is one data
point (`D x Np`); `values` holds the corresponding data values.

Fits a grid-constrained convolution interpolant with `fit_scattered` and wraps
it in the standard fast interpolant, so evaluation, derivatives, and integrals
work exactly as for gridded data. All `fit_scattered` keyword arguments are
accepted, plus `derivative`, which is forwarded to `FastConvolutionInterpolation`.
The deprecated `precompute` and `subgrid` keywords are still accepted and forwarded,
but have no effect. The boundary condition is fixed to `:poly` for consistency
with the fit.

```julia
pts = rand(2, 500)                      # 500 scattered points in 2D
vals = [sin(3p[1]) * cos(2p[2]) for p in eachcol(pts)]
itp = convolution_interpolation(pts, vals)
itp(0.4, 0.6)
```
"""
function convolution_interpolation(points::AbstractMatrix, values::AbstractVector;
                                   kernel::Union{Symbol,Tuple{Vararg{Symbol}}}=:auto,
                                   mode::Symbol=:exact,
                                   gridsize::Union{Symbol,Int,Tuple{Vararg{Int}}}=:auto,
                                   oversample::Real=0.0,
                                   q::Union{Int,Symbol}=:auto,
                                   solver::Symbol=:cholesky,
                                   W::Real=1e6, iters::Int=20, tol::Real=1e-11,
                                   verbose::Bool=false,
                                   precompute=nothing,
                                   derivative::Union{Int,Tuple{Vararg{Int}}}=0,
                                   subgrid=nothing,
                                   extrap::Union{Symbol,AbstractExtrapolation}=Throw())
    D = size(points, 1)
    kernel_res = kernel === :auto ? _default_scattered_kernel(D) : kernel
    knots, vals, _ = fit_scattered(points, values;
                                   kernel = kernel_res, mode, gridsize,
                                   oversample, q, solver, W, iters, tol, verbose)
    itp = FastConvolutionInterpolation(knots, vals;
                                       kernel = kernel_res, bc = :poly,
                                       precompute, derivative, subgrid)
    return ConvolutionExtrapolation(itp, _extrap_type(extrap))
end

# =============================================================================
# internals
# =============================================================================

"""Scattered-path default kernel: `:b5` for 1-3D (fastest and, with the
default `:cholesky` solver, also the deepest-converging in benchmarks),
`:a4` for 4-5D, `:a3` above."""
_default_scattered_kernel(D::Int) = D <= 3 ? :b5 : D <= 5 ? :a4 : :a3

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
    fac = _scattered_spd_factorize(Hs)
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