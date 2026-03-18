"""
    lazy_ghost_value(values::AbstractArray{T,N}, virtual_idx::NTuple{N,Int},
                     eqs::Int, kernel_type::Symbol;
                     kernel_bc::Union{Symbol,Vector{Tuple{Symbol,Symbol}}}=:auto) where {T,N}

Compute the coefficient value at a virtual (expanded-array) index on the fly,
without materializing the expanded array.

Virtual indexing per dimension d (1-based Julia indexing):
    [1 .. ng]                    = left ghost points
    [ng+1 .. ng+size(d)]         = interior (raw data)
    [ng+size(d)+1 .. end]        = right ghost points

where `ng = eqs - 1`.

Processes dimensions sequentially (1, 2, ..., N), matching the eager
`apply_boundary_conditions_for_dim!` ordering exactly. Ghost indices are
resolved into weighted sums of interior points (polynomial path) or
evaluated as scalars (recursive path) — no recursion in either case.

For corners (ghost in multiple dimensions), the sequential processing
produces identical results to the eager materialized fill: dims 1..d-1
are fully resolved before dim d is processed.

## Resolution methods:
- **Polynomial path** (default for grids with sufficient points):
  Expands ghost into a deterministic weighted sum over interior points
  using `POLYNOMIAL_GHOST_COEFFS`. Fully decomposed into weights.

- **Recursive path** (fallback for small grids):
  Evaluates the 1D slice concretely (with earlier dims already resolved),
  computes ghost value as a scalar via `get_recursive_coefs`.
  The scalar is accumulated separately from the weight/index pairs.

Both methods apply mean-centering matching the eager code exactly.

See also: [`is_boundary_stencil`](@ref), [`POLYNOMIAL_GHOST_COEFFS`](@ref).
"""
function lazy_ghost_value(values::AbstractArray{T,N}, virtual_idx::NTuple{N,Int},
                          eqs::Int, kernel_type::Symbol;
                          kernel_bc::Union{Symbol,Vector{Tuple{Symbol,Symbol}}}=:auto) where {T,N}
    ng = eqs - 1
    factor_nd = N == 1 ? one(T) : -one(T)

    # Fast path: all interior — direct access
    all_interior = true
    @inbounds for d in 1:N
        vid = virtual_idx[d]
        if vid <= ng || vid > ng + size(values, d)
            all_interior = false
            break
        end
    end
    if all_interior
        raw_idx = ntuple(d -> virtual_idx[d] - ng, Val(N))
        return @inbounds values[raw_idx...]
    end

    ghost_matrix = get_polynomial_ghost_coeffs(kernel_type)
    n_stencil = size(ghost_matrix, 2)

    # Track weighted index pairs (polynomial) and scalar accumulator (recursive)
    weights = T[one(T)]
    indices = NTuple{N,Int}[virtual_idx]
    scalar_accum = zero(T)

    for d in 1:N
        n_d = size(values, d)
        bc_pair = kernel_bc isa Symbol ? (kernel_bc, kernel_bc) : kernel_bc[d]

        new_weights = T[]
        new_indices = NTuple{N,Int}[]

        for (w, ix) in zip(weights, indices)
            vid = ix[d]

            if vid > ng && vid <= ng + n_d
                # Interior in this dim — pass through
                push!(new_weights, w)
                push!(new_indices, ix)
                continue
            end

            # Ghost in this dimension
            ghost_side = vid <= ng ? :left : :right
            ghost_layer = ghost_side === :left ? (ng + 1 - vid) : (vid - ng - n_d)
            bc_side = ghost_side === :left ? bc_pair[1] : bc_pair[2]
            use_polynomial = (bc_side == :poly) || (bc_side == :auto && n_d >= n_stencil)
            factor = ghost_side === :right ? factor_nd : one(T)

            if use_polynomial
                # ── Polynomial path: expand into weights ──
                # ghost = y_mean + factor * Σ coef[m] * (stencil[m] - y_mean)
                #       = (1 - factor * Σcoef) * y_mean + factor * Σ coef[m] * stencil[m]
                # where y_mean = (1/n_d) * Σ interior[k]
                ns = min(n_stencil, n_d)
                coef_sum = zero(T)
                for m in 1:ns
                    coef_sum += ghost_matrix[ghost_layer, m]
                end
                mean_weight = one(T) - factor * coef_sum

                # Mean contribution distributed over all interior points
                for k in 1:n_d
                    nv = ntuple(dd -> dd == d ? ng + k : ix[dd], Val(N))
                    push!(new_weights, w * mean_weight / n_d)
                    push!(new_indices, nv)
                end

                # Stencil contributions
                for m in 1:ns
                    nv = if ghost_side === :left
                        ntuple(dd -> dd == d ? ng + m : ix[dd], Val(N))
                    elseif N == 1
                        # 1D right: reversed order (matches eager reverse(vs))
                        ntuple(dd -> dd == d ? ng + n_d + 1 - m : ix[dd], Val(N))
                    else
                        # ND right: natural order (matches eager double-reversal)
                        ntuple(dd -> dd == d ? ng + m : ix[dd], Val(N))
                    end
                    push!(new_weights, w * factor * ghost_matrix[ghost_layer, m])
                    push!(new_indices, nv)
                end

            else
                # ── Recursive path: evaluate slice concretely ──
                # Dims 1..d-1 are already resolved to interior indices.
                # Dims d+1..N may still be ghost — eager reads 0 there (unfilled).
                y_slice = Vector{T}(undef, n_d)
                for k in 1:n_d
                    nv = ntuple(dd -> dd == d ? ng + k : ix[dd], Val(N))
                    all_ok = true
                    for dd in (d+1):N
                        if nv[dd] <= ng || nv[dd] > ng + size(values, dd)
                            all_ok = false
                            break
                        end
                    end
                    if all_ok
                        raw = ntuple(dd -> nv[dd] - ng, Val(N))
                        y_slice[k] = @inbounds values[raw...]
                    else
                        y_slice[k] = zero(T)  # matches eager: unfilled ghost slots = 0
                    end
                end

                y_mean = sum(y_slice) / n_d
                y_centered = y_slice .- y_mean

                if ghost_side === :right && N == 1
                    reverse!(y_centered)
                end

                h_d = one(T)
                coef = get_recursive_coefs(y_centered, h_d,
                           bc_side == :auto ? :auto : bc_side, :left)

                y_ext = Vector{T}(undef, n_d + ng)
                y_ext[ng+1:ng+n_d] .= y_centered
                for j in 1:ghost_layer
                    y_ext[ng + 1 - j] = sum(coef[kk] * y_ext[ng + 1 - j + kk]
                                            for kk in 1:length(coef))
                end

                ghost_val = y_mean + factor * y_ext[ng + 1 - ghost_layer]
                scalar_accum += w * ghost_val
            end
        end

        weights = new_weights
        indices = new_indices
    end

    # Sum: scalar contributions (recursive) + weighted raw values (polynomial)
    result = scalar_accum
    @inbounds for (w, ix) in zip(weights, indices)
        raw = ntuple(d -> ix[d] - ng, Val(N))
        result += w * values[raw...]
    end
    return result
end

"""
    is_boundary_stencil(cell::Int, n::Int, eqs::Int) -> Bool

Check if cell index `cell` (1-based, in raw data coordinates) has its
evaluation stencil `[cell-(eqs-1) .. cell+eqs]` reaching outside `[1, n]`.

Used to dispatch between the fast interior path (direct array access)
and the slow boundary path (lazy ghost resolution).
"""
@inline function is_boundary_stencil(cell::Int, n::Int, eqs::Int)
    return (cell - (eqs - 1) < 1) | (cell + eqs > n)
end

# Helper to extract kernel symbol from degree type parameter (zero-cost dispatch)
@inline _kernel_sym(::Val{S}) where S = S
@inline _kernel_sym(::HigherOrderKernel{S}) where S = S
