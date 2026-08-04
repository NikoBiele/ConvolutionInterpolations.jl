"""
    BoundaryWorkspace{T,N}

Preallocated workspace for efficient boundary condition computation in N-dimensional arrays.

# Fields
- `ghost_vals::Vector{T}`: Temporary storage for computed ghost point values
- `y_temp::Vector{T}`: Temporary storage for reversed signal data
- `slice::Vector{T}`: Storage for extracted 1D slices from N-D arrays
- `slice_offset::Vector{CartesianIndex{N}}`: Precomputed offsets for slice extraction
- `c_offset::Vector{CartesianIndex{N}}`: Precomputed offsets for ghost point placement

# Details
This workspace structure eliminates allocations during boundary condition application by
preallocating all necessary temporary arrays. It is reused across all boundary dimensions
and all boundary points, making the boundary coefficient computation allocation-free.

See also: `BoundaryWorkspace(T, N, max_eqs, max_dim_size)`.
"""
# Workspace for boundary condition computation
struct BoundaryWorkspace{T,N}
    ghost_vals::Vector{T}
    y_temp::Vector{T}
    slice::Vector{T}
    slice_offset::Vector{CartesianIndex{N}}
    c_offset::Vector{CartesianIndex{N}}
end

"""
    BoundaryWorkspace(T::Type, N::Int, max_eqs::Int, max_dim_size::Int)

Construct a preallocated workspace for boundary condition computation.

# Arguments
- `T::Type`: Element type for arrays (typically `Float64` or `Float32`)
- `N::Int`: Number of dimensions in the array
- `max_eqs::Int`: Maximum number of equations (kernel support size)
- `max_dim_size::Int`: Maximum size along any dimension

# Returns
`BoundaryWorkspace{T,N}` with appropriately sized preallocated arrays

# Details
Creates workspace arrays sized to handle the worst-case scenario across all dimensions,
enabling allocation-free boundary computation throughout the interpolation setup.
"""

function BoundaryWorkspace(T::Type, ::Val{N}, max_eqs::Int, max_dim_size::Int) where N
    BoundaryWorkspace{T,N}(
        zeros(T, max_eqs - 1),
        zeros(T, max_dim_size),
        zeros(T, max_dim_size),
        Vector{CartesianIndex{N}}(undef, max_dim_size),
        Vector{CartesianIndex{N}}(undef, max_eqs - 1),
    )
end

"""
    get_boundary_indices(c_size::NTuple{N,Int}, dim::Int, eqs::Int) where N

Compute CartesianIndex ranges for left and right boundary positions in a given dimension.
Returns a tuple `(left_indices, right_indices)` of `CartesianIndices` objects.

Uses `ntuple` with `Val(N)` for compile-time specialization — zero allocations.
"""

function get_boundary_indices(c_size::NTuple{N,Int}, dim::Int, eqs::Int) where N
    left_ranges = ntuple(d -> d == dim ? ((eqs):(eqs)) : (1:c_size[d]), Val(N))
    right_ranges = ntuple(d -> d == dim ? ((c_size[d] - (eqs-1)):(c_size[d] - (eqs-1))) : (1:c_size[d]), Val(N))
    return CartesianIndices(left_ranges), CartesianIndices(right_ranges)
end

"""
    create_convolutional_coefs(vs::AbstractArray{T,N}, h::NTuple{N,T}, eqs::Int,
                               kernel_bc::Union{Symbol,Vector{Tuple{Symbol,Symbol}}}, 
                               kernel_type::Symbol) where {T,N}

Create coefficient array with ghost points computed from boundary conditions.

# Arguments
- `vs::AbstractArray{T,N}`: Input data values on the interior grid
- `h::NTuple{N,T}`: Grid spacing in each dimension
- `eqs::Int`: Number of equations (kernel support size)
- `kernel_bc`: Boundary condition specification
  - `Symbol`: Same boundary condition for all dimensions and both sides
  - `Vector{Tuple{Symbol,Symbol}}`: Per-dimension (left, right) boundary conditions
- `kernel_type::Symbol`: Kernel degree (`:a3`, `:b5`, etc.)

# Returns
For `:a0` and `:a1` kernels: Returns `vs` unchanged (no ghost points needed)

For higher-order kernels: Expanded coefficient array with dimensions `size(vs) .+ 2*(eqs-1)`,
where ghost points outside the original domain have been filled according to the specified
boundary conditions.

# Details
This is the main entry point for boundary condition application.

**For nearest neighbor (`:a0`) and linear (`:a1`) kernels:**
- No boundary processing is required since these kernels have minimal support
- Returns the input array directly for efficiency

**For higher-order kernels (`:a3`, `:a4`, `:b5`, etc.):**
1. Allocates an expanded array with space for ghost points
2. Copies interior values to the center
3. Applies boundary conditions dimension-by-dimension using a preallocated workspace
4. Returns the complete coefficient array ready for convolution

The dimension-by-dimension approach ensures proper corner and edge treatment in
multidimensional cases.

# Examples
```julia
# Linear interpolation - no ghost points needed
vs = rand(50, 50)
h = (0.1, 0.1)
coefs = create_convolutional_coefs(vs, h, 2, :detect, :a1)  # Returns vs unchanged

# Higher-order kernel with ghost points
coefs = create_convolutional_coefs(vs, h, 5, :poly, :b5)
# Returns expanded array of size (58, 58) with ghost points

# Different boundary conditions per dimension
bc = [(:poly, :poly), (:linear, :quadratic)]
coefs = create_convolutional_coefs(vs, h, 5, bc, :b5)
```

See also: `apply_boundary_conditions_for_dim!`, `boundary_coefs`.
"""

function create_convolutional_coefs(vs::AbstractArray{T,N}, h::NTuple{N,T}, 
                                    eqs::NTuple{N,Int},
                                    kernel_bc::NTuple{N,Tuple{Symbol,Symbol}}, 
                                    kernel_types::NTuple{N,Symbol},
                                    ::Val{UG}) where {T,N,UG}
    new_dims = ntuple(d -> size(vs, d) + 2*(eqs[d]-1), N)
    c = zeros(T, new_dims)
    inner_indices = ntuple(d -> (1+(eqs[d]-1)):(new_dims[d]-(eqs[d]-1)), N)
    c[inner_indices...] = vs

    max_eqs = maximum(eqs)
    max_dim_size = maximum(size(vs))
    workspace = BoundaryWorkspace(T, Val(N), max_eqs, max_dim_size)

    for fixed_dim in 1:N
        apply_boundary_conditions_for_dim!(c, vs, fixed_dim, h, eqs[fixed_dim], kernel_bc, 
                                           kernel_types[fixed_dim], workspace, size(vs), Val(UG))
    end
    return c
end
function create_convolutional_coefs(vs::AbstractArray{T,N}, h::NTuple{N,T}, eqs::Int,
                                    kernel_bc, kernel_type::Symbol, ::Val{UG}) where {T,N,UG}
    create_convolutional_coefs(vs, h, ntuple(_ -> eqs, N), kernel_bc,
                               ntuple(_ -> kernel_type, N), Val(UG))
end


## ===========================================================================
## Boundary-condition guard for the :detect path
##
## Rejects the polynomial ghost extension when the Delta^ns residual exceeds
## kappa times the local data range, where ns = size(G, 2) is the interior
## stencil width (4 for a-kernels, 6 for :b5, 8 for :b7/:b9/:b11/:b13).
##
## Delta^ns annihilates exactly the space the extension reproduces
## (degree <= ns-1), so it is the leading residual of the extension itself
## rather than a proxy for smoothness. This replaces a d1/d2/d3 sign-change
## heuristic.
##
## Ghost matrix convention: size(G) == (ng, ns); G[r, d] weights interior
## point d into ghost g_{-r}. Only the first eqs-1 rows are ever applied.
## The right boundary is traversed base = n_dim, step = -1, matching
## fill_ghost_points_polynomial!, which feeds y_centered[end - k + 1].
## ===========================================================================

const BC_GUARD_K = Ref(1.0) # carefully measured, 20k random signals, flat optimum

"""
Stencil traversal helper. Points are read as

    y[base], y[base+step], ..., y[base+step*(ns-1)]

Left boundary:  base = 1,     step = +1
Right boundary: base = n_dim, step = -1

Returns (lo, hi, hi-lo, max|y|) over the stencil.
"""
@inline function bc_stencil_range(y, base::Int, step::Int, ns::Int, ::Type{T}) where {T}
    lo = typemax(T); hi = typemin(T); amax = zero(T)
    @inbounds for d in 0:ns-1
        v = y[base + step*d]
        lo = min(lo, v); hi = max(hi, v); amax = max(amax, abs(v))
    end
    return lo, hi, hi - lo, amax
end


"""
Single entry point for the :detect decision.

    G          ghost matrix, (ng, ns) -- only size(G, 2) is used
    y          mean-centered slice
    base, step side selector: left is (1, +1), right is (n_dim, -1)
    n_avail    points available in the slice (= n_dim)
    num_ghost  ghost points actually filled (= eqs-1); unused by :diff
    hd         grid spacing in this dimension; :legacy only

Window width is k+3, giving three overlapping Delta^k windows: one on the
stencil the extension uses, two just inside it. Widening makes the test
stricter, since the verdict is a max over windows.
"""
@inline function bc_accept_polynomial(::Type{T}, G, y, base::Int, step::Int,
                                      n_avail::Int, num_ghost::Int,
                                      hd::T, workspace) where {T}
    k = size(G, 2)
    n_avail < k + 1 && return false
    return bc_guard_diff(T, y, base, step, min(n_avail, k + 3), k,
                         T(BC_GUARD_K[]))
end

const BC_W4 = (1.0, -4.0, 6.0, -4.0, 1.0)
const BC_W6 = (1.0, -6.0, 15.0, -20.0, 15.0, -6.0, 1.0)
const BC_W8 = (1.0, -8.0, 28.0, -56.0, 70.0, -56.0, 28.0, -8.0, 1.0)

@inline function _bc_diff_scan(::Type{T}, y, base::Int, step::Int, nwin::Int,
                               W::NTuple{M,Float64}, tol) where {T,M}
    k = M - 1
    Tacc = promote_type(T, Float64)
    @inbounds for start in 0:(nwin - k - 1)
        s = zero(Tacc)
        for d in 0:k
            s = muladd(Tacc(W[d+1]), Tacc(y[base + step*(start + k - d)]), s)
        end
        abs(s) > tol && return false
    end
    return true
end

@inline function _bc_diff_generic(::Type{T}, y, base::Int, step::Int,
                                  nwin::Int, k::Int, tol) where {T}
    Tacc = promote_type(T, Float64)
    @inbounds for start in 0:(nwin - k - 1)
        s = zero(Tacc); c = 1; sgn = 1
        for j in 0:k
            s += Tacc(sgn * c) * Tacc(y[base + step*(start + k - j)])
            c = div(c * (k - j), j + 1)
            sgn = -sgn
        end
        abs(s) > tol && return false
    end
    return true
end

"""
Difference test. Delta^k annihilates exactly the space the ghost extension
reproduces (degree < k), so |Delta^k y| is its leading residual. Accepted
when that residual stays below `kappa` times the local data range, over
every window of k+1 consecutive points in the stencil.

Weights for k = 4, 6, 8 are compile-time constants; other k falls back to a
binomial recurrence. Allocation-free on every path. Accumulation is in
>= Float64 so the test does not degrade at Float32 for wide stencils.
"""
@inline function bc_guard_diff(::Type{T}, y, base::Int, step::Int,
                               nwin::Int, k::Int, kappa::T) where {T}
    nwin < k + 1 && return false
    Tacc = promote_type(T, Float64)
    lo, hi, rng, amax = bc_stencil_range(y, base, step, nwin, T)
    tol = Tacc(kappa) * Tacc(rng) +
          Tacc(eps(T)) * Tacc(amax) * Tacc(1 << min(k, 30))
    k == 8 && return _bc_diff_scan(T, y, base, step, nwin, BC_W8, tol)
    k == 6 && return _bc_diff_scan(T, y, base, step, nwin, BC_W6, tol)
    k == 4 && return _bc_diff_scan(T, y, base, step, nwin, BC_W4, tol)
    return _bc_diff_generic(T, y, base, step, nwin, k, tol)
end


"""
    apply_boundary_conditions_for_dim!(...)

Apply boundary conditions along one dimension. Fills ghost points on both
ends.

Only the first (or last) `m = min(n_dim, max(ns+3, eqs))` samples along the
line are touched: `ns+3` is the widest window the guard reads, `eqs` covers
the Gaussian-fallback fit. The slice buffer holds those `m` values in
increasing index order on both sides, so the right-hand path indexes from
`end` exactly as before.

Mean-centering uses the mean of that window rather than of the whole line.
Ghost matrix rows sum to 1, so the choice of offset is mathematically
irrelevant and affects only roundoff; a local mean also removes the local
trend, which is marginally better conditioned.
"""
function apply_boundary_conditions_for_dim!(c::AbstractArray{T,N}, vs::AbstractArray, dim::Int,
                                            h::NTuple{N,T}, eqs::Int,
                                            kernel_bc::NTuple{N,Tuple{Symbol,Symbol}},
                                            kernel_type::Symbol,
                                            workspace::BoundaryWorkspace{T,N},
                                            vs_size::NTuple{N,Int}, ::Val{UG}) where {T,N,UG}

    if eqs == 1
        return  # :a0 and :a1 need no ghost points
    end

    kernel_boundary_condition = kernel_bc[dim]
    left_indices, right_indices = get_boundary_indices(size(c), dim, eqs)

    ghost_matrix = UG ? nothing : get_polynomial_ghost_coeffs(:not_used, kernel_type)
    n_dim        = size(vs, dim)
    n_interior   = UG ? eqs : size(ghost_matrix, 2)
    few_points   = n_dim < n_interior
    num_ghost    = eqs - 1
    hd           = h[dim]

    # Window actually read: guard needs n_interior+3, the Gaussian fit needs
    # up to eqs, the linear/quadratic matrices need <= 3.
    m = min(n_dim, max(n_interior + 3, eqs))

    # slice_offset[j] is a step of (j-1) along dim; only m are needed now.
    for j in 1:m
        workspace.slice_offset[j] = CartesianIndex(ntuple(d -> d == dim ? j-1 : 0, Val(N)))
    end
    for j in 1:(eqs-1)
        workspace.c_offset[j] = CartesianIndex(ntuple(d -> d == dim ? j : 0, Val(N)))
    end

    c_offset_view = view(workspace.c_offset, 1:(eqs-1))
    slice_view    = view(workspace.slice, 1:m)

    # ---- left boundary ----------------------------------------------------
    for idx in left_indices
        if N == 1
            @inbounds for j in 1:m
                workspace.slice[j] = vs[j]
            end
        else
            @inbounds for j in 1:m
                workspace.slice[j] = c[idx + workspace.slice_offset[j]]
            end
        end

        y_mean = zero(T)
        @inbounds for j in 1:m
            y_mean += workspace.slice[j]
        end
        y_mean /= m
        @inbounds for j in 1:m
            workspace.slice[j] -= y_mean
        end

        bcL = kernel_boundary_condition[1]
        use_polynomial_left = if few_points || UG
            false
        elseif bcL === :poly
            true
        elseif bcL === :detect
            bc_accept_polynomial(T, ghost_matrix, slice_view, 1, 1,
                                 m, num_ghost, hd, workspace)
        else
            false
        end

        if use_polynomial_left
            fill_ghost_points_polynomial!(c, idx, c_offset_view, ghost_matrix, y_mean, slice_view, :left, eqs, workspace)
        elseif UG
            fill_ghost_points_gaussian!(T, c, idx, c_offset_view, slice_view, y_mean, eqs, workspace, :left)
        elseif bcL === :linear || bcL === :quadratic
            bc_matrix = get_polynomial_ghost_coeffs(bcL, kernel_type)
            fill_ghost_points_polynomial!(c, idx, c_offset_view, bc_matrix, y_mean, slice_view, :left, eqs, workspace)
        elseif few_points || bcL === :poly || bcL === :detect
            bc_matrix = get_polynomial_ghost_coeffs(:linear, kernel_type)
            fill_ghost_points_polynomial!(c, idx, c_offset_view, bc_matrix, y_mean, slice_view, :left, eqs, workspace)
        else
            error("Unsupported boundary condition: $(bcL)")
        end
    end

    # ---- right boundary ---------------------------------------------------
    # slice[j] holds interior sample (n_dim - m + j), i.e. the last m values
    # in increasing order, so slice_view[end] is the final sample.
    for idx in right_indices
        if N == 1
            @inbounds for j in 1:m
                workspace.slice[j] = vs[n_dim - m + j]
            end
        else
            @inbounds for j in 1:m
                workspace.slice[j] = c[idx - workspace.slice_offset[m - j + 1]]
            end
        end

        y_mean = zero(T)
        @inbounds for j in 1:m
            y_mean += workspace.slice[j]
        end
        y_mean /= m
        @inbounds for j in 1:m
            workspace.slice[j] -= y_mean
        end

        bcR = kernel_boundary_condition[2]
        use_polynomial_right = if few_points || UG
            false
        elseif bcR === :poly
            true
        elseif bcR === :detect
            bc_accept_polynomial(T, ghost_matrix, slice_view, m, -1,
                                 m, num_ghost, hd, workspace)
        else
            false
        end

        if use_polynomial_right
            fill_ghost_points_polynomial!(c, idx, c_offset_view, ghost_matrix, y_mean, slice_view, :right, eqs, workspace)
        elseif UG
            fill_ghost_points_gaussian!(T, c, idx, c_offset_view, slice_view, y_mean, eqs, workspace, :right)
        elseif bcR === :linear || bcR === :quadratic
            bc_matrix = get_polynomial_ghost_coeffs(bcR, kernel_type)
            fill_ghost_points_polynomial!(c, idx, c_offset_view, bc_matrix, y_mean, slice_view, :right, eqs, workspace)
        elseif few_points || bcR === :poly || bcR === :detect
            bc_matrix = get_polynomial_ghost_coeffs(:linear, kernel_type)
            fill_ghost_points_polynomial!(c, idx, c_offset_view, bc_matrix, y_mean, slice_view, :right, eqs, workspace)
        else
            error("Unsupported boundary condition: $(bcR)")
        end
    end
end

"""
    fill_ghost_points_polynomial!(c::AbstractArray{T}, idx::CartesianIndex, 
                                  c_offset::AbstractVector{CartesianIndex{N}},
                                  coef::Matrix, y_offset::T, y_centered::Vector{T},
                                  side::Symbol, eqs::Int,
                                  workspace::BoundaryWorkspace{T,N}) where {T,N}

Fill ghost points using polynomial boundary conditions (non-recursive, direct computation).

# Arguments
- `c::AbstractArray{T}`: Coefficient array to modify
- `idx::CartesianIndex`: Index of the boundary point (interior grid point nearest to boundary)
- `c_offset::AbstractVector{CartesianIndex{N}}`: Precomputed offsets for ghost point positions
- `coef::Matrix`: Coefficient matrix where row j gives coefficients for ghost point g_{-j}
- `y_offset::T`: Mean offset of the signal
- `y_centered::Vector{T}`: Mean-centered signal values
- `side::Symbol`: `:left` or `:right` boundary
- `eqs::Int`: Number of equations (determines number of ghost points: eqs-1)
- `workspace::BoundaryWorkspace{T,N}`: Workspace for temporary arrays

# Details
Polynomial boundary conditions compute ghost points directly from interior values using
optimal kernel-specific coefficient matrices. This method:

- Preserves the polynomial reproduction property of the kernel
- Uses matrix-vector multiplication for efficiency
- Handles signal reversal automatically for right boundaries
- Adds back the mean offset to get final ghost point values

The computation is: `ghost[j] = y_offset + coef[j,:] ⋅ y_centered`

This is the recommended boundary condition method for most use cases.

See also: `fill_ghost_points_recursive!`, `get_polynomial_ghost_coeffs`.
"""

function fill_ghost_points_polynomial!(c::AbstractArray{T}, idx::CartesianIndex, 
                                      c_offset::AbstractVector{CartesianIndex{N}},
                                      coef::Matrix, y_offset::T, y_centered::AbstractVector{T},
                                      side::Symbol, eqs::Int,
                                      workspace::BoundaryWorkspace{T,N}) where {T,N}
    num_interior = size(coef, 2)
    num_ghost = eqs - 1  # Always use exactly eqs-1, not the full matrix
    
    if side == :left
        y_view = view(y_centered, 1:num_interior)
        ghost_vals_view = view(workspace.ghost_vals, 1:num_ghost)
        coef_view = view(coef, 1:num_ghost, :)  # Only use first eqs-1 rows
        
        mul!(ghost_vals_view, coef_view, y_view)

        for j in 1:num_ghost
            c[idx - c_offset[j]] = y_offset + workspace.ghost_vals[j]
        end
    else  # :right
        # Reverse into workspace
        for k in 1:num_interior
            workspace.y_temp[k] = y_centered[end - k + 1]
        end
        y_temp_view = view(workspace.y_temp, 1:num_interior)
        ghost_vals_view = view(workspace.ghost_vals, 1:num_ghost)
        coef_view = view(coef, 1:num_ghost, :)  # Only use first eqs-1 rows
        
        mul!(ghost_vals_view, coef_view, y_temp_view)
        
        for j in 1:num_ghost
            c[idx + c_offset[j]] = y_offset + workspace.ghost_vals[j]
        end
    end
end

function fill_ghost_points_gaussian!(T, c, idx, c_offset_view, slice_view, y_mean, eqs, workspace, vs_side)
    if vs_side == :left
        n_fit = min(max(6, eqs÷3), length(slice_view))
        # accumulate sums for normal equations
        sx = zero(T); sy = zero(T); sxx = zero(T); sxy = zero(T)
        for k in 1:n_fit
            xk = T(k)
            yk = slice_view[k]
            sx  += xk
            sy  += yk
            sxx += xk * xk
            sxy += xk * yk
        end
        # solve 2x2 system: [n sx; sx sxx] * [a; b] = [sy; sxy]
        denom = n_fit * sxx - sx * sx
        b = (n_fit * sxy - sx * sy) / denom  # slope
        a = (sy - b * sx) / n_fit            # intercept

        # evaluate at ghost point positions (0, -1, -2, ...)
        # and store mean-centered values in workspace
        for k in 1:(eqs-1)
            workspace.y_temp[k] = a + b * T(1 - k) # x = 0, -1, -2, ...
        end

        # now fill ghost points directly without bc_matrix
        for j in 1:(eqs-1)
            c[idx - c_offset_view[j]] = y_mean + workspace.y_temp[j]
        end
    else # vs_side == :right
        n_fit = min(max(6, eqs÷3), length(slice_view))
        sx = zero(T); sy = zero(T); sxx = zero(T); sxy = zero(T)
        for k in 1:n_fit
            xk = T(k)
            yk = slice_view[end - k + 1]
            sx  += xk
            sy  += yk
            sxx += xk * xk
            sxy += xk * yk
        end
        denom = n_fit * sxx - sx * sx
        b = (n_fit * sxy - sx * sy) / denom
        a = (sy - b * sx) / n_fit

        for k in 1:(eqs-1)
            workspace.y_temp[k] = a + b * T(1 - k)
        end

        for j in 1:(eqs-1)
            c[idx + c_offset_view[j]] = y_mean + workspace.y_temp[j]
        end
    end
end