"""
    BoundaryWorkspace{T,N}

Preallocated workspace for efficient boundary condition computation in N-dimensional arrays.

# Fields
- `ghost_vals::Vector{T}`: Temporary storage for computed ghost point values
- `y_temp::Vector{T}`: Temporary storage for reversed signal data
- `slice::Vector{T}`: Storage for extracted 1D slices from N-D arrays
- `slice_offset::Vector{CartesianIndex{N}}`: Precomputed offsets for slice extraction
- `c_offset::Vector{CartesianIndex{N}}`: Precomputed offsets for ghost point placement
- `y_extended::Vector{T}`: Extended signal array for recursive boundary computation

# Details
This workspace structure eliminates allocations during boundary condition application by
preallocating all necessary temporary arrays. It is reused across all boundary dimensions
and all boundary points, making the boundary coefficient computation allocation-free.

See also: `BoundaryWorkspace(T, N, max_eqs, max_dim_size)`.
"""
# Workspace for boundary condition computation
struct BoundaryWorkspace{T,N}
    # For polynomial method
    ghost_vals::Vector{T}
    y_temp::Vector{T}
    
    # For slice extraction
    slice::Vector{T}
    slice_offset::Vector{CartesianIndex{N}}
    
    # For ghost point offsets
    c_offset::Vector{CartesianIndex{N}}
    
    # For recursive method
    y_extended::Vector{T}
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
        zeros(T, max_eqs - 1),                # ghost_vals - always eqs-1
        zeros(T, max_dim_size),               # y_temp
        zeros(T, max_dim_size),               # slice
        Vector{CartesianIndex{N}}(undef, max_dim_size), # slice_offset
        Vector{CartesianIndex{N}}(undef, max_eqs - 1),  # c_offset - always eqs-1
        zeros(T, max_dim_size + 2*(max_eqs-1))  # y_extended
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

"""
    apply_boundary_conditions_for_dim!(c::AbstractArray{T,N}, vs::AbstractArray, dim::Int,
        h::NTuple{N,T}, eqs::Int,
        kernel_bc::Union{Symbol,Vector{Tuple{Symbol,Symbol}}},
        kernel_type::Symbol,
        workspace::BoundaryWorkspace{T,N}) where {T,N}

Apply boundary conditions along a single dimension of the coefficient array. Fills ghost
points on both left and right boundaries.

# Arguments
- `c`: Coefficient array to modify (expanded with ghost point slots)
- `vs`: Original interior data values
- `dim`: Dimension to process (1 to N)
- `h`: Grid spacing tuple
- `eqs`: Number of equations (kernel support size)
- `kernel_bc`: Boundary condition specification (Symbol or per-dimension tuple vector)
- `kernel_type`: Kernel degree (`:a3`, `:b5`, etc.)
- `workspace`: Preallocated `BoundaryWorkspace` for temporary arrays

# Algorithm
1. Fetches the polynomial ghost coefficient matrix once (shared across all boundary points)
2. Determines polynomial vs recursive path per side based on grid size vs matrix requirements
3. Precomputes `CartesianIndex` offsets using `ntuple(Val(N))` (zero allocations)
4. For each boundary point: extracts a 1D slice into the workspace, computes mean-centering
   in-place, then fills ghost points via polynomial (`mul!`) or recursive fallback

The polynomial path (default for grids with sufficient points) is fully allocation-free.
The recursive fallback path may allocate for signal analysis.

See also: `fill_ghost_points_polynomial!`, `fill_ghost_points_recursive!`, `get_recursive_coefs`.
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
    
    # Precompute slice_offset and c_offset once (reuse workspace arrays)
    for j in 0:size(vs, dim)-1
        workspace.slice_offset[j+1] = CartesianIndex(ntuple(d -> d == dim ? j : 0, Val(N)))
    end
    
    for j in 1:(eqs-1)
        workspace.c_offset[j] = CartesianIndex(ntuple(d -> d == dim ? j : 0, Val(N)))
    end

    # Fetch ghost matrix once (same for all boundary points)
    ghost_matrix = UG ? nothing : get_polynomial_ghost_coeffs(:not_used, kernel_type)
    n_dim = size(vs, dim)
    n_interior = UG ? eqs : size(ghost_matrix, 2)
    few_points = n_dim < n_interior
    
    c_offset_view = view(workspace.c_offset, 1:(eqs-1))
    
    # Process left boundary
    for idx in left_indices
        if N == 1
            workspace.slice[1:length(vs)] .= vs
        else
            for j in 1:n_dim
                workspace.slice[j] = c[idx + workspace.slice_offset[j]]
            end
        end
        
        slice_view = view(workspace.slice, 1:n_dim)
        y_mean = sum(slice_view) / n_dim
        for k in 1:n_dim
            workspace.slice[k] -= y_mean
        end
    
        if few_points || UG || !(kernel_boundary_condition[1] in (:detect, :poly))
            use_polynomial_left = false
        else
            ns_left = n_interior
            # d1
            mn1, mx1 = typemax(T), typemin(T)
            for d in 1:(ns_left-1)
                v = slice_view[d+1] - slice_view[d]
                workspace.y_temp[d] = v
                mn1 = min(mn1, v)
                mx1 = max(mx1, v)
            end
            periodic_boundary_left = (mn1 * mx1) / h[dim]^2 < -1/10
            # d2
            mn2, mx2 = typemax(T), typemin(T)
            for d in 1:(ns_left-2)
                v = workspace.y_temp[d+1] - workspace.y_temp[d]
                workspace.y_extended[d] = v
                mn2 = min(mn2, v)
                mx2 = max(mx2, v)
            end
            high_curvature_left = abs(mx2 - mn2) / h[dim] > 1/2
            # d3 - only meaningful for ns_left > 4 (b-kernels)
            high_d3_left = if ns_left > 4
                mn3, mx3 = typemax(T), typemin(T)
                for d in 1:(ns_left-3)
                    v = workspace.y_extended[d+1] - workspace.y_extended[d]
                    mn3 = min(mn3, v)
                    mx3 = max(mx3, v)
                end
                abs(mx3 - mn3) / h[dim] > 1/2
            else
                false
            end            
            use_polynomial_left = (kernel_boundary_condition[1] == :poly ||
                    (kernel_boundary_condition[1] == :detect && 
                    !periodic_boundary_left && !high_curvature_left && !high_d3_left))
        end

        if use_polynomial_left
            fill_ghost_points_polynomial!(c, idx, c_offset_view, ghost_matrix, y_mean, slice_view, :left, eqs, workspace)
        elseif UG
            fill_ghost_points_gaussian!(T, c, idx, c_offset_view, slice_view, y_mean, eqs, workspace, :left)
        elseif kernel_boundary_condition[1] == :linear || kernel_boundary_condition[1] == :quadratic
            bc_matrix = get_polynomial_ghost_coeffs(kernel_boundary_condition[1], kernel_type)
            fill_ghost_points_polynomial!(c, idx, c_offset_view, bc_matrix, y_mean, slice_view, :left, eqs, workspace)
        elseif few_points || kernel_boundary_condition[1] == :poly || kernel_boundary_condition[1] == :detect
            bc_matrix = get_polynomial_ghost_coeffs(:linear, kernel_type)
            fill_ghost_points_polynomial!(c, idx, c_offset_view, bc_matrix, y_mean, slice_view, :left, eqs, workspace)
        else
            error("Unsupported boundary condition: $(kernel_boundary_condition[1])")
        end
    end
    
    # Process right boundary
    for idx in right_indices
        if N == 1
            workspace.slice[1:length(vs)] .= vs
        else
            for j in 1:n_dim
                workspace.slice[j] = c[idx - workspace.slice_offset[n_dim - j + 1]]
            end
        end

        slice_view = view(workspace.slice, 1:n_dim)
        y_mean = sum(slice_view) / n_dim
        for k in 1:n_dim
            workspace.slice[k] -= y_mean
        end

        if few_points || UG || !(kernel_boundary_condition[2] in (:detect, :poly))
            use_polynomial_right = false
        else
            ns_right = n_interior
            # d1 - note: right boundary uses the last ns_right points
            mn1, mx1 = typemax(T), typemin(T)
            for d in 1:(ns_right-1)
                v = slice_view[end-(ns_right-1)+d] - slice_view[end-(ns_right-1)+d-1]
                workspace.y_temp[d] = v
                mn1 = min(mn1, v)
                mx1 = max(mx1, v)
            end
            periodic_boundary_right = (mn1 * mx1) / h[dim]^2 < -1/10
            # d2
            mn2, mx2 = typemax(T), typemin(T)
            for d in 1:(ns_right-2)
                v = workspace.y_temp[d+1] - workspace.y_temp[d]
                workspace.y_extended[d] = v
                mn2 = min(mn2, v)
                mx2 = max(mx2, v)
            end
            high_curvature_right = abs(mx2 - mn2) / h[dim] > 1/2
            # d3 - only meaningful for ns_right > 4 (b-kernels)
            high_d3_right = if ns_right > 4
                mn3, mx3 = typemax(T), typemin(T)
                for d in 1:(ns_right-3)
                    v = workspace.y_extended[d+1] - workspace.y_extended[d]
                    mn3 = min(mn3, v)
                    mx3 = max(mx3, v)
                end
                abs(mx3 - mn3) / h[dim] > 1/2
            else
                false
            end
            use_polynomial_right = (kernel_boundary_condition[2] == :poly ||
                                    (kernel_boundary_condition[2] == :detect && 
                                    !periodic_boundary_right && !high_curvature_right && !high_d3_right))
        end

        if use_polynomial_right
            fill_ghost_points_polynomial!(c, idx, c_offset_view, ghost_matrix, y_mean, slice_view, :right, eqs, workspace)
        elseif UG
            fill_ghost_points_gaussian!(T, c, idx, c_offset_view, slice_view, y_mean, eqs, workspace, :right)
        elseif kernel_boundary_condition[2] == :linear || kernel_boundary_condition[2] == :quadratic
            bc_matrix = get_polynomial_ghost_coeffs(kernel_boundary_condition[2], kernel_type)
            fill_ghost_points_polynomial!(c, idx, c_offset_view, bc_matrix, y_mean, slice_view, :right, eqs, workspace)
        elseif few_points || kernel_boundary_condition[2] == :poly || kernel_boundary_condition[2] == :detect
            bc_matrix = get_polynomial_ghost_coeffs(:linear, kernel_type)
            fill_ghost_points_polynomial!(c, idx, c_offset_view, bc_matrix, y_mean, slice_view, :right, eqs, workspace)
        else
            error("Unsupported boundary condition: $(kernel_boundary_condition[2])")
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
            workspace.y_temp[k] = a + b * T(1 - k) - y_mean  # x = 0, -1, -2, ...
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
            workspace.y_temp[k] = a + b * T(1 - k) - y_mean
        end

        for j in 1:(eqs-1)
            c[idx + c_offset_view[j]] = y_mean + workspace.y_temp[j]
        end
    end
end