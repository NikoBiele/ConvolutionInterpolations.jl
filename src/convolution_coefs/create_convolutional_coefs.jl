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

function BoundaryWorkspace(T::Type, N::Int, max_eqs::Int, max_dim_size::Int)
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
coefs = create_convolutional_coefs(vs, h, 2, :auto, :a1)  # Returns vs unchanged

# Higher-order kernel with ghost points
coefs = create_convolutional_coefs(vs, h, 5, :polynomial, :b5)
# Returns expanded array of size (58, 58) with ghost points

# Different boundary conditions per dimension
bc = [(:polynomial, :polynomial), (:linear, :quadratic)]
coefs = create_convolutional_coefs(vs, h, 5, bc, :b5)
```

See also: `apply_boundary_conditions_for_dim!`, `boundary_coefs`.
"""

function create_convolutional_coefs(vs::AbstractArray{T,N}, h::NTuple{N,T}, eqs::P,
                                kernel_bc::Union{Symbol,Vector{Tuple{Symbol,Symbol}}},
                                kernel_type::Symbol) where {T,N,P}
       
    new_dims = size(vs) .+ 2*(eqs-1)
    c = zeros(T, new_dims)
    inner_indices = map(d -> (1+(eqs-1)):(d-(eqs-1)), new_dims)
    c[inner_indices...] = vs

    # Create workspace once
    max_dim_size = maximum(size(vs))
    workspace = BoundaryWorkspace(T, N, eqs, max_dim_size)

    # Apply boundary conditions in order
    for fixed_dim in 1:N
        apply_boundary_conditions_for_dim!(c, vs, fixed_dim, h, eqs, kernel_bc, kernel_type, workspace)
    end

    return c
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
                                           kernel_bc::Union{Symbol,Vector{Tuple{Symbol,Symbol}}}, 
                                           kernel_type::Symbol,
                                           workspace::BoundaryWorkspace{T,N}) where {T,N}
    
    kernel_boundary_condition = if kernel_bc isa Symbol
        (kernel_bc, kernel_bc)
    else
        kernel_bc[dim]
    end
    
    left_indices, right_indices = get_boundary_indices(size(c), dim, eqs)
    
    # Precompute slice_offset and c_offset once (reuse workspace arrays)
    for j in 0:size(vs, dim)-1
        workspace.slice_offset[j+1] = CartesianIndex(ntuple(d -> d == dim ? j : 0, Val(N)))
    end
    
    for j in 1:(eqs-1)
        workspace.c_offset[j] = CartesianIndex(ntuple(d -> d == dim ? j : 0, Val(N)))
    end

    # Fetch ghost matrix once (same for all boundary points)
    ghost_matrix = get_polynomial_ghost_coeffs(kernel_type)
    n_dim = size(vs, dim)
    n_interior = size(ghost_matrix, 2)
    use_polynomial_left = kernel_boundary_condition[1] == :polynomial || 
                          (kernel_boundary_condition[1] == :auto && n_dim >= n_interior)
    use_polynomial_right = kernel_boundary_condition[2] == :polynomial || 
                           (kernel_boundary_condition[2] == :auto && n_dim >= n_interior)
    
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
        
        if use_polynomial_left
            fill_ghost_points_polynomial!(c, idx, c_offset_view, ghost_matrix, y_mean, slice_view, :left, eqs, workspace)
        else
            coef = get_recursive_coefs(slice_view, h[dim], kernel_boundary_condition[1], :left)
            fill_ghost_points_recursive!(c, idx, c_offset_view, coef, y_mean, slice_view, :left, eqs, size(vs), workspace)
        end
    end
    
    # Process right boundary
    for idx in right_indices
        if N == 1
            workspace.slice[1:length(vs)] .= vs
        else
            for j in 1:n_dim
                workspace.slice[j] = c[idx - workspace.slice_offset[j]]
            end
        end
        
        slice_view = view(workspace.slice, 1:n_dim)
        y_mean = sum(slice_view) / n_dim
        for k in 1:n_dim
            workspace.slice[k] -= y_mean
        end
        
        if use_polynomial_right
            fill_ghost_points_polynomial!(c, idx, c_offset_view, ghost_matrix, y_mean, slice_view, :right, eqs, workspace)
        else
            coef = get_recursive_coefs(slice_view, h[dim], kernel_boundary_condition[2], :right)
            fill_ghost_points_recursive!(c, idx, c_offset_view, coef, y_mean, slice_view, :right, eqs, size(vs), workspace)
        end
    end
end

"""
    fill_ghost_points_polynomial!(c::AbstractArray{T}, idx::CartesianIndex,
        c_offset::AbstractVector{CartesianIndex{N}},
        coef::Matrix, y_offset::T, y_centered::AbstractVector{T},
        side::Symbol, eqs::Int,
        workspace::BoundaryWorkspace{T,N}) where {T,N}

Fill ghost points using polynomial boundary conditions via direct matrix-vector multiplication.

Computes `ghost[j] = y_offset + coef[j,:] ⋅ y_centered` using preallocated workspace arrays.
For right boundaries, reverses the signal into `workspace.y_temp` before multiplication.
Uses `mul!` into `workspace.ghost_vals` for allocation-free computation.

`y_centered` is an `AbstractVector{T}` (typically a view into the workspace slice).

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
        coef_view = view(coef, 1:num_ghost, 1:num_interior)  # Only use first eqs-1 rows

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
        coef_view = view(coef, 1:num_ghost, 1:num_interior)  # Only use first eqs-1 rows
        factor = length(size(c)) == 1 ? one(T) : -one(T)
        
        mul!(ghost_vals_view, coef_view, y_temp_view)
            
        for j in 1:num_ghost
            c[idx + c_offset[j]] = y_offset + factor * workspace.ghost_vals[j]
        end
    end
end

"""
    fill_ghost_points_recursive!(c::AbstractArray{T}, idx::CartesianIndex,
        c_offset::AbstractVector{CartesianIndex{N}},
        coef::Vector, y_offset::T, y_centered::AbstractVector{T},
        side::Symbol, eqs::Int, vs_size::NTuple,
        workspace::BoundaryWorkspace{T,N}) where {T,N}

Fill ghost points using recursive (iterative) extrapolation.

Each ghost point depends on previously computed ghost points and interior values:
`ghost[j] = sum(coef[k] * extended[j-k] for k in 1:length(coef))`.

Uses `workspace.y_extended` for the extended signal array. Fallback method for small grids
or non-polynomial boundary conditions.

`y_centered` is an `AbstractVector{T}` (typically a view into the workspace slice).

See also: `fill_ghost_points_polynomial!`, `get_recursive_coefs`.
"""

function fill_ghost_points_recursive!(c::AbstractArray{T}, idx::CartesianIndex,
                                     c_offset::AbstractVector{CartesianIndex{N}},
                                     coef::Vector, y_offset::T, y_centered::AbstractVector{T},
                                     side::Symbol, eqs::Int, vs_size::NTuple,
                                     workspace::BoundaryWorkspace{T,N}) where {T,N}
    if side == :left
        y_len = length(y_centered)
        y_ext_view = view(workspace.y_extended, 1:((eqs-1) + y_len))
        y_ext_view[1+(eqs-1):end] .= y_centered
        
        for j in 1:(eqs-1)
            y_ext_view[1+(eqs-1)-j] = sum(coef[k] * y_ext_view[1 + (eqs-1) - j + k] for k = 1:length(coef))
            c[idx - c_offset[j]] = y_offset + y_ext_view[1+(eqs-1)-j]
        end
    else  # :right
        factor = length(vs_size) == 1 ? one(T) : -one(T)
        y_len = length(y_centered)
        y_ext_view = view(workspace.y_extended, 1:(y_len + eqs - 1))
        y_ext_view[1:end-(eqs-1)] .= y_centered
        
        for j in 1:(eqs-1)
            y_ext_view[y_len+j] = sum(coef[k] * y_ext_view[y_len + j - k] for k = 1:length(coef))
            c[idx + c_offset[j]] = y_offset + factor * y_ext_view[y_len+j]
        end
    end
end