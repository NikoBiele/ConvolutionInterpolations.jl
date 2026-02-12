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

Compute CartesianIndex ranges for left and right boundary regions in a given dimension.

# Arguments
- `c_size::NTuple{N,Int}`: Size of the coefficient array (including ghost points)
- `dim::Int`: Dimension along which to extract boundary indices (1 to N)
- `eqs::Int`: Number of equations (kernel support size)

# Returns
Tuple of `(left_indices, right_indices)` where each is a `CartesianIndices` object
spanning the (N-1)-dimensional boundary hyperplane at the fixed boundary position.

# Details
The boundary indices correspond to the first and last interior grid points along the
specified dimension, extended across all other dimensions. These indices are used to
iterate over all points where boundary conditions must be applied.

# Examples
```julia
# For a 3D array of size (50, 60, 70) with eqs=5
left_idx, right_idx = get_boundary_indices((50, 60, 70), 1, 5)
# left_idx: all points at x[1] = 5 (the left interior boundary)
# right_idx: all points at x[1] = 46 (the right interior boundary)
```
"""

function get_boundary_indices(c_size::NTuple{N,Int}, dim::Int, eqs::Int) where N
    # Create ranges for all dimensions
    ranges = [1:s for s in c_size]
    
    # Left boundary: fix dim to left boundary position
    left_ranges = copy(ranges)
    left_ranges[dim] = (1 + (eqs-1)):(1 + (eqs-1))
    left_indices = CartesianIndices(Tuple(left_ranges))
    
    # Right boundary: fix dim to right boundary position  
    right_ranges = copy(ranges)
    right_ranges[dim] = (c_size[dim] - (eqs-1)):(c_size[dim] - (eqs-1))
    right_indices = CartesianIndices(Tuple(right_ranges))
    
    return left_indices, right_indices
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
    
    if kernel_type == :a0 || kernel_type == :a1
        return vs
    end
       
    new_dims = size(vs) .+ 2*(eqs-1)
    c = zeros(T, new_dims)
    inner_indices = map(d -> (1+(eqs-1)):(d-(eqs-1)), new_dims)
    c[inner_indices...] = vs

    # Create workspace once
    max_dim_size = maximum(size(vs))
    workspace = BoundaryWorkspace(T, N, eqs, max_dim_size)

    # Apply boundary conditions in order
    for fixed_dim in 1:N
        apply_boundary_conditions_for_dim!(c, vs, fixed_dim, h, eqs, kernel_bc, kernel_type, N, workspace)
    end

    return c
end

"""
    apply_boundary_conditions_for_dim!(c::AbstractArray{T,N}, vs::AbstractArray, dim::Int, 
                                       h::NTuple{N,T}, eqs::Int,
                                       kernel_bc::Union{Symbol,Vector{Tuple{Symbol,Symbol}}}, 
                                       kernel_type::Symbol, ndims::Int,
                                       workspace::BoundaryWorkspace{T,N}) where {T,N}

Apply boundary conditions along a single dimension of the coefficient array.

# Arguments
- `c::AbstractArray{T,N}`: Coefficient array to modify (with ghost points)
- `vs::AbstractArray`: Original interior data values
- `dim::Int`: Dimension to process (1 to N)
- `h::NTuple{N,T}`: Grid spacing in each dimension
- `eqs::Int`: Number of equations (kernel support size)
- `kernel_bc`: Boundary condition specification
- `kernel_type::Symbol`: Kernel degree (`:a3`, `:b5`, etc.)
- `ndims::Int`: Total number of dimensions
- `workspace::BoundaryWorkspace{T,N}`: Preallocated workspace for temporary arrays

# Details
This function processes one dimension at a time, filling ghost points on both the left
and right boundaries. For each boundary point on the (N-1)-dimensional hyperplane:

1. Extracts a 1D slice along the specified dimension
2. Computes boundary coefficients using `boundary_coefs`
3. Fills ghost points using either polynomial or recursive method
4. Repeats for all points on the boundary hyperplane

The workspace enables allocation-free computation by reusing temporary arrays across
all boundary points and dimensions.

# Algorithm
For each boundary (left and right):
- Extract 1D slices perpendicular to the boundary
- Compute ghost point values from interior points
- Write ghost values into the coefficient array

Boundary conditions can be different for left vs. right sides of each dimension.

See also: `fill_ghost_points_polynomial!`, `fill_ghost_points_recursive!`
"""

function apply_boundary_conditions_for_dim!(c::AbstractArray{T,N}, vs::AbstractArray, dim::Int, 
                                           h::NTuple{N,T}, eqs::Int,
                                           kernel_bc::Union{Symbol,Vector{Tuple{Symbol,Symbol}}}, 
                                           kernel_type::Symbol, ndims::Int,
                                           workspace::BoundaryWorkspace{T,N}) where {T,N}
    
    kernel_boundary_condition = if kernel_bc isa Symbol
        (kernel_bc, kernel_bc)
    else
        kernel_bc[dim]
    end
    
    left_indices, right_indices = get_boundary_indices(size(c), dim, eqs)
    
    # Precompute slice_offset and c_offset once (reuse workspace arrays)
    for j in 0:size(vs, dim)-1
        workspace.slice_offset[j+1] = CartesianIndex(ntuple(d -> d == dim ? j : 0, ndims))
    end
    
    for j in 1:(eqs-1)
        workspace.c_offset[j] = CartesianIndex(ntuple(d -> d == dim ? j : 0, ndims))
    end
    
    # Process left boundary
    for idx in left_indices
        # Extract slice using preallocated array
        if length(size(vs)) == 1
            workspace.slice[1:length(vs)] .= vs
        else
            for j in 1:size(vs, dim)
                workspace.slice[j] = c[idx + workspace.slice_offset[j]]
            end
        end
        
        slice_view = view(workspace.slice, 1:size(vs, dim))
        coef, y_offset, y_centered = boundary_coefs(slice_view, h[dim], kernel_boundary_condition[1], :left, kernel_type)
        
        c_offset_view = view(workspace.c_offset, 1:(eqs-1))
        
        if coef isa Matrix
            fill_ghost_points_polynomial!(c, idx, c_offset_view, coef, y_offset, y_centered, :left, eqs, workspace)
        else
            fill_ghost_points_recursive!(c, idx, c_offset_view, coef, y_offset, y_centered, :left, eqs, size(vs), workspace)
        end
    end
    
    # Process right boundary
    for idx in right_indices
        # Extract slice using preallocated array
        if length(size(vs)) == 1
            workspace.slice[1:length(vs)] .= vs
        else
            for j in 1:size(vs, dim)
                workspace.slice[j] = c[idx - workspace.slice_offset[j]]
            end
        end
        
        slice_view = view(workspace.slice, 1:size(vs, dim))
        coef, y_offset, y_centered = boundary_coefs(slice_view, h[dim], kernel_boundary_condition[2], :right, kernel_type)
        
        c_offset_view = view(workspace.c_offset, 1:(eqs-1))
        
        if coef isa Matrix
            fill_ghost_points_polynomial!(c, idx, c_offset_view, coef, y_offset, y_centered, :right, eqs, workspace)
        else
            fill_ghost_points_recursive!(c, idx, c_offset_view, coef, y_offset, y_centered, :right, eqs, size(vs), workspace)
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
                                      coef::Matrix, y_offset::T, y_centered::Vector{T},
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
        
        mul!(ghost_vals_view, coef_view, y_temp_view)
            
        for j in 1:num_ghost
            c[idx + c_offset[j]] = y_offset + workspace.ghost_vals[j]
        end
    end
end

"""
    fill_ghost_points_recursive!(c::AbstractArray{T}, idx::CartesianIndex,
                                 c_offset::AbstractVector{CartesianIndex{N}},
                                 coef::Vector, y_offset::T, y_centered,
                                 side::Symbol, eqs::Int, vs_size::NTuple,
                                 workspace::BoundaryWorkspace{T,N}) where {T,N}

Fill ghost points using recursive boundary conditions (iterative extrapolation).

# Arguments
- `c::AbstractArray{T}`: Coefficient array to modify
- `idx::CartesianIndex`: Index of the boundary point
- `c_offset::AbstractVector{CartesianIndex{N}}`: Precomputed offsets for ghost point positions
- `coef::Vector`: Prediction coefficients for recursive computation
- `y_offset::T`: Mean offset of the signal
- `y_centered`: Mean-centered signal values
- `side::Symbol`: `:left` or `:right` boundary
- `eqs::Int`: Number of equations (determines number of ghost points: eqs-1)
- `vs_size::NTuple`: Size of original data array
- `workspace::BoundaryWorkspace{T,N}`: Workspace for temporary arrays

# Details
Recursive boundary conditions compute ghost points iteratively, where each ghost point
depends on previously computed ghost points and interior values. Common methods include:

- Linear extrapolation: `coef = [2.0, -1.0]`
- Quadratic extrapolation: `coef = [3.0, -3.0, 1.0]`
- Auto-detected periodic patterns

The recursion proceeds outward from the interior:
```
ghost[j] = sum(coef[k] * (ghost[j-k] or interior[...]) for k in 1:length(coef))
```

This method is used as a fallback when polynomial coefficients are unavailable or for
small grids where polynomial method is not applicable.

See also: `fill_ghost_points_polynomial!`, `detect_boundary_signal_fast`.
"""

function fill_ghost_points_recursive!(c::AbstractArray{T}, idx::CartesianIndex,
                                     c_offset::AbstractVector{CartesianIndex{N}},
                                     coef::Vector, y_offset::T, y_centered,
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