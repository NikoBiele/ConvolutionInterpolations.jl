"""
    create_convolutional_coefs(vs::AbstractArray{T,N}, h::NTuple{N,T}, eqs::Int) where {T,N}

Creates an extended array with boundary conditions applied for convolution operations.

# Arguments
- `vs::AbstractArray{T,N}`: Input array of dimension N
- `h::NTuple{N,T}`: Tuple of scaling parameters for each dimension
- `eqs::Int`: Order of the boundary equations (extension size)

# Returns
- Extended array with boundary conditions applied

# Details
This function:
1. Creates a new array with dimensions expanded by `2*(eqs-1)` in each direction
2. Copies the original data to the center of the new array
3. Applies boundary conditions dimension by dimension
4. Uses linear prediction to extend the signal at boundaries in a way that preserves signal characteristics

Boundary conditions are applied sequentially by dimension to ensure proper handling of corners and edges.
"""
function create_convolutional_coefs(vs::AbstractArray{T,N}, h::NTuple{N,T}, eqs::P, kernel_bc) where {T,N,P}
    new_dims = size(vs) .+ 2*(eqs-1)
    c = zeros(T, new_dims)
    inner_indices = map(d -> (1+(eqs-1)):(d-(eqs-1)), new_dims)
    c[inner_indices...] = vs
    factor = length(size(vs)) == 1 ? 1 : -1

    # helper function to apply boundary condition along specific dimensions
    function apply_boundary_condition!(c, fixed_dims)
        for idx in CartesianIndices(size(c))
            is_boundary = any(idx[dim] in (1+(eqs - 1), size(c, dim) - (eqs - 1)) for dim in fixed_dims)
            if !is_boundary
                continue
            end

            kernel_boundary_condition = Vector{Symbol}(undef, 2)
            if kernel_bc isa Symbol
                kernel_boundary_condition[1] = kernel_bc
                kernel_boundary_condition[2] = kernel_bc
            else
                kernel_boundary_condition = kernel_bc[fixed_dims]
            end

            for dim in fixed_dims

                if idx[dim] == 1 + (eqs - 1)

                    slice_offset = [CartesianIndex(ntuple(d -> d == dim ? i : 0, N)) for i = 0:size(vs, dim)-1]
                    slice = length(size(vs)) == 1 ? vs : [c[idx+slice_offset[i]] for i in 1:size(vs, dim)]
                    coef, y_offset, y_centered = boundary_coefs(slice[1:end], h[dim], kernel_boundary_condition[1])
                    c_offset = [CartesianIndex(ntuple(d -> d == dim ? i : 0, N)) for i = 1:(eqs-1)]
                    y_extended = zeros((eqs-1)+length(y_centered))
                    y_extended[1+(eqs-1):end] .= y_centered
                    for j in 1:(eqs-1)
                        y_extended[1+(eqs-1)-j] = sum( coef[k] * y_extended[1 + (eqs-1) - j + k] for k = 1:length(coef))
                        c[idx-c_offset[j]] = y_offset + y_extended[1+(eqs-1)-j]
                    end

                elseif idx[dim] == size(c, dim) - (eqs - 1)

                    slice_offset = [CartesianIndex(ntuple(d -> d == dim ? i : 0, N)) for i = 0:size(vs, dim)-1]
                    slice = length(size(vs)) == 1 ? vs : [c[idx-slice_offset[i]] for i in 1:size(vs, dim)]
                    coef, y_offset, y_centered = boundary_coefs(slice[1:end], h[dim], kernel_boundary_condition[2])
                    c_offset = [CartesianIndex(ntuple(d -> d == dim ? i : 0, N)) for i = 1:eqs-1]
                    y_extended = zeros(length(y_centered)+eqs-1)
                    y_extended[1:end-(eqs-1)] .= y_centered
                    for j in 1:eqs-1
                        y_extended[length(y_centered)+j] = sum( coef[k] * y_extended[length(y_centered) + j - k] for k = 1:length(coef))
                        c[idx+c_offset[j]] = y_offset + factor * y_extended[length(y_centered)+j]
                    end    
                end
            end
        end
    end

    # Apply boundary conditions in order: 1 dimension, then 2, then 3, etc.
    for fixed_dims in 1:N
        apply_boundary_condition!(c, fixed_dims)
    end

    return c
end
