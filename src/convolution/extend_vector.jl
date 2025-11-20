"""
    extend_vector(x::AbstractVector, n_extra::Integer)

Extend a vector of knot points by adding points at both ends.

# Arguments
- `x::AbstractVector`: The vector of knot points to extend
- `n_extra::Integer`: The number of extra points to add at each end

# Returns
- A new vector with extra points added at both ends

# Details
This function extends a vector of knot points (e.g., a grid of coordinates) by adding
`n_extra` points at each end. The spacing of the added points matches the spacing at the
respective end of the original vector.

This is primarily used internally to extend the domain for boundary condition handling
in convolution interpolation.
"""
function extend_vector(x::AbstractVector{T}, n_extra::Integer) where T
    step_start = x[2] - x[1]
    step_end = x[end] - x[end-1]
    
    # Preserve the type T throughout
    start_extension = T[x[1] - i * step_start for i in n_extra:-1:1]
    end_extension = T[x[end] + i * step_end for i in 1:n_extra]
    
    return vcat(start_extension, x, end_extension)
end