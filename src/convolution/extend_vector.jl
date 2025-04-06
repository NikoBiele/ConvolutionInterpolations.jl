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
function extend_vector(x::AbstractVector, n_extra::Integer)
    step_start = x[2] - x[1]
    step_end = x[end] - x[end-1]
    
    start_extension = range(x[1] - n_extra * step_start, step=step_start, length=n_extra)
    end_extension = range(x[end] + step_end, step=step_end, length=n_extra)
    
    return vcat(start_extension, x, end_extension)
end