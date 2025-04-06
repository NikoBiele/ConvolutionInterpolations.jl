"""
    expand_knots(knots::NTuple{N,AbstractVector}, n_extra::Integer) where N

Expand a tuple of knot point vectors in multiple dimensions.

# Arguments
- `knots::NTuple{N,AbstractVector}`: Tuple of knot point vectors for each dimension
- `n_extra::Integer`: The number of extra points to add at each end of each dimension

# Returns
- A new tuple of extended knot point vectors

# Details
This function extends the knot points in all dimensions by applying `extend_vector` to
each component of the input tuple. The result is a new tuple of vectors where each
vector has been extended by `n_extra` points at both ends.

This is used internally to prepare the domain for proper boundary condition handling
in convolution interpolation.
"""
function expand_knots(knots::NTuple{N,AbstractVector}, n_extra::Integer) where N
    knots_new = ntuple(i -> extend_vector(knots[i], n_extra), N)
    return knots_new
end