"""
    expand_knots(knots::NTuple{N,AbstractVector}, n_extra::Integer) where N

Expand a tuple of knot vectors by adding `n_extra` ghost points at each end of each dimension.

Applies `extend_vector` to each component. Used internally to prepare the domain for
boundary condition handling in convolution interpolation.
"""

function expand_knots(knots::NTuple{N,AbstractVector}, n_extra::Integer) where N
    knots_new = ntuple(i -> extend_vector(knots[i], n_extra), N)
    return knots_new
end