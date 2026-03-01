"""
    extend_vector(x::AbstractVector{T}, n_extra::Integer) where T

Extend a knot vector by adding `n_extra` points at each end, matching the spacing at
the respective boundary. Returns a new vector of length `length(x) + 2*n_extra`.

Used internally by `expand_knots` for boundary condition setup.
"""

function extend_vector(x::AbstractVector{T}, n_extra::Integer) where T
    step_start = x[2] - x[1]
    step_end = x[end] - x[end-1]
    
    # Preserve the type T throughout
    start_extension = T[x[1] - i * step_start for i in n_extra:-1:1]
    end_extension = T[x[end] + i * step_end for i in 1:n_extra]
    
    return vcat(start_extension, x, end_extension)
end