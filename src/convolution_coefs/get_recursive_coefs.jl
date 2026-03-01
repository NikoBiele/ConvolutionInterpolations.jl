
"""
    get_recursive_coefs(y_centered, h::T, bc::Symbol, which_end::Symbol) where T

Get prediction coefficients for recursive (non-polynomial) boundary conditions.

This is the fallback path used when polynomial boundary conditions are unavailable,
e.g. for small grids where the number of interior points is less than the polynomial
ghost coefficient matrix requires.

# Arguments
- `y_centered`: Mean-centered signal values (already in-place in workspace)
- `h::T`: Grid spacing
- `bc::Symbol`: Boundary condition type (`:auto`, `:detect`, `:linear`, `:quadratic`, `:periodic`)
- `which_end::Symbol`: `:left` or `:right` boundary

# Returns
`Vector{T}` of prediction coefficients for iterative ghost point computation.

# Methods
- `:auto` / `:detect`: Analyzes signal structure via `detect_boundary_signal_fast`
- `:linear`: Returns `[2, -1, 0]` (first-order extrapolation)
- `:quadratic`: Returns `[3, -3, 1]` (second-order extrapolation)
- `:periodic`: Blends autocorrelation prediction with quadratic extrapolation

# Note
This function may allocate (coefficient vectors, signal analysis). It is only called
on the non-polynomial fallback path, which is rare for typical grid sizes. The hot path
(polynomial boundary conditions) bypasses this entirely.

See also: `apply_boundary_conditions_for_dim!`, `fill_ghost_points_recursive!`.
"""

function get_recursive_coefs(y_centered, h::T, bc::Symbol, which_end::Symbol) where T
    n = length(y_centered)
    if bc == :auto || bc == :detect
        return detect_boundary_signal_fast(y_centered, n, h; which_end=which_end)
    elseif bc == :linear
        return T[T(2), T(-1), T(0)]
    elseif bc == :quadratic
        return T[T(3), T(-3), T(1)]
    elseif bc == :periodic
        return periodic_boundary(y_centered)
    else
        error("Unsupported kernel boundary condition: kernel_bc = $bc. " *
              "Supported: :polynomial, :auto, :detect, :linear, :quadratic, :periodic.")
    end
end