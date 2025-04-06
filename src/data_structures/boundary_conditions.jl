"""
    Flat <: BoundaryCondition

Extrapolation boundary condition that returns the value at the nearest boundary point.

When a point outside the interpolation domain is evaluated, this boundary condition
returns the exact value at the closest point on the boundary, without any modification.
This creates a flat extension beyond the boundaries.
"""
struct Flat <: BoundaryCondition end

"""
    Periodic <: BoundaryCondition

Extrapolation boundary condition that wraps coordinates using the period of the domain.

When a point outside the interpolation domain is evaluated, this boundary condition
wraps the coordinates as if the data were periodic, with period equal to the domain size.
This is useful for inherently periodic data like angles or cyclic signals.
"""
struct Periodic <: BoundaryCondition end

"""
    Line <: BoundaryCondition

Extrapolation boundary condition that performs linear extrapolation from the boundary.

When a point outside the interpolation domain is evaluated, this boundary condition
computes a gradient at the nearest boundary point and extends linearly in that direction.
This provides a fourth-order continuous extension beyond the boundaries.
"""
struct Line <: BoundaryCondition end

"""
    Reflect <: BoundaryCondition

Extrapolation boundary condition that reflects coordinates across the boundary.

When a point outside the interpolation domain is evaluated, this boundary condition
reflects the coordinates across the nearest boundary, as if the data were mirrored.
This is useful for preserving symmetry and avoiding discontinuities at boundaries.
"""
struct Reflect <: BoundaryCondition end

"""
    Throw <: BoundaryCondition

Extrapolation boundary condition that raises an error for points outside the domain.

When a point outside the interpolation domain is evaluated, this boundary condition
raises an error rather than attempting to extrapolate. This is useful when extrapolation
is not appropriate for the data or application.
"""
struct Throw <: BoundaryCondition end