"""
    BoundaryCondition

Abstract type representing different strategies for handling points outside the interpolation domain.

# Subtypes
- `Flat`: Uses the value at the nearest boundary point
- `Periodic`: Wraps coordinates with the period of the domain
- `Line`: Performs linear extrapolation using the gradient at the boundary
- `Reflect`: Reflects coordinates across the boundary
- `Throw`: Raises an error for points outside the domain

These boundary conditions can be used with the `extrapolate` function to extend
interpolation objects beyond their original domain.
"""
abstract type BoundaryCondition end