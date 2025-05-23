"""
    DEGREE_TO_EQUATIONS

A mapping from polynomial degree to the number of piecewise equations used in convolution kernels.

This dictionary defines the relationship between the degree of the polynomial used in a 
convolution kernel and the number of piecewise equations (segments) needed to properly 
define that kernel across its support range.

# Structure
- Keys: Polynomial degrees (3, 5, 7, 9, 11, 13)
- Values: Number of piecewise equations required

# Details
This mapping specifies:
- Degree 3: 2 equations with support [-2,2]
- Degree 5: 5 equations with support [-5,5]
- Degree 7: 6 equations with support [-6,6]
- Degree 9: 7 equations with support [-7,7]
- Degree 11: 8 equations with support [-8,8]
- Degree 13: 9 equations with support [-9,9]

The support range of a kernel of degree D equals [-E,E] where E is the number of equations.
This dictionary is used internally to determine the appropriate kernel configuration based
on the specified degree.
"""
const DEGREE_TO_EQUATIONS = Dict(
    :a0 => 1,
    :a1 => 1,
    :a3 => 2,
    :a5 => 3,
    :a7 => 4,
    :b3 => 3,
    :b5 => 5,
    :b7 => 6,
    :b9 => 7,
    :b11 => 8,
    :b13 => 9
)