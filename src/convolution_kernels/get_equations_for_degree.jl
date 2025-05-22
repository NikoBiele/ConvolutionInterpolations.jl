"""
    get_equations_for_degree(degree::Integer)

Get the number of piecewise equations required for a convolution kernel of the specified degree.

# Arguments
- `degree::Symbol`: The polynomial degree of the convolution kernel

# Returns
- The number of piecewise equations used in the kernel of the given degree

# Throws
- `ArgumentError`: If the specified degree is not supported

# Details
This function provides a safe interface to access the `DEGREE_TO_EQUATIONS` mapping,
with proper error handling for unsupported degree values. It returns the number of
piecewise equations that define a convolution kernel of the requested degree.

Supported degrees are:
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
    :b13 => 9,
corresponding to nearest neighbor, linear, cubic, quintic, septic,
nonic, and higher-order kernels.

# Examples
```julia
# Get number of equations for a cubic kernel
eqs = get_equations_for_degree(3)  # Returns 3

# Get number of equations for a quintic kernel
eqs = get_equations_for_degree(5)  # Returns 5
```
"""
function get_equations_for_degree(degree::Symbol)
    haskey(DEGREE_TO_EQUATIONS, degree) || throw(ArgumentError("Degree $degree not supported. Supported degrees: $(sort(collect(keys(DEGREE_TO_EQUATIONS))))"))
    return DEGREE_TO_EQUATIONS[degree]
end