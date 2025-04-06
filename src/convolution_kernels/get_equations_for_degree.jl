"""
    get_equations_for_degree(degree::Integer)

Get the number of piecewise equations required for a convolution kernel of the specified degree.

# Arguments
- `degree::Integer`: The polynomial degree of the convolution kernel

# Returns
- The number of piecewise equations used in the kernel of the given degree

# Throws
- `ArgumentError`: If the specified degree is not supported

# Details
This function provides a safe interface to access the `DEGREE_TO_EQUATIONS` mapping,
with proper error handling for unsupported degree values. It returns the number of
piecewise equations that define a convolution kernel of the requested degree.

Supported degrees are 3, 5, 7, 9, 11, and 13, corresponding to cubic, quintic, septic,
nonic, and higher-order kernels.

# Examples
```julia
# Get number of equations for a cubic kernel
eqs = get_equations_for_degree(3)  # Returns 3

# Get number of equations for a quintic kernel
eqs = get_equations_for_degree(5)  # Returns 5
```
"""
function get_equations_for_degree(degree::Integer)
    haskey(DEGREE_TO_EQUATIONS, degree) || throw(ArgumentError("Degree $degree not supported. Supported degrees: $(sort(collect(keys(DEGREE_TO_EQUATIONS))))"))
    return DEGREE_TO_EQUATIONS[degree]
end