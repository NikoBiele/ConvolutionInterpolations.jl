"""
    get_equations_for_degree(degree::Symbol)

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
    :a4 => 3,
    :a5 => 3,
    :a7 => 4,
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
eqs = get_equations_for_degree(:a4)  # Returns 3

# Get number of equations for a quintic kernel
eqs = get_equations_for_degree(:b5)  # Returns 5
```
"""
function get_equations_for_degree(degree::Symbol)
    haskey(DEGREE_TO_EQUATIONS, degree) || throw(ArgumentError("Degree $degree not supported. Supported degrees: $(sort(collect(keys(DEGREE_TO_EQUATIONS))))"))
    return DEGREE_TO_EQUATIONS[degree]
end

function _default_kernel(N::Int)
    if N <= 2
        return :b7
    elseif N == 3
        return :b5
    elseif N <= 5
        return :a4
    else
        return :a3
    end
end

"""
    LAZY_MAX_EAGER_COEFFICIENTS

Size of eager mode's coefficient array (the data copied, plus `eqs − 1` ghost points on each side of
every axis) above which `lazy=nothing` chooses lazy mode, to save memory: 2²⁷ coefficients, 1 GiB in
`Float64`.
"""
const LAZY_MAX_EAGER_COEFFICIENTS = 2^27

"""
    LAZY_MAX_EAGER_GHOSTS

Number of ghost points eager mode would compute above which `lazy=nothing` chooses lazy mode, to save
construction time: 2²⁴ ghost points, roughly a second of construction (each ghost point is a
compensated sum over the interior values, plus the `:detect` test of its line).
"""
const LAZY_MAX_EAGER_GHOSTS = 2^24

"""
    _default_lazy(sizes, kernels, derivatives)

The choice `lazy=nothing` makes on the uniform fast path: lazy when eager expansion would use too
much memory (more than `LAZY_MAX_EAGER_COEFFICIENTS` coefficients in total) or take too long to
build (more than `LAZY_MAX_EAGER_GHOSTS` ghost points), since that then outweighs lazy mode's slower
boundary cells; eager otherwise, and always eager with antiderivatives, which lazy mode does not
support.
"""
function _default_lazy(sizes::NTuple{N,Int}, kernels::NTuple{N,Symbol},
                       derivatives::NTuple{N,Int}) where {N}
    any(d -> derivatives[d] < 0, 1:N) && return false    # antiderivatives need eager mode
    # counts in Float64, so that no product of axis lengths can overflow
    data = prod(d -> Float64(sizes[d]), 1:N)             # the data values, copied by eager
    total = prod(d -> Float64(sizes[d] + 2 * (get(DEGREE_TO_EQUATIONS, kernels[d], 1) - 1)), 1:N)
    ghosts = total - data                                # the ghost points eager computes
    return total > LAZY_MAX_EAGER_COEFFICIENTS || ghosts > LAZY_MAX_EAGER_GHOSTS
end