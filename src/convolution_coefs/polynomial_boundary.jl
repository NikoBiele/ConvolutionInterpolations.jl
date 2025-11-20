"""
    POLYNOMIAL_GHOST_COEFFS

Dictionary containing precomputed coefficient matrices for polynomial-reproduction boundary conditions.

# Structure
Maps kernel symbols to integer coefficient matrices where each row contains the coefficients
for computing one ghost point from interior values.

For a kernel of type `:kernel`, `POLYNOMIAL_GHOST_COEFFS[:kernel]` is a matrix where:
- Row `j` contains coefficients for ghost point `g_{-j}` (left boundary)
- Each coefficient multiplies consecutive interior points starting from the boundary

# Available Kernels
- `:a3`: Cubic reproduction, 2 ghost points, uses 4 interior points
- `:a5`: Cubic reproduction, 3 ghost points, uses 4 interior points
- `:a7`: Cubic reproduction, 4 ghost points, uses 4 interior points
- `:b3`: Cubic reproduction, 3 ghost points, uses 4 interior points
- `:b5`: Quintic reproduction, 5 ghost points, uses 6 interior points
- `:b7`: Septic reproduction, 6 ghost points, uses 8 interior points
- `:b9`: 7th-order convergence, 7 ghost points, uses 8 interior points
- `:b11`: 7th-order convergence, 8 ghost points, uses 8 interior points
- `:b13`: 7th-order convergence, 9 ghost points, uses 8 interior points

# Computation Method
Coefficients are computed by solving Vandermonde systems to ensure exact reproduction
of polynomials up to the specified degree. For example, `:b5` coefficients guarantee
that quintic polynomials are interpolated exactly even near boundaries.

# Usage
These coefficients are used internally by the boundary condition system. The ghost point
computation is: `g_{-j} = sum(coef[j,k] * f[k] for k in 1:num_interior)`

See also: `get_polynomial_ghost_coeffs`, `fill_ghost_points_polynomial!`.
"""

const POLYNOMIAL_GHOST_COEFFS = Dict{Symbol, Matrix{Int}}(
    
    # a3: Reproduces polynomials up to degree 3
    # Support: [-2, 2], needs 2 ghost points
    # Uses 4 interior points (degree + 1)
    :a3 => [
        4  -6  4  -1;  # g_{-1}
        10  -20  15  -4;   # g_{-2}
    ],

    # A5 Kernel: Reproduces cubics (degree 3)
    # Support: [-3, 3], needs 3 ghost points
    # Uses 4 interior points (degree + 1)
    :a5 => [
        4  -6   4  -1;     # g_{-1} = 4*f0 - 6*f1 + 4*f2 - 1*f3
        10 -20  15  -4;    # g_{-2} = 10*f0 - 20*f1 + 15*f2 - 4*f3
        20 -45  36 -10;    # g_{-3} = 20*f0 - 45*f1 + 36*f2 - 10*f3
    ],
    
    # a7: Reproduces polynomials up to degree 3
    # Support: [-4, 4], needs 4 ghost points
    # Uses 4 interior points (degree + 1)
    :a7 => [
        4  -6  4  -1;  # g_{-1}
        10  -20  15  -4;  # g_{-2}
        20  -45  36  -10;  # g_{-3}
        35  -84  70  -20;   # g_{-4}
    ],

    # b3: Reproduces polynomials up to degree 3
    # Support: [-3, 3], needs 3 ghost points
    # Uses 4 interior points (degree + 1)
    :b3 => [
        4  -6  4  -1;  # g_{-1}
        10  -20  15  -4;  # g_{-2}
        20  -45  36  -10;   # g_{-3}
    ],
    
    # B5 Kernel: Reproduces quintics (degree 5)
    # Support: [-5, 5], needs 5 ghost points
    # Uses 6 interior points (degree + 1)
    :b5 => [
        6    -15    20   -15    6   -1;      # g_{-1}
        21   -70   105   -84   35   -6;      # g_{-2}
        56  -210   336  -280  120  -21;      # g_{-3}
        126 -504   840  -720  315  -56;      # g_{-4}
        252 -1050  1800 -1575  700 -126;     # g_{-5}
    ],
    
    # :b7: Reproduces polynomials up to degree 7
    # Support: [-6, 6], needs 6 ghost points
    # Uses 8 interior points (degree + 1)
    :b7 => [
        8  -28  56  -70  56  -28  8  -1;  # g_{-1}
        36  -168  378  -504  420  -216  63  -8;  # g_{-2}
        120  -630  1512  -2100  1800  -945  280  -36;  # g_{-3}
        330  -1848  4620  -6600  5775  -3080  924  -120;  # g_{-4}
        792  -4620  11880  -17325  15400  -8316  2520  -330;  # g_{-5}
        1716  -10296  27027  -40040  36036  -19656  6006  -792;   # g_{-6}
    ],

    # :b9: Reproduces polynomials up to degree 7
    # Support: [-7, 7], needs 7 ghost points
    # Uses 8 interior points (degree + 1)
    :b9 => [
        8  -28  56  -70  56  -28  8  -1;  # g_{-1}
        36  -168  378  -504  420  -216  63  -8;  # g_{-2}
        120  -630  1512  -2100  1800  -945  280  -36;  # g_{-3}
        330  -1848  4620  -6600  5775  -3080  924  -120;  # g_{-4}
        792  -4620  11880  -17325  15400  -8316  2520  -330;  # g_{-5}
        1716  -10296  27027  -40040  36036  -19656  6006  -792;  # g_{-6}
        3432  -21021  56056  -84084  76440  -42042  12936  -1716;   # g_{-7}
    ],

    # :b11: Reproduces polynomials up to degree 7
    # Support: [-8, 8], needs 8 ghost points
    # Uses 8 interior points (degree + 1)
    :b11 => [
        8  -28  56  -70  56  -28  8  -1;  # g_{-1}
        36  -168  378  -504  420  -216  63  -8;  # g_{-2}
        120  -630  1512  -2100  1800  -945  280  -36;  # g_{-3}
        330  -1848  4620  -6600  5775  -3080  924  -120;  # g_{-4}
        792  -4620  11880  -17325  15400  -8316  2520  -330;  # g_{-5}
        1716  -10296  27027  -40040  36036  -19656  6006  -792;  # g_{-6}
        3432  -21021  56056  -84084  76440  -42042  12936  -1716;  # g_{-7}
        6435  -40040  108108  -163800  150150  -83160  25740  -3432;   # g_{-8}
    ],

    # :b13: Reproduces polynomials up to degree 7
    # Support: [-9, 9], needs 9 ghost points
    # Uses 8 interior points (degree + 1)
    :b13 => [
        8  -28  56  -70  56  -28  8  -1;  # g_{-1}
        36  -168  378  -504  420  -216  63  -8;  # g_{-2}
        120  -630  1512  -2100  1800  -945  280  -36;  # g_{-3}
        330  -1848  4620  -6600  5775  -3080  924  -120;  # g_{-4}
        792  -4620  11880  -17325  15400  -8316  2520  -330;  # g_{-5}
        1716  -10296  27027  -40040  36036  -19656  6006  -792;  # g_{-6}
        3432  -21021  56056  -84084  76440  -42042  12936  -1716;  # g_{-7}
        6435  -40040  108108  -163800  150150  -83160  25740  -3432;  # g_{-8}
        11440  -72072  196560  -300300  277200  -154440  48048  -6435;   # g_{-9}
    ]
)

"""
    get_polynomial_ghost_coeffs(kernel_type::Symbol)

Retrieve precomputed polynomial-reproduction ghost point coefficients for a given kernel.

# Arguments
- `kernel_type::Symbol`: Kernel identifier (`:a3`, `:b5`, `:b7`, etc.)

# Returns
Integer coefficient matrix where row `j` contains coefficients for computing ghost point
`g_{-j}` from interior values. The number of rows equals the number of ghost points needed
(kernel support radius), and the number of columns equals the number of interior points
used for the computation.

# Throws
`ArgumentError` if `kernel_type` is not available in `POLYNOMIAL_GHOST_COEFFS`

# Examples
```julia
# Get coefficients for b5 kernel
coefs = get_polynomial_ghost_coeffs(:b5)
# Returns a 5×6 matrix for computing 5 ghost points from 6 interior points

# Compute first ghost point for some data
f = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]  # Interior points
g_minus_1 = sum(coefs[1, k] * f[k] for k in 1:6)
```

See also: `POLYNOMIAL_GHOST_COEFFS`, `boundary_coefs`.
"""

function get_polynomial_ghost_coeffs(kernel_type::Symbol)
    if !haskey(POLYNOMIAL_GHOST_COEFFS, kernel_type)
        available = join(keys(POLYNOMIAL_GHOST_COEFFS), ", ")
        throw(ArgumentError(
            "Unknown kernel type: $kernel_type. " *
            "Available polynomial-reproduction coefficients: $available"
        ))
    end
    return POLYNOMIAL_GHOST_COEFFS[kernel_type]
end