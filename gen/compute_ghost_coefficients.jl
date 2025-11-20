"""
Compute Ghost Point Coefficients for Any Convolution Kernel

This script determines the polynomial reproduction degree of a kernel and computes
the optimal ghost point coefficients via Vandermonde system solution.

Usage:
    result = compute_kernel_ghost_coeffs(kernel_function, support_radius)
    
Example:
    coeffs = compute_kernel_ghost_coeffs(my_kernel, 3)
"""

using LinearAlgebra

"""
    test_polynomial_reproduction(kernel::Function, support_radius::Int; max_degree=13, verbose=true)

Determine the polynomial reproduction degree of a kernel.

# Arguments
- `kernel::Function`: Kernel function that takes a distance s and returns weight
- `support_radius::Int`: Half-width of kernel support (e.g., 3 for support [-3,3])
- `max_degree::Int`: Maximum polynomial degree to test (default: 10)
- `verbose::Bool`: Print detailed output (default: true)

# Returns
- `Int`: Highest polynomial degree that the kernel reproduces exactly
"""
function test_polynomial_reproduction(kernel::Function, support_radius::Int; max_degree=13, verbose=true)
    if verbose
        println("\n" * "="^80)
        println("Testing polynomial reproduction (support ±$support_radius)")
        println("="^80)
    end
    
    # Test at a non-integer position to avoid trivial interpolation
    x_test = 0.5
    
    for degree in 0:max_degree
        # Sample polynomial at integer grid points
        positions = -support_radius:support_radius
        values = [i^degree for i in positions]
        
        # Kernel weights at distances from x_test
        weights = [kernel(abs(i - x_test)) for i in positions]
        
        # Interpolation at x_test
        result = sum(w * v for (w, v) in zip(weights, values))
        expected = x_test^degree
        error = abs(result - expected)
        threshold = max(1e-3, abs(expected) * 1e-3)  # Relative tolerance
        
        if verbose
            status = error > threshold && error/max(abs(expected), 1e-15) > 1e-3 ? "✓" : "✗"
            println("  Degree $degree: error = $(round(error, sigdigits=1)) $status")
        end
        
        if error > threshold && error/max(abs(expected), 1e-15) > 1e-3
            if verbose
                println("\nKernel reproduces polynomials up to degree $(degree - 1)")
            end
            return degree - 1
        end
    end
    
    if verbose
        println("\nKernel reproduces polynomials up to at least degree $max_degree")
    end
    return max_degree
end

"""
    compute_ghost_coefficients(poly_degree::Int, num_ghost_points::Int; verbose=true)

Compute universal ghost point coefficients via Vandermonde system.

# Arguments
- `poly_degree::Int`: Polynomial reproduction degree of the kernel
- `num_ghost_points::Int`: Number of ghost points needed (usually = support radius)
- `verbose::Bool`: Print detailed output (default: true)

# Returns
- `Matrix{Float64}`: Coefficient matrix where row j contains coefficients for g_{-j}
"""
function compute_ghost_coefficients(poly_degree::Int, num_ghost_points::Int; verbose=true)
    n = poly_degree
    m = n + 1  # Number of interior points to use
    
    if verbose
        println("\n" * "="^80)
        println("Computing ghost point coefficients")
        println("="^80)
        println("  Polynomial degree: $n")
        println("  Interior points used: $m")
        println("  Ghost points needed: $num_ghost_points")
    end
    
    # Build Vandermonde matrix
    V = zeros(n + 1, m)
    for i in 0:n
        for j in 0:(m-1)
            V[i+1, j+1] = j^i
        end
    end
    
    if verbose
        println("\nVandermonde matrix V:")
        display(V)
        println()
    end
    
    # Solve for each ghost point
    ghost_matrix = zeros(num_ghost_points, m)
    
    for k in 1:num_ghost_points
        # Right-hand side: polynomial values at position -k
        b = [(-k)^i for i in 0:n]
        
        # Solve V * α = b
        α = V \ b
        ghost_matrix[k, :] = α
        
        if verbose
            println("Ghost point g_{-$k}:")
            println("  α = $α")
            
            # Format as equation
            terms = ["$(round(α[j], digits=1))*f$(j-1)" for j in 1:m]
            eq = join(terms, " + ")
            eq = replace(eq, "+ -" => "- ")
            println("  g_{-$k} = $eq")
            println()
        end
    end
    
    return ghost_matrix
end

"""
    compute_kernel_ghost_coeffs(kernel::Function, support_radius::Int; 
                                max_degree=13, verbose=true)

Complete pipeline: test polynomial reproduction and compute ghost point coefficients.

# Arguments
- `kernel::Function`: Kernel function that takes distance s and returns weight
- `support_radius::Int`: Half-width of kernel support
- `max_degree::Int`: Maximum polynomial degree to test (default: 10)
- `verbose::Bool`: Print detailed output (default: true)

# Returns
- Named tuple with fields:
  - `poly_degree::Int`: Polynomial reproduction degree
  - `ghost_matrix::Matrix{Float64}`: Ghost point coefficient matrix
  - `num_interior::Int`: Number of interior points used
  - `num_ghost::Int`: Number of ghost points

# Examples
```julia
# Define the kernel
function my_kernel(s)
    # ... kernel implementation
end

# Compute ghost point coefficients
result = compute_kernel_ghost_coeffs(my_kernel, 3)

# Access results
println("Polynomial degree: ", result.poly_degree)
println("Ghost matrix: ", result.ghost_matrix)

# Copy-paste ready for polynomial_boundary.jl
print_julia_code(result)
```
"""
function compute_kernel_ghost_coeffs(kernel::Function, support_radius::Int; 
                                    max_degree=13, verbose=true)
    # Step 1: Determine polynomial reproduction degree
    poly_degree = test_polynomial_reproduction(kernel, support_radius; 
                                              max_degree=max_degree, 
                                              verbose=verbose)
    
    # Step 2: Compute ghost point coefficients
    num_ghost_points = support_radius
    ghost_matrix = compute_ghost_coefficients(poly_degree, num_ghost_points; 
                                             verbose=verbose)
    
    # Step 3: Validate on test polynomials
    if verbose
        println("\n" * "="^80)
        println("VALIDATION: Testing on polynomials")
        println("="^80)
        
        # Test on a polynomial of the reproduction degree
        test_degree = poly_degree
        x_data = collect(0:9)
        y_data = x_data.^test_degree
        
        println("\nTest polynomial: p(x) = x^$test_degree")
        println("Data at x=0,1,2,...: $(y_data[1:5])...")
        
        # Compute ghost points
        y_mean = sum(y_data) / length(y_data)
        y_centered = y_data .- y_mean
        
        num_interior = size(ghost_matrix, 2)
        println("\nComputed ghost points:")
        
        for j in 1:num_ghost_points
            ghost_val = sum(ghost_matrix[j, k] * y_centered[k] for k in 1:num_interior)
            ghost_val += y_mean
            expected = (-j)^test_degree
            error = abs(ghost_val - expected)
            threshold = max(1e-3, abs(expected) * 1e-3)  # Relative tolerance
            status = error > threshold && error/max(abs(expected), 1e-15) > 1e-3 ? "✓" : "✗"
            println("  g_{-$j} = $(round(ghost_val, digits=1)), expected = $expected, error = $(round(error, sigdigits=2)) $status")
        end
    end
    
    return (
        poly_degree = poly_degree,
        ghost_matrix = ghost_matrix,
        num_interior = size(ghost_matrix, 2),
        num_ghost = size(ghost_matrix, 1)
    )
end

"""
    print_julia_code(result; kernel_name="my_kernel")

Print Julia code ready to copy-paste into polynomial_boundary.jl

# Arguments
- `result`: Output from compute_kernel_ghost_coeffs
- `kernel_name::String`: Name for the kernel (default: "my_kernel")
"""
function print_julia_code(result; kernel_name="my_kernel")
    println("\n" * "="^80)
    println("COPY-PASTE READY CODE FOR polynomial_boundary.jl")
    println("="^80)
    println()
    
    # Print comment
    println("    # $kernel_name: Reproduces polynomials up to degree $(result.poly_degree)")
    println("    # Support: [$(-(result.num_ghost)), $(result.num_ghost)], needs $(result.num_ghost) ghost points")
    println("    # Uses $(result.num_interior) interior points (degree + 1)")
    
    # Print matrix with proper formatting
    println("    :$(kernel_name) => [")
    
    for j in 1:result.num_ghost
        row = result.ghost_matrix[j, :]
        
        # Format each element to avoid scientific notation for clean integers
        formatted = map(row) do val
            if abs(val - round(val)) < 1e-3  # Nearly integer
                return string(Int(round(val))) * ".0"
            else
                return string(round(val, digits=1))
            end
        end
        
        row_str = join(formatted, "  ")
        
        # Add comment
        comment = "# g_{-$j}"
        if j < result.num_ghost
            println("        $row_str;  $comment")
        else
            println("        $row_str;   $comment")
        end
    end
    
    println("    ],")
    println()
end

"""
    format_ghost_matrix(ghost_matrix::Matrix; kernel_name="kernel")

Format ghost matrix as a clean Julia array literal.

# Arguments
- `ghost_matrix::Matrix`: Ghost point coefficient matrix
- `kernel_name::String`: Name for display purposes
"""
function format_ghost_matrix(ghost_matrix::Matrix; kernel_name="kernel")
    println("\nFormatted ghost matrix for $kernel_name:")
    println("[")
    for i in 1:size(ghost_matrix, 1)
        row = ghost_matrix[i, :]
        formatted = map(row) do val
            abs(val - round(val)) < 1e-3 ? Int(round(val)) : round(val, digits=1)
        end
        println("  $formatted;  # g_{-$i}")
    end
    println("]")
end

# ==============================================================================
# EXAMPLE USAGE
# ==============================================================================

"""
Example: Compute ghost coefficients for a custom kernel
"""
function example_usage()
    println("="^80)
    println("EXAMPLE: Computing ghost coefficients for a kernel")
    println("="^80)
    
    # Example: A5 kernel
    function a5_kernel(s::Real)
        s_abs = abs(s)
        a = 3/64
        
        if s_abs < 1.0
            coef = [1, 0, 8*a-5/2, 0, -18*a+45/16, 10*a-21/16]
            return sum(coef[i+1] * s_abs^i for i in 0:5)
        elseif s_abs < 2.0
            coef = [-66*a+5, 265*a-15, -392*a+35/2, 270*a-10, -88*a+45/16, 11*a-5/16]
            return sum(coef[i+1] * s_abs^i for i in 0:5)
        elseif s_abs < 3.0
            coef = [-162*a, 297*a, -216*a, 78*a, -14*a, a]
            return sum(coef[i+1] * s_abs^i for i in 0:5)
        else
            return zero(T)
        end
    end
    
    # Compute ghost coefficients
    result = compute_kernel_ghost_coeffs(a5_kernel, 3)
    
    # Print copy-paste ready code
    print_julia_code(result; kernel_name="a5")
    
    return result
end

# Run example if script is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    example_usage()
end

println("\n" * "="^80)
println("READY TO USE!")
println("="^80)
println("\nTo compute ghost coefficients for a kernel:")
println("  1. Define the kernel function")
println("  2. Call: result = compute_kernel_ghost_coeffs(kernel, support_radius)")
println("  3. Call: print_julia_code(result, kernel_name=\"kernel_name\")")
println("  4. Copy-paste the output into polynomial_boundary.jl")
println()