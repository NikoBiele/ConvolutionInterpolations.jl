"""
    boundary_coefs(y, h::T, kernel_bc::Symbol, which_end::Symbol, kernel_type::Symbol=:unknown) where T<:Number

Compute boundary coefficients for handling domain boundaries in convolution interpolation.

# Arguments
- `y`: Vector representing the signal slice near the boundary
- `h`: Grid spacing (of type T<:Number)
- `kernel_bc::Symbol`: Boundary condition method to use
- `which_end::Symbol`: Which boundary (`:left` or `:right`)
- `kernel_type::Symbol`: Kernel degree (`:a3`, `:b5`, etc.) - required for `:polynomial` mode

# Returns
Tuple containing:
1. **Coefficients**: Matrix or Vector depending on method
   - `:polynomial`: Matrix where row j contains coefficients for ghost point g_{-j}
   - Other methods: Vector of prediction coefficients for recursive computation
2. **y_mean**: Mean value of the signal (offset component)
3. **y_centered**: Mean-centered signal values (fluctuation component)

# Boundary Condition Methods
- `:auto` (default): Automatically selects optimal method based on signal and grid size
  - Uses `:polynomial` for grids with sufficient points
  - Falls back to `:detect` for small grids
- `:polynomial`: Optimal polynomial-reproduction boundary conditions (recommended)
  - Kernel-specific coefficients that preserve polynomial reproduction property
  - Non-recursive: directly computes ghost points from interior values
  - Requires `kernel_type` parameter
- `:detect`: Auto-detection of signal characteristics
  - Identifies periodic patterns and applies appropriate treatment
  - Blends autocorrelation-based prediction with quadratic extrapolation
  - Adapts to signal structure automatically
- `:linear`: Linear extrapolation `[2.0, -1.0, 0.0]`
  - Simple first-order accurate boundary
- `:quadratic`: Quadratic extrapolation `[3.0, -3.0, 1.0]`
  - Second-order accurate boundary
- `:periodic`: Advanced periodic signal handling
  - Blends autocorrelation predictor with quadratic extrapolation
  - Optimized for periodic and quasi-periodic signals

# Details
This function handles the creation of "ghost points" outside the computational domain,
which are needed for convolution kernels near domain boundaries. The signal
is decomposed into mean (trend) and centered (fluctuation) components for improved accuracy.

For `:polynomial` mode, returns a coefficient matrix where each row corresponds to one ghost
point, enabling direct computation without recursion. This method preserves the polynomial
reproduction properties of the underlying kernel.

# Examples
```julia
# Optimal polynomial boundary (recommended)
coef_matrix, y_mean, y_centered = boundary_coefs(y, h, :polynomial, :left, :b5)
# coef_matrix[j, :] gives coefficients for ghost point g_{-j}

# Automatic selection
coef, y_mean, y_centered = boundary_coefs(y, h, :auto, :left, :b5)

# Simple linear extrapolation
coef_vec, y_mean, y_centered = boundary_coefs(y, h, :linear, :left)

# Periodic signal handling
coef_vec, y_mean, y_centered = boundary_coefs(y, h, :periodic, :left)
```

See also: `get_polynomial_ghost_coeffs`, `detect_boundary_signal_fast`
"""

function boundary_coefs(y, h::T, kernel_bc::Symbol, which_end::Symbol, kernel_type::Symbol=:unknown) where T<:Number

    n = length(y)
    y_mean = sum(y)/n
    y_centered = y .- y_mean
    ghost_matrix = get_polynomial_ghost_coeffs(kernel_type)
    
    if kernel_bc == :polynomial || (kernel_bc == :auto && length(y) >= size(ghost_matrix, 2))
        return ghost_matrix, y_mean, y_centered
    elseif kernel_bc == :auto && length(y) < size(ghost_matrix, 2)
        return detect_boundary_signal_fast(y_centered, n, h; which_end=which_end), y_mean, y_centered
    else
        if kernel_bc == :detect
            return detect_boundary_signal_fast(y_centered, n, h; which_end=which_end), y_mean, y_centered
        elseif kernel_bc == :quadratic
            return T[3.0, -3.0, 1.0], y_mean, y_centered
        elseif kernel_bc == :linear
            return T[2.0, -1.0, 0.0], y_mean, y_centered
        elseif kernel_bc == :periodic
            return periodic_boundary(y_centered), y_mean, y_centered
        else
            error("Unsupported kernel boundary condition: kernel_bc = $(kernel_bc). Supported kernel boundary conditions are :detect, :quadratic, :linear, :periodic, and :polynomial.")
        end
    end
end