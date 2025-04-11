"""
    boundary_coefs(y, h::T, kernel_bc::Symbol, which_end::Symbol) where T<:Number

Computes boundary coefficients for a given 1D signal slice.

# Arguments
- `y`: Vector representing the signal slice
- `h`: Scaling parameter (of type T<:Number)
- `kernel_bc::Symbol`: Symbol representing the kernel boundary condition
- `which_end::Symbol`: Symbol representing which end of the signal to analyze

# Returns
- Tuple containing:
  1. Vector of prediction coefficients
  2. Mean value of the signal (offset)
  3. Mean-centered signal values

# Details
This function uses the specified boundary condition for the kernel (defaults to detect).
1. 'detect': auto-detects periodic signals and uses appropriate boundary conditions
2. Quadratic prediction `[3.0, -3.0, 1.0]`
3. Linear prediction `[2.0, -1.0, 0.0]`
4. Periodic boundary conditions

The function decomposes the signal into trend (offset) and fluctuation components for more accurate boundary handling.
"""
function boundary_coefs(y, h::T, kernel_bc::Symbol, which_end::Symbol) where T<:Number

    n = length(y)
    y_mean = sum(y)/n
    y_centered = y .- y_mean
    if kernel_bc == :detect
        return detect_boundary_signal_fast(y_centered, n, h; which_end=which_end), y_mean, y_centered
    elseif kernel_bc == :quadratic
        return [3.0, -3.0, 1.0], y_mean, y_centered
    elseif kernel_bc == :linear
        return [2.0, -1.0, 0.0], y_mean, y_centered
    elseif kernel_bc == :periodic
        return periodic_boundary(y_centered), y_mean, y_centered
    else
        error("Unsupported kernel boundary condition: kernel_bc = $(kernel_bc). Supported kernel boundary conditions are :detect, :quadratic, :linear, and :periodic.")
    end
end