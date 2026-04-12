"""
    is_boundary_stencil(cell::Int, n::Int, eqs::Int) -> Bool

Check if cell index `cell` (1-based, in raw data coordinates) has its
evaluation stencil `[cell-(eqs-1) .. cell+eqs]` reaching outside `[1, n]`.

Used to dispatch between the fast interior path (direct array access)
and the slow boundary path (lazy ghost resolution).
"""
@inline function is_boundary_stencil(cell::Int, n::Int, eqs::Int)
    return (cell - (eqs - 1) < 1) | (cell + eqs > n)
end

# Helper to extract kernel symbol from degree type parameter (zero-cost dispatch)
@inline _kernel_sym(::Val{S}) where S = S
@inline _kernel_sym(::HigherOrderKernel{S}) where S = S
@inline _kernel_sym(::LowerOrderKernel{S}) where S = S
@inline _kernel_sym(::HigherOrderMixedKernel{S}) where S = S
@inline _kernel_sym(::LowerOrderMixedKernel{S}) where S = S

# equation tuple helpers
@inline _eqs_d(eqs::Int, d) = eqs
@inline _eqs_d(eqs::Tuple, d) = eqs[d]

# kernel tuple helpers
@inline _kernel_d(kernel, d) = kernel
@inline _kernel_d(kernel::Tuple, d) = kernel[d]