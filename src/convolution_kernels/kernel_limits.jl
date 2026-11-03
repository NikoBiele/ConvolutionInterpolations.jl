# The highest derivative and integral orders each kernel supports.

# Highest integral order per kernel: m − 1 must not exceed the degree up to which the kernel's
# moments vanish, so that outside the support Kₘ is exactly the integrated step (the far field
# used by the anchoring and the tails). Validated exactly for every kernel up to these orders.
const _max_integral_order = Dict(:a0 => 2, :a1 => 2,
                                 :a3 => 4, :a4 => 4, :a5 => 4, :a7 => 4,
                                 :b5 => 6,
                                 :b7 => 8, :b9 => 8, :b11 => 8, :b13 => 8)

"""
    _max_derivative_order(kernel, nonuniform) -> Int

Highest derivative order a kernel supports on a uniform or a nonuniform grid: the smaller of
the kernel's continuity order and `R - 1`, where `R` is the polynomial degree the kernel
reproduces on that grid. Derivative `k` of the interpolant converges at order `R + 1 - k`,
so `k <= R - 1` keeps at least second-order convergence, and `k` at most the continuity order
keeps the derivative a continuous function. On uniform grids the `b` kernels reproduce
degree 6 (`R - 1 = 5`); on nonuniform grids they reproduce their full degree `p`, so their
continuity `p - 2` is the binding limit. Rounding in the data is amplified by `h^-k`, which
no cap can express: high orders on fine grids need extended precision.
"""
function _max_derivative_order(kernel::Symbol, nonuniform::Bool)
    if nonuniform
        return _max_derivative_nonuniform[kernel]
    end
    return _max_derivative_uniform[kernel]
end

const _max_derivative_uniform = Dict(:a0 => 0, :a1 => 0,
                                     :a3 => 1, :a4 => 1, :a5 => 1, :a7 => 1,
                                     :b5 => 3,
                                     :b7 => 5, :b9 => 5, :b11 => 5, :b13 => 5)

const _max_derivative_nonuniform = Dict(:a1 => 0, :n3 => 0,
                                        :b5 => 3, :b7 => 5, :b9 => 7, :b11 => 9, :b13 => 11)