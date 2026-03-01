# Auto-generated lookup for precomputed kernel tables
# These tables are computed once at exact rational precision and shipped with the package.
# For :linear subgrid (high resolution) and top derivatives, tables are computed and cached on demand.

include("pre_range.jl")
include("kernel_a0.jl")
include("kernel_a1.jl")
include("kernel_a3.jl")
include("kernel_a4.jl")
include("kernel_a5.jl")
include("kernel_a7.jl")
include("kernel_b5.jl")
include("kernel_b7.jl")
include("kernel_b9.jl")
include("kernel_b11.jl")
include("kernel_b13.jl")

function get_shipped_kernel_tables(degree::Symbol, derivative::Int, ::Type{T}) where T
    pre_range = T.(PRE_RANGE_101)
    
    # Special cases: a0 and a1 use 2-point range
    if degree == :a0 || degree == :a1
        pre_range = T.(PRE_RANGE_2)
    end
    if degree == :a0
        if derivative == 0
            kp = T.(KERNEL_PRE_a0_d0)
            kd1 = nothing
            kd2 = nothing
            return pre_range, kp, kd1, kd2
        end
    elseif degree == :a1
        if derivative == 0
            kp = T.(KERNEL_PRE_a1_d0)
            kd1 = nothing
            kd2 = nothing
            return pre_range, kp, kd1, kd2
        end
    elseif degree == :a3
        if derivative == 0
            kp = T.(KERNEL_PRE_a3_d0)
            kd1 = T.(KERNEL_D1_PRE_a3_d0)
            kd2 = nothing
            return pre_range, kp, kd1, kd2
        end
    elseif degree == :a4
        if derivative == 0
            kp = T.(KERNEL_PRE_a4_d0)
            kd1 = T.(KERNEL_D1_PRE_a4_d0)
            kd2 = nothing
            return pre_range, kp, kd1, kd2
        end
    elseif degree == :a5
        if derivative == 0
            kp = T.(KERNEL_PRE_a5_d0)
            kd1 = T.(KERNEL_D1_PRE_a5_d0)
            kd2 = nothing
            return pre_range, kp, kd1, kd2
        end
    elseif degree == :a7
        if derivative == 0
            kp = T.(KERNEL_PRE_a7_d0)
            kd1 = T.(KERNEL_D1_PRE_a7_d0)
            kd2 = nothing
            return pre_range, kp, kd1, kd2
        end
    elseif degree == :b5
        if derivative == 0
            kp = T.(KERNEL_PRE_b5_d0)
            kd1 = T.(KERNEL_D1_PRE_b5_d0)
            kd2 = T.(KERNEL_D2_PRE_b5_d0)
            return pre_range, kp, kd1, kd2
        elseif derivative == 1
            kp = T.(KERNEL_PRE_b5_d1)
            kd1 = T.(KERNEL_D1_PRE_b5_d1)
            kd2 = T.(KERNEL_D2_PRE_b5_d1)
            return pre_range, kp, kd1, kd2
        elseif derivative == 2
            kp = T.(KERNEL_PRE_b5_d2)
            kd1 = T.(KERNEL_D1_PRE_b5_d2)
            kd2 = nothing
            return pre_range, kp, kd1, kd2
        end
    elseif degree == :b7
        if derivative == 0
            kp = T.(KERNEL_PRE_b7_d0)
            kd1 = T.(KERNEL_D1_PRE_b7_d0)
            kd2 = T.(KERNEL_D2_PRE_b7_d0)
            return pre_range, kp, kd1, kd2
        elseif derivative == 1
            kp = T.(KERNEL_PRE_b7_d1)
            kd1 = T.(KERNEL_D1_PRE_b7_d1)
            kd2 = T.(KERNEL_D2_PRE_b7_d1)
            return pre_range, kp, kd1, kd2
        elseif derivative == 2
            kp = T.(KERNEL_PRE_b7_d2)
            kd1 = T.(KERNEL_D1_PRE_b7_d2)
            kd2 = T.(KERNEL_D2_PRE_b7_d2)
            return pre_range, kp, kd1, kd2
        elseif derivative == 3
            kp = T.(KERNEL_PRE_b7_d3)
            kd1 = T.(KERNEL_D1_PRE_b7_d3)
            kd2 = nothing
            return pre_range, kp, kd1, kd2
        end
    elseif degree == :b9
        if derivative == 0
            kp = T.(KERNEL_PRE_b9_d0)
            kd1 = T.(KERNEL_D1_PRE_b9_d0)
            kd2 = T.(KERNEL_D2_PRE_b9_d0)
            return pre_range, kp, kd1, kd2
        elseif derivative == 1
            kp = T.(KERNEL_PRE_b9_d1)
            kd1 = T.(KERNEL_D1_PRE_b9_d1)
            kd2 = T.(KERNEL_D2_PRE_b9_d1)
            return pre_range, kp, kd1, kd2
        elseif derivative == 2
            kp = T.(KERNEL_PRE_b9_d2)
            kd1 = T.(KERNEL_D1_PRE_b9_d2)
            kd2 = T.(KERNEL_D2_PRE_b9_d2)
            return pre_range, kp, kd1, kd2
        elseif derivative == 3
            kp = T.(KERNEL_PRE_b9_d3)
            kd1 = T.(KERNEL_D1_PRE_b9_d3)
            kd2 = T.(KERNEL_D2_PRE_b9_d3)
            return pre_range, kp, kd1, kd2
        elseif derivative == 4
            kp = T.(KERNEL_PRE_b9_d4)
            kd1 = T.(KERNEL_D1_PRE_b9_d4)
            kd2 = nothing
            return pre_range, kp, kd1, kd2
        end
    elseif degree == :b11
        if derivative == 0
            kp = T.(KERNEL_PRE_b11_d0)
            kd1 = T.(KERNEL_D1_PRE_b11_d0)
            kd2 = T.(KERNEL_D2_PRE_b11_d0)
            return pre_range, kp, kd1, kd2
        elseif derivative == 1
            kp = T.(KERNEL_PRE_b11_d1)
            kd1 = T.(KERNEL_D1_PRE_b11_d1)
            kd2 = T.(KERNEL_D2_PRE_b11_d1)
            return pre_range, kp, kd1, kd2
        elseif derivative == 2
            kp = T.(KERNEL_PRE_b11_d2)
            kd1 = T.(KERNEL_D1_PRE_b11_d2)
            kd2 = T.(KERNEL_D2_PRE_b11_d2)
            return pre_range, kp, kd1, kd2
        elseif derivative == 3
            kp = T.(KERNEL_PRE_b11_d3)
            kd1 = T.(KERNEL_D1_PRE_b11_d3)
            kd2 = T.(KERNEL_D2_PRE_b11_d3)
            return pre_range, kp, kd1, kd2
        elseif derivative == 4
            kp = T.(KERNEL_PRE_b11_d4)
            kd1 = T.(KERNEL_D1_PRE_b11_d4)
            kd2 = T.(KERNEL_D2_PRE_b11_d4)
            return pre_range, kp, kd1, kd2
        elseif derivative == 5
            kp = T.(KERNEL_PRE_b11_d5)
            kd1 = T.(KERNEL_D1_PRE_b11_d5)
            kd2 = nothing
            return pre_range, kp, kd1, kd2
        end
    elseif degree == :b13
        if derivative == 0
            kp = T.(KERNEL_PRE_b13_d0)
            kd1 = T.(KERNEL_D1_PRE_b13_d0)
            kd2 = T.(KERNEL_D2_PRE_b13_d0)
            return pre_range, kp, kd1, kd2
        elseif derivative == 1
            kp = T.(KERNEL_PRE_b13_d1)
            kd1 = T.(KERNEL_D1_PRE_b13_d1)
            kd2 = T.(KERNEL_D2_PRE_b13_d1)
            return pre_range, kp, kd1, kd2
        elseif derivative == 2
            kp = T.(KERNEL_PRE_b13_d2)
            kd1 = T.(KERNEL_D1_PRE_b13_d2)
            kd2 = T.(KERNEL_D2_PRE_b13_d2)
            return pre_range, kp, kd1, kd2
        elseif derivative == 3
            kp = T.(KERNEL_PRE_b13_d3)
            kd1 = T.(KERNEL_D1_PRE_b13_d3)
            kd2 = T.(KERNEL_D2_PRE_b13_d3)
            return pre_range, kp, kd1, kd2
        elseif derivative == 4
            kp = T.(KERNEL_PRE_b13_d4)
            kd1 = T.(KERNEL_D1_PRE_b13_d4)
            kd2 = T.(KERNEL_D2_PRE_b13_d4)
            return pre_range, kp, kd1, kd2
        elseif derivative == 5
            kp = T.(KERNEL_PRE_b13_d5)
            kd1 = T.(KERNEL_D1_PRE_b13_d5)
            kd2 = nothing
            return pre_range, kp, kd1, kd2
        end
    end
    error("No shipped kernel table for degree=$degree, derivative=$derivative")
end
