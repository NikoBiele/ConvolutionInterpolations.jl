using ConvolutionInterpolations
using CairoMakie
using Chairmarks
using Printf
using Scratch
# Scratch.clear_scratchspaces!()

# --- Benchmark setup ---
kernels = [:a0, :a1, :a3, :a4, :a5, :a7, :b5, :b7, :b9, :b11, :b13]
kernel_labels = string.(kernels)
dims = [1, 2, 3, 4]
N = 10  # grid points per dimension

init_times = zeros(length(kernels), length(dims))   # in μs
eval_times = zeros(length(kernels), length(dims))    # in ns

for (di, d) in enumerate(dims)
    println("Benchmarking dimension $d...")
    
    # Create grid and data
    knots = ntuple(_ -> range(0.0, 1.0, length=N), d)
    data = rand(ntuple(_ -> N, d)...)
    
    for (ki, kernel) in enumerate(kernels)
        print("  $kernel... ")
        
        # Benchmark initialization
        b_init = @b convolution_interpolation($knots, $data; degree=$(kernel), subgrid=:linear, kernel_bc=:linear)
        init_times[ki, di] = b_init.time / 1e-6  # s → μs
        
        # Benchmark evaluation
        itp = convolution_interpolation(knots, data; degree=kernel, subgrid=:linear)
        point = ntuple(_ -> 0.5, d)
        b_eval = nothing
        if d == 1
            b_eval = @b $itp.itp($point[1])
        elseif d == 2
            b_eval = @b $itp.itp($point[1], $point[2])
        elseif d == 3
            b_eval = @b $itp.itp($point[1], $point[2], $point[3])
        else
            b_eval = @b $itp.itp($point[1], $point[2], $point[3], $point[4])
        end
        eval_times[ki, di] = b_eval.time / 1e-9  # s → ns
        
        # Check allocations
        allocs = b_eval.allocs
        bytes = b_eval.bytes
        
        @printf("init: %.0f μs, eval: %.0f ns, allocs: %d (%d bytes)\n", 
                init_times[ki, di], eval_times[ki, di], allocs, bytes)
    end
end

# --- Plotting ---
fig = Figure(size=(1400, 600))

for (col, (matrix, title_str, unit)) in enumerate([
    (init_times, "Initialization Time (μs)", "μs"),
    (eval_times, "Single Evaluation Time (ns)", "ns")
])
    ax = Axis(fig[1, col],
        title=title_str,
        xlabel="Dimension",
        ylabel="Kernel",
        xticks=(1:4, string.(dims)),
        yticks=(1:length(kernels), kernel_labels),
        yreversed=false
    )
    
    # Log-scale colors
    log_matrix = log10.(matrix)
    
    hm = heatmap!(ax, 1:length(dims), 1:length(kernels), log_matrix',
        colormap=:viridis,
        colorrange=(minimum(log_matrix), maximum(log_matrix))
    )
    
    # Add text labels with outline
    for ki in 1:length(kernels)
        for di in 1:length(dims)
            val = matrix[ki, di]
            label = @sprintf("%.0f", val)
            
            text!(ax, di, ki; text=label,
                align=(:center, :center),
                fontsize=32,
                color=:white,
                strokewidth=1.5,
                strokecolor=:black)
        end
    end
    
end

fig

# save("fig/kernel_performance_comparison.png", fig, px_per_unit=3.0)