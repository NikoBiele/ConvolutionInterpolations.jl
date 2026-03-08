using ConvolutionInterpolations
using CairoMakie
using LaTeXStrings
using Scratch
# Scratch.clear_scratchspaces!()

G = Float64
runge(x) = G(1)./G(G(1)+25*x^2)
runge_antideriv(x) = 1/5 * atan(5*x) + 1/5 * atan(5)
xf = [G(-1) + 2*G(i)/G(11_000) for i in 0:11_000]
yf = runge.(xf)
using Plots
Plots.plot(xf, yf)
n = 100
ns = unique(trunc.(Int, 10.0 .^([1 + 3.0 * i/(n-1) for i in 0:(n-1)])))
subgrid = :cubic # :linear
precompute = 101 # 10_000
itpa4_fast_errors = zeros(G, length(ns))
itpb5_fast_errors = zeros(G, length(ns))
itpb7_fast_errors = zeros(G, length(ns))
itpb9_fast_errors = zeros(G, length(ns))
itpb11_fast_errors = zeros(G, length(ns))
itpb13_fast_errors = zeros(G, length(ns))

count = 0
for n in ns
    count += 1
    println("n = $n")
    xr = [G(-1.0)+G(2*i)/G(n-1) for i in 0:(n-1)]
    itpa4_fast = convolution_interpolation(xr, runge.(xr); degree=:a4, fast=true, derivative=-1, subgrid=subgrid, precompute=precompute, kernel_bc=[(:polynomial,:polynomial)]);
    itpb5_fast = convolution_interpolation(xr, runge.(xr); degree=:b5, fast=true, derivative=-1, subgrid=subgrid, precompute=precompute, kernel_bc=[(:polynomial,:polynomial)]);
    itpb7_fast = convolution_interpolation(xr, runge.(xr); degree=:b7, fast=true, derivative=-1, subgrid=subgrid, precompute=precompute, kernel_bc=[(:polynomial,:polynomial)]);
    itpb9_fast = convolution_interpolation(xr, runge.(xr); degree=:b9, fast=true, derivative=-1, subgrid=subgrid, precompute=precompute, kernel_bc=[(:polynomial,:polynomial)]);
    itpb11_fast = convolution_interpolation(xr, runge.(xr); degree=:b11, fast=true, derivative=-1, subgrid=subgrid, precompute=precompute, kernel_bc=[(:polynomial,:polynomial)]);
    itpb13_fast = convolution_interpolation(xr, runge.(xr); degree=:b13, fast=true, derivative=-1, subgrid=subgrid, precompute=precompute, kernel_bc=[(:polynomial,:polynomial)]);

    itpa4_fast_errors[count] = maximum(abs.(itpa4_fast.(xf) - runge_antideriv.(xf)))
    itpb5_fast_errors[count] = maximum(abs.(itpb5_fast.(xf) - runge_antideriv.(xf)))
    itpb7_fast_errors[count] = maximum(abs.(itpb7_fast.(xf) - runge_antideriv.(xf)))
    itpb9_fast_errors[count] = maximum(abs.(itpb9_fast.(xf) - runge_antideriv.(xf)))
    itpb11_fast_errors[count] = maximum(abs.(itpb11_fast.(xf) - runge_antideriv.(xf)))
    itpb13_fast_errors[count] = maximum(abs.(itpb13_fast.(xf) - runge_antideriv.(xf)))
end

fig = Makie.Figure(; size=(1200, 800))
ax = Makie.Axis(fig[1, 1],
    xlabel = L"\text{Total Number of Sample Points}",
    ylabel = L"\text{Maximum Absolute Error}",
    title = L"Convergence of integration of $f(x) = 1/(1 + 25x²),\; x \in [-1;1]$",
    titlesize = 24,
    xlabelsize = 20,
    ylabelsize = 20,
    xscale = log10,
    yscale = log10,
    limits = ((10, 10^4), (1e-16, 1e0))
)

# Plot the lines
# thicker lines with markers
Makie.lines!(ax, ns, itpa4_fast_errors, label="Convolution ':a4' (fast)", linewidth=3, color=:green, linestyle=:solid)
Makie.lines!(ax, ns, itpb5_fast_errors, label="Convolution ':b5' (fast)", linewidth=3, color=:red, linestyle=:solid)
Makie.lines!(ax, ns, itpb7_fast_errors, label="Convolution ':b7' (fast)", linestyle=:solid)
Makie.lines!(ax, ns, itpb9_fast_errors, label="Convolution ':b9' (fast)", linestyle=:solid)
Makie.lines!(ax, ns, itpb11_fast_errors, label="Convolution ':b11' (fast)", linestyle=:solid)
Makie.lines!(ax, ns, itpb13_fast_errors, label="Convolution ':b13' (fast)", linestyle=:solid)

Makie.lines!(ax, [10, 10000], [1e-5*(1000/100)^7, 1e-5*(100/10000)^7],
       color=:black, linestyle=:dash, label=    L"\mathcal{O}(h^7)", linewidth=3)

# Add legend
Legend(fig[1, 2], ax, labelsize=20)

fig

# save("fig/integration_1d_runge.png", fig, px_per_unit=3)