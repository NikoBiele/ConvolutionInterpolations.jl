using ConvolutionInterpolations
using CairoMakie
using Interpolations
using FastChebInterp
using LaTeXStrings
using Scratch
# Scratch.clear_scratchspaces!()

G = Float64
runge(x) = G(1)./G(G(1)+25*x^2)
xf = [G(-1) + 2*G(i)/G(11_000) for i in 0:11_000]
yf = runge.(xf)

n = 100
ns = unique(trunc.(Int, 10.0 .^([1 + 3.0 * i/(n-1) for i in 0:(n-1)])))
subgrid = :cubic # :linear
cerrors = zeros(G, length(ns))
cubic_spline_errors = zeros(G, length(ns))
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
    xc = G.(chebpoints(n, -1,1))
    c = chebinterp(runge.(xc), -1,1)
    xr = [G(-1.0)+G(2*i)/G(n-1) for i in 0:(n-1)]
    xr_spline = range(-1.0, 1.0, length=n)
    cubic_spline = Interpolations.cubic_spline_interpolation(xr_spline, runge.(xr))
    itpa4_fast = convolution_interpolation(xr, runge.(xr); kernel=:a4, fast=true, subgrid=subgrid, bc=:poly);
    itpb5_fast = convolution_interpolation(xr, runge.(xr); kernel=:b5, fast=true, subgrid=subgrid, bc=:poly);
    itpb7_fast = convolution_interpolation(xr, runge.(xr); kernel=:b7, fast=true, subgrid=subgrid, bc=:poly);
    itpb9_fast = convolution_interpolation(xr, runge.(xr); kernel=:b9, fast=true, subgrid=subgrid, bc=:poly);
    itpb11_fast = convolution_interpolation(xr, runge.(xr); kernel=:b11, fast=true, subgrid=subgrid, bc=:poly);
    itpb13_fast = convolution_interpolation(xr, runge.(xr); kernel=:b13, fast=true, subgrid=subgrid, bc=:poly);

    cerrors[count] = maximum(abs.(c.(xf) - yf))
    cubic_spline_errors[count] = maximum(abs.(cubic_spline.(xf) - yf))
    itpa4_fast_errors[count] = maximum(abs.(itpa4_fast.(xf) - yf))
    itpb5_fast_errors[count] = maximum(abs.(itpb5_fast.(xf) - yf))
    itpb7_fast_errors[count] = maximum(abs.(itpb7_fast.(xf) - yf))
    itpb9_fast_errors[count] = maximum(abs.(itpb9_fast.(xf) - yf))
    itpb11_fast_errors[count] = maximum(abs.(itpb11_fast.(xf) - yf))
    itpb13_fast_errors[count] = maximum(abs.(itpb13_fast.(xf) - yf))
end

fig = Makie.Figure(; size=(1200, 800))
ax = Makie.Axis(fig[1, 1],
    xlabel = L"\text{Total Number of Sample Points}",
    ylabel = L"\text{Maximum Absolute Error}",
    title = L"Convergence of interpolation on $f(x) = 1/(1 + 25x²),\; x \in [-1;1]$",
    titlesize = 24,
    xlabelsize = 20,
    ylabelsize = 20,
    xscale = log10,
    yscale = log10,
    limits = ((10, 10^4), (1e-16, 1e0))
)

# Plot the lines
# thicker lines with markers
Makie.lines!(ax, ns, cubic_spline_errors, label="Cubic Spline", linewidth=3, color=:orange)
Makie.lines!(ax, ns, itpa4_fast_errors, label="':a4' (fast)", linewidth=3, color=:green, linestyle=:solid)
Makie.lines!(ax, ns, itpb5_fast_errors, label="':b5' (fast)", linewidth=3, color=:red, linestyle=:solid)
Makie.lines!(ax, ns, itpb7_fast_errors, label="':b7' (fast)", linestyle=:solid)
Makie.lines!(ax, ns, itpb9_fast_errors, label="':b9' (fast)", linestyle=:solid)
Makie.lines!(ax, ns, itpb11_fast_errors, label="':b11' (fast)", linestyle=:solid)
Makie.lines!(ax, ns, itpb13_fast_errors, label="':b13' (fast)", linestyle=:solid)
Makie.lines!(ax, ns, cerrors, label="Chebyshev", linewidth=3, color=:blue)

Makie.lines!(ax, [10, 10000], [1e-5*(1000/100)^7, 1e-5*(100/10000)^7],
       color=:black, linestyle=:dash, label=    L"\mathcal{O}(h^7)", linewidth=3)

# Add legend
Legend(fig[1, 2], ax, labelsize=20)

fig

# save("fig/convergence_interpolation_1d_runge.png", fig, px_per_unit=3)