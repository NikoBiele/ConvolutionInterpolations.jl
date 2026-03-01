using ConvolutionInterpolations
using CairoMakie
# using Interpolations
# using FastChebInterp
using LaTeXStrings
using Scratch
# Scratch.clear_scratchspaces!()

G = BigFloat # Float64
runge(x) = G(1)./G(G(1)+25*x^2)
xf = [G(-1) + 2*G(i)/G(11_000) for i in 0:11_000]
yf = runge.(xf)

n = 100
ns = unique(trunc.(Int, 10.0 .^([1 + 3.0 * i/(n-1) for i in 0:(n-1)])))
# cerrors = zeros(G, length(ns))
# cubic_spline_errors = zeros(G, length(ns))
itpb5_fast_errors = zeros(G, length(ns))
itpb7_fast_errors = zeros(G, length(ns))
itpb9_fast_errors = zeros(G, length(ns))
itpb11_fast_errors = zeros(G, length(ns))
itpb13_fast_errors = zeros(G, length(ns))

count = 0
for n in ns
    count += 1
    println("n = $n")
    # xc = G.(chebpoints(n, -1,1))
    # c = chebinterp(runge.(xc), -1,1)
    xr = [G(-1.0)+G(2*i)/G(n-1) for i in 0:(n-1)]
    # xr_spline = range(-1.0, 1.0, length=n)
    # cubic_spline = Interpolations.cubic_spline_interpolation(xr_spline, runge.(xr))
    itpb5_fast = convolution_interpolation(xr, runge.(xr); degree=:b5, fast=true, subgrid=:cubic, kernel_bc=[(:polynomial,:polynomial)]);
    itpb7_fast = convolution_interpolation(xr, runge.(xr); degree=:b7, fast=true, subgrid=:cubic, kernel_bc=[(:polynomial,:polynomial)]);
    itpb9_fast = convolution_interpolation(xr, runge.(xr); degree=:b9, fast=true, subgrid=:cubic, kernel_bc=[(:polynomial,:polynomial)]);
    itpb11_fast = convolution_interpolation(xr, runge.(xr); degree=:b11, fast=true, subgrid=:cubic, kernel_bc=[(:polynomial,:polynomial)]);
    itpb13_fast = convolution_interpolation(xr, runge.(xr); degree=:b13, fast=true, subgrid=:cubic, kernel_bc=[(:polynomial,:polynomial)]);

    # cerrors[count] = maximum(abs.(c.(xf) - yf))
    # cubic_spline_errors[count] = maximum(abs.(cubic_spline.(xf) - yf))
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
    limits = ((10, 10^4), (1e-20, 1e0))
)

# Plot the lines
# thicker lines with markers
# Makie.lines!(ax, ns, cubic_spline_errors, label="Cubic Spline", linewidth=3, color=:orange)
# Makie.lines!(ax, ns, cerrors, label="Chebyshev", linewidth=3, color=:blue)
Makie.lines!(ax, ns, itpb5_fast_errors, label="Convolution 5th (b) (fast)", linewidth=3, color=:green, linestyle=:solid)
Makie.lines!(ax, ns, itpb7_fast_errors, label="Convolution 7th (b) (fast)", linestyle=:solid)
Makie.lines!(ax, ns, itpb9_fast_errors, label="Convolution 9th (b) (fast)", linestyle=:solid)
Makie.lines!(ax, ns, itpb11_fast_errors, label="Convolution 11th (b) (fast)", linestyle=:solid)
Makie.lines!(ax, ns, itpb13_fast_errors, label="Convolution 13th (b) (fast)", linestyle=:solid)

Makie.lines!(ax, [10, 10000], [1e-5*(1000/100)^7, 1e-5*(100/10000)^7],
       color=:black, linestyle=:dash, label=    L"\mathcal{O}(h^7)", linewidth=3)

# Add legend
Legend(fig[1, 2], ax, fontsize=20)

fig

# save("fig/interpolation_1d_runge.png", fig, px_per_unit=3)