using ConvolutionInterpolations
using CairoMakie
using LaTeXStrings
using Scratch
# Scratch.clear_scratchspaces!()

############ 1D ################

G = Float64
runge(x) = G(1)./G(G(1)+25*x^2)
runge_antideriv(x) = 1/5 * atan(5*x) + 1/5 * atan(5)
xf = [G(-1) + 2*G(i)/G(11_000) for i in 0:11_000]
yf = runge.(xf)
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

######################## 2D ############################

# ── 2D convergence ────────────────────────────────────────────────────────────
using QuadGK

f2 = (x, y) -> 1/(1 + 25*(x^2 + y^2))
f2_int, _ = quadgk(y -> quadgk(x -> f2(x,y), -1, 1, rtol=1e-14)[1], -1, 1, rtol=1e-14)

ns_2d = unique(trunc.(Int, 10.0 .^([1 + 2.0 * i/(n-1) for i in 0:(n-1)])))

itpb5_2d_errors  = zeros(G, length(ns_2d))
itpb7_2d_errors  = zeros(G, length(ns_2d))
itpb9_2d_errors  = zeros(G, length(ns_2d))
itpb11_2d_errors = zeros(G, length(ns_2d))
itpb13_2d_errors = zeros(G, length(ns_2d))

count = 0
for n in ns_2d
    count += 1
    println("2D n = $n")
    xr = [G(-1.0) + G(2*i)/G(n-1) for i in 0:(n-1)]
    vals = [f2(x, y) for x in xr, y in xr]
    for (errors, deg) in zip(
            [itpb5_2d_errors, itpb7_2d_errors, itpb9_2d_errors, itpb11_2d_errors, itpb13_2d_errors],
            [:b5, :b7, :b9, :b11, :b13])
        itp = convolution_interpolation((xr, xr), vals; degree=deg, fast=true, derivative=-1,
                                        subgrid=:cubic, precompute=101,
                                        kernel_bc=[(:polynomial,:polynomial),(:polynomial,:polynomial)])
        errors[count] = abs(itp(1.0, 1.0) - f2_int)
    end
end

fig2 = Makie.Figure(; size=(1200, 800))
ax2 = Makie.Axis(fig2[1, 1],
    xlabel = L"\text{Number of Sample Points per Dimension}",
    ylabel = L"\text{Absolute Error}",
    title = L"Convergence of 2D integration of $f(x,y) = 1/(1 + 25(x^2+y^2)),\; [-1;1]^2$",
    titlesize = 24,
    xlabelsize = 20,
    ylabelsize = 20,
    xscale = log10,
    yscale = log10,
    limits = ((10, 10^3), (1e-16, 1e0))
)

Makie.lines!(ax2, ns_2d, itpb5_2d_errors,  label="':b5'",  linewidth=3, color=:red)
Makie.lines!(ax2, ns_2d, itpb7_2d_errors,  label="':b7'",  linewidth=3)
Makie.lines!(ax2, ns_2d, itpb9_2d_errors,  label="':b9'",  linewidth=3)
Makie.lines!(ax2, ns_2d, itpb11_2d_errors, label="':b11'", linewidth=3)
Makie.lines!(ax2, ns_2d, itpb13_2d_errors, label="':b13'", linewidth=3)

Makie.lines!(ax2, [10, 1000], [1e-3*(10/10)^9, 1e-3*(10/1000)^9],
       color=:black, linestyle=:dash, label=L"\mathcal{O}(h^9)", linewidth=3)

Legend(fig2[1, 2], ax2, labelsize=20)
fig2
# save("fig/convergence_integration_2d.png", fig2, px_per_unit=3)

# ── 3D convergence ────────────────────────────────────────────────────────────
f3 = (x, y, z) -> exp(-(x^2 + y^2 + z^2))
val_1d_3, _ = quadgk(x -> exp(-x^2), -1, 1, rtol=1e-14)  # factorizes!
f3_int = val_1d_3^3

ns_3d = unique(trunc.(Int, 10.0 .^([1 + 1.2 * i/(n-1) for i in 0:(n-1)])))

itpb5_3d_errors  = zeros(G, length(ns_3d))
itpb7_3d_errors  = zeros(G, length(ns_3d))
itpb9_3d_errors  = zeros(G, length(ns_3d))
itpb11_3d_errors = zeros(G, length(ns_3d))
itpb13_3d_errors = zeros(G, length(ns_3d))

count = 0
for n in ns_3d
    count += 1
    println("3D n = $n")
    xr = [G(-1.0) + G(2*i)/G(n-1) for i in 0:(n-1)]
    vals = [f3(x, y, z) for x in xr, y in xr, z in xr]
    for (errors, deg) in zip(
            [itpb5_3d_errors, itpb7_3d_errors, itpb9_3d_errors, itpb11_3d_errors, itpb13_3d_errors],
            [:b5, :b7, :b9, :b11, :b13])
        itp = convolution_interpolation((xr, xr, xr), vals; degree=deg, fast=true, derivative=-1,
                                        subgrid=:cubic, precompute=101,
                                        kernel_bc=[(:polynomial,:polynomial),(:polynomial,:polynomial),(:polynomial,:polynomial)])
        errors[count] = abs(itp(1.0, 1.0, 1.0) - f3_int)
    end
end

fig3 = Makie.Figure(; size=(1200, 800))
ax3 = Makie.Axis(fig3[1, 1],
    xlabel = L"\text{Number of Sample Points per Dimension}",
    ylabel = L"\text{Absolute Error}",
    title = L"Convergence of 3D integration of $f(x,y,z) = e^{-(x^2+y^2+z^2)},\; [-1;1]^3$",
    titlesize = 24,
    xlabelsize = 20,
    ylabelsize = 20,
    xscale = log10,
    yscale = log10,
    limits = ((10, 10^2.2), (1e-16, 1e0))
)

Makie.lines!(ax3, ns_3d, itpb5_3d_errors,  label="':b5'",  linewidth=3, color=:red)
Makie.lines!(ax3, ns_3d, itpb7_3d_errors,  label="':b7'",  linewidth=3)
Makie.lines!(ax3, ns_3d, itpb9_3d_errors,  label="':b9'",  linewidth=3)
Makie.lines!(ax3, ns_3d, itpb11_3d_errors, label="':b11'", linewidth=3)
Makie.lines!(ax3, ns_3d, itpb13_3d_errors, label="':b13'", linewidth=3)

Makie.lines!(ax3, [10, 300], [1e-1*(10/10)^11, 1e-1*(10/300)^11],
       color=:black, linestyle=:dash, label=L"\mathcal{O}(h^{11})", linewidth=3)

Legend(fig3[1, 2], ax3, labelsize=20)
fig3

# save("fig/convergence_integration_3d.png", fig3, px_per_unit=3)