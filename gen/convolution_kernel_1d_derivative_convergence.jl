using ConvolutionInterpolations
using CairoMakie
using LaTeXStrings
using Scratch
Scratch.clear_scratchspaces!()

G = Float64
# Analytical derivatives
runge(x) = one(G)/G(one(G)+G(25)*G(x)^2)
Nf = 11_000
xf = [-one(G) + G(2*i)/G(Nf-1) for i in 0:Nf-1]
yf = 1 ./(1 .+ 25*xf.^2)
yf_prime = -50 .*xf ./ (1 .+ 25*xf.^2).^2
yf_double_prime = 50*(75 .*xf.^2 .- 1) ./ (1 .+ 25*xf.^2).^3
yf_triple_prime = 15000*xf.*(1 .- 25*xf.^2) ./ (1 .+ 25*xf.^2).^4

n = 100
ns = unique(trunc.(Int, 10.0 .^([1 + 3.0 * i/(n-1) for i in 0:(n-1)])))

itpb5_fast_d0_errors = zeros(G, length(ns))
itpb7_fast_d0_errors = zeros(G, length(ns))
itpb9_fast_d0_errors = zeros(G, length(ns))
itpb11_fast_d0_errors = zeros(G, length(ns))

itpb5_fast_d1_errors = zeros(G, length(ns))
itpb7_fast_d1_errors = zeros(G, length(ns))
itpb9_fast_d1_errors = zeros(G, length(ns))
itpb11_fast_d1_errors = zeros(G, length(ns))

itpb5_fast_d2_errors = zeros(G, length(ns))
itpb7_fast_d2_errors = zeros(G, length(ns))
itpb9_fast_d2_errors = zeros(G, length(ns))
itpb11_fast_d2_errors = zeros(G, length(ns))

itpb5_fast_d3_errors = zeros(G, length(ns))
itpb7_fast_d3_errors = zeros(G, length(ns))
itpb9_fast_d3_errors = zeros(G, length(ns))
itpb11_fast_d3_errors = zeros(G, length(ns))

count = 0
for n in ns
    count += 1
    println("n = $n")
    xr = [-one(G)+G(2)*G(i)/(n-1) for i in 0:(n-1)]
    itpb5_fast_d0 = convolution_interpolation(xr, runge.(xr); degree=:b5, fast=true, kernel_bc=:polynomial, derivative=0, precompute=100, subgrid=:quintic);
    itpb7_fast_d0 = convolution_interpolation(xr, runge.(xr); degree=:b7, fast=true, kernel_bc=:polynomial, derivative=0, precompute=100, subgrid=:quintic);
    itpb9_fast_d0 = convolution_interpolation(xr, runge.(xr); degree=:b9, fast=true, kernel_bc=:polynomial, derivative=0, precompute=100, subgrid=:quintic);
    itpb11_fast_d0 = convolution_interpolation(xr, runge.(xr); degree=:b11, fast=true, kernel_bc=:polynomial, derivative=0, precompute=100, subgrid=:cubic);
    itpb5_fast_d1 = convolution_interpolation(xr, runge.(xr); degree=:b5, fast=true, kernel_bc=:polynomial, derivative=1, precompute=100, subgrid=:quintic);
    itpb7_fast_d1 = convolution_interpolation(xr, runge.(xr); degree=:b7, fast=true, kernel_bc=:polynomial, derivative=1, precompute=100, subgrid=:quintic);
    itpb9_fast_d1 = convolution_interpolation(xr, runge.(xr); degree=:b9, fast=true, kernel_bc=:polynomial, derivative=1, precompute=100, subgrid=:quintic);
    itpb11_fast_d1 = convolution_interpolation(xr, runge.(xr); degree=:b11, fast=true, kernel_bc=:polynomial, derivative=1, precompute=100, subgrid=:cubic);
    itpb5_fast_d2 = convolution_interpolation(xr, runge.(xr); degree=:b5, fast=true, kernel_bc=:polynomial, derivative=2, precompute=100, subgrid=:cubic);
    itpb7_fast_d2 = convolution_interpolation(xr, runge.(xr); degree=:b7, fast=true, kernel_bc=:polynomial, derivative=2, precompute=100, subgrid=:quintic);
    itpb9_fast_d2 = convolution_interpolation(xr, runge.(xr); degree=:b9, fast=true, kernel_bc=:polynomial, derivative=2, precompute=100, subgrid=:quintic);
    itpb11_fast_d2 = convolution_interpolation(xr, runge.(xr); degree=:b11, fast=true, kernel_bc=:polynomial, derivative=2, precompute=100, subgrid=:cubic);
    itpb5_fast_d3 = convolution_interpolation(xr, runge.(xr); degree=:b5, fast=true, kernel_bc=:polynomial, derivative=3, precompute=1000, subgrid=:linear);
    itpb7_fast_d3 = convolution_interpolation(xr, runge.(xr); degree=:b7, fast=true, kernel_bc=:polynomial, derivative=3, precompute=100, subgrid=:cubic);
    itpb9_fast_d3 = convolution_interpolation(xr, runge.(xr); degree=:b9, fast=true, kernel_bc=:polynomial, derivative=3, precompute=100, subgrid=:quintic);
    itpb11_fast_d3 = convolution_interpolation(xr, runge.(xr); degree=:b11, fast=true, kernel_bc=:polynomial, derivative=3, precompute=100, subgrid=:cubic);

    itpb5_fast_d0_errors[count] = maximum(abs.(itpb5_fast_d0.(xf) - yf))
    itpb7_fast_d0_errors[count] = maximum(abs.(itpb7_fast_d0.(xf) - yf))
    itpb9_fast_d0_errors[count] = maximum(abs.(itpb9_fast_d0.(xf) - yf))
    itpb11_fast_d0_errors[count] = maximum(abs.(itpb11_fast_d0.(xf) - yf))
    itpb5_fast_d1_errors[count] = maximum(abs.(itpb5_fast_d1.(xf) - yf_prime))
    itpb7_fast_d1_errors[count] = maximum(abs.(itpb7_fast_d1.(xf) - yf_prime))
    itpb9_fast_d1_errors[count] = maximum(abs.(itpb9_fast_d1.(xf) - yf_prime))
    itpb11_fast_d1_errors[count] = maximum(abs.(itpb11_fast_d1.(xf) - yf_prime))
    itpb5_fast_d2_errors[count] = maximum(abs.(itpb5_fast_d2.(xf) - yf_double_prime))
    itpb7_fast_d2_errors[count] = maximum(abs.(itpb7_fast_d2.(xf) - yf_double_prime))
    itpb9_fast_d2_errors[count] = maximum(abs.(itpb9_fast_d2.(xf) - yf_double_prime))
    itpb11_fast_d2_errors[count] = maximum(abs.(itpb11_fast_d2.(xf) - yf_double_prime))
    itpb5_fast_d3_errors[count] = maximum(abs.(itpb5_fast_d3.(xf) - yf_triple_prime))
    itpb7_fast_d3_errors[count] = maximum(abs.(itpb7_fast_d3.(xf) - yf_triple_prime))
    itpb9_fast_d3_errors[count] = maximum(abs.(itpb9_fast_d3.(xf) - yf_triple_prime))
    itpb11_fast_d3_errors[count] = maximum(abs.(itpb11_fast_d3.(xf) - yf_triple_prime))
end

update_theme!(
    fontsize=14,
    Axis=(xlabelsize=30, ylabelsize=30, titlesize=30),
    Legend=(labelsize=30,)
)

fig = Makie.Figure(; size=(1200, 800))
ax = Makie.Axis(fig[1, 1],
    xlabel = L"\text{Total Number of Sample Points}",
    ylabel = L"\text{Maximum Absolute Error}",
    title = L"\text{Interpolation on }f(x) = 1/(1 + 25x²),\; x \in [-1;1]",
    xscale = log10,
    yscale = log10,
    limits = ((10, 10^4), (1e-15, 1e5)),
    titlealign = :left
)


# Plot the lines
colors = [:blue, :orange, :green, :purple]  # One per derivative
linestyles = [:solid, :dash]  # b5=solid, b7=dashed

labels = [
    L"b5, \, f",
    L"b5, \, \partial_{x} \, f",
    L"b5, \, \partial_{xx} \, f",
    L"b5, \, \partial_{xxx} \, f",
    L"b7, \, f",
    L"b7, \, \partial_{x} \, f",
    L"b7, \, \partial_{xx} \, f",
    L"b7, \, \partial_{xxx} \, f",
    L"b9, \, f",
    L"b9, \, \partial_{x} \, f",
    L"b9, \, \partial_{xx} \, f",
    L"b9, \, \partial_{xxx} \, f",
    L"b11, \, f",
    L"b11, \, \partial_{x} \, f",
    L"b11, \, \partial_{xx} \, f",
    L"b11, \, \partial_{xxx} \, f",
    L"\mathcal{O}(h^7)"
]

Makie.lines!(ax, ns, itpb5_fast_d0_errors, label=labels[1], linestyle=:solid, color=colors[1])
Makie.lines!(ax, ns, itpb7_fast_d0_errors, label=labels[5], linestyle=:dash, color=colors[1])
Makie.lines!(ax, ns, itpb9_fast_d0_errors, label=labels[9], linestyle=:dot, color=colors[1])
Makie.lines!(ax, ns, itpb11_fast_d0_errors, label=labels[13], linestyle=:dashdot, color=colors[1])
Makie.lines!(ax, ns, itpb5_fast_d1_errors, label=labels[2], linestyle=:solid, color=colors[2])
Makie.lines!(ax, ns, itpb7_fast_d1_errors, label=labels[6], linestyle=:dash, color=colors[2])
Makie.lines!(ax, ns, itpb9_fast_d1_errors, label=labels[10], linestyle=:dot, color=colors[2])
Makie.lines!(ax, ns, itpb11_fast_d1_errors, label=labels[14], linestyle=:dashdot, color=colors[2])
Makie.lines!(ax, ns, itpb5_fast_d2_errors, label=labels[3], linestyle=:solid, color=colors[3])
Makie.lines!(ax, ns, itpb7_fast_d2_errors, label=labels[7], linestyle=:dash, color=colors[3])
Makie.lines!(ax, ns, itpb9_fast_d2_errors, label=labels[11], linestyle=:dot, color=colors[3])
Makie.lines!(ax, ns, itpb11_fast_d2_errors, label=labels[15], linestyle=:dashdot, color=colors[3])
Makie.lines!(ax, ns, itpb5_fast_d3_errors, label=labels[4], linestyle=:solid, color=colors[4])
Makie.lines!(ax, ns, itpb7_fast_d3_errors, label=labels[8], linestyle=:dash, color=colors[4])
Makie.lines!(ax, ns, itpb9_fast_d3_errors, label=labels[12], linestyle=:dot, color=colors[4])
Makie.lines!(ax, ns, itpb11_fast_d3_errors, label=labels[16], linestyle=:dashdot, color=colors[4])
Makie.lines!(ax, [10, 10000], [1e-5*(1000/100)^7, 1e-5*(100/10000)^7], 
       color=:black, linestyle=:dash, label=labels[17], linewidth=3)

# Legend outside to the right
Legend(fig[1, 2], ax, framevisible=false, nbanks=4)

fig

# save("kernel_derivatives_1d_runge.png", fig, px_per_unit=3)  # High DPI
