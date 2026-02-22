using ConvolutionInterpolations
using Plots

f(x) = cos(x^2 / 9.0)

# Data points
x_data = range(0, 10, length=16)
y_data = f.(x_data)
x_fine = range(0, 10, length=1000)
y_true = f.(x_fine)

# --- Left panel: ':a' type degrees ---
a_kernels = [:a0, :a1, :a3, :a4, :a5, :a7]
a_colors = [:blue, :green, :red, :orange, :magenta]

p1 = scatter(x_data, y_data, label="Data points", color=:red, markersize=6, markerstrokewidth=1.5)
plot!(p1, x_fine, y_true, label="True function", color=:black, linewidth=1.5)
for (k, c) in zip(a_kernels, a_colors)
    itp = convolution_interpolation(x_data, y_data; degree=k)
    plot!(p1, x_fine, itp.(x_fine), label="degree=:$k", color=c, linewidth=1.2)
end
plot!(p1, title="Lower order kernels", xlabel="x", ylabel="y")

# --- Right panel: ':b' type degrees ---
b_kernels = [:b5, :b7, :b9, :b11, :b13]
b_colors = [:blue, :green, :red, :orange, :magenta, :cyan]

p2 = scatter(x_data, y_data, label="Data points", color=:red, markersize=6, markerstrokewidth=1.5)
plot!(p2, x_fine, y_true, label="True function", color=:black, linewidth=1.5)
for (k, c) in zip(b_kernels, b_colors)
    itp = convolution_interpolation(x_data, y_data; degree=k)
    plot!(p2, x_fine, itp.(x_fine), label="degree=:$k", color=c, linewidth=1.2)
end
plot!(p2, title="Higher order kernels", xlabel="x", ylabel="y")

plot(p1, p2, layout=(1, 2), size=(1200, 400), dpi=500)
savefig("fig/convolution_interpolation_kernels.png")