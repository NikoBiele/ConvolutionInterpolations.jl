using ConvolutionInterpolations
using Plots

# Non-uniform grid with 23 points
x = [0.0, 0.15, 0.4, 0.7, 1.5, 2.5, 3.8, 4.2, 4.6, 4.8, 5.0,
     5.3, 5.6, 6.0, 6.5, 7.0, 7.3, 7.8, 8.2, 8.5, 9.0, 9.5, 10.0]

# f(x) = sin(x) * exp(-x/5)
# f^(n)(x) = Im[(i - 1/5)^n * exp((i - 1/5)x)]  (complex exponential trick)
function fk(x, n)
    a = 1/5
    z = (im - a)^n * exp((im - a) * x)
    return imag(z)
end

y = fk.(x, 0)
x_fine = range(0.0, 10.0, length=300)

colors = [:royalblue, :orangered, :green4, :purple3, :deeppink, :goldenrod]
labels = ["f(x)", "f'(x)", "f''(x)", "f'''(x)", "f⁴(x)", "f⁵(x)"]

p = plot(; size=(900, 500), dpi=300, legend=:topright,
         legendfontsize=14,
         title="Nonuniform b11 kernel: function and derivatives 1-5",
         background_color=:white, grid=true, gridalpha=0.3)

for d in 0:5
    y_dots = fk.(x, d)
    itp = convolution_interpolation(x, y; degree=:b11, derivative=d)
    y_itp = itp.(x_fine)
    
    scatter!(p, x, y_dots, color=colors[d+1], markersize=3, markerstrokewidth=0, label=labels[d+1])
    plot!(p, x_fine, y_itp, color=colors[d+1], linewidth=1.5, label="")
end

display(p)
# savefig(p, "fig/nonuniform_derivatives.png")