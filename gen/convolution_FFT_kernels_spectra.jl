using GenericFFT
using CairoMakie
using ConvolutionInterpolations
using LaTeXStrings

# Number of points (power of 2 for FFT efficiency)
n = 2^16

# Create sample points
L = 100.0
dx = 2L / n
x = [-L + i * dx for i in 0:n-1]

# Set up the figure
fig = Figure(size=(1500, 1000), fontsize=16)
ax = Axis(fig[1, 1],
    title=L"\text{FFT of ConvolutionInterpolations.jl kernels}",
    titlesize=24,
    xlabel=L"\text{Frequency [Hz]}",
    ylabel=L"\text{Magnitude}",
    yscale=log10,
    limits=((-5, 5), (1e-7, 1.5)),
    xticks=-5:1:5,
    xtickformat=xs -> [L"\mathbf{%$(x)}" for x in xs],
    yticks=[1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0],
    ytickformat=ys -> [L"\mathbf{10^{%$(Int(log10(y)))}}" for y in ys],
    xlabelsize=18,
    ylabelsize=18)

# Define degrees
degrees = [:a0, :a1, :a3, :a4, :a5, :a7, :b5, :b7, :b9, :b11, :b13]

# Darker colors for b-kernels
b_colors = [:darkblue, :darkred, :darkgreen, :purple, :black]
b_markers = [:circle, :rect, :diamond, :utriangle, :cross]
# Lighter colors for a-kernels
a_colors = [:lightblue, :lightsalmon, :lightgreen, :plum, :silver, :wheat]

b_idx = 0
a_idx = 0
legend_entries = []
legend_labels = String[]

for degree in degrees
    println("Computing FFT for $degree kernel...")
    itp = convolution_interpolation(x, [1/(1+25*xi^2) for xi in x]; degree=degree, fast=false)
    kernel_vals = itp.itp.kernel.(x)

    # GenericFFT handles BigFloat natively
    spectrum = fft(kernel_vals)

    # Shift for centered plot
    spectrum_shifted = fftshift(spectrum)
    freqs = fftshift(fftfreq(n, 1/Float64(dx)))

    # Compute normalized magnitudes
    magnitudes = Float64.(abs.(spectrum_shifted))
    magnitudes ./= maximum(magnitudes)

    # Determine line style and color
    is_b_kernel = startswith(string(degree), "b")
    if is_b_kernel
        b_idx += 1
        color = b_colors[mod1(b_idx, length(b_colors))]
        linestyle = :solid
        linewidth = 2.0
    else
        a_idx += 1
        color = a_colors[mod1(a_idx, length(a_colors))]
        linestyle = :dash
        linewidth = 1.5
    end

    if is_b_kernel
        # Subsample for markers (every ~200th point in the view range)
        mask = findall(f -> -5 <= f <= 5, freqs)
        marker_stride = max(1, length(mask) ÷ 15)
        marker_indices = mask[1:marker_stride:end]
        
        # Plot full line and markers separately, group for legend
        l = lines!(ax, freqs, magnitudes,
            color=color,
            linewidth=linewidth)
        s = scatter!(ax, freqs[marker_indices], magnitudes[marker_indices],
            marker=b_markers[mod1(b_idx, length(b_markers))],
            color=color,
            markersize=18)
        # Group them for a combined legend entry
        push!(legend_entries, [l, s])
        push!(legend_labels, string(degree, " kernel"))
    else
        l = lines!(ax, freqs, magnitudes,
            linestyle=linestyle,
            color=color,
            linewidth=linewidth)
        push!(legend_entries, l)
        push!(legend_labels, string(degree, " kernel"))
    end
end

# Add ideal sinc (brickwall) filter
# The sinc function has compact support in frequency domain: rect function
# Cutoff at frequency = 0.5 (Nyquist for unit spacing)
sinc_cutoff = 0.5
freqs_plot = range(-5, 5, length=1000)
brickwall = [abs(f) <= sinc_cutoff ? 1.0 : 1e-20 for f in freqs_plot]

l_sinc = lines!(ax, freqs_plot, brickwall,
    linestyle=:dot,
    color=:black,
    linewidth=2.5)
push!(legend_entries, l_sinc)
push!(legend_labels, "sinc (ideal)")

# Add legend at bottom
Legend(fig[2, 1], legend_entries, legend_labels, orientation=:horizontal, nbanks=1, framevisible=false)

fig
# save("FFT_kernels_spectra.png", fig, px_per_unit=3.0)
