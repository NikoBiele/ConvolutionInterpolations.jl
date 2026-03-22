###########################################################################
### NONUNIFORM DERIVATIVES ################################################
###########################################################################

### Nonuniform b-kernel derivatives
nu_deriv_kernels = [:b5] #, :b7, :b9, :b11]
N_nu_deriv = 40
tolerance_nu_deriv = 1e-4

# shared helper
function make_nonuniform_grid_derivatives(n; a=0.0, b=1.0, strength=0.3)
    x = collect(range(a, b, length=n))
    h = (b - a) / (n - 1)
    for i in 2:n-1
        x[i] += strength * h * sin(3π * (x[i] - a) / (b - a))
    end
    sort!(x)
    return x
end

println("Testing nonuniform derivative kernel interpolation in 1D, 2D...")
@testset "1D nonuniform derivative d1" begin
    for kernel in nu_deriv_kernels
        println("Testing 1D nonuniform derivative d1: ", kernel)
        x_nu = make_nonuniform_grid_derivatives(N_nu_deriv; a=0.0, b=2π)
        vals = sin.(x_nu)
        itp_d1 = convolution_interpolation(x_nu, vals; kernel=kernel, derivative=1)

        test_points = [(x_nu[i] + x_nu[i+1]) / 2 for i in 1:N_nu_deriv-1]
        analytical = cos.(test_points)
        interpolated = Float64[itp_d1(x) for x in test_points]
        @test interpolated ≈ analytical atol=tolerance_nu_deriv
    end
end

@testset "1D nonuniform derivative d2" begin
    for kernel in nu_deriv_kernels
        println("Testing 1D nonuniform derivative d2: ", kernel)
        x_nu = make_nonuniform_grid_derivatives(N_nu_deriv*2; a=0.0, b=2π) # double the number of points
        vals = sin.(x_nu)
        itp_d2 = convolution_interpolation(x_nu, vals; kernel=kernel, derivative=2)

        test_points = [(x_nu[i] + x_nu[i+1]) / 2 for i in 1:(N_nu_deriv*2-1)]
        analytical = -sin.(test_points)
        interpolated = Float64[itp_d2(x) for x in test_points]
        @test interpolated ≈ analytical atol=tolerance_nu_deriv
    end
end

@testset "2D nonuniform derivative d1" begin
    for kernel in nu_deriv_kernels
        println("Testing 2D nonuniform derivative d1: ", kernel)
        x_nu = make_nonuniform_grid_derivatives(N_nu_deriv; a=0.0, b=2π)
        y_nu = make_nonuniform_grid_derivatives(N_nu_deriv; a=0.0, b=2π, strength=0.2)
        vals = [sin(x) * sin(y) for x in x_nu, y in y_nu]
        itp_d1 = convolution_interpolation((x_nu, y_nu), vals; kernel=kernel, derivative=1)

        test_x = [(x_nu[i] + x_nu[i+1]) / 2 for i in 1:N_nu_deriv-1]
        test_y = [(y_nu[i] + y_nu[i+1]) / 2 for i in 1:N_nu_deriv-1]
        analytical = [cos(x) * cos(y) for x in test_x, y in test_y]
        interpolated = Float64[itp_d1(x, y) for x in test_x, y in test_y]
        @test interpolated ≈ analytical atol=tolerance_nu_deriv
    end
end
