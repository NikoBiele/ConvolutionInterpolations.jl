###########################################################################
### NONUNIFORM DERIVATIVES ################################################
###########################################################################

### Nonuniform b-kernel derivatives
nu_deriv_kernels = [:b5, :b7, :b9, :b11]
N_nu_deriv = 12
tolerance_nu_deriv = 0.1

println("Testing nonuniform derivative kernel interpolation in 1D, 2D...")
@testset "1D nonuniform derivative d1" begin
    for kernel in nu_deriv_kernels
        println("Testing 1D nonuniform derivative d1: ", kernel)
        x_nu = make_nonuniform_grid(N_nu_deriv; a=0.0, b=2π)
        vals = sin.(x_nu)
        itp_d1 = convolution_interpolation(x_nu, vals; degree=kernel, derivative=1)

        test_points = [(x_nu[i] + x_nu[i+1]) / 2 for i in 4:N_nu_deriv-4]
        analytical = cos.(test_points)
        interpolated = Float64[itp_d1(x) for x in test_points]
        @test interpolated ≈ analytical atol=tolerance_nu_deriv
    end
end

@testset "1D nonuniform derivative d2" begin
    for kernel in nu_deriv_kernels
        println("Testing 1D nonuniform derivative d2: ", kernel)
        x_nu = make_nonuniform_grid(N_nu_deriv; a=0.0, b=2π)
        vals = sin.(x_nu)
        itp_d2 = convolution_interpolation(x_nu, vals; degree=kernel, derivative=2)

        test_points = [(x_nu[i] + x_nu[i+1]) / 2 for i in 4:N_nu_deriv-4]
        analytical = -sin.(test_points)
        interpolated = Float64[itp_d2(x) for x in test_points]
        @test interpolated ≈ analytical atol=tolerance_nu_deriv
    end
end

@testset "2D nonuniform derivative d1" begin
    for kernel in nu_deriv_kernels
        println("Testing 2D nonuniform derivative d1: ", kernel)
        x_nu = make_nonuniform_grid(N_nu_deriv; a=0.0, b=2π)
        y_nu = make_nonuniform_grid(N_nu_deriv; a=0.0, b=2π, strength=0.2)
        vals = [sin(x) * sin(y) for x in x_nu, y in y_nu]
        itp_d1 = convolution_interpolation((x_nu, y_nu), vals; degree=kernel, derivative=1)

        test_x = [(x_nu[i] + x_nu[i+1]) / 2 for i in 5:N_nu_deriv-5]
        test_y = [(y_nu[i] + y_nu[i+1]) / 2 for i in 5:N_nu_deriv-5]
        analytical = [cos(x) * cos(y) for x in test_x, y in test_y]
        interpolated = Float64[itp_d1(x, y) for x in test_x, y in test_y]
        @test interpolated ≈ analytical atol=tolerance_nu_deriv
    end
end
