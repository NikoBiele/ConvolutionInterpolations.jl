###########################################################################
### NONUNIFORM PER-DIM DERIVATIVES ########################################
###########################################################################

### Per-dimension nonuniform b-kernel derivatives
nu_pd_kernels = [:b5, :b7, :b9, :b11]
N_nu_pd = 20
tolerance_nu_pd = 0.005

# shared helper
function make_nonuniform_grid_perdim_derivatives(n; a=0.0, b=1.0, strength=0.3)
    x = collect(range(a, b, length=n))
    h = (b - a) / (n - 1)
    for i in 2:n-1
        x[i] += strength * h * sin(3π * (x[i] - a) / (b - a))
    end
    sort!(x)
    return x
end

# f(x,y) = sin(x)*cos(y), exact derivatives:
#   (0,0) → sin(x)*cos(y)
#   (1,0) → cos(x)*cos(y)
#   (0,1) → -sin(x)*sin(y)
#   (1,1) → -cos(x)*sin(y)
#   (2,0) → -sin(x)*cos(y)
#   (0,2) → -sin(x)*cos(y)

println("Testing per-dim nonuniform b-kernel interpolation in 2D...")

@testset "2D nonuniform per-dim (0,0) — interpolation" begin
    println("Testing 2D nonuniform per-dim (0,0) — interpolation...")
    for kx in nu_pd_kernels, ky in nu_pd_kernels
        x_nu = make_nonuniform_grid_perdim_derivatives(N_nu_pd; a=0.0, b=2π)
        y_nu = make_nonuniform_grid_perdim_derivatives(N_nu_pd; a=0.0, b=2π, strength=0.2)
        vals = [sin(x) * cos(y) for x in x_nu, y in y_nu]
        itp = convolution_interpolation((x_nu, y_nu), vals; degree=(kx, ky), derivative=(0, 0))
        test_x = [(x_nu[i] + x_nu[i+1]) / 2 for i in 5:N_nu_pd-5]
        test_y = [(y_nu[i] + y_nu[i+1]) / 2 for i in 5:N_nu_pd-5]
        err = maximum(abs(itp(x, y) - sin(x)*cos(y)) for x in test_x, y in test_y)
        @test err < tolerance_nu_pd
    end
end

@testset "2D nonuniform per-dim (1,0) — d/dx" begin
    println("Testing 2D nonuniform per-dim (1,0) — d/dx...")
    for kx in nu_pd_kernels, ky in nu_pd_kernels
        x_nu = make_nonuniform_grid_perdim_derivatives(N_nu_pd; a=0.0, b=2π)
        y_nu = make_nonuniform_grid_perdim_derivatives(N_nu_pd; a=0.0, b=2π, strength=0.2)
        vals = [sin(x) * cos(y) for x in x_nu, y in y_nu]
        itp = convolution_interpolation((x_nu, y_nu), vals; degree=(kx, ky), derivative=(1, 0))
        test_x = [(x_nu[i] + x_nu[i+1]) / 2 for i in 5:N_nu_pd-5]
        test_y = [(y_nu[i] + y_nu[i+1]) / 2 for i in 5:N_nu_pd-5]
        err = maximum(abs(itp(x, y) - cos(x)*cos(y)) for x in test_x, y in test_y)
        @test err < tolerance_nu_pd
    end
end

@testset "2D nonuniform per-dim (0,1) — d/dy" begin
    println("Testing 2D nonuniform per-dim (0,1) — d/dy...")
    for kx in nu_pd_kernels, ky in nu_pd_kernels
        x_nu = make_nonuniform_grid_perdim_derivatives(N_nu_pd; a=0.0, b=2π)
        y_nu = make_nonuniform_grid_perdim_derivatives(N_nu_pd; a=0.0, b=2π, strength=0.2)
        vals = [sin(x) * cos(y) for x in x_nu, y in y_nu]
        itp = convolution_interpolation((x_nu, y_nu), vals; degree=(kx, ky), derivative=(0, 1))
        test_x = [(x_nu[i] + x_nu[i+1]) / 2 for i in 5:N_nu_pd-5]
        test_y = [(y_nu[i] + y_nu[i+1]) / 2 for i in 5:N_nu_pd-5]
        err = maximum(abs(itp(x, y) - (-sin(x)*sin(y))) for x in test_x, y in test_y)
        @test err < tolerance_nu_pd
    end
end

@testset "2D nonuniform per-dim (1,1) — d²/dxdy" begin
    println("Testing 2D nonuniform per-dim (1,1) — d²/dxdy...")
    for kx in nu_pd_kernels, ky in nu_pd_kernels
        x_nu = make_nonuniform_grid_perdim_derivatives(N_nu_pd; a=0.0, b=2π)
        y_nu = make_nonuniform_grid_perdim_derivatives(N_nu_pd; a=0.0, b=2π, strength=0.2)
        vals = [sin(x) * cos(y) for x in x_nu, y in y_nu]
        itp = convolution_interpolation((x_nu, y_nu), vals; degree=(kx, ky), derivative=(1, 1))
        test_x = [(x_nu[i] + x_nu[i+1]) / 2 for i in 5:N_nu_pd-5]
        test_y = [(y_nu[i] + y_nu[i+1]) / 2 for i in 5:N_nu_pd-5]
        err = maximum(abs(itp(x, y) - (-cos(x)*sin(y))) for x in test_x, y in test_y)
        @test err < tolerance_nu_pd
    end
end

@testset "2D nonuniform per-dim (2,0) — d²/dx²" begin
    println("Testing 2D nonuniform per-dim (2,0) — d²/dx²...")
    for kx in [:b7, :b9, :b11], ky in nu_pd_kernels   # b5 d2 at N=12 may be marginal
        x_nu = make_nonuniform_grid_perdim_derivatives(N_nu_pd; a=0.0, b=2π)
        y_nu = make_nonuniform_grid_perdim_derivatives(N_nu_pd; a=0.0, b=2π, strength=0.2)
        vals = [sin(x) * cos(y) for x in x_nu, y in y_nu]
        itp = convolution_interpolation((x_nu, y_nu), vals; degree=(kx, ky), derivative=(2, 0))
        test_x = [(x_nu[i] + x_nu[i+1]) / 2 for i in 5:N_nu_pd-5]
        test_y = [(y_nu[i] + y_nu[i+1]) / 2 for i in 5:N_nu_pd-5]
        err = maximum(abs(itp(x, y) - (-sin(x)*cos(y))) for x in test_x, y in test_y)
        @test err < tolerance_nu_pd
    end
end

@testset "2D nonuniform per-dim consistency: tuple vs scalar same kernel" begin
    println("Testing 2D nonuniform per-dim consistency: tuple vs scalar same kernel...")
    for kernel in nu_pd_kernels
        for deriv in [(0,0), (1,0), (0,1), (1,1)]
            x_nu = make_nonuniform_grid_perdim_derivatives(N_nu_pd; a=0.0, b=2π)
            y_nu = make_nonuniform_grid_perdim_derivatives(N_nu_pd; a=0.0, b=2π, strength=0.2)
            vals = [sin(x) * cos(y) for x in x_nu, y in y_nu]
            itp_tuple  = convolution_interpolation((x_nu, y_nu), vals; degree=(kernel, kernel), derivative=deriv)
            itp_scalar = convolution_interpolation((x_nu, y_nu), vals; degree=kernel,           derivative=deriv)
            test_x = [(x_nu[i] + x_nu[i+1]) / 2 for i in 5:N_nu_pd-5]
            test_y = [(y_nu[i] + y_nu[i+1]) / 2 for i in 5:N_nu_pd-5]
            err = maximum(abs(itp_tuple(x, y) - itp_scalar(x, y)) for x in test_x, y in test_y)
            @test err < 1e-10
        end
    end
end