###########################################################################
### TEST NONUNIFORM INTERPOLATION #########################################
###########################################################################    

### Nonuniform b-kernel interpolation
nu_kernels = [:n3, :b5] #, :b7, :b9, :b11] # :a3 kernel triggers nonuniform lower order kernel
N_nu = 40 # 1d and 2d
N_nu_3d = 10
N_nu_4d = 6
tolerance_nu = 1e-6

# shared helper
function make_nonuniform_grid_interpolation(n; a=0.0, b=1.0, strength=0.3)
    x = collect(range(a, b, length=n))
    h = (b - a) / (n - 1)
    for i in 2:n-1
        x[i] += strength * h * sin(3π * (x[i] - a) / (b - a))
    end
    sort!(x)
    return x
end

println("Testing nonuniform kernel interpolation in 1D, 2D, 3D, 4D...")
@testset "1D nonuniform kernels" begin
    for kernel in nu_kernels
        println("Testing 1D nonuniform kernel: ", kernel)
        # grid point reproduction with random data
        x_nu = make_nonuniform_grid_interpolation(N_nu)
        vals_rand = rand(N_nu)
        itp_rand = convolution_interpolation(x_nu, vals_rand; kernel=kernel)
        @test itp_rand.itp.(x_nu) ≈ vals_rand atol=tolerance_nu

        # linear midpoint reproduction
        vals_linear = x_nu
        itp_linear = convolution_interpolation(x_nu, vals_linear; kernel=kernel)
        midpoints = [(x_nu[i] + x_nu[i+1]) / 2 for i in 1:N_nu-1]
        @test itp_linear.itp.(midpoints) ≈ midpoints atol=tolerance_nu
    end
end

@testset "2D nonuniform kernels" begin
    for kernel in nu_kernels
        println("Testing 2D nonuniform kernel: ", kernel)
        x_nu = make_nonuniform_grid_interpolation(N_nu)
        y_nu = make_nonuniform_grid_interpolation(N_nu; strength=0.2)

        # grid point reproduction with random data
        vals_rand = rand(N_nu, N_nu)
        itp_rand = convolution_interpolation((x_nu, y_nu), vals_rand; kernel=kernel)
        @test [itp_rand.itp(x_nu[i], y_nu[j]) for i in 1:N_nu for j in 1:N_nu] ≈ 
                [vals_rand[i,j] for i in 1:N_nu for j in 1:N_nu] atol=tolerance_nu

        # bilinear midpoint reproduction
        vals_linear = [x_nu[i] + y_nu[j] for i in 1:N_nu, j in 1:N_nu]
        itp_linear = convolution_interpolation((x_nu, y_nu), vals_linear; kernel=kernel)
        test_x = [(x_nu[i] + x_nu[i+1]) / 2 for i in 3:N_nu-3]
        test_y = [(y_nu[i] + y_nu[i+1]) / 2 for i in 3:N_nu-3]
        analytical = [tx + ty for tx in test_x, ty in test_y]
        interpolated = [itp_linear.itp(tx, ty) for tx in test_x, ty in test_y]
        @test interpolated ≈ analytical atol=tolerance_nu
    end
end

@testset "3D nonuniform kernels" begin
    for kernel in nu_kernels
        println("Testing 3D nonuniform kernel: ", kernel)
        x_nu = make_nonuniform_grid_interpolation(N_nu_3d)
        y_nu = make_nonuniform_grid_interpolation(N_nu_3d; strength=0.2)
        z_nu = make_nonuniform_grid_interpolation(N_nu_3d; strength=0.25)

        # grid point reproduction with random data
        vals_rand = rand(N_nu_3d, N_nu_3d, N_nu_3d)
        itp_rand = convolution_interpolation((x_nu, y_nu, z_nu), vals_rand; kernel=kernel)
        interpolated = [itp_rand.itp(x_nu[i], y_nu[j], z_nu[k]) for i in 1:N_nu_3d for j in 1:N_nu_3d for k in 1:N_nu_3d]
        true_vals = [vals_rand[i,j,k] for i in 1:N_nu_3d for j in 1:N_nu_3d for k in 1:N_nu_3d]
        @test interpolated ≈ true_vals atol=tolerance_nu
    end
end

@testset "4D nonuniform kernels" begin
    for kernel in nu_kernels[1:2] # to save time, test only :n3 and :b5
        println("Testing 4D nonuniform kernel: ", kernel)
        x_nu = make_nonuniform_grid_interpolation(N_nu_4d)
        y_nu = make_nonuniform_grid_interpolation(N_nu_4d; strength=0.2)
        z_nu = make_nonuniform_grid_interpolation(N_nu_4d; strength=0.25)
        w_nu = make_nonuniform_grid_interpolation(N_nu_4d)

        # grid point reproduction with random data
        vals_rand = rand(N_nu_4d, N_nu_4d, N_nu_4d, N_nu_4d)
        itp_rand = convolution_interpolation((x_nu, y_nu, z_nu, w_nu), vals_rand; kernel=kernel, bc=:linear)
        interpolated = [itp_rand.itp(x_nu[i], y_nu[j], z_nu[k], w_nu[l]) for i in 1:N_nu_4d for j in 1:N_nu_4d for k in 1:N_nu_4d for l in 1:N_nu_4d]
        true_vals = [vals_rand[i,j,k,l] for i in 1:N_nu_4d for j in 1:N_nu_4d for k in 1:N_nu_4d for l in 1:N_nu_4d]
        @test interpolated ≈ true_vals atol=tolerance_nu
    end
end
