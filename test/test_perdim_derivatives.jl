#######################################################################
### TEST PER-DIMENSION DERIVATIVE ORDERS ############################
#######################################################################
#
# Tests DerivativeOrder{(d1,d2,...)} dispatch — separate from the scalar
# derivative=1/2 path. Covers 2D and 3D, fast and slow, for all
# combinations of per-dim derivative orders supported by b-series kernels.
#
# f(x,y)   = sin(x)*sin(y)
# (1,0) →   cos(x)*sin(y)
# (0,1) →   sin(x)*cos(y)
# (1,1) →   cos(x)*cos(y)
# (2,0) →  -sin(x)*sin(y)
# (0,2) →  -sin(x)*sin(y)
#
# f(x,y,z) = sin(x)*sin(y)*sin(z)
# (1,0,0) →  cos(x)*sin(y)*sin(z)
# (0,1,0) →  sin(x)*cos(y)*sin(z)
# (0,0,1) →  sin(x)*sin(y)*cos(z)
# (1,1,0) →  cos(x)*cos(y)*sin(z)
# (1,0,1) →  cos(x)*sin(y)*cos(z)
# (0,1,1) →  sin(x)*cos(y)*cos(z)
#######################################################################

perdim_deriv_kernels = [:b5, :b7, :b9, :b11]
N_pd = 20
tolerance_pd = 0.005

println("Testing per-dimension derivative orders in 2D and 3D...")

# ── 2D helpers ────────────────────────────────────────────────────────

const perdim_2d_cases = [
    ((1, 0), (x, y) -> cos(x)*sin(y)),
    ((0, 1), (x, y) -> sin(x)*cos(y)),
    ((1, 1), (x, y) -> cos(x)*cos(y)),
    ((2, 0), (x, y) -> -sin(x)*sin(y)),
    ((0, 2), (x, y) -> -sin(x)*sin(y)),
]

# ── 2D direct ─────────────────────────────────────────────────────────

@testset "2D per-dim derivative direct" begin
    for kernel in perdim_deriv_kernels
        println("Testing 2D per-dim derivative direct: ", kernel)
        r = range(0.0, 2π, length=N_pd)
        vals = [sin(x)*sin(y) for x in r, y in r]
        test_pts = [(r[i] + r[i+1]) / 2 for i in 5:N_pd-5]
        for (deriv, exact) in perdim_2d_cases
            itp = convolution_interpolation((r, r), vals; degree=(kernel,kernel),
                      derivative=deriv, fast=false, kernel_bc=:polynomial)
            interpolated = [itp(x, y) for x in test_pts, y in test_pts]
            analytical   = [exact(x, y) for x in test_pts, y in test_pts]
            @test interpolated ≈ analytical atol=tolerance_pd
        end
    end
end

# ── 2D fast ───────────────────────────────────────────────────────────

@testset "2D per-dim derivative fast" begin
    for kernel in perdim_deriv_kernels
        println("Testing 2D per-dim derivative fast: ", kernel)
        r = range(0.0, 2π, length=N_pd)
        vals = [sin(x)*sin(y) for x in r, y in r]
        test_pts = [(r[i] + r[i+1]) / 2 for i in 5:N_pd-5]
        for (deriv, exact) in perdim_2d_cases
            itp = convolution_interpolation((r, r), vals; degree=(kernel,kernel),
                      derivative=deriv, fast=true, kernel_bc=:polynomial)
            interpolated = [itp(x, y) for x in test_pts, y in test_pts]
            analytical   = [exact(x, y) for x in test_pts, y in test_pts]
            @test interpolated ≈ analytical atol=tolerance_pd
        end
    end
end

# ── 2D fast vs direct agreement ───────────────────────────────────────

@testset "2D per-dim derivative fast vs direct" begin
    for kernel in perdim_deriv_kernels
        println("Testing 2D per-dim derivative fast vs direct: ", kernel)
        r = range(0.0, 2π, length=N_pd)
        vals = [sin(x)*sin(y) for x in r, y in r]
        test_pts = [(r[i] + r[i+1]) / 2 for i in 5:N_pd-5]
        for (deriv, _) in perdim_2d_cases
            itp_fast   = convolution_interpolation((r, r), vals; degree=(kernel,kernel),
                             derivative=deriv, fast=true,  kernel_bc=:polynomial)
            itp_direct = convolution_interpolation((r, r), vals; degree=kernel,
                             derivative=deriv, fast=false, kernel_bc=:polynomial)
            err = maximum(abs(itp_fast(x, y) - itp_direct(x, y))
                          for x in test_pts, y in test_pts)
            @test err < 1e-4
        end
    end
end

# ── 3D helpers ────────────────────────────────────────────────────────

const perdim_3d_cases = [
    ((1, 0, 0), (x, y, z) ->  cos(x)*sin(y)*sin(z)),
    ((0, 1, 0), (x, y, z) ->  sin(x)*cos(y)*sin(z)),
    ((0, 0, 1), (x, y, z) ->  sin(x)*sin(y)*cos(z)),
    ((1, 1, 0), (x, y, z) ->  cos(x)*cos(y)*sin(z)),
    ((1, 0, 1), (x, y, z) ->  cos(x)*sin(y)*cos(z)),
    ((0, 1, 1), (x, y, z) ->  sin(x)*cos(y)*cos(z)),
]

# ── 3D direct ─────────────────────────────────────────────────────────

@testset "3D per-dim derivative direct" begin
    for kernel in perdim_deriv_kernels
        println("Testing 3D per-dim derivative direct: ", kernel)
        r = range(0.0, 2π, length=N_pd)
        vals = [sin(x)*sin(y)*sin(z) for x in r, y in r, z in r]
        test_pts = [(r[i] + r[i+1]) / 2 for i in 6:N_pd-6]
        for (deriv, exact) in perdim_3d_cases
            itp = convolution_interpolation((r, r, r), vals; degree=(kernel,kernel,kernel),
                      derivative=deriv, fast=false, kernel_bc=:polynomial)
            interpolated = [itp(x, y, z) for x in test_pts, y in test_pts, z in test_pts]
            analytical   = [exact(x, y, z) for x in test_pts, y in test_pts, z in test_pts]
            @test interpolated ≈ analytical atol=tolerance_pd
        end
    end
end

# ── 3D fast ───────────────────────────────────────────────────────────

@testset "3D per-dim derivative fast" begin
    for kernel in perdim_deriv_kernels
        println("Testing 3D per-dim derivative fast: ", kernel)
        r = range(0.0, 2π, length=N_pd)
        vals = [sin(x)*sin(y)*sin(z) for x in r, y in r, z in r]
        test_pts = [(r[i] + r[i+1]) / 2 for i in 6:N_pd-6]
        for (deriv, exact) in perdim_3d_cases
            itp = convolution_interpolation((r, r, r), vals; degree=(kernel,kernel,kernel),
                      derivative=deriv, fast=true, kernel_bc=:polynomial)
            interpolated = [itp(x, y, z) for x in test_pts, y in test_pts, z in test_pts]
            analytical   = [exact(x, y, z) for x in test_pts, y in test_pts, z in test_pts]
            @test interpolated ≈ analytical atol=tolerance_pd
        end
    end
end