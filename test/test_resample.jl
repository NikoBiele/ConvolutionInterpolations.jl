println("Testing convolution_resample...")
@testset "convolution_resample" begin

    @testset "1D resample preserves signal" begin
        xs_in  = range(0.0, 2π, length=50)
        xs_out = range(0.0, 2π, length=100)
        signal = sin.(xs_in)
        result = convolution_resample((xs_in,), (xs_out,), signal)
        analytical = sin.(xs_out)
        @test length(result) == 100
        @test eltype(result) == Float64
        @test maximum(abs.(result .- analytical)) < 1e-5
    end

    @testset "1D resample kernel accuracy ordering" begin
        xs_in  = range(0.0, 2π, length=50)
        xs_out = range(0.0, 2π, length=100)
        signal = sin.(xs_in)
        analytical = sin.(xs_out)
        errors = map((:a1, :a3, :a4, :b5)) do k
            subgrid = k == :a1 ? :linear : :cubic
            r = convolution_resample((xs_in,), (xs_out,), signal; kernel=k, subgrid=subgrid)
            maximum(abs.(r .- analytical))
        end
        # higher order kernels should be at least as accurate
        @test errors[1] >= errors[2] >= errors[3] >= errors[4]
    end

    @testset "2D resample preserves signal" begin
        xs_in  = range(0.0, 2π, length=50)
        xs_out = range(0.0, 2π, length=60)
        signal = [sin(x)*cos(y) for x in xs_in, y in xs_in]
        result = convolution_resample((xs_in, xs_in), (xs_out, xs_out), signal)
        analytical = [sin(x)*cos(y) for x in xs_out, y in xs_out]
        @test size(result) == (60, 60)
        @test eltype(result) == Float64
        @test maximum(abs.(result .- analytical)) < 1e-4
    end

    @testset "3D resample preserves signal" begin
        xs_in  = range(0.0, 2π, length=40)
        xs_out = range(0.0, 2π, length=60)
        signal = [sin(x)*cos(y)*sin(z) for x in xs_in, y in xs_in, z in xs_in]
        result = convolution_resample((xs_in, xs_in, xs_in),
                    (xs_out, xs_out, xs_out), signal; kernel=:b7)
        analytical = [sin(x)*cos(y)*sin(z) for x in xs_out, y in xs_out, z in xs_out]
        @test size(result) == (60, 60, 60)
        @test maximum(abs.(result .- analytical)) < 1e-4
    end

    @testset "resample output size" begin
        for N in (1, 2, 3)
            n_in  = 20
            n_out = 35
            ks_in  = ntuple(_ -> range(0.0, 2π, length=n_in),  N)
            ks_out = ntuple(_ -> range(0.0, 2π, length=n_out), N)
            vs     = randn(ntuple(_ -> n_in, N)...)
            result = convolution_resample(ks_in, ks_out, vs)
            @test size(result) == ntuple(_ -> n_out, N)
            @test eltype(result) == Float64
        end
    end

    @testset "resample derivative" begin
        xs_in  = range(0.0, 2π, length=50)
        xs_out = range(0.0, 2π, length=100)
        signal = sin.(xs_in)
        result = convolution_resample((xs_in,), (xs_out,), signal; kernel=:b7, derivative=1)
        analytical = cos.(xs_out)
        @test maximum(abs.(result .- analytical)) < 1e-3
    end

    @testset "invalid subgrid/kernel combination" begin
        xs_in  = range(0.0, 2π, length=50)
        xs_out = range(0.0, 2π, length=100)
        signal = sin.(xs_in)
        @test_throws ErrorException convolution_resample(
            (xs_in,), (xs_out,), signal;
            kernel=:a1, subgrid=:cubic)
    end

end