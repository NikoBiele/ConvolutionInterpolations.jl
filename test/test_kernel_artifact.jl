println("\n" * "-"^60)
println("Testing kernels artifact...")
println("-"^60)

# The shipped kernel tables (the "kernel_tables" artifact) must equal an exact
# rational recomputation rounded to Float64, for every shipped (kernel, derivative).
@testset "Kernel table artifact" begin
    CI = ConvolutionInterpolations
    for kernel in (:a0, :a1, :a3, :a4, :a5, :a7, :b5, :b7, :b9, :b11, :b13)
        for derivative in CI._shipped_derivatives(kernel)
            # Tables as read from the artifact: (pre_range, kp, kd1, kd2)
            shipped = CI._build_kernel_tables(kernel, derivative, Float64)
            # Same tables recomputed at exact rational precision
            exact = CI.precompute_kernel_and_range(kernel; precompute=101,
                                                   F=Float64, derivative=derivative)
            @test shipped[1] == exact[1]
            @test shipped[2] == exact[2]
            @test shipped[3] == exact[3]
            @test shipped[4] == exact[4]
        end
    end
end