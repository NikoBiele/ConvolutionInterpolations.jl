# A pure N-D integral (derivative = -1 in every dimension) of separable data must equal the
# product of the 1D antiderivatives exactly: the eager ghost extension is linear and acts per
# dimension, so the N-D interpolant is the product of the 1D interpolants, and so is its
# anchored N-fold antiderivative. Evaluation points include the last knots, where missing
# coefficients in the N-D summation would show up first.
@testset "4D integral of separable data equals product of 1D integrals" begin
    # A different grid size and function per dimension, so dimensions cannot be confused
    x1 = range(0.0, 2.0, length=12)
    x2 = range(-1.0, 1.0, length=11)
    x3 = range(0.5, 3.0, length=10)
    x4 = range(0.0, 1.5, length=13)
    g1(x) = sin(x) + 2.0
    g2(y) = cos(2y) + 1.5
    g3(z) = exp(z / 3)
    g4(w) = 1.0 + w^2

    # Separable 4D data and its 4-fold antiderivative interpolant
    vs = [g1(a) * g2(b) * g3(c) * g4(d) for a in x1, b in x2, c in x3, d in x4]
    itp4 = convolution_interpolation((x1, x2, x3, x4), vs; kernel=:b5, derivative=-1, bc=:poly)

    # The four 1D antiderivative interpolants of the same kernel and boundary condition
    itp1 = convolution_interpolation(x1, g1.(x1); kernel=:b5, derivative=-1, bc=:poly)
    itp2 = convolution_interpolation(x2, g2.(x2); kernel=:b5, derivative=-1, bc=:poly)
    itp3 = convolution_interpolation(x3, g3.(x3); kernel=:b5, derivative=-1, bc=:poly)
    itp4_1d = convolution_interpolation(x4, g4.(x4); kernel=:b5, derivative=-1, bc=:poly)

    # Evaluation points: the anchor, the far corner, points just inside the far ends, and
    # random interior points
    points = [(x1[1], x2[1], x3[1], x4[1]),
              (x1[end], x2[end], x3[end], x4[end]),
              (x1[end] - 0.3 * step(x1), x2[end] - 0.3 * step(x2),
               x3[end] - 0.3 * step(x3), x4[end] - 0.3 * step(x4)),
              (x1[end], x2[1], x3[end], x4[1]),
              (x1[1], x2[end], x3[1], x4[end])]
    rng = Random.MersenneTwister(7)
    for _ in 1:8
        push!(points, (x1[1] + rand(rng) * (x1[end] - x1[1]),
                       x2[1] + rand(rng) * (x2[end] - x2[1]),
                       x3[1] + rand(rng) * (x3[end] - x3[1]),
                       x4[1] + rand(rng) * (x4[end] - x4[1])))
    end

    for (a, b, c, d) in points
        expected = itp1(a) * itp2(b) * itp3(c) * itp4_1d(d)
        @test isapprox(itp4(a, b, c, d), expected; rtol=1e-11, atol=1e-12)
    end
end