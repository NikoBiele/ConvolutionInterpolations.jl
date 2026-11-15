# Deterministic test data, identical on every Julia version and platform.
#
# The streams of Julia's seeded generators are only stable within a minor version, so a test
# calibrated on one draw can fail on another version with nothing wrong in the package. The
# functions here use integer arithmetic and a few exactly rounded floating-point operations
# only, so every test sees the same numbers everywhere:
#
#   halton_points(D, n)        scattered points: evenly spread, no two close together
#   stable_rand(seed, dims...)   uniform numbers in [0, 1)
#   stable_randn(seed, dims...)  standard normal numbers, for noise
#
# Different seeds give unrelated streams; the same seed always gives the same numbers.

"""Radical inverse of `i` in `base` (van der Corput): the digits of `i` in that base, mirrored
about the point. A number in [0, 1); consecutive `i` fill the interval evenly."""
function vdc(i::Int, base::Int)
    r = 0.0
    f = 1.0 / base
    k = i
    while k > 0
        r += f * (k % base)
        k = div(k, base)
        f /= base
    end
    return r
end

const HALTON_BASES = (2, 3, 5, 7, 11, 13)   # one prime per axis, up to six dimensions

"""`D x n` matrix of Halton points in [0, 1)^D, one point per column: axis `d` is the van der
Corput sequence in the `d`-th prime. `skip` drops that many leading points, which gives a
second point set that shares no point with the first (data and test points, for instance)."""
function halton_points(D::Int, n::Int; skip::Int = 0)
    return [vdc(i + skip, HALTON_BASES[d]) for d in 1:D, i in 1:n]
end

"""SplitMix64: a 64-bit mixer in integer arithmetic. Nearby inputs give unrelated outputs."""
function splitmix64(x::UInt64)
    x += 0x9e3779b97f4a7c15
    z = xor(x, x >> 30) * 0xbf58476d1ce4e5b9
    z = xor(z, z >> 27) * 0x94d049bb133111eb
    return xor(z, z >> 31)
end

"""Array of the given `dims` of uniform numbers in [0, 1): entry `i` is SplitMix64 of a
counter, with the stream chosen by the nonnegative integer `seed`."""
function stable_rand(seed::Int, dims::Int...)
    out = Array{Float64}(undef, dims...)
    stream = splitmix64(UInt64(seed))                      # unrelated streams for nearby seeds
    for i in eachindex(out)
        bits = splitmix64(stream + UInt64(i)) >> 11        # the top 53 bits
        out[i] = bits * 2.0^-53
    end
    return out
end

"""Array of the given `dims` of standard normal numbers (Box-Muller on `stable_rand`), with
the stream chosen by `seed`."""
function stable_randn(seed::Int, dims::Int...)
    out = Array{Float64}(undef, dims...)
    u = stable_rand(seed, 2, length(out))                  # two uniform numbers per normal one
    for i in eachindex(out)
        out[i] = sqrt(-2 * log(1 - u[1, i])) * cos(2 * pi * u[2, i])
    end
    return out
end