@inline _to_eltype(::Type{T}, x) where T = eltype(x) === T ? x : T.(x)

# Keyword-free methods of `convolution_interpolation`: the defaults of the keyword method (kernel,
# :detect boundaries, no derivative, throwing extrapolation), built directly on the uniform fast
# path, lazy or eager as the keyword method's default `lazy=nothing` chooses (`_default_lazy`).
# Nonuniform knots are handed to the keyword method with its default kernel, which selects the
# nonuniform path, so a call without keywords behaves exactly like one with default keywords.
@inline _all_uniform(knots) = all(k -> is_uniform_grid(k), knots)

function convolution_interpolation(knots::AbstractVector, values::Array{T,1}) where {T}
   is_uniform_grid(knots) || return convolution_interpolation(knots, values; kernel=:auto)
   knots_t = (eltype(knots) === T ? knots : T.(knots),)
   kernels = (:b7,)
   lazy = _default_lazy(size(values), kernels, (0,))
   return ConvolutionExtrapolation(
       _build_fast_uniform_convolution(knots_t, values,
           ((:detect, :detect),),
           Val(kernels), Val(lazy), Val((0,))),
       Throw())
end

function convolution_interpolation(knots::NTuple{1,AbstractVector}, values::Array{T,1}) where {T}
    _all_uniform(knots) || return convolution_interpolation(knots, values; kernel=:auto)
    knots_t = ntuple(d -> _to_eltype(T, knots[d]), 1)
    kernels = (:b7,)
    lazy = _default_lazy(size(values), kernels, (0,))
    return ConvolutionExtrapolation(
        _build_fast_uniform_convolution(knots_t, values,
            ((:detect, :detect),),
            Val(kernels), Val(lazy), Val((0,))),
        Throw())
end

function convolution_interpolation(knots::NTuple{2,AbstractVector}, values::Array{T,2}) where {T}
    _all_uniform(knots) || return convolution_interpolation(knots, values; kernel=:auto)
    knots_t = ntuple(d -> _to_eltype(T, knots[d]), 2)
    kernels = (:b7, :b7)
    lazy = _default_lazy(size(values), kernels, (0, 0))
    return ConvolutionExtrapolation(
        _build_fast_uniform_convolution(knots_t, values,
            ((:detect, :detect), (:detect, :detect)),
            Val(kernels), Val(lazy), Val((0, 0))),
        Throw())
end

function convolution_interpolation(knots::NTuple{3,AbstractVector}, values::Array{T,3}) where {T}
    _all_uniform(knots) || return convolution_interpolation(knots, values; kernel=:auto)
    knots_t = ntuple(d -> _to_eltype(T, knots[d]), 3)
    kernels = (:b5, :b5, :b5)
    lazy = _default_lazy(size(values), kernels, (0, 0, 0))
    return ConvolutionExtrapolation(
        _build_fast_uniform_convolution(knots_t, values,
            ((:detect, :detect), (:detect, :detect), (:detect, :detect)),
            Val(kernels), Val(lazy), Val((0, 0, 0))),
        Throw())
end

function convolution_interpolation(knots::NTuple{4,AbstractVector}, values::Array{T,4}) where {T}
    _all_uniform(knots) || return convolution_interpolation(knots, values; kernel=:auto)
    knots_t = ntuple(d -> _to_eltype(T, knots[d]), 4)
    kernels = (:a4, :a4, :a4, :a4)
    lazy = _default_lazy(size(values), kernels, (0, 0, 0, 0))
    return ConvolutionExtrapolation(
        _build_fast_uniform_convolution(knots_t, values,
            ((:detect, :detect), (:detect, :detect), (:detect, :detect), (:detect, :detect)),
            Val(kernels), Val(lazy), Val((0, 0, 0, 0))),
        Throw())
end

function convolution_interpolation(knots::NTuple{5,AbstractVector}, values::Array{T,5}) where {T}
    _all_uniform(knots) || return convolution_interpolation(knots, values; kernel=:auto)
    knots_t = ntuple(d -> _to_eltype(T, knots[d]), 5)
    kernels = (:a4, :a4, :a4, :a4, :a4)
    lazy = _default_lazy(size(values), kernels, (0, 0, 0, 0, 0))
    return ConvolutionExtrapolation(
        _build_fast_uniform_convolution(knots_t, values,
            ((:detect, :detect), (:detect, :detect), (:detect, :detect), (:detect, :detect), (:detect, :detect)),
            Val(kernels), Val(lazy), Val((0, 0, 0, 0, 0))),
        Throw())
end

function convolution_interpolation(knots::NTuple{6,AbstractVector}, values::Array{T,6}) where {T}
    _all_uniform(knots) || return convolution_interpolation(knots, values; kernel=:auto)
    knots_t = ntuple(d -> _to_eltype(T, knots[d]), 6)
    kernels = (:a3, :a3, :a3, :a3, :a3, :a3)
    lazy = _default_lazy(size(values), kernels, (0, 0, 0, 0, 0, 0))
    return ConvolutionExtrapolation(
        _build_fast_uniform_convolution(knots_t, values,
            ((:detect, :detect), (:detect, :detect), (:detect, :detect), (:detect, :detect), (:detect, :detect), (:detect, :detect)),
            Val(kernels), Val(lazy), Val((0, 0, 0, 0, 0, 0))),
        Throw())
end

function convolution_interpolation(knots::NTuple{7,AbstractVector}, values::Array{T,7}) where {T}
    _all_uniform(knots) || return convolution_interpolation(knots, values; kernel=:auto)
    knots_t = ntuple(d -> _to_eltype(T, knots[d]), 7)
    kernels = (:a3, :a3, :a3, :a3, :a3, :a3, :a3)
    lazy = _default_lazy(size(values), kernels, (0, 0, 0, 0, 0, 0, 0))
    return ConvolutionExtrapolation(
        _build_fast_uniform_convolution(knots_t, values,
            ((:detect, :detect), (:detect, :detect), (:detect, :detect), (:detect, :detect), (:detect, :detect), (:detect, :detect), (:detect, :detect)),
            Val(kernels), Val(lazy), Val((0, 0, 0, 0, 0, 0, 0))),
        Throw())
end

function convolution_interpolation(knots::NTuple{8,AbstractVector}, values::Array{T,8}) where {T}
    _all_uniform(knots) || return convolution_interpolation(knots, values; kernel=:auto)
    knots_t = ntuple(d -> _to_eltype(T, knots[d]), 8)
    kernels = (:a3, :a3, :a3, :a3, :a3, :a3, :a3, :a3)
    lazy = _default_lazy(size(values), kernels, (0, 0, 0, 0, 0, 0, 0, 0))
    return ConvolutionExtrapolation(
        _build_fast_uniform_convolution(knots_t, values,
            ((:detect, :detect), (:detect, :detect), (:detect, :detect), (:detect, :detect), (:detect, :detect), (:detect, :detect), (:detect, :detect), (:detect, :detect)),
            Val(kernels), Val(lazy), Val((0, 0, 0, 0, 0, 0, 0, 0))),
        Throw())
end

function convolution_interpolation(knots::NTuple{9,AbstractVector}, values::Array{T,9}) where {T}
    _all_uniform(knots) || return convolution_interpolation(knots, values; kernel=:auto)
    knots_t = ntuple(d -> _to_eltype(T, knots[d]), 9)
    kernels = (:a3, :a3, :a3, :a3, :a3, :a3, :a3, :a3, :a3)
    lazy = _default_lazy(size(values), kernels, (0, 0, 0, 0, 0, 0, 0, 0, 0))
    return ConvolutionExtrapolation(
        _build_fast_uniform_convolution(knots_t, values,
            ((:detect, :detect), (:detect, :detect), (:detect, :detect), (:detect, :detect), (:detect, :detect), (:detect, :detect), (:detect, :detect), (:detect, :detect), (:detect, :detect)),
            Val(kernels), Val(lazy), Val((0, 0, 0, 0, 0, 0, 0, 0, 0))),
        Throw())
end

function convolution_interpolation(knots::NTuple{10,AbstractVector}, values::Array{T,10}) where {T}
    _all_uniform(knots) || return convolution_interpolation(knots, values; kernel=:auto)
    knots_t = ntuple(d -> _to_eltype(T, knots[d]), 10)
    kernels = (:a3, :a3, :a3, :a3, :a3, :a3, :a3, :a3, :a3, :a3)
    lazy = _default_lazy(size(values), kernels, (0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
    return ConvolutionExtrapolation(
        _build_fast_uniform_convolution(knots_t, values,
            ((:detect, :detect), (:detect, :detect), (:detect, :detect), (:detect, :detect), (:detect, :detect), (:detect, :detect), (:detect, :detect), (:detect, :detect), (:detect, :detect), (:detect, :detect)),
            Val(kernels), Val(lazy), Val((0, 0, 0, 0, 0, 0, 0, 0, 0, 0))),
        Throw())
end