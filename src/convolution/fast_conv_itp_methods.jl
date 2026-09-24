@inline _to_eltype(::Type{T}, x) where T = eltype(x) === T ? x : T.(x)

function convolution_interpolation(knots::AbstractVector, values::Array{T,1}) where {T}
   knots_t = (eltype(knots) === T ? knots : T.(knots),)
   return ConvolutionExtrapolation(
       _build_fast_uniform_convolution(knots_t, values,
           ((:detect, :detect),), false,
           Val((:b7,)), Val(false), Val((0,))),
       Throw())
end

function convolution_interpolation(knots::NTuple{1,AbstractVector}, values::Array{T,1}) where {T}
    knots_t = ntuple(d -> _to_eltype(T, knots[d]), 1)
    return ConvolutionExtrapolation(
        _build_fast_uniform_convolution(knots_t, values,
            ((:detect, :detect),), false,
            Val((:b7,)), Val(false), Val((0,))),
        Throw())
end

function convolution_interpolation(knots::NTuple{2,AbstractVector}, values::Array{T,2}) where {T}
    knots_t = ntuple(d -> _to_eltype(T, knots[d]), 2)
    return ConvolutionExtrapolation(
        _build_fast_uniform_convolution(knots_t, values,
            ((:detect, :detect), (:detect, :detect)), false,
            Val((:b7, :b7)), Val(false), Val((0, 0))),
        Throw())
end

function convolution_interpolation(knots::NTuple{3,AbstractVector}, values::Array{T,3}) where {T}
    knots_t = ntuple(d -> _to_eltype(T, knots[d]), 3)
    return ConvolutionExtrapolation(
        _build_fast_uniform_convolution(knots_t, values,
            ((:detect, :detect), (:detect, :detect), (:detect, :detect)), false,
            Val((:b5, :b5, :b5)), Val(false), Val((0, 0, 0))),
        Throw())
end

function convolution_interpolation(knots::NTuple{4,AbstractVector}, values::Array{T,4}) where {T}
    knots_t = ntuple(d -> _to_eltype(T, knots[d]), 4)
    return ConvolutionExtrapolation(
        _build_fast_uniform_convolution(knots_t, values,
            ((:detect, :detect), (:detect, :detect), (:detect, :detect), (:detect, :detect)), true,
            Val((:a4, :a4, :a4, :a4)), Val(true), Val((0, 0, 0, 0))),
        Throw())
end

function convolution_interpolation(knots::NTuple{5,AbstractVector}, values::Array{T,5}) where {T}
    knots_t = ntuple(d -> _to_eltype(T, knots[d]), 5)
    return ConvolutionExtrapolation(
        _build_fast_uniform_convolution(knots_t, values,
            ((:detect, :detect), (:detect, :detect), (:detect, :detect), (:detect, :detect), (:detect, :detect)), true,
            Val((:a4, :a4, :a4, :a4, :a4)), Val(true), Val((0, 0, 0, 0, 0))),
        Throw())
end

function convolution_interpolation(knots::NTuple{6,AbstractVector}, values::Array{T,6}) where {T}
    knots_t = ntuple(d -> _to_eltype(T, knots[d]), 6)
    return ConvolutionExtrapolation(
        _build_fast_uniform_convolution(knots_t, values,
            ((:detect, :detect), (:detect, :detect), (:detect, :detect), (:detect, :detect), (:detect, :detect), (:detect, :detect)), true,
            Val((:a3, :a3, :a3, :a3, :a3, :a3)), Val(true), Val((0, 0, 0, 0, 0, 0))),
        Throw())
end

function convolution_interpolation(knots::NTuple{7,AbstractVector}, values::Array{T,7}) where {T}
    knots_t = ntuple(d -> _to_eltype(T, knots[d]), 7)
    return ConvolutionExtrapolation(
        _build_fast_uniform_convolution(knots_t, values,
            ((:detect, :detect), (:detect, :detect), (:detect, :detect), (:detect, :detect), (:detect, :detect), (:detect, :detect), (:detect, :detect)), true,
            Val((:a3, :a3, :a3, :a3, :a3, :a3, :a3)), Val(true), Val((0, 0, 0, 0, 0, 0, 0))),
        Throw())
end

function convolution_interpolation(knots::NTuple{8,AbstractVector}, values::Array{T,8}) where {T}
    knots_t = ntuple(d -> _to_eltype(T, knots[d]), 8)
    return ConvolutionExtrapolation(
        _build_fast_uniform_convolution(knots_t, values,
            ((:detect, :detect), (:detect, :detect), (:detect, :detect), (:detect, :detect), (:detect, :detect), (:detect, :detect), (:detect, :detect), (:detect, :detect)), true,
            Val((:a3, :a3, :a3, :a3, :a3, :a3, :a3, :a3)), Val(true), Val((0, 0, 0, 0, 0, 0, 0, 0))),
        Throw())
end

function convolution_interpolation(knots::NTuple{9,AbstractVector}, values::Array{T,9}) where {T}
    knots_t = ntuple(d -> _to_eltype(T, knots[d]), 9)
    return ConvolutionExtrapolation(
        _build_fast_uniform_convolution(knots_t, values,
            ((:detect, :detect), (:detect, :detect), (:detect, :detect), (:detect, :detect), (:detect, :detect), (:detect, :detect), (:detect, :detect), (:detect, :detect), (:detect, :detect)), true,
            Val((:a3, :a3, :a3, :a3, :a3, :a3, :a3, :a3, :a3)), Val(true), Val((0, 0, 0, 0, 0, 0, 0, 0, 0))),
        Throw())
end

function convolution_interpolation(knots::NTuple{10,AbstractVector}, values::Array{T,10}) where {T}
    knots_t = ntuple(d -> _to_eltype(T, knots[d]), 10)
    return ConvolutionExtrapolation(
        _build_fast_uniform_convolution(knots_t, values,
            ((:detect, :detect), (:detect, :detect), (:detect, :detect), (:detect, :detect), (:detect, :detect), (:detect, :detect), (:detect, :detect), (:detect, :detect), (:detect, :detect), (:detect, :detect)), true,
            Val((:a3, :a3, :a3, :a3, :a3, :a3, :a3, :a3, :a3, :a3)), Val(true), Val((0, 0, 0, 0, 0, 0, 0, 0, 0, 0))),
        Throw())
end