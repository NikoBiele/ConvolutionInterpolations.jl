"""
(itp::FastConvolutionInterpolation{T,2,...})(x::Number, y::Number) — IntegralOrder
Evaluate 2D fast antiderivative at coordinates (x, y).

The 2D domain decomposes into regions around the local stencil box:
  center (K̃×K̃)       — O(eqs²) double loop over local stencil
  left strips (K̃×tail) — O(eqs) per strip via 1D prefix sum lookup
  left corner (tail×tail) — O(1) via precomputed 2D cross-sum array

Regions right of the stencil in either dimension contribute nothing: those coefficients lie
right of the anchor's stencil too, where K̃ is saturated at −½ both at x and at the anchor, so
their anchored weight is exactly zero.

K̃ weights come from exact column polynomials, evaluated once per dimension
(see `_kernel_weights`, `_antiderivative_weight`).

Result is (center + strips + corner) * h[1] * h[2], anchored to zero at the
leftmost interior knot in each dimension. O(1) in grid size, allocation-free.
See also: FastConvolutionInterpolation, convolution_fast_integration_1d.
"""

@inline function (itp::FastConvolutionInterpolation{T,2,2,TCoefs,Axs,KA,Val{2},
                    DG,EQ,KBC,FastIntegralOrder,FD,SD,Val{SG},Val{false},Val{2}})(x::Vararg{Number,2}) where 
                    {T<:AbstractFloat,TCoefs<:AbstractArray{T,2},
                    Axs<:Tuple{<:AbstractVector,<:AbstractVector},
                    KA<:Tuple{<:Nothing,<:Nothing},DG,EQ<:Tuple{Int,Int},KBC<:Tuple{<:Tuple{Symbol,Symbol},<:Tuple{Symbol,Symbol}},
                    FD,SD,SG}

    x = T.(x)
    eqs_1 = itp.eqs[1]
    eqs_2 = itp.eqs[2]
    result = zero(T)

    i1_float = (x[1] - itp.knots[1][1]) / itp.h[1] + one(T)
    i1       = clamp(floor(Int, i1_float), eqs_1, size(itp.coefs, 1) - eqs_1)
    i2_float = (x[2] - itp.knots[2][1]) / itp.h[2] + one(T)
    i2       = clamp(floor(Int, i2_float), eqs_2, size(itp.coefs, 2) - eqs_2)

    # K̃ weights per dimension at τ = t (column c ↔ coefficient j = i + eqs + 1 − c)
    w1, w2 = _column_weights_per_dim(Val(_kernel_sym(itp.kernel_sym)), Val((-1, -1)),
                                     (i1_float - T(i1), i2_float - T(i2)))

    # left tail boundary indices
    l1 = i1 - eqs_1;      l1_ok = l1 >= 1
    l2 = i2 - eqs_2;      l2_ok = l2 >= 1

    # ── center: K̃×K̃, and left strip in dim 1: tail×K̃ ───────────────
    @inbounds for j2 in (i2 - eqs_2 + 1):(i2 + eqs_2)
        kt2 = _antiderivative_weight(w2, i2, eqs_2, j2)
        lv2 = kt2 - itp.left_values[2][j2]

        # center: K̃×K̃
        @inbounds for j1 in (i1 - eqs_1 + 1):(i1 + eqs_1)
            kt1 = _antiderivative_weight(w1, i1, eqs_1, j1)
            result += itp.coefs[j1, j2] * (kt1 - itp.left_values[1][j1]) * lv2
        end

        # left strip in dim 1: O(1) lookup weighted by lv2
        result += (l1_ok ? itp.tail1_left[1][l1, j2] : zero(T)) * lv2
    end

    # ── left strip in dim 2: K̃×tail, and the left corner: tail×tail ──
    if l2_ok
        @inbounds for j1 in (i1 - eqs_1 + 1):(i1 + eqs_1)
            kt1 = _antiderivative_weight(w1, i1, eqs_1, j1)
            result += itp.tail1_left[2][j1, l2] * (kt1 - itp.left_values[1][j1])
        end
        result += l1_ok ? itp.tail2_ll[l1, l2] : zero(T)
    end

    return result * itp.h[1] * itp.h[2]
end