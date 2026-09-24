"""
(itp::FastConvolutionInterpolation{T,3,...})(x::Number, y::Number, z::Number) — IntegralOrder
Evaluate 3D fast antiderivative at coordinates (x, y, z).

The 3D domain decomposes into regions around the local stencil box:
  center (K̃×K̃×K̃)          — O(eqs³) triple loop over local stencil
  left faces (tail×K̃×K̃)    — O(eqs²) per face via 1D prefix sum lookup
  left edges (tail×tail×K̃) — O(eqs) per edge via 2D cross-sum lookup
  left corner (tail×tail×tail) — O(1) via 3D cross-sum lookup

Regions right of the stencil in any dimension contribute nothing: those coefficients lie right
of the anchor's stencil too, where K̃ is saturated at −½ both at x and at the anchor, so their
anchored weight is exactly zero.

K̃ weights come from exact column polynomials, evaluated once per dimension
(see `_kernel_weights`, `_antiderivative_weight`).

Result is (center + faces + edges + corner) * h[1] * h[2] * h[3], anchored to zero at the
leftmost interior knot in each dimension. O(1) in grid size, allocation-free.
See also: FastConvolutionInterpolation, convolution_fast_integration_2d.
"""
@inline function (itp::FastConvolutionInterpolation{T,3,3,TCoefs,Axs,KA,Val{3},
                    DG,EQ,KBC,FastIntegralOrder,FD,SD,Val{SG},Val{false},Val{3}})(x::Vararg{Number,3}) where
                    {T<:AbstractFloat,TCoefs<:AbstractArray{T,3},
                    Axs<:Tuple{<:AbstractVector,<:AbstractVector,<:AbstractVector},
                    KA<:Tuple{<:Nothing,<:Nothing,<:Nothing},DG,EQ<:Tuple{Int,Int,Int},
                    KBC<:Tuple{<:Tuple{Symbol,Symbol},<:Tuple{Symbol,Symbol},<:Tuple{Symbol,Symbol}},
                    FD,SD,SG}

    x = T.(x)
    result = zero(T)

    i1_float = (x[1] - itp.knots[1][1]) / itp.h[1] + one(T)
    i1       = clamp(floor(Int, i1_float), itp.eqs[1], size(itp.coefs, 1) - itp.eqs[1])
    i2_float = (x[2] - itp.knots[2][1]) / itp.h[2] + one(T)
    i2       = clamp(floor(Int, i2_float), itp.eqs[2], size(itp.coefs, 2) - itp.eqs[2])
    i3_float = (x[3] - itp.knots[3][1]) / itp.h[3] + one(T)
    i3       = clamp(floor(Int, i3_float), itp.eqs[3], size(itp.coefs, 3) - itp.eqs[3])

    # K̃ weights per dimension at τ = t (column c ↔ coefficient j = i + eqs + 1 − c)
    w1, w2, w3 = _column_weights_per_dim(Val(_kernel_sym(itp.kernel_sym)), Val((-1, -1, -1)),
                                         (i1_float - T(i1), i2_float - T(i2), i3_float - T(i3)))

    # left tail boundary indices
    l1 = i1 - itp.eqs[1];      l1_ok = l1 >= 1
    l2 = i2 - itp.eqs[2];      l2_ok = l2 >= 1
    l3 = i3 - itp.eqs[3];      l3_ok = l3 >= 1

    # ── center: K̃×K̃×K̃, left face in dim 1, left edge along dim 3 ─────────────
    @inbounds for j3 in (i3 - itp.eqs[3] + 1):(i3 + itp.eqs[3])
        lv3 = _antiderivative_weight(w3, i3, itp.eqs[3], j3) - itp.left_values[3][j3]

        @inbounds for j2 in (i2 - itp.eqs[2] + 1):(i2 + itp.eqs[2])
            lv2 = _antiderivative_weight(w2, i2, itp.eqs[2], j2) - itp.left_values[2][j2]
            lv23 = lv2 * lv3

            # center: K̃₁ × K̃₂ × K̃₃
            @inbounds for j1 in (i1 - itp.eqs[1] + 1):(i1 + itp.eqs[1])
                result += itp.coefs[j1, j2, j3] *
                    (_antiderivative_weight(w1, i1, itp.eqs[1], j1) - itp.left_values[1][j1]) * lv23
            end

            # left face in dim 1: tail1[1] × K̃₂ × K̃₃
            result += (l1_ok ? itp.tail1_left[1][l1, j2, j3] : zero(T)) * lv23
        end

        # left face in dim 2: K̃₁ × tail1[2] × K̃₃
        if l2_ok
            @inbounds for j1 in (i1 - itp.eqs[1] + 1):(i1 + itp.eqs[1])
                lv1 = _antiderivative_weight(w1, i1, itp.eqs[1], j1) - itp.left_values[1][j1]
                result += itp.tail1_left[2][j1, l2, j3] * lv1 * lv3
            end
        end

        # left edge along dim 3: tail × tail × K̃₃ (saturated in dims 1 & 2)
        result += (l1_ok && l2_ok ? itp.tail3_edge_ll[3][l1, l2, j3] : zero(T)) * lv3
    end

    # left face in dim 3: K̃₁ × K̃₂ × tail1[3]
    if l3_ok
        @inbounds for j2 in (i2 - itp.eqs[2] + 1):(i2 + itp.eqs[2])
            lv2 = _antiderivative_weight(w2, i2, itp.eqs[2], j2) - itp.left_values[2][j2]
            @inbounds for j1 in (i1 - itp.eqs[1] + 1):(i1 + itp.eqs[1])
                lv1 = _antiderivative_weight(w1, i1, itp.eqs[1], j1) - itp.left_values[1][j1]
                result += itp.tail1_left[3][j1, j2, l3] * lv1 * lv2
            end
        end
    end

    # left edge along dim 2: tail × K̃₂ × tail (saturated in dims 1 & 3)
    if l1_ok && l3_ok
        @inbounds for j2 in (i2 - itp.eqs[2] + 1):(i2 + itp.eqs[2])
            lv2 = _antiderivative_weight(w2, i2, itp.eqs[2], j2) - itp.left_values[2][j2]
            result += itp.tail3_edge_ll[2][l1, j2, l3] * lv2
        end
    end

    # left edge along dim 1: K̃₁ × tail × tail (saturated in dims 2 & 3)
    if l2_ok && l3_ok
        @inbounds for j1 in (i1 - itp.eqs[1] + 1):(i1 + itp.eqs[1])
            lv1 = _antiderivative_weight(w1, i1, itp.eqs[1], j1) - itp.left_values[1][j1]
            result += itp.tail3_edge_ll[1][j1, l2, l3] * lv1
        end
    end

    # ── left corner: tail × tail × tail ─────────────────────────────────────────
    result += (l1_ok && l2_ok && l3_ok ? itp.tail3_corner_lll[l1, l2, l3] : zero(T))

    return result * itp.h[1] * itp.h[2] * itp.h[3]
end