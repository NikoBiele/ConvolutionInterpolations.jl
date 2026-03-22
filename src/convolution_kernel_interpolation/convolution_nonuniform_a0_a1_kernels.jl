# ═══════════════════════════════════════════════════════════════
# 1D eval
# ═══════════════════════════════════════════════════════════════

@inline function (itp::ConvolutionInterpolation{T,1,TCoefs,IT,Axs,KA,Val{1},
                    NonUniformNonMixedLowKernel{DG},EQ,KBC,DerivativeOrder{DO},
                    FD,SD,Val{(:not_used)},Val{:nonuniform},Val{true}})(x::Vararg{Number,1}) where 
                    {T,TCoefs,IT,Axs,KA<:Tuple{<:Nothing},DG,EQ<:Tuple{Int},KBC,DO,FD,SD}

    n_k = length(itp.knots[1])
    n_c = length(itp.coefs)
    ng = (n_k - n_c) ÷ 2

    if DG === :a0
        i = clamp(searchsortedlast(itp.knots[1], x[1]), 1 + ng, n_k - 1 - ng)
        xi = (x[1] - itp.knots[1][i]) <= (itp.knots[1][i+1] - itp.knots[1][i]) / 2 ? i : i + 1
        return itp.coefs[xi - ng]

    elseif DG === :a1
        i = clamp(searchsortedlast(itp.knots[1], x[1]), 1 + ng, n_k - 1 - ng)
        t = (T(x[1]) - itp.knots[1][i]) / (itp.knots[1][i+1] - itp.knots[1][i])
        return (one(T) - t) * itp.coefs[i - ng] + t * itp.coefs[i + 1 - ng]
    end
end

# ═══════════════════════════════════════════════════════════════
# 2D eval
# ═══════════════════════════════════════════════════════════════

function (itp::ConvolutionInterpolation{T,2,TCoefs,IT,Axs,KA,Val{2},
                    NonUniformNonMixedLowKernel{DG},EQ,KBC,DerivativeOrder{DO},
                    FD,SD,SG,Val{:nonuniform},Val{true}})(x::Vararg{Number,2}) where 
                    {T,TCoefs,IT,Axs,KA<:Tuple{<:Nothing,<:Nothing},DG,EQ<:Tuple{Int,Int},KBC,DO,FD,SD,SG}

    ng = ntuple(d -> (length(itp.knots[d]) - size(itp.coefs,d)) ÷ 2, 2)

    if DG === :a0
        ids = ntuple(2) do d
            i = clamp(searchsortedlast(itp.knots[d], x[d]), 1 + ng[d], length(itp.knots[d]) - 1 - ng[d])
            xi = (x[d] - itp.knots[d][i]) <= (itp.knots[d][i+1] - itp.knots[d][i]) / 2 ? i : i + 1
            xi - ng[d]
        end
        return @inbounds itp.coefs[ids...]

    elseif DG === :a1
        iw = ntuple(2) do d
            i = clamp(searchsortedlast(itp.knots[d], x[d]), 1 + ng[d], length(itp.knots[d]) - 1 - ng[d])
            t = (T(x[d]) - itp.knots[d][i]) / (itp.knots[d][i+1] - itp.knots[d][i])
            (i - ng[d], t)
        end
        i1, t1 = iw[1]; i2, t2 = iw[2]
        @inbounds return (one(T)-t1)*(one(T)-t2)*itp.coefs[i1,i2] +
                            t1*(one(T)-t2)*itp.coefs[i1+1,i2] +
                            (one(T)-t1)*t2*itp.coefs[i1,i2+1] +
                            t1*t2*itp.coefs[i1+1,i2+1]
    end
end

# ═══════════════════════════════════════════════════════════════
# 3D eval
# ═══════════════════════════════════════════════════════════════

function (itp::ConvolutionInterpolation{T,3,TCoefs,IT,Axs,KA,Val{3},
                    NonUniformNonMixedLowKernel{DG},EQ,KBC,DerivativeOrder{DO},
                    FD,SD,SG,Val{:nonuniform},Val{true}})(x::Vararg{Number,3}) where 
                    {T,TCoefs,IT,Axs,KA<:Tuple{<:Nothing,<:Nothing,<:Nothing},DG,EQ<:Tuple{Int,Int,Int},KBC,DO,FD,SD,SG}

    ng = ntuple(d -> (length(itp.knots[d]) - size(itp.coefs,d)) ÷ 2, 3)

    if DG === :a0
        ids = ntuple(3) do d
            i = clamp(searchsortedlast(itp.knots[d], x[d]), 1 + ng[d], length(itp.knots[d]) - 1 - ng[d])
            xi = (x[d] - itp.knots[d][i]) <= (itp.knots[d][i+1] - itp.knots[d][i]) / 2 ? i : i + 1
            xi - ng[d]
        end
        return @inbounds itp.coefs[ids...]

    elseif DG === :a1
        iw = ntuple(3) do d
            i = clamp(searchsortedlast(itp.knots[d], x[d]), 1 + ng[d], length(itp.knots[d]) - 1 - ng[d])
            t = (T(x[d]) - itp.knots[d][i]) / (itp.knots[d][i+1] - itp.knots[d][i])
            (i - ng[d], t)
        end
        result = zero(T)
        @inbounds for c3 in 0:1, c2 in 0:1, c1 in 0:1
            w = (c1 == 0 ? one(T) - iw[1][2] : iw[1][2]) *
                (c2 == 0 ? one(T) - iw[2][2] : iw[2][2]) *
                (c3 == 0 ? one(T) - iw[3][2] : iw[3][2])
            result += w * itp.coefs[iw[1][1]+c1, iw[2][1]+c2, iw[3][1]+c3]
        end
        return result
    end
end

# ═══════════════════════════════════════════════════════════════
# ND eval
# ═══════════════════════════════════════════════════════════════

function (itp::ConvolutionInterpolation{T,N,TCoefs,IT,Axs,KA,HigherDimension{N},
                    NonUniformNonMixedLowKernel{DG},EQ,KBC,DerivativeOrder{DO},
                    FD,SD,SG,Val{:nonuniform},Val{true}})(x::Vararg{Number,N}) where 
                    {T,N,TCoefs,IT,KA<:NTuple{N,<:Nothing},Axs,DG,EQ<:NTuple{N,Int},KBC,DO,FD,SD,SG}

    ng = ntuple(d -> (length(itp.knots[d]) - size(itp.coefs,d)) ÷ 2, N)

    if DG === Val{:a0}
        ids = ntuple(N) do d
            i = clamp(searchsortedlast(itp.knots[d], x[d]), 1 + ng[d], length(itp.knots[d]) - 1 - ng[d])
            xi = (x[d] - itp.knots[d][i]) <= (itp.knots[d][i+1] - itp.knots[d][i]) / 2 ? i : i + 1
            xi - ng[d]
        end
        return @inbounds itp.coefs[ids...]

    elseif DG === Val{:a1}
        iw = ntuple(N) do d
            i = clamp(searchsortedlast(itp.knots[d], x[d]), 1 + ng[d], length(itp.knots[d]) - 1 - ng[d])
            t = (T(x[d]) - itp.knots[d][i]) / (itp.knots[d][i+1] - itp.knots[d][i])
            (i - ng[d], t)
        end
        result = zero(T)
        @inbounds for offsets in Iterators.product(ntuple(_ -> 0:1, N)...)
            w = prod(d -> offsets[d] == 0 ? one(T) - iw[d][2] : iw[d][2], 1:N)
            idx = ntuple(d -> iw[d][1] + offsets[d], N)
            result += w * itp.coefs[idx...]
        end
        return result
    end
end