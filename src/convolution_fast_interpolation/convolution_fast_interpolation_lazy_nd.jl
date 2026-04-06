@inline function (itp::FastConvolutionInterpolation{T,N,0,TCoefs,Axs,KA,HigherDimension{N},
                    LowerOrderKernel{DG},EQ,PR,KP,KBC,DerivativeOrder{DO},FD,SD,Val{SG},
                    Val{true},Val{0}})(x::Vararg{Number,N}) where {T<:AbstractFloat,N,TCoefs<:AbstractArray{T,N},
                    KA<:NTuple{N,<:Nothing},Axs<:NTuple{N,<:AbstractVector},DG,EQ<:NTuple{N,Int},
                    PR<:NTuple{N,<:AbstractVector},KP,KBC<:NTuple{N,Tuple{Symbol,Symbol}},
                    DO,FD,SD,SG}
    
    # same as eager path
    if DG[1] == :a0
        return _eval_a0_nd(itp, x)
    elseif DG[1] == :a1
        return _eval_a1_nd(itp, x, Val{DO}())
    end
end

@inline function (itp::FastConvolutionInterpolation{T,N,0,TCoefs,Axs,KA,HigherDimension{N},
            DG,EQ,PR,KP,KBC,DerivativeOrder{DO},FD,SD,Val{SG},
                    Val{true},Val{0}})(x::Vararg{Number,N}) where {T<:AbstractFloat,N,TCoefs<:AbstractArray{T,N},
                    KA<:NTuple{N,<:Nothing},Axs<:NTuple{N,<:AbstractVector},DG,
                    EQ<:NTuple{N,Int},PR<:NTuple{N,<:AbstractVector},KP,
                    KBC<:NTuple{N,Tuple{Symbol,Symbol}},DO,FD,SD,SG}
                    
    # specialized dispatch for N-dimensional higher-order kernel
    
    # Compute i_float once per dimension
    i_floats = ntuple(d -> (x[d] - itp.knots[d][1]) / itp.h[d] + one(T), N)
    
    # Find knot indices for each dimension
    pos_ids = ntuple(d -> clamp(floor(Int, i_floats[d]), 1, length(itp.knots[d]) - 1), N)
    
    is_boundary = ntuple(d -> is_boundary_stencil(pos_ids[d], size(itp.coefs, d), itp.eqs[d]), N)

    if any(is_boundary)

        # Compute normalized left distances - recompute from actual knot positions
        weights = ntuple(d -> (x[d] - itp.knots[d][pos_ids[d]]) / itp.h[d], N)
        
        # Build up: for each "slice" in remaining dimensions, 
        # accumulate the interpolated result
        result = zero(T)
        
        # Iterate over all 2^(N-1) combinations of the last N-1 dimensions
        @inbounds for corner in 0:(2^(N-1) - 1)
            # Build indices for dimensions 2:N
            tail_indices = ntuple(d -> (corner >> (d-1)) & 1 == 0 ? pos_ids[d+1] : pos_ids[d+1]+1, N-1)
            
            # Get the two values along dimension 1
            idx0 = (pos_ids[1], tail_indices...)
            idx1 = (pos_ids[1]+1, tail_indices...)
            
            # Interpolate along dimension 1
            interp_val = (one(T) - weights[1]) * itp.coefs[idx0...] + weights[1] * itp.coefs[idx1...]
            
            # Weight by all the other dimensions
            tail_weight = prod(ntuple(d -> (corner >> (d-1)) & 1 == 0 ? (one(T) - weights[d+1]) : weights[d+1], N-1))
            
            result += tail_weight * interp_val
        end
        
        return @inbounds @fastmath result * prod((-one(T)/itp.h[d])^DO[d] for d in 1:N)

    else

        # Compute normalized left distances - recompute from actual knot positions
        diff_left = ntuple(d -> (x[d] - itp.knots[d][pos_ids[d]]) / itp.h[d], N)
        diff_right = ntuple(d -> one(T) - diff_left[d], N)

        idx_lower = ntuple(d -> clamp(floor(Int, diff_right[d] * (length(itp.pre_range[d]) - one(Int64))) + one(Int64),
                                    one(Int64), length(itp.pre_range[d]) - one(Int64)), N)
        
        idx_upper = ntuple(d -> idx_lower[d] + 1, N)
        t = ntuple(d -> (diff_right[d] - itp.pre_range[d][idx_lower[d]]) / 
                        (itp.pre_range[d][idx_upper[d]] - itp.pre_range[d][idx_lower[d]]), N)

        result = zero(T)
        
        @inbounds for offsets in Iterators.product(ntuple(d -> -(itp.eqs[d]-1):itp.eqs[d], N)...)
            idxs = ntuple(d -> pos_ids[d] + offsets[d], N)
            coef = itp.coefs[idxs...]

            kernel_val = one(T)
            @inbounds for d in 1:N
                k_lower = itp.kernel_pre[d][idx_lower[d], offsets[d]+itp.eqs[d]]
                k_upper = itp.kernel_pre[d][idx_upper[d], offsets[d]+itp.eqs[d]]
                kernel_val *= (one(T) - t[d]) * k_lower + t[d] * k_upper
            end
            
            result += coef * kernel_val
        end
        
        return  @inbounds @fastmath result * prod((-one(T)/itp.h[d])^DO[d] for d in 1:N)
    end
end