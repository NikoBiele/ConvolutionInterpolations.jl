"""
    Base.getindex(itp::ConvolutionInterpolation{T,N,TCoefs,IT,Axs,ConvolutionKernel{DG},DT}, 
                 I::Vararg{Integer,N}) where {T,N,TCoefs,IT,Axs,DT,DG}
    Base.getindex(itp::FastConvolutionInterpolation{T,N,TCoefs,IT,Axs,ConvolutionKernel{DG},DT}, 
                 I::Vararg{Integer,N}) where {T,N,TCoefs,IT,Axs,DT,DG}

Access coefficient values directly using integer indices.

# Arguments
- `itp`: The interpolation object
- `I::Vararg{Integer,N}`: The indices to access (one for each dimension)

# Returns
- The coefficient value at the specified indices

This method allows direct access to the underlying coefficient array using integer indices.
No interpolation is performed; this merely returns the exact coefficient value.
"""
function Base.getindex(itp::ConvolutionInterpolation{T,N,TCoefs,IT,Axs,ConvolutionKernel{DG},DT}, I::Vararg{Integer,N}) where {T,N,TCoefs,IT,Axs,DT,DG}
    return itp.coefs[I...]
end
function Base.getindex(itp::FastConvolutionInterpolation{T,N,TCoefs,IT,Axs,ConvolutionKernel{DG},DT}, I::Vararg{Integer,N}) where {T,N,TCoefs,IT,Axs,DT,DG}
    return itp.coefs[I...]
end

"""
    (itp::ConvolutionInterpolation{T,1,TCoefs,IT,Axs,ConvolutionKernel{DG},Val{1}})(i::Integer) where {T,TCoefs,IT,Axs,DG}
    (itp::FastConvolutionInterpolation{T,1,TCoefs,IT,Axs,ConvolutionKernel{DG},Val{1}})(i::Integer) where {T,TCoefs,IT,Axs,DG}

Access coefficient values directly using a single integer index for 1D interpolation.

# Arguments
- `itp`: The one-dimensional interpolation object
- `i::Integer`: The index to access

# Returns
- The coefficient value at the specified index

This method provides a convenient function-call syntax for accessing coefficient values
in one-dimensional interpolation objects. No interpolation is performed.
"""
function (itp::ConvolutionInterpolation{T,1,TCoefs,IT,Axs,ConvolutionKernel{DG},Val{1}})(i::Integer) where {T,TCoefs,IT,Axs,DG}
    return itp.coefs[i]
end
function (itp::FastConvolutionInterpolation{T,1,TCoefs,IT,Axs,ConvolutionKernel{DG},Val{1}})(i::Integer) where {T,TCoefs,IT,Axs,DG}
    return itp.coefs[i]
end


"""
    (itp::ConvolutionInterpolation{T,N,TCoefs,IT,Axs,ConvolutionKernel{DG},DT})(I::Vararg{Integer,N}) where {T,N,TCoefs,IT,Axs,DT,DG}
    (itp::FastConvolutionInterpolation{T,N,TCoefs,IT,Axs,ConvolutionKernel{DG},DT})(I::Vararg{Integer,N}) where {T,N,TCoefs,IT,Axs,DT,DG}

Access coefficient values directly using integer indices with function-call syntax.

# Arguments
- `itp`: The interpolation object
- `I::Vararg{Integer,N}`: The indices to access (one for each dimension)

# Returns
- The coefficient value at the specified indices

This method provides a convenient function-call syntax for accessing coefficient values
in multi-dimensional interpolation objects. No interpolation is performed.
"""
function (itp::ConvolutionInterpolation{T,N,TCoefs,IT,Axs,ConvolutionKernel{DG},DT})(I::Vararg{Integer,N}) where {T,N,TCoefs,IT,Axs,DT,DG}
    return itp.coefs[I...]
end

function (itp::FastConvolutionInterpolation{T,N,TCoefs,IT,Axs,ConvolutionKernel{DG},DT})(I::Vararg{Integer,N}) where {T,N,TCoefs,IT,Axs,DT,DG}
    return itp.coefs[I...]
end