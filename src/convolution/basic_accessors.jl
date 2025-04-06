"""
    getknots(itp::ConvolutionInterpolation)
    getknots(itp::FastConvolutionInterpolation)

Get the knot points (grid coordinates) of an interpolation object.

# Arguments
- `itp`: The interpolation object

# Returns
- A tuple of vectors containing the knot points for each dimension
"""
getknots(itp::ConvolutionInterpolation) = itp.knots
getknots(itp::FastConvolutionInterpolation) = itp.knots

"""
    Base.axes(itp::ConvolutionInterpolation)
    Base.axes(itp::FastConvolutionInterpolation)

Get the axes of the coefficient array in an interpolation object.

# Arguments
- `itp`: The interpolation object

# Returns
- The axes of the coefficient array
"""
Base.axes(itp::ConvolutionInterpolation) = axes(itp.coefs)
Base.axes(itp::FastConvolutionInterpolation) = axes(itp.coefs)

"""
    Base.size(itp::ConvolutionInterpolation)
    Base.size(itp::FastConvolutionInterpolation)

Get the size of the coefficient array in an interpolation object.

# Arguments
- `itp`: The interpolation object

# Returns
- A tuple containing the size of the coefficient array in each dimension
"""
Base.size(itp::ConvolutionInterpolation) = size(itp.coefs)
Base.size(itp::FastConvolutionInterpolation) = size(itp.coefs)

"""
    lbounds(itp::ConvolutionInterpolation)
    lbounds(itp::FastConvolutionInterpolation)

Get the lower bounds (first knot point in each dimension) of an interpolation object.

# Arguments
- `itp`: The interpolation object

# Returns
- A tuple containing the first knot point value in each dimension
"""
lbounds(itp::ConvolutionInterpolation) = first.(itp.knots)
lbounds(itp::FastConvolutionInterpolation) = first.(itp.knots)

"""
    ubounds(itp::ConvolutionInterpolation)
    ubounds(itp::FastConvolutionInterpolation)

Get the upper bounds (last knot point in each dimension) of an interpolation object.

# Arguments
- `itp`: The interpolation object

# Returns
- A tuple containing the last knot point value in each dimension
"""
ubounds(itp::ConvolutionInterpolation) = last.(itp.knots)
ubounds(itp::FastConvolutionInterpolation) = last.(itp.knots)

"""
    itpflag(::Type{<:ConvolutionInterpolation{T,N,TCoefs,IT}}) where {T,N,TCoefs,IT}
    itpflag(itp::ConvolutionInterpolation)
    itpflag(::Type{<:FastConvolutionInterpolation{T,N,TCoefs,IT}}) where {T,N,TCoefs,IT}
    itpflag(itp::FastConvolutionInterpolation)

Get the interpolation method type flag(s) for an interpolation object.

# Arguments
- `itp`: The interpolation object or its type

# Returns
- The interpolation method identifier (ConvolutionMethod or tuple of them)

This function is used for method dispatch in the interpolation system.
"""
itpflag(::Type{<:ConvolutionInterpolation{T,N,TCoefs,IT}}) where {T,N,TCoefs,IT} = IT()
itpflag(itp::ConvolutionInterpolation) = itp.it
itpflag(::Type{<:FastConvolutionInterpolation{T,N,TCoefs,IT}}) where {T,N,TCoefs,IT} = IT()
itpflag(itp::FastConvolutionInterpolation) = itp.it

"""
    coefficients(itp::ConvolutionInterpolation)
    coefficients(itp::FastConvolutionInterpolation)

Get the coefficient array of an interpolation object.

# Arguments
- `itp`: The interpolation object

# Returns
- The coefficient array containing interpolation data with boundary extensions
"""
coefficients(itp::ConvolutionInterpolation) = itp.coefs
coefficients(itp::FastConvolutionInterpolation) = itp.coefs

"""
    Base.length(itp::ConvolutionInterpolation)
    Base.length(itp::FastConvolutionInterpolation)

Get the total number of elements in the coefficient array of an interpolation object.

# Arguments
- `itp`: The interpolation object

# Returns
- The total number of elements in the coefficient array
"""
Base.length(itp::ConvolutionInterpolation) = length(itp.coefs)
Base.length(itp::FastConvolutionInterpolation) = length(itp.coefs)

"""
    Base.iterate(itp::ConvolutionInterpolation, state=1)
    Base.iterate(itp::FastConvolutionInterpolation, state=1)

Iterate through the coefficient values of an interpolation object.

# Arguments
- `itp`: The interpolation object
- `state=1`: The current iteration state

# Returns
- A tuple of (value, next_state) or nothing when iteration is complete

This method allows interpolation objects to be used in for loops and other iteration contexts.
"""
Base.iterate(itp::ConvolutionInterpolation, state=1) = state > length(itp) ? nothing : (itp[state], state+1)
Base.iterate(itp::FastConvolutionInterpolation, state=1) = state > length(itp) ? nothing : (itp[state], state+1)

"""
    lbound(ax::AbstractRange, itp::ConvolutionMethod)

Get the lower bound of an axis for a given interpolation method.

# Arguments
- `ax::AbstractRange`: The axis range
- `itp::ConvolutionMethod`: The interpolation method

# Returns
- The lower bound of the axis for the specified interpolation method

This function is used internally for determining valid interpolation bounds.
"""
lbound(ax::AbstractRange, itp::ConvolutionMethod) = lbound(ax, itpflag(itp))