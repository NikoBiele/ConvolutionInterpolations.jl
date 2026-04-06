"""
    autocorrelation(signal::NTuple{NS,T}) where {NS,T}

Computes the autocorrelation function of a signal.

# Arguments
- `signal::Vector{T}`: Input signal vector where T<:Number

# Returns
- Vector containing the autocorrelation values

# Details
This function:
1. Centers the signal by subtracting the mean
2. Computes the signal variance (with minimum threshold to prevent division by zero)
3. Calculates the normalized autocorrelation function for lags 0 to n-1, where n is the signal length

The autocorrelation values provide information about signal periodicity and structure, which is used for determining appropriate boundary conditions.
"""
function autocorrelation(signal::NTuple{NS,T}) where {NS,T}
    s = sum(signal) / NS
    signal_centered = ntuple(i -> signal[i] - s, NS)
    variance = max(T(1e-6), sum(x -> x^2, signal_centered) / NS)
    return ntuple(k -> sum(i -> signal_centered[i] * signal_centered[i+k], 1:NS-k) / (NS * variance), 6)
end