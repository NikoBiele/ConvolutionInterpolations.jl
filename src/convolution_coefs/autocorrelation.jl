"""
    autocorrelation(signal::Vector{T}) where T<:Number

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
function autocorrelation(signal::Vector{T}) where T
    n = length(signal)
    signal_centered = signal .- sum(signal)/length(signal)
    variance = max(1e-6, sum(abs2, signal_centered) / n)
    acf = T[sum(signal_centered[1:n-k] .* signal_centered[k+1:end]) for k in 0:n-1] ./ (n * variance)
    return acf
end