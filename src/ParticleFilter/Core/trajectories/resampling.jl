#= DISCLAIMER!:
	Original functions for resampling taken from: https://github.com/TuringLang/AdvancedPS.jl/blob/master/src/resampling.jl
=#

############################################################################################
# Types and structs - Resampling Methods
struct Simple <: ParticleResampling end
struct Systematic <: ParticleResampling end
struct Stratified <: ParticleResampling end
struct Residual <: ParticleResampling end

############################################################################################
"""
$(SIGNATURES)
Resample particles, dispatched on ParticleResampling subtypes.

# Examples
```julia
```

"""
function resample! end
function resample!(
    _rng::Random.AbstractRNG,
    type::S,
    container::AbstractMatrix{<:Integer},
    iter::Integer,
    weights::Vector{<:Real},
    n::Integer=length(weights),
) where {S<:ParticleResampling}
    return resample!(_rng, type, view(container, :, iter), iter, weights, n)
end

############################################################################################
function resample!(
    _rng::Random.AbstractRNG,
    type::Systematic,
    container::AbstractVector{<:Integer},
    iter::Integer,
    weights::Vector{<:Real},
    n::Integer=length(weights),
)
    # check input
    m = length(weights)
    m > 0 || error("weight vector is empty")
    # pre-calculations
    @inbounds v = n * weights[1]
    u = oftype(v, rand(_rng))
    sample = 1
    @inbounds for i in 1:n
        # as long as we have not found the next sample
        while v < u
            # increase and check the sample
            sample += 1
            sample > m &&
                error("sample could not be selected (are the weights normalized?)")
            # update the cumulative sum of weights (scaled by `n`)
            v += n * weights[sample]
        end
        # save the next sample
        container[i] = sample
        # update `u`
        u += one(u)
    end
    return nothing
end

############################################################################################
function resample!(
    _rng::Random.AbstractRNG,
    type::Stratified,
    container::AbstractVector{<:Integer},
    iter::Integer,
    weights::Vector{<:Real},
    n::Integer=length(weights),
)
    # check input
    m = length(weights)
    m > 0 || error("weight vector is empty")
    # pre-calculations
    @inbounds v = n * weights[1]
    sample = 1
    @inbounds for i in 1:n
        # sample next `u` (scaled by `n`)
        u = oftype(v, i - 1 + rand(_rng))
        # as long as we have not found the next sample
        while v < u
            # increase and check the sample
            sample += 1
            sample > m &&
                error("sample could not be selected (are the weights normalized?)")
            # update the cumulative sum of weights (scaled by `n`)
            v += n * weights[sample]
        end
        # save the next sample
        container[i] = sample
    end
    return nothing
end

############################################################################################
function resample!(
    _rng::Random.AbstractRNG,
    type::Residual,
    container::AbstractVector{<:Integer},
    iter::Integer,
    weights::Vector{<:Real},
    n::Integer=length(weights),
)
    # deterministic assignment
    residuals = similar(weights)
    i = 1
    @inbounds for j in 1:length(weights)
        x = n * weights[j]
        floor_x = floor(Int, x)
        for k in 1:floor_x
            container[i] = j
            i += 1
        end
        residuals[j] = x - floor_x
    end
    # sampling from residuals
    if i <= n
        residuals ./= sum(residuals)
        rand!(_rng, Distributions.Categorical(residuals), view(container, 1:n))
    end
    return nothing
end

############################################################################################
function resample!(
    _rng::Random.AbstractRNG,
    type::Simple,
    container::AbstractVector{<:Integer},
    iter::Integer,
    weights::Vector{<:Real},
    n::Integer=length(weights),
)
    rand!(_rng, Distributions.Categorical(weights), container)
    return nothing
end

############################################################################################
#= DISCLAIMER:
	Original function taken from: https://github.com/TuringLang/AdvancedPS.jl/blob/master/src/resampling.jl
=#
"""
$(SIGNATURES)
More stable, faster version of rand(Categorical) if weights sum up to 1.

# Examples
```julia
```

"""
function randcat(_rng::Random.AbstractRNG, p::AbstractVector{<:Real})
    T = eltype(p)
    r = rand(_rng, T)
    cp = p[1]
    s = 1
    n = length(p)
    while cp <= r && s < n
        @inbounds cp += p[s += 1]
    end
    return s
end

############################################################################################
#Export
export resample!, randcat, Simple, Systematic, Stratified, Residual
