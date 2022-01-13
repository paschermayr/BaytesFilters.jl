#=
Some notes to myself because I always forget it:
-> so want to estimate normalizing marginal likelihood term in P(s_1:t | e_1:t) via mean(w_1:t)
-> calculate w_1:t iteratively via w_t-k:t
-> in SIS we calculate normalizing W_t ∝ W_t-1 * w_t-k:t
-> in PF we always calculate normalizing W_t ∝ w_t-k:t, because W_t-1 is constant from resampling step.
-> in adaptive Res. PF, we still need normalizing W_t ∝ W_t-1 * w_t-k:t. In case resampled, W_t is constant, if not need SIS calculation.
-> Sometimes people define unnormalized w_t via w_t-1 * g(..) -> but this refers to w_1:t for each step! Not what we need in PF, as we calculate normalizing term iteratively.
=#
############################################################################################
"""
$(TYPEDEF)

Container for weights and normalized particles weights at current time step t.

# Fields
$(TYPEDFIELDS)
"""
struct ParticleWeights
    "Used for log likelihood calculation."
    ℓweights::Vector{Float64}
    "exp(ℓweightsₙ) used for resampling particles and sampling trajectories ~ kept in log space for numerical stability."
    ℓweightsₙ::Vector{Float64}
    "A buffer vector with same size as ℓweights and ℓweightsₙ."
    buffer::Vector{Float64}
    function ParticleWeights(Nparticles::I) where {I<:Integer}
        ArgCheck.@argcheck Nparticles > 0 "Number of particles need to be positive."
        return new(
            fill(log(1.0 / Nparticles), Nparticles),
            fill(log(1.0 / Nparticles), Nparticles),
            fill(1.0 / Nparticles, Nparticles),
        )
    end
end

############################################################################################
"""
$(SIGNATURES)
Returns log weights and normalized log weights at time > 1.

# Examples
```julia
```

"""
function (weights::ParticleWeights)(ℓevidenceₜ::AbstractVector{T}) where {T<:Real}
    weights.ℓweights .= ℓevidenceₜ #Incremental weight
    normalize!(weights)
    return nothing
end

############################################################################################
"""
$(SIGNATURES)
Set weights back to equal.

# Examples
```julia
```

"""
function update!(weights::ParticleWeights)
    weights.ℓweights .= weights.ℓweightsₙ .= log(1.0 / length(weights.buffer))
    return nothing
end

############################################################################################
"""
$(SIGNATURES)
Inplace-Normalize Particle weights, accounting for ℓweightsₙ at previous iteration.

# Examples
```julia
```

"""
function normalize!(weights::ParticleWeights)
    # Wₜ ∝ Wₜ₋₁×wₜ #!NOTE: Relevant if no resampling step performed
    weights.ℓweightsₙ .= (weights.ℓweightsₙ .+ weights.ℓweights)
    weights.ℓweightsₙ .-= logsumexp(weights.ℓweightsₙ)
    return nothing
end

############################################################################################
"""
$(SIGNATURES)
Compute effectice sample size of particle filter via normalized log weights.

# Examples
```julia
```

"""
function computeESS(weightsₙ::Vector{T}) where {T<:AbstractFloat}
    return 1.0 / sum(abs2, weightsₙ)
end
function computeESS(weights::ParticleWeights)
    weights.buffer .= exp.(weights.ℓweightsₙ)
    return 1.0 / sum(abs2, weights.buffer)
end

############################################################################################
"""
$(SIGNATURES)
Draw proposal trajectory index.

# Examples
```julia
```

"""
function draw!(_rng::Random.AbstractRNG, weights::ParticleWeights)
    ## Draw new path
    weights.buffer .= exp.(weights.ℓweightsₙ)
    path = randcat(_rng, weights.buffer)
    ## Return trajectory
    return path
end

############################################################################################
# Types and structs - Reweighting Methods
struct Bootstrap <: ParticleWeighting end

"""
$(SIGNATURES)
Weight function that is dispatched on ParticleWeighting types.

# Examples
```julia
```

"""
function weight!() end

############################################################################################
# Bootstrap Method
function weight!(
    method::Bootstrap,
    weights::ParticleWeights,
    dataₜ::D,
    kernel::ParticleKernel,
    val::AbstractMatrix{P},
    iter::Integer,
) where {D,P}
    ##Calculate weights
    ℓevidence!(weights.ℓweights, dataₜ, kernel, val, iter)
    ##Normalize weights
    normalize!(weights)
    return nothing
end

############################################################################################
#export
export ParticleWeights, draw!, normalize!, computeESS, Bootstrap, weight!
