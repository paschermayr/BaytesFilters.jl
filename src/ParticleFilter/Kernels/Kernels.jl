############################################################################################
#Define general functions that need to be dispatched for new particle kernels

"""
$(SIGNATURES)
Initiate particle with given initial distribution.

# Examples
```julia
```

"""
function initial(_rng::Random.AbstractRNG, kernel::ParticleKernel) end

"""
$(SIGNATURES)
Propagate particle forward given current particle.

# Examples
```julia
```

"""
function transition(
    _rng::Random.AbstractRNG, kernel::ParticleKernel, val::AbstractMatrix{P}, iter::Integer
) where {P} end

"""
$(SIGNATURES)
Calculate log transtion probability from particle given particle history.

# Examples
```julia
```

"""
function ℓtransition(
    _rng::Random.AbstractRNG,
    kernel::ParticleKernel,
    valₜ::Union{P,AbstractVector{P}},
    val::AbstractMatrix{P},
    iter::Integer,
) where {P} end

############################################################################################
#Include
include("markov.jl")
include("semimarkov.jl")

############################################################################################
"""
$(SIGNATURES)
Inplace initiate particle given kernel.

# Examples
```julia
```

"""
function initial!(
    _rng::Random.AbstractRNG, kernel::ParticleKernel, val::AbstractMatrix{P}, iter::Integer
) where {P}
    @inbounds for idx in Base.OneTo(size(val, 1))
        val[idx, iter] = initial(_rng, kernel)
    end
    return nothing
end

"""
$(SIGNATURES)
Inplace propagate particle forward given current particle and kernel.

# Examples
```julia
```

"""
function transition!(
    _rng::Random.AbstractRNG, kernel::ParticleKernel, val::AbstractMatrix{P}, iter::Integer
) where {P}
    #!NOTE: Transition to s_t ~ s_1:t-1, trajectories:t-1
    @inbounds for idx in Base.OneTo(size(val, 1))
        val[idx, iter] = transition(_rng, kernel, view(val, idx, :), iter)
    end
    return nothing
end

############################################################################################
"""
$(SIGNATURES)
Inplace calculate log transtion probability from current particle given trajectory.

# Examples
```julia
```

"""
function ℓtransition!(
    ℓcontainer::Vector{F},
    valₜ::Union{P,AbstractArray{P}},
    kernel::ParticleKernel,
    val::AbstractMatrix{P},
    iter::Integer,
) where {F<:AbstractFloat,P}
    @inbounds for idx in Base.OneTo(size(val, 1))
        ℓcontainer[idx] = ℓtransition(valₜ, kernel, view(val, idx, :), iter)
    end
    return ℓcontainer
end

############################################################################################

"""
$(SIGNATURES)
Evaluate data given particle trajectory.

# Examples
```julia
```

"""
function ℓevidence(
    dataₜ::D, kernel::ParticleKernel, val::AbstractVector{P}, iter::Integer
) where {D,P}
    return logpdf(kernel.evidence(val, iter), dataₜ)
end

"""
$(SIGNATURES)
Inplace logpdf evaluation of data given current particle trajectory.

# Examples
```julia
```

"""
function ℓevidence!(
    ℓcontainer::Vector{F},
    dataₜ::D,
    kernel::ParticleKernel,
    val::AbstractMatrix{P},
    iter::Integer,
) where {F<:AbstractFloat,D,P}
    @inbounds for idx in Base.OneTo(size(val, 1))
        #!NOTE: Need to evaluate e_t ~ s_1:t, e_1:t-1
        ℓcontainer[idx] = ℓevidence(dataₜ, kernel, view(val, idx, :), iter)
    end
    return ℓcontainer
end

############################################################################################
#export
export initial,
    transition, ℓtransition, initial!, transition!, ℓtransition!, ℓevidence, ℓevidence!
