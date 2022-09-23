############################################################################################
"""
$(TYPEDEF)

Contains Particle container for propagation.

# Fields
$(TYPEDFIELDS)
"""
mutable struct Particles{R,P<:ParticleKernel,B,I<:Integer} <: AbstractParticles
    "Particle trajectories, for a discussion about possible shapes for the trajectories."
    #=
    Field that hold Nparticles Ndata times, CRITERIA:
    1) Must be able to sort values efficiently for a Column Major ~ Base Vector{Vector{T}} not performant
    2) must be able to be extended in Ndata dimension if new data arrives
    -> ElasticMatrix can do both
    + Can easily return a single particle trajectory
    -> Need to use view(elasticmatrix,...)
    + Can resize in Nparticles dimension for tuning
    -> not possible at the moment
    3) Interesting conversations:
    https://discourse.julialang.org/t/preallocating-vector-of-vectors/69304/10."
    https://discourse.julialang.org/t/elasticarrays-arraysofarrays-for-vector-of-arrays/70426/2
    -> a VectorOfVectors of SimilarArrays that is push!-able sounds interesting, but how
    would we depict multivariate trajectories in this case? The Matrix Variant would just store a vector in each field and thats it.
    =#
    val::ElasticMatrix{B,Vector{B}}
    "Saved ancestors of resampling step in pf"
    ancestor::ElasticMatrix{I,Vector{I}}
    "Distributions to propagate particle forward."
    kernel::P
    "Particle weights."
    weights::BaytesCore.ParameterWeights
    "Necessary buffer values for resampling particles."
    buffer::ParticleBuffer{B,I,R}
    "Log likelihood estimate."
    ℓobjective::Accumulator
    function Particles(
        reference::AbstractArray{T},
        prediction::Vector{R},
        kernel::P,
        ancestortype::Type{I},
        Nparticles::Integer,
        Ndata::Integer
    ) where {T,R,P<:ParticleKernel,I<:Integer}
        ## Create val
        val = ElasticMatrix{T}(undef, Nparticles, Ndata)
        ## Create ancestors - switch Nparticles, Ndata for easier access in resampling step
        ancestor = ElasticMatrix{ancestortype}(undef, Nparticles, Ndata)
        ## Create initial weights
        weights = BaytesCore.ParameterWeights(Nparticles)
        ## Create buffer
        buffer = ParticleBuffer(prediction, reference, Nparticles, Ndata, ancestortype)
        ## Return Particles
        return new{R,P,T,I}(val, ancestor, kernel, weights, buffer, Accumulator())
    end
end

function initial!(_rng::Random.AbstractRNG, particles::Particles, tune::ParticleFilterTune)
    return initial!(_rng, particles.kernel, particles.val, tune.iter.current)
end
function transition!(
    _rng::Random.AbstractRNG, particles::Particles, tune::ParticleFilterTune
)
    return transition!(_rng, particles.kernel, particles.val, tune.iter.current)
end

function weight!(dataₜ::D, particles::Particles, tune::ParticleFilterTune) where {D}
    return weight!(
        tune.weighting,
        particles.weights,
        dataₜ,
        particles.kernel,
        particles.val,
        tune.iter.current,
    )
end

#!NOTE: Here we can directly input tune.iter.current-1 for ancestors
function ancestors!(
    _rng::Random.AbstractRNG,
    particles::Particles,
    tune::ParticleFilterTune,
    ancestor=particles.ancestor,
)
    return ancestors!(
        _rng,
        tune.referencing,
        tune.resampling,
        ancestor,
        tune.iter.current - 1,
        tune.chains.Nchains,
        particles.weights.buffer,
    )
end
#!NOTE: This step is AFTER resample!() [ancestors!/get_reference!] during particle propagation hence tune.iter.current is used
function set_reference!(
    particles::Particles, tune::ParticleFilterTune, reference::AbstractArray{P}
) where {P}
    return set_reference!(tune.referencing, particles.val, reference, tune.iter.current)
end

function shuffle_forward!(particles::Particles, tune::ParticleFilterTune)
    return shuffle_forward!(
        particles.val,
        particles.buffer.parameter.val,
        particles.ancestor,
        particles.buffer.parameter.index,
        max(tune.memory), #tune.memory.latent,
        tune.iter.current - 1,
    )
end
function shuffle_backward!(particles::Particles, tune::ParticleFilterTune)
    return shuffle_backward!(
        particles.val, particles.buffer.parameter.val, particles.ancestor, particles.buffer.parameter.index
    )
end

############################################################################################
"""
$(SIGNATURES)
Resize particles struct with new `Nparticles` and `Ndata` size.

# Examples
```julia
```

"""
function resize!(
    particles::Particles,
    kernel::P,
    reference::AbstractArray{T},
    Nparticles::Integer,
    Ndata::Integer,
) where {T,P<:ParticleKernel}
    ## Change val
    particles.val = ElasticMatrix{T}(undef, Nparticles, Ndata)
    ## Change ancestor size
    F = eltype(particles.ancestor)
    particles.ancestor = ElasticMatrix{F}(undef, Nparticles, Ndata)
    ## Update buffer
    update!(particles.buffer, Nparticles, Ndata)
    ## Compute new initial weights
    particles.weights = BaytesCore.ParameterWeights(Nparticles)
    ##
    return nothing
end

############################################################################################
"""
$(SIGNATURES)
Update particles struct with new number of particles. Note that kernel and log likelihood are adjusted in another step.

# Examples
```julia
```

"""
function update!(
    particles::Particles,
    ParticleUpdate::BaytesCore.UpdateFalse,
    DataUpdate::BaytesCore.UpdateFalse,
    kernel::P,
    reference::AbstractArray{T},
    Nparticles::Integer,
    Ndata::Integer,
) where {T,P<:ParticleKernel}
    ## Set loglikelihood counter back to 0
    init!(particles.ℓobjective)
    ## Update kernel
    particles.kernel = kernel
    ## Return
    return nothing
end
function update!(
    particles::Particles,
    ParticleUpdate::BaytesCore.UpdateTrue,
    DataUpdate::D,
    kernel::P,
    reference::AbstractArray{T},
    Nparticles::Integer,
    Ndata::Integer,
) where {D<:BaytesCore.UpdateBool,T,P<:ParticleKernel}
    ## Set loglikelihood counter back to 0
    init!(particles.ℓobjective)
    ## Update kernel
    particles.kernel = kernel
    ## Resize Particles with new Nparticles and Ndata
    resize!(particles, kernel, reference, Nparticles, Ndata)
    ## Return
    return nothing
end
function update!(
    particles::Particles,
    ParticleUpdate::BaytesCore.UpdateFalse,
    DataUpdate::BaytesCore.UpdateTrue,
    kernel::P,
    reference::AbstractArray{T},
    Nparticles::Integer,
    Ndata::Integer,
) where {T,P<:ParticleKernel}
    ## Set loglikelihood counter back to 0
    init!(particles.ℓobjective)
    ## Update kernel
    particles.kernel = kernel
    ## Resize val and ancestor size
    resize!(particles.val, Nparticles, Ndata)
    resize!(particles.ancestor, Nparticles, Ndata)
    ## Resize buffer
    update!(particles.buffer, Nparticles, Ndata)
    ##
    return nothing
end

############################################################################################
"""
$(SIGNATURES)
Resample particle ancestors and shuffle current particles dependency.

# Examples
```julia
```

"""
function resample!(
    _rng::Random.AbstractRNG,
    particles::Particles,
    tune::ParticleFilterTune,
    reference::AbstractArray{P},
) where {P}
    #!NOTE: Resampling step for t-1 done at iteration t, before propagation starts. So ancestors_t-1 for particle_t are stored
    if BaytesCore.issmaller(
        BaytesCore.computeESS(particles.weights), tune.chains.Nchains * tune.chains.threshold
    )
        ## Resample ancestors
        ancestors!(_rng, particles, tune)
        ## Compute new reference ancestor
        get_reference!(_rng, particles, tune, reference)
        ## Set resampled == true in corresponding iteration -> happens at start of iteration, so iter-1 for resampled particles
        particles.buffer.resampled[tune.iter.current - 1] = true
        ## Equal weight normalized weights for next iteration memory
        Base.fill!(particles.weights.ℓweightsₙ, log(1.0 / tune.chains.Nchains))
        ## Shuffle particles according to resampled indices
        #!NOTE: We shuffle particles at iter-1 since we are doing this before transition at iter
        shuffle_forward!(particles, tune)
    else
        ## Set resampled to false
        particles.buffer.resampled[tune.iter.current - 1] = false
        ## Assign default order to ancestors
        @inbounds for Nrow in Base.OneTo(length(particles.buffer.parameter.val)) #size(particles.ancestor[tune.iter.current-1], 1) )
            particles.ancestor[Nrow, tune.iter.current - 1] = Nrow
        end
    end
    return nothing
end

"""
$(SIGNATURES)
Initialize particles.

# Examples
```julia
```

"""
function initial!(
    _rng::AbstractRNG,
    particles::Particles,
    tune::ParticleFilterTune,
    reference::AbstractArray{P},
    objective::Objective,
) where {P}
    ## Start iterating over initial distributions of particles ~ depends on memory of particles, usually 1
    for t in 1:tune.memory.initial #max(1, tune.memory.latent)
        ## Sample from initial distribution
        initial!(_rng, particles, tune)
        ## Update Last particle value based on Particle Filter settings (Marginal/Conditiona/Ancestral)
        set_reference!(particles, tune, reference)
        ## Calculate particle weights and log likelihood
        @inbounds if t > tune.memory.data #maxmemory
            weight!(BaytesCore.grab(objective.data, t, tune.config.data), particles, tune)
            ℓobjectiveₜ = logmeanexp(particles.weights.ℓweights)
            particles.buffer.ℓobjectiveᵥ[t] = ℓobjectiveₜ
            update!(particles.ℓobjective, ℓobjectiveₜ)
        end
        #!NOTE: No Resample step at init! for particle ancestors ~ first resampling should happen in propagate!() before particle propagation starts
        ## Update current iteration
        update!(tune.iter)
        ## Assign default order to ancestors
        @inbounds for Nrow in Base.OneTo(length(particles.buffer.parameter.val))
            particles.ancestor[Nrow, t] = Nrow
        end
    end
    return nothing
end

############################################################################################
"""
$(SIGNATURES)
Propagate particles forward.

# Examples
```julia
```

"""
function propagate!(
    _rng::AbstractRNG,
    particles::Particles,
    tune::ParticleFilterTune,
    reference::AbstractArray{P},
    objective::Objective,
) where {P}
    #!NOTE: Ndata a bit more flexible than size(reference, 1), as one could only propagate a few iterations even if more samples are known.
    for t in (tune.iter.current):(tune.chains.Ndata)
        ## Resample particle ancestors if resampling criterion fullfiled
        resample!(_rng, particles, tune, reference)
        ## Transition particles
        transition!(_rng, particles, tune)
        ## Update Last particle value based on Particle Filter settings (Marginal/Conditiona/Ancestral)
        set_reference!(particles, tune, reference)
        ## Calculate particle weights and log likelihood
        #!NOTE: cannot do this at the same time if particles are resampled adaptively, as normalized weights will change if not resampled
        @inbounds if t > tune.memory.data #maxmemory
            weight!(
                BaytesCore.grab(objective.data, tune.iter.current, tune.config.data),
                particles,
                tune,
            )
            ℓobjectiveₜ = logmeanexp(particles.weights.ℓweights)
            particles.buffer.ℓobjectiveᵥ[t] = ℓobjectiveₜ
            update!(particles.ℓobjective, ℓobjectiveₜ)
        end
        ## Update current iteration to Ndata+1 for propagation step
        update!(tune.iter)
    end
    ## Assign default order to ancestors for final index
    @inbounds for Nrow in Base.OneTo(length(particles.buffer.parameter.val))
        particles.ancestor[Nrow, tune.iter.current - 1] = Nrow
    end
    #!NOTE: NO more resampling step at final iteration, as we (1) resample ancestor (2) propagate particles, not (1) propagate particles (2) resample particles
    return nothing
end

############################################################################################
"""
$(SIGNATURES)
Predict new latent and observed data.

# Examples
```julia
```

"""

function predict(
    _rng::Random.AbstractRNG,
    trajectory::AbstractArray{P},
    kernel::ParticleKernel,
    reference::AbstractArray{P},
    iter::Integer,
) where {P}
    ## Predict new state
    stateᵗ⁺¹ = transition(_rng, kernel, trajectory, iter)
    ## Add prediction to proposal
    push!(trajectory, stateᵗ⁺¹)
    ## Predict new data
    dataᵗ⁺¹ = rand(_rng, kernel.evidence(trajectory, iter))
    return (eltype(reference)(stateᵗ⁺¹), dataᵗ⁺¹)
end
function predict(
    _rng,
    particles::Particles,
    tune::ParticleFilterTune,
    reference::AbstractArray{P},
    path::Integer,
) where {P}
    return predict(
        _rng,
        assign!(particles.buffer.proposal, view(particles.val, path, :)),
        particles.kernel,
        reference,
        tune.iter.current,
    )
end

"""
$(SIGNATURES)
Assign 'trajectory' elements to 'buffer' up until index 'iter'.

# Examples
```julia
```

"""
function assign!(buffer::AbstractVector{P}, trajectory::AbstractVector{T}) where {P,T}
    ## Resize buffer
    resize!(buffer, length(trajectory))
    ## Add trajectory elements to buffer
    @inbounds @simd for iter in eachindex(trajectory)
        buffer[iter] = trajectory[iter]
    end
    return buffer
end

############################################################################################
#export
export Particles, resample!, initial!, propagate!, predict
