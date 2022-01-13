############################################################################################
"""
$(TYPEDEF)
Contains maximum memory for latent and observed data. Relevant for number of
propagation steps from initial particle distribution and for first time particles
are weighted with data point.

# Fields
$(TYPEDFIELDS)
"""
struct ParticleFilterMemory
    "Latent variable memory. Number of times that initial particle distribution is used."
    latent::Int64
    "Observed data memory. First particle weighting taken at this point."
    data::Int64
    function ParticleFilterMemory(latent::Int64, data::Int64)
        ArgCheck.@argcheck latent >= 0 "Latent memory cannot be negative."
        ArgCheck.@argcheck data >= 0 "Data memory cannot be negative."
        return new(latent, data)
    end
end

function max(memory::ParticleFilterMemory)
    return max(memory.latent, memory.data)
end

############################################################################################
"""
$(SIGNATURES)
Propagate forward a single trajectory given ParticleKernel.

# Examples
```julia
```

"""
function propagate(
    _rng, kernel::ParticleKernel, memory::ParticleFilterMemory, reference::AbstractArray{P}
) where {P}
    trajectory = Vector{P}(undef, size(reference, 1))
    Ninitial = max(1, memory.latent)
    ## Initialize particle
    for iter in 1:Ninitial
        trajectory[iter] = initial(_rng, kernel)
    end
    ## Propoagate particle forward
    for iter in (Ninitial + 1):size(trajectory, 1)
        trajectory[iter] = transition(_rng, kernel, trajectory, iter)
    end
    return trajectory
end

############################################################################################
"""
$(SIGNATURES)
Check if function is exectuable.

# Examples
```julia
```

"""
function _try(f::Function, input...)
    try
        f(input...)
    catch err
        return err
    end
end

"""
$(SIGNATURES)
Return first iteration at which function is executable.

# Examples
```julia
```

"""
function _checkmemory(f::Function; maxiter=1000)
    for iter in Base.OneTo(maxiter)
        val = _try(f, iter)
        if !isa(val, Exception)
            return iter
        end
    end
    return "Memory for Objective exceeds 1000 - check your dynamics(objective) implementation."
end

"""
$(SIGNATURES)
Guess memory of ParticleKernel.

# Examples
```julia
```

"""
function _guessmemory(
    _rng::Random.AbstractRNG, kernel::ParticleKernel, reference::AbstractArray{T}
) where {T}
    ## Initialize state and evidence transition distribution as a function of the current iteration
    guess_state(iter) = transition(_rng, kernel, reference, iter)
    #!NOTE: We dont need to use more general ℓevidence here as evidence distribution independent of Particle type
    guess_evidence(iter) = kernel.evidence(reference, iter)
    ## Compute first iteration at which kernel works
    Mparticle = _checkmemory(guess_state)
    Mevidence = _checkmemory(guess_evidence)
    ## Check if dependency structure could be determined
    ArgCheck.@argcheck isa(Mparticle, Integer) "Could not determine particle dependency - check kernel.transition in dynamics, or consider writing an issue in BaytesFilters."
    ArgCheck.@argcheck isa(Mevidence, Integer) "Could not determine data dependency - check kernel.observation in dynamics, or consider writing an issue in BaytesFilters."
    ## Return ParticleFilterMemory - subtract 1 from memory as iterations start with 1, which would mean memoryless transition.
    return ParticleFilterMemory(Mparticle - 1, Mevidence - 1)
end

################################################################################
#export
export ParticleFilterMemory
