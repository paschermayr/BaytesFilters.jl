############################################################################################
"""
$(TYPEDEF)
Holds information for array structure of data and reference particle.

# Fields
$(TYPEDFIELDS)
"""
mutable struct ParticleFilterConfig{A,B,C,D}
    "Data size and sorting information."
    data::BaytesCore.ArrayConfig{A,B}
    "Particle size and sorting information."
    particle::BaytesCore.ArrayConfig{C,D}
    function ParticleFilterConfig(
        dataconfig::ArrayConfig{A,B}, particleconfig::ArrayConfig{C,D}
    ) where {A,B,C,D}
        return new{A,B,C,D}(dataconfig, particleconfig)
    end
end
function ParticleFilterConfig(
    data::AbstractArray{T}, particle::AbstractArray{F}
) where {T,F}
    return ParticleFilterConfig(
        BaytesCore.ArrayConfig(data), BaytesCore.ArrayConfig(particle)
    )
end

function update!(
    config::ParticleFilterConfig, data::AbstractArray{T}, particle::AbstractArray{F}
) where {T,F}
    config.data = BaytesCore.ArrayConfig(data)
    config.particle = BaytesCore.ArrayConfig(particle)
    return nothing
end

############################################################################################
"""
$(TYPEDEF)
Holds tuning information for Particle Filter.

# Fields
$(TYPEDFIELDS)
"""
struct ParticleFilterTune{
    T<:Tagged,
    A<:BaytesCore.ParameterWeighting,B<:BaytesCore.ResamplingMethod,C<:ParticleReferencing,
    D,E,F,G,
    U<:BaytesCore.UpdateBool
} <: AbstractTune
    "Tagged Model parameter."
    tagged::T
    "Weighting Methods for particles."
    weighting::A
    "Resampling methods for particle trajectories."
    resampling::B
    "Referencing type for last particle at each iteration - either Conditional, Ancestral or Marginal Implementation."
    referencing::C
    "Contains data and reference size and sorting."
    config::ParticleFilterConfig{D,E,F,G}
    "Number of particle chains and tuning information."
    chains::BaytesCore.ChainsTune
    "Memory for latent and observed data."
    memory::ParticleFilterMemory
    "Boolean if generated quantities should be generated while sampling"
    generated::U
    "Current iteration."
    iter::Iterator
    function ParticleFilterTune(
        objective::O,
        weighting::A,
        resampling::B,
        referencing::C,
        config::ParticleFilterConfig{D,E,F,G},
        chains::BaytesCore.ChainsTune,
        memory::ParticleFilterMemory,
        generated::U,
        iter::Iterator,
    ) where {
        O<:Objective,
        A<:BaytesCore.ParameterWeighting,
        B<:BaytesCore.ResamplingMethod,
        C<:ParticleReferencing,
        D,
        E,
        F,
        G,
        U<:BaytesCore.UpdateBool
    }
        return new{typeof(objective.tagged),A,B,C,D,E,F,G,U}(
            objective.tagged,
            weighting,
            resampling,
            referencing,
            config,
            chains,
            memory,
            generated,
            iter,
        )
    end
end

############################################################################################

function update!(tune::ParticleFilterTune, data::AbstractArray, reference::AbstractArray)
    ## Update Iteration count
    init!(tune.iter, 1)
    ## Update Data/Particle Configuration
    update!(tune.config, data, reference)
    Ndata = maximum(tune.config.data.size)
    ## Update Number of Particles
    update_Nparticles, update_Ndata = update!(tune.chains, Ndata)
    ## Return if adjustments in either dimension are needed
    return update_Nparticles, update_Ndata, Ndata
end

############################################################################################
#export
export ParticleFilterTune
