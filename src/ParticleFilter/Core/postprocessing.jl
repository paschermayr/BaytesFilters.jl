############################################################################################
"""
$(SIGNATURES)
Callable struct to make initializing ParticleFilter sampler easier in sampling library.

# Examples
```julia
```

"""
struct ParticleFilterConstructor{
    S<:Union{Symbol,NTuple{k,Symbol} where k},D<:ParticleFilterDefault
} <: AbstractConstructor
    "Parameter to be tracked in filter."
    sym::S
    "PF Default Arguments."
    default::D
    function ParticleFilterConstructor(
        sym::S, default::D
    ) where {S<:Union{Symbol,NTuple{k,Symbol} where k},D<:ParticleFilterDefault}
        return new{S,D}(sym, default)
    end
end
function (constructor::ParticleFilterConstructor)(
    _rng::Random.AbstractRNG, model::ModelWrapper, data::D,
    Nchains::Integer, temperdefault::BaytesCore.TemperDefault{B, F}
) where {D, B<:BaytesCore.UpdateBool, F<:AbstractFloat}
    return ParticleFilter(
        _rng, Objective(model, data, constructor.sym), Nchains, temperdefault; default=constructor.default
    )
end
function ParticleFilter(sym::S; kwargs...) where {S<:Union{Symbol,NTuple{k,Symbol} where k}}
    return ParticleFilterConstructor(sym, ParticleFilterDefault(; kwargs...))
end

############################################################################################
"""
$(SIGNATURES)
Infer ParticleFilter diagnostics type.

# Examples
```julia
```

"""
function infer(
    _rng::Random.AbstractRNG,
    diagnostics::Type{AbstractDiagnostics},
    pf::ParticleFilter,
    model::ModelWrapper,
    data::D,
) where {D}
    return ParticleDiagnostics{infer(_rng, pf, model, data)}
end

"""
$(SIGNATURES)
Infer type of predictions of kernel.

# Examples
```julia
```

"""
function infer(
    _rng::Random.AbstractRNG, pf::ParticleFilter, model::ModelWrapper, data::D
) where {D}
    reference = get_reference(pf.tune.referencing, model, data, pf.tune.tagged)
    trajectory = propagate(_rng, pf.particles.kernel, pf.tune.memory, reference)
    ## Chose first available iterations where we can predict
    iter = max(pf.tune.memory) + 1
    return typeof(predict(_rng, trajectory, pf.particles.kernel, reference, iter))
end

"""
$(SIGNATURES)
Return summary statistics for PF diagnostics.

# Examples
```julia
```

"""
function results(
    diagnosticsᵛ::AbstractVector{P},
    pf::ParticleFilter,
    Ndigits::Integer,
    quantiles::Vector{T},
) where {T<:Real,P<:ParticleDiagnostics}
    ## Print Trace
    println(
        "### ",
        Base.nameof(typeof(pf)),
        " parameter target: ",
        keys(pf.tune.tagged.parameter),
    )
    println(
        "Initial ℓlikelihood: ",
        round(mean(diagnosticsᵛ[begin].ℓℒ); digits=Ndigits),
        ", final ℓlikelihood: ",
        round(mean(diagnosticsᵛ[end].ℓℒ); digits=Ndigits),
        ".",
    )
    println(
        "Avg. number of used particles: ",
        round(
            mean(diagnosticsᵛ[iter].Nparticles for iter in eachindex(diagnosticsᵛ));
            digits=0,
        ),
        ".",
    )
    return println(
        "Avg. resampling steps: ",
        round(
            mean(diagnosticsᵛ[iter].resampled for iter in eachindex(diagnosticsᵛ)) * 100;
            digits=Ndigits,
        ),
        "%.",
    )
end

############################################################################################
function get_result(pf::ParticleFilter)
    return nothing
end

function get_tagged(pf::ParticleFilter)
    return pf.tune.tagged
end

function get_loglik(pf::ParticleFilter)
    return pf.particles.ℓℒ.cumulative
end

function get_prediction(diagnostics::ParticleDiagnostics)
    return diagnostics.prediction
end

function get_phase(pf::ParticleFilter)
    return nothing
end

function get_iteration(pf::ParticleFilter)
    return nothing
end

############################################################################################
#export
export ParticleFilterConstructor, infer
