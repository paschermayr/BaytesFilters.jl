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
    _rng::Random.AbstractRNG,
    model::ModelWrapper,
    data::D,
    temperature::F,
    info::BaytesCore.SampleDefault
) where {D, F<:AbstractFloat}
    return ParticleFilter(
        _rng,
        Objective(model, data, Tagged(model, constructor.sym), temperature),
        constructor.default,
        info
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
    TTemperature = model.info.flattendefault.output
    TPrediction = infer(_rng, pf, model, data)
    return ParticleFilterDiagnostics{TPrediction, TTemperature}
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
) where {T<:Real,P<:ParticleFilterDiagnostics}
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
function result!(pf::ParticleFilter, result::L) where {L<:ℓObjectiveResult}
    return error("Not implemented for ParticleFilter.")
end
function get_result(pf::ParticleFilter)
    return error("Not implemented for ParticleFilter.")
end
function get_ℓweight(pf::ParticleFilter)
    return pf.particles.ℓℒ.cumulative
end

############################################################################################
#export
export ParticleFilterConstructor, infer
