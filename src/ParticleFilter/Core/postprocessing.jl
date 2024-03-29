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
        tup = BaytesCore.to_Tuple(sym)
        return new{typeof(tup),D}(tup, default)
    end
end
function (constructor::ParticleFilterConstructor)(
    _rng::Random.AbstractRNG,
    model::ModelWrapper,
    data::D,
    proposaltune::P,
    info::BaytesCore.SampleDefault
) where {D, P<:BaytesCore.ProposalTune}
    return ParticleFilter(
        _rng,
        Objective(model, data, Tagged(model, constructor.sym), proposaltune.temperature),
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
    #TTemperature = model.info.flattendefault.output
    TPrediction = infer(_rng, pf, model, data)
    TGenerated, TGenerated_algorithm = infer_generated(_rng, pf, model, data)
    return ParticleFilterDiagnostics{TPrediction, TGenerated, TGenerated_algorithm}
end

"""
$(SIGNATURES)
Infer type of predictions of kernel.

# Examples
```julia
```

"""
function infer(_rng::Random.AbstractRNG,
    kernel::ParticleKernel,
    tune::ParticleFilterTune,
    model::ModelWrapper,
    data::D
) where {D}
    reference = get_reference(tune.referencing, model, data, tune.tagged)
    trajectory = propagate(_rng, kernel, tune.memory, reference)
    ## Chose first available iterations where we can predict
    iter = max(tune.memory) + 1
    return typeof(predict(_rng, kernel, trajectory, reference, iter))
end
function infer(
    _rng::Random.AbstractRNG, pf::ParticleFilter, model::ModelWrapper, data::D
) where {D}
    kernel = dynamics_propose(Objective(model, data, pf.tune.tagged))
    return infer(_rng, kernel, pf.tune, model, data)
end


"""
$(SIGNATURES)
Infer type of generated quantities of PF sampler.

# Examples
```julia
```

"""
function infer_generated(
    _rng::Random.AbstractRNG, pf::ParticleFilter, model::ModelWrapper, data::D
) where {D}
    objective = Objective(model, data, pf.tune.tagged)
    TGenerated = typeof(generate(_rng, objective, pf.tune.generated))
    TGenerated_algorithm = typeof(generate(_rng, pf, objective, pf.tune.generated))
    return TGenerated, TGenerated_algorithm
end

############################################################################################
"""
$(SIGNATURES)
Generate statistics for algorithm given model parameter and data.

# Examples
```julia
```

"""
function generate(_rng::Random.AbstractRNG, algorithm::ParticleFilter, objective::Objective)
    return nothing
end
function generate(_rng::Random.AbstractRNG, algorithm::ParticleFilter, objective::Objective, gen::BaytesCore.UpdateTrue)
    return generate(_rng, algorithm, objective)
end
function generate(_rng::Random.AbstractRNG, algorithm::ParticleFilter, objective::Objective, gen::BaytesCore.UpdateFalse)
    return nothing
end

############################################################################################
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
        round(mean(diagnosticsᵛ[begin].base.ℓobjective); digits=Ndigits),
        ", final ℓlikelihood: ",
        round(mean(diagnosticsᵛ[end].base.ℓobjective); digits=Ndigits),
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
function result!(pf::ParticleFilter, result)
    return error("Not implemented for ParticleFilter.")
end
function get_result(pf::ParticleFilter)
    return error("Not implemented for ParticleFilter.")
end

function predict(_rng::Random.AbstractRNG, kernel::ParticleKernel, pf::ParticleFilter, objective::Objective)
    path = BaytesCore.draw!(_rng, pf.particles.weights)
    reference = get_reference(pf.tune.referencing,
        Objective(objective.model, objective.data, pf.tune.tagged, objective.temperature)
    )
    return predict(_rng, kernel, pf.particles, pf.tune, reference, path)
end
function predict(_rng::Random.AbstractRNG, pf::ParticleFilter, objective::Objective)
    obj = Objective(objective.model, objective.data, pf.tune.tagged, objective.temperature)
    return predict(_rng, dynamics_propose(obj), pf, obj)
end

############################################################################################
#export
export ParticleFilterConstructor, infer
