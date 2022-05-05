############################################################################################
"""
$(TYPEDEF)

Default arguments for Particle Filter constructor.

# Fields
$(TYPEDFIELDS)
"""
struct ParticleFilterDefault{
    M<:Union{Nothing, ParticleFilterMemory},
    A<:BaytesCore.ParameterWeighting,
    B<:BaytesCore.ResamplingMethod,
    C<:ParticleReferencing,
    T<:Integer,
    U<:BaytesCore.UpdateBool
}
    "Memory for particles and data."
    memory::M
    "Weighting Methods for particles."
    weighting::A
    "Resampling methods for particle trajectories."
    resampling::B
    "Referencing type for last particle at each iteration - either Conditional, Ancestral or Marginal Implementation."
    referencing::C
    "Coverage of Particles/datapoints."
    coverage::Float64
    "ESS threshold for resampling particle trajectories."
    threshold::Float64
    "Type for ancestors indices"
    ancestortype::Type{T}
    "Boolean if initial parameter are fixed or resampled."
    TunedModel::Bool
    "Boolean if generate(_rng, objective) for corresponding model is stored in PF Diagnostics."
    generated::U
    function ParticleFilterDefault(;
        memory::M=nothing,
        weighting::A=Bootstrap(),
        resampling::B=Systematic(),
        referencing::C=Marginal(),
        coverage=0.50,
        threshold=0.75,
        ancestortype::Type{T}=Int32,
        TunedModel=true,
        generated=BaytesCore.UpdateFalse()
    ) where {
        M<:Union{Nothing, ParticleFilterMemory},
        A<:BaytesCore.ParameterWeighting,
        B<:BaytesCore.ResamplingMethod,
        C<:ParticleReferencing,
        T<:Integer
    }
        ArgCheck.@argcheck 0.0 < coverage
        ArgCheck.@argcheck 0.0 <= threshold <= 1.0
        return new{M,A,B,C,T,typeof(generated)}(
            memory,
            weighting,
            resampling,
            referencing,
            coverage,
            threshold,
            ancestortype,
            TunedModel,
            generated
        )
    end
end

############################################################################################
"""
$(TYPEDEF)

Contains particles, transition kernel and all other relevant tuning information for particle propagation.

# Fields
$(TYPEDFIELDS)
"""
struct ParticleFilter{A<:Particles,B<:ParticleFilterTune} <: AbstractAlgorithm
    "Particle values and kernel."
    particles::A
    "Tuning configuration for particles."
    tune::B
    function ParticleFilter(
        particles::A, tune::B
    ) where {A<:Particles,B<:ParticleFilterTune}
        return new{A,B}(particles, tune)
    end
end

function ParticleFilter(
    _rng::Random.AbstractRNG,
    objective::Objective,
    default::ParticleFilterDefault=ParticleFilterDefault(),
    info::BaytesCore.SampleDefault = BaytesCore.SampleDefault()
)
    ## Checks before algorithm is initiated
    ArgCheck.@argcheck hasmethod(dynamics, Tuple{typeof(objective)}) "No Filter dynamics given your objective exists - assign dynamics(objective::Objective{MyModel})"
    @unpack memory, weighting, resampling, referencing, coverage, threshold, ancestortype, TunedModel, generated = default
    @unpack model, data, tagged = objective
    ## Assign model dynamics
    kernel = ModelWrappers.dynamics(objective)
    ## Initiate a valid reference given model.data and tagged.parameter
    reference = get_reference(referencing, objective)
    ## Guess particle and data memory
    memory = memory isa ParticleFilterMemory ? memory : _guessmemory(_rng, kernel, reference)
    ArgCheck.@argcheck memory isa ParticleFilterMemory
    ## Forward sample new reference if TunedModel == false
    if !TunedModel
        reference = propagate(_rng, kernel, memory, reference)
    end
    ## Assign tuning struct
    Ndata = maximum(size(data))
    Nparticles = Int64(floor(coverage * Ndata))
    tune = ParticleFilterTune(
        objective,
        weighting,
        resampling,
        referencing,
        ParticleFilterConfig(data, reference),
        BaytesCore.ChainsTune(coverage, threshold, Nparticles, Ndata),
        memory,
        generated,
        Iterator(1),
    )
    ## Prediction buffer
    Tprediction = infer(_rng, tune, kernel, model, data)
    prediction_buffer = Vector{Tprediction}(undef, 1)
    ## Create Particles container
    particles = Particles(reference, prediction_buffer, kernel, ancestortype, Nparticles, Ndata)
    ## Return Particle filter
    return ParticleFilter(particles, tune)
end

############################################################################################
"""
$(SIGNATURES)
Run particle filter and do not change reference trajectory.

# Examples
```julia
```

"""
function propose(
    _rng::Random.AbstractRNG,
    pf::ParticleFilter,
    objective::Objective,
    reference::AbstractArray{P}=get_reference(pf.tune.referencing, objective),
) where {P}
    ## Set iterations and log likelihood to initial state and resize proposal
    init!(pf.particles.ℓobjective)
    init!(pf.tune.iter, 1)
    ## Initialize particles
    initial!(_rng, pf.particles, pf.tune, reference, objective)
    ## Propagate particles forward
    propagate!(_rng, pf.particles, pf.tune, reference, objective)
    ## Sort all particles back into correct order
    BaytesCore.shuffle_backward!(pf.particles, pf.tune)
    ## Draw proposal path, update proposal with corresponding path and predict future state
    path = BaytesCore.draw!(_rng, pf.particles.weights)
    prediction = predict(_rng, pf.particles, pf.tune, reference, path)
    pf.particles.buffer.prediction[1] = prediction
    ## Update model parameter with reference trajectory
    ModelWrappers.fill!(
        objective.model,
        objective.tagged,
        #!NOTE: Create new Array with current iteration length to keep same type as in Model
        BaytesCore.to_NamedTuple(
            keys(objective.tagged.parameter),
            pf.particles.val[path, 1:(pf.tune.iter.current - 1)],
        ),
    )
    ## Create Diagnostics and return output
    diagnostics = ParticleFilterDiagnostics(
        BaytesCore.BaseDiagnostics(
            pf.particles.ℓobjective.cumulative,
            objective.temperature,
            prediction,
            pf.tune.iter.current-1
        ),
        pf.particles.ℓobjective.current,
        pf.tune.chains.Nchains,
        mean(pf.particles.buffer.resampled),
        ModelWrappers.generate(_rng, objective, pf.tune.generated)
    )
    ## Return new model parameter and diagnostics
    return objective.model.val, diagnostics
end

############################################################################################
"""
$(SIGNATURES)
Run particle filter and change reference trajectory with proposal trajectory.

# Examples
```julia
```

"""
function propose!(
    _rng::Random.AbstractRNG,
    pf::ParticleFilter,
    model::ModelWrapper,
    data::D,
    temperature::T = model.info.reconstruct.default.output(1.0),
    update::U=BaytesCore.UpdateTrue(),
) where {D, T<:AbstractFloat, U<:BaytesCore.UpdateBool}
    ## Update PF parameter
    objective = Objective(model, data, pf.tune.tagged, temperature)
    ## Collect reference
    reference = get_reference(pf.tune.referencing, objective)
    ## Check if Ndata and Nparticles have to be adjusted
    if update isa BaytesCore.UpdateTrue
        ## Check if number of particles need to be updated
        update_Nparticles, update_Ndata, Ndata = update!(pf.tune, data, reference)
        ## Resize particles if needed
        update!(
            pf.particles,
            update_Nparticles,
            update_Ndata,
            dynamics(objective),
            reference,
            pf.tune.chains.Nchains,
            Ndata,
        )
    end
    ## Propagate particles forward
    val, diagnostics = propose(_rng, pf, objective, reference)
    ## Update ModelWrapper parameter with reference trajectory
    model.val = val
    ## Return diagnostics
    return val, diagnostics
end

############################################################################################
"""
$(SIGNATURES)
Propagate particle filter forward with new data.

# Examples
```julia
```

"""
function propagate!(
    _rng::Random.AbstractRNG,
    pf::ParticleFilter,
    model::ModelWrapper,
    data::D,
    temperature::T = model.info.reconstruct.default.output(1.0)
) where {D, T<:AbstractFloat}
    ## Check if pf and data fulfill requirements for particle propagation
    ArgCheck.@argcheck isa(pf.tune.referencing, Marginal) "PF propagation only allowed for marginal particle filter"
    ## Assign new dynamics in case particles dependent on data
    objective = Objective(model, data, pf.tune.tagged, temperature)
    pf.particles.kernel = ModelWrappers.dynamics(objective)
    ## Collect reference trajectory
    reference = get_reference(pf.tune.referencing, objective)
    # Check if reference no larger than data dimension
    ArgCheck.@argcheck size(reference, 1) <= maximum(size(objective.data)) "Reference trajectory is longer than provided data input."
    ##Set last trajectory as reference trajectory, so at least 1 trajectory is compatible to extend model parameter.
    #!NOTE: This is necessary for the prediction of the latent variable, where we can take into account ALL particles.
    #!NOTE: This is different than just forward sampling, as after resampling particle counts may be greatly different than mere transition distribution parameter.
    @inbounds @simd for iter in eachindex(reference)
        pf.particles.val[end, iter] = reference[iter]
    end
    ## Update maximal number of iterations and resize particles ~ Need to manually resize as want to keep weights
    pf.tune.chains.Ndata = maximum(size(objective.data))
    resize!(pf.particles.val, pf.tune.chains.Nchains, pf.tune.chains.Ndata)
    resize!(pf.particles.ancestor, pf.tune.chains.Nchains, pf.tune.chains.Ndata)
    update!(pf.particles.buffer, pf.tune.chains.Nchains, pf.tune.chains.Ndata)
    ## Assign temporary variables so visible outside of data loop
    path = size(pf.particles.val, 1)
    ## Iterate through new data
    for iter in (pf.tune.iter.current):(pf.tune.chains.Ndata)
        ## Resample particle ancestors if resampling criterion fullfiled
        resample!(_rng, pf.particles, pf.tune, reference)
        ## Transition particles
        transition!(_rng, pf.particles.kernel, pf.particles.val, iter)
        ## Calculate particle weights and log likelihood
        #!NOTE: cannot do this at the same time if particles are resampled adaptively, as normalized weights will change if not resampled
        weight!(
            BaytesCore.grab(objective.data, pf.tune.iter.current, pf.tune.config.data),
            pf.particles,
            pf.tune,
        )
        update!(pf.particles.ℓobjective, logmeanexp(pf.particles.weights.ℓweights))
        ## Compute proposal weights based on reference trajectory ~ Differs from regular setup
        @inbounds for idx in Base.OneTo(size(pf.particles.val, 1))
            pf.particles.weights.buffer[idx] = ℓtransition(
                pf.particles.val[idx, iter],
                pf.particles.kernel,
                reference,
                pf.tune.iter.current,
            )
        end
        pf.particles.weights.buffer .-= logsumexp(pf.particles.weights.buffer)
        pf.particles.weights.buffer .= exp.(pf.particles.weights.buffer)
        path = randcat(_rng, pf.particles.weights.buffer)
        #!NOTE: Will overwrite model.val.reference
        push!(reference, pf.particles.val[path, iter])
        ## Update current iteration
        update!(pf.tune.iter)
    end
    ## Assign default order to ancestors for final index
    @inbounds for Nrow in Base.OneTo(length(pf.particles.buffer.parameter.val))
        pf.particles.ancestor[Nrow, pf.tune.iter.current - 1] = Nrow
    end
    ## Sort all particles back into correct order
    BaytesCore.shuffle_backward!(pf.particles, pf.tune)
    ## Predict new state and observation
    prediction = predict(_rng, pf.particles, pf.tune, reference, path)
    pf.particles.buffer.prediction[1] = prediction

    ## Update model parameter with reference trajectory
    # NOTE: Updated NamedTuple needed so no pointer issues when storing in trace
    ModelWrappers.fill!(
        model,
        pf.tune.tagged,
        #!NOTE: Create new Array with current iteration length to keep same type as in Model
        BaytesCore.to_NamedTuple(
            keys(pf.tune.tagged.parameter),
            pf.particles.val[path, 1:(pf.tune.iter.current - 1)],
        ),
    )
    ## Create Diagnostics and return output
    diagnostics = ParticleFilterDiagnostics(
        BaytesCore.BaseDiagnostics(
            pf.particles.ℓobjective.cumulative,
            temperature,
            prediction,
            pf.tune.iter.current-1
        ),
        pf.particles.ℓobjective.current,
        pf.tune.chains.Nchains,
        mean(pf.particles.buffer.resampled),
        ModelWrappers.generate(_rng, objective, pf.tune.generated)
    )
    return model.val, diagnostics
end

############################################################################################
#export
export ParticleFilter, ParticleFilterDefault, propose, propose!, propagate!
