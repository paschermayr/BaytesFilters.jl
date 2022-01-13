############################################################################################
# Types to distinguish between Gibbs and Metropolis updating
"Referencing struct: no updating via reference index"
struct Marginal <: ParticleReferencing end
"Referencing struct: last element in particle history always reference index"
struct Conditional <: ParticleReferencing end
"Referencing struct: last element in particle history weighted via reference index"
struct Ancestral <: ParticleReferencing end #PG with Ancestor Sampling

############################################################################################
"""
$(SIGNATURES)
Assign reference, depending on whether reference is tracked in Model.

# Examples
```julia
```

"""
function get_reference(referencing, model::ModelWrapper, data::D, tagged::Tagged) where {D}
    ## Check if only 1 parameter is tracked that has equal number of samples as data
    ArgCheck.@argcheck length(tagged.parameter) == 1 "Only 1 parameter can be tracked for Particle Filter."
    referenceᵖ = getindex(model.val, keys(tagged.parameter)[begin])
    return referenceᵖ
end
function get_reference(referencing, objective::Objective)
    return get_reference(referencing, objective.model, objective.data, objective.tagged)
end

############################################################################################
"""
$(SIGNATURES)
Inplace calculate ancestor weights.

# Examples
```julia
```

"""
function ancestralweight!(
    weights::ParticleWeights,
    valₜ::Union{P,AbstractArray{P}},
    kernel::ParticleKernel,
    val::AbstractArray,
    iter::Integer,
) where {P}
    #!NOTE: buffer first overwritten in ℓtransition! and then assigned again on LHS
    weights.buffer .=
        weights.ℓweights .+ ℓtransition!(weights.buffer, valₜ, kernel, val, iter)
    weights.buffer .-= logsumexp(weights.buffer)
    return nothing
end

############################################################################################
"""
$(SIGNATURES)
Sample ancestor reference for particle history.

# Examples
```julia
```

"""
function sample_ancestor(
    _rng::Random.AbstractRNG,
    weights::ParticleWeights,
    valₜ::Union{P,AbstractArray{P}},
    kernel::ParticleKernel,
    val::AbstractArray,
    iter::Integer,
) where {P}
    #!NOTE: buffer will be first overwritten on right side to inplace calculate new weights, and then assigned inplace on left side
    ancestralweight!(weights, valₜ, kernel, val, iter)
    #!NOTE Need to exp() particles.weights.buffer due to ancestralweights! calculation in log space
    weights.buffer .= exp.(weights.buffer)
    return randcat(_rng, weights.buffer)
end

############################################################################################
"""
$(SIGNATURES)
Resampling function for particles, dispatched on ParticleReferencing subtype.

# Examples
```julia
```

"""
function ancestors!(
    _rng::Random.AbstractRNG,
    referencing::ParticleReferencing,
    resampling::R,
    ancestor::AbstractArray, #referencing::Marginal
    iter::Integer,
    Nparticles::Integer,
    weights::AbstractVector{F},
) where {R<:ParticleResampling,F<:AbstractFloat}
    #!NOTE: Weights.buffer already exp.(weightsₙ) during resample_maybe( computeESS(particles.weights), X) in previous step
    resample!(
        _rng,
        resampling,
        ancestor,
        iter,
        weights,
        isa(referencing, Marginal) ? Nparticles : Nparticles - 1,
    )
    return nothing
end

############################################################################################
"""
$(SIGNATURES)
Set last particle at current iteration.

# Examples
```julia
```

"""
function set_reference!(
    referencing::Marginal,
    val::AbstractVector{P},
    reference::AbstractVector{P},
    iter::Integer,
) where {P}
    #!NOTE: Nothing to be done in Marginal case
    return nothing
end
function set_reference!(
    referencing::R, val::AbstractVector{P}, reference::AbstractVector{P}, iter::Integer
) where {R<:Union{Conditional,Ancestral},P}
    #!NOTE: Reference particle at t is the same for PG and PGAS
    val[end] = reference[iter]
    return nothing
end
function set_reference!(
    referencing, val::AbstractMatrix{P}, reference::AbstractVector{P}, iter::Integer
) where {P}
    return set_reference!(referencing, view(val, :, iter), reference, iter)
end

############################################################################################
"""
$(SIGNATURES)
Compute new reference index from current reference index.

# Examples
```julia
```

"""
function get_reference!(
    _rng::Random.AbstractRNG,
    referencing::Marginal,
    ancestor::AbstractVector{I},
    weights::ParticleWeights,
    referenceₜ::Union{P,AbstractArray{P}},
    kernel::ParticleKernel,
    val::AbstractArray,
    iter::Integer,
) where {I<:Integer,P}
    return nothing
end
function get_reference!(
    _rng::Random.AbstractRNG,
    referencing::Conditional,
    ancestor::AbstractVector{I},
    weights::ParticleWeights,
    referenceₜ::Union{P,AbstractArray{P}},
    kernel::ParticleKernel,
    val::AbstractArray,
    iter::Integer,
) where {I<:Integer,P}
    ancestor[end] = length(ancestor) #size(val, 1)
    return nothing
end
function get_reference!(
    _rng::Random.AbstractRNG,
    referencing::Ancestral,
    ancestor::AbstractVector{I},
    weights::ParticleWeights,
    referenceₜ::Union{P,AbstractArray{P}},
    kernel::ParticleKernel,
    val::AbstractArray,
    iter::Integer,
) where {I<:Integer,P}
    #!NOTE: Weights.buffer already exp.(weightsₙ) during resample_maybe( computeESS(particles.weights), X) in previous step
    ancestor[end] = sample_ancestor(_rng, weights, referenceₜ, kernel, val, iter)
    return nothing
end

function get_reference!(
    _rng::Random.AbstractRNG,
    referencing,
    ancestor::AbstractMatrix{I},
    weights::ParticleWeights,
    referenceₜ::Union{P,AbstractArray{P}},
    kernel::ParticleKernel,
    val::AbstractArray,
    iter::Integer,
) where {I<:Integer,P}
    return get_reference!(
        _rng,
        referencing,
        view(ancestor, :, iter - 1),
        weights,
        referenceₜ,
        kernel,
        val,
        iter,
    )
end

############################################################################################
#export
export Marginal,
    Conditional, Ancestral, get_reference, get_reference!, set_reference!, sample_ancestor
