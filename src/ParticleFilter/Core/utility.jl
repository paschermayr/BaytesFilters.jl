############################################################################################
"""
$(TYPEDEF)

Contains temporary buffer values to avoid allocations during particle propagation.

# Fields
$(TYPEDFIELDS)
"""
struct ParticleBuffer{A,I,P}
    "Contains buffer values for particles and ancestor for one iteration."
    parameter::BaytesCore.ParameterBuffer{A, I}
    "Proposal trajectory and predicted latent variable."
    proposal::Vector{A}
    "Predicted latent and oberved data"
    prediction::Vector{P}
    "Stores boolean if resampled at each iteration."
    resampled::Vector{Bool}
    "Stores incremental log target estimates for each iteration."
    ℓobjectiveᵥ::Vector{Float64}
    function ParticleBuffer(
        prediction::Vector{P}, reference::AbstractVector{A}, Nparticles::Integer, Ndata::Integer, F::Type{I}
    ) where {P,A,I<:Integer}
        parameter = BaytesCore.ParameterBuffer(reference, Nparticles, F)
        proposal = typeof(reference)(undef, Ndata)
        resampled = zeros(Bool, Ndata)
        ℓobjectiveᵥ = zeros(Float64, Ndata)
        return new{A,I,P}(parameter, proposal, prediction, resampled, ℓobjectiveᵥ)
    end
end
function update!(buffer::ParticleBuffer, Nparticles::Integer, Ndata::Integer)
    resize!(buffer.parameter, Nparticles)
    resize!(buffer.proposal, Ndata)
    resize!(buffer.resampled, Ndata)
    resize!(buffer.ℓobjectiveᵥ, Ndata)
    return nothing
end

############################################################################################
#export
export ParticleBuffer, update!
