############################################################################################
"""
$(TYPEDEF)

Contains temporary buffer values to avoid allocations during particle propagation.

# Fields
$(TYPEDFIELDS)
"""
struct ParticleBuffer{A,I}
    "Contains buffer values for particles and ancestor for one iteration."
    parameter::BaytesCore.ParameterBuffer{A, I}
    "Proposal trajectory and predicted latent variable."
    proposal::Vector{A}
    "Stores boolean if resampled at each iteration."
    resampled::Vector{Bool}
    function ParticleBuffer(
        reference::AbstractVector{A}, Nparticles::Integer, Ndata::Integer, F::Type{I}
    ) where {A,I<:Integer}
        parameter = BaytesCore.ParameterBuffer(reference, Nparticles, F)
        proposal = typeof(reference)(undef, Ndata)
        resampled = zeros(Bool, Ndata)
        return new{A,I}(parameter, proposal, resampled)
    end
end
function update!(buffer::ParticleBuffer, Nparticles::Integer, Ndata::Integer)
    resize!(buffer.parameter, Nparticles)
    resize!(buffer.proposal, Ndata)
    resize!(buffer.resampled, Ndata)
    return nothing
end

############################################################################################
#export
export ParticleBuffer, update!
