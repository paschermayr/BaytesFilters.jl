############################################################################################
"""
$(TYPEDEF)

Contains temporary buffer values to avoid allocations during particle propagation.

# Fields
$(TYPEDFIELDS)
"""
struct ParticleBuffer{A,I}
    "Placeholder for particle trajectory if resampling is applied."
    val::Vector{A}
    "Buffer for resampled particle trajectory indices."
    ancestor::Vector{I}
    "Proposal trajectory and predicted latent variable."
    proposal::Vector{A}
    "Stores boolean if resampled at each iteration."
    resampled::Vector{Bool}
    function ParticleBuffer(
        reference::AbstractVector{A}, Nparticles::Integer, Ndata::Integer, F::Type{I}
    ) where {A,I<:Integer}
        val = typeof(reference)(undef, Nparticles)
        ancestor = Vector{F}(undef, Nparticles)
        proposal = typeof(reference)(undef, Ndata)
        resampled = zeros(Bool, Ndata)
        return new{A,I}(val, ancestor, proposal, resampled)
    end
end
function update!(buffer::ParticleBuffer, Nparticles::Integer, Ndata::Integer)
    resize!(buffer.val, Nparticles)
    resize!(buffer.ancestor, Nparticles)
    resize!(buffer.proposal, Ndata)
    resize!(buffer.resampled, Ndata)
    return nothing
end

############################################################################################
#export
export ParticleBuffer, update!
