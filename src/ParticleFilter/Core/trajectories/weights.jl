############################################################################################
# Types and structs - Reweighting Methods
struct Bootstrap <: BaytesCore.ParameterWeighting end

############################################################################################
# Bootstrap Method
function weight!(
    method::Bootstrap,
    kernel::ParticleKernel,
    weights::BaytesCore.ParameterWeights,
    dataₜ::D,
    val::AbstractMatrix{P},
    iter::Integer,
) where {D,P}
    ##Calculate weights
    ℓevidence!(kernel, weights.ℓweights, dataₜ, val, iter)
    ##Normalize weights
    BaytesCore.normalize!(weights)
    return nothing
end

############################################################################################
#export
export Bootstrap, weight!
