############################################################################################
# Types and structs - Reweighting Methods
struct Bootstrap <: BaytesCore.ParameterWeighting end

############################################################################################
# Bootstrap Method
function weight!(
    method::Bootstrap,
    weights::BaytesCore.ParameterWeights,
    dataₜ::D,
    kernel::ParticleKernel,
    val::AbstractMatrix{P},
    iter::Integer,
) where {D,P}
    ##Calculate weights
    ℓevidence!(weights.ℓweights, dataₜ, kernel, val, iter)
    ##Normalize weights
    BaytesCore.normalize!(weights)
    return nothing
end

############################################################################################
#export
export Bootstrap, weight!
