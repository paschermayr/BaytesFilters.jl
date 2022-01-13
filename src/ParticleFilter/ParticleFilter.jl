############################################################################################
# Define abstract types
"""
$(TYPEDEF)
Available Particle Filter Kernels.
"""
abstract type ParticleKernel <: AbstractKernel end

"""
$(TYPEDEF)
Super type for various particle weighting techniques.
"""
abstract type ParticleWeighting end

"""
$(TYPEDEF)
Super type for various particle resampling techniques.
"""
abstract type ParticleResampling end

"""
$(TYPEDEF)
Super type for various particle referencing techniques.
"""
abstract type ParticleReferencing end

############################################################################################
# Import sub-folder
include("Core/Core.jl")
include("Kernels/Kernels.jl")

############################################################################################
# Export
export update!,
    init,
    init!,
    propose,
    propose!,
    propagate,
    propagate!,
    predict,
    ParticleKernel,
    ParticleWeighting,
    ParticleResampling,
    ParticleReferencing
