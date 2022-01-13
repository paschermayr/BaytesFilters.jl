############################################################################################
"""
$(TYPEDEF)
Holds tuning information for number of particles used in next run.

# Fields
$(TYPEDFIELDS)
"""
mutable struct ParticlesTune
    "Proposed coverage of number of particles/ number of data points."
    coverage::Float64
    "Threshold for resampling, between 0.00 and 1.00."
    threshold::Float64
    "Number of particles."
    Nparticles::Int64
    "Number of data points."
    Ndata::Int64
    function ParticlesTune(
        coverage::Float64, threshold::Float64, Nparticles::Int64, Ndata::Int64
    )
        ArgCheck.@argcheck coverage > 0.0 "Coverage needs to be positive"
        ArgCheck.@argcheck Nparticles > 0 "Nparticles needs to be positive"
        ArgCheck.@argcheck Ndata > 0 "Ndata needs to be positive"
        ArgCheck.@argcheck threshold >= 0.0 "Threshold needs to be positive"
        return new(coverage, threshold, Nparticles, Ndata)
    end
end

############################################################################################
function update!(ptune::ParticlesTune, Ndata::Integer)
    ## Check if data size has changed
    update_Ndata = ptune.Ndata != Ndata ? UpdateTrue() : UpdateFalse()
    if update_Ndata isa UpdateTrue
        ptune.Ndata = Ndata
    end
    ## Check if particle size has changed
    Nparticlesᵖ = Int64(floor(ptune.coverage * Ndata))
    #!NOTE: Update Nparticles if coverage falls below data size.
    update_Nparticles = ptune.Nparticles != Nparticlesᵖ ? UpdateTrue() : UpdateFalse()
    @inbounds if update_Nparticles isa UpdateTrue
        ptune.Nparticles = Nparticlesᵖ
    end
    return update_Nparticles, update_Ndata
end

############################################################################################
#export
export ParticlesTune
