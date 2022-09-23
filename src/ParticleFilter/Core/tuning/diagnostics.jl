############################################################################################
"""
$(SIGNATURES)
Contains information about log-likelihood, expected sample size and proposal trajectory.

# Examples
```julia
```

"""
struct ParticleFilterDiagnostics{P,G,A} <: AbstractDiagnostics
    "Diagnostics used for all Baytes kernels"
    base::BaytesCore.BaseDiagnostics{P}
    "Incremental log objective at current iteration."
    ℓincrement::Float64
    "Number of particles used in proposal steps."
    Nparticles::Int64
    "Average number of resampling steps."
    resampled::Float64
    "Generated quantities specified for objective"
    generated::G
    "Generated quantities specified for algorithm"
    generated_algorithm::A
    function ParticleFilterDiagnostics(
        base::BaytesCore.BaseDiagnostics{P},
        ℓincrement::Float64,
        Nparticles::Int64,
        resampled::Float64,
        generated::G,
        generated_algorithm::A
    ) where {P, G, A}
        return new{P, G, A}(base, ℓincrement, Nparticles, resampled, generated, generated_algorithm)
    end
end

############################################################################################
"""
$(SIGNATURES)
Show relevant diagnostic results.

# Examples
```julia
```

"""
function generate_showvalues(diagnostics::D) where {D<:ParticleFilterDiagnostics}
    return function showvalues()
        return (:pf, "diagnostics"),
        (:loglik_estimate, diagnostics.base.ℓobjective),
        (:Temperature, diagnostics.base.temperature),
        (:Nparticles, diagnostics.Nparticles),
        (:resampled, diagnostics.resampled),
        (:generated, diagnostics.generated),
        (:generated_algorithm, diagnostics.generated_algorithm)
    end
end

############################################################################################
#export
export ParticleFilterDiagnostics, generate_showvalues
