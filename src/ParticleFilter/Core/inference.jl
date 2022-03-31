############################################################################################
"""
$(SIGNATURES)
Function that checks for number of particles to achieve target variance of log target estimate.

# Examples
```julia
```

"""

function estimate_Nparticles(
    _rng::Random.AbstractRNG,
    pf::ParticleFilterConstructor,
    objective::Objective,
    variance::AbstractFloat;
    Nchains::Int64 = Threads.nthreads(),
    margin::Float64 = 0.25,
    itermax::Int64 = 100,
    mincoverage::Float64 = 0.01
)
    ArgCheck.@argcheck Nchains > 0
    ArgCheck.@argcheck margin > 0
    ArgCheck.@argcheck itermax > 0
## Assign all kernels
    algorithms = map(iter -> pf(_rng, objective.model, objective.data, objective.temperature, SampleDefault()), Base.OneTo(Nchains))
    models = map(iter -> deepcopy(objective.model), Base.OneTo(Nchains))
## Preallocate buffer for variance and mean of ℓobjective estimate
    ℓobjective_variance = zeros(itermax)
    ℓobjective = zeros(Nchains, itermax)
    Ncoverage = zeros(itermax)
    Ncoverage[1] = pf.default.coverage
    iter = 1
## Run first test for estimates
    for chain in Base.OneTo(Nchains)
        _, diagnostics = propose!(_rng, algorithms[chain], models[chain], objective.data, objective.temperature, UpdateTrue())
        ℓobjective[chain, iter] = diagnostics.base.ℓobjective
    end
    ℓobjective_variance[iter] = var(view(ℓobjective, :, iter))
    while iter < itermax
        iter += 1
    ## Compute variance and mean of ℓobjective estimate
        Threads.@threads for chain in Base.OneTo(Nchains)
            _, diagnostics = propose!(_rng, algorithms[chain], models[chain], objective.data, objective.temperature, UpdateTrue())
            ℓobjective[chain, iter] = diagnostics.base.ℓobjective
        end
        ℓobjective_variance[iter] = var(view(ℓobjective, :, iter))
    ## Check if conditions are fullfilled
        _margin = ℓobjective_variance[iter]*margin
        boundaries = (ℓobjective_variance[iter] - _margin, ℓobjective_variance[iter] + _margin)
        if boundaries[begin] < variance < boundaries[end]
            break
        end
    ## Adjust number of particles
        if ℓobjective_variance[iter] > variance
            #Note: Double particles at the beginning and then slowly decrease change over iterations
            Ncoverage[iter] = Ncoverage[iter-1] + Ncoverage[iter-1] * 2.0/iter
        else
            Ncoverage[iter] = max(mincoverage, Ncoverage[iter-1] - Ncoverage[iter-1] * 2.0/iter)
        end
        for chain in Base.OneTo(Nchains)
            algorithms[chain].tune.chains.coverage = Ncoverage[iter]
        end
    ## Print current run diagnostics
    println(
        "(Target variance ", variance,
        ", Iteration ", iter,
        ") - Coverage set to ", round(Ncoverage[iter]; digits=3),
        " for variance of ", round(ℓobjective_variance[iter]; digits=2), "."
        #". Number of particles: ", size(algorithms[1].particles.val, 1)
    )
    end
    ## Return Nparticles
    return (
        Coverage = Ncoverage,
        ℓobjective_variance = ℓobjective_variance,
        ℓobjective = ℓobjective
        )
end

############################################################################################
#export
export estimate_Nparticles
