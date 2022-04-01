############################################################################################
for iter in eachindex(objectives)
    @testset "Likelihood estimate, all models" begin
        ## Resampling methods
        for _resample in resamplemethods
            _obj = deepcopy(objectives[iter])
            ## Sample data given model and create new objective
            dat, lat = simulate(_rng, _obj.model; Nsamples = 1000)
            _tagged = Tagged(_obj.model, :latent)
            fill!(_obj.model, _tagged, (; latent = lat))
            _obj = Objective(_obj.model, dat, _tagged)

            ## Define PF default tuning parameter
            pfdefault = ParticleFilterDefault(;
                coverage = 2.00,
                threshold = 0.75,
                referencing = Marginal(),
                resampling = _resample
            )
            ## Initialize kernel and check if it can be run
            pfkernel = ParticleFilter(
                _rng,
                _obj,
                pfdefault
            )
            _val, _diag = propose(_rng, pfkernel, _obj)
            ## Check Marginal PF likelihood
            _type = infer(_rng, BaytesFilters.AbstractDiagnostics, pfkernel, _obj.model, _obj.data)
            diags = Vector{_type}(undef, 50)
            for idx in eachindex(diags)
                _, diags[idx] = propose(_rng, pfkernel, _obj)
            end
            ℓobjective_approx = [diags[idx].base.ℓobjective for idx in eachindex(diags)]
            _, ℓobjective_exact = filter_forward(_obj)
            @test ℓobjective_exact ≈ mean(ℓobjective_approx) atol = 50.0 #!NOTE: Initial distribution slightly different in forward filter for HSMM, rest should be atol ~ 1.0
        end
    end
end

############################################################################################
# Number of Particle estimate
function estimate_Nparticles(
    _rng::Random.AbstractRNG,
    pf::ParticleFilterConstructor,
    objective::Objective,
    variance::AbstractFloat;
    Nchains::Int64 = Threads.nthreads(),
    margin::Float64 = 0.25,
    itermax::Int64 = 100,
    mincoverage::Float64 = 0.01,
    printoutput = true
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
        if printoutput
            println(
                "(Target variance ", variance,
                ", Iteration ", iter,
                ") - Coverage set to ", round(Ncoverage[iter]; digits=3),
                " for variance of ", round(ℓobjective_variance[iter]; digits=2), "."
            )
        end
    end
    ## Return Nparticles
    return (
        Coverage = Ncoverage,
        ℓobjective_variance = ℓobjective_variance,
        ℓobjective = ℓobjective,
        itermax = iter,
        variance = variance
        )
end

for iter in eachindex(objectives)
    @testset "Number Particle estimation, all models" begin
        ## Resampling methods
        for _resample in resamplemethods
            ## Referencing methods
            for _reference in referencemethods
                _obj = deepcopy(objectives[iter])
                ## Sample data given model and create new objective
                dat, lat = simulate(_rng, _obj.model; Nsamples = 500)
                _tagged = Tagged(_obj.model, :latent)
                fill!(_obj.model, _tagged, (; latent = lat))
                _obj = Objective(_obj.model, dat, _tagged)
                ## Define PF default tuning parameter
                pfdefault = ParticleFilterDefault(;
                    referencing = _reference,
                    resampling = _resample
                )
                ## Check if we can initiate from Constructor
                constructor = ParticleFilterConstructor(:latent, pfdefault)
                ## Check number of particles
                _ℓobjectivevariance = 0.50
                _maxiter = 10
                _margin = 1.0
                _NparticlesEstimate = estimate_Nparticles(
                    _rng, constructor, _obj, _ℓobjectivevariance;
                    Nchains = 4,
                    margin = _margin,
                    itermax = _maxiter,
                    mincoverage = 0.1,
                    printoutput = false
                )
                _variance = _NparticlesEstimate.ℓobjective_variance[_NparticlesEstimate.itermax]
                _boundaries = (_variance - _margin, _variance + _margin)
                if _NparticlesEstimate.itermax < _maxiter
                    @test _boundaries[begin] < _variance < _boundaries[end]
                end
            end
        end
    end
end
