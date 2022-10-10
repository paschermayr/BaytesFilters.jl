############################################################################################
for iter in eachindex(objectives)
    @testset "Likelihood estimate, all models" begin
        ## Resampling methods
        for _resample in resamplemethods
            _obj = deepcopy(objectives[iter])
            ## Sample data given model and create new objective
            dat, lat = simulate(_rng, _obj.model; Nsamples = 500)
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
            _val, _diag = propose(_rng, ModelWrappers.dynamics(_obj), pfkernel, _obj)
            ## Check Marginal PF likelihood
            _type = infer(_rng, BaytesFilters.AbstractDiagnostics, pfkernel, _obj.model, _obj.data)
            diags = Vector{_type}(undef, 50)
            for idx in eachindex(diags)
                _, diags[idx] = propose(_rng, ModelWrappers.dynamics(_obj), pfkernel, _obj)
            end
            ℓobjective_approx = [diags[idx].base.ℓobjective for idx in eachindex(diags)]
            _, ℓobjective_exact = filter_forward(_obj)
            @test ℓobjective_exact ≈ mean(ℓobjective_approx) atol = 50.0 #!NOTE: Initial distribution slightly different in forward filter for HSMM, rest should be atol ~ 1.0
            #Check for results
            results(diags, pfkernel, 2, [.1, .2, .5, .8, .9])
        end
    end
end

############################################################################################
# Number of Particle estimate
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
