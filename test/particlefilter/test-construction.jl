############################################################################################
# Proposal steps and post processing
#_resample = resamplemethods[1]
#_reference = referencemethods[1]
for iter in eachindex(objectives)
    @testset "Kernel construction and propose, all models" begin
        ## Resampling methods
        for _resample in resamplemethods
            ## Referencing methods
            for _reference in referencemethods
                _obj = deepcopy(objectives[iter])
                ## Define PF default tuning parameter
                pfdefault = ParticleFilterDefault(;
                    referencing = _reference,
                    resampling = _resample
                )
                ## Check if Default options work
                ParticleFilter(_rng, _obj, pfdefault)
                ParticleFilter(_rng, _obj, pfdefault, SampleDefault())
                ## Check if we can initiate from Constructor
                constructor = ParticleFilterConstructor(:latent, pfdefault)
                constructor(_rng, _obj.model, _obj.data, 1., SampleDefault())
                ParticleFilter(:latent)
                ## Initialize kernel and check if it can be run
                pfkernel = ParticleFilter(
                    _rng,
                    _obj,
                    pfdefault
                )
                _val, _diag = propose(_rng, pfkernel, _obj)
                @test size(pfkernel.particles.val, 2) == length(_obj.data)
                ## Check if all particles are correct ~ Easy to check for Semi-Markov particles
                if _obj.model.id isa HSMM
                    @test check_correctness(pfkernel.particles.kernel, pfkernel.particles.val) == 0
                end
                ## Check if all ancestors are correct
                @test check_ancestors(pfkernel.particles.ancestor)
                ## Postprocessing
                BaytesFilters.generate_showvalues(_diag)()
                _type = infer(_rng, BaytesFilters.AbstractDiagnostics, pfkernel, _obj.model, _obj.data)
                @test _diag isa _type
                _type = infer(_rng, pfkernel.tune, pfkernel.particles.kernel, _obj.model, _obj.data)
                @test _diag.base.prediction isa _type
                _type = infer(_rng, pfkernel, _obj.model, _obj.data)
                @test _diag.base.prediction isa _type
                _type = BaytesFilters.infer_generated(_rng, pfkernel, _obj.model, _obj.data)
                @test _diag.generated isa _type
                @test predict(_rng, pfkernel, _obj) isa typeof(_diag.base.prediction)
            end
        end
    end
end

############################################################################################
# Propagation steps
for iter in eachindex(objectives)
    @testset "Kernel propagation, all models" begin
        data2 = vcat(objectives[iter].data, objectives[iter].data[1:10]) #randn(length(objectives[iter].data)+10)
        ## Resampling methods
        for _resample in resamplemethods
            ## Referencing methods
            _reference = Marginal()
            _obj = deepcopy(objectives[iter])
            ## Define PF default tuning parameter
            pfdefault = ParticleFilterDefault(;
                referencing = _reference,
                resampling = _resample
            )
            ## Initialize kernel and check if it can be run
            pfkernel = ParticleFilter(
                _rng,
                _obj,
                pfdefault
            )
            propose(_rng, pfkernel, _obj)
            ## Propagate forward
            propagate!(_rng, pfkernel, _obj.model, data2)
            @test size(pfkernel.particles.val, 2) == length(data2)
        end
    end
end
