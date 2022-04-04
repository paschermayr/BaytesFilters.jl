############################################################################################
# Proposal steps and post processing
#=
iter=1
_resample = resamplemethods[1]
_reference = referencemethods[1]
generated = generating[2]
=#
for iter in eachindex(objectives)
    @testset "Kernel construction and propose, all models" begin
        ## Resampling methods
        for _resample in resamplemethods
            ## Referencing methods
            for _reference in referencemethods
                for generated in generating
                    _obj = deepcopy(objectives[iter])
                    ## Define PF default tuning parameter
                    pfdefault = ParticleFilterDefault(;
                        referencing = _reference,
                        resampling = _resample,
                        generated = generated
                    )
                    ## Check if Default options work
                    ParticleFilter(_rng, _obj, pfdefault)
                    ParticleFilter(_rng, _obj, pfdefault, SampleDefault())
                    ## Check if we can initiate from Constructor
                    constructor = ParticleFilterConstructor(:latent, pfdefault)
                    constructor(_rng, _obj.model, _obj.data, 1., SampleDefault())
                    ParticleFilter((:latent,))
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
end

############################################################################################
# Propagation steps
for iter in eachindex(objectives)
    Ndata_extension = [1, 10]
    @testset "Kernel propagation, all models" begin
        for newdata in Ndata_extension
        # First extend 1 data point, then 10
            # -> once PF will only extend Ndata, then Nparticles and Ndata
            data2 = vcat(objectives[iter].data, objectives[iter].data[1:newdata]) #randn(length(objectives[iter].data)+10)
            ## Resampling methods
            for _resample in resamplemethods
                ## Referencing methods
                _reference = Marginal()
                _obj = deepcopy(objectives[iter])
                ## Define PF default tuning parameter
                pfdefault = ParticleFilterDefault(;
                    #!NOTE: So Nparticles does not change if 1 more data point is added
                    coverage = .5,
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
                propose!(_rng, pfkernel, _obj.model, data2)
                ## Check if Nparticles change for more data accordingly even if not propagated
                pfkernel2 = ParticleFilter(
                    _rng,
                    _obj,
                    pfdefault
                )
                ChainsInit = pfkernel2.tune.chains.Nchains
                @test ChainsInit == size(pfkernel2.particles.val,1)
                propose!(_rng, pfkernel2, deepcopy(objectives[iter].model), data2)
                @test size(pfkernel2.particles.val,1) <= size(data2,1)*pfkernel2.tune.chains.coverage
                ## Check if Nparticles change for less data accordingly even if not propagated
                propose!(_rng, pfkernel2, deepcopy(objectives[iter].model), data2[1:Int(round(length(data2)/2))])
                @test size(pfkernel2.particles.val,1) <= size(data2[1:Int(round(length(data2)/2))],1)*pfkernel2.tune.chains.coverage
            end
        end
    end
end
