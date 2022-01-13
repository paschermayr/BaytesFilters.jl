############################################################################################
# Models to be used in construction
objectives = [deepcopy(markov_objective), deepcopy(semimarkov_objective)]

resamplemethods = [Systematic(), Residual(), Stratified(), Simple()]
referencemethods = [Conditional(), Ancestral(), Marginal()]

## Make model
for iter in eachindex(objectives)
    @testset "Kernel construction and propagation, all models" begin
        ## Resampling methods
        for _resample in resamplemethods
            ## Referencing methods
            for _reference in referencemethods
                _obj = deepcopy(objectives[iter])
                #println(_resample, _reference)
                ## Define PF default tuning parameter
                pfdefault = ParticleFilterDefault(;
                    referencing = _reference,
                    resampling = _resample
                )
                ## Check if Default options work
                ParticleFilter(_rng, _obj; default = pfdefault)
                ParticleFilter(_rng, _obj, 1; default = pfdefault)
                ParticleFilter(_rng, _obj, 1, TemperDefault(); default = pfdefault)
                ## Check if we can initiate from Constructor
                constructor = ParticleFilterConstructor(:latent, pfdefault)
                constructor(_rng, _obj.model, _obj.data, 1, TemperDefault())
                ## Initialize kernel and check if it can be run
                pfkernel = ParticleFilter(
                    _rng,
                    _obj,
                    1,
                    TemperDefault(UpdateTrue(), 0.5);
                    default = pfdefault
                )
                propose(_rng, pfkernel, _obj)
                # Check if we can propagate data forward
                data2 = randn(length(_obj.data)+10)
                @test size(pfkernel.particles.val, 2) == length(_obj.data)
#=
                if _reference isa Marginal
                    propagate!(_rng, pfkernel, _obj.model, _obj.data)
                    propagate!(_rng, pfkernel, _obj.model, data2)
                    @test size(pfkernel.particles.val, 2) == length(data2)
                end
=#
            end
        end
    end
end
