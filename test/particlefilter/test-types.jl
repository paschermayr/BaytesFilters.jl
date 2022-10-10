############################################################################################
# Check types for Markov Kernel
markov_pf = ParticleFilter(_rng, markov_objective)
_vals, _diag = propose(_rng, ModelWrappers.dynamics(markov_objective), markov_pf, markov_objective)
@testset "PF - Types - Markov" begin
    @test eltype(markov_latent) == eltype(markov_pf.particles.val)
    @test typeof(markov_latent) == typeof(markov_pf.particles.buffer.parameter.val)
    @test typeof(markov_latent) == typeof(markov_pf.particles.buffer.proposal)
    ## Check predictions
    #@test typeof(_diag.prediction) == Tuple{eltype(markov_latent),eltype(markov_data)}
end

############################################################################################
# Check types for SemiMarkov Kernel
semimarkov_pf = ParticleFilter(_rng, semimarkov_objective)
_vals, _diag = propose(_rng, ModelWrappers.dynamics(semimarkov_objective), semimarkov_pf, semimarkov_objective)
@testset "PF - Types - SemiMarkov" begin
    @test eltype(semimarkov_latent) == eltype(semimarkov_pf.particles.val)
    @test typeof(semimarkov_latent) == typeof(semimarkov_pf.particles.buffer.parameter.val)
    @test typeof(semimarkov_latent) == typeof(semimarkov_pf.particles.buffer.proposal)
    ## Check predictions
    #@test typeof(_diag.prediction) == Tuple{eltype(semimarkov_latent),eltype(semimarkov_data)}
end
