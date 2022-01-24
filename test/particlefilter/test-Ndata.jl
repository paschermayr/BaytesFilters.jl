############################################################################################
# Check if data changes, PF settings change
markov_data_small = randn(_rng, Float16, 50)
markov_data_large = randn(_rng, Float16, 10000)

markov_pf = ParticleFilter(_rng, deepcopy(markov_objective))
_vals, _diag = propose(_rng, markov_pf, markov_objective)

@testset "PF - Markov Allocations - Data smaller than initiation" begin
    Ndata = length(markov_data_small)
    propose!(_rng, markov_pf, markov_objective.model, markov_data_small)
    @test length(markov_objective.model.val.latent) == Ndata
    @test size(markov_pf.particles.val,2) == Ndata
    @test size(markov_pf.particles.ancestor,2) == Ndata
    @test length(markov_pf.particles.buffer.resampled) == Ndata
    @test markov_pf.tune.chains.Ndata == Ndata
end

#=
markov_pf = ParticleFilter(_rng, deepcopy(markov_objective))
_vals, _diag = propose(_rng, markov_pf, markov_objective)

@testset "PF - SemiMarkov Allocations - Data larger than initiation" begin
    Ndata = length(markov_data_large)
    propose!(_rng, markov_pf, markov_objective.model, markov_data_large)
    @test length(markov_objective.model.val.latent) == Ndata
    @test size(markov_pf.particles.val,2) == Ndata
    @test size(markov_pf.particles.ancestor,2) == Ndata
    @test length(markov_pf.particles.buffer.resampled) == Ndata
    @test markov_pf.tune.particles.Ndata == Ndata
end
=#
