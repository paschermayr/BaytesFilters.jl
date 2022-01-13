############################################################################################
# Latent memory 1, data memory 0
markov_kernel = dynamics(markov_objective)
markov_memory = BaytesFilters._guessmemory(_rng, markov_kernel, markov_latent)
@testset "Markov Memory - Latent 1, data 0" begin
    @test markov_memory.latent == 1
    @test markov_memory.data == 0
end

semimarkov_kernel = dynamics(semimarkov_objective)
semimarkov_memory = BaytesFilters._guessmemory(_rng, semimarkov_kernel, semimarkov_latent)
@testset "SemiMarkov Memory - Latent 1, data 0" begin
    @test semimarkov_memory.latent == 1
    @test semimarkov_memory.data == 0
end

############################################################################################
# Latent memory 1, data memory 1
markov_HO_kernel = dynamics(markov_HO_objective)
markov_HO_memory = BaytesFilters._guessmemory(_rng, markov_HO_kernel, HO_latent)
@testset "HO Markov Memory - Latent 2, data 3" begin
    @test markov_HO_memory.latent == 2
    @test markov_HO_memory.data == 3
end
