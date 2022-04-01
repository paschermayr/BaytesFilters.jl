############################################################################################
# Higher Order Data Memory
HO_latent = randn(_rng, N)
HO_data = randn(_rng, size(HO_latent, 1))
HO_param = (
    μ₀=Param(0.0, Normal(0.0, 10.0)),
    μ=Param(-0.50, Normal(0.0, 10.0)),
    σ=Param(0.250, Gamma(0.5, 1.0)),
    ϕ=Param(0.90, Uniform(-1.0, 1.0)),
    latent=Param(HO_latent, [Normal(0.0, 10) for i in Base.OneTo(N)]),
)
struct Markov_HO <: ModelName end
markov_HO = ModelWrapper(Markov_HO(), HO_param)
markov_HO_objective = Objective(markov_HO, HO_data, :latent)

function BaytesFilters.dynamics(objective::Objective{<:ModelWrapper{<:Markov_HO}})
    @unpack model, data = objective
    @unpack μ₀, μ, σ, ϕ = model.val

    initialˢ = Normal(μ, σ / sqrt((1 - ϕ^2)))
    function transition(particles, iter)
        return Normal(μ + ϕ * (mean(view(particles, (iter - 2):(iter - 1))) - μ), σ)
    end
    function evidence(particles, iter)
        return Normal(μ₀, sqrt(exp(mean(view(particles, (iter - 3):iter)))))
    end
    return Markov(initialˢ, transition, evidence)
end
############################################################################################
