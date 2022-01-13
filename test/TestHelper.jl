############################################################################################
# Constants
"RNG for sampling based solutions"
const _rng = Random.GLOBAL_RNG   # shorthand
Random.seed!(_rng, 1)

"Tolerance for stochastic solutions"
const _TOL = 1.0e-6

"Number of samples"
N = 10^3

############################################################################################
#Markov Kernel
markov_latent = Int32.(rand(_rng, Distributions.Categorical(3), N))
markov_data = randn(_rng, Float16, size(markov_latent, 1))
markov_param = (;
    μ=Param([-.1, 0.0, 0.1], fill(Normal(0.0, 10), 3)),
    σ=Param([5.0, 3.0, 2.0], fill(truncated(Normal(3.0, 10.0), 0.0, 10.0), 3)),
    p=Param(
        [[0.2, 0.6, 0.2], [0.2, 0.6, 0.2], [0.2, 0.6, 0.2]], [Dirichlet(3, 3) for i in 1:3]
    ),
    latent=Param(markov_latent, Fixed()),
)
struct HMM <: AbstractModel end
hmm = ModelWrapper(HMM(), markov_param)
markov_objective = Objective(hmm, markov_data, :latent)

function get_dynamics(model::ModelWrapper{<:HMM}, θ)
    @unpack μ, σ, p = θ
    dynamicsᵉ = [Normal(μ[iter], σ[iter]) for iter in eachindex(μ)]
    dynamicsˢ = [Categorical(p[iter]) for iter in eachindex(μ)]
    return dynamicsᵉ, dynamicsˢ
end
function BaytesFilters.dynamics(objective::Objective{<:ModelWrapper{<:HMM}})
    @unpack model, data = objective
    dynamicsᵉ, dynamicsˢ = get_dynamics(model, model.val)

    initialˢ = Categorical(fill(1 / length(dynamicsᵉ), length(dynamicsᵉ)))
    transition(particles, iter) = dynamicsˢ[particles[iter - 1]]
    evidence(particles, iter) = dynamicsᵉ[particles[iter]]

    return Markov(initialˢ, transition, evidence)
end
dynamics(markov_objective)

############################################################################################
#SemiMarkov Kernel
semimarkov_latent = [
    (2, 2), (2, 1), (2, 0), (1, 2), (1, 1), (1, 0), (2, 3), (2, 2), (2, 1), (2, 0), (1, 5)
]
#!NOTE: Convert data so we can check if particles have correct type assigned
semimarkov_latent = convert(Vector{Tuple{Int32,Int16}}, semimarkov_latent)
semimarkov_data = randn(_rng, size(semimarkov_latent, 1))
semimarkov_param = (;
    μ=Param(
        [-.2, 0.2],
        [truncated(Normal(-2.0, 5), -10.0, 0.0), truncated(Normal(2.0, 5), 0.0, 10.0)],
    ),
    σ=Param(
        [5.0, 2.0],
        [truncated(Normal(2.5, 10.0), 0.0, 10.0), truncated(Normal(5.0, 10.0), 0.0, 10.0)],
    ),
    λ=Param(
        [10.0, 50.0],
        [
            truncated(Normal(10.0, 100.0), 0.0, 100.0),
            truncated(Normal(20.0, 100.0), 0.0, 100.0),
        ],
    ),
    p=Param([[1], [1]], Fixed()),
    latent=Param(semimarkov_latent, Fixed()),
)
struct HSMM <: AbstractModel end
hsmm = ModelWrapper(HSMM(), semimarkov_param)
semimarkov_objective = Objective(hsmm, semimarkov_data, :latent)

function extend_state(transition::T, state::Integer) where {T}
    transitionⁿᵉʷ = zeros(eltype(transition), length(transition) + 1)
    transitionⁿᵉʷ[1:end .!= state] .= transition
    return transitionⁿᵉʷ
end
function get_dynamics(model::ModelWrapper{<:HSMM}, θ)
    @unpack μ, σ, λ, p = θ
    dynamicsᵈ = [Poisson(λ[iter]) for iter in eachindex(μ)]
    dynamicsᵉ = [Normal(μ[iter], σ[iter]) for iter in eachindex(μ)]
    dynamicsˢ = [Categorical(extend_state(p[iter], iter)) for iter in eachindex(μ)]
    return dynamicsᵉ, dynamicsˢ, dynamicsᵈ
end
function BaytesFilters.dynamics(objective::Objective{<:ModelWrapper{<:HSMM}})
    @unpack model, data = objective
    dynamicsᵉ, dynamicsˢ, dynamicsᵈ = get_dynamics(model, model.val)

    initialˢ = Categorical(fill(1 / length(dynamicsᵉ), length(dynamicsᵉ)))
    initialᵈ(sₜ) = dynamicsᵈ[sₜ]
    initial = SemiMarkovInitiation(initialˢ, initialᵈ)

    state(particles, iter) = dynamicsˢ[particles[iter - 1][1]]
    duration(s, iter) = dynamicsᵈ[s]
    transition = SemiMarkovTransition(state, duration)

    observation(particles, iter) = dynamicsᵉ[particles[iter][1]]
    return SemiMarkov(initial, transition, observation)
end
dynamics(semimarkov_objective)

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
struct Markov_HO <: AbstractModel end
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
