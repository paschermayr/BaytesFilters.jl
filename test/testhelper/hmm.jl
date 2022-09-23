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
struct HMM <: ModelName end
hmm = ModelWrapper(HMM(), markov_param)
markov_objective = Objective(hmm, markov_data, :latent)

function ModelWrappers.generate(_rng::Random.AbstractRNG, objective::Objective{<:ModelWrapper{HMM}})
    @unpack model, data = objective
    @unpack μ, σ, latent = model.val
    return rand(_rng, Normal(Float32(2.), Float32(3.)))
end
function ModelWrappers.generate(_rng::Random.AbstractRNG, pf::ParticleFilter, objective::Objective{<:ModelWrapper{HMM}})
    @unpack model, data = objective
    @unpack μ, σ, latent = model.val
    return pf.particles.buffer.ℓobjectiveᵥ
end

function ModelWrappers.predict(_rng::Random.AbstractRNG, objective::Objective{<:ModelWrapper{HMM}})
    @unpack model, data = objective
    @unpack μ, σ, latent = model.val
    #NOTE: Check if correct prediction is applied even if standard predict is overloaded
	return Int(3) #rand(_rng, Normal(μ[latent[end]], σ[latent[end]]))
end

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
"Forward Filter HMM"
function filter_forward(objective::Objective{<:ModelWrapper{HMM}})
    @unpack model, data = objective
## Map Parameter to observation and state probabilities
    @unpack p = model.val
    dynamicsᵉ, _ = get_dynamics(model, model.val)
#!NOTE: Working straight with parameter instead of distributions here as easier to implement
    dynamicsˢ           = transpose( reduce(hcat, p ) ) #[ Categorical(p[iter]) for iter in eachindex(μ) ]
    initialˢ            = get_stationary( dynamicsˢ )
    structure = BaytesCore.ByRows()
## Assign Log likelihood
    ℓℒᵈᵃᵗᵃ = zeros(Float64, size(data,1), size(dynamicsˢ, 1) )
    Base.Threads.@threads for state in Base.OneTo( size(ℓℒᵈᵃᵗᵃ, 2) )
    for iter in Base.OneTo( size(ℓℒᵈᵃᵗᵃ, 1) )
            ℓℒᵈᵃᵗᵃ[iter, state] += logpdf( dynamicsᵉ[state], grab(data, iter, structure) )
        end
    end
## Initialize
    Nstates = size(dynamicsˢ, 1)
    α = zeros( size(ℓℒᵈᵃᵗᵃ) ) # α represents P( sₜ | e₁:ₜ ) for each t, which is numerically more stable than classical forward probabilities p(sₜ, e₁:ₜ)
    c = vec_maximum( @view(ℓℒᵈᵃᵗᵃ[1,:]) ) # Used for stable logsum() calculations for incremental log likelihood addition, i.e.: log( exp(x) + exp(y) ) == x + log(1 + exp(y -x ) ) for y >= x.
    ℓℒ = 0.0 #Log likelihood container
## Calculate initial probabilities
    for stateₜ in Base.OneTo(Nstates)
         α[1,stateₜ] += initialˢ[stateₜ] * exp(ℓℒᵈᵃᵗᵃ[1,stateₜ]-c) #Calculate initial p(s₁, e₁) ∝ P(s₁) * p(e₁ | s₁)
    end
## Normalize P( s₁, e₁ ) and return normalizing constant P(e₁), which can be used to calculate ℓℒ = P(e₁:ₜ) = p(e₁) ∏_k p(eₖ | e₁:ₖ₋₁)
    norm = sum( @view(α[1,:]) )
    α[1,:] ./= norm
    ℓℒ += log(norm) + c# log(norm) = p(eₜ | e₁:ₜ₋₁), c is constant that is added back after removing on top for numerical stability
## Loop through sequence
    for t = 2:size(α, 1)
        c = vec_maximum( @view(ℓℒᵈᵃᵗᵃ[t,:]) ) # Used for stable logsum() calculations, i.e.: log( exp(x) + exp(y) ) == x + log(1 + exp(y -x ) ) for y >= x.
        ## Calculate ∑_sₜ₋₁ P( sₜ | sₜ₋₁) * P(sₜ₋₁ | e₁:ₜ₋₁) - we sum over sₜ₋₁ - states are per row
        for stateₜ in Base.OneTo(Nstates)
            for stateₜ₋₁ in Base.OneTo(Nstates)
                α[t,stateₜ] += dynamicsˢ[stateₜ₋₁, stateₜ] * α[t-1,stateₜ₋₁] # * for both log and non-log version as inside log( sum(probability(...)))
            end
            ## Then multiply with ℒᵈᵃᵗᵃ corrector P( eₜ | sₜ )
            α[t,stateₜ] *= exp(ℓℒᵈᵃᵗᵃ[t,stateₜ]-c) # - c for higher numerical stability, will be added back to likelihood increment below
        end
        ## Normalize α and obtain P( eₜ | e₁:ₜ₋₁)
        norm = sum( @view(α[t,:]) )
        α[t,:] ./= norm
        ## Add normalizing constant p(eₜ | e₁:ₜ₋₁) to likelihood term
        ℓℒ += log(norm)+c
    end
    return (α, ℓℒ)
end
filter_forward(markov_objective)

############################################################################################
function ModelWrappers.simulate(rng::Random.AbstractRNG, model::ModelWrapper{HMM}; Nsamples = 1000)
    dynamicsᵉ, dynamicsˢ = get_dynamics(model, model.val)
    latentⁿᵉʷ = Vector{eltype(model.val.latent)}(undef, Nsamples)
    observedⁿᵉʷ = zeros(Nsamples)

    stateₜ = rand(rng, dynamicsˢ[ rand(1:length(dynamicsˢ) ) ] )
    latentⁿᵉʷ[1] = stateₜ
    observedⁿᵉʷ[1] = rand(rng, dynamicsᵉ[stateₜ])

    for iter in 2:size(observedⁿᵉʷ,1)
            stateₜ = rand(rng, dynamicsˢ[ stateₜ ]) #stateₜ for t-1 overwritten
            latentⁿᵉʷ[iter] = stateₜ
            observedⁿᵉʷ[iter] = rand(rng, dynamicsᵉ[stateₜ])
    end
    return observedⁿᵉʷ, latentⁿᵉʷ
end
