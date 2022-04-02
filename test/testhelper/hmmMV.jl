############################################################################################
#Markov Kernel
markov_MV_data = [randn(_rng, 2) for _ in 1:N]
param_HMM_MV = (;
    μ = Param([[-2., -2.], [0., 0.], [2., 2.]],
        [MvNormal( [i, i], [1., 1.] ) for i in -1.0:1:1.]),
    σ = Param([[ 2.2 0.2 ; 0.2 2.2], [ 1.1 0.1 ; 0.1 1.1], [ 2.2 0.2 ; 0.2 2.2] ],
        [InverseWishart(10, [0.8 0.5 ; 0.5 .8] ) for i in 1:3]),
    p = Param([[.6, .2, .2],  [.1, .1, .8], [.1, .1, .8]], [Dirichlet(3,3) for i in 1:3]),
    latent = Param(markov_latent, Fixed()),
)
struct HMM_MV <: ModelName end
hmm_MV = ModelWrapper(HMM_MV(), param_HMM_MV)
markov_MV_objective = Objective(hmm_MV, markov_MV_data, :latent)

################################################################################
function get_dynamics(model::ModelWrapper{<:HMM_MV}, θ)
    @unpack μ, σ, p = θ
    dynamicsᵉ = [ MvNormal(μ[iter], σ[iter]) for iter in eachindex(μ) ]
    dynamicsˢ = [ Categorical( p[iter] ) for iter in eachindex(μ) ]
    return dynamicsᵉ, dynamicsˢ
end

function BaytesFilters.dynamics(objective::Objective{<:ModelWrapper{<:HMM_MV}})
    @unpack model, data = objective
    dynamicsᵉ, dynamicsˢ = get_dynamics(model, model.val)

    initialˢ = Categorical(fill(1 / length(dynamicsᵉ), length(dynamicsᵉ)))
    transition(particles, iter) = dynamicsˢ[particles[iter - 1]]
    evidence(particles, iter) = dynamicsᵉ[particles[iter]]

    return Markov(initialˢ, transition, evidence)
end
#dynamics(markov_MV_objective)

############################################################################################
"Forward Filter HMM"
function filter_forward(objective::Objective{<:ModelWrapper{HMM_MV}})
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
filter_forward(markov_MV_objective)

############################################################################################
function ModelWrappers.simulate(rng::Random.AbstractRNG, model::ModelWrapper{HMM_MV}; Nsamples = 1000)
    dynamicsᵉ, dynamicsˢ = get_dynamics(model, model.val)
    latentⁿᵉʷ = Vector{eltype(model.val.latent)}(undef, Nsamples)
    observedⁿᵉʷ = [zeros(2) for _ in 1:Nsamples]

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
