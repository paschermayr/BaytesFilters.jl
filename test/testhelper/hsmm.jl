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
        [truncated(Normal(-2.0, 5), -10.0, 0.0), truncated(Normal(2.0, 5), 0.0, 10.0)],
        [-.2, 0.2],
    ),
    σ=Param(
        [truncated(Normal(2.5, 10.0), 0.0, 10.0), truncated(Normal(5.0, 10.0), 0.0, 10.0)],
        [5.0, 2.0],
    ),
    λ=Param(
        [
            truncated(Normal(10.0, 100.0), 0.0, 100.0),
            truncated(Normal(20.0, 100.0), 0.0, 100.0),
        ],
        [10.0, 50.0],
    ),
    p=Param(Fixed(), [[1], [1]], ),
    latent=Param(Fixed(),semimarkov_latent, ),
)
struct HSMM <: ModelName end
hsmm = ModelWrapper(HSMM(), semimarkov_param)
semimarkov_objective = Objective(hsmm, semimarkov_data, :latent)

############################################################################################
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
function check_correctness(val::AbstractVector{T}) where {T}
## Get relevant fields
    s = getfield.(val, 1)
    d = getfield.(val, 2)
## Initate container that holds time when state changes
    StateIter = Int64[]
    DurationIter = Int64[]
## Compute all state changes
    statechanges = [s[iter]-s[iter-1] for iter in 2:length(s)]
    durationchanges = [d[iter]-d[iter-1] for iter in 2:length(d)]
## Get all iterations where s changes
    for iter in eachindex(statechanges)
        if statechanges[iter] != 0
            push!(StateIter, iter)
        end
    end
## Get all iterations where d changes
    for iter in eachindex(durationchanges)
        if durationchanges[iter] != -1
            push!(DurationIter, iter)
        end
    end
## Get all changes that are correct
    changes = [ StateIter[iter] == DurationIter[iter] for iter in eachindex(StateIter) ]
## Return total changes - correct changes (should b 0)
    return length(StateIter) - sum(changes)
end

#Check if HSMM has impossible transitions
function check_correctness(kernel::SemiMarkov, val::Vector{<:AbstractArray{T}}) where {T}
    return sum([check_correctness(val[iter]) for iter in eachindex(val)])
end
function check_correctness(kernel::SemiMarkov, val::AbstractMatrix{T}) where {T}
    return sum([check_correctness(val[iter, :]) for iter in Base.OneTo(size(val, 1))])
end

############################################################################################
"Forward Filter HSMM - target filtering distributions P( sₜ | e₁:ₜ ) instead of forward probabilities P( sₜ, e₁:ₜ ) for numerical stability"
function filter_forward(objective::Objective{<:ModelWrapper{HSMM}}; dmax = size(objective.data, 1))
    @unpack model, data = objective
## Map Parameter to observation and state probabilities
    @unpack p = model.val
    dynamicsᵉ, _, dynamicsᵈ = get_dynamics(model, model.val)
    #!NOTE: Working straight with parameter instead of distributions here as easier to implement
    dynamicsˢ = transpose( reduce(hcat, [  extend_state(p[iter], iter) for iter in eachindex(p) ] ) )
    initialˢ            = get_stationary( dynamicsˢ )
    sorting = BaytesCore.ByRows()
## Assign Log likelihood
    ℓℒᵈᵃᵗᵃ = zeros(Float64, size(data,1), size(dynamicsˢ, 1) ) #can be a Matrix instead of Array{k, 3} because eₜ | sₜ, dₜ independent of dₜ
    Base.Threads.@threads for state in Base.OneTo( size(ℓℒᵈᵃᵗᵃ, 2) )
    for iter in Base.OneTo( size(ℓℒᵈᵃᵗᵃ, 1) )
            ℓℒᵈᵃᵗᵃ[iter, state] = logpdf( dynamicsᵉ[state], grab(data, iter, sorting) )
        end
    end
## Initialize
    Nstates = size(dynamicsˢ, 1)
    α       = zeros( size(ℓℒᵈᵃᵗᵃ, 1), size(ℓℒᵈᵃᵗᵃ, 2), dmax ) # α represents P( sₜ, dₜ | e₁:ₜ ) for each t, where dmax is the maximum duration for a theoretically infinite upper bound for dₜ - numerically more stable than classical forward probabilities p(sₜ, dₜ, e₁ₜ)
    ℓℒ      = 0.0 #Log likelihood container
    c       = vec_maximum( @view(ℓℒᵈᵃᵗᵃ[1,:]) )
## Calculate initial probabilities P( s₁, d₁, e₁ ) ∝ P( s₁ ) * P( d₁ | s₁ ) * P( e₁ | s₁ )
    for stateₜ in Base.OneTo(Nstates) #Start with P( s₁ ) * P( e₁ | s₁ )
        α[1,stateₜ, :] .+= initialˢ[stateₜ] * exp(ℓℒᵈᵃᵗᵃ[1,stateₜ] - c )
        for durationₜ in 0:(dmax-1) #Proceed with P( d₁ | s₁ )
            α[1,stateₜ, durationₜ+1] *= pdf(dynamicsᵈ[ stateₜ ], durationₜ )
        end
    end
## Normalize P( s₁, d₁, e₁ ) and return normalizing constant P( e₁), which can be used to calculate ℓℒ = P(e₁:ₜ) = p(e₁) ∏_k p(eₖ | e₁:ₖ₋₁)
    norm = sum( @view(α[1,:, :]) )
    α[1, :, :] ./= norm
    ℓℒ += log(norm) + c # log(norm) = p(eₜ | e₁:ₜ₋₁), c is constant that is added back after removing on top for numerical stability
## Loop through sequence
    for t = 2:size(α, 1)
        c = vec_maximum( @view(ℓℒᵈᵃᵗᵃ[t,:]) )
        Base.Threads.@threads for stateₜ in Base.OneTo(Nstates)
            for durationₜ in 0:(dmax-1)
                ## First ∑_dₜ₋₁ P(dₜ | sₜ, dₜ₋₁) * (lower term)
                for durationₜ₋₁ in 0:(dmax-1)
                    ## Then calculate ∑_sₜ₋₁ P( sₜ | sₜ₋₁) * P(sₜ₋₁ | e₁:ₜ₋₁) - we sum over sₜ₋₁, which states are per row
                    for stateₜ₋₁ in Base.OneTo(Nstates)
                        if durationₜ₋₁ > 0 #Dirac delta function for duration and state
                            α[t,stateₜ, durationₜ+1] += ( (durationₜ₋₁-1) == durationₜ) * (stateₜ₋₁ == stateₜ) * α[t-1, stateₜ₋₁, durationₜ₋₁+1]
                        else #State and duration probabilities - self-transitions have 0 probabilities if durationₜ₋₁ == 0 via transition[i,i] == 0
                            α[t,stateₜ, durationₜ+1] += pdf(dynamicsᵈ[ stateₜ ], durationₜ) * dynamicsˢ[stateₜ₋₁, stateₜ] * α[t-1, stateₜ₋₁, durationₜ₋₁+1]
                        end
                    end
                end
                ## Then multiply with ℒᵈᵃᵗᵃ corrector P( eₜ | sₜ )
                α[t, stateₜ, durationₜ+1] *= exp(ℓℒᵈᵃᵗᵃ[t,stateₜ] - c ) # - c for higher numerical stability, will be added back to likelihood increment below
            end
        end
        ## Normalize α
        norm = sum( @view(α[t,:, :]) )
        α[t, :, :] ./= norm
        ## Add normalizing constant p(eₜ | e₁:ₜ₋₁) to likelihood term
        ℓℒ += log(norm) + c
    end
    return (α, ℓℒ)
end
filter_forward(semimarkov_objective)

############################################################################################
function ModelWrappers.simulate(rng::Random.AbstractRNG, model::ModelWrapper{HSMM}; Nsamples = 1000)
    dynamicsᵉ, dynamicsˢ, dynamicsᵈ = get_dynamics(model, model.val)
    latentⁿᵉʷ = Vector{eltype(model.val.latent)}(undef, Nsamples)
    observedⁿᵉʷ = zeros(Nsamples)

    stateₜ = rand(rng, dynamicsˢ[ rand(1:length(dynamicsˢ) ) ] )
    durationₜ = rand(rng, dynamicsᵈ[stateₜ])
    latentⁿᵉʷ[1] = (stateₜ, durationₜ)
    observedⁿᵉʷ[1] = rand(rng, dynamicsᵉ[stateₜ])

    for iter in 2:size(observedⁿᵉʷ,1)
        if durationₜ > 0
            durationₜ -=  1
            latentⁿᵉʷ[iter] = (stateₜ, durationₜ)
            observedⁿᵉʷ[iter] = rand(rng, dynamicsᵉ[stateₜ])
        else
            stateₜ = rand(rng, dynamicsˢ[ stateₜ ] ) #stateₜ for t-1 overwritten
            durationₜ = rand(rng, dynamicsᵈ[stateₜ])
            latentⁿᵉʷ[iter] = (stateₜ, durationₜ)
            observedⁿᵉʷ[iter] = rand(rng, dynamicsᵉ[stateₜ])
        end
    end
    return observedⁿᵉʷ, latentⁿᵉʷ
end
