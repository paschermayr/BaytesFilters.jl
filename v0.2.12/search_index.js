var documenterSearchIndex = {"docs":
[{"location":"intro/#Introduction","page":"Introduction","title":"Introduction","text":"","category":"section"},{"location":"intro/","page":"Introduction","title":"Introduction","text":"Yet to be properly done.","category":"page"},{"location":"","page":"Home","title":"Home","text":"CurrentModule = BaytesFilters","category":"page"},{"location":"#BaytesFilters","page":"Home","title":"BaytesFilters","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for BaytesFilters.","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [BaytesFilters]","category":"page"},{"location":"#BaytesFilters.BaytesFilters","page":"Home","title":"BaytesFilters.BaytesFilters","text":"Particle Filter module.\n\n\n\n\n\n","category":"module"},{"location":"#BaytesFilters.AbstractParticles","page":"Home","title":"BaytesFilters.AbstractParticles","text":"abstract type AbstractParticles\n\nSuper type for various particle referencing techniques.\n\n\n\n\n\n","category":"type"},{"location":"#BaytesFilters.Ancestral","page":"Home","title":"BaytesFilters.Ancestral","text":"Referencing struct: last element in particle history weighted via reference index\n\n\n\n\n\n","category":"type"},{"location":"#BaytesFilters.Conditional","page":"Home","title":"BaytesFilters.Conditional","text":"Referencing struct: last element in particle history always reference index\n\n\n\n\n\n","category":"type"},{"location":"#BaytesFilters.InitialTrajectory","page":"Home","title":"BaytesFilters.InitialTrajectory","text":"struct InitialTrajectory{K<:ParticleKernel, C<:ParticleReferencing, M<:ParticleFilterMemory}\n\nContains information to obtain reference trajectory for sampling process.\n\nFields\n\nkernel::ParticleKernel\nKernel for particle propagation\nreferencing::ParticleReferencing\nReferencing method\nmemory::ParticleFilterMemory\nMemory for observed and latent data in PF.\n\n\n\n\n\n","category":"type"},{"location":"#BaytesFilters.Marginal","page":"Home","title":"BaytesFilters.Marginal","text":"Referencing struct: no updating via reference index\n\n\n\n\n\n","category":"type"},{"location":"#BaytesFilters.Markov","page":"Home","title":"BaytesFilters.Markov","text":"struct Markov{A, B, C} <: ParticleKernel\n\nMarkov Kernel for particle propagation.\n\nFields\n\ninitial::Any\nInitial distribution, function of iter only.\ntransition::Any\nTransition distribution, function of full particle trajectory and current iteration count.\nevidence::Any\nData distribution to weight particles. Function of full data, particle trajectory and current iteration count.\n\n\n\n\n\n","category":"type"},{"location":"#BaytesFilters.ParticleBuffer","page":"Home","title":"BaytesFilters.ParticleBuffer","text":"struct ParticleBuffer{A, I, P}\n\nContains temporary buffer values to avoid allocations during particle propagation.\n\nFields\n\nparameter::BaytesCore.ParameterBuffer{A, I} where {A, I}\nContains buffer values for particles and ancestor for one iteration.\nproposal::Vector\nProposal trajectory and predicted latent variable.\nprediction::Vector\nPredicted latent and oberved data\nresampled::Vector{Bool}\nStores boolean if resampled at each iteration.\nℓobjectiveᵥ::Vector{Float64}\nStores incremental log target estimates for each iteration.\n\n\n\n\n\n","category":"type"},{"location":"#BaytesFilters.ParticleFilter","page":"Home","title":"BaytesFilters.ParticleFilter","text":"struct ParticleFilter{A<:Particles, B<:ParticleFilterTune} <: BaytesCore.AbstractAlgorithm\n\nContains particles, transition kernel and all other relevant tuning information for particle propagation.\n\nFields\n\nparticles::Particles\nParticle values and kernel.\ntune::ParticleFilterTune\nTuning configuration for particles.\n\n\n\n\n\n","category":"type"},{"location":"#BaytesFilters.ParticleFilterConfig","page":"Home","title":"BaytesFilters.ParticleFilterConfig","text":"mutable struct ParticleFilterConfig{A, B, C, D}\n\nHolds information for array structure of data and reference particle.\n\nFields\n\ndata::BaytesCore.ArrayConfig{A, B} where {A, B}\nData size and sorting information.\nparticle::BaytesCore.ArrayConfig{C, D} where {C, D}\nParticle size and sorting information.\n\n\n\n\n\n","category":"type"},{"location":"#BaytesFilters.ParticleFilterConstructor","page":"Home","title":"BaytesFilters.ParticleFilterConstructor","text":"Callable struct to make initializing ParticleFilter sampler easier in sampling library.\n\nExamples\n\n\n\n\n\n\n\n","category":"type"},{"location":"#BaytesFilters.ParticleFilterDefault","page":"Home","title":"BaytesFilters.ParticleFilterDefault","text":"struct ParticleFilterDefault{M<:Union{Nothing, ParticleFilterMemory}, A<:ParameterWeighting, B<:ResamplingMethod, C<:ParticleReferencing, T<:Integer, I<:AbstractInitialization, U<:UpdateBool}\n\nDefault arguments for Particle Filter constructor.\n\nFields\n\nmemory::Union{Nothing, ParticleFilterMemory}\nMemory for particles and data.\nweighting::ParameterWeighting\nWeighting Methods for particles.\nresampling::ResamplingMethod\nResampling methods for particle trajectories.\nreferencing::ParticleReferencing\nReferencing type for last particle at each iteration - either Conditional, Ancestral or Marginal Implementation.\ncoverage::Float64\nCoverage of Particles/datapoints.\nthreshold::Float64\nESS threshold for resampling particle trajectories.\nancestortype::Type{T} where T<:Integer\nType for ancestors indices\ninit::AbstractInitialization\nMethod to obtain initial Modelparameter, see 'AbstractInitialization'.\ngenerated::UpdateBool\nBoolean if generate(_rng, objective) for corresponding model is stored in PF Diagnostics.\n\n\n\n\n\n","category":"type"},{"location":"#BaytesFilters.ParticleFilterDiagnostics","page":"Home","title":"BaytesFilters.ParticleFilterDiagnostics","text":"Contains information about log-likelihood, expected sample size and proposal trajectory.\n\nExamples\n\n\n\n\n\n\n\n","category":"type"},{"location":"#BaytesFilters.ParticleFilterMemory","page":"Home","title":"BaytesFilters.ParticleFilterMemory","text":"struct ParticleFilterMemory\n\nContains maximum memory for latent and observed data. Relevant for number of propagation steps from initial particle distribution and for first time particles are weighted with data point.\n\nFields\n\nlatent::Int64\nLatent variable memory. Number of times that initial particle distribution is used.\ndata::Int64\nObserved data memory. First particle weighting taken at this point.\ninitial::Int64\nNumber of times when sampled from initial distribution. Per default equal to latent.\n\n\n\n\n\n","category":"type"},{"location":"#BaytesFilters.ParticleFilterTune","page":"Home","title":"BaytesFilters.ParticleFilterTune","text":"struct ParticleFilterTune{T<:ModelWrappers.Tagged, A<:ParameterWeighting, B<:ResamplingMethod, C<:ParticleReferencing, D, E, F, G, U<:UpdateBool} <: BaytesCore.AbstractTune\n\nHolds tuning information for Particle Filter.\n\nFields\n\ntagged::ModelWrappers.Tagged\nTagged Model parameter.\nweighting::ParameterWeighting\nWeighting Methods for particles.\nresampling::ResamplingMethod\nResampling methods for particle trajectories.\nreferencing::ParticleReferencing\nReferencing type for last particle at each iteration - either Conditional, Ancestral or Marginal Implementation.\nconfig::BaytesFilters.ParticleFilterConfig\nContains data and reference size and sorting.\nchains::BaytesCore.ChainsTune\nNumber of particle chains and tuning information.\nmemory::ParticleFilterMemory\nMemory for latent and observed data.\ngenerated::UpdateBool\nBoolean if generated quantities should be generated while sampling\niter::BaytesCore.Iterator\nCurrent iteration.\n\n\n\n\n\n","category":"type"},{"location":"#BaytesFilters.ParticleKernel","page":"Home","title":"BaytesFilters.ParticleKernel","text":"abstract type ParticleKernel <: BaytesCore.AbstractKernel\n\nAvailable Particle Filter Kernels.\n\n\n\n\n\n","category":"type"},{"location":"#BaytesFilters.ParticleReferencing","page":"Home","title":"BaytesFilters.ParticleReferencing","text":"abstract type ParticleReferencing\n\nSuper type for various particle referencing techniques.\n\n\n\n\n\n","category":"type"},{"location":"#BaytesFilters.Particles","page":"Home","title":"BaytesFilters.Particles","text":"mutable struct Particles{R, B, I<:Integer} <: AbstractParticles\n\nContains Particle container for propagation.\n\nFields\n\nval::ElasticArrays.ElasticArray{B, 2, 1, Vector{B}} where B\nParticle trajectories, for a discussion about possible shapes for the trajectories.\nancestor::ElasticArrays.ElasticArray{I, 2, 1, Vector{I}} where I<:Integer\nSaved ancestors of resampling step in pf\nweights::BaytesCore.ParameterWeights\nParticle weights.\nbuffer::ParticleBuffer{B, I, R} where {R, B, I<:Integer}\nNecessary buffer values for resampling particles.\nℓobjective::BaytesCore.Accumulator\nLog likelihood estimate.\n\n\n\n\n\n","category":"type"},{"location":"#BaytesFilters.SemiMarkov","page":"Home","title":"BaytesFilters.SemiMarkov","text":"struct SemiMarkov{A<:SemiMarkovInitiation, B<:SemiMarkovTransition, C} <: ParticleKernel\n\nSemi-Markov Kernel for particle propagation.\n\nFields\n\ninitial::SemiMarkovInitiation\nInitial distribution, function of iter only.\ntransition::SemiMarkovTransition\nTransition distribution, function of full particle trajectory and current iteration count.\nevidence::Any\nData distribution to weight particles. Function of full data, particle trajectory and current iteration count.\n\n\n\n\n\n","category":"type"},{"location":"#BaytesFilters.SemiMarkovInitiation","page":"Home","title":"BaytesFilters.SemiMarkovInitiation","text":"struct SemiMarkovInitiation{A, B}\n\nInitial distribution for state and duration of semi-Markov kernel.\n\nFields\n\nstate::Any\nInitial distribution of state variable.\nduration::Any\nInitial distribution of duration variable.\n\n\n\n\n\n","category":"type"},{"location":"#BaytesFilters.SemiMarkovTransition","page":"Home","title":"BaytesFilters.SemiMarkovTransition","text":"struct SemiMarkovTransition{A, B}\n\nTransition distributions for state and duration of semi-Markov kernel.\n\nFields\n\nstate::Any\nTransition distribution of state variable.\nduration::Any\nTransition distribution of duration variable.\n\n\n\n\n\n","category":"type"},{"location":"#Base.resize!-Union{Tuple{T}, Tuple{Particles, AbstractArray{T}, Integer, Integer}} where T","page":"Home","title":"Base.resize!","text":"resize!(particles, reference, Nparticles, Ndata)\n\n\nResize particles struct with new Nparticles and Ndata size.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#BaytesCore.generate-Tuple{Random.AbstractRNG, ParticleFilter, ModelWrappers.Objective}","page":"Home","title":"BaytesCore.generate","text":"generate(_rng, algorithm, objective)\n\n\nGenerate statistics for algorithm given model parameter and data.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#BaytesCore.generate_showvalues-Tuple{D} where D<:ParticleFilterDiagnostics","page":"Home","title":"BaytesCore.generate_showvalues","text":"generate_showvalues(diagnostics)\n\n\nShow relevant diagnostic results.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#BaytesCore.infer-Union{Tuple{D}, Tuple{Random.AbstractRNG, ParticleKernel, ParticleFilterTune, ModelWrappers.ModelWrapper, D}} where D","page":"Home","title":"BaytesCore.infer","text":"infer(_rng, kernel, tune, model, data)\n\n\nInfer type of predictions of kernel.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#BaytesCore.infer-Union{Tuple{D}, Tuple{Random.AbstractRNG, Type{BaytesCore.AbstractDiagnostics}, ParticleFilter, ModelWrappers.ModelWrapper, D}} where D","page":"Home","title":"BaytesCore.infer","text":"infer(_rng, diagnostics, pf, model, data)\n\n\nInfer ParticleFilter diagnostics type.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#BaytesCore.propagate!-Union{Tuple{P}, Tuple{Random.AbstractRNG, ParticleKernel, Particles, ParticleFilterTune, AbstractArray{P}, ModelWrappers.Objective}} where P","page":"Home","title":"BaytesCore.propagate!","text":"propagate!(_rng, kernel, particles, tune, reference, objective)\n\n\nPropagate particles forward.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#BaytesCore.propagate!-Union{Tuple{T}, Tuple{D}, Tuple{Random.AbstractRNG, ParticleFilter, ModelWrappers.ModelWrapper, D}, Tuple{Random.AbstractRNG, ParticleFilter, ModelWrappers.ModelWrapper, D, T}} where {D, T<:BaytesCore.ProposalTune}","page":"Home","title":"BaytesCore.propagate!","text":"propagate!(_rng, pf, model, data)\npropagate!(_rng, pf, model, data, proposaltune)\n\n\nPropagate particle filter forward with new data.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#BaytesCore.propagate-Union{Tuple{P}, Tuple{Any, ParticleKernel, ParticleFilterMemory, AbstractArray{P}}} where P","page":"Home","title":"BaytesCore.propagate","text":"propagate(_rng, kernel, memory, reference)\n\n\nPropagate forward a single trajectory given ParticleKernel.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#BaytesCore.propose!-Union{Tuple{T}, Tuple{D}, Tuple{Random.AbstractRNG, ParticleFilter, ModelWrappers.ModelWrapper, D}, Tuple{Random.AbstractRNG, ParticleFilter, ModelWrappers.ModelWrapper, D, T}} where {D, T<:BaytesCore.ProposalTune}","page":"Home","title":"BaytesCore.propose!","text":"Run particle filter and change reference trajectory with proposal trajectory.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#BaytesCore.propose-Union{Tuple{P}, Tuple{Random.AbstractRNG, ParticleKernel, ParticleFilter, ModelWrappers.Objective}, Tuple{Random.AbstractRNG, ParticleKernel, ParticleFilter, ModelWrappers.Objective, AbstractArray{P}}} where P","page":"Home","title":"BaytesCore.propose","text":"propose(_rng, kernel, pf, objective)\npropose(_rng, kernel, pf, objective, reference)\n\n\nRun particle filter and do not change reference trajectory.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#BaytesCore.resample!-Union{Tuple{P}, Tuple{Random.AbstractRNG, ParticleKernel, Particles, ParticleFilterTune, AbstractArray{P}}} where P","page":"Home","title":"BaytesCore.resample!","text":"resample!(_rng, kernel, particles, tune, reference)\n\n\nResample particle ancestors and shuffle current particles dependency.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#BaytesCore.results-Union{Tuple{P}, Tuple{T}, Tuple{AbstractVector{P}, ParticleFilter, Integer, Vector{T}}} where {T<:Real, P<:ParticleFilterDiagnostics}","page":"Home","title":"BaytesCore.results","text":"results(diagnosticsᵛ, pf, Ndigits, quantiles)\n\n\nReturn summary statistics for PF diagnostics.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#BaytesCore.update!-Union{Tuple{T}, Tuple{Particles, UpdateFalse, UpdateFalse, AbstractArray{T}, Integer, Integer}} where T","page":"Home","title":"BaytesCore.update!","text":"update!(particles, ParticleUpdate, DataUpdate, reference, Nparticles, Ndata)\n\n\nUpdate particles struct with new number of particles. Note that kernel and log likelihood are adjusted in another step.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#BaytesFilters._checkmemory-Tuple{Function}","page":"Home","title":"BaytesFilters._checkmemory","text":"_checkmemory(f; maxiter)\n\n\nReturn first iteration at which function is executable.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#BaytesFilters._guessmemory-Union{Tuple{T}, Tuple{Random.AbstractRNG, ParticleKernel, AbstractArray{T}}} where T","page":"Home","title":"BaytesFilters._guessmemory","text":"_guessmemory(_rng, kernel, reference)\n\n\nGuess memory of ParticleKernel.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#BaytesFilters._try-Tuple{Function, Vararg{Any}}","page":"Home","title":"BaytesFilters._try","text":"_try(f, input)\n\n\nCheck if function is exectuable.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#BaytesFilters.ancestors!-Union{Tuple{F}, Tuple{R}, Tuple{Random.AbstractRNG, ParticleReferencing, R, AbstractArray, Integer, Integer, AbstractVector{F}}} where {R<:ResamplingMethod, F<:AbstractFloat}","page":"Home","title":"BaytesFilters.ancestors!","text":"ancestors!(_rng, referencing, resampling, ancestor, iter, Nparticles, weights)\n\n\nResampling function for particles, dispatched on ParticleReferencing subtype.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#BaytesFilters.ancestralweight!-Union{Tuple{P}, Tuple{ParticleKernel, BaytesCore.ParameterWeights, Union{AbstractArray{P}, P}, AbstractArray, Integer}} where P","page":"Home","title":"BaytesFilters.ancestralweight!","text":"ancestralweight!(kernel, weights, valₜ, val, iter)\n\n\nInplace calculate ancestor weights.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#BaytesFilters.assign!-Union{Tuple{T}, Tuple{P}, Tuple{AbstractVector{P}, AbstractVector{T}}} where {P, T}","page":"Home","title":"BaytesFilters.assign!","text":"assign!(buffer, trajectory)\n\n\nAssign 'trajectory' elements to 'buffer' up until index 'iter'.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#BaytesFilters.estimate_Nparticles-Tuple{Random.AbstractRNG, ParticleFilterConstructor, ModelWrappers.Objective, AbstractFloat}","page":"Home","title":"BaytesFilters.estimate_Nparticles","text":"estimate_Nparticles(_rng, pf, objective, variance; Nchains, margin, itermax, mincoverage, printoutput)\n\n\nFunction that checks for number of particles to achieve target variance of log target estimate.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#BaytesFilters.get_reference!-Union{Tuple{P}, Tuple{F}, Tuple{I}, Tuple{Random.AbstractRNG, ParticleKernel, Marginal, AbstractVector{I}, F, ParticleFilterTune, Union{AbstractArray{P}, P}}} where {I<:Integer, F<:AbstractParticles, P}","page":"Home","title":"BaytesFilters.get_reference!","text":"get_reference!(_rng, kernel, referencing, ancestor, particles, tune, reference)\n\n\nCompute new reference index from current reference index.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#BaytesFilters.get_reference-Union{Tuple{D}, Tuple{Any, ModelWrappers.ModelWrapper, D, ModelWrappers.Tagged}} where D","page":"Home","title":"BaytesFilters.get_reference","text":"get_reference(referencing, model, data, tagged)\n\n\nAssign reference, depending on whether reference is tracked in Model.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#BaytesFilters.infer_generated-Union{Tuple{D}, Tuple{Random.AbstractRNG, ParticleFilter, ModelWrappers.ModelWrapper, D}} where D","page":"Home","title":"BaytesFilters.infer_generated","text":"infer_generated(_rng, pf, model, data)\n\n\nInfer type of generated quantities of PF sampler.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#BaytesFilters.initial!-Union{Tuple{P}, Tuple{Random.AbstractRNG, ParticleKernel, AbstractMatrix{P}, Integer}} where P","page":"Home","title":"BaytesFilters.initial!","text":"Inplace initiate particle given kernel.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#BaytesFilters.initial!-Union{Tuple{P}, Tuple{Random.AbstractRNG, ParticleKernel, Particles, ParticleFilterTune, AbstractArray{P}, ModelWrappers.Objective}} where P","page":"Home","title":"BaytesFilters.initial!","text":"Initialize particles.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#BaytesFilters.initial-Tuple{Random.AbstractRNG, ParticleKernel}","page":"Home","title":"BaytesFilters.initial","text":"initial(_rng, kernel)\n\n\nInitiate particle with given initial distribution.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#BaytesFilters.sample_ancestor-Union{Tuple{P}, Tuple{Random.AbstractRNG, ParticleKernel, BaytesCore.ParameterWeights, Union{AbstractArray{P}, P}, AbstractArray, Integer}} where P","page":"Home","title":"BaytesFilters.sample_ancestor","text":"sample_ancestor(_rng, kernel, weights, valₜ, val, iter)\n\n\nSample ancestor reference for particle history.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#BaytesFilters.set_reference!-Union{Tuple{P}, Tuple{Marginal, AbstractVector{P}, AbstractVector{P}, Integer}} where P","page":"Home","title":"BaytesFilters.set_reference!","text":"set_reference!(referencing, val, reference, iter)\n\n\nSet last particle at current iteration.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#BaytesFilters.transition!-Union{Tuple{P}, Tuple{Random.AbstractRNG, ParticleKernel, AbstractMatrix{P}, Integer}} where P","page":"Home","title":"BaytesFilters.transition!","text":"Inplace propagate particle forward given current particle and kernel.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#BaytesFilters.transition-Union{Tuple{P}, Tuple{Random.AbstractRNG, ParticleKernel, AbstractMatrix{P}, Integer}} where P","page":"Home","title":"BaytesFilters.transition","text":"transition(_rng, kernel, val, iter)\n\n\nPropagate particle forward given current particle.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#BaytesFilters.ℓevidence!-Union{Tuple{P}, Tuple{D}, Tuple{F}, Tuple{ParticleKernel, Vector{F}, D, AbstractMatrix{P}, Integer}} where {F<:AbstractFloat, D, P}","page":"Home","title":"BaytesFilters.ℓevidence!","text":"ℓevidence!(kernel, ℓcontainer, dataₜ, val, iter)\n\n\nInplace logpdf evaluation of data given current particle trajectory.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#BaytesFilters.ℓevidence-Union{Tuple{P}, Tuple{D}, Tuple{ParticleKernel, D, AbstractVector{P}, Integer}} where {D, P}","page":"Home","title":"BaytesFilters.ℓevidence","text":"ℓevidence(kernel, dataₜ, val, iter)\n\n\nEvaluate data given particle trajectory.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#BaytesFilters.ℓtransition!-Union{Tuple{P}, Tuple{F}, Tuple{ParticleKernel, Vector{F}, Union{AbstractArray{P}, P}, AbstractMatrix{P}, Integer}} where {F<:AbstractFloat, P}","page":"Home","title":"BaytesFilters.ℓtransition!","text":"ℓtransition!(kernel, ℓcontainer, valₜ, val, iter)\n\n\nInplace calculate log transtion probability from current particle given trajectory.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#BaytesFilters.ℓtransition-Union{Tuple{P}, Tuple{ParticleKernel, Union{AbstractVector{P}, P}, AbstractMatrix{P}, Integer}} where P","page":"Home","title":"BaytesFilters.ℓtransition","text":"ℓtransition(kernel, valₜ, val, iter)\n\n\nCalculate log transtion probability from particle given particle history.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"}]
}
