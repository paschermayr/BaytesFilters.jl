# BaytesFilters

<!---
![logo](docs/src/assets/logo.svg)
[![CI](xxx)](xxx)
[![arXiv article](xxx)](xxx)

-->
[![Documentation, Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://paschermayr.github.io/BaytesFilters.jl/)
[![Build Status](https://github.com/paschermayr/BaytesFilters.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/paschermayr/BaytesFilters.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/paschermayr/BaytesFilters.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/paschermayr/BaytesFilters.jl)
[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://github.com/SciML/ColPrac)

BaytesFilters.jl is a library to perform particle filtering for one parameter in a `ModelWrapper` struct, see [ModelWrappers.jl](https://github.com/paschermayr/ModelWrappers.jl).

## Introduction
Let us start with creating a univariate normal Mixture model with two states via [ModelWrappers.jl](https://github.com/paschermayr/ModelWrappers.jl):
```julia
using ModelWrappers, BaytesFilters
using Distributions, Random, UnPack
_rng = Random.MersenneTwister(1)
N = 10^3
# Parameter
μ = [-2., 2.]
σ = [1., 1.]
p = [.05, .95]
# Latent data
latent = rand(_rng, Categorical(p), N)
data = [rand(_rng, Normal(μ[iter], σ[iter])) for iter in latent]

# Create ModelWrapper struct, assuming we do not know latent
latent_init = rand(_rng, Categorical(p), N)
myparameter = (;
    μ = Param(μ, [Normal(-2., 5), Normal(2., 5)]),
    σ = Param(σ, [Gamma(2.,2.), Gamma(2.,2.)]),
    p = Param(p, Dirichlet(2, 2)),
    latent = Param(latent_init, [Categorical(p) for _ in Base.OneTo(N)]),
)
mymodel = ModelWrapper(myparameter)
myobjective = Objective(mymodel, data)
```

BaytesFilters.jl let's you target one parameter of your model with equal dimension of the provided data. In order to create a `ParticleFilter` struct, you first have to assign the model dynamics by dispatching your model on the following function:

```julia
function BaytesFilters.dynamics(objective::Objective{<:ModelWrapper{BaseModel}})
    @unpack model, data = objective
    @unpack μ, σ, p = model.val

    initial_latent = Categorical(p)
    transition_latent(particles, iter) = initial_latent
    transition_data(particles, iter) = Normal(μ[particles[iter]], σ[particles[iter]])

    return Markov(initial_latent, transition_latent, transition_data)
end
dynamics(myobjective)
```

The model dynamics consist of initial and transition latent dynamics, as well as data dynamics. Note that the return struct `Markov` is very flexible, and can be of higher order as well. Some more remarks:
1. `transition_latent(particles, iter)` is a function of a full particle trajectory `particles` and the current iteration `iter`. In the mixture case, there is no Markov structure in the latent process, but
you can also define higher order Markov dependencies, i.e.,
```julia
transition_latent(particles, iter) = Normal(mean(@view(particles[iter-5:iter-1])), 1)
```
2. `transition_data(particles, iter)` is a function of a full particle trajectory `particles` and the current iteration `iter`.
3. Note that at each iteration, `transition_latent` and `transition_data` have access to the underlying data, so it is easy to adjust the filter for dependency. For instance, an auto-regressive data structure could look like:
```julia
transition_data(particles, iter) = Normal(mean(@view(particles[iter-5:iter-1])), mean(@view(data[iter-2:iter-1])))
```

## Estimating particle trajectories
Let us now create a `ParticleFilter`, and estimate the latent trajectory given all other model parameter. Note that we can only target one parameter via a `ParticleFilter`, so we slightly adjust our objective to
```julia
myobjective2 = Objective(mymodel, data, :latent)
myfilter = ParticleFilter(myobjective2)
```
Proposal steps works exactly as in BaytesMCMC.jl, you can either use `propose(_rng, algorithm, objective)` or `propose!(_rng, algorithm, mode, data)` depending on the use case. You can check out the BaytesMCMC.jl if you need further clarification on the difference of these two functions. Let us run the filter to get a new estimate for the latent data sequence:
```julia
_val, _diagnostics = propose(_rng, myfilter, myobjective2)

using Plots
plot(latent, label = "true")
plot!(_val.latent, label = "filter estimate")
```
Very close, nice! Another interesting output we get is an estimate from the model evidence. Luckily, for a discrete mixture model we can also evaluate the evidence analytically relatively easy. Let us check how good the estimate is against the analtyical solution:
```julia
# Define analytical form for mixture likelihood
using LogExpFunctions
function (objective::Objective{<:ModelWrapper{BaseModel}})(θ::NamedTuple)
    @unpack model, data = objective
    @unpack μ, σ, p = model.val
    Nstates = length(μ)
    dynamics_latent = Categorical(p)
    dynamics_data = [Normal(μ[iter], σ[iter]) for iter in Base.OneTo(Nstates)]
    ll = 0.0
    for time in Base.OneTo(length(data))
        ll += logsumexp(logpdf(dynamics_latent, iter) + logpdf(dynamics_data[iter], data[time]) for iter in Base.OneTo(Nstates))
    end
    return ll
end

# Compare 1 Filter run with analytical likelihood
_diagnostics.ℓℒ #~-1633.01
myobjective2(_val) #-1633.05
```
There are more output statistics stored in the diagnostics struct, such as the number of resampling steps or a one-step-ahead prediction for the next latent and observed data point. If you want to do further inference on the likelihood estimate, you might want to run the filter several times to check the variance of this estimate.

## Configuration

You can configure the `ParticleFilter` with the `ParticleFilterDefault` struct.
```julia
pfdefault = ParticleFilterDefault(;
    weighting=Bootstrap(), #Weighting Methods for particles
    resampling=Systematic(), #Resampling methods for particle trajectories
    referencing=Marginal(), #Referencing type for last particle at each iteration - either Conditional, Ancestral or Marginal Implementation.
    coverage=0.50, #Coverage of Nparticles/Ndata.
    threshold=0.75, #ESS threshold for resampling particle trajectories.
)
myfilter = ParticleFilter(myobjective2; default = pfdefault )
```

There are a variety of methods for `ParticleFilterDefault` fields. For now, you have to check all options in the code base if you want to adjust the default arguments.

## Scaling

The particle filter implementation scales linearly in both the number of particles as well as the number of data points. We can verify this by comparing the time spent in the proposal steps for different coverage ratios:
```julia
using BenchmarkTools
pfdefault1 = ParticleFilterDefault(; coverage=0.5)
pfdefault2 = ParticleFilterDefault(; coverage=1.0)
myfilter1 = ParticleFilter(myobjective2; default=pfdefault1)
myfilter2 = ParticleFilter(myobjective2; default=pfdefault2)
@btime propose($_rng, $myfilter1, $myobjective2) #12.522 ms (8 allocations: 16.20 KiB)
@btime propose($_rng, $myfilter2, $myobjective2) #25.992 ms (8 allocations: 16.20 KiB)
```
Moreover, if your `dynamics(objective)` function is efficiently implemented, there should be only very few allocations during the proposal step.

## Going Forward

This package is still highly experimental - suggestions and comments are always welcome!

<!---
# Citing Baytes.jl

If you use Baytes.jl for your own research, please consider citing the following publication: ...
-->
