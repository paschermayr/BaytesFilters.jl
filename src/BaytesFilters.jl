"Particle Filter module."
module BaytesFilters

############################################################################################
#Import modules

import BaytesCore:
    BaytesCore,
    update!,
    infer,
    results,
    init,
    init!,
    propose,
    propose!,
    propagate,
    propagate!,
    result!,
    get_result,
    generate_showvalues,
    ParameterWeighting,
    ResamplingMethod,
    resample!,
    weight!,
    shuffle!,
    shuffle_forward!,
    shuffle_backward!

using BaytesCore:
    BaytesCore,
    AbstractAlgorithm,
    AbstractTune,
    AbstractConfiguration,
    AbstractDiagnostics,
    AbstractKernel,
    AbstractConstructor,
    Updater,
    Iterator,
    Accumulator,
    UpdateBool,
    UpdateTrue,
    UpdateFalse,
    logsumexp,
    logaddexp,
    logmeanexp,
    issmaller,
    grab,
    ArrayConfig,
    to_NamedTuple,
    update,
    ParameterWeights,
    draw!,
    normalize!,
    computeESS,
    randcat,
    ChainsTune,
    ParameterBuffer,
    SampleDefault,
    to_Tuple

import ModelWrappers:
    ModelWrappers,
    predict,
    dynamics,
    AbstractInitialization,
    NoInitialization,
    PriorInitialization,
    OptimInitialization

using ModelWrappers:
    ModelWrappers, ModelWrapper, Tagged, Objective, DiffObjective, ℓObjectiveResult, generate

import Base: max, push!, resize!
import Random: Random, rand!

using DocStringExtensions:
    DocStringExtensions, TYPEDEF, TYPEDFIELDS, FIELDS, SIGNATURES, FUNCTIONNAME
using ArgCheck: ArgCheck, @argcheck
using UnPack: UnPack, @unpack, @pack!

using Random: Random, AbstractRNG, GLOBAL_RNG
using Distributions: Distributions, logpdf, Categorical
using ElasticArrays: ElasticArrays, ElasticMatrix
using Statistics: Statistics, mean, var

############################################################################################
# Import sub-folder
include("ParticleFilter/ParticleFilter.jl")
include("KalmanFilter/KalmanFilter.jl")

############################################################################################
# Export
export
    # BaytesCore
    UpdateBool,
    UpdateTrue,
    UpdateFalse,
    propose,
    propose!,
    propagate,
    propagate!,
    resample!,
    ParameterWeighting,
    ResamplingMethod,
    shuffle!,
    shuffle_forward!,
    shuffle_backward!,
    SampleDefault

    # ModelWrappers
    predict,
    dynamics,
    AbstractInitialization,
    NoInitialization,
    PriorInitialization,
    OptimInitialization


end
