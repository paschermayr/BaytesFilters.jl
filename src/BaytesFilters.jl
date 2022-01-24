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
    get_result,
    get_tagged,
    result!,
    get_loglik,
    get_prediction,
    get_phase,
    get_iteration,
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
    islarger,
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
    ParameterBuffer

import ModelWrappers: ModelWrappers, predict
using ModelWrappers:
    ModelWrappers, ModelWrapper, Tagged, Objective, DiffObjective, ℓObjectiveResult

import Base: max, push!, resize!
import Random: Random, rand!
#import Distributions: Distributions, Multinomial

using DocStringExtensions:
    DocStringExtensions, TYPEDEF, TYPEDFIELDS, FIELDS, SIGNATURES, FUNCTIONNAME
using ArgCheck: ArgCheck, @argcheck
using UnPack: UnPack, @unpack, @pack!

using Random: Random, AbstractRNG, GLOBAL_RNG
using Distributions: Distributions, logpdf, Categorical
using ElasticArrays: ElasticArrays, ElasticMatrix
using Statistics: Statistics, mean

############################################################################################
# Define functions to be dispatched so Models can be used via BaytesFilters
function dynamics end

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

    # BaytesFilters
    dynamics

end
