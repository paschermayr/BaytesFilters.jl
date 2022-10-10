############################################################################################
"""
$(TYPEDEF)
Markov Kernel for particle propagation.

# Fields
$(TYPEDFIELDS)
"""
struct Markov{A,B,C} <: ParticleKernel
    "Initial distribution, function of iter only."
    initial::A
    "Transition distribution, function of full particle trajectory and current iteration count."
    transition::B
    "Data distribution to weight particles. Function of full data, particle trajectory and current iteration count."
    evidence::C
    function Markov(initial::A, transition::B, evidence::C) where {A,B,C}
        return new{A,B,C}(initial, transition, evidence)
    end
end

############################################################################################
function initial(_rng::Random.AbstractRNG, kernel::Markov)
    return rand(_rng, kernel.initial)
end

############################################################################################
function transition(
    _rng::Random.AbstractRNG, kernel::Markov, val::AbstractArray{P}, iter::Integer
) where {P}
    return rand(_rng, kernel.transition(val, iter))
end

############################################################################################
function ℓtransition(
    kernel::Markov, valₜ::Union{P,AbstractArray{P}}, val::AbstractArray{P}, iter::Integer
) where {P}
    return logpdf(kernel.transition(val, iter), valₜ)
end

############################################################################################
export Markov
