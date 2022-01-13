############################################################################################
"""
$(TYPEDEF)
Initial distribution for state and duration of semi-Markov kernel.

# Fields
$(TYPEDFIELDS)
"""
struct SemiMarkovInitiation{A,B}
    "Initial distribution of state variable."
    state::A
    "Initial distribution of duration variable."
    duration::B
    function SemiMarkovInitiation(state::A, duration::B) where {A,B}
        return new{A,B}(state, duration)
    end
end

############################################################################################
"""
$(TYPEDEF)
Transition distributions for state and duration of semi-Markov kernel.

# Fields
$(TYPEDFIELDS)
"""
struct SemiMarkovTransition{A,B}
    "Transition distribution of state variable."
    state::A
    "Transition distribution of duration variable."
    duration::B
    function SemiMarkovTransition(state::A, duration::B) where {A,B}
        return new{A,B}(state, duration)
    end
end

############################################################################################
"""
$(TYPEDEF)
Semi-Markov Kernel for particle propagation.

# Fields
$(TYPEDFIELDS)
"""
struct SemiMarkov{A<:SemiMarkovInitiation,B<:SemiMarkovTransition,C} <: ParticleKernel
    "Initial distribution, function of iter only."
    initial::A
    "Transition distribution, function of full particle trajectory and current iteration count."
    transition::B
    "Data distribution to weight particles. Function of full data, particle trajectory and current iteration count."
    evidence::C
    function SemiMarkov(
        initial::A, transition::B, evidence::C
    ) where {A<:SemiMarkovInitiation,B<:SemiMarkovTransition,C}
        return new{A,B,C}(initial, transition, evidence)
    end
end

############################################################################################
function initial(_rng::Random.AbstractRNG, kernel::SemiMarkov)
    s = rand(_rng, kernel.initial.state)
    d = rand(_rng, kernel.initial.duration(s))
    return (s, d)
end

############################################################################################
function transition(
    _rng::Random.AbstractRNG, kernel::SemiMarkov, val::AbstractArray{P}, iter::Integer
) where {P}
    if val[iter - 1][2] > 0
        return (val[iter - 1][1], val[iter - 1][2] - 1)
    else
        #!NOTE: Inconsistent that s depends on whole Vector, but d only on current s - but no other way as of now
        sₜ = rand(_rng, kernel.transition.state(val, iter))
        dₜ = rand(_rng, kernel.transition.duration(sₜ, iter))
        return (sₜ, dₜ)
    end
end
############################################################################################
function ℓtransition(
    valₜ::Union{P,AbstractArray{P}},
    kernel::SemiMarkov,
    val::AbstractArray{P},
    iter::Integer,
) where {P}
    sₜ₋₁, dₜ₋₁ = val[iter - 1]
    sₜ, dₜ = valₜ
    #!NOTE: If duration at t-1 is 0, and particles states are not the same from t-1 to t
    if dₜ₋₁ == 0 && sₜ₋₁ != sₜ
        ℓπ = logpdf(kernel.transition.state(val, iter), sₜ) #current state given past state
        ℓπ += logpdf(kernel.transition.duration(sₜ, iter), dₜ) #duration given current state
        return ℓπ
    elseif (dₜ₋₁ - 1) == (dₜ) && sₜ₋₁ == sₜ
        return 0.0 #log(1.0) = 0
    else
        return -Inf
    end
end

############################################################################################
export SemiMarkovInitiation, SemiMarkovTransition, SemiMarkov
