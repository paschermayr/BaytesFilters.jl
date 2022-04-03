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
"Normalize Vector inplace and return normalizing constant"
@inline function normalize!(vec::AbstractVector)
    norm = sum(vec)
    vec ./= norm
    return norm
end
"Faster max function for Viterbi algorithm in HMM. Not exported."
function vec_maximum(vec::AbstractVector)
    m = vec[1]
    @inbounds for i = Base.OneTo(length(vec))
        if vec[i] > m
            m = vec[i]
        end
    end
    return m
end

############################################################################################
"Compute Stationary distribution for given transition matrix"
function get_stationary!(Transition::AbstractMatrix{T}) where T<:Real
    # From: https://github.com/QuantEcon/QuantEcon.jl/blob/f454d4dfbaf52f550ddd52eff52471e4b8fddb9d/src/markov/mc_tools.jl
    # Grassmann-Taksar-Heyman (GTH) algorithm (Grassmann, Taksar, and Heyman 1985)
    n = size(Transition, 1)
    x = zeros(T, n)

    @inbounds for k in 1:n-1
        scale = sum(Transition[k, k+1:n])
        if scale <= zero(T)
            # There is one (and only one) recurrent class contained in
            # {1, ..., k};
            # compute the solution associated with that recurrent class.
            n = k
            break
        end
        Transition[k+1:n, k] /= scale

        for j in k+1:n, i in k+1:n
            Transition[i, j] += Transition[i, k] * Transition[k, j]
        end
    end

    # backsubstitution
    x[n] = 1
    @inbounds for k in n-1:-1:1, i in k+1:n
        x[k] += x[i] * Transition[i, k]
    end

    # normalisation
    x /= sum(x)

    return x
end
get_stationary(Transition::AbstractMatrix{T}) where {T<:Real} = get_stationary!(copy(Transition))

############################################################################################
"Check if ancestors are all in correct order - Not that first column can have different order as ancestors not resampled at 0."
function check_ancestors(ancestors::AbstractMatrix{I}) where {I<:Integer}
    for Ncol in 2:size(ancestors, 2)
        for Nrow in Base.OneTo(size(ancestors, 1))
            if ancestors[Nrow, Ncol] != Nrow
                return false
            end
        end
    end
    return true
end

############################################################################################
include("hmm.jl")
include("hmmMV.jl")
include("hsmm.jl")
include("HigherOrderMarkov.jl")

############################################################################################
resamplemethods = [Systematic(), Residual(), Stratified(), BaytesFilters.Multinomial()]
referencemethods = [Conditional(), Ancestral(), Marginal()]
objectives = [deepcopy(markov_objective), deepcopy(markov_MV_objective), deepcopy(semimarkov_objective)]
generating = [UpdateFalse(), UpdateTrue()]
