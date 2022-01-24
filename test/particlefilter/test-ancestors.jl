############################################################################################
function check_correctness(val::AbstractVector{T}) where {T}
    ## Get relevant fields
    s = getfield.(val, 1)
    d = getfield.(val, 2)
    ## Initate container that holds time when state changes
    StateIter = Int64[]
    DurationIter = Int64[]
    ## Compute all state changes
    statechanges = [s[iter] - s[iter - 1] for iter in 2:length(s)]
    durationchanges = [d[iter] - d[iter - 1] for iter in 2:length(d)]
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
    changes = [StateIter[iter] == DurationIter[iter] for iter in eachindex(StateIter)]
    ## Return total changes - correct changes (should b 0)
    return length(StateIter) - sum(changes)
end
#Check if HSMM has impossible transitions
function check_correctness(kernel::SemiMarkov, val::AbstractMatrix{T}) where {T}
    Nparticles = size(val, 1)
    return sum([check_correctness(val[iter, :]) for iter in Base.OneTo(Nparticles)])
end

############################################################################################
# run PF and check if correct state trajectories
semimarkov_pf = ParticleFilter(_rng, semimarkov_objective)
_vals, _diag = propose(_rng, semimarkov_pf, semimarkov_objective)

@testset "PF - Ancestors - SemiMarkov" begin
    @test check_correctness(semimarkov_pf.particles.kernel, semimarkov_pf.particles.val) ==
          0
end
