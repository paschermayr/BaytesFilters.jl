
############################################################################################
"""
$(SIGNATURES)
Sort val according to index order without allocations using buffer as temporary storage. Used in propagate!(pf) when all but the last ancestor are already sorted.

# Examples
```julia
```

"""
function shuffle!(
    val::AbstractMatrix{P}, buffer::AbstractArray{P}, ancestor::AbstractVector{I}
) where {P,I<:Integer}
    ## Assign temporary variables
    Nparticles = size(ancestor, 1)
    Ndata = size(val, 2)
    ## Reshuffle val - can be done inplace via buffer
    @inbounds for t in Ndata:-1:1
        for idx in Base.OneTo(Nparticles)
            #!NOTE: Ancestor is only a buffer, so we do not need change order of it after shuffling.
            buffer[idx] = val[ancestor[idx], t]
        end
        for idx in Base.OneTo(Nparticles)
            val[idx, t] = buffer[idx]
        end
    end
    return nothing
end

############################################################################################
"""
$(SIGNATURES)
Uses 'ancestor' as Matrix instead of as Vector of Vector. Faster approach, but code for other functions more difficult and contrieved. Kept in code-base for now.

# Examples
```julia
```

"""
function shuffle_forward!(
    val::AbstractMatrix{P},
    buffer_val::AbstractArray{P},
    ancestor::AbstractMatrix{I},
    buffer_ancestor::AbstractVector{I},
    lookback::Integer,
    iterₘₐₓ::Integer,
) where {P,I<:Integer}
    ## Assign temporary variables
    Nparticles = size(buffer_val, 1)
    ## 1 Resample current particles
    @inbounds for idx in Base.OneTo(Nparticles)
        #!NOTE: buffer_ancestor assigned for higher order memory case below
        buffer_ancestor[idx] = ancestor[idx, iterₘₐₓ]
        buffer_val[idx] = val[buffer_ancestor[idx], iterₘₐₓ]
    end
    ## Assign resampled particle
    @inbounds for idx in Base.OneTo(Nparticles)
        val[idx, iterₘₐₓ] = buffer_val[idx]
    end
    ## 2 If higher order memory, resample relevant previous particles, and adjust ancestors.
    if lookback > 1
        ## Assign last possible time point for lookback,
        #!NOTE: i.e. lookback == 2 means current particles AND particles at iterₘₐₓ-1 are resampled.
        iter₀ = iterₘₐₓ - lookback + 1
        @inbounds for t in (iterₘₐₓ - 1):-1:iter₀
            ## Assign buffer for resampled particle from ancestor at iterₘₐₓ
            for idx in Base.OneTo(Nparticles)
                buffer_val[idx] = val[buffer_ancestor[idx], t]
            end
            ## Assign resampled particle and set current ancestor as non-resampled
            for idx in Base.OneTo(Nparticles)
                ## All but the last ancestors are set back to default order 1,2,3,4,..
                ancestor[idx, t] = idx
                val[idx, t] = buffer_val[idx]
            end
        end
        ## last ancestor in lookback inherits the ancestors of the current iteration
        @inbounds for idx in Base.OneTo(Nparticles)
            ancestor[idx, iter₀] = ancestor[idx, iterₘₐₓ]
            #!NOTE: Set back ancestors at iterₘₐₓ to default order as particles before already resampled
            ancestor[idx, iterₘₐₓ] = idx
        end
    end
    return nothing
end

"""
$(SIGNATURES)
Uses 'ancestor' as Matrix instead of as Vector of Vector. Faster approach, but code for other functions more difficult and contrieved. Kept in code-base for now.

# Examples
```julia
```

"""
function shuffle_backward!(
    val::AbstractMatrix{P},
    buffer_val::AbstractArray{P},
    ancestor::AbstractMatrix{I},
    buffer_ancestor::AbstractVector{I},
) where {P,I<:Integer}
    ## Assign relevant looping variables
    Nparticles = size(buffer_val, 1)
    iter₀ = 1
    iterₘₐₓ = size(ancestor, 2)
    ## 1 Assign Ancestors theoretical iterₘₐₓ+1 = [1,2,3,...]
    @inbounds for iter in Base.OneTo(Nparticles)
        buffer_ancestor[iter] = iter
    end
    ## 2 Resample particles in correct order
    @inbounds for t in (iterₘₐₓ - 1):-1:iter₀
        ## Reassign current ancestor
        for idx in Base.OneTo(Nparticles)
            buffer_ancestor[idx] = ancestor[buffer_ancestor[idx], t + 1]
            buffer_val[idx] = val[buffer_ancestor[idx], t]
        end
        ## Assign resampled particle and set current ancestor as non-resampled
        for idx in Base.OneTo(Nparticles)
            #!NOTE: If this line is used, we can recall shuffle_backward!() as many times as we want and particles are in correct order
            ancestor[idx, t + 1] = idx
            val[idx, t] = buffer_val[idx]
        end
    end
    return nothing
end

############################################################################################
#export
export shuffle!, shuffle_forward!, shuffle_backward!
