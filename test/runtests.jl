############################################################################################
# Import External PackagesJK
using Test
using Random: Random, AbstractRNG, seed!
using ArgCheck: ArgCheck, @argcheck
using UnPack: UnPack, @unpack
using Distributions

############################################################################################
# Import Baytes Packages
using ModelWrappers, BaytesFilters

############################################################################################
# Include Files
include("TestHelper.jl")

############################################################################################
# Run Tests
@testset "All tests" begin
    # General
    include("test-memory.jl")
    # PF specific
    include("particlefilter/test-construction.jl")
    include("particlefilter/test-types.jl")
    include("particlefilter/test-ancestors.jl")
    include("particlefilter/test-Ndata.jl")
end
