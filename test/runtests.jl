############################################################################################
# Import External PackagesJK
using Test
using Random: Random, AbstractRNG, seed!
using ArgCheck: ArgCheck, @argcheck
using SimpleUnPack: SimpleUnPack, @unpack
using Distributions

############################################################################################
# Import Baytes Packages
using BaytesCore, ModelWrappers, BaytesFilters
#using .BaytesFilters
############################################################################################
# Include Files
include("testhelper/TestHelper.jl");

############################################################################################
# Run Tests
@testset "All tests" begin
    # General
    include("test-memory.jl")
    # PF specific
    include("particlefilter/test-construction.jl")
    include("particlefilter/test-estimate.jl")
    include("particlefilter/test-types.jl")
    include("particlefilter/test-ancestors.jl")
    include("particlefilter/test-Ndata.jl")
end
