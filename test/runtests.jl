# runtests.jl
using Test

@testset "CausalMixtures.jl Tests" begin
    include("testgibbsinput.jl")
    include("testgibbsstate.jl")
    include("testgibbsout.jl")
    include("testCausalMixtures.jl")
    include("testgibbsppd.jl")
    # ... include other test files as needed
    
    # Integration tests
    include("integration/test_hedonic_model.jl")
end
