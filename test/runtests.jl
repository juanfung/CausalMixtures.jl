# runtests.jl
using Test

println("Running CausalMixtures.jl Tests...")
println("="^50)

result = @testset "CausalMixtures.jl Tests" begin
    # test constructors
    include("testgibbsinput.jl")
    include("testgibbsstate.jl")
    include("testgibbsout.jl")
    include("testCausalMixtures.jl")

    # test functionality
    include("testgibbsppd.jl")
    include("test_priors.jl")
    include("test_samplers.jl")
    include("test_simulation_utils.jl")

    # ... include other test files as needed
    
    # Integration tests
    include("integration/test_hedonic_model.jl")
end

# Print summary display (simple for julia 1.0 since no failed field in result)
println("\n🎯 TEST RESULTS")
println("="^50)
if result.anynonpass
    println("❌ Some tests failed")
else
    println("🎉 ALL TESTS PASSED!")
end
println("📊 Test object: $result")
println("="^50)
