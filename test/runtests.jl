# runtests.jl
using Test
include("TestHelpers.jl")
using .TestHelpers

println("Running CausalMixtures.jl Tests...")
println("="^50)

result = @testset "CausalMixtures.jl Tests" begin
    # test constructors
    include("testgibbsinput.jl")
    include("testgibbsstate.jl")
    include("testgibbsout.jl")
    include("testCausalMixtures.jl")
    include("testdpm_init.jl")

    # statistical test helpers
    include("test_helpers.jl")

    # test functionality
    include("testgibbsppd.jl")
    include("test_priors.jl")
    include("test_samplers.jl")
    include("test_simulation_utils.jl")

    # Phase 1 correctness fixes
    include("test_phase1_fixes.jl")
    include("test_phase1_followups.jl")

    # Phase 2 prior predictive simulator
    include("test_prior_predictive.jl")

    # multi-chain diagnostics
    include("test_diagnostics.jl")

    # Design A sampler (Phase 4a)
    include("test_designA.jl")
    include("test_geweke.jl")  # smoke; CM_GEWEKE=1 for the long runs

    # Phase 3 state refactor
    include("test_phase3_refactor.jl")

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
