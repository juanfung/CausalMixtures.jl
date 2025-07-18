using Test
using CausalMixtures

# Test the dpm_init function
@testset "dpm_init function" begin
    # Test with valid inputs
    df = DataFrame(:y => [1, 2, 3], :x1 => [4, 5, 6], :x2 => [7, 8, 9], :d => [0, 1, 0])
    y_form = @formula(y ~ x1 + x2)
    d_form = @formula(d ~ x1 + x2)
    raw_data = CausalMixtures.RawData(y_form, d_form, df)
    prior_dp = CausalMixtures.PriorDP(alpha = 1.0, J = 20)
    prior_theta = CausalMixtures.PriorTheta()
    prior = CausalMixtures.InputPriors(prior_dp = prior_dp, prior_theta = prior_theta)
    param = CausalMixtures.InputParams(M = 1, scale_data = (true, true), verbose = true)
    state, input, output = CausalMixtures.dpm_init(raw_data, prior, param)
    @test state isa CausalMixtures.GibbsState
    @test input isa CausalMixtures.GibbsInput
    @test output isa CausalMixtures.GibbsOut
    
    # Test with different input scenarios
    param = CausalMixtures.InputParams(M = 1000, scale_data = (true, true), verbose = false)
    state, input, output = CausalMixtures.dpm_init(raw_data, prior, param)
    @test state isa CausalMixtures.GibbsState
    @test input isa CausalMixtures.GibbsInput
    @test output isa CausalMixtures.GibbsOut
    
    # Test with edge cases (e.g., invalid model)
    param = CausalMixtures.InputParams(M = 1, scale_data = (true, true), verbose = true, model = "invalid")
    @test_throws ErrorException CausalMixtures.dpm_init(raw_data, prior, param)
end