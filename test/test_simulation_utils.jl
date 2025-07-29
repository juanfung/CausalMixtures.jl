# test/test_simulation_utils.jl
using Test
using CausalMixtures

@testset "generate_hedonic_data function" begin
    # Test basic functionality
    @test_nowarn data = CausalMixtures.generate_hedonic_data(100, 1234)
    
    data = CausalMixtures.generate_hedonic_data(50, 42)
    
    # Test data structure
    @test size(data.df, 1) == 50
    @test "Y_obs" in names(data.df)
    @test "D_obs" in names(data.df)
    @test "X" in names(data.df)
    @test "Z" in names(data.df)
    
    # Test design info
    @test data.design_info.n == 50
    @test data.design_info.n_treated + data.design_info.n_control == 50
    @test data.design_info.n_treated > 0
    @test data.design_info.n_control > 0
    
    # Test true effects are reasonable
    @test abs(data.true_effects.ate) < 20
    @test !isnan(data.true_effects.ate)
    
    # Test reproducibility with same seed
    data1 = CausalMixtures.generate_hedonic_data(30, 999)
    data2 = CausalMixtures.generate_hedonic_data(30, 999)
    @test data1.true_effects.ate == data2.true_effects.ate
end
    
@testset "setup_default_priors function" begin
    # Test basic functionality
    @test_nowarn priors = CausalMixtures.setup_default_priors(7)
    
    priors = CausalMixtures.setup_default_priors(5)
    
    # Test structure exists (Julia 1.0 compatible)
    @test :prior_dp in fieldnames(typeof(priors))
    @test :prior_theta in fieldnames(typeof(priors))
    @test :prior_beta in fieldnames(typeof(priors.prior_theta))
    @test :prior_Sigma in fieldnames(typeof(priors.prior_theta))
    
    # Test custom parameters (just test it doesn't crash)
    @test_nowarn priors_custom = CausalMixtures.setup_default_priors(7, beta_nu=50.0, rho=6.0, r=2.0)
    
    # Test invalid input (Julia 1.0 compatible error types)
    #@test_throws ErrorException CausalMixtures.setup_default_priors(0)
    #@test_throws ErrorException CausalMixtures.setup_default_priors(-1)
end
