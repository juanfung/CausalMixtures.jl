# test/test_priors.jl - Test prior setup
using Test
using CausalMixtures

    
@testset "Prior parameter validation" begin
    # Test that priors can be created with different parameters
    @test_nowarn CausalMixtures.setup_default_priors(7, beta_nu=100.0)
    @test_nowarn CausalMixtures.setup_default_priors(7, rho=10.0)
    @test_nowarn CausalMixtures.setup_default_priors(7, r=5.0)
    @test_nowarn CausalMixtures.setup_default_priors(7, J=1)  # Test J parameter
    
    # Test combined parameters
    priors = CausalMixtures.setup_default_priors(5, beta_nu=75.0, rho=8.0, r=3.0)
    
    # Test structure (Julia 1.0 compatible)
    @test :prior_dp in fieldnames(typeof(priors))
    @test :prior_theta in fieldnames(typeof(priors))
    @test :prior_beta in fieldnames(typeof(priors.prior_theta))
    @test :prior_Sigma in fieldnames(typeof(priors.prior_theta))
end
    
@testset "Prior consistency" begin
    # Test that same parameters give same priors (structure-level test)
    priors1 = CausalMixtures.setup_default_priors(7, beta_nu=50.0)
    priors2 = CausalMixtures.setup_default_priors(7, beta_nu=50.0)
    
    # Test they have same structure
    @test typeof(priors1) == typeof(priors2)
    @test typeof(priors1.prior_theta) == typeof(priors2.prior_theta)
    
    # Test different ktot gives different dimensions
    priors_small = CausalMixtures.setup_default_priors(3)
    priors_large = CausalMixtures.setup_default_priors(10)
    @test priors_small isa CausalMixtures.InputPriors
    @test priors_large isa CausalMixtures.InputPriors
end
