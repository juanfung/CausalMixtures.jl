# test/test_priors.jl - default priors (C3, docs/priors.md) and the deprecated ktot method
using Test
using CausalMixtures, LinearAlgebra, Random, Statistics, StatsModels
isdefined(Main, :TestHelpers) || include("TestHelpers.jl")
using .TestHelpers

@testset "Default priors are C3" begin
    pr = CausalMixtures.setup_default_priors(2, 3)
    pb, ps, pd = pr.prior_theta.prior_beta, pr.prior_theta.prior_Sigma, pr.prior_dp
    ## beta = (gamma (kz = 3), beta1 (kx = 2), beta0 (kx = 2)), intercepts first in each block
    @test pb.mu == zeros(7)
    @test pb.Vinv
    @test pb.Vcov ≈ Diagonal([0.5^2, 0.5^2, 0.5^2, 1.0, 0.5^2, 1.0, 0.5^2])
    @test pb.V ≈ inv(pb.Vcov)
    @test ps.rho == 10
    @test ps.R ≈ Diagonal([0.1, 0.153, 0.153])
    @test (pd.J, pd.alpha, pd.alpha_shape, pd.alpha_rate) == (20, 1.0, 2.0, 2.0)

    ## the same diagonal as the Phase 2 simulator's C3 spec
    c3 = PriorSpec(name="C3", sd_intercept=1, sd_intercept_d=0.5, sd_probit=0.5, sd_outcome=0.5,
                   rho=10, r=0.153, alpha_shape=2.0, alpha_rate=2.0)
    for (kx, kz) in ((1, 1), (2, 3), (4, 2))
        p = CausalMixtures.setup_default_priors(kx, kz)
        @test diag(p.prior_theta.prior_beta.Vcov) ≈ g0_V0(c3, kx, kz)
        @test p.prior_theta.prior_Sigma.R ≈ c3.R
    end

    ## keyword overrides
    p = CausalMixtures.setup_default_priors(2, 3; J=5, alpha_shape=0.0, sd_probit=1.0, rho=6, r=0.13)
    @test p.prior_dp.J == 5 && p.prior_dp.alpha_shape == 0.0
    @test diag(p.prior_theta.prior_beta.Vcov)[2:3] ≈ [1.0, 1.0]
    @test p.prior_theta.prior_Sigma.R ≈ Diagonal([1/6, 0.13, 0.13])

    ## intercept positions: none, or not in column 1
    p = CausalMixtures.setup_default_priors(2, 3; xintercept=nothing, zintercept=2)
    @test diag(p.prior_theta.prior_beta.Vcov) ≈ [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25]
    p = CausalMixtures.setup_default_priors(2, 3; xintercept=2, zintercept=3, sd_probit=1.0)
    @test diag(p.prior_theta.prior_beta.Vcov) ≈ [1.0, 1.0, 0.25, 0.25, 1.0, 0.25, 1.0]
    @test_throws ArgumentError CausalMixtures.setup_default_priors(2, 3; zintercept=4)
    @test_throws ArgumentError CausalMixtures.setup_default_priors(0, 3)
end

@testset "Default priors from RawData read the intercepts by name" begin
    data = generate_hedonic_data(30, 3)
    df = data.df
    raw = CausalMixtures.RawData(@formula(Y_obs ~ X), @formula(D_obs ~ X + Z), df)
    @test CausalMixtures.setup_default_priors(raw).prior_theta.prior_beta.Vcov ≈
          CausalMixtures.setup_default_priors(2, 3).prior_theta.prior_beta.Vcov
    ## no intercept in the selection equation
    raw0 = CausalMixtures.RawData(@formula(Y_obs ~ X), @formula(D_obs ~ 0 + X + Z), df)
    p0 = CausalMixtures.setup_default_priors(raw0)
    @test diag(p0.prior_theta.prior_beta.Vcov) ≈ [0.25, 0.25, 1.0, 0.25, 1.0, 0.25]
    ## a one-term formula
    raw1 = CausalMixtures.RawData(@formula(Y_obs ~ 1), @formula(D_obs ~ Z), df)
    @test diag(CausalMixtures.setup_default_priors(raw1).prior_theta.prior_beta.Vcov) ≈ [0.25, 0.25, 1.0, 1.0]
    ## kwargs pass through
    @test CausalMixtures.setup_default_priors(raw; J=4).prior_dp.J == 4
end

@testset "Deprecated setup_default_priors(ktot) keeps the old prior" begin
    pr = @test_logs (:warn, r"deprecated") CausalMixtures.setup_default_priors(7)
    @test pr.prior_theta.prior_beta.Vcov ≈ Matrix(100.0I, 7, 7)
    @test pr.prior_theta.prior_Sigma.rho == 6
    @test pr.prior_theta.prior_Sigma.R ≈ Diagonal([1/6, 2.0, 2.0])
    @test pr.prior_dp.alpha_shape == 0.0
    pr = @test_logs (:warn, r"deprecated") CausalMixtures.setup_default_priors(5, beta_nu=75.0, rho=8.0, r=3.0, J=1)
    @test pr.prior_theta.prior_beta.Vcov ≈ Matrix(75.0I, 5, 5)
    @test pr.prior_theta.prior_Sigma.R ≈ Diagonal([1/8, 3.0, 3.0])
    @test pr.prior_dp.J == 1
end
