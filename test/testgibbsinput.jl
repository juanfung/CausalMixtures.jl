using Test
using CausalMixtures
using DataFrames
using StatsModels
using LinearAlgebra


"""
Test the RawData constructor.

This test checks that the RawData constructor creates a valid RawData object.
"""
@testset "RawData constructor" begin
    y_form = @formula(y ~ x1 + x2)
    d_form = @formula(d ~ x1 + x2)
    df = DataFrame(y = [1, 2, 3], x1 = [4, 5, 6], x2 = [7, 8, 9], d = [0, 1, 0])
    raw_data = CausalMixtures.RawData(y_form, d_form, df)
    @test raw_data isa CausalMixtures.RawData
    @test raw_data.y_form == y_form
    @test raw_data.d_form == d_form
    @test raw_data.df == df
end

"""
Test the PriorDP constructor.

This test checks that the PriorDP constructor creates a valid PriorDP object.
"""
@testset "PriorDP constructor" begin
    prior_dp = CausalMixtures.PriorDP(alpha = 1.0, J = 20)
    @test prior_dp isa CausalMixtures.PriorDP
    @test prior_dp.alpha == 1.0
    @test prior_dp.J == 20
end

"""
Test the PriorBeta constructor.

Test that PriorBeta constructor creates a valid PriorBeta object.
"""
@testset "PriorBeta constructor" begin
    prior_beta = CausalMixtures.PriorBeta(mu = zeros(3), V = Matrix(1.0I, 3, 3))
    @test prior_beta isa CausalMixtures.PriorBeta
    @test prior_beta.mu == zeros(3)
    @test prior_beta.V == Matrix(1.0I, 3, 3)
end

"""
Test the PriorSigma constructor.

Test that PriorSigma constructor creates a valid PriorSigma object.
"""
@testset "PriorSigma constructor" begin
    prior_sigma = CausalMixtures.PriorSigma(rho = 5.0, R = Matrix(1.0I, 3, 3))
    @test prior_sigma isa CausalMixtures.PriorSigma
    @test prior_sigma.rho == 5.0
    @test prior_sigma.R == Matrix(1.0I, 3, 3)
end

"""
Test the PriorTheta constructor

Test that PriorTheta constructor creates a valid PriorTheta object
"""
@testset "PriorTheta constructor" begin
    prior_beta = CausalMixtures.PriorBeta(mu = zeros(3), V = Matrix(1.0I, 3, 3))
    prior_sigma = CausalMixtures.PriorSigma(rho = 5.0, R = Matrix(1.0I, 3, 3))
    prior_theta = CausalMixtures.PriorTheta(prior_beta = prior_beta, prior_Sigma = prior_sigma)
    @test prior_theta isa CausalMixtures.PriorTheta
    @test prior_theta.prior_beta == prior_beta
    @test prior_theta.prior_Sigma == prior_sigma
end

"""
Test the InputPriors constructor

Test that InputPriors constructor creates a valid InputPriors object
"""
@testset "InputPriors constructor" begin
    prior_dp = CausalMixtures.PriorDP(alpha = 1.0, J = 20)
    prior_theta = CausalMixtures.PriorTheta()
    input_priors = CausalMixtures.InputPriors(prior_dp = prior_dp, prior_theta = prior_theta)
    @test input_priors isa CausalMixtures.InputPriors
    @test input_priors.prior_dp == prior_dp
    @test input_priors.prior_theta == prior_theta
end

"""
Test the InputParams constructor

Test that InputParams constructor creates a valid InputParams object
"""
@testset "InputParams constructor" begin
    input_params = CausalMixtures.InputParams(M = 1000, scale_data = (true, true), verbose = true)
    @test input_params isa CausalMixtures.InputParams
    @test input_params.M == 1000
    @test input_params.scale_data == (true, true)
    @test input_params.verbose == true
end