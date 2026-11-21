# test/test_helpers.jl -- sanity checks for the statistical test helpers
using Test, LinearAlgebra, Random
isdefined(Main, :TestHelpers) || include("TestHelpers.jl")
using .TestHelpers

@testset "TestHelpers" begin
    rng = test_rng()
    @test rand(test_rng(), 3) == rand(test_rng(), 3)

    mu = [1.0, -2.0, 0.5]
    Sigma = [2.0 0.6 -0.3; 0.6 1.0 0.2; -0.3 0.2 0.5]
    L = cholesky(Sigma).L
    N = 50_000
    X = (mu .+ L * randn(rng, 3, N))'   # N × 3, rows are draws

    @test means_match(X, mu)
    @test covs_match(X, Sigma)
    @test moments_match(X, mu, Sigma)
    @test covs_match(X, Sigma; mean=mu)

    # detects a shift of 0.05 sd at N = 50_000 (z ≈ 11) and a 5% variance error (z ≈ 7)
    @test !means_match(X, mu .+ [0.05sqrt(2.0), 0, 0])
    S_bad = copy(Sigma); S_bad[1, 1] *= 1.05
    @test !covs_match(X, S_bad)

    # vector input and ess scaling
    x = X[:, 1]
    @test means_match(x, [mu[1]])
    z_full = mean_zscores(x, [mu[1] + 0.01])[1]
    z_half = mean_zscores(x, [mu[1] + 0.01]; ess=N / 4)[1]
    @test z_half ≈ z_full / 2
end
