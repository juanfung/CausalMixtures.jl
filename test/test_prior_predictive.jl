# test/test_prior_predictive.jl -- the standalone prior predictive simulator (Phase 2)
using Test, CausalMixtures, LinearAlgebra, Random, Statistics, Distributions, SparseArrays
isdefined(Main, :TestHelpers) || include("TestHelpers.jl")
using .TestHelpers

@testset "Prior predictive simulator" begin

@testset "NobileWishart(rng, ...) is the same algorithm as the global-RNG method" begin
    S = [5.0 1 -1; 1 4 0.5; -1 0.5 3]
    seed_global!(4); a = [NobileWishart(9, S) for _ in 1:5]
    b = (rng = Xoshiro(4); [NobileWishart(rng, 9, S) for _ in 1:5])
    @test a == b
end

@testset "CRP number of clusters" begin
    n, alpha = 500, 1.0
    ks = crp_num_clusters(PriorSpec(alpha=alpha), n; reps=20_000, rng=test_rng())
    Ek = sum(alpha / (alpha + i - 1) for i in 1:n)        # exact E[k | alpha, n]
    @test means_match(Float64.(ks), [Ek])
    @test all(==(1), crp_num_clusters(PriorSpec(alpha=1e-12), 50; reps=200, rng=test_rng()))
end

@testset "prior_predictive moments" begin
    rng = test_rng()
    n = 40
    X = hcat(ones(n), randn(rng, n))
    W = hcat(ones(n), randn(rng, n), randn(rng, n))
    xbar = vec(mean(X, dims=1))
    for slopes in (:cluster, :common)
        s = PriorSpec(slopes=slopes, sd_intercept=0.7, sd_probit=0.4, sd_outcome=0.5, rho=8, r=0.5,
                      alpha_shape=2.0, alpha_rate=2.0)
        pp = prior_predictive(s, X, W; reps=3000, rng=rng)
        nclust = sum(pp.k)
        @test length(pp.s22) == nclust == length(pp.effect)
        @test length(pp.prop) == 3000 * n
        ## cluster effect xbar'(beta1 - beta0) is N(0, 2 (sd_int^2 + sum_k xbar_k^2 sd_out^2));
        ## draws are independent across clusters under :cluster only
        v_eff = 2 * (0.7^2 + sum(xbar[2:end] .^ 2) * 0.5^2)
        if slopes == :cluster
            @test means_match(pp.effect, [0.0])
            @test covs_match(pp.effect, fill(v_eff, 1, 1); mean=[0.0])
        end
        ## cluster mean of y1 has sd sqrt(sd_int^2 + ...) and zero mean
        @test abs(mean(pp.mu1)) < 0.1
        @test all(x -> 0 <= x <= 1, pp.prop)
        @test all(>(0), pp.s22)
    end
    ## all slopes and intercepts zero: every propensity is exactly 0.5
    s0 = PriorSpec(sd_intercept=0.0, sd_probit=0.0, sd_outcome=0.0)
    @test all(==(0.5), prior_predictive(s0, X, W; reps=5, rng=rng).prop)
end

@testset "g0_logpredictive matches brute-force Monte Carlo over beta and Sigma" begin
    rng = test_rng()
    x = [1.0, 0.8, -0.3]
    w = [1.0, -0.5, 1.2]
    z = [0.4, 1.1, -0.6]
    for slopes in (:cluster, :common)
        s = PriorSpec(slopes=slopes, sd_intercept=0.8, sd_probit=0.6, sd_outcome=0.5, rho=7, r=0.4)
        b = slopes == :common ? [0.3, -0.2, 0.5, 0.1, -0.4, 0.2] : nothing
        sig = draw_sigmas(s, 4000; rng=rng)
        lp = g0_logpredictive(s, z, x, w, sig; b=b)
        ## brute force: for each Sigma, average N(z | H beta, Sigma) over 200 beta draws
        H = Matrix(blockdiag(sparse(w'), sparse(x'), sparse(x')))
        v = sqrt.(g0_V0(s, length(x), length(w)))
        acc = 0.0
        for Sg in sig
            d = MvNormal(zeros(3), Symmetric(Sg))
            for _ in 1:200
                if slopes == :cluster
                    m = H * (v .* randn(rng, length(v)))
                else
                    m = [dot(w[2:end], b[1:2]), dot(x[2:end], b[3:4]), dot(x[2:end], b[5:6])] .+ 0.8 .* randn(rng, 3)
                end
                acc += pdf(d, z - m)
            end
        end
        @test isapprox(lp, log(acc / (200 * length(sig))); atol=0.03)
    end
end

end # Prior predictive simulator
