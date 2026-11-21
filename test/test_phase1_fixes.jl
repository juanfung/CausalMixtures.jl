# test/test_phase1_fixes.jl -- one testset per Phase 1 correctness fix
# (docs/implementation_plan.md, section 03).
using Test, CausalMixtures, LinearAlgebra, Random, Statistics
isdefined(Main, :TestHelpers) || include("TestHelpers.jl")
using .TestHelpers

@testset "Phase 1 fixes" begin

@testset "Fix 1: prior beta draws use the covariance V0" begin
    mu = [0.5, -1.0, 2.0, 0.0]
    A = [1.0 0.3 0.0 0.2; 0.0 2.0 -0.4 0.0; 0.0 0.0 0.7 0.1; 0.0 0.0 0.0 3.0]
    V0 = A' * A                                    # non-diagonal covariance
    sigma = CausalMixtures.PriorSigma(rho=6, R=Matrix(Diagonal([1/6, 2.0, 2.0])))
    N = 100_000
    for Vinv in (true, false)
        V_given = Vinv ? inv(V0) : V0              # precision or covariance, as the user passes it
        pt = CausalMixtures.PriorTheta(CausalMixtures.PriorBeta(mu=mu, V=V_given, Vinv=Vinv), sigma)
        @test pt.prior_beta.Vcov ≈ V0
        @test pt.prior_beta.Lcov * pt.prior_beta.Lcov' ≈ V0
        seed_global!()
        th = CausalMixtures.StateTheta()
        B = Matrix{Float64}(undef, N, length(mu))
        for r in 1:N
            CausalMixtures.sample_prior_theta!(th, pt, 1)
            B[r, :] = th[1].beta
        end
        @test moments_match(B, mu, V0)
    end

    ## defaults: V0 = 100 I, so prior sd 10 (it was 0.1 before the fix)
    pr = setup_default_priors(7)
    @test pr.prior_theta.prior_beta.V ≈ Matrix(0.01I, 7, 7)
    @test pr.prior_theta.prior_beta.Vcov ≈ Matrix(100.0I, 7, 7)
    seed_global!()
    th = CausalMixtures.StateTheta()
    B = reduce(vcat, (CausalMixtures.sample_prior_theta!(th, pr.prior_theta, 1)[1].beta' for _ in 1:20_000))
    @test covs_match(B, Matrix(100.0I, 7, 7); mean=zeros(7))
end

@testset "Fix 3: Escobar-West alpha step uses a random mixture indicator" begin
    ## Holding k and n fixed, repeated alpha updates form a Markov chain whose stationary law is
    ## p(alpha | k, n) ∝ Gamma(alpha; a, rate b) * alpha^k * Gamma(alpha) / Gamma(alpha + n).
    ## Compare its mean and variance with quadrature of that density.
    function alpha_posterior_moments(a, b, k, n)
        grid = range(1e-6, 60.0, length=600_001)
        logp = [(a + k - 1) * log(x) - b * x - sum(log(x + i) for i in 0:n-1) for x in grid]
        w = exp.(logp .- maximum(logp))
        m1 = sum(w .* grid) / sum(w)
        m2 = sum(w .* grid .^ 2) / sum(w)
        return m1, m2 - m1^2
    end

    a, b, n, M = 2.0, 2.0, 50, 100_000
    for k in (1, 4, 12), update! in (CausalMixtures.update_alpha!, CausalMixtures.update_marginal_alpha!)
        input = CausalMixtures.GibbsInput(dims=CausalMixtures.InputDims(n=n),
                    priors=CausalMixtures.InputPriors(prior_dp=CausalMixtures.PriorDP(J=k, alpha=1.0,
                                                      alpha_shape=a, alpha_rate=b)))
        state = CausalMixtures.GibbsState(state_dp=CausalMixtures.StateDP(J=k, alpha=1.0))
        seed_global!()
        draws = [update!(state, input).state_dp.alpha for _ in 1:M]
        Ea, Va = alpha_posterior_moments(a, b, k, n)
        ess = ess_batch(draws)
        @test means_match(draws, [Ea]; ess=ess)
        @test covs_match(draws, fill(Va, 1, 1); mean=[Ea], ess=ess)
    end
end

end # Phase 1 fixes
