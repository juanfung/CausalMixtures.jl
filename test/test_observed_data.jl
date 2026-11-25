# test/test_observed_data.jl -- Phase 5b: the observed-data posterior of the Gaussian selection
# model (no data augmentation)
using Test, CausalMixtures, LinearAlgebra, Random, Statistics, Distributions
isdefined(Main, :TestHelpers) || include("TestHelpers.jl")
using .TestHelpers
const CMO = CausalMixtures

@testset "Observed-data posterior (Phase 5b)" begin

@testset "log-likelihood = brute-force integral of the trivariate normal" begin
    rng = test_rng()
    kz, kx = 2, 2
    for rep in 1:6
        S = CMO.NobileWishart(rng, 6, [1.0 0 0; 0 2.0 0; 0 0 2.0])   # sigma11 = 1, sizeable correlations
        beta = randn(rng, kz + 2kx)
        x, w = [1.0, randn(rng)], [1.0, randn(rng)]
        mu = [dot(w, beta[1:kz]), dot(x, beta[kz+1:kz+kx]), dot(x, beta[kz+kx+1:end])]
        N3 = MvNormal(mu, Symmetric(S))
        for d in (0, 1)
            o, m = d == 1 ? (2, 3) : (3, 2)
            yo = mu[o] + sqrt(S[o, o]) * randn(rng)
            ## 2-d trapezoid over (d* on the side given by d, y_miss)
            sd1, sdm = 1.0, sqrt(S[m, m])
            ds = d == 1 ? range(0.0, mu[1] + 10sd1, length=1501) : range(mu[1] - 10sd1, 0.0, length=1501)
            ym = range(mu[m] - 10sdm, mu[m] + 10sdm, length=1501)
            step(ds) <= 0 && continue
            z = zeros(3)
            f = [begin z[1] = a; z[o] = yo; z[m] = b; pdf(N3, z) end for a in ds, b in ym]
            wa = fill(step(ds), length(ds)); wa[[1, end]] ./= 2
            wb = fill(step(ym), length(ym)); wb[[1, end]] ./= 2
            integral = wa' * f * wb
            X, W = reshape(x, 1, :), reshape(w, 1, :)
            ll = CMO.obs_loglik(beta, S, X, W, [yo], [d])
            @test isapprox(ll, log(integral); atol=2e-4) || (rep, d, ll, log(integral))
        end
    end
end

@testset "log-Cholesky Jacobian and the (delta, Omega) prior = the NobileWishart density" begin
    rng = test_rng()
    rho, Psi = 9, [5.0 1 -1; 1 4 0.5; -1 0.5 3]
    rp = CMO.RidgePrior(CMO.PriorSigma(rho, Psi ./ rho))
    ktot = 2
    for _ in 1:5
        th = vcat(zeros(ktot), randn(rng, 2) .* 0.5, randn(rng, 3) .* 0.4)
        ## numerical Jacobian of eta -> (O11, O21, O22)
        g(e) = (O = CMO.obsdata_unpack(vcat(zeros(ktot + 2), e), ktot)[3]; [O[1, 1], O[2, 1], O[2, 2]])
        e = th[ktot+3:end]; h = 1e-6
        J = hcat([(g(e + h * I[1:3, k]) - g(e - h * I[1:3, k])) / 2h for k in 1:3]...)
        @test log(abs(det(J))) ≈ CMO.obsdata_unpack(th, ktot)[4] atol=1e-6
    end
    ## prior in (delta, Omega) = NobileWishart density of Sigma (Jacobian 1 between them), up to a constant
    lnobile(S) = -(rho + 4) / 2 * logdet(S) - tr(Psi * inv(S)) / 2
    lprior(dl, O) = logpdf(MvNormal(rp.c, Symmetric(O ./ rp.Psi11)), dl) + logpdf(InverseWishart(rp.nu, rp.Psi221), O)
    Ss = [CMO.NobileWishart(rng, rho, Psi) for _ in 1:6]
    a = [lnobile(S) for S in Ss]
    b = [lprior(CMO.sigma_delta_omega(S)...) for S in Ss]
    @test (a .- a[1]) ≈ (b .- b[1]) atol=1e-8
end

@testset "sampler: with no data it recovers the prior; draws round-trip" begin
    data = generate_hedonic_data(40, 2)
    raw = CMO.RawData(data.formulas.formula_y, data.formulas.formula_d, data.df)
    params = CMO.InputParams(M=1, scale_data=(true, true), verbose=false, model="gaussian")
    _, input, _ = CMO.dpm_init(raw, setup_default_priors(2, 3; J=1), params; rng=Xoshiro(1))
    ## no units: the posterior is the prior
    pb = input.priors.prior_theta.prior_beta
    th0 = CMO.obsdata_pack(zeros(7), [1.0 0 0; 0 0.3 0; 0 0 0.3])
    draws, acc = CMO.obsdata_sampler(Xoshiro(2), input, th0; M=400_000, burn=20_000, thin=10, idx=Int[])
    @test 0.1 < acc < 0.5
    rng = test_rng()
    ps = input.priors.prior_theta.prior_Sigma
    P = reduce(hcat, [begin S = CMO.NobileWishart(rng, ps.rho, ps.rho .* ps.R); vcat(pb.mu + pb.Lcov * randn(rng, 7), S[1, 2], S[1, 3], S[2, 2]) end for _ in 1:40_000])'
    D = reduce(hcat, [begin b, dl, O, _ = CMO.obsdata_unpack(draws[s, :], 7); vcat(b, dl, O[1, 1] + dl[1]^2) end for s in 1:size(draws, 1)])'
    ## MCMC draws: compare means with a generous MC error (ESS well below the draw count)
    se = sqrt.(vec(var(P, dims=1)) ./ 2000)
    @test maximum(abs, (vec(mean(D, dims=1)) - vec(mean(P, dims=1))) ./ se) < 5
    @test maximum(abs, vec(std(D, dims=1)) ./ vec(std(P, dims=1)) .- 1) < 0.15
end

end # Phase 5b
