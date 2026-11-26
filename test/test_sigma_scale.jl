# test/test_sigma_scale.jl -- Phase 6a': the learned Sigma scale (sigma_scale = :hier)
using Test, CausalMixtures, LinearAlgebra, Random, Statistics, Distributions
isdefined(Main, :TestHelpers) || include("TestHelpers.jl")
using .TestHelpers
const CMS = CausalMixtures

## a state with K clusters of the given Sigmas, for update_sigma_scale!
function scale_state(Sigmas; a=2.0, b=2 / 0.153)
    data = generate_hedonic_data(60, 3)
    raw = CMS.RawData(data.formulas.formula_y, data.formulas.formula_d, data.df)
    p = CMS.InputParams(M=1, scale_data=(true, true), verbose=false, model="marginal", sigma_scale=:hier, r_shape=a, r_rate=b)
    st, input, _ = CMS.dpm_init(raw, setup_default_priors(2, 3; J=4), p; rng=Xoshiro(1))
    K = length(Sigmas)
    st.state_theta = [CMS.Theta(beta=st.state_theta[1].beta, Sigma=S) for S in Sigmas]
    st.state_dp.njs = fill(5, K); st.state_dp.J = K
    return st, input
end

@testset "Learned Sigma scale (Phase 6a')" begin
    @testset "r_k | Omega_1..K = brute-force grid of prior x IW_2 densities" begin
        rng = test_rng()
        rho, a, b = 10, 2.0, 2 / 0.153
        Sigmas = [CMS.NobileWishart(rng, rho, rho .* Matrix(Diagonal([0.1, 0.2, 0.3]))) for _ in 1:4]
        Os = [CMS.sigma_delta_omega(S)[2] for S in Sigmas]
        st, input = scale_state(Sigmas; a=a, b=b)
        K = length(Os)
        for (k, rother) in ((1, 0.3), (2, 0.2))
            ## brute force: log Gamma prior + sum_j log IW_2(Omega_j | rho, diag(rho r1, rho r0)), other r fixed
            grid = range(1e-4, 1.5, length=30_001)
            lp = map(grid) do r
                rr = k == 1 ? (r, rother) : (rother, r)
                logpdf(Gamma(a, 1 / b), r) + sum(logpdf(InverseWishart(rho, Matrix(Diagonal([rho * rr[1], rho * rr[2]]))), Matrix(Symmetric(O))) for O in Os)
            end
            w = exp.(lp .- maximum(lp)); w ./= sum(w) * step(grid)
            G = Gamma(a + K * rho / 2, 1 / (b + rho / 2 * sum(inv(O)[k, k] for O in Os)))
            @test maximum(abs, w .- pdf.(G, grid)) < 1e-3 * maximum(w)
            ## the sampler's draws follow G
            N = 100_000
            draws = [(CMS.update_sigma_scale!(rng, st, input); st.state_hyper.r[k]) for _ in 1:N]
            @test abs(mean(draws) - mean(G)) < 4.5 * std(G) / sqrt(N)
            @test abs(var(draws) / var(G) - 1) < 0.03
        end
    end

    @testset "invariance: r ~ prior, Sigma_j | r ~ NobileWishart, one update -> prior again; shape + K rho fails" begin
        rng = test_rng()
        rho, a, b, K = 10, 2.0, 2 / 0.153, 3
        st, input = scale_state([Matrix(1.0I, 3, 3) for _ in 1:K])
        function after(shape_extra)
            out = zeros(50_000, 2)
            for t in axes(out, 1)
                r = [rand(rng, Gamma(a, 1 / b)) for _ in 1:2]
                st.state_theta = [CMS.Theta(beta=st.state_theta[1].beta,
                                  Sigma=CMS.NobileWishart(rng, rho, rho .* Matrix(Diagonal([1 / rho, r[1], r[2]])))) for _ in 1:K]
                if shape_extra == 0
                    CMS.update_sigma_scale!(rng, st, input)
                    out[t, :] = st.state_hyper.r
                else   # planted: shape a + K rho instead of a + K rho / 2
                    s = [sum(inv(CMS.sigma_delta_omega(th.Sigma)[2])[k, k] for th in st.state_theta) for k in 1:2]
                    out[t, :] = [rand(rng, Gamma(a + K * rho, 1 / (b + rho / 2 * s[k]))) for k in 1:2]
                end
            end
            return out
        end
        P = hcat([rand(rng, Gamma(a, 1 / b), 50_000) for _ in 1:2]...)
        sq(X) = (X .- mean(X, dims=1)) .^ 2
        A = after(0)
        @test maximum(abs, two_sample_zscores(log.(A), log.(P))) < 4.5
        @test maximum(abs, two_sample_zscores(sq(log.(A)), sq(log.(P)))) < 4.5
        @test maximum(abs, two_sample_zscores(log.(after(1)), log.(P))) > 10
    end

    @testset "option wiring: r starts at the priors' R, is saved, and both samplers run" begin
        @test CMS.InputParams().sigma_scale == :fixed
        @test_throws ArgumentError CMS.InputParams(sigma_scale=:nope)
        data = generate_hedonic_data(80, 4)
        raw = CMS.RawData(data.formulas.formula_y, data.formulas.formula_d, data.df)
        for model in ("marginal", "gaussian")
            p = CMS.InputParams(M=20, scale_data=(true, true), verbose=false, model=model, sigma_scale=:hier)
            rng = Xoshiro(2)
            init = CMS.dpm_init(raw, setup_default_priors(raw; J = model == "gaussian" ? 1 : 10), p; rng=rng)
            @test init[1].state_hyper.r == [0.153, 0.153]
            st, input, out = CMS.dpm!(init...; rng=rng)
            @test all(length.(out.out_r) .== 2) && all(x -> all(>(0), x), out.out_r)
            @test out.out_r[end] == st.state_hyper.r
            @test CMS.sigma_R(st, input)[2, 2] == st.state_hyper.r[1]
            @test all(isfinite, CMS.effects(out, input; rng=rng).pate)
        end
        ## :fixed keeps the priors' R and saves no r
        p = CMS.InputParams(M=5, scale_data=(true, true), verbose=false, model="marginal")
        st, input, out = CMS.dpm!(CMS.dpm_init(raw, setup_default_priors(raw), p; rng=Xoshiro(3))...; rng=Xoshiro(3))
        @test isempty(st.state_hyper.r) && all(isempty, out.out_r) && CMS.sigma_R(st, input) === input.priors.prior_theta.prior_Sigma.R
        @test_throws ArgumentError CMS.dpm_init(raw, setup_default_priors(raw),
            CMS.InputParams(M=5, verbose=false, model="blocked", sigma_scale=:hier); rng=Xoshiro(1))
    end
end
