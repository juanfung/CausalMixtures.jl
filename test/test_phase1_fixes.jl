# test/test_phase1_fixes.jl -- one testset per Phase 1 correctness fix
# (docs/implementation_plan.md, section 03).
using Test, CausalMixtures, LinearAlgebra, Random, Statistics, StatsModels, Distributions
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

    ## defaults (C3): V0 = diag(0.5², 0.5², 0.5², 1, 0.5², 1, 0.5²) for (gamma, beta1, beta0) with
    ## kz = 3, kx = 2 (the precision was used as the covariance before the fix)
    pr = setup_default_priors(2, 3)
    V0 = Matrix(Diagonal([0.25, 0.25, 0.25, 1.0, 0.25, 1.0, 0.25]))
    @test pr.prior_theta.prior_beta.V ≈ inv(V0)
    @test pr.prior_theta.prior_beta.Vcov ≈ V0
    seed_global!()
    th = CausalMixtures.StateTheta()
    B = reduce(vcat, (CausalMixtures.sample_prior_theta!(th, pr.prior_theta, 1)[1].beta' for _ in 1:20_000))
    @test covs_match(B, V0; mean=zeros(7))
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

@testset "Fix 5: label probabilities in log space" begin
    lse = CausalMixtures.logsumexp
    @test lse([log(1.0), log(2.0), log(3.0)]) ≈ log(6.0)
    @test lse([-1e4, -1e4]) ≈ -1e4 + log(2)
    @test lse([1e4, 1e4]) ≈ 1e4 + log(2)
    @test lse([-Inf, 0.0]) == 0.0

    ## extreme log weights: exp() underflows to 0 for all of them, but the ratios are fine
    seed_global!()
    logw = [-2e4, -2e4 + log(3), -3e4]
    draws = [CausalMixtures.sample_logweights(logw) for _ in 1:40_000]
    @test all(in((1, 2)), draws)
    @test means_match(Float64.(draws .== 2), [0.75])
    @test_throws ErrorException CausalMixtures.sample_logweights([-Inf, -Inf])
    @test_throws ErrorException CausalMixtures.sample_logweights([0.0, NaN])

    ## every label sampler survives a unit whose latent data sit far from all components
    data = generate_hedonic_data(60, 4321)
    raw = CausalMixtures.RawData(data.formulas.formula_y, data.formulas.formula_d, data.df)
    for model in ("dpm", "fmn", "blocked", "marginal")
        seed_global!()
        priors = setup_default_priors(2, 3, J=3)
        params = CausalMixtures.InputParams(M=1, verbose=false, model=model)
        state, input, out = CausalMixtures.dpm_init(raw, priors, params)
        state.state_data.y1[1] = 1e4
        state.state_data.y0[1] = -1e4
        update! = Dict("dpm" => CausalMixtures.update_labels!,
                       "fmn" => CausalMixtures.update_fmn_labels!,
                       "blocked" => CausalMixtures.update_blocked_labels!,
                       "marginal" => CausalMixtures.update_marginal_labels!)[model]
        state = update!(state, input)
        labs = [state.state_dp.labels[i] for i in 1:input.dims.n]
        @test all(l -> l >= 1, labs)
        @test sum(values(state.state_dp.njs)) == input.dims.n
    end
end

@testset "Fix 6: njs has an entry for every component, including empty ones" begin
    ## n = 12 units spread over J = 30 components: most components start empty
    data = generate_hedonic_data(12, 2024)
    raw = CausalMixtures.RawData(data.formulas.formula_y, data.formulas.formula_d, data.df)
    priors = setup_default_priors(2, 3, J=30)
    for model in ("dpm", "fmn", "blocked", "marginal")
        seed_global!()
        params = CausalMixtures.InputParams(M=3, verbose=false, model=model)
        state, input, out = CausalMixtures.dpm_init(raw, priors, params)
        dp = state.state_dp
        @test sort(collect(keys(dp.njs))) == collect(1:dp.J)
        @test sum(values(dp.njs)) == 12
        @test length(state.state_theta) == dp.J
        if model == "dpm"
            ## Polya urn: only occupied components, labelled 1:k
            @test all(>(0), values(dp.njs))
            @test dp.J == length(unique(values(dp.labels)))
        else
            @test dp.J == 30
            @test any(==(0), values(dp.njs))
        end
        state, input, out = CausalMixtures.dpm!(state, input, out)
        @test length(out.out_dp) == 3
    end
end

@testset "Fix 10: init with a covariance prior (Vinv = false)" begin
    data = generate_hedonic_data(80, 99)
    raw = CausalMixtures.RawData(data.formulas.formula_y, data.formulas.formula_d, data.df)
    K = data.true_params.ktot
    V0 = Matrix(Diagonal(collect(1.0:K))) + 0.1 * ones(K, K)
    R = Matrix(Diagonal([1/7, 1.5, 2.5]))
    priors = CausalMixtures.InputPriors(CausalMixtures.PriorDP(J=4, alpha=1.0),
                 CausalMixtures.PriorTheta(CausalMixtures.PriorBeta(mu=fill(0.5, K), V=V0, Vinv=false),
                                           CausalMixtures.PriorSigma(rho=7, R=R)))
    for model in ("dpm", "blocked")
        seed_global!()
        params = CausalMixtures.InputParams(M=2, verbose=false, model=model)
        state, input, out = CausalMixtures.dpm_init(raw, priors, params)
        pb = input.priors.prior_theta.prior_beta
        @test pb.Vinv
        @test pb.V ≈ inv(V0)
        @test pb.Vcov == V0
        @test state.state_sampler.Vmu ≈ inv(V0) * fill(0.5, K)
        @test input.priors.prior_theta.prior_Sigma.rho == 7
        @test input.priors.prior_theta.prior_Sigma.R == R
        state, input, out = CausalMixtures.dpm!(state, input, out)
        @test length(out.out_theta) == 2
    end
end

@testset "Fix P1: saved blocked weights do not alias across iterations" begin
    data = generate_hedonic_data(100, 7)
    raw = CausalMixtures.RawData(data.formulas.formula_y, data.formulas.formula_d, data.df)
    priors = setup_default_priors(2, 3, J=6)
    params = CausalMixtures.InputParams(M=20, verbose=false, model="blocked")
    seed_global!()
    state, input, out = CausalMixtures.dpm!(CausalMixtures.dpm_init(raw, priors, params)...)
    M = params.M
    ## saved weights must vary over iterations (before the fix all equalled the last iteration's)
    v1 = [out.out_dp[m].ws[1].v for m in 1:M]
    @test length(unique(v1)) == M
    @test out.out_dp[M].ws[1].v == state.state_dp.ws[1].v
    @test out.out_dp[1].ws[1].v != state.state_dp.ws[1].v
    for m in 1:M
        ws = out.out_dp[m].ws
        v = [ws[j].v for j in 1:6]
        w = [ws[j].w for j in 1:6]
        ## stick-breaking identity holds within each saved iteration, and the weights sum to 1
        @test w ≈ [v[j] * prod(1 .- v[1:j-1]) for j in 1:6]
        @test sum(w) ≈ 1
        @test v[6] == 1.0
    end
    ## the BlockedWeights algebra
    b1, b2 = CausalMixtures.BlockedWeights(0.5, 0.25), CausalMixtures.BlockedWeights(0.125, 0.5)
    @test (1 - b1).w == 0.5 && (1 - b1).v == 0.75
end

@testset "Fix P5: PPD covariates built by column name, averaged over all n" begin
    data = generate_hedonic_data(150, 31)
    df = data.df
    yf = @formula(Y_obs ~ X)
    raw_xz = CausalMixtures.RawData(yf, @formula(D_obs ~ X + Z), df)
    raw_zx = CausalMixtures.RawData(yf, @formula(D_obs ~ Z + X), df)
    priors = setup_default_priors(2, 3, J=1)
    params = CausalMixtures.InputParams(M=25, verbose=false, model="gaussian")

    seed_global!()
    _, in_xz, out_xz = CausalMixtures.dpm!(CausalMixtures.dpm_init(raw_xz, priors, params)...)
    _, in_zx, _ = CausalMixtures.dpm_init(raw_zx, priors, params)
    @test in_xz.data.znames == ["(Intercept)", "X", "Z"]
    @test in_zx.data.znames == ["(Intercept)", "Z", "X"]
    @test in_zx.data.xnames == ["(Intercept)", "X"]

    ## covariate means over all n (scale_data = false, so on the data scale)
    xbar, wbar = mean_covariates(in_zx)
    @test xbar ≈ [1.0, mean(df.X)]
    @test wbar ≈ [1.0, mean(df.Z), mean(df.X)]

    ## the same posterior expressed in the Z + X column order
    kz, kx = in_xz.dims.kz, in_xz.dims.kx
    perm = [findfirst(==(nm), in_xz.data.znames) for nm in in_zx.data.znames]
    out_zx = CausalMixtures.GibbsOut(params.M)
    for m in 1:params.M
        th = out_xz.out_theta[m][1]
        b = copy(th.beta)
        b[1:kz] = th.beta[perm]
        out_zx.out_theta[m] = CausalMixtures.StateTheta(1 => CausalMixtures.Theta(b, th.Sigma))
        out_zx.out_dp[m] = out_xz.out_dp[m]
        out_zx.out_data[m] = out_xz.out_data[m]
    end

    ## same PPD from both orders, via (xbar, wbar) and via the selection covariates alone
    for args in ((mean_covariates(in_xz)...,), (mean_covariates(in_xz)[2],))
        args_zx = length(args) == 2 ? mean_covariates(in_zx) : (mean_covariates(in_zx)[2],)
        seed_global!(1); p_xz = CausalMixtures.rand_ppd(out_xz, in_xz, args...)
        seed_global!(1); p_zx = CausalMixtures.rand_ppd(out_zx, in_zx, args_zx...)
        @test p_xz ≈ p_zx
    end
    ## the old znew[1:kx] slicing would have used (1, Zbar) as the outcome covariates here
    h = CausalMixtures.ppd_design(in_zx, wbar)
    @test Matrix(h)[2, (kz+1):(kz+kx)] ≈ [1.0, mean(df.X)]

    ## an outcome column missing from the selection equation needs xnew and wnew explicitly
    raw_w = CausalMixtures.RawData(@formula(Y_obs ~ X), @formula(D_obs ~ Z), df)
    _, in_w, _ = CausalMixtures.dpm_init(raw_w, setup_default_priors(2, 2, J=1), params)
    @test_throws ArgumentError CausalMixtures.ppd_design(in_w, [1.0, 0.5])
    @test size(CausalMixtures.ppd_design(in_w, [1.0, 0.2], [1.0, 0.5])) == (3, 6)
    @test_throws DimensionMismatch CausalMixtures.ppd_design(in_w, [1.0], [1.0, 0.5])
end

@testset "Odds and ends: BlockedWeights minus, RawData export" begin
    b1, b2 = CausalMixtures.BlockedWeights(0.5, 0.25), CausalMixtures.BlockedWeights(0.125, 0.5)
    @test (b1 - b2).w == 0.375 && (b1 - b2).v == -0.25
    @test (b1 + b2).w == 0.625 && (b1 + b2).v == 0.75
    @test :RawData in names(CausalMixtures)
    @test !(:RawRata in names(CausalMixtures))
end

@testset "Odds and ends: test = true paths update theta and keep latent data fixed" begin
    data = generate_hedonic_data(60, 4321)
    raw = CausalMixtures.RawData(data.formulas.formula_y, data.formulas.formula_d, data.df)
    for model in ("dpm", "blocked", "marginal", "fmn", "gaussian")
        seed_global!()
        priors = setup_default_priors(2, 3, J = model == "gaussian" ? 1 : 3)
        state, input, out = CausalMixtures.dpm_init(raw, priors,
                                CausalMixtures.InputParams(M=3, verbose=false, model=model))
        latent = copy(state.state_data)
        state, input, out = CausalMixtures.dpm!(state, input, out; test=true)
        @test state.state_data.dstar == latent.dstar
        @test state.state_data.y1 == latent.y1
        @test state.state_data.y0 == latent.y0
        @test out.out_theta[1][1].beta != out.out_theta[3][1].beta
    end
end

@testset "Odds and ends: pre-1.0 helpers (check_y, batch_var, dpm_dump!)" begin
    data = generate_hedonic_data(50, 11)
    raw = CausalMixtures.RawData(data.formulas.formula_y, data.formulas.formula_d, data.df)
    seed_global!()
    params = CausalMixtures.InputParams(M=4, scale_data=(true, true), verbose=false, model="blocked")
    state, input, out = CausalMixtures.dpm!(CausalMixtures.dpm_init(raw, setup_default_priors(2, 3, J=3), params)...)
    ## the observed outcome is never imputed, so check_y recovers it on the original scale
    @test CausalMixtures.check_y(input, out) ≈ data.df.Y_obs

    a = collect(1.0:12.0)
    @test CausalMixtures.batch_var(a, 3) ≈ var([2.0, 5.0, 8.0, 11.0]) / 4
    @test_throws ArgumentError CausalMixtures.batch_var(a, 5)

    ## dpm_dump! (used gc()) writes the batch to disk and continues the chain
    mktempdir() do dir
        state2, input2, out2 = CausalMixtures.dpm_dump!(state, input, out; fname="chain", dir=dir * "/")
        @test isfile(joinpath(dir, "chain.jld2"))
        @test length(out2.out_dp) == params.M
    end

    @test !isdefined(CausalMixtures, :make_table)
    @test !isdefined(CausalMixtures, :ppd_cdf)
    @test !isdefined(CausalMixtures, :rescale_output)
end

@testset "Odds and ends: PPD prior draws work when Vinv = false" begin
    data = generate_hedonic_data(60, 5)
    raw = CausalMixtures.RawData(data.formulas.formula_y, data.formulas.formula_d, data.df)
    seed_global!()
    params = CausalMixtures.InputParams(M=6, verbose=false, model="dpm")
    state, input, out = CausalMixtures.dpm!(CausalMixtures.dpm_init(raw, setup_default_priors(2, 3, J=3), params)...)
    V0 = Matrix(Diagonal(collect(1.0:7.0)))
    sig = input.priors.prior_theta.prior_Sigma
    pt_cov = CausalMixtures.PriorTheta(CausalMixtures.PriorBeta(mu=zeros(7), V=V0, Vinv=false), sig)
    pt_prec = CausalMixtures.PriorTheta(CausalMixtures.PriorBeta(mu=zeros(7), V=inv(V0), Vinv=true), sig)
    hnew = CausalMixtures.ppd_design(input, mean_covariates(input)...)
    seed_global!(3); a = CausalMixtures.rand_dpm(out, pt_cov, hnew)
    seed_global!(3); b = CausalMixtures.rand_dpm(out, pt_prec, hnew)
    @test size(a) == (3, params.M)
    @test a ≈ b
    znew = mean_covariates(input)[2]
    seed_global!(3); c = CausalMixtures.dpm_ppd(out, input, znew; prior_theta=pt_cov)
    seed_global!(3); d = CausalMixtures.dpm_ppd(out, input, znew; prior_theta=pt_prec)
    @test c ≈ d
end

@testset "Odds and ends: parallel_ppd retired, marginal deprecated" begin
    @test !isdefined(CausalMixtures, :parallel_ppd)
    @test !isdefined(CausalMixtures, :single_dpm_ppd)
    data = generate_hedonic_data(40, 8)
    raw = CausalMixtures.RawData(data.formulas.formula_y, data.formulas.formula_d, data.df)
    priors = setup_default_priors(2, 3, J=3)
    @test_logs (:warn, r"deprecated") CausalMixtures.dpm_init(raw, priors,
        CausalMixtures.InputParams(M=1, verbose=false, model="marginal"))
    @test_logs CausalMixtures.dpm_init(raw, priors, CausalMixtures.InputParams(M=1, verbose=false, model="blocked"))
end

@testset "NobileWishart matches brute-force IW draws conditioned on sigma11 = 1" begin
    ## Draw Sigma ~ InverseWishart(nu, S) (density ∝ |Σ|^{-(nu+4)/2} exp(-tr(S Σ^{-1})/2)),
    ## keep draws with |sigma11 - 1| < 0.02, and compare with NobileWishart(nu, S).
    ## Uses Distributions' sampler, independent of NobileWishart's Bartlett construction.
    entries(X) = [X[2, 2], X[3, 3], X[1, 2], X[1, 3], X[2, 3]]
    settings = ((9, [5.0 1 -1; 1 4 0.5; -1 0.5 3]), (14, [10.0 3 -2; 3 8 1; -2 1 6]))
    for (nu, S) in settings
        rng = test_rng()
        IW = InverseWishart(nu, S)
        kept = Vector{Vector{Float64}}()
        for _ in 1:1_500_000
            X = rand(rng, IW)
            abs(X[1, 1] - 1) < 0.02 && push!(kept, entries(X))
        end
        B = reduce(hcat, kept)'
        seed_global!()
        Nw = reduce(hcat, [entries(CausalMixtures.NobileWishart(nu, S)) for _ in 1:200_000])'
        @test size(B, 1) > 25_000
        @test all(x -> x[1, 1] == 1.0, (CausalMixtures.NobileWishart(nu, S) for _ in 1:10))
        ## means and variances of sigma22, sigma33, sigma12, sigma13, sigma23
        @test maximum(abs, two_sample_zscores(B, Nw)) < 4.5
        sq(X) = (X .- mean(X, dims=1)) .^ 2
        @test maximum(abs, two_sample_zscores(sq(B), sq(Nw))) < 4.5
        ## power: a wrong degrees of freedom is detected
        seed_global!()
        Nw1 = reduce(hcat, [entries(CausalMixtures.NobileWishart(nu + 1, S)) for _ in 1:200_000])'
        @test maximum(abs, two_sample_zscores(B, Nw1)) > 10
    end
end

end # Phase 1 fixes
