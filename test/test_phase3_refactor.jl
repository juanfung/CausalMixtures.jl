# test/test_phase3_refactor.jl -- Phase 3 state refactor (docs/implementation_plan.md, section 05)
using Test, CausalMixtures, LinearAlgebra, Random, Statistics, StatsModels, Distributions, SparseArrays
isdefined(Main, :TestHelpers) || include("TestHelpers.jl")
using .TestHelpers

@testset "Phase 3 refactor" begin

## the state invariants every label sweep must leave behind
function check_state(state, input; occupied_only, fixed_J=false)
    dp = state.state_dp
    K = length(dp.njs)
    @test length(dp.labels) == input.dims.n
    @test all(l -> 1 <= l <= K, dp.labels)
    @test dp.njs == [count(==(j), dp.labels) for j in 1:K]
    @test length(dp.members) == K
    @test all(j -> dp.members[j] == findall(==(j), dp.labels), 1:K)
    @test length(state.state_theta) >= K
    if occupied_only
        @test all(>(0), dp.njs)
        @test dp.J == K == length(state.state_theta)
    elseif fixed_J
        @test dp.J == K  # fmn: J is the number of components, occupied or not
    else
        @test dp.J == count(>(0), dp.njs)
    end
end

@testset "labels, counts and member lists are vectors kept in sync" begin
    data = generate_hedonic_data(80, 21)
    raw = CausalMixtures.RawData(data.formulas.formula_y, data.formulas.formula_d, data.df)
    for model in ("dpm", "blocked", "fmn", "marginal")
        seed_global!()
        params = CausalMixtures.InputParams(M=1, scale_data=(true, true), verbose=false, model=model)
        state, input, out = CausalMixtures.dpm_init(raw, setup_default_priors(2, 3, J=8), params)
        @test state.state_dp.labels isa Vector{Int}
        check_state(state, input; occupied_only = model == "dpm", fixed_J = model == "fmn")
        for _ in 1:15
            state, input, out = CausalMixtures.dpm!(state, input, out)
            check_state(state, input; occupied_only = model == "dpm", fixed_J = model == "fmn")
        end
    end
end

@testset "dpm sweep: emptied components are dropped and the rest relabelled in order" begin
    data = generate_hedonic_data(30, 5)
    raw = CausalMixtures.RawData(data.formulas.formula_y, data.formulas.formula_d, data.df)
    params = CausalMixtures.InputParams(M=1, scale_data=(true, true), verbose=false, model="dpm")
    seed_global!()
    state, input, _ = CausalMixtures.dpm_init(raw, setup_default_priors(2, 3, J=3), params)
    ## empty component 2 by hand; compaction keeps components 1 and 3 (as 1 and 2) and their thetas
    th1, th3 = state.state_theta[1], state.state_theta[3]
    dp = state.state_dp
    dp.labels[dp.labels .== 2] .= 1
    dp.njs .= [count(==(j), dp.labels) for j in 1:3]
    CausalMixtures.compact_components!(state)
    @test dp.J == 2 && dp.njs == [count(==(j), dp.labels) for j in 1:2]
    @test state.state_theta[1] === th1 && state.state_theta[2] === th3
    @test all(j -> dp.members[j] == findall(==(j), dp.labels), 1:2)
end

@testset "kernels: dense design, block-identity precision, 3-d log density" begin
    rng = test_rng()
    data = generate_hedonic_data(40, 77)
    raw = CausalMixtures.RawData(data.formulas.formula_y, @formula(D_obs ~ Z + X), data.df)
    params = CausalMixtures.InputParams(M=1, scale_data=(true, true), verbose=false, model="blocked")
    seed_global!()
    state, input, _ = CausalMixtures.dpm_init(raw, setup_default_priors(2, 3, J=4), params)
    n, kx, kz = input.dims.n, input.dims.kx, input.dims.kz
    H = input.data.Hmat
    @test input.data.X ≈ Matrix(H[(n+1):(2n), (kz+1):(kz+kx)])
    @test input.data.W ≈ Matrix(H[1:n, 1:kz])
    beta = randn(rng, kz + 2kx)
    A = randn(rng, 3, 3); S = A * A' + I
    S ./= S[1, 1]
    c = CausalMixtures.Chol3(S)
    @test [c.l11 0 0; c.l21 c.l22 0; c.l31 c.l32 c.l33] ≈ cholesky(S).L
    for i in (1, 17, n)
        Hi = Matrix(H[[i, i + n, i + 2n], :])
        @test collect(CausalMixtures.unit_mean(input.data.X, input.data.W, beta, i)) ≈ Hi * beta
        z = (randn(rng), randn(rng), randn(rng))
        @test CausalMixtures.logpdf_unit(c, beta, input.data.X, input.data.W, z, i) ≈
              logpdf(MvNormal(Hi * beta, S), collect(z))
    end
    ## H_j'(Sinv ⊗ I)H_j and H_j'(Sinv ⊗ I)z_j from cross-products match the dense kron build
    idx = [2, 3, 5, 8, 13, 21, 34]
    Hj = Matrix(H[vcat(idx, idx .+ n, idx .+ 2n), :])
    Sinv = inv(S)
    Wj, Xj = input.data.W[idx, :], input.data.X[idx, :]
    @test CausalMixtures.design_precision(Sinv, Wj' * Wj, Wj' * Xj, Xj' * Xj) ≈ Hj' * kron(Sinv, I(length(idx))) * Hj
    ## the latent-data conditionals are those of N3(H_i beta, Sigma)
    lm = CausalMixtures.LatentMoments(S)
    function gauss_cond(S, a, bs)
        K = S[a, bs]' / S[bs, bs]
        return K, S[a, a] - (K * S[bs, a])[1]
    end
    K, v = gauss_cond(S, 2, [1, 3]); @test [lm.c1d lm.c1y] ≈ K && lm.omega1 ≈ v
    K, v = gauss_cond(S, 3, [1, 2]); @test [lm.c0d lm.c0y] ≈ K && lm.omega0 ≈ v
    K, v = gauss_cond(S, 1, [2, 3]); @test [lm.sd1 lm.sd0] ≈ K && lm.omega_d ≈ v
end

@testset "update_theta! draws from the dense-build conditional of beta given Sigma" begin
    ## fix Sigma by making NobileWishart's draw irrelevant: compare the beta draws' moments with the
    ## dense formula given the Sigma that was drawn (beta | Sigma is exactly normal)
    data = generate_hedonic_data(60, 3)
    raw = CausalMixtures.RawData(data.formulas.formula_y, data.formulas.formula_d, data.df)
    params = CausalMixtures.InputParams(M=1, scale_data=(true, true), verbose=false, model="gaussian")
    seed_global!()
    state, input, _ = CausalMixtures.dpm_init(raw, setup_default_priors(2, 3, J=1), params)
    n = input.dims.n
    idx = collect(1:n)
    H = Matrix(input.data.Hmat)
    zv = vcat(state.state_data.dstar, state.state_data.y1, state.state_data.y0)
    pb = input.priors.prior_theta.prior_beta
    N = 20_000
    B = zeros(N, input.dims.ktot); Z = zeros(N, input.dims.ktot)
    for r in 1:N
        CausalMixtures.update_theta!(Random.default_rng(), state, input, 1, idx)
        th = state.state_theta[1]
        Sinv = inv(th.Sigma)
        V = inv(H' * kron(Sinv, I(n)) * H + pb.V)
        m = V * (H' * kron(Sinv, I(n)) * zv + pb.V * pb.mu)
        Z[r, :] = cholesky(Symmetric(V)).L \ (th.beta - m)   # standard normal if the draw is right
    end
    @test maximum(abs, mean(Z, dims=1)) < 4.5 / sqrt(N)
    @test cov(Z) ≈ I atol = 0.05
end

@testset "explicit rng: reproducible and independent of the global RNG" begin
    data = generate_hedonic_data(60, 9)
    raw = CausalMixtures.RawData(data.formulas.formula_y, data.formulas.formula_d, data.df)
    function run(model, seed, global_seed)
        Random.seed!(global_seed)
        rng = Xoshiro(seed)
        params = CausalMixtures.InputParams(M=6, scale_data=(true, true), verbose=false, model=model)
        priors = setup_default_priors(2, 3, J = model == "gaussian" ? 1 : 5)
        state, input, out = CausalMixtures.dpm!(CausalMixtures.dpm_init(raw, priors, params; rng=rng)...; rng=rng)
        ## continue the chain, then the PPD
        state, input, out = CausalMixtures.dpm!(state, input, out; rng=rng)
        ppd = rand_ppd(rng, out, input, mean_covariates(input)...)
        g = rand()  # the global RNG must be untouched by all of the above
        return [t.beta for th in out.out_theta for t in th], [dp.alpha for dp in out.out_dp], ppd, g
    end
    for model in ("dpm", "blocked", "fmn", "gaussian", "marginal")
        a = run(model, 1, 100)
        b = run(model, 1, 200)
        @test a[1] == b[1] && a[2] == b[2] && a[3] == b[3]
        Random.seed!(100); @test a[4] == rand()
        c = run(model, 2, 100)
        @test a[1] != c[1]
    end
    ## dpm_ppd with an rng
    params = CausalMixtures.InputParams(M=4, scale_data=(true, true), verbose=false, model="dpm")
    rng = Xoshiro(3)
    state, input, out = CausalMixtures.dpm!(CausalMixtures.dpm_init(raw, setup_default_priors(2, 3, J=3), params; rng=rng)...; rng=rng)
    w = mean_covariates(input)[2]
    @test dpm_ppd(Xoshiro(4), out, input, w) == dpm_ppd(Xoshiro(4), out, input, w)
    @test CausalMixtures.sample_logweights(Xoshiro(5), [0.0, 1.0]) == CausalMixtures.sample_logweights(Xoshiro(5), [0.0, 1.0])
end

@testset "thinned output: (theta, n_j, alpha) every thin-th draw, latent data optional" begin
    data = generate_hedonic_data(50, 13)
    raw = CausalMixtures.RawData(data.formulas.formula_y, data.formulas.formula_d, data.df)
    priors = setup_default_priors(2, 3, J=4)
    for model in ("dpm", "blocked", "gaussian")
        pr = model == "gaussian" ? setup_default_priors(2, 3, J=1) : priors
        ## the same chain saved three ways: every draw, thinned, thinned with latent data
        outs = map([(1, 0), (3, 0), (3, 4)]) do (thin, lt)
            params = CausalMixtures.InputParams(M=12, scale_data=(true, true), verbose=false, model=model,
                                                thin=thin, latent_thin=lt)
            rng = Xoshiro(8)
            CausalMixtures.dpm!(CausalMixtures.dpm_init(raw, pr, params; rng=rng)...; rng=rng)
        end
        (_, _, full), (_, _, thinned), (st, _, withlat) = outs
        @test length(full.out_dp) == 12 && isempty(full.out_data) && isempty(full.out_labels)
        @test length(thinned.out_dp) == length(thinned.out_theta) == 4
        @test isempty(thinned.out_data)
        @test [d.alpha for d in thinned.out_dp] == [full.out_dp[m].alpha for m in 3:3:12]
        @test [d.njs for d in thinned.out_dp] == [full.out_dp[m].njs for m in 3:3:12]
        @test all(thinned.out_theta[s][j].beta == full.out_theta[3s][j].beta
                  for s in 1:4 for j in eachindex(thinned.out_theta[s]))
        @test length(withlat.out_data) == length(withlat.out_labels) == 3
        @test withlat.out_data[3].y1 == st.state_data.y1 && withlat.out_labels[3] == st.state_dp.labels
        @test withlat.out_data[1].y1 != withlat.out_data[2].y1
        ## saved draws do not alias the live state
        @test full.out_dp[1] isa CausalMixtures.DrawDP
        @test full.out_dp[1].njs !== full.out_dp[2].njs
    end
    ## continuing a chain appends, with the same thinning
    params = CausalMixtures.InputParams(M=6, scale_data=(true, true), verbose=false, model="blocked", thin=2)
    rng = Xoshiro(1)
    st, inp, out = CausalMixtures.dpm!(CausalMixtures.dpm_init(raw, priors, params; rng=rng)...; rng=rng)
    st, inp, out = CausalMixtures.dpm!(st, inp, out; rng=rng)
    @test length(out.out_dp) == 6
    @test_throws ArgumentError CausalMixtures.InputParams(M=6, thin=0)
end

@testset "every struct field has a concrete type" begin
    M = CausalMixtures
    types = Any[]
    for nm in names(M; all=true)
        isdefined(M, nm) || continue
        T = getfield(M, nm)
        T isa DataType && isstructtype(T) && parentmodule(T) === M && !startswith(string(nm), "#") && push!(types, T)
    end
    ## the parametric ones, as the package instantiates them
    data = generate_hedonic_data(20, 1)
    raw = M.RawData(data.formulas.formula_y, data.formulas.formula_d, data.df)
    params = M.InputParams(M=1, scale_data=(true, true), verbose=false, model="dpm")
    _, input, _ = M.dpm_init(raw, setup_default_priors(2, 3, J=2), params)
    append!(types, [typeof(raw), typeof(input.data.y), M.ScaleData{Matrix{Float64}}])
    @test length(types) > 20
    for T in types, (f, ft) in zip(fieldnames(T), fieldtypes(T))
        @test isconcretetype(ft) || (T, f, ft)
    end
    @test input.data.y isa M.ScaleData{Vector{Float64}}
end

end # Phase 3 refactor
