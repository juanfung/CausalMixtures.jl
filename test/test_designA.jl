# test/test_designA.jl -- Design A sampler (model = "marginal"), Phase 4a unit tests
using Test, CausalMixtures, LinearAlgebra, Random, Statistics, Distributions
isdefined(Main, :TestHelpers) || include("TestHelpers.jl")
using .TestHelpers
const CM = CausalMixtures

@testset "Design A (Phase 4a)" begin

function setup(sl; n=40, seed=3, J=4)
    data = generate_hedonic_data(n, seed)
    raw = CM.RawData(data.formulas.formula_y, data.formulas.formula_d, data.df)
    params = CM.InputParams(M=1, scale_data=(true, true), verbose=false, model="marginal", slopes=sl)
    rng = Xoshiro(11)
    state, input, _ = CM.dpm_init(raw, setup_default_priors(2, 3, J=J, slopes=sl), params; rng=rng)
    ## a few sweeps so that the latent data and clusters are not at their initial values
    D = DesignA(input)
    for _ in 1:5
        CM.marginal_sweep!(rng, state, input, D)
    end
    return state, input, D, rng
end

## dense versions of the cluster and global designs of unit i
Hc_dense(D, i) = CM.cluster_row(D.Wc, D.Xc, i)
Hg_dense(D, i) = CM.cluster_row(D.Wg, D.Xg, i)

@testset "design split and priors ($sl)" for sl in (:common, :cluster)
    state, input, D, _ = setup(sl)
    kx, kz = input.dims.kx, input.dims.kz
    H = Matrix(input.data.Hmat)
    n = input.dims.n
    @test sort(vcat(D.cidx, D.gidx)) == 1:(kz + 2kx)
    pb = input.priors.prior_theta.prior_beta
    D0 = DesignA(input)  # fresh: the fixed prior (hierarchical: (m, T) at the hyperprior means)
    @test D0.Vc ≈ pb.Vcov[D.cidx, D.cidx] && D0.mu_c ≈ pb.mu[D.cidx]
    ## after sweeps, D's intercept block is the state's (m, diag(tau2)); the slopes keep the fixed prior
    h = state.state_hyper
    @test D.mu_c[D.icidx] ≈ h.m && diag(D.Vc)[D.icidx] ≈ h.tau2
    oth = setdiff(1:length(D.cidx), D.icidx)
    @test D.Vc[oth, oth] ≈ pb.Vcov[D.cidx[oth], D.cidx[oth]]
    for i in (1, 7, n)
        Hi = H[[i, i + n, i + 2n], :]
        ## H_i beta = H^c_i beta[cidx] + H^g_i beta[gidx]
        beta = randn(kz + 2kx)
        @test Hc_dense(D, i) * beta[D.cidx] + (isempty(D.gidx) ? zeros(3) : Hg_dense(D, i) * beta[D.gidx]) ≈ Hi * beta
    end
    if sl == :common
        @test D.cidx == [1, kz + 1, kz + kx + 1]                 # intercepts
        @test diag(D0.Vc) ≈ [0.25, 1, 1]
        @test diag(inv(D.Vg_inv)) ≈ [0.25, 0.25, 1, 1]          # probit slopes .5², outcome slopes 1
        ## the shared slopes are the same in every cluster
        @test all(t -> t.beta[D.gidx] == state.state_theta[1].beta[D.gidx], state.state_theta)
    else
        @test isempty(D.gidx) && D.cidx == 1:(kz + 2kx)
    end
end

@testset "statistics: incremental = rebuilt, block identity = dense build ($sl)" for sl in (:common, :cluster)
    state, input, D, rng = setup(sl)
    b = CM.global_slopes(state, D)
    idx = state.state_dp.members[1]
    st = CM.cluster_stats(state, D, b, idx)
    ## add and remove units incrementally
    st2 = CM.ClusterStats(D)
    for i in 1:input.dims.n
        CM.stats_add!(st2, D, i, CM.ztilde(state, D, b, i), 1)
    end
    for i in setdiff(1:input.dims.n, idx)
        CM.stats_add!(st2, D, i, CM.ztilde(state, D, b, i), -1)
    end
    @test st2.n == st.n
    for f in (:WW, :WX, :XX, :Wz, :Xz)
        @test getfield(st2, f) ≈ getfield(st, f) atol = 1e-10
    end
    ## Q and r against the dense sums over the members
    A = randn(rng, 3, 3); Sig = A * A' + I; Sig ./= Sig[1, 1]
    Sinv = inv(Sig)
    Q, r = CM.cluster_Qr(st, D, Sinv)
    Qd, rd = copy(D.Vc_inv), copy(D.Vc_inv_mu)
    for i in idx
        Hc = Hc_dense(D, i)
        Qd += Hc' * Sinv * Hc
        rd += Hc' * Sinv * collect(CM.ztilde(state, D, b, i))
    end
    @test Q ≈ Qd
    @test r ≈ rd
end

@testset "collapsed predictives match Monte Carlo over beta^c ($sl)" for sl in (:common, :cluster)
    state, input, D, rng = setup(sl)
    b = CM.global_slopes(state, D)
    j = 1
    Sig = state.state_theta[j].Sigma
    idx = state.state_dp.members[j]
    i = idx[1]
    st = CM.cluster_stats(state, D, b, idx[2:end])  # the cluster without unit i
    C, m = CM.cluster_post(st, D, inv(Sig))
    z = CM.ztilde(state, D, b, i)
    Hc = Hc_dense(D, i)
    lp = CM.log_predictive(C, m, Sig, Hc, D, i, z)
    ## E over beta^c ~ N(Q^-1 r, Q^-1) of N3(z | H^c beta^c, Sigma)
    N = 200_000
    Vpost = inv(Symmetric(Matrix(C)))
    Lp = cholesky(Symmetric(Vpost)).L
    mc = mean(pdf(MvNormal(Hc * (m + Lp * randn(rng, length(m))), Sig), collect(z)) for _ in 1:N)
    @test isapprox(lp, log(mc); atol=0.02)
    ## auxiliary cluster: E over beta^c ~ N(mu_c, V_c)
    am, ac = D.aux_mean[i], D.aux_cov[i]
    lpa = logpdf(MvNormal(collect(am), Symmetric(Sig + ac)), collect(z))
    mca = mean(pdf(MvNormal(Hc * (D.mu_c + D.Lc * randn(rng, length(D.mu_c))), Sig), collect(z)) for _ in 1:N)
    @test isapprox(lpa, log(mca); atol=0.03)
end

@testset "label-step posterior cache and predictive = cluster_post and the dense predictive ($sl)" for sl in (:common, :cluster)
    state, input, D, rng = setup(sl)
    b = CM.global_slopes(state, D)
    kc = length(D.cidx)
    for j in 1:state.state_dp.J, i in first(state.state_dp.members[j], 2)
        Sig = state.state_theta[j].Sigma
        st = CM.cluster_stats(state, D, b, state.state_dp.members[j])
        C, m = CM.cluster_post(st, D, inv(Sig))
        pc = CM.post_cache!(CM.PostCache(kc), st, D, inv(Sig))
        @test pc.Qinv ≈ inv(Matrix(C)) && pc.Qinv == pc.Qinv'
        @test pc.m ≈ m
        S = zeros(3, 3)
        mu = CM.predictive!(S, pc, Sig, D, i)
        Hc = Hc_dense(D, i)
        @test collect(mu) ≈ Hc * m
        @test S ≈ Sig + Hc * inv(Matrix(C)) * Hc' && S == S'
        z = CM.ztilde(state, D, b, i)
        @test CM.logpdf3(CM.Chol3(S), (collect(z) - collect(mu))...) ≈ CM.log_predictive(C, m, Sig, Hc, D, i, z)
    end
end

@testset "NobileWishart with a precomputed factor = the direct draw" begin
    V = [1.0 0.2 0.1; 0.2 1.53 0.3; 0.1 0.3 1.53]
    F = CM.NobileFactor(V)
    for s in 1:5
        @test CM.NobileWishart(Xoshiro(s), 10, V) == CM.NobileWishart(Xoshiro(s), 10, F)
    end
    @test CM.NobileWishart(Xoshiro(1), 10, F)[1, 1] ≈ 1
end

@testset "beta^c and b draws follow their dense full conditionals ($sl)" for sl in (:common, :cluster)
    state, input, D, rng = setup(sl)
    N = 10_000
    ## beta^c_1 | Sigma_1, z, b
    Zc = zeros(N, length(D.cidx))
    for r in 1:N
        CM.update_marginal_coefs!(rng, state, D)
        th = state.state_theta[1]
        b = CM.global_slopes(state, D)
        Sinv = inv(th.Sigma)
        Q, rr = copy(D.Vc_inv), copy(D.Vc_inv_mu)
        for i in state.state_dp.members[1]
            Hc = Hc_dense(D, i)
            Q += Hc' * Sinv * Hc
            rr += Hc' * Sinv * collect(CM.ztilde(state, D, b, i))
        end
        Zc[r, :] = cholesky(Symmetric(Q)).U * (th.beta[D.cidx] - Q \ rr)  # N(0, I) if right
    end
    @test maximum(abs, mean(Zc, dims=1)) < 4.5 / sqrt(N)
    @test maximum(abs, cov(Zc) - I) < 0.06
    sl == :common || continue
    ## b | beta^c, Sigma, z
    Zg = zeros(N, length(D.gidx))
    sd = state.state_data
    for r in 1:N
        CM.update_global_slopes!(rng, state, D)
        P, s = copy(D.Vg_inv), copy(D.Vg_inv_mu)
        for (j, th) in enumerate(state.state_theta), i in state.state_dp.members[j]
            Sinv = inv(th.Sigma)
            Hg, Hc = Hg_dense(D, i), Hc_dense(D, i)
            u = [sd.dstar[i], sd.y1[i], sd.y0[i]] - Hc * th.beta[D.cidx]
            P += Hg' * Sinv * Hg
            s += Hg' * Sinv * u
        end
        Zg[r, :] = cholesky(Symmetric(P)).U * (CM.global_slopes(state, D) - P \ s)
    end
    @test maximum(abs, mean(Zg, dims=1)) < 4.5 / sqrt(N)
    @test maximum(abs, cov(Zg) - I) < 0.06
end

@testset "alpha fixed tiny: one cluster throughout" begin
    data = generate_hedonic_data(80, 2)
    raw = CM.RawData(data.formulas.formula_y, data.formulas.formula_d, data.df)
    for sl in (:common, :cluster)
        pr = setup_default_priors(2, 3; J=1, alpha=1e-12, alpha_shape=0.0, slopes=sl)
        params = CM.InputParams(M=60, scale_data=(true, true), verbose=false, model="marginal", slopes=sl)
        rng = Xoshiro(4)
        _, input, out = CM.dpm!(CM.dpm_init(raw, pr, params; rng=rng)...; rng=rng)
        @test all(==(1), trace_k(out))
        y = rand_ppd(rng, out, input, mean_covariates(input)...)
        @test size(y) == (3, 60) && all(isfinite, y)
    end
end

@testset "R2 observed-data score = integral of the predictive over the unobserved coordinates" begin
    rng = test_rng()
    ## gd is the global part of the selection index: the cut-off on z~_d is -gd
    for rep in 1:12, d in (0, 1), gd in (0.0, 0.8 * randn(rng))
        A = randn(rng, 3, 3); S = A * A' + 0.3I
        mu = Tuple(randn(rng, 3))
        o = d == 1 ? 2 : 3
        yo = mu[o] + sqrt(S[o, o]) * randn(rng)
        ## y_miss integrates out exactly; integrate the (d*, y_o) bivariate density over the
        ## half-line of d* given by d with the trapezoid rule
        B = MvNormal([mu[1], mu[o]], Symmetric(S[[1, o], [1, o]]))
        sdd = sqrt(S[1, 1])
        lo, hi = d == 1 ? (-gd, mu[1] + 12sdd) : (mu[1] - 12sdd, -gd)
        if hi <= lo
            continue
        end
        g = range(lo, hi, length=400_001)
        f = [pdf(B, [x, yo]) for x in g]
        integral = step(g) * (sum(f) - (f[1] + f[end]) / 2)
        sc = CM.r2_score(mu, S, yo, d, gd)
        @test isapprox(sc, log(integral); atol=1e-6) || (rep, d, gd, sc, log(integral))
    end
    ## far in the tail: logcdf keeps the score finite
    S = [1.0 0.5 0.2; 0.5 1.0 0.1; 0.2 0.1 1.0]
    @test isfinite(CM.r2_score((40.0, 0.0, 0.0), S, 0.0, 0))
    @test CM.r2_score((40.0, 0.0, 0.0), S, 0.0, 0) < -500
end

@testset "R2 draws follow the truncated predictive conditional" begin
    rng = test_rng()
    for (S, mu) in (([1.0 0.6 -0.4; 0.6 1.5 0.2; -0.4 0.2 0.8], (0.3, -0.2, 0.5)),
                    ([1.0 -0.7 0.7; -0.7 1.0 -0.1; 0.7 -0.1 1.0], (-1.0, 1.0, 0.0))), d in (0, 1), gd in (0.0, 0.6, -0.9)
        o, mi = d == 1 ? (2, 3) : (3, 2)
        yo = mu[o] + 0.7
        N = 100_000
        draws = reduce(vcat, [collect(CM.r2_draw(rng, mu, S, yo, d, gd))' for _ in 1:N])
        @test all(draws[:, o] .== yo)
        @test all(d == 1 ? draws[:, 1] .> -gd : draws[:, 1] .< -gd)
        ## reference: (d*, y_miss) | y_o by generic Gaussian conditioning, then rejection on the sign
        u = [1, mi]
        K = S[u, [o]] / S[o, o]
        Cm = vec(collect(mu)[u] .+ K .* (yo - mu[o]))
        Cv = Symmetric(S[u, u] - K * S[[o], u])
        ## accumulate accepted reference draws in batches (the cut-off can sit in the tail)
        R = MvNormal(Cm, Cv)
        acc = Matrix{Float64}[]
        nacc, ntot = 0, 0
        while nacc < N && ntot < 40_000_000
            x = rand(rng, R, 1_000_000)
            keep = x[:, (x[1, :] .> -gd) .== (d == 1)]
            push!(acc, permutedims(keep)); nacc += size(keep, 2); ntot += 1_000_000
        end
        ref = reduce(vcat, acc)
        @test size(ref, 1) > 5000
        X = draws[:, [1, mi]]
        @test maximum(abs, two_sample_zscores(X, ref)) < 4.5
        sq(Y) = (Y .- mean(Y, dims=1)) .^ 2
        @test maximum(abs, two_sample_zscores(sq(X), sq(ref))) < 4.5
        @test abs(cor(X[:, 1], X[:, 2]) - cor(ref[:, 1], ref[:, 2])) < 0.02
    end
end

@testset "label_step = :r2 is the default; :augmented keeps the 4a step" begin
    @test CM.InputParams().label_step == :r2
    @test_throws ArgumentError CM.InputParams(label_step=:nope)
    data = generate_hedonic_data(60, 5)
    raw = CM.RawData(data.formulas.formula_y, data.formulas.formula_d, data.df)
    for ls in (:r2, :augmented)
        params = CM.InputParams(M=20, scale_data=(true, true), verbose=false, model="marginal", label_step=ls)
        rng = Xoshiro(2)
        st, input, out = CM.dpm!(CM.dpm_init(raw, setup_default_priors(2, 3, slopes=:common), params; rng=rng)...; rng=rng)
        ## the observed outcome is never changed and d* keeps the sign of d
        y1, y0 = st.state_data.y1, st.state_data.y0
        @test ifelse.(input.data.d .== 1, y1, y0) == input.data.y.a
        @test all((st.state_data.dstar .> 0) .== (input.data.d .== 1))
    end
end

end # Design A
