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
    @test D.Vc ≈ pb.Vcov[D.cidx, D.cidx]
    for i in (1, 7, n)
        Hi = H[[i, i + n, i + 2n], :]
        ## H_i beta = H^c_i beta[cidx] + H^g_i beta[gidx]
        beta = randn(kz + 2kx)
        @test Hc_dense(D, i) * beta[D.cidx] + (isempty(D.gidx) ? zeros(3) : Hg_dense(D, i) * beta[D.gidx]) ≈ Hi * beta
    end
    if sl == :common
        @test D.cidx == [1, kz + 1, kz + kx + 1]                 # intercepts
        @test diag(D.Vc) ≈ [0.25, 1, 1]
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

end # Design A
