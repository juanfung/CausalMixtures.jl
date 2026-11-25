# test/test_effects.jl -- Phase 5: per-draw label-invariant effects and partition summaries
using Test, CausalMixtures, LinearAlgebra, Random, Statistics, Distributions
isdefined(Main, :TestHelpers) || include("TestHelpers.jl")
using .TestHelpers
const CME = CausalMixtures

## a short run of one sampler on hedonic data
function short_run(model, sl; n=120, M=40, latent_thin=1, seed=3, ys=1.0)
    data = generate_hedonic_data(n, 7)
    df = copy(data.df); df.Y_obs .*= ys
    raw = CME.RawData(data.formulas.formula_y, data.formulas.formula_d, df)
    p = CME.InputParams(M=M, scale_data=(true, true), verbose=false, model=model, slopes=sl, latent_thin=latent_thin)
    rng = Xoshiro(seed)
    pr = setup_default_priors(2, 3; J = model == "gaussian" ? 1 : 10, slopes=sl)
    _, input, out = CME.dpm!(CME.dpm_init(raw, pr, p; rng=rng)...; rng=rng)
    return input, out
end

@testset "Effects (Phase 5)" begin

@testset "closed-form TT and TUT = Monte Carlo over the mixture" begin
    rng = test_rng()
    kz, kx = 2, 2
    x, w = [1.0, 0.4], [1.0, -0.3]
    betas = [[0.2, 0.5, 1.0, 2.0, 0.0, 0.5], [-0.6, 1.0, 3.0, 0.5, 1.0, -1.0], [0.9, -0.4, -1.0, 1.0, 0.5, 2.0]]
    Sigmas = [[1.0 0.6 -0.4; 0.6 1.5 0.2; -0.4 0.2 0.8], [1.0 -0.5 0.3; -0.5 1.0 0.1; 0.3 0.1 2.0],
              [1.0 0.2 0.7; 0.2 0.6 0.0; 0.7 0.0 1.2]]
    pis = [0.5, 0.3, 0.2]
    TT, TUT = CME.te_point(betas, Sigmas, pis, x, w, kz, kx)
    N = 2_000_000
    C = Categorical(pis)
    Ls = [cholesky(S).L for S in Sigmas]
    st, nt, su, nu = 0.0, 0, 0.0, 0
    g = zeros(N)
    for r in 1:N
        j = rand(rng, C)
        b = betas[j]
        e = Ls[j] * randn(rng, 3)
        ds = dot(w, b[1:kz]) + e[1]
        gain = dot(x, b[kz+1:kz+kx] - b[kz+kx+1:end]) + e[2] - e[3]
        if ds > 0
            st += gain; nt += 1
        else
            su += gain; nu += 1
        end
        g[r] = gain
    end
    se = std(g) / sqrt(min(nt, nu))
    @test abs(st / nt - TT) < 4.5se
    @test abs(su / nu - TUT) < 4.5se
end

@testset "k = 1 (gaussian): PATE = x'(beta1 - beta0), Heckman TT and TUT" begin
    input, out = short_run("gaussian", :common)
    e = CME.effects(out, input; rng=Xoshiro(1))
    kz, kx = input.dims.kz, input.dims.kx
    X, W, d, sy = input.data.X, input.data.W, input.data.d, input.data.y.s[1]
    s = length(out.out_dp)
    th = out.out_theta[s][1]
    b, S = th.beta, th.Sigma
    Delta(i) = dot(X[i, :], b[kz+1:kz+kx] - b[kz+kx+1:end])
    idx(i) = dot(W[i, :], b[1:kz])
    lam(v) = pdf(Normal(), v) / cdf(Normal(), v)
    c = S[1, 2] - S[1, 3]
    @test e.pate[s] ≈ sy * mean(Delta(i) for i in 1:input.dims.n)
    @test e.patt[s] ≈ sy * mean(Delta(i) + c * lam(idx(i)) for i in findall(==(1), d))
    @test e.patu[s] ≈ sy * mean(Delta(i) - c * pdf(Normal(), idx(i)) / (1 - cdf(Normal(), idx(i))) for i in findall(==(0), d))
end

@testset "label invariance: permuting the clusters changes nothing" begin
    for sl in (:common, :cluster)
        input, out = short_run("marginal", sl; M=30)
        e1 = CME.effects(out, input; rng=Xoshiro(5), weights=:mean)
        out2 = deepcopy(out)
        for s in eachindex(out2.out_dp)
            dp = out2.out_dp[s]
            K = length(dp.njs)
            perm = shuffle(Xoshiro(s), 1:K)
            out2.out_dp[s] = CME.DrawDP(dp.J, dp.njs[perm], dp.ws, dp.alpha, dp.eta)
            out2.out_theta[s] = out2.out_theta[s][perm]
        end
        e2 = CME.effects(out2, input; rng=Xoshiro(5), weights=:mean)
        @test e1.pate ≈ e2.pate && e1.patt ≈ e2.patt && e1.patu ≈ e2.patu
        @test any(s -> length(out.out_dp[s].njs) > 1, eachindex(out.out_dp)) || sl == :cluster
    end
end

@testset "Dirichlet weights use n_j and alpha" begin
    rng = test_rng()
    njs, a = [30, 12, 5], 0.7
    N = 200_000
    P = reduce(hcat, [CME.draw_weights(rng, njs, a) for _ in 1:N])'
    A = vcat(njs, a); A0 = sum(A)
    m = A ./ A0
    v = A .* (A0 .- A) ./ (A0^2 * (A0 + 1))
    @test maximum(abs, (vec(mean(P, dims=1)) - m) ./ sqrt.(v ./ N)) < 4.5
    @test maximum(abs, vec(var(P, dims=1)) ./ v .- 1) < 0.02
    @test CME.draw_weights(rng, njs, a; weights=:mean) ≈ m
end

@testset "effects are on the original y scale" begin
    i1, o1 = short_run("marginal", :common; M=20)
    i2, o2 = short_run("marginal", :common; M=20, ys=10.0)
    e1 = CME.effects(o1, i1; rng=Xoshiro(2), weights=:mean)
    e2 = CME.effects(o2, i2; rng=Xoshiro(2), weights=:mean)
    @test i2.data.y.s[1] ≈ 10 * i1.data.y.s[1]
    @test e2.pate ≈ 10 .* e1.pate rtol=1e-6
    @test e2.patt ≈ 10 .* e1.patt rtol=1e-6
    @test e2.satt ≈ 10 .* e1.satt rtol=1e-6
end

@testset "per-draw outputs, summary, SATT/SATU and partition summaries" begin
    for (model, sl) in (("marginal", :common), ("gaussian", :common), ("blocked", :cluster), ("dpm", :cluster))
        input, out = short_run(model, sl; M=24, latent_thin=2)
        e = CME.effects(out, input; rng=Xoshiro(1), burn=4)
        @test length(e.pate) == 20 && all(isfinite, e.pate) && all(isfinite, e.patt) && all(isfinite, e.patu)
        @test length(e.satt) == 10      # latent draws every 2nd iteration, after the same burn-in share
        ## SATT recomputed from the saved latent data
        z = out.out_data[end]; d = input.data.d; sy = input.data.y.s[1]
        @test e.satt[end] ≈ sy * mean((z.y1 - z.y0)[d .== 1])
        s1 = CME.effects_summary(e)
        @test s1.pate.q[1] <= s1.pate.q[2] <= s1.pate.q[3] && isnan(s1.pate.rhat)
        s2 = CME.effects_summary([e, e])
        @test s2.pate.mean ≈ s1.pate.mean && isfinite(s2.pate.rhat)
    end
    input, out = short_run("marginal", :common; M=30, latent_thin=1)
    ps = CME.partition_summary(out; burn=10)
    n = input.dims.n
    @test size(ps.psm) == (n, n) && ps.psm == ps.psm' && all(diag(ps.psm) .== 1)
    @test length(ps.point) == n && minimum(ps.point) == 1
    @test sum(values(ps.k)) ≈ 1 && 0 <= ps.separated <= 1
    ## the point partition has the smallest expected Binder loss among the saved ones
    L(c) = sum(abs((c[i] == c[j]) - ps.psm[i, j]) for j in 2:n for i in 1:(j - 1))
    @test ps.binder ≈ L(ps.point)
    @test all(L(out.out_labels[s]) >= ps.binder - 1e-9 for s in 11:30)
    @test_throws ArgumentError CME.partition_summary(short_run("marginal", :common; M=5, latent_thin=0)[2])
end

end # Phase 5
