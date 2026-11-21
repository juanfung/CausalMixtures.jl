# bench/prior_calibration.jl -- Phase 2 prior calibration analysis (docs/priors.md).
#
# Run from the repo root:  julia --project=. bench/prior_calibration.jl [M] [extra]
#   M      sampler iterations for the posterior check (default 6000; burn-in M ÷ 3)
#   extra  "C3" to include the extra candidate defined below
# Prints markdown tables. All simulation uses explicit seeded RNGs; the samplers do not take an
# rng yet (Phase 3), so the posterior check seeds the global RNG before each run.
using CausalMixtures, Random, Statistics, LinearAlgebra, Distributions, DataFrames, StatsModels, Printf

const M_ITER = length(ARGS) > 0 ? parse(Int, ARGS[1]) : 6000
const BURN = M_ITER ÷ 3
const EXTRA = length(ARGS) > 1 ? ARGS[2:end] : String[]
const SEED = 20261121

q(v, p) = quantile(v, p)
f2(x) = @sprintf("%.2f", x)
f3(x) = @sprintf("%.3f", x)
row(xs...) = println("| ", join(xs, " | "), " |")

## --------------------------------------------------------------------------- #
## data on the standardized scale
data = generate_hedonic_data(500, 1234)
raw = CausalMixtures.RawData(@formula(Y_obs ~ X), @formula(D_obs ~ X + Z), data.df)
Random.seed!(SEED)
_, inp, _ = CausalMixtures.dpm_init(raw, setup_default_priors(7, J=1),
                                    CausalMixtures.InputParams(M=1, scale_data=(true, true), verbose=false, model="gaussian"))
n, kx, kw = inp.dims.n, inp.dims.kx, inp.dims.kz
H = inp.data.Hmat
W = Matrix(H[1:n, 1:kw])
X = Matrix(H[(n+1):(2n), (kw+1):(kw+kx)])
ys = inp.data.y.a
dvec = inp.data.d
println("data: n = $n, kx = $kx ($(inp.data.xnames)), kw = $kw ($(inp.data.znames)); ",
        "sd(y_raw) = $(f3(inp.data.y.s[1])); treated share = $(f3(mean(dvec)))")

## --------------------------------------------------------------------------- #
## candidates; r chosen so that the prior median of a cluster's Var(y1) is 0.3
## (sigma22 scales linearly in r when R = diag(1/rho, r, r))
function r_for_median(rho, target; S=200_000, rng=Xoshiro(SEED))
    sig = draw_sigmas(PriorSpec(rho=rho, r=1.0), S; rng=rng)
    return target / median(s[2, 2] for s in sig)
end
r6, r10 = r_for_median(6, 0.3), r_for_median(10, 0.3)
println("r for median Var(y1) = 0.3: rho = 6 -> r = $(f3(r6)); rho = 10 -> r = $(f3(r10))")

C0 = PriorSpec(name="C0", sd_intercept=10, sd_probit=10, sd_outcome=10, rho=6, r=2.0, alpha=1.0)
C1 = PriorSpec(name="C1", sd_intercept=1, sd_probit=1, sd_outcome=1, rho=6, r=r6, alpha_shape=2.0, alpha_rate=2.0)
C2 = PriorSpec(name="C2", sd_intercept=1, sd_probit=1, sd_outcome=0.5, rho=10, r=r10, alpha_shape=2.0, alpha_rate=2.0)
cands = [C0, C1, C2]
include(joinpath(@__DIR__, "prior_calibration_extra.jl"))   # defines extra_candidates(r6, r10)
append!(cands, extra_candidates(r6, r10, EXTRA))

## --------------------------------------------------------------------------- #
println("\n## (a)-(c) prior predictive, 1000 replicates of n = $n")
println()
row("Candidate", "slopes", "prop < .05 or > .95", "clusters with mean prop < .05 or > .95", "median Var(y1) [5%, 95%]", "median Var(y0) [5%, 95%]",
    "sd of cluster mean y1", "sd of cluster mean y0", "cluster effect xbar'(b1-b0) [5%, 95%]",
    "median sd(y_obs) [5%, 95%]", "treated share [5%, 95%]")
row(fill("---", 11)...)
pp_store = Dict{Tuple{String,Symbol},Any}()
for s in cands, slopes in (:cluster, :common)
    sp = PriorSpec(s; slopes=slopes)
    pp = prior_predictive(sp, X, W; reps=1000, rng=Xoshiro(SEED))
    pp_store[(s.name, slopes)] = pp
    ext = mean((pp.prop .< 0.05) .| (pp.prop .> 0.95))
    cext = mean((pp.cluster_prop .< 0.05) .| (pp.cluster_prop .> 0.95))
    row(s.name, string(slopes), f2(ext), f2(cext),
        "$(f2(median(pp.s22))) [$(f2(q(pp.s22, .05))), $(f2(q(pp.s22, .95)))]",
        "$(f2(median(pp.s33))) [$(f2(q(pp.s33, .05))), $(f2(q(pp.s33, .95)))]",
        f2(std(pp.mu1)), f2(std(pp.mu0)),
        "$(f2(q(pp.effect, .05))), $(f2(q(pp.effect, .95)))",
        "$(f2(median(pp.sd_y))) [$(f2(q(pp.sd_y, .05))), $(f2(q(pp.sd_y, .95)))]",
        "$(f2(q(pp.share_treated, .05))), $(f2(q(pp.share_treated, .95)))")
end
println("\nObserved: sd(y_obs) = 1 (standardized), treated share = $(f3(mean(dvec))).")

## --------------------------------------------------------------------------- #
## (d) new-cluster penalty
rng = Xoshiro(SEED + 1)
dstar = [rand(rng, truncated(Normal(), di == 1 ? 0.0 : -Inf, di == 1 ? Inf : 0.0)) for di in dvec]
t1, t0 = findall(==(1), dvec), findall(==(0), dvec)
bh1 = X[t1, :] \ ys[t1]; s1 = std(ys[t1] - X[t1, :] * bh1)
bh0 = X[t0, :] \ ys[t0]; s0 = std(ys[t0] - X[t0, :] * bh0)
y1c = [dvec[i] == 1 ? ys[i] : dot(X[i, :], bh1) + s1 * randn(rng) for i in 1:n]
y0c = [dvec[i] == 0 ? ys[i] : dot(X[i, :], bh0) + s0 * randn(rng) for i in 1:n]
gh = W \ dstar; b1h = X \ y1c; b0h = X \ y0c
E = hcat(dstar - W * gh, y1c - X * b1h, y0c - X * b0h)
Sh = Symmetric(E' * E / n)
bshared = vcat(gh[2:end], b1h[2:end], b0h[2:end])
idx = randperm(rng, n)[1:200]
Z = hcat(dstar, y1c, y0c)
lp_exist = [logpdf(MvNormal([dot(W[i, :], gh), dot(X[i, :], b1h), dot(X[i, :], b0h)], Sh), Z[i, :]) for i in idx]
println("\n## (d) new-cluster penalty, 200 data points")
println("\nSingle-Gaussian fit: Sigma-hat = ", round.(Matrix(Sh), digits=3),
        "; median log N(z_i | H_i beta-hat, Sigma-hat) = $(f2(median(lp_exist)))\n")
row("Candidate", "slopes", "median log p(z_i given G0)", "median gap", "gap [10%, 90%]",
    "log odds new vs one cluster of 500, alpha = 1")
row(fill("---", 6)...)
gaps = Dict{Tuple{String,Symbol},Vector{Float64}}()
for s in cands, slopes in (:cluster, :common)
    sp = PriorSpec(s; slopes=slopes)
    sig = draw_sigmas(sp, 2000; rng=Xoshiro(SEED + 2))
    lp_new = [g0_logpredictive(sp, Z[i, :], X[i, :], W[i, :], sig; b=bshared) for i in idx]
    g = lp_exist .- lp_new
    gaps[(s.name, slopes)] = g
    row(s.name, string(slopes), f2(median(lp_new)), f2(median(g)), "$(f2(q(g, .1))), $(f2(q(g, .9)))",
        f2(log(1 / 500) - median(g)))
end

## --------------------------------------------------------------------------- #
## (e) CRP prior on k
println("\n## (e) CRP prior on k\n")
row("alpha prior", "n", "E[k]", "median k", "k [5%, 95%]")
row(fill("---", 5)...)
for (lab, s) in (("alpha = 1 fixed", PriorSpec(alpha=1.0)), ("alpha ~ Gamma(2, 2)", PriorSpec(alpha_shape=2.0, alpha_rate=2.0))),
    nn in (500, 2000)
    ks = crp_num_clusters(s, nn; reps=4000, rng=Xoshiro(SEED + 3))
    row(lab, nn, f2(mean(ks)), median(ks), "$(q(ks, .05)), $(q(ks, .95))")
end

## --------------------------------------------------------------------------- #
## (f) posterior check with the blocked sampler (slopes = :cluster)
function to_inputpriors(s::PriorSpec, kx, kw; J=20)
    V0 = g0_V0(s, kx, kw)
    pb = CausalMixtures.PriorBeta(mu=zeros(length(V0)), V=Matrix(Diagonal(1 ./ V0)), Vinv=true)
    ps = CausalMixtures.PriorSigma(rho=s.rho, R=s.R)
    a0 = s.alpha_shape > 0 ? s.alpha_shape / s.alpha_rate : s.alpha
    pd = CausalMixtures.PriorDP(J=J, alpha=a0, alpha_shape=s.alpha_shape, alpha_rate=s.alpha_rate)
    return CausalMixtures.InputPriors(pd, CausalMixtures.PriorTheta(pb, ps))
end

function two_component_data(nn, rng)
    x, z = rand(rng, nn), rand(rng, nn)
    c = rand(rng, nn) .< 0.5
    S = [1.0 0.7 -0.7; 0.7 1.0 -0.1; -0.7 -0.1 1.0]
    Ee = (cholesky(S).L * randn(rng, 3, nn))'
    ds = -1 .+ x .+ z .+ Ee[:, 1]
    y1 = 2 .+ 10x .+ 2c .+ Ee[:, 2]        # component 2 shifts both outcome intercepts by 2 error sds
    y0 = 1 .+ 2x .+ 2c .+ Ee[:, 3]
    d = Int.(ds .> 0)
    return DataFrame(X=x, Z=z, Y_obs=ifelse.(d .== 1, y1, y0), D_obs=d), mean(y1 - y0)
end

function run_blocked(df, s::PriorSpec; seed=SEED)
    rawd = CausalMixtures.RawData(@formula(Y_obs ~ X), @formula(D_obs ~ X + Z), df)
    Random.seed!(seed)
    st, ip, ot = CausalMixtures.dpm_init(rawd, to_inputpriors(s, 2, 3),
                     CausalMixtures.InputParams(M=M_ITER, scale_data=(true, true), verbose=false, model="blocked"))
    t = @elapsed (st, ip, ot = CausalMixtures.dpm!(st, ip, ot))
    kz, kxx = ip.dims.kz, ip.dims.kx
    xb = mean_covariates(ip)[1]
    sy = ip.data.y.s[1]
    occ, ateG, ateO, alph, sep = Float64[], Float64[], Float64[], Float64[], Float64[]
    dv = ip.data.d
    nn = ip.dims.n
    for m in (BURN+1):M_ITER
        dp, th = ot.out_dp[m], ot.out_theta[m]
        eff(j) = dot(xb, th[j].beta[kz+1:kz+kxx] - th[j].beta[kz+kxx+1:end])
        push!(occ, count(>(0), values(dp.njs)))
        push!(ateG, sy * sum(dp.ws[j].w * eff(j) for j in keys(dp.ws)))
        push!(ateO, sy * sum(dp.njs[j] / ip.dims.n * eff(j) for j in keys(dp.njs) if dp.njs[j] > 0))
        push!(alph, dp.alpha)
        ## share of units in clusters that are < 5% or > 95% treated (missing arm prior-driven)
        tr = Dict{Int,Int}()
        for i in 1:nn
            tr[dp.labels[i]] = get(tr, dp.labels[i], 0) + dv[i]
        end
        push!(sep, sum(dp.njs[j] for j in keys(tr) if !(0.05 <= tr[j] / dp.njs[j] <= 0.95); init=0) / nn)
    end
    return (occ=occ, ateG=ateG, ateO=ateO, alpha=alph, sep=sep, time=t)
end

df2, ate2 = two_component_data(500, Xoshiro(SEED + 4))
println("\n## (f) blocked sampler posterior, J = 20, M = $M_ITER, burn-in $BURN\n")
println("True ATE: hedonic sample $(f3(data.true_effects.ate)) (population 5); two-component sample $(f3(ate2)) (population 5).\n")
row("Data", "Candidate", "mean occupied k", "k [5%, 95%]", "mean alpha", "ATE (G weights) mean [2.5%, 97.5%]",
    "ATE (occupied, n_j/n) mean [2.5%, 97.5%]", "ATE (occupied) 1st / 2nd half", "units in separated clusters", "time (s)")
row(fill("---", 10)...)
for (dname, df) in (("hedonic(500, 1234)", data.df), ("two-component", df2)), s in cands
    r = run_blocked(df, s)
    row(dname, s.name, f2(mean(r.occ)), "$(round(Int, q(r.occ, .05))), $(round(Int, q(r.occ, .95)))", f2(mean(r.alpha)),
        "$(f2(mean(r.ateG))) [$(f2(q(r.ateG, .025))), $(f2(q(r.ateG, .975)))]",
        "$(f2(mean(r.ateO))) [$(f2(q(r.ateO, .025))), $(f2(q(r.ateO, .975)))]",
        "$(f2(mean(r.ateO[1:end÷2]))) / $(f2(mean(r.ateO[end÷2+1:end])))", f2(mean(r.sep)),
        @sprintf("%.1f", r.time))
end
