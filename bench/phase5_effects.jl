# bench/phase5_effects.jl -- Phase 5 demonstration: per-draw label-invariant effects with 95%
# credible intervals against the sample and population truths.
#   julia --project=. bench/phase5_effects.jl <data> <sampler> [chains] [M] [burn]
# data: hedonic (generate_hedonic_data(500, 1234)) or twocomp (bench/prior_calibration.jl's
# two-component data, two_component_data(500, Xoshiro(20261121 + 4))).
# sampler: marginal_common, marginal_cluster, gaussian, blocked. C3 priors, R2, hierarchical,
# R3 and the ridge block off (the defaults); 8 chains x 3000 iterations, the first 1000 saved
# draws discarded; latent draws saved every 10th iteration (SATT, SATU). Chain c uses Xoshiro(20261121 + c).
# Population truths: the same generators at n = 10^6.
using CausalMixtures, Random, Statistics, Printf, DataFrames, StatsModels, LinearAlgebra
const CM = CausalMixtures

dname = ARGS[1]; sampler = ARGS[2]
chains = length(ARGS) > 2 ? parse(Int, ARGS[3]) : 8
M = length(ARGS) > 3 ? parse(Int, ARGS[4]) : 3000
burn = length(ARGS) > 4 ? parse(Int, ARGS[5]) : 1000
const SEED = 20261121

function two_component_data(nn, rng)
    x, z = rand(rng, nn), rand(rng, nn)
    c = rand(rng, nn) .< 0.5
    S = [1.0 0.7 -0.7; 0.7 1.0 -0.1; -0.7 -0.1 1.0]
    Ee = (cholesky(S).L * randn(rng, 3, nn))'
    ds = -1 .+ x .+ z .+ Ee[:, 1]
    y1 = 2 .+ 10x .+ 2c .+ Ee[:, 2]
    y0 = 1 .+ 2x .+ 2c .+ Ee[:, 3]
    d = Int.(ds .> 0)
    g = y1 - y0
    return DataFrame(X=x, Z=z, Y_obs=ifelse.(d .== 1, y1, y0), D_obs=d), (ate=mean(g), tt=mean(g[d .== 1]), tut=mean(g[d .== 0]))
end

if dname == "hedonic"
    data = generate_hedonic_data(500, 1234)
    raw = CM.RawData(data.formulas.formula_y, data.formulas.formula_d, data.df)
    samp = (ate=data.true_effects.ate, tt=data.true_effects.tt, tut=data.true_effects.tut)
    big = generate_hedonic_data(1_000_000, 4321)
    pop = (ate=big.true_effects.ate, tt=big.true_effects.tt, tut=big.true_effects.tut)
else
    df2, samp = two_component_data(500, Xoshiro(SEED + 4))
    raw = CM.RawData(@formula(Y_obs ~ X), @formula(D_obs ~ X + Z), df2)
    _, pop = two_component_data(1_000_000, Xoshiro(4321))
end

model, sl = sampler == "marginal_common" ? ("marginal", :common) : sampler == "marginal_cluster" ? ("marginal", :cluster) :
            sampler == "gaussian" ? ("gaussian", :common) : ("blocked", :cluster)
pr = setup_default_priors(2, 3; J = model == "gaussian" ? 1 : 20, slopes=sl)
params = CM.InputParams(M=M, scale_data=(true, true), verbose=false, model=model, slopes=sl, latent_thin=10)
effs = Effects[]
pred = Float64[]
for c in 1:chains
    rng = Xoshiro(SEED + c)
    _, input, out = CM.dpm!(CM.dpm_init(raw, pr, params; rng=rng)...; rng=rng)
    push!(effs, effects(out, input; rng=Xoshiro(SEED + 100 + c), burn=burn))
    if c == 1   # the old predictive route at the covariate means, for contrast
        xb, wb = mean_covariates(input)
        g = dpm_ate(rand_ppd(Xoshiro(7), out, input, xb, wb), input).ate
        append!(pred, g[(burn + 1):end])
    end
end
s = effects_summary(effs)
@printf("%s | %s | sample ATE %.3f, TT %.3f, TUT %.3f | population ATE %.3f, TT %.3f, TUT %.3f\n",
        dname, sampler, samp.ate, samp.tt, samp.tut, pop.ate, pop.tt, pop.tut)
for (k, v) in pairs(s)
    v === nothing && continue
    @printf("  %s: %.3f [%.3f, %.3f] (width %.2f), ESS %.0f, R-hat %.3f\n", k, v.mean, v.q[1], v.q[3], v.q[3] - v.q[1], v.ess_bulk, v.rhat)
end
qp = quantile(pred, [0.025, 0.975])
@printf("  predictive gain at the means (rand_ppd + dpm_ate, chain 1): %.3f [%.3f, %.3f] (width %.2f)\n", mean(pred), qp[1], qp[2], qp[2] - qp[1])
