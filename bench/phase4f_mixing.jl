# bench/phase4f_mixing.jl -- Phase 4f: mixing with the ridge block on and off, against the Phase
# 4e baseline (same data variants, samplers and seeds).
#   julia --project=. bench/phase4f_mixing.jl <dgp> [chains] [M] [burn]
# dgp as in bench/phase4e_diagnosis.jl: base, noise (R^2 0.61), z03 (gamma_Z = 0.3), z3 (gamma_Z = 3).
# Samplers: gaussian; marginal :common and :cluster (R2, hierarchical, R3 off). 8 chains x 5000
# iterations (1000 burn-in), seeds 20261121 + c, C3, scale_data = (true, true).
# Per configuration: ATE mean (sd), bulk ESS / 1000 draws, R-hat; stuck chains (chain mean ATE more
# than 3 pooled sds, or 1.5 median within-chain sds, below the median of the chain means);
# bulk ESS / 1000 of the previously slow parameters of unit 1's cluster (sigma12, sigma13, the
# outcome intercepts, sigma23) and the ridge correlations corr(sigma12, beta1 intercept) and
# corr(sigma13, beta0 intercept); k and its R-hat under marginal.
using CausalMixtures, Random, Statistics, Printf, LinearAlgebra
const CM = CausalMixtures

dgp = length(ARGS) > 0 ? ARGS[1] : "base"
chains = length(ARGS) > 1 ? parse(Int, ARGS[2]) : 8
M = length(ARGS) > 2 ? parse(Int, ARGS[3]) : 5000
burn = length(ARGS) > 3 ? parse(Int, ARGS[4]) : 1000
const SEED = 20261121

S0 = [1.0 0.7 -0.7; 0.7 1.0 -0.1; -0.7 -0.1 1.0]
data = if dgp == "base"
    generate_hedonic_data(500, 1234)
elseif dgp == "noise"
    Ds = Diagonal([1, 3.3, 3.3]); generate_hedonic_data(500, 1234; Sigma_true=Matrix(Ds * S0 * Ds))
elseif dgp == "z03"
    generate_hedonic_data(500, 1234; gamma=[-1.0, 1.0, 0.3])
elseif dgp == "z3"
    generate_hedonic_data(500, 1234; gamma=[-1.0, 1.0, 3.0])
else
    error("unknown dgp $dgp")
end
raw = CM.RawData(data.formulas.formula_y, data.formulas.formula_d, data.df)

println("# Phase 4f mixing: dgp = $dgp (sample ATE $(round(data.true_effects.ate, digits=3))), $chains chains x $M iterations ($burn burn-in)\n")
println("| Sampler | Ridge | ATE mean (sd) | ATE ESS / 1000 | R-hat ATE | stuck (A) / (B) | ESS σ12 | ESS σ13 | ESS β1 int | ESS β0 int | ESS σ23 | corr(σ12, β1 int) | corr(σ13, β0 int) | k (R-hat) |")
println("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for (model, sl) in (("gaussian", :cluster), ("marginal", :common), ("marginal", :cluster)), rb in (false, true)
    pr = setup_default_priors(2, 3; J = model == "gaussian" ? 1 : 20, slopes = model == "gaussian" ? :common : sl)
    params = CM.InputParams(M=M, scale_data=(true, true), verbose=false, model=model,
                            slopes = model == "gaussian" ? :common : sl, latent_thin=1, ridge_block=rb)
    tr = Dict(k => Vector{Vector{Float64}}() for k in ("ATE", "s12", "s13", "b1", "b0", "s23", "k"))
    for c in 1:chains
        rng = Xoshiro(SEED + c)
        _, input, out = CM.dpm!(CM.dpm_init(raw, pr, params; rng=rng)...; rng=rng)
        keep = (burn + 1):M
        kz, kx = input.dims.kz, input.dims.kx
        th = [out.out_theta[s][out.out_labels[s][1]] for s in keep]
        push!(tr["ATE"], trace_ate(out, input)[keep])
        push!(tr["s12"], [t.Sigma[1, 2] for t in th]); push!(tr["s13"], [t.Sigma[1, 3] for t in th])
        push!(tr["s23"], [t.Sigma[2, 3] for t in th])
        push!(tr["b1"], [t.beta[kz + 1] for t in th]); push!(tr["b0"], [t.beta[kz + kx + 1] for t in th])
        push!(tr["k"], Float64.(trace_k(out)[keep]))
    end
    nd = chains * (M - burn)
    e(k) = 1000 * ess_bulk(tr[k]) / nd
    pool(k) = reduce(vcat, tr[k])
    a = pool("ATE")
    cm = mean.(tr["ATE"]); med = median(cm)
    stA = count(<(med - 3std(a)), cm); stB = count(<(med - 1.5median(std.(tr["ATE"]))), cm)
    kstr = model == "marginal" ? @sprintf("%.2f (%.3f)", mean(pool("k")), rhat(tr["k"])) : "1"
    @printf("| %s | %s | %.3f (%.3f) | %.1f | %.3f | %d / %d | %.1f | %.1f | %.1f | %.1f | %.1f | %.2f | %.2f | %s |\n",
            model == "gaussian" ? "gaussian" : "marginal $(sl)", rb ? "on" : "off", mean(a), std(a), e("ATE"),
            rhat(tr["ATE"]), stA, stB, e("s12"), e("s13"), e("b1"), e("b0"), e("s23"),
            cor(pool("s12"), pool("b1")), cor(pool("s13"), pool("b0")), kstr)
    flush(stdout)
end
