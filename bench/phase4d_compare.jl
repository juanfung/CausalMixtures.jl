# bench/phase4d_compare.jl -- Phase 4d: hierarchical intercepts on and off, Design A (R2, R3 off),
# both slope modes, on generate_hedonic_data(500, 1234) and on the Phase 2 two-component data
# (bench/prior_calibration.jl: the same generator with both outcome intercepts shifted by +2 for a
# random half of the units; two_component_data(500, Xoshiro(20261121 + 4))). C3 priors.
#   julia --project=. bench/phase4d_compare.jl [chains] [M] [burn]
# Per configuration, over the post-burn-in draws of all chains: ATE mean (sd) and R-hat, bulk ESS
# per 1000 draws, k mean and R-hat, the separated-units share (out_sep) mean, and under
# hierarchical the posterior means of m and tau2.
using CausalMixtures, Random, Statistics, Printf, DataFrames, StatsModels, LinearAlgebra
const CM = CausalMixtures

chains = length(ARGS) > 0 ? parse(Int, ARGS[1]) : 8
M = length(ARGS) > 1 ? parse(Int, ARGS[2]) : 3000
burn = length(ARGS) > 2 ? parse(Int, ARGS[3]) : 1000
const SEED = 20261121

## bench/prior_calibration.jl, unchanged
function two_component_data(nn, rng)
    x, z = rand(rng, nn), rand(rng, nn)
    c = rand(rng, nn) .< 0.5
    S = [1.0 0.7 -0.7; 0.7 1.0 -0.1; -0.7 -0.1 1.0]
    Ee = (cholesky(S).L * randn(rng, 3, nn))'
    ds = -1 .+ x .+ z .+ Ee[:, 1]
    y1 = 2 .+ 10x .+ 2c .+ Ee[:, 2]
    y0 = 1 .+ 2x .+ 2c .+ Ee[:, 3]
    d = Int.(ds .> 0)
    return DataFrame(X=x, Z=z, Y_obs=ifelse.(d .== 1, y1, y0), D_obs=d), mean(y1 - y0)
end

hed = generate_hedonic_data(500, 1234)
df2, ate2 = two_component_data(500, Xoshiro(SEED + 4))
datasets = (("hedonic(500, 1234)", CM.RawData(hed.formulas.formula_y, hed.formulas.formula_d, hed.df), hed.true_effects.ate),
            ("two-component", CM.RawData(@formula(Y_obs ~ X), @formula(D_obs ~ X + Z), df2), ate2))

println("| Data (sample ATE) | Slopes | Hierarchical | ATE mean (sd) | ATE ESS / 1000 | R-hat ATE | k | R-hat k | separated share | m (mean) | tau2 (mean) |")
println("|---|---|---|---|---|---|---|---|---|---|---|")
for (dname, raw, sate) in datasets, sl in (:common, :cluster), hier in (false, true)
    pr = setup_default_priors(2, 3; slopes=sl)
    mk(Mi) = CM.InputParams(M=Mi, scale_data=(true, true), verbose=false, model="marginal", slopes=sl, hierarchical=hier)
    ates, ks, seps, hyp = Vector{Vector{Float64}}(), Vector{Vector{Float64}}(), Float64[], Vector{Float64}[]
    for c in 1:chains
        rng = Xoshiro(SEED + c)
        _, input, out = CM.dpm!(CM.dpm_init(raw, pr, mk(M); rng=rng)...; rng=rng)
        keep = (burn + 1):M
        push!(ates, trace_ate(out, input)[keep]); push!(ks, Float64.(trace_k(out)[keep]))
        append!(seps, out.out_sep[keep])
        hier && append!(hyp, out.out_hyper[keep])
    end
    all_ate = reduce(vcat, ates)
    nd = chains * (M - burn)
    hm = hier ? join([@sprintf("%.2f", x) for x in mean(hyp)[1:3]], ", ") : "—"
    ht = hier ? join([@sprintf("%.2f", x) for x in mean(hyp)[4:6]], ", ") : "—"
    @printf("| %s (%.3f) | %s | %s | %.3f (%.3f) | %.1f | %.3f | %.2f | %.3f | %.3f | %s | %s |\n",
            dname, sate, sl, hier ? "on" : "off", mean(all_ate), std(all_ate), 1000 * ess_bulk(ates) / nd,
            rhat(ates), mean(mean.(ks)), rhat(ks), mean(seps), hm, ht)
    println("  chain means ATE: ", join([@sprintf("%.2f", mean(a)) for a in ates], " "),
            "; chain means k: ", join([@sprintf("%.2f", mean(k)) for k in ks], " "))
    flush(stdout)
end
