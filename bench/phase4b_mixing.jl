# bench/phase4b_mixing.jl -- Phase 4b exit check (4): mixing of the Design A sampler with the 4a label
# step (label_step = :augmented) against R2 (label_step = :r2), in both slope modes, on
# generate_hedonic_data(500, 1234), C3 priors, scale_data = (true, true).
#   julia --project=. bench/phase4b_mixing.jl [chains] [M] [burn]
# Per configuration: pooled posterior mean of the per-draw ATE and of k; bulk ESS of ATE per 1000
# post-burn-in draws; R-hat of ATE, k and alpha; and the number of chains stuck in the low-ATE mode:
#   (A) chain mean ATE more than 3 pooled posterior sds below the median of the chain means;
#   (B) chain mean ATE more than 1.5 median within-chain sds below that median (more sensitive).
using CausalMixtures, Random, Statistics, Printf

const SEED = 20261121
chains = length(ARGS) > 0 ? parse(Int, ARGS[1]) : 16
M = length(ARGS) > 1 ? parse(Int, ARGS[2]) : 3000
burn = length(ARGS) > 2 ? parse(Int, ARGS[3]) : 1000

data = generate_hedonic_data(500, 1234)
raw = CausalMixtures.RawData(data.formulas.formula_y, data.formulas.formula_d, data.df)

println("| Label step | Slopes | ATE | k | ATE bulk ESS / 1000 draws | R-hat ATE | R-hat k | R-hat α | stuck (A) | stuck (B) | lowest chain ATE | ms / iter |")
println("|---|---|---|---|---|---|---|---|---|---|---|---|")
for ls in (:augmented, :r2), sl in (:common, :cluster)
    pr = setup_default_priors(2, 3; slopes=sl)
    params = CausalMixtures.InputParams(M=M, scale_data=(true, true), verbose=false, model="marginal",
                                        slopes=sl, label_step=ls)
    let rng = Xoshiro(1), p2 = CausalMixtures.InputParams(M=3, scale_data=(true, true), verbose=false,
                                                          model="marginal", slopes=sl, label_step=ls)
        CausalMixtures.dpm!(CausalMixtures.dpm_init(raw, pr, p2; rng=rng)...; rng=rng)
    end
    ates, ks, als, secs = Vector{Vector{Float64}}(), Vector{Vector{Float64}}(), Vector{Vector{Float64}}(), Float64[]
    for c in 1:chains
        rng = Xoshiro(SEED + c)
        t = @elapsed (_, input, out) = CausalMixtures.dpm!(CausalMixtures.dpm_init(raw, pr, params; rng=rng)...; rng=rng)
        keep = (burn + 1):M
        push!(ates, trace_ate(out, input)[keep]); push!(ks, Float64.(trace_k(out)[keep]))
        push!(als, trace_alpha(out)[keep]); push!(secs, t)
    end
    cmeans = mean.(ates)
    med = median(cmeans)
    pooled_sd = std(reduce(vcat, ates))
    within = median(std.(ates))
    stuckA = count(<(med - 3pooled_sd), cmeans)
    stuckB = count(<(med - 1.5within), cmeans)
    ndraws = chains * (M - burn)
    @printf("| %s | %s | %.3f | %.2f | %.1f | %.3f | %.3f | %.3f | %d | %d | %.2f | %.2f |\n", ls, sl,
            mean(cmeans), mean(mean.(ks)), 1000 * ess_bulk(ates) / ndraws, rhat(ates), rhat(ks), rhat(als),
            stuckA, stuckB, minimum(cmeans), 1000 * mean(secs) / M)
    println("  chain means: ", join([@sprintf("%.2f", x) for x in cmeans], " "))
    flush(stdout)
end
