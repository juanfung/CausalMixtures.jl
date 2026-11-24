# bench/phase4c_mixing.jl -- Phase 4c exit check (4): mixing with and without R3 (the collapsed
# sigma23 step), for Design A (R2 label step) in both slope modes and for gaussian, on
# generate_hedonic_data(500, 1234), C3 priors, scale_data = (true, true).
#   julia --project=. bench/phase4c_mixing.jl [chains] [M] [burn] [seed] [only]
# (seed: chain c uses Xoshiro(seed + c), default 20261121; only: run the configurations whose
# sampler or slope name contains this string)
# Per configuration, over the post-burn-in draws of all chains:
# - ATE: pooled mean and sd, bulk ESS per 1000 draws, R-hat; R-hat of k and alpha (Design A);
# - sigma23 of unit 1's cluster (gaussian: the one Sigma): mean, sd, bulk ESS per 1000 draws;
# - "cond. prior": sigma23's prior conditional given the draw's other Sigma entries (the R3
#   target), averaged over every 10th draw: by the law of total variance its implied mean and sd
#   should match the posterior mean and sd of sigma23, since the likelihood does not involve it;
# - stuck chains: chain mean ATE more than (A) 3 pooled sds or (B) 1.5 median within-chain sds
#   below the median of the chain means.
using CausalMixtures, Random, Statistics, Printf
const CM = CausalMixtures

SEED = length(ARGS) > 3 ? parse(Int, ARGS[4]) : 20261121
ONLY = length(ARGS) > 4 ? ARGS[5] : ""
chains = length(ARGS) > 0 ? parse(Int, ARGS[1]) : 16
M = length(ARGS) > 1 ? parse(Int, ARGS[2]) : 3000
burn = length(ARGS) > 2 ? parse(Int, ARGS[3]) : 1000

data = generate_hedonic_data(500, 1234)
raw = CM.RawData(data.formulas.formula_y, data.formulas.formula_d, data.df)

## mean and variance of sigma23's prior conditional given the other entries of S (grid)
function cond_prior_moments(S, V, nu; G=2001)
    L, U = CM.sigma23_interval(S)
    g = range(L, U, length=G)[2:end-1]
    lf = [CM.sigma23_logdens(S, s, V, nu) for s in g]
    w = exp.(lf .- maximum(lf)); w ./= sum(w)
    m = sum(w .* g)
    return m, sum(w .* (g .- m) .^ 2)
end

fmt(x) = isfinite(x) ? @sprintf("%.3f", x) : "—"

println("| Sampler | Slopes | R3 | ATE mean (sd) | ATE ESS / 1000 | R-hat ATE | R-hat k | R-hat α | k | σ23 mean (sd) | σ23 ESS / 1000 | cond. prior implied σ23 mean (sd) | stuck (A) / (B) | ms / iter |")
println("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for (model, sl, r3) in (("marginal", :common, false), ("marginal", :common, true),
                        ("marginal", :cluster, false), ("marginal", :cluster, true),
                        ("gaussian", :cluster, false), ("gaussian", :cluster, true))
    occursin(ONLY, "$model $sl") || continue
    pr = setup_default_priors(2, 3; J = model == "gaussian" ? 1 : 20, slopes=sl)
    mk(Mi) = CM.InputParams(M=Mi, scale_data=(true, true), verbose=false, model=model, slopes=sl,
                            sigma23_step=r3, latent_thin=1)
    let rng = Xoshiro(1)
        CM.dpm!(CM.dpm_init(raw, pr, mk(3); rng=rng)...; rng=rng)
    end
    ates, ks, als, s23s = [Vector{Vector{Float64}}() for _ in 1:4]
    cps, secs = Matrix{Float64}[], Float64[]
    ps = nothing
    for c in 1:chains
        rng = Xoshiro(SEED + c)
        t = @elapsed (_, input, out) = CM.dpm!(CM.dpm_init(raw, pr, mk(M); rng=rng)...; rng=rng)
        ps = input.priors.prior_theta.prior_Sigma
        keep = (burn + 1):M
        S1 = [out.out_theta[s][out.out_labels[s][1]].Sigma for s in keep]
        push!(ates, trace_ate(out, input)[keep]); push!(ks, Float64.(trace_k(out)[keep]))
        push!(als, trace_alpha(out)[keep]); push!(s23s, [S[2, 3] for S in S1]); push!(secs, t)
        push!(cps, reduce(vcat, [collect(cond_prior_moments(S, ps.rho .* ps.R, ps.rho))' for S in S1[1:10:end]]))
    end
    nd = chains * (M - burn)
    cmeans = mean.(ates); med = median(cmeans)
    all_ate, all_s23 = reduce(vcat, ates), reduce(vcat, s23s)
    stuckA = count(<(med - 3std(all_ate)), cmeans)
    stuckB = count(<(med - 1.5median(std.(ates))), cmeans)
    CP = reduce(vcat, cps)
    cp_mean = mean(CP[:, 1]); cp_sd = sqrt(mean(CP[:, 2]) + var(CP[:, 1]))
    dpm = model == "marginal"
    @printf("| %s | %s | %s | %.3f (%.3f) | %.1f | %.3f | %s | %s | %.2f | %.3f (%.3f) | %.1f | %.3f (%.3f) | %d / %d | %.2f |\n",
            model, dpm ? string(sl) : "—", r3 ? "on" : "off", mean(all_ate), std(all_ate),
            1000 * ess_bulk(ates) / nd, rhat(ates), dpm ? fmt(rhat(ks)) : "—", dpm ? fmt(rhat(als)) : "—",
            mean(mean.(ks)), mean(all_s23), std(all_s23), 1000 * ess_bulk(s23s) / nd, cp_mean, cp_sd,
            stuckA, stuckB, 1000 * mean(secs) / M)
    println("  chain means ATE: ", join([@sprintf("%.2f", x) for x in cmeans], " "))
    flush(stdout)
end
