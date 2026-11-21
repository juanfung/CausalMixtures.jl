# bench/phase4a_checks.jl -- Phase 4a exit checks (b)-(e) on generate_hedonic_data(500, 1234).
#   julia --project=. bench/phase4a_checks.jl [chains] [M] [burn]
# Configurations, `chains` chains each (seeds SEED + c), scale_data = (true, true), C3 priors:
#   gaussian            the one-component model
#   marginal_k1         Design A, slopes = :cluster, alpha fixed at 1e-12, J = 1: k stays 1, so
#                       the model is the gaussian one ((b))
#   marginal_cluster    Design A, slopes = :cluster ((c), (d))
#   marginal_common     Design A, slopes = :common ((d))
#   blocked             J = 20 ((c))
# Per chain: post-burn-in mean and sd of the per-draw ATE (trace_ate), mean k and alpha, and the
# n_j-weighted Sigma entries; across chains: R-hat and bulk ESS of ATE, k and alpha.
using CausalMixtures, Random, Statistics, Printf

const SEED = 20261121
chains = length(ARGS) > 0 ? parse(Int, ARGS[1]) : 8
M = length(ARGS) > 1 ? parse(Int, ARGS[2]) : 3000
burn = length(ARGS) > 2 ? parse(Int, ARGS[3]) : 1000

data = generate_hedonic_data(500, 1234)
raw = CausalMixtures.RawData(data.formulas.formula_y, data.formulas.formula_d, data.df)

configs = [
    ("gaussian", "gaussian", :cluster, setup_default_priors(2, 3; J=1)),
    ("marginal_k1", "marginal", :cluster, setup_default_priors(2, 3; J=1, alpha=1e-12, alpha_shape=0.0, slopes=:cluster)),
    ("marginal_cluster", "marginal", :cluster, setup_default_priors(2, 3; slopes=:cluster)),
    ("marginal_common", "marginal", :common, setup_default_priors(2, 3; slopes=:common)),
    ("blocked", "blocked", :cluster, setup_default_priors(2, 3; J=20)),
]

function sigma_trace(out, n)
    S = zeros(length(out.out_dp), 5)
    for (m, (dp, th)) in enumerate(zip(out.out_dp, out.out_theta))
        for j in eachindex(dp.njs)
            dp.njs[j] > 0 || continue
            Sg = th[j].Sigma
            S[m, :] .+= dp.njs[j] / n .* [Sg[2,2], Sg[3,3], Sg[1,2], Sg[1,3], Sg[2,3]]
        end
    end
    return S
end

results = Dict{String,Any}()
for (name, model, slopes, pr) in configs
    params = CausalMixtures.InputParams(M=M, scale_data=(true, true), verbose=false, model=model, slopes=slopes)
    ## compile
    let rng = Xoshiro(1), p2 = CausalMixtures.InputParams(M=3, scale_data=(true, true), verbose=false, model=model, slopes=slopes)
        CausalMixtures.dpm!(CausalMixtures.dpm_init(raw, pr, p2; rng=rng)...; rng=rng)
    end
    ates, ks, alphas, sig, secs = Vector{Vector{Float64}}(), Vector{Vector{Float64}}(), Vector{Vector{Float64}}(), Matrix{Float64}[], Float64[]
    input = nothing
    for c in 1:chains
        rng = Xoshiro(SEED + c)
        t = @elapsed (st, input, out) = CausalMixtures.dpm!(CausalMixtures.dpm_init(raw, pr, params; rng=rng)...; rng=rng)
        keep = (burn + 1):M
        push!(ates, trace_ate(out, input)[keep])
        push!(ks, Float64.(trace_k(out)[keep]))
        push!(alphas, trace_alpha(out)[keep])
        push!(sig, sigma_trace(out, input.dims.n)[keep, :])
        push!(secs, t)
        @printf("%-17s chain %d: ATE %.3f (sd %.3f)  k %.2f  alpha %.2f  %.1f s\n", name, c, mean(ates[end]),
                std(ates[end]), mean(ks[end]), mean(alphas[end]), t)
        flush(stdout)
    end
    results[name] = (ates=ates, ks=ks, alphas=alphas, sig=sig, secs=secs)
end

cm(v) = mean(mean.(v))
cse(v) = std(mean.(v)) / sqrt(length(v))
println("\n## Per configuration (", chains, " chains, M = ", M, ", burn-in ", burn, ")\n")
println("| Configuration | ATE mean ± se | ATE sd (mean over chains) | k | alpha | R-hat ATE | R-hat k | R-hat alpha | bulk ESS ATE | ms / iter |")
println("|---|---|---|---|---|---|---|---|---|---|")
fmt(x) = isnan(x) ? "—" : @sprintf("%.3f", x)
for (name, _, _, _) in configs
    r = results[name]
    @printf("| %s | %.3f ± %.3f | %.3f | %.2f | %.2f | %s | %s | %s | %.0f | %.2f |\n", name, cm(r.ates), cse(r.ates),
            mean(std.(r.ates)), cm(r.ks), cm(r.alphas), fmt(rhat(r.ates)), fmt(rhat(r.ks)), fmt(rhat(r.alphas)),
            ess_bulk(r.ates), 1000 * mean(r.secs) / M)
end

function compare(a, b, label)
    ra, rb = results[a], results[b]
    println("\n## ", label, ": ", a, " vs ", b, "\n")
    println("| Statistic | ", a, " | ", b, " | z |")
    println("|---|---|---|---|")
    rows = [("ATE mean", mean.(ra.ates), mean.(rb.ates)), ("ATE sd", std.(ra.ates), std.(rb.ates))]
    for (s, nm) in enumerate(("s22", "s33", "s12", "s13", "s23"))
        push!(rows, (nm, [mean(x[:, s]) for x in ra.sig], [mean(x[:, s]) for x in rb.sig]))
    end
    for (nm, x, y) in rows
        z = (mean(x) - mean(y)) / sqrt(var(x) / length(x) + var(y) / length(y))
        @printf("| %s | %.4f ± %.4f | %.4f ± %.4f | %+.2f |\n", nm, mean(x), std(x) / sqrt(length(x)),
                mean(y), std(y) / sqrt(length(y)), z)
    end
end
compare("marginal_k1", "gaussian", "(b) alpha fixed tiny (k = 1)")
compare("marginal_cluster", "blocked", "(c) Design A (:cluster) vs blocked")
compare("marginal_common", "marginal_cluster", "(d) :common vs :cluster")
