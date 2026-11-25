# bench/phase4e_diagnosis.jl -- Phase 4e: what limits the mixing of the ATE? Analysis only.
#   julia --project=. bench/phase4e_diagnosis.jl <dgp> [chains] [M] [burn]
# dgp: base     generate_hedonic_data(500, 1234)
#      noise    outcome block of Sigma scaled by 3.3 (sds 3.3, same correlations): R^2 of Y_obs on
#               (1, X, D, X D) about 0.61 instead of 0.94
#      z03, z3  instrument coefficient (gamma_Z) 0.3 or 3 instead of 1
# Samplers: gaussian (k = 1) and marginal (:common, R2, hierarchical, R3 off), C3 priors,
# scale_data = (true, true), 8 chains x 5000 iterations (1000 burn-in), seeds 20261121 + c.
# Per sampler: bulk ESS per 1000 post-burn-in draws of every scalar parameter (coefficients on the
# standardized scale, unit 1's cluster under marginal), the ATE, and the latent d* and missing
# outcome of six fixed units; lag 1-50 autocorrelations (averaged over chains) of the five
# slowest; the posterior correlations of the five slowest with the ATE.
using CausalMixtures, Random, Statistics, StatsBase, Printf, LinearAlgebra
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
dobs = data.df.D_obs
units = vcat(findall(==(1), dobs)[1:3], findall(==(0), dobs)[1:3])

function run_config(model)
    sl = :common
    pr = setup_default_priors(2, 3; J = model == "gaussian" ? 1 : 20, slopes=sl)
    params = CM.InputParams(M=M, scale_data=(true, true), verbose=false, model=model, slopes=sl, latent_thin=1)
    traces = Dict{String,Vector{Vector{Float64}}}()
    names = String[]
    put!(nm, v) = (haskey(traces, nm) || (traces[nm] = Vector{Float64}[]; push!(names, nm)); push!(traces[nm], v))
    local input
    for c in 1:chains
        rng = Xoshiro(SEED + c)
        _, input, out = CM.dpm!(CM.dpm_init(raw, pr, params; rng=rng)...; rng=rng)
        keep = (burn + 1):M
        kz, kx = input.dims.kz, input.dims.kx
        coef = vcat(["γ[$(n)]" for n in input.data.znames], ["β1[$(n)]" for n in input.data.xnames],
                    ["β0[$(n)]" for n in input.data.xnames])
        c1 = [out.out_labels[s][1] for s in keep]
        th = [out.out_theta[s][c1[t]] for (t, s) in enumerate(keep)]
        put!("ATE", trace_ate(out, input)[keep])
        for (p, nm) in enumerate(coef)
            put!(nm, [t.beta[p] for t in th])
        end
        for (nm, (a, b)) in zip(("σ12", "σ13", "σ22", "σ33", "σ23"), ((1, 2), (1, 3), (2, 2), (3, 3), (2, 3)))
            put!(nm, [t.Sigma[a, b] for t in th])
        end
        if model == "marginal"
            put!("k", Float64.(trace_k(out)[keep]))
            put!("α", trace_alpha(out)[keep])
            for (l, nm) in enumerate(("m_d", "m_1", "m_0", "τ²_d", "τ²_1", "τ²_0"))
                put!(nm, [out.out_hyper[s][l] for s in keep])
            end
        end
        for i in units
            tag = dobs[i] == 1 ? "T" : "C"
            put!("d*[$i,$tag]", [out.out_data[s].dstar[i] for s in keep])
            put!("ymiss[$i,$tag]", [dobs[i] == 1 ? out.out_data[s].y0[i] : out.out_data[s].y1[i] for s in keep])
        end
    end
    nd = chains * (M - burn)
    ess = Dict(nm => 1000 * ess_bulk(traces[nm]) / nd for nm in names if std(reduce(vcat, traces[nm])) > 0)
    return traces, [nm for nm in names if haskey(ess, nm)], ess
end

println("# Phase 4e diagnosis: dgp = $dgp, $chains chains x $M iterations ($burn burn-in)\n")
println("Fixed units (T treated, C control; ymiss is the unobserved outcome): ", join(units, ", "), "\n")
for model in ("gaussian", "marginal")
    traces, names, ess = run_config(model)
    println("## $model\n")
    println("| Parameter | bulk ESS / 1000 | posterior mean | sd |")
    println("|---|---|---|---|")
    for nm in sort(names, by = nm -> ess[nm])
        x = reduce(vcat, traces[nm])
        @printf("| %s | %.1f | %.3f | %.3f |\n", nm, ess[nm], mean(x), std(x))
    end
    slow = sort(filter(!=("ATE"), names), by = nm -> ess[nm])[1:5]
    lags = [1, 2, 3, 5, 10, 20, 30, 40, 50]
    println("\nAutocorrelation (mean over chains), five slowest and the ATE:\n")
    println("| Parameter | ", join(["lag $l" for l in lags], " | "), " |")
    println("|---|", repeat("---|", length(lags)))
    for nm in vcat(slow, "ATE")
        ac = mean(autocor(t, 1:50) for t in traces[nm])
        println("| $nm | ", join([@sprintf("%.3f", ac[l]) for l in lags], " | "), " |")
    end
    sel = vcat("ATE", slow)
    X = hcat([reduce(vcat, traces[nm]) for nm in sel]...)
    C = cor(X)
    println("\nPosterior correlations (pooled draws):\n")
    println("| | ", join(sel, " | "), " |")
    println("|---|", repeat("---|", length(sel)))
    for (a, nm) in enumerate(sel)
        println("| $nm | ", join([@sprintf("%.2f", C[a, b]) for b in eachindex(sel)], " | "), " |")
    end
    println()
    flush(stdout)
end
