# bench/phase3_equivalence.jl -- Phase 3 exit check (a): the refactored samplers give the same
# posteriors as before the refactor, under the C3 default priors.
#
#   julia --project=. bench/phase3_equivalence.jl run <label> [chains] [M] [burn]
#       runs gaussian, blocked and dpm on generate_hedonic_data(500, 1234), `chains` chains each
#       (seeds SEED + c), and writes per-chain summaries to bench/phase3_equivalence_<label>.csv
#   julia --project=. bench/phase3_equivalence.jl compare <label1> <label2>
#       prints a markdown table comparing the two runs
#
# Per-draw, label-invariant statistics (scale_data = (true, true); effects on the data scale):
#   ate   = sy * sum_j (n_j / n) xbar'(beta1_j - beta0_j)        (occupied clusters)
#   k     = number of occupied clusters, alpha
#   s22.. = sum_j (n_j / n) Sigma_j[a, b]                         (n_j-weighted Sigma entries)
#   ppd   = dpm_ate(rand_ppd(out, input, xbar, wbar)): predictive ITE draws at the covariate means
# Each chain contributes its post-burn-in mean (and sd for ate and ppd). The comparison uses the
# between-chain spread of the chain means, which covers the slow mixing of k.
using CausalMixtures, Random, Statistics, Printf

const SEED = 20261121
const STATS = ["ate_mean", "ate_sd", "k", "alpha", "s22", "s33", "s12", "s13", "s23", "ppd_mean", "ppd_sd", "secs"]

function chain_summary(model, seed, M, burn)
    data = generate_hedonic_data(500, 1234)
    raw = CausalMixtures.RawData(data.formulas.formula_y, data.formulas.formula_d, data.df)
    priors = setup_default_priors(2, 3; J = model == "gaussian" ? 1 : 20)
    params = CausalMixtures.InputParams(M=M, scale_data=(true, true), verbose=false, model=model)
    Random.seed!(seed)
    t = @elapsed (state, input, out) = CausalMixtures.dpm!(CausalMixtures.dpm_init(raw, priors, params)...)
    n, kx, kz = input.dims.n, input.dims.kx, input.dims.kz
    sy = input.data.y.s[1]
    xbar, wbar = mean_covariates(input)
    i1, i0 = (kz+1):(kz+kx), (kz+kx+1):(kz+2kx)
    keep = (burn+1):length(out.out_dp)
    ate = Float64[]; k = Float64[]; alpha = Float64[]; S = zeros(0, 5)
    for m in keep
        njs, th = out.out_dp[m].njs, out.out_theta[m]
        a = 0.0; s = zeros(5); kk = 0
        for j in 1:length(njs)
            njs[j] > 0 || continue
            kk += 1
            w = njs[j] / n
            b = th[j].beta; Sg = th[j].Sigma
            a += w * (xbar' * (b[i1] - b[i0]))
            s .+= w .* [Sg[2,2], Sg[3,3], Sg[1,2], Sg[1,3], Sg[2,3]]
        end
        push!(ate, sy * a); push!(k, kk); push!(alpha, out.out_dp[m].alpha); S = vcat(S, s')
    end
    Random.seed!(seed + 7)
    sub = CausalMixtures.GibbsOut(length(keep))
    for (r, m) in enumerate(keep)
        sub.out_dp[r] = out.out_dp[m]; sub.out_theta[r] = out.out_theta[m]
    end
    ppd = CausalMixtures.dpm_ate(rand_ppd(sub, input, xbar, wbar), input).ate
    return [mean(ate), std(ate), mean(k), mean(alpha), vec(mean(S, dims=1))..., mean(ppd), std(ppd), t]
end

function run(label, chains, M, burn)
    file = joinpath(@__DIR__, "phase3_equivalence_$(label).csv")
    chain_summary("blocked", SEED, 3, 1)  # compile
    open(file, "w") do io
        println(io, "model,chain,", join(STATS, ","))
        for model in ("gaussian", "blocked", "dpm"), c in 1:chains
            r = chain_summary(model, SEED + c, M, burn)
            println(io, model, ",", c, ",", join((@sprintf("%.6g", x) for x in r), ","))
            flush(io)
            @printf("%-8s chain %d: ate %.3f (sd %.3f)  k %.2f  alpha %.2f  %.1f s\n", model, c, r[1], r[2], r[3], r[4], r[end])
        end
    end
    println("wrote ", file)
end

function load(label)
    lines = readlines(joinpath(@__DIR__, "phase3_equivalence_$(label).csv"))
    hdr = split(lines[1], ",")
    rows = [split(l, ",") for l in lines[2:end]]
    return Dict(m => [parse.(Float64, r[3:end]) for r in rows if r[1] == m] for m in ("gaussian", "blocked", "dpm")), hdr[3:end]
end

function compare(l1, l2)
    d1, hdr = load(l1); d2, _ = load(l2)
    println("| Model | Statistic | $(l1): mean ± se | $(l2): mean ± se | z |")
    println("|---|---|---|---|---|")
    for m in ("gaussian", "blocked", "dpm")
        A = reduce(hcat, d1[m])'; B = reduce(hcat, d2[m])'
        for (s, nm) in enumerate(hdr)
            nm == "secs" && continue
            m == "gaussian" && nm in ("k", "alpha") && continue
            a, b = A[:, s], B[:, s]
            sa, sb = std(a) / sqrt(length(a)), std(b) / sqrt(length(b))
            z = (mean(a) - mean(b)) / sqrt(sa^2 + sb^2)
            @printf("| %s | %s | %.4f ± %.4f | %.4f ± %.4f | %+.2f |\n", m, nm, mean(a), sa, mean(b), sb, z)
        end
    end
    for m in ("gaussian", "blocked", "dpm")
        @printf("%s: mean chain time %.2f s -> %.2f s\n", m, mean(r[end] for r in d1[m]), mean(r[end] for r in d2[m]))
    end
end

if ARGS[1] == "run"
    run(ARGS[2], length(ARGS) > 2 ? parse(Int, ARGS[3]) : 8, length(ARGS) > 3 ? parse(Int, ARGS[4]) : 3000,
        length(ARGS) > 4 ? parse(Int, ARGS[5]) : 1000)
else
    compare(ARGS[2], ARGS[3])
end
