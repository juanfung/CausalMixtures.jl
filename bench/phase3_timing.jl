# bench/phase3_timing.jl -- Phase 3 exit check (b): timing and allocation, n = 500, M = 1000,
# generate_hedonic_data(500, 1234), scale_data = (true, true), Random.seed!(20261121) before init.
# Runs under the old prior C0 (the bench/baseline.md configuration: J = 20, alpha = 1 fixed) and
# the C3 defaults (J = 20, alpha ~ Gamma(2, 2)). One warm-up run, then the median of 3 runs.
# Works on the code before and after the refactor:  julia --project=. bench/phase3_timing.jl [label]
using CausalMixtures, Random, Statistics, Printf

const M = 1000
const N = 500
label = length(ARGS) > 0 ? ARGS[1] : "current"

data = generate_hedonic_data(N, 1234)
raw = CausalMixtures.RawData(data.formulas.formula_y, data.formulas.formula_d, data.df)
priors = Dict("C0" => (J -> setup_default_priors(7; J=J)), "C3" => (J -> setup_default_priors(2, 3; J=J)))

function run(model, prior, M; seed=20261121)
    p = CausalMixtures.InputParams(M=M, scale_data=(true, true), verbose=false, model=model)
    pr = priors[prior](model == "gaussian" ? 1 : 20)
    Random.seed!(seed)
    init = CausalMixtures.dpm_init(raw, pr, p)
    GC.gc()
    stats = @timed CausalMixtures.dpm!(init...)
    out = stats.value[3]
    return (time=stats.time, gctime=stats.gctime, bytes=stats.bytes, meanJ=mean(out.out_dp[m].J for m in 1:M))
end

println("julia ", VERSION, ", threads = ", Threads.nthreads(), ", label = ", label)
println("| Code | Prior | Model | Time (s) | ms / iter | GC | Allocated | Mean J |")
println("|---|---|---|---|---|---|---|---|")
for prior in ("C0", "C3"), model in ("dpm", "blocked", "gaussian")
    run(model, prior, 2)  # compile
    rs = [run(model, prior, M) for _ in 1:3]
    r = rs[sortperm([x.time for x in rs])[2]]
    @printf("| %s | %s | `%s` | %.2f | %.2f | %.1f%% | %.2f GiB | %.2f |\n", label, prior, model, r.time,
            1000r.time / M, 100r.gctime / r.time, r.bytes / 2^30, r.meanJ)
    flush(stdout)
end
