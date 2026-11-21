# bench/baseline.jl -- Phase 0 timing baseline (see docs/implementation_plan.md).
#
# Times `dpm` and `blocked` at n = 500, M = 1000 on generate_hedonic_data(500, 1234).
# Run from the repo root:  julia --project=. bench/baseline.jl [M]
using CausalMixtures, Random, Statistics, Printf

const M = length(ARGS) > 0 ? parse(Int, ARGS[1]) : 1000
const N = 500

data = generate_hedonic_data(N, 1234)
raw = CausalMixtures.RawData(data.formulas.formula_y, data.formulas.formula_d, data.df)
priors = setup_default_priors(data.true_params.ktot)

function run(model, M; seed=20261121)
    p = CausalMixtures.InputParams(M=M, scale_data=(true, true), verbose=false, model=model)
    Random.seed!(seed)
    init = CausalMixtures.dpm_init(raw, priors, p)
    GC.gc()
    stats = @timed CausalMixtures.dpm!(init...)
    state, input, out = stats.value
    Js = [out.out_dp[m].J for m in 1:M]
    return (time=stats.time, gctime=stats.gctime, bytes=stats.bytes,
            meanJ=mean(Js), finalJ=Js[end], alpha=out.out_dp[end].alpha)
end

println("julia ", VERSION, ", threads = ", Threads.nthreads())
for model in ("dpm", "blocked")
    run(model, 2)  # compile
    r = run(model, M)
    @printf("%-8s n=%d M=%d  time=%.2f s  (%.2f ms/iter)  gc=%.1f%%  alloc=%.2f GiB  mean J=%.2f  final J=%d\n",
            model, N, M, r.time, 1000r.time / M, 100r.gctime / r.time, r.bytes / 2^30, r.meanJ, r.finalJ)
end
