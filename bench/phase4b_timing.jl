# bench/phase4b_timing.jl -- Phase 4b exit check (5): time per 1000 iterations, n = 500,
# generate_hedonic_data(500, 1234), C3 priors, scale_data = (true, true). Median of 3 runs of
# M = 1000 after a warm-up, each from a fresh init with Xoshiro(20261121). Same protocol as
# bench/phase4a_timing.jl; the Design A rows use the optimized label step (both label steps).
using CausalMixtures, Random, Statistics, Printf
data = generate_hedonic_data(500, 1234)
raw = CausalMixtures.RawData(data.formulas.formula_y, data.formulas.formula_d, data.df)
cfgs = [("blocked (J = 20)", "blocked", :cluster, :r2, setup_default_priors(2, 3)),
        ("marginal, :common, 4a step (:augmented)", "marginal", :common, :augmented, setup_default_priors(2, 3; slopes=:common)),
        ("marginal, :common, R2", "marginal", :common, :r2, setup_default_priors(2, 3; slopes=:common)),
        ("marginal, :cluster, 4a step (:augmented)", "marginal", :cluster, :augmented, setup_default_priors(2, 3; slopes=:cluster)),
        ("marginal, :cluster, R2", "marginal", :cluster, :r2, setup_default_priors(2, 3; slopes=:cluster))]
function run(model, sl, ls, pr, M)
    p = CausalMixtures.InputParams(M=M, scale_data=(true, true), verbose=false, model=model, slopes=sl, label_step=ls)
    rng = Xoshiro(20261121)
    init = CausalMixtures.dpm_init(raw, pr, p; rng=rng)
    GC.gc()
    s = @timed CausalMixtures.dpm!(init...; rng=rng)
    return s.time, s.bytes, mean(trace_k(s.value[3]))
end
println("| Sampler | s / 1000 iter | Allocated / 1000 iter | mean k |")
println("|---|---|---|---|")
for (name, model, sl, ls, pr) in cfgs
    run(model, sl, ls, pr, 3)
    rs = [run(model, sl, ls, pr, 1000) for _ in 1:3]
    r = rs[sortperm(first.(rs))[2]]
    @printf("| %s | %.2f | %.2f GiB | %.2f |\n", name, r[1], r[2] / 2^30, r[3])
    flush(stdout)
end
