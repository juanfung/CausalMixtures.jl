# bench/phase4c_timing.jl -- Phase 4c exit check (5): time per 1000 iterations, n = 500,
# generate_hedonic_data(500, 1234), C3 priors, scale_data = (true, true). Median of 3 runs of
# M = 1000 after a warm-up, each from a fresh init with Xoshiro(20261121). Same protocol as
# bench/phase4a_timing.jl; R3 (sigma23_step) off and on, R2 label step.
using CausalMixtures, Random, Statistics, Printf
data = generate_hedonic_data(500, 1234)
raw = CausalMixtures.RawData(data.formulas.formula_y, data.formulas.formula_d, data.df)
cfgs = [("blocked (J = 20)", "blocked", :cluster, false, setup_default_priors(2, 3)),
        ("marginal, :common, R3 off", "marginal", :common, false, setup_default_priors(2, 3; slopes=:common)),
        ("marginal, :common, R3 on", "marginal", :common, true, setup_default_priors(2, 3; slopes=:common)),
        ("marginal, :cluster, R3 off", "marginal", :cluster, false, setup_default_priors(2, 3; slopes=:cluster)),
        ("marginal, :cluster, R3 on", "marginal", :cluster, true, setup_default_priors(2, 3; slopes=:cluster)),
        ("gaussian, R3 off", "gaussian", :cluster, false, setup_default_priors(2, 3; J=1)),
        ("gaussian, R3 on", "gaussian", :cluster, true, setup_default_priors(2, 3; J=1))]
function run(model, sl, ls, pr, M)
    p = CausalMixtures.InputParams(M=M, scale_data=(true, true), verbose=false, model=model, slopes=sl, sigma23_step=ls)
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
