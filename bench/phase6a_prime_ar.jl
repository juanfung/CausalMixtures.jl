# bench/phase6a_prime_ar.jl -- Phase 6a': sensitivity of the learned Sigma scale to its hyperprior
# shape a_r (mean fixed at C3's 0.153: b_r = a_r / 0.153), on simulate_selection(1000; :gaussian)
# at r2 = 0.5 and 0.95 (iv = 1), marginal :common, 2 chains x 4000 (first 1000 discarded).
using CausalMixtures, Random, Statistics, Printf
const CM = CausalMixtures
println("| r2 | a_r | r1 post | r0 post | r (data) | k | k (>= 5%) | PATE | PATT | PATU |")
println("|---|---|---|---|---|---|---|---|---|---|")
for r2 in (0.5, 0.95)
    s = simulate_selection(1000; scenario=:gaussian, r2=r2, rng=Xoshiro(11))
    raw = CM.RawData(s.formulas.formula_y, s.formulas.formula_d, s.df)
    rd = auto_sigma_scale(raw).r
    for a in (1.0, 2.0, 5.0)
        effs, rs, ks, k5 = CM.Effects[], Vector{Float64}[], Float64[], Float64[]
        for c in 1:2
            p = CM.InputParams(M=4000, scale_data=(true, true), verbose=false, model="marginal", sigma_scale=:hier,
                               r_shape=a, r_rate=a / 0.153)
            rng = Xoshiro(100 + c)
            _, input, out = CM.dpm!(CM.dpm_init(raw, setup_default_priors(raw), p; rng=rng)...; rng=rng)
            push!(effs, effects(out, input; rng=rng, burn=1000, n_g0=5))
            append!(rs, out.out_r[1001:end])
            append!(ks, [count(>(0), dp.njs) for dp in out.out_dp[1001:end]])
            append!(k5, [count(>=(50), dp.njs) for dp in out.out_dp[1001:end]])
        end
        e = effects_summary(effs)
        f(v) = @sprintf("%.3f [%.3f, %.3f]", v.mean, v.q[1], v.q[3])
        @printf("| %.2f | %.0f | %.3f | %.3f | %.3f, %.3f | %.2f | %.2f | %s | %s | %s |\n", r2, a, mean(x[1] for x in rs), mean(x[2] for x in rs),
                rd[1], rd[2], mean(ks), mean(k5), f(e.pate), f(e.patt), f(e.patu))
        flush(stdout)
    end
    @printf("| %.2f | truth | | | | 1 | 1 | %.3f | %.3f | %.3f |\n", r2, s.truth.population.pate, s.truth.population.patt, s.truth.population.patu)
end
