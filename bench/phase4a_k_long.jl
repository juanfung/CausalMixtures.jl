# bench/phase4a_k_long.jl -- 4 chains x 30,000 iterations (thin 10) of blocked and Design A (:cluster)
# on generate_hedonic_data(500, 1234): does blocked's k settle where Design A's does?
using CausalMixtures, Random, Statistics, Printf
data = generate_hedonic_data(500, 1234)
raw = CausalMixtures.RawData(data.formulas.formula_y, data.formulas.formula_d, data.df)
M = 30_000
for (name, model, sl, pr) in (("blocked", "blocked", :cluster, setup_default_priors(2, 3; J=20)),
                              ("marginal_cluster", "marginal", :cluster, setup_default_priors(2, 3; slopes=:cluster)))
    ks = Vector{Vector{Float64}}(); as = Vector{Vector{Float64}}()
    for c in 1:4
        rng = Xoshiro(20261121 + c)
        p = CausalMixtures.InputParams(M=M, scale_data=(true, true), verbose=false, model=model, slopes=sl, thin=10)
        _, inp, out = CausalMixtures.dpm!(CausalMixtures.dpm_init(raw, pr, p; rng=rng)...; rng=rng)
        k = Float64.(trace_k(out)); a = trace_alpha(out)
        push!(ks, k[301:end]); push!(as, a[301:end])
        q = length(k) ÷ 4
        @printf("%s chain %d: k by quarter %s  alpha by quarter %s\n", name, c,
                join([@sprintf("%.2f", mean(k[(i-1)*q+1:i*q])) for i in 1:4], " "),
                join([@sprintf("%.2f", mean(a[(i-1)*q+1:i*q])) for i in 1:4], " "))
        flush(stdout)
    end
    @printf("%s: k %.2f (R-hat %.3f), alpha %.2f (R-hat %.3f)\n", name, mean(mean.(ks)), rhat(ks), mean(mean.(as)), rhat(as))
end
