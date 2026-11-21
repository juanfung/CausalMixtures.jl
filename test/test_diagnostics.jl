# test/test_diagnostics.jl -- R-hat and ESS for multi-chain scalar traces
using Test, CausalMixtures, Random, Statistics
isdefined(Main, :TestHelpers) || include("TestHelpers.jl")
using .TestHelpers

@testset "MCMC diagnostics" begin
    rng = test_rng()
    ar1(n, phi, mu=0.0) = (x = zeros(n); x[1] = randn(rng) / sqrt(1 - phi^2);
                           for t in 2:n; x[t] = phi * x[t-1] + randn(rng); end; x .+ mu)
    ## iid chains: R-hat about 1, ESS about the number of draws
    ch = [randn(rng, 2000) for _ in 1:4]
    @test rhat(ch) < 1.01
    @test isapprox(ess_bulk(ch), 8000; rtol=0.15)
    @test isapprox(ess_mean(ch), 8000; rtol=0.15)
    ## AR(1): ESS = N M (1 - phi) / (1 + phi)
    for phi in (0.5, 0.9)
        ch = [ar1(20_000, phi) for _ in 1:4]
        target = 80_000 * (1 - phi) / (1 + phi)
        @test isapprox(ess_mean(ch), target; rtol=0.15)
        @test isapprox(ess_bulk(ch), target; rtol=0.15)
        @test rhat(ch) < 1.01
    end
    ## one chain stuck elsewhere: R-hat flags it
    ch = [ar1(2000, 0.5) for _ in 1:7]
    push!(ch, ar1(2000, 0.5, 2.0))
    @test rhat(ch) > 1.1
    ## a chain that drifts (first half differs from second): split R-hat flags it
    @test rhat([vcat(randn(rng, 1000), randn(rng, 1000) .+ 1.5)]) > 1.1
    ## a different spread only: the folded R-hat flags it
    @test rhat([randn(rng, 2000), 4 .* randn(rng, 2000)]) > 1.05
    ## constant traces
    @test isnan(rhat([ones(100), ones(100)]))
    ## a single vector is one chain
    @test rhat(randn(rng, 1000)) < 1.02

    ## traces from a run
    data = generate_hedonic_data(60, 4)
    raw = CausalMixtures.RawData(data.formulas.formula_y, data.formulas.formula_d, data.df)
    params = CausalMixtures.InputParams(M=30, scale_data=(true, true), verbose=false, model="blocked")
    outs = map(1:2) do c
        r = Xoshiro(c)
        CausalMixtures.dpm!(CausalMixtures.dpm_init(raw, setup_default_priors(2, 3, J=5), params; rng=r)...; rng=r)
    end
    input = outs[1][2]
    @test length(trace_ate(outs[1][3], input)) == 30
    @test trace_k(outs[1][3]) == [count(>(0), d.njs) for d in outs[1][3].out_dp]
    dg = chain_diagnostics([o[3] for o in outs], input; burn=10)
    @test Set(keys(dg)) == Set([:ate, :k, :alpha])
    @test length(dg[:ate].chain_means) == 2 && isfinite(dg[:ate].rhat)
end
