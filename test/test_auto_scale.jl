# test/test_auto_scale.jl -- Phase 6a: the data-adaptive Sigma scale (setup_default_priors(raw; r = :auto))
using Test, CausalMixtures, LinearAlgebra, Random, Statistics, DataFrames, StatsModels
isdefined(Main, :TestHelpers) || include("TestHelpers.jl")
using .TestHelpers
const CMA = CausalMixtures

@testset "Data-adaptive Sigma scale (Phase 6a)" begin
    @testset "r_k = 0.153 s_k^2 / 0.3 on the standardized scale, clamped" begin
        rng = Xoshiro(1)
        n = 4000
        x, z = randn(rng, n), randn(rng, n)
        d = Int.(0.5x .+ z .+ randn(rng, n) .> 0)
        e = ifelse.(d .== 1, 0.3, 0.6) .* randn(rng, n)          # treated residual sd 0.3, controls 0.6
        df = DataFrame(Y=1 .+ 2x .+ e, D=d, X=x, Z=z)
        raw = CMA.RawData(@formula(Y ~ X), @formula(D ~ X + Z), df)
        a = auto_sigma_scale(raw)
        ## independent computation: per-arm residual variance by normal equations / var(Y)
        for (k, arm) in enumerate((1, 0))
            i = d .== arm
            Xa = hcat(ones(sum(i)), x[i])
            bhat = (Xa' * Xa) \ (Xa' * df.Y[i])
            s2 = sum(abs2, df.Y[i] - Xa * bhat) / (sum(i) - 2) / var(df.Y)
            @test a.s2[k] ≈ s2
            @test a.r[k] ≈ clamp(0.153 * s2 / 0.3, 0.01, 0.5)
        end
        @test a.r[2] / a.r[1] ≈ (0.6 / 0.3)^2 rtol=0.1           # r scales with the residual variance
        ## invariant to the scale of y; the priors carry R = diag(1/rho, r1, r0)
        df10 = copy(df); df10.Y .*= 10
        @test auto_sigma_scale(CMA.RawData(@formula(Y ~ X), @formula(D ~ X + Z), df10)).r == a.r ||
              all(auto_sigma_scale(CMA.RawData(@formula(Y ~ X), @formula(D ~ X + Z), df10)).r .≈ a.r)
        p = setup_default_priors(raw; r=:auto)
        @test p.prior_theta.prior_Sigma.R ≈ Diagonal([0.1, a.r[1], a.r[2]])
        @test setup_default_priors(raw).prior_theta.prior_Sigma.R ≈ Diagonal([0.1, 0.153, 0.153])   # default unchanged
        @test setup_default_priors(2, 3; r=(0.02, 0.04)).prior_theta.prior_Sigma.R ≈ Diagonal([0.1, 0.02, 0.04])
        ## clamping: an almost noiseless outcome hits the floor
        df0 = copy(df); df0.Y = 1 .+ 2x .+ 1e-4 .* randn(rng, n)
        @test auto_sigma_scale(CMA.RawData(@formula(Y ~ X), @formula(D ~ X + Z), df0)).r == (0.01, 0.01)
    end

    @testset ":auto on the hedonic data: PATT and PATU cover both truths (Phase 5b)" begin
        data = generate_hedonic_data(500, 1234)
        raw = CMA.RawData(data.formulas.formula_y, data.formulas.formula_d, data.df)
        pr = setup_default_priors(raw; r=:auto, J=1)
        params = CMA.InputParams(M=20_000, thin=4, scale_data=(true, true), verbose=false, model="gaussian")
        effs = map(1:4) do c
            rng = Xoshiro(20261121 + c)
            _, input, out = CMA.dpm!(CMA.dpm_init(raw, pr, params; rng=rng)...; rng=rng)
            CMA.effects(out, input; rng=rng, burn=1000)   # 4000 iterations
        end
        s = CMA.effects_summary(effs)
        ## sample truth (TT 6.671, TUT 3.565) and population truth (6.528, 3.474; Phase 5)
        for (tr, pop, v) in ((data.true_effects.tt, 6.528, s.patt), (data.true_effects.tut, 3.474, s.patu))
            @test v.q[1] < tr < v.q[3] && v.q[1] < pop < v.q[3]
        end
        @test s.patt.rhat < 1.05 && s.patu.rhat < 1.05 || (s.patt.rhat, s.patu.rhat)
    end
end
