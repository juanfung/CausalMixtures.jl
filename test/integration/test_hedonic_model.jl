# test/integration/test_hedonic_model.jl -- recovery tests (Phase 6a), replacing the old 30-50
# iteration integration runs. Fixed seeds; simulate_selection(...; diagnostic = true) data.
# Default suite (about 1.5 minutes): gaussian to an ATE ESS of about 200, and a shorter
# marginal :common check. CM_LONG=1 adds the full marginal :common recovery (4 chains x 10k at
# n = 2000, ATE ESS about 200; 5-12 minutes depending on load).
using CausalMixtures, Test, Statistics, Random
const CMI = CausalMixtures

const LONG_RECOVERY = get(ENV, "CM_LONG", "0") == "1"

function recover(model, n, M; chains=4, seed=2026, scenario=:gaussian)
    s = simulate_selection(n; scenario=scenario, diagnostic=true, rng=Xoshiro(seed))
    raw = CMI.RawData(s.formulas.formula_y, s.formulas.formula_d, s.df)
    pr = setup_default_priors(raw; J = model == "gaussian" ? 1 : 20)
    effs = map(1:chains) do c
        p = CMI.InputParams(M=M, thin = max(1, M ÷ 5000), scale_data=(true, true), verbose=false, model=model)
        rng = Xoshiro(seed + 100c)
        _, input, out = CMI.dpm!(CMI.dpm_init(raw, pr, p; rng=rng)...; rng=rng)
        CMI.effects(out, input; rng=rng, burn = length(out.out_dp) ÷ 4)
    end
    return CMI.effects_summary(effs), s.truth.population
end

@testset "Recovery tests" begin
    @testset "legacy generate_hedonic_data still works" begin
        data = CMI.generate_hedonic_data(100, 2099)
        @test size(data.df, 1) == 100 && all(in(names(data.df)), ["Y_obs", "D_obs", "X", "Z"])
        @test data.design_info.n_treated + data.design_info.n_control == 100
    end
    @testset "gaussian recovers the population PATE (n = 2000)" begin
        e, pop = recover("gaussian", 2000, 25_000)
        @test e.pate.q[1] < pop.pate < e.pate.q[3]
        @test e.pate.rhat < 1.05 || e.pate.rhat
        @test e.pate.ess_bulk > 150 || e.pate.ess_bulk
    end
    @testset "marginal :common: PATE interval covers the population truth (n = 1000)" begin
        e, pop = recover("marginal", 1000, 3000)
        @test e.pate.q[1] < pop.pate < e.pate.q[3]
        @test e.pate.rhat < 1.1 || e.pate.rhat
        @test abs(e.pate.mean - pop.pate) < 0.1           # diagnostic effects are 0.5 log points
    end
    if LONG_RECOVERY
        @testset "marginal :common full recovery (CM_LONG=1; n = 2000, 4 x 10k)" begin
            e, pop = recover("marginal", 2000, 10_000)
            @test e.pate.q[1] < pop.pate < e.pate.q[3]
            @test e.pate.rhat < 1.05 || e.pate.rhat
            @test e.pate.ess_bulk > 150 || e.pate.ess_bulk
        end
    end
end
