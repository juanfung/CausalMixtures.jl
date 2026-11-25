# test/test_simulate_selection.jl -- Phase 6a: the simulate_selection DGPs
using Test, CausalMixtures, Random, Statistics, StatsModels
isdefined(Main, :TestHelpers) || include("TestHelpers.jl")
using .TestHelpers

@testset "simulate_selection (Phase 6a)" begin
    @testset "reproducible with an explicit rng; formulas built by name" begin
        a = simulate_selection(300; scenario=:mixture, rng=Xoshiro(5))
        b = simulate_selection(300; scenario=:mixture, rng=Xoshiro(5))
        @test a.df == b.df && a.types == b.types && a.truth.sample == b.truth.sample
        @test a.truth.population == b.truth.population   # independent fixed stream, cached
        @test simulate_selection(300; rng=Xoshiro(6)).df != a.df
        @test string(a.formulas.formula_y) == "Y_obs ~ log_sqft + age + log_lot + bedrooms + income"
        @test occursin("distance", string(a.formulas.formula_d)) && !occursin("distance", string(a.formulas.formula_y))
        @test all(in((0, 1)), a.df.D_obs) && a.df.Y_obs == ifelse.(a.df.D_obs .== 1, a.potential.y1, a.potential.y0)
        @test 11 < mean(a.df.Y_obs) < 13      # log price around 12
    end
    @testset "achieved R^2, treated share, scale ($sc)" for sc in (:gaussian, :mixture, :skewed, :roy, :weak_iv)
        for r2 in (0.5, 0.95)
            s = simulate_selection(40_000; scenario=sc, rng=Xoshiro(1), r2=r2)
            @test abs(s.info.r2_y1 - r2) < 0.03
            sc == :mixture || @test abs(s.info.r2_y0 - r2) < 0.03
            @test 0.3 < s.info.treated_share < 0.6
            @test 0.02 < s.truth.population.ate < 0.1           # realistic effects, log points
        end
        d = simulate_selection(2000; scenario=sc, rng=Xoshiro(2), diagnostic=true)
        @test 0.3 < d.truth.population.ate < 0.7
    end
    @testset "orderings, types and instrument strength" begin
        p = simulate_selection(1000; scenario=:roy, rng=Xoshiro(1)).truth.population
        @test p.att > p.ate > p.atu
        m = simulate_selection(60_000; scenario=:mixture, rng=Xoshiro(3))
        @test sort(unique(m.types)) == [1, 2, 3]
        @test maximum(abs, [mean(m.types .== t) for t in 1:3] .- [0.3, 0.4, 0.3]) < 0.01
        @test 0 < m.truth.population.share_gain < 1                  # winners and losers
        w = simulate_selection(2000; scenario=:weak_iv, rng=Xoshiro(4))
        st = simulate_selection(2000; scenario=:gaussian, rng=Xoshiro(4))
        @test abs(w.info.first_stage_z) < abs(st.info.first_stage_z) / 2
        @test w.info.params.iv == 0.2
        @test st.info.first_stage_F ≈ st.info.first_stage_z^2
    end
    @testset "sample truth is consistent with population truth at large n ($sc)" for sc in (:gaussian, :mixture, :skewed, :roy)
        s = simulate_selection(200_000; scenario=sc, rng=Xoshiro(9))
        q, p = s.truth.sample, s.truth.population
        g = s.potential.y1 .- s.potential.y0
        se = std(g) / sqrt(length(g))
        @test abs(q.ate - p.ate) < 4.5se
        @test abs(q.att - p.att) < 4.5se * 2 && abs(q.atu - p.atu) < 4.5se * 2
        @test abs(q.share_gain - p.share_gain) < 0.01
        @test maximum(abs, q.gain_q .- p.gain_q) < 0.01
        @test p.pate == p.ate && p.patt == p.att && p.patu == p.atu
    end
end
