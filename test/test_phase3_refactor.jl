# test/test_phase3_refactor.jl -- Phase 3 state refactor (docs/implementation_plan.md, section 05)
using Test, CausalMixtures, LinearAlgebra, Random, Statistics, StatsModels
isdefined(Main, :TestHelpers) || include("TestHelpers.jl")
using .TestHelpers

@testset "Phase 3 refactor" begin

## the state invariants every label sweep must leave behind
function check_state(state, input; occupied_only, fixed_J=false)
    dp = state.state_dp
    K = length(dp.njs)
    @test length(dp.labels) == input.dims.n
    @test all(l -> 1 <= l <= K, dp.labels)
    @test dp.njs == [count(==(j), dp.labels) for j in 1:K]
    @test length(dp.members) == K
    @test all(j -> dp.members[j] == findall(==(j), dp.labels), 1:K)
    @test length(state.state_theta) >= K
    if occupied_only
        @test all(>(0), dp.njs)
        @test dp.J == K == length(state.state_theta)
    elseif fixed_J
        @test dp.J == K  # fmn: J is the number of components, occupied or not
    else
        @test dp.J == count(>(0), dp.njs)
    end
end

@testset "labels, counts and member lists are vectors kept in sync" begin
    data = generate_hedonic_data(80, 21)
    raw = CausalMixtures.RawData(data.formulas.formula_y, data.formulas.formula_d, data.df)
    for model in ("dpm", "blocked", "fmn", "marginal")
        seed_global!()
        params = CausalMixtures.InputParams(M=1, scale_data=(true, true), verbose=false, model=model)
        state, input, out = CausalMixtures.dpm_init(raw, setup_default_priors(2, 3, J=8), params)
        @test state.state_dp.labels isa Vector{Int}
        check_state(state, input; occupied_only = model == "dpm", fixed_J = model == "fmn")
        for _ in 1:15
            state, input, out = CausalMixtures.dpm!(state, input, out)
            check_state(state, input; occupied_only = model == "dpm", fixed_J = model == "fmn")
        end
    end
end

@testset "dpm sweep: emptied components are dropped and the rest relabelled in order" begin
    data = generate_hedonic_data(30, 5)
    raw = CausalMixtures.RawData(data.formulas.formula_y, data.formulas.formula_d, data.df)
    params = CausalMixtures.InputParams(M=1, scale_data=(true, true), verbose=false, model="dpm")
    seed_global!()
    state, input, _ = CausalMixtures.dpm_init(raw, setup_default_priors(2, 3, J=3), params)
    ## empty component 2 by hand; compaction keeps components 1 and 3 (as 1 and 2) and their thetas
    th1, th3 = state.state_theta[1], state.state_theta[3]
    dp = state.state_dp
    dp.labels[dp.labels .== 2] .= 1
    dp.njs .= [count(==(j), dp.labels) for j in 1:3]
    CausalMixtures.compact_components!(state)
    @test dp.J == 2 && dp.njs == [count(==(j), dp.labels) for j in 1:2]
    @test state.state_theta[1] === th1 && state.state_theta[2] === th3
    @test all(j -> dp.members[j] == findall(==(j), dp.labels), 1:2)
end

end # Phase 3 refactor
