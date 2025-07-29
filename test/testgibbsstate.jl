using Test
using CausalMixtures
using DataFrames
using StatsModels
using LinearAlgebra

# Test the StateDP constructor
# This test checks that the StateDP constructor creates a valid StateDP object
@testset "StateDP constructor" begin
    state_dp = CausalMixtures.StateDP()
    @test state_dp isa CausalMixtures.StateDP
    @test state_dp.J > 0
    @test state_dp.labels isa Dict
    @test state_dp.njs isa Dict
    @test state_dp.alpha == 1.0
end

# Test the StateSampler constructor
# This test checks that the StateSampler constructor creates a valid StateSampler object
@testset "StateSampler constructor" begin
    state_sampler = CausalMixtures.StateSampler()
    @test state_sampler isa CausalMixtures.StateSampler
    @test state_sampler.batch_m == 0
    @test state_sampler.batch_n == 1
end

# Test the StateTheta constructor
# This test checks that the StateTheta constructor creates a valid StateTheta object
@testset "StateTheta constructor" begin
    state_theta = CausalMixtures.StateTheta()
    @test state_theta isa Dict{Int64, CausalMixtures.Theta}
    for theta in values(state_theta)
        @test theta.beta isa Vector{Float64}
        @test theta.Sigma isa Matrix{Float64}
    end
end

# Test the GibbsState constructor
# This test checks that the GibbsState constructor creates a valid GibbsState object
@testset "GibbsState constructor" begin
    state = CausalMixtures.GibbsState()
    @test state isa CausalMixtures.GibbsState
end

# Test the state_dp field of GibbsState
# This test checks that the state_dp field is initialized correctly
@testset "state_dp field of GibbsState" begin
    # Test with J > 1
    state_dp = CausalMixtures.StateDP(J = 5)
    state = CausalMixtures.GibbsState(state_dp = state_dp)
    @test state.state_dp isa CausalMixtures.StateDP
    @test state.state_dp.J == 5
    @test state.state_dp.labels isa Dict
    @test state.state_dp.njs isa Dict
    
    # Test with J = 1
    state_dp = CausalMixtures.StateDP(J = 1)
    state = CausalMixtures.GibbsState(state_dp = state_dp)
    @test state.state_dp.J == 1
end

# Test the state_sampler field of GibbsState
# This test checks that the state_sampler field is initialized correctly
@testset "state_sampler field of GibbsState" begin
    state = CausalMixtures.GibbsState()
    @test state.state_sampler isa CausalMixtures.StateSampler
    @test state.state_sampler.batch_m == 0
    @test state.state_sampler.batch_n == 1
end

# Test the state_theta field of GibbsState
# This test checks that the state_theta field is initialized correctly
@testset "state_theta field of GibbsState" begin
    state = CausalMixtures.GibbsState()
    @test state.state_theta isa Dict{Int64, CausalMixtures.Theta}
    for theta in values(state.state_theta)
        @test theta.beta isa Vector{Float64}
        @test theta.Sigma isa Matrix{Float64}
    end
end
