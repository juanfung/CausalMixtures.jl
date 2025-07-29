using Test
using CausalMixtures
using DataFrames
using StatsModels
using LinearAlgebra

# Test the GibbsOut constructor
# This test checks that the GibbsOut constructor creates a valid GibbsOut object
@testset "GibbsOut constructor" begin
    gout = CausalMixtures.GibbsOut(10)
    @test gout isa CausalMixtures.GibbsOut
    @test length(gout.out_data) == 10
    @test length(gout.out_dp) == 10
    @test length(gout.out_theta) == 10
end

# Test the default GibbsOut constructor
# This test checks that the default GibbsOut constructor creates a valid GibbsOut object
@testset "Default GibbsOut constructor" begin
    gout = CausalMixtures.GibbsOut()
    @test gout isa CausalMixtures.GibbsOut
    @test length(gout.out_data) == 0
    @test length(gout.out_dp) == 0
    @test length(gout.out_theta) == 0
end