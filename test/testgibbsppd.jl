using Test
using CausalMixtures

# Test the TreatmentEffects constructor
@testset "TreatmentEffects constructor" begin
    # Test that TreatmentEffects constructor creates a valid TreatmentEffects object
    te = CausalMixtures.TreatmentEffects()
    @test te isa CausalMixtures.TreatmentEffects
    @test te.ate isa Vector{Float64}
    @test te.tt isa Vector{Float64}
    
    # Test with keyword arguments
    te = CausalMixtures.TreatmentEffects(ate = [1.0, 2.0], tt = [3.0, 4.0])
    @test te.ate == [1.0, 2.0]
    @test te.tt == [3.0, 4.0]
end

# Test the PosteriorPredictive constructor
@testset "PosteriorPredictive constructor" begin
    # Test that PosteriorPredictive constructor creates a valid PosteriorPredictive object
    ppd = CausalMixtures.PosteriorPredictive()
    @test ppd isa CausalMixtures.PosteriorPredictive
    @test ppd.grid isa LinRange{Float64}
    @test ppd.ate isa Array{Float64}
    @test ppd.tt isa Array{Float64}
    @test ppd.late isa Array{Float64}
    
    # Test with keyword arguments
    grid = LinRange(-2, 2, 10)
    ate = zeros(10)
    tt = zeros(10)
    late = zeros(10)
    ppd = CausalMixtures.PosteriorPredictive(grid = grid, ate = ate, tt = tt, late = late)
    @test ppd.grid == grid
    @test ppd.ate == ate
    @test ppd.tt == tt
    @test ppd.late == late
end