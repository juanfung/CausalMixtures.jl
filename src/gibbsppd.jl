## --------------------------------------------------------------------------- #
## PPD objects

mutable struct TreatmentEffects
    ate::Vector{Float64}
    tt::Vector{Float64}
end

##TE = TreatmentEffects

TreatmentEffects(; ate=Array(Float64,0), tt=Array(Float64,0) ) = TreatmentEffects(ate, tt)

mutable struct PosteriorPredictive
    grid::LinRange{Float64}
    ate::Array{Float64}
    tt::Array{Float64}
    late::Array{Float64}
end

PosteriorPredictive(;grid=linspace(-2,2,2), ate=zeros(2), tt=zeros(2), late=zeros(2)) = PPD(grid, ate, tt, late)

PPD = PosteriorPredictive

## --------------------------------------------------------------------------- #
export TreatmentEffects, PosteriorPredictive, PPD
