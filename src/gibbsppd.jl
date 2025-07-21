## --------------------------------------------------------------------------- #
## PPD objects

mutable struct TreatmentEffects
    ate::Vector{Float64}
    tt::Vector{Float64}
end

##TE = TreatmentEffects

TreatmentEffects(; ate=Float64[], tt=Float64[] ) = TreatmentEffects(ate, tt)

mutable struct PosteriorPredictive
    grid::LinRange{Float64}
    ate::Array{Float64}
    tt::Array{Float64}
    late::Array{Float64}
end

PosteriorPredictive(;grid=LinRange(-2,2,2), ate=zeros(2), tt=zeros(2), late=zeros(2)) = PosteriorPredictive(grid, ate, tt, late)


PPD = PosteriorPredictive

## --------------------------------------------------------------------------- #
export TreatmentEffects, PosteriorPredictive, PPD
