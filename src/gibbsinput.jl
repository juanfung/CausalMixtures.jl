## --------------------------------------------------------------------------- #
## Data

## raw data input
struct RawData
    y_form::Formula
    d_form::Formula
    df::DataFrame
end

## standardized data
struct ScaleData
    a::Array{Float64}
    m::Array{Float64,1}
    s::Array{Float64,1}
end

ScaleData(; a=[0.0], m=[0.0], s=[1.0]) = ScaleData(a, m, s)

## dimensions
struct InputDims
    n::Int64
    kx::Int64
    kz::Int64
    ktot::Int64
end

InputDims(; n=1, kx=1, kz=1, ktot=1) = InputDims(n, kx, kz, ktot)

## transformed data for sampler
struct InputData
    y::Union{Vector{Float64},ScaleData}
    d::Vector{Int64}
    lower::Vector{Float64}
    upper::Vector{Float64}
    Hmat::SparseMatrixCSC{Float64,Int64}
end

InputData(; y=[0.0], d=[0], lower=[0.0], upper=[0.0], Hmat=sparse([0.0])) = InputData(y, d, lower, upper, Hmat)
## TODO: define InputData with/without {xmat,zmat}

## --------------------------------------------------------------------------- #
## Parameters

mutable struct InputParams
    M::Int64
    scale_data::Tuple{Bool,Bool}
    verbose::Bool
    model::String
end

InputParams(; M=100, scale_data=(false,false), verbose=true, model="dpm") = InputParams(M, scale_data, verbose, model)
## set default model to dpm
InputParams(M::Int64, scale_data::Tuple{Bool,Bool}, verbose::Bool) = InputParams(M, scale_data, verbose, model="dpm")

## --------------------------------------------------------------------------- #
## Priors

struct PriorDP
    J::Int64
    alpha::Float64
    alpha_shape::Float64
    alpha_rate::Float64
end

PriorDP(; J=2, alpha=1.0, alpha_shape=0.0, alpha_rate=0.0 ) = PriorDP(J, alpha, alpha_shape, alpha_rate)

struct PriorBeta
    mu::Vector{Float64} # prior mean
    V::Matrix{Float64} # prior precision
    Vinv::Bool # if true, precision; else, covariance
end

PriorBeta(; mu=ones(2), V=eye(2), Vinv=true) = PriorBeta(mu, V, Vinv)

struct PriorSigma
    rho::Int64
    R::Matrix{Float64}
end

PriorSigma(; rho=1, R=eye(3) ) = PriorSigma(rho, R)

struct PriorTheta
    prior_beta::PriorBeta
    prior_Sigma::PriorSigma
end

PriorTheta(; prior_beta=PriorBeta(), prior_Sigma=PriorSigma()) = PriorTheta(prior_beta, prior_Sigma)

## collect priors
struct InputPriors
    prior_dp::PriorDP
    prior_theta::PriorTheta
end

InputPriors(; prior_dp = PriorDP(), prior_theta=PriorTheta()) = InputPriors(prior_dp, prior_theta)

## --------------------------------------------------------------------------- #
## collect all inputs

struct GibbsInput
    data::InputData # transformed data
    dims::InputDims # data dimensions
    params::InputParams # sampler parameters
    priors::InputPriors # priors
end

GibbsInput(; data=InputData(), dims=InputDims(), params=InputParams(), priors=InputPriors()) =
    GibbsInput(data, dims, params, priors)

## --------------------------------------------------------------------------- #
export RawRata, ScaleData, InputData, InputPriors, InputParams, InputDims
export PriorDP, PriorBeta, PriorSigma, PriorTheta
export GibbsInput
