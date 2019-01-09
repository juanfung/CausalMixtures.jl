## Module for causal inference via mixture models

##__precompile__()

module CausalMixtures

## load packages
using DataFrames, DataStructures, Distributions, SparseArrays, StatsModels
using JLD, HDF5
## using Rebugger

import Base.+, Base.-, Base.*

## --------------------------------------------------------------------------- #
## Objects:
## GibbsInput: {Data, Params, Priors}
## GibbsState: {DP, Theta, LatentData}
## GibbsOut: Array{DP[m], Theta[m], LatentData[m], m=1:M}

## where

## DP: {J, labels, njs, alpha, eta}, where
##      J = num active components
##      labels = component memberships
##      njs = counts of component membership
## Theta: {Beta, Sigma}
## LatentData: {dstar, y1, y0}
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
## State

mutable struct BlockedWeights
    w::Float64
    v::Float64
    ##nj::Int64
    BlockedWeights(w::Float64, v::Float64) = new(w, v)
end
BlockedWeights(; w=1.0, v=1.0) = BlockedWeights(w, v)

## extend sum and prodcut methods:
+(a::Real, b::BlockedWeights) = BlockedWeights(a+b.w, a+b.v)
+(b1::BlockedWeights, b2::BlockedWeights) = BlockedWeights(b1.w+b2.w, b1.v+b2.v)

-(a::Real, b::BlockedWeights) = BlockedWeights(a-b.w, a-b.v)
-(b1::BlockedWeights, b2::BlockedWeights) = BlockedWeights(b1.w+b2.w, b1.v+b2.v)

*(a::Real, b::BlockedWeights) = BlockedWeights(a*b.w, a*b.v)
*(b1::BlockedWeights, b2::BlockedWeights) = BlockedWeights(b1.w*b2.w, b1.v*b2.v)

function sumbw{T<:Associative}(b::Base.ValueIterator{T}; i::Int64=1)
    s = reduce(+, b)
    if i == 1 return s.w else return s.v end
end

## collect DP state
mutable struct StateDP
    J::Int64 # number of active components
    labels::Dict{Int64,Int64} # individual component membership
    njs::Dict{Int64,Int64} # component membership counts
    ws::DataStructures.OrderedDict{Int64,BlockedWeights} # stick-breaking weights
    alpha::Float64 # concentration parameter
    eta::Float64 # auxiliary variable for sampling alpha
    StateDP(J::Int64, labels::Dict{Int64,Int64}, njs::Dict{Int64,Int64}, ws::DataStructures.OrderedDict{Int64,BlockedWeights}, alpha::Float64, eta::Float64) = new(J, labels, njs, ws, alpha, eta)
end

## outer constructor
StateDP(; J=2,
        labels=Dict{Int64,Int64}(),
        njs=Dict{Int64,Int64}(),
        ws=DataStructures.OrderedDict{Int64,BlockedWeights}(),
        alpha=1.0,
        eta=0.0) = StateDP(J, labels, njs, ws, alpha, eta)

## default constructor
##StateDP(; J=2, labels=Dict{Int64,Int64}(), njs=Dict{Int64,Int64}(), alpha=1.0, eta=0.0) =
##    StateDP(J, labels, njs, alpha, eta)

## copy method
Base.copy(m::StateDP) = StateDP([ copy(getfield(m, k)) for k = 1:length(fieldnames(m)) ]...)

## collect theta state, where theta = beta-Sigma pair
mutable struct Theta
    beta::Vector{Float64} # latent data mean vector
    Sigma::Matrix{Float64} # latent data covariance matrix
end

Theta(; beta=zeros(2), Sigma=eye(2)) = Theta(beta,Sigma)

## collect thetas for all components
##type StateTheta
##    state_theta::Dict{Int64,Theta}
##end

##StateTheta(; state_theta=Dict{Int64,Theta}() ) = StateTheta(state_theta)
typealias StateTheta Dict{Int64,Theta}

mutable struct NjTheta
    nj::Int64
    theta::Theta    
end

NjTheta(; nj=0, theta=Theta()) = NjTheta(nj, theta)

typealias StateNjTheta Dict{Int64,NjTheta}

## collect latent data
mutable struct StateData
    dstar::Vector{Float64} # latent selection outcome
    y1::Vector{Float64} # latent treatment outcome
    y0::Vector{Float64} # latent control outcome
end

StateData(; dstar=[0.0], y1=[0.0], y0=[0.0]) = StateData(dstar, y1, y0)
Base.copy(m::StateData) = StateData([ copy(getfield(m, k)) for k = 1:length(fieldnames(m)) ]...)

## collect sampler state
mutable struct StateSampler
    chain::Bool # is current chain continuing from previous chain?
    batch_n::Int64 # current batch number
    batch_m::Int64 # current batch iteration
    batch_1::Int64 # initial batch iteration
    Vmu::Vector{Float64} #
    zdenom::Float64 # 
    ##rng::AbstractRNG # rng state
end

StateSampler(; chain=false, batch_n=1, batch_m=0, batch_1=1, Vmu=[0.0], zdenom=0.0) =
    StateSampler(chain, batch_n, batch_m, batch_1, Vmu, zdenom)

## current state
mutable struct GibbsState
    state_data::StateData
    state_dp::StateDP
    state_theta::StateTheta
    state_sampler::StateSampler
end

GibbsState(; state_data=StateData(), state_dp=StateDP(), state_theta=StateTheta(), state_sampler=StateSampler()) =
    GibbsState(state_data, state_dp, state_theta, state_sampler)


## --------------------------------------------------------------------------- #
## Output

mutable struct GibbsOut
    out_data::Array{StateData}
    out_dp::Array{StateDP}
    out_theta::Array{StateTheta}
end

GibbsOut(; out_data=Array(StateData,0), out_dp=Array(StateDP,0), out_theta=Array(StateTheta,0)) =
    GibbsOut(out_data, out_dp, out_theta)

## generate GibbsOut given num iterations, M (or input.params?)
GibbsOut(M::Int64) = GibbsOut(Array(StateData, M), Array(StateDP, M), Array(StateTheta, M))

## TODO: output as subtype of AbstractArray
## TODO: DO NOT SAVE state_sampler ...
##typealias GibbsOut Array{GibbsState, 1}

## ---------------------------------------------------------------------------#
## collect {State, Input, Out} into tuple

##type GibbsTuple
##    state::GibbsState
##    input::GibbsInput
##    out::GibbsOut
##end

##GibbsTuple(; state=GibbsState(), input=GibbsInput(), out=GibbsOut()) = GibbsTuple(state, input, out)

typealias GibbsTuple Tuple{GibbsState, GibbsInput, GibbsOut}
##GibbsTuple(; state=GibbsState(), input=GibbsInput(), out=GibbsOut()) = GibbsTuple(state, input, out)

## --------------------------------------------------------------------------- #
## PPD objects

mutable struct TreatmentEffects
    ate::Vector{Float64}
    tt::Vector{Float64}
end

##typealias TE TreatmentEffects

TreatmentEffects(; ate=Array(Float64,0), tt=Array(Float64,0) ) = TreatmentEffects(ate, tt)

mutable struct PosteriorPredictive
    grid::LinSpace{Float64}
    ate::Array{Float64}
    tt::Array{Float64}
    late::Array{Float64}
end

typealias PPD PosteriorPredictive

PPD(;grid=linspace(-2,2,2), ate=zeros(2), tt=zeros(2), late=zeros(2)) = PPD(grid, ate, tt, late)

## --------------------------------------------------------------------------- #
## export to Main

## export types
export RawData, ScaleData, InputData, InputPriors, InputParams, InputDims
export PriorDP, StateDP
export PriorBeta, PriorSigma, PriorTheta, Theta, StateTheta
export StateData
export StateSampler
export GibbsInput, GibbsState, GibbsOut
export PPD, TreatmentEffects

## export functions
export flip_mat, NobileWishart, truncnorm, standardize #
export dpm_init, dpm_chain!, dpm_gibbs!, dpm_dump!, dpm!, dpm_blocked! # DPM
export dpm_fmn!, dpm_gaussian! # finite models
export rescale_beta, rescale_output, acf_var, batch_var, ppd_cdf # processing output
export dpm_ppd, rand_ppd, rand_dpm, rand_blocked, rand_gaussian, dpm_ate # posterior predictive treatment effects
# functions for parallel ppd
export select_ppd, single_dpm_ppd, single_blocked_ppd, single_fmn_ppd, single_gaussian_ppd, setup_ppd, parallel_ppd, parallel_rand_ppd

## --------------------------------------------------------------------------- #
## load main functions
include("misc_functions.jl")
include("sampler_functions.jl")
include("dpm_init.jl")

## polya-urn sampler
include("dpm_gibbs.jl")

## blocked gibbs sampler
include("dpm_blocked.jl")

## fmn gibbs sampler
include("dpm_fmn.jl")

## benchmark guassian gibbs sampler
include("dpm_gaussian.jl")

## posterior predictives
include("dpm_ppd.jl")

## parallelized ppd
include("parallel_ppd.jl")

end
