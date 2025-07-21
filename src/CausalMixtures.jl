## Module for causal inference via mixture models

##__precompile__()

module CausalMixtures

## load packages
using DataFrames, DataStructures, Distributions, LinearAlgebra, Random, SparseArrays, StatsBase, StatsModels
using JLD2, Printf
## using Rebugger
import Base.+, Base.-, Base.*, Base.Iterators.take

export DataFrames, Distributions, LinearAlgebra, StatsBase, StatsModels, Random, JLD2

## --------------------------------------------------------------------------- #
## Objects:
## GibbsInput: {Data, Params, Priors}
include("gibbsinput.jl")
## GibbsState: {DP, Theta, LatentData}
include("gibbsstate.jl")
## GibbsOut: Array{DP[m], Theta[m], LatentData[m], m=1:M}
include("gibbsout.jl")

## where
## DP: {J, labels, njs, alpha, eta}, where
##      J = num active components
##      labels = component memberships
##      njs = counts of component membership
## Theta: {Beta, Sigma}
## LatentData: {dstar, y1, y0}

## --------------------------------------------------------------------------- #
## Collect {State, Input, Out} into tuple

GibbsTuple = Tuple{GibbsState, GibbsInput, GibbsOut}
##GibbsTuple(; state=GibbsState(), input=GibbsInput(), out=GibbsOut()) = GibbsTuple(state, input, out)

export GibbsTuple

## --------------------------------------------------------------------------- #
## Posterior Predictive Distribution and Treatment Effects
include("gibbsppd.jl")

## --------------------------------------------------------------------------- #
## load main functions
include("misc_functions.jl")
include("sampler_functions.jl")
include("dpm_init.jl")

## polya-urn sampler
include("dpm_gibbs.jl")
## blocked gibbs sampler
include("dpm_blocked.jl")
## marginal gibbs sampler
include("dpm_marginal.jl")
## fmn gibbs sampler
include("dpm_fmn.jl")
## benchmark guassian gibbs sampler
include("dpm_gaussian.jl")

## posterior predictives
include("dpm_ppd.jl")
## parallelized ppd
include("parallel_ppd.jl")

end
