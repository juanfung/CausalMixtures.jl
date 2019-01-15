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

function sumbw(b::Base.ValueIterator{T}; i::Int64=1) where T<:AbstractDict
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
StateTheta = Dict{Int64,Theta}

mutable struct NjTheta
    nj::Int64
    theta::Theta    
end

NjTheta(; nj=0, theta=Theta()) = NjTheta(nj, theta)

StateNjTheta = Dict{Int64,NjTheta}

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
export BlockedWeights, StateDP
export Theta, NjTheta, StateNjTheta
export StateData, StateSampler, GibbsState
