## --------------------------------------------------------------------------- #
## State

## immutable, so a StateDP saved by update_out! cannot be changed by later updates
struct BlockedWeights
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
-(b1::BlockedWeights, b2::BlockedWeights) = BlockedWeights(b1.w-b2.w, b1.v-b2.v)

*(a::Real, b::BlockedWeights) = BlockedWeights(a*b.w, a*b.v)
*(b1::BlockedWeights, b2::BlockedWeights) = BlockedWeights(b1.w*b2.w, b1.v*b2.v)

function sumbw(b::Base.ValueIterator{T}; i::Int64=1) where T<:AbstractDict
    s = reduce(+, b)
    if i == 1 return s.w else return s.v end
end

## collect DP state
## Components are indexed 1:length(njs). labels[i] is unit i's component, njs[j] its count, and
## members[j] the sorted units in component j (rebuilt after every label sweep, so the parameter
## step never scans the labels). ws holds the stick-breaking weights (blocked sampler only).
mutable struct StateDP
    J::Int64 # number of active components
    labels::Vector{Int64} # individual component membership
    njs::Vector{Int64} # component membership counts
    ws::Vector{BlockedWeights} # stick-breaking weights
    alpha::Float64 # concentration parameter
    eta::Float64 # auxiliary variable for sampling alpha
    members::Vector{Vector{Int64}} # units in each component
end

StateDP(J::Integer, labels::AbstractVector, njs::AbstractVector, ws::AbstractVector, alpha::Real, eta::Real) =
    StateDP(J, labels, njs, ws, alpha, eta, Vector{Int64}[])

## outer constructor
StateDP(; J=2,
        labels=Int64[],
        njs=Int64[],
        ws=BlockedWeights[],
        alpha=1.0,
        eta=0.0,
        members=Vector{Int64}[]) = StateDP(J, labels, njs, ws, alpha, eta, members)

## copy method
Base.copy(m::StateDP) = StateDP(m.J, copy(m.labels), copy(m.njs), copy(m.ws), m.alpha, m.eta, copy.(m.members))

## rebuild the member lists from the labels: one pass over the units
function rebuild_members!(s::StateDP)
    K = length(s.njs)
    length(s.members) == K || resize!(s.members, K)
    @inbounds for j in 1:K
        isassigned(s.members, j) ? empty!(s.members[j]) : (s.members[j] = Int64[])
        sizehint!(s.members[j], s.njs[j])
    end
    @inbounds for (i, l) in enumerate(s.labels)
        push!(s.members[l], i)
    end
    return s
end

## collect theta state, where theta = beta-Sigma pair. Immutable: the samplers replace a
## component's Theta rather than change it, so saved draws can share it without copying.
struct Theta
    beta::Vector{Float64} # latent data mean vector
    Sigma::Matrix{Float64} # latent data covariance matrix
end

Theta(; beta=zeros(2), Sigma=Matrix{Float64}(I, 2, 2)) = Theta(beta,Sigma)

## collect thetas for all components
##type StateTheta
##    state_theta::Dict{Int64,Theta}
##end

##StateTheta(; state_theta=Dict{Int64,Theta}() ) = StateTheta(state_theta)
StateTheta = Vector{Theta}

## set component j of a StateTheta, growing it by one if j is the next index
function setcomp!(v::Vector, j::Integer, x)
    if j == length(v) + 1
        push!(v, x)
    elseif 1 <= j <= length(v)
        v[j] = x
    else
        throw(BoundsError(v, j))
    end
    return v
end

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
Base.copy(m::StateData) = StateData([ copy(getfield(m, k)) for k = 1:length(fieldnames(typeof(m))) ]...)

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
