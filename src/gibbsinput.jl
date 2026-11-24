## --------------------------------------------------------------------------- #
## Data

## raw data input
struct RawData{FY<:FormulaTerm,FD<:FormulaTerm}
    y_form::FY
    d_form::FD
    df::DataFrame
end

## standardized data: a = (A .- m') ./ s' (a vector for y, a matrix for X and W)
struct ScaleData{A<:Array{Float64}}
    a::A
    m::Vector{Float64}
    s::Vector{Float64}
end

ScaleData(; a=[0.0], m=[0.0], s=[1.0]) = ScaleData(Array{Float64}(a), Vector{Float64}(m), Vector{Float64}(s))

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
    y::ScaleData{Vector{Float64}} # outcome (y.a), and its centering and scale
    d::Vector{Int64}
    lower::Vector{Float64}
    upper::Vector{Float64}
    Hmat::SparseMatrixCSC{Float64,Int64}
    X::Matrix{Float64} # n x kx outcome-equation design (dense; the samplers use X and W, not Hmat)
    W::Matrix{Float64} # n x kz selection-equation design
    xnames::Vector{String} # outcome-equation columns of Hmat, in order
    znames::Vector{String} # selection-equation columns of Hmat, in order
end

InputData(; y=ScaleData(a=[0.0]), d=[0], lower=[0.0], upper=[0.0], Hmat=sparse([0.0]), X=zeros(0, 0), W=zeros(0, 0),
          xnames=String[], znames=String[]) =
    InputData(y, d, lower, upper, Hmat, X, W, xnames, znames)
## TODO: define InputData with/without {xmat,zmat}

## --------------------------------------------------------------------------- #
## Parameters

## M iterations; save the component parameters every `thin`-th iteration and the latent data
## and labels every `latent_thin`-th iteration (0 = never). `slopes` is the slope mode of the
## Design A sampler (model = "marginal"): :common (shared covariate slopes, the DPM over the
## intercepts and Sigma) or :cluster (the full beta per cluster); other models ignore it.
## `label_step` is Design A's label step: :r2 (the joint label-and-latent step, Phase 4b) or
## :augmented (scored on the current latent data, Phase 4a). `sigma23_step` turns on R3, the
## collapsed sigma23 step (Phase 4c), in the marginal and gaussian samplers (off by default: it
## did not improve mixing). `hierarchical` makes Design A's cluster intercepts hierarchical,
## a_j ~ N(m, diag(tau2)) with m and tau2 drawn every sweep (Phase 4d).
mutable struct InputParams
    M::Int64
    scale_data::Tuple{Bool,Bool}
    verbose::Bool
    model::String
    thin::Int64
    latent_thin::Int64
    slopes::Symbol
    label_step::Symbol
    sigma23_step::Bool
    hierarchical::Bool
    function InputParams(M, scale_data, verbose, model, thin, latent_thin, slopes, label_step, sigma23_step, hierarchical)
        thin >= 1 || throw(ArgumentError("thin must be >= 1, got $thin"))
        latent_thin >= 0 || throw(ArgumentError("latent_thin must be >= 0, got $latent_thin"))
        slopes in (:common, :cluster) || throw(ArgumentError("slopes must be :common or :cluster, got $slopes"))
        label_step in (:r2, :augmented) || throw(ArgumentError("label_step must be :r2 or :augmented, got $label_step"))
        return new(M, scale_data, verbose, model, thin, latent_thin, slopes, label_step, sigma23_step, hierarchical)
    end
end

InputParams(; M=100, scale_data=(false,false), verbose=true, model="dpm", thin=1, latent_thin=0, slopes=:common,
            label_step=:r2, sigma23_step=false, hierarchical=true) =
    InputParams(M, scale_data, verbose, model, thin, latent_thin, slopes, label_step, sigma23_step, hierarchical)
## set default model to dpm
InputParams(m, s, v) = InputParams(M=m, scale_data=s, verbose=v, model="dpm")

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
    V::Matrix{Float64} # prior precision (after dpm_init); as given by the user before
    Vinv::Bool # if true, V is a precision; else, a covariance
    Vcov::Matrix{Float64} # prior covariance V0 (always a covariance)
    Lcov::LowerTriangular{Float64,Matrix{Float64}} # lower Cholesky factor of V0, for prior draws
end

## covariance and its Cholesky factor from V, which is a precision if Vinv is true
function PriorBeta(mu::AbstractVector, V::AbstractMatrix, Vinv::Bool)
    Vcov = Vinv ? Matrix(Symmetric(inv(Symmetric(Matrix{Float64}(V))))) : Matrix{Float64}(V)
    return PriorBeta(mu, V, Vinv, Vcov, cholesky(Symmetric(Vcov)).L)
end

PriorBeta(; mu=ones(2), V=Matrix(1.0I, 2, 2), Vinv=true) = PriorBeta(mu, V, Vinv)

struct PriorSigma
    rho::Int64
    R::Matrix{Float64}
end

PriorSigma(; rho=1, R=Matrix(1.0I, 3, 3) ) = PriorSigma(rho, R)

struct PriorTheta
    prior_beta::PriorBeta
    prior_Sigma::PriorSigma
end

PriorTheta(; prior_beta=PriorBeta(), prior_Sigma=PriorSigma()) = PriorTheta(prior_beta, prior_Sigma)

## collect priors. `slopes` records the slope mode the prior was built for by
## setup_default_priors (:cluster, :common), or :unspecified.
struct InputPriors
    prior_dp::PriorDP
    prior_theta::PriorTheta
    slopes::Symbol
end

InputPriors(prior_dp::PriorDP, prior_theta::PriorTheta) = InputPriors(prior_dp, prior_theta, :unspecified)
InputPriors(; prior_dp = PriorDP(), prior_theta=PriorTheta(), slopes=:unspecified) = InputPriors(prior_dp, prior_theta, slopes)

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
export RawData, ScaleData, InputData, InputPriors, InputParams, InputDims
export PriorDP, PriorBeta, PriorSigma, PriorTheta
export GibbsInput
