## ---------------------------------------------------------------------------#
## Output
##
## Saved draws, thinned. Every `thin`-th iteration stores the component parameters and the DP
## state needed downstream: out_theta[s] (the Theta of every component; Theta is immutable and
## the samplers replace it, so the saved vector shares it without copying) and out_dp[s] (J, the
## counts n_j, the stick-breaking weights and alpha). The latent data and labels (n values each)
## are stored only every `latent_thin`-th iteration, and not at all when latent_thin = 0.

## one saved iteration of the DP state (the labels are saved separately, with the latent data)
struct DrawDP
    J::Int64
    njs::Vector{Int64}
    ws::Vector{BlockedWeights}
    alpha::Float64
    eta::Float64
end

DrawDP(s::StateDP) = DrawDP(s.J, copy(s.njs), copy(s.ws), s.alpha, s.eta)

mutable struct GibbsOut
    out_data::Vector{StateData}
    out_dp::Vector{DrawDP}
    out_theta::Vector{StateTheta}
    out_labels::Vector{Vector{Int64}}
    thin::Int64
    latent_thin::Int64
end

## storage for M iterations
function GibbsOut(M::Integer; thin::Integer=1, latent_thin::Integer=0)
    thin >= 1 || throw(ArgumentError("thin must be >= 1, got $thin"))
    latent_thin >= 0 || throw(ArgumentError("latent_thin must be >= 0, got $latent_thin"))
    nl = latent_thin == 0 ? 0 : M ÷ latent_thin
    return GibbsOut(Vector{StateData}(undef, nl), Vector{DrawDP}(undef, M ÷ thin),
                    Vector{StateTheta}(undef, M ÷ thin), Vector{Vector{Int64}}(undef, nl), thin, latent_thin)
end

## storage for a run with the thinning in params
GibbsOut(params::InputParams) = GibbsOut(params.M; thin=params.thin, latent_thin=params.latent_thin)

## default constructor with no arguments
GibbsOut() = GibbsOut(0)

## --------------------------------------------------------------------------- #
export GibbsOut, DrawDP
