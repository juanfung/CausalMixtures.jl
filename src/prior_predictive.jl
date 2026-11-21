## --------------------------------------------------------------------------- #
## Prior predictive simulation of the generative model (plan section 04, Phase 2).
##
## Written independently of sample_prior_theta! and of every sampler, so it can check
## them. Everything is on the standardized scale: X (n x kx) and W (n x kw) hold the
## outcome and selection covariates with an intercept in column 1, and
##     z_i = (d*_i, y1_i, y0_i)' = H_i beta_{c_i} + e_i,  e_i ~ N3(0, Sigma_{c_i}), sigma11 = 1,
##     H_i = blockdiag(w_i', x_i', x_i'),  beta = (gamma', beta1', beta0'),
##     c ~ CRP(alpha),  Sigma_j ~ NobileWishart(rho, rho R),  all prior means zero.
## slopes = :cluster: the whole beta_j is drawn per cluster.
## slopes = :common:  only the three intercepts are drawn per cluster; the slopes are
##                    drawn once and shared by all clusters.

export PriorSpec, prior_predictive, g0_V0, draw_sigmas, g0_logpredictive, crp_num_clusters

"""
    PriorSpec(; name, slopes=:cluster, sd_intercept, sd_intercept_d=sd_intercept, sd_probit,
              sd_outcome, rho, r=2.0, R=diag(1/rho, r, r), alpha=1.0, alpha_shape=0.0, alpha_rate=0.0)

A candidate base measure G0 and DP prior on the standardized scale. Intercepts, probit slopes
and outcome slopes are independent N(0, sd^2); the selection-equation intercept has sd
`sd_intercept_d` and the two outcome intercepts `sd_intercept`. Sigma ~ NobileWishart(rho, rho*R). alpha is fixed
at `alpha` when `alpha_shape == 0`, and otherwise alpha ~ Gamma(alpha_shape, rate = alpha_rate).
"""
struct PriorSpec
    name::String
    slopes::Symbol
    sd_intercept::Float64
    sd_intercept_d::Float64
    sd_probit::Float64
    sd_outcome::Float64
    rho::Int
    R::Matrix{Float64}
    alpha::Float64
    alpha_shape::Float64
    alpha_rate::Float64
end

function PriorSpec(; name::String="", slopes::Symbol=:cluster, sd_intercept::Real=1.0,
                   sd_intercept_d::Real=sd_intercept, sd_probit::Real=1.0, sd_outcome::Real=1.0, rho::Integer=6, r::Real=2.0,
                   R::AbstractMatrix=Matrix(Diagonal([1/rho, r, r])), alpha::Real=1.0,
                   alpha_shape::Real=0.0, alpha_rate::Real=0.0)
    slopes in (:cluster, :common) || throw(ArgumentError("slopes must be :cluster or :common"))
    return PriorSpec(name, slopes, sd_intercept, sd_intercept_d, sd_probit, sd_outcome, rho, Matrix{Float64}(R),
                     alpha, alpha_shape, alpha_rate)
end

## copy of a spec with some fields changed
function PriorSpec(s::PriorSpec; kw...)
    f = merge((name=s.name, slopes=s.slopes, sd_intercept=s.sd_intercept,
               sd_intercept_d=s.sd_intercept_d, sd_probit=s.sd_probit,
               sd_outcome=s.sd_outcome, rho=s.rho, R=s.R, alpha=s.alpha,
               alpha_shape=s.alpha_shape, alpha_rate=s.alpha_rate), values(kw))
    return PriorSpec(; f...)
end

_draw_alpha(rng::AbstractRNG, s::PriorSpec) =
    s.alpha_shape > 0 ? rand(rng, Gamma(s.alpha_shape, 1/s.alpha_rate)) : s.alpha

"""
    g0_V0(spec, kx, kw) -> Vector

Diagonal of the prior covariance of the full beta = (gamma, beta1, beta0), intercepts first in
each block (the :cluster layout).
"""
function g0_V0(s::PriorSpec, kx::Integer, kw::Integer)
    blk(sd) = vcat(s.sd_intercept^2, fill(sd^2, kx - 1))
    return vcat(s.sd_intercept_d^2, fill(s.sd_probit^2, kw - 1), blk(s.sd_outcome), blk(s.sd_outcome))
end

## Chinese restaurant process labels for n units
function _crp(rng::AbstractRNG, n::Integer, alpha::Real)
    labels = Vector{Int}(undef, n)
    counts = Int[]
    for i in 1:n
        u = rand(rng) * (i - 1 + alpha)
        j = 0
        acc = 0.0
        for (jj, c) in enumerate(counts)
            acc += c
            if u < acc
                j = jj
                break
            end
        end
        if j == 0
            push!(counts, 1)
            labels[i] = length(counts)
        else
            counts[j] += 1
            labels[i] = j
        end
    end
    return labels, counts
end

"""
    crp_num_clusters(spec, n; reps=10_000, rng) -> Vector{Int}

Draws of the number of clusters k among n units under the spec's prior on alpha.
"""
crp_num_clusters(s::PriorSpec, n::Integer; reps::Integer=10_000, rng::AbstractRNG=Random.default_rng()) =
    [length(_crp(rng, n, _draw_alpha(rng, s))[2]) for _ in 1:reps]

"""
    prior_predictive(spec, X, W; reps=1000, rng) -> NamedTuple

Simulate `reps` data sets of size n = size(X, 1) from the generative model. Returns
- `k`, `alpha`: clusters and concentration per replicate;
- `prop`: every unit's propensity Phi(w_i' gamma_{c_i}), all replicates pooled;
- per cluster draw (pooled over replicates): `s22`, `s33` (within-cluster Var(y1), Var(y0)),
  `mu1`, `mu0` (cluster means of y1, y0 at the covariate means xbar) and `effect`
  (xbar'(beta1 - beta0)), `cluster_prop` (the cluster's mean propensity over all n covariate rows;
  near 0 or 1 means a cluster that is almost all controls or all treated, whose missing arm is
  identified only by the prior);
- per replicate: `sd_y` (sd of the observed outcome), `share_treated`.
"""
function prior_predictive(s::PriorSpec, X::AbstractMatrix, W::AbstractMatrix;
                          reps::Integer=1000, rng::AbstractRNG=Random.default_rng())
    n, kx = size(X)
    kw = size(W, 2)
    size(W, 1) == n || throw(DimensionMismatch("X and W must have the same number of rows"))
    xbar = vec(mean(X, dims=1))
    rhoR = s.rho * s.R
    ks, alphas = Int[], Float64[]
    prop = Float64[]
    s22, s33, mu1, mu0, effect = Float64[], Float64[], Float64[], Float64[], Float64[]
    cluster_prop = Float64[]
    sd_y, share_treated = Float64[], Float64[]
    slopes(sd, k) = sd .* randn(rng, k)
    for _ in 1:reps
        alpha = _draw_alpha(rng, s)
        labels, counts = _crp(rng, n, alpha)
        k = length(counts)
        push!(ks, k); push!(alphas, alpha)
        ## shared slopes for :common
        g_sl, b1_sl, b0_sl = slopes(s.sd_probit, kw - 1), slopes(s.sd_outcome, kx - 1), slopes(s.sd_outcome, kx - 1)
        gam = Vector{Vector{Float64}}(undef, k)
        b1 = Vector{Vector{Float64}}(undef, k)
        b0 = Vector{Vector{Float64}}(undef, k)
        L = Vector{Matrix{Float64}}(undef, k)
        for j in 1:k
            a = [s.sd_intercept_d, s.sd_intercept, s.sd_intercept] .* randn(rng, 3)
            if s.slopes == :cluster
                g_sl, b1_sl, b0_sl = slopes(s.sd_probit, kw - 1), slopes(s.sd_outcome, kx - 1), slopes(s.sd_outcome, kx - 1)
            end
            gam[j] = vcat(a[1], g_sl); b1[j] = vcat(a[2], b1_sl); b0[j] = vcat(a[3], b0_sl)
            Sig = NobileWishart(rng, s.rho, rhoR)
            L[j] = cholesky(Symmetric(Sig)).L
            push!(s22, Sig[2, 2]); push!(s33, Sig[3, 3])
            push!(mu1, dot(xbar, b1[j])); push!(mu0, dot(xbar, b0[j]))
            push!(effect, dot(xbar, b1[j] - b0[j]))
            ## the cluster's mean propensity over all n covariate rows
            push!(cluster_prop, mean(cdf.(Normal(), W * gam[j])))
        end
        yobs = Vector{Float64}(undef, n)
        ntreat = 0
        for i in 1:n
            j = labels[i]
            x = view(X, i, :); w = view(W, i, :)
            idx = dot(w, gam[j])
            push!(prop, cdf(Normal(), idx))
            e = L[j] * randn(rng, 3)
            dstar = idx + e[1]
            if dstar > 0
                yobs[i] = dot(x, b1[j]) + e[2]; ntreat += 1
            else
                yobs[i] = dot(x, b0[j]) + e[3]
            end
        end
        push!(sd_y, std(yobs)); push!(share_treated, ntreat / n)
    end
    return (k=ks, alpha=alphas, prop=prop, s22=s22, s33=s33, mu1=mu1, mu0=mu0, effect=effect,
            cluster_prop=cluster_prop, sd_y=sd_y, share_treated=share_treated)
end

"""
    draw_sigmas(spec, S; rng) -> Vector{Matrix}

S draws of Sigma ~ NobileWishart(rho, rho R), for Monte Carlo over Sigma in `g0_logpredictive`.
"""
draw_sigmas(s::PriorSpec, S::Integer; rng::AbstractRNG=Random.default_rng()) =
    [NobileWishart(rng, s.rho, s.rho * s.R) for _ in 1:S]

"""
    g0_logpredictive(spec, z, x, w, sigmas; b=nothing) -> Float64

log p(z | G0) for one unit, with beta integrated in closed form and Sigma by Monte Carlo over
`sigmas`:  log mean_s N3(z | m, Sigma_s + C).
- :cluster: m = H mu = 0 and C = H V0 H'.
- :common: only the intercepts are new, so m = H^g b with the shared slopes
  b = (gamma_slopes, beta1_slopes, beta0_slopes) given, and C = sd_intercept^2 I.
"""
function g0_logpredictive(s::PriorSpec, z::AbstractVector, x::AbstractVector, w::AbstractVector,
                          sigmas::AbstractVector{<:AbstractMatrix}; b=nothing)
    kx, kw = length(x), length(w)
    if s.slopes == :cluster
        v = g0_V0(s, kx, kw)
        vg, v1, v0 = v[1:kw], v[kw+1:kw+kx], v[kw+kx+1:end]
        C = Diagonal([sum(w .^ 2 .* vg), sum(x .^ 2 .* v1), sum(x .^ 2 .* v0)])
        m = zeros(3)
    else
        b === nothing && throw(ArgumentError(":common needs the shared slopes b"))
        bg, bb1, bb0 = b[1:kw-1], b[kw:kw+kx-2], b[kw+kx-1:end]
        m = [dot(w[2:end], bg), dot(x[2:end], bb1), dot(x[2:end], bb0)]
        C = Diagonal([s.sd_intercept_d^2, s.sd_intercept^2, s.sd_intercept^2])
    end
    lps = [logpdf(MvNormal(m, Symmetric(Sig + C)), z) for Sig in sigmas]
    return logsumexp(lps) - log(length(lps))
end
