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

export PriorSpec, prior_predictive, prior_draw, g0_V0, draw_sigmas, g0_logpredictive, crp_num_clusters

"""
    PriorSpec(; name, slopes=:cluster, sd_intercept, sd_intercept_d=sd_intercept, sd_probit,
              sd_outcome, rho, r=2.0, R=diag(1/rho, r, r), alpha=1.0, alpha_shape=0.0, alpha_rate=0.0)

A candidate base measure G0 and DP prior on the standardized scale. Intercepts, probit slopes
and outcome slopes are independent N(0, sd^2); the selection-equation intercept has sd
`sd_intercept_d` and the two outcome intercepts `sd_intercept`. Sigma ~ NobileWishart(rho, rho*R). alpha is fixed
at `alpha` when `alpha_shape == 0`, and otherwise alpha ~ Gamma(alpha_shape, rate = alpha_rate).
`hierarchical = true` (Phase 4d): the cluster intercepts a_j = (a_d, a_1, a_0) ~ N(m, diag(tau2)) with
m ~ N(0, diag(v)) and tau2_l ~ IG(3, 2 v_l), v = (sd_intercept_d^2, sd_intercept^2, sd_intercept^2).
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
    hierarchical::Bool
end

function PriorSpec(; name::String="", slopes::Symbol=:cluster, sd_intercept::Real=1.0,
                   sd_intercept_d::Real=sd_intercept, sd_probit::Real=1.0, sd_outcome::Real=1.0, rho::Integer=6, r::Real=2.0,
                   R::AbstractMatrix=Matrix(Diagonal([1/rho, r, r])), alpha::Real=1.0,
                   alpha_shape::Real=0.0, alpha_rate::Real=0.0, hierarchical::Bool=false)
    slopes in (:cluster, :common) || throw(ArgumentError("slopes must be :cluster or :common"))
    return PriorSpec(name, slopes, sd_intercept, sd_intercept_d, sd_probit, sd_outcome, rho, Matrix{Float64}(R),
                     alpha, alpha_shape, alpha_rate, hierarchical)
end

## copy of a spec with some fields changed
function PriorSpec(s::PriorSpec; kw...)
    f = merge((name=s.name, slopes=s.slopes, sd_intercept=s.sd_intercept,
               sd_intercept_d=s.sd_intercept_d, sd_probit=s.sd_probit,
               sd_outcome=s.sd_outcome, rho=s.rho, R=s.R, alpha=s.alpha,
               alpha_shape=s.alpha_shape, alpha_rate=s.alpha_rate, hierarchical=s.hierarchical), values(kw))
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
    prior_draw(spec, X, W; rng, partition=:crp, J=20) -> NamedTuple

One draw of everything in the generative model, for n = size(X, 1) units with covariates X
(n x kx) and W (n x kw), intercepts in column 1:
- `alpha`, `labels` (1:K), `counts`;
- per component j: `gam[j]`, `b1[j]`, `b0[j]` (full coefficient vectors, intercept first) and
  `Sigma[j]`; under :common the slopes are shared and also returned as `slopes =
  (gamma slopes, beta1 slopes, beta0 slopes)` (empty vectors under :cluster);
- per unit: `z` (n x 3: d*, y1, y0), `d = 1{d* > 0}` and the observed outcome `y`.

`partition = :crp`: c ~ CRP(alpha), K = number of clusters. `partition = :stick`: the truncated
stick-breaking prior of the blocked sampler, v_j ~ Beta(1, alpha) for j < J, v_J = 1, and
c_i ~ Categorical(w); all K = J components get parameters, occupied or not.
Independent of the samplers' own prior draws (used by the Geweke tests).
"""
function prior_draw(s::PriorSpec, X::AbstractMatrix, W::AbstractMatrix;
                    rng::AbstractRNG=Random.default_rng(), partition::Symbol=:crp, J::Integer=20)
    n, kx = size(X)
    kw = size(W, 2)
    size(W, 1) == n || throw(DimensionMismatch("X and W must have the same number of rows"))
    rhoR = s.rho * s.R
    slopes(sd, k) = sd .* randn(rng, k)
    alpha = _draw_alpha(rng, s)
    if partition == :crp
        labels, counts = _crp(rng, n, alpha)
    elseif partition == :stick
        v = [j < J ? rand(rng, Beta(1, alpha)) : 1.0 for j in 1:J]
        w = [v[j] * prod(1 .- v[1:j-1]) for j in 1:J]
        labels = [rand(rng, Categorical(w ./ sum(w))) for _ in 1:n]
        counts = [count(==(j), labels) for j in 1:J]
    else
        throw(ArgumentError("partition must be :crp or :stick"))
    end
    K = length(counts)
    ## hierarchical intercepts: (m, tau2) once per draw
    vint = [s.sd_intercept_d, s.sd_intercept, s.sd_intercept] .^ 2
    if s.hierarchical
        m = sqrt.(vint) .* randn(rng, 3)
        tau2 = [rand(rng, InverseGamma(3.0, 2v)) for v in vint]
    else
        m, tau2 = zeros(3), vint
    end
    ## shared slopes for :common (drawn in both modes, to keep one random stream)
    g_sl, b1_sl, b0_sl = slopes(s.sd_probit, kw - 1), slopes(s.sd_outcome, kx - 1), slopes(s.sd_outcome, kx - 1)
    shared = s.slopes == :common ? (copy(g_sl), copy(b1_sl), copy(b0_sl)) : (Float64[], Float64[], Float64[])
    gam = Vector{Vector{Float64}}(undef, K)
    b1 = Vector{Vector{Float64}}(undef, K)
    b0 = Vector{Vector{Float64}}(undef, K)
    Sigma = Vector{Matrix{Float64}}(undef, K)
    L = Vector{Matrix{Float64}}(undef, K)
    for j in 1:K
        a = m .+ sqrt.(tau2) .* randn(rng, 3)
        if s.slopes == :cluster
            g_sl, b1_sl, b0_sl = slopes(s.sd_probit, kw - 1), slopes(s.sd_outcome, kx - 1), slopes(s.sd_outcome, kx - 1)
        end
        gam[j] = vcat(a[1], g_sl); b1[j] = vcat(a[2], b1_sl); b0[j] = vcat(a[3], b0_sl)
        Sigma[j] = NobileWishart(rng, s.rho, rhoR)
        L[j] = cholesky(Symmetric(Sigma[j])).L
    end
    z = Matrix{Float64}(undef, n, 3)
    for i in 1:n
        j = labels[i]
        x = view(X, i, :); w = view(W, i, :)
        e = L[j] * randn(rng, 3)
        z[i, 1] = dot(w, gam[j]) + e[1]
        z[i, 2] = dot(x, b1[j]) + e[2]
        z[i, 3] = dot(x, b0[j]) + e[3]
    end
    d = Int.(z[:, 1] .> 0)
    y = ifelse.(d .== 1, z[:, 2], z[:, 3])
    return (alpha=alpha, labels=labels, counts=counts, gam=gam, b1=b1, b0=b0, Sigma=Sigma,
            slopes=shared, z=z, d=d, y=y, m = s.hierarchical ? m : Float64[], tau2 = s.hierarchical ? tau2 : Float64[])
end

"""
    prior_predictive(spec, X, W; reps=1000, rng) -> NamedTuple

Simulate `reps` data sets of size n = size(X, 1) from the generative model (`prior_draw`).
Returns
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
    xbar = vec(mean(X, dims=1))
    ks, alphas = Int[], Float64[]
    prop = Float64[]
    s22, s33, mu1, mu0, effect = Float64[], Float64[], Float64[], Float64[], Float64[]
    cluster_prop = Float64[]
    sd_y, share_treated = Float64[], Float64[]
    for _ in 1:reps
        r = prior_draw(s, X, W; rng=rng)
        push!(ks, length(r.counts)); push!(alphas, r.alpha)
        for j in eachindex(r.counts)
            push!(s22, r.Sigma[j][2, 2]); push!(s33, r.Sigma[j][3, 3])
            push!(mu1, dot(xbar, r.b1[j])); push!(mu0, dot(xbar, r.b0[j]))
            push!(effect, dot(xbar, r.b1[j] - r.b0[j]))
            push!(cluster_prop, mean(cdf.(Normal(), W * r.gam[j])))
        end
        for i in eachindex(r.labels)
            push!(prop, cdf(Normal(), dot(view(W, i, :), r.gam[r.labels[i]])))
        end
        push!(sd_y, std(r.y)); push!(share_treated, mean(r.d))
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
