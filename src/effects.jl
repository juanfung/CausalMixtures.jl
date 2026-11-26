## --------------------------------------------------------------------------- #
## Phase 5: per-draw, label-invariant treatment effects and partition summaries.
##
## For each saved draw t, the fitted mixture is sum_j pi_j N3(H beta_j, Sigma_j) with component
## weights pi drawn per draw:
##   marginal, dpm: (pi_1..pi_k, pi_0) ~ Dir(n_1, ..., n_k, alpha) over the occupied clusters,
##     pi_0 being the mass of G off the occupied atoms, represented by n_g0 fresh draws from G0
##     with weight pi_0 / n_g0 each (marginal: G0 given the draw's (m, tau2) and shared b);
##   blocked: the stick weights w_j; fmn: n_j / n; gaussian: pi = 1.
## With Delta_ij = x_i'(beta1_j - beta0_j), s_ij = w_i'gamma_j, Phi_ij = Phi(s_ij), phi_ij = phi(s_ij)
## (sigma11 = 1, so E[u_k | v > -s] = sigma_1k phi(s)/Phi(s) and E[u_k | v < -s] = -sigma_1k phi(s)/(1 - Phi(s))):
##   PATE = (1/n) sum_i sum_j pi_j Delta_ij,
##   TT(x, w)  = sum_j pi_j Phi_j [Delta_j + (s12_j - s13_j) phi_j / Phi_j] / sum_j pi_j Phi_j,
##   TUT(x, w) = sum_j pi_j (1 - Phi_j) [Delta_j - (s12_j - s13_j) phi_j / (1 - Phi_j)] / sum_j pi_j (1 - Phi_j),
##   PATT = TT averaged over the treated units' own (x, w); PATU = TUT over the untreated units'.
## All on the original y scale (times sd(y) when the data were standardized). With saved latent
## draws, SATT and SATU average y1 - y0 (observed outcome, imputed counterfactual) over the treated
## and untreated units. These are posterior draws of population and sample estimands: their
## quantiles are credible intervals. Contrast rand_ppd / dpm_ate, which draw a new individual's
## gain (a predictive distribution that depends on sigma23).

export Effects, effects, effects_summary, partition_summary

"""
    Effects

Per-draw treatment effects on the original y scale: `pate`, `patt`, `patu` (one value per saved
draw used), and `satt`, `satu` (one per saved latent draw; empty without latent draws).
"""
struct Effects
    pate::Vector{Float64}
    patt::Vector{Float64}
    patu::Vector{Float64}
    satt::Vector{Float64}
    satu::Vector{Float64}
end

## the mixture's TT and TUT at one unit (x, w), for components (beta, Sigma) with weights pi; on
## the model scale. kz, kx: lengths of gamma and of beta1 = beta0.
function te_point(betas::AbstractVector, Sigmas::AbstractVector, pis::AbstractVector{Float64},
                  x::AbstractVector{Float64}, w::AbstractVector{Float64}, kz::Integer, kx::Integer)
    nt = dt = nu = du = 0.0
    @inbounds for j in eachindex(pis)
        b = betas[j]
        S = Sigmas[j]
        s = 0.0
        for a in 1:kz
            s += w[a] * b[a]
        end
        Delta = 0.0
        for a in 1:kx
            Delta += x[a] * (b[kz + a] - b[kz + kx + a])
        end
        c = S[1, 2] - S[1, 3]
        lP, lQ, lp = logcdf(Normal(), s), logccdf(Normal(), s), logpdf(Normal(), s)
        P, Q = exp(lP), exp(lQ)
        nt += pis[j] * (P * Delta + c * exp(lp))     # pi Phi [Delta + c phi / Phi]
        dt += pis[j] * P
        nu += pis[j] * (Q * Delta - c * exp(lp))     # pi (1 - Phi) [Delta - c phi / (1 - Phi)]
        du += pis[j] * Q
    end
    return nt / dt, nu / du
end

## component weights of one saved draw: Dir(n_1..n_k, alpha) (:dirichlet) or its mean (:mean)
function draw_weights(rng::AbstractRNG, njs::AbstractVector{<:Integer}, alpha::Real; weights::Symbol=:dirichlet)
    a = vcat(Float64.(njs), alpha > 0 ? Float64(alpha) : Float64[])
    if weights == :dirichlet
        return rand(rng, Dirichlet(a))
    elseif weights == :mean
        return a ./ sum(a)
    else
        throw(ArgumentError("weights must be :dirichlet or :mean, got $weights"))
    end
end

## R of saved draw s: diag(1/rho, r1, r0) with the draw's learned r (sigma_scale = :hier), or the prior's R
function draw_R(out::GibbsOut, input::GibbsInput, s::Integer)
    R = input.priors.prior_theta.prior_Sigma.R
    (isassigned(out.out_r, s) && length(out.out_r[s]) == 2) || return R
    Rc = copy(R)
    Rc[2, 2], Rc[3, 3] = out.out_r[s][1], out.out_r[s][2]
    return Rc
end

## the components (beta, Sigma) and weights of saved draw s
function draw_mixture(rng::AbstractRNG, out::GibbsOut, input::GibbsInput, s::Integer, D, G0Sigma::NobileFactor;
                      n_g0::Integer=20, weights::Symbol=:dirichlet)
    model = input.params.model
    dp, th = out.out_dp[s], out.out_theta[s]
    rho = input.priors.prior_theta.prior_Sigma.rho
    if model == "gaussian"
        return [th[1].beta], [th[1].Sigma], [1.0]
    elseif model == "blocked"
        J = length(th)
        return [th[j].beta for j in 1:J], [th[j].Sigma for j in 1:J], [dp.ws[j].w for j in 1:J]
    elseif model == "fmn"
        occ = [j for j in eachindex(dp.njs) if dp.njs[j] > 0]
        return [th[j].beta for j in occ], [th[j].Sigma for j in occ], dp.njs[occ] ./ sum(dp.njs[occ])
    elseif model in ("marginal", "dpm")
        occ = [j for j in eachindex(dp.njs) if dp.njs[j] > 0]
        pis = draw_weights(rng, dp.njs[occ], n_g0 > 0 ? dp.alpha : 0.0; weights=weights)
        betas = [th[j].beta for j in occ]
        Sigmas = [th[j].Sigma for j in occ]
        if length(pis) > length(occ)
            p0 = pis[end] / n_g0
            pis = vcat(pis[1:end-1], fill(p0, n_g0))
            if model == "marginal"
                if D.hier && isassigned(out.out_hyper, s) && length(out.out_hyper[s]) == 6
                    set_cprior!(D, out.out_hyper[s][1:3], out.out_hyper[s][4:6])
                end
                b = th[occ[1]].beta[D.gidx]
                for _ in 1:n_g0
                    push!(betas, full_beta(D, D.mu_c + D.Lc * randn(rng, length(D.mu_c)), b))
                    push!(Sigmas, NobileWishart(rng, rho, G0Sigma))
                end
            else
                pb = input.priors.prior_theta.prior_beta
                for _ in 1:n_g0
                    push!(betas, pb.mu + pb.Lcov * randn(rng, length(pb.mu)))
                    push!(Sigmas, NobileWishart(rng, rho, G0Sigma))
                end
            end
        end
        return betas, Sigmas, pis
    else
        throw(ArgumentError("no effects method for model $model"))
    end
end

"""
    effects(out, input; rng, n_g0 = 20, burn = 0, weights = :dirichlet) -> Effects

Per-draw, label-invariant treatment effects of a fitted model, on the original y scale, for the
saved draws after the first `burn`:
- `pate`: (1/n) sum_i sum_j pi_j x_i'(beta1_j - beta0_j);
- `patt`, `patu`: the mixture's closed-form TT and TUT, averaged over the treated and the untreated
  units' own covariates;
- `satt`, `satu` (when latent draws were saved, `latent_thin > 0`): the mean of y1 - y0 over the
  treated and untreated units, observed outcome against imputed counterfactual.

The weights pi of each draw are drawn as (pi_1..pi_k, pi_0) ~ Dir(n_1..n_k, alpha) for `marginal`
and `dpm` (`weights = :mean` uses the Dirichlet mean instead), with pi_0 spread over `n_g0`
fresh draws from G0 (`n_g0 = 0` drops it); the stick weights for `blocked`; pi = 1 for
`gaussian`. Quantiles of the draws are credible intervals (`effects_summary`). For the predictive
distribution of one new unit's gain, which depends on sigma23, see `rand_ppd` and `dpm_ate`.
"""
function effects(out::GibbsOut, input::GibbsInput; rng::AbstractRNG=Random.default_rng(), n_g0::Integer=20,
                 burn::Integer=0, weights::Symbol=:dirichlet)
    X, W = input.data.X, input.data.W
    kz, kx, n = input.dims.kz, input.dims.kx, input.dims.n
    sy = input.data.y.s[1]
    d = input.data.d
    tr, un = findall(==(1), d), findall(==(0), d)
    ps = input.priors.prior_theta.prior_Sigma
    G0S = NobileFactor(ps.rho .* ps.R)
    D = input.params.model == "marginal" ? DesignA(input) : nothing
    S = length(out.out_dp)
    keep = (burn + 1):S
    pate, patt, patu = zeros(length(keep)), zeros(length(keep)), zeros(length(keep))
    xbar = vec(mean(X, dims=1))
    for (t, s) in enumerate(keep)
        G0s = isassigned(out.out_r, s) && length(out.out_r[s]) == 2 ? NobileFactor(ps.rho .* draw_R(out, input, s)) : G0S
        betas, Sigmas, pis = draw_mixture(rng, out, input, s, D, G0s; n_g0=n_g0, weights=weights)
        pate[t] = sy * sum(pis[j] * dot(xbar, betas[j][kz+1:kz+kx] - betas[j][kz+kx+1:end]) for j in eachindex(pis))
        at, au = 0.0, 0.0
        for i in tr
            at += te_point(betas, Sigmas, pis, view(X, i, :), view(W, i, :), kz, kx)[1]
        end
        for i in un
            au += te_point(betas, Sigmas, pis, view(X, i, :), view(W, i, :), kz, kx)[2]
        end
        patt[t] = sy * at / max(length(tr), 1)
        patu[t] = sy * au / max(length(un), 1)
    end
    satt, satu = Float64[], Float64[]
    if out.latent_thin > 0
        lburn = burn * out.thin ÷ out.latent_thin   # the same share of the run
        for s in (lburn + 1):length(out.out_data)
            isassigned(out.out_data, s) || continue
            z = out.out_data[s]
            push!(satt, sy * mean(z.y1[i] - z.y0[i] for i in tr))
            push!(satu, sy * mean(z.y1[i] - z.y0[i] for i in un))
        end
    end
    return Effects(pate, patt, patu, satt, satu)
end

"""
    effects_summary(effs; probs = (0.025, 0.5, 0.975)) -> NamedTuple of NamedTuples

Summary of each estimand over one or several chains (`effs` an `Effects` or a vector of them, one
per chain): mean, sd, the quantiles in `probs`, bulk ESS, and the rank-normalized split R-hat
(with two or more chains; NaN otherwise).
"""
function effects_summary(effs::AbstractVector{Effects}; probs=(0.025, 0.5, 0.975))
    function one(f)
        ch = [getfield(e, f) for e in effs if !isempty(getfield(e, f))]
        isempty(ch) && return nothing
        x = reduce(vcat, ch)
        q = quantile(x, collect(probs))
        L = minimum(length.(ch))
        chs = [c[1:L] for c in ch]
        return (mean=mean(x), sd=std(x), q=Tuple(q), probs=Tuple(probs),
                ess_bulk = L >= 8 ? ess_bulk(chs) : NaN, rhat = length(chs) >= 2 && L >= 8 ? rhat(chs) : NaN)
    end
    return NamedTuple{(:pate, :patt, :patu, :satt, :satu)}(Tuple(one(f) for f in (:pate, :patt, :patu, :satt, :satu)))
end

effects_summary(e::Effects; kw...) = effects_summary([e]; kw...)

"""
    partition_summary(out; burn = 0, max_candidates = 500) -> NamedTuple

Partition summaries from the saved labels (needs `latent_thin > 0`), after the first `burn` saved
label draws:
- `psm`: the n x n posterior similarity matrix, P(c_i = c_j);
- `point`: the saved partition minimizing the expected Binder loss sum_{i<j} |1{c_i = c_j} - psm_ij|
  (searched over up to `max_candidates` evenly spaced saved partitions), with its `binder` loss;
- `k`: the posterior of the number of occupied clusters, as a Dict k => probability (from the
  saved draws, after the same share of burn-in);
- `separated`: the posterior mean of the separated-units share (`out.out_sep`).
"""
function partition_summary(out::GibbsOut; burn::Integer=0, max_candidates::Integer=500)
    out.latent_thin > 0 || throw(ArgumentError("partition_summary needs saved labels: run with InputParams(latent_thin = k > 0)"))
    labs = [out.out_labels[s] for s in (burn + 1):length(out.out_labels) if isassigned(out.out_labels, s)]
    isempty(labs) && throw(ArgumentError("no saved labels after burn-in"))
    n = length(labs[1])
    psm = zeros(n, n)
    for c in labs
        @inbounds for j in 1:n, i in 1:n
            psm[i, j] += c[i] == c[j]
        end
    end
    psm ./= length(labs)
    cand = unique(round.(Int, range(1, length(labs), length=min(max_candidates, length(labs)))))
    best, bestloss = 0, Inf
    for s in cand
        c = labs[s]
        loss = 0.0
        @inbounds for j in 2:n, i in 1:(j - 1)
            loss += abs((c[i] == c[j]) - psm[i, j])
        end
        loss < bestloss && ((best, bestloss) = (s, loss))
    end
    ## relabel the point partition 1..K in order of first appearance
    c = labs[best]
    map_ = Dict{Int,Int}()
    point = [get!(map_, l, length(map_) + 1) for l in c]
    dburn = burn * out.latent_thin ÷ out.thin
    ks = [count(>(0), out.out_dp[s].njs) for s in (dburn + 1):length(out.out_dp)]
    kpost = Dict(k => count(==(k), ks) / length(ks) for k in unique(ks))
    sep = [out.out_sep[s] for s in (dburn + 1):length(out.out_sep) if isassigned(out.out_sep, s)]
    return (psm=psm, point=point, binder=bestloss, k=kpost, separated = isempty(sep) ? NaN : mean(sep))
end
