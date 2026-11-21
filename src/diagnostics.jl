## --------------------------------------------------------------------------- #
## Multi-chain MCMC diagnostics for scalar, label-invariant traces (ATE per draw, k, alpha).
##
## rhat:     rank-normalized split R-hat, the larger of the bulk and folded versions
##           (Vehtari, Gelman, Simpson, Carpenter and Bürkner 2021).
## ess_bulk: effective sample size of the rank-normalized split chains (same paper).
## ess_mean: effective sample size of the raw split chains, for the Monte Carlo error of a mean.
## Both ESS use the multi-chain autocorrelation of Stan with Geyer's initial monotone sequence.
## `chains` is a vector of equal-length traces, one per chain (a single chain is fine).

export rhat, ess_bulk, ess_mean, trace_ate, trace_k, trace_alpha, chain_diagnostics

_aschains(chains::AbstractVector{<:AbstractVector{<:Real}}) = [Float64.(c) for c in chains]
_aschains(x::AbstractVector{<:Real}) = [Float64.(x)]

## split every chain into its two halves (dropping the middle draw of an odd-length chain)
function _split(chains::Vector{Vector{Float64}})
    N = minimum(length, chains)
    h = N ÷ 2
    out = Vector{Vector{Float64}}()
    for c in chains
        push!(out, c[1:h])
        push!(out, c[(N - h + 1):N])
    end
    return out
end

## rank-normalize the pooled draws (ranks with ties averaged, then normal scores)
function _ranknorm(chains::Vector{Vector{Float64}})
    lens = length.(chains)
    r = StatsBase.tiedrank(reduce(vcat, chains))
    S = length(r)
    z = quantile.(Normal(), (r .- 0.375) ./ (S + 0.25))
    out = Vector{Vector{Float64}}()
    k = 0
    for L in lens
        push!(out, z[(k + 1):(k + L)])
        k += L
    end
    return out
end

## within-chain variance W and the pooled estimate var_plus (Stan's definitions)
function _wvar(chains::Vector{Vector{Float64}})
    N = length(chains[1])
    W = mean(var.(chains))
    B = length(chains) > 1 ? var(mean.(chains)) : 0.0
    return W, (N - 1) / N * W + B
end

function _rhat(chains::Vector{Vector{Float64}})
    W, vp = _wvar(chains)
    return W > 0 ? sqrt(vp / W) : NaN
end

"""
    rhat(chains) -> Float64

Rank-normalized split R-hat: the maximum of the bulk (rank-normalized) and tail (folded)
versions. Values above about 1.01 mean the chains disagree. NaN if every draw is equal.
"""
function rhat(chains)
    c = _split(_aschains(chains))
    all(x -> all(==(c[1][1]), x), c) && return NaN
    bulk = _rhat(_ranknorm(c))
    med = median(reduce(vcat, c))
    tail = _rhat(_ranknorm([abs.(x .- med) for x in c]))
    return max(bulk, tail)
end

## multi-chain ESS (Stan): rho_t = 1 - (W - mean_m acov_m(t)) / var_plus, summed with Geyer's
## initial monotone sequence over pairs of lags
function _ess(chains::Vector{Vector{Float64}}; block::Int=64)
    M = length(chains)
    N = length(chains[1])
    W, vp = _wvar(chains)
    (vp > 0 && N >= 4) || return NaN
    tau = -1.0
    Pprev = Inf
    t = 0
    while t + 1 <= N - 1
        lags = collect(t:min(t + 2block - 1, N - 1))
        ## autocovariances with the 1/N normalisation (as Stan), averaged over chains
        ac = mean(StatsBase.autocov(c, lags; demean=true) for c in chains)
        rho = 1 .- (W .- ac) ./ vp
        for k in 1:2:(length(rho) - 1)
            P = rho[k] + rho[k + 1]
            P < 0 && return M * N / max(tau, 1 / log10(M * N))
            P = min(P, Pprev)
            Pprev = P
            tau += 2P
        end
        t = last(lags) + 1 - (isodd(length(lags)) ? 1 : 0)
    end
    return M * N / max(tau, 1 / log10(M * N))
end

"""
    ess_bulk(chains) -> Float64

Bulk effective sample size: the ESS of the rank-normalized split chains.
"""
ess_bulk(chains) = _ess(_ranknorm(_split(_aschains(chains))))

"""
    ess_mean(chains) -> Float64

Effective sample size of the raw split chains, for the Monte Carlo standard error of the mean,
sd / sqrt(ess_mean).
"""
ess_mean(chains) = _ess(_split(_aschains(chains)))

## --------------------------------------------------------------------------- #
## label-invariant traces from a run

"""
    trace_ate(out, input) -> Vector{Float64}

Per-draw ATE on the data scale, sy * sum_j (n_j / n) xbar'(beta1_j - beta0_j), over the occupied
components, with xbar the covariate means.
"""
function trace_ate(out::GibbsOut, input::GibbsInput)
    kx, kz, n = input.dims.kx, input.dims.kz, input.dims.n
    xbar = vec(mean(input.data.X, dims=1))
    sy = input.data.y.s[1]
    i1, i0 = (kz + 1):(kz + kx), (kz + kx + 1):(kz + 2kx)
    return [sy * sum(dp.njs[j] / n * dot(xbar, view(th[j].beta, i1) .- view(th[j].beta, i0))
                     for j in eachindex(dp.njs) if dp.njs[j] > 0)
            for (dp, th) in zip(out.out_dp, out.out_theta)]
end

"Per-draw number of occupied components."
trace_k(out::GibbsOut) = [count(>(0), dp.njs) for dp in out.out_dp]

"Per-draw concentration parameter."
trace_alpha(out::GibbsOut) = [dp.alpha for dp in out.out_dp]

"""
    chain_diagnostics(outs, input; burn=0) -> Dict

For several chains `outs` (GibbsOut, same input), the mean, sd, R-hat, bulk ESS and mean ESS of
the per-draw ATE, k and alpha after discarding `burn` saved draws.
"""
function chain_diagnostics(outs::AbstractVector{GibbsOut}, input::GibbsInput; burn::Integer=0)
    res = Dict{Symbol,NamedTuple}()
    for (nm, f) in ((:ate, o -> trace_ate(o, input)), (:k, trace_k), (:alpha, trace_alpha))
        ch = [f(o)[(burn + 1):end] for o in outs]
        all_ = reduce(vcat, ch)
        res[nm] = (mean=mean(all_), sd=std(all_), rhat=rhat(ch), ess_bulk=ess_bulk(ch), ess_mean=ess_mean(ch),
                   chain_means=mean.(ch))
    end
    return res
end
