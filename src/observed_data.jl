## --------------------------------------------------------------------------- #
## Observed-data posterior of one Gaussian component, without data augmentation (Phase 5b; the
## k = 1 prototype of option D). For unit i with selection index s = w'gamma:
##   treated: log N(y | x'b1, s22) + log Phi((s + (s12/s22)(y - x'b1)) / sqrt(1 - s12^2/s22))
##   control: log N(y | x'b0, s33) + log Phi(-(s + (s13/s33)(y - x'b0)) / sqrt(1 - s13^2/s33))
## (sigma11 = 1; the unobserved outcome and d* integrated out). sigma23 does not enter.
## Parameters theta = (beta (gamma, b1, b0), delta = (s12, s13), eta), with
## Omega = Var(y1, y0 | d*) = L L', L = [exp(eta1) 0; eta2 exp(eta3)] (log-Cholesky), and
## Sigma = [1 delta'; delta Omega + delta delta'] (src/ridge_block.jl). Prior: beta ~ N(mu, V0),
## delta | Omega ~ N(c, Omega/Psi11), Omega ~ IW_2(rho, Psi_22.1) (the NobileWishart(rho, rho R)
## prior, Phase 4f), plus the log-Jacobian of eta -> Omega, log 4 + 3 eta1 + 2 eta3.

export obs_loglik, obsdata_logpost, obsdata_sampler, obsdata_out

## observed-data log-likelihood of the units idx under one component (model scale)
function obs_loglik(beta::AbstractVector{Float64}, Sigma::AbstractMatrix{Float64}, X::AbstractMatrix{Float64},
                    W::AbstractMatrix{Float64}, y::AbstractVector{Float64}, d::AbstractVector{<:Integer},
                    idx=eachindex(y))
    kz, kx = size(W, 2), size(X, 2)
    s22, s33, s12, s13 = Sigma[2, 2], Sigma[3, 3], Sigma[1, 2], Sigma[1, 3]
    (s22 > s12^2 && s33 > s13^2) || return -Inf
    c1, c0 = s12 / s22, s13 / s33
    q1, q0 = sqrt(1 - s12^2 / s22), sqrt(1 - s13^2 / s33)
    ll = 0.0
    @inbounds for i in idx
        s = 0.0
        for a in 1:kz
            s += W[i, a] * beta[a]
        end
        m = 0.0
        if d[i] == 1
            for a in 1:kx
                m += X[i, a] * beta[kz + a]
            end
            r = y[i] - m
            ll += -0.5 * (log(2pi * s22) + r^2 / s22) + logcdf(Normal(), (s + c1 * r) / q1)
        else
            for a in 1:kx
                m += X[i, a] * beta[kz + kx + a]
            end
            r = y[i] - m
            ll += -0.5 * (log(2pi * s33) + r^2 / s33) + logcdf(Normal(), -(s + c0 * r) / q0)
        end
    end
    return ll
end

## theta -> (beta, delta, Omega, log-Jacobian of eta -> Omega)
function obsdata_unpack(theta::AbstractVector{Float64}, ktot::Integer)
    beta = theta[1:ktot]
    d = theta[ktot+1:ktot+2]
    eta1, eta2, eta3 = theta[ktot+3], theta[ktot+4], theta[ktot+5]
    l11, l21, l22 = exp(eta1), eta2, exp(eta3)
    O = [l11^2 l11 * l21; l11 * l21 l21^2 + l22^2]
    return beta, d, O, log(4.0) + 3 * eta1 + 2 * eta3
end

function obsdata_pack(beta::AbstractVector{Float64}, Sigma::AbstractMatrix{Float64})
    d, O = sigma_delta_omega(Sigma)
    L = cholesky(Symmetric(O)).L
    return vcat(beta, d, log(L[1, 1]), L[2, 1], log(L[2, 2]))
end

"""
    obsdata_logpost(theta, X, W, y, d, pb, rp; idx) -> Float64

Log posterior (up to a constant) of one Gaussian component on the observed data of units `idx`,
in the parameterization of `obsdata_unpack`; `pb` the prior of beta, `rp` the `RidgePrior`.
"""
function obsdata_logpost(theta::AbstractVector{Float64}, X, W, y, d, pb::PriorBeta, rp::RidgePrior; idx=eachindex(y))
    ktot = length(pb.mu)
    beta, dl, O, lj = obsdata_unpack(theta, ktot)
    S = sigma_from_delta_omega(dl, O)
    ll = obs_loglik(beta, S, X, W, y, d, idx)
    isfinite(ll) || return -Inf
    lp = logpdf(MvNormal(pb.mu, Symmetric(pb.Vcov)), beta)
    lp += logpdf(MvNormal(rp.c, Symmetric(O ./ rp.Psi11)), dl)
    lp += logpdf(InverseWishart(rp.nu, rp.Psi221), Matrix(Symmetric(O)))
    return ll + lp + lj
end

"""
    obsdata_sampler(rng, input, theta0; M, burn, thin = 1) -> (draws, accept)

Random-walk Metropolis on the observed-data posterior of the k = 1 Gaussian selection model
(`obsdata_logpost`, the C3 priors of `input`, the sampler's standardized data). The proposal is
N(0, 2.38^2/p C) with C adapted from the chain during the first `burn` iterations (Haario et
al. 2001) and frozen afterwards; only post-burn-in draws are returned (every `thin`-th), as rows
of `draws`. `accept` is the post-burn-in acceptance rate. `idx`: the units whose data enter (all
by default; empty gives the prior).
"""
function obsdata_sampler(rng::AbstractRNG, input::GibbsInput, theta0::AbstractVector{Float64};
                         M::Integer=100_000, burn::Integer=20_000, thin::Integer=1,
                         idx::AbstractVector{<:Integer}=1:input.dims.n)
    X, W, y, d = input.data.X, input.data.W, input.data.y.a, input.data.d
    pb = input.priors.prior_theta.prior_beta
    rp = RidgePrior(input.priors.prior_theta.prior_Sigma)
    p = length(theta0)
    lpf(t) = obsdata_logpost(t, X, W, y, d, pb, rp; idx=idx)
    th = copy(theta0)
    lp = lpf(th)
    isfinite(lp) || throw(ArgumentError("theta0 has zero posterior density"))
    sc = 2.38^2 / p
    C = Matrix(0.01I, p, p)
    mu, Sacc, nacc = copy(th), zeros(p, p), 0
    L = cholesky(Symmetric(sc .* C)).L
    draws = Matrix{Float64}(undef, M ÷ thin, p)
    acc = 0
    for it in 1:(burn + M)
        prop = th + L * randn(rng, p)
        lq = lpf(prop)
        if log(rand(rng)) < lq - lp
            th, lp = prop, lq
            it > burn && (acc += 1)
        end
        if it <= burn
            ## recursive mean and covariance of the chain so far; refresh the proposal every 200
            nacc += 1
            dlt = th - mu
            mu .+= dlt ./ nacc
            Sacc .+= (nacc - 1) / nacc .* (dlt * dlt')
            if it % 200 == 0 && nacc > 2p
                C = Sacc ./ (nacc - 1) + 1e-8I
                L = cholesky(Symmetric(sc .* C)).L
            end
        elseif (it - burn) % thin == 0
            draws[(it - burn) ÷ thin, :] = th
        end
    end
    return draws, acc / M
end

"""
    obsdata_out(draws, input) -> GibbsOut

Saved draws of `obsdata_sampler` as a one-component `GibbsOut`, so `effects(out, input)` (with a
`gaussian` input) gives PATE, PATT and PATU per draw.
"""
function obsdata_out(draws::AbstractMatrix{Float64}, input::GibbsInput)
    ktot = length(input.priors.prior_theta.prior_beta.mu)
    S = size(draws, 1)
    out = GibbsOut(S)
    for s in 1:S
        beta, dl, O, _ = obsdata_unpack(draws[s, :], ktot)
        out.out_theta[s] = [Theta(beta=beta, Sigma=sigma_from_delta_omega(dl, O))]
        out.out_dp[s] = DrawDP(1, [input.dims.n], BlockedWeights[], 0.0, 0.0)
        out.out_hyper[s] = Float64[]
        out.out_sep[s] = 0.0
        out.out_r[s] = Float64[]
    end
    return out
end
