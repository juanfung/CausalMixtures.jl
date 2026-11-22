## --------------------------------------------------------------------------- #
## Design A sampler (model = "marginal"), Phase 4a: docs/implementation_plan.md section 06.
##
## Unified parameterization:  z_i = H^c_i beta^c_{c_i} + H^g_i b + e_i,  e_i ~ N3(0, Sigma_{c_i}).
##   slopes = :cluster: H^c_i = H_i = blockdiag(w_i', x_i', x_i'), beta^c = (gamma, beta1, beta0);
##                      no global part.
##   slopes = :common:  H^c_i = I_3, beta^c_j = (a_d, a_1, a_0) holds the cluster intercepts;
##                      H^g_i = blockdiag(w~_i', x~_i', x~_i') holds the shared slopes b (the
##                      design without its intercept columns).
## Priors: beta^c ~ N(mu_c, V_c), b ~ N(mu_g, V_g) (the intercept and slope blocks of
## prior_beta, which must not be correlated with each other), Sigma ~ NobileWishart(rho, rho R),
## alpha ~ Gamma(shape, rate).
##
## One sweep (no R3 collapsed sigma23 step yet):
##  (i)   labels: Neal's Algorithm 8 over Sigma with beta^c integrated out. Default (Phase 4b,
##        label_step = :r2): the joint label-and-latent step R2, which scores the observed data
##        (d_i, y_i) only and redraws (d*_i, y_miss,i) from the chosen cluster's predictive.
##        label_step = :augmented (Phase 4a): scored on the full augmented z~_i = z_i - H^g_i b. Existing cluster j: log n_{-i,j} +
##        log N3(z~_i | H^c_i Q_{j,-i}^-1 r_{j,-i}, Sigma_j + H^c_i Q_{j,-i}^-1 H^c_i'); auxiliary
##        cluster: log(alpha/m) + log N3(z~_i | H^c_i mu_c, Sigma_aux + H^c_i V_c H^c_i'), m = 2,
##        a singleton's Sigma in auxiliary slot 1. Per-cluster Sigma-free statistics are updated
##        incrementally as units move.
##  (ii)  beta^c_j | Sigma_j, z, b ~ N(Q_j^-1 r_j, Q_j^-1)
##  (iii) b | everything (:common only), accumulated per cluster with Sigma_j^-1 weights
##  (iv)  Sigma_j ~ NobileWishart(rho + n_j, rho R + sum e e')
##  (v)   latent data: the y_miss and d* conditionals of update_latent!
##  (vi)  alpha: Escobar-West with the random indicator (update_alpha!)
## Steps (i)-(ii) are a partially collapsed Gibbs kernel: (i) samples c | Sigma, z, b with beta^c
## integrated out, and (ii) redraws every beta^c from its full conditional.
##
## The state keeps the FULL coefficient vector of each cluster, beta = (gamma, beta1, beta0), in
## state_theta[j].beta: beta[cidx] = beta^c_j and beta[gidx] = b (the same b in every cluster).
## So the saved output, the PPD and trace_ate read it like the other samplers' output.

export dpm_marginal!, DesignA

## --------------------------------------------------------------------------- #
## fixed quantities for a run

struct DesignA
    slopes::Symbol
    cidx::Vector{Int}              # positions of beta^c in the full beta
    gidx::Vector{Int}              # positions of b in the full beta
    Wc::Matrix{Float64}            # cluster design: selection rows (:cluster W, :common ones)
    Xc::Matrix{Float64}            # cluster design: outcome rows (:cluster X, :common ones)
    Wg::Matrix{Float64}            # global design: selection rows without the intercept
    Xg::Matrix{Float64}            # global design: outcome rows without the intercept
    mu_c::Vector{Float64}
    Vc::Matrix{Float64}
    Vc_inv::Matrix{Float64}
    Vc_inv_mu::Vector{Float64}
    Lc::LowerTriangular{Float64,Matrix{Float64}} # factor of V_c, for prior draws
    mu_g::Vector{Float64}
    Vg_inv::Matrix{Float64}
    Vg_inv_mu::Vector{Float64}
    aux_mean::Vector{NTuple{3,Float64}} # H^c_i mu_c
    aux_cov::Vector{Matrix{Float64}}    # H^c_i V_c H^c_i'
    m_aux::Int
end

function DesignA(input::GibbsInput; m_aux::Integer=2)
    slopes = input.params.slopes
    kx, kz, n = input.dims.kx, input.dims.kz, input.dims.n
    X, W = input.data.X, input.data.W
    pb = input.priors.prior_theta.prior_beta
    ktot = kz + 2kx
    if slopes == :cluster
        cidx = collect(1:ktot)
        Wc, Xc = W, X
        Wg, Xg = zeros(n, 0), zeros(n, 0)
    elseif slopes == :common
        xi = findfirst(==("(Intercept)"), input.data.xnames)
        zi = findfirst(==("(Intercept)"), input.data.znames)
        (xi === nothing || zi === nothing) &&
            throw(ArgumentError("slopes = :common needs an intercept in both formulas"))
        cidx = [zi, kz + xi, kz + kx + xi]
        Wc, Xc = ones(n, 1), ones(n, 1)
        Wg, Xg = W[:, setdiff(1:kz, zi)], X[:, setdiff(1:kx, xi)]
    else
        throw(ArgumentError("slopes must be :common or :cluster, got $slopes"))
    end
    gidx = setdiff(1:ktot, cidx)
    V0 = pb.Vcov
    maximum(abs, V0[cidx, gidx]; init=0.0) == 0 ||
        throw(ArgumentError("slopes = :common needs a prior with no correlation between intercepts and slopes"))
    mu_c, Vc = pb.mu[cidx], Matrix(Symmetric(V0[cidx, cidx]))
    Vc_inv = Matrix(Symmetric(inv(Symmetric(Vc))))
    mu_g, Vg = pb.mu[gidx], V0[gidx, gidx]
    Vg_inv = isempty(gidx) ? zeros(0, 0) : Matrix(Symmetric(inv(Symmetric(Vg))))
    aux_mean = Vector{NTuple{3,Float64}}(undef, n)
    aux_cov = Vector{Matrix{Float64}}(undef, n)
    for i in 1:n
        Hc = cluster_row(Wc, Xc, i)
        aux_mean[i] = unit_mean(Xc, Wc, mu_c, i)
        aux_cov[i] = Hc * Vc * Hc'
    end
    return DesignA(slopes, cidx, gidx, Wc, Xc, Wg, Xg, mu_c, Vc, Vc_inv, Vc_inv * mu_c,
                   cholesky(Symmetric(Vc)).L, mu_g, Vg_inv, Vg_inv * mu_g, aux_mean, aux_cov, m_aux)
end

## the 3 x kc design blockdiag(w_i', x_i', x_i') for the rows of W and X
function cluster_row(W::AbstractMatrix{Float64}, X::AbstractMatrix{Float64}, i::Integer)
    kz, kx = size(W, 2), size(X, 2)
    H = zeros(3, kz + 2kx)
    @inbounds for a in 1:kz
        H[1, a] = W[i, a]
    end
    @inbounds for a in 1:kx
        H[2, kz + a] = X[i, a]
        H[3, kz + kx + a] = X[i, a]
    end
    return H
end

## the shared slopes b, read from any cluster's full beta
global_slopes(state::GibbsState, D::DesignA) = state.state_theta[1].beta[D.gidx]

## z~_i = z_i - H^g_i b
@inline function ztilde(state::GibbsState, D::DesignA, b::AbstractVector{Float64}, i::Integer)
    z = unit_z(state.state_data, i)
    isempty(b) && return z
    gd, g1, g0 = unit_mean(D.Xg, D.Wg, b, i)
    return (z[1] - gd, z[2] - g1, z[3] - g0)
end

## --------------------------------------------------------------------------- #
## per-cluster Sigma-free statistics of the cluster design and z~

mutable struct ClusterStats
    n::Int
    WW::Matrix{Float64}
    WX::Matrix{Float64}
    XX::Matrix{Float64}
    Wz::Matrix{Float64} # sum_i w_i z~_i'  (kcz x 3)
    Xz::Matrix{Float64} # sum_i x_i z~_i'  (kcx x 3)
end

ClusterStats(D::DesignA) = (kz = size(D.Wc, 2); kx = size(D.Xc, 2);
    ClusterStats(0, zeros(kz, kz), zeros(kz, kx), zeros(kx, kx), zeros(kz, 3), zeros(kx, 3)))

## add (s = +1) or remove (s = -1) unit i with augmented latent data z
function stats_add!(st::ClusterStats, D::DesignA, i::Integer, z::NTuple{3,Float64}, s::Int)
    Wc, Xc = D.Wc, D.Xc
    kz, kx = size(Wc, 2), size(Xc, 2)
    st.n += s
    @inbounds for a in 1:kz
        wa = s * Wc[i, a]
        for b in 1:kz
            st.WW[a, b] += wa * Wc[i, b]
        end
        for b in 1:kx
            st.WX[a, b] += wa * Xc[i, b]
        end
        for c in 1:3
            st.Wz[a, c] += wa * z[c]
        end
    end
    @inbounds for a in 1:kx
        xa = s * Xc[i, a]
        for b in 1:kx
            st.XX[a, b] += xa * Xc[i, b]
        end
        for c in 1:3
            st.Xz[a, c] += xa * z[c]
        end
    end
    return st
end

function cluster_stats(state::GibbsState, D::DesignA, b::AbstractVector{Float64}, idx::AbstractVector{<:Integer})
    st = ClusterStats(D)
    for i in idx
        stats_add!(st, D, i, ztilde(state, D, b, i), 1)
    end
    return st
end

## Q = V_c^-1 + H'(Sinv ⊗ I)H and r = V_c^-1 mu_c + H'(Sinv ⊗ I) z~ from the statistics
function cluster_Qr(st::ClusterStats, D::DesignA, Sinv::AbstractMatrix{Float64})
    Q = D.Vc_inv + design_precision(Sinv, st.WW, st.WX, st.XX)
    r = D.Vc_inv_mu + vcat(st.Wz * Sinv[:, 1], st.Xz * Sinv[:, 2], st.Xz * Sinv[:, 3])
    return Q, r
end

## the cluster posterior of beta^c given Sigma: Cholesky of Q and the mean Q^-1 r
function cluster_post(st::ClusterStats, D::DesignA, Sinv::AbstractMatrix{Float64})
    Q, r = cluster_Qr(st, D, Sinv)
    C = cholesky(Symmetric(Q))
    return C, C \ r
end

## log N3(z~_i | H^c_i m, Sigma + H^c_i Q^-1 H^c_i'), the beta-collapsed predictive
function log_predictive(C::Cholesky, m::Vector{Float64}, Sigma::AbstractMatrix{Float64},
                        Hc::Matrix{Float64}, D::DesignA, i::Integer, z::NTuple{3,Float64})
    mu = unit_mean(D.Xc, D.Wc, m, i)
    S = Sigma + Hc * (C \ Matrix(Hc'))
    return logpdf3(Chol3(S), z[1] - mu[1], z[2] - mu[2], z[3] - mu[3])
end

## --------------------------------------------------------------------------- #
## (i) labels

## --------------------------------------------------------------------------- #
## R2 (Phase 4b): score a candidate cluster on the observed data only, then draw the latent data
## from its predictive. With z~ ~ N3(mu, S) the predictive of a candidate cluster, o = 2 if d = 1
## and 3 if d = 0, s = 2d - 1, y~ the observed outcome minus its global part, and g_d = (H^g_i b)_d
## the global part of the selection index (zero under slopes = :cluster),
##   p(d, y~) = N(y~ | mu_o, S_oo) * Phi(s * (mu_d + g_d + S_do/S_oo (y~ - mu_o)) / sqrt(S_dd - S_do^2/S_oo)),
## which is the integral of N3(z~ | mu, S) over the unobserved outcome and the half-line of
## z~_d that agrees with d: d = 1{d* > 0} with d* = z~_d + g_d, so the cut-off on z~_d is -g_d.

## log p(d, y~ | mu, S): the R2 observed-data score. `gd` is the global part of the selection
## index, (H^g_i b)_d: d = 1{d* > 0} with d* = z~_d + gd, so the threshold on z~_d is -gd.
function r2_score(mu::NTuple{3,Float64}, S::AbstractMatrix{Float64}, yo::Real, d::Integer, gd::Real=0.0)
    o = d == 1 ? 2 : 3
    s = 2d - 1
    Soo = S[o, o]
    cm = mu[1] + gd + S[1, o] / Soo * (yo - mu[o])
    cv = S[1, 1] - S[1, o]^2 / Soo
    return -0.5 * (log(2pi * Soo) + (yo - mu[o])^2 / Soo) + logcdf(Normal(), s * cm / sqrt(cv))
end

## draw (d*, y_miss) from N3(mu, S) given the observed y~_o and the sign of d*; returns z~
function r2_draw(rng::AbstractRNG, mu::NTuple{3,Float64}, S::AbstractMatrix{Float64}, yo::Real, d::Integer,
                 gd::Real=0.0)
    o, mi = d == 1 ? (2, 3) : (3, 2)
    Soo = S[o, o]
    ## z~_d | y~_o, truncated to the side of -gd given by d (d* = z~_d + gd on the side of 0)
    cm = mu[1] + S[1, o] / Soo * (yo - mu[o])
    cv = S[1, 1] - S[1, o]^2 / Soo
    ds = rand(rng, truncated(Normal(cm, sqrt(cv)), d == 1 ? -gd : -Inf, d == 1 ? Inf : -gd))
    ## y_miss | d*, y~_o: Gaussian conditional on the coordinates (1, o)
    a11, a12, a22 = S[1, 1], S[1, o], Soo
    det = a11 * a22 - a12^2
    c1, c2 = S[mi, 1], S[mi, o]
    k1 = (c1 * a22 - c2 * a12) / det # S_{m,(1,o)} S_{(1,o),(1,o)}^-1
    k2 = (c2 * a11 - c1 * a12) / det
    ym = mu[mi] + k1 * (ds - mu[1]) + k2 * (yo - mu[o]) + sqrt(max(S[mi, mi] - k1 * c1 - k2 * c2, 0.0)) * randn(rng)
    return d == 1 ? (ds, Float64(yo), ym) : (ds, ym, Float64(yo))
end

## The label step keeps, per slot, Q^-1 and m = Q^-1 r of the cluster posterior of beta^c given
## Sigma. Q = V_c^-1 + H'(Sigma^-1 (x) I)H depends on the members' design and Sigma only, not on
## z~, so when unit i returns to the cluster it left, Q^-1 from before its removal is restored
## and only r and m are recomputed; Q is refactored on a removal and on a move.
mutable struct PostCache
    Qinv::Matrix{Float64}
    r::Vector{Float64}
    m::Vector{Float64}
end
PostCache(kc::Integer) = PostCache(zeros(kc, kc), zeros(kc), zeros(kc))

## Q^-1 from the statistics (design part only)
function post_Q!(pc::PostCache, st::ClusterStats, D::DesignA, Sinv::AbstractMatrix{Float64})
    kz, kx = size(st.WX)
    g, b1, b0 = coef_blocks(kx, kz)
    Q = pc.Qinv
    copyto!(Q, D.Vc_inv)
    @inbounds for (ra, ia) in ((g, 1), (b1, 2), (b0, 3)), (rb, ib) in ((g, 1), (b1, 2), (b0, 3))
        s = Sinv[ia, ib]
        for (q, cb) in enumerate(rb), (p, ca) in enumerate(ra)
            Q[ca, cb] += s * (ia == 1 ? (ib == 1 ? st.WW[p, q] : st.WX[p, q]) :
                              (ib == 1 ? st.WX[q, p] : st.XX[p, q]))
        end
    end
    _, info = LAPACK.potrf!('U', Q)
    info == 0 || throw(PosDefException(info))
    LAPACK.potri!('U', Q)
    @inbounds for c in axes(Q, 2), a in (c + 1):size(Q, 1)
        Q[a, c] = Q[c, a]
    end
    return pc
end

## r = V_c^-1 mu_c + H'(Sigma^-1 (x) I) z~ and m = Q^-1 r
function post_m!(pc::PostCache, st::ClusterStats, D::DesignA, Sinv::AbstractMatrix{Float64})
    kz, kx = size(st.WX)
    r = pc.r
    copyto!(r, D.Vc_inv_mu)
    @inbounds for c in 1:3
        for a in 1:kz
            r[a] += st.Wz[a, c] * Sinv[c, 1]
        end
        for a in 1:kx
            r[kz + a] += st.Xz[a, c] * Sinv[c, 2]
            r[kz + kx + a] += st.Xz[a, c] * Sinv[c, 3]
        end
    end
    mul!(pc.m, pc.Qinv, r)
    return pc
end

post_cache!(pc, st, D, Sinv) = post_m!(post_Q!(pc, st, D, Sinv), st, D, Sinv)

## the beta^c-collapsed predictive of z~_i for a cluster: mean H^c_i m and, into S,
## Sigma + H^c_i Q^-1 H^c_i', with H^c_i = blockdiag(w_i', x_i', x_i')
function predictive!(S::Matrix{Float64}, pc::PostCache, Sigma::AbstractMatrix{Float64}, D::DesignA, i::Integer)
    Wc, Xc = D.Wc, D.Xc
    kz, kx = size(Wc, 2), size(Xc, 2)
    V = pc.Qinv
    @inbounds for ib in 1:3, ia in 1:ib
        oa = ia == 1 ? 0 : (ia == 2 ? kz : kz + kx)
        ob = ib == 1 ? 0 : (ib == 2 ? kz : kz + kx)
        na = ia == 1 ? kz : kx
        nb = ib == 1 ? kz : kx
        acc = 0.0
        for q in 1:nb
            hq = ib == 1 ? Wc[i, q] : Xc[i, q]
            t = 0.0
            for p in 1:na
                t += (ia == 1 ? Wc[i, p] : Xc[i, p]) * V[oa + p, ob + q]
            end
            acc += t * hq
        end
        S[ia, ib] = Sigma[ia, ib] + acc
        S[ib, ia] = S[ia, ib]
    end
    return unit_mean(Xc, Wc, pc.m, i)
end

## (i) labels. label_step = :r2 (default): the joint label-and-latent step R2, scoring the
## observed data and redrawing unit i's latent data from the chosen cluster's predictive.
## label_step = :augmented (Phase 4a): score the current augmented z~_i, latent data unchanged.
function update_marginal_labels!(rng::AbstractRNG, state::GibbsState, input::GibbsInput, D::DesignA;
                                 r2::Bool = input.params.label_step == :r2)
    dp = state.state_dp
    ps = input.priors.prior_theta.prior_Sigma
    G0 = NobileFactor(ps.rho .* ps.R)
    b = global_slopes(state, D)
    K = length(dp.njs)
    sd = state.state_data
    y, dobs = input.data.y.a, input.data.d
    kc = length(D.cidx)
    ## per-slot statistics, Sigma^-1 and posterior caches
    stats = [cluster_stats(state, D, b, dp.members[j]) for j in 1:K]
    Sinv = [inv(state.state_theta[j].Sigma) for j in 1:K]
    post = [post_cache!(PostCache(kc), stats[j], D, Sinv[j]) for j in 1:K]
    saved = zeros(kc, kc) # Q^-1 of unit i's cluster before its removal
    m = D.m_aux
    aux = Vector{Matrix{Float64}}(undef, m)
    logw = Float64[]
    mus = NTuple{3,Float64}[]
    Ss = Matrix{Float64}[]
    @inbounds for i in 1:input.dims.n
        z = ztilde(state, D, b, i)
        j0 = dp.labels[i]
        dp.labels[i] = 0
        dp.njs[j0] -= 1
        stats_add!(stats[j0], D, i, z, -1)
        ## auxiliary Sigmas: a singleton's own Sigma in slot 1, fresh G0 draws elsewhere
        if dp.njs[j0] == 0
            aux[1] = state.state_theta[j0].Sigma
        else
            aux[1] = NobileWishart(rng, ps.rho, G0)
            copyto!(saved, post[j0].Qinv)
            post_cache!(post[j0], stats[j0], D, Sinv[j0])
        end
        for a in 2:m
            aux[a] = NobileWishart(rng, ps.rho, G0)
        end
        K = length(dp.njs)
        while length(Ss) < K + m
            push!(Ss, zeros(3, 3))
        end
        resize!(logw, K + m); resize!(mus, K + m)
        ## the observed outcome minus its global part
        g = isempty(b) ? (0.0, 0.0, 0.0) : unit_mean(D.Xg, D.Wg, b, i)
        di = dobs[i]
        yo = y[i] - (di == 1 ? g[2] : g[3])
        free = 0
        for j in 1:K
            if dp.njs[j] > 0
                mus[j] = predictive!(Ss[j], post[j], state.state_theta[j].Sigma, D, i)
                logw[j] = log(dp.njs[j]) + (r2 ? r2_score(mus[j], Ss[j], yo, di, g[1]) :
                    logpdf3(Chol3(Ss[j]), z[1] - mus[j][1], z[2] - mus[j][2], z[3] - mus[j][3]))
            else
                logw[j] = -Inf
                free == 0 && (free = j)
            end
        end
        for a in 1:m
            mu, S = D.aux_mean[i], Ss[K + a]
            S .= aux[a] .+ D.aux_cov[i]
            mus[K + a] = mu
            logw[K + a] = log(dp.alpha / m) + (r2 ? r2_score(mu, S, yo, di, g[1]) :
                logpdf3(Chol3(S), z[1] - mu[1], z[2] - mu[2], z[3] - mu[3]))
        end
        ji = sample_logweights(rng, logw)
        if r2
            ## the latent data from the chosen cluster's predictive, given d and y~
            z = r2_draw(rng, mus[ji], Ss[ji], yo, di, g[1])
            sd.dstar[i], sd.y1[i], sd.y0[i] = z[1] + g[1], z[2] + g[2], z[3] + g[3]
        end
        stay = ji == j0 # j0 is occupied here: an emptied slot scores -Inf
        if ji > K
            ## open a cluster with the chosen auxiliary Sigma
            Sig = aux[ji - K]
            th = Theta(beta=copy(state.state_theta[1].beta), Sigma=Sig) # beta^c is redrawn in (ii)
            if free > 0
                ji = free
                state.state_theta[ji] = th
                stats[ji] = ClusterStats(D)
                Sinv[ji] = inv(Sig)
            else
                push!(dp.njs, 0)
                push!(state.state_theta, th)
                push!(stats, ClusterStats(D))
                push!(Sinv, inv(Sig))
                push!(post, PostCache(kc))
                ji = length(dp.njs)
            end
        end
        dp.labels[i] = ji
        dp.njs[ji] += 1
        stats_add!(stats[ji], D, i, z, 1)
        if stay
            copyto!(post[ji].Qinv, saved)
            post_m!(post[ji], stats[ji], D, Sinv[ji])
        else
            post_cache!(post[ji], stats[ji], D, Sinv[ji])
        end
    end
    compact_components!(state)
    return state
end

## --------------------------------------------------------------------------- #
## (ii)-(iv) cluster coefficients, shared slopes, Sigma

## full beta of a cluster from beta^c and b
function full_beta(D::DesignA, bc::AbstractVector{Float64}, b::AbstractVector{Float64})
    beta = Vector{Float64}(undef, length(D.cidx) + length(D.gidx))
    beta[D.cidx] = bc
    beta[D.gidx] = b
    return beta
end

## (ii) beta^c_j | Sigma_j, z, b for every cluster
function update_marginal_coefs!(rng::AbstractRNG, state::GibbsState, D::DesignA)
    b = global_slopes(state, D)
    for j in 1:state.state_dp.J
        th = state.state_theta[j]
        st = cluster_stats(state, D, b, state.state_dp.members[j])
        C, mj = cluster_post(st, D, inv(th.Sigma))
        bc = mj + C.U \ randn(rng, length(mj))
        state.state_theta[j] = Theta(beta=full_beta(D, bc, b), Sigma=th.Sigma)
    end
    return state
end

## (iii) b | beta^c, Sigma, z (:common only):
## P = V_g^-1 + sum_j H^g_j'(Sinv_j ⊗ I)H^g_j, s = V_g^-1 mu_g + sum_j H^g_j'(Sinv_j ⊗ I)(z - H^c beta^c)
function update_global_slopes!(rng::AbstractRNG, state::GibbsState, D::DesignA)
    isempty(D.gidx) && return state
    P = copy(D.Vg_inv)
    s = copy(D.Vg_inv_mu)
    sd = state.state_data
    for j in 1:state.state_dp.J
        idx = state.state_dp.members[j]
        th = state.state_theta[j]
        Sinv = inv(th.Sigma)
        bc = th.beta[D.cidx]
        Wg, Xg = D.Wg[idx, :], D.Xg[idx, :]
        U = Matrix{Float64}(undef, length(idx), 3) # z - H^c beta^c
        for (r, i) in enumerate(idx)
            cd, c1, c0 = unit_mean(D.Xc, D.Wc, bc, i)
            U[r, 1] = sd.dstar[i] - cd
            U[r, 2] = sd.y1[i] - c1
            U[r, 3] = sd.y0[i] - c0
        end
        P .+= design_precision(Sinv, Wg' * Wg, Wg' * Xg, Xg' * Xg)
        US = U * Sinv
        s .+= vcat(Wg' * US[:, 1], Xg' * US[:, 2], Xg' * US[:, 3])
    end
    C = cholesky(Symmetric(P))
    b = C \ s + C.U \ randn(rng, length(s))
    for j in 1:state.state_dp.J
        th = state.state_theta[j]
        state.state_theta[j] = Theta(beta=full_beta(D, th.beta[D.cidx], b), Sigma=th.Sigma)
    end
    return state
end

## (iv) Sigma_j | beta_j, b, z with the full residuals
function update_marginal_sigmas!(rng::AbstractRNG, state::GibbsState, input::GibbsInput)
    ps = input.priors.prior_theta.prior_Sigma
    X, W = input.data.X, input.data.W
    sd = state.state_data
    for j in 1:state.state_dp.J
        th = state.state_theta[j]
        U = ps.rho .* ps.R
        for i in state.state_dp.members[j]
            md, m1, m0 = unit_mean(X, W, th.beta, i)
            e = (sd.dstar[i] - md, sd.y1[i] - m1, sd.y0[i] - m0)
            for a in 1:3, c in 1:3
                U[a, c] += e[a] * e[c]
            end
        end
        state.state_theta[j] = Theta(beta=th.beta, Sigma=NobileWishart(rng, ps.rho + state.state_dp.njs[j], U))
    end
    return state
end

## one Design A sweep, (i)-(vi); test = true keeps the latent data fixed
function marginal_sweep!(rng::AbstractRNG, state::GibbsState, input::GibbsInput, D::DesignA; test::Bool=false)
    ## (i) R2 by default; test = true keeps the latent data fixed, so it scores z~ instead
    update_marginal_labels!(rng, state, input, D; r2 = input.params.label_step == :r2 && !test)
    update_marginal_coefs!(rng, state, D)                # (ii)
    update_global_slopes!(rng, state, D)                 # (iii)
    update_marginal_sigmas!(rng, state, input)           # (iv)
    if !test                                             # (v)
        for j in 1:state.state_dp.J
            update_latent!(rng, state, input, j, state.state_dp.members[j])
        end
    end
    if input.priors.prior_dp.alpha_shape != 0.0          # (vi)
        update_alpha!(rng, state, input)
    end
    return state
end

function dpm_marginal!(state::GibbsState, input::GibbsInput, out::GibbsOut; test=false,
                       rng::AbstractRNG=Random.default_rng())
    M = input.params.M
    verbose = input.params.verbose
    D = DesignA(input)
    if verbose
        @printf("Batch %d (Design A, slopes = %s)\nInitial J = %d\nInitial alpha = %f\nBegin sampler...",
                state.state_sampler.batch_n, D.slopes, state.state_dp.J, state.state_dp.alpha)
    end
    for m in 1:M
        marginal_sweep!(rng, state, input, D; test=test)
        if verbose && mod(m, max(1, M ÷ 10)) == 0
            @printf("\nIteration %d: J = %d, alpha = %f", m + state.state_sampler.batch_m,
                    state.state_dp.J, state.state_dp.alpha)
        end
        out = update_out!(state, out, m)
    end
    if verbose @printf("\nSampler run complete.\n") end
    state = update_sampler!(state, M)
    return (state, input, out)
end

## --------------------------------------------------------------------------- #
## posterior predictive draws for a new unit with design hnew (3 x ktot)

function rand_marginal(rng::AbstractRNG, out::GibbsOut, input::GibbsInput, hnew::SparseMatrixCSC{Float64,Int64})
    D = DesignA(input)
    ps = input.priors.prior_theta.prior_Sigma
    rhoR = ps.rho .* ps.R
    ynew = Matrix{Float64}(undef, 3, length(out.out_dp))
    for m in eachindex(out.out_dp)
        dp, th = out.out_dp[m], out.out_theta[m]
        w = Float64[dp.njs[j] for j in eachindex(dp.njs)]
        push!(w, dp.alpha)
        jm = rand(rng, Categorical(w ./ sum(w)))
        if jm <= length(dp.njs)
            beta, Sigma = th[jm].beta, th[jm].Sigma
        else
            ## a new cluster: beta^c from its prior, the shared b from the draw, Sigma from G0
            beta = full_beta(D, D.mu_c + D.Lc * randn(rng, length(D.mu_c)), th[1].beta[D.gidx])
            Sigma = NobileWishart(rng, ps.rho, rhoR)
        end
        ynew[:, m] = hnew * beta + cholesky(Symmetric(Sigma)).L * randn(rng, 3)
    end
    return ynew
end
