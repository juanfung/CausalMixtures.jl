## functions to update GibbsState
##
## Model: z_i = (d*_i, y1_i, y0_i)' = H_i beta_{c_i} + e_i, e_i ~ N3(0, Sigma_{c_i}), sigma11 = 1,
## H_i = blockdiag(w_i', x_i', x_i'), beta = (gamma, beta1, beta0). The design is used through the
## dense X (n x kx) and W (n x kz) in InputData: per-unit means are dot products with a row, and
## a component's precision H_j'(Sigma^-1 ⊗ I)H_j is built from the Sigma-free cross-products
## W_j'W_j, W_j'X_j, X_j'X_j (design_precision).

## --------------------------------------------------------------------------- #
## design helpers

## positions of gamma, beta1, beta0 in beta = (gamma, beta1, beta0)
@inline coef_blocks(kx::Integer, kz::Integer) = (1:kz, (kz+1):(kz+kx), (kz+kx+1):(kz+2kx))

## the mean H_i beta = (w_i'gamma, x_i'beta1, x_i'beta0) of unit i
@inline function unit_mean(X::AbstractMatrix{Float64}, W::AbstractMatrix{Float64},
                           beta::AbstractVector{Float64}, i::Integer)
    kx, kz = size(X, 2), size(W, 2)
    md = 0.0
    @inbounds for a in 1:kz
        md += W[i, a] * beta[a]
    end
    m1 = 0.0
    m0 = 0.0
    @inbounds for a in 1:kx
        x = X[i, a]
        m1 += x * beta[kz + a]
        m0 += x * beta[kz + kx + a]
    end
    return md, m1, m0
end

## latent data of unit i
@inline unit_z(sd::StateData, i::Integer) = @inbounds (sd.dstar[i], sd.y1[i], sd.y0[i])

## log N3(z_i | H_i beta, Sigma), given the Cholesky factor c of Sigma
@inline function logpdf_unit(c::Chol3, beta::AbstractVector{Float64}, X, W, z::NTuple{3,Float64}, i::Integer)
    md, m1, m0 = unit_mean(X, W, beta, i)
    return logpdf3(c, z[1] - md, z[2] - m1, z[3] - m0)
end

## H'(Sinv ⊗ I)H for H = blockdiag(W, X, X), from WW = W'W, WX = W'X, XX = X'X
function design_precision(Sinv::AbstractMatrix{Float64}, WW::AbstractMatrix{Float64},
                          WX::AbstractMatrix{Float64}, XX::AbstractMatrix{Float64})
    kz, kx = size(WX)
    g, b1, b0 = coef_blocks(kx, kz)
    P = Matrix{Float64}(undef, kz + 2kx, kz + 2kx)
    @views begin
        P[g, g] .= Sinv[1,1] .* WW
        P[g, b1] .= Sinv[1,2] .* WX
        P[g, b0] .= Sinv[1,3] .* WX
        P[b1, g] .= Sinv[2,1] .* WX'
        P[b0, g] .= Sinv[3,1] .* WX'
        P[b1, b1] .= Sinv[2,2] .* XX
        P[b1, b0] .= Sinv[2,3] .* XX
        P[b0, b1] .= Sinv[3,2] .* XX
        P[b0, b0] .= Sinv[3,3] .* XX
    end
    return P
end

## Cholesky factors of every component's Sigma, computed once per sweep
component_chols(state_theta::StateTheta) = Chol3[Chol3(t.Sigma) for t in state_theta]

## --------------------------------------------------------------------------- #
## main sampler functions

## update component labels and number of components.
## A component emptied during the sweep keeps its slot (njs = 0, skipped by sample_label!) and
## a new component reuses a free slot; the occupied components are relabelled 1:k at the end.
## This is the same Polya-urn kernel as removing the component at once, without the O(n)
## relabelling scan each time a component empties.
function update_labels!(rng::AbstractRNG, state::GibbsState, input::GibbsInput)
    dp = state.state_dp
    chols = component_chols(state.state_theta)
    @inbounds for i in 1:input.dims.n
        ji = dp.labels[i]
        dp.labels[i] = 0 # remove i from component ji
        dp.njs[ji] -= 1
        ## sample component label
        state = sample_label!(rng, state, input, i, chols)
    end
    compact_components!(state)
    return state
end

## drop empty components, relabel the occupied ones 1:k in their current order, rebuild members
function compact_components!(state::GibbsState)
    dp = state.state_dp
    K = length(dp.njs)
    newlab = zeros(Int64, K)
    k = 0
    @inbounds for j in 1:K
        if dp.njs[j] > 0
            k += 1
            newlab[j] = k
            dp.njs[k] = dp.njs[j]
            state.state_theta[k] = state.state_theta[j]
        end
    end
    resize!(dp.njs, k)
    resize!(state.state_theta, k)
    @inbounds for i in eachindex(dp.labels)
        dp.labels[i] = newlab[dp.labels[i]]
    end
    dp.J = k
    rebuild_members!(dp)
    return state
end

## sample new theta from its posterior given unit i's latent data z alone
function sample_new_theta(rng::AbstractRNG, p::PriorTheta, state::GibbsState, input::GibbsInput, i::Integer,
                          z::NTuple{3,Float64})
    X, W = input.data.X, input.data.W
    ## i. Sigma ~ NobileWishart(rho + 1, u u' + rho R), u = z - H_i mu
    md, m1, m0 = unit_mean(X, W, p.prior_beta.mu, i)
    u = (z[1] - md, z[2] - m1, z[3] - m0)
    U = [u[a] * u[b] for a in 1:3, b in 1:3] .+ p.prior_Sigma.rho .* p.prior_Sigma.R
    Sigma_j = NobileWishart(rng, p.prior_Sigma.rho + 1, U)
    ## ii. beta ~ N(vj (H_i' Sinv z + V mu), vj), vj = (H_i' Sinv H_i + V)^-1
    Sinv = inv(Sigma_j)
    w = W[i, :]
    x = X[i, :]
    vj = inv(design_precision(Sinv, w * w', w * x', x * x') + p.prior_beta.V)
    s1 = Sinv[1,1]*z[1] + Sinv[1,2]*z[2] + Sinv[1,3]*z[3]
    s2 = Sinv[2,1]*z[1] + Sinv[2,2]*z[2] + Sinv[2,3]*z[3]
    s3 = Sinv[3,1]*z[1] + Sinv[3,2]*z[2] + Sinv[3,3]*z[3]
    mj = vj * (vcat(s1 .* w, s2 .* x, s3 .* x) .+ state.state_sampler.Vmu)
    beta_j = mj + cholesky( Hermitian(vj) ).U' * randn(rng, length(mj))
    return Theta(beta=beta_j, Sigma=Sigma_j)
end

## sample label for i
function sample_label!(rng::AbstractRNG, state::GibbsState, input::GibbsInput, i::Int64, chols::Vector{Chol3})
    X, W = input.data.X, input.data.W
    z = unit_z(state.state_data, i)
    ## log weights over the occupied components (empty slots get -Inf); the common factor
    ## 1/zdenom cancels
    dp = state.state_dp
    K = length(dp.njs)
    logw = Array{Float64}(undef, K + 1)
    free = 0
    @inbounds for j in 1:K
        if dp.njs[j] > 0
            logw[j] = log( dp.njs[j] ) + logpdf_unit(chols[j], state.state_theta[j].beta, X, W, z, i)
        else
            logw[j] = -Inf
            free == 0 && (free = j)
        end
    end

    ## Prob(label i = new component)
    theta_j = sample_new_theta(rng, input.priors.prior_theta, state, input, i, z)
    chol_j = Chol3(theta_j.Sigma)
    logw[K + 1] = log( dp.alpha ) + logpdf_unit(chol_j, theta_j.beta, X, W, z, i)

    ## sample component
    ji = sample_logweights(rng, logw)

    ## a new component goes into a free slot if there is one
    if ji == K + 1
        if free > 0
            ji = free
            state.state_theta[ji] = theta_j
            chols[ji] = chol_j
        else
            push!(dp.njs, 0)
            push!(state.state_theta, theta_j)
            push!(chols, chol_j)
        end
    end
    dp.labels[i] = ji

    ## update counts
    dp.njs[ji] += 1

    return state

end

## update parameters {LatentData, Theta}
function update_params!(rng::AbstractRNG, state::GibbsState, input::GibbsInput)
    @inbounds for j in 1:state.state_dp.J
        if state.state_dp.njs[j] == 0 continue end
        idx = state.state_dp.members[j]
        ## update theta
        state = update_theta!(rng, state, input, j, idx)
        ## update latent data
        state = update_latent!(rng, state, input, j, idx)
    end
    return state
end

## update theta_j = (beta_j, Sigma_j) given the latent data of its members idx:
## Sigma_j | beta_j, then beta_j | Sigma_j (sigma = false: beta_j | Sigma_j only)
function update_theta!(rng::AbstractRNG, state::GibbsState, input::GibbsInput, j::Integer, idx::AbstractVector{<:Integer};
                       sigma::Bool=true)
    p = input.priors.prior_theta
    kx, kz = input.dims.kx, input.dims.kz
    g, b1, b0 = coef_blocks(kx, kz)
    nj = length(idx)
    sd = state.state_data
    Wj = input.data.W[idx, :] # nj x kz
    Xj = input.data.X[idx, :] # nj x kx
    dj = sd.dstar[idx]
    y1j = sd.y1[idx]
    y0j = sd.y0[idx]

    ## 1. Sigma ~ NobileWishart(nj + rho, E'E + rho R), E = residuals at the current beta
    beta = state.state_theta[j].beta
    ed = dj .- Wj * beta[g]
    e1 = y1j .- Xj * beta[b1]
    e0 = y0j .- Xj * beta[b0]
    U = [dot(ed, ed) dot(ed, e1) dot(ed, e0);
         dot(e1, ed) dot(e1, e1) dot(e1, e0);
         dot(e0, ed) dot(e0, e1) dot(e0, e0)]
    U .+= p.prior_Sigma.rho .* p.prior_Sigma.R
    Sigma_j = sigma ? NobileWishart(rng, (nj + p.prior_Sigma.rho), U ) : state.state_theta[j].Sigma

    ## 2. beta ~ N(vj (H_j'(Sinv ⊗ I) z_j + V mu), vj), vj = (H_j'(Sinv ⊗ I)H_j + V)^-1
    Sinv = inv(Sigma_j)
    vj = inv(design_precision(Sinv, Wj' * Wj, Wj' * Xj, Xj' * Xj) + p.prior_beta.V)
    r = vcat(Wj' * (Sinv[1,1] .* dj .+ Sinv[1,2] .* y1j .+ Sinv[1,3] .* y0j),
             Xj' * (Sinv[2,1] .* dj .+ Sinv[2,2] .* y1j .+ Sinv[2,3] .* y0j),
             Xj' * (Sinv[3,1] .* dj .+ Sinv[3,2] .* y1j .+ Sinv[3,3] .* y0j))
    mj = vj * (r .+ state.state_sampler.Vmu)
    beta_j = mj + cholesky( Hermitian(vj) ).U' * randn(rng, length(mj))

    state.state_theta[j] = Theta(beta=beta_j, Sigma=Sigma_j)
    return state
end

## conditional moments of the latent data given Sigma (the formulas of the original
## update_latent!, verified in the review): the missing outcome given d* and the observed
## outcome, and d* given (y1, y0)
struct LatentMoments
    c1d::Float64; c1y::Float64; omega1::Float64 # y1 | d*, y0
    c0d::Float64; c0y::Float64; omega0::Float64 # y0 | d*, y1
    sd1::Float64; sd0::Float64; omega_d::Float64 # d* | y1, y0
end

function LatentMoments(S::AbstractMatrix{Float64})
    mid = S[2,3] * S[1,3] * S[1,2]
    denom1 = S[2,2] - S[1,2]^2
    denom0 = S[3,3] - S[1,3]^2
    denomd = S[2,2] * S[3,3] - S[2,3]^2
    return LatentMoments(
        ( (S[3,3] * S[1,2]) - (S[2,3] * S[1,3]) ) / denom0,
        ( S[2,3] - (S[1,3] * S[1,2]) ) / denom0,
        S[2,2] - ( ( (S[1,2]^2 * S[3,3]) - 2*mid + (S[2,3]^2) ) / denom0 ),
        ( (S[2,2] * S[1,3]) - (S[2,3] * S[1,2]) ) / denom1,
        ( S[2,3] - (S[1,3] * S[1,2]) ) / denom1,
        S[3,3] - ( ( (S[1,3]^2 * S[2,2]) - 2*mid + (S[2,3]^2) ) / denom1 ),
        ( (S[3,3] * S[1,2]) - (S[2,3] * S[1,3]) ) / denomd,
        ( (S[2,2] * S[1,3]) - (S[2,3] * S[1,2]) ) / denomd,
        1 - ( ( (S[1,2]^2 * S[3,3]) - 2*mid + (S[1,3]^2 * S[2,2]) ) / denomd ))
end

## draw unit i's missing outcome given d*_i and its observed outcome yi; returns (y1, y0)
@inline function draw_ymiss(rng::AbstractRNG, lm::LatentMoments, md, m1, m0, dstar, yi, di)
    if di == 1
        return yi, m0 + (dstar - md) * lm.c0d + (yi - m1) * lm.c0y + sqrt(lm.omega0) * randn(rng)
    else
        return m1 + (dstar - md) * lm.c1d + (yi - m0) * lm.c1y + sqrt(lm.omega1) * randn(rng), yi
    end
end

## update the latent data (y_miss, then d*) of component j's members idx. Units are
## conditionally independent given theta_j, so each unit's two draws are done in turn.
function update_latent!(rng::AbstractRNG, state::GibbsState, input::GibbsInput, j::Integer, idx::AbstractVector{<:Integer})
    th = state.state_theta[j]
    lm = LatentMoments(th.Sigma)
    sdd = sqrt(lm.omega_d)
    X, W = input.data.X, input.data.W
    y, d = input.data.y.a, input.data.d
    lower, upper = input.data.lower, input.data.upper
    sd = state.state_data
    @inbounds for i in idx
        md, m1, m0 = unit_mean(X, W, th.beta, i)
        ## 1. missing outcome
        y1, y0 = draw_ymiss(rng, lm, md, m1, m0, sd.dstar[i], y[i], d[i])
        sd.y1[i] = y1
        sd.y0[i] = y0
        ## 2. selection outcome
        mu_d = md + (y1 - m1) * lm.sd1 + (y0 - m0) * lm.sd0
        sd.dstar[i] = rand(rng, truncated(Normal(mu_d, sdd), lower[i], upper[i]) )
    end
    return state
end

## update concentration parameter
function update_alpha!(rng::AbstractRNG, state::GibbsState, input::GibbsInput)

    ## sample auxiliary variable
    state.state_dp.eta = rand(rng, Distributions.Beta(state.state_dp.alpha + 1, input.dims.n) )

    ## update shape and rate parameters
    b_star = input.priors.prior_dp.alpha_rate - log(state.state_dp.eta)
    a_star = input.priors.prior_dp.alpha_shape + state.state_dp.J
    A = (input.priors.prior_dp.alpha_shape + state.state_dp.J - 1) / (input.dims.n * b_star)
    ## Escobar & West (1995): shape a+k with probability pi_eta = A/(1+A), else a+k-1
    a_star = rand(rng) < A/(1 + A) ? a_star : a_star - 1

    ## sample alpha
    state.state_dp.alpha = rand(rng, Distributions.Gamma(a_star, 1/b_star) )

    ## update zdenom
    state.state_sampler.zdenom = state.state_dp.alpha + input.dims.n - 1

    return state

end

## size the output for a run of M iterations with the thinning in params
function prepare_out!(out::GibbsOut, params::InputParams)
    out.thin, out.latent_thin = params.thin, params.latent_thin
    nl = params.latent_thin == 0 ? 0 : params.M ÷ params.latent_thin
    resize!(out.out_dp, params.M ÷ params.thin)
    resize!(out.out_theta, params.M ÷ params.thin)
    resize!(out.out_hyper, params.M ÷ params.thin)
    resize!(out.out_sep, params.M ÷ params.thin)
    resize!(out.out_data, nl)
    resize!(out.out_labels, nl)
    return out
end

## the separated-units share: the share of units in clusters whose treated share is under 5% or
## over 95% (Phase 4d diagnostic: in such clusters the treated or the control outcome is
## identified from almost no data)
function separated_share(dp::StateDP, d::AbstractVector{<:Integer}; lo::Real=0.05, hi::Real=0.95)
    n = length(dp.labels)
    nsep = 0
    for idx in dp.members
        isempty(idx) && continue
        t = count(i -> d[i] == 1, idx) / length(idx)
        (t < lo || t > hi) && (nsep += length(idx))
    end
    return nsep / n
end

## save iteration m: theta, counts and alpha every thin-th iteration (the Theta objects are
## shared, not copied), latent data and labels every latent_thin-th iteration
function update_out!(state::GibbsState, out::GibbsOut, m::Int64, input::GibbsInput)
    if m % out.thin == 0
        s = m ÷ out.thin
        out.out_dp[s] = DrawDP(state.state_dp)
        out.out_theta[s] = copy(state.state_theta)
        out.out_hyper[s] = vcat(state.state_hyper.m, state.state_hyper.tau2)
        out.out_sep[s] = separated_share(state.state_dp, input.data.d)
    end
    if out.latent_thin > 0 && m % out.latent_thin == 0
        s = m ÷ out.latent_thin
        out.out_data[s] = copy(state.state_data)
        out.out_labels[s] = copy(state.state_dp.labels)
    end
    return out
end

## update sampler state after a run
function update_sampler!(state::GibbsState, M::Int64)
    state.state_sampler.chain = true
    state.state_sampler.batch_m += M
    return state
end

## --------------------------------------------------------------------------- #
## R3: the collapsed sigma23 step (Phase 4c). Each unit observes one outcome, so with y_miss
## integrated out the likelihood of (d*, y_obs) involves sigma12, sigma13, sigma22, sigma33 only,
## and the full conditional of sigma23 is the PRIOR conditional of the restricted inverse
## Wishart, NobileWishart(rho, rho R):
##   p(sigma23 | rest) ∝ |Sigma|^{-(rho+4)/2} exp(-tr(rho R Sigma^-1) / 2)
## on the positive-definite interval |sigma23 - s12 s13| < sqrt((s22 - s12^2)(s33 - s13^2)).
## sigma23 is drawn by slice sampling on that bounded interval (shrinkage from the full interval,
## exact), then y_miss of every member is redrawn given (d*, y_obs) and the new Sigma: a
## partially collapsed step, so nothing after it may condition on the old y_miss.

## log density of sigma23 = s under |Sigma|^{-(nu+4)/2} exp(-tr(V Sigma^-1)/2), other entries of
## Sigma fixed (sigma11 = 1), up to a constant; -Inf outside the positive-definite interval
function sigma23_logdens(S::AbstractMatrix{Float64}, s::Real, V::AbstractMatrix{Float64}, nu::Real)
    a, b, c, e = S[1, 2], S[1, 3], S[2, 2], S[3, 3]
    det = (c - a^2) * (e - b^2) - (s - a * b)^2
    det > 0 || return -Inf
    ## adjugate of [1 a b; a c s; b s e]
    i11, i22, i33 = c * e - s^2, e - b^2, c - a^2
    i12, i13, i23 = b * s - a * e, a * s - b * c, a * b - s
    tr = (V[1, 1] * i11 + V[2, 2] * i22 + V[3, 3] * i33 +
          2 * (V[1, 2] * i12 + V[1, 3] * i13 + V[2, 3] * i23)) / det
    return -(nu + 4) / 2 * log(det) - tr / 2
end

## the positive-definite interval of sigma23 given the other entries
function sigma23_interval(S::AbstractMatrix{Float64})
    a, b = S[1, 2], S[1, 3]
    w = sqrt((S[2, 2] - a^2) * (S[3, 3] - b^2))
    return a * b - w, a * b + w
end

## one slice-sampling update of sigma23 from its prior conditional; shrinkage on the bounded
## interval (Neal 2003), so no stepping out
function slice_sigma23(rng::AbstractRNG, S::AbstractMatrix{Float64}, V::AbstractMatrix{Float64}, nu::Real)
    L, U = sigma23_interval(S)
    x0 = S[2, 3]
    logy = sigma23_logdens(S, x0, V, nu) - randexp(rng)
    for _ in 1:10_000
        x1 = L + (U - L) * rand(rng)
        sigma23_logdens(S, x1, V, nu) > logy && return x1
        x1 < x0 ? (L = x1) : (U = x1)
    end
    error("slice_sigma23: no acceptance after 10000 shrinkage steps")
end

## R3 for component j with members idx: sigma23 from its prior conditional, then y_miss | d*,
## y_obs, theta_j for every member
function update_sigma23!(rng::AbstractRNG, state::GibbsState, input::GibbsInput, j::Integer, idx::AbstractVector{<:Integer})
    ps = input.priors.prior_theta.prior_Sigma
    th = state.state_theta[j]
    S = copy(th.Sigma)
    S[2, 3] = S[3, 2] = slice_sigma23(rng, S, ps.rho .* ps.R, ps.rho)
    state.state_theta[j] = Theta(beta=th.beta, Sigma=S)
    return update_ymiss_only!(rng, state, input, j, idx)
end

## --------------------------------------------------------------------------- #
## specialized functions for Gaussian model

function update_gaussian_params!(rng::AbstractRNG, state::GibbsState, input::GibbsInput)
    idx = 1:input.dims.n
    ## update theta: with the ridge block, beta | Sigma and then the joint (beta1, beta0, sigma12,
    ## sigma13) | Omega, gamma and Omega | rest (Phase 4f); otherwise Sigma | beta, beta | Sigma
    if input.params.ridge_block
        state = update_theta!(rng, state, input, 1, idx; sigma=false)
        update_gaussian_ridge!(rng, state, input)
    else
        state = update_theta!(rng, state, input, 1, idx)
    end
    ## R3: collapsed sigma23 step (and y_miss redraw)
    input.params.sigma23_step && update_sigma23!(rng, state, input, 1, idx)
    ## update latent data
    state = update_latent!(rng, state, input, 1, idx)
    return state
end

## --------------------------------------------------------------------------- #
## specialized functions for FMN

## update labels for fmn
function update_fmn_labels!(rng::AbstractRNG, state::GibbsState, input::GibbsInput)
    chols = component_chols(state.state_theta)
    ## update component labels for each i
    @inbounds for i in 1:input.dims.n
        ji = state.state_dp.labels[i] # i's current component
        if ji == 0 error("Null label! Oops") end
        state.state_dp.njs[ji] = state.state_dp.njs[ji] - 1 # decrease ji count
        if state.state_dp.njs[ji] < 0 error("Negative label count! Oops") end
        ## sample component label
        state = sample_fmn_label!(rng, state, input, i, chols)
    end
    rebuild_members!(state.state_dp)
    return state
end

## sample fmn label for i
function sample_fmn_label!(rng::AbstractRNG, state::GibbsState, input::GibbsInput, i::Int64, chols::Vector{Chol3})
    X, W = input.data.X, input.data.W
    z = unit_z(state.state_data, i)
    logw = Array{Float64}(undef, state.state_dp.J)
    @inbounds for j in 1:state.state_dp.J
        logw[j] = log( state.state_dp.njs[j] + state.state_dp.alpha ) +
                  logpdf_unit(chols[j], state.state_theta[j].beta, X, W, z, i)
    end
    ## sample component
    ji = sample_logweights(rng, logw)
    state.state_dp.labels[i] = ji
    ## update counts
    state.state_dp.njs[ji] +=  1
    return state
end


## --------------------------------------------------------------------------- #
## specialized functions for blocked sampler

## v_j ~ Beta(1 + n_j, alpha + sum_{l>j} n_l) for j < J; v_J = 1
function compute_v(rng::AbstractRNG, s::StateDP, j::Int64)
    if j < length(s.njs)
        njj = sum( @view s.njs[(j+1):end] )
        v = rand(rng, Distributions.Beta( 1 + s.njs[j], s.alpha + njj))
        ## a Beta draw can round to exactly 1.0 when alpha + njj is tiny; keep 1 - v > 0 so that
        ## log(1 - v) in update_blocked_alpha! stays finite (-36.7 at the cap)
        v = min(v, prevfloat(1.0))
    else
        v = 1.0
    end
    return v
end

## w_j = v_j * prod_{l<j} (1 - v_l), using the already-updated v_1..v_{j-1}
function compute_w(s::StateDP, j::Int64, v::Float64)
    return (j == 1) ? v : v * prod( 1 - s.ws[l].v for l in 1:(j-1) )
end

## store a new BlockedWeights object: saved copies of earlier states must not see this update
function update_ws!(rng::AbstractRNG, s::StateDP, j::Int64)
    v = compute_v(rng, s, j)
    s.ws[j] = BlockedWeights(compute_w(s, j, v), v)
    return s
end

## update stick-breaking weights
function compute_weights!(rng::AbstractRNG, state_dp::StateDP)
    length(state_dp.ws) == length(state_dp.njs) || resize!(state_dp.ws, length(state_dp.njs))
    for j in 1:length(state_dp.njs)
        state_dp = update_ws!(rng, state_dp, j)
    end
    return state_dp
end

function update_blocked_labels!(rng::AbstractRNG, state::GibbsState, input::GibbsInput)
    chols = component_chols(state.state_theta)
    for i in 1:input.dims.n
        state = sample_blocked_label!(rng, state, input, i, chols)
    end

    state.state_dp = compute_weights!(rng, state.state_dp)
    state.state_dp = update_J!(state.state_dp)
    rebuild_members!(state.state_dp)
    return state
end

function sample_blocked_label!(rng::AbstractRNG, state::GibbsState, input::GibbsInput, i::Int64, chols::Vector{Chol3})
    X, W = input.data.X, input.data.W
    z = unit_z(state.state_data, i)
    logw = Array{Float64}(undef, length(state.state_dp.njs))
    # remove i from current component
    state.state_dp.njs[state.state_dp.labels[i]] -= 1
    ## 1. compute log Pr(label i = j)
    @inbounds for j in 1:length(state.state_dp.njs)
        logw[j] = log( state.state_dp.ws[j].w ) + logpdf_unit(chols[j], state.state_theta[j].beta, X, W, z, i)
    end
    ## 2. sample ji
    ji = sample_logweights(rng, logw)
    state.state_dp.labels[i] = ji
    ## update counts
    state.state_dp.njs[ji] += 1
    return state
end

## montitor number of active components
function update_J!(s::StateDP)
    s.J = count( >(0), s.njs )
    return s
end

## sample theta from prior
function sample_prior_theta!(rng::AbstractRNG, state_theta::StateTheta, prior_theta::PriorTheta, j::Int64)
    ## beta ~ N(mu, V0): use the covariance factor, not the precision stored in prior_beta.V
    beta_j = prior_theta.prior_beta.mu + prior_theta.prior_beta.Lcov*randn(rng, length(prior_theta.prior_beta.mu))
    Sigma_j = NobileWishart(rng, prior_theta.prior_Sigma.rho, prior_theta.prior_Sigma.rho * prior_theta.prior_Sigma.R )
    setcomp!(state_theta, j, Theta(beta=beta_j, Sigma=Sigma_j))
    return state_theta
end

## update parameters {LatentData, Theta}
function update_blocked_params!(rng::AbstractRNG, state::GibbsState, input::GibbsInput)
    @inbounds for j in 1:input.priors.prior_dp.J
        if state.state_dp.njs[j] == 0
            ## sample from prior
            state.state_theta = sample_prior_theta!(rng, state.state_theta, input.priors.prior_theta, j)
        else
            ## sample from posterior
            idx = state.state_dp.members[j]
            state = update_theta!(rng, state, input, j, idx)
            state = update_latent!(rng, state, input, j, idx)
        end
    end
    return state
end

## update concentration parameter
function fv(d::AbstractVector{BlockedWeights}; f::Function=log)
    return sum( f(1 - d[j].v) for j in 1:(length(d)-1); init=0.0 )
end

function update_blocked_alpha!(rng::AbstractRNG, state::GibbsState, input::GibbsInput)
    a_star = input.priors.prior_dp.alpha_shape + input.priors.prior_dp.J - 1
    b_star = input.priors.prior_dp.alpha_rate - (fv(state.state_dp.ws))
    state.state_dp.alpha = rand(rng, Distributions.Gamma(a_star, 1/b_star) )
    state.state_sampler.zdenom = state.state_dp.alpha + input.dims.n - 1
    return state
end


## --------------------------------------------------------------------------- #
## test functions: dpm_gibbs

## update latent data, given true theta
function update_latent_only!(rng::AbstractRNG, state::GibbsState, input::GibbsInput)
    @inbounds for j in 1:state.state_dp.J
        if state.state_dp.njs[j] == 0 continue end
        state = update_ymiss_only!(rng, state, input, j, state.state_dp.members[j])
    end
    return state
end

## update theta, given true latent data
function update_theta_only!(rng::AbstractRNG, state::GibbsState, input::GibbsInput)
    @inbounds for j in 1:state.state_dp.J
        if state.state_dp.njs[j] == 0 continue end
        state = update_theta!(rng, state, input, j, state.state_dp.members[j])
    end
    return state
end

## update the missing outcomes only, d* fixed
function update_ymiss_only!(rng::AbstractRNG, state::GibbsState, input::GibbsInput, j::Integer, idx::AbstractVector{<:Integer})
    th = state.state_theta[j]
    lm = LatentMoments(th.Sigma)
    X, W = input.data.X, input.data.W
    y, d = input.data.y.a, input.data.d
    sd = state.state_data
    @inbounds for i in idx
        md, m1, m0 = unit_mean(X, W, th.beta, i)
        sd.y1[i], sd.y0[i] = draw_ymiss(rng, lm, md, m1, m0, sd.dstar[i], y[i], d[i])
    end
    return state
end

## --------------------------------------------------------------------------- #
## test functions: dpm_blocked

## update {Theta} only
function update_blocked_theta_only!(rng::AbstractRNG, state::GibbsState, input::GibbsInput)
    @inbounds for j in 1:input.priors.prior_dp.J
        if state.state_dp.njs[j] == 0
            ## sample from prior
            state.state_theta = sample_prior_theta!(rng, state.state_theta, input.priors.prior_theta, j)
        else
            state = update_theta!(rng, state, input, j, state.state_dp.members[j])
        end
    end
    return state
end
