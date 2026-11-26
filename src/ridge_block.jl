## --------------------------------------------------------------------------- #
## Phase 4f, option A: the ridge block. A joint complete-data draw of each cluster's outcome
## coefficients and selection covariances, which breaks the intercept-sigma_1k ridge (Phase 4e).
##
## Reparameterization (sigma11 = 1): Sigma = [1 delta'; delta Omega + delta delta'], with
## delta = (sigma12, sigma13)' and Omega = Var(y1, y0 | d*), one-to-one with Jacobian 1.
## Prior: Sigma ~ NobileWishart(rho, Psi), Psi = rho R, i.e. density |Sigma|^{-(rho+4)/2}
## exp(-tr(Psi Sigma^-1)/2) on sigma11 = 1. With |Sigma| = |Omega| and
##   tr(Psi Sigma^-1) = const + Psi11 (delta - c)' Omega^-1 (delta - c) + tr(Psi_22.1 Omega^-1),
## c = Psi_21 / Psi11, Psi_22.1 = Psi_22 - Psi_21 Psi_12 / Psi11, it factors as
##   delta | Omega ~ N(c, Omega / Psi11)          (C3: N(0, Omega))
##   Omega ~ IW_2(rho, Psi_22.1)                  (|Omega|^{-(rho+3)/2}: nu_Omega = rho)
## Complete data: d*_i ~ N(mu_d, 1), which involves neither (B, delta) nor Omega, and
##   (y1_i, y0_i) | d*_i ~ N2(outcome means + delta v_i, Omega),  v_i = d*_i - mu_d,i.
## Full conditionals (B = outcome coefficients of the cluster, prior N(mu_B, V_B) given the
## selection coefficients):
##   (B, delta) | Omega, rest ~ Gaussian (a two-equation regression with regressors (x_i, v_i)),
##   Omega | B, delta, rest ~ IW_2(rho + n_j + 1, Psi_22.1 + E'E + Psi11 (delta - c)(delta - c)'),
## the +1 and the Psi11 term coming from the delta | Omega prior.

## prior constants of the (delta, Omega) parameterization
struct RidgePrior
    Psi11::Float64
    c::Vector{Float64}
    Psi221::Matrix{Float64}
    nu::Float64
end

function RidgePrior(ps::PriorSigma)
    Psi = ps.rho .* ps.R
    Psi11 = Psi[1, 1]
    c = Psi[2:3, 1] ./ Psi11
    Psi221 = Matrix(Symmetric(Psi[2:3, 2:3] - Psi[2:3, 1] * Psi[1, 2:3]' ./ Psi11))
    return RidgePrior(Psi11, c, Psi221, Float64(ps.rho))
end

## (delta, Omega) of Sigma and back
function sigma_delta_omega(S::AbstractMatrix{Float64})
    d = [S[1, 2], S[1, 3]]
    O = [S[2, 2] - d[1]^2  S[2, 3] - d[1] * d[2];
         S[2, 3] - d[1] * d[2]  S[3, 3] - d[2]^2]
    return d, O
end

sigma_from_delta_omega(d::AbstractVector{Float64}, O::AbstractMatrix{Float64}) =
    [1.0 d[1] d[2];
     d[1] O[1, 1] + d[1]^2 O[1, 2] + d[1] * d[2];
     d[2] O[1, 2] + d[1] * d[2] O[2, 2] + d[2]^2]

## the Gaussian full conditional of theta = (B1, B0, delta1, delta2) given Omega: precision P and
## P * mean. ztf(i) is unit i's latent vector minus its global part; the cluster design is
## blockdiag(Wc_i', Xc_i', Xc_i') with coefficients bc = (selection, B1, B0); mu_c, Vc its prior.
function ridge_theta_post(idx::AbstractVector{<:Integer}, ztf, Wc::AbstractMatrix{Float64},
                          Xc::AbstractMatrix{Float64}, bc::AbstractVector{Float64}, mu_c::AbstractVector{Float64},
                          Vc::AbstractMatrix{Float64}, O::AbstractMatrix{Float64}, rp::RidgePrior)
    kz, kx = size(Wc, 2), size(Xc, 2)
    g, b1, b0 = coef_blocks(kx, kz)
    ob = vcat(collect(b1), collect(b0))
    p = 2kx
    q = p + 2
    Oi = inv(Symmetric(Matrix(O)))
    w11, w12, w22 = Oi[1, 1], Oi[1, 2], Oi[2, 2]
    P = zeros(q, q)
    r = zeros(q)
    g1, g2 = zeros(q), zeros(q)
    @inbounds for i in idx
        z = ztf(i)
        v = z[1]
        for a in 1:kz
            v -= Wc[i, a] * bc[g[a]]
        end
        fill!(g1, 0.0); fill!(g2, 0.0)
        for a in 1:kx
            g1[a] = Xc[i, a]
            g2[kx + a] = Xc[i, a]
        end
        g1[p + 1] = v
        g2[p + 2] = v
        t1 = w11 * z[2] + w12 * z[3]
        t2 = w12 * z[2] + w22 * z[3]
        for b in 1:q, a in 1:q
            P[a, b] += w11 * g1[a] * g1[b] + w12 * (g1[a] * g2[b] + g2[a] * g1[b]) + w22 * g2[a] * g2[b]
        end
        for a in 1:q
            r[a] += g1[a] * t1 + g2[a] * t2
        end
    end
    ## prior of B given the selection coefficients (Gaussian conditioning of N(mu_c, Vc))
    VBB, VBg, Vgg = Vc[ob, ob], Vc[ob, g], Vc[g, g]
    if maximum(abs, VBg; init=0.0) == 0
        mB, VB = mu_c[ob], VBB
    else
        K = VBg / Symmetric(Vgg)
        mB, VB = mu_c[ob] + K * (bc[g] - mu_c[g]), VBB - K * VBg'
    end
    VBi = inv(Symmetric(VB))
    P[1:p, 1:p] .+= VBi
    r[1:p] .+= VBi * mB
    ## delta | Omega ~ N(c, Omega / Psi11)
    P[p+1:q, p+1:q] .+= rp.Psi11 .* Oi
    r[p+1:q] .+= rp.Psi11 .* (Oi * rp.c)
    return Matrix(Symmetric(P)), r, ob
end

## the IW_2 full conditional of Omega given theta: degrees of freedom and scale
function ridge_omega_post(idx::AbstractVector{<:Integer}, ztf, Wc::AbstractMatrix{Float64},
                          Xc::AbstractMatrix{Float64}, bc::AbstractVector{Float64},
                          d::AbstractVector{Float64}, rp::RidgePrior)
    kz, kx = size(Wc, 2), size(Xc, 2)
    g, b1, b0 = coef_blocks(kx, kz)
    S = copy(rp.Psi221)
    @inbounds for i in idx
        z = ztf(i)
        v = z[1]
        for a in 1:kz
            v -= Wc[i, a] * bc[g[a]]
        end
        e1, e0 = z[2] - v * d[1], z[3] - v * d[2]
        for a in 1:kx
            e1 -= Xc[i, a] * bc[b1[a]]
            e0 -= Xc[i, a] * bc[b0[a]]
        end
        S[1, 1] += e1^2; S[1, 2] += e1 * e0; S[2, 2] += e0^2
    end
    S[2, 1] = S[1, 2]
    dc = d .- rp.c
    S .+= rp.Psi11 .* (dc * dc')
    return rp.nu + length(idx) + 1, Matrix(Symmetric(S))
end

## one ridge-block update of a cluster: (B, delta) | Omega jointly, then Omega | (B, delta).
## Returns the new cluster coefficients bc and Sigma.
function ridge_draw(rng::AbstractRNG, idx::AbstractVector{<:Integer}, ztf, Wc::AbstractMatrix{Float64},
                    Xc::AbstractMatrix{Float64}, bc::AbstractVector{Float64}, mu_c::AbstractVector{Float64},
                    Vc::AbstractMatrix{Float64}, Sigma::AbstractMatrix{Float64}, rp::RidgePrior)
    _, O = sigma_delta_omega(Sigma)
    P, r, ob = ridge_theta_post(idx, ztf, Wc, Xc, bc, mu_c, Vc, O, rp)
    C = cholesky(Symmetric(P))
    theta = C \ r + C.U \ randn(rng, length(r))
    p = length(ob)
    bc_new = copy(bc)
    bc_new[ob] = theta[1:p]
    d = theta[p+1:p+2]
    df, S = ridge_omega_post(idx, ztf, Wc, Xc, bc_new, d, rp)
    O_new = rand(rng, InverseWishart(df, S))
    return bc_new, sigma_from_delta_omega(d, Matrix(Symmetric(O_new)))
end

## the ridge block for the gaussian sampler (one component, every unit)
function update_gaussian_ridge!(rng::AbstractRNG, state::GibbsState, input::GibbsInput)
    pb = input.priors.prior_theta.prior_beta
    th = state.state_theta[1]
    sd = state.state_data
    rp = RidgePrior(PriorSigma(input.priors.prior_theta.prior_Sigma.rho, sigma_R(state, input)))
    beta, Sigma = ridge_draw(rng, 1:input.dims.n, i -> unit_z(sd, i), input.data.W, input.data.X,
                             th.beta, pb.mu, pb.Vcov, th.Sigma, rp)
    state.state_theta[1] = Theta(beta=beta, Sigma=Sigma)
    return state
end

## the ridge block for Design A: every cluster, on z~ = z - H^g b, the prior of beta^c from D
function update_marginal_ridge!(rng::AbstractRNG, state::GibbsState, input::GibbsInput, D) # D::DesignA
    b = global_slopes(state, D)
    rp = RidgePrior(PriorSigma(input.priors.prior_theta.prior_Sigma.rho, sigma_R(state, input)))
    for j in 1:state.state_dp.J
        idx = state.state_dp.members[j]
        th = state.state_theta[j]
        bc, Sigma = ridge_draw(rng, idx, i -> ztilde(state, D, b, i), D.Wc, D.Xc, th.beta[D.cidx],
                               D.mu_c, D.Vc, th.Sigma, rp)
        state.state_theta[j] = Theta(beta=full_beta(D, bc, b), Sigma=Sigma)
    end
    return state
end
