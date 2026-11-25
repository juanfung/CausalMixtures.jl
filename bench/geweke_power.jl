# bench/geweke_power.jl -- power check of the Geweke harness: plant one bug by redefining a
# method, then run the Geweke test (50k sweeps vs 100k forward draws) for Design A in both slope
# modes and for gaussian.
#   julia --project=. bench/geweke_power.jl <bug> [M_sc] [M_mc]   (default 50000 100000)
# bugs: none, sigma_dof, no_beta_unc, no_phi, no_shift, alpha_fixed_indicator (Phases 4a-4b);
# r3_posterior, r3_no_redraw, r3_no_redraw_dfirst, r3_dfirst, r3_scale_R, r3_dof (Phase 4c, R3;
# R3 is off by default since 4d, so these need sigma23_step = true to have an effect);
# hier_shape_k, hier_pred_fixed (Phase 4d, hierarchical intercepts);
# ridge_omega_noprior, ridge_separate, ridge_wrong_nu (Phase 4f, ridge block);
# latent_c0y_flip, latent_trunc_side (the latent refresh; testable since the harness runs it first).
# The r3_* bugs run with R3 on. CM_RIDGE=0 runs with the ridge block off.
include(joinpath(@__DIR__, "..", "test", "GewekeHarness.jl")); using .GewekeHarness
using CausalMixtures, Random, LinearAlgebra
const CM = CausalMixtures
bug = ARGS[1]
GewekeHarness.R3[] = startswith(bug, "r3_")
GewekeHarness.RIDGE[] = get(ENV, "CM_RIDGE", "1") == "1"
M_sc = length(ARGS) > 1 ? parse(Int, ARGS[2]) : 50_000
M_mc = length(ARGS) > 2 ? parse(Int, ARGS[3]) : 100_000
if bug == "sigma_dof"      # Sigma_j ~ NobileWishart(rho + n_j + 1, ...)
    @eval CM function update_marginal_sigmas!(rng::AbstractRNG, state::GibbsState, input::GibbsInput)
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
            state.state_theta[j] = Theta(beta=th.beta, Sigma=NobileWishart(rng, ps.rho + state.state_dp.njs[j] + 1, U))
        end
        return state
    end
elseif bug == "no_beta_unc"  # label-step predictive without the beta^c uncertainty term H Q^-1 H'
    @eval CM function predictive!(S::Matrix{Float64}, pc::PostCache, Sigma::AbstractMatrix{Float64}, D::DesignA, i::Integer)
        S .= Sigma
        return unit_mean(D.Xc, D.Wc, pc.m, i)
    end
elseif bug == "no_phi"  # R2 score without the Phi term (Phase 4b planted bug)
    @eval CM function r2_score(mu::NTuple{3,Float64}, S::AbstractMatrix{Float64}, yo::Real, d::Integer, gd::Real=0.0)
        o = d == 1 ? 2 : 3
        Soo = S[o, o]
        return -0.5 * (log(2pi * Soo) + (yo - mu[o])^2 / Soo)
    end
elseif bug == "no_shift"  # R2 with the plan's original cut-off at 0: the global shift g_d dropped
    @eval CM function r2_score(mu::NTuple{3,Float64}, S::AbstractMatrix{Float64}, yo::Real, d::Integer, gd::Real=0.0)
        o = d == 1 ? 2 : 3; s = 2d - 1; Soo = S[o, o]
        cm = mu[1] + S[1, o] / Soo * (yo - mu[o])
        cv = S[1, 1] - S[1, o]^2 / Soo
        return -0.5 * (log(2pi * Soo) + (yo - mu[o])^2 / Soo) + logcdf(Normal(), s * cm / sqrt(cv))
    end
    @eval CM function r2_draw(rng::AbstractRNG, mu::NTuple{3,Float64}, S::AbstractMatrix{Float64}, yo::Real, d::Integer,
                              gd::Real=0.0)
        o, mi = d == 1 ? (2, 3) : (3, 2)
        Soo = S[o, o]
        cm = mu[1] + S[1, o] / Soo * (yo - mu[o])
        cv = S[1, 1] - S[1, o]^2 / Soo
        ds = rand(rng, truncated(Normal(cm, sqrt(cv)), d == 1 ? 0.0 : -Inf, d == 1 ? Inf : 0.0))
        a11, a12, a22 = S[1, 1], S[1, o], Soo
        det = a11 * a22 - a12^2
        c1, c2 = S[mi, 1], S[mi, o]
        k1 = (c1 * a22 - c2 * a12) / det
        k2 = (c2 * a11 - c1 * a12) / det
        ym = mu[mi] + k1 * (ds - mu[1]) + k2 * (yo - mu[o]) + sqrt(max(S[mi, mi] - k1 * c1 - k2 * c2, 0.0)) * randn(rng)
        return d == 1 ? (ds, Float64(yo), ym) : (ds, ym, Float64(yo))
    end
elseif bug == "r3_posterior"  # R3 with the posterior (rho + n_j, rho R + sum e e') instead of the prior
    @eval CM function update_sigma23!(rng::AbstractRNG, state::GibbsState, input::GibbsInput, j::Integer, idx::AbstractVector{<:Integer})
        ps = input.priors.prior_theta.prior_Sigma
        th = state.state_theta[j]
        X, W = input.data.X, input.data.W
        sd = state.state_data
        U = ps.rho .* ps.R
        for i in idx
            md, m1, m0 = unit_mean(X, W, th.beta, i)
            e = (sd.dstar[i] - md, sd.y1[i] - m1, sd.y0[i] - m0)
            for a in 1:3, c in 1:3
                U[a, c] += e[a] * e[c]
            end
        end
        S = copy(th.Sigma)
        S[2, 3] = S[3, 2] = slice_sigma23(rng, S, U, ps.rho + length(idx))
        state.state_theta[j] = Theta(beta=th.beta, Sigma=S)
        return update_ymiss_only!(rng, state, input, j, idx)
    end
elseif bug in ("r3_scale_R", "r3_dof")  # R3 with a wrong prior target: scale R (rho dropped), or rho + 6 dof
    @eval CM function update_sigma23!(rng::AbstractRNG, state::GibbsState, input::GibbsInput, j::Integer, idx::AbstractVector{<:Integer})
        ps = input.priors.prior_theta.prior_Sigma
        th = state.state_theta[j]
        S = copy(th.Sigma)
        V, nu = $(bug == "r3_scale_R") ? (ps.R, ps.rho) : (ps.rho .* ps.R, ps.rho + 6)
        S[2, 3] = S[3, 2] = slice_sigma23(rng, S, V, nu)
        state.state_theta[j] = Theta(beta=th.beta, Sigma=S)
        return update_ymiss_only!(rng, state, input, j, idx)
    end
elseif bug in ("r3_no_redraw", "r3_no_redraw_dfirst")  # R3 without the y_miss redraw
    @eval CM function update_sigma23!(rng::AbstractRNG, state::GibbsState, input::GibbsInput, j::Integer, idx::AbstractVector{<:Integer})
        ps = input.priors.prior_theta.prior_Sigma
        th = state.state_theta[j]
        S = copy(th.Sigma)
        S[2, 3] = S[3, 2] = slice_sigma23(rng, S, ps.rho .* ps.R, ps.rho)
        state.state_theta[j] = Theta(beta=th.beta, Sigma=S)
        return state
    end
end
if bug in ("r3_no_redraw_dfirst", "r3_dfirst")  # latent refresh draws d* before y_miss
    @eval CM function update_latent!(rng::AbstractRNG, state::GibbsState, input::GibbsInput, j::Integer, idx::AbstractVector{<:Integer})
        th = state.state_theta[j]
        lm = LatentMoments(th.Sigma)
        sdd = sqrt(lm.omega_d)
        X, W = input.data.X, input.data.W
        y, d = input.data.y.a, input.data.d
        lower, upper = input.data.lower, input.data.upper
        sd = state.state_data
        @inbounds for i in idx
            md, m1, m0 = unit_mean(X, W, th.beta, i)
            mu_d = md + (sd.y1[i] - m1) * lm.sd1 + (sd.y0[i] - m0) * lm.sd0
            sd.dstar[i] = rand(rng, truncated(Normal(mu_d, sdd), lower[i], upper[i]))
            sd.y1[i], sd.y0[i] = draw_ymiss(rng, lm, md, m1, m0, sd.dstar[i], y[i], d[i])
        end
        return state
    end
end
if bug == "hier_shape_k"  # tau2 | a, m with shape 3 + k instead of 3 + k/2
    @eval CM function update_hyper!(rng::AbstractRNG, state::GibbsState, D::DesignA)
        h = state.state_hyper
        pos = D.cidx[D.icidx]
        occ = [j for j in eachindex(state.state_dp.njs) if state.state_dp.njs[j] > 0]
        k = length(occ)
        for l in 1:3
            sa = sum(state.state_theta[j].beta[pos[l]] for j in occ)
            P = 1 / D.Vm[l] + k / h.tau2[l]
            h.m[l] = (D.mu_m[l] / D.Vm[l] + sa / h.tau2[l]) / P + randn(rng) / sqrt(P)
        end
        for l in 1:3
            ss = sum((state.state_theta[j].beta[pos[l]] - h.m[l])^2 for j in occ)
            h.tau2[l] = rand(rng, InverseGamma(D.tau_shape + k, D.tau_scale[l] + ss / 2))
        end
        return set_cprior!(D, h.m, h.tau2)
    end
elseif bug == "hier_pred_fixed"  # label step (collapsed and auxiliary predictives) under the fixed (mu, V), not (m, T)
    @eval CM function marginal_sweep!(rng::AbstractRNG, state::GibbsState, input::GibbsInput, D::DesignA; test::Bool=false)
        set_cprior!(D, D.mu_m, D.Vm)
        update_marginal_labels!(rng, state, input, D; r2 = input.params.label_step == :r2 && !test)
        sync_hyper!(state, D)
        update_marginal_coefs!(rng, state, D)
        update_global_slopes!(rng, state, D)
        update_marginal_sigmas!(rng, state, input)
        if !test
            for j in 1:state.state_dp.J
                update_latent!(rng, state, input, j, state.state_dp.members[j])
            end
        end
        input.priors.prior_dp.alpha_shape != 0.0 && update_alpha!(rng, state, input)
        D.hier && update_hyper!(rng, state, D)
        return state
    end
end
if bug == "ridge_omega_noprior"  # Omega update without the +1 and the Psi11 (delta - c)(delta - c)' term
    @eval CM function ridge_omega_post(idx::AbstractVector{<:Integer}, ztf, Wc::AbstractMatrix{Float64},
                                       Xc::AbstractMatrix{Float64}, bc::AbstractVector{Float64},
                                       d::AbstractVector{Float64}, rp::RidgePrior)
        kz, kx = size(Wc, 2), size(Xc, 2)
        g, b1, b0 = coef_blocks(kx, kz)
        S = copy(rp.Psi221)
        for i in idx
            z = ztf(i)
            v = z[1] - sum(Wc[i, a] * bc[g[a]] for a in 1:kz)
            e1 = z[2] - v * d[1] - sum(Xc[i, a] * bc[b1[a]] for a in 1:kx)
            e0 = z[3] - v * d[2] - sum(Xc[i, a] * bc[b0[a]] for a in 1:kx)
            S[1, 1] += e1^2; S[1, 2] += e1 * e0; S[2, 2] += e0^2
        end
        S[2, 1] = S[1, 2]
        return rp.nu + length(idx), Matrix(Symmetric(S))
    end
elseif bug == "ridge_separate"  # B | delta, Omega, then delta | B, Omega (two exact conditional draws: valid)
    @eval CM function ridge_draw(rng::AbstractRNG, idx::AbstractVector{<:Integer}, ztf, Wc::AbstractMatrix{Float64},
                                 Xc::AbstractMatrix{Float64}, bc::AbstractVector{Float64}, mu_c::AbstractVector{Float64},
                                 Vc::AbstractMatrix{Float64}, Sigma::AbstractMatrix{Float64}, rp::RidgePrior)
        d0, O = sigma_delta_omega(Sigma)
        P, r, ob = ridge_theta_post(idx, ztf, Wc, Xc, bc, mu_c, Vc, O, rp)
        p = length(ob)
        Bi, Di = 1:p, p+1:p+2
        ## B | delta = d0: precision P_BB, mean P_BB^-1 (r_B - P_BD d0)
        CB = cholesky(Symmetric(P[Bi, Bi]))
        B = CB \ (r[Bi] - P[Bi, Di] * d0) + CB.U \ randn(rng, p)
        ## delta | B
        CD = cholesky(Symmetric(P[Di, Di]))
        d = CD \ (r[Di] - P[Di, Bi] * B) + CD.U \ randn(rng, 2)
        bc_new = copy(bc); bc_new[ob] = B
        df, S = ridge_omega_post(idx, ztf, Wc, Xc, bc_new, d, rp)
        O_new = rand(rng, InverseWishart(df, S))
        return bc_new, sigma_from_delta_omega(d, Matrix(Symmetric(O_new)))
    end
elseif bug == "ridge_wrong_nu"  # nu_Omega = rho - 1 instead of rho
    @eval CM function RidgePrior(ps::PriorSigma)
        Psi = ps.rho .* ps.R
        Psi11 = Psi[1, 1]
        c = Psi[2:3, 1] ./ Psi11
        Psi221 = Matrix(Symmetric(Psi[2:3, 2:3] - Psi[2:3, 1] * Psi[1, 2:3]' ./ Psi11))
        return RidgePrior(Psi11, c, Psi221, Float64(ps.rho - 1))
    end
end
if bug == "latent_c0y_flip"  # y0 | d*, y1: the coefficient on (y1 - m1) with its sign flipped
    @eval CM function LatentMoments(S::AbstractMatrix{Float64})
        mid = S[2,3] * S[1,3] * S[1,2]
        denom1 = S[2,2] - S[1,2]^2
        denom0 = S[3,3] - S[1,3]^2
        denomd = S[2,2] * S[3,3] - S[2,3]^2
        return LatentMoments(
            ( (S[3,3] * S[1,2]) - (S[2,3] * S[1,3]) ) / denom0,
            ( S[2,3] - (S[1,3] * S[1,2]) ) / denom0,
            S[2,2] - ( ( (S[1,2]^2 * S[3,3]) - 2*mid + (S[2,3]^2) ) / denom0 ),
            ( (S[2,2] * S[1,3]) - (S[2,3] * S[1,2]) ) / denom1,
            -( S[2,3] - (S[1,3] * S[1,2]) ) / denom1,
            S[3,3] - ( ( (S[1,3]^2 * S[2,2]) - 2*mid + (S[2,3]^2) ) / denom1 ),
            ( (S[3,3] * S[1,2]) - (S[2,3] * S[1,3]) ) / denomd,
            ( (S[2,2] * S[1,3]) - (S[2,3] * S[1,2]) ) / denomd,
            1 - ( ( (S[1,2]^2 * S[3,3]) - 2*mid + (S[1,3]^2 * S[2,2]) ) / denomd ))
    end
elseif bug == "latent_trunc_side"  # d* truncated to the wrong side of 0
    @eval CM function update_latent!(rng::AbstractRNG, state::GibbsState, input::GibbsInput, j::Integer, idx::AbstractVector{<:Integer})
        th = state.state_theta[j]
        lm = LatentMoments(th.Sigma)
        sdd = sqrt(lm.omega_d)
        X, W = input.data.X, input.data.W
        y, d = input.data.y.a, input.data.d
        sd = state.state_data
        @inbounds for i in idx
            md, m1, m0 = unit_mean(X, W, th.beta, i)
            y1, y0 = draw_ymiss(rng, lm, md, m1, m0, sd.dstar[i], y[i], d[i])
            sd.y1[i] = y1
            sd.y0[i] = y0
            mu_d = md + (y1 - m1) * lm.sd1 + (y0 - m0) * lm.sd0
            sd.dstar[i] = rand(rng, d[i] == 1 ? truncated(Normal(mu_d, sdd), -Inf, 0.0) : truncated(Normal(mu_d, sdd), 0.0, Inf))
        end
        return state
    end
end
if bug == "alpha_fixed_indicator"  # Escobar-West always with shape a + k (Phase 1 bug 3)
    @eval CM function update_alpha!(rng::AbstractRNG, state::GibbsState, input::GibbsInput)
        state.state_dp.eta = rand(rng, Distributions.Beta(state.state_dp.alpha + 1, input.dims.n))
        b_star = input.priors.prior_dp.alpha_rate - log(state.state_dp.eta)
        a_star = input.priors.prior_dp.alpha_shape + state.state_dp.J
        state.state_dp.alpha = rand(rng, Distributions.Gamma(a_star, 1/b_star))
        return state
    end
end
for (model, s) in (("marginal", :common), ("marginal", :cluster), ("gaussian", :cluster))
    r = geweke_run(model, s; M_sc=M_sc, M_mc=M_mc)
    worst = sort(r.rows, by = x -> -abs(x.z))[1:3]
    println(bug, " ", model == "gaussian" ? "gaussian" : s, " max|z| = ", round(r.maxz, digits=1), "  worst: ", join(["$(x.stat) $(round(x.z, digits=1))" for x in worst], ", "))
end
