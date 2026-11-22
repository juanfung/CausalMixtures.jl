# bench/geweke_power.jl -- power check of the Geweke harness: plant one bug by redefining a
# method, then run the Design A Geweke test (50k sweeps vs 100k forward draws) in both slope modes.
#   julia --project=. bench/geweke_power.jl <bug>
# bugs: none, sigma_dof, no_beta_unc, no_phi, no_shift, alpha_fixed_indicator
include(joinpath(@__DIR__, "..", "test", "GewekeHarness.jl")); using .GewekeHarness
using CausalMixtures, Random, LinearAlgebra
const CM = CausalMixtures
bug = ARGS[1]
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
elseif bug == "alpha_fixed_indicator"  # Escobar-West always with shape a + k (Phase 1 bug 3)
    @eval CM function update_alpha!(rng::AbstractRNG, state::GibbsState, input::GibbsInput)
        state.state_dp.eta = rand(rng, Distributions.Beta(state.state_dp.alpha + 1, input.dims.n))
        b_star = input.priors.prior_dp.alpha_rate - log(state.state_dp.eta)
        a_star = input.priors.prior_dp.alpha_shape + state.state_dp.J
        state.state_dp.alpha = rand(rng, Distributions.Gamma(a_star, 1/b_star))
        return state
    end
end
for s in (:common, :cluster)
    r = geweke_run("marginal", s; M_sc=50_000, M_mc=100_000)
    worst = sort(r.rows, by = x -> -abs(x.z))[1:3]
    println(bug, " ", s, " max|z| = ", round(r.maxz, digits=1), "  worst: ", join(["$(x.stat) $(round(x.z, digits=1))" for x in worst], ", "))
end
