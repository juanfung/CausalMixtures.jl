# test/GewekeHarness.jl -- Geweke (2004) "getting it right" test for the samplers.
#
# Marginal-conditional simulator: independent forward draws of (alpha, c, theta, z) from the prior
# with prior_draw (src/prior_predictive.jl), which never calls the samplers' own prior draws.
# Successive-conditional simulator: alternate
#     (1) simulate the data (d, y) and the latent z from the likelihood given the current (c, theta);
#     (2) one full sampler sweep given (d, y),
# starting from a forward draw. If the sweep leaves the posterior invariant, both simulators draw
# the same joint distribution, so every statistic g(theta) has the same mean. We compare the means
# of label-invariant statistics and their squares with z = (mean_mc - mean_sc) / se, where the
# successive-conditional MC error uses its effective sample size (ess_mean).
module GewekeHarness

using CausalMixtures, Random, Statistics, LinearAlgebra, Distributions
const CM = CausalMixtures

export geweke_run, geweke_table, HIER

## hierarchical intercepts in the marginal sampler (Phase 4d default); set HIER[] = false for
## the fixed-prior sampler
const HIER = Ref(true)

## the model's prior as a PriorSpec (C3; shared outcome slopes N(0, 1) under :common)
function harness_spec(model::String, slopes::Symbol)
    if model == "gaussian"
        return PriorSpec(slopes=:cluster, sd_intercept=1, sd_intercept_d=0.5, sd_probit=0.5, sd_outcome=0.5,
                         rho=10, r=0.153, alpha=1e-12, alpha_shape=0.0)
    end
    return PriorSpec(slopes=slopes, sd_intercept=1, sd_intercept_d=0.5, sd_probit=0.5,
                     sd_outcome = slopes == :common ? 1.0 : 0.5, rho=10, r=0.153,
                     alpha_shape=2.0, alpha_rate=2.0, hierarchical = model == "marginal" && HIER[])
end

## sampler input for n units with the standardized covariates of generate_hedonic_data(n, seed)
function harness_input(model::String, slopes::Symbol; n::Int=20, seed::Int=1, J::Int=10)
    data = generate_hedonic_data(n, seed)
    raw = CM.RawData(data.formulas.formula_y, data.formulas.formula_d, data.df)
    params = CM.InputParams(M=1, scale_data=(true, true), verbose=false, model=model, slopes=slopes,
                            hierarchical = model == "marginal" && HIER[])
    pr = setup_default_priors(2, 3; J = model == "gaussian" ? 1 : J,
                              slopes = model == "marginal" ? slopes : :cluster)
    state, input, _ = CM.dpm_init(raw, pr, params; rng=Xoshiro(seed))
    return state, input
end

## positions of the intercepts in beta = (gamma, beta1, beta0) (intercepts in column 1)
intercepts(input) = (1, input.dims.kz + 1, input.dims.kz + input.dims.kx + 1)
slope_idx(input) = setdiff(1:(input.dims.kz + 2input.dims.kx), collect(intercepts(input)))

const SIGMA_NAMES = ("s12", "s13", "s22", "s23", "s33")
sigma_entries(S) = (S[1, 2], S[1, 3], S[2, 2], S[2, 3], S[3, 3])

## label-invariant statistics: alpha, k, unit 1's intercepts and Sigma, the shared slopes,
## mean_i (a_1 - a_0)_{c_i}, and the share of units in unit 1's cluster
function stat_names(model, slopes, input)
    nm = String[]
    model == "gaussian" || append!(nm, ["alpha", "k"])
    append!(nm, ["a_d", "a_1", "a_0"])
    append!(nm, collect(SIGMA_NAMES))
    if slopes == :common && model == "marginal"
        append!(nm, ["b$(j)" for j in eachindex(slope_idx(input))])
    end
    if model == "marginal" && HIER[]
        append!(nm, ["m_d", "m_1", "m_0", "log tau2_d", "log tau2_1", "log tau2_0"])
    end
    push!(nm, "mean(a1 - a0)")
    model == "gaussian" || push!(nm, "share(c_1)")
    return nm
end

function stats_from(model, slopes, input, alpha, labels, njs, betas, Sigmas, hyper=Float64[])
    n = length(labels)
    i_d, i_1, i_0 = intercepts(input)
    c1 = labels[1]
    v = Float64[]
    model == "gaussian" || append!(v, [alpha, count(>(0), njs)])
    append!(v, [betas[c1][i_d], betas[c1][i_1], betas[c1][i_0]])
    append!(v, collect(sigma_entries(Sigmas[c1])))
    if slopes == :common && model == "marginal"
        append!(v, betas[c1][slope_idx(input)])
    end
    if model == "marginal" && HIER[]
        append!(v, hyper[1:3], log.(hyper[4:6]))
    end
    push!(v, mean(betas[labels[i]][i_1] - betas[labels[i]][i_0] for i in 1:n))
    model == "gaussian" || push!(v, njs[c1] / n)
    return v
end

## statistics of one forward draw
function forward_stats(rng, model, slopes, input, spec, J)
    X, W = input.data.X, input.data.W
    r = prior_draw(spec, X, W; rng=rng, partition = model == "blocked" ? :stick : :crp, J=J)
    betas = [vcat(r.gam[j], r.b1[j], r.b0[j]) for j in eachindex(r.counts)]
    return stats_from(model, slopes, input, r.alpha, r.labels, r.counts, betas, r.Sigma, vcat(r.m, r.tau2)), r
end

## put a forward draw into the sampler state and the data into the input
function load_draw!(rng, state, input, r, model)
    K = length(r.counts)
    dp = state.state_dp
    dp.labels = copy(r.labels)
    dp.njs = copy(r.counts)
    dp.alpha = r.alpha
    state.state_theta = [CM.Theta(beta=vcat(r.gam[j], r.b1[j], r.b0[j]), Sigma=copy(r.Sigma[j])) for j in 1:K]
    if model == "blocked"
        dp.ws = fill(CM.BlockedWeights(), K)
        CM.compute_weights!(rng, dp)          # v | c, alpha: completes a joint draw
        CM.update_J!(dp)
    else
        dp.J = K
    end
    CM.rebuild_members!(dp)
    state.state_hyper = CM.StateHyper(m=copy(r.m), tau2=copy(r.tau2))
    set_data!(input, state, r.z)
    return state
end

## data (d, y) and latent z into the input and the state
function set_data!(input, state, z::AbstractMatrix)
    d = Int.(z[:, 1] .> 0)
    input.data.d .= d
    input.data.y.a .= ifelse.(d .== 1, z[:, 2], z[:, 3])
    input.data.lower .= ifelse.(d .== 1, 0.0, -Inf)
    input.data.upper .= ifelse.(d .== 1, Inf, 0.0)
    state.state_data.dstar .= z[:, 1]
    state.state_data.y1 .= z[:, 2]
    state.state_data.y0 .= z[:, 3]
    return input
end

## simulate z_i ~ N3(H_i beta_{c_i}, Sigma_{c_i}) for every unit, given the state
function simulate_z(rng, state, input)
    n = input.dims.n
    z = Matrix{Float64}(undef, n, 3)
    for i in 1:n
        th = state.state_theta[state.state_dp.labels[i]]
        md, m1, m0 = CM.unit_mean(input.data.X, input.data.W, th.beta, i)
        e = cholesky(Symmetric(th.Sigma)).L * randn(rng, 3)
        z[i, :] = [md, m1, m0] .+ e
    end
    return z
end

function sweep!(rng, state, input, model, D)
    if model == "marginal"
        CM.marginal_sweep!(rng, state, input, D)
    elseif model == "gaussian"
        CM.update_gaussian_params!(rng, state, input)
    elseif model == "blocked"
        CM.update_blocked_labels!(rng, state, input)
        CM.update_blocked_params!(rng, state, input)
        CM.update_blocked_alpha!(rng, state, input)
    else
        error("no Geweke sweep for $model")
    end
    return state
end

"""
    geweke_run(model, slopes; M_sc, M_mc, n=20, J=10, seed=1, rng) -> NamedTuple

Run both simulators and return the z-scores for every statistic (and its square with the
default `powers = (1, 2)`) and the summaries (means, the successive-conditional ESS).
"""
function geweke_run(model::String, slopes::Symbol=:cluster; M_sc::Int=100_000, M_mc::Int=100_000,
                    n::Int=20, J::Int=10, seed::Int=1, rng::AbstractRNG=Xoshiro(seed), powers=(1, 2))
    state, input = harness_input(model, slopes; n=n, seed=seed, J=J)
    spec = harness_spec(model, slopes)
    names = stat_names(model, slopes, input)
    ## marginal-conditional
    G_mc = Matrix{Float64}(undef, M_mc, length(names))
    for m in 1:M_mc
        G_mc[m, :] = forward_stats(rng, model, slopes, input, spec, J)[1]
    end
    ## successive-conditional, started at a forward draw
    _, r = forward_stats(rng, model, slopes, input, spec, J)
    load_draw!(rng, state, input, r, model)
    D = model == "marginal" ? DesignA(input) : nothing
    G_sc = Matrix{Float64}(undef, M_sc, length(names))
    t = @elapsed for m in 1:M_sc
        set_data!(input, state, simulate_z(rng, state, input))
        sweep!(rng, state, input, model, D)
        dp = state.state_dp
        G_sc[m, :] = stats_from(model, slopes, input, dp.alpha, dp.labels, dp.njs,
                                [t.beta for t in state.state_theta], [t.Sigma for t in state.state_theta],
                                vcat(state.state_hyper.m, state.state_hyper.tau2))
    end
    ## z-scores for g and g^2 (k and share: their own squares are informative too)
    rows = NamedTuple[]
    for (s, nm) in enumerate(names), pw in powers
        a, b = G_mc[:, s] .^ pw, G_sc[:, s] .^ pw
        ess = ess_mean(b)
        se = sqrt(var(a) / length(a) + var(b) / ess)
        push!(rows, (stat = pw == 1 ? nm : "$(nm)^2", mean_mc=mean(a), mean_sc=mean(b),
                     ess=ess, z=(mean(a) - mean(b)) / se))
    end
    return (model=model, slopes=slopes, rows=rows, maxz=maximum(abs(r.z) for r in rows), secs=t, M_sc=M_sc, M_mc=M_mc)
end

## markdown table of one run
function geweke_table(io::IO, res)
    println(io, "| Statistic | forward mean | sampler mean | sampler ESS | z |")
    println(io, "|---|---|---|---|---|")
    for r in res.rows
        println(io, "| ", r.stat, " | ", round(r.mean_mc, sigdigits=4), " | ", round(r.mean_sc, sigdigits=4),
                " | ", round(Int, r.ess), " | ", round(r.z, digits=2), " |")
    end
end

end # module
