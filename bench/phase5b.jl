# bench/phase5b.jl -- Phase 5b: is the shrinkage of sigma12, sigma13 (and of PATT - PATU) on
# generate_hedonic_data(500, 1234) the posterior, or a mixing/initialization artifact of the
# data-augmentation (DA) samplers?
#   julia --project=. bench/phase5b.jl gold [sd_outcome] [sd_intercept] [rho] [r]
#   julia --project=. bench/phase5b.jl da <gaussian|marginal> <default|heckman|overdispersed> [M]
# gold: the observed-data posterior of the k = 1 Gaussian model (src/observed_data.jl, no DA),
#       4 chains from dispersed starts, 50k adaptive burn-in + 400k frozen RWM iterations (thin 20).
# da:   8 chains of gaussian (M = 100k) or marginal :common (M = 50k) from the given init:
#       default (dpm_init: y_miss = y_obs, so sigma1k ~ 0), heckman (two-step gamma, beta,
#       sigma1k; one cluster; latent data imputed by 20 latent passes given theta), overdispersed
#       (heckman beta + N(0, 0.5^2) noise, selection correlations ~ U(-0.9, 0.9), variances x
#       exp(N(0, 0.5^2)); one cluster; latent imputed). The first 20% of saved draws are burn-in.
# All on the standardized scale (scale_data = (true, true)); sigma1k for marginal is the
# n_j-weighted mean over clusters. Effects from effects() (original y scale).
using CausalMixtures, Random, Statistics, Printf, LinearAlgebra, Distributions
const CM = CausalMixtures
const SEED = 20261121

mode = ARGS[1]
data = generate_hedonic_data(500, 1234)
raw = CM.RawData(data.formulas.formula_y, data.formulas.formula_d, data.df)

## Heckman two-step on the sampler's (standardized) data: gamma, beta1, beta0, Sigma
function heckman(input)
    X, W, y, d = input.data.X, input.data.W, input.data.y.a, input.data.d
    g = zeros(size(W, 2))
    for _ in 1:50
        s = W * g; q = 2 .* d .- 1
        lam = q .* pdf.(Normal(), q .* s) ./ cdf.(Normal(), q .* s)
        H = -transpose(W) * (lam .* (lam .+ q .* s) .* W)
        g -= H \ (transpose(W) * lam)
    end
    s = W * g; lt = pdf.(Normal(), s) ./ cdf.(Normal(), s); lu = -pdf.(Normal(), s) ./ (1 .- cdf.(Normal(), s))
    t, u = d .== 1, d .== 0
    b1 = hcat(X[t, :], lt[t]) \ y[t]; b0 = hcat(X[u, :], lu[u]) \ y[u]
    kx = size(X, 2)
    s12, s13 = b1[end], b0[end]
    e1 = y[t] - hcat(X[t, :], lt[t]) * b1; e0 = y[u] - hcat(X[u, :], lu[u]) * b0
    s22, s33 = mean(e1 .^ 2) + s12^2, mean(e0 .^ 2) + s13^2
    S = [1.0 s12 s13; s12 s22 s12 * s13; s13 s12 * s13 s33]
    return vcat(g, b1[1:kx], b0[1:kx]), S
end

## a single-cluster state at (beta, Sigma) with latent data imputed given theta
function set_state!(rng, state, input, beta, Sigma)
    n = input.dims.n
    dp = state.state_dp
    dp.labels = ones(Int, n); dp.njs = [n]; dp.J = 1
    state.state_theta = [CM.Theta(beta=beta, Sigma=Sigma)]
    CM.rebuild_members!(dp)
    for _ in 1:20
        CM.update_latent!(rng, state, input, 1, 1:n)
    end
    return state
end

function overdisperse(rng, beta, S)
    b = beta .+ 0.5 .* randn(rng, length(beta))
    v2, v3 = S[2, 2] * exp(0.5randn(rng)), S[3, 3] * exp(0.5randn(rng))
    r2, r3 = rand(rng, Uniform(-0.9, 0.9)), rand(rng, Uniform(-0.9, 0.9))
    s12, s13 = r2 * sqrt(v2), r3 * sqrt(v3)
    return b, [1.0 s12 s13; s12 v2 s12 * s13; s13 s12 * s13 v3]
end

fmt(v) = @sprintf("%.3f [%.3f, %.3f]", mean(v), quantile(v, 0.025), quantile(v, 0.975))

if mode == "gold"
    sdo = length(ARGS) > 1 ? parse(Float64, ARGS[2]) : 1.0
    sdi = length(ARGS) > 2 ? parse(Float64, ARGS[3]) : 1.0
    rho = length(ARGS) > 3 ? parse(Int, ARGS[4]) : 10
    rr = length(ARGS) > 4 ? parse(Float64, ARGS[5]) : 0.153
    params = CM.InputParams(M=1, scale_data=(true, true), verbose=false, model="gaussian")
    pr = setup_default_priors(2, 3; J=1, sd_outcome=sdo, sd_intercept=sdi, rho=rho, r=rr)
    _, input, _ = CM.dpm_init(raw, pr, params; rng=Xoshiro(1))
    bh, Sh = heckman(input)
    starts = [(bh, Sh), overdisperse(Xoshiro(11), bh, Sh), overdisperse(Xoshiro(12), bh, Sh),
              (bh .* 0, [1.0 0 0; 0 Sh[2, 2] 0; 0 0 Sh[3, 3]])]
    s12, s13, effs, accs, Ds = Vector{Float64}[], Vector{Float64}[], CM.Effects[], Float64[], Matrix{Float64}[]
    for (c, (b0, S0)) in enumerate(starts)
        draws, acc = obsdata_sampler(Xoshiro(SEED + c), input, CM.obsdata_pack(b0, S0); M=400_000, burn=50_000, thin=20)
        push!(accs, acc); push!(Ds, draws)
        out = obsdata_out(draws, input)
        push!(s12, [t[1].Sigma[1, 2] for t in out.out_theta]); push!(s13, [t[1].Sigma[1, 3] for t in out.out_theta])
        push!(effs, effects(out, input; rng=Xoshiro(SEED + 100 + c)))
    end
    es = effects_summary(effs)
    println("# gold standard (observed-data RWM, no DA), sd_outcome = $sdo, sd_intercept = $sdi, rho = $rho, r = $rr (prior mean of Omega_kk = rho r/(rho - 3) = $(round(rho * rr / (rho - 3), digits=3)))")
    @printf("acceptance %s; R-hat over all %d parameters: max %.4f\n", join([@sprintf("%.2f", a) for a in accs], ", "),
            size(Ds[1], 2), maximum(rhat([D[:, k] for D in Ds]) for k in 1:size(Ds[1], 2)))
    @printf("sigma12 %s (R-hat %.4f, ESS %.0f)\n", fmt(reduce(vcat, s12)), rhat(s12), ess_bulk(s12))
    @printf("sigma13 %s (R-hat %.4f, ESS %.0f)\n", fmt(reduce(vcat, s13)), rhat(s13), ess_bulk(s13))
    for k in (:pate, :patt, :patu)
        v = getfield(es, k)
        @printf("%s %.3f [%.3f, %.3f] (R-hat %.4f, ESS %.0f)\n", k, v.mean, v.q[1], v.q[3], v.rhat, v.ess_bulk)
    end
    @printf("Heckman two-step (standardized): sigma12 %.3f, sigma13 %.3f\n", Sh[1, 2], Sh[1, 3])
else
    sampler, init = ARGS[2], ARGS[3]
    model = sampler
    M = length(ARGS) > 3 ? parse(Int, ARGS[4]) : (model == "gaussian" ? 100_000 : 50_000)
    thin = max(1, M ÷ 10_000)
    params = CM.InputParams(M=M, thin=thin, scale_data=(true, true), verbose=false, model=model, slopes=:common)
    pr = setup_default_priors(2, 3; J = model == "gaussian" ? 1 : 20)
    s12, s13, effs = Vector{Float64}[], Vector{Float64}[], CM.Effects[]
    for c in 1:8
        rng = Xoshiro(SEED + c)
        state, input, out = CM.dpm_init(raw, pr, params; rng=rng)
        if init != "default"
            bh, Sh = heckman(input)
            b0, S0 = init == "heckman" ? (bh, Sh) : overdisperse(rng, bh, Sh)
            set_state!(rng, state, input, b0, S0)
        end
        _, input, out = CM.dpm!(state, input, out; rng=rng)
        w(dp, th, a, b) = sum(dp.njs[j] * th[j].Sigma[a, b] for j in eachindex(dp.njs) if dp.njs[j] > 0) / sum(dp.njs)
        push!(s12, [w(out.out_dp[s], out.out_theta[s], 1, 2) for s in eachindex(out.out_dp)])
        push!(s13, [w(out.out_dp[s], out.out_theta[s], 1, 3) for s in eachindex(out.out_dp)])
        push!(effs, effects(out, input; rng=Xoshiro(SEED + 100 + c)))
    end
    S = length(s12[1]); burn = S ÷ 5
    post(v) = [x[(burn + 1):end] for x in v]
    println("# DA $model, init = $init, 8 chains x $M iterations (thin $thin; first 20% of saved draws burn-in)")
    @printf("sigma12 %s (R-hat %.3f)\n", fmt(reduce(vcat, post(s12))), rhat(post(s12)))
    @printf("sigma13 %s (R-hat %.3f)\n", fmt(reduce(vcat, post(s13))), rhat(post(s13)))
    for k in (:pate, :patt, :patu)
        v = post([getfield(e, k) for e in effs])
        @printf("%s %s (R-hat %.3f, ESS %.0f)\n", k, fmt(reduce(vcat, v)), rhat(v), ess_bulk(v))
    end
    ## trajectories: mean over chains in 10 windows of the whole run (burn-in included)
    win = [((k - 1) * S ÷ 10 + 1):(k * S ÷ 10) for k in 1:10]
    traj(v) = join([@sprintf("%.3f", mean(mean(x[r]) for x in v)) for r in win], " ")
    println("trajectory (window means over chains, 10 windows of ", M ÷ 10, " iterations):")
    println("  sigma12: ", traj(s12)); println("  sigma13: ", traj(s13))
    println("  patt:    ", traj([e.patt for e in effs])); println("  patu:    ", traj([e.patu for e in effs]))
    ## per-chain final-window means, to show spread across chains
    println("  per-chain sigma13 in the last window: ", join([@sprintf("%.3f", mean(x[win[end]])) for x in s13], " "))
    ## save the post-burn-in pooled draws for the cross-init R-hat
    open("bench/phase5b_draws_$(model)_$(init).txt", "w") do io
        for c in 1:8, (a, b, p, q, r) in zip(post(s12)[c], post(s13)[c], post([e.pate for e in effs])[c], post([e.patt for e in effs])[c], post([e.patu for e in effs])[c])
            println(io, c, " ", a, " ", b, " ", p, " ", q, " ", r)
        end
    end
end
