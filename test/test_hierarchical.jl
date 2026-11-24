# test/test_hierarchical.jl -- Phase 4d: hierarchical cluster intercepts and the separated-units
# diagnostic
using Test, CausalMixtures, LinearAlgebra, Random, Statistics, Distributions
isdefined(Main, :TestHelpers) || include("TestHelpers.jl")
using .TestHelpers
const CMH = CausalMixtures

function hier_setup(sl; n=40, seed=3, hierarchical=true)
    data = generate_hedonic_data(n, seed)
    raw = CMH.RawData(data.formulas.formula_y, data.formulas.formula_d, data.df)
    params = CMH.InputParams(M=1, scale_data=(true, true), verbose=false, model="marginal", slopes=sl,
                             hierarchical=hierarchical)
    rng = Xoshiro(11)
    state, input, _ = CMH.dpm_init(raw, setup_default_priors(2, 3, J=4, slopes=sl), params; rng=rng)
    return state, input, CMH.DesignA(input)
end

## put k clusters with intercepts A[:, j] into the state (other coefficients from cluster 1)
function set_intercepts!(state, D, A)
    k = size(A, 2)
    base = state.state_theta[1]
    pos = D.cidx[D.icidx]
    state.state_theta = [begin b = copy(base.beta); b[pos] = A[:, j]; CMH.Theta(beta=b, Sigma=base.Sigma) end for j in 1:k]
    state.state_dp.njs = ones(Int, k)
    state.state_dp.J = k
    return state
end

@testset "Hierarchical intercepts (Phase 4d)" begin

@testset "m | a, tau2 and tau2 | a, m match their closed forms ($sl)" for sl in (:common, :cluster)
    state, input, D = hier_setup(sl)
    rng = test_rng()
    @test D.hier && D.mu_m == zeros(3) && D.Vm ≈ [0.25, 1, 1] && D.tau_scale ≈ [0.5, 2, 2]
    A = [0.3 -0.2 0.5 0.1; 1.2 0.4 -0.7 0.9; -0.5 0.0 0.8 0.3]   # k = 4 clusters
    k = size(A, 2)
    set_intercepts!(state, D, A)
    tau2 = [0.2, 0.7, 1.5]
    N = 100_000
    ms, ts = zeros(N, 3), zeros(N, 3)
    for r in 1:N
        state.state_hyper = CMH.StateHyper(m=zeros(3), tau2=copy(tau2))
        CMH.update_hyper!(rng, state, D)
        ms[r, :] = state.state_hyper.m
        ts[r, :] = state.state_hyper.tau2
    end
    ## m: P_m = V_m^-1 + k T^-1, mean P_m^-1 (V_m^-1 mu_m + T^-1 sum_j a_j) (dense)
    Vm, T = Diagonal(D.Vm), Diagonal(tau2)
    Pm = inv(Vm) + k * inv(T)
    mm = Pm \ (inv(Vm) * D.mu_m + inv(T) * vec(sum(A, dims=2)))
    Cm = inv(Pm)
    @test maximum(abs, (vec(mean(ms, dims=1)) - mm) ./ sqrt.(diag(Cm) / N)) < 4.5
    @test maximum(abs, vec(var(ms, dims=1)) ./ diag(Cm) .- 1) < 0.03
    ## tau2 given the drawn m: the probability integral transform under IG(3 + k/2, 2 s^2 + SS/2) is U(0, 1)
    U = [cdf(InverseGamma(3 + k / 2, D.tau_scale[l] + sum((A[l, :] .- ms[r, l]) .^ 2) / 2), ts[r, l]) for r in 1:N, l in 1:3]
    for l in 1:3
        u = sort(U[:, l])
        @test maximum(abs, u .- (1:N) ./ N) < 1.95 / sqrt(N)   # KS at alpha = 0.001
    end
    ## the prior of beta^c follows: intercept block (m, diag(tau2)), slopes unchanged
    @test D.mu_c[D.icidx] == state.state_hyper.m && diag(D.Vc)[D.icidx] == state.state_hyper.tau2
end

@testset "invariance: (m, tau2) | a leaves the hierarchical prior invariant; shape 3 + k fails" begin
    state, input, D = hier_setup(:common)
    rng = test_rng()
    k, N = 3, 100_000
    draw_prior() = (m = sqrt.(D.Vm) .* randn(rng, 3); t = [rand(rng, InverseGamma(3.0, s)) for s in D.tau_scale]; (m, t))
    stats(m, t) = vcat(m, log.(t))
    function after_step(shape)
        R = zeros(N, 6)
        for r in 1:N
            m, t = draw_prior()
            A = m .+ sqrt.(t) .* randn(rng, 3, k)
            set_intercepts!(state, D, A)
            state.state_hyper = CMH.StateHyper(m=m, tau2=t)
            D2 = shape == 3.0 ? D : CMH.DesignA(D.slopes, D.cidx, D.gidx, D.Wc, D.Xc, D.Wg, D.Xg, D.mu_c, D.Vc,
                D.Vc_inv, D.Vc_inv_mu, D.Lc, D.mu_g, D.Vg_inv, D.Vg_inv_mu, D.aux_mean, D.aux_cov, D.m_aux,
                D.hier, D.icidx, D.mu_m, D.Vm, shape, D.tau_scale)
            CMH.update_hyper!(rng, state, D2)
            R[r, :] = stats(state.state_hyper.m, state.state_hyper.tau2)
        end
        return R
    end
    P = reduce(vcat, [stats(draw_prior()...)' for _ in 1:N])
    R = after_step(3.0)
    sq(X) = (X .- mean(X, dims=1)) .^ 2
    @test maximum(abs, two_sample_zscores(R, P)) < 4.5
    @test maximum(abs, two_sample_zscores(sq(R), sq(P))) < 4.5
    ## power: the planted shape (3 + k/2 -> 3 + k, here via tau_shape 3 + k/2) is detected
    Rb = after_step(3.0 + k / 2)
    @test maximum(abs, two_sample_zscores(Rb, P)) > 10
end

@testset "hierarchical = false keeps the fixed prior" begin
    state, input, D = hier_setup(:common; hierarchical=false)
    @test !D.hier && isempty(state.state_hyper.m)
    rng = Xoshiro(1)
    Vc0, mu0 = copy(D.Vc), copy(D.mu_c)
    for _ in 1:5
        CMH.marginal_sweep!(rng, state, input, D)
    end
    @test D.Vc == Vc0 && D.mu_c == mu0 && isempty(state.state_hyper.m)
end

@testset "separated-units share" begin
    dp = CMH.StateDP(J=3, labels=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3], njs=[3, 2, 5], ws=CMH.BlockedWeights[], alpha=1.0, eta=0.0)
    CMH.rebuild_members!(dp)
    ## cluster 1 all treated (separated), cluster 2 half, cluster 3 none treated (separated)
    d = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
    @test CMH.separated_share(dp, d) ≈ 0.8
    @test CMH.separated_share(dp, [1, 0, 1, 1, 0, 1, 0, 0, 0, 0]) ≈ 0.0
    ## stored per saved draw, with (m, tau2); a run with a separated share above 5% warns
    data = generate_hedonic_data(80, 2)
    raw = CMH.RawData(data.formulas.formula_y, data.formulas.formula_d, data.df)
    params = CMH.InputParams(M=6, thin=2, scale_data=(true, true), verbose=false, model="marginal", slopes=:common)
    rng = Xoshiro(3)
    st, input, out = CMH.dpm!(CMH.dpm_init(raw, setup_default_priors(2, 3, slopes=:common), params; rng=rng)...; rng=rng)
    @test length(out.out_sep) == 3 && all(0 .<= out.out_sep .<= 1)
    @test length(out.out_hyper) == 3 && all(length.(out.out_hyper) .== 6) && out.out_hyper[end] == vcat(st.state_hyper.m, st.state_hyper.tau2)
    out.out_sep .= 0.5
    @test_logs (:warn, r"separated") CMH.warn_separated(out)
    out.out_sep .= 0.01
    @test_logs CMH.warn_separated(out)
end

end # Phase 4d
