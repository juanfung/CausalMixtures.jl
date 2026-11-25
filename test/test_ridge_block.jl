# test/test_ridge_block.jl -- Phase 4f: the ridge block (joint outcome coefficients and
# delta = (sigma12, sigma13) | Omega, then Omega)
using Test, CausalMixtures, LinearAlgebra, Random, Statistics, Distributions
isdefined(Main, :TestHelpers) || include("TestHelpers.jl")
using .TestHelpers
const CMR = CausalMixtures

@testset "Ridge block (Phase 4f)" begin

@testset "(delta, Omega) <-> Sigma round trip" begin
    rng = test_rng()
    for _ in 1:20
        S = CMR.NobileWishart(rng, 10, Matrix(Diagonal([1.0, 1.53, 1.53])))
        d, O = CMR.sigma_delta_omega(S)
        @test CMR.sigma_from_delta_omega(d, O) ≈ S
        @test isposdef(O)
    end
end

@testset "prior mapping: NobileWishart(rho, Psi) = N(c, Omega/Psi11) x IW_2(rho, Psi_22.1) ($nm)" for (nm, rho, Psi) in (
        ("C3", 10, 10 .* Matrix(Diagonal([0.1, 0.153, 0.153]))),
        ("general Psi", 9, [5.0 1 -1; 1 4 0.5; -1 0.5 3]))
    rng = test_rng()
    rp = CMR.RidgePrior(CMR.PriorSigma(rho, Psi ./ rho))
    @test rp.nu == rho
    nm == "C3" && @test rp.Psi11 ≈ 1 && rp.c == [0.0, 0.0] && rp.Psi221 ≈ Diagonal([1.53, 1.53])
    N = 200_000
    ent(d, O) = [d[1], d[2], O[1, 1], O[1, 2], O[2, 2]]
    A = reduce(hcat, [ent(CMR.sigma_delta_omega(CMR.NobileWishart(rng, rho, Psi))...) for _ in 1:N])'
    function claimed(nuO)
        IW = InverseWishart(nuO, rp.Psi221)
        reduce(hcat, [begin O = rand(rng, IW); d = rp.c + cholesky(Symmetric(O ./ rp.Psi11)).L * randn(rng, 2); ent(d, O) end for _ in 1:N])'
    end
    B = claimed(rho)
    sq(X) = (X .- mean(X, dims=1)) .^ 2
    @test maximum(abs, two_sample_zscores(A, B)) < 4.5
    @test maximum(abs, two_sample_zscores(sq(A), sq(B))) < 4.5
    ## power: nu_Omega = rho - 1 or rho + 1 is detected
    for nuO in (rho - 1, rho + 1)
        Bw = claimed(nuO)
        @test max(maximum(abs, two_sample_zscores(A, Bw)), maximum(abs, two_sample_zscores(sq(A), sq(Bw)))) > 10
    end
end

## brute force: the log joint density of the original model in (beta^c, Sigma), for n units with
## z_i ~ N3(H_i beta^c, Sigma), beta^c ~ N(mu_c, Vc), Sigma ~ NobileWishart(rho, Psi) (unnormalized)
function log_joint(Z, W, X, bc, Sigma, mu_c, Vc, rho, Psi)
    kz, kx = size(W, 2), size(X, 2)
    lp = logpdf(MvNormal(mu_c, Symmetric(Vc)), bc)
    lp += -(rho + 4) / 2 * logdet(Sigma) - tr(Psi * inv(Sigma)) / 2
    for i in axes(Z, 1)
        m = [dot(W[i, :], bc[1:kz]), dot(X[i, :], bc[kz+1:kz+kx]), dot(X[i, :], bc[kz+kx+1:end])]
        lp += logpdf(MvNormal(m, Symmetric(Sigma)), Z[i, :])
    end
    return lp
end

@testset "conditionals = brute-force log joint ($(corr ? "correlated" : "independent") prior)" for corr in (false, true)
    rng = test_rng()
    n, kz, kx = 15, 2, 2
    W, X = hcat(ones(n), randn(rng, n, kz - 1)), hcat(ones(n), randn(rng, n, kx - 1))
    q = kz + 2kx
    mu_c = randn(rng, q)
    A = randn(rng, q, q)
    Vc = corr ? A * A' / q + 0.5I : Matrix(Diagonal(0.5 .+ rand(rng, q)))
    rho, Psi = 9, [5.0 1 -1; 1 4 0.5; -1 0.5 3]
    rp = CMR.RidgePrior(CMR.PriorSigma(rho, Psi ./ rho))
    Z = randn(rng, n, 3) .* 2
    ztf = i -> (Z[i, 1], Z[i, 2], Z[i, 3])
    bc0 = randn(rng, q)
    S0 = CMR.NobileWishart(rng, rho, Psi)
    _, O = CMR.sigma_delta_omega(S0)
    P, r, ob = CMR.ridge_theta_post(1:n, ztf, W, X, bc0, mu_c, Vc, O, rp)
    post = MvNormal(P \ r, Symmetric(inv(P)))
    p = length(ob)
    ## (B, delta) | Omega, selection coefficients: differences of the log joint over theta equal
    ## differences of the Gaussian log density
    th(t) = (bc = copy(bc0); bc[ob] = t[1:p]; (bc, CMR.sigma_from_delta_omega(t[p+1:end], O)))
    ts = [rand(rng, post) for _ in 1:6]
    L = [log_joint(Z, W, X, th(t)..., mu_c, Vc, rho, Psi) for t in ts]
    G = [logpdf(post, t) for t in ts]
    @test (L .- L[1]) ≈ (G .- G[1]) rtol=1e-8 atol=1e-8
    ## Omega | B, delta: differences over Omega equal those of IW_2(rho + n + 1, ...)
    t = ts[2]
    bc, _ = th(t)
    d = t[p+1:end]
    df, Sc = CMR.ridge_omega_post(1:n, ztf, W, X, bc, d, rp)
    @test df == rho + n + 1
    IWp = InverseWishart(df, Sc)
    Os = [rand(rng, IWp) for _ in 1:6]
    L2 = [log_joint(Z, W, X, bc, CMR.sigma_from_delta_omega(d, Matrix(Symmetric(Oa))), mu_c, Vc, rho, Psi) for Oa in Os]
    G2 = [logpdf(IWp, Oa) for Oa in Os]
    @test (L2 .- L2[1]) ≈ (G2 .- G2[1]) rtol=1e-8 atol=1e-8
    ## ridge_draw: the joint theta draws follow the Gaussian conditional; Sigma valid
    N = 50_000
    D = zeros(N, p + 2)
    for k in 1:N
        bcn, Sn = CMR.ridge_draw(rng, 1:n, ztf, W, X, bc0, mu_c, Vc, S0, rp)
        D[k, :] = vcat(bcn[ob], Sn[1, 2], Sn[1, 3])
        k <= 100 && (@test Sn[1, 1] == 1 && isposdef(Sn) && bcn[1:kz] == bc0[1:kz])
    end
    @test maximum(abs, (vec(mean(D, dims=1)) - mean(post)) ./ sqrt.(diag(cov(post)) / N)) < 4.5
    @test maximum(abs, cov(D) - cov(post)) < 0.05 * maximum(abs, cov(post))
end

@testset "ridge_block option: default on, both samplers run" begin
    @test CMR.InputParams().ridge_block == true
    data = generate_hedonic_data(60, 5)
    raw = CMR.RawData(data.formulas.formula_y, data.formulas.formula_d, data.df)
    for (model, sl) in (("marginal", :common), ("marginal", :cluster), ("gaussian", :cluster)), rb in (true, false)
        params = CMR.InputParams(M=10, scale_data=(true, true), verbose=false, model=model, slopes=sl, ridge_block=rb)
        rng = Xoshiro(3)
        st, input, out = CMR.dpm!(CMR.dpm_init(raw, setup_default_priors(2, 3; J = model == "gaussian" ? 1 : 20, slopes=sl), params; rng=rng)...; rng=rng)
        @test all(t -> t.Sigma[1, 1] ≈ 1 && isposdef(t.Sigma), st.state_theta)  # NobileWishart: 1 up to rounding
        @test all(isfinite, trace_ate(out, input))
    end
end

end # Phase 4f
