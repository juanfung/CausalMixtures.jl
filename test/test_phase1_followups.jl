# test/test_phase1_followups.jl -- follow-ups requested after the Phase 1 review
using Test, CausalMixtures, LinearAlgebra, Random, Statistics, StatsModels
isdefined(Main, :TestHelpers) || include("TestHelpers.jl")
using .TestHelpers

@testset "Phase 1 follow-ups" begin

@testset "dpm_ppd builds its design by column name" begin
    data = generate_hedonic_data(120, 17)
    df = data.df
    raw_xz = CausalMixtures.RawData(@formula(Y_obs ~ X), @formula(D_obs ~ X + Z), df)
    raw_zx = CausalMixtures.RawData(@formula(Y_obs ~ X), @formula(D_obs ~ Z + X), df)
    priors = setup_default_priors(data.true_params.ktot, J=3)
    params = CausalMixtures.InputParams(M=8, verbose=false, model="dpm")
    seed_global!()
    _, in_xz, out_xz = CausalMixtures.dpm!(CausalMixtures.dpm_init(raw_xz, priors, params)...)
    _, in_zx, _ = CausalMixtures.dpm_init(raw_zx, priors, params)

    ## the same posterior, every component's selection coefficients in Z + X order
    kz = in_xz.dims.kz
    perm = [findfirst(==(nm), in_xz.data.znames) for nm in in_zx.data.znames]
    out_zx = CausalMixtures.GibbsOut(params.M)
    for m in 1:params.M
        th = CausalMixtures.StateTheta()
        for (j, t) in out_xz.out_theta[m]
            b = copy(t.beta); b[1:kz] = t.beta[perm]
            th[j] = CausalMixtures.Theta(b, t.Sigma)
        end
        out_zx.out_theta[m] = th
        out_zx.out_dp[m] = out_xz.out_dp[m]
        out_zx.out_data[m] = out_xz.out_data[m]
    end

    _, w_xz = mean_covariates(in_xz)
    _, w_zx = mean_covariates(in_zx)
    seed_global!(5); a = dpm_ppd(out_xz, in_xz, w_xz)
    seed_global!(5); b = dpm_ppd(out_zx, in_zx, w_zx)
    @test size(a) == (3, params.M)
    @test a ≈ b
    ## (xnew, wnew) form agrees with the znew form
    seed_global!(5); c = dpm_ppd(out_zx, in_zx, mean_covariates(in_zx)...)
    @test c ≈ b
    ## and with rand_ppd for the dpm model
    seed_global!(5); d = rand_ppd(out_zx, in_zx, w_zx)
    @test d ≈ b
end

end # Phase 1 follow-ups
