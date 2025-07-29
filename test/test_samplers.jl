# test/test_samplers.jl - Test individual sampler functions
using Test
using CausalMixtures

# Create test data
data_test = CausalMixtures.generate_hedonic_data(200, 1111)
    
@testset "Gaussian Sampler Smoke Test" begin
    # FIXED: Set J=1 in priors, not params
    priors_gaussian = CausalMixtures.setup_default_priors(data_test.true_params.ktot, J=1)
    raw_data_test = CausalMixtures.RawData(data_test.formulas.formula_y, 
                                           data_test.formulas.formula_d, data_test.df)
    params_gaussian = CausalMixtures.InputParams(M=5, verbose=false, model="gaussian")

    # pre-declare for scoping
    init_gaussian = nothing
    out_gaussian = nothing
    # Test initialization doesn't crash
    @test_nowarn init_gaussian = CausalMixtures.dpm_init(raw_data_test, priors_gaussian, params_gaussian)
    @test_nowarn out_gaussian = CausalMixtures.dpm!(init_gaussian...)
    
    # Test output structure is reasonable
    state, input, output = out_gaussian #CausalMixtures.dpm!(init_gaussian...)
    @test length(output.out_dp) == params_gaussian.M
    @test length(output.out_theta) == params_gaussian.M
end
    
@testset "FMN Sampler Smoke Test" begin
    priors_test = CausalMixtures.setup_default_priors(data_test.true_params.ktot)
    raw_data_test = CausalMixtures.RawData(data_test.formulas.formula_y, 
                                           data_test.formulas.formula_d, data_test.df)
    params_fmn = CausalMixtures.InputParams(M=5, verbose=false, model="fmn")
    
    # pre-declaring
    init_fmn = nothing
    out_fmn = nothing

    # Test initialization doesn't crash
    @test_nowarn init_fmn = CausalMixtures.dpm_init(raw_data_test, priors_test, params_fmn)
    @test_nowarn out_fmn = CausalMixtures.dpm!(init_fmn...)
    
    # Test output structure is reasonable
    state, input, output = out_fmn
    @test length(output.out_dp) == params_fmn.M
    @test length(output.out_theta) == params_fmn.M
end
    
@testset "Blocked Sampler Smoke Test" begin
    priors_test = CausalMixtures.setup_default_priors(data_test.true_params.ktot)
    raw_data_test = CausalMixtures.RawData(data_test.formulas.formula_y, 
                                           data_test.formulas.formula_d, data_test.df)
    params_blocked = CausalMixtures.InputParams(M=5, verbose=false, model="blocked")
    
    # pre-declare variables to avoid scoping issues
    init_blocked = nothing
    out_blocked = nothing

    # Test initialization doesn't crash
    @test_nowarn init_blocked = CausalMixtures.dpm_init(raw_data_test, priors_test, params_blocked)
    
    # Test sampling doesn't crash (very short run)
    @test_nowarn out_blocked = CausalMixtures.dpm!(init_blocked...)
    
    # Test output structure is reasonable
    state, input, output = out_blocked
    @test length(output.out_dp) == params_blocked.M
    @test length(output.out_theta) == params_blocked.M
end
    
@testset "DPM Sampler Smoke Test" begin
    priors_test = CausalMixtures.setup_default_priors(data_test.true_params.ktot)
    raw_data_test = CausalMixtures.RawData(data_test.formulas.formula_y, 
                                           data_test.formulas.formula_d, data_test.df)
    params_dpm = CausalMixtures.InputParams(M=5, verbose=false, model="dpm")
    
    # pre-declare
    init_dpm = nothing
    out_dpm = nothing

    # Test initialization doesn't crash
    @test_nowarn init_dpm = CausalMixtures.dpm_init(raw_data_test, priors_test, params_dpm)
    @test_nowarn out_dpm = CausalMixtures.dpm!(init_dpm...)
    
    # Test output structure is reasonable
    state, input, output = out_dpm #CausalMixtures.dpm!(init_dpm...)
    @test length(output.out_dp) == params_dpm.M
    @test length(output.out_theta) == params_dpm.M
end
    
@testset "PPD and ATE Smoke Test" begin
    # Use blocked sampler for this test
    priors_test = CausalMixtures.setup_default_priors(data_test.true_params.ktot)
    raw_data_test = CausalMixtures.RawData(data_test.formulas.formula_y, 
                                           data_test.formulas.formula_d, data_test.df)
    params_test = CausalMixtures.InputParams(M=10, verbose=false, model="blocked")
    
    init_test = nothing
    out_test = nothing
    
    @test_nowarn init_test = CausalMixtures.dpm_init(raw_data_test, priors_test, params_test)
    @test_nowarn out_test = CausalMixtures.dpm!(init_test...)
    
    state, input, output = out_test
    
    # Test PPD generation doesn't crash AND get the result
    znew_test = mean(input.data.Hmat[1:100, 1:3], dims=1)'
    @test_nowarn CausalMixtures.rand_ppd(output, input, znew_test[:,1])  # ← Test it doesn't crash
    ynew_test = CausalMixtures.rand_ppd(output, input, znew_test[:,1])   # ← Get actual result
    
    # Test ATE calculation doesn't crash AND get the result  
    @test_nowarn CausalMixtures.dpm_ate(ynew_test, input)  # ← Test it doesn't crash
    tes_test = CausalMixtures.dpm_ate(ynew_test, input)    # ← Get actual result
    
    # Test ATE structure
    @test length(tes_test.ate) == params_test.M
    @test !any(isnan.(tes_test.ate))
    @test !any(isinf.(tes_test.ate))
end
