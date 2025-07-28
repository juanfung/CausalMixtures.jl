# test/integration/test_hedonic_model.jl

using CausalMixtures
using Test, Statistics, Random

@testset "Hedonic Model Integration Tests" begin
    
    @testset "Data Generation" begin
        @test_nowarn data = CausalMixtures.generate_hedonic_data(100, 2099)
        
        data = CausalMixtures.generate_hedonic_data(100, 2099)
        
        # Basic data structure tests
        @test size(data.df, 1) == 100
        @test "Y_obs" in names(data.df)
        @test "D_obs" in names(data.df)
        @test "X" in names(data.df)
        @test "Z" in names(data.df)
        
        @test abs(data.true_effects.ate_mean) < 20
        @test data.design_info.n == 100
        @test data.design_info.n_treated + data.design_info.n_control == 100
    end
    
    @testset "Blocked Sampler Integration" begin
        # Test blocked sampler specifically
        data_test = CausalMixtures.generate_hedonic_data(200, 1234)
        priors_test = CausalMixtures.setup_default_priors(data_test.true_params.ktot, 
                                                         beta_nu=100.0, rho=6.0, r=2.0)
        raw_data_test = CausalMixtures.RawData(data_test.formulas.formula_y, 
                                             data_test.formulas.formula_d, data_test.df)
        
        params_test = CausalMixtures.InputParams(M=50, scale_data=(true,true), 
                                               verbose=false, model="blocked")
        
        @test_nowarn jit_init_test = CausalMixtures.dpm_init(raw_data_test, priors_test, params_test)
        
        # FIXED: Define variable outside @test_nowarn, then assign inside
        jit_init_fresh = CausalMixtures.dpm_init(raw_data_test, priors_test, params_test)
        out_test = nothing  # ← Pre-declare variable
        @test_nowarn out_test = CausalMixtures.dpm!(jit_init_fresh...)  # ← Assign inside
        
        state_test, input_test, output_test = out_test  # ← Now accessible
        @test length(output_test.out_dp) == 50
        @test length(output_test.out_theta) == 50
        
        # Test PPD computation
        znew_test = mean(input_test.data.Hmat[1:100, 1:3], dims=1)'
        ynew_test = nothing  # ← Pre-declare
        @test_nowarn ynew_test = CausalMixtures.rand_ppd(output_test, input_test, znew_test[:,1])
        
        tes_test = nothing  # ← Pre-declare
        @test_nowarn tes_test = CausalMixtures.dpm_ate(ynew_test, input_test)
        
        # Test treatment effects are reasonable
        @test length(tes_test.ate) == 50
        @test abs(mean(tes_test.ate) - data_test.true_effects.ate_mean) < 8.0
        @test !any(isnan.(tes_test.ate))
        @test !any(isinf.(tes_test.ate))
        
        println("blocked sampler ATE: $(round(mean(tes_test.ate), digits=3))")
    end
    
    @testset "DPM Sampler Integration" begin
        # Test dpm sampler specifically  
        data_test2 = CausalMixtures.generate_hedonic_data(200, 5678)
        priors_test2 = CausalMixtures.setup_default_priors(data_test2.true_params.ktot, 
                                                          beta_nu=100.0, rho=6.0, r=2.0)
        raw_data_test2 = CausalMixtures.RawData(data_test2.formulas.formula_y, 
                                              data_test2.formulas.formula_d, data_test2.df)
        
        params_test2 = CausalMixtures.InputParams(M=50, scale_data=(true,true), 
                                                verbose=false, model="dpm")
        
        @test_nowarn jit_init_test2 = CausalMixtures.dpm_init(raw_data_test2, priors_test2, params_test2)
        
        # FIXED: Pre-declare variables
        jit_init_fresh2 = CausalMixtures.dpm_init(raw_data_test2, priors_test2, params_test2)
        out_test2 = nothing
        @test_nowarn out_test2 = CausalMixtures.dpm!(jit_init_fresh2...)
        
        state_test2, input_test2, output_test2 = out_test2
        @test length(output_test2.out_dp) == 50
        @test length(output_test2.out_theta) == 50
        
        # Test PPD computation
        znew_test2 = mean(input_test2.data.Hmat[1:100, 1:3], dims=1)'
        ynew_test2 = nothing
        @test_nowarn ynew_test2 = CausalMixtures.rand_ppd(output_test2, input_test2, znew_test2[:,1])
        
        tes_test2 = nothing
        @test_nowarn tes_test2 = CausalMixtures.dpm_ate(ynew_test2, input_test2)
        
        # Test treatment effects are reasonable
        @test length(tes_test2.ate) == 50
        @test abs(mean(tes_test2.ate) - data_test2.true_effects.ate_mean) < 8.0
        @test !any(isnan.(tes_test2.ate))
        @test !any(isinf.(tes_test2.ate))
        
        println("dpm sampler ATE: $(round(mean(tes_test2.ate), digits=3))")
    end
    
    @testset "Sampler Consistency Comparison" begin
        # Compare blocked vs dpm on same data
        data_comp = CausalMixtures.generate_hedonic_data(150, 42)
        priors_comp = CausalMixtures.setup_default_priors(data_comp.true_params.ktot)
        raw_data_comp = CausalMixtures.RawData(data_comp.formulas.formula_y, 
                                             data_comp.formulas.formula_d, data_comp.df)
        
        # Test blocked sampler
        params_blocked = CausalMixtures.InputParams(M=30, verbose=false, model="blocked")
        init_blocked = CausalMixtures.dpm_init(raw_data_comp, priors_comp, params_blocked)
        out_blocked = CausalMixtures.dpm!(init_blocked...)
        
        # Test dpm sampler  
        params_dpm = CausalMixtures.InputParams(M=30, verbose=false, model="dpm")
        init_dpm = CausalMixtures.dpm_init(raw_data_comp, priors_comp, params_dpm)
        out_dpm = CausalMixtures.dpm!(init_dpm...)
        
        # Compute treatment effects
        znew_comp = mean(out_blocked[2].data.Hmat[1:100, 1:3], dims=1)'
        
        ynew_blocked = CausalMixtures.rand_ppd(out_blocked[3], out_blocked[2], znew_comp[:,1])
        tes_blocked = CausalMixtures.dpm_ate(ynew_blocked, out_blocked[2])
        
        ynew_dpm = CausalMixtures.rand_ppd(out_dpm[3], out_dpm[2], znew_comp[:,1])
        tes_dpm = CausalMixtures.dpm_ate(ynew_dpm, out_dpm[2])
        
        # Test reasonable results and consistency
        @test abs(mean(tes_blocked.ate) - data_comp.true_effects.ate_mean) < 6.0
        @test abs(mean(tes_dpm.ate) - data_comp.true_effects.ate_mean) < 6.0
        @test abs(mean(tes_blocked.ate) - mean(tes_dpm.ate)) < 4.0
    end
end
