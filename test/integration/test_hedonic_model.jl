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
        
        @test abs(data.true_effects.ate) < 20
        @test data.design_info.n == 100
        @test data.design_info.n_treated + data.design_info.n_control == 100
    end
    
    @testset "Blocked Sampler Integration" begin
        # Test blocked sampler specifically
        data_test = CausalMixtures.generate_hedonic_data(200, 1234)
        priors_test = CausalMixtures.setup_default_priors(2, 3)
        raw_data_test = CausalMixtures.RawData(data_test.formulas.formula_y, 
                                             data_test.formulas.formula_d, data_test.df)
        
        params_test = CausalMixtures.InputParams(M=50, scale_data=(true,true), 
                                               verbose=false, model="blocked")
        
        @test_nowarn jit_init_test = CausalMixtures.dpm_init(raw_data_test, priors_test, params_test)
        
        jit_init_fresh = CausalMixtures.dpm_init(raw_data_test, priors_test, params_test)
        out_test = nothing
        @test_nowarn out_test = CausalMixtures.dpm!(jit_init_fresh...)
        
        state_test, input_test, output_test = out_test
        @test length(output_test.out_dp) == 50
        @test length(output_test.out_theta) == 50
        
        # Test PPD computation
        znew_test = CausalMixtures.mean_covariates(input_test)  # (xbar, wbar) over all n
        ynew_test = nothing
        @test_nowarn ynew_test = CausalMixtures.rand_ppd(output_test, input_test, znew_test...)
        
        tes_test = nothing
        @test_nowarn tes_test = CausalMixtures.dpm_ate(ynew_test, input_test)
        
        # Test treatment effects are reasonable
        @test length(tes_test.ate) == 50
        @test abs(mean(tes_test.ate) - data_test.true_effects.ate) < 8.0
        @test !any(isnan.(tes_test.ate))
        @test !any(isinf.(tes_test.ate))
        
        println("blocked sampler ATE: $(round(mean(tes_test.ate), digits=3))")
    end
    
    @testset "DPM Sampler Integration" begin
        # Test dpm sampler specifically  
        data_test2 = CausalMixtures.generate_hedonic_data(200, 5678)
        priors_test2 = CausalMixtures.setup_default_priors(2, 3)
        raw_data_test2 = CausalMixtures.RawData(data_test2.formulas.formula_y, 
                                              data_test2.formulas.formula_d, data_test2.df)
        
        params_test2 = CausalMixtures.InputParams(M=50, scale_data=(true,true), 
                                                verbose=false, model="dpm")
        
        @test_nowarn jit_init_test2 = CausalMixtures.dpm_init(raw_data_test2, priors_test2, params_test2)
        
        jit_init_fresh2 = CausalMixtures.dpm_init(raw_data_test2, priors_test2, params_test2)
        out_test2 = nothing
        @test_nowarn out_test2 = CausalMixtures.dpm!(jit_init_fresh2...)
        
        state_test2, input_test2, output_test2 = out_test2
        @test length(output_test2.out_dp) == 50
        @test length(output_test2.out_theta) == 50
        
        # Test PPD computation
        znew_test2 = CausalMixtures.mean_covariates(input_test2)  # (xbar, wbar) over all n
        ynew_test2 = nothing
        @test_nowarn ynew_test2 = CausalMixtures.rand_ppd(output_test2, input_test2, znew_test2...)
        
        tes_test2 = nothing
        @test_nowarn tes_test2 = CausalMixtures.dpm_ate(ynew_test2, input_test2)
        
        # Test treatment effects are reasonable
        @test length(tes_test2.ate) == 50
        @test abs(mean(tes_test2.ate) - data_test2.true_effects.ate) < 8.0
        @test !any(isnan.(tes_test2.ate))
        @test !any(isinf.(tes_test2.ate))
        
        println("dpm sampler ATE: $(round(mean(tes_test2.ate), digits=3))")
    end
    
    @testset "FMN Sampler Integration" begin
        # Test fmn sampler specifically  
        data_test3 = CausalMixtures.generate_hedonic_data(200, 9999)
        priors_test3 = CausalMixtures.setup_default_priors(2, 3)
        raw_data_test3 = CausalMixtures.RawData(data_test3.formulas.formula_y, 
                                              data_test3.formulas.formula_d, data_test3.df)
        
        params_test3 = CausalMixtures.InputParams(M=50, scale_data=(true,true), 
                                                verbose=false, model="fmn")
        
        @test_nowarn jit_init_test3 = CausalMixtures.dpm_init(raw_data_test3, priors_test3, params_test3)
        
        jit_init_fresh3 = CausalMixtures.dpm_init(raw_data_test3, priors_test3, params_test3)
        out_test3 = nothing
        @test_nowarn out_test3 = CausalMixtures.dpm!(jit_init_fresh3...)
        
        state_test3, input_test3, output_test3 = out_test3
        @test length(output_test3.out_dp) == 50
        @test length(output_test3.out_theta) == 50
        
        # Test PPD computation
        znew_test3 = CausalMixtures.mean_covariates(input_test3)  # (xbar, wbar) over all n
        ynew_test3 = nothing
        @test_nowarn ynew_test3 = CausalMixtures.rand_ppd(output_test3, input_test3, znew_test3...)
        
        tes_test3 = nothing
        @test_nowarn tes_test3 = CausalMixtures.dpm_ate(ynew_test3, input_test3)
        
        # Test treatment effects are reasonable
        @test length(tes_test3.ate) == 50
        @test abs(mean(tes_test3.ate) - data_test3.true_effects.ate) < 8.0
        @test !any(isnan.(tes_test3.ate))
        @test !any(isinf.(tes_test3.ate))
        
        println("fmn sampler ATE: $(round(mean(tes_test3.ate), digits=3))")
    end
    
    @testset "Gaussian Sampler Integration" begin
        # Test gaussian sampler specifically  
        data_test4 = CausalMixtures.generate_hedonic_data(200, 7777)
        ## SET J = 1 manually!
        priors_test4 = CausalMixtures.setup_default_priors(2, 3, J=1)
        raw_data_test4 = CausalMixtures.RawData(data_test4.formulas.formula_y, 
                                                data_test4.formulas.formula_d, data_test4.df)
        params_test4 = CausalMixtures.InputParams(M=50, scale_data=(true,true), 
                                                  verbose=false, model="gaussian")
        
        @test_nowarn jit_init_test4 = CausalMixtures.dpm_init(raw_data_test4, priors_test4, params_test4)
        
        jit_init_fresh4 = CausalMixtures.dpm_init(raw_data_test4, priors_test4, params_test4)
        out_test4 = nothing
        @test_nowarn out_test4 = CausalMixtures.dpm!(jit_init_fresh4...)
        
        state_test4, input_test4, output_test4 = out_test4
        @test length(output_test4.out_dp) == 50
        @test length(output_test4.out_theta) == 50
        
        # Test PPD computation
        znew_test4 = CausalMixtures.mean_covariates(input_test4)  # (xbar, wbar) over all n
        ynew_test4 = nothing
        @test_nowarn ynew_test4 = CausalMixtures.rand_ppd(output_test4, input_test4, znew_test4...)
        
        tes_test4 = nothing
        @test_nowarn tes_test4 = CausalMixtures.dpm_ate(ynew_test4, input_test4)
        
        # Test treatment effects are reasonable
        @test length(tes_test4.ate) == 50
        @test abs(mean(tes_test4.ate) - data_test4.true_effects.ate) < 8.0
        @test !any(isnan.(tes_test4.ate))
        @test !any(isinf.(tes_test4.ate))
        
        println("gaussian sampler ATE: $(round(mean(tes_test4.ate), digits=3))")
    end
    
    @testset "Multi-Sampler Consistency Comparison" begin
        # Compare all working samplers on same data
        data_comp = CausalMixtures.generate_hedonic_data(150, 42)
        priors_comp = CausalMixtures.setup_default_priors(2, 3)
        raw_data_comp = CausalMixtures.RawData(data_comp.formulas.formula_y, 
                                             data_comp.formulas.formula_d, data_comp.df)
        
        # Test all four working samplers
        samplers = ["blocked", "dpm", "fmn", "gaussian"]
        results = Dict()
        
        for sampler in samplers
            if sampler == "gaussian"
                priors_sampler = CausalMixtures.setup_default_priors(2, 3, J=1)
            else
                priors_sampler = priors_comp
            end
            params = CausalMixtures.InputParams(M=30, verbose=false, model=sampler)
            init = CausalMixtures.dpm_init(raw_data_comp, priors_sampler, params)
            out = CausalMixtures.dpm!(init...)
            
            # Compute treatment effects
            znew_comp = CausalMixtures.mean_covariates(out[2])  # (xbar, wbar) over all n
            ynew = CausalMixtures.rand_ppd(out[3], out[2], znew_comp...)
            tes = CausalMixtures.dpm_ate(ynew, out[2])
            
            results[sampler] = mean(tes.ate)
            
            # Test each sampler gives reasonable results
            @test abs(results[sampler] - data_comp.true_effects.ate) < 6.0
        end
        
        # Test that all samplers are somewhat consistent with each other
        ate_values = collect(values(results))
        @test maximum(ate_values) - minimum(ate_values) < 8.0  # Range shouldn't be too wide
        
        println("Sampler ATE comparison:")
        for (sampler, ate) in results
            println("  $(sampler): $(round(ate, digits=3))")
        end
    end
end
