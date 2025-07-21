## marginal sampler for DPM
export dpm_marginal!

function dpm_marginal!(state::GibbsState, input::GibbsInput, out::GibbsOut; 
                      test=false, marginal_type="finite")
    
    ## sampler parameters
    M = input.params.M
    verbose = input.params.verbose
    
    ## choose marginal approach
    if marginal_type == "finite"
        update_labels_fn! = update_marginal_labels_finite!
        update_params_fn! = update_marginal_params_finite!
        update_alpha_fn! = update_marginal_alpha_finite!
    elseif marginal_type == "infinite" 
        update_labels_fn! = update_marginal_labels_infinite!
        update_params_fn! = update_marginal_params_infinite!
        update_alpha_fn! = update_marginal_alpha_infinite!
    else
        error("marginal_type must be 'finite' or 'infinite'")
    end
    
    ## begin the sampler
    if verbose
        @printf("Batch %d (Marginal: %s)\nInitial J = %d\nInitial alpha = %f\nBegin sampler...",
                state.state_sampler.batch_n, marginal_type, state.state_dp.J, state.state_dp.alpha)
    end
    
    @inbounds for m in 1:M
        
        ## 1. update labels
        if verbose && mod(m, max(1, M/10)) == 0 
            @printf("\nIteration: %d\nUpdating labels...", m + state.state_sampler.batch_m)
        end
        
        state = update_labels_fn!(state, input)
        
        ## 2. update theta and latent data
        if verbose && mod(m, max(1, M/10)) == 0
            @printf("\nActive J = %d", state.state_dp.J)
            if marginal_type == "finite"
                @printf("\nEmpty components = %d", input.priors.prior_dp.J - state.state_dp.J)
            end
            @printf("\nUpdating component parameters and latent data...")
        end

        ## test using observed data as latent data?
        if test
            state = update_marginal_theta_only!(state, input)
        else
            state = update_params_fn!(state, input)
        end        
        
        ## 3. update alpha?
        if input.priors.prior_dp.alpha_shape != 0.0
            if verbose && mod(m, max(1, M/10)) == 0 @printf("\nUpdating alpha...") end
            
            state = update_alpha_fn!(state, input)
            if verbose && mod(m, max(1, M/10)) == 0 @printf("\nCurrent alpha = %f", state.state_dp.alpha) end            
        end
        
        ## 4. save iteration m draws
        if verbose && mod(m, max(1, M/10)) == 0 @printf("\nDone!") end
        
        out = update_out!(state, out, m)
        
    end
    
    if verbose @printf("\nSampler run complete.\n") end

    ## update sampler state
    state = update_sampler!(state, M)
    
    return (state, input, out)
end
