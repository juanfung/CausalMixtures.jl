## improved sampler for DPM
export dpm_gibbs!

function dpm_gibbs!(state::GibbsState, input::GibbsInput, out::GibbsOut; test=false)
    
    ## sampler parameters
    M = input.params.M
    verbose = input.params.verbose
    
    ## begin the sampler
    if verbose
        @printf("Batch %d\nInitial J = %d\nInitial alpha = %f\nBegin sampler...",
                state.state_sampler.batch_n, state.state_dp.J, state.state_dp.alpha )
    end
    
    @inbounds for m in 1:M
        
        ## 1. update labels
        if verbose && mod(m, max(1, M/10)) == 0
            @printf("\nIteration: %d\nUpdating labels...", m + state.state_sampler.batch_m )
        end
        
        state = update_labels!(state, input)
        if verbose && mod(m, max(1, M/10)) == 0 @printf("\nCurrent J = %d", state.state_dp.J) end
        
        ## 2. update theta and latent data
        if verbose && mod(m, max(1, M/10)) == 0 @printf("\nUpdating component parameters and latent data...") end

        ## test using observed data as latent data?
        if test
            state = update_theta_only!(state, input)
        else
            state = update_params!(state, input)
        end
        
        ## 3. update alpha?
        if input.priors.prior_dp.alpha_shape != 0.0
            if verbose && mod(m, max(1, M/10)) == 0 @printf("\nUpdating alpha...") end
            
            ## sample alpha
            state = update_alpha!(state, input)
            if verbose && mod(m, max(1, M/10)) == 0 @printf("\nCurrent alpha = %f", state.state_dp.alpha) end            
        end
        
        ## 4. save iteration m draws
        if verbose && mod(m, max(1, M/10)) == 0 @printf("\nDone!") end
        
        out = update_out!(state, out, m)
        ##out[m] = state
        
    end
    
    if verbose @printf("\nSampler run complete.\n") end

    ## update sampler state
    state = update_sampler!(state, M)
    
    return (state, input, out)
    
end
