## blocked sampler for DPM

function dpm_blocked!(state::GibbsState, input::GibbsInput, out::GibbsOut)
    
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
        if verbose && mod(M, m) == 0 
            @printf("\nIteration: %d\nUpdating labels...", m + state.state_sampler.batch_m )
        end
        
        state = update_blocked_labels!(state, input)
        
        ## 2. update theta and latent data
        if verbose && mod(M, m) == 0
            @printf("\nActive J = %d", state.state_dp.J)
            @printf("\nEmpty components = %d", input.priors.prior_dp.J - state.state_dp.J)
            @printf("\nUpdating component parameters and latent data...")
        end
        
        state = update_blocked_params!(state, input)
        
        ## 3. update alpha?
        if input.priors.prior_dp.alpha_shape != 0.0
            if verbose && mod(M, m) == 0 @printf("\nUpdating alpha...") end
            
            ## sample alpha
            state = update_blocked_alpha!(state, input)
            if verbose && mod(M, m) == 0 @printf("\nCurrent alpha = %f", state.state_dp.alpha) end            
        end
        
        ## 4. save iteration m draws
        if verbose && mod(M, m) == 0 @printf("\nDone!") end
        
        out = update_out!(state, out, m)
        ##out[m] = state
        
    end
    
    if verbose @printf("\nSampler run complete.\n") end

    ## update sampler state
    state = update_sampler!(state, M)
    
    return (state, input, out)
    
end
