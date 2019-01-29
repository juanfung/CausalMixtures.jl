## benchmark Gaussian model
export dpm_gaussian!

function dpm_gaussian!(state::GibbsState, input::GibbsInput, out::GibbsOut; test=false)
    
    ## sampler parameters
    M = input.params.M
    verbose = input.params.verbose    
    
    ## begin the sampler
    if verbose
        @printf("Batch %d\nInitial J = %d\nInitial alpha = %f\nBegin sampler...",
                state.state_sampler.batch_n, state.state_dp.J, state.state_dp.alpha )
    end
    
    @inbounds for m in 1:M
        
        ## 1. update theta and latent data
        if verbose && mod(M, m) == 0
            @printf("\nIteration: %d\nUpdating parameters and latent data...", m + state.state_sampler.batch_m )
        end
        
        state = update_gaussian_params!(state, input)
        
        ## 2. save iteration m draws
        if verbose && mod(M, m) == 0 @printf("\nDone!") end
        
        out = update_out!(state, out, m)
        
    end
    
    if verbose @printf("\nSampler run complete.\n") end

    ## update sampler state
    state = update_sampler!(state, M)
    
    return (state, input, out)
    
end
