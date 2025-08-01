export dpm_init, dpm_chain!, dpm_dump!, dpm!

## initialize, chain, dump, etc.

## function to select sampler
function set_sampler(model::String)

    if model == "dpm"
        f = dpm_gibbs!
    elseif model == "blocked"
        f = dpm_blocked!
    elseif model == "fmn"
        f = dpm_fmn!
    elseif model == "gaussian"
        f = dpm_gaussian!
    elseif model == "marginal"
        f = dpm_marginal!
    else
        throw(DomainError(model, "chosen model does not match known models"))
    end

    return f

end

## initialize sampler
function dpm_init(data::RawData, priors::InputPriors, params::InputParams; xmats_out=false)
    
    ## Inputs:
    ## data (data type object): DataFrame and Formula objects
    ## prior (prior type object): prior hyperparameters
    ## param (param type object): sampler parameters
    
    ## Output:
    ## GibbsState : initial sampler state
    ## GibbsInput : fixed objects (data, params, priors)
    ## GibbsOut : pre-allocated storage
    
    M = params.M
    scale_data = params.scale_data
    verbose = params.verbose
    
    if verbose  println("M = $(M)\nscale data = $(scale_data)\nverbose = $(verbose)") end
    
    ## 1. transform raw data -> model matrix
    ##    * Optional: scale data    
    y = convert(Array, data.df[!, data.y_form.lhs.sym])
    d = convert(Array, data.df[!, data.d_form.lhs.sym])
    
    xmat = ModelMatrix( ModelFrame(data.y_form, data.df) ).m
    if verbose println("Xmat dim:\nN = ", size(xmat, 1), ", K = ", size(xmat, 2)) end
    
    zmat = ModelMatrix( ModelFrame(data.d_form, data.df) ).m
    if verbose println("Zmat dim:\nN = ", size(zmat, 1), ", K = ", size(zmat, 2)) end
    
    ## get dimensions
    n = size(y, 1)
    kx = size(xmat, 2)
    kz = size(zmat, 2)
    ktot = 2*kx + kz
    
    dims = InputDims(n=n, kx=kx, kz=kz, ktot=ktot)
    
    ## scale response?
    if scale_data[1]
        if verbose println("Scaling response...") end
        ys = standardize(Float64.(y))
    else
        ys = ScaleData(a=y)
    end
    ## scale inputs?
    if scale_data[2]
        if verbose println("Scaling inputs...") end
        xmats = standardize(xmat)
        zmats = standardize(zmat)
    else
        xmats = ScaleData(a=xmat, m=zeros(kx), s=ones(kx))
        zmats = ScaleData(a=zmat, m=zeros(kz), s=ones(kz))
    end
    
    ## construct block-diag H
    Hmat = blockdiag(sparse(zmats.a), sparse(xmats.a), sparse(xmats.a))
    if verbose println("Hmat dim:\nN = ", size(Hmat, 1), ", K = ", ktot) end
    
    ## truncated normal support for latent data
    lower = [ di == 1 ?  0 : -Inf for di in d ]
    upper = [ di == 1 ? Inf : 0 for di in d ]
    
    ## collect data objects
    input_data = InputData(y=ys, d=d, lower=lower, upper=upper, Hmat=Hmat)

    model = params.model
    
    ## collect all inputs
    if priors.prior_theta.prior_beta.Vinv
        ## convert to covariance
        V = inv(priors.prior_theta.prior_beta.V)
    else
        ## save covariance and convert to precision
        V = priors.prior_theta.prior_beta.V
        ##prior_beta = PriorBeta(priors.prior_theta.prior_beta.mu, V\eye(ktot), true)
        priors = InputPriors(priors.prior_dp,
                             PriorTheta(PriorBeta(mu=priors.prior_theta.prior_beta.mu, V=inv(V), Vinv=true),
                                        priors.prior_Sigma))
    end

    # ADDED: Auto-fix J=1 for gaussian sampler
    if model == "gaussian" && priors.prior_dp.J != 1
        @warn "Gaussian sampler requires J=1 (single component). Automatically setting J=1."
        # Create new priors with J=1
        new_prior_dp = PriorDP(alpha=priors.prior_dp.alpha, J=1, 
                               alpha_shape=priors.prior_dp.alpha_shape, 
                               alpha_rate=priors.prior_dp.alpha_rate)
        priors = InputPriors(prior_dp=new_prior_dp, prior_theta=priors.prior_theta)
    end
    
    input = GibbsInput(data=input_data, dims=dims, params=params, priors=priors)
    
    ## 2. initialize sampler state
    if verbose println("Initializing state...") end  
    
    ## 2a. initialize latent data:
    ## sample selection outcome
    dstar = zeros(n)
    @inbounds for i in 1:n
        dstar[i] = rand( TruncatedNormal(0, 1, lower[i], upper[i]) )
    end
    ## initialize potential outcomes using observed data
    state_data = StateData(dstar=dstar, y1=copy(ys.a), y0=copy(ys.a))
    
    ## 2b. initialize DP:
    ## assign component membership uniformly
    labels = Dict{Int64,Int64}()
    if priors.prior_dp.J != 1        
        for i in 1:n
            labels[i] = rand(1:priors.prior_dp.J)
        end
    else
        for i in 1:n
            labels[i] = 1
        end
    end
    ## count component memberships
    njs = StatsBase.countmap(collect(values(labels)))
    ## compute stick-breaking weights?
    ws = DataStructures.OrderedDict{Int64,BlockedWeights}()
    ## collect DP hyperparameters
    ## initialize alpha to prior alpha, eta = 0
    state_dp = StateDP(J=priors.prior_dp.J, labels=labels, njs=njs, ws=ws, alpha=priors.prior_dp.alpha, eta=0.0)    
    if input.params.model == "blocked"
        ## initialize weights
        for k in sort(collect(keys(state_dp.njs))) get!(state_dp.ws, k, BlockedWeights()) end
        state_dp = compute_weights!(state_dp)
    elseif input.params.model == "fmn"
        state_dp.alpha = state_dp.alpha / float(state_dp.J)
    end
        
    ## 2c. initialize theta:
    ## sample theta from prior
    state_theta = Dict{Int64,Theta}()
    for j in 1:priors.prior_dp.J
        ##beta_j = priors.prior_theta.prior_beta.mu + chol(priors.prior_theta.prior_beta.V)'*randn(ktot)
        ##Sigma_j = NobileWishart(priors.prior_theta.prior_Sigma.rho,
        ##                        priors.prior_theta.prior_Sigma.rho * priors.prior_theta.prior_Sigma.R )
        ##state_theta[j] = Theta(beta=beta_j, Sigma=Sigma_j)
        state = sample_prior_theta!(state_theta, priors.prior_theta, j)
    end
    
    ## 2d. collect sampler state parameters
    ## pre-compute constant factors for sampler
    Vmu = *(priors.prior_theta.prior_beta.V, priors.prior_theta.prior_beta.mu)
    zdenom = priors.prior_dp.alpha + n - 1
    ## set the initial sampler state
    ##state_sampler = StateSampler(); state_sampler.Vmu=Vmu; state_sampler.zdenom=zdenom
    state_sampler = StateSampler(chain=false, batch_n=1, batch_m=0, batch_1=1, Vmu=Vmu, zdenom=zdenom)
    
    ## 2e. collect all state variables
    state = GibbsState(state_data=state_data, state_dp=state_dp,
                       state_theta=state_theta, state_sampler=state_sampler)
    
    ## 3. pre-allocate output
    out = GibbsOut(M)
    
    if verbose println("Ready!") end
    
    if xmats_out
        return (state, input, out, xmats, zmats)
    else
        return (state, input, out)
    end
    
end

## append new output to old
function dpm_chain!(state::GibbsState, input::GibbsInput, out::GibbsOut; test=false)
    ## 1. pre-allocate output
    ##out = GibbsOut( [out, Array(StateTheta, input.params.M) ] )    
    dpm_sampler = set_sampler(input.params.model)
    ## pre-allocate storage for new chain
    out_new = GibbsOut(input.params.M)
    ## 2. run sampler
    println("Continuing from last state...")
    state, input, out_new = dpm_sampler( state, input, out_new, test=test )
    ## append new output to old output
    ##out.out_M += out_new.out_M
    append!(out.out_data, out_new.out_data)
    out_data = 0
    append!(out.out_dp, out_new.out_dp)
    out_dp = 0
    append!(out.out_theta, out_new.out_theta)
    out_new = 0    
    return (state, input, out)
end

## wrapper to call sampler
function dpm!(state::GibbsState, input::GibbsInput, out::GibbsOut=GibbsOut(0); test=false)
    if ( state.state_sampler.chain && length(out.out_data) > 0 )
        ## append output
        out_tup = dpm_chain!(state, input, out, test = test)
    else
        ## no out given?
        if length(out.out_data) == 0 ## || state.state_sampler.chain == false )
            out = GibbsOut(input.params.M)
        end
        ## run new chain
        dpm_sampler = set_sampler(input.params.model)
        out_tup = dpm_sampler(state, input, out, test=test)
    end    
    return out_tup    
end

## return to previous state from iteration output
## TODO: how to recover state.state_sampler?
function get_state(out::GibbsOut)
    state = GibbsState(state_data = out.out_data[end],
                       state_dp = out.out_dp[end],
                       state_theta = out.out_theta[end])
    return state
end

## reset state to iteration m
function reset_state!(out::GibbsOut, state::GibbsState, m::Int64)
    state.state_data = out.out_data[m]
    state.state_dp = out.out_dp[m]
    state.state_theta = out.out_theta[m]
    return state
end


## calls:
## 1a. collect raw data, priors, params, setting prior_beta.V = inv(V)
## 1b. open connection for saving? o = open(...)
## 2. call init:
##    state, input, out = dpm_init(raw_data, priors, params)
## 3. save {state, input}?
## 
## 4. call sampler:
##    state, input, out = dpm((state, input, out)...)
##    out_tup = dpm_gibbs(dpm_init(raw_data, priors, params)...)
## 5. save {out}?
##
## 6. write to disk and dump?
##

## dump output to disk and continue chain
## new: JLD2 version
## TODO: Check for consistency with chain_dpm.jl
function dpm_dump!(state::GibbsState, input::GibbsInput, out::GibbsOut;
                   fname::String="out", dir::String="./")
    old_m = state.state_sampler.batch_m
    out_m = state.state_sampler.batch_m
    out_1 = old_m - input.params.M + 1
    filename = dir * fname * ".jld2"
    if isfile(filename)
        ## 1. append to existing file
        if input.params.verbose println("Updating existing file...") end
        
        # Load existing data
        existing_data = load(filename)
        
        # Update state
        existing_data["state"] = state
        
        # Add new output batch
        batch_key = "out-$(out_1):$(out_m)"
        existing_data[batch_key] = out
        
        # Save updated data
        save(filename, existing_data)
        
    else
        ## 2. create new file
        if input.params.verbose println("Creating new file $filename...") end
        
        batch_key = "out-$(out_1):$(out_m)"
        save_data = Dict(
            "state" => state,
            "input" => input,
            batch_key => out
        )
        save(filename, save_data)
    end
    ## 3. run sampler with empty out
    out = GibbsOut(input.params.M)
    gc()
    state.state_sampler.batch_n += 1
    state.state_sampler.batch_1 += state.state_sampler.batch_m # increment starting point
    dpm_sampler = set_sampler(input.params.model)
    if input.params.verbose println("Continuing from last state...") end
    state, input, out = dpm_sampler( state, input, out)
    return (state, input, out)
end

## new: load saved results from disk
function dpm_load(fname::String="out", dir::String="./")
    filename = dir * fname * ".jld2"
    
    if !isfile(filename)
        error("File $filename does not exist!")
    end
    
    data = load(filename)
    
    # Extract core objects
    state = data["state"]
    input = data["input"]
    
    # Find all output batches
    output_keys = filter(k -> startswith(string(k), "out-"), keys(data))
    
    if length(output_keys) == 1
        # Single batch
        out = data[first(output_keys)]
        return (state, input, out)
    else
        # Multiple batches - return as dictionary
        outputs = Dict()
        for key in output_keys
            outputs[string(key)] = data[key]
        end
        return (state, input, outputs)
    end
end

## new: convenience function to load just the final state
function dpm_load_state(fname::String="out", dir::String="./")
    filename = dir * fname * ".jld2"
    
    if !isfile(filename)
        error("File $filename does not exist!")
    end
    
    return load(filename, "state")
end

## new: convenience function to combine multiple output batches
function dpm_combine_outputs(outputs::Dict)
    if length(outputs) == 1
        return first(values(outputs))
    end
    
    # Sort by batch number and combine
    sorted_keys = sort(collect(keys(outputs)))
    combined_out = outputs[sorted_keys[1]]
    
    for key in sorted_keys[2:end]
        batch_out = outputs[key]
        append!(combined_out.out_data, batch_out.out_data)
        append!(combined_out.out_dp, batch_out.out_dp)
        append!(combined_out.out_theta, batch_out.out_theta)
    end
    
    return combined_out
end

    
## pre-define functions in sampler?
##update_labels!(state::GibbsState) = update_labels!(state::GibbsState, input)
##update_params!(state::GibbsState) = update_params!(state::GibbsState, input)
##if input.priors.prior_dp.alpha_shape != 0.0 update_alpha!(state::GibbsState) = update_alpha!(state::GibbsState, input) end
