## initialize, chain, dump, etc.

## function to select sampler
function set_sampler(model::ASCIIString)

    if model == "dpm"
        f = dpm_gibbs!
    elseif model == "blocked"
        f = dpm_blocked!
    elseif model == "fmn"
        f =  dpm_fmn!
    elseif model == "gaussian"
        f = dpm_gaussian!
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
    y = convert(Array, data.df[data.y_form.lhs])
    d = convert(Array, data.df[data.d_form.lhs])
    
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
        ys = standardize(y)
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
    Hmat = blkdiag(sparse(zmats.a), sparse(xmats.a), sparse(xmats.a))
    if verbose println("Hmat dim:\nN = ", size(Hmat, 1), ", K = ", ktot) end
    
    ## truncated normal support for latent data
    lower = ifelse( d .== 1, 0, -Inf )
    upper = ifelse( d .== 1, Inf, 0 )
    
    ## collect data objects
    input_data = InputData(y=ys, d=d, lower=lower, upper=upper, Hmat=Hmat)
    
    ## collect all inputs
    if priors.prior_theta.prior_beta.Vinv
        ## convert to covariance
        V = priors.prior_theta.prior_beta.V\eye(ktot)
    else
        ## save covariance and convert to precision
        V = priors.prior_theta.prior_beta.V
        ##prior_beta = PriorBeta(priors.prior_theta.prior_beta.mu, V\eye(ktot), true)
        priors = InputPriors(priors.prior_dp,
                             PriorTheta(PriorBeta(mu=priors.prior_theta.prior_beta.mu, V=V\eye(ktot), Vinv=true),
                                        priors.prior_Sigma))
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
function dpm_chain!(state::GibbsState, input::GibbsInput, out::GibbsOut)
    ## 1. pre-allocate output
    ##out = GibbsOut( [out, Array(StateTheta, input.params.M) ] )    
    dpm_sampler = set_sampler(input.params.model)
    ## pre-allocate storage for new chain
    out_new = GibbsOut(input.params.M)
    ## 2. run sampler
    println("Continuing from last state...")
    state, input, out_new = dpm_sampler( state, input, out_new )
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

## dump output to disk and continue chain
## NB: Assumes fname exists!
## TODO: Re-write for consistency with chain_dpm.jl
function dpm_dump!(state::GibbsState, input::GibbsInput, out::GibbsOut;
                   fname::ASCIIString="out", dir::ASCIIString="./")
    old_m = state.state_sampler.batch_m
    ##batch_n = state.state_sampler.batch_n
    ##batch_1 = state.state_sampler.batch_1
    out_m = state.state_sampler.batch_m
    out_1 = batch_m - input.params.M + 1
    if in( fname * ".jld", readdir(dir) ) # fname * ".jld" in readdir(dir) # 
        ## 1. write current output to disk using JLD-HDF5        
        if input.params.verbose println("Dumping old output...") end        
        JLD.jldopen(dir * fname * ".jld", "r+", compress=true) do file
            delete!(file, "state")
            write(file, "state", state)
            write(file, "output/out-$(out_1):$(out_m)", out) # add sims
        end        
    else
        ##error("$(fname) does not exist in $(dir)!")
        if input.params.verbose println("Creating new file " * dir * fname * ".jld ...") end
        ## TODO: create and write file, saving state and input
        JLD.jldopen(dir * fname * ".jld", "w", compress=true) do file
            addrequire(file, CausalMixtures)
            write(file, "state", state) # save state 
            write(file, "input", input) # save inputs
            g = g_create(file, "output")
            g["out-$(out_1):$(out_m)"] = out # save first run output
        end
    end
    ## 2. run sampler with empty out
    out = GibbsOut(input.params.M)
    gc()
    state.state_sampler.batch_n += 1
    state.state_sampler.batch_1 += state.state_sampler.batch_m # increment starting point
    dpm_sampler = set_sampler(input.params.model)
    if input.params.verbose println("Continuing from last state...") end
    state, input, out = dpm_sampler( state, input, out )
    return (state, input, out)
end

## wrapper to call sampler
function dpm!(state::GibbsState, input::GibbsInput, out::GibbsOut=GibbsOut(0))
    if ( state.state_sampler.chain && length(out.out_data) > 0 )
        ## append output
        out_tup = dpm_chain!(state, input, out)
    else
        ## no out given?
        if length(out.out_data) == 0 ## || state.state_sampler.chain == false )
            out = GibbsOut(input.params.M)
        end
        ## run new chain
        dpm_sampler = set_sampler(input.params.model)
        out_tup = dpm_sampler(state, input, out)
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


    
## pre-define functions in sampler?
##update_labels!(state::GibbsState) = update_labels!(state::GibbsState, input)
##update_params!(state::GibbsState) = update_params!(state::GibbsState, input)
##if input.priors.prior_dp.alpha_shape != 0.0 update_alpha!(state::GibbsState) = update_alpha!(state::GibbsState, input) end
