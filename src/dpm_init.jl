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
function dpm_init(data::RawData, priors::InputPriors, params::InputParams; xmats_out=false,
                  rng::AbstractRNG=Random.default_rng())
    
    ## Inputs:
    ## data (data type object): DataFrame and Formula objects
    ## prior (prior type object): prior hyperparameters
    ## param (param type object): sampler parameters
    ## rng: random number generator for the initial state (pass the same one to dpm!)
    
    ## Output:
    ## GibbsState : initial sampler state
    ## GibbsInput : fixed objects (data, params, priors)
    ## GibbsOut : pre-allocated storage
    
    ## fail fast on an unknown model (throws DomainError)
    set_sampler(params.model)
    
    M = params.M
    scale_data = params.scale_data
    verbose = params.verbose
    
    if verbose  println("M = $(M)\nscale data = $(scale_data)\nverbose = $(verbose)") end
    
    ## 1. transform raw data -> model matrix
    ##    * Optional: scale data    
    y = Vector{Float64}(data.df[!, data.y_form.lhs.sym])
    d = Vector{Int64}(data.df[!, data.d_form.lhs.sym])
    
    xframe = ModelFrame(data.y_form, data.df)
    xmat = ModelMatrix(xframe).m
    if verbose println("Xmat dim:\nN = ", size(xmat, 1), ", K = ", size(xmat, 2)) end
    
    zframe = ModelFrame(data.d_form, data.df)
    zmat = ModelMatrix(zframe).m
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
    Hmat = blockdiag(sparse(zmats.a), sparse(xmats.a), sparse(xmats.a))
    if verbose println("Hmat dim:\nN = ", size(Hmat, 1), ", K = ", ktot) end
    
    ## truncated normal support for latent data
    lower = [ di == 1 ?  0 : -Inf for di in d ]
    upper = [ di == 1 ? Inf : 0 for di in d ]
    
    ## collect data objects
    input_data = InputData(y=ys, d=d, lower=lower, upper=upper, Hmat=Hmat,
                           X=Matrix{Float64}(xmats.a), W=Matrix{Float64}(zmats.a),
                           xnames=string.(coefnames(xframe)), znames=string.(coefnames(zframe)))

    model = params.model
    if model == "marginal" && priors.slopes != :unspecified && priors.slopes != params.slopes
        @warn "the priors were built for slopes = :$(priors.slopes) but the sampler runs slopes = " *
              ":$(params.slopes); use setup_default_priors(...; slopes = :$(params.slopes))"
    end
    
    ## collect all inputs
    ## prior_beta.V is the precision from here on; prior_beta.Vcov and .Lcov hold the covariance V0
    if !priors.prior_theta.prior_beta.Vinv
        ## keep the given covariance and its factor, convert V to precision
        pb = priors.prior_theta.prior_beta
        priors = InputPriors(priors.prior_dp,
                             PriorTheta(PriorBeta(pb.mu, inv(pb.Vcov), true, pb.Vcov, pb.Lcov),
                                        priors.prior_theta.prior_Sigma), priors.slopes)
    end

    # ADDED: Auto-fix J=1 for gaussian sampler
    if model == "gaussian" && priors.prior_dp.J != 1
        @warn "Gaussian sampler requires J=1 (single component). Automatically setting J=1."
        # Create new priors with J=1
        new_prior_dp = PriorDP(alpha=priors.prior_dp.alpha, J=1, 
                               alpha_shape=priors.prior_dp.alpha_shape, 
                               alpha_rate=priors.prior_dp.alpha_rate)
        priors = InputPriors(prior_dp=new_prior_dp, prior_theta=priors.prior_theta, slopes=priors.slopes)
    end
    
    input = GibbsInput(data=input_data, dims=dims, params=params, priors=priors)
    
    ## 2. initialize sampler state
    if verbose println("Initializing state...") end  
    
    ## 2a. initialize latent data:
    ## sample selection outcome
    dstar = zeros(n)
    @inbounds for i in 1:n
        dstar[i] = rand(rng, truncated(Normal(0, 1), lower[i], upper[i]) )
    end
    ## initialize potential outcomes using observed data
    state_data = StateData(dstar=dstar, y1=copy(ys.a), y0=copy(ys.a))
    
    ## 2b. initialize DP:
    ## assign component membership uniformly
    J = priors.prior_dp.J
    labels = J != 1 ? [rand(rng, 1:J) for i in 1:n] : ones(Int64, n)
    if model in ("dpm", "marginal")
        ## the Polya-urn and Design A samplers keep only occupied components (J = number of clusters),
        ## so relabel the occupied ones as 1:k
        occupied = sort(unique(labels))
        relabel = zeros(Int64, J)
        relabel[occupied] = 1:length(occupied)
        labels = relabel[labels]
        J = length(occupied)
    end
    ## count component memberships, with a (possibly zero) count for every component 1:J
    njs = zeros(Int64, J)
    for l in labels
        njs[l] += 1
    end
    ## collect DP hyperparameters
    ## initialize alpha to prior alpha, eta = 0
    state_dp = StateDP(J=J, labels=labels, njs=njs, ws=BlockedWeights[], alpha=priors.prior_dp.alpha, eta=0.0)
    rebuild_members!(state_dp)
    if input.params.model == "blocked"
        ## initialize weights
        state_dp.ws = fill(BlockedWeights(), J)
        state_dp = compute_weights!(rng, state_dp)
    elseif input.params.model == "fmn"
        state_dp.alpha = state_dp.alpha / float(state_dp.J)
    end
        
    ## 2c. initialize theta:
    ## sample theta from prior
    state_theta = StateTheta()
    for j in 1:state_dp.J
        ##beta_j = priors.prior_theta.prior_beta.mu + chol(priors.prior_theta.prior_beta.V)'*randn(ktot)
        ##Sigma_j = NobileWishart(priors.prior_theta.prior_Sigma.rho,
        ##                        priors.prior_theta.prior_Sigma.rho * priors.prior_theta.prior_Sigma.R )
        ##state_theta[j] = Theta(beta=beta_j, Sigma=Sigma_j)
        sample_prior_theta!(rng, state_theta, priors.prior_theta, j)
    end
    if model == "marginal"
        ## Design A: one shared b (the slopes of component 1's prior draw) under :common
        D = DesignA(input)
        b = state_theta[1].beta[D.gidx]
        for j in 1:state_dp.J
            state_theta[j] = Theta(beta=full_beta(D, state_theta[j].beta[D.cidx], b), Sigma=state_theta[j].Sigma)
        end
        state_hyper = D.hier ? StateHyper(m=copy(D.mu_m), tau2=D.tau_scale ./ (D.tau_shape - 1)) : StateHyper()
    else
        state_hyper = StateHyper()
    end
    ## the learned Sigma scale starts at the priors' r (Phase 6a')
    if params.sigma_scale == :hier
        model in ("marginal", "gaussian") || throw(ArgumentError("sigma_scale = :hier is implemented for the marginal and gaussian samplers"))
        Rp = priors.prior_theta.prior_Sigma.R
        maximum(abs, Rp - Diagonal(diag(Rp))) == 0 || throw(ArgumentError("sigma_scale = :hier needs a diagonal R"))
        state_hyper.r = [Rp[2, 2], Rp[3, 3]]
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
                       state_theta=state_theta, state_sampler=state_sampler, state_hyper=state_hyper)
    
    ## 3. pre-allocate output
    out = GibbsOut(params)
    
    if verbose println("Ready!") end
    
    if xmats_out
        return (state, input, out, xmats, zmats)
    else
        return (state, input, out)
    end
    
end

## append new output to old
function dpm_chain!(state::GibbsState, input::GibbsInput, out::GibbsOut; test=false,
                    rng::AbstractRNG=Random.default_rng())
    ## 1. pre-allocate output
    ##out = GibbsOut( [out, Array(StateTheta, input.params.M) ] )    
    dpm_sampler = set_sampler(input.params.model)
    ## pre-allocate storage for new chain
    out_new = GibbsOut(input.params)
    ## 2. run sampler
    if input.params.verbose println("Continuing from last state...") end
    state, input, out_new = dpm_sampler( state, input, out_new, test=test, rng=rng )
    ## append new output to old output
    ##out.out_M += out_new.out_M
    append!(out.out_data, out_new.out_data)
    append!(out.out_dp, out_new.out_dp)
    append!(out.out_theta, out_new.out_theta)
    append!(out.out_labels, out_new.out_labels)
    append!(out.out_hyper, out_new.out_hyper)
    append!(out.out_sep, out_new.out_sep)
    append!(out.out_r, out_new.out_r)
    return (state, input, out)
end

## wrapper to call sampler
## rng: the random number generator for every draw (default: the global RNG)
function dpm!(state::GibbsState, input::GibbsInput, out::GibbsOut=GibbsOut(0); test=false,
              rng::AbstractRNG=Random.default_rng())
    if ( state.state_sampler.chain && length(out.out_dp) > 0 )
        ## append output
        out_tup = dpm_chain!(state, input, out, test = test, rng = rng)
    else
        ## new chain: (re)size the output for M iterations
        prepare_out!(out, input.params)
        ## run new chain
        dpm_sampler = set_sampler(input.params.model)
        out_tup = dpm_sampler(state, input, out, test=test, rng=rng)
    end
    input.params.model == "marginal" && warn_separated(out_tup[3])
    return out_tup
end

## warn when the posterior mean of the separated-units share (units in clusters under 5% or
## over 95% treated) exceeds `level`
function warn_separated(out::GibbsOut; level::Real=0.05)
    sep = filter(isfinite, out.out_sep)
    isempty(sep) && return nothing
    s = mean(sep)
    s > level && @warn @sprintf("%.1f%% of units sit, on average, in clusters that are under 5%% or over 95%% treated: their treated or control outcome is identified from little data (separated-units diagnostic, out.out_sep)", 100s)
    return nothing
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
                   fname::String="out", dir::String="./", rng::AbstractRNG=Random.default_rng())
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
    out = GibbsOut(input.params)
    GC.gc()
    state.state_sampler.batch_n += 1
    state.state_sampler.batch_1 += state.state_sampler.batch_m # increment starting point
    dpm_sampler = set_sampler(input.params.model)
    if input.params.verbose println("Continuing from last state...") end
    state, input, out = dpm_sampler( state, input, out, rng=rng)
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
        append!(combined_out.out_labels, batch_out.out_labels)
    end
    
    return combined_out
end

    
## pre-define functions in sampler?
##update_labels!(state::GibbsState) = update_labels!(state::GibbsState, input)
##update_params!(state::GibbsState) = update_params!(state::GibbsState, input)
##if input.priors.prior_dp.alpha_shape != 0.0 update_alpha!(state::GibbsState) = update_alpha!(state::GibbsState, input) end
