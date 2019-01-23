## functions to update GibbsState

## --------------------------------------------------------------------------- #
## main sampler functions

## update component labels and number of components
function update_labels!(state::GibbsState, input::GibbsInput)
    ## update component labels for each i
    @inbounds for i in 1:input.dims.n
        ji = copy( state.state_dp.labels[i] ) # i's current component        
        state.state_dp.labels[i] = 0 # remove i from component ji
        state.state_dp.njs[ji] = state.state_dp.njs[ji] - 1 # decrease ji count
        ## remove empty component!
        if state.state_dp.njs[ji] == 0
            state = remove_component!(state, ji)
        end
        ## sample component label
        state = sample_label!(state, input, i)
    end
    return state
end

## remove empty component and relabel components
function remove_component!(state::GibbsState, ji::Int64)    
    ## 1. shift labels
    state.state_dp = shift_labels!(state.state_dp, ji)    
    ## 2. update counts    
    state.state_dp = shift_njs!(state.state_dp, ji)    
    ## 3. shift components
    state.state_theta = shift_theta!(state.state_theta, ji)    
    ## 4. shift J
    state.state_dp.J -= 1
    return state
end

function shift_labels!(state_dp::StateDP, ji::Int64)
    ##for ii in collect(keys(state_dp.labels))[collect(values(state_dp.labels)) .> ji]
    @inbounds for ii in keys(state_dp.labels)
        if state_dp.labels[ii] > ji
            state_dp.labels[ii] -= 1
        end        
    end    
    return state_dp
end

function shift_njs!(state_dp::StateDP, ji::Int64)
    @inbounds for j in ji:(length(state_dp.njs) - 1)
        state_dp.njs[j] = state_dp.njs[j+1]
    end
    pop!(state_dp.njs, length(state_dp.njs))
    return state_dp
end

function shift_theta!(state_theta::StateTheta, ji::Int64)
    @inbounds for j in ji:(length(state_theta) - 1)
        state_theta[j] = state_theta[j+1]
    end
    pop!(state_theta, length(state_theta))
    return state_theta
end

## sample new theta from prior predictive
function sample_new_theta(prior_theta::PriorTheta, state::GibbsState,
                          Hi::SparseMatrixCSC{Float64,Int64}, yi::Vector{Float64})
    ## i. sample new Sigma
    #Ui = yi - Hi*prior_theta.prior_beta.mu # 3 x 1
    #Ui = Ui*Ui' + prior_theta.prior_Sigma.rho*prior_theta.prior_Sigma.R # 3 x 3    
    #Sigma_j = NobileWishart(prior_theta.prior_Sigma.rho+1, Ui)
    ## ii. sample new beta
    #xb = *(Hi', Sigma_j\eye(3)) # ktot x 3
    #vj = xb*Hi + prior_theta.prior_beta.V # ktot x ktot
    #vj = vj\eye(vj)
    #mj = *( vj, xb*yi + state.state_sampler.Vmu ) # ktot x 1
    #beta_j = mj + chol(vj)'*randn(length(mj)) # ktot x 1
    Sigma_j = sample_new_Sigma(prior_theta, Hi, yi)
    beta_j = sample_new_beta(new_beta_cov(prior_theta, Hi, yi, Sigma_j)..., state.state_sampler.Vmu, yi)
    return Theta(beta=beta_j, Sigma=Sigma_j)
end

function sample_new_Sigma(p::PriorTheta, Hi::SparseMatrixCSC{Float64,Int64}, yi::Vector{Float64})
    ## i. sample new Sigma
    Ui = yi - Hi*p.prior_beta.mu # 3 x 1
    Ui = Ui*Ui' + p.prior_Sigma.rho*p.prior_Sigma.R # 3 x 3
    ##if !isposdef(Ui) Ui += eps(Float64)*eye(3) end
    ##if !isposdef(Ui) error("Ui not postive definite!")
    Sigma_j = NobileWishart(p.prior_Sigma.rho+1, Ui)
    return Sigma_j
end

function new_beta_cov(p::PriorTheta, Hi::SparseMatrixCSC{Float64,Int64}, yi::Vector{Float64}, S::Matrix{Float64})
    xb = *(Hi', inv(S)) # ktot x 3
    vj = xb*Hi + p.prior_beta.V # ktot x ktot
    return ( xb, inv(vj) )
end

function sample_new_beta(xb::Matrix{Float64}, vj::Matrix{Float64}, Vmu::Vector{Float64}, yi::Vector{Float64})
    mj = *( vj, xb*yi + Vmu ) # ktot x 1 
    return mj + cholesky( Hermitian(vj) ).U'*randn(length(mj)) # ktot x 1
end

## compute probability of theta
function prob_theta(theta::Theta, Hi::SparseMatrixCSC{Float64,Int64}, yi::Vector{Float64})
    w = logpdf( MvNormal( Hi*theta.beta, theta.Sigma), yi)
end

## sample label for i
function sample_label!(state::GibbsState, input::GibbsInput, i::Int64)
    
    ## Prob(label i = existing component)
    Hi = input.data.Hmat[vcat(i, i+input.dims.n, i+2*input.dims.n), :] # 3 x ktot
    yi = [state.state_data.dstar[i], state.state_data.y1[i], state.state_data.y0[i]] # 3 x 1
    w = Array{Float64}(undef, state.state_dp.J)
    @inbounds for j in 1:state.state_dp.J
        w[j] = state.state_dp.njs[j] * exp( prob_theta(state.state_theta[j], Hi, yi) ) / state.state_sampler.zdenom
    end
    
    ## Prob(label i = new component)
    theta_j = sample_new_theta(input.priors.prior_theta, state, Hi, yi)
    w_new = state.state_dp.alpha * exp( prob_theta(theta_j, Hi, yi) ) / state.state_sampler.zdenom    
    
    ## combine and normalize probabilities
    push!(w, w_new)
    rmul!(w, 1/sum(w))
    
    ## sample component
    ji = rand( Distributions.Categorical(w) )
    state.state_dp.labels[i] = ji
    ## setindex!(state.state_dp.labels, ji, i)
    
    ## update J
    if ji == state.state_dp.J + 1
        state.state_dp.J += 1
        ##state.state_dp.njs[ji] = 0
        get!(state.state_dp.njs, ji, 0)
        ##state.state_theta[ji] = theta_j
        get!(state.state_theta, ji, theta_j)
    end
    
    ## update counts
    state.state_dp.njs[ji] +=  1

    return state

end

## update parameters {LatentData, Theta}
function update_params!(state::GibbsState, input::GibbsInput)

    @inbounds for j in 1:state.state_dp.J # k in keys(state.state_theta) or keys(state.state_dp.njs)
        
        if state.state_dp.njs[j] == 0 continue end
        
        idx = sort( collect( keys( ( filter( (k,v) -> v == j, state.state_dp.labels ) ) ) ) )
        Hj = input.data.Hmat[vcat(idx, idx .+ input.dims.n, idx .+ 2*input.dims.n), :] # 3nj x ktot
        
        ## update theta
        state = update_theta!(state, input, j, idx, Hj)
        
        ## update latent data        
        state = update_latent!(state, input, j, idx, Hj)
        
    end
    
    return state
    
end

function update_theta!(state::GibbsState, input::GibbsInput, j::Int64, idx::Vector{Int64},
                       Hj::SparseMatrixCSC{Float64,Int64}) ##T<:AbstractArray{Int64,1} UnitRange{Int64}

    nj = state.state_dp.njs[j]
    
    dstarj = view(state.state_data.dstar, idx) # nj x 1
    y1j = view(state.state_data.y1, idx) # nj x 1
    y0j = view(state.state_data.y0, idx) # nj x 1
    
    yij = vcat(dstarj, y1j, y0j) # 3nj x 1
    ##Hj = input.data.Hmat[vcat(idx, idx+input.dims.n, idx+2*input.dims.n),:] # 3nj x ktot

    ## 1. update Sigma
    function sample_Sigma( prior_theta::PriorTheta, theta::Theta )
        ##, Hj::SparseMatrixCSC{Float64,Int64}, yij::Vector{Float64})
        ##nj = div(length(yij),3)
        ##Hb = Hj*state_theta.beta # 3nj x 1
        Uj = yij - Hj*theta.beta # 3nj x 1
        if length(Uj) != 3nj error("Dimension mismatch! Oops") end
        Uj = reshape(Uj, nj, 3) # nj x 3
        Uj = Uj'*Uj + prior_theta.prior_Sigma.rho*prior_theta.prior_Sigma.R # 3 x 3        
        Sigma_j = NobileWishart( (nj + prior_theta.prior_Sigma.rho), Uj )
        return Sigma_j
    end
    
    Sigma_j = sample_Sigma(input.priors.prior_theta, state.state_theta[j])
    
    ## 2. update beta
    function sample_beta( prior_theta::PriorTheta, state::GibbsState )
        sigXi = kron( inv(Sigma_j), sparse(1.0I, nj, nj) ) # 3nj x 3nj
        xb = Hj'*sigXi # ktot x 3nj
        vj = xb*Hj + prior_theta.prior_beta.V # ktot x ktot
        vj = inv(vj) # ktot x ktot
        mj = *( vj, xb*yij + state.state_sampler.Vmu ) # ktot x 1
        beta_j = mj + cholesky( Hermitian(vj) ).U'*randn(length(mj))
        return beta_j
    end
    
    beta_j = sample_beta(input.priors.prior_theta, state)
    
    state.state_theta[j] = Theta(beta=beta_j, Sigma=Sigma_j)
    
    return state

end

function update_latent!(state::GibbsState, input::GibbsInput, j::Int64, idx::Vector{Int64},
                        Hj::SparseMatrixCSC{Float64,Int64})

    nj = state.state_dp.njs[j]
    
    dstarj = view(state.state_data.dstar, idx) # nj x 1
    
    yj = view(input.data.y.a, idx) # nj x 1
    dj = view(input.data.d, idx) # nj x 1
    
    ##Hb = input.data.Hmat[vcat(idx, idx+input.dims.n, idx+2*input.dims.n),:] * state.state_theta[j].beta # 3nj x 1
    Hb = Hj * state.state_theta[j].beta
    zbD = view(Hb, 1:nj) # nj x 1
    xb1 = view(Hb, (nj+1):(2nj)) # nj x 1
    xb0 = view(Hb, (2nj+1):(3nj)) # nj x 1
    
    ## pre-compute constants
    mid = (state.state_theta[j].Sigma[2,3])*(state.state_theta[j].Sigma[1,3])*(state.state_theta[j].Sigma[1,2])
    
    ## 1. update missing outcomes -> y1, y0
    function sample_ymiss(Sigma::Matrix{Float64})
        ## compute denominators
        denom1 = Sigma[2,2] - (Sigma[1,2]^2)
        denom0 = Sigma[3,3] - (Sigma[1,3]^2)
        ## compute potential outcome means
        mu1 = xb1 +
        (dstarj - zbD) * ( ( (Sigma[3,3] * Sigma[1,2]) - (Sigma[2,3] * Sigma[1,3]) ) / denom0 ) +
        (yj - xb0) * ( ( Sigma[2,3] - (Sigma[1,3] * Sigma[1,2]) ) / denom0 )
        mu0 = xb0 +
        (dstarj - zbD) * ( ( (Sigma[2,2] * Sigma[1,3]) - (Sigma[2,3] * Sigma[1,2]) ) / denom1 ) +
        (yj - xb1) * ( ( Sigma[2,3] - (Sigma[1,3] * Sigma[1,2]) ) / denom1 )        
        ## compute potential outcome variances
        omega1 = Sigma[2,2] - ( ( (Sigma[1,2]^2 * Sigma[3,3]) - 2*mid + (Sigma[2,3]^2) ) / denom0 )
        omega0 = Sigma[3,3] - ( ( (Sigma[1,3]^2 * Sigma[2,2]) - 2*mid + (Sigma[2,3]^2) ) / denom1 )
        ## compute missing outcome mean and variance
        mu_miss = (1 .- dj).*mu1 + dj.*mu0
        var_miss = (1 .- dj)*omega1 + dj*omega0
        ## compute missing outcome
        ymiss_j = mu_miss + sqrt.(var_miss) .* randn(nj)
        return ymiss_j
    end
    
    ymiss_j = sample_ymiss(state.state_theta[j].Sigma)
    
    ## update potential outcomes
    y1j = dj.*yj + (1 .- dj).*ymiss_j
    y0j = dj.*ymiss_j + (1 .- dj).*yj
    
    ## 2. update selection outcome
    function update_dstar!(Sigma::Matrix{Float64})
        ## compute denominator and other factors
        denomd = (Sigma[2,2] * Sigma[3,3]) - (Sigma[2,3]^2 )
        sd1 = ( ( (Sigma[3,3] * Sigma[1,2]) - (Sigma[2,3] * Sigma[1,3]) ) / denomd )
        sd0 = ( ( (Sigma[2,2] * Sigma[1,3]) - (Sigma[2,3] * Sigma[1,2]) ) / denomd )
        ## compute mean
        mu_d = zbD + (y1j - xb1)*sd1 + (y0j - xb0)*sd0
        ## compute variance
        omega_d = 1 - ( ( (Sigma[1,2]^2 * Sigma[3,3]) - 2*mid + (Sigma[1,3]^2 * Sigma[2,2]) ) / denomd )        
        @inbounds for i in 1:nj
            dstarj[i] = rand( Distributions.TruncatedNormal(mu_d[i], sqrt(omega_d),
                                                            input.data.lower[idx[i]], input.data.upper[idx[i]]) )
        end
        return dstarj
    end
    
    dstarj = update_dstar!(state.state_theta[j].Sigma)
    
    setindex!(state.state_data.dstar, dstarj, idx)
    setindex!(state.state_data.y1, y1j, idx)
    setindex!(state.state_data.y0, y0j, idx)
    
    return state
    
end

## update concentration parameter
function update_alpha!(state::GibbsState, input::GibbsInput)

    ## sample auxiliary variable
    state.state_dp.eta = rand( Distributions.Beta(state.state_dp.alpha + 1, input.dims.n) )
    
    ## update shape and rate parameters
    b_star = input.priors.prior_dp.alpha_rate - log(state.state_dp.eta)
    a_star = input.priors.prior_dp.alpha_shape + state.state_dp.J
    A = (input.priors.prior_dp.alpha_shape + state.state_dp.J - 1) / (input.dims.n * b_star)
    ##pi_eta = A/(1 + A)
    a_star = A/(1 + A) > 0.5 ? a_star : a_star - 1
    
    ## sample alpha
    state.state_dp.alpha = rand( Distributions.Gamma(a_star, 1/b_star) )
    
    ## update zdenom
    state.state_sampler.zdenom = state.state_dp.alpha + input.dims.n - 1

    return state

end

## update output
function update_out!(state::GibbsState, out::GibbsOut, m::Int64)
    out.out_data[m] = copy(state.state_data)
    out.out_dp[m] = copy(state.state_dp)
    out.out_theta[m] = copy(state.state_theta)
    ## push!(out.out_theta, state.state_theta)
    return out
end

## update sampler state after a run
function update_sampler!(state::GibbsState, M::Int64)
    state.state_sampler.chain = true
    ##state.state_sampler.batch_1 += state.state_sampler.batch_m
    state.state_sampler.batch_m += M
    return state
end

## update sampler state after data dump

## --------------------------------------------------------------------------- #
## specialized functions for Gaussian model

function update_gaussian_params!(state::GibbsState, input::GibbsInput)
    idx = collect(1:input.dims.n)
    ## update theta
    state = update_theta!(state, input, 1, idx, input.data.Hmat)
    ## update latent data        
    state = update_latent!(state, input, 1, idx, input.data.Hmat)
    return state
end    

## --------------------------------------------------------------------------- #
## specialized functions for FMN

## update labels for fmn
function update_fmn_labels!(state::GibbsState, input::GibbsInput)
    ## update component labels for each i
    @inbounds for i in 1:input.dims.n
        ji = state.state_dp.labels[i] # i's current component
        if ji == 0 error("Null label! Oops") end
        state.state_dp.njs[ji] = state.state_dp.njs[ji] - 1 # decrease ji count
        if state.state_dp.njs[ji] < 0 error("Negative label count! Oops") end
        ## sample component label
        state = sample_fmn_label!(state, input, i)
    end
    return state
end

## sample fmn label for i
function sample_fmn_label!(state::GibbsState, input::GibbsInput, i::Int64)    
    Hi = input.data.Hmat[vcat(i, i+input.dims.n, i+2*input.dims.n), :] # 3 x ktot
    yi = [state.state_data.dstar[i], state.state_data.y1[i], state.state_data.y0[i]] # 3 x 1
    w = Array(Float64, state.state_dp.J)
    @inbounds for j in 1:state.state_dp.J        
        w[j] = ( state.state_dp.njs[j] + state.state_dp.alpha ) *
        exp( prob_theta(state.state_theta[j], Hi, yi) ) / state.state_sampler.zdenom
    end    
    rmul!(w, 1/sum(w))    
    ## sample component
    ji = rand( Distributions.Categorical(w) )
    state.state_dp.labels[i] = ji    
    ## update counts
    state.state_dp.njs[ji] +=  1
    return state
end


## --------------------------------------------------------------------------- #
## specialized functions for blocked sampler

## update stick-breaking weights, v.1
function update_weights!(state::GibbsState, input::GibbsInput)
    ##for j in sort(collect(keys(filter( (k,v) -> k < state.state_dp.J, state.state_dp.njs ))))
    v, w = ones(state.state_dp.J), zeros(state.state_dp.J)
    for j in 1:state.state_dp.J # for j in sort(collect(keys(state.state_dp.njs)))
        if j < state.state_dp.J
            njj = sum( values( filter( (k,v) -> k > j, state.state_dp.njs) ) )
            ##njj = reduce(+, drop( values(state.state_dp.njs), j ) ).nj
            ##state.state_dp.njs[j].v =
            v[j] = rand( Distributions.Beta( 1 + state.state_dp.njs[j], state.state_dp.alpha + njj ) )
        end
        w[j] = v[j]*prod(1 - v[1:j-1])
        ##state.state_dp.njs[j].w = state.state_dp.njs[j].v*prod( map( v -> 1 - v, take(values(state.state_dp.njs), j-1) ) ).v
        setindex!(state.state_dp.weights, BlockedWeights(w[j], v[j]), j)
    end
    return state
end

## another approach
function compute_v(s::StateDP, j::Int64)
    if j < length(s.njs)
        njj = sum( values( filter( (k,v) -> k > j, s.njs) ) )
        v = rand( Distributions.Beta( 1 + s.njs[j], s.alpha + njj))
    else
        v = 1.0
    end
    return v
end

function compute_w(s::StateDP, j::Int64)
    w = (j == 1) ? s.ws[j].v : s.ws[j].v * prod( map( val -> 1-val, take(values(s.ws), j-1) ) ).v
    ##s.ws[j].w = j == 1? s.ws[j].v: s.ws[j].v * prod( map( val -> 1-val.v, take(values(s.ws), j-1) ) )
    ## reduce(*, map(v->1-v, take(valus(d), j-1))).v
    return w
end

function update_ws!(s::StateDP, j::Int64)
    get!(s.ws, j, BlockedWeights())
    s.ws[j].v = compute_v(s, j)
    s.ws[j].w = compute_w(s, j)
    return s
end

## update stick-breaking weights, v.2
function compute_weights!(state_dp::StateDP)
    for j in 1:length(state_dp.njs)
        state_dp = update_ws!(state_dp, j)
    end
    return state_dp
end

function update_blocked_labels!(state::GibbsState, input::GibbsInput)
    for i in 1:input.dims.n
        state = sample_blocked_label!(state, input, i)
    end

    state.state_dp = compute_weights!(state.state_dp)
    state.state_dp = update_J!(state.state_dp)
    return state
end

function sample_blocked_label!(state::GibbsState, input::GibbsInput, i::Int64)
    Hi = input.data.Hmat[vcat(i, i+input.dims.n, i+2*input.dims.n), :]
    yi = [state.state_data.dstar[i], state.state_data.y1[i], state.state_data.y0[i]]
    w = Array(Float64, input.priors.prior_dp.J)
    # remove i from current component
    state.state_dp.njs[state.state_dp.labels[i]] -= 1
    ## 1. compute Pr(label i = j)
    @inbounds for j in 1:length(state.state_dp.njs)
        w[j] = state.state_dp.ws[j].w * exp( prob_theta(state.state_theta[j], Hi, yi) ) #/ state.state_sampler.zdenom
    end
    rmul!(w, 1/sum(w))
    ## 2. sample ji
    ji = rand( Distributions.Categorical(w) )
    state.state_dp.labels[i] = ji
    ## update counts
    state.state_dp.njs[ji] += 1
    return state
end

## montitor number of active components
function update_J!(s::StateDP)
    s.J = length( filter( (k, val) -> val != 0, s.njs) )
    return s
end

## sample theta from prior
function sample_prior_theta!(state_theta::StateTheta, prior_theta::PriorTheta, j::Int64)
    beta_j = prior_theta.prior_beta.mu + cholesky(prior_theta.prior_beta.V).U'*randn(length(prior_theta.prior_beta.mu))
    Sigma_j = NobileWishart(prior_theta.prior_Sigma.rho, prior_theta.prior_Sigma.rho * prior_theta.prior_Sigma.R )
    state_theta[j] = Theta(beta=beta_j, Sigma=Sigma_j)
    return state_theta
end

## update parameters {LatentData, Theta}
function update_blocked_params!(state::GibbsState, input::GibbsInput)
    #for k in keys(state.state_dp.ws) or keys(state.state_dp.njs)
    @inbounds for j in 1:length(input.priors.prior_dp.J) 
        if state.state_dp.njs[j] == 0
            ## sample from prior
            state.state_theta = sample_prior_theta!(state.state_theta, input.priors.prior_theta, j)
        else
            ## sample from posterior
            idx = sort( collect( keys( ( filter( (k,v) -> v == j, state.state_dp.labels ) ) ) ) )
            Hj = input.data.Hmat[vcat(idx, idx+input.dims.n, idx+2*input.dims.n),:] # 3nj x ktot        
            ## update theta
            state = update_theta!(state, input, j, idx, Hj)
            ## update latent data        
            state = update_latent!(state, input, j, idx, Hj)
        end
    end
    return state
end

## update concentration parameter
function fv(d::OrderedDict{Int64,BlockedWeights}; f::Function=log) #, j::Int64; f::Function=identity)
    return reduce(+, map(val -> f(1-val.v), take(values(d), length(d)-1)))
    ##return map(v -> f(1-v), take(values(d), j-1) )
end

function update_blocked_alpha!(state::GibbsState, input::GibbsInput)
    a_star = input.priors.prior_dp.alpha_shape + input.priors.prior_dp.J - 1
    ##v = reduce(+, map(val -> log(1-val), take(values(state.state_dp.ws), state.state_dp.J-1))).v
    b_star = input.priors.prior_dp.alpha_rate - fv(state.state_dp.ws)
    ##b_star = input.priors.prior_dp.alpha_rate-reduce(+,map_v(state.state_dp.njs,state.state_dp.J,f=v->log(1-v))).v
    state.state_dp.alpha = rand( Distributions.Gamma(a_star, 1/b_star) )
    state.state_sampler.zdenom = state.state_dp.alpha + input.dims.n - 1
    return state
end


## --------------------------------------------------------------------------- #
## test functions: dpm_gibbs

## update latent data, given true theta
function update_latent_only!(state::GibbsState, input::GibbsInput)
    @inbounds for j in 1:state.state_dp.J # k in keys(state.state_theta) or keys(state.state_dp.njs)        
        if state.state_dp.njs[j] == 0 continue end        
        idx = sort( collect( keys( ( filter( (k,v) -> v == j, state.state_dp.labels ) ) ) ) )
        ##Hj = input.data.Hmat[vcat(idx, idx+input.dims.n, idx+2*input.dims.n),:] # 3nj x ktot
        ## update latent data
        state = update_ymiss_only!(state, input, j, idx)
    end    
    return state    
end

## update theta, given true latent data
function update_theta_only!(state::GibbsState, input::GibbsInput)
    @inbounds for j in 1:state.state_dp.J # k in keys(state.state_theta) or keys(state.state_dp.njs)        
        if state.state_dp.njs[j] == 0 continue end        
        idx = sort( collect( keys( ( filter( (k,v) -> v == j, state.state_dp.labels ) ) ) ) )
        Hj = input.data.Hmat[vcat(idx, idx+input.dims.n, idx+2*input.dims.n),:] # 3nj x ktot        
        ## update theta
        state = update_theta!(state, input, j, idx, Hj)        
    end    
    return state    
end

## no update, check assignment
function update_ymiss_only!(state::GibbsState, input::GibbsInput, j::Int64, idx::Vector{Int64})

    nj = state.state_dp.njs[j]

    dstarj = view(state.state_data.dstar, idx) # nj x 1
    y1j = view(state.state_data.y1, idx) # nj x 1
    y0j = view(state.state_data.y0, idx) # nj x 1
    
    yj = view(input.data.y.a, idx) # nj x 1
    dj = view(input.data.d, idx) # nj x 1
    
    Hb = input.data.Hmat[vcat(idx, idx+input.dims.n, idx+2*input.dims.n),:] * state.state_theta[j].beta # 3nj x 1
    zbD = view(Hb, 1:nj) # nj x 1
    xb1 = view(Hb, (nj+1):(2nj)) # nj x 1
    xb0 = view(Hb, (2nj+1):(3nj)) # nj x 1
    
    ## pre-compute constants
    mid = (state.state_theta[j].Sigma[2,3])*(state.state_theta[j].Sigma[1,3])*(state.state_theta[j].Sigma[1,2])
    
    ## 1. update missing outcomes -> y1, y0
    function sample_ymiss(Sigma::Matrix{Float64})
        ## compute denominators
        denom1 = (Sigma[2,2]) - (Sigma[1,2])^2
        denom0 = (Sigma[3,3]) - (Sigma[1,3])^2
        ## compute potential outcome means
        mu1 = xb1 +
        (dstarj - zbD) * ( ( (Sigma[3,3]) * (Sigma[1,2]) - (Sigma[2,3]) * (Sigma[1,3]) ) / denom0 ) +
        (yj - xb0) * ( ( (Sigma[2,3]) - (Sigma[1,3]) * (Sigma[1,2]) ) / denom0 )        
        mu0 = xb0 +
        (dstarj - zbD) * ( ( (Sigma[2,2]) * (Sigma[1,3]) - (Sigma[2,3]) * (Sigma[1,2]) ) / denom1 ) +
        (yj - xb1) * ( ( (Sigma[2,3]) - (Sigma[1,3]) * (Sigma[1,2]) ) / denom1 )        
        ## compute potential outcome variances
        omega1 = (Sigma[2,2]) - ( ( (Sigma[1,2])^2 * (Sigma[3,3]) - 2*mid + (Sigma[2,3])^2 ) / denom0 )
        omega0 = (Sigma[3,3]) - ( ( (Sigma[1,3])^2 * (Sigma[2,2]) - 2*mid + (Sigma[2,3])^2 ) / denom1 )
        ## compute missing outcome mean and variance
        mu_miss = ( (1 .- dj) .* mu1 ) + ( dj .* mu0 )  
        var_miss = ( (1 .- dj) * omega1 ) + ( dj * omega0 )
        ## compute missing outcome
        return mu_miss + sqrt.(var_miss) .* randn(nj)
    end
    
    ymiss_j = sample_ymiss(state.state_theta[j].Sigma)
    
    ## update potential outcomes
    y1j = dj.*yj + (1 .- dj).*ymiss_j
    y0j = dj.*ymiss_j + (1 .- dj).*yj
    
    ## fake update potential outcomes
    ####ymiss_j = [ dj[i] == 1 ? y0j[i] : y1j[i] for i in 1:nj ]
    ##y1j = dj.*yj + (1 .- dj).*y1j
    ##y0j = dj.*y0j + (1 .- dj).*yj
    
    ##setindex!(state.state_data.dstar, dstarj, idx)
    setindex!(state.state_data.y1, y1j, idx)
    setindex!(state.state_data.y0, y0j, idx)
    
    return state
    
end

## --------------------------------------------------------------------------- #
## test functions: dpm_blocked

## update {Theta} only
function update_blocked_theta_only!(state::GibbsState, input::GibbsInput)
    ##for k in keys(state.state_dp.ws) or keys(state.state_dp.njs)
    @inbounds for j in 1:length(input.priors.prior_dp.J) 
        if state.state_dp.njs[j] == 0
            ## sample from prior
            state.state_theta = sample_prior_theta!(state.state_theta, input.priors.prior_theta, j)
        else
            ## sample from posterior
            idx = sort( collect( keys( ( filter( (k,v) -> v == j, state.state_dp.labels ) ) ) ) )
            Hj = input.data.Hmat[vcat(idx, idx+input.dims.n, idx+2*input.dims.n),:] # 3nj x ktot        
            ## update theta
            state = update_theta!(state, input, j, idx, Hj)
        end
    end
    return state
end
