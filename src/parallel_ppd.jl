## parallelized ppd draws
export select_ppd, single_dpm_ppd, single_blocked_ppd, single_fmn_ppd, single_gaussian_ppd, setup_ppd, parallel_ppd, parallel_rand_ppd

## select random draw function
function select_ppd(model::String="dpm")
    # select ppd function
    if model == "dpm"
        return single_dpm_ppd
    elseif model == "blocked"
        return single_blocked_ppd
    elseif model == "fmn"
        return single_fmn_ppd
    elseif model == "gaussian"
        return single_gaussian_ppd
    else
        error("No method for model $(model)!")
    end
end

## draw from polya urn ppd
function single_dpm_ppd(out::GibbsOut, hnew::SparseMatrixCSC{Float64,Int64}, m::Int64,
                        rho::Real, rhoR::Matrix{Float64}, mu::Vector{Float64}, V::Matrix{Float64})
    
    ynew = Array(Float64, 0)
    w = Array(Float64, 0)
    
    alpha = out.out_dp[m].alpha
    J = out.out_dp[m].J
    njs = out.out_dp[m].njs
    theta = out.out_theta[m]
    
    ## draw from ppd
    for j in 1:J
        hb = hnew*theta[j].beta
        append!( ynew, hb + chol(theta[j].Sigma)'*randn(3) )
        push!( w, njs[j]*exp(logpdf(MvNormal(hb, theta[j].Sigma), ynew[(3*(j-1)+1):3*j])) )
    end
    
    ## draw from prior
    Sigma = NobileWishart(rho, rhoR)
    beta = mu + chol(V)'*randn(length(mu))
    hb = hnew*beta
    append!( ynew, hb + chol(Sigma)'*randn(3) )
    push!( w, alpha*exp(logpdf(MvNormal(hb, Sigma), ynew[(3*J+1):3*(J+1)])) )
    
    ## select component
    jm = rand( Distributions.Categorical(w/sum(w)) )    
    return ynew[(3*(jm-1)+1):3*jm]
    
end

## draw from blocked dpm ppd
function single_blocked_ppd(out::GibbsOut, hnew::SparseMatrixCSC{Float64,Int64}, m::Int64, sdummy::Float64)
    ynew = Array(Float64, 0)
    w = Array(Float64, 0)
    theta = out.out_theta[m]
    ws = out.out_dp[m].ws
    for j in keys(ws)
        wj = ws[j].w
        hb = hnew * theta[j].beta
        append!( ynew, hb + chol(theta[j].Sigma)'*randn(3) )
        push!( w, wj*exp(logpdf(MvNormal(hb, theta[j].Sigma), ynew[(3*(j-1)+1):3*j])) )
    end
    jm = rand( Distributions.Categorical(w/sum(w)) )
    return ynew[(3*(jm-1)+1):3*jm]
end

## draw from fmn ppd
function single_fmn_ppd(out::GibbsOut, hnew::SparseMatrixCSC{Float64,Int64}, m::Int64,
                        J::Int64, aJ::Float64)
    ynew = Array(Float64, 0)
    w = Array(Float64, 0)
    theta = out.out_theta[m]
    njs = out.out_dp[m].njs
    for j in 1:J
        hb = hnew * theta[j].beta
        append!( ynew, hb + chol(theta[j].Sigma)'*randn(3) )
        push!( w, (njs[j] + aJ)*exp(logpdf(MvNormal(hb, theta[j].Sigma), ynew[(3*(j-1)+1):3*j])) )
    end
    jm = rand( Distributions.Categorical(w/sum(w)) )
    return ynew[(3*(jm-1)+1):3*jm]
end

## draw from gaussian ppd
function single_gaussian_ppd(out::GibbsOut, hnew::SparseMatrixCSC{Float64,Int64}, m::Int64, sdummy::Float64)
    ynew = Array(Float64, 0)
    hb = hnew * out.out_theta[m][1].beta
    return hb + chol(out.out_theta[m][1].Sigma)'*randn(3)
end

## pre-compute necessary constants
function setup_ppd(out::GibbsOut, input::GibbsInput)
    if input.params.model == "dpm"
        if input.priors.prior_theta.prior_beta.Vinv
        ## convert precision to covariance
            V = input.priors.prior_theta.prior_beta.V\Matrix{Float64}(I, input.priors.prior_theta.prior_beta.V, input.priors.prior_theta.prior_beta.V)
        end
        rho = input.priors.prior_theta.prior_Sigma.rho
        rhoR = rho*input.priors.prior_theta.prior_Sigma.R
        mu = input.priors.prior_theta.prior_beta.mu
        return rho, rhoR, mu, V
    elseif input.params.model == "fmn"
        J = input.priors.prior_dp.J
        aJ = input.priors.prior_dp.alpha/float(J)
        return J, aJ
    else
        return 0.
    end
end

## parallel draws
function parallel_ppd(out::GibbsOut, input::GibbsInput, znew::Vector{Float64})

    ## construct predictor matrix
    hnew = blkdiag(sparse(znew'), sparse(znew[1:input.dims.kx]'), sparse(znew[1:input.dims.kx]'))
    
    ## total samples
    M = length(out.out_dp)

    ## pre-compute
    setup = setup_ppd(out, input)

    ## select sampling function
    ppd = select_ppd(input.params.model)

    ## compile to given (out, hnew, setup...)
    function single_ppd(m::Int)
        ## draw from ppd for mth MCMC sample
        ## return ynew_out (3 x 1)
        return ppd(out, hnew, m, setup...)
    end
    
    ## parallel draws
    ##ynew::Array{Any,1} = pmap( single_ppd, Array(Int64, M) ) # - nonsense m!
    ynew = pmap( single_ppd, 1:M ) # Array(Array(Float64,3),M) - (M x 1)
    
    ## return Array{Float64,2}
    return hcat(ynew...) # (3 x M)
    
end

function parallel_rand_ppd(out_path::AbstractString, z_path::AbstractString)

    state = JLD2.jldopen( joinpath(out_path, "state.jld2"), "r")
    input = read(state, "input")
    close(state)

    znew = vec( readcsv(z_path) )

    path_to_output = joinpath(out_path, "output")    
    outs = readdir( path_to_output )

    function dist_rand_ppd(p::AbstractString) # or ByteString
        
        o = JLD2.jldopen( joinpath(path_to_output, p), "r")
        
        out = read(o, "out")
        
        close(o)
        
        ynew = CausalMixtures.rand_ppd(out, input, znew)
        
        return ynew
        
    end

    ynew_par = pmap(dist_rand_ppd, outs)

    return hcat(ynew_par...)

end


function parallel_out_J(out_path::AbstractString)    
    
    path_to_output = joinpath(out_path, "output")    
    outs = readdir( path_to_output )

    function dist_out_J(p::AbstractString) # or ByteString        
        o = JLD2.jldopen( joinpath(path_to_output, p), "r")        
        out = read(o, "out")        
        close(o)        
        outJ = CausalMixtures.out_J(out)        
        return outJ
        
    end
    
    outJ_par = pmap(dist_out_J, outs)

    return vcat(outJ_par...)

end
    
