## treatment effect posterior predictives
export dpm_ppd, rand_ppd, rand_dpm, rand_blocked, rand_gaussian, dpm_ate 

## --------------------------------------------------------------------------- #
## TODO: dpm_ppd computes posterior predictive density values
function dpm_ppd(out::GibbsOut, znew::Vector{Float64}, prior_theta::PriorTheta;
                 ##hnew::SparseMatrixCSC{Float64,Int64}
                 xid::UnitRange=1:(length(znew)-1))
    
    rho = prior_theta.prior_Sigma.rho
    rhoR = prior_theta.prior_Sigma.rho*prior_theta.prior_Sigma.R
    mu = prior_theta.prior_beta.mu
    if prior_theta.prior_beta.Vinv
        ## convert precision to covariance
        V = inv(prior_theta.prior_beta.V)
    end
    
    ## predictors
    hnew = blockdiag(sparse(znew'), sparse(znew[xid]'), sparse(znew[xid]')) # 3 x ktot
    
    ## pre-allocate storage
    ynew_out = Array{Float64}(undef, 0)
    ynew = Array{Float64}(undef, 0)
    w = Array{Float64}(undef, 0)
    
    for m in 1:length(out.out_dp)
        
        alpha = out.out_dp[m].alpha
        J = out.out_dp[m].J
        njs = out.out_dp[m].njs
        theta = out.out_theta[m]
        
        for j in 1:J
            hb = hnew*theta[j].beta
            append!( ynew, hb + cholesky(theta[j].Sigma).U'*randn(3) )
            push!( w, njs[j] )
        end
        
        ## draw from prior
        Sigma = NobileWishart(rho, rhoR)
        beta = mu + cholesky(V).U'*randn(length(mu))
        hb = hnew*beta
        append!( ynew, hb + cholesky(Sigma).U'*randn(3) )
        push!( w, alpha )
        
        jm = rand( Distributions.Categorical(w/sum(w)) )
        append!( ynew_out, ynew[(3*(jm-1)+1):3*jm] )
        
        ## reset storage
        resize!(ynew, 0)
        resize!(w, 0)
        
    end
    
    return reshape(ynew_out, 3, length(out.out_dp))
    
end

## --------------------------------------------------------------------------- #
## sample latent data from posterior
function rand_dpm(out::GibbsOut, p::PriorTheta, hnew::SparseMatrixCSC{Float64,Int64}; 
                  component_filter::Function = j -> true)  # Default: use all components
    ## setup...
    rho = p.prior_Sigma.rho
    rhoR = rho*p.prior_Sigma.R
    mu = p.prior_beta.mu
    if p.prior_beta.Vinv
        ## convert precision to covariance
        V = inv(p.prior_beta.V)
    end
    ## pre-allocate storage
    ynew_out = Array{Float64}(undef, 0)
    ynew = Array{Float64}(undef, 0)
    w = Array{Float64}(undef, 0)    
    for m in 1:length(out.out_dp)        
        alpha = out.out_dp[m].alpha
        J = out.out_dp[m].J
        njs = out.out_dp[m].njs
        theta = out.out_theta[m]        
        for j in 1:J
            if component_filter(j) && njs[j] > 0  # Apply filter
                hb = hnew*theta[j].beta
                append!( ynew, hb + cholesky(theta[j].Sigma).U'*randn(3) )
                push!( w, njs[j] )
            end
        end 
        ## draw from prior
        Sigma = NobileWishart(rho, rhoR)
        beta = mu + cholesky(V).U'*randn(length(mu))
        hb = hnew*beta
        append!( ynew, hb + cholesky(Sigma).U'*randn(3) )
        push!( w, alpha )
        ## select component
        jm = rand( Distributions.Categorical(w/sum(w)) )
        append!( ynew_out, ynew[(3*(jm-1)+1):3*jm] )
        ## reset storage
        resize!(ynew, 0)
        resize!(w, 0)        
    end    
    return reshape(ynew_out, 3, length(out.out_dp))
end

function rand_blocked(out::GibbsOut, hnew::SparseMatrixCSC{Float64,Int64}) # also works for fmn
    ## setup...
    ## pre-allocate storage
    ynew_out = Array{Float64}(undef, 0)
    ynew = Array{Float64}(undef, 0)
    w = Array{Float64}(undef, 0)
    for m in 1:length(out.out_dp)
        theta = out.out_theta[m]
        ws = out.out_dp[m].ws
        for j in keys(ws)
            wj = ws[j].w
            hb = hnew * theta[j].beta
            append!( ynew, hb + cholesky(theta[j].Sigma).U'*randn(3) )
            push!( w, wj )
        end
        jm = rand( Distributions.Categorical(w/sum(w)) )
        append!( ynew_out, ynew[(3*(jm-1)+1):3*jm] )
        ## reset storage
        resize!(ynew, 0)
        resize!(w, 0)
    end
    return reshape(ynew_out, 3, length(out.out_dp)) 
end

function rand_fmn(out::GibbsOut, input::GibbsInput, hnew::SparseMatrixCSC{Float64,Int64})
    ## setup...
    J = input.priors.prior_dp.J
    aJ = input.priors.prior_dp.alpha/float(J)
    ## pre-allocate storage
    ynew_out = Array{Float64}(undef, 0)
    ynew = Array{Float64}(undef, 0)
    w = Array{Float64}(undef, 0)
    for m in 1:length(out.out_dp)
        theta = out.out_theta[m]
        njs = out.out_dp[m].njs
        for j in 1:J
            hb = hnew * theta[j].beta
            append!( ynew, hb + cholesky(theta[j].Sigma).U'*randn(3) )
            push!( w, njs[j] + aJ )
        end
        jm = rand( Distributions.Categorical(w/sum(w)) )
        append!( ynew_out, ynew[(3*(jm-1)+1):3*jm] )
        ## reset storage
        resize!(ynew, 0)
        resize!(w, 0)
    end
    return reshape(ynew_out, 3, length(out.out_dp))            
end

function rand_gaussian(out::GibbsOut, hnew::SparseMatrixCSC{Float64,Int64})
    ## setup...
    ## pre-allocate storage
    ynew_out = Array{Float64}(undef, 0)
    ynew = Array{Float64}(undef, 0)
    for m in 1:length(out.out_dp)
        hb = hnew * out.out_theta[m][1].beta
        append!( ynew_out, hb + cholesky(out.out_theta[m][1].Sigma).U'*randn(3) )
        ## reset storage
        resize!(ynew, 0)
    end
    return reshape(ynew_out, 3, length(out.out_dp))
end

function rand_ppd(out::GibbsOut, input::GibbsInput, znew::Vector{Float64})
    hnew = blockdiag(sparse(znew'), sparse(znew[1:input.dims.kx]'), sparse(znew[1:input.dims.kx]'))
    if input.params.model == "dpm"
        ppd = rand_dpm(out, input.priors.prior_theta, hnew)
    elseif input.params.model == "marginal"
        ppd = rand_dpm(out, input.priors.prior_theta, hnew, 
                       component_filter = j -> out.out_dp[end].njs[j] > 0)
    elseif input.params.model == "blocked"
        ppd = rand_blocked(out, hnew)
    elseif input.params.model == "fmn"
        ppd = rand_fmn(out, input, hnew)
    elseif input.params.model == "gaussian"
        ppd = rand_gaussian(out, hnew)
    else
        error("No method for model $(input.params.model)!")
    end
    return ppd
end

## given draws from posterior predictive, compute treatment effects
function dpm_ate(ynew::Matrix{Float64}, input::GibbsInput)
    sy = input.data.y.s[1]
    dNew = vec(ynew[1,:])
    ateNew = vec(ynew[2,:] - ynew[3,:])*sy
    idx = findall(d -> d > 0.0, dNew)

    ttNew = ateNew[idx]
    return TreatmentEffects(ate=ateNew, tt=ttNew)
end

## --------------------------------------------------------------------------- #
## compute density values
function dpm_density(ynew::Matrix{Float64}, input::GibbsInput)

    sy = input.data.y.s[1]
    ## ynew -> ate -> ate.dens
    dNew = vec(ynew[1,:])
    ateNew = vec(ynew[2,:] - ynew[3,:])*sy
    ## ynew -> tt -> tt.dens
    idx = findall(d -> d > 0.0, dNew)
    ttNew = ateNew[idx]
    ## PPD(grid, ate.dens, tt.dens)
    
    return PPD(grid=range(1, 1, length=1), ate=ateNew, tt=ttNew)

end

## --------------------------------------------------------------------------- #
## output active components
function out_J(out::GibbsOut)
    outJ = Array{Int64}(undef, length(out.out_dp))
    for m in 1:length(out.out_dp)
        outJ[m] = out.out_dp[m].J
    end
    return outJ
end
