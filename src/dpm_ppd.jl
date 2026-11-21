## treatment effect posterior predictives
export dpm_ppd, rand_ppd, rand_dpm, rand_blocked, rand_gaussian, dpm_ate, ppd_design, mean_covariates

## --------------------------------------------------------------------------- #
## dpm_ppd: posterior predictive draws for the Polya-urn (dpm) output at a new unit.
## The design is built by column name (ppd_design), as in rand_ppd; the draws come from rand_dpm.
## prior_theta defaults to the one the sampler used.
dpm_ppd(out::GibbsOut, input::GibbsInput, znew::AbstractVector{<:Real};
        prior_theta::PriorTheta=input.priors.prior_theta) =
    rand_dpm(out, prior_theta, ppd_design(input, znew))

dpm_ppd(out::GibbsOut, input::GibbsInput, xnew::AbstractVector{<:Real}, wnew::AbstractVector{<:Real};
        prior_theta::PriorTheta=input.priors.prior_theta) =
    rand_dpm(out, prior_theta, ppd_design(input, xnew, wnew))

## --------------------------------------------------------------------------- #
## sample latent data from posterior
function rand_dpm(out::GibbsOut, p::PriorTheta, hnew::SparseMatrixCSC{Float64,Int64}; 
                  component_filter::Function = j -> true)  # Default: use all components
    ## setup...
    rho = p.prior_Sigma.rho
    rhoR = rho*p.prior_Sigma.R
    mu = p.prior_beta.mu
    Lcov = p.prior_beta.Lcov # factor of the prior covariance V0, whatever Vinv is
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
        beta = mu + Lcov*randn(length(mu))
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

## --------------------------------------------------------------------------- #
## covariates for the posterior predictive

"""
    mean_covariates(input) -> (xbar, wbar)

Means over all n units of the outcome-equation covariates `x` and the selection-equation
covariates `w`, in the column order of `input.data.xnames` and `input.data.znames`
(on the sampler's scale, so standardized if `scale_data` was set).
"""
function mean_covariates(input::GibbsInput)
    n, kx, kz = input.dims.n, input.dims.kx, input.dims.kz
    H = input.data.Hmat
    wbar = vec(mean(Matrix(H[1:n, 1:kz]), dims=1))
    xbar = vec(mean(Matrix(H[(n+1):(2n), (kz+1):(kz+kx)]), dims=1))
    return xbar, wbar
end

"""
    ppd_design(input, xnew, wnew)

The 3 x ktot design `blockdiag(wnew', xnew', xnew')` for a new unit.
"""
function ppd_design(input::GibbsInput, xnew::AbstractVector{<:Real}, wnew::AbstractVector{<:Real})
    length(xnew) == input.dims.kx || throw(DimensionMismatch("xnew has length $(length(xnew)), expected $(input.dims.kx)"))
    length(wnew) == input.dims.kz || throw(DimensionMismatch("wnew has length $(length(wnew)), expected $(input.dims.kz)"))
    return blockdiag(sparse(Float64.(wnew)'), sparse(Float64.(xnew)'), sparse(Float64.(xnew)'))
end

"""
    ppd_design(input, znew)

Design for a new unit given only its selection-equation covariates `znew` (in the order of
`input.data.znames`). The outcome covariates are taken from `znew` by column name, so the
order of terms in either formula does not matter. Every outcome-equation column must also
appear in the selection equation; otherwise pass `xnew` and `wnew` separately.
"""
function ppd_design(input::GibbsInput, znew::AbstractVector{<:Real})
    xnames, znames = input.data.xnames, input.data.znames
    idx = map(xnames) do nm
        k = findfirst(==(nm), znames)
        k === nothing && throw(ArgumentError("outcome column $(nm) is not in the selection equation; " *
                                             "call rand_ppd(out, input, xnew, wnew)"))
        k
    end
    return ppd_design(input, znew[idx], znew)
end

rand_ppd(out::GibbsOut, input::GibbsInput, znew::AbstractVector{<:Real}) =
    rand_ppd(out, input, ppd_design(input, znew))

rand_ppd(out::GibbsOut, input::GibbsInput, xnew::AbstractVector{<:Real}, wnew::AbstractVector{<:Real}) =
    rand_ppd(out, input, ppd_design(input, xnew, wnew))

function rand_ppd(out::GibbsOut, input::GibbsInput, hnew::SparseMatrixCSC{Float64,Int64})
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
