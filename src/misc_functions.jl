## misc functions
export flip_mat, NobileWishart, truncnorm, standardize #
export rescale_beta, rescale_output, check_y, rand_cdf, acf_var, acf_var2, batch_var, ppd_cdf # processing output
export make_table

## --------------------------------------------------------------------------- #
## helper functions

## flip rows 1,3 and columns 1,3 for 3x3 matrix
function flip_mat(A::Array) reverse(reverse(A, dims=1), dims=2) end


## sample 3x3 A ~ IW(nu, V) with constraint A(1,1)=1
function NobileWishart(nu::Int64, V::Array{Float64})
    
    ## lower triangular Cholesky factor for V^-1
    I3 = Matrix(1.0I, 3, 3)
    ##L = chol(flip_mat(V)\I3, Val{:L}) ## deprecated
    L = adjoint( cholesky( Hermitian(flip_mat(V)\I3) ).U )

    ## initialize lower triangular A
    A = zeros(3, 3)
    A[2,1], A[3,1], A[3,2] = randn(3)

    ## fill in diagonal
    A[1,1] = sqrt( rand( Chisq(nu)) )
    A[2,2] = sqrt( rand( Chisq(nu-1)) )
    A[3,3] = 1/L[3,3]

    ## construct out = (L^-1)'(A^-1)'(A^-1)(L^-1)
    L = L\I3
    A = A\I3 
    A = *(A, L) # (A^-1)(L^-1)
    A = *(A', A) # Sigma

    return flip_mat(A)

end


## sample truncated normals at 0, given d = {0, 1}
function truncnorm(mu::Array, sigma::Array, d::Array)
    
    u = rand(size(mu,1))

    c = cdf(Normal(), -mu./sigma)

    p = ( c.*(1 - d) + (1 - c).*d ).*u + c.*d

    z = mu + sigma.*quantile(Normal(), p)

    return z
    
end


## center and scale continuous data, A
function standardize(A::Array{Float64}; B=2)

    ## inputs
    ## A (n x p) : data matrix
    ## B (Integer): bound on unique values for factors
    
    ## outputs
    ## a (n x p): scaled data matrix
    ## m (p x 1): centering for each column of A
    ## s (p x ): scaling for each column of A
    
    p = size(A, 2)
    s = zeros(p)
    m = zeros(p)
    a = zeros(size(A))
    
    for k in 1:p
        if size(unique(A[:,k]), 1) <= B ##eltype(A[:,k]) <: Integer
            ##next
            s[k] = 1.0
            m[k] = 0.0            
        else  
            s[k] = std(A[:,k])
            m[k] = mean(A[:,k])            
        end
        a[:,k] = (A[:,k] .- m[k])/s[k]
    end
    
    return ScaleData(a=a, m=m, s=s)
    
end


## re-scale coefficients, redux
function rescale_beta(beta::Vector{Float64}, xs::ScaleData, ys::ScaleData)

    x = xs.a
    mx = xs.m
    sx = xs.s

    y = ys.a
    my = ys.m[1]
    sy = ys.s[1]
    
    p = size(beta, 1)
    
    b = zeros(size(beta))
    
    ##b[1] = sy*( beta[1] + my - sum(beta[2:p].*mx[2:p]./sx[2:p]) ) # NB: mx[1]=0
    b[1] = sy*( beta[1] + my - dot( beta, mx./sx ) )
    for k in 2:p
        b[k] = beta[k]*sy/sx[k]
    end    
    
    return b
    
end


## --------------------------------------------------------------------------- #
## post-processing output


## rescale MCMC output
function rescale_output(out::GibbsOut)
    rescaled_out = GibbsOut(out_data=out.out_data, out_dp=out.out_dp, out_theta=out.out_theta)
    for m in 1:length(out.out_theta)
        for j in 1:length(rescaled_out.out_theta[m])
            rescaled_out.out_theta[m][j].beta = rescale_beta(rescaled_out.out_theta[m][j].beta)
        end
    end    
    return rescaled_out
end


## check data augmentation
function check_y(input::GibbsInput, out::GibbsOut)
    my, sy = input.data.y.m[1], input.data.y.s[1]
    y_out = Array{Float64}(undef, input.dims.n, length(out.out_data) )
    for m in 1:length(out.out_data)
        y_out[:,m] = input.data.d .* out.out_data[m].y1 + (1 - input.data.d) .* out.out_data[m].y0
    end
    return mean( y_out, 2)*sy + my
end


## evaluate CDF at y, F(y) = Pr(Y<y), by integrating PPD
function ppd_cdf(ppd::PPD; ate=true)
    
    ## inputs:
    ## ppd::PPD : such that sums=true
    ## ate::Bool : if true, CDF for ATE, else CDF for TT
    
    ## output:
    ## cdf::Function

    if ate
        a = ppd.ate
    else
        a = ppd.tt
    end    
    
    function cdf(y::Float64)        
        ## Input:        
        ## y::Float64 : value at which CDF evaluated
        ## Output:
        ## F(y)::Float64        
        idx = findall(xi -> xi < y, ppd.grid)        
        lens = diff(ppd.grid[idx])
        b = sub(a, idx)
        b = [ mean(b[i:i+1]) for i in 1:(length(b)-1) ]        
        return dot(lens, b)        
    end
    
    return cdf
    
end


## compute empirical CDF for vector of posterior samples, a
## returns cdf::Function
rand_cdf(a::Vector{Float64}) = StatsBase.ecdf(a)


## compute kernel density for vector of posterior samples, a
## returns kde::UnivariateKDE{FloatRange{Float64}}
#rand_pdf(a::Vector{Float64}) = KernelDensity.kde(a)


## compute Monte Carlo variance for scalar MCMC draws:
## 1. MC variance using autocorrelation function
acf_var(a::Vector{Float64}) = var(a)*( 1 + 2*sum(StatsBase.autocor(a)) )/length(a)


acf_var2(a::Vector{Float64}) = var(a)*( 1 + 2*sum(StatsBase.autocor(a, 0:(length(a)-1))) )/length(a)


## 2. MC variance by batching
function batch_var(a::Vector{Float64}, k::Int64)

    ## Inputs:
    ## a::Vector : vector of scalar MCMC draws
    ## k::Int64 : batch size
    
    ## Outputs:
    ## vb::Float64 : batch variance
    
    M = length(a)
    if mod(M, k) != 0
        return println("Warning:\nk must be a multiple of M")
    else
        n = div(M, k)
        a = reshape(a, k, n)
        batches = mean(a, 1)
        return var(batches)/n
    end
    
end

## --------------------------------------------------------------------------- #
## function to make tables
##function make_table(t::TreatmentEffects; expo::Bool=false, ...)
function make_table(a::Vector, b::Vector; tab_name::String="tab", tab_path::String=pwd(),
                    y::Float64=0.0, output::Bool=true)
    
    ## Input:
    ## a,b : ate, tt
    
    ## Output:
    ## a table

    tab = DataFrames.DataFrame()
    aCDF = StatsBase.ecdf(a)
    bCDF = StatsBase.ecdf(b)

    tab[:stat] = ["mean",
                   "median",
                   "variance",
                   "95pct credible set (lower)",
                   "95pct credible set (upper)",
                   "F($(y))" ]

    tab[:ate] = round([
                        mean(a),
                        median(a),
                        DPMixture.acf_var(a),
                        StatsBase.percentile(a, 2.5),
                        StatsBase.percentile(a, 97.5),
                        aCDF(y) ], 6)

    tab[:tt] = round([
                       mean(b),
                       median(b),
                       DPMixture.acf_var(b),
                       StatsBase.percentile(b, 2.5),
                       StatsBase.percentile(b, 97.5),
                       bCDF(y) ], 6)

    DataFrames.writetable(joinpath(tab_path, tab_name * ".dat"),
                          tab,
                          separator = ';',
                          quotemark = ' ',
                          header = true)

    println("Table " * tab_name * ": done!")

    if output
        return tab, aCDF, bCDF
    end
end
