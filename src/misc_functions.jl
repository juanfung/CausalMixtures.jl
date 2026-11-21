## misc functions
export flip_mat, NobileWishart, truncnorm, standardize #
export rescale_beta, check_y, rand_cdf, acf_var, acf_var2, batch_var # processing output

## --------------------------------------------------------------------------- #
## helper functions

## flip rows 1,3 and columns 1,3 for 3x3 matrix
function flip_mat(A::Array) reverse(reverse(A, dims=1), dims=2) end


## sample 3x3 A ~ IW(nu, V) with constraint A(1,1)=1
NobileWishart(nu::Int64, V::Array{Float64}) = NobileWishart(Random.default_rng(), nu, V)

function NobileWishart(rng::AbstractRNG, nu::Int64, V::Array{Float64})
    
    ## lower triangular Cholesky factor for V^-1
    I3 = Matrix(1.0I, 3, 3)
    ##L = chol(flip_mat(V)\I3, Val{:L}) ## deprecated
    L = adjoint( cholesky( Hermitian(flip_mat(V)\I3) ).U )

    ## initialize lower triangular A
    A = zeros(3, 3)
    A[2,1], A[3,1], A[3,2] = randn(rng, 3)

    ## fill in diagonal
    A[1,1] = sqrt( rand(rng, Chisq(nu)) )
    A[2,2] = sqrt( rand(rng, Chisq(nu-1)) )
    A[3,3] = 1/L[3,3]

    ## construct out = (L^-1)'(A^-1)'(A^-1)(L^-1)
    L = L\I3
    A = A\I3 
    A = *(A, L) # (A^-1)(L^-1)
    A = *(A', A) # Sigma

    return flip_mat(A)

end


## Cholesky factor L (Sigma = L L') of a 3x3 positive-definite matrix, held as scalars, and
## log|Sigma|. Built once per component per sweep, so the label steps evaluate the 3-dimensional
## normal density with a few flops and no allocation.
struct Chol3
    l11::Float64
    l21::Float64
    l31::Float64
    l22::Float64
    l32::Float64
    l33::Float64
    logdet::Float64
end

function Chol3(S::AbstractMatrix{<:Real})
    l11 = sqrt(S[1,1])
    l21 = S[2,1] / l11
    l31 = S[3,1] / l11
    l22 = sqrt(S[2,2] - l21^2)
    l32 = (S[3,2] - l31 * l21) / l22
    l33 = sqrt(S[3,3] - l31^2 - l32^2)
    return Chol3(l11, l21, l31, l22, l32, l33, 2 * (log(l11) + log(l22) + log(l33)))
end

const LOG2PI = log(2pi)

## log N3(e | 0, Sigma) for the residual e = (e1, e2, e3), given Sigma's factor c
@inline function logpdf3(c::Chol3, e1::Real, e2::Real, e3::Real)
    u1 = e1 / c.l11
    u2 = (e2 - c.l21 * u1) / c.l22
    u3 = (e3 - c.l31 * u1 - c.l32 * u2) / c.l33
    return -0.5 * (u1^2 + u2^2 + u3^2) - 0.5 * c.logdet - 1.5 * LOG2PI
end


## log(sum(exp.(logw))) without overflow or underflow
function logsumexp(logw::AbstractVector{<:Real})
    m = maximum(logw)
    isfinite(m) || return m
    return m + log(sum(x -> exp(x - m), logw))
end

## sample an index with probability proportional to exp.(logw), computed stably
function sample_logweights(logw::AbstractVector{<:Real})
    p = exp.(logw .- logsumexp(logw))
    all(isfinite, p) || error("invalid label log weights: $(logw)")
    return rand( Distributions.Categorical(p ./ sum(p)) )
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


## check data augmentation
function check_y(input::GibbsInput, out::GibbsOut)
    my, sy = input.data.y.m[1], input.data.y.s[1]
    y_out = Array{Float64}(undef, input.dims.n, length(out.out_data) )
    for m in 1:length(out.out_data)
        y_out[:,m] = input.data.d .* out.out_data[m].y1 .+ (1 .- input.data.d) .* out.out_data[m].y0
    end
    return vec(mean(y_out, dims=2)) .* sy .+ my
end


## compute empirical CDF for vector of posterior samples, a
## returns cdf::Function
rand_cdf(a::Vector{Float64}) = StatsBase.ecdf(a)


## compute kernel density for vector of posterior samples, a
## returns kde::UnivariateKDE{FloatRange{Float64}}
#rand_pdf(a::Vector{Float64}) = KernelDensity.kde(a)


## compute Monte Carlo variance for scalar MCMC draws, var(a) * tau / n, where
## tau = 1 + 2 * sum_{k>=1} rho_k is the integrated autocorrelation time.
## (StatsBase.autocor includes lag 0, which must not enter the sum.)

## 1. MC variance using autocorrelation function, lags 1:L with StatsBase's default
##    window L = min(n - 1, round(10 log10 n))
function acf_var(a::Vector{Float64})
    n = length(a)
    L = min(n - 1, round(Int, 10*log10(n)))
    return var(a)*( 1 + 2*sum(StatsBase.autocor(a, 1:L)) )/n
end

## 1b. MC variance with Geyer's (1992) initial positive sequence: tau = -1 + 2 sum_{k=0}^{K} G_k,
##     G_k = rho_{2k} + rho_{2k+1}, stopping before the first G_k <= 0. Summing every lag up to
##     n - 1 instead is useless: those sample autocorrelations sum to exactly -1/2, giving 0.
function acf_var2(a::Vector{Float64}; block::Int=64)
    n = length(a)
    tau = -1.0
    lag0 = 0
    while lag0 + 1 <= n - 1
        lags = lag0:min(lag0 + 2*block - 1, n - 1)
        r = StatsBase.autocor(a, collect(lags))
        for k in 1:2:(length(r) - 1)
            G = r[k] + r[k+1]
            G <= 0 && return var(a)*tau/n
            tau += 2G
        end
        lag0 = last(lags) + 1 - (isodd(length(lags)) ? 1 : 0)
    end
    return var(a)*tau/n
end


## 2. MC variance by batching
function batch_var(a::Vector{Float64}, k::Int64)

    ## Inputs:
    ## a::Vector : vector of scalar MCMC draws
    ## k::Int64 : batch size
    
    ## Outputs:
    ## vb::Float64 : batch variance
    
    M = length(a)
    mod(M, k) == 0 || throw(ArgumentError("the batch size k = $k must divide M = $M"))
    n = div(M, k)
    batches = vec(mean(reshape(a, k, n), dims=1))
    return var(batches)/n
    
end
