## --------------------------------------------------------------------------- #
## Phase 6: simulate_selection, a hedonic-flavoured selection-model simulator with sample and
## population truth. Replaces generate_hedonic_data (kept, as legacy).
##
## Covariates (raw scale), from a Gaussian copula: log_sqft ~ N(7.4, 0.35^2), age ~ Gamma(2, 15),
## log_lot ~ N(8.8, 0.5^2), bedrooms in 1:5 (probabilities 0.05, 0.20, 0.40, 0.25, 0.10),
## income (neighbourhood median, $10k) ~ LogNormal(log 6.5, 0.35), and the excluded instrument
## distance (km to the nearest choice school) ~ Exponential(3). The model is linear in the
## covariates standardized by their population moments (x~, w~), hence linear in the raw ones:
##   d* = g0_t + x~'g_x + iv g_w w~ + v,   d = 1{d* > 0}
##   y_k = mu_k,t + x~'b_k + u_k,   k = 1 (treated), 0,   log price around 12.
## t is the latent type (:mixture only). The outcome noise is set so that the population R^2 of
## each potential outcome on x is about r2 (under :mixture the type means count as noise; if they
## alone exceed the budget, the achieved R^2 is below r2). REALISTIC mode: ATE about 0.05 log
## points; diagnostic = true multiplies the effects by 10 and separates the types. Under :mixture
## the zoned-school quality raises y0 by 0.06 q and lowers the gain by the same 0.06 (q - qbar),
## so y1 does not depend on the type and y0 carries the whole type separation (in diagnostic mode
## the achieved R^2 of y0 is far below r2).

export simulate_selection

const _SS_PROBS = (0.1, 0.25, 0.5, 0.75, 0.9)
const _SS_COPULA = [1.0 -0.1 0.4 0.7 0.3 0.0;
                    -0.1 1.0 0.1 -0.05 -0.2 0.1;
                    0.4 0.1 1.0 0.3 0.1 0.2;
                    0.7 -0.05 0.3 1.0 0.2 0.0;
                    0.3 -0.2 0.1 0.2 1.0 -0.1;
                    0.0 0.1 0.2 0.0 -0.1 1.0]
const _SS_BEDP = [0.05, 0.20, 0.40, 0.25, 0.10]
const _SS_NAMES = (:log_sqft, :age, :log_lot, :bedrooms, :income, :distance)

## raw covariates (n x 6, the last column the instrument) from the copula
function _ss_covariates(rng::AbstractRNG, n::Integer)
    L = cholesky(Symmetric(_SS_COPULA)).L
    Z = randn(rng, n, 6) * L'
    U = cdf.(Normal(), Z)
    cb = cumsum(_SS_BEDP)
    A = Matrix{Float64}(undef, n, 6)
    A[:, 1] = 7.4 .+ 0.35 .* Z[:, 1]
    A[:, 2] = quantile.(Gamma(2.0, 15.0), U[:, 2])
    A[:, 3] = 8.8 .+ 0.5 .* Z[:, 3]
    A[:, 4] = [Float64(findfirst(>=(u), cb)) for u in U[:, 4]]
    A[:, 5] = quantile.(LogNormal(log(6.5), 0.35), U[:, 5])
    A[:, 6] = quantile.(Exponential(3.0), U[:, 6])
    return A
end

## population means and sds of the six covariates
function _ss_moments()
    bm = sum((1:5) .* _SS_BEDP); bs = sqrt(sum(((1:5) .- bm) .^ 2 .* _SS_BEDP))
    m = [7.4, mean(Gamma(2.0, 15.0)), 8.8, bm, mean(LogNormal(log(6.5), 0.35)), 3.0]
    s = [0.35, std(Gamma(2.0, 15.0)), 0.5, bs, std(LogNormal(log(6.5), 0.35)), 3.0]
    return m, s
end

## the covariance of the standardized covariates (Pearson, after the marginal transforms),
## computed once from a fixed internal draw
const _SS_XCOV = Ref{Matrix{Float64}}()
function _ss_xcov()
    if !isassigned(_SS_XCOV)
        m, s = _ss_moments()
        A = _ss_covariates(Xoshiro(424242), 400_000)
        _SS_XCOV[] = cov((A .- m') ./ s')
    end
    return _SS_XCOV[]
end

## the parameters of a scenario
function _ss_params(scenario::Symbol, r2::Real, iv::Real, diagnostic::Bool)
    scenario in (:gaussian, :mixture, :skewed, :roy, :weak_iv) ||
        throw(ArgumentError("scenario must be :gaussian, :mixture, :skewed, :roy or :weak_iv, got $scenario"))
    0 < r2 < 1 || throw(ArgumentError("r2 must be in (0, 1), got $r2"))
    scenario == :weak_iv && (iv = 0.2)
    f = diagnostic ? 10.0 : 1.0
    b0 = [0.20, -0.04, 0.05, 0.03, 0.12]
    b1 = b0 .+ f .* [0.0, 0.0, 0.0, 0.0, 0.02]
    tau = 0.05f
    gx = [0.1, 0.0, 0.0, 0.0, 0.3]
    scenario == :roy && (gx = gx .+ (b1 .- b0) .* (5 / f))      # selection on the observed gain too
    gw = -0.8 * iv
    C = _ss_xcov()[1:5, 1:5]
    vs1, vs0 = dot(b1, C * b1), dot(b0, C * b0)               # Var(x~'b_k)
    if scenario == :mixture
        pt = [0.3, 0.4, 0.3]
        q = diagnostic ? [-2.0, 0.0, 2.0] : [-1.0, 0.0, 1.0]
        kappa = 0.06f
        qbar = dot(pt, q)
        mu0 = 12.0 .+ 0.06f .* q
        mu1 = mu0 .+ tau .- kappa .* (q .- qbar)
        g0 = -0.3 .- 0.5 .* q
        rho = [(0.5, -0.3), (0.2, 0.0), (-0.1, 0.3)]
        mult = [0.8, 1.0, 1.2]
        tv(mu) = dot(pt, (mu .- dot(pt, mu)) .^ 2)
        e1 = max(vs1 * (1 - r2) / r2 - tv(mu1), 0.1 * vs1 * (1 - r2) / r2)
        e0 = max(vs0 * (1 - r2) / r2 - tv(mu0), 0.1 * vs0 * (1 - r2) / r2)
        m2 = dot(pt, mult .^ 2)
        sd1, sd0 = mult .* sqrt(e1 / m2), mult .* sqrt(e0 / m2)
    else
        pt = [1.0]; q = [0.0]; kappa = 0.0
        mu0 = [12.0]; mu1 = [12.0 + tau]; g0 = [-0.3]
        rho = scenario == :roy ? [(0.6, -0.6)] : [(0.4, -0.2)]
        sd1, sd0 = [sqrt(vs1 * (1 - r2) / r2)], [sqrt(vs0 * (1 - r2) / r2)]
    end
    r10 = scenario == :roy ? 0.2 : 0.5
    return (scenario=scenario, r2=r2, iv=iv, diagnostic=diagnostic, b0=b0, b1=b1, gx=gx, gw=gw, pt=pt, q=q,
            mu0=mu0, mu1=mu1, g0=g0, rho=rho, r10=r10, sd1=sd1, sd0=sd0, skew=scenario == :skewed)
end

## one draw of n units: raw covariates, types, potential outcomes and d
function _ss_draw(rng::AbstractRNG, n::Integer, P)
    A = _ss_covariates(rng, n)
    m, s = _ss_moments()
    Xt = (A[:, 1:5] .- m[1:5]') ./ s[1:5]'
    wt = (A[:, 6] .- m[6]) ./ s[6]
    T = length(P.pt)
    types = T == 1 ? ones(Int, n) : rand(rng, Categorical(P.pt), n)
    E = randn(rng, n, 3)
    y1, y0, dstar = zeros(n), zeros(n), zeros(n)
    for t in 1:T
        r1, r0 = P.rho[t]
        Ct = [1.0 r1 r0; r1 1.0 P.r10; r0 P.r10 1.0]
        Lt = cholesky(Symmetric(Ct)).L
        idx = findall(==(t), types)
        e = E[idx, :] * Lt'
        v, u1, u0 = e[:, 1], e[:, 2], e[:, 3]
        if P.skew
            nu, dl = 4.0, 0.9
            v = v .* sqrt.((nu - 2) ./ rand(rng, Chisq(nu), length(idx)))        # t_4, variance 1
            sn(e) = (dl .* abs.(randn(rng, length(e))) .+ sqrt(1 - dl^2) .* e .- dl * sqrt(2 / pi)) ./ sqrt(1 - 2dl^2 / pi)
            u1, u0 = sn(u1), sn(u0)                                               # skew-normal, variance 1
        end
        y1[idx] = P.mu1[t] .+ Xt[idx, :] * P.b1 .+ P.sd1[t] .* u1
        y0[idx] = P.mu0[t] .+ Xt[idx, :] * P.b0 .+ P.sd0[t] .* u0
        dstar[idx] = P.g0[t] .+ Xt[idx, :] * P.gx .+ P.gw .* wt[idx] .+ v
    end
    d = Int.(dstar .> 0)
    return A, types, y1, y0, d
end

function _ss_truth(y1, y0, d)
    g = y1 .- y0
    tr, un = d .== 1, d .== 0
    pr = collect(_SS_PROBS)
    return (ate=mean(g), att=mean(g[tr]), atu=mean(g[un]), share_gain=mean(g .> 0),
            gain_q=quantile(g, pr), qte=quantile(y1, pr) .- quantile(y0, pr), probs=_SS_PROBS, treated_share=mean(d))
end

## population truth, cached per DGP: one N = 10^6 draw from an independent fixed stream
const _SS_POP = Dict{Any,Any}()
function _ss_population(P; N::Integer=1_000_000, pop_seed::Integer=20261125)
    key = (P.scenario, P.r2, P.iv, P.diagnostic, N, pop_seed)
    get!(_SS_POP, key) do
        _, _, y1, y0, d = _ss_draw(Xoshiro(pop_seed), N, P)
        t = _ss_truth(y1, y0, d)
        merge(t, (pate=t.ate, patt=t.att, patu=t.atu, N=N))
    end
end

## probit of d on [1 X] by Newton; coefficients and standard errors
function _ss_probit(Z::AbstractMatrix{Float64}, d::AbstractVector{<:Integer})
    g = zeros(size(Z, 2))
    H = zeros(size(Z, 2), size(Z, 2))
    for _ in 1:100
        s = Z * g; q = 2 .* d .- 1
        lt = exp.(logpdf.(Normal(), q .* s) .- logcdf.(Normal(), q .* s))   # phi(qs)/Phi(qs)
        H = -transpose(Z) * ((lt .* (lt .+ q .* s)) .* Z)                   # d2/ds2 log Phi(qs) = -lt (lt + qs)
        step = H \ (transpose(Z) * (q .* lt))
        g -= step
        maximum(abs, step) < 1e-10 && break
    end
    return g, sqrt.(diag(inv(-H)))
end

"""
    simulate_selection(n; scenario = :gaussian, rng = Random.default_rng(), r2 = 0.7, iv = 1.0,
                       diagnostic = false, pop_N = 10^6, pop_seed = 20261125) -> NamedTuple

Simulate `n` units of a hedonic-flavoured selection model (log house price around 12; the
treatment is, for example, choosing a school outside the zoned one), with sample and population
truth. Scenarios:
- `:gaussian`: one trivariate normal (v, u1, u0), corr(v, u1) = 0.4, corr(v, u0) = -0.2;
- `:mixture`: 3 latent types with zoned-school quality q (-1, 0, 1); the gain is
  tau - kappa (q - qbar), so there are winners and losers; type-specific selection intercepts and
  error covariances (including the selection covariances);
- `:skewed`: skew-normal outcome errors and a t_4 selection error (not a finite normal mixture);
- `:roy`: selection on gains, corr(v, u1) = 0.6, corr(v, u0) = -0.6, and the selection index
  loading on the observed gain;
- `:weak_iv`: `:gaussian` with `iv = 0.2`.

`iv` scales the instrument's (distance's) coefficient; `r2` sets the outcome noise so that the
population R^2 of each potential outcome on x is about `r2`. `diagnostic = true`: effects x 10
and well-separated types, for checking samplers rather than realism.

Returns `df` (raw covariates `log_sqft, age, log_lot, bedrooms, income, distance`, `Y_obs`,
`D_obs`), `formulas` (`formula_y`, `formula_d`, the instrument only in the latter), `types`
(the latent types; all 1 outside `:mixture`), `potential` (`y1`, `y0`), `truth.sample` (ATE,
ATT, ATU, the share with y1 > y0, quantiles of y1 - y0 and QTEs at `probs`, from the simulated
potential outcomes), `truth.population` (the same from one `pop_N` draw with the independent
stream `Xoshiro(pop_seed)`, cached, plus `pate`, `patt`, `patu`, which are what `effects`
estimates averaged over the covariate distribution), and `info` (treated share, the achieved
sample R^2 of each potential outcome on x, and the first-stage probit z-statistic of the
instrument with its square as a pseudo-F).
"""
function simulate_selection(n::Integer; scenario::Symbol=:gaussian, rng::AbstractRNG=Random.default_rng(),
                            r2::Real=0.7, iv::Real=1.0, diagnostic::Bool=false, pop_N::Integer=1_000_000,
                            pop_seed::Integer=20261125)
    P = _ss_params(scenario, r2, iv, diagnostic)
    A, types, y1, y0, d = _ss_draw(rng, n, P)
    df = DataFrame(A, collect(_SS_NAMES))
    df.Y_obs = ifelse.(d .== 1, y1, y0)
    df.D_obs = d
    fy = @formula(Y_obs ~ log_sqft + age + log_lot + bedrooms + income)
    fd = @formula(D_obs ~ log_sqft + age + log_lot + bedrooms + income + distance)
    Z = hcat(ones(n), (A .- mean(A, dims=1)) ./ std(A, dims=1))   # probit on standardized columns (same z)
    X = hcat(ones(n), A[:, 1:5])
    r2s(y) = (r = y - X * (X \ y); 1 - var(r) / var(y))
    ## first-stage strength (needs enough units of both kinds for the probit)
    zst = n >= 20 * size(Z, 2) && 0 < sum(d) < n ? (gse = _ss_probit(Z, d); gse[1][end] / gse[2][end]) : NaN
    info = (treated_share=mean(d), r2_y1=r2s(y1), r2_y0=r2s(y0), first_stage_z=zst, first_stage_F=zst^2,
            params=P)
    return (df=df, formulas=(formula_y=fy, formula_d=fd), types=types, potential=(y1=y1, y0=y0),
            truth=(sample=_ss_truth(y1, y0, d), population=_ss_population(P; N=pop_N, pop_seed=pop_seed)),
            info=info)
end
