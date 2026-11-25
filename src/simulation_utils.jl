# src/simulation_utils.jl

"""
Simulation utilities for testing and examples in CausalMixtures.jl
"""

using Random, Distributions, StatsBase, DataFrames, CategoricalArrays, StatsModels, LinearAlgebra

"""
    generate_hedonic_data(n::Int, seed::Int; kwargs...)

Generate synthetic data for a Bayesian hedonic selection model with known treatment effects.

# Arguments
- `n::Int`: Sample size
- `seed::Int`: Random seed for reproducibility

# Keyword Arguments
- `gamma::Vector{Float64}`: Selection equation coefficients (default: [-1.0, 1.0, 1.0])
- `beta_1::Vector{Float64}`: Treatment outcome coefficients (default: [2.0, 10.0])
- `beta_0::Vector{Float64}`: Control outcome coefficients (default: [1.0, 2.0])
- `Sigma_true::Matrix{Float64}`: Error covariance matrix (default: 3x3 with specified correlations)

# Returns
A named tuple containing:
- `df::DataFrame`: Generated dataset with observed outcomes and treatments
- `true_effects::NamedTuple`: True treatment effects (ate, tt, tut)
- `true_params::NamedTuple`: True DGP parameters used
- `formulas::NamedTuple`: StatsModels formulas for outcome and selection equations
- `design_info::NamedTuple`: Additional information about the design
**Legacy** (Phase 6): kept for old scripts and tests. Use `simulate_selection` for new work
(scenarios, sample and population truth, controllable R² and instrument strength).
"""
function generate_hedonic_data(n::Int, seed::Int;
                              gamma::Vector{Float64} = [-1.0, 1.0, 1.0],
                              beta_1::Vector{Float64} = [2.0, 10.0], 
                              beta_0::Vector{Float64} = [1.0, 2.0],
                              Sigma_true::Matrix{Float64} = [1.0 0.7 -0.7; 0.7 1.0 -0.1; -0.7 -0.1 1.0])
    
    # Set random seed for reproducibility
    Random.seed!(seed)
    
    # Generate predictors
    x = rand(n)  # continuous predictor
    z = rand(n)  # instrument
    f = StatsBase.sample(["a", "b"], n)  # categorical factor
    ydummy = rand(0:1, n)  # dummy variable (not used in final model)
    
    # Create initial DataFrame
    df = DataFrame(
        X = x,
        Z = z, 
        F = categorical(f),
        Y_dummy = categorical(ydummy)
    )
    
    # Generate correlated errors
    errs = rand(MvNormal(Sigma_true), n)
    v = errs[1, :]   # selection error
    u1 = errs[2, :]  # treatment outcome error  
    u0 = errs[3, :]  # control outcome error
    
    # Construct design matrices
    formula_d = @formula(Y_dummy ~ X + Z)
    mm_d = ModelMatrix(ModelFrame(formula_d, df))
    
    formula_y = @formula(Y_dummy ~ X) 
    mm_y = ModelMatrix(ModelFrame(formula_y, df))
    
    # Generate latent selection variable
    d_star = mm_d.m * gamma + v
    
    # Generate potential outcomes
    y_1 = mm_y.m * beta_1 + u1  # treatment outcome
    y_0 = mm_y.m * beta_0 + u0  # control outcome
    
    # Observed treatment (selection rule)
    d_obs = [di > 0 ? 1 : 0 for di in d_star]
    
    # Observed outcome (switching equation)
    y_obs = d_obs .* y_1 + (1 .- d_obs) .* y_0
    
    # Add observed variables to DataFrame
    df[!, :Y_obs] = y_obs
    df[!, :D_obs] = d_obs
    
    # Compute true treatment effects
    ate_true = y_1 .- y_0  # individual treatment effects
    tt_true = sum(d_obs .* (y_1 - y_0)) / sum(d_obs)  # treatment on treated
    tut_true = sum((1 .- d_obs) .* (y_1 - y_0)) / sum(1 .- d_obs)  # treatment on untreated
    
    # Create final formulas for estimation
    formula_y_final = @formula(Y_obs ~ X)
    formula_d_final = @formula(D_obs ~ X + Z)
    
    # Compute total number of parameters
    ktot = length(gamma) + length(beta_1) + length(beta_0)

    potential_outcomes = (
        Y1 = y_1,           # Treatment outcomes for all units
        Y0 = y_0           # Control outcomes for all units  
    )
    
    return (
        df = df,
        true_effects = (
            potential_outcomes = potential_outcomes,
            ite = ate_true, # individual treatment effects
            ate = mean(ate_true), # mean ATE
            tt = tt_true, # TT
            tut = tut_true # TUT
        ),
        true_params = (
            gamma = gamma,
            beta_1 = beta_1,
            beta_0 = beta_0, 
            Sigma = Sigma_true,
            ktot = ktot
        ),
        formulas = (
            formula_y = formula_y_final,
            formula_d = formula_d_final
        ),
        design_info = (
            n = n,
            seed = seed,
            n_treated = sum(d_obs),
            n_control = sum(1 .- d_obs)
        )
    )
end

"""
    setup_default_priors(kx, kz; J=20, alpha=1.0, alpha_shape=2.0, alpha_rate=2.0,
                         sd_intercept=1.0, sd_intercept_d=0.5, sd_probit=0.5, slopes=:common,
                         sd_outcome = slopes == :common ? 1.0 : 0.5, rho=10, r=0.153,
                         xintercept=1, zintercept=1)
    setup_default_priors(raw::RawData; kwargs...)

Default priors (candidate C3 of `docs/priors.md`) for a model with `kx` outcome-equation columns
and `kz` selection-equation columns. The coefficient vector is `beta = (gamma, beta1, beta0)`,
of length `ktot = kz + 2kx`, in the column order of the model matrices.

**The defaults are on the standardized scale: run the sampler with `scale_data = (true, true)`.**
On raw data with sd(y) far from 1 they shrink the coefficients too much.

| Parameter | Prior | Why |
|---|---|---|
| selection intercept (`d*`) | N(0, `sd_intercept_d`² = 0.5²) | keeps cluster propensities inside (0.05, 0.95); a cluster that is all treated or all controls identifies its missing arm only through the prior |
| outcome intercepts (`y1`, `y0`) | N(0, `sd_intercept`² = 1) | cluster means of standardized y spread over about ±2 |
| probit slopes | N(0, `sd_probit`² = 0.5²) | with standardized w, about 6% of prior propensity mass outside (0.05, 0.95) (C0: 92%) |
| outcome slopes | N(0, `sd_outcome`²): 0.5² per cluster (`slopes = :cluster`), 1 when shared (`slopes = :common`) | implied sd(y) about 1.1, against 1 in the data; a shared slope is identified by all n units |
| `Sigma` | `NobileWishart(rho, rho*R)`, `rho = 10`, `R = diag(1/rho, r, r)`, `r = 0.153` | prior median within-cluster Var(y1) and Var(y0) of 0.3 |
| `alpha` | Gamma(`alpha_shape` = 2, rate `alpha_rate` = 2), started at `alpha` = 1 | E[k] about 6.6 at n = 500, wide but not degenerate at 1 |

All prior means are zero and the coefficients are independent. `xintercept` and `zintercept`
give the position of the intercept column in each equation (`nothing` if there is none); the
`RawData` method reads them from the formulas. Set `alpha_shape = 0` to fix alpha at `alpha`.
`J` is the truncation level for `blocked` and `fmn`, and the initial number of components for
`dpm` and `marginal`. `slopes` is the slope mode of the Design A sampler (`model = "marginal"`,
`InputParams(slopes = ...)`); both default to `:common`. It sets the outcome-slope default and is
recorded in the priors so that `dpm_init` can warn about a mismatch. Pass `slopes = :cluster` for
per-cluster slopes (outcome slope sd 0.5), with `InputParams(slopes = :cluster)` for `marginal`.

The calibration, its targets and the comparison with the other candidates are in
`docs/priors.md`.
"""
function setup_default_priors(kx::Integer, kz::Integer;
                              J::Integer = 20,
                              alpha::Real = 1.0,
                              alpha_shape::Real = 2.0,
                              alpha_rate::Real = 2.0,
                              sd_intercept::Real = 1.0,
                              sd_intercept_d::Real = 0.5,
                              sd_probit::Real = 0.5,
                              slopes::Symbol = :common,
                              sd_outcome::Real = slopes == :common ? 1.0 : 0.5,
                              rho::Integer = 10,
                              r::Real = 0.153,
                              xintercept::Union{Integer,Nothing} = 1,
                              zintercept::Union{Integer,Nothing} = 1)
    kx >= 1 && kz >= 1 || throw(ArgumentError("kx and kz must be positive, got kx = $kx, kz = $kz"))
    slopes in (:cluster, :common) || throw(ArgumentError("slopes must be :cluster or :common, got $slopes"))
    for (pos, k, nm) in ((xintercept, kx, "xintercept"), (zintercept, kz, "zintercept"))
        pos === nothing || 1 <= pos <= k || throw(ArgumentError("$nm = $pos is not a column in 1:$k"))
    end
    sd_d = fill(Float64(sd_probit), kz)
    zintercept === nothing || (sd_d[zintercept] = sd_intercept_d)
    sd_y = fill(Float64(sd_outcome), kx)
    xintercept === nothing || (sd_y[xintercept] = sd_intercept)
    V0 = Diagonal(vcat(sd_d, sd_y, sd_y) .^ 2)

    prior_dp = PriorDP(J=Int(J), alpha=Float64(alpha), alpha_shape=Float64(alpha_shape),
                       alpha_rate=Float64(alpha_rate))
    prior_beta = PriorBeta(mu=zeros(kz + 2kx), V=Matrix(inv(V0)), Vinv=true)
    prior_sigma = PriorSigma(rho=Int(rho), R=Matrix(Diagonal([1/rho, r, r])))
    return InputPriors(prior_dp=prior_dp, prior_theta=PriorTheta(prior_beta=prior_beta, prior_Sigma=prior_sigma),
                       slopes=slopes)
end

## intercept positions read from the formulas, by column name
function setup_default_priors(raw::RawData; kwargs...)
    xnames = string.(coefnames(ModelFrame(raw.y_form, raw.df)))
    znames = string.(coefnames(ModelFrame(raw.d_form, raw.df)))
    xnames isa AbstractVector || (xnames = [xnames])
    znames isa AbstractVector || (znames = [znames])
    return setup_default_priors(length(xnames), length(znames);
                                xintercept=findfirst(==("(Intercept)"), xnames),
                                zintercept=findfirst(==("(Intercept)"), znames), kwargs...)
end

"""
    setup_default_priors(ktot::Int; alpha=1.0, J=20, beta_nu=100.0, rho=6.0, r=2.0)

Deprecated: the pre-calibration prior (candidate C0 of `docs/priors.md`), beta ~ N(0, beta_nu I),
`Sigma ~ NobileWishart(rho, rho*diag(1/rho, r, r))`, alpha fixed. It is kept unchanged for old
scripts. Use `setup_default_priors(kx, kz)` or `setup_default_priors(raw::RawData)`.
"""
function setup_default_priors(ktot::Int;
                             alpha::Float64 = 1.0,
                             J::Int = 20,
                             beta_nu::Float64 = 100.0,
                             rho::Float64 = 6.0,
                             r::Float64 = 2.0)
    @warn "setup_default_priors(ktot) is deprecated: it returns the old diffuse prior (C0 in " *
          "docs/priors.md). Use setup_default_priors(kx, kz) or setup_default_priors(raw::RawData) " *
          "for the calibrated defaults."
    
    # Beta prior
    beta_mu = zeros(ktot)
    beta_V = beta_nu * Matrix(1.0I, ktot, ktot)
    
    # Sigma prior  
    R = Matrix(Diagonal([1/rho, r, r]))
    
    # Construct prior objects
    prior_dp = PriorDP(alpha=alpha, J=J, alpha_shape=0.0, alpha_rate=0.0)
    prior_beta = PriorBeta(mu=beta_mu, V=inv(beta_V), Vinv=true)
    prior_sigma = PriorSigma(rho=rho, R=R)
    prior_theta = PriorTheta(prior_beta=prior_beta, prior_Sigma=prior_sigma)
    
    return InputPriors(prior_dp=prior_dp, prior_theta=prior_theta)
end
