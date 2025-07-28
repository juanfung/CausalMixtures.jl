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
    
    return (
        df = df,
        true_effects = (
            ate = ate_true,
            tt = tt_true, 
            tut = tut_true,
            ate_mean = mean(ate_true)
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
    setup_default_priors(ktot::Int; kwargs...)

Set up default prior specifications for hedonic model testing.

# Arguments  
- `ktot::Int`: Total number of parameters

# Keyword Arguments
- `alpha::Float64`: DP concentration parameter (default: 1.0)
- `J::Int`: Number of mixture components (default: 20)
- `beta_nu::Float64`: Prior precision scaling (default: 100.0)
- `rho::Float64`: Wishart degrees of freedom (default: 6.0)
- `r::Float64`: Wishart scale parameter (default: 2.0)

# Returns
`InputPriors` object ready for sampler initialization
"""
function setup_default_priors(ktot::Int;
                             alpha::Float64 = 1.0,
                             J::Int = 20,
                             beta_nu::Float64 = 100.0,
                             rho::Float64 = 6.0,
                             r::Float64 = 2.0)
    
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
