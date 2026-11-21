# CausalMixtures.jl

Bayesian nonparametric causal inference using Dirichlet Process Mixtures,
inspired by:

Li, M., Poirier, D.J. and Tobias, J.L. (2004), Do dropouts suffer from dropping
out? Estimation and prediction of outcome gains in generalized selection models.
J. Appl. Econ., 19: 203-225. https://doi.org/10.1002/jae.731

## Quick Reference - Working Workflow

### 1. Generate/Prepare Data
```julia
using CausalMixtures

# Option A: Use simulation utility (for testing)
data = CausalMixtures.generate_hedonic_data(500, 1234)  # n=500, seed=1234

# Option B: Real data - needs DataFrame with:
# - Y_obs: continuous outcome
# - D_obs: binary treatment (0/1) 
# - X, Z: covariates/instruments
```

### 2. Set Up Model
```julia
# Define formulas
formula_y = @formula(Y_obs ~ X)        # Outcome equation
formula_d = @formula(D_obs ~ Z + X)    # Selection equation

# Create data object
raw_data = RawData(formula_y, formula_d, data.df)

# Set up priors: the calibrated defaults (docs/priors.md), on the standardized scale,
# so run with scale_data=(true,true). Intercept columns are found by name.
priors = setup_default_priors(raw_data)
# equivalently setup_default_priors(kx, kz) with kx = 2 outcome and kz = 3 selection columns
```

### 3. Run Sampler
```julia
# Configure sampler - USE THESE RELIABLE OPTIONS:
params = InputParams(
    M=1000,                    # MCMC iterations
    scale_data=(true,true),    # Scale data
    verbose=true,              # Show progress
    model="blocked"            # ✅ RELIABLE (or "dpm")
    # model="marginal"         # ❌ UNSTABLE - avoid!
)

# Initialize and run
init = dpm_init(raw_data, priors, params)
state, input, output = dpm!(init...)
```

### 4. Calculate Treatment Effects
```julia
# Set prediction point: covariate means over all n units, matched to each equation by name
xbar, wbar = mean_covariates(input)

# Generate posterior predictive draws
ynew = rand_ppd(output, input, xbar, wbar)

# Calculate treatment effects
tes = dpm_ate(ynew, input)

# Results
println("ATE: $(round(mean(tes.ate), digits=3))")
println("95% CI: $(round.(quantile(tes.ate, [0.025, 0.975]), digits=3))")
```

## Key Functions Reference

| Function | Purpose | Notes |
|----------|---------|-------|
| `generate_hedonic_data(n, seed)` | Simulate test data | Returns data object with true effects |
| `setup_default_priors(raw_data)` | Default priors (C3, standardized scale) | or `setup_default_priors(kx, kz)`; the old `setup_default_priors(ktot)` is deprecated |
| `dpm_init()` | Initialize sampler | Returns (state, input, output) |
| `dpm!()` | Run MCMC sampler | Main sampling function |
| `rand_ppd()` | Posterior predictive draws | Size: (3 × M) matrix |
| `dpm_ate()` | Calculate treatment effects | Returns ATE, ATT, ATC |

## Sampler Status

| Sampler | Status | When to Use |
|---------|--------|-------------|
| `"blocked"` | ✅ **Reliable** | Default choice, stable |
| `"dpm"` | ✅ **Reliable** | More flexible, research use |
| `"marginal"` | ⚠️ **Deprecated** | Do not use; to be replaced by the Design A sampler (`docs/implementation_plan.md`) |

## Testing

```julia
# Run all tests
include("test/runtests.jl")

```

## File Structure
```
├── examples
│   └── hedonic_selection
├── git_push_all.sh
├── README.md
├── src
│   ├── CausalMixtures.jl
│   ├── dpm_blocked.jl
│   ├── dpm_fmn.jl
│   ├── dpm_gaussian.jl
│   ├── dpm_gibbs.jl
│   ├── dpm_init.jl            # Initialization functions
│   ├── dpm_marginal.jl
│   ├── dpm_ppd.jl             # rand_ppd(), dpm_ate()
│   ├── gibbsinput.jl
│   ├── gibbsout.jl
│   ├── gibbsppd.jl
│   ├── gibbsstate.jl
│   ├── misc_functions.jl
│   ├── sampler_functions.jl.   # MCMC sampler helper functions (blocked, dpm work; marginal broken)
│   └── simulation_utils.jl     # generate_hedonic_data(), setup_default_priors()
└── test
    ├── integration
    │   └── test_hedonic_model.jl # integreation tests for working samplers
    ├── runtests.jl             # run suite of unit tests + integration tests
    ├── testCausalMixtures.jl
    ├── testdpm_init.jl
    ├── testgibbsinput.jl
    ├── testgibbsout.jl
    ├── testgibbsppd.jl
    └── testgibbsstate.jl
```

## Notes

- **Always use `"blocked"` sampler** unless you need `"dpm"` flexibility
- **ktot = length of parameter vector** (gamma + beta_1 + beta_0)
- **PPD matrix is (3 × M)**: [selection, treated outcome, control outcome]
