# CausalMixtures.jl

Bayesian nonparametric causal inference using Dirichlet Process Mixtures,
inspired by:

Li, M., Poirier, D.J. and Tobias, J.L. (2004), Do dropouts suffer from dropping
out? Estimation and prediction of outcome gains in generalized selection models.
J. Appl. Econ., 19: 203-225. https://doi.org/10.1002/jae.731

## Quick Reference - Working Workflow

### 1. Generate/Prepare Data
```julia
using CausalMixtures, Random

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
    model="blocked",           # ✅ RELIABLE (or "dpm")
    # model="marginal"         # Design A (Phase 4a, in validation); slopes=:common or :cluster
    thin=1,                    # save (theta_j, n_j, alpha) every thin-th iteration
    latent_thin=0              # save latent data and labels every k-th iteration (0 = never)
)

# Initialize and run, with an explicit random number generator (default: the global RNG)
rng = Xoshiro(2026)
init = dpm_init(raw_data, priors, params; rng=rng)
state, input, output = dpm!(init...; rng=rng)
```

### 4. Calculate Treatment Effects
```julia
# Per-draw, label-invariant average effects on the original y scale (discard burn-in draws)
eff = effects(output, input; rng=rng, burn=1000)
s = effects_summary(eff)          # or effects_summary([eff1, eff2, ...]) over several chains

# Posterior mean and 95% credible interval of the population ATE, ATT and ATU
for (name, v) in pairs(s)
    v === nothing && continue      # satt/satu need saved latent draws (latent_thin > 0)
    println("$(name): $(round(v.mean, digits=3))  95% CI $(round.(v.q[[1, 3]], digits=3))")
end
```

`effects` returns, per saved draw, PATE (the average of x'(β1 − β0) over the sample's
covariates), PATT and PATU (the closed-form TT and TUT averaged over the treated and untreated
units' own covariates), and, when latent draws are saved, the sample SATT and SATU. Their
quantiles are **credible intervals** for these averages.

The older `rand_ppd` + `dpm_ate` route gives something different: draws from the **predictive
distribution of one new unit's gain** y1 − y0 at chosen covariates. Its spread includes that unit's
own errors and depends on σ23, which the data do not identify, so its quantiles are predictive
intervals for an individual, not confidence or credible intervals for the ATE:
```julia
xbar, wbar = mean_covariates(input)
gains = dpm_ate(rand_ppd(rng, output, input, xbar, wbar), input)   # one new unit's gain
```

Partition summaries (with `latent_thin > 0`): `partition_summary(output; burn)` gives the posterior
similarity matrix, a point partition (minimum expected Binder loss), the posterior of k and the
separated-units share.

## Key Functions Reference

| Function | Purpose | Notes |
|----------|---------|-------|
| `generate_hedonic_data(n, seed)` | Simulate test data | Returns data object with true effects |
| `setup_default_priors(raw_data)` | Default priors (C3, standardized scale) | or `setup_default_priors(kx, kz)`; the old `setup_default_priors(ktot)` is deprecated |
| `dpm_init()` | Initialize sampler | Returns (state, input, output) |
| `dpm!()` | Run MCMC sampler | Main sampling function |
| `effects()` | Per-draw PATE, PATT, PATU (and SATT, SATU) | Original y scale; `effects_summary()` for credible intervals |
| `partition_summary()` | Similarity matrix, point partition, posterior of k | Needs `latent_thin > 0` |
| `rand_ppd()` | Predictive draws of a new unit's (d*, y1, y0) | Size: (3 × M) matrix |
| `dpm_ate()` | A new unit's predictive gain y1 − y0 | Predictive intervals (depend on σ23), not ATE credible intervals |

## Sampler Status

| Sampler | Status | When to Use |
|---------|--------|-------------|
| `"blocked"` | ✅ **Reliable** | Default choice, stable |
| `"dpm"` | ✅ **Reliable** | More flexible, research use |
| `"marginal"` | 🧪 **In validation** | Design A (`docs/implementation_plan.md` §06): β integrated out of the label step; `slopes=:common` (default) or `:cluster`. Use `setup_default_priors(raw_data; slopes=:common)` |

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
│   ├── dpm_marginal.jl        # Design A sampler (model = "marginal")
│   ├── dpm_ppd.jl             # rand_ppd(), dpm_ate()
│   ├── gibbsinput.jl
│   ├── gibbsout.jl
│   ├── gibbsppd.jl
│   ├── gibbsstate.jl
│   ├── misc_functions.jl
│   ├── sampler_functions.jl    # MCMC sampler helper functions
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
