# CausalMixtures.jl

Bayesian nonparametric causal inference using Dirichlet Process Mixtures.

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

# Set up priors (use defaults)
ktot = 7  # Number of parameters (adjust based on your model)
priors = setup_default_priors(ktot, beta_nu=100.0, rho=6.0, r=2.0)
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
# Set prediction point (usually sample mean)
znew = mean(input.data.Hmat[1:100, 1:3], dims=1)'

# Generate posterior predictive draws
ynew = rand_ppd(output, input, znew[:,1])

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
| `setup_default_priors(ktot)` | Create default priors | ktot = # parameters in model |
| `dpm_init()` | Initialize sampler | Returns (state, input, output) |
| `dpm!()` | Run MCMC sampler | Main sampling function |
| `rand_ppd()` | Posterior predictive draws | Size: (3 × M) matrix |
| `dpm_ate()` | Calculate treatment effects | Returns ATE, ATT, ATC |

## Sampler Status

| Sampler | Status | When to Use |
|---------|--------|-------------|
| `"blocked"` | ✅ **Reliable** | Default choice, stable |
| `"dpm"` | ✅ **Reliable** | More flexible, research use |
| `"marginal"` | ❌ **Unstable** | Avoid until Julia 1.6+ |

## Common Issues & Solutions

**Wrong ATE sign/magnitude:**
- Check treatment coding (1=treated, 0=control)
- Verify formula specifications
- Try different sampler

**High variance in results:**
- Increase MCMC iterations (`M`)
- Check for convergence issues
- Consider different priors

**Sampler errors:**
- Use `"blocked"` instead of `"marginal"`
- Check data format (no missing values)
- Verify ktot matches model parameters

## Testing

```julia
# Run integration tests
include("test/integration/test_hedonic_model.jl")

# Should see: "32 tests passed"
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
│   ├── parallel_ppd.jl
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

## Notes for Future Me

- **Always use `"blocked"` sampler** unless you need `"dpm"` flexibility
- **The marginal sampler is fundamentally broken** - gives different results with same inputs
- **Integration tests prove blocked/dpm work reliably** 
- **ktot = length of parameter vector** (gamma + beta_1 + beta_0)
- **PPD matrix is (3 × M)**: [selection, treated outcome, control outcome]
=======
