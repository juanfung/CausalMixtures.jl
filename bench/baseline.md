# Phase 0 baseline

Recorded 2026-11-21 on branch `sampler-rewrite`. Code at `e817e7b`, plus the
environment-only fix in commit `08730c4` (`TruncatedNormal` → `truncated(Normal(0, 1), l, u)`).
Machine: Apple M2, Julia 1.12.0, 1 thread. Resolved packages include Distributions 0.25.134,
StatsBase 0.33.21, StatsModels 0.6.33, DataFrames 1.8.3, DataStructures 0.18.22.

## Test suite (`julia --project=. test/runtests.jl`)

| State | Pass | Error | Total |
|---|---|---|---|
| `e817e7b` as is | 106 | 10 | 116 |
| after `08730c4` | 169 | 1 | 170 |

- **At `e817e7b`:** every test that calls `dpm_init` errors with
  `UndefVarError: TruncatedNormal not defined`. Distributions 0.25.1xx removed that constructor.
  The 10 errors are the 5 sampler smoke tests and the 5 sampler integration tests. All
  constructor, prior and data-generation tests pass.
- **After `08730c4`:** one error remains, in "Multi-Sampler Consistency Comparison":
  `KeyError: key 16 not found` in `compute_v` ← `compute_weights!` ← `dpm_init` (blocked).
  The uniform initial labels left component 16 empty, so `countmap` gave it no `njs` key.
  This is plan item 6. The test uses the global RNG without a seed, so whether it fails depends
  on run order.
- `test/testdpm_init.jl` is not included in `runtests.jl`.

The integration tests print these ATE estimates (population ATE is 1 + 8·E[X] = 5; tolerance is ±8):
blocked 2.578, dpm 2.993, fmn 4.515, gaussian 3.767.

## Timing (`julia --project=. bench/baseline.jl`)

`generate_hedonic_data(500, 1234)`, `setup_default_priors(ktot)` (J = 20, α = 1 fixed),
`scale_data = (true, true)`, `Random.seed!(20261121)` before init, one warm-up run first.

| Model | n | M | Time | ms / iter | GC | Allocated | Mean J | Final J |
|---|---|---|---|---|---|---|---|---|
| `dpm` | 500 | 1000 | 6.63 s | 6.63 | 11.6% | 14.33 GiB | 8.33 | 9 |
| `blocked` | 500 | 1000 | 5.78 s | 5.78 | 10.5% | 10.88 GiB | 3.64 | 4 |

The allocation volume (about 11–15 MiB per sweep at n = 500) is the main target for the Phase 3
refactor. Mean J for `dpm` is inflated by the new-cluster bug (review §03, bug 2), which Phase 1
does not touch.
