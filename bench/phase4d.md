# Phase 4d: hierarchical intercepts and the separated-units diagnostic

Recorded 2026-11-24 on an Apple M2, Julia 1.12.0. The outputs are `geweke_4d_long.md` and
`phase4d_compare_output.md`.

## What changed

- **Hierarchical intercepts** (`InputParams(hierarchical = true)`, the default; Design A only).
  - The cluster intercepts a_j = (a_d, a_1, a_0)_j ~ N(m, diag(τ²)), with
    m ~ N(0, V_m), V_m = diag(0.5², 1, 1), and τ²_l ~ IG(3, 2 s_l²), s² = (0.5², 1, 1).
  - The hyperpriors are read from the fixed intercept prior, so C3 gives exactly these values.
  - Under `:cluster` only the intercepts are hierarchical; the slopes keep their C3 normals.
- **Where (m, T) is used.** It is the intercept block of the β^c prior everywhere it appears:
  - the collapsed predictive of existing clusters;
  - the auxiliary clusters' predictive;
  - the β^c update.

  `set_cprior!` rewrites Design A's prior fields in place, and `sync_hyper!` runs at the start
  of each sweep.
- **New last step (vii):**
  - m_l ~ N((μ_l/V_l + Σ_j a_lj/τ²_l)/P, 1/P), with P = 1/V_l + k/τ²_l;
  - τ²_l ~ IG(3 + k/2, 2 s_l² + Σ_j (a_lj − m_l)²/2).
- **State and output.**
  - The state holds the hyperparameters in `state_hyper`.
  - Each saved draw stores `out_hyper[s] = vcat(m, τ²)`.
  - `rand_marginal` draws new clusters from that draw's (m, τ²).
- **Separated-units diagnostic.**
  - `out_sep[s]` is the share of units in clusters that are under 5% or over 95% treated. It is
    saved by every sampler.
  - `dpm!` warns after a `marginal` run when its posterior mean exceeds 5%.
- **R3** is now off by default (`sigma23_step = false`).

## Tests

**Unit tests** (`test/test_hierarchical.jl`):
- m | a, τ² matches the dense closed form, in mean and variance.
- τ² | a, m passes a probability-integral-transform KS test against IG(3 + k/2, …), in both slope
  modes.
- **Invariance:** with (m, τ²) drawn from the hyperprior and a_j ~ N(m, T), one update leaves the
  hyperprior invariant. Shape 3 + k fails the same check (|z| > 10).
- `hierarchical = false` keeps the fixed prior.
- The diagnostic, its storage and the warning are tested.

**Geweke** (`bench/geweke.jl`, 500k sweeps against 500k forward draws, n = 20). m_d, m_1, m_0 and
log τ²_d, log τ²_1, log τ²_0 are added as statistics, together with their squares.

| Sampler | Statistics | max \|z\| | Largest | Hyperparameter statistics |
|---|---|---|---|---|
| `marginal`, `:common` | 44 | 2.12 | s33² +2.12 | all within ±1.69 |
| `marginal`, `:cluster` | 36 | 2.88 | mean(a1 − a0)² +2.88 | all within ±1.63 |
| `gaussian` | 18 | 2.13 | s23² (identical to 4b: R3 off) | — |
| `blocked` | 24 | 1.79 | a_1² | — |

**Planted bugs** (`bench/geweke_power.jl`, 50k sweeps):

| Planted bug | `:common` | `:cluster` |
|---|---|---|
| None (control) | 1.4 | 2.1 |
| (a) τ² shape 3 + k instead of 3 + k/2 | **93.7** (log τ²_d) | **92.8** (log τ²_1) |
| (b) label step under the fixed (0, V_m), β^c update under (m, T) | **34.8** (m_1²) | **33.3** (m_1²) |
| Earlier: Φ dropped | 14.1 | 13.0 |
| Earlier: shift g_d dropped | 76.1 | 2.1 (negative control) |
| Earlier: Σ dof | 11.8 | 7.7 |
| Earlier: no H Q⁻¹ H′ | 7.8 | 14.3 |
| Earlier: Escobar–West indicator | 44.8 | 42.9 |

Both requested bugs are real errors, and both fail:
- **(a)** has the wrong posterior shape.
- **(b)** scores labels and opens clusters under one β^c prior while drawing β^c under another,
  so the label step no longer targets its conditional.

**Suite:** 2146/2146.

## Hierarchical on and off

`bench/phase4d_compare.jl`: 8 chains × 3000 iterations (1000 burn-in), R2, C3. The two-component
data is Phase 2's (`bench/prior_calibration.jl`).

| Data (sample ATE) | Slopes | Hier. | ATE mean (sd) | ATE ESS / 1000 | R-hat ATE | k | R-hat k | Separated share | m | τ² |
|---|---|---|---|---|---|---|---|---|---|---|
| hedonic (5.155) | `:common` | off | 5.264 (0.237) | 17.5 | 1.025 | 1.12 | 1.001 | 0.000 | — | — |
| hedonic (5.155) | `:common` | on | 5.280 (0.236) | 13.4 | 1.040 | 1.14 | 1.000 | 0.000 | 0.02, 0.35, −0.50 | 0.22, 0.90, 0.91 |
| hedonic (5.155) | `:cluster` | off | 5.267 (0.228) | 16.7 | 1.025 | 1.10 | 1.001 | 0.000 | — | — |
| hedonic (5.155) | `:cluster` | on | 5.267 (0.241) | 16.1 | 1.023 | 1.10 | 1.000 | 0.000 | 0.02, 0.34, −0.50 | 0.22, 0.90, 0.90 |
| two-component (4.920) | `:common` | off | 5.418 (0.455) | 6.7 | 1.065 | 1.28 | 1.123 | 0.000 | — | — |
| two-component (4.920) | `:common` | on | 5.452 (0.427) | 7.1 | 1.063 | 1.37 | 1.125 | 0.002 | −0.06, 0.34, −0.52 | 0.23, 0.87, 0.88 |
| two-component (4.920) | `:cluster` | off | 5.478 (0.415) | 10.5 | 1.056 | 1.18 | 1.005 | 0.000 | — | — |
| two-component (4.920) | `:cluster` | on | 5.503 (0.419) | 7.2 | 1.052 | 1.19 | 1.001 | 0.000 | −0.05, 0.33, −0.52 | 0.22, 0.89, 0.90 |

The m and τ² columns are posterior means, in the order (d, 1, 0).

- **Little changes with the hierarchy on.**
  - The ATE moves by 0.00–0.035, which is well inside its sd.
  - k moves by at most 0.09.
  - ATE ESS changes by no more than the seed-to-seed spread seen in Phase 4c (up to about 40%).
- **Why.** With C3 and these data the posterior has k ≈ 1.1–1.4, so there is about one cluster
  to learn (m, T) from.
  - τ² stays near its prior mean: 0.22–0.23 against 0.25 for the selection intercept, and
    0.87–0.91 against 1 for the outcome intercepts.
  - m moves toward the data's intercepts.
  - The hierarchy is in place for data with more clusters, and costs nothing here.
- **Separated share.** It is 0–0.2%, far below the 5% warning level, both on and off. The C3
  probit priors already prevent the separation that C0 and C1 produced in Phase 2.
- **Two-component data.**
  - Design A puts almost everything in one cluster (k 1.2–1.4). One `:common` chain visits
    k ≈ 2 (chain means 1.72 and 2.07), hence R-hat for k of 1.12.
  - The ATE is 5.42–5.50 against a sample ATE of 4.92: about 1.2 posterior sds high, in both
    modes and with or without the hierarchy.
  - The blocked C3 runs of Phase 2 gave 5.24–5.49 on the same data.
