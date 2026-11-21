# Phase 4a: Design A core and Geweke harness, exit checks

Recorded 2026-11-21 on an Apple M2, Julia 1.12.0, 1 thread, at the commit that adds this file.
The full outputs are in `geweke_4a_long.md`, `phase4a_checks_output.md` and
`phase4a_k_long_output.txt`.

## (a) Geweke "getting it right"

`bench/geweke.jl` (harness in `test/GewekeHarness.jl`) used n = 20 units with the standardized
covariates of `generate_hedonic_data(20, 1)`, C3 priors, 500,000 sampler sweeps and 500,000
independent forward draws from `prior_draw`.

Each test is z = (forward mean − sampler mean) / se. The sampler's Monte Carlo error comes from
its ESS (`ess_mean`). Every statistic is tested with its square.

| Sampler | Statistics | max \|z\| | Largest |
|---|---|---|---|
| `marginal`, `:common` | 32 | 2.09 | b2² 2.09, s12² 1.48 |
| `marginal`, `:cluster` | 24 | 2.42 | s33 2.42, s13² 2.41 |
| `gaussian` (sanity) | 18 | 2.13 | s23² 2.13 |
| `blocked` (sanity) | 24 | 1.79 | a_1² −1.79 |

Per-statistic maximum |z| for Design A (both moments, worse of the two):

| Statistic | `:common` | `:cluster` |
|---|---|---|
| α | 1.26 | 0.43 |
| k | 0.76 | 0.28 |
| a_d, a_1, a_0 (unit 1's cluster) | 0.48, 0.42, 1.11 | 0.51, 1.26, 0.55 |
| σ12, σ13, σ22, σ23, σ33 (unit 1's cluster) | 1.48, 0.38, 1.11, 0.71, 0.62 | 1.10, 2.41, 1.11, 1.28, 2.42 |
| shared slopes b1–b4 | 1.06, 2.09, 0.42, 0.32 | — |
| mean_i (a_1 − a_0) | 1.19 | 1.42 |
| share of units in unit 1's cluster | 0.71 | 0.37 |

**Power check** (50k sweeps). Each planted bug was a method redefined in a throwaway script:

| Planted bug | `:common` max \|z\| | `:cluster` max \|z\| |
|---|---|---|
| Σ_j with ρ + n_j + 1 degrees of freedom | 11.5 (σ22) | 10.4 (σ22) |
| Collapsed predictive without the H Q⁻¹ H′ term | 14.7 (k², α) | 37.6 (k) |
| Escobar–West without the random indicator | 33.4 (α) | 31.1 (α) |
| None (control) | 2.4 | 1.8 |

The default suite runs a smoke version: 10k sweeps, first moments only, |z| < 5.
`CM_GEWEKE=1` runs the long version. At a few thousand sweeps, the squares of the heavy-tailed
Σ entries are unreliable: s23² gave z = 6.2 at 4k sweeps and 2.15 at 100k on the same
configuration. That is why the smoke version skips squares.

## (b)–(d) On `generate_hedonic_data(500, 1234)`

`bench/phase4a_checks.jl` ran 8 chains per configuration, M = 3000 with a burn-in of 1000, C3 priors
and `scale_data = (true, true)`. ATE is the per-draw `trace_ate`. R-hat is the rank-normalized
split R-hat; ESS is the bulk ESS over the 8 chains.

| Configuration | ATE mean ± se | ATE sd | k | α | R-hat ATE | R-hat k | R-hat α | bulk ESS ATE |
|---|---|---|---|---|---|---|---|---|
| `gaussian` | 5.263 ± 0.015 | 0.216 | 1.00 | — | 1.033 | — | — | 177 |
| `marginal`, α = 1e-12 fixed, J = 1 | 5.244 ± 0.018 | 0.222 | 1.00 | — | 1.056 | — | — | 156 |
| `marginal`, `:cluster` | 5.229 ± 0.019 | 0.237 | 1.15 | 0.26 | 1.035 | 1.068 | 1.005 | 197 |
| `marginal`, `:common` | 5.205 ± 0.050 | 0.283 | 1.16 | 0.26 | 1.144 | 1.095 | 1.006 | 41 |
| `blocked` (J = 20) | 5.190 ± 0.053 | 0.290 | 2.65 | 0.79 | 1.122 | 1.479 | 1.475 | 46 |

Sample ATE 5.155; population ATE 5.

**(b) α fixed tiny, so k = 1 in every draw, against `gaussian`.**

| | ATE mean | ATE sd | σ22 | σ33 | σ12 | σ13 | σ23 |
|---|---|---|---|---|---|---|---|
| z | −0.75 | +0.45 | +0.67 | −0.62 | +0.62 | +0.72 | −0.84 |

**(c) `marginal` (`:cluster`) against `blocked`.**

| | ATE mean | ATE sd | σ22 | σ33 | σ12 | σ13 | σ23 |
|---|---|---|---|---|---|---|---|
| z | +0.69 | −0.85 | −1.15 | −3.51 | −1.06 | +1.20 | −0.21 |

- The ATE and its sd agree.
- k and α do not: `blocked` gives 2.65 and 0.79 with R-hat 1.48; Design A gives 1.15 and 0.26 with
  R-hat 1.005.
- The σ33 gap follows from k, since σ is averaged with n_j weights over a different number of
  clusters.

**Long runs** (`phase4a_k_long.jl`: 4 chains × 30,000 iterations each, thinned by 10, the first
3,000 iterations discarded):
- Design A gives k = 1.10 (R-hat 1.001) and α = 0.25 (R-hat 1.000), the same in every chain and
  quarter.
- `blocked`'s chains drift down and disagree: final-quarter k of 1.09, 1.55, 1.42 and 2.66, R-hat
  1.32. Its one well-mixed chain matches Design A.
- The disagreement on k is therefore `blocked` mixing slowly over partitions, not a difference
  between the models.

**(d) `:common` against `:cluster`.**

| | ATE mean | ATE sd | σ22 | σ33 | σ12 | σ13 | σ23 |
|---|---|---|---|---|---|---|---|
| z | −0.45 | +0.78 | −0.71 | +1.12 | −0.32 | +0.29 | +0.18 |

- ATE 5.205 vs 5.229; k 1.16 vs 1.15.
- One `:common` chain out of 8 stuck in the low-ATE mode: ATE 4.87, sd 0.69, k 1.42. That pushes
  `:common`'s R-hat for ATE to 1.14.
- No `:cluster` chain did.

## (e) Timing

`bench/phase4a_timing.jl`: n = 500, C3, median of 3 runs of M = 1000.

| Sampler | s / 1000 iter | Allocated / 1000 iter |
|---|---|---|
| `blocked` (J = 20) | 0.39 | 0.59 GiB |
| `marginal`, `:cluster` | 2.54 | 5.11 GiB |
| `marginal`, `:common` | 2.15 | 4.01 GiB |
| `gaussian` | 0.07 | 0.08 GiB |

Design A is 5.5–6.5× slower per sweep than `blocked`, with about 4–5 MiB allocated per sweep.
Almost all of it is the label step, which costs the same with k = 1 (2.55 ms/iter in (b)).

The per-unit work is:
- two auxiliary `NobileWishart` draws;
- two refactorizations of Q (the cluster the unit leaves and the one it joins);
- one `Q⁻¹H′` solve per candidate cluster.

None of this was optimized: 4b replaces the scoring with the R2 observed-data score, so it would
be optimized there.
