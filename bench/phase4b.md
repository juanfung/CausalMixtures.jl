# Phase 4b: R2 joint label-and-latent step, exit checks

Recorded 2026-11-22 on an Apple M2, Julia 1.12.0, 1 thread. The full outputs are in
`geweke_4b_long.md`, `phase4b_mixing_output.md` and `phase4b_timing_output.md`.

## Correction to plan §06 (confirmed by the coordinator)

As written, §06 scores and truncates on the shifted scale z̃ = z − H^g_i b with the cut-off at 0.
Treatment is d = 1{d* > 0}, and d* = z̃_d + g_d with g_d = (H^g_i b)_d, so on that scale the
cut-off is −g_d.

The score is

  log N(ỹ | μ_o, S_oo) + log Φ( s (μ_d + g_d + S_do/S_oo (ỹ − μ_o)) / √(S_dd − S_do²/S_oo) )

and z̃_d is drawn truncated at −g_d.

Under `:cluster`, g = 0 and §06 is unchanged. Under `:common`, the formula as written failed
Geweke with max |z| = 62.3 (a_d −62.3, b2 −31.7). The corrected step passes; see below.

## (1)–(2) Unit tests (`test/test_designA.jl`)

- **Score.** The R2 score equals the log of the predictive integrated over the unobserved
  outcome and the half-line of z̃_d beyond −g_d. This uses the trapezoid rule, atol 1e-6, with 12
  random (μ, S) for each d and g_d ∈ {0, random}. A far-tail case stays finite.
- **Draws.** The (z̃_d, y_miss) draws match rejection sampling from the Gaussian conditional on
  ỹ_o with the cut-off at −g_d, for g_d ∈ {0, 0.6, −0.9}. The comparison covers means, centred
  squares and the correlation.

## (3) Geweke "getting it right"

`bench/geweke.jl` uses the same set-up as Phase 4a: n = 20, C3 priors, 500,000 sweeps and 500,000
independent forward draws. Each statistic is tested together with its square. R2 is the default
label step.

| Sampler | Statistics | max \|z\| | Largest |
|---|---|---|---|
| `marginal`, `:common` | 32 | 1.98 | mean(a1 − a0) −1.98, b3 −1.86, a_0 +1.82 |
| `marginal`, `:cluster` | 24 | 2.11 | s13 +2.11, a_d² +1.81, k² +1.57 |
| `gaussian` (sanity) | 18 | 2.13 | s23² +2.13 |
| `blocked` (sanity) | 24 | 1.79 | a_1² −1.79 |

**Power check** (`bench/geweke_power.jl`, 50k sweeps against 100k forward draws, with one planted
bug per run):

| Planted bug | `:common` max \|z\| | `:cluster` max \|z\| |
|---|---|---|
| None (control) | 1.8 | 2.3 |
| Global shift g_d dropped: the cut-off at 0, as §06 had it | **62.3** (a_d, b2) | 2.3, identical to the control because g = 0 (negative control) |
| Φ term dropped from the R2 score | 14.9 (a_d²) | 11.9 (a_d²) |
| Label-step predictive without the H Q⁻¹ H′ term | 7.2 (k²) | 16.5 (k) |
| Σ_j with ρ + n_j + 1 degrees of freedom | 10.5 (σ33) | 9.2 (σ22) |
| Escobar–West without the random indicator | 44.6 (α) | 42.8 (α) |

Every planted bug fails. Dropping the shift fails only under `:common`, as it should.

## (4) Mixing: the 4a label step against R2

`bench/phase4b_mixing.jl` ran on `generate_hedonic_data(500, 1234)` with C3 priors and
`scale_data = (true, true)`: 16 chains × 3000 iterations, 1000 of them burn-in, with chain seeds
20261121 + c.

- ESS is the bulk ESS of the per-draw ATE per 1000 post-burn-in draws, pooled over 32,000 draws.
- R-hat is the rank-normalized split R-hat.
- "Stuck" counts chains whose mean ATE falls (A) more than 3 pooled sds, or (B) more than 1.5
  median within-chain sds, below the median of the chain means.

| Label step | Slopes | ATE | k | ATE ESS / 1000 | R-hat ATE | R-hat k | R-hat α | stuck (A) / (B) | lowest chain ATE |
|---|---|---|---|---|---|---|---|---|---|
| 4a (`:augmented`) | `:common` | 5.238 | 1.16 | 10.4 | 1.039 | 1.016 | 1.001 | 0 / 0 | 5.15 |
| 4a (`:augmented`) | `:cluster` | 5.235 | 1.15 | 4.0 | 1.095 | 1.145 | 1.007 | 0 / 0 | 5.05 |
| R2 | `:common` | 5.269 | 1.12 | 19.2 | 1.021 | 1.001 | 1.000 | 0 / 0 | 5.17 |
| R2 | `:cluster` | 5.268 | 1.10 | 15.2 | 1.031 | 1.000 | 1.000 | 0 / 0 | 5.18 |

Sample ATE 5.155; population ATE 5.

- **ESS.** R2 has about 1.8× the ATE ESS of the 4a step under `:common` and 3.8× under
  `:cluster`.
- **R-hat for k.** The 4a step's R-hat for k is 1.145 under `:cluster`; R2's is 1.000.
- **Posterior means.**
  - The ATE means differ by 0.03 (4a 5.24, R2 5.27). Using the Monte Carlo standard errors
    implied by the ESS and Phase 4a's posterior sds (0.28 and 0.24), that is z ≈ 1.6 under
    `:common` and 1.4 under `:cluster`, so it is not significant.
  - k is 1.10–1.12 under R2 and 1.15–1.16 under 4a.
- **Stuck chains.** No chain was stuck in any configuration, so this run says nothing about the
  low-ATE mode: 1 of 8 4a `:common` chains was stuck in Phase 4a, with different seeds. The 4a
  `:cluster` chains have the widest spread of chain means (lowest chain 5.05, R-hat 1.095).

The ms/iter column in `phase4b_mixing_output.md` was measured while other jobs were running.
Use (5) for timing.

## (5) Timing and the label-step optimization

The optimizations, all in `update_marginal_labels!`, are:
- **Q⁻¹ cache.** Each slot keeps Q⁻¹ and m = Q⁻¹r. Q depends on the design and Σ only, not on
  z̃, so a unit that returns to its own cluster gets the pre-removal Q⁻¹ back, and only r and m
  are recomputed. Q is refactored in place (potrf/potri) only on a removal and on a move.
- **In-place predictive.** `predictive!` builds Σ + H^c Q⁻¹ H^c′ from the block structure without
  allocating.
- **Precomputed Wishart factor.** The two auxiliary G0 draws use a `NobileFactor` of ρR, which is
  precomputed once per sweep.

The general rank-3 updates and the separate 3×3 closed form for `:common` were not needed: Q has
size kc = 3 (`:common`) or 7 (`:cluster`), and the cache already takes away most of the cost.

On the same RNG stream as the previous step, the new step gives identical labels and latent data
equal to within 5e-15, in both label steps and both slope modes.

`bench/phase4b_timing.jl` used the same protocol as Phase 4a: n = 500, C3 priors, the median of 3
runs of M = 1000, on an idle machine.

| Sampler | s / 1000 iter | Allocated / 1000 iter | Phase 4a (s / 1000 iter) |
|---|---|---|---|
| `blocked` (J = 20) | 0.41 | 0.61 GiB | 0.39 |
| `marginal`, `:common`, 4a step | 0.92 | 1.23 GiB | 2.15 |
| `marginal`, `:common`, R2 | 0.96 | 1.20 GiB | — |
| `marginal`, `:cluster`, 4a step | 1.11 | 1.18 GiB | 2.54 |
| `marginal`, `:cluster`, R2 | 1.14 | 1.15 GiB | — |

- R2 costs about the same as the 4a step: about 4% more per iteration, for the logΦ and the
  truncated-normal draw.
- With the optimization, Design A is 2.3–2.8× slower per iteration than `blocked`, down from
  5.5–6.5× in Phase 4a.
- The label step is still almost all of the sweep. The other updates take under 0.2 ms per
  iteration. Most of what is left is allocation in the two `NobileWishart` draws per unit.

## (6) Suite

`julia --project=. test/runtests.jl` passes 1876/1876 tests, including the Geweke smoke runs.
