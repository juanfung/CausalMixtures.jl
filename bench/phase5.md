# Phase 5: outputs (per-draw effects, partition summaries, predictive docs)

Recorded 2026-11-25 on an Apple M2, Julia 1.12.0. The raw output is in
`phase5_effects_output.md` and the script in `phase5_effects.jl`.

## API

**`effects(out, input; rng, n_g0 = 20, burn = 0, weights = :dirichlet) -> Effects`** returns
per-draw, label-invariant effects on the original y scale.

Component weights per saved draw:
- `marginal` and `dpm`: (π_1..π_k, π_0) ~ Dir(n_1..n_k, α). π_0 is spread over `n_g0` fresh G0
  draws; for `marginal`, G0 uses the draw's (m, τ²) and its shared b.
- `blocked`: the stick weights.
- `fmn`: n_j/n.
- `gaussian`: π = 1.

Per-draw effects:
- **PATE** = (1/n) Σ_i Σ_j π_j x_i′(β1_j − β0_j).
- **PATT and PATU:** the closed-form mixture TT and TUT, averaged over the treated and untreated
  units' own covariates. With σ11 = 1 and Φ_j, φ_j at w′γ_j:
  - TT = Σ π_j Φ_j[Δ_j + (σ12 − σ13)_j φ_j/Φ_j] / Σ π_j Φ_j;
  - TUT = Σ π_j (1 − Φ_j)[Δ_j − (σ12 − σ13)_j φ_j/(1 − Φ_j)] / Σ π_j (1 − Φ_j).
- **SATT and SATU:** with saved latent draws, the mean of y1 − y0 (observed outcome, imputed
  counterfactual) over the treated and untreated units.

**Other functions:**
- **`effects_summary(effs)`** gives the mean, sd, 2.5/50/97.5% quantiles, bulk ESS, and R-hat
  over chains.
- **`partition_summary(out; burn)`** gives:
  - the posterior similarity matrix;
  - the saved partition with the smallest expected Binder loss;
  - the posterior of k;
  - the separated-units share.
- **`rand_ppd` and `dpm_ate`** are documented as predictive draws of *one new unit's gain*, which
  depend on σ23. The README now uses `effects` for credible intervals (bug 12).

**`ridge_block` now defaults to false** (the user's decision, made in its own commit).

## Tests (`test/test_effects.jl`, 42 checks)

1. **Closed form against simulation.** TT and TUT match a 2M-unit Monte Carlo of a 3-component
   mixture at a fixed (x, w).
2. **Label invariance.** Permuting cluster labels changes no effect, in both slope modes.
3. **The k = 1 case.** `gaussian` gives x′(β1 − β0) and the textbook Heckman TT and TUT, per
   draw.
4. **Weights.** The Dirichlet weights' Monte Carlo moments match Dir(n_j, α).
5. **Scale.** Multiplying Y by 10 multiplies every effect by 10.
6. **Other checks:**
   - the per-draw lengths are right with burn-in;
   - SATT matches a recomputation from the saved latent data;
   - the summaries work for one and for several chains;
   - the partition summaries are consistent, and the Binder point is minimal among the saved
     partitions.

**Suite:** 2474/2474.

## Demonstration

Each configuration ran 8 chains × 3000 iterations, with the first 1000 discarded. The defaults
were C3, R2, hierarchical intercepts, and R3 and the ridge block off. Intervals are 95% credible
intervals pooled over chains.

**Hedonic data:** sample ATE 5.155, TT 6.671, TUT 3.565. Population (n = 10⁶): 5.001, 6.528,
3.474.

| Sampler | PATE | PATT | PATU | R-hat (max) |
|---|---|---|---|---|
| `marginal :common` | 5.25 [4.77, 5.70] | 6.13 [5.38, 6.71] | 4.33 [3.79, 4.96] | 1.06 |
| `marginal :cluster` | 5.28 [4.79, 5.76] | 6.12 [5.39, 6.75] | 4.40 [3.85, 5.05] | 1.03 |
| `gaussian` | 5.26 [4.85, 5.70] | 6.15 [5.50, 6.76] | 4.33 [3.80, 4.90] | 1.04 |
| `blocked` | 5.20 [4.72, 5.66] | 6.09 [5.43, 6.68] | 4.27 [3.68, 4.93] | 1.12 |

**Two-component data:** sample ATE 4.920, TT 6.653, TUT 3.383. Population (n = 10⁶): 4.997,
6.533, 3.470.

| Sampler | PATE | PATT | PATU | R-hat (max) |
|---|---|---|---|---|
| `marginal :common` | 5.44 [4.60, 6.26] | 7.07 [5.79, 8.09] | 3.99 [2.86, 5.21] | 1.11 |
| `marginal :cluster` | 5.50 [4.61, 6.37] | 7.16 [5.95, 8.07] | 4.02 [2.84, 5.35] | 1.11 |
| `gaussian` | 5.50 [4.61, 6.48] | 7.17 [5.93, 8.08] | 4.01 [2.81, 5.41] | 1.15 |
| `blocked` | 5.40 [4.46, 6.28] | 6.84 [4.57, 8.02] | 4.12 [2.99, 5.39] | 1.34 |

SATT and SATU agree with PATT and PATU to within 0.07 in every row.

**Interval width.** The credible interval for the PATE is 0.85–0.96 wide on the hedonic data and
1.7–1.9 wide on the two-component data. The old predictive route (`rand_ppd` + `dpm_ate` at the
covariate means, which the README presented as a "95% CI") gives 5.1–5.5 and 8.9–9.3: about 5.5×
wider. That is the spread of one new unit's gain, not uncertainty about an average.

**Coverage on these two data sets.**
- **Two-component data:** every interval covers both the sample and the population truths.
- **Hedonic data:**
  - PATE covers both truths in every sampler.
  - PATT covers both, but the posterior mean is about 0.5 below the sample TT.
  - PATU misses both truths in every sampler: the lower bounds are 3.68–3.85, against 3.565 and
    3.474.
- **Watch item.** The posterior pulls TT and TUT toward the ATE: its σ12 and σ13 are about half
  the size the data suggest.
  - **Two-step comparison:** a classical Heckman two-step on the same sample gives σ12 = 0.64 and
    σ13 = −0.89 (true ±0.7), with TT 6.94 and TUT 3.71.
  - **Bayesian posterior:** on the standardized scale its means imply correlations of about 0.29
    and −0.40, against ±0.7 true.
  - **Not the Σ prior.** Under C3 the implied prior correlation has sd 0.59 and P(|r| > 0.7) =
    0.33, and it doesn't depend on ρ. So the pull comes from the weakly identified ridge (Phase
    4e/4f) together with the coefficient priors, or from sampling variation in this one data set.
  - **`gaussian` shows it too,** and it is the correctly specified model for these data.
  - **Next step:** one data set can't separate the two explanations. The Phase 6 coverage study
    should; it is the first thing to check there.
- **`blocked`** mixes worst (R-hat up to 1.34 on PATT for the two-component data). That is
  consistent with Phase 4a.
