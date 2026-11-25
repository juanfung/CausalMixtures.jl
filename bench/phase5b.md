# Phase 5b: is the PATU miss a mixing artifact or the posterior?

Recorded 2026-11-25 on an Apple M2, Julia 1.12.0. Scripts: `phase5b.jl` and `phase5b_mle.jl`; the
raw output is in `phase5b_output.md`. Data: `generate_hedonic_data(500, 1234)`, C3 priors, on
the standardized scale.

## Verdict

**It is the genuine posterior, not mixing.**
- A gold-standard sampler of the observed-data posterior, with no data augmentation, gives the
  same answer as the data-augmentation (DA) samplers to the third decimal.
- The DA chains agree with it from every initialization, with no drift.

The shrinkage of σ12 and σ13, and with it TT and TUT being pulled toward the ATE, comes from the
**C3 prior on Σ**, specifically its scale r. It does not come from the coefficient priors.
- **The mechanism:** C3's prior on the within-arm outcome variance Ω (prior mean 0.22) is
  calibrated for noisy outcomes. These data have an outcome R² of 0.94, so the actual within-arm
  variance is about 0.06.
- **Why that shrinks σ1k:** the data pin σ_kk = Ω_kk + σ1k², and the selection covariances are
  weakly identified (the Phase 4e ridge). So a prior that pushes Ω up pushes σ1k² toward 0.
- **The fix it suggests:** with the Ω prior on the data's scale, the 95% intervals cover the true
  TT and TUT again.

Option D therefore isn't needed for this problem. It is still the mixing remedy from Phase 4f.

## 1. Gold standard: the observed-data posterior without DA

`src/observed_data.jl` implements the per-unit likelihood:
- treated units: log N(y | x′β1, σ22) + log Φ((w′γ + σ12/σ22 · (y − x′β1)) / √(1 − σ12²/σ22));
- controls: the same with (β0, σ33, σ13) and the sign flipped.

The rest of the set-up:
- **Parameters:** (β, δ, log-Cholesky Ω), with the 4f prior δ | Ω ~ N(0, Ω/Ψ11) and
  Ω ~ IW₂(ρ, Ψ22·1), plus the Jacobian.
- **Sampler:** random-walk Metropolis with its proposal covariance adapted during the 50k burn-in
  only, then 400k frozen iterations, thinned by 20.
- **Chains:** 4, started at the two-step estimates, at two overdispersed points, and at zero
  correlation.
- **Unit tests:**
  - the likelihood equals a 2-D quadrature of the trivariate normal;
  - the Jacobian matches finite differences;
  - the prior equals the `NobileWishart` density;
  - with no data the sampler recovers the prior.
- **Reuse:** the code is written per unit subset (`idx`), so option D can reuse it per cluster.

**Results.** Max R-hat over all 12 parameters is 1.0003. ESS is 25–28k on σ1k and the effects.
Acceptance is 0.14–0.25.

| Quantity | Gold standard (C3) |
|---|---|
| σ12 | 0.066 [−0.043, 0.160] |
| σ13 | −0.104 [−0.206, 0.017] |
| PATE | 5.259 [4.796, 5.710] |
| PATT | 6.140 [5.442, 6.741] |
| PATU | 4.335 [3.779, 4.973] |

Sample truth: ATE 5.155, TT 6.671, TUT 3.565. Population: 5.001, 6.528, 3.474.

## 2. The DA samplers from three initializations

Each configuration ran 8 chains: `gaussian` for 100k iterations, `marginal :common` for 50k. The
first 20% of saved draws were discarded.
- **default:** `dpm_init`, which sets y_miss = y_obs, so σ1k ≈ 0 at the start.
- **heckman:** the two-step γ, β and σ1k, with the latent data imputed by 20 latent passes.
- **overdispersed:** the two-step β plus N(0, 0.5²) noise, selection correlations drawn from
  U(−0.9, 0.9), and variances multiplied by exp(N(0, 0.5²)).

| Sampler, init | σ12 | σ13 | PATT | PATU | R-hat (max) |
|---|---|---|---|---|---|
| `gaussian`, default | 0.064 [−0.045, 0.159] | −0.103 [−0.205, 0.022] | 6.135 [5.418, 6.735] | 4.345 [3.786, 4.976] | 1.002 |
| `gaussian`, heckman | 0.065 [−0.044, 0.158] | −0.103 [−0.207, 0.021] | 6.136 [5.418, 6.744] | 4.337 [3.784, 4.979] | 1.001 |
| `gaussian`, overdispersed | 0.065 [−0.044, 0.159] | −0.103 [−0.207, 0.022] | 6.135 [5.416, 6.744] | 4.340 [3.784, 4.974] | 1.002 |
| `marginal`, default | 0.067 [−0.040, 0.160] | −0.104 [−0.208, 0.022] | 6.142 [5.415, 6.744] | 4.328 [3.775, 4.953] | 1.001 |
| `marginal`, heckman | 0.065 [−0.044, 0.160] | −0.103 [−0.207, 0.021] | 6.134 [5.418, 6.742] | 4.340 [3.782, 4.979] | 1.003 |
| `marginal`, overdispersed | 0.065 [−0.042, 0.158] | −0.105 [−0.208, 0.016] | 6.149 [5.443, 6.747] | 4.340 [3.783, 4.975] | 1.003 |

- **R-hat across all 24 chains** (3 inits × 8): `gaussian` ≤ 1.0012 and `marginal` ≤ 1.0022 for
  σ12, σ13, PATE, PATT and PATU.
- **Trajectories:** the window means over chains, in windows of 10k iterations (`gaussian`) or
  5k (`marginal`), are flat from the first window. For example, σ13 under `gaussian` default reads
  −0.100 −0.101 −0.103 … −0.103, and the heckman chains, which start near −0.24, are already at
  −0.103 in their first window.
- **The DA chains agree within the first few thousand iterations.** Their slow mixing along the
  ridge costs ESS (Phases 4e and 4f), but it doesn't bias where they settle.
- **So the hypothesis is ruled out.** The (d*, y_miss) ↔ θ coupling only slows these chains down;
  it doesn't keep them away from the posterior.

## 3. The posterior is prior-driven: the Σ prior's scale

**Maximum likelihood** (`phase5b_mle.jl`, Newton ascent on the same observed-data likelihood with
no prior): σ12 = 0.152 and σ13 = −0.218, i.e. correlations 0.63 and −0.77 (true ±0.7), at
log-likelihood −274.57. A second start stopped at a worse point (−279.30), which I discarded. The
Heckman two-step gives 0.176 and −0.244.

**Prior sensitivity**, using the gold standard with 4 chains each (R-hat ≤ 1.0009):

| Prior | Prior mean of Ω_kk | σ12 | σ13 | PATT | PATU |
|---|---|---|---|---|---|
| C3: outcome slopes sd 1, intercepts sd 1, ρ = 10, r = 0.153 | 0.22 | 0.066 | −0.104 | 6.140 [5.442, 6.741] | 4.335 [3.779, 4.973] |
| outcome slopes sd 2 | 0.22 | 0.066 | −0.104 | 6.141 [5.433, 6.742] | 4.335 [3.782, 4.970] |
| intercepts sd 3 | 0.22 | 0.065 | −0.105 | 6.150 [5.439, 6.747] | 4.341 [3.782, 4.981] |
| outcome slopes sd 2, intercepts sd 3 | 0.22 | 0.065 | −0.106 | 6.153 [5.441, 6.751] | 4.342 [3.784, 4.976] |
| ρ = 5, r = 0.153 | 0.38 | 0.085 | −0.131 | 6.294 [5.542, 6.891] | 4.225 [3.651, 4.913] |
| ρ = 5, r = 0.03 | 0.075 | 0.128 | −0.185 | 6.605 [5.853, 7.149] | 3.982 [3.420, 4.741] |
| ρ = 10, r = 0.03 | 0.043 | 0.129 | −0.190 | 6.632 [5.913, 7.156] | 3.973 [3.427, 4.715] |
| ρ = 10, r = 0.01 | 0.014 | 0.153 | −0.210 | 6.747 [6.071, 7.245] | 3.840 [3.315, 4.584] |
| Maximum likelihood | — | 0.152 | −0.218 | | |

**Reading the table:**
- **The coefficient priors don't matter:** with outcome slopes at sd 1 or 2 and intercepts at sd
  1 or 3, the results move by at most 0.01.
- **The Σ scale r decides it.**
  - **At r = 0.03,** prior mean Ω_kk is 0.04–0.08, close to the data's within-arm variance. There
    σ1k is about 85% of the MLE, and both TT and TUT intervals cover their sample and population
    truths: PATT [5.91, 7.16] contains 6.671 and 6.528; PATU [3.43, 4.72] contains 3.565 and
    3.474.
  - **At r = 0.01,** the posterior sits at the MLE.
  - **PATE barely moves** (5.26–5.33).
- **Lowering ρ alone, at C3's r,** helps only a little: lower ρ gives a less informative but
  *larger* prior mean of Ω.

**What follows.**
- **C3's r = 0.153** was calibrated for a prior median within-cluster variance of 0.3. The
  outcome is standardized but not residualized, so with strong covariates the residual variance
  is far smaller, and the prior on Ω then shrinks the selection covariances.
- **Candidate fixes**, for the user or coordinator to decide:
  - (i) a data-adaptive r, set from pilot per-arm OLS residual variances on the standardized
    scale (here about 0.06, i.e. r ≈ 0.03);
  - (ii) a prior on (δ, Ω) whose δ part doesn't tighten as Ω grows, for example a prior on the
    correlation and Ω separately;
  - (iii) leaving C3 and documenting the sensitivity.
- **For Phase 6:** the coverage study should include high-R² scenarios. Coverage of TT and TUT
  under C3 is expected to fall short there and to be fine in the R² ≈ 0.6 variant.
