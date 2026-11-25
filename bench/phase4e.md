# Phase 4e: why the ATE mixes slowly (analysis only)

Recorded 2026-11-24 on an Apple M2, Julia 1.12.0. The script is `bench/phase4e_diagnosis.jl`; the
full outputs are `phase4e_base.md`, `phase4e_noise.md`, `phase4e_z03.md` and `phase4e_z3.md`.

## Set-up

- **Samplers:**
  - `gaussian` (k = 1);
  - `marginal` (`:common`, the R2 label step, hierarchical intercepts, R3 off).
- **Runs:** 8 chains × 5000 iterations (1000 burn-in), C3 priors, `scale_data = (true, true)`,
  so the coefficients are on the standardized scale.
- **Parameters:**
  - The `marginal` coefficients and Σ are those of unit 1's cluster. Its "β1[(Intercept)]" and
    "β0[(Intercept)]" are the cluster intercepts a_1 and a_0.
  - The latent d* and the unobserved outcome are recorded for six fixed units (three treated,
    three controls).
- **DGPs:** all from `generate_hedonic_data(500, 1234)`:

| DGP | Change | Treated share | Sample ATE |
|---|---|---|---|
| base | — | 0.51 | 5.155 |
| noise | outcome sds × 3.3, same correlations: R² of Y on (1, X, D, X·D) 0.61 instead of 0.94 | 0.51 | 5.090 |
| z03 | instrument coefficient γ_Z = 0.3 instead of 1 | 0.39 | 5.155 |
| z3 | γ_Z = 3 | 0.77 | 5.155 |

## Evidence

**Bulk ESS per 1000 draws** (`gaussian` / `marginal`):

| DGP | ATE | σ12 | β1 intercept | σ13 | β0 intercept | σ23 | γ_Z | d* (range) | y_miss (range) |
|---|---|---|---|---|---|---|---|---|---|
| base | 10.5 / 15.5 | 10.2 / 13.9 | 10.5 / 15.0 | 10.5 / 13.4 | 10.8 / 14.3 | 19 / 26 | 243 / 240 | 460–890 / 590–990 | 55–115 / 80–150 |
| noise | 7.9 / 9.0 | 6.2 / 11.7 | 6.5 / 19.0 | 4.6 / 6.7 | 5.2 / 8.9 | 3.7 / 10.2 | 78 / 26 | 290–720 / 230–750 | 21–62 / 27–92 |
| z03 | 6.0 / 9.2 | 5.2 / 8.5 | 5.2 / 8.7 | 6.0 / 8.7 | 6.2 / 9.1 | 20 / 30 | 198 / 177 | 167–950 / 240–980 | 24–27 / 37–47 |
| z3 | 12.4 / 25.5 | 41.7 / 48.7 | 85 / 117 | 11.4 / 21.9 | 11.5 / 22.6 | 21 / 32 | 85 / 82 | 440–900 / 710–990 | 155–470 / 290–620 |

Other parameters:
- The slope coefficients mix at 13–540 per 1000 draws. The fastest is β1[X] under z3 (465 / 542).
- Under `marginal`, k and α reach 350–800 in every DGP except noise, where k is at 9.2
  (mean k 3.3).
- The hierarchical m and τ² reach 700–940, and 72–256 under noise.

**Paired posterior correlations** (`gaussian` / `marginal`):

| DGP | σ12 with β1 intercept | σ13 with β0 intercept |
|---|---|---|
| base | −0.94 / −0.75 | +0.94 / +0.93 |
| noise | (σ12 not among the five slowest) | +0.93 / +0.80 |
| z03 | −0.97 / −0.90 | +0.96 / +0.95 |
| z3 | (σ12 and β1 fast) | +0.88 / +0.87 |

**Correlation with the ATE.** The ATE is correlated with every slow member, |r| = 0.55–0.95.
Under z3 its correlation with the β0 intercept is −0.95 (`gaussian`) and −0.94 (`marginal`).

**Autocorrelation.** The slow group has lag-1 autocorrelation 0.93–0.98 and lag 50 of 0.27–0.60
(`gaussian`). The ATE's autocorrelation curve follows the slowest pair's in every DGP.

## Diagnosis

The ATE is slow because of one ridge per outcome arm: the outcome intercept against the
selection covariance σ1k. The probit block and the latent data are not the cause.

1. **The ridge, and why the sampler crawls along it.**
   - Given d*, each outcome equation is a regression with the selection residual as a
     regressor:
     y_k = x′β_k + σ_1k (d* − w′γ) + e_k, with Var(e_k) = σ_kk − σ_1k².
   - Within the arm that observes y_k, the regressor d* − w′γ has mean λ(w′γ) ≠ 0 (an inverse
     Mills ratio). λ varies little across units unless the instrument moves w′γ a lot.
   - So the regressor is close to collinear with the intercept, and (β_k intercept, σ_1k) is
     a posterior ridge with |r| = 0.75–0.97.
   - The sampler draws β | Σ, z and Σ | β, z in separate blocks, so it moves along the ridge
     in small zig-zag steps. With two blocks at correlation r, lag-1 autocorrelation is about
     r² ≈ 0.88–0.94, which is what we see.
   - The ATE is a contrast of the two arms' intercepts and slopes, so it inherits the
     slowness.
2. **Identification sets how bad it is.**
   - The ridge's correlation, and with it the ESS, follows how well the selection correction is
     identified in each arm.
   - **Weak instrument (z03):** ESS falls from 10–15 to 5–9, and |r| rises to 0.97.
   - **Strong instrument (z3):** the treated arm, now 77% of units, frees up: σ12 at 42–49 and
     its intercept at 85–117. The control arm, now 23% of units, stays on its ridge (11–23, r =
     0.88). The ATE improves only to 12–25, because it is limited by the minority arm.
   - **More noise:** the ridge loosens and σ23's positive-definite interval widens, so σ23
     joins the slow group (3.7–10). This is the only regime where R3 could help.
   - So the slowness is weak identification of the selection correction, made into a mixing
     problem by blocking β and Σ apart. A sampler that draws each ridge pair jointly would avoid
     it.
3. **What is not the cause.**
   - **γ and d\*:** they mix at 80–990 per 1000 draws, with γ_Z lowest under strong
     instrument and noise.
   - **y_miss:** 20–620.
   - **k and α:** 350–800, except under noise.
   - **The hierarchical hyperparameters:** 70–940.
   - **σ23:** pinned under base (Phase 4c).
   - **The mixture:** `gaussian`, with k = 1, is as slow as `marginal`. `marginal` is in fact
     1.3–2× faster on the slow group, plausibly because R2 redraws each unit's (d*, y_miss)
     jointly with its label from a β^c-collapsed predictive.
   - **The one exception is noise,** where `marginal` has k ≈ 3.3 with ESS 9 (lag-50
     autocorrelation 0.30). There, partition moves are a second slow mode on top of the ridge.

## Candidate remedies, ranked (none implemented)

1. **Draw each arm's (β_k, σ_1k) jointly in the conditional-regression parameterization, with
   y_miss integrated out.**
   - **The step.** Given (γ, d*), each arm's observed outcome is the regression
     y_k = x′β_k + σ_1k (d* − w′γ) + e_k with Var(e_k) = ω_k = σ_kk − σ_1k². With y_miss
     integrated out, the two arms separate (σ23 drops out, as in Phase 4c). So (β_k, σ_1k) can
     be drawn as one Gaussian block from the arm's observed units, and then ω_k.
   - **Conjugacy.** The Nobile prior is conditionally conjugate here. For an inverse Wishart,
     Σ11 is independent of (Σ11⁻¹Σ12, Σ22·1), so fixing σ11 = 1 keeps the Gaussian conditional
     prior of (σ12, σ13) given Σ22·1, and an inverse-Wishart or inverse-gamma prior on the rest.
   - **σ23** is then drawn by R3, already implemented and Geweke-tested, from its prior
     conditional, followed by a y_miss redraw.
   - **Why first.** This removes the ridge exactly, in both arms and every regime tested. Under
     Design A it replaces steps (ii)–(iv) per cluster, and the label step is untouched.
   - **The remaining limit** is the dependence on d*, which mixes fast here.
2. **Interweaving (ASIS; Yu & Meng 2011) of the outcome block.** Alternate the current
   (β, Σ | complete z) update with an update in the residual parameterization (β_k, σ_1k given
   e = d* − w′γ). This is less exact than (1), but it plugs into the existing steps and can be
   tested quickly. It is worth it if the hierarchical intercepts or the `:common` and
   `:cluster` splits make (1) awkward.
3. **Probit-block moves:** marginal data augmentation (Imai & van Dyk 2005) or a joint
   (γ, d*) draw.
   - These target the selection block, which mixes 10–50× faster than the slow group in every
     DGP. Expect little gain on the ATE.
   - They could become relevant only after (1), or when the instrument dominates (z3: γ_Z ESS
     82–85).

I did not test the joint draw over (β, σ12, σ13) combined with a y_miss-collapsed σ23 (R3) as a
full scheme. That would be Phase 5 work.
