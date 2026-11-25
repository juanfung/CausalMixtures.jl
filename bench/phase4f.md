# Phase 4f: the ridge block (option A), the review fixes, and a corrected diagnosis

Recorded 2026-11-25 on an Apple M2, Julia 1.12.0, with 1 thread (`JULIA_NUM_GC_THREADS=1` for the
parallel runs).

The outputs are:
- `geweke_4f_long.md` (ridge block on) and `geweke_4f_long_ridge_off.md`;
- `phase4f_mixing_{base,noise,z03,z3}.md`;
- `phase4f_timing_output.md`.

## Summary

- **Correctness.** Option A, the joint draw of (outcome coefficients, δ = (σ12, σ13)) given
  Ω = Var(y1, y0 | d*), followed by Ω, is implemented and correct:
  - the prior-mapping and brute-force conditional tests pass;
  - Geweke passes with the latent steps now exercised;
  - its invalid planted bugs fail.
- **It does not improve mixing.** ATE ESS is unchanged in every sampler and DGP.
- **Why:** the measured correlation between the intercept and σ_1k *given the complete latent
  data* is 0.00 (range −0.09 to 0.10). The −0.94 ridge of Phase 4e exists only in the posterior
  with d* integrated out.
- **What that means.** The slow coupling is between θ and the latent data (d*, y_miss), not
  between the β and Σ blocks as Phase 4e said. Any step that conditions on d*, including option
  B, cannot remove it. The candidates that can are listed at the end.

## Review fixes (done first)

1. **Harness gap.** Each Geweke iteration overwrote the latent data with `simulate_z`, so the
   sweep's final latent refresh (and R3's redraw) was never tested.
   - The harness now rotates each sweep so the latent steps run first:
     `marginal_sweep!(latent_first = true)`, `update_gaussian_params!(latent_first = true)`,
     and the latent refresh before the labels for `blocked`.
   - The samplers' own order is unchanged.
2. **Coverage.** Unit 1's cluster slopes, and their squares, are statistics wherever slopes are
   per cluster: `marginal :cluster`, `gaussian` and `blocked`.
3. **Docs.** `docs/implementation_plan.md` is refreshed from the plan artifact.
4. **Defaults.** `setup_default_priors` now defaults to `slopes = :common`, as `InputParams`
   does.
5. **Cleanups:**
   - `inv(Symmetric(Σ))` throughout;
   - one prior refresh per sweep;
   - a guard on degenerate σ23 intervals;
   - `rand_marginal` tolerates missing `out_hyper` entries.

## The ridge block

With σ11 = 1, write Σ = [1 δ′; δ Ω + δδ′]. The map from (δ, Σ_uu) to (δ, Ω) has Jacobian 1, and
|Σ| = |Ω|.

**Prior.** The `NobileWishart(ρ, Ψ)` density, |Σ|^{−(ρ+4)/2} exp(−½ tr ΨΣ⁻¹), factors as:
- δ | Ω ~ N(c, Ω/Ψ11), with c = Ψ21/Ψ11;
- Ω ~ IW₂(ρ, Ψ22·1).

δ's normalizer takes |Ω|^{−1/2}, which leaves |Ω|^{−(ρ+3)/2}, so **ν_Ω = ρ**. Under C3, Ψ11 = 1
and c = 0, so δ | Ω ~ N(0, Ω).

**Conditionals.** d* ~ N(μ_d, 1) involves neither block, so both conditionals come from y | d*:
- (B, δ) | Ω, γ, b, z is Gaussian: a two-equation regression on (x_i, v_i), with
  v_i = d*_i − μ_d,i. B's prior is the current β^c prior conditioned on the selection block, with
  the hierarchical intercepts included.
- Ω | B, δ ~ IW₂(ρ + n_j + 1, Ψ22·1 + E′E + Ψ11(δ − c)(δ − c)′).

**Where it runs.**
- Design A: step (iv), replacing Σ | β.
- `gaussian`: β | Σ, then the block.
- Option: `InputParams(ridge_block = true)`, the default.

**Validity.** Every step draws from a full conditional of the same joint posterior, in coordinates
related one-to-one by a Jacobian-1 map, so the composition is valid.

### Unit tests (`test/test_ridge_block.jl`, 274 checks)

- **Prior mapping.** 200k `NobileWishart` draws mapped to (δ, Ω) match N(c, Ω/Ψ11) × IW₂(ρ, Ψ22·1)
  in means and centred squares. This holds for C3 and for a general Ψ with c ≠ 0.
  - ν_Ω = ρ − 1 and ρ + 1 both fail (|z| > 10), so **ν_Ω = ρ is confirmed numerically**.
- **Conditionals against a brute-force log joint.** The log joint is the original model's
  N₃(z | Hβ^c, Σ) × N(β^c) × restricted-IW(Σ).
  - Its differences over (B, δ) equal those of the Gaussian conditional, to 1e−8.
  - Its differences over Ω equal those of IW₂(ρ + n + 1, …).
  - Both hold with independent and with correlated coefficient priors.
- **Draws.** The joint draws match the Gaussian conditional. Σ keeps σ11 = 1 and is positive
  definite.

### Geweke with the rotated harness

`bench/geweke.jl`: 500k sweeps against 500k forward draws, n = 20, C3, hierarchical intercepts.

| Sampler | Statistics | Before 4f (ridge off) | After 4f (ridge on) |
|---|---|---|---|
| `marginal`, `:common` | 44 | 3.42 (b2²) | 3.56 (log τ²_1) |
| `marginal`, `:common`, seeds 2 and 3 | 44 | 2.01, 1.66 | 2.19, 2.27 |
| `marginal`, `:cluster` | 44 | 2.45 | 2.88 |
| `gaussian` | 26 | 1.78 | 1.95 |
| `blocked` | 32 | 1.97 | 1.97 |

Seed 1's 3.4–3.6 under `:common` is under the 4.5 threshold. Its top statistic differs between the
two runs, and two more seeds give 1.7–2.3, so it was multiple-testing noise across 88 z-scores.

**Planted bugs** (`bench/geweke_power.jl`, 50k sweeps; R3 on for the R3 rows):

| Planted bug | `:common` | `:cluster` | `gaussian` | Expected |
|---|---|---|---|---|
| None (ridge on / ridge off) | 2.3 / 3.9 | 2.5 / 2.9 | 2.1 / 3.0 | pass |
| Latent refresh: wrong truncation side | **78.1** | **62.1** | **153.3** | fail |
| Latent refresh: c0y sign flipped (ridge on / off) | 4.5 / 4.6 | 4.3 / 3.6 | **16.8 / 11.2** | fail; see below |
| R3 without the redraw, d* drawn first (on / off) | 2.3 / 2.4 | 1.5 / 2.3 | **8.9 / 6.6** | fail; see below |
| R3 without the redraw | 2.1 | 2.3 | 1.5 | valid (4c) |
| R3 with the posterior target | 3.1 | 2.7 | 2.5 | valid (4c) |
| Ridge: Ω without the +1 and the Ψ11 δδ′ term | **7.3** | **7.4** | **5.7** | fail |
| Ridge: ν_Ω = ρ − 1 | **10.6** | **10.5** | **7.5** | fail |
| Ridge: B and δ drawn separately | 2.2 | 2.0 | 1.7 | valid |

**Notes on the planted bugs.**
- **The two latent-step bugs are strong in `gaussian` and weak in `marginal`.** Design A's R2
  label step redraws every unit's (d*, y_miss) right after the latent refresh. So the refresh
  affects Design A only through the cluster statistics of the other units.
  - The two 500k reruns for `marginal` are in the next section.
- **Drawing B and δ separately is valid.** B | δ, Ω followed by δ | B, Ω are two exact
  conditional draws, so it is a Gibbs sampler, just a less blocked one. It cannot fail Geweke.

**500k reruns of the two weak cases** (ridge on):

| Planted bug | `:common` | `:cluster` | `gaussian` |
|---|---|---|---|
| Latent refresh: c0y sign flipped | **13.3** (s12²) | **14.4** (s12²) | **33.3** (s23²) |
| R3 without the redraw, d* drawn first | 2.1 | 1.9 | **15.6** (s33) |

- **The c0y flip fails clearly** in every sampler.
- **The stale-y_miss bug fails in `gaussian`** (8.9 at 50k, 15.6 at 500k), where the latent refresh
  feeds straight into the θ update.
- **In Design A it stays invisible even at 500k.** The R2 label step that follows redraws every
  unit's (d*, y_miss) from the collapsed predictive before any coefficient or Σ step uses them.
  The corrupted latent data reach R2 only through the other units' cluster statistics during the
  scan.
- **So the bug is real, but in this sampler it has almost no effect.** The harness now sees the
  latent refresh, as the c0y and truncation bugs show.

## Mixing: ridge block off and on

`bench/phase4f_mixing.jl` uses the Phase 4e set-up: 8 chains × 5000 iterations (1000 burn-in),
seeds 20261121 + c, C3, R2, hierarchical, R3 off. Parameters are those of unit 1's cluster.

ATE bulk ESS per 1000 draws (off → on):

| DGP | `gaussian` | `marginal :common` | `marginal :cluster` |
|---|---|---|---|
| base | 10.5 → 8.8 | 14.4 → 16.3 | 15.2 → 14.1 |
| noise (R² 0.61) | 7.9 → 7.4 | 7.0 → 8.3 | 8.8 → 8.8 |
| z03 (weak instrument) | 6.0 → 7.0 | 7.4 → 7.6 | 7.6 → 3.9 |
| z3 (strong instrument) | 12.4 → 15.1 | 23.1 → 18.4 | 21.1 → 20.5 |

The previously slow parameters on base (ESS per 1000 draws, off → on), with the ridge
correlations (off / on):

| Sampler | σ12 | σ13 | β1 intercept | β0 intercept | corr(σ12, β1 int) | corr(σ13, β0 int) |
|---|---|---|---|---|---|---|
| `gaussian` | 10.2 → 10.4 | 10.5 → 7.9 | 10.5 → 10.6 | 10.8 → 8.2 | −0.94 / −0.94 | 0.94 / 0.94 |
| `marginal :common` | 16.9 → 16.1 | 11.2 → 13.5 | 18.5 → 17.3 | 11.8 → 14.4 | −0.84 / −0.86 | 0.92 / 0.92 |
| `marginal :cluster` | 16.6 → 15.2 | 11.9 → 12.0 | 17.6 → 16.3 | 12.5 → 12.8 | −0.92 / −0.83 | 0.95 / 0.94 |

**Results:**
- **Mixing:** ESS changes are within the seed-to-seed spread (Phase 4c showed up to about 40%), in
  both directions. The marginal ridge correlations are unchanged.
- **Estimates:**
  - ATE means and sds are unchanged: base 5.255–5.267 (sd 0.23–0.25).
  - The weak-instrument `:cluster` mean moves from 5.379 to 5.296, with sd 0.70–0.73.
  - R-hat is 1.007–1.064.
  - No chain was stuck in any configuration.
- **Reproducibility:** the ridge-off `gaussian` rows reproduce Phase 4e exactly.

## Why option A cannot work, and what can

The ridge block draws (B, δ) given the complete latent data. At 75 posterior states (`gaussian`,
base), the conditional correlation it samples from is:
- corr(β1 intercept, σ12 | Ω, γ, z): mean 0.00, range −0.09 to 0.10;
- corr(β0 intercept, σ13 | Ω, γ, z): the same.

Given d*, the regressor v_i = d*_i − w′γ varies within each arm: treated mean 0.64 (sd 0.69),
controls −0.80 (sd 0.60). So it is not collinear with the intercept, and there is no ridge to
break.

The −0.94 ridge appears only when d* is integrated out, where v_i becomes λ(w′γ_i), nearly
constant within an arm. The slow coupling is therefore between θ and the latent data:
- given d*, (β, σ_1k) is pinned;
- given (β, σ_1k), d* is pinned;
- so the chain moves along the observed-data ridge only as fast as the augmentation lets it.

Phase 4e's "β | Σ against Σ | β zig-zag" mechanism was wrong. The ridge it measured is real, but it
lives in the observed-data posterior.

**Consequences for the roadmap.**
- **Option B is ruled out.** It is the per-arm (β_k, σ_1k) draw with y_miss integrated out but
  d* conditioned on, so the same argument applies.
- **D, a collapsed move on the observed-data ridge** (new, recommended):
  - per cluster and arm, update (β_k intercept, σ_1k), or (β_k, σ_1k), by Metropolis–Hastings
    or slice sampling on the observed-data likelihood with d* and y_miss integrated out;
  - that likelihood is explicit: N(y_k | x′β_k, σ_kk) · Φ(±(w′γ + σ_1k/σ_kk·(y_k − x′β_k)) /
    √(1 − σ_1k²/σ_kk));
  - follow it immediately with a joint redraw of (d*, y_miss) from their conditional, which is
    the `r2_draw` construction;
  - it is a partially collapsed step like R2 and R3;
  - it targets the measured ridge directly, and a proposal along the ridge direction (estimated
    during burn-in) should move freely.
- **C, ASIS** (the fallback): interweave the current, sufficient augmentation with an ancillary
  one that holds the standardized residuals fixed while θ moves. ASIS is known to fix this kind of
  augmentation slowness in probit-type models (Yu & Meng 2011), but it needs a non-conjugate θ
  update under the ancillary augmentation.
- **MDA** (Imai & van Dyk 2005), with a working parameter that moves d* together with
  (intercept, σ_1k), is a third route along the same lines.
- **What not to repeat:** any further joint draw that conditions on d*.

**The ridge block itself** is valid and costs nothing (timing below). It is kept as an option
defaulting to on, as specified, but it buys nothing. Making it default to off is a reasonable
alternative.

## Timing

`bench/phase4f_timing.jl`: n = 500, the median of 3 runs of M = 1000, idle machine.

| Sampler | Ridge off | Ridge on |
|---|---|---|
| `marginal`, `:common` | 0.96 | 0.92 |
| `marginal`, `:cluster` | 1.15 | 1.10 |
| `gaussian` | 0.07 | 0.09 |
| `blocked` (J = 20) | 0.41 | — |

The figures are seconds per 1000 iterations.

## Suite

`julia --project=. test/runtests.jl` passes 2417/2417.
