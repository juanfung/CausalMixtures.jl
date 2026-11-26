# Phase 6a′: a learned Σ scale (`sigma_scale = :hier`)

Recorded 2026-11-26 on an Apple M2, Julia 1.12.0. The outputs are:
- `geweke_6ap_long.md`;
- `coverage_pilot2*.csv`, summarized in `coverage_pilot2_summary.md`;
- `phase6a_prime_ar_output.md`.

## Summary

- **`sigma_scale = :hier` is implemented, tested and Geweke-validated**, in `marginal` and
  `gaussian`.
  - It learns the Σ scale r = (r1, r0) with r_k ~ Gamma(2, rate 2/0.153), whose mean is C3's
    0.153.
  - It recovers about the data's scale: around 0.22 at r2 = 0.5 and around 0.045 at r2 = 0.95,
    against C3's fixed 0.153.
- **It fixes the high-R² problem without `:auto`'s cost.** At r2 = 0.95 with a strong instrument
  on Gaussian data:
  - C3 covers PATE and PATU in only 70% of replications;
  - `:auto` restores coverage but doubles the interval widths and the RMSE;
  - `:hier` covers 100% with C3's narrow widths and the lowest RMSE.
- **The over-clustering diagnosis was only partly right.**
  - Much of the "k = 4–6 on one Gaussian" comes from tiny clusters (the DP's α·log n tail), not
    from splitting the bulk.
  - Clusters holding at least 5% of units number about 2.2 under C3 and still about 2.1 under
    `:hier`, even though `:hier` learns r ≈ the data's scale.
  - So the Σ scale isn't what drives it. The effects are unaffected either way.
- **Recommendation:** make `:hier` the default prior, subject to tier 1 confirming it. Run tier 1
  as C3 against `:hier`, with `:auto` as an optional reduced arm.

## 1. The step

**Derivation.** I verified it before coding; it matches the coordinator's.
- With R = diag(1/ρ, r1, r0), Ψ = ρR has Ψ11 = 1 and Ψ12 = 0.
- In the Phase 4f form, δ_j | Ω_j ~ N(0, Ω_j) doesn't involve r, and
  Ω_j ~ IW₂(ρ, diag(ρr1, ρr0)).
- The restricted-IW prior is exactly the product of these normalized densities (Phase 4f's
  prior-mapping test). So r_k enters only through |Ψ22·1|^{ρ/2} ∝ r_k^{ρ/2} and
  exp(−½ ρ r_k (Ω_j⁻¹)_kk).
- With K occupied clusters:

  r_k | Ω_1..Ω_K ~ Gamma(a_r + Kρ/2, b_r + (ρ/2) Σ_j (Ω_j⁻¹)_kk).

**Implementation.**
- **Options:** `InputParams(sigma_scale = :fixed | :hier, r_shape = 2, r_rate = 2/0.153)`.
  `:auto` stays a prior construction: `setup_default_priors(raw; r = :auto)` with `:fixed`.
- **State and output:** the state holds r, starting at the priors' R, and `out_r` saves it per
  draw.
- **Where the current R is used:** `sigma_R(state, input)` supplies it to:
  - the label step's auxiliary G0 draws;
  - the Σ update;
  - R3;
  - the ridge block.
- **Saved draws:** `effects()` and `rand_marginal` use each saved draw's own r for their G0 draws.
- **Sweep:** a new last step (viii) in `marginal`, and last in `gaussian`.

## 2. Tests

**Unit tests** (`test/test_sigma_scale.jl`, 23 checks):
- The r_k conditional equals a brute-force grid of Gamma prior × Σ_j IW₂(Ω_j | ρ, diag(ρr1, ρr0))
  log densities. The sampler's draws match it in mean and variance.
- **Invariance:** with r drawn from its prior, the clusters' Σ drawn from `NobileWishart` given r,
  and one update, the prior on log r is unchanged. The shape a + Kρ fails the same check
  (|z| > 10).
- **Wiring:**
  - `:fixed` is the default;
  - r starts at the priors' R and is saved;
  - `sigma_R` follows it;
  - `marginal` and `gaussian` run;
  - `blocked` refuses.

**Long Geweke** (500k sweeps against 500k forward draws, with r learned; forward
r ~ Gamma(2, 2/0.153); log r1 and log r0 among the statistics):

| Sampler | Statistics | max \|z\|, seed 1 | Seeds 2 and 3 | log r statistics |
|---|---|---|---|---|
| `marginal :common` | 48 | 3.73 (share(c₁)²) | 2.51, 2.40 | within ±1.8 |
| `marginal :cluster` | 48 | 3.43 (α²) | 2.32, 1.66 | within ±1.8 |
| `gaussian` | 30 | 2.80 (log r0²) | — | within ±2.8 |

- Seed 1's 3.4–3.7 lands on the correlated partition statistics (α, k, share), with opposite
  signs in the two slope modes.
- In seeds 2 and 3 those statistics are all within ±1.5. So it is noise, as with seed 1 in Phase
  4f.
- `blocked` is unchanged (1.97).

**Planted bugs** (50k sweeps):

| Planted bug | `:common` | `:cluster` | `gaussian` |
|---|---|---|---|
| None, r learned (control) | 2.6 | 2.6 | 3.8 (s22²) |
| (a) shape a_r + Kρ instead of a_r + Kρ/2 | **368.7** | **365.5** | **312.5** |
| (b) label-step auxiliary draws at C3's r | **11.4** | **10.0** | 3.8 (no label step: inert) |

Both bugs are real errors.
- **(b) is invalid** because Algorithm 8 needs the auxiliary parameters drawn from the actual base
  measure G0(r).

**Suite:** 2569/2569.

## 3. Pilot: C3, `:auto` and `:hier`

The pilot used the 16 cells of Phase 6a, each now with three priors (24 cells): realistic mode,
n = 1000, `:gaussian` and `:mixture` × r2 ∈ {0.5, 0.95} × iv ∈ {0.3, 1}. Each replication ran
`marginal :common` with 4 chains × 3000 iterations and `n_g0 = 5`, 10 replications per cell. Seeds
match the first pilot. There were 240 replications with no failures, in 2.8 h on 8 processes
together with the Geweke runs.

With 10 replications the coverage standard error is ±0.09–0.14, so these are signals, not claims.

**Strong-instrument cells (iv = 1).**

Coverage of the population truth for PATE / PATT / PATU, then mean width and RMSE over the three
estimands:

| Cell | Prior | Coverage | Width | RMSE | k (≥ 5%) | r posterior (data) |
|---|---|---|---|---|---|---|
| gaussian, r2 0.95 | C3 | 0.70 / 1.00 / 0.70 | 0.054 | 0.021 | 1.00 | 0.153 fixed (0.024) |
| gaussian, r2 0.95 | auto | 0.80 / 0.90 / 1.00 | 0.118 | 0.046 | 1.27 | 0.023 fixed |
| gaussian, r2 0.95 | hier | **1.00 / 1.00 / 1.00** | 0.059 | **0.012** | 1.04 | 0.051 (0.024) |
| gaussian, r2 0.5 | C3 | 0.80 / 1.00 / 0.90 | 0.270 | 0.083 | 2.28 | 0.153 fixed (0.242) |
| gaussian, r2 0.5 | auto | 0.90 / 1.00 / 1.00 | 0.258 | 0.068 | 1.83 | 0.240 fixed |
| gaussian, r2 0.5 | hier | **1.00 / 1.00 / 1.00** | 0.254 | **0.043** | 2.19 | 0.217 (0.249) |
| mixture, r2 0.95 | C3 | **1.00 / 1.00 / 1.00** | 0.057 | **0.014** | 1.00 | 0.153 fixed (0.023) |
| mixture, r2 0.95 | auto | 1.00 / 0.90 / 0.90 | 0.136 | 0.055 | 1.76 | 0.025 fixed |
| mixture, r2 0.95 | hier | 0.80 / 0.90 / 0.90 | 0.073 | 0.020 | 1.26 | 0.047 (0.024) |
| mixture, r2 0.5 | C3 | 0.90 / 1.00 / 1.00 | 0.294 | 0.081 | 2.99 | 0.153 fixed (0.242) |
| mixture, r2 0.5 | auto | 0.90 / 1.00 / 1.00 | 0.289 | 0.074 | 2.44 | 0.227 fixed |
| mixture, r2 0.5 | hier | 1.00 / 0.90 / 0.90 | 0.295 | 0.076 | 2.93 | 0.206 (0.248) |

**Weak-ish instrument cells (iv = 0.3, F ≈ 25).**
- R-hat exceeds 1.1 in 20–90% of replications under every prior, so the comparisons are
  unreliable. These stay held for option D.
- At r2 = 0.95 `:hier` again has the best RMSE on Gaussian data: 0.015–0.029, against 0.020–0.035
  for C3 and 0.03–0.09 for `:auto`.
- On `:mixture` at r2 = 0.95, `:hier` has the narrowest intervals with full coverage.

**Reading.**
- **The high-R² problem from Phase 5b is real and `:hier` fixes it.**
  - On Gaussian data at r2 = 0.95, C3's fixed scale under-covers PATE and PATU (0.70) with a
    positive bias.
  - `:hier` learns r ≈ 0.05, far below C3's 0.153, and covers fully. Its widths are C3's, and its
    RMSE is about half of C3's.
- **`:auto` is the wrong fix.** It doubles widths and RMSE at high R², and it raises the number of
  substantial clusters (1.27–1.76).
  - Its r equals the data's residual variance scaled by C3's median rule.
  - With per-cluster Σ that small, the label step opens more clusters, and each cluster's
    selection covariances become poorly determined.
- **`:mixture` at r2 = 0.95 favours C3 slightly** (coverage 1.0 against 0.8–0.9 for `:hier`).
  - With 10 replications this is within noise: a 0.8 has a standard error of about 0.13.
  - Here the three types' outcome means differ little, so a small r lets the model split them,
    which helps C3's coarser fit less than it costs.
  - Tier 1 needs to resolve this; it is the main open question for the default.
- **The over-clustering at r2 = 0.5 is not about the Σ scale.**
  - Raw k is 4–8, but clusters holding at least 5% of units number about 2.2 (Gaussian, truth 1)
    and about 3 (mixture, truth 3).
  - Learning r (0.21–0.22, close to the data's 0.24–0.25) leaves this unchanged.
  - The excess raw k is the DP's tail of tiny clusters: α ≈ 0.5–0.9 at n = 1000 implies about
    α·log(1 + n/α) ≈ 4 occupied clusters.
  - The extra roughly one substantial cluster on Gaussian data is unexplained, and the effects
    aren't affected. For reporting, k (≥ 5%) is the right partition summary.
- **Hyperprior sensitivity** (`phase6a_prime_ar_output.md`, single Gaussian data sets):
  - Across a_r ∈ {1, 2, 5} (mean 0.153), PATE, PATT and PATU move by at most 0.02, and every
    interval covers the truth.
  - The posterior r is 0.18–0.28 at r2 = 0.5 (data 0.23–0.25) and 0.04–0.07 at r2 = 0.95 (data
    0.022–0.025). The larger a_r pulls harder toward 0.153.
  - The learned r runs about twice `:auto`'s at high R² because the calibrations differ. The
    learned r matches Ω's prior *mean* (E Ω = ρr/(ρ − 3)), while `:auto` matches the prior
    *median* of σ_kk to s².
- **Runtime:** `:hier` costs the same as C3 (78–249 s per replication under load, with no
  systematic difference).

## 4. Recommendation and the final tier-1 design

**Default prior: `:hier`** (learned Σ scale, Gamma(2, 2/0.153)), conditional on tier 1.
- It is the only option that handles both R² regimes with C3's interval widths.
- It is fully Bayesian.
- It costs nothing extra.
- Keep C3 (`:fixed`) available, and `:auto` as an option, but don't make `:auto` the default.

**Tier 1 (diagnostic mode, for coverage claims):**

| Setting | Value |
|---|---|
| Cells | `:gaussian`, `:mixture` × n ∈ {500, 2000} × r2 ∈ {0.5, 0.95} × iv = 1 × prior ∈ {C3, hier}: 16 cells |
| Weak instrument | iv = 0.3 and `:weak_iv` held for option D, as instructed |
| Replications | 100 per cell (coverage standard error ±0.022 at 95%) |
| Chains | `marginal :common`, 4 chains × 6000 iterations (strong instrument; at 3000 R-hat > 1.1 in 0–40% of reps), first 25% discarded, 2000 saved draws per chain, `n_g0 = 5` |
| Recorded | coverage, width and bias against population and sample truth; RMSE; R-hat and ESS; k and k (≥ 5%); r posterior and data-implied r; separated share; runtime |
| Cost | about 30 s per replication serial at n = 500 and about 110 s at n = 2000. 1600 replications ≈ 1.12 × 10⁵ s serial, about **7–8 h** on the laptop's 8 processes |
| Optional | `:auto` as a third arm at 50 replications, adding about 3.5 h |

**Suggested order:**
1. n = 2000 first (about 5.5 h). It answers C3 against `:hier`, including the `:mixture` r2 =
   0.95 question.
2. Then n = 500.
3. Tiers 2 and 3 (misspecified scenarios, realistic spot checks) follow with the chosen default,
   at about 7 h.
4. Tier 4 waits for option D.
