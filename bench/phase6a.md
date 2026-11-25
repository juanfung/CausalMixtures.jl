# Phase 6a: simulator, adaptive Σ scale, recovery tests, coverage harness, pilot

Recorded 2026-11-25/26 on an Apple M2 (4 performance and 4 efficiency cores), Julia 1.12.0.

## What was built

1. **`simulate_selection(n; scenario, rng, r2 = 0.7, iv = 1.0, diagnostic = false)`**
   (`src/simulate_selection.jl`).
   - **Covariates:** hedonic-style, drawn from a Gaussian copula: log sqft, age, log lot,
     bedrooms, neighbourhood income, and the excluded instrument (distance to the nearest
     choice school).
   - **Outcome:** log price around 12.
   - **Scenarios:** `:gaussian`, `:mixture`, `:skewed`, `:roy` and `:weak_iv`.
     - `:mixture` has 3 zoned-school types, with gain τ − κ(q − q̄), so there are winners and
       losers.
     - `:skewed` has skew-normal outcome errors and a t₄ selection error.
     - `:roy` selects on gains.
   - **Controls:** `iv` scales the instrument; `r2` sets the outcome noise; `diagnostic = true`
     multiplies the effects by 10 and separates the types.
   - **Sample truth:** ATE, ATT, ATU, the share with y1 > y0, quantiles of y1 − y0, and quantile
     treatment effects.
   - **Population truth:** the same quantities from one 10⁶ draw on an independent fixed stream,
     cached, plus PATE/PATT/PATU.
   - **Info:** treated share, achieved R², and the first-stage probit z-statistic with z² as a
     pseudo-F. For reference at n = 1000, F ≈ 120 at iv = 1, 25–29 at iv = 0.3, and about 12 for
     `:weak_iv`.
   - **Legacy:** `generate_hedonic_data` is kept but marked legacy.
   - **Tests:** 77 checks:
     - reproducibility;
     - achieved R² within 0.03 of the target;
     - treated share;
     - ATT > ATE > ATU under `:roy`;
     - mixture type shares;
     - the weak first stage;
     - sample truth consistent with population truth at n = 200k.
2. **The data-adaptive Σ scale, an option:** `setup_default_priors(raw; r = :auto)`, which calls
   `auto_sigma_scale`.
   - **Rule:** on the standardized scale, fit per-arm OLS of y on x and take the residual
     variances s_k². Then r_k = 0.153 · s_k²/0.3, clamped to [0.01, 0.5], and R = diag(1/ρ, r1, r0).
   - **Default:** the fixed C3 value is unchanged.
   - **On the hedonic data:** r = (0.023, 0.027). With `gaussian`, both the PATT and PATU
     intervals cover the sample and the population truths, as r = 0.03 did in Phase 5b.
   - **Tests:** the scaling rule and the coverage check above.
3. **Recovery tests** (`test/integration/test_hedonic_model.jl`) replace the 30–50-iteration
   runs. They use diagnostic `:gaussian` data with fixed seeds.
   - **`gaussian`:** n = 2000, 4 × 25k iterations; coverage of the population PATE, R-hat < 1.05,
     ESS > 150.
   - **`marginal :common`:** n = 1000, 4 × 3000; coverage and R-hat < 1.1.
   - **Runtime:** about 1.5 minutes in the default suite.
   - **Opt-in (`CM_LONG=1`):** `marginal` at n = 2000, 4 × 10k, meeting R-hat < 1.05 and ESS >
     150. It passes, taking 5–12 minutes.
4. **`bench/coverage.jl`**, the coverage harness. Per replication it simulates, fits
   `marginal :common`, calls `effects()`, and records:
   - coverage, width and bias against the population and the sample truths;
   - R-hat and ESS;
   - the separated share and k;
   - the Σ scale used, the first-stage F, and the runtime.

   Results are written to a CSV as each replication finishes, and a rerun resumes. It shards over
   single-threaded processes (`shard=i/K`). Julia threads are also supported but contend for the
   garbage collector: 7 threads ran slower than one process. `bench/coverage_summary.jl` makes
   the tables.

## Pilot

The pilot is in `coverage_pilot*.csv` and summarized in `coverage_pilot_summary.md`.
- **Cells:** realistic mode, n = 1000, `:gaussian` and `:mixture` × r2 ∈ {0.5, 0.95} × iv ∈
  {0.3, 1} × prior ∈ {C3, auto}: 16 cells.
- **Replications:** 10 per cell (11 in one cell, from the stopped threaded run).
- **Chains:** `marginal :common`, 4 chains × 3000 iterations, the first 25% of saved draws
  discarded.

With 10 replications, coverage has a standard error of about ±0.09–0.14, so the pilot measures
cost and flags problems; it makes no coverage claims.

**What it shows:**
- **Coverage.**
  - 95% interval coverage of the population truth is 0.70–1.00 across cells; the sample truth is
    similar. Nothing is clearly broken at this precision.
  - The lowest is C3 at r2 = 0.95, iv = 1 on `:gaussian`: 0.70 for PATE and PATU. Those intervals
    are narrow (0.045 and 0.065) and the bias is positive (+0.010 and +0.017, a quarter of the
    width). That matches the Phase 5b mechanism: with strong covariates, C3's Σ scale pulls the
    selection covariances toward 0.
  - With `:auto` in that cell, coverage is 0.80–1.00, but the intervals are twice as wide and k
    rises from 1.18 to 1.74. The smaller Σ scale lets more clusters in.
  - Whether `:auto` is a net improvement is exactly what the full study has to settle.
- **Over-clustering at r2 = 0.5.**
  - Mean k is 4–6 on `:gaussian` data (true k = 1) and 6–8 on `:mixture` (true 3), under both
    priors.
  - The effects are unaffected (coverage is nominal), but the posterior of k is not interpretable
    there.
  - This needs a look in Phase 6: possibly the hierarchical intercepts or the Escobar–West prior
    on α, combined with these covariates.
- **Mixing.**
  - At 4 × 3000 iterations, R-hat for the PATE exceeds 1.1 in 50–90% of replications in the
    weak-instrument cells (iv = 0.3), against 0–40% at iv = 1.
  - Weak-instrument cells need roughly 4× longer chains. This is the Phase 4e ridge, which
    worsens as the instrument weakens.
- **Separated share:** 0–2% everywhere.
- **Interval widths** in realistic mode are large relative to the effects at r2 = 0.5:
  - PATE width 0.21–0.46 against an ATE of 0.05;
  - at r2 = 0.95 they shrink to 0.045–0.30.
  - Realistic effects are hard to detect, as expected. Coverage claims use diagnostic mode, as
    planned.

**Cost.**
- One chain at n = 1000 takes 6–8 s per 3000 iterations, and about 2 ms per iteration at k ≈ 1–5.
  Cost scales linearly in n.
- `effects()` on 2250 draws takes 3–4 s with 20 base-measure draws and 1–1.5 s with 5.
- Throughput is 3.4× serial with 4 processes and 4.3× with 8.
- The pilot took 63.6 minutes for 139 replications, about 27.5 s per replication effective.

## Proposed full design

The full design needs your approval before it runs.

**Common settings:**
- `marginal :common`, diagnostic mode for the coverage claims, 100 replications per cell.
  - At 95% coverage the standard error is ±0.022, so a cell at or below about 0.91 is flagged.
- Chains: 4 per replication.
  - Iterations per chain: M = 6000 at iv = 1 and M = 12000 at iv = 0.3, so most replications
    reach R-hat < 1.05.
  - Saved draws: thinned to 2000 per chain, with the first 25% discarded.
- `effects` with `n_g0 = 5`.
- Each replication records its R-hat; coverage is reported for all replications and for those
  with R-hat < 1.05.

**Cost per replication** (serial, 4 chains, including `effects`):

| n | iv = 1 (M = 6000) | iv = 0.3 (M = 12000) |
|---|---|---|
| 500 | about 30 s | about 55 s |
| 2000 | about 110 s | about 210 s |

The laptop's effective speed-up with 8 processes is about 4.3×.

**Tiers:**

| Tier | Cells | Replications | Estimated wall clock |
|---|---|---|---|
| 1 (core) | `:gaussian`, `:mixture` × n {500, 2000} × r2 {0.5, 0.95} × iv {0.3, 1} × prior {C3, auto}: 32 cells | 3200 | about 21 h |
| 2 (misspecification) | `:skewed`, `:roy` × n = 2000 × r2 {0.5, 0.95} × iv = 1 × prior {C3, auto}: 8 cells | 800 | about 6 h |
| 3 (realistic spot checks) | `:gaussian`, `:mixture` at n = 2000, r2 = 0.7, iv = 1, both priors, realistic mode: 4 cells | 200 | about 1.5 h |
| 4 (weak IV) | `:weak_iv` × n = 2000 × r2 {0.5, 0.95} × both priors: 4 cells | 400 | about 5.5 h |

**Total:** about 34 hours of laptop time for all tiers.

**Suggested order:**
1. Tier 1 at n = 2000 first. It covers the key questions (C3 vs auto at high R², mixture vs
   Gaussian) in about 14 h.
2. Then Tier 1 at n = 500.
3. Then Tiers 2, 3 and 4.

**If the budget is tight:**
- 50 replications per cell for the first look (standard error ±0.03) halves everything.
- Dropping r2 = 0.5 from Tier 1 saves a quarter, since the pilot shows no problems there beyond
  over-clustering.

**Before the full run:**
- Decide whether to look into the over-clustering at r2 = 0.5 first, since it affects the k
  summaries.
- Decide whether the weak-instrument cells should wait for option D (Phase 4g), which targets
  exactly their slow mixing.
