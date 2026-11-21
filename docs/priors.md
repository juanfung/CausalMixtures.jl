# Prior calibration (Phase 2)

Analysis only: `setup_default_priors` is unchanged. Reproduce with

```
julia --project=. bench/prior_calibration.jl 6000 C3
```

The simulator is `src/prior_predictive.jl` (`PriorSpec`, `prior_predictive`, `g0_logpredictive`,
`crp_num_clusters`). It is written independently of `sample_prior_theta!` and the samplers, and is
tested against brute-force Monte Carlo in `test/test_prior_predictive.jl`. Every simulation uses an
explicit seeded `Xoshiro`. The samplers don't take an `rng` yet (Phase 3), so the posterior check
seeds the global RNG before each run.

**Data.** `generate_hedonic_data(500, 1234)` with `Y_obs ~ X` and `D_obs ~ X + Z`, standardized as
the sampler does with `scale_data = (true, true)`. Covariates have mean 0 and sd 1; y has sd 1
(raw sd 3.649); the treated share is 0.512.

## Candidates

All prior means are 0. V0 is the prior covariance of β, and Σ ~ `NobileWishart(ρ, ρR)` with
R = diag(1/ρ, r, r). Because σ22 scales linearly in r, r is set so that the prior median of a
cluster's Var(y1) is 0.3.

| | intercepts (d*, y1, y0) | probit slopes | outcome slopes | ρ | r | α |
|---|---|---|---|---|---|---|
| **C0** (current) | N(0, 10²) | N(0, 10²) | N(0, 10²) | 6 | 2 | 1 fixed |
| **C1** | N(0, 1) | N(0, 1) | N(0, 1) | 6 | 0.130 | Gamma(2, 2) |
| **C2** | N(0, 1) | N(0, 1) | N(0, 0.5²) | 10 | 0.153 | Gamma(2, 2) |
| **C3** (added) | d*: N(0, 0.5²); y: N(0, 1) | N(0, 0.5²) | N(0, 0.5²) | 10 | 0.153 | Gamma(2, 2) |

**Why C3.** C1 and C2 put about a third of the prior propensity mass outside (0.05, 0.95), which
misses the plan's target. The posterior check also shows the cost: clusters that are almost all
treated or all controls. C3 is C2 with the selection equation tightened to N(0, 0.5²), so
Var(w'γ) ≈ 0.75 on standardized covariates.

Slope structures: `:cluster` draws the full β per cluster. `:common` draws the three intercepts per
cluster and one shared set of slopes. Marginally a unit's propensity, a cluster's variances, and
its effect at x̄ (where x̄ = (1, 0) after standardizing) have the same prior in both modes. So
(a)–(c) agree up to Monte Carlo error, and the slope structure only matters for (d).

## (a)–(c) Prior predictive (1000 replicates of n = 500)

- **(a)** "prop" is a unit's propensity Φ(w_i'γ_{c_i}), pooled over replicates. "Clusters with mean
  prop" is a cluster's propensity averaged over all 500 covariate rows. A value near 0 or 1 means
  a cluster whose missing arm only the prior identifies.
- **(b)** Within-cluster Var(y1) = σ22 and Var(y0) = σ33, against the total Var(y) = 1.
- **(c)** Spread over G0 draws of the cluster means x̄'β1 and x̄'β0, and of the cluster effect
  x̄'(β1 − β0).

| Candidate | slopes | prop < .05 or > .95 | clusters with mean prop < .05 or > .95 | median Var(y1) [5%, 95%] | median Var(y0) [5%, 95%] | sd of cluster mean y1 | sd of cluster mean y0 | cluster effect [5%, 95%] | median sd(y_obs) [5%, 95%] | treated share [5%, 95%] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C0 | cluster | 0.92 | 0.14 | 4.60 [1.45, 24.29] | 4.75 [1.45, 23.28] | 9.96 | 9.87 | -22.75, 23.01 | 12.10 [6.93, 18.96] | 0.20, 0.82 |
| C0 | common | 0.92 | 0.14 | 4.66 [1.42, 23.73] | 4.62 [1.41, 24.12] | 9.92 | 9.91 | -22.87, 23.45 | 11.55 [6.05, 19.91] | 0.17, 0.82 |
| C1 | cluster | 0.32 | 0.02 | 0.30 [0.09, 1.58] | 0.30 [0.09, 1.59] | 0.99 | 0.99 | -2.30, 2.34 | 1.38 [0.84, 2.10] | 0.22, 0.78 |
| C1 | common | 0.32 | 0.02 | 0.29 [0.09, 1.57] | 0.31 [0.09, 1.55] | 0.99 | 0.98 | -2.31, 2.26 | 1.34 [0.79, 2.08] | 0.21, 0.74 |
| C2 | cluster | 0.32 | 0.03 | 0.30 [0.11, 1.20] | 0.31 [0.12, 1.20] | 1.01 | 0.99 | -2.34, 2.30 | 1.10 [0.67, 1.57] | 0.24, 0.76 |
| C2 | common | 0.32 | 0.03 | 0.30 [0.11, 1.21] | 0.30 [0.12, 1.13] | 1.00 | 1.00 | -2.36, 2.31 | 1.10 [0.67, 1.58] | 0.23, 0.80 |
| C3 | cluster | 0.06 | 0.00 | 0.30 [0.11, 1.20] | 0.31 [0.12, 1.20] | 1.01 | 0.99 | -2.34, 2.30 | 1.11 [0.68, 1.59] | 0.31, 0.69 |
| C3 | common | 0.06 | 0.00 | 0.30 [0.11, 1.21] | 0.30 [0.12, 1.13] | 1.00 | 1.00 | -2.36, 2.31 | 1.10 [0.69, 1.60] | 0.31, 0.70 |

- C0's implied sd(y_obs) is about 12 on data whose sd is 1, and 92% of its propensities are
  essentially 0 or 1.
- For this data set the within-cluster target of 0.3 is generous. A single Gaussian fit leaves a
  residual variance of only 0.045–0.05 for y (R² ≈ 0.95), because the synthetic outcome is almost
  deterministic in X. Real hedonic data will have much lower R². See the open questions below.

## (d) New-cluster penalty (200 data points)

For each data point, d* is drawn from its truncated N(0, 1) prior. The missing outcome is imputed
crudely as the other arm's OLS fit plus noise at that arm's residual sd, ignoring selection.

- The **existing cluster** is a single Gaussian fit to all completed data, by per-equation OLS with
  the residual covariance: Σ̂ = [0.885 0.008 −0.004; 0.008 0.045 −0.002; −0.004 −0.002 0.05], and
  median log N(z_i | H_i β̂, Σ̂) = −0.93.
- **log p(z_i | G0)** integrates β in closed form, N(H_i μ, Σ + H_i V0 H_i'), and averages over
  2000 Σ draws. Under `:common` only the intercepts are new: the mean is H^g_i b̂, with the fitted
  shared slopes b̂, and V_c = diag of the intercept variances.
- **Gap** = existing − new. The last column adds log(α / n_j) for α = 1 and one cluster of 500.

| Candidate | slopes | median log p(z_i given G0) | median gap | gap [10%, 90%] | log odds new vs one cluster of 500, α = 1 |
| --- | --- | --- | --- | --- | --- |
| C0 | cluster | -10.75 | 9.79 | 7.92, 11.26 | -16.00 |
| C0 | common | -9.75 | 8.82 | 7.16, 9.77 | -15.03 |
| C1 | cluster | -4.74 | 3.79 | 2.10, 5.02 | -10.00 |
| C1 | common | -3.97 | 3.21 | 1.57, 3.86 | -9.42 |
| C2 | cluster | -4.50 | 3.63 | 2.00, 4.64 | -9.84 |
| C2 | common | -3.96 | 3.20 | 1.56, 3.85 | -9.41 |
| C3 | cluster | -4.20 | 3.40 | 1.80, 4.24 | -9.62 |
| C3 | common | -3.77 | 3.15 | 1.59, 3.65 | -9.37 |

C0's penalty of about 10 nats matches the review's estimate (log ∫ f dG0 ≈ −9.9). The calibrated
priors cut the penalty by about 6 nats. After that, log(α/n_j) ≈ −6.2 dominates, which is
appropriate for a point the big cluster already fits well. `:common` is a further ~0.3–1 nat
cheaper, because a new cluster only has to explain an intercept.

## (e) CRP prior on k (context)

| α prior | n | E[k] | median k | k [5%, 95%] |
| --- | --- | --- | --- | --- |
| α = 1 fixed | 500 | 6.76 | 7 | 3, 11 |
| α = 1 fixed | 2000 | 8.21 | 8 | 4, 13 |
| α ~ Gamma(2, 2) | 500 | 6.64 | 6 | 1, 14 |
| α ~ Gamma(2, 2) | 2000 | 7.84 | 7 | 2, 17 |

## (f) Posterior check: blocked sampler (`slopes = :cluster`)

J = 20, M = 6000, burn-in 2000. I ran longer than the suggested M = 2000 with burn-in 500 because
traces under C1 were still drifting at iteration ~1500 (ATE moved from 2.0 to 5.3). Each run takes
about 35 s.

- **Two-component data:** the same generator as `generate_hedonic_data` (x, z ~ U(0, 1), the same
  γ, β and Σ), except that half the units have both outcome intercepts shifted by +2, which is
  2 error sds. Sample ATE 4.920; population ATE 5.
- **Hedonic data:** sample ATE 5.155; population ATE 5.
- **ATE columns:** per draw, sy·Σ_j π_j x̄'(β1j − β0j), with π_j either the stick-breaking weights
  ("G weights") or n_j/n over occupied clusters.
- **"Units in separated clusters":** the share of units whose cluster is < 5% or > 95% treated.

| Data | Candidate | mean occupied k | k [5%, 95%] | mean α | ATE (G weights) mean [2.5%, 97.5%] | ATE (occupied) mean [2.5%, 97.5%] | ATE (occupied) 1st / 2nd half | units in separated clusters |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hedonic | C0 | 2.50 | 2, 4 | 1.00 | 9.55 [-1.00, 24.20] | 9.63 [-0.84, 25.03] | 9.78 / 9.48 | 1.00 |
| hedonic | C1 | 2.44 | 1, 4 | 0.81 | 5.24 [4.80, 5.73] | 5.29 [4.85, 5.77] | 5.32 / 5.26 | 0.00 |
| hedonic | C2 | 2.24 | 1, 4 | 0.65 | 5.19 [4.73, 5.60] | 5.23 [4.77, 5.63] | 5.24 / 5.21 | 0.00 |
| hedonic | C3 | 2.20 | 1, 4 | 0.67 | 5.20 [4.70, 5.65] | 5.24 [4.75, 5.69] | 5.22 / 5.25 | 0.00 |
| two-component | C0 | 2.50 | 2, 4 | 1.00 | 10.75 [-4.27, 24.26] | 11.11 [-4.35, 24.72] | 14.93 / 7.29 | 1.00 |
| two-component | C1 | 3.23 | 2, 5 | 0.66 | 5.13 [3.28, 7.33] | 5.17 [3.35, 7.34] | 4.87 / 5.46 | 0.21 |
| two-component | C2 | 3.08 | 1, 5 | 0.69 | 5.41 [4.72, 6.12] | 5.44 [4.75, 6.16] | 5.42 / 5.46 | 0.01 |
| two-component | C3 | 9.25 | 3, 14 | 7.67 | 5.24 [4.39, 6.22] | 5.37 [4.55, 6.28] | 5.25 / 5.49 | 0.02 |

**Seed sensitivity, two-component data.** Each row is a separate chain started from a different
global seed. (The +0 rows differ from the table above only because r was rounded to 0.153 here;
the chains are that sensitive.)

| Candidate | seed | mean occupied k | k [5%, 95%] | mean α | ATE (occupied) mean [2.5%, 97.5%] | 1st / 2nd half | separated |
| --- | --- | --- | --- | --- | --- | --- | --- |
| C2 | +0 | 8.19 | 5, 11 | 1.92 | 5.18 [4.44, 5.91] | 5.08 / 5.28 | 0.02 |
| C2 | +100 | 4.85 | 2, 8 | 1.13 | 5.46 [4.77, 6.20] | 5.53 / 5.39 | 0.01 |
| C2 | +200 | 1.17 | 1, 2 | 0.29 | 5.64 [4.80, 6.34] | 5.58 / 5.70 | 0.00 |
| C3 | +0 | 2.52 | 1, 4 | 0.53 | 5.47 [4.67, 6.29] | 5.34 / 5.59 | 0.00 |
| C3 | +100 | 1.17 | 1, 2 | 0.28 | 5.49 [4.61, 6.55] | 5.68 / 5.30 | 0.00 |
| C3 | +200 | 1.21 | 1, 2 | 0.29 | 5.49 [4.70, 6.37] | 5.47 / 5.51 | 0.00 |

**Reading (f).**

1. **C0 fails.** Every unit ends up in a cluster that is all treated or all controls. Each such
   cluster's probit intercept is about ±7.5, and its missing arm has σ of about 7–10, because only
   the N(0, 100) prior pins it down. So the ATE wanders over [−1, 24] (hedonic) and [−4, 24]
   (two-component), even though the gaussian sampler gets 5.15–5.20 on the same data.
2. **The separation is a posterior phenomenon that the probit prior controls.** The prior puts only
   2–3% of C1 and C2 clusters there, but under C1 21% of units end up in separated clusters on the
   two-component data. A cluster that is all treated fits y1 freely, and the prior is the only cost
   for its unobserved y0.
3. **C2 and C3 give stable ATEs** (5.2–5.6, every interval covering the sample ATE 4.92) with ≤ 2%
   separated units.
4. **The number of clusters is not identified by a single blocked chain.** For one data set and
   prior, mean k ranges from 1.2 to 8–9 across seeds, with α following it. This is a mixing
   problem: the k-trace is effectively stuck within each chain. It is the motivation for Design A
   (R2/R3), not something priors can fix. Don't compare candidates on k from these runs.

## Recommendation

**C3.** Intercepts N(0, 1) for y1 and y0 and N(0, 0.5²) for d*; probit and outcome slopes
N(0, 0.5²); ρ = 10, R = diag(1/10, 0.153, 0.153) (prior median within-cluster Var(y) = 0.3); and
α ~ Gamma(2, 2).

- **It is the only candidate that meets the plan's propensity target:** 6% of mass outside
  (0.05, 0.95), against 32% for C1 and C2 and 92% for C0. It puts essentially no prior mass on
  clusters that are all treated or all controls, which is the failure mode that wrecks C0 and
  degrades C1 in the posterior check.
- **Its implied sd(y_obs) is 1.10 [0.68, 1.60]**, against the observed 1. C1 gives 1.38 and C0 gives 12.
- **It has the lowest new-cluster penalty:** 3.40 nats under `:cluster` and 3.15 under `:common`,
  against 9.8 for C0.
- **Its ATE posterior is stable across seeds** and covers the truth. C2 is close behind. C3 differs
  only in the probit prior, which is the part (a) and (f) say matters.

## Open questions for review

1. **Treatment-separated clusters are a model-level risk, in both slope modes.** A cluster-specific
   probit intercept lets a cluster be all treated or all controls, leaving its missing arm
   prior-driven. C3 makes that unlikely, but a strong enough signal in real data could still pull
   it there. Options for Phase 4:
   - a shared probit intercept under `:common`;
   - reporting the "units in separated clusters" diagnostic with every fit;
   - a hierarchical prior.
2. **Within-cluster variance target.** 0.3 is generic. This synthetic DGP has a residual variance
   of about 0.05, so a cluster's Σ is far tighter than the prior median. That is harmless here
   (ρ = 10 lets n_j dominate), but it argues for making the `simulate_selection` scenarios in
   Phase 6 realistic (R² ≈ 0.5–0.8) before fine-tuning r.
3. **`:common` priors.** Phase 4 needs V_c (intercepts) and V_g (shared slopes). The C3 values map
   directly: V_c = diag(0.5², 1, 1) and V_g = diag of the slope variances.
