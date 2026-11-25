# CausalMixtures Implementation Plan

*CausalMixtures.jl · implementation plan · v2 · updated Nov 21, 2026*

> Copied from the claude.ai artifact <https://claude.ai/artifact/D95pcHqUCeSUAyH7pPzbkG>.
> The companion math review is <https://claude.ai/artifact/3koW6iaGsEvZiLYDH4RH8b>
> ("CausalMixtures Sampler Math"). Section numbers such as "review §04" refer to that page.

A phase-by-phase plan for rebuilding the samplers. It is written so that a coding session can
follow it without the earlier conversation. The math and the reasoning behind each choice are on
the companion review page.

## ✓ Progress

| Phase | Status |
|---|---|
| 0 · Baseline | ✅ Done |
| 1 · Local fixes | ✅ Done, plus follow-ups (`dpm_ppd` by name, `acf_var`, `testdpm_init`) |
| 2 · Priors | ✅ Analysis done (`docs/priors.md`). **C3 adopted**; switching the defaults is the first task of Phase 3. |
| 3 · Refactor | ✅ Done. C3 is the default; posteriors match before and after (max \|z\| 1.11 over 31 comparisons). `blocked` is 14× faster with 19× fewer allocations; `dpm` 2.8× (still dominated by bug 2). Suite 1508/1508. About 1 chain in 8 gets stuck in a low-ATE mode, both before and after. |
| 4a · Design A core + Geweke harness | ✅ Done. **Geweke:** passes in both slope modes (max \|z\| 2.09 `:common`, 2.42 `:cluster`, 500k sweeps); three planted bugs each gave \|z\| of 10–38. **Agreement:** matches `gaussian` when $k=1$ and agrees with `blocked` on ATE. **Problems found:** `blocked` is unreliable for $k$ (chains drift; R-hat 1.32 on $k$); ATE ESS is only about 150–200 per 16,000 draws for every sampler, `gaussian` included; 1 `:common` chain in 8 sticks in a low-ATE mode. **Speed:** 2.2 s per 1000 iterations, against 0.39 s for `blocked`. |
| 4b · R2 joint label/latent step | ✅ Done. **Formula fix:** the selection-index shift $g_i$ was missing; the Geweke test caught it (\|z\| 62). **Geweke:** passes (max \|z\| 1.98 `:common`, 2.11 `:cluster`); six planted bugs give \|z\| of 7–62. **Mixing:** ATE ESS up 1.8× (`:common`) and 3.8× (`:cluster`), and R-hat for $k$ is 1.000–1.001; no chain got stuck in this run. **Speed:** 0.96 s per 1000 iterations (`:common`) and 1.14 s (`:cluster`), against 0.41 s for `blocked`. |
| 4c · R3 collapsed $\sigma_{23}$ step | ✅ Done, but **no mixing gain**. **Correctness:** unit tests and Geweke pass (1.50 / 1.95 / 2.13). **Mixing:** ATE ESS is unchanged; $\sigma_{23}$ is pinned near $\sigma_{12}\sigma_{13}$ (posterior sd 0.010), so it isn't the bottleneck. The step defaults to off and is kept for the $\sigma_{23}$ sensitivity tool. **Lesson:** the coordinator's two "must fail" planted bugs were actually valid samplers: the augmented-posterior target is the ordinary Gibbs conditional, and the latent refresh redraws $y_{\text{miss}}$ anyway. |
| 4d · Hierarchical intercepts + separated-units diagnostic | ✅ Done. Hierarchical prior on by default. **Geweke:** passes (max \|z\| 2.12 `:common`, 2.88 `:cluster`); both planted bugs fail hard (\|z\| 33–94). **Effect on these data:** negligible (ATE ≤ 0.035, $k$ ≤ 0.09, separated share 0–0.2%); with $k\approx1$–1.4 there is little to pool over. **Watch item:** the two-component ATE is 5.42–5.50 against a sample ATE of 4.92 (about 1.2 sd high); Phase 6 coverage runs should check this. |
| 4e · Mixing diagnosis | ✅ Done. **Bottleneck:** one ridge per outcome arm, between the intercept and $\sigma_{1k}=\mathrm{Cov}(v,u_k)$ ($\|r\|$ 0.75–0.97). Given $d^*$, $y_k=\mathbf x'\beta_k+\sigma_{1k}(d^*-\mathbf w'\gamma)+e$; the selection residual's mean is nearly constant within an arm, so drawing β given Σ and then Σ given β zig-zags along the ridge. **Severity tracks instrument strength:** ESS is 5–9 per 1000 draws with a weak instrument. **Everything else mixes 10–50× faster:** $\gamma$, $d^*$, $y_{\text{miss}}$, $k$, $\alpha$, $\mathbf m$ and $\tau^2$. |
| 4f · Fix the ridge | ▶ In progress: **option A**, a joint complete-data draw of the outcome coefficients and $\boldsymbol\delta=(\sigma_{12},\sigma_{13})$ given $\Omega=\mathrm{Cov}(u\mid v)$, then $\Omega$ from its inverse-Wishart. **Fallback:** C, ASIS interweaving. **Kept on the roadmap:** B, the per-arm joint draw with $y_{\text{miss}}$ integrated out; revisit it after A. |
| Review of 4a–4e code | ✅ Done. **Sampler:** no correctness bugs found in Design A. **Harness gap (confirmed):** the Geweke harness overwrites the latent data at the start of each iteration, so the sweep's final latent refresh and R3's redraw were never tested, in any sampler. Fix: run the latent refresh first, add planted latent-step bugs, and re-run the long Geweke tests. **Also fixed as part of 4f:** cluster-slope statistics under `:cluster`, the repo copy of this plan, and `setup_default_priors` defaulting to `:common`, plus minor cleanups. |
| 5–7 | Not started |

## 00 Ground rules for the coding session

- Work on a branch such as `sampler-rewrite`, with one commit per task or small group of tasks.
  Stop at the end of each phase for review.
- Every behaviour change ships with a test. Statistical tests use fixed seeds through an explicit
  `rng` and tolerances set by Monte Carlo error.
- Don't change the math beyond what this plan specifies. If something in the code contradicts the
  plan, stop and report it rather than improvising.
- Target the current Julia LTS, and update `Project.toml` compat bounds (they currently say
  `julia = "1.0"`).
- Line references are against commit `e817e7b`.

## 01 Decisions already made

| Topic | Decision |
|---|---|
| Main sampler | Design A: Neal Algorithm 8 with $G$ and the cluster-level β integrated out, plus a joint label/latent step (R2) and a collapsed $\sigma_{23}$ step (R3). It replaces the current `marginal` and, once validated, `dpm`. |
| Slopes | **`slopes = :common` is the default**: covariate slopes are shared and the DPM is over the intercepts and Σ. `slopes = :cluster` is an option: the full β per cluster, as today. |
| Priors | **C3** from Phase 2, on the standardized scale: $d^*$ intercept $N(0,0.5^2)$; $y$ intercepts $N(0,1)$; probit slopes $N(0,0.5^2)$; outcome slopes $N(0,0.5^2)$ per cluster, or $N(0,1)$ when shared; $\rho=10$, $R=\mathrm{diag}(0.1,\,0.153,\,0.153)$; $\alpha\sim\mathrm{Gamma}(2,2)$. Cluster intercepts get a hierarchical prior (Phase 4). |
| Diagnostics | Every fit reports the share of units in treatment-separated clusters: clusters that are under 5% or over 95% treated. |
| Design B (Chib 2007) | Deferred, possibly indefinitely, if R3 delivers the mixing. |
| Individual gains | Later: a $\sigma_{23}$ prior-sensitivity table plus Fréchet–Makarov bounds. |
| `blocked` | Kept as an independent cross-check, with its bugs fixed. |
| `model="did"` | Roadmap only, after Design A is validated. |

## 02 Phase 0: baseline and setup

1. Create the branch. Run the existing suite (`test/runtests.jl`) and record what passes and fails.
2. Record a timing baseline: `dpm` and `blocked` at $n=500$, $M=1000$ on
   `generate_hedonic_data(500, 1234)`. Save the numbers in `bench/baseline.md`.
3. Add a test helper module with a seeded `rng` and moment-comparison helpers (mean and covariance
   with MC-error tolerances).

*Exit: a baseline report and a green or documented-red suite.*

## 03 Phase 1: local correctness fixes

These are small and independent. Each one gets a test.

| # | Fix | Where | Test |
|---|---|---|---|
| 1 | Prior β draws must use the *covariance* $V_0$. Store $V_0$ and its Cholesky factor in the input, next to the precision. | `sampler_functions.jl:479-484` | Empirical covariance of $10^5$ draws ≈ $V_0$ |
| 3 | Escobar–West: use shape $a+k$ with probability $\pi_\eta=A/(1+A)$, chosen at random. | `:310`, `:709` | Compare with a direct grid or quadrature posterior of α given $k$, $n$ |
| 5 | Compute label probabilities in log space with a log-sum-exp helper everywhere. | `:121,126,377,461,660,664` | Extreme log densities give no NaN |
| 6 | Initialise `njs` for every `1:J`, including zero counts. | `dpm_init.jl:144` | Tiny $n$ with large $J$ runs without a KeyError |
| 10 | Use `priors.prior_theta.prior_Sigma` in the covariance branch. | `dpm_init.jl:104` | Init with `Vinv=false` |
| P1 | Store new `BlockedWeights` objects on each update (or deep-copy) so saved iterations don't alias. | `gibbsstate.jl:51`, `sampler_functions.jl:428-433` | Saved `ws` differ across iterations |
| P5 | Build $\mathbf x$ and $\mathbf w$ for the PPD from the model frames by column name, not `znew[1:kx]`. Average over all $n$. | `dpm_ppd.jl:167` | `D ~ Z + X` and `D ~ X + Z` give the same PPD |
| — | Odds and ends: `-` on `BlockedWeights`, `export RawData`, broken `test=true` paths (fix or delete), pre-1.0 helpers (`rescale_output`, `check_y`, `ppd_cdf`, `make_table`, `gc()`), and `V` undefined when `Vinv=false` in the PPD. | various | Existing tests |
| — | Retire `parallel_ppd.jl`, which has wrong weights and pre-1.0 code. Mark the old `marginal` as deprecated. | `parallel_ppd.jl` | — |
| — | Port the brute-force `NobileWishart` check: compare against IW draws with $\lvert\sigma_{11}-1\rvert<0.02$ for two $(\nu,S)$ settings. | `misc_functions.jl:14` | Moments of $\sigma_{22},\sigma_{33},\sigma_{12},\sigma_{13},\sigma_{23}$ within MC error |

The reference NumPy script for the NobileWishart check is `chk/nobile.py` in the review session's
scratchpad. The coding session can rewrite it from this description.

## 04 Phase 2: prior calibration (step 0)

On standardized data, the defaults ($V_0=100I$, $\rho=6$, $R=\mathrm{diag}(1/\rho,2,2)$) make a
correct sampler almost never open clusters. The prior's $E[\sigma_{22}]\approx8$ when
$\mathrm{Var}(y)=1$, and $\gamma\sim N(0,100)$ is far too diffuse for probit slopes.

1. Write `prior_predictive(priors, X, W; n, reps, rng)`. It simulates $\alpha \to$ partition
   $\to \theta_j \to \mathbf z$ and reports: the number of clusters $k$, the distribution of
   propensities $\Phi(\mathbf w'\gamma)$, within-cluster outcome variances, and the implied spread
   of $y$.
2. Choose new defaults on the standardized scale with that tool. Starting points to tune, not
   validated values: intercepts $N(0,1)$; probit slopes $N(0,1)$; outcome slopes $N(0,1)$;
   $(\rho,R)$ such that the prior median within-cluster variance of $y$ is about 0.2–0.5; and
   $\alpha\sim\mathrm{Gamma}(2,2)$. Targets: propensities spread over $(0.05,0.95)$, and a prior on
   $k$ that is wide but not degenerate at 1.
3. Update `setup_default_priors` and document every choice in its docstring.

*Exit: prior-predictive plots and table in `docs/priors.md`, plus new defaults.*

> **Done: outcome**
>
> - **C3 adopted.** It is the only candidate with under 10% of propensity mass outside (0.05, 0.95),
>   the closest implied sd(y), the lowest new-cluster penalty (3.4 nats, against 9.8 before), and a
>   stable ATE.
> - **Treatment-separated clusters are a model-level risk** (C0 sent every unit there). That led to
>   the hierarchical intercept prior and the separated-units diagnostic.
> - A single `blocked` chain does not settle $k$, so don't compare priors on $k$.

## 05 Phase 3: state refactor (no math change)

**First task:** switch `setup_default_priors` to C3, in the current per-cluster layout. Update the
docstring, and update tests whose tolerances assumed C0.

- `labels::Vector{Int}`, counts in a `Vector{Int}`, and per-cluster member lists or sufficient
  statistics. No `Dict` scans.
- Dense `W` ($n\times k_w$) and `X` ($n\times k_x$). Never row-slice the sparse `Hmat` per unit.
- Cache `cholesky(Σ_j)` once per sweep, and use an inline 3-dimensional log density.
- Thread an explicit `rng::AbstractRNG` through init, all samplers and the PPD.
- Store output as a thinned array of $(\theta_j, n_j, \alpha)$. Saving latent data is optional and
  thinned. No per-iteration deep copies.
- Concrete field types, with no `Union{Vector,ScaleData}`.

*Exit: `gaussian`, `blocked` and `dpm` give distributionally the same results as before, with the
Phase 1 fixes applied. Record the timing against the Phase 0 baseline.*

## 06 Phase 4: Design A sampler

### Unified parameterization for both slope modes

Split each unit's design into a *cluster* part and a *global* part:

$$\mathbf z_i = H^{c}_i\,\beta^{c}_{c_i} + H^{g}_i\,b + \varepsilon_i,\qquad \varepsilon_i\sim N_3(0,\Sigma_{c_i}),\ \sigma_{11}=1.$$

- `:common`: $H^c_i=I_3$, so $\beta^c_j=(a_{d,j},a_{1,j},a_{0,j})$ holds cluster intercepts.
  $H^g_i=\mathrm{blockdiag}(\tilde{\mathbf w}_i',\tilde{\mathbf x}_i',\tilde{\mathbf x}_i')$ holds
  the shared slopes, without intercept columns.
- `:cluster`: $H^c_i=H_i$ (the full design) and there is no global part.
- Priors: $b\sim N(\mu_g,V_g)$, and Σ has the restricted IW prior. The cluster intercepts
  $\mathbf a_j$ (the intercept entries of $\beta^c_j$) get a **hierarchical prior**:
  $\mathbf a_j\sim N(\mathbf m,T)$ with $T=\mathrm{diag}(\tau_d^2,\tau_1^2,\tau_0^2)$,
  $\mathbf m\sim N(0,V_m)$ with $V_m=\mathrm{diag}(0.5^2,1,1)$, and
  $\tau_l^2\sim IG(3,\,2s_l^2)$, whose prior mean $s_l^2$ equals the C3 intercept variance.
  Under `:cluster`, the non-intercept entries of $\beta^c_j$ keep independent C3 normals. The
  collapsed predictive and the auxiliary clusters use $(\mathbf m,T)$ in place of $(\mu_c,V_c)$
  for the intercept block.

Let $\tilde{\mathbf z}_i=\mathbf z_i-H^g_ib$. With $\Sigma_j$ fixed, each cluster keeps
$Q_j=V_c^{-1}+\sum_{i\in S_j}H^{c\prime}_i\Sigma_j^{-1}H^c_i$ and
$\mathbf r_j=V_c^{-1}\mu_c+\sum_{i\in S_j}H^{c\prime}_i\Sigma_j^{-1}\tilde{\mathbf z}_i$. Build
these from Σ-free cross-products using the block identity on the review page (§04). Under
`:common`, $Q_j=V_c^{-1}+n_j\Sigma_j^{-1}$, a 3×3 matrix.

### One sweep

1. **Joint label and latent step (R2)**, for each $i$:

   (a) Remove $i$ from its cluster. If that empties the cluster, keep its $\Sigma$ as auxiliary
   slot 1. Fill the other slots up to $m$ (default 2) from the restricted $IW(\rho,\rho R)$.

   (b) For each candidate cluster, form the β-collapsed predictive
   $\tilde{\mathbf z}_i\sim N_3(\boldsymbol\mu,S)$:

   $$\boldsymbol\mu=H^c_iQ_{j,-i}^{-1}\mathbf r_{j,-i},\qquad S=\Sigma_j+H^c_iQ_{j,-i}^{-1}H^{c\prime}_i.$$

   For an auxiliary cluster, use $\mu_c$ and $V_c$ in place of the cluster posterior.

   (c) Score only the *observed* data. Let $o=2$ if $d_i=1$ and $o=3$ if $d_i=0$, let
   $s_i=2d_i-1$, and let $g_i=(H^g_ib)_d=\tilde{\mathbf w}_i'\tilde\gamma$ be the global part of
   the selection index ($g_i=0$ under `:cluster`). Then

   $$\log p_j=\log N(\tilde y_i\mid\mu_o,S_{oo})+\log\Phi\!\Big(s_i\,\frac{\mu_d+g_i+S_{do}S_{oo}^{-1}(\tilde y_i-\mu_o)}{\sqrt{S_{dd}-S_{do}^2/S_{oo}}}\Big).$$

   *Corrected in Phase 4b.* On the shifted scale $\tilde z_d=d^*-g_i$, treatment means
   $\tilde z_d>-g_i$, not $\tilde z_d>0$. The original formula omitted $g_i$; the Geweke test
   caught it under `:common` (\|z\| = 62 before the fix, 1.8 after).

   Add $\log n_{-i,j}$ for an existing cluster, or $\log(\alpha/m)$ for an auxiliary one. Here
   $\tilde y_i$ is the observed outcome minus its global part.

   (d) Sample $c_i$ with log-sum-exp. Then draw $\tilde z_{d,i}$ from its normal conditional on
   $\tilde y_i$, truncated at $-g_i$ (above it if $d_i=1$, below it otherwise), which is the same
   as drawing $d_i^*$ truncated at 0. Then draw $y_{\text{miss},i}$ from the predictive
   conditional on $(d^*_i,\tilde y_i)$. Add $i$'s statistics to its new cluster.

2. **Cluster coefficients:** $\beta^c_j\sim N(Q_j^{-1}\mathbf r_j,\ Q_j^{-1})$.
3. **Global slopes** (`:common` only): $b\sim N(P^{-1}\mathbf s,P^{-1})$ with
   $P=V_g^{-1}+\sum_iH^{g\prime}_i\Sigma_{c_i}^{-1}H^g_i$ and
   $\mathbf s=V_g^{-1}\mu_g+\sum_iH^{g\prime}_i\Sigma_{c_i}^{-1}(\mathbf z_i-H^c_i\beta^c_{c_i})$.
   Accumulate these per cluster with the block identity.
4. **Σ:** $\Sigma_j\sim$ `NobileWishart`$(\rho+n_j,\ \rho R+\sum_{i\in S_j}\mathbf e_i\mathbf e_i')$,
   where $\mathbf e_i$ is the full residual.
5. **Collapsed $\sigma_{23}$ (R3)**, for each cluster. With $y_{\text{miss}}$ integrated out,
   $\sigma_{23}$ is absent from the likelihood. Draw it from its conditional prior given the other
   entries by slice sampling, then redraw $y_{\text{miss}}$ for the cluster's members. The density
   in $\sigma_{23}$ is the restricted-IW density:

   $$p(\sigma_{23}\mid\cdot)\propto|\Sigma|^{-(\rho+4)/2}\exp\!\big(-\tfrac12\mathrm{tr}(\rho R\,\Sigma^{-1})\big),$$

   supported on the positive-definite interval

   $$\big|\sigma_{23}-\sigma_{12}\sigma_{13}\big|<\sqrt{(\sigma_{22}-\sigma_{12}^2)(\sigma_{33}-\sigma_{13}^2)}.$$

   Check the exponent $(\rho+p+1)/2$ with $p=3$ against the `NobileWishart` parameterization
   before coding.
6. **Latent refresh** (optional, cheap): one pass of the existing $y_{\text{miss}}$ and $d^*$
   conditionals (`sampler_functions.jl:241-287`, verified correct).
7. **α:** Escobar–West with the random indicator, using $k$ = number of clusters.
8. **Intercept hyperparameters**, given the $k$ cluster intercepts:

   $$\mathbf m\sim N\big(P_m^{-1}T^{-1}\textstyle\sum_j\mathbf a_j,\ P_m^{-1}\big),\quad P_m=V_m^{-1}+kT^{-1};\qquad \tau_l^2\sim IG\big(3+\tfrac k2,\ 2s_l^2+\tfrac12\textstyle\sum_j(a_{l,j}-m_l)^2\big).$$

   This lets a cluster that is all treated or all controls borrow its missing-arm intercept from
   the other clusters, instead of falling back on the raw prior.

> **Validity.** Steps 1–2 are a partially collapsed Gibbs (PCG) kernel: β<sup>c</sup> is
> marginalized in step 1 and redrawn in step 2. Step 5 marginalizes $y_{\text{miss}}$ and redraws it
> immediately. Don't condition on an old β<sup>c</sup> or an old $y_{\text{miss}}$ after the step
> that integrated it out. In step 1, $\sigma_{23}$ still enters indirectly through $Q_{j,-i}$, which
> is built from the other members' imputed data. That's expected.

*Exit: Phase 6's Geweke test passes for both slope modes, and the sampler agrees with `blocked` on
label-invariant summaries.*

## 07 Phase 5: outputs

- **Per-draw, label-invariant effects.** Draw weights
  $(\pi_1,\dots,\pi_k,\pi_0)\sim\mathrm{Dir}(n_1,\dots,n_k,\alpha)$, with $\pi_0$ covering a fresh
  $G_0$ draw estimated by a few Monte Carlo draws. Then compute:
  - **PATE** as the average over sample covariates of $\sum_j\pi_j\,\mathbf x'(\beta_{1j}-\beta_{0j})$;
  - **ATT** by averaging the closed-form TT over *treated units' own* covariates;
  - **ATU** analogously.

  Report credible intervals from the per-draw values.
- **The closed-form TT**, from review page §07 P4, with $\lambda=\phi/\Phi$:

  $$\text{TT}(\mathbf x,\mathbf w)=\frac{\sum_j\pi_j\Phi(\mathbf w'\gamma_j)\big[\mathbf x'(\beta_{1j}-\beta_{0j})+(\sigma_{12,j}-\sigma_{13,j})\lambda(\mathbf w'\gamma_j)\big]}{\sum_j\pi_j\Phi(\mathbf w'\gamma_j)}.$$

  Under `:common`, $\mathbf x'(\beta_1-\beta_0)$ becomes $(a_{1j}-a_{0j})+\tilde{\mathbf x}'(b_1-b_0)$.
- **Keep predictive draws of individual gains**, labelled clearly as depending on $\sigma_{23}$.
  Fix the README's interval wording.
- **Partition summaries:** posterior similarity matrix, a point partition (Binder or VI loss), and
  the posterior of $k$.
- **Separated-units diagnostic:** the per-draw share of units in clusters that are under 5% or over
  95% treated, with a warning when its posterior mean exceeds 5%.

## 08 Phase 6: tests and synthetic data

- **Geweke "getting it right"** test for `gaussian`, `blocked` and Design A in both slope modes.
  Use tiny $n$, tight proper priors and a proper α prior. The prior simulator must be written
  independently of `sample_prior_theta!`. Compare label-invariant statistics.
- **Unit tests:**
  - the collapsed predictive matches Monte Carlo over $\beta^c\sim N(\mu_c,V_c)$;
  - the block-identity $Q,\mathbf r$ match a dense `kron` build;
  - the R2 observed-data score matches numerical integration over $d^*$;
  - the $\sigma_{23}$ slice sampler matches the conditional from brute-force IW draws.
- **Agreement tests:** Design A with α → 0 matches `gaussian`; Design A matches `blocked` on PATE,
  ATT and $E[k]$.
- **`simulate_selection(n; scenario, rng)`**, replacing `generate_hedonic_data`:
  - *diagnostic* scenarios with well-separated components;
  - *realistic* scenarios: `:gaussian`, `:mixture` (with the school-quality flattening, so there
    are winners and losers), `:skewed`, `:roy` and `:weak_iv`;
  - ground truth computed from the potential outcomes, for both the sample and a large population
    draw.
- **Coverage study:** 100+ replications, as an offline script in `bench/`, not in CI. Use
  well-specified scenarios only for coverage claims. Don't treat $k$ as meaningful under `:skewed`.

## 09 Phase 7: documentation and roadmap

- Rewrite the README workflow, with formulas listing X terms first or built by name, correct
  interval wording, and the slope-mode option.
- Add `docs/sampler_math.md`, a condensed version of the review page.
- Roadmap, unchanged:
  - a $\sigma_{23}$ sensitivity tool plus Fréchet–Makarov bounds;
  - `model="did"`, a mixture DiD (triple-difference framing for Champaign–Urbana, with the
    few-clusters caveat);
  - Design B, only if needed.

## 10 Kickoff prompt for the coding session

Paste this into the Claude Code session on your laptop, in the repo directory:

```text
Read the implementation plan at https://claude.ai/artifact/D95pcHqUCeSUAyH7pPzbkG and the math review at https://claude.ai/artifact/3koW6iaGsEvZiLYDH4RH8b (use the Artifact tool's read action). Save a copy of the plan to docs/implementation_plan.md. Then:
1. Create branch sampler-rewrite.
2. Do Phase 0 (baseline) and Phase 1 (local correctness fixes) only, one commit per fix, each with its test.
3. Don't change math beyond the plan; if the code contradicts the plan, stop and tell me.
4. Stop after Phase 1 and summarize: what changed, test results, anything surprising.
```
