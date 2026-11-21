<!-- Generated from the review page https://claude.ai/artifact/3koW6iaGsEvZiLYDH4RH8b (Nov 21, 2026).
     Math uses GitHub $...$ / $$...$$ syntax. The companion plan is docs/implementation_plan.md. -->

# CausalMixtures Sampler Math

The model as coded, what each sampler is, the math errors, and a design for a fast marginal sampler. **Status (Nov 21):** Phases 0–2 of the [implementation plan](https://claude.ai/artifact/D95pcHqUCeSUAyH7pPzbkG) are done on branch `sampler-rewrite`. Progress and Phase 2 findings are in §13, and the full bug list with fix status is in Appendix A.

## §00 Your answers, and what follows

*“Integrate out G or β — don't remember which (or both)”*

**Both.** Design A (§4) integrates out $G$ (the Pólya-urn view) and also integrates out $\beta$ analytically. $\Sigma$ stays in the state and new clusters get auxiliary $\Sigma$ draws (Neal's Algorithm 8). *Corrected after review:* the obstacle to integrating Σ is the *independent* (semi-conjugate) prior on β, not the $\sigma_{11}=1$ restriction. The restricted IW is itself Normal–IW in $(\sigma_{12},\sigma_{13})$ and $\mathrm{Cov}(u\mid v)$.

*“What would be the value of keeping the individual gains?”*

See §5. In short: you need the joint law of $(y_1,y_0)$ only for questions about the *distribution* of gains, such as who gains and by how much. You can run the faster sampler and still answer those questions by drawing $\sigma_{10}$ afterwards.

------------------------------------------------------------------------

## §01 The model as implemented

For unit $i$, the latent vector, its regression, and the observation rule are:

$$\mathbf z_i=(d_i^*,\,y_{1i},\,y_{0i})' = H_i\beta_{c_i}+\varepsilon_i,\qquad \varepsilon_i\sim N_3(0,\Sigma_{c_i}),\qquad \sigma_{11}=1$$

$$H_i=\begin{pmatrix}\mathbf w_i'&0&0\\0&\mathbf x_i'&0\\0&0&\mathbf x_i'\end{pmatrix},\qquad \beta=(\gamma',\beta_1',\beta_0')',\qquad K=k_w+2k_x$$

Observed: $d_i=1(d_i^*>0)$ and $y_i=d_iy_{1i}+(1-d_i)y_{0i}$. Mixing: $\theta_i=(\beta_i,\Sigma_i)\mid G\sim G$, $G\sim DP(\alpha,G_0)$, where

$$G_0:\quad \beta\sim N(\mu,V_0),\qquad \Sigma\sim IW(\rho,\rho R)\ \text{restricted to}\ \sigma_{11}=1.$$

The sampler augments $d_i^*$ (truncated normal) and the missing potential outcome $y_{\text{miss},i}$ (normal).

**Verified correct**

- The $y_{\text{miss}}\mid d^*,y_{\text{obs}}$ and $d^*\mid y_1,y_0$ conditionals: means and variances match the 3×3 normal.
- $\Sigma_j\mid\beta_j\sim IW(\rho+n_j,\ \rho R+\sum e e')$, restricted.
- `NobileWishart`: the Bartlett factor with $A_{33}=1/L_{33}$ gives $\Sigma_{33}=1$ after the flip. Imai & van Dyk (2005) report errors in Nobile's MNP derivation. I haven't checked whether that applies to this step, so a Monte Carlo check is in the test plan.
- The blocked stick-breaking update for $v_j$ and the blocked $\alpha$ update.

## §02 What each sampler actually is

| model=       | Literature name                                                                | Status                                                                                 |
|--------------|--------------------------------------------------------------------------------|----------------------------------------------------------------------------------------|
| `"dpm"`      | Pólya-urn sampler with $G$ integrated out (Neal 2000, non-conjugate) | Already a “marginal” sampler. Its new-cluster step is wrong (bug 2).                   |
| `"blocked"`  | Truncated stick-breaking (Ishwaran & James 2001)                               | Structure is right. Prior draws are wrong (bug 1).                                     |
| `"fmn"`      | Finite mixture, $Dir(\alpha/J)$ weights integrated out                           | Empty components go stale (bug 4).                                                     |
| `"marginal"` | Mix of `fmn` and `dpm`                                                         | Inherits bug 2. Its PPD and $\alpha$ update don't match the finite model it runs. |
| `"gaussian"` | $J=1$ benchmark                                                        | Fine. Use it as the baseline for correctness checks.                                   |

## §03 Math errors that matter most

- **1 · Prior β draws use the precision as the covariance**  
  `sampler_functions.jl:480 · sample_prior_theta!`

  `prior_beta.V` holds $V_0^{-1}$ after `dpm_init`, but the draw is $\mu+\mathrm{chol}(V_0^{-1})'\,\xi$. With defaults, β gets sd 0.1 instead of 10. This affects initial values and every empty component in `blocked`, so that sampler's stationary distribution is not the stated model.

- **2 · New-cluster probability is biased upward**  
  `sampler_functions.jl:63–78, 125 · used by dpm and marginal`

  The correct weight is

  $$\Pr(c_i=\text{new})\ \propto\ \alpha\int f(\mathbf z_i\mid\theta)\,dG_0(\theta).$$

  The code draws $\theta^\star$ from roughly $p(\theta\mid\mathbf z_i)$ and uses $\alpha f(\mathbf z_i\mid\theta^\star)$. A posterior draw fits $\mathbf z_i$ by construction, so the weight is too large. You get too many tiny clusters, near-singular $\Sigma$ updates, and the wrong posterior. This is the likely cause of the “unstable” marginal sampler.

- **3 · Escobar–West α step is deterministic**  
  `sampler_functions.jl:310, 709`

  It should draw $\text{shape}=a+k$ with probability $\pi_\eta=A/(1+A)$, otherwise $a+k-1$. The code instead picks the larger one whenever $\pi_\eta>0.5$.

- **4 · `fmn` never redraws empty components (downgraded: mixing, not correctness)**  
  `sampler_functions.jl:158`

  *Corrected after review:* skipping θ for an empty component leaves the target invariant, and the stale values are valid posterior draws. Redrawing them from $G_0$ only helps mixing.

- **5 · Probabilities computed as exp(logpdf)**  
  `sampler_functions.jl:121, 126, 377, 461, 660, 664`

  These can underflow to all zeros and produce NaN in `Categorical`. Sample with log-sum-exp instead.

**Interpretation**

`quantile(tes.ate, [.025,.975])` in the README is the predictive spread of an *individual's* gain $y_1-y_0$. It is not a credible interval for the ATE, and it depends on the unidentified $\sigma_{10}$. A credible interval for the ATE comes from per-draw values:

$$\text{ATE}^{(m)}=\sum_j \pi^{(m)}_j\,\bar{\mathbf x}'\big(\beta^{(m)}_{1j}-\beta^{(m)}_{0j}\big).$$

Bugs 6–13 (PPD indexing, blocked init, the `test=true` paths, formula-order slicing and others) are in Appendix A, which also gives the fix status of every bug.

## §04 Design A: integrate out $G$ and $\beta$

This keeps the same model and prior and replaces both `dpm` and the current `marginal`. The state is: labels $c$, $\Sigma_j$ per cluster, latent $\mathbf z$, and $\alpha$. β is regenerated each sweep.

### Cluster statistics that do not depend on Σ

Because $H_i$ is block-diagonal, with rows $\mathbf h_{1i}=\mathbf w_i$ and $\mathbf h_{2i}=\mathbf h_{3i}=\mathbf x_i$:

$$\sum_{i\in S_j}H_i'\Sigma^{-1}H_i=\Big[(\Sigma^{-1})_{ab}\textstyle\sum_{i\in S_j}\mathbf h_{ai}\mathbf h_{bi}'\Big]_{a,b},\qquad \sum_{i\in S_j}H_i'\Sigma^{-1}\mathbf z_i=\Big[\textstyle\sum_b(\Sigma^{-1})_{ab}\sum_{i\in S_j}\mathbf h_{ai}z_{bi}\Big]_a$$

Each cluster therefore keeps $\sum\mathbf w\mathbf w'$, $\sum\mathbf w\mathbf x'$, $\sum\mathbf x\mathbf x'$, $\sum\mathbf w\mathbf z'$ and $\sum\mathbf x\mathbf z'$. Adding or removing a unit costs $O(K^2)$. From these,

$$Q_j=V_0^{-1}+\sum_{i\in S_j}H_i'\Sigma_j^{-1}H_i,\qquad \mathbf r_j=V_0^{-1}\mu+\sum_{i\in S_j}H_i'\Sigma_j^{-1}\mathbf z_i,\qquad \beta_j\mid\cdot\sim N(Q_j^{-1}\mathbf r_j,\ Q_j^{-1}).$$

This also replaces the $3n_j\times3n_j$ `kron` in the current β update.

### Predictive densities with β integrated out

Existing cluster $j$, with unit $i$ removed:

$$p(\mathbf z_i\mid \text{rest of }j,\Sigma_j)=N_3\big(\mathbf z_i\mid H_iQ_{j,-i}^{-1}\mathbf r_{j,-i},\ \Sigma_j+H_iQ_{j,-i}^{-1}H_i'\big)$$

Auxiliary (new) cluster with $\Sigma^{\text{aux}}\sim G_{0,\Sigma}$:

$$p(\mathbf z_i\mid\Sigma^{\text{aux}})=N_3\big(\mathbf z_i\mid H_i\mu,\ \Sigma^{\text{aux}}+H_iV_0H_i'\big)$$

$H_i\mu$ and $H_iV_0H_i'$ never change, so precompute them once.

### One sweep

1.  **Labels.** For each $i$: remove it from its cluster and subtract its statistics. If the cluster is now empty, keep its $\Sigma$ as auxiliary slot 1. Fill the other $m-1$ slots with draws from `NobileWishart(ρ, ρR)`. Then sample with log weights

    $$\log n_{-i,j}+\log N_3(\text{existing }j),\qquad \log(\alpha/m)+\log N_3(\text{aux }k),$$

    add $i$ to the chosen cluster, and refactor $Q$ only for the two clusters that changed. Cost per sweep: $O\big(n(J+m)K^2\big)$.

2.  **β.** $\beta_j\sim N(Q_j^{-1}\mathbf r_j,\ Q_j^{-1})$ for each cluster.

3.  **Σ.** $\Sigma_j\sim$ restricted $IW\big(\rho+n_j,\ \rho R+\sum_{i\in S_j}\mathbf e_i\mathbf e_i'\big)$ with $\mathbf e_i=\mathbf z_i-H_i\beta_j$. Steps 2 and 3 can be repeated once or twice.

4.  **Latent data.** Draw $y_{\text{miss}}$, then $d^*$, using the existing conditionals (already verified). Then recompute the cluster statistics.

5.  **α.** Escobar–West with a random mixture indicator, using $k$ = number of clusters.

**Why this is valid**

It is a partially collapsed Gibbs sampler (van Dyk & Park 2008). Step 1 samples $c\mid\Sigma,\mathbf z$ with β integrated out. Step 2 then draws β from its full conditional. This is a valid “marginalize, then trim” partially collapsed kernel. Step 1 is a sequential sweep, not a single blocked draw. Order matters: never condition on the *old* β after step 1. The Σ update also needs the cluster's $\sum\mathbf z\mathbf z'$ (3×3) as a sufficient statistic.

## §05 Individual gains: what they buy you

Most treatment-effect summaries depend only on the marginal law of each potential outcome and the selection equation. None of these need $\sigma_{10}$:

- ATE, ATT, ATU, LATE and MTE-type means.
- Quantile treatment effects, meaning differences between the marginal quantiles of $y_1$ and $y_0$.
- The full predictive distribution of $y_1$ and of $y_0$ separately.

These require the **joint** law of $(y_1,y_0)$, and so $\sigma_{10}$:

- The share who benefit, $\Pr(y_1>y_0)$, or the share harmed.
- Quantiles and the variance of the individual gain $y_1-y_0$, i.e. how heterogeneous gains are and whether people sort on them (the Roy-model question).
- A predicted gain distribution for a particular person, such as the predicted loss for a given dropout. This is the question in the title of Li, Poirier & Tobias.

Poirier & Tobias (2003) show that $\sigma_{10}$ is not identified. The data inform it only through the other parameters, via the positive-definiteness constraint. When that learning is weak, the spread of the gain distribution mostly reflects the prior.

**Key consequence for the sampler**

The observed-data likelihood never involves $\sigma_{10}$, which is $\sigma_{23}$ in the code's indexing $(d^*,y_1,y_0)$. Given everything *except* $y_{\text{miss}}$, its posterior is its prior, conditioned on the identified entries:

$$p(\sigma_{23,j}\mid \text{data},\text{rest}\setminus y_{\text{miss}})=p(\sigma_{23,j}\mid \sigma_{12,j},\sigma_{13,j},\sigma_{22,j},\sigma_{33,j}).$$

The restricted-IW prior already enforces positive definiteness. Given $y_{\text{miss}}$, $\sigma_{23}$ *is* informed, and that coupling is the slow part.

So you can sample *without* $y_{\text{miss}}$ (Design B), which removes the slowest-mixing part of the chain, and then draw $\sigma_{10}$ for each saved iteration and cluster in a one-dimensional post-processing step. That recovers the gain distribution without slowing the sampler down.

### Design B (optional): drop $y_{\text{miss}}$ (Chib 2007)

For each arm $k\in\{0,1\}$, let $\sigma_k=\mathrm{Cov}(u_k,v)$ and $\omega_k=\mathrm{Var}(u_k)-\sigma_k^2$. Then

$$d_i^*=\mathbf w_i'\gamma+v_i,\quad v_i\sim N(0,1);\qquad y_i\mid d_i^*,d_i=k\ \sim\ N\big(\mathbf x_i'\beta_k+\sigma_k(d_i^*-\mathbf w_i'\gamma),\ \omega_k\big).$$

With a Normal–Inverse-Gamma prior on $(\beta_k,\sigma_k,\omega_k)$, each arm is conjugate given $(\gamma_j,d^*)$. You can integrate it out (Student-$t$ predictive) and use auxiliary draws for $\gamma$ only, which is close to a fully collapsed sampler. The cost is a different prior on Σ. Matching the restricted 3×3 IW exactly would require the conditional above, which is non-standard. You would instead put a prior on $\rho_{10}$ directly, for example uniform on its PD-feasible interval.

**Mixture caveat**

In a mixture, the shared label $c_i$ also links $y_1$ and $y_0$: units in different clusters have different $\beta_1-\beta_0$. So the gain distribution has a between-cluster part as well as a within-cluster $\sigma_{10}$ part. The between-cluster part rests on the assumption that a unit has the same cluster in both arms. Treat it as model-driven, like $\sigma_{10}$.

## §06 How to know it's right

- **Geweke “getting it right” test** on tiny $n$: alternate simulating data given parameters with one sweep given data, then compare the moments of $\alpha$, $k$, β and Σ with the prior. This catches bugs 1–4.
- **Unit checks:** prior β draws have covariance $V_0$; the collapsed predictive matches a Monte Carlo average of $f(\mathbf z_i\mid\beta,\Sigma)$ over $\beta\sim N(\mu,V_0)$; the sufficient-statistic $Q_j$ matches the current `kron` result; and `NobileWishart` moments match a brute-force sampler.
- **Agreement:** `gaussian` ≈ Design A as $\alpha\to0$, and `blocked` ≈ Design A on the posterior ATE and $E[k]$.
- **Recovery with real tolerances:** the current ATE test passes anywhere within 8 of a true value near 5. Replace it with a coverage check of the 95% credible interval across about 20 seeds.

## §07 Posterior predictive code review

The target, for a new unit with design row $h$ and each saved MCMC state, is

$$p(\mathbf z_{\text{new}}\mid\text{state})=\sum_j \frac{n_j}{n+\alpha}N\big(h\beta_j,\Sigma_j\big)+\frac{\alpha}{n+\alpha}\int N\big(h\beta,\Sigma\big)\,dG_0 .$$

To sample it, pick a component with probability $n_j/(n+\alpha)$, or the prior with probability $\alpha/(n+\alpha)$, then draw $\mathbf z_{\text{new}}$ from that component.

| Function                  | Verdict                                                                                                                                                                                                                          |
|---------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `rand_dpm`                | ✅ Correct. Weights are $n_j$ and $\alpha$ and don't depend on the drawn value. Prior β uses the inverted precision, which is right here. It is wasteful, though: it draws from every component and then picks one. |
| `rand_gaussian`           | ✅ Correct.                                                                                                                                                                                                                      |
| `rand_blocked`            | ⚠️ The logic is right, but the saved weights are wrong. See P1.                                                                                                                                                                  |
| `rand_fmn`                | ✅ The weights $n_j+\alpha/J$ are right. Empty components carry old but valid θ (bug 4 was downgraded).                                                                                                                           |
| `rand_dpm` for `marginal` | ❌ Filters with the last iteration's $n_j$ and loops only $1{:}J_{\text{active}}$ over sparse labels.                                                                                                                              |
| `parallel_ppd.jl`         | ❌ Wrong weights (P2), and pre-1.0 code (`chol`, `blkdiag`, `Array(Float64,0)`, `readcsv`) that won't run.                                                                                                                       |
| `dpm_ate`                 | ✅ for the ATE. ⚠️ for TT (P3).                                                                                                                                                                                                  |

- **P1 · Every saved iteration shares the same blocked weights**  
  `gibbsstate.jl:51 copy(StateDP) · sampler_functions.jl:428–433 update_ws!`

  `copy(::StateDP)` copies the `ws` OrderedDict shallowly, and `BlockedWeights` is mutable. `update_ws!` then changes the existing objects in place (`s.ws[j].v = …`). So every `out.out_dp[m].ws` points to the same objects, and they all hold the *final* iteration's weights. The blocked PPD therefore mixes iteration-$m$ θ with last-iteration weights. Fix: store new `BlockedWeights` objects each update, or use `deepcopy`. θ and the latent data don't have this problem, because they are replaced, not mutated.

- **P2 · parallel_ppd weights by the density of the drawn value**  
  `parallel_ppd.jl:36, 44, 62, 78`

  It uses $w_j\propto n_j\,f(\tilde{\mathbf z}_j\mid\theta_j)$, where $\tilde{\mathbf z}_j$ is the value just drawn from component $j$. That favours components that happened to produce high-density draws, which in practice means tight Σ, so the result is not the predictive distribution. Commit `7f789f1` fixed this in `rand_*` but not here. Retire this file or rebuild it on the fixed `rand_*`.

- **P3 · TT is evaluated at mean covariates, which is not the ATT**  
  `dpm_ate · dpm_ppd.jl:181–188`

  For the mixture, the ATE is linear in $x$, so the ATE at $\bar{\mathbf x}$ equals the average ATE. TT is not linear in $x$ because of the selection term, so TT at $\bar{\mathbf w}$ is not the population ATT. Average over the treated units' own covariates instead.

- **P4 · One noisy draw per iteration where an exact value is available**

  `mean(tes.ate)` is a valid estimate, but its Monte Carlo error includes the outcome noise Σ, and its quantiles are the spread of individual gains, not a credible interval. Given the state, both effects have closed forms:

  $$\text{ATE}(\mathbf x)=\sum_j \pi_j\,\mathbf x'(\beta_{1j}-\beta_{0j})$$

  $$\text{TT}(\mathbf x,\mathbf w)=\frac{\sum_j \pi_j\,\Phi(\mathbf w'\gamma_j)\big[\mathbf x'(\beta_{1j}-\beta_{0j})+(\sigma_{12,j}-\sigma_{13,j})\,\lambda(\mathbf w'\gamma_j)\big]}{\sum_j \pi_j\,\Phi(\mathbf w'\gamma_j)},\qquad \lambda=\phi/\Phi$$

  Here $\pi_j$ is $n_j/(n+\alpha)$, or the blocked $w_j$. The extra $\alpha/(n+\alpha)$ term uses prior means for the ATE and a few Monte Carlo draws from $G_0$ for TT. Computing these per draw gives true credible intervals for ATE and ATT, with less Monte Carlo noise. Keep the current per-draw sampling of individual gains only for the gains distribution.

- **P5 · `znew` construction**

  Outcome rows come from `znew[1:kx]`, which assumes `D ~ X + Z` order, while the README shows `Z + X`. The README and tests also average only `Hmat[1:100, …]`. Build $\mathbf x$ and $\mathbf w$ from the model frames by name, and average over all $n$, or over treated units for TT.

## §08 Use cases (draft)

The model fits when all of these hold: a **binary** treatment chosen by the unit, a **continuous** outcome, at least one variable that shifts the choice but not the outcome (an exclusion restriction), and a reason to doubt normal errors or a single common effect. Without an excluded instrument, identification rests on functional form, which is weak here because the error distribution is the flexible part.

| Setting                   | Treatment $d$                                                   | Outcome $y$             | Candidate instrument $w$                                     |
|---------------------------|---------------------------------------------------------------------------|-----------------------------------|------------------------------------------------------------------------|
| School choice and housing | Home is assigned to, or uses, a choice school instead of the zoned school | Log sale price, or test scores    | Distance to the nearest choice school; lottery offer; program boundary |
| Returns to schooling      | Completes high school or college                                          | Log wage                          | Distance to college; local tuition; local labour market at age 17      |
| Union or sector premium   | Union member                                                              | Log wage                          | Industry and region coverage rates                                     |
| Technology adoption       | Adopts a practice or seed                                                 | Yield or farm income              | Access to extension services; information shocks                       |
| Health treatment choice   | Intensive vs standard treatment                                           | Cost or a continuous health index | Distance to a specialty facility                                       |

What the DPM adds over a Gaussian selection model:

- Skewed or heavy-tailed outcomes, such as prices and earnings, without a transformation.
- Latent subpopulations with different effects and different selection, for example neighbourhood or household types.
- Distributional summaries: quantile effects, plus the share who benefit or are harmed, reported with the $\sigma_{10}$ sensitivity analysis.

The closest published methods: Chib & Hamilton's DPM treatment model (union premium), and Conley, Hansen, McCulloch & Rossi (2008) DPM instrumental variables. The last two rows of the table are illustrative and not tied to a specific paper.

## §09 A more realistic synthetic data generator (draft)

The current `generate_hedonic_data` draws errors from a **single** multivariate normal. That gives the mixture nothing to find, and a Gaussian model is already correct for it. It also builds unused columns (`F`, `Y_dummy`), uses uniform(0, 1) covariates and effects about 100× the noise, and calls the global `Random.seed!`.

### Proposed structure

`simulate_selection(n; scenario, rng, …)` keeps one structural template and switches the error law and heterogeneity by scenario:

$$d_i^*=\gamma_0+\mathbf x_i'\gamma_x+\gamma_w w_i+v_i,\qquad y_{ki}=\beta_{k0}+\mathbf x_i'\beta_{kx}+u_{ki},\quad k\in\{0,1\}.$$

- **Covariates**, with a hedonic flavour: log square footage, house age, lot size, bedrooms and neighbourhood income, correlated through a Gaussian copula. Realistic scales: log price around 12, effects of 0.02–0.10 log points against noise of about 0.3.
- **Instrument** $w$: distance to the nearest choice school (exponential) or a lottery offer (Bernoulli). It is excluded from both outcome equations, and its strength is adjustable.
- **RNG**: an explicit `rng::AbstractRNG` argument, so the global seed is untouched and tests can be reproduced.

### Scenarios

| Scenario    | Errors and heterogeneity                                                                                     | What it tests                                                                  |
|-------------|--------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------|
| `:gaussian` | One trivariate normal (today's generator)                                                                    | Baseline; all samplers should agree with `gaussian`                            |
| `:mixture`  | Three latent types with their own $\beta_1-\beta_0$ and Σ; one type is harmed on average                         | Recovering clusters, ATE vs ATT vs ATU, and the share harmed                   |
| `:skewed`   | Skew-normal or skew-$t$ outcome errors and heavy-tailed $v$; not a finite normal mixture | DPM flexibility under misspecification, which is the realistic case for prices |
| `:roy`      | Units select on their own gains: $v$ strongly correlated with $u_1-u_0$                        | Essential heterogeneity: ATT ≠ ATE ≠ ATU                                       |
| `:weak_iv`  | Instrument strength near zero                                                                                | What breaks when identification rests on functional form                       |

### Hedonic flavour, kept light

The scenarios stay generic. One mechanism from the school-choice setting is still worth borrowing for `:mixture`: give each type a latent zoned-school quality $q$, and let the gain be $y_1-y_0\approx\delta-\kappa(q-\bar q)$. Choice flattens the school-quality premium (Black 1999 puts that premium at roughly 2–2.5% of price per 5% in test scores), so high-$q$ types lose and low-$q$ types gain. That produces a realistic share harmed without a separate engineered scenario. Use a sample-size grid such as $n\in\{1{,}000,\ 3{,}000,\ 10{,}000\}$.

### Ground truth returned

Truth is computed from the simulated potential outcomes, not from formulas, so it works for every scenario:

- **Sample truth:** ATE, ATT, ATU, the share with $y_1>y_0$, quantiles of $y_1-y_0$, quantile treatment effects, and CATE by covariate group.
- **Population truth:** the same quantities from one large draw ($N=10^6$) of the same data-generating process, which is what credible intervals should cover.
- The true type labels in `:mixture`, for checking clusters, and the formulas with `X` terms listed first, so P5 cannot bite.

## §10 Applying the model to the Champaign–Urbana study

**Design:** house sales in Champaign and Urbana before and after 1997. Champaign moved to controlled choice and Urbana kept neighbourhood zoning. The outcome is log sale price, the treatment is *a post-1997 sale in Champaign*, and school zones come from hand-built maps. This is a difference-in-differences natural experiment. The selection model in this package rests on different assumptions, which raises these issues:

- **1 · Treatment is determined by place and date**

  $d=\text{Champaign}\times\text{post}$. If the city and period indicators are in the selection equation, $d^*$ is perfectly predicted, which gives probit separation and a degenerate selection equation. If they are left out of the price equation, city and period effects fall into the errors and are read as selection.

- **2 · No credible exclusion restriction**

  Champaign-specific attributes (taxes, services, amenities) plausibly affect prices directly, so they can't be excluded from the price equation. In this design the identifying assumption is parallel trends, not an excluded instrument.

- **3 · The estimand doesn't match**

  The control group pools Champaign pre-1997 sales with all Urbana sales. The selection model has no notion of a before–after comparison, so its ATT is not the DiD ATT.

- **4 · Market-level treatment and spillovers**

  Choice changes equilibrium prices across all of Champaign, and buyers can substitute across the city line. So Urbana may be indirectly treated, violating the no-interference assumption (SUTVA) and contaminating the control group.

- **5 · Repeated cross-sections of sales**

  The houses sold before and after 1997 differ. A change in which houses sell can mimic a price effect unless hedonic controls absorb it. Repeat sales would help.

- **6 · Timing**

  The 1997 agreement and the 2002 consent decree are two possible event dates, with room for anticipation and phase-in.

- **7 · Zone measurement error**

  Hand-built zone maps can misclassify a house's zoned school. That attenuates exactly the heterogeneity (gains by zone quality) the mixture is meant to find.

- **8 · Spatial correlation**

  Prices are correlated within neighbourhoods and zones. The DPM treats units as conditionally independent, so add zone covariates or zone-level random effects.

**What would fit this design**

- **Mixture DiD:** $y_{it}=\alpha_{\text{city}}+\lambda_t+\mathbf x_{it}'\beta+\tau_i D_{it}+\varepsilon_{it}$, with a DPM over $(\tau_i, \varepsilon)$. Effects are heterogeneous and identified by parallel trends, not by exclusion. A possible future `model="did"`, reusing the Design A machinery.
- **Changes-in-changes** (Athey & Imbens 2006): recovers the whole counterfactual price distribution for the treated group from repeated cross-sections, giving quantile effects. The share who benefit still needs an assumption linking the two potential outcomes, such as rank invariance, or bounds. That is the same problem as $\sigma_{10}$ in a different form.

## §11 Independent review round

A fresh-context reviewer re-derived the math from the source, without seeing our conversation. It also ran NumPy Monte Carlo checks; Julia was not available. **Verdict:** the bug hunt is accurate, Design A is valid, but the plan misses the two biggest mixing levers and overreaches on the C–U application.

### Confirmed, with new evidence

- **Bug 2 is large.** Under the default prior the code inflates the new-cluster weight about **210–230×** ($\log\int f\,dG_0\approx-9.9$ against $\approx-4.5$ for the code's estimate). The code also discards a singleton's θ instead of keeping it as an auxiliary, as Algorithm 8 requires.
- **`NobileWishart` is exact.** Its moments match brute-force IW draws conditioned on $\sigma_{11}\approx1$, so the Imai–van Dyk critique doesn't apply to this step.
- Bugs 1, 3, 5–11, P1–P5 and the TT formula with $\sigma_{12}=\mathrm{Cov}(v,u_1)$, $\sigma_{13}=\mathrm{Cov}(v,u_0)$ are all confirmed. Bug 6 is wider than stated: `dpm`, `fmn` and `marginal` can also throw a KeyError.

### Corrections I accept (fixed above)

- Bug 4 is a mixing issue, not a correctness bug.
- Σ can't be integrated because of the independent β prior, not the $\sigma_{11}$ restriction.
- `dpm` is a broken Algorithm 8 with $m=1$, not “Algorithm 2 family”.
- Line references: `NobileWishart` is at `misc_functions.jl:14`, `copy(StateDP)` at `gibbsstate.jl:51`, and the PPD lines are `dpm_ppd.jl:167, 172`.
- “Effects 100× the noise” was overstated; the signal-to-noise ratio is about 8:1 in variance.
- The 10–100× speedup is unmeasured; the `kron` is sparse, not dense.

### The two biggest gaps

- **R1 · The prior $G_0$ decides the number of clusters**

  With $V_0=100I$ on standardized data, a new cluster's prior predictive is about $e^{-6}$ times an existing one's. So a *correct* sampler will almost never open clusters, and results hinge on $V_0$, $R$ and $\alpha$. The prior $E[\sigma_{22}]\approx 8$ when $\mathrm{Var}(y)=1$, and $\gamma\sim N(0,100)$ is far too diffuse for probit slopes. The README's “blocked is reliable” may partly reflect bug 1, whose sd-0.1 prior draws happen to sit near the data.

- **R2 · Labels are sticky because they condition on imputed $y_{\text{miss}}$**

  Draw $(c_i, d_i^*, y_{\text{miss},i})$ jointly. Weight each cluster by the β-collapsed predictive of the *observed* $(d_i,y_i)$: the bivariate marginal of $(d^*,y_{d_i})$, with $d^*$ integrated to a $\Phi$. Then draw the latent values from the chosen cluster. This is a closed-form, valid block step. My caveat: $\sigma_{23}$ still enters indirectly through $Q_{j,-i}$, which is built from the other members' imputed data, so this weakens the coupling rather than removing it entirely.

- **R3 · A collapsed $\sigma_{23}$ step can replace Design B**

  Integrate out $y_{\text{miss}}$, draw $\sigma_{23,j}$ from its 1-D conditional prior by slice sampling on the positive-definite interval, then redraw $y_{\text{miss}}$. This is a valid partially collapsed step. It gets most of Design B's mixing gain *without changing the model*.

### Other additions

- **Label-invariant outputs only.** Geweke tests, agreement tests, R-hat and ESS must use functionals such as ATE, ATT and the partition summaries, not per-cluster parameters.
- **Per-draw ATE/ATT with the uncertainty in $G$:** draw $(\pi_1,\dots,\pi_k,\pi_0)\sim\mathrm{Dir}(n_1,\dots,n_k,\alpha)$, and define PATT versus SATT.
- **Fréchet–Makarov bounds** on $\Pr(y_1>y_0)$ from the identified marginals, shown next to the $\sigma_{23}$ sensitivity table.
- **Synthetic data:** separate *diagnostic* scenarios (well-separated components) from *realistic* ones, where small effects mean types aren't separable. Score clusters with posterior similarity and ARI. Run 100+ coverage replications offline. Under `:skewed` the number of clusters $k$ is not meaningful (Miller & Harrison).
- **Infrastructure before $n=10{,}000$:** thread the RNG through the sampler, store a thinned state, and use vector labels.
- **Use cases:** drop the “lottery offer” instrument for house prices and the union coverage-rate instrument. Note that the DPM keeps $E[y\mid x]$ linear. Add van Hasselt (2011) as close prior work.
- **Champaign–Urbana:** two cities means one treated cluster, and few-cluster DiD inference is unreliable (Conley & Taber 2011). The credible estimand is a *triple difference*: the change in the price–zoned-school-quality gradient in Champaign after the reform, relative to Urbana. Check whether Urbana had any choice elements.

### Proposed plan revisions (your call)

1.  **New step 0: priors first.** Fix bug 1, rescale $V_0$ (especially γ) and $R$ to standardized data, put a proper prior on α, and run prior-predictive checks of $k$, propensities and outcome variances. Open question: keep per-cluster slopes, or move to **common slopes with a DPM over intercepts and Σ** (Conley et al. 2008 style)? That is a model change.

2.  **Design A, amended:** add the joint $(c_i,d^*_i,y_{\text{miss},i})$ label step (R2) and the collapsed $\sigma_{23}$ step (R3). Store $\sum\mathbf z\mathbf z'$.

3.  **Design B deferred**, possibly indefinitely, if R3 delivers the mixing.

4.  **Outputs rebuilt** as label-invariant per-draw functionals with Dirichlet weights. Fix P1, P2 and P5, and add Fréchet bounds.

5.  **Tests:** a Geweke test with an independent prior simulator, the ported NobileWishart check, collapsed-predictive and $Q/r$ unit tests, and offline coverage runs.

## §12 Decisions so far

- **Build Design A first**: the same model, with $G$ and β integrated out and Neal's Algorithm 8 for Σ.
- **Individual gains later**, by drawing $\sigma_{10}$ after sampling, with a prior-sensitivity table for the shares who benefit and are harmed.
- **Keep `blocked`** as an independent cross-check.
- **After the review round:**
  - Prior calibration becomes step 0.
  - `slopes = :common` (shared slopes, with the DPM over intercepts and Σ) is the default, and `slopes = :cluster` is an option.
  - Design A gains the joint label/latent step (R2) and the collapsed $\sigma_{23}$ step (R3).
  - Design B is deferred.
- **Synthetic data:** generic scenarios only (`:gaussian`, `:mixture`, `:skewed`, `:roy`, `:weak_iv`). `:mixture` borrows the “choice flattens the school premium” mechanism so there are winners and losers.
- **Champaign–Urbana:** documented in §10 as a design the current selection model doesn't fit cleanly.
- **Roadmap: `model="did"`** (future). This is a mixture difference-in-differences: $y_{it}=\alpha_{g}+\lambda_t+\mathbf x_{it}'\beta+\tau_i D_{it}+\varepsilon_{it}$ with a DPM over $(\tau_i,\varepsilon_{it})$, identified by parallel trends.
  - **Reuse:** the Design A machinery (labels, auxiliary components, conjugate sufficient statistics). The outcome equation is univariate here, so the Normal–Inverse-Gamma base measure is fully conjugate and Neal's Algorithm 3 applies, with $(\tau,\sigma^2)$ integrated out exactly.
  - **Outputs:** ATT, the distribution of $\tau_i$ among treated units, and the share with $\tau_i>0$.
  - **Correction to an earlier claim:** with repeated cross-sections each house is seen once, so a treated sale shows only $\tau_i+\varepsilon_i$. The distribution of $\tau$ is then a *deconvolution*. It needs (i) parallel trends in *distribution*, so the treated group's counterfactual $\varepsilon$ law comes from the other three city–period cells, and (ii) $\tau_i$ independent of $\varepsilon_i$. The share with $\tau_i>0$ rests on those assumptions. It is the DiD counterpart of the $\sigma_{10}$ problem, not free of it.
  - **Settled for the C–U study:**
    - **Repeated cross-sections** of sales. Hedonic controls absorb changes in which houses sell, and the deconvolution caveat above applies.
    - **Event date 2002** (consent decree). Controlled choice was already agreed in 1997, so 1997–2001 is partly treated. Treat it as a transition window to drop, or as its own period in an event study, not as pre-period.
    - **Pre-trends:** only visual inspection was done. The DiD model should include leads, i.e. placebo effects at pre-period dates. Pre-trend tests have low power and pretesting can distort inference (Roth 2022), so report posterior bands on the leads instead of a pass/fail test.
    - **Spatial correlation:** not handled previously. Add neighbourhood or zone random effects, or a spatial term; this matters most for uncertainty.
  - **Sequencing:** after Design A is correct and tested.
- **After Phase 2:**
  - Adopt prior **C3** as the default (details in §13).
  - Add a **hierarchical prior on cluster intercepts** to Phase 4, plus a separated-units diagnostic reported with every fit.
- Code work happens on your laptop. This session coordinates and reviews.

## §13 Progress and Phase 2 findings

| Phase                | Status                       | Notes                                                                                                                                                                                                                                                |
|----------------------|------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 0 · Baseline         | ✅ Done                      | The suite was red at `e817e7b` because Distributions dropped `TruncatedNormal(μ,σ,l,u)`. Compat raised to Julia 1.10. Baseline: about 6 ms per iteration and 11–14 GiB allocated for $M=1000$, $n=500$.                                   |
| 1 · Local fixes      | ✅ Done                      | Every item in Appendix A is fixed except bugs 2, 7 and 8, which Design A replaces. 382 tests pass.                                                                                                                                                   |
| Follow-ups           | ✅ Done                      | `dpm_ppd` builds its design by name. `acf_var` no longer counts lag 0 twice, and `acf_var2` now uses Geyer's initial positive sequence (summing every sample autocorrelation gives exactly −½). `testdpm_init.jl` runs in the suite. 821 tests pass. |
| Bug found in passing | ✅ Fixed                     | `blocked` crashed whenever α had a prior: a stick-breaking draw could round to $v_j=1$, making $\log(1-v_j)=-\infty$. The draw is now capped at `prevfloat(1.0)`.                                                                                      |
| 2 · Priors           | ✅ Analysis done; C3 adopted | Details are in `docs/priors.md` and summarized below.                                                                                                                                                                                                |
| 3 · Refactor         | Next                         | —                                                                                                                                                                                                                                                    |

### Prior candidates (standardized scale)

|                  | Intercepts                                                 | Probit slopes | Outcome slopes                                      | $\rho$ | $r$ | $\alpha$ |
|------------------|------------------------------------------------------------|---------------|-----------------------------------------------------|---------------|---------------|---------------|
| C0 (old)         | $N(0,10^2)$                                              | $N(0,10^2)$ | $N(0,10^2)$                                       | 6             | 2             | 1 fixed       |
| C1               | $N(0,1)$                                              | $N(0,1)$ | $N(0,1)$                                       | 6             | 0.130         | Gamma(2, 2)   |
| C2               | $N(0,1)$                                              | $N(0,1)$ | $N(0,0.5^2)$                                       | 10            | 0.153         | Gamma(2, 2)   |
| **C3 (adopted)** | $d^*$: $N(0,0.5^2)$; $y$: $N(0,1)$ | $N(0,0.5^2)$ | $N(0,0.5^2)$; $N(0,1)$ when slopes are shared | 10            | 0.153         | Gamma(2, 2)   |

Here $R=\mathrm{diag}(1/\rho, r, r)$, with $r$ set so the prior median within-cluster $\mathrm{Var}(y)$ is 0.3.

|                                                                  | C0               | C1        | C2        | C3            |
|------------------------------------------------------------------|------------------|-----------|-----------|---------------|
| Propensity mass outside (0.05, 0.95)                             | 92%              | 32%       | 32%       | **6%**        |
| Implied sd($y_{\text{obs}}$) (data: 1)                              | 12.1             | 1.38      | 1.10      | 1.11          |
| New-cluster penalty, nats (`:cluster` / `:common`)               | 9.8 / 8.8        | 3.8 / 3.2 | 3.6 / 3.2 | **3.4 / 3.2** |
| Blocked ATE, hedonic data (sample ATE 5.16)                      | 9.6 \[−0.8, 25\] | 5.29      | 5.23      | 5.24          |
| Units in all-treated or all-control clusters, two-component data | 100%             | 21%       | 1%        | 2%            |

### What Phase 2 changed in the plan

- **Treatment-separated clusters are a model-level risk.** A cluster can end up all treated or all controls, leaving its missing arm pinned only by the prior. C0 sends every unit there, and its ATE wanders over \[−1, 25\]. C3 makes this rare but cannot rule it out. *Decision:* Phase 4 gives cluster intercepts a hierarchical prior, so a separated cluster's missing arm borrows strength from the other clusters, and every fit reports the share of units in separated clusters.
- **A single blocked chain does not settle the number of clusters.** Across seeds, mean $k$ ranged from 1.2 to 8.2 on the same data while the ATE stayed at 5.2–5.6. This is the sticky-label problem that R2 and R3 target. Don't compare priors on $k$.
- **Slope shrinkage.** The true standardized outcome slope in the test data is about 0.8. A per-cluster N(0, 0.5²) prior shrinks it noticeably in small clusters. *Decision:* shared slopes get N(0, 1). Check this in Phase 6 with realistic $R^2$; the test DGP's $R^2\approx0.95$ is unrealistically high.

## §A Appendix A: full bug list with status

| \#      | Bug                                                                                                                                                                                        | Status                                                    |
|---------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------|
| 1       | Prior β draws use the precision as the covariance (`sample_prior_theta!`).                                                                                                                 | ✅ Fixed, Phase 1                                         |
| 2       | Biased new-cluster weight in `dpm`/`marginal` (about 220× under the old defaults), and a singleton's θ is discarded instead of kept as an auxiliary.                                       | Replaced by Design A (Phase 4)                            |
| 3       | Escobar–West α step chooses its mixture component deterministically.                                                                                                                       | ✅ Fixed, Phase 1                                         |
| 4       | `fmn` never redraws empty components. This is a mixing issue, not a correctness bug.                                                                                                       | Downgraded; no change                                     |
| 5       | Label probabilities computed as `exp(logpdf)`, with no log-sum-exp.                                                                                                                        | ✅ Fixed, Phase 1                                         |
| 6       | `countmap` init leaves `njs` missing keys: KeyError, plus uninitialized weights in `blocked`.                                                                                              | ✅ Fixed, Phase 1                                         |
| 7       | Marginal PPD filters with the last iteration's counts and loops over sparse labels.                                                                                                        | Replaced by Design A                                      |
| 8       | Marginal α update is for the infinite DP, but the labels are finite $Dir(\alpha/J)$.                                                                                                         | Replaced by Design A                                      |
| 9       | `test=true` paths crash (`idx + n`; `1:length(J)`).                                                                                                                                        | ✅ Fixed, Phase 1                                         |
| 10      | `priors.prior_Sigma` doesn't exist in the `Vinv=false` branch.                                                                                                                             | ✅ Fixed, Phase 1                                         |
| 11 / P5 | PPD covariates sliced by position (`znew[1:kx]`), and the README averaged only the first 100 rows.                                                                                         | ✅ Fixed: Phase 1 for `rand_ppd`, follow-up for `dpm_ppd` |
| 12      | The README's `quantile(tes.ate)` is a predictive interval for individual gains, not a credible interval for the ATE.                                                                       | Phase 5 (outputs) and Phase 7 (README)                    |
| 13      | Odds and ends: `-` on `BlockedWeights` adds; `export RawRata`; pre-1.0 helpers (`rescale_output`, `check_y`, `ppd_cdf`, `make_table`, `gc()`); undefined `V` in the PPD when `Vinv=false`. | ✅ Fixed or removed, Phase 1                              |
| P1      | Saved blocked weights alias the live objects.                                                                                                                                              | ✅ Fixed, Phase 1                                         |
| P2      | `parallel_ppd.jl` weights components by the density of the value it just drew.                                                                                                             | ✅ File retired, Phase 1                                  |
| P3      | TT evaluated at mean covariates is not the population ATT.                                                                                                                                 | Phase 5                                                   |
| P4      | One predictive draw per iteration instead of the closed-form per-draw ATE/ATT.                                                                                                             | Phase 5                                                   |
| —       | `acf_var` counts lag 0 twice.                                                                                                                                                              | ✅ Fixed, follow-up                                       |
| —       | `blocked` crashes when α has a prior: $v_j$ rounds to 1.                                                                                                                           | ✅ Fixed, Phase 2                                         |

------------------------------------------------------------------------

### References

- Neal (2000), Markov chain sampling methods for Dirichlet process mixture models, JCGS 9:249–265.
- Ishwaran & James (2001), Gibbs sampling methods for stick-breaking priors, JASA.
- Escobar & West (1995), Bayesian density estimation and inference using mixtures, JASA.
- van Dyk & Park (2008), Partially collapsed Gibbs samplers, JASA.
- Chib (2007), Analysis of treatment response data without the joint distribution of potential outcomes, J. Econometrics 140:401–412.
- Poirier & Tobias (2003), On the predictive distributions of outcome gains in the presence of an unidentified parameter, JBES 21:258–268.
- Li, Poirier & Tobias (2004), Do dropouts suffer from dropping out?, J. Applied Econometrics 19:203–225.
- Nobile (2000), J. Econometrics 99:335–345; Imai & van Dyk (2005), J. Econometrics.
- Geweke (2004), Getting it right: joint distribution tests of posterior simulators, JASA.
