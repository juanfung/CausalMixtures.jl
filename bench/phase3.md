# Phase 3: state refactor, exit checks

Recorded 2026-11-21 on an Apple M2, Julia 1.12.0, 1 thread. "Pre" is commit `4b4a2f4`, which has the C3
defaults and the code as before the refactor. "Post" is `0d211ca`, after the six refactor commits.
The refactor changes no math.

## (a) Same posteriors before and after, under C3

`bench/phase3_equivalence.jl` ran `gaussian`, `blocked` (J = 20) and `dpm` on
`generate_hedonic_data(500, 1234)` with `scale_data = (true, true)` and the C3 defaults. Each sampler ran
8 chains with seeds 20261121 + c, each M = 3000 with a burn-in of 1000. The per-chain summaries are in
`phase3_equivalence_pre.csv` and `phase3_equivalence_post.csv`.

Per-draw statistics are label-invariant:
- `ate` = sy Σ_j (n_j/n) x̄'(β1_j − β0_j), on the data scale;
- `k` is the number of occupied clusters;
- `s..` are the n_j-weighted Σ entries;
- `ppd` is the `dpm_ate` predictive draw at the covariate means.

Each chain contributes its post-burn-in mean, and its sd for `ate` and `ppd`. The standard error is the
between-chain spread of the chain means over √8. That spread covers the slow, seed-dependent mixing of k.
z = (pre − post) / √(se_pre² + se_post²).

| Model | Statistic | pre: mean ± se | post: mean ± se | z |
|---|---|---|---|---|
| gaussian | ate_mean | 5.2496 ± 0.0188 | 5.2625 ± 0.0155 | -0.53 |
| gaussian | ate_sd | 0.2052 ± 0.0121 | 0.2163 ± 0.0113 | -0.67 |
| gaussian | s22 | 0.0539 ± 0.0004 | 0.0533 ± 0.0003 | +1.11 |
| gaussian | s33 | 0.0662 ± 0.0009 | 0.0663 ± 0.0003 | -0.03 |
| gaussian | s12 | 0.0651 ± 0.0041 | 0.0617 ± 0.0042 | +0.59 |
| gaussian | s13 | -0.0997 ± 0.0070 | -0.1009 ± 0.0028 | +0.16 |
| gaussian | s23 | -0.0070 ± 0.0007 | -0.0061 ± 0.0005 | -1.03 |
| gaussian | ppd_mean | 5.2601 ± 0.0221 | 5.2713 ± 0.0166 | -0.41 |
| gaussian | ppd_sd | 1.3542 ± 0.0113 | 1.3405 ± 0.0068 | +1.04 |
| blocked | ate_mean | 5.1129 ± 0.0948 | 5.1900 ± 0.0531 | -0.71 |
| blocked | ate_sd | 0.3814 ± 0.1378 | 0.2896 ± 0.0604 | +0.61 |
| blocked | k | 3.2563 ± 0.4263 | 2.6497 ± 0.4315 | +1.00 |
| blocked | alpha | 1.0034 ± 0.1485 | 0.7870 ± 0.1564 | +1.00 |
| blocked | s22 | 0.0631 ± 0.0052 | 0.0597 ± 0.0043 | +0.49 |
| blocked | s33 | 0.0689 ± 0.0007 | 0.0685 ± 0.0006 | +0.52 |
| blocked | s12 | 0.0673 ± 0.0042 | 0.0697 ± 0.0048 | -0.38 |
| blocked | s13 | -0.0954 ± 0.0072 | -0.1006 ± 0.0038 | +0.64 |
| blocked | s23 | -0.0065 ± 0.0007 | -0.0062 ± 0.0006 | -0.35 |
| blocked | ppd_mean | 5.0617 ± 0.1007 | 5.1620 ± 0.0647 | -0.84 |
| blocked | ppd_sd | 1.8332 ± 0.1873 | 1.6501 ± 0.1228 | +0.82 |
| dpm | ate_mean | 5.1817 ± 0.0535 | 5.0972 ± 0.1727 | +0.47 |
| dpm | ate_sd | 0.3051 ± 0.0689 | 0.2616 ± 0.0406 | +0.54 |
| dpm | k | 2.9337 ± 0.0854 | 3.1169 ± 0.3637 | -0.49 |
| dpm | alpha | 0.4916 ± 0.0122 | 0.5149 ± 0.0481 | -0.47 |
| dpm | s22 | 0.0556 ± 0.0004 | 0.0664 ± 0.0112 | -0.97 |
| dpm | s33 | 0.0715 ± 0.0031 | 0.0684 ± 0.0008 | +0.98 |
| dpm | s12 | 0.0642 ± 0.0028 | 0.0642 ± 0.0055 | +0.00 |
| dpm | s13 | -0.0937 ± 0.0069 | -0.0993 ± 0.0078 | +0.54 |
| dpm | s23 | -0.0062 ± 0.0004 | -0.0065 ± 0.0006 | +0.31 |
| dpm | ppd_mean | 5.1877 ± 0.0517 | 5.0810 ± 0.1822 | +0.56 |
| dpm | ppd_sd | 1.4900 ± 0.0863 | 1.6979 ± 0.2410 | -0.81 |

All 31 |z| values are below 1.2. The two versions are not bitwise comparable: drawing per unit and
threading an rng changes the random stream, not the kernel.

**Stuck chains, in both versions.** One chain out of 8 sometimes sits in a mode with a lower ATE and a
wider posterior:
- pre `blocked` chain 1: ATE 4.47, sd 1.32;
- pre `dpm` chain 2: ATE 4.83, sd 0.78;
- post `blocked` chain 8: ATE 4.83, sd 0.71;
- post `dpm` chain 6: ATE 3.90, sd 0.54, k 5.6.

These excursions dominate the between-chain standard errors of `blocked` and `dpm`. They are the
multimodality that Phase 2 saw in k, and the reason for Design A. The refactor did not introduce them.

## (b) Timing and allocation

`bench/phase3_timing.jl` uses the `bench/baseline.md` configuration: n = 500, M = 1000,
`generate_hedonic_data(500, 1234)`, `scale_data = (true, true)`, J = 20 (1 for `gaussian`), and
`Random.seed!(20261121)` before init. Each entry is the median of 3 runs after a warm-up. C0 is the old
prior with α = 1 fixed, which is the baseline configuration; C3 is the new default.

| Prior | Model | Pre: time | Pre: alloc | Post: time | Post: alloc | Speed-up | Alloc ratio |
|---|---|---|---|---|---|---|---|
| C0 | `dpm` | 7.08 s | 15.18 GiB | 2.36 s | 5.89 GiB | 3.0× | 2.6× less |
| C0 | `blocked` | 5.83 s | 11.03 GiB | 0.44 s | 0.55 GiB | 13.3× | 20× less |
| C0 | `gaussian` | 0.27 s | 0.89 GiB | 0.07 s | 0.08 GiB | 3.9× | 11× less |
| C3 | `dpm` | 6.12 s | 13.55 GiB | 2.21 s | 5.72 GiB | 2.8× | 2.4× less |
| C3 | `blocked` | 6.00 s | 11.21 GiB | 0.42 s | 0.59 GiB | 14.3× | 19× less |
| C3 | `gaussian` | 0.27 s | 0.89 GiB | 0.07 s | 0.08 GiB | 3.9× | 11× less |

Against the Phase 0 baseline at `e817e7b`: `dpm` went from 6.63 s / 14.33 GiB to 2.36 s / 5.89 GiB,
and `blocked` from 5.78 s / 10.88 GiB to 0.44 s / 0.55 GiB (C0).

**Where `dpm`'s time still goes.** Its new-cluster weight draws a fresh θ from p(θ | z_i) for every unit
in every sweep (`sample_new_theta`: a `NobileWishart` and a ktot × ktot inverse, about 12 KB each). That
is about 6 MB per sweep. This step is the biased new-cluster rule of review bug 2. Design A replaces it,
so I did not optimise it here.

GC share: 12–13% for `dpm`, about 5% for `blocked`.

## (c) Test suite

`julia --project=. test/runtests.jl`: 1508 / 1508 pass.
