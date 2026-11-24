# Phase 4c: R3, the collapsed σ23 step: exit checks

Recorded 2026-11-24 on an Apple M2, Julia 1.12.0, 1 thread. The full outputs are in
`geweke_4c_long.md`, `phase4c_mixing_output.md`, `phase4c_mixing_seed2_output.md` and
`phase4c_timing_output.md`.

## The step

The R3 step (`update_sigma23!` in `src/sampler_functions.jl`, option
`InputParams(sigma23_step = true)`) runs as follows.

- **Where it runs.** In Design A it is step (iv'), after the Σ update and before the latent
  refresh. In `gaussian` it runs after `update_theta!`. It is skipped with `test = true`.
- **Target.** Each unit observes only one outcome, so once y_miss is integrated out the likelihood
  doesn't involve σ23. Its full conditional is therefore the prior conditional of
  NobileWishart(ρ, ρR):

  |Σ|^{-(ρ+4)/2} exp(−tr(ρR Σ⁻¹)/2)

  on the interval |σ23 − σ12σ13| < √((σ22 − σ12²)(σ33 − σ13²)). The exponent is the one in the
  Phase 1 test, which checks NobileWishart against inverse-Wishart draws conditioned on σ11 = 1.
- **Draw.** σ23 is drawn by slice sampling that shrinks from the full bounded interval, so the
  draw is exact. y_miss is then redrawn for every member, given (d*, y_obs) and the new Σ.

## (1)–(2) Unit tests (`test/test_sigma23.jl`)

- **Log density.** It equals the dense restricted-IW log density and is −Inf outside the
  positive-definite interval.
- **Invariance.**
  - I drew 200,000 Σ from NobileWishart(ρ, ρR) and applied one slice update to each. The
    result matches fresh draws in the means and centred squares of all five entries
    (|z| < 4.5). Only σ23 moves.
  - **Power:** five slice steps with ρ + 50 in the target fail the same check.
- **Conditional.** Two fixed sets of (σ12, σ13, σ22, σ33) were used. One slice step from exact
  starts, and a chain from a fixed start, both pass a KS test at α = 0.001 against the
  normalized density on a 200k-point grid.
- **Scope.** `update_sigma23!` changes σ23 and the members' y_miss only. The option runs in
  `marginal` under both slope modes and in `gaussian`.

## (3) Geweke

**Long run** (`bench/geweke.jl`, 500k sweeps against 500k forward draws, R2 and R3 on):

| Sampler | max \|z\| | Largest | s23 z |
|---|---|---|---|
| `marginal`, `:common` | 1.50 | a_d −1.50, mean(a1 − a0) −1.36, s23 +1.35 | 1.35 |
| `marginal`, `:cluster` | 1.95 | s33² +1.95 | 0.02 |
| `gaussian` | 2.13 | s33 +2.13 | 0.04 |
| `blocked` (no R3) | 1.79 | a_1² −1.79 | 0.82 |

**Planted bugs** (`bench/geweke_power.jl`, max |z|):

| Planted bug | `:common` | `:cluster` | `gaussian` | Expected |
|---|---|---|---|---|
| None (control) | 2.9 | 2.4 | 1.6 | pass |
| Earlier bugs: Σ dof | 7.8 | 10.9 | — | fail |
| Earlier bugs: no H Q⁻¹ H′ | 6.2 | 18.7 | — | fail |
| Earlier bugs: Φ dropped | 15.9 | 10.8 | — | fail |
| Earlier bugs: shift g_d dropped | **60.6** | 2.4 | — | fail `:common` only |
| Earlier bugs: Escobar–West indicator | 48.0 | 41.1 | — | fail |
| R3 target: posterior (ρ + n_j, ρR + Σee′), 50k / **500k** | 1.5 / 3.3 | 1.8 / 2.1 | 1.2 / 2.2 | **valid, see below** |
| R3 without the y_miss redraw, 50k / **500k** | 2.8 / 1.7 | 2.6 / 1.7 | 2.1 / 2.8 | **valid, see below** |
| R3 without the redraw, latent refresh drawing d* first, 50k / **500k** | 2.6 / 2.4 | 2.5 / 2.7 | 3.3 / 2.7 | invalid, not detected |
| Redraw kept, latent refresh drawing d* first (control), 50k | 2.4 | 1.8 | 1.9 | pass |
| R3 target with scale R instead of ρR, 50k / **500k** | 2.2 / **8.6** | 3.8 / 2.4 | 2.7 / **7.3** | fail |
| R3 target with ρ + 6 dof, 50k / **500k** | 2.2 / **6.2** | 3.1 / 2.5 | 2.2 / **4.7** | fail |

The "—" cells are runs where the bug lies in code `gaussian` doesn't use, so `gaussian` is
identical to the control there.

The two planted bugs requested as "must fail" are valid samplers, so Geweke cannot fail them:

- **The posterior target is a valid step.** NobileWishart(ρ + n_j, ρR + Σee′), with the
  current y_miss in e, is exactly the full conditional of Σ given the complete latent data, and
  its σ23 conditional is the full conditional of σ23. Slice-sampling from it, then redrawing
  y_miss, is an ordinary Gibbs step: valid, but not collapsed. The prior target is right only
  because y_miss is integrated out. Using the prior while still conditioning on y_miss would be
  the actual error; that is the next row.
- **Skipping the redraw is valid in this sweep order.** The step right after R3 is the latent
  refresh, which first draws y_miss from the same conditional as R3's redraw. Two draws in a row
  from one conditional are the same as one.
- **What actually goes wrong.** A later step must condition on the stale y_miss. Skipping the
  redraw while the refresh draws d* first does exactly that. It is invalid, but the error is
  below what Geweke can resolve at n = 20 and 500k sweeps: σ23 barely affects d*'s conditional,
  and the next sweep's full Σ update redraws σ23. The control (redraw kept, d* first) is valid
  and passes.

Geweke has little power on R3's target, for the same reason: the next sweep's full Σ update
redraws σ23 from the correct conditional, which dilutes R3's error. A wrong target is detected
only at 500k sweeps, through s23², and not under `:cluster`. The unit tests in (1) are the strong
check on the target.

## (4) Mixing: R3 off and on

`bench/phase4c_mixing.jl` ran on `generate_hedonic_data(500, 1234)` with C3 priors: 16 chains ×
3000 iterations (1000 burn-in), the R2 label step, and chain seeds 20261121 + c.

- σ23 is taken from unit 1's cluster (`gaussian`: its one Σ).
- "Implied" is the mean and sd of σ23 implied by its prior conditional given each draw's other
  entries, averaged over every 10th draw using the law of total variance. Because the likelihood
  doesn't involve σ23, these should equal its posterior mean and sd.

| Sampler | R3 | ATE mean (sd) | ATE ESS / 1000 | R-hat ATE / k / α | k | σ23 mean (sd) | σ23 ESS / 1000 | Implied σ23 mean (sd) | Stuck |
|---|---|---|---|---|---|---|---|---|---|
| `marginal`, `:common` | off | 5.269 (0.232) | 19.2 | 1.021 / 1.001 / 1.000 | 1.12 | −0.007 (0.011) | 22.8 | −0.007 (0.012) | 0 |
| `marginal`, `:common` | on | 5.252 (0.239) | 8.9 | 1.043 / 1.001 / 1.000 | 1.12 | −0.007 (0.010) | 25.3 | −0.007 (0.010) | 0 |
| `marginal`, `:cluster` | off | 5.268 (0.229) | 15.2 | 1.031 / 1.000 / 1.000 | 1.10 | −0.007 (0.010) | 22.1 | −0.007 (0.010) | 0 |
| `marginal`, `:cluster` | on | 5.282 (0.231) | 15.2 | 1.030 / 1.001 / 1.001 | 1.10 | −0.007 (0.010) | 30.6 | −0.007 (0.010) | 0 |
| `gaussian` | off | 5.256 (0.240) | 7.7 | 1.060 / — / — | 1 | −0.007 (0.010) | 16.5 | −0.007 (0.010) | 0 |
| `gaussian` | on | 5.254 (0.238) | 7.5 | 1.062 / — / — | 1 | −0.006 (0.010) | 9.4 | −0.006 (0.010) | 0 |

**Second seed set** (20271121 + c), `:common`:

| R3 | ATE ESS / 1000 | σ23 ESS / 1000 |
|---|---|---|
| off | 18.4 | 27.3 |
| on | 14.5 | 32.4 |

The R3-off rows reproduce Phase 4b exactly (19.2 and 15.2), from the same seeds.

- **R3 doesn't improve ATE mixing** in any sampler:
  - `:cluster`: 15.2 against 15.2;
  - `gaussian`: 7.7 against 7.5;
  - `:common`: 19.2 against 8.9, and 18.4 against 14.5 with the second seeds.

  The `:common` drop is within the spread between seed sets, which is up to about 40%.
- **σ23 isn't what limits mixing.**
  - Its posterior sd is 0.010, against a prior sd of about 0.3.
  - Given the other entries, its positive-definite interval is narrow, so σ23 is pinned near
    σ12σ13.
  - Its ESS is already higher than the ATE's with R3 off, and R3 changes it by no more than
    seed noise (`gaussian`: 16.5 → 9.4; Design A: 22–27 → 25–32).
- **σ23 matches its conditional prior.** In all six configurations, the posterior mean and sd of
  σ23 equal the values implied by its prior conditional.
- **No stuck chains** in any configuration. ATE posterior means are 5.25–5.28.

## (5) Timing

`bench/phase4c_timing.jl` used the Phase 4a protocol: n = 500, the median of 3 runs of M = 1000,
on an idle machine.

| Sampler | R3 off | R3 on |
|---|---|---|
| `marginal`, `:common` | 0.96 | 0.95 |
| `marginal`, `:cluster` | 1.14 | 1.15 |
| `gaussian` | 0.07 | 0.08 |
| `blocked` (J = 20) | 0.42 | — |

The figures are seconds per 1000 iterations. R3 costs under 1 ms per iteration for Design A: one
slice draw per cluster and one y_miss redraw per unit.

## (6) Suite

`julia --project=. test/runtests.jl` passes 2079/2079 tests.
