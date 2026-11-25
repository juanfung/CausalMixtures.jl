## bench/phase5b_mle.jl -- Phase 5b: maximum-likelihood estimate of the k = 1 observed-data likelihood
## gradients from several starts; compared with the posterior and the Heckman two-step
using CausalMixtures, Random, Statistics, LinearAlgebra, Printf
const CM = CausalMixtures
data = generate_hedonic_data(500, 1234)
raw = CM.RawData(data.formulas.formula_y, data.formulas.formula_d, data.df)
_, input, _ = CM.dpm_init(raw, setup_default_priors(2, 3; J=1), CM.InputParams(M=1, scale_data=(true, true), verbose=false, model="gaussian"); rng=Xoshiro(1))
X, W, y, d = input.data.X, input.data.W, input.data.y.a, input.data.d
function ll(t)
    b, dl, O, _ = CM.obsdata_unpack(t, 7)
    return CM.obs_loglik(b, CM.sigma_from_delta_omega(dl, O), X, W, y, d)
end
function ascend(t)
    f = ll(t)
    for it in 1:3000
        g = [(ll(t + 1e-6 * I[1:12, k]) - ll(t - 1e-6 * I[1:12, k])) / 2e-6 for k in 1:12]
        H = zeros(12, 12)   # quasi-Newton step via numerical Hessian every iteration (small problem)
        for k in 1:12
            H[:, k] = [(ll(t + 1e-4 * I[1:12, k] + 1e-4 * I[1:12, l]) - ll(t + 1e-4 * I[1:12, k] - 1e-4 * I[1:12, l]) -
                        ll(t - 1e-4 * I[1:12, k] + 1e-4 * I[1:12, l]) + ll(t - 1e-4 * I[1:12, k] - 1e-4 * I[1:12, l])) / 4e-8 for l in 1:12]
        end
        H = (H + H') / 2
        step = -(H - 1e-6I) \ g
        all(isfinite, step) && (minimum(eigvals(Symmetric(H))) < 0) || (step = 0.01 .* g)
        a = 1.0
        while a > 1e-8 && !(ll(t + a * step) > f)
            a /= 2
        end
        a <= 1e-8 && break
        t = t + a * step; fn = ll(t)
        abs(fn - f) < 1e-10 && (f = fn; break)
        f = fn
    end
    return t, f
end
for (nm, S0) in (("zero corr", [1.0 0 0; 0 0.06 0; 0 0 0.07]), ("two-step-like", [1.0 0.18 -0.24; 0.18 0.09 -0.04; -0.24 -0.04 0.12]))
    t0 = CM.obsdata_pack([0.0, 0.25, 0.27, 0.58, 0.8, -0.86, 0.17], S0)
    t, f = ascend(t0)
    b, dl, O, _ = CM.obsdata_unpack(t, 7); S = CM.sigma_from_delta_omega(dl, O)
    @printf("MLE from %s: loglik %.4f, sigma12 %.3f, sigma13 %.3f, sigma22 %.3f, sigma33 %.3f; corr %.2f, %.2f\n", nm, f, S[1, 2], S[1, 3], S[2, 2], S[3, 3], S[1, 2] / sqrt(S[2, 2]), S[1, 3] / sqrt(S[3, 3]))
end
## profile: the log-likelihood maximized over everything else at fixed (sigma12, sigma13) is
## not needed; report the loglik at the posterior mean for scale
