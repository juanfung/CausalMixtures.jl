# bench/prior_calibration_extra.jl -- optional extra candidates for bench/prior_calibration.jl
# extra_candidates(r6, r10, names) returns the PriorSpecs named in `names`.
#
# C3: C2 with a tighter selection equation. Under C1 and C2, N(0, 1) probit intercepts and slopes
# put about a third of the prior propensity mass outside (0.05, 0.95), and the posterior check
# shows clusters that are almost all treated or all controls, whose missing arm is identified
# only by the prior. N(0, 0.5^2) on the probit intercept and slopes gives Var(w'gamma) of about
# 0.75 on standardized covariates, so roughly 5% of the propensity mass lies outside (0.05, 0.95).
function extra_candidates(r6, r10, names)
    out = PriorSpec[]
    if "C3" in names
        push!(out, PriorSpec(name="C3", sd_intercept=1, sd_intercept_d=0.5, sd_probit=0.5,
                             sd_outcome=0.5, rho=10, r=r10, alpha_shape=2.0, alpha_rate=2.0))
    end
    return out
end
