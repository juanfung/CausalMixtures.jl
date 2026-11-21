# test/TestHelpers.jl -- shared helpers for statistical tests.
#
# Include once (runtests.jl does this); test files then `using .TestHelpers`.
#
# Moment checks compare sample moments with targets using z-scores built from
# the Monte Carlo standard error of each moment, so tolerances scale with the
# number of draws instead of being hand-tuned constants. Draws are rows.

module TestHelpers

using Random, Statistics, LinearAlgebra

export TEST_SEED, test_rng, seed_global!,
       mean_zscores, cov_zscores, means_match, covs_match, moments_match

"Default seed for statistical tests."
const TEST_SEED = 20261121

"A fresh, explicitly seeded RNG for tests."
test_rng(seed::Integer=TEST_SEED) = Xoshiro(seed)

"""
    seed_global!(seed=TEST_SEED)

Seed the task-local global RNG, for code that does not take an `rng` argument yet
(the samplers before Phase 3).
"""
seed_global!(seed::Integer=TEST_SEED) = Random.seed!(seed)

_rows(x::AbstractVector) = reshape(x, :, 1)
_rows(x::AbstractMatrix) = x

"""
    mean_zscores(draws, target; ess=nrows) -> Vector

z-scores of the sample mean of each column of `draws` (N × p, one draw per row)
against `target`, using the standard error sd/√ess. Pass `ess` for correlated
(MCMC) draws.
"""
function mean_zscores(draws::AbstractVecOrMat, target; ess::Real=size(_rows(draws), 1))
    X = _rows(draws)
    m = vec(mean(X, dims=1))
    s = vec(std(X, dims=1))
    return (m .- vec(collect(target))) ./ (s ./ sqrt(ess))
end

"""
    cov_zscores(draws, target; mean=nothing, ess=nrows) -> Matrix

Entrywise z-scores of the sample covariance of `draws` (N × p) against the
p × p `target`. The standard error of entry (a, b) is the sd of the products
(x_a - m_a)(x_b - m_b) divided by √ess. If `mean` is given (the known
population mean), the products are centred at it instead of the sample mean.
"""
function cov_zscores(draws::AbstractVecOrMat, target::AbstractMatrix;
                     mean=nothing, ess::Real=size(_rows(draws), 1))
    X = _rows(draws)
    N, p = size(X)
    mu = mean === nothing ? vec(Statistics.mean(X, dims=1)) : vec(collect(mean))
    Xc = X .- mu'
    z = zeros(p, p)
    for a in 1:p, b in a:p
        prod_ab = Xc[:, a] .* Xc[:, b]
        se = std(prod_ab) / sqrt(ess)
        z[a, b] = z[b, a] = (Statistics.mean(prod_ab) - target[a, b]) / se
    end
    return z
end

"""
    means_match(draws, target; z=4.5, ess=nrows) -> Bool

True if every column mean is within `z` Monte Carlo standard errors of its
target. The default z = 4.5 keeps the family-wise false-alarm rate small for a
few dozen simultaneous checks.
"""
means_match(draws, target; z::Real=4.5, kw...) = maximum(abs, mean_zscores(draws, target; kw...)) < z

"""
    covs_match(draws, target; z=4.5, mean=nothing, ess=nrows) -> Bool

True if every covariance entry is within `z` Monte Carlo standard errors of its target.
"""
covs_match(draws, target; z::Real=4.5, kw...) = maximum(abs, cov_zscores(draws, target; kw...)) < z

"""
    moments_match(draws, mu, Sigma; z=4.5, ess=nrows) -> Bool

Both `means_match` and `covs_match`.
"""
moments_match(draws, mu, Sigma; z::Real=4.5, ess::Real=size(_rows(draws), 1)) =
    means_match(draws, mu; z=z, ess=ess) && covs_match(draws, Sigma; z=z, ess=ess)

end # module
