# test/test_geweke.jl -- Geweke "getting it right" tests (test/GewekeHarness.jl).
# The default suite runs a short smoke version. Set CM_GEWEKE=1 for the long runs:
#   CM_GEWEKE=1 julia --project=. test/runtests.jl
using Test, CausalMixtures, Random
isdefined(Main, :GewekeHarness) || include("GewekeHarness.jl")
using .GewekeHarness

const GEWEKE_LONG = get(ENV, "CM_GEWEKE", "0") == "1"

@testset "Geweke: $(GEWEKE_LONG ? "long" : "smoke")" begin
    ## smoke: first moments only; squares of the heavy-tailed Sigma entries need long runs
    M_sc, M_mc, zmax, powers = GEWEKE_LONG ? (500_000, 500_000, 4.5, (1, 2)) : (10_000, 20_000, 5.0, (1,))
    configs = (("marginal", :common), ("marginal", :cluster), ("gaussian", :cluster), ("blocked", :cluster))
    for (model, slopes) in configs
        r = geweke_run(model, slopes; M_sc=M_sc, M_mc=M_mc, seed=GEWEKE_LONG ? 1 : 2, powers=powers)
        @test all(x -> isfinite(x.z), r.rows)
        @test r.maxz < zmax || (model, slopes, r.maxz, sort(r.rows, by = x -> -abs(x.z))[1])
        GEWEKE_LONG && println("Geweke $model $slopes: max |z| = $(round(r.maxz, digits=2))")
    end
end
