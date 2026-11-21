# bench/geweke.jl -- long Geweke runs (Phase 4a exit check (a)); writes markdown tables to stdout.
#   julia --project=. bench/geweke.jl [M_sc] [M_mc]
include(joinpath(@__DIR__, "..", "test", "GewekeHarness.jl"))
using .GewekeHarness, Printf
M_sc = length(ARGS) > 0 ? parse(Int, ARGS[1]) : 500_000
M_mc = length(ARGS) > 1 ? parse(Int, ARGS[2]) : 500_000
println("Geweke runs: n = 20 (generate_hedonic_data(20, 1) covariates, standardized), C3 priors, ",
        "M_sc = $M_sc sweeps, M_mc = $M_mc forward draws, julia ", VERSION, "\n")
for (m, s) in (("marginal", :common), ("marginal", :cluster), ("gaussian", :cluster), ("blocked", :cluster))
    r = geweke_run(m, s; M_sc=M_sc, M_mc=M_mc)
    @printf("### %s, slopes = %s: max |z| = %.2f (%d statistics; %.0f s of sweeps)\n\n", m, s, r.maxz, length(r.rows), r.secs)
    geweke_table(stdout, r)
    println()
    flush(stdout)
end
