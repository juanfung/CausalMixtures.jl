# bench/coverage.jl -- Phase 6 coverage harness. Per replication: simulate_selection -> fit
# marginal :common (C3 or r = :auto priors) -> effects() -> record, for PATE, PATT and PATU, the
# 95% interval's coverage of the population and the sample truth, its width, the bias, R-hat,
# bulk ESS, the separated-units share, mean k and the runtime. Results are appended to a CSV as
# each replication finishes, so a run can be stopped and resumed (finished (cell, rep) pairs are
# skipped). Replications run in parallel over Julia threads.
#
#   julia --project=. -t 8 bench/coverage.jl key=value ...
# keys (comma-separated lists make a grid of cells):
#   scenario=gaussian,mixture  n=1000  r2=0.5,0.95  iv=0.3,1  prior=C3,auto  diagnostic=false
#   reps=10  chains=4  M=3000  out=bench/coverage.csv  seed=20261126
# Each replication's data and chains are seeded from hash((cell, rep, seed)), so results do not
# depend on the thread schedule.
using CausalMixtures, Random, Statistics, Printf
const CM = CausalMixtures

const ARGD = Dict(split(a, "=")[1] => split(a, "=")[2] for a in ARGS)
lst(k, d) = split(get(ARGD, k, d), ",")
scenarios = Symbol.(lst("scenario", "gaussian,mixture"))
ns = parse.(Int, lst("n", "1000"))
r2s = parse.(Float64, lst("r2", "0.5,0.95"))
ivs = parse.(Float64, lst("iv", "0.3,1"))
priors = String.(lst("prior", "C3,auto"))
diagnostic = parse(Bool, get(ARGD, "diagnostic", "false"))
reps = parse(Int, get(ARGD, "reps", "10"))
chains = parse(Int, get(ARGD, "chains", "4"))
M = parse(Int, get(ARGD, "M", "3000"))
outfile = get(ARGD, "out", "bench/coverage.csv")
seed0 = parse(Int, get(ARGD, "seed", "20261126"))

const HEADER = "scenario,n,r2,iv,prior,diagnostic,rep,seed,estimand,mean,lo,hi,width,pop_truth,sample_truth," *
               "cover_pop,cover_sample,bias_pop,bias_sample,rhat,ess_bulk,sep_share,k_mean,r1,r0,first_stage_F,secs"

cells = [(sc, n, r2, iv, pr) for sc in scenarios for n in ns for r2 in r2s for iv in ivs for pr in priors]
cellkey(c) = join(string.(c), "|") * "|$diagnostic"

## finished (cell, rep) pairs from an existing file
done = Set{Tuple{String,Int}}()
if isfile(outfile)
    for line in Iterators.drop(eachline(outfile), 1)
        f = split(line, ",")
        length(f) < 8 && continue
        push!(done, (join(f[1:6], "|"), parse(Int, f[7])))
    end
else
    open(io -> println(io, HEADER), outfile, "w")
end
tasks = [(c, r) for c in cells for r in 1:reps if !((cellkey(c), r) in done)]
println("coverage: $(length(cells)) cells x $reps reps; $(length(tasks)) to run on $(Threads.nthreads()) threads -> $outfile")
flush(stdout)

## fill the simulator's caches (population truth per DGP) before threading
for (sc, n, r2, iv, pr) in cells
    simulate_selection(10 * 7 * 20; scenario=sc, r2=r2, iv=iv, diagnostic=diagnostic, rng=Xoshiro(1))
end

const LOCK = ReentrantLock()
function run_rep(c, rep)
    sc, n, r2, iv, pr = c
    seed = Int(hash((cellkey(c), rep, seed0)) % typemax(Int32))
    t = @elapsed begin
        rng = Xoshiro(seed)
        s = simulate_selection(n; scenario=sc, r2=r2, iv=iv, diagnostic=diagnostic, rng=rng)
        raw = CM.RawData(s.formulas.formula_y, s.formulas.formula_d, s.df)
        priors_ = pr == "auto" ? setup_default_priors(raw; r=:auto) : setup_default_priors(raw)
        R = priors_.prior_theta.prior_Sigma.R
        effs, seps, ks = CM.Effects[], Float64[], Float64[]
        for ch in 1:chains
            p = CM.InputParams(M=M, thin=max(1, M ÷ 2000), scale_data=(true, true), verbose=false, model="marginal")
            crng = Xoshiro(seed + 1000ch)
            _, input, out = CM.dpm!(CM.dpm_init(raw, priors_, p; rng=crng)...; rng=crng)
            b = length(out.out_dp) ÷ 4
            push!(effs, CM.effects(out, input; rng=crng, burn=b))
            append!(seps, out.out_sep[(b + 1):end])
            append!(ks, [count(>(0), dp.njs) for dp in out.out_dp[(b + 1):end]])
        end
        e = CM.effects_summary(effs)
    end
    rows = String[]
    for (est, ptruth, struth) in ((:pate, s.truth.population.pate, s.truth.sample.ate),
                                  (:patt, s.truth.population.patt, s.truth.sample.att),
                                  (:patu, s.truth.population.patu, s.truth.sample.atu))
        v = getfield(e, est)
        lo, hi = v.q[1], v.q[3]
        push!(rows, join(Any[sc, n, r2, iv, pr, diagnostic, rep, seed, est,
                             @sprintf("%.6f", v.mean), @sprintf("%.6f", lo), @sprintf("%.6f", hi), @sprintf("%.6f", hi - lo),
                             @sprintf("%.6f", ptruth), @sprintf("%.6f", struth), Int(lo < ptruth < hi), Int(lo < struth < hi),
                             @sprintf("%.6f", v.mean - ptruth), @sprintf("%.6f", v.mean - struth),
                             @sprintf("%.4f", v.rhat), @sprintf("%.1f", v.ess_bulk), @sprintf("%.4f", mean(seps)),
                             @sprintf("%.3f", mean(ks)), @sprintf("%.4f", R[2, 2]), @sprintf("%.4f", R[3, 3]),
                             @sprintf("%.1f", s.info.first_stage_F), @sprintf("%.1f", t)], ","))
    end
    lock(LOCK) do
        open(outfile, "a") do io
            foreach(r -> println(io, r), rows)
        end
        @printf("done %s rep %d (%.0f s)\n", cellkey(c), rep, t)
        flush(stdout)
    end
end

Threads.@threads :dynamic for (c, rep) in tasks
    try
        run_rep(c, rep)
    catch err
        lock(LOCK) do
            println("FAILED $(cellkey(c)) rep $rep: ", sprint(showerror, err))
            flush(stdout)
        end
    end
end
println("coverage: finished")
