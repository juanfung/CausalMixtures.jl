# bench/coverage_summary.jl -- summarize a coverage run (bench/coverage.jl output: the base CSV and
# its shards). Per cell and estimand: replications, coverage of the population and the sample truth
# (with its binomial se), mean interval width, bias and RMSE against the population truth, median
# R-hat and the share with R-hat > 1.1, mean separated share and k, mean runtime per replication.
#   julia --project=. bench/coverage_summary.jl bench/coverage_pilot.csv
using Statistics, Printf

base = replace(ARGS[1], r"\.csv$" => "")
dir, bn = dirname(base), basename(base)
rows = Vector{Vector{String}}()
header = String[]
for f in readdir(isempty(dir) ? "." : dir; join=true)
    (startswith(basename(f), bn) && endswith(f, ".csv")) || continue
    lines = readlines(f)
    global header = String.(split(lines[1], ","))
    append!(rows, [String.(split(l, ",")) for l in lines[2:end] if !isempty(l)])
end
col(nm) = findfirst(==(nm), header)
cells = unique([r[1:6] for r in rows])
sort!(cells)
println("| Scenario | n | r2 | iv | prior | estimand | reps | cover pop | cover sample | width | bias (pop) | RMSE | med R-hat | R-hat > 1.1 | sep | k | s / rep |")
println("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for c in cells, est in ("pate", "patt", "patu")
    sub = [r for r in rows if r[1:6] == c && r[col("estimand")] == est]
    isempty(sub) && continue
    f(nm) = parse.(Float64, [r[col(nm)] for r in sub])
    R = length(sub)
    cp, cs = mean(f("cover_pop")), mean(f("cover_sample"))
    se = sqrt(max(cp * (1 - cp), 0.0225 * 0.975) / R)
    rh = f("rhat")
    @printf("| %s | %s | %s | %s | %s | %s | %d | %.2f ± %.2f | %.2f | %.4f | %+.4f | %.4f | %.3f | %.2f | %.3f | %.2f | %.0f |\n",
            c[1], c[2], c[3], c[4], c[5], est, R, cp, se, cs, mean(f("width")), mean(f("bias_pop")),
            sqrt(mean(f("bias_pop") .^ 2)), median(rh), mean(rh .> 1.1), mean(f("sep_share")), mean(f("k_mean")), mean(f("secs")))
end
