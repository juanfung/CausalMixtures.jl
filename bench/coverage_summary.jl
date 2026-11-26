# bench/coverage_summary.jl -- summarize a coverage run (bench/coverage.jl output: the base CSV and
# its shards). Per cell and estimand: replications, coverage of the population and the sample truth
# (with its binomial se), mean interval width, bias and RMSE against the population truth, median
# R-hat and the share with R-hat > 1.1, mean separated share and k, mean runtime per replication.
# "flag" marks population coverage below 0.90 (*). A second table compares prior arms within each
# (scenario, n, r2, iv, estimand) when the run has both C3 and hier: the coverage difference and its
# MC se, flagged (**) when they differ by more than 2 se, and the RMSE ratio.
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
hasr = col("k5_mean") !== nothing
println("| Scenario | n | r2 | iv | prior | estimand | reps | cover pop | cover sample | width | bias (pop) | RMSE | med R-hat | R-hat > 1.1 | sep | k | ", hasr ? "k (>= 5%) | r post (data) | " : "", "s / rep | flag |")
println("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|", hasr ? "---|---|" : "", "---|---|")
summ = Dict{Tuple,Tuple{Float64,Int,Float64}}()
for c in cells, est in ("pate", "patt", "patu")
    sub = [r for r in rows if r[1:6] == c && r[col("estimand")] == est]
    isempty(sub) && continue
    f(nm) = parse.(Float64, [r[col(nm)] for r in sub])
    R = length(sub)
    cp, cs = mean(f("cover_pop")), mean(f("cover_sample"))
    se = sqrt(max(cp * (1 - cp), 0.0225 * 0.975) / R)
    rh = f("rhat")
    extra = ""
    if hasr
        rpo, rda = f("r1_post"), f("r1_data")
        rpo0, rda0 = f("r0_post"), f("r0_data")
        extra = @sprintf("%.2f | %s | ", mean(f("k5_mean")),
                         all(isnan, rpo) ? @sprintf("— (%.3f, %.3f)", mean(rda), mean(rda0)) :
                                           @sprintf("%.3f, %.3f (%.3f, %.3f)", mean(rpo), mean(rpo0), mean(rda), mean(rda0)))
    end
    rmse = sqrt(mean(f("bias_pop") .^ 2))
    summ[(c[1:4]..., c[6], est, c[5])] = (cp, R, rmse)
    @printf("| %s | %s | %s | %s | %s | %s | %d | %.3f ± %.3f | %.3f | %.4f | %+.4f | %.4f | %.3f | %.2f | %.3f | %.2f | %s%.0f | %s |\n",
            c[1], c[2], c[3], c[4], c[5], est, R, cp, se, cs, mean(f("width")), mean(f("bias_pop")),
            rmse, median(rh), mean(rh .> 1.1), mean(f("sep_share")), mean(f("k_mean")), extra, mean(f("secs")),
            cp < 0.90 ? "*" : "")
end

pairs = sort(unique([k[1:6] for k in keys(summ) if haskey(summ, (k[1:6]..., "C3")) && haskey(summ, (k[1:6]..., "hier"))]))
if !isempty(pairs)
    println("\n| Scenario | n | r2 | iv | estimand | cover C3 | cover hier | hier − C3 ± se | RMSE hier / C3 | flag |")
    println("|---|---|---|---|---|---|---|---|---|---|")
    for k in pairs
        (c1, n1, m1), (c2, n2, m2) = summ[(k..., "C3")], summ[(k..., "hier")]
        sed = sqrt(max(c1 * (1 - c1), 0.0225 * 0.975) / n1 + max(c2 * (1 - c2), 0.0225 * 0.975) / n2)
        @printf("| %s | %s | %s | %s | %s | %.3f | %.3f | %+.3f ± %.3f | %.2f | %s |\n", k[1], k[2], k[3], k[4], k[6],
                c1, c2, c2 - c1, sed, m2 / m1, abs(c2 - c1) > 2sed ? "**" : "")
    end
end
