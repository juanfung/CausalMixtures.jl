#!/bin/sh
# bench/phase6b_run.sh -- Phase 6b, tier 1 coverage study: {gaussian, mixture} x r2 {0.5, 0.95} x iv 1
# x prior {C3 fixed, hier}, 100 reps per cell, diagnostic mode, 4 chains x 6000 iterations (2000 saved
# draws per chain, first 25% discarded), effects() with n_g0 = 5. The n = 2000 block runs first, then
# n = 500, each over K single-threaded processes (bench/coverage.jl shard=i/K).
#
# Resumable: rerun the same command. Every finished (cell, rep) is already in
# bench/coverage_tier1_n<N>*.csv and is skipped; keep K unchanged so the shards split the rest as before
# (a different K still skips finished pairs, it only rebalances).
#
#   nohup caffeinate -ims sh bench/phase6b_run.sh > bench/logs/tier1_launcher.log 2>&1 &
#
# Writes bench/coverage_tier1_n<N>_summary.md and touches bench/logs/tier1_n<N>.done after each block.
cd "$(dirname "$0")/.." || exit 1
mkdir -p bench/logs
K=${K:-8}
COMMON="scenario=gaussian,mixture r2=0.5,0.95 iv=1 prior=C3,hier diagnostic=true reps=100 chains=4 M=6000 draws=2000 n_g0=5"
for N in ${NS:-2000 500}; do
    OUT=bench/coverage_tier1_n$N.csv
    echo "$(date '+%F %T') start n = $N, K = $K"
    i=1
    while [ "$i" -le "$K" ]; do
        OPENBLAS_NUM_THREADS=1 julia --project=. -t 1 bench/coverage.jl $COMMON n=$N out=$OUT shard=$i/$K \
            >> bench/logs/tier1_n${N}_shard$i.log 2>&1 &
        i=$((i + 1))
    done
    wait
    julia --project=. bench/coverage_summary.jl $OUT > bench/coverage_tier1_n${N}_summary.md
    touch bench/logs/tier1_n$N.done
    echo "$(date '+%F %T') done n = $N"
done
