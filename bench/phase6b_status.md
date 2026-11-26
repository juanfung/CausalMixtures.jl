# Phase 6b status: tier 1 coverage run

**Progress:**
- **n = 2000:** done 2026-11-26 18:39 (8.6 h; 800/800 reps, no failures). Results are in
  `bench/coverage_tier1_n2000_summary.md`, and the coordinator was messaged.
- **n = 500:** started 18:39, estimated at about 2 h.

A handoff note, so a fresh session can pick the run up.

**What runs:** `bench/phase6b_run.sh`, launched 2026-11-26 under `caffeinate -ims` (the Mac must stay
on AC with the lid open). It runs 8 single-threaded `bench/coverage.jl` processes on the n = 2000
block, then the n = 500 block.
- Cells: {gaussian, mixture} × r2 {0.5, 0.95} × iv 1 × prior {C3, hier}, diagnostic mode.
- 100 reps per cell, 4 chains × 6000 iterations, n_g0 = 5.
- Estimated wall clock: about 5.5–6 h for n = 2000, then about 1.5–2 h for n = 500.

**Where results are:**
- Per-replication rows: `bench/coverage_tier1_n2000.shard<i>.csv` and `bench/coverage_tier1_n500.shard<i>.csv`.
- Logs: `bench/logs/tier1_n<N>_shard<i>.log` and `bench/logs/tier1_launcher.log` (not committed).
- Each finished block writes `bench/coverage_tier1_n<N>_summary.md` and touches `bench/logs/tier1_n<N>.done`.

**Check progress:** `grep -c '^done' bench/logs/tier1_n2000_shard*.log` (100 per shard when the block
is done), or `pgrep -fl coverage.jl`.

**Resume after sleep, reboot or a killed run:** check that nothing is still running
(`pgrep -fl coverage.jl`), then rerun
`nohup caffeinate -ims sh bench/phase6b_run.sh > bench/logs/tier1_launcher.log 2>&1 &`.
Finished (cell, rep) pairs are skipped. A partly written last line in a shard CSV (from a hard kill)
should be deleted first. To run only one block, set `NS=500`.

**After each block:**
1. Report the per-cell summary.
2. Message "Repository readme review".
3. After the n = 500 block, write `bench/phase6b.md` with the final summary. The coordinator asked
   for, per cell and estimand:
   - coverage with its MC standard error, sample-truth coverage, width, bias and RMSE;
   - median R-hat and the share with R-hat > 1.1;
   - k(≥5%), the separated share, the posterior r against the data value, and the time per replication.

   Flag every cell where coverage is below 0.90, or where C3 and hier differ by more than 2 MC
   standard errors. End with the recommendation on making `:hier` the default, looking hardest at
   the mixture r2 = 0.95 cells. The default stays C3 until then; don't change it.
4. Then STOP. Weak-IV cells are held for option D (Phase 4g).
