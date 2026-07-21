# C1 smoke — command (recorded BEFORE launch)

Frozen threshold: **MIN_FREE_MB = 26,355 MiB** (= max fit peaks 22,145/22,259 + 4,096;
peak JSON `exp09_2026-07-21_11-42-41_c1_fit_peak.json` in the exp_06 folder; 112
samples/GPU; note the peaks are TOTAL-used incl. B-F's resident ~15.9 GiB, so the gate
double-counts the co-tenant — conservative direction). Frozen in
`c1_frozen_min_free.txt` (this commit). Pins: EXPECT_PACKAGE_SHA `3e416db…` (unchanged,
cyl repo frozen); EXPECT_EXP09_SHA = THIS records commit (self-ref convention,
post-hoc verified). Co-tenant with B-F (unchanged disclosure).

```bash
cd /home/yixunhu/codespace/exp-09-cyl-dinov3-no-ssl
export PATH=/home/yixunhu/miniconda3/envs/flac/bin:$PATH   # flac env (attempt-1 lesson)
export EXPECT_PACKAGE_SHA=3e416db1b6933dd842a3667432ff21436e7089ca
export EXPECT_EXP09_SHA=$(git rev-parse HEAD)
bash worklog/worklog_yixun/exp_09_cyl_no_ssl/c1_smoke.sh 26355 \
  /home/yixunhu/codespace/cylindrical-dinov3/worklog/worklog_yixun/exp_06_flac_no_ssl_claude
```
Acceptance (all verifier-gated, exit nonzero on any miss): pin gate ALL PASS; both
GPUs free ≥ 26,355; 100 steps with ckpt ≤ step 100; finite loss; strict reload;
sustained ≥ 0.0395 steps/s (ceil'd monotonic wall / observed==declared steps);
backbone forwards = 9/batch (1 source + K=8 context), zero extra frame passes.
