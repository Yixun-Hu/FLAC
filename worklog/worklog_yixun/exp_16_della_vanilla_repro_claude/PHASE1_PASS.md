# PHASE1_PASS — exp_16 Phase-1 calibration gate verdict

**Verdict: PASS (8/8 pre-registered criteria)** · Evaluated 2026-08-12 23:05 EDT by the Planner session against plan §5 + Rev 3 §2.

- **Gating cell:** job 12287666 (`exp16-eval-unseen_s42`), COMPLETED 00:07:52 rc=0, NVIDIA A100 80GB PCIe, EXPECT_SHA `e2d77a2`, log `della_vanilla_repro_2026-08-12_22:29:10_eval_jid12287666.log`.
- **Evidence:** loader "Found 6337 files in 17 subfolders" (announcement 01 full split); `--expected-stream-count 6337` stream check passed (sidecar written); clean checkpoint load (no `--allow-partial-load`; rc=0); `EVALRESULT cell=unseen_s42 rc=0`.
- **Metrics record:** `weights/FLAC/FLAC_EMA_metrics_1_1.0_exp16_calib_unseen_K8_seed42.json` (committed alongside this verdict). Reference: exp_01's `..._exp01_unseen_K8_seed42.json` (A6000, commit 0bd5da0). All-sample aggregate, per Rev 3 §2's disclosed exception.

| metric (JSON path) | exp16 (A100) | exp01 (A6000) | Δ | threshold (abs) | verdict |
|---|---|---|---|---|---|
| metrics.T60 | 8.6238 | 8.6238 | +0.0000 | 0.086 | PASS |
| metrics.C50 | 0.9688 | 0.9687 | +0.0001 | 0.0097 | PASS |
| metrics.EDT | 37.0803 | 37.0786 | +0.0017 | 0.371 | PASS |
| metrics.FD | 0.3053 | 0.3053 | +0.0000 | 0.0031 | PASS |
| metrics.RIR_to_GT_RIR_R@1 | 7.0696 | 7.1012 | −0.0316 | 0.30 | PASS |
| metrics.RIR_to_GT_RIR_R@5 | 19.4256 | 19.3940 | +0.0316 | 0.19 | PASS |
| metrics.RIR_to_GT_RIR_R@10 | 27.0317 | 27.0948 | −0.0631 | 0.27 | PASS |
| metrics["Invalid T60"] | 0.0 | == 0.0 | — | exact | PASS |

Companion cells 12287676 (unseen_s43, diagnostic) and 12287677 (seen_s42, descriptive) report separately in `_results.md`; per plan they do not gate.

**Authorization chain:** plan Rev 2 approved (Yixun 2026-08-11); Phase-2 submission on gate pass pre-authorized (Yixun 2026-08-12, worklog 10:20 entry); two-leg race strategy approved (Yixun 2026-08-12, worklog 18:15 entry).
