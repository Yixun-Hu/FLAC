# Lab notebook — exp_10_fa_scratch_resume

## 2026-08-01 — scaffold
- **Goal** — resume exp_07's B-F (fa_invariant from-scratch, SyncBN-64 DDP recipe) from `outputs_FLAC/exp07_BF/FLAC_exp07_BF/exp07_BF/checkpoints/epoch=8-step=40000.ckpt`; all screens under the fa protocol (`--cond-method fa_invariant`), correcting the exp_07 mismatch.
- **Version Control** — branch check-equivariance-necessity, base = exp_09 closure (`93449cb`).
- **Result** — `launched` (planning). Plan → Codex plan review → Yixun approval → launch.

## 2026-08-01T14:55:18-04:00 — plan Rev 2 APPROVED by Yixun → implementation round 1 (bf_resume_launch.sh)
- Coder = Opus 5 max seat; guard-tests + Codex review before use; then 15-step probe → launch.

## 2026-08-01T15:15:02-04:00 — round 1 SHIP (26/26 guards) → probe PASS (all lineage asserts; lr 4.903e-5 analytic; 15 fa steps) → COMMITTED RUN LAUNCHED (40k→67.5k, wandb)
- Acceptance criteria (pre-launch): full-state restore @40000, fa path active, 27,500 steps to 67,500, ckpt/2500, no OOM/NaN; screens per 2,500 with `--cond-method fa_invariant`; hard aborts only.
- Monitors armed (ckpt arrivals + death guard). ETA ~2.3 d → screens → R1–R3 readouts.

## 2026-08-04T11:21:39-04:00 — DECOMPOSITION CELL (vanilla P1@40k under fa eval, 5-seed both K): ensembling confound REFUTED — the invariance effect is training-side
- P1@40k + fa-eval: K8 8.817/1.0009/42.283/R4.049 · K1 10.257/1.0824/45.436/R3.926 — test-time 4-view averaging DEGRADES a vanilla model (R@1 −1.12, EDT +1.63 at K8 vs its own protocol; the model never learned averaged conditioning).
- Same-protocol matched-step comparison: fa-TRAINED 8.202/0.9778/38.79/R5.39 beats vanilla+fa-eval on all 12 cells (T60 −0.62, EDT −3.49, R@1 +1.34 at K8) → **fa's advantage is TRAINING-side invariance, not inference ensembling; "invariance makes the difference" licensed at matched steps (one training seed; 40k point).**

## 2026-08-05T15:45:00-04:00 — STALL DISCOVERED: run dead at step 65,000/67,500 (found during exp_11 reconnaissance)

- **Goal** — routine cross-check of exp_10 state while scaffolding exp_11 (orbit-size sweep).
- **Result** — `partial` (run stalled short of the pre-registered 67.5k endpoint):
  - Last checkpoint `epoch=14-step=65000.ckpt` written **2026-08-05T09:33:22-04:00**; no `train.py` process alive; all 8 GPUs at 0 MiB / 0% at 15:27.
  - Screens on disk through S62500 (fa protocol, K8 s42); S65000 not yet screened.
  - **The tee'd train log is GONE** (`fa_scratch_resume_2026-08-01_*_exp10_BF_train.log` absent from the exp folder), and every `wandb/run-*` dir from before 2026-08-05 is gone, while ALL tracked files carry mtime 2026-08-04T23:17 — i.e. an untracked-file wipe / re-checkout of this working tree happened at ~23:17 on Aug 4 (a second session's smoke runs `flac_readme_smoke_neuronic` appear in wandb from 00:39 Aug 5 onward). Checkpoints under `outputs_FLAC/` survived.
- **Analysis** — **infrastructure, not a real bug**, on the available evidence: the training process survived the Aug-4 wipe (ckpts kept landing on cadence: 60k @19:10 Aug 4, 62.5k @02:22 Aug 5, 65k @09:33 Aug 5) and died between 09:33 and ~15:00 Aug 5 with its stdout/stderr log destroyed, so the proximate cause (kill, crash, OOM from cotenant pressure) is not diagnosable. Step cadence had also slowed to ~0.095 steps/s (vs ~0.14 planned), consistent with cotenancy load. The 65k checkpoint is a clean resume boundary; `bf_resume_launch.sh` RESTART mode (EXPECTED_STEP=65000, ckpt inside `outputs_FLAC/exp10_BF/`) covers exactly this case — ~7.3 h × 2 GPUs to reach 67,500.
- **Next** — restart decision deferred to Yixun (surfaced 2026-08-05 together with the exp_11 plan): the R1 primary readout (B-F@67.5k vs P1@67.5k, 5-seed both K) is blocked until the endpoint exists. Also flagged: wandb lineage for this run is broken (log + run dirs destroyed); the on-disk ckpt/metric JSONs are the surviving record.

## 2026-08-05T21:0x — RECONCILIATION (two machines, one experiment)

- The stall entry above describes the CLUSTER copy of exp_10 (8-GPU machine; post-Aug-4-wipe, log/wandb loss, dead at 65k). **The ORIGINAL run on the A6000 box (this session) never stalled: it completed 67,500 at 2026-08-05 ~18:5x EDT and the endpoint gate block ran there** — the entry below records those gates. Checkpoints/metrics of record live on the A6000 box under `outputs_FLAC/exp10_BF/`; the cluster restart decision is MOOT (no restart needed — the run is done). Cross-machine metric-JSON consolidation per the pending metrics_json proposal.

## 2026-08-05T20:49:19-04:00 — ENDPOINT GATES: R3 PASS (exact C₄; 45° breaks) · candidate rule NO QUALIFIER · R1 endpoint = band-worst draw → tier **SHORT** by the fixed rule; window reading contradicts the endpoint (pre-registered R1b)
- **R1 (fixed endpoint, 5-seed):** BF@67.5k K8 10.0412±0.0118/1.0050±0.0012/42.1068±0.0373/R6.6656±0.0607 vs P1@67.5k 8.757/0.9753/36.962/6.154 — P1 better T60/C50/EDT (z +15…+79), **BF better R@1 at BOTH K (z +2.9/+3.2)**. K1 same shape. → **SHORT** per the bounded tiers.
- **Endpoint anomaly (honest note):** the 67.5k draw (T60 10.04) is far outside BF's own screen band (8.58–9.42 over 42.5k–65k; 62.5k read 8.582) — a band-worst endpoint, the same phenomenon as exp_07 B-V's 67.5k endpoint (band max). The fixed-rule verdict stands (that is what pre-registration means), but R1b (window, exploratory): BF's screens track P1's curve on T60/C50/EDT and lead on R@1 from 50k onward.
- **R2:** 0/8 (endpoint). **R3: PASS** — C₄ spreads ≤0.0315 (T60 0.0009/C50 0.0001/EDT 0.0011), 45° control breaks (T60 +2.04, R@1 −2.05). Equivariance exact at the endpoint.
- **No extension offer** (plan: requires VIABLE-or-better). Standing insight preserved: at matched steps ≤65k, fa-scratch ≈ vanilla on error metrics + ahead on retrieval (the 40k 5-seed 12/12 result + decomposition remain the paper-grade evidence); the endpoint draw does not retract those.
- Next: closing package (results/analysis/HTML/closure review) — tomorrow morning.
