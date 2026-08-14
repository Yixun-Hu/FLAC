# Lab notebook — exp_14_fa_drawshare

## 2026-08-12 — scaffold + plan Rev 2
- **Goal** — does the operational chunk plan (per-angle vs fully-shared RoPE draws) change FA training outcomes at 40k? Single delta, seed 42, sequential arms.
- **Decisions on record** — Yixun approved the discriminating training ("做，要把因果钉死"), then, after the plan review showed 12 days cannot support a general causal claim, chose **option B** (run the scaled-down version, scope the claim to the seed-42 trajectory). Sequencing "顺序跑".
- **Rev 2 applied all review findings**: cap 32 vs **96** (1/3 vs **3/3**, matching exp_11's draw-sharing TOPOLOGY — the earlier "micro-32 tops out at 2/3" was my arithmetic error); config key instead of env override; from-scratch launcher; claim scoping; DS2 downgraded to a cross-era replication check; registered trajectory statistic; fit probe gate for cap 96.
- **Result** — `planning` (NOTHING launched; no code written, no GPU used). Rev 3 after the re-review's two blockers → Yixun's final go → cap-96 fit probe → DS-PA → admission/parity audit → DS-CS3.

## 2026-08-14T15:06:15-04:00 — DS-PA PAUSED CLEANLY at step 5,000 (Yixun-ordered); campaign fully idle
- Watcher fired at 15:05:13: ckpt `step=5000` on disk, all 14 ranks exited, GPUs released (GPU0 42.6 GB free). Resume command recorded in `fa_drawshare_command.md`; RESTART gate will re-verify embedded config, cap, and full optimizer/scheduler/EMA state.
- **Both exp_14 arms are now held:** DS-PA paused at 5,000/40,000 (7.6 d of training still owed when resumed); DS-CS3 never started (its cap_fit evidence is stamped and still valid while HEAD is unchanged — note any new commit to the fingerprinted paths voids it and it must be re-stamped).
- wandb run for DS-PA will show "crashed" — that is this deliberate pause, not a failure.
