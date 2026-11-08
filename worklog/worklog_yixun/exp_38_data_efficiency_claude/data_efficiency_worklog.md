# exp_38 data_efficiency — lab notebook (append-only)

## 2026-11-07T20:24:00-05:00 — scaffold: worktree, branch, folder
- **Goal** — start exp_38 per the SOP (query, plan, reviews before code).
- **Version Control** — branch `exp-38-data-efficiency` cut from `localization-exp` `2b42468c8f429b1d93da93b3e4ddb7bd6f8f1932` (= `origin/localization-exp` after `git fetch`, 20:24 EST); worktree `~/codespace/exp-38-data-efficiency`; untracked asset links: `AcousticRooms -> /media/diskstation/yixunhu/FLAC/AcousticRooms`, `weights/AGREE -> ~/codespace/FLAC/weights/AGREE`, `weights/FLAC/VAE.{safetensors,ckpt} -> ~/codespace/FLAC/weights/FLAC/…` (the tracked `weights/FLAC/*.json` stay).
- **Result** — scaffolded `data_efficiency_yixun_query.md`, this notebook, `commits_data_efficiency.md`.
- **Next** — investigation → plan r0.

## 2026-11-07T20:55:00-05:00 — investigation → plan r0
- **Goal** — establish, with evidence, the three 100 % recipes, exp_14's subset mechanism, composition with yaw_aug / fa_invariant, platform differences, estimates.
- **Command / Validation** — read-only: git history/blobs of `exp-14-data-curve`, `localization-exp`, `origin/exp17-yawaug-scratch`; the exp_07/exp_14/exp_15/exp_17/exp_25/exp_35 records; CPU `torch.load` of the three 100 % checkpoints (embedded `model_config`, step, scheduler); recomputation of the Table 3(b) cells from raw JSONs; sha256 of every file to be ported at all three exp_14 launch commits.
- **Result** — key findings (all in `plan_data_efficiency.md`):
  1. The YAW 100 % checkpoint `ac1f2603…` is **exp_17's 2×A6000 arm** (embedded config == `exp_15…/FLAC_AR_YAWAUG.json`, grad-ckpt ON), not exp_15's 8×L40 arm — the brief's recipe pointer is corrected.
  2. P1, YAW, B-F 100 % and dc_van ×3 were ALL trained at 2 ranks × 32 / GPU, SyncBN, bf16-mixed, seed 42 on the A6000 box.
  3. The subset mechanism is data-side only; it lives only on `exp-14-data-curve`; yaw_aug + the FA chunk-cap key live only on `localization-exp` → cut from `localization-exp` + port exp_14's data side (AR_md.py byte copy; dataset.py +7 lines; data files byte copies).
  4. B-F 100 % = legacy per-angle orbit; today's default cap 64 at micro 32 would share {90°,180°} RoPE draws → our B-F config must set `training.frame_avg_max_fwd_samples` = micro (32).
  5. The 8×8 neuronic rung is a recipe change (per-rank micro / RoPE-draw granularity; same-model evidence VANL 8×8 vs P1 2×32: 0.95 T60 gap) → recommend 2×L40 × 32 on neuronic.
  6. Our base's evaluator is byte-identical to the one that produced all three 100 % anchors (exp_25 `d6ca307a`, exp_35 `5b065801`); dc_van used exp_14's evaluator → E0 replay gates.
- **Next** — Codex plan review r1 (1 of max 3).

## 2026-11-07T21:05:00-05:00 — Codex plan review r1 (1 of 3) = REVISE → plan r1
- **Goal** — independent plan review before code (SOP; cap 3 plan rounds).
- **Command / Validation** — `codex exec -s read-only -m gpt-5.6-sol -c model_reasoning_effort=xhigh -C ~/codespace/exp-38-data-efficiency -o <scratch>/plan_r1_review.md "<prompt>" < /dev/null`; transcript `data_efficiency_2026-11-07_20-30-50_codex_plan_r1.log.gz`; review verbatim `data_efficiency_codex_plan_r1_review.md`.
- **Result** — REVISE: 4 BLOCKING (platform confounding of the default allocation; mid-epoch chaining; submit-exactly-once; fail-open flash gate), 13 HIGH, 6 NIT. All addressed in plan r1 (§15 changelog); r0 archived as `plan_history/plan_data_efficiency_r0.md`.
- **Analysis** — the decisive change: the primary design moves all six runs onto this box (every reference point trained here at 2 × 32); neuronic becomes a labelled sensitivity platform; chains are dropped (monolithic only). Process slip recorded (Codex HIGH 13): round-A port + first kit drafts were written in the worktree while r1 ran — uncommitted, provisional, reconciled with r1 and sent to the code review as code.
- **Next** — Codex plan review r2 (2 of 3); implementation of plan r1 continues in parallel, r2 findings bind both plan and code.

## 2026-11-07T21:30:00-05:00 — Codex plan review r2 (2 of 3) = REVISE → plan r2; subset audit PASS
- **Result** — r2 REVISE: 2 BLOCKING (E0 incomplete and not upstream of FULL; the "Yixun admits a resumed run" exception), 12 HIGH, 5 NIT; all folded into plan r2 (§15). Review verbatim `data_efficiency_codex_plan_r2_review.md`, transcript `data_efficiency_2026-11-07_20-48-56_codex_plan_r2.log.gz`; r1 archived.
- **Key r2 design moves** — E0 (8 cells, exact, input-hash pinned) produces a bound receipt that FULL requires; crash ⇒ re-run from 0; an observe-only in-process evidence callback (`--evidence-json`, default off) proves on every rank the initial-state digest, Sobol/RNG states, active SyncBN, non-reentrant ViT checkpointing, flash use in the DiT and B-F's per-angle conditioner calls; an immutable campaign worktree + closure digest + symlink allowlist + environment receipt; a card lock shared by training and eval; Slurm intent tokens; eval `--num-workers 4` (= every reference cell; the context draws depend on it).
- **Subset audit (rung 3, first run, pre-r2 code):** PASS in 60 s (`data_efficiency_2026-11-07_20-56-16_subset_audit.log`): 73,473 / 145,665 / 218,463 targets, 243 rooms, 0 empty restricted pools, per-room histograms == exp_14's manifest, ordered-target sha256 recorded; **at 100 % zero on-disk candidates lie outside `train.json`** (seen-eval hold-outs are whole receivers; pools are same-receiver) — the 100 % anchors' pools were effectively split-restricted too. To be re-run with the r2 audit code.
- **Next** — implement plan r2 (production: the evidence callback + flag; kit; eval; collector), tests, dry-runs, GPU-0 smoke; then Codex code review 1 of 3 on the whole diff. Plan review r3 kept in reserve.
