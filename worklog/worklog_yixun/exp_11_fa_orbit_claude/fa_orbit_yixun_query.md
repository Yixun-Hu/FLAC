# Yixun's queries — exp_11_fa_orbit

## Query 1 (2026-08-05) — commission

### Verbatim

> I need you to run a new experiment as a follow up of @worklog/worklog_yixun/exp_07_fa_scratch_claude/ fa_invariant conditioning and @worklog/worklog_yixun/exp_10_fa_scratch_resume_claude/. What I need you to do is to verify this hyposithes: currently we have a consistent better result for fa FLAC @worklog/worklog_yixun/model_comparison.md , but current fa method is C4, so I want you to try C8, C16 and C32(don't need to change the other model config, data config and training config, only change the averaging over the orbit number) to see whether we can get better results with more precise equivariance/invariance to the yaw.

### Summary

Sweep the frame-averaging orbit size of the fa_invariant conditioning: the current fa FLAC (exp_07 B-F / exp_10 resume) averages the ViT depth-path conditioners over the C4 yaw subgroup (0/90/180/270°). Train otherwise-identical from-scratch arms at C8, C16 and C32 and test whether a finer orbit — a closer approximation to full SO(2) yaw invariance — improves on the C4 results in `model_comparison.md`.

### Assumption / hypothesis (Yixun's, recorded faithfully)

fa FLAC's consistent advantage comes from training-side yaw invariance of the conditioning (established by the exp_10 decomposition cell). C4 invariance is exact only on the 90° subgroup; averaging over a finer orbit (C8/C16/C32) makes the conditioning more precisely invariant to arbitrary yaw, which may translate into further metric gains.

### Scope constraint (verbatim-derived)

Only `training.frame_avg_angles` changes. Model config, data config and training config otherwise stay byte-identical to the exp_07 `FLAC_AR_BF.json` recipe (SyncBN-64 DDP, eff-batch 64, seed 42, InverseLR, EMA on).

### Why this experiment needs to run

The C4 choice in exp_03/exp_07 was driven by exact-roll convenience (90° = 128 columns of the W=512 panorama), not by an optimality argument. The panorama/patch geometry admits finer exact subgroups (45°/22.5°/11.25° = 64/32/16-px rolls — all still aligned to the 16-px ViT patch grid; C32 is the finest patch-aligned orbit). Whether the fa advantage saturates at C4 or keeps growing with orbit precision is an open, decision-relevant question for the method's final form.

## Query 2 (2026-08-05, mid-turn) — GO with fast-recipe amendment

### Verbatim

> I don't need you to restart the exp_10. This experiment will be run on another computer. I need you to run exp_11 with the fastest training recipe (multi-card parallel withtout gradient checkpointing), and you should do profiling to test what blocks the training time first, similiar to option B

### Summary

(1) exp_10's 65k→67.5k completion is out of scope here (it moves to another machine). (2) exp_11 is a GO, amended: use the FASTEST training recipe — multi-GPU data parallelism WITHOUT ViT gradient checkpointing — rather than exp_07's 2-GPU + grad-ckpt recipe. (3) Before launching arms, PROFILE the training step to find what dominates wall-clock; then stage launches per the plan's Option B shape.

### Assumption / hypothesis (Yixun's, recorded faithfully)

Training wall-clock (dominated by the orbit's ViT passes) is the binding constraint on the sweep; more cards and no checkpoint-recompute should shrink it materially, and profiling will reveal the true bottleneck before GPU-weeks are committed.

### Planner note on recipe semantics (recorded at amendment time)

"Fastest" is constrained by recipe identity: with SyncBN, BN statistics see micro-batch × N_GPU, so keeping BOTH effective batch = 64 AND BN batch = 64 requires micro × N = 64 with accum 1 (candidate rungs 8×8, 16×4, 32×2). Grad-ckpt off changes memory/speed, not math. num_gpus/micro-batch/grad-ckpt are hereby recipe-free knobs (superseding Q1's "don't change training config" for exactly these throughput knobs); optimizer, schedule, seed, eff-batch, BN-batch semantics stay fixed. All four arms (C4L bridge + C8/C16/C32) train under the SAME chosen fast recipe, so internal comparisons remain single-delta in the orbit.

## Query 3 (2026-08-05, ~20:55 EDT) — recipe decision after P0 feasibility escalation

### Verbatim

> go with uniform grad-ckpt at the fastest rung

### Summary

Resolves the P0 escalation (no-ckpt infeasible for C8+ on 46 GB L40s: C8 OOM even at micro-8; C4L barely fits at 8×8 only). All four arms train with ViT gradient checkpointing ON, at the single fastest rung measured by the official ckpt-recipe P0 matrix. Consequence: the arm configs revert to a pure single-delta vs exp_07's `FLAC_AR_BF.json` (orbit angles only; C4L byte-identical), which strengthens the original design.

## Query 4 (2026-08-05, ~21:45 EDT) — C32 authorized

### Verbatim

> go for C32 as well

### Summary

Resolves the Option-B C32 gate: all four arms (C4L, C8, C16, C32) launch at the P0-selected rung under the uniform grad-ckpt recipe (Q3), to the 40k matched-step primary budget. No open staging decisions remain; the sweep runs as a complete four-arm commission.

## Query 5 (2026-08-06) — batched orbit acceleration

### Verbatim

> Is there any method to accelarate the C8, C16 and C32 training? 9 days is too long

*(options presented: batch the orbit through the ViT in large chunks — P0 utilization traces showed ~30% GPU busy, latency-bound sequential micro-forwards — vs launch as-is)*

> batch (option a)

### Summary

Adopt the batched-orbit execution of `invariant_conditioning`: identical averaging math, reordered execution (large chunked ViT forwards instead of per-angle micro-forwards), equivalence-gated against the loop implementation, used by ALL arms. Expected 2–3× on orbit-dominated arms (C32 ~9 d → ~3–4.5 d, single segment). Launches wait for: implementation + review + real-data equivalence probe + spot re-measurement + re-pin + smoke re-run under the new path.

## Query 6 (2026-08-06) — batched orbit adopted as disclosed recipe change

### Verbatim

> proceed

*(context: batching shares the train-mode DINOv3 stochastic RoPE rescale draw across a chunk's angles instead of per-angle independent draws — reviewer finding; presented as proceed-with-disclosure vs revert-to-loop)*

### Summary

The batched-orbit implementation is adopted as part of the exp_11 recipe, disclosed: identical averaging arithmetic; chunk-shared train-mode RoPE draws; applied identically to all four arms and the C4L bridge (sole inferential comparator); historical C4 rows labeled legacy-loop; eval-mode equivalence gated (fp32, 14 cells) since the augmentation is off at eval.

## Query 8 (2026-08-07, ~23:50 EDT) — overnight blanket approval

### Verbatim

> Currently I will go to bed, so I will approve everythin after your recommendation util I wake up, potential 10h from now. Please go ahead

### Summary

Standing approval (~10 h window) for the Planner's recommendations within the exp_11 framework: conf blocks + table rows as arms land, trajectory screens and gate bookkeeping, the pre-registered R2/R3/D measurement cells (including the small reviewed driver extension they require), and analyses. Scope stays inside the approved plan; anything outside it waits for morning.
