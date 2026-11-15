# exp_38 data_efficiency — Yixun's driving query

## Query 1 — 2026-11-07 (relayed by the main session; the verbatim Chinese/English original is held by the main session's transcript)

**As relayed to the Coder/Planner (verbatim brief text):**

> Yixun (the researcher) asked for: an acoustic-metric-accuracy vs training-data-amount comparison (25 / 50 / 75 / 100 % of the AcousticRooms (AR) training data) for three models — FLAC (vanilla, "P1"), Yaw-Augmented FLAC ("YAW") and OrbitRIR ("B-F", fa_invariant C4 frame-averaged conditioning) — at K_ctx = 8, reported as mean ± SD over evaluation seeds 42–46 (EDT ms, C50 dB, T60 %), like Table 3(b) of the CERPA paper.

> The job: 6 new 40k-step training runs — YAW and B-F at 25 / 50 / 75 %, using EXACTLY the same nested subsets as exp_14's dc_van runs (same subset definition/manifests, same context restriction), each model's own 100 % recipe otherwise unchanged (seed 42, eff batch 64, 40k steps, its conditioning/augmentation, its precision), so each fraction is directly comparable to dc_van_fXXX and to the 100 % checkpoints. Then 5-seed K8 evaluation (seeds 42–46; also K1 if it is free in the existing eval tooling) of each final checkpoint with each model's own conditioner (vanilla for YAW, fa_invariant C4 for B-F), same evaluator/protocol as exp_14 used for dc_van so numbers line up with the table.

> Platforms: (a) this machine's 2×A6000 (DDP 2×32) and (b) neuronic (Princeton slurm cluster, repo at /n/fs/gatrdp/codespace/FLAC, 8×L40 nodes; exp_15's sbatch is the template). [...] Every script must be able to run any of the 6 runs on either platform. Recommended default allocation (Yixun may change it): neuronic runs all 3 YAW and B-F 50 % / 75 %; this machine runs B-F 25 %. Record exact per-platform differences (micro-batch × GPUs, precision) and make the effective batch / SyncBN batch identical (64).

Standing constraints restated in the brief: never touch `main`; at most 3 Codex plan-review and 3 code-review rounds; no real 40k training started by the Coder; GPU 1 is busy with someone else's evaluation jobs until ≈ 04:00 Nov 8 (never use, signal or stop them); a ≤ 15-min GPU smoke on GPU 0 only is allowed (a few YAW f025 / B-F f025 steps at the real batch).

### Summary
Extend Table 3(b)'s data-efficiency curve (today only FLAC: exp_14 `dc_van_f025/f050/f075` + the P1@40k anchor) to Yaw-Aug FLAC and OrbitRIR: train each at 25 / 50 / 75 % of the AR training split on exp_14's nested subsets with exp_14's context restriction, each with its own 100 % recipe, and evaluate the 40k endpoints on the full unseen split (K = 8 headline, K = 1 too), 5 eval seeds, the same protocol as the table's existing cells.

### Yixun's hypothesis (as understood by the Planner)
The yaw-equivariant OrbitRIR (exact C4 invariance by frame averaging) and, to a lesser degree, the yaw-augmented FLAC should be more data-efficient than vanilla FLAC — their acoustic error should degrade less as training data shrinks.

### Why the experiment needs to run
No YAW or B-F checkpoint trained on less than 100 % of the AR split exists; the table needs them to make a data-efficiency claim for the paper.

### Assumptions logged by the Planner (each surfaced in `plan_data_efficiency.md` §13 Decisions)
- "Exactly the same subsets" = exp_14's committed split JSONs and dataset configs byte-for-byte (sha256-pinned), including `acoustic_context.restrict_to_split: true`.
- "Each model's own 100 % recipe" = the recipe that produced the checkpoint each table row uses (P1 `c4c67882…`, YAW `ac1f2603…`, B-F `5319feb4…`) — verified from the checkpoints' embedded configs and the experiment records, see plan §3. **Correction surfaced:** the YAW 100 % checkpoint is exp_17's 2×A6000 arm, not exp_15's 8×L40 arm.
