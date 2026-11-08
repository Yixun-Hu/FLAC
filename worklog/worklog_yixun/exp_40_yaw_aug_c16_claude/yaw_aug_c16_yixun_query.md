# exp_40 yaw_aug_c16 — Yixun's driving query

## Query 1 — 2026-11-07 (relayed by the main session to the Coder/Planner seat)

**Verbatim (as relayed):**

> Yixun asked for another yaw-augmentation training run that does NOT use all 512 pixel offsets: Yaw-Aug FLAC with Δp ∈ {0, 32, 64, …, 480} — 16 discrete column offsets, i.e. yaw θ = k·22.5°, k uniform over {0,…,15} (the C16 group), instead of exp_15's d uniform over {0,…,511}.

The relayed brief adds: the new arm must differ from the existing Yaw-Aug arm in exactly one thing (the offset set); same code, seeds and recipe; a new optional `yaw_aug` key (e.g. `"offset_step": 32`) whose absence keeps the existing behaviour byte-identical (regression test reproducing the existing draws); config `FLAC_AR_YAWAUG_C16.json` = `FLAC_AR_YAWAUG.json` + that one key; a neuronic sbatch (Yixun submits) and a 2×A6000 fallback launcher, both with DRY_RUN; checkpoints every 2,500 steps, the 40k checkpoint copied to the NAS; an evaluation plan with ready-to-run commands on this machine reusing the reviewed exp_25/exp_36 machinery (θ = 0, C4, 45°-grid, off-grid and random-yaw blocks, K ∈ {1, 8}); Codex review within the 3 + 3 round cap; at most a ≤ 15-min smoke on GPU 0; do not start the real 40k training.

**Summary:** train and evaluate a C16-discrete Yaw-Aug FLAC arm — the existing per-sample yaw augmentation restricted to the 16 column offsets that are multiples of 32 — and place it next to the existing Yaw-Aug (uniform over 512 columns) and vanilla P1 arms in the yaw-robustness figures.

**Assumption / hypothesis (surfaced, not Yixun's words):** a discrete C16 augmentation may buy (near-)invariance on its own 22.5° grid at least as well as the continuous one, but its behaviour BETWEEN grid points (the off-grid angles 11.25°, 33.75°, 56.25°, 101.25°, each exactly half a grid step from the nearest training angle) is the open question: a model that interpolates smoothly in yaw should stay flat there; one that memorises the 16 training orientations should degrade there, towards vanilla.

**Why this experiment needs to run:** it separates "augmentation over a finite rotation group" from "augmentation over the full circle" at the same budget — the data-side analogue of the C4/C8/C16 frame-averaging orbit sizes already in the record — and tests whether the continuous augmentation's off-grid flatness (exp_17's 45° probe) is something a 16-element group already delivers.

**Planner correction to the brief (see plan §0):** the YAW checkpoint in the existing figures (`ac1f2603…`) is exp_17's 2×A6000 arm, not exp_15's neuronic 8×L40 arm (`16b964ec…`). The recipe to match for "drops into the existing figures" is therefore exp_17's 2 ranks × 32, which makes the platform choice a decision for Yixun (plan §0 D1).
