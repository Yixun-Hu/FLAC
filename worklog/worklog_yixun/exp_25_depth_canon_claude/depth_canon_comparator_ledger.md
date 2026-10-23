# exp_25 — comparator ledger (plan §5; review H5/H6)

Every number from another experiment that exp_25 quotes or pairs against is listed here with its artifact-level provenance. **Status column:** `raw` = the per-seed metric JSON exists on this box and feeds `gen_model_comparison.py`; `md` = only the committed markdown table survives (raw JSONs lost or on another machine); `re-measured` = exp_25 re-evaluates the checkpoint at the exp_25 evaluator pin (plan §5 P1-pin / YAW-pin blocks). Updated as blocks complete.

## Checkpoints (NAS `/media/diskstation/yixunhu/FLAC/checkpoints/ar_40k_endpoints/`, manifest `MANIFEST.sha256`, archived 2026-08-21)

| Arm | File | sha256 | Embedded `model_config.training` keys of note |
|---|---|---|---|
| P1 (vanilla) | `P1/epoch=8-step=40000.ckpt` | `c4c678826cddda37fa4977926aadee530afd037b3abb110918b52a342ce9845c` | none of `cond_method` / `yaw_aug` / `canonicalize` (checked 2026-10-23) |
| B-F (fa_invariant) | `BF/epoch=8-step=40000.ckpt` | `5319feb4af874624859e87105ddd8ab06d4b449769d1e054f712b2b1c0542328` | `cond_method: fa_invariant`, `frame_avg_angles` |
| YAW (yaw-aug) | `YAW/epoch=8-step=40000.ckpt` | `ac1f26034e4f341fe0c2cb4638e2eb473959d66ddd2fd95d184dc2fd4f264de7` | `yaw_aug` |
| (B-V, unused) | `BV/…` | `ace9f735…` | — |

Local working copies: `~/codespace/FLAC/weights/exp20/{P1,BF,YAW}_40k.ckpt` (exp_20 admissions `loc_crossarm_admission_*.json` carry the same shas).

## Table rows (θ = 0, unseen split, 5 eval seeds 42–46, item-weighted mean ± sd)

| Row | Source experiment | Evaluator commit / protocol | Raw files | Status |
|---|---|---|---|---|
| P1 @40k K1/K8 | exp_07 (P1 arm), 5-seed fill in exp_10 | `eval_FLAC.py`, vanilla, bf16 cond autocast, cfg 1.0, 1 step, EMA | `outputs_FLAC/exp07_P1/**/…exp07_P140_K{1,8}_s4[2-6]*.json` (9) + `…exp07_P1_screen_S40000_ema.json` (K8 s42) | `raw` — AND `re-measured` at the exp_25 pin (P1-pin T block) |
| B-F @40k K1/K8 | exp_07 (B-F arm), 5-seed fill in exp_10 | `--cond-method fa_invariant`, legacy per-angle loop (pre-`1479304`), bf16 | `outputs_FLAC/exp07_BF/**/…exp10_BF40_K{1,8}_s4[2-6]*.json` (9) + screen | `raw` (historical; NOT re-measured — fa eval; orbit execution now `batched`, numerically equal in eval mode per `yaw_rotation.py` docstring but not re-verified here) |
| YAW @40k K1/K8 (9.397 / 7.965 …) | exp_17 (A6000 arm, grad-ckpt ON) | vanilla, bf16, cfg 1.0, 1 step, EMA; evaluated from the removed worktree `~/codespace/exp-17-yawaug-a6000` | `outputs_FLAC/exp17_YAWAUG_extras/` in that worktree — **absent** from this box and the NAS (searched 2026-10-23) | `md` (`origin/exp17-yawaug-scratch:…/results_yaw_aug_a6000_endpoint_rows.md`) — AND `re-measured` (YAW-pin T block). ⚠️ `model_comparison.md`'s current "yaw-aug vanilla YAWAUG @40k (exp_15)" row is the **neuronic 8×L40** arm (10.067 / 8.503), a different recipe — never used as the comparator here |

## Fixed-angle invariance references

| Quantity | Source | Value | Status |
|---|---|---|---|
| P1 Δ_C4 @40k, seed 42 (K1 / K8) | exp_17 P1 control orbit (`results_yaw_aug_a6000_p1ctrl_grid_table.md`, branch `origin/exp17-yawaug-scratch`) | ΔT60 0.922 / 0.903; ΔC50 0.133 / 0.134; ΔEDT 6.497 / 6.807; ΔR@1 0.584 / 0.473 | `md` — AND `re-measured` (P1-pin C4/45 cells) |
| YAW Δ_C4 @40k, seed 42 (K1 / K8) | exp_17 `results_yaw_aug_a6000_c4_grid_table.md` | ΔT60 0.061 / 0.075; ΔC50 0.011 / 0.013; ΔEDT 0.384 / 0.349; ΔR@1 0.047 / 0.237 | `md` — AND `re-measured` (YAW-pin C4/45 cells) |
| B-F C₄ exactness | exp_08 H-A2/H-A3 (fine-tuned fa_invariant A-F ckpt, not the B-F endpoint): waveform relL2 0.0023 at 90/180/270° (floor 0.0093), 0.206 at 45°; metrics flat to ΔT60 ≤ 0.001 | by construction of the C₄ frame average (exp_03 conditioning-level 4.9e-8) | `md` (structural; no B-F@40k rotated cells on this box — `outputs_FLAC/exp07_BF` holds only `rot0` and exp_11 cross-orbit cells) |
| YAW 45° probe (K8, seed 42) | exp_17 (c) | T60 7.996, EDT 40.523, C50 1.020, R@1 5.207 | `md` — AND `re-measured` |

## Random-yaw references (exp_14 protocol)

Only neuronic-recipe arms exist (exp_14 VANL/C4L/C8/C16/C32; exp_15 YAWAUG) — **cross-recipe, context only**. exp_25 measures random-yaw blocks for CANON, P1 and YAW at one pin (plan §5); B-F has none (fa eval out of scope).

## Trajectory references (seed 42, C4 grid per checkpoint, 2.5k → 40k)

P1 and YAW grids: exp_17 md tables above (`md`); CANON: exp_25 TRAJ block (plan §5).
