# exp_24 — HAA orientation-cue fairness ablation

**Commissioned by Yixun 2026-09-26 (verbatim):** "Control for the additional loudspeaker-orientation information used by
CERPA during HAA fine-tuning. Currently, CYLDINO receives an explicit loudspeaker-facing cue, while the vanilla FLAC
baseline does not, so improvements on HAA may be partially attributable to the additional conditioning rather than
CERPA itself. Fine-tune and evaluate the following variants under the same HAA protocol: FLAC; FLAC + orientation cue;
Yaw-Augmented FLAC + orientation cue; CYLDINO + orientation cue. The key controlled comparison is FLAC + orientation cue
vs. CYLDINO + orientation cue."

**Seat:** Claude Fable 5.1 (cylindrical-dinov3 main session), LEAN mode. Everything reproducible from this folder.

## Arms (all: exp_19 registered HAA recipe verbatim; eval = exp_19/exp_23 protocol)

| arm | backbone | init (AR-40k EMA) | cue | conditioning path | status |
|---|---|---|---|---|---|
| FLAC (`P1`) | stock DINOv3-S | exp07_P1 | no | vanilla | exp_19 record (committed) |
| **FLAC + cue (`P1ORI27`)** | stock DINOv3-S, patch conv widened 3→6 | same weights, zero extra | yes, s=27 | vanilla | **this experiment** |
| Yaw-aug FLAC (`YAW`, aug ON in FT) | stock DINOv3-S | exp17_YAWAUG | no | vanilla | exp_19 record |
| **Yaw-aug FLAC + cue (`YAWORI27`)** | stock, widened | same, zero extra | yes, s=27 | vanilla (yaw aug rotates the cue) | **this experiment** |
| CylDINO (`CYL`) | cylindrical DINOv3-S | exp09_cylNoSSL | no | fa_invariant [0] | exp_19 record |
| CylDINO + cue (`CYLORI27`) | cylindrical, widened | same, zero extra | yes, s=27 | fa_invariant [0] | exp_23 record |
| (ref) `YNA` = yaw-aug init, stock FT (aug OFF) | stock | exp17_YAWAUG | no | vanilla | exp_19 record |

Key comparison: **P1ORI27 vs CYLORI27** (same information available to both; isolates the encoder). Secondary:
P1ORI27 vs P1 (does the cue alone help the world-frame model?), YAWORI27 vs YAW (does the cue restore what yaw
augmentation removed?).

## Design notes (read before interpreting)

- The cue is identical across arms: `md['facing']` (world-frame unit vector, `haa_speaker_facing.json`; −y in every
  Base room) appended as a second XYZ triple `s·f`, `s = 27`, to the geometry-encoder input, with the patch conv widened
  by three ZERO-initialised channels so step 0 equals the base arm exactly (`smoke_construct_vanilla.py`: max|Δ| = 0.0
  for both new arms on a real hallway sample; `exp24_init_sha.txt` = widened inits `HAA_init_P1ORI.ckpt`,
  `HAA_init_YAWORI.ckpt`, built with exp_23's `widen_init.py` from the sha-verified exp_19 inits).
- On the stock (world-frame) backbone the field is constant over pixels AND over the Base rooms, so it carries no
  information the world-frame layout does not already give; the fair-control question is exactly whether the extra
  parameters/bias help vanilla FLAC. On CylDINO the gauge turns the same vector into `Rz(−θ_u) f`, a column-relative
  field — that is the information the invariant encoder had lost. Under yaw augmentation (`YAW`, aug ON) the facing is
  rotated with the scene (`rotate_scene_metadata`), so for that arm the cue is informative at train time; at test time
  (rotate 0) it equals the constant field.
- Implementation: `src/models/conditioners.py` — new `_widen_hf_patch_conv()` on the stock HF path when the conditioner
  config carries `orientation_field` (default-inert; banner "orientation_field ENABLED (vanilla backbone)…"); the
  orientation-field agreement guard now also covers the vanilla path. The cylindrical package is NOT on the path for
  these arms (`PYTHONPATH=""`), enforced by the launcher's banner checks.
- Configs: `FLAC_HAA_finetune_P1ORI27.json` = stock `FLAC_HAA_finetune.json` + `orientation_field/scale` on both
  ViTCoordinates blocks; `FLAC_HAA_finetune_YAWORI27.json` = exp_19 `FLAC_HAA_finetune_YAW.json` + the same. Dataset
  configs = exp_23's `haa_{train,val,test,test_1}_ori.json` (stock + `HAA_md_ori.py`, which only adds `md['facing']`).
- Recipe: 1,000 steps, batch 16 × accum 4, AdamW 5e-6 + InverseLR, VAE frozen, val every 10, ckpt every 100, seed 42,
  bf16-mixed, one GPU (co-tenant with the exp_13 tier-L trainings). Eval: `--cond-method vanilla`, bf16 autocast,
  cfg 1.0, one step, per-scene records; ckpt-1000 × K∈{8,1} × seeds 42–46 + K=8 seed-42 curve (100…900).
- Scripts: `haa_ft_ori_launch.sh` (SMOKE/FULL), `haa_ft_ori_eval.sh`, `chain_arm.sh` (FULL → evals → results),
  `exp24_write_results.py` → `results_fairness.md` (tables, controlled comparisons, per-room, curves, pooled, LaTeX rows).

## Status
- 2026-09-26 17:31 EDT: both FULL chains launched detached (`chain_arm.sh`): P1ORI27 on GPU 1 (co-tenant with the
  stock-L training), YAWORI27 on GPU 0 (co-tenant with the CylDINO-L training). SMOKE (3 steps) passed for both with the
  required banners (vanilla backbone; "orientation_field ENABLED (vanilla backbone): … 6 input channels (scale=27.0)";
  no cylindrical backbone; YAWORI27 additionally "yaw_aug ENABLED img_w=512 seed=42"). CPU smoke: step-0 outputs
  bit-identical to P1 / YAW. Observed footprint: **~20.5 GB per fine-tune** (the "~4 GiB" note inherited from the exp_23
  launcher is wrong for this recipe; VRAM_FLOOR should be ≥ 22000 next time). Expected: FULL ≈ 5 h each (3.9 h solo in
  exp_23, slower as a co-tenant), evals ≈ 45 min, results ≈ 23:45 (P1ORI27) / 00:30 Sep 27 (YAWORI27).
