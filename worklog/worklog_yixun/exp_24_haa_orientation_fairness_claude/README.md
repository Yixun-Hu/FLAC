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
- 2026-09-27 00:13 EDT: **both chains DONE** (P1ORI27 FULL 21:39, 19/19 cells rc=0 at 23:53; YAWORI27 FULL 21:51,
  19/19 cells rc=0 at 00:12). Results: `results_fairness.md`. Headline (ckpt-1000, K=8, paper convention):

  | arm | T60↓ | C50↓ | EDT↓ | R@1↑ | R@10↑ | hallway T60 |
  |---|---|---|---|---|---|---|
  | FLAC (P1) | 3.413 | 2.202 | 85.0 | 5.18 | 31.69 | 3.43 |
  | FLAC + cue (P1ORI27) | 3.320 | **1.869** | 76.2 | 4.68 | 30.73 | 3.35 |
  | Yaw-aug FLAC (YAW, aug ON) | 4.092 | 2.777 | 91.8 | 4.13 | 27.34 | 5.69 |
  | Yaw-aug FLAC + cue (YAWORI27) | **3.301** | 1.930 | **73.0** | 4.91 | 30.76 | **3.29** |
  | CylDINO (CYL) | 5.411 | 3.442 | 119.5 | 4.10 | 27.65 | 8.85 |
  | CylDINO + cue (CYLORI27) | 3.533 | 2.158 | 85.8 | 4.96 | 31.45 | 4.02 |

  **Key controlled comparison (CYLORI27 vs P1ORI27, information matched):** CylDINO + cue trails FLAC + cue on the decay
  metrics — T60 +6.4 %, C50 +15.5 %, EDT +12.6 % (K=1: +5.5 / +15.8 / +13.6 %) — and leads slightly on retrieval
  (R@1 +6.2 %, R@10 +2.3 %). So the exp_23 "parity with vanilla" headline was partly the cue helping in general: the cue
  also improves vanilla FLAC (C50 −15 %, EDT −10 %, T60 −3 %), even though on the world-frame backbone the field is a
  constant. Mechanism (per-room): P1's gains are in the classroom/complex/dampened C50–EDT, NOT in the hallway (3.43 →
  3.35) — i.e. the zero-init channels × s=27 act as a fast-learning extra bias on the patch embedding (10× effective LR
  under Adam), an optimisation/capacity effect; for CylDINO the same cue acts where the information was missing (hallway
  8.85 → 4.02) and for yaw-aug FLAC it restores what augmentation removed (hallway 5.69 → 3.29, the best hallway of all
  arms; YAWORI27 is the best arm overall on T60 and EDT). Retrieval: the cue costs vanilla FLAC R@1 (5.18 → 4.68) but
  helps every invariant/augmented arm. Caveats: one training seed per arm; s=27 was tuned on CylDINO and copied to the
  stock arms unchanged; the facing direction is the same vector in every room (dataset-supplied).
- 2026-09-27 12:05 EDT: **zero-information control arms launched** (Yixun: "Run the zero-field control"). A literal zero
  field is a no-op (the extra conv weights receive gradient ∝ the field, so they stay zero and the arm IS plain FLAC),
  so the control is the same-scale constant field with no orientation content: the world UP vector (0,0,1)·27 in every
  room (`haa_zup_field.json`, `HAA_md_zup.py` = HAA_md_ori.py with the table swapped, `haa_{train,val,test,test_1}_zup.json`).
  z is invariant under the cylindrical gauge Rz, so the field is a constant on BOTH backbones. Arms: `P1ZUP27` (stock,
  widened conv, P1ORI init, GPU 1) and `CYLZUP27` (cylindrical, widened conv, CYLORI init, GPU 0); recipe/eval identical
  to the facing arms; ckpt cadence 1000 (endpoint only — disk was at 19 GB, see below), so no steps curve for the controls.
  Reading rule: P1ORI27 ≈ P1ZUP27 ⇒ the vanilla-FLAC gain is the bias/extra-parameter effect; CYLORI27 ≪ CYLZUP27 ⇒ the
  CylDINO gain is the orientation information. SMOKE passed for both (banners + `HAA_md_zup.py (zup field)`).
  Disk note: the box hit 19 GB free at 11:49 (the exp_13 disk guard stopped the stock-L training); pip/uv caches purged
  (→ 27 GB) before launching; nothing deleted from any run.
- 2026-09-27 18:50 EDT: **control arms DONE** (P1ZUP27: FULL 15:55, 10/10 cells 16:30; CYLZUP27: FULL 18:07, 10/10 cells
  18:50; all rc=0). `results_fairness.md` regenerated (9 arms). Verdict (ckpt-1000, K=8, paper convention):

  | arm | T60↓ | C50↓ | EDT↓ | R@1↑ | R@10↑ | hallway T60 |
  |---|---|---|---|---|---|---|
  | FLAC | 3.413 | 2.202 | 85.0 | 5.18 | 31.69 | 3.43 |
  | FLAC + facing cue | 3.320 | 1.869 | 76.2 | 4.68 | 30.73 | 3.35 |
  | FLAC + constant field (control) | 3.334 | 1.864 | 75.6 | 4.66 | 30.81 | 3.32 |
  | CylDINO | 5.411 | 3.442 | 119.5 | 4.10 | 27.65 | 8.85 |
  | CylDINO + facing cue | 3.533 | 2.158 | 85.8 | 4.96 | 31.45 | 4.02 |
  | CylDINO + constant field (control) | 5.673 | 3.551 | 124.1 | 3.79 | 26.21 | 9.45 |

  - **Vanilla FLAC: facing ≈ constant** (P1ORI27 vs P1ZUP27: T60 −0.4 %, C50 +0.2 %, EDT +0.8 %, R@1 +0.3 %; K=1 the
    same) ⇒ the whole vanilla gain from the cue (C50 −15 %, EDT −10 %) is the extra-parameter / fast-learning-bias
    effect of the widened, s=27, zero-init conv channels, not orientation information (which the world-frame backbone
    already has). It also costs vanilla retrieval either way (R@1 −10 %).
  - **CylDINO: facing ≫ constant** (CYLORI27 vs CYLZUP27: T60 −38 %, C50 −39 %, EDT −31 %, R@1 +31 %, R@10 +20 %;
    hallway T60 4.02 vs 9.45) ⇒ CylDINO's gain is the orientation information itself. The constant field alone is
    slightly HARMFUL on CylDINO (CYLZUP27 vs CYL: T60 +4.8 %, C50 +3.2 %, EDT +3.9 %, R@1 −7.5 %), so the bias bonus
    that vanilla FLAC enjoys does not transfer to the cylindrical backbone — the two backbones do not share the same
    "free" capacity effect, which is worth stating next to the key comparison.
  - Key controlled comparison stands: CylDINO + cue trails FLAC + cue (and FLAC + constant) by T60 +6 %, C50 +16 %,
    EDT +13 % with retrieval +2–6 %; against plain FLAC it is at parity (exp_23). Caveats: one training seed per arm;
    s=27 tuned on CylDINO; endpoint-only checkpoints for the controls (no steps curve).
