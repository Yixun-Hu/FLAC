# Params — exp_07_fa_scratch (final, per the committed launch manifest in `fa_scratch_config_identity_audit.md` §6)

## Both arms (identical by construction; asserted by `assert_arm_configs.py` v4)

| Param | Value | Provenance |
|---|---|---|
| model/training config | B-V: `FLAC_AR_BV.json` (byte-copy of release config) · B-F: `FLAC_AR_BF.json` (+`cond_method: fa_invariant`, `frame_avg_angles: [0,90,180,270]`) | audit §4 |
| dataset config | `src/configs/dataset_configs/AR/train/acousticroom_train.json` (291,210 items / 243 rooms; `augs: true`; K=8; `single_channel_ir_1`) | release |
| VAE pretransform | `weights/FLAC/VAE.safetensors` (frozen) | release |
| steps | `--max-steps 67500` (TDD round 1 flag) | ckpt-recorded budget |
| effective batch | 64 (micro×accum from M0 — same pair BOTH arms, chosen by B-F's fit: 64×1 → 32×2 → 16×4) | ckpt counter-proof + paper |
| optimizer | AdamW 5e-5, betas (0.9, 0.999), wd 1e-3 + InverseLR(1e6, 0.5, 0.99) | config = ckpt state |
| EMA | on (wrapper: beta 0.9999, power ¾) | paper + ckpt |
| precision | bf16-mixed | paper + defaults.ini |
| seed | 42 | choice (released seed unknowable) |
| checkpoints | every 2,500 steps | ckpt-recorded cadence |
| validation | `--val-dataset-config` omitted; `val_every` −1 (screens external) | choice (RNG purity) |
| workers | 6 | choice (defaults.ini) |
| grad clip / strategy | 0.0 / auto, single GPU (GPU 1) | choice-default / paper |
| ViT init | DINOv3-S16 rev `114c1379…` sha256 `4610ad75…`; `HF_HUB_OFFLINE=1`; fail-closed gate pre-launch | pinned choice |
| env records at launch | matmul/TF32 flags, `pip freeze` → launch log | manifest |

## Screens & gate

- Screening: external `eval_FLAC.py` at every 10k-step checkpoint, K=8, eval-seed 42, full unseen split (6,337/17), bf16 cond-autocast; EMA weights via default path; online via `use_ema=false` eval-config copy (to be committed before first screen).
- **Gate (pre-registered, plan §2c):** B-V@67,500 within 2σ (exp_01 5-seed σ) of released Table-1 per T60/C50/EDT at K=8 → B-F launches; else STOP and ask Yixun.
- Final protocol (post-B-F): 5 eval-seeds × K∈{1,8} both arms + H1/H2 rotation sweeps on B-F + bf16 floor re-registration (announcement 01 full-split everywhere).

## Throughput anchors (to be replaced by M0 measurements)

vanilla ~10 samples/s free-GPU → B-V ≈ 5.0 d; fa_invariant ~3 samples/s → B-F ≈ 16.7 d. M0 measures EMA-on from-scratch actuals; ETAs re-anchored at launch.
