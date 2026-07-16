# B-F from-scratch, 2-GPU DDP — training params for Yixun verification (PRE-LAUNCH)

**Status:** AWAITING Yixun verification (his directive 2026-07-16: 2-GPU DDP so VRAM/batch matches the paper setting; params verified before launch). Nothing launches until approved.
**Blocker to schedule around:** GPU 0 currently runs Yixun's other-session `FLAC_aug291k` training (PID 1284685, step 42,500, ~3h39m per 2,500 steps → done ~Jul 18 ~02:00 EDT if its target is 67,500).

## A. Fixed — identical to the release recipe / B-V phase-1 manifest

| Param | Value | Provenance |
|---|---|---|
| model config | `FLAC_AR_BF.json` = **byte-copy of release `FLAC_AR.json`** + `cond_method: "fa_invariant"` + `frame_avg_angles: [0, 90, 180, 270]` (the ONLY two key diffs; init-identical to B-V under seed 42 — state_dict sha256 asserted fail-closed at launch) | audit §4; `assert_arm_configs.py` |
| dataset config | `src/configs/dataset_configs/AR/train/acousticroom_train.json` — 291,210 items / 243 rooms, K=8 context, `augs: true`, `single_channel_ir_1` | release |
| VAE pretransform | `weights/FLAC/VAE.safetensors`, frozen | release |
| total optimizer steps | `--max-steps 67500` | ckpt-recorded release budget |
| **effective (global) batch** | **64** | ckpt counter-proof + paper |
| optimizer | AdamW lr 5e-5, betas (0.9, 0.999), weight-decay 1e-3 | config = ckpt state |
| LR schedule | InverseLR(inv_gamma 1e6, power 0.5, final_lr_ratio 0.99) — lr ≈ 4.84e-5 at 67.5k | config = ckpt state |
| EMA | on (wrapper: beta 0.9999, power ¾) | paper + ckpt |
| precision | bf16-mixed (from `defaults.ini`; no explicit flag — exact flag identity with B-V) | paper + defaults |
| seed | 42 | same as B-V (released seed unknowable) |
| checkpoints | every 2,500 steps | release cadence |
| validation | off (`--val-dataset-config` omitted); screens external per 10k ckpt (EMA+online, K=8 s42, full 6,337-item unseen split) | phase-1 protocol |
| grad clip | 0.0 | default (as B-V) |
| DINOv3 init | pinned rev `114c1379…`, sha256 `4610ad75…`; `HF_HUB_OFFLINE=1` on gate AND training | pinned choice |
| conda env | **`rir2rir`** (NOT `flac`) — pip-freeze manifest identity with B-V phase 1 | flagged 2026-07-16 |
| logger | **wandb** — account yh4742@princeton.edu (verified), project `FLAC_exp07_BF`, run `exp07_BF`; fail-closed identity gate; key self-extracted past `.bashrc`'s interactive guard | Yixun directive |
| save dir | `outputs_FLAC/exp07_BF` (wandb nests ckpts under `<save-dir>/FLAC_exp07_BF/exp07_BF/checkpoints/`) | train.py:129 |
| workers | 6 per rank (12 total; 48 cores, aug291k uses 8 — no contention) | as B-V |

## B. NEW — the DDP block (what Yixun asked to change)

| Param | Value | Note |
|---|---|---|
| GPUs | `--num-gpus 2` (`CUDA_VISIBLE_DEVICES=0,1`, 2× A6000 48 GB) | release: 1× H100 80 GB |
| strategy | `--strategy ddp_find_unused_parameters_true` — passed EXPLICITLY | REQUIRED: `defaults.ini` has `strategy="auto"`, which train.py:159–170 forwards verbatim (the `num_gpus>1 → ddp_find_unused…` fallback at train.py:172 never fires); plain DDP would crash on unused params (CFG dropout). Same value as the README multi-GPU example. |
| **micro×accum ladder** (M1 fit probe, first-fit-wins) | **①** `--batch-size 32 --accum-batches 1` → 32/GPU × 2 × 1 = 64 · **②** `16 × 2 × 2` = 64 · **③** `8 × 2 × 4` = 64 | ① is the release-closest decomposition (accum 1, one optimizer step per batch, BN batch 32/GPU); single-GPU B-F OOM'd at micro 32/16 with ~8–9 GB *fragmentation* waste, so DDP fit is genuinely open → probe decides |
| allocator | plain first; `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` ONLY if no plain rung fits (numerics-neutral, would be disclosed in manifest) | targets the observed fragmentation |
| steps/epoch | 291,210 ÷ 64 = 4,550 opt-steps/epoch → epoch 14 at 67.5k — scheduler/step cadence **identical to release & B-V** | ✓ |
| est. duration | ~5 d (2× the single-GPU ~9.6 d estimate at ~90 % DDP scaling; re-anchored from the first ~200 steps at launch) | |

## C. Honest deltas (pre-registered disclosure)

1. **vs release:** 2×A6000 DDP vs 1×H100 single-GPU → gradient all-reduce gives the same *mean* gradient (fp summation order differs, not bit-equal); **BN batch = micro/GPU (32 at rung ①) vs release 64** — no SyncBatchNorm (release code has none; adding it would be a bigger deviation); per-rank BN running stats, rank-0's saved; DistributedSampler shards (same 291,210 samples/epoch globally, different order).
2. **vs the 8×8 B-V control:** B-F-DDP differs from B-V-8×8 in micro/BN-batch + DDP + logger — **the single-delta matched-arm design is broken by this reorder.** Fix (proposed): **repoint P1 to run B-V at the IDENTICAL DDP recipe** (same rung, same 2-GPU strategy, wandb) — P1 then serves double duty: the micro-parity causal test AND the clean matched control for B-F. The existing 8×8 B-V remains as the endpoint-anchored corroborating row.
3. **Logger wandb vs none (B-V phase 1):** observation-only (`wandb.watch` reads gradients; no RNG consumption) — noted for completeness.

## D. Execution plan after Yixun's OK

1. **M1 DDP fit probe** (~20–30 min, both GPUs): 15 opt steps per rung, ladder ①→②→③, per-GPU 1-s VRAM samplers, CUDA-OOM-only descent (any other failure hard-aborts), FIT = rc 0 + "max_steps=15 reached" + finite loss. Winning rung is pinned and reported.
2. **Launch** `bf_scratch_launch.sh` (updated to DDP; GPU-free guard now checks BOTH GPUs; wandb + pin gates unchanged) — teed `*_BF_train.log`, wandb live.
3. Screens per 10k ckpt (recursive find handles wandb nesting); ckpt-arrival monitor; ETA re-anchored and reported.
