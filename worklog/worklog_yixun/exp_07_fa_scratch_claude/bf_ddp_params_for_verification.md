# B-F from-scratch, 2-GPU DDP + SyncBN — training params (Yixun-verified 2026-07-16, LAUNCH-GATED)

**Status:** params APPROVED by Yixun 2026-07-16 with one amendment — **SyncBatchNorm is MANDATORY** ("B-F DDP: SyncBN required so BN effective batch = 64 … deliberate deviation from release (no SyncBN) to match paper BN=64"). Schedule: **wait for aug291k** to free GPU 0 (~Jul 18 ~02:00 EDT) → M1 probe → **report the measured rung → WAIT for Yixun's go** → only then train.
**GPU-1 interim (pre-approved):** screen extend ckpts 72.5k/75k/77.5k + resume the extend 77.5k→100k until GPU 0 frees.

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
| conda env | **`flac`** (Yixun 2026-07-17 "conda activate flac"; supersedes the 2026-07-16 rir2rir row). Pre-flight verified: torch 2.7.0+cu126 / PL 2.1.0 / transformers 4.57.0 / wandb 0.26.1 **version-identical to rir2rir**; import graph OK; 40/40 tests green; wandb identity yh4742 ✓. **Delta: flash_attn 2.7.4 present → DiT auto-uses FlashAttention kernels** (`src/models/transformer.py:13,429`) — exact algorithm, fp-rounding-level numerics, faster, arguably closer to the release env; P1 shares the env so the matched pair stays internally consistent. **Evals/screens stay in `rir2rir`** (metric-chain comparability with exp_01 + all screens). | Yixun 2026-07-17 |
| logger | **wandb** — account yh4742@princeton.edu (verified), project `FLAC_exp07_BF`, run `exp07_BF`; fail-closed identity gate; key self-extracted past `.bashrc`'s interactive guard | Yixun directive |
| save dir | `outputs_FLAC/exp07_BF` (wandb nests ckpts under `<save-dir>/FLAC_exp07_BF/exp07_BF/checkpoints/`) | train.py:129 |
| workers | 6 per rank (12 total; 48 cores, aug291k uses 8 — no contention) | as B-V |

## B. NEW — the DDP block (what Yixun asked to change)

| Param | Value | Note |
|---|---|---|
| GPUs | `--num-gpus 2` (`CUDA_VISIBLE_DEVICES=0,1`, 2× A6000 48 GB) | release: 1× H100 80 GB |
| strategy | `--strategy ddp_find_unused_parameters_true` — passed EXPLICITLY | REQUIRED: `defaults.ini` has `strategy="auto"`, which train.py:159–170 forwards verbatim (the `num_gpus>1 → ddp_find_unused…` fallback at train.py:172 never fires); plain DDP would crash on unused params (CFG dropout). Same value as the README multi-GPU example. |
| **SyncBatchNorm** | **MANDATORY (Yixun 2026-07-16).** PL `Trainer(sync_batchnorm=True)` via a new `--sync-batchnorm` flag (TDD-wired; ini default `false` so all existing recipes are byte-identical). Cross-rank BN sync → **effective BN batch = 32/GPU × 2 = 64 = paper**. Multi-GPU only; **fail-closed** (train.py refuses `sync_batchnorm=true` with `num_gpus < 2`; launch + M1 probe pass the same flag; launch aborts if the flag is off). ⚠️ Deliberate deviation from the release code (release trained single-GPU, no SyncBN — its BN-64 came from micro-64); we match the paper's BN statistics, not the release's code path. Covers all 20 BatchNorm2d modules in the RIR encoder. | ① rung: SyncBN batch 64 exact. ② 16×2×2: SyncBN gives 32 per BN update (2 updates/opt-step) — closer but not 64; ③ 8×2×4: 16 per update. Rung ① is the only paper-exact decomposition → strongly preferred. |
| **micro×accum** (M1 fit probe) | **SINGLE compliant rung: `--batch-size 32 --accum-batches 1`** → 32/GPU × 2 × 1 = 64, `--sync-batchnorm true`. *(Review correction: gradient accumulation never contributes to BN statistics — SyncBN batch = micro/GPU × world_size only — so the former ②16×2×2/③8×2×4 rungs would silently violate BN=64 and are ELIMINATED.)* | If 32×2×1 does not fit under the plain allocator, the probe STOPS and reports; escalation options (expandable_segments allocator / revisiting the mandate) are Yixun's call. Launch script enforces MB×2==64 fail-closed. |
| allocator | plain first; `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` ONLY if no plain rung fits (numerics-neutral, would be disclosed in manifest) | targets the observed fragmentation |
| steps/epoch | 291,210 ÷ 64 = 4,550 opt-steps/epoch → epoch 14 at 67.5k — scheduler/step cadence **identical to release & B-V** | ✓ |
| est. duration | ~5 d (2× the single-GPU ~9.6 d estimate at ~90 % DDP scaling; re-anchored from the first ~200 steps at launch) | |

## C. Honest deltas (pre-registered disclosure)

1. **vs release:** 2×A6000 DDP+SyncBN vs 1×H100 single-GPU-no-SyncBN → gradient all-reduce gives the same *mean* gradient (fp summation order differs, not bit-equal); **SyncBN makes BN statistics batch-64 like the paper** (at rung ①) while the release achieved BN-64 via micro-64 on one card — same statistics, different mechanism (deliberate, Yixun-mandated); SyncBN'd running stats are rank-consistent (saved once); DistributedSampler shards (same 291,210 samples/epoch globally, different order).
2. **vs the 8×8 B-V control (CONFIRMED by Yixun):** "Any B-V/P1 compared to this B-F must use the same DDP+SyncBN recipe" → **P1 = B-V at the IDENTICAL DDP+SyncBN recipe** (same rung, same flags, wandb) — micro-parity causal test AND B-F's clean matched control in one run. The existing 8×8 B-V demotes to a corroborating row; `plan_bv_parity.md` gets a formal amendment before P1 launches.
3. **Logger wandb vs none (B-V phase 1):** observation-only (`wandb.watch` reads gradients; no RNG consumption) — noted for completeness.

## D. Execution plan (per Yixun's approval terms)

0. **Interim (GPU 1, pre-approved):** screen extend ckpts 72.5k/75k/77.5k; resume extend 77.5k→100k (done ~Jul 17 ~17:00, before GPU 0 frees). **Prep:** coder subagent wires `--sync-batchnorm` (TDD, `src/tests/`), Codex reviews wiring + updated launch/probe scripts, committed before any run.
1. **M1 DDP+SyncBN fit probe** (~30 min, both GPUs, after aug291k ends ~Jul 18 ~02:00): 15 opt steps per rung ①→②→③ (all `--sync-batchnorm true`), per-GPU 1-s VRAM samplers, CUDA-OOM-only descent (other failures hard-abort), FIT = rc 0 + "max_steps=15 reached" + finite loss.
2. **REPORT the measured rung + per-GPU VRAM + re-anchored ETA → WAIT for Yixun's explicit go** (his instruction: "after the probe, report the rung first, then wait for go"). NO training before that go.
3. On go: launch `bf_scratch_launch.sh` (DDP+SyncBN; both-GPU-free guard; wandb + pin gates) — teed `*_BF_train.log`, wandb live; screens per 10k ckpt; ckpt monitor; ETA re-anchored in the launch report.
