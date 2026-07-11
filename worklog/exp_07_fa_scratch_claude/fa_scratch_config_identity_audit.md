# Config-identity audit — exp_07 (B-F ≡ B-V ≡ FLAC-as-released/described)

**Author:** Fable 5 (main session — planning/analysis seat restored from Opus 4.8 per Yixun's `/model` switch, 2026-07-10) · **Date:** 2026-07-10
**Commissioned by Yixun:** "before going [exp_07], I need to confirm that the exp_07 B-F arm has the same data, training and model configuration as B-V, which should be the same as FLAC as described inside FLAC_pdf.md."

**Evidence sources (all four, cross-checked):** (1) the paper text `FLAC_pdf.md` (§5.1, §B.1); (2) the repo configs (`FLAC_AR.json`, `acousticroom_train.json`, `defaults.ini`) + current-branch code; (3) **the released `weights/FLAC/FLAC.ckpt`'s own internal records** (embedded `model_config`, optimizer/scheduler state, all loop-counter phases — probed by `probe_released_ckpt.py` v2, canonical log `fa_scratch_2026-07-10_23:59:06_ckpt_probe_v2.log`; the earlier `..._23:35:28_ckpt_probe.log` is superseded, its config-diff section having been produced inline — provenance deviation recorded in `fa_scratch_command.md`); (4) arm instantiation asserts (`assert_arm_configs.py` v3 — v2 checks + the fail-closed ViT pin gate; canonical log `fa_scratch_2026-07-11_00:11:31_arm_asserts_v3.log`).

---

## Verdict (one screen)

| Claim | Verdict |
|---|---|
| **B-F ≡ B-V** (identical data/model/training; conditioning method the only delta) | **PROVEN** — byte-copy + exactly 2 `training` keys; instantiation asserts v2: identical 64.50M/753-tensor architecture, **init-identical under seed 42 (state-dict sha256 match)**, factory wiring exercised via `configure_optimizers()`; `fa_invariant` is forward-time-only (`src/training/diffusion.py:206-221`) |
| **B-V ≡ released FLAC (config level)** | **Config-identical up to no-op keys, with ONE unresolved training-relevant caveat**: the trainable DINOv3's initializer provenance (§3 — authors loaded a local snapshot of unknowable revision; our pin: rev `114c1379…`, sha256 `4610ad75…`, identical across arms). All other diffs are metrics/demo/logging or behavior-identical defaults (itemized §3) |
| **B-V ≡ paper description** | **VERIFIED on every stated number** (§2). Ckpt loop counters prove **effective batch 64** (conditional on the shipped split + `drop_last`, `src/data/dataset.py:405`); the **64-micro × 1-GPU decomposition is paper-specified**, not counter-provable (2×32×1 yields identical counters) (§1) |
| Residual unknowables | 5, enumerated §5 — none config-addressable (seed of released run; authors' data/env snapshot; ViT initializer revision; checkpoint selection; launch settings not recorded in the ckpt) |
| **Plan correction** | original recipe is **effective batch 64** (paper: micro 64 × 1 H100; accum=1 counter-proven) — the exp_07 plan's "effective batch 128" traced to the README example, which is **incompatible with the released ckpt** (§1); wall-clock roughly **halves** (§6) |

---

## 1. The released checkpoint is a complete training record

`FLAC.ckpt` (PL 2.1.0 — matches `pyproject.toml` pin) records `epoch=14`, `global_step=67,500`. **Counter phases matter** (the checkpoint callback fires after `processed` increments, before `completed` — so `completed` lags by 1): like-for-like `processed` counters give micro `67,500` / optimizer `67,500` → **accumulation = 1.000000 exactly**; per-epoch: (67,500 − 3,800 current-epoch) / 14 completed epochs = **4,550.0 steps/epoch = floor(291,210 / 64)** with `drop_last=True` (`src/data/dataset.py:405`), shuffle on (`:344`).

**What the counters prove and don't:** effective batch 128 ⇒ 2,275 steps/epoch ⇒ epoch ≈ 29 at step 67,500 — incompatible; accumulation 2 ⇒ optimizer ≠ micro counters — incompatible. So **global effective batch 64 with accum 1 is counter-proven** (conditional on the shipped 291,210-item split). The counters **cannot** distinguish the decomposition: `1 GPU × micro 64` and `2 GPUs × micro 32` yield identical per-rank counters — **"batch 64 on a single H100" rests on the paper text** (§B.1), which we adopt.

**README adjudication (provenance of the old eff-128 assumption):** `README.md:114`'s example command (`--batch-size 32 --accum-batches 2 --num-gpus 2` = eff 128, workers 8, `--val-every 2500`, ddp strategy) is **incompatible with the released ckpt on two independent counters** (epoch-at-step and accum ratio). It is a convenience example, not the released recipe. Paper + `defaults.ini` + ckpt agree: eff 64, accum 1.

Optimizer state: AdamW `initial_lr 5e-5, betas (0.9, 0.999), eps 1e-8, wd 1e-3`. Scheduler state: `InverseLR(inv_gamma=1e6, power=0.5, warmup=0.99)`, `_last_lr = 4.839339184958273e-5` — analytic check with the **full formula** (`src/training/utils.py:56`): `(1 − 0.99^{67501}) · 5e-5 · (1 + 67500/10^6)^{-0.5}` — the warmup term is `1 − 2.35e-295` ≡ 1.0 in float64 → `4.839339184958273e-5`, **exact to the last digit** ✓. (The warmup term is live from step 0: the freshly constructed schedule starts at `0.01 × 5e-5 = 5e-7` — verified by the arm asserts.) EMA present (212 keys, `step=67,500`). `ModelCheckpoint(every_n_train_steps=2500)` → 67,500 = 27 × 2,500: the release is a **periodic checkpoint** (mid-epoch-15), consistent with exp_06's surviving "source-side checkpoint selection" hypothesis and making **67,500 steps the pre-registered parity budget**.

## 2. B-V vs the paper text (§B.1, §5.1)

| Knob | Paper | Our B-V | Match |
|---|---|---|---|
| DiT | 12 blocks, 8 heads, width 256 | `depth 12, num_heads 8, embed_dim 256` | ✓ |
| Objective | flow matching; α∼N(−1.2, 2²), t=σ(−α) | `rectified_flow`; `log_snr` sampler, defaults (−1.2, 2.0) (`diffusion.py:103-105`) | ✓ |
| Optimizer | lr 5e-5, AdamW | `AdamW 5e-5` (+ betas/wd from ckpt, paper-silent) | ✓ |
| Batch | **64, single H100** | `defaults.ini: batch_size 64, accum 1, num_gpus 1` | ✓ (eff-64/accum-1 ckpt-proven; 64×1-GPU decomposition paper-specified, §1) |
| EMA / precision | EMA; BF16 | `use_ema true` (wrapper: beta .9999, power ¾); `precision bf16-mixed` | ✓ |
| Augmentation | time shift ≤10 samples; pink noise SNR 40–60 dB | `RandomTimeShift(10, p=.5) + AddNoise(pink, 40-60, p=.5)` via `augs:true` (`dataset.py:191-195`) | ✓ (p=0.5 paper-silent) |
| Dataset total | 260 rooms, "over 300k" RIRs, 22,050 Hz | `all_data.json`: **302,671 items / 260 rooms**; sample_rate 22050 | ✓ |
| Train split | 243 seen rooms | `data/AR/train.json`: **291,210 items / 243 rooms / 10 categories**; `single_channel_ir_1` | ✓ (291,210 is the *training subset* of the >300k total) |
| Seen eval | 6,217 instances / 131 rooms | `seen_eval.json`: 6,217 / 131 | ✓ exact |
| Unseen eval | paper text says **5,244** | shipped `unseen_eval.json`: **6,337 / 17 rooms** | ⚠ paper-text discrepancy — the *shipped* split is authoritative: exp_01 reproduced released Table-1 numbers on 6,337, and announcement 01 pins it. No action. |

## 3. B-V vs the ckpt-embedded `model_config` (the strongest identity check)

Diff of `ck["model_config"]` vs repo `FLAC_AR.json` — **every training-relevant key identical** (pretransform, conditioning dims, DiT block, objective, cfg_dropout 0.1, mask_padding, timestep_sampler, use_ema, optimizer+scheduler configs, sample_size/rate). The complete list of differences:

| Diff | Class | Assessment |
|---|---|---|
| ViT `hf_model_name_or_path`: `./Models/dinov3-vits16-pretrain-lvd1689m` (ckpt) vs `facebook/dinov3-vits16-pretrain-lvd1689m` (repo) | **training-relevant caveat — NOT a proven no-op** | The ViT is *trainable* (`freeze:false`), so its **initial weights are lineage-relevant**. Name-identical model, but the authors' local-snapshot revision is unknowable from the ckpt. Mitigation: our initializer is pinned — revision `114c1379950215c8b35dfcd4e90a5c251dde0d32`, `model.safetensors` sha256 `4610ad75edef83e75afdebf162d148dc628045ea6cbb83d67d4708c709c4f91d` — and **identical across both arms** (matched comparison unaffected; the absolute B-V-vs-Table-1 read carries this caveat, §5.iii) |
| repo adds `from_scratch:false, img_h:256, img_w:512` | behavior-identical defaults | `from_scratch` defaults False (`conditioners.py:374`); `img_h/w` are read only in the **SimpleViT branch taken when no HF path is supplied** (`:403-404`) — never reached by these configs |
| ckpt has `training.structured_noise:false, structured_noise_beta:1.0` | dead option | **false** in ckpt; string absent from our entire codebase |
| ckpt has `training.demo.*` | authors'-internal demo logging | not consumed by our train path (grep empty) |
| metrics blocks differ (ckpt: DRR/RFD/T60band/l1 flags, no AGREE; repo: FD/retrieval + AGREE ckpt) | eval/logging only | metrics never touch gradients |

## 4. B-F ≡ B-V

- `FLAC_AR_BV.json` = byte-copy of repo `FLAC_AR.json`; `FLAC_AR_BF.json` differs by **exactly** `training.cond_method="fa_invariant"` + `training.frame_avg_angles=[0,90,180,270]` (explicit for provenance; equals `DEFAULT_FRAME_ANGLES`). JSON-diff in the probe log.
- Instantiation asserts **v2** (both arms, CPU): cond_method wiring correct; wrapper-level `cfg_dropout_prob`/`optimizer_configs`/sampler asserts (factory output, not JSON re-reads); a real `configure_optimizers()` call whose returned objects are field-checked — `AdamW(initial_lr 5e-5, betas (0.9,0.999), wd 1e-3)` + `InverseLR(1e6, 0.5, 0.99)` at `interval=step`, with the step-0 live lr `5e-7 = (1−0.99¹)·5e-5` confirming the warmup schedule is active from construction exactly as in the released run; **identical parameter-name set (753 tensors) and count (64.50M)**; and **seeded init-identity — under seed 42 both arms' full state_dicts hash identically (sha256 `44a2f6aadd7d2180…`)**. The FA path changes only how the conditioner is *called* per forward (one full pass + |G|−1 ViT-only passes; cylindrical re-parameterization of the same pose inputs), never the module tree (`factory.py:75-76`, `diffusion.py:206-221`).
- Same dataset config file, same split json, same metadata module, same `augs`, same seed policy (42), same trainer flags → **data/training identity by construction**; the dataloader consumes metadata identically in both arms (FA transforms happen inside the training step).

## 5. Residual unknowables (not config-addressable — pre-registered as caveats)

i. **Training seed of the released run** — not recorded in the ckpt; ours pinned at 42 for both arms (matched comparison unaffected; absolute-parity read carries this caveat).
ii. **Authors' exact data/env snapshot** — exp_06's surviving lineage explanation; exactly what exp_07's B-V-vs-Table-1 gate measures.
iii. **DINOv3 initializer revision** (training-relevant, §3) — authors trained from a local snapshot of unknowable revision; both arms share our pinned initializer (rev `114c1379…`, sha256 `4610ad75…`) → matched comparison unaffected; absolute-parity read carries it.
iv. **Checkpoint selection** — the release is one periodic checkpoint (§1); our 2,500-cadence checkpoints let us mirror any selection curve.
v. **Launch settings not recorded in the ckpt** — micro×GPU decomposition, `num_workers`, validation cadence, dataloader RNG stream details. These are **choices, not recoverable identity** — pinned explicitly in the §6 manifest and held identical across arms.

## 6. Corrected budget + launch identity (supersedes plan §2 table)

67,500 steps × 64 samples = **4.32M samples/arm** (half the plan's eff-128 assumption):

| Arm | Throughput anchor (free GPU) | Wall-clock |
|---|---|---|
| B-V | ~10 samples/s | **~5.0 d** |
| B-F | ~3 samples/s (4× conditioner) | **~16.7 d** |
| Sequential total | | **~21.7 d** (was ~40 d) — hybrid (c) gate after B-V week unchanged |

### Launch manifest (explicit; every setting a *choice* is labeled as such; **identical across arms by construction**)

| Setting | Value | Status vs released run |
|---|---|---|
| `--max-steps` | 67,500 | ckpt-recorded budget (**pending the TDD round** — `train.py:161` hardcodes 1e6) |
| effective batch / accum | 64 / micro×accum from M0 (see rule below) | eff-64+accum-1 ckpt-proven; decomposition paper-specified |
| `--precision` | `bf16-mixed` | paper + defaults.ini |
| `--seed` | 42 | **choice** (released seed unknowable) |
| `--checkpoint-every` | 2,500 | ckpt-recorded cadence (overrides defaults.ini's 10,000) |
| `--num-workers` | 6 (defaults.ini) | **choice** — README example says 8; workers shift per-worker RNG streams (context-RIR `np.random.choice`, `AR_md.py:104`; stochastic augs) |
| `--val-every` / `--val-dataset-config` | −1 / **omitted entirely** (no val loader is even built — `train.py:52-58` makes it conditional; screens are external `eval_FLAC.py` jobs per plan §3) | **choice** — README says 2,500 + seeneval config, but in-train validation consumes RNG (noise sampling, `diffusion.py:415`), perturbing the training stream; external screens keep the training RNG sequence pure |
| shuffle / drop_last | True / True (hardcoded, `dataset.py:344,405`) | matches ckpt arithmetic (§1) |
| `--gradient-clip-val` | 0.0 (defaults.ini) | **choice** (not ckpt-recoverable; defaults.ini value adopted) |
| strategy / nodes | auto / 1 (single GPU, no DDP) | **choice** consistent with the paper's single-GPU statement (not ckpt-recoverable) |
| matmul/TF32 policy | record `torch.get_float32_matmul_precision()` + TF32 flags in the launch log | **environment record** |
| deps | torch 2.7.0, PL 2.1.0, transformers 4.57.0 (pyproject pins; PL version matches the ckpt's recorded 2.1.0) + **full `pip freeze` captured in the launch log** | **environment record** (authors' full env unknowable; PL ckpt-consistent) |
| ViT initializer | rev `114c1379…`, sha256 `4610ad75…` (§3) — **fail-closed**: `assert_arm_configs.py::assert_vit_pin()` refuses launch unless the cache holds exactly this snapshot; launch env sets `HF_HUB_OFFLINE=1` | **pinned choice** (authors' revision unknowable) |
| hardware | A6000-48GB | **deviation** — authors: H100-80GB |

**Micro×accum rule (review fix):** the M0 fit probes measure the max micro-batch for **B-F** (the more constrained arm: 4× ViT forwards); then **both arms run the identical micro×accum pair** chosen from B-F's constraint — `64×1` if B-F fits it, else `32×2`, else `16×4` — never an asymmetric pair (an asymmetric configuration would be a separately-declared ablation, not the primary comparison). If the common pair ≠ `64×1`, the **B-V-vs-released lineage read** carries a micro-batch deviation caveat; the **B-F-vs-B-V comparison does not** (arms identical).

**What a micro-batch deviation actually changes** (corrected per review, with one rebuttal): (a) **BatchNorm running-stat updates** — the review claimed no BatchNorm exists; **rebutted with evidence**: the instantiated model contains **20 `BatchNorm2d` modules** under `conditioner.conditioners.context_audio.net.cnn.*` (torchvision resnet18 inside `AudioResNet18`, `conditioners.py:19/37/148` — invisible to source-grep because the classes live in torchvision; the same modules exp_05/exp_08's `FreezeBN` discovered and froze). Their per-micro-batch statistics differ with micro size during from-scratch training. Plus the review's additions: (b) gradient-accumulation summation order (bf16 reduction-order effects), (c) per-worker/stochastic-aug RNG sequencing, (d) data order under resume.

## 7. Yixun's in-flight `FLAC_vanilla291k` (GPU 0) — can it serve as B-V?

**No — close but not certified.** Same training-relevant model config (its `FLAC_AR_noAGREE.json` differs from `FLAC_AR.json` only in metrics: `eval_FD/eval_retrieval:false`, no AGREE ckpt) ✓; eff-batch 64 ✓; checkpoint cadence 2,500 ✓. But: micro 16 × accum 4 (released: 64×1 — different BN micro-stats/data order), **`folder_name: single_channel_ir` ≠ repo `single_channel_ir_1`** (a genuine data-provenance difference), third-party copies of `train.json`/`AR_md.py` (content unverified from this repo), 291k-step target. **exp_07 runs its own B-V.** Opportunistic bonus (optional, ~25 min GPU): screen that run's step-67,500 checkpoint with our eval as a *corroborating* approximate lineage read — never a substitute.

## 8. Pre-launch checklist (all still open)

1. `--max-steps` TDD round (Opus 4.8 max codes; **Codex `gpt-5.6-sol` xhigh reviews** — reviewer model updated per Yixun 2026-07-10, codex-cli upgraded 0.142.5 → 0.144.1).
2. M0 EMA-on fit/throughput probes, both arms → fixes micro×accum + real ETAs.
3. A free GPU (both currently occupied by Yixun's runs).
4. Screening eval-config copies (`use_ema=false` variant) per plan §3.
5. Consolidated Codex review of this audit's probe scripts (`probe_released_ckpt.py`, `assert_arm_configs.py`, arm configs) — universal-review rule.
