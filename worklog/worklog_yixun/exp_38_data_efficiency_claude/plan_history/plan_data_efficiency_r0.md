# plan_data_efficiency — exp_38: YAW and OrbitRIR (B-F) data-efficiency curve at 25 / 50 / 75 % of AR

**Author:** Claude Opus 5.5 (Coder/Planner subagent, max effort) · **Date:** 2026-11-07 · **Status:** r0 — before Codex plan review r1 · **Branch / worktree:** `exp-38-data-efficiency` in `~/codespace/exp-38-data-efficiency`, cut from `localization-exp` HEAD `2b42468c8f429b1d93da93b3e4ddb7bd6f8f1932` (= `origin/localization-exp` at 20:24 EST Nov 7). `main` is never touched.

Every factual claim below carries its evidence (file:line, commit, checkpoint field or measured number). "Verified" = re-checked by the Planner in this session; "inferred" = argued from records, not directly observable.

---

## 1. Question and estimand

**Question (Yixun, `data_efficiency_yixun_query.md`):** how do Yaw-Augmented FLAC (YAW) and OrbitRIR (B-F, `fa_invariant` C4 frame averaging) degrade when trained on 25 / 50 / 75 % of the AcousticRooms (AR) training split, compared with vanilla FLAC (P1), whose curve already exists (exp_14 `dc_van_f025/f050/f075` + P1@40k)?

**Estimand.** For model `m ∈ {YAW, B-F}`, fraction `f ∈ {0.25, 0.50, 0.75}`, context size `K ∈ {8 (headline), 1}`: the 40,000-step EMA endpoint of ONE training run (seed 42) evaluated on the full unseen split (`data/AR/unseen_eval.json`, 6,337 items / 17 rooms, announcement 01), metrics T60 %, C50 dB, EDT ms (plus FD, R@k reported), **split-level item-weighted** aggregation (the convention of every cell in Table 3(b), §9.4), mean ± sample SD over eval seeds 42–46. Each fraction's training recipe = that model's own 100 % recipe with the dataset config swapped for exp_14's subset config — nothing else.

**Scope of inference (mandatory in results/analysis):** one training seed per cell; the 5 eval seeds measure diffusion/eval variability only, not training-run variability (exp_14's recorded step-to-step band on this stack is ≈ ±0.5 T60, `data_curve_results.md:3`). Fixed-compute estimand: 40k × 64 samples everywhere, so effective epochs are 34.8 / 17.6 / 11.7 / 8.8 at 25 / 50 / 75 / 100 % (exp_14 `data_curve_params_set_up.md`).

## 2. What already exists (reused, never re-run)

| Curve point | Source | K=8 T60 / C50 / EDT (item-weighted, mean ± SD, s42–46) | Evaluator / backend |
|---|---|---|---|
| FLAC 25 % | exp_14 `dc_van_f025` (ckpt sha `aa1a2cd6…`) | 10.252±0.012 / 1.048±0.002 / 42.914±0.058 | exp_14 evaluator (`exp-14-data-curve`, `66901577` lineage), SDPA (train log prints `flash_attn not installed`) |
| FLAC 50 % | `dc_van_f050` | 9.450±0.010 / 1.044±0.002 / 41.587±0.064 | same |
| FLAC 75 % | `dc_van_f075` | 8.913±0.009 / 1.104±0.002 / 41.636±0.065 | same |
| FLAC 100 % | P1@40k `c4c67882…`, exp_25 P1-pin cells | 8.993±0.011 / 1.009±0.004 / 40.650±0.103 | exp_25 pin `d6ca307a`, SDPA |
| YAW 100 % | YAW@40k `ac1f2603…`, exp_25 YAW-pin cells | 7.965±0.015 / 1.013±0.002 / 39.923±0.058 | exp_25 pin `d6ca307a`, SDPA |
| B-F 100 % | B-F@40k `5319feb4…`, exp_35 BF-pin T cells | 8.202±0.017 / 0.978±0.002 / 38.794±0.074 | exp_35 pin H35 `5b065801`, SDPA (no-flash overlay) |

Recomputed by the Planner from the raw JSONs (dc_van: `localization-exp:outputs_FLAC/data_curve_import/dc_van_f0XX/*_K8_s4[2-6].json`; P1/YAW: `outputs_FLAC/exp25_depth_canon/{P1,YAW}/*exp25_*_T_S40000_K8_s4[2-6].json`; B-F: `~/codespace/exp-35-orbitrir-yaw/outputs_FLAC/exp25_depth_canon/BF/*exp35_BF_T_S40000_K8_s4[2-6]_fa_invariant_a4.json`); they reproduce Yixun's Table 3(b) DINOv3 rows (25 % 42.914/1.048/10.253, …, 100 % 40.650/1.009/8.993; the 75 % EDT 41.640 is the 2-dp rounding 41.64 of 41.6362). K=1 values for every row are in the same files (e.g. dc_van_f025 K1 11.463/1.117/45.036).

**Evaluator identity.** `git diff --stat 5b065801 2b42468c -- eval_FLAC.py src/{models,data,inference,metrics,training}` is empty and `git diff d6ca307a 2b42468c` over the same paths adds only `overlays/no_flash/` (verified): the evaluator + model code at our base is byte-identical to the one that produced the P1/YAW/B-F 100 % anchors. dc_van used the older exp_14 evaluator; §9.3's E0 replay gate bridges that.

## 3. The 100 % recipes (verified from the checkpoints' embedded `model_config` and the records)

All three were loaded on CPU in this session (`torch.load(..., weights_only=False)` on the exp-25 cache copies under `~/codespace/exp-36-run/outputs_FLAC/exp25_depth_canon/.ckpt_cache/`): each embeds `model_config`, `global_step == 40000`, `epoch == 8`, scheduler `InverseLR(inv_gamma 1e6, power 0.5, warmup 0.99)`, `last_epoch 40000`, `_last_lr 4.9029e-5`.

| | P1 (FLAC) | YAW (Yaw-Aug FLAC) | B-F (OrbitRIR) |
|---|---|---|---|
| ckpt sha256 | `c4c678826cdd…` | `ac1f26034e4f…` | `5319feb4af87…` |
| embedded config == | `exp_07…/FLAC_AR_BVp1.json` (sha `733ca52b…`) | `exp_15…/FLAC_AR_YAWAUG.json` (sha `b14caae2…`) = BVp1 + `training.yaw_aug {enabled true, img_w 512, seed 42}` | `exp_07…/FLAC_AR_BF.json` (sha `3e677204…`) = release config + `cond_method fa_invariant`, `frame_avg_angles [0,90,180,270]` |
| ViT grad-ckpt (both ViT conditioners) | true | true | true |
| experiment | exp_07 P1 | **exp_17 yaw_aug_a6000** (NOT exp_15) | exp_07 B-F |
| hardware / rung | 2×A6000, 32/GPU × 2 × accum 1, DDP `ddp_find_unused_parameters_true`, SyncBN (BN 64) | same (exp_17 params: "32/GPU × 2 GPUs × accum 1 → eff/BN batch 64 (PINNED)") | same (`bf_scratch_launch.sh`: `MB=32 ACC=1` literal pin, `--num-gpus 2 --sync-batchnorm true`) |
| precision | bf16-mixed (ini default) | `--precision bf16-mixed` | bf16-mixed (ini default) |
| seed / steps / ckpt cadence | 42 / 67.5k budget, 40k ckpt used / 2,500 | 42 / 40k / 2,500 | 42 / 67.5k budget, stopped at 40,249, 40k ckpt used / 2,500 |
| lineage | resumed once at 32.5k (legs `e50d098` → `1b56e7a`; exp_25 `depth_canon_training_path_diff_audit.md`) | monolithic, 21 h 44 m exclusive on both cards (`42cbddaf`) | monolithic (`bf_stop_record_and_p1_amendment.md`) |
| training commit | `e50d098` / `1b56e7a` (check-equivariance-necessity) | the `exp-17-yawaug-a6000` worktree (removed), Aug 15 2026; inferred training closure = `41aa31dc` (no closure commit between `41aa31dc` and Oct's exp_25 commits, `git log 41aa31dc..localization-exp -- train.py defaults.ini src/…`) | `f59f5a4` era (Jul 19; `commits_fa_scratch.md`) |
| FA chunk plan (ann. 06) | n/a | n/a | **legacy per-angle loop** (trained before the batched orbit `1479304`/`8094d60`, Aug 6) |
| attention backend (DiT) | flash-attn (env `flac` from 6596121 on; inferred) | flash-attn (inferred: the exp_17 eval logs of Aug 16 lack `flash_attn not installed`) | flash-attn (`6596121`, "flash-attn kernel delta disclosed") |
| logger | wandb | wandb | wandb |

Notes. (i) The max-steps budget (67.5k) does not alter the trajectory up to 40k: `InverseLR` depends only on `inv_gamma/power/warmup` and the step (`src/training/utils.py`), nothing reads `max_steps` — so `--max-steps 40000` is the same schedule, as exp_14's dc_van runs already assumed. (ii) **Correction to the brief:** the table's YAW row is exp_17's matched 2×A6000 arm (`ac1f2603…`, PROVENANCE `/media/diskstation/yixunhu/FLAC/checkpoints/ar_40k_endpoints/PROVENANCE.md`, exp_25 comparator ledger). exp_15's neuronic YAWAUG (8×L40 × 8, `model_comparison.md` rows "yaw-aug vanilla YAWAUG @40k (exp_15)") is a different recipe and is NOT the 100 % point. (iii) dc_van (exp_14) trained with P1's config byte-for-byte (`FLAC_AR_exp14_vanS.json` sha `733ca52b…`) and P1's rung, but with SDPA (Sept env, no flash-attn), at `66901577`/`f609976d`/`6a4b8b08`, `--logger none`.

## 4. How exp_14 built and applied the subsets (verified)

**Built** (exp_14 plan §3 + Amendment 1; `src/tools/make_ar_train_subsets.py` on `exp-14-data-curve`): nested per-room stratified random subsets of `data/AR/train.json` (291,210 items / 243 rooms, sha `aa4e52d6…`), seed 2026, eligibility `sampler_faithful_S00int_v1` (a target is kept only if `AR_md.py`'s `S00{int}` name rebuild can find ≥ 1 in-split context; `S010` is unreachable), top-ups to keep every target's pool non-empty; manifest `data/AR/train_frac_manifest_s2026.json` (sha `553534ea…`):

| tag | split file | sha256 | items | rooms | ctx histogram 0 / 1–7 / ≥ 8 |
|---|---|---|---|---|---|
| 025 | `data/AR/train_frac025_s2026.json` | `ce33f34f…` | 73,473 | 243 | 0 / 16,890 / 56,583 |
| 050 | `data/AR/train_frac050_s2026.json` | `bf330846…` | 145,665 | 243 | 0 / 14,303 / 131,362 |
| 075 | `data/AR/train_frac075_s2026.json` | `57388ce8…` | 218,463 | 243 | 0 / 17,430 / 201,033 |

**Applied** (exp_14 `data_curve_params_set_up.md`; dc_van at-launch records): dataset configs `src/configs/dataset_configs/AR/train/acousticroom_train_frac{025,050,075}.json` (sha `7ac8243f…` / `b74687f7…` / `cc0e5fb8…`) = `acousticroom_train.json` with `json_file_path` swapped to the split and `modalities.acoustic_context.restrict_to_split: true`. `AR_md.py::get_custom_metadata` then filters the K = 8 reference-context candidates (otherwise drawn from the whole on-disk room folder) to the split's basenames of that room; an empty pool or a room the split does not list raises `DatasetContractError`, which `SampleDataset.__getitem__` re-raises instead of silently resampling (`src/data/dataset.py` on `66901577`). The training logs print `Found 73473 / 145665 / 218463 files in 243 subfolders` once per rank (verified in the preserved dc_van logs).

**Identity across exp_14's three launches (verified):** all 11 data-path files (3 splits, manifest + `.sha256`, 3 dataset configs, `AR_md.py`, `dataset.py`, the restrict-context test) hash identically at `66901577`, `f609976d`, `6a4b8b08` and the branch head `46a9c3d0`.

## 5. Does the subset mechanism compose with yaw_aug and fa_invariant without code changes?

**On the training side, yes; but no single existing commit carries both halves, so a data-side port is required.**

- The mechanism is purely data-side (dataset config + `AR_md.py` + `dataset.py`). `yaw_aug` acts in `DiffusionCondTrainingWrapper.training_step` on the collated metadata (`_check_yaw_aug_metadata` / `_apply_yaw_aug`, `src/training/diffusion.py`) and `fa_invariant` acts in `_compute_conditioning` → `invariant_conditioning` (`src/data/yaw_rotation.py:390`). Both read `depth` and the four pose fields; neither cares how many context poses were drawn or from which pool (the yaw-aug schema guard requires trailing dimension 3 only). The subset changes only *which* items and *which* contexts enter the batch.
- **Where the code lives:** `restrict_to_split` exists only on `exp-14-data-curve` (cut from `7bbd8aa`, diverged from `localization-exp` at `824c2b68`, Jul 22). `yaw_aug` (exp_15, Aug 11–12) and the FA chunk-cap key `training.frame_avg_max_fwd_samples` (FLAC-repo exp_14 fa_drawshare, `d427175`) exist only on the `check-equivariance-necessity` → `localization-exp` line (`git grep` verified both ways).
- **Choice: cut from `localization-exp` and port exp_14's data side** (not the reverse), because (a) the YAW and B-F 100 % recipes need the yaw-aug code and the chunk-cap key, which are reviewed and regression-tested there (exp_15 golden disabled-path fixture, exp_14-FLAC red-proven literal four-arg default call); (b) the evaluator that produced all three 100 % anchors is byte-identical at that base (§2); (c) the port is tiny and exactly checkable: `AR_md.py` is byte-identical at `7bbd8aa` and `localization-exp` (sha `4846b653…` both), so exp_14's version (`fb2f6882…`) drops in verbatim; `dataset.py` needs exp_14's two hunks (+7 lines: the `DatasetContractError` class and its re-raise before the generic `except Exception` resample), which touch regions the later exp_16 ARE edits did not.
- **Training-path diff audit (each 100 % run's closure vs ours), all gated off for our arms:** since `41aa31dc` (YAW) only exp_25's `canonicalize` commits (gated by `training.canonicalize`, absent); since `f59f5a4` (B-F) additionally the batched orbit (handled by the chunk-cap key below), exp_14-FLAC cap key (default-preserving), exp_15 yaw_aug (gated; and refused with `fa_invariant` by `_parse_yaw_aug_config`), exp_16 ARE (gated by `training.are_lambda`, absent). The exp_25 audit (`depth_canon_training_path_diff_audit.md`) already proved the vanilla path unchanged from P1's `e50d098` to `08ac314`; nothing on the closure changed between `08ac314` and our base except exp_25's gated commits.
- **The one recipe-relevant subtlety — B-F's chunk plan (announcement 06).** B-F 100 % trained under the legacy per-angle loop (one ViT forward, i.e. one train-mode DINOv3 RoPE rescale draw, per orbit angle). On today's code the default cap 64 at micro 32 gives `angles_per_chunk = max(1, 64 // 32) = 2` → {90°, 180°} would share a draw: a different method. Setting `training.frame_avg_max_fwd_samples` = the per-rank micro-batch gives `angles_per_chunk = 1` = the per-angle loop's call structure exactly (`_orbit_average_batched`, `yaw_rotation.py:523-580`: whole angles per chunk, ascending order, identical accumulation; the loop is kept as the equivalence reference at `:504`). `drop_last=True` (`dataset.py:457`) means no tail batch ever changes the partition during training. So B-F's arm config = `FLAC_AR_BF.json` + exactly one key: `frame_avg_max_fwd_samples: 32` (2×32 rung) or `8` (8×8 rung) — the key is checkpoint-embedded, as announcement 06 wants.

## 6. Run matrix (6 required runs + 1 optional bridge)

Run ids follow exp_14 (`dc_van_f025` → `de_<arm>_f<tag>`; never the substring `exp14_`, which `gen_model_comparison.py:207` claims). A non-default rung appends `_r8x8`.

| run id | model config (sha-pinned) | dataset config | items | final ckpt name |
|---|---|---|---|---|
| `de_yaw_f025` / `_f050` / `_f075` | `exp_15…/FLAC_AR_YAWAUG.json` (`b14caae2…`, used in place, = YAW 100 %'s embedded config) | `acousticroom_train_frac{025,050,075}.json` | 73,473 / 145,665 / 218,463 | `epoch=34-step=40000.ckpt` / `epoch=17-…` / `epoch=11-…` |
| `de_bf_f025` / `_f050` / `_f075` | `exp_38…/FLAC_AR_BF_cap32.json` (= `FLAC_AR_BF.json` + `training.frame_avg_max_fwd_samples: 32`; `FLAC_AR_BF_cap8.json` for the 8×8 rung only) | same | same | same |
| `de_p1_f025` (OPTIONAL platform bridge, D2) | `exp_07…/FLAC_AR_BVp1.json` (`733ca52b…`, = dc_van's config) | `acousticroom_train_frac025.json` | 73,473 | `epoch=34-step=40000.ckpt` |

Steps/epoch: 1,148 / 2,276 / 3,413 at both 2×32 and 8×8 (`DistributedSampler` pads to a multiple of the world size, `drop_last` per rank: ⌈n/2⌉/32 and ⌈n/8⌉/8 floor to the same counts; f025 matches dc_van's log `968/1148` in epoch 34).

**Literal training argv (A6000, fresh start)** — dc_van's argv shape exactly (at-launch record `at_launch_dc_van_f025_…txt`) minus exp_14's `--run-contract-json` (an exp_14-branch-only provenance flag with byte-identical Trainer kwargs, not ported) plus an explicit `--precision bf16-mixed` (= the ini default and exp_17's literal flag):

```
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0,1 python train.py \
  --model-config <arm config> --dataset-config src/configs/dataset_configs/AR/train/acousticroom_train_frac<TAG>.json \
  --pretransform-ckpt-path weights/FLAC/VAE.safetensors \
  --max-steps 40000 --batch-size 32 --accum-batches 1 --num-workers 6 --seed 42 \
  --num-gpus 2 --strategy ddp_find_unused_parameters_true --sync-batchnorm true --precision bf16-mixed \
  --logger none --checkpoint-every 2500 --name <run> --experiment-name <run> --save-dir <run dir>
```

Neuronic: the same argv under `torchrun --standalone --nnodes=1 --nproc_per_node=<NGPU> train.py …` (exp_15's launcher form; PL 2.1 elects TorchElastic; `SLURM_PROCID/LOCALID/NODEID/NTASKS*` unset so `train.py:100-101` cannot add a rank offset to the seed and SLURMEnvironment cannot claim the job), with `--batch-size <MB> --num-gpus <NGPU>` of the rung. A resumed leg adds `--ckpt-path <run dir>/epoch=*-step=<S>.ckpt --max-steps <S+L>` (exact per PL 2.1, CLAUDE.md "Chunked chain training").

`--logger none` (dc_van's choice): no numeric effect; avoids the `flac` env's wandb-0.15.4 online failure (CLAUDE.md exp_25 trap) and W&B resume conflicts for chained legs; loss is visible in the teed tqdm log.

**Announcement 06 chunk plan (B-F):** C4 orbit (4 angles, 3 non-zero), per-rank micro 32, `frame_avg_max_fwd_samples` 32 → `angles_per_chunk` 1, 0 shared RoPE draws = the B-F 100 % (legacy per-angle) plan. On the 8×8 rung: micro 8, cap 8 → also per-angle. Eval cap 64 (eval mode has no RoPE draw; exp_35 used 64).

## 7. Platforms — exact differences and the recommendation

| | A6000 box (this machine) | neuronic 2×L40 (**recommended**) | neuronic 8×L40 (opt-in only) |
|---|---|---|---|
| ranks × micro × accum | 2 × 32 × 1 | 2 × 32 × 1 | 8 × 8 × 1 |
| eff batch / SyncBN batch | 64 / 64 | 64 / 64 | 64 / 64 |
| per-rank micro (RoPE-draw granularity: 1 draw per ViT forward per rank, HF DINOv3 `augment_patches_center_coordinates`) | 32 | 32 | 8 (4× more, smaller-batch draws) |
| B-F chunk cap → plan | 32 → per-angle | 32 → per-angle | 8 → per-angle (config `FLAC_AR_BF_cap8.json`) |
| yaw-aug draw keys `(42, step, rank, index)` | = YAW 100 % | = YAW 100 % (same ranks/indices) | different keys (same distribution) |
| launcher | `python train.py` (PL subprocess DDP) | `torchrun --standalone` | `torchrun --standalone` |
| GPU | RTX A6000 48 GB (Ampere) | L40 48 GB (Ada) | L40 |
| precision | bf16-mixed | bf16-mixed | bf16-mixed |
| attention backend (DiT) | flash-attn 2.7.4.post1 importable since Nov 1 → flash | whatever `/n/fs/gatrdp/envs/flac` has (unknown; gated, D4) | same |
| env | `~/miniconda3/envs/flac` (torch 2.7.0+cu126, PL 2.1.0) | `/n/fs/gatrdp/envs/flac` (exp_15 pins torch 2.7.0+cu126, PL 2.1.0) | same |
| ckpt destination | NAS run dir directly (exp_14 practice) | `/n/fs/gatrdp/codespace/FLAC/outputs_FLAC/exp38_data_efficiency/<run>/` → copied to the NAS with sha256 verification | same |

**Why 8×8 is opt-in only.** Every reference curve point (P1, dc_van ×3, YAW, B-F) was trained at 2 ranks × 32. The 8×8 rung keeps eff/BN batch 64 but changes the per-rank micro-batch, hence the number and batch-sharing of train-mode RoPE draws (announcement 06 treats exactly this as a method change), the yaw-aug draw keys and the platform. The only same-model evidence in the record: exp_11 VANL@40k (vanilla, 8×8 L40) K=8 = 8.048 / 1.022 / 37.32 vs P1@40k (2×32 A6000) 8.993 / 1.009 / 40.65 (`model_comparison.md:49,71`) — a 0.95 T60 / 3.3 ms EDT gap at matched steps (confounded with code era and backend, but of the size of the whole 25 → 100 % FLAC effect). A neuronic 8×8 cell would therefore not be "directly comparable" in the sense Yixun asked for. The scripts support it (`RUNG=8x8 ALLOW_NONMATCHED_RUNG=1`, run id suffix `_r8x8`) so the choice stays Yixun's (D1).

**2×L40 still changes the GPU model.** There is no 2×32 L40 run in the record to bound that effect. Recommended mitigation (D2): the optional bridge `de_p1_f025` on neuronic 2×L40, compared with dc_van_f025 (A6000) at the same subset — one ≈ 1-day 2-GPU job; a gross platform offset would show beyond the ±0.5 T60 band.

## 8. Time, disk, destinations

**Measured anchors (2 × A6000 × 32):** YAW 100 % = 21 h 44 m wall for 40k with both cards exclusive and flash-attn (1.96 s/step, `42cbddaf`); P1 July 0.259 steps/s (3.86 s/step, shared); dc_van Sept 3.7 s/step (two DDP jobs on the same two cards, SDPA); exp_25 CANON (vanilla recipe) 6–19 s/step next to 100 %-utilization co-tenants; B-F July 0.079 steps/s (12.7 s/step, shared); paper appendix: 40.3 h FLAC, 124.9 h OrbitRIR (3.6 / 11.2 s/step, "cumulative active wall-clock"). B-F/vanilla cost ratio under like conditions: 3.1 (paper) – 3.5 (July). exp_11 on 8×L40 × 8: 0.96 s/step vanilla, 2.75 s/step C4 (batched, 3 angles in one chunk).

| run (40k steps) | A6000 2×32, exclusive | A6000 2×32, shared | neuronic 2×L40 ×32 (est.) | neuronic 8×L40 ×8 (est., opt-in) |
|---|---|---|---|---|
| YAW (any fraction) | ≈ 22 h (measured exp_17) | ≈ 40–45 h | ≈ 17–22 h (L40 ≈ 1.0–1.25× A6000 bf16; data loading per rank unchanged) | ≈ 11–12 h (exp_11 0.96–1.05 s/step) |
| B-F (any fraction) | ≈ 61–77 h (1.96 s/step × 3.1–3.5) | ≈ 125–141 h | ≈ 50–75 h | ≈ 35–55 h (per-angle at micro 8 is latency-bound: ≥ 3 × the 2.75 s/step chunk-shared cost per extra forward is the upper end) |
| P1 bridge (optional) | ≈ 22 h | — | ≈ 17–22 h | — |

The steady-state rate is re-measured by every run's first 1,000 steps (logged); the GPU-0 smoke (§11) gives a single-rank lower bound. **Earliest start on this box:** both cards free — GPU 1 holds the host-A exp_30 evaluations (pids 3599091, 3602030, cwd `~/codespace/flac-fork-attn*`, expected to end ≈ 04:00 Nov 8) and the cross-session handshake in `~/codespace/claude_gpu_yield/` (REQUEST by the FLAC session for exp_36/exp_37 still standing) — the main session must sequence our B-F f025 through that handshake. B-F f025 exclusive from ≈ 04:00 Nov 8 → ETD ≈ Nov 10 17:00 – Nov 11 09:00 EST (≈ Nov 13 if shared). Neuronic: queue-dependent; ≈ 1 day (YAW) and ≈ 2.5–3 days (B-F) of service time after start at 2×L40.

**Disk.** A wrapped checkpoint is 723,922,667 B (B-F/YAW, verified); 16 cadence checkpoints/run ≈ 11.6 GB; 6 runs ≈ 70 GB (+ 11.6 GB for the bridge). NAS `/media/diskstation` has 14 TB free (df). Eval: metric JSON + stream sidecar ≈ 1 MB/cell; scored-waveform dumps 6,337 × 10,240 float32 ≈ 260 MB/cell × 60 cells ≈ 15.6 GB on the NAS. Neuronic scratch: ≈ 11.6 GB per run under `/n/fs/gatrdp/codespace/FLAC/outputs_FLAC/exp38_data_efficiency/` until copied.

**Destinations (announcement 07).** `/media/diskstation/yixunhu/FLAC/checkpoints/exp38_data_efficiency/<run id>/` — the A6000 runs write there directly (`--save-dir`); neuronic runs are copied there after completion and verified against the `CHECKPOINTS.sha256` the final leg writes. Waveform dumps: `/media/diskstation/yixunhu/FLAC/exp38_pred_waveforms/` (announcement 08).

## 9. Evaluation protocol

### 9.1 Evaluator, backend, flags
`eval_FLAC.py` at the exp_38 branch pin (byte-identical eval path to exp_25 `d6ca307a` / exp_35 `5b065801`; our data-side port is inert at eval: no eval config sets `restrict_to_split`, and `DatasetContractError` is never raised there), launched through the reviewed exp_35 no-flash launcher (`PYTHONDONTWRITEBYTECODE=1 python overlays/no_flash/exp35_no_flash_launch.py eval_FLAC.py …`, sentinel line `EXP35_NO_FLASH_LAUNCHER attention_backend=sdpa …` required in every cell log) → SDPA, the backend of dc_van's, exp_25's and exp_35's cells. On this A6000 box only (eval hardware = the anchors' hardware).

Per cell (exactly exp_25 / exp_35's argv shape, verified from `depth_canon_cells.json` and exp_35's dry-run log):
- YAW (and P1 bridge): `--model-config <the run's training config> --dataset-config src/configs/dataset_configs/AR/eval/acousticroom_unseeneval{,_1}.json --cond-method vanilla --cond-autocast bf16 --cfg-scale 1.0 --steps 1 --batch-size 64 --device cuda --record-per-scene --record-stream --expected-stream-count 6337 --seed <S> --rotate-mode fixed --rotate-deg 0 --eval-name <run>_K<K>_s<S> --ckpt-path <farm link> --dump-scored-waveforms /media/diskstation/yixunhu/FLAC/exp38_pred_waveforms`
- B-F: the same plus `--cond-method fa_invariant --frame-avg-angles 0,90,180,270 --frame-avg-max-fwd-samples 64`.

The eval protocol is bound to the checkpoint: the driver refuses unless the checkpoint's embedded `model_config` equals the `--model-config` file (canonical JSON) and `--cond-method` equals the embedded `training.cond_method` (absent ⇒ vanilla) — announcement 05.

### 9.2 Cells
6 runs × K ∈ {8, 1} × seeds 42–46 = **60 cells** (+ 10 for the optional bridge). K=1 is free in this tooling (same driver, `acousticroom_unseeneval_1.json`). Outputs land next to a per-run farm symlink `outputs_FLAC/exp38_data_efficiency/eval/<run>/<ckpt name>` → NAS checkpoint (eval_FLAC writes beside `--ckpt-path`'s lexical directory, `eval_FLAC.py:629-637`), never in another experiment's directory. Cost: ≈ 6.5–20 min (vanilla) / ≈ 25 min (B-F, exp_35 cadence) per cell → ≈ 20 GPU-h, ≈ 10 h wall on two cards.

### 9.3 E0 replay hard stops (run first; the campaign does not start unless both pass)
- **E0a (bridges the exp_14 evaluator):** dc_van_f025 @40k (`aa1a2cd6…`) K=8 s42 with `FLAC_AR_BVp1.json` (= dc_van's config, sha `733ca52b…`) under our §9.1 vanilla argv, compared with exp_14's record (NAS + git copy, sha `4136ac75…`): PASS iff |Δ| ≤ T60 0.001, C50 0.0005, EDT 0.005, FD 0.0005, R@k 0.2 (exp_35 E2's tolerances); exact equality reported.
- **E0b (binds the fa path to exp_35):** B-F@40k (`5319feb4…`) K=8 s42 under §9.1's B-F argv vs exp_35's cell (sha `f167470e…`): PASS iff every `metrics` value is exactly equal (same code, same backend, same GPU class — exp_35's E1 showed exact replays).
A failure stops the campaign and goes to Yixun; nothing is adjusted silently.

### 9.4 Aggregation and the table
`exp38_collect.py` reads only validated cells (protocol fields, `n_samples == 6337`, `ckpt_sha256_loaded` == the run's audited final checkpoint, stream sidecar present, dump COMPLETE) and writes `data_efficiency_results.md` + `.json`: per model × fraction × K the mean ± sample SD over seeds 42–46 of the top-level `metrics` (split-level item-weighted — what every Table 3(b) cell uses), next to the reference rows of §2 (read from their raw JSONs, sha-recorded). Incomplete runs render PENDING, never a number. `model_comparison.md` rows (announcement 04) are added in the results phase (regeneration happens on neuronic, CLAUDE.md table-regen trap).

## 10. Implementation (per file; TDD — every function's test is written first and seen RED)

### Round A — port exp_14's data side (production data path)
1. `src/tests/test_ar_md_restrict_context.py` — byte copy of `66901577`'s (sha `5f63591e…`; hermetic toy-tree fixtures; B1 bitwise default-path vs `7bbd8aa`, B2–B9 restriction, fail-closed, worker/process propagation). RED on our base (no `DatasetContractError`).
2. `src/data/dataset.py` — exp_14's two hunks (+7): `class DatasetContractError(RuntimeError)`; `except DatasetContractError: raise` before `except Exception`. `src/configs/dataset_configs/custom_metadata/AR_md.py` — byte copy of `fb2f6882…`. GREEN.
3. Data files, byte copies (sha-pinned): the 3 splits, `train_frac_manifest_s2026.{json,sha256}`, the 3 dataset configs. `src/tests/test_exp38_subset_port.py` (new): (a) every ported file hashes to its exp_14 launch blob (constants); (b) `AR_md.py` equals the exp_14 blob and the base `AR_md.py` equals `7bbd8aa`'s (so the port is exactly exp_14's diff); (c) each split: item count, 243 rooms, nesting f025 ⊂ f050 ⊂ f075 ⊂ `train.json` per room; (d) each frac config parses and differs from `acousticroom_train.json` only in `json_file_path` and `acoustic_context.restrict_to_split: true`; (e) `dataset.py` re-raises `DatasetContractError` and still resamples a plain exception (behavioural, toy tree via test 1's fixtures).

### Round B — the exp_38 kit (`worklog/worklog_yixun/exp_38_data_efficiency_claude/`; tests in `src/tests/test_exp38_*.py`)
4. `exp38_registry.py` — the single source of truth; pure functions + a CLI. Tests `test_exp38_registry.py`:
   - `RUNS`, `ARMS`, `FRACS`, `RUNGS`, `PINS` constants (configs + sha256, split/dataset-config sha256 + counts, final ckpt names, rungs `a6000_2x32` / `neuronic_2x32` / `neuronic_8x8` with `allowed_by_default`, VAE sha `8d82159e…`, DINOv3 rev `114c1379…` + file shas, NAS/neuronic roots). Test: constants equal what the files on disk hash to.
   - `derive_bf_config(base, cap)` / `bf_config_path(rung)` — test: committed `FLAC_AR_BF_cap{32,8}.json` bytes == regenerated bytes; parsed diff vs `FLAC_AR_BF.json` is exactly `training.frame_avg_max_fwd_samples`; the factory parses it (wrapper `frame_avg_max_fwd_samples == cap`).
   - `chunk_plan(arm, rung)` — test: B-F → cap = micro, `angles_per_chunk == 1`, shared draws 0 for both rungs; a spy conditioner run through `_orbit_average_batched` with that cap makes exactly 3 calls of B samples each, in angle order, equal to `_orbit_average_loop`'s calls (announcement 06 evidence); YAW → `None`.
   - `run_id(arm, tag, rung)`; `build_train_argv(run, rung, save_dir, resume=None)` — tests: the A6000 argv equals the literal of §6 token-for-token (and dc_van's at-launch argv minus `--run-contract-json` plus `--precision bf16-mixed` for the P1 bridge); 8×8 refused without the explicit allow; resume adds exactly `--ckpt-path`/`--max-steps`, leg ends on the 2,500 cadence, ≤ 40,000; no token contains `exp14_`.
   - `check_subset(tag, repo)` — split/config sha + counts; tampered fixture fails.
   - `check_run_dir(dir, start_step)` — fresh start refuses any `*.ckpt` (or a non-empty dir other than our own at-launch records); resume requires exactly the cadence set ≤ start_step and nothing beyond.
   - `audit_ckpt(path, run, step)` — embedded `model_config` == arm config (canonical), `global_step == step`, `diffusion_ema` keys present, scheduler `last_epoch == step` (tiny synthetic checkpoint fixtures).
   - `log_gate(text, run, rung, phase, backend)` — start: `Found <n> files in 243 subfolders` × NGPU, DDP `MEMBER: k/NGPU` for every k (or torchrun's world size line), YAW banner `yaw_aug ENABLED img_w=512 seed=42` as a whole line before the first step line, backend (`flash`: no `flash_attn not installed`; `sdpa`: ≥ NGPU such lines); end: Lightning's `` `Trainer.fit` stopped: `max_steps=<N>` reached. `` and no non-finite `train/loss`. Tests on synthetic logs, one per failure mode.
   - `vit_pin_check(cache)` / `vae_check(path)` — the exp_07/exp_25 DINOv3 snapshot pin (rev + both file shas), VAE sha.
   - CLI `dry-run --platform {a6000,neuronic} [--rung …]`: for every run prints the exact argv, config path + sha, dataset config + split sha + item count, chunk plan, run-dir state (must be checkpoint-free), and fails non-zero on any check; `argv`, `check`, `audit-ckpt`, `log-gate`, `ckpt-manifest` (sha256 of the 16 cadence checkpoints).
5. `exp38_train_local.sh` — the A6000 launcher, thin bash over the registry. `MODE=DRY_RUN|SMOKE|FULL|RESUME`, `RUN=<run id>`, `EXPECT_SHA` (FULL/RESUME: must equal HEAD; tracked tree clean), `GPU_POLICY=idle|shared` (idle = refuse while any foreign compute process is on GPU 0 or 1; shared = per-card free-memory floor 21,900 MiB B-F / 12,288 MiB vanilla, co-tenants disclosed), `ATTN_BACKEND=flash|sdpa` (sdpa = `PYTHONPATH=overlays/no_flash` + `PYTHONDONTWRITEBYTECODE=1`, children inherit it). Gates: interpreter under `envs/flac`, registry `check`, VAE + DINOv3 pins, NAS mount (CIFS share in `findmnt`) + write probe + ≥ 30 GiB free, run dir fresh (FULL) / resume preflight (RESUME, `RESUME_STEP`). Launch `setsid python train.py <argv>` teed to `exp38_<run>_<ts>_train.log` in the experiment folder (kept untracked while running; tqdm-heavy), at-launch record `at_launch_<run>_<ts>.txt` (argv, HEAD, file shas, host, GPU UUIDs, `pip freeze` sha, backend, co-tenants) written before the fork; a start watcher (`exp38_registry.py log-gate --phase start --wait`) kills ONLY our own process group if a start gate fails; after exit: end gate, `audit-ckpt` of `step=40000`, cadence set {2500…40000}, `CHECKPOINTS.sha256`. SMOKE: one GPU (`SMOKE_GPU`, default 0), `--num-gpus 1` (no SyncBN — a single-rank smoke cannot exercise DDP/SyncBN; disclosed), micro 32, `SMOKE_STEPS` (default 20), `--checkpoint-every 1000000`, save-dir under `outputs_FLAC/exp38_smoke/`, asserts banner/dataset count/finite loss/zero checkpoints, records peak memory (nvidia-smi sampler) and the last tqdm rate. DRY_RUN: every gate that does not need a GPU, then prints the argv; spawns nothing.
6. `exp38_train.sbatch` — the neuronic leg worker (exp_15's sbatch conventions: `/n/fs/gatrdp/envs/flac/bin/python` pinned, `HF_HOME=/n/fs/gatrdp/hf_cache`, `HF_HUB_OFFLINE=1`, `PYTHONNOUSERSITE=1`, transcript under the experiment folder). Inputs: `RUN`, `RUNG`, `EXPECT_SHA`, `LEG_START`, `LEG_END`, `ATTN_BACKEND`. Gates: HEAD == `EXPECT_SHA`, tracked tree clean, registry `check`, pins, allocated GPUs == NGPU and all L40, run dir state (LEG_START = 0 fresh; else exactly the cadence set ≤ LEG_START and `audit-ckpt` of the LEG_START checkpoint). torchrun launch + start watcher + end gate + `audit-ckpt` of LEG_END; at 40,000 also `CHECKPOINTS.sha256`. Exit codes documented; the job never edits git.
7. `exp38_submit_neuronic.sh` — derives the Slurm resources from the rung (`--gres=gpu:l40:N --cpus-per-task 8+7N --mem` …, exp_15's formula, raised for micro 32: 2×32 → 22 CPUs / 64 G; 8×8 → 64 CPUs / 108 G), `--time` from `TIME` (Yixun's partition limit, D3), splits `[START, 40000]` into legs of `LEG_STEPS` (multiple of 2,500; default 40,000 = monolithic), submits the legs as one `afterok` chain (each leg's resume path is deterministic), refuses if `squeue` shows a job of the same name (query failure is fatal), appends every submission line to `exp38_neuronic_submissions.log`. `DRY_RUN=1` prints the chain and submits nothing. Recovery after a failed leg = re-run the same command with `START=<last good boundary>`.
8. `exp38_eval.py` — `plan` (the 60 (+10) cells + E0a/E0b with full argv as JSON), `prepare <run>` (verify the NAS final checkpoint against `CHECKPOINTS.sha256` + `audit-ckpt`, create the farm symlink), `run --gpu G --cells …` (sequential; refuses while a foreign compute process is on G unless `--shared`; skip a cell only if its outputs validate; each cell teed to `exp38_eval_<cell>_<ts>.log`; sentinel + `n_samples` + embedded-config + `ckpt_sha256_loaded` checks), `e0` (the two replay verdicts → `exp38_e0_verdict.json`). Tests `test_exp38_eval.py` (argv per arm, cond-method binding refuses a mismatch, skip-only-if-valid, E0 tolerance logic on synthetic records).
9. `exp38_collect.py` — `validate_cell`, `aggregate(cells) → mean ± SD`, `render_table` (golden markdown), reference rows from §2's raw JSONs (sha-recorded), PENDING on incompleteness. Tests `test_exp38_collect.py`.
10. Records: `data_efficiency_worklog.md`, `commits_data_efficiency.md`, `data_efficiency_command.md` (every launch appended at launch time), `data_efficiency_params_set_up.md` (at launch), `data_efficiency_neuronic_commands.md` (Yixun's command sheet).

Commit discipline: < 200 changed code lines per commit (the byte-copied data files and the byte-copied exp_14 test are data/test copies, not new code), one commit per red→green cycle, SHAs in `commits_data_efficiency.md`; `git pull --rebase` before each push.

## 11. Validation ladder (CPU-only except rung 6)

1. Static: `py_compile`, `bash -n`, `json.load` of every config, `git diff --check`.
2. pytest: the new tests + the existing suites touching the changed/used code (`test_yaw_aug_*`, `test_fa_*`/`test_yaw_rotation*`, `test_train_*`, `test_sync_batchnorm*`, `test_eval_*`, `test_exp35_no_flash_launch.py`, dataset tests) — all green (pre-existing GPU-only failures, if any, listed).
3. Real-data readback (CPU): one real batch per fraction through `create_dataloader_from_config` (2 workers, micro 4): `Found <n> files in 243 subfolders`; every drawn context basename ∈ the split's room set; no `DatasetContractError`.
4. Tiny synthetic forward (CPU): build the YAW and B-F-cap32 wrappers, one `training_step` on a seeded synthetic batch — YAW banner path + rotation applied; B-F: 3 orbit chunk calls of B samples.
5. `DRY_RUN` of every launch path: `exp38_registry.py dry-run --platform a6000` and `--platform neuronic` (all 6 runs, both rungs where allowed), `MODE=DRY_RUN exp38_train_local.sh` for each run, `DRY_RUN=1 exp38_submit_neuronic.sh` for each run (monolithic and chained), the sbatch's gate section in a dry mode on this box with the neuronic paths substituted (`EXP38_SBATCH_DRYRUN=1`), `exp38_eval.py plan`.
6. GPU smoke (≤ 15 min, GPU 0 only, Yixun-allowed): `MODE=SMOKE RUN=de_yaw_f025` and `RUN=de_bf_f025`, 20 steps each at micro 32, no checkpoints — banner, finite loss, peak memory (B-F expected ≈ 15–16 GiB/rank with grad-ckpt, exp_07 M1 15,712 MiB), single-rank rate.
7. Full runs — NOT started by the Coder; launched only after Yixun's decisions (§13) and the Codex code reviews.

## 12. Launch acceptance criteria (per run; judged at the first log lines and at the end)

HEAD == the pushed launch SHA; tracked tree clean; 2 ranks (`MEMBER: 1/2`, `2/2`) or the rung's NGPU; per-rank micro 32 (8 only on the opt-in rung); accum 1; SyncBN on; bf16-mixed; dataset count line == the fraction's item count, 243 subfolders, per rank; YAW: banner line before step 0; B-F: config embeds `frame_avg_max_fwd_samples` = micro; backend as declared; ≥ 1 optimizer step with finite loss; checkpoints exactly every 2,500 steps; `step=40000` checkpoint audited (embedded config == arm config, global_step 40,000, EMA present) and `CHECKPOINTS.sha256` written. A chained neuronic run additionally: every leg's boundary checkpoint audited before the next leg starts (`afterok`), and the chain disclosed (not bit-equivalent to a monolithic run: PL restores optimizer/scheduler/EMA/loop counters, not RNG streams or the mid-epoch dataloader position).

## 13. Decisions for Yixun (with recommendations)

- **D1 — neuronic rung.** Recommend **2×L40 × 32/GPU** (per-rank recipe identical to every reference curve point; easier to schedule than 8-GPU jobs). 8×8 only as an explicitly labelled cross-recipe run (§7 evidence: 0.95 T60 VANL-vs-P1 gap).
- **D2 — allocation / platform bridge.** (a) the default split (neuronic: YAW ×3 + B-F f050/f075 at 2×L40×32; this box: B-F f025) **plus the optional `de_p1_f025` bridge on neuronic 2×L40** [recommended]; (b) everything on this box (exact hardware identity; ≈ 3 × 22 h + 3 × 61–77 h ≈ 12–13 days exclusive, sequential); (c) the default split without a bridge.
- **D3 — neuronic walltime.** `LEG_STEPS=40000` (monolithic, like the references) if the partition allows ≈ 30 h (YAW) / ≈ 80 h (B-F); otherwise the longest cadence-aligned legs that fit (e.g. YAW 20,000 × 2 and B-F 10,000 × 4 at 24 h), disclosed as chained.
- **D4 — attention backend for training.** Recommend **flash** (= the P1/B-F/YAW 100 % runs; dc_van trained SDPA — a pre-existing difference inside the FLAC curve); the gate refuses a backend mismatch, so on neuronic the run fails closed if flash-attn is absent unless `ATTN_BACKEND=sdpa` is chosen. Measured eval-side size of the backend effect: P1@40k K8 per-seed flash vs SDPA differ by ≤ 0.003 T60 / ≤ 0.007 EDT (exp_07 vs exp_25 files).
- **D5 — this box's GPU policy for B-F f025.** Recommend **idle** (both cards free of foreign compute processes; sequenced through the `claude_gpu_yield` handshake by the main session); `shared` only with Yixun's go (≈ 2× slower, co-tenants slowed too).
- **D6 — eval.** Recommend §9 as written: the anchors' evaluator at our pin, SDPA via the exp_35 no-flash launcher, E0a/E0b hard stops, K=1 included, all evals on this box.

## 14. Risks and disclosures

- Single training seed per cell (§1). Matched steps, not matched epochs or compute.
- Hardware/backend/code-era differences between our runs and the references are enumerated in §3/§7 and repeated in the results.
- dc_van (FLAC 25/50/75) trained SDPA at the exp_14 pin; our runs train with the declared backend at our pin — the FLAC row and the new rows are not one-pin; disclosed.
- Concurrent writers: `localization-exp` moves; we never rebase a launched pin; any closure change after a launch = a new run, never a resume.
- Never edit a running bash script; never touch another session's processes or the handshake files; the Coder starts no real training.
