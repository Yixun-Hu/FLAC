# plan_data_efficiency — exp_38: YAW and OrbitRIR (B-F) data-efficiency curve at 25 / 50 / 75 % of AR

**Author:** Claude Opus 5.5 (Coder/Planner subagent, max effort) · **Date:** 2026-11-07 · **Status:** **r1** — all Codex plan-review r1 findings addressed (§15 changelog); r0 archived in `plan_history/`. Plan review 1 of max 3 used. **Branch / worktree:** `exp-38-data-efficiency` in `~/codespace/exp-38-data-efficiency`, cut from `localization-exp` `2b42468c8f429b1d93da93b3e4ddb7bd6f8f1932` (= `origin/localization-exp` at 20:24 EST Nov 7). `main` is never touched.

"Verified" = re-checked by the Planner in this session; "inferred" = argued from records; "unknown" = not recoverable from any surviving record.

**Process note (Codex r1 HIGH 13):** while review r1 ran, the Coder drafted the round-A port and a first registry / launcher / sbatch / submitter in the worktree (uncommitted). Those drafts are provisional: they are reconciled with this r1 plan and go to the Codex code review as code; nothing launches before it.

---

## 1. Question and estimand

**Question (`data_efficiency_yixun_query.md`):** how do Yaw-Augmented FLAC (YAW) and OrbitRIR (B-F, `fa_invariant` C4 frame averaging) degrade when trained on 25 / 50 / 75 % of the AcousticRooms (AR) training split, next to vanilla FLAC (P1), whose curve exists (exp_14 `dc_van_f025/f050/f075` + P1@40k)?

**Estimand.** For model `m ∈ {YAW, B-F}`, fraction `f ∈ {0.25, 0.50, 0.75}`, `K ∈ {8 (headline), 1}`: the 40,000-step EMA endpoint of ONE monolithic training run (seed 42), evaluated on the full unseen split (`data/AR/unseen_eval.json`, 6,337 items / 17 rooms; announcement 01): T60 %, C50 dB, EDT ms (+ FD, R@k), **split-level item-weighted** (every Table 3(b) cell's convention, §9.4), mean ± sample SD over eval seeds 42–46. Recipe per fraction = the model's own 100 % recipe with the dataset config swapped for exp_14's subset config; nothing else.

**What the comparison is (Codex r1 Q10):** a **historical matched-subset, fixed-optimizer-step-budget comparison** — same subsets, same 40k × 64 budget, same batch / rung, same evaluator — against reference runs trained earlier at other commits. Residual historical asymmetries (§14) are disclosed; no unqualified causal claim. One training seed per cell (eval seeds measure eval variability only; exp_14 recorded a ≈ ±0.5 T60 step-to-step band, `data_curve_results.md:3`). Effective epochs 34.8 / 17.6 / 11.7 / 8.8 at 25 / 50 / 75 / 100 %.

## 2. What already exists (reused, never re-run)

| Curve point | Source | K=8 T60 / C50 / EDT (item-weighted, mean ± SD, s42–46) | Evaluator / eval backend |
|---|---|---|---|
| FLAC 25 % | exp_14 `dc_van_f025` (ckpt `aa1a2cd6…`) | 10.252±0.012 / 1.048±0.002 / 42.914±0.058 | exp_14 evaluator (`66901577` lineage), SDPA |
| FLAC 50 % | `dc_van_f050` | 9.450±0.010 / 1.044±0.002 / 41.587±0.064 | same |
| FLAC 75 % | `dc_van_f075` | 8.913±0.009 / 1.104±0.002 / 41.636±0.065 | same |
| FLAC 100 % | P1@40k `c4c67882…`, exp_25 P1-pin | 8.993±0.011 / 1.009±0.004 / 40.650±0.103 | exp_25 pin `d6ca307a`, SDPA |
| YAW 100 % | YAW@40k `ac1f2603…`, exp_25 YAW-pin | 7.965±0.015 / 1.013±0.002 / 39.923±0.058 | exp_25 pin `d6ca307a`, SDPA |
| B-F 100 % | B-F@40k `5319feb4…`, exp_35 BF-pin T | 8.202±0.017 / 0.978±0.002 / 38.794±0.074 | exp_35 pin `5b065801`, SDPA (no-flash overlay) |

Recomputed by the Planner from the raw JSONs; they reproduce Yixun's Table 3(b) DINOv3 rows (75 % EDT 41.640 = 2-dp rounding of 41.6362). The exp_35 B-F JSONs are not tracked on any branch, so they (and every other reference JSON the collector reads) are **imported into `ref_cells/` with a sha256 manifest** (§10 B9).

**Evaluator identity (NIT 1 corrected).** `git diff 5b065801 2b42468c` and `git diff d6ca307a 2b42468c` over `eval_FLAC.py src/{models,data,inference,metrics,training}` are both empty (verified); `overlays/no_flash/` is absent at `d6ca307a` and present at `5b065801` and at our base. So the evaluator/model code that produced all three 100 % anchors is byte-identical to ours; dc_van used the older exp_14 evaluator, bridged by the E0 replays (§9.3).

## 3. The 100 % recipes (verified from each checkpoint's embedded `model_config` and the records)

All three loaded on CPU in this session: each embeds `model_config`, `global_step 40000`, `epoch 8`, `InverseLR(1e6, 0.5, 0.99)`, `last_epoch 40000`, `_last_lr 4.9029e-5`.

| | P1 (FLAC) | YAW | B-F (OrbitRIR) |
|---|---|---|---|
| ckpt sha256 | `c4c678826cdd…` | `ac1f26034e4f…` | `5319feb4af87…` |
| embedded config == (canonical JSON) | `exp_07…/FLAC_AR_BVp1.json` (`733ca52b…`) | `exp_15…/FLAC_AR_YAWAUG.json` (`b14caae2…`) = BVp1 + `training.yaw_aug {true, 512, 42}` | `exp_07…/FLAC_AR_BF.json` (`3e677204…`) = release + `cond_method fa_invariant`, `frame_avg_angles` C4 |
| ViT grad-ckpt (both ViTs) | true | true | true |
| experiment | exp_07 P1 | **exp_17 yaw_aug_a6000, grad-ckpt-ON arm** (NOT exp_15's 8×L40 arm; NOT exp_17's retired grad-ckpt-OFF variant on `origin/exp17-yawaug-scratch`) | exp_07 B-F |
| rung | 2×A6000, 32/GPU × 2 × accum 1, `ddp_find_unused_parameters_true`, SyncBN (BN 64) | same (exp_17 params + worklog 2026-08-15T16:10) | same (`bf_scratch_launch.sh` literal `MB=32 ACC=1`) |
| precision | bf16-mixed (ini default) | bf16-mixed | bf16-mixed (ini default) |
| seed / steps / cadence | 42 / 67.5k budget, 40k ckpt / 2,500 | 42 / 40k / 2,500 | 42 / 67.5k budget, stopped 40,249, 40k ckpt / 2,500 |
| lineage | **resumed once at 32.5k** (legs `e50d098` → `1b56e7a`) | monolithic, 21 h 44 m with both cards exclusive (`42cbddaf`; 1.93–1.97 s/step) | monolithic |
| training commit | `e50d098` / `1b56e7a` | **unknown** — the removed `exp-17-yawaug-a6000` worktree's launch HEAD was never recorded in any surviving file (searched: exp_17 worklog/params/analysis on `origin/exp17-yawaug-scratch`, PROVENANCE.md, exp_25 ledger); the closure of `41aa31dc` (its planned base) is the best inference | `f59f5a4` era (Jul 19) |
| FA chunk plan (ann. 06) | n/a | n/a | legacy per-angle loop (before `1479304`/`8094d60`, Aug 6) |
| DiT attention backend | flash-attn (documented: `6596121` "training env switched to conda flac … flash-attn kernel delta disclosed") | **unknown** (inferred flash: the exp_17 eval logs of Aug 16 in the same env lack `flash_attn not installed`; the training log is lost) | flash-attn (documented, `6596121` precedes the Jul 19 launch) |
| logger | wandb | wandb | wandb |

Notes. (i) `--max-steps 40000` is **schedule**-equivalent to the 67.5k-budget runs' 40k checkpoints (`InverseLR` reads only its own parameters and the step), not **trajectory**-equivalent to P1's resumed history. (ii) The YAW "same 100 % recipe" claim is scoped to what is bound: the embedded config, rung, precision, seed, budget and cadence; launch source and backend are disclosed as unknown/inferred (Codex r1 HIGH 1). (iii) dc_van: P1's config byte-for-byte and P1's rung, but SDPA (Sept env), at `66901577`/`f609976d`/`6a4b8b08`, `--logger none`.

## 4. How exp_14 built and applied the subsets (verified)

**Built** (`src/tools/make_ar_train_subsets.py` on `exp-14-data-curve`; exp_14 plan §3 + Amendment 1): nested per-room stratified random subsets of `data/AR/train.json` (291,210 / 243 rooms, `aa4e52d6…`), seed 2026, eligibility `sampler_faithful_S00int_v1`, top-ups so no target's restricted context pool is empty; manifest `data/AR/train_frac_manifest_s2026.json` (`553534ea…`, per-room 0 / 1–7 / ≥ 8 histograms).

| tag | split | sha256 | items | rooms | ctx histogram 0 / 1–7 / ≥ 8 |
|---|---|---|---|---|---|
| 025 | `data/AR/train_frac025_s2026.json` | `ce33f34f…` | 73,473 | 243 | 0 / 16,890 / 56,583 |
| 050 | `data/AR/train_frac050_s2026.json` | `bf330846…` | 145,665 | 243 | 0 / 14,303 / 131,362 |
| 075 | `data/AR/train_frac075_s2026.json` | `57388ce8…` | 218,463 | 243 | 0 / 17,430 / 201,033 |

**Applied:** `acousticroom_train_frac{025,050,075}.json` (`7ac8243f…` / `b74687f7…` / `cc0e5fb8…`) = `acousticroom_train.json` + split path swap + `acoustic_context.restrict_to_split: true`; `AR_md.py` filters the K = 8 context candidates (otherwise the whole on-disk room folder — at 100 % this includes seen-eval hold-outs, ≈ 1.8 % more than `train.json`, exp_14 query) to the split's basenames of that room; `DatasetContractError` (empty pool, unlisted room, unreadable split) is re-raised by `SampleDataset.__getitem__` instead of resampled. `json_scandir` publishes `info['json_file_path']` (unchanged on our base, verified by Codex r1 Q2). Training logs print `Found <n> files in 243 subfolders` once per rank.

**Identity across exp_14's three launches:** all 11 data-path files hash identically at `66901577`, `f609976d`, `6a4b8b08` and `46a9c3d0` (verified). Codex r1 Q2: no other exp_14 change (names.py, run contract, generator) affects runtime sampling.

## 5. Composition with yaw_aug and fa_invariant

**Training side: composes without code changes. Code location: a data-side port is required** (no single commit carries both halves).

- The mechanism is data-side; `yaw_aug` acts on collated metadata in `training_step`, `fa_invariant` in `_compute_conditioning` → `invariant_conditioning`; neither depends on which or how many contexts were drawn (the yaw schema guard needs trailing dim 3 only). At 2×32 the yaw-aug key set `(42, global_step, global_rank, index)` is the 100 % run's; the subset changes which sample receives a key, not the keys (Codex r1 Q4) — valid only for monolithic, uninterrupted step numbering at 2 ranks × 32.
- `restrict_to_split` exists only on `exp-14-data-curve` (diverged from `localization-exp` at `824c2b68`); `yaw_aug` and `training.frame_avg_max_fwd_samples` only on `localization-exp`. **Choice: cut from `localization-exp`, port exp_14's data side** — `AR_md.py` was byte-identical at `7bbd8aa` and our base (`4846b653…`), so exp_14's `AR_md.py` (`fb2f6882…`) drops in verbatim; `dataset.py` takes exp_14's two hunks (+7); data files byte copies.
- **Training-path audit since each reference commit (NIT 2 corrected):** model/conditioning changes are gated by keys our arms do not set (`training.canonicalize` exp_25, `training.are_lambda` exp_16) or exercised identically (`training.yaw_aug` exp_15 — and refused together with `fa_invariant` by `_parse_yaw_aug_config`; the chunk-cap key exp_14-FLAC, default-preserving when absent, set deliberately for B-F). The exp_16 `dataset.py`/`utils.py` changes are **unconditional but default-preserving**: `RandomTimeShift.last_shift` is read back after the existing draw (no extra RNG, audio untouched) and `info["time_shift"]` is published but read by no conditioner (exp_25 `depth_canon_training_path_diff_audit.md`).
- **B-F chunk plan (ann. 06).** Legacy per-angle = one ViT forward (one train-mode DINOv3 RoPE draw, HF `augment_patches_center_coordinates`, a `(1,)` draw per call) per orbit angle. Today's default cap 64 at micro 32 gives `max(1, 64 // 32) = 2` angles per chunk → {90°,180°} share a draw. `frame_avg_max_fwd_samples` = per-rank micro → 1 angle per chunk = the loop's call structure (`_orbit_average_batched` `yaw_rotation.py:523`, reference loop `:504`; ascending order; additions not reassociated; masks from the base pass; autocast scope unchanged; non-reentrant checkpointing preserves RNG). `drop_last=True` (`dataset.py:457`) → no tail batch in training. Codex r1 Q3 agrees semantically; the equivalence is **proven by a seeded training-mode test** (§10 B4) and the CUDA bit-identity is not claimed. Config: `FLAC_AR_BF.json` + exactly `training.frame_avg_max_fwd_samples: 32` (`FLAC_AR_BF_cap32.json`; `_cap8` exists only for the opt-in 8×8 rung).

## 6. Run matrix — monolithic only

| run id | model config | dataset config | items | final ckpt |
|---|---|---|---|---|
| `de_yaw_f025` / `_f050` / `_f075` | `exp_15…/FLAC_AR_YAWAUG.json` (`b14caae2…`, in place) | `acousticroom_train_frac{025,050,075}.json` | 73,473 / 145,665 / 218,463 | `epoch=34-step=40000.ckpt` / `epoch=17-…` / `epoch=11-…` |
| `de_bf_f025` / `_f050` / `_f075` | `exp_38…/FLAC_AR_BF_cap32.json` | same | same | same |
| `de_p1_f025` (bridge; MANDATORY only if any primary run trains on a platform other than this box, D2) | `exp_07…/FLAC_AR_BVp1.json` (`733ca52b…` = dc_van's) | `acousticroom_train_frac025.json` | 73,473 | `epoch=34-step=40000.ckpt` |

Run ids follow exp_14's (`de_…`; never `exp14_`, `gen_model_comparison.py:207`); a non-default rung appends `_r8x8`; a crash-resumed run is marked in its run dir (`RESUMED_AT_<step>`) and is **not primary** unless Yixun admits it (Codex r1 BLOCKING 2). Checkpoint names: `epoch=(step−1)//steps_per_epoch` with 1,148 / 2,276 / 3,413 steps/epoch (matches dc_van's dir, verified).

**Literal argv (A6000)** — dc_van's argv shape (its at-launch record) minus exp_14's `--run-contract-json`, plus explicit recipe-defining defaults (NIT 3):

```
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0,1 python train.py \
  --model-config <arm config> --dataset-config src/configs/dataset_configs/AR/train/acousticroom_train_frac<TAG>.json \
  --pretransform-ckpt-path weights/FLAC/VAE.safetensors \
  --max-steps 40000 --batch-size 32 --accum-batches 1 --num-workers 6 --seed 42 \
  --num-gpus 2 --num-nodes 1 --strategy ddp_find_unused_parameters_true --sync-batchnorm true \
  --precision bf16-mixed --val-every -1 --gradient-clip-val 0.0 \
  --logger none --checkpoint-every 2500 --name <run> --experiment-name <run> --save-dir <run dir>
```
(`--pretrained-ckpt-path` / `--remove-pretransform-weight-norm` / `--ckpt-path` stay at their empty ini defaults, checked by the parsed-args test.) Neuronic: the same argv under `torchrun --standalone --nnodes=1 --nproc_per_node=<NGPU>`, `SLURM_NTASKS SLURM_JOB_NAME SLURM_PROCID SLURM_LOCALID SLURM_NODEID` unset before torchrun (exp_15 line 1043; `train.py:100-101` would otherwise add SLURM_PROCID to the seed), their values recorded first.

**Chunk plan record (ann. 06):** B-F: C4 (4 angles, 3 non-zero), per-rank micro 32, cap 32 → 1 angle per chunk, 0 shared RoPE draws = B-F 100 %; eval cap 64 (eval mode has no draw).

## 7. Platforms — and why the primary curve stays on this box

| | A6000 box (primary) | neuronic 2×L40 × 32 (sensitivity) | neuronic 8×L40 × 8 (opt-in only) |
|---|---|---|---|
| ranks × micro × accum, eff / SyncBN batch | 2 × 32 × 1, 64 / 64 | 2 × 32 × 1, 64 / 64 | 8 × 8 × 1, 64 / 64 |
| RoPE draws per step per ViT forward | 2 (one per rank, 32 samples each) | 2 | 8 (8 samples each) |
| B-F cap → plan | 32 → per-angle | 32 → per-angle | 8 → per-angle (`_cap8`) |
| yaw-aug keys | = YAW 100 % | = YAW 100 % | different keys |
| sampler (DistributedSampler, seed 42) | world 2: pads ⌈n/2⌉·2 − n; global batch t = perm[64t : 64t+64] | identical to A6000 | world 8: pads ⌈n/8⌉·8 − n, different rank partition; same steps/epoch, different epoch-end drops (§10 B4 test records the per-fraction counts) |
| launcher | `python train.py` (PL subprocess DDP) | `torchrun --standalone` (logically the same sampler/seeding at world 2; test-demonstrated, §10 B4) | torchrun |
| GPU / driver / CUDA / cuDNN / NCCL | RTX A6000 (Ampere); recorded at launch | L40 (Ada); recorded at launch | L40 |
| precision / matmul | bf16-mixed; `torch.set_float32_matmul_precision('medium')` (`train.py`) | same | same |
| DiT backend | flash-attn 2.7.4.post1 (importable since Nov 1) | whatever `/n/fs/gatrdp/envs/flac` has — positively asserted, refused if ≠ declared | same |
| ckpt destination | NAS run dir directly | `/n/fs/gatrdp/codespace/FLAC/outputs_FLAC/exp38_data_efficiency/<run>/` → staged, sha-verified copy to the NAS | same |

**Why (Codex r1 BLOCKING 1):** every reference point (P1, dc_van ×3, YAW, B-F) trained on this box at 2 ranks × 32. Any other platform puts a hardware (and possibly backend) change *inside* a curve — at minimum between the 75 % and the 100 % point — and the default allocation in the brief would additionally split the B-F curve across A6000 (25 %) and L40 (50/75 %). The only same-model evidence of how large such shifts can be is exp_11 VANL@40k (8×8 L40) vs P1@40k (2×32 A6000), K=8: 8.048 / 1.022 / 37.32 vs 8.993 / 1.009 / 40.65 (`model_comparison.md:49,71`) — **a warning about the size of compound confounding (rung + platform + code era + backend), not a platform-effect estimate.** A `de_p1_f025` bridge on another platform measures such a compound difference for P1 at 25 % only; it cannot bound method- or fraction-specific interactions. Hence the recommendation (D2-a): **all six runs on this box**; neuronic only as a labelled sensitivity platform (or if Yixun accepts the caveats, D2-b/c).

## 8. Time, disk, destinations

**Rates on this box (2 × A6000 × 32):** YAW 100 %: 1.93–1.97 s/step steady with both cards exclusive and flash (21 h 44 m wall); P1 July 3.86 s/step (shared); dc_van Sept 3.7 s/step (two DDP jobs on the same two cards, SDPA); exp_25 CANON 6–19 s/step beside 100 %-utilization co-tenants; B-F July 12.7 s/step (shared); paper appendix: 40.3 h FLAC / 124.9 h OrbitRIR (3.6 / 11.2 s/step, "cumulative active wall-clock"). B-F/vanilla cost ratio under like conditions 3.1 (paper) – 3.5 (July).

| run (40k) | A6000, exclusive | A6000, shared | 2×L40 ×32 (unvalidated projection) | 8×L40 ×8 (unvalidated projection) |
|---|---|---|---|---|
| YAW | ≈ 22 h (measured exp_17) | ≈ 40–45 h | ≈ 17–25 h | ≈ 11–14 h (exp_11 0.96–1.05 s/step) |
| B-F | ≈ 61–77 h (1.96 s/step × 3.1–3.5) | ≈ 125–141 h | — (pending the 2-rank pilot smoke) | — (pending a pilot) |
| P1 bridge | ≈ 22 h | — | ≈ 17–25 h | — |

**Primary schedule (D2-a), sequential, exclusive:** 3 × 22 h + 3 × (61–77) h ≈ **249–297 h ≈ 10.4–12.4 days** of both cards. Earliest start: both cards free of foreign compute processes — GPU 1 holds host-A's exp_30 evaluations (pids 3599091, 3602030, cwd `~/codespace/flac-fork-attn*`, expected ≈ 04:00 Nov 8), and the FLAC session's REQUEST for exp_36 / exp_37 in `~/codespace/claude_gpu_yield/` still stands: **the main session sequences exp_38 through that handshake** (the launcher only refuses non-idle cards; it never reads or writes the handshake files). Proposed order: B-F f025 → YAW f025 → B-F f050 → YAW f050 → B-F f075 → YAW f075 (a complete pair per fraction as early as possible). If started ≈ 04:00 Nov 8: B-F f025 ETD ≈ Nov 10 17:00 – Nov 11 09:00 EST; all six ≈ Nov 18 16:00 – Nov 20 16:00 EST (exclusive; ≈ 2× longer if shared). Every run's steady-state rate is re-measured from its first 1,000 steps and the ETD re-announced.

**Disk (NIT 4: decimal GB and GiB).** Checkpoint 723,922,667 B; 16 per run = 11.58 GB (10.79 GiB); 6 runs 69.5 GB (64.7 GiB); + bridge 11.6 GB. Waveform dumps 6,337 × 10,240 × 4 B = 259.6 MB (247.6 MiB) per cell: 60 cells 15.6 GB (14.5 GiB); E0 replays (8 cells, §9.3) dump-free; bridge cells +2.6 GB. Local (this box): metric JSON + stream sidecar + cell logs ≈ 2 MB/cell ≈ 0.2 GB; train logs ≈ 13 MB each; farm = symlinks only (repeated NAS reads of a 724 MB checkpoint per cell, ≈ 43 GB read over the campaign — negligible against the CIFS mount). NAS free 14 TB; the launcher requires ≥ 30 GiB free at launch. Neuronic (only if used): ≤ 11.6 GB per run under the save root + a staged copy of the same size during publication; the submitter checks free space ≥ 2 × 12 GB × concurrent runs.

**Destinations (ann. 07).** `/media/diskstation/yixunhu/FLAC/checkpoints/exp38_data_efficiency/<run id>/`. A6000 runs write there directly; neuronic runs are published by a staged, no-clobber copy (`<run>.incoming/` → `sha256sum -c CHECKPOINTS.sha256` → atomic rename) — commands in the neuronic command sheet. Dumps: `/media/diskstation/yixunhu/FLAC/exp38_pred_waveforms/` (ann. 08).

## 9. Evaluation protocol

### 9.1 Evaluator, backend, flags
`eval_FLAC.py` at the exp_38 pin (eval path byte-identical to `d6ca307a` / `5b065801`; the data-side port is inert at eval: no eval config sets `restrict_to_split`; `DatasetContractError` never fires), through the reviewed exp_35 no-flash launcher (`PYTHONDONTWRITEBYTECODE=1 python overlays/no_flash/exp35_no_flash_launch.py eval_FLAC.py …`; exactly one `EXP35_NO_FLASH_LAUNCHER attention_backend=sdpa overlay_sha256=0d0e52ca… launcher_sha256=bbd2847d… target=eval_FLAC.py` line required) → SDPA = the backend of dc_van's, exp_25's and exp_35's cells; on this box only.

Per cell (exp_25 / exp_35 argv shape, verified from `depth_canon_cells.json` and exp_35's dry-run log):
- YAW / P1: `--model-config <the run's training config> --dataset-config src/configs/dataset_configs/AR/eval/acousticroom_unseeneval{,_1}.json --cond-method vanilla --cond-autocast bf16 --cfg-scale 1.0 --steps 1 --batch-size 64 --device cuda --record-per-scene --record-stream --expected-stream-count 6337 --seed <S> --rotate-mode fixed --rotate-deg 0 --eval-name <run>_K<K>_s<S> --ckpt-path <farm link> --dump-scored-waveforms /media/diskstation/yixunhu/FLAC/exp38_pred_waveforms`
- B-F: the same plus `--cond-method fa_invariant --frame-avg-angles 0,90,180,270 --frame-avg-max-fwd-samples 64`.

The driver refuses unless the checkpoint's embedded `model_config` canonically equals `--model-config` and `--cond-method` equals the embedded `training.cond_method` (absent ⇒ vanilla) — announcement 05.

### 9.2 Cells
6 runs × K ∈ {8, 1} × seeds 42–46 = **60 cells** (+ 10 for the bridge if trained). Outputs land beside a per-run farm symlink `outputs_FLAC/exp38_data_efficiency/eval/<run>/<ckpt name>` → NAS checkpoint (eval_FLAC writes beside `--ckpt-path`'s lexical dir, `eval_FLAC.py:629-637`). ≈ 6.5–20 min (vanilla) / ≈ 25 min (B-F) per cell ≈ 20 GPU-h ≈ 10 h on two cards. Evaluation of a run starts as soon as its checkpoints are admitted (both cards are then busy with the next training — so evals run either on a card the training does not use, i.e. never during a primary training run, or between runs: the plan schedules each run's 10 cells right after it, ≈ 1.5–2.5 h of both cards).

### 9.3 E0 replay hard stops (Codex r1 HIGH 7) — run first; the campaign does not start unless all pass
Exact equality of the `metrics` dict (and `by_scene` where both carry it), key sets pinned:

| gate | checkpoint | config | cells | reference (imported into `ref_cells/`, sha-pinned) |
|---|---|---|---|---|
| E0a (bridges the exp_14 evaluator) | dc_van_f025@40k `aa1a2cd6…` | `FLAC_AR_BVp1.json` (= dc_van's) | K8 s42, K1 s42 | exp_14 records (`4136ac75…` K8; K1 likewise) |
| E0b (fa path) | B-F@40k `5319feb4…` | `FLAC_AR_BF.json` | K8 s42, K1 s42 | exp_35 BF-pin T cells (`f167470e…` K8) |
| E0c (YAW path) | YAW@40k `ac1f2603…` | `FLAC_AR_YAWAUG.json` | K8 s42, K1 s42 | exp_25 YAW-pin T cells |
| E0d (P1 100 %) | P1@40k `c4c67882…` | `FLAC_AR_BVp1.json` | K8 s42 | exp_25 P1-pin T cell |

Plus: every E0 and campaign cell's fixed-mode `input_hash` must equal the pinned exp_25 value for its (K, seed) (verified: P1, YAW and B-F agree per (K, seed); e.g. K8 s42 `4c950c1e…`, K1 s42 `37d2f569…`). E0a has no stream sidecar on the exp_14 side; its `input_hash` is checked against the pinned value only. **If E0a is not exact**, the campaign stops and Yixun decides (report-with-labelled-difference vs re-evaluating dc_van ×3 at our pin, 30 cells); no tolerance is invented after the fact. E0 cells are dump-free diagnostics (ann. 08 applies to result cells).

### 9.4 Aggregation, validation and the table
`exp38_collect.py` admits a cell only as one transaction (Codex r1 HIGH 8): metrics schema (exact key set), protocol fields (`cond_method`, `frame_avg_angles`, `frame_avg_fwd_cap`, `cond_autocast bf16`, `rotate_deg 0`, `seed`, `steps 1`, `cfg_scale 1.0`, `batch_size 64`, `n_samples 6337`, `dataset_config`, `eval_name`, `weights_source ema`, `source_sha` = the eval pin), `ckpt_sha256_loaded` = the run's audited final checkpoint, stream sidecar present with `stream_count 6337` and the pinned `input_hash`, the driver's cell receipt (model-config sha, dataset-config sha, launcher sentinel, rc 0), dump `COMPLETE` + `manifest.json` digests + `waveforms.npy` shape `[6337, 10240]` float32 + sha; exactly seeds 42–46 per (run, K), no duplicates. It writes `data_efficiency_results.md` + `.json` (mean ± sample SD of the top-level `metrics` per model × fraction × K) beside the §2 reference rows read from `ref_cells/` (manifest-checked). Incomplete → PENDING, never a number.

**`model_comparison.md` (Codex r1 HIGH 12, ann. 04):** after each run's complete ten-cell block passes the collector, the same commit adds (a) the raw cell JSONs force-added under `outputs_FLAC/exp38_data_efficiency/eval/<run>/`, (b) their lines in `A6000_METRICS_SHA256SUMS.txt`, (c) two row specs in `gen_model_comparison.py` (`("data-eff YAW f=25% @40k (exp_38)", "vanilla eval", K, ["outputs_FLAC/exp38_data_efficiency/eval/de_yaw_f025/*_K{K}_s4[2-6].json"])`, B-F rows `"fa eval"`), then push; the regeneration itself runs on neuronic (the CLAUDE.md table-regen trap: the generator refuses on this box), so each push carries a one-line regen request for Yixun.

## 10. Implementation (per file; TDD)

### Round A — port exp_14's data side (production data path)
- A1 `src/tests/test_ar_md_restrict_context.py` — byte copy (`5f63591e…`; hermetic toy tree; B1 bitwise default path, B2–B9 restriction / fail-closed / worker + process propagation incl. relative-vs-absolute split paths). RED on the base (ImportError).
- A2 `src/data/dataset.py` (+7, exp_14's two hunks) and `AR_md.py` (byte copy `fb2f6882…`). GREEN.
- A3 data files (3 splits, manifest + `.sha256`, 3 dataset configs; byte copies) + `src/tests/test_exp38_subset_port.py`: file hashes == exp_14 blobs (self-contained constants, no git refs — NIT 6), manifest `.sha256` consistent, counts/rooms/nesting, config diff == split path + flag, re-raise order.
- A4 `exp38_subset_audit.py` (CPU, reads the on-disk AR tree read-only; tests on toy trees) — the exhaustive identity audit (Codex r1 HIGH 2): per fraction (i) the complete **ordered** target list `SampleDataset.filenames` built by our `create_dataloader_from_config` path, its sha256, count, and its equality with the list built by exp_14's own `json_scandir` (its function text is unchanged; the audit also hashes the function source of both commits from the pinned blob constants); (ii) for EVERY target, AR_md's unrestricted candidate pool (same receiver, other sources, `S00{int}` rebuild, `os.path.exists`) and the restricted pool (∩ split basenames of that room) — zero empty restricted pools, the per-room 0 / 1–7 / ≥ 8 histograms equal to the committed manifest's, restricted ⊆ unrestricted, every removed candidate ∉ split; (iii) for `train.json` (100 %, unrestricted) the count of on-disk candidates outside `train.json` (the disclosed ≈ 1.8 %); writes `data_efficiency_subset_audit.json` (+ `.sha256`), committed; the launchers refuse unless it is present, PASS and bound to the current split/config/AR_md hashes. Base-commit (`2b42468c` / `7bbd8aa`) blob checks live here, not in pytest.

### Round B — the exp_38 kit (`worklog/worklog_yixun/exp_38_data_efficiency_claude/`; tests `src/tests/test_exp38_*.py`)
- B1 `exp38_registry.py` — single source of truth (constants, configs, chunk plan, run ids, argv, legs removed, run-dir states, checkpoint names) + CLI `list | check | argv | smoke-argv | run-dir-check | audit-ckpt | ckpt-manifest | pins | log-gate | snapshot | dry-run`. Tests `test_exp38_registry.py`: constants vs disk; committed `_cap32/_cap8` == regenerated bytes and == base + one key; factory passes the cap; argv == dc_van's at-launch argv with only the registered differences (token-for-token) and its parsed `prefigure` values (`num_nodes 1`, `val_every -1`, `gradient_clip_val 0.0`, empty `ckpt_path` / `pretrained_ckpt_path`); 8×8 refused without the opt-in; per-run argv; run-dir fresh / resume / foreign entries; log gates.
- B2 `exp38_registry.py::audit_ckpt` (Codex r1 HIGH 6) — ONE read of the bytes (sha256 + `torch.load(BytesIO)`): name, `global_step`, `epoch == (step−1)//spe`, embedded config canonical-equal, EMA `diffusion_ema.ema_model.*` suffix set == online `diffusion.model.*` suffix set with equal shapes/dtypes + `diffusion_ema.step` present, one optimizer with non-empty state and AdamW groups (lr == scheduler `_last_lr`, betas (0.9, 0.999), wd 1e-3, `initial_lr` 5e-5), scheduler `InverseLR(1e6, 0.5, 0.99)`, `last_epoch == step` and `_last_lr` == the closed-form value, fit-loop `optim_progress.optimizer.step.total.completed == step`, `ModelCheckpoint` state `dirpath` == the run dir and `every_n_train_steps` 2500. Tests on synthetic checkpoints (one per failure) + a real-data readback on dc_van_f025's step-40000 checkpoint (skipped if the NAS is absent).
- B3 `exp38_init_digest.py` (CPU) — seed-42 initial model + wrapper exactly as `train.py` builds it (seed_everything 42 → `create_model_from_config` → VAE load → `create_training_wrapper_from_config`) → sha256 over the ordered state_dict + parameter inventory; asserts YAW == B-F == P1 initial digests (same architecture and seed; paper's "same initialization") and records it in the at-launch record. Test: two builds equal; a perturbed config differs.
- B4 tests of recipe mechanics (no new production code): (a) `test_exp38_bf_orbit_equivalence.py` — the real B-F conditioner (DINOv3-S from the pinned HF cache, CPU, fp32, train mode, B = 2, non-reentrant checkpointing on) under identical seeded RNG: `_orbit_average_loop` vs `_orbit_average_batched(cap=B)` → bit-equal outputs, masks, post-call RNG states (torch CPU), loss and parameter gradients; spy: exactly one B-sample call per non-zero angle; the default cap 64 at B = 32 shares two angles (the reason for the key). (b) `test_exp38_sampler.py` — `DistributedSampler(seed 42)` index traces: world 2 × 32 global batch t == perm[64t : 64t+64] for both rank-launch styles (rank/world from PL's subprocess env vs torchrun env give the same sampler kwargs); per fraction the padding and per-rank drop counts at world 2 and 8 (recorded in the plan's §7 table by the test's output).
- B5 `exp38_backend_probe.py` (Codex r1 BLOCKING 4) — positive backend proof in the training env: interpreter path, `flash_attn.__file__` + `flash_attn.__version__` (pinned 2.7.4.post1 on this box; neuronic's recorded and must equal `EXPECT_FLASH_VERSION`), `src.models.transformer.flash_attn_func is not None` (flash) / `is None` with the overlay resolved (sdpa), and on GPU one bf16 forward of the arm's real DiT with a counting wrapper around `src.models.transformer.flash_attn_func` and `F.scaled_dot_product_attention`: flash ⇒ flash calls > 0 and SDPA calls == 0 in the DiT; sdpa ⇒ flash calls == 0. Run in every SMOKE and in the FULL preflight (cards idle; ≈ 1 min); a mismatch refuses (silently falling back is a recipe change). Tests: CPU parts with monkeypatched specs.
- B6 `exp38_train_local.sh` — A6000 launcher: `MODE=DRY_RUN | SMOKE (1 rank, GPU 0, storage-light) | SMOKE_DDP (2 ranks × 32, SyncBN, 30 steps, storage-light, the production-topology smoke — Codex r1 HIGH 5) | FULL | RESUME (ALLOW_RESUME=1; marks the run non-primary)`. Gates: interpreter, registry `check`, subset-audit artifact, pins + backend probe, init digest, GPU idle (default) / shared floor, EXPECT_SHA == HEAD and tracked tree clean (FULL/RESUME/SMOKE_DDP), FULL requires a PASS `SMOKE_DDP` evidence for the same run id at the same HEAD, NAS mount + write probe + ≥ 30 GiB, run dir fresh, an exclusive run lock (`mkdir <run dir>/.exp38_lock`, holder pid + host + ts; refused if present). Before the fork: the at-launch record (argv, HEAD, file shas, host, GPU names/UUIDs, driver/CUDA/cuDNN/NCCL/torch/PL/transformers/flash-attn versions, matmul/TF32 flags, CPU count, rank env, co-tenants, `pip freeze` sha, backend probe, init digest) and an **immutable tool snapshot** `<run dir>/.exp38_tools/` (registry + configs + dataset config + split + audit artifact, sha-listed); the start watcher, end gate and checkpoint audits execute the snapshot (`--repo <snapshot>`), never the live checkout (Codex r1 HIGH 10). Launch `setsid python train.py …` → log; start watcher (kills ONLY our process group on failure); end gate (Lightning's marker; ≥ 50 parsed finite `train/loss` values and none non-finite; banner / members / dataset lines; exit statuses of train.py and the watcher recorded); cadence set {2500…40000}; final audit; `CHECKPOINTS.sha256`. Guard tests (`test_exp38_launchers.py`) drive the real script with `MODE=DRY_RUN` and synthetic environments (fake `nvidia-smi` on PATH, temp NAS root via the DRY_RUN-only hook) for every refusal.
- B7 `exp38_train.sbatch` + `exp38_submit_neuronic.sh` — neuronic, **one monolithic job per run** (no chain; Codex r1 BLOCKING 2/3): the submitter refuses unless HEAD is pushed and clean, takes the per-run submission lock atomically (`mkdir <run dir>/.exp38_submitted` with the intent token, then `sbatch --parsable`, then the job id written inside; a second submission for the same run dir is refused by construction; recovery after a failed job = archive the run dir, new submission), records every line in `exp38_neuronic_submissions.log`; `SMOKE=1` submits the 2-rank 30-step pilot (rate + memory, storage-light; required before FULL on that platform). The worker: all §10 B6 gates adapted (L40 × NGPU, 1 node / 1 task, rank-env record then unset, exclusive run lock, snapshot tools, backend probe, init digest), torchrun, watcher, end gate, audit, manifest. `DRY_RUN=1` / `EXP38_SBATCH_DRYRUN=1` print everything and submit/spawn nothing.
- B8 `exp38_eval.py` — `plan` (60 (+10) cells + E0 rows with full argv, JSON), `prepare <run>` (sha256 of the NAS final checkpoint == `CHECKPOINTS.sha256` line, `audit-ckpt`, farm symlink), `run --gpu G --cells …` (sequential, a per-GPU `flock`, refuses a non-idle card unless `--shared`, skip only if the cell validates, per-cell log + receipt), `e0` (verdict JSON). Tests `test_exp38_eval.py` (argv per arm, cond-method binding refusal, skip-only-if-valid, E0 exactness, input-hash pins).
- B9 `exp38_collect.py` + `ref_cells/` (imported reference JSONs + sha manifest) — §9.4. Tests `test_exp38_collect.py` (transactional admission per failure mode, aggregation, golden table, PENDING).
- B10 records: `data_efficiency_worklog.md`, `commits_data_efficiency.md`, `data_efficiency_command.md`, `data_efficiency_params_set_up.md` (at launch), `data_efficiency_neuronic_commands.md` (Yixun's sheet: worktree at the pin, SMOKE, FULL, staged publication to the NAS).

Commits < 200 changed code lines, one per red→green cycle, SHAs in `commits_data_efficiency.md`; `git pull --rebase` before every push.

## 11. Validation ladder

1. Static: `py_compile`, `bash -n`, `json.load`, `git diff --check`.
2. pytest: all new tests + the existing suites touching the changed / used code (`test_yaw_aug_*`, `test_frame_avg_cap_config.py`, `test_exp11_orbit_configs.py`, `test_train_*`, `test_exp35_no_flash_launch.py`, `test_eval_*`, dataset tests) — green; pre-existing GPU-only failures listed.
3. Subset audit (CPU, real AR tree): `exp38_subset_audit.py` → PASS artifact committed.
4. CPU init digest (B3) and the CPU training-mode orbit test (B4a) on the real DINOv3 weights.
5. `DRY_RUN` of every launch path: `exp38_registry.py dry-run --platform a6000|neuronic`, `MODE=DRY_RUN exp38_train_local.sh` for each run, `DRY_RUN=1 exp38_submit_neuronic.sh` and `EXP38_SBATCH_DRYRUN=1` for each run, `exp38_eval.py plan`.
6. GPU-0 single-rank smoke (≤ 15 min total, Yixun-allowed now): `MODE=SMOKE` for `de_yaw_f025` and `de_bf_f025` (20 steps, micro 32, no checkpoint): banner, finite loss, backend probe, peak memory (B-F expect ≈ 15–16 GiB/rank, exp_07 M1 15,712 MiB; paper 15.3 GiB), single-rank rate (lower bound only — a single rank promotes nothing).
7. `MODE=SMOKE_DDP` per run on both cards — at launch time, when the cards are idle (not now: GPU 1 is busy).
8. FULL — not started by the Coder; after Yixun's decisions (§13) and the Codex code review(s).

## 12. Launch acceptance criteria (per run)

HEAD == the pushed launch SHA, tracked tree clean, launched from a dedicated run worktree at that SHA (`~/codespace/exp-38-run`, never committed in while a run is live); 1 node; 2 ranks (`MEMBER: 1/2`, `2/2`); micro 32; accum 1; `--sync-batchnorm true` (no runtime print exists for the SyncBN conversion; evidence = argv + `test_train_sync_batchnorm.py`); `val_every −1`; clip 0.0; bf16-mixed; both ViT checkpoint wrappers on (config + `test_vit_gradient_checkpointing.py`); optimizer / scheduler hyper-parameters (config check + final audit); initial-model digest == the registered one; dataset line == the fraction's count / 243 on every rank; YAW banner before step 0; B-F config embeds cap = micro; backend probe = declared; ≥ 1 optimizer step with finite loss, then ≥ 50 parsed finite losses by the end and none non-finite; checkpoints exactly every 2,500 steps; the `step=40000` audit (B2) passes; `CHECKPOINTS.sha256` written; exit statuses recorded. Monolithic, no resume (else marked non-primary).

## 13. Decisions for Yixun (with recommendations)

- **D1 — rung.** 2 ranks × 32 everywhere (recommended); 8×8 only as an explicitly labelled cross-recipe run (changes RoPE-draw granularity, yaw-aug keys and sample order).
- **D2 — platform / allocation.** (a) **all six on this box, sequential, exclusive** (recommended — the only design with every point of each curve, incl. the 100 % anchors and dc_van, on one platform and rung; ≈ 10.4–12.4 days of both cards); (b) B-F × 3 here (the B-F curve stays single-platform with its anchor) + YAW × 3 on neuronic 2×L40 with the **mandatory** `de_p1_f025` platform+backend+code-era bridge, the YAW 25–75 % points labelled L40-trained (≈ 7.6–9.6 days here); (c) the brief's default split — not recommended (the B-F curve would mix A6000 and L40); if chosen, at minimum add B-F f025 on 2×L40 so each new curve's 25–75 % points share a platform, the bridge mandatory, all A6000/L40 comparisons labelled sensitivity.
- **D3 — monolithic only.** Crash-resumed runs are not primary unless Yixun admits them (disclosed like P1's own 32.5k resume). On neuronic this requires a walltime ≥ the run (YAW ≈ 1 day; B-F ≈ 2.5–3.5 days at 2×L40, pending the pilot).
- **D4 — DiT backend for training.** flash (recommended; = the documented P1/B-F backend and the inferred YAW one); SDPA would match dc_van's training. The probe refuses a mismatch either way.
- **D5 — GPU policy here.** idle (recommended), sequenced by the main session through the `claude_gpu_yield` handshake; shared only with Yixun's go.
- **D6 — eval.** §9 as written; on an E0a mismatch, report the labelled difference or re-evaluate dc_van × 3 at our pin (30 cells) — Yixun's call at that moment.

## 14. Risks and disclosures

- One training seed per cell; matched steps, not epochs or compute.
- Historical asymmetries (Codex r1 Q10): dc_van trained SDPA (ours flash, D4); different code eras (§5 audit); P1 100 % resumed mid-run; YAW 100 %'s launch source and training backend unknown; any non-A6000 platform (D2). The E0 replays remove evaluation-side doubt only, never training-path confounding.
- Concurrent writers: `localization-exp` moves; runs launch from a frozen run worktree at the pushed pin; a closure change after a launch = a new run, never a resume.
- Never edit a running bash script (bodies are single functions called on the last line); never touch another session's processes or the handshake files; the Coder starts no real training.

## 15. Changelog r0 → r1 (Codex plan review r1, verdict REVISE)

| Finding | Disposition |
|---|---|
| B1 platform confounding | §7 rationale + D2: primary = all six on this box; split options require single-platform curves + mandatory bridge, labelled sensitivity |
| B2 mid-epoch chaining | chains removed; monolithic only; resumed runs marked non-primary (§6, §10 B6/B7, D3) |
| B3 submit-exactly-once | one job per run; atomic per-run submission lock + exclusive run lock; recovery = archive + resubmit (§10 B7) |
| B4 flash gate fail-open | positive backend probe incl. a counted DiT forward (§10 B5) |
| H1 YAW source/backend | marked unknown/inferred; claim scoped (§3) |
| H2 subset proof | exhaustive subset audit artifact required by the launchers (§10 A4) |
| H3 B-F equivalence | seeded training-mode loop-vs-cap test on the real conditioner (§10 B4a) |
| H4 sampler / launcher | sampler-trace test + per-fraction padding/drop record (§10 B4b, §7) |
| H5 production-topology smoke | `SMOKE_DDP` / neuronic `SMOKE=1` required before FULL (§10 B6/B7, §11) |
| H6 checkpoint admission | single-read full-state audit + initial-model digest (§10 B2/B3) |
| H7 E0 coverage / tolerance | E0a–E0d at both K, exact equality, input-hash pins (§9.3) |
| H8 collector binding | transactional admission + imported, manifest-checked references (§9.4, §10 B9) |
| H9 resources | aggregate / local / NAS / staged-copy budgets and checks; pilot before walltime (§8, §10 B7) |
| H10 late-bound tools | immutable run worktree + snapshot tools executed for audits (§10 B6, §12) |
| H11 acceptance completeness | §12 rewritten; explicit argv defaults; finite-loss count; exit statuses |
| H12 table publication | per-run transaction + row specs; regen on neuronic (§9.4) |
| H13 implementation during review | acknowledged (header); drafts provisional, reconciled, reviewed as code |
| N1–N6 | ancestry wording (§2), gated vs default-preserving (§5), explicit defaults (§6), GB/GiB (§8), 8×8 timing relabelled (§8), self-contained pytest (§10 A3/A4) |
