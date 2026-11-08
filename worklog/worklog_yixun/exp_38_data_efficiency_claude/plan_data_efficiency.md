# plan_data_efficiency — exp_38: YAW and OrbitRIR (B-F) data-efficiency curve at 25 / 50 / 75 % of AR

**Author:** Claude Opus 5.5 (Coder/Planner subagent, max effort) · **Date:** 2026-11-07 · **Status:** **r2** — Codex plan reviews r1 (REVISE) and r2 (REVISE) folded (§15); plan reviews used 2 of 3. r0/r1 in `plan_history/`. **Branch / worktree:** `exp-38-data-efficiency` in `~/codespace/exp-38-data-efficiency`, cut from `localization-exp` `2b42468c8f429b1d93da93b3e4ddb7bd6f8f1932`. `main` is never touched.

"Verified" = re-checked by the Planner in this session; "inferred" = argued from records; "unknown" = not recoverable from any surviving record.

**Process (Codex r1 HIGH 13, r2 HIGH 12):** implementation drafts and the round-A commits (`fac501b0`, `4187ef77`) were made while plan reviews r1/r2 ran. They are provisional: after this r2 revision they are reconciled with it, and the ENTIRE exp_38 diff (`2b42468c..HEAD`) goes to the bounded code reviews. Nothing launches before that.

---

## 1. Question and estimand

**Question:** how do Yaw-Augmented FLAC (YAW) and OrbitRIR (B-F, `fa_invariant` C4 frame averaging) degrade when trained on 25 / 50 / 75 % of the AR training split, next to vanilla FLAC (P1), whose curve exists (exp_14 `dc_van_f025/f050/f075` + P1@40k)?

**Estimand.** For `m ∈ {YAW, B-F}`, `f ∈ {0.25, 0.50, 0.75}`, `K ∈ {8 (headline), 1}`: the 40,000-step EMA endpoint of ONE monolithic, never-resumed training run (seed 42), evaluated on the full unseen split (6,337 items / 17 rooms; ann. 01): T60 %, C50 dB, EDT ms (+ FD, R@k), split-level item-weighted (§9.4), mean ± sample SD over eval seeds 42–46. **Mathematical recipe per fraction = the model's own 100 % recipe with exp_14's subset dataset config**; operational differences from the 100 % runs (logger none vs W&B — no numeric effect, initialization identical by the init receipt, §10 B3; source era; environment) are listed in §14 (Codex r2 NIT 2).

**Framing (Codex r1 Q10):** a historical matched-subset, fixed-optimizer-step-budget comparison. One training seed per cell (eval seeds measure eval variability only; exp_14's ≈ ±0.5 T60 step-to-step band, `data_curve_results.md:3`). Effective epochs 34.8 / 17.6 / 11.7 / 8.8 at 25 / 50 / 75 / 100 %.

## 2. What already exists (reused, never re-run)

| Curve point | Source | K=8 T60 / C50 / EDT (item-weighted, mean ± SD, s42–46) | Evaluator / eval backend |
|---|---|---|---|
| FLAC 25 / 50 / 75 % | exp_14 `dc_van_f025` (`aa1a2cd6…`) / `_f050` / `_f075` | 10.252±0.012 / 1.048±0.002 / 42.914±0.058 · 9.450±0.010 / 1.044±0.002 / 41.587±0.064 · 8.913±0.009 / 1.104±0.002 / 41.636±0.065 | exp_14 evaluator (`66901577` lineage), SDPA |
| FLAC 100 % | P1@40k `c4c67882…`, exp_25 P1-pin | 8.993±0.011 / 1.009±0.004 / 40.650±0.103 | exp_25 pin `d6ca307a`, SDPA |
| YAW 100 % | YAW@40k `ac1f2603…`, exp_25 YAW-pin | 7.965±0.015 / 1.013±0.002 / 39.923±0.058 | `d6ca307a`, SDPA |
| B-F 100 % | B-F@40k `5319feb4…`, exp_35 BF-pin T | 8.202±0.017 / 0.978±0.002 / 38.794±0.074 | exp_35 pin `5b065801`, SDPA (no-flash overlay) |

Recomputed by the Planner from the raw JSONs (they reproduce Table 3(b); 75 % EDT 41.640 = 2-dp of 41.6362). Every reference JSON the collector reads is imported into `ref_cells/` with a sha256 manifest (the exp_35 B-F cells are tracked on no branch).

**Evaluator identity:** `git diff 5b065801 2b42468c` and `git diff d6ca307a 2b42468c` over `eval_FLAC.py src/{models,data,inference,metrics,training}` are empty (verified); `overlays/no_flash/` is absent at `d6ca307a`, present at `5b065801` and our base.

## 3. The 100 % recipes (verified from each checkpoint's embedded `model_config` and the records)

| | P1 (FLAC) | YAW | B-F (OrbitRIR) |
|---|---|---|---|
| ckpt sha256 | `c4c678826cdd…` | `ac1f26034e4f…` | `5319feb4af87…` |
| embedded config == | `exp_07…/FLAC_AR_BVp1.json` (`733ca52b…`) | `exp_15…/FLAC_AR_YAWAUG.json` (`b14caae2…`) = BVp1 + `training.yaw_aug {true, 512, 42}` | `exp_07…/FLAC_AR_BF.json` (`3e677204…`) = release + `cond_method fa_invariant`, C4 angles |
| `.model` sections | canonically identical across the three (Codex r2 verified); only training dispatch keys differ | | |
| experiment | exp_07 P1 | exp_17 yaw_aug_a6000, grad-ckpt-ON arm (not exp_15's 8×L40 arm; not exp_17's retired OFF variant) | exp_07 B-F |
| rung | 2×A6000, 32/GPU × 2 × accum 1, `ddp_find_unused_parameters_true`, SyncBN (BN 64), ViT grad-ckpt on | same | same |
| precision / seed / cadence | bf16-mixed / 42 / 2,500 | same | same |
| lineage | 67.5k budget, **resumed once at 32.5k** (`e50d098` → `1b56e7a`) | monolithic 40k, 21 h 44 m exclusive (`42cbddaf`; 1.93–1.97 s/step) | 67.5k budget, monolithic, stopped 40,249 |
| training commit | `e50d098` / `1b56e7a` | **unknown** (the removed `exp-17-yawaug-a6000` worktree's launch HEAD is in no surviving file; `41aa31dc`'s closure is the best inference) | `f59f5a4` era |
| FA chunk plan | n/a | n/a | legacy per-angle loop (pre-`1479304`) |
| DiT backend | flash-attn (documented, `6596121`) | **unknown** (inferred flash: same-env eval logs of Aug 16 lack `flash_attn not installed`; training log lost) | flash-attn (documented) |

`--max-steps 40000` is schedule-equivalent (`InverseLR` reads only its own parameters and the step) to the 67.5k-budget runs' 40k checkpoints, not trajectory-equivalent to P1's resumed history. The YAW "own recipe" claim is scoped to what is bound (embedded config, rung, precision, seed, budget, cadence).

## 4. How exp_14 built and applied the subsets (verified)

Built: nested per-room stratified random subsets of `data/AR/train.json` (291,210 / 243 rooms, `aa4e52d6…`), seed 2026, eligibility `sampler_faithful_S00int_v1`, top-ups so no restricted context pool is empty; manifest `train_frac_manifest_s2026.json` (`553534ea…`).

| tag | split (sha256) | items | rooms | ctx 0 / 1–7 / ≥ 8 |
|---|---|---|---|---|
| 025 | `train_frac025_s2026.json` (`ce33f34f…`) | 73,473 | 243 | 0 / 16,890 / 56,583 |
| 050 | `train_frac050_s2026.json` (`bf330846…`) | 145,665 | 243 | 0 / 14,303 / 131,362 |
| 075 | `train_frac075_s2026.json` (`57388ce8…`) | 218,463 | 243 | 0 / 17,430 / 201,033 |

Applied: `acousticroom_train_frac{025,050,075}.json` (`7ac8243f…` / `b74687f7…` / `cc0e5fb8…`) = `acousticroom_train.json` + split path + `acoustic_context.restrict_to_split: true`; `AR_md.py` filters the same-receiver context candidates to the split's basenames of that room; `DatasetContractError` is re-raised instead of resampled. **New fact from the exp_38 subset audit (§10 A4, PASS, 2026-11-07):** at 100 % the unrestricted pools contain **zero** candidates outside `train.json` — the seen-eval hold-outs are whole receivers, and pools are same-receiver — so the 100 % anchors' context pools were effectively split-restricted too (exp_14's "≈ 1.8 % larger room folders" never reached a pool).

## 5. Composition with yaw_aug and fa_invariant

Training side composes without code changes; a data-side port is required because no commit carries both halves. `restrict_to_split` lives only on `exp-14-data-curve`; `yaw_aug` and `training.frame_avg_max_fwd_samples` only on `localization-exp`. Choice: cut from `localization-exp`, port exp_14's data side (`AR_md.py` byte copy — it was byte-identical at `7bbd8aa` and our base; `dataset.py` +7; data files byte copies).

Training-path audit since each reference commit: model/conditioning changes are gated by keys our arms do not set (`canonicalize`, `are_lambda`) or exercised identically (`yaw_aug`; the chunk-cap key, set deliberately for B-F); the exp_16 `dataset.py`/`utils.py` changes are unconditional but default-preserving (`last_shift` read after the existing draw; `info["time_shift"]` consumed by no conditioner).

B-F chunk plan (ann. 06): legacy per-angle = one ViT forward (one train-mode DINOv3 RoPE `(1,)` draw) per orbit angle. Default cap 64 at micro 32 → 2 angles per chunk ({90°,180°} share a draw). `frame_avg_max_fwd_samples` = per-rank micro → 1 angle per chunk = the loop's call structure; `drop_last=True` → no tail batch. Config `FLAC_AR_BF_cap32.json` = `FLAC_AR_BF.json` + exactly that key (`_cap8` only for the opt-in 8×8 rung). Proven by a seeded CPU training-mode test (§10 B4a) and observed on the GPU by the run-evidence callback (§10 B3: conditioner call shapes in the first step); CUDA bit identity not claimed.

## 6. Run matrix — monolithic, never resumed

| run id | model config | dataset config | items | final ckpt |
|---|---|---|---|---|
| `de_yaw_f025` / `_f050` / `_f075` | `exp_15…/FLAC_AR_YAWAUG.json` (`b14caae2…`) | `acousticroom_train_frac{025,050,075}.json` | 73,473 / 145,665 / 218,463 | `epoch=34-` / `epoch=17-` / `epoch=11-step=40000.ckpt` |
| `de_bf_f025` / `_f050` / `_f075` | `exp_38…/FLAC_AR_BF_cap32.json` | same | same | same |
| `de_p1_f025` (bridge; MANDATORY only under D2-b/c) | `exp_07…/FLAC_AR_BVp1.json` (= dc_van's) | `acousticroom_train_frac025.json` | 73,473 | `epoch=34-step=40000.ckpt` |

**Crash policy (Codex r2 BLOCKING 2):** a primary run that crashes is archived (its run dir renamed `<run>.failed_<ts>`) and **re-run from step 0**. A crash-resumed continuation, if ever made for diagnostics, gets the distinct run id `<run>_resumed<step>`, a `RESUMED_AT_<step>` marker, a sensitivity label in every table, and can never replace a primary cell (PL restores neither the iterator nor all RNG streams). No exception path.

Run ids follow exp_14's (`de_…`; never `exp14_`); checkpoint names `epoch=(step−1)//spe-step=<step>.ckpt` with 1,148 / 2,276 / 3,413 steps/epoch (matches dc_van's dir).

**Literal argv (A6000):**
```
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0,1 python train.py \
  --model-config <arm config> --dataset-config src/configs/dataset_configs/AR/train/acousticroom_train_frac<TAG>.json \
  --pretransform-ckpt-path weights/FLAC/VAE.safetensors \
  --max-steps 40000 --batch-size 32 --accum-batches 1 --num-workers 6 --seed 42 \
  --num-gpus 2 --num-nodes 1 --strategy ddp_find_unused_parameters_true --sync-batchnorm true \
  --precision bf16-mixed --val-every -1 --gradient-clip-val 0.0 \
  --logger none --checkpoint-every 2500 --name <run> --experiment-name <run> --save-dir <run dir> \
  --evidence-json <run dir>/evidence/run_evidence.json
```
= dc_van's argv shape minus exp_14's `--run-contract-json`, plus explicit recipe defaults and the exp_38 observe-only evidence flag (§10 B3). Neuronic: the same under `torchrun --standalone --nnodes=1 --nproc_per_node=<NGPU>`, the SLURM rank variables recorded then unset.

## 7. Platforms — the primary curve stays on this box

| | A6000 box (primary) | neuronic 2×L40 × 32 (sensitivity) | neuronic 8×L40 × 8 (opt-in only) |
|---|---|---|---|
| ranks × micro × accum; eff / SyncBN batch | 2 × 32 × 1; 64 / 64 | 2 × 32 × 1; 64 / 64 | 8 × 8 × 1; 64 / 64 |
| RoPE draws per ViT call per step | 2 (32 samples each) | 2 | 8 (8 samples each) |
| B-F cap → plan | 32 → per-angle | 32 → per-angle | 8 → per-angle (`_cap8`) |
| yaw-aug keys | = YAW 100 % | = YAW 100 % | different |
| sampler (corrected, Codex r2 HIGH 5) | `DistributedSampler` (seed 42, set_epoch e) pads ⌈n/2⌉·2 − n = 1 / 1 / 1 at 25 / 50 / 75 %; rank lists are strided (rank r: perm[r::2]); global step t = the rank-interleaving of perm[64t : 64t+64] | identical | pads 7 / 7 / 1 (rank r: perm[r::8]); the same unique prefix is consumed and the same unique suffix dropped (1 / 1 / 31 items per epoch) — padded entries fall in the dropped tails; differences are rank assignment, yaw keys, worker streams and RoPE sharing |
| launcher | `python train.py` (PL subprocess DDP) | `torchrun --standalone` (in-process equivalence test, §10 B4b) | torchrun |
| GPU / stack | exactly two "NVIDIA RTX A6000" visible (asserted); versions pinned by the campaign environment receipt | two L40 (asserted) | eight L40 |
| DiT backend | flash-attn 2.7.4.post1 (asserted, counted in-process) | asserted against `EXPECT_FLASH_VERSION` | same |
| ckpt destination | NAS run dir | neuronic save root → staged, sha-verified NAS copy | same |

**Why (Codex r1 BLOCKING 1):** every reference point trained on this box at 2 × 32; any other platform puts a hardware (and possibly backend) change inside a curve. exp_11 VANL@40k (8×8 L40) vs P1@40k (2×32 A6000), K=8: 8.048 / 1.022 / 37.32 vs 8.993 / 1.009 / 40.65 (`model_comparison.md:49,71`) is a warning about the size of compound confounding (rung + platform + code era + backend), not a platform-effect estimate. A `de_p1_f025` bridge measures a compound difference for P1 at 25 % only.

## 8. Time, disk, destinations

Rates on this box (2 × A6000 × 32): YAW 100 % 1.93–1.97 s/step exclusive with flash; P1 July 3.86 s/step (shared); dc_van 3.7 s/step (two DDP jobs on the same cards, SDPA); exp_25 CANON 6–19 s/step shared; B-F July 12.7 s/step (shared); paper appendix 40.3 h FLAC / 124.9 h OrbitRIR (3.6 / 11.2 s/step, cumulative active wall-clock). B-F/vanilla ratio under like conditions 3.1–3.5.

| block | duration (exclusive) |
|---|---|
| E0 replays (8 cells, §9.3) | ≈ 2–3 h (both cards) |
| SMOKE_DDP per run (30 steps + start-up) | ≈ 5–15 min × 6 |
| YAW training (each) | ≈ 22 h (measured exp_17) |
| B-F training (each) | ≈ 61–77 h (1.96 s/step × 3.1–3.5; projection) |
| 10 eval cells after each run | ≈ 1.5–2.5 h (both cards) |

**Primary campaign (D2-a), sequential, exclusive:** E0 → [SMOKE_DDP → FULL → admission → 10 evals → collect + publish] × 6 in the order B-F f025, YAW f025, B-F f050, YAW f050, B-F f075, YAW f075 ≈ 3 + 6 × 0.2 + 3 × 22 + 3 × (61–77) + 6 × (1.5–2.5) ≈ **262–318 h ≈ 11–13.3 days** of both cards (≈ 2× if shared). Earliest start: both cards free of foreign compute processes — GPU 1 holds host-A's exp_30 evaluations (pids 3599091, 3602030, cwd `~/codespace/flac-fork-attn*`, expected ≈ 04:00 Nov 8) and the FLAC session's REQUEST for exp_36 / exp_37 in `~/codespace/claude_gpu_yield/` stands: **the main session sequences exp_38 through that handshake** (exp_38's launchers only refuse non-idle cards; they never read or write the handshake files). From ≈ 04:00 Nov 8: B-F f025 ETD ≈ Nov 10 20:00 – Nov 11 12:00 EST (E0 + smoke first); campaign end ≈ Nov 19 – Nov 21 EST. Each run's steady-state rate is re-measured from its first 1,000 steps and the ETD re-announced.

**Disk (decimal GB / GiB).** Checkpoint 723,922,667 B; 16 per run 11.58 GB (10.79 GiB); 6 runs 69.5 GB (64.7 GiB); bridge +11.6 GB. Dumps 259.6 MB (247.6 MiB) per result cell, 60 cells 15.6 GB (14.5 GiB), bridge +2.6 GB; E0 cells dump-free. Local: ≈ 2 MB per cell + ≈ 13 MB per train log; farm = symlinks (≈ 43 GB of NAS reads over the campaign). Gates: training NAS ≥ 30 GiB free; each eval block checks the dump root ≥ 10 × 260 MB × 1.5 free, the AGREE checkpoint sha and the NAS mount; neuronic (only if used): free ≥ 2 × 12 GB per concurrent run.

**Destinations (ann. 07):** `/media/diskstation/yixunhu/FLAC/checkpoints/exp38_data_efficiency/<run id>/` (A6000 runs write there directly; neuronic via `<run>.incoming/` → `sha256sum -c CHECKPOINTS.sha256` → atomic rename); dumps `/media/diskstation/yixunhu/FLAC/exp38_pred_waveforms/` (ann. 08).

## 9. Evaluation protocol

### 9.1 Evaluator, backend, flags
`eval_FLAC.py` at the campaign pin (eval path byte-identical to `d6ca307a` / `5b065801`; the port is inert at eval), through the reviewed exp_35 no-flash launcher (`PYTHONDONTWRITEBYTECODE=1 python overlays/no_flash/exp35_no_flash_launch.py eval_FLAC.py …`; exactly one sentinel line `EXP35_NO_FLASH_LAUNCHER attention_backend=sdpa overlay_sha256=0d0e52ca… launcher_sha256=bbd2847d… target=eval_FLAC.py`) → SDPA, the backend of dc_van's, exp_25's and exp_35's cells; this box only.

Per cell (exp_25 / exp_35 argv shape, verified), with the worker count made explicit (Codex r2 NIT 4): **`--num-workers 4`** — the evaluator default every reference cell ran with (exp_14, exp_25, exp_35 pass none); the context draws, hence `input_hash`, depend on it:
- YAW / P1: `--model-config <the run's training config> --dataset-config src/configs/dataset_configs/AR/eval/acousticroom_unseeneval{,_1}.json --cond-method vanilla --cond-autocast bf16 --cfg-scale 1.0 --steps 1 --batch-size 64 --num-workers 4 --device cuda --record-per-scene --record-stream --expected-stream-count 6337 --seed <S> --rotate-mode fixed --rotate-deg 0 --eval-name <run>_K<K>_s<S> --ckpt-path <farm link> --dump-scored-waveforms /media/diskstation/yixunhu/FLAC/exp38_pred_waveforms`
- B-F: + `--cond-method fa_invariant --frame-avg-angles 0,90,180,270 --frame-avg-max-fwd-samples 64`.
The driver refuses unless the checkpoint's embedded `model_config` canonically equals `--model-config` and `--cond-method` equals the embedded `training.cond_method` (ann. 05).

### 9.2 Cells and schedule (Codex r2 NIT 1)
6 runs × K ∈ {8, 1} × seeds 42–46 = 60 cells (+10 bridge). Sequence per run: train → checkpoint admission → its 10 cells on both cards (two lanes) → collect + publish → next run's SMOKE_DDP. Outputs land beside a per-run farm symlink `outputs_FLAC/exp38_data_efficiency/eval/<run>/<ckpt name>` → NAS checkpoint.

### 9.3 E0 replay hard stops — mechanically upstream of every FULL (Codex r2 BLOCKING 1)

| gate | checkpoint | config | cells | reference (in `ref_cells/`, sha-pinned) | compared |
|---|---|---|---|---|---|
| E0a (exp_14 evaluator bridge) | dc_van_f025@40k `aa1a2cd6…` | `FLAC_AR_BVp1.json` | K8 s42, K1 s42 | exp_14 records (K8 `4136ac75…`) | `metrics` (exp_14 records carry no `by_scene`) |
| E0b (fa path) | B-F@40k `5319feb4…` | `FLAC_AR_BF.json` | K8 s42, K1 s42 | exp_35 BF-pin T (K8 `f167470e…`) | `metrics` + `by_scene` |
| E0c (YAW path) | YAW@40k `ac1f2603…` | `FLAC_AR_YAWAUG.json` | K8 s42, K1 s42 | exp_25 YAW-pin T | `metrics` + `by_scene` |
| E0d (P1 100 %) | P1@40k `c4c67882…` | `FLAC_AR_BVp1.json` | K8 s42, K1 s42 | exp_25 P1-pin T | `metrics` + `by_scene` |

Equality is recursive and type-strict (same keys, same types, `==` on every leaf; `by_scene` required where the reference has it). Every E0 and campaign cell's `input_hash` must equal the pinned exp_25 value for its (K, seed) — all ten pinned in `exp38_eval.py` (P1, YAW and B-F agree per (K, seed); e.g. K8 s42 `4c950c1e…`, K1 s42 `37d2f569…`). On PASS, `exp38_eval.py e0` publishes `data_efficiency_e0_receipt.json` (+ `.sha256`) atomically, bound to: the campaign closure digest (§10 B6), the evaluator + no-flash file hashes, the campaign environment receipt, the four checkpoint shas, the reference manifest sha and the input-hash map. **FULL (local) and the neuronic submitter refuse unless that PASS receipt exists and its bindings equal the current campaign tree.** If E0a is not exact, the campaign stops; Yixun's decision is recorded as a separate signed-off decision file (`data_efficiency_e0a_decision.md`) and the FLAC-row estimand is relabelled accordingly (report with the labelled difference, or re-evaluate dc_van × 3 at our pin, 30 cells). E0 cells are dump-free diagnostics.

### 9.4 Admission, aggregation, publication
`exp38_collect.py` admits a cell as one transaction (Codex r1 HIGH 8, r2 HIGH 9): exact raw `metrics` key set, all finite; protocol fields (`cond_method`, `frame_avg_angles`, `frame_avg_fwd_cap`, `orbit_execution`, `cond_autocast bf16`, `rotate_deg 0`, `seed`, `steps 1`, `cfg_scale 1.0`, `batch_size 64`, `n_samples 6337`, `device cuda`, `dataset_config`, `eval_name`, `weights_source ema`, `source_sha`, `per_scene_schema 1`, `scene_count 10` and the ten expected scene keys); `ckpt_sha256_loaded` / `ckpt_bytes_loaded` = the run's admitted final checkpoint; stream sidecar (`stream_count 6337`, pinned `input_hash`); the driver's cell receipt (model / dataset config shas, `data/AR/unseen_eval.json` and AGREE checkpoint shas, launcher sentinel, rc 0, metrics + stream + dump paths and shas — the cross-links); dump `COMPLETE` + `manifest.json` + `waveforms.npy` `[6337, 10240]` float32 + sha; exactly seeds 42–46 per (run, K), no duplicates. Runs with a `RESUMED_AT_*` marker never enter the primary table. Results `data_efficiency_results.{json,md}` are written under a collector lock as one atomic pair (temp files → fsync → renames, JSON last-renamed-first-read with a manifest sha in the MD), with a campaign manifest of every admitted file. Incomplete → PENDING.

**`model_comparison.md` (Codex r1 HIGH 12, r2 HIGH 10):** twelve preregistered row specs (YAW/B-F × 25/50/75 % × K 1/8; exact globs `…/eval/de_yaw_f025/*_metrics_1_1.0_de_yaw_f025_K{K}_s4[2-6].json` and for B-F `…_K{K}_s4[2-6]_fa_invariant_a4.json`) with an `exp38` contract that calls the collector's offline validator on the imported evidence (metrics, stream sidecars, cell receipts, checkpoint admission, compact dump evidence: manifest + sha) and publishes only a complete K=1 + K=8 five-seed transaction — added and code-reviewed **before the first FULL** (code round 2). Announcement 04 vs the table-regen trap: on this box `gen_model_comparison.py` refuses (rc 4/3) because 23 existing rows resolve only on neuronic (CLAUDE.md); the repo's standing practice (exp_25) is raws + specs committed here and the regeneration run by Yixun on neuronic. exp_38 keeps that practice and makes it a decision (D7): each results commit carries the raws, specs and a one-command neuronic regen file, and the regenerated table follows in the next commit from neuronic; the alternative (regenerating here with the row-regression overrides) needs Yixun's explicit order.

## 10. Implementation (per file; TDD)

### Round A — port exp_14's data side (committed `fac501b0`, `4187ef77`; provisional until the code review)
- A1/A2 `src/tests/test_ar_md_restrict_context.py` (byte copy `5f63591e…`; red ImportError → 50 green); `src/data/dataset.py` +7; `AR_md.py` byte copy `fb2f6882…`.
- A3 data files byte copies + `src/tests/test_exp38_subset_port.py` (self-contained hash/count/nesting/config-diff/re-raise tests).
- A4 `exp38_subset_audit.py` (+ `src/tests/test_exp38_subset_audit.py`: the audit's pool == AR_md's own captured `np.random.choice` pool on toy trees incl. the `S010` case; histograms; empty-pool and missing-file detection). Per fraction and for 100 %: ordered target list (sha, count) via `get_audio_filenames`; source hashes of `json_scandir`, `get_audio_filenames` and AR_md's four functions == exp_14's blob; for every target the unrestricted and restricted pools; zero empty pools; per-room histograms == the manifest. **r2 additions (Codex r2 HIGH 1):** bound map = canonical `realpath(AcousticRooms)`, the audit tool's own sha, `dataset.py`, `AR_md.py`, `train.json`, manifest, all splits and dataset configs; an ordered aggregate digest of (target, restricted pool) vectors per split; a **root fingerprint** = sha over every room's sorted listing. `--verify <artifact>` recomputes the fingerprint (243 listings, ≈ 10 s) and the bound map; every FULL runs it; the launcher checks the complete bound map, not selected hashes. First real run (pre-r2 code): PASS (73,473 / 145,665 / 218,463 targets, 243 rooms, 0 empty pools, histograms equal, 0 out-of-split candidates at 100 %); re-run with the final code.

### Round B — the exp_38 kit (`worklog/worklog_yixun/exp_38_data_efficiency_claude/`; tests `src/tests/test_exp38_*.py`)
- **B1 `exp38_registry.py`** — single source of truth (constants, configs incl. optimizer/scheduler/sampler hyper-parameters, chunk plan, run ids, argv, run-dir states, checkpoint names, log gates incl. a strict finite-loss parse with ≥ 50 values at the end, snapshot, subset-audit binding); CLI. Tests: constants vs disk; `_cap32/_cap8` bytes == regenerated; argv == dc_van's at-launch argv with only the registered differences; parsed `prefigure` values (`num_nodes 1`, `val_every −1`, `gradient_clip_val 0.0`, empty `ckpt_path` / `pretrained_ckpt_path` / `remove_pretransform_weight_norm`); run-dir states (fresh / foreign / `.exp38_lock` / `.exp38_submitted`); log gates per failure mode.
- **B2 `audit_ckpt`** (Codex r1 HIGH 6, r2 HIGH 2) — one byte read (sha256 + `torch.load(BytesIO)`); name, `global_step`, `epoch`; embedded config canonical-equal; the **entire** `state_dict` key/shape/dtype inventory == a freshly constructed wrapper's (built on CPU from the arm config, cached per arm) and `strict=True` load into it; every tensor finite; EMA: `diffusion_ema.ema_model.*` == `diffusion.model.*` suffixes/shapes/dtypes, `diffusion_ema.step` and `initted` present; optimizer: one AdamW whose state covers every trainable parameter of the fresh wrapper's `configure_optimizers()` param group in order, `exp_avg`/`exp_avg_sq` shaped as their parameters and finite, `step` == global step, group hyper-parameters (lr == scheduler `_last_lr`, betas, eps 1e-8, wd 1e-3, `initial_lr` 5e-5); scheduler all fields (`inv_gamma 1e6`, `power 0.5`, `warmup 0.99`, `final_lr 0.0`, `base_lrs [5e-5]`, `last_epoch == step`, `_step_count == step + 1`, `_last_lr` == closed form); loops (`optim_progress.optimizer.step.total.completed == step`, `batch_progress.total.ready == processed == step`, `epoch_progress.current.processed == epoch` consistency); the complete ModelCheckpoint callback key (`monitor None, mode min, every_n_train_steps 2500, every_n_epochs 0, train_time_interval None`) with `dirpath` == the run dir. Synthetic per-failure tests + a real-data readback on dc_van_f025's endpoint.
- **B3 run evidence (Codex r2 HIGH 3/4/6; replaces the out-of-process init digest as the primary proof)** — a new observe-only PL callback `src/training/run_evidence.py::RunEvidenceCallback`, enabled only by a new default-empty flag `--evidence-json` (`train.py` + `defaults.ini`; absent ⇒ the callbacks list and Trainer kwargs are byte-identical to today — red-proven test, the exp_07/exp_14 flag precedent). On every rank: at `on_fit_start` — sha256 over the full wrapper `state_dict` (keys, shapes, dtypes, bytes; EMA and pretransform included), the Sobol engine state (`wrapper.rng`: `quasi`, `shift`, `num_generated`), Python / NumPy / torch CPU / CUDA RNG state digests, world size, rank, SyncBatchNorm vs BatchNorm module counts in `trainer.strategy.model`, the number of ViT layers whose `forward` is the non-reentrant checkpoint wrapper, `flash_attn_func is not None`; it then wraps (pass-through, numerics untouched) `src.models.transformer.flash_attn_func`, `F.scaled_dot_product_attention` as seen by `src.models.transformer`, `torch.utils.checkpoint.checkpoint` (recording `use_reentrant`) and the conditioner's `forward` (recording batch length and `only_ids`) for the FIRST optimizer step; at `on_train_batch_end` of step 1 it writes `<path>.rank<r>.json` and unwraps. Tests: flag-off identity; synthetic module (fake trainer) evidence content; B-F call-shape recorder on a toy conditioner. Gates (start watcher): every rank's evidence present; all ranks report the same `state_dict` digest and equal to the registered seed-42 initial digest of the architecture (identical for YAW / B-F / P1, whose `.model` sections are canonically equal); SyncBN count > 0 and BatchNorm (non-sync) count 0 in DDP; checkpoint wrappers on both ViTs and every recorded checkpoint call `use_reentrant=False`; flash calls > 0 and SDPA calls == 0 inside the DiT for `ATTN_BACKEND=flash` (flash 0 for `sdpa`); for B-F the conditioner calls in step 1 = one full call of B = 32 samples + three `only_ids` calls of 32 samples each (no angle sharing); world size 2, micro 32. The registered initial digest is produced by `exp38_init_digest.py` (CPU, `train.py`'s construction order: seed_everything 42 → dataloader → model → VAE load → wrapper) and pinned after the first SMOKE_DDP confirms equality.
- **B4 recipe-mechanics tests** — (a) `test_exp38_bf_orbit_equivalence.py`: the real B-F conditioner (DINOv3-S from the pinned HF cache, CPU fp32, train mode, B = 2, legal 256×512 inputs, non-reentrant checkpointing on), two independent identical module copies under identical seeded RNG: `_orbit_average_loop` vs `_orbit_average_batched(cap=B)` → bit-equal outputs, masks, post-forward and post-backward RNG states, loss and every parameter gradient; a spy asserts every checkpoint call is `use_reentrant=False`; the default cap 64 at B = 32 shares two angles. (b) `test_exp38_sampler.py`: PL 2.1's own sampler replacement (`_update_dataloader` / `DistributedSampler` kwargs from `LightningEnvironment` vs `TorchElasticEnvironment` under the env each launcher sets), `set_epoch`, `pl_worker_init_function` seeds per (rank, worker), and first-batch index traces at world 2 for both environments; the §7 padding / drop numbers asserted. A real two-process trace is not run (CPU DDP harness out of scope); disclosed.
- **B5 `exp38_backend_probe.py`** — static: interpreter, `flash_attn.__file__` + `__version__` (pinned 2.7.4.post1 here; `EXPECT_FLASH_VERSION` on neuronic), `src.models.transformer.flash_attn_func` set (flash) / unset with the overlay resolved (sdpa); GPU: one bf16 `apply_attn` per DiT attention module with counted calls. Used as the pre-launch check; the in-process evidence (B3) is the binding proof.
- **B6 `exp38_train_local.sh`** — `MODE=DRY_RUN | SMOKE (1 rank on GPU 0; preliminary only) | SMOKE_DDP (2 × 32 production topology, 30 steps, smoke namespace) | FULL`. No RESUME mode (crash policy §6; a diagnostic `_resumed<step>` continuation is a separate explicit tool, not built now). Gates: (1) **campaign tree** — launched from the immutable campaign worktree `~/codespace/exp-38-campaign` (detached at the campaign pin); `EXPECT_SHA` == HEAD; tracked tree clean; **campaign closure digest** = sha over (path, sha256) of every tracked file of `train.py`, `defaults.ini`, `eval_FLAC.py`, `pyproject.toml`, `src/**` (minus tests), `overlays/**`, `data/AR/*.json`, the arm / base configs and every `exp38_*` file + `FLAC_AR_BF_cap*.json`; untracked entries limited to an allowlist of exact symlinks (`AcousticRooms` → `/media/diskstation/yixunhu/FLAC/AcousticRooms`, `weights/AGREE` → …, `weights/FLAC/VAE.{safetensors,ckpt}` → …) with every other untracked `*.py` / package / `sitecustomize` / `usercustomize` anywhere under the repo refused; (2) env sanitized (`PYTHONPATH` unset except the sdpa overlay, `PYTHONNOUSERSITE=1`, `ENABLE_TORCH_COMPILE` must be unset, `HF_HUB_OFFLINE=1`); (3) **campaign environment receipt** (python, torch + CUDA, cuDNN, NCCL, PL, transformers, flash-attn, torchaudio, numpy, prefigure, ema-pytorch, driver) written at E0, and every later launch refuses drift; (4) exactly two visible GPUs, both "NVIDIA RTX A6000"; (5) registry `check`, subset audit `--verify`, pins + backend probe; (6) **E0 receipt** PASS with matching bindings (FULL only); (7) **SMOKE_DDP receipt** (schema below) PASS for this run at this closure digest (FULL only); (8) the **exp_38 card lock** `~/codespace/.exp38_cards.lock` (`flock`, shared by every exp_38 training and evaluation entry point, held from the GPU check through the final audit; GPU ownership re-checked after acquisition) and GPU idle (default) / shared floor; (9) NAS mount (CIFS) + write probe + ≥ 30 GiB; (10) run dir fresh + the per-run lock `mkdir <run dir>/.exp38_lock`. Before the fork: at-launch record + the **immutable tool snapshot** `<run dir>/.exp38_tools/` = every file of the closure digest + the subset-audit artifact + E0/SMOKE receipts, `SNAPSHOT.sha256`; the start watcher, end gate and audits execute the snapshot (`python <snap>/…/exp38_registry.py --repo <snap> …`). Launch `setsid python train.py …` → log; start watcher (log gate + evidence gate; on failure terminates ONLY our process group); end gate; cadence set; B2 audit; `CHECKPOINTS.sha256`; lock released. **SMOKE_DDP receipt schema** (`data_efficiency_smoke_<run>_<ts>.json` + `.sha256`, smoke dir under `outputs_FLAC/exp38_smoke/`, never the NAS run namespace): run, platform, closure digest, environment-receipt sha, production-relevant argv projection (the FULL argv minus max-steps / checkpoint-every / save-dir / names), arm / dataset / split shas, backend probe, the B3 evidence summary of both ranks, GPU names + UUIDs, DDP members 1/2 + 2/2, ≥ 1 optimizer step, finite losses, zero checkpoints, exit statuses, verdict. Guard tests (`test_exp38_launchers.py`) drive the real script in DRY_RUN with synthetic environments (fake `nvidia-smi` on PATH, temp roots via DRY_RUN-only hooks) for every refusal.
- **B7 neuronic (sensitivity only) `exp38_train.sbatch` + `exp38_submit_neuronic.sh`** — one monolithic job per run; the same gates adapted (two or eight L40, 1 node / 1 task, rank env recorded then unset, closure digest, environment receipt of that platform, E0 receipt from this box bound by closure digest, a SMOKE pilot on that platform first). Submission ownership (Codex r2 HIGH 8): the submitter creates `<save root>/<run>/.exp38_submitted/` atomically with `intent.json` {token, run, head, closure digest, state `submitting`}, submits with `--job-name exp38-<run>-<token8> --comment exp38:<token>`, then records the job id (`state submitted`); `reconcile` resolves an interrupted submission by the token through `squeue` and `sacct -o JobID,JobName%-60,State%-40` (any query failure fatal); `abandon` renames the marker only after the token's job is proven absent or terminal (never while pending/running); smoke and FULL use distinct namespaces (`<run>_smoke<ts>`). `DRY_RUN=1` / `EXP38_SBATCH_DRYRUN=1` print everything, submit / spawn nothing.
- **B8 `exp38_eval.py`** — `plan` (60 (+10) cells + the 8 E0 cells, full argv, JSON); `prepare <run>` (NAS final checkpoint sha == `CHECKPOINTS.sha256`, B2 audit, farm symlink); `run --gpu G --cells …` (holds the exp_38 card lock; refuses a non-idle card unless `--shared`; checks dump-root space, AGREE sha and NAS per block; skip only if the cell validates; per-cell log + receipt); `e0` (verdict + receipt); `env-receipt` (writes / checks the campaign environment receipt). Tests `test_exp38_eval.py` (argv per arm, cond-method binding refusal, skip-only-if-valid, E0 recursive type-strict equality, input-hash pins, receipt bindings).
- **B9 `exp38_collect.py` + `ref_cells/`** — §9.4; tests `test_exp38_collect.py` (admission per failure mode, aggregation, golden table, PENDING, resumed-run exclusion, atomic pair).
- **B10 (code round 2, before the first FULL)** — `gen_model_comparison.py` twelve row specs + the `exp38` contract hook (§9.4), with tests.
- **B11 records** — `data_efficiency_worklog.md`, `commits_data_efficiency.md`, `data_efficiency_command.md`, `data_efficiency_params_set_up.md` (at launch), `data_efficiency_neuronic_commands.md`, and after results `data_efficiency_results.md`, `data_efficiency_analysis.md` and the SOP HTML page `data_efficiency_01_results.html` + `data_efficiency_results_assets/`, generated only from the admitted results JSON (Codex r2 NIT 5).

Commits < 200 changed code lines, one per red→green cycle, SHAs in `commits_data_efficiency.md`; `git pull --rebase` before every push.

## 11. Validation ladder

1. Static (`py_compile`, `bash -n`, `json.load`, `git diff --check`). 2. pytest: all new tests + the existing suites on the touched / used code. 3. Subset audit + `--verify` on the real tree. 4. CPU: init digest, B4a on the real DINOv3 weights. 5. DRY_RUN of every launch path (registry `dry-run` both platforms, `MODE=DRY_RUN` per run, `DRY_RUN=1` submitter and `EXP38_SBATCH_DRYRUN=1` per run, `exp38_eval.py plan`). 6. GPU-0 single-rank smoke now (≤ 15 min total, Yixun-allowed; YAW f025 and B-F f025, 20 steps, micro 32, no checkpoint; evidence callback on): banner, finite loss, flash counted, B-F call shapes, peak memory (B-F ≈ 15–16 GiB/rank expected), single-rank rate (lower bound only; promotes nothing). 7. At campaign start, on idle cards: environment receipt → E0 (8 cells) → SMOKE_DDP per run → FULL. 8. Nothing beyond rung 6 is run by the Coder.

## 12. Launch acceptance criteria (per run)

All §10 B6 gates pass; the start watcher sees DDP members 1/2 + 2/2, the dataset line (count / 243) on both ranks, the YAW banner before step 0 (YAW), finite losses, and the B3 evidence of both ranks satisfying every listed condition; checkpoints exactly every 2,500 steps; the end gate (Lightning's `max_steps=40000` marker, ≥ 50 finite parsed losses, none non-finite); the step-40000 B2 audit; `CHECKPOINTS.sha256`; exit statuses of train.py and the watcher recorded; never resumed.

## 13. Decisions for Yixun (with recommendations)

- **D1 rung:** 2 ranks × 32 everywhere (recommended); 8×8 only as a labelled cross-recipe run.
- **D2 platform / allocation:** (a) **all six on this box, sequential, exclusive** (recommended; ≈ 11–13.3 days incl. E0, smokes and evals); (b) B-F × 3 here + YAW × 3 on neuronic 2×L40 with the mandatory P1 bridge — the YAW 25–75 % points share a platform but its 100 % anchor stays A6000-trained (Codex r2 NIT 3), so that curve is cross-platform at 100 %; (c) the brief's split — not recommended (the B-F curve mixes A6000 and L40 within 25–75 %), and if chosen the bridge is mandatory and every A6000/L40 comparison is labelled sensitivity.
- **D3 crash policy:** fixed by §6 (re-run from 0; resumed = sensitivity only). No decision needed unless Yixun wants exact-resume machinery built.
- **D4 training backend:** flash (recommended; = the documented P1 / B-F backend); SDPA would match dc_van's training. The evidence gate refuses a mismatch either way.
- **D5 GPU policy here:** idle (recommended), sequenced by the main session through the `claude_gpu_yield` handshake.
- **D6 E0a mismatch:** decided at that moment by Yixun (§9.3).
- **D7 table regeneration:** neuronic regen after each results commit (recommended; repo practice) vs here with the row-regression overrides (needs his explicit order).

## 14. Risks and disclosures

One training seed per cell; matched steps, not epochs or compute. Historical asymmetries: dc_van trained SDPA (ours flash, D4); different code / dependency eras (§5); P1 100 % resumed at 32.5k; YAW 100 %'s launch source and training backend unknown; W&B vs `--logger none` (operational); any non-A6000 platform (D2). E0 removes evaluation-side doubt only. Concurrent writers: runs launch from the frozen campaign worktree; a closure change after a launch = a new run. Never edit a running bash script (bodies are single functions called on the last line); never touch another session's processes or the handshake files; the Coder starts no real training.

## 15. Changelog

**r0 → r1 (Codex plan r1, REVISE):** B1 primary = all six here; B2 chains removed; B3 one job per run + locks; B4 positive backend probe; H1 YAW provenance scoped; H2 exhaustive subset audit; H3 B-F CPU equivalence test; H4 sampler test; H5 SMOKE_DDP; H6 checkpoint audit + init digest; H7 E0a–d; H8 collector binding; H9 resources; H10 snapshot; H11 acceptance; H12 table rows; H13 acknowledged; N1–N6.

**r1 → r2 (Codex plan r2, REVISE):**

| Finding | Disposition |
|---|---|
| B1 E0 incomplete / not upstream | E0d both K (8 cells); recursive type-strict equality; ten input hashes pinned; E0 receipt bound to closure / env / evaluator / refs; FULL + submitter refuse without it; E0a decision file (§9.3) |
| B2 resume exception | removed; crash → re-run from 0; `_resumed<step>` = sensitivity only (§6, D3) |
| H1 subset-audit binding | realpath, tool sha, full bound map, pool digests, root fingerprint, `--verify` before every FULL (§10 A4) |
| H2 full-state audit | full inventory vs a fresh wrapper, strict load, finiteness, optimizer coverage, all scheduler / loop / callback fields (§10 B2) |
| H3 init proof | in-process evidence callback on every rank incl. Sobol + RNG states; registered digest (§10 B3) |
| H4 B-F GPU structure | conditioner call shapes recorded in-process in step 1 (§10 B3) |
| H5 sampler | §7 corrected (strided ranks, 1/1/31 drops, 1/1/1 vs 7/7/1 pads); PL replacement / set_epoch / worker-seed / trace test (§10 B4b) |
| H6 SMOKE receipt + active recipe | receipt schema; SyncBN / checkpoint / flash observed in-process; smoke namespace (§10 B3, B6) |
| H7 immutable closure | campaign worktree + closure digest + symlink allowlist + shadow refusal + env sanitization + version pins + two-A6000 assertion (§10 B6) |
| H8 locks | exp_38 card lock shared by training + eval; Slurm intent token + reconcile + abandon; distinct smoke namespace (§10 B6–B8) |
| H9 collector | finite / exact keys / orbit / device / bytes / schema / scene keys / cross-links / unseen + AGREE shas / atomic pair (§9.4) |
| H10 table | twelve specs + `exp38` contract before FULL (B10, code round 2); regen practice made explicit (D7) |
| H11 resources | ETD incl. E0 / smokes / evals; dump-root and AGREE checks per eval block; neuronic walltime from its pilot (§8) |
| H12 implementation during review | acknowledged; the whole diff goes to the code review (header) |
| N1–N5 | schedule sequence (§9.2); "nothing else" qualified (§1, §14); D2(b)/(c) wording (§13); eval `--num-workers 4` explicit, with the reason (§9.1); analysis + HTML artifacts (§10 B11) |
