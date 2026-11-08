# plan_yaw_aug_c16 — exp_40: C16-discrete Yaw-Aug FLAC (offsets Δp ∈ {0, 32, …, 480})

**Author:** Claude Opus 5.5 (Coder/Planner seat, subagent of the main session) · **Date:** 2026-11-07 · **Status:** r1 — Codex plan review r1 (REVISE: 2 BLOCKING / 12 HIGH / 4 NIT, `yaw_aug_c16_codex_plan_r1_review.md`) folded in; every finding is mapped in §12. r0 archived in `plan_history/`. · **Branch / worktree:** `exp-40-yaw-aug-c16` in `~/codespace/exp-40-yaw-aug-c16`, cut from `localization-exp` `2b42468c8f429b1d93da93b3e4ddb7bd6f8f1932`; `main` is never touched.

"Verified" = re-checked in this session against files/commits; "inferred" = argued from records; "unknown" = not recoverable.

---

## 0. Corrections to the brief and the decisions they create

**C1 — the YAW checkpoint in the existing figures is exp_17's A6000 arm, not exp_15's neuronic arm (verified; Codex r1 Q1 confirmed).**

| | exp_15 YAWAUG (neuronic) | exp_17 Yaw-Aug (A6000) = **YAW in the figures** |
|---|---|---|
| 40k sha256 | `16b964ec…` (leg-16 transcript `AUDITED_SHA256`) | `ac1f2603…` (exp_20 admission, NAS `ar_40k_endpoints/PROVENANCE.md`, exp_25 ledger) |
| ranks × micro × accum | 8 × 8 × 1 (SyncBN 64) | 2 × 32 × 1 (SyncBN 64) |
| launcher | `torchrun --standalone`, SLURM rank vars unset | PL-spawned subprocess DDP (`python train.py --num-gpus 2`), inferred from exp_17's plan/params (exact launcher script unknown — removed worktree) |
| execution | 16 chained 2.5k legs (PL resume; iterator/RNG not restored) | monolithic 0→40k (exp_17 worklog 14:54:52 Aug 15 → 42cbddaf "training COMPLETE") |
| config origin | `exp_15…/FLAC_AR_YAWAUG.json` `b14caae2…` | the same file (admission `config_path` + `config_sha256`); embedded canonical `0761a677…` |
| grad-ckpt | ON | ON (the retired checkpointing-OFF variant on `origin/exp17-yawaug-scratch` is excluded) |
| code | `fb4fc749` … `88897e02` | audited base `41aa31dc`; exact launch commit **unknown** |
| attention backend | flash inferred (no "flash_attn not installed" line in the leg transcripts) | flash inferred (none in the Aug-16 eval logs either); **unknown** for the training process |
| environment / sampler-worker trace | neuronic `flac` env, torchrun DistributedSampler world 8 | A6000 `flac` env of Aug 2026 (versions not recorded); world-2 sampler; **unknown** trace |
| figures / cells | not in the exp_25/35/36 figures; `model_comparison.md` row "(exp_15)" | every exp_25 YAW-pin cell, exp_36 YAW cells, exp_20/22 YAW |

**Scope of the equivalence claim (Codex r1 HIGH 1).** On the 2×32 rung C16 and exp_17 share the same *logical* draw keys `(seed 42, global_step, rank ∈ {0,1}, within-rank index 0..31)`, and at every key the C16 offset is the reference offset rounded down onto the 32-column grid (§2.1). It is NOT claimed that the historical exp_17 run assigned each key to the same training sample: its launch commit, environment, backend and sampler/worker trace are unknown. The claim is "same audited logical algorithm and draws, quantised", never "bitwise reproduction of the exp_17 trajectory".

**D1 (Yixun) — platform / rung.** Both kits are built; the platform and rung are launch arguments, never defaults.
- **(a) RECOMMENDED (primary): 2×A6000 on this box, 2×32, monolithic, exclusive cards.** The only option matching the figure arm's platform, topology, micro-batch, SyncBN batch and launcher form. ≈ 22 h exclusive (exp_17 measured 1.966 s/step). The cards are shared with exp_38's queue (its plan r2: six 2-card runs, ≈ 10–12 days), the host-A exp_30 evaluations on GPU 1 (until ≈ 04:00 Nov 8) and the `~/codespace/claude_gpu_yield/` handshake — the ordering is Yixun's.
- **(b) neuronic 2×L40, 2×32, monolithic** — same logical draw keys; GPU model and launcher form differ (no 2×32 L40 run in the record bounds that effect). A *platform sensitivity*, labelled so in every artifact. Needs a partition wall limit ≥ the timed pilot's projection (§4.4).
- **(c) neuronic 8×L40, 8×8** — exp_15's recipe as literally asked. Its comparator is exp_15's `16b964ec` (not in the figures); rank partition and draw keys differ from exp_17's. Not recommended for the figure question.

**D2 (Yixun) — the brief's "default platform: neuronic".** The kit refuses to run without an explicit `PLATFORM`/`RUNG`; the command sheet lists (a), (b), (c) in that order.

**D3 (Yixun) — crash policy.** Primary = one uninterrupted 0→40k run, exclusive cards. A crash → rerun from step 0 under a fresh `_r<n>` run id. A resumed run (PL `--ckpt-path`) is a different trajectory (iterator and RNG streams are not restored) and **can never become the primary arm** unless exact iterator + RNG restoration is first implemented and tested; the kit has no resume mode, and the command sheet shows the manual resume command only under a distinct sensitivity run id `…_resumed<N>`.

**D4 (Yixun) — neuronic checkpoint publication.** This box cannot be assumed to reach neuronic and the job cannot reach the NAS. On neuronic the kit writes each finished cadence checkpoint's sha256 into an in-job manifest as soon as the next checkpoint exists (§4.5); publication to the NAS is an *external cadence pull* by Yixun (one command, repeatable at any time, publishes every finished checkpoint not yet on the NAS). If he does not want to pull during the run, the alternative is an explicit "publish once at the end" relaxation — his decision.

**D5 (Yixun) — off-grid angles.** The brief suggested {11.25°, 33.75°, 56.25°, 101.25°}. Codex r1 HIGH 8: those cluster in the first 101.25° and each is equidistant from two grid points. Planned instead: the balanced C4-spaced half-step coset **{11.25°, 101.25°, 191.25°, 281.25°}** (16/144/272/400 columns), each paired by a pre-registered *floor* rule with {0°, 90°, 180°, 270°}, whose cells already exist (T + C4 blocks). The brief's set can be added as 24 more cells (C16/YAW/P1 × 4 angles × 2 K, ≈ 5 GPU-h) if Yixun wants it.

**D6 (Yixun) — the paper figure.** The paper repo's plotting script (`-Neurips-2026-workshop-Sound-Localization/scripts/plot_yaw_c4_three_models.py`) hard-codes its arms, sources, angles and seeds. This seat does not edit the paper repo; it delivers a validated per-cell CSV (§5.4) whose schema the script can import, and the exact change the script needs is written in the results note. Who makes that change is Yixun's call.

---

## 1. Question and estimand

**Question.** Restricting the existing per-sample yaw augmentation to the C16 group (column offsets d ∈ {0, 32, …, 480}, θ = k·22.5°): (i) what does it cost or gain at θ = 0, (ii) is the model as yaw-robust as YAW on its own grid, and (iii) does robustness survive between grid points?

**Arms (EMA, vanilla conditioning).** **C16** (new) · **YAW** = exp_17 `ac1f2603…` (uniform over 512 columns) · **P1** = exp_07 `c4c67882…` (no augmentation). One training run per arm (seed 42); eval seeds estimate evaluation variability only.

**Estimand.** Per (arm, K ∈ {1, 8}, rotation cell, eval seed): the top-level item-weighted `metrics` of `eval_FLAC.py` on the full unseen split (6,337 items / 17 rooms, announcement 01) — the convention of every exp_17/25/36 YAW cell. Readouts (descriptive; single training seed, no confirmatory test):
- θ = 0 table: mean ± sample SD over seeds 42–46 (C16, YAW, P1 all have five seeds).
- C4 block: per-angle metrics; degradation vs the arm's own θ = 0 cell at the same seed; five-seed C16 vs YAW contrasts (YAW s43–46 from exp_36), seed-42-only contrasts against P1 (P1 has C4 at s42 only).
- 45°-grid block (on the C16 grid, k = 2, 6, 10, 14): seed-42 paired contrasts, all three arms.
- **Off-grid test (pre-registered):** for each arm and K, `Δoff(θ) = m(θ) − m(θ − 11.25°)` for θ ∈ {11.25, 101.25, 191.25, 281.25} (floor pairing), seed 42, averaged over the four θ; the C16-minus-YAW difference of `Δoff` is the off-grid penalty of the discrete augmentation, read against each arm's seed-42 C4 spread (max − min over {0, 90, 180, 270}). "Flat between grid points" ⇔ C16's `Δoff` is within YAW's ± its C4 spread; otherwise "degrades between grid points". Both outcomes are reportable.
- Random-yaw block (exp_14/15 R protocol): five-seed means, degradation vs θ = 0 (all three arms have five seeds).

---

## 2. The treatment: one new optional key

### 2.1 Semantics (coupled quantisation)

`training.yaw_aug.offset_step = s` (int). With the key present the offset of sample `i` at `(global_step, global_rank)` is `u_i` = the existing draw (`torch.randint(0, img_w)` from an MT19937 seeded by `_yaw_aug_step_seed(seed, step, rank)`), then `d_i = s · ⌊u_i / s⌋`. Because `s | img_w`, `⌊u/s⌋` is **exactly uniform on {0, …, img_w/s − 1}** (each value has exactly `s` preimages among the 512 equiprobable `u`), independent across samples; nothing new is drawn and no global RNG stream is touched (`_apply_yaw_aug` + quantisation leave the global RNG untouched; a whole `training_step` legitimately advances other streams — tests compare the before/after digests with the golden's). Resume-exact per logical key, exactly as before. s = 32, img_w = 512: 16 offsets, θ = k·22.5°. `s = 1` reproduces the absent path exactly (tested).

Validation (fail-closed `ValueError`, exp_15 convention), one helper used by the factory and the wrapper: `s` an `int`, not a `bool`; `1 ≤ s < img_w`; `img_w % s == 0`. Policies (Codex NIT 1): (i) `s == img_w` is refused although it defines the trivial group: it would silently disable the augmentation; (ii) a *disabled* factory block (`enabled: false`) returns `{}` before reading `img_w`/`seed`/`offset_step` — the exp_15 contract that a disabled block is the pre-change call, unchanged — while direct construction with `yaw_aug_offset_step` and `yaw_aug_enabled=False` is refused (there it can only be a programming error).

### 2.2 Absent key ⇒ the reference behaviour, exactly

"Byte-identical" here means: for every valid existing configuration, identical captured behaviour and draws (the E1/E2 golden), not identical bytecode or error texts. Mechanism: the factory passes **no new kwarg** when the key is absent; the wrapper's new parameter `yaw_aug_offset_step=None` is appended LAST; one guarded line after the draw (`if s is not None: offsets = quantize_yaw_offsets(...)`); the banner is the exp_15 line verbatim when absent (`yaw_aug ENABLED img_w=512 seed=42`) and `yaw_aug ENABLED img_w=512 seed=42 offset_step=32` when present. Other sites (Codex r1 Q2): `validation_step`/`test_step` never augment (unchanged; exp_15 test stays green); `eval_FLAC.py` builds the wrapper from the C16 config through `create_training_wrapper_from_config` — tested to construct and never call `_apply_yaw_aug` in eval; `_parse_canonicalize_config` keeps its yaw ⊕ canonicalize refusal with the extended parser — tested.

### 2.3 Config

`exp_40…/FLAC_AR_YAWAUG_C16.json` = the bytes of `exp_15…/FLAC_AR_YAWAUG.json` (`b14caae2…`, read-only) with exactly `"seed": 42\n        }\n    }\n}` → `"seed": 42,\n            "offset_step": 32\n        }\n    }\n}` (sha `b497dfe9…`). Pytest: forward byte construction, base and arm shas pinned, type-strict single delta, recipe pins (EMA, no `cond_method`/`canonicalize`/`are_*`/`frame_avg_*`, grad-ckpt on both ViTs, ViT 256×512).

---

## 3. Equivalence to the reference arms' training code

**E1 — enabled-path golden at four source pins (done).** The capture tool `src/tests/exp40_yaw_aug_golden_capture.py` (`capture --src-root <git-archive export>` / `assemble`) records from a source tree: four tiny CPU `training_step`s (exp_15 golden harness model, real factories, real DiT, flash neutralised) with `yaw_aug` enabled at `(step, rank)` ∈ {(0,0), (7,0), (7,1), (2500,3)} — per-sample offsets reaching `rotate_scene_metadata`, metadata and conditioning digests reaching the model, loss `float.hex`, global RNG digests either side — and the offsets `_apply_yaw_aug` draws at width 512 / seed 42 for both rungs (8×8, 2×32) at steps {0, 1, 2, 3, 1000, 2499, 2500, 39999}, a rotated-batch digest and global-RNG isolation. **Recaptured with the committed tool bytes** (Codex HIGH 2) at `fb4fc749`, `88897e02`, `41aa31dc`, `2b42468c`: all four canonical records identical, sha256 `76c1f634…`. The fixture `src/tests/fixtures/exp40_yaw_aug_enabled_golden.json` binds the four full pins, the per-pin record sha, the tool's sha256, python/torch versions; a test refuses a tool whose bytes differ from the recorded sha (any tool edit forces a recapture).

**E2 — regression + quantised-path tests at HEAD (§6).** Absent key ⇒ capture == fixture; `s = 1` ⇒ == fixture; `s = 32` (real) ⇒ offsets == `32⌊g/32⌋` elementwise, support ⊆ the 16 offsets; `s = 4` (tiny, img_w 16) ⇒ offsets == `4⌊g/4⌋`, metadata digests == `rotate_scene_metadata` by those offsets, RNG digests == golden's. **Independent pre-rotated reference (Codex HIGH 2):** the same tiny step run (i) with `offset_step = 4` and (ii) with yaw_aug *disabled* on metadata pre-rotated by first principles (`torch.roll` + explicit z-rotation matrices) by the snapped offsets, both from the same seeded state ⇒ identical conditioning tensors, loss and every parameter gradient (exact equality on CPU). The exp_15 disabled-path golden stays green.

**E3 — closure diff audit (verified).** `41aa31dc → 2b42468c`, training closure (`train.py`, `defaults.ini`, `src/` minus tests, `data/AR`): `train.py`, `defaults.ini`, `data/AR` unchanged; `src/training/diffusion.py` (+223) / `factory.py` (+69) = exp_25 canonicalisation, **gated** by `training.canonicalize` (absent; `transfer_batch_to_device` returns the parent call when disabled); `src/data/yaw_rotation.py` (+259) = canonicalisation helpers only (unused); `src/metrics/*` = RAF branches (dataset ≠ RAF; metrics are eval-only); RAF configs, `src/tools/depth_canon_stats.py`, `src/localization/*` = not imported by training. `fb4fc749 → 41aa31dc`: exp_16 ARE (**gated** by `training.are_lambda`, absent) plus one **unconditional, default-preserving** change: `dataset.py`/`utils.py` publish `RandomTimeShift.last_shift` as `info["time_shift"]` (no extra RNG value, audio bit-identical, an int key the conditioners ignore). Our diff adds only §2.

---

## 4. Training protocol and launch kit

### 4.1 Literal argv

```
train.py --model-config worklog/worklog_yixun/exp_40_yaw_aug_c16_claude/FLAC_AR_YAWAUG_C16.json
  --dataset-config src/configs/dataset_configs/AR/train/acousticroom_train.json
  --pretransform-ckpt-path weights/FLAC/VAE.safetensors
  --max-steps 40000 --batch-size <32|8> --accum-batches 1 --num-workers 6 --seed 42
  --num-gpus <2|8> --num-nodes 1 --strategy ddp_find_unused_parameters_true --sync-batchnorm true
  --precision bf16-mixed --val-every -1 --val-dataset-config '' --gradient-clip-val 0.0
  --logger wandb --checkpoint-every 2500
  --name FLAC_exp40_C16 --experiment-name <RUN_ID> --save-dir <ABS SAVE ROOT>
```

= exp_15's literal leg-1 argv shape with config, names and rung substituted; `--pretrained-ckpt-path` / `--remove-pretransform-weight-norm` / `--ckpt-path` absent (ini defaults `''`, enforced by the kit's parsed-argv check). `WANDB_MODE=offline` (flac's wandb 0.15.4 cannot create online runs on this box; same on neuronic so an online failure cannot kill the job; logging is not part of the recipe — `logger.watch` and the run-id draw do not touch the seeded torch streams, and P1/YAW were also trained with `--logger wandb`). RUN_ID ∈ {`exp40_C16_a6000_2x32`, `exp40_C16_neuronic_2x32`, `exp40_C16_neuronic_8x8`} (+ `_r<n>` after a crash); smoke runs get their own namespace `<RUN_ID>_<mode>_<stamp>` under a separate smoke root.

### 4.2 One immutable run tree per launch (Codex HIGH 3)

Every SMOKE / DDP_SMOKE / FULL runs from `RUNTREE = <save root>/runtrees/<RUN_ID or smoke id>__<sha12>/` created by the launcher (A6000) or the submitter (neuronic, at submission time, so a queued job is immune to checkout changes): `git archive <EXPECT_SHA>` extracted there, made read-only (`chmod -R a-w`), plus exactly three allowlisted links — `AcousticRooms` → the dataset realpath, `weights/FLAC/VAE.safetensors` → its realpath — and nothing else (training needs no AGREE: it is loaded only in `set_test_config`). The gate (`exp40_kit.py check --tree RUNTREE`) verifies: every regular file's git blob hash equals `git ls-tree -r <EXPECT_SHA>` and the file set is exactly the commit's plus the allowlisted links; links resolve to the recorded realpaths; the dataset passes the **AR sample digest** (16 deterministic train items × {IR wav, depth npy, metadata json}, sha256 `a6ec8195…`, computed on this box; neuronic must reproduce it); the VAE sha `8d82159e…`; the DINOv3 `model.safetensors` sha `4610ad75…` in the HF cache (`HF_HUB_OFFLINE=1`). train.py, the smoke/evidence driver and every post-run audit execute from RUNTREE (its own `exp40_kit.py`), never from the live checkout.

### 4.3 Gates (all fail-closed, exit 2, nothing spawned)

1. **Pin:** `EXPECT_SHA` full 40-hex == the source checkout's HEAD; the closure (`train.py defaults.ini src data/AR` + configs + kit) committed and no untracked file under it; RUNTREE verified as in §4.2. Neuronic: `EXPECT_SHA` reachable from `origin/exp-40-yaw-aug-c16` after `git fetch` (pushed), Codex HIGH 12.
2. **Environment (Codex HIGH 4):** `PYTHONPATH`/`PYTHONSTARTUP`/`PYTHONHOME`/`PYTHONUSERBASE` unset, `PYTHONNOUSERSITE=1`, `PYTHONDONTWRITEBYTECODE=1`; interpreter = the platform's pinned python; distribution versions pinned on the A6000 (python 3.10.20, torch 2.7.0 (+cu126), torchaudio 2.7.0, pytorch-lightning 2.1.0, transformers 4.57.0, flash-attn 2.7.4.post1, numpy 1.23.5, ema-pytorch 0.2.3, prefigure 0.0.9, safetensors 0.7.0, huggingface-hub 0.35.3, einops 0.8.2; CUDA runtime 12.6, cuDNN 90501, NCCL 2.26.2, driver 590.48.01); every module's origin under the env prefix; `torch.backends.cuda.matmul.allow_tf32` / `cudnn.allow_tf32` recorded. On neuronic (values not readable from here) the same fields are **recorded** by the SMOKE/pilot job and the FULL job refuses any drift from the smoke receipt's environment digest. **Positive flash proof:** `src.models.transformer.flash_attn_func is not None` from RUNTREE, plus the counted flash calls of the smoke (gate 7).
3. **Config contract** through the real factory parsers: `_parse_yaw_aug_config` → `{enabled, 512, 42, offset_step 32}`; `_parse_canonicalize_config`, `_parse_frame_avg_cap_config`, `_parse_are_config` → `{}`; EMA on; grad-ckpt on both ViTs; ViT img_w 512.
4. **Parsed argv:** the argv re-parsed with train.py's own `prefigure` defaults equals the literal argv's intended values for every recipe key (incl. the empty `pretrained_ckpt_path`, `remove_pretransform_weight_norm`, `ckpt_path`, `val_dataset_config`).
5. **GPUs:** A6000 — exactly two visible `NVIDIA RTX A6000`; FULL is **idle-only** (any compute process on GPU 0 or 1 refuses; uuid-mapped; query failure fatal); `GPU_POLICY=shared` exists only for SMOKE/DDP_SMOKE and is recorded. The check runs after the exp_40 card lock is taken and again immediately before the spawn (Codex HIGH 6). The `claude_gpu_yield/` handshake files are hashed into the launch record; sequencing the cards through it is the main session's / Yixun's job. Neuronic — `SLURM_NNODES == 1`, `SLURM_NTASKS == 1` (recorded with every `SLURM_*` rank variable before clearing them), N visible `NVIDIA L40`.
6. **Namespace / disk:** FULL run namespace absent (fresh); NAS cadence directory absent (A6000); ≥ 30 GiB local free; per-run flock.
7. **Smoke receipt (FULL only, Codex HIGH 5):** a PASS receipt `receipts/ddp_smoke_<platform>_<rung>_<EXPECT_SHA>.json` whose schema is validated field by field: same platform/rung/HEAD/RUNTREE-file digest/environment digest; recipe projection (argv minus `--max-steps`, `--checkpoint-every`, `--save-dir`, `--experiment-name`) equal to FULL's; ranks exactly 0..N−1; steps 0..S−1 each exactly once per rank, each with exactly `micro` offsets equal to the counter-based recomputation snapped to the grid and to the E1 fixture where the fixture covers the step; GPU UUIDs/model per rank; `SyncBatchNorm` module count > 0 on every rank; non-reentrant ViT checkpoint calls > 0 on every rank (and no reentrant call); flash calls > 0 on every rank; per-rank initial-state signature (full wrapper `state_dict` digest + python/numpy/torch-CPU/CUDA RNG digests + Sobol engine state) with the parameter digest equal across ranks; finite numeric losses ≥ S/2 observations; train / tee / watcher exit statuses all 0.

### 4.4 Execution

**A6000 (`exp40_train_local.sh`):** `MODE=DRY_RUN|SMOKE|DDP_SMOKE|FULL`, `PLATFORM=a6000`, `RUNG=2x32`, `RUN_ID`, `EXPECT_SHA`, `GPU_POLICY` (FULL: idle only). DRY_RUN runs gates 1–4 and 6 against a freshly built RUNTREE and prints the argv; it spawns nothing and removes nothing. Launch: `setsid python <RUNTREE>/…/exp40_evidence_driver.py` (which runs `train.py` in-process) with cwd RUNTREE, teed to `yaw_aug_c16_<ts>_<MODE>_train.log` in the experiment folder (untracked while running), an at-launch record (`…_launch.json`: argv, HEAD, RUNTREE digest, environment, GPU UUIDs, co-tenants, handshake digests, host) written before the fork and committed with the command file; the **start watcher** (`exp40_kit.py watch`): the exact C16 banner and N `MEMBER k/N` lines must precede the first progress line with a completed step, else it terminates only its own process group (exit 8; timeout 9); on FULL, a **cadence publisher** (§4.5) runs alongside; after exit the **end audit** (§4.6). SMOKE: one rank on GPU 0 (`CUDA_VISIBLE_DEVICES=0`, `--num-gpus 1 --strategy auto`, no SyncBN — a preliminary rung only, it cannot promote a FULL), micro 32, 20 steps, no checkpoints. DDP_SMOKE: both cards, the production rung, 30 steps, no checkpoints; its PASS receipt is what FULL requires.

**Neuronic (`exp40_submit_neuronic.sh` + `exp40_train.sbatch`):** `MODE=PILOT|FULL`, `RUNG=2x32|8x8` (required). PILOT = a DDP_SMOKE on the target rung with 300 steps (post-warm-up rate measured over steps 100→300, a rate window the 20-step smokes cannot give); FULL requires the PILOT receipt (gate 7) and `--time = ceil(projected 40k time × 1.15 + 45 min)`, refused if it exceeds the partition's `MaxTime` (`sinfo -h -p all -o %l`; query failure fatal) — then D1(b)/(c) cannot run monolithically and the decision returns to Yixun. Resources: 2x32 → 2 GPUs / 22 CPUs / 64 G; 8x8 → 8 / 64 / 108 G (exp_15's formula, raised for micro 32). **Submit exactly once (Codex BLOCKING 1):** a per-(run id, mode) state file `receipts/neuronic_<RUN_ID>_<MODE>.state.json`, every transition written atomically (temp + fsync + rename) under one flock: `prepared(intent T)` → `sbatch --parsable --job-name exp40-<RUN_ID>-<MODE>-<T> --comment <T>` → `submitted(job id)`; a re-invocation in any state first reconciles T through **both** `squeue -h -n <name>` and `sacct -n -X -P --name <name> -o JobID,State%-40` (any query failure fatal; `sacct` states normalised by prefix, ambiguous prefixes fatal): a live job ⇒ refuse; `prepared` with no job in either ⇒ the previous `sbatch` never happened ⇒ re-submit with the same T; `submitted` + terminal ⇒ recorded, and a new submission needs `NEW_INTENT=1` (fresh T) only if the previous job is terminal-failed — never while it is pending/running; FULL after a COMPLETED FULL is refused (a rerun needs a new `_r<n>` run id). SMOKE/PILOT and FULL use distinct job names and state files. `DRY_RUN=1` prints the state transition and the exact `sbatch` line, submits nothing. The sbatch asserts its own job name carries the state file's intent token before training.

### 4.5 Every 2,500-step checkpoint to the NAS (Codex BLOCKING 2)

Destination (announcement 07): `/media/diskstation/yixunhu/FLAC/checkpoints/exp40_yaw_aug_c16/<RUN_ID>/`. A checkpoint `step=S` is **finished** once `step=S+2500` exists, or — for S = 40000 — once the training process has exited rc 0 (PL writes each checkpoint once, in one `torch.save`; the next cadence file appears ≈ 80 min later, so a finished file is never being written). **A6000:** `exp40_kit.py publish-cadence --watch` runs next to FULL from RUNTREE: for every finished, unpublished S it hashes the source, copies to `.<name>.partial-<pid>` on the NAS, re-hashes the copy, refuses an existing target, renames, then rewrites `MANIFEST.sha256` atomically (temp + rename); the source is re-hashed after the copy (unchanged or refuse). It never deletes local files. The end audit re-verifies the 16-file NAS inventory against the local `CHECKPOINTS.sha256`. **Neuronic (D4):** the job's own `publish-cadence --local-manifest` writes `CHECKPOINTS.partial.sha256` lines for finished checkpoints on neuronic; Yixun pulls the checkpoint directory to this box at any time (`rsync -a --partial` command in the command sheet) and runs `exp40_kit.py publish-cadence --src <staging> --manifest <pulled CHECKPOINTS.partial.sha256>` here, which publishes every listed, hash-verified, not-yet-published checkpoint with the same no-clobber protocol.

### 4.6 End audit and admission (Codex HIGH 7)

From ONE read of the 40k checkpoint bytes (`torch.load(BytesIO)` on CPU): `global_step == 40000`, `epoch == 8`; embedded `model_config` canonically == the arm config; the complete wrapper `state_dict` inventory (keys, shapes, dtypes) == a freshly constructed wrapper's; strict load clean; every tensor finite; EMA `initted == True`, `step` > 0, EMA keys/shapes/dtypes mirror `diffusion.model.*`; optimizer: one state, one param group with the pinned AdamW hyperparameters, a state for every trainable parameter (index-mapped to `diffusion.parameters()`), `exp_avg`/`exp_avg_sq` shaped like the parameter and finite, `step == 40000`; scheduler: `inv_gamma 1e6, power 0.5, warmup 0.99, final_lr 0, base_lrs [5e-5], last_epoch 40000, _step_count 40001`; loops: `fit_loop` epoch/batch progress consistent with 40000 optimizer steps; callbacks: exactly the `ModelCheckpoint{… every_n_train_steps: 2500 …}` key with `dirpath` == the run's checkpoint dir; cadence set = exactly {2500, …, 40000}; `CHECKPOINTS.sha256` written. The per-rank initial-state signatures emitted by the FULL run's evidence driver before step 0 are checked equal across ranks (parameters) and recorded. **Admission** (`exp40_kit.py admit`) re-runs the checkpoint checks on the NAS copy, verifies `MANIFEST.sha256`, and writes the committed `exp40_C16_admission.json` (run id, platform, rung, train HEAD, RUNTREE digest, NAS path, sha256, bytes, embedded-config sha, EMA inventory sha, audit verdict sha) that the evaluation reads.

### 4.7 Evidence driver (all modes)

`exp40_evidence_driver.py` runs `train.py`'s `main()` in-process with `sys.argv[0]` = the driver, so PL's subprocess DDP re-executes the driver for rank 1 (both ranks instrumented). Observe-only wrappers (each calls the original and returns its result unchanged): `DiffusionCondTrainingWrapper._apply_yaw_aug` + `src.training.diffusion.rotate_scene_metadata` (per-step offsets), `src.models.transformer.flash_attn_func` (call count), `torch.utils.checkpoint.checkpoint` (calls and `use_reentrant`), `on_fit_start` (world size, rank, GPU UUID/name, `SyncBatchNorm` count, the initial-state signature). Evidence is **rank-private**: rank r appends only to `<evidence dir>/rank<r>.jsonl` and writes `rank<r>.summary.json` atomically at `on_train_end`; aggregation happens after all ranks exit. FULL keeps the per-step offsets for all 40,000 steps (≈ 5 MB per rank), so the audit recomputes the whole treatment.

---

## 5. Evaluation plan (this box, after admission)

### 5.1 Protocol

Literally the exp_25 YAW-pin protocol: each argv is derived from the committed exp_25 manifest row of the same kind (`exp_25…/depth_canon_cells.json` YAW-pin rows, file sha pinned), substituting only `--model-config`, `--seed`, `--rotate-*`, `--eval-name`: `--cond-method vanilla --cond-autocast bf16 --cfg-scale 1.0 --steps 1 --batch-size 64 --device cuda --record-per-scene --record-stream --expected-stream-count 6337`, `acousticroom_unseeneval_1.json` (K=1) / `acousticroom_unseeneval.json` (K=8), EMA, `--dump-scored-waveforms /media/diskstation/yixunhu/FLAC/exp40_pred_waveforms` (announcement 08). **SDPA** through the committed, hash-pinned no-flash launcher `overlays/no_flash/exp35_no_flash_launch.py` (exp_25's YAW cells ran without flash-attn; exp_36 reproduced them bit-exactly this way). The evaluator code (`eval_FLAC.py`, `src/` minus tests/localization, `data/AR`) is byte-identical between `d6ca307a`, `localization-exp` and `exp-36-yaw-c4-spread` (verified); this branch adds §2 only (eval constructs the wrapper, never augments).

A thin driver `exp40_eval.py` (Codex r1: the right call) reuses the exp_25/36 pieces — the manifest rows as argv source, `eval_FLAC.py`, the no-flash launcher + its `--preflight` + sentinel, a content-addressed local checkpoint cache + hard-link farm, receipts, skip-only-verified, the exp_36 replay comparator semantics — without touching exp_25/35/36 files or the live `~/codespace/exp-36-run`.

### 5.2 Cells (85 new; reuse pinned)

| Block | Arm | Angles | K | Seeds | New cells |
|---|---|---|---|---|---|
| E0 replay (HARD STOP, first) | YAW | T 0° K1; C4 90° K8; R K8 | — | 42 | 3 |
| T (θ = 0) | C16 | 0 | 1, 8 | 42–46 | 10 |
| C4 | C16 | 90, 180, 270 | 1, 8 | 42–46 | 30 |
| G45 (on the C16 grid) | C16 | 45, 135, 225, 315 | 1, 8 | 42 | 8 |
| OFF (balanced half-step coset, D5) | C16 | 11.25, 101.25, 191.25, 281.25 | 1, 8 | 42 | 8 |
| R (random yaw over 512 columns, rotate seed = eval seed) | C16 | random | 1, 8 | 42–46 | 10 |
| FILL (comparators that exist nowhere) | YAW, P1 | 11.25, 101.25, 191.25, 281.25 | 1, 8 | 42 | 16 |

**Reused, never re-run (Codex HIGH 10), pinned in a committed comparator registry `exp40_comparators.json`** (built by `exp40_eval.py register-comparators`; per cell: metrics/stream/receipt paths + sha256, `source_sha`, checkpoint sha/bytes, model/dataset config shas, `input_hash`/`assignment_hash`, dump dir + `waveforms.sha256` + manifest sha, or a documented dump exemption): exp_25 YAW and P1 T (s42–46), R (s42–46), C4 + 45° (s42) in `~/codespace/exp-25-depth-canon/outputs_FLAC/exp25_depth_canon/{YAW,P1}/`; exp_36 YAW and P1 135°/225°/315° K∈{1,8} s42 (12 cells, `~/codespace/exp-36-run/outputs_FLAC/exp25_depth_canon/{YAW,P1}/`, `exp25_YAW_C4_…`/`exp35_P1_OFF_…`); exp_36 YAW C4 s43–46 (24 cells, produced dump-free under exp_36's plan — recorded as a historical dump exemption). The registry refuses incomplete or mixed sets per block and is built only once exp_36 has published; the collector re-hashes every registered file. Evaluator equivalence across pins: E0 proves this pin reproduces exp_25 exactly; exp_36's four replay hard stops proved the same for its pin.

**E0.** Re-run `exp25_YAW_T_S40000_K1_s42`, `exp25_YAW_C4_S40000_K8_s42_th90`, `exp25_YAW_R_S40000_K8_s42` at this pin; references pinned by sha256 (metrics, stream sidecar, dump `waveforms.sha256`). EQUAL ⇔ `metrics` and `by_scene` recursively, type-strictly equal; every other metrics key equal except `source_sha`, `eval_name`, `ckpt_path`; stream sidecar sha256 equal; `waveforms.sha256` content equal. The verdict is an atomic **E0 receipt** bound to HEAD, the closure digest, `eval_FLAC.py` + no-flash overlay/launcher shas, the environment digest, the reference shas, the checkpoint/config pins and the per-(K, seed) input-hash map; `run` refuses every non-E0 cell — including `--only` selections — without an EQUAL receipt for the current HEAD and closure. E0 needs no C16 checkpoint and can run as soon as one card is free.

### 5.3 Per-cell verification (Codex HIGH 9)

rc 0; the cell log carries exactly one `EXP35_NO_FLASH_LAUNCHER` sentinel with backend `sdpa` and the pinned overlay/launcher shas; the metrics JSON has exactly the exp_25 key set and: `cond_method "vanilla"`, `cond_autocast "bf16"`, `frame_avg_angles null`, `frame_avg_fwd_cap null`, `orbit_execution "n/a"`, `batch_size 64`, `cfg_scale 1.0`, `steps 1`, `device "cuda"`, `weights_source "ema"`, `n_samples 6337`, `scene_count 10`, `per_scene_schema 1`, `by_scene` with exactly the ten room-type keys, `seed`/`rotate_deg` as specified (random: `rotate_deg null`), `dataset_config` == the row's, `source_sha` == HEAD, `ckpt_sha256_loaded`/`ckpt_bytes_loaded` == the pin, every metric key of the exp_25 set present and finite; stream sidecar: schema 1, `stream_count 6337`, `img_w 512`, `rotate_mode`/`rotate_deg`/`rotate_seed` as specified, recomputed `input_hash`/`assignment_hash` from its tuples equal to the stored ones, fixed-mode `offsets` all equal to the angle's column shift, random-mode `offsets` equal to the eval-seed draw, and `input_hash` (fixed) / `assignment_hash` (random) equal to the pinned exp_25 YAW value for the same (K, seed); dump: `COMPLETE`, `manifest.json` with `n 6337`, `shape [6337, 10240]`, `dtype float32`, `input_hash` == the stream's, `metrics_stem` == the cell's, `metrics_sha256`/`stream_sidecar_sha256` == the files', `waveforms.sha256` == `manifest.sha256` == the npy's re-hash. A receipt `<stem>.receipt.json` is written atomically only after all of it; `run` skips only a cell whose receipt and artifacts re-verify; a cell with artifacts but no receipt is refused, never overwritten. Lanes: a card with no compute process and ≥ 10 GiB free at spawn (re-checked immediately before), else wait; nothing is ever signalled; the evaluation takes the same exp_40 card lock as training.

### 5.4 Collect, figure, table (Codex HIGH 11)

`exp40_eval.py collect` → `yaw_aug_c16_cells_raw.csv` (one row per validated new or registered cell: arm, block, K, seed, rotate mode/deg, every metric, metrics/stream sha, source pin) + `yaw_aug_c16_results_tables.md` (θ = 0 table, C4/G45/OFF/R per-angle tables, the §1 off-grid readout); blocks missing any registered cell render PENDING. Figure: D6. `model_comparison.md` (announcement 04): this round adds to `gen_model_comparison.py` the row specs "yaw-aug C16 vanilla @40k (exp_40)" K=1/K=8 (`outputs_FLAC/exp40_yaw_aug_c16/farm/C16/*_metrics_1_1.0_exp40_C16_T_S40000_K{k}_s4[2-6].json`) under a contract `exp40` that calls `exp40_eval.py`'s offline validator (metrics + stream + receipt + admission, no checkpoint access); at results time the ten T metrics JSONs, stream sidecars, receipts and the admission record are force-added to git with a sha256 manifest, and the regenerated table (on neuronic, per the standing table-regen trap) + evidence + commit + push form one results update.

### 5.5 GPU time

Exclusive-card per-cell times (exp_35 table, YAW, no-flash): K=1 11.9 min, K=8 12.6 min. New cells: 42 at K=1, 43 at K=8 ⇒ 42 × 11.9 + 43 × 12.6 ≈ 1,042 min ≈ **17.4 GPU-h ≈ 8.7 h wall on two exclusive cards** (≈ 17.4 h on one; 1.6–5× longer next to a 100 %-utilisation co-tenant). E0 alone ≈ 37 min. NAS dumps ≈ 85 × 260 MB ≈ 22 GB (14 TB free). Adding the brief's off-grid set (D5) would add 24 cells ≈ 4.9 GPU-h.

---

## 6. Per-file plan (TDD; tests in `src/tests/`)

| File | Change / tests |
|---|---|
| `src/data/yaw_rotation.py` | `check_yaw_offset_step`, `quantize_yaw_offsets` (§2.1). |
| `src/training/factory.py` | `YAW_AUG_KEYS += ("offset_step",)`; when enabled and present → `yaw_aug_offset_step`. |
| `src/training/diffusion.py` | `yaw_aug_offset_step=None` appended last; validation; banner suffix; the guarded quantise line. |
| `src/tests/exp40_yaw_aug_golden_capture.py` + `fixtures/exp40_yaw_aug_enabled_golden.json` | E1 (§3). |
| `src/tests/test_yaw_aug_c16.py` | fixture binding (4 pins, record sha, tool sha == committed tool); absent key == golden; s = 1 == golden; s = 32 snapped draws + 16-point support; s = 4 tiny step (offsets, metadata vs `rotate_scene_metadata`, RNG digests); **pre-rotated independent reference: conditioning, loss and all gradients equal**; resume-exactness; exact uniformity over `range(512)`; helper validation; factory (absent → the exp_15 kwargs; present; invalid; disabled block ignores; unknown keys); wrapper (default, disabled-arm refusal, invalid, last parameter, exact banners); `eval_FLAC`-style construction from the C16 config + eval never calls `_apply_yaw_aug`; `_parse_canonicalize_config` refusal with the C16 block. |
| `src/tests/test_yaw_aug_c16_config.py` | §2.3. |
| `exp_40…/exp40_kit.py` | argv, recipe projection, parsed-argv check, RUNTREE build/verify, AR sample digest, environment/versions/origins, flash proof, pins, DINO, config contract, GPU gate, start watcher, smoke/pilot receipt (schema + aggregation of rank-private evidence), cadence publisher (watch / pulled manifest), end audit, admission, neuronic state machine helpers (intent, reconcile, sacct normalisation, partition max, time request). |
| `exp_40…/exp40_evidence_driver.py` | §4.7. |
| `exp_40…/exp40_train_local.sh`, `exp40_train.sbatch`, `exp40_submit_neuronic.sh` | thin bash over the kit (§4.4); DRY_RUN everywhere. |
| `src/tests/test_exp40_kit.py` | every gate's refusal paths (wrong HEAD / dirty closure / untracked file / RUNTREE extra or modified file / wrong link target / AR digest / env var / version / missing flash / config drift / parsed-argv drift / wrong GPU / occupied card / shared FULL / existing namespace / missing or mismatching receipt fields); watcher (pass; step before banner; wrong banner; wrong world; timeout; own-group kill only — a dummy process group); evidence aggregation (pass; missing rank; duplicate or missing step; wrong count; non-multiple of 32; value ≠ recomputation; fixture mismatch; reentrant checkpoint; zero flash; no SyncBN in DDP; rank signatures differ; non-finite loss); cadence publisher (finished-only rule; no clobber; source change; manifest atomicity; pulled-manifest mode); audit + admit on a tiny synthetic checkpoint (pass + each failure field); submitter state machine with stub `sbatch`/`squeue`/`sacct`/`sinfo` (crash after `sbatch` before record → adopted, not resubmitted; live job → refused; prepared-but-absent → same intent resubmitted; terminal-failed → `NEW_INTENT=1` only; query failure fatal; `OUT_OF_ME+` normalised; ambiguous prefix fatal; TIME > MaxTime refused; unpushed commit refused); DRY_RUN of both launchers. |
| `exp_40…/exp40_eval.py`, `exp40_cells.json`, `exp40_comparators.json` (built at eval time) | §5. |
| `src/tests/test_exp40_eval.py` | the 85-row list (counts per block/K, names, `rot_token` stems, argv == the exp_25 template except the substituted flags, no duplicate stems); E0 comparator (EQUAL; each mismatch kind; missing reference refused); E0 receipt binding and non-bypass (`--only` refused without it; stale HEAD/closure refused); every §5.3 verification field; skip / refuse logic; comparator registry (complete; missing; mixed pin; hash drift); collect (PENDING; off-grid readout formula on synthetic values); the offline `model_comparison` validator; a stub-evaluator harness for `run` (no GPU). |
| `worklog/worklog_yixun/gen_model_comparison.py` | the two exp_40 row specs + contract `exp40` (offline validator); test in `test_exp40_eval.py`. |
| `exp_40…/yaw_aug_c16_neuronic_commands.md` | command sheet (branch onto neuronic, DRY_RUN, PILOT, FULL, state recovery, pull + publish + admit). |
| records | `_worklog.md`, `commits_yaw_aug_c16.md`, `_command.md`, `_params_set_up.md` (at launch), `_results.md`, `_analysis.md`, HTML (after results). |

## 7. Validation ladder

1. Static: `py_compile`, `bash -n`, `json.load`, `git diff --check`.
2. Pytest: new tests + existing yaw/factory/train/eval suites (CPU).
3. Real-data readback (CPU): one real AR batch (micro 4) through the dataloader + `_apply_yaw_aug` with s = 32 (offsets multiples of 32; depth `[3, 256, 512]` rolled by exactly d; pose norms and z preserved); the AR sample digest.
4. DRY_RUN of every path (local launcher all modes; submitter + sbatch gates with stubbed Slurm; `exp40_eval.py run` DRY_RUN with E0 + all cells against a harness admission record).
5. GPU smoke (this session, GPU 0 only, ≤ 15 min, waits while another process holds GPU 0): `MODE=SMOKE`, 20 steps, micro 32, evidence driver.
6. Before FULL (not this session): A6000 DDP_SMOKE on both cards, or the neuronic PILOT; then FULL on Yixun's go.
7. Eval: E0 → C16 blocks → FILL → registry → collect.

## 8. Acceptance criteria (FULL)

The launch record's HEAD == EXPECT_SHA and RUNTREE digest == the receipt's; N ranks on the registered GPU model; micro 32 (8), accum 1, SyncBN active; the exact C16 banner and N members before the first step; flash calls counted on every rank (evidence); ≥ 1 optimizer step with finite loss; 16 cadence checkpoints, each on the NAS (A6000) or in the cadence manifest (neuronic); end audit PASS; admission written. Eval: E0 EQUAL; every cell verified (§5.3); comparator registry complete.

## 9. Risks and disclosures

Single training seed; exp_17's launch commit, environment, sampler trace and both reference backends unknown/inferred (§0); a neuronic run differs in GPU model and launcher form; C16 vs YAW share logical draw keys on 2×32, not sample assignment; FILL is seed 42 only; the exp_36 YAW C4 s43–46 cells have no dumps (exemption recorded). GPU 1 belongs to the host-A exp_30 evaluations until ≈ 04:00 Nov 8 — never used, signalled or waited on by this seat; exp_38/exp_39 worktrees are never touched.

## 10. Notebook chronology (Codex NIT 4)

The notebook's E1 entry (21:21) was appended after the 21:28 provenance entry; an append-only clarification records the actual order (E1 capture 21:20–21:21, provenance write-up 21:28).

## 11. Review budget

Plan: r0 → r1 used 1 of 3; this r1 goes to r2. Code: ≤ 3 rounds, ALL code of the experiment (training change, tests, kit, driver, eval driver, collector, generator rows) handed over in round 1.

## 12. Codex plan review r1 — disposition

| Finding | Disposition |
|---|---|
| B1 neuronic submit not exactly-once | §4.4 intent-token state machine (prepared → submitted → reconciled via squeue + sacct), adoption after a crash, NEW_INTENT only after terminal failure, distinct PILOT/FULL namespaces, tests for every transition |
| B2 cadence checkpoints not on the NAS | §4.5 finished-checkpoint publisher (A6000, alongside FULL) + in-job cadence manifest and external pull (neuronic, D4); end audit re-verifies the NAS inventory |
| H1 claim too broad / provenance table | §0 table expanded (launcher, execution, config origin, grad-ckpt, backend, environment, sampler); scoped claim paragraph |
| H2 E1 not bound to its generator; quantised reference too weak | §3: recaptured with the committed tool; fixture binds the tool sha, versions, full pins; test refuses tool drift; pre-rotated independent reference (conditioning, loss, gradients) |
| H3 mutable closure / shadows / asset links | §4.2 immutable RUNTREE from `git archive`, blob-hash verification, exact allowlisted links + realpaths, AR sample digest, VAE/DINO shas, audits from RUNTREE |
| H4 environment + active backend | §4.3-2 pinned versions/runtime/origins/TF32, neuronic drift vs smoke receipt; §4.7 counted flash calls on every rank |
| H5 smoke receipt / shared JSONL | §4.7 rank-private evidence; §4.3-7 full receipt schema |
| H6 exclusivity / crash identity | FULL idle-only, lock then re-check before spawn, shared only for smokes; D3 mechanical (no resume mode; resumed runs never primary) |
| H7 admission not full-state | §4.6 full inventory, strict load, finiteness, EMA, optimizer coverage, scheduler, loops, callback, cadence; per-rank initial signatures |
| H8 off-grid estimand | D5 + §1: balanced coset {11.25, 101.25, 191.25, 281.25}, floor pairing, Δoff readout |
| H9 E0 + verification contract | §5.2 atomic E0 receipt, non-bypass; §5.3 full field list |
| H10 comparator reuse / FILL facts | §5.2 reuse the 12 exp_36 cells + exp_25 cells via a pinned registry; FILL reduced to the 16 off-grid comparator cells; dump exemption recorded; seed alignment per block in §1 |
| H11 figure + table paths | D6 (paper repo) + §5.4 validated CSV; generator row specs + offline validator this round; one results update |
| H12 neuronic contract / wall time | §4.3-1 pushed check; §4.3-5 one node/task, rank vars recorded; §4.4 PILOT-timed `--time`, partition MaxTime refusal |
| NIT 1 policies | §2.1 (i)/(ii) |
| NIT 2 RNG wording | §2.1 |
| NIT 3 totals | §5.2/§5.5 recomputed: 85 cells, 17.4 GPU-h |
| NIT 4 chronology | §10 + notebook clarification |
