# plan_yaw_aug_c16 — exp_40: C16-discrete Yaw-Aug FLAC (offsets Δp ∈ {0, 32, …, 480})

**Author:** Claude Opus 5.5 (Coder/Planner seat, subagent of the main session) · **Date:** 2026-11-07 · **Status:** r0 — before Codex plan review r1 · **Branch / worktree:** `exp-40-yaw-aug-c16` in `~/codespace/exp-40-yaw-aug-c16`, cut from `localization-exp` HEAD `2b42468c8f429b1d93da93b3e4ddb7bd6f8f1932`; `main` is never touched. Experiment folder `worklog/worklog_yixun/exp_40_yaw_aug_c16_claude/`.

"Verified" = re-checked in this session against files/commits; "inferred" = argued from records, not directly observable.

---

## 0. Corrections to the brief and the decisions they create

**C1 — the YAW checkpoint in the existing figures is exp_17's A6000 arm, not exp_15's neuronic arm (verified).** The brief names "exp_15 … its 40k checkpoint sha256 prefix is ac1f26034e4f". Those are two different runs:

| | exp_15 YAWAUG (neuronic) | exp_17 Yaw-Aug (A6000) = **YAW in the figures** |
|---|---|---|
| 40k sha256 | `16b964ec…` (`slurm_train_exp15-YAWAUG-leg40000…_3716203.out`: `AUDITED_SHA256 16b964ec…`) | `ac1f2603…` (`exp_20…/loc_crossarm_admission_YAW.json`, NAS `ar_40k_endpoints/PROVENANCE.md`, `exp_25…/depth_canon_comparator_ledger.md`) |
| rung | 8 ranks × micro 8 × accum 1 (BN batch 8/rank, SyncBN 64), torchrun, 8×L40, 16 chained legs | 2 ranks × micro 32 × accum 1 (SyncBN 64), PL-spawned DDP, 2×RTX A6000, monolithic |
| config | `exp_15…/FLAC_AR_YAWAUG.json` sha `b14caae2…` | the SAME file (admission `config_path`, `config_sha256 b14caae2…`; embedded canonical config `0761a677…`) |
| code | `fb4fc749` (legs 1–?) … `88897e02` (leg 16) | removed worktree `~/codespace/exp-17-yawaug-a6000`, audited base `41aa31dc`; exact launch commit **unrecoverable** |
| figures / cells | not in the exp_25/35/36 figures; `model_comparison.md` row "yaw-aug vanilla YAWAUG @40k (exp_15)" K=8 T60 8.503 | every exp_25 YAW-pin cell, exp_36 YAW-C4S, exp_20/22 YAW; K=8 T60 7.965 ± 0.015 |

The rung is not innocuous: vanilla exp_11 VANL (8×8 L40) vs P1 (2×32 A6000) differ by 0.95 T60 at matched steps (`model_comparison.md`), and the two Yaw-Aug arms by ≈ 0.5 T60 at K=8 (confounded with evaluator era). **So "drops into the existing figures" requires exp_17's recipe: 2 ranks × 32.** With 2 ranks × 32 the counter-based draw keys `(42, global_step, rank ∈ {0,1}, index ∈ 0..31)` are exactly exp_17's, so with the coupled quantisation of §2 every C16 offset is exp_17's offset rounded down onto the 32-column grid.

**D1 (Yixun) — platform / rung of the training run.** Both kits are built; the choice is a launch-time argument, never a default.
- **(a) RECOMMENDED: 2×A6000 on this box, 2×32, monolithic** — identical platform, rung, launcher form (PL-spawned DDP) and draw keys as YAW (exp_17), P1 and B-F. ≈ 22 h exclusive on both cards (exp_17 measured 1.966 s/step steady state). Conflicts with exp_38's queue on the same two cards (its plan r2 puts six 2-card runs there, ≈ 10–12 days) and with the GPU-1 evaluations until ≈ 04:00 Nov 8 — the ordering is Yixun's call.
- **(b) neuronic 2×L40, 2×32, monolithic** — same rung and draw keys; GPU model differs (no 2×32 L40 run in the record bounds that effect) and the launcher is torchrun (exp_15's form). Labelled a *platform sensitivity*, still the right rung. Est. 17–25 h service time; needs a partition wall limit ≥ ≈ 26 h, otherwise it cannot run monolithically (see D3).
- **(c) neuronic 8×L40, 8×8** — exp_15's recipe as the brief literally asked. Its comparator is then exp_15's `16b964ec` (not in the figures, whose cells live on neuronic under another evaluator era) and the draw keys/rank partition differ from exp_17's. ≈ 8–12 h. Not recommended for the figure question.

**D2 (Yixun) — the brief's "default platform: neuronic".** Kept as documentation of the request; the launch kit refuses to run without an explicit platform/rung, and the command sheet lists (a) first, (b) second, (c) third.

**D3 (Yixun) — crash policy.** Primary = one uninterrupted 0→40k run. PL resume restores model/optimizer/scheduler/EMA/loop counters but not the dataloader iterator or the RNG streams (CLAUDE.md "Chunked chain training"), so a resumed run is a different trajectory. Default: a crash → rerun from step 0 under a fresh run id. Alternative (Yixun may choose): resume and label the arm "resumed at N" everywhere. The kit implements only the default; the command sheet shows the labelled resume command for completeness.

---

## 1. Question and estimand

**Question.** Restricting the existing per-sample yaw augmentation to the C16 group (column offsets d ∈ {0, 32, …, 480}, θ = k·22.5°): (i) what does it cost or gain at θ = 0, (ii) is the model as yaw-robust as YAW on its own training grid, and (iii) — the real test — does robustness survive *between* grid points?

**Arms (all EMA, vanilla conditioning).** **C16** (new, this experiment) · **YAW** = exp_17 `ac1f2603…` (uniform over 512 columns) · **P1** = exp_07 `c4c67882…` (no augmentation). One training run per arm (seed 42); five eval seeds estimate evaluation variability only (scope statement, as in exp_15/17).

**Estimand.** Per (arm, K ∈ {1, 8}, rotation cell, eval seed): the top-level item-weighted `metrics` of `eval_FLAC.py` on the full unseen split (6,337 items / 17 rooms — announcement 01; the convention of every exp_17/25/36 YAW cell). Readouts: θ = 0 table (mean ± sample SD over seeds 42–46); per-angle metrics and the degradation relative to the arm's own θ = 0 cell; for each arm the spread over the C4 / 45°-grid / off-grid blocks; and the **off-grid contrast** `m(C16, off-grid θ) − m(C16, nearest on-grid θ)` next to the same contrast for YAW. No new hypothesis test is pre-registered beyond descriptive paired contrasts (single training seed); the analysis will state which of "flat between grid points" / "degrades between grid points" the data show, read off the C16-minus-YAW off-grid gap against the arms' own seed-42 θ = 0 → C4 spread.

---

## 2. The treatment: one new optional key

### 2.1 Semantics (coupled quantisation)

`training.yaw_aug.offset_step = s` (int). With the key present the offset of sample `i` at `(global_step, global_rank)` is

```
u_i = the existing draw: torch.randint(0, img_w, generator = MT19937 seeded by _yaw_aug_step_seed(seed, step, rank))[i]
d_i = s * floor(u_i / s)
```

Because `s | img_w`, `k_i = floor(u_i / s)` is **exactly uniform on {0, …, img_w/s − 1}** (each k has exactly s preimages among the 512 equiprobable u), independent across samples, and counter-based/resume-exact exactly as before (nothing new is drawn; no RNG stream is touched). For s = 32, img_w = 512: 16 offsets {0, 32, …, 480}, θ = k·22.5°. The coupling property: at every `(step, rank, index)` the C16 offset is the reference arm's offset rounded down onto the grid — on the 2×32 rung (D1 a/b) the arm differs from exp_17 only in the offset set, even draw by draw (same data order modulo platform nondeterminism). `s = 1` reproduces the absent path exactly (tested). Rejected alternative: a second `randint(0, img_w/s)` — equally uniform, but decoupled from the reference draws for no gain.

Validation (fail-closed `ValueError`, the exp_15 convention): `s` an int and not a bool; `1 ≤ s < img_w`; `img_w % s == 0`. Checked in the factory parser and again in the wrapper constructor (same helper); an `offset_step` passed to the wrapper with `yaw_aug_enabled=False` is refused (it would be a silent no-op).

### 2.2 Absent key ⇒ exp_15/exp_17 behaviour byte-identical

- The factory passes **no new kwarg** when the key is absent (the wrapper call for every existing config is literally the pre-change call; the existing exp_15 kwarg tests stay green).
- The wrapper's new parameter is appended LAST in the signature (`yaw_aug_offset_step=None`), so no positional caller shifts.
- `_apply_yaw_aug` gains one guarded line after the draw: `if self.yaw_aug_offset_step is not None: offsets = quantize_yaw_offsets(offsets, img_w, s)`.
- The banner is the exp_15 line verbatim when the key is absent (`yaw_aug ENABLED img_w=512 seed=42`) and `yaw_aug ENABLED img_w=512 seed=42 offset_step=32` when present — distinct whole lines, so a launcher gate can never accept one arm's log for the other.

### 2.3 Config

`exp_40…/FLAC_AR_YAWAUG_C16.json` = the bytes of `exp_15…/FLAC_AR_YAWAUG.json` (sha `b14caae2…`, read-only) with exactly `"seed": 42\n` → `"seed": 42,\n            "offset_step": 32\n` inside `training.yaw_aug`. Pytest: forward byte construction from the base file (unique anchor, exact bytes, no trailing newline as in the base); parsed objects type-strictly equal after deleting `offset_step`; the base sha pinned; `gradient_checkpointing: true` on both ViT conditioners, EMA on, no `cond_method`, ViT `img_w` 512 == `yaw_aug.img_w`.

---

## 3. Equivalence to the reference arms' training code (the "same code" proof)

**E1 — enabled-path golden, captured at four source trees (done, evidence for this plan).** A capture tool (planned file `src/tests/exp40_yaw_aug_golden_capture.py`, draft run from the scratchpad) records from a given source tree: (tiny) four whole `training_step`s of the exp_15 golden harness's tiny CPU model (real factories, real DiT, flash neutralised) with `yaw_aug` enabled at `(step, rank)` ∈ {(0,0), (7,0), (7,1), (2500,3)} — per-sample offsets reaching `rotate_scene_metadata`, metadata and conditioning digests reaching the model, loss (`float.hex`), global RNG digests either side; (real) the offsets `_apply_yaw_aug` draws at img_w 512, seed 42, for **both** rungs (8 ranks × 8 and 2 ranks × 32) at steps {0, 1, 2, 3, 1000, 2499, 2500, 39999}, a digest of one rotated batch, and global-RNG isolation. Run against `git archive` exports of **`fb4fc749`** (exp_15's first training pin), **`88897e02`** (exp_15's last leg), **`41aa31dc`** (exp_17's audited base) and **`2b42468c`** (our base): **all four records are byte-identical**, canonical sha256 `76c1f63419f1038d4ceb22cc788941ef5843ac5699c44603f17caa6181427909`. (Note: the tiny model's loss is the same in all four cases because its DiT's conditioning paths are zero-initialised; the metadata and conditioning digests are what bind the transform — they differ across cases as they must.) The record becomes the fixture `src/tests/fixtures/exp40_yaw_aug_enabled_golden.json`, with the four pins and the per-pin record sha in `_meta`.

**E2 — regression tests at HEAD (planned, §6):** offset_step absent ⇒ the capture equals the fixture exactly; offset_step = 32 (real) / 4 (tiny, img_w 16) ⇒ offsets equal `s·⌊golden/s⌋` elementwise, metadata into the conditioner equal to `rotate_scene_metadata` by those offsets, RNG digests equal to the golden's; offset_step = 1 ⇒ equal to the golden. The existing exp_15 disabled-path golden (`test_yaw_aug_training.py::test_disabled_path_matches_golden`) stays green.

**E3 — closure diff audit (verified, `git diff --stat`).** `41aa31dc → 2b42468c`, training closure (`train.py`, `defaults.ini`, `src/` minus tests, `data/AR`): `train.py`, `defaults.ini`, `data/AR` unchanged; `src/training/diffusion.py` (+223) and `factory.py` (+69) = exp_25 canonicalisation, gated by `training.canonicalize` (absent here; `transfer_batch_to_device` returns the parent call when disabled); `src/data/yaw_rotation.py` (+259) = canonicalisation helpers only; `src/metrics/*` = RAF dataset branches (dataset ≠ RAF); RAF configs/`RAF_md.py`, `src/tools/depth_canon_stats.py`, `src/localization/*` = not imported by training. `fb4fc749 → 41aa31dc`: exp_16 ARE (gated by `training.are_lambda`, absent) **plus one unconditional, default-preserving change**: `dataset.py`/`utils.py` publish `RandomTimeShift.last_shift` as `info["time_shift"]` (no extra RNG value, audio bit-identical; an extra int key the conditioners ignore). Our own diff adds only §2.

**What cannot be proven (disclosed):** exp_17's exact launch commit (removed worktree; E1 covers its audited base and both sides of it), and both reference arms' training attention backend (no "flash_attn not installed" line in exp_15's training log or exp_17's Aug-16 eval logs ⇒ flash-attn inferred, not proven). Our training runs with flash-attn **positively asserted** (§4.2). CUDA training is not bitwise reproducible anyway; the equivalence claim is "same algorithm, same seeds, same recipe, same draws (quantised)", not trajectory identity.

---

## 4. Training protocol and launch kit

### 4.1 Literal argv (both platforms; only the launcher prefix and `--num-gpus/--batch-size` differ)

```
train.py --model-config worklog/worklog_yixun/exp_40_yaw_aug_c16_claude/FLAC_AR_YAWAUG_C16.json
  --dataset-config src/configs/dataset_configs/AR/train/acousticroom_train.json
  --pretransform-ckpt-path weights/FLAC/VAE.safetensors
  --max-steps 40000 --batch-size <32|8> --accum-batches 1 --num-workers 6 --seed 42
  --num-gpus <2|8> --num-nodes 1 --strategy ddp_find_unused_parameters_true --sync-batchnorm true
  --precision bf16-mixed --val-every -1 --val-dataset-config '' --gradient-clip-val 0.0
  --logger wandb --checkpoint-every 2500
  --name FLAC_exp40_C16 --experiment-name <RUN_ID> --save-dir outputs_FLAC/exp40_yaw_aug_c16
```

= exp_15's literal argv shape (its leg-1 transcript) with the config, names and rung substituted. W&B: `WANDB_MODE=offline` (the flac env's wandb 0.15.4 cannot create online runs on this box — CLAUDE.md exp_25 trap; synced later with `~/.venvs/wandb-sync`); same offline mode on neuronic (an online-mode failure would kill the job; logging is not part of the recipe). RUN_ID ∈ {`exp40_C16_a6000_2x32`, `exp40_C16_neuronic_2x32`, `exp40_C16_neuronic_8x8`} (+ `_r<n>` for a rerun after a crash); never containing `exp14_`. Checkpoints land in `outputs_FLAC/exp40_yaw_aug_c16/FLAC_exp40_C16/<RUN_ID>/checkpoints/epoch=*-step=<2500·j>.ckpt` (16 files).

### 4.2 `exp40_train_local.sh` — the 2×A6000 launcher (this box)

`MODE=DRY_RUN|SMOKE|DDP_SMOKE|FULL`, `EXPECT_SHA=<40-hex>` (FULL/DDP_SMOKE: must equal HEAD), `RUN_ID`. Gates, all before any spawn and all fail-closed (exit 2, nothing started):
1. HEAD == EXPECT_SHA; `git status --porcelain --untracked-files=all` empty for the closure (`train.py defaults.ini src data/AR` + the arm config + the kit), so no untracked import shadow under `src/`; `PYTHONPATH`/`PYTHONSTARTUP` unset; no `flash_attn*`/`sitecustomize*` entry at the repo root.
2. Interpreter `/home/yixunhu/miniconda3/envs/flac/bin/python`; versions torch 2.7.0, pytorch_lightning 2.1.0, transformers 4.57.0, flash_attn 2.7.4.post1; **positive flash proof**: `src.models.transformer.flash_attn_func is not None` in that interpreter with the launch environment.
3. Pins (sha256): arm config (pinned in the kit), base config `b14caae2…`, `acousticroom_train.json`, `data/AR/train.json`, VAE `8d82159e…`; the DINOv3 snapshot directory present in the HF cache (`HF_HUB_OFFLINE=1`).
4. Parsed config contract via the real factory (`_parse_yaw_aug_config` → `{enabled, 512, 42, offset_step 32}`; no `cond_method`/`canonicalize`/`are_lambda`/`frame_avg_max_fwd_samples`; EMA on; grad-ckpt on both ViTs).
5. GPUs (FULL/DDP_SMOKE): exactly two visible `NVIDIA RTX A6000`; `GPU_POLICY=idle` (default) refuses if ANY compute process is on GPU 0 or 1 (uuid-mapped, query failure fatal); `shared` (explicit opt-in) requires ≥ 24 GiB free per card and records the co-tenants. A machine-wide flock `outputs_FLAC/exp40_yaw_aug_c16/.cards.lock` is held until the end audit (our trainings and evaluations never overlap each other).
6. Run namespace fresh (the RUN_ID dir must not exist); ≥ 30 GiB free on the local disk; per-run flock.
7. A DDP_SMOKE PASS receipt for the same HEAD and RUN_ID-independent recipe projection (argv minus names/max-steps/checkpoint cadence) is required before FULL.

Launch: `setsid` + `tee` into `worklog/…/exp_40…/yaw_aug_c16_<ts>_<MODE>_train.log` (kept untracked while running); an at-launch record `yaw_aug_c16_<ts>_<MODE>_launch.json` (argv, HEAD, file shas, versions, `pip freeze` sha, GPU UUIDs, co-tenants, host) written before the fork and committed with the command file. **Start watcher:** the exact whole line `yaw_aug ENABLED img_w=512 seed=42 offset_step=32` must appear before the first training-progress line and two `MEMBER: n/2` rank lines must appear, else the watcher terminates ONLY its own process group (exit 8). **End audit** (`exp40_ckpt.py audit`): rc 0; the 16 cadence checkpoints exactly; the 40k checkpoint read once: `global_step == 40000`, embedded `model_config` canonical == the arm config, `training.yaw_aug` literal, EMA inventory mirrors the online DiT inventory (key/shape/dtype), optimizer + `InverseLR` scheduler state present, strict load into a freshly built wrapper on CPU; `CHECKPOINTS.sha256` written for all 16; a terminal JSON verdict.

`SMOKE` (the only GPU use this session, GPU 0 only, ≤ 15 min): one rank, `--num-gpus 1` (no SyncBN — disclosed: a single-rank smoke cannot exercise DDP/SyncBN), micro 32, 20 steps, `--checkpoint-every 1000000` (zero checkpoints asserted), output under `outputs_FLAC/exp40_smoke/`, run through **`exp40_smoke_driver.py`**: it wraps `DiffusionCondTrainingWrapper._apply_yaw_aug` + `src.training.diffusion.rotate_scene_metadata` (observe-only) and appends `{rank, global_step, offsets}` per call to a JSONL; the post-check recomputes every line independently (`_yaw_aug_step_seed` → `draw_yaw_offsets` → `·32⌊/32⌋`) and requires exact equality, all offsets ∈ {0, 32, …, 480}, contiguous steps 0..19, plus finite losses, the banner, peak memory (nvidia-smi sampler) and the post-warm-up tqdm rate. `DDP_SMOKE`: the same on both cards with the production rung (2×32, SyncBN, `ddp_find_unused_parameters_true`); PL re-executes `sys.argv[0]` for rank 1, so the driver instruments both ranks; additionally asserts 2 members and `SyncBatchNorm` modules present (the driver counts them in `on_fit_start` of each rank). DDP_SMOKE needs both cards free — not run this session (GPU 1 is someone else's).

### 4.3 Neuronic: `exp40_train.sbatch` + `exp40_submit_neuronic.sh`

Yixun submits; this seat has no neuronic access. `RUNG=2x32|8x8` (required, no default), `MODE=SMOKE|FULL`, `EXPECT_SHA` (must equal the checked-out HEAD), `TIME` (default 26:00:00 for 2x32, 16:00:00 for 8x8; Yixun's partition limit governs — D1/D3). The sbatch carries exp_15's environment conventions verbatim (`/n/fs/gatrdp/envs/flac/bin/python`, `PYTHONNOUSERSITE=1`, `HF_HOME=/n/fs/gatrdp/hf_cache`, `HF_HUB_OFFLINE=1`, `PYTHONUNBUFFERED=1`, `unset PYTHONPATH`, SLURM rank variables unset before `torchrun --standalone --nnodes=1 --nproc_per_node=N` so `train.py:100` cannot add a rank offset to the seed) and the same gates as §4.2 1–4 and 6 (GPU gate: `nvidia-smi` names == N × `NVIDIA L40`, `--gres=gpu:l40:N`), the same start watcher (N members), end audit and `CHECKPOINTS.sha256`. Resources (exp_15's formula, raised for micro 32): 2x32 → 2 GPUs / 22 CPUs / 64 G; 8x8 → 8 / 64 / 108 G. The submitter holds a flock, refuses if `squeue -h -u $USER -n exp40_C16_<rung>` lists anything (query failure fatal), submits ONE monolithic job, appends the line to `yaw_aug_c16_neuronic_submissions.log`; `DRY_RUN=1` prints the exact `sbatch` line and runs the sbatch's gates in `EXP40_DRY_GATES=1` mode, submitting nothing. SMOKE = 20 steps on the same rung with the smoke driver, no checkpoints (one short job before FULL).

### 4.4 Checkpoint destination (announcement 07)

`/media/diskstation/yixunhu/FLAC/checkpoints/exp40_yaw_aug_c16/<RUN_ID>/` via `exp40_ckpt.py publish --src <checkpoints dir> --run-id <RUN_ID>`: verifies every source file against the run's `CHECKPOINTS.sha256`, copies each to a temp name in the destination, re-hashes the NAS copy, renames without clobbering (refuses an existing target), writes `MANIFEST.sha256` (+ `PROVENANCE.md`: run id, HEAD, platform, rung, argv). A6000: run on this box after the end audit. Neuronic: Yixun first pulls the run's checkpoint dir to this box (command sheet: `rsync -av --partial <user>@neuronic.cs.princeton.edu:/n/fs/gatrdp/codespace/FLAC/outputs_FLAC/exp40_yaw_aug_c16/FLAC_exp40_C16/<RUN_ID>/checkpoints/ <staging>`), then runs `publish` from the staging dir. Then `exp40_ckpt.py admit` writes the committed admission record `exp40_C16_admission.json` (the §4.2 audit fields + NAS path + sha) that the evaluation reads.

---

## 5. Evaluation plan (this box, after admission)

**Protocol = the exp_25 YAW-pin cells' protocol, literally.** Every argv is derived from the committed exp_25 manifest row of the same kind (`worklog/…/exp_25_depth_canon_claude/depth_canon_cells.json`, YAW-pin rows `exp25_YAW_{T,R,C4}_…`, file sha pinned), substituting only `--model-config` (the arm's config), `--seed`, `--rotate-*` and `--eval-name`: `--cond-method vanilla --cond-autocast bf16 --cfg-scale 1.0 --steps 1 --batch-size 64 --device cuda --record-per-scene --record-stream --expected-stream-count 6337`, unseen split via `acousticroom_unseeneval_1.json` (K=1) / `acousticroom_unseeneval.json` (K=8), EMA (the checkpoint's `use_ema`), `--dump-scored-waveforms /media/diskstation/yixunhu/FLAC/exp40_pred_waveforms` (announcement 08). **Backend = SDPA** through the committed, hash-pinned no-flash launcher `overlays/no_flash/exp35_no_flash_launch.py` (exp_25's YAW cells ran without flash-attn; exp_36 reproduced them bit-exactly this way). The evaluator code (`eval_FLAC.py`, `src/` minus tests/localization, `data/AR`) is byte-identical between the exp_25 pin `d6ca307a`, `localization-exp` and `exp-36-yaw-c4-spread` (verified `git diff --stat`, empty); our branch adds §2 only.

**Why a thin exp_40 driver instead of the exp_25/36 campaign runner itself.** That runner is hard-wired to the exp_25 folder, its manifest and farm, and the no-flash routing for new blocks exists only on the unmerged `exp-35`/`exp-36` branches (≈ 10k lines not in `localization-exp`); its live run worktree `~/codespace/exp-36-run` must not be committed to. `exp40_eval.py` reuses its pieces — the manifest rows as the argv source, `eval_FLAC.py`, the no-flash launcher and its sentinel, the content-addressed checkpoint cache + hard-link farm, the receipt/verify/skip-only-verified discipline, the exp_36 replay comparator semantics — without touching exp_25/35/36 files.

### 5.1 Cells (97)

| Block | Arm | Angles | K | Seeds | Cells |
|---|---|---|---|---|---|
| E0 replay (HARD STOP, runs first) | YAW | T 0° K1; C4 90° K8; R K8 | — | 42 | 3 |
| T (θ = 0) | C16 | 0 | 1, 8 | 42–46 | 10 |
| C4 | C16 | 90, 180, 270 | 1, 8 | 42–46 | 30 |
| G45 (45°-grid; ON the C16 grid, k = 2, 6, 10, 14) | C16 | 45, 135, 225, 315 | 1, 8 | 42 | 8 |
| OFF (half-steps, 16/48/80/144 columns, 11.25° from the grid) | C16 | 11.25, 33.75, 56.25, 101.25 | 1, 8 | 42 | 8 |
| R (random yaw, uniform over 512 columns, rotate seed = eval seed; exp_14/15 R protocol) | C16 | random | 1, 8 | 42–46 | 10 |
| FILL (comparators missing at these angles) | YAW, P1 | 135, 225, 315, 11.25, 33.75, 56.25, 101.25 | 1, 8 | 42 | 28 |

Existing comparator cells reused (never re-run; verified on disk in `~/codespace/exp-25-depth-canon/outputs_FLAC/exp25_depth_canon/{YAW,P1}/`): YAW and P1 T (s42–46), R (s42–46), C4 + 45° (s42); YAW C4 s43–46 from the exp_36 chain once it publishes. The R cells for YAW and P1 therefore already exist — none are added. FILL re-measures P1 135°/225° even though the exp_36 chain also produces them, so every new angle sits at one evaluator pin.

E0: the three cells re-run `exp25_YAW_T_S40000_K1_s42`, `exp25_YAW_C4_S40000_K8_s42_th90`, `exp25_YAW_R_S40000_K8_s42` at our pin; exact equality required against the archived exp_25 JSONs (pinned sha256): `metrics` and `by_scene` strictly equal (recursive, type-strict), every other key equal except `source_sha`, `eval_name`, `ckpt_path`; the `.stream.json` sha256 equal; the dump's `waveforms.sha256` equal to the archived exp_25 dump's. Any mismatch stops the campaign (exit 1, nothing else runs) and is reported to Yixun. E0 does not need the C16 checkpoint and can run as soon as one card is free.

Names: `exp40_<ARM>_<BLOCK>_S40000_K<K>_s<seed>[_th<tok>]` with `tok` = `eval_FLAC.rot_token` (11.25 → `11p25`); metrics files `<ckpt stem>_metrics_1_1.0_<name>[_rot<tok>|_rotrand<seed>].json` in the exp_40 farm `outputs_FLAC/exp40_yaw_aug_c16/farm/<ARM>/`.

### 5.2 `exp40_eval.py` (driver) — subcommands

`plan` (writes/prints the deterministic 97-row cell list; `--check` against the committed `exp40_cells.json`), `run --lanes 0[,1] [--blocks …] [--only …]` (EXPECT_HEAD == HEAD; closure clean; no-flash overlay + launcher sha pinned and CPU `--preflight` run; checkpoint → local cache `<farm>/.ckpt_cache/<sha>.ckpt` copied, re-hashed, 0444, then a hard link per arm, re-hashed before each spawn; per lane: no compute process on the card and ≥ 10 GiB free at spawn, else wait (poll) — never signals anything; per cell log in the experiment folder; after rc 0 verify → atomic receipt `<stem>.receipt.json`; skip only a verified cell; a cell with artifacts but no receipt is refused, never overwritten; E0 runs first and gates the rest), `verify` (every receipt re-checked), `replay-check` (E0 verdict file), `collect` (validated cells → `yaw_aug_c16_cells_raw.csv` + a summary markdown table in the shape of `analysis/yaw_robustness/generated/yaw_c4_metrics_raw.csv`; incomplete blocks render PENDING). DRY_RUN prints each cell's exact spawn command. Cell verification: rc 0; the sentinel line `EXP35_NO_FLASH_LAUNCHER … backend=sdpa`; metrics JSON fields `cond_method == "vanilla"`, `cond_autocast == "bf16"`, `rotate_deg`/`seed` as specified (random: `rotate_deg null` and `_rotrand<seed>`), `n_samples == 6337`, `scene_count == 10`, `weights_source == "ema"`, `ckpt_sha256_loaded`/`ckpt_bytes_loaded` == the admitted pin, `source_sha == HEAD`, finite metric values with the exact exp_25 key set; stream sidecar present; dump dir with `COMPLETE` and `waveforms.sha256` matching `manifest.json`.

### 5.3 GPU-time estimate

Exclusive-card per-cell times measured for YAW under the no-flash backend (exp_35 cell-time table): K=1 median 11.9 min, K=8 12.6 min (+ dump write ≈ 260 MB/cell to the NAS). 97 cells ≈ 48 × 11.9 + 49 × 12.6 min ≈ **19.8 GPU-h ≈ 10 h wall on two exclusive cards** (≈ 20 h on one; 1.6–5× longer next to a 100 %-utilisation co-tenant). E0 alone: 3 cells ≈ 37 min on one card. NAS: ≈ 25 GB of dumps (14 TB free).

---

## 6. Per-file plan (TDD; tests in `src/tests/`)

| File | Change |
|---|---|
| `src/data/yaw_rotation.py` | `check_yaw_offset_step(offset_step, img_w)` (validation, returns the int) and `quantize_yaw_offsets(offsets, img_w, offset_step)` (int64 1-D in `[0, img_w)` → `offset_step * (offsets // offset_step)`, same dtype/device, input untouched). ≈ +35 lines. |
| `src/training/factory.py` | `YAW_AUG_KEYS` += `"offset_step"`; when enabled and present: `check_yaw_offset_step` → `kwargs["yaw_aug_offset_step"]`. ≈ +8. |
| `src/training/diffusion.py` | `yaw_aug_offset_step=None` appended last; validation (refused without `yaw_aug_enabled`); banner suffix; the one guarded quantise line in `_apply_yaw_aug`. ≈ +20. |
| `src/tests/exp40_yaw_aug_golden_capture.py` | the E1 capture tool (helper module, not collected). |
| `src/tests/fixtures/exp40_yaw_aug_enabled_golden.json` | the four-pin fixture. |
| `src/tests/test_yaw_aug_c16.py` | (1) HEAD capture == fixture (absent key); (2) fixture `_meta`: 4 pins, equal per-pin record sha == canonical sha of the record; (3) real offsets with s = 32 == `32⌊g/32⌋`, support ⊆ {0,…,480}; (4) tiny with s = 4: offsets, metadata digests vs a direct `rotate_scene_metadata` reference, conditioning shapes, RNG digests == golden; (5) s = 1 == golden; (6) exact uniformity: `quantize` maps `range(512)` onto each of the 16 grid points exactly 32 times; (7) helper validation (0, −1, 30, 512, 600, True, 32.0, "32", out-of-range / non-int64 / 2-D offsets); (8) factory: absent → exactly the three exp_15 kwargs; present → + `yaw_aug_offset_step`; each invalid value raises; disabled block with the key → `{}`; (9) wrapper: default None; refused without enabled; invalid refused; exact banner lines both ways; (10) resume-exactness with s = 32 (fresh wrapper at step N == sequential); (11) global RNG untouched by quantisation. |
| `src/tests/test_yaw_aug_c16_config.py` | §2.3 byte construction + parsed single delta + base sha + recipe pins + factory parse of the real file. |
| `exp_40…/FLAC_AR_YAWAUG_C16.json` | §2.3. |
| `exp_40…/exp40_train_local.sh`, `exp40_smoke_driver.py`, `exp40_ckpt.py` (`audit`, `publish`, `admit`), `exp40_train.sbatch`, `exp40_submit_neuronic.sh` | §4. Tests: `src/tests/test_exp40_kit.py` — DRY_RUN of the local launcher and of the sbatch gates under stubbed `nvidia-smi`/`git`-clean fixtures (refusals: wrong HEAD, dirty closure, untracked shadow, wrong GPU names, occupied card, existing run dir, missing smoke receipt, wrong rung/arg); smoke-driver checker on synthetic JSONL (pass; a non-multiple of 32; a wrong value; a missing step); `exp40_ckpt.py audit/publish/admit` on a tiny synthetic checkpoint (pass; wrong step; config drift; EMA mismatch; sha mismatch; clobber refused). |
| `exp_40…/exp40_eval.py`, `exp40_cells.json` | §5. Tests: `src/tests/test_exp40_eval.py` — the 97-row list (counts per block, names, argv equal to the exp_25 template except the substituted flags, off-grid tokens, no duplicate stems); replay comparator (equal / each mismatch kind / missing reference → refused); verify (each field failure); skip/refuse logic; collect (PENDING on incomplete blocks). A stub-evaluator harness for `run` (no GPU). |
| `exp_40…/yaw_aug_c16_neuronic_commands.md` | command sheet (getting the branch onto neuronic, DRY_RUN, SMOKE, FULL, the transfer + publish + admit steps). |
| records | `_worklog.md`, `commits_yaw_aug_c16.md`, `_command.md`, `_params_set_up.md` (at launch), `_results.md`, `_analysis.md`, HTML (after results). |

`model_comparison.md` (announcement 04): when the C16 T block is complete at both K, a row spec "yaw-aug C16 vanilla @40k (exp_40)" (glob `exp40_C16_T_S40000_K{1,8}_s4[2-6]`, top-level metrics) is added to `gen_model_comparison.py` and the table regenerated — on neuronic, per the standing table-regen trap — in the same results commit.

## 7. Validation ladder

1. Static: `py_compile`, `bash -n`, `json.load`, `git diff --check`.
2. Pytest: the new tests + the existing yaw/factory/train suites (`test_yaw_aug_training.py`, `test_yaw_aug_arm_config.py`, `test_cond_dispatch.py`, `test_train_*`, `test_depth_canon_*` factory parts) green on CPU.
3. Real-data readback (CPU): one real AR batch (micro 4) through the dataloader + `_apply_yaw_aug` with s = 32: offsets multiples of 32, depth `[3, 256, 512]` rolled by exactly `d`, pose norms and z preserved.
4. DRY_RUN of every launch path (local launcher all modes; sbatch gates for both rungs; submitter; `exp40_eval.py run` DRY_RUN with E0 + all 97 cells against a placeholder admission record in a harness dir).
5. GPU smoke (GPU 0, ≤ 15 min, this session): `MODE=SMOKE` 20 steps at micro 32 (§4.2).
6. Before FULL (not this session): DDP_SMOKE on both A6000s (A6000 path) or the neuronic SMOKE job (neuronic path); then FULL by Yixun's go.
7. Eval: E0 (3 cells) → C16 blocks T, C4, G45, OFF, R → FILL.

## 8. Acceptance criteria (FULL)

HEAD == EXPECT_SHA reported in the log; 2 ranks (or N on neuronic) on the registered GPU model; micro 32 (8), accum 1, SyncBN on; the exact C16 banner before the first step; flash backend positively asserted; ≥ 1 optimizer step with finite loss; 16 cadence checkpoints; 40k end audit PASS; `CHECKPOINTS.sha256`; NAS publish + admission. Eval: E0 EQUAL; every cell verified (§5.2).

## 9. Risks and disclosures

Single training seed; exp_17's launch commit and both reference backends inferred (E3); a neuronic run differs in GPU model and launcher form (D1 b/c); C16 vs YAW on 2×32 share draw keys but CUDA nondeterminism and data-loader worker timing make the trajectories non-identical; FILL cells for P1 135/225 duplicate the exp_36 chain's (same pin family, reported as a consistency check if both exist). GPU 1 belongs to the exp_30 evaluations until ≈ 04:00 Nov 8 — never used, signalled or waited on by this seat; exp_38/exp_39 worktrees are never touched.

## 10. Review budget

Plan: ≤ 3 Codex rounds (this is r0 → r1). Code: ≤ 3 Codex rounds, ALL code of the experiment (training change, tests, kit, eval driver, collector) handed over in round 1; fixes in rounds 2–3; anything left after round 3 is fixed, tested and logged as fixed-without-re-review.
