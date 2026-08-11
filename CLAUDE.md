# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Wait-time reporting (MANDATORY, every response)

Whenever any runs are in flight that Yixun must wait on — code reviews, training, evals, probes, any background execution — END the response with: (1) an estimated completion time for EVERY running item (wall-clock ETA, not just duration), and (2) **the earliest time Yixun needs to be present** (the next moment a human decision could be required; state "no presence needed until X" when everything is autonomous).

## Universal code review (MANDATORY)

EVERY piece of code written in this repo — by the Coder subagent OR by the main session (one-off scripts, page/visual generators, probe drivers, worklog tooling, anything executable) — goes through the Codex review → revision loop before its round closes. No "it's just a throwaway script" exemption: if it runs and its output informs a decision or artifact, it gets reviewed. Small scripts may be batched into one consolidated review round.

## Session handoff & compaction protocol (MANDATORY)

**Triggers:** Yixun says **"handoff"**, **"new session"**, or **"wrap up"** — OR the session is about to **/compact** (context summarization) in the same session — OR the active **model changes** (via `/model`, e.g. Fable 5 → Opus 4.8) or the current model **reaches its usage limit** and a different model takes over. On ANY trigger, BEFORE proceeding with other work, update ALL of:

1. **`CLAUDE.md`** (this file) — refresh any guidance this session made stale (new flags, moved paths, changed conventions).
2. **`worklog/worklog_yixun/master_experiment_tracker.md`** — the living per-experiment index: status, headline verdicts/numbers, key commits, what's in flight.
3. **`worklog/worklog_yixun/issue_report.md`** — open issues, known caveats/bugs, and decisions currently awaiting Yixun.
4. **`worklog/worklog_yixun/HANDOFF.md`** — the handoff doc: **log ALL working memory** — every in-flight run (exact command, PID/task id, log path, wall-clock ETA, what to do when it finishes), current experiment state and next steps, pending approvals, environment/GPU state, and any gotchas a fresh session would need to resume losslessly.

The handoff doc is the contract with the next session: assume the reader has NO memory beyond the repo + these four files.

**Automation:** `.claude/hooks/model_change_handoff.py` (wired in `.claude/settings.local.json` on `SessionStart`+`UserPromptSubmit`) auto-detects a model-*family* change (fable/opus/…), snapshots the four docs to `worklog/worklog_yixun/handoff_snapshots/` + logs `handoff_log.md`, and injects a reminder into the incoming model's context. It is the *detector/archiver only* — the live model still authors the four-doc refresh.

## Experiment SOP (MANDATORY)

All experiment and code-writing work in this repo follows **`worklog/experiment_SOP.md`** — read it before planning, coding, or running anything. Non-negotiable core:

- Read every standing directive in `worklog/worklog_yixun/announcement/` first. In particular: comparisons against FLAC always use the full existing dataset configs under `src/configs/dataset_configs/` (unseen eval = all 6337 items / 17 rooms of `data/AR/unseen_eval.json`); never create new or subsampled eval configurations.
- Per experiment: scaffold `worklog/worklog_yixun/exp_<NN>_<exp name>_claude/`, log the user's query (verbatim + summary + assumption + why), write `plan_<exp name>.md` (English plan + per-file planned code) and get it approved BEFORE implementing.
- Role split: the main session model plans and analyzes; an **Opus 5 max-effort** subagent writes the code (per Yixun 2026-07-25; supersedes Opus 4.8); **OpenAI Codex `gpt-5.6-sol` at xhigh reviews it** via `~/.local/bin/codex exec -s read-only … < /dev/null` (review saved as `<exp name>_codex_code_review.md`). If Codex is unavailable, the declared fallback reviewer is Claude Opus 5 at max effort — say so in the artifact by-line. `-s read-only` does **not** protect the environment (it once permitted a `pip install` into conda base), so every review prompt must explicitly forbid installing or modifying environments.
- Every run's terminal output is teed to a timestamped `.log` in the experiment folder; params, command, results, and a closing reliability analysis are recorded as separate files per SOP.
- Develop commit-by-commit from base `0bd5da0`: commits generally < 200 changed code lines, one-or-more commits per experiment, SHAs logged in `commits_<exp name>.md`. Archive (don't destroy) superseded code under `worklog/worklog_yixun/archive_*`.

## Project

Reference implementation for *Few-shot Acoustic Synthesis with Multimodal Flow Matching* (CVPR 2026). Two components live in the same repo:

- **FLAC** — one-shot room impulse response (RIR) generator. A conditional rectified-flow DiT operating in a VAE latent space over 1-channel RIR waveforms.
- **AGREE** — CLIP-style acoustic–geometry embedding, used both as a metric (FD / Recall) and as a pretrained backbone. Code in `AGREE/`.

Datasets supported: AcousticRooms (AR, primary training) and Hearing Anything Anywhere (HAA, finetune target). See `README.md` for download / preparation instructions.

**Shared machine, concurrent writers.** A second Claude session works this same repo and branch (`check-equivariance-necessity`) from another machine and pushes frequently — **always `git pull --rebase` before committing**, and never rewrite files another session owns. ⚠️ **Queued Slurm jobs bind to commits**: exp_11 training legs verify their `EXPECT_SHA` at start via the *content-scoped* gate in `fa_orbit_train.sbatch` (Codex-approved, `da7ee7f`) — record/worklog commits are safe while legs pend, but commits touching the training closure (`train.py`, `defaults.ini`, `src/`, `data/AR`, arm configs, the launcher + its helpers) still abort them fail-closed. Never revert that gate to HEAD-identity. On neuronic, sbatch has **no `SBATCH_EXCLUDE` env var** — node exclusion works only as an explicit `--exclude`/`EXCLUDE=` argument (screen submitter only; the training submitter has no exclusion path). ECC-flaky nodes (Aug 2026): neu301/303/305/306/317/319/322/332. On the A6000 box, Yixun also runs experiments from *sibling checkouts* (`~/codespace/exp-12-arms`, `exp-08-cylvit-pe-cnn`, `exp-09-cyl-dinov3-no-ssl`, `exp-10-cyl-distill`) whose jobs share those GPUs and use their own experiment numbering. **Before assuming any `train.py` process belongs to this worktree, check `readlink /proc/<pid>/cwd`; never kill or edit a run you did not launch.**

## Install & environment

```bash
conda create -n flac python=3.10 && conda activate flac
pip install .
```

Pinned versions live in `pyproject.toml` (torch 2.7.0, pytorch_lightning 2.1.0, transformers 4.57.0, etc.). Flash-Attention requires torch ≥ 2.5. VAE training needs an H100 (80GB); everything else fits on a 24GB GPU.

## Common commands

All entry points are at the repo root and consume **two JSON configs** (model + dataset) plus CLI flags. CLI defaults come from `defaults.ini` (parsed by `prefigure`), so any flag listed there can be omitted.

```bash
# Download pretrained weights (HF) into ./weights
bash download_weights.sh

# Train FLAC on AR (see README for full multi-GPU example; minimum below)
python train.py \
  --dataset-config src/configs/dataset_configs/AR/train/acousticroom_train.json \
  --val-dataset-config src/configs/dataset_configs/AR/eval/acousticroom_seeneval.json \
  --model-config src/configs/model_configs/FLAC/AR/FLAC_AR.json \
  --pretransform-ckpt-path weights/FLAC/VAE.safetensors \
  --save-dir ./outputs_FLAC --name FLAC --experiment-name FLAC_training

# Evaluate FLAC (standalone script; preferred for paper-style numbers)
python eval_FLAC.py \
  --model-config src/configs/model_configs/FLAC/AR/FLAC_AR.json \
  --dataset-config src/configs/dataset_configs/AR/eval/acousticroom_seeneval.json \
  --ckpt-path weights/FLAC/FLAC_EMA.ckpt \
  --cfg-scale 1.0 --steps 1 --eval-name eval_FLAC

# Evaluate via the PL test loop (alternate path; uses model_config['test_setup'])
python eval_pl.py --model-config ... --val-dataset-config ... --ckpt-path ...

# Evaluate the VAE alone (note: .ckpt, not .safetensors)
python eval_VAE.py --model-config ... --dataset-config ... --ckpt-path ...

# Baselines (KNN / LinearInterp / RdnAcross / RdnSame). Runs as a module.
python -m baselines.eval_baselines --dataset {AR,HAA} --baseline KNN ...

# Unwrap a training-wrapper checkpoint into a plain model checkpoint
python unwrap_model.py --model-config ... --ckpt-path ... --name out --use-safetensors
```

AGREE has its **own** CLI entry-point and **must be run from the `AGREE/` directory** so its relative paths resolve:

```bash
cd AGREE
python -m AGREE_train.main --dataset-type {AR,HAA} --model {dinoV3,xRIR,openclip} ...
```

Upstream ships no lint config or formatter — don't invent one. Tests: per `worklog/worklog_yixun/announcement/02_test_driven_development.md`, all NEW code in this fork is developed test-first (pytest, in `src/tests/` — location chosen by Yixun); upstream release code is not retroactively covered.

## Architecture

### Config-driven model construction

Every model is built by `src/models/factory.py::create_model_from_config(model_config)` and wrapped for training by `src/training/factory.py::create_training_wrapper_from_config`. The same JSON describes *both* what `nn.Module` is built and how it is trained, so when changing model behavior, expect to edit both the architecture block and the `training` block of the config — not the Python.

`model_type` switches between two top-level kinds:
- `autoencoder` → builds the **VAE pretransform** (oobleck encoder/decoder, VAE bottleneck, ratio 1024, latent dim 32). Trained by `AutoencoderTrainingWrapper`.
- `diffusion_cond` → builds **FLAC** itself: a conditioned DiT (`src/models/dit.py`) wrapped by `ConditionedDiffusionModelWrapper` (`src/models/diffusion.py`) with the VAE attached as `pretransform`. Trained by `DiffusionCondTrainingWrapper` (`src/training/diffusion.py`).

The DiT consumes:
- **global conditioning** (adaLN) — listed in `diffusion.global_cond_ids` (e.g. `source`, `source_vit`).
- **cross-attention conditioning** — listed in `diffusion.cross_attention_cond_ids` (e.g. `context_poses_vit`, `context_poses`, `context_audio`).

Conditioners are registered in `model.conditioning.configs[].id/type`; the most important types are `dist_embedder` (Fourier features for 3D positions), `ViTCoordinates` (DINOv3 / xRIR / OpenCLIP ViT over depth panoramas), and `rir` (mel-style STFT encoder over reference RIR waveforms). The `id` field is the contract with `cross_attention_cond_ids` / `global_cond_ids` *and* with the metadata keys produced by the dataset's `custom_metadata_module`.

### `flow_source` is load-bearing

`training.flow_source` controls the starting point of the rectified-flow trajectory and is read with bracket access in `src/training/factory.py:66` — a missing key crashes with `KeyError` by design. Two modes:

- `"gaussian"` — standard rectified flow from noise. Used by `FLAC_AR.json`, `FLAC_AR_S.json`, `FLAC_AR_AllCA.json`, `FLAC_AR_InContext.json`, `FLAC_AR_VAECtxt.json`, `FLAC_AR_noGeom.json`, `FLAC_HAA_finetune.json`.
- `"nearest_ref"` — flow starts from the VAE encoding of the nearest reference RIR (`_pick_nearest_reference` in `src/training/diffusion.py`). Used by `FLAC_AR_nearestRef.json`. Training drops the picked reference from cross-attn context; inference keeps it (so K=1 deployment works). `eval_FLAC.py` mirrors the same dispatch.

When adding a third mode, update **both** dispatch sites (`training_step`/`validation_step`/`test_step` in `src/training/diffusion.py` and the inference path in `eval_FLAC.py`) — there is no central whitelist by design.

### Dataset & metadata pipeline

`src/data/dataset.py::create_dataloader_from_config` reads `dataset_config["datasets"]`, where each entry points at a `custom_metadata_module` Python file (e.g. `src/configs/dataset_configs/custom_metadata/AR_md.py`). That module exposes `get_custom_metadata(info, audio)` returning a per-sample dict consumed by both the conditioners (matched by `id`) and the metric callback (`scene`, `depth`, `source`).

`dataset_config["modalities"]` toggles which fields the metadata module produces (`acoustic_context.max_context` controls K, the number of reference RIRs per sample). To change K at eval, switch to an `acousticroom_*eval_<K>.json` config rather than editing code.

The AR depth panorama is taken at the **listener** position; HAA reverses this and renders at the **source** position. There is a deliberate sign convention in `src/configs/dataset_configs/custom_metadata/HAA_md.py:70` — flipping it can improve HAA metrics (see README "Performance Tip" and the paper supplementary).

### Metrics

`src/metrics/metric_callback.py::AcousticMetricsCallback` is constructed from `training.metrics` in the model config and produces T60, C50, EDT, l1/multires-l1, FD (AGREE-based), and retrieval recall. FD/retrieval require an AGREE checkpoint at `metrics.AGREE_ckpt`. Use `AGREE_full{AR,HAA}.pt` *only* for evaluation (data leakage); use `AGREE_AR.pt` as a downstream backbone.

Paper headline numbers are computed by **averaging per-scene results**, not over all samples. The evaluation script prints both; use the per-scene mean for comparisons.

**⚠️ Eval-protocol flags are part of the experiment, not defaults.** `eval_FLAC.py` takes `--cond-method {vanilla,fa_invariant}` (plus `--frame-avg-angles`, `--rotate-deg` for C₄ sweeps and the 45° negative control, `--cond-autocast {default,bf16,off}`). **The flag must match how the checkpoint was trained.** A mismatch produces plausible-looking but catastrophically wrong numbers in both directions — the fa-trained B-F@40k reads `8.202/0.978/38.79/R5.39` under fa eval and `10.652/2.082/80.86/R0.68` under vanilla eval. Evaluating fa checkpoints with the default (vanilla) conditioning caused exp_09's protocol error and one retracted exp_07 conclusion. **Put the eval-protocol flags in every launch/screen manifest; never rely on the default.**

Cross-experiment results live in `worklog/worklog_yixun/model_comparison.md`, regenerated **only** by `worklog/worklog_yixun/gen_model_comparison.py` (rows are glob specs aggregated from raw per-seed metric JSONs; single-seed screens are structurally excluded). Per announcement 04, regenerate + commit + push on every model-results update.

### Checkpoints

Trained checkpoints embed the PL training wrapper, which makes them large and key-mangled. `eval_FLAC.py` strips the `diffusion.` / `diffusion_ema.ema_model.` prefixes inline. For external use (and as input to FLAC training), run `unwrap_model.py` to export a clean model. ⚠️ `unwrap_model.py` currently imports from `stable_audio_tools` (not in `pyproject.toml`); the in-repo equivalents live under `src/`, so the script needs adapting before it will run as-is.

When loading a VAE as a pretransform for FLAC training, pass the unwrapped `.safetensors` via `--pretransform-ckpt-path`; do not pass the wrapped `.ckpt`.

### Checkpoint surgery on warm resume (`src/tools/`, learned exp_09 + exp_13)

⚠️ **A PL checkpoint's LR-scheduler state overrides the model config on resume.** `InverseLR` (`src/training/utils.py`) keeps `inv_gamma`/`power`/`warmup`/`final_lr` as instance attributes, so `LRScheduler.state_dict()` serializes them and PL's `load_state_dict` (`self.__dict__.update`) writes them back over whatever the incoming config built. **Changing a schedule in the JSON therefore has no effect on a warm resume** — measured: an intended 1.28e-5 decay tail silently stayed at 4.77e-5 for the whole run. To change a schedule mid-run you must rewrite the checkpoint:

- `src/tools/retune_lr_state.py` — writes a **copy** with new scheduler hyperparameters *and* the matching `param_groups[0]["lr"]` (so the first resumed step is already on the new curve); re-derives the target lr from the checkpoint's own `base_lrs`/`warmup`/`last_epoch` and refuses to write if it disagrees.
- `src/tools/strip_optimizer_state.py` — fresh-Adam resume. **Keep the optimizer entry and clear only its `state`**: an *absent* `optimizer_states` key makes PL raise `KeyError` on restore, and an *empty list* silently runs the first update at the step-0 warmup lr (5e-7, ~96× under schedule) because the freshly-constructed scheduler writes that value into `param_groups`.

Both tools are copy-only (never touch their input) and TDD-covered in `src/tests/`. Resumes are never bit-exact — PL restores no RNG or dataloader position; disclose it.

### HAA finetuning quirk

Training length is set by `--max-steps` (ini key `max_steps` in `defaults.ini`, default 1,000,000 — same as the old hardcode, so existing recipes are unaffected). HAA finetuning expects 1000 steps: pass `--max-steps 1000` when running the HAA recipe in the README — never edit code for it. (Flag landed in exp_07 TDD round 1, replacing the hardcoded `train.py:161`; enforced by `src/tests/test_train_max_steps.py`.)

### Multi-GPU BatchNorm (`--sync-batchnorm`)

`--sync-batchnorm` (ini key `sync_batchnorm` in `defaults.ini`, default `false`) forwards to PL `Trainer(sync_batchnorm=True)`. Default off leaves the Trainer kwargs **byte-identical** to the pre-flag dict, so existing recipes are unaffected. It is multi-GPU-only: enabling it with `num_gpus < 2` is a **fail-closed `ValueError`** (Yixun mandate), not a silent no-op, and `val_args` may not smuggle the key past that guard. (Landed exp_07, commit `f362673`, TDD, 40 tests.)

⚠️ **Gradient accumulation never feeds BN statistics** — BN sees only the per-device micro-batch, so accumulation cannot reconstruct a target BN batch. To get BN over 64 you need 32/GPU × 2 GPUs × accum 1; `8×8` does **not** give BN=64 no matter the accumulation. Pin the micro-batch literally when a BN batch size is part of the recipe.

### Diffusion objectives

`model.diffusion.diffusion_objective` selects the sampling path used at inference time in `eval_FLAC.py`:
- `"rectified_flow"` → `sample_discrete_euler`
- `"v"` → `sample`
- `"rf_denoiser"` → `sample_flow_pingpong` with logSNR-spaced sigmas

All FLAC configs in the repo currently use `"rectified_flow"`. Inference paths live in `src/inference/sampling.py`.

## Repo layout (only the non-obvious bits)

- `src/configs/model_configs/FLAC/AR/*.json` — paper variants; see README table for the conditioning topology of each.
- `src/configs/dataset_configs/AR/eval/acousticroom_*eval_{1,4,8}.json` — reduced-context variants; the suffix is K.
- `src/configs/dataset_configs/custom_metadata/{AR,HAA}_md.py` — *not* configs; these are Python hooks loaded dynamically by the dataloader.
- `AGREE/` — self-contained subproject. Its `AGREE_train.main` is independent of `src/training/`. Always invoke from inside `AGREE/`.
- `baselines/eval_baselines.py` — unified KNN/RdnAcross/RdnSame/LinearInterp; reads AGREE for the FD metric.
- `data/{AR,HAA}/*.json` — train/eval split files (not raw data). HAA depth maps ship in `data/HAA/depth/`.
- `src/tools/` — checkpoint-surgery CLIs (`strip_optimizer_state.py`, `retune_lr_state.py`); copy-only, TDD-covered. See "Checkpoint surgery" above.
- `worklog/worklog_yixun/` — experiment records plus three living artifacts: `model_comparison.md` (+ `gen_model_comparison.py`), `A6000_METRICS_SHA256SUMS.txt` (raw metric JSONs are force-added past the `outputs_FLAC/` ignore so the table regenerates from git), and the trajectory figures `trajectories_all_arms*.{png,pdf,html}`.
- `weights/`, `AcousticRooms/`, `HAA/`, `outputs_FLAC/`, `wandb/` — gitignored runtime locations; create or symlink as needed.

## Sibling repo: `cylindrical-dinov3` (new 2026-07-16)

`~/codespace/cylindrical-dinov3` (GitHub `Yixun-Hu/cylindrical-dinov3`, **private**) — a standalone package building a **cylindrical azimuth-equivariant DINOv3 ViT** intended to replace FLAC's geometry backbone. Deliberately a *sibling*, not a FLAC subfolder: it is `pip install -e`'d into the same env so both the continued-SSL training scripts and FLAC import one copy (`from cylindrical_dinov3 import ...`), and weights move via `save_pretrained()`/`from_pretrained()`. Do **not** fork Transformers or edit the installed copy in site-packages.

- It carries its **own** copy of `worklog/experiment_SOP.md` (same portable SOP) and its own `worklog/worklog_yixun/` namespace — experiment bookkeeping for that work lives there, not here.
- `ai_conversations/claude_context_dinov3_cylindrical_conversation_from_codex.md` — the Codex design transcript that specifies the port (RoPE harmonics, prefix/CLS handling, XYZ gauge alignment, SSL plan). Read before touching that repo.
- `vanilla_dinov3/` — read-only vendored copy of `transformers==4.57.0` `dinov3_vit`, hash-verified against the pip RECORD; the diff baseline. **Port against it, not against Transformers `main`** — v4.57.0 stores blocks as `self.layer`, `main` renamed it to `self.model`.
- FLAC-side integration when it lands: a loader branch in `src/models/conditioners.py` keyed on `ViT.implementation`, still inside the existing `if vit_model is None:` block (FLAC shares one ViT between `source_vit` and `context_poses_vit`). ⚠️ Load order — a full `FLAC_EMA.ckpt` load will **overwrite** a geometry backbone loaded earlier; the SSL backbone must be applied *after* the FLAC state dict.
