# plan_della_vanilla_repro — exp_16: reproduce vanilla FLAC on della (A100 80GB; H100 unavailable — see Rev 3)

**Status:** REV 3 — Rev 2 was approved by Yixun 2026-08-11 and then amended twice: the hardware amendment (§4d) and this Rev 3 correction block from the full integrative Codex review (`della_vanilla_repro_codex_code_full_review.md`).

## Rev 3 corrections (full-review findings; no recipe change)

1. **Hardware consistency (review N2):** every "H100" in the launch-acceptance and title text is superseded by the §4d amendment — Phase 2 runs on **1× A100-SXM4-80GB**. The §7 OOM fallback is ViT gradient checkpointing only; the "H200" fallback is struck (H200s sit in `ailab*`, inaccessible to this account). Smoke correction: the 300-step smoke ran **six** validation passes (val-every 50), not one; the ETA arithmetic already used the measured per-pass cost (86 s), so the projection stands. The first 2,500-step checkpoint of the full leg is a runtime monitor (arrival confirms save-path + cadence), not an acceptance criterion.
2. **Phase-1 gate bookkeeping (review B1):** the gate is evaluated on exact JSON paths of the metrics record — `metrics.T60`, `metrics.C50`, `metrics.EDT`, `metrics.FD`, `metrics["Invalid T60"]`, and the **`RIR_to_GT_RIR_R@{1,5,10}`** family (never `RIR_to_geom_R@k`). The anchor comparison is the **top-level all-sample aggregate** (exp_01 established this is the aggregation behind the paper's AR table; `FLAC_AR.json` does not enable per-scene eval) — this is a deliberate, disclosed exception to the per-scene-means note in CLAUDE.md, which exp_01 showed applies to the HAA path. The gate verdict is written to a committed **`PHASE1_PASS.md`** (values, deltas vs thresholds, loader/count/load evidence, verdict line) and `della_submit.sh` refuses a non-smoke `train` submission unless that file exists at HEAD — the wrapper now enforces the plan's "Phase 1 gates all training".
3. **Interruption runbook (review B2):** measured cluster policy (2026-08-11, `scontrol show config`): **`JobRequeue = 0`, `PreemptMode = OFF`** — della never auto-requeues and gpu-medium is not preemptible; a failed/timed-out leg is terminal until an operator acts. Both sbatch files additionally pin `#SBATCH --no-requeue`. Recovery procedure after a mid-run death: (1) confirm the job is terminal (`sacct -j <id>`) and read its log tail; (2) classify infra vs real bug per SOP — only infra failures proceed to resume; (3) validate the newest checkpoint under the production glob (readable, `epoch=*-step=*.ckpt`, step < 67500); (4) if valid: `della_submit.sh train --resume --time <remaining×1.5>` (closure unchanged + pushed; RESUME=1 is the explicit opt-in; disclosure: PL resume is not bit-exact — no RNG/dataloader state restore); (5) if no checkpoint: resubmit without `--resume`; (6) record the new job id + W&B run in the command manifest and worklog.
4. **Records (review N4):** `commits_della_vanilla_repro.md` is kept complete through HEAD from now on; `_params_set_up.md`'s training section is filled at Phase-2 submission with the wrapper-recorded command.
**Branch:** `della-flac-chequity` (cut from `check-equivariance-necessity` @ `e603947`).
**Planner:** Claude Fable 5 (main session, xhigh). **Coder:** Claude Opus 5 subagent. **Reviewer:** Claude Opus 5 (declared fallback — no Codex/npm/node on della; per CLAUDE.md fallback rule).

**Rev 2 changes:** rung-0 env repair + import smoke (B1); AGREE hub-fetch covered by populating the real HF cache, resolver demoted to defense-in-depth for the FLAC conditioner only (B2/B3); Phase-3 bands recalibrated on the three prior full-budget vanilla runs + P1@67.5k made co-primary reference + contribution restated (B4); budget arithmetic corrected (291,210 items, 4,550 steps/epoch, "≥67,500, snapshot adopted", epoch-14 batch-size witness banked) (B5); `/AcousticRooms` gitignored with `/models` (B6); Phase-1 gate fully pre-registered with `max(3σ, 1%)` thresholds, Run A demoted to descriptive (B7); resolver root-resolution + resolved-path logging + split call-site tests (N1–N3); interpreter-provenance sbatch gate (N4); smoke re-specified ≥300 steps + priced validation + sbatch resources (N5); OOM contingency pre-registered (N6); default QOS gpu-medium (N7); metrics-into-repo + SHA256SUMS pre-registered (N8); endpoint-draw screen at 62.5k/65k (N9); Run B at seeds 42+43 (N10); `flow_source` removed from parity list (N11); flash-attn + val-RNG/val-set + SLURM_PROCID added to irreproducibility list (N12/N13); `--record-stream --expected-stream-count` adopted for calibration cells (N14).

---

## 1. Goal & headline design decision

Reproduce the released vanilla FLAC (`FLAC_EMA.ckpt`) by training from scratch on one della H100 with the paper's reported recipe, then evaluate on the full published splits and quantify how close a faithful re-run lands.

**Recovered budget (corrected per review B5):** the paper omits the training length. The released wrapped checkpoint `FLAC.ckpt` embeds `global_step = 67500`, `epoch = 14`, optimizer step 67500, and the exact `InverseLR` state of `FLAC_AR.json`'s scheduler (`_last_lr = 4.839339184958273e-05`, reproduced analytically by the reviewer to the last digit). The release ran **≥ 67,500 steps**; the released artifact is the `every_n_train_steps=2500` periodic snapshot at 67,500 (27 × 2500), which we **adopt as the pre-registered budget**. Split arithmetic: `data/AR/train.json` = 291,210 items / 243 rooms ⇒ 4,550 steps/epoch at global batch 64 with `drop_last=True`; 67,500 = 14 × 4,550 + 3,800, matching the checkpoint's own `batch_progress.current.processed = 3800` mid-epoch-14 state. That arithmetic is also an **independent witness of global batch 64** (at 2×32 or 2×64 the epoch counter would read differently), on top of the release run-directory name `…_dinoViTS16FT_BS64`.

**Scope statement (review B4):** this repo already holds three full-budget from-scratch vanilla runs (P1@67.5k, P1-rerun@87.5k, B-V 8×8@67.5k in `model_comparison.md`). exp_16's contribution is therefore (a) validating the **della pipeline** end-to-end, (b) the **literal paper recipe** — batch 64 on ONE GPU, BN batch 64/512 as published — which no prior run used exactly (P1's provenance differs), and (c) a same-recipe anchor for future della experiments. It is **pipeline validation with a reproduction readout**, not the first reproduction.

**Known-irreproducible factors (disclosed up front):** release RNG seed; dataloader worker scheduling; GPU arch/kernel nondeterminism; **no flash-attn on della** (release used 2.7.4.post1; della runs the math fallback — note exp_01's anchor numbers also ran without flash-attn, so Phase 1 is apples-to-apples) (N12); **validation cadence perturbs the training RNG stream** (`validation_step` draws `torch.randn_like` from the global RNG) and the release validated on a different/smaller set (82 val batches in its loop state vs our seeneval's 98) (N13); `train.py` adds `SLURM_PROCID` to the seed — the effective seed is recorded and the leg never launches under `srun -n>1` (N13). Reproduction claims are statistical, never bit-exact.

## 2. Facts the plan builds on (verified 2026-08-11; corrections per review)

- Config parity: `FLAC.ckpt['model_config']` ≡ repo `FLAC_AR.json` in every architecture/optimizer field. Differences: ViT path (`./Models/dinov3-…` local dir in release vs `facebook/dinov3-…` hub id in repo — the release itself trained from a local directory), release-only `training.demo` block, metric-callback flags, `structured_noise: false`. (`flow_source` claim removed — not in `FLAC_AR.json`, not read by this branch's factory; the CLAUDE.md note is stale — N11.)
- Recipe fields in `FLAC_AR.json` + `defaults.ini`: DiT depth 12 / heads 8 / `embed_dim` 256; AdamW lr 5e-5 β(0.9,0.999) wd 1e-3; InverseLR(inv_gamma 1e6, power 0.5, warmup 0.99) stepped per optimizer step — effectively constant-LR (decay removes 3.2% over the run; warmup done by step ~458); batch 64; `use_ema: true` (EMA β 0.9999, power 3/4, update_every 1); `precision bf16-mixed`; `diffusion_objective: rectified_flow`; `timestep_sampler: log_snr`; cfg_dropout 0.1. Silently inherited defaults recorded in `_params_set_up.md`: `gradient_clip_val 0.0` (release value unknowable), `num_sanity_val_steps 0`, `log_every_n_steps 100`, `save_top_k -1`, `accum_batches 1`, `checkpoint_every 2500` (now release-confirmed).
- **DINOv3 is locally available ONCE** (correction, B3): `models/dinov3-vits16-pretrain-lvd1689m/` (repo symlink → scratch). The HF-cache entry is an empty 94 KB stub (refs only, no blobs) — populating it is rung 0 work, not an existing mitigation.
- **The env is currently broken** (B1): `setuptools 83` removed `pkg_resources` → `import k_diffusion → clip` dies; pytest absent. Rung 0 repairs both.
- **Two hub-id call sites** (B2): FLAC conditioner (`src/models/conditioners.py:455/458`) AND AGREE metric model (`AGREE/AGREE/hf_model.py:30` via `metric_callback.py:432-437`). Both must load offline for any FD/retrieval eval.
- Slurm: account `blanchette`; QOS `gpu-short` 1d / `gpu-medium` 3d / `gpu-long` 6d; H100 (`gpu:h100:4|8`) and H200 nodes in partition `gpu`; login node GPU-free; compute nodes offline.
- Eval references: exp_01 per-seed unseen JSONs (A6000, `0bd5da0`) + paper Table 1; exp_01 5-seed noise floors (K8: T60 ±0.012, C50 ±0.003, EDT ±0.07, R@1 ±0.10). Prior full-budget vanilla runs for Phase-3 calibration (B4 table).
- `eval_FLAC.py` writes metrics + predictions beside `--ckpt-path`; wrapped checkpoints load cleanly (whitelist `("diffusion_ema.", "losses.")`, eval_FLAC.py:734) — no unwrap step needed. Calibration evals run **without** `--store_predictions` (README deviation, disclosed: ~GB of decoded audio would land in `$HOME`; metrics unaffected).

## 3. Rung 0 — environment repair & offline-load preflight (login node, before any code)

1. `pip install "setuptools<81" pytest` into the flac env (login node has network). Record exact versions + full `pip freeze` snapshot in `della_vanilla_repro_params_set_up.md` (B1).
2. Populate the real HF cache: `HF_HOME=/scratch/gpfs/BLANCHETTE/yh4742/hf_cache hf download facebook/dinov3-vits16-pretrain-lvd1689m` (B2 preferred fix — covers the AGREE call site with zero code changes; requires the gated-repo token already stored in `hf_cache/token`).
3. Import smoke (CPU, ~10 s): `python -c "import train, eval_FLAC"` from the repo root.
4. Offline-load proof (CPU, login node): `HF_HUB_OFFLINE=1 HF_HOME=… python -c` loading BOTH call sites — the conditioner ViT (via resolver → `models/…`) and the AGREE `hf_model` path (via cache) — must succeed with network disabled. This rung, not a Slurm job, is what discovers residual fetch paths.

## 4. Della accommodations (code changes, all on `della-flac-chequity`)

### 4a. `src/models/conditioners.py` — local ViT snapshot resolution (TDD; defense-in-depth per B2/B3)

With the cache populated, the resolver is redundancy, not the only path — but it is kept deliberately: it honors Yixun's `models/` setup, survives a cache wipe on shared scratch, and mirrors the release's own local-dir loading. Design revised per N1/N2:

```python
def resolve_vit_model_path(name_or_path, local_root=None):
    """Prefer a local snapshot under <local_root>/<basename> when present.

    Root resolution (N1): explicit arg → $FLAC_LOCAL_MODEL_ROOT → <repo_root>/models
    derived from this file's location (NOT the CWD). Returns (resolved_path,
    source_tag) so the call site can log which rule fired. An input that is
    already an existing directory, or has no local snapshot, passes through
    unchanged (hub cache / offline behavior then applies).
    """
```

Call-site changes (N2): resolve BEFORE the existing print; log `original → resolved (source_tag)` plus the snapshot's `model.safetensors` size and sha256 prefix — the run log's only ViT-weights provenance. The `'convnext' in model_name_or_path` check stays on the **original** string. JSON configs stay byte-unchanged (hub id embeds into checkpoints via `ModelConfigEmbedderCallback` — portability preserved). AGREE subproject is **not** modified (cache covers it; scope call, stated per B2).

**Tests (`src/tests/test_vit_local_resolution.py`, red→green; revised per N3):**
1. `test_existing_dir_returned_unchanged`
2. `test_hub_id_resolves_to_local_snapshot` (explicit `local_root`)
3. `test_hub_id_without_snapshot_unchanged`
4. `test_env_var_root_wins_over_repo_root`
5. `test_repo_root_derived_from_file_not_cwd` (chdir to tmp; resolver still finds `<repo>/models`)
6. `test_callsite_from_pretrained_uses_resolved_path` (from_scratch **false** branch → `AutoModel.from_pretrained` recorded arg)
7. `test_callsite_from_config_uses_resolved_path` (from_scratch **true** branch → `AutoConfig.from_pretrained` recorded arg)
8. `test_callsite_logs_resolved_path` (captured stdout names both original and resolved)

### 4b. `.gitignore` — TWO lines: `/models` and `/AcousticRooms` (B6: the symlink is not matched by the existing `AcousticRooms/` dir-only rule). `git pull --rebase` immediately before this commit specifically (shared file).

### 4c. Storage moves (runtime, no commits; recorded in worklog)

Released checkpoints → `/scratch/gpfs/BLANCHETTE/yh4742/FLAC/weights/`, file-level symlinks back into `weights/FLAC/` & `weights/AGREE/` (repo paths in configs/commands unchanged; metric JSONs stay real files). Training outputs → `--save-dir /scratch/…/FLAC/checkpoints/exp16_vanilla_repro`. W&B: `WANDB_MODE=offline`, `WANDB_DIR=/scratch/…/FLAC/wandb`.

### 4d. exp_16 Slurm kit (in the exp folder; deliberate scope call per review G: gated by `bash -n` + `DRYRUN=1` + the pytest-covered resolver, not a bespoke pytest runner — single-writer checkout, no lease machinery)

Common gates in both sbatch scripts: `EXPECT_SHA` == `git rev-parse HEAD` and clean tree; **interpreter-provenance gate** (N4): `python -c "import src.models.conditioners as c; assert c.__file__.startswith('$REPO_ROOT')"`; env exports (`HF_HOME`, `HF_HUB_OFFLINE=1`, `WANDB_MODE=offline`, `WANDB_DIR`); eval/train argv echoed; tee'd timestamped log into the exp folder; `DRYRUN=1` runs every gate and exits.

- `della_eval.sbatch` — 1 GPU, `--cpus-per-task=6 --mem=32G --time=04:00:00` (eval loaders: `--num-workers 4` default untouched per N10), no `--allow-partial-load`.
- `della_train.sbatch` — `--gres=gpu:a100:1 --constraint=gpu80 --cpus-per-task=10 --mem=64G` (persistent_workers ⇒ ~16 resident workers, N5), `--time` from the smoke-measured ETA ×1.5, QOS `gpu-medium` default (N7); resume-aware (`--ckpt-path` latest if present; PL resume non-bit-exact — disclosed); refuses `srun -n>1` (N13). **NO explicit `--partition`** — della rejects it for GPU jobs and auto-routes from GRES.
  **HARDWARE AMENDMENT (Yixun-approved 2026-08-11):** della H100s live only in `pli*`/`cryoem`/`ailab` partitions, none reachable by the `blanchette` account (probes: "Invalid qos specification"). Phase 2 runs on an **A100 80GB** (`gpu80` constraint; the 40GB A100s are excluded). Recipe deviation = GPU architecture only (same 80GB memory, BF16 supported, math-attention on both — the release's flash-attn absence already made kernels non-identical). Disclosed in every artifact; ~2× slower than H100 ⇒ ETA re-estimated at smoke.
- `della_submit.sh` — verifies committed+pushed SHA; records job id + argv into `della_vanilla_repro_command.md` at submit time.

## 5. Phase plan

### Phase 1 — eval calibration on the released EMA (gates all training compute)

Runs (H100 or A100 — arch recorded; disclosed in results):

- **Run B (gating, ×2 seeds per N10):** full unseen split, K=8, seeds **42 and 43**:
  `python eval_FLAC.py --model-config src/configs/model_configs/FLAC/AR/FLAC_AR.json --dataset-config src/configs/dataset_configs/AR/eval/acousticroom_unseeneval.json --ckpt-path weights/FLAC/FLAC_EMA.ckpt --cfg-scale 1.0 --steps 1 --eval-name exp16_calib_unseen_K8_seed<S> --seed <S> --cond-method vanilla --cond-autocast default --record-stream --expected-stream-count 6337` (N14; `--rotate-deg 0.0` default, written in the params file per announcement 05). Eval batch-size/workers stay at defaults (64/4) to match exp_01's protocol.
- **Run A (descriptive only, per B7):** the README lines 40-49 seen-split command with the same explicit protocol flags and `--expected-stream-count 6217`, no `--store_predictions` (deviation disclosed in §2). Compared narratively against the paper's seen-set supplementary tables; **it does not gate**.

**Pre-registered gate (B7) — Run B seed 42 vs exp_01's stored seed-42 JSON (8.6238 / 0.9687 / 37.0786 / FD 0.3053 / R@1 7.1012 / R@5 19.394 / R@10 27.0948), thresholds = `max(3σ_exp01, 1% relative)` as absolutes:**

| metric | threshold (abs) |
|---|---|
| T60 | ±0.086 |
| C50 | ±0.0097 |
| EDT | ±0.371 |
| R@1 | ±0.30 |
| R@5 | ±0.19 |
| R@10 | ±0.27 |
| FD | ±0.0031 |
| Invalid T60 | == 0.0 |

Plus: loader prints "6337 files in 17 subfolders"; stream count check passes; clean checkpoint load. Seed 43 is diagnostic (constant machine offset vs seed-level draw). Any gate failure ⇒ STOP, diagnose before Phase 2.

### Phase 2 — training reproduction (67,500 steps)

```
python train.py \
  --dataset-config src/configs/dataset_configs/AR/train/acousticroom_train.json \
  --val-dataset-config src/configs/dataset_configs/AR/eval/acousticroom_seeneval.json \
  --model-config src/configs/model_configs/FLAC/AR/FLAC_AR.json \
  --pretransform-ckpt-path weights/FLAC/VAE.safetensors \
  --save-dir /scratch/gpfs/BLANCHETTE/yh4742/FLAC/checkpoints/exp16_vanilla_repro \
  --name FLAC_vanilla_repro --experiment-name exp16_della_vanilla_repro \
  --batch-size 64 --num-gpus 1 --accum-batches 1 --num-workers 8 \
  --checkpoint-every 2500 --val-every 2500 --max-steps 67500 --seed 42
```

- Batch 64 on ONE GPU (paper-literal; BN batches 64 DiT-side / 512 RIR-encoder-side). Validation cannot contaminate BN/EMA (reviewer-verified mechanism: EMA in `on_before_zero_grad` only; PL `eval()` during val; pretransform re-`train()` no-op since `enable_grad=False`) — it only perturbs the RNG stream, which is disclosed (N13).
- **Ladder:** rung 0 (§3) → static checks → `python -m pytest src/tests -q` → `DRYRUN=1` both sbatch scripts → **smoke** (gputest: `--max-steps 300 --val-every 50`, checkpointing effectively off, rate from steps 20–300, one priced validation pass — N5) → ETA = 67500/rate + 27×val_cost, ×1.5 margin → `--time`, QOS `gpu-medium` (escalate to `gpu-long` only if ETA×1.5 > 3 d) → full leg.
- **OOM contingency (pre-registered, N6):** first ViT `gradient_checkpointing` (numerically identical), then an H200 node. **Never gradient accumulation** (BN batch is part of the recipe).
- Launch acceptance (worklog, before submit): pushed SHA verified; 1× H100; global batch 64; effective seed recorded (incl. SLURM_PROCID term); ≥1 optimizer step, no OOM/NaN; checkpoint at 2500 lands on scratch; offline W&B run dir created.

### Phase 3 — evaluate the reproduction + report

- Headline: final `epoch=14-step=67500.ckpt` (EMA weights via eval_FLAC's inline strip), full unseen split, K ∈ {1, 8}, seeds 42-46, `--cond-method vanilla --cond-autocast default`; seen split K8 seed 42 descriptive.
- **Endpoint-draw control (N9):** single-seed (42) unseen K8 screens at 62.5k and 65k (free checkpoints) — descriptive only, separates "della reproduced badly" from "endpoint is a draw".
- **References (B4):** judged against the release (exp_01 5-seed) AND reported side-by-side with `P1 vanilla @67.5k` (closest same-budget prior).
- **Pre-registered bands (review F), both K, vs release:** C50 ±3%, T60 ±8%, EDT ±8%, FD ±5%, R@1/R@5/R@10 ±20%. **Structural criteria:** K=8 beats K=1 on T60/C50/EDT; every metric ≥ the worst prior full-budget from-scratch run. PASS = all bands + both structural; PARTIAL = ≤2 band misses with structure intact; FAIL otherwise. Verdict language: "consistent with the release under an unknown seed" — n=1 establishes pipeline validity, not a reproduction hypothesis test.
- **Table plumbing (N8):** copy per-seed metric JSONs from scratch into the repo (`outputs_FLAC/exp16_vanilla_repro/`), `git add -f`, extend `DELLA_METRICS_SHA256SUMS.txt` (new, analogue of the A6000 file), add the `gen_model_comparison.py` row spec, regenerate, commit + push (announcement 04).
- `_results.md`, `_analysis.md`, `della_vanilla_repro_01_results.html` (+assets) per SOP.

## 6. Commit plan (each < 200 changed lines; SHAs → `commits_della_vanilla_repro.md`)

1. ✅ exp_16 scaffold (query/worklog/plan) + CLAUDE.md /init refresh — `0a0cd23`.
2. Review round 1 artifacts: plan review + this Rev 2 + worklog entry.
3. `.gitignore`: `/models` + `/AcousticRooms` (rebase immediately before).
4. TDD red→green: `test_vit_local_resolution.py` + `resolve_vit_model_path` + call-site wiring (may split test/impl).
5. `della_eval.sbatch` + `della_submit.sh`.
6. `della_train.sbatch`.
7+. Params/command/results/analysis/reviews as they land. Rung-0 env changes are not commits; recorded in `_params_set_up.md` + worklog.

Reviews: one fallback-reviewer round per Coder round; `full` integrative review before the Phase-2 launch; every review file carries the Opus 5 fallback by-line.

## 7. Risks & mitigations

- **Throughput unknown** → re-specified smoke (N5) prices both training and validation before QOS commit; resume-aware sbatch (non-bit-exact resume disclosed if used).
- **Offline compute nodes** → rung 0 proves both hub-id call sites load with `HF_HUB_OFFLINE=1` on the login node before any job queues.
- **Cache wipe on shared scratch** → resolver keeps the conditioner loadable from `models/`; AGREE would need a re-download (accepted residual risk, stated).
- **Two-writer repo** → all work on `della-flac-chequity`; `git pull --rebase` before every commit (especially the `.gitignore` one); exp_11/14/15 kits and gates untouched.
- **Release seed unknown; three prior from-scratch runs miss release R@1 by 9.5–13.5%** → bands calibrated to that envelope (B4/F); P1@67.5k co-reference keeps the verdict honest.

## Rev 4 — chunked chain execution for Phase 2 (Yixun directive 2026-08-13; replaces the two-leg race)

**Motivation:** account fairshare exhaustion makes long jobs queue for days; short jobs backfill. Yixun directive: split training into a self-resuming chain with a watchdog. Approved parameters: **CHUNK = 7500 steps, 9 legs** (last leg 7500 → 67,500), all gpu-short (`--time=04:30:00`, QOS weight 2000 vs medium's 800), racers 12313040/12313041 cancelled at chain submission.

**Verified foundations (worklog 2026-08-13):** `trainer.fit(ckpt_path=…)` restores model/optimizer/InverseLR/EMA/global_step (train.py:230); checkpoints land at a leg-independent path each 2500 steps, so 7500-step boundaries are native checkpoint boundaries; PL 2.1 mid-epoch resume restarts the dataloader (training_epoch_loop.py:155 warning) and **same-seed legs draw byte-identical permutations** (empirical probe) — so each leg must reseed.

**Design (new files only — the queued racers' closure gate does not cover new files, so they stay valid until deliberately cancelled):**

- `della_chain.sbatch` — one chain leg. Gates as della_train.sbatch (closure content-scoped over the 13 core paths + the two chain files, interpreter provenance, 1×A100-80GB, ntasks 1, no-requeue, offline env) plus a **mutual-exclusion gate** (any RUNNING exp16-train*/exp16-chain job other than self ⇒ GATEFAIL). Leg logic: S = step of newest valid ckpt (0 if none); S ≥ CHAIN_TOTAL ⇒ CHAINDONE exit 0 + scancel remaining manifest ids; TARGET = min(S+CHUNK, TOTAL); LEG = S/CHUNK; **SEED = 42 + LEG** (deterministic; retries reuse it; leg 1 = recipe seed 42); argv = Phase-2 argv + `--max-steps TARGET --seed SEED` + (`--ckpt-path <newest>` when S>0). Watchdog half 1 (in-leg): attempts stamp per S — a second consecutive attempt at the same S ⇒ CHAINHALT (scancel remaining chain, distinct exit); post-run assert newest step == TARGET.
- `della_chain_submit.sh` — computes N legs, submits leg 1 free + legs 2..N `--dependency=afterany:<prev>` (afterany + self-computing TARGET ⇒ a crashed leg is retried by its successor from the last good checkpoint — watchdog half 2); each job held→flock'd record (chain manifest with all ids + command block)→released; refuses unless: closure clean, HEAD pushed, `PHASE1_PASS.md` tracked at HEAD (interlock §Rev 3 preserved), and **no exp16-train\*/exp16-chain jobs present** (enforces the racer swap). `--probe` mode: TOTAL=300/CHUNK=100/3 legs/`--time=00:30:00`, separate save-dir `exp16_vanilla_repro_chainprobe`.
- Validation ladder for the chain itself: bash -n → DRYRUN matrix → **3-leg GPU probe** (acceptance: leg boundaries 100/200/300 hit; RESUME lines at legs 2-3; seeds 42/43/44 echoed; LR continues the InverseLR curve across seams; CHAINDONE on a 4th invocation) → racers cancelled → real chain.

**Disclosed deviations added to §1's irreproducibility list:** 8 resume seams (no RNG/dataloader-position restore at each), per-leg reseed (leg k trains on a fresh permutation of the full set; expected per-sample visit count matches continuous training; P(never seen) ≈ 4e-10), one W&B offline run per leg (stitched for the results page), and validation RNG perturbation points shifted relative to a continuous run.
