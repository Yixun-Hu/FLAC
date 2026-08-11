# plan_della_vanilla_repro — exp_16: reproduce vanilla FLAC on della (H100)

**Status:** DRAFT — awaiting fallback plan review (Opus 5 max-effort; Codex unavailable on della) and Yixun approval.
**Branch:** `della-flac-chequity` (cut from `check-equivariance-necessity` @ `e603947`).
**Planner:** Claude Fable 5 (main session, xhigh). **Coder:** Claude Opus 5 subagent. **Reviewer:** Claude Opus 5 (declared fallback — no Codex/npm/node on della; per CLAUDE.md fallback rule).

---

## 1. Goal & headline design decision

Reproduce the released vanilla FLAC (`FLAC_EMA.ckpt`) by training from scratch on one della H100 with the paper's reported recipe, then evaluate on the full published splits and quantify how close a faithful re-run lands.

**Recovered budget:** the paper omits the training length. The released wrapped checkpoint `FLAC.ckpt` embeds `global_step = 67500`, `epoch = 14`, optimizer step tensor 67500, and the exact `InverseLR` state of `FLAC_AR.json`'s scheduler at base lr 5e-5. **The reproduction budget is therefore pre-registered at 67,500 optimizer steps at batch 64** (≈15 epochs, 4500 steps/epoch ⇒ ≈288k train samples). No other budget will be reported as "the reproduction".

**Known-irreproducible factors (disclosed up front):** release RNG seed, dataloader worker scheduling, GPU arch/kernel nondeterminism, and possibly minor dataset-version drift. The reproduction claim is statistical (metrics within tolerance of the release on the full splits), never bit-exact.

## 2. Facts the plan builds on (all verified 2026-08-11, see worklog)

- Config parity: `FLAC.ckpt['model_config']` ≡ repo `FLAC_AR.json` in every architecture/optimizer field. Differences are only: ViT path (`./Models/dinov3-…` local dir in release vs `facebook/dinov3-…` in repo — the release itself trained from a local directory), a release-only `training.demo` block, metric-callback flags, `structured_noise: false`.
- Recipe fields, all already in `FLAC_AR.json` + `defaults.ini`: DiT depth 12 / heads 8 / width 256 (`embed_dim`), AdamW lr 5e-5 β(0.9,0.999) wd 1e-3, InverseLR(inv_gamma 1e6, power 0.5, warmup 0.99), batch 64, `use_ema: true`, `precision bf16-mixed`, flow matching (`diffusion_objective: rectified_flow`, `flow_source: gaussian`), `timestep_sampler: log_snr`, cfg_dropout 0.1.
- DINOv3 ViT-S/16 present at `models/dinov3-vits16-pretrain-lvd1689m/` (repo symlink → scratch) **and** in the shared HF cache. Loader: `src/models/conditioners.py:455-458`.
- Slurm: account `blanchette`; QOS `gpu-short` 1d / `gpu-medium` 3d / `gpu-long` 6d; H100s in partition `gpu`; compute nodes assumed offline (HF/W&B must not need network). Login node GPU-free ⇒ all runs via sbatch.
- Eval references: exp_01's per-seed unseen JSONs (identical protocol, A6000, commit 0bd5da0) + paper Table 1; exp_01 noise floors (K8: T60 ±0.012, C50 ±0.003, EDT ±0.07, R@1 ±0.10).
- `eval_FLAC.py` writes metrics + predictions beside `--ckpt-path` ⇒ calibration evals run **without** `--store_predictions` (README lines 40-49 include it; deviation disclosed — predictions are heavy, land in home, and are not needed to verify metrics).

## 3. Della accommodations (the only code changes; all on `della-flac-chequity`)

### 3a. `src/models/conditioners.py` — local ViT snapshot resolution (TDD)

New module-level pure function, called at the two consumption points (`AutoConfig.from_pretrained`, `AutoModel.from_pretrained`):

```python
def resolve_vit_model_path(name_or_path, local_root="models"):
    """Prefer a local snapshot under ``<local_root>/<basename>`` when present.

    The FLAC release itself trained its ViT from a local directory
    ('./Models/dinov3-vits16-pretrain-lvd1689m', per the released checkpoint's
    embedded model_config); on cluster nodes without network access this maps a
    hub id like 'facebook/dinov3-vits16-pretrain-lvd1689m' onto the repo-level
    ``models/`` symlink. An input that is already an existing directory, or has
    no local snapshot, is returned unchanged (hub-cache/offline behavior then
    applies).
    """
    if os.path.isdir(name_or_path):
        return name_or_path
    candidate = os.path.join(local_root, os.path.basename(name_or_path))
    if os.path.isdir(candidate):
        return candidate
    return name_or_path
```

The JSON configs stay **byte-unchanged** (`facebook/dinov3-…`), so checkpoints trained here embed the portable hub id and the A6000/neuronic checkouts are unaffected (no `models/` dir there ⇒ function is a no-op). Belt-and-braces: launch scripts also export `HF_HOME=/scratch/gpfs/BLANCHETTE/yh4742/hf_cache` + `HF_HUB_OFFLINE=1`.

**Tests (`src/tests/test_vit_local_resolution.py`, written first, red→green):**
1. `test_existing_dir_returned_unchanged` — an existing directory path is returned as-is (release-style `./Models/...` input).
2. `test_hub_id_resolves_to_local_snapshot` — with `tmp_path/models/dinov3-vits16-pretrain-lvd1689m` present and `local_root` pointed there, `facebook/dinov3-vits16-pretrain-lvd1689m` resolves to the local dir.
3. `test_hub_id_without_snapshot_unchanged` — no local dir ⇒ input returned unchanged.
4. `test_plain_name_no_slash` — a bare name (`some-model`) with a matching local dir resolves; without one, unchanged.
5. `test_loader_callsite_uses_resolver` — monkeypatch `AutoModel.from_pretrained`/`AutoConfig.from_pretrained` to record the path argument; drive the `ViTCoordinates` branch of `create_multi_conditioner` with a fake `models/` root and assert the recorded path is the resolved one (no network, no real weights).

### 3b. `.gitignore` — one line: `/models` (the scratch symlink; mirrors `weights/`, `outputs_FLAC/`).

### 3c. Storage moves (runtime, no commits; recorded in worklog)

Released checkpoints move to scratch, file-level symlinks keep repo paths valid (configs/commands unchanged):
```
mv weights/FLAC/{FLAC.ckpt,FLAC_EMA.ckpt,FLAC_HAA.ckpt,VAE.ckpt,VAE.safetensors} /scratch/gpfs/BLANCHETTE/yh4742/FLAC/weights/
mv weights/AGREE/{AGREE_AR.pt,AGREE_fullAR.pt,AGREE_fullHAA.pt}                  /scratch/gpfs/BLANCHETTE/yh4742/FLAC/weights/
ln -s <scratch>/... back into weights/FLAC/, weights/AGREE/   # one symlink per file
```
Metric JSONs (tiny, force-added) stay as real files in `weights/FLAC/`. Training outputs go to `--save-dir /scratch/gpfs/BLANCHETTE/yh4742/FLAC/checkpoints/exp16_vanilla_repro` (flag only, no code change). W&B: `WANDB_MODE=offline`, `WANDB_DIR=/scratch/gpfs/BLANCHETTE/yh4742/FLAC/wandb`.

### 3d. exp_16 Slurm kit (in the exp folder, della-style)

- `della_eval.sbatch` — parameterized calibration/eval driver: 1 GPU, conda env activation, env exports (HF/W&B offline), fail-closed gates (expected SHA, dataset item count printed by the loader, clean checkpoint load i.e. **no** `--allow-partial-load`), eval argv echoed, tee'd timestamped log into the exp folder, `DRYRUN=1` support.
- `della_train.sbatch` — the 67.5k-step training leg: same gating + env, `--gres=gpu:h100:1`, resume-aware (`--ckpt-path` latest checkpoint if present so a preempted/timed-out leg continues; PL resume is not bit-exact — disclosed), tee'd log.
- `della_submit.sh` — thin submit wrapper: verifies committed+pushed SHA, records job id + argv into `della_vanilla_repro_command.md` at submit time.
No worktree-lease machinery (single-writer checkout on della); the sbatch gate refuses to run if `git rev-parse HEAD` ≠ the submitted `EXPECT_SHA` or the tree is dirty — simpler than exp_11's content-scoped gate but sufficient while only this session commits here. Scripts are Coder-written, reviewed like all code; `bash -n` + `DRYRUN=1` are their test rungs.

## 4. Phase plan

### Phase 1 — eval calibration on the released EMA (user-mandated, before any training)

Two runs on one H100 (or A100 if queueing is faster — arch disclosed in results):

- **Run A (README lines 40-49, protocol flags explicit per announcement 05):**
  `python eval_FLAC.py --model-config src/configs/model_configs/FLAC/AR/FLAC_AR.json --dataset-config src/configs/dataset_configs/AR/eval/acousticroom_seeneval.json --ckpt-path weights/FLAC/FLAC_EMA.ckpt --cfg-scale 1.0 --steps 1 --eval-name exp16_calib_seen_K8_seed42 --seed 42 --cond-method vanilla --cond-autocast default` (no `--store_predictions`; deviation from README as justified in §2).
- **Run B (cross-machine exactness anchor):** same but `--dataset-config .../acousticroom_unseeneval.json`, `--eval-name exp16_calib_unseen_K8_seed42` — directly comparable to exp_01's stored `..._unseen_K8_seed42.json` (T60 8.6238 / C50 0.9687 / EDT 37.0786 / FD 0.3053 / R@1 7.1012).

**Acceptance (pre-registered):** loader prints the full split ("6337 files in 17 subfolders" for unseen); clean checkpoint load; Run B within **3× exp_01's 5-seed noise floor** per metric of exp_01's seed-42 values (same seed, different GPU arch ⇒ small numeric drift only); Run A lands in the paper's seen-set table range (Tab. A.4/A.5, extracted at analysis time). Failure ⇒ stop, diagnose before any training.

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

- Batch 64 on ONE GPU matches the paper literally (single H100, batch 64) and the BN rule (BN sees the full 64; no accumulation games). `precision` stays `bf16-mixed` (defaults.ini). `--max-steps 67500` uses the exp_07 flag; no code edits.
- Val cadence is measurement-only (PL runs `model.eval()`; BN stats and EMA are untouched by validation) — matches README's 2500. ~27 checkpoints × 724 MB ≈ 20 GB on scratch.
- **Ladder before the full leg:** static checks → `python -m pytest src/tests -q` (new tests + regressions) → `DRYRUN=1` sbatch gates → **smoke** (gpu-test/gputest: `--max-steps 50`, checkpointing off via `--checkpoint-every` high, measure steps/s) → wall-clock ETA = 67500/steps_per_s × 1.5 margin → pick QOS (`gpu-short` if <1 d, else `gpu-medium`; resume chain if beyond) → full leg.
- Acceptance at launch (recorded in worklog before submit): job runs the pushed SHA; 1× H100; global batch 64; first optimizer step completes with no OOM/NaN; loss curve logged offline to scratch W&B; checkpoint lands at step 2500.

### Phase 3 — evaluate the reproduction + report

- Export/eval the final checkpoint (`epoch=14-step=67500.ckpt`): `eval_FLAC.py` loads the wrapped checkpoint and strips EMA prefixes inline (verified exp_01 path). **EMA weights are the deployment artifact**, matching the release.
- Full protocol per announcements 01/04: full unseen split, K ∈ {1, 8}, seeds 42-46 (10 runs), `--cond-method vanilla --cond-autocast default`; then seen split K8 seed 42 for the Table-A parity row.
- Add a `gen_model_comparison.py` row spec for the repro checkpoint, regenerate `model_comparison.md`, commit + push (announcement 04).
- **Reproduction verdict (pre-registered):** vs the release's exp_01 numbers — PASS if every headline unseen metric (T60, C50, EDT, R@1, FD; both K) is within **±10% relative**; PARTIAL if within ±20% and ordering vs baselines is preserved; FAIL otherwise. Training-seed variance is unknown (n=1 by necessity); the verdict language will say "consistent with the release under an unknown seed", never "bit-identical".
- `_results.md`, `_analysis.md`, `della_vanilla_repro_01_results.html` (+assets), commit log per SOP.

## 5. Commit plan (each < 200 changed lines, SHAs to `commits_della_vanilla_repro.md`)

1. exp_16 scaffold: query + worklog + this plan (+ CLAUDE.md /init refresh carried from session start).
2. `.gitignore` `/models` line.
3. TDD red→green: `src/tests/test_vit_local_resolution.py` + `resolve_vit_model_path` + call-site wiring.
4. `della_eval.sbatch` + `della_submit.sh` (+ DRYRUN gates).
5. `della_train.sbatch`.
6+. Params/command/results/analysis/review artifacts as they land.

Reviews: one fallback-reviewer round per Coder round (3-5 grouped smallly), a `full` integrative review before the Phase-2 launch; every review file opens with the Opus 5 fallback by-line.

## 6. Risks & mitigations

- **Throughput unknown** → smoke-measured before committing QOS; resume-aware sbatch if a leg dies (PL resume non-bit-exact — disclosed in results).
- **Compute nodes offline** → HF_HUB_OFFLINE + local `models/` + offline W&B; calibration eval exercises all three before training does.
- **`models` symlink absence on other machines** → resolver no-ops; configs unchanged; tests enforce.
- **Two-writer repo** → work stays on `della-flac-chequity`; `git pull --rebase` before every commit; no edits to exp_11/14/15 kits or their gates.
- **Seen-eval reference numbers** are in the paper's supplementary tables; if extraction is ambiguous, Run B (exact exp_01 anchor) alone gates Phase 2, Run A becomes descriptive.
- **Release seed unknown** → statistical acceptance bands (§4 Phase 3), stated in every artifact.
