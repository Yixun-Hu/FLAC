# Evaluation Plan — V3 Ablation vs Baseline on Arbitrary-Context K=8 / K=1

## Context

Two models trained side-by-side at effective batch-size 32 on AcousticRooms:

- **Ablation V3** — `FLAC_AR_arbRIR_v0.json`, trained on `AR_md_arbRIR_v0.py` (arbitrary `(s_i, r_i)` context, source-excluded, 3-token fused-pose design).
- **Baseline** — `FLAC_AR.json`, trained on `AR_md.py` (same-receiver context, classic `context_poses` design). The baseline run crashed once and was resumed from `epoch=4-step=40000.ckpt`.

Goal: measure how well **each** model reconstructs a query RIR when given **arbitrary-receiver context** (the V3 distribution) at two context budgets, **K=8** and **K=1**, on the AR seen and unseen eval splits. This isolates whether the V3 coordinate formulation actually helps when context comes from arbitrary `(s_i, r_i)` pairs, vs. the baseline which never saw arbitrary receivers in training.

## The compatibility problem (why we cannot reuse the existing configs)

`eval_FLAC.py` builds the model from `--model-config` and the dataloader (incl. its `custom_metadata_module`) from `--dataset-config`. `MultiConditioner` pulls **only** the metadata keys named in each model's `conditioning.configs[].id`, and raises `ValueError` if a required key is missing; extra keys are silently ignored.

| | cross-attn keys the model consumes | provided by `AR_md_arbRIR_v0.py`? |
|---|---|---|
| **Ablation V3** (`FLAC_AR_arbRIR_v0.json`) | `context_poses_vit`, `context_fused_pose`, `context_audio` | ✅ all present |
| **Baseline** (`FLAC_AR.json`) | `context_poses_vit`, **`context_poses`**, `context_audio` | ❌ `context_poses` **missing** |

`AR_md_arbRIR_v0.py` emits `context_audio`, `context_poses_vit`, `context_fused_pose` (+ `source`, `source_vit`, `depth`, `scene`) — **no `context_poses`**. So the baseline model cannot run on the existing `acousticroom_*eval_arbRIR_v0.json` configs. Conversely the baseline's own eval configs use same-receiver sampling (`AR_md.py`), which is **not** the arbitrary distribution we want to test.

**Solution:** *superset* eval metadata modules that do the V3 arbitrary, source-excluded sampling **once per sample** and emit **both** key families from the same picked `(s_i, r_i)` set:

- `context_audio`            — picked RIR waveforms (both models)
- `context_poses_vit`        — `s_i − r_q` (both models, ViT stream)
- `context_fused_pose`       — `{pair_local: s_i−r_i, src_qrel: s_i−r_q, recs_qrel: r_i−r_q}` (ablation only)
- `context_poses`            — Scheme-dependent (baseline only; see below)
- `source`, `source_vit`, `depth`, `scene` — unchanged query side

The picks must be **byte-identical across all 12 runs, both schemes, and both models** for the comparison to be apples-to-apples. Relying on RNG reproducibility for this is **not safe in this codebase** — see the next section. Instead the sampling is done **once, offline, single-threaded**, frozen to a manifest on disk, and the eval modules only *read* it (no `np.random.choice` in the dataloader path).

## Serious Issue: the RNG determinism assumption is too strong — use a frozen manifest

The earlier draft claimed "`--seed 42`, `--num-workers 4`, `--batch-size 32` ensures both models see exactly the same context." **This is unsafe in this codebase.** Code-verified:

- `eval_FLAC.py:99` consumes the loader with a **manual `for batch in eval_dl` loop**, *not* a PL `Trainer.test()`.
- `create_dataloader_from_config` (`src/data/dataset.py:411`) constructs `torch.utils.data.DataLoader(... num_workers=N, persistent_workers=True ...)` with **no `worker_init_fn`**.
- `pl.seed_everything(seed, workers=True)` (`eval_FLAC.py:74`) only seeds the main process and sets `PL_SEED_WORKERS=1`. PL's per-worker `pl_worker_init_function` is attached **only** by the Trainer's data connector — which is bypassed here. PyTorch's default worker init seeds `torch`/`random` per worker but **not numpy**.

Consequence: `np.random.choice` inside the metadata module runs in worker processes whose **numpy** RNG is seeded only by fork-inheritance of the parent's numpy state (Linux `fork` start method). With `num_workers>1` all workers fork the *same* parent numpy state (the classic DataLoader-numpy footgun): sampling is correlated across workers, and run-to-run / A-vs-B identity holds only *incidentally* and breaks under any change to worker count, batch size, dataset length, import order, or upstream numpy consumption. Resting the scientific comparison on this is exactly the "too strong" assumption.

**Fix (primary mechanism, not a fallback): persist the sampled context to a frozen manifest and reuse it.**

1. A standalone pre-pass script (`tools/build_arbRIR_eval_manifest.py`) iterates each eval split **single-threaded in the main process** with an explicit local `np.random.RandomState(seed)`, runs the *exact* source-excluded `np.random.choice` once per query, and writes a JSON manifest keyed by the **scene triple** (see contract below): `{ "<scene_name>/<scene_id>/<filename>" : [ctx_wav_filename_0, …, ctx_wav_filename_7] }`. K=1 is the **prefix** of the K=8 list (nested), so K1 ⊂ K8 by construction.
2. The A/B eval metadata modules **do not sample**. `get_custom_metadata` derives the same scene-triple key from `info['relpath']`, looks it up in the manifest, and loads exactly those files in listed order. Identical context is then guaranteed *by construction* — independent of `num_workers`, `batch_size`, fork semantics, and the A↔B swap (both schemes read the same manifest; scheme only changes the post-load `context_poses` assignment).
3. The determinism check (Verification #3) is retained as a **mandatory hard gate** on top of this — belt-and-suspenders, must pass before any matrix run.

This removes RNG from the hot path entirely and makes "both models see exactly the same context" a structural guarantee rather than a probabilistic hope.

### Manifest-key contract (Remaining-Bug-2 mitigation)

The builder runs **offline**; the lookup happens at **dataloader time**. `info['relpath']` (`dataset.py:276`) is `path.relpath(audio_filename, root_path)` — its exact string depends on `self.root_paths` and the on-disk prefix, so keying the manifest by a raw build-time path risks a total lookup miss (every query → `KeyError`, eval dies). Mitigation, code-verified safe:

- **Key = `"<scene_name>/<scene_id>/<filename>"`** — the last three path components, exactly what `AR_md_arbRIR_v0.py:39-41` already parses from `info['relpath']` (`rel_path.split("/")[-3:]`, dropping the extension on the filename). This triple is invariant to root-path/prefix conventions, so build-time and runtime keys match by construction. The builder parses query paths with the *same* `split("/")[-3:]` logic; the module reuses its existing parse.
- **seen/unseen resolver = fail-loud.** The module selects the manifest from `info['json_file_path']` (`dataset.py:267`, e.g. `data/AR/seen_eval.json` → `data/AR/arbRIR_v0_eval_manifest_seen.json`); cross-check against `info['seeneval']`/`info['unseeneval']` (`:265-266`). If neither resolves unambiguously, **raise** — never silently load the wrong/another manifest.
- **No-missing-key guarantee.** Verification #1 asserts every query in `data/AR/{seen,unseen}_eval.json` has a manifest entry and every module lookup resolves (no `KeyError` over the full split) before any matrix run.

`info['seeneval']`, `info['unseeneval']`, `info['json_file_path']`, `info['modalities']`, `info['relpath']` are all confirmed present in `info` before `get_custom_metadata` is invoked (`dataset.py:265-267, 286, 274-276` → call at `:288`), so this path needs no new config field and no `dataset.py` change.

## Decision: run BOTH Scheme A and Scheme B (no longer "A primary, B optional")

In `AR_md.py` the baseline's `context_poses = src_loc − rec_loc` (pair-local `s_i − r_i`), which equals `s_i − r_q` only because same-receiver training forces `r_i = r_q`. With arbitrary `r_i ≠ r_q` the two diverge. **We run both definitions as primary deliverables** and report them side-by-side.

### The two schemes have different geometric meanings

| Item | Scheme A: `s_i − r_q` | Scheme B: `s_i − r_i` |
|---|---|---|
| Geometric meaning | Context source position in the **query receiver** coordinate frame | Context source position in its **own context receiver** frame, i.e. pair-local |
| When receiver is shared (`r_i = r_q`) | Equals `s_i − r_q` | Equals `s_i − r_q` |
| In the arbitrary setting (`r_i ≠ r_q`) | Still uses `r_q` as the origin | Uses the context pair's own receiver `r_i` as origin |

**Key point: when `r_i = r_q` the two are exactly the same. The difference only surfaces under arbitrary-pair evaluation** — which is precisely the distribution this study probes, so both must be measured.

### Concrete example

Query pair `(s_q=[5,1,2], r_q=[1,1,2])`, context pair `(s_i=[3,1,4], r_i=[2,1,3])`:

```text
Scheme A: context_poses = s_i − r_q = [3,1,4] − [1,1,2] = [2, 0, 2]
Scheme B: context_poses = s_i − r_i = [3,1,4] − [2,1,3] = [1, 0, 1]
```

### Why both, not one

- **Scheme A** = the geometric quantity the baseline *effectively learned* (its training distribution had `r_i = r_q`, so `s_i − r_i` collapsed to `s_i − r_q`); same frame as its `context_poses_vit` stream. Measures the baseline given the "fairest" pose vector — isolates the *fused-coordinate* difference vs. V3.
- **Scheme B** = the *literal* `AR_md.py` formula applied unchanged. Measures the baseline under exactly its training-time code path, which under arbitrary `r_i` injects an additional origin-shift distribution gap. Quantifies how much the moved origin alone costs.
- Reporting both A and B brackets the baseline's true behaviour and makes the V3 comparison robust to the `context_poses`-semantics choice rather than contingent on it.

The V3 ablation model **ignores `context_poses`** entirely (it consumes `context_poses_vit`, `context_fused_pose`, `context_audio`). Since the RNG call order is identical between the Scheme-A and Scheme-B modules, the ablation produces **identical** results on either — so the ablation is evaluated **once per (K, split)** cell (run on the Scheme-A configs); only the **baseline** is run twice per cell (A and B).

## Checkpoint selection — latest common step, same budget for both

**Rule (do not hardcode a step number):** at execution time, enumerate both checkpoint dirs, take the set of optimizer steps present in **both** runs, and pick the **maximum** such step. Use that single step for **all** runs in the matrix so every cell compares equal training budget. Both runs are still training and emitting new checkpoints, so this is re-evaluated the moment execution starts and then **frozen** for the whole matrix (do not re-pick mid-matrix, or ablation and baseline cells would land on different steps).

```bash
ABL_DIR=outputs_FLAC/FLAC_arbRIR_v0/FLAC_arbRIR_v0_training/checkpoints
BASE_DIR=outputs_FLAC/FLAC_AR_baseline_short/FLAC_AR_baseline_short_training/checkpoints
COMMON_STEP=$(comm -12 \
  <(ls "$ABL_DIR"  | sed -n 's/.*-step=\([0-9]*\)\.ckpt/\1/p' | sort -n) \
  <(ls "$BASE_DIR" | sed -n 's/.*-step=\([0-9]*\)\.ckpt/\1/p' | sort -n) \
  | sort -n | tail -1)
CKPT_ABL=$(ls "$ABL_DIR"/*-step=${COMMON_STEP}.ckpt)
CKPT_BASE=$(ls "$BASE_DIR"/*-step=${COMMON_STEP}.ckpt)
```

Snapshot at planning time (illustrative only — recompute at run time): ablation had `…-step=170000`, baseline `…-step=145000`, so the latest common step was **145000** (`epoch=15-step=145000.ckpt` for both). The number will be higher by execution; the rule above always yields the correct, equal-budget pair.

## Artifacts to create

| Purpose | Path | Notes |
|---|---|---|
| Manifest builder (pre-pass) | `tools/build_arbRIR_eval_manifest.py` | Single-process, explicit `np.random.RandomState(42)`. Replicates the exact source-excluded candidate enumeration + `np.random.choice` from `AR_md_arbRIR_v0.py:sample_arbitrary_context`. Emits one manifest per split at K=8; K=1 = first element of each K=8 list (nested). Idempotent; refuses to overwrite an existing manifest unless `--force`. |
| Frozen manifest — seen | `data/AR/arbRIR_v0_eval_manifest_seen.json` | `{"<scene>/<scene_id>/<filename>": [ctx_wav_fn × 8]}` (scene-triple key — see Manifest-key contract) over `data/AR/seen_eval.json`. Checked in (small, text) so the exact context set is reproducible and reviewable. |
| Frozen manifest — unseen | `data/AR/arbRIR_v0_eval_manifest_unseen.json` | Same over `data/AR/unseen_eval.json`. |
| Eval module — Scheme A | `src/configs/dataset_configs/custom_metadata/AR_md_arbRIR_v0_eval_A.py` | Clone `AR_md_arbRIR_v0.py` query side; **replace `sample_arbitrary_context` with a manifest lookup** keyed by the scene-triple parsed from `info['relpath']` (`split("/")[-3:]`; no `np.random.choice`). Loads the manifest's first `max_context` files in listed order, computes pair_local/src_qrel/recs_qrel, emits all V3 keys + `md['context_poses'] = src_qrel` (`s_i − r_q`). Manifest resolved fail-loud from `info['json_file_path']` (see contract). Serves ablation **and** baseline-A. |
| Eval module — Scheme B | `src/configs/dataset_configs/custom_metadata/AR_md_arbRIR_v0_eval_B.py` | Identical to A except `md['context_poses'] = pair_local` (`s_i − r_i`, literal `AR_md.py` formula). Reads the **same** manifest. Serves baseline-B only. |
| Dataset cfg — A, seen, K=8 | `acousticroom_seeneval_arbRIR_v0evalA_8.json` | module A; `max_context: 8` |
| Dataset cfg — A, seen, K=1 | `acousticroom_seeneval_arbRIR_v0evalA_1.json` | module A; `max_context: 1` |
| Dataset cfg — A, unseen, K=8 | `acousticroom_unseeneval_arbRIR_v0evalA_8.json` | module A; `unseeneval: true`; `max_context: 8` |
| Dataset cfg — A, unseen, K=1 | `acousticroom_unseeneval_arbRIR_v0evalA_1.json` | module A; `max_context: 1` |
| Dataset cfg — B, seen, K=8 | `acousticroom_seeneval_arbRIR_v0evalB_8.json` | module B; `max_context: 8` |
| Dataset cfg — B, seen, K=1 | `acousticroom_seeneval_arbRIR_v0evalB_1.json` | module B; `max_context: 1` |
| Dataset cfg — B, unseen, K=8 | `acousticroom_unseeneval_arbRIR_v0evalB_8.json` | module B; `unseeneval: true`; `max_context: 8` |
| Dataset cfg — B, unseen, K=1 | `acousticroom_unseeneval_arbRIR_v0evalB_1.json` | module B; `max_context: 1` |

All eval configs (8 total) are clones of `acousticroom_{seen,unseen}eval_arbRIR_v0.json` changing **only** `custom_metadata_module` and `acoustic_context.max_context`. The module resolves which manifest to load (seen vs unseen) from `info['json_file_path']`, cross-checked against `info['seeneval']`/`info['unseeneval']`, raising on ambiguity — no extra config field needed (see Manifest-key contract). No model-config or other Python changes — `eval_FLAC.py` and both model configs are reused untouched.

**K=1 ⊂ K=8 by construction:** the K=1 run loads the **first** entry of the same manifest list the K=8 run uses, so the single K=1 context is exactly the first of the eight — context budgets are nested, not independently sampled. `context_fused_pose` tensors are `[1,3]`, `context_audio` `[1,1,T]` — conditioners already handle `N=1`. Source-exclusion (`s_i ≠ s_q`) is enforced once, in the manifest builder. Note that K=1 here is the first element of an 8-permutation, not an independent K=1 draw. Both are statistically valid single-context samples (source-exclusion preserved); we use the nested form to keep K=1 ⊂ K=8 for paired analysis.

## Evaluation matrix

The ablation runs **once** per (K, split) (Scheme-invariant — it ignores `context_poses`); the baseline runs **twice** per (K, split) (A and B). Splits = {seen, unseen}, K = {8, 1}.

| Model | configs used | runs |
|---|---|---|
| Ablation V3 (`FLAC_AR_arbRIR_v0.json`) | A configs (Scheme irrelevant) | 2 K × 2 splits = **4** |
| Baseline-A (`FLAC_AR.json`) | A configs | 2 K × 2 splits = **4** |
| Baseline-B (`FLAC_AR.json`) | B configs | 2 K × 2 splits = **4** |

**Total = 12 runs**, all from the single locked `…-step=${COMMON_STEP}.ckpt` for each model.

| # | Model config | Dataset config | ckpt | eval-name |
|---|---|---|---|---|
| 1 | `FLAC_AR_arbRIR_v0.json` | `…_seeneval_arbRIR_v0evalA_8.json` | `$CKPT_ABL` | `arbRIR_v0_seen_K8` |
| 2 | `FLAC_AR_arbRIR_v0.json` | `…_seeneval_arbRIR_v0evalA_1.json` | `$CKPT_ABL` | `arbRIR_v0_seen_K1` |
| 3 | `FLAC_AR_arbRIR_v0.json` | `…_unseeneval_arbRIR_v0evalA_8.json` | `$CKPT_ABL` | `arbRIR_v0_unseen_K8` |
| 4 | `FLAC_AR_arbRIR_v0.json` | `…_unseeneval_arbRIR_v0evalA_1.json` | `$CKPT_ABL` | `arbRIR_v0_unseen_K1` |
| 5 | `FLAC_AR.json` | `…_seeneval_arbRIR_v0evalA_8.json` | `$CKPT_BASE` | `baseline_A_seen_K8` |
| 6 | `FLAC_AR.json` | `…_seeneval_arbRIR_v0evalA_1.json` | `$CKPT_BASE` | `baseline_A_seen_K1` |
| 7 | `FLAC_AR.json` | `…_unseeneval_arbRIR_v0evalA_8.json` | `$CKPT_BASE` | `baseline_A_unseen_K8` |
| 8 | `FLAC_AR.json` | `…_unseeneval_arbRIR_v0evalA_1.json` | `$CKPT_BASE` | `baseline_A_unseen_K1` |
| 9 | `FLAC_AR.json` | `…_seeneval_arbRIR_v0evalB_8.json` | `$CKPT_BASE` | `baseline_B_seen_K8` |
| 10 | `FLAC_AR.json` | `…_seeneval_arbRIR_v0evalB_1.json` | `$CKPT_BASE` | `baseline_B_seen_K1` |
| 11 | `FLAC_AR.json` | `…_unseeneval_arbRIR_v0evalB_8.json` | `$CKPT_BASE` | `baseline_B_unseen_K8` |
| 12 | `FLAC_AR.json` | `…_unseeneval_arbRIR_v0evalB_1.json` | `$CKPT_BASE` | `baseline_B_unseen_K1` |

## Exact commands

`max_steps`/`train.py` are irrelevant here. Pin eval to **one GPU** via `CUDA_VISIBLE_DEVICES` (both A6000s currently run training, ~21 GB / 49 GB each; ~28 GB free on each — eval fits but slows the training run sharing that GPU; see Operational notes).

```bash
ABL_DIR=outputs_FLAC/FLAC_arbRIR_v0/FLAC_arbRIR_v0_training/checkpoints
BASE_DIR=outputs_FLAC/FLAC_AR_baseline_short/FLAC_AR_baseline_short_training/checkpoints
COMMON_STEP=$(comm -12 \
  <(ls "$ABL_DIR"  | sed -n 's/.*-step=\([0-9]*\)\.ckpt/\1/p' | sort -n) \
  <(ls "$BASE_DIR" | sed -n 's/.*-step=\([0-9]*\)\.ckpt/\1/p' | sort -n) \
  | sort -n | tail -1)
CKPT_ABL=$(ls "$ABL_DIR"/*-step=${COMMON_STEP}.ckpt)
CKPT_BASE=$(ls "$BASE_DIR"/*-step=${COMMON_STEP}.ckpt)
echo "Locked common step=$COMMON_STEP  ABL=$CKPT_ABL  BASE=$CKPT_BASE"

# Run #1 (ablation, seen, K=8) — template; the other 11 vary
#   --model-config, --dataset-config, --ckpt-path, --eval-name per the matrix.
CUDA_VISIBLE_DEVICES=1 python eval_FLAC.py \
  --model-config src/configs/model_configs/FLAC/AR/FLAC_AR_arbRIR_v0.json \
  --dataset-config src/configs/dataset_configs/AR/eval/acousticroom_seeneval_arbRIR_v0evalA_8.json \
  --ckpt-path "$CKPT_ABL" --cfg-scale 1.0 --steps 1 \
  --batch-size 32 --num-workers 4 --seed 42 \
  --eval-name arbRIR_v0_seen_K8
```

**Determinism is guaranteed by the frozen manifest, not by RNG flags.** Context identity across all 12 runs / both schemes / both models holds *by construction* because every module reads the same checked-in `data/AR/arbRIR_v0_eval_manifest_{seen,unseen}.json` (built once, single-process, with an explicit `np.random.RandomState(42)`). `--num-workers`/`--batch-size`/`shuffle=False` no longer affect *which* context is drawn (no `np.random.choice` in the dataloader path) — keep `--batch-size 32 --num-workers 4` for speed consistency, but correctness does not depend on them. Build the manifest **before** any eval run; never regenerate it mid-matrix (treat it as immutable like the checkpoints). `--seed 42` is still passed to `eval_FLAC.py` for the model-side noise draw (`torch.randn` at sampling).

## Metrics & interpretation

Both model configs already carry the same `training.metrics` block (`eval_T60/C50/EDT/FD/retrieval`, `AGREE_ckpt: weights/AGREE/AGREE_fullAR.pt`). `eval_FLAC.py` writes `<ckpt_dir>/<ckpt>_metrics_1_1.0_<eval-name>.json` and prints to stdout.

- **Use the per-scene mean** for headline comparison (CLAUDE.md: paper numbers average per-scene results; the script also prints an all-samples mean which differs — don't mix them).
- `AGREE_fullAR.pt` is the eval-only AGREE — correct for FD/Recall here; never as a downstream backbone.
- Report, **per split**, a table with rows = {Ablation V3, Baseline-A (`s_i−r_q`), Baseline-B (`s_i−r_i`)}, cols = {K=8, K=1}, for each of T60 / C50 / EDT / l1 / FD / Recall. Signals of interest: (i) does V3's gap over *both* baseline schemes widen at K=1 and under the arbitrary distribution; (ii) the A−B spread quantifies how much the moved-origin distribution shift alone hurts the baseline.

## Operational notes

- Both training runs are live and own both GPUs (~21 GB / 49 GB each). Eval adds a few GB; it runs but **slows the training run on whichever GPU it shares**. `max_steps=1000000` ⇒ neither training run finishes on its own, so "wait for a free GPU" is not viable — pin eval to one GPU and accept the slowdown.
- Eval is read-only w.r.t. training state and checkpoints — safe to run alongside; the locked `step=${COMMON_STEP}` checkpoints are immutable on disk, so eval is not time-sensitive.
- 12 runs; at `--steps 1` rectified-flow each run is short (minutes), dominated by AGREE FD/retrieval embedding passes.

## Verification (before the full matrix)

1. **Build + sanity-check the manifests** — run `tools/build_arbRIR_eval_manifest.py` for seen and unseen. Assert: every query in `data/AR/{seen,unseen}_eval.json` has exactly 8 context filenames; no context shares the query source node (source-exclusion holds); all listed files exist on disk; the K=1 prefix is a valid single entry. **Key-contract assertion:** for every query, the scene-triple key the *module* would compute from `info['relpath']` (`split("/")[-3:]`) is present in the manifest — i.e. a dry-run lookup over the full split raises **zero** `KeyError`. Manifests are then frozen (read-only) for the whole matrix.
2. **Superset-module shape check** — import `AR_md_arbRIR_v0_eval_A.get_custom_metadata` and `…_eval_B.…` on one real AR sample at `max_context` 8 then 1; assert keys `{scene, source, source_vit, depth, context_audio, context_poses_vit, context_fused_pose, context_poses}` and shapes (`context_audio [N,1,9600]`; all pose tensors `[N,3]`). Module A: `context_poses == context_poses_vit == context_fused_pose['src_qrel']`. Module B: `context_poses == context_fused_pose['pair_local']` and (generically) `≠ context_poses_vit`. Assert the K=1 tensors equal the K=8 tensors sliced to `[:1]` (nesting).
3. **Determinism hard gate — MANDATORY, do not skip; blocks the matrix.** Build the K=8 dataloader **twice with deliberately different `num_workers` (`1` vs `4`) and `batch_size`**, collect all `context_audio` for the full split each time, and assert they are **bit-identical** — this proves context is manifest-bound and *independent* of dataloader plumbing (the property the old RNG plan lacked). (Use `1` vs `4`, not `0` vs `4`: `create_dataloader_from_config` hardcodes `persistent_workers=True` at `dataset.py:411`, and PyTorch rejects `num_workers=0` with that option — both contrast values must be `>0`.) Then assert the Scheme-A and Scheme-B modules yield bit-identical `context_audio`, `context_poses_vit`, and `context_fused_pose` (only `context_poses` may differ). If any assertion fails, **stop** — do not run the matrix.
4. **Cross-model key check** — run baseline cell #5 for one batch; confirm `MultiConditioner` does not raise on `context_poses`.
5. **Smoke eval** — run cell #1 to completion; confirm metrics JSON written and FD/Recall finite.
6. Only after 1–5 pass: run the full 12-cell matrix and tabulate per-scene means.

## Critical files

- `eval_FLAC.py` — eval entrypoint (CLI: `--model-config --dataset-config --ckpt-path --cfg-scale --steps --batch-size --num-workers --seed --eval-name --store_predictions`); K comes from the dataset config. **`:74`** `pl.seed_everything(seed, workers=True)` (main-process only here); **`:99`** manual `for batch in eval_dl` loop (no Trainer) — why PL per-worker seeding never attaches.
- `src/data/dataset.py:411` — `DataLoader(... num_workers, persistent_workers=True ...)` built with **no `worker_init_fn`** → numpy unseeded in workers → the reason RNG determinism is unsafe and the manifest is required.
- `src/configs/dataset_configs/custom_metadata/AR_md_arbRIR_v0.py` — `sample_arbitrary_context` (lines 123–188) is the exact candidate-enumeration + source-exclusion + `np.random.choice` logic the manifest builder must replicate; query-side keys (lines 75–87) are cloned into the eval modules.
- `src/configs/dataset_configs/custom_metadata/AR_md.py` — reference for the baseline `context_poses` semantic (Scheme-A vs Scheme-B distinction).
- `src/configs/dataset_configs/AR/eval/acousticroom_{seen,unseen}eval_arbRIR_v0.json` — structural templates for the 8 new eval configs.
- `src/models/conditioners.py::MultiConditioner.forward` — confirms extra metadata keys are ignored and missing required keys raise (why the superset modules are necessary and sufficient).
- Model configs `FLAC_AR_arbRIR_v0.json` / `FLAC_AR.json` — unchanged; their `cross_attention_cond_ids` define which keys each eval consumes.
