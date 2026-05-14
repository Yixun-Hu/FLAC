# Ablation Plan — Arbitrary-RIR Context with v0 Coordinate Formulation (V3, geometry-fused)

## Context

The current FLAC AR pipeline restricts context RIRs to those sharing **the same receiver** as the query (`src/configs/dataset_configs/custom_metadata/AR_md.py:98`). Each context's geometry is encoded as a single 3-vector — implicitly $s_i^g - r_i^g$ (`AR_md.py:122`), which equals $s_i^g - r_q^g$ only because $r_i = r_q$ by construction.

The v0 ablation lifts that restriction: each context RIR may be drawn from an **arbitrary $(s_i, r_i)$ pair** in the scene (subject to source exclusion below). Each context contributes **three cross-attention tokens** — one audio, one ViT, one fused-pose:

1. **Audio token** — from the existing `RIRConditioner` (unchanged): spectrogram-encoded RIR waveform $E(h_i)$.
2. **ViT token** — from the existing `GeometryConditioner` (unchanged): query-relative source $s_i^g - r_q^g$ passed through the DINOv3 ViT against the query's depth panorama $G_{r_q}$.
3. **Fused-pose token** — a *new* `FusedPoseConditioner` that Fourier-encodes all three pose vectors $(s_i^g - r_i^g,\ s_i^g - r_q^g,\ r_i^g - r_q^g)$ in the query-receiver frame and projects them to a single token.

All geometry is expressed in the **query-receiver-centered frame**. The query token is unchanged: $q = \phi_q(s_q - r_q,\; G_{r_q})$, fed to adaLN exactly as in baseline FLAC.

Goal: a side-by-side ablation against `FLAC_AR.json` — same DiT, same VAE, same training recipe, **same cross-attention sequence length** ($3N$ tokens for $K=N$ context RIRs); the only differences are (a) context RIRs come from arbitrary $(s_i, r_i)$ and (b) the geometric Fourier stream is replaced by a fused-pose token carrying the three v0 vectors.

## Why V3 (vs. V1 all-separate or V2 strict-fusion)

|  | tokens / context | cross-attn cost vs. baseline | audio path independent | new code |
|---|---|---|---|---|
| V1 (all separate) | 5 | 1.67× | yes | none |
| V2 (strict fusion) | 1 | 0.33× | no | one new class |
| **V3 (this plan)** | **3** | **1× (matches baseline)** | **yes** | one new class |

- **Apples-to-apples vs. baseline** — same cross-attention sequence length, so any metric delta is attributable to the *coordinate formulation*, not to attention budget.
- **Audio path stays independent** — the pretrained `RIRConditioner` behavior is preserved; CFG-style "drop audio, keep geometry" remains a clean inference-time ablation.
- **Explicit geometric grouping** — the three pose vectors are bound into one token, telling the model "these three vectors describe the same context entity" rather than having it discover that from data (V1's weakness).

## Resolved decisions

1. **Three tokens per context** (V3): audio + ViT + fused-pose.
2. **Source-exclusion**: context pool = all $(s_i, r_i)$ in the scene where $s_i \neq s_q$. Pairs sharing the query receiver but using a different source are allowed (they degenerate to $r_i - r_q = 0$); pairs sharing the query source are forbidden. Probes stronger cross-source generalization.
3. **Arbitrary sampling at eval too**: `AR_md_arbRIR_v0.py` is used by both train and eval dataset configs; no separate "same-receiver-only" eval pool.
4. **`flow_source = "gaussian"`** to match the baseline. A `nearest_ref` variant is trivial to add later by swapping the model config only.

## Architectural fit

- **Conditioner factory** (`src/models/conditioners.py:333-431`) dispatches by `type` string with `output_dim` auto-injected from `model.conditioning.cond_dim`. Adding a new `"fused_pose"` branch is a 2-line edit, mirroring the existing `"context_fusion"`-style pattern.
- **Reusable pieces** in the same file: `DistEmbedderConditioner` Fourier-features pattern (lines 250-271), `GeometryConditioner` (cross-attn ViT, unchanged), `RIRConditioner` (cross-attn audio, unchanged).
- **Metadata routing**: `MultiConditioner.forward` (`conditioners.py:288-329`) passes dict-typed metadata values through unchanged on the non-Geometry path (lines 317-322, since `isinstance(dict, (list, tuple))` is False). The dataloader collator (`src/data/dataset.py:328-341`) leaves dicts alone (line 339 `else: b = b`). **Therefore the new metadata module can pre-bundle the three pose vectors into a single dict-valued key, and no `MultiConditioner.forward` patch is required.**
- **Token sequence assembly** (`src/models/diffusion.py:139-157`) concatenates whatever cross-attention IDs are listed; the V3 list has exactly three entries (matching baseline cardinality).

## Scope (delta vs. baseline)

| Purpose | Path | Action |
|---|---|---|
| New conditioner class | `src/models/conditioners.py` | **Add** `FusedPoseConditioner` (~30 lines, before `MultiConditioner`) and one `elif` branch in `create_multi_conditioner_from_conditioning_config` (~2 lines, after the `dist_embedder` branch) |
| Metadata module | `src/configs/dataset_configs/custom_metadata/AR_md_arbRIR_v0.py` | **New** |
| Model config | `src/configs/model_configs/FLAC/AR/FLAC_AR_arbRIR_v0.json` | **New** |
| Train dataset config | `src/configs/dataset_configs/AR/train/acousticroom_train_arbRIR_v0.json` | **New** (one-line `custom_metadata_module` change) |
| Eval dataset configs | `acousticroom_seeneval_arbRIR_v0.json`, `acousticroom_unseeneval_arbRIR_v0.json` (+ optional K-variants) | **New** |

Baseline files untouched: `AR_md.py`, `FLAC_AR.json`, existing dataset configs. No changes to `src/training/*`, `src/data/*`, `src/inference/*`, `train.py`, `eval_FLAC.py`, the VAE, or AGREE.

## File-by-file changes

### 1. `src/models/conditioners.py` — new class + factory branch

**`FusedPoseConditioner`** — subclass of `Conditioner`. Constructor signature:

```python
FusedPoseConditioner(
    output_dim: int,          # auto-injected by factory (= cond_dim, 256)
    num_freqs: int = 20,
    max_freq: int = 10,       # Fourier features match DistEmbedderConditioner
    pose_max_val: float = 5.0,
    name: str = "FusedPoseConditioner",
)
```

Internals:
- Fourier-frequency table identical to `DistEmbedderConditioner:243` (`Parameter(2.0**linspace(0, max_freq, num_freqs), requires_grad=False)`).
- Per-pose Fourier dim = `3 * (1 + 2 * num_freqs)` = 123 at the defaults.
- `self.proj = nn.Linear(3 * 123, output_dim)` — one shared projection over the concatenated three-vector Fourier embedding.

`forward(batch_metadata: List[Dict], device) -> Tuple[Tensor, Tensor]`:
- Each entry is a dict `{'pair_local': [N,3], 'src_qrel': [N,3], 'recs_qrel': [N,3]}`.
- Stack across batch → three tensors of shape `[B, N, 3]`.
- For each pose tensor: `p = p / pose_max_val`; build `[p, sin(p[..., None] * freqs).flatten(-2), cos(p[..., None] * freqs).flatten(-2)]` → `[B, N, 123]`. Concat the three → `[B, N, 369]`.
- `out = proj(concat)` → `[B, N, output_dim]`; mask = `torch.ones(B, 1, device=device)`.
- Return `[out, mask]`.

**Factory branch** — insert after the `dist_embedder` branch in `create_multi_conditioner_from_conditioning_config`:

```python
elif conditioner_type == "fused_pose":
    conditioners[id] = FusedPoseConditioner(**conditioner_config)
```

`conditioner_config` already has `output_dim` injected; the rest comes from the JSON `config` block.

### 2. `src/configs/dataset_configs/custom_metadata/AR_md_arbRIR_v0.py` — new

Clone `AR_md.py` for the query side (`source`, `source_vit`, `depth`, `scene` — unchanged). Replace the context sampler with arbitrary-pair selection:

- Get query's `src_loc_query`, `rec_loc_query` from `get_receiver_source_location` (matches `AR_md.py:77-88`). Parse `src_node_query` from the filename.
- Enumerate the scene_id directory: `os.listdir(dir_name)` → list of `S00X_R00Y_hybrid_IR.wav`. Build a candidate list **excluding all entries with `src_node == src_node_query`** (source-exclusion).
- `np.random.choice(candidates, num_ref, replace=False)` with `replace=True` fallback if `len(candidates) < num_ref` (mirrors `AR_md.py:103-106`).
- For each picked context, load `src_loc`, `rec_loc` from its `S00X_R00Y.json`. Compute three vectors as **world-frame differences** (do *not* reuse `get_3d_point_camera_coord` blindly — its `lis_xyz` argument plays the role of the new frame origin; we want explicit subtraction):
  - `pair_local = src_loc - rec_loc`         →  $s_i - r_i$
  - `src_qrel   = src_loc - rec_loc_query`   →  $s_i - r_q$
  - `recs_qrel  = rec_loc - rec_loc_query`   →  $r_i - r_q$
- Load each context RIR waveform with the same pad/crop logic as the baseline (`AR_md.py:107-117`).
- Bundle the three pose vectors into one dict-valued key:

```python
md['context_fused_pose'] = {
    'pair_local': pair_local_tensor,  # [N, 3]
    'src_qrel':   src_qrel_tensor,    # [N, 3]
    'recs_qrel':  recs_qrel_tensor,   # [N, 3]
}
md['context_poses_vit'] = src_qrel_tensor   # [N, 3] — also feeds the ViT against depth at r_q
md['context_audio']     = all_ref_irs       # [N, 1, max_len]
```

Final `md` shape contract:

| key | shape | conditioner | semantic |
|---|---|---|---|
| `scene` | str | (metrics) | unchanged |
| `source` | `[3]` | `dist_embedder` (adaLN) | $s_q - r_q$, unchanged |
| `source_vit` | `[1, 3]` | `ViTCoordinates` (adaLN) | $s_q - r_q$, unchanged |
| `depth` | `[3, 256, 512]` | (depth input to ViT) | $G_{r_q}$, unchanged |
| `context_poses_vit` | `[N, 3]` | `ViTCoordinates` (cross-attn) | $s_i - r_q$, fed to ViT against $G_{r_q}$ |
| `context_fused_pose` | dict of 3× `[N, 3]` | `fused_pose` (cross-attn) | **new** geometric token, $(s_i-r_i, s_i-r_q, r_i-r_q)$ |
| `context_audio` | `[N, 1, T]` | `rir` (cross-attn) | unchanged, picked RIR waveforms |

Note: the baseline's `context_poses` (Fourier $s_i - r_q$) is **removed** in V3. Its information is preserved inside `context_fused_pose` (as the `src_qrel` component) alongside the two new vectors. Cross-attention sequence length stays at $3N$ — same as baseline.

### 3. `src/configs/model_configs/FLAC/AR/FLAC_AR_arbRIR_v0.json` — new

Clone `FLAC_AR.json`. In `model.conditioning.configs[]`:

- **Keep** `source` (dist_embedder), `source_vit` (ViTCoordinates) — query side untouched.
- **Keep** `context_poses_vit` (ViTCoordinates) — fed by the new `context_poses_vit` key (now semantically $s_i - r_q$ in the V3 module).
- **Keep** `context_audio` (rir) — unchanged.
- **Remove** `context_poses` (was Fourier $s_i - r_q$; subsumed by the fused-pose token).
- **Add** one new entry:

```json
{
  "id": "context_fused_pose",
  "type": "fused_pose",
  "config": {
    "num_freqs": 20,
    "max_freq": 10,
    "pose_max_val": 5.0
  }
}
```

In `model.diffusion`:

- `cross_attention_cond_ids: ["context_poses_vit", "context_fused_pose", "context_audio"]` (three entries, matching baseline cardinality).
- `global_cond_ids: ["source", "source_vit"]` — unchanged.
- Everything else (`type: "dit"`, `diffusion_objective: "rectified_flow"`, DiT depth/heads, `cond_dim: 256`, etc.) — unchanged.

In `training`:

- `flow_source: "gaussian"`. Same optimizer, scheduler, metrics, AGREE checkpoint.

### 4. Dataset configs — new

Clone `acousticroom_train.json`, `acousticroom_seeneval.json`, `acousticroom_unseeneval.json` (+ K=1/4/8 eval variants if needed). Change exactly one field per file:

```json
"custom_metadata_module": "src/configs/dataset_configs/custom_metadata/AR_md_arbRIR_v0.py"
```

Keep `acoustic_context.max_context = 8`, `max_len = 9600`, `modalities.depth.load = true`.

## What does NOT change

- DiT, VAE pretransform, `DiffusionCondTrainingWrapper`, EMA, `_pick_nearest_reference` (unused here), `eval_FLAC.py`, sampling code, the metric callback, AGREE FD/Recall.
- `MultiConditioner.forward` — the dict-valued metadata key flows through unchanged on the non-Geometry path.
- `src/data/dataset.py` — collator's `else: b = b` branch handles dicts already.
- `GeometryConditioner` and `RIRConditioner` — both reused as-is.
- The factory in `src/models/factory.py` and the diffusion-cond construction in `src/models/diffusion.py:312-316`.

## Verification

1. **Shape sanity (one-shot REPL)** — confirm the metadata module emits the V3 keys with the right shapes:
   ```python
   import json
   from src.data.dataset import create_dataloader_from_config
   with open("src/configs/dataset_configs/AR/train/acousticroom_train_arbRIR_v0.json") as f:
       cfg = json.load(f)
   dl = create_dataloader_from_config(cfg, batch_size=2, num_workers=2,
                                      sample_rate=22050, sample_size=10240, audio_channels=1)
   audio, md = next(iter(dl))
   for m in md:
       fp = m['context_fused_pose']
       assert isinstance(fp, dict)
       assert fp['pair_local'].shape == (8, 3)
       assert fp['src_qrel'].shape   == (8, 3)
       assert fp['recs_qrel'].shape  == (8, 3)
       assert m['context_poses_vit'].shape == (8, 3)
       assert m['context_audio'].shape     == (8, 1, 9600)
       assert m['source'].shape            == (3,)
       assert m['source_vit'].shape        == (1, 3)
       assert m['depth'].shape             == (3, 256, 512)
   ```
2. **Forward smoke test** — build the model from `FLAC_AR_arbRIR_v0.json`, run one `training_step`:
   - `cross_attention_input` after `ConditionedDiffusionModelWrapper.get_conditioning_inputs` should be `[B, 3N, cond_dim] = [2, 24, 256]` — **same length as baseline**.
   - Loss is finite; gradients propagate into `FusedPoseConditioner.proj` (~95K params for the linear).
3. **Source-exclusion test** — instrument the metadata module over ~100 samples to assert `src_node_i != src_node_q` for every drawn context.
4. **Short training run** — 5k–10k steps with `--val-every 1000 --checkpoint-every 5000` on a 24GB GPU. Compare validation loss trajectory against an identically-budgeted baseline `FLAC_AR.json` run.
5. **Eval against baseline** — run `eval_FLAC.py` on `acousticroom_seeneval_arbRIR_v0.json` (and unseen, K-variants if generated). Caveat: the arbitrary-receiver + source-exclusion eval distribution differs from the baseline eval. For a fair head-to-head, also run the baseline FLAC against the `_arbRIR_v0` eval configs (tests baseline's robustness to the new context distribution). Document which protocol is used next to the metrics.

## Critical files to inspect during implementation

- `src/models/conditioners.py:226-271` — `DistEmbedderConditioner` (Fourier-features reference).
- `src/models/conditioners.py:288-329` — `MultiConditioner.forward` (dict-passthrough on lines 317-324).
- `src/models/conditioners.py:333-431` — factory `create_multi_conditioner_from_conditioning_config` (add `fused_pose` branch after the `dist_embedder` branch).
- `src/configs/dataset_configs/custom_metadata/AR_md.py` — clone target; lines 87-123 are the sampler to rewrite.
- `src/configs/model_configs/FLAC/AR/FLAC_AR.json` — clone target.
- `src/data/dataset.py:328-341` — collator's dict-passthrough.
- `src/models/diffusion.py:139-157`, `312-316` — cross-attention assembly + factory call.
