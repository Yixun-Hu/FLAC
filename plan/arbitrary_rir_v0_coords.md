# Ablation Plan — Arbitrary-RIR Context with v0 Coordinate Formulation (strict-fusion)

## Context

The current FLAC AR pipeline restricts context RIRs to those sharing **the same receiver** as the query (`src/configs/dataset_configs/custom_metadata/AR_md.py:98`). Each context's geometry is encoded as a single 3-vector — implicitly $s_i^g - r_i^g$ (`AR_md.py:122`), which equals $s_i^g - r_q^g$ only because $r_i = r_q$ by construction.

The v0 ablation lifts that restriction: each context RIR may be drawn from an **arbitrary $(s_i, r_i)$ pair** in the scene (subject to source exclusion below). Each context becomes a *single* fused token

$$c_i = \phi_{\mathrm{ctx}}\bigl(E(h_i),\; s_i^g - r_i^g,\; s_i^g - r_q^g,\; r_i^g - r_q^g\bigr) \in \mathbb{R}^{\text{cond\_dim}}$$

where $E$ is an STFT+ResNet18 audio encoder and the three pose vectors live in the **query-receiver-centered frame**. The query token is unchanged: $q = \phi_q(s_q - r_q,\; G_{r_q})$, fed to adaLN exactly as in baseline FLAC.

Goal: a side-by-side ablation against `FLAC_AR.json` — same DiT, same VAE, same training recipe; only the context-side conditioning is replaced.

## Resolved decisions

1. **Strict-fusion**: one fused token per context via a new `ContextFusionConditioner` class. This *replaces* the three baseline context streams (`context_poses_vit`, `context_poses`, `context_audio`) with a single stream of `N` tokens. Cross-attention sequence length drops from `3N → N` (for K=8: 24 → 8 tokens).
2. **Source-exclusion**: context pool = all $(s_i, r_i)$ in the scene where $s_i \neq s_q$. Pairs that share the query receiver but use a different source are allowed (they degenerate to $r_i - r_q = 0$); pairs that share the query source are forbidden. This probes stronger cross-source generalization.
3. **Arbitrary sampling at eval too**: the same `AR_md_arbRIR_v0.py` is used by both train and eval dataset configs; no separate "same-receiver-only" eval pool.

## Architectural fit

- **Conditioner factory** (`src/models/conditioners.py:333-431`) dispatches by `type` string. `output_dim` is auto-injected from the global `model.conditioning.cond_dim`. Adding a new `"context_fusion"` branch is a 2-line edit.
- **Reusable pieces** in the same file: `AudioResNet18` (lines 19-64, `forward(spec[B,C,F,T]) -> [B, 512]`), the Fourier-features pattern from `DistEmbedderConditioner` (lines 250-271), and the STFT+magnitude pattern from `RIRConditioner` (lines 149-168).
- **Metadata routing**: `MultiConditioner.forward` (`conditioners.py:288-329`) passes dict-typed metadata values through unchanged on the non-Geometry path (lines 317-322, since `isinstance(dict, (list, tuple))` is False). The dataloader collator (`src/data/dataset.py:328-341`) also leaves dicts alone (line 339 `else: b = b`). **Therefore the metadata module can pre-bundle audio + 3 pose vectors into a single dict-valued key, and no `MultiConditioner.forward` patch is required.**
- **Token sequence assembly** (`src/models/diffusion.py:139-157`) concatenates whatever cross-attention IDs are listed; reducing to one ID is supported.

## Scope (delta vs. baseline)

| Purpose | Path | Action |
|---|---|---|
| New conditioner class | `src/models/conditioners.py` | **Add** `ContextFusionConditioner` class (~50 lines, before `MultiConditioner`) and one `elif` branch in `create_multi_conditioner_from_conditioning_config` (~3 lines, after the `dist_embedder` branch at line 411) |
| Metadata module | `src/configs/dataset_configs/custom_metadata/AR_md_arbRIR_v0.py` | **New** |
| Model config | `src/configs/model_configs/FLAC/AR/FLAC_AR_arbRIR_v0.json` | **New** |
| Train dataset config | `src/configs/dataset_configs/AR/train/acousticroom_train_arbRIR_v0.json` | **New** (one-line change vs. baseline: `custom_metadata_module` pointer) |
| Eval dataset configs | `acousticroom_seeneval_arbRIR_v0.json`, `acousticroom_unseeneval_arbRIR_v0.json` (+ optional `_1`/`_4`/`_8` K-variants) | **New** |

Baseline files untouched: `AR_md.py`, `FLAC_AR.json`, the existing dataset configs. No changes to `src/training/*`, `src/data/*`, `src/inference/*`, `train.py`, `eval_FLAC.py`, the VAE, or AGREE.

## File-by-file changes

### 1. `src/models/conditioners.py` — new class + factory branch

**`ContextFusionConditioner`** — subclass of `Conditioner`. Constructor signature:

```python
ContextFusionConditioner(
    output_dim: int,          # auto-injected by factory (= cond_dim, 256)
    in_channels: int = 1,
    n_fft: int = 124,
    win_length: int = 31,
    hop_length: int = 62,     # STFT params (match RIRConditioner defaults from FLAC_AR.json)
    num_freqs: int = 20,
    max_freq: int = 10,       # Fourier features for poses (match DistEmbedderConditioner)
    pose_max_val: float = 5.0,
    fusion_hidden_dim: int = 512,
    name: str = "ContextFusionConditioner",
)
```

Internals:
- `self.audio_net = AudioResNet18(in_channels)` (reuse, conditioners.py:19-64) — outputs 512-d per RIR.
- `self.stft = torchaudio.transforms.Spectrogram(n_fft, win_length, hop_length, power=None)` (same as `RIRConditioner`).
- `self.register_buffer("freqs", 2.0**torch.linspace(0, max_freq, num_freqs))` — replicates the Fourier-frequency table from `DistEmbedderConditioner:243`. Per-pose Fourier dim = `3 * (1 + 2 * num_freqs)` (raw xyz + sin/cos pairs) = 123 at the defaults.
- `self.proj = nn.Linear(512 + 3*123, output_dim)` (single linear; can swap to a small MLP if it helps).

`forward(batch_metadata: List[Dict], device) -> Tuple[Tensor, Tensor]`:
- Each entry is a dict `{'audio': [N,1,T], 'pair_local': [N,3], 'src_qrel': [N,3], 'recs_qrel': [N,3]}`.
- Stack across batch → `audios [B,N,1,T]`, three pose tensors `[B,N,3]`.
- `audios = audios.view(B*N, 1, T)` → STFT → magnitude → `audio_feat = audio_net(spec)` → `[B*N, 512]`.
- For each pose tensor: `p = p.view(B*N, 3) / pose_max_val`; `feat = cat([p, sin(p[…, None] * freqs).flatten(-2), cos(p[…, None] * freqs).flatten(-2)], dim=-1)` → `[B*N, 123]`. Concat the three → `[B*N, 369]`.
- `fused = cat([audio_feat, pose_feat], dim=-1)` → `[B*N, 881]` → `proj` → `[B*N, output_dim]`.
- `out = fused.view(B, N, output_dim)`; mask = `torch.ones(B, 1, device=device)`.
- Return `[out, mask]`.

**Factory branch** — insert after line 411 in `create_multi_conditioner_from_conditioning_config`:

```python
elif conditioner_type == "context_fusion":
    conditioners[id] = ContextFusionConditioner(**conditioner_config)
```

`conditioner_config` already has `output_dim` injected (lines 354-355); the rest comes from the JSON `config` block.

### 2. `src/configs/dataset_configs/custom_metadata/AR_md_arbRIR_v0.py` — new

Clone `AR_md.py` for the query side (`source`, `source_vit`, `depth`, `scene` — unchanged). Replace the context sampler with arbitrary-pair selection:

- Get query's `src_loc_query`, `rec_loc_query` from the same `get_receiver_source_location` (AR_md.py:77-88). Parse `src_node_query`, `rec_node_query` from the filename.
- Enumerate the scene-id directory once: `os.listdir(dir_name)` → list of `S00X_R00Y_hybrid_IR.wav`. Build a candidate list of `(src_node, rec_node, path)` triples **excluding all entries with `src_node == src_node_query`** (the source-exclusion decision).
- `np.random.choice(candidates, num_ref, replace=False)` with `replace=True` fallback if `len(candidates) < num_ref` (mirroring the existing fallback at `AR_md.py:103-106`).
- For each picked context, load `src_loc`, `rec_loc` from its `S00X_R00Y.json`. Compute three vectors as world-frame differences (do *not* reuse `get_3d_point_camera_coord` here; it conflates "translate by negative of arg" with "express in camera frame"):
  - `src_qrel  = src_loc - rec_loc_query`   →  $s_i - r_q$
  - `pair_local = src_loc - rec_loc`         →  $s_i - r_i$
  - `recs_qrel = rec_loc - rec_loc_query`   →  $r_i - r_q$
- Load each context RIR waveform with the same pad/crop logic as the baseline (`AR_md.py:107-117`).
- Bundle into one metadata key:

```python
md['context_token'] = {
    'audio':      all_ref_irs,        # [N, 1, max_len]
    'pair_local': pair_local_tensor,  # [N, 3]
    'src_qrel':   src_qrel_tensor,    # [N, 3]
    'recs_qrel':  recs_qrel_tensor,   # [N, 3]
}
```

Final `md` shape contract:
| key | shape | notes |
|---|---|---|
| `scene` | str | unchanged |
| `source` | `[3]` | unchanged, $s_q - r_q$ (for adaLN) |
| `source_vit` | `[1, 3]` | unchanged (for ViT adaLN) |
| `depth` | `[3, 256, 512]` | unchanged, $G_{r_q}$ |
| `context_token` | dict (above) | **new bundled key** |

### 3. `src/configs/model_configs/FLAC/AR/FLAC_AR_arbRIR_v0.json` — new

Clone `FLAC_AR.json`. In `model.conditioning.configs[]`:
- **Keep** `source` (dist_embedder) and `source_vit` (ViTCoordinates) — query side untouched.
- **Remove** `context_poses_vit`, `context_poses`, `context_audio`.
- **Add** one entry:

```json
{
  "id": "context_token",
  "type": "context_fusion",
  "config": {
    "in_channels": 1,
    "n_fft": 124, "win_length": 31, "hop_length": 62,
    "num_freqs": 20, "max_freq": 10,
    "pose_max_val": 5.0,
    "fusion_hidden_dim": 512
  }
}
```

In `model.diffusion`:
- `cross_attention_cond_ids: ["context_token"]` (was three entries).
- `global_cond_ids: ["source", "source_vit"]` — unchanged.
- Everything else (`type: "dit"`, `diffusion_objective: "rectified_flow"`, DiT depth/heads, etc.) — unchanged.

In `training`:
- `flow_source: "gaussian"` — unchanged. Same optimizer, scheduler, metrics, AGREE checkpoint.

### 4. Dataset configs — new

Clone `acousticroom_train.json`, `acousticroom_seeneval.json`, `acousticroom_unseeneval.json` (+ the K=1/4/8 eval variants if needed). Change exactly one field per file:

```json
"custom_metadata_module": "src/configs/dataset_configs/custom_metadata/AR_md_arbRIR_v0.py"
```

Keep `acoustic_context.max_context = 8`, `max_len = 9600`, `modalities.depth.load = true`.

## What does NOT change

- DiT, VAE pretransform, `DiffusionCondTrainingWrapper`, EMA, `_pick_nearest_reference` (unused here), `eval_FLAC.py`, sampling code, the metric callback, AGREE FD/Recall.
- `MultiConditioner.forward` — the dict-valued metadata key flows through unchanged.
- `src/data/dataset.py` — collator's `else: b = b` branch handles dicts already.
- The factory in `src/models/factory.py` and the diffusion-cond construction in `src/models/diffusion.py:312-316`.

## Verification

1. **Shape sanity (one-shot REPL)** — confirm the bundled dict survives the dataloader and contains the right shapes:
   ```python
   import json
   from src.data.dataset import create_dataloader_from_config
   with open("src/configs/dataset_configs/AR/train/acousticroom_train_arbRIR_v0.json") as f:
       cfg = json.load(f)
   dl = create_dataloader_from_config(cfg, batch_size=2, num_workers=0,
                                      sample_rate=22050, sample_size=10240, audio_channels=1)
   audio, md = next(iter(dl))
   for m in md:
       t = m['context_token']
       assert isinstance(t, dict)
       assert t['audio'].shape      == (8, 1, 9600)
       assert t['pair_local'].shape == (8, 3)
       assert t['src_qrel'].shape   == (8, 3)
       assert t['recs_qrel'].shape  == (8, 3)
       assert m['depth'].shape      == (3, 256, 512)
       # source-exclusion invariant: no context shares the query source
       # (verify via separate path: enumerate src_node from candidates' filenames)
   ```
2. **Forward smoke test** — build the model from `FLAC_AR_arbRIR_v0.json`, run one `training_step`:
   - `cross_attention_input` after `ConditionedDiffusionModelWrapper.get_conditioning_inputs` should be `[B, N, cond_dim] = [2, 8, 256]` (single stream, N tokens — *not* 3N).
   - Loss is finite; gradients propagate into `ContextFusionConditioner.audio_net`, `proj`, and (no learnable freqs).
3. **Source-exclusion test** — instrument the metadata module (or run a one-shot sanity check) over ~100 samples to assert `src_node_i != src_node_q` for every drawn context.
4. **Short training run** — 5k–10k steps with `--val-every 1000 --checkpoint-every 5000` on a 24GB GPU. Compare validation loss trajectory against an identically-budgeted baseline `FLAC_AR.json` run.
5. **Eval against baseline** — run `eval_FLAC.py` on `acousticroom_seeneval_arbRIR_v0.json` (and unseen, K-variants if generated). Caveat: the arbitrary-receiver + source-exclusion eval distribution differs from the baseline eval. For a fair head-to-head, also run the baseline FLAC against the `_arbRIR_v0` eval configs (tests baseline's robustness to the new context distribution). Document which protocol is used next to the metrics.

## Critical files to inspect during implementation

- `src/models/conditioners.py:19-64` — `AudioResNet18` (reuse target).
- `src/models/conditioners.py:136-174` — `RIRConditioner` (STFT pattern reference).
- `src/models/conditioners.py:226-271` — `DistEmbedderConditioner` (Fourier-features reference).
- `src/models/conditioners.py:288-329` — `MultiConditioner.forward` (dict-passthrough on lines 317-324).
- `src/models/conditioners.py:333-431` — factory `create_multi_conditioner_from_conditioning_config` (add `context_fusion` branch after line 411).
- `src/configs/dataset_configs/custom_metadata/AR_md.py` — clone target; lines 77-123 are the sampler to rewrite.
- `src/configs/model_configs/FLAC/AR/FLAC_AR.json` — clone target.
- `src/data/dataset.py:328-341` — confirm collator's dict-passthrough.
- `src/models/diffusion.py:139-157`, `312-316` — confirm single-id cross-attention assembly + factory call.
