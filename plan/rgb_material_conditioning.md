# Plan — RGB-Material Conditioning for FLAC (Context Replacement Experiment)

## Context

The context-ablation control (commit `7c48157`, results in Notion + `outputs_FLAC/arbRIR_eval_logs/RESULTS_context_ablation.md`) quantitatively demonstrated:

- **Context audio is the dominant input to baseline FLAC** (~80% of performance). Zeroing it: T60 seen 8.7→44.9 (+415%), unseen 10.3→45.7 (+344%).
- **The signal it carries is room-specific**: a wrong-room RIR is catastrophic (8.7→37.3 T60), only marginally better than silence.
- **Geometry-only → context gap is ~5×** (T60 ~45 → ~8.7), overwhelmingly room-specific material/acoustic information.

This converts the advisor hypothesis ("context RIR = material proxy") from plausible to **quantitatively demonstrated**. The natural follow-on: can we **replace** the context RIR with a different, more realistic material proxy — concretely, an **RGB observation of the room** — and recover the same gap-closing? If yes: the RGB direction is validated, the K-context dataset requirement of FLAC is relaxed, and the path to physics-aware / interpretable architecture (the group theme) opens up.

## The hypothesis chain to test

Let `f` be the learned mapping FLAC implements. We have evidence that, in the baseline:

```
f(depth, source_pos, context_RIR) ≈ f(depth, source_pos, material(room))    (H1)
```

The RGB-material experiment tests:

```
material(room) ≈ g(RGB_panorama at r_q)         for some learnable g
```

Combined: if RGB can supply the same material information, then `f(depth, src, RGB)` should land near `f(depth, src, context_RIR)` and far above `f(depth, src, ∅)`.

**Three possible outcomes** (pre-registered, mirroring the H1/H2 structure of the context-ablation):

| Outcome (seen T60 ranges anchored to the ablation: ~8.7 correct, ~45 floor) | Verdict |
|---|---|
| RGB ≈ correct (~8.7) | **Strong H1 confirmed.** RGB fully recovers the material signal. Paper-worthy result; basis for the physics-aware architecture direction. |
| zeroctx < RGB < correct (clearly between) | **Partial.** RGB recovers some but not all of the material info. Quantify the fraction; future work to improve the RGB encoder / data. |
| RGB ≈ zeroctx (~45) | **RGB-as-material-proxy fails.** Either AR's RGB cannot encode the relevant material info, or the encoder couldn't learn to extract it. H1 stays alive (material IS in context) but RGB isn't the right substitute → consider explicit material conditioning instead. |

## The data problem — RGB is not available locally (verified)

| dataset | depth panoramas | **RGB / color / texture** |
|---|---|---|
| AcousticRooms (local) | ✅ `.npy`, 256×512, 1-ch | ❌ **none** (untextured simulation meshes; FLAC paper confirms) |
| HAA (local) | ✅ rendered from surface annotations | ❌ **none** (`HAA/` contains only `.npy`: RIRs, xyz, music — 125 npy / 0 images verified by `find`) |

The "RGB-material experiment" as commonly framed is therefore **blocked at the data layer in our current setting**. Three acquisition routes, with their tradeoffs:

| Route | What it requires | Scientific strength | Engineering cost | Confound risk |
|---|---|---|---|---|
| **A. Synthetic-from-material** — acquire AR's full release (`room_mesh_obj_format/`, `material_library/`, `simulation_info/`), render an equirect RGB panorama by mapping each surface's material → pseudo-color, aligned with the existing depth | full AR release; rendering pipeline; deterministic material→color map | Medium | Medium | **High — circular**: the "RGB" literally encodes the material label, so it's essentially material-oracle dressed up as RGB. Useful as an *upper bound* on what RGB-conditioning could ever achieve on AR, not as evidence about real RGB. |
| **B. Real-photo RGB on HAA** — HAA scenes are real rooms; the original `hearinganythinganywhere` repo has RGB photos at the source positions (not in our local subset). Acquire those, finetune FLAC with RGB conditioning on HAA. | HAA RGB photos; HAA finetuning recipe (FLAC paper already supports this); a HAA RGB metadata module | High (real photos, real materials) | Medium-High | **Dataset shift confound**: HAA was originally a finetune target (small, 10 scenes); not directly comparable to the AR step=145000 numbers. |
| **C. Treble simulation** — generate a new controlled dataset with aligned RGB + depth + material + RIR | Treble SDK + scene authoring + RIR sim + RGB render | Highest (controllable material perturbations) | Highest (~weeks) | Lowest |

**Logically-prior gate (Stage 0) — material-oracle first.** All three RGB routes test "*can we extract material from a particular visual modality*". They share an upstream assumption: that conditioning on ground-truth material itself recovers the audio-context performance. If it doesn't, the RGB experiment is a wild goose chase. The cheapest decisive test is therefore the **material-oracle**:

```
condition on the actual per-room (or per-surface) material absorption vector μ
```

This needs only the public `material_library/` + scene-material assignment from AR's full release — no rendering, no RGB, no Treble. It is the **upper bound** on what any RGB-derived material proxy can achieve on AR. If the oracle reaches ~ correct (8.7 T60) → RGB pursuit is justified, plus we know the target. If the oracle plateaus closer to zeroctx (~45) → either H1 is wrong or material absorption alone is the wrong representation (e.g., spatial layout of materials matters; coefficients are insufficient; etc.). **Either result is informative; both save the RGB-implementation effort.**

## Recommended sequencing (resolved 2026-05-20)

1. **Stage 0 (material-oracle) — required gate. [USER-CONFIRMED]** Acquire AR material library + simulation info; build **both** a per-room μ̄_m vector **and** a per-surface visible μ_k (panorama-raycast); train **both** a fine-tune from step=145000 **and** a from-scratch run in parallel. **4 trained models** (2 vectors × 2 training strategies), evaluated against the same correct/zeroctx anchors. Fine-tune cells deliver in ~hours; from-scratch in ~3–5 days. **Decisive outcome on H1 with two independent training-strategy data points.**
2. **Stage 1 (synthetic-RGB on AR) — IF Stage 0 confirms H1.** Render material→pseudo-color RGB panoramas aligned with depth, replace the context RIR stream with an RGB encoder (DINOv3 ViT, mirroring the existing depth path). Provides the *upper bound* for RGB on AR (since synthetic-RGB perfectly encodes materials). ~3–5 days.
3. **Stage 2 (real-photo RGB on HAA) — DEFERRED [USER: later project, not in scope for this plan].** Tracked for completeness only; will plan separately when promoted.

This plan covers all three stages but **prioritizes Stage 0 as the immediate work**. Anyone reading should be able to start there on green-light without rereading.

---

## Stage 0 — Material-oracle (the gating experiment)

**Hypothesis to falsify.** `f(depth, src, μ̄_m) ≈ f(depth, src, context_RIR)`, i.e. conditioning on the room's mean absorption vector recovers the audio-context performance.

### Data — what we need to acquire

From `github.com/facebookresearch/AcousticRooms`:

- `material_library/` — per-material absorption (and reflection) coefficients, presumably keyed by material name. Likely small (<100 MB).
- `simulation_info/` — per-scene assignment of materials to surfaces / surface groups. Confirms which 332 materials × 11 categories were used per room.
- `room_mesh_obj_format/` — meshes with `usemtl` tags (needed if we go per-surface rather than per-room).

**Schema-verification step before any modeling:** open one `simulation.json` and the matching material entry; confirm:
1. Are absorption coefficients given as 8-band (or n-band) per octave?
2. Is there a per-face (or per-surface-group) material assignment, or only a global material list?
3. The join key — does `simulation.json` reference materials by name (matching `material_library/`), and does it reference the same scene_id we already have (`Cafe_idx_1`, etc.)?

This is verbatim the Manifest-Key-Contract pattern from `plan/eval_arbRIR_v0_vs_baseline_K1_K8.md` — answer it first, no loader code until it's pinned.

### Material vector design (two flavors)

| Vector | Shape | Interpretation |
|---|---|---|
| **μ̄_m** (room-level, area-weighted mean) | `[8]` (8 octave bands) | Single global token; the cheapest oracle |
| **μ_k** (visible-surface, panorama-aligned) | `[K_pano, 8]` | One per panorama patch via mesh raycast from r_q (matches the advisor's proposal #2) |

**Per user direction (2026-05-20): Stage 0 runs BOTH vectors as primary deliverables**, not sequentially. μ̄_m isolates "is global mean absorption enough"; μ_k tests whether the spatial/visible distribution of materials matters (the advisor's proposal #2). They are independent models with different conditioner topologies and are evaluated side-by-side.

### Architecture — minimal-delta change (TWO new conditioner classes)

Conditioner pattern is already established (V3 added `fused_pose` as a new `type`). Stage 0 adds two:

- **`MaterialAbsorptionConditioner`** for μ̄_m (≈ 30 LoC; cloned from `DistEmbedderConditioner` since it's a fixed-dim vector → MLP → fixed-dim token). Plugged in as an **AdaLN global token** (`global_cond_ids`), same path as `source`.
- **`MaterialVisibleConditioner`** for μ_k (≈ 50 LoC; takes `[B, K_pano, 8]` → per-panorama-patch MLP → `[B, K_pano, cond_dim]`). Plugged in as a **cross-attention token sequence** (`cross_attention_cond_ids`), same path as `context_poses_vit`. `K_pano` set initially to 64 (8×8 grid over the panorama) — tunable.

Both share the conditioner-factory branch pattern; both `type: "material_absorption_global"` / `"material_absorption_visible"` registered in `create_multi_conditioner_from_conditioning_config`.

**Model configs (two new):**
- `FLAC_AR_material_global.json` — clones `FLAC_AR.json`, drops `context_audio` (replace, not add — the cleanest test of "does material *replace* the context-audio role?"), adds `material_global` to `global_cond_ids`.
- `FLAC_AR_material_visible.json` — clones `FLAC_AR.json`, drops `context_audio`, adds `material_visible` to `cross_attention_cond_ids`.

**Optional `+ctx_audio` additivity variants** (cheap to add on top): same configs but keep `context_audio` too. Run only on the fine-tune track if budget allows; clarifies whether material ADDS over context or merely REPLACES it.

### Training strategy — fine-tune AND from-scratch in parallel (user-directed)

Per user direction (2026-05-20), run **both** strategies in parallel, on **both** vectors → **4 trained models** total:

1. **Fine-tune from `epoch=15-step=145000.ckpt`** — new conditioner heads + AdaLN/cross-attn injection params randomly initialized; everything else loaded with `strict=False`. 5k–10k steps, same batch/accum/LR as baseline. Cost: ~few hours/A6000 per model. **Lands first**; isolates "does adding this signal on top of an existing baseline help?".
2. **From-scratch training** — fresh init, full training budget (target step≥145k to match the existing baseline apples-to-apples). Cost: ~3–5 days/A6000 per model. **Lands ~3–5 days later**; tests "does this signal help when the model can learn to use it from epoch 0?".

Trained models on disk:
- `outputs_FLAC/FLAC_material_global_ft/...`
- `outputs_FLAC/FLAC_material_global_scratch/...`
- `outputs_FLAC/FLAC_material_visible_ft/...`
- `outputs_FLAC/FLAC_material_visible_scratch/...`

**Operational reality — GPU contention (decision point at launch):** GPUs 0 and 1 are currently fully occupied by the existing ablation + baseline training runs (`max_steps=1000000`; effectively never finish). Launching 2 new from-scratch runs in parallel requires either (a) sharing GPUs with the existing training (documented slowdown — both training streams continue but ~halved throughput), (b) stopping one or both of the existing runs (they're well past step=145000 already; little marginal value from continuing), or (c) acquiring more GPU capacity. **Flag at execution time; don't pre-commit.** Fine-tune cells (~hours) can run sequentially on a single shared GPU without significant disruption.

### Eval matrix — extends the 3-way control

Same `seen` and `unseen`, K=1 only (matches the 3-way control we already have). 5 cells per split = **10 new evals** (and we reuse `correct`/`wrongroom`/`zeroctx` from the control):

Reference anchors (already in the matrix from the context-ablation control):
- `correct` (depth + src + ctx_audio, K=1): seen T60 8.73, unseen 10.29
- `zeroctx` (depth + src, no ctx): seen T60 44.94, unseen 45.69 (≈ geometry-only floor)

New cells (2 vectors × 2 training strategies × 2 splits = **8 evals minimum**, plus ~4 optional additivity cells):

| Cell | training | conditioning | eval-name suffix |
|---|---|---|---|
| `material_global / ft` | fine-tune from 145k | depth + src + μ̄_m (no ctx_audio) | `mat_global_ft_{seen,unseen}_K1` |
| `material_global / scratch` | from-scratch | depth + src + μ̄_m (no ctx_audio) | `mat_global_scratch_{seen,unseen}_K1` |
| `material_visible / ft` | fine-tune from 145k | depth + src + μ_k panorama tokens (no ctx_audio) | `mat_visible_ft_{seen,unseen}_K1` |
| `material_visible / scratch` | from-scratch | depth + src + μ_k panorama tokens (no ctx_audio) | `mat_visible_scratch_{seen,unseen}_K1` |
| (optional) `mat_global + ctx_audio / ft` | fine-tune | depth + src + μ̄_m + ctx_audio | additive sanity |
| (optional) `mat_visible + ctx_audio / ft` | fine-tune | depth + src + μ_k + ctx_audio | additive sanity |

**Same checkpoint-step locking rule as before** (`plan/eval_arbRIR_v0_vs_baseline_K1_K8.md`): fine-tune cells eval'd at the highest fine-tune step (e.g. 145k+10k = 155k); from-scratch cells eval'd at the highest step common to both new from-scratch runs (re-pick at execution time). All cells reuse `eval_FLAC.py` with `per_scene=True` patch already in place.

### Verification gates (before any training)

1. **Schema gate**: confirm absorption coefficient shape (8-band assumed; check), scene→material join key matches our existing `scene_id`, material name uniqueness across rooms.
2. **Frozen material manifests** (mirror of wrong-room manifest pattern):
   - `data/AR/material_global_manifest.json` — `{scene_id: μ̄_m_list[8]}` (room-level area-weighted mean).
   - `data/AR/material_visible_manifest_{seen,unseen}.json` — `{query_key: μ_k_list[K_pano, 8]}` (per receiver position; pre-rendered raycast hits).
   Both idempotent (seed=42 if any RNG involved); checked in.
3. **Sanity**: μ̄_m differs across rooms (not constant); μ_k spatial variance is non-degenerate (multiple distinct materials visible per panorama).
4. **Conditioner shape gate**: forward pass on one batch for each conditioner — token shapes match, no NaN, gradients flow.
5. **Validity gate (mirror of context-ablation)**: a smoke fine-tune for ~100 steps with material=zeros vs material=real produces different loss trajectories → the model actually attends to the new signal. Run **for each of the 4 trained models**.

### Pre-registered decision rules (Stage 0)

Let `T60_X` be the T60 (per-scene mean, seen) of model X. Anchors: correct=8.73, zeroctx=44.94 → gap to close = 36.2 T60 points.

Per-model classification:
- `T60_X ∈ [8.73, 12]` (≥90% gap closure) → **strong**.
- `T60_X ∈ [12, 25]` (50–90% closure) → **partial**.
- `T60_X ∈ [25, 45]` (modest closure) → **weak**.
- `T60_X ≥ 45` → **null / bug**.

Same buckets apply per-metric for C50/EDT/FD/retrieval.

**With 4 trained models we get a 2×2 grid (vector × training-strategy).** Cross-model patterns and their implications:

| Pattern | Implication |
|---|---|
| All 4 strong | **H1 fully confirmed**, RGB Stage 1 green-lit on the strongest variant. |
| `_scratch` strong, `_ft` partial/weak (both vectors) | Existing baseline params can't adapt to the new signal; **fine-tune is the wrong strategy** for Stage 1 RGB too — plan to train RGB from scratch only. |
| `_ft` strong, `_scratch` strong-but-slower | Fine-tune adaptation works; from-scratch is more expensive than necessary. Stage 1 RGB can fine-tune. |
| `visible_*` strictly stronger than `global_*` | Spatial layout of materials matters; Stage 1 RGB should keep panorama spatial structure (no global pooling). |
| `global_*` ≈ `visible_*` | 8-band global μ is sufficient; Stage 1 RGB can collapse to a global token. |
| All 4 partial / weak | **H1 weakened**: material absorption alone isn't the dominant signal. Diagnose before any RGB work — possible candidates: spatial impulse-response structure that material can't capture, source-receiver acoustic path info, etc. |
| `*_ft` ≈ correct **AND** `*_scratch` ≪ correct | Suspect: fine-tune is leaking via the inherited model state (i.e., the model already learned to use context, and the new material conditioner is just a coincidental shortcut). Run the validity gate's zero-material trajectory check more carefully. |

---

## Stage 1 — Synthetic RGB on AR (only if Stage 0 green-lights it)

**Why this is included only conditionally:** AR's meshes are untextured; "RGB" here is fabricated from material labels. Useful as an *upper bound* for what an RGB-derived material proxy can do on AR (since the synthetic RGB perfectly encodes material), **not** as evidence that real-world RGB can. Be honest about this when writing up.

### Data

Render an equirect RGB panorama at each receiver position by raycasting against the mesh and coloring hits by the assigned material's pseudo-color (a deterministic name→RGB table). Same `H=256, W=512` to align with depth. One `.png` (or `.npy`) per receiver-position, mirroring `depth_map/<scene>/<scene_id>/<rec_node>.npy`.

### Architecture

- **New conditioner**: `RGBPanoramaConditioner` — cloned from `ViTCoordinates` (which already wraps DINOv3-S/16; `src/models/conditioners.py:423`). 3 input channels, same 256×512 geometry, optional `freeze=True` initially.
- New config `FLAC_AR_rgb_replace_audio.json`: in `cross_attention_cond_ids`, swap `context_audio` for `context_rgb`. Single global RGB panorama → single cross-attn token (or a small grid of tokens if the ViT outputs spatial features).

### Eval matrix

Same as Stage 0 plus `rgb` and `rgb + zeroctx` cells. Compare:

- `rgb` vs `correct` (does RGB recover audio-context performance?)
- `rgb` vs `material_global` (does spatially-resolved RGB do better than a single global vector?)
- `rgb` vs `zeroctx` (basic "RGB helps over nothing")

### Pre-registered decision (Stage 1)

Stage 1 is an **upper bound** for AR-RGB. If it fails (rgb ≈ zeroctx), real RGB on AR is hopeless and Stage 2 (HAA) becomes the only path. If it succeeds (rgb ≈ correct), Stage 2 tests whether *real* photos rather than material-derived synthetic RGB can replicate the result.

---

## Stage 2 — Real-photo RGB on HAA (only after Stages 0+1)

Use HAA's real photos (acquire from `hearinganythinganywhere`) as RGB conditioning, finetune FLAC from the AR baseline. The convention difference (HAA depth at source, AR depth at receiver — CLAUDE.md `HAA_md.py:70`) needs matching for RGB. Smaller, real-world test; cross-dataset comparison. Detailed plan deferred until Stage 1 lands.

---

## Architecture decisions (apply to all stages)

- **Run fine-tune AND from-scratch in parallel** (user-directed) — both serve as training-strategy controls. Fine-tune lands first.
- **Always keep the cross-attention sequence length comparable** to the baseline's K=1 (one material token / one RGB token ≈ one audio token), so we are not testing "more tokens = better".
- **Always run the existing 3-way control (`correct`/`wrongroom`/`zeroctx`) on every new finetuned model after training** to confirm the new conditioner hasn't broken the established baseline behavior (sanity check).

## Artifacts to create (concrete file list)

Stage 0 — code & data:

- `tools/acquire_ar_full_release.sh` — wraps the AR repo's download for `material_library/`, `simulation_info/`, `room_mesh_obj_format/`.
- `tools/build_material_global_manifest.py` → `data/AR/material_global_manifest.json` (per `scene_id` → 8-band μ̄_m).
- `tools/build_material_visible_manifest.py` → `data/AR/material_visible_manifest_{seen,unseen}.json` (per query scene-triple → `[K_pano, 8]` from mesh raycast at r_q against the AR mesh; reuses the panorama coordinate frame used by `depth_map/`).
- `src/models/conditioners.py`:
  - `MaterialAbsorptionConditioner` (cloned from `DistEmbedderConditioner`) — `type: "material_absorption_global"`, AdaLN global token.
  - `MaterialVisibleConditioner` (≈50 LoC, new pattern) — `type: "material_absorption_visible"`, cross-attn tokens.
  - Two factory branches.
- `src/configs/model_configs/FLAC/AR/FLAC_AR_material_global.json`, `FLAC_AR_material_visible.json` — two new model configs (drops `context_audio`).
- `src/configs/dataset_configs/custom_metadata/AR_md_material_{global,visible}.py` — two new metadata modules emitting `md['material_global']` or `md['material_visible']` from the frozen manifest.
- Eval-side configs (for the new material trained models): `acousticroom_{seen,unseen}eval_material_{global,visible}_1.json` — 4 dataset configs.

Stage 0 — training & eval drivers:

- `tools/launch_material_finetune.sh` (4 finetune runs: global+ft, global+ft+ctx_audio additivity (optional), visible+ft, visible+ft+ctx_audio additivity (optional)).
- `tools/launch_material_scratch.sh` (2 from-scratch runs: global+scratch, visible+scratch).
- `tools/verify_material_oracle.py` — Verification gates 1–5 (schema + manifests + sanity + conditioner shape + zero-vs-real loss-trajectory validity).
- `tools/run_material_oracle_eval.sh` + `tools/tabulate_material_oracle.py` — eval all 4 models on seen+unseen K=1, GFM-valid + auto-emit the 2×2 grid + per-row classification per the decision rules above.

Stage 1 (template only; finalized after Stage 0 result):

- `tools/render_synthetic_rgb_pano.py` (mesh raycast + material→color mapping; same panorama coord frame as `depth_map/`).
- `AcousticRooms/rgb_pano/<scene>/<scene_id>/<rec_node>.png` (or `.npy`).
- `src/models/conditioners.py` — `RGBPanoramaConditioner` (clone of `ViTCoordinates`).
- `FLAC_AR_rgb_replace_audio.json` (and `_scratch` variant if Stage 0 says from-scratch is required).

## Risks & confounds (must address in the writeup)

1. **Synthetic-RGB ≡ material-oracle** in information content if rendered from material labels — flag this honestly; Stage 1 alone cannot claim "real RGB works".
2. **Mid-training caveat persists**: baseline at step=145000 was ~0.4–0.5 loss; finetune-from-145k inherits this. Document it on every result.
3. **Material library coverage**: AR's 332 materials × 11 categories — confirm coverage of all eval rooms; missing-material handling fail-loud.
4. **Receiver vs source convention drift across stages**: AR depth is at receiver, HAA at source. RGB rendering must match the dataset's convention or the model will see misaligned RGB+depth.
5. **DiT capacity vs new signal**: the existing DiT was trained without material conditioning. A small fine-tune may underfit the new signal; a long fine-tune may distort the audio-context path. Watch the `correct`-cell metric on the finetuned model — if it degrades materially, the new signal is interfering rather than helping.

## Critical files to inspect

- `src/models/conditioners.py:226-271` (`DistEmbedderConditioner`) — template for `MaterialAbsorptionConditioner`.
- `src/models/conditioners.py:423` (ViT branch in factory) — template for RGB encoder.
- `src/training/diffusion.py` — to confirm fine-tune from `--ckpt-path` works with new conditioning ids.
- `eval_FLAC.py:91` — already patched for `per_scene=True`; new evals inherit.
- `RESULTS_context_ablation.md` — the anchor numbers every Stage-0 cell is compared against.

## Resolved decisions (user-confirmed 2026-05-20)

1. ✅ **Stage 0 first** — material-oracle as the gating experiment.
2. ✅ **AR full release acquisition** — proceed; storage budget approved.
3. ⏸ **HAA Stage 2 deferred** — not in scope for this plan execution; tracked as a later project.
4. ✅ **Fine-tune AND from-scratch in parallel** — both as primary deliverables. 4 trained models total (2 vectors × 2 training strategies).
5. ✅ **Both material granularities (μ̄_m and μ_k)** — parallel primary experiments, not sequential.

**Plan is approved for execution.** Next concrete step on green-light: acquire AR's full release (`room_mesh_obj_format/`, `material_library/`, `simulation_info/`) and run the schema verification gate before any modeling. Then in order: build the two manifests → verify gates 1–5 → launch the 4 trainings (fine-tunes can start immediately and share GPU; from-scratch runs need a GPU decision: share/stop one of the existing training runs/acquire more capacity).
