# plan_bf_fa_cartesian.md — exp_21: Cartesian frame-averaged B-F (full C4 orbit, no cylindrical features)

**Author:** Claude Fable 5 (Planner seat) · 2026-08-21
**Revision:** Rev 3 — addresses all 5 blockers + nits of the round-2 review (`bf_fa_cartesian_codex_plan_review_r2.md`, verdict REVISE; method core, `only_ids` support, T2 validity and cap-32 arithmetic all CONFIRMED there). Rev 2 superseded in place. Round-0 record: `plan_bf_target_frame.md` + `bf_target_frame_codex_plan_review.md` (canonicalization reading, rejected by Yixun's Query-2 redirect).
**Status:** awaiting Yixun approval (+ D5/D6 below). NO implementation before sign-off.
**Base commit at planning time:** `6ac7b8a` on `exp17-yawaug-scratch` (A6000 box).

---

## 1. The arm (Yixun's definition, Query 2 — verbatim contract)

> Keep C4 frame averaging and remove cylindrical_pose_features. For every C4 frame, jointly rotate depth and all four pose keys; feed the rotated source/context_poses into the unchanged DistEmbedder as Cartesian xyz, and frame-average all four pose/geometry conditioner outputs. Keep context_audio single-pass. Do not use target-aligned −φ_t canonicalization or a single ViT pass.

vs the registered B-F (`fa_invariant`), exactly ONE mechanism changes: the **pose branch's symmetrization scheme** — B-F's intrinsically-invariant cylindrical triplets through the out-of-domain Cartesian `/5` embedder (meters/radians mixed, ±π discontinuity) become **C4-frame-averaged in-domain Cartesian embeddings**. The ViT branch is structurally unchanged (r2 review §b confirms the code paths guarantee it: the cylindrical step never touched `*_vit`/`depth`, GeometryConditioner reads only its own `*_vit` pose + depth, and the extra DistEmbedder orbit calls consume no RNG) — pinned numerically by test T2. Clean single-mechanism test of the representation-defect hypothesis.

**Invariance class:** ideal exact invariance on the **C4 subgroup only**, all conditioning branches; numerically allclose in float32. **45° is a genuine negative control** (expected to BREAK, and *required* to break past the C4 tolerance — §5). No reference azimuth exists; round-0 findings 2/4 are moot (r2 confirms, with 4's depth-fail-closed clause carried).

**Hypothesis:** replacing the pose representation closes part of B-F@40k's gap to P1@40k while keeping the C4-flat orbit. If not, the remaining prime suspect (per Yixun) is premature C4 averaging in the ViT branch — a later, separate arm.

Out of scope: Cyl-PE arm, canonicalization, HAA finetuning (`finetune_cond.py:35/76/469` intentionally NOT extended; must keep rejecting `fa_cartesian` until the HAA round), launch timing beyond D3 (co-tenant — Yixun, 13:45). `eval_pl.py` inherits the wrapper dispatch but lacks registered-eval provenance: **forbidden for headline rows** (r2 nit).

## 2. Method definition (exact)

```
base   = conditioner(metadata, device)              # RAW Cartesian metadata; ALL ids incl. context_audio (once)
orbit  = POSE_KEYS = ("source", "source_vit", "context_poses", "context_poses_vit")
for g in angles[1:]:                                # angles = training.frame_avg_angles = [0, 90, 180, 270]
    rotated_g = rotate_scene_metadata(md, radians(g), img_w, pose_keys=orbit)   # depth + ALL FOUR pose keys, jointly
    part_g    = conditioner(rotated_g, device, only_ids=orbit)
out[id] = (base[id] + Σ_g part_g[id]) / len(angles)  for id in orbit      # global ids (source, source_vit) and
out[context_audio] = base[context_audio]             #   cross-attn ids averaged identically (r2 §a); masks from base
```

- **Reuses the audited orbit machinery wholesale:** `rotate_scene_metadata`, `_rotated_variants`, `_orbit_average_batched` (+ `_orbit_average_loop` as equivalence reference) are generic over the id set (r2 §a: `MultiConditioner.only_ids` filters by id, conditioners.py:367/374; the executor accumulates every `present[i][0]` identically, yaw_rotation.py:547/568; base-mask retention is correct — DistEmbedder masks are constant all-ones and the DiT doesn't consume cross-attn masks, but T-mask pins it anyway). `cylindrical_pose_features`, `invariant_conditioning`: **zero edits**.
- **Angle 0 = base pass**; accumulation order/discipline identical to the existing executor.
- **Depth REQUIRED, fail-closed** (missing depth would silently degrade the orbit → error, not fallback); per-sample img_w from each sample's `depth.shape[-1]`, cross-sample mismatch raises.
- **Chunk plans — TRAINING and EVAL are separate declarations (r2 blocker 3):**
  - **Training:** micro-batch 32/rank, `training.frame_avg_max_fwd_samples: 32` → `angles_per_chunk = max(1, 32//32) = 1` → three separate non-zero-angle conditioner calls + the base call, matching B-F's legacy per-angle loop exactly (r2 §c: arithmetic confirmed; `drop_last=True` on the AR train loader means no tail-batch partition change; precisely, the train-mode DINOv3 RoPE draw is per Geometry/DINO coordinate call — at K=8, 4 × (1 source + 8 context) = 36 DINO calls/draws per conditioning step under both legacy and cap-32). **D5 recommendation: cap 32** for draw-schedule parity with the comparator; default 64 (2 angles/chunk) is the exp_17-era protocol but a disclosed schedule difference vs B-F. Verify B-F's training-era execution from the exp_07 record at parity-audit time.
  - **Eval:** batch 64 requires cap ≥ 64 (`_orbit_average_batched` raises when batch > cap, yaw_rotation.py:555): every eval invocation passes `--frame-avg-max-fwd-samples 64` explicitly, and the admission validator requires it. Eval mode has no RoPE draw, so eval numbers are cap-independent; the flag is pinned for protocol identity, not numerics.

**Names:** experiment `bf_fa_cartesian`; `cond_method: "fa_cartesian"`; config `FLAC_AR_BFC.json` (Yixun vetoes at approval if he prefers other strings).

## 3. Files to change (planned code, per file)

Complete cond_method inventory (r2 nit: production path is complete once the helper imports land at `diffusion.py:22` and `eval_FLAC.py:61`): 3a–3d + intentionally-unchanged `finetune_cond.py`; `unwrap_model.py` (broken upstream import, export out of scope) and `baselines/` (non-FLAC) need nothing.

### 3a. `src/data/yaw_rotation.py`

```python
def fa_cartesian_conditioning(conditioner, metadata, device,
                              angles=DEFAULT_FRAME_ANGLES,
                              max_fwd_samples=None):
    """Full-C4 Cartesian frame averaging (exp_21).

    Validation contract identical to invariant_conditioning (angles non-empty,
    angles[0]==0.0, cap validated via resolve_frame_avg_cap BEFORE any
    short-circuit). NO cylindrical step: base = conditioner(metadata, device)
    on raw Cartesian metadata. Orbit ids = POSE_KEYS (all four); asserts
    set(base) ⊇ POSE_KEYS (r2 nit); depth absent -> ValueError (fail-closed).
    Accumulation via _orbit_average_batched with present=POSE_KEYS; masks from
    base for every id; context_audio and any other id: base pass only.
    """
```

### 3b. `src/training/diffusion.py`

- ctor whitelist (~195): add `"fa_cartesian"`. yaw-aug guard (~240): also reject with `fa_cartesian`.
- `_compute_conditioning` (~510): branch mirroring fa_invariant's **two-call-shape discipline** (no-cap → four-argument form verbatim; declared cap → keyword), calling `fa_cartesian_conditioning`. Import added at line ~22.

### 3c. `src/training/factory.py`

- `_parse_yaw_aug_config` (~49): rejection extends to `("fa_invariant", "fa_cartesian")`.
- `_parse_frame_avg_cap_config`: pin by test that a declared cap and `frame_avg_angles` are read for `fa_cartesian` exactly as for `fa_invariant`.

### 3d. `eval_FLAC.py`

- argparse choices + fail-fast (~1087): add `fa_cartesian`. Import at ~61.
- Conditioning site (~1265): `fa_cartesian_conditioning` branch; frame-angle + cap resolution shared with fa_invariant; `--rotate-deg/--rotate-mode` protocol rotation upstream, unchanged.
- Suffix (~375): verify the existing rule already yields `_fa_cartesian_a4`; don't fork.
- Provenance (`orbit_provenance`, `build_metrics_record`, `build_predictions_meta`): fa_cartesian records like fa_invariant — `orbit_execution: "batched"`, real `frame_avg_angles`, real cap.

### 3e. New model config — `exp_21.../FLAC_AR_BFC.json`

Copy of `exp_07_fa_scratch_claude/FLAC_AR_BF.json`; training-block edits: `"cond_method": "fa_cartesian"`; `frame_avg_angles: [0.0, 90.0, 180.0, 270.0]` kept; `frame_avg_max_fwd_samples: 32` per D5 (declared knob postdating B-F's training, not a method change). Everything else byte-identical.

### 3f. Launcher — `exp_21.../bfc_launch.sh` + `bfc_launch_guardtests.sh`

**Recipe source = B-F's own launcher, `exp_07_fa_scratch_claude/bf_scratch_launch.sh`** (r2 correction — not the exp_09 file Rev 2 named); generic modern safety gates transplanted on top. Pinned manifest:

- AR train config only — **NO `--val-dataset-config` and no validation loader** (r2 blocker 1: B-F ran without one, `bf_scratch_launch.sh:88`; validation batches draw RNG noise and would perturb subsequent training draws — including one would break single-delta parity);
- `weights/FLAC/VAE.safetensors`; no resume/pretrained ckpt (from scratch);
- `--batch-size 32 --num-gpus 2 --accum-batches 1 --strategy ddp_find_unused_parameters_true --sync-batchnorm true --precision bf16-mixed --num-workers 6 --seed 42 --max-steps 40000 --checkpoint-every 2500` (strategy pinned explicitly; `defaults.ini:30` now says `auto`) — every remaining flag byte-matched against the exp_07 launcher at parity-audit time;
- wandb identity gate, ViT grad-ckpt per B-F recipe, offline DINOv3 pin, conda/PL asserts, VRAM/df floors, teed timestamped log.
- **Init-identity audit:** seed-42 instantiation checksum BFC ≡ BF initial state dicts (identical architecture ⇒ must pass; mismatch aborts).

### 3g. `worklog/worklog_yixun/gen_model_comparison.py`

Row specs (K=8, K=1) for BFC@40k, protocol label `fa_cartesian eval`. **Admission validator (r2 blocker 4 extension):** exact dataset path per K, `n_samples == 6337`, **ten room-family keys** (NOT `scene_count == 17` — `--record-per-scene` groups on `md["scene"]` = scene_name → 10 families, AR_md.py:23, generator convention gen_model_comparison.py:787), seeds exactly {42..46} (no dup/missing), EMA, `cond_autocast bf16`, batch 64, `cfg_scale 1.0`, `steps 1`, `rotate_deg 0`, cond_method `fa_cartesian`, `frame_avg_angles [0,90,180,270]`, **eval cap 64**, step-40000 checkpoint identity (sha), one evaluator pin. Generator tests extended. Staged here; regeneration cluster-only, as established.

### 3h. Tests — `src/tests/test_fa_cartesian.py` (+ extensions), TDD red→green

1. **T-C4-invariance** — full conditioning dict allclose-invariant (atol 1e-5, eval mode) under 90°/180°/270° pre-rotation; context_audio unchanged.
2. **T-vit-branch-pinned** — averaged `*_vit` outputs of `fa_cartesian_conditioning` **allclose** (r2 nit: not bit-equal on GPU) to `invariant_conditioning`'s on identical input, eval mode. (Conditioner-level pin only; training gradients still differ downstream — disclosed.)
3. **T-45-breaks** — 45° pre-rotation changes averaged outputs beyond T1's tolerance (deterministic, eval mode).
4. **T-audio-single-pass** — `context_audio` conditioner executes exactly once (counting mock; BN running-stats protection).
5. **T-dist-orbit-correct** — averaged `source`/`context_poses` equal the hand-computed mean of the four rotated-pose DistEmbedder outputs (loop reference, present=POSE_KEYS).
6. **T-batched≡loop** — batched executor equals loop reference for the extended id set at **batch 32** with caps {32, 64} (r2 nit: the two caps exercise different chunk partitions at this batch).
7. **T-mask-preserved** — every averaged id retains the exact base-pass mask object/values (r2 §a: tensor tests alone can't detect mask replacement); plus the `set(base) ⊇ POSE_KEYS` assert path.
8. **T-depth-required** — missing depth raises; per-sample width mismatch raises.
9. **T-dispatch** — ctor accept/reject; yaw_aug×fa_cartesian rejected at wrapper AND factory; cap accepted+plumbed; call-shape discipline mirrored; never calls `invariant_conditioning`.
10. **T-eval-protocol** — suffix `_fa_cartesian_a4`; record schema (cond_method, real angles, `"batched"`, cap); flag resolution shared with fa_invariant; eval-cap-64-with-batch-64 accepted, batch-64-with-cap-32 raises (pins blocker 3's split).
11. **Regression set** — `test_invariant_conditioning`, `test_yaw_symmetry`, `test_cond_dispatch`, `test_finetune_cond` (whitelist still rejects fa_cartesian), `test_yaw_aug_training`, `test_eval_paths`, `test_yaw_random_eval`, `test_exp14_fixed_mode_snapshot`, `test_frame_avg_cap_config`.

## 4. Validation ladder & parity audit

1. Static: `py_compile`; `FLAC_AR_BFC.json` parse; `bash -n`.
2. Full pytest: §3h + regression set.
3. Tiny synthetic forward: `_compute_conditioning` on synthetic metadata (CPU).
4. Real-data readback: AR samples through one `fa_cartesian_conditioning` call; C4 spot-assert on real tensors.
5. Smoke ~25 steps 2-GPU DDP, ckpt off (not a throughput measurement).
6. Rate check: windowed steps/s over ≥200 real-run steps under co-tenancy with exp12A; expectation ≈ B-F's historical pace (~3.5× P1's per-step cost — r2 nit, disclosed in §5) + negligible dist-embedder orbit cost.
7. Parity audit vs B-F: config diff = declared edits only; optimizer/scheduler/EMA/metrics byte-identical; launcher flag-by-flag match to `bf_scratch_launch.sh` (incl. NO val loader); **seed-42 init checksum BFC≡BF**; verify B-F's training-era orbit execution from the exp_07 record and reconcile with D5; announcement-06 chunk plans (train + eval separately) in the launch manifest.

Per-round Codex reviews each Coder round; integrative `full` review before launch.

## 5. Run & evaluation design

- **Training:** 1 arm, from scratch, seed 42, `FLAC_AR_BFC.json`, 40k steps, 2×A6000 **co-tenant with exp12A** (D3). Wall ETA set after rung 6 (FA-class step cost ≈3.5× vanilla; co-tenancy stretch measured, not guessed).
- **Registered eval at 40k — two executable templates (r2 blocker 2), one per K; SEED ∈ {42,43,44,45,46}:**

```bash
# K=8 (K=1: swap dataset config to acousticroom_unseeneval_1.json and K8→K1 in --eval-name)
python eval_FLAC.py --model-config worklog/worklog_yixun/exp_21_bf_fa_cartesian_claude/FLAC_AR_BFC.json \
  --dataset-config src/configs/dataset_configs/AR/eval/acousticroom_unseeneval.json \
  --ckpt-path outputs_FLAC/exp21_BFC/.../epoch=..-step=40000.ckpt \
  --cond-method fa_cartesian --frame-avg-angles 0,90,180,270 --frame-avg-max-fwd-samples 64 \
  --rotate-mode fixed --rotate-deg 0 --cond-autocast bf16 \
  --batch-size 64 --cfg-scale 1.0 --steps 1 --record-per-scene \
  --seed ${SEED} --eval-name exp21_BFC_S40000_K8_s${SEED}
```

  `--eval-name` carries K and seed (unique output per cell — `build_output_paths` does not add them itself, eval_FLAC.py:373). Flat split-level metrics = comparator estimand; per-scene payload (10 room families) preserved as the paper-style estimand, never compared to flat-only history.
- **Invariance grid (K=8, seed 42):** the registered K8/s42 cell is the 0° member; four more cells add `--rotate-deg {45,90,180,270}` with `--eval-name exp21_BFC_S40000_K8_s42_rot${DEG}` and **`--record-stream --expected-stream-count 6337`** on all five (identical ordering + full coverage proof). **Pre-registered acceptance (r2 blocker 5): absolute limits, not seed-range** — C4 spread (max−min over {0,90,180,270}) ≤ **T60 0.005 / C50 0.0005 / EDT 0.006 / R@1 0.15** (≈5× exp_10's measured B-F C4 floor of 0.0009/0.0001/0.0011/few-hundredths, margin for executor/precision drift); the **45° cell must EXCEED the C4 limit on T60 or EDT** (predeclared statistic) — a 45°-flat result fails the negative control and blocks interpretation.
- **Comparators (r2 blocker 4) — D6 for Yixun:** historical B-F@40k / P1@40k rows are legacy-executor, different-`source_sha` artifacts; `model_comparison.md` itself marks legacy-loop and batched rows non-interchangeable. **Recommended (D6-a): re-evaluate BOTH comparators at the current evaluator pin** — B-F@40k with `--cond-method fa_invariant --frame-avg-angles 0,90,180,270 --frame-avg-max-fwd-samples 64`, P1@40k with `--cond-method vanilla`, all other flags identical to the BFC template, 5 seeds × both K (20 cells, ~2–4 h on 2 GPUs) — making the paired per-seed deltas evaluator-clean. Fallback (D6-b): skip re-eval and demote historical rows to *contextual* comparators (no paired-delta claims). Chunk-plan disclosure: BFC train cap-32 / eval cap-64 batched; B-F training legacy-loop (verified at rung 7), re-eval batched; P1 no orbit.
- **Readout (pre-registered):** paired per-seed deltas BFC−BF and BFC−P1 (under D6-a), mean ± std; **primary metrics T60 + EDT at K=8**; all six table metrics + all K=1 cells reported; conclusions limited to the 40k checkpoint (optional 37.5k s42-K8 band-context screen — 42.5k does not exist, training stops at 40k); 5 eval seeds = sampling variability only, ONE training seed per arm; BFC/B-F ≈ matched-compute, **BFC/P1 matched-steps NOT matched-compute** (FA ≈3.5×/step — disclosed alongside any P1 comparison).
- **Follow-ons (separate approvals):** Cyl-PE arm; ViT-branch-averaging ablation; HAA finetune round.

## 6. Sequencing & ETAs

This plan (Rev 3) → Yixun approval + D5/D6 → Opus 5 (max) Coder TDD rounds w/ per-round Codex reviews → ladder → integrative `full` review → parity/init audits → commit+push → launch (co-tenant) → 40k train → eval block (**14 unique BFC cells** — 10 registered + 4 extra grid angles, the K8/s42/0° cell shared; + 20 comparator cells under D6-a) → results/analysis/HTML → commits log. Coding+reviews ≈ 0.5–1 day.

## 7. Decisions for Yixun

- **D5 — Training chunk cap:** `frame_avg_max_fwd_samples: 32` (**recommended** — reproduces B-F's per-angle draw schedule; r2-confirmed arithmetic) vs default 64. Training-only; eval always cap 64 (required at batch 64).
- **D6 — Comparator hygiene:** (a) **recommended** — re-evaluate B-F@40k + P1@40k at the current evaluator pin (20 cells, ~2–4 h) for evaluator-clean paired deltas; (b) skip and demote historical rows to contextual (weaker claims, zero extra compute).
- **Name check:** `fa_cartesian` / `FLAC_AR_BFC.json` / `exp_21_bf_fa_cartesian_claude` — veto here if you want different strings.
- Settled: D1–D4 (method per your Query 2; co-tenant launch; superseded name void) + numbering **renumbered exp_20 → exp_21** (Yixun 2026-08-21 ~23:25 — the concurrent session's RAF work owns exp_20; NAS `exp20_raf_*` artifacts extant).
