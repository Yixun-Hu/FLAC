# plan_loc_invert — exp_18: Inverting Vanilla FLAC for Source Localization (preflight)

**Author:** Claude Fable 5 (Planner seat). **Rev 3, 2026-08-18** — Rev 2 folded all 11 Codex review-of-record findings (`loc_invert_codex_plan_review.md`, C1–C11); Rev 3 folds the 23 supplementary Opus findings (`loc_invert_opus_plan_review.md`, O1–O23; disposition in `loc_invert_worklog.md`; O1/O2/O4/O14 independently re-verified by the Planner in code/data). Status: **awaiting Yixun approval before any implementation**.

## 1. Objective & research question

Test whether a frozen, pretrained Vanilla FLAC contains enough source-position information to localize the source of a held-out RIR in an **unseen room**, purely by analysis-by-synthesis inversion. exp_18 also delivers the protocol + code the later cross-arm experiment (FA B-F, yaw-aug, cyl-DINOv3) reuses unchanged.

**Registered success criteria (C1/O1/O10):** FLAC must beat BOTH (i) the **context-conditioned random baseline** (information-matched: the 8 context source positions are visible in the conditioning, so the non-context eligible set is only ~2 candidates — verified: 16/17 unseen rooms have 10 sources; `LivingRoomsWithHallway_idx_30` has 9, where elimination alone names the target) and (ii) the **non-generative nearest-context control** (§2.6). Wiring controls (§2.8) must prove the candidate coordinate is load-bearing.

## 2. Protocol (registered)

### 2.1 Queries & identity audit
- Split: full `acousticroom_unseeneval.json` → `data/AR/unseen_eval.json`, 6,337 items / 17 rooms (announcement 01). Query: h_obs = h(S_gt, R), receiver known, source hidden.
- Context: standard pipeline (8 refs from OTHER sources at the SAME receiver; exclusion by construction). The `sample_context_ids` fingerprint is recorded as a **regression/provenance guard** (O7 — it is not an independent leakage proof) and feeds each candidate's context-membership flag.
- **Fail-closed identity audit (C2/O15):** pre-generation enumeration; first `sample_target_id` mismatch aborts; headline artifacts only after exactly 6,337 identities / 17 rooms verified; split hash recorded.
- **Context-draw determinism (O8):** the per-item `np.random.choice` runs in worker processes, so the draw depends on `batch_size` AND `num_workers` — both are **pinned protocol values** (registered at launch), recorded in every manifest/output; `pl.seed_everything(seed)`, `shuffle=False`.

### 2.2 Candidate set (metadata-defined)
- C = valid sources enumerated from `metadata/` pair JSONs (C7), GT included; M ≈ 10 (9 in LRH_idx_30), logged. Numeric, naming-tolerant id parsing (wav-name vs metadata-name are separate namespaces — the readback rung establishes the real metadata naming rather than assuming the `"S00"+str` quirk; O6). `src_loc` uniqueness + cross-receiver consistency asserted.
- **Per-query geometry invariant (O6):** the GT candidate's camera-frame projection must equal the loader's `md['source']` exactly; mismatch aborts. Readback cross-checks metadata sources vs RIR files (Cafe_idx_1 has 922/1000 pairs — missing pairs shrink only the oracle's reported eligibility, never C or the headline denominator).
- Per-candidate conditioning: `source`/`source_vit` swapped (shallow-copy metadata variant, shared tensors for untouched keys, base-not-mutated test; O19); `depth`/`context_*` shared.

### 2.3 Generation
- Frozen FLAC, EMA weights (the `resolve_weights_source` outcome recorded; O9), rectified flow, `steps=1`, `cfg_scale=1.0`, `torch.set_float32_matmul_precision('medium')` as in `eval_FLAC` (O9). Announcement-05 flags pinned: `--cond-method vanilla --rotate-deg 0 --cond-autocast default`, fa-angles `n/a`; fail-closed on nonzero rotate-deg, ARE checkpoints, non-RF objectives (C5/C8).
- K = 8; **deterministic noise bank** keyed `(seed, query_id, k)` via dedicated `torch.Generator`s (never the global stream), shared across candidates (common random numbers; C10). Permutation/batch-split equivalence + resume-reproducibility tests.
- Conditioner runs once per query over the M metadata dicts. **Optional optimization** (compute the 8 shared context encodings once instead of M×; O13) is admissible ONLY behind a bit-parity test against the monolithic call, decided after R0 timing.
- One-query **numerical parity test** vs the `eval_FLAC` generation path (same ckpt/metadata/noise ⇒ identical waveforms; C8).

### 2.4 Scoring (deterministic readout; established preprocessing)
- **Verified (O2):** AGREE's audio tower ends in a sampling `VAEBottleneck` (`audio_model.py:201` → `vae_sample` → `randn_like`, no eval guard) — the stock `encode_audio` is stochastic and consumes global RNG. **Registered scorer: the deterministic VAE-MEAN readout** — `layers(x)` → `chunk(2)[0]` (mean) → flatten → `project` → L2-normalize — implemented in `src/localization/agree_embed.py` without editing AGREE code. Tests: determinism; zero global-RNG consumption; stub-weights equality with the sampled path at stdev→0; batch-size invariance. R0 additionally **measures** the sampled readout's noise (100 draws, pairwise-cos distribution) as a labelled diagnostic (§2.8).
- Preprocessing = the established AR metric route (C6, code-verified): clamp [-1,1] → first-8,000 samples (`max_len`) → tower's 10,240 handling; preprocessing-tensor equality test vs the real `update_metrics→Retrieval` route (exact-embedding equality vs the stock path is impossible under its sampling — compared at mean-readout instead).
- AGREE `.eval().requires_grad_(False)`, `torch.inference_mode()`; ckpt sha256 recorded. s_{m,k} = cos; S_m = LME_τ via `torch.logsumexp` (τ=0.02 stability test; O18); argmax w/ lowest-index tie-break. All s_{m,k} logged at full float32 round-trip precision (hex or 17-sig-digit repr; offline-vs-online re-aggregation equality test; O18).

### 2.5 Hyperparameter registration
- **LME with K=8 is the registered method** (C4); dev selects **τ only** from {0.02, 0.05, 0.1, 0.2, 0.5}, objective = dev **pooled mean e_loc** (median is a step function of top-1 at M≈10; O11), tie-break smallest τ. Registered in `_params_set_up.md`, **committed before R2 with its SHA recorded in the R2 manifest** (O17).
- Dev scope: full seen split if R0's measured cost permits, else a labelled bounded dev slice (tuning-only, never in `_results.md`; O23) — decided at R0, recorded either way.
- mean/max/K′∈{1,2,4}: labelled offline sensitivity only.

### 2.6 Metrics & baselines
- **Primary: pooled median e_loc over the 6,337 queries** (C3). Also pooled mean, success@0.5 m, success@1.0 m (noting at M≈10 these are near-deterministic functions of top-1 — reported with that caveat; O11). Room key = `scene_name/scene_id` (17 rooms). Labelled secondaries: equal-room macro stats, top-1, MRR.
- **Baselines, identical weighting/conventions (C1/C3/O5):**
  - *Uniform-over-C* (spec's literal lower bound; exact).
  - *Context-conditioned* (REGISTERED comparison target; exact, uniform over non-context candidates). Eligible-set-size distribution reported; **LRH_idx_30 (9 sources ⇒ eligible set = {GT}, baseline 100%, zero headroom) is reported separately and excluded from the information-matched aggregate, with that exclusion stated in `_results.md`**. Context-member-prediction rate reported.
  - *Non-generative nearest-context control (O10)*: pick the context source whose measured RIR best matches h_obs in E_a; predict the candidate nearest that source. Needs no FLAC — attribution control for "is the generator adding anything".
- **Statistics (C3/O12):** per-seed values; seed mean ± SD (variability, never a CI); 17-room clustered bootstrap CI on the primary; **paired per-query comparisons** (FLAC vs each baseline, room-clustered) pre-registered.
- **Oracle (O4, revised):** measured-RIR runs need NO FLAC checkpoint ⇒ they run **the moment the dataset lands** (run R-1) together with the baselines. Identity variant = pipeline sanity only, and under the sampled-readout diagnostic its cos<1 scorer-noise is itself informative; under the registered mean readout identity-cos = 1 by construction. No `single_channel_ir_2+` is referenced anywhere in-repo — a second-measurement oracle exists only if dataset inspection finds one (readback rung); otherwise explicitly reported as unavailable.

### 2.7 Heatmaps
As Rev 2 (rule-selected gallery, T_disp = registered τ, candidate-extent labelling) + optional true room silhouette from the `md['depth']` point-cloud floor projection (O20).

### 2.8 Wiring & sensitivity controls (O3/O21 — run before the headline is interpreted)
1. **Constant-source control:** regenerate a bounded, pre-registered slice with `source`/`source_vit` frozen at the room centroid for all candidates → localization must collapse to the context-conditioned baseline. Proves the coordinate conditioning is load-bearing.
2. **Between/within-candidate power statistic:** with common random numbers, report var_m(mean_k s_{m,k}) / mean_m var_k(s_{m,k}) — candidate identity must move similarities more than sampling noise.
3. **Scorer-noise measurement:** sampled-readout pairwise-cos distribution (R0) quantifies what the mean readout removes.
4. **`--cond-autocast off` diagnostic** on a labelled slice to rule out fp16 ranking artifacts (O21).

## 3. Checkpoints & assets (decisions §8)
- FLAC ckpt for the exp_18 headline: **recommendation changed (O22) to released `FLAC_EMA.ckpt`** — it IS "pretrained Vanilla FLAC" per the spec, Table-1-verified on this exact box (exp_01), and HF-downloadable now (unblocks everything before any rsync). exp07_P1 / VANL@40k rows move to the cross-arm experiment.
- Scorer: `AGREE_AR.pt` primary, `AGREE_fullAR.pt` labelled diagnostic.
- **Additional assets verified as required (O14):** `weights/FLAC/VAE.ckpt` (wrapped ckpt; AGREE's audio tower loads it at construction, CWD-relative, before the AGREE state dict overwrites it) — comes with `download_weights.sh`; **gated DINOv3 HF access** (this box has no HF cache, `HF_HOME` unset) — needs `huggingface-cli login` (Yixun) or an HF-cache rsync; AcousticRooms with `metadata/` + `single_channel_ir_1/`.

## 4. Implementation plan (per file, per-function tests)

As Rev 2 §4 with these deltas (full contracts live in the Rev 2 tables, which remain binding):
- `candidates.py`: + `assert_gt_matches_loader(cand_set, md)` (O6; test: exact match passes, 1e-6 perturbation aborts); `candidate_metadata` becomes shallow-copy-with-key-swap (O19; base-intact test). `enumerate_metadata_sources` remains the candidate authority.
- `scoring.py`: `aggregate` via `torch.logsumexp` + τ=0.02 stability test (O18); + `context_conditioned_baseline` handles the empty-eligible/GT-only case explicitly (LRH_idx_30 test); + `nearest_context_baseline(cand_xyz, ctx_xyz, ctx_sims)` (O10; hand-example test); + `paired_room_clustered_test(records_a, records_b)` (O12; synthetic test); + `power_statistic(sims)` (§2.8.2; test).
- `agree_embed.py`: `embed_rirs(..., readout={'mean','sample'})`, mean = registered; tests: determinism, global-RNG-state unchanged (O2), stub stdev→0 equality, batch invariance, preprocessing-tensor equality vs real callback route (C6), pads-only-never-crops edge (O18).
- `eval_localization.py`: + `--context-k` passthrough by dataset-config choice only (existing `_1/_4` configs); + pinned `--batch-size --num-workers` recorded in provenance (O8); + smoke query ids pinned to SEEN rooms (O16); + constant-source control mode `--control constant_source` (§2.8.1); + serialization precision round-trip test (O18); everything else per Rev 2 (audit, noise bank, layout, parity harness, gt_rir mode, smoke guard).
- `loc_invert_heatmaps.py`: as Rev 2 + optional depth-silhouette helper (pure-function test on synthetic point cloud).

## 5. Run matrix (resequenced — O4 makes R-1 checkpoint-free)
| Run | Needs | Split | Purpose |
|---|---|---|---|
| **R-1 dataset gate** | dataset + AGREE only (no FLAC ckpt) | full unseen | readback rung + oracle sanity + BOTH random baselines + nearest-context control + eligible-set stats — runs the moment AcousticRooms lands |
| R0 probe | + FLAC ckpt | pre-registered SEEN-room queries, `--smoke` | end-to-end pipe, parity harness, **fit & timing probe** (peak mem + per-component times at batch 64, max-M, K=8), scorer-noise measurement, autocast-off diagnostic |
| R1 dev-tune | 〃 | seen (full or labelled slice, R0-gated) | τ selection (§2.5) |
| R2 registered | 〃 | full unseen | headline; seeds 42/43/44, one per GPU; params SHA pre-committed |
| R2b (if approved, §8.4) | 〃 | full unseen, `acousticroom_unseeneval_1.json` | K_ctx=1 registered secondary: eligible set 9/10 ⇒ the strong test of acoustic (vs elimination) information |
| R3 controls | 〃 | pre-registered slices | constant-source control + power statistic |

## 6. Validation ladder
As Rev 2, with rung 4 (readback) additionally establishing: real metadata file naming (namespaces; O6), `single_channel_ir_2+` existence, gated-DINOv3/HF load, VAE.ckpt presence, per-room source counts vs the split-derived expectation (10×16 + 9×1). Rung 6 (fit/timing probe) not waived.

## 7. Integrity controls
Rev 2 list + deterministic mean-readout scorer (no hidden scorer noise, no RNG cross-talk); pinned dataloader parallelism; geometry invariant per query; information-matched + non-generative controls as registered comparison targets; zero-headroom room handled openly; pre-registration commit-then-run (O17).

## 8. Open decisions for Yixun (recommendation first)
1. **Approve Rev 3 protocol** as registered.
2. **Headline ckpt = released `FLAC_EMA.ckpt`** (recommendation CHANGED per O22: spec-faithful "pretrained Vanilla FLAC", Table-1-verified on this box, downloadable now; program ckpts move to the cross-arm exp) — confirm, or name exp07_P1 / VANL@40k instead.
3. **Scorer** = `AGREE_AR.pt` primary + fullAR labelled diagnostic; **deterministic mean readout** registered (the stock sampled readout becomes a diagnostic) — confirm.
4. **K_ctx=1 secondary sweep (R2b)** via the existing `_1` config — recommended YES (it is the strong version of the scientific claim given the eligible-set-2 problem; ~doubles GPU cost) — approve or drop.
5. **Seeds 42/43/44**, K=8 — confirm.
6. **Dataset route**: rsync AcousticRooms (incl. `metadata/`) vs fresh download; **HF access for gated DINOv3** on this box (`huggingface-cli login` yourself, or rsync `~/.cache/huggingface` from the other box); FLAC/AGREE weights via `download_weights.sh` unless you prefer rsync.

## 9. Compute budget (corrected basis; O13/C9)
exp_01 K_ctx=8 eval ≈ **13 min**/seed (not the K=1 run's 6.5). Naive ×80 sample-count scaling bounds a sweep at ~17 h; expected with per-query conditioner sharing ~10–15 h; the optional context-encoding amortization (5× fewer ViT forwards, parity-gated) could bring it to a few hours. R0 measures; >12 h/sweep projection returns to Yixun before R1. R-1 and R4-style baselines: CPU/1-GPU minutes.

## 10. Deliverables
As Rev 2. Cross-arm table out of scope.

---

## Rev 3.1 amendment (2026-08-20, Planner; post rung-4 readback + integrative full review — factual corrections and enforcement hardenings, no change to the approved science protocol)

1. **LRH_idx_30 corrected (rung 4 fact):** ALL 17 unseen rooms have 10 metadata-defined sources ⇒ M=10 everywhere; LRH's source 10 has metadata but no wavs ⇒ its eligible set = {GT, S10} = 2, the same 50% information-matched chance as every other room. §2.6's gt_only exclusion clause is retained purely as a guard; it is expected never to fire. Oracle eligibility shrinks by one in LRH by design. No second measurement channel exists ⇒ identity-oracle is sanity-only (the §2.6 fallback stands).
2. **Frozen candidate manifest (full-review F1):** candidates are precomputed once per run into a room-level manifest (nodes, coordinates, wav availability), consistency-checked, sha256-hashed into provenance, and consumed from memory per query — no per-query disk enumeration. The same manifest hash is required across seeds/arms (enforced via item 4).
3. **Reviewed entry points (full-review F2):** `--mode readback` (R-1 gate), always-on per-query component timing + CUDA peak-memory in run summaries (R0's probe = its smoke summary), `--mode scorer-noise` (§2.8.3 measurement), `--mode reaggregate` (R1's offline τ/agg/K′ sweep + the registered smallest-τ selection) — all TDD'd and Codex-reviewed like everything else.
4. **Machine-checked registration (full-review F4):** registered unseen runs require a committed JSON registration manifest locking {config hashes, ckpt/scorer shas, K, τ, agg, cond-method, autocast, steps, cfg-scale, seeds, readout, candidate-manifest hash}; the driver verifies `--registration-sha` is a real commit containing that exact manifest and refuses any locked-field mismatch BEFORE loading models.
5. **Cell-unique artifact names + no silent overwrite (full-review F5); complete device provenance (F6); all validation before model loads (F7); fail-closed context evidence for AR rows (F3); finite `--frame-avg-angles` (Part-1 #9 leftover).**
6. **R0 smoke identities:** the exact first-N seen-split identities are recorded in `_params_set_up.md` at R0 launch (per full-review launch conditions).

## Rev 3.2 amendment (2026-08-19 ~21:45 EDT, Planner; forced by measured data, headline unaffected)

**Duplicate-position sources merge into one candidate.** R0's manifest gate aborted on a real property of the SEEN split: 2/131 seen rooms (`Bathrooms_idx_11`: S9≡S10; `Bathrooms_idx_16`: S4≡S7) have two source nodes at one position (≤1e-6 m). The 17 unseen rooms are clean (proven by R-1b's full manifest build), so the registered headline protocol is untouched. Amendment: the manifest merges position-duplicate nodes into a single candidate (canonical = lowest node id; merge map recorded in manifest, rows, and provenance); GT maps to its merged candidate (e_loc identical by construction); context fingerprints resolve duplicate poses to the one merged index; gt_rir uses the canonical node's file (fallback: any member's, recorded). Near-duplicates beyond 1e-6 remain distinct candidates. Rationale: candidate identity is positional — two labels at one point are one localization hypothesis; the alternative (excluding 2 rooms) would subset the dev split.
