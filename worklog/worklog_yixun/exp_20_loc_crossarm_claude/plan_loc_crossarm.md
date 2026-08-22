# plan_loc_crossarm — exp_20: cross-arm analysis-by-synthesis localization at matched 40k (PROPOSAL)

**Author:** Claude Fable 5 (Planner), 2026-08-21. Protocol = exp_18's registered machinery verbatim; this plan specifies only the deltas. Status: → Codex plan review → **Yixun approval** → runs.

## 1. Arms & inputs (NAS `checkpoints/ar_40k_endpoints/`, MANIFEST verified 4/4 OK)
| Arm | ckpt sha256 (verified) | Training lineage (PROVENANCE.md) | Eval conditioning |
|---|---|---|---|
| **P1-VANL@40k** | `c4c67882…` | exp_07 P1 vanilla, this recipe family | `--cond-method vanilla` |
| **BF-FA@40k** | `5319feb4…` | exp_07 B-F fa_invariant (C4 frame-avg) | **`--cond-method fa_invariant`** (announcement 05 — mismatch is catastrophic; announcement 06 orbit provenance recorded) |
| **YAW@40k** | `ac1f2603…` | **exp_17 A6000 yaw-aug** (recipe-matched to P1/BF; deliberately NOT exp_15's cluster arm `16b964ec` — different run, different recipe; disclosed in every artifact) | `--cond-method vanilla` |
| (BV@40k) | `ace9f735…` | exp_07 B-V | delivered, NOT in scope unless Yixun adds it |
All wrapped PL ckpts (optimizer+EMA): acceptance criterion per run — `weights_source` resolves **"ema"** (unlike released-EMA's "online").

## 2. Cells (per arm; identical to exp_18's registered protocol)
R2 (K_ctx=8) seeds 42/43/44 + R2b (K_ctx=1) seeds 42/43/44 — full unseen split, same split/candidate digests, same pinned loader (4/4), same noise keys, K=8, LME τ=0.02, mean-readout AGREE scorer; waveform dumps ON (announcement 08, `exp18_pred_waveforms/`-style per-cell NAS dirs under a new `exp20_pred_waveforms/`); **metrics inline** (`--metrics`: all five frozen families ride the same pass — no replays needed; m2 is the registered secondary scorer, incl. the m2-K1 cells for every arm).

## 3. Registration
Per arm × regime: 6 new protocol manifests (exp_18 template, new ckpt sha) + per arm: metric manifest with **constants copied VERBATIM from the frozen `d6dbf00` config** (Δmax=8, μ/σ, backend — nothing recalibrated) and only the binding digests (protocol-manifest digests, ckpt sha) rebound. All committed pre-run; machine-gates as in exp_18.

## 4. New code (one TDD round + one Codex review)
- **fa-parity gate**: extend the parity harness to `--cond-method fa_invariant` — one real query, BF ckpt, driver's fa conditioning vs an `eval_FLAC`-faithful fa replay, bitwise (autocast-off) — REQUIRED green before any registered BF row (exp_18 only ever exercised vanilla dispatch).
- Per-arm manifest generation script (exp folder tooling, reviewed).
- Nothing else changes; `src/localization/` and the driver are frozen at exp_18's reviewed state.

## 5. Analysis (pre-registered)
- **Primary contrasts (Holm over 4):** BF vs P1 and YAW vs P1, per regime, AGREE scorer, paired per-query room-clustered (top-1 indicator + e_loc), matched noise keys/contexts (same seeds ⇒ same context streams ⇒ paired at query level).
- Secondaries (labelled): all m2/other-family contrasts, BF-vs-YAW, comparisons to exp_18's released-EMA rows (step-mismatch disclosed), per-room structure (the small-room effect), ctx-member failure mode per arm, sparse-regime margin per arm.
- Deliverables per SOP: results/analysis/HTML gallery extension, living cross-arm localization table in the exp folder.

## 6. Validation ladder
1–2: suite green (existing 2,688) + new fa-parity tests. 3: per-arm CPU ckpt probe (key inventory: `diffusion_ema.*` present; BF conditioning config sanity). 4: per-arm 2-query seen smoke (BF under fa) incl. weights_source=ema check. 5: fa-parity gate green. 6: registered runs.

## 7. Compute & schedule
~3.5 h/cell (metrics inline) × 6 cells × 3 arms ≈ 63 h GPU ≈ **~32 h wall on 2 GPUs** (arm-parallel; BV would add ~11 h wall). Order: smokes+parity (~1 h) → P1 + BF first (both GPUs), YAW follows.

## 8. Open items for Yixun (recommendation first)
1. **YAW = exp_17's A6000 arm** (your rsync's choice; recipe-matched — recommended) — confirm.
2. **BV@40k**: hold (recommended; add later as one command if wanted) vs include as 4th arm (+11 h).
3. **Metrics inline everywhere** (recommended; gives m2-K1 per arm at zero extra passes) — confirm.
4. Seeds 42/43/44 — confirm.

---

## Rev 2 (2026-08-21, Planner; folds all 7 findings of `loc_crossarm_codex_plan_review.md`)

**B1 — FA protocol binding & parity.** The BF registration locks, machine-checked: `cond_method=fa_invariant`, `frame_avg_angles=[0,90,180,270]`, `rotate_deg=0`, `cond_autocast=default` (the registered path), `FRAME_AVG_MAX_FWD_SAMPLES = candidate micro-batch` ⇒ **per-angle execution** (matching exp_07 BF's training-era per-angle behavior; announcement 06 §3 — the chunk plan is declared, not derived), plus batch/workers/orbit provenance. The fa-parity gate runs under the **registered autocast** (bit- or tolerance-bound as measured, with autocast-off as the exactness diagnostic), on a real query, BF ckpt, vs an `eval_FLAC`-faithful fa replay. Refusal tests: BF→vanilla, P1/YAW→fa_invariant, any mutated FA field vs the manifest.

**B2 — Checkpoint admission (exp_15 contract, ported).** Per arm, CPU, before its manifests are committed: `global_step==40000`; embedded model config canonically equals the arm's config file; EMA↔online suffix sets/shapes/dtypes one-to-one and complete (partial-EMA ⇒ refusal); full load-integrity (0 missing/unexpected); sha256 + resolved identity written to an admission record committed with the manifests. Tests: partial-EMA fixture, step mismatch, config mismatch.

**B3 — Paired-inference gate + statistics.** Paired contrasts are computed only after a validator proves, per (regime, seed), exact equality across arms of: query id/order stream, context-stream digest (or full fingerprints), split + candidate-manifest digests, loader settings, and noise-key arrays; any mismatch blocks paired reporting (unpaired fallback labelled). **Seed aggregation: per-query mean across the three seeds, then room-clustered inference** (seeds are replicates, never independent queries). **Multiplicity: top-1 is the sole confirmatory endpoint — Holm over exactly 4 contrasts** (BF vs P1, YAW vs P1 × two regimes); e_loc and every other metric supportive/descriptive.

**M4 — Planned code & per-function tests (announcement 02).** New reviewed units: `fa_parity_gate` (+registered-autocast tolerance record), `admit_checkpoint` (B2), `validate_pairing` (B3), `aggregate_seeds_per_query`, `build_holm_family`, per-arm manifest generator; red→green tests for partial EMA, step/config mismatch, cond-method refusal matrix, mutated orbit fields, context/noise mismatch, incomplete cells, seed aggregation, Holm construction.

**M5 — Metric inheritance.** One canonical scorer subdocument inherited from `d6dbf00` by deep-equality gate; only binding fields may differ per arm; per-arm recalibration prohibited; framed as **fixed external scorers** with the transport caveat stated (Δmax/M4-norms calibrated on released-ckpt seen generations; validity beyond that domain not claimed; AGREE primary unaffected).

**M6 — YAW binding.** The admission record (B2) binds `ac1f2603…` to step 40,000 + canonical exp_17 config; lineage cited to immutable exp_17 completion commits (`42cbdda`, `f378775`) + NAS PROVENANCE. All cross-arm conclusions framed as conditional on single historical training runs per arm (no replicated-training causal claim).

**M7 — Compute/storage.** Timed **100-query pilots** (vanilla and BF per-angle) before the schedule is finalized; the 32 h estimate is provisional until then (BF per-angle conditioning is ~4× the vanilla ViT work — plan for 40–55 h wall). Storage registered: ~374 GB dumps + margin, **500 GiB free-space floor** checked at launch, unique per-cell dirs, manifest completion checks, partial-run recovery via the existing atomic-publish machinery.
