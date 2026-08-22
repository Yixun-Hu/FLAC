# plan_raf_mappingA — exp_21: Mapping-A (unseen-source) cross-arm evaluation on RAF

**Author:** Claude Fable 5 (Planner seat). **Rev 1, 2026-08-21** — pre-review draft. Status: awaiting Codex review, then Yixun approval BEFORE implementation. Branch `raf-mapping-a` (from `raf-finetune-exp` @ `263ef27` — inherits the full reviewed exp_19 pipeline, 557-test suite, canonical publications).

## 1. Objective & claim under test

Evaluate P1 (vanilla), YAW (yaw-aug), BF (FA) 40k AR checkpoints zero-shot on RAF under **Mapping A**: listener-centered conditioning where context = the *same microphone* hearing *other speaker positions* at the same array placement, and the model predicts the RIR for an **unseen speaker position**. This is the AR-native relation (context over sources at a fixed receiver), RAF's own benchmark task, and the capability the localization program consumes. Paired with exp_20's Mapping-H rows (same checkpoints, same rooms, receiver-interpolation task), the comparison isolates source-generalization from receiver-interpolation on identical real data.

## 2. Data facts the protocol builds on (measured in exp_19; re-verified at readback rung)

EmptyRoom: 139 placements × median 9 tx (max 108); FurnishedRoom: 121 placements. Placements re-occupied sub-cm; each tx-group has exactly 36 captures (one 72-group excluded as before). Within a placement, the mic rig is fixed ⇒ for a target (tx*, mic m) the captures (tx_j, mic m), j≠*, exist at sub-cm-identical receiver positions — the exact AR "same receiver, other sources" relation. Readback rung verifies: per-placement mic correspondence (36 mics match across the placement's tx-groups within a registered tolerance, default 2 cm — measured before any item is built), per-placement tx counts, and K-eligibility.

## 3. Protocol (registered)

- **Item** = (placement p, mic m, held-out target tx*): predict RIR(tx*, m) given context {RIR(tx_j, m)} for K=8 other tx at p. **Conditioning (AR semantics, reusing `AR_md`'s frame convention):** frame centered at the LISTENER m; `source` = tx* − pos(m); `context_poses` = {tx_j − pos(m)}; `context_audio` = the same-mic RIRs; `depth` = equirect panorama rendered at pos(m) (AR convention: listener).
- **Eligibility:** placements with ≥ 9 tx-groups (K=8 + 1 target). **Selection (deterministic):** FPS over placement centroids, **16 placements/room**; per placement, ALL 36 mics; **target tx = the FPS-last tx of that placement's tx set** (deterministic, spatially extremal); context = 8 drawn deterministically per item (stable hash seed, exp_19 machinery). ⇒ **576 items/room, 1,152 total**, fixed and published as `data/RAF/mappingA_eval.json` + a splits-record with the full provenance (counts, distances, exclusions).
- **Leakage:** the target tx's captures never appear in context (different tx by construction); nothing is trained, so the only integrity surface is identity — full stream audit (`--expected-stream-count 1152`), capture-ID fingerprints, per-item identity as in exp_19/20.
- **Arms & flags (announcement 05):** P1 & YAW `--cond-method vanilla`; BF `--cond-method fa_invariant` (default C₄ angles, fwd-cap 64) — note fa frame-averaging rotates listener-centered panoramas, its native convention. All: `--rotate-deg 0 --cond-autocast default --cfg-scale 1.0 --steps 1`, 5 seeds (42–46), batch 64. Optional labelled context row: exp_19's RAF-finetuned ckpt (trained under Mapping H — labelled cross-mapping transfer, not a like-for-like row) — decision §7.3.

## 4. New code (TDD; the only new surfaces — everything else reused verbatim)

1. `data/RAF/prepare_mappingA.py` — placement mic-correspondence audit; eligibility + FPS selection; item manifest + `mappingA_eval.json` (HAA-shape, target wavs) + runtime `mappingA_metadata.json` (item → {mic pos_p, target tx_p, context capture ids, depth file}); publishes under the existing `PublishTransaction` with its own marker kind + parameter identity (16 placements, 36 mics, K=8, tolerance 2 cm, readback digest, generation binding to the existing audio publication — REUSES the published scaled WAVs, no re-resample).
2. `data/RAF/render_depth.py` — a `--positions-from mappingA` mode rendering at the 1,152 mic positions (576/room; ~10 min; same QA machinery, miss cap, no-flipud, mask-derived audits; sightline recorded-only as registered).
3. `src/configs/dataset_configs/custom_metadata/RAF_A_md.py` — Mapping-A hook (listener frame; deterministic eval-mode context by capture id; same publication gate pattern pointed at the mappingA marker; contract tests incl. the real-conditioner pass).
4. `src/configs/dataset_configs/RAF/eval/raf_mappingA.json` + a model-config clone if any metrics field differs (expected: none — RAF policy applies unchanged).
5. Tests mirror exp_19 patterns (synthetic fixtures; determinism; gate bypass negatives; identity plumbing).

## 5. Validation ladder & review

Readback rung (mic correspondence + eligibility stats, committed record) → prep dry-run (1 placement) → canonical mappingA publication → depth renders + QA → smoke eval (pre-registered 8 items, one arm) → full 3-arm × 5-seed sweep. Universal Codex review: plan (this doc) → code round(s) → focused delta if amendments. Threat model & convergence rules of exp_19 (Amendments 6/8) carry over verbatim.

## 6. Compute & timeline

Prep: CPU ~10 min (no resampling — reuses published WAVs). Renders: 1,152 maps ≈ 15 min. Eval: 1,152 items ≈ 1.5× exp_20's 768-item rows ⇒ ~5 min/seed vanilla, ~15 min/seed fa ⇒ full sweep ~2.5 h on two GPUs. End-to-end (after plan approval + code round): roughly one working day including reviews.

## 7. Open decisions for Yixun (recommendation first)

1. **Approve the §3 protocol** (16 placements/room × 36 mics × FPS-last target tx; 1,152 items; K=8).
2. **Depth at every mic** (1,152 renders, faithful AR convention — recommended) vs one per placement centroid (72 renders, ~1.4 m position error inside the rig).
3. **Include the RAF-finetuned (Mapping-H-trained) checkpoint as a labelled transfer row** — recommended YES (cheap, and it shows whether H-finetuning helps or hurts unseen-source prediction).
4. **BV checkpoint**: include as a fourth arm (one more sweep, ~30 min) — recommended YES for completeness if BV is the vanilla twin of BF.
5. Seeds 42–46, batch 64 — confirm.
