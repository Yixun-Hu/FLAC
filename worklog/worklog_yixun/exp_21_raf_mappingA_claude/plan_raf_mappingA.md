# plan_raf_mappingA — exp_21: Mapping-A (unseen-source) cross-arm evaluation on RAF

**Author:** Claude Fable 5 (Planner seat). **Rev 2, 2026-08-21** — folds all 10 Codex findings (`raf_mappingA_codex_plan_review.md`, M1–M10). Status: **awaiting Yixun approval before any implementation.** Branch `raf-mapping-a` @ base `263ef27` (full exp_19 pipeline inherited).

## 1. Objective & licensed claims (M6 folded)

Evaluate P1 / YAW / BF 40k checkpoints zero-shot on RAF under Mapping A: predict the RIR of an **unseen source position** at a fixed microphone, given the same mic's recordings of other sources at that array placement. **Primary inference: the cross-arm ranking WITHIN Mapping A** (identical items/seeds/pipeline per arm). exp_20's Mapping-H rows appear only as a non-matched descriptive reference — absolute H-vs-A deltas are confounded by item construction and are NOT interpreted. (A matched H/A paired protocol over identical target RIRs is a possible future experiment, out of scope.)

## 2. Data preparation (M1/M2 folded — the two structural blockers)

- **Audio union build (M1):** exp_19 published only 21 selected tx-groups/room; Mapping A needs the full union of its target+context captures (≈ 324/placement × 32 ≈ **10,368 WAVs, ~1.4 GB float32**). `prepare_mappingA.py` enumerates the exact union, resamples raw 48 kHz → 22,050 float32, applies the ×3 scalar **only after a pre-publication amplitude audit over that exact union** (no-clip at ×3, finite/nonzero, nothing under the −60 dB loader gate post-scale). Any violation **stops the run for a registered amplitude-policy decision** — never drop/substitute items. Provenance binds raw capture ids, output hashes, scalar, and the exp_19 generation for any file genuinely shared with the Mapping-H publication.
- **Placement + mic correspondence (M2) — preregistered algorithm, replacing the informational centroid-rounding:** (i) cluster tx-groups by receiver-set proximity with complete linkage (no transitive chaining), cap 5 cm; (ii) deterministic medoid template per cluster; (iii) per tx-group a 36×36 one-to-one assignment (Hungarian) to the template; require unique matches with p95 displacement ≤ 1 cm, hard anomaly cap 2 cm, and ambiguity margin ≥ 3× the matched displacement to the next-nearest mic (inter-mic spacing ≫ this); (iv) record displacement p50/p95/max + rigid-array residuals per placement. A failing tx-group is excluded BEFORE eligibility; eligibility = ≥ 9 passing, **source-xyz-distinct** groups; any post-publication correspondence violation aborts the publication (never silently shrinks the item count). The quoted 139/121 placement counts are re-derived by this algorithm at the readback rung, not assumed.

## 3. Item construction & conditioning semantics (M3/M5/M8 folded)

- **Item** = (placement, mic slot m, held-out target tx-pose). **Registered formulas (exactly `AR_md`'s, per-capture own-rx):** `source = tx*_p − rx_target_p`; `context_pose_j = tx_j_p − rx_context_capture_j_p` (each context uses ITS OWN capture's rx, not a nominal mic position); `depth = panorama(rx_target_p)`; `*_vit` fields copied correspondingly. Every `rx_context_j − rx_target` displacement is recorded; the protocol is described as an **approximate same-listener realization bounded by the correspondence audit** — never "exact AR relation".
- **Unseen SOURCE POSITION (M5):** context excludes every group sharing the target's source xyz (not just the target pose) — quaternion-only duplicates are excluded too (the exp_19 quat-per-xyz audit feeds this).
- **Target selection (M8, recommendation changed):** targets chosen by **stable uniform hash** over each placement's eligible poses (FPS retained only for placement coverage) — the estimand is general unseen-source performance, not spatial-stress. Target-to-context distance distributions published either way. (§8.1 offers the FPS-last "spatially extreme" variant as a labelled secondary if Yixun wants both.)
- Selection: 16 placements/room (FPS over validated placement centroids) × 36 mic slots × 1 hash-chosen target ⇒ **1,152 items**, K=8 deterministic per-item context. Published as `data/RAF/mappingA_eval.json` + manifest + splits-record.
- **Static manifest validator (M5):** 1,152 unique targets; exactly 8 distinct contexts each; no target capture / target-xyz group in context; matched mic slot; displacement bounds; no duplicate atoms. Runtime: exact stream counts, positional audit, schema-2 capture ids, `input_hash` identical across all arm×seed cells, publication generations/digests identical.

## 4. Publication engineering (M4 folded — listed code surface)

Mapping-A gets **disjoint split/runtime/depth roots** and flavor-scoped markers (`mappingA_prepare`, `mappingA_depth`) with registered canonical identities (placement count, K, matching algorithm version + tolerances, correspondence digest, audio-union digest, renderer params, readback digest, scalar + derivation). `publish.py` verifier work is in scope: `RAF_A_md` verifies the Mapping-A combined publication while `RAF_md` continues verifying Mapping-H; composition tests: H→A, A→H, republish, injected crash — both flavors must remain simultaneously valid (exp_19 r4-T4 history).

## 5. Renderer & FA specifics (M9 folded)

`render_depth.py` gains a listener-mode: parameterized by target receiver position, **independent raw RAF receiver height** for the nadir gate, and transmitter endpoints for the recorded sightline diagnostic; listener-map containment/nadir/bounds/scale gates; exact no-flipud equirect convention; QA provenance labels the map as listener-positioned. Tests: real conditioner pass under BOTH vanilla and fa_invariant on Mapping-A metadata + a C₄ rotation-invariance test (the FA machinery is AR-native for listener-centered panoramas — verified applicable, `yaw_rotation.py:389`).

## 6. Arms, statistics (M7 folded)

- Arms: P1/YAW `--cond-method vanilla`; BF `fa_invariant` (default C₄, cap 64); 5 seeds 42–46; batch 64; `--rotate-deg 0 --cond-autocast default`. Optional labelled rows (§8): BV; exp_19 RAF-finetuned as a **transfer diagnostic** with published capture/tx-pose/xyz/placement-level overlap vs its training supports (any overlap ⇒ explicitly not an unseen-source result).
- **Statistics: placement is the clustering unit** (32 placements). Report: every placement; each room; the fixed two-room macro. Arm contrasts: exact item/seed pairing, aggregate within placement first, room-stratified cluster bootstrap + paired placement-level randomization intervals. Seed SD reported separately as Monte-Carlo variability. No item-i.i.d. n=1152 intervals; no generalization claims beyond these two rooms.

## 7. Validation ladder & compute (M10 folded)

Rungs: correspondence/readback record (committed) → amplitude scan over the exact union → audio dry-run (one placement) → canonical mappingA publication → 36-map listener-render rung (measured) → full renders → one full-batch vanilla + one FA smoke (measured) → 3-arm sweep. **All compute quotes are extrapolated from those measured rungs only**; provisional envelope from exp_19/20 measurements: audio ~10k resamples ≈ 30–60 min CPU; renders 1,152 ≈ minutes; vanilla ≈ ~1–2 min/seed + startup; FA ≈ 3–5×; sweep well under an evening on two GPUs; storage ≈ 1.4 GB (NAS).

## 8. Open decisions for Yixun (recommendation first)

1. **Approve Rev 2 protocol** — notably the hash-uniform target rule (M8; the FPS-last spatial-stress variant available as a labelled secondary), the approximate-same-listener labeling (M3), and the within-protocol-only inference scope (M6).
2. **Depth at every target mic** (1,152 listener renders — faithful AR convention; recommended) vs per-placement centroid (72; ~1.4 m误差 inside the rig).
3. **Labelled transfer row** for the RAF-finetuned ckpt with overlap disclosure — recommended YES.
4. **BV as fourth arm** — recommended YES if it is BF's vanilla twin.
5. Seeds 42–46, batch 64 — confirm.
6. **Amplitude contingency pre-authorization:** if the union audit finds files where ×3 clips or stays sub-threshold, I stop and bring you the measured options (per-M1 rule) — confirm this stop-and-ask behavior.
