# loc_meshgrid_worklog — exp_22 lab notebook (append-only)

## 2026-08-25T00:40:00-0400 — Yixun's decision set (verbatim in query file); assets fetched; kickoff

- **Decisions (2026-08-24/25):** 2a fetch all 16 official OBJs + rsync frozen exp_09 artifacts for byte-parity; **2b option ii — P1 arm first, then decide BF+YAW**; 2c scorer = inherited `AGREE_fullAR` pin (**leakage caveat, stated everywhere: fullAR saw the full dataset incl. unseen rooms; acceptable here because the scorer is frozen, identical across arms/candidates, and pinned by the approved exp_09 protocol — but absolute levels are not leak-free and must never be compared against AGREE_AR-scored exp_18/exp_20 rows without that label**); 2d our admitted wrapped 40k ckpts, hash-checked against their EMA extract on rsync arrival; 2e announcement-08 exemption approved — dump ONLY the 16 off-grid probe queries + quantile-selected visualization cases (all sims/scores logged as always).
- **Assets:** all 16 room OBJs fetched from official `clean-main` to `/media/diskstation/yixunhu/FLAC/AcousticRooms/room_mesh_obj_format/` (shas in fetch log; Cafe byte-identical to Yixun's upload, e7a0b7b9…). `ListeningRoom_idx_2.obj` confirmed 404 on clean-main ⇒ inherited 5,337/16 subset stands. Open3D 0.19.0 already in env.
- **Awaiting Yixun's rsync** (destination `/media/diskstation/yixunhu/FLAC/checkpoints/exp22_exp09_parity_artifacts/`): ① exp_09 frozen context manifest (D1, content-hashed); ② G1 candidate manifests + mesh-audit report; ③ `P1_40k_clean_hybrid_EMA.ckpt` (sha `da127485…`). Not blocking code rounds; blocking only the byte-parity cross-checks.
- **Next** — Coder round exp22-r1: port D1 (exp_01-RNG context materializer) + G1 (geometry primitives) per the inherited plan's per-function test lists, adapted to this repo's paths/announcement-02 conventions.

## 2026-08-25T01:15:00-0400 — exp22-r1 delivered; Planner ruling on the anchor-prior ambiguity

- **Version Control** — `9b362a2` (D1) `e05d0de` (G1) `76d45d4` (ledger); full tree **2,840 passed / 0 failed**. Real D1 census reproduces the inherited histograms EXACTLY ({6:91,7:429,8:5263,9:554} → {6:91,7:429,8:4363,9:454}); real Cafe G1 smoke: all anchors parity-valid, sources clear the prior (min 0.550 m), lattice 9,996 → 6,273 valid.
- **RULING (anchor prior):** the 0.20 m clearance applies to SOURCES ONLY — adopted as the registered reading. Grounds: §1.2 names it a *source-distribution* prior; §1.3 rule 3 restricts the candidate predicate to *source* anchors while rule 2 asks only free-space classification of all anchors; physically receivers legitimately sit near walls (2/100 Cafe receivers at 0.100 m); and the exp_09 checkout's own G1 ran and was approved with Cafe included, implying their implementation read it identically. The rsynced exp_09 mesh-audit report will verify this equivalence on arrival (recorded as a pending cross-check).
- Also accepted: `self_intersecting: None` (O(n²) infeasible at 366k tris; disclosed), EXCLUDED_ROOM naming per the split file.
- **Next** — Codex exp22-r1 review → D1 frozen-manifest generation (full pass, teed) + G1 16-room audit → post-G1 cost gate to Yixun.

## 2026-08-25T02:05:00-0400 — r1 review: 4 BLOCKERs; RSYNC NOW GATES G1

- F1 (worker-RNG call graph) and F2 (substitution guard) are code-fixable now. **F3 changes the rsync's status: the inherited plan freezes "31 directions" by reference, not value — our locally generated set fails a real anchor (MeetingRoom_idx_32 receiver, 15/31 odd votes at 0.250 m clearance) that exp_09's approved G1 evidently passed. Their frozen direction set (or G1 code/audit) is REQUIRED for parity; without it we can only pin our own set and re-open the anchor rule.** F4 = the audit/cost-report driver, spec'd, buildable now.

## 2026-08-25T03:50:00-0400 — r2 delivered; Planner verification + rulings

- **Version Control** — `6f169c5 9eef028 a9f796b bc790f5`; tree 2,863 green. Call-graph parity proven on real data (identical RNG state + fingerprints + context-audio digests across ours vs the eval_FLAC-faithful reference; counter-test bites). Directions pinned literally at the reviewed digest `9ab4339f…`; MeetingRoom discrepancy encoded as a loud, unresolved, room-blocking marker.
- **Rulings:** ANCHOR_TOLERANCE 1e-3 for the context-join ACCEPTED (recovery join, not a geometric boundary; separately named). AGREE-path resolution recording ACCEPTED (verified on disk: the configured `weights/AGREE/AGREE_fullAR.pt` exists here, so resolution = "configured" on this box; the fallback guard is for other checkouts).
- **Next** — focused r2 re-review → D1 full-pass manifest (CPU) → 16-room audit stays BLOCKED on the exp_09 rsync (MeetingRoom cross-check).

## 2026-08-25T06:35:00-0400 — r3 delivered; D1 FULL PASS LAUNCHED (reviewer-preauthorized)

- **Version Control** — `d061f64 1d1e491 22c512d`; tree 2,877 green. Planner rulings: the inf/NaN branch-rule distinction ACCEPTED (meaningful-empty-band inf disqualifies; NaN refuses — both pinned).
- **Acceptance criteria (D1 pass, pre-launch):** enumeration gate 6,337 unique ordered identities; per-position idx+relpath guard clean; completed census exactly {6:91,7:429,8:5263,9:554}; filtered stream 5,337 with only ListeningRoom_idx_2 removed; manifest hashes recorded (full + filtered); byte-stable reload check. Cross-check vs exp_09's manifest: PENDING the rsync (recorded).
- **Result** — launched (CPU; log loc_meshgrid_<TS>_d1_manifest.log). Focused r3 re-review running in parallel; any F2 finding ⇒ regenerate (cheap, deterministic).

## 2026-08-25T04:05:00-0400 (box ~02:20) — r4 delivered; D1 REGENERATED (hashes stable); final re-review launched

- `3b27412 6297e88`; tree 2,889 green. D1 manifest regenerated post-F2-fix: full `15d229c0…`, filtered `99f8da60…` — identical to the superseded pass (comparison fix, not draw change; correctly reasoned), census asserted, reload verified, AGREE resolution "configured".

## 2026-08-25T02:50:00-0400 — r5 delivered; code phase COMPLETE pending rsync

- `60e91e4 b0314d0`; tree 2,892 green. Staged-verify-then-atomic-publish implements the r4 reviewer's precisely-stated condition; closed under Planner verification (precedent exp_18 r5b), with the next Codex review (audit outputs at the cost gate) covering this commit again in situ.
- **exp_22 state:** D1 manifest ACCEPTED + committed; geometry primitives + fail-closed audit driver review-hardened through 5 rounds; 16 meshes on NAS; **sole blocker = Yixun's rsync** (direction set → MeetingRoom ruling → audit → cost gate).

## 2026-08-25 — Yixun clarification: P1 checkpoint identity resolved

- **Yixun (verbatim):** "P1_40k_clean_hybrid_EMA.ckpt is our trained P1 40k checkpoint, you use the same checkpoint for the P1 vanilla 40k."
- **Consequence:** exp_22's P1 arm = our admitted wrapped `weights/exp20/P1_40k.ckpt` (sha `c4c67882…`), EMA-resolved at load — the same weights as the inherited plan's clean-EMA extract (their file = the EMA branch of this training run, exported). Rsync item ④ DROPPED from the blocking list; an EMA-tensor cross-check against their extract remains an optional nicety if the file ever arrives.
- **Remaining rsync/transfer needs (shrunk):** ① the frozen 31-direction constants (DECISIVE — gates the MeetingRoom ruling + G1 audit); ② their D1 context manifest + G1 candidate manifests/mesh-audit report (parity cross-checks). All small files — any channel works, incl. Zhixuan pasting the constants.

## 2026-08-25 — Yixun directive (relaying Zhixuan): SELF-AUTHORITATIVE; finish the experiment

- **Yixun (near-verbatim):** the 31 directions exist only to test interior free space (≥16/31 odd parity = inside); no need to get them from Zhixuan — define our own; the two JSONs are OUR generated artifacts. "Just go ahead and finish this experiment."
- **Consequences:** (1) the pending exp_09 parity cross-checks are RESOLVED-BY-AUTHORITY — exp_22's D1 manifest and geometry artifacts are the registered originals; (2) the MeetingRoom_idx_32 anchor discrepancy is OURS to resolve, pre-generation: **registered rule = choose the direction set by a deterministic, anchor-driven selection — the smallest generator seed whose 31-direction set passes strict-majority parity for EVERY metadata source+receiver anchor in all 16 rooms** (anchors are known-interior points; selecting geometry constants for classifier self-consistency BEFORE any FLAC generation is pre-registration, not tuning — same class as the plan's own pre-generation z-branch rule); (3) the ≥16/31 strict-majority rule is confirmed verbatim by Yixun.
- **Cost-gate interpretation (stated for veto):** Yixun already approved P1-first (~140 GPU-h ≈ 3 days) and now says finish; the audit's measured projection will be compared against that envelope — proceed without another stop if within ~125% of it, stop and ask if above.

## 2026-08-25 — r6 delivered (seed 1 frozen; 700/700 anchors pass); G1 AUDIT LAUNCHED

- `910ead7 96ba96e`; tree 2,900 green. Old set = build_directions(seed 0), failing exactly the one reviewer-found anchor at 16-room scale; seed 1 passes all 700 (MeetingRoom receivers at exactly 16/31 — borderline-interior, recorded as a robustness caveat). Selection reproducible (`select_direction_seed.py`), report committed.
- **Acceptance criteria (G1 audit):** all 16 required rooms accepted; every query nonempty with finite full-height oracle; both z-branch distributions computed honestly (band ∞ counted); the pre-registered branch rule applied globally; staged-verify-then-atomic publish; cost report with the four gate numbers. NOTE: the earlier candidate-grid PNG used the seed-0 set; regenerate under seed 1 post-audit.

## 2026-08-25 — G1 AUDIT PASSED; POST-G1 COST GATE

- **Audit:** 16/16 rooms accepted (seed-1 directions), 5,337/5,337 queries nonempty with finite oracles; staged-verify-then-atomic publish clean. **Branch rule selected z_band** (identical over-threshold count to full height: 50 queries with e_oracle>0.5 m, 0.94% — the pre-registered no-new-unwinnable condition holds). Oracle median 0.241 m both branches.
- **Gate numbers (chosen z_band):** 8,896,540 candidate-query pairs; 966,147 unique receiver-candidate pairs ⇒ ~966k source-conditioner calls (receiver-union cache); artifacts 284.7 MB. Full-height comparison: 15.73M pairs (z-band saves 43%).
- **Cost projection vs the approved P1 envelope (~140 GPU-h):** 71.2M generated waveforms (pairs × K=8 nested). At the inherited plan's measured rate (~7 ms/waveform, large batches) ≈ 140 GPU-h — ON the envelope; at exp_20's small-batch rate (~13 ms) ≈ 257 GPU-h — over. **The binding decision stays with the pre-registered throughput probe** (cache-enabled, no-quality, ladder step): proceed if its projection ≤ 175 GPU-h (125% envelope, per the stop-rule Yixun saw), else stop and ask.
- JSON reports copied to g1_audit_reports/ (npz coordinate sidecars stay in outputs_loc, hashes committed via the reports).
