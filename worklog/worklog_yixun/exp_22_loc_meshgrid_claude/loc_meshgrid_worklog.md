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
