
## 2026-08-22T00:05:00-0400 — APPROVAL: plan Rev 2 approved by Yixun ("approve the plan")
Blanket approval adopts every §8 recommendation: (1) Rev 2 protocol (hash-uniform targets; approximate-same-listener labeling; within-protocol inference); (2) depth at every target mic (1,152 listener renders); (3) RAF-finetuned labelled transfer row WITH overlap disclosure; (4) BV as fourth arm; (5) seeds 42–46, batch 64; (6) amplitude stop-and-ask PRE-AUTHORIZED (union audit violation ⇒ halt + measured options to Yixun). Implementation contracts next; Coder dispatched.

## Readback rung — placement + microphone correspondence (2026-08-22, Coder)

Ran the tested `mappingA_common` clustering/matching over BOTH rooms' full tx-group
populations from the raw archived metadata (read-only). Artifact:
`mappingA_correspondence_record.json`; log `mappingA_readback_20260822_004005.log`;
driver `run_mappingA_readback.py` (thin harness over the TDD'd components).

| | EmptyRoom | FurnishedRoom |
|---|---|---|
| captures / tx-groups | 47,484 / 1,319 | 39,132 / 1,086 |
| groups excluded (size != 36) | 0 | 1 (the recorded 72-capture group) |
| **placements (complete linkage, 5 cm)** | **74** | **91** |
| informal centroid-rounding count | 139 | 121 |
| groups passing correspondence | 892 | 927 |
| groups failing | 427 | 158 |
| **eligible placements (>= 9 passing source-distinct groups)** | **73 of 74** | **86 of 91** |
| median matched displacement (p50 of per-group p50) | 0.75 mm | 0.17 mm |
| median per-group p95 / max | 1.56 mm / 1.72 mm | 0.32 mm / 0.35 mm |
| median ambiguity margin | 116 | 536 |

**Verdict: the registered `n_items = 16 x 36 x 2 = 1,152` identity is ACHIEVABLE** —
both rooms exceed the 16-eligible-placement requirement by a wide margin (73 and 86).

Notes for the record:

- The informal centroid key over-counted placements by ~1.7x (139 vs 74, 121 vs 91),
  which is exactly M2's concern: independent 1 cm rounding of a centroid SPLITS one
  re-occupation across adjacent bins. The re-derived counts are the ones the FPS
  placement selection will draw from.
- The correspondence is strongly bimodal: passing groups match to sub-millimetre
  (median p95 1.6 mm / 0.3 mm), while failures sit at 4-7 cm — i.e. a 5 cm placement
  cluster can still contain genuinely different array positions, and the 1 cm match
  gate is what separates them. 32% of EmptyRoom groups and 15% of FurnishedRoom
  groups fail; they are excluded BEFORE eligibility, never after item construction.
- Placement sizes are very uneven (largest EmptyRoom placement holds 108 tx-groups,
  107 of them passing and source-distinct; many placements hold 36 or fewer). FPS
  over validated placement centroids will therefore have ample choice, but the
  16 selected placements should be reported with their group counts.

## 2026-08-22T06:50:00-0400 — CODE ROUND CLOSED (R1 verified by Planner probes); canonical chain begins
r1→r4 + closing passes: N1-N9, P1-P3, Q1-Q3, R1 all closed; 849 mappingA+raf tests + 487 consumer tests green; correspondence record independently recomputed twice by the reviewer. Residuals: audio-union pin outstanding (dry-run→pin→canonical, next), content-not-byte shared-audio identity, rigid residual recorded. Chain: full-parameter --non-canonical dry run (measures union digest) → pin → canonical prep → 1,152 listener renders → smoke → single-commit 5-arm sweep.
