# loc_crossarm_worklog — exp_20 lab notebook (append-only)

## 2026-08-21 — Scaffold, plan, review, APPROVAL (compressed early record)

- Scaffold + query committed; plan Rev 1 → Codex plan review REQUEST-CHANGES (3 BLOCKER / 4 MAJOR) → Rev 2 folded all findings → **Yixun approved Rev 2 verbatim: "Approved Rev 2. Confirm YAW = exp_17 A6000 arm; hold BV@40k; metrics inline everywhere; seeds 42/43/44. Proceed with TDD, code review, pilots, registration freezes, and campaign."**
- Inputs verified: NAS ar_40k_endpoints MANIFEST 4/4 OK (P1 c4c67882…, BF 5319feb4…, YAW ac1f2603…, BV ace9f735… held).
- **Acceptance criteria (TDD phase):** every Rev 2 M4 unit red→green; refusal matrix complete; suite stays green (2,688 baseline + new); Codex code review clean before any probe/pilot executes.

## 2026-08-21T23:45:00-0400 — exp20-r1 delivered; all arms ADMITTED; findings

- **Version Control** — `9967aed 0e5f32e 8587e88 317ed65 0f2ac21 6d1500f`; suite 2,738 green (+50 new); admission records committed for P1/BF/YAW (step 40000, config canonical equality vs committed per-arm configs — NOT FLAC_AR.json (gradient_checkpointing delta); EMA mirror 210/210 ×3, identical inventory digest across arms).
- **Findings of record:** (1) BF's embedded training config carries `cond_method: fa_invariant` + angles ⇒ cond-method refusal is CHECKPOINT-BOUND for exp_20 arms (released EMA remains manifest-bound, honestly recorded). (2) Driver FA cap bug caught by TDD: default cap 64 ⇒ all 4 angles one forward; now cap = candidate micro-batch ⇒ per-angle, plan-flip proven by test. (3) Per-arm --model-config = the arm's committed config file (P1: exp_11 VANCKPT.json; BF: exp_07 FLAC_AR_BF.json; YAW: exp_15 FLAC_AR_YAWAUG.json).
- **Planner confirmations:** metric manifests inherit exp_18's r2_identity_digest + candidate_manifest_sha256 — CORRECT (same split, same candidates). Pilots = existing `--smoke --max-queries 100` (documented, no new flag).
- **Next** — Codex exp20-r1 review → fa-parity real execution + pilots → registration freezes → campaign.

## 2026-08-22T01:15:00-0400 — exp20-r2 delivered (all 7 closed); rulings; re-review launched

- **Version Control** — `0bc70af 6290a72 38a6ab4 9186586 7d6ca1c 5aa0dbb 7eda137`; suite 2,785 green; 97 crossarm tests; all three admission records regenerated + re-ADMITTED at this runtime; all nine manifests generated end-to-end and passing the REAL frozen verifier in test.
- **Planner rulings on the two deviations:** (1) r7 firewall +1 key (`cond_method_binding`) with everything-else-still-fails — ACCEPTED (a tightening). (2) two-tier batch/workers locking (required for arm-manifests; checked-when-present otherwise) — ACCEPTED (retroactive invalidation of exp_18's published registrations would be wrong).
- **Next** — focused re-review (mandatory per r1 reviewer) → BF fa-parity real execution → pilots → freezes → campaign.

## 2026-08-22T03:00:00-0400 — exp20-r3 delivered (four residuals closed); final gate re-review launched

- **Version Control** — `d1a54fa 50a46ad fe6bbe4 c42c232`; suite 2,797 green; 109 crossarm tests. Descriptor-bound deserialization tradeoff (no lazy mmap during admission — no new peak) accepted and recorded.

## 2026-08-22T04:15:00-0400 — r4 micro-round delivered; narrow re-review launched
- `5d4bab0` `1a332bf`; suite 2,802 green; 114 crossarm tests. Ratchet 7→4→2→0-claimed.
