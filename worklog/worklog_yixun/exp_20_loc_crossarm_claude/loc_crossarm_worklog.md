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
