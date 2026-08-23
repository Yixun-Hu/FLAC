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

## 2026-08-22T04:50:00-0400 — r4 re-review: F3/F4b closed; F4a one-liner → r5; PARITY EXECUTION GO (launched)
- **Acceptance criteria (fa-parity gate, pre-launch):** autocast-off bitwise match=True; registered-autocast within the preregistered 2e-2 tolerance, match=True; evidence record complete (both partitions [10,10,10], per-side dtypes/finiteness, shas); exit 0.

## 2026-08-22T05:25:00-0400 — FA-PARITY GATE PASSED (bitwise both modes); pilots launched; r5 re-review launched

- **Result (parity)** — PASSED all pre-registered criteria and exceeded them: autocast-off max_abs_diff=0.0 AND registered-autocast max_abs_diff=0.0 (bitwise both), record outputs_loc/exp20/exp20_bf_parity_fa_parity.json. First attempt was a Planner shell hang (stdin-blocked cat; disclosed in command.md) — infra, no run occurred.
- **Acceptance criteria (pilots, pre-launch)** — 100-query seen smokes, dumps ON (to *_pilot NAS dirs): complete cleanly; probe components present; per-query means become the campaign schedule basis; BF pilot additionally exercises the executed-partition end gate ([10]x3 per query).

## 2026-08-22T05:55:00-0400 — Pilots + r5 GO; MANIFEST FREEZE; campaign launch

- **Pilots:** P1 1.19 s/q (cell ≈2.1 h), BF per-angle 1.52 s/q (cell ≈2.7 h), peak 1.5 GB. Schedule basis: with inline metrics+dump (measured in exp_18's R4 passes at ≈+1.0 s/q), cells ≈2.2 h (vanilla) / 2.8 h (BF) ⇒ campaign ≈43 h GPU ≈ **22 h wall on 2 GPUs**. BF pilot exercised the executed-partition end gate ([10]×3 per query) clean.
- **r5 narrow re-review: APPROVE/GO** (coercion channel closed; ledger nit batched). FREEZE = the commit adding registrations/ (9 manifests, post-r5 fa_source_shas).
- **Acceptance criteria (every campaign cell):** both registration gates pass at the freeze sha; identity gate 6,337/6,337; BF cells: fa executed-partition gate green per query; dumps complete to the cell's NAS dir; metrics-JSONL publishes after all gates; weights_source resolves "ema".
- **Launch order:** 9 GPU-pairs chained by watcher — (P1-K8, BF-K8) seeds 42/43/44, then (YAW-K8 ×3 + P1-K1 ×3) interleaved, then (BF-K1, YAW-K1) ×3.

## 2026-08-22T05:35:00-0400 — Pair 1 PASSED (P1 0.4946 / BF 0.5085 K8 top-1, paired); pair 2 launched

## 2026-08-22T13:45:00-0400 — K8 P1-vs-BF complete (3 seeds): BF +0.0182 paired top-1, p=0.0002

- P1 0.4948±0.0002, BF 0.5087±0.0007 macro; paired (per-query seed-means, 17-room clustered, stat=mean): top-1 +0.0182 CI [+0.0086,+0.0260] p=0.0002 (12/17 rooms), e_loc −0.090 m p=0.017. Pairing gate PROVEN per seed. Interim label: Holm-4 verdict pending YAW+K1. Pair 4 (YAW-K8-42 + P1-K1-42) running.

## 2026-08-22T21:30:00-0400 — GPU HANDOVER: campaign PAUSED after pair 6 (Yixun's senior needs the GPUs)

- **Decision** — Clean stop: the two running cells (YAW-K8-44, YAW-K1-42) run to completion (~22:40 EDT); **pair 7 is NOT launched**. Zero compute lost; campaign paused at 12/18 cells. Peer session confirms zero GPU footprint on their side (exp_21 held). Resume = pairs 7–9 (P1/BF/YAW K1 seeds 43/44; ~10 h pair-time) on Yixun's release; all registrations remain valid (freeze a92ff5d; the pause has no protocol effect — cells are independent).
