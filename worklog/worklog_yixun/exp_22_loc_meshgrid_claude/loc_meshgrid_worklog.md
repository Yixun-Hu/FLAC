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

## 2026-08-25 — r7 engine delivered (2,989 green); Planner rulings; review launched

- **Rulings on the Coder's flags:** (1) **Noise key = shared_across_candidates (CRN) — REGISTERED**: inherited §1.1's "candidates share … seeds" is explicit, and it matches exp_18's C10 discipline; the dispatched per-candidate key was my error. (2) mean readout deviation stands (reviewer to weigh). (3) fp16 batch-shape nondeterminism (~ulp): batching knobs = advisory binding tier, ACCEPTED — but all production cells pin identical `--batch-rows`/`--source-chunk` in the params file, so cross-cell comparability is by construction. (4) per-query sidecars, bounded dump rule, relpath probe ordering — ACCEPTED. (5) --probe-room ACCEPTED (diagnostics-only). (6) keep mean_scores_hex (inherited N4 mandates it; 1.34 GB fine). (7) fa_invariant refusal fine for P1-first.
- **Known follow-up for the review round:** a production `--rooms` shard filter + a census-gated merge (two GPUs want room-disjoint shards; per-query atomic artifacts + the 5,337-census gate make this safe) — reviewer to treat as in-scope addition.
- Probe fact recorded: 7.67 ms/wf @ batch 64 ⇒ ~152 GPU-h; 83% decode ⇒ registered probe sweeps --batch-rows.

## 2026-08-25 — r8 delivered (all 8; 3,036 green; real-stack leakage guard + cache parity clean); final gate review launched
- Interpretation ACCEPTED: driver refuses per_candidate always (no unregistered driver mode); engine-level opt-in test-only.
- **Acceptance criteria (registered probe sweep, pre-launch):** reviewer's exact commands, batch-rows {64,128,256} on cuda:0; artifacts carry full binding+digest; projection via project_cost against REGISTERED_TOTALS; select smallest batch-rows whose projection minimizes GPU-h; LAUNCH P1 iff projection ≤ 175 GPU-h (else stop → Yixun).

## Erratum — session clock drift (logged 2026-08-25 19:30 EDT)
Earlier entries this session were stamped under a drifted session clock (believed 2026-08-18).
System `date` verified 2026-08-25; all exp_22 I1 execution (D1 manifest 02:17, G1 audit 16:41,
cache parity / probe smoke ~17:38, r8b fixes + push `1e3ed22`, probe sweep launch 19:25) occurred
on **2026-08-25 EDT**. Log filenames carrying other dates in their stems name the same runs.
Per standing rule: verify `date` before stamping.

## I1 probe sweep verdict + batch-rows selection (2026-08-25 19:5x EDT)
Registered sweep (Bathrooms_idx_14, 10 queries / 1 receiver group / 32 union rows / 2,208 waveforms per config) complete; diagnostics copied beside this worklog. **Independent rederivation from raw per-record timings (4 buckets: waveform-scaled sampling+decode+embed+scoring x 71,172,320; pair-scaled conditioning x 8,896,540; union-row-scaled source cache x 966,147, deduped per receiver_id with intra-group stamp-equality asserted; per-query context x 5,337) agrees with the engine projection to 0.1 GPU-h on all three configs:**
- br64: 157.8 GPU-h (7.904 ms/wf)
- br128: 156.0 GPU-h (7.815 ms/wf)
- **br256: 148.4 GPU-h (7.438 ms/wf) — unique minimum -> WINNER (smallest batch-rows attaining min)**
Gate: 148.4 <= 175 GPU-h -> **P1 LAUNCH AUTHORIZED** per pre-registered criteria (worklog r8 section) under Yixun's "2b: ii — run P1 first". Wall estimate: Cafe shard 51.2% of pairs -> ~76 h -> results ETD ~Aug 29 morning. Sample-size caveat: projection extrapolates one receiver group; the engine re-stamps per-row batching so drift is auditable in the full pass.

## P1 full pass LAUNCHED + R1 Coder round dispatched (2026-08-25 19:55 EDT)
- **P1 launch** (per auto-launch rule, br256): RUN_TAG `P1_CRN_br256_20260825_194053`, reviewer's verbatim call 2. Cafe/GPU0 (4,559,398 pairs) + rest15/GPU1 (4,337,142 pairs); master script runs the mandatory census-gated merge after both shards (merge NOT run on any nonzero shard exit). Both shards passed startup gates: Seed 42 + "release call graph reproduced ... RNG state matches the D1 pass". Logs: `outputs_loc/exp22/i1_P1_CRN_br256_20260825_194053_{cafe,rest15}.log`, master `loc_meshgrid_p1_full.log`. Wall ETD ~76 h → results ~Aug 29 morning.
- **exp22-r9 (R1 reporting) dispatched to Coder** (Opus, max effort): `meshgrid_report.py` (census/digest gates → e_loc/e_oracle/e_excess, raw+oracle-normalized success @0.5/1.0 m, room-first aggregation, 95% room-bootstrap CI seed 20260825×10k, random baseline seeds 101–105, LME+S_mean cross-check vs stored argmax, quantile viz-case selection) + `meshgrid_offgrid_probe.py` (16 off-grid truth probes + real-vs-generated calibration, waveform dumps per announcement 08) + TDD. New files only; engine untouched (live run). Codex review follows delivery.

## r9 delivered; Codex r9 review + r9b dispatched (2026-08-25 ~21:45 EDT)
- **r9 (Coder, Opus max)**: `meshgrid_report.py` (1,644 L) + `meshgrid_offgrid_probe.py` (704 L) + 2 test files; full suite **3,142 passed / 11 skipped** (+98 over r8b); commits `bc1ebea` `2d44d8e` (new-files-only, live run untouched), pushed. Key resolutions logged by the Coder: (1) G1 manifests carry no continuous x*_s → post-hoc resolver via `candidates.find_pair_metadata` with dual cross-checks (e_oracle ==1e-9; rec_loc ==1e-6), new `--metadata-root`; (2) rows carry generation-loop timing only → `LATENCY_SCOPE_NOTE` disclosure (merge drops run_summary timings); (3) float16 sidecar vs float32 row scores → argmax-flip policy `explained` (default, names flips excusable at margin≤2·ulp-deviation; inflated-margin spoof detector) vs `strict`; (4) random baseline keyed sha256(seed,query_id) → walk-order independent; (5) fixture-only `require_manifest_census` relaxation, CLI always strict; (6) §2 sparse AGREE-retrieval control NOT in r9 scope → reports name it outstanding; (7) run_probe reads room manifests twice by design.
- **Codex r9 review launched** (gpt-5.6-sol xhigh, read-only, install/modify forbidden), scope bc1ebea+2d44d8e; verdict → `loc_meshgrid_codex_code_r9_review.md`.
- **r9b dispatched** (Coder, Opus max): the missing §2 control — sparse/metadata-bank AGREE retrieval (real RIRs, same-receiver-other-sources bank, own-pair excluded, sparse-bank oracle, same metric family + bootstrap) + ingest handoff into the r9 report; only permitted existing-file edit: the controls_elsewhere wording.

## r9b delivered (2026-08-25 ~23:30 EDT)
Sparse/metadata-bank AGREE-retrieval control built + 76 tests; full suite **3,218 / 11 skipped / 0 fail**; commits `07a7242 dc46a70 9ba13cf b3f08e0`, pushed. Registered bank rule `numeric_identity` (candidates.py convention); `released_eligible_pool` kept as sensitivity check (S010 invisible to the released selector — bank=eligible+1 for 4,593/5,337 queries, disclosed as mildly favoring retrieval). Membership = pair metadata ∧ wav exists (901 missing wavs counted, Cafe_idx_1 + LRWH_idx_30). Real-data read-only validation: 5,337 banks sizes 7-9, zero refusals; **sparse-bank oracle median 1.470 m (88.8% > 0.5 m)** — never comparable to the dense-grid oracle, labeled so. Truth-pin gate vs real G1: max |Δoracle| = 0, receiver drift 0. Control NOT yet run (needs AGREE ckpt walk) — report says "run pending". r9b also pre-closed 3 Codex-r9 finding-patterns inside its own scope (G1 dup/partial join, truth authentication, registered-settings banner); offgrid_probe/report fixes remain r9c's.

## r9c delivered (2026-08-26 ~00:45 EDT)
All 9 Codex r9 findings + 4 minors closed; suite **3,260 / 11 / 0** (+42); commits `0a06416 f92cb26`, pushed. Highlights: hash-join of supplied D1/G1/room manifests to the binding's pins + REQUIRED merge_report re-join (`--single-shard` relaxes only the merge report, stamped non-canonical); exact 5,337 identity join D1≡G1≡rows before any metric; TruthResolver metadata-bank digest (recorded-then-pinnable via `--expect-metadata-bank-sha256`) + full-vector truth check in the probe (scalar-oracle non-injectivity documented); probe applies the full report ladder + `assert_published_matches`; gate-before-device; `REGISTERED_PROTOCOL` enforcement (all ten constants + agree/model-config shas verified to match the live P1 binding; `ckpt_sha256` recorded-not-pinned per Yixun 2d — plan §1.4's EMA-extract literal would refuse the admitted wrapped ckpt); `argmax_flip_within_2dev` rename + float16 dtype + absolute half-ulp bound; room-first latency w/ named missing components; staged-then-published NPZ dumps with embedded labels. Fixture rewritten to be exploit-capable (per-room shards through the engine's own merge_shards). Cross-pin test locks rc/mr `assert_grid_oracle` to identical verdicts (collapse deferred).

## Codex r9c re-review: REJECT, narrowing (2026-08-25 ~23:05 EDT) — and two Planner rulings
Resolved: B2 + 3 minors. Residual PARTIALLYs (B1 copyable merge receipt, B5 import-time cuda default, M6-M9, one disclosure) + B3 NOT (TOFU digest ≠ origin) in report/probe; 3 new BLOCKERs + 2 MAJORs in the r9b retrieval control (unbound bank inputs; scalar-spoofable truth claim; model_config_sha256 omitted; K=1 LME float path; binding semantics). Review saved `loc_meshgrid_codex_code_r9c_review.md`. Fix rounds r9d (report/probe, r9c author) + r9e (retrieval control, r9b author) dispatched in parallel — disjoint files.
**PLANNER RULING 1 (M6 pin + worklog erratum):** the entry above dated 00:40 said the admitted wrapped ckpts would be "hash-checked against their EMA extract on rsync arrival" — that rsync never arrived; Yixun later resolved identity by authority ("P1_40k_clean_hybrid_EMA.ckpt is our trained P1 40k checkpoint") and decision 2d admits our wrapped 40k ckpts. Therefore the REGISTERED admissible-arm ckpt registry is pinned to the shas of `weights/exp20/{P1,BF,YAW}_40k.ckpt` (P1 = `c4c67882…`, already the live binding); canonical reports refuse any other ckpt sha without an explicit deviation flag. Dataset-config sha joins the registered set.
**PLANNER RULING 2 (B3 origin):** full independence from the AR pair-metadata tree is impossible — the tree IS the truth authority (loader md['source'] and G1 both derive from it). The honest closure is PRE-REGISTRATION, not provenance: the metadata-bank digest (and the sparse-bank wav-bytes digest) are computed on the real tree and committed BEFORE the P1 merge exists or any localization quality is read, making post-hoc adversarial selection impossible; the canonical report then REQUIRES the pre-registered digest (TOFU mode demoted to non-canonical). r9d/r9e implement digest CLIs; the main session computes and freezes the values immediately after they land.

## r9d delivered; PRE-REGISTERED DIGESTS FROZEN (2026-08-25 ~23:20 EDT; file stamp UTC)
r9d closed all 8 residuals + both rulings (ckpt registry P1/BF/YAW pinned by real digest; dataset-config sha registered; receipt re-derivation from rows + independent G1 source-row derivation; both-neighbor float16 ulp; canonical-latency completeness; fsync-then-publish with rollback). Commits `52cb570 cda347e`; suite **3,318 / 11 / 0** (localization slice at HEAD: 411/1). **Digests frozen before the P1 merge exists** in `loc_meshgrid_preregistered_digests.json`: metadata bank `9f1322e5…` (5,337 pair files), sparse bank `39f0a119…` (47,132 entries; reproduced twice independently). Codex verify pass r9f launched over r9d+r9e.

## Codex r9f: REJECT, minimal blocking set of 3 (2026-08-26 ~00:15 EDT)
RESOLVED: B2 B5 M6 M7 M8 + disclosures + retrieval model_config + K=1 cosine. Blocking: (1) probe gate lacks derive_run_facts/uniform-batching and empty batching stamps pass via `if found`; (2) probe stores --expect-metadata-bank-sha256 without computing/comparing, and the CLI drops the expectation at publication (JSON/md vs NPZ canonicality divergence); (3) retrieval digests bytes under --dataset-root but scores obs from the dataset-config root (divergent roots), and re-reads truth/bank bytes post-gate without hash comparison; plus M9 os.replace outside rollback; input-surface device/alpha/loader-config residuals. r9g (probe/report) + r9h (retrieval) dispatched.

## r9g + r9h delivered (2026-08-26 ~01:10 EDT)
All r9f blocking items + residuals closed. Suite **3,358 / 11 / 0** at final HEAD; report+probe 184 tests, retrieval control 130. Frozen pre-registered digests UNCHANGED by the refactor (sparse bank recomputed = 39f0a119…, roots agree on the released config, loader values == REGISTERED_LOADER). Final Codex verify pass r9i launched.

## Codex r9i: REJECT, 4-item minimal set (2026-08-26 ~02:20 EDT)
12/18 rows RESOLVED. Blocking: (1) pair-JSON verify-then-REOPEN (retrieval :1278, report resolver :924) — parse the verified bytes; (2) off-grid live obs_wav unbound to the obs behind the frozen grid rows; (3) NPZ rename/bookkeeping crash gap; (4) fail-open canonicality joins (--non-canonical ignored in JSON/md path; retrieval top-level canonical ignores walk-derived digest_verified). r9j (report/probe) + r9k (retrieval) dispatched.

## r9j + r9k + r9j2 delivered; THIRD DIGEST FROZEN (2026-08-26 01:47 EDT, `date`-verified; earlier ~04:15 stamp was drift)
All four r9i items closed + observation pin added. Suite **3,394 / 11 / 0** (localization slice 487/1). Observation bank `ee2ba80a…` frozen in `loc_meshgrid_preregistered_digests.json` (independently reproduced twice; chronology note: added after the first freeze but still before the P1 merge exists / any quality read — genuine pre-registration). Canonical probe runs now require BOTH --expect-metadata-bank-sha256 9f1322e5… AND --expect-observation-bank-sha256 ee2ba80a…. Final Codex verify r9l launched over r9j/r9k/r9j2.

## Codex r9l: REJECT, 3 items — read-once family, report/probe only (2026-08-26 ~02:20 EDT)
Retrieval control fully RESOLVED (all rows). Blocking: (1) probe's fresh resolver consumes runtime truth bytes post-gate; (2) observation pin reopens after decode + one-candidate functional tie non-injective — need single-buffer hash→decode; (3) NEW: report rows/sidecars verified-then-reopened. r9m dispatched (report/probe author).

## r9m delivered (2026-08-26 ~02:55 EDT)
All three r9l items closed: probe truth from frozen-bank-verified buffers; observation byte->tensor single path (decode bit-identity to the released loader verified on real data); report rows+sidecars single-buffer parse cross-pinned to the engine's verifiers. Suite **3,409 / 11 / 0** (slice 502/1). All three frozen digests recomputed UNCHANGED. Verify pass r9n launched.

## Codex r9n: REJECT, ONE blocker (2026-08-26 ~03:30 EDT)
Items 1-2 RESOLVED. Sole remaining: probe CLI census verifies-then-discards rows/sidecars; the walk reopens them — a coherent row+sidecar replacement between phases can pass. r9m2 dispatched (bind walk reads to the census snapshot digests).

## r9m2 delivered (2026-08-26 ~04:00 EDT)
Last blocker closed: census captures per-artifact BYTE digests (not self-recomputable claims-digests — proven by an exploit test that a coherent swap verifies against itself); every walk read bound to the snapshot. Suite **3,415 / 11 / 0** (slice 508/1). Frozen digests unchanged. Verify r9o launched.

## R1 REPORTING STACK APPROVED (2026-08-26 ~04:30 EDT) — review campaign closed
Codex r9o: **RESOLVED — APPROVE.** The stack (meshgrid_report / meshgrid_offgrid_probe / meshgrid_retrieval_control) may consume the P1 merged run as canonical with the three frozen digests (metadata 9f1322e5, sparse bank 39f0a119, observation ee2ba80a). Campaign r9→r9o: 8 fix rounds, 6 review passes, suite 3,044 → **3,415** (+371). Stack now FROZEN until the merge lands. On merge: (1) canonical report with both pins + --expect-ckpt-sha256; (2) off-grid probe run (GPU) with all three pins; (3) retrieval control run (GPU) with its pin; (4) quantile viz cases + results/analysis/HTML with subset + leakage labels; then Yixun's BF+YAW option-ii decision.
