**Reviewer:** OpenAI Codex (gpt-5.6-sol, xhigh, codex exec, read-only sandbox) · **Date:** 2026-11-06 · **Round:** exp_37 code review 1 (round 1 of 3; dca796dd → 41e8576d + the Planner's exp_37 records scripts)

**Verdict:** REQUEST-CHANGES

## BLOCKING

1. **Required predicted-waveform artifacts are entirely missing.**

   Announcement 08 requires every generative localization/eval run to save the exact generated, decoded, clamped tensors that were scored, as float32 `[M,K,10240]`, with per-query path/SHA and a dump manifest (`worklog/worklog_yixun/announcement/08_save_pred_waveforms.md:7-10`). The runner currently saves only candidates and similarities (`src/localization/runner.py:219-236`); the exact `generated` tensor is discarded after scoring (`src/localization/runner.py:402-425`). This affects G0, B1/B2 accuracy runs, B3 localization, nested/c2f timing runs, and the separate synthesis work.

   **Fix:** add a write-once waveform artifact contract: per-query `.npy` `[M,1,10240]` float32 containing the exact output of `prepare_generated_audio`, per-query SHA/path in the hashed query record, candidate/full-grid index mapping, and a hashed dump manifest bound to the run manifest, checkpoint, seed, backend, and query coverage. For already completed runs, announcement 08 permits deterministic regeneration only if the regenerated similarities reproduce the published per-sample similarities bitwise.

   Copying/dumping after the measured batch window is not automatically timing-neutral: D2H copies, page-cache pressure, memory residency, disk I/O, and altered idle/thermal history can perturb later batches and queries even if not charged directly to the timer. Preserve publishable timing with a separate deterministic artifact replay linked to the timing run, or obtain an explicit timing-only exemption from Yixun. Do not add dumping to the guarded headline timed region.

2. **The c2f Pareto time is not statistically comparable to the other points.**

   The ordinary points use a median-of-three selector vector (`tools/exp37_selector_latency.py:58-81`), but c2f performs AGREE inline once and sums that single observation into `c2f_total` (`src/localization/exp37_c2f.py:57-119`); the aggregator consumes it directly (`tools/exp37_aggregate_b2b3.py:93-101`). This violates §10’s “same definition for every point” and median-of-three selector requirement (`plan_loc_efficiency.md:134-139`).

   **Fix:** retain one generator execution but measure each c2f selector component three times on the exact generated tensors: observed encoding, coarse generated encoding/similarity, top-k/neighbour selection, refined encoding/similarity, and union argmax. Record repeats and use component/per-query medians, asserting stable scores/top-k/winner. Otherwise exclude c2f from the registered Pareto comparison and label it exploratory. Disclosure alone is insufficient for the registered figure.

3. **Co-tenancy is not actually fail-closed over the whole timed interval.**

   Only start/loaded/end snapshots are written (`src/localization/runner.py:608-620,730-732`). A co-tenant can start and finish between them undetected. CPU processes are truncated to the top 20 (`src/localization/run_provenance.py:37,108-124`) and are never part of `timing_valid`; the verdict checks only sampled GPU apps (`src/localization/exp37_latency.py:111-123`). Host-record JSON is also unhashed and not checked against its run-manifest SHA (`src/localization/run_provenance.py:137-148`).

   **Fix:** require a continuous or sufficiently frequent run-bound monitor/attestation covering the entire timing interval, with no observation gaps or command failures; bind every sample cryptographically to the run manifest and require the run PID on its card throughout GPU residency. Add an explicit, verifiable host gate that exp_36 has ended, and record the full relevant CPU-process set. Selector measurements need the same rule.

4. **The B2/B3 aggregator can emit an incomplete or non-publishable headline figure.**

   `_stats` accepts any seed set (`tools/exp37_aggregate_b2b3.py:59-61`); B2 and B3 never require the registered `{42,43,44}` coverage or the complete arm/grid/orbit/θ Cartesian products (`:64-119,131-156`). Missing cells can therefore disappear while the CLI exits successfully. Separately, invalid timing points are still included in the frontier (`:113-116`) and plotted (`:231-262`); JSON/Markdown/figures are written before exit 3 (`:298-310`).

   **Fix:** require exactly:

   - B2: two arms × `{0.50,0.75}` × three seeds, plus both nested and c2f timing passes.
   - B3 accuracy: three orbits × two θ values × three seeds.
   - B3 latency: all three registered orbits, exactly once.

   Require `n=3` for every accuracy statistic. Never calculate or draw a publication frontier using invalid timings. Either refuse publication outputs before creating the figure or produce a clearly separate diagnostic artifact containing no frontier.

5. **The runbooks are not yet an exact, safely executable command sequence.**

   - B3 lists only the seed-42 spelling and says “×3”; the seed-43/44 commands and names are absent (`exp37_runbook_bc.md:85-91`).
   - The proposed aggregation JSON contains placeholder strings such as `"… 12 entries"` inside arrays; it is not a usable aggregator spec (`:108-122`).
   - The selector dirty-tree check only prints a refusal and continues, while the trailing `echo` masks the selector’s exit status (`exp37_runbook_bb.md:16-22`). The B-c selector loop has the same exit-masking problem (`exp37_runbook_bc.md:43-50`).
   - The experiment’s eight new B3 synthesis commands and exact reused-cell paths are absent; only a vague separate-leaf note is given (`exp37_runbook_bc.md:135-138`).

   **Fix:** enumerate every command with its exact seed/name in execution order; commit a complete machine-readable aggregation spec or deterministic spec builder; capture and return the actual command status; abort on dirty tree/failure; and provide the exact H35 leaf commit, runbook commands, and reused artifact paths for B3 synthesis.

## HIGH

1. **Summary loaders do not enforce the full registered provenance.**

   `TIMING_IDENTITY`/`_arm_identity` omit the model-config SHA, AGREE SHA, sample seed, sampler, no-flash launcher/overlay identity, mesh digest map, and several selection bindings (`src/localization/exp37_latency.py:22-29,135-166`). Thus B1/B3 latency can accept a wrong-seed or wrong-AGREE run. `load_selector` likewise ignores its recorded AGREE checkpoint digest and context/selection provenance (`:67-85`).

   **Fix:** centralize the complete registered identity and compare every field used by G0, including the expected AGREE/model-config hashes, seed 42 where required, sampler, audit/context/pilot hashes, SDPA launcher digests, mesh map, and timing flags. Validate the selector’s AGREE/context hashes and permitted selection-superset relationship.

2. **The c2f artifact misstates and under-validates its candidate identity.**

   `_build_query_result` copies the full pilot’s `candidate_indices_sha256` (`src/localization/runner.py:268-287`) even when the c2f result contains only the union (`src/localization/exp37_c2f.py:110-127`). `load_c2f_run` checks only the union-list length, protocol name, and top-k (`src/localization/exp37_latency.py:236-257`); it does not recompute the union SHA, validate ordering/range/radius, confirm `generated_count`, or bind stored candidates to `full[union]`.

   **Fix:** record parent full-grid identity separately and make the active candidate hash describe the actual union. Recompute and validate the union index SHA, sorted uniqueness/range, radius `0.5`, generated/union counts, candidate array, and winner.

3. **External and aggregate source contents are not bound into the summary.**

   The exp_23 JSON and latency CSV are read without validating/pinning their content hashes (`tools/exp37_aggregate_b2b3.py:159-166`). The resulting summary mostly binds path strings through `spec`, not an immutable inventory of source hashes.

   **Fix:** validate the exp_23 canonical SHA, record the CSV byte SHA, and include all run-manifest, query-artifact, selector, subset/audit/pilot, and host-attestation SHAs in the summary payload.

4. **B1 seed-42 reuse is valid for accuracy but not for the registered B2 timing source.**

   The 64 pilot records are an exact ordered subsequence of the 128 selection with identical query/candidate identities, so deterministic seed-42 accuracy can be read from B1. However §4 pre-registers six 64-query 0.5 m runs and their own timings (`plan_loc_efficiency.md:86-93`); the 128-query execution has different intervening queries and thermal/runtime history. Current loaders also reject this reuse because they require an exact pilot SHA/query list (`src/localization/exp37_latency.py:135-166`; `src/localization/exp37_grid.py:77-86`).

   **Fix:** if saving GPU time, add an explicit validated-superset mode for accuracy only, including waveform coverage. Keep dedicated B2.1/B2.2 seed-42 timing runs unless Yixun amends the registered timing protocol before execution. Qualify the current proposal at `exp37_runbook_ba.md:135-137`.

## NIT

1. **“Reproduces all 64 recorded K=1 metrics” is stronger than the check.**

   The readout compares only `prediction_index` and `localization_error_m` (`src/localization/exp37_grid.py:92-96`), while the notebook claims all recorded K=1 metrics (`loc_efficiency_worklog.md:110-116`).

   **Fix:** compare oracle error, excess, success flags, resolution-aware success, prediction coordinates, and score fields too, or narrow the claim.

2. **The optional K_ctx=1 split wording conflicts with the fixed summarizer command.**

   The runbook says K1 splits may be dropped (`exp37_runbook_bb.md:43-46`), but its only summary command requires all six split directories (`:60-66`).

   **Fix:** provide an explicit alternate invocation without the optional `--split` arguments.

## Question verdicts

| Question | Result | Review |
|---|---|---|
| Q1 B1 fidelity | **PASS** | The guarded headline region remains bundle-equivalent; the `frame_angles` and default full-range `candidate_indices` deltas are acceptable (`src/tests/test_exp37_headline_timer_guard.py:96-139`). K_ctx=1 preserves the eight-context filter (`runner.py:61-76`). Selector/statistics formulas match the bundle (`exp37_latency.py:43-64`). Stage boundaries and max-not-sum memory handling are correct (`stage_split.py:85-209`). The code currently gates the split strictly on per-query **and** aggregate 2%, not aggregate-only (`exp37_latency.py:195-213`). Keep the strict gate: aggregate-only contradicts binding §10. If jitter withholds the split, report the comparison but do not publish the breakdown. |
| Q2 B2 | **FAIL** | Nested subset, k=5/ties/Chebyshev refinement/fresh oracles are correct (`exp37_grid.py:33-66`). The 0.75 m artifact is pinned to z-band with 5,337 queries and the same 64-query pilot. The failure is the single-repeat inline c2f selector timing described above. |
| Q3 B3 | **PASS** | Immutable base, once-rotated query, candidates built from base then rotated once, global-frame metrics, whole-column gate, and θ=0 headline dispatch are correctly implemented (`exp37_rotation.py:34-102`; `runner.py:549-556,628-637`). Frame-angle identity is recorded and checked. |
| Q4 G0/provenance/exclusivity | **FAIL** | G0 pins/query/threshold/YAW-first logic pass (`tools/exp37_g0_compare.py:53-67,117-202`); no-flash refusal and mesh remap pass (`exp37_no_flash_launch.py:166-197`; `mesh_remap.py:27-49`). Full-run co-tenancy and downstream provenance validation fail for the reasons above. |
| Q5 runbooks | **FAIL** | Model commands use the correct launcher/flags and timing passes are ordered behind the host gate, but the command set/spec is incomplete and some failures are masked. B1→B2 reuse is accuracy-valid only, not a registered timing substitution. |
| Q6 Planner records scripts | **PASS** | `docs_1106zh_exp37_plan.py:6-34` pins helper/source hashes and produced commit `54bc7d59`; `docs_1106zi_a1.py:21-63` pins the review/amendment inputs and produced `ce2412c5`. Their done manifests record the expected source/output SHAs and commits. No finding. |
| Q7 other | **FAIL** | Missing waveforms, incomplete aggregation gates, weak source binding, c2f candidate provenance, and runbook gaps require correction. |

## What I verified

Static inspection only—no scripts, tests, localization, benchmark, GPU, package, process, or write operations were performed.

I inspected the exact pushed range, binding plan/review/research materials, bundle reference timer/selector/summarizer, all changed production tools and relevant tests, the three runbooks, committed 0.75 m/nested artifacts, ledger/notebook, announcement 08, both Planner scripts, their transaction manifests, and commits `54bc7d59`/`ce2412c5`. The committed artifacts statically report 5,337 included 0.75 m queries, 11,537 pilot pairs, 64 unchanged query IDs, and 6,610 nested pairs.
