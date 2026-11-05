**Reviewer:** OpenAI Codex (gpt-5.6-sol, xhigh, codex exec, read-only sandbox) · **Date:** 2026-11-04 · **Round:** plan r0 (exp_35 orbitrir_yaw_evidence: plan DRAFT r0 at 603b1094)

**Verdict:** REVISE

The scientific objective is sound, but r0 is not ready for GPU execution. The principal blockers are the unavoidable 28-cell minimum for an exp_25-standard B-F Tier 3 row, the unpinned flash-attention shadow, incomplete Tier 2/preregistration semantics, the roughly 15.4-hour phase-(i) estimate versus Yixun’s expected three hours, and the resulting change required to exp_22’s already-approved launch procedure.

## P1 — Minimal scope

**Status: BLOCKING**

Yes: the literal “10 random-yaw cells plus T-dup and Tier 2 dumps” is insufficient for a Tier 3 row computed exactly as `depth_canon_collect.py` does.

The collector requires:

- Five θ=0 values per K for `sd(m_Z)`.
- Five same-seed θ=0/R pairs per K for `Δ_R`.
- Same-input θ=0/90/180/270 at seed 42 for `Δ_C4`.
- A same-input θ=0/45 pair for the 45° probe.

These are explicit at [depth_canon_collect.py:1087](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_collect.py:1087), and pairing requires the same arm/K/seed/step plus equal `input_hash` at [depth_canon_collect.py:727](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_collect.py:727). Tier 2 additionally requires both the rotated cell and θ=0 reference to have scored dumps at [depth_canon_collect.py:1005](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_collect.py:1005).

I confirmed both frozen CSV campaigns have lost every rotated source JSON:

- `yaw_metrics_raw.csv`: 40 rotated paths at 45°/90°, zero remain.
- `yaw_c4_metrics_raw.csv`: 60 rotated paths at 90°/180°/270°, zero remain.
- Only the 20 θ=0 JSONs remain across each manifest.

Therefore the smallest exact Tier 3 design is:

- B-F θ=0: 10 cells.
- B-F random yaw: 10 cells.
- B-F seed-42 90°/180°/270°/45°: 8 cells.

That is **28 Tier 3 measurement cells**. Add two K-specific duplicates to reproduce exp_25’s Tier 2 determinism coverage: **30 B-F result cells**. The two P1 controls bring the launch phase to **32 cells**, plus the B-F control probe.

A 22-cell design—T10 + R10 + two duplicates—could splice `Δ_C4` and 45° scalars from the old CSV, but it would be mixed-backend, unpaired, and without `input_hash`; it is not a Tier 3 row “exactly as the collector” computes one.

**Concrete fix:** State 28 as the mathematical minimum for Tier 3, 30 for Tier 3 plus two-K Tier 2 determinism, and 32 including backend controls. Present the historical 22-cell splice only as a clearly non-registered alternative.

**Changes what Yixun must decide:** **Yes.** He must approve the expanded 30-result-cell phase or accept a non-exp_25-standard historical splice/drop Tier 2.

## P2 — B-F protocol and arm-aware changes

**Status: HIGH**

The proposed flags are correct for the new exp_35 evaluation:

```text
--cond-method fa_invariant
--frame-avg-angles 0,90,180,270
--frame-avg-max-fwd-samples 64
```

The B-F configuration binds `fa_invariant` and the C4 angles at [FLAC_AR_BF.json:195](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_07_fa_scratch_claude/FLAC_AR_BF.json:195). The evaluator exposes those flags at [eval_FLAC.py:2268](/home/yixunhu/codespace/FLAC/eval_FLAC.py:2268); 64 is the registered batched-evaluation default at [yaw_rotation.py:47](/home/yixunhu/codespace/FLAC/src/data/yaw_rotation.py:47), with angle-order-preserving accumulation at [yaw_rotation.py:523](/home/yixunhu/codespace/FLAC/src/data/yaw_rotation.py:523).

Qualification: the original exp_10 A6/table cells used the legacy per-angle loop. The new cells will be `orbit_execution=batched`, cap 64. Thus the flags preserve the intended C4 conditioning recipe, but not the original execution implementation. This is why E2 is needed.

The arm-aware refactor is minimal in concept because exp_25 currently hardcodes vanilla in both record validation and output paths:

- [depth_canon_collect.py:123](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_collect.py:123)
- [depth_canon_collect.py:453](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_collect.py:453)
- [depth_canon_make_cells.py:49](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_make_cells.py:49)

It is safe only if guarded by exact regressions.

**Concrete fix:** Require:

1. The original 230 generated rows remain byte-identical.
2. The old exp_25 mode reproduces `depth_canon_results.json` byte-for-byte.
3. Arm protocols affect record validation, output suffixes, probe binding and rendering consistently.
4. BF records must bind angles `[0.0,90.0,180.0,270.0]`, batched execution, cap 64, bf16 conditioning and checkpoint/config hashes.

**Changes what Yixun must decide:** **No.** This is an implementation contract once the expanded scope is approved.

## P3 — B-F Tier 2 semantics

**Status: BLOCKING**

CANON’s composition control is not a valid numerical control for B-F. In the committed CANON probe, the evaluation-control p99 values at the quarter-turn offsets are exactly zero, while d64/random are nonzero:

```text
d128 = d256 = d384 = 0
d64 = 0.014003...
random = 0.011007...
pooled = 0.010524...
```

The resulting CANON rows are visible at [depth_canon_results.md:21](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_results.md:21). CANON makes the quarter-turn inputs identical after canonicalization; B-F instead permutes and sums the four C4 conditioner passes, so its own arithmetic floor must be measured.

The faithful analog is:

```text
A = FA(md)
B = FA(rotate_d(md))
d ∈ {128, 256, 384 panorama columns}
```

using the exact B-F checkpoint/config, four registered angles, cap 64, bf16 evaluation conditioning, fixed item stream and the same probe stages. No canonicalization should run.

B-F’s Tier 2 conclusion should then be split:

- **Tier 2-C4:** registered invariance verdict at 90°/180°/270°, judged against B-F’s own FA composition control and the absolute cap; T-dup remains exact.
- **45° and random yaw:** report waveform distances descriptively as off-orbit non-invariance. They are outside the C4 invariance claim. They may exceed the C4 threshold “by construction,” but that is not a failed C4-invariance verdict.
- **Tier 3:** still applies the registered C4 and random-yaw rules, so the overall B-F Tier 3 verdict is expected to fail if `|Δ_R| > sd(m_Z)`.

The draft’s E4 phrase “Tier 2 fails at 45°/R” conflates an intentionally unsupported transformation with failure of the claimed C4 invariance.

**Concrete fix:** Register “Tier 2-C4” explicitly; exclude 45°/random from its pooled verdict; retain their dump distances as descriptive diagnostics.

**Changes what Yixun must decide:** **Yes.** He must approve C4-only Tier 2 semantics for OrbitRIR.

## P4 — Attention backend and import shadow

**Status: BLOCKING**

The planner’s backend findings are confirmed:

- `flash-attn` version `2.7.4.post1` is installed, with package, metadata and CUDA extension timestamps `2026-11-01 23:14:14 EST`.
- FLAC imports it at [transformer.py:12](/home/yixunhu/codespace/FLAC/src/models/transformer.py:12).
- The diffusion transformer selects it whenever importable, otherwise SDPA, at [transformer.py:406](/home/yixunhu/codespace/FLAC/src/models/transformer.py:406).
- exp_25 logs and exp_22’s August logs print `No module named 'flash_attn'` and `flash_attn not installed`.

A top-level `flash_attn` shadow that raises `ImportError` is functionally sufficient to force FLAC’s diffusion transformer onto SDPA without modifying the installed package. I found no other code in the `eval_FLAC.py` path that would change B-F behavior because of the shadow. The DINO conditioner is loaded through `AutoModel` without an explicit flash implementation at [conditioners.py:445](/home/yixunhu/codespace/FLAC/src/models/conditioners.py:445); its current default remains SDPA.

But the proposed **scratch** shadow is not provenance-safe. The exp_25 runner hashes only its enumerated relevant tree at [depth_canon_eval_run.sh:111](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_eval_run.sh:111). An external scratch directory is absent from the campaign digest, so the evaluated backend input would be uncommitted and unhashed.

Also, `eval_localization.py` records `flash_attn_available` using `find_spec`, not the actual transformer dispatch, at [eval_localization.py:719](/home/yixunhu/codespace/FLAC/eval_localization.py:719) and [eval_localization.py:863](/home/yixunhu/codespace/FLAC/eval_localization.py:863). A shadow package still has a discoverable spec, so exp_22 would misleadingly record “available” even though the FLAC import fell back.

The two exact P1 controls are a proportionate launch gate if they compare the old/new waveform manifests and stream hashes bit-for-bit. Combined with the source diff—only `src/localization/*` changed since `d6ca307a`—they establish the intended P1 evaluator-path equivalence. They do not prove universal equivalence, and “inside the vanilla envelope” is not an acceptable fallback for E1.

Running with flash instead would make new exp_35 cells internally coherent but would break direct numerical comparability to exp_25’s no-flash P1/CANON thresholds. Restoring that comparison would require same-backend comparator re-runs and would no longer be minimal.

**Concrete fix:** Commit the minimal shadow, include it in the runner’s relevant-tree hash and receipts, record the actual selected FLAC backend (`flash_attn_func is None` / `sdpa`), and make exact CTL equality a hard stop. Do not use an unpinned scratch directory.

**Changes what Yixun must decide:** **Yes.** He must approve a hash-pinned no-flash overlay and the small runner/provenance adjustment, or accept a substantially larger same-flash remeasurement campaign.

## P5 — θ=0 admissibility and E2

**Status: HIGH**

The requested vanilla agreement statement is correct.

My independent per-seed comparison of the exp_10 CSV against exp_25 P1 produced:

| K | T60 max | C50 max | EDT max | FD max | R@1 max | R@5 max | R@10 max |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.0003 | 0.0001 | 0.0031 | 0.0001 | 0.0947 pp | 0.1263 pp | 0.0474 pp |
| 8 | 0.0004 | 0 | 0.0033 | 0.0001 | 0.0947 pp | 0.0789 pp | 0.1736 pp |

The means agree to three decimals for T60, C50 and EDT. The exp_25 means are reported at [depth_canon_results.md:93](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_results.md:93).

E2’s bounds—0.001 T60, 0.0005 C50, 0.005 EDT, 0.0005 FD and 0.2 percentage points for retrieval—are sensible as an admissibility sentinel.

However, B-F K8 seed 42 has two historical θ=0 sources:

- A6/C4-backfill CSV: T60 `8.1890`.
- Historical table source `exp07_BF40_fae_rot0`: T60 `8.1902`.

Their 0.0012 difference already exceeds E2’s T60 threshold. E2 must identify one artifact, not the phrase “exp_10.”

**Concrete fix:** Pin the A6 CSV source recommended for the figure—SHA-256 `0ae554c09f45ce89b8957cf75a3d15e44f4173f4f8987497dbbebf16bbfb23da`—as E2’s reference, per K/seed. Alternatively widen T60 to 0.002 before observing new results if the intent is to admit either historical θ=0 source. Define retrieval units explicitly as percentage points.

**Changes what Yixun must decide:** **Yes, narrowly.** Choose the A6 CSV as the controlling reference, or approve a broader threshold that admits both historical sources.

## P6 — exp_35 collector mode

**Status: HIGH**

The proposed separate exp_35 mode is sound if old results are treated as immutable external evidence, not merged into the new cell campaign.

The exp_25 results JSON is pinned at:

```text
ec3fcd08e3d210524b50246d9bfdf084e72835432a5d06450085e0cad12eb6e0
```

It is labelled partial only because 20 seen-split cells are missing; the relevant unseen rows and Tier 3 values are complete, as documented at [depth_canon_results.md:1](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_results.md:1).

The main incompatibility risk is campaign mixing. The existing collector requires one campaign digest at [depth_canon_collect.py:1288](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_collect.py:1288). exp_35 must not silently treat derived exp_25 rows as if they were cells from H35.

**Concrete fix:** Fail closed unless:

- The external JSON hash, schema and pin equal the registered values.
- `partial` has exactly the 20 known seen-cell omissions and no invalid cells.
- Every reused unseen/Tier 3 field is finite and non-null.
- Old rows remain in an `external_reference` namespace.
- New H35 cells have exactly one campaign digest.
- B-F T/R streams reproduce the expected per-K/seed `input_hash`.
- Random-yaw cells also reproduce the registered `assignment_hash`, not merely an arbitrary valid SHA.
- Historical exp_10 CSV values are never admitted into the registered B-F Tier 3 calculation.
- CTL comparison checks old receipt waveform/dump hashes exactly.

**Changes what Yixun must decide:** **No**, assuming the no-flash backend choice is settled. These are collector integrity requirements.

## P7 — Off-orbit figure

**Status: HIGH**

Using exp_10’s 45° FLAC/OrbitRIR points beside new no-flash 135°/225° points is acceptable with explicit disclosure because Yixun expressly required reuse of the existing 45° cells. It is not fully homogeneous, but the θ=0 agreement and E2 gate make it defensible.

The raw/summary figure inputs must retain per-point provenance: source campaign, evaluator pin, attention backend, legacy-loop versus batched orbit execution, source-file hash, K and seed. The paper’s existing mapping confirms `Vanilla P1@40k → FLAC` and `FA B-F@40k → OrbitRIR`.

A full 45° rerun for FLAC and B-F is therefore **not mandatory**. It becomes mandatory only if E2 fails or if Yixun chooses a homogeneous-backend figure; that would add 20 cells.

The draft’s actual error is calling the eight CANON 45° seeds 43–46 cells “optional” at [plan_orbitrir_yaw_evidence.md:42](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude/plan_orbitrir_yaw_evidence.md:42). Yixun requested five-seed mean/error bars for all three models at 45° in [orbitrir_yaw_evidence_yixun_query.md:13](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude/orbitrir_yaw_evidence_yixun_query.md:13). exp_25 supplies only CANON seed 42, so the remaining eight are required.

**Concrete fix:** Make CANON-45 ×8 mandatory. Keep historical FLAC/B-F 45° only if E2 passes, with visible mixed-provenance disclosure and provenance columns in the raw CSV.

**Changes what Yixun must decide:** **Only conditionally.** No 45° rerun is needed now; if E2 fails, he must choose between 20 new 45° cells and a separately labelled historical series.

## P8 — GPU plan and ETA

**Status: BLOCKING**

The arithmetic is correct:

- Controls: approximately 0.61 h.
- B-F T10: approximately 4.92 h.
- B-F R10: approximately 4.92 h.
- Duplicates: approximately 0.98 h.
- C4+45 ×8: approximately 3.93 h.
- Probe: approximately 0.08 h.

Total phase (i): approximately **15.4 h**, not three hours. The off-orbit P1/B-F cells are approximately 16 h; CANON 135°/225° plus required CANON-45 are approximately 8.6 h, giving **22–25 co-tenant lane-hours** for the figure and approximately 40 total lane-hours.

I independently parsed the archived exp_30 progress log:

- Unshared 500-step windows: 4.208–4.234 s/step.
- During the s44 table job: 7.610–9.712 s/step.
- First 4,000-step mean: 8.521 s/step.

Thus the proposed guard is grounded in Yixun’s own job. It should use the unrounded registered values—9.712 and 8.521—or state its rounding rule.

The operational issue is latency: a two-cell B-F chunk may run about an hour before the plan checks the training guard, and the first 500-step window itself takes over an hour when shared. That is too slow to detect an adverse B-F-specific effect.

The exp_22 watcher also needs explicit reservation coordination. Yixun put exp_35 ahead of exp_22 at [loc_meshgrid_worklog.md:842](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_22_loc_meshgrid_claude/loc_meshgrid_worklog.md:842), but a free-card handoff could otherwise race the watcher.

**Concrete fix:**

- Start with one cell per chunk: CTL K1, CTL K8, B-F K1, B-F K8.
- Add a shorter rolling early-warning window while retaining the 500-step final guard.
- Do not permit two-cell chunks until both B-F K values have measured time/memory impact.
- Reserve/release the card explicitly to the exp_22 watcher.
- Limit co-tenancy initially to phase (i); do not assume another 22–25 shared hours without approval.
- Report the expected phase-(i) training-progress cost: approximately 7.7 h.

**Changes what Yixun must decide:** **Yes.** He must approve approximately 15.4 co-tenant hours and its estimated 7.7-hour training-progress cost versus the requested three-hour expectation.

## P9 — Pre-registration E1–E5

**Status: BLOCKING**

The preregistration must be fixed before any measurement cell runs.

- **E1:** Exact CTL waveform/dump hash equality must be a launch gate. Delete the fallback “else inside the vanilla envelope.” A mismatch stops the campaign.
- **E2:** Name the exact historical B-F files/CSV hash, K/seed rows, units and thresholds. Resolve the K8 seed-42 source ambiguity in advance.
- **E3:** Replace “retrieval within 1–2 item flips” with the actual registered rule: `Δ_C4(BF) <= 0.1 × Δ_C4(P1)` for every registered metric and both K. Do not substitute an informal item count after seeing results.
- **E4:** Separate expectation from verdict. B-F is expected to pass Tier 2-C4 and fail Tier 3 through random yaw; 45°/random are outside the C4 Tier 2 claim.
- **E5:** Either define numerical tolerances for “≈” or mark the relationships descriptive and non-verdict-bearing.
- **FD:** The requested presentation table has six metrics, but the registered Tier 3 rule includes FD at [depth_canon_collect.py:48](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_collect.py:48). FD must remain in the Tier 3 verdict even if omitted from the six-panel figure.
- Freeze the exact cell list, ordering, H35 pin, checkpoint/config hashes, backend overlay hash, exp_25 JSON hash, CSV hashes, BF probe contract, stop gates and scheduler handoff before GPU execution.

**Changes what Yixun must decide:** **Yes.** The substantive decisions are scope, backend, Tier 2-C4 semantics and E2 reference. The remaining fixes are mandatory preregistration hygiene.

## P10 — exp_22 consequence

**Status: BLOCKING**

Yes. The flash-attention finding changes exp_22’s launch requirements.

The August P1-era logs show no flash-attention—for example [loc_meshgrid cache parity log:25](</home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_22_loc_meshgrid_claude/loc_meshgrid_2026-08-25_17:20:00_i1_cache_parity.log:25>) and [probe smoke log:17](</home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_22_loc_meshgrid_claude/loc_meshgrid_2026-08-25_17:45:00_i1_probe_smoke.log:17>). G0 is explicitly a bit-exact replay against that frozen P1 campaign at [loc_meshgrid_worklog.md:705](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_22_loc_meshgrid_claude/loc_meshgrid_worklog.md:705). It must therefore run with the same no-flash FLAC backend.

The eventual YAW and BF passes are paired against P1, so they also must use the same no-flash backend. For one coherent R2 campaign, the preceding gates and BF throughput probe should use it too.

The authoritative command list at [loc_meshgrid_worklog.md:691](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_22_loc_meshgrid_claude/loc_meshgrid_worklog.md:691) does not pin the attention backend. It cannot safely run unchanged in the current environment. A shadow alone also causes the `find_spec` provenance problem described in P4.

**Concrete fix:** Suspend the watcher launch until exp_22 has a reviewed, hash-bound no-flash mechanism and records the actual FLAC backend. This may require a new reviewed pin or a genuinely isolated environment in which `flash_attn` is undiscoverable; the current command list’s pin/backend contract must be amended explicitly.

**Changes what Yixun must decide:** **Yes.** The previously approved exp_22 launch list must be reopened for this backend correction.

## Findings ranked

1. **BLOCKING — Backend provenance:** The scratch flash shadow affects numerics but is outside the runner digest; exact CTLs do not repair an unpinned runtime input.
2. **BLOCKING — Scope/ETA mismatch:** An exact B-F Tier 3 row requires 28 cells; Tier 2 determinism brings it to 30, versus the assumed 11-ish cells/three hours.
3. **BLOCKING — Tier 2 and preregistration:** The plan conflates C4 invariance with expected off-orbit non-invariance and leaves E1–E5 partly data-dependent.
4. **BLOCKING — GPU authorization:** Phase (i) is approximately 15.4 co-tenant hours with an estimated 7.7-hour training-progress cost; the current check cadence is too slow.
5. **BLOCKING — exp_22 drift:** Its frozen replay and paired passes were designed against no-flash P1, but the authoritative commands do not enforce that backend.
6. **HIGH — Collector campaign separation:** Old derived results must remain hash-pinned external references, with strict partial-result and stream/assignment validation.
7. **HIGH — Figure completeness:** CANON 45° seeds 43–46 are mandatory, not optional; the historical/new backend mix is acceptable only after E2 and with per-point provenance.
8. **HIGH — E2 source ambiguity:** B-F K8 seed 42 has two historical θ=0 values whose T60 difference already exceeds the proposed tolerance.

## Minimal decision list for Yixun

1. Approve the exact-standard phase: **30 B-F result cells + 2 P1 controls + one BF probe**, approximately 15.4 shared lane-hours, or explicitly choose a non-standard historical splice.
2. Approve a **committed, hash-bound no-flash overlay** and exact-CTL hard stop.
3. Approve **Tier 2-C4** as OrbitRIR’s invariance claim, with 45°/random dump distances descriptive only.
4. Select the **A6 CSV as E2’s controlling historical B-F source**; recommended, because it is the figure source. Otherwise widen T60 before running.
5. Confirm the **eight CANON-45 cells are mandatory** and historical FLAC/B-F 45° remains acceptable unless E2 fails.
6. Approve co-tenancy only for phase (i), or explicitly authorize further figure co-tenancy after the measured slowdown report.
7. Reopen exp_22’s authoritative commands to require the same **no-flash backend** before its watcher may launch.

## What I verified

All probes were read-only. I did not import Torch, query a GPU, run any evaluator/runner/collector/probe/test, touch a process, modify a file, or install/change a package.

- `git branch --show-current`, `git rev-parse HEAD`, `git status`, and scoped `git diff`: branch `localization-exp`, HEAD `603b10942b8185273eeaa7db523a348904cf7d34`; since `d6ca307a`, only localization modules/tests changed under the inspected evaluator tree.
- `sha256sum`: main/exp-25-worktree kit files match byte-for-byte; exp_25 results JSON SHA and exp_10 CSV hashes recorded above.
- `sed`/`nl`/`rg`: collector pairing/Tier 2/Tier 3 contracts, runner digest and GPU gate, BF config, evaluator flags, FA batching, transformer dispatch, exp_22 authoritative commands and historical no-flash logs.
- `stat` plus direct metadata reading: `flash-attn 2.7.4.post1`, installed at the stated Nov. 1 timestamp; no package import.
- Standard-library Python with `PYTHONDONTWRITEBYTECODE=1`: checked CSV source existence, compared exp_10/exp_25 per-seed metrics, inspected JSON probe/control values and parsed archived exp_30 progress text.
- Legacy-source census: zero surviving rotated source JSONs in both generated raw CSV manifests; all 20 θ=0 sources remain.
- Archived timing parse: 4.208–4.234 s/step unshared, 7.610–9.712 shared, 8.521 mean over the first 4,000 shared steps.
- Paper-repo read-only inspection: existing plot maps `Vanilla P1@40k` to FLAC and `FA B-F@40k` to OrbitRIR and states that plotted angles are measured rather than C4-replicated.
