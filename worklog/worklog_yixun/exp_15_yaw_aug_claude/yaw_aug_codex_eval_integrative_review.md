# exp_15 yaw_aug — Codex SECOND INTEGRATIVE review (eval surface, pre-campaign)

**Reviewer:** OpenAI Codex `gpt-5.6-sol` xhigh (codex-cli 0.146.0, read-only) · **Date:** 2026-08-16 · **Eval surface at:** `5077502` · **Verdict: NO-GO** — 1 BLOCKING (campaign pin does not bind the executing control plane), 4 MAJOR (K=1 rendered confirmatory; V-cell gate leakage; ratified Invalid-T60 routing dropped; §6.9–6.10 have no executable path), 2 MINOR, 1 NIT. Seam audit PASSED (one schema, 42-basename round-trip, DRYRUN exact). Includes the authoritative 13-step post-40k runbook.

# Second integrative review — exp_15 `yaw_aug`

**Verdict: NO-GO.** Eval must not launch merely because 40k lands and a pin file is created. The artifact/data-plane seams are consistent, but the campaign pin does not currently bind the complete executable control plane, and several pre-registered statistical/gating requirements are not faithfully implemented.

The requested ancestry range does not apply cleanly to the rewritten current history, so I used the permitted fallback: files at `5077502`, their history since `1deb10e`, and the current versions. The requested eval surface is byte-identical between `5077502` and current HEAD.

## Findings

1. **BLOCKING — The campaign pin does not bind the driver and control-plane code actually executed.**

   The wave script enumerates and classifies cells using the validator in the moving main checkout ([yaw_aug_submit_grid.sh:109](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_submit_grid.sh:109), [yaw_aug_submit_grid.sh:245](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_submit_grid.sh:245), [yaw_aug_submit_grid.sh:311](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_submit_grid.sh:311)). The single-cell submitter similarly renders the identity/contract with the main-tree validator before preparing the pinned worktree ([yaw_aug_screen_submit.sh:313](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_screen_submit.sh:313), [yaw_aug_screen_submit.sh:431](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_screen_submit.sh:431)).

   Most importantly, `sbatch` receives the main-checkout driver, not the copy in the pinned worktree ([yaw_aug_screen_submit.sh:531](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_screen_submit.sh:531)). That driver then records the pinned commit as `source_sha` ([yaw_aug_screen.sbatch:382](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_screen.sbatch:382), [yaw_aug_screen.sbatch:532](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_screen.sbatch:532)). A post-pin edit to the main driver can therefore affect execution while the artifact claims the pinned commit. This is the announcement-05 mismatch class that the pin is meant to eliminate.

   **Fix:** make the main entry point a minimal bootstrap that reads the pin, prepares the worktree, and re-execs the grid/single submitter from that worktree. Submit `$WT/.../yaw_aug_screen.sbatch`, and use the pinned validator for enumeration, classification, intent rendering, and execution. Add a regression that deliberately makes the main driver/validator differ from the pin and proves only pinned content is used and attributed.

2. **MAJOR — K=1 is incorrectly analyzed and rendered as confirmatory H1.**

   The plan registers exactly one confirmatory family: H1’s two K=8 co-primaries. K=1 is descriptive only ([plan_yaw_aug.md:88](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/plan_yaw_aug.md:88), [plan_yaw_aug.md:90](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/plan_yaw_aug.md:90)).

   The collector unconditionally labels H1 confirmatory and applies `contrast_rows`/Holm for whichever K it receives ([yaw_aug_collect.py:816](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_collect.py:816), [yaw_aug_collect.py:842](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_collect.py:842)). It builds hypotheses for both K=8 and K=1 ([yaw_aug_collect.py:1070](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_collect.py:1070)), while the renderer hardcodes the heading “H1 (K=8, co-primaries)” even for the K=1 section ([yaw_aug_collect.py:954](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_collect.py:954)).

   **Fix:** branch explicitly on K. K=8 gets Holm-2 and the registered verdict vocabulary. K=1 gets descriptive estimates/unadjusted CIs, no confirmatory family, no superiority/inferiority verdict, and a correct heading. Add golden report and JSON tests for both K sections.

3. **MAJOR — YAWAUG V@90° has acquired a gate role despite the explicit prohibition.**

   The plan says YAWAUG V is descriptive/mechanistic only and carries no gate role ([plan_yaw_aug.md:66](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/plan_yaw_aug.md:66), [plan_yaw_aug.md:107](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/plan_yaw_aug.md:107)). It requires cross-arm `assignment_hash` equality for R cells, not V cells ([plan_yaw_aug.md:84](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/plan_yaw_aug.md:84)).

   The collector registers assignment obligations for both R and V ([yaw_aug_collect.py:369](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_collect.py:369)). G3 PENDING globally suppresses hypothesis output, and V hash failures are routed into H2/H3 blocking scopes ([yaw_aug_collect.py:697](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_collect.py:697), [yaw_aug_collect.py:724](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_collect.py:724)). Thus a missing or mismatched YAWAUG V cell can block inference.

   **Fix:** remove V from hypothesis-required G3 obligations. Continue validating each V artifact and report its hashes locally, but let V defects suppress only that mechanism readout. Required assignment equality should cover registered R pairs only; T↔R input equality should scope only the affected H2 contrasts.

4. **MAJOR — §13’s ratified `Invalid T60` routing is silently dropped.**

   §13 explicitly places `Invalid T60` in the acoustic family, requiring the ten-room-family mean ([plan_yaw_aug.md:244](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/plan_yaw_aug.md:244), [plan_yaw_aug.md:248](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/plan_yaw_aug.md:248)). The validator requires it at split level but omits it from the required per-scene schema ([exp15_validate_cell.py:147](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/exp15_validate_cell.py:147)). The imported `HEADLINE_METRICS` set excludes it, and exp_15 uses that set for routing and every block table ([yaw_aug_collect.py:775](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_collect.py:775), [yaw_aug_collect.py:1071](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_collect.py:1071)).

   **Fix:** require finite per-scene `Invalid T60` values for all ten groups, route their ten-group mean, and include them in descriptive markdown/JSON tables. Keep them outside the confirmatory family.

5. **MINOR — The exp_11 external reproduction check is absent.**

   The plan requires non-halting comparisons against both exp_11 VANL and exp_14 Z ([plan_yaw_aug.md:106](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/plan_yaw_aug.md:106)). The exp_11 Q9 VANL rows are registered and populated as a distinct contract ([gen_model_comparison.py:92](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/gen_model_comparison.py:92)). Exp_15’s collector only discovers and compares exp_14 Z evidence ([yaw_aug_collect.py:887](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_collect.py:887), [yaw_aug_collect.py:911](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_collect.py:911)).

   **Fix:** add a separately labeled, validated exp_11 Q9 comparison using the same pre-declared tolerance. It remains descriptive and non-halting.

6. **MINOR — Planned direct V/probe submissions can double-run and are not appended to `yaw_aug_command.md`.**

   The grid path implements classification, valid-cell skipping, queue/lease detection, and durable command logging ([yaw_aug_submit_grid.sh:311](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_submit_grid.sh:311), [yaw_aug_submit_grid.sh:425](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_submit_grid.sh:425)). The single-cell path used by the planned first V/probe submissions has no equivalent classify/in-flight guard and writes only an intent manifest before submission ([yaw_aug_screen_submit.sh:531](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_screen_submit.sh:531), [yaw_aug_screen_submit.sh:572](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_screen_submit.sh:572)). This misses §6.7’s “every submission appended” requirement ([plan_yaw_aug.md:150](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/plan_yaw_aug.md:150)).

   **Fix:** give the single-cell path the same validate-before-skip and in-flight checks, with durable command-log append before `sbatch`; alternatively provide a one-cell mode in the grid and use that exclusively.

7. **MAJOR — §6.9–§6.10 have prose triggers but no executable completion path.**

   The YAWAUG model-row transaction must fire after its T cells reach 5/5 at both K and their applicable gates pass ([plan_yaw_aug.md:171](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/plan_yaw_aug.md:171)). The generator has no exp_15 row or validator; its registration currently ends with exp_14 rows ([gen_model_comparison.py:120](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/gen_model_comparison.py:120)). It also averages top-level metric fields ([gen_model_comparison.py:15](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/gen_model_comparison.py:15)), which cannot simply be reused for §13’s scene-routed acoustic estimand.

   The collector only prints markdown to stdout or optionally writes a JSON bundle ([yaw_aug_collect.py:1097](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_collect.py:1097)); it does not create `yaw_aug_results.md`, analysis, HTML, assets, or commits. The current params and command documents contain training only ([yaw_aug_params_set_up.md:1](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_params_set_up.md:1), [yaw_aug_command.md:23](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_command.md:23)).

   **Fix:** before launch, add a tested YAWAUG two-K table transaction using exp_15 validation and §13 routing, plus a concrete publication command/checklist naming who writes each §6.10 artifact. It must have a T-only readiness predicate; the global collector remains PENDING while R blocks are absent.

8. **NIT — G2 does not encode the planned YAWAUG probe identity.**

   The ladder names YAWAUG K=8 seed-42 R as the probe ([plan_yaw_aug.md:188](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/plan_yaw_aug.md:188)). The collector accepts the first K=8 seed-42 R artifact from either arm ([yaw_aug_collect.py:1052](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_collect.py:1052)).

   **Fix:** select the exact registered YAWAUG/rrob/K8/s42 cell and name it in G2’s report.

## Seam audit

Aside from Finding 1’s provenance/control-plane problem, the artifact seams pass:

- The validator defines the exact metrics filename and sidecars in one place: metrics, `.stream.json`, and `.json.screenmeta.json` ([exp15_validate_cell.py:333](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/exp15_validate_cell.py:333), [exp15_validate_cell.py:350](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/exp15_validate_cell.py:350)).
- The driver resolves the actual path through `eval_FLAC.build_output_paths`, writes the expected screenmeta—including both literal and effective frame-angle fields—and runs the shared validator before publishing success ([yaw_aug_screen.sbatch:503](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_screen.sbatch:503), [yaw_aug_screen.sbatch:532](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_screen.sbatch:532), [yaw_aug_screen.sbatch:587](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_screen.sbatch:587)).
- The collector reads the same three artifacts and delegates protocol/schema validation to `exp15_validate_cell.validate_payloads`; there is no second schema ([yaw_aug_collect.py:172](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_collect.py:172), [yaw_aug_collect.py:734](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_collect.py:734)).
- The 42 cells originate in `expected_grid`, the grid consumes that enumeration, and the collector reconstructs every basename through the validator’s own `metrics_path` ([exp15_validate_cell.py:182](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/exp15_validate_cell.py:182), [yaw_aug_submit_grid.sh:245](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_submit_grid.sh:245), [yaw_aug_collect.py:135](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_collect.py:135)). The committed DRYRUN transcript contains exactly 20 T, 20 R, and 2 V cells.

## Plan §5 / §13 status

| Requirement | Status |
|---|---|
| Five seed-paired differences, paired-t CI, direction orientation | **Satisfied in code; runtime pending.** |
| H1 K=8, Holm over two co-primaries | **Satisfied for K=8.** |
| K=1 descriptive only | **Not satisfied — Finding 2.** |
| H2 degradation contrast and H3 absolute-R contrast | **Satisfied in code; runtime pending.** |
| G1 seed-42 positive control | **Satisfied in code; runtime pending** ([yaw_aug_collect.py:306](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_collect.py:306)). |
| G2 golden stream | **Implemented; exact probe identity needs Finding 8.** |
| G3 scoped hash integrity | **Partial — V improperly gates H-readouts, Finding 3.** |
| G4 checkpoint admission for every cell | **Satisfied and fail-closed; YAWAUG pending 40k** ([yaw_aug_collect.py:490](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_collect.py:490)). |
| G5 eight complete T/R blocks | **Satisfied in code; runtime pending** ([yaw_aug_collect.py:537](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_collect.py:537)). |
| Exp_11 and exp_14 external checks | **Partial — exp_14 only, Finding 5.** |
| §13 acoustic/split routing, audio-to-audio R@1, geom quarantine | **Satisfied except `Invalid T60`, Finding 4.** |
| Mandatory inference-scope statement | **Satisfied in report/bundle; analysis artifact pending.** |

## §8 acceptance checklist

| §8 item | Status |
|---|---|
| Training job reports pin; allowlist green | **Satisfied for completed legs.** Initial chain acceptance is recorded at [yaw_aug_worklog.md:212](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_worklog.md:212). |
| 1×8 L40, micro 8, effective 64, SyncBN, grad checkpointing | **Satisfied for completed training legs** ([yaw_aug_params_set_up.md:5](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_params_set_up.md:5)). |
| Exact yaw banner before step 0; optimizer progress; no OOM/NaN | **Satisfied for completed legs** ([yaw_aug_worklog.md:214](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_worklog.md:214)). |
| Rate gate | **Satisfied:** 1.005 steps/s versus 0.849 floor ([yaw_aug_worklog.md:218](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_worklog.md:218)). |
| 2500 cadence; final 40k checkpoint and digest | **Cadence satisfied through 30k; final pending.** Registry still has null `final_ckpt_sha256` and `final_step` ([yaw_aug_launch_registry.json:8](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_launch_registry.json:8)). |
| Eval argv manifest, three artifacts, §5 gates | **Implementation partial/runtime pending.** Artifact schema passes the seam audit, but Findings 1–6 prevent acceptance. |

## Exact post-40k runbook

1. Let the final leg finish its completion audit. Verify exactly one audited `step=40000` leg, `final_step: 40000`, and a non-null `final_ckpt_sha256`; commit and push the registry/worklog closure.

2. Implement Findings 1–8 with focused TDD and close their fix review. No eval submission before that closure.

3. Choose the campaign pin. It should be the **full 40-character SHA of the reviewed, pushed commit containing both the final 40k registry state and all integrative fixes**. It must not be `5077502`, current HEAD, or any commit whose registry has a null final digest.

4. Write that SHA, once, to `yaw_aug_screen_campaign_pin`. Do not use `PIN_SHA` as a substitute—the scripts correctly treat it only as an equality assertion ([yaw_aug_screen_submit.sh:202](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_screen_submit.sh:202), [yaw_aug_submit_grid.sh:217](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_submit_grid.sh:217)).

5. From the pinned tree, run the torch-free admission expectations for both arms. Do not load checkpoints on the login node. The first compute cell for each arm will run the heavy recomputing G4 admission before evaluation ([yaw_aug_screen.sbatch:404](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_screen.sbatch:404)).

6. Submit only the VANL positive control:

   `bash .../yaw_aug_screen_submit.sh ARM=VANL CELL=vctl STEP=40000 SEED=42 K=8 ROTATE_DEG=90`

   Wait for metrics, stream, and screenmeta to validate. This exercises VANL admission, provisioning, fixed rotation, and artifact landing.

7. Submit only the YAWAUG probe:

   `bash .../yaw_aug_screen_submit.sh ARM=YAWAUG CELL=rrob STEP=40000 SEED=42 K=8 ROTATE_DEG=0`

   Wait for validation and require G2 PASS. This exercises YAWAUG admission and supplies timing. Stop on any admission or G2 failure.

8. Run `WAVE=vctl`. Validate-before-skip must skip the landed VANL cell and submit only YAWAUG V. Wait for it to validate; it remains non-gating.

9. Run `WAVE=tbl MAX_INFLIGHT=16`. Wait for all 20 T cells. Evaluate G1 and the T-scoped G3/G4/G5 obligations. G1 failure halts the campaign before the full R wave.

10. Once YAWAUG T is 5/5 at both K and its applicable gates pass, execute the tested §6.9 transaction: add the two YAWAUG row specs with §13 routing, regenerate `model_comparison.md`, then commit and push immediately. Do not repin the running eval campaign.

11. Run `WAVE=rrob MAX_INFLIGHT=16`. Validate-before-skip must skip the landed YAWAUG probe and submit the other 19 R cells. Wait for all to validate.

12. Run `WAVE=all` as a resume/audit sweep. Expected outcome: 42 VALID, zero new submissions, zero in-flight cells, zero invalid cells.

13. Run the reviewed pinned collector with the full campaign SHA and repository root. Require G1–G5 PASS before accepting any hypothesis numbers. Produce both `yaw_aug_results.md` and strict JSON; then produce analysis, HTML/assets, params/command updates, and `commits_yaw_aug.md`.

## Scientific-integrity conclusion

Seed pairing, rotation-seed assignment, exact 6,337-item stream accounting, fixed checkpoint endpoint, audio-to-audio R@1, and the grid’s validate-before-skip behavior are sound. The remaining integrity risks are concrete rather than speculative: unpinned executable shell/control logic, K=1 multiplicity inflation, V-cell gate leakage, omitted ratified `Invalid T60`, and the direct-submit duplicate path.

Committed evidence reports 89 collector tests passing ([pytest transcript:102](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-16_16-08-48_pytest_evalr2fix.log:102)), 174/0/0 strict guard results ([guard transcript:205](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-16_16-13-13_guardtests_evalr2fix.log:205)), and 175/175 union coverage ([union transcript:5](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-16_16-14-44_union_coverage_evalr2fix.log:5)). Per the review constraints, I did not rerun them or execute any driver/DRYRUN/checkpoint-loading operation.