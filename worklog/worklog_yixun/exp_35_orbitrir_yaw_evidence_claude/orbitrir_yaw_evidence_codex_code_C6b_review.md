**Reviewer:** OpenAI Codex (gpt-5.6-sol, xhigh, codex exec, read-only sandbox) · **Date:** 2026-11-06 · **Round:** code C6b (exp_35 round 6b: the off-orbit figure + E5; 5b065801 → 7e34a61c)

**Verdict:** REQUEST-CHANGES

## Findings

No BLOCKING findings. Three HIGH provenance defects prevent SHIP.

1. **HIGH — H35 is a default, not an enforced registered pin.**  
   [`yaw_offorbit_sources.py:81`](/home/yixunhu/codespace/exp-35-round6b/analysis/yaw_robustness/yaw_offorbit_sources.py:81) accepts an arbitrary `pin`; [`:369`](/home/yixunhu/codespace/exp-35-round6b/analysis/yaw_robustness/yaw_offorbit_sources.py:369) exposes it as `--pin`; and [`:379`](/home/yixunhu/codespace/exp-35-round6b/analysis/yaw_robustness/yaw_offorbit_sources.py:379) forwards it without requiring `pin == H35`. A complete, internally consistent set of receipts produced at another commit would therefore pass, even though §6 registers all exp_35 cells at H35. The existing “wrong pin” test only changes the requested pin while leaving fixture receipts at the original pin, proving mismatch detection—not binding to H35.

   **Fix:** reject any exp_35 pin other than `H35` before verification, preferably removing the production `--pin` override. Add a test whose complete fixture is coherently stamped at another 40-hex pin and require refusal.

2. **HIGH — the receipt consumed by the new code is reread after verification but is never itself revalidated.**  
   The frozen collector validates the files at [`depth_canon_collect.py:1797`](/home/yixunhu/codespace/exp-35-round6b/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_collect.py:1797), then independently rereads metrics, stream, and receipt at [`:1801`](/home/yixunhu/codespace/exp-35-round6b/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_collect.py:1801). The wrapper consumes that second receipt for backend, dump evidence, and campaign digest at [`yaw_offorbit_sources.py:158`](/home/yixunhu/codespace/exp-35-round6b/analysis/yaw_robustness/yaw_offorbit_sources.py:158), while checking only the second metrics/stream hashes against it. A coordinated replacement of metrics, stream, and receipt between the verification and reread can therefore introduce unverified metric values, backend provenance, and campaign digest. The race tests at [`test_plot_yaw_offorbit.py:645`](/home/yixunhu/codespace/exp-35-round6b/src/tests/test_plot_yaw_offorbit.py:645) mutate only `metrics_raw` or `stream_raw` without updating the parsed receipt, so they do not cover this case.

   **Fix:** construct one stable snapshot of metrics/stream/receipt, validate that exact receipt through the collector, and bracket validation with byte-for-byte rereads so movement refuses. Add receipt-only and coordinated metrics+stream+receipt substitution tests.

3. **HIGH — the CANON checkpoint index is not parsed from the bytes whose hash was checked.**  
   [`yaw_offorbit_sources.py:77`](/home/yixunhu/codespace/exp-35-round6b/analysis/yaw_robustness/yaw_offorbit_sources.py:77) verifies the manifest bytes, discards them, and [`:78`](/home/yixunhu/codespace/exp-35-round6b/analysis/yaw_robustness/yaw_offorbit_sources.py:78) calls `load_canon_index(path)`, which reopens the path. A change between those reads can supply an index other than the pinned bytes. The current test only alters the file before verification.

   **Fix:** parse the raw bytes returned by `_pinned_bytes` directly, using the collector’s validation rules, and test mutation between hash and parse.

## K1–K9

| Item | Status | Review |
|---|---|---|
| **K1** | **FAIL** | Layout, source mapping, H6 fields, N7 values, E2 source switch, and A2 E5 placement otherwise conform. The configurable exp_35 pin violates the registered H35 source identity. |
| **K2** | **FAIL** | Metrics/stream hashing, registry checks, backend checks, missing/duplicate refusal, and campaign membership are well structured, but findings 1–3 leave pin and snapshot provenance fail-open. |
| **K3** | **PASS** | Source selection and `statistics.mean/stdev` are correct. E5 pairs on equal `input_hash`, computes cell−reference differences, calls `_paired(cell, reference)`, reports mean/median/p99/max for relL2/relL∞, and handles the CANON cross-pin comparison correctly. |
| **K4** | **PASS** | Frozen H35 files are untouched. Git diff and the `git show 5b065801` regression test support unchanged legacy/c4 behavior; only argument dispatch and the new off-orbit path were added. |
| **K5** | **FAIL** | The committed RED/GREEN evidence, real-schema fixtures, deletion/tamper cases, legacy byte comparison, and real-data readback are strong. Tests do not cover the three HIGH cases above. |
| **K6** | **PASS** | Input worktrees/NAS are read-only in the implementation; figure/E5 output creation occurs after input collection succeeds; refusal is exit 3. Invoke the hand-back commands through `/home/yixunhu/miniconda3/envs/flac/bin/python -B` from the round-6b checkout. |
| **K7** | **PASS** | Worktree clean; 9/9 trailers; diff hygiene clean. Production-code changes by the committed counter are 166, 80, 89, 198, 5, 9, and 25 changed lines; the last two commits are logs only. No environment-changing code or dependency change exists in the range. |
| **K8** | **FAIL** | Readings 7 and 8 are not fully fail-closed; assessment below. |
| **K9** | **REQUEST-CHANGES** | Do not run the real figure/E5 or merge this round with the campaign records until the three HIGH findings are fixed, regression-tested, and re-reviewed. |

## The 17 readings

1. **Agree.** θ=0 FLAC/CANON use full-precision pinned JSON values, each type-strictly cross-checked against the corresponding cell record.
2. **Agree.** Raw/summary metrics are the six plotted metrics plus FD; no geometry retrieval metrics enter.
3. **Agree.** A6 vanilla is `n/a`/recorded and A6 OrbitRIR is `batched` with the specified inferred basis; receipt-backed cells use recorded values.
4. **Agree.** exp_35 backend comes from receipt-v2 plus log, exp_25 from the log’s SDPA pair, and A6 flash/pin metadata follows the registered historical assertion.
5. **Agree.** `a6` requires E2 PASS and absence of every E2-FAIL artifact; `one-pin` requires E2 FAIL and the complete 16-cell block.
6. **Agree.** Both scripts require E1 equality at K1 and K8 through `collect_points`.
7. **Disagree as an unqualified fail-closed claim.** The intended cell coverage—including CTL and E5 pair cells—and exp_25 registry checks are present, but the campaign digest is taken from the unvalidated second receipt snapshot described in HIGH 2.
8. **Disagree.** The cells manifest is parsed from pinned bytes; the CANON manifest is hash-checked and then independently reread, as in HIGH 3.
9. **Agree.** Summary mean/sample SD use `statistics`.
10. **Agree.** Provenance row keys and paths are explicit and absolute.
11. **Agree.** Off-orbit mode returns before the legacy README writer.
12. **Agree.** E5 P1-vs-angle reuses the figure’s FLAC series and computes differences of means without inventing A6 seed pairing.
13. **Agree.** E5 first verifies the entire figure point set.
14. **Agree.** Pair orientation is rotated cell minus the registered B-F 45° or CANON 0° reference.
15. **Agree.** The disclosure names both pins, identifies E1 as another-arm evidence, and states that no same-pin CANON 0° cell exists.
16. **Agree as an implementation reading.** CANON is `#1BAF7A` with triangles; no Node/npm/npx runtime was available, so palette-validator certification was not possible. Marker shape supplies an independent identity channel.
17. **Agree.** The title is exactly “Off-Orbit Yaw Robustness”.

## Planner audit and evidence verified

The repository portion of the Planner audit is confirmed:

- HEAD/base/merge-base are exactly `7e34a61c`, `5b065801`, and `5b065801`.
- Nine commits are in range; all nine have the Opus 5.5 trailer.
- Excluding worklog, the diff contains exactly the three analysis files and two new tests.
- No frozen collector, manifest/results, runner, overlay, `eval_FLAC.py`, or non-test `src/` file changed.
- `git diff --check` is clean.
- Runner `RELEVANT` excludes `src/tests` and contains no `analysis/` path, so C6b cannot change the running campaign digest.
- Pinned hashes directly recomputed as:
  - cells manifest `440eb1be…7037d`
  - CANON manifest `1e481447…18cba`
  - exp_25 JSON `ec3fcd08…2eb6e0`
  - registry `9ec31f32…65a24`
  - A6 CSV `0ae554c0…fb23da`
- Manifest counts are 346 total cells, 116 exp_35 cells.
- Final regression log hashes to `556afdfd…` and reports `496 passed`.
- Final readback log hashes to `f7d05e41…`; its embedded snippet reports E1 equal, E2 PASS 14/14, 213 missing-artifact reasons over exactly 71 unrun cells, one digest per observed campaign, and two real dump byte checks.
- No committed or local real `yaw_offorbit_metrics*` or `orbitrir_yaw_e5.{json,md}` outputs exist.
- Worktree is clean and the branch has no configured upstream.

Probes used were read-only: `git status/rev-parse/merge-base/log/show/diff`, `rg`, `sed`, `nl`, `wc`, `sha256sum`, `jq`, `find`, `command -v`, and the read-only line counter under `PYTHONDONTWRITEBYTECODE=1`/`python -B`. I did not run pytest, either analysis script, the collector, the supplied readback, any evaluator/runner, or any process/GPU probe.
