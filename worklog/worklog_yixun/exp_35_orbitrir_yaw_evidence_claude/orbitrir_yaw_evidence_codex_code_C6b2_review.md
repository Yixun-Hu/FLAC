**Reviewer:** OpenAI Codex (gpt-5.6-sol, xhigh, codex exec, read-only sandbox) · **Date:** 2026-11-06 · **Round:** code C6b-2 (exp_35 round 6b: the C6b fixes; 7e34a61c → ca65b463)

**Verdict:** SHIP

## Findings

No BLOCKING or HIGH findings.

1. **NIT — the whole-range `git diff --check` claim is inaccurate.** The command reports trailing whitespace in captured worklogs, beginning at [H1 RED log:15](/home/yixunhu/codespace/exp-35-round6b/worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude/orbitrir_yaw_evidence_2026-11-06_00-40-34_r6b_C6b_H1_enforced_pins_red.log:15), and also at [real-data readback:7](/home/yixunhu/codespace/exp-35-round6b/worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude/orbitrir_yaw_evidence_2026-11-06_00-57-22_r6b_C6b_real_data_readback_at_b9e0e79e.log:7). The five Python files themselves pass `git diff --check`.

   **Concrete correction:** preserve the verbatim evidence logs, but change the audit statement to “`git diff --check -- '*.py'` clean; whole-range check reports whitespace in captured logs.”

## F1–F6

### F1 — RESOLVED

- `H35` and `EXP25_PIN` are literal production constants at [yaw_offorbit_sources.py:51](/home/yixunhu/codespace/exp-35-round6b/analysis/yaw_robustness/yaw_offorbit_sources.py:51).
- `context()` has no pin parameter at [yaw_offorbit_sources.py:108](/home/yixunhu/codespace/exp-35-round6b/analysis/yaw_robustness/yaw_offorbit_sources.py:108).
- Cell verification selects only those constants at [yaw_offorbit_sources.py:199](/home/yixunhu/codespace/exp-35-round6b/analysis/yaw_robustness/yaw_offorbit_sources.py:199), while E1/E2 explicitly use `H35` at [yaw_offorbit_sources.py:373](/home/yixunhu/codespace/exp-35-round6b/analysis/yaw_robustness/yaw_offorbit_sources.py:373).
- The run refuses if the frozen collector’s exp_25 constant differs at [yaw_offorbit_sources.py:367](/home/yixunhu/codespace/exp-35-round6b/analysis/yaw_robustness/yaw_offorbit_sources.py:367).
- Neither CLI exposes `--pin`; the coherent-other-pin fixture is refused and succeeds only after the test deliberately changes the registered fixture constant at [test_plot_yaw_offorbit.py:779](/home/yixunhu/codespace/exp-35-round6b/src/tests/test_plot_yaw_offorbit.py:779).

The fixture-only `S.H35` patches do not create a production override.

### F2 — RESOLVED

- `_snapshot()` records bytes plus `(st_dev, st_ino)` from the same descriptor at [yaw_offorbit_sources.py:169](/home/yixunhu/codespace/exp-35-round6b/analysis/yaw_robustness/yaw_offorbit_sources.py:169).
- Metrics, stream, receipt, applicable canon sidecar, and receipt-named log are captured once; the collector receives the exact parsed snapshot receipt through `receipt_doc` at [yaw_offorbit_sources.py:193](/home/yixunhu/codespace/exp-35-round6b/analysis/yaw_robustness/yaw_offorbit_sources.py:193).
- Every captured file is reread afterward and compared by bytes and identity, while the consumed snapshot metrics/stream/canon are independently hashed to the snapshot receipt at [yaw_offorbit_sources.py:226](/home/yixunhu/codespace/exp-35-round6b/analysis/yaw_robustness/yaw_offorbit_sources.py:226).
- E5 validates its on-disk dump against that same snapshot receipt and parses waveform bytes only after matching its receipt-bound hash at [exp35_e5_offorbit_pairs.py:59](/home/yixunhu/codespace/exp-35-round6b/analysis/yaw_robustness/exp35_e5_offorbit_pairs.py:59).
- Receipt-only, coordinated before/after substitution, and identical-bytes/new-inode cases are covered at [test_plot_yaw_offorbit.py:684](/home/yixunhu/codespace/exp-35-round6b/src/tests/test_plot_yaw_offorbit.py:684); E5’s exact receipt handoff is covered at [test_exp35_e5_offorbit_pairs.py:236](/home/yixunhu/codespace/exp-35-round6b/src/tests/test_exp35_e5_offorbit_pairs.py:236).

The stated residual is acceptable here: an adversarial in-place change restored to identical bytes on the identical inode entirely inside a frozen verifier read cannot be observed externally. Ordinary persistent writes and atomic replacements refuse. This would need stronger hardening under a hostile-writer threat model, but it does not block this experiment workflow.

### F3 — RESOLVED

`canon_index_from_bytes()` implements the collector’s namespace, SHA-256, checkpoint-pattern, and unique-step rules directly on the pinned buffer at [yaw_offorbit_sources.py:79](/home/yixunhu/codespace/exp-35-round6b/analysis/yaw_robustness/yaw_offorbit_sources.py:79). `pinned_canon_index()` parses that buffer without reopening through `load_canon_index`, then verifies that the path still contains those bytes at [yaw_offorbit_sources.py:96](/home/yixunhu/codespace/exp-35-round6b/analysis/yaw_robustness/yaw_offorbit_sources.py:96).

The tests compare the real index with the frozen collector, exercise its malformed cases, forbid `load_canon_index`, and mutate the path between hash and parse at [test_plot_yaw_offorbit.py:808](/home/yixunhu/codespace/exp-35-round6b/src/tests/test_plot_yaw_offorbit.py:808).

### F4 — RESOLVED

The unrequested gate fix is correct and directly applies F2’s invariant to E1/E2:

- CTL and B-F gate cells are snapshot-verified first and cached; exp_25 reference files are held at [yaw_offorbit_sources.py:259](/home/yixunhu/codespace/exp-35-round6b/analysis/yaw_robustness/yaw_offorbit_sources.py:259).
- All held files are reread after E1/E2 at [yaw_offorbit_sources.py:381](/home/yixunhu/codespace/exp-35-round6b/analysis/yaw_robustness/yaw_offorbit_sources.py:381).
- Gate-cell failures now enter `reasons`; de-duplication prevents repeated reporting at [yaw_offorbit_sources.py:430](/home/yixunhu/codespace/exp-35-round6b/analysis/yaw_robustness/yaw_offorbit_sources.py:430).
- Tests cover the CTL metrics, B-F receipt, and exp_25 reference stream race classes at [test_plot_yaw_offorbit.py:840](/home/yixunhu/codespace/exp-35-round6b/src/tests/test_plot_yaw_offorbit.py:840).

This fixes the prior path where a control-cell failure discovered after E1 could be silently ignored.

### F5 — RESOLVED

Readings 7 and 8 now hold.

- Reading 7: all used cells are bound to their registered pin, stable snapshot, receipt artifact hashes, stream registry, backend proof, and one campaign digest; CTL cells participate in that digest.
- Reading 8: both pinned manifests are parsed from the bytes whose hashes were checked; the CANON index no longer rereads before parsing.

No static regression found in:

- legacy/c4 dispatch, which remains isolated from off-orbit mode at [plot_yaw_curves.py:447](/home/yixunhu/codespace/exp-35-round6b/analysis/yaw_robustness/plot_yaw_curves.py:447);
- mean/sample-SD statistics at [yaw_offorbit_sources.py:442](/home/yixunhu/codespace/exp-35-round6b/analysis/yaw_robustness/yaw_offorbit_sources.py:442);
- E5 pair orientation, equal-input pairing, metric differences, or distance summaries at [exp35_e5_offorbit_pairs.py:92](/home/yixunhu/codespace/exp-35-round6b/analysis/yaw_robustness/exp35_e5_offorbit_pairs.py:92).

The committed regression log reports 509 passed. Phase (ii) and the final production figure/E5 remain unrun, as expected.

### F6 — SHIP

After phase (ii), run from `/home/yixunhu/codespace/exp-35-round6b`:

```bash
CUDA_VISIBLE_DEVICES= /home/yixunhu/miniconda3/envs/flac/bin/python -B analysis/yaw_robustness/plot_yaw_curves.py --mode offorbit --exp35-base-dir /home/yixunhu/codespace/exp-35-orbitrir-yaw --exp25-base-dir /home/yixunhu/codespace/exp-25-depth-canon

CUDA_VISIBLE_DEVICES= /home/yixunhu/miniconda3/envs/flac/bin/python -B analysis/yaw_robustness/exp35_e5_offorbit_pairs.py --exp35-base-dir /home/yixunhu/codespace/exp-35-orbitrir-yaw --exp25-base-dir /home/yixunhu/codespace/exp-25-depth-canon
```

## What I verified

- HEAD is `ca65b463`; merge-base with the requested start is `7e34a61c`; exactly five commits are in range.
- All 5/5 commits carry the Claude Opus 5.5 trailer.
- Excluding worklogs, exactly the three analysis scripts and two tests changed.
- `eval_FLAC.py`, `overlays/`, non-test `src/`, and `worklog/.../exp_25_depth_canon_claude/` are unchanged versus `5b065801`.
- The AST-based committed counter reproduces every Coder count exactly:
  - HIGH 1: production `+14/−15`, tests `+51/−16`
  - HIGH 2: `+64/−32`, tests `+56/−12`
  - HIGH 3: `+20/−2`, tests `+25/−0`
  - gate fix: `+37/−16`, tests `+25/−0`
- Regression log SHA-256 is `bba714bbfaa09d2186a263f26688271f3a82270c53b0407de6a536b0ac5b9c9e` and reports `509 passed`.
- Readback log SHA-256 is `6ad41a5a8e280b8e642959ea3d6b4c558fc6654c62b2f8bc72f0269143e8cb83`; its embedded output reports E1 equal at K1/K8, E2 PASS 14/14, no gate reasons, exactly 238 expected absence reasons over 70 unrun cells, 22 exp_25 plus 10 exp_35 verified cells with one digest per campaign, and both sampled NAS dumps bound.
- Worktree is clean.

Read-only probes used: `git status/log/show/diff/merge-base/rev-list/diff --check`, `rg`, `sed`, `nl`, `sha256sum`, and the committed AST line counter under `python -B`. I did not run pytest, the figure, E5, the collector, runner, readback snippet, any GPU/process command, or inspect the running phase.
