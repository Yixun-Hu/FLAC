**Reviewer:** OpenAI Codex (gpt-5.6-sol, xhigh, codex exec, read-only sandbox) · **Date:** 2026-11-05 · **Round:** plan A7 r2 (exp_22 R2: the no-flash backend, revised)

**Verdict:** REVISE

The detector + Planner-run launcher is a faithful implementation of Yixun’s instruction in principle; an autonomous GPU-taking watcher is not required. The detector can autonomously notice availability while the reviewed launcher remains the sole authority that reserves and launches. However, A7 r2 still has one immediately fatal sentinel rule and several acquisition/liveness details that must be made executable before arming.

## R1 — FAIL

Most of BLOCKING 1 is resolved at [plan_loc_meshgrid_r2_bf_yaw.md:107](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_22_loc_meshgrid_claude/plan_loc_meshgrid_r2_bf_yaw.md:107):

- The pinned pytest target exists at `/home/yixunhu/miniconda3/envs/flac/lib/python3.10/site-packages/pytest/__main__.py` and is a valid `runpy.run_path` target.
- The eleven selected cases exactly equal the eleven unique failed node IDs in the triage log; no IDs are missing or extra. Its SHA-256 is the registered `08c362ed…`.
- `-x`, `-q`, and `-p no:cacheprovider` are executable.
- Every G line keeps the governing arguments and uses logical `--device cuda:0` behind one physical `CUDA_VISIBLE_DEVICES` card.
- V and A are correctly exempted as pure-CPU `-c` operations.
- P is correctly restricted to the one-card form and factor 1.5.
- Deferring all shard/report/crossarm lines is correct. Re-registering the shards later with both cards visible and logical `cuda:0`/`cuda:1` preserves the original layout.
- A fresh gate out-dir changes only where new evidence is written. The G0 replay input remains `i1_P1_CRN_br256_20260825_194053_merged`, and the context manifest, audit, checkpoints, registered anchor and all protocol arguments remain unchanged ([worklog:700–715](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_22_loc_meshgrid_claude/loc_meshgrid_worklog.md:700)).

The remaining failure is admission of T: pytest returns zero when selected tests skip. A7 records the summary but does not require all eleven cases to have run and passed. Missing assets or a broken cache/symlink could therefore yield a successful pre-flight without re-verifying the failures.

Concrete fix: require exactly eleven collected and eleven passed, with zero skipped, failed, errored, xfailed or deselected cases. Prefer a fresh `--junitxml` artifact, validate its counts and exact node IDs, and include its path/SHA-256 in T’s dispatch record.

## R2 — FAIL

The proposed dispatch record is otherwise sufficient to bind an accepted line to its artifacts: O_EXCL creation, full argv/environment/card identity/times/rc, exact sentinel, log hash, output hashes, launcher/overlay hashes, followed by committing the record hash beside the evidence hashes, closes the substitution problem identified in A7 r1.

But the “log’s FIRST line equals the sentinel” condition at [plan:107](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_22_loc_meshgrid_claude/plan_loc_meshgrid_r2_bf_yaw.md:107) is false for the pinned launcher:

- The launcher calls `transformer_reasons()` before printing the sentinel ([exp35_no_flash_launch.py:188–193](/home/yixunhu/codespace/exp-35-orbitrir-yaw/overlays/no_flash/exp35_no_flash_launch.py:188)).
- Importing the transformer prints the overlay `ImportError`, then `flash_attn not installed, disabling Flash Attention` ([transformer.py:12–18](/home/yixunhu/codespace/FLAC/src/models/transformer.py:12)).
- The pinned launcher test explicitly requires that pair before the sentinel ([test_exp35_no_flash_launch.py:107–112](/home/yixunhu/codespace/exp-35-orbitrir-yaw/src/tests/test_exp35_no_flash_launch.py:107)).

Thus every correct T/G/P launch would fail the A7 r2 check.

Concrete fix: keep the byte-identical launcher and `EXP35_NO_FLASH_LAUNCHER` name. Require:

1. Exactly one full-line sentinel equal to the dynamically constructed target-specific value.
2. Exactly one ordered fallback pair before it.
3. No `NO-FLASH LAUNCH REFUSED`.
4. Hash the complete log in the dispatch record.

The pinned launcher hash and code order establish that the sentinel precedes `runpy`, so it need not be the first physical log line.

There is also a record-location ambiguity: T precedes GV, but every dispatch record is placed in `<run dir>` while the gate run directory “must not exist before GV.” Define a separate fresh orchestration directory such as `r2_preflight_<RUN>/dispatch/`; keep `r2_gates_<RUN>` and `r2_probe_BF_<RUN>` strictly scientific-output directories.

## R3 — FAIL

The architectural change is acceptable and faithful. It does not require retaining an autonomous GPU-taking watcher.

The following parts are sound:

- GPU 0 is excluded whenever the exp_35 flock is actually held, regardless of state contents. A dead supervisor releases the flock even if the lock file remains. The existing reviewed implementation demonstrates the correct read-only probe: open the existing file read-only and attempt a nonblocking shared flock ([exp35_supervisor.py:976–990](/home/yixunhu/codespace/exp-35-orbitrir-yaw/worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude/exp35_supervisor.py:976)).
- GPU 1’s exact `EVAL_DONE ACEattn s43 ` evidence plus absence of `exp30_attn_eval.sh` prevents admission during a late-starting or still-running s43 table evaluation.
- Planner-written phase-(ii) O_EXCL reservations give exp_35 priority before its runner starts.
- O_EXCL acquisition, post-acquisition checks, pre-line checks, foreign-process latching, no signalling, authenticated byte/inode release, and crash-stays behavior are the correct fail-closed policy.
- A future GPU-0 training job is not an implicit blocker once the registered 30-minute hand-over window passes. Any known queued job that must retain priority needs an intent/reservation before the detector declares candidacy.

Three details remain open:

1. The detector’s 30-minute observation is not durably or freshly bound to the launcher. Because the detector never writes, the launcher has no specified certificate showing the card was empty at every 60-second poll. “Re-check live” could mean only one instantaneous query—or, read literally, another 30-minute wait before every line.

   Concrete fix: make the launcher’s capture phase independently collect and seal the 31 empty observations for the chosen UUID, then require launch within a short registered maximum age and recheck immediately before and after O_EXCL. Alternatively, directly pipe the reviewed detector’s timestamped transcript into an O_EXCL capture record and seal its hash.

2. The detector’s execution and wake channel are unspecified. A foreground shell remains subject to the two-hour session limit; a detached detector without a named notification channel cannot wake the Planner.

   Concrete fix: name the session-independent monitor mechanism and wake channel, its single-instance identity, and how the Planner verifies a fresh notification. The detector may remain read-only; only the launcher should write capture/seal/reservation state.

3. The supervisor state transition is not sufficiently ordered. “Reservation and state stay” does not state that durable state exists immediately after O_EXCL and before detachment/spawn, nor that each line is journaled before its child can survive a supervisor crash.

   Concrete fix: specify and fsync an atomic state machine:

   `reserved → supervisor_started → line_pending → child_identified → line_complete → released`

   `line_pending` must contain argv, environment, cwd, card UUID/index, reservation hash/inode, log and expected outputs. `child_identified` must add PID, `/proc` start time, cwd and cmdline hash. Inject failures at every transition; no crash may automatically release or adopt.

The exact absolute exp_35 lock path and both card UUIDs should also be pinned. The reviewed exp_35 launch entry already records them at [exp35_phase_i_launch.sh:23–35](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude/exp35_phase_i_launch.sh:23).

## R4 — FAIL

The scientific pre-registrations are otherwise correct: one-card factor 1.5, exact node IDs, `pytest -x`, 60-second/30-minute detector timing, fresh gate/probe paths, unchanged order and gate arguments, A4–A6, the 150-hour cap, A5-above-cap stopping for Yixun, and no reachable shard.

Before approval, add:

- the exact two UUID literals and UUID-to-current-index resolution;
- “eleven passed, zero skipped” as a hard stop;
- the sealed 30-minute candidate-evidence rule and freshness bound;
- the orchestration/dispatch directory;
- unreadable compute-table, lock, queue-log, intent or reservation inputs are busy/refusal states.

## R5 — FAIL

The Coder round is broad but needs these additions:

- a real pinned-launcher test proving fallback pair → dynamic sentinel → target output;
- pytest success, skip and deselection cases, preferably through the JUnit artifact;
- detector-to-capture freshness and seal binding;
- detector lifetime/wake behavior beyond two hours;
- held, unheld-leftover, absent and unreadable exp_35 lock cases;
- fault injection at every reservation/state/detach/spawn transition;
- a foreign process appearing during a running line, proving the current child is untouched and no later line starts;
- dispatch records stored outside fresh scientific-output directories;
- proof that a nonzero wrapped target is recorded before stop;
- the existing PID-reuse, exact-release and no-shard tests retained.

## R6 — REVISE

The autonomous GPU-taking watcher does not need to return. Correct the sentinel rule and make the detector-to-launch acquisition and liveness handoff explicit; then the detector + Planner-run reviewed launcher is the safer faithful implementation.

## Ranked findings

- **BLOCKING 1:** The first-line sentinel check rejects every valid pinned-launcher run. Match the exact sentinel once after the pinned fallback pair.
- **BLOCKING 2:** The launcher has no authenticated fresh proof of the detector’s 30-minute observation. Seal that observation or independently repeat it during capture.
- **HIGH 1:** T can succeed with skipped tests. Require exactly eleven passed and bind a machine-readable result artifact.
- **HIGH 2:** Detector persistence and its Planner wake channel remain unspecified, so HIGH 2’s two-hour limitation is not fully closed.
- **HIGH 3:** Crash-stays is not backed by an ordered, pre-spawn durable state schema. Define and fault-test every transition.
- **HIGH 4:** `<run dir>` conflicts with the requirement that the gate directory not exist before GV. Use a separate orchestration/dispatch directory.
- **NIT 1:** In the JSON schema, separate `selected_node_ids`, `pytest_summary` and `junit_xml` from `output_paths`; node IDs and a summary line are not paths. Keep the sentinel name unchanged.

## What I verified

Read-only probes only:

- Branch `localization-exp`, HEAD `c6d34d516a568e2e8a969a21ed9dff00b363d091`; tracked worktree and index clean. Existing untracked files were untouched.
- A7 r2 is the final §9 bullet; the matching notebook entry is [loc_meshgrid_worklog.md:852–855](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_22_loc_meshgrid_claude/loc_meshgrid_worklog.md:852).
- The prior A7 review and governing command list at lines 691–733.
- The eleven plan IDs equal the eleven unique triage IDs.
- The pytest `__main__.py` path and contents, without executing it.
- The four PIN2 hashes: overlay `0d0e52ca…`, launcher `bbd2847d…`, launcher test `d61a3c02…`, and `pyproject.toml` `6422eae5…`.
- The pinned launcher’s actual output ordering and its test contract.
- The reviewed `2143fecc` capture/seal/launch pattern, exp_35 flock semantics, O_EXCL reservation creation, authenticated exact release and GPU-0 hand-over path.
- Commands used: `git status`, `rev-parse`, `diff`, `show`, `log`, `cat-file`, `ls-tree`, `merge-base`; `rg`, `nl`, `sed`, `sha256sum`; and `python -B -c` text/spec inspection.

I did not run pytest, any launcher, driver, gate, probe, GPU query/command, or inspect, signal, attach to, or otherwise touch any running process.
