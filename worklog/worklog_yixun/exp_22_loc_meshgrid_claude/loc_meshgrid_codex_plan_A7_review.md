**Reviewer:** OpenAI Codex (gpt-5.6-sol, xhigh, codex exec, read-only sandbox) · **Date:** 2026-11-05 · **Round:** plan A7 (exp_22 R2: the no-flash backend for the reopened command list)

**Verdict:** REVISE

A7 has the correct backend mechanism and four-file PIN2 constitution, but the command transformation, backend-to-artifact binding, and watcher exclusivity/recovery contract are not yet complete enough to arm.

## B1 — FAIL: mechanism and exact dispatch

The H35 launcher’s core guarantee is sound:

- It prepends the overlay and checkout, removes inherited `PYTHONPATH`, and passes only the overlay to children ([exp35_no_flash_launch.py:84–104](/home/yixunhu/codespace/exp-35-orbitrir-yaw/overlays/no_flash/exp35_no_flash_launch.py:84), [181–198](/home/yixunhu/codespace/exp-35-orbitrir-yaw/overlays/no_flash/exp35_no_flash_launch.py:181)).
- It verifies that this checkout’s transformer imported with `flash_attn_func is None` ([exp35_no_flash_launch.py:137–154](/home/yixunhu/codespace/exp-35-orbitrir-yaw/overlays/no_flash/exp35_no_flash_launch.py:137)); that selects `F.scaled_dot_product_attention` ([transformer.py:429–440](/home/yixunhu/codespace/FLAC/src/models/transformer.py:429)).
- H35 tests forked and spawned DataLoader workers and verified the overlay origin and SDPA state in both ([test_exp35_no_flash_launch.py:390–405](/home/yixunhu/codespace/exp-35-orbitrir-yaw/src/tests/test_exp35_no_flash_launch.py:390)).
- `runpy` sets the target’s original argument vector. `localize_meshgrid.main()` consequently still sees only its original arguments ([localize_meshgrid.py:584–594](/home/yixunhu/codespace/FLAC/localize_meshgrid.py:584)); emit/run records strip and compare the same launch flags as before ([meshgrid_offgrid_probe.py:560–573](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_offgrid_probe.py:560), [614–620](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_offgrid_probe.py:614)). PIN, tracked-clean, hostname, CUDA-runtime UUID and `--expect-arm` semantics therefore survive.

A7’s proposed command rewrite is not executable as written:

1. The launcher accepts a filesystem target and calls `runpy.run_path`; it does not implement Python `-c` or `-m` ([exp35_no_flash_launch.py:169–199](/home/yixunhu/codespace/exp-35-orbitrir-yaw/overlays/no_flash/exp35_no_flash_launch.py:169)). The governing list still contains re-verification/arithmetic `-c` lines ([worklog:707,713–714](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_22_loc_meshgrid_claude/loc_meshgrid_worklog.md:707)) and report/crossarm `-m` lines ([worklog:731–733](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_22_loc_meshgrid_claude/loc_meshgrid_worklog.md:731)). A7 nevertheless says all CPU lines are wrapped “the same way” ([plan:106](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_22_loc_meshgrid_claude/plan_loc_meshgrid_r2_bf_yaw.md:106)).

2. A7’s exact sentinel ends in `target=localize_meshgrid.py`, but pytest necessarily emits the actual pytest `__main__.py` target. Any helper target would likewise have another value.

3. `CUDA_VISIBLE_DEVICES=<card>` plus unchanged arguments breaks the rest15 shard: the existing command uses `--device cuda:1` ([worklog:720–726](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_22_loc_meshgrid_claude/loc_meshgrid_worklog.md:720)); exposing one physical card makes only logical `cuda:0` available.

Concrete fix: publish a complete authoritative A7 command list, not a prose substitution rule. Wrap all model-importing `localize_meshgrid.py` and pytest lines. Either explicitly exempt the pure CPU `-c`/`-m` operations and retain them exactly, or add a reviewed file-target adapter that reproduces `-m`/`-c` semantics. Match the sentinel’s full line to the actual target. For later two-shard passes, preserve `cuda:0/1` by exposing both cards, or explicitly re-register single-visible-card `cuda:0` arguments.

## B2 — PASS: PIN2 constitution

The four H35 files are the right non-record delta, byte-identically:

| Path | SHA-256 |
|---|---|
| `overlays/no_flash/flash_attn.py` | `0d0e52ca693e236a2e0a44cd61f61539fdf995cff5132d1f621181a0f3e7c023` |
| `overlays/no_flash/exp35_no_flash_launch.py` | `bbd2847d3a50685420750d066195c8656a1509b4e905883886c0568d244043de` |
| `src/tests/test_exp35_no_flash_launch.py` | `d61a3c02a7e1295a48b3edb60e8abccab508cadb6f4c11bdca9872234bd961dd` |
| `pyproject.toml` | `6422eae5482773db252372760825749f41ac061487be892e350fda997b619a74` |

The current `pyproject.toml` differs from H35 only by the intended `exclude = ["overlays*"]` and final newline. The launcher requires `overlays/no_flash` to contain exactly its two files ([launcher:62–81](/home/yixunhu/codespace/exp-35-orbitrir-yaw/overlays/no_flash/exp35_no_flash_launch.py:62)); the copied test enforces that exact tracked set ([test:125–131](/home/yixunhu/codespace/exp-35-orbitrir-yaw/src/tests/test_exp35_no_flash_launch.py:125)). Therefore no exp_22 helper may be placed under `overlays/`.

No gate or report constant expects `4576b562`; launch provenance requires one consistent live SHA, not that old SHA ([meshgrid_segments.py:49–70](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_segments.py:49)). PIN2 launch records will correctly carry PIN2. A later exp_35 merge meets identical content on the four paths.

The watcher, adapters and their tests should live under the exp_22 worklog and be committed at PIN2, while the non-record diff remains exactly these four paths as A7 specifies.

## B3 — FAIL: backend evidence is not bound to artifacts

Exactly one sentinel plus a successful pipeline proves that the logged process entered its target after the launcher asserted SDPA. It does not bind that log to the evidence JSON subsequently cited.

This matters because gate and probe writers atomically replace existing paths rather than refusing pre-existing output ([meshgrid_engine.py:1935–1941](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_engine.py:1935), [2390–2409](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_engine.py:2390)). A log could be moved, edited, or paired with another JSON after the sentinel count passed.

Concrete fix:

- Hard-stop if an expected log or output artifact already exists.
- After each successful line and re-verification, write an immutable backend-dispatch record containing PIN2, full target/argv, relevant environment, GPU UUID, exit code, exact sentinel, log path and SHA-256, output path and SHA-256, and launcher/overlay hashes.
- Hash that record and append its hash to the worklog. For tests, bind the log to the exact eleven node IDs; for the probe and six gates, bind each produced JSON.
- Use an exact full-line sentinel comparison with the actual target, not a substring count.
- Apply the same scheme later to launch-record emit/run logs and their launch-record SHA.

The frozen evidence JSON schema need not change.

## B4 — FAIL: watcher cannot yet guarantee exclusivity

The 30-minute no-process rule, reservation awareness, no signals, stop-on-failure and no-shard scope are correct foundations. They are not sufficient by themselves.

Required fixes:

- Recheck the compute table, every reservation and exp_35 state after winning the O_EXCL reservation and immediately before every spawn. If anything changed, authenticate and release only the exact exp_22 record and reset the 30-minute clock.
- Authenticate reservation bytes and inode before unlinking, following H35’s pattern ([exp35_supervisor.py:295–348](/home/yixunhu/codespace/exp-35-orbitrir-yaw/worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude/exp35_supervisor.py:295)).
- Treat GPU 0 as unavailable whenever the exp_35 supervisor lock is held and phase (i) is incomplete with state absent, unreadable, `lane=None`, or shared GPU 1—not merely when a readable state already says shared GPU 1. H35 can hand over from that initial state ([exp35_supervisor.py:588–667](/home/yixunhu/codespace/exp-35-orbitrir-yaw/worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude/exp35_supervisor.py:588)).
- Do not admit GPU 1 until the known s43 table job has terminally completed; a transient 30-minute gap alone is not an adequate queue-completion proof. Likewise, establish exp_35 phase-(ii) reservations or an authenticated phase-(ii) intent/lock before exp_22 is eligible, so O_EXCL timing cannot give exp_22 priority.
- Persist state before spawn: command, PID, `/proc` start time, cwd, cmdline hash, card UUID, reservation hash, log and expected output. On watcher death, leave the reservation in place while a child may survive. Restart must refuse/adopt by exact identity, never infer from PID alone and never delete a stale-looking reservation automatically.
- Hold a single-instance lock. Simulated asynchronous exceptions between reservation/state/spawn and during a line must leave a recoverable busy state.
- Run the watcher as a reviewed detached supervisor with heartbeat/state/logging; it cannot depend on a session’s two-hour background lifetime.
- Poll during each line and latch “no further command” if an unexpected process appears. With the mandated no-signal rule, absolute prevention of a malicious/uncooperative late co-tenant is impossible; the defensible guarantee is cooperative reservation exclusivity plus no launch on a known busy/queued lane.

## B5 — PASS: order and scientific scope

The controlling decisions remain unchanged:

- Eleven GPU-only tests → GV/G1/G2/G3/G3D/G0, each immediately re-verified → BF probe → A5/A6 arithmetic.
- A5/A6 retain `0.5145`, the 150-hour cap, and the existing one-/two-lane factors ([plan:103–104](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_22_loc_meshgrid_claude/plan_loc_meshgrid_r2_bf_yaw.md:103)).
- The watcher starts no shard. YAW and BF passes still require Yixun’s subsequent explicit go ([worklog:843–844](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_22_loc_meshgrid_claude/loc_meshgrid_worklog.md:843)).

Pre-register now:

- the watcher uses the one-card probe and factor `1.5`; it must not silently switch to A6’s two-lane form;
- exact eleven node IDs, `pytest -x`, card-selection priority by UUID, poll interval, fresh output paths and all hard-stop conditions;
- the backend-dispatch record schema described under B3.

No scientific threshold or estimand change is needed.

## B6 — FAIL: Coder round contents

The Coder round must contain:

1. PIN2 with the four exact H35 non-record files and no meshgrid production change.
2. A fully expanded authoritative A7 command list resolving every target, sentinel and CUDA-visibility case.
3. A detached exp_22 watcher/supervisor, arming script and durable state/reservation implementation.
4. A small backend-dispatch record helper binding logs to artifacts.
5. TDD coverage for:
   - localize argv and emit/run launch-record identity through `runpy`;
   - pytest `__main__.py` and all eleven exact node IDs;
   - dynamic sentinel targets and refusal/nonzero propagation;
   - output/log no-clobber and hash binding;
   - process/reset timing, unreadable nvidia/reservation/state inputs, O_EXCL races and post-acquisition rechecks;
   - exp_35 initial/shared/hand-over/phase-(ii) states and the s43 table job;
   - crash/restart, PID reuse, surviving child, exact-record release and two-hour-independent detachment;
   - stop at the first failure and proof that no shard command is reachable.

Codex code review must pass before arming. No new Yixun decision is required.

## Ranked findings

- **BLOCKING 1:** A7 does not yet define executable, semantics-preserving targets for `-c`, `-m`, pytest, and the `cuda:1` shard under single-card visibility. Fix with a fully expanded command list and explicit adapters/exemptions.
- **BLOCKING 2:** Sentinel-only logs are not cryptographically joined to the gate/probe JSONs. Add fresh-output checks and hashed backend-dispatch records.
- **BLOCKING 3:** Reservation acquisition, exp_35 initial/hand-over state, crash recovery and exact release have uncovered windows. Implement the durable fail-closed supervisor contract above.
- **HIGH 1:** Explicitly gate the known GPU-1 table job and exp_35 phase-(ii) priority; 30-minute idleness alone does not prove those scheduled consumers are finished.
- **HIGH 2:** The watcher needs a detached launch/heartbeat mechanism independent of the two-hour session limit.
- **NIT:** Keep the byte-identical `EXP35_NO_FLASH_LAUNCHER` name; do not rename it for exp_22. Only its expected `target=` field must be dynamic.

## What I verified

Read-only probes only:

- `git status`, `rev-parse`, `log`, `show`, `diff`, `merge-base`, `merge-tree`, `rg`, `sed`, `nl`, `find`, and SHA-256 streams from `git show`.
- A7 at requested commit `4b2c718e`; tracked tree was clean.
- `localize_meshgrid.py` and `src/localization/` remain byte-identical to `701926e5` and contain no flash reference.
- `localize_meshgrid.py`, `eval_localization.py`, and `src/models/transformer.py` match their H35 bytes.
- The four H35 hashes above and the current-to-H35 `pyproject.toml` delta.
- `build_provenance` is called by `eval_localization.py`, not the meshgrid driver.
- The eleven exact failed node IDs from the Nov 4 triage.
- During review, `localization-exp` advanced from `4b2c718e` to records-only child `7ecac76d`; it did not alter A7 or any reviewed production path.

I did not run pytest, any launcher, driver, gate, probe, GPU query/command, or inspect/signal/attach to a running process.
