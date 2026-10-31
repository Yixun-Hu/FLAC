**Reviewer:** OpenAI Codex (gpt-5.6-sol, xhigh, codex exec, read-only sandbox) · **Date:** 2026-10-26 · **Round:** code rC (review of fix round rC-F1: shared-GPU launch mode, plan v14)

**Verdict:** REQUEST-CHANGES — no BLOCKING finding; one pre-arm HIGH in the SMOKE-evidence acceptance contract. The shared-GPU gate itself is implemented correctly.

## Findings

### HIGH — FULL does not enforce the new schema-v2 GPU provenance

[depth_canon_launch.sh:296](worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_launch.sh:296) requires `schema_version == 2`, but `NEED` omits:

- `gpu_policy`
- `min_gpu_free_mib`
- `free_mib_at_launch`
- `co_tenants_at_launch`
- also `max_projected_hours == 120`

Thus a schema-2 document with missing, false, wrongly typed, or stale GPU disclosure is accepted by FULL. The launcher-sha binding proves which launcher is expected, but not that an arbitrary mutable JSON was actually emitted unmodified by that launcher. The log hash does not help because FULL does not parse the log’s GPU disclosure.

Concrete fix before arming:

- Pass the current policy, floor, and projection cap into the FULL evidence checker.
- Require `gpu_policy == GPU_OCCUPANCY_POLICY`, `min_gpu_free_mib == MIN_GPU_FREE_MIB`, and `max_projected_hours == MAX_PROJECTED_HOURS`.
- Require real SMOKE evidence to contain exactly `{"0": int, "1": int}`, with both values at least the floor; reject `null`.
- Type-check every co-tenant record as `{gpu: 0|1, pid: positive int, used_mib: nonnegative int, cwd: nonempty str}`.
- Add negative E-cases deleting each key and supplying wrong types/values. Current E10 only proves schema 1 is rejected at [depth_canon_guardtests.sh:483](worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_guardtests.sh:483).

This changes the launcher sha, so a fresh final-HEAD DRY_RUN pass is mandatory.

### NIT — watcher accepts telemetry that the launcher strictly rejects

At [depth_canon_autolaunch.sh:181](worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_autolaunch.sh:181), the watcher accepts arbitrary `used_memory` text. A persistent `[N/A]` is therefore “clean”; the launcher rejects it at [depth_canon_launch.sh:407](worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_launch.sh:407), eventually consuming six SMOKE attempts and entering `FAILED`.

It cannot launch wrongly, and real rows are currently numeric, so I rank this NIT. Fold it into rC-F2: add `nounits`, require numeric MiB, and test `[N/A]` as a dirty poll that waits with an explicit reason.

### NIT — stale v13 prose remains in plan v14

[plan_depth_canon.md:105](worklog/worklog_yixun/exp_25_depth_canon_claude/plan_depth_canon.md:105) still says the watcher requires zero compute processes and 23,040 MiB; [plan_depth_canon.md:184](worklog/worklog_yixun/exp_25_depth_canon_claude/plan_depth_canon.md:184) still says the smoke threshold is 60 h. The v14 clauses and changelog correctly supersede these, but the stale operational paragraphs should be updated.

### HIGH for the later scientific claim, not caused by rC-F1

The Planner’s gate characterization is correct: the production audit gate rebuilds the census and summaries, then hashes map bytes at [depth_canon_audit_gate.py:277](worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_audit_gate.py:277) and [depth_canon_audit_gate.py:293](worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_audit_gate.py:293); it does not rerun `audit_map`’s cloud construction at [depth_canon_tie_audit.py:233](worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_tie_audit.py:233).

Consequently, the intermittent cloud mismatch does not invalidate the gate’s byte-integrity claim, but it does leave the stronger registered-audit/Tier-1 premise at risk. Tier 1 will detect unequal hashes rather than silently pass, but “equal by construction” is not established until the mismatch is localized and shown not to affect the recovered proxy, candidates, `j_star`, or canonical hash. This does not arise from rC-F1 and need not block shared-mode training, but it should block an unqualified Tier-1 exactness claim. The promised `issue_report.md` investigation entry is not present at current HEAD.

## C1–C8

- **C1 — RESOLVED.** Literals are pinned at [depth_canon_launch.sh:69](worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_launch.sh:69), [depth_canon_launch.sh:73](worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_launch.sh:73), and [depth_canon_launch.sh:76](worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_launch.sh:76), with the policy self-check at line 86. Watcher equality is asserted at [test_depth_canon_autolaunch.py:495](src/tests/test_depth_canon_autolaunch.py:495). Twelve GiB leaves about 1.7 GiB above the measured 10.3 GiB/card peak; defensible as an admission floor, not a reservation. A 120 h cap is sane against ≈43 h uncontended and ≈86 h at half rate.

- **C2 — RESOLVED.** All refusal paths use the sole `gpu_refuse` emitter at [depth_canon_launch.sh:89](worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_launch.sh:89). The strict three-field query, numeric MiB, unknown-UUID rejection, and status checks are at lines 378–423. Shared occupancy is disclosed, not refused; `idle` retains the v10 refusal behavior at lines 424–429. Its per-process wording changed as disclosed. There is no `nvidia-smi` call after the validated gate answers.

- **C3 — NOT RESOLVED.** The writer correctly emits schema 2 and all fields at [depth_canon_launch.sh:658](worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_launch.sh:658), and the six-sha bind remains intact at lines 284–288. FULL’s acceptance predicate does not enforce the new fields. The `null`/`[]` DRY_RUN representation is dead code because DRY_RUN exits at line 483 before evidence generation.

- **C4 — PARTIAL.** Shared occupancy, memory-floor control, three clean polls, token rechecks, rc-7 cap, literal `MB=32 ACC=1 DRY_RUN=0`, and tee draining are preserved at [depth_canon_autolaunch.sh:167](worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_autolaunch.sh:167) and lines 227–273. The numeric-`used_memory` mismatch is the NIT above. Three polls still establish sustained pre-launch free memory, but cannot reserve it; a co-tenant can grow after either the final poll or launcher gate. That is the registered bidirectional-OOM/TOCTOU consequence, not a new wrong-launch path. Infinite waiting with heartbeats is intentional when the floor never clears.

- **C5 — PARTIAL.** The RED logs are load-bearing: watcher 5 failed/45 passed and guard 24 passed/55 failed/1 skipped. Green evidence is 50 watcher tests, 185/0/0 guard, and the reported Planner 2405/3/1 regression. U1/U14/U15/U16/U17/U18, R1b–f and R6b are present. Missing-key/wrong-type evidence tests and a watcher `[N/A]` case are absent. U15 also reads the source working-tree watcher at [depth_canon_guardtests.sh:567](worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_guardtests.sh:567), weakening clone isolation, although the current tracked watcher is clean and matches.

- **C6 — RESOLVED.** Launch logs disclose PID/card/MiB/cwd and per-card free memory; SMOKE evidence records the same snapshot. Plan v14 carries co-tenant FULL/campaign, relaxed constants, OOM, speed, and W&B near-miss disclosures at [plan_depth_canon.md:387](worklog/worklog_yixun/exp_25_depth_canon_claude/plan_depth_canon.md:387). `ALLOW_SHARED_GPU=1` is documented at line 133 and already implemented in the unchanged eval driver; the driver diff is empty.

- **C7 — RESOLVED as implementation, with a mandatory pre-token action.** DRY_RUN skips: NAS write probe (line 343), live GPU gate (372), W&B identity (445), and launch/post-run/evidence generation (483). Independent coverage: non-DRY harness cases exercise successful/failed write probes, GPU branches, evidence writing, and post-run checks; the W&B path has only the reported standalone identity check, not a committed log. I did not repeat it because this was static review. It must be rerun immediately before every actual token as required by [plan_depth_canon.md:187](worklog/worklog_yixun/exp_25_depth_canon_claude/plan_depth_canon.md:187).

- **C8 — RESOLVED with registered limitations.** The 25-step projection measures the currently contended rate, but only over approximately steps 10–25; it is a point-in-time sanity check, not a 40k forecast under bursty neighbors. Co-tenant processes cannot automatically inject into this launcher’s tee, namespace, or evidence; deliberate same-path writes remain outside the admission threat model. `COT_N[AIDX]` is safe, empty-array expansion is safe under Bash 5.2.21, and the quoted four-argument protocol preserves cwd spaces and `?`.

## Coder deviations

1. Accepted: the 27-file regression is the established superset.
2. Accepted: message and 120 h test updates are necessary.
3. Accepted: extra cases strengthen coverage.
4. Accepted: watcher policy self-check is appropriate.
5. Accepted: idle behavior is preserved; only disclosure detail changed.
6. Not accepted: missing FULL validation is the HIGH above.
7. Accepted as accurately disclosed dead code; no DRY_RUN evidence is published.
8. Accepted: manifest remains 48 entries; launcher is bound through its sha.
9. Accepted with the U15 isolation NIT.
10. Accepted: CONFIG ERROR text matches.
11. Accepted: Coder pass was at `3ed2ab2`; Planner pass at current HEAD `6f7cf92` is recorded. Any subsequent fix or records commit requires another final-HEAD pass before the token.

The Planner’s omitted `CUDA_VISIBLE_DEVICES=""` is a disclosed protocol deviation. The result is a four-test superset—2405/3 versus 2401/7—not evidence of a regression failure.

## Read-only verification performed

- `git diff --stat/name-status/numstat 234bc81 6f7cf92`: no non-test `src/`, `train.py`, `defaults.ini`, config, `AR_md.py`, operator, or eval-driver change.
- Manifest regenerated to stdout and compared in-memory: byte-equal, 48 entries, sha256 `631da22f…`.
- `bash -n` on launcher, watcher, guard, and pin generator: clean.
- Code-scope `git diff --check`: clean.
- Tracked worktree: clean; existing untracked logs/lock/artifacts were untouched.
- Current hashes: launcher `4b72da29…`, watcher `8544935a…`, guard `b3d79b32…`, audit gate `452ece21…`.
- Commit sizes: 1a 78+/9−, 1b 87+/17−, launcher 66+/33−, watcher 29+/12−; each code commit is below 200 changed lines.
- Inspected all specified red/green/regression/guard/ABORTED/DRY_RUN logs and the Planner’s untracked verification logs.
- Ran only read-only shell/git/hash/syntax probes and a side-effect-free Bash array expansion check. No test, launcher, watcher, driver, audit/guard suite, GPU command, process probe, package command, or file write was performed.
