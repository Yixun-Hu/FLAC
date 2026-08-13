Reading additional input from stdin...
OpenAI Codex v0.146.0
--------
workdir: /n/fs/gatrdp/codespace/FLAC
model: gpt-5.6-sol
provider: openai
approval: on-request
sandbox: danger-full-access
reasoning effort: xhigh
reasoning summaries: none
session id: 019ffc05-b5d8-7cf3-88ec-eb0879bf3fd7
--------
user
Round-5-fix re-review (read-only; do NOT install anything or modify environments/files; output = review text only). Repo /n/fs/gatrdp/codespace/FLAC. Your round-5 review of the exp_11 chunked-training kit (EXPDIR=worklog/worklog_yixun/exp_11_fa_orbit_claude, your text preserved in EXPDIR/fa_orbit_codex_chunk_kit_review.md) returned REVISE with blockings B1-B7. Scope ruling: B5 (eval-side verify_chain admission of chunk checkpoints) is DEFERRED to the next round — evaluate the rest.

Fixes implemented (cumulative diff + full watchdog at /tmp/claude-374349/-n-fs-gatrdp-codespace-FLAC/66305eb9-88c7-41df-95da-dde30a426dc2/scratchpad/chunk_kit_r2.diff; live files in EXPDIR): B1 registry excluded from both drift gates via :(exclude) pathspec; B2 flock singleton + squeue-failure=UNKNOWN-skip + ARMS dedupe + by-ID lastjob re-check; B3 terminal-only settlement, CANCELLED=>freeze (operator signal), manifest located by settled job id (chunk_end cross-check), unique-chunk_end fallback only with clean state; B4 record-before-DONE, DONE requires disk==record==TARGET, S>TARGET freezes; B6 job-side endpoint attestation appended post-classification (endpoint_ckpt/step/sha256, '<none>' on non-success) + recorder demands and re-hash-verifies it + no predecessor fallback + uuid/manifest-sha uniqueness + preflight full-ancestry validation (link 0 vs INITIAL anchor, link-to-link continuity, strictly increasing steps); B7 watchdog startup anchor validation (VANL was anchored in the registry today, now dirty — itself a live demo of B1). Non-blocking adoptions: state-write failure exits, freeze-clear resets streak only when count>=threshold (deviation: literal always-reset would disable the counter — assess), per-arm CHUNK_<ARM> overrides with VANL default 5000 (your overhead finding; explicit CHUNK= does not override CHUNK_VANL — deviation, assess), guardtests now 123 pass / 5 pre-existing dirty-tree fails.

Verify each in-scope blocking is properly resolved, assess the two deviations, and hunt fresh defects in the changed code (attestation append point vs classification classes; recorder attestation checks; ancestry validator edge cases: single link, empty chain; watchdog lock fd lifetime; settled-jobid manifest fallback). Verdict: APPROVED or REVISE with findings.
codex
I’ll perform a read-only re-review against the preserved round-5 findings, the cumulative diff, and the live kit. I’ll inspect each requested edge case and report only the review verdict/text—no file or environment changes.
exec
/usr/bin/bash -lc 'pwd && git status --short --untracked-files=all && find worklog/worklog_yixun/exp_11_fa_orbit_claude -maxdepth 2 -type f -print | sort && wc -l /tmp/claude-374349/-n-fs-gatrdp-codespace-FLAC/66305eb9-88c7-41df-95da-dde30a426dc2/scratchpad/chunk_kit_r2.diff worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_chunk_kit_review.md' in /n/fs/gatrdp/codespace/FLAC
 succeeded in 31ms:
  5116 worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_chunk_kit_review.md
  6810 total

2026-08-13T16:47:54.746602Z ERROR codex_models_manager::manager: failed to renew cache TTL: missing field `base_instructions` at line 94 column 5
exec
/usr/bin/bash -lc "rg -n '"'^#{1,4} |'"\\bB[1-7]\\b|REVISE|APPROVED|Blocking|blocking' worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_chunk_kit_review.md | head -n 240 && rg -n '"'^diff --git|''^@@'"' /tmp/claude-374349/-n-fs-gatrdp-codespace-FLAC/66305eb9-88c7-41df-95da-dde30a426dc2/scratchpad/chunk_kit_r2.diff" in /n/fs/gatrdp/codespace/FLAC
 exited 127 in 0ms:
/usr/bin/bash: line 1: rg: command not found
/usr/bin/bash: line 1: rg: command not found

2026-08-13T16:48:00.442183Z ERROR codex_models_manager::manager: failed to renew cache TTL: missing field `base_instructions` at line 94 column 5
exec
/usr/bin/bash -lc "grep -nE '"'^#{1,4} |'"\\bB[1-7]\\b|REVISE|APPROVED|Blocking|blocking' worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_chunk_kit_review.md | head -n 240; grep -nE '"'^diff --git|''^@@'"' /tmp/claude-374349/-n-fs-gatrdp-codespace-FLAC/66305eb9-88c7-41df-95da-dde30a426dc2/scratchpad/chunk_kit_r2.diff" in /n/fs/gatrdp/codespace/FLAC
 succeeded in 0ms:
20:REVIEW FOCUS: (a) can any sequence of watchdog + recorder + preflight actions launder an unrecorded/foreign checkpoint into the lineage, or double-submit an arm (race between squeue check and submit; job-name collisions with the five OLD long legs still queued as exp11-<ARM>-train — we cancel them before starting the watchdog, is anything else vulnerable)? (b) recorder correctness: link append under concurrent invocation (store lock?), atomic write, the mine/at indexing, chunk_end vs budget/cadence edges; (c) preflight --chain: any bypass via crafted registry/manifest (note the resume file is always re-hashed); (d) launcher: STOPSTEP interactions with the completion literal, argv-parity section G, wandb fresh-id (r4) under many chunks, TIME_PIN selection; (e) watchdog: sacct settle logic (CANCELLED counts as failure — is that right when a human scancels?), leg_manifest_for uniqueness under repeated same-boundary attempts after a failed leg (two manifests with chunk_end==S -> permanent freeze — is the recovery path documented/sound?), state-file semantics; (f) the 2500-step chunk size: startup overhead ~10-15min/leg at 8 GPUs — flag if you consider the overhead-to-compute ratio for any arm unacceptable vs a 5000-step chunk. Verdict: BLOCKING/NON-BLOCKING findings then APPROVED or REVISE.
694: # Environment pins (round-3 B6) — measured on the reviewed environment:
726:+# --- B2. CHUNK_END: where THIS job stops (round 5) ----------------------------
947:# ============================================================================
948:# fa_orbit_chunk_watchdog.sh — login-side driver for exp_11's CHUNKED legs.
950:# WHY. The partition never backfills a 34-160 h allocation, so the 40k -> 100k
951:# extension legs sat PD indefinitely. A chunked leg asks for hours instead: it
952:# trains to the next 2500-step boundary, saves, and exits. This script is what
953:# turns a sequence of such jobs into one continuous run — it watches for an arm
954:# with no live job, records the chunk that just finished, and submits the next.
956:# WHAT IT MAY DO. Exactly three things: read (squeue/sacct/ls/the registry),
957:# append to its own log and state file, and call the two sanctioned tools —
958:# fa_orbit_record_restart.py and fa_orbit_submit.sh. It NEVER calls sbatch
959:# itself, never deletes or rewrites a checkpoint, a manifest or the registry
960:# (the recorder owns the registry), and never edits the tracked tree.
962:# FAIL-CLOSED. Every step is refused rather than guessed: an ambiguous manifest
963:# or checkpoint, a recorder refusal, or a submitter refusal counts as a failure
964:# for that arm, and MAX_CONSEC_FAIL consecutive failures FREEZE the arm until a
965:# human removes its frozen_<ARM> line from the state file. A frozen arm is
966:# skipped loudly, so the log says why nothing is progressing. The chunk chain
967:# itself is guarded independently of this script: fa_orbit_ckpt_preflight.py
968:# --chain refuses a chunk whose predecessor was never recorded, so a watchdog
969:# bug cannot launder an unrecorded checkpoint into the lineage.
971:# USAGE (KEY=VALUE arguments only; every key whitelisted, no value is ever eval'd)
972:#   bash fa_orbit_chunk_watchdog.sh                       # the pinned defaults
973:#   bash fa_orbit_chunk_watchdog.sh ARMS=C4L,C8 POLL=600
974:#   bash fa_orbit_chunk_watchdog.sh ONESHOT=1 DRYRUN=1    # one pass, submits nothing
976:#   ARMS=C4L,C8,C16,C32,VANL  CHUNK=2500  TARGET=100000  POLL=300
977:#   MAX_CONSEC_FAIL=2  ONESHOT=0  DRYRUN=0
978:#   TEST HOOKS (change no decision, only where the script reads/writes):
979:#   OUTPUT_ROOT=outputs_FLAC  REGISTRY=<expdir>/arm_launch_registry.json
980:#   STATE=<expdir>/.chunk_watchdog_state  LOG=<expdir>/fa_orbit_chunk_watchdog.log
982:# COST. One `squeue` and one `ls` per arm per poll (plus, once an arm is above
983:# 40000, one short python read of the registry). Nothing else runs on the login
984:# node; the training itself is entirely inside Slurm.
985:# ============================================================================
1002:# --- argument parsing: whitelist the KEY, shape-check the VALUE, never eval ---
1040:# --- state: a plain KV file a human can read and edit ------------------------
1041:# keys: fail_<ARM> <n> | frozen_<ARM> <when>|<reason> | lastjob_<ARM> <jid> |
1042:#       done_<ARM> <step>.  Removing a frozen_<ARM> line un-freezes that arm.
2540:     5	# Round-3 review B1: an operator must never hand-assemble --gres/--cpus/--mem/
2766:   173	# --- B. the pins decide the recipe (round-3 B1) -------------------------------
2815:   222	# --- B2. CHUNK_END: where THIS job stops (round 5) ----------------------------
2841:   248	# still matches (content-gate review B2) — data/AR (the split JSONs the
3079:   486	# --- H. Slurm allocation must match the pins (round-3 B1) ---------------------
3106:   513	# --- I. environment identity gates (round-3 B6) -------------------------------
3137:   544	# --- K. exclusive run ownership via flock (round-3 B3 residual) --------------
3156:   563	# --- L. RESTART preflight (round-3 B2) ---------------------------------------
3187:   594	# --- M. wandb: scrub, pin the destination, fix the run id (round-3 B7) --------
3205:   612	# B7 residual: pin the destination account explicitly instead of leaving it implicit
3230:   637	# --- O. atomic manifest, duplicated to the save-dir (round-3 B5) --------------
3269:   676	# B5 residual: a failed environment dump must not be silently hashed into the record
3338:   745	# --- Q. W&B run identity verification (round-3 B7 residual) ------------------
3367:   774	# --- S. durable final record (round-3 B5 residual) ---------------------------
3508:   486	# --- H. Slurm allocation must match the pins (round-3 B1) ---------------------
3529:     2	"""exp_11 RESTART checkpoint preflight (round-3 review B2).
3633:   106	    # Fail-CLOSED (round-3 B2 residual): an absent or empty manifest commit is not
4126:REVIEW FOCUS: (a) can any sequence of watchdog + recorder + preflight actions launder an unrecorded/foreign checkpoint into the lineage, or double-submit an arm (race between squeue check and submit; job-name collisions with the five OLD long legs still queued as exp11-<ARM>-train — we cancel them before starting the watchdog, is anything else vulnerable)? (b) recorder correctness: link append under concurrent invocation (store lock?), atomic write, the mine/at indexing, chunk_end vs budget/cadence edges; (c) preflight --chain: any bypass via crafted registry/manifest (note the resume file is always re-hashed); (d) launcher: STOPSTEP interactions with the completion literal, argv-parity section G, wandb fresh-id (r4) under many chunks, TIME_PIN selection; (e) watchdog: sacct settle logic (CANCELLED counts as failure — is that right when a human scancels?), leg_manifest_for uniqueness under repeated same-boundary attempts after a failed leg (two manifests with chunk_end==S -> permanent freeze — is the recovery path documented/sound?), state-file semantics; (f) the 2500-step chunk size: startup overhead ~10-15min/leg at 8 GPUs — flag if you consider the overhead-to-compute ratio for any arm unacceptable vs a 5000-step chunk. Verdict: BLOCKING/NON-BLOCKING findings then APPROVED or REVISE.
4932:   568	echo "--- I. flock run ownership, both contention directions (round-3 B3 residual) ---"
5053:## BLOCKING
5073:## NON-BLOCKING
5081:## VERDICT
5083:REVISE.
5086:## BLOCKING
5106:## NON-BLOCKING
5114:## VERDICT
5116:REVISE.
1:diff --git a/worklog/worklog_yixun/exp_11_fa_orbit_claude/arm_launch_registry.json b/worklog/worklog_yixun/exp_11_fa_orbit_claude/arm_launch_registry.json
5:@@ -119,7 +119,11 @@
18:diff --git a/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py b/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py
22:@@ -20,6 +20,16 @@ restores the exp_10 depth for the sweep:
39:@@ -120,31 +130,24 @@ def canonical_ckpt_dir(save_dir, arm, repo_root):
85:@@ -186,6 +189,44 @@ def check_extension_binding(manifest_path, registry_path, arm, rung, config_path
130:@@ -196,14 +237,98 @@ def check_extension_binding(manifest_path, registry_path, arm, rung, config_path
236:@@ -221,13 +346,25 @@ def main(argv=None):
263:@@ -277,7 +414,18 @@ def main(argv=None):
283:@@ -287,12 +435,19 @@ def main(argv=None):
303:@@ -308,10 +463,22 @@ def main(argv=None):
327:diff --git a/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_record_restart.py b/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_record_restart.py
331:@@ -26,14 +26,40 @@ else in the manifest was checked at all. Now:
372:@@ -41,6 +67,9 @@ import fa_orbit_producer_manifest as pm            # noqa: E402
382:@@ -70,12 +99,19 @@ def kvs(man, key):
405:@@ -104,13 +140,18 @@ def check_identity(arm, man, initial, pins, repo_root):
429:@@ -148,6 +189,213 @@ def resolve(root, path):
643:@@ -191,6 +439,11 @@ def record(args, arm, pins):
655:diff --git a/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submit.sh b/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submit.sh
659:@@ -9,9 +9,17 @@
677:@@ -28,19 +36,35 @@ PLACEHOLDER="TO-PIN-AFTER-P0"
715:@@ -59,11 +83,17 @@ else
734:@@ -78,8 +108,16 @@ MB="${RUNG%x*}"; NGPU="${RUNG#*x}"
751:@@ -94,9 +132,10 @@ ARGS=(
763:@@ -120,6 +159,7 @@ TMP="$(mktemp "${MANIFEST}.XXXXXX")" || exit 3
771:diff --git a/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch b/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch
775:@@ -23,6 +23,11 @@
787:@@ -95,6 +100,17 @@ PINNED_TIME_LIMIT_RESTART_C8="51:00:00"     # 60k/0.4351 = 38.3 h
805:@@ -137,6 +153,10 @@ else
816:@@ -167,7 +187,12 @@ else
830:@@ -193,6 +218,22 @@ if [ "$SMOKE" != "1" ]; then
853:@@ -206,8 +247,14 @@ EXPREL="${EXPDIR#"$REPO"/}"; EXP07REL="${EXP07#"$REPO"/}"
868:@@ -345,14 +392,14 @@ else
885:@@ -530,8 +577,20 @@ if [ "$MODE" = "RESTART" ]; then
908:@@ -640,6 +699,7 @@ rm -f "$PIPFREEZE_FILE"
916:@@ -707,12 +767,63 @@ if [ "$WANDB_CHECK_RC" -ne 0 ]; then
981:@@ -723,7 +834,9 @@ FINAL_RECORD="$(
991:diff --git a/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh b/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh
995:@@ -430,6 +430,202 @@ printf 'tamper\n' >> "${TMP}/ext_launch_manifest.txt"

2026-08-13T16:48:06.041533Z ERROR codex_models_manager::manager: failed to renew cache TTL: missing field `base_instructions` at line 94 column 5
exec
/usr/bin/bash -lc "sed -n '5035,5125p' worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_chunk_kit_review.md; sed -n '1,180p' /tmp/claude-374349/-n-fs-gatrdp-codespace-FLAC/66305eb9-88c7-41df-95da-dde30a426dc2/scratchpad/chunk_kit_r2.diff" in /n/fs/gatrdp/codespace/FLAC
 succeeded in 0ms:
   501	    import fa_orbit_producer_manifest as pm
   502	    h = hashlib.sha256()
   503	    with open(ckpt, "rb") as fh:
   504	        for chunk in iter(lambda: fh.read(1 << 20), b""):
   505	            h.update(chunk)
   506	    chain_bad, note = pm.verify_chain(json.load(open(reg_path)), arm, step_i, ckpt,
   507	                                      h.hexdigest(), os.path.dirname(os.path.abspath(reg_path)),
   508	                                      main_repo)
   509	    bad += chain_bad
   510	    if note:
   511	        print(note)
   512	if bad:
   513	    sys.exit("ARM LINEAGE GATE: " + "; ".join(bad))
   514	print(f"arm lineage OK: {arm} bound to AUDITED launch job {reg['job']} ({reg['mode']}, uuid "
   515	      f"{reg['launch_uuid'][:8]}, commit {reg['commit'][:12]}, rung {reg['rung']}, "

2026-08-13T16:24:34.745649Z ERROR codex_models_manager::manager: failed to renew cache TTL: missing field `base_instructions` at line 94 column 5
codex
## BLOCKING

1. The watchdog cannot automatically progress beyond the first chunk. The recorder modifies tracked `arm_launch_registry.json`, while the submitter refuses any modified experiment JSON. After recording 42,500, the immediate submission fails; the next poll fails again and freezes the arm. This also risks aborting other arms that start while the registry is dirty. See [fa_orbit_record_restart.py](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_record_restart.py:300) and [fa_orbit_submit.sh](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submit.sh:111). The dynamic chain needs storage compatible with the cleanliness gate, or an explicit publication/commit workflow before further submission.

2. Double submission remains possible. The watchdog has no singleton/per-arm lock, and `squeue` plus submit is a TOCTOU sequence. Worse, an `squeue` error is silently interpreted as an empty queue. Two watchdogs, duplicate arms in `ARMS`, or a concurrent manual submission can therefore queue the same leg twice. The job’s run-directory `flock` occurs only after scheduling and cannot prevent duplicate queued jobs or sequential duplicate execution. Canceling the five old jobs removes only the known name collision. See [fa_orbit_chunk_watchdog.sh](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_chunk_watchdog.sh:171).

3. Scheduler settlement is not actually settled. Empty, lagging, or nonterminal `sacct` output causes `lastjob` to be discarded, after which the same boundary can be resubmitted immediately. A failed attempt that published its launcher manifest followed by a successful retry creates two manifests with the same `chunk_end`; `leg_manifest_for` then refuses forever. Removing `frozen_<ARM>` does not resolve that ambiguity. Retain `lastjob` until an explicit terminal state and select the producer manifest by the settled job ID, not merely `chunk_end`. See [watchdog lines 174–185](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_chunk_watchdog.sh:174) and [manifest lookup](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_chunk_watchdog.sh:152).

   Treating `CANCELLED` as non-success is correct, but automatically replacing a human-cancelled job is not. `CANCELLED` should produce an immediate hold/freeze or require an explicit pause/resume action. Currently the first cancellation is resubmitted, and its failure can even be cleared if an endpoint checkpoint exists.

4. The terminal chunk is never recorded. `S >= TARGET` returns DONE before the `tip < S` reconciliation, so the 97,500→100,000 link is absent. DONE should require an exact, validated, recorded tip at `TARGET`; `S > TARGET` should refuse rather than declare success. See [watchdog line 200](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_chunk_watchdog.sh:200).

5. Chunk checkpoints are incompatible with the existing evaluation lineage gate. The chunk recorder writes `arms.<ARM>.chain`, but `fa_orbit_screen.sbatch` still calls `fa_orbit_producer_manifest.verify_chain`, which only accepts entries under `restarts.<ARM>` with producer manifests. Consequently every chunk-produced checkpoint above 40k is rejected. See [fa_orbit_screen.sbatch](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen.sbatch:489) and [fa_orbit_producer_manifest.py](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_producer_manifest.py:223).

6. The lineage can still attribute a stale/foreign compatible checkpoint to the wrong job. A launcher manifest is published before training, and the recorder associates it with whichever exactly-named endpoint is present in the shared directory. It does not require `sacct=COMPLETED`, the classifier completion record, or a job-generated endpoint attestation. Thus a failed job’s manifest plus a pre-existing same-arm/config checkpoint can become a link; rehashing proves current bytes, not which job produced them.

   Separately, `--chain` validates only `chain[-1]`, not ancestry from the INITIAL anchor. A crafted registry can insert an arbitrary tip hash without valid link-to-link continuity. The recorder also treats a missing predecessor hash/step as `None`, which falls back to the INITIAL anchor. Require full-chain validation and a post-classification producer attestation binding job, UUID, resume hash, endpoint path/step/hash, and successful completion.

7. VANL is not launch-ready. Its live registry row lacks both `final_ckpt_sha256` and `final_step`, while the default watchdog includes VANL. Its first chunk will be submitted and then rejected by `--extension`. The 40k checkpoint must be audited and anchored before startup, and the watchdog should validate all selected anchors before submitting anything. See [arm_launch_registry.json](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/arm_launch_registry.json:110).

## NON-BLOCKING

- Recorder concurrency is otherwise sound for cooperative writers: the exclusive store lock covers registry load through same-directory atomic replace, and the `mine`/`at` indexing correctly handles append and idempotent re-record. Cadence, ordering, and budget checks are correct. Consider enforcing uniqueness of launch UUID and manifest SHA as well as job ID.
- `STOPSTEP` is correctly passed to both `train.py` and the completion classifier. Section G permits `--max-steps` to vary, fresh W&B IDs are collision-resistant, and submitter/job/recorder CHUNK time-pin selection agrees.
- State publication failure currently logs a warning and continues; that should stop the watchdog. Clearing only `frozen_<ARM>` leaves the old failure count, and `done_<ARM>` becomes stale if `TARGET` changes.
- The watchdog guard case deliberately exercises no orchestration path and can pass merely by seeing a live job or empty fixture directory. Add deterministic scheduler fakes covering settlement lag, concurrent watchdogs, failed retry, dirty registry, terminal recording, cancellation, and submission failure.
- At 2,500 steps, 10–15 minutes of startup adds roughly 20–28% of total leg time for VANL and 14–19% for C4L; VANL is unacceptably overhead-heavy and C4L is borderline. A 5,000-step VANL chunk fits its 2.5-hour pin comfortably; C4L would be close to its 3-hour pin and deserves slightly more margin. C16/C32 should remain at 2,500. The current global `CHUNK` option should not permit 5,000 universally: C32 cannot fit 5,000 steps in its 9-hour pin.

## VERDICT

REVISE.
tokens used
245,952
## BLOCKING

1. The watchdog cannot automatically progress beyond the first chunk. The recorder modifies tracked `arm_launch_registry.json`, while the submitter refuses any modified experiment JSON. After recording 42,500, the immediate submission fails; the next poll fails again and freezes the arm. This also risks aborting other arms that start while the registry is dirty. See [fa_orbit_record_restart.py](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_record_restart.py:300) and [fa_orbit_submit.sh](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submit.sh:111). The dynamic chain needs storage compatible with the cleanliness gate, or an explicit publication/commit workflow before further submission.

2. Double submission remains possible. The watchdog has no singleton/per-arm lock, and `squeue` plus submit is a TOCTOU sequence. Worse, an `squeue` error is silently interpreted as an empty queue. Two watchdogs, duplicate arms in `ARMS`, or a concurrent manual submission can therefore queue the same leg twice. The job’s run-directory `flock` occurs only after scheduling and cannot prevent duplicate queued jobs or sequential duplicate execution. Canceling the five old jobs removes only the known name collision. See [fa_orbit_chunk_watchdog.sh](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_chunk_watchdog.sh:171).

3. Scheduler settlement is not actually settled. Empty, lagging, or nonterminal `sacct` output causes `lastjob` to be discarded, after which the same boundary can be resubmitted immediately. A failed attempt that published its launcher manifest followed by a successful retry creates two manifests with the same `chunk_end`; `leg_manifest_for` then refuses forever. Removing `frozen_<ARM>` does not resolve that ambiguity. Retain `lastjob` until an explicit terminal state and select the producer manifest by the settled job ID, not merely `chunk_end`. See [watchdog lines 174–185](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_chunk_watchdog.sh:174) and [manifest lookup](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_chunk_watchdog.sh:152).

   Treating `CANCELLED` as non-success is correct, but automatically replacing a human-cancelled job is not. `CANCELLED` should produce an immediate hold/freeze or require an explicit pause/resume action. Currently the first cancellation is resubmitted, and its failure can even be cleared if an endpoint checkpoint exists.

4. The terminal chunk is never recorded. `S >= TARGET` returns DONE before the `tip < S` reconciliation, so the 97,500→100,000 link is absent. DONE should require an exact, validated, recorded tip at `TARGET`; `S > TARGET` should refuse rather than declare success. See [watchdog line 200](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_chunk_watchdog.sh:200).

5. Chunk checkpoints are incompatible with the existing evaluation lineage gate. The chunk recorder writes `arms.<ARM>.chain`, but `fa_orbit_screen.sbatch` still calls `fa_orbit_producer_manifest.verify_chain`, which only accepts entries under `restarts.<ARM>` with producer manifests. Consequently every chunk-produced checkpoint above 40k is rejected. See [fa_orbit_screen.sbatch](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen.sbatch:489) and [fa_orbit_producer_manifest.py](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_producer_manifest.py:223).

6. The lineage can still attribute a stale/foreign compatible checkpoint to the wrong job. A launcher manifest is published before training, and the recorder associates it with whichever exactly-named endpoint is present in the shared directory. It does not require `sacct=COMPLETED`, the classifier completion record, or a job-generated endpoint attestation. Thus a failed job’s manifest plus a pre-existing same-arm/config checkpoint can become a link; rehashing proves current bytes, not which job produced them.

   Separately, `--chain` validates only `chain[-1]`, not ancestry from the INITIAL anchor. A crafted registry can insert an arbitrary tip hash without valid link-to-link continuity. The recorder also treats a missing predecessor hash/step as `None`, which falls back to the INITIAL anchor. Require full-chain validation and a post-classification producer attestation binding job, UUID, resume hash, endpoint path/step/hash, and successful completion.

7. VANL is not launch-ready. Its live registry row lacks both `final_ckpt_sha256` and `final_step`, while the default watchdog includes VANL. Its first chunk will be submitted and then rejected by `--extension`. The 40k checkpoint must be audited and anchored before startup, and the watchdog should validate all selected anchors before submitting anything. See [arm_launch_registry.json](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/arm_launch_registry.json:110).

## NON-BLOCKING

- Recorder concurrency is otherwise sound for cooperative writers: the exclusive store lock covers registry load through same-directory atomic replace, and the `mine`/`at` indexing correctly handles append and idempotent re-record. Cadence, ordering, and budget checks are correct. Consider enforcing uniqueness of launch UUID and manifest SHA as well as job ID.
- `STOPSTEP` is correctly passed to both `train.py` and the completion classifier. Section G permits `--max-steps` to vary, fresh W&B IDs are collision-resistant, and submitter/job/recorder CHUNK time-pin selection agrees.
- State publication failure currently logs a warning and continues; that should stop the watchdog. Clearing only `frozen_<ARM>` leaves the old failure count, and `done_<ARM>` becomes stale if `TARGET` changes.
- The watchdog guard case deliberately exercises no orchestration path and can pass merely by seeing a live job or empty fixture directory. Add deterministic scheduler fakes covering settlement lag, concurrent watchdogs, failed retry, dirty registry, terminal recording, cancellation, and submission failure.
- At 2,500 steps, 10–15 minutes of startup adds roughly 20–28% of total leg time for VANL and 14–19% for C4L; VANL is unacceptably overhead-heavy and C4L is borderline. A 5,000-step VANL chunk fits its 2.5-hour pin comfortably; C4L would be close to its 3-hour pin and deserves slightly more margin. C16/C32 should remain at 2,500. The current global `CHUNK` option should not permit 5,000 universally: C32 cannot fit 5,000 steps in its 9-hour pin.

## VERDICT

REVISE.
diff --git a/worklog/worklog_yixun/exp_11_fa_orbit_claude/arm_launch_registry.json b/worklog/worklog_yixun/exp_11_fa_orbit_claude/arm_launch_registry.json
index 34af91e..b398126 100644
--- a/worklog/worklog_yixun/exp_11_fa_orbit_claude/arm_launch_registry.json
+++ b/worklog/worklog_yixun/exp_11_fa_orbit_claude/arm_launch_registry.json
@@ -119,7 +119,11 @@
       "vae_sha256": "8d82159eec35210198246f449bec6561fc19b514922f340a17515050daf7f0b9",
       "p0_manifest_sha256": "72607b922177208d56055d604b292d697b643ef3b7ab48261ab2e23a0cc2b53b",
       "save_dir": "outputs_FLAC/exp11_VANL",
-      "training_seed": 42
+      "training_seed": 42,
+      "final_ckpt_sha256": "1095f49330b4e7b9c469d69fdbaab1772586055236964b5e347604e712988507",
+      "final_step": 40000,
+      "final_ckpt_path": "outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=8-step=40000.ckpt",
+      "anchored_at": "2026-08-13T12:30:38-04:00"
     }
   },
   "restarts": {}
diff --git a/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py b/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py
index fc72f83..f03859a 100644
--- a/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py
+++ b/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py
@@ -20,6 +20,16 @@ restores the exp_10 depth for the sweep:
   - optionally, the arm's ORIGINAL launch manifest is re-read and the restart is
     bound to the same rung, commit and budget.
 
+Three resume contracts share those structural checks and differ only in what the
+resume file must BE:
+  * default   — a crash restart of the same launch (same budget, same commit);
+  * --extension — the Q10 40k -> 100k leg: the audited INITIAL launch identity,
+    resuming that launch's audited final checkpoint;
+  * --chain    — a CHUNK (round 5): the same INITIAL identity, resuming the TIP
+    of ``arms.<ARM>.chain``, the per-chunk links fa_orbit_record_restart.py
+    writes. An empty chain refuses, so chunk N+1 cannot start until chunk N is
+    recorded.
+
 Prints the checkpoint sha256 (for the restart manifest) and a lineage summary.
 Exit 0 = admissible; nonzero = refuse to launch.
 """
@@ -120,31 +130,24 @@ def canonical_ckpt_dir(save_dir, arm, repo_root):
     return os.path.realpath(os.path.join(base, f"FLAC_exp11_{arm}", f"exp11_{arm}", "checkpoints"))
 
 
-def check_extension_binding(manifest_path, registry_path, arm, rung, config_path, ckpt_path,
-                            ckpt_sha, expected_step, max_steps, repo_root="."):
-    """The 40k -> 100k EXTENSION contract (re-pin review, required fix 1).
+def _load_registry_row(registry_path, arm):
+    """(row, problems) for the arm's INITIAL registry row."""
+    if not os.path.isfile(registry_path):
+        return None, [f"audited launch registry not found: {registry_path}"]
+    row = json.load(open(registry_path)).get("arms", {}).get(arm)
+    if row is None:
+        return None, [f"{arm} is not in the audited launch registry {registry_path}"]
+    return row, []
 
-    A crash restart continues the SAME launch: same budget, same reviewed commit,
-    so `check_manifest_binding` demands both. An extension breaks both BY DESIGN
-    — it raises the budget from 40000 to 100000 and runs later reviewed code —
-    and demanding equality there is exactly what gave jobs 3662828-30 their third
-    hard-abort path.
 
-    What an extension must still prove is the ORIGINAL LAUNCH IDENTITY, and it
-    proves it against the COMMITTED registry rather than the mutable manifest
-    alone: the INITIAL manifest byte-for-byte as audited, the same job/uuid/
-    launch commit/rung/config/save-dir/training seed, and a resumed checkpoint
-    that IS that launch's audited final checkpoint, sitting in that launch's own
-    canonical run directory. Budget and running commit may move; nothing that
-    identifies the run may.
+def check_initial_identity(man, manifest_path, reg, arm, rung, config_path, max_steps):
+    """Everything an EXTENSION and a CHAIN leg must BOTH prove about the INITIAL launch.
+
+    The two contracts differ only in their RESUME ANCHOR (the extension chains to
+    the audited 40k checkpoint; a chain leg chains to the last recorded chunk),
+    so the launch-identity half lives here and is shared verbatim.
     """
     problems = []
-    if not os.path.isfile(registry_path):
-        return [f"audited launch registry not found: {registry_path}"], {}
-    reg = json.load(open(registry_path)).get("arms", {}).get(arm)
-    if reg is None:
-        return [f"{arm} is not in the audited launch registry {registry_path}"], {}
-    man = parse_manifest(manifest_path)
     kv, jkv = kv_line(man, "arm"), kv_line(man, "job")
 
     got_sha = sha256_file(manifest_path)
@@ -186,6 +189,44 @@ def check_extension_binding(manifest_path, registry_path, arm, rung, config_path
     if sha256_file(config_path) != reg.get("config_sha256"):
         problems.append(f"{config_path} sha256 != the registered config_sha256 "
                         f"{str(reg.get('config_sha256'))[:12]}")
+    return problems
+
+
+def check_canonical_dir(man, arm, ckpt_path, repo_root):
+    """The resume file sits in the REGISTERED launch's own run directory."""
+    save_dir = man.get("save_dir", "")
+    if not save_dir:
+        return ["manifest records no save_dir"]
+    canon = canonical_ckpt_dir(save_dir, arm, repo_root)
+    if os.path.realpath(os.path.dirname(ckpt_path)) != canon:
+        return [f"resume checkpoint {ckpt_path} does not live in the registered "
+                f"launch's canonical run directory {canon}"]
+    return []
+
+
+def check_extension_binding(manifest_path, registry_path, arm, rung, config_path, ckpt_path,
+                            ckpt_sha, expected_step, max_steps, repo_root="."):
+    """The 40k -> 100k EXTENSION contract (re-pin review, required fix 1).
+
+    A crash restart continues the SAME launch: same budget, same reviewed commit,
+    so `check_manifest_binding` demands both. An extension breaks both BY DESIGN
+    — it raises the budget from 40000 to 100000 and runs later reviewed code —
+    and demanding equality there is exactly what gave jobs 3662828-30 their third
+    hard-abort path.
+
+    What an extension must still prove is the ORIGINAL LAUNCH IDENTITY, and it
+    proves it against the COMMITTED registry rather than the mutable manifest
+    alone: the INITIAL manifest byte-for-byte as audited, the same job/uuid/
+    launch commit/rung/config/save-dir/training seed, and a resumed checkpoint
+    that IS that launch's audited final checkpoint, sitting in that launch's own
+    canonical run directory. Budget and running commit may move; nothing that
+    identifies the run may.
+    """
+    reg, problems = _load_registry_row(registry_path, arm)
+    if problems:
+        return problems, {}
+    man = parse_manifest(manifest_path)
+    problems = check_initial_identity(man, manifest_path, reg, arm, rung, config_path, max_steps)
     # the resumed checkpoint IS the audited anchor, in the audited run directory
     anchor, final_step = reg.get("final_ckpt_sha256"), reg.get("final_step")
     if not anchor:
@@ -196,14 +237,98 @@ def check_extension_binding(manifest_path, registry_path, arm, rung, config_path
                         f"{anchor[:12]} — this leg does not continue that run")
     if final_step is not None and int(final_step) != int(expected_step):
         problems.append(f"EXPECTED_STEP {expected_step} != the registered final_step {final_step}")
-    save_dir = man.get("save_dir", "")
-    if not save_dir:
-        problems.append("manifest records no save_dir")
+    problems += check_canonical_dir(man, arm, ckpt_path, repo_root)
+    return problems, man
+
+
+def check_chain_ancestry(reg, chain):
+    """EVERY link, from the audited INITIAL anchor to the tip (round-5 B6).
+
+    Checking only ``chain[-1]`` accepted a crafted registry: append a link whose
+    ``final_ckpt_sha256`` is whatever file you want to run, and the tip check
+    passes while nothing connects it to the audited 40k anchor. The chain is a
+    lineage, so it is verified as one — link 0 must resume the INITIAL anchor,
+    every later link must resume its predecessor's endpoint, and the steps must
+    strictly increase. Any break names the offending link INDEX.
+    """
+    problems = []
+    prev_sha, prev_step = reg.get("final_ckpt_sha256"), reg.get("final_step")
+    prev_label = "the audited INITIAL anchor"
+    for i, link in enumerate(chain):
+        job = link.get("job")
+        r_sha, r_step = link.get("resume_ckpt_sha256"), link.get("resume_step")
+        f_sha, f_step = link.get("final_ckpt_sha256"), link.get("final_step")
+        if not f_sha or f_step is None:
+            problems.append(f"chain link {i} (job {job!r}) carries no final_ckpt_sha256/final_step — "
+                            "it is not evidence of a checkpoint")
+        if prev_sha and r_sha != prev_sha:
+            problems.append(f"chain link {i} (job {job!r}) resume_ckpt_sha256 {str(r_sha)[:12]} != "
+                            f"{prev_label}'s final_ckpt_sha256 {str(prev_sha)[:12]} — the chain is "
+                            f"BROKEN at link {i}: it does not continue what precedes it")
+        if prev_step is not None and str(r_step) != str(prev_step):
+            problems.append(f"chain link {i} (job {job!r}) resume_step {r_step!r} != {prev_label}'s "
+                            f"final_step {prev_step!r} — the chain is BROKEN at link {i}")
+        try:
+            if prev_step is not None and f_step is not None and int(f_step) <= int(prev_step):
+                problems.append(f"chain link {i} (job {job!r}) final_step {f_step!r} does not exceed "
+                                f"{prev_label}'s {prev_step!r} — chunk steps only ever increase")
+        except (TypeError, ValueError):
+            problems.append(f"chain link {i} (job {job!r}) has a non-integer step "
+                            f"(resume {r_step!r}, final {f_step!r})")
+        prev_sha, prev_step, prev_label = f_sha, f_step, f"chain link {i}"
+    return problems
+
+
+def check_chain_binding(manifest_path, registry_path, arm, rung, config_path, ckpt_path,
+                        ckpt_sha, expected_step, max_steps, repo_root="."):

2026-08-13T16:48:11.021073Z ERROR codex_models_manager::manager: failed to renew cache TTL: missing field `base_instructions` at line 94 column 5
exec
/usr/bin/bash -lc "wc -l worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_record_restart.py worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submit.sh worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_chunk_watchdog.sh; nl -ba worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py | sed -n '1,560p'" in /n/fs/gatrdp/codespace/FLAC
 succeeded in 0ms:
   489 worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py
   518 worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_record_restart.py
   181 worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submit.sh
   868 worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch
   496 worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_chunk_watchdog.sh
  2552 total
     1	#!/usr/bin/env python3
     2	"""exp_11 RESTART checkpoint preflight (round-3 review B2).
     3	
     4	exp_10's `bf_resume_launch.sh` proved a restart with ONE CPU-side ``torch.load``
     5	before spending an allocation; round 3 shipped only a path check, which would
     6	have accepted a zero-byte file or a renamed C4 checkpoint under the C16 root
     7	(orbit size does not change the module tree, so it would even have loaded). This
     8	restores the exp_10 depth for the sweep:
     9	
    10	  - the checkpoint's embedded ``global_step`` equals EXPECTED_STEP exactly;
    11	  - its embedded ``model_config`` deep-equals this arm's config file, so the
    12	    orbit, conditioning method and architecture it was trained under are the
    13	    ones this run would continue;
    14	  - full warm optimizer state (non-empty ``state``), ``lr_schedulers``, and EMA
    15	    weights are present — a stripped/weights-only file is the wrong file, since
    16	    exp_11 has no optimizer-reset lineage;
    17	  - the target budget still lies ahead (``global_step < max_steps``), so a
    18	    "restart" cannot terminate immediately on Lightning's ``>=`` stop rule while
    19	    printing the completion literal;
    20	  - optionally, the arm's ORIGINAL launch manifest is re-read and the restart is
    21	    bound to the same rung, commit and budget.
    22	
    23	Three resume contracts share those structural checks and differ only in what the
    24	resume file must BE:
    25	  * default   — a crash restart of the same launch (same budget, same commit);
    26	  * --extension — the Q10 40k -> 100k leg: the audited INITIAL launch identity,
    27	    resuming that launch's audited final checkpoint;
    28	  * --chain    — a CHUNK (round 5): the same INITIAL identity, resuming the TIP
    29	    of ``arms.<ARM>.chain``, the per-chunk links fa_orbit_record_restart.py
    30	    writes. An empty chain refuses, so chunk N+1 cannot start until chunk N is
    31	    recorded.
    32	
    33	Prints the checkpoint sha256 (for the restart manifest) and a lineage summary.
    34	Exit 0 = admissible; nonzero = refuse to launch.
    35	"""
    36	import argparse
    37	import hashlib
    38	import json
    39	import os
    40	import sys
    41	
    42	
    43	def sha256_file(path):
    44	    h = hashlib.sha256()
    45	    with open(path, "rb") as fh:
    46	        for chunk in iter(lambda: fh.read(1 << 20), b""):
    47	            h.update(chunk)
    48	    return h.hexdigest()
    49	
    50	
    51	def load_ckpt_config(path):
    52	    """The model_config embedded in a Lightning checkpoint, on CPU.
    53	
    54	    Shared with the screen driver (fa_orbit_screen.sbatch), which asserts the
    55	    checkpoint's own orbit before spending an evaluation on it: a screen that
    56	    silently evaluated the wrong arm's checkpoint would poison a futility gate."""
    57	    import torch
    58	    ck = torch.load(path, map_location="cpu", weights_only=False)
    59	    if not isinstance(ck, dict):
    60	        raise RuntimeError(f"not a Lightning checkpoint: {path}")
    61	    cfg = ck.get("model_config")
    62	    if not isinstance(cfg, dict):
    63	        raise RuntimeError(f"checkpoint carries no embedded model_config: {path}")
    64	    return cfg, ck.get("global_step")
    65	
    66	
    67	def load_ckpt_state_keys(path):
    68	    """The checkpoint's state_dict KEYS (used to prove EMA weights exist before a
    69	    screen spends a GPU: eval_FLAC silently evaluates online weights when the EMA
    70	    entries are absent)."""
    71	    import torch
    72	    ck = torch.load(path, map_location="cpu", weights_only=False)
    73	    if not isinstance(ck, dict):
    74	        raise RuntimeError(f"not a Lightning checkpoint: {path}")
    75	    return list((ck.get("state_dict") or {}).keys())
    76	
    77	
    78	def parse_manifest(path):
    79	    """The launcher's own manifest format: whitespace-separated `key value...`."""
    80	    out = {}
    81	    with open(path, "r") as fh:
    82	        for line in fh:
    83	            line = line.strip()
    84	            if not line or line.startswith("#"):
    85	                continue
    86	            key, _, rest = line.partition(" ")
    87	            out[key] = rest.strip()
    88	    return out
    89	
    90	
    91	def check_manifest_binding(manifest_path, arm, rung, commit, maxsteps):
    92	    man = parse_manifest(manifest_path)
    93	    problems = []
    94	    fields = man.get("arm", "")
    95	    # `arm <ARM> rung <RUNG> micro <MB> ngpu <N> max_steps <S> ...`
    96	    tokens = ("arm " + fields).split()
    97	    kv = {tokens[i]: tokens[i + 1] for i in range(0, len(tokens) - 1, 2)}
    98	    if kv.get("arm") != arm:
    99	        problems.append(f"manifest arm {kv.get('arm')!r} != {arm!r}")
   100	    if kv.get("rung") != rung:
   101	        problems.append(f"manifest rung {kv.get('rung')!r} != {rung!r} "
   102	                        "(a restart may not change the rung: it would change rank count, "
   103	                        "sampler partitioning and worker seeding mid-lineage)")
   104	    if kv.get("max_steps") != str(maxsteps):
   105	        problems.append(f"manifest max_steps {kv.get('max_steps')!r} != {maxsteps}")
   106	    # Fail-CLOSED (round-3 B2 residual): an absent or empty manifest commit is not
   107	    # "no opinion", it is missing provenance — the restart must not proceed on it.
   108	    man_commit = man.get("commit", "").strip()
   109	    if not man_commit:
   110	        problems.append("launch manifest carries no 'commit' line — cannot bind the restart "
   111	                        "to the lineage that produced this checkpoint")
   112	    elif not commit:
   113	        problems.append("no running commit supplied to compare against the manifest commit")
   114	    elif man_commit != commit:
   115	        problems.append(f"manifest commit {man_commit[:12]} != running commit {commit[:12]}")
   116	    return problems, man
   117	
   118	
   119	def kv_line(man, key):
   120	    """One manifest line's `k v k v ...` pairs (the launcher's `arm ...`/`job ...`)."""
   121	    f = (f"{key} " + man.get(key, "")).split()
   122	    return {f[i]: f[i + 1] for i in range(0, len(f) - 1, 2)}
   123	
   124	
   125	def canonical_ckpt_dir(save_dir, arm, repo_root):
   126	    """<save_dir>/FLAC_exp11_<ARM>/exp11_<ARM>/checkpoints, as the launcher builds it.
   127	
   128	    save_dir is recorded relative to the repo root, so it is resolved against it."""
   129	    base = save_dir if os.path.isabs(save_dir) else os.path.join(repo_root, save_dir)
   130	    return os.path.realpath(os.path.join(base, f"FLAC_exp11_{arm}", f"exp11_{arm}", "checkpoints"))
   131	
   132	
   133	def _load_registry_row(registry_path, arm):
   134	    """(row, problems) for the arm's INITIAL registry row."""
   135	    if not os.path.isfile(registry_path):
   136	        return None, [f"audited launch registry not found: {registry_path}"]
   137	    row = json.load(open(registry_path)).get("arms", {}).get(arm)
   138	    if row is None:
   139	        return None, [f"{arm} is not in the audited launch registry {registry_path}"]
   140	    return row, []
   141	
   142	
   143	def check_initial_identity(man, manifest_path, reg, arm, rung, config_path, max_steps):
   144	    """Everything an EXTENSION and a CHAIN leg must BOTH prove about the INITIAL launch.
   145	
   146	    The two contracts differ only in their RESUME ANCHOR (the extension chains to
   147	    the audited 40k checkpoint; a chain leg chains to the last recorded chunk),
   148	    so the launch-identity half lives here and is shared verbatim.
   149	    """
   150	    problems = []
   151	    kv, jkv = kv_line(man, "arm"), kv_line(man, "job")
   152	
   153	    got_sha = sha256_file(manifest_path)
   154	    if got_sha != reg.get("manifest_sha256"):
   155	        problems.append(f"launch manifest sha256 {got_sha[:12]} != audited "
   156	                        f"{str(reg.get('manifest_sha256'))[:12]} — the manifest changed after it "
   157	                        "was registered")
   158	    for label, got_v, want_v in (("arm", kv.get("arm"), arm),
   159	                                 ("job", jkv.get("job"), reg.get("job")),
   160	                                 ("launch mode", jkv.get("mode"), "INITIAL"),
   161	                                 ("launch_uuid", jkv.get("launch_uuid"), reg.get("launch_uuid")),
   162	                                 ("rung", kv.get("rung"), reg.get("rung")),
   163	                                 ("rung (this run)", rung, reg.get("rung")),
   164	                                 ("config_sha256", man.get("config_sha256"), reg.get("config_sha256")),
   165	                                 ("save_dir", man.get("save_dir"), reg.get("save_dir"))):
   166	        if got_v != want_v:
   167	            problems.append(f"{label} {got_v!r} != registered {want_v!r}")
   168	    man_commit = man.get("commit", "").strip()
   169	    if not man_commit:
   170	        problems.append("launch manifest carries no 'commit' line — cannot bind the extension to "
   171	                        "the lineage that produced this checkpoint")
   172	    elif man_commit != reg.get("commit"):
   173	        problems.append(f"manifest commit {man_commit[:12]} != the registered launch commit "
   174	                        f"{str(reg.get('commit'))[:12]}")
   175	    if int(reg.get("training_seed", -1)) != 42:
   176	        problems.append(f"registered training seed {reg.get('training_seed')!r} != 42")
   177	    # The INITIAL budget is the manifest's and the registry's; the extension's is
   178	    # this run's, and it must strictly cover the resume point without shrinking.
   179	    initial_budget = reg.get("max_steps")
   180	    if kv.get("max_steps") != initial_budget:
   181	        problems.append(f"manifest max_steps {kv.get('max_steps')!r} != registered "
   182	                        f"{initial_budget!r} (the INITIAL budget, which an extension preserves)")
   183	    try:
   184	        if max_steps < int(initial_budget):
   185	            problems.append(f"extension budget {max_steps} does not extend the registered "
   186	                            f"{initial_budget} — an extension may only raise the budget")
   187	    except (TypeError, ValueError):
   188	        problems.append(f"registered max_steps {initial_budget!r} is not an integer")
   189	    if sha256_file(config_path) != reg.get("config_sha256"):
   190	        problems.append(f"{config_path} sha256 != the registered config_sha256 "
   191	                        f"{str(reg.get('config_sha256'))[:12]}")
   192	    return problems
   193	
   194	
   195	def check_canonical_dir(man, arm, ckpt_path, repo_root):
   196	    """The resume file sits in the REGISTERED launch's own run directory."""
   197	    save_dir = man.get("save_dir", "")
   198	    if not save_dir:
   199	        return ["manifest records no save_dir"]
   200	    canon = canonical_ckpt_dir(save_dir, arm, repo_root)
   201	    if os.path.realpath(os.path.dirname(ckpt_path)) != canon:
   202	        return [f"resume checkpoint {ckpt_path} does not live in the registered "
   203	                f"launch's canonical run directory {canon}"]
   204	    return []
   205	
   206	
   207	def check_extension_binding(manifest_path, registry_path, arm, rung, config_path, ckpt_path,
   208	                            ckpt_sha, expected_step, max_steps, repo_root="."):
   209	    """The 40k -> 100k EXTENSION contract (re-pin review, required fix 1).
   210	
   211	    A crash restart continues the SAME launch: same budget, same reviewed commit,
   212	    so `check_manifest_binding` demands both. An extension breaks both BY DESIGN
   213	    — it raises the budget from 40000 to 100000 and runs later reviewed code —
   214	    and demanding equality there is exactly what gave jobs 3662828-30 their third
   215	    hard-abort path.
   216	
   217	    What an extension must still prove is the ORIGINAL LAUNCH IDENTITY, and it
   218	    proves it against the COMMITTED registry rather than the mutable manifest
   219	    alone: the INITIAL manifest byte-for-byte as audited, the same job/uuid/
   220	    launch commit/rung/config/save-dir/training seed, and a resumed checkpoint
   221	    that IS that launch's audited final checkpoint, sitting in that launch's own
   222	    canonical run directory. Budget and running commit may move; nothing that
   223	    identifies the run may.
   224	    """
   225	    reg, problems = _load_registry_row(registry_path, arm)
   226	    if problems:
   227	        return problems, {}
   228	    man = parse_manifest(manifest_path)
   229	    problems = check_initial_identity(man, manifest_path, reg, arm, rung, config_path, max_steps)
   230	    # the resumed checkpoint IS the audited anchor, in the audited run directory
   231	    anchor, final_step = reg.get("final_ckpt_sha256"), reg.get("final_step")
   232	    if not anchor:
   233	        problems.append(f"{arm} has no audited final_ckpt_sha256 in the registry — the extension "
   234	                        "has nothing to chain to (audit the arm's final checkpoint first)")
   235	    elif ckpt_sha != anchor:
   236	        problems.append(f"resume checkpoint sha256 {ckpt_sha[:12]} != the audited final checkpoint "
   237	                        f"{anchor[:12]} — this leg does not continue that run")
   238	    if final_step is not None and int(final_step) != int(expected_step):
   239	        problems.append(f"EXPECTED_STEP {expected_step} != the registered final_step {final_step}")
   240	    problems += check_canonical_dir(man, arm, ckpt_path, repo_root)
   241	    return problems, man
   242	
   243	
   244	def check_chain_ancestry(reg, chain):
   245	    """EVERY link, from the audited INITIAL anchor to the tip (round-5 B6).
   246	
   247	    Checking only ``chain[-1]`` accepted a crafted registry: append a link whose
   248	    ``final_ckpt_sha256`` is whatever file you want to run, and the tip check
   249	    passes while nothing connects it to the audited 40k anchor. The chain is a
   250	    lineage, so it is verified as one — link 0 must resume the INITIAL anchor,
   251	    every later link must resume its predecessor's endpoint, and the steps must
   252	    strictly increase. Any break names the offending link INDEX.
   253	    """
   254	    problems = []
   255	    prev_sha, prev_step = reg.get("final_ckpt_sha256"), reg.get("final_step")
   256	    prev_label = "the audited INITIAL anchor"
   257	    for i, link in enumerate(chain):
   258	        job = link.get("job")
   259	        r_sha, r_step = link.get("resume_ckpt_sha256"), link.get("resume_step")
   260	        f_sha, f_step = link.get("final_ckpt_sha256"), link.get("final_step")
   261	        if not f_sha or f_step is None:
   262	            problems.append(f"chain link {i} (job {job!r}) carries no final_ckpt_sha256/final_step — "
   263	                            "it is not evidence of a checkpoint")
   264	        if prev_sha and r_sha != prev_sha:
   265	            problems.append(f"chain link {i} (job {job!r}) resume_ckpt_sha256 {str(r_sha)[:12]} != "
   266	                            f"{prev_label}'s final_ckpt_sha256 {str(prev_sha)[:12]} — the chain is "
   267	                            f"BROKEN at link {i}: it does not continue what precedes it")
   268	        if prev_step is not None and str(r_step) != str(prev_step):
   269	            problems.append(f"chain link {i} (job {job!r}) resume_step {r_step!r} != {prev_label}'s "
   270	                            f"final_step {prev_step!r} — the chain is BROKEN at link {i}")
   271	        try:
   272	            if prev_step is not None and f_step is not None and int(f_step) <= int(prev_step):
   273	                problems.append(f"chain link {i} (job {job!r}) final_step {f_step!r} does not exceed "
   274	                                f"{prev_label}'s {prev_step!r} — chunk steps only ever increase")
   275	        except (TypeError, ValueError):
   276	            problems.append(f"chain link {i} (job {job!r}) has a non-integer step "
   277	                            f"(resume {r_step!r}, final {f_step!r})")
   278	        prev_sha, prev_step, prev_label = f_sha, f_step, f"chain link {i}"
   279	    return problems
   280	
   281	
   282	def check_chain_binding(manifest_path, registry_path, arm, rung, config_path, ckpt_path,
   283	                        ckpt_sha, expected_step, max_steps, repo_root="."):
   284	    """The CHUNKED-LEG contract: resume the TIP of this arm's recorded chain.
   285	
   286	    A chunk leg is an extension leg that stops early, so it must prove exactly
   287	    the same INITIAL launch identity — but its resume point is no longer the
   288	    audited 40k anchor: it is the endpoint of the previous chunk, recorded as the
   289	    last link of ``arms.<ARM>.chain`` by fa_orbit_record_restart.py.
   290	
   291	    Fail-closed by construction: an EMPTY or ABSENT chain refuses, so chunk N+1
   292	    cannot run until chunk N has been recorded, and every link's final_step /
   293	    final_ckpt_sha256 was written by the recorder from the file on disk. The
   294	    resume file is re-hashed by the caller (``ckpt_sha``); a manifest's claimed
   295	    hash is never trusted here or anywhere else in this chain.
   296	
   297	    Round-5 review B6: the WHOLE ancestry is validated (check_chain_ancestry),
   298	    not merely ``chain[-1]`` — otherwise a crafted registry could append a tip
   299	    with an arbitrary hash and no link back to the audited 40k anchor.
   300	    """
   301	    reg, problems = _load_registry_row(registry_path, arm)
   302	    if problems:
   303	        return problems, {}
   304	    man = parse_manifest(manifest_path)
   305	    problems = check_initial_identity(man, manifest_path, reg, arm, rung, config_path, max_steps)
   306	    if not reg.get("final_ckpt_sha256"):
   307	        problems.append(f"{arm} has no audited final_ckpt_sha256 in the registry — a chain of "
   308	                        "chunks must still descend from the audited INITIAL run")
   309	    chain = reg.get("chain") or []
   310	    if not chain:
   311	        problems.append(f"{arm} has no recorded chain link in {registry_path}: a chunk may only "
   312	                        "resume the tip of a RECORDED chain, so chunk N+1 is inadmissible until "
   313	                        "chunk N is recorded — run fa_orbit_record_restart.py on the previous "
   314	                        "chunk's launcher manifest first")
   315	    else:
   316	        # B6: the FULL ancestry, not just the tip — a tip alone can be crafted.
   317	        problems += check_chain_ancestry(reg, chain)
   318	        tip = chain[-1]
   319	        tip_step, tip_sha = tip.get("final_step"), tip.get("final_ckpt_sha256")
   320	        if str(tip_step) != str(expected_step):
   321	            problems.append(f"EXPECTED_STEP {expected_step} != the last recorded chain link's "
   322	                            f"final_step {tip_step!r} (link job {tip.get('job')!r}) — a chunk "
   323	                            "resumes the TIP of the chain, never an earlier link")
   324	        if not tip_sha:
   325	            problems.append(f"the last recorded chain link (job {tip.get('job')!r}) carries no "
   326	                            "final_ckpt_sha256 — it is not evidence of a checkpoint")
   327	        elif ckpt_sha != tip_sha:
   328	            problems.append(f"resume checkpoint sha256 {ckpt_sha[:12]} != the last recorded chain "
   329	                            f"link's final_ckpt_sha256 {str(tip_sha)[:12]} — this file is not the "
   330	                            "checkpoint that chunk produced")
   331	    problems += check_canonical_dir(man, arm, ckpt_path, repo_root)
   332	    return problems, man
   333	
   334	
   335	def main(argv=None):
   336	    ap = argparse.ArgumentParser(description="exp_11 restart checkpoint preflight")
   337	    ap.add_argument("--ckpt", required=True)
   338	    ap.add_argument("--expected-step", type=int, required=True)
   339	    ap.add_argument("--config", required=True, help="this arm's model config json")
   340	    ap.add_argument("--max-steps", type=int, required=True)
   341	    ap.add_argument("--arm", required=True)
   342	    ap.add_argument("--rung", required=True)
   343	    ap.add_argument("--commit", default="")
   344	    ap.add_argument("--launch-manifest", default="",
   345	                    help="the arm's original launch manifest (binds rung/commit/budget)")
   346	    ap.add_argument("--extension", action="store_true",
   347	                    help="this restart is the Q10 40k->100k EXTENSION: bind it to the audited "
   348	                         "INITIAL launch identity instead of requiring an equal budget/commit")
   349	    ap.add_argument("--chain", action="store_true",
   350	                    help="this restart is a CHUNK continuing an already-recorded chain: same "
   351	                         "INITIAL identity as --extension, but the resume anchor is the LAST "
   352	                         "recorded arms.<ARM>.chain link instead of the audited 40k checkpoint")
   353	    ap.add_argument("--chunk-end", type=int, default=None,
   354	                    help="this leg's stop step (a chunk boundary): EXPECTED_STEP < N <= max-steps "
   355	                         "and a multiple of 2500")
   356	    ap.add_argument("--launch-registry", default="",
   357	                    help="the committed arm launch registry (required with --extension/--chain)")
   358	    ap.add_argument("--repo-root", default=".",
   359	                    help="root the registry's relative save_dir is resolved against")
   360	    args = ap.parse_args(argv)
   361	    if args.extension and args.chain:
   362	        ap.error("--extension and --chain are mutually exclusive: a leg either resumes the audited "
   363	                 "40k anchor or the tip of the recorded chunk chain, never both")
   364	    if args.extension and not args.launch_registry:
   365	        ap.error("--extension requires --launch-registry (the audited INITIAL launch row)")
   366	    if args.chain and not args.launch_registry:
   367	        ap.error("--chain requires --launch-registry (the recorded chunk chain lives in it)")
   368	
   369	    if not os.path.isfile(args.ckpt):
   370	        print(f"PREFLIGHT: checkpoint not found: {args.ckpt}")
   371	        return 2
   372	
   373	    import torch  # deferred: keeps --help and unit imports cheap
   374	
   375	    try:
   376	        ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
   377	    except Exception as exc:                      # truncated, empty or not a torch file
   378	        print(f"PREFLIGHT: {args.ckpt} is not loadable as a checkpoint: "
   379	              f"{type(exc).__name__}: {exc}")
   380	        return 2
   381	    if not isinstance(ck, dict):
   382	        print(f"PREFLIGHT: not a Lightning checkpoint: {args.ckpt}")
   383	        return 2
   384	
   385	    problems = []
   386	    gs = ck.get("global_step")
   387	    if gs != args.expected_step:
   388	        problems.append(f"global_step {gs} != EXPECTED_STEP {args.expected_step}")
   389	    if isinstance(gs, int) and gs >= args.max_steps:
   390	        problems.append(f"global_step {gs} >= max_steps {args.max_steps}: Lightning would stop "
   391	                        "immediately and still print the completion literal")
   392	
   393	    mc = ck.get("model_config")
   394	    if not isinstance(mc, dict):
   395	        problems.append("checkpoint carries no embedded 'model_config' dict")
   396	    else:
   397	        want = json.load(open(args.config))
   398	        if mc != want:
   399	            tr = mc.get("training", {}) if isinstance(mc.get("training"), dict) else {}
   400	            problems.append(
   401	                f"embedded model_config != {args.config} (parsed-object mismatch; embedded "
   402	                f"cond_method={tr.get('cond_method')!r} angles={tr.get('frame_avg_angles')!r})")
   403	
   404	    opts = ck.get("optimizer_states") or []
   405	    if len(opts) != 1:
   406	        problems.append(f"expected exactly 1 optimizer entry, found {len(opts)}")
   407	    elif not opts[0].get("state"):
   408	        problems.append("optimizer state is CLEARED (stripped checkpoint); exp_11 restarts are "
   409	                        "WARM continuations and have no optimizer-reset lineage")
   410	    if not ck.get("lr_schedulers"):
   411	        problems.append("no 'lr_schedulers' -> PL 2.1 KeyErrors on resume")
   412	    sd = ck.get("state_dict") or {}
   413	    n_ema = sum(1 for k in sd if k.startswith("diffusion_ema."))
   414	    if not n_ema:
   415	        problems.append("no EMA weights in state_dict")
   416	
   417	    # The per-job stop step is checked against the BUDGET (--max-steps), which the
   418	    # launcher still pins at 100000: a chunk narrows where this leg stops, never
   419	    # what the campaign is allowed to reach.
   420	    if args.chunk_end is not None:
   421	        if args.chunk_end % 2500 != 0:
   422	            problems.append(f"--chunk-end {args.chunk_end} is not a multiple of 2500 (the pinned "
   423	                            "checkpoint cadence: a chunk must end ON a checkpoint)")
   424	        if not args.expected_step < args.chunk_end <= args.max_steps:
   425	            problems.append(f"--chunk-end {args.chunk_end} must satisfy EXPECTED_STEP "
   426	                            f"{args.expected_step} < chunk_end <= max_steps {args.max_steps}")
   427	
   428	    digest = sha256_file(args.ckpt)     # needed by the extension/chain anchor checks
   429	    man = {}
   430	    if args.launch_manifest:
   431	        if not os.path.isfile(args.launch_manifest):
   432	            problems.append(f"launch manifest not found: {args.launch_manifest}")
   433	        elif args.extension:
   434	            more, man = check_extension_binding(
   435	                args.launch_manifest, args.launch_registry, args.arm, args.rung, args.config,
   436	                args.ckpt, digest, args.expected_step, args.max_steps, args.repo_root)
   437	            problems += more
   438	        elif args.chain:
   439	            more, man = check_chain_binding(
   440	                args.launch_manifest, args.launch_registry, args.arm, args.rung, args.config,
   441	                args.ckpt, digest, args.expected_step, args.max_steps, args.repo_root)
   442	            problems += more
   443	        else:
   444	            more, man = check_manifest_binding(args.launch_manifest, args.arm, args.rung,
   445	                                               args.commit, args.max_steps)
   446	            problems += more
   447	    elif args.extension:
   448	        problems.append("--extension requires --launch-manifest (the audited INITIAL manifest)")
   449	    elif args.chain:
   450	        problems.append("--chain requires --launch-manifest (the audited INITIAL manifest)")
   451	
   452	    if problems:
   453	        print("PREFLIGHT FAILED:")
   454	        for p in problems:
   455	            print(f"  !! {p}")
   456	        return 2
   457	
   458	    tr = mc.get("training", {})
   459	    print(f"restart lineage OK: {args.ckpt}")
   460	    print(f"  global_step={gs} epoch={ck.get('epoch')} cond_method={tr.get('cond_method')!r} "
   461	          f"angles={tr.get('frame_avg_angles')}")
   462	    print(f"  optimizer_state=FULL ({len(opts[0]['state'])} entries) "
   463	          f"lr={opts[0]['param_groups'][0].get('lr')} "
   464	          f"sched_last_epoch={ck['lr_schedulers'][0].get('last_epoch')} ema_entries={n_ema}")
   465	    if man and args.extension:
   466	        print(f"extension lineage OK: {args.arm} {args.expected_step} -> "
   467	              f"{args.chunk_end if args.chunk_end is not None else args.max_steps} continues "
   468	              f"the audited launch job {kv_line(man, 'job').get('job')} "
   469	              f"(launch commit {man.get('commit', '')[:12]}, running commit {args.commit[:12] or '<none>'})")
   470	        print(f"  bound to the audited launch manifest: {args.launch_manifest}")
   471	        if args.chunk_end is not None:
   472	            print(f"  chunk leg: stops at {args.chunk_end} of the {args.max_steps} budget")
   473	    elif man and args.chain:
   474	        print(f"chain lineage OK: {args.arm} {args.expected_step} -> "
   475	              f"{args.chunk_end if args.chunk_end is not None else args.max_steps} resumes the tip "
   476	              f"of the recorded chunk chain, under the audited launch job "
   477	              f"{kv_line(man, 'job').get('job')} "
   478	              f"(launch commit {man.get('commit', '')[:12]}, running commit {args.commit[:12] or '<none>'})")
   479	        print(f"  bound to the audited launch manifest: {args.launch_manifest}")
   480	        if args.chunk_end is not None:
   481	            print(f"  chunk leg: stops at {args.chunk_end} of the {args.max_steps} budget")
   482	    elif man:
   483	        print(f"  bound to launch manifest: {args.launch_manifest}")
   484	    print(f"CKPT_SHA256 {digest}")
   485	    return 0
   486	
   487	
   488	if __name__ == "__main__":
   489	    sys.exit(main())

2026-08-13T16:48:17.328806Z ERROR codex_models_manager::manager: failed to renew cache TTL: missing field `base_instructions` at line 94 column 5
codex
The main fixes are present, including full-chain traversal and post-run endpoint fields. I’m now tracing whether their fail-closed behavior holds under malformed/empty ancestry and scheduler/manifest ambiguity, because those are the most likely places for a nominal fix to retain a bypass.
exec
/usr/bin/bash -lc "nl -ba worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_record_restart.py | sed -n '1,560p'" in /n/fs/gatrdp/codespace/FLAC
 succeeded in 0ms:
     1	#!/usr/bin/env python3
     2	"""Record a RESTART leg in arm_launch_registry.json from its PUBLISHED manifest.
     3	
     4	    python3 fa_orbit_record_restart.py C4L outputs_FLAC/exp11_C4L/<manifest>.txt
     5	    python3 fa_orbit_record_restart.py C4L <manifest> --extend   # later, as the leg saves more
     6	
     7	A restart is only admissible if it provably continues the audited INITIAL run, so
     8	this refuses unless the resume checkpoint ON DISK -- always re-hashed, never
     9	trusted from the manifest -- equals that arm's recorded final_ckpt_sha256.
    10	
    11	Re-pin review, required fix 3. The previous version was fail-OPEN: it re-hashed
    12	only `if os.path.isfile(resume_path)`, so a manifest naming a file that could not
    13	be resolved was recorded on the strength of its own claimed hash, and nothing
    14	else in the manifest was checked at all. Now:
    15	
    16	  * the canonical resume file MUST exist, sit in the audited launch's own
    17	    checkpoint directory, and is ALWAYS re-hashed;
    18	  * every identity field is validated against the INITIAL registry row (arm, job,
    19	    uuid, commit, rung, config sha, VAE and P0 manifest shas, save-dir, seed) and
    20	    against the Q10 pins read out of the launcher itself (budget 100000, resume
    21	    step = the audited final step, and the arm's RESTART wall pin), so recorder
    22	    and launcher cannot disagree;
    23	  * publication is atomic (tmp + rename) under the store lock;
    24	  * duplicates are refused -- one leg, one row.
    25	
    26	It also publishes the leg's PRODUCER MANIFEST (fix 2): every checkpoint this leg
    27	produced, re-hashed from disk, into an append-only per-leg file the screen
    28	verifies each >40k checkpoint against. Re-run with --extend as the leg saves more.
    29	
    30	ROUND 5 — CHUNKED legs. The cluster never backfills a 34-160 h allocation, so a
    31	leg now trains to the next 2500-step boundary and exits (`chunk_end` in its
    32	manifest). Such a leg is recorded as a CHAIN LINK in `arms.<ARM>.chain`:
    33	
    34	    INITIAL anchor (40000) <- link(40000 -> 42500) <- link(42500 -> 45000) <- ...
    35	
    36	Each link is admissible only if it resumed the TIP of the chain (or the audited
    37	INITIAL anchor, for the first link) with the file that is on disk NOW, re-hashed.
    38	The endpoint checkpoint the leg produced is likewise located in the audited
    39	canonical directory and hashed by this recorder, which is what makes it the next
    40	link's anchor: fa_orbit_ckpt_preflight.py --chain refuses a chunk whose resume
    41	file is not the last link's `final_ckpt_sha256`, so chunk N+1 cannot start until
    42	chunk N is recorded here. Appending a link never touches the INITIAL fields, is
    43	atomic (tmp+rename under the store lock), and is idempotent per job.
    44	
    45	ROUND-5 REVIEW B6 — ATTRIBUTION. Location + re-hashing prove which bytes are on
    46	disk, not WHICH JOB WROTE THEM: a failed leg's (pre-published) manifest plus a
    47	pre-existing checkpoint of the right name was enough to mint a link. So a chunk
    48	is now recorded only on the producing job's own post-classification attestation
    49	(`endpoint_ckpt … endpoint_step … endpoint_sha256 …`, appended by
    50	fa_orbit_train.sbatch to its own manifest), checked against this recorder's
    51	independent re-hash. And a chunk that HAS a predecessor is never re-parented
    52	onto the INITIAL anchor when that predecessor is incomplete — it is refused.
    53	"""
    54	import argparse
    55	import fcntl
    56	import glob
    57	import hashlib
    58	import json
    59	import os
    60	import re
    61	import sys
    62	from datetime import datetime, timezone
    63	
    64	HERE = os.path.dirname(os.path.abspath(__file__))
    65	sys.path.insert(0, HERE)
    66	import fa_orbit_producer_manifest as pm            # noqa: E402
    67	from fa_orbit_ckpt_preflight import canonical_ckpt_dir    # noqa: E402
    68	
    69	PIN_RE = re.compile(r'^(PINNED_[A-Z0-9_]+)=(?:"([^"]*)"|(\S+))')
    70	# The pinned checkpoint cadence: a chunk may only end ON a saved checkpoint,
    71	# otherwise the next chunk has nothing to resume.
    72	CHUNK_STEP = 2500
    73	
    74	
    75	def read_pins(launcher):
    76	    """The launcher's own PINNED_* values, so the recorder cannot drift from them."""
    77	    pins = {}
    78	    with open(launcher) as fh:
    79	        for line in fh:
    80	            m = PIN_RE.match(line)
    81	            if m:
    82	                pins[m.group(1)] = m.group(2) if m.group(2) is not None else m.group(3)
    83	    return pins
    84	
    85	
    86	def parse_manifest(path):
    87	    raw = open(path, "rb").read()
    88	    man = {}
    89	    for line in raw.decode().splitlines():
    90	        line = line.strip()
    91	        if line and not line.startswith("#"):
    92	            k, _, rest = line.partition(" ")
    93	            man[k] = rest.strip()
    94	    return raw, man
    95	
    96	
    97	def kvs(man, key):
    98	    f = (f"{key} " + man.get(key, "")).split()
    99	    return {f[i]: f[i + 1] for i in range(0, len(f) - 1, 2)}
   100	
   101	
   102	def check_identity(arm, man, initial, pins, repo_root,
   103	                   want_anchor=None, want_step=None, want_time=None):
   104	    """Every field of the RESTART manifest, against the audited INITIAL row + Q10 pins.
   105	
   106	    A CHUNK leg proves the same identity but resumes the tip of the recorded
   107	    chain rather than the audited 40k anchor, and is walled by the arm's CHUNK
   108	    pin rather than its RESTART pin, so the caller may override those three
   109	    expectations. Everything else is identical for both leg kinds."""
   110	    jk, ak, rk = kvs(man, "job"), kvs(man, "arm"), kvs(man, "resume_ckpt")
   111	    tk = kvs(man, "time_limit")
   112	    problems = []
   113	    anchor = want_anchor if want_anchor is not None else initial.get("final_ckpt_sha256")
   114	    final_step = want_step if want_step is not None else initial.get("final_step")
   115	    if not anchor:
   116	        problems.append(f"{arm} has no audited final_ckpt_sha256 to chain from — audit the "
   117	                        "INITIAL run's final checkpoint before recording a leg")
   118	    if jk.get("mode") != "RESTART":
   119	        problems.append(f"manifest mode is {jk.get('mode')!r}, not RESTART")
   120	    for field, got in (("job", jk.get("job")), ("launch_uuid", jk.get("launch_uuid")),
   121	                       ("commit", man.get("commit"))):
   122	        if not got:
   123	            problems.append(f"manifest records no {field} — a leg with no identity is not a record")
   124	    if jk.get("job") and initial.get("job") == jk.get("job"):
   125	        problems.append(f"manifest job {jk.get('job')} IS the INITIAL job — that is the launch "
   126	                        "already registered, not a restart leg")
   127	    for label, got, want in (("arm", ak.get("arm"), arm),
   128	                             ("rung", ak.get("rung"), initial.get("rung")),
   129	                             ("micro", ak.get("micro"), pins.get("PINNED_MB")),
   130	                             ("ngpu", ak.get("ngpu"), pins.get("PINNED_NGPU")),
   131	                             ("config_sha256", man.get("config_sha256"), initial.get("config_sha256")),
   132	                             ("vae_sha256", man.get("vae_sha256"), initial.get("vae_sha256")),
   133	                             ("p0_manifest_sha256", man.get("p0_manifest_sha256"),
   134	                              initial.get("p0_manifest_sha256")),
   135	                             ("save_dir", man.get("save_dir"), initial.get("save_dir"))):
   136	        if got != want:
   137	            problems.append(f"manifest {label} {got!r} != the audited INITIAL run's {want!r}")
   138	    if ak.get("rung") != pins.get("PINNED_RUNG"):
   139	        problems.append(f"manifest rung {ak.get('rung')!r} != the pinned {pins.get('PINNED_RUNG')!r}")
   140	    if ak.get("max_steps") != pins.get("PINNED_MAXSTEPS"):
   141	        problems.append(f"manifest max_steps {ak.get('max_steps')!r} != the Q10 budget pin "
   142	                        f"{pins.get('PINNED_MAXSTEPS')!r}")
   143	    step_label = "audited final step" if want_step is None else "recorded chain tip's final step"
   144	    if final_step is not None and str(rk.get("expected_step")) != str(final_step):
   145	        problems.append(f"manifest expected_step {rk.get('expected_step')!r} != the {step_label} "
   146	                        f"{final_step!r} — a leg resumes where the run it continues ended")
   147	    pin_label = "RESTART wall pin"
   148	    if want_time is None:
   149	        want_time = pins.get(f"PINNED_TIME_LIMIT_RESTART_{arm}")
   150	    else:
   151	        pin_label = "CHUNK wall pin"
   152	    if tk.get("time_limit") != want_time:
   153	        problems.append(f"manifest time_limit {tk.get('time_limit')!r} != the arm's {pin_label} "
   154	                        f"{want_time!r}")
   155	    if int(initial.get("training_seed", -1)) != 42:
   156	        problems.append(f"registered training seed {initial.get('training_seed')!r} != 42")
   157	    # the config the leg names must still hash to the audited value
   158	    cfg_path = man.get("model_config", "")
   159	    if not cfg_path or not os.path.isfile(cfg_path):
   160	        problems.append(f"manifest model_config {cfg_path!r} does not exist")
   161	    elif hashlib.sha256(open(cfg_path, "rb").read()).hexdigest() != initial.get("config_sha256"):
   162	        problems.append(f"{cfg_path} no longer hashes to the audited config_sha256")
   163	    # --- the resume file itself: MUST exist, MUST be canonical, ALWAYS re-hashed --
   164	    resume_path = (man.get("resume_ckpt", "").split() or [""])[0]
   165	    resume_real = ""
   166	    if not resume_path or resume_path == "<none>":
   167	        problems.append("manifest records no resume_ckpt — a RESTART that resumed nothing is not "
   168	                        "a continuation of the audited run")
   169	    elif not os.path.isfile(resolve(repo_root, resume_path)):
   170	        problems.append(f"the resume checkpoint {resume_path} does not exist — the recorder does "
   171	                        "NOT accept the manifest's claimed hash in its place")
   172	    else:
   173	        resume_real = resolve(repo_root, resume_path)
   174	        canon = canonical_ckpt_dir(initial.get("save_dir", ""), arm, repo_root)
   175	        if os.path.realpath(os.path.dirname(resume_real)) != canon:
   176	            problems.append(f"the resume checkpoint is not in the audited launch's canonical "
   177	                            f"directory {canon}")
   178	        got = pm.sha256_file(resume_real)
   179	        if anchor and got != anchor:
   180	            problems.append(f"the resume file on disk hashes {got[:12]}, not the audited "
   181	                            f"{anchor[:12]} — this leg does not continue that run")
   182	        if rk.get("resume_ckpt_sha256") != got:
   183	            problems.append(f"manifest resume_ckpt_sha256 {str(rk.get('resume_ckpt_sha256'))[:12]} "
   184	                            f"!= the file's actual {got[:12]}")
   185	    return problems, resume_real
   186	
   187	
   188	def resolve(root, path):
   189	    return path if os.path.isabs(path) else os.path.join(root, path)
   190	
   191	
   192	def chunk_end_of(man):
   193	    """The leg's per-job stop step as written by the launcher, or None.
   194	
   195	    A manifest with no `chunk_end` line (or the literal `<none>`) is a
   196	    whole-budget RESTART leg and takes the original recording path untouched."""
   197	    v = (man.get("chunk_end", "").split() or [""])[0]
   198	    return None if not v or v == "<none>" else v
   199	
   200	
   201	def utc_now():
   202	    return datetime.now(timezone.utc).isoformat(timespec="seconds")
   203	
   204	
   205	def check_endpoint_attestation(man, chunk_end, endpoint, endpoint_sha, repo_root):
   206	    """The producing job's own statement of what it wrote (round-5 review B6).
   207	
   208	    The launcher publishes its manifest BEFORE training, so the manifest alone
   209	    proves intent, never authorship: a FAILED leg's manifest plus a pre-existing
   210	    checkpoint of the right name would otherwise become a chain link, and
   211	    re-hashing proves only which bytes are on disk now. After classification the
   212	    job appends to its own manifest
   213	
   214	        endpoint_ckpt <path> endpoint_step <n> endpoint_sha256 <sha>
   215	
   216	    (or `endpoint_ckpt <none> ...` if it did not reach the success class), and
   217	    this function makes that attestation MANDATORY and fail-closed:
   218	
   219	      * no attestation at all           -> refuse (name the launcher append);
   220	      * `<none>`                        -> refuse (the leg says it produced none);
   221	      * endpoint_step != chunk_end      -> refuse (it attests another boundary);
   222	      * attested sha != our own re-hash -> refuse (it is not that file);
   223	      * attested path != the file we found in the canonical dir -> refuse.
   224	    """
   225	    ak = kvs(man, "endpoint_ckpt")
   226	    if "endpoint_ckpt" not in man:
   227	        return ["the manifest carries no endpoint attestation — a chunk is recorded only on the "
   228	                "PRODUCING JOB's own statement of what it wrote. fa_orbit_train.sbatch appends "
   229	                "`endpoint_ckpt <path> endpoint_step <n> endpoint_sha256 <sha>` to this manifest "
   230	                "after classification; a manifest without that line was either written by an older "
   231	                "launcher or belongs to a leg that never finished"]
   232	    attested = ak.get("endpoint_ckpt", "")
   233	    if not attested or attested == "<none>":
   234	        return [f"the producing job attested `endpoint_ckpt <none>` (class "
   235	                f"{ak.get('endpoint_class', '?')}) — it did not reach the success class, or wrote no "
   236	                "single checkpoint at its chunk end; a leg that attests nothing is never a chain link"]
   237	    problems = []
   238	    if str(ak.get("endpoint_step")) != str(chunk_end):
   239	        problems.append(f"the attested endpoint_step {ak.get('endpoint_step')!r} != this leg's "
   240	                        f"chunk_end {chunk_end!r} — the job attests a different boundary")
   241	    if endpoint is None:
   242	        return problems      # the missing/ambiguous endpoint is already a problem upstream
   243	    if os.path.realpath(resolve(repo_root, attested)) != os.path.realpath(endpoint):
   244	        problems.append(f"the attested endpoint {attested} is not the checkpoint found in the audited "
   245	                        f"canonical directory ({endpoint})")
   246	    if ak.get("endpoint_sha256") != endpoint_sha:
   247	        problems.append(f"the attested endpoint_sha256 "
   248	                        f"{str(ak.get('endpoint_sha256'))[:12]} != this recorder's own re-hash of "
   249	                        f"{os.path.basename(endpoint)} ({str(endpoint_sha)[:12]}) — the file on disk "
   250	                        "is not the file that job attested writing")
   251	    return problems
   252	
   253	
   254	def record_chunk(args, arm, reg, initial, man, man_sha, pins, chunk_raw):
   255	    """Append ONE chain link for a chunked leg: <resume_step> -> <chunk_end>.
   256	
   257	    The link is what makes the NEXT chunk admissible (preflight --chain), so it
   258	    is written only when this leg provably resumed the tip of the chain and its
   259	    endpoint checkpoint is on disk, unambiguous, and hashed here from that disk.
   260	    """
   261	    jk, rk = kvs(man, "job"), kvs(man, "resume_ckpt")
   262	    job = jk.get("job")
   263	    problems = []
   264	    if args.extend:
   265	        problems.append("--extend extends a producer manifest and has no meaning for a CHUNK leg; "
   266	                        "each chunk is recorded once, as one immutable chain link")
   267	    if not job:
   268	        problems.append("manifest records no job — a leg with no identity is not a record")
   269	
   270	    # --- the chunk boundary itself ------------------------------------------
   271	    chunk_end = resume_step = None
   272	    try:
   273	        chunk_end = int(chunk_raw)
   274	    except (TypeError, ValueError):
   275	        problems.append(f"manifest chunk_end {chunk_raw!r} is not an integer")
   276	    try:
   277	        resume_step = int(rk.get("expected_step"))
   278	    except (TypeError, ValueError):
   279	        problems.append(f"manifest expected_step {rk.get('expected_step')!r} is not an integer")
   280	    budget = None
   281	    try:
   282	        budget = int(pins.get("PINNED_MAXSTEPS"))
   283	    except (TypeError, ValueError):
   284	        problems.append(f"the launcher's PINNED_MAXSTEPS {pins.get('PINNED_MAXSTEPS')!r} is not an "
   285	                        "integer — the chunk boundary cannot be checked against the budget")
   286	    if chunk_end is not None:
   287	        if chunk_end % CHUNK_STEP:
   288	            problems.append(f"manifest chunk_end {chunk_end} is not a multiple of {CHUNK_STEP} (the "
   289	                            "pinned checkpoint cadence: a chunk must end ON a checkpoint)")
   290	        if resume_step is not None and chunk_end <= resume_step:
   291	            problems.append(f"manifest chunk_end {chunk_end} does not exceed the resume step "
   292	                            f"{resume_step} — the leg would produce no new checkpoint")
   293	        if budget is not None and chunk_end > budget:
   294	            problems.append(f"manifest chunk_end {chunk_end} exceeds the pinned budget {budget}")
   295	
   296	    # --- which link (or the audited anchor) this chunk continues ------------
   297	    chain = initial.get("chain") or []
   298	    mine = [i for i, link in enumerate(chain) if link.get("job") == job]
   299	    if len(mine) > 1:
   300	        raise SystemExit(f"{arm} has {len(mine)} chain links claiming job {job} — the registry is "
   301	                         "inconsistent; fix it before recording")
   302	    at = mine[0] if mine else len(chain)
   303	    # Round-5 review NON-BLOCKING: one leg, one link — by job AND by identity. A
   304	    # different job id carrying an already-recorded launch uuid or manifest is an
   305	    # inconsistent registry, not a second chunk.
   306	    if not mine:
   307	        for i, link in enumerate(chain):
   308	            if jk.get("launch_uuid") and link.get("launch_uuid") == jk.get("launch_uuid"):
   309	                problems.append(f"chain link {i} (job {link.get('job')!r}) already carries launch_uuid "
   310	                                f"{jk.get('launch_uuid')!r} — a different job with the same launch uuid "
   311	                                "means the registry is inconsistent; fix it before recording")
   312	            if link.get("manifest_sha256") == man_sha:
   313	                problems.append(f"chain link {i} (job {link.get('job')!r}) was recorded from a manifest "
   314	                                f"with this exact sha256 {man_sha[:12]} — the same manifest cannot be "
   315	                                "two chunks")
   316	    prev = chain[at - 1] if at > 0 else None
   317	    if prev is None:
   318	        want_anchor = want_step = None                 # the audited INITIAL anchor
   319	        prev_desc = f"the audited INITIAL anchor at step {initial.get('final_step')}"
   320	    else:
   321	        want_anchor, want_step = prev.get("final_ckpt_sha256"), prev.get("final_step")
   322	        prev_desc = f"chain link job {prev.get('job')} ending at step {prev.get('final_step')}"
   323	        # Round-5 review B6: a predecessor that records no endpoint is NOT
   324	        # evidence, and falling back to the INITIAL 40k anchor here would silently
   325	        # re-parent this chunk onto the anchor — the chain's whole point is that
   326	        # every link continues the one before it. Refuse; the fallback exists
   327	        # only for the FIRST link (at == 0).
   328	        if not want_anchor or want_step is None:
   329	            problems.append(f"the predecessor chain link (index {at - 1}, job {prev.get('job')!r}) "
   330	                            "carries no final_ckpt_sha256/final_step — it is not evidence of a "
   331	                            "checkpoint, and a chunk with a predecessor is NEVER re-parented onto the "
   332	                            "audited INITIAL anchor; repair or re-record that link first")
   333	            print("RECORD REFUSED:")
   334	            for p in problems:
   335	                print(f"  !! {p}")
   336	            return 2
   337	    want_time = pins.get(f"PINNED_TIME_LIMIT_CHUNK_{arm}")
   338	    if not want_time:
   339	        problems.append(f"the launcher carries no PINNED_TIME_LIMIT_CHUNK_{arm} pin, so this "
   340	                        "chunk leg's wall time cannot be bound to a pin")
   341	    ident, resume_real = check_identity(arm, man, initial, pins, args.repo_root,
   342	                                        want_anchor=want_anchor, want_step=want_step,
   343	                                        want_time=want_time)
   344	    problems += ident
   345	
   346	    # --- the endpoint checkpoint this leg produced --------------------------
   347	    endpoint = endpoint_sha = None
   348	    if chunk_end is not None:
   349	        ckpt_dir = canonical_ckpt_dir(initial.get("save_dir", ""), arm, args.repo_root)
   350	        hits = sorted(glob.glob(os.path.join(ckpt_dir, f"*-step={chunk_end}.ckpt")))
   351	        if len(hits) != 1:
   352	            problems.append(f"expected exactly 1 checkpoint at the chunk end step {chunk_end} in "
   353	                            f"{ckpt_dir}, found {len(hits)}"
   354	                            + (": " + ", ".join(os.path.basename(h) for h in hits) if hits else ""))
   355	        else:
   356	            endpoint = hits[0]
   357	            endpoint_sha = pm.sha256_file(endpoint)
   358	    problems += check_endpoint_attestation(man, chunk_end, endpoint, endpoint_sha, args.repo_root)
   359	    if problems:
   360	        print("RECORD REFUSED:")
   361	        for p in problems:
   362	            print(f"  !! {p}")
   363	        return 2
   364	
   365	    # check_identity already re-hashed the resume file and proved it IS this hash
   366	    resume_sha = want_anchor if want_anchor is not None else initial.get("final_ckpt_sha256")
   367	    link = {"job": job, "launch_uuid": jk.get("launch_uuid"),
   368	            "manifest_path": os.path.abspath(args.manifest), "manifest_sha256": man_sha,
   369	            "resume_step": resume_step, "resume_ckpt_sha256": resume_sha,
   370	            "final_step": chunk_end, "final_ckpt_sha256": endpoint_sha,
   371	            "recorded_utc": utc_now()}
   372	
   373	    if mine:
   374	        old = chain[at]
   375	        differing = sorted(k for k in link if k != "recorded_utc"
   376	                           and str(old.get(k)) != str(link[k]))
   377	        if differing:
   378	            print("RECORD REFUSED:")
   379	            print(f"  !! {arm} job {job} is ALREADY a chain link whose content differs "
   380	                  f"({', '.join(differing)}) — a recorded link is immutable, and rewriting one "
   381	                  "would re-parent every chunk recorded after it")
   382	            return 2
   383	        print(f"{arm} chunk link job {job} ({resume_step} -> {chunk_end}) is already recorded, "
   384	              "byte-identical — no-op")
   385	        return 0
   386	
   387	    initial.setdefault("chain", []).append(link)
   388	    if not args.dry_run:
   389	        pm.write_atomic(args.registry, reg)
   390	    print(f"recorded {arm} chunk link job {job}: {resume_step} -> {chunk_end}, continuing "
   391	          f"{prev_desc} " + ("(dry run, nothing written)" if args.dry_run else "(published)"))
   392	    print(f"  resume {str(resume_sha)[:12]} -> endpoint {link['final_ckpt_sha256'][:12]} "
   393	          f"({pm.rel_to(args.repo_root, endpoint)})")
   394	    print(f"  {arm} chain is now {len(initial['chain'])} link(s), tip at step {chunk_end}"
   395	          f" of the {budget} budget")
   396	    return 0
   397	
   398	
   399	def main(argv=None):
   400	    ap = argparse.ArgumentParser(description="record an exp_11 RESTART leg")
   401	    ap.add_argument("arm")
   402	    ap.add_argument("manifest")
   403	    ap.add_argument("--registry", default=os.path.join(HERE, "arm_launch_registry.json"))
   404	    ap.add_argument("--launcher", default=os.path.join(HERE, "fa_orbit_train.sbatch"),
   405	                    help="where the Q10 pins are read from")
   406	    ap.add_argument("--producer-dir", default=HERE,
   407	                    help="where the per-leg producer manifests are published")
   408	    # HERE = <repo>/worklog/worklog_<user>/exp_11_fa_orbit_claude
   409	    ap.add_argument("--repo-root", default=os.path.dirname(os.path.dirname(os.path.dirname(HERE))),
   410	                    help="root the manifest's relative paths resolve against")
   411	    ap.add_argument("--extend", action="store_true",
   412	                    help="this leg is already recorded: extend its producer manifest only")
   413	    ap.add_argument("--rehash-all", action="store_true",
   414	                    help="re-hash published checkpoints too (full audit, expensive)")
   415	    ap.add_argument("--dry-run", action="store_true", help="validate and report, publish nothing")
   416	    args = ap.parse_args(argv)
   417	
   418	    arm = args.arm
   419	    pins = read_pins(args.launcher)
   420	    if not pins.get("PINNED_MAXSTEPS"):
   421	        raise SystemExit(f"no PINNED_* values found in {args.launcher}")
   422	
   423	    # One writer at a time, and the lock is the registry's own DIRECTORY: no lock
   424	    # file to leave behind in a tracked tree, and it still covers the tmp+rename.
   425	    store = os.path.dirname(os.path.abspath(args.registry)) or "."
   426	    lock_fd = os.open(store, os.O_RDONLY)
   427	    try:
   428	        fcntl.flock(lock_fd, fcntl.LOCK_EX)
   429	        return record(args, arm, pins)
   430	    finally:
   431	        os.close(lock_fd)
   432	
   433	
   434	def record(args, arm, pins):
   435	    reg = json.load(open(args.registry))
   436	    initial = reg.get("arms", {}).get(arm)
   437	    if initial is None:
   438	        raise SystemExit(f"{arm} has no INITIAL registry entry")
   439	
   440	    raw, man = parse_manifest(args.manifest)
   441	    man_sha = hashlib.sha256(raw).hexdigest()
   442	    # Round 5: a leg that carries a `chunk_end` line stopped at a chunk boundary
   443	    # and is recorded as a CHAIN LINK, not as a whole-budget RESTART row.
   444	    chunk_raw = chunk_end_of(man)
   445	    if chunk_raw is not None:
   446	        return record_chunk(args, arm, reg, initial, man, man_sha, pins, chunk_raw)
   447	    problems, resume_real = check_identity(arm, man, initial, pins, args.repo_root)
   448	    jk, ak, rk = kvs(man, "job"), kvs(man, "arm"), kvs(man, "resume_ckpt")
   449	    job = jk.get("job")
   450	
   451	    legs = reg.setdefault("restarts", {}).setdefault(arm, [])
   452	    same = [l for l in legs if l.get("job") == job or l.get("launch_uuid") == jk.get("launch_uuid")
   453	            or l.get("manifest_sha256") == man_sha]
   454	    if same and not args.extend:
   455	        raise SystemExit(f"{arm} job {job} is ALREADY recorded ({len(same)} matching leg(s)) — "
   456	                         "one leg, one row; use --extend to extend its producer manifest")
   457	    if len(same) > 1:
   458	        problems.append(f"{len(same)} registry rows already claim this leg — the registry is "
   459	                        "inconsistent; fix it before recording")
   460	    if args.extend and not same:
   461	        problems.append(f"--extend given but {arm} job {job} is not recorded yet")
   462	    if problems:
   463	        print("RECORD REFUSED:")
   464	        for p in problems:
   465	            print(f"  !! {p}")
   466	        return 2
   467	
   468	    anchor = initial["final_ckpt_sha256"]
   469	    producer = pm.manifest_name(arm, job)
   470	    row = {
   471	        "manifest_path": args.manifest, "manifest_sha256": man_sha,
   472	        "job": job, "mode": "RESTART", "launch_uuid": jk.get("launch_uuid"),
   473	        "arm": arm, "commit": man.get("commit"), "rung": ak.get("rung"),
   474	        "config_sha256": man.get("config_sha256"), "save_dir": man.get("save_dir"),
   475	        "resume_ckpt": resume_real, "resume_ckpt_sha256": anchor,
   476	        "expected_step": rk.get("expected_step"), "max_steps": ak.get("max_steps"),
   477	        "time_limit": kvs(man, "time_limit").get("time_limit"),
   478	        "producer_manifest": producer, "chains_to": anchor,
   479	        "recorded_at": pm._now(),
   480	    }
   481	    header = {"arm": arm, "job": job, "launch_uuid": jk.get("launch_uuid"), "mode": "RESTART",
   482	              "commit": man.get("commit"), "resume_ckpt_sha256": anchor,
   483	              "expected_step": rk.get("expected_step"), "max_steps": ak.get("max_steps"),
   484	              "save_dir": man.get("save_dir"), "config_sha256": man.get("config_sha256"),
   485	              "chains_to": anchor, "leg_manifest_sha256": man_sha}
   486	
   487	    ckpt_dir = canonical_ckpt_dir(initial["save_dir"], arm, args.repo_root)
   488	    prod_path = os.path.join(args.producer_dir, producer)
   489	    known = (pm.load(prod_path) or {}).get("checkpoints", {})
   490	    found, scan_problems = pm.scan_checkpoints(
   491	        ckpt_dir, int(rk["expected_step"]), int(ak["max_steps"]), known=known,
   492	        rehash_all=args.rehash_all, repo_root=args.repo_root)
   493	    added, kept, pub_problems = pm.publish(prod_path, header, found, dry_run=args.dry_run)
   494	    if scan_problems or pub_problems:
   495	        print("RECORD REFUSED:")
   496	        for p in scan_problems + pub_problems:
   497	            print(f"  !! {p}")
   498	        return 2
   499	
   500	    if args.extend:
   501	        for i, leg in enumerate(legs):
   502	            if leg.get("job") == job:
   503	                legs[i] = {**leg, "producer_manifest": producer}
   504	    else:
   505	        legs.append(row)
   506	    if not args.dry_run:
   507	        pm.write_atomic(args.registry, reg)
   508	    verb = "extended" if args.extend else "recorded"
   509	    print(f"{verb} {arm} RESTART job {job} chaining to {anchor[:12]} "
   510	          f"({'dry run, nothing written' if args.dry_run else 'published'})")
   511	    print(f"  producer manifest {producer}: {len(added)} checkpoint(s) added, "
   512	          f"{len(kept)} already published"
   513	          + (f" (steps {added[0]}..{added[-1]})" if added else ""))
   514	    return 0
   515	
   516	
   517	if __name__ == "__main__":
   518	    sys.exit(main())

2026-08-13T16:48:29.309478Z ERROR codex_models_manager::manager: failed to renew cache TTL: missing field `base_instructions` at line 94 column 5
exec
/usr/bin/bash -lc "nl -ba worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch | sed -n '1,280p'; nl -ba worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch | sed -n '280,620p'; nl -ba worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch | sed -n '620,900p'" in /n/fs/gatrdp/codespace/FLAC
 succeeded in 0ms:
     1	#!/bin/bash
     2	# ============================================================================
     3	# exp_11 ARM TRAINING LAUNCHER — one orbit arm at the ONE pinned P0 rung.
     4	#
     5	# Round-3 review (fa_orbit_codex_code_r3_review.md) rebuilt this script: the
     6	# recipe is now literally pinned rather than operator-supplied, restarts get
     7	# exp_10's torch.load depth, the run directory is locked, the exit taxonomy is a
     8	# tested function, and the environment/W&B destinations are gated.
     9	#
    10	# RECIPE (post-P0, Yixun 2026-08-05): every arm runs WITH ViT gradient
    11	# checkpointing. P0 measured the no-ckpt recipe INFEASIBLE for C8 and richer
    12	# orbits (OOM at micro-8, 45,457 MiB) while checkpointed peaks at ~9.4 GB, so all
    13	# four arms share one checkpointed rung and the ONLY delta between them is the
    14	# averaging orbit.
    15	#
    16	# SUBMIT: never by hand — `fa_orbit_submit.sh <ARM>` derives every Slurm flag
    17	# from the pins below, so an operator cannot mis-enter --gres/--mem/--time.
    18	#   ARM                C4L | C8 | C16 | C32
    19	#   EXPECT_SHA         full 40-hex reviewed commit OID (required). Binding is
    20	#                      by CONTENT of the training surfaces, not HEAD identity:
    21	#                      a launch is accepted when HEAD == EXPECT_SHA, or when
    22	#                      the training closure is byte-identical between the two
    23	#                      (two writers commit to this checkout; worklog/record
    24	#                      commits must not kill a queued leg).
    25	#   RESUME_CKPT/EXPECTED_STEP   crash restart only (see LINEAGE)
    26	#   CHUNK_END          optional, RESTART only: stop THIS job at that step (a
    27	#                      multiple of the 2500 checkpoint cadence, <= the budget) and
    28	#                      let the next job resume from it. The budget pin does not
    29	#                      move; only where this leg stops does. Chunk legs are
    30	#                      chained by fa_orbit_chunk_watchdog.sh via the submitter.
    31	#   SMOKE=1            the reviewed multi-GPU smoke (see SMOKE MODE)
    32	# RUNG / MAXSTEPS / MIN_FREE_MB / time limit are NOT operator inputs any more.
    33	#
    34	# LINEAGE (fail-closed, exactly two stories):
    35	#   INITIAL  no RESUME_CKPT, EXPECTED_STEP unset/0, run directory absent.
    36	#   RESTART  EXPECTED_STEP > 0 AND RESUME_CKPT inside this arm's OWN
    37	#            <RUNDIR>/checkpoints/ AND the checkpoint passes
    38	#            fa_orbit_ckpt_preflight.py (embedded step/config/optimizer/
    39	#            scheduler/EMA + binding to the original launch manifest).
    40	#
    41	# WORLD SIZE: no absence timer (round-3 B4 — a cold start with W&B has no
    42	# measured bound, and `scancel` bypassed classification). Instead: a watcher that
    43	# terminates the torchrun process group the moment Lightning reports the WRONG
    44	# rank count, plus the post-hoc classification in fa_orbit_classify.py.
    45	#
    46	# torchrun: PL 2.1.0 elects TorchElastic before SLURMEnvironment, so the ranks
    47	# torchrun starts are used as-is; the SLURM rank variables are unset so
    48	# SLURMEnvironment cannot claim the job. train.py is unmodified and rank-safe:
    49	# WandbLogger.experiment is @rank_zero_experiment, and ModelCheckpoint.setup
    50	# broadcasts rank 0's dirpath to every rank.
    51	#
    52	# SMOKE MODE (SMOKE=1): the reviewed pre-launch smoke. Bypasses ONLY the "pins
    53	# must be pinned" gate; every other gate still runs. Uses SMOKE_RUNG,
    54	# SMOKE_MAXSTEPS (small), SMOKE_MIN_FREE_MB, its own identity
    55	# (FLAC_exp11_smoke_<ARM> / exp11_smoke_<ARM>) and its own save-dir prefix, so a
    56	# smoke can never touch or resume an arm's real lineage.
    57	#
    58	# TEST HOOK: OUTPUT_ROOT (default outputs_FLAC) relocates the output namespace so
    59	# the guard tests never write under a production prefix. It changes no gate.
    60	# ============================================================================
    61	#SBATCH --partition=all
    62	#SBATCH --nodes=1
    63	#SBATCH --ntasks=1
    64	#SBATCH --output=/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_train_%x_%j.out
    65	# TRANSCRIPT POLICY. This file is written by Slurm for the whole life of the run.
    66	# During the run it is deliberately UNTRACKED (the job removes it from the index
    67	# at launch, see the untrack block below): a tracked file that a running job
    68	# appends to is one a git checkout/stash can unlink out from under the job's file
    69	# descriptor, freezing the visible transcript while the run continues. Completed
    70	# transcripts are committed by the OPERATOR at run closure with `git add -f`.
    71	
    72	set -uo pipefail
    73	
    74	# ============================ PINNED RECIPE =================================
    75	# Filled from the reviewed P0 report; until then every value is the literal
    76	# placeholder and the launcher refuses to run (except under SMOKE=1).
    77	PIN_PLACEHOLDER="TO-PIN-AFTER-P0"
    78	PINNED_RUNG="8x8"                          # P0 run 1334933 + spot 9bf1936: fastest uniform rung where ALL arms fit (C32 peak 30,817 MiB)
    79	PINNED_MB="8"                              # micro-batch per GPU (8 x 8 = 64 = eff = BN batch)
    80	PINNED_NGPU="8"                            # ranks
    81	PINNED_MAXSTEPS=100000                     # Q10: extended budget (was 40000, the
    82	                                           # plan §2 primary matched step, which
    83	                                           # remains the TABLE step — the extension
    84	                                           # adds trajectory, it does not move the
    85	                                           # registered comparison point)
    86	PINNED_CHECKPOINT_EVERY=2500               # exp_07 cadence
    87	PINNED_MIN_FREE_MB="36500"                 # batched C32 peak 32,063 MiB + ~4.4 GB margin (max-across-arms floor)
    88	PINNED_TIME_LIMIT_C4L="24:00:00"           # batched 40k/0.6598 = 16.8 h x1.3 + startup
    89	PINNED_TIME_LIMIT_C8="35:00:00"            # batched 40k/0.4351 = 25.5 h x1.3 + startup
    90	PINNED_TIME_LIMIT_C16="60:00:00"           # batched 40k/0.2454 = 45.3 h x1.3 + startup
    91	PINNED_TIME_LIMIT_C32="112:00:00"          # batched 40k/0.1308 = 84.9 h x1.3 + startup — SINGLE segment (no wall-split needed)
    92	# VANL is the vanilla-conditioning arm of the SAME recipe (Q9): its cost comes
    93	# from the official P0 VAN_8x8 rate, not from an orbit slope, because it makes no
    94	# orbit passes at all — 40k/1.07 steps/s = 10.4 h x1.3 + startup.
    95	PINNED_TIME_LIMIT_VANL="14:00:00"
    96	# Q10 RESTART legs: 40k -> 100k is 60,000 further steps at the batched rates,
    97	# x1.3 + startup. Each must sit under the 168 h partition cap, and each does.
    98	PINNED_TIME_LIMIT_RESTART_C4L="34:00:00"    # 60k/0.6598 = 25.3 h
    99	PINNED_TIME_LIMIT_RESTART_C8="51:00:00"     # 60k/0.4351 = 38.3 h
   100	PINNED_TIME_LIMIT_RESTART_C16="89:00:00"    # 60k/0.2454 = 67.9 h
   101	PINNED_TIME_LIMIT_RESTART_C32="160:00:00"   # 60k/0.1308 = 127.4 h (cap 168 h)
   102	PINNED_TIME_LIMIT_RESTART_VANL="19:00:00"   # 60k/1.0722 = 15.5 h
   103	# Round 5 CHUNKED legs: the partition never backfills a 34-160 h allocation, so a
   104	# leg trains to the next 2500-step boundary and exits. One chunk at the batched
   105	# rates is 2500/rate, rounded up with the same x1.3 margin plus a generous
   106	# startup allowance (cold start + torch/PL import + the 40k-checkpoint load are a
   107	# fixed cost per leg and now amortise over 2500 steps, not 60000):
   108	#   C4L 2500/0.6598 = 1.05 h | C8 1.60 h | C16 2.83 h | C32 5.31 h | VANL 0.65 h
   109	PINNED_TIME_LIMIT_CHUNK_C4L="03:00:00"
   110	PINNED_TIME_LIMIT_CHUNK_C8="04:00:00"
   111	PINNED_TIME_LIMIT_CHUNK_C16="06:00:00"
   112	PINNED_TIME_LIMIT_CHUNK_C32="09:00:00"
   113	PINNED_TIME_LIMIT_CHUNK_VANL="02:30:00"
   114	PINNED_P0_MANIFEST_SHA256="72607b922177208d56055d604b292d697b643ef3b7ab48261ab2e23a0cc2b53b"  # batched matrix manifest bd96575-…-a3ed28eb; spot manifest sha in the commit message
   115	# Environment pins (round-3 B6) — measured on the reviewed environment:
   116	PINNED_PYTHON="/n/fs/gatrdp/envs/flac/bin/python"
   117	PINNED_PL_VERSION="2.1.0"
   118	PINNED_TORCH_VERSION="2.7.0+cu126"
   119	PINNED_VAE_SHA256="8d82159eec35210198246f449bec6561fc19b514922f340a17515050daf7f0b9"
   120	# ============================================================================
   121	
   122	REPO=/n/fs/gatrdp/codespace/FLAC
   123	# TEST HOOK (guard tests only): sbatch copies this script to a spool dir, so the
   124	# repo path must be absolute; FA_ORBIT_REPO_OVERRIDE lets the guard suite point a
   125	# dry run at a worktree. It is honoured ONLY outside a Slurm job and scrubbed
   126	# immediately, so it can never influence a real launch.
   127	if [ -n "${FA_ORBIT_REPO_OVERRIDE:-}" ] && [ -z "${SLURM_JOB_ID:-}" ]; then
   128	  REPO="$FA_ORBIT_REPO_OVERRIDE"
   129	fi
   130	unset FA_ORBIT_REPO_OVERRIDE
   131	EXPDIR="$REPO/worklog/worklog_yixun/exp_11_fa_orbit_claude"
   132	EXP07="$REPO/worklog/worklog_yixun/exp_07_fa_scratch_claude"
   133	cd "$REPO" || exit 3
   134	unset PYTHONPATH PYTHONOPTIMIZE
   135	export PATH=/n/fs/gatrdp/envs/flac/bin:$PATH
   136	export PYTHONNOUSERSITE=1
   137	export HF_HOME=/n/fs/gatrdp/hf_cache
   138	export HF_HUB_OFFLINE=1
   139	
   140	DRYRUN="${DRYRUN:-0}"
   141	SMOKE="${SMOKE:-0}"
   142	# NEW-2: the production output namespace is not operator state. Inside a Slurm
   143	# job it is the literal below; an ambient value that disagrees aborts. The
   144	# override exists only for non-Slurm guard dry runs.
   145	PRODUCTION_OUTPUT_ROOT="outputs_FLAC"
   146	if [ -n "${SLURM_JOB_ID:-}" ]; then
   147	  if [ -n "${OUTPUT_ROOT:-}" ] && [ "$OUTPUT_ROOT" != "$PRODUCTION_OUTPUT_ROOT" ]; then
   148	    echo "ambient OUTPUT_ROOT='${OUTPUT_ROOT}' != the production literal '${PRODUCTION_OUTPUT_ROOT}' - abort"; exit 2
   149	  fi
   150	  OUTPUT_ROOT="$PRODUCTION_OUTPUT_ROOT"
   151	else
   152	  OUTPUT_ROOT="${OUTPUT_ROOT:-$PRODUCTION_OUTPUT_ROOT}"
   153	fi
   154	RESUME_CKPT="${RESUME_CKPT:-}"
   155	EXPECTED_STEP="${EXPECTED_STEP:-0}"
   156	# CHUNK_END (round 5): the step THIS job stops at. Empty = the leg runs to the
   157	# budget, which is every pre-round-5 leg. It narrows train.py's --max-steps only;
   158	# PINNED_MAXSTEPS remains the campaign budget and its gate is untouched.
   159	CHUNK_END="${CHUNK_END:-}"
   160	TS="$(date '+%Y-%m-%d_%H-%M-%S')"
   161	
   162	die() { echo "$1"; exit "${2:-2}"; }
   163	
   164	# --- A. parameters ------------------------------------------------------------
   165	[ -n "${ARM:-}" ] || die "ARM must be exported (C4L|C8|C16|C32|VANL) - abort"
   166	[ -n "${EXPECT_SHA:-}" ] || die "EXPECT_SHA (full reviewed commit sha) must be exported - abort"
   167	case "$ARM" in
   168	  C4L|C8|C16|C32|VANL) ;;
   169	  *) die "ARM '${ARM}' is not a legal exp_11 arm — C4L|C8|C16|C32 only (FA1/VAN/CKPT4 are P0 profiling cells, never arms) - abort" ;;
   170	esac
   171	case "$EXPECTED_STEP" in ''|*[!0-9]*) die "EXPECTED_STEP '${EXPECTED_STEP}' must be a non-negative integer - abort";; esac
   172	
   173	# --- B. the pins decide the recipe (round-3 B1) -------------------------------
   174	if [ "$SMOKE" = "1" ]; then
   175	  RUNG="${SMOKE_RUNG:-}"; MAXSTEPS="${SMOKE_MAXSTEPS:-30}"; MIN_FREE_MB="${SMOKE_MIN_FREE_MB:-}"
   176	  CHECKPOINT_EVERY="${SMOKE_CHECKPOINT_EVERY:-10}"
   177	  [ -n "$RUNG" ] || die "SMOKE=1 requires SMOKE_RUNG (32x2|16x4|8x8) - abort"
   178	  [ -n "$MIN_FREE_MB" ] || die "SMOKE=1 requires SMOKE_MIN_FREE_MB (per-GPU floor) - abort"
   179	  TIME_LIMIT="${SMOKE_TIME:-00:30:00}"; TIME_PIN_NAME="SMOKE_TIME"
   180	  NAME="FLAC_exp11_smoke_${ARM}"; EXPNAME="exp11_smoke_${ARM}"
   181	  SAVEDIR="${OUTPUT_ROOT}/exp11_smoke/${ARM}"
   182	  echo "=== SMOKE MODE: pins bypassed, EVERY other gate active; identity ${EXPNAME} ==="
   183	else
   184	  # Q10 / re-pin fix 1: the wall pin follows the LEG, not the arm. A restart leg
   185	  # is 60,000 further steps, not 40,000 from scratch, so the submitter allocates
   186	  # PINNED_TIME_LIMIT_RESTART_<ARM>. The job selected PINNED_TIME_LIMIT_<ARM>
   187	  # regardless and then rejected its own (correct) allocation in gate H — the
   188	  # third hard-abort path the re-pin review found on jobs 3662828-30. The JOB now
   189	  # selects the same pin the submitter did and enforces THAT one.
   190	  # Round 5: a CHUNK leg is shorter again — it stops at the next boundary — so it
   191	  # carries the arm's CHUNK pin. Submitter and job select by the same rule, and
   192	  # gate H still enforces whichever pin was selected here.
   193	  if [ "$EXPECTED_STEP" -gt 0 ] && [ -n "$CHUNK_END" ]; then
   194	    TIME_PIN_NAME="PINNED_TIME_LIMIT_CHUNK_${ARM}"
   195	  elif [ "$EXPECTED_STEP" -gt 0 ]; then
   196	    TIME_PIN_NAME="PINNED_TIME_LIMIT_RESTART_${ARM}"
   197	  else
   198	    TIME_PIN_NAME="PINNED_TIME_LIMIT_${ARM}"
   199	  fi
   200	  for PIN_NAME in PINNED_RUNG PINNED_MB PINNED_NGPU PINNED_MIN_FREE_MB PINNED_P0_MANIFEST_SHA256 \
   201	                  "$TIME_PIN_NAME"; do
   202	    eval "PIN_VAL=\${$PIN_NAME}"
   203	    [ "$PIN_VAL" != "$PIN_PLACEHOLDER" ] || die "${PIN_NAME} is still '${PIN_PLACEHOLDER}': the P0 report has not been pinned into this launcher yet — no arm may launch (use SMOKE=1 for the pre-launch smoke) - abort"
   204	  done
   205	  RUNG="$PINNED_RUNG"; MAXSTEPS="$PINNED_MAXSTEPS"; MIN_FREE_MB="$PINNED_MIN_FREE_MB"
   206	  CHECKPOINT_EVERY="$PINNED_CHECKPOINT_EVERY"
   207	  eval "TIME_LIMIT=\${${TIME_PIN_NAME}}"
   208	  NAME="FLAC_exp11_${ARM}"; EXPNAME="exp11_${ARM}"; SAVEDIR="${OUTPUT_ROOT}/exp11_${ARM}"
   209	fi
   210	
   211	case "$RUNG" in
   212	  32x2|16x4|8x8) ;;
   213	  *) die "rung '${RUNG}' must be 32x2, 16x4 or 8x8 - abort" ;;
   214	esac
   215	MB="${RUNG%x*}"; NGPU="${RUNG#*x}"
   216	[ "$((MB * NGPU))" -eq 64 ] || die "rung ${RUNG}: MB*NGPU = $((MB*NGPU)) != 64 (micro x N pin, plan §10) - abort"
   217	if [ "$SMOKE" != "1" ]; then
   218	  [ "$MB" = "$PINNED_MB" ] && [ "$NGPU" = "$PINNED_NGPU" ] || die "pin inconsistency: rung ${RUNG} vs PINNED_MB=${PINNED_MB}/PINNED_NGPU=${PINNED_NGPU} - abort"
   219	  [ "$MAXSTEPS" = "100000" ] || die "PINNED_MAXSTEPS is ${MAXSTEPS}, the registered budget is 100000 - abort"
   220	fi
   221	
   222	# --- B2. CHUNK_END: where THIS job stops (round 5) ----------------------------
   223	# The budget gate above is untouched on purpose: CHUNK_END is the per-job stop,
   224	# PINNED_MAXSTEPS stays the campaign budget. STOPSTEP is what train.py is given.
   225	# Fail-closed on every edge, because a chunk that does not land ON a checkpoint
   226	# leaves the chain with nothing to resume.
   227	STOPSTEP="$MAXSTEPS"
   228	if [ -n "$CHUNK_END" ]; then
   229	  [ "$SMOKE" != "1" ] || die "CHUNK_END is a production chunk-chain input and has no meaning under SMOKE=1 (SMOKE_MAXSTEPS is the smoke's budget) - abort"
   230	  case "$CHUNK_END" in ''|*[!0-9]*) die "CHUNK_END '${CHUNK_END}' must be a positive integer - abort";; esac
   231	  [ "$EXPECTED_STEP" -gt 0 ] || die "CHUNK_END ${CHUNK_END} declares a chunk, but only a RESTART leg may be chunked (EXPECTED_STEP is 0) - abort"
   232	  [ "$CHUNK_END" -gt "$EXPECTED_STEP" ] || die "CHUNK_END ${CHUNK_END} must exceed the resume step ${EXPECTED_STEP} — this leg would produce no new checkpoint - abort"
   233	  [ "$CHUNK_END" -le "$MAXSTEPS" ] || die "CHUNK_END ${CHUNK_END} exceeds the budget ${MAXSTEPS} - abort"
   234	  [ "$((CHUNK_END % PINNED_CHECKPOINT_EVERY))" -eq 0 ] || die "CHUNK_END ${CHUNK_END} is not a multiple of the ${PINNED_CHECKPOINT_EVERY}-step checkpoint cadence: the chunk must end ON a checkpoint or the next chunk has nothing to resume - abort"
   235	  STOPSTEP="$CHUNK_END"
   236	fi
   237	RUNDIR="${SAVEDIR}/${NAME}/${EXPNAME}"
   238	echo "=== exp_11 arm ${ARM} @ rung ${RUNG} (MB ${MB} x ${NGPU} GPU, grad-ckpt ON) — ${TS} — host $(hostname) ==="
   239	
   240	# --- C. commit binding + tracked-surface drift --------------------------------
   241	HEAD_SHA="$(git rev-parse HEAD 2>/dev/null)" || HEAD_SHA=""
   242	EXPREL="${EXPDIR#"$REPO"/}"; EXP07REL="${EXP07#"$REPO"/}"
   243	# The drift gate is scoped to CODE surfaces, not the whole exp folder: the four
   244	# arms are running and Slurm appends to their tracked *.out logs continuously, so
   245	# a folder-wide check would abort every screen on a live-log write. Configs,
   246	# drivers and validators are still fully covered. The patterns are QUOTED so
   247	# git, not the shell, expands them — a tracked file deleted from the worktree
   248	# still matches (content-gate review B2) — data/AR (the split JSONs the
   249	# dataloader opens) is covered, and a failing git status is fail-closed.
   250	# arm_launch_registry.json is EXCLUDED (round-5 review B1): it is a lineage
   251	# RECORD the reviewed recorder writes as each chunk finishes, not a measurement
   252	# surface, and it is already outside the commit-binding closure below. Gating it
   253	# deadlocks the chunk chain — the recorder dirties it at 42500 and every later
   254	# leg, on every arm, would abort here until a human committed.
   255	DRIFT="$(git status --porcelain --untracked-files=no -- train.py defaults.ini src ":(exclude)src/tests" data/AR \
   256	          "$EXPREL/*.json" "$EXPREL/*.py" "$EXPREL/*.sbatch" "$EXPREL/*.sh" \
   257	          ":(exclude)$EXPREL/arm_launch_registry.json" \
   258	          "$EXP07REL/FLAC_AR_BF.json" 2>&1)" \
   259	  || die "git status for the drift gate failed: ${DRIFT} - abort"
   260	# Commit binding is CONTENT-scoped: HEAD identity is sufficient but not
   261	# necessary. Two sessions commit to this checkout, so a pending leg must
   262	# survive commits that leave the training closure untouched — and abort on
   263	# any commit that changes it. The closure is what the job actually loads:
   264	# train.py, defaults.ini, src/, the data/AR split JSONs, the five arm
   265	# configs (enumerated — a shell glob would silently drop a config deleted
   266	# since EXPECT_SHA), this launcher, the four runtime helper scripts it
   267	# invokes, and exp_07's FLAC_AR_BF.json (C4L parity baseline).
   268	# Record/analysis files (registry, manifests, gen_*/validators, worklog)
   269	# are deliberately OUTSIDE the closure. Fail-closed on every edge:
   270	# EXPECT_SHA must be the full 40-hex commit OID (a symbolic ref like HEAD
   271	# would defeat the binding), the diff runs against the CAPTURED HEAD OID,
   272	# and HEAD is re-read afterwards to close the mid-gate-commit race.
   273	# src/tests is EXCLUDED from both gates: pytest-only code, never imported by
   274	# train.py at runtime, and the TDD sessions land test files continuously —
   275	# gating them reproduced the exact kill this gate exists to prevent
   276	# (legs 3680875-78 died on a src/tests-only commit, 2026-08-12).
   277	surface_diff_vs_expect() {
   278	  git diff --name-only "${EXPECT_SHA}" "${HEAD_SHA}" -- train.py defaults.ini src ":(exclude)src/tests" data/AR \
   279	      "$EXPDIR"/FLAC_AR_BF_C4L.json "$EXPDIR"/FLAC_AR_BF_C8.json \
   280	      "$EXPDIR"/FLAC_AR_BF_C16.json "$EXPDIR"/FLAC_AR_BF_C32.json \
   280	      "$EXPDIR"/FLAC_AR_BF_C16.json "$EXPDIR"/FLAC_AR_BF_C32.json \
   281	      "$EXPDIR"/FLAC_AR_VANCKPT.json "$EXPDIR"/fa_orbit_train.sbatch \
   282	      "$EXPDIR"/fa_orbit_ckpt_preflight.py "$EXPDIR"/assert_arm_configs_exp11.py \
   283	      "$EXPDIR"/fa_orbit_wandb_readback.py "$EXPDIR"/fa_orbit_classify.py \
   284	      "$EXP07/FLAC_AR_BF.json"
   285	}
   286	GATE_FAIL=""; GATE_OK_MSG=""
   287	if [ -z "$HEAD_SHA" ]; then
   288	  GATE_FAIL="cannot resolve HEAD"
   289	elif ! printf '%s\n' "$EXPECT_SHA" | grep -qE '^[0-9a-f]{40}$'; then
   290	  GATE_FAIL="EXPECT_SHA '${EXPECT_SHA}' is not a full lowercase 40-hex commit id"
   291	elif [ "$(git rev-parse --verify -q "${EXPECT_SHA}^{commit}" 2>/dev/null)" != "$EXPECT_SHA" ]; then
   292	  GATE_FAIL="EXPECT_SHA ${EXPECT_SHA} is not a commit known to this repo"
   293	elif [ "$HEAD_SHA" = "$EXPECT_SHA" ]; then
   294	  GATE_OK_MSG="commit binding OK: ${HEAD_SHA}"
   295	elif SD="$(surface_diff_vs_expect 2>&1)"; then
   296	  if [ -z "$SD" ]; then
   297	    GATE_OK_MSG="commit binding OK (content): training surfaces identical, EXPECT_SHA ${EXPECT_SHA} HEAD ${HEAD_SHA}"
   298	  else
   299	    GATE_FAIL="training surfaces changed since EXPECT_SHA ${EXPECT_SHA} (HEAD ${HEAD_SHA}): ${SD}"
   300	  fi
   301	else
   302	  GATE_FAIL="surface diff vs EXPECT_SHA failed: ${SD}"
   303	fi
   304	if [ -z "$GATE_FAIL" ] && [ "$(git rev-parse HEAD 2>/dev/null)" != "$HEAD_SHA" ]; then
   305	  GATE_FAIL="HEAD moved during the commit-binding check (was ${HEAD_SHA})"
   306	fi
   307	if [ "$DRYRUN" = "1" ]; then
   308	  [ -z "$GATE_FAIL" ] && echo "${GATE_OK_MSG} (dry run)" \
   309	    || echo "DRY-RUN ADVISORY: ${GATE_FAIL} (a real launch aborts here)"
   310	  [ -z "$DRIFT" ] || echo "DRY-RUN ADVISORY: tracked measurement surfaces are modified (a real launch aborts here)"
   311	else
   312	  [ -n "${SLURM_JOB_ID:-}" ] || die "a real launch must run under sbatch (no SLURM_JOB_ID) - abort"
   313	  [ -z "$GATE_FAIL" ] || die "${GATE_FAIL} - abort"
   314	  echo "$GATE_OK_MSG"
   315	  [ -z "$DRIFT" ] || { echo "tracked measurement surfaces modified since review - abort:"; echo "$DRIFT"; exit 2; }
   316	fi
   317	
   318	# --- D. arm -> config (single source) + semantic gate -------------------------
   319	arm_config_for() {
   320	  case "$1" in
   321	    C4L|C8|C16|C32) echo "$EXPDIR/FLAC_AR_BF_$1.json" ;;
   322	    VANL)           echo "$EXPDIR/FLAC_AR_VANCKPT.json" ;;
   323	    *) return 1 ;;
   324	  esac
   325	}
   326	MODEL_CONFIG="$(arm_config_for "$ARM")" || die "no config mapped for arm '${ARM}' - abort"
   327	MODEL_CONFIG_ABS="$(readlink -f "$MODEL_CONFIG" 2>/dev/null)"
   328	[ -n "$MODEL_CONFIG_ABS" ] && [ -f "$MODEL_CONFIG_ABS" ] || die "arm config '${MODEL_CONFIG}' does not exist - abort"
   329	CONFIG_SHA="$(sha256sum "$MODEL_CONFIG_ABS" | awk '{print $1}')"
   330	echo "config for ${ARM}: ${MODEL_CONFIG_ABS} sha256 ${CONFIG_SHA}"
   331	
   332	python3 - "$MODEL_CONFIG_ABS" "$ARM" <<'PY' || die "arm/config semantic gate FAILED - abort"
   333	import json, sys
   334	cfg = json.load(open(sys.argv[1])); arm = sys.argv[2]
   335	t = cfg.get("training", {}); bad = []
   336	# VANL is the same recipe with the conditioning removed, so its gate is the
   337	# MIRROR IMAGE of the orbit arms': the orbit keys must be ABSENT, not merely
   338	# different. A vanilla config that carried a stray frame_avg_angles would be a
   339	# silently fa-flavoured baseline, which would destroy the single-delta claim.
   340	if arm == "VANL":
   341	    cm = t.get("cond_method")
   342	    if cm not in (None, "vanilla"):
   343	        bad.append(f"cond_method={cm!r} (want absent or 'vanilla')")
   344	    if "frame_avg_angles" in t:
   345	        bad.append(f"frame_avg_angles is present ({t['frame_avg_angles']!r}) — a vanilla arm has no orbit")
   346	    want = None
   347	else:
   348	    want = {"C4L": 4, "C8": 8, "C16": 16, "C32": 32}[arm]
   349	    angles = t.get("frame_avg_angles")
   350	    if t.get("cond_method") != "fa_invariant":
   351	        bad.append(f"cond_method={t.get('cond_method')!r} (want fa_invariant)")
   352	    if not isinstance(angles, list) or len(angles) != want:
   353	        bad.append(f"frame_avg_angles has {angles and len(angles)} entries (want {want})")
   354	    elif angles != [k * 360.0 / want for k in range(want)]:
   355	        bad.append(f"frame_avg_angles are not the uniform C{want} orbit")
   356	if t.get("use_ema") is not True:
   357	    bad.append(f"use_ema={t.get('use_ema')!r} (want True)")
   358	vits = [c for c in cfg["model"]["conditioning"]["configs"] if c["type"] == "ViTCoordinates"]
   359	if sorted(c["id"] for c in vits) != ["context_poses_vit", "source_vit"]:
   360	    bad.append(f"ViT conditioner ids {sorted(c['id'] for c in vits)} != the expected two")
   361	# Post-P0: grad-ckpt ON for every arm; the KEY must exist and be literally True
   362	for c in vits:
   363	    if "gradient_checkpointing" not in c["config"]:
   364	        bad.append(f"{c['id']}: gradient_checkpointing key absent (want literal true)")
   365	    elif c["config"]["gradient_checkpointing"] is not True:
   366	        bad.append(f"{c['id']}: gradient_checkpointing={c['config']['gradient_checkpointing']!r} (want True)")
   367	if bad:
   368	    sys.exit("ARM/CONFIG GATE: " + "; ".join(bad))
   369	if arm == "VANL":
   370	    print(f"gate OK: {arm} is vanilla (no cond_method, no orbit), grad-ckpt True, EMA on")
   371	else:
   372	    print(f"gate OK: {arm} carries the uniform C{want} orbit, grad-ckpt True, EMA on")
   373	PY
   374	
   375	# --- E. lineage: INITIAL vs RESTART -------------------------------------------
   376	SAVEDIR_REAL="$(realpath -m "$SAVEDIR")"
   377	CKPT_DIR_REAL="$(realpath -m "${RUNDIR}/checkpoints")"
   378	LAUNCH_MANIFEST_LINK="${SAVEDIR}/launch_manifest.txt"     # written by the INITIAL launch
   379	if [ "$EXPECTED_STEP" -eq 0 ]; then
   380	  MODE="INITIAL"
   381	  [ -z "$RESUME_CKPT" ] || die "INITIAL launch must not carry RESUME_CKPT (set EXPECTED_STEP > 0 to declare a RESTART) - abort"
   382	  [ ! -e "$RUNDIR" ] || die "run directory ${RUNDIR} already exists — an INITIAL launch never clobbers a previous run - abort"
   383	else
   384	  MODE="RESTART"
   385	  [ -n "$RESUME_CKPT" ] || die "EXPECTED_STEP ${EXPECTED_STEP} declares a RESTART, but RESTART requires RESUME_CKPT - abort"
   386	  [ -f "$RESUME_CKPT" ] || die "RESUME_CKPT not found: ${RESUME_CKPT} - abort"
   387	  RESUME_REAL="$(realpath -m "$RESUME_CKPT")"
   388	  # exactly this arm's own checkpoints directory — not merely somewhere below the save root
   389	  case "$RESUME_REAL" in
   390	    "${CKPT_DIR_REAL}"/*.ckpt) ;;
   391	    *) die "a RESTART may only resume a checkpoint from ${CKPT_DIR_REAL}/ (got ${RESUME_REAL}) - abort" ;;
   392	  esac
   393	  [ "$MAXSTEPS" -gt "$EXPECTED_STEP" ] || die "MAXSTEPS ${MAXSTEPS} must exceed the resume step ${EXPECTED_STEP} - abort"
   394	fi
   395	echo "lineage: ${MODE} (expected_step ${EXPECTED_STEP}, max_steps ${MAXSTEPS}, chunk_end ${CHUNK_END:-<none>}, stops at ${STOPSTEP}, ckpt every ${CHECKPOINT_EVERY}, time pin ${TIME_PIN_NAME}=${TIME_LIMIT})"
   396	
   397	# --- F. the exact train.py argv ----------------------------------------------
   398	ARGV=(
   399	  --model-config "$MODEL_CONFIG_ABS"
   400	  --dataset-config src/configs/dataset_configs/AR/train/acousticroom_train.json
   401	  --pretransform-ckpt-path weights/FLAC/VAE.safetensors
   402	  --max-steps "$STOPSTEP" --batch-size "$MB" --accum-batches 1 --num-workers 6 --seed 42
   403	  --num-gpus "$NGPU" --num-nodes 1
   404	  --strategy ddp_find_unused_parameters_true --sync-batchnorm true --precision bf16-mixed
   405	  --val-every -1 --val-dataset-config ''
   406	  --gradient-clip-val 0.0
   407	  --logger wandb --checkpoint-every "$CHECKPOINT_EVERY"
   408	  --name "$NAME" --experiment-name "$EXPNAME" --save-dir "$SAVEDIR"
   409	)
   410	[ "$MODE" = "RESTART" ] && ARGV+=(--ckpt-path "$RESUME_CKPT")
   411	
   412	# --- G. argv-parity dry run (plan N13; round-3 N9 tightened) ------------------
   413	ARGV_FILE="$(mktemp)" || die "mktemp failed - abort" 3
   414	printf '%s\n' "${ARGV[@]}" > "$ARGV_FILE" || die "could not write the argv file - abort" 3
   415	python3 - "$ARGV_FILE" "$MODE" <<'PY'
   416	import sys
   417	# The exp_07 B-F reference argv (bf_scratch_launch.sh) — the lineage this sweep continues.
   418	REF = """--model-config worklog/worklog_yixun/exp_07_fa_scratch_claude/FLAC_AR_BF.json
   419	--dataset-config src/configs/dataset_configs/AR/train/acousticroom_train.json
   420	--pretransform-ckpt-path weights/FLAC/VAE.safetensors
   421	--max-steps 67500 --batch-size 32 --accum-batches 1 --num-workers 6 --seed 42
   422	--num-gpus 2 --strategy ddp_find_unused_parameters_true --sync-batchnorm true
   423	--logger wandb --checkpoint-every 2500
   424	--name FLAC_exp07_BF --experiment-name exp07_BF --save-dir outputs_FLAC/exp07_BF""".split()
   425	# Flags whose VALUE may differ from exp_07 (identity, budget, rung, resume):
   426	ALLOWED_DIFF = {"--model-config", "--name", "--experiment-name", "--save-dir", "--max-steps",
   427	                "--num-gpus", "--batch-size", "--logger", "--checkpoint-every", "--ckpt-path"}
   428	# Flags exp_07 left to defaults.ini and we state explicitly — whitelisted with their
   429	# EXACT expected values (round-3 N9: no "equals the mutable ini" escape hatch):
   430	ALLOWED_ADD = {"--num-nodes": "1", "--precision": "bf16-mixed", "--val-every": "-1",
   431	               "--val-dataset-config": "", "--gradient-clip-val": "0.0", "--ckpt-path": None}
   432	tokens = [t for t in open(sys.argv[1]).read().split("\n")]
   433	if tokens and tokens[-1] == "":
   434	    tokens.pop()
   435	mode = sys.argv[2]
   436	
   437	def as_map(toks):
   438	    out, i = {}, 0
   439	    while i < len(toks):
   440	        flag = toks[i]
   441	        if not flag.startswith("--"):
   442	            raise SystemExit(f"ARGV PARITY: stray token {flag!r}")
   443	        val = toks[i + 1] if i + 1 < len(toks) and not toks[i + 1].startswith("--") else ""
   444	        if flag in out:
   445	            raise SystemExit(f"ARGV PARITY: duplicate flag {flag}")
   446	        out[flag] = val
   447	        i += 2 if (i + 1 < len(toks) and not toks[i + 1].startswith("--")) else 1
   448	    return out
   449	
   450	ref, new = as_map(REF), as_map(tokens)
   451	violations, allowed, explicit = [], [], []
   452	for flag in sorted(set(ref) | set(new)):
   453	    if flag in ref and flag in new:
   454	        if ref[flag] != new[flag]:
   455	            (allowed if flag in ALLOWED_DIFF else violations).append(
   456	                f"{flag}: exp_07 {ref[flag]!r} -> exp_11 {new[flag]!r}")
   457	    elif flag in new:
   458	        if flag == "--ckpt-path":
   459	            (allowed if mode == "RESTART" else violations).append(
   460	                f"--ckpt-path: {new[flag]!r} (RESTART only)")
   461	        elif flag in ALLOWED_ADD and ALLOWED_ADD[flag] == new[flag]:
   462	            explicit.append(f"{flag}={new[flag]!r} (whitelisted explicit default)")
   463	        else:
   464	            violations.append(f"{flag}: added with {new[flag]!r}, not a whitelisted addition "
   465	                              f"(expected {ALLOWED_ADD.get(flag, '<not allowed>')!r})")
   466	    else:
   467	        violations.append(f"{flag}: present in exp_07 ({ref[flag]!r}), MISSING here")
   468	
   469	print("--- train.py argv ---")
   470	print(" ".join(f"{k} {v!r}" if v == "" else f"{k} {v}" for k, v in new.items()))
   471	print("--- argv parity vs exp_07 B-F ---")
   472	for d in allowed:
   473	    print(f"  allowed  {d}")
   474	for d in explicit:
   475	    print(f"  explicit {d}")
   476	if violations:
   477	    print("ARGV PARITY VIOLATIONS:")
   478	    for v in violations:
   479	        print(f"  !! {v}")
   480	    raise SystemExit(2)
   481	print(f"ARGV PARITY OK ({mode}): only whitelisted differences and additions")
   482	PY
   483	parity=$?
   484	rm -f "$ARGV_FILE"
   485	[ "$parity" -eq 0 ] || die "argv parity check FAILED - abort"
   486	if [ "$DRYRUN" = "1" ]; then
   487	  echo "DRY RUN complete: gates A–G passed for ARM=${ARM} RUNG=${RUNG} MODE=${MODE} SMOKE=${SMOKE}"
   488	  echo "  (Slurm/GPU/VRAM/env/wandb/ViT/lock gates and training are skipped in DRYRUN)"
   489	  exit 0
   490	fi
   491	
   492	# --- H. Slurm allocation must match the pins (round-3 B1) ---------------------
   493	[ "${SLURM_JOB_NUM_NODES:-1}" = "1" ] || die "expected 1 node, got ${SLURM_JOB_NUM_NODES} - abort"
   494	[ "${SLURM_NTASKS:-1}" = "1" ] || die "expected 1 task, got ${SLURM_NTASKS} - abort"
   495	WANT_CPUS="$((8 + 7 * NGPU))"; WANT_MEM_MB="$(((12 * NGPU + 12) * 1024))"
   496	GOT_CPUS="${SLURM_CPUS_PER_TASK:-${SLURM_CPUS_ON_NODE:-0}}"
   497	GOT_MEM_MB="${SLURM_MEM_PER_NODE:-0}"
   498	[ "$GOT_CPUS" = "$WANT_CPUS" ] || die "allocated ${GOT_CPUS} CPUs, the pinned rung needs ${WANT_CPUS} — submit via fa_orbit_submit.sh - abort"
   499	[ "$GOT_MEM_MB" = "$WANT_MEM_MB" ] || die "allocated ${GOT_MEM_MB} MB RAM, the pinned rung needs ${WANT_MEM_MB} — submit via fa_orbit_submit.sh - abort"
   500	GOT_TIME="$(squeue -h -j "$SLURM_JOB_ID" -o %l 2>/dev/null | tr -d ' ')"
   501	norm_minutes() { awk -v t="$1" 'BEGIN{d=0; if (t ~ /-/) {split(t,p,"-"); d=p[1]; t=p[2]} n=split(t,c,":");
   502	  if (n==3) m=c[1]*60+c[2]+c[3]/60; else if (n==2) m=c[1]+c[2]/60; else m=t; printf "%d", d*1440+m}'; }
   503	# The pin this ${MODE} leg is entitled to — an INITIAL allocation handed to a
   504	# RESTART leg (or the reverse) is refused here, in the job, not merely intended
   505	# by the submitter.
   506	[ "$(norm_minutes "$GOT_TIME")" = "$(norm_minutes "$TIME_LIMIT")" ] || die "allocated time ${GOT_TIME} != the ${TIME_PIN_NAME} pin ${TIME_LIMIT} this ${MODE} leg requires — submit via fa_orbit_submit.sh - abort"
   507	echo "allocation matches the pins: ${GOT_CPUS} cpus, ${GOT_MEM_MB} MB, ${GOT_TIME} (${TIME_PIN_NAME})"
   508	
   509	mapfile -t GPU_ROWS < <(nvidia-smi --query-gpu=uuid,name --format=csv,noheader,nounits)
   510	[ "${#GPU_ROWS[@]}" -eq "$NGPU" ] || die "rung ${RUNG} needs exactly ${NGPU} visible GPU(s), got ${#GPU_ROWS[@]} - abort"
   511	UUIDS=()
   512	for ROW in "${GPU_ROWS[@]}"; do
   513	  U="$(echo "$ROW" | cut -d, -f1 | tr -d ' ')"; N="$(echo "$ROW" | cut -d, -f2 | sed 's/^ *//')"
   514	  case "$N" in *L40*) ;; *) die "allocated GPU '${N}' is not an L40 - abort";; esac
   515	  UUIDS+=("$U"); echo "GPU: ${N} ${U}"
   516	done
   517	UUID_CSV="$(IFS=,; echo "${UUIDS[*]}")"
   518	
   519	# --- I. environment identity gates (round-3 B6) -------------------------------
   520	REAL_PY="$(command -v python)"
   521	[ "$(readlink -f "$REAL_PY")" = "$(readlink -f "$PINNED_PYTHON")" ] || die "python resolves to ${REAL_PY}, pinned ${PINNED_PYTHON} - abort"
   522	python3 - "$PINNED_PL_VERSION" "$PINNED_TORCH_VERSION" <<'PY' || die "runtime version gate FAILED - abort"
   523	import sys
   524	import pytorch_lightning as pl, torch
   525	want_pl, want_torch = sys.argv[1], sys.argv[2]
   526	bad = []
   527	if pl.__version__ != want_pl:
   528	    bad.append(f"pytorch_lightning {pl.__version__} != pinned {want_pl}")
   529	if torch.__version__ != want_torch:
   530	    bad.append(f"torch {torch.__version__} != pinned {want_torch}")
   531	if bad:
   532	    sys.exit("; ".join(bad))
   533	print(f"runtime gate OK: pl {pl.__version__}, torch {torch.__version__}, cuda {torch.version.cuda}")
   534	PY
   535	VAE_SHA="$(sha256sum weights/FLAC/VAE.safetensors | awk '{print $1}')"
   536	[ "$VAE_SHA" = "$PINNED_VAE_SHA256" ] || die "weights/FLAC/VAE.safetensors sha256 ${VAE_SHA} != pinned ${PINNED_VAE_SHA256} - abort"
   537	echo "VAE pin OK: ${VAE_SHA}"
   538	DRIVER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
   539	CUDA_VER="$(python3 -c 'import torch; print(torch.version.cuda)')"
   540	
   541	# --- J. per-GPU free-VRAM gate (floor from the pins) --------------------------
   542	for U in "${UUIDS[@]}"; do
   543	  FREE="$(nvidia-smi --id="$U" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | tr -dc '0-9')"
   544	  [ -n "$FREE" ] || die "nvidia-smi free-mem query failed on ${U} - refusing to launch blind"
   545	  [ "$FREE" -ge "$MIN_FREE_MB" ] || die "GPU ${U} free ${FREE} MiB < required ${MIN_FREE_MB} MiB - refusing to launch"
   546	done
   547	echo "--- co-tenancy disclosure at launch ---"
   548	nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true
   549	
   550	# --- K. exclusive run ownership via flock (round-3 B3 residual) --------------
   551	# mkdir + stale recovery had two races: a contender could arrive between mkdir
   552	# and the owner write, and release removed the directory without checking whose
   553	# it was. flock has neither: the kernel holds the lock while the fd is open and
   554	# releases it on close (including on kill), so there is no stale state to
   555	# recover and no recovery path to get wrong.
   556	mkdir -p "$OUTPUT_ROOT" || die "could not create ${OUTPUT_ROOT} - abort" 3
   557	LOCKFILE="${OUTPUT_ROOT}/exp11_${ARM}.lock"
   558	exec 9>"$LOCKFILE" || die "could not open the lock file ${LOCKFILE} - abort" 3
   559	if ! flock -n 9; then
   560	  OWNER="$(tr '\n' ' ' < "$LOCKFILE" 2>/dev/null)"
   561	  die "arm ${ARM} is locked by another live job (${OWNER:-<no metadata>}) - refusing a concurrent writer - abort"
   562	fi
   563	LAUNCH_UUID="$(cat /proc/sys/kernel/random/uuid)"
   564	{ echo "job ${SLURM_JOB_ID}"; echo "uuid ${LAUNCH_UUID}"; echo "arm ${ARM}"; echo "mode ${MODE}"; echo "acquired ${TS}"; } >&9 \
   565	  || die "could not write the lock owner metadata - abort" 3
   566	echo "lock acquired: ${LOCKFILE} (flock on fd 9, released on exit)"
   567	mkdir -p "$SAVEDIR" || die "could not create ${SAVEDIR} - abort" 3
   568	
   569	# --- L. RESTART preflight (round-3 B2) ---------------------------------------
   570	CKPT_SHA=""
   571	if [ "$MODE" = "RESTART" ]; then
   572	  PRE_ARGS=(--ckpt "$RESUME_CKPT" --expected-step "$EXPECTED_STEP" --config "$MODEL_CONFIG_ABS"
   573	            --max-steps "$MAXSTEPS" --arm "$ARM" --rung "$RUNG" --commit "$HEAD_SHA")
   574	  [ -n "$LAUNCH_MANIFEST_LINK" ] && PRE_ARGS+=(--launch-manifest "$LAUNCH_MANIFEST_LINK")
   575	  # Q10 / re-pin fix 1: a real arm's restart is the 40k -> 100k EXTENSION, whose
   576	  # contract binds the ORIGINAL launch identity (audited manifest bytes, job,
   577	  # uuid, launch commit, config, save-dir, seed, and the 40k anchor itself)
   578	  # without demanding that the INITIAL budget/commit equal this leg's. SMOKE
   579	  # restarts have no registered launch and keep the ordinary contract.
   580	  #
   581	  # Round 5: only the FIRST leg off the anchor resumes the audited 40k checkpoint.
   582	  # A later chunk resumes the endpoint of the previous chunk, which is evidence
   583	  # only once fa_orbit_record_restart.py has written it as a chain link — so those
   584	  # legs bind with --chain, and an unrecorded predecessor refuses the launch.
   585	  if [ "$SMOKE" != "1" ]; then
   586	    if [ "$EXPECTED_STEP" -eq 40000 ]; then
   587	      PRE_ARGS+=(--extension)
   588	    else
   589	      PRE_ARGS+=(--chain)
   590	    fi
   591	    PRE_ARGS+=(--launch-registry "$EXPDIR/arm_launch_registry.json" --repo-root "$REPO")
   592	    [ -n "$CHUNK_END" ] && PRE_ARGS+=(--chunk-end "$CHUNK_END")
   593	  fi
   594	  PRE_OUT="$(python3 "$EXPDIR/fa_orbit_ckpt_preflight.py" "${PRE_ARGS[@]}" 2>&1)"
   595	  echo "$PRE_OUT"
   596	  echo "$PRE_OUT" | grep -q "^CKPT_SHA256 " || die "restart preflight FAILED - abort"
   597	  CKPT_SHA="$(echo "$PRE_OUT" | awk '/^CKPT_SHA256 /{print $2}')"
   598	fi
   599	
   600	# --- M. wandb: scrub, pin the destination, fix the run id (round-3 B7) --------
   601	unset WANDB_MODE WANDB_DISABLED WANDB_ENTITY WANDB_RUN_ID WANDB_RESUME WANDB_DIR WANDB_PROJECT WANDB_NAME
   602	export WANDB_DIR="$REPO/$SAVEDIR"
   603	export WANDB_MODE=online
   604	eval "$(grep -E '^[[:space:]]*export[[:space:]]+WANDB_API_KEY=' ~/.bashrc 2>/dev/null | tail -1)"
   605	WANDB_ENTITY_SEEN="$(python3 - <<'PY'
   606	import sys
   607	try:
   608	    import wandb
   609	    v = wandb.Api().viewer
   610	except Exception as e:
   611	    sys.exit(f"wandb identity check FAILED: {e}")
   612	if v.email != "yh4742@princeton.edu":
   613	    sys.exit(f"wandb identity {v.email} != yh4742@princeton.edu")
   614	print(v.entity)
   615	PY
   616	)" || die "wandb identity gate FAILED (no logger fallback: the arms train with wandb) - abort"
   617	[ -n "$WANDB_ENTITY_SEEN" ] || die "wandb returned an empty entity - abort"
   618	# B7 residual: pin the destination account explicitly instead of leaving it implicit
   619	export WANDB_ENTITY="$WANDB_ENTITY_SEEN"
   620	echo "wandb identity OK: entity ${WANDB_ENTITY} (exported), mode ${WANDB_MODE}"
   620	echo "wandb identity OK: entity ${WANDB_ENTITY} (exported), mode ${WANDB_MODE}"
   621	if [ "$MODE" = "RESTART" ]; then
   622	  ORIG_WANDB_RUN_ID="$(awk '/^wandb_run_id /{print $2}' "$LAUNCH_MANIFEST_LINK" 2>/dev/null)"
   623	  [ -n "$ORIG_WANDB_RUN_ID" ] || die "RESTART needs the original wandb run id from ${LAUNCH_MANIFEST_LINK} - abort"
   624	  # A RESUMED wandb run refuses config changes: prefigure's push_wandb_config
   625	  # calls config.update() without allow_val_change, and a restart legitimately
   626	  # changes max_steps (40000->100000) and ckpt_path — legs 3684149/3684150
   627	  # died at exactly train.py:193 on this. So a restart runs as a FRESH run
   628	  # carrying the lineage in its name; curves split across runs but panels
   629	  # align on the step axis. train.py stays byte-unchanged (exp_15 admission
   630	  # pins it).
   631	  WANDB_RUN_ID="exp11-${ARM}-r${EXPECTED_STEP}-$(date +%s%N)-$(cut -c1-8 /proc/sys/kernel/random/uuid)"
   632	  export WANDB_RUN_ID
   633	  echo "wandb: new RESTART run id ${WANDB_RUN_ID} (continues ${ORIG_WANDB_RUN_ID}; not resumed — a resumed run refuses the changed max_steps/ckpt_path config)"
   634	else
   635	  WANDB_RUN_ID="exp11-${ARM}-$(date +%s%N)-$(cut -c1-8 /proc/sys/kernel/random/uuid)"
   636	  export WANDB_RUN_ID
   637	  echo "wandb: new run id ${WANDB_RUN_ID}"
   638	fi
   639	
   640	# --- N. DINOv3 pin + init-identity gate (inside the allocation) ---------------
   641	HF_HUB_OFFLINE=1 python3 "$EXPDIR/assert_arm_configs_exp11.py" "$ARM" || die "ViT-pin / init-identity GATE FAILED - abort" 1
   642	
   643	# --- O. atomic manifest, duplicated to the save-dir (round-3 B5) --------------
   644	# --- the live transcript must not be a TRACKED file while it is being written --
   645	# A tracked file that a running job appends to is a file git will happily
   646	# replace. `git rebase --autostash` (remote coordination between machines) stashes
   647	# the dirty working tree, checks out, and restores — and each of those steps
   648	# UNLINKS the path and writes a new inode. The job's stdout descriptor still
   649	# points at the old, now-nameless inode, so every subsequent line is written to a
   650	# file with no name and the visible transcript freezes forever. That is exactly
   651	# what happened to C16 (job 3648696): its transcript stops at Epoch 5 while the
   652	# run went on to step 40000, and all four arm transcripts froze at the same
   653	# instant, 02:04:07, during a rebase cycle.
   654	#
   655	# So: at launch, the job removes its OWN Slurm transcript from the index. The
   656	# file stays on disk untouched; it is simply no longer something git will move.
   657	# The operator commits completed transcripts at closure (git add -f).
   658	SLURM_OUT_AT_LAUNCH="$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null \
   659	                        | tr ' ' '\n' | awk -F= '$1=="StdOut"{print $2; exit}')"
   660	UNTRACK_STATE="not-attempted"
   661	if [ -n "$SLURM_OUT_AT_LAUNCH" ]; then
   662	  if git -C "$REPO" ls-files --error-unmatch "$SLURM_OUT_AT_LAUNCH" >/dev/null 2>&1; then
   663	    if git -C "$REPO" rm --cached --quiet -- "$SLURM_OUT_AT_LAUNCH" 2>/dev/null; then
   664	      UNTRACK_STATE="untracked-at-launch"
   665	      echo "live transcript untracked for the duration of this run: ${SLURM_OUT_AT_LAUNCH}"
   666	      echo "  (the file is untouched on disk; commit it at closure with git add -f)"
   667	    else
   668	      UNTRACK_STATE="untrack-FAILED"
   669	      echo "WARNING: could not untrack ${SLURM_OUT_AT_LAUNCH}; a git working-tree"
   670	      echo "         operation during this run can still detach the transcript"
   671	    fi
   672	  else
   673	    UNTRACK_STATE="already-untracked"      # the normal steady state
   674	  fi
   675	else
   676	  UNTRACK_STATE="stdout-path-unknown"
   677	fi
   678	
   679	TRAINLOG="${EXPDIR}/fa_orbit_${TS}_${ARM}_${RUNG}_jid${SLURM_JOB_ID}_train.log"
   680	SAVEDIR_LOG="${SAVEDIR}/fa_orbit_${TS}_${ARM}_${RUNG}_jid${SLURM_JOB_ID}_train.log"
   681	MANIFEST="${EXPDIR}/fa_orbit_${TS}_${ARM}_${RUNG}_jid${SLURM_JOB_ID}_manifest.txt"
   682	# B5 residual: a failed environment dump must not be silently hashed into the record
   683	PIPFREEZE_FILE="$(mktemp)" || die "mktemp failed - abort" 3
   684	pip freeze > "$PIPFREEZE_FILE" 2>/dev/null || die "pip freeze FAILED - the environment digest would be a lie - abort" 7
   685	ENV_SHA="$(sha256sum "$PIPFREEZE_FILE" | awk '{print $1}')"
   686	rm -f "$PIPFREEZE_FILE"
   687	{
   688	  echo "# exp_11 arm launch manifest"
   689	  echo "timestamp ${TS}"
   690	  echo "job ${SLURM_JOB_ID} host $(hostname) mode ${MODE} launch_uuid ${LAUNCH_UUID}"
   691	  echo "arm ${ARM} rung ${RUNG} micro ${MB} ngpu ${NGPU} max_steps ${MAXSTEPS} ckpt_every ${CHECKPOINT_EVERY}"
   692	  echo "commit ${HEAD_SHA}"
   693	  echo "p0_manifest_sha256 ${PINNED_P0_MANIFEST_SHA256}"
   694	  echo "model_config ${MODEL_CONFIG_ABS}"
   695	  echo "config_sha256 ${CONFIG_SHA}"
   696	  echo "vae_sha256 ${VAE_SHA}"
   697	  echo "env_pip_freeze_sha256 ${ENV_SHA}"
   698	  echo "torch_version $(python3 -c 'import torch; print(torch.__version__)') cuda ${CUDA_VER} driver ${DRIVER}"
   699	  echo "gpu_uuids ${UUID_CSV}"
   700	  echo "time_limit ${TIME_LIMIT} min_free_mb ${MIN_FREE_MB}"
   701	  echo "resume_ckpt ${RESUME_CKPT:-<none>} expected_step ${EXPECTED_STEP} resume_ckpt_sha256 ${CKPT_SHA:-<none>}"
   702	  echo "chunk_end ${CHUNK_END:-<none>}"
   703	  echo "save_dir ${SAVEDIR}"
   704	  echo "slurm_transcript ${SLURM_OUT_AT_LAUNCH:-<unknown>} untrack ${UNTRACK_STATE}"
   705	  echo "train_log ${TRAINLOG}"
   706	  echo "train_log_copy ${SAVEDIR_LOG}"
   707	  echo "wandb_entity ${WANDB_ENTITY_SEEN} wandb_project ${NAME} wandb_name ${EXPNAME}"
   708	  echo "wandb_run_id ${WANDB_RUN_ID}"
   709	  echo "command torchrun --standalone --nnodes=1 --nproc_per_node=${NGPU} train.py ${ARGV[*]}"
   710	} > "${MANIFEST}.tmp" || die "manifest write FAILED - abort" 3
   711	mv "${MANIFEST}.tmp" "$MANIFEST" || die "manifest publication FAILED - abort" 3
   712	cp "$MANIFEST" "${SAVEDIR}/$(basename "$MANIFEST")" || die "manifest copy to the save-dir FAILED - abort" 3
   713	[ "$MODE" = "INITIAL" ] && { cp "$MANIFEST" "$LAUNCH_MANIFEST_LINK" || die "launch-manifest link write FAILED - abort" 3; }
   714	echo "manifest: ${MANIFEST} (copied to ${SAVEDIR})"
   715	
   716	# --- P. training: torchrun + FIFO tee (both statuses captured) ----------------
   717	unset SLURM_NTASKS SLURM_JOB_NAME SLURM_PROCID SLURM_LOCALID SLURM_NODEID
   718	: > "$TRAINLOG" || die "could not create ${TRAINLOG} - abort" 3
   719	: > "$SAVEDIR_LOG" || die "could not create ${SAVEDIR_LOG} - abort" 3
   720	FIFO="$(mktemp)" || die "mktemp failed - abort" 3
   721	rm -f "$FIFO" && mkfifo "$FIFO" || die "mkfifo failed - abort" 3
   722	trap 'rm -f "$FIFO"' EXIT
   723	tee -a "$TRAINLOG" "$SAVEDIR_LOG" < "$FIFO" &
   724	TEE_PID=$!
   725	echo "=== launching ${ARM} ${RUNG}: torchrun --standalone --nproc_per_node=${NGPU} train.py (${MODE}) ==="
   726	torchrun --standalone --nnodes=1 --nproc_per_node="$NGPU" train.py "${ARGV[@]}" > "$FIFO" 2>&1 &
   727	TR_PID=$!
   728	
   729	# world-size watcher: terminate the moment Lightning reports the WRONG rank count
   730	WORLD_RE="All distributed processes registered\. Starting with [0-9]+ processes"
   731	(
   732	  while kill -0 "$TR_PID" 2>/dev/null; do
   733	    if grep -Eq "$WORLD_RE" "$TRAINLOG" 2>/dev/null; then
   734	      GOT="$(grep -Eo "$WORLD_RE" "$TRAINLOG" | head -1 | grep -Eo '[0-9]+')"
   735	      if [ "$GOT" != "$NGPU" ]; then
   736	        echo "WORLD-SIZE WATCHER: Lightning started with ${GOT} processes, expected ${NGPU} — terminating torchrun"
   737	        pkill -TERM -P "$TR_PID" 2>/dev/null; kill -TERM "$TR_PID" 2>/dev/null
   738	      fi
   739	      exit 0
   740	    fi
   741	    sleep 5
   742	  done
   743	) &
   744	WATCHER=$!
   745	
   746	wait "$TR_PID"; rc=$?
   747	kill "$WATCHER" 2>/dev/null; wait "$WATCHER" 2>/dev/null
   748	wait "$TEE_PID"; tee_rc=$?
   749	rm -f "$FIFO"
   750	
   751	# --- Q. W&B run identity verification (round-3 B7 residual) ------------------
   752	# The gate above proves WHO we are; this proves WHERE the run actually landed.
   753	# train.py:165 builds WandbLogger(project=, name=) with NO save_dir, so PL passes
   754	# its default save_dir='.' into wandb.init and that OVERRIDES the exported
   755	# WANDB_DIR: in job 3646734 the run went to $REPO/wandb/run-<ts>-<id> while this
   756	# check looked under $WANDB_DIR/wandb and found nothing (training was green, the
   757	# job still classified 7). We keep exporting WANDB_DIR — other wandb artifacts do
   758	# respect it — but locate the run by the collision-proof id WE generated, which
   759	# wandb embeds in the directory name, across both candidate roots. Exactly one
   760	# match is required; a mismatch is a provenance failure (class 7), not a footnote.
   761	WANDB_CHECK_RC=0
   762	python3 "$EXPDIR/fa_orbit_wandb_readback.py" --run-id "$WANDB_RUN_ID" \
   763	  --root "$REPO" --root "$WANDB_DIR" \
   764	  --entity "$WANDB_ENTITY" --project "$NAME" --name "$EXPNAME" || WANDB_CHECK_RC=$?
   765	if [ "$WANDB_CHECK_RC" -ne 0 ]; then
   766	  echo "W&B run identity could not be verified against the manifest - provenance failure"
   767	fi
   768	
   769	# --- R. classification (tested taxonomy; see fa_orbit_classify.py) ------------
   770	# The completion literal Lightning prints is the stop step it was GIVEN, so the
   771	# classifier is told the step this leg was asked to reach. Without a chunk that
   772	# is MAXSTEPS, byte-identical to every pre-round-5 leg; with one it is the chunk
   773	# boundary, and a successful chunk classifies COMPLETE instead of NO-MARKER.
   774	CLASSIFY_OUT="$(python3 "$EXPDIR/fa_orbit_classify.py" --rc "$rc" --tee-rc "$tee_rc" --ngpu "$NGPU" \
   775	  --maxsteps "$STOPSTEP" --log "$TRAINLOG" --log-copy "$SAVEDIR_LOG" 2>&1)"
   776	final_rc=$?
   777	echo "$CLASSIFY_OUT"
   778	LAST_CKPT="$(ls -1t "${RUNDIR}/checkpoints"/*.ckpt 2>/dev/null | head -1)"
   779	
   780	# --- R2. ENDPOINT ATTESTATION: the job attests what it produced (round-5 B6) --
   781	# The manifest is published BEFORE training, so on its own it proves only what
   782	# this job intended. The recorder then had to associate it with whichever
   783	# exactly-named checkpoint happened to sit in the shared directory — a failed
   784	# job's manifest plus a pre-existing same-arm checkpoint could become a chain
   785	# link, and re-hashing proves current bytes, not authorship. So the job itself
   786	# appends, AFTER classification, what it produced: the endpoint path, its step,
   787	# and its sha256 as this job sees it on disk. fa_orbit_record_restart.py REFUSES
   788	# any chunk manifest without this attestation, and refuses one whose attested
   789	# sha does not match its own re-hash. A leg that did NOT reach the success class
   790	# attests `<none>`, which is likewise a refusal — a failed leg can never be
   791	# laundered into the chain by a checkpoint it did not write.
   792	ATTEST_LINE=""
   793	if [ -n "$CHUNK_END" ]; then
   794	  ENDPOINT_CKPT=""
   795	  if [ "$final_rc" -eq 0 ]; then
   796	    for CAND in "${RUNDIR}/checkpoints"/*-step="${CHUNK_END}".ckpt; do
   797	      [ -f "$CAND" ] || continue
   798	      if [ -n "$ENDPOINT_CKPT" ]; then ENDPOINT_CKPT="<ambiguous>"; break; fi
   799	      ENDPOINT_CKPT="$CAND"
   800	    done
   801	  fi
   802	  if [ "$final_rc" -ne 0 ]; then
   803	    ATTEST_LINE="endpoint_ckpt <none> endpoint_step ${CHUNK_END} endpoint_class ${final_rc}"
   804	    echo "endpoint attestation: this leg did NOT classify as complete (rc=${final_rc}) — attesting <none>"
   805	  elif [ -z "$ENDPOINT_CKPT" ] || [ "$ENDPOINT_CKPT" = "<ambiguous>" ]; then
   806	    ATTEST_LINE="endpoint_ckpt <none> endpoint_step ${CHUNK_END} endpoint_class ${final_rc}"
   807	    echo "endpoint attestation: no single *-step=${CHUNK_END}.ckpt under ${RUNDIR}/checkpoints (found '${ENDPOINT_CKPT:-none}') — attesting <none>"
   808	  else
   809	    ENDPOINT_SHA="$(sha256sum "$ENDPOINT_CKPT" | awk '{print $1}')"
   810	    if [ -n "$ENDPOINT_SHA" ]; then
   811	      ATTEST_LINE="endpoint_ckpt ${ENDPOINT_CKPT} endpoint_step ${CHUNK_END} endpoint_sha256 ${ENDPOINT_SHA}"
   812	      echo "endpoint attestation: ${ENDPOINT_CKPT} step ${CHUNK_END} sha256 ${ENDPOINT_SHA}"
   813	    else
   814	      ATTEST_LINE="endpoint_ckpt <none> endpoint_step ${CHUNK_END} endpoint_class ${final_rc}"
   815	      echo "endpoint attestation: sha256sum of ${ENDPOINT_CKPT} FAILED — attesting <none>"
   816	    fi
   817	  fi
   818	  # The manifest was written pre-run by THIS job; this is the same job appending
   819	  # to its own record. Both copies are kept identical.
   820	  if ! printf '%s\n' "$ATTEST_LINE" >> "$MANIFEST"; then
   821	    echo "ENDPOINT ATTESTATION COULD NOT BE APPENDED to ${MANIFEST} — the recorder will refuse this chunk (fail-closed); record it by hand only after auditing the checkpoint"
   822	  fi
   823	  printf '%s\n' "$ATTEST_LINE" >> "${SAVEDIR}/$(basename "$MANIFEST")" \
   824	    || echo "WARNING: could not append the endpoint attestation to the save-dir manifest copy"
   825	fi
   826	
   827	# --- S. durable final record (round-3 B5 residual) ---------------------------
   828	# The classifier verdict and this summary must survive in BOTH copies; the tee
   829	# status is captured (it used to be discarded) and a failure is class 7. The
   830	# whole preflight transcript (this job's Slurm output so far) is copied into the
   831	# save-dir too, so the durable record is not just torchrun's stdout.
   832	FINAL_RECORD="$(
   833	  echo "=== exp_11 ${ARM} ${RUNG} finished at $(date '+%Y-%m-%d %H:%M:%S') ==="
   834	  echo "torchrun rc=${rc} tee rc=${tee_rc} wandb_identity_rc=${WANDB_CHECK_RC} classified rc=${final_rc}"
   835	  echo "$CLASSIFY_OUT"
   836	  echo "last checkpoint: ${LAST_CKPT:-<none>}"
   837	  [ -n "$CHUNK_END" ] && echo "endpoint attestation: ${ATTEST_LINE:-<not attempted>}"
   838	  echo "restart: fa_orbit_submit.sh ${ARM} --resume ${LAST_CKPT:-<path>} --expected-step <step>"
   839	  [ -n "$CHUNK_END" ] && echo "next chunk: record this leg (fa_orbit_record_restart.py ${ARM} ${MANIFEST}), then fa_orbit_submit.sh ${ARM} --resume <ckpt at ${CHUNK_END}> --expected-step ${CHUNK_END} --chunk-end <${CHUNK_END}+${PINNED_CHECKPOINT_EVERY}>"
   840	)"
   841	printf '%s\n' "$FINAL_RECORD" | tee -a "$TRAINLOG" "$SAVEDIR_LOG" > /dev/null
   842	final_tee_rc="${PIPESTATUS[1]}"
   843	# The transcript path is NOT guessed from the #SBATCH pattern (job 3646734 logged
   844	# "slurm_train_job_3646734.out not found" because SLURM_JOB_NAME is unset inside
   845	# the job): ask Slurm itself.
   846	# --- BEGIN slurm-stdout-helper (guard-tested by fa_orbit_equivprobe_wrapper_test.sh) --
   847	parse_slurm_stdout() {   # `scontrol show job` text on stdin -> the StdOut path
   848	  tr ' ' '\n' | awk -F= '$1=="StdOut"{print $2; exit}'
   849	}
   850	slurm_stdout_path() {    # $1 = job id
   851	  scontrol show job "$1" 2>/dev/null | parse_slurm_stdout
   852	}
   853	# --- END slurm-stdout-helper --------------------------------------------------
   854	SLURM_OUT="$(slurm_stdout_path "$SLURM_JOB_ID")"
   855	PREFLIGHT_COPY_RC=0
   856	if [ -n "$SLURM_OUT" ] && [ -f "$SLURM_OUT" ]; then
   857	  cp "$SLURM_OUT" "${SAVEDIR}/preflight_${TS}_jid${SLURM_JOB_ID}.out" || PREFLIGHT_COPY_RC=$?
   858	  [ "$PREFLIGHT_COPY_RC" -eq 0 ] && echo "preflight transcript duplicated from ${SLURM_OUT}"
   859	else
   860	  echo "PREFLIGHT TRANSCRIPT: scontrol reported StdOut='${SLURM_OUT}' which is not a readable file"
   861	  PREFLIGHT_COPY_RC=1
   862	fi
   863	if [ "$final_tee_rc" -ne 0 ] || [ "$PREFLIGHT_COPY_RC" -ne 0 ] || [ "$WANDB_CHECK_RC" -ne 0 ]; then
   864	  echo "LOG-PROVENANCE: final record tee rc=${final_tee_rc}, preflight copy rc=${PREFLIGHT_COPY_RC}, wandb identity rc=${WANDB_CHECK_RC}"
   865	  [ "$final_rc" -eq 0 ] && final_rc=7
   866	fi
   867	printf '%s\n' "$FINAL_RECORD"
   868	exit "$final_rc"

2026-08-13T16:48:38.289439Z ERROR codex_models_manager::manager: failed to renew cache TTL: missing field `base_instructions` at line 94 column 5
exec
/usr/bin/bash -lc "nl -ba worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_chunk_watchdog.sh | sed -n '1,560p'" in /n/fs/gatrdp/codespace/FLAC
 succeeded in 0ms:
     1	#!/usr/bin/env bash
     2	# ============================================================================
     3	# fa_orbit_chunk_watchdog.sh — login-side driver for exp_11's CHUNKED legs.
     4	#
     5	# WHY. The partition never backfills a 34-160 h allocation, so the 40k -> 100k
     6	# extension legs sat PD indefinitely. A chunked leg asks for hours instead: it
     7	# trains to the next 2500-step boundary, saves, and exits. This script is what
     8	# turns a sequence of such jobs into one continuous run — it watches for an arm
     9	# with no live job, records the chunk that just finished, and submits the next.
    10	#
    11	# WHAT IT MAY DO. Exactly three things: read (squeue/sacct/ls/the registry),
    12	# append to its own log and state file, and call the two sanctioned tools —
    13	# fa_orbit_record_restart.py and fa_orbit_submit.sh. It NEVER calls sbatch
    14	# itself, never deletes or rewrites a checkpoint, a manifest or the registry
    15	# (the recorder owns the registry), and never edits the tracked tree. It also
    16	# never runs fa_orbit_add_anchor.py: anchoring an arm is an OPERATOR action
    17	# (it audits a 40k checkpoint into the lineage), so an arm with no anchor is
    18	# frozen at startup instead (round-5 review B7).
    19	#
    20	# FAIL-CLOSED. Every step is refused rather than guessed: an ambiguous manifest
    21	# or checkpoint, a recorder refusal, or a submitter refusal counts as a failure
    22	# for that arm, and MAX_CONSEC_FAIL consecutive failures FREEZE the arm until a
    23	# human removes its frozen_<ARM> line from the state file. A frozen arm is
    24	# skipped loudly, so the log says why nothing is progressing. The chunk chain
    25	# itself is guarded independently of this script: fa_orbit_ckpt_preflight.py
    26	# --chain refuses a chunk whose predecessor was never recorded, so a watchdog
    27	# bug cannot launder an unrecorded checkpoint into the lineage.
    28	#
    29	# ROUND-5 REVIEW FIXES.
    30	#   B2 singleton + no TOCTOU: one instance per experiment folder (flock on
    31	#      .chunk_watchdog.lock, held for the whole run on fd 8), the arm list is
    32	#      de-duplicated, a FAILING squeue is UNKNOWN (skip) rather than "empty",
    33	#      and the last submitted job is re-checked BY ID before a new submission.
    34	#   B3 settlement: lastjob_<ARM> is retained until sacct reports a TERMINAL
    35	#      state; an empty/lagging/nonterminal sacct skips the arm this poll. The
    36	#      finished chunk's manifest is located BY THE SETTLED JOB ID, so a failed
    37	#      attempt and its retry can never be confused. CANCELLED is a human
    38	#      intervention signal: the arm freezes immediately, it is never retried.
    39	#   B4 terminal chunk: the record reconciliation runs BEFORE the DONE test, so
    40	#      the last chunk (97500 -> 100000) is recorded like every other one. DONE
    41	#      requires the newest checkpoint to be exactly TARGET *and* the recorded
    42	#      chain tip to be TARGET; a checkpoint past TARGET freezes the arm.
    43	#   B7 anchors: every selected arm's audited anchor (final_ckpt_sha256 +
    44	#      final_step) is validated at STARTUP, before anything is submitted.
    45	#
    46	# USAGE (KEY=VALUE arguments only; every key whitelisted, no value is ever eval'd)
    47	#   bash fa_orbit_chunk_watchdog.sh                       # the pinned defaults
    48	#   bash fa_orbit_chunk_watchdog.sh ARMS=C4L,C8 POLL=600
    49	#   bash fa_orbit_chunk_watchdog.sh ONESHOT=1 DRYRUN=1    # one pass, submits nothing
    50	#
    51	#   ARMS=C4L,C8,C16,C32,VANL  CHUNK=2500  TARGET=100000  POLL=300
    52	#   MAX_CONSEC_FAIL=2  ONESHOT=0  DRYRUN=0
    53	#   PER-ARM CHUNK: CHUNK_C4L= CHUNK_C8= CHUNK_C16= CHUNK_C32= CHUNK_VANL=5000
    54	#     A leg pays 10-15 min of startup whatever its length, so a 2500-step chunk
    55	#     is 20-28% overhead for the fast arm (VANL) and ~4% for the slow one (C32).
    56	#     VANL therefore DEFAULTS to 5000 (it still fits its 02:30:00 chunk pin);
    57	#     every other arm defaults to the global CHUNK. An explicit CHUNK= sets the
    58	#     default for the other four arms only — to change VANL, pass CHUNK_VANL=.
    59	#     The submitted chunk end is capped at TARGET either way.
    60	#   TEST HOOKS (change no decision, only where the script reads/writes):
    61	#   OUTPUT_ROOT=outputs_FLAC  REGISTRY=<expdir>/arm_launch_registry.json
    62	#   STATE=<expdir>/.chunk_watchdog_state  LOG=<expdir>/fa_orbit_chunk_watchdog.log
    63	#
    64	# COST. One `squeue` and one `ls` per arm per poll (plus, once an arm is above
    65	# 40000, one short python read of the registry). Nothing else runs on the login
    66	# node; the training itself is entirely inside Slurm.
    67	# ============================================================================
    68	set -uo pipefail
    69	cd "$(git -C "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" rev-parse --show-toplevel)" || exit 3
    70	
    71	EXPDIR="worklog/worklog_yixun/exp_11_fa_orbit_claude"
    72	SUBMITTER="${EXPDIR}/fa_orbit_submit.sh"
    73	RECORDER="${EXPDIR}/fa_orbit_record_restart.py"
    74	PY=/n/fs/gatrdp/envs/flac/bin/python
    75	ANCHOR_STEP=40000                 # where every arm's INITIAL run ended
    76	
    77	ARMS="C4L,C8,C16,C32,VANL"; CHUNK=2500; TARGET=100000; POLL=300
    78	MAX_CONSEC_FAIL=2; ONESHOT=0; DRYRUN=0
    79	# Per-arm chunk size. Empty = "use the global CHUNK"; VANL is 5000 by default
    80	# (its 2500-step leg would be ~a quarter startup overhead — round-5 NON-BLOCKING).
    81	CHUNK_C4L=""; CHUNK_C8=""; CHUNK_C16=""; CHUNK_C32=""; CHUNK_VANL=5000
    82	OUTPUT_ROOT="outputs_FLAC"
    83	REGISTRY="${EXPDIR}/arm_launch_registry.json"
    84	STATE="${EXPDIR}/.chunk_watchdog_state"
    85	LOG="${EXPDIR}/fa_orbit_chunk_watchdog.log"
    86	# B2: the singleton lock is FIXED to the experiment folder — the resource being
    87	# protected is "the arms of exp_11", not any particular state file, so it must
    88	# not be relocatable by an argument.
    89	LOCKFILE="${EXPDIR}/.chunk_watchdog.lock"
    90	
    91	# --- argument parsing: whitelist the KEY, shape-check the VALUE, never eval ---
    92	reject()   { echo "$1" >&2; exit 2; }
    93	is_num()   { case "${1:-}" in ''|*[!0-9]*) return 1 ;; esac; }
    94	is_armset() {
    95	  local v="$1" a
    96	  [ -n "$v" ] || return 1
    97	  local IFS=,
    98	  for a in $v; do
    99	    case "$a" in C4L|C8|C16|C32|VANL) ;; *) return 1 ;; esac
   100	  done
   101	}
   102	for kv in "$@"; do
   103	  case "$kv" in *=*) ;; *) reject "argument '${kv}' is not KEY=VALUE" ;; esac
   104	  key="${kv%%=*}"; val="${kv#*=}"
   105	  case "$key" in
   106	    ARMS)            is_armset "$val" || reject "ARMS='${val}' is not a comma-separated list of C4L|C8|C16|C32|VANL" ;;
   107	    CHUNK|TARGET|POLL|MAX_CONSEC_FAIL|CHUNK_C4L|CHUNK_C8|CHUNK_C16|CHUNK_C32|CHUNK_VANL)
   108	                     is_num "$val" || reject "${key}='${val}' is not a non-negative integer" ;;
   109	    ONESHOT|DRYRUN)  case "$val" in 0|1) ;; *) reject "${key}='${val}' must be 0 or 1" ;; esac ;;
   110	    OUTPUT_ROOT|REGISTRY|STATE|LOG)
   111	                     case "$val" in
   112	                       ''|*[!A-Za-z0-9/._-]*) reject "${key}='${val}' has unsafe characters" ;;
   113	                     esac ;;
   114	    *)               reject "unknown argument '${kv}' (expected ARMS=/CHUNK=/CHUNK_<ARM>=/TARGET=/POLL=/MAX_CONSEC_FAIL=/ONESHOT=/DRYRUN=/OUTPUT_ROOT=/REGISTRY=/STATE=/LOG=)" ;;
   115	  esac
   116	  printf -v "$key" '%s' "$val"      # name whitelisted above; value never parsed
   117	done
   118	[ "$CHUNK" -gt 0 ] || reject "CHUNK must be positive"
   119	[ "$((CHUNK % 2500))" -eq 0 ] || reject "CHUNK=${CHUNK} is not a multiple of the 2500-step checkpoint cadence"
   120	# every per-arm override gets the SAME shape checks as the global one
   121	for A in C4L C8 C16 C32 VANL; do
   122	  eval "PERARM=\${CHUNK_${A}}"                       # name from a literal whitelist
   123	  [ -n "$PERARM" ] || continue
   124	  [ "$PERARM" -gt 0 ] || reject "CHUNK_${A} must be positive"
   125	  [ "$((PERARM % 2500))" -eq 0 ] || reject "CHUNK_${A}=${PERARM} is not a multiple of the 2500-step checkpoint cadence"
   126	done
   127	[ "$TARGET" -gt "$ANCHOR_STEP" ] || reject "TARGET=${TARGET} must exceed the ${ANCHOR_STEP} anchor"
   128	[ "$((TARGET % 2500))" -eq 0 ] || reject "TARGET=${TARGET} is not a multiple of 2500"
   129	[ "$MAX_CONSEC_FAIL" -ge 1 ] || reject "MAX_CONSEC_FAIL must be at least 1"
   130	[ -f "$SUBMITTER" ] || reject "missing ${SUBMITTER}"
   131	[ -f "$RECORDER" ] || reject "missing ${RECORDER}"
   132	[ -f "$REGISTRY" ] || reject "missing ${REGISTRY}"
   133	
   134	# --- B2(a): ONE watchdog per experiment folder --------------------------------
   135	# Two watchdogs are a double-submission engine: both see "no live job", both
   136	# submit the same boundary, and the job's own run-directory flock cannot help
   137	# because it is taken long after scheduling. The lock is held on fd 8 for this
   138	# process's whole lifetime and released by the kernel when it exits, so a killed
   139	# watchdog leaves nothing stale behind.
   140	exec 8>"$LOCKFILE" || reject "could not open the watchdog lock ${LOCKFILE}"
   141	flock -n 8 || reject "another chunk watchdog already holds ${LOCKFILE} — refusing to start a second instance (two watchdogs would submit the same chunk twice)"
   142	
   143	log() { printf '%s %s\n' "$(date -Is)" "$1" >> "$LOG"; echo "$1"; }
   144	
   145	chunk_for() {   # <arm> -> that arm's chunk size (per-arm override, else global)
   146	  local v; eval "v=\${CHUNK_$1:-}"         # $1 is whitelisted by is_armset
   147	  [ -n "$v" ] && { printf '%s' "$v"; return 0; }
   148	  printf '%s' "$CHUNK"
   149	}
   150	
   151	# --- state: a plain KV file a human can read and edit ------------------------
   152	# keys: fail_<ARM> <n> | frozen_<ARM> <when>|<reason> | lastjob_<ARM> <jid> |
   153	#       done_<ARM> <step>.  Removing a frozen_<ARM> line un-freezes that arm AND
   154	#       clears its failure streak (see load_state).
   155	declare -A ST=()
   156	load_state() {
   157	  ST=()
   158	  [ -f "$STATE" ] || return 0
   159	  local k v a
   160	  while read -r k v; do
   161	    case "$k" in ''|\#*) continue ;; esac
   162	    ST["$k"]="$v"
   163	  done < "$STATE"
   164	  # NON-BLOCKING adoption: a human who deletes a frozen_<ARM> line means "try
   165	  # this arm again", but the fail counter that caused the freeze survived and
   166	  # the very next failure re-froze the arm immediately. Clearing a freeze now
   167	  # clears the streak too. Deliberately NARROWER than "reset whenever frozen is
   168	  # absent": a sub-threshold streak (fail < MAX_CONSEC_FAIL, never frozen) must
   169	  # survive across polls or MAX_CONSEC_FAIL could never be reached at all.
   170	  for a in C4L C8 C16 C32 VANL; do
   171	    if [ -z "${ST[frozen_$a]:-}" ] && [ "${ST[fail_$a]:-0}" -ge "$MAX_CONSEC_FAIL" ] 2>/dev/null; then
   172	      ST["fail_$a"]="0"
   173	      log "${a}: frozen_${a} was cleared by hand — its ${MAX_CONSEC_FAIL}-failure streak is reset too"
   174	    fi
   175	  done
   176	}
   177	save_state() {
   178	  local tmp="${STATE}.tmp.$$" k
   179	  {
   180	    echo "# exp_11 chunk watchdog state (plain KV, rewritten atomically)."
   181	    echo "# Delete a frozen_<ARM> line to let the watchdog submit that arm again;"
   182	    echo "# doing so also resets that arm's fail_<ARM> streak to 0 on the next poll."
   183	    if [ "${#ST[@]}" -gt 0 ]; then
   184	      for k in "${!ST[@]}"; do printf '%s %s\n' "$k" "${ST[$k]}"; done | sort
   185	    fi
   186	  } > "$tmp" || { echo "could not write ${tmp}" >&2; return 1; }
   187	  mv -f "$tmp" "$STATE" || { echo "could not publish ${STATE}" >&2; return 1; }
   188	}
   189	
   190	bump_fail() {   # <arm> <reason>
   191	  local arm="$1" reason="$2" n
   192	  n=$(( ${ST[fail_$arm]:-0} + 1 ))
   193	  ST["fail_$arm"]="$n"
   194	  log "${arm}: FAILURE (${reason}) — ${n}/${MAX_CONSEC_FAIL} consecutive"
   195	  if [ "$n" -ge "$MAX_CONSEC_FAIL" ]; then
   196	    ST["frozen_$arm"]="$(date -Is)|${reason}"
   197	    log "${arm}: !!! FROZEN after ${n} consecutive failures (${reason}). No further submission for this arm until its frozen_${arm} line is removed from ${STATE}."
   198	  fi
   199	}
   200	
   201	freeze_now() {  # <arm> <reason> — a freeze that is NOT a retryable failure
   202	  local arm="$1" reason="$2"
   203	  ST["frozen_$arm"]="$(date -Is)|${reason}"
   204	  log "${arm}: !!! FROZEN immediately (${reason}). No further submission for this arm until its frozen_${arm} line is removed from ${STATE}."
   205	}
   206	
   207	clear_fail() {  # <arm> <why>
   208	  if [ "${ST[fail_$1]:-0}" != "0" ]; then
   209	    log "${1}: failure streak cleared (${2})"
   210	  fi
   211	  ST["fail_$1"]="0"
   212	}
   213	
   214	newest_ckpt_step() {   # <ckpt dir> -> the largest step with a checkpoint file
   215	  ls -1 "$1" 2>/dev/null \
   216	    | sed -n 's/^epoch=[0-9]\{1,\}-step=\([0-9]\{1,\}\)\.ckpt$/\1/p' | sort -n | tail -1
   217	}
   218	
   219	chain_tip_step() {     # <arm> -> the arm's recorded chain tip (or its anchor, or 0)
   220	  "$PY" - "$REGISTRY" "$1" <<'PY' 2>/dev/null
   221	import json, sys
   222	row = (json.load(open(sys.argv[1])).get("arms") or {}).get(sys.argv[2]) or {}
   223	chain = row.get("chain") or []
   224	print(int((chain[-1].get("final_step") if chain else row.get("final_step")) or 0))
   225	PY
   226	}
   227	
   228	anchor_state() {       # <arm> -> OK | MISSING | UNREADABLE (B7 startup validation)
   229	  "$PY" - "$REGISTRY" "$1" <<'PY' 2>/dev/null || echo UNREADABLE
   230	import json, sys
   231	row = (json.load(open(sys.argv[1])).get("arms") or {}).get(sys.argv[2]) or {}
   232	print("OK" if row.get("final_ckpt_sha256") and row.get("final_step") is not None else "MISSING")
   233	PY
   234	}
   235	
   236	leg_manifest_by_job() {  # <arm> <jobid> -> the manifest THAT job published
   237	  local arm="$1" jid="$2" f
   238	  local -a hits=()
   239	  for f in "$EXPDIR"/fa_orbit_*_"${arm}"_*_jid"${jid}"_manifest.txt; do
   240	    [ -f "$f" ] && hits+=("$f")
   241	  done
   242	  [ "${#hits[@]}" -eq 1 ] || return 1
   243	  printf '%s\n' "${hits[0]}"
   244	}
   245	
   246	leg_manifest_for() {   # <arm> <chunk_end> -> the ONE launcher manifest for that chunk
   247	  local arm="$1" want="$2" f ce
   248	  local -a hits=()
   249	  for f in "$EXPDIR"/fa_orbit_*_"${arm}"_8x8_jid*_manifest.txt; do
   250	    [ -f "$f" ] || continue
   251	    ce="$(awk '/^chunk_end /{print $2; exit}' "$f" 2>/dev/null)"
   252	    [ "$ce" = "$want" ] && hits+=("$f")
   253	  done
   254	  [ "${#hits[@]}" -eq 1 ] || return 1
   255	  printf '%s\n' "${hits[0]}"
   256	}
   257	
   258	manifest_chunk_end() { # <manifest> -> its chunk_end line
   259	  awk '/^chunk_end /{print $2; exit}' "$1" 2>/dev/null
   260	}
   261	
   262	process_arm() {
   263	  local arm="$1"
   264	  if [ -n "${ST[frozen_$arm]:-}" ]; then
   265	    log "${arm}: FROZEN (${ST[frozen_$arm]}) — skipping; clear it in ${STATE} to resume"
   266	    return 0
   267	  fi
   268	
   269	  # B2(b): a FAILING squeue is UNKNOWN, never "the queue is empty". Reading a
   270	  # scheduler hiccup as "no live job" is how a running leg gets a twin.
   271	  local queued qrc
   272	  queued="$(squeue -h -u "$USER" -n "exp11-${arm}-train" -o '%i %T' 2>/dev/null)"; qrc=$?
   273	  if [ "$qrc" -ne 0 ]; then
   274	    log "${arm}: squeue failed (rc=${qrc}) — the queue state is UNKNOWN, skipping this poll (an unknown queue is never read as an empty one)"
   275	    return 0
   276	  fi
   277	
   278	  # B3: settle the leg we last submitted — but only once the SCHEDULER says it
   279	  # is really over. An empty or lagging sacct is not a verdict: the arm is left
   280	  # alone (lastjob_<ARM> retained) so the same boundary cannot be resubmitted
   281	  # while the previous attempt is still settling.
   282	  local last="${ST[lastjob_$arm]:-}" st settled_job=""
   283	  if [ -n "$last" ] && [ -z "$queued" ]; then
   284	    st="$(sacct -X -n -P -j "$last" -o State 2>/dev/null | head -1)"
   285	    st="${st%%$'\n'*}"
   286	    case "$st" in
   287	      COMPLETED)
   288	        clear_fail "$arm" "job ${last} COMPLETED"
   289	        settled_job="$last"; unset "ST[lastjob_$arm]" ;;
   290	      CANCELLED*)
   291	        # A cancellation is a HUMAN acting on this run. Replacing the job the
   292	        # operator just killed is the one thing the watchdog must never do.
   293	        settled_job="$last"; unset "ST[lastjob_$arm]"
   294	        freeze_now "$arm" "job ${last} was cancelled by operator — human intervention signal, not a retryable failure"
   295	        return 0 ;;
   296	      FAILED|TIMEOUT|NODE_FAIL|OUT_OF_MEMORY|BOOT_FAIL|DEADLINE|PREEMPTED)
   297	        settled_job="$last"; unset "ST[lastjob_$arm]"
   298	        bump_fail "$arm" "job ${last} ended ${st}" ;;
   299	      "")
   300	        log "${arm}: sacct reports nothing yet for job ${last} — NOT settled, skipping this poll (lastjob_${arm} retained)"
   301	        return 0 ;;
   302	      *)
   303	        log "${arm}: job ${last} is ${st}, which is not a terminal state — NOT settled, skipping this poll (lastjob_${arm} retained)"
   304	        return 0 ;;
   305	    esac
   306	    [ -n "${ST[frozen_$arm]:-}" ] && return 0
   307	  fi
   308	
   309	  if [ -n "$queued" ]; then
   310	    log "${arm}: live job (${queued//$'\n'/; }) — nothing to do"
   311	    return 0
   312	  fi
   313	
   314	  local ckdir="${OUTPUT_ROOT}/exp11_${arm}/FLAC_exp11_${arm}/exp11_${arm}/checkpoints"
   315	  local S; S="$(newest_ckpt_step "$ckdir")"
   316	  if [ -z "$S" ]; then
   317	    log "${arm}: no epoch=*-step=*.ckpt under ${ckdir} — nothing to resume, skipping"
   318	    return 0
   319	  fi
   320	  # B4: a checkpoint PAST the target is an impossible state for a chunk chain
   321	  # (every leg stops ON a boundary <= TARGET). It is not success — freeze.
   322	  if [ "$S" -gt "$TARGET" ]; then
   323	    freeze_now "$arm" "newest checkpoint is step ${S}, PAST the target ${TARGET} — impossible for a chunk chain; a human must explain this checkpoint before the chain continues"
   324	    return 0
   325	  fi
   326	
   327	  # B4: a chunk that finished but was never recorded must be recorded FIRST —
   328	  # BEFORE the DONE test, or the terminal chunk (e.g. 97500 -> 100000) would be
   329	  # declared done and never recorded. The next chunk's preflight binds to the
   330	  # recorded chain, not to whatever is on disk.
   331	  local tip
   332	  if [ "$S" -gt "$ANCHOR_STEP" ]; then
   333	    tip="$(chain_tip_step "$arm")"
   334	    is_num "$tip" || { log "${arm}: could not read the chain tip from ${REGISTRY}"; bump_fail "$arm" "registry unreadable"; return 0; }
   335	    if [ "$tip" -lt "$S" ]; then
   336	      local man ce
   337	      # B3(c): the producer manifest is identified by the JOB THAT SETTLED, not
   338	      # by its chunk_end — a failed attempt and its successful retry publish two
   339	      # manifests with the SAME chunk_end, and chunk_end matching then refuses
   340	      # forever. chunk_end is kept as a cross-check on the job we picked.
   341	      if [ -n "$settled_job" ]; then
   342	        if ! man="$(leg_manifest_by_job "$arm" "$settled_job")"; then
   343	          log "${arm}: job ${settled_job} published no single manifest in ${EXPDIR} (zero or ambiguous) — cannot record the finished chunk"
   344	          bump_fail "$arm" "no unique manifest for settled job ${settled_job}"
   345	          return 0
   346	        fi
   347	        ce="$(manifest_chunk_end "$man")"
   348	        if [ "$ce" != "$S" ]; then
   349	          log "${arm}: job ${settled_job}'s manifest declares chunk_end ${ce:-<none>}, but the newest checkpoint on disk is step ${S} — refusing to attribute that checkpoint to this job"
   350	          bump_fail "$arm" "manifest chunk_end ${ce:-<none>} != on-disk step ${S}"
   351	          return 0
   352	        fi
   353	      elif ! man="$(leg_manifest_for "$arm" "$S")"; then
   354	        # no settled job id in state (e.g. the watchdog was restarted with a
   355	        # clean state file): fall back to UNIQUE chunk_end matching, and refuse
   356	        # the moment it is ambiguous.
   357	        log "${arm}: no single launcher manifest with chunk_end ${S} in ${EXPDIR} (zero or ambiguous) and no settled job id in the state — cannot record the finished chunk"
   358	        bump_fail "$arm" "manifest for chunk_end ${S} not uniquely identifiable"
   359	        return 0
   360	      fi
   361	      local -a rec=("$RECORDER" "$arm" "$man" --registry "$REGISTRY")
   362	      [ "$DRYRUN" = "1" ] && rec+=(--dry-run)
   363	      log "${arm}: recording finished chunk ${tip} -> ${S} from $(basename "$man")"
   364	      local out rc
   365	      out="$("$PY" "${rec[@]}" 2>&1)"; rc=$?
   366	      if [ "$rc" -ne 0 ]; then
   367	        log "${arm}: RECORDER REFUSED (rc=${rc}): $(printf '%s' "$out" | head -4 | tr '\n' ' ')"
   368	        bump_fail "$arm" "recorder rc=${rc}"
   369	        return 0
   370	      fi
   371	      clear_fail "$arm" "chunk ${tip} -> ${S} recorded"
   372	      if [ "$DRYRUN" = "1" ]; then
   373	        log "${arm}: DRYRUN — the chunk link was validated but not written, so the next chunk is not yet admissible"
   374	      else
   375	        tip="$(chain_tip_step "$arm")"
   376	      fi
   377	    fi
   378	  fi
   379	
   380	  # B4: DONE means BOTH the disk and the RECORD reached the target. A target
   381	  # reached on disk but missing from the chain is an unfinished lineage, not a
   382	  # finished campaign.
   383	  if [ "$S" -eq "$TARGET" ]; then
   384	    tip="$(chain_tip_step "$arm")"
   385	    if [ "$tip" = "$TARGET" ]; then
   386	      if [ "${ST[done_$arm]:-}" != "$S" ]; then
   387	        ST["done_$arm"]="$S"
   388	        log "${arm}: DONE — newest checkpoint is step ${S} == target ${TARGET}, and the recorded chain tip is ${tip}"
   389	      fi
   390	      return 0
   391	    fi
   392	    log "${arm}: newest checkpoint is the target ${TARGET} but the recorded chain tip is ${tip} — NOT declaring DONE until the terminal chunk is recorded"
   393	    [ "$DRYRUN" = "1" ] || bump_fail "$arm" "terminal chunk at ${TARGET} is not recorded"
   394	    return 0
   395	  fi
   396	
   397	  # B2(d): the name-scoped squeue above is not authoritative for the job WE
   398	  # submitted (a rename, a held job, a scheduler that lists it late). Re-check
   399	  # the last job BY ID before adding another one to the queue.
   400	  if [ -n "${ST[lastjob_$arm]:-}" ]; then
   401	    local byid brc
   402	    byid="$(squeue -h -j "${ST[lastjob_$arm]}" -o '%i %T' 2>/dev/null)"; brc=$?
   403	    if [ "$brc" -ne 0 ]; then
   404	      log "${arm}: squeue -j ${ST[lastjob_$arm]} failed (rc=${brc}) — UNKNOWN, skipping this poll rather than risk a duplicate submission"
   405	      return 0
   406	    fi
   407	    if [ -n "$byid" ]; then
   408	      log "${arm}: job ${ST[lastjob_$arm]} is still ${byid} by ID — skipping (the name-scoped queue check missed it)"
   409	      return 0
   410	    fi
   411	  fi
   412	
   413	  # submit the next chunk through the ONLY sanctioned submitter
   414	  local step; step="$(chunk_for "$arm")"
   415	  local next=$(( S + step ))
   416	  [ "$next" -gt "$TARGET" ] && next="$TARGET"
   417	  local -a ck=("$ckdir"/*-step="${S}".ckpt)
   418	  if [ "${#ck[@]}" -ne 1 ] || [ ! -f "${ck[0]}" ]; then
   419	    log "${arm}: expected exactly one checkpoint file at step ${S} in ${ckdir}, found ${#ck[@]}"
   420	    bump_fail "$arm" "ambiguous checkpoint at step ${S}"
   421	    return 0
   422	  fi
   423	  log "${arm}: submitting chunk ${S} -> ${next} (chunk size ${step}): bash ${SUBMITTER} ${arm} --resume ${ck[0]} --expected-step ${S} --chunk-end ${next} (DRYRUN=${DRYRUN})"
   424	  local out rc jid
   425	  out="$(DRYRUN="$DRYRUN" bash "$SUBMITTER" "$arm" --resume "${ck[0]}" \
   426	          --expected-step "$S" --chunk-end "$next" 2>&1)"; rc=$?
   427	  if [ "$rc" -ne 0 ]; then
   428	    log "${arm}: SUBMIT REFUSED (rc=${rc}): $(printf '%s' "$out" | head -4 | tr '\n' ' ')"
   429	    bump_fail "$arm" "submit rc=${rc}"
   430	    return 0
   431	  fi
   432	  jid="$(printf '%s' "$out" | awk '/^submitted /{print $NF}')"
   433	  if [ -n "$jid" ]; then
   434	    ST["lastjob_$arm"]="$jid"
   435	    log "${arm}: submitted chunk ${S} -> ${next} as job ${jid}"
   436	  else
   437	    log "${arm}: submitter returned 0 without a job id (DRYRUN=${DRYRUN}) — nothing queued"
   438	  fi
   439	  return 0
   440	}
   441	
   442	# --- B2(c): the arm list is DE-DUPLICATED ------------------------------------
   443	# ARMS=C8,C8 is two passes over one arm in one poll: the first submits, the
   444	# second sees the job it just queued only if the scheduler is already listing it.
   445	IFS=',' read -r -a ARM_RAW <<< "$ARMS"
   446	ARM_LIST=()
   447	for A in "${ARM_RAW[@]}"; do
   448	  SEEN=0
   449	  for B in ${ARM_LIST[@]+"${ARM_LIST[@]}"}; do [ "$A" = "$B" ] && SEEN=1; done
   450	  [ "$SEEN" -eq 0 ] && ARM_LIST+=("$A")
   451	done
   452	[ "${#ARM_LIST[@]}" -gt 0 ] || reject "ARMS='${ARMS}' selected no arm"
   453	
   454	log "=== chunk watchdog start: arms ${ARM_LIST[*]} chunk ${CHUNK} (VANL ${CHUNK_VANL}) target ${TARGET} poll ${POLL}s max_consec_fail ${MAX_CONSEC_FAIL} oneshot ${ONESHOT} dryrun ${DRYRUN} pid $$ ==="
   455	
   456	# --- B7: every selected arm must already carry an AUDITED anchor -------------
   457	# VANL's registry row has neither final_ckpt_sha256 nor final_step, so its first
   458	# chunk would be submitted, burn a queue slot, and be refused by the preflight.
   459	# Anchoring is an OPERATOR action (fa_orbit_add_anchor.py audits the 40k
   460	# checkpoint into the lineage) — the watchdog must never do it, so it freezes
   461	# the arm here, before anything is submitted.
   462	load_state
   463	for ARM in "${ARM_LIST[@]}"; do
   464	  [ -n "${ST[frozen_$ARM]:-}" ] && continue
   465	  ANCHOR="$(anchor_state "$ARM")"
   466	  case "$ANCHOR" in
   467	    OK) ;;
   468	    MISSING)    freeze_now "$ARM" "no audited anchor — run fa_orbit_add_anchor.py ${ARM} first" ;;
   469	    *)          freeze_now "$ARM" "the registry row for ${ARM} could not be read from ${REGISTRY} — no audited anchor could be confirmed" ;;
   470	  esac
   471	done
   472	save_state || { echo "the state file ${STATE} could not be written — exiting rather than run with failure counters that do not persist" >&2; exit 3; }
   473	
   474	while :; do
   475	  load_state
   476	  for ARM in "${ARM_LIST[@]}"; do
   477	    process_arm "$ARM"
   478	  done
   479	  # NON-BLOCKING adoption: an unwritable state file means the next poll would
   480	  # re-decide from stale state (re-submitting a boundary, losing a freeze). Stop.
   481	  save_state || { log "FATAL: the state file ${STATE} could not be written — exiting rather than continue with unpersisted state"; exit 3; }
   482	
   483	  REMAINING=0
   484	  for ARM in "${ARM_LIST[@]}"; do
   485	    [ -n "${ST[done_$ARM]:-}" ] && continue
   486	    [ -n "${ST[frozen_$ARM]:-}" ] && continue
   487	    REMAINING=$((REMAINING + 1))
   488	  done
   489	  if [ "$REMAINING" -eq 0 ]; then
   490	    log "=== every arm is DONE or FROZEN — watchdog exiting ==="
   491	    break
   492	  fi
   493	  [ "$ONESHOT" = "1" ] && { log "=== ONESHOT: one pass complete, exiting ==="; break; }
   494	  sleep "$POLL"
   495	done
   496	exit 0

exec
/usr/bin/bash -lc "nl -ba worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submit.sh | sed -n '1,240p'; grep -RIn --exclude='fa_orbit_codex_chunk_kit_review.md' --exclude='*.log' --exclude='*.out' -E 'chunk_watchdog.lock|flock.*(watchdog|chunk)|lastjob_|CHUNK_VANL|endpoint_ckpt' worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submit.sh worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh | head -n 260" in /n/fs/gatrdp/codespace/FLAC
 succeeded in 0ms:
     1	#!/usr/bin/env bash
     2	# ============================================================================
     3	# fa_orbit_submit.sh — the ONLY sanctioned way to submit an exp_11 arm.
     4	#
     5	# Round-3 review B1: an operator must never hand-assemble --gres/--cpus/--mem/
     6	# --time. Every resource flag is derived here from the pins inside
     7	# fa_orbit_train.sbatch (read out of the script itself, so the two can never
     8	# disagree), and the job is refused unless the tracked tree is clean.
     9	#
    10	#   ./fa_orbit_submit.sh C8
    11	#   ./fa_orbit_submit.sh C8 --resume <ckpt> --expected-step 12500
    12	#   ./fa_orbit_submit.sh C8 --resume <ckpt> --expected-step 40000 --chunk-end 42500
    13	#   SMOKE=1 SMOKE_RUNG=16x4 SMOKE_MIN_FREE_MB=14000 ./fa_orbit_submit.sh C4L
    14	#   DRYRUN=1 ./fa_orbit_submit.sh C8        # print the sbatch line, submit nothing
    15	#
    16	# CHUNKED legs (round 5). The partition never backfills a 34-160 h allocation, so
    17	# a leg may declare --chunk-end <n>: it trains to that boundary, saves, and exits,
    18	# and the next leg resumes from there. A chunk leg is walled by
    19	# PINNED_TIME_LIMIT_CHUNK_<ARM> (hours, not days), which is what makes it
    20	# schedulable. The chain is driven by fa_orbit_chunk_watchdog.sh, which submits
    21	# every chunk through THIS script — never through sbatch directly.
    22	#
    23	# Resources per rung (micro x N = 64): --gres=gpu:l40:N, --cpus-per-task=8+7N,
    24	# --mem=(12N+12)G, --time=<the arm's pinned limit>. Each submission is recorded
    25	# in an atomic, no-clobber manifest next to the launcher.
    26	# ============================================================================
    27	set -uo pipefail
    28	cd "$(git -C "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" rev-parse --show-toplevel)" || exit 3
    29	
    30	EXPDIR="worklog/worklog_yixun/exp_11_fa_orbit_claude"
    31	SBATCH_FILE="${EXPDIR}/fa_orbit_train.sbatch"
    32	DRYRUN="${DRYRUN:-0}"
    33	SMOKE="${SMOKE:-0}"
    34	PLACEHOLDER="TO-PIN-AFTER-P0"
    35	
    36	[ -f "$SBATCH_FILE" ] || { echo "missing ${SBATCH_FILE} - abort"; exit 3; }
    37	
    38	ARM="${1:-}"
    39	[ -n "$ARM" ] || { echo "usage: $0 <C4L|C8|C16|C32|VANL> [--resume <ckpt> --expected-step <n> [--chunk-end <n>]] - abort"; exit 2; }
    40	shift
    41	case "$ARM" in C4L|C8|C16|C32|VANL) ;; *) echo "ARM '${ARM}' must be C4L|C8|C16|C32|VANL - abort"; exit 2;; esac
    42	
    43	RESUME_CKPT=""; EXPECTED_STEP=0; CHUNK_END=""
    44	while [ $# -gt 0 ]; do
    45	  case "$1" in
    46	    --resume) RESUME_CKPT="${2:?--resume needs a path}"; shift 2 ;;
    47	    --expected-step) EXPECTED_STEP="${2:?--expected-step needs a number}"; shift 2 ;;
    48	    --chunk-end) CHUNK_END="${2:?--chunk-end needs a number}"; shift 2 ;;
    49	    *) echo "unknown argument '$1' - abort"; exit 2 ;;
    50	  esac
    51	done
    52	case "$EXPECTED_STEP" in ''|*[!0-9]*) echo "--expected-step must be a non-negative integer - abort"; exit 2;; esac
    53	# --- --chunk-end: shape-checked HERE, and again inside the job ---------------
    54	# A chunk is meaningful only for a leg that resumes: it narrows where THIS job
    55	# stops, never what the campaign may reach (the budget pin stays 100000). It must
    56	# land on a saved checkpoint, or the next chunk has nothing to resume from.
    57	CHUNK_BUDGET="$(awk -F= '/^PINNED_MAXSTEPS=/{split($2,a," "); print a[1]; exit}' "$SBATCH_FILE")"
    58	case "$CHUNK_BUDGET" in ''|*[!0-9]*) echo "could not read PINNED_MAXSTEPS from ${SBATCH_FILE} - abort"; exit 3;; esac
    59	if [ -n "$CHUNK_END" ]; then
    60	  case "$CHUNK_END" in ''|*[!0-9]*) echo "--chunk-end must be a positive integer - abort"; exit 2;; esac
    61	  [ "$SMOKE" != "1" ] || { echo "--chunk-end is a production chunk-chain input and has no meaning under SMOKE=1 - abort"; exit 2; }
    62	  { [ -n "$RESUME_CKPT" ] && [ "$EXPECTED_STEP" -gt 0 ]; } \
    63	    || { echo "--chunk-end is valid only together with --resume/--expected-step - abort"; exit 2; }
    64	  [ "$((CHUNK_END % 2500))" -eq 0 ] || { echo "--chunk-end ${CHUNK_END} is not a multiple of 2500 (the pinned checkpoint cadence) - abort"; exit 2; }
    65	  [ "$CHUNK_END" -gt "$EXPECTED_STEP" ] || { echo "--chunk-end ${CHUNK_END} must exceed --expected-step ${EXPECTED_STEP} - abort"; exit 2; }
    66	  [ "$CHUNK_END" -le "$CHUNK_BUDGET" ] || { echo "--chunk-end ${CHUNK_END} exceeds the pinned budget ${CHUNK_BUDGET} - abort"; exit 2; }
    67	fi
    68	
    69	# --- pins are read FROM the launcher, so submitter and job cannot disagree ----
    70	pin() {  # read one PINNED_* value out of the launcher (quoted or bare)
    71	  awk -v k="$1" '$0 ~ "^"k"=" {
    72	        if (match($0, /"[^"]*"/)) { print substr($0, RSTART + 1, RLENGTH - 2) }
    73	        else { split($0, a, "="); split(a[2], b, " "); print b[1] }
    74	        exit }' "$SBATCH_FILE"
    75	}
    76	if [ "$SMOKE" = "1" ]; then
    77	  RUNG="${SMOKE_RUNG:?SMOKE=1 requires SMOKE_RUNG (32x2|16x4|8x8)}"
    78	  TIME_LIMIT="${SMOKE_TIME:-00:30:00}"
    79	  [ -n "${SMOKE_MIN_FREE_MB:-}" ] || { echo "SMOKE=1 requires SMOKE_MIN_FREE_MB - abort"; exit 2; }
    80	  JOBNAME="exp11-smoke-${ARM}"
    81	else
    82	  RUNG="$(pin PINNED_RUNG)"
    83	  # A RESTART leg is a different budget from the INITIAL one: 60k further steps,
    84	  # not 40k from scratch. Selecting the INITIAL limit for a restart would wall-kill
    85	  # every arm partway through the extension.
    86	  # A CHUNK leg is shorter still: it stops at the next boundary, so it is walled
    87	  # by the arm's CHUNK pin (hours) rather than its whole-extension RESTART pin
    88	  # (days) — the whole point of chunking is an allocation the scheduler backfills.
    89	  if [ -n "$CHUNK_END" ]; then
    90	    TIME_LIMIT="$(pin "PINNED_TIME_LIMIT_CHUNK_${ARM}")"
    91	  elif [ -n "${EXPECTED_STEP:-}" ] && [ "${EXPECTED_STEP:-0}" -gt 0 ]; then
    92	    TIME_LIMIT="$(pin "PINNED_TIME_LIMIT_RESTART_${ARM}")"
    93	  else
    94	    TIME_LIMIT="$(pin "PINNED_TIME_LIMIT_${ARM}")"
    95	  fi
    96	  [ -n "$TIME_LIMIT" ] || { echo "the launcher carries no wall pin for this ${ARM} leg - abort"; exit 2; }
    97	  for V in "$RUNG" "$TIME_LIMIT" "$(pin PINNED_MIN_FREE_MB)" "$(pin PINNED_P0_MANIFEST_SHA256)"; do
    98	    [ "$V" != "$PLACEHOLDER" ] || { echo "the launcher still carries ${PLACEHOLDER} pins: the P0 report has not been pinned yet — no arm may be submitted (use SMOKE=1 for the smoke) - abort"; exit 2; }
    99	  done
   100	  JOBNAME="exp11-${ARM}-train"
   101	fi
   102	case "$RUNG" in 32x2|16x4|8x8) ;; *) echo "rung '${RUNG}' must be 32x2|16x4|8x8 - abort"; exit 2;; esac
   103	MB="${RUNG%x*}"; NGPU="${RUNG#*x}"
   104	[ "$((MB * NGPU))" -eq 64 ] || { echo "rung ${RUNG}: MB*NGPU != 64 - abort"; exit 2; }
   105	
   106	# --- drift gate: a queued job must run reviewed, committed code --------------
   107	# The drift gate is scoped to CODE surfaces, not the whole exp folder: the four
   108	# arms are running and Slurm appends to their tracked *.out logs continuously, so
   109	# a folder-wide check would abort every screen on a live-log write. Configs,
   110	# drivers and validators are still fully covered.
   111	#
   112	# arm_launch_registry.json is EXCLUDED (round-5 review B1). It is a lineage
   113	# RECORD written by the reviewed recorder as each chunk finishes — a measurement
   114	# *product*, not a measurement *surface* — and it is already outside the
   115	# launcher's commit-binding closure. Gating it deadlocks the chunk chain: the
   116	# recorder dirties it at 42500, so the very next submission (and every later
   117	# one, for every arm) is refused until a human commits. Keep it outside.
   118	DRIFT="$(git status --porcelain --untracked-files=no -- train.py defaults.ini src \
   119	         "$EXPDIR"/*.json "$EXPDIR"/*.py "$EXPDIR"/*.sbatch "$EXPDIR"/*.sh \
   120	         ":(exclude)${EXPDIR}/arm_launch_registry.json" \
   121	         worklog/worklog_yixun/exp_07_fa_scratch_claude/FLAC_AR_BF.json 2>/dev/null)"
   122	[ -z "$DRIFT" ] || { echo "tracked measurement surfaces have uncommitted changes - commit first, abort:"; echo "$DRIFT"; exit 2; }
   123	SHA="$(git rev-parse HEAD)"
   124	
   125	ARGS=(
   126	  --job-name="$JOBNAME"
   127	  --gres="gpu:l40:${NGPU}"
   128	  --cpus-per-task="$((8 + 7 * NGPU))"
   129	  --mem="$(((12 * NGPU + 12)))G"
   130	  --time="$TIME_LIMIT"
   131	  --export="ALL,ARM=${ARM},EXPECT_SHA=${SHA},OUTPUT_ROOT=outputs_FLAC"
   132	)
   133	[ "$SMOKE" = "1" ] && ARGS[5]="${ARGS[5]},SMOKE=1,SMOKE_RUNG=${SMOKE_RUNG},SMOKE_MIN_FREE_MB=${SMOKE_MIN_FREE_MB},SMOKE_MAXSTEPS=${SMOKE_MAXSTEPS:-30},SMOKE_TIME=${TIME_LIMIT}"
   134	[ -n "$RESUME_CKPT" ] && ARGS[5]="${ARGS[5]},RESUME_CKPT=${RESUME_CKPT},EXPECTED_STEP=${EXPECTED_STEP}"
   135	[ -n "$CHUNK_END" ] && ARGS[5]="${ARGS[5]},CHUNK_END=${CHUNK_END}"
   136	ARGS+=("$SBATCH_FILE")
   137	
   138	echo "arm ${ARM} | rung ${RUNG} (${MB}x${NGPU}) | time ${TIME_LIMIT} | commit ${SHA} | smoke ${SMOKE} | chunk_end ${CHUNK_END:-<none>}"
   139	if [ "$DRYRUN" = "1" ]; then
   140	  echo "DRYRUN sbatch ${ARGS[*]}"
   141	  exit 0
   142	fi
   143	
   144	# --- NEW-3: publish the INTENT before submitting -----------------------------
   145	# The provenance record must exist before the job can exist, otherwise a local
   146	# write failure leaves a queued job nobody recorded. The intent manifest carries
   147	# the exact command and pins; the job id is appended afterwards, and if that
   148	# append fails the exact job we just created is cancelled.
   149	INTENT_ID="$(date +%s%N)-$(cut -c1-8 /proc/sys/kernel/random/uuid)"
   150	MANIFEST="${EXPDIR}/fa_orbit_submission_${ARM}_${INTENT_ID}.txt"
   151	[ ! -e "$MANIFEST" ] || { echo "submission manifest ${MANIFEST} already exists - abort"; exit 2; }
   152	TMP="$(mktemp "${MANIFEST}.XXXXXX")" || exit 3
   153	{
   154	  echo "# exp_11 arm submission (intent published BEFORE sbatch)"
   155	  echo "intent_id ${INTENT_ID}"
   156	  echo "submitted_at $(date -Is)"
   157	  echo "arm ${ARM} rung ${RUNG} micro ${MB} ngpu ${NGPU}"
   158	  echo "jobname ${JOBNAME} time ${TIME_LIMIT} smoke ${SMOKE}"
   159	  echo "commit ${SHA}"
   160	  echo "pins rung=${RUNG} maxsteps=$(pin PINNED_MAXSTEPS) ckpt_every=$(pin PINNED_CHECKPOINT_EVERY) min_free_mb=$(pin PINNED_MIN_FREE_MB) p0_manifest_sha256=$(pin PINNED_P0_MANIFEST_SHA256)"
   161	  echo "resume ${RESUME_CKPT:-<none>} expected_step ${EXPECTED_STEP}"
   162	  echo "chunk_end ${CHUNK_END:-<none>}"
   163	  echo "sbatch sbatch ${ARGS[*]}"
   164	} >> "$TMP" || { echo "intent manifest write failed - abort"; exit 3; }
   165	mv -n "$TMP" "$MANIFEST" || { echo "intent manifest publication failed - abort"; exit 2; }
   166	[ -e "$MANIFEST" ] || { echo "intent manifest ${MANIFEST} did not appear - abort"; exit 2; }
   167	echo "intent manifest: ${MANIFEST}"
   168	
   169	OUT="$(sbatch "${ARGS[@]}" 2>&1)"; JID="$(echo "$OUT" | awk '/Submitted batch job/ {print $NF}')"
   170	if [ -z "$JID" ]; then
   171	  echo "SUBMIT FAILED: ${OUT}"
   172	  echo "submit_failed $(date -Is)" >> "$MANIFEST"
   173	  exit 1
   174	fi
   175	echo "submitted ${ARM} -> job ${JID}"
   176	if ! echo "jobid ${JID}" >> "$MANIFEST"; then
   177	  echo "could not append job id ${JID} to ${MANIFEST} — cancelling the job rather than leave it unrecorded"
   178	  scancel "$JID" || echo "scancel ${JID} FAILED — cancel it by hand NOW"
   179	  exit 2
   180	fi
   181	echo "submission recorded: ${MANIFEST} (job ${JID})"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh:481:attest = f"endpoint_ckpt {p42} endpoint_step 42500 endpoint_sha256 {sha(p42)}\n"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh:485:    body + "endpoint_ckpt <none> endpoint_step 42500 endpoint_class 4\n")
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh:487:    body + f"endpoint_ckpt {p42} endpoint_step 42500 endpoint_sha256 {'e' * 64}\n")
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh:509:expect_cmd "recorder refuses a leg that attested it produced nothing" 2 "endpoint_ckpt <none>" -- \
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh:581:expect_cmd "watchdog rejects an off-cadence per-arm CHUNK_VANL" 2 "not a multiple" -- \
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh:582:  bash "$WD" ONESHOT=1 CHUNK_VANL=2600
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh:589:# lock with a background flock holder and prove the watchdog refuses to start.
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh:590:WD_LOCK="${EXPDIR}/.chunk_watchdog.lock"

exec
/usr/bin/bash -lc "nl -ba worklog/worklog_yixun/exp_11_fa_orbit_claude/arm_launch_registry.json | sed -n '1,180p'; nl -ba worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_add_anchor.py | sed -n '1,280p'; sed -n '430,640p' worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh" in /n/fs/gatrdp/codespace/FLAC
 succeeded in 0ms:
     1	{
     2	  "_comment": [
     3	    "AUDITED exp_11 arm launch registry (final GO-check item 4 / review (b)).",
     4	    "The launch manifests live under gitignored outputs_FLAC and are therefore",
     5	    "MUTABLE evidence: binding a screen to 'whatever the manifest says now' proves",
     6	    "nothing. This committed registry pins each arm's manifest by sha256 plus the",
     7	    "launch facts a screen must re-check (job, INITIAL mode, launch uuid, commit,",
     8	    "rung, config/VAE/P0 hashes, canonical save-dir, training seed).",
     9	    "VANL (Q9, job 3661520) recorded from its PUBLISHED launch manifest after the",
    10	    "INITIAL job started -- an sbatch return alone does not prove a launch happened.",
    11	    "Recorded from the live manifests of the running arms (3648694-97, 3661520).",
    12	    "RESTART legs (Q10, 40k -> 100k) are recorded under 'restarts' as a CHAIN:",
    13	    "each entry's resume_ckpt_sha256 must equal the INITIAL entry's",
    14	    "final_ckpt_sha256, which is the audited 40k checkpoint that leg resumed",
    15	    "from. A checkpoint above 40k is therefore admissible only if the restart",
    16	    "that produced it provably descends from the audited INITIAL run.",
    17	    "Populate with fa_orbit_record_restart.py once a leg's manifest publishes.",
    18	    "Chaining alone is NOT sufficient (re-pin review, finding 2): a leg is only",
    19	    "evidence for the checkpoints it actually PRODUCED, so every leg carries a",
    20	    "'producer_manifest' -- fa_orbit_producer_<ARM>_job<JOB>.json, append-only,",
    21	    "step -> sha256 re-hashed from disk by the recorder. The screen re-hashes the",
    22	    "checkpoint it is about to evaluate and admits it only on an exact",
    23	    "step/sha256/path match published by a leg whose every field here",
    24	    "re-validates against the arm's INITIAL row."
    25	  ],
    26	  "recorded_at": "2026-08-09",
    27	  "training_seed": 42,
    28	  "arms": {
    29	    "C4L": {
    30	      "manifest_path": "outputs_FLAC/exp11_C4L/launch_manifest.txt",
    31	      "manifest_sha256": "d49df42d2f7f9c3f39f1aeb6631da84ef0e0a392c22a8271edadbd83885e814a",
    32	      "job": "3648694",
    33	      "mode": "INITIAL",
    34	      "launch_uuid": "ceb40a63-6ce3-4d38-a2b8-d6c21f1b8cc7",
    35	      "commit": "2b78f995a6d377676bd9d9fb60635ab90032b52d",
    36	      "rung": "8x8",
    37	      "micro": "8",
    38	      "ngpu": "8",
    39	      "max_steps": "40000",
    40	      "config_sha256": "3e677204902c543801679610b58d818de3f6353e7b95baded2667078135ea328",
    41	      "vae_sha256": "8d82159eec35210198246f449bec6561fc19b514922f340a17515050daf7f0b9",
    42	      "p0_manifest_sha256": "72607b922177208d56055d604b292d697b643ef3b7ab48261ab2e23a0cc2b53b",
    43	      "save_dir": "outputs_FLAC/exp11_C4L",
    44	      "training_seed": 42,
    45	      "final_ckpt_sha256": "ed9d7a869ecded98cab78ecc4cef83e579df6643c8ffe564912a9e8ec5c88de8",
    46	      "final_step": 40000
    47	    },
    48	    "C8": {
    49	      "manifest_path": "outputs_FLAC/exp11_C8/launch_manifest.txt",
    50	      "manifest_sha256": "fa1037c300fa3f1100667634864653690049271bd4e2815e419fb205c9068388",
    51	      "job": "3648695",
    52	      "mode": "INITIAL",
    53	      "launch_uuid": "1009ec7f-e5e2-443d-881e-57ae15525676",
    54	      "commit": "2b78f995a6d377676bd9d9fb60635ab90032b52d",
    55	      "rung": "8x8",
    56	      "micro": "8",
    57	      "ngpu": "8",
    58	      "max_steps": "40000",
    59	      "config_sha256": "af4c2f21c2d03bc51448a3c161878443a8e9afd1314695530baf4032e808b3f1",
    60	      "vae_sha256": "8d82159eec35210198246f449bec6561fc19b514922f340a17515050daf7f0b9",
    61	      "p0_manifest_sha256": "72607b922177208d56055d604b292d697b643ef3b7ab48261ab2e23a0cc2b53b",
    62	      "save_dir": "outputs_FLAC/exp11_C8",
    63	      "training_seed": 42,
    64	      "final_ckpt_sha256": "4b58787774b5f15852f2603295d57bbe7a1e2d4e40ba217e548f55d643a52c4c",
    65	      "final_step": 40000
    66	    },
    67	    "C16": {
    68	      "manifest_path": "outputs_FLAC/exp11_C16/launch_manifest.txt",
    69	      "manifest_sha256": "deb07b532fea037d9354b5c635e9ad6a80ad8c022dabdc6dbe0a879a27be3979",
    70	      "job": "3648696",
    71	      "mode": "INITIAL",
    72	      "launch_uuid": "87e0d920-a64f-46f4-a085-2807ebf0f13c",
    73	      "commit": "2b78f995a6d377676bd9d9fb60635ab90032b52d",
    74	      "rung": "8x8",
    75	      "micro": "8",
    76	      "ngpu": "8",
    77	      "max_steps": "40000",
    78	      "config_sha256": "5d8b5c4390dd0e87685bba6ed06566fabe2e1ce5074cce3f6a1f38c442829a80",
    79	      "vae_sha256": "8d82159eec35210198246f449bec6561fc19b514922f340a17515050daf7f0b9",
    80	      "p0_manifest_sha256": "72607b922177208d56055d604b292d697b643ef3b7ab48261ab2e23a0cc2b53b",
    81	      "save_dir": "outputs_FLAC/exp11_C16",
    82	      "training_seed": 42,
    83	      "final_ckpt_sha256": "a8a82309fe4a654efa8cbb00c6ffeeaa2fc765c5916b46cf12aaf2ec2b2d01f0",
    84	      "final_step": 40000
    85	    },
    86	    "C32": {
    87	      "manifest_path": "outputs_FLAC/exp11_C32/launch_manifest.txt",
    88	      "manifest_sha256": "b2d08bc0f27583bd78845e281380906b7f05a737444525f4e32cafa5106b395e",
    89	      "job": "3648697",
    90	      "mode": "INITIAL",
    91	      "launch_uuid": "64c9d2e0-88b7-4730-b36d-e6320274b1e2",
    92	      "commit": "2b78f995a6d377676bd9d9fb60635ab90032b52d",
    93	      "rung": "8x8",
    94	      "micro": "8",
    95	      "ngpu": "8",
    96	      "max_steps": "40000",
    97	      "config_sha256": "8584498e8e3b5d7014e61bc4c48ef71ca62139cdc9f2f30de031cae8da34abdf",
    98	      "vae_sha256": "8d82159eec35210198246f449bec6561fc19b514922f340a17515050daf7f0b9",
    99	      "p0_manifest_sha256": "72607b922177208d56055d604b292d697b643ef3b7ab48261ab2e23a0cc2b53b",
   100	      "save_dir": "outputs_FLAC/exp11_C32",
   101	      "training_seed": 42,
   102	      "final_ckpt_sha256": "1beca05a02ffc7b72884e54671ff2d6836789fadcfe2aa6f0ce27ca2f739c134",
   103	      "final_step": 40000,
   104	      "final_ckpt_path": "outputs_FLAC/exp11_C32/FLAC_exp11_C32/exp11_C32/checkpoints/epoch=8-step=40000.ckpt",
   105	      "anchored_at": "2026-08-10T23:40:52-04:00"
   106	    },
   107	    "VANL": {
   108	      "manifest_path": "outputs_FLAC/exp11_VANL/launch_manifest.txt",
   109	      "manifest_sha256": "113d06a284c6198cf9487e99a2efb7ccde94ae13e656a403fe2af0281d3de8b1",
   110	      "job": "3661520",
   111	      "mode": "INITIAL",
   112	      "launch_uuid": "7337b7ae-7d43-47aa-86b3-87b1daad9be7",
   113	      "commit": "81ddac372076ea92751ae09cbaf371df70f396e5",
   114	      "rung": "8x8",
   115	      "micro": "8",
   116	      "ngpu": "8",
   117	      "max_steps": "40000",
   118	      "config_sha256": "733ca52b66c43538e1b9e603e979678af95ac05d89fd1d481ebb472a285a49d8",
   119	      "vae_sha256": "8d82159eec35210198246f449bec6561fc19b514922f340a17515050daf7f0b9",
   120	      "p0_manifest_sha256": "72607b922177208d56055d604b292d697b643ef3b7ab48261ab2e23a0cc2b53b",
   121	      "save_dir": "outputs_FLAC/exp11_VANL",
   122	      "training_seed": 42,
   123	      "final_ckpt_sha256": "1095f49330b4e7b9c469d69fdbaab1772586055236964b5e347604e712988507",
   124	      "final_step": 40000,
   125	      "final_ckpt_path": "outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=8-step=40000.ckpt",
   126	      "anchored_at": "2026-08-13T12:30:38-04:00"
   127	    }
   128	  },
   129	  "restarts": {}
   130	}
     1	#!/usr/bin/env python3
     2	"""Record an arm's AUDITED final checkpoint (the anchor) in arm_launch_registry.json.
     3	
     4	    python3 fa_orbit_add_anchor.py C32
     5	    python3 fa_orbit_add_anchor.py C32 --dry-run
     6	
     7	The anchor -- ``final_ckpt_sha256`` + ``final_step`` -- is what every later leg
     8	chains to: the restart preflight refuses an extension whose resume checkpoint is
     9	not this hash, the recorder refuses a leg that does not resume it, and the screen
    10	refuses a >40k checkpoint whose leg does not descend from it. It is therefore
    11	written with the same rigor as a leg (fa_orbit_record_restart.py), never by hand:
    12	
    13	  * the arm's INITIAL launch manifest must still hash to the value the registry
    14	    audited, and every identity field in it must equal the registry row -- so the
    15	    anchor is added to a launch whose provenance is still intact, not to whatever
    16	    the mutable manifest under gitignored outputs_FLAC says now;
    17	  * the checkpoint is located by the registry's own save-dir (canonical run
    18	    directory), must be the EXACTLY ONE file at the registered final step, and is
    19	    re-hashed from disk;
    20	  * it is opened once and audited: embedded ``global_step`` == the registered
    21	    budget, embedded ``model_config`` deep-equals the arm's config, warm optimizer
    22	    state, ``lr_schedulers`` and EMA weights all present -- a stripped or foreign
    23	    file is not an anchor;
    24	  * publication is tmp+rename under the same exclusive lock the recorder takes;
    25	  * an arm that already carries a DIFFERENT anchor is refused (re-anchoring would
    26	    silently re-parent every leg already chained to the old one). Re-running with
    27	    the same checkpoint is a no-op.
    28	"""
    29	import argparse
    30	import fcntl
    31	import glob
    32	import hashlib
    33	import json
    34	import os
    35	import sys
    36	
    37	HERE = os.path.dirname(os.path.abspath(__file__))
    38	sys.path.insert(0, HERE)
    39	import fa_orbit_producer_manifest as pm                   # noqa: E402
    40	from fa_orbit_ckpt_preflight import canonical_ckpt_dir    # noqa: E402
    41	from fa_orbit_record_restart import kvs, parse_manifest, read_pins, resolve   # noqa: E402
    42	
    43	
    44	def check_launch_chain(arm, row, pins, repo_root):
    45	    """The INITIAL manifest, byte-for-byte as audited, field for field as recorded."""
    46	    problems = []
    47	    man_path = resolve(repo_root, str(row.get("manifest_path", "")))
    48	    if not row.get("manifest_path") or not os.path.isfile(man_path):
    49	        return [f"the registered INITIAL launch manifest {man_path} does not exist — the anchor "
    50	                "cannot be bound to the launch that produced it"], {}
    51	    raw, man = parse_manifest(man_path)
    52	    got = hashlib.sha256(raw).hexdigest()
    53	    if got != row.get("manifest_sha256"):
    54	        problems.append(f"launch manifest sha256 {got[:12]} != audited "
    55	                        f"{str(row.get('manifest_sha256'))[:12]} — it changed after registration")
    56	    jk, ak = kvs(man, "job"), kvs(man, "arm")
    57	    for label, a, b in (("arm", ak.get("arm"), arm),
    58	                        ("job", jk.get("job"), row.get("job")),
    59	                        ("mode", jk.get("mode"), "INITIAL"),
    60	                        ("launch_uuid", jk.get("launch_uuid"), row.get("launch_uuid")),
    61	                        ("commit", man.get("commit"), row.get("commit")),
    62	                        ("rung", ak.get("rung"), row.get("rung")),
    63	                        ("max_steps", ak.get("max_steps"), row.get("max_steps")),
    64	                        ("config_sha256", man.get("config_sha256"), row.get("config_sha256")),
    65	                        ("vae_sha256", man.get("vae_sha256"), row.get("vae_sha256")),
    66	                        ("p0_manifest_sha256", man.get("p0_manifest_sha256"),
    67	                         row.get("p0_manifest_sha256")),
    68	                        ("save_dir", man.get("save_dir"), row.get("save_dir"))):
    69	        if a != b:
    70	            problems.append(f"manifest {label} {a!r} != the registered {b!r}")
    71	    if ak.get("rung") != pins.get("PINNED_RUNG"):
    72	        problems.append(f"manifest rung {ak.get('rung')!r} != the pinned {pins.get('PINNED_RUNG')!r}")
    73	    if man.get("vae_sha256") != pins.get("PINNED_VAE_SHA256"):
    74	        problems.append("manifest vae_sha256 != the launcher's PINNED_VAE_SHA256")
    75	    if int(row.get("training_seed", -1)) != 42:
    76	        problems.append(f"registered training seed {row.get('training_seed')!r} != 42")
    77	    cfg = man.get("model_config", "")
    78	    if not cfg or not os.path.isfile(cfg):
    79	        problems.append(f"manifest model_config {cfg!r} does not exist")
    80	    elif hashlib.sha256(open(cfg, "rb").read()).hexdigest() != row.get("config_sha256"):
    81	        problems.append(f"{cfg} no longer hashes to the audited config_sha256")
    82	    return problems, man
    83	
    84	
    85	def find_final_ckpt(row, arm, step, repo_root):
    86	    """The EXACTLY ONE checkpoint at the registered final step, in the canonical dir."""
    87	    ckpt_dir = canonical_ckpt_dir(str(row.get("save_dir", "")), arm, repo_root)
    88	    hits = sorted(glob.glob(os.path.join(ckpt_dir, f"*-step={step}.ckpt")))
    89	    if len(hits) != 1:
    90	        return None, [f"expected exactly 1 checkpoint at step {step} in {ckpt_dir}, found "
    91	                      f"{len(hits)}{': ' + ', '.join(os.path.basename(h) for h in hits) if hits else ''}"]
    92	    return hits[0], []
    93	
    94	
    95	def audit_ckpt(path, step, config_path):
    96	    """One torch.load: this file IS that run's finished checkpoint, warm and EMA-bearing."""
    97	    import torch                      # deferred: --help and unit imports stay cheap
    98	    problems = []
    99	    try:
   100	        ck = torch.load(path, map_location="cpu", weights_only=False)
   101	    except Exception as exc:
   102	        return [f"{path} is not loadable as a checkpoint: {type(exc).__name__}: {exc}"]
   103	    if not isinstance(ck, dict):
   104	        return [f"not a Lightning checkpoint: {path}"]
   105	    if ck.get("global_step") != step:
   106	        problems.append(f"embedded global_step {ck.get('global_step')} != the registered final "
   107	                        f"step {step}")
   108	    mc = ck.get("model_config")
   109	    if not isinstance(mc, dict):
   110	        problems.append("checkpoint carries no embedded 'model_config' dict")
   111	    elif mc != json.load(open(config_path)):
   112	        problems.append(f"embedded model_config != {config_path} (parsed-object mismatch)")
   113	    opts = ck.get("optimizer_states") or []
   114	    if len(opts) != 1 or not opts[0].get("state"):
   115	        problems.append("optimizer state is missing or CLEARED — a stripped file is not the "
   116	                        "checkpoint a restart would resume")
   117	    if not ck.get("lr_schedulers"):
   118	        problems.append("no 'lr_schedulers' — PL 2.1 would KeyError on resume from this file")
   119	    sd = ck.get("state_dict") or {}
   120	    if not any(str(k).startswith("diffusion_ema.") for k in sd):
   121	        problems.append("no EMA weights in state_dict")
   122	    return problems
   123	
   124	
   125	def main(argv=None):
   126	    ap = argparse.ArgumentParser(description="record an exp_11 arm's audited final checkpoint")
   127	    ap.add_argument("arm")
   128	    ap.add_argument("--step", type=int, default=None,
   129	                    help="cross-check only; must equal the arm's registered max_steps")
   130	    ap.add_argument("--registry", default=os.path.join(HERE, "arm_launch_registry.json"))
   131	    ap.add_argument("--launcher", default=os.path.join(HERE, "fa_orbit_train.sbatch"))
   132	    # HERE = <repo>/worklog/worklog_<user>/exp_11_fa_orbit_claude
   133	    ap.add_argument("--repo-root", default=os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
   134	    ap.add_argument("--dry-run", action="store_true", help="audit and report, publish nothing")
   135	    args = ap.parse_args(argv)
   136	
   137	    pins = read_pins(args.launcher)
   138	    if not pins.get("PINNED_RUNG"):
   139	        raise SystemExit(f"no PINNED_* values found in {args.launcher}")
   140	    store = os.path.dirname(os.path.abspath(args.registry)) or "."
   141	    lock_fd = os.open(store, os.O_RDONLY)     # the recorder's lock: one writer, one store
   142	    try:
   143	        fcntl.flock(lock_fd, fcntl.LOCK_EX)
   144	        return anchor(args, pins)
   145	    finally:
   146	        os.close(lock_fd)
   147	
   148	
   149	def anchor(args, pins):
   150	    arm = args.arm
   151	    reg = json.load(open(args.registry))
   152	    row = reg.get("arms", {}).get(arm)
   153	    if row is None:
   154	        raise SystemExit(f"{arm} has no INITIAL registry entry — an anchor belongs to a launch")
   155	
   156	    try:
   157	        step = int(row["max_steps"])
   158	    except (KeyError, TypeError, ValueError):
   159	        raise SystemExit(f"{arm}'s registry row has no integer max_steps to anchor at")
   160	    if args.step is not None and args.step != step:
   161	        raise SystemExit(f"--step {args.step} != {arm}'s registered budget {step}: the anchor is "
   162	                         "the run's FINAL checkpoint, not an arbitrary one")
   163	
   164	    problems, man = check_launch_chain(arm, row, pins, args.repo_root)
   165	    ckpt, more = find_final_ckpt(row, arm, step, args.repo_root)
   166	    problems += more
   167	    # Only load the checkpoint once the chain itself is sound: a broken chain has
   168	    # already refused the run, and a 724 MB read should not be spent proving it
   169	    # twice. (check_launch_chain has already flagged a missing/unhashable config,
   170	    # so reaching here means model_config exists and is the audited one.)
   171	    if ckpt and not problems:
   172	        problems += audit_ckpt(ckpt, step, man["model_config"])
   173	    if problems:
   174	        print("ANCHOR REFUSED:")
   175	        for p in problems:
   176	            print(f"  !! {p}")
   177	        return 2
   178	
   179	    digest = pm.sha256_file(ckpt)
   180	    old = row.get("final_ckpt_sha256")
   181	    if old and old != digest:
   182	        print("ANCHOR REFUSED:")
   183	        print(f"  !! {arm} is already anchored at {old[:12]} and this checkpoint hashes "
   184	              f"{digest[:12]} — re-anchoring would silently re-parent every leg already chained "
   185	              "to the old anchor")
   186	        return 2
   187	    if old == digest and str(row.get("final_step")) == str(step):
   188	        print(f"{arm} is already anchored at {digest[:12]} (step {step}) — unchanged")
   189	        return 0
   190	
   191	    row["final_ckpt_sha256"] = digest
   192	    row["final_step"] = step
   193	    row["final_ckpt_path"] = pm.rel_to(args.repo_root, ckpt)
   194	    row["anchored_at"] = pm._now()
   195	    if not args.dry_run:
   196	        pm.write_atomic(args.registry, reg)
   197	    print(f"anchored {arm} at step {step}: {digest}")
   198	    print(f"  {row['final_ckpt_path']} (launch job {row.get('job')}, commit "
   199	          f"{str(row.get('commit'))[:12]})"
   200	          + ("  [dry run, nothing written]" if args.dry_run else ""))
   201	    return 0
   202	
   203	
   204	if __name__ == "__main__":
   205	    sys.exit(main())
expect_cmd "extension refuses a manifest that drifted after registration" 2 "changed after it was registered" -- \
  "${EXT[@]}" --max-steps 100000

echo "--- G4. Round 5: CHUNKED legs — chain preflight, chunk recorder, gates ---"
# A fresh fixture set (G3's manifest was deliberately tampered above): the same
# INITIAL identity, plus the 42500 endpoint checkpoint a first chunk produces
# and the launcher manifest that chunk leg would leave behind.
CH_ROOT="${TMP}/chain"; CH_SAVE="${CH_ROOT}/exp11_C8"
CH_CKPT_DIR="${CH_SAVE}/FLAC_exp11_C8/exp11_C8/checkpoints"
mkdir -p "$CH_CKPT_DIR"
CHUNK_PIN_C8="$(awk -F'"' '/^PINNED_TIME_LIMIT_CHUNK_C8=/{print $2; exit}' "$LAUNCHER")"
$PY - "$TMP" "${EXPDIR}/FLAC_AR_BF_C8.json" "$CH_CKPT_DIR" "$CH_SAVE" "$LAUNCHER" "$CHUNK_PIN_C8" <<'PY'
import hashlib, json, os, re, sys, torch
tmp, cfg_path, ckpt_dir, save_dir, launcher, chunk_pin = sys.argv[1:7]
vae_sha = re.search(r'^PINNED_VAE_SHA256="([^"]*)"', open(launcher).read(), re.M).group(1)
cfg = json.load(open(cfg_path))
def ck(step, epoch):
    return {"global_step": step, "epoch": epoch, "model_config": cfg,
            "state_dict": {"diffusion.x": torch.zeros(1), "diffusion_ema.x": torch.zeros(1)},
            "optimizer_states": [{"state": {0: {"step": 1}}, "param_groups": [{"lr": 1e-5}]}],
            "lr_schedulers": [{"last_epoch": step}]}
p40 = os.path.join(ckpt_dir, "epoch=8-step=40000.ckpt"); torch.save(ck(40000, 8), p40)
p42 = os.path.join(ckpt_dir, "epoch=9-step=42500.ckpt"); torch.save(ck(42500, 9), p42)
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
cfg_sha = hashlib.sha256(open(cfg_path, "rb").read()).hexdigest()
man = os.path.join(tmp, "chain_launch_manifest.txt")
with open(man, "w") as fh:
    fh.write("job 3648695 host neu000 mode INITIAL launch_uuid ext-uuid-c8\n")
    fh.write("arm C8 rung 8x8 micro 8 ngpu 8 max_steps 40000 ckpt_every 2500\n")
    fh.write("commit " + "2" * 40 + "\n")
    fh.write(f"model_config {cfg_path}\nconfig_sha256 {cfg_sha}\nvae_sha256 {vae_sha}\n")
    fh.write(f"save_dir {save_dir}\nwandb_run_id exp11-C8-chain\n")
reg = {"arms": {"C8": {
    "manifest_path": man, "manifest_sha256": sha(man),
    "job": "3648695", "mode": "INITIAL", "launch_uuid": "ext-uuid-c8",
    "commit": "2" * 40, "rung": "8x8", "max_steps": "40000",
    "config_sha256": cfg_sha, "vae_sha256": vae_sha, "save_dir": save_dir,
    "training_seed": 42, "final_ckpt_sha256": sha(p40), "final_step": 40000}}, "restarts": {}}
json.dump(reg, open(os.path.join(tmp, "chain_registry.json"), "w"), indent=2)
# the launcher manifest a finished 40000->42500 chunk leg leaves behind, in four
# variants: the real one (with the producing job's ENDPOINT ATTESTATION, round-5
# B6), one with no attestation at all (a pre-B6 launcher, or a leg that never
# finished), one attesting `<none>` (the leg did not reach the success class),
# and one whose attested sha is not the file's.
body = ("job 3999001 host neu001 mode RESTART launch_uuid chunk-uuid-1\n"
        "arm C8 rung 8x8 micro 8 ngpu 8 max_steps 100000 ckpt_every 2500\n"
        "commit " + "3" * 40 + "\n"
        f"model_config {cfg_path}\nconfig_sha256 {cfg_sha}\nvae_sha256 {vae_sha}\n"
        f"save_dir {save_dir}\n"
        f"resume_ckpt {p40} expected_step 40000 resume_ckpt_sha256 {sha(p40)}\n"
        f"time_limit {chunk_pin}\nchunk_end 42500\n")
attest = f"endpoint_ckpt {p42} endpoint_step 42500 endpoint_sha256 {sha(p42)}\n"
open(os.path.join(tmp, "chunk_leg_manifest.txt"), "w").write(body + attest)
open(os.path.join(tmp, "chunk_leg_manifest_noattest.txt"), "w").write(body)
open(os.path.join(tmp, "chunk_leg_manifest_noneattest.txt"), "w").write(
    body + "endpoint_ckpt <none> endpoint_step 42500 endpoint_class 4\n")
open(os.path.join(tmp, "chunk_leg_manifest_badattest.txt"), "w").write(
    body + f"endpoint_ckpt {p42} endpoint_step 42500 endpoint_sha256 {'e' * 64}\n")
print("chain fixture written")
PY
CH_CKPT42="${CH_CKPT_DIR}/epoch=9-step=42500.ckpt"
chain_pf() {  # <registry> [extra preflight args...]
  local reg="$1"; shift
  $PY "$PREFLIGHT" --config "${EXPDIR}/FLAC_AR_BF_C8.json" --arm C8 --rung 8x8 \
    --max-steps 100000 --ckpt "$CH_CKPT42" --expected-step 42500 --commit "$HEAD_SHA" \
    --launch-manifest "${TMP}/chain_launch_manifest.txt" --chain --launch-registry "$reg" "$@"
}
expect_cmd "chain refuses an UNRECORDED predecessor (the fail-closed core)" 2 "no recorded chain link" -- \
  chain_pf "${TMP}/chain_registry.json"
recorder_with() { $PY "${EXPDIR}/fa_orbit_record_restart.py" C8 "$1" \
  --registry "${TMP}/chain_registry.json" --launcher "$LAUNCHER" --repo-root "$PWD" "${@:2}"; }
recorder() { recorder_with "${TMP}/chunk_leg_manifest.txt" "$@"; }
# B6: the manifest is published BEFORE training, so it proves intent, not
# authorship. Without the producing job's post-classification attestation, a
# FAILED leg's manifest plus a pre-existing 42500 checkpoint would mint a link.
# These three run BEFORE the successful record, so `mine` is empty and the
# attestation gate — not the immutability gate — is what refuses them.
expect_cmd "recorder refuses a manifest with NO endpoint attestation" 2 "no endpoint attestation" -- \
  recorder_with "${TMP}/chunk_leg_manifest_noattest.txt"
expect_cmd "recorder refuses a leg that attested it produced nothing" 2 "endpoint_ckpt <none>" -- \
  recorder_with "${TMP}/chunk_leg_manifest_noneattest.txt"
expect_cmd "recorder refuses an attested sha that is not the file's" 2 "attested endpoint_sha256" -- \
  recorder_with "${TMP}/chunk_leg_manifest_badattest.txt"
expect_cmd "recorder dry run validates but writes nothing" 0 "dry run, nothing written" -- recorder --dry-run
expect_cmd "chain still refuses after a dry run" 2 "no recorded chain link" -- \
  chain_pf "${TMP}/chain_registry.json"
expect_cmd "recorder records the finished chunk as a chain link" 0 "recorded C8 chunk link" -- recorder
expect_cmd "recorder is idempotent per job" 0 "already recorded" -- recorder
expect_cmd "chain accepts the recorded tip" 0 "restart lineage OK" -- chain_pf "${TMP}/chain_registry.json"
expect_cmd "chain + valid --chunk-end accepted" 0 "restart lineage OK" -- \
  chain_pf "${TMP}/chain_registry.json" --chunk-end 45000
expect_cmd "chain refuses a --chunk-end off the checkpoint cadence" 2 "not a multiple of 2500" -- \
  chain_pf "${TMP}/chain_registry.json" --chunk-end 43000
expect_cmd "chain refuses a --chunk-end at or below the resume step" 2 "chunk_end <= max_steps" -- \
  chain_pf "${TMP}/chain_registry.json" --chunk-end 42500
printf '# drift\n' >> "${TMP}/chunk_leg_manifest.txt"
expect_cmd "recorder refuses to rewrite a recorded link" 2 "immutable" -- recorder
$PY - "${TMP}/chain_registry.json" "${TMP}/chain_reg_badsha.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1])); r["arms"]["C8"]["chain"][-1]["final_ckpt_sha256"] = "d" * 64
json.dump(r, open(sys.argv[2], "w"), indent=2)
PY
expect_cmd "chain refuses a resume that is not the recorded tip's checkpoint" 2 "not the checkpoint that chunk produced" -- \
  chain_pf "${TMP}/chain_reg_badsha.json"
expect_cmd "chain refuses a non-tip resume step" 2 "resumes the TIP" -- \
  $PY "$PREFLIGHT" --config "${EXPDIR}/FLAC_AR_BF_C8.json" --arm C8 --rung 8x8 \
    --max-steps 100000 --ckpt "${CH_CKPT_DIR}/epoch=8-step=40000.ckpt" --expected-step 40000 \
    --commit "$HEAD_SHA" --launch-manifest "${TMP}/chain_launch_manifest.txt" --chain \
    --launch-registry "${TMP}/chain_registry.json"
# B6: validating only the TIP accepted a crafted registry — append a link whose
# final hash is whatever file you want to run and nothing ties it to the audited
# anchor. The WHOLE ancestry is checked now, and a break names its link index.
$PY - "${TMP}/chain_registry.json" "${TMP}/chain_reg_badanchor.json" \
     "${TMP}/chain_reg_badancestry.json" <<'PY'
import json, sys
src, badanchor, badancestry = sys.argv[1:4]
r = json.load(open(src))                      # link 0 does not resume the INITIAL anchor
r["arms"]["C8"]["chain"][0]["resume_ckpt_sha256"] = "e" * 64
json.dump(r, open(badanchor, "w"), indent=2)
r = json.load(open(src))                      # link 1 does not resume link 0's endpoint
link0 = r["arms"]["C8"]["chain"][0]
r["arms"]["C8"]["chain"].append({
    "job": "3999002", "launch_uuid": "chunk-uuid-2", "manifest_path": "/dev/null",
    "manifest_sha256": "0" * 64, "resume_step": link0["final_step"],
    "resume_ckpt_sha256": "e" * 64, "final_step": 45000, "final_ckpt_sha256": "f" * 64,
    "recorded_utc": "2026-08-13T00:00:00+00:00"})
json.dump(r, open(badancestry, "w"), indent=2)
PY
expect_cmd "chain refuses a first link that does not descend from the audited anchor" 2 "BROKEN at link 0" -- \
  chain_pf "${TMP}/chain_reg_badanchor.json"
expect_cmd "chain refuses a BROKEN ancestry (link 1 does not continue link 0)" 2 "BROKEN at link 1" -- \
  chain_pf "${TMP}/chain_reg_badancestry.json"
# launcher-side CHUNK_END gates (parameter gates run under DRYRUN)
case_run "CHUNK_END on an INITIAL launch refused" 2 "only a RESTART leg may be chunked" \
  -- DRYRUN=1 ARM=C4L "EXPECT_SHA=${HEAD_SHA}" "OUTPUT_ROOT=${OUT_ROOT}" "${REPO_ENV[@]}" CHUNK_END=42500
case_run "CHUNK_END under SMOKE refused" 2 "no meaning under SMOKE" \
  -- "${SMOKE_ENV[@]}" ARM=C4L CHUNK_END=42500 EXPECTED_STEP=40000
# submitter-side --chunk-end shape gates (refused before any pin/sbatch work)
expect_cmd "submitter: --chunk-end without --resume refused" 2 "valid only together" -- \
  env DRYRUN=1 bash "$SUBMITTER" C8 --chunk-end 42500
expect_cmd "submitter: --chunk-end off cadence refused" 2 "not a multiple of 2500" -- \
  env DRYRUN=1 bash "$SUBMITTER" C8 --resume x.ckpt --expected-step 40000 --chunk-end 42600
expect_cmd "submitter: --chunk-end must exceed the resume step" 2 "must exceed" -- \
  env DRYRUN=1 bash "$SUBMITTER" C8 --resume x.ckpt --expected-step 42500 --chunk-end 42500
expect_cmd "submitter: --chunk-end above the budget refused" 2 "exceeds the pinned budget" -- \
  env DRYRUN=1 bash "$SUBMITTER" C8 --resume x.ckpt --expected-step 40000 --chunk-end 102500
# watchdog argument safety + the no-checkpoint skip (ONESHOT, everything in TMP)
WD="${EXPDIR}/fa_orbit_chunk_watchdog.sh"
WD_TMP="${TMP}/wd"; mkdir -p "${WD_TMP}/outputs"
expect_cmd "watchdog rejects an off-cadence CHUNK" 2 "not a multiple" -- \
  bash "$WD" ONESHOT=1 CHUNK=2600
expect_cmd "watchdog rejects an off-cadence per-arm CHUNK_VANL" 2 "not a multiple" -- \
  bash "$WD" ONESHOT=1 CHUNK_VANL=2600
expect_cmd "watchdog rejects an unknown arm" 2 "not a comma-separated list" -- \
  bash "$WD" ONESHOT=1 ARMS=C4L,BOGUS
expect_cmd "watchdog rejects an unknown key" 2 "unknown argument" -- \
  bash "$WD" ONESHOT=1 SBATCH_EXTRA=x
# B2(a): a SECOND watchdog is a double-submission engine (both see "no live job",
# both submit the same boundary). Rather than race two watchdogs, hold the real
# lock with a background flock holder and prove the watchdog refuses to start.
WD_LOCK="${EXPDIR}/.chunk_watchdog.lock"
WD_RELEASE="${WD_TMP}/release_the_lock"      # the holder exits NORMALLY on this
( flock -n 9 || exit 1; while [ ! -e "$WD_RELEASE" ]; do sleep 0.2; done ) 9>"$WD_LOCK" &
WD_HOLDER=$!
sleep 0.5
expect_cmd "watchdog refuses a second concurrent instance" 2 "refusing to start a second instance" -- \
  bash "$WD" ONESHOT=1 DRYRUN=1 ARMS=C8 "OUTPUT_ROOT=${WD_TMP}/outputs" \
    "REGISTRY=${TMP}/chain_registry.json" "STATE=${WD_TMP}/state" "LOG=${WD_TMP}/log"
: > "$WD_RELEASE"
wait "$WD_HOLDER" 2>/dev/null
if flock -n 9 9>"$WD_LOCK" 2>/dev/null; then
  echo "PASS  the watchdog lock is free once the holder exits"; PASS=$((PASS+1))
else
  echo "FAIL  the watchdog lock is still held after the holder exited"; FAIL=$((FAIL+1))
fi
# B7: an arm whose registry row carries no audited anchor (VANL, live) must be
# FROZEN at startup — not submitted and then refused by the preflight. Anchoring
# is an operator action, so the watchdog names the tool instead of running it.
$PY - "${TMP}/chain_registry.json" "${TMP}/chain_reg_noanchor.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
r["arms"]["C8"].pop("final_ckpt_sha256", None); r["arms"]["C8"].pop("final_step", None)
json.dump(r, open(sys.argv[2], "w"), indent=2)
PY
expect_cmd "watchdog freezes an arm with no audited anchor before submitting" 0 "no audited anchor" -- \
  bash "$WD" ONESHOT=1 DRYRUN=1 ARMS=C8 "OUTPUT_ROOT=${WD_TMP}/outputs" \
    "REGISTRY=${TMP}/chain_reg_noanchor.json" "STATE=${WD_TMP}/state_noanchor" \
    "LOG=${WD_TMP}/log_noanchor"
# NOTE: this case runs against the REAL queue (no squeue fake): with a live
# exp11-C8-train job it logs "live job — nothing to do", without one it logs
# "nothing to resume, skipping" (the fixture OUTPUT_ROOT is empty). Either way
# a full ONESHOT pass must complete cleanly without submitting anything.
expect_cmd "watchdog completes a ONESHOT pass without submitting" 0 "ONESHOT: one pass complete" -- \
  bash "$WD" ONESHOT=1 DRYRUN=1 ARMS=C8 "OUTPUT_ROOT=${WD_TMP}/outputs" \
    "REGISTRY=${TMP}/chain_registry.json" "STATE=${WD_TMP}/state" "LOG=${WD_TMP}/log"
expect_cmd "watchdog defaults VANL to a 5000-step chunk (startup overhead)" 0 "chunk 2500 (VANL 5000)" -- \
  bash "$WD" ONESHOT=1 DRYRUN=1 ARMS=C8 "OUTPUT_ROOT=${WD_TMP}/outputs" \
    "REGISTRY=${TMP}/chain_registry.json" "STATE=${WD_TMP}/state" "LOG=${WD_TMP}/log"

echo "--- H. the submitter refuses un-pinned submission ---"
# RETIRED for the same reason as the launcher case above: all pins are concrete,
# so the submitter's placeholder refusal is unreachable on the real file.
if grep -qE '^PINNED_[A-Z_]+="TO-PIN-AFTER-P0"' "$SUBMITTER"; then
  echo "FAIL  a submitter pin is still the placeholder"; FAIL=$((FAIL+1))
else
  echo "PASS  every submitter pin holds a concrete value"; PASS=$((PASS+1))
fi
grep -q 'PLACEHOLDER="TO-PIN-AFTER-P0"' "$SUBMITTER" \
  && { echo "PASS  the submitter still refuses a placeholder pin if one returns"; PASS=$((PASS+1)); } \
  || { echo "FAIL  the submitter placeholder refusal is gone"; FAIL=$((FAIL+1)); }
expect_cmd "submitter rejects a bad arm" 2 "must be C4L" -- env DRYRUN=1 bash "$SUBMITTER" FA1

exec
/usr/bin/bash -lc "git diff -- worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submit.sh worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch | sed -n '1,420p'; git status --porcelain --untracked-files=no -- train.py defaults.ini src worklog/worklog_yixun/exp_11_fa_orbit_claude/*.json worklog/worklog_yixun/exp_11_fa_orbit_claude/*.py worklog/worklog_yixun/exp_11_fa_orbit_claude/*.sbatch worklog/worklog_yixun/exp_11_fa_orbit_claude/*.sh ':(exclude)worklog/worklog_yixun/exp_11_fa_orbit_claude/arm_launch_registry.json' worklog/worklog_yixun/exp_07_fa_scratch_claude/FLAC_AR_BF.json" in /n/fs/gatrdp/codespace/FLAC
 succeeded in 0ms:
diff --git a/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submit.sh b/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submit.sh
index 011f5bc..bde3f1c 100644
--- a/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submit.sh
+++ b/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submit.sh
@@ -9,9 +9,17 @@
 #
 #   ./fa_orbit_submit.sh C8
 #   ./fa_orbit_submit.sh C8 --resume <ckpt> --expected-step 12500
+#   ./fa_orbit_submit.sh C8 --resume <ckpt> --expected-step 40000 --chunk-end 42500
 #   SMOKE=1 SMOKE_RUNG=16x4 SMOKE_MIN_FREE_MB=14000 ./fa_orbit_submit.sh C4L
 #   DRYRUN=1 ./fa_orbit_submit.sh C8        # print the sbatch line, submit nothing
 #
+# CHUNKED legs (round 5). The partition never backfills a 34-160 h allocation, so
+# a leg may declare --chunk-end <n>: it trains to that boundary, saves, and exits,
+# and the next leg resumes from there. A chunk leg is walled by
+# PINNED_TIME_LIMIT_CHUNK_<ARM> (hours, not days), which is what makes it
+# schedulable. The chain is driven by fa_orbit_chunk_watchdog.sh, which submits
+# every chunk through THIS script — never through sbatch directly.
+#
 # Resources per rung (micro x N = 64): --gres=gpu:l40:N, --cpus-per-task=8+7N,
 # --mem=(12N+12)G, --time=<the arm's pinned limit>. Each submission is recorded
 # in an atomic, no-clobber manifest next to the launcher.
@@ -28,19 +36,35 @@ PLACEHOLDER="TO-PIN-AFTER-P0"
 [ -f "$SBATCH_FILE" ] || { echo "missing ${SBATCH_FILE} - abort"; exit 3; }
 
 ARM="${1:-}"
-[ -n "$ARM" ] || { echo "usage: $0 <C4L|C8|C16|C32|VANL> [--resume <ckpt> --expected-step <n>] - abort"; exit 2; }
+[ -n "$ARM" ] || { echo "usage: $0 <C4L|C8|C16|C32|VANL> [--resume <ckpt> --expected-step <n> [--chunk-end <n>]] - abort"; exit 2; }
 shift
 case "$ARM" in C4L|C8|C16|C32|VANL) ;; *) echo "ARM '${ARM}' must be C4L|C8|C16|C32|VANL - abort"; exit 2;; esac
 
-RESUME_CKPT=""; EXPECTED_STEP=0
+RESUME_CKPT=""; EXPECTED_STEP=0; CHUNK_END=""
 while [ $# -gt 0 ]; do
   case "$1" in
     --resume) RESUME_CKPT="${2:?--resume needs a path}"; shift 2 ;;
     --expected-step) EXPECTED_STEP="${2:?--expected-step needs a number}"; shift 2 ;;
+    --chunk-end) CHUNK_END="${2:?--chunk-end needs a number}"; shift 2 ;;
     *) echo "unknown argument '$1' - abort"; exit 2 ;;
   esac
 done
 case "$EXPECTED_STEP" in ''|*[!0-9]*) echo "--expected-step must be a non-negative integer - abort"; exit 2;; esac
+# --- --chunk-end: shape-checked HERE, and again inside the job ---------------
+# A chunk is meaningful only for a leg that resumes: it narrows where THIS job
+# stops, never what the campaign may reach (the budget pin stays 100000). It must
+# land on a saved checkpoint, or the next chunk has nothing to resume from.
+CHUNK_BUDGET="$(awk -F= '/^PINNED_MAXSTEPS=/{split($2,a," "); print a[1]; exit}' "$SBATCH_FILE")"
+case "$CHUNK_BUDGET" in ''|*[!0-9]*) echo "could not read PINNED_MAXSTEPS from ${SBATCH_FILE} - abort"; exit 3;; esac
+if [ -n "$CHUNK_END" ]; then
+  case "$CHUNK_END" in ''|*[!0-9]*) echo "--chunk-end must be a positive integer - abort"; exit 2;; esac
+  [ "$SMOKE" != "1" ] || { echo "--chunk-end is a production chunk-chain input and has no meaning under SMOKE=1 - abort"; exit 2; }
+  { [ -n "$RESUME_CKPT" ] && [ "$EXPECTED_STEP" -gt 0 ]; } \
+    || { echo "--chunk-end is valid only together with --resume/--expected-step - abort"; exit 2; }
+  [ "$((CHUNK_END % 2500))" -eq 0 ] || { echo "--chunk-end ${CHUNK_END} is not a multiple of 2500 (the pinned checkpoint cadence) - abort"; exit 2; }
+  [ "$CHUNK_END" -gt "$EXPECTED_STEP" ] || { echo "--chunk-end ${CHUNK_END} must exceed --expected-step ${EXPECTED_STEP} - abort"; exit 2; }
+  [ "$CHUNK_END" -le "$CHUNK_BUDGET" ] || { echo "--chunk-end ${CHUNK_END} exceeds the pinned budget ${CHUNK_BUDGET} - abort"; exit 2; }
+fi
 
 # --- pins are read FROM the launcher, so submitter and job cannot disagree ----
 pin() {  # read one PINNED_* value out of the launcher (quoted or bare)
@@ -59,11 +83,17 @@ else
   # A RESTART leg is a different budget from the INITIAL one: 60k further steps,
   # not 40k from scratch. Selecting the INITIAL limit for a restart would wall-kill
   # every arm partway through the extension.
-  if [ -n "${EXPECTED_STEP:-}" ] && [ "${EXPECTED_STEP:-0}" -gt 0 ]; then
+  # A CHUNK leg is shorter still: it stops at the next boundary, so it is walled
+  # by the arm's CHUNK pin (hours) rather than its whole-extension RESTART pin
+  # (days) — the whole point of chunking is an allocation the scheduler backfills.
+  if [ -n "$CHUNK_END" ]; then
+    TIME_LIMIT="$(pin "PINNED_TIME_LIMIT_CHUNK_${ARM}")"
+  elif [ -n "${EXPECTED_STEP:-}" ] && [ "${EXPECTED_STEP:-0}" -gt 0 ]; then
     TIME_LIMIT="$(pin "PINNED_TIME_LIMIT_RESTART_${ARM}")"
   else
     TIME_LIMIT="$(pin "PINNED_TIME_LIMIT_${ARM}")"
   fi
+  [ -n "$TIME_LIMIT" ] || { echo "the launcher carries no wall pin for this ${ARM} leg - abort"; exit 2; }
   for V in "$RUNG" "$TIME_LIMIT" "$(pin PINNED_MIN_FREE_MB)" "$(pin PINNED_P0_MANIFEST_SHA256)"; do
     [ "$V" != "$PLACEHOLDER" ] || { echo "the launcher still carries ${PLACEHOLDER} pins: the P0 report has not been pinned yet — no arm may be submitted (use SMOKE=1 for the smoke) - abort"; exit 2; }
   done
@@ -78,8 +108,16 @@ MB="${RUNG%x*}"; NGPU="${RUNG#*x}"
 # arms are running and Slurm appends to their tracked *.out logs continuously, so
 # a folder-wide check would abort every screen on a live-log write. Configs,
 # drivers and validators are still fully covered.
+#
+# arm_launch_registry.json is EXCLUDED (round-5 review B1). It is a lineage
+# RECORD written by the reviewed recorder as each chunk finishes — a measurement
+# *product*, not a measurement *surface* — and it is already outside the
+# launcher's commit-binding closure. Gating it deadlocks the chunk chain: the
+# recorder dirties it at 42500, so the very next submission (and every later
+# one, for every arm) is refused until a human commits. Keep it outside.
 DRIFT="$(git status --porcelain --untracked-files=no -- train.py defaults.ini src \
          "$EXPDIR"/*.json "$EXPDIR"/*.py "$EXPDIR"/*.sbatch "$EXPDIR"/*.sh \
+         ":(exclude)${EXPDIR}/arm_launch_registry.json" \
          worklog/worklog_yixun/exp_07_fa_scratch_claude/FLAC_AR_BF.json 2>/dev/null)"
 [ -z "$DRIFT" ] || { echo "tracked measurement surfaces have uncommitted changes - commit first, abort:"; echo "$DRIFT"; exit 2; }
 SHA="$(git rev-parse HEAD)"
@@ -94,9 +132,10 @@ ARGS=(
 )
 [ "$SMOKE" = "1" ] && ARGS[5]="${ARGS[5]},SMOKE=1,SMOKE_RUNG=${SMOKE_RUNG},SMOKE_MIN_FREE_MB=${SMOKE_MIN_FREE_MB},SMOKE_MAXSTEPS=${SMOKE_MAXSTEPS:-30},SMOKE_TIME=${TIME_LIMIT}"
 [ -n "$RESUME_CKPT" ] && ARGS[5]="${ARGS[5]},RESUME_CKPT=${RESUME_CKPT},EXPECTED_STEP=${EXPECTED_STEP}"
+[ -n "$CHUNK_END" ] && ARGS[5]="${ARGS[5]},CHUNK_END=${CHUNK_END}"
 ARGS+=("$SBATCH_FILE")
 
-echo "arm ${ARM} | rung ${RUNG} (${MB}x${NGPU}) | time ${TIME_LIMIT} | commit ${SHA} | smoke ${SMOKE}"
+echo "arm ${ARM} | rung ${RUNG} (${MB}x${NGPU}) | time ${TIME_LIMIT} | commit ${SHA} | smoke ${SMOKE} | chunk_end ${CHUNK_END:-<none>}"
 if [ "$DRYRUN" = "1" ]; then
   echo "DRYRUN sbatch ${ARGS[*]}"
   exit 0
@@ -120,6 +159,7 @@ TMP="$(mktemp "${MANIFEST}.XXXXXX")" || exit 3
   echo "commit ${SHA}"
   echo "pins rung=${RUNG} maxsteps=$(pin PINNED_MAXSTEPS) ckpt_every=$(pin PINNED_CHECKPOINT_EVERY) min_free_mb=$(pin PINNED_MIN_FREE_MB) p0_manifest_sha256=$(pin PINNED_P0_MANIFEST_SHA256)"
   echo "resume ${RESUME_CKPT:-<none>} expected_step ${EXPECTED_STEP}"
+  echo "chunk_end ${CHUNK_END:-<none>}"
   echo "sbatch sbatch ${ARGS[*]}"
 } >> "$TMP" || { echo "intent manifest write failed - abort"; exit 3; }
 mv -n "$TMP" "$MANIFEST" || { echo "intent manifest publication failed - abort"; exit 2; }
diff --git a/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch b/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch
index 3bfa2e7..e5db7da 100644
--- a/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch
+++ b/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch
@@ -23,6 +23,11 @@
 #                      (two writers commit to this checkout; worklog/record
 #                      commits must not kill a queued leg).
 #   RESUME_CKPT/EXPECTED_STEP   crash restart only (see LINEAGE)
+#   CHUNK_END          optional, RESTART only: stop THIS job at that step (a
+#                      multiple of the 2500 checkpoint cadence, <= the budget) and
+#                      let the next job resume from it. The budget pin does not
+#                      move; only where this leg stops does. Chunk legs are
+#                      chained by fa_orbit_chunk_watchdog.sh via the submitter.
 #   SMOKE=1            the reviewed multi-GPU smoke (see SMOKE MODE)
 # RUNG / MAXSTEPS / MIN_FREE_MB / time limit are NOT operator inputs any more.
 #
@@ -95,6 +100,17 @@ PINNED_TIME_LIMIT_RESTART_C8="51:00:00"     # 60k/0.4351 = 38.3 h
 PINNED_TIME_LIMIT_RESTART_C16="89:00:00"    # 60k/0.2454 = 67.9 h
 PINNED_TIME_LIMIT_RESTART_C32="160:00:00"   # 60k/0.1308 = 127.4 h (cap 168 h)
 PINNED_TIME_LIMIT_RESTART_VANL="19:00:00"   # 60k/1.0722 = 15.5 h
+# Round 5 CHUNKED legs: the partition never backfills a 34-160 h allocation, so a
+# leg trains to the next 2500-step boundary and exits. One chunk at the batched
+# rates is 2500/rate, rounded up with the same x1.3 margin plus a generous
+# startup allowance (cold start + torch/PL import + the 40k-checkpoint load are a
+# fixed cost per leg and now amortise over 2500 steps, not 60000):
+#   C4L 2500/0.6598 = 1.05 h | C8 1.60 h | C16 2.83 h | C32 5.31 h | VANL 0.65 h
+PINNED_TIME_LIMIT_CHUNK_C4L="03:00:00"
+PINNED_TIME_LIMIT_CHUNK_C8="04:00:00"
+PINNED_TIME_LIMIT_CHUNK_C16="06:00:00"
+PINNED_TIME_LIMIT_CHUNK_C32="09:00:00"
+PINNED_TIME_LIMIT_CHUNK_VANL="02:30:00"
 PINNED_P0_MANIFEST_SHA256="72607b922177208d56055d604b292d697b643ef3b7ab48261ab2e23a0cc2b53b"  # batched matrix manifest bd96575-…-a3ed28eb; spot manifest sha in the commit message
 # Environment pins (round-3 B6) — measured on the reviewed environment:
 PINNED_PYTHON="/n/fs/gatrdp/envs/flac/bin/python"
@@ -137,6 +153,10 @@ else
 fi
 RESUME_CKPT="${RESUME_CKPT:-}"
 EXPECTED_STEP="${EXPECTED_STEP:-0}"
+# CHUNK_END (round 5): the step THIS job stops at. Empty = the leg runs to the
+# budget, which is every pre-round-5 leg. It narrows train.py's --max-steps only;
+# PINNED_MAXSTEPS remains the campaign budget and its gate is untouched.
+CHUNK_END="${CHUNK_END:-}"
 TS="$(date '+%Y-%m-%d_%H-%M-%S')"
 
 die() { echo "$1"; exit "${2:-2}"; }
@@ -167,7 +187,12 @@ else
   # regardless and then rejected its own (correct) allocation in gate H — the
   # third hard-abort path the re-pin review found on jobs 3662828-30. The JOB now
   # selects the same pin the submitter did and enforces THAT one.
-  if [ "$EXPECTED_STEP" -gt 0 ]; then
+  # Round 5: a CHUNK leg is shorter again — it stops at the next boundary — so it
+  # carries the arm's CHUNK pin. Submitter and job select by the same rule, and
+  # gate H still enforces whichever pin was selected here.
+  if [ "$EXPECTED_STEP" -gt 0 ] && [ -n "$CHUNK_END" ]; then
+    TIME_PIN_NAME="PINNED_TIME_LIMIT_CHUNK_${ARM}"
+  elif [ "$EXPECTED_STEP" -gt 0 ]; then
     TIME_PIN_NAME="PINNED_TIME_LIMIT_RESTART_${ARM}"
   else
     TIME_PIN_NAME="PINNED_TIME_LIMIT_${ARM}"
@@ -193,6 +218,22 @@ if [ "$SMOKE" != "1" ]; then
   [ "$MB" = "$PINNED_MB" ] && [ "$NGPU" = "$PINNED_NGPU" ] || die "pin inconsistency: rung ${RUNG} vs PINNED_MB=${PINNED_MB}/PINNED_NGPU=${PINNED_NGPU} - abort"
   [ "$MAXSTEPS" = "100000" ] || die "PINNED_MAXSTEPS is ${MAXSTEPS}, the registered budget is 100000 - abort"
 fi
+
+# --- B2. CHUNK_END: where THIS job stops (round 5) ----------------------------
+# The budget gate above is untouched on purpose: CHUNK_END is the per-job stop,
+# PINNED_MAXSTEPS stays the campaign budget. STOPSTEP is what train.py is given.
+# Fail-closed on every edge, because a chunk that does not land ON a checkpoint
+# leaves the chain with nothing to resume.
+STOPSTEP="$MAXSTEPS"
+if [ -n "$CHUNK_END" ]; then
+  [ "$SMOKE" != "1" ] || die "CHUNK_END is a production chunk-chain input and has no meaning under SMOKE=1 (SMOKE_MAXSTEPS is the smoke's budget) - abort"
+  case "$CHUNK_END" in ''|*[!0-9]*) die "CHUNK_END '${CHUNK_END}' must be a positive integer - abort";; esac
+  [ "$EXPECTED_STEP" -gt 0 ] || die "CHUNK_END ${CHUNK_END} declares a chunk, but only a RESTART leg may be chunked (EXPECTED_STEP is 0) - abort"
+  [ "$CHUNK_END" -gt "$EXPECTED_STEP" ] || die "CHUNK_END ${CHUNK_END} must exceed the resume step ${EXPECTED_STEP} — this leg would produce no new checkpoint - abort"
+  [ "$CHUNK_END" -le "$MAXSTEPS" ] || die "CHUNK_END ${CHUNK_END} exceeds the budget ${MAXSTEPS} - abort"
+  [ "$((CHUNK_END % PINNED_CHECKPOINT_EVERY))" -eq 0 ] || die "CHUNK_END ${CHUNK_END} is not a multiple of the ${PINNED_CHECKPOINT_EVERY}-step checkpoint cadence: the chunk must end ON a checkpoint or the next chunk has nothing to resume - abort"
+  STOPSTEP="$CHUNK_END"
+fi
 RUNDIR="${SAVEDIR}/${NAME}/${EXPNAME}"
 echo "=== exp_11 arm ${ARM} @ rung ${RUNG} (MB ${MB} x ${NGPU} GPU, grad-ckpt ON) — ${TS} — host $(hostname) ==="
 
@@ -206,8 +247,14 @@ EXPREL="${EXPDIR#"$REPO"/}"; EXP07REL="${EXP07#"$REPO"/}"
 # git, not the shell, expands them — a tracked file deleted from the worktree
 # still matches (content-gate review B2) — data/AR (the split JSONs the
 # dataloader opens) is covered, and a failing git status is fail-closed.
+# arm_launch_registry.json is EXCLUDED (round-5 review B1): it is a lineage
+# RECORD the reviewed recorder writes as each chunk finishes, not a measurement
+# surface, and it is already outside the commit-binding closure below. Gating it
+# deadlocks the chunk chain — the recorder dirties it at 42500 and every later
+# leg, on every arm, would abort here until a human committed.
 DRIFT="$(git status --porcelain --untracked-files=no -- train.py defaults.ini src ":(exclude)src/tests" data/AR \
           "$EXPREL/*.json" "$EXPREL/*.py" "$EXPREL/*.sbatch" "$EXPREL/*.sh" \
+          ":(exclude)$EXPREL/arm_launch_registry.json" \
           "$EXP07REL/FLAC_AR_BF.json" 2>&1)" \
   || die "git status for the drift gate failed: ${DRIFT} - abort"
 # Commit binding is CONTENT-scoped: HEAD identity is sufficient but not
@@ -345,14 +392,14 @@ else
   esac
   [ "$MAXSTEPS" -gt "$EXPECTED_STEP" ] || die "MAXSTEPS ${MAXSTEPS} must exceed the resume step ${EXPECTED_STEP} - abort"
 fi
-echo "lineage: ${MODE} (expected_step ${EXPECTED_STEP}, max_steps ${MAXSTEPS}, ckpt every ${CHECKPOINT_EVERY}, time pin ${TIME_PIN_NAME}=${TIME_LIMIT})"
+echo "lineage: ${MODE} (expected_step ${EXPECTED_STEP}, max_steps ${MAXSTEPS}, chunk_end ${CHUNK_END:-<none>}, stops at ${STOPSTEP}, ckpt every ${CHECKPOINT_EVERY}, time pin ${TIME_PIN_NAME}=${TIME_LIMIT})"
 
 # --- F. the exact train.py argv ----------------------------------------------
 ARGV=(
   --model-config "$MODEL_CONFIG_ABS"
   --dataset-config src/configs/dataset_configs/AR/train/acousticroom_train.json
   --pretransform-ckpt-path weights/FLAC/VAE.safetensors
-  --max-steps "$MAXSTEPS" --batch-size "$MB" --accum-batches 1 --num-workers 6 --seed 42
+  --max-steps "$STOPSTEP" --batch-size "$MB" --accum-batches 1 --num-workers 6 --seed 42
   --num-gpus "$NGPU" --num-nodes 1
   --strategy ddp_find_unused_parameters_true --sync-batchnorm true --precision bf16-mixed
   --val-every -1 --val-dataset-config ''
@@ -530,8 +577,20 @@ if [ "$MODE" = "RESTART" ]; then
   # uuid, launch commit, config, save-dir, seed, and the 40k anchor itself)
   # without demanding that the INITIAL budget/commit equal this leg's. SMOKE
   # restarts have no registered launch and keep the ordinary contract.
-  [ "$SMOKE" != "1" ] && PRE_ARGS+=(--extension --launch-registry "$EXPDIR/arm_launch_registry.json"
-                                    --repo-root "$REPO")
+  #
+  # Round 5: only the FIRST leg off the anchor resumes the audited 40k checkpoint.
+  # A later chunk resumes the endpoint of the previous chunk, which is evidence
+  # only once fa_orbit_record_restart.py has written it as a chain link — so those
+  # legs bind with --chain, and an unrecorded predecessor refuses the launch.
+  if [ "$SMOKE" != "1" ]; then
+    if [ "$EXPECTED_STEP" -eq 40000 ]; then
+      PRE_ARGS+=(--extension)
+    else
+      PRE_ARGS+=(--chain)
+    fi
+    PRE_ARGS+=(--launch-registry "$EXPDIR/arm_launch_registry.json" --repo-root "$REPO")
+    [ -n "$CHUNK_END" ] && PRE_ARGS+=(--chunk-end "$CHUNK_END")
+  fi
   PRE_OUT="$(python3 "$EXPDIR/fa_orbit_ckpt_preflight.py" "${PRE_ARGS[@]}" 2>&1)"
   echo "$PRE_OUT"
   echo "$PRE_OUT" | grep -q "^CKPT_SHA256 " || die "restart preflight FAILED - abort"
@@ -640,6 +699,7 @@ rm -f "$PIPFREEZE_FILE"
   echo "gpu_uuids ${UUID_CSV}"
   echo "time_limit ${TIME_LIMIT} min_free_mb ${MIN_FREE_MB}"
   echo "resume_ckpt ${RESUME_CKPT:-<none>} expected_step ${EXPECTED_STEP} resume_ckpt_sha256 ${CKPT_SHA:-<none>}"
+  echo "chunk_end ${CHUNK_END:-<none>}"
   echo "save_dir ${SAVEDIR}"
   echo "slurm_transcript ${SLURM_OUT_AT_LAUNCH:-<unknown>} untrack ${UNTRACK_STATE}"
   echo "train_log ${TRAINLOG}"
@@ -707,12 +767,63 @@ if [ "$WANDB_CHECK_RC" -ne 0 ]; then
 fi
 
 # --- R. classification (tested taxonomy; see fa_orbit_classify.py) ------------
+# The completion literal Lightning prints is the stop step it was GIVEN, so the
+# classifier is told the step this leg was asked to reach. Without a chunk that
+# is MAXSTEPS, byte-identical to every pre-round-5 leg; with one it is the chunk
+# boundary, and a successful chunk classifies COMPLETE instead of NO-MARKER.
 CLASSIFY_OUT="$(python3 "$EXPDIR/fa_orbit_classify.py" --rc "$rc" --tee-rc "$tee_rc" --ngpu "$NGPU" \
-  --maxsteps "$MAXSTEPS" --log "$TRAINLOG" --log-copy "$SAVEDIR_LOG" 2>&1)"
+  --maxsteps "$STOPSTEP" --log "$TRAINLOG" --log-copy "$SAVEDIR_LOG" 2>&1)"
 final_rc=$?
 echo "$CLASSIFY_OUT"
 LAST_CKPT="$(ls -1t "${RUNDIR}/checkpoints"/*.ckpt 2>/dev/null | head -1)"
 
+# --- R2. ENDPOINT ATTESTATION: the job attests what it produced (round-5 B6) --
+# The manifest is published BEFORE training, so on its own it proves only what
+# this job intended. The recorder then had to associate it with whichever
+# exactly-named checkpoint happened to sit in the shared directory — a failed
+# job's manifest plus a pre-existing same-arm checkpoint could become a chain
+# link, and re-hashing proves current bytes, not authorship. So the job itself
+# appends, AFTER classification, what it produced: the endpoint path, its step,
+# and its sha256 as this job sees it on disk. fa_orbit_record_restart.py REFUSES
+# any chunk manifest without this attestation, and refuses one whose attested
+# sha does not match its own re-hash. A leg that did NOT reach the success class
+# attests `<none>`, which is likewise a refusal — a failed leg can never be
+# laundered into the chain by a checkpoint it did not write.
+ATTEST_LINE=""
+if [ -n "$CHUNK_END" ]; then
+  ENDPOINT_CKPT=""
+  if [ "$final_rc" -eq 0 ]; then
+    for CAND in "${RUNDIR}/checkpoints"/*-step="${CHUNK_END}".ckpt; do
+      [ -f "$CAND" ] || continue
+      if [ -n "$ENDPOINT_CKPT" ]; then ENDPOINT_CKPT="<ambiguous>"; break; fi
+      ENDPOINT_CKPT="$CAND"
+    done
+  fi
+  if [ "$final_rc" -ne 0 ]; then
+    ATTEST_LINE="endpoint_ckpt <none> endpoint_step ${CHUNK_END} endpoint_class ${final_rc}"
+    echo "endpoint attestation: this leg did NOT classify as complete (rc=${final_rc}) — attesting <none>"
+  elif [ -z "$ENDPOINT_CKPT" ] || [ "$ENDPOINT_CKPT" = "<ambiguous>" ]; then
+    ATTEST_LINE="endpoint_ckpt <none> endpoint_step ${CHUNK_END} endpoint_class ${final_rc}"
+    echo "endpoint attestation: no single *-step=${CHUNK_END}.ckpt under ${RUNDIR}/checkpoints (found '${ENDPOINT_CKPT:-none}') — attesting <none>"
+  else
+    ENDPOINT_SHA="$(sha256sum "$ENDPOINT_CKPT" | awk '{print $1}')"
+    if [ -n "$ENDPOINT_SHA" ]; then
+      ATTEST_LINE="endpoint_ckpt ${ENDPOINT_CKPT} endpoint_step ${CHUNK_END} endpoint_sha256 ${ENDPOINT_SHA}"
+      echo "endpoint attestation: ${ENDPOINT_CKPT} step ${CHUNK_END} sha256 ${ENDPOINT_SHA}"
+    else
+      ATTEST_LINE="endpoint_ckpt <none> endpoint_step ${CHUNK_END} endpoint_class ${final_rc}"
+      echo "endpoint attestation: sha256sum of ${ENDPOINT_CKPT} FAILED — attesting <none>"
+    fi
+  fi
+  # The manifest was written pre-run by THIS job; this is the same job appending
+  # to its own record. Both copies are kept identical.
+  if ! printf '%s\n' "$ATTEST_LINE" >> "$MANIFEST"; then
+    echo "ENDPOINT ATTESTATION COULD NOT BE APPENDED to ${MANIFEST} — the recorder will refuse this chunk (fail-closed); record it by hand only after auditing the checkpoint"
+  fi
+  printf '%s\n' "$ATTEST_LINE" >> "${SAVEDIR}/$(basename "$MANIFEST")" \
+    || echo "WARNING: could not append the endpoint attestation to the save-dir manifest copy"
+fi
+
 # --- S. durable final record (round-3 B5 residual) ---------------------------
 # The classifier verdict and this summary must survive in BOTH copies; the tee
 # status is captured (it used to be discarded) and a failure is class 7. The
@@ -723,7 +834,9 @@ FINAL_RECORD="$(
   echo "torchrun rc=${rc} tee rc=${tee_rc} wandb_identity_rc=${WANDB_CHECK_RC} classified rc=${final_rc}"
   echo "$CLASSIFY_OUT"
   echo "last checkpoint: ${LAST_CKPT:-<none>}"
+  [ -n "$CHUNK_END" ] && echo "endpoint attestation: ${ATTEST_LINE:-<not attempted>}"
   echo "restart: fa_orbit_submit.sh ${ARM} --resume ${LAST_CKPT:-<path>} --expected-step <step>"
+  [ -n "$CHUNK_END" ] && echo "next chunk: record this leg (fa_orbit_record_restart.py ${ARM} ${MANIFEST}), then fa_orbit_submit.sh ${ARM} --resume <ckpt at ${CHUNK_END}> --expected-step ${CHUNK_END} --chunk-end <${CHUNK_END}+${PINNED_CHECKPOINT_EVERY}>"
 )"
 printf '%s\n' "$FINAL_RECORD" | tee -a "$TRAINLOG" "$SAVEDIR_LOG" > /dev/null
 final_tee_rc="${PIPESTATUS[1]}"
 M worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py
 M worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_record_restart.py
 M worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submit.sh
 M worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch
 M worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh

exec
/usr/bin/bash -lc "grep -RIn --exclude='*.log' --exclude='*.out' -E 'ModelCheckpoint|enable_version_counter|filename=.*step|save_top_k|every_n_train_steps' train.py src defaults.ini | head -n 200; grep -RIn --exclude='*.log' --exclude='*.out' -E 'step=\\{|step=' src train.py | head -n 120; /n/fs/gatrdp/envs/flac/bin/python -c 'import inspect,pytorch_lightning as pl; from pytorch_lightning.callbacks import ModelCheckpoint; print(pl.__version__); print(inspect.signature(ModelCheckpoint.__init__))' 2>/dev/null" in /n/fs/gatrdp/codespace/FLAC
2026-08-13T16:51:29.540527Z ERROR codex_models_manager::manager: failed to renew cache TTL: missing field `base_instructions` at line 94 column 5
exec
/usr/bin/bash -lc "grep -RIn -E 'enable_version_counter|CHECKPOINT_JOIN_CHAR|_find_last_checkpoints' /n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/lightning/pytorch/callbacks/model_checkpoint.py /n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/pytorch_lightning/callbacks/model_checkpoint.py 2>/dev/null | head -n 120; sed -n '160,200p' train.py" in /n/fs/gatrdp/codespace/FLAC
 succeeded in 0ms:
/n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/pytorch_lightning/callbacks/model_checkpoint.py:95:            unless ``enable_version_counter`` is set to False.
/n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/pytorch_lightning/callbacks/model_checkpoint.py:133:        enable_version_counter: Whether to append a version to the existing file name.
/n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/pytorch_lightning/callbacks/model_checkpoint.py:139:        - ``CHECKPOINT_JOIN_CHAR = "-"``
/n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/pytorch_lightning/callbacks/model_checkpoint.py:205:    CHECKPOINT_JOIN_CHAR = "-"
/n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/pytorch_lightning/callbacks/model_checkpoint.py:226:        enable_version_counter: bool = True,
/n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/pytorch_lightning/callbacks/model_checkpoint.py:236:        self._enable_version_counter = enable_version_counter
/n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/pytorch_lightning/callbacks/model_checkpoint.py:519:            filename = "{epoch}" + cls.CHECKPOINT_JOIN_CHAR + "{step}"
/n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/pytorch_lightning/callbacks/model_checkpoint.py:542:            filename = cls.CHECKPOINT_JOIN_CHAR.join([prefix, filename])
/n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/pytorch_lightning/callbacks/model_checkpoint.py:582:            filename = self.CHECKPOINT_JOIN_CHAR.join((filename, f"v{ver}"))
/n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/pytorch_lightning/callbacks/model_checkpoint.py:617:    def _find_last_checkpoints(self, trainer: "pl.Trainer") -> Set[str]:
/n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/pytorch_lightning/callbacks/model_checkpoint.py:637:        if self._enable_version_counter:
/n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/pytorch_lightning/callbacks/model_checkpoint.py:661:        if self._enable_version_counter:
    training_wrapper = create_training_wrapper_from_config(model_config, model)

    exc_callback = ExceptionCallback()

    if args.logger == 'wandb':
        logger = pl.loggers.WandbLogger(project=args.name, name=args.experiment_name)
        logger.watch(training_wrapper)
    
        if args.save_dir and isinstance(logger.experiment.id, str):
            checkpoint_dir = os.path.join(args.save_dir, logger.experiment.project, logger.experiment.name, "checkpoints") 
        else:
            checkpoint_dir = None
    elif args.logger == 'comet':
        logger = pl.loggers.CometLogger(project_name=args.name)
        if args.save_dir and isinstance(logger.version, str):
            checkpoint_dir = os.path.join(args.save_dir, logger.name, logger.version, "checkpoints") 
        else:
            checkpoint_dir = args.save_dir if args.save_dir else None
    else:
        logger = None
        checkpoint_dir = args.save_dir if args.save_dir else None
        
    ckpt_callback = pl.callbacks.ModelCheckpoint(every_n_train_steps=args.checkpoint_every, dirpath=checkpoint_dir, save_top_k=-1)
    save_model_config_callback = ModelConfigEmbedderCallback(model_config)
        
    #Combine args and config dicts
    args_dict = vars(args)
    args_dict.update({"model_config": model_config})
    args_dict.update({"dataset_config": dataset_config})
    args_dict.update({"val_dataset_config": val_dataset_config})

    # Logger 
    if args.logger == 'wandb':
        push_wandb_config(logger, args_dict)
    elif args.logger == 'comet':
        logger.log_hyperparams(args_dict)

    #Set multi-GPU strategy if specified
    if args.strategy:
        if args.strategy == "deepspeed":
            from pytorch_lightning.strategies import DeepSpeedStrategy

2026-08-13T16:51:35.822938Z ERROR codex_models_manager::manager: failed to renew cache TTL: missing field `base_instructions` at line 94 column 5
exec
/usr/bin/bash -lc "nl -ba /n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/pytorch_lightning/callbacks/model_checkpoint.py | sed -n '210,245p'; nl -ba /n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/pytorch_lightning/callbacks/model_checkpoint.py | sed -n '560,595p'; nl -ba /n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/pytorch_lightning/callbacks/model_checkpoint.py | sed -n '625,670p'" in /n/fs/gatrdp/codespace/FLAC
 succeeded in 0ms:
   210	
   211	    def __init__(
   212	        self,
   213	        dirpath: Optional[_PATH] = None,
   214	        filename: Optional[str] = None,
   215	        monitor: Optional[str] = None,
   216	        verbose: bool = False,
   217	        save_last: Optional[bool] = None,
   218	        save_top_k: int = 1,
   219	        save_weights_only: bool = False,
   220	        mode: str = "min",
   221	        auto_insert_metric_name: bool = True,
   222	        every_n_train_steps: Optional[int] = None,
   223	        train_time_interval: Optional[timedelta] = None,
   224	        every_n_epochs: Optional[int] = None,
   225	        save_on_train_epoch_end: Optional[bool] = None,
   226	        enable_version_counter: bool = True,
   227	    ):
   228	        super().__init__()
   229	        self.monitor = monitor
   230	        self.verbose = verbose
   231	        self.save_last = save_last
   232	        self.save_top_k = save_top_k
   233	        self.save_weights_only = save_weights_only
   234	        self.auto_insert_metric_name = auto_insert_metric_name
   235	        self._save_on_train_epoch_end = save_on_train_epoch_end
   236	        self._enable_version_counter = enable_version_counter
   237	        self._last_global_step_saved = 0  # no need to save when no steps were taken
   238	        self._last_time_checked: Optional[float] = None
   239	        self.current_score: Optional[Tensor] = None
   240	        self.best_k_models: Dict[str, Tensor] = {}
   241	        self.kth_best_model_path = ""
   242	        self.best_model_score: Optional[Tensor] = None
   243	        self.best_model_path = ""
   244	        self.last_model_path = ""
   245	        self._last_checkpoint_saved = ""
   560	            >>> ckpt = ModelCheckpoint(dirpath=tmpdir, filename='{epoch}-{val_loss:.2f}')
   561	            >>> os.path.basename(ckpt.format_checkpoint_name(dict(epoch=2, val_loss=0.123456)))
   562	            'epoch=2-val_loss=0.12.ckpt'
   563	            >>> os.path.basename(ckpt.format_checkpoint_name(dict(epoch=2, val_loss=0.12), filename='{epoch:d}'))
   564	            'epoch=2.ckpt'
   565	            >>> ckpt = ModelCheckpoint(dirpath=tmpdir,
   566	            ... filename='epoch={epoch}-validation_loss={val_loss:.2f}',
   567	            ... auto_insert_metric_name=False)
   568	            >>> os.path.basename(ckpt.format_checkpoint_name(dict(epoch=2, val_loss=0.123456)))
   569	            'epoch=2-validation_loss=0.12.ckpt'
   570	            >>> ckpt = ModelCheckpoint(dirpath=tmpdir, filename='{missing:d}')
   571	            >>> os.path.basename(ckpt.format_checkpoint_name({}))
   572	            'missing=0.ckpt'
   573	            >>> ckpt = ModelCheckpoint(filename='{step}')
   574	            >>> os.path.basename(ckpt.format_checkpoint_name(dict(step=0)))
   575	            'step=0.ckpt'
   576	
   577	        """
   578	        filename = filename or self.filename
   579	        filename = self._format_checkpoint_name(filename, metrics, auto_insert_metric_name=self.auto_insert_metric_name)
   580	
   581	        if ver is not None:
   582	            filename = self.CHECKPOINT_JOIN_CHAR.join((filename, f"v{ver}"))
   583	
   584	        ckpt_name = f"{filename}{self.FILE_EXTENSION}"
   585	        return os.path.join(self.dirpath, ckpt_name) if self.dirpath else ckpt_name
   586	
   587	    def __resolve_ckpt_dir(self, trainer: "pl.Trainer") -> _PATH:
   588	        """Determines model checkpoint save directory at runtime. Reference attributes from the trainer's logger to
   589	        determine where to save checkpoints. The path for saving weights is set in this priority:
   590	
   591	        1.  The ``ModelCheckpoint``'s ``dirpath`` if passed in
   592	        2.  The ``Logger``'s ``log_dir`` if the trainer has loggers
   593	        3.  The ``Trainer``'s ``default_root_dir`` if the trainer has no loggers
   594	
   595	        The path gets extended with subdirectory "checkpoints".
   625	            }
   626	        return set()
   627	
   628	    def __warn_if_dir_not_empty(self, dirpath: _PATH) -> None:
   629	        if self.save_top_k != 0 and _is_dir(self._fs, dirpath, strict=True) and len(self._fs.ls(dirpath)) > 0:
   630	            rank_zero_warn(f"Checkpoint directory {dirpath} exists and is not empty.")
   631	
   632	    def _get_metric_interpolated_filepath_name(
   633	        self, monitor_candidates: Dict[str, Tensor], trainer: "pl.Trainer", del_filepath: Optional[str] = None
   634	    ) -> str:
   635	        filepath = self.format_checkpoint_name(monitor_candidates)
   636	
   637	        if self._enable_version_counter:
   638	            version_cnt = self.STARTING_VERSION
   639	            while self.file_exists(filepath, trainer) and filepath != del_filepath:
   640	                filepath = self.format_checkpoint_name(monitor_candidates, ver=version_cnt)
   641	                version_cnt += 1
   642	
   643	        return filepath
   644	
   645	    def _monitor_candidates(self, trainer: "pl.Trainer") -> Dict[str, Tensor]:
   646	        monitor_candidates = deepcopy(trainer.callback_metrics)
   647	        # cast to int if necessary because `self.log("epoch", 123)` will convert it to float. if it's not a tensor
   648	        # or does not exist we overwrite it as it's likely an error
   649	        epoch = monitor_candidates.get("epoch")
   650	        monitor_candidates["epoch"] = epoch.int() if isinstance(epoch, Tensor) else torch.tensor(trainer.current_epoch)
   651	        step = monitor_candidates.get("step")
   652	        monitor_candidates["step"] = step.int() if isinstance(step, Tensor) else torch.tensor(trainer.global_step)
   653	        return monitor_candidates
   654	
   655	    def _save_last_checkpoint(self, trainer: "pl.Trainer", monitor_candidates: Dict[str, Tensor]) -> None:
   656	        if not self.save_last:
   657	            return
   658	
   659	        filepath = self.format_checkpoint_name(monitor_candidates, self.CHECKPOINT_NAME_LAST)
   660	
   661	        if self._enable_version_counter:
   662	            version_cnt = self.STARTING_VERSION
   663	            while self.file_exists(filepath, trainer) and filepath != self.last_model_path:
   664	                filepath = self.format_checkpoint_name(monitor_candidates, self.CHECKPOINT_NAME_LAST, ver=version_cnt)
   665	                version_cnt += 1
   666	
   667	        # set the last model path before saving because it will be part of the state.
   668	        previous, self.last_model_path = self.last_model_path, filepath
   669	        if self._fs.protocol == "file" and self._last_checkpoint_saved and self.save_top_k != 0:
   670	            self._link_checkpoint(trainer, self._last_checkpoint_saved, filepath)

exec
/usr/bin/bash -lc "nl -ba worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_classify.py | sed -n '1,300p'; grep -nE 'state|terminal|CANCELLED|FAILED|COMPLETED|SPECIAL_EXIT|REVOKED' worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh | tail -n 100" in /n/fs/gatrdp/codespace/FLAC
 succeeded in 0ms:
     1	#!/usr/bin/env python3
     2	"""exp_11 arm-launch outcome classification (round-3 review B5).
     3	
     4	The launcher's exit taxonomy lives here, as a pure function of (torchrun status,
     5	tee status, the two log copies), so every class is unit-testable from fake logs
     6	instead of only from a multi-day run:
     7	
     8	    6  WORLD-SIZE      Lightning never reported the expected rank count, or
     9	                       reported a different one  (the run is not the recipe)
    10	    3  OOM             nonzero exit whose log carries a CUDA OOM
    11	    4  NO-MARKER       exit 0 without Lightning's exact completion literal
    12	                       (an early stop is not a finished budget)
    13	    7  LOG-PROVENANCE  tee failed, a log copy is missing, or the two copies are
    14	                       not byte-identical (the durable record is not durable)
    15	    rc otherwise       the raw torchrun status, preserved
    16	
    17	Precedence is world-size > OOM > no-marker > log-provenance > raw, so the most
    18	specific statement about WHY the run is unusable wins. Usage:
    19	
    20	    fa_orbit_classify.py --rc 0 --tee-rc 0 --ngpu 4 --maxsteps 40000 \\
    21	        --log <path> --log-copy <path>
    22	"""
    23	import argparse
    24	import filecmp
    25	import os
    26	import re
    27	import sys
    28	
    29	# Literals verified in the installed PL 2.1.0:
    30	#   lightning_fabric/utilities/distributed.py:296
    31	#   pytorch_lightning/loops/fit_loop.py:167
    32	WORLD_RE = re.compile(r"All distributed processes registered\. Starting with (\d+) processes")
    33	OOM_RE = re.compile(r"CUDA out of memory|OutOfMemoryError")
    34	
    35	EXIT_OOM, EXIT_NO_MARKER, EXIT_WORLD_SIZE, EXIT_LOG = 3, 4, 6, 7
    36	
    37	
    38	def _read(path):
    39	    try:
    40	        with open(path, "r", errors="replace") as fh:
    41	            return fh.read()
    42	    except OSError:
    43	        return None
    44	
    45	
    46	def classify(rc, tee_rc, ngpu, maxsteps, log_path, log_copy_path):
    47	    """Return ``(exit_code, [messages])``; never raises on missing files."""
    48	    msgs = []
    49	    text = _read(log_path)
    50	    if text is None:
    51	        return EXIT_LOG, [f"LOG-PROVENANCE: training log {log_path} is unreadable"]
    52	
    53	    world = [int(m) for m in WORLD_RE.findall(text)]
    54	    done = f"`Trainer.fit` stopped: `max_steps={maxsteps}` reached." in text
    55	    oom = bool(OOM_RE.search(text))
    56	
    57	    log_bad = []
    58	    if tee_rc:
    59	        log_bad.append(f"tee exited {tee_rc}")
    60	    for p in (log_path, log_copy_path):
    61	        if not os.path.isfile(p):
    62	            log_bad.append(f"missing log copy {p}")
    63	    if not log_bad and not filecmp.cmp(log_path, log_copy_path, shallow=False):
    64	        log_bad.append("the two log copies are not byte-identical")
    65	
    66	    if not world:
    67	        msgs.append(f"WORLD-SIZE: Lightning never reported rank registration "
    68	                    f"(expected {ngpu} processes) — this run did not train the recipe")
    69	        return EXIT_WORLD_SIZE, msgs
    70	    if any(w != ngpu for w in world):
    71	        msgs.append(f"WORLD-SIZE: reported {world}, expected {ngpu} processes")
    72	        return EXIT_WORLD_SIZE, msgs
    73	    msgs.append(f"world size OK: {ngpu} processes registered")
    74	
    75	    if rc != 0 and oom:
    76	        msgs.append(f"OOM: torchrun exited {rc} with a CUDA out-of-memory in the log")
    77	        return EXIT_OOM, msgs
    78	    if rc == 0 and not done:
    79	        msgs.append(f"NO-MARKER: exit 0 without `max_steps={maxsteps}` reached — "
    80	                    "an early stop, not a finished budget")
    81	        return EXIT_NO_MARKER, msgs
    82	    if log_bad:
    83	        msgs.append("LOG-PROVENANCE: " + "; ".join(log_bad))
    84	        return EXIT_LOG, msgs
    85	    if rc != 0:
    86	        msgs.append(f"RUNTIME: torchrun exited {rc} (no OOM signature)")
    87	        return rc, msgs
    88	    msgs.append(f"COMPLETE: {maxsteps} steps reached, dual logs verified identical")
    89	    return 0, msgs
    90	
    91	
    92	def main(argv=None):
    93	    ap = argparse.ArgumentParser(description="classify an exp_11 arm run")
    94	    ap.add_argument("--rc", type=int, required=True)
    95	    ap.add_argument("--tee-rc", type=int, default=0)
    96	    ap.add_argument("--ngpu", type=int, required=True)
    97	    ap.add_argument("--maxsteps", type=int, required=True)
    98	    ap.add_argument("--log", required=True)
    99	    ap.add_argument("--log-copy", required=True)
   100	    args = ap.parse_args(argv)
   101	    code, msgs = classify(args.rc, args.tee_rc, args.ngpu, args.maxsteps,
   102	                          args.log, args.log_copy)
   103	    for m in msgs:
   104	        print(m)
   105	    return code
   106	
   107	
   108	if __name__ == "__main__":
   109	    sys.exit(main())
83:# never fire. Replaced by the end state it was protecting, plus proof that the
215:  echo "FAIL  the suite changed tracked state:"; diff <(echo "$TRACKED_BEFORE") <(echo "$TRACKED_AFTER") | sed 's/^/        | /'
254:         "state_dict": {"diffusion.x": torch.zeros(1)},
255:         "optimizer_states": [{"state": {0: {"step": 1}} if opt else {},
259:        d["state_dict"]["diffusion_ema.x"] = torch.zeros(1)
276:expect_cmd "preflight rejects a stripped optimizer" 2 "optimizer state is CLEARED" -- "${PRE[@]}" --ckpt "${TMP}/stripped.ckpt" --expected-step 5000
348:      "state_dict": {"diffusion.x": torch.zeros(1), "diffusion_ema.x": torch.zeros(1)},
349:      "optimizer_states": [{"state": {0: {"step": 1}}, "param_groups": [{"lr": 1e-5}]}],
448:            "state_dict": {"diffusion.x": torch.zeros(1), "diffusion_ema.x": torch.zeros(1)},
449:            "optimizer_states": [{"state": {0: {"step": 1}}, "param_groups": [{"lr": 1e-5}]}],
504:# FAILED leg's manifest plus a pre-existing 42500 checkpoint would mint a link.
597:    "REGISTRY=${TMP}/chain_registry.json" "STATE=${WD_TMP}/state" "LOG=${WD_TMP}/log"
616:    "REGISTRY=${TMP}/chain_reg_noanchor.json" "STATE=${WD_TMP}/state_noanchor" \
624:    "REGISTRY=${TMP}/chain_registry.json" "STATE=${WD_TMP}/state" "LOG=${WD_TMP}/log"
627:    "REGISTRY=${TMP}/chain_registry.json" "STATE=${WD_TMP}/state" "LOG=${WD_TMP}/log"
680:grep -q 'OUTPUT_ROOT=outputs_FLAC' "$SUBMITTER" && { echo "PASS  submitter exports the fixed root, not ambient state"; PASS=$((PASS+1)); } \

exec
/usr/bin/bash -lc "tail -n 80 worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-13_12-42-29_guardtests.log; grep -nE '"'^PASS|''^FAIL|passed|failed|SUMMARY|guard'"' worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-13_12-42-29_guardtests.log | tail -n 30; stat -c '%y %s %n' worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-13_12-42-29_guardtests.log" in /n/fs/gatrdp/codespace/FLAC
 succeeded in 0ms:
PASS  recorder refuses a leg that attested it produced nothing  (rc=2)
PASS  recorder refuses an attested sha that is not the file's  (rc=2)
PASS  recorder dry run validates but writes nothing  (rc=0)
PASS  chain still refuses after a dry run  (rc=2)
PASS  recorder records the finished chunk as a chain link  (rc=0)
PASS  recorder is idempotent per job  (rc=0)
PASS  chain accepts the recorded tip  (rc=0)
PASS  chain + valid --chunk-end accepted  (rc=0)
PASS  chain refuses a --chunk-end off the checkpoint cadence  (rc=2)
PASS  chain refuses a --chunk-end at or below the resume step  (rc=2)
PASS  recorder refuses to rewrite a recorded link  (rc=2)
PASS  chain refuses a resume that is not the recorded tip's checkpoint  (rc=2)
PASS  chain refuses a non-tip resume step  (rc=2)
PASS  chain refuses a first link that does not descend from the audited anchor  (rc=2)
PASS  chain refuses a BROKEN ancestry (link 1 does not continue link 0)  (rc=2)
PASS  CHUNK_END on an INITIAL launch refused  (rc=2)
PASS  CHUNK_END under SMOKE refused  (rc=2)
PASS  submitter: --chunk-end without --resume refused  (rc=2)
PASS  submitter: --chunk-end off cadence refused  (rc=2)
PASS  submitter: --chunk-end must exceed the resume step  (rc=2)
PASS  submitter: --chunk-end above the budget refused  (rc=2)
PASS  watchdog rejects an off-cadence CHUNK  (rc=2)
PASS  watchdog rejects an off-cadence per-arm CHUNK_VANL  (rc=2)
PASS  watchdog rejects an unknown arm  (rc=2)
PASS  watchdog rejects an unknown key  (rc=2)
PASS  watchdog refuses a second concurrent instance  (rc=2)
PASS  the watchdog lock is free once the holder exits
PASS  watchdog freezes an arm with no audited anchor before submitting  (rc=0)
PASS  watchdog completes a ONESHOT pass without submitting  (rc=0)
PASS  watchdog defaults VANL to a 5000-step chunk (startup overhead)  (rc=0)
--- H. the submitter refuses un-pinned submission ---
PASS  every submitter pin holds a concrete value
PASS  the submitter still refuses a placeholder pin if one returns
PASS  submitter rejects a bad arm  (rc=2)
FAIL  submitter derives smoke flags: want rc=0 + '--gres=gpu:l40:4', got rc=2
        |  M worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py
        |  M worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_record_restart.py
        |  M worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submit.sh
        |  M worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch
        |  M worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh
FAIL  submitter derives cpus/mem from the rung: want rc=0 + '--cpus-per-task=36', got rc=2
        |  M worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py
        |  M worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_record_restart.py
        |  M worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submit.sh
        |  M worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch
        |  M worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh
FAIL  submitter derives 8x8 resources: want rc=0 + '--mem=108G', got rc=2
        |  M worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py
        |  M worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_record_restart.py
        |  M worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submit.sh
        |  M worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch
        |  M worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh
--- I. flock run ownership, both contention directions (round-3 B3 residual) ---
PASS  contender refused while the lock is held
PASS  lock free after the holder exits (no stale-recovery path needed)
PASS  launcher uses flock, not mkdir+stale recovery
PASS  no rmdir-based lock release remains
--- J. OUTPUT_ROOT is a literal inside a Slurm job (NEW-2) ---
PASS  ambient OUTPUT_ROOT rejected under Slurm  (rc=2)
PASS  launcher pins the production root literally
PASS  submitter exports the fixed root, not ambient state
--- K. the submitter publishes intent BEFORE sbatch (NEW-3) ---
FAIL  dry run publishes no submission manifest: want rc=0 + 'DRYRUN sbatch', got rc=2
        |  M worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py
        |  M worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_record_restart.py
        |  M worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submit.sh
        |  M worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch
        |  M worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh
PASS  a dry run leaves no submission manifest behind
PASS  intent manifest is published before the sbatch call
PASS  an unrecordable job is cancelled
--- L. FIFO and pip-freeze plumbing (NEW-4, B5 residual) ---
PASS  FIFO no longer uses mktemp -u
PASS  FIFO removal is in the exit trap
PASS  pip freeze status is checked before hashing
PASS  the final record's tee status is captured
PASS  the approved wandb entity is exported
PASS  the launcher runs the wandb readback and gates on its result

=== guard tests: 123 passed, 5 failed ===
118:PASS  watchdog completes a ONESHOT pass without submitting  (rc=0)
119:PASS  watchdog defaults VANL to a 5000-step chunk (startup overhead)  (rc=0)
121:PASS  every submitter pin holds a concrete value
122:PASS  the submitter still refuses a placeholder pin if one returns
123:PASS  submitter rejects a bad arm  (rc=2)
124:FAIL  submitter derives smoke flags: want rc=0 + '--gres=gpu:l40:4', got rc=2
129:        |  M worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh
130:FAIL  submitter derives cpus/mem from the rung: want rc=0 + '--cpus-per-task=36', got rc=2
135:        |  M worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh
136:FAIL  submitter derives 8x8 resources: want rc=0 + '--mem=108G', got rc=2
141:        |  M worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh
143:PASS  contender refused while the lock is held
144:PASS  lock free after the holder exits (no stale-recovery path needed)
145:PASS  launcher uses flock, not mkdir+stale recovery
146:PASS  no rmdir-based lock release remains
148:PASS  ambient OUTPUT_ROOT rejected under Slurm  (rc=2)
149:PASS  launcher pins the production root literally
150:PASS  submitter exports the fixed root, not ambient state
152:FAIL  dry run publishes no submission manifest: want rc=0 + 'DRYRUN sbatch', got rc=2
157:        |  M worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh
158:PASS  a dry run leaves no submission manifest behind
159:PASS  intent manifest is published before the sbatch call
160:PASS  an unrecordable job is cancelled
162:PASS  FIFO no longer uses mktemp -u
163:PASS  FIFO removal is in the exit trap
164:PASS  pip freeze status is checked before hashing
165:PASS  the final record's tee status is captured
166:PASS  the approved wandb entity is exported
167:PASS  the launcher runs the wandb readback and gates on its result
169:=== guard tests: 123 passed, 5 failed ===
2026-08-13 12:44:12.993517000 -0400 9462 worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-13_12-42-29_guardtests.log

codex
Two material residuals have emerged: the submission singleton is not shared with manual use of the sanctioned submitter, and the endpoint attestation can still bind the wrong bytes on same-boundary retries because Lightning versions colliding filenames while every kit glob ignores those versioned files. I’m checking the live VANL anchor and the exact ancestry trust boundary before finalizing severity.
exec
/usr/bin/bash -lc "sha256sum outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=8-step=40000.ckpt; stat -c '%s %y %n' outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=8-step=40000.ckpt outputs_FLAC/exp11_VANL/launch_manifest.txt; sha256sum outputs_FLAC/exp11_VANL/launch_manifest.txt worklog/worklog_yixun/exp_11_fa_orbit_claude/FLAC_AR_VANCKPT.json" in /n/fs/gatrdp/codespace/FLAC
exec
/usr/bin/bash -lc 'sha256sum outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=8-step=40000.ckpt' in /n/fs/gatrdp/codespace/FLAC
2026-08-13T16:54:03.637823Z ERROR codex_models_manager::manager: failed to renew cache TTL: missing field `base_instructions` at line 94 column 5
 succeeded in 23896ms:
1095f49330b4e7b9c469d69fdbaab1772586055236964b5e347604e712988507  outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=8-step=40000.ckpt

 succeeded in 58962ms:
1095f49330b4e7b9c469d69fdbaab1772586055236964b5e347604e712988507  outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=8-step=40000.ckpt
723922539 2026-08-09 23:24:06.241382000 -0400 outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=8-step=40000.ckpt
2472 2026-08-09 12:08:15.009186000 -0400 outputs_FLAC/exp11_VANL/launch_manifest.txt
113d06a284c6198cf9487e99a2efb7ccde94ae13e656a403fe2af0281d3de8b1  outputs_FLAC/exp11_VANL/launch_manifest.txt
733ca52b66c43538e1b9e603e979678af95ac05d89fd1d481ebb472a285a49d8  worklog/worklog_yixun/exp_11_fa_orbit_claude/FLAC_AR_VANCKPT.json

2026-08-13T16:54:16.834593Z ERROR codex_models_manager::manager: failed to renew cache TTL: missing field `base_instructions` at line 94 column 5
exec
/usr/bin/bash -lc "sha256sum outputs_FLAC/exp11_VANL/launch_manifest.txt worklog/worklog_yixun/exp_11_fa_orbit_claude/FLAC_AR_VANCKPT.json; stat -c '%s %y %n' outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=8-step=40000.ckpt outputs_FLAC/exp11_VANL/launch_manifest.txt" in /n/fs/gatrdp/codespace/FLAC
 succeeded in 0ms:
113d06a284c6198cf9487e99a2efb7ccde94ae13e656a403fe2af0281d3de8b1  outputs_FLAC/exp11_VANL/launch_manifest.txt
733ca52b66c43538e1b9e603e979678af95ac05d89fd1d481ebb472a285a49d8  worklog/worklog_yixun/exp_11_fa_orbit_claude/FLAC_AR_VANCKPT.json
723922539 2026-08-09 23:24:06.241382000 -0400 outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=8-step=40000.ckpt
2472 2026-08-09 12:08:15.009186000 -0400 outputs_FLAC/exp11_VANL/launch_manifest.txt

2026-08-13T16:54:21.998478Z ERROR codex_models_manager::manager: failed to renew cache TTL: missing field `base_instructions` at line 94 column 5
exec
/usr/bin/bash -lc "PYTHONDONTWRITEBYTECODE=1 /n/fs/gatrdp/envs/flac/bin/python -c \"import sys; sys.path.insert(0,'worklog/worklog_yixun/exp_11_fa_orbit_claude'); import fa_orbit_ckpt_preflight as p; a='a'*64; h='b'*64; reg={'final_ckpt_sha256':a,'final_step':40000}; valid=[{'job':'forged','resume_ckpt_sha256':a,'resume_step':40000,'final_ckpt_sha256':h,'final_step':42500}]; print('empty',p.check_chain_ancestry(reg,[])); print('single_structural_forgery',p.check_chain_ancestry(reg,valid))\"" in /n/fs/gatrdp/codespace/FLAC
 succeeded in 0ms:
empty []
single_structural_forgery []

2026-08-13T16:54:32.974307Z ERROR codex_models_manager::manager: failed to renew cache TTL: missing field `base_instructions` at line 94 column 5
exec
/usr/bin/bash -lc "sacct -X -n -P -j 3684153 -o State 2>/dev/null | sed -n '1,5l'; squeue -h -j 3684153 -o '%i %T' 2>/dev/null | sed -n '1,5l'" in /n/fs/gatrdp/codespace/FLAC
 succeeded in 0ms:
FAILED$

exec
/usr/bin/bash -lc "if [ -f worklog/worklog_yixun/exp_11_fa_orbit_claude/.chunk_watchdog_state ]; then nl -ba worklog/worklog_yixun/exp_11_fa_orbit_claude/.chunk_watchdog_state; else echo '<no state file>'; fi; ls -l worklog/worklog_yixun/exp_11_fa_orbit_claude/.chunk_watchdog.lock 2>/dev/null" in /n/fs/gatrdp/codespace/FLAC
 succeeded in 0ms:
<no state file>
-rw-r--r--. 1 yh4742 PU:SEA:clusters:ca15lab:members 0 Aug 13 12:44 worklog/worklog_yixun/exp_11_fa_orbit_claude/.chunk_watchdog.lock

 succeeded in 309399ms:
train.py:182:    ckpt_callback = pl.callbacks.ModelCheckpoint(every_n_train_steps=args.checkpoint_every, dirpath=checkpoint_dir, save_top_k=-1)
src/tests/test_retune_lr_state.py:110:        "callbacks": {"ModelCheckpoint{'monitor': None}": {"best_model_score": None}},
src/tests/test_strip_optimizer_state.py:57:        "callbacks": {"ModelCheckpoint{'monitor': None}": {"best_model_score": None}},
src/tests/test_finetune_cond.py:315:    Lightning cannot inject its default ModelCheckpoint) AND no ModelCheckpoint in
src/tests/test_finetune_cond.py:322:    assert not any(isinstance(cb, pl.callbacks.ModelCheckpoint) for cb in kw["callbacks"])
src/tests/test_finetune_cond.py:331:    assert any(isinstance(cb, pl.callbacks.ModelCheckpoint) for cb in kw["callbacks"])
defaults.ini:21:save_top_k = -1
src/tests/test_exp11_validate_rows.py:40:        "epoch=2-step=10000.ckpt")
src/tests/test_exp11_validate_rows.py:91:    name = name or "epoch=2-step=10000_metrics_1_1.0_exp11_C8_screen_S10000_s42_K8_fa_invariant_a8.json"
src/tests/test_exp11_validate_rows.py:201:        name="epoch=2-step=10000_metrics_1_1.0_exp11_C8_screen_S10000_s42_K1_fa_invariant_a8.json")
src/tests/test_exp11_validate_rows.py:207:    other = CKPT.replace("step=10000", "step=12500")
src/tests/test_exp11_validate_rows.py:236:def _seed_row(tmp_path, seed, arm="C8", step=10000, k=8, cell="conf"):
src/tests/test_exp11_validate_rows.py:241:    name = f"epoch=2-step={step}_metrics_1_1.0_{ev}_fa_invariant_a{n_ang}.json"
src/tests/test_exp11_validate_rows.py:243:          f"epoch=2-step={step}.ckpt")
src/tests/test_exp11_validate_rows.py:248:        side=_sidecar(seed=seed, eval_name=ev, arm=arm, step=step, K=k,
src/tests/test_exp11_validate_rows.py:255:    rows, problems = V.validate_cell(paths, arm="C8", step=10000, k=8, contract="table")
src/tests/test_exp11_validate_rows.py:262:    _rows, problems = V.validate_cell(paths, arm="C8", step=10000, k=8, contract="table")
src/tests/test_exp11_validate_rows.py:268:    dup = _seed_row(tmp_path, 46, step=10000)   # same (arm, step, seed) twice
src/tests/test_exp11_validate_rows.py:269:    _rows, problems = V.validate_cell(paths + [dup], arm="C8", step=10000, k=8, contract="table")
src/tests/test_exp11_validate_rows.py:276:                                     step=10000, k=8, contract="futility")
src/tests/test_exp11_validate_rows.py:282:    _rows, problems = V.validate_cell(paths, arm="C8", step=10000, k=8, contract="table")
src/tests/test_exp11_validate_rows.py:333:    path = tmp_path / ("epoch=2-step=10000_metrics_1_1.0_exp11_C8_screen_S10000_s42_K8"
src/tests/test_exp11_validate_rows.py:343:    path = tmp_path / ("epoch=2-step=10000_metrics_1_1.0_exp11_C8_screen_S10000_s42_K8"
src/tests/test_exp11_validate_rows.py:360:    bad = ("epoch=2-step=10000_metrics_1_1.0_exp11_C8_screen_S10000_s42_K8"
src/tests/test_exp11_validate_rows.py:415:                     name="epoch=2-step=10000_metrics_1_1.0_exp11_C8_screen_S10000_s42_K8_fa_invariant_a8.json")
src/tests/test_exp11_validate_rows.py:434:    _rows, problems = V.validate_cell(paths, arm="C8", step=10000, k=8, contract="table")
src/tests/test_exp11_validate_rows.py:440:    _rows, problems = V.validate_cell(paths, arm="C8", step=10000, k=8, contract="futility")
src/tests/test_exp11_validate_rows.py:446:    _rows, problems = V.validate_cell(paths, arm="C8", step=10000, k=8, contract="table")
src/tests/test_exp11_validate_rows.py:457:        name="epoch=2-step=10000_metrics_1_1.0_exp11_C8_conf_S10000_s46_K8_fa_invariant_a8.json")
src/tests/test_exp11_validate_rows.py:458:    _rows, problems = V.validate_cell(paths + [odd], arm="C8", step=10000, k=8, contract="table")
src/tests/test_exp11_validate_rows.py:468:        name="epoch=2-step=10000_metrics_1_1.0_exp11_C8_conf_S10000_s46_K8_fa_invariant_a8.json")
src/tests/test_exp11_validate_rows.py:469:    _rows, problems = V.validate_cell(paths + [odd], arm="C8", step=10000, k=8, contract="table")
src/tests/test_exp11_validate_rows.py:553:def _r3_row(tmp_path, rot, arm="C32", step=40000, k=8):
src/tests/test_exp11_validate_rows.py:561:    name = f"epoch=8-step={step}_metrics_1_1.0_{ev}_fa_invariant_a{n_ang}{suffix}.json"
src/tests/test_exp11_validate_rows.py:562:    ck = f"outputs_FLAC/exp11_{arm}/FLAC_exp11_{arm}/exp11_{arm}/checkpoints/epoch=8-step={step}.ckpt"
src/tests/test_exp11_validate_rows.py:567:        side=_sidecar(arm=arm, step=step, K=k, eval_name=ev, frame_avg_angles=ang, ckpt_path=ck),
src/tests/test_exp11_validate_rows.py:573:    rows, problems = V.validate_cell(paths, arm="C32", step=40000, k=8, contract="r3")
src/tests/test_exp11_validate_rows.py:578:    _rows, problems = V.validate_cell(short, arm="C32", step=40000, k=8, contract="r3")
src/tests/test_exp11_validate_rows.py:586:    _rows, problems = V.validate_cell(paths, arm="C32", step=40000, k=8, contract="r3")
src/tests/test_exp11_validate_rows.py:607:    ck = ck_dir / "epoch=2-step=10000.ckpt"
src/tests/test_exp11_validate_rows.py:628:                       name="epoch=2-step=10000_metrics_1_1.0_exp11_C8_screen_S10000_s42_K8_fa_invariant_a8.json")
src/tests/test_exp11_validate_rows.py:637:def _cross_row(tmp_path, eval_orbit, arm="C8", step=40000, k=8):
src/tests/test_exp11_validate_rows.py:640:    name = f"epoch=8-step={step}_metrics_1_1.0_{ev}_fa_invariant_a{eval_orbit}.json"
src/tests/test_exp11_validate_rows.py:642:          f"epoch=8-step={step}.ckpt")
src/tests/test_exp11_validate_rows.py:647:        side=_sidecar(arm=arm, step=step, K=k, eval_name=ev, frame_avg_angles=ang,
src/tests/test_exp11_validate_rows.py:663:    rows, problems = V.validate_cell(paths, arm="C8", step=40000, k=8, contract="cross")
src/tests/test_exp11_validate_rows.py:669:    _rows, problems = V.validate_cell(short, arm="C8", step=40000, k=8, contract="cross")
src/tests/test_exp11_validate_rows.py:684:    name = f"epoch=8-step={step}_metrics_1_1.0_{ev}_fa_invariant_a{eval_orbit}.json"
src/tests/test_exp11_validate_rows.py:685:    ck = f"outputs_FLAC/exp11_{arm}/FLAC_exp11_{arm}/exp11_{arm}/checkpoints/epoch=8-step={step}.ckpt"
src/tests/test_exp11_validate_rows.py:687:    side = _sidecar(arm=arm, step=step, K=8, eval_name=ev, frame_avg_angles=ang, ckpt_path=ck)
src/tests/test_exp11_validate_rows.py:699:    name = f"epoch=8-step={step}_metrics_1_1.0_{ev}_fa_invariant_a16.json"
src/tests/test_exp11_validate_rows.py:700:    ck = f"outputs_FLAC/exp11_{arm}/FLAC_exp11_{arm}/exp11_{arm}/checkpoints/epoch=8-step={step}.ckpt"
src/tests/test_exp11_validate_rows.py:703:                      side=_sidecar(arm=arm, step=step, K=8, eval_name=ev, frame_avg_angles=ang4,
src/tests/test_exp11_validate_rows.py:711:    paths = [_cross_row(tmp_path, n, step=40000) for n in (4, 16, 32)]
src/tests/test_exp11_validate_rows.py:712:    _rows, problems = V.validate_cell(paths, arm="C8", step=30000, k=8, contract="cross")
src/tests/test_exp11_validate_rows.py:721:    name = f"epoch=8-step={step}_metrics_1_1.0_{ev}_fa_invariant_a32_rot11p25.json"
src/tests/test_exp11_validate_rows.py:722:    ck = f"outputs_FLAC/exp11_{arm}/FLAC_exp11_{arm}/exp11_{arm}/checkpoints/epoch=8-step={step}.ckpt"
src/tests/test_exp11_validate_rows.py:726:                      side=_sidecar(arm=arm, step=step, K=8, eval_name=ev, frame_avg_angles=ang,
src/tests/test_exp11_validate_rows.py:747:def _vanl_row(tmp_path, seed=42, step=40000, k=8, cell="conf", **over):
src/tests/test_exp11_validate_rows.py:749:    name = f"epoch=8-step={step}_metrics_1_1.0_{ev}.json"      # no _fa_invariant_aN
src/tests/test_exp11_validate_rows.py:750:    ck = f"outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=8-step={step}.ckpt"
src/tests/test_exp11_validate_rows.py:759:    side = _sidecar(arm="VANL", step=step, K=k, seed=seed, eval_name=ev, ckpt_path=ck,
src/tests/test_exp11_validate_rows.py:809:    rows, problems = V.validate_cell(paths, arm="VANL", step=40000, k=8, contract="table")
src/tests/test_exp11_validate_rows.py:866:    _rows, problems = V.validate_cell(paths, arm="C8", step=40000, k=8, contract="q9")
src/tests/test_exp11_validate_rows.py:872:    rows, problems = V.validate_cell(paths, arm="VANL", step=40000, k=8, contract="q9")
src/tests/test_exp11_validate_rows.py:889:    name = f"epoch=2-step=10000_metrics_1_1.0_{ev}_fa_invariant_a8.json"
src/tests/test_exp11_validate_rows.py:890:    ck = "outputs_FLAC/exp11_C8/FLAC_exp11_C8/exp11_C8/checkpoints/epoch=2-step=10000.ckpt"
src/tests/test_exp11_validate_rows.py:895:                      side=_sidecar(arm="C8", step=10000, K=1, eval_name=ev,
src/tests/test_exp11_validate_rows.py:923:        _rows, problems = V.validate_cell([], arm="C8", step=step, k=8, contract="traj")
src/tests/test_exp11_validate_rows.py:931:        _rows, problems = V.validate_cell([], arm="C8", step=step, k=8, contract="traj")
src/tests/test_exp11_validate_rows.py:933:    _rows, problems = V.validate_cell([], arm="C8", step=42500, k=8, contract="traj")
src/tests/test_exp11_restart_record.py:93:        p = os.path.join(ckpt_dir, f"epoch={step // 5000}-step={step}.ckpt")
src/tests/test_exp11_restart_record.py:174:    gone = os.path.join(world["ckpt_dir"], "epoch=8-step=40000.ckpt")
src/tests/test_exp11_restart_record.py:191:    stray = write(os.path.join(world["root"], "elsewhere", "epoch=8-step=40000.ckpt"),
src/tests/test_exp11_restart_record.py:278:    new = os.path.join(world["ckpt_dir"], "epoch=9-step=47500.ckpt")
src/tests/test_exp11_restart_record.py:295:    write(os.path.join(world["ckpt_dir"], "epoch=30-step=102500.ckpt"), "beyond the budget\n")
src/tests/test_exp11_restart_record.py:329:    stray = write(os.path.join(world["ckpt_dir"], "epoch=20-step=60000.ckpt"), "unpublished\n")
src/tests/test_exp11_restart_record.py:336:    copy = write(os.path.join(world["root"], "elsewhere", "epoch=9-step=45000.ckpt"),
src/tests/test_exp11_restart_record.py:409:def _ckpt_blob(cfg, step=40000, opt=True, sched=True, ema=True):
src/tests/test_exp11_restart_record.py:465:    assert row["final_ckpt_path"].endswith("epoch=8-step=40000.ckpt")
src/tests/test_exp11_restart_record.py:518:    torch.save(_ckpt_blob(cfg), os.path.join(unanchored["ckpt_dir"], "epoch=9-step=40000.ckpt"))
src/tests/test_yaw_aug_record_control.py:50:    def _make(global_step=40000, with_ema=True, model_config="control",
src/tests/test_yaw_aug_record_control.py:68:        path = tmp_path / f"epoch=8-step={global_step}.ckpt"
src/tests/test_yaw_aug_record_control.py:116:    rc.build_record(synthetic_ckpt(), CONTROL_CONFIG, expect_step=40000)
src/tests/test_yaw_aug_record_control.py:136:        rc.build_record(corrupt, CONTROL_CONFIG, expect_step=40000)
src/tests/test_yaw_aug_record_control.py:145:    other = synthetic_ckpt(global_step=37500)
src/tests/test_yaw_aug_record_control.py:154:        rc.build_record(ckpt, CONTROL_CONFIG, expect_step=40000)
src/tests/test_yaw_aug_record_control.py:170:        rc.build_record(ckpt, CONTROL_CONFIG, expect_step=40000)
src/tests/test_yaw_aug_record_control.py:184:    rc.build_record(synthetic_ckpt(), CONTROL_CONFIG, expect_step=40000)
src/tests/test_yaw_aug_record_control.py:193:    record = rc.build_record(ckpt, CONTROL_CONFIG, expect_step=40000)
src/tests/test_yaw_aug_record_control.py:236:    record = rc.build_record(synthetic_ckpt(), CONTROL_CONFIG, expect_step=40000)
src/tests/test_yaw_aug_record_control.py:251:                        expect_step=40000)
src/tests/test_yaw_aug_record_control.py:257:                        expect_step=40000)
src/tests/test_yaw_aug_record_control.py:297:        rc.build_record(ckpt, CONTROL_CONFIG, expect_step=40000)
src/tests/test_yaw_aug_record_control.py:301:    record = rc.build_record(synthetic_ckpt(), CONTROL_CONFIG, expect_step=40000)
src/tests/test_yaw_aug_record_control.py:309:    ckpt = synthetic_ckpt(global_step=40000)
src/tests/test_yaw_aug_record_control.py:314:        rc.build_record(ckpt, CONTROL_CONFIG, expect_step=40000)
src/tests/test_yaw_aug_record_control.py:322:        rc.build_record(ckpt, CONTROL_CONFIG, expect_step=40000), tmp_path / "rec.json"
src/tests/test_yaw_aug_record_control.py:334:    record = rc.build_record(synthetic_ckpt(), CONTROL_CONFIG, expect_step=40000)
src/tests/test_yaw_aug_record_control.py:349:    record = rc.build_record(synthetic_ckpt(), CONTROL_CONFIG, expect_step=40000)
src/tests/test_yaw_aug_record_control.py:359:    record = rc.build_record(synthetic_ckpt(), CONTROL_CONFIG, expect_step=40000)
src/tests/test_yaw_aug_record_control.py:375:    ckpt = synthetic_ckpt(global_step=37500)
src/tests/test_yaw_aug_record_control.py:377:        rc.build_record(ckpt, CONTROL_CONFIG, expect_step=40000)
src/tests/test_yaw_aug_record_control.py:384:        rc.build_record(synthetic_ckpt(), impostor, expect_step=40000)
src/tests/test_yaw_aug_record_control.py:390:        rc.build_record(ckpt, CONTROL_CONFIG, expect_step=40000)
src/tests/test_yaw_aug_record_control.py:427:        rc.build_record(ckpt, CONTROL_CONFIG, expect_step=40000)
src/tests/test_yaw_aug_record_control.py:431:    a = rc.build_record(synthetic_ckpt(), CONTROL_CONFIG, expect_step=40000)
src/tests/test_yaw_aug_record_control.py:432:    b = rc.build_record(synthetic_ckpt(), CONTROL_CONFIG, expect_step=40000)
src/tests/test_yaw_aug_record_control.py:447:        rc.build_record(synthetic_ckpt(global_step=1), CONTROL_CONFIG, expect_step=40000)
src/tests/test_yaw_aug_record_control.py:491:        "outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=8-step=40000.ckpt"
src/tests/test_yaw_gen_collect.py:158:    ckpt = os.path.join(ck_dir, f"epoch=8-step={int(cell.step)}.ckpt")
src/tests/test_yaw_gen_collect.py:1388:def _exp11_conf_rows(root, arm, k, value, seeds=V.SEEDS, step=V.STEP):
src/tests/test_yaw_gen_collect.py:1396:        name = (f"epoch=8-step={step}_metrics_1_1.0_exp11_{arm}_conf_S{step}"
src/tests/test_gen_model_comparison_gate.py:95:def _write_valid_cell(tmp_path, arm="C8", step=40000, k=8, seeds=(42, 43, 44, 45, 46),
src/tests/test_gen_model_comparison_gate.py:115:    ck_file = ck_dir / f"epoch=8-step={step}.ckpt"
src/tests/test_gen_model_comparison_gate.py:130:        name = f"epoch=8-step={step}_metrics_1_1.0_{ev}{suffix}.json"
src/tests/test_gen_model_comparison_gate.py:358:def _tree_with_exp11_evidence(tmp_path, k=8, arm="C8", step=40000, seeds=(42, 43, 44, 45, 46)):
src/tests/test_gen_model_comparison_gate.py:375:    paths = _write_valid_cell(ev_dir, arm=arm, step=step, k=k, seeds=seeds)
src/tests/test_gen_model_comparison_gate.py:422:    _write_valid_cell(root / "outputs_FLAC" / "exp11_C8", arm="C8", step=40000, k=1)
src/tests/test_gen_model_comparison_gate.py:582:    return body.replace("*", "epoch=8-step=40000_metrics_1_1.0_", 1).replace("*", "")
src/tests/test_gen_model_comparison_gate.py:628:    return _write_valid_cell(ev_root, arm=arm, step=40000, k=k, seeds=seeds,
src/tests/test_exp14_fixed_mode_snapshot.py:208:    "ckpt_path": "/o/epoch=8-step=40000.ckpt",
src/tests/test_exp14_fixed_mode_snapshot.py:231:        payload, "/o/epoch=8-step=40000.ckpt", 0.0, "vanilla", None, by_scene=by_scene)
src/tests/test_eval_paths.py:613:    pattern of outputs_FLAC/ft_vanilla/epoch=0-step=2000.ckpt) pass; a dropped
src/tests/test_yaw_aug_training.py:121:def _attach_stub_trainer(wrapper, global_step=0, global_rank=0):
src/tests/test_yaw_aug_training.py:130:        global_step=global_step,
src/tests/test_yaw_aug_training.py:214:    _attach_stub_trainer(wrapper, global_step=7, global_rank=0)
2.1.0
(self, dirpath: Union[str, pathlib.Path, NoneType] = None, filename: Optional[str] = None, monitor: Optional[str] = None, verbose: bool = False, save_last: Optional[bool] = None, save_top_k: int = 1, save_weights_only: bool = False, mode: str = 'min', auto_insert_metric_name: bool = True, every_n_train_steps: Optional[int] = None, train_time_interval: Optional[datetime.timedelta] = None, every_n_epochs: Optional[int] = None, save_on_train_epoch_end: Optional[bool] = None, enable_version_counter: bool = True)

codex
## BLOCKING

1. **B2 remains unresolved for manual/concurrent submission.** The singleton lock is held only by the watchdog ([fa_orbit_chunk_watchdog.sh:140](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_chunk_watchdog.sh:140)); the sanctioned submitter does not acquire it before `sbatch` ([fa_orbit_submit.sh:169](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submit.sh:169)). A manual invocation can still race the watchdog’s `squeue`→submit interval and queue the same boundary twice. The job-side flock only serializes execution; a duplicate can start immediately after the first releases it.

   The advertised by-ID recheck is effectively unreachable: when `lastjob` exists and the name query is empty, settlement either unsets `lastjob` or returns; therefore [the later by-ID branch](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_chunk_watchdog.sh:400) cannot normally observe it. Also, a new `lastjob` is persisted only after all arms finish processing ([line 481](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_chunk_watchdog.sh:481)), leaving a crash window after successful submission. Submission needs a reservation/lock shared by every submitter, including manual use.

2. **The positive endpoint attestation is written before the job’s final classification is known.** A chunk with `final_rc=0` appends a positive endpoint at [fa_orbit_train.sbatch:793](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:793), but final-record tee, preflight-copy, or W&B failure can subsequently change `final_rc` to 7 at [line 863](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:863). The recorder checks only non-`<none>` path/step/hash—not a success class or settled `COMPLETED` state ([fa_orbit_record_restart.py:225](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_record_restart.py:225)). Consequently a Slurm `FAILED` job can carry an admissible positive attestation; with a fresh failure streak, the watchdog bumps once, records it, clears the streak, and proceeds. This violates B6’s successful-completion requirement.

3. **Same-boundary retries can attribute the predecessor’s checkpoint to the retry job.** `train.py` uses `ModelCheckpoint` without disabling its default version counter ([train.py:182](/n/fs/gatrdp/codespace/FLAC/train.py:182); [default is `True`](/n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/pytorch_lightning/callbacks/model_checkpoint.py:226)). If `epoch=…-step=42500.ckpt` already exists, a retry writes `epoch=…-step=42500-v1.ckpt`. However:

   - job attestation searches only `*-step=42500.ckpt` ([fa_orbit_train.sbatch:796](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:796));
   - the recorder uses the same exact pattern ([fa_orbit_record_restart.py:350](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_record_restart.py:350));
   - the watchdog’s checkpoint parser also ignores versioned names ([fa_orbit_chunk_watchdog.sh:214](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_chunk_watchdog.sh:214)).

   A successful retry therefore positively attests the stale unversioned file, and the recorder’s rehash agrees—binding another job’s bytes to the retry. This defeats both B3 retry recovery and B6 attribution.

4. **Full-chain validation proves continuity, not link authenticity.** Empty chains are correctly refused by the wrapper, and broken/non-increasing ancestry is rejected. But a single structurally valid forged link—resume hash/step equal to the INITIAL anchor, increasing final step, arbitrary final hash—produces no ancestry error. [check_chain_ancestry](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py:244) never verifies each link’s manifest bytes/SHA, endpoint attestation, job identity, UUID uniqueness, or manifest-SHA uniqueness. The latter checks exist only while appending a new link ([fa_orbit_record_restart.py:306](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_record_restart.py:306)).

   Because B1 deliberately makes the registry mutable and ignored by both drift gates, a crafted single link whose final hash matches a compatible checkpoint in the canonical directory is accepted by `--chain`. B6 therefore still has a crafted-registry laundering path.

## Disposition of prior blockings

- **B1:** Resolved. The registry exclusion works in both drift gates.
- **B2:** Not resolved; see blocking 1.
- **B3:** Terminal retention, `CANCELLED` freeze, job-ID manifest lookup, and unique fallback are sound. Same-boundary checkpoint handling remains broken; see blocking 3.
- **B4:** Resolved. Reconciliation precedes DONE; DONE requires disk and chain tip exactly at TARGET; `S > TARGET` freezes.
- **B5:** Deferred as ruled.
- **B6:** Not resolved; see blockings 2–4.
- **B7:** The live VANL anchor is valid: its on-disk checkpoint SHA-256 is `1095f493…988507`, matching the registry, and its launch-manifest/config hashes also match. Startup checking is weaker than advertised, however: [anchor_state](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_chunk_watchdog.sh:228) checks only field presence, not `final_step == 40000`, hash shape, canonical path, or on-disk hash.

## Deviations and non-blocking notes

- The narrower freeze-clear reset is correct. Resetting every absent-freeze counter would erase sub-threshold failures each poll and prevent the threshold from being reached.
- Keeping VANL at 5000 unless `CHUNK_VANL` is explicitly supplied is acceptable and documented. However, global/per-arm chunk inputs still lack time-pin compatibility checks: `CHUNK=5000` or `CHUNK_C32=5000` is accepted even though C32’s fixed nine-hour pin was sized for 2500 steps.
- FD 8 remains open for the parent lifetime, but it is inherited by external children such as `sleep`. Killing only the watchdog can therefore leave the lock held until that child exits—safe against duplication, but not literally “released when the watchdog is killed.”
- `done_<ARM>` remains unqualified by TARGET. Reusing a state file with a larger TARGET can submit one new leg and then exit because the old nonempty `done` entry is skipped in the remaining-arm count.
- The recorded guard result is indeed 123 pass / 5 dirty-tree failures, but it does not exercise the manual-submit race, final-status promotion after attestation, versioned retry filenames, or a structurally valid forged chain link.

## VERDICT

**REVISE.**
tokens used
166,167
## BLOCKING

1. **B2 remains unresolved for manual/concurrent submission.** The singleton lock is held only by the watchdog ([fa_orbit_chunk_watchdog.sh:140](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_chunk_watchdog.sh:140)); the sanctioned submitter does not acquire it before `sbatch` ([fa_orbit_submit.sh:169](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submit.sh:169)). A manual invocation can still race the watchdog’s `squeue`→submit interval and queue the same boundary twice. The job-side flock only serializes execution; a duplicate can start immediately after the first releases it.

   The advertised by-ID recheck is effectively unreachable: when `lastjob` exists and the name query is empty, settlement either unsets `lastjob` or returns; therefore [the later by-ID branch](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_chunk_watchdog.sh:400) cannot normally observe it. Also, a new `lastjob` is persisted only after all arms finish processing ([line 481](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_chunk_watchdog.sh:481)), leaving a crash window after successful submission. Submission needs a reservation/lock shared by every submitter, including manual use.

2. **The positive endpoint attestation is written before the job’s final classification is known.** A chunk with `final_rc=0` appends a positive endpoint at [fa_orbit_train.sbatch:793](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:793), but final-record tee, preflight-copy, or W&B failure can subsequently change `final_rc` to 7 at [line 863](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:863). The recorder checks only non-`<none>` path/step/hash—not a success class or settled `COMPLETED` state ([fa_orbit_record_restart.py:225](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_record_restart.py:225)). Consequently a Slurm `FAILED` job can carry an admissible positive attestation; with a fresh failure streak, the watchdog bumps once, records it, clears the streak, and proceeds. This violates B6’s successful-completion requirement.

3. **Same-boundary retries can attribute the predecessor’s checkpoint to the retry job.** `train.py` uses `ModelCheckpoint` without disabling its default version counter ([train.py:182](/n/fs/gatrdp/codespace/FLAC/train.py:182); [default is `True`](/n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/pytorch_lightning/callbacks/model_checkpoint.py:226)). If `epoch=…-step=42500.ckpt` already exists, a retry writes `epoch=…-step=42500-v1.ckpt`. However:

   - job attestation searches only `*-step=42500.ckpt` ([fa_orbit_train.sbatch:796](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:796));
   - the recorder uses the same exact pattern ([fa_orbit_record_restart.py:350](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_record_restart.py:350));
   - the watchdog’s checkpoint parser also ignores versioned names ([fa_orbit_chunk_watchdog.sh:214](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_chunk_watchdog.sh:214)).

   A successful retry therefore positively attests the stale unversioned file, and the recorder’s rehash agrees—binding another job’s bytes to the retry. This defeats both B3 retry recovery and B6 attribution.

4. **Full-chain validation proves continuity, not link authenticity.** Empty chains are correctly refused by the wrapper, and broken/non-increasing ancestry is rejected. But a single structurally valid forged link—resume hash/step equal to the INITIAL anchor, increasing final step, arbitrary final hash—produces no ancestry error. [check_chain_ancestry](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py:244) never verifies each link’s manifest bytes/SHA, endpoint attestation, job identity, UUID uniqueness, or manifest-SHA uniqueness. The latter checks exist only while appending a new link ([fa_orbit_record_restart.py:306](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_record_restart.py:306)).

   Because B1 deliberately makes the registry mutable and ignored by both drift gates, a crafted single link whose final hash matches a compatible checkpoint in the canonical directory is accepted by `--chain`. B6 therefore still has a crafted-registry laundering path.

## Disposition of prior blockings

- **B1:** Resolved. The registry exclusion works in both drift gates.
- **B2:** Not resolved; see blocking 1.
- **B3:** Terminal retention, `CANCELLED` freeze, job-ID manifest lookup, and unique fallback are sound. Same-boundary checkpoint handling remains broken; see blocking 3.
- **B4:** Resolved. Reconciliation precedes DONE; DONE requires disk and chain tip exactly at TARGET; `S > TARGET` freezes.
- **B5:** Deferred as ruled.
- **B6:** Not resolved; see blockings 2–4.
- **B7:** The live VANL anchor is valid: its on-disk checkpoint SHA-256 is `1095f493…988507`, matching the registry, and its launch-manifest/config hashes also match. Startup checking is weaker than advertised, however: [anchor_state](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_chunk_watchdog.sh:228) checks only field presence, not `final_step == 40000`, hash shape, canonical path, or on-disk hash.

## Deviations and non-blocking notes

- The narrower freeze-clear reset is correct. Resetting every absent-freeze counter would erase sub-threshold failures each poll and prevent the threshold from being reached.
- Keeping VANL at 5000 unless `CHUNK_VANL` is explicitly supplied is acceptable and documented. However, global/per-arm chunk inputs still lack time-pin compatibility checks: `CHUNK=5000` or `CHUNK_C32=5000` is accepted even though C32’s fixed nine-hour pin was sized for 2500 steps.
- FD 8 remains open for the parent lifetime, but it is inherited by external children such as `sleep`. Killing only the watchdog can therefore leave the lock held until that child exits—safe against duplication, but not literally “released when the watchdog is killed.”
- `done_<ARM>` remains unqualified by TARGET. Reusing a state file with a larger TARGET can submit one new leg and then exit because the old nonempty `done` entry is skipped in the remaining-arm count.
- The recorded guard result is indeed 123 pass / 5 dirty-tree failures, but it does not exercise the manual-submit race, final-status promotion after attestation, versioned retry filenames, or a structurally valid forged chain link.

## VERDICT

**REVISE.**
