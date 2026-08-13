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
session id: 019ffbe6-3613-79a1-8b57-379a9c6f1c76
--------
user
Round-5 review (read-only; do NOT install anything or modify environments/files; output = review text only). Repo /n/fs/gatrdp/codespace/FLAC, exp_11 chunked-training kit. Cumulative diff + the new watchdog file: /tmp/claude-374349/-n-fs-gatrdp-codespace-FLAC/66305eb9-88c7-41df-95da-dde30a426dc2/scratchpad/chunk_kit.diff (live files in worklog/worklog_yixun/exp_11_fa_orbit_claude/).

MOTIVATION: the cluster never backfills 34-160h 8-GPU allocations; five 40k->100k legs sat PENDING >16h. The kit splits each extension into 2500-step chunk jobs (hours-long walltimes: C4L 3h/C8 4h/C16 6h/C32 9h/VANL 2.5h pins) chained by a login-side watchdog.

DESIGN: (1) fa_orbit_ckpt_preflight.py --chain: same INITIAL-identity binding as --extension but the resume anchor is the TIP of arms.<ARM>.chain (links written only by the recorder); empty chain refuses; optional --chunk-end validated (multiple of 2500, EXPECTED_STEP < N <= budget). (2) fa_orbit_record_restart.py: a manifest carrying chunk_end is recorded as an immutable chain link {job, uuid, manifest sha, resume_step/sha, final_step=chunk_end, final_ckpt_sha256 hashed from the on-disk endpoint ckpt}; idempotent per job; content-drift on re-record refuses; identity checks parameterized (anchor/tip, CHUNK wall pin). (3) fa_orbit_submit.sh --chunk-end: shape checks + CHUNK time pin + exports CHUNK_END. (4) fa_orbit_train.sbatch: STOPSTEP=CHUNK_END feeds train.py --max-steps AND the completion-literal classifier; budget pin unmoved; preflight dispatch: EXPECTED_STEP==40000 -> --extension, >40000 -> --chain; CHUNK pins; manifest chunk_end line. (5) NEW fa_orbit_chunk_watchdog.sh: KV-whitelist args, per-arm: skip if job named exp11-<ARM>-train queued; find newest canonical ckpt step S; S>=TARGET -> done; if S>40000 and registry tip<S record the finished chunk first (unique manifest by chunk_end==S, recorder refusal -> failure); submit next chunk via the sanctioned submitter only; 2 consecutive failures freeze the arm (state file, human-clearable); ONESHOT/DRYRUN test hooks. (6) 22 new guard cases, all passing (113 pass total; 5 pre-existing dirty-tree submitter cases clear on commit).

REVIEW FOCUS: (a) can any sequence of watchdog + recorder + preflight actions launder an unrecorded/foreign checkpoint into the lineage, or double-submit an arm (race between squeue check and submit; job-name collisions with the five OLD long legs still queued as exp11-<ARM>-train — we cancel them before starting the watchdog, is anything else vulnerable)? (b) recorder correctness: link append under concurrent invocation (store lock?), atomic write, the mine/at indexing, chunk_end vs budget/cadence edges; (c) preflight --chain: any bypass via crafted registry/manifest (note the resume file is always re-hashed); (d) launcher: STOPSTEP interactions with the completion literal, argv-parity section G, wandb fresh-id (r4) under many chunks, TIME_PIN selection; (e) watchdog: sacct settle logic (CANCELLED counts as failure — is that right when a human scancels?), leg_manifest_for uniqueness under repeated same-boundary attempts after a failed leg (two manifests with chunk_end==S -> permanent freeze — is the recovery path documented/sound?), state-file semantics; (f) the 2500-step chunk size: startup overhead ~10-15min/leg at 8 GPUs — flag if you consider the overhead-to-compute ratio for any arm unacceptable vs a 5000-step chunk. Verdict: BLOCKING/NON-BLOCKING findings then APPROVED or REVISE.
codex
I’ll audit the diff and live scripts read-only, trace the lineage and concurrency invariants end to end, then report only actionable findings and a verdict.
exec
/usr/bin/bash -lc 'wc -l /tmp/claude-374349/-n-fs-gatrdp-codespace-FLAC/66305eb9-88c7-41df-95da-dde30a426dc2/scratchpad/chunk_kit.diff && rg --files worklog/worklog_yixun/exp_11_fa_orbit_claude | sort && git status --short' in /n/fs/gatrdp/codespace/FLAC
 succeeded in 0ms:
1172 /tmp/claude-374349/-n-fs-gatrdp-codespace-FLAC/66305eb9-88c7-41df-95da-dde30a426dc2/scratchpad/chunk_kit.diff
/usr/bin/bash: line 1: rg: command not found
 M worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py
 M worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_record_restart.py
 M worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submit.sh
 M worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch
 M worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh
?? AGENTS.md
?? AcousticRooms
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-12_18-23-10_guardtests.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-13_12-06-48_guardtests.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-13_12-08-41_guardtests.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-13_12-10-57_guardtests.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_chunk_watchdog.sh
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_chunk_kit_review.md
?? worklog/worklog_yixun/exp_15_yaw_aug_claude/slurm_train_exp15-smoke-YAWAUG_3685989.out
?? worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-12_11-59-14_readback.log
?? worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-12_11-59-42_readback.log
?? worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-12_12-05-35_guardtests_fullfix.log
?? worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-12_12-35-37_guardtests_f3fix.log
?? worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-12_12-40-22_guardtests_f3fix.log
?? worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-12_17-49-31_YAWAUG_8x8_jid3685989_manifest.txt
?? worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_smoke_registry.json

exec
/usr/bin/bash -lc "sed -n '1,420p' /tmp/claude-374349/-n-fs-gatrdp-codespace-FLAC/66305eb9-88c7-41df-95da-dde30a426dc2/scratchpad/chunk_kit.diff" in /n/fs/gatrdp/codespace/FLAC
 succeeded in 0ms:
diff --git a/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py b/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py
index fc72f83..c9b6ed2 100644
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
@@ -196,14 +237,54 @@ def check_extension_binding(manifest_path, registry_path, arm, rung, config_path
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
+def check_chain_binding(manifest_path, registry_path, arm, rung, config_path, ckpt_path,
+                        ckpt_sha, expected_step, max_steps, repo_root="."):
+    """The CHUNKED-LEG contract: resume the TIP of this arm's recorded chain.
+
+    A chunk leg is an extension leg that stops early, so it must prove exactly
+    the same INITIAL launch identity — but its resume point is no longer the
+    audited 40k anchor: it is the endpoint of the previous chunk, recorded as the
+    last link of ``arms.<ARM>.chain`` by fa_orbit_record_restart.py.
+
+    Fail-closed by construction: an EMPTY or ABSENT chain refuses, so chunk N+1
+    cannot run until chunk N has been recorded, and every link's final_step /
+    final_ckpt_sha256 was written by the recorder from the file on disk. The
+    resume file is re-hashed by the caller (``ckpt_sha``); a manifest's claimed
+    hash is never trusted here or anywhere else in this chain.
+    """
+    reg, problems = _load_registry_row(registry_path, arm)
+    if problems:
+        return problems, {}
+    man = parse_manifest(manifest_path)
+    problems = check_initial_identity(man, manifest_path, reg, arm, rung, config_path, max_steps)
+    if not reg.get("final_ckpt_sha256"):
+        problems.append(f"{arm} has no audited final_ckpt_sha256 in the registry — a chain of "
+                        "chunks must still descend from the audited INITIAL run")
+    chain = reg.get("chain") or []
+    if not chain:
+        problems.append(f"{arm} has no recorded chain link in {registry_path}: a chunk may only "
+                        "resume the tip of a RECORDED chain, so chunk N+1 is inadmissible until "
+                        "chunk N is recorded — run fa_orbit_record_restart.py on the previous "
+                        "chunk's launcher manifest first")
     else:
-        canon = canonical_ckpt_dir(save_dir, arm, repo_root)
-        if os.path.realpath(os.path.dirname(ckpt_path)) != canon:
-            problems.append(f"resume checkpoint {ckpt_path} does not live in the registered "
-                            f"launch's canonical run directory {canon}")
+        tip = chain[-1]
+        tip_step, tip_sha = tip.get("final_step"), tip.get("final_ckpt_sha256")
+        if str(tip_step) != str(expected_step):
+            problems.append(f"EXPECTED_STEP {expected_step} != the last recorded chain link's "
+                            f"final_step {tip_step!r} (link job {tip.get('job')!r}) — a chunk "
+                            "resumes the TIP of the chain, never an earlier link")
+        if not tip_sha:
+            problems.append(f"the last recorded chain link (job {tip.get('job')!r}) carries no "
+                            "final_ckpt_sha256 — it is not evidence of a checkpoint")
+        elif ckpt_sha != tip_sha:
+            problems.append(f"resume checkpoint sha256 {ckpt_sha[:12]} != the last recorded chain "
+                            f"link's final_ckpt_sha256 {str(tip_sha)[:12]} — this file is not the "
+                            "checkpoint that chunk produced")
+    problems += check_canonical_dir(man, arm, ckpt_path, repo_root)
     return problems, man
 
 
@@ -221,13 +302,25 @@ def main(argv=None):
     ap.add_argument("--extension", action="store_true",
                     help="this restart is the Q10 40k->100k EXTENSION: bind it to the audited "
                          "INITIAL launch identity instead of requiring an equal budget/commit")
+    ap.add_argument("--chain", action="store_true",
+                    help="this restart is a CHUNK continuing an already-recorded chain: same "
+                         "INITIAL identity as --extension, but the resume anchor is the LAST "
+                         "recorded arms.<ARM>.chain link instead of the audited 40k checkpoint")
+    ap.add_argument("--chunk-end", type=int, default=None,
+                    help="this leg's stop step (a chunk boundary): EXPECTED_STEP < N <= max-steps "
+                         "and a multiple of 2500")
     ap.add_argument("--launch-registry", default="",
-                    help="the committed arm launch registry (required with --extension)")
+                    help="the committed arm launch registry (required with --extension/--chain)")
     ap.add_argument("--repo-root", default=".",
                     help="root the registry's relative save_dir is resolved against")
     args = ap.parse_args(argv)
+    if args.extension and args.chain:
+        ap.error("--extension and --chain are mutually exclusive: a leg either resumes the audited "
+                 "40k anchor or the tip of the recorded chunk chain, never both")
     if args.extension and not args.launch_registry:
         ap.error("--extension requires --launch-registry (the audited INITIAL launch row)")
+    if args.chain and not args.launch_registry:
+        ap.error("--chain requires --launch-registry (the recorded chunk chain lives in it)")
 
     if not os.path.isfile(args.ckpt):
         print(f"PREFLIGHT: checkpoint not found: {args.ckpt}")
@@ -277,7 +370,18 @@ def main(argv=None):
     if not n_ema:
         problems.append("no EMA weights in state_dict")
 
-    digest = sha256_file(args.ckpt)     # needed by the extension contract's anchor check
+    # The per-job stop step is checked against the BUDGET (--max-steps), which the
+    # launcher still pins at 100000: a chunk narrows where this leg stops, never
+    # what the campaign is allowed to reach.
+    if args.chunk_end is not None:
+        if args.chunk_end % 2500 != 0:
+            problems.append(f"--chunk-end {args.chunk_end} is not a multiple of 2500 (the pinned "
+                            "checkpoint cadence: a chunk must end ON a checkpoint)")
+        if not args.expected_step < args.chunk_end <= args.max_steps:
+            problems.append(f"--chunk-end {args.chunk_end} must satisfy EXPECTED_STEP "
+                            f"{args.expected_step} < chunk_end <= max_steps {args.max_steps}")
+
+    digest = sha256_file(args.ckpt)     # needed by the extension/chain anchor checks
     man = {}
     if args.launch_manifest:
         if not os.path.isfile(args.launch_manifest):
@@ -287,12 +391,19 @@ def main(argv=None):
                 args.launch_manifest, args.launch_registry, args.arm, args.rung, args.config,
                 args.ckpt, digest, args.expected_step, args.max_steps, args.repo_root)
             problems += more
+        elif args.chain:
+            more, man = check_chain_binding(
+                args.launch_manifest, args.launch_registry, args.arm, args.rung, args.config,
+                args.ckpt, digest, args.expected_step, args.max_steps, args.repo_root)
+            problems += more
         else:
             more, man = check_manifest_binding(args.launch_manifest, args.arm, args.rung,
                                                args.commit, args.max_steps)
             problems += more
     elif args.extension:
         problems.append("--extension requires --launch-manifest (the audited INITIAL manifest)")
+    elif args.chain:
+        problems.append("--chain requires --launch-manifest (the audited INITIAL manifest)")
 
     if problems:
         print("PREFLIGHT FAILED:")
@@ -308,10 +419,22 @@ def main(argv=None):
           f"lr={opts[0]['param_groups'][0].get('lr')} "
           f"sched_last_epoch={ck['lr_schedulers'][0].get('last_epoch')} ema_entries={n_ema}")
     if man and args.extension:
-        print(f"extension lineage OK: {args.arm} {args.expected_step} -> {args.max_steps} continues "
+        print(f"extension lineage OK: {args.arm} {args.expected_step} -> "
+              f"{args.chunk_end if args.chunk_end is not None else args.max_steps} continues "
               f"the audited launch job {kv_line(man, 'job').get('job')} "
               f"(launch commit {man.get('commit', '')[:12]}, running commit {args.commit[:12] or '<none>'})")
         print(f"  bound to the audited launch manifest: {args.launch_manifest}")
+        if args.chunk_end is not None:
+            print(f"  chunk leg: stops at {args.chunk_end} of the {args.max_steps} budget")
+    elif man and args.chain:
+        print(f"chain lineage OK: {args.arm} {args.expected_step} -> "
+              f"{args.chunk_end if args.chunk_end is not None else args.max_steps} resumes the tip "
+              f"of the recorded chunk chain, under the audited launch job "
+              f"{kv_line(man, 'job').get('job')} "
+              f"(launch commit {man.get('commit', '')[:12]}, running commit {args.commit[:12] or '<none>'})")
+        print(f"  bound to the audited launch manifest: {args.launch_manifest}")
+        if args.chunk_end is not None:
+            print(f"  chunk leg: stops at {args.chunk_end} of the {args.max_steps} budget")
     elif man:
         print(f"  bound to launch manifest: {args.launch_manifest}")
     print(f"CKPT_SHA256 {digest}")
diff --git a/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_record_restart.py b/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_record_restart.py
index c20019c..86b3ab8 100644
--- a/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_record_restart.py
+++ b/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_record_restart.py
@@ -26,14 +26,31 @@ else in the manifest was checked at all. Now:
 It also publishes the leg's PRODUCER MANIFEST (fix 2): every checkpoint this leg
 produced, re-hashed from disk, into an append-only per-leg file the screen
 verifies each >40k checkpoint against. Re-run with --extend as the leg saves more.
+
+ROUND 5 — CHUNKED legs. The cluster never backfills a 34-160 h allocation, so a
+leg now trains to the next 2500-step boundary and exits (`chunk_end` in its
+manifest). Such a leg is recorded as a CHAIN LINK in `arms.<ARM>.chain`:
+
+    INITIAL anchor (40000) <- link(40000 -> 42500) <- link(42500 -> 45000) <- ...
+
+Each link is admissible only if it resumed the TIP of the chain (or the audited
+INITIAL anchor, for the first link) with the file that is on disk NOW, re-hashed.
+The endpoint checkpoint the leg produced is likewise located in the audited
+canonical directory and hashed by this recorder, which is what makes it the next
+link's anchor: fa_orbit_ckpt_preflight.py --chain refuses a chunk whose resume
+file is not the last link's `final_ckpt_sha256`, so chunk N+1 cannot start until
+chunk N is recorded here. Appending a link never touches the INITIAL fields, is
+atomic (tmp+rename under the store lock), and is idempotent per job.
 """
 import argparse
 import fcntl
+import glob
 import hashlib
 import json
 import os
 import re
 import sys
+from datetime import datetime, timezone
 
 HERE = os.path.dirname(os.path.abspath(__file__))
 sys.path.insert(0, HERE)
@@ -41,6 +58,9 @@ import fa_orbit_producer_manifest as pm            # noqa: E402
 from fa_orbit_ckpt_preflight import canonical_ckpt_dir    # noqa: E402
 
 PIN_RE = re.compile(r'^(PINNED_[A-Z0-9_]+)=(?:"([^"]*)"|(\S+))')
+# The pinned checkpoint cadence: a chunk may only end ON a saved checkpoint,
+# otherwise the next chunk has nothing to resume.
+CHUNK_STEP = 2500
 
 
 def read_pins(launcher):
@@ -70,12 +90,19 @@ def kvs(man, key):
     return {f[i]: f[i + 1] for i in range(0, len(f) - 1, 2)}
 
 
-def check_identity(arm, man, initial, pins, repo_root):
-    """Every field of the RESTART manifest, against the audited INITIAL row + Q10 pins."""
+def check_identity(arm, man, initial, pins, repo_root,
+                   want_anchor=None, want_step=None, want_time=None):
+    """Every field of the RESTART manifest, against the audited INITIAL row + Q10 pins.
+
+    A CHUNK leg proves the same identity but resumes the tip of the recorded
+    chain rather than the audited 40k anchor, and is walled by the arm's CHUNK
+    pin rather than its RESTART pin, so the caller may override those three
+    expectations. Everything else is identical for both leg kinds."""
     jk, ak, rk = kvs(man, "job"), kvs(man, "arm"), kvs(man, "resume_ckpt")
     tk = kvs(man, "time_limit")
     problems = []
-    anchor, final_step = initial.get("final_ckpt_sha256"), initial.get("final_step")
+    anchor = want_anchor if want_anchor is not None else initial.get("final_ckpt_sha256")
+    final_step = want_step if want_step is not None else initial.get("final_step")
     if not anchor:
         problems.append(f"{arm} has no audited final_ckpt_sha256 to chain from — audit the "
                         "INITIAL run's final checkpoint before recording a leg")
@@ -104,13 +131,18 @@ def check_identity(arm, man, initial, pins, repo_root):
     if ak.get("max_steps") != pins.get("PINNED_MAXSTEPS"):
         problems.append(f"manifest max_steps {ak.get('max_steps')!r} != the Q10 budget pin "
                         f"{pins.get('PINNED_MAXSTEPS')!r}")
+    step_label = "audited final step" if want_step is None else "recorded chain tip's final step"
     if final_step is not None and str(rk.get("expected_step")) != str(final_step):
-        problems.append(f"manifest expected_step {rk.get('expected_step')!r} != the audited final "
-                        f"step {final_step!r} — a leg resumes where the INITIAL run ended")
-    want_time = pins.get(f"PINNED_TIME_LIMIT_RESTART_{arm}")
+        problems.append(f"manifest expected_step {rk.get('expected_step')!r} != the {step_label} "
+                        f"{final_step!r} — a leg resumes where the run it continues ended")
+    pin_label = "RESTART wall pin"
+    if want_time is None:
+        want_time = pins.get(f"PINNED_TIME_LIMIT_RESTART_{arm}")
+    else:
+        pin_label = "CHUNK wall pin"
     if tk.get("time_limit") != want_time:
-        problems.append(f"manifest time_limit {tk.get('time_limit')!r} != the arm's RESTART wall "
-                        f"pin {want_time!r}")
+        problems.append(f"manifest time_limit {tk.get('time_limit')!r} != the arm's {pin_label} "
+                        f"{want_time!r}")
     if int(initial.get("training_seed", -1)) != 42:
         problems.append(f"registered training seed {initial.get('training_seed')!r} != 42")
     # the config the leg names must still hash to the audited value
@@ -148,6 +180,135 @@ def resolve(root, path):
     return path if os.path.isabs(path) else os.path.join(root, path)
 
 
+def chunk_end_of(man):
+    """The leg's per-job stop step as written by the launcher, or None.
+
+    A manifest with no `chunk_end` line (or the literal `<none>`) is a
+    whole-budget RESTART leg and takes the original recording path untouched."""
+    v = (man.get("chunk_end", "").split() or [""])[0]
+    return None if not v or v == "<none>" else v
+
+
+def utc_now():
+    return datetime.now(timezone.utc).isoformat(timespec="seconds")
+
+
+def record_chunk(args, arm, reg, initial, man, man_sha, pins, chunk_raw):
+    """Append ONE chain link for a chunked leg: <resume_step> -> <chunk_end>.
+
+    The link is what makes the NEXT chunk admissible (preflight --chain), so it
+    is written only when this leg provably resumed the tip of the chain and its
+    endpoint checkpoint is on disk, unambiguous, and hashed here from that disk.
+    """
+    jk, rk = kvs(man, "job"), kvs(man, "resume_ckpt")
+    job = jk.get("job")
+    problems = []
+    if args.extend:
+        problems.append("--extend extends a producer manifest and has no meaning for a CHUNK leg; "
+                        "each chunk is recorded once, as one immutable chain link")
+    if not job:
+        problems.append("manifest records no job — a leg with no identity is not a record")
+
+    # --- the chunk boundary itself ------------------------------------------
+    chunk_end = resume_step = None
+    try:
+        chunk_end = int(chunk_raw)
+    except (TypeError, ValueError):
+        problems.append(f"manifest chunk_end {chunk_raw!r} is not an integer")
+    try:
+        resume_step = int(rk.get("expected_step"))
+    except (TypeError, ValueError):
+        problems.append(f"manifest expected_step {rk.get('expected_step')!r} is not an integer")
+    budget = None
+    try:
+        budget = int(pins.get("PINNED_MAXSTEPS"))
+    except (TypeError, ValueError):
+        problems.append(f"the launcher's PINNED_MAXSTEPS {pins.get('PINNED_MAXSTEPS')!r} is not an "
+                        "integer — the chunk boundary cannot be checked against the budget")
+    if chunk_end is not None:
+        if chunk_end % CHUNK_STEP:
+            problems.append(f"manifest chunk_end {chunk_end} is not a multiple of {CHUNK_STEP} (the "
+                            "pinned checkpoint cadence: a chunk must end ON a checkpoint)")
+        if resume_step is not None and chunk_end <= resume_step:
+            problems.append(f"manifest chunk_end {chunk_end} does not exceed the resume step "
+                            f"{resume_step} — the leg would produce no new checkpoint")
+        if budget is not None and chunk_end > budget:
+            problems.append(f"manifest chunk_end {chunk_end} exceeds the pinned budget {budget}")
+
+    # --- which link (or the audited anchor) this chunk continues ------------
+    chain = initial.get("chain") or []
+    mine = [i for i, link in enumerate(chain) if link.get("job") == job]

exec
/usr/bin/bash -lc "sed -n '421,840p' /tmp/claude-374349/-n-fs-gatrdp-codespace-FLAC/66305eb9-88c7-41df-95da-dde30a426dc2/scratchpad/chunk_kit.diff" in /n/fs/gatrdp/codespace/FLAC
 succeeded in 0ms:
+    if len(mine) > 1:
+        raise SystemExit(f"{arm} has {len(mine)} chain links claiming job {job} — the registry is "
+                         "inconsistent; fix it before recording")
+    at = mine[0] if mine else len(chain)
+    prev = chain[at - 1] if at > 0 else None
+    if prev is None:
+        want_anchor = want_step = None                 # the audited INITIAL anchor
+        prev_desc = f"the audited INITIAL anchor at step {initial.get('final_step')}"
+    else:
+        want_anchor, want_step = prev.get("final_ckpt_sha256"), prev.get("final_step")
+        prev_desc = f"chain link job {prev.get('job')} ending at step {prev.get('final_step')}"
+    want_time = pins.get(f"PINNED_TIME_LIMIT_CHUNK_{arm}")
+    if not want_time:
+        problems.append(f"the launcher carries no PINNED_TIME_LIMIT_CHUNK_{arm} pin, so this "
+                        "chunk leg's wall time cannot be bound to a pin")
+    ident, resume_real = check_identity(arm, man, initial, pins, args.repo_root,
+                                        want_anchor=want_anchor, want_step=want_step,
+                                        want_time=want_time)
+    problems += ident
+
+    # --- the endpoint checkpoint this leg produced --------------------------
+    endpoint = None
+    if chunk_end is not None:
+        ckpt_dir = canonical_ckpt_dir(initial.get("save_dir", ""), arm, args.repo_root)
+        hits = sorted(glob.glob(os.path.join(ckpt_dir, f"*-step={chunk_end}.ckpt")))
+        if len(hits) != 1:
+            problems.append(f"expected exactly 1 checkpoint at the chunk end step {chunk_end} in "
+                            f"{ckpt_dir}, found {len(hits)}"
+                            + (": " + ", ".join(os.path.basename(h) for h in hits) if hits else ""))
+        else:
+            endpoint = hits[0]
+    if problems:
+        print("RECORD REFUSED:")
+        for p in problems:
+            print(f"  !! {p}")
+        return 2
+
+    # check_identity already re-hashed the resume file and proved it IS this hash
+    resume_sha = want_anchor if want_anchor is not None else initial.get("final_ckpt_sha256")
+    link = {"job": job, "launch_uuid": jk.get("launch_uuid"),
+            "manifest_path": os.path.abspath(args.manifest), "manifest_sha256": man_sha,
+            "resume_step": resume_step, "resume_ckpt_sha256": resume_sha,
+            "final_step": chunk_end, "final_ckpt_sha256": pm.sha256_file(endpoint),
+            "recorded_utc": utc_now()}
+
+    if mine:
+        old = chain[at]
+        differing = sorted(k for k in link if k != "recorded_utc"
+                           and str(old.get(k)) != str(link[k]))
+        if differing:
+            print("RECORD REFUSED:")
+            print(f"  !! {arm} job {job} is ALREADY a chain link whose content differs "
+                  f"({', '.join(differing)}) — a recorded link is immutable, and rewriting one "
+                  "would re-parent every chunk recorded after it")
+            return 2
+        print(f"{arm} chunk link job {job} ({resume_step} -> {chunk_end}) is already recorded, "
+              "byte-identical — no-op")
+        return 0
+
+    initial.setdefault("chain", []).append(link)
+    if not args.dry_run:
+        pm.write_atomic(args.registry, reg)
+    print(f"recorded {arm} chunk link job {job}: {resume_step} -> {chunk_end}, continuing "
+          f"{prev_desc} " + ("(dry run, nothing written)" if args.dry_run else "(published)"))
+    print(f"  resume {str(resume_sha)[:12]} -> endpoint {link['final_ckpt_sha256'][:12]} "
+          f"({pm.rel_to(args.repo_root, endpoint)})")
+    print(f"  {arm} chain is now {len(initial['chain'])} link(s), tip at step {chunk_end}"
+          f" of the {budget} budget")
+    return 0
+
+
 def main(argv=None):
     ap = argparse.ArgumentParser(description="record an exp_11 RESTART leg")
     ap.add_argument("arm")
@@ -191,6 +352,11 @@ def record(args, arm, pins):
 
     raw, man = parse_manifest(args.manifest)
     man_sha = hashlib.sha256(raw).hexdigest()
+    # Round 5: a leg that carries a `chunk_end` line stopped at a chunk boundary
+    # and is recorded as a CHAIN LINK, not as a whole-budget RESTART row.
+    chunk_raw = chunk_end_of(man)
+    if chunk_raw is not None:
+        return record_chunk(args, arm, reg, initial, man, man_sha, pins, chunk_raw)
     problems, resume_real = check_identity(arm, man, initial, pins, args.repo_root)
     jk, ak, rk = kvs(man, "job"), kvs(man, "arm"), kvs(man, "resume_ckpt")
     job = jk.get("job")
diff --git a/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submit.sh b/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submit.sh
index 011f5bc..67a22d5 100644
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
@@ -94,9 +124,10 @@ ARGS=(
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
@@ -120,6 +151,7 @@ TMP="$(mktemp "${MANIFEST}.XXXXXX")" || exit 3
   echo "commit ${SHA}"
   echo "pins rung=${RUNG} maxsteps=$(pin PINNED_MAXSTEPS) ckpt_every=$(pin PINNED_CHECKPOINT_EVERY) min_free_mb=$(pin PINNED_MIN_FREE_MB) p0_manifest_sha256=$(pin PINNED_P0_MANIFEST_SHA256)"
   echo "resume ${RESUME_CKPT:-<none>} expected_step ${EXPECTED_STEP}"
+  echo "chunk_end ${CHUNK_END:-<none>}"
   echo "sbatch sbatch ${ARGS[*]}"
 } >> "$TMP" || { echo "intent manifest write failed - abort"; exit 3; }
 mv -n "$TMP" "$MANIFEST" || { echo "intent manifest publication failed - abort"; exit 2; }
diff --git a/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch b/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch
index 3bfa2e7..24654d4 100644
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
 
@@ -345,14 +386,14 @@ else
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
@@ -530,8 +571,20 @@ if [ "$MODE" = "RESTART" ]; then
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
@@ -640,6 +693,7 @@ rm -f "$PIPFREEZE_FILE"
   echo "gpu_uuids ${UUID_CSV}"
   echo "time_limit ${TIME_LIMIT} min_free_mb ${MIN_FREE_MB}"
   echo "resume_ckpt ${RESUME_CKPT:-<none>} expected_step ${EXPECTED_STEP} resume_ckpt_sha256 ${CKPT_SHA:-<none>}"
+  echo "chunk_end ${CHUNK_END:-<none>}"
   echo "save_dir ${SAVEDIR}"
   echo "slurm_transcript ${SLURM_OUT_AT_LAUNCH:-<unknown>} untrack ${UNTRACK_STATE}"
   echo "train_log ${TRAINLOG}"
@@ -707,8 +761,12 @@ if [ "$WANDB_CHECK_RC" -ne 0 ]; then
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
@@ -724,6 +782,7 @@ FINAL_RECORD="$(
   echo "$CLASSIFY_OUT"
   echo "last checkpoint: ${LAST_CKPT:-<none>}"
   echo "restart: fa_orbit_submit.sh ${ARM} --resume ${LAST_CKPT:-<path>} --expected-step <step>"
+  [ -n "$CHUNK_END" ] && echo "next chunk: record this leg (fa_orbit_record_restart.py ${ARM} ${MANIFEST}), then fa_orbit_submit.sh ${ARM} --resume <ckpt at ${CHUNK_END}> --expected-step ${CHUNK_END} --chunk-end <${CHUNK_END}+${PINNED_CHECKPOINT_EVERY}>"
 )"
 printf '%s\n' "$FINAL_RECORD" | tee -a "$TRAINLOG" "$SAVEDIR_LOG" > /dev/null
 final_tee_rc="${PIPESTATUS[1]}"
diff --git a/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh b/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh
index b487055..770a9f2 100755
--- a/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh
+++ b/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh
@@ -430,6 +430,122 @@ printf 'tamper\n' >> "${TMP}/ext_launch_manifest.txt"
 expect_cmd "extension refuses a manifest that drifted after registration" 2 "changed after it was registered" -- \
   "${EXT[@]}" --max-steps 100000
 
+echo "--- G4. Round 5: CHUNKED legs — chain preflight, chunk recorder, gates ---"
+# A fresh fixture set (G3's manifest was deliberately tampered above): the same
+# INITIAL identity, plus the 42500 endpoint checkpoint a first chunk produces
+# and the launcher manifest that chunk leg would leave behind.
+CH_ROOT="${TMP}/chain"; CH_SAVE="${CH_ROOT}/exp11_C8"
+CH_CKPT_DIR="${CH_SAVE}/FLAC_exp11_C8/exp11_C8/checkpoints"
+mkdir -p "$CH_CKPT_DIR"
+CHUNK_PIN_C8="$(awk -F'"' '/^PINNED_TIME_LIMIT_CHUNK_C8=/{print $2; exit}' "$LAUNCHER")"
+$PY - "$TMP" "${EXPDIR}/FLAC_AR_BF_C8.json" "$CH_CKPT_DIR" "$CH_SAVE" "$LAUNCHER" "$CHUNK_PIN_C8" <<'PY'
+import hashlib, json, os, re, sys, torch
+tmp, cfg_path, ckpt_dir, save_dir, launcher, chunk_pin = sys.argv[1:7]
+vae_sha = re.search(r'^PINNED_VAE_SHA256="([^"]*)"', open(launcher).read(), re.M).group(1)
+cfg = json.load(open(cfg_path))
+def ck(step, epoch):
+    return {"global_step": step, "epoch": epoch, "model_config": cfg,
+            "state_dict": {"diffusion.x": torch.zeros(1), "diffusion_ema.x": torch.zeros(1)},
+            "optimizer_states": [{"state": {0: {"step": 1}}, "param_groups": [{"lr": 1e-5}]}],
+            "lr_schedulers": [{"last_epoch": step}]}
+p40 = os.path.join(ckpt_dir, "epoch=8-step=40000.ckpt"); torch.save(ck(40000, 8), p40)
+p42 = os.path.join(ckpt_dir, "epoch=9-step=42500.ckpt"); torch.save(ck(42500, 9), p42)
+sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
+cfg_sha = hashlib.sha256(open(cfg_path, "rb").read()).hexdigest()
+man = os.path.join(tmp, "chain_launch_manifest.txt")
+with open(man, "w") as fh:
+    fh.write("job 3648695 host neu000 mode INITIAL launch_uuid ext-uuid-c8\n")
+    fh.write("arm C8 rung 8x8 micro 8 ngpu 8 max_steps 40000 ckpt_every 2500\n")
+    fh.write("commit " + "2" * 40 + "\n")
+    fh.write(f"model_config {cfg_path}\nconfig_sha256 {cfg_sha}\nvae_sha256 {vae_sha}\n")
+    fh.write(f"save_dir {save_dir}\nwandb_run_id exp11-C8-chain\n")
+reg = {"arms": {"C8": {
+    "manifest_path": man, "manifest_sha256": sha(man),
+    "job": "3648695", "mode": "INITIAL", "launch_uuid": "ext-uuid-c8",
+    "commit": "2" * 40, "rung": "8x8", "max_steps": "40000",
+    "config_sha256": cfg_sha, "vae_sha256": vae_sha, "save_dir": save_dir,
+    "training_seed": 42, "final_ckpt_sha256": sha(p40), "final_step": 40000}}, "restarts": {}}
+json.dump(reg, open(os.path.join(tmp, "chain_registry.json"), "w"), indent=2)
+# the launcher manifest a finished 40000->42500 chunk leg leaves behind
+cman = os.path.join(tmp, "chunk_leg_manifest.txt")
+with open(cman, "w") as fh:
+    fh.write("job 3999001 host neu001 mode RESTART launch_uuid chunk-uuid-1\n")
+    fh.write("arm C8 rung 8x8 micro 8 ngpu 8 max_steps 100000 ckpt_every 2500\n")
+    fh.write("commit " + "3" * 40 + "\n")
+    fh.write(f"model_config {cfg_path}\nconfig_sha256 {cfg_sha}\nvae_sha256 {vae_sha}\n")
+    fh.write(f"save_dir {save_dir}\n")
+    fh.write(f"resume_ckpt {p40} expected_step 40000 resume_ckpt_sha256 {sha(p40)}\n")
+    fh.write(f"time_limit {chunk_pin}\nchunk_end 42500\n")
+print("chain fixture written")
+PY
+CH_CKPT42="${CH_CKPT_DIR}/epoch=9-step=42500.ckpt"
+chain_pf() {  # <registry> [extra preflight args...]
+  local reg="$1"; shift
+  $PY "$PREFLIGHT" --config "${EXPDIR}/FLAC_AR_BF_C8.json" --arm C8 --rung 8x8 \
+    --max-steps 100000 --ckpt "$CH_CKPT42" --expected-step 42500 --commit "$HEAD_SHA" \
+    --launch-manifest "${TMP}/chain_launch_manifest.txt" --chain --launch-registry "$reg" "$@"
+}
+expect_cmd "chain refuses an UNRECORDED predecessor (the fail-closed core)" 2 "no recorded chain link" -- \
+  chain_pf "${TMP}/chain_registry.json"
+recorder() { $PY "${EXPDIR}/fa_orbit_record_restart.py" C8 "${TMP}/chunk_leg_manifest.txt" \
+  --registry "${TMP}/chain_registry.json" --launcher "$LAUNCHER" --repo-root "$PWD" "$@"; }
+expect_cmd "recorder dry run validates but writes nothing" 0 "dry run, nothing written" -- recorder --dry-run
+expect_cmd "chain still refuses after a dry run" 2 "no recorded chain link" -- \
+  chain_pf "${TMP}/chain_registry.json"
+expect_cmd "recorder records the finished chunk as a chain link" 0 "recorded C8 chunk link" -- recorder
+expect_cmd "recorder is idempotent per job" 0 "already recorded" -- recorder
+expect_cmd "chain accepts the recorded tip" 0 "restart lineage OK" -- chain_pf "${TMP}/chain_registry.json"
+expect_cmd "chain + valid --chunk-end accepted" 0 "restart lineage OK" -- \
+  chain_pf "${TMP}/chain_registry.json" --chunk-end 45000
+expect_cmd "chain refuses a --chunk-end off the checkpoint cadence" 2 "not a multiple of 2500" -- \
+  chain_pf "${TMP}/chain_registry.json" --chunk-end 43000
+expect_cmd "chain refuses a --chunk-end at or below the resume step" 2 "chunk_end <= max_steps" -- \
+  chain_pf "${TMP}/chain_registry.json" --chunk-end 42500
+printf '# drift\n' >> "${TMP}/chunk_leg_manifest.txt"
+expect_cmd "recorder refuses to rewrite a recorded link" 2 "immutable" -- recorder
+$PY - "${TMP}/chain_registry.json" "${TMP}/chain_reg_badsha.json" <<'PY'
+import json, sys

exec
/usr/bin/bash -lc "sed -n '841,1240p' /tmp/claude-374349/-n-fs-gatrdp-codespace-FLAC/66305eb9-88c7-41df-95da-dde30a426dc2/scratchpad/chunk_kit.diff" in /n/fs/gatrdp/codespace/FLAC
 succeeded in 0ms:
+r = json.load(open(sys.argv[1])); r["arms"]["C8"]["chain"][-1]["final_ckpt_sha256"] = "d" * 64
+json.dump(r, open(sys.argv[2], "w"), indent=2)
+PY
+expect_cmd "chain refuses a resume that is not the recorded tip's checkpoint" 2 "not the checkpoint that chunk produced" -- \
+  chain_pf "${TMP}/chain_reg_badsha.json"
+expect_cmd "chain refuses a non-tip resume step" 2 "resumes the TIP" -- \
+  $PY "$PREFLIGHT" --config "${EXPDIR}/FLAC_AR_BF_C8.json" --arm C8 --rung 8x8 \
+    --max-steps 100000 --ckpt "${CH_CKPT_DIR}/epoch=8-step=40000.ckpt" --expected-step 40000 \
+    --commit "$HEAD_SHA" --launch-manifest "${TMP}/chain_launch_manifest.txt" --chain \
+    --launch-registry "${TMP}/chain_registry.json"
+# launcher-side CHUNK_END gates (parameter gates run under DRYRUN)
+case_run "CHUNK_END on an INITIAL launch refused" 2 "only a RESTART leg may be chunked" \
+  -- DRYRUN=1 ARM=C4L "EXPECT_SHA=${HEAD_SHA}" "OUTPUT_ROOT=${OUT_ROOT}" "${REPO_ENV[@]}" CHUNK_END=42500
+case_run "CHUNK_END under SMOKE refused" 2 "no meaning under SMOKE" \
+  -- "${SMOKE_ENV[@]}" ARM=C4L CHUNK_END=42500 EXPECTED_STEP=40000
+# submitter-side --chunk-end shape gates (refused before any pin/sbatch work)
+expect_cmd "submitter: --chunk-end without --resume refused" 2 "valid only together" -- \
+  env DRYRUN=1 bash "$SUBMITTER" C8 --chunk-end 42500
+expect_cmd "submitter: --chunk-end off cadence refused" 2 "not a multiple of 2500" -- \
+  env DRYRUN=1 bash "$SUBMITTER" C8 --resume x.ckpt --expected-step 40000 --chunk-end 42600
+expect_cmd "submitter: --chunk-end must exceed the resume step" 2 "must exceed" -- \
+  env DRYRUN=1 bash "$SUBMITTER" C8 --resume x.ckpt --expected-step 42500 --chunk-end 42500
+expect_cmd "submitter: --chunk-end above the budget refused" 2 "exceeds the pinned budget" -- \
+  env DRYRUN=1 bash "$SUBMITTER" C8 --resume x.ckpt --expected-step 40000 --chunk-end 102500
+# watchdog argument safety + the no-checkpoint skip (ONESHOT, everything in TMP)
+WD="${EXPDIR}/fa_orbit_chunk_watchdog.sh"
+WD_TMP="${TMP}/wd"; mkdir -p "${WD_TMP}/outputs"
+expect_cmd "watchdog rejects an off-cadence CHUNK" 2 "not a multiple" -- \
+  bash "$WD" ONESHOT=1 CHUNK=2600
+expect_cmd "watchdog rejects an unknown arm" 2 "not a comma-separated list" -- \
+  bash "$WD" ONESHOT=1 ARMS=C4L,BOGUS
+expect_cmd "watchdog rejects an unknown key" 2 "unknown argument" -- \
+  bash "$WD" ONESHOT=1 SBATCH_EXTRA=x
+# NOTE: this case runs against the REAL queue (no squeue fake): with a live
+# exp11-C8-train job it logs "live job — nothing to do", without one it logs
+# "nothing to resume, skipping" (the fixture OUTPUT_ROOT is empty). Either way
+# a full ONESHOT pass must complete cleanly without submitting anything.
+expect_cmd "watchdog completes a ONESHOT pass without submitting" 0 "ONESHOT: one pass complete" -- \
+  bash "$WD" ONESHOT=1 DRYRUN=1 ARMS=C8 "OUTPUT_ROOT=${WD_TMP}/outputs" \
+    "REGISTRY=${TMP}/chain_registry.json" "STATE=${WD_TMP}/state" "LOG=${WD_TMP}/log"
+
 echo "--- H. the submitter refuses un-pinned submission ---"
 # RETIRED for the same reason as the launcher case above: all pins are concrete,
 # so the submitter's placeholder refusal is unreachable on the real file.
=== NEW FILE: fa_orbit_chunk_watchdog.sh ===
#!/usr/bin/env bash
# ============================================================================
# fa_orbit_chunk_watchdog.sh — login-side driver for exp_11's CHUNKED legs.
#
# WHY. The partition never backfills a 34-160 h allocation, so the 40k -> 100k
# extension legs sat PD indefinitely. A chunked leg asks for hours instead: it
# trains to the next 2500-step boundary, saves, and exits. This script is what
# turns a sequence of such jobs into one continuous run — it watches for an arm
# with no live job, records the chunk that just finished, and submits the next.
#
# WHAT IT MAY DO. Exactly three things: read (squeue/sacct/ls/the registry),
# append to its own log and state file, and call the two sanctioned tools —
# fa_orbit_record_restart.py and fa_orbit_submit.sh. It NEVER calls sbatch
# itself, never deletes or rewrites a checkpoint, a manifest or the registry
# (the recorder owns the registry), and never edits the tracked tree.
#
# FAIL-CLOSED. Every step is refused rather than guessed: an ambiguous manifest
# or checkpoint, a recorder refusal, or a submitter refusal counts as a failure
# for that arm, and MAX_CONSEC_FAIL consecutive failures FREEZE the arm until a
# human removes its frozen_<ARM> line from the state file. A frozen arm is
# skipped loudly, so the log says why nothing is progressing. The chunk chain
# itself is guarded independently of this script: fa_orbit_ckpt_preflight.py
# --chain refuses a chunk whose predecessor was never recorded, so a watchdog
# bug cannot launder an unrecorded checkpoint into the lineage.
#
# USAGE (KEY=VALUE arguments only; every key whitelisted, no value is ever eval'd)
#   bash fa_orbit_chunk_watchdog.sh                       # the pinned defaults
#   bash fa_orbit_chunk_watchdog.sh ARMS=C4L,C8 POLL=600
#   bash fa_orbit_chunk_watchdog.sh ONESHOT=1 DRYRUN=1    # one pass, submits nothing
#
#   ARMS=C4L,C8,C16,C32,VANL  CHUNK=2500  TARGET=100000  POLL=300
#   MAX_CONSEC_FAIL=2  ONESHOT=0  DRYRUN=0
#   TEST HOOKS (change no decision, only where the script reads/writes):
#   OUTPUT_ROOT=outputs_FLAC  REGISTRY=<expdir>/arm_launch_registry.json
#   STATE=<expdir>/.chunk_watchdog_state  LOG=<expdir>/fa_orbit_chunk_watchdog.log
#
# COST. One `squeue` and one `ls` per arm per poll (plus, once an arm is above
# 40000, one short python read of the registry). Nothing else runs on the login
# node; the training itself is entirely inside Slurm.
# ============================================================================
set -uo pipefail
cd "$(git -C "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" rev-parse --show-toplevel)" || exit 3

EXPDIR="worklog/worklog_yixun/exp_11_fa_orbit_claude"
SUBMITTER="${EXPDIR}/fa_orbit_submit.sh"
RECORDER="${EXPDIR}/fa_orbit_record_restart.py"
PY=/n/fs/gatrdp/envs/flac/bin/python
ANCHOR_STEP=40000                 # where every arm's INITIAL run ended

ARMS="C4L,C8,C16,C32,VANL"; CHUNK=2500; TARGET=100000; POLL=300
MAX_CONSEC_FAIL=2; ONESHOT=0; DRYRUN=0
OUTPUT_ROOT="outputs_FLAC"
REGISTRY="${EXPDIR}/arm_launch_registry.json"
STATE="${EXPDIR}/.chunk_watchdog_state"
LOG="${EXPDIR}/fa_orbit_chunk_watchdog.log"

# --- argument parsing: whitelist the KEY, shape-check the VALUE, never eval ---
reject()   { echo "$1" >&2; exit 2; }
is_num()   { case "${1:-}" in ''|*[!0-9]*) return 1 ;; esac; }
is_armset() {
  local v="$1" a
  [ -n "$v" ] || return 1
  local IFS=,
  for a in $v; do
    case "$a" in C4L|C8|C16|C32|VANL) ;; *) return 1 ;; esac
  done
}
for kv in "$@"; do
  case "$kv" in *=*) ;; *) reject "argument '${kv}' is not KEY=VALUE" ;; esac
  key="${kv%%=*}"; val="${kv#*=}"
  case "$key" in
    ARMS)            is_armset "$val" || reject "ARMS='${val}' is not a comma-separated list of C4L|C8|C16|C32|VANL" ;;
    CHUNK|TARGET|POLL|MAX_CONSEC_FAIL)
                     is_num "$val" || reject "${key}='${val}' is not a non-negative integer" ;;
    ONESHOT|DRYRUN)  case "$val" in 0|1) ;; *) reject "${key}='${val}' must be 0 or 1" ;; esac ;;
    OUTPUT_ROOT|REGISTRY|STATE|LOG)
                     case "$val" in
                       ''|*[!A-Za-z0-9/._-]*) reject "${key}='${val}' has unsafe characters" ;;
                     esac ;;
    *)               reject "unknown argument '${kv}' (expected ARMS=/CHUNK=/TARGET=/POLL=/MAX_CONSEC_FAIL=/ONESHOT=/DRYRUN=/OUTPUT_ROOT=/REGISTRY=/STATE=/LOG=)" ;;
  esac
  printf -v "$key" '%s' "$val"      # name whitelisted above; value never parsed
done
[ "$CHUNK" -gt 0 ] || reject "CHUNK must be positive"
[ "$((CHUNK % 2500))" -eq 0 ] || reject "CHUNK=${CHUNK} is not a multiple of the 2500-step checkpoint cadence"
[ "$TARGET" -gt "$ANCHOR_STEP" ] || reject "TARGET=${TARGET} must exceed the ${ANCHOR_STEP} anchor"
[ "$((TARGET % 2500))" -eq 0 ] || reject "TARGET=${TARGET} is not a multiple of 2500"
[ "$MAX_CONSEC_FAIL" -ge 1 ] || reject "MAX_CONSEC_FAIL must be at least 1"
[ -f "$SUBMITTER" ] || reject "missing ${SUBMITTER}"
[ -f "$RECORDER" ] || reject "missing ${RECORDER}"
[ -f "$REGISTRY" ] || reject "missing ${REGISTRY}"

log() { printf '%s %s\n' "$(date -Is)" "$1" >> "$LOG"; echo "$1"; }

# --- state: a plain KV file a human can read and edit ------------------------
# keys: fail_<ARM> <n> | frozen_<ARM> <when>|<reason> | lastjob_<ARM> <jid> |
#       done_<ARM> <step>.  Removing a frozen_<ARM> line un-freezes that arm.
declare -A ST=()
load_state() {
  ST=()
  [ -f "$STATE" ] || return 0
  local k v
  while read -r k v; do
    case "$k" in ''|\#*) continue ;; esac
    ST["$k"]="$v"
  done < "$STATE"
}
save_state() {
  local tmp="${STATE}.tmp.$$" k
  {
    echo "# exp_11 chunk watchdog state (plain KV, rewritten atomically)."
    echo "# Delete a frozen_<ARM> line to let the watchdog submit that arm again."
    if [ "${#ST[@]}" -gt 0 ]; then
      for k in "${!ST[@]}"; do printf '%s %s\n' "$k" "${ST[$k]}"; done | sort
    fi
  } > "$tmp" || { echo "could not write ${tmp}" >&2; return 1; }
  mv -f "$tmp" "$STATE" || { echo "could not publish ${STATE}" >&2; return 1; }
}

bump_fail() {   # <arm> <reason>
  local arm="$1" reason="$2" n
  n=$(( ${ST[fail_$arm]:-0} + 1 ))
  ST["fail_$arm"]="$n"
  log "${arm}: FAILURE (${reason}) — ${n}/${MAX_CONSEC_FAIL} consecutive"
  if [ "$n" -ge "$MAX_CONSEC_FAIL" ]; then
    ST["frozen_$arm"]="$(date -Is)|${reason}"
    log "${arm}: !!! FROZEN after ${n} consecutive failures (${reason}). No further submission for this arm until its frozen_${arm} line is removed from ${STATE}."
  fi
}

clear_fail() {  # <arm> <why>
  if [ "${ST[fail_$1]:-0}" != "0" ]; then
    log "${1}: failure streak cleared (${2})"
  fi
  ST["fail_$1"]="0"
}

newest_ckpt_step() {   # <ckpt dir> -> the largest step with a checkpoint file
  ls -1 "$1" 2>/dev/null \
    | sed -n 's/^epoch=[0-9]\{1,\}-step=\([0-9]\{1,\}\)\.ckpt$/\1/p' | sort -n | tail -1
}

chain_tip_step() {     # <arm> -> the arm's recorded chain tip (or its anchor, or 0)
  "$PY" - "$REGISTRY" "$1" <<'PY' 2>/dev/null
import json, sys
row = (json.load(open(sys.argv[1])).get("arms") or {}).get(sys.argv[2]) or {}
chain = row.get("chain") or []
print(int((chain[-1].get("final_step") if chain else row.get("final_step")) or 0))
PY
}

leg_manifest_for() {   # <arm> <chunk_end> -> the ONE launcher manifest for that chunk
  local arm="$1" want="$2" f ce
  local -a hits=()
  for f in "$EXPDIR"/fa_orbit_*_"${arm}"_8x8_jid*_manifest.txt; do
    [ -f "$f" ] || continue
    ce="$(awk '/^chunk_end /{print $2; exit}' "$f" 2>/dev/null)"
    [ "$ce" = "$want" ] && hits+=("$f")
  done
  [ "${#hits[@]}" -eq 1 ] || return 1
  printf '%s\n' "${hits[0]}"
}

process_arm() {
  local arm="$1"
  if [ -n "${ST[frozen_$arm]:-}" ]; then
    log "${arm}: FROZEN (${ST[frozen_$arm]}) — skipping; clear it in ${STATE} to resume"
    return 0
  fi

  local queued
  queued="$(squeue -h -u "$USER" -n "exp11-${arm}-train" -o '%i %T' 2>/dev/null)"

  # settle the leg we last submitted, once it has left the queue
  local last="${ST[lastjob_$arm]:-}" st
  if [ -n "$last" ] && [ -z "$queued" ]; then
    st="$(sacct -X -n -P -j "$last" -o State 2>/dev/null | head -1 | awk '{print $1}')"
    case "$st" in
      COMPLETED)          clear_fail "$arm" "job ${last} COMPLETED" ;;
      FAILED|TIMEOUT|NODE_FAIL|OUT_OF_MEMORY|BOOT_FAIL|DEADLINE|PREEMPTED|CANCELLED)
                          bump_fail "$arm" "job ${last} ended ${st}" ;;
      "")                 log "${arm}: sacct reports nothing for job ${last} — not counted either way" ;;
      *)                  log "${arm}: job ${last} ended ${st} — not counted" ;;
    esac
    unset "ST[lastjob_$arm]"
    [ -n "${ST[frozen_$arm]:-}" ] && return 0
  fi

  if [ -n "$queued" ]; then
    log "${arm}: live job (${queued//$'\n'/; }) — nothing to do"
    return 0
  fi

  local ckdir="${OUTPUT_ROOT}/exp11_${arm}/FLAC_exp11_${arm}/exp11_${arm}/checkpoints"
  local S; S="$(newest_ckpt_step "$ckdir")"
  if [ -z "$S" ]; then
    log "${arm}: no epoch=*-step=*.ckpt under ${ckdir} — nothing to resume, skipping"
    return 0
  fi
  if [ "$S" -ge "$TARGET" ]; then
    if [ "${ST[done_$arm]:-}" != "$S" ]; then
      ST["done_$arm"]="$S"
      log "${arm}: DONE — newest checkpoint is step ${S} >= target ${TARGET}"
    fi
    return 0
  fi

  # a chunk that finished but was never recorded must be recorded FIRST: the next
  # chunk's preflight binds to the recorded chain, not to whatever is on disk.
  if [ "$S" -gt "$ANCHOR_STEP" ]; then
    local tip; tip="$(chain_tip_step "$arm")"
    is_num "$tip" || { log "${arm}: could not read the chain tip from ${REGISTRY}"; bump_fail "$arm" "registry unreadable"; return 0; }
    if [ "$tip" -lt "$S" ]; then
      local man
      if ! man="$(leg_manifest_for "$arm" "$S")"; then
        log "${arm}: no single launcher manifest with chunk_end ${S} in ${EXPDIR} (zero or ambiguous) — cannot record the finished chunk"
        bump_fail "$arm" "manifest for chunk_end ${S} not uniquely identifiable"
        return 0
      fi
      local -a rec=("$RECORDER" "$arm" "$man" --registry "$REGISTRY")
      [ "$DRYRUN" = "1" ] && rec+=(--dry-run)
      log "${arm}: recording finished chunk ${tip} -> ${S} from $(basename "$man")"
      local out rc
      out="$("$PY" "${rec[@]}" 2>&1)"; rc=$?
      if [ "$rc" -ne 0 ]; then
        log "${arm}: RECORDER REFUSED (rc=${rc}): $(printf '%s' "$out" | head -4 | tr '\n' ' ')"
        bump_fail "$arm" "recorder rc=${rc}"
        return 0
      fi
      clear_fail "$arm" "chunk ${tip} -> ${S} recorded"
      if [ "$DRYRUN" = "1" ]; then
        log "${arm}: DRYRUN — the chunk link was validated but not written, so the next chunk is not yet admissible"
      fi
    fi
  fi

  # submit the next chunk through the ONLY sanctioned submitter
  local next=$(( S + CHUNK ))
  [ "$next" -gt "$TARGET" ] && next="$TARGET"
  local -a ck=("$ckdir"/*-step="${S}".ckpt)
  if [ "${#ck[@]}" -ne 1 ] || [ ! -f "${ck[0]}" ]; then
    log "${arm}: expected exactly one checkpoint file at step ${S} in ${ckdir}, found ${#ck[@]}"
    bump_fail "$arm" "ambiguous checkpoint at step ${S}"
    return 0
  fi
  log "${arm}: submitting chunk ${S} -> ${next}: bash ${SUBMITTER} ${arm} --resume ${ck[0]} --expected-step ${S} --chunk-end ${next} (DRYRUN=${DRYRUN})"
  local out rc jid
  out="$(DRYRUN="$DRYRUN" bash "$SUBMITTER" "$arm" --resume "${ck[0]}" \
          --expected-step "$S" --chunk-end "$next" 2>&1)"; rc=$?
  if [ "$rc" -ne 0 ]; then
    log "${arm}: SUBMIT REFUSED (rc=${rc}): $(printf '%s' "$out" | head -4 | tr '\n' ' ')"
    bump_fail "$arm" "submit rc=${rc}"
    return 0
  fi
  jid="$(printf '%s' "$out" | awk '/^submitted /{print $NF}')"
  if [ -n "$jid" ]; then
    ST["lastjob_$arm"]="$jid"
    log "${arm}: submitted chunk ${S} -> ${next} as job ${jid}"
  else
    log "${arm}: submitter returned 0 without a job id (DRYRUN=${DRYRUN}) — nothing queued"
  fi
  return 0
}

IFS=',' read -r -a ARM_LIST <<< "$ARMS"
log "=== chunk watchdog start: arms ${ARMS} chunk ${CHUNK} target ${TARGET} poll ${POLL}s max_consec_fail ${MAX_CONSEC_FAIL} oneshot ${ONESHOT} dryrun ${DRYRUN} ==="
while :; do
  load_state
  for ARM in "${ARM_LIST[@]}"; do
    process_arm "$ARM"
  done
  save_state || log "WARNING: the state file could not be written — failure counters will not persist"

  REMAINING=0
  for ARM in "${ARM_LIST[@]}"; do
    [ -n "${ST[done_$ARM]:-}" ] && continue
    [ -n "${ST[frozen_$ARM]:-}" ] && continue
    REMAINING=$((REMAINING + 1))
  done
  if [ "$REMAINING" -eq 0 ]; then
    log "=== every arm is DONE or FROZEN — watchdog exiting ==="
    break
  fi
  [ "$ONESHOT" = "1" ] && { log "=== ONESHOT: one pass complete, exiting ==="; break; }
  sleep "$POLL"
done
exit 0

codex
The first pass confirms the intended fail-closed checks are present. I’m now checking the less obvious failure modes: registry-lock lifetime, watchdog instance races, manifest lookup ambiguity, scheduler-state settling, and whether launcher parity actually covers `STOPSTEP`.
exec
/usr/bin/bash -lc "nl -ba worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_record_restart.py | sed -n '1,560p'
nl -ba worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_producer_manifest.py | sed -n '1,320p'
nl -ba worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_chunk_watchdog.sh | sed -n '1,380p'" in /n/fs/gatrdp/codespace/FLAC
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
    44	"""
    45	import argparse
    46	import fcntl
    47	import glob
    48	import hashlib
    49	import json
    50	import os
    51	import re
    52	import sys
    53	from datetime import datetime, timezone
    54	
    55	HERE = os.path.dirname(os.path.abspath(__file__))
    56	sys.path.insert(0, HERE)
    57	import fa_orbit_producer_manifest as pm            # noqa: E402
    58	from fa_orbit_ckpt_preflight import canonical_ckpt_dir    # noqa: E402
    59	
    60	PIN_RE = re.compile(r'^(PINNED_[A-Z0-9_]+)=(?:"([^"]*)"|(\S+))')
    61	# The pinned checkpoint cadence: a chunk may only end ON a saved checkpoint,
    62	# otherwise the next chunk has nothing to resume.
    63	CHUNK_STEP = 2500
    64	
    65	
    66	def read_pins(launcher):
    67	    """The launcher's own PINNED_* values, so the recorder cannot drift from them."""
    68	    pins = {}
    69	    with open(launcher) as fh:
    70	        for line in fh:
    71	            m = PIN_RE.match(line)
    72	            if m:
    73	                pins[m.group(1)] = m.group(2) if m.group(2) is not None else m.group(3)
    74	    return pins
    75	
    76	
    77	def parse_manifest(path):
    78	    raw = open(path, "rb").read()
    79	    man = {}
    80	    for line in raw.decode().splitlines():
    81	        line = line.strip()
    82	        if line and not line.startswith("#"):
    83	            k, _, rest = line.partition(" ")
    84	            man[k] = rest.strip()
    85	    return raw, man
    86	
    87	
    88	def kvs(man, key):
    89	    f = (f"{key} " + man.get(key, "")).split()
    90	    return {f[i]: f[i + 1] for i in range(0, len(f) - 1, 2)}
    91	
    92	
    93	def check_identity(arm, man, initial, pins, repo_root,
    94	                   want_anchor=None, want_step=None, want_time=None):
    95	    """Every field of the RESTART manifest, against the audited INITIAL row + Q10 pins.
    96	
    97	    A CHUNK leg proves the same identity but resumes the tip of the recorded
    98	    chain rather than the audited 40k anchor, and is walled by the arm's CHUNK
    99	    pin rather than its RESTART pin, so the caller may override those three
   100	    expectations. Everything else is identical for both leg kinds."""
   101	    jk, ak, rk = kvs(man, "job"), kvs(man, "arm"), kvs(man, "resume_ckpt")
   102	    tk = kvs(man, "time_limit")
   103	    problems = []
   104	    anchor = want_anchor if want_anchor is not None else initial.get("final_ckpt_sha256")
   105	    final_step = want_step if want_step is not None else initial.get("final_step")
   106	    if not anchor:
   107	        problems.append(f"{arm} has no audited final_ckpt_sha256 to chain from — audit the "
   108	                        "INITIAL run's final checkpoint before recording a leg")
   109	    if jk.get("mode") != "RESTART":
   110	        problems.append(f"manifest mode is {jk.get('mode')!r}, not RESTART")
   111	    for field, got in (("job", jk.get("job")), ("launch_uuid", jk.get("launch_uuid")),
   112	                       ("commit", man.get("commit"))):
   113	        if not got:
   114	            problems.append(f"manifest records no {field} — a leg with no identity is not a record")
   115	    if jk.get("job") and initial.get("job") == jk.get("job"):
   116	        problems.append(f"manifest job {jk.get('job')} IS the INITIAL job — that is the launch "
   117	                        "already registered, not a restart leg")
   118	    for label, got, want in (("arm", ak.get("arm"), arm),
   119	                             ("rung", ak.get("rung"), initial.get("rung")),
   120	                             ("micro", ak.get("micro"), pins.get("PINNED_MB")),
   121	                             ("ngpu", ak.get("ngpu"), pins.get("PINNED_NGPU")),
   122	                             ("config_sha256", man.get("config_sha256"), initial.get("config_sha256")),
   123	                             ("vae_sha256", man.get("vae_sha256"), initial.get("vae_sha256")),
   124	                             ("p0_manifest_sha256", man.get("p0_manifest_sha256"),
   125	                              initial.get("p0_manifest_sha256")),
   126	                             ("save_dir", man.get("save_dir"), initial.get("save_dir"))):
   127	        if got != want:
   128	            problems.append(f"manifest {label} {got!r} != the audited INITIAL run's {want!r}")
   129	    if ak.get("rung") != pins.get("PINNED_RUNG"):
   130	        problems.append(f"manifest rung {ak.get('rung')!r} != the pinned {pins.get('PINNED_RUNG')!r}")
   131	    if ak.get("max_steps") != pins.get("PINNED_MAXSTEPS"):
   132	        problems.append(f"manifest max_steps {ak.get('max_steps')!r} != the Q10 budget pin "
   133	                        f"{pins.get('PINNED_MAXSTEPS')!r}")
   134	    step_label = "audited final step" if want_step is None else "recorded chain tip's final step"
   135	    if final_step is not None and str(rk.get("expected_step")) != str(final_step):
   136	        problems.append(f"manifest expected_step {rk.get('expected_step')!r} != the {step_label} "
   137	                        f"{final_step!r} — a leg resumes where the run it continues ended")
   138	    pin_label = "RESTART wall pin"
   139	    if want_time is None:
   140	        want_time = pins.get(f"PINNED_TIME_LIMIT_RESTART_{arm}")
   141	    else:
   142	        pin_label = "CHUNK wall pin"
   143	    if tk.get("time_limit") != want_time:
   144	        problems.append(f"manifest time_limit {tk.get('time_limit')!r} != the arm's {pin_label} "
   145	                        f"{want_time!r}")
   146	    if int(initial.get("training_seed", -1)) != 42:
   147	        problems.append(f"registered training seed {initial.get('training_seed')!r} != 42")
   148	    # the config the leg names must still hash to the audited value
   149	    cfg_path = man.get("model_config", "")
   150	    if not cfg_path or not os.path.isfile(cfg_path):
   151	        problems.append(f"manifest model_config {cfg_path!r} does not exist")
   152	    elif hashlib.sha256(open(cfg_path, "rb").read()).hexdigest() != initial.get("config_sha256"):
   153	        problems.append(f"{cfg_path} no longer hashes to the audited config_sha256")
   154	    # --- the resume file itself: MUST exist, MUST be canonical, ALWAYS re-hashed --
   155	    resume_path = (man.get("resume_ckpt", "").split() or [""])[0]
   156	    resume_real = ""
   157	    if not resume_path or resume_path == "<none>":
   158	        problems.append("manifest records no resume_ckpt — a RESTART that resumed nothing is not "
   159	                        "a continuation of the audited run")
   160	    elif not os.path.isfile(resolve(repo_root, resume_path)):
   161	        problems.append(f"the resume checkpoint {resume_path} does not exist — the recorder does "
   162	                        "NOT accept the manifest's claimed hash in its place")
   163	    else:
   164	        resume_real = resolve(repo_root, resume_path)
   165	        canon = canonical_ckpt_dir(initial.get("save_dir", ""), arm, repo_root)
   166	        if os.path.realpath(os.path.dirname(resume_real)) != canon:
   167	            problems.append(f"the resume checkpoint is not in the audited launch's canonical "
   168	                            f"directory {canon}")
   169	        got = pm.sha256_file(resume_real)
   170	        if anchor and got != anchor:
   171	            problems.append(f"the resume file on disk hashes {got[:12]}, not the audited "
   172	                            f"{anchor[:12]} — this leg does not continue that run")
   173	        if rk.get("resume_ckpt_sha256") != got:
   174	            problems.append(f"manifest resume_ckpt_sha256 {str(rk.get('resume_ckpt_sha256'))[:12]} "
   175	                            f"!= the file's actual {got[:12]}")
   176	    return problems, resume_real
   177	
   178	
   179	def resolve(root, path):
   180	    return path if os.path.isabs(path) else os.path.join(root, path)
   181	
   182	
   183	def chunk_end_of(man):
   184	    """The leg's per-job stop step as written by the launcher, or None.
   185	
   186	    A manifest with no `chunk_end` line (or the literal `<none>`) is a
   187	    whole-budget RESTART leg and takes the original recording path untouched."""
   188	    v = (man.get("chunk_end", "").split() or [""])[0]
   189	    return None if not v or v == "<none>" else v
   190	
   191	
   192	def utc_now():
   193	    return datetime.now(timezone.utc).isoformat(timespec="seconds")
   194	
   195	
   196	def record_chunk(args, arm, reg, initial, man, man_sha, pins, chunk_raw):
   197	    """Append ONE chain link for a chunked leg: <resume_step> -> <chunk_end>.
   198	
   199	    The link is what makes the NEXT chunk admissible (preflight --chain), so it
   200	    is written only when this leg provably resumed the tip of the chain and its
   201	    endpoint checkpoint is on disk, unambiguous, and hashed here from that disk.
   202	    """
   203	    jk, rk = kvs(man, "job"), kvs(man, "resume_ckpt")
   204	    job = jk.get("job")
   205	    problems = []
   206	    if args.extend:
   207	        problems.append("--extend extends a producer manifest and has no meaning for a CHUNK leg; "
   208	                        "each chunk is recorded once, as one immutable chain link")
   209	    if not job:
   210	        problems.append("manifest records no job — a leg with no identity is not a record")
   211	
   212	    # --- the chunk boundary itself ------------------------------------------
   213	    chunk_end = resume_step = None
   214	    try:
   215	        chunk_end = int(chunk_raw)
   216	    except (TypeError, ValueError):
   217	        problems.append(f"manifest chunk_end {chunk_raw!r} is not an integer")
   218	    try:
   219	        resume_step = int(rk.get("expected_step"))
   220	    except (TypeError, ValueError):
   221	        problems.append(f"manifest expected_step {rk.get('expected_step')!r} is not an integer")
   222	    budget = None
   223	    try:
   224	        budget = int(pins.get("PINNED_MAXSTEPS"))
   225	    except (TypeError, ValueError):
   226	        problems.append(f"the launcher's PINNED_MAXSTEPS {pins.get('PINNED_MAXSTEPS')!r} is not an "
   227	                        "integer — the chunk boundary cannot be checked against the budget")
   228	    if chunk_end is not None:
   229	        if chunk_end % CHUNK_STEP:
   230	            problems.append(f"manifest chunk_end {chunk_end} is not a multiple of {CHUNK_STEP} (the "
   231	                            "pinned checkpoint cadence: a chunk must end ON a checkpoint)")
   232	        if resume_step is not None and chunk_end <= resume_step:
   233	            problems.append(f"manifest chunk_end {chunk_end} does not exceed the resume step "
   234	                            f"{resume_step} — the leg would produce no new checkpoint")
   235	        if budget is not None and chunk_end > budget:
   236	            problems.append(f"manifest chunk_end {chunk_end} exceeds the pinned budget {budget}")
   237	
   238	    # --- which link (or the audited anchor) this chunk continues ------------
   239	    chain = initial.get("chain") or []
   240	    mine = [i for i, link in enumerate(chain) if link.get("job") == job]
   241	    if len(mine) > 1:
   242	        raise SystemExit(f"{arm} has {len(mine)} chain links claiming job {job} — the registry is "
   243	                         "inconsistent; fix it before recording")
   244	    at = mine[0] if mine else len(chain)
   245	    prev = chain[at - 1] if at > 0 else None
   246	    if prev is None:
   247	        want_anchor = want_step = None                 # the audited INITIAL anchor
   248	        prev_desc = f"the audited INITIAL anchor at step {initial.get('final_step')}"
   249	    else:
   250	        want_anchor, want_step = prev.get("final_ckpt_sha256"), prev.get("final_step")
   251	        prev_desc = f"chain link job {prev.get('job')} ending at step {prev.get('final_step')}"
   252	    want_time = pins.get(f"PINNED_TIME_LIMIT_CHUNK_{arm}")
   253	    if not want_time:
   254	        problems.append(f"the launcher carries no PINNED_TIME_LIMIT_CHUNK_{arm} pin, so this "
   255	                        "chunk leg's wall time cannot be bound to a pin")
   256	    ident, resume_real = check_identity(arm, man, initial, pins, args.repo_root,
   257	                                        want_anchor=want_anchor, want_step=want_step,
   258	                                        want_time=want_time)
   259	    problems += ident
   260	
   261	    # --- the endpoint checkpoint this leg produced --------------------------
   262	    endpoint = None
   263	    if chunk_end is not None:
   264	        ckpt_dir = canonical_ckpt_dir(initial.get("save_dir", ""), arm, args.repo_root)
   265	        hits = sorted(glob.glob(os.path.join(ckpt_dir, f"*-step={chunk_end}.ckpt")))
   266	        if len(hits) != 1:
   267	            problems.append(f"expected exactly 1 checkpoint at the chunk end step {chunk_end} in "
   268	                            f"{ckpt_dir}, found {len(hits)}"
   269	                            + (": " + ", ".join(os.path.basename(h) for h in hits) if hits else ""))
   270	        else:
   271	            endpoint = hits[0]
   272	    if problems:
   273	        print("RECORD REFUSED:")
   274	        for p in problems:
   275	            print(f"  !! {p}")
   276	        return 2
   277	
   278	    # check_identity already re-hashed the resume file and proved it IS this hash
   279	    resume_sha = want_anchor if want_anchor is not None else initial.get("final_ckpt_sha256")
   280	    link = {"job": job, "launch_uuid": jk.get("launch_uuid"),
   281	            "manifest_path": os.path.abspath(args.manifest), "manifest_sha256": man_sha,
   282	            "resume_step": resume_step, "resume_ckpt_sha256": resume_sha,
   283	            "final_step": chunk_end, "final_ckpt_sha256": pm.sha256_file(endpoint),
   284	            "recorded_utc": utc_now()}
   285	
   286	    if mine:
   287	        old = chain[at]
   288	        differing = sorted(k for k in link if k != "recorded_utc"
   289	                           and str(old.get(k)) != str(link[k]))
   290	        if differing:
   291	            print("RECORD REFUSED:")
   292	            print(f"  !! {arm} job {job} is ALREADY a chain link whose content differs "
   293	                  f"({', '.join(differing)}) — a recorded link is immutable, and rewriting one "
   294	                  "would re-parent every chunk recorded after it")
   295	            return 2
   296	        print(f"{arm} chunk link job {job} ({resume_step} -> {chunk_end}) is already recorded, "
   297	              "byte-identical — no-op")
   298	        return 0
   299	
   300	    initial.setdefault("chain", []).append(link)
   301	    if not args.dry_run:
   302	        pm.write_atomic(args.registry, reg)
   303	    print(f"recorded {arm} chunk link job {job}: {resume_step} -> {chunk_end}, continuing "
   304	          f"{prev_desc} " + ("(dry run, nothing written)" if args.dry_run else "(published)"))
   305	    print(f"  resume {str(resume_sha)[:12]} -> endpoint {link['final_ckpt_sha256'][:12]} "
   306	          f"({pm.rel_to(args.repo_root, endpoint)})")
   307	    print(f"  {arm} chain is now {len(initial['chain'])} link(s), tip at step {chunk_end}"
   308	          f" of the {budget} budget")
   309	    return 0
   310	
   311	
   312	def main(argv=None):
   313	    ap = argparse.ArgumentParser(description="record an exp_11 RESTART leg")
   314	    ap.add_argument("arm")
   315	    ap.add_argument("manifest")
   316	    ap.add_argument("--registry", default=os.path.join(HERE, "arm_launch_registry.json"))
   317	    ap.add_argument("--launcher", default=os.path.join(HERE, "fa_orbit_train.sbatch"),
   318	                    help="where the Q10 pins are read from")
   319	    ap.add_argument("--producer-dir", default=HERE,
   320	                    help="where the per-leg producer manifests are published")
   321	    # HERE = <repo>/worklog/worklog_<user>/exp_11_fa_orbit_claude
   322	    ap.add_argument("--repo-root", default=os.path.dirname(os.path.dirname(os.path.dirname(HERE))),
   323	                    help="root the manifest's relative paths resolve against")
   324	    ap.add_argument("--extend", action="store_true",
   325	                    help="this leg is already recorded: extend its producer manifest only")
   326	    ap.add_argument("--rehash-all", action="store_true",
   327	                    help="re-hash published checkpoints too (full audit, expensive)")
   328	    ap.add_argument("--dry-run", action="store_true", help="validate and report, publish nothing")
   329	    args = ap.parse_args(argv)
   330	
   331	    arm = args.arm
   332	    pins = read_pins(args.launcher)
   333	    if not pins.get("PINNED_MAXSTEPS"):
   334	        raise SystemExit(f"no PINNED_* values found in {args.launcher}")
   335	
   336	    # One writer at a time, and the lock is the registry's own DIRECTORY: no lock
   337	    # file to leave behind in a tracked tree, and it still covers the tmp+rename.
   338	    store = os.path.dirname(os.path.abspath(args.registry)) or "."
   339	    lock_fd = os.open(store, os.O_RDONLY)
   340	    try:
   341	        fcntl.flock(lock_fd, fcntl.LOCK_EX)
   342	        return record(args, arm, pins)
   343	    finally:
   344	        os.close(lock_fd)
   345	
   346	
   347	def record(args, arm, pins):
   348	    reg = json.load(open(args.registry))
   349	    initial = reg.get("arms", {}).get(arm)
   350	    if initial is None:
   351	        raise SystemExit(f"{arm} has no INITIAL registry entry")
   352	
   353	    raw, man = parse_manifest(args.manifest)
   354	    man_sha = hashlib.sha256(raw).hexdigest()
   355	    # Round 5: a leg that carries a `chunk_end` line stopped at a chunk boundary
   356	    # and is recorded as a CHAIN LINK, not as a whole-budget RESTART row.
   357	    chunk_raw = chunk_end_of(man)
   358	    if chunk_raw is not None:
   359	        return record_chunk(args, arm, reg, initial, man, man_sha, pins, chunk_raw)
   360	    problems, resume_real = check_identity(arm, man, initial, pins, args.repo_root)
   361	    jk, ak, rk = kvs(man, "job"), kvs(man, "arm"), kvs(man, "resume_ckpt")
   362	    job = jk.get("job")
   363	
   364	    legs = reg.setdefault("restarts", {}).setdefault(arm, [])
   365	    same = [l for l in legs if l.get("job") == job or l.get("launch_uuid") == jk.get("launch_uuid")
   366	            or l.get("manifest_sha256") == man_sha]
   367	    if same and not args.extend:
   368	        raise SystemExit(f"{arm} job {job} is ALREADY recorded ({len(same)} matching leg(s)) — "
   369	                         "one leg, one row; use --extend to extend its producer manifest")
   370	    if len(same) > 1:
   371	        problems.append(f"{len(same)} registry rows already claim this leg — the registry is "
   372	                        "inconsistent; fix it before recording")
   373	    if args.extend and not same:
   374	        problems.append(f"--extend given but {arm} job {job} is not recorded yet")
   375	    if problems:
   376	        print("RECORD REFUSED:")
   377	        for p in problems:
   378	            print(f"  !! {p}")
   379	        return 2
   380	
   381	    anchor = initial["final_ckpt_sha256"]
   382	    producer = pm.manifest_name(arm, job)
   383	    row = {
   384	        "manifest_path": args.manifest, "manifest_sha256": man_sha,
   385	        "job": job, "mode": "RESTART", "launch_uuid": jk.get("launch_uuid"),
   386	        "arm": arm, "commit": man.get("commit"), "rung": ak.get("rung"),
   387	        "config_sha256": man.get("config_sha256"), "save_dir": man.get("save_dir"),
   388	        "resume_ckpt": resume_real, "resume_ckpt_sha256": anchor,
   389	        "expected_step": rk.get("expected_step"), "max_steps": ak.get("max_steps"),
   390	        "time_limit": kvs(man, "time_limit").get("time_limit"),
   391	        "producer_manifest": producer, "chains_to": anchor,
   392	        "recorded_at": pm._now(),
   393	    }
   394	    header = {"arm": arm, "job": job, "launch_uuid": jk.get("launch_uuid"), "mode": "RESTART",
   395	              "commit": man.get("commit"), "resume_ckpt_sha256": anchor,
   396	              "expected_step": rk.get("expected_step"), "max_steps": ak.get("max_steps"),
   397	              "save_dir": man.get("save_dir"), "config_sha256": man.get("config_sha256"),
   398	              "chains_to": anchor, "leg_manifest_sha256": man_sha}
   399	
   400	    ckpt_dir = canonical_ckpt_dir(initial["save_dir"], arm, args.repo_root)
   401	    prod_path = os.path.join(args.producer_dir, producer)
   402	    known = (pm.load(prod_path) or {}).get("checkpoints", {})
   403	    found, scan_problems = pm.scan_checkpoints(
   404	        ckpt_dir, int(rk["expected_step"]), int(ak["max_steps"]), known=known,
   405	        rehash_all=args.rehash_all, repo_root=args.repo_root)
   406	    added, kept, pub_problems = pm.publish(prod_path, header, found, dry_run=args.dry_run)
   407	    if scan_problems or pub_problems:
   408	        print("RECORD REFUSED:")
   409	        for p in scan_problems + pub_problems:
   410	            print(f"  !! {p}")
   411	        return 2
   412	
   413	    if args.extend:
   414	        for i, leg in enumerate(legs):
   415	            if leg.get("job") == job:
   416	                legs[i] = {**leg, "producer_manifest": producer}
   417	    else:
   418	        legs.append(row)
   419	    if not args.dry_run:
   420	        pm.write_atomic(args.registry, reg)
   421	    verb = "extended" if args.extend else "recorded"
   422	    print(f"{verb} {arm} RESTART job {job} chaining to {anchor[:12]} "
   423	          f"({'dry run, nothing written' if args.dry_run else 'published'})")
   424	    print(f"  producer manifest {producer}: {len(added)} checkpoint(s) added, "
   425	          f"{len(kept)} already published"
   426	          + (f" (steps {added[0]}..{added[-1]})" if added else ""))
   427	    return 0
   428	
   429	
   430	if __name__ == "__main__":
   431	    sys.exit(main())
     1	#!/usr/bin/env python3
     2	"""Per-leg PRODUCER manifests: which checkpoints a RESTART leg actually produced.
     3	
     4	Re-pin review, required fix 2. The >40k lineage gate was EXISTENTIAL: once any
     5	registry leg for an arm carried `mode=RESTART` and the right 40k resume hash,
     6	every later checkpoint in that arm's canonical directory passed, because nothing
     7	bound the evaluated checkpoint's own bytes to the leg that produced it. A
     8	same-config checkpoint from a wrong restart, copied into the canonical directory,
     9	was admissible.
    10	
    11	DESIGN CHOICE (the review allowed two; this is the second, and why).
    12	  Rejected: the restart leg's JOB hashes each checkpoint as it saves. That is the
    13	  tighter binding, but it means editing fa_orbit_train.sbatch's training path --
    14	  a hashing sidecar running beside torchrun -- while jobs 3662828-30 sit queued
    15	  against that launcher, which this round is forbidden to do (and which would put
    16	  sustained multi-GB reads next to a live training job on a shared filesystem).
    17	  Chosen: the RECORDER captures the leg's checkpoint inventory (step -> sha256,
    18	  re-hashed from disk) and publishes it into an APPEND-ONLY, COMMITTED per-leg
    19	  file next to the audited registry. The screen re-hashes the checkpoint it is
    20	  about to evaluate and requires an exact step -> sha256 -> leg match.
    21	
    22	Why that is still immutable evidence: the file lives in the tracked experiment
    23	directory, exactly like arm_launch_registry.json, and screens read it from the
    24	PINNED worktree -- so a row can only enter it through a commit, and only a
    25	commit that is in the campaign pin can be used as evidence. Within the file the
    26	recorder is append-only: a step already published may never change its sha256 or
    27	its path, and the header (arm, job, launch uuid, resume anchor, save-dir, config
    28	sha, budget) may never change at all.
    29	"""
    30	import hashlib
    31	import json
    32	import os
    33	import re
    34	import tempfile
    35	from datetime import datetime
    36	
    37	CKPT_RE = re.compile(r"^epoch=(\d+)-step=(\d+)\.ckpt$")
    38	# Header fields that identify the LEG. Once published they are frozen: a
    39	# republish that disagrees on any of them is a different leg wearing this file's
    40	# name, not an extension of it.
    41	HEADER_FIELDS = ("arm", "job", "launch_uuid", "mode", "commit", "resume_ckpt_sha256",
    42	                 "expected_step", "max_steps", "save_dir", "config_sha256",
    43	                 "chains_to", "leg_manifest_sha256")
    44	
    45	
    46	def sha256_file(path):
    47	    h = hashlib.sha256()
    48	    with open(path, "rb") as fh:
    49	        for chunk in iter(lambda: fh.read(1 << 20), b""):
    50	            h.update(chunk)
    51	    return h.hexdigest()
    52	
    53	
    54	def manifest_name(arm, job):
    55	    """The per-leg file name. Flat in the experiment directory, like the registry
    56	    and the backfill manifest, so the launcher/submitter drift gates (`$EXPDIR/*.json`)
    57	    cover it."""
    58	    return f"fa_orbit_producer_{arm}_job{job}.json"
    59	
    60	
    61	def load(path):
    62	    if not os.path.isfile(path):
    63	        return None
    64	    with open(path) as fh:
    65	        return json.load(fh)
    66	
    67	
    68	def rel_to(root, path):
    69	    """Repo-relative when possible (portable across the pinned worktrees), else absolute."""
    70	    real, root_real = os.path.realpath(path), os.path.realpath(root)
    71	    return os.path.relpath(real, root_real) if real.startswith(root_real + os.sep) else real
    72	
    73	
    74	def resolve(root, path):
    75	    return path if os.path.isabs(path) else os.path.join(root, path)
    76	
    77	
    78	def scan_checkpoints(ckpt_dir, after_step, max_step, known=None, rehash_all=False, repo_root="."):
    79	    """Re-hash the leg's checkpoints from DISK: {str(step): {path, sha256, bytes}}.
    80	
    81	    Only steps strictly after the resume point and no further than the budget are
    82	    the leg's own output -- the resume checkpoint itself belongs to the INITIAL
    83	    run and is already anchored in the registry. Steps already published are not
    84	    re-read by default (they are immutable evidence, and each is ~724 MB on a
    85	    shared filesystem); ``rehash_all`` forces a full audit."""
    86	    known = known or {}
    87	    out, problems = {}, []
    88	    if not os.path.isdir(ckpt_dir):
    89	        return {}, [f"checkpoint directory not found: {ckpt_dir}"]
    90	    for name in sorted(os.listdir(ckpt_dir)):
    91	        m = CKPT_RE.match(name)
    92	        if not m:
    93	            continue
    94	        step = int(m.group(2))
    95	        if step <= after_step or step > max_step:
    96	            continue
    97	        path = os.path.join(ckpt_dir, name)
    98	        key = str(step)
    99	        if key in out:
   100	            problems.append(f"two checkpoint files claim step {step} in {ckpt_dir}")
   101	            continue
   102	        if key in known and not rehash_all:
   103	            out[key] = dict(known[key])
   104	            continue
   105	        out[key] = {"path": rel_to(repo_root, path), "sha256": sha256_file(path),
   106	                    "bytes": os.path.getsize(path)}
   107	    return out, problems
   108	
   109	
   110	def publish(path, header, checkpoints, dry_run=False):
   111	    """Append-only publication of a leg's inventory. Returns (added, kept, problems).
   112	
   113	    The header is frozen and a published step may not change; the write is
   114	    tmp+rename in the destination directory, so a reader never sees a partial
   115	    file and a failed write leaves the previous one intact."""
   116	    problems, added, kept = [], [], []
   117	    old = load(path)
   118	    if old is not None:
   119	        for field in HEADER_FIELDS:
   120	            if str(old.get(field)) != str(header.get(field)):
   121	                problems.append(f"{os.path.basename(path)} is already published with "
   122	                                f"{field}={old.get(field)!r}, not {header.get(field)!r} — a "
   123	                                "producer manifest is immutable")
   124	        for step, entry in sorted(old.get("checkpoints", {}).items()):
   125	            new = checkpoints.get(step)
   126	            if new is None:
   127	                continue        # a published step whose file is gone stays published
   128	            if new["sha256"] != entry["sha256"]:
   129	                problems.append(f"step {step} is published with sha256 {entry['sha256'][:12]} but "
   130	                                f"now hashes {new['sha256'][:12]} — a published checkpoint may "
   131	                                "never change")
   132	            elif os.path.basename(str(new["path"])) != os.path.basename(str(entry["path"])):
   133	                problems.append(f"step {step} is published at {entry['path']} but now at "
   134	                                f"{new['path']}")
   135	    if problems:
   136	        return [], [], problems
   137	    merged = dict((old or {}).get("checkpoints", {}))
   138	    for step, entry in checkpoints.items():
   139	        (kept if step in merged else added).append(step)
   140	        merged.setdefault(step, entry)
   141	    doc = {k: header.get(k) for k in HEADER_FIELDS}
   142	    doc["_comment"] = [
   143	        "APPEND-ONLY producer manifest for one exp_11 RESTART leg (re-pin fix 2).",
   144	        "Every checkpoint this leg produced, re-hashed from disk by",
   145	        "fa_orbit_record_restart.py. The screen admits a >40k checkpoint only when",
   146	        "its own sha256 matches this file's entry for exactly that step, so a",
   147	        "valid restart row no longer vouches for any later same-config file.",
   148	        "A published step never changes; the header never changes.",
   149	    ]
   150	    doc["first_published"] = (old or {}).get("first_published") or _now()
   151	    doc["last_extended"] = _now()
   152	    doc["checkpoints"] = dict(sorted(merged.items(), key=lambda kv: int(kv[0])))
   153	    if not dry_run:
   154	        write_atomic(path, doc)
   155	    return sorted(added, key=int), sorted(kept, key=int), []
   156	
   157	
   158	def write_atomic(path, doc):
   159	    d = os.path.dirname(os.path.abspath(path)) or "."
   160	    fd, tmp = tempfile.mkstemp(prefix=".{}.".format(os.path.basename(path)), dir=d)
   161	    try:
   162	        with os.fdopen(fd, "w") as fh:
   163	            json.dump(doc, fh, indent=2)
   164	            fh.write("\n")
   165	            fh.flush()
   166	            os.fsync(fh.fileno())
   167	        os.replace(tmp, path)
   168	    except BaseException:
   169	        if os.path.exists(tmp):
   170	            os.unlink(tmp)
   171	        raise
   172	
   173	
   174	def _now():
   175	    return datetime.now().astimezone().isoformat(timespec="seconds")
   176	
   177	
   178	def validate_leg(leg, row, arm, step=None):
   179	    """Every field of a registry RESTART row, against the audited INITIAL row.
   180	
   181	    The old gate checked two of them (mode, resume sha) and only existentially."""
   182	    problems = []
   183	    anchor = row.get("final_ckpt_sha256")
   184	    for label, got, want in (("arm", leg.get("arm"), arm),
   185	                             ("mode", leg.get("mode"), "RESTART"),
   186	                             ("resume_ckpt_sha256", leg.get("resume_ckpt_sha256"), anchor),
   187	                             ("chains_to", leg.get("chains_to"), anchor),
   188	                             ("save_dir", leg.get("save_dir"), row.get("save_dir")),
   189	                             ("config_sha256", leg.get("config_sha256"), row.get("config_sha256"))):
   190	        if got != want:
   191	            problems.append(f"restart leg {label} {got!r} != the audited INITIAL row's {want!r}")
   192	    for field in ("job", "launch_uuid", "commit", "manifest_sha256", "producer_manifest"):
   193	        if not leg.get(field):
   194	            problems.append(f"restart leg records no {field}")
   195	    try:
   196	        expected, budget = int(leg["expected_step"]), int(leg["max_steps"])
   197	    except (KeyError, TypeError, ValueError):
   198	        problems.append(f"restart leg has no integer expected_step/max_steps "
   199	                        f"({leg.get('expected_step')!r}/{leg.get('max_steps')!r})")
   200	        return problems
   201	    if str(row.get("final_step")) != str(expected):
   202	        problems.append(f"restart leg resumed at step {expected}, but the audited INITIAL run "
   203	                        f"ended at {row.get('final_step')}")
   204	    if budget <= expected:
   205	        problems.append(f"restart leg budget {budget} does not exceed its resume step {expected}")
   206	    if step is not None and not expected < int(step) <= budget:
   207	        problems.append(f"step {step} is outside this leg's output range "
   208	                        f"({expected} < step <= {budget})")
   209	    return problems
   210	
   211	
   212	def verify_chain(reg, arm, step, ckpt_path, ckpt_sha, base_dir, repo_root="."):
   213	    """Bind ONE checkpoint to the leg that produced it. Returns (problems, note).
   214	
   215	    ``base_dir`` is the directory the registry was read from, so the per-leg files
   216	    travel with it (pinned worktree, or a synthetic registry under test)."""
   217	    row = (reg.get("arms") or {}).get(arm)
   218	    if row is None:
   219	        return [f"{arm} is not in the audited launch registry"], ""
   220	    if not row.get("final_ckpt_sha256"):
   221	        return [f"{arm} has no audited final_ckpt_sha256, so a >40k checkpoint cannot be "
   222	                "chained to its INITIAL run"], ""
   223	    legs = (reg.get("restarts") or {}).get(arm) or []
   224	    if not legs:
   225	        return [f"checkpoint at step {step} is above the INITIAL budget but {arm} has no RESTART "
   226	                "entry in the audited registry — record the leg with fa_orbit_record_restart.py "
   227	                "first"], ""
   228	    why = []
   229	    for i, leg in enumerate(legs):
   230	        bad = validate_leg(leg, row, arm, step=step)
   231	        if bad:
   232	            why.append(f"leg {leg.get('job', i)}: " + "; ".join(bad))
   233	            continue
   234	        # The leg's OWN restart manifest is mutable evidence under gitignored
   235	        # outputs_FLAC, exactly like the INITIAL one the screen already re-hashes:
   236	        # it must still be there, and still be the bytes that were recorded.
   237	        leg_man = resolve(repo_root, str(leg.get("manifest_path")))
   238	        if not os.path.isfile(leg_man):
   239	            why.append(f"leg {leg.get('job')}: the registered RESTART manifest {leg_man} is gone")
   240	            continue
   241	        got = sha256_file(leg_man)
   242	        if got != leg.get("manifest_sha256"):
   243	            why.append(f"leg {leg.get('job')}: RESTART manifest {leg_man} now hashes {got[:12]}, "
   244	                       f"not the registered {str(leg.get('manifest_sha256'))[:12]} — it changed "
   245	                       "after it was recorded")
   246	            continue
   247	        man_path = resolve(base_dir, str(leg.get("producer_manifest")))
   248	        man = load(man_path)
   249	        if man is None:
   250	            why.append(f"leg {leg.get('job')}: producer manifest {man_path} is missing")
   251	            continue
   252	        head_bad = [f"producer manifest {f}={man.get(f)!r} != the registry leg's {leg.get(f)!r}"
   253	                    for f in ("arm", "job", "launch_uuid", "resume_ckpt_sha256", "chains_to")
   254	                    if str(man.get(f)) != str(leg.get(f))]
   255	        if str(man.get("leg_manifest_sha256")) != str(leg.get("manifest_sha256")):
   256	            head_bad.append("producer manifest is not the one this registry row published")
   257	        if head_bad:
   258	            why.append(f"leg {leg.get('job')}: " + "; ".join(head_bad))
   259	            continue
   260	        entry = (man.get("checkpoints") or {}).get(str(step))
   261	        if entry is None:
   262	            why.append(f"leg {leg.get('job')}: produced no checkpoint at step {step} "
   263	                       f"(published: {sorted((man.get('checkpoints') or {}), key=int)})")
   264	            continue
   265	        if entry.get("sha256") != ckpt_sha:
   266	            why.append(f"leg {leg.get('job')}: step {step} was published as "
   267	                       f"{str(entry.get('sha256'))[:12]}, this file hashes {ckpt_sha[:12]} — this "
   268	                       "is NOT the checkpoint that leg produced")
   269	            continue
   270	        if os.path.realpath(resolve(repo_root, str(entry.get("path")))) != os.path.realpath(ckpt_path):
   271	            why.append(f"leg {leg.get('job')}: step {step} was published at {entry.get('path')}, "
   272	                       f"not {ckpt_path}")
   273	            continue
   274	        return [], (f"producer binding OK: step {step} ({ckpt_sha[:12]}) was produced by RESTART "
   275	                    f"job {leg.get('job')}, which resumed the audited "
   276	                    f"{str(row['final_ckpt_sha256'])[:12]} and published it in "
   277	                    f"{os.path.basename(man_path)}")
   278	    return [f"no validated RESTART leg for {arm} published step {step} with sha256 "
   279	            f"{ckpt_sha[:12]} — " + " | ".join(why)], ""
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
    15	# (the recorder owns the registry), and never edits the tracked tree.
    16	#
    17	# FAIL-CLOSED. Every step is refused rather than guessed: an ambiguous manifest
    18	# or checkpoint, a recorder refusal, or a submitter refusal counts as a failure
    19	# for that arm, and MAX_CONSEC_FAIL consecutive failures FREEZE the arm until a
    20	# human removes its frozen_<ARM> line from the state file. A frozen arm is
    21	# skipped loudly, so the log says why nothing is progressing. The chunk chain
    22	# itself is guarded independently of this script: fa_orbit_ckpt_preflight.py
    23	# --chain refuses a chunk whose predecessor was never recorded, so a watchdog
    24	# bug cannot launder an unrecorded checkpoint into the lineage.
    25	#
    26	# USAGE (KEY=VALUE arguments only; every key whitelisted, no value is ever eval'd)
    27	#   bash fa_orbit_chunk_watchdog.sh                       # the pinned defaults
    28	#   bash fa_orbit_chunk_watchdog.sh ARMS=C4L,C8 POLL=600
    29	#   bash fa_orbit_chunk_watchdog.sh ONESHOT=1 DRYRUN=1    # one pass, submits nothing
    30	#
    31	#   ARMS=C4L,C8,C16,C32,VANL  CHUNK=2500  TARGET=100000  POLL=300
    32	#   MAX_CONSEC_FAIL=2  ONESHOT=0  DRYRUN=0
    33	#   TEST HOOKS (change no decision, only where the script reads/writes):
    34	#   OUTPUT_ROOT=outputs_FLAC  REGISTRY=<expdir>/arm_launch_registry.json
    35	#   STATE=<expdir>/.chunk_watchdog_state  LOG=<expdir>/fa_orbit_chunk_watchdog.log
    36	#
    37	# COST. One `squeue` and one `ls` per arm per poll (plus, once an arm is above
    38	# 40000, one short python read of the registry). Nothing else runs on the login
    39	# node; the training itself is entirely inside Slurm.
    40	# ============================================================================
    41	set -uo pipefail
    42	cd "$(git -C "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" rev-parse --show-toplevel)" || exit 3
    43	
    44	EXPDIR="worklog/worklog_yixun/exp_11_fa_orbit_claude"
    45	SUBMITTER="${EXPDIR}/fa_orbit_submit.sh"
    46	RECORDER="${EXPDIR}/fa_orbit_record_restart.py"
    47	PY=/n/fs/gatrdp/envs/flac/bin/python
    48	ANCHOR_STEP=40000                 # where every arm's INITIAL run ended
    49	
    50	ARMS="C4L,C8,C16,C32,VANL"; CHUNK=2500; TARGET=100000; POLL=300
    51	MAX_CONSEC_FAIL=2; ONESHOT=0; DRYRUN=0
    52	OUTPUT_ROOT="outputs_FLAC"
    53	REGISTRY="${EXPDIR}/arm_launch_registry.json"
    54	STATE="${EXPDIR}/.chunk_watchdog_state"
    55	LOG="${EXPDIR}/fa_orbit_chunk_watchdog.log"
    56	
    57	# --- argument parsing: whitelist the KEY, shape-check the VALUE, never eval ---
    58	reject()   { echo "$1" >&2; exit 2; }
    59	is_num()   { case "${1:-}" in ''|*[!0-9]*) return 1 ;; esac; }
    60	is_armset() {
    61	  local v="$1" a
    62	  [ -n "$v" ] || return 1
    63	  local IFS=,
    64	  for a in $v; do
    65	    case "$a" in C4L|C8|C16|C32|VANL) ;; *) return 1 ;; esac
    66	  done
    67	}
    68	for kv in "$@"; do
    69	  case "$kv" in *=*) ;; *) reject "argument '${kv}' is not KEY=VALUE" ;; esac
    70	  key="${kv%%=*}"; val="${kv#*=}"
    71	  case "$key" in
    72	    ARMS)            is_armset "$val" || reject "ARMS='${val}' is not a comma-separated list of C4L|C8|C16|C32|VANL" ;;
    73	    CHUNK|TARGET|POLL|MAX_CONSEC_FAIL)
    74	                     is_num "$val" || reject "${key}='${val}' is not a non-negative integer" ;;
    75	    ONESHOT|DRYRUN)  case "$val" in 0|1) ;; *) reject "${key}='${val}' must be 0 or 1" ;; esac ;;
    76	    OUTPUT_ROOT|REGISTRY|STATE|LOG)
    77	                     case "$val" in
    78	                       ''|*[!A-Za-z0-9/._-]*) reject "${key}='${val}' has unsafe characters" ;;
    79	                     esac ;;
    80	    *)               reject "unknown argument '${kv}' (expected ARMS=/CHUNK=/TARGET=/POLL=/MAX_CONSEC_FAIL=/ONESHOT=/DRYRUN=/OUTPUT_ROOT=/REGISTRY=/STATE=/LOG=)" ;;
    81	  esac
    82	  printf -v "$key" '%s' "$val"      # name whitelisted above; value never parsed
    83	done
    84	[ "$CHUNK" -gt 0 ] || reject "CHUNK must be positive"
    85	[ "$((CHUNK % 2500))" -eq 0 ] || reject "CHUNK=${CHUNK} is not a multiple of the 2500-step checkpoint cadence"
    86	[ "$TARGET" -gt "$ANCHOR_STEP" ] || reject "TARGET=${TARGET} must exceed the ${ANCHOR_STEP} anchor"
    87	[ "$((TARGET % 2500))" -eq 0 ] || reject "TARGET=${TARGET} is not a multiple of 2500"
    88	[ "$MAX_CONSEC_FAIL" -ge 1 ] || reject "MAX_CONSEC_FAIL must be at least 1"
    89	[ -f "$SUBMITTER" ] || reject "missing ${SUBMITTER}"
    90	[ -f "$RECORDER" ] || reject "missing ${RECORDER}"
    91	[ -f "$REGISTRY" ] || reject "missing ${REGISTRY}"
    92	
    93	log() { printf '%s %s\n' "$(date -Is)" "$1" >> "$LOG"; echo "$1"; }
    94	
    95	# --- state: a plain KV file a human can read and edit ------------------------
    96	# keys: fail_<ARM> <n> | frozen_<ARM> <when>|<reason> | lastjob_<ARM> <jid> |
    97	#       done_<ARM> <step>.  Removing a frozen_<ARM> line un-freezes that arm.
    98	declare -A ST=()
    99	load_state() {
   100	  ST=()
   101	  [ -f "$STATE" ] || return 0
   102	  local k v
   103	  while read -r k v; do
   104	    case "$k" in ''|\#*) continue ;; esac
   105	    ST["$k"]="$v"
   106	  done < "$STATE"
   107	}
   108	save_state() {
   109	  local tmp="${STATE}.tmp.$$" k
   110	  {
   111	    echo "# exp_11 chunk watchdog state (plain KV, rewritten atomically)."
   112	    echo "# Delete a frozen_<ARM> line to let the watchdog submit that arm again."
   113	    if [ "${#ST[@]}" -gt 0 ]; then
   114	      for k in "${!ST[@]}"; do printf '%s %s\n' "$k" "${ST[$k]}"; done | sort
   115	    fi
   116	  } > "$tmp" || { echo "could not write ${tmp}" >&2; return 1; }
   117	  mv -f "$tmp" "$STATE" || { echo "could not publish ${STATE}" >&2; return 1; }
   118	}
   119	
   120	bump_fail() {   # <arm> <reason>
   121	  local arm="$1" reason="$2" n
   122	  n=$(( ${ST[fail_$arm]:-0} + 1 ))
   123	  ST["fail_$arm"]="$n"
   124	  log "${arm}: FAILURE (${reason}) — ${n}/${MAX_CONSEC_FAIL} consecutive"
   125	  if [ "$n" -ge "$MAX_CONSEC_FAIL" ]; then
   126	    ST["frozen_$arm"]="$(date -Is)|${reason}"
   127	    log "${arm}: !!! FROZEN after ${n} consecutive failures (${reason}). No further submission for this arm until its frozen_${arm} line is removed from ${STATE}."
   128	  fi
   129	}
   130	
   131	clear_fail() {  # <arm> <why>
   132	  if [ "${ST[fail_$1]:-0}" != "0" ]; then
   133	    log "${1}: failure streak cleared (${2})"
   134	  fi
   135	  ST["fail_$1"]="0"
   136	}
   137	
   138	newest_ckpt_step() {   # <ckpt dir> -> the largest step with a checkpoint file
   139	  ls -1 "$1" 2>/dev/null \
   140	    | sed -n 's/^epoch=[0-9]\{1,\}-step=\([0-9]\{1,\}\)\.ckpt$/\1/p' | sort -n | tail -1
   141	}
   142	
   143	chain_tip_step() {     # <arm> -> the arm's recorded chain tip (or its anchor, or 0)
   144	  "$PY" - "$REGISTRY" "$1" <<'PY' 2>/dev/null
   145	import json, sys
   146	row = (json.load(open(sys.argv[1])).get("arms") or {}).get(sys.argv[2]) or {}
   147	chain = row.get("chain") or []
   148	print(int((chain[-1].get("final_step") if chain else row.get("final_step")) or 0))
   149	PY
   150	}
   151	
   152	leg_manifest_for() {   # <arm> <chunk_end> -> the ONE launcher manifest for that chunk
   153	  local arm="$1" want="$2" f ce
   154	  local -a hits=()
   155	  for f in "$EXPDIR"/fa_orbit_*_"${arm}"_8x8_jid*_manifest.txt; do
   156	    [ -f "$f" ] || continue
   157	    ce="$(awk '/^chunk_end /{print $2; exit}' "$f" 2>/dev/null)"
   158	    [ "$ce" = "$want" ] && hits+=("$f")
   159	  done
   160	  [ "${#hits[@]}" -eq 1 ] || return 1
   161	  printf '%s\n' "${hits[0]}"
   162	}
   163	
   164	process_arm() {
   165	  local arm="$1"
   166	  if [ -n "${ST[frozen_$arm]:-}" ]; then
   167	    log "${arm}: FROZEN (${ST[frozen_$arm]}) — skipping; clear it in ${STATE} to resume"
   168	    return 0
   169	  fi
   170	
   171	  local queued
   172	  queued="$(squeue -h -u "$USER" -n "exp11-${arm}-train" -o '%i %T' 2>/dev/null)"
   173	
   174	  # settle the leg we last submitted, once it has left the queue
   175	  local last="${ST[lastjob_$arm]:-}" st
   176	  if [ -n "$last" ] && [ -z "$queued" ]; then
   177	    st="$(sacct -X -n -P -j "$last" -o State 2>/dev/null | head -1 | awk '{print $1}')"
   178	    case "$st" in
   179	      COMPLETED)          clear_fail "$arm" "job ${last} COMPLETED" ;;
   180	      FAILED|TIMEOUT|NODE_FAIL|OUT_OF_MEMORY|BOOT_FAIL|DEADLINE|PREEMPTED|CANCELLED)
   181	                          bump_fail "$arm" "job ${last} ended ${st}" ;;
   182	      "")                 log "${arm}: sacct reports nothing for job ${last} — not counted either way" ;;
   183	      *)                  log "${arm}: job ${last} ended ${st} — not counted" ;;
   184	    esac
   185	    unset "ST[lastjob_$arm]"
   186	    [ -n "${ST[frozen_$arm]:-}" ] && return 0
   187	  fi
   188	
   189	  if [ -n "$queued" ]; then
   190	    log "${arm}: live job (${queued//$'\n'/; }) — nothing to do"
   191	    return 0
   192	  fi
   193	
   194	  local ckdir="${OUTPUT_ROOT}/exp11_${arm}/FLAC_exp11_${arm}/exp11_${arm}/checkpoints"
   195	  local S; S="$(newest_ckpt_step "$ckdir")"
   196	  if [ -z "$S" ]; then
   197	    log "${arm}: no epoch=*-step=*.ckpt under ${ckdir} — nothing to resume, skipping"
   198	    return 0
   199	  fi
   200	  if [ "$S" -ge "$TARGET" ]; then
   201	    if [ "${ST[done_$arm]:-}" != "$S" ]; then
   202	      ST["done_$arm"]="$S"
   203	      log "${arm}: DONE — newest checkpoint is step ${S} >= target ${TARGET}"
   204	    fi
   205	    return 0
   206	  fi
   207	
   208	  # a chunk that finished but was never recorded must be recorded FIRST: the next
   209	  # chunk's preflight binds to the recorded chain, not to whatever is on disk.
   210	  if [ "$S" -gt "$ANCHOR_STEP" ]; then
   211	    local tip; tip="$(chain_tip_step "$arm")"
   212	    is_num "$tip" || { log "${arm}: could not read the chain tip from ${REGISTRY}"; bump_fail "$arm" "registry unreadable"; return 0; }
   213	    if [ "$tip" -lt "$S" ]; then
   214	      local man
   215	      if ! man="$(leg_manifest_for "$arm" "$S")"; then
   216	        log "${arm}: no single launcher manifest with chunk_end ${S} in ${EXPDIR} (zero or ambiguous) — cannot record the finished chunk"
   217	        bump_fail "$arm" "manifest for chunk_end ${S} not uniquely identifiable"
   218	        return 0
   219	      fi
   220	      local -a rec=("$RECORDER" "$arm" "$man" --registry "$REGISTRY")
   221	      [ "$DRYRUN" = "1" ] && rec+=(--dry-run)
   222	      log "${arm}: recording finished chunk ${tip} -> ${S} from $(basename "$man")"
   223	      local out rc
   224	      out="$("$PY" "${rec[@]}" 2>&1)"; rc=$?
   225	      if [ "$rc" -ne 0 ]; then
   226	        log "${arm}: RECORDER REFUSED (rc=${rc}): $(printf '%s' "$out" | head -4 | tr '\n' ' ')"
   227	        bump_fail "$arm" "recorder rc=${rc}"
   228	        return 0
   229	      fi
   230	      clear_fail "$arm" "chunk ${tip} -> ${S} recorded"
   231	      if [ "$DRYRUN" = "1" ]; then
   232	        log "${arm}: DRYRUN — the chunk link was validated but not written, so the next chunk is not yet admissible"
   233	      fi
   234	    fi
   235	  fi
   236	
   237	  # submit the next chunk through the ONLY sanctioned submitter
   238	  local next=$(( S + CHUNK ))
   239	  [ "$next" -gt "$TARGET" ] && next="$TARGET"
   240	  local -a ck=("$ckdir"/*-step="${S}".ckpt)
   241	  if [ "${#ck[@]}" -ne 1 ] || [ ! -f "${ck[0]}" ]; then
   242	    log "${arm}: expected exactly one checkpoint file at step ${S} in ${ckdir}, found ${#ck[@]}"
   243	    bump_fail "$arm" "ambiguous checkpoint at step ${S}"
   244	    return 0
   245	  fi
   246	  log "${arm}: submitting chunk ${S} -> ${next}: bash ${SUBMITTER} ${arm} --resume ${ck[0]} --expected-step ${S} --chunk-end ${next} (DRYRUN=${DRYRUN})"
   247	  local out rc jid
   248	  out="$(DRYRUN="$DRYRUN" bash "$SUBMITTER" "$arm" --resume "${ck[0]}" \
   249	          --expected-step "$S" --chunk-end "$next" 2>&1)"; rc=$?
   250	  if [ "$rc" -ne 0 ]; then
   251	    log "${arm}: SUBMIT REFUSED (rc=${rc}): $(printf '%s' "$out" | head -4 | tr '\n' ' ')"
   252	    bump_fail "$arm" "submit rc=${rc}"
   253	    return 0
   254	  fi
   255	  jid="$(printf '%s' "$out" | awk '/^submitted /{print $NF}')"
   256	  if [ -n "$jid" ]; then
   257	    ST["lastjob_$arm"]="$jid"
   258	    log "${arm}: submitted chunk ${S} -> ${next} as job ${jid}"
   259	  else
   260	    log "${arm}: submitter returned 0 without a job id (DRYRUN=${DRYRUN}) — nothing queued"
   261	  fi
   262	  return 0
   263	}
   264	
   265	IFS=',' read -r -a ARM_LIST <<< "$ARMS"
   266	log "=== chunk watchdog start: arms ${ARMS} chunk ${CHUNK} target ${TARGET} poll ${POLL}s max_consec_fail ${MAX_CONSEC_FAIL} oneshot ${ONESHOT} dryrun ${DRYRUN} ==="
   267	while :; do
   268	  load_state
   269	  for ARM in "${ARM_LIST[@]}"; do
   270	    process_arm "$ARM"
   271	  done
   272	  save_state || log "WARNING: the state file could not be written — failure counters will not persist"
   273	
   274	  REMAINING=0
   275	  for ARM in "${ARM_LIST[@]}"; do
   276	    [ -n "${ST[done_$ARM]:-}" ] && continue
   277	    [ -n "${ST[frozen_$ARM]:-}" ] && continue
   278	    REMAINING=$((REMAINING + 1))
   279	  done
   280	  if [ "$REMAINING" -eq 0 ]; then
   281	    log "=== every arm is DONE or FROZEN — watchdog exiting ==="
   282	    break
   283	  fi
   284	  [ "$ONESHOT" = "1" ] && { log "=== ONESHOT: one pass complete, exiting ==="; break; }
   285	  sleep "$POLL"
   286	done
   287	exit 0

exec
/usr/bin/bash -lc "nl -ba worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_record_restart.py | sed -n '312,520p'
grep -n \"def write_atomic\\|flock\\|LOCK_\\|lock\" worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_producer_manifest.py
nl -ba worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_producer_manifest.py | sed -n '1,170p'
nl -ba worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submit.sh | sed -n '1,230p'" in /n/fs/gatrdp/codespace/FLAC
 succeeded in 0ms:
   312	def main(argv=None):
   313	    ap = argparse.ArgumentParser(description="record an exp_11 RESTART leg")
   314	    ap.add_argument("arm")
   315	    ap.add_argument("manifest")
   316	    ap.add_argument("--registry", default=os.path.join(HERE, "arm_launch_registry.json"))
   317	    ap.add_argument("--launcher", default=os.path.join(HERE, "fa_orbit_train.sbatch"),
   318	                    help="where the Q10 pins are read from")
   319	    ap.add_argument("--producer-dir", default=HERE,
   320	                    help="where the per-leg producer manifests are published")
   321	    # HERE = <repo>/worklog/worklog_<user>/exp_11_fa_orbit_claude
   322	    ap.add_argument("--repo-root", default=os.path.dirname(os.path.dirname(os.path.dirname(HERE))),
   323	                    help="root the manifest's relative paths resolve against")
   324	    ap.add_argument("--extend", action="store_true",
   325	                    help="this leg is already recorded: extend its producer manifest only")
   326	    ap.add_argument("--rehash-all", action="store_true",
   327	                    help="re-hash published checkpoints too (full audit, expensive)")
   328	    ap.add_argument("--dry-run", action="store_true", help="validate and report, publish nothing")
   329	    args = ap.parse_args(argv)
   330	
   331	    arm = args.arm
   332	    pins = read_pins(args.launcher)
   333	    if not pins.get("PINNED_MAXSTEPS"):
   334	        raise SystemExit(f"no PINNED_* values found in {args.launcher}")
   335	
   336	    # One writer at a time, and the lock is the registry's own DIRECTORY: no lock
   337	    # file to leave behind in a tracked tree, and it still covers the tmp+rename.
   338	    store = os.path.dirname(os.path.abspath(args.registry)) or "."
   339	    lock_fd = os.open(store, os.O_RDONLY)
   340	    try:
   341	        fcntl.flock(lock_fd, fcntl.LOCK_EX)
   342	        return record(args, arm, pins)
   343	    finally:
   344	        os.close(lock_fd)
   345	
   346	
   347	def record(args, arm, pins):
   348	    reg = json.load(open(args.registry))
   349	    initial = reg.get("arms", {}).get(arm)
   350	    if initial is None:
   351	        raise SystemExit(f"{arm} has no INITIAL registry entry")
   352	
   353	    raw, man = parse_manifest(args.manifest)
   354	    man_sha = hashlib.sha256(raw).hexdigest()
   355	    # Round 5: a leg that carries a `chunk_end` line stopped at a chunk boundary
   356	    # and is recorded as a CHAIN LINK, not as a whole-budget RESTART row.
   357	    chunk_raw = chunk_end_of(man)
   358	    if chunk_raw is not None:
   359	        return record_chunk(args, arm, reg, initial, man, man_sha, pins, chunk_raw)
   360	    problems, resume_real = check_identity(arm, man, initial, pins, args.repo_root)
   361	    jk, ak, rk = kvs(man, "job"), kvs(man, "arm"), kvs(man, "resume_ckpt")
   362	    job = jk.get("job")
   363	
   364	    legs = reg.setdefault("restarts", {}).setdefault(arm, [])
   365	    same = [l for l in legs if l.get("job") == job or l.get("launch_uuid") == jk.get("launch_uuid")
   366	            or l.get("manifest_sha256") == man_sha]
   367	    if same and not args.extend:
   368	        raise SystemExit(f"{arm} job {job} is ALREADY recorded ({len(same)} matching leg(s)) — "
   369	                         "one leg, one row; use --extend to extend its producer manifest")
   370	    if len(same) > 1:
   371	        problems.append(f"{len(same)} registry rows already claim this leg — the registry is "
   372	                        "inconsistent; fix it before recording")
   373	    if args.extend and not same:
   374	        problems.append(f"--extend given but {arm} job {job} is not recorded yet")
   375	    if problems:
   376	        print("RECORD REFUSED:")
   377	        for p in problems:
   378	            print(f"  !! {p}")
   379	        return 2
   380	
   381	    anchor = initial["final_ckpt_sha256"]
   382	    producer = pm.manifest_name(arm, job)
   383	    row = {
   384	        "manifest_path": args.manifest, "manifest_sha256": man_sha,
   385	        "job": job, "mode": "RESTART", "launch_uuid": jk.get("launch_uuid"),
   386	        "arm": arm, "commit": man.get("commit"), "rung": ak.get("rung"),
   387	        "config_sha256": man.get("config_sha256"), "save_dir": man.get("save_dir"),
   388	        "resume_ckpt": resume_real, "resume_ckpt_sha256": anchor,
   389	        "expected_step": rk.get("expected_step"), "max_steps": ak.get("max_steps"),
   390	        "time_limit": kvs(man, "time_limit").get("time_limit"),
   391	        "producer_manifest": producer, "chains_to": anchor,
   392	        "recorded_at": pm._now(),
   393	    }
   394	    header = {"arm": arm, "job": job, "launch_uuid": jk.get("launch_uuid"), "mode": "RESTART",
   395	              "commit": man.get("commit"), "resume_ckpt_sha256": anchor,
   396	              "expected_step": rk.get("expected_step"), "max_steps": ak.get("max_steps"),
   397	              "save_dir": man.get("save_dir"), "config_sha256": man.get("config_sha256"),
   398	              "chains_to": anchor, "leg_manifest_sha256": man_sha}
   399	
   400	    ckpt_dir = canonical_ckpt_dir(initial["save_dir"], arm, args.repo_root)
   401	    prod_path = os.path.join(args.producer_dir, producer)
   402	    known = (pm.load(prod_path) or {}).get("checkpoints", {})
   403	    found, scan_problems = pm.scan_checkpoints(
   404	        ckpt_dir, int(rk["expected_step"]), int(ak["max_steps"]), known=known,
   405	        rehash_all=args.rehash_all, repo_root=args.repo_root)
   406	    added, kept, pub_problems = pm.publish(prod_path, header, found, dry_run=args.dry_run)
   407	    if scan_problems or pub_problems:
   408	        print("RECORD REFUSED:")
   409	        for p in scan_problems + pub_problems:
   410	            print(f"  !! {p}")
   411	        return 2
   412	
   413	    if args.extend:
   414	        for i, leg in enumerate(legs):
   415	            if leg.get("job") == job:
   416	                legs[i] = {**leg, "producer_manifest": producer}
   417	    else:
   418	        legs.append(row)
   419	    if not args.dry_run:
   420	        pm.write_atomic(args.registry, reg)
   421	    verb = "extended" if args.extend else "recorded"
   422	    print(f"{verb} {arm} RESTART job {job} chaining to {anchor[:12]} "
   423	          f"({'dry run, nothing written' if args.dry_run else 'published'})")
   424	    print(f"  producer manifest {producer}: {len(added)} checkpoint(s) added, "
   425	          f"{len(kept)} already published"
   426	          + (f" (steps {added[0]}..{added[-1]})" if added else ""))
   427	    return 0
   428	
   429	
   430	if __name__ == "__main__":
   431	    sys.exit(main())
158:def write_atomic(path, doc):
     1	#!/usr/bin/env python3
     2	"""Per-leg PRODUCER manifests: which checkpoints a RESTART leg actually produced.
     3	
     4	Re-pin review, required fix 2. The >40k lineage gate was EXISTENTIAL: once any
     5	registry leg for an arm carried `mode=RESTART` and the right 40k resume hash,
     6	every later checkpoint in that arm's canonical directory passed, because nothing
     7	bound the evaluated checkpoint's own bytes to the leg that produced it. A
     8	same-config checkpoint from a wrong restart, copied into the canonical directory,
     9	was admissible.
    10	
    11	DESIGN CHOICE (the review allowed two; this is the second, and why).
    12	  Rejected: the restart leg's JOB hashes each checkpoint as it saves. That is the
    13	  tighter binding, but it means editing fa_orbit_train.sbatch's training path --
    14	  a hashing sidecar running beside torchrun -- while jobs 3662828-30 sit queued
    15	  against that launcher, which this round is forbidden to do (and which would put
    16	  sustained multi-GB reads next to a live training job on a shared filesystem).
    17	  Chosen: the RECORDER captures the leg's checkpoint inventory (step -> sha256,
    18	  re-hashed from disk) and publishes it into an APPEND-ONLY, COMMITTED per-leg
    19	  file next to the audited registry. The screen re-hashes the checkpoint it is
    20	  about to evaluate and requires an exact step -> sha256 -> leg match.
    21	
    22	Why that is still immutable evidence: the file lives in the tracked experiment
    23	directory, exactly like arm_launch_registry.json, and screens read it from the
    24	PINNED worktree -- so a row can only enter it through a commit, and only a
    25	commit that is in the campaign pin can be used as evidence. Within the file the
    26	recorder is append-only: a step already published may never change its sha256 or
    27	its path, and the header (arm, job, launch uuid, resume anchor, save-dir, config
    28	sha, budget) may never change at all.
    29	"""
    30	import hashlib
    31	import json
    32	import os
    33	import re
    34	import tempfile
    35	from datetime import datetime
    36	
    37	CKPT_RE = re.compile(r"^epoch=(\d+)-step=(\d+)\.ckpt$")
    38	# Header fields that identify the LEG. Once published they are frozen: a
    39	# republish that disagrees on any of them is a different leg wearing this file's
    40	# name, not an extension of it.
    41	HEADER_FIELDS = ("arm", "job", "launch_uuid", "mode", "commit", "resume_ckpt_sha256",
    42	                 "expected_step", "max_steps", "save_dir", "config_sha256",
    43	                 "chains_to", "leg_manifest_sha256")
    44	
    45	
    46	def sha256_file(path):
    47	    h = hashlib.sha256()
    48	    with open(path, "rb") as fh:
    49	        for chunk in iter(lambda: fh.read(1 << 20), b""):
    50	            h.update(chunk)
    51	    return h.hexdigest()
    52	
    53	
    54	def manifest_name(arm, job):
    55	    """The per-leg file name. Flat in the experiment directory, like the registry
    56	    and the backfill manifest, so the launcher/submitter drift gates (`$EXPDIR/*.json`)
    57	    cover it."""
    58	    return f"fa_orbit_producer_{arm}_job{job}.json"
    59	
    60	
    61	def load(path):
    62	    if not os.path.isfile(path):
    63	        return None
    64	    with open(path) as fh:
    65	        return json.load(fh)
    66	
    67	
    68	def rel_to(root, path):
    69	    """Repo-relative when possible (portable across the pinned worktrees), else absolute."""
    70	    real, root_real = os.path.realpath(path), os.path.realpath(root)
    71	    return os.path.relpath(real, root_real) if real.startswith(root_real + os.sep) else real
    72	
    73	
    74	def resolve(root, path):
    75	    return path if os.path.isabs(path) else os.path.join(root, path)
    76	
    77	
    78	def scan_checkpoints(ckpt_dir, after_step, max_step, known=None, rehash_all=False, repo_root="."):
    79	    """Re-hash the leg's checkpoints from DISK: {str(step): {path, sha256, bytes}}.
    80	
    81	    Only steps strictly after the resume point and no further than the budget are
    82	    the leg's own output -- the resume checkpoint itself belongs to the INITIAL
    83	    run and is already anchored in the registry. Steps already published are not
    84	    re-read by default (they are immutable evidence, and each is ~724 MB on a
    85	    shared filesystem); ``rehash_all`` forces a full audit."""
    86	    known = known or {}
    87	    out, problems = {}, []
    88	    if not os.path.isdir(ckpt_dir):
    89	        return {}, [f"checkpoint directory not found: {ckpt_dir}"]
    90	    for name in sorted(os.listdir(ckpt_dir)):
    91	        m = CKPT_RE.match(name)
    92	        if not m:
    93	            continue
    94	        step = int(m.group(2))
    95	        if step <= after_step or step > max_step:
    96	            continue
    97	        path = os.path.join(ckpt_dir, name)
    98	        key = str(step)
    99	        if key in out:
   100	            problems.append(f"two checkpoint files claim step {step} in {ckpt_dir}")
   101	            continue
   102	        if key in known and not rehash_all:
   103	            out[key] = dict(known[key])
   104	            continue
   105	        out[key] = {"path": rel_to(repo_root, path), "sha256": sha256_file(path),
   106	                    "bytes": os.path.getsize(path)}
   107	    return out, problems
   108	
   109	
   110	def publish(path, header, checkpoints, dry_run=False):
   111	    """Append-only publication of a leg's inventory. Returns (added, kept, problems).
   112	
   113	    The header is frozen and a published step may not change; the write is
   114	    tmp+rename in the destination directory, so a reader never sees a partial
   115	    file and a failed write leaves the previous one intact."""
   116	    problems, added, kept = [], [], []
   117	    old = load(path)
   118	    if old is not None:
   119	        for field in HEADER_FIELDS:
   120	            if str(old.get(field)) != str(header.get(field)):
   121	                problems.append(f"{os.path.basename(path)} is already published with "
   122	                                f"{field}={old.get(field)!r}, not {header.get(field)!r} — a "
   123	                                "producer manifest is immutable")
   124	        for step, entry in sorted(old.get("checkpoints", {}).items()):
   125	            new = checkpoints.get(step)
   126	            if new is None:
   127	                continue        # a published step whose file is gone stays published
   128	            if new["sha256"] != entry["sha256"]:
   129	                problems.append(f"step {step} is published with sha256 {entry['sha256'][:12]} but "
   130	                                f"now hashes {new['sha256'][:12]} — a published checkpoint may "
   131	                                "never change")
   132	            elif os.path.basename(str(new["path"])) != os.path.basename(str(entry["path"])):
   133	                problems.append(f"step {step} is published at {entry['path']} but now at "
   134	                                f"{new['path']}")
   135	    if problems:
   136	        return [], [], problems
   137	    merged = dict((old or {}).get("checkpoints", {}))
   138	    for step, entry in checkpoints.items():
   139	        (kept if step in merged else added).append(step)
   140	        merged.setdefault(step, entry)
   141	    doc = {k: header.get(k) for k in HEADER_FIELDS}
   142	    doc["_comment"] = [
   143	        "APPEND-ONLY producer manifest for one exp_11 RESTART leg (re-pin fix 2).",
   144	        "Every checkpoint this leg produced, re-hashed from disk by",
   145	        "fa_orbit_record_restart.py. The screen admits a >40k checkpoint only when",
   146	        "its own sha256 matches this file's entry for exactly that step, so a",
   147	        "valid restart row no longer vouches for any later same-config file.",
   148	        "A published step never changes; the header never changes.",
   149	    ]
   150	    doc["first_published"] = (old or {}).get("first_published") or _now()
   151	    doc["last_extended"] = _now()
   152	    doc["checkpoints"] = dict(sorted(merged.items(), key=lambda kv: int(kv[0])))
   153	    if not dry_run:
   154	        write_atomic(path, doc)
   155	    return sorted(added, key=int), sorted(kept, key=int), []
   156	
   157	
   158	def write_atomic(path, doc):
   159	    d = os.path.dirname(os.path.abspath(path)) or "."
   160	    fd, tmp = tempfile.mkstemp(prefix=".{}.".format(os.path.basename(path)), dir=d)
   161	    try:
   162	        with os.fdopen(fd, "w") as fh:
   163	            json.dump(doc, fh, indent=2)
   164	            fh.write("\n")
   165	            fh.flush()
   166	            os.fsync(fh.fileno())
   167	        os.replace(tmp, path)
   168	    except BaseException:
   169	        if os.path.exists(tmp):
   170	            os.unlink(tmp)
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
   111	DRIFT="$(git status --porcelain --untracked-files=no -- train.py defaults.ini src \
   112	         "$EXPDIR"/*.json "$EXPDIR"/*.py "$EXPDIR"/*.sbatch "$EXPDIR"/*.sh \
   113	         worklog/worklog_yixun/exp_07_fa_scratch_claude/FLAC_AR_BF.json 2>/dev/null)"
   114	[ -z "$DRIFT" ] || { echo "tracked measurement surfaces have uncommitted changes - commit first, abort:"; echo "$DRIFT"; exit 2; }
   115	SHA="$(git rev-parse HEAD)"
   116	
   117	ARGS=(
   118	  --job-name="$JOBNAME"
   119	  --gres="gpu:l40:${NGPU}"
   120	  --cpus-per-task="$((8 + 7 * NGPU))"
   121	  --mem="$(((12 * NGPU + 12)))G"
   122	  --time="$TIME_LIMIT"
   123	  --export="ALL,ARM=${ARM},EXPECT_SHA=${SHA},OUTPUT_ROOT=outputs_FLAC"
   124	)
   125	[ "$SMOKE" = "1" ] && ARGS[5]="${ARGS[5]},SMOKE=1,SMOKE_RUNG=${SMOKE_RUNG},SMOKE_MIN_FREE_MB=${SMOKE_MIN_FREE_MB},SMOKE_MAXSTEPS=${SMOKE_MAXSTEPS:-30},SMOKE_TIME=${TIME_LIMIT}"
   126	[ -n "$RESUME_CKPT" ] && ARGS[5]="${ARGS[5]},RESUME_CKPT=${RESUME_CKPT},EXPECTED_STEP=${EXPECTED_STEP}"
   127	[ -n "$CHUNK_END" ] && ARGS[5]="${ARGS[5]},CHUNK_END=${CHUNK_END}"
   128	ARGS+=("$SBATCH_FILE")
   129	
   130	echo "arm ${ARM} | rung ${RUNG} (${MB}x${NGPU}) | time ${TIME_LIMIT} | commit ${SHA} | smoke ${SMOKE} | chunk_end ${CHUNK_END:-<none>}"
   131	if [ "$DRYRUN" = "1" ]; then
   132	  echo "DRYRUN sbatch ${ARGS[*]}"
   133	  exit 0
   134	fi
   135	
   136	# --- NEW-3: publish the INTENT before submitting -----------------------------
   137	# The provenance record must exist before the job can exist, otherwise a local
   138	# write failure leaves a queued job nobody recorded. The intent manifest carries
   139	# the exact command and pins; the job id is appended afterwards, and if that
   140	# append fails the exact job we just created is cancelled.
   141	INTENT_ID="$(date +%s%N)-$(cut -c1-8 /proc/sys/kernel/random/uuid)"
   142	MANIFEST="${EXPDIR}/fa_orbit_submission_${ARM}_${INTENT_ID}.txt"
   143	[ ! -e "$MANIFEST" ] || { echo "submission manifest ${MANIFEST} already exists - abort"; exit 2; }
   144	TMP="$(mktemp "${MANIFEST}.XXXXXX")" || exit 3
   145	{
   146	  echo "# exp_11 arm submission (intent published BEFORE sbatch)"
   147	  echo "intent_id ${INTENT_ID}"
   148	  echo "submitted_at $(date -Is)"
   149	  echo "arm ${ARM} rung ${RUNG} micro ${MB} ngpu ${NGPU}"
   150	  echo "jobname ${JOBNAME} time ${TIME_LIMIT} smoke ${SMOKE}"
   151	  echo "commit ${SHA}"
   152	  echo "pins rung=${RUNG} maxsteps=$(pin PINNED_MAXSTEPS) ckpt_every=$(pin PINNED_CHECKPOINT_EVERY) min_free_mb=$(pin PINNED_MIN_FREE_MB) p0_manifest_sha256=$(pin PINNED_P0_MANIFEST_SHA256)"
   153	  echo "resume ${RESUME_CKPT:-<none>} expected_step ${EXPECTED_STEP}"
   154	  echo "chunk_end ${CHUNK_END:-<none>}"
   155	  echo "sbatch sbatch ${ARGS[*]}"
   156	} >> "$TMP" || { echo "intent manifest write failed - abort"; exit 3; }
   157	mv -n "$TMP" "$MANIFEST" || { echo "intent manifest publication failed - abort"; exit 2; }
   158	[ -e "$MANIFEST" ] || { echo "intent manifest ${MANIFEST} did not appear - abort"; exit 2; }
   159	echo "intent manifest: ${MANIFEST}"
   160	
   161	OUT="$(sbatch "${ARGS[@]}" 2>&1)"; JID="$(echo "$OUT" | awk '/Submitted batch job/ {print $NF}')"
   162	if [ -z "$JID" ]; then
   163	  echo "SUBMIT FAILED: ${OUT}"
   164	  echo "submit_failed $(date -Is)" >> "$MANIFEST"
   165	  exit 1
   166	fi
   167	echo "submitted ${ARM} -> job ${JID}"
   168	if ! echo "jobid ${JID}" >> "$MANIFEST"; then
   169	  echo "could not append job id ${JID} to ${MANIFEST} — cancelling the job rather than leave it unrecorded"
   170	  scancel "$JID" || echo "scancel ${JID} FAILED — cancel it by hand NOW"
   171	  exit 2
   172	fi
   173	echo "submission recorded: ${MANIFEST} (job ${JID})"

exec
/usr/bin/bash -lc "nl -ba worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch | sed -n '120,840p'" in /n/fs/gatrdp/codespace/FLAC
 succeeded in 0ms:
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
   250	DRIFT="$(git status --porcelain --untracked-files=no -- train.py defaults.ini src ":(exclude)src/tests" data/AR \
   251	          "$EXPREL/*.json" "$EXPREL/*.py" "$EXPREL/*.sbatch" "$EXPREL/*.sh" \
   252	          "$EXP07REL/FLAC_AR_BF.json" 2>&1)" \
   253	  || die "git status for the drift gate failed: ${DRIFT} - abort"
   254	# Commit binding is CONTENT-scoped: HEAD identity is sufficient but not
   255	# necessary. Two sessions commit to this checkout, so a pending leg must
   256	# survive commits that leave the training closure untouched — and abort on
   257	# any commit that changes it. The closure is what the job actually loads:
   258	# train.py, defaults.ini, src/, the data/AR split JSONs, the five arm
   259	# configs (enumerated — a shell glob would silently drop a config deleted
   260	# since EXPECT_SHA), this launcher, the four runtime helper scripts it
   261	# invokes, and exp_07's FLAC_AR_BF.json (C4L parity baseline).
   262	# Record/analysis files (registry, manifests, gen_*/validators, worklog)
   263	# are deliberately OUTSIDE the closure. Fail-closed on every edge:
   264	# EXPECT_SHA must be the full 40-hex commit OID (a symbolic ref like HEAD
   265	# would defeat the binding), the diff runs against the CAPTURED HEAD OID,
   266	# and HEAD is re-read afterwards to close the mid-gate-commit race.
   267	# src/tests is EXCLUDED from both gates: pytest-only code, never imported by
   268	# train.py at runtime, and the TDD sessions land test files continuously —
   269	# gating them reproduced the exact kill this gate exists to prevent
   270	# (legs 3680875-78 died on a src/tests-only commit, 2026-08-12).
   271	surface_diff_vs_expect() {
   272	  git diff --name-only "${EXPECT_SHA}" "${HEAD_SHA}" -- train.py defaults.ini src ":(exclude)src/tests" data/AR \
   273	      "$EXPDIR"/FLAC_AR_BF_C4L.json "$EXPDIR"/FLAC_AR_BF_C8.json \
   274	      "$EXPDIR"/FLAC_AR_BF_C16.json "$EXPDIR"/FLAC_AR_BF_C32.json \
   275	      "$EXPDIR"/FLAC_AR_VANCKPT.json "$EXPDIR"/fa_orbit_train.sbatch \
   276	      "$EXPDIR"/fa_orbit_ckpt_preflight.py "$EXPDIR"/assert_arm_configs_exp11.py \
   277	      "$EXPDIR"/fa_orbit_wandb_readback.py "$EXPDIR"/fa_orbit_classify.py \
   278	      "$EXP07/FLAC_AR_BF.json"
   279	}
   280	GATE_FAIL=""; GATE_OK_MSG=""
   281	if [ -z "$HEAD_SHA" ]; then
   282	  GATE_FAIL="cannot resolve HEAD"
   283	elif ! printf '%s\n' "$EXPECT_SHA" | grep -qE '^[0-9a-f]{40}$'; then
   284	  GATE_FAIL="EXPECT_SHA '${EXPECT_SHA}' is not a full lowercase 40-hex commit id"
   285	elif [ "$(git rev-parse --verify -q "${EXPECT_SHA}^{commit}" 2>/dev/null)" != "$EXPECT_SHA" ]; then
   286	  GATE_FAIL="EXPECT_SHA ${EXPECT_SHA} is not a commit known to this repo"
   287	elif [ "$HEAD_SHA" = "$EXPECT_SHA" ]; then
   288	  GATE_OK_MSG="commit binding OK: ${HEAD_SHA}"
   289	elif SD="$(surface_diff_vs_expect 2>&1)"; then
   290	  if [ -z "$SD" ]; then
   291	    GATE_OK_MSG="commit binding OK (content): training surfaces identical, EXPECT_SHA ${EXPECT_SHA} HEAD ${HEAD_SHA}"
   292	  else
   293	    GATE_FAIL="training surfaces changed since EXPECT_SHA ${EXPECT_SHA} (HEAD ${HEAD_SHA}): ${SD}"
   294	  fi
   295	else
   296	  GATE_FAIL="surface diff vs EXPECT_SHA failed: ${SD}"
   297	fi
   298	if [ -z "$GATE_FAIL" ] && [ "$(git rev-parse HEAD 2>/dev/null)" != "$HEAD_SHA" ]; then
   299	  GATE_FAIL="HEAD moved during the commit-binding check (was ${HEAD_SHA})"
   300	fi
   301	if [ "$DRYRUN" = "1" ]; then
   302	  [ -z "$GATE_FAIL" ] && echo "${GATE_OK_MSG} (dry run)" \
   303	    || echo "DRY-RUN ADVISORY: ${GATE_FAIL} (a real launch aborts here)"
   304	  [ -z "$DRIFT" ] || echo "DRY-RUN ADVISORY: tracked measurement surfaces are modified (a real launch aborts here)"
   305	else
   306	  [ -n "${SLURM_JOB_ID:-}" ] || die "a real launch must run under sbatch (no SLURM_JOB_ID) - abort"
   307	  [ -z "$GATE_FAIL" ] || die "${GATE_FAIL} - abort"
   308	  echo "$GATE_OK_MSG"
   309	  [ -z "$DRIFT" ] || { echo "tracked measurement surfaces modified since review - abort:"; echo "$DRIFT"; exit 2; }
   310	fi
   311	
   312	# --- D. arm -> config (single source) + semantic gate -------------------------
   313	arm_config_for() {
   314	  case "$1" in
   315	    C4L|C8|C16|C32) echo "$EXPDIR/FLAC_AR_BF_$1.json" ;;
   316	    VANL)           echo "$EXPDIR/FLAC_AR_VANCKPT.json" ;;
   317	    *) return 1 ;;
   318	  esac
   319	}
   320	MODEL_CONFIG="$(arm_config_for "$ARM")" || die "no config mapped for arm '${ARM}' - abort"
   321	MODEL_CONFIG_ABS="$(readlink -f "$MODEL_CONFIG" 2>/dev/null)"
   322	[ -n "$MODEL_CONFIG_ABS" ] && [ -f "$MODEL_CONFIG_ABS" ] || die "arm config '${MODEL_CONFIG}' does not exist - abort"
   323	CONFIG_SHA="$(sha256sum "$MODEL_CONFIG_ABS" | awk '{print $1}')"
   324	echo "config for ${ARM}: ${MODEL_CONFIG_ABS} sha256 ${CONFIG_SHA}"
   325	
   326	python3 - "$MODEL_CONFIG_ABS" "$ARM" <<'PY' || die "arm/config semantic gate FAILED - abort"
   327	import json, sys
   328	cfg = json.load(open(sys.argv[1])); arm = sys.argv[2]
   329	t = cfg.get("training", {}); bad = []
   330	# VANL is the same recipe with the conditioning removed, so its gate is the
   331	# MIRROR IMAGE of the orbit arms': the orbit keys must be ABSENT, not merely
   332	# different. A vanilla config that carried a stray frame_avg_angles would be a
   333	# silently fa-flavoured baseline, which would destroy the single-delta claim.
   334	if arm == "VANL":
   335	    cm = t.get("cond_method")
   336	    if cm not in (None, "vanilla"):
   337	        bad.append(f"cond_method={cm!r} (want absent or 'vanilla')")
   338	    if "frame_avg_angles" in t:
   339	        bad.append(f"frame_avg_angles is present ({t['frame_avg_angles']!r}) — a vanilla arm has no orbit")
   340	    want = None
   341	else:
   342	    want = {"C4L": 4, "C8": 8, "C16": 16, "C32": 32}[arm]
   343	    angles = t.get("frame_avg_angles")
   344	    if t.get("cond_method") != "fa_invariant":
   345	        bad.append(f"cond_method={t.get('cond_method')!r} (want fa_invariant)")
   346	    if not isinstance(angles, list) or len(angles) != want:
   347	        bad.append(f"frame_avg_angles has {angles and len(angles)} entries (want {want})")
   348	    elif angles != [k * 360.0 / want for k in range(want)]:
   349	        bad.append(f"frame_avg_angles are not the uniform C{want} orbit")
   350	if t.get("use_ema") is not True:
   351	    bad.append(f"use_ema={t.get('use_ema')!r} (want True)")
   352	vits = [c for c in cfg["model"]["conditioning"]["configs"] if c["type"] == "ViTCoordinates"]
   353	if sorted(c["id"] for c in vits) != ["context_poses_vit", "source_vit"]:
   354	    bad.append(f"ViT conditioner ids {sorted(c['id'] for c in vits)} != the expected two")
   355	# Post-P0: grad-ckpt ON for every arm; the KEY must exist and be literally True
   356	for c in vits:
   357	    if "gradient_checkpointing" not in c["config"]:
   358	        bad.append(f"{c['id']}: gradient_checkpointing key absent (want literal true)")
   359	    elif c["config"]["gradient_checkpointing"] is not True:
   360	        bad.append(f"{c['id']}: gradient_checkpointing={c['config']['gradient_checkpointing']!r} (want True)")
   361	if bad:
   362	    sys.exit("ARM/CONFIG GATE: " + "; ".join(bad))
   363	if arm == "VANL":
   364	    print(f"gate OK: {arm} is vanilla (no cond_method, no orbit), grad-ckpt True, EMA on")
   365	else:
   366	    print(f"gate OK: {arm} carries the uniform C{want} orbit, grad-ckpt True, EMA on")
   367	PY
   368	
   369	# --- E. lineage: INITIAL vs RESTART -------------------------------------------
   370	SAVEDIR_REAL="$(realpath -m "$SAVEDIR")"
   371	CKPT_DIR_REAL="$(realpath -m "${RUNDIR}/checkpoints")"
   372	LAUNCH_MANIFEST_LINK="${SAVEDIR}/launch_manifest.txt"     # written by the INITIAL launch
   373	if [ "$EXPECTED_STEP" -eq 0 ]; then
   374	  MODE="INITIAL"
   375	  [ -z "$RESUME_CKPT" ] || die "INITIAL launch must not carry RESUME_CKPT (set EXPECTED_STEP > 0 to declare a RESTART) - abort"
   376	  [ ! -e "$RUNDIR" ] || die "run directory ${RUNDIR} already exists — an INITIAL launch never clobbers a previous run - abort"
   377	else
   378	  MODE="RESTART"
   379	  [ -n "$RESUME_CKPT" ] || die "EXPECTED_STEP ${EXPECTED_STEP} declares a RESTART, but RESTART requires RESUME_CKPT - abort"
   380	  [ -f "$RESUME_CKPT" ] || die "RESUME_CKPT not found: ${RESUME_CKPT} - abort"
   381	  RESUME_REAL="$(realpath -m "$RESUME_CKPT")"
   382	  # exactly this arm's own checkpoints directory — not merely somewhere below the save root
   383	  case "$RESUME_REAL" in
   384	    "${CKPT_DIR_REAL}"/*.ckpt) ;;
   385	    *) die "a RESTART may only resume a checkpoint from ${CKPT_DIR_REAL}/ (got ${RESUME_REAL}) - abort" ;;
   386	  esac
   387	  [ "$MAXSTEPS" -gt "$EXPECTED_STEP" ] || die "MAXSTEPS ${MAXSTEPS} must exceed the resume step ${EXPECTED_STEP} - abort"
   388	fi
   389	echo "lineage: ${MODE} (expected_step ${EXPECTED_STEP}, max_steps ${MAXSTEPS}, chunk_end ${CHUNK_END:-<none>}, stops at ${STOPSTEP}, ckpt every ${CHECKPOINT_EVERY}, time pin ${TIME_PIN_NAME}=${TIME_LIMIT})"
   390	
   391	# --- F. the exact train.py argv ----------------------------------------------
   392	ARGV=(
   393	  --model-config "$MODEL_CONFIG_ABS"
   394	  --dataset-config src/configs/dataset_configs/AR/train/acousticroom_train.json
   395	  --pretransform-ckpt-path weights/FLAC/VAE.safetensors
   396	  --max-steps "$STOPSTEP" --batch-size "$MB" --accum-batches 1 --num-workers 6 --seed 42
   397	  --num-gpus "$NGPU" --num-nodes 1
   398	  --strategy ddp_find_unused_parameters_true --sync-batchnorm true --precision bf16-mixed
   399	  --val-every -1 --val-dataset-config ''
   400	  --gradient-clip-val 0.0
   401	  --logger wandb --checkpoint-every "$CHECKPOINT_EVERY"
   402	  --name "$NAME" --experiment-name "$EXPNAME" --save-dir "$SAVEDIR"
   403	)
   404	[ "$MODE" = "RESTART" ] && ARGV+=(--ckpt-path "$RESUME_CKPT")
   405	
   406	# --- G. argv-parity dry run (plan N13; round-3 N9 tightened) ------------------
   407	ARGV_FILE="$(mktemp)" || die "mktemp failed - abort" 3
   408	printf '%s\n' "${ARGV[@]}" > "$ARGV_FILE" || die "could not write the argv file - abort" 3
   409	python3 - "$ARGV_FILE" "$MODE" <<'PY'
   410	import sys
   411	# The exp_07 B-F reference argv (bf_scratch_launch.sh) — the lineage this sweep continues.
   412	REF = """--model-config worklog/worklog_yixun/exp_07_fa_scratch_claude/FLAC_AR_BF.json
   413	--dataset-config src/configs/dataset_configs/AR/train/acousticroom_train.json
   414	--pretransform-ckpt-path weights/FLAC/VAE.safetensors
   415	--max-steps 67500 --batch-size 32 --accum-batches 1 --num-workers 6 --seed 42
   416	--num-gpus 2 --strategy ddp_find_unused_parameters_true --sync-batchnorm true
   417	--logger wandb --checkpoint-every 2500
   418	--name FLAC_exp07_BF --experiment-name exp07_BF --save-dir outputs_FLAC/exp07_BF""".split()
   419	# Flags whose VALUE may differ from exp_07 (identity, budget, rung, resume):
   420	ALLOWED_DIFF = {"--model-config", "--name", "--experiment-name", "--save-dir", "--max-steps",
   421	                "--num-gpus", "--batch-size", "--logger", "--checkpoint-every", "--ckpt-path"}
   422	# Flags exp_07 left to defaults.ini and we state explicitly — whitelisted with their
   423	# EXACT expected values (round-3 N9: no "equals the mutable ini" escape hatch):
   424	ALLOWED_ADD = {"--num-nodes": "1", "--precision": "bf16-mixed", "--val-every": "-1",
   425	               "--val-dataset-config": "", "--gradient-clip-val": "0.0", "--ckpt-path": None}
   426	tokens = [t for t in open(sys.argv[1]).read().split("\n")]
   427	if tokens and tokens[-1] == "":
   428	    tokens.pop()
   429	mode = sys.argv[2]
   430	
   431	def as_map(toks):
   432	    out, i = {}, 0
   433	    while i < len(toks):
   434	        flag = toks[i]
   435	        if not flag.startswith("--"):
   436	            raise SystemExit(f"ARGV PARITY: stray token {flag!r}")
   437	        val = toks[i + 1] if i + 1 < len(toks) and not toks[i + 1].startswith("--") else ""
   438	        if flag in out:
   439	            raise SystemExit(f"ARGV PARITY: duplicate flag {flag}")
   440	        out[flag] = val
   441	        i += 2 if (i + 1 < len(toks) and not toks[i + 1].startswith("--")) else 1
   442	    return out
   443	
   444	ref, new = as_map(REF), as_map(tokens)
   445	violations, allowed, explicit = [], [], []
   446	for flag in sorted(set(ref) | set(new)):
   447	    if flag in ref and flag in new:
   448	        if ref[flag] != new[flag]:
   449	            (allowed if flag in ALLOWED_DIFF else violations).append(
   450	                f"{flag}: exp_07 {ref[flag]!r} -> exp_11 {new[flag]!r}")
   451	    elif flag in new:
   452	        if flag == "--ckpt-path":
   453	            (allowed if mode == "RESTART" else violations).append(
   454	                f"--ckpt-path: {new[flag]!r} (RESTART only)")
   455	        elif flag in ALLOWED_ADD and ALLOWED_ADD[flag] == new[flag]:
   456	            explicit.append(f"{flag}={new[flag]!r} (whitelisted explicit default)")
   457	        else:
   458	            violations.append(f"{flag}: added with {new[flag]!r}, not a whitelisted addition "
   459	                              f"(expected {ALLOWED_ADD.get(flag, '<not allowed>')!r})")
   460	    else:
   461	        violations.append(f"{flag}: present in exp_07 ({ref[flag]!r}), MISSING here")
   462	
   463	print("--- train.py argv ---")
   464	print(" ".join(f"{k} {v!r}" if v == "" else f"{k} {v}" for k, v in new.items()))
   465	print("--- argv parity vs exp_07 B-F ---")
   466	for d in allowed:
   467	    print(f"  allowed  {d}")
   468	for d in explicit:
   469	    print(f"  explicit {d}")
   470	if violations:
   471	    print("ARGV PARITY VIOLATIONS:")
   472	    for v in violations:
   473	        print(f"  !! {v}")
   474	    raise SystemExit(2)
   475	print(f"ARGV PARITY OK ({mode}): only whitelisted differences and additions")
   476	PY
   477	parity=$?
   478	rm -f "$ARGV_FILE"
   479	[ "$parity" -eq 0 ] || die "argv parity check FAILED - abort"
   480	if [ "$DRYRUN" = "1" ]; then
   481	  echo "DRY RUN complete: gates A–G passed for ARM=${ARM} RUNG=${RUNG} MODE=${MODE} SMOKE=${SMOKE}"
   482	  echo "  (Slurm/GPU/VRAM/env/wandb/ViT/lock gates and training are skipped in DRYRUN)"
   483	  exit 0
   484	fi
   485	
   486	# --- H. Slurm allocation must match the pins (round-3 B1) ---------------------
   487	[ "${SLURM_JOB_NUM_NODES:-1}" = "1" ] || die "expected 1 node, got ${SLURM_JOB_NUM_NODES} - abort"
   488	[ "${SLURM_NTASKS:-1}" = "1" ] || die "expected 1 task, got ${SLURM_NTASKS} - abort"
   489	WANT_CPUS="$((8 + 7 * NGPU))"; WANT_MEM_MB="$(((12 * NGPU + 12) * 1024))"
   490	GOT_CPUS="${SLURM_CPUS_PER_TASK:-${SLURM_CPUS_ON_NODE:-0}}"
   491	GOT_MEM_MB="${SLURM_MEM_PER_NODE:-0}"
   492	[ "$GOT_CPUS" = "$WANT_CPUS" ] || die "allocated ${GOT_CPUS} CPUs, the pinned rung needs ${WANT_CPUS} — submit via fa_orbit_submit.sh - abort"
   493	[ "$GOT_MEM_MB" = "$WANT_MEM_MB" ] || die "allocated ${GOT_MEM_MB} MB RAM, the pinned rung needs ${WANT_MEM_MB} — submit via fa_orbit_submit.sh - abort"
   494	GOT_TIME="$(squeue -h -j "$SLURM_JOB_ID" -o %l 2>/dev/null | tr -d ' ')"
   495	norm_minutes() { awk -v t="$1" 'BEGIN{d=0; if (t ~ /-/) {split(t,p,"-"); d=p[1]; t=p[2]} n=split(t,c,":");
   496	  if (n==3) m=c[1]*60+c[2]+c[3]/60; else if (n==2) m=c[1]+c[2]/60; else m=t; printf "%d", d*1440+m}'; }
   497	# The pin this ${MODE} leg is entitled to — an INITIAL allocation handed to a
   498	# RESTART leg (or the reverse) is refused here, in the job, not merely intended
   499	# by the submitter.
   500	[ "$(norm_minutes "$GOT_TIME")" = "$(norm_minutes "$TIME_LIMIT")" ] || die "allocated time ${GOT_TIME} != the ${TIME_PIN_NAME} pin ${TIME_LIMIT} this ${MODE} leg requires — submit via fa_orbit_submit.sh - abort"
   501	echo "allocation matches the pins: ${GOT_CPUS} cpus, ${GOT_MEM_MB} MB, ${GOT_TIME} (${TIME_PIN_NAME})"
   502	
   503	mapfile -t GPU_ROWS < <(nvidia-smi --query-gpu=uuid,name --format=csv,noheader,nounits)
   504	[ "${#GPU_ROWS[@]}" -eq "$NGPU" ] || die "rung ${RUNG} needs exactly ${NGPU} visible GPU(s), got ${#GPU_ROWS[@]} - abort"
   505	UUIDS=()
   506	for ROW in "${GPU_ROWS[@]}"; do
   507	  U="$(echo "$ROW" | cut -d, -f1 | tr -d ' ')"; N="$(echo "$ROW" | cut -d, -f2 | sed 's/^ *//')"
   508	  case "$N" in *L40*) ;; *) die "allocated GPU '${N}' is not an L40 - abort";; esac
   509	  UUIDS+=("$U"); echo "GPU: ${N} ${U}"
   510	done
   511	UUID_CSV="$(IFS=,; echo "${UUIDS[*]}")"
   512	
   513	# --- I. environment identity gates (round-3 B6) -------------------------------
   514	REAL_PY="$(command -v python)"
   515	[ "$(readlink -f "$REAL_PY")" = "$(readlink -f "$PINNED_PYTHON")" ] || die "python resolves to ${REAL_PY}, pinned ${PINNED_PYTHON} - abort"
   516	python3 - "$PINNED_PL_VERSION" "$PINNED_TORCH_VERSION" <<'PY' || die "runtime version gate FAILED - abort"
   517	import sys
   518	import pytorch_lightning as pl, torch
   519	want_pl, want_torch = sys.argv[1], sys.argv[2]
   520	bad = []
   521	if pl.__version__ != want_pl:
   522	    bad.append(f"pytorch_lightning {pl.__version__} != pinned {want_pl}")
   523	if torch.__version__ != want_torch:
   524	    bad.append(f"torch {torch.__version__} != pinned {want_torch}")
   525	if bad:
   526	    sys.exit("; ".join(bad))
   527	print(f"runtime gate OK: pl {pl.__version__}, torch {torch.__version__}, cuda {torch.version.cuda}")
   528	PY
   529	VAE_SHA="$(sha256sum weights/FLAC/VAE.safetensors | awk '{print $1}')"
   530	[ "$VAE_SHA" = "$PINNED_VAE_SHA256" ] || die "weights/FLAC/VAE.safetensors sha256 ${VAE_SHA} != pinned ${PINNED_VAE_SHA256} - abort"
   531	echo "VAE pin OK: ${VAE_SHA}"
   532	DRIVER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
   533	CUDA_VER="$(python3 -c 'import torch; print(torch.version.cuda)')"
   534	
   535	# --- J. per-GPU free-VRAM gate (floor from the pins) --------------------------
   536	for U in "${UUIDS[@]}"; do
   537	  FREE="$(nvidia-smi --id="$U" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | tr -dc '0-9')"
   538	  [ -n "$FREE" ] || die "nvidia-smi free-mem query failed on ${U} - refusing to launch blind"
   539	  [ "$FREE" -ge "$MIN_FREE_MB" ] || die "GPU ${U} free ${FREE} MiB < required ${MIN_FREE_MB} MiB - refusing to launch"
   540	done
   541	echo "--- co-tenancy disclosure at launch ---"
   542	nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true
   543	
   544	# --- K. exclusive run ownership via flock (round-3 B3 residual) --------------
   545	# mkdir + stale recovery had two races: a contender could arrive between mkdir
   546	# and the owner write, and release removed the directory without checking whose
   547	# it was. flock has neither: the kernel holds the lock while the fd is open and
   548	# releases it on close (including on kill), so there is no stale state to
   549	# recover and no recovery path to get wrong.
   550	mkdir -p "$OUTPUT_ROOT" || die "could not create ${OUTPUT_ROOT} - abort" 3
   551	LOCKFILE="${OUTPUT_ROOT}/exp11_${ARM}.lock"
   552	exec 9>"$LOCKFILE" || die "could not open the lock file ${LOCKFILE} - abort" 3
   553	if ! flock -n 9; then
   554	  OWNER="$(tr '\n' ' ' < "$LOCKFILE" 2>/dev/null)"
   555	  die "arm ${ARM} is locked by another live job (${OWNER:-<no metadata>}) - refusing a concurrent writer - abort"
   556	fi
   557	LAUNCH_UUID="$(cat /proc/sys/kernel/random/uuid)"
   558	{ echo "job ${SLURM_JOB_ID}"; echo "uuid ${LAUNCH_UUID}"; echo "arm ${ARM}"; echo "mode ${MODE}"; echo "acquired ${TS}"; } >&9 \
   559	  || die "could not write the lock owner metadata - abort" 3
   560	echo "lock acquired: ${LOCKFILE} (flock on fd 9, released on exit)"
   561	mkdir -p "$SAVEDIR" || die "could not create ${SAVEDIR} - abort" 3
   562	
   563	# --- L. RESTART preflight (round-3 B2) ---------------------------------------
   564	CKPT_SHA=""
   565	if [ "$MODE" = "RESTART" ]; then
   566	  PRE_ARGS=(--ckpt "$RESUME_CKPT" --expected-step "$EXPECTED_STEP" --config "$MODEL_CONFIG_ABS"
   567	            --max-steps "$MAXSTEPS" --arm "$ARM" --rung "$RUNG" --commit "$HEAD_SHA")
   568	  [ -n "$LAUNCH_MANIFEST_LINK" ] && PRE_ARGS+=(--launch-manifest "$LAUNCH_MANIFEST_LINK")
   569	  # Q10 / re-pin fix 1: a real arm's restart is the 40k -> 100k EXTENSION, whose
   570	  # contract binds the ORIGINAL launch identity (audited manifest bytes, job,
   571	  # uuid, launch commit, config, save-dir, seed, and the 40k anchor itself)
   572	  # without demanding that the INITIAL budget/commit equal this leg's. SMOKE
   573	  # restarts have no registered launch and keep the ordinary contract.
   574	  #
   575	  # Round 5: only the FIRST leg off the anchor resumes the audited 40k checkpoint.
   576	  # A later chunk resumes the endpoint of the previous chunk, which is evidence
   577	  # only once fa_orbit_record_restart.py has written it as a chain link — so those
   578	  # legs bind with --chain, and an unrecorded predecessor refuses the launch.
   579	  if [ "$SMOKE" != "1" ]; then
   580	    if [ "$EXPECTED_STEP" -eq 40000 ]; then
   581	      PRE_ARGS+=(--extension)
   582	    else
   583	      PRE_ARGS+=(--chain)
   584	    fi
   585	    PRE_ARGS+=(--launch-registry "$EXPDIR/arm_launch_registry.json" --repo-root "$REPO")
   586	    [ -n "$CHUNK_END" ] && PRE_ARGS+=(--chunk-end "$CHUNK_END")
   587	  fi
   588	  PRE_OUT="$(python3 "$EXPDIR/fa_orbit_ckpt_preflight.py" "${PRE_ARGS[@]}" 2>&1)"
   589	  echo "$PRE_OUT"
   590	  echo "$PRE_OUT" | grep -q "^CKPT_SHA256 " || die "restart preflight FAILED - abort"
   591	  CKPT_SHA="$(echo "$PRE_OUT" | awk '/^CKPT_SHA256 /{print $2}')"
   592	fi
   593	
   594	# --- M. wandb: scrub, pin the destination, fix the run id (round-3 B7) --------
   595	unset WANDB_MODE WANDB_DISABLED WANDB_ENTITY WANDB_RUN_ID WANDB_RESUME WANDB_DIR WANDB_PROJECT WANDB_NAME
   596	export WANDB_DIR="$REPO/$SAVEDIR"
   597	export WANDB_MODE=online
   598	eval "$(grep -E '^[[:space:]]*export[[:space:]]+WANDB_API_KEY=' ~/.bashrc 2>/dev/null | tail -1)"
   599	WANDB_ENTITY_SEEN="$(python3 - <<'PY'
   600	import sys
   601	try:
   602	    import wandb
   603	    v = wandb.Api().viewer
   604	except Exception as e:
   605	    sys.exit(f"wandb identity check FAILED: {e}")
   606	if v.email != "yh4742@princeton.edu":
   607	    sys.exit(f"wandb identity {v.email} != yh4742@princeton.edu")
   608	print(v.entity)
   609	PY
   610	)" || die "wandb identity gate FAILED (no logger fallback: the arms train with wandb) - abort"
   611	[ -n "$WANDB_ENTITY_SEEN" ] || die "wandb returned an empty entity - abort"
   612	# B7 residual: pin the destination account explicitly instead of leaving it implicit
   613	export WANDB_ENTITY="$WANDB_ENTITY_SEEN"
   614	echo "wandb identity OK: entity ${WANDB_ENTITY} (exported), mode ${WANDB_MODE}"
   615	if [ "$MODE" = "RESTART" ]; then
   616	  ORIG_WANDB_RUN_ID="$(awk '/^wandb_run_id /{print $2}' "$LAUNCH_MANIFEST_LINK" 2>/dev/null)"
   617	  [ -n "$ORIG_WANDB_RUN_ID" ] || die "RESTART needs the original wandb run id from ${LAUNCH_MANIFEST_LINK} - abort"
   618	  # A RESUMED wandb run refuses config changes: prefigure's push_wandb_config
   619	  # calls config.update() without allow_val_change, and a restart legitimately
   620	  # changes max_steps (40000->100000) and ckpt_path — legs 3684149/3684150
   621	  # died at exactly train.py:193 on this. So a restart runs as a FRESH run
   622	  # carrying the lineage in its name; curves split across runs but panels
   623	  # align on the step axis. train.py stays byte-unchanged (exp_15 admission
   624	  # pins it).
   625	  WANDB_RUN_ID="exp11-${ARM}-r${EXPECTED_STEP}-$(date +%s%N)-$(cut -c1-8 /proc/sys/kernel/random/uuid)"
   626	  export WANDB_RUN_ID
   627	  echo "wandb: new RESTART run id ${WANDB_RUN_ID} (continues ${ORIG_WANDB_RUN_ID}; not resumed — a resumed run refuses the changed max_steps/ckpt_path config)"
   628	else
   629	  WANDB_RUN_ID="exp11-${ARM}-$(date +%s%N)-$(cut -c1-8 /proc/sys/kernel/random/uuid)"
   630	  export WANDB_RUN_ID
   631	  echo "wandb: new run id ${WANDB_RUN_ID}"
   632	fi
   633	
   634	# --- N. DINOv3 pin + init-identity gate (inside the allocation) ---------------
   635	HF_HUB_OFFLINE=1 python3 "$EXPDIR/assert_arm_configs_exp11.py" "$ARM" || die "ViT-pin / init-identity GATE FAILED - abort" 1
   636	
   637	# --- O. atomic manifest, duplicated to the save-dir (round-3 B5) --------------
   638	# --- the live transcript must not be a TRACKED file while it is being written --
   639	# A tracked file that a running job appends to is a file git will happily
   640	# replace. `git rebase --autostash` (remote coordination between machines) stashes
   641	# the dirty working tree, checks out, and restores — and each of those steps
   642	# UNLINKS the path and writes a new inode. The job's stdout descriptor still
   643	# points at the old, now-nameless inode, so every subsequent line is written to a
   644	# file with no name and the visible transcript freezes forever. That is exactly
   645	# what happened to C16 (job 3648696): its transcript stops at Epoch 5 while the
   646	# run went on to step 40000, and all four arm transcripts froze at the same
   647	# instant, 02:04:07, during a rebase cycle.
   648	#
   649	# So: at launch, the job removes its OWN Slurm transcript from the index. The
   650	# file stays on disk untouched; it is simply no longer something git will move.
   651	# The operator commits completed transcripts at closure (git add -f).
   652	SLURM_OUT_AT_LAUNCH="$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null \
   653	                        | tr ' ' '\n' | awk -F= '$1=="StdOut"{print $2; exit}')"
   654	UNTRACK_STATE="not-attempted"
   655	if [ -n "$SLURM_OUT_AT_LAUNCH" ]; then
   656	  if git -C "$REPO" ls-files --error-unmatch "$SLURM_OUT_AT_LAUNCH" >/dev/null 2>&1; then
   657	    if git -C "$REPO" rm --cached --quiet -- "$SLURM_OUT_AT_LAUNCH" 2>/dev/null; then
   658	      UNTRACK_STATE="untracked-at-launch"
   659	      echo "live transcript untracked for the duration of this run: ${SLURM_OUT_AT_LAUNCH}"
   660	      echo "  (the file is untouched on disk; commit it at closure with git add -f)"
   661	    else
   662	      UNTRACK_STATE="untrack-FAILED"
   663	      echo "WARNING: could not untrack ${SLURM_OUT_AT_LAUNCH}; a git working-tree"
   664	      echo "         operation during this run can still detach the transcript"
   665	    fi
   666	  else
   667	    UNTRACK_STATE="already-untracked"      # the normal steady state
   668	  fi
   669	else
   670	  UNTRACK_STATE="stdout-path-unknown"
   671	fi
   672	
   673	TRAINLOG="${EXPDIR}/fa_orbit_${TS}_${ARM}_${RUNG}_jid${SLURM_JOB_ID}_train.log"
   674	SAVEDIR_LOG="${SAVEDIR}/fa_orbit_${TS}_${ARM}_${RUNG}_jid${SLURM_JOB_ID}_train.log"
   675	MANIFEST="${EXPDIR}/fa_orbit_${TS}_${ARM}_${RUNG}_jid${SLURM_JOB_ID}_manifest.txt"
   676	# B5 residual: a failed environment dump must not be silently hashed into the record
   677	PIPFREEZE_FILE="$(mktemp)" || die "mktemp failed - abort" 3
   678	pip freeze > "$PIPFREEZE_FILE" 2>/dev/null || die "pip freeze FAILED - the environment digest would be a lie - abort" 7
   679	ENV_SHA="$(sha256sum "$PIPFREEZE_FILE" | awk '{print $1}')"
   680	rm -f "$PIPFREEZE_FILE"
   681	{
   682	  echo "# exp_11 arm launch manifest"
   683	  echo "timestamp ${TS}"
   684	  echo "job ${SLURM_JOB_ID} host $(hostname) mode ${MODE} launch_uuid ${LAUNCH_UUID}"
   685	  echo "arm ${ARM} rung ${RUNG} micro ${MB} ngpu ${NGPU} max_steps ${MAXSTEPS} ckpt_every ${CHECKPOINT_EVERY}"
   686	  echo "commit ${HEAD_SHA}"
   687	  echo "p0_manifest_sha256 ${PINNED_P0_MANIFEST_SHA256}"
   688	  echo "model_config ${MODEL_CONFIG_ABS}"
   689	  echo "config_sha256 ${CONFIG_SHA}"
   690	  echo "vae_sha256 ${VAE_SHA}"
   691	  echo "env_pip_freeze_sha256 ${ENV_SHA}"
   692	  echo "torch_version $(python3 -c 'import torch; print(torch.__version__)') cuda ${CUDA_VER} driver ${DRIVER}"
   693	  echo "gpu_uuids ${UUID_CSV}"
   694	  echo "time_limit ${TIME_LIMIT} min_free_mb ${MIN_FREE_MB}"
   695	  echo "resume_ckpt ${RESUME_CKPT:-<none>} expected_step ${EXPECTED_STEP} resume_ckpt_sha256 ${CKPT_SHA:-<none>}"
   696	  echo "chunk_end ${CHUNK_END:-<none>}"
   697	  echo "save_dir ${SAVEDIR}"
   698	  echo "slurm_transcript ${SLURM_OUT_AT_LAUNCH:-<unknown>} untrack ${UNTRACK_STATE}"
   699	  echo "train_log ${TRAINLOG}"
   700	  echo "train_log_copy ${SAVEDIR_LOG}"
   701	  echo "wandb_entity ${WANDB_ENTITY_SEEN} wandb_project ${NAME} wandb_name ${EXPNAME}"
   702	  echo "wandb_run_id ${WANDB_RUN_ID}"
   703	  echo "command torchrun --standalone --nnodes=1 --nproc_per_node=${NGPU} train.py ${ARGV[*]}"
   704	} > "${MANIFEST}.tmp" || die "manifest write FAILED - abort" 3
   705	mv "${MANIFEST}.tmp" "$MANIFEST" || die "manifest publication FAILED - abort" 3
   706	cp "$MANIFEST" "${SAVEDIR}/$(basename "$MANIFEST")" || die "manifest copy to the save-dir FAILED - abort" 3
   707	[ "$MODE" = "INITIAL" ] && { cp "$MANIFEST" "$LAUNCH_MANIFEST_LINK" || die "launch-manifest link write FAILED - abort" 3; }
   708	echo "manifest: ${MANIFEST} (copied to ${SAVEDIR})"
   709	
   710	# --- P. training: torchrun + FIFO tee (both statuses captured) ----------------
   711	unset SLURM_NTASKS SLURM_JOB_NAME SLURM_PROCID SLURM_LOCALID SLURM_NODEID
   712	: > "$TRAINLOG" || die "could not create ${TRAINLOG} - abort" 3
   713	: > "$SAVEDIR_LOG" || die "could not create ${SAVEDIR_LOG} - abort" 3
   714	FIFO="$(mktemp)" || die "mktemp failed - abort" 3
   715	rm -f "$FIFO" && mkfifo "$FIFO" || die "mkfifo failed - abort" 3
   716	trap 'rm -f "$FIFO"' EXIT
   717	tee -a "$TRAINLOG" "$SAVEDIR_LOG" < "$FIFO" &
   718	TEE_PID=$!
   719	echo "=== launching ${ARM} ${RUNG}: torchrun --standalone --nproc_per_node=${NGPU} train.py (${MODE}) ==="
   720	torchrun --standalone --nnodes=1 --nproc_per_node="$NGPU" train.py "${ARGV[@]}" > "$FIFO" 2>&1 &
   721	TR_PID=$!
   722	
   723	# world-size watcher: terminate the moment Lightning reports the WRONG rank count
   724	WORLD_RE="All distributed processes registered\. Starting with [0-9]+ processes"
   725	(
   726	  while kill -0 "$TR_PID" 2>/dev/null; do
   727	    if grep -Eq "$WORLD_RE" "$TRAINLOG" 2>/dev/null; then
   728	      GOT="$(grep -Eo "$WORLD_RE" "$TRAINLOG" | head -1 | grep -Eo '[0-9]+')"
   729	      if [ "$GOT" != "$NGPU" ]; then
   730	        echo "WORLD-SIZE WATCHER: Lightning started with ${GOT} processes, expected ${NGPU} — terminating torchrun"
   731	        pkill -TERM -P "$TR_PID" 2>/dev/null; kill -TERM "$TR_PID" 2>/dev/null
   732	      fi
   733	      exit 0
   734	    fi
   735	    sleep 5
   736	  done
   737	) &
   738	WATCHER=$!
   739	
   740	wait "$TR_PID"; rc=$?
   741	kill "$WATCHER" 2>/dev/null; wait "$WATCHER" 2>/dev/null
   742	wait "$TEE_PID"; tee_rc=$?
   743	rm -f "$FIFO"
   744	
   745	# --- Q. W&B run identity verification (round-3 B7 residual) ------------------
   746	# The gate above proves WHO we are; this proves WHERE the run actually landed.
   747	# train.py:165 builds WandbLogger(project=, name=) with NO save_dir, so PL passes
   748	# its default save_dir='.' into wandb.init and that OVERRIDES the exported
   749	# WANDB_DIR: in job 3646734 the run went to $REPO/wandb/run-<ts>-<id> while this
   750	# check looked under $WANDB_DIR/wandb and found nothing (training was green, the
   751	# job still classified 7). We keep exporting WANDB_DIR — other wandb artifacts do
   752	# respect it — but locate the run by the collision-proof id WE generated, which
   753	# wandb embeds in the directory name, across both candidate roots. Exactly one
   754	# match is required; a mismatch is a provenance failure (class 7), not a footnote.
   755	WANDB_CHECK_RC=0
   756	python3 "$EXPDIR/fa_orbit_wandb_readback.py" --run-id "$WANDB_RUN_ID" \
   757	  --root "$REPO" --root "$WANDB_DIR" \
   758	  --entity "$WANDB_ENTITY" --project "$NAME" --name "$EXPNAME" || WANDB_CHECK_RC=$?
   759	if [ "$WANDB_CHECK_RC" -ne 0 ]; then
   760	  echo "W&B run identity could not be verified against the manifest - provenance failure"
   761	fi
   762	
   763	# --- R. classification (tested taxonomy; see fa_orbit_classify.py) ------------
   764	# The completion literal Lightning prints is the stop step it was GIVEN, so the
   765	# classifier is told the step this leg was asked to reach. Without a chunk that
   766	# is MAXSTEPS, byte-identical to every pre-round-5 leg; with one it is the chunk
   767	# boundary, and a successful chunk classifies COMPLETE instead of NO-MARKER.
   768	CLASSIFY_OUT="$(python3 "$EXPDIR/fa_orbit_classify.py" --rc "$rc" --tee-rc "$tee_rc" --ngpu "$NGPU" \
   769	  --maxsteps "$STOPSTEP" --log "$TRAINLOG" --log-copy "$SAVEDIR_LOG" 2>&1)"
   770	final_rc=$?
   771	echo "$CLASSIFY_OUT"
   772	LAST_CKPT="$(ls -1t "${RUNDIR}/checkpoints"/*.ckpt 2>/dev/null | head -1)"
   773	
   774	# --- S. durable final record (round-3 B5 residual) ---------------------------
   775	# The classifier verdict and this summary must survive in BOTH copies; the tee
   776	# status is captured (it used to be discarded) and a failure is class 7. The
   777	# whole preflight transcript (this job's Slurm output so far) is copied into the
   778	# save-dir too, so the durable record is not just torchrun's stdout.
   779	FINAL_RECORD="$(
   780	  echo "=== exp_11 ${ARM} ${RUNG} finished at $(date '+%Y-%m-%d %H:%M:%S') ==="
   781	  echo "torchrun rc=${rc} tee rc=${tee_rc} wandb_identity_rc=${WANDB_CHECK_RC} classified rc=${final_rc}"
   782	  echo "$CLASSIFY_OUT"
   783	  echo "last checkpoint: ${LAST_CKPT:-<none>}"
   784	  echo "restart: fa_orbit_submit.sh ${ARM} --resume ${LAST_CKPT:-<path>} --expected-step <step>"
   785	  [ -n "$CHUNK_END" ] && echo "next chunk: record this leg (fa_orbit_record_restart.py ${ARM} ${MANIFEST}), then fa_orbit_submit.sh ${ARM} --resume <ckpt at ${CHUNK_END}> --expected-step ${CHUNK_END} --chunk-end <${CHUNK_END}+${PINNED_CHECKPOINT_EVERY}>"
   786	)"
   787	printf '%s\n' "$FINAL_RECORD" | tee -a "$TRAINLOG" "$SAVEDIR_LOG" > /dev/null
   788	final_tee_rc="${PIPESTATUS[1]}"
   789	# The transcript path is NOT guessed from the #SBATCH pattern (job 3646734 logged
   790	# "slurm_train_job_3646734.out not found" because SLURM_JOB_NAME is unset inside
   791	# the job): ask Slurm itself.
   792	# --- BEGIN slurm-stdout-helper (guard-tested by fa_orbit_equivprobe_wrapper_test.sh) --
   793	parse_slurm_stdout() {   # `scontrol show job` text on stdin -> the StdOut path
   794	  tr ' ' '\n' | awk -F= '$1=="StdOut"{print $2; exit}'
   795	}
   796	slurm_stdout_path() {    # $1 = job id
   797	  scontrol show job "$1" 2>/dev/null | parse_slurm_stdout
   798	}
   799	# --- END slurm-stdout-helper --------------------------------------------------
   800	SLURM_OUT="$(slurm_stdout_path "$SLURM_JOB_ID")"
   801	PREFLIGHT_COPY_RC=0
   802	if [ -n "$SLURM_OUT" ] && [ -f "$SLURM_OUT" ]; then
   803	  cp "$SLURM_OUT" "${SAVEDIR}/preflight_${TS}_jid${SLURM_JOB_ID}.out" || PREFLIGHT_COPY_RC=$?
   804	  [ "$PREFLIGHT_COPY_RC" -eq 0 ] && echo "preflight transcript duplicated from ${SLURM_OUT}"
   805	else
   806	  echo "PREFLIGHT TRANSCRIPT: scontrol reported StdOut='${SLURM_OUT}' which is not a readable file"
   807	  PREFLIGHT_COPY_RC=1
   808	fi
   809	if [ "$final_tee_rc" -ne 0 ] || [ "$PREFLIGHT_COPY_RC" -ne 0 ] || [ "$WANDB_CHECK_RC" -ne 0 ]; then
   810	  echo "LOG-PROVENANCE: final record tee rc=${final_tee_rc}, preflight copy rc=${PREFLIGHT_COPY_RC}, wandb identity rc=${WANDB_CHECK_RC}"
   811	  [ "$final_rc" -eq 0 ] && final_rc=7
   812	fi
   813	printf '%s\n' "$FINAL_RECORD"
   814	exit "$final_rc"

exec
/usr/bin/bash -lc "nl -ba worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch | sed -n '391,505p'
nl -ba worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py | sed -n '1,470p'" in /n/fs/gatrdp/codespace/FLAC
 succeeded in 0ms:
   391	# --- F. the exact train.py argv ----------------------------------------------
   392	ARGV=(
   393	  --model-config "$MODEL_CONFIG_ABS"
   394	  --dataset-config src/configs/dataset_configs/AR/train/acousticroom_train.json
   395	  --pretransform-ckpt-path weights/FLAC/VAE.safetensors
   396	  --max-steps "$STOPSTEP" --batch-size "$MB" --accum-batches 1 --num-workers 6 --seed 42
   397	  --num-gpus "$NGPU" --num-nodes 1
   398	  --strategy ddp_find_unused_parameters_true --sync-batchnorm true --precision bf16-mixed
   399	  --val-every -1 --val-dataset-config ''
   400	  --gradient-clip-val 0.0
   401	  --logger wandb --checkpoint-every "$CHECKPOINT_EVERY"
   402	  --name "$NAME" --experiment-name "$EXPNAME" --save-dir "$SAVEDIR"
   403	)
   404	[ "$MODE" = "RESTART" ] && ARGV+=(--ckpt-path "$RESUME_CKPT")
   405	
   406	# --- G. argv-parity dry run (plan N13; round-3 N9 tightened) ------------------
   407	ARGV_FILE="$(mktemp)" || die "mktemp failed - abort" 3
   408	printf '%s\n' "${ARGV[@]}" > "$ARGV_FILE" || die "could not write the argv file - abort" 3
   409	python3 - "$ARGV_FILE" "$MODE" <<'PY'
   410	import sys
   411	# The exp_07 B-F reference argv (bf_scratch_launch.sh) — the lineage this sweep continues.
   412	REF = """--model-config worklog/worklog_yixun/exp_07_fa_scratch_claude/FLAC_AR_BF.json
   413	--dataset-config src/configs/dataset_configs/AR/train/acousticroom_train.json
   414	--pretransform-ckpt-path weights/FLAC/VAE.safetensors
   415	--max-steps 67500 --batch-size 32 --accum-batches 1 --num-workers 6 --seed 42
   416	--num-gpus 2 --strategy ddp_find_unused_parameters_true --sync-batchnorm true
   417	--logger wandb --checkpoint-every 2500
   418	--name FLAC_exp07_BF --experiment-name exp07_BF --save-dir outputs_FLAC/exp07_BF""".split()
   419	# Flags whose VALUE may differ from exp_07 (identity, budget, rung, resume):
   420	ALLOWED_DIFF = {"--model-config", "--name", "--experiment-name", "--save-dir", "--max-steps",
   421	                "--num-gpus", "--batch-size", "--logger", "--checkpoint-every", "--ckpt-path"}
   422	# Flags exp_07 left to defaults.ini and we state explicitly — whitelisted with their
   423	# EXACT expected values (round-3 N9: no "equals the mutable ini" escape hatch):
   424	ALLOWED_ADD = {"--num-nodes": "1", "--precision": "bf16-mixed", "--val-every": "-1",
   425	               "--val-dataset-config": "", "--gradient-clip-val": "0.0", "--ckpt-path": None}
   426	tokens = [t for t in open(sys.argv[1]).read().split("\n")]
   427	if tokens and tokens[-1] == "":
   428	    tokens.pop()
   429	mode = sys.argv[2]
   430	
   431	def as_map(toks):
   432	    out, i = {}, 0
   433	    while i < len(toks):
   434	        flag = toks[i]
   435	        if not flag.startswith("--"):
   436	            raise SystemExit(f"ARGV PARITY: stray token {flag!r}")
   437	        val = toks[i + 1] if i + 1 < len(toks) and not toks[i + 1].startswith("--") else ""
   438	        if flag in out:
   439	            raise SystemExit(f"ARGV PARITY: duplicate flag {flag}")
   440	        out[flag] = val
   441	        i += 2 if (i + 1 < len(toks) and not toks[i + 1].startswith("--")) else 1
   442	    return out
   443	
   444	ref, new = as_map(REF), as_map(tokens)
   445	violations, allowed, explicit = [], [], []
   446	for flag in sorted(set(ref) | set(new)):
   447	    if flag in ref and flag in new:
   448	        if ref[flag] != new[flag]:
   449	            (allowed if flag in ALLOWED_DIFF else violations).append(
   450	                f"{flag}: exp_07 {ref[flag]!r} -> exp_11 {new[flag]!r}")
   451	    elif flag in new:
   452	        if flag == "--ckpt-path":
   453	            (allowed if mode == "RESTART" else violations).append(
   454	                f"--ckpt-path: {new[flag]!r} (RESTART only)")
   455	        elif flag in ALLOWED_ADD and ALLOWED_ADD[flag] == new[flag]:
   456	            explicit.append(f"{flag}={new[flag]!r} (whitelisted explicit default)")
   457	        else:
   458	            violations.append(f"{flag}: added with {new[flag]!r}, not a whitelisted addition "
   459	                              f"(expected {ALLOWED_ADD.get(flag, '<not allowed>')!r})")
   460	    else:
   461	        violations.append(f"{flag}: present in exp_07 ({ref[flag]!r}), MISSING here")
   462	
   463	print("--- train.py argv ---")
   464	print(" ".join(f"{k} {v!r}" if v == "" else f"{k} {v}" for k, v in new.items()))
   465	print("--- argv parity vs exp_07 B-F ---")
   466	for d in allowed:
   467	    print(f"  allowed  {d}")
   468	for d in explicit:
   469	    print(f"  explicit {d}")
   470	if violations:
   471	    print("ARGV PARITY VIOLATIONS:")
   472	    for v in violations:
   473	        print(f"  !! {v}")
   474	    raise SystemExit(2)
   475	print(f"ARGV PARITY OK ({mode}): only whitelisted differences and additions")
   476	PY
   477	parity=$?
   478	rm -f "$ARGV_FILE"
   479	[ "$parity" -eq 0 ] || die "argv parity check FAILED - abort"
   480	if [ "$DRYRUN" = "1" ]; then
   481	  echo "DRY RUN complete: gates A–G passed for ARM=${ARM} RUNG=${RUNG} MODE=${MODE} SMOKE=${SMOKE}"
   482	  echo "  (Slurm/GPU/VRAM/env/wandb/ViT/lock gates and training are skipped in DRYRUN)"
   483	  exit 0
   484	fi
   485	
   486	# --- H. Slurm allocation must match the pins (round-3 B1) ---------------------
   487	[ "${SLURM_JOB_NUM_NODES:-1}" = "1" ] || die "expected 1 node, got ${SLURM_JOB_NUM_NODES} - abort"
   488	[ "${SLURM_NTASKS:-1}" = "1" ] || die "expected 1 task, got ${SLURM_NTASKS} - abort"
   489	WANT_CPUS="$((8 + 7 * NGPU))"; WANT_MEM_MB="$(((12 * NGPU + 12) * 1024))"
   490	GOT_CPUS="${SLURM_CPUS_PER_TASK:-${SLURM_CPUS_ON_NODE:-0}}"
   491	GOT_MEM_MB="${SLURM_MEM_PER_NODE:-0}"
   492	[ "$GOT_CPUS" = "$WANT_CPUS" ] || die "allocated ${GOT_CPUS} CPUs, the pinned rung needs ${WANT_CPUS} — submit via fa_orbit_submit.sh - abort"
   493	[ "$GOT_MEM_MB" = "$WANT_MEM_MB" ] || die "allocated ${GOT_MEM_MB} MB RAM, the pinned rung needs ${WANT_MEM_MB} — submit via fa_orbit_submit.sh - abort"
   494	GOT_TIME="$(squeue -h -j "$SLURM_JOB_ID" -o %l 2>/dev/null | tr -d ' ')"
   495	norm_minutes() { awk -v t="$1" 'BEGIN{d=0; if (t ~ /-/) {split(t,p,"-"); d=p[1]; t=p[2]} n=split(t,c,":");
   496	  if (n==3) m=c[1]*60+c[2]+c[3]/60; else if (n==2) m=c[1]+c[2]/60; else m=t; printf "%d", d*1440+m}'; }
   497	# The pin this ${MODE} leg is entitled to — an INITIAL allocation handed to a
   498	# RESTART leg (or the reverse) is refused here, in the job, not merely intended
   499	# by the submitter.
   500	[ "$(norm_minutes "$GOT_TIME")" = "$(norm_minutes "$TIME_LIMIT")" ] || die "allocated time ${GOT_TIME} != the ${TIME_PIN_NAME} pin ${TIME_LIMIT} this ${MODE} leg requires — submit via fa_orbit_submit.sh - abort"
   501	echo "allocation matches the pins: ${GOT_CPUS} cpus, ${GOT_MEM_MB} MB, ${GOT_TIME} (${TIME_PIN_NAME})"
   502	
   503	mapfile -t GPU_ROWS < <(nvidia-smi --query-gpu=uuid,name --format=csv,noheader,nounits)
   504	[ "${#GPU_ROWS[@]}" -eq "$NGPU" ] || die "rung ${RUNG} needs exactly ${NGPU} visible GPU(s), got ${#GPU_ROWS[@]} - abort"
   505	UUIDS=()
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
   244	def check_chain_binding(manifest_path, registry_path, arm, rung, config_path, ckpt_path,
   245	                        ckpt_sha, expected_step, max_steps, repo_root="."):
   246	    """The CHUNKED-LEG contract: resume the TIP of this arm's recorded chain.
   247	
   248	    A chunk leg is an extension leg that stops early, so it must prove exactly
   249	    the same INITIAL launch identity — but its resume point is no longer the
   250	    audited 40k anchor: it is the endpoint of the previous chunk, recorded as the
   251	    last link of ``arms.<ARM>.chain`` by fa_orbit_record_restart.py.
   252	
   253	    Fail-closed by construction: an EMPTY or ABSENT chain refuses, so chunk N+1
   254	    cannot run until chunk N has been recorded, and every link's final_step /
   255	    final_ckpt_sha256 was written by the recorder from the file on disk. The
   256	    resume file is re-hashed by the caller (``ckpt_sha``); a manifest's claimed
   257	    hash is never trusted here or anywhere else in this chain.
   258	    """
   259	    reg, problems = _load_registry_row(registry_path, arm)
   260	    if problems:
   261	        return problems, {}
   262	    man = parse_manifest(manifest_path)
   263	    problems = check_initial_identity(man, manifest_path, reg, arm, rung, config_path, max_steps)
   264	    if not reg.get("final_ckpt_sha256"):
   265	        problems.append(f"{arm} has no audited final_ckpt_sha256 in the registry — a chain of "
   266	                        "chunks must still descend from the audited INITIAL run")
   267	    chain = reg.get("chain") or []
   268	    if not chain:
   269	        problems.append(f"{arm} has no recorded chain link in {registry_path}: a chunk may only "
   270	                        "resume the tip of a RECORDED chain, so chunk N+1 is inadmissible until "
   271	                        "chunk N is recorded — run fa_orbit_record_restart.py on the previous "
   272	                        "chunk's launcher manifest first")
   273	    else:
   274	        tip = chain[-1]
   275	        tip_step, tip_sha = tip.get("final_step"), tip.get("final_ckpt_sha256")
   276	        if str(tip_step) != str(expected_step):
   277	            problems.append(f"EXPECTED_STEP {expected_step} != the last recorded chain link's "
   278	                            f"final_step {tip_step!r} (link job {tip.get('job')!r}) — a chunk "
   279	                            "resumes the TIP of the chain, never an earlier link")
   280	        if not tip_sha:
   281	            problems.append(f"the last recorded chain link (job {tip.get('job')!r}) carries no "
   282	                            "final_ckpt_sha256 — it is not evidence of a checkpoint")
   283	        elif ckpt_sha != tip_sha:
   284	            problems.append(f"resume checkpoint sha256 {ckpt_sha[:12]} != the last recorded chain "
   285	                            f"link's final_ckpt_sha256 {str(tip_sha)[:12]} — this file is not the "
   286	                            "checkpoint that chunk produced")
   287	    problems += check_canonical_dir(man, arm, ckpt_path, repo_root)
   288	    return problems, man
   289	
   290	
   291	def main(argv=None):
   292	    ap = argparse.ArgumentParser(description="exp_11 restart checkpoint preflight")
   293	    ap.add_argument("--ckpt", required=True)
   294	    ap.add_argument("--expected-step", type=int, required=True)
   295	    ap.add_argument("--config", required=True, help="this arm's model config json")
   296	    ap.add_argument("--max-steps", type=int, required=True)
   297	    ap.add_argument("--arm", required=True)
   298	    ap.add_argument("--rung", required=True)
   299	    ap.add_argument("--commit", default="")
   300	    ap.add_argument("--launch-manifest", default="",
   301	                    help="the arm's original launch manifest (binds rung/commit/budget)")
   302	    ap.add_argument("--extension", action="store_true",
   303	                    help="this restart is the Q10 40k->100k EXTENSION: bind it to the audited "
   304	                         "INITIAL launch identity instead of requiring an equal budget/commit")
   305	    ap.add_argument("--chain", action="store_true",
   306	                    help="this restart is a CHUNK continuing an already-recorded chain: same "
   307	                         "INITIAL identity as --extension, but the resume anchor is the LAST "
   308	                         "recorded arms.<ARM>.chain link instead of the audited 40k checkpoint")
   309	    ap.add_argument("--chunk-end", type=int, default=None,
   310	                    help="this leg's stop step (a chunk boundary): EXPECTED_STEP < N <= max-steps "
   311	                         "and a multiple of 2500")
   312	    ap.add_argument("--launch-registry", default="",
   313	                    help="the committed arm launch registry (required with --extension/--chain)")
   314	    ap.add_argument("--repo-root", default=".",
   315	                    help="root the registry's relative save_dir is resolved against")
   316	    args = ap.parse_args(argv)
   317	    if args.extension and args.chain:
   318	        ap.error("--extension and --chain are mutually exclusive: a leg either resumes the audited "
   319	                 "40k anchor or the tip of the recorded chunk chain, never both")
   320	    if args.extension and not args.launch_registry:
   321	        ap.error("--extension requires --launch-registry (the audited INITIAL launch row)")
   322	    if args.chain and not args.launch_registry:
   323	        ap.error("--chain requires --launch-registry (the recorded chunk chain lives in it)")
   324	
   325	    if not os.path.isfile(args.ckpt):
   326	        print(f"PREFLIGHT: checkpoint not found: {args.ckpt}")
   327	        return 2
   328	
   329	    import torch  # deferred: keeps --help and unit imports cheap
   330	
   331	    try:
   332	        ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
   333	    except Exception as exc:                      # truncated, empty or not a torch file
   334	        print(f"PREFLIGHT: {args.ckpt} is not loadable as a checkpoint: "
   335	              f"{type(exc).__name__}: {exc}")
   336	        return 2
   337	    if not isinstance(ck, dict):
   338	        print(f"PREFLIGHT: not a Lightning checkpoint: {args.ckpt}")
   339	        return 2
   340	
   341	    problems = []
   342	    gs = ck.get("global_step")
   343	    if gs != args.expected_step:
   344	        problems.append(f"global_step {gs} != EXPECTED_STEP {args.expected_step}")
   345	    if isinstance(gs, int) and gs >= args.max_steps:
   346	        problems.append(f"global_step {gs} >= max_steps {args.max_steps}: Lightning would stop "
   347	                        "immediately and still print the completion literal")
   348	
   349	    mc = ck.get("model_config")
   350	    if not isinstance(mc, dict):
   351	        problems.append("checkpoint carries no embedded 'model_config' dict")
   352	    else:
   353	        want = json.load(open(args.config))
   354	        if mc != want:
   355	            tr = mc.get("training", {}) if isinstance(mc.get("training"), dict) else {}
   356	            problems.append(
   357	                f"embedded model_config != {args.config} (parsed-object mismatch; embedded "
   358	                f"cond_method={tr.get('cond_method')!r} angles={tr.get('frame_avg_angles')!r})")
   359	
   360	    opts = ck.get("optimizer_states") or []
   361	    if len(opts) != 1:
   362	        problems.append(f"expected exactly 1 optimizer entry, found {len(opts)}")
   363	    elif not opts[0].get("state"):
   364	        problems.append("optimizer state is CLEARED (stripped checkpoint); exp_11 restarts are "
   365	                        "WARM continuations and have no optimizer-reset lineage")
   366	    if not ck.get("lr_schedulers"):
   367	        problems.append("no 'lr_schedulers' -> PL 2.1 KeyErrors on resume")
   368	    sd = ck.get("state_dict") or {}
   369	    n_ema = sum(1 for k in sd if k.startswith("diffusion_ema."))
   370	    if not n_ema:
   371	        problems.append("no EMA weights in state_dict")
   372	
   373	    # The per-job stop step is checked against the BUDGET (--max-steps), which the
   374	    # launcher still pins at 100000: a chunk narrows where this leg stops, never
   375	    # what the campaign is allowed to reach.
   376	    if args.chunk_end is not None:
   377	        if args.chunk_end % 2500 != 0:
   378	            problems.append(f"--chunk-end {args.chunk_end} is not a multiple of 2500 (the pinned "
   379	                            "checkpoint cadence: a chunk must end ON a checkpoint)")
   380	        if not args.expected_step < args.chunk_end <= args.max_steps:
   381	            problems.append(f"--chunk-end {args.chunk_end} must satisfy EXPECTED_STEP "
   382	                            f"{args.expected_step} < chunk_end <= max_steps {args.max_steps}")
   383	
   384	    digest = sha256_file(args.ckpt)     # needed by the extension/chain anchor checks
   385	    man = {}
   386	    if args.launch_manifest:
   387	        if not os.path.isfile(args.launch_manifest):
   388	            problems.append(f"launch manifest not found: {args.launch_manifest}")
   389	        elif args.extension:
   390	            more, man = check_extension_binding(
   391	                args.launch_manifest, args.launch_registry, args.arm, args.rung, args.config,
   392	                args.ckpt, digest, args.expected_step, args.max_steps, args.repo_root)
   393	            problems += more
   394	        elif args.chain:
   395	            more, man = check_chain_binding(
   396	                args.launch_manifest, args.launch_registry, args.arm, args.rung, args.config,
   397	                args.ckpt, digest, args.expected_step, args.max_steps, args.repo_root)
   398	            problems += more
   399	        else:
   400	            more, man = check_manifest_binding(args.launch_manifest, args.arm, args.rung,
   401	                                               args.commit, args.max_steps)
   402	            problems += more
   403	    elif args.extension:
   404	        problems.append("--extension requires --launch-manifest (the audited INITIAL manifest)")
   405	    elif args.chain:
   406	        problems.append("--chain requires --launch-manifest (the audited INITIAL manifest)")
   407	
   408	    if problems:
   409	        print("PREFLIGHT FAILED:")
   410	        for p in problems:
   411	            print(f"  !! {p}")
   412	        return 2
   413	
   414	    tr = mc.get("training", {})
   415	    print(f"restart lineage OK: {args.ckpt}")
   416	    print(f"  global_step={gs} epoch={ck.get('epoch')} cond_method={tr.get('cond_method')!r} "
   417	          f"angles={tr.get('frame_avg_angles')}")
   418	    print(f"  optimizer_state=FULL ({len(opts[0]['state'])} entries) "
   419	          f"lr={opts[0]['param_groups'][0].get('lr')} "
   420	          f"sched_last_epoch={ck['lr_schedulers'][0].get('last_epoch')} ema_entries={n_ema}")
   421	    if man and args.extension:
   422	        print(f"extension lineage OK: {args.arm} {args.expected_step} -> "
   423	              f"{args.chunk_end if args.chunk_end is not None else args.max_steps} continues "
   424	              f"the audited launch job {kv_line(man, 'job').get('job')} "
   425	              f"(launch commit {man.get('commit', '')[:12]}, running commit {args.commit[:12] or '<none>'})")
   426	        print(f"  bound to the audited launch manifest: {args.launch_manifest}")
   427	        if args.chunk_end is not None:
   428	            print(f"  chunk leg: stops at {args.chunk_end} of the {args.max_steps} budget")
   429	    elif man and args.chain:
   430	        print(f"chain lineage OK: {args.arm} {args.expected_step} -> "
   431	              f"{args.chunk_end if args.chunk_end is not None else args.max_steps} resumes the tip "
   432	              f"of the recorded chunk chain, under the audited launch job "
   433	              f"{kv_line(man, 'job').get('job')} "
   434	              f"(launch commit {man.get('commit', '')[:12]}, running commit {args.commit[:12] or '<none>'})")
   435	        print(f"  bound to the audited launch manifest: {args.launch_manifest}")
   436	        if args.chunk_end is not None:
   437	            print(f"  chunk leg: stops at {args.chunk_end} of the {args.max_steps} budget")
   438	    elif man:
   439	        print(f"  bound to launch manifest: {args.launch_manifest}")
   440	    print(f"CKPT_SHA256 {digest}")
   441	    return 0
   442	
   443	
   444	if __name__ == "__main__":
   445	    sys.exit(main())

exec
/usr/bin/bash -lc 'git ls-files --error-unmatch worklog/worklog_yixun/exp_11_fa_orbit_claude/arm_launch_registry.json >/dev/null 2>&1; echo registry_tracked_rc=$?
git status --short --untracked-files=no -- worklog/worklog_yixun/exp_11_fa_orbit_claude/arm_launch_registry.json
sed -n '"'1,260p' worklog/worklog_yixun/exp_11_fa_orbit_claude/arm_launch_registry.json
sed -n '1,260p' worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_chunk_kit_review.md" in /n/fs/gatrdp/codespace/FLAC
 succeeded in 0ms:
registry_tracked_rc=0
{
  "_comment": [
    "AUDITED exp_11 arm launch registry (final GO-check item 4 / review (b)).",
    "The launch manifests live under gitignored outputs_FLAC and are therefore",
    "MUTABLE evidence: binding a screen to 'whatever the manifest says now' proves",
    "nothing. This committed registry pins each arm's manifest by sha256 plus the",
    "launch facts a screen must re-check (job, INITIAL mode, launch uuid, commit,",
    "rung, config/VAE/P0 hashes, canonical save-dir, training seed).",
    "VANL (Q9, job 3661520) recorded from its PUBLISHED launch manifest after the",
    "INITIAL job started -- an sbatch return alone does not prove a launch happened.",
    "Recorded from the live manifests of the running arms (3648694-97, 3661520).",
    "RESTART legs (Q10, 40k -> 100k) are recorded under 'restarts' as a CHAIN:",
    "each entry's resume_ckpt_sha256 must equal the INITIAL entry's",
    "final_ckpt_sha256, which is the audited 40k checkpoint that leg resumed",
    "from. A checkpoint above 40k is therefore admissible only if the restart",
    "that produced it provably descends from the audited INITIAL run.",
    "Populate with fa_orbit_record_restart.py once a leg's manifest publishes.",
    "Chaining alone is NOT sufficient (re-pin review, finding 2): a leg is only",
    "evidence for the checkpoints it actually PRODUCED, so every leg carries a",
    "'producer_manifest' -- fa_orbit_producer_<ARM>_job<JOB>.json, append-only,",
    "step -> sha256 re-hashed from disk by the recorder. The screen re-hashes the",
    "checkpoint it is about to evaluate and admits it only on an exact",
    "step/sha256/path match published by a leg whose every field here",
    "re-validates against the arm's INITIAL row."
  ],
  "recorded_at": "2026-08-09",
  "training_seed": 42,
  "arms": {
    "C4L": {
      "manifest_path": "outputs_FLAC/exp11_C4L/launch_manifest.txt",
      "manifest_sha256": "d49df42d2f7f9c3f39f1aeb6631da84ef0e0a392c22a8271edadbd83885e814a",
      "job": "3648694",
      "mode": "INITIAL",
      "launch_uuid": "ceb40a63-6ce3-4d38-a2b8-d6c21f1b8cc7",
      "commit": "2b78f995a6d377676bd9d9fb60635ab90032b52d",
      "rung": "8x8",
      "micro": "8",
      "ngpu": "8",
      "max_steps": "40000",
      "config_sha256": "3e677204902c543801679610b58d818de3f6353e7b95baded2667078135ea328",
      "vae_sha256": "8d82159eec35210198246f449bec6561fc19b514922f340a17515050daf7f0b9",
      "p0_manifest_sha256": "72607b922177208d56055d604b292d697b643ef3b7ab48261ab2e23a0cc2b53b",
      "save_dir": "outputs_FLAC/exp11_C4L",
      "training_seed": 42,
      "final_ckpt_sha256": "ed9d7a869ecded98cab78ecc4cef83e579df6643c8ffe564912a9e8ec5c88de8",
      "final_step": 40000
    },
    "C8": {
      "manifest_path": "outputs_FLAC/exp11_C8/launch_manifest.txt",
      "manifest_sha256": "fa1037c300fa3f1100667634864653690049271bd4e2815e419fb205c9068388",
      "job": "3648695",
      "mode": "INITIAL",
      "launch_uuid": "1009ec7f-e5e2-443d-881e-57ae15525676",
      "commit": "2b78f995a6d377676bd9d9fb60635ab90032b52d",
      "rung": "8x8",
      "micro": "8",
      "ngpu": "8",
      "max_steps": "40000",
      "config_sha256": "af4c2f21c2d03bc51448a3c161878443a8e9afd1314695530baf4032e808b3f1",
      "vae_sha256": "8d82159eec35210198246f449bec6561fc19b514922f340a17515050daf7f0b9",
      "p0_manifest_sha256": "72607b922177208d56055d604b292d697b643ef3b7ab48261ab2e23a0cc2b53b",
      "save_dir": "outputs_FLAC/exp11_C8",
      "training_seed": 42,
      "final_ckpt_sha256": "4b58787774b5f15852f2603295d57bbe7a1e2d4e40ba217e548f55d643a52c4c",
      "final_step": 40000
    },
    "C16": {
      "manifest_path": "outputs_FLAC/exp11_C16/launch_manifest.txt",
      "manifest_sha256": "deb07b532fea037d9354b5c635e9ad6a80ad8c022dabdc6dbe0a879a27be3979",
      "job": "3648696",
      "mode": "INITIAL",
      "launch_uuid": "87e0d920-a64f-46f4-a085-2807ebf0f13c",
      "commit": "2b78f995a6d377676bd9d9fb60635ab90032b52d",
      "rung": "8x8",
      "micro": "8",
      "ngpu": "8",
      "max_steps": "40000",
      "config_sha256": "5d8b5c4390dd0e87685bba6ed06566fabe2e1ce5074cce3f6a1f38c442829a80",
      "vae_sha256": "8d82159eec35210198246f449bec6561fc19b514922f340a17515050daf7f0b9",
      "p0_manifest_sha256": "72607b922177208d56055d604b292d697b643ef3b7ab48261ab2e23a0cc2b53b",
      "save_dir": "outputs_FLAC/exp11_C16",
      "training_seed": 42,
      "final_ckpt_sha256": "a8a82309fe4a654efa8cbb00c6ffeeaa2fc765c5916b46cf12aaf2ec2b2d01f0",
      "final_step": 40000
    },
    "C32": {
      "manifest_path": "outputs_FLAC/exp11_C32/launch_manifest.txt",
      "manifest_sha256": "b2d08bc0f27583bd78845e281380906b7f05a737444525f4e32cafa5106b395e",
      "job": "3648697",
      "mode": "INITIAL",
      "launch_uuid": "64c9d2e0-88b7-4730-b36d-e6320274b1e2",
      "commit": "2b78f995a6d377676bd9d9fb60635ab90032b52d",
      "rung": "8x8",
      "micro": "8",
      "ngpu": "8",
      "max_steps": "40000",
      "config_sha256": "8584498e8e3b5d7014e61bc4c48ef71ca62139cdc9f2f30de031cae8da34abdf",
      "vae_sha256": "8d82159eec35210198246f449bec6561fc19b514922f340a17515050daf7f0b9",
      "p0_manifest_sha256": "72607b922177208d56055d604b292d697b643ef3b7ab48261ab2e23a0cc2b53b",
      "save_dir": "outputs_FLAC/exp11_C32",
      "training_seed": 42,
      "final_ckpt_sha256": "1beca05a02ffc7b72884e54671ff2d6836789fadcfe2aa6f0ce27ca2f739c134",
      "final_step": 40000,
      "final_ckpt_path": "outputs_FLAC/exp11_C32/FLAC_exp11_C32/exp11_C32/checkpoints/epoch=8-step=40000.ckpt",
      "anchored_at": "2026-08-10T23:40:52-04:00"
    },
    "VANL": {
      "manifest_path": "outputs_FLAC/exp11_VANL/launch_manifest.txt",
      "manifest_sha256": "113d06a284c6198cf9487e99a2efb7ccde94ae13e656a403fe2af0281d3de8b1",
      "job": "3661520",
      "mode": "INITIAL",
      "launch_uuid": "7337b7ae-7d43-47aa-86b3-87b1daad9be7",
      "commit": "81ddac372076ea92751ae09cbaf371df70f396e5",
      "rung": "8x8",
      "micro": "8",
      "ngpu": "8",
      "max_steps": "40000",
      "config_sha256": "733ca52b66c43538e1b9e603e979678af95ac05d89fd1d481ebb472a285a49d8",
      "vae_sha256": "8d82159eec35210198246f449bec6561fc19b514922f340a17515050daf7f0b9",
      "p0_manifest_sha256": "72607b922177208d56055d604b292d697b643ef3b7ab48261ab2e23a0cc2b53b",
      "save_dir": "outputs_FLAC/exp11_VANL",
      "training_seed": 42
    }
  },
  "restarts": {}
}
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
session id: 019ffbe6-3613-79a1-8b57-379a9c6f1c76
--------
user
Round-5 review (read-only; do NOT install anything or modify environments/files; output = review text only). Repo /n/fs/gatrdp/codespace/FLAC, exp_11 chunked-training kit. Cumulative diff + the new watchdog file: /tmp/claude-374349/-n-fs-gatrdp-codespace-FLAC/66305eb9-88c7-41df-95da-dde30a426dc2/scratchpad/chunk_kit.diff (live files in worklog/worklog_yixun/exp_11_fa_orbit_claude/).

MOTIVATION: the cluster never backfills 34-160h 8-GPU allocations; five 40k->100k legs sat PENDING >16h. The kit splits each extension into 2500-step chunk jobs (hours-long walltimes: C4L 3h/C8 4h/C16 6h/C32 9h/VANL 2.5h pins) chained by a login-side watchdog.

DESIGN: (1) fa_orbit_ckpt_preflight.py --chain: same INITIAL-identity binding as --extension but the resume anchor is the TIP of arms.<ARM>.chain (links written only by the recorder); empty chain refuses; optional --chunk-end validated (multiple of 2500, EXPECTED_STEP < N <= budget). (2) fa_orbit_record_restart.py: a manifest carrying chunk_end is recorded as an immutable chain link {job, uuid, manifest sha, resume_step/sha, final_step=chunk_end, final_ckpt_sha256 hashed from the on-disk endpoint ckpt}; idempotent per job; content-drift on re-record refuses; identity checks parameterized (anchor/tip, CHUNK wall pin). (3) fa_orbit_submit.sh --chunk-end: shape checks + CHUNK time pin + exports CHUNK_END. (4) fa_orbit_train.sbatch: STOPSTEP=CHUNK_END feeds train.py --max-steps AND the completion-literal classifier; budget pin unmoved; preflight dispatch: EXPECTED_STEP==40000 -> --extension, >40000 -> --chain; CHUNK pins; manifest chunk_end line. (5) NEW fa_orbit_chunk_watchdog.sh: KV-whitelist args, per-arm: skip if job named exp11-<ARM>-train queued; find newest canonical ckpt step S; S>=TARGET -> done; if S>40000 and registry tip<S record the finished chunk first (unique manifest by chunk_end==S, recorder refusal -> failure); submit next chunk via the sanctioned submitter only; 2 consecutive failures freeze the arm (state file, human-clearable); ONESHOT/DRYRUN test hooks. (6) 22 new guard cases, all passing (113 pass total; 5 pre-existing dirty-tree submitter cases clear on commit).

REVIEW FOCUS: (a) can any sequence of watchdog + recorder + preflight actions launder an unrecorded/foreign checkpoint into the lineage, or double-submit an arm (race between squeue check and submit; job-name collisions with the five OLD long legs still queued as exp11-<ARM>-train — we cancel them before starting the watchdog, is anything else vulnerable)? (b) recorder correctness: link append under concurrent invocation (store lock?), atomic write, the mine/at indexing, chunk_end vs budget/cadence edges; (c) preflight --chain: any bypass via crafted registry/manifest (note the resume file is always re-hashed); (d) launcher: STOPSTEP interactions with the completion literal, argv-parity section G, wandb fresh-id (r4) under many chunks, TIME_PIN selection; (e) watchdog: sacct settle logic (CANCELLED counts as failure — is that right when a human scancels?), leg_manifest_for uniqueness under repeated same-boundary attempts after a failed leg (two manifests with chunk_end==S -> permanent freeze — is the recovery path documented/sound?), state-file semantics; (f) the 2500-step chunk size: startup overhead ~10-15min/leg at 8 GPUs — flag if you consider the overhead-to-compute ratio for any arm unacceptable vs a 5000-step chunk. Verdict: BLOCKING/NON-BLOCKING findings then APPROVED or REVISE.
codex
I’ll audit the diff and live scripts read-only, trace the lineage and concurrency invariants end to end, then report only actionable findings and a verdict.
exec
/usr/bin/bash -lc 'wc -l /tmp/claude-374349/-n-fs-gatrdp-codespace-FLAC/66305eb9-88c7-41df-95da-dde30a426dc2/scratchpad/chunk_kit.diff && rg --files worklog/worklog_yixun/exp_11_fa_orbit_claude | sort && git status --short' in /n/fs/gatrdp/codespace/FLAC
 succeeded in 0ms:
1172 /tmp/claude-374349/-n-fs-gatrdp-codespace-FLAC/66305eb9-88c7-41df-95da-dde30a426dc2/scratchpad/chunk_kit.diff
/usr/bin/bash: line 1: rg: command not found
 M worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py
 M worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_record_restart.py
 M worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submit.sh
 M worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch
 M worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh
?? AGENTS.md
?? AcousticRooms
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-12_18-23-10_guardtests.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-13_12-06-48_guardtests.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-13_12-08-41_guardtests.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-13_12-10-57_guardtests.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_chunk_watchdog.sh
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_chunk_kit_review.md
?? worklog/worklog_yixun/exp_15_yaw_aug_claude/slurm_train_exp15-smoke-YAWAUG_3685989.out
?? worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-12_11-59-14_readback.log
?? worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-12_11-59-42_readback.log
?? worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-12_12-05-35_guardtests_fullfix.log
?? worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-12_12-35-37_guardtests_f3fix.log
?? worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-12_12-40-22_guardtests_f3fix.log
?? worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-12_17-49-31_YAWAUG_8x8_jid3685989_manifest.txt
?? worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_smoke_registry.json

exec
/usr/bin/bash -lc "sed -n '1,420p' /tmp/claude-374349/-n-fs-gatrdp-codespace-FLAC/66305eb9-88c7-41df-95da-dde30a426dc2/scratchpad/chunk_kit.diff" in /n/fs/gatrdp/codespace/FLAC
 succeeded in 0ms:
diff --git a/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py b/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py
index fc72f83..c9b6ed2 100644
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
@@ -196,14 +237,54 @@ def check_extension_binding(manifest_path, registry_path, arm, rung, config_path
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
+def check_chain_binding(manifest_path, registry_path, arm, rung, config_path, ckpt_path,
+                        ckpt_sha, expected_step, max_steps, repo_root="."):
+    """The CHUNKED-LEG contract: resume the TIP of this arm's recorded chain.
+
+    A chunk leg is an extension leg that stops early, so it must prove exactly
+    the same INITIAL launch identity — but its resume point is no longer the
+    audited 40k anchor: it is the endpoint of the previous chunk, recorded as the
+    last link of ``arms.<ARM>.chain`` by fa_orbit_record_restart.py.
+
+    Fail-closed by construction: an EMPTY or ABSENT chain refuses, so chunk N+1
+    cannot run until chunk N has been recorded, and every link's final_step /
+    final_ckpt_sha256 was written by the recorder from the file on disk. The
+    resume file is re-hashed by the caller (``ckpt_sha``); a manifest's claimed
+    hash is never trusted here or anywhere else in this chain.
+    """
+    reg, problems = _load_registry_row(registry_path, arm)
+    if problems:
+        return problems, {}
+    man = parse_manifest(manifest_path)
+    problems = check_initial_identity(man, manifest_path, reg, arm, rung, config_path, max_steps)
+    if not reg.get("final_ckpt_sha256"):
+        problems.append(f"{arm} has no audited final_ckpt_sha256 in the registry — a chain of "
+                        "chunks must still descend from the audited INITIAL run")
+    chain = reg.get("chain") or []
+    if not chain:
+        problems.append(f"{arm} has no recorded chain link in {registry_path}: a chunk may only "
+                        "resume the tip of a RECORDED chain, so chunk N+1 is inadmissible until "
+                        "chunk N is recorded — run fa_orbit_record_restart.py on the previous "
+                        "chunk's launcher manifest first")
     else:
-        canon = canonical_ckpt_dir(save_dir, arm, repo_root)
-        if os.path.realpath(os.path.dirname(ckpt_path)) != canon:
-            problems.append(f"resume checkpoint {ckpt_path} does not live in the registered "
-                            f"launch's canonical run directory {canon}")
+        tip = chain[-1]
+        tip_step, tip_sha = tip.get("final_step"), tip.get("final_ckpt_sha256")
+        if str(tip_step) != str(expected_step):
+            problems.append(f"EXPECTED_STEP {expected_step} != the last recorded chain link's "
+                            f"final_step {tip_step!r} (link job {tip.get('job')!r}) — a chunk "
+                            "resumes the TIP of the chain, never an earlier link")
+        if not tip_sha:
+            problems.append(f"the last recorded chain link (job {tip.get('job')!r}) carries no "
+                            "final_ckpt_sha256 — it is not evidence of a checkpoint")
+        elif ckpt_sha != tip_sha:
+            problems.append(f"resume checkpoint sha256 {ckpt_sha[:12]} != the last recorded chain "
+                            f"link's final_ckpt_sha256 {str(tip_sha)[:12]} — this file is not the "
+                            "checkpoint that chunk produced")
+    problems += check_canonical_dir(man, arm, ckpt_path, repo_root)
     return problems, man
 
 
@@ -221,13 +302,25 @@ def main(argv=None):
     ap.add_argument("--extension", action="store_true",
                     help="this restart is the Q10 40k->100k EXTENSION: bind it to the audited "
                          "INITIAL launch identity instead of requiring an equal budget/commit")
+    ap.add_argument("--chain", action="store_true",
+                    help="this restart is a CHUNK continuing an already-recorded chain: same "
+                         "INITIAL identity as --extension, but the resume anchor is the LAST "
+                         "recorded arms.<ARM>.chain link instead of the audited 40k checkpoint")
+    ap.add_argument("--chunk-end", type=int, default=None,
+                    help="this leg's stop step (a chunk boundary): EXPECTED_STEP < N <= max-steps "
+                         "and a multiple of 2500")
     ap.add_argument("--launch-registry", default="",
-                    help="the committed arm launch registry (required with --extension)")
+                    help="the committed arm launch registry (required with --extension/--chain)")
     ap.add_argument("--repo-root", default=".",
                     help="root the registry's relative save_dir is resolved against")
     args = ap.parse_args(argv)
+    if args.extension and args.chain:
+        ap.error("--extension and --chain are mutually exclusive: a leg either resumes the audited "
+                 "40k anchor or the tip of the recorded chunk chain, never both")
     if args.extension and not args.launch_registry:
         ap.error("--extension requires --launch-registry (the audited INITIAL launch row)")
+    if args.chain and not args.launch_registry:
+        ap.error("--chain requires --launch-registry (the recorded chunk chain lives in it)")
 
     if not os.path.isfile(args.ckpt):
         print(f"PREFLIGHT: checkpoint not found: {args.ckpt}")
@@ -277,7 +370,18 @@ def main(argv=None):
     if not n_ema:
         problems.append("no EMA weights in state_dict")
 
-    digest = sha256_file(args.ckpt)     # needed by the extension contract's anchor check
+    # The per-job stop step is checked against the BUDGET (--max-steps), which the
+    # launcher still pins at 100000: a chunk narrows where this leg stops, never

codex
Two operational blockers have emerged: recording a link dirties the tracked registry that the submitter requires to be clean, and retry attempts at the same boundary make manifest selection permanently ambiguous. I’m validating scheduler parsing and checkpoint/registry trust boundaries before assigning final severities.
exec
/usr/bin/bash -lc 'sacct -X -n -P -j 3648695 -o JobIDRaw,State 2>/dev/null | head -5
sacct -X -n -P -j 3648695 -o State 2>/dev/null | head -5
squeue -h -u "$USER" -n exp11-C8-train -o '"'%i %T' 2>/dev/null | head -10" in /n/fs/gatrdp/codespace/FLAC
 succeeded in 0ms:
3648695|COMPLETED
COMPLETED
3687570 PENDING

2026-08-13T16:15:12.925557Z ERROR codex_models_manager::manager: failed to renew cache TTL: missing field `base_instructions` at line 94 column 5
exec
/usr/bin/bash -lc "nl -ba worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_classify.py | sed -n '1,280p'
grep -R \"max_steps\" -n train.py src/training | head -60
grep -R \"save_top_k\\|ModelCheckpoint\\|checkpoint_callback\" -n train.py src | head -80
grep -n \"TIME_LIMIT_CHUNK\\|CHECKPOINT_EVERY\\|MAXSTEPS\" worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch | head -40" in /n/fs/gatrdp/codespace/FLAC
 succeeded in 2922ms:
train.py:46:    the only behavioral change is that max_steps is now sourced from args.max_steps
train.py:78:        "max_steps": args.max_steps, # HAA finetune recipe: --max-steps 1000
grep: src/tests/__pycache__/test_strip_optimizer_state.cpython-310-pytest-9.1.1.pyc: binary file matches
grep: src/tests/__pycache__/test_retune_lr_state.cpython-310-pytest-9.1.1.pyc: binary file matches
grep: src/tests/__pycache__/test_finetune_cond.cpython-310-pytest-9.1.1.pyc: binary file matches
train.py:182:    ckpt_callback = pl.callbacks.ModelCheckpoint(every_n_train_steps=args.checkpoint_every, dirpath=checkpoint_dir, save_top_k=-1)
src/tests/test_retune_lr_state.py:110:        "callbacks": {"ModelCheckpoint{'monitor': None}": {"best_model_score": None}},
src/tests/test_strip_optimizer_state.py:57:        "callbacks": {"ModelCheckpoint{'monitor': None}": {"best_model_score": None}},
src/tests/test_finetune_cond.py:315:    Lightning cannot inject its default ModelCheckpoint) AND no ModelCheckpoint in
src/tests/test_finetune_cond.py:322:    assert not any(isinstance(cb, pl.callbacks.ModelCheckpoint) for cb in kw["callbacks"])
src/tests/test_finetune_cond.py:331:    assert any(isinstance(cb, pl.callbacks.ModelCheckpoint) for cb in kw["callbacks"])
32:# RUNG / MAXSTEPS / MIN_FREE_MB / time limit are NOT operator inputs any more.
54:# SMOKE_MAXSTEPS (small), SMOKE_MIN_FREE_MB, its own identity
81:PINNED_MAXSTEPS=100000                     # Q10: extended budget (was 40000, the
86:PINNED_CHECKPOINT_EVERY=2500               # exp_07 cadence
109:PINNED_TIME_LIMIT_CHUNK_C4L="03:00:00"
110:PINNED_TIME_LIMIT_CHUNK_C8="04:00:00"
111:PINNED_TIME_LIMIT_CHUNK_C16="06:00:00"
112:PINNED_TIME_LIMIT_CHUNK_C32="09:00:00"
113:PINNED_TIME_LIMIT_CHUNK_VANL="02:30:00"
158:# PINNED_MAXSTEPS remains the campaign budget and its gate is untouched.
175:  RUNG="${SMOKE_RUNG:-}"; MAXSTEPS="${SMOKE_MAXSTEPS:-30}"; MIN_FREE_MB="${SMOKE_MIN_FREE_MB:-}"
176:  CHECKPOINT_EVERY="${SMOKE_CHECKPOINT_EVERY:-10}"
194:    TIME_PIN_NAME="PINNED_TIME_LIMIT_CHUNK_${ARM}"
205:  RUNG="$PINNED_RUNG"; MAXSTEPS="$PINNED_MAXSTEPS"; MIN_FREE_MB="$PINNED_MIN_FREE_MB"
206:  CHECKPOINT_EVERY="$PINNED_CHECKPOINT_EVERY"
219:  [ "$MAXSTEPS" = "100000" ] || die "PINNED_MAXSTEPS is ${MAXSTEPS}, the registered budget is 100000 - abort"
224:# PINNED_MAXSTEPS stays the campaign budget. STOPSTEP is what train.py is given.
227:STOPSTEP="$MAXSTEPS"
229:  [ "$SMOKE" != "1" ] || die "CHUNK_END is a production chunk-chain input and has no meaning under SMOKE=1 (SMOKE_MAXSTEPS is the smoke's budget) - abort"
233:  [ "$CHUNK_END" -le "$MAXSTEPS" ] || die "CHUNK_END ${CHUNK_END} exceeds the budget ${MAXSTEPS} - abort"
234:  [ "$((CHUNK_END % PINNED_CHECKPOINT_EVERY))" -eq 0 ] || die "CHUNK_END ${CHUNK_END} is not a multiple of the ${PINNED_CHECKPOINT_EVERY}-step checkpoint cadence: the chunk must end ON a checkpoint or the next chunk has nothing to resume - abort"
387:  [ "$MAXSTEPS" -gt "$EXPECTED_STEP" ] || die "MAXSTEPS ${MAXSTEPS} must exceed the resume step ${EXPECTED_STEP} - abort"
389:echo "lineage: ${MODE} (expected_step ${EXPECTED_STEP}, max_steps ${MAXSTEPS}, chunk_end ${CHUNK_END:-<none>}, stops at ${STOPSTEP}, ckpt every ${CHECKPOINT_EVERY}, time pin ${TIME_PIN_NAME}=${TIME_LIMIT})"
401:  --logger wandb --checkpoint-every "$CHECKPOINT_EVERY"
567:            --max-steps "$MAXSTEPS" --arm "$ARM" --rung "$RUNG" --commit "$HEAD_SHA")
685:  echo "arm ${ARM} rung ${RUNG} micro ${MB} ngpu ${NGPU} max_steps ${MAXSTEPS} ckpt_every ${CHECKPOINT_EVERY}"
766:# is MAXSTEPS, byte-identical to every pre-round-5 leg; with one it is the chunk
785:  [ -n "$CHUNK_END" ] && echo "next chunk: record this leg (fa_orbit_record_restart.py ${ARM} ${MANIFEST}), then fa_orbit_submit.sh ${ARM} --resume <ckpt at ${CHUNK_END}> --expected-step ${CHUNK_END} --chunk-end <${CHUNK_END}+${PINNED_CHECKPOINT_EVERY}>"

exec
/usr/bin/bash -lc "grep -R \"chain\\|verify_chain\\|producer\" -n worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen.sbatch worklog/worklog_yixun/exp_11_fa_orbit_claude/*.py | head -160
grep -R \"arm_launch_registry\" -n worklog/worklog_yixun/exp_11_fa_orbit_claude | head -100" in /n/fs/gatrdp/codespace/FLAC
exec
/usr/bin/bash -lc "nl -ba worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen.sbatch | sed -n '455,525p'
git diff -- worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen.sbatch" in /n/fs/gatrdp/codespace/FLAC
 succeeded in 0ms:
   455	for label, got_v, want_v in (("job", jkv.get("job"), reg["job"]),
   456	                             ("launch mode", jkv.get("mode"), reg["mode"]),
   457	                             ("launch_uuid", jkv.get("launch_uuid"), reg["launch_uuid"]),
   458	                             ("commit", man.get("commit"), reg["commit"]),
   459	                             ("rung", kv.get("rung"), reg["rung"]),
   460	                             ("max_steps", kv.get("max_steps"), reg["max_steps"]),
   461	                             ("config_sha256", man.get("config_sha256"), reg["config_sha256"]),
   462	                             ("vae_sha256", man.get("vae_sha256"), reg["vae_sha256"]),
   463	                             ("p0_manifest_sha256", man.get("p0_manifest_sha256"), reg["p0_manifest_sha256"]),
   464	                             ("save_dir", man.get("save_dir"), reg["save_dir"])):
   465	    if got_v != want_v:
   466	        bad.append(f"{label} {got_v!r} != registered {want_v!r}")
   467	if jkv.get("mode") != "INITIAL":
   468	    bad.append(f"launch mode {jkv.get('mode')!r} is not INITIAL (a restart is not a registered launch)")
   469	if int(reg.get("training_seed", -1)) != 42:
   470	    bad.append(f"registered training seed {reg.get('training_seed')!r} != 42")
   471	if kv.get("arm") != arm:
   472	    bad.append(f"manifest arm {kv.get('arm')!r} != {arm!r}")
   473	if man.get("config_sha256") != cfg_sha:
   474	    bad.append(f"manifest config_sha256 {man.get('config_sha256', '')[:12]} != the config we would "
   475	               f"evaluate {cfg_sha[:12]}")
   476	save_dir = man.get("save_dir", "")
   477	if not save_dir:
   478	    bad.append("manifest records no save_dir")
   479	else:
   480	    want_dir = os.path.realpath(ckpt_dir)
   481	    # save_dir is recorded relative to the MAIN tree; under worktree-pinned
   482	    # execution the cwd is the pinned code root, so resolve it explicitly.
   483	    save_abs = save_dir if os.path.isabs(save_dir) else os.path.join(main_repo, save_dir)
   484	    canon = os.path.realpath(os.path.join(save_abs, f"FLAC_exp11_{arm}", f"exp11_{arm}", "checkpoints"))
   485	    if want_dir != canon:
   486	        bad.append(f"checkpoint dir {want_dir} is not the manifest's canonical {canon}")
   487	    if os.path.realpath(os.path.dirname(ckpt)) != canon:
   488	        bad.append(f"checkpoint {ckpt} does not live in the canonical run directory {canon}")
   489	# --- Q10: a checkpoint ABOVE 40k came from a RESTART leg, not the INITIAL run --
   490	# The INITIAL manifest cannot vouch for it. The first version of this gate was
   491	# EXISTENTIAL (re-pin review, finding 2): once ONE leg carried mode=RESTART and
   492	# the right resume hash, every later same-config checkpoint in this directory
   493	# passed, because nothing bound the evaluated FILE to the leg that produced it —
   494	# a checkpoint from a wrong restart, copied in, was admissible. Now the binding
   495	# is per checkpoint: this file is re-hashed and must appear, at exactly this
   496	# step, in the append-only PRODUCER MANIFEST of a leg whose every registry field
   497	# re-validates against the audited INITIAL row (see fa_orbit_producer_manifest.py).
   498	step_i = int(os.path.basename(ckpt).split("step=")[1].split(".ckpt")[0])
   499	if step_i > 40000:
   500	    sys.path.insert(0, expdir)
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
   516	      f"max_steps {reg['max_steps']}, training seed {reg['training_seed']}); manifest "
   517	      f"{got_sha[:12]} as registered; config {cfg_sha[:12]}")
   518	PY
   519	fi
   520	
   521	# --- E. the checkpoint's OWN embedded config must be this arm's ---------------
   522	# NOTE: TRAIN_ANGLES, not N_ANGLES — a cross cell evaluates a different orbit,
   523	# but the checkpoint must still BE this arm, trained on this arm's orbit.
   524	python3 - "$CKPT" "$MODEL_CONFIG" "$TRAIN_ANGLES" "$STEP" "$EXPDIR" "$COND_METHOD" <<'PY' || die "checkpoint/arm identity gate FAILED - abort"
   525	import json, sys

 succeeded in 82180ms:
grep: worklog/worklog_yixun/exp_11_fa_orbit_claude/__pycache__/fa_orbit_record_restart.cpython-310.pyc: binary file matches
grep: worklog/worklog_yixun/exp_11_fa_orbit_claude/__pycache__/fa_orbit_producer_manifest.cpython-310.pyc: binary file matches
grep: worklog/worklog_yixun/exp_11_fa_orbit_claude/__pycache__/fa_orbit_add_anchor.cpython-310.pyc: binary file matches
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:585:    PRE_ARGS+=(--launch-registry "$EXPDIR/arm_launch_registry.json" --repo-root "$REPO")
worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-VANL-screen-7500-s42-K8_3662403.out:10:ARM LINEAGE GATE: VANL is not in the audited launch registry /n/fs/gatrdp/codespace/FLAC/.measure_worktrees/0c6e9ffb616cbd788b420e67d62638ad40a7b13c/worklog/worklog_yixun/exp_11_fa_orbit_claude/arm_launch_registry.json
worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-VANL-screen-2500-s42-K8_3662805.out:10:ARM LINEAGE GATE: VANL is not in the audited launch registry /n/fs/gatrdp/codespace/FLAC/.measure_worktrees/0c6e9ffb616cbd788b420e67d62638ad40a7b13c/worklog/worklog_yixun/exp_11_fa_orbit_claude/arm_launch_registry.json
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_gocheck2.md:17:| **Immutable launch binding** | **CLOSED** | The committed registry pins each live arm’s complete manifest digest and launch identity ([arm_launch_registry.json:13](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/arm_launch_registry.json:13)). The screen checks the digest plus job, `INITIAL` mode, UUID, commit, rung, limits, config/VAE/P0 hashes, save directory, and registered seed ([fa_orbit_screen.sbatch:302](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen.sbatch:302)). The recorded manifests currently match those digests, including running C4L job `3648694`. The override seam is acceptably gated: Slurm forces the production output root, and a registry override is refused for that root ([fa_orbit_screen.sbatch:111](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen.sbatch:111), [fa_orbit_screen.sbatch:275](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen.sbatch:275)). |
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:124:with open(os.path.join(out, "arm_launch_registry.json"), "w") as fh:
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:130:      "FA_ORBIT_ARM_REGISTRY=${OUT_ROOT}/arm_launch_registry.json")
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:133:  $PY - "$1" "${OUT_ROOT}/exp11_$1/launch_manifest.txt" "${OUT_ROOT}/arm_launch_registry.json" <<'PY'
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:322:$PY - "${OUT_ROOT}/arm_launch_registry.json" "${OUT_ROOT}/exp11_C8" <<'PY'
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:358:r = json.load(open(os.path.join(out, "arm_launch_registry.json")))
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:360:json.dump(r, open(os.path.join(out, "arm_launch_registry.json"), "w"), indent=2)
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:364:    --registry "${OUT_ROOT}/arm_launch_registry.json" \
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:394:tamper_leg() { $PY - "${OUT_ROOT}/arm_launch_registry.json" "$1" "$2" <<'PY'
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:401:cp "${OUT_ROOT}/arm_launch_registry.json" "${TMP}/reg_recorded.json"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:405:cp "${TMP}/reg_recorded.json" "${OUT_ROOT}/arm_launch_registry.json"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:409:cp "${TMP}/reg_recorded.json" "${OUT_ROOT}/arm_launch_registry.json"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:1047:r=json.load(open('${EXPDIR}/arm_launch_registry.json'))['arms']
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:1432:REG="${EXPDIR}/arm_launch_registry.json"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:1467:  $PY - "$MAN" "${OUT_ROOT}/arm_launch_registry.json" <<'PY'
worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-VANL-screen-5000-s42-K1_3662402.out:10:ARM LINEAGE GATE: VANL is not in the audited launch registry /n/fs/gatrdp/codespace/FLAC/.measure_worktrees/0c6e9ffb616cbd788b420e67d62638ad40a7b13c/worklog/worklog_yixun/exp_11_fa_orbit_claude/arm_launch_registry.json
worklog/worklog_yixun/exp_11_fa_orbit_claude/commits_fa_orbit.md:50:| 41 | `b23fccd` | GO-check NO-GO closure. **ASSETS (the crasher):** `fa_orbit_measure_worktree.sh` links `AcousticRooms` and `weights` into every pinned worktree at the main tree's own targets — a fresh worktree holds only TRACKED files, so the eval's relative `AcousticRooms` / `weights/AGREE/AGREE_fullAR.pt` did not exist and a pinned screen would have died at startup; the screen now stats each required asset before spending a slot. **LIFECYCLE:** fixed-count pruning (which could delete a QUEUED job's tree) is replaced by LEASES — `.leases/<jobid>` written under an `flock` before `sbatch` returns and promoted to the real id, validated by the job at start, released by a trap at exit; a tree is prunable only with zero live leases (stale ids double-checked against `squeue`) and never the tree being handed out. New `fa_orbit_screen_submit.sh` makes pin+lease+submit atomic. **IMMUTABLE LAUNCH BINDING:** the launch manifest lives under gitignored outputs and is mutable, so it is now bound to a committed `arm_launch_registry.json` by sha256 plus job, mode INITIAL, launch uuid, commit, rung, max_steps, config/VAE/P0 hashes, save-dir and seed 42 — the seed used to be printed, not checked. **TABLE:** the two-K gate is transactional (aborts without writing; `--allow-partial-exp11` still renders WITHHELD, never numbers), `build_header()` is actually called by `main()` so the LABEL-MIGRATION-DEFERRED note reaches the file, and `--repo-root` resolves evidence/validator/output against the MAIN tree from any cwd. Also: MEASURE_ROOT must be a detached, managed worktree of THIS repo under `.measure_worktrees/`, and the sidecar records a repo-relative model-config path. |
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_chunk_watchdog.sh:34:#   OUTPUT_ROOT=outputs_FLAC  REGISTRY=<expdir>/arm_launch_registry.json
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_chunk_watchdog.sh:53:REGISTRY="${EXPDIR}/arm_launch_registry.json"
worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-VANL-screen-2500-s42-K8_3662399.out:10:ARM LINEAGE GATE: VANL is not in the audited launch registry /n/fs/gatrdp/codespace/FLAC/.measure_worktrees/0c6e9ffb616cbd788b420e67d62638ad40a7b13c/worklog/worklog_yixun/exp_11_fa_orbit_claude/arm_launch_registry.json
worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-VANL-screen-7500-s42-K8_3662809.out:10:ARM LINEAGE GATE: VANL is not in the audited launch registry /n/fs/gatrdp/codespace/FLAC/.measure_worktrees/0c6e9ffb616cbd788b420e67d62638ad40a7b13c/worklog/worklog_yixun/exp_11_fa_orbit_claude/arm_launch_registry.json
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_chunk_kit_review.md:765:-  [ "$SMOKE" != "1" ] && PRE_ARGS+=(--extension --launch-registry "$EXPDIR/arm_launch_registry.json"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_chunk_kit_review.md:778:+    PRE_ARGS+=(--launch-registry "$EXPDIR/arm_launch_registry.json" --repo-root "$REPO")
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_chunk_kit_review.md:979:#   OUTPUT_ROOT=outputs_FLAC  REGISTRY=<expdir>/arm_launch_registry.json
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_chunk_kit_review.md:998:REGISTRY="${EXPDIR}/arm_launch_registry.json"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_chunk_kit_review.md:1242:     2	"""Record a RESTART leg in arm_launch_registry.json from its PUBLISHED manifest.
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_chunk_kit_review.md:1556:   316	    ap.add_argument("--registry", default=os.path.join(HERE, "arm_launch_registry.json"))
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_chunk_kit_review.md:1694:    23	directory, exactly like arm_launch_registry.json, and screens read it from the
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_chunk_kit_review.md:1984:    34	#   OUTPUT_ROOT=outputs_FLAC  REGISTRY=<expdir>/arm_launch_registry.json
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_chunk_kit_review.md:2003:    53	REGISTRY="${EXPDIR}/arm_launch_registry.json"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_chunk_kit_review.md:2249:   316	    ap.add_argument("--registry", default=os.path.join(HERE, "arm_launch_registry.json"))
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_chunk_kit_review.md:2388:    23	directory, exactly like arm_launch_registry.json, and screens read it from the
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_chunk_kit_review.md:3178:   585	    PRE_ARGS+=(--launch-registry "$EXPDIR/arm_launch_registry.json" --repo-root "$REPO")
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_chunk_kit_review.md:3975:/usr/bin/bash -lc 'git ls-files --error-unmatch worklog/worklog_yixun/exp_11_fa_orbit_claude/arm_launch_registry.json >/dev/null 2>&1; echo registry_tracked_rc=$?
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_chunk_kit_review.md:3976:git status --short --untracked-files=no -- worklog/worklog_yixun/exp_11_fa_orbit_claude/arm_launch_registry.json
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_chunk_kit_review.md:3977:sed -n '"'1,260p' worklog/worklog_yixun/exp_11_fa_orbit_claude/arm_launch_registry.json
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_chunk_kit_review.md:4428:grep -R \"arm_launch_registry\" -n worklog/worklog_yixun/exp_11_fa_orbit_claude | head -100" in /n/fs/gatrdp/codespace/FLAC
worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-VANL-screen-7500-s42-K1_3662404.out:10:ARM LINEAGE GATE: VANL is not in the audited launch registry /n/fs/gatrdp/codespace/FLAC/.measure_worktrees/0c6e9ffb616cbd788b420e67d62638ad40a7b13c/worklog/worklog_yixun/exp_11_fa_orbit_claude/arm_launch_registry.json
worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-VANL-screen-5000-s42-K1_3662808.out:10:ARM LINEAGE GATE: VANL is not in the audited launch registry /n/fs/gatrdp/codespace/FLAC/.measure_worktrees/0c6e9ffb616cbd788b420e67d62638ad40a7b13c/worklog/worklog_yixun/exp_11_fa_orbit_claude/arm_launch_registry.json
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_add_anchor.py:2:"""Record an arm's AUDITED final checkpoint (the anchor) in arm_launch_registry.json.
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_add_anchor.py:130:    ap.add_argument("--registry", default=os.path.join(HERE, "arm_launch_registry.json"))
worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-VANL-screen-2500-s42-K1_3662806.out:10:ARM LINEAGE GATE: VANL is not in the audited launch registry /n/fs/gatrdp/codespace/FLAC/.measure_worktrees/0c6e9ffb616cbd788b420e67d62638ad40a7b13c/worklog/worklog_yixun/exp_11_fa_orbit_claude/arm_launch_registry.json
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:208:worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:533:  [ "$SMOKE" != "1" ] && PRE_ARGS+=(--extension --launch-registry "$EXPDIR/arm_launch_registry.json"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:297:worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:124:with open(os.path.join(out, "arm_launch_registry.json"), "w") as fh:
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:298:worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:130:      "FA_ORBIT_ARM_REGISTRY=${OUT_ROOT}/arm_launch_registry.json")
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:300:worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:133:  $PY - "$1" "${OUT_ROOT}/exp11_$1/launch_manifest.txt" "${OUT_ROOT}/arm_launch_registry.json" <<'PY'
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:315:worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:322:$PY - "${OUT_ROOT}/arm_launch_registry.json" "${OUT_ROOT}/exp11_C8" <<'PY'
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:324:worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:358:r = json.load(open(os.path.join(out, "arm_launch_registry.json")))
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:325:worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:360:json.dump(r, open(os.path.join(out, "arm_launch_registry.json"), "w"), indent=2)
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:327:worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:364:    --registry "${OUT_ROOT}/arm_launch_registry.json" \
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:331:worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:394:tamper_leg() { $PY - "${OUT_ROOT}/arm_launch_registry.json" "$1" "$2" <<'PY'
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:332:worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:401:cp "${OUT_ROOT}/arm_launch_registry.json" "${TMP}/reg_recorded.json"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:334:worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:405:cp "${TMP}/reg_recorded.json" "${OUT_ROOT}/arm_launch_registry.json"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:336:worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:409:cp "${TMP}/reg_recorded.json" "${OUT_ROOT}/arm_launch_registry.json"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:350:worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:1047:r=json.load(open('${EXPDIR}/arm_launch_registry.json'))['arms']
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:357:worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:1432:REG="${EXPDIR}/arm_launch_registry.json"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:372:worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:1467:  $PY - "$MAN" "${OUT_ROOT}/arm_launch_registry.json" <<'PY'
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:391:worklog/worklog_yixun/exp_11_fa_orbit_claude/arm_launch_registry.json:3:    "AUDITED exp_11 arm launch registry (final GO-check item 4 / review (b)).",
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:392:worklog/worklog_yixun/exp_11_fa_orbit_claude/arm_launch_registry.json:4:    "The launch manifests live under gitignored outputs_FLAC and are therefore",
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:393:worklog/worklog_yixun/exp_11_fa_orbit_claude/arm_launch_registry.json:5:    "MUTABLE evidence: binding a screen to 'whatever the manifest says now' proves",
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:394:worklog/worklog_yixun/exp_11_fa_orbit_claude/arm_launch_registry.json:6:    "nothing. This committed registry pins each arm's manifest by sha256 plus the",
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:395:worklog/worklog_yixun/exp_11_fa_orbit_claude/arm_launch_registry.json:9:    "VANL (Q9, job 3661520) recorded from its PUBLISHED launch manifest after the",
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:396:worklog/worklog_yixun/exp_11_fa_orbit_claude/arm_launch_registry.json:11:    "Recorded from the live manifests of the running arms (3648694-97, 3661520).",
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:397:worklog/worklog_yixun/exp_11_fa_orbit_claude/arm_launch_registry.json:12:    "RESTART legs (Q10, 40k -> 100k) are recorded under 'restarts' as a CHAIN:",
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:398:worklog/worklog_yixun/exp_11_fa_orbit_claude/arm_launch_registry.json:17:    "Populate with fa_orbit_record_restart.py once a leg's manifest publishes.",
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:399:worklog/worklog_yixun/exp_11_fa_orbit_claude/arm_launch_registry.json:20:    "'producer_manifest' -- fa_orbit_producer_<ARM>_job<JOB>.json, append-only,",
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:400:worklog/worklog_yixun/exp_11_fa_orbit_claude/arm_launch_registry.json:30:      "manifest_path": "outputs_FLAC/exp11_C4L/launch_manifest.txt",
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:401:worklog/worklog_yixun/exp_11_fa_orbit_claude/arm_launch_registry.json:31:      "manifest_sha256": "d49df42d2f7f9c3f39f1aeb6631da84ef0e0a392c22a8271edadbd83885e814a",
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:402:worklog/worklog_yixun/exp_11_fa_orbit_claude/arm_launch_registry.json:42:      "p0_manifest_sha256": "72607b922177208d56055d604b292d697b643ef3b7ab48261ab2e23a0cc2b53b",
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:403:worklog/worklog_yixun/exp_11_fa_orbit_claude/arm_launch_registry.json:49:      "manifest_path": "outputs_FLAC/exp11_C8/launch_manifest.txt",
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:404:worklog/worklog_yixun/exp_11_fa_orbit_claude/arm_launch_registry.json:50:      "manifest_sha256": "fa1037c300fa3f1100667634864653690049271bd4e2815e419fb205c9068388",
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:405:worklog/worklog_yixun/exp_11_fa_orbit_claude/arm_launch_registry.json:61:      "p0_manifest_sha256": "72607b922177208d56055d604b292d697b643ef3b7ab48261ab2e23a0cc2b53b",
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:406:worklog/worklog_yixun/exp_11_fa_orbit_claude/arm_launch_registry.json:68:      "manifest_path": "outputs_FLAC/exp11_C16/launch_manifest.txt",
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:407:worklog/worklog_yixun/exp_11_fa_orbit_claude/arm_launch_registry.json:69:      "manifest_sha256": "deb07b532fea037d9354b5c635e9ad6a80ad8c022dabdc6dbe0a879a27be3979",
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:408:worklog/worklog_yixun/exp_11_fa_orbit_claude/arm_launch_registry.json:80:      "p0_manifest_sha256": "72607b922177208d56055d604b292d697b643ef3b7ab48261ab2e23a0cc2b53b",
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:409:worklog/worklog_yixun/exp_11_fa_orbit_claude/arm_launch_registry.json:87:      "manifest_path": "outputs_FLAC/exp11_C32/launch_manifest.txt",
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:410:worklog/worklog_yixun/exp_11_fa_orbit_claude/arm_launch_registry.json:88:      "manifest_sha256": "b2d08bc0f27583bd78845e281380906b7f05a737444525f4e32cafa5106b395e",
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:411:worklog/worklog_yixun/exp_11_fa_orbit_claude/arm_launch_registry.json:99:      "p0_manifest_sha256": "72607b922177208d56055d604b292d697b643ef3b7ab48261ab2e23a0cc2b53b",
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:412:worklog/worklog_yixun/exp_11_fa_orbit_claude/arm_launch_registry.json:108:      "manifest_path": "outputs_FLAC/exp11_VANL/launch_manifest.txt",
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:413:worklog/worklog_yixun/exp_11_fa_orbit_claude/arm_launch_registry.json:109:      "manifest_sha256": "113d06a284c6198cf9487e99a2efb7ccde94ae13e656a403fe2af0281d3de8b1",
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:414:worklog/worklog_yixun/exp_11_fa_orbit_claude/arm_launch_registry.json:120:      "p0_manifest_sha256": "72607b922177208d56055d604b292d697b643ef3b7ab48261ab2e23a0cc2b53b",
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:534:worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_add_anchor.py:2:"""Record an arm's AUDITED final checkpoint (the anchor) in arm_launch_registry.json.
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:556:worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_add_anchor.py:130:    ap.add_argument("--registry", default=os.path.join(HERE, "arm_launch_registry.json"))
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:577:worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_record_restart.py:2:"""Record a RESTART leg in arm_launch_registry.json from its PUBLISHED manifest.
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:607:worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_record_restart.py:155:    ap.add_argument("--registry", default=os.path.join(HERE, "arm_launch_registry.json"))
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:644:worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_producer_manifest.py:23:directory, exactly like arm_launch_registry.json, and screens read it from the
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:762:worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen.sbatch:422:  ARM_REGISTRY="$EXPDIR/arm_launch_registry.json"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:1210:   533	  [ "$SMOKE" != "1" ] && PRE_ARGS+=(--extension --launch-registry "$EXPDIR/arm_launch_registry.json"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:1748:     2	"""Record a RESTART leg in arm_launch_registry.json from its PUBLISHED manifest.
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:1901:   155	    ap.add_argument("--registry", default=os.path.join(HERE, "arm_launch_registry.json"))
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:2014:/usr/bin/bash -lc "python3 -m json.tool worklog/worklog_yixun/exp_11_fa_orbit_claude/arm_launch_registry.json | sed -n '1,260p'; nl -ba worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py | sed -n '60,225p;270,330p'; nl -ba worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_producer_manifest.py | sed -n '160,280p'" in /n/fs/gatrdp/codespace/FLAC
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:3801:worklog/worklog_yixun/exp_11_fa_orbit_claude/arm_launch_registry.json:12:    "RESTART legs (Q10, 40k -> 100k) are recorded under 'restarts' as a CHAIN:",
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:3802:worklog/worklog_yixun/exp_11_fa_orbit_claude/arm_launch_registry.json:30:      "manifest_path": "outputs_FLAC/exp11_C4L/launch_manifest.txt",
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:3803:worklog/worklog_yixun/exp_11_fa_orbit_claude/arm_launch_registry.json:49:      "manifest_path": "outputs_FLAC/exp11_C8/launch_manifest.txt",
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:3804:worklog/worklog_yixun/exp_11_fa_orbit_claude/arm_launch_registry.json:68:      "manifest_path": "outputs_FLAC/exp11_C16/launch_manifest.txt",
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:3805:worklog/worklog_yixun/exp_11_fa_orbit_claude/arm_launch_registry.json:87:      "manifest_path": "outputs_FLAC/exp11_C32/launch_manifest.txt",
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:3806:worklog/worklog_yixun/exp_11_fa_orbit_claude/arm_launch_registry.json:108:      "manifest_path": "outputs_FLAC/exp11_VANL/launch_manifest.txt",

exec
/usr/bin/bash -lc "find outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints -maxdepth 1 -type f -name '*-step=40000.ckpt' -printf '%f %s bytes\\n' 2>/dev/null
find outputs_FLAC -path '*exp11_VANL*' -name '*step=40000.ckpt' -type f -printf '%p %s bytes\\n' 2>/dev/null
squeue -h -u \""'$USER" -n exp11-VANL-train -o '"'%i %T %l' 2>/dev/null" in /n/fs/gatrdp/codespace/FLAC
 succeeded in 0ms:
epoch=8-step=40000.ckpt 723922539 bytes
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=8-step=40000.ckpt 723922539 bytes
3687573 PENDING 19:00:00

2026-08-13T16:18:43.284829Z ERROR codex_models_manager::manager: failed to renew cache TTL: missing field `base_instructions` at line 94 column 5
exec
/usr/bin/bash -lc "nl -ba worklog/worklog_yixun/exp_11_fa_orbit_claude/arm_launch_registry.json | sed -n '110,175p'
nl -ba worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_chunk_watchdog.sh | sed -n '55,290p'" in /n/fs/gatrdp/codespace/FLAC
 succeeded in 0ms:
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
   122	      "training_seed": 42
   123	    }
   124	  },
   125	  "restarts": {}
   126	}
    55	LOG="${EXPDIR}/fa_orbit_chunk_watchdog.log"
    56	
    57	# --- argument parsing: whitelist the KEY, shape-check the VALUE, never eval ---
    58	reject()   { echo "$1" >&2; exit 2; }
    59	is_num()   { case "${1:-}" in ''|*[!0-9]*) return 1 ;; esac; }
    60	is_armset() {
    61	  local v="$1" a
    62	  [ -n "$v" ] || return 1
    63	  local IFS=,
    64	  for a in $v; do
    65	    case "$a" in C4L|C8|C16|C32|VANL) ;; *) return 1 ;; esac
    66	  done
    67	}
    68	for kv in "$@"; do
    69	  case "$kv" in *=*) ;; *) reject "argument '${kv}' is not KEY=VALUE" ;; esac
    70	  key="${kv%%=*}"; val="${kv#*=}"
    71	  case "$key" in
    72	    ARMS)            is_armset "$val" || reject "ARMS='${val}' is not a comma-separated list of C4L|C8|C16|C32|VANL" ;;
    73	    CHUNK|TARGET|POLL|MAX_CONSEC_FAIL)
    74	                     is_num "$val" || reject "${key}='${val}' is not a non-negative integer" ;;
    75	    ONESHOT|DRYRUN)  case "$val" in 0|1) ;; *) reject "${key}='${val}' must be 0 or 1" ;; esac ;;
    76	    OUTPUT_ROOT|REGISTRY|STATE|LOG)
    77	                     case "$val" in
    78	                       ''|*[!A-Za-z0-9/._-]*) reject "${key}='${val}' has unsafe characters" ;;
    79	                     esac ;;
    80	    *)               reject "unknown argument '${kv}' (expected ARMS=/CHUNK=/TARGET=/POLL=/MAX_CONSEC_FAIL=/ONESHOT=/DRYRUN=/OUTPUT_ROOT=/REGISTRY=/STATE=/LOG=)" ;;
    81	  esac
    82	  printf -v "$key" '%s' "$val"      # name whitelisted above; value never parsed
    83	done
    84	[ "$CHUNK" -gt 0 ] || reject "CHUNK must be positive"
    85	[ "$((CHUNK % 2500))" -eq 0 ] || reject "CHUNK=${CHUNK} is not a multiple of the 2500-step checkpoint cadence"
    86	[ "$TARGET" -gt "$ANCHOR_STEP" ] || reject "TARGET=${TARGET} must exceed the ${ANCHOR_STEP} anchor"
    87	[ "$((TARGET % 2500))" -eq 0 ] || reject "TARGET=${TARGET} is not a multiple of 2500"
    88	[ "$MAX_CONSEC_FAIL" -ge 1 ] || reject "MAX_CONSEC_FAIL must be at least 1"
    89	[ -f "$SUBMITTER" ] || reject "missing ${SUBMITTER}"
    90	[ -f "$RECORDER" ] || reject "missing ${RECORDER}"
    91	[ -f "$REGISTRY" ] || reject "missing ${REGISTRY}"
    92	
    93	log() { printf '%s %s\n' "$(date -Is)" "$1" >> "$LOG"; echo "$1"; }
    94	
    95	# --- state: a plain KV file a human can read and edit ------------------------
    96	# keys: fail_<ARM> <n> | frozen_<ARM> <when>|<reason> | lastjob_<ARM> <jid> |
    97	#       done_<ARM> <step>.  Removing a frozen_<ARM> line un-freezes that arm.
    98	declare -A ST=()
    99	load_state() {
   100	  ST=()
   101	  [ -f "$STATE" ] || return 0
   102	  local k v
   103	  while read -r k v; do
   104	    case "$k" in ''|\#*) continue ;; esac
   105	    ST["$k"]="$v"
   106	  done < "$STATE"
   107	}
   108	save_state() {
   109	  local tmp="${STATE}.tmp.$$" k
   110	  {
   111	    echo "# exp_11 chunk watchdog state (plain KV, rewritten atomically)."
   112	    echo "# Delete a frozen_<ARM> line to let the watchdog submit that arm again."
   113	    if [ "${#ST[@]}" -gt 0 ]; then
   114	      for k in "${!ST[@]}"; do printf '%s %s\n' "$k" "${ST[$k]}"; done | sort
   115	    fi
   116	  } > "$tmp" || { echo "could not write ${tmp}" >&2; return 1; }
   117	  mv -f "$tmp" "$STATE" || { echo "could not publish ${STATE}" >&2; return 1; }
   118	}
   119	
   120	bump_fail() {   # <arm> <reason>
   121	  local arm="$1" reason="$2" n
   122	  n=$(( ${ST[fail_$arm]:-0} + 1 ))
   123	  ST["fail_$arm"]="$n"
   124	  log "${arm}: FAILURE (${reason}) — ${n}/${MAX_CONSEC_FAIL} consecutive"
   125	  if [ "$n" -ge "$MAX_CONSEC_FAIL" ]; then
   126	    ST["frozen_$arm"]="$(date -Is)|${reason}"
   127	    log "${arm}: !!! FROZEN after ${n} consecutive failures (${reason}). No further submission for this arm until its frozen_${arm} line is removed from ${STATE}."
   128	  fi
   129	}
   130	
   131	clear_fail() {  # <arm> <why>
   132	  if [ "${ST[fail_$1]:-0}" != "0" ]; then
   133	    log "${1}: failure streak cleared (${2})"
   134	  fi
   135	  ST["fail_$1"]="0"
   136	}
   137	
   138	newest_ckpt_step() {   # <ckpt dir> -> the largest step with a checkpoint file
   139	  ls -1 "$1" 2>/dev/null \
   140	    | sed -n 's/^epoch=[0-9]\{1,\}-step=\([0-9]\{1,\}\)\.ckpt$/\1/p' | sort -n | tail -1
   141	}
   142	
   143	chain_tip_step() {     # <arm> -> the arm's recorded chain tip (or its anchor, or 0)
   144	  "$PY" - "$REGISTRY" "$1" <<'PY' 2>/dev/null
   145	import json, sys
   146	row = (json.load(open(sys.argv[1])).get("arms") or {}).get(sys.argv[2]) or {}
   147	chain = row.get("chain") or []
   148	print(int((chain[-1].get("final_step") if chain else row.get("final_step")) or 0))
   149	PY
   150	}
   151	
   152	leg_manifest_for() {   # <arm> <chunk_end> -> the ONE launcher manifest for that chunk
   153	  local arm="$1" want="$2" f ce
   154	  local -a hits=()
   155	  for f in "$EXPDIR"/fa_orbit_*_"${arm}"_8x8_jid*_manifest.txt; do
   156	    [ -f "$f" ] || continue
   157	    ce="$(awk '/^chunk_end /{print $2; exit}' "$f" 2>/dev/null)"
   158	    [ "$ce" = "$want" ] && hits+=("$f")
   159	  done
   160	  [ "${#hits[@]}" -eq 1 ] || return 1
   161	  printf '%s\n' "${hits[0]}"
   162	}
   163	
   164	process_arm() {
   165	  local arm="$1"
   166	  if [ -n "${ST[frozen_$arm]:-}" ]; then
   167	    log "${arm}: FROZEN (${ST[frozen_$arm]}) — skipping; clear it in ${STATE} to resume"
   168	    return 0
   169	  fi
   170	
   171	  local queued
   172	  queued="$(squeue -h -u "$USER" -n "exp11-${arm}-train" -o '%i %T' 2>/dev/null)"
   173	
   174	  # settle the leg we last submitted, once it has left the queue
   175	  local last="${ST[lastjob_$arm]:-}" st
   176	  if [ -n "$last" ] && [ -z "$queued" ]; then
   177	    st="$(sacct -X -n -P -j "$last" -o State 2>/dev/null | head -1 | awk '{print $1}')"
   178	    case "$st" in
   179	      COMPLETED)          clear_fail "$arm" "job ${last} COMPLETED" ;;
   180	      FAILED|TIMEOUT|NODE_FAIL|OUT_OF_MEMORY|BOOT_FAIL|DEADLINE|PREEMPTED|CANCELLED)
   181	                          bump_fail "$arm" "job ${last} ended ${st}" ;;
   182	      "")                 log "${arm}: sacct reports nothing for job ${last} — not counted either way" ;;
   183	      *)                  log "${arm}: job ${last} ended ${st} — not counted" ;;
   184	    esac
   185	    unset "ST[lastjob_$arm]"
   186	    [ -n "${ST[frozen_$arm]:-}" ] && return 0
   187	  fi
   188	
   189	  if [ -n "$queued" ]; then
   190	    log "${arm}: live job (${queued//$'\n'/; }) — nothing to do"
   191	    return 0
   192	  fi
   193	
   194	  local ckdir="${OUTPUT_ROOT}/exp11_${arm}/FLAC_exp11_${arm}/exp11_${arm}/checkpoints"
   195	  local S; S="$(newest_ckpt_step "$ckdir")"
   196	  if [ -z "$S" ]; then
   197	    log "${arm}: no epoch=*-step=*.ckpt under ${ckdir} — nothing to resume, skipping"
   198	    return 0
   199	  fi
   200	  if [ "$S" -ge "$TARGET" ]; then
   201	    if [ "${ST[done_$arm]:-}" != "$S" ]; then
   202	      ST["done_$arm"]="$S"
   203	      log "${arm}: DONE — newest checkpoint is step ${S} >= target ${TARGET}"
   204	    fi
   205	    return 0
   206	  fi
   207	
   208	  # a chunk that finished but was never recorded must be recorded FIRST: the next
   209	  # chunk's preflight binds to the recorded chain, not to whatever is on disk.
   210	  if [ "$S" -gt "$ANCHOR_STEP" ]; then
   211	    local tip; tip="$(chain_tip_step "$arm")"
   212	    is_num "$tip" || { log "${arm}: could not read the chain tip from ${REGISTRY}"; bump_fail "$arm" "registry unreadable"; return 0; }
   213	    if [ "$tip" -lt "$S" ]; then
   214	      local man
   215	      if ! man="$(leg_manifest_for "$arm" "$S")"; then
   216	        log "${arm}: no single launcher manifest with chunk_end ${S} in ${EXPDIR} (zero or ambiguous) — cannot record the finished chunk"
   217	        bump_fail "$arm" "manifest for chunk_end ${S} not uniquely identifiable"
   218	        return 0
   219	      fi
   220	      local -a rec=("$RECORDER" "$arm" "$man" --registry "$REGISTRY")
   221	      [ "$DRYRUN" = "1" ] && rec+=(--dry-run)
   222	      log "${arm}: recording finished chunk ${tip} -> ${S} from $(basename "$man")"
   223	      local out rc
   224	      out="$("$PY" "${rec[@]}" 2>&1)"; rc=$?
   225	      if [ "$rc" -ne 0 ]; then
   226	        log "${arm}: RECORDER REFUSED (rc=${rc}): $(printf '%s' "$out" | head -4 | tr '\n' ' ')"
   227	        bump_fail "$arm" "recorder rc=${rc}"
   228	        return 0
   229	      fi
   230	      clear_fail "$arm" "chunk ${tip} -> ${S} recorded"
   231	      if [ "$DRYRUN" = "1" ]; then
   232	        log "${arm}: DRYRUN — the chunk link was validated but not written, so the next chunk is not yet admissible"
   233	      fi
   234	    fi
   235	  fi
   236	
   237	  # submit the next chunk through the ONLY sanctioned submitter
   238	  local next=$(( S + CHUNK ))
   239	  [ "$next" -gt "$TARGET" ] && next="$TARGET"
   240	  local -a ck=("$ckdir"/*-step="${S}".ckpt)
   241	  if [ "${#ck[@]}" -ne 1 ] || [ ! -f "${ck[0]}" ]; then
   242	    log "${arm}: expected exactly one checkpoint file at step ${S} in ${ckdir}, found ${#ck[@]}"
   243	    bump_fail "$arm" "ambiguous checkpoint at step ${S}"
   244	    return 0
   245	  fi
   246	  log "${arm}: submitting chunk ${S} -> ${next}: bash ${SUBMITTER} ${arm} --resume ${ck[0]} --expected-step ${S} --chunk-end ${next} (DRYRUN=${DRYRUN})"
   247	  local out rc jid
   248	  out="$(DRYRUN="$DRYRUN" bash "$SUBMITTER" "$arm" --resume "${ck[0]}" \
   249	          --expected-step "$S" --chunk-end "$next" 2>&1)"; rc=$?
   250	  if [ "$rc" -ne 0 ]; then
   251	    log "${arm}: SUBMIT REFUSED (rc=${rc}): $(printf '%s' "$out" | head -4 | tr '\n' ' ')"
   252	    bump_fail "$arm" "submit rc=${rc}"
   253	    return 0
   254	  fi
   255	  jid="$(printf '%s' "$out" | awk '/^submitted /{print $NF}')"
   256	  if [ -n "$jid" ]; then
   257	    ST["lastjob_$arm"]="$jid"
   258	    log "${arm}: submitted chunk ${S} -> ${next} as job ${jid}"
   259	  else
   260	    log "${arm}: submitter returned 0 without a job id (DRYRUN=${DRYRUN}) — nothing queued"
   261	  fi
   262	  return 0
   263	}
   264	
   265	IFS=',' read -r -a ARM_LIST <<< "$ARMS"
   266	log "=== chunk watchdog start: arms ${ARMS} chunk ${CHUNK} target ${TARGET} poll ${POLL}s max_consec_fail ${MAX_CONSEC_FAIL} oneshot ${ONESHOT} dryrun ${DRYRUN} ==="
   267	while :; do
   268	  load_state
   269	  for ARM in "${ARM_LIST[@]}"; do
   270	    process_arm "$ARM"
   271	  done
   272	  save_state || log "WARNING: the state file could not be written — failure counters will not persist"
   273	
   274	  REMAINING=0
   275	  for ARM in "${ARM_LIST[@]}"; do
   276	    [ -n "${ST[done_$ARM]:-}" ] && continue
   277	    [ -n "${ST[frozen_$ARM]:-}" ] && continue
   278	    REMAINING=$((REMAINING + 1))
   279	  done
   280	  if [ "$REMAINING" -eq 0 ]; then
   281	    log "=== every arm is DONE or FROZEN — watchdog exiting ==="
   282	    break
   283	  fi
   284	  [ "$ONESHOT" = "1" ] && { log "=== ONESHOT: one pass complete, exiting ==="; break; }
   285	  sleep "$POLL"
   286	done
   287	exit 0

exec
/usr/bin/bash -lc "stat -f -c '%T %m' worklog/worklog_yixun/exp_11_fa_orbit_claude
findmnt -T /n/fs/gatrdp/codespace/FLAC -o TARGET,FSTYPE,OPTIONS -n 2>/dev/null" in /n/fs/gatrdp/codespace/FLAC
 succeeded in 0ms:
nfs ?
/n/fs/gatrdp nfs    rw,relatime,vers=3,rsize=131072,wsize=524288,namlen=255,hard,noacl,proto=tcp,timeo=600,retrans=2,sec=sys,mountaddr=128.112.155.142,mountvers=3,mountport=300,mountproto=udp,local_lock=none,addr=128.112.155.142

exec
/usr/bin/bash -lc "grep -n \"flock\\|write_atomic\\|os.open\" worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_add_anchor.py worklog/worklog_yixun/exp_11_fa_orbit_claude/*.py" in /n/fs/gatrdp/codespace/FLAC
 succeeded in 0ms:
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_add_anchor.py:141:    lock_fd = os.open(store, os.O_RDONLY)     # the recorder's lock: one writer, one store
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_add_anchor.py:143:        fcntl.flock(lock_fd, fcntl.LOCK_EX)
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_add_anchor.py:196:        pm.write_atomic(args.registry, reg)
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_add_anchor.py:141:    lock_fd = os.open(store, os.O_RDONLY)     # the recorder's lock: one writer, one store
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_add_anchor.py:143:        fcntl.flock(lock_fd, fcntl.LOCK_EX)
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_add_anchor.py:196:        pm.write_atomic(args.registry, reg)
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_producer_manifest.py:154:        write_atomic(path, doc)
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_producer_manifest.py:158:def write_atomic(path, doc):
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_record_restart.py:302:        pm.write_atomic(args.registry, reg)
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_record_restart.py:339:    lock_fd = os.open(store, os.O_RDONLY)
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_record_restart.py:341:        fcntl.flock(lock_fd, fcntl.LOCK_EX)
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_record_restart.py:420:        pm.write_atomic(args.registry, reg)

2026-08-13T16:22:16.498831Z ERROR codex_models_manager::manager: failed to renew cache TTL: missing field `base_instructions` at line 94 column 5
exec
/usr/bin/bash -lc "nl -ba worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh | sed -n '540,590p'
nl -ba worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_producer_manifest.py | sed -n '212,282p'
nl -ba worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen.sbatch | sed -n '489,515p'" in /n/fs/gatrdp/codespace/FLAC
 succeeded in 0ms:
   540	  bash "$WD" ONESHOT=1 SBATCH_EXTRA=x
   541	# NOTE: this case runs against the REAL queue (no squeue fake): with a live
   542	# exp11-C8-train job it logs "live job — nothing to do", without one it logs
   543	# "nothing to resume, skipping" (the fixture OUTPUT_ROOT is empty). Either way
   544	# a full ONESHOT pass must complete cleanly without submitting anything.
   545	expect_cmd "watchdog completes a ONESHOT pass without submitting" 0 "ONESHOT: one pass complete" -- \
   546	  bash "$WD" ONESHOT=1 DRYRUN=1 ARMS=C8 "OUTPUT_ROOT=${WD_TMP}/outputs" \
   547	    "REGISTRY=${TMP}/chain_registry.json" "STATE=${WD_TMP}/state" "LOG=${WD_TMP}/log"
   548	
   549	echo "--- H. the submitter refuses un-pinned submission ---"
   550	# RETIRED for the same reason as the launcher case above: all pins are concrete,
   551	# so the submitter's placeholder refusal is unreachable on the real file.
   552	if grep -qE '^PINNED_[A-Z_]+="TO-PIN-AFTER-P0"' "$SUBMITTER"; then
   553	  echo "FAIL  a submitter pin is still the placeholder"; FAIL=$((FAIL+1))
   554	else
   555	  echo "PASS  every submitter pin holds a concrete value"; PASS=$((PASS+1))
   556	fi
   557	grep -q 'PLACEHOLDER="TO-PIN-AFTER-P0"' "$SUBMITTER" \
   558	  && { echo "PASS  the submitter still refuses a placeholder pin if one returns"; PASS=$((PASS+1)); } \
   559	  || { echo "FAIL  the submitter placeholder refusal is gone"; FAIL=$((FAIL+1)); }
   560	expect_cmd "submitter rejects a bad arm" 2 "must be C4L" -- env DRYRUN=1 bash "$SUBMITTER" FA1
   561	expect_cmd "submitter derives smoke flags" 0 "--gres=gpu:l40:4" -- \
   562	  env DRYRUN=1 SMOKE=1 SMOKE_RUNG=16x4 SMOKE_MIN_FREE_MB=14000 bash "$SUBMITTER" C4L
   563	expect_cmd "submitter derives cpus/mem from the rung" 0 "--cpus-per-task=36" -- \
   564	  env DRYRUN=1 SMOKE=1 SMOKE_RUNG=16x4 SMOKE_MIN_FREE_MB=14000 bash "$SUBMITTER" C4L
   565	expect_cmd "submitter derives 8x8 resources" 0 "--mem=108G" -- \
   566	  env DRYRUN=1 SMOKE=1 SMOKE_RUNG=8x8 SMOKE_MIN_FREE_MB=14000 bash "$SUBMITTER" C4L
   567	
   568	echo "--- I. flock run ownership, both contention directions (round-3 B3 residual) ---"
   569	LOCKF="${TMP}/exp11_LOCKTEST.lock"
   570	FIFO_HOLD="${TMP}/holder.fifo"; mkfifo "$FIFO_HOLD"
   571	# held by a live process -> a contender must fail to acquire
   572	# the holder keeps the fd open for its whole lifetime, exactly like the launcher
   573	( flock -n 9 || exit 1; read -r _ < "$FIFO_HOLD" ) 9>"$LOCKF" &
   574	HOLDER=$!
   575	sleep 0.5
   576	if flock -n 9 9>"$LOCKF" 2>/dev/null; then
   577	  echo "FAIL  a second holder acquired a held flock"; FAIL=$((FAIL+1))
   578	else
   579	  echo "PASS  contender refused while the lock is held"; PASS=$((PASS+1))
   580	fi
   581	echo go > "$FIFO_HOLD"        # let the holder exit, closing fd 9
   582	wait "$HOLDER" 2>/dev/null
   583	# holder died (kill -9 equivalent) -> the lock must be free immediately, no stale dir
   584	if flock -n 9 9>"$LOCKF" 2>/dev/null; then
   585	  echo "PASS  lock free after the holder exits (no stale-recovery path needed)"; PASS=$((PASS+1))
   586	else
   587	  echo "FAIL  lock still held after the holder exited"; FAIL=$((FAIL+1))
   588	fi
   589	grep -q 'flock -n 9' "$LAUNCHER" && { echo "PASS  launcher uses flock, not mkdir+stale recovery"; PASS=$((PASS+1)); } \
   590	  || { echo "FAIL  launcher does not use flock"; FAIL=$((FAIL+1)); }
   212	def verify_chain(reg, arm, step, ckpt_path, ckpt_sha, base_dir, repo_root="."):
   213	    """Bind ONE checkpoint to the leg that produced it. Returns (problems, note).
   214	
   215	    ``base_dir`` is the directory the registry was read from, so the per-leg files
   216	    travel with it (pinned worktree, or a synthetic registry under test)."""
   217	    row = (reg.get("arms") or {}).get(arm)
   218	    if row is None:
   219	        return [f"{arm} is not in the audited launch registry"], ""
   220	    if not row.get("final_ckpt_sha256"):
   221	        return [f"{arm} has no audited final_ckpt_sha256, so a >40k checkpoint cannot be "
   222	                "chained to its INITIAL run"], ""
   223	    legs = (reg.get("restarts") or {}).get(arm) or []
   224	    if not legs:
   225	        return [f"checkpoint at step {step} is above the INITIAL budget but {arm} has no RESTART "
   226	                "entry in the audited registry — record the leg with fa_orbit_record_restart.py "
   227	                "first"], ""
   228	    why = []
   229	    for i, leg in enumerate(legs):
   230	        bad = validate_leg(leg, row, arm, step=step)
   231	        if bad:
   232	            why.append(f"leg {leg.get('job', i)}: " + "; ".join(bad))
   233	            continue
   234	        # The leg's OWN restart manifest is mutable evidence under gitignored
   235	        # outputs_FLAC, exactly like the INITIAL one the screen already re-hashes:
   236	        # it must still be there, and still be the bytes that were recorded.
   237	        leg_man = resolve(repo_root, str(leg.get("manifest_path")))
   238	        if not os.path.isfile(leg_man):
   239	            why.append(f"leg {leg.get('job')}: the registered RESTART manifest {leg_man} is gone")
   240	            continue
   241	        got = sha256_file(leg_man)
   242	        if got != leg.get("manifest_sha256"):
   243	            why.append(f"leg {leg.get('job')}: RESTART manifest {leg_man} now hashes {got[:12]}, "
   244	                       f"not the registered {str(leg.get('manifest_sha256'))[:12]} — it changed "
   245	                       "after it was recorded")
   246	            continue
   247	        man_path = resolve(base_dir, str(leg.get("producer_manifest")))
   248	        man = load(man_path)
   249	        if man is None:
   250	            why.append(f"leg {leg.get('job')}: producer manifest {man_path} is missing")
   251	            continue
   252	        head_bad = [f"producer manifest {f}={man.get(f)!r} != the registry leg's {leg.get(f)!r}"
   253	                    for f in ("arm", "job", "launch_uuid", "resume_ckpt_sha256", "chains_to")
   254	                    if str(man.get(f)) != str(leg.get(f))]
   255	        if str(man.get("leg_manifest_sha256")) != str(leg.get("manifest_sha256")):
   256	            head_bad.append("producer manifest is not the one this registry row published")
   257	        if head_bad:
   258	            why.append(f"leg {leg.get('job')}: " + "; ".join(head_bad))
   259	            continue
   260	        entry = (man.get("checkpoints") or {}).get(str(step))
   261	        if entry is None:
   262	            why.append(f"leg {leg.get('job')}: produced no checkpoint at step {step} "
   263	                       f"(published: {sorted((man.get('checkpoints') or {}), key=int)})")
   264	            continue
   265	        if entry.get("sha256") != ckpt_sha:
   266	            why.append(f"leg {leg.get('job')}: step {step} was published as "
   267	                       f"{str(entry.get('sha256'))[:12]}, this file hashes {ckpt_sha[:12]} — this "
   268	                       "is NOT the checkpoint that leg produced")
   269	            continue
   270	        if os.path.realpath(resolve(repo_root, str(entry.get("path")))) != os.path.realpath(ckpt_path):
   271	            why.append(f"leg {leg.get('job')}: step {step} was published at {entry.get('path')}, "
   272	                       f"not {ckpt_path}")
   273	            continue
   274	        return [], (f"producer binding OK: step {step} ({ckpt_sha[:12]}) was produced by RESTART "
   275	                    f"job {leg.get('job')}, which resumed the audited "
   276	                    f"{str(row['final_ckpt_sha256'])[:12]} and published it in "
   277	                    f"{os.path.basename(man_path)}")
   278	    return [f"no validated RESTART leg for {arm} published step {step} with sha256 "
   279	            f"{ckpt_sha[:12]} — " + " | ".join(why)], ""
   489	# --- Q10: a checkpoint ABOVE 40k came from a RESTART leg, not the INITIAL run --
   490	# The INITIAL manifest cannot vouch for it. The first version of this gate was
   491	# EXISTENTIAL (re-pin review, finding 2): once ONE leg carried mode=RESTART and
   492	# the right resume hash, every later same-config checkpoint in this directory
   493	# passed, because nothing bound the evaluated FILE to the leg that produced it —
   494	# a checkpoint from a wrong restart, copied in, was admissible. Now the binding
   495	# is per checkpoint: this file is re-hashed and must appear, at exactly this
   496	# step, in the append-only PRODUCER MANIFEST of a leg whose every registry field
   497	# re-validates against the audited INITIAL row (see fa_orbit_producer_manifest.py).
   498	step_i = int(os.path.basename(ckpt).split("step=")[1].split(".ckpt")[0])
   499	if step_i > 40000:
   500	    sys.path.insert(0, expdir)
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
