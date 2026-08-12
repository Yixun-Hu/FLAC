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
session id: 019ff816-10d4-7ca1-9f51-846c0c81b402
--------
user
Round-4 delta review of worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch in /n/fs/gatrdp/codespace/FLAC (read-only; do NOT install anything or modify environments/files; output = review text only). Production failure: RESTART legs 3684149/3684150 died at train.py:193 — the launcher resumed the INITIAL leg's wandb run (WANDB_RESUME=must, same run id) and prefigure's push_wandb_config calls config.update() WITHOUT allow_val_change, so the restart's legitimately-changed config (max_steps 40000->100000, then ckpt_path) raises ConfigError. Full traceback in outputs_FLAC/exp11_C4L/fa_orbit_2026-08-12_18-11-38_C4L_8x8_jid3684149_train.log lines 129-151.

Delta (diff at /tmp/claude-374349/-n-fs-gatrdp-codespace-FLAC/66305eb9-88c7-41df-95da-dde30a426dc2/scratchpad/wandb_restart.diff): RESTART legs now mint a FRESH run id 'exp11-${ARM}-r${EXPECTED_STEP}-<ns>-<uuid8>' instead of resuming; the original id is still read from the launch manifest (fail-closed die retained) and echoed as lineage; WANDB_RESUME is no longer exported. Constraints honored: train.py byte-unchanged (exp_15's admission record pins it), prefigure untouched. Known cost: wandb curves split across runs per leg (accepted; figures come from eval JSONs).

Check: (1) any residual WANDB_RESUME/resume path that could still collide; (2) the run-id readback gate at ~line 670 (fa_orbit_wandb_readback.py --run-id + run-id glob over $REPO/wandb/run-*) works with the new id shape (r-token, dashes); (3) the manifest writer (line ~648 'wandb_run_id ${WANDB_RUN_ID}') now records the NEW id — does anything downstream (record_restart/registry/readback, future crash-restarts reading LAUNCH_MANIFEST_LINK) break or mislink when the manifest chain's wandb ids differ per leg; (4) WANDB_RUN_ID char limit (wandb caps run ids at 128 chars? verify the constructed id length is safe); (5) fresh holes. Verdict: APPROVED or REVISE.
2026-08-12T22:27:46.234795Z ERROR codex_models_manager::manager: failed to renew cache TTL: missing field `base_instructions` at line 94 column 5
codex
I’ll do a read-only delta review against the stated failure mode, inspecting the patch plus the surrounding launcher, manifest, registry, and readback paths. I won’t modify files or environments.
  → Inspect the supplied delta and current sbatch restart/W&B logic
  • Trace manifest, registry, readback, and future crash-restart consumers
  • Verify W&B run-ID constraints and issue a focused verdict
2026-08-12T22:28:03.944362Z ERROR codex_models_manager::manager: failed to renew cache TTL: missing field `base_instructions` at line 94 column 5
exec
/usr/bin/bash -lc "git status --short && git diff -- worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch && sed -n '1,240p' /tmp/claude-374349/-n-fs-gatrdp-codespace-FLAC/66305eb9-88c7-41df-95da-dde30a426dc2/scratchpad/wandb_restart.diff" in /n/fs/gatrdp/codespace/FLAC
 succeeded in 8946ms:
 M worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch
?? AGENTS.md
?? AcousticRooms
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-12_18-11-38_C4L_8x8_jid3684149_manifest.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-12_18-19-40_C8_8x8_jid3684150_manifest.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-12_18-21-40_C16_8x8_jid3684151_manifest.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-12_18-23-10_guardtests.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md
?? worklog/worklog_yixun/exp_15_yaw_aug_claude/slurm_train_exp15-smoke-YAWAUG_3685989.out
?? worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-12_11-59-14_readback.log
?? worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-12_11-59-42_readback.log
?? worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-12_12-05-35_guardtests_fullfix.log
?? worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-12_12-35-37_guardtests_f3fix.log
?? worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-12_12-40-22_guardtests_f3fix.log
?? worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-12_17-49-31_YAWAUG_8x8_jid3685989_manifest.txt
?? worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_smoke_registry.json
diff --git a/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch b/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch
index 185d911..3bfa2e7 100644
--- a/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch
+++ b/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch
@@ -560,10 +560,18 @@ PY
 export WANDB_ENTITY="$WANDB_ENTITY_SEEN"
 echo "wandb identity OK: entity ${WANDB_ENTITY} (exported), mode ${WANDB_MODE}"
 if [ "$MODE" = "RESTART" ]; then
-  WANDB_RUN_ID="$(awk '/^wandb_run_id /{print $2}' "$LAUNCH_MANIFEST_LINK" 2>/dev/null)"
-  [ -n "$WANDB_RUN_ID" ] || die "RESTART needs the original wandb run id from ${LAUNCH_MANIFEST_LINK} - abort"
-  export WANDB_RUN_ID WANDB_RESUME=must
-  echo "wandb: resuming run ${WANDB_RUN_ID} (resume=must)"
+  ORIG_WANDB_RUN_ID="$(awk '/^wandb_run_id /{print $2}' "$LAUNCH_MANIFEST_LINK" 2>/dev/null)"
+  [ -n "$ORIG_WANDB_RUN_ID" ] || die "RESTART needs the original wandb run id from ${LAUNCH_MANIFEST_LINK} - abort"
+  # A RESUMED wandb run refuses config changes: prefigure's push_wandb_config
+  # calls config.update() without allow_val_change, and a restart legitimately
+  # changes max_steps (40000->100000) and ckpt_path — legs 3684149/3684150
+  # died at exactly train.py:193 on this. So a restart runs as a FRESH run
+  # carrying the lineage in its name; curves split across runs but panels
+  # align on the step axis. train.py stays byte-unchanged (exp_15 admission
+  # pins it).
+  WANDB_RUN_ID="exp11-${ARM}-r${EXPECTED_STEP}-$(date +%s%N)-$(cut -c1-8 /proc/sys/kernel/random/uuid)"
+  export WANDB_RUN_ID
+  echo "wandb: new RESTART run id ${WANDB_RUN_ID} (continues ${ORIG_WANDB_RUN_ID}; not resumed — a resumed run refuses the changed max_steps/ckpt_path config)"
 else
   WANDB_RUN_ID="exp11-${ARM}-$(date +%s%N)-$(cut -c1-8 /proc/sys/kernel/random/uuid)"
   export WANDB_RUN_ID
diff --git a/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch b/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch
index 185d911..3bfa2e7 100644
--- a/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch
+++ b/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch
@@ -560,10 +560,18 @@ PY
 export WANDB_ENTITY="$WANDB_ENTITY_SEEN"
 echo "wandb identity OK: entity ${WANDB_ENTITY} (exported), mode ${WANDB_MODE}"
 if [ "$MODE" = "RESTART" ]; then
-  WANDB_RUN_ID="$(awk '/^wandb_run_id /{print $2}' "$LAUNCH_MANIFEST_LINK" 2>/dev/null)"
-  [ -n "$WANDB_RUN_ID" ] || die "RESTART needs the original wandb run id from ${LAUNCH_MANIFEST_LINK} - abort"
-  export WANDB_RUN_ID WANDB_RESUME=must
-  echo "wandb: resuming run ${WANDB_RUN_ID} (resume=must)"
+  ORIG_WANDB_RUN_ID="$(awk '/^wandb_run_id /{print $2}' "$LAUNCH_MANIFEST_LINK" 2>/dev/null)"
+  [ -n "$ORIG_WANDB_RUN_ID" ] || die "RESTART needs the original wandb run id from ${LAUNCH_MANIFEST_LINK} - abort"
+  # A RESUMED wandb run refuses config changes: prefigure's push_wandb_config
+  # calls config.update() without allow_val_change, and a restart legitimately
+  # changes max_steps (40000->100000) and ckpt_path — legs 3684149/3684150
+  # died at exactly train.py:193 on this. So a restart runs as a FRESH run
+  # carrying the lineage in its name; curves split across runs but panels
+  # align on the step axis. train.py stays byte-unchanged (exp_15 admission
+  # pins it).
+  WANDB_RUN_ID="exp11-${ARM}-r${EXPECTED_STEP}-$(date +%s%N)-$(cut -c1-8 /proc/sys/kernel/random/uuid)"
+  export WANDB_RUN_ID
+  echo "wandb: new RESTART run id ${WANDB_RUN_ID} (continues ${ORIG_WANDB_RUN_ID}; not resumed — a resumed run refuses the changed max_steps/ckpt_path config)"
 else
   WANDB_RUN_ID="exp11-${ARM}-$(date +%s%N)-$(cut -c1-8 /proc/sys/kernel/random/uuid)"
   export WANDB_RUN_ID

exec
/usr/bin/bash -lc "rg -n -i 'wandb|manifest|record_restart|registry|run-id|run_id|LAUNCH_MANIFEST|RESTART|EXPECTED_STEP' worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch worklog/worklog_yixun/exp_11_fa_orbit_claude --glob '*.py' --glob '*.sh' --glob '*.sbatch' --glob '*.json' --glob '*.txt'" in /n/fs/gatrdp/codespace/FLAC
 exited 127 in 5649ms:
/usr/bin/bash: line 1: rg: command not found

exec
/usr/bin/bash -lc "grep -RInE 'wandb|manifest|record_restart|registry|run-id|run_id|LAUNCH_MANIFEST|RESTART|EXPECTED_STEP' worklog/worklog_yixun/exp_11_fa_orbit_claude --include='*.py' --include='*.sh' --include='*.sbatch' --include='*.json' --include='*.txt'" in /n/fs/gatrdp/codespace/FLAC
 succeeded in 12000ms:
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C8_1786056695036465420-2fc0ef50.txt:7:pins rung=8x8 maxsteps=40000 ckpt_every=2500 min_free_mb=36500 p0_manifest_sha256=72607b922177208d56055d604b292d697b643ef3b7ab48261ab2e23a0cc2b53b
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_submit.sh:237:# --- intent manifest: what was submitted, at which commit, with what exclusions
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_submit.sh:248:} > "$INTENT" || echo "WARNING: could not write the intent manifest ${INTENT}" >&2
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_submit.sh:249:echo "intent manifest: ${INTENT}"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C16_1786310422371467848-a776b47c.txt:7:pins rung=8x8 maxsteps=100000 ckpt_every=2500 min_free_mb=36500 p0_manifest_sha256=72607b922177208d56055d604b292d697b643ef3b7ab48261ab2e23a0cc2b53b
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C16_1786310422371467848-a776b47c.txt:9:sbatch sbatch --job-name=exp11-C16-train --gres=gpu:l40:8 --cpus-per-task=64 --mem=108G --time=89:00:00 --export=ALL,ARM=C16,EXPECT_SHA=c85bc612c4f431cc8f55e937907e36d846cd7085,OUTPUT_ROOT=outputs_FLAC,RESUME_CKPT=outputs_FLAC/exp11_C16/FLAC_exp11_C16/exp11_C16/checkpoints/epoch=8-step=40000.ckpt,EXPECTED_STEP=40000 worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C32_1786465302839671032-280f410d.txt:7:pins rung=8x8 maxsteps=100000 ckpt_every=2500 min_free_mb=36500 p0_manifest_sha256=72607b922177208d56055d604b292d697b643ef3b7ab48261ab2e23a0cc2b53b
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C32_1786465302839671032-280f410d.txt:9:sbatch sbatch --job-name=exp11-C32-train --gres=gpu:l40:8 --cpus-per-task=64 --mem=108G --time=160:00:00 --export=ALL,ARM=C32,EXPECT_SHA=0f0acb2e87debd872f14de91613a89e5760908df,OUTPUT_ROOT=outputs_FLAC,RESUME_CKPT=outputs_FLAC/exp11_C32/FLAC_exp11_C32/exp11_C32/checkpoints/epoch=8-step=40000.ckpt,EXPECTED_STEP=40000 worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py:2:"""exp_11 RESTART checkpoint preflight (round-3 review B2).
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py:10:  - the checkpoint's embedded ``global_step`` equals EXPECTED_STEP exactly;
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py:20:  - optionally, the arm's ORIGINAL launch manifest is re-read and the restart is
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py:23:Prints the checkpoint sha256 (for the restart manifest) and a lineage summary.
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py:68:def parse_manifest(path):
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py:69:    """The launcher's own manifest format: whitespace-separated `key value...`."""
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py:81:def check_manifest_binding(manifest_path, arm, rung, commit, maxsteps):
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py:82:    man = parse_manifest(manifest_path)
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py:89:        problems.append(f"manifest arm {kv.get('arm')!r} != {arm!r}")
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py:91:        problems.append(f"manifest rung {kv.get('rung')!r} != {rung!r} "
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py:95:        problems.append(f"manifest max_steps {kv.get('max_steps')!r} != {maxsteps}")
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py:96:    # Fail-CLOSED (round-3 B2 residual): an absent or empty manifest commit is not
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py:100:        problems.append("launch manifest carries no 'commit' line — cannot bind the restart "
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py:103:        problems.append("no running commit supplied to compare against the manifest commit")
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py:105:        problems.append(f"manifest commit {man_commit[:12]} != running commit {commit[:12]}")
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py:110:    """One manifest line's `k v k v ...` pairs (the launcher's `arm ...`/`job ...`)."""
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py:123:def check_extension_binding(manifest_path, registry_path, arm, rung, config_path, ckpt_path,
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py:128:    so `check_manifest_binding` demands both. An extension breaks both BY DESIGN
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py:134:    proves it against the COMMITTED registry rather than the mutable manifest
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py:135:    alone: the INITIAL manifest byte-for-byte as audited, the same job/uuid/
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py:142:    if not os.path.isfile(registry_path):
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py:143:        return [f"audited launch registry not found: {registry_path}"], {}
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py:144:    reg = json.load(open(registry_path)).get("arms", {}).get(arm)
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py:146:        return [f"{arm} is not in the audited launch registry {registry_path}"], {}
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py:147:    man = parse_manifest(manifest_path)
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py:150:    got_sha = sha256_file(manifest_path)
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py:151:    if got_sha != reg.get("manifest_sha256"):
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py:152:        problems.append(f"launch manifest sha256 {got_sha[:12]} != audited "
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py:153:                        f"{str(reg.get('manifest_sha256'))[:12]} — the manifest changed after it "
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py:167:        problems.append("launch manifest carries no 'commit' line — cannot bind the extension to "
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py:170:        problems.append(f"manifest commit {man_commit[:12]} != the registered launch commit "
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py:174:    # The INITIAL budget is the manifest's and the registry's; the extension's is
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py:178:        problems.append(f"manifest max_steps {kv.get('max_steps')!r} != registered "
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py:192:        problems.append(f"{arm} has no audited final_ckpt_sha256 in the registry — the extension "
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py:198:        problems.append(f"EXPECTED_STEP {expected_step} != the registered final_step {final_step}")
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py:201:        problems.append("manifest records no save_dir")
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py:219:    ap.add_argument("--launch-manifest", default="",
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py:220:                    help="the arm's original launch manifest (binds rung/commit/budget)")
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py:224:    ap.add_argument("--launch-registry", default="",
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py:225:                    help="the committed arm launch registry (required with --extension)")
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py:227:                    help="root the registry's relative save_dir is resolved against")
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py:229:    if args.extension and not args.launch_registry:
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py:230:        ap.error("--extension requires --launch-registry (the audited INITIAL launch row)")
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py:251:        problems.append(f"global_step {gs} != EXPECTED_STEP {args.expected_step}")
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py:282:    if args.launch_manifest:
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py:283:        if not os.path.isfile(args.launch_manifest):
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py:284:            problems.append(f"launch manifest not found: {args.launch_manifest}")
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py:287:                args.launch_manifest, args.launch_registry, args.arm, args.rung, args.config,
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py:291:            more, man = check_manifest_binding(args.launch_manifest, args.arm, args.rung,
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py:295:        problems.append("--extension requires --launch-manifest (the audited INITIAL manifest)")
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py:314:        print(f"  bound to the audited launch manifest: {args.launch_manifest}")
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py:316:        print(f"  bound to launch manifest: {args.launch_manifest}")
worklog/worklog_yixun/exp_11_fa_orbit_claude/p0_manifest_aa4bc18-1785968431124626318-df9602ea.txt:1:# exp_11 P0 submission manifest (consumed by p0_collect.py --manifest)
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:25:#   RESUME_CKPT/EXPECTED_STEP   crash restart only (see LINEAGE)
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:30:#   INITIAL  no RESUME_CKPT, EXPECTED_STEP unset/0, run directory absent.
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:31:#   RESTART  EXPECTED_STEP > 0 AND RESUME_CKPT inside this arm's OWN
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:34:#            scheduler/EMA + binding to the original launch manifest).
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:91:# Q10 RESTART legs: 40k -> 100k is 60,000 further steps at the batched rates,
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:93:PINNED_TIME_LIMIT_RESTART_C4L="34:00:00"    # 60k/0.6598 = 25.3 h
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:94:PINNED_TIME_LIMIT_RESTART_C8="51:00:00"     # 60k/0.4351 = 38.3 h
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:95:PINNED_TIME_LIMIT_RESTART_C16="89:00:00"    # 60k/0.2454 = 67.9 h
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:96:PINNED_TIME_LIMIT_RESTART_C32="160:00:00"   # 60k/0.1308 = 127.4 h (cap 168 h)
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:97:PINNED_TIME_LIMIT_RESTART_VANL="19:00:00"   # 60k/1.0722 = 15.5 h
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:98:PINNED_P0_MANIFEST_SHA256="72607b922177208d56055d604b292d697b643ef3b7ab48261ab2e23a0cc2b53b"  # batched matrix manifest bd96575-…-a3ed28eb; spot manifest sha in the commit message
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:139:EXPECTED_STEP="${EXPECTED_STEP:-0}"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:151:case "$EXPECTED_STEP" in ''|*[!0-9]*) die "EXPECTED_STEP '${EXPECTED_STEP}' must be a non-negative integer - abort";; esac
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:166:  # PINNED_TIME_LIMIT_RESTART_<ARM>. The job selected PINNED_TIME_LIMIT_<ARM>
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:170:  if [ "$EXPECTED_STEP" -gt 0 ]; then
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:171:    TIME_PIN_NAME="PINNED_TIME_LIMIT_RESTART_${ARM}"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:221:# Record/analysis files (registry, manifests, gen_*/validators, worklog)
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:236:      "$EXPDIR"/fa_orbit_wandb_readback.py "$EXPDIR"/fa_orbit_classify.py \
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:328:# --- E. lineage: INITIAL vs RESTART -------------------------------------------
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:331:LAUNCH_MANIFEST_LINK="${SAVEDIR}/launch_manifest.txt"     # written by the INITIAL launch
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:332:if [ "$EXPECTED_STEP" -eq 0 ]; then
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:334:  [ -z "$RESUME_CKPT" ] || die "INITIAL launch must not carry RESUME_CKPT (set EXPECTED_STEP > 0 to declare a RESTART) - abort"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:337:  MODE="RESTART"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:338:  [ -n "$RESUME_CKPT" ] || die "EXPECTED_STEP ${EXPECTED_STEP} declares a RESTART, but RESTART requires RESUME_CKPT - abort"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:344:    *) die "a RESTART may only resume a checkpoint from ${CKPT_DIR_REAL}/ (got ${RESUME_REAL}) - abort" ;;
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:346:  [ "$MAXSTEPS" -gt "$EXPECTED_STEP" ] || die "MAXSTEPS ${MAXSTEPS} must exceed the resume step ${EXPECTED_STEP} - abort"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:348:echo "lineage: ${MODE} (expected_step ${EXPECTED_STEP}, max_steps ${MAXSTEPS}, ckpt every ${CHECKPOINT_EVERY}, time pin ${TIME_PIN_NAME}=${TIME_LIMIT})"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:360:  --logger wandb --checkpoint-every "$CHECKPOINT_EVERY"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:363:[ "$MODE" = "RESTART" ] && ARGV+=(--ckpt-path "$RESUME_CKPT")
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:376:--logger wandb --checkpoint-every 2500
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:412:            (allowed if mode == "RESTART" else violations).append(
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:413:                f"--ckpt-path: {new[flag]!r} (RESTART only)")
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:441:  echo "  (Slurm/GPU/VRAM/env/wandb/ViT/lock gates and training are skipped in DRYRUN)"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:457:# RESTART leg (or the reverse) is refused here, in the job, not merely intended
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:522:# --- L. RESTART preflight (round-3 B2) ---------------------------------------
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:524:if [ "$MODE" = "RESTART" ]; then
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:525:  PRE_ARGS=(--ckpt "$RESUME_CKPT" --expected-step "$EXPECTED_STEP" --config "$MODEL_CONFIG_ABS"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:527:  [ -n "$LAUNCH_MANIFEST_LINK" ] && PRE_ARGS+=(--launch-manifest "$LAUNCH_MANIFEST_LINK")
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:529:  # contract binds the ORIGINAL launch identity (audited manifest bytes, job,
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:533:  [ "$SMOKE" != "1" ] && PRE_ARGS+=(--extension --launch-registry "$EXPDIR/arm_launch_registry.json"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:541:# --- M. wandb: scrub, pin the destination, fix the run id (round-3 B7) --------
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:549:    import wandb
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:550:    v = wandb.Api().viewer
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:552:    sys.exit(f"wandb identity check FAILED: {e}")
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:554:    sys.exit(f"wandb identity {v.email} != yh4742@princeton.edu")
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:557:)" || die "wandb identity gate FAILED (no logger fallback: the arms train with wandb) - abort"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:558:[ -n "$WANDB_ENTITY_SEEN" ] || die "wandb returned an empty entity - abort"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:561:echo "wandb identity OK: entity ${WANDB_ENTITY} (exported), mode ${WANDB_MODE}"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:562:if [ "$MODE" = "RESTART" ]; then
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:563:  ORIG_WANDB_RUN_ID="$(awk '/^wandb_run_id /{print $2}' "$LAUNCH_MANIFEST_LINK" 2>/dev/null)"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:564:  [ -n "$ORIG_WANDB_RUN_ID" ] || die "RESTART needs the original wandb run id from ${LAUNCH_MANIFEST_LINK} - abort"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:565:  # A RESUMED wandb run refuses config changes: prefigure's push_wandb_config
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:572:  WANDB_RUN_ID="exp11-${ARM}-r${EXPECTED_STEP}-$(date +%s%N)-$(cut -c1-8 /proc/sys/kernel/random/uuid)"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:574:  echo "wandb: new RESTART run id ${WANDB_RUN_ID} (continues ${ORIG_WANDB_RUN_ID}; not resumed — a resumed run refuses the changed max_steps/ckpt_path config)"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:578:  echo "wandb: new run id ${WANDB_RUN_ID}"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:584:# --- O. atomic manifest, duplicated to the save-dir (round-3 B5) --------------
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:622:MANIFEST="${EXPDIR}/fa_orbit_${TS}_${ARM}_${RUNG}_jid${SLURM_JOB_ID}_manifest.txt"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:629:  echo "# exp_11 arm launch manifest"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:634:  echo "p0_manifest_sha256 ${PINNED_P0_MANIFEST_SHA256}"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:642:  echo "resume_ckpt ${RESUME_CKPT:-<none>} expected_step ${EXPECTED_STEP} resume_ckpt_sha256 ${CKPT_SHA:-<none>}"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:647:  echo "wandb_entity ${WANDB_ENTITY_SEEN} wandb_project ${NAME} wandb_name ${EXPNAME}"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:648:  echo "wandb_run_id ${WANDB_RUN_ID}"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:650:} > "${MANIFEST}.tmp" || die "manifest write FAILED - abort" 3
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:651:mv "${MANIFEST}.tmp" "$MANIFEST" || die "manifest publication FAILED - abort" 3
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:652:cp "$MANIFEST" "${SAVEDIR}/$(basename "$MANIFEST")" || die "manifest copy to the save-dir FAILED - abort" 3
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:653:[ "$MODE" = "INITIAL" ] && { cp "$MANIFEST" "$LAUNCH_MANIFEST_LINK" || die "launch-manifest link write FAILED - abort" 3; }
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:654:echo "manifest: ${MANIFEST} (copied to ${SAVEDIR})"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:694:# its default save_dir='.' into wandb.init and that OVERRIDES the exported
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:695:# WANDB_DIR: in job 3646734 the run went to $REPO/wandb/run-<ts>-<id> while this
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:696:# check looked under $WANDB_DIR/wandb and found nothing (training was green, the
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:697:# job still classified 7). We keep exporting WANDB_DIR — other wandb artifacts do
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:699:# wandb embeds in the directory name, across both candidate roots. Exactly one
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:702:python3 "$EXPDIR/fa_orbit_wandb_readback.py" --run-id "$WANDB_RUN_ID" \
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:706:  echo "W&B run identity could not be verified against the manifest - provenance failure"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:723:  echo "torchrun rc=${rc} tee rc=${tee_rc} wandb_identity_rc=${WANDB_CHECK_RC} classified rc=${final_rc}"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:751:  echo "LOG-PROVENANCE: final record tee rc=${final_tee_rc}, preflight copy rc=${PREFLIGHT_COPY_RC}, wandb identity rc=${WANDB_CHECK_RC}"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C4L_1786038172477244742-627d96c6.txt:7:pins rung=8x8 maxsteps=40000 ckpt_every=2500 min_free_mb=35500 p0_manifest_sha256=b2aeaf9c1e797d5268f02faa594cc416cb59c113e8c3a8be70ad7f34e242208e
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C4L_1786047832741741064-d5f916b0.txt:7:pins rung=8x8 maxsteps=40000 ckpt_every=2500 min_free_mb=36500 p0_manifest_sha256=72607b922177208d56055d604b292d697b643ef3b7ab48261ab2e23a0cc2b53b
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_VANL_1786476227361783047-3dca770a.txt:7:pins rung=8x8 maxsteps=100000 ckpt_every=2500 min_free_mb=36500 p0_manifest_sha256=72607b922177208d56055d604b292d697b643ef3b7ab48261ab2e23a0cc2b53b
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_VANL_1786476227361783047-3dca770a.txt:9:sbatch sbatch --job-name=exp11-VANL-train --gres=gpu:l40:8 --cpus-per-task=64 --mem=108G --time=19:00:00 --export=ALL,ARM=VANL,EXPECT_SHA=da7ee7f3556aee541344dea6ce76479b4495e529,OUTPUT_ROOT=outputs_FLAC,RESUME_CKPT=outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=8-step=40000.ckpt,EXPECTED_STEP=40000 worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C16_1786054560564868965-5fd4c1e1.txt:7:pins rung=8x8 maxsteps=40000 ckpt_every=2500 min_free_mb=36500 p0_manifest_sha256=72607b922177208d56055d604b292d697b643ef3b7ab48261ab2e23a0cc2b53b
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C8_1786054560465501451-9ffdd4d5.txt:7:pins rung=8x8 maxsteps=40000 ckpt_every=2500 min_free_mb=36500 p0_manifest_sha256=72607b922177208d56055d604b292d697b643ef3b7ab48261ab2e23a0cc2b53b
worklog/worklog_yixun/exp_11_fa_orbit_claude/p0_submit_matrix.sh:16:#   workers <FAM> <RUNG> the 0-vs-6-worker pair for one cell, in ONE manifest
worklog/worklog_yixun/exp_11_fa_orbit_claude/p0_submit_matrix.sh:19:# random hex) and an ATOMIC, NO-CLOBBER manifest listing runid, mode, commit sha
worklog/worklog_yixun/exp_11_fa_orbit_claude/p0_submit_matrix.sh:64:manifest_begin() {  # $1 = mode
worklog/worklog_yixun/exp_11_fa_orbit_claude/p0_submit_matrix.sh:65:  MANIFEST="$EXPDIR/p0_manifest_${RUNID}.txt"
worklog/worklog_yixun/exp_11_fa_orbit_claude/p0_submit_matrix.sh:66:  [ ! -e "$MANIFEST" ] || { echo "manifest ${MANIFEST} already exists - abort (run id collision)"; exit 2; }
worklog/worklog_yixun/exp_11_fa_orbit_claude/p0_submit_matrix.sh:70:    echo "# exp_11 P0 submission manifest (consumed by p0_collect.py --manifest)"
worklog/worklog_yixun/exp_11_fa_orbit_claude/p0_submit_matrix.sh:79:manifest_publish() {
worklog/worklog_yixun/exp_11_fa_orbit_claude/p0_submit_matrix.sh:80:  [ ! -e "$MANIFEST" ] || { echo "manifest ${MANIFEST} appeared during submission - abort"; exit 2; }
worklog/worklog_yixun/exp_11_fa_orbit_claude/p0_submit_matrix.sh:81:  mv -n "$MANIFEST_TMP" "$MANIFEST" || { echo "manifest publication failed - abort"; exit 2; }
worklog/worklog_yixun/exp_11_fa_orbit_claude/p0_submit_matrix.sh:82:  [ -e "$MANIFEST" ] || { echo "manifest ${MANIFEST} was not published - abort"; exit 2; }
worklog/worklog_yixun/exp_11_fa_orbit_claude/p0_submit_matrix.sh:84:  echo "manifest: ${MANIFEST}"
worklog/worklog_yixun/exp_11_fa_orbit_claude/p0_submit_matrix.sh:85:  echo "collect with: python ${EXPDIR}/p0_collect.py --manifest ${MANIFEST}"
worklog/worklog_yixun/exp_11_fa_orbit_claude/p0_submit_matrix.sh:130:    manifest_begin matrix
worklog/worklog_yixun/exp_11_fa_orbit_claude/p0_submit_matrix.sh:134:    manifest_publish
worklog/worklog_yixun/exp_11_fa_orbit_claude/p0_submit_matrix.sh:139:    manifest_begin spot
worklog/worklog_yixun/exp_11_fa_orbit_claude/p0_submit_matrix.sh:142:    manifest_publish
worklog/worklog_yixun/exp_11_fa_orbit_claude/p0_submit_matrix.sh:148:    # ONE manifest: the halves differ only in worker count, and the collector keys
worklog/worklog_yixun/exp_11_fa_orbit_claude/p0_submit_matrix.sh:150:    manifest_begin workers
worklog/worklog_yixun/exp_11_fa_orbit_claude/p0_submit_matrix.sh:153:    manifest_publish
worklog/worklog_yixun/exp_11_fa_orbit_claude/p0_submit_matrix.sh:160:  echo "${FAILURES} submission(s) FAILED - the manifest records them as SUBMIT_FAILED and collection will refuse to report"
worklog/worklog_yixun/exp_11_fa_orbit_claude/p0_manifest_bd96575-1786045321510462046-a3ed28eb.txt:1:# exp_11 P0 submission manifest (consumed by p0_collect.py --manifest)
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-06_20-55-01_C8_8x8_jid3648695_manifest.txt:1:# exp_11 arm launch manifest
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-06_20-55-01_C8_8x8_jid3648695_manifest.txt:6:p0_manifest_sha256 72607b922177208d56055d604b292d697b643ef3b7ab48261ab2e23a0cc2b53b
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-06_20-55-01_C8_8x8_jid3648695_manifest.txt:18:wandb_entity yh4742-princeton-university wandb_project FLAC_exp11_C8 wandb_name exp11_C8
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-06_20-55-01_C8_8x8_jid3648695_manifest.txt:19:wandb_run_id exp11-C8-1786064131292302937-6d92e299
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-06_20-55-01_C8_8x8_jid3648695_manifest.txt:20:command torchrun --standalone --nnodes=1 --nproc_per_node=8 train.py --model-config /n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/FLAC_AR_BF_C8.json --dataset-config src/configs/dataset_configs/AR/train/acousticroom_train.json --pretransform-ckpt-path weights/FLAC/VAE.safetensors --max-steps 40000 --batch-size 8 --accum-batches 1 --num-workers 6 --seed 42 --num-gpus 8 --num-nodes 1 --strategy ddp_find_unused_parameters_true --sync-batchnorm true --precision bf16-mixed --val-every -1 --val-dataset-config  --gradient-clip-val 0.0 --logger wandb --checkpoint-every 2500 --name FLAC_exp11_C8 --experiment-name exp11_C8 --save-dir outputs_FLAC/exp11_C8
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C16_1786531886943324938-89cca289.txt:7:pins rung=8x8 maxsteps=100000 ckpt_every=2500 min_free_mb=36500 p0_manifest_sha256=72607b922177208d56055d604b292d697b643ef3b7ab48261ab2e23a0cc2b53b
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C16_1786531886943324938-89cca289.txt:9:sbatch sbatch --job-name=exp11-C16-train --gres=gpu:l40:8 --cpus-per-task=64 --mem=108G --time=89:00:00 --export=ALL,ARM=C16,EXPECT_SHA=2b75036651c1d23a095a32d48117747c633e6008,OUTPUT_ROOT=outputs_FLAC,RESUME_CKPT=outputs_FLAC/exp11_C16/FLAC_exp11_C16/exp11_C16/checkpoints/epoch=8-step=40000.ckpt,EXPECTED_STEP=40000 worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C8_1786476226855877847-ed87fb45.txt:7:pins rung=8x8 maxsteps=100000 ckpt_every=2500 min_free_mb=36500 p0_manifest_sha256=72607b922177208d56055d604b292d697b643ef3b7ab48261ab2e23a0cc2b53b
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C8_1786476226855877847-ed87fb45.txt:9:sbatch sbatch --job-name=exp11-C8-train --gres=gpu:l40:8 --cpus-per-task=64 --mem=108G --time=51:00:00 --export=ALL,ARM=C8,EXPECT_SHA=da7ee7f3556aee541344dea6ce76479b4495e529,OUTPUT_ROOT=outputs_FLAC,RESUME_CKPT=outputs_FLAC/exp11_C8/FLAC_exp11_C8/exp11_C8/checkpoints/epoch=8-step=40000.ckpt,EXPECTED_STEP=40000 worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C16_1786465302622561406-f725e951.txt:7:pins rung=8x8 maxsteps=100000 ckpt_every=2500 min_free_mb=36500 p0_manifest_sha256=72607b922177208d56055d604b292d697b643ef3b7ab48261ab2e23a0cc2b53b
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C16_1786465302622561406-f725e951.txt:9:sbatch sbatch --job-name=exp11-C16-train --gres=gpu:l40:8 --cpus-per-task=64 --mem=108G --time=89:00:00 --export=ALL,ARM=C16,EXPECT_SHA=0f0acb2e87debd872f14de91613a89e5760908df,OUTPUT_ROOT=outputs_FLAC,RESUME_CKPT=outputs_FLAC/exp11_C16/FLAC_exp11_C16/exp11_C16/checkpoints/epoch=8-step=40000.ckpt,EXPECTED_STEP=40000 worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_VANL_1786289214348203957-1be1edd5.txt:7:pins rung=8x8 maxsteps=40000 ckpt_every=2500 min_free_mb=36500 p0_manifest_sha256=72607b922177208d56055d604b292d697b643ef3b7ab48261ab2e23a0cc2b53b
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C4L_1786052743947331089-816fe670.txt:7:pins rung=8x8 maxsteps=40000 ckpt_every=2500 min_free_mb=36500 p0_manifest_sha256=72607b922177208d56055d604b292d697b643ef3b7ab48261ab2e23a0cc2b53b
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:95:# launch manifests so the LATER gates (identity, EMA) are the ones under test;
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:96:# the "no manifest" case below uses an arm deliberately left without one.
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:98:def manifest(arm, cfg_path):
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:102:    with open(os.path.join(d, "launch_manifest.txt"), "w") as fh:
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:106:        fh.write(f"p0_manifest_sha256 {'a' * 64}\n")
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:111:        "manifest_path": os.path.join(d, "launch_manifest.txt"),
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:112:        "manifest_sha256": hashlib.sha256(
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:113:            open(os.path.join(d, "launch_manifest.txt"), "rb").read()).hexdigest(),
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:117:        "p0_manifest_sha256": "a" * 64, "save_dir": d, "training_seed": 42,
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:121:manifest("C4L", os.path.join(expdir, "FLAC_AR_BF_C4L.json"))
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:122:manifest("C16", os.path.join(expdir, "FLAC_AR_BF_C16.json"))
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:123:manifest("VANL", os.path.join(expdir, "FLAC_AR_VANCKPT.json"))
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:124:with open(os.path.join(out, "arm_launch_registry.json"), "w") as fh:
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:130:      "FA_ORBIT_ARM_REGISTRY=${OUT_ROOT}/arm_launch_registry.json")
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:132:register_manifest() {  # <arm> — record the manifest as it stands, faithfully
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:133:  $PY - "$1" "${OUT_ROOT}/exp11_$1/launch_manifest.txt" "${OUT_ROOT}/arm_launch_registry.json" <<'PY'
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:147:    "manifest_path": man_path, "manifest_sha256": hashlib.sha256(raw).hexdigest(),
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:152:    "p0_manifest_sha256": man.get("p0_manifest_sha256"), "save_dir": man.get("save_dir"),
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:203:# the temp root has NO launch manifests, so every arm screen must refuse there
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:204:case_run "an arm ckpt with no launch manifest is refused" 2 "launch manifest missing" \
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:206:# ...and with a manifest whose config hash is another arm's, the lineage gate fires
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:208:write_c8_manifest() {  # $1 = which arm's config hash to record
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:212:    echo "p0_manifest_sha256 aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:215:    echo "save_dir ${OUT_ROOT}/exp11_C8"; } > "${OUT_ROOT}/exp11_C8/launch_manifest.txt"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:216:  register_manifest C8            # audited AS WRITTEN: the field checks are the test
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:218:write_c8_manifest C4L
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:219:case_run "a launch manifest for another config is refused" 2 "ARM LINEAGE GATE" \
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:221:# a correct manifest lets the same screen through
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:222:write_c8_manifest C8
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:304:# The synthetic registry is built from the fixture manifests, so it has no
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:322:$PY - "${OUT_ROOT}/arm_launch_registry.json" "${OUT_ROOT}/exp11_C8" <<'PY'
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:330:r["restarts"] = {"C8": [{"mode": "RESTART", "job": "999", "resume_ckpt_sha256": "b" * 64,
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:334:case_run "a RESTART resuming elsewhere is refused" 2 "no validated RESTART leg" \
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:336:# --- fix 2: the leg is RECORDED by the real recorder, so the registry row and the
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:337:# per-leg producer manifest come from the same audited pipeline the operator uses.
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:346:open(os.path.join(out, "restart_manifest_C8.txt"), "w").write("\n".join([
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:347:    "# exp_11 arm launch manifest",
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:348:    "job 3662829 host synthetic mode RESTART launch_uuid leg-uuid-c8",
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:351:    "p0_manifest_sha256 " + "a" * 64,
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:358:r = json.load(open(os.path.join(out, "arm_launch_registry.json")))
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:360:json.dump(r, open(os.path.join(out, "arm_launch_registry.json"), "w"), indent=2)
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:363:  $PY "$EXPDIR/fa_orbit_record_restart.py" C8 "${OUT_ROOT}/restart_manifest_C8.txt" \
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:364:    --registry "${OUT_ROOT}/arm_launch_registry.json" \
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:368:expect_cmd "the recorder publishes the leg and its producer manifest" 0 "checkpoint(s) added" \
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:388:# a registry row without its producer manifest is no longer evidence
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:390:case_run "a leg with no producer manifest admits nothing" 2 "producer manifest" \
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:394:tamper_leg() { $PY - "${OUT_ROOT}/arm_launch_registry.json" "$1" "$2" <<'PY'
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:401:cp "${OUT_ROOT}/arm_launch_registry.json" "${TMP}/reg_recorded.json"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:403:case_run "a leg whose save_dir is not the audited one is refused" 2 "no validated RESTART leg" \
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:405:cp "${TMP}/reg_recorded.json" "${OUT_ROOT}/arm_launch_registry.json"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:407:case_run "a leg that did not resume the audited final step is refused" 2 "no validated RESTART leg" \
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:409:cp "${TMP}/reg_recorded.json" "${OUT_ROOT}/arm_launch_registry.json"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:410:# the leg's OWN restart manifest is mutable evidence and is re-hashed too
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:411:cp "${OUT_ROOT}/restart_manifest_C8.txt" "${TMP}/restart_manifest_C8.txt"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:412:echo "tampered_field yes" >> "${OUT_ROOT}/restart_manifest_C8.txt"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:413:case_run "a RESTART manifest edited after recording is refused" 2 "changed after it was recorded" \
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:415:cp "${TMP}/restart_manifest_C8.txt" "${OUT_ROOT}/restart_manifest_C8.txt"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:446:# The backfill checkpoint is bound to the AUDITED manifest (path + sha256), so a
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:499:# the audited manifest now registers the legacy D2 endpoint
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:502:m=json.load(open('${EXPDIR}/c4_backfill_manifest.json'))
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:509:  echo "PASS  the audited backfill manifest registers the 40k D2 endpoint"; PASS=$((PASS + 1))
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:515:m=json.load(open('${EXPDIR}/c4_backfill_manifest.json'))
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:522:  echo "PASS  the audited backfill manifest is well-formed (20k/30k, seed 42, live paths, config hash)"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:525:  echo "FAIL  the audited backfill manifest is malformed or its files are missing"; FAIL=$((FAIL + 1))
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:1044:  # the registry must carry VANL, recorded from the PUBLISHED manifest
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:1047:r=json.load(open('${EXPDIR}/arm_launch_registry.json'))['arms']
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:1051:assert len(v['manifest_sha256'])==64 and v['save_dir']=='outputs_FLAC/exp11_VANL'
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:1054:    echo "PASS  the registry carries VANL from its published launch manifest"; PASS=$((PASS + 1))
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:1056:    echo "FAIL  the VANL registry entry is missing or wrong"; FAIL=$((FAIL + 1))
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:1084:      echo "PASS  the intent manifest records the pin"; PASS=$((PASS + 1))
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:1086:      echo "FAIL  the intent manifest does not record the pin"; FAIL=$((FAIL + 1))
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:1431:echo "--- arm launch registry binding ---"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:1432:REG="${EXPDIR}/arm_launch_registry.json"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:1434:  echo "PASS  the audited arm launch registry is committed"; PASS=$((PASS + 1))
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:1441:    for f in ("manifest_sha256", "job", "mode", "launch_uuid", "commit", "rung",
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:1442:              "max_steps", "config_sha256", "vae_sha256", "p0_manifest_sha256", "save_dir"):
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:1445:    assert len(v["manifest_sha256"]) == 64
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:1449:  else echo "FAIL  the registry is incomplete"; FAIL=$((FAIL + 1))
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:1451:  # a TAMPERED manifest must be caught: same fields, different bytes
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:1452:  # tamper with the (synthetic) manifest AFTER it was registered: same fields,
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:1454:  MAN="${OUT_ROOT}/exp11_C8/launch_manifest.txt"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:1455:  cp "$MAN" "${TMP}/manifest.bak"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:1459:    echo "PASS  a launch manifest edited after registration is rejected"; PASS=$((PASS + 1))
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:1461:    echo "FAIL  a tampered launch manifest passed the gate (rc=${rc})"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:1464:  cp "${TMP}/manifest.bak" "$MAN"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:1465:  # ...and a RESTART launch (mode != INITIAL) is not a registered launch
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:1466:  sed -i 's/mode INITIAL/mode RESTART/' "$MAN"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:1467:  $PY - "$MAN" "${OUT_ROOT}/arm_launch_registry.json" <<'PY'
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:1470:reg["arms"]["C8"]["manifest_sha256"] = hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest()
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:1475:    echo "PASS  a RESTART launch is refused as a screen lineage"; PASS=$((PASS + 1))
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:1479:  cp "${TMP}/manifest.bak" "$MAN"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:1480:  grep -q 'reg\["manifest_sha256"\]' "$SCREEN" \
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:1481:    && { echo "PASS  the screen binds the manifest by sha256, not by content alone"; PASS=$((PASS + 1)); } \
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:1482:    || { echo "FAIL  the screen still trusts the mutable manifest"; FAIL=$((FAIL + 1)); }
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:1488:    || { echo "PASS  the seed claim is verified against the registry, not printed"; PASS=$((PASS + 1)); }
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:1490:  echo "FAIL  no arm launch registry"; FAIL=$((FAIL + 1))
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:1550:  echo "PASS  the untrack outcome is recorded in the launch manifest"; PASS=$((PASS + 1))
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:1552:  echo "FAIL  the manifest does not record the untrack state"; FAIL=$((FAIL + 1))
worklog/worklog_yixun/exp_11_fa_orbit_claude/p0_manifest_9bf1936-1786033425104073952-d8d84328.txt:1:# exp_11 P0 submission manifest (consumed by p0_collect.py --manifest)
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_equivprobe_wrapper_test.sh:154:grep -q 'fa_orbit_wandb_readback.py' "$LAUNCHER" \
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_equivprobe_wrapper_test.sh:155:  && { echo "PASS  the launcher uses the id-based wandb readback"; PASS=$((PASS + 1)); } \
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-06_20-55-50_C16_8x8_jid3648696_manifest.txt:1:# exp_11 arm launch manifest
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-06_20-55-50_C16_8x8_jid3648696_manifest.txt:6:p0_manifest_sha256 72607b922177208d56055d604b292d697b643ef3b7ab48261ab2e23a0cc2b53b
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-06_20-55-50_C16_8x8_jid3648696_manifest.txt:18:wandb_entity yh4742-princeton-university wandb_project FLAC_exp11_C16 wandb_name exp11_C16
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-06_20-55-50_C16_8x8_jid3648696_manifest.txt:19:wandb_run_id exp11-C16-1786064168022803862-f44c29b2
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-06_20-55-50_C16_8x8_jid3648696_manifest.txt:20:command torchrun --standalone --nnodes=1 --nproc_per_node=8 train.py --model-config /n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/FLAC_AR_BF_C16.json --dataset-config src/configs/dataset_configs/AR/train/acousticroom_train.json --pretransform-ckpt-path weights/FLAC/VAE.safetensors --max-steps 40000 --batch-size 8 --accum-batches 1 --num-workers 6 --seed 42 --num-gpus 8 --num-nodes 1 --strategy ddp_find_unused_parameters_true --sync-batchnorm true --precision bf16-mixed --val-every -1 --val-dataset-config  --gradient-clip-val 0.0 --logger wandb --checkpoint-every 2500 --name FLAC_exp11_C16 --experiment-name exp11_C16 --save-dir outputs_FLAC/exp11_C16
worklog/worklog_yixun/exp_11_fa_orbit_claude/arm_launch_registry.json:3:    "AUDITED exp_11 arm launch registry (final GO-check item 4 / review (b)).",
worklog/worklog_yixun/exp_11_fa_orbit_claude/arm_launch_registry.json:4:    "The launch manifests live under gitignored outputs_FLAC and are therefore",
worklog/worklog_yixun/exp_11_fa_orbit_claude/arm_launch_registry.json:5:    "MUTABLE evidence: binding a screen to 'whatever the manifest says now' proves",
worklog/worklog_yixun/exp_11_fa_orbit_claude/arm_launch_registry.json:6:    "nothing. This committed registry pins each arm's manifest by sha256 plus the",
worklog/worklog_yixun/exp_11_fa_orbit_claude/arm_launch_registry.json:9:    "VANL (Q9, job 3661520) recorded from its PUBLISHED launch manifest after the",
worklog/worklog_yixun/exp_11_fa_orbit_claude/arm_launch_registry.json:11:    "Recorded from the live manifests of the running arms (3648694-97, 3661520).",
worklog/worklog_yixun/exp_11_fa_orbit_claude/arm_launch_registry.json:12:    "RESTART legs (Q10, 40k -> 100k) are recorded under 'restarts' as a CHAIN:",
worklog/worklog_yixun/exp_11_fa_orbit_claude/arm_launch_registry.json:17:    "Populate with fa_orbit_record_restart.py once a leg's manifest publishes.",
worklog/worklog_yixun/exp_11_fa_orbit_claude/arm_launch_registry.json:20:    "'producer_manifest' -- fa_orbit_producer_<ARM>_job<JOB>.json, append-only,",
worklog/worklog_yixun/exp_11_fa_orbit_claude/arm_launch_registry.json:30:      "manifest_path": "outputs_FLAC/exp11_C4L/launch_manifest.txt",
worklog/worklog_yixun/exp_11_fa_orbit_claude/arm_launch_registry.json:31:      "manifest_sha256": "d49df42d2f7f9c3f39f1aeb6631da84ef0e0a392c22a8271edadbd83885e814a",
worklog/worklog_yixun/exp_11_fa_orbit_claude/arm_launch_registry.json:42:      "p0_manifest_sha256": "72607b922177208d56055d604b292d697b643ef3b7ab48261ab2e23a0cc2b53b",
worklog/worklog_yixun/exp_11_fa_orbit_claude/arm_launch_registry.json:49:      "manifest_path": "outputs_FLAC/exp11_C8/launch_manifest.txt",
worklog/worklog_yixun/exp_11_fa_orbit_claude/arm_launch_registry.json:50:      "manifest_sha256": "fa1037c300fa3f1100667634864653690049271bd4e2815e419fb205c9068388",
worklog/worklog_yixun/exp_11_fa_orbit_claude/arm_launch_registry.json:61:      "p0_manifest_sha256": "72607b922177208d56055d604b292d697b643ef3b7ab48261ab2e23a0cc2b53b",
worklog/worklog_yixun/exp_11_fa_orbit_claude/arm_launch_registry.json:68:      "manifest_path": "outputs_FLAC/exp11_C16/launch_manifest.txt",
worklog/worklog_yixun/exp_11_fa_orbit_claude/arm_launch_registry.json:69:      "manifest_sha256": "deb07b532fea037d9354b5c635e9ad6a80ad8c022dabdc6dbe0a879a27be3979",
worklog/worklog_yixun/exp_11_fa_orbit_claude/arm_launch_registry.json:80:      "p0_manifest_sha256": "72607b922177208d56055d604b292d697b643ef3b7ab48261ab2e23a0cc2b53b",
worklog/worklog_yixun/exp_11_fa_orbit_claude/arm_launch_registry.json:87:      "manifest_path": "outputs_FLAC/exp11_C32/launch_manifest.txt",
worklog/worklog_yixun/exp_11_fa_orbit_claude/arm_launch_registry.json:88:      "manifest_sha256": "b2d08bc0f27583bd78845e281380906b7f05a737444525f4e32cafa5106b395e",
worklog/worklog_yixun/exp_11_fa_orbit_claude/arm_launch_registry.json:99:      "p0_manifest_sha256": "72607b922177208d56055d604b292d697b643ef3b7ab48261ab2e23a0cc2b53b",
worklog/worklog_yixun/exp_11_fa_orbit_claude/arm_launch_registry.json:108:      "manifest_path": "outputs_FLAC/exp11_VANL/launch_manifest.txt",
worklog/worklog_yixun/exp_11_fa_orbit_claude/arm_launch_registry.json:109:      "manifest_sha256": "113d06a284c6198cf9487e99a2efb7ccde94ae13e656a403fe2af0281d3de8b1",
worklog/worklog_yixun/exp_11_fa_orbit_claude/arm_launch_registry.json:120:      "p0_manifest_sha256": "72607b922177208d56055d604b292d697b643ef3b7ab48261ab2e23a0cc2b53b",
worklog/worklog_yixun/exp_11_fa_orbit_claude/c4_backfill_manifest.json:3:    "AUDITED exp_07 B-F backfill manifest (exp_11 plan \u00a73/\u00a75, round-4 review B6).",
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C4L_1786050012428592254-16b4b108.txt:7:pins rung=8x8 maxsteps=40000 ckpt_every=2500 min_free_mb=36500 p0_manifest_sha256=72607b922177208d56055d604b292d697b643ef3b7ab48261ab2e23a0cc2b53b
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-06_16-47-14_C4L_8x8_jid3646734_manifest.txt:1:# exp_11 arm launch manifest
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-06_16-47-14_C4L_8x8_jid3646734_manifest.txt:6:p0_manifest_sha256 72607b922177208d56055d604b292d697b643ef3b7ab48261ab2e23a0cc2b53b
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-06_16-47-14_C4L_8x8_jid3646734_manifest.txt:18:wandb_entity yh4742-princeton-university wandb_project FLAC_exp11_smoke_C4L wandb_name exp11_smoke_C4L
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-06_16-47-14_C4L_8x8_jid3646734_manifest.txt:19:wandb_run_id exp11-C4L-1786049318048844980-bd40da20
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-06_16-47-14_C4L_8x8_jid3646734_manifest.txt:20:command torchrun --standalone --nnodes=1 --nproc_per_node=8 train.py --model-config /n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/FLAC_AR_BF_C4L.json --dataset-config src/configs/dataset_configs/AR/train/acousticroom_train.json --pretransform-ckpt-path weights/FLAC/VAE.safetensors --max-steps 30 --batch-size 8 --accum-batches 1 --num-workers 6 --seed 42 --num-gpus 8 --num-nodes 1 --strategy ddp_find_unused_parameters_true --sync-batchnorm true --precision bf16-mixed --val-every -1 --val-dataset-config  --gradient-clip-val 0.0 --logger wandb --checkpoint-every 10 --name FLAC_exp11_smoke_C4L --experiment-name exp11_smoke_C4L --save-dir outputs_FLAC/exp11_smoke/C4L
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C4L_1786465302140788027-ae06a985.txt:7:pins rung=8x8 maxsteps=100000 ckpt_every=2500 min_free_mb=36500 p0_manifest_sha256=72607b922177208d56055d604b292d697b643ef3b7ab48261ab2e23a0cc2b53b
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C4L_1786465302140788027-ae06a985.txt:9:sbatch sbatch --job-name=exp11-C4L-train --gres=gpu:l40:8 --cpus-per-task=64 --mem=108G --time=34:00:00 --export=ALL,ARM=C4L,EXPECT_SHA=0f0acb2e87debd872f14de91613a89e5760908df,OUTPUT_ROOT=outputs_FLAC,RESUME_CKPT=outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=8-step=40000.ckpt,EXPECTED_STEP=40000 worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C8_1786465302361183738-e9235c57.txt:7:pins rung=8x8 maxsteps=100000 ckpt_every=2500 min_free_mb=36500 p0_manifest_sha256=72607b922177208d56055d604b292d697b643ef3b7ab48261ab2e23a0cc2b53b
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C8_1786465302361183738-e9235c57.txt:9:sbatch sbatch --job-name=exp11-C8-train --gres=gpu:l40:8 --cpus-per-task=64 --mem=108G --time=51:00:00 --export=ALL,ARM=C8,EXPECT_SHA=0f0acb2e87debd872f14de91613a89e5760908df,OUTPUT_ROOT=outputs_FLAC,RESUME_CKPT=outputs_FLAC/exp11_C8/FLAC_exp11_C8/exp11_C8/checkpoints/epoch=8-step=40000.ckpt,EXPECTED_STEP=40000 worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh:120:case_run "restart w/o ckpt" 2 "RESTART requires RESUME_CKPT" \
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh:121:  -- "${SMOKE_ENV[@]}" ARM=C8 EXPECTED_STEP=5000
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh:123:  -- "${SMOKE_ENV[@]}" ARM=C8 EXPECTED_STEP=5000 "RESUME_CKPT=${TMP}/nope.ckpt"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh:125:  -- "${SMOKE_ENV[@]}" ARM=C8 EXPECTED_STEP=5000 "RESUME_CKPT=${TMP}/foreign.ckpt"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh:132:  -- "${SMOKE_ENV[@]}" ARM=C8 EXPECTED_STEP=5000 SMOKE_MAXSTEPS=6000 \
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh:135:  -- "${SMOKE_ENV[@]}" ARM=C8 EXPECTED_STEP=5000 SMOKE_MAXSTEPS=6000 "RESUME_CKPT=${SMOKE_RUN}/notes.txt"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh:137:  -- "${SMOKE_ENV[@]}" ARM=C8 EXPECTED_STEP=5000 SMOKE_MAXSTEPS=30 \
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh:282:# manifest binding: same rung passes, changed rung fails
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh:283:cat > "${TMP}/launch_manifest.txt" <<EOF
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh:284:# exp_11 arm launch manifest
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh:287:wandb_run_id exp11-C8-test
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh:289:expect_cmd "preflight binds to the launch manifest" 0 "bound to launch manifest" -- \
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh:290:  "${PRE[@]}" --ckpt "${TMP}/good.ckpt" --expected-step 5000 --commit "$HEAD_SHA" --launch-manifest "${TMP}/launch_manifest.txt"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh:291:expect_cmd "preflight rejects a rung change" 2 "manifest rung" -- \
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh:293:     --ckpt "${TMP}/good.ckpt" --expected-step 5000 --launch-manifest "${TMP}/launch_manifest.txt"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh:294:# B2 residual: a manifest with no commit, or a different commit, must fail CLOSED
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh:295:grep -v '^commit ' "${TMP}/launch_manifest.txt" > "${TMP}/manifest_nocommit.txt"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh:296:expect_cmd "preflight rejects a manifest without a commit" 2 "no 'commit' line" -- \
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh:298:     --launch-manifest "${TMP}/manifest_nocommit.txt"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh:299:sed 's/^commit .*/commit 0000000000000000000000000000000000000000/' "${TMP}/launch_manifest.txt" > "${TMP}/manifest_othercommit.txt"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh:302:     --launch-manifest "${TMP}/manifest_othercommit.txt"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh:305:     --launch-manifest "${TMP}/launch_manifest.txt"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh:307:echo "--- G2. Q10: the JOB selects and enforces the RESTART time pin (re-pin fix 1) ---"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh:315:case_run "a RESTART leg selects the RESTART pin" 0 "time pin PINNED_TIME_LIMIT_RESTART_C8=51:00:00" \
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh:316:  -- "${Q10_ENV[@]}" ARM=C8 EXPECTED_STEP=40000 \
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh:326:SUB_RESTART="$(env DRYRUN=1 bash "$SUBMITTER" C16 --resume "${Q10_RUN}/checkpoints/epoch=8-step=40000.ckpt" --expected-step 40000 2>&1)"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh:327:if echo "$SUB_RESTART" | grep -q "time 89:00:00"; then
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh:328:  echo "PASS  submitter and job agree on the C16 RESTART pin"; PASS=$((PASS+1))
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh:330:  echo "FAIL  the submitter no longer allocates the C16 RESTART pin"; FAIL=$((FAIL+1))
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh:334:# The ordinary restart contract requires manifest max_steps == this run's budget
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh:335:# and manifest commit == the running commit. An extension violates both BY
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh:337:# (audited manifest bytes, job/uuid/commit/config/save-dir/seed, and the resumed
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh:356:man = os.path.join(tmp, "ext_launch_manifest.txt")
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh:358:    fh.write("# exp_11 arm launch manifest\n")
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh:366:    fh.write("wandb_run_id exp11-C8-ext\n")
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh:368:    "manifest_path": man,
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh:369:    "manifest_sha256": hashlib.sha256(open(man, "rb").read()).hexdigest(),
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh:375:json.dump(reg, open(os.path.join(tmp, "ext_registry.json"), "w"), indent=2)
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh:381:     --launch-manifest "${TMP}/ext_launch_manifest.txt" --extension
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh:382:     --launch-registry "${TMP}/ext_registry.json")
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh:383:expect_cmd "the ORDINARY contract refuses the extension (the bug)" 2 "manifest max_steps" -- \
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh:386:     --launch-manifest "${TMP}/ext_launch_manifest.txt"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh:393:     --launch-manifest "${TMP}/ext_launch_manifest.txt" --extension \
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh:394:     --launch-registry "${TMP}/ext_registry.json"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh:395:$PY - "${TMP}/ext_registry.json" "${TMP}/reg_noanchor.json" "${TMP}/reg_wronganchor.json" \
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh:408:  --launch-manifest "${TMP}/ext_launch_manifest.txt" --extension --launch-registry "$1"; }
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh:412:# SAME registry that just refused is anchored and then accepted. This is C32's
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh:414:add_anchor() { $PY "${EXPDIR}/fa_orbit_add_anchor.py" C8 --registry "$1" \
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh:423:expect_cmd "add_anchor refuses a manifest that disagrees with the registry" 2 "!= the registered" -- \
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh:427:expect_cmd "extension refuses a manifest commit that is not the registered one" 2 "registered launch commit" -- \
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh:429:printf 'tamper\n' >> "${TMP}/ext_launch_manifest.txt"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh:430:expect_cmd "extension refuses a manifest that drifted after registration" 2 "changed after it was registered" -- \
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh:489:expect_cmd "dry run publishes no submission manifest" 0 "DRYRUN sbatch" -- \
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh:492:[ "$INTENT_BEFORE" = "$INTENT_AFTER" ] && { echo "PASS  a dry run leaves no submission manifest behind"; PASS=$((PASS+1)); } \
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh:493:  || { echo "FAIL  a dry run created a submission manifest"; FAIL=$((FAIL+1)); }
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh:495:  && { echo "PASS  intent manifest is published before the sbatch call"; PASS=$((PASS+1)); } \
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh:496:  || { echo "FAIL  the manifest is still published after sbatch"; FAIL=$((FAIL+1)); }
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh:509:grep -q 'WANDB_ENTITY="\$WANDB_ENTITY_SEEN"' "$LAUNCHER" && { echo "PASS  the approved wandb entity is exported"; PASS=$((PASS+1)); } \
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh:511:# RETIRED: the readback moved out of the launcher into fa_orbit_wandb_readback.py
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh:513:# launcher for wandb-metadata.json tested the superseded shape. Assert the real
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh:515:if grep -q 'fa_orbit_wandb_readback.py' "$LAUNCHER" \
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh:518:  echo "PASS  the launcher runs the wandb readback and gates on its result"; PASS=$((PASS+1))
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh:520:  echo "FAIL  no post-run wandb identity verification"; FAIL=$((FAIL+1))
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submit.sh:17:# in an atomic, no-clobber manifest next to the launcher.
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submit.sh:35:RESUME_CKPT=""; EXPECTED_STEP=0
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submit.sh:39:    --expected-step) EXPECTED_STEP="${2:?--expected-step needs a number}"; shift 2 ;;
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submit.sh:43:case "$EXPECTED_STEP" in ''|*[!0-9]*) echo "--expected-step must be a non-negative integer - abort"; exit 2;; esac
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submit.sh:59:  # A RESTART leg is a different budget from the INITIAL one: 60k further steps,
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submit.sh:62:  if [ -n "${EXPECTED_STEP:-}" ] && [ "${EXPECTED_STEP:-0}" -gt 0 ]; then
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submit.sh:63:    TIME_LIMIT="$(pin "PINNED_TIME_LIMIT_RESTART_${ARM}")"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submit.sh:96:[ -n "$RESUME_CKPT" ] && ARGS[5]="${ARGS[5]},RESUME_CKPT=${RESUME_CKPT},EXPECTED_STEP=${EXPECTED_STEP}"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submit.sh:107:# write failure leaves a queued job nobody recorded. The intent manifest carries
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submit.sh:112:[ ! -e "$MANIFEST" ] || { echo "submission manifest ${MANIFEST} already exists - abort"; exit 2; }
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submit.sh:121:  echo "pins rung=${RUNG} maxsteps=$(pin PINNED_MAXSTEPS) ckpt_every=$(pin PINNED_CHECKPOINT_EVERY) min_free_mb=$(pin PINNED_MIN_FREE_MB) p0_manifest_sha256=$(pin PINNED_P0_MANIFEST_SHA256)"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submit.sh:122:  echo "resume ${RESUME_CKPT:-<none>} expected_step ${EXPECTED_STEP}"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submit.sh:124:} >> "$TMP" || { echo "intent manifest write failed - abort"; exit 3; }
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submit.sh:125:mv -n "$TMP" "$MANIFEST" || { echo "intent manifest publication failed - abort"; exit 2; }
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submit.sh:126:[ -e "$MANIFEST" ] || { echo "intent manifest ${MANIFEST} did not appear - abort"; exit 2; }
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submit.sh:127:echo "intent manifest: ${MANIFEST}"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C4L_1786054560338820300-09f373e3.txt:7:pins rung=8x8 maxsteps=40000 ckpt_every=2500 min_free_mb=36500 p0_manifest_sha256=72607b922177208d56055d604b292d697b643ef3b7ab48261ab2e23a0cc2b53b
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C8_1786310422260085470-2e58ce21.txt:7:pins rung=8x8 maxsteps=100000 ckpt_every=2500 min_free_mb=36500 p0_manifest_sha256=72607b922177208d56055d604b292d697b643ef3b7ab48261ab2e23a0cc2b53b
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C8_1786310422260085470-2e58ce21.txt:9:sbatch sbatch --job-name=exp11-C8-train --gres=gpu:l40:8 --cpus-per-task=64 --mem=108G --time=51:00:00 --export=ALL,ARM=C8,EXPECT_SHA=c85bc612c4f431cc8f55e937907e36d846cd7085,OUTPUT_ROOT=outputs_FLAC,RESUME_CKPT=outputs_FLAC/exp11_C8/FLAC_exp11_C8/exp11_C8/checkpoints/epoch=8-step=40000.ckpt,EXPECTED_STEP=40000 worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch
worklog/worklog_yixun/exp_11_fa_orbit_claude/p0_manifest_86a752b-1785980874148140138-06d348d6.txt:1:# exp_11 P0 submission manifest (consumed by p0_collect.py --manifest)
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C16_1786476227007058126-933e8fc4.txt:7:pins rung=8x8 maxsteps=100000 ckpt_every=2500 min_free_mb=36500 p0_manifest_sha256=72607b922177208d56055d604b292d697b643ef3b7ab48261ab2e23a0cc2b53b
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C16_1786476227007058126-933e8fc4.txt:9:sbatch sbatch --job-name=exp11-C16-train --gres=gpu:l40:8 --cpus-per-task=64 --mem=108G --time=89:00:00 --export=ALL,ARM=C16,EXPECT_SHA=da7ee7f3556aee541344dea6ce76479b4495e529,OUTPUT_ROOT=outputs_FLAC,RESUME_CKPT=outputs_FLAC/exp11_C16/FLAC_exp11_C16/exp11_C16/checkpoints/epoch=8-step=40000.ckpt,EXPECTED_STEP=40000 worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch
worklog/worklog_yixun/exp_11_fa_orbit_claude/p0_manifest_bd96575-1786045321895684456-ae4c2f92.txt:1:# exp_11 P0 submission manifest (consumed by p0_collect.py --manifest)
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C8_1786531886816349959-e2b269ee.txt:7:pins rung=8x8 maxsteps=100000 ckpt_every=2500 min_free_mb=36500 p0_manifest_sha256=72607b922177208d56055d604b292d697b643ef3b7ab48261ab2e23a0cc2b53b
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C8_1786531886816349959-e2b269ee.txt:9:sbatch sbatch --job-name=exp11-C8-train --gres=gpu:l40:8 --cpus-per-task=64 --mem=108G --time=51:00:00 --export=ALL,ARM=C8,EXPECT_SHA=2b75036651c1d23a095a32d48117747c633e6008,OUTPUT_ROOT=outputs_FLAC,RESUME_CKPT=outputs_FLAC/exp11_C8/FLAC_exp11_C8/exp11_C8/checkpoints/epoch=8-step=40000.ckpt,EXPECTED_STEP=40000 worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch
worklog/worklog_yixun/exp_11_fa_orbit_claude/p0_manifest_72a8114-1785969226421855487-c8d5b51f.txt:1:# exp_11 P0 submission manifest (consumed by p0_collect.py --manifest)
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C4L_1786056694890618883-d777d6ef.txt:7:pins rung=8x8 maxsteps=40000 ckpt_every=2500 min_free_mb=36500 p0_manifest_sha256=72607b922177208d56055d604b292d697b643ef3b7ab48261ab2e23a0cc2b53b
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C4L_1786310422143759413-7d512809.txt:7:pins rung=8x8 maxsteps=100000 ckpt_every=2500 min_free_mb=36500 p0_manifest_sha256=72607b922177208d56055d604b292d697b643ef3b7ab48261ab2e23a0cc2b53b
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C4L_1786310422143759413-7d512809.txt:9:sbatch sbatch --job-name=exp11-C4L-train --gres=gpu:l40:8 --cpus-per-task=64 --mem=108G --time=34:00:00 --export=ALL,ARM=C4L,EXPECT_SHA=c85bc612c4f431cc8f55e937907e36d846cd7085,OUTPUT_ROOT=outputs_FLAC,RESUME_CKPT=outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=8-step=40000.ckpt,EXPECTED_STEP=40000 worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C4L_1786531886688280327-a2124841.txt:7:pins rung=8x8 maxsteps=100000 ckpt_every=2500 min_free_mb=36500 p0_manifest_sha256=72607b922177208d56055d604b292d697b643ef3b7ab48261ab2e23a0cc2b53b
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C4L_1786531886688280327-a2124841.txt:9:sbatch sbatch --job-name=exp11-C4L-train --gres=gpu:l40:8 --cpus-per-task=64 --mem=108G --time=34:00:00 --export=ALL,ARM=C4L,EXPECT_SHA=2b75036651c1d23a095a32d48117747c633e6008,OUTPUT_ROOT=outputs_FLAC,RESUME_CKPT=outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=8-step=40000.ckpt,EXPECTED_STEP=40000 worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C4L_1786476226703002708-005bbf54.txt:7:pins rung=8x8 maxsteps=100000 ckpt_every=2500 min_free_mb=36500 p0_manifest_sha256=72607b922177208d56055d604b292d697b643ef3b7ab48261ab2e23a0cc2b53b
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C4L_1786476226703002708-005bbf54.txt:9:sbatch sbatch --job-name=exp11-C4L-train --gres=gpu:l40:8 --cpus-per-task=64 --mem=108G --time=34:00:00 --export=ALL,ARM=C4L,EXPECT_SHA=da7ee7f3556aee541344dea6ce76479b4495e529,OUTPUT_ROOT=outputs_FLAC,RESUME_CKPT=outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=8-step=40000.ckpt,EXPECTED_STEP=40000 worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-07_00-16-30_C32_8x8_jid3648697_manifest.txt:1:# exp_11 arm launch manifest
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-07_00-16-30_C32_8x8_jid3648697_manifest.txt:6:p0_manifest_sha256 72607b922177208d56055d604b292d697b643ef3b7ab48261ab2e23a0cc2b53b
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-07_00-16-30_C32_8x8_jid3648697_manifest.txt:18:wandb_entity yh4742-princeton-university wandb_project FLAC_exp11_C32 wandb_name exp11_C32
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-07_00-16-30_C32_8x8_jid3648697_manifest.txt:19:wandb_run_id exp11-C32-1786076295103433762-98dd1f9b
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-07_00-16-30_C32_8x8_jid3648697_manifest.txt:20:command torchrun --standalone --nnodes=1 --nproc_per_node=8 train.py --model-config /n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/FLAC_AR_BF_C32.json --dataset-config src/configs/dataset_configs/AR/train/acousticroom_train.json --pretransform-ckpt-path weights/FLAC/VAE.safetensors --max-steps 40000 --batch-size 8 --accum-batches 1 --num-workers 6 --seed 42 --num-gpus 8 --num-nodes 1 --strategy ddp_find_unused_parameters_true --sync-batchnorm true --precision bf16-mixed --val-every -1 --val-dataset-config  --gradient-clip-val 0.0 --logger wandb --checkpoint-every 2500 --name FLAC_exp11_C32 --experiment-name exp11_C32 --save-dir outputs_FLAC/exp11_C32
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C32_1786531887062468286-c65636ea.txt:7:pins rung=8x8 maxsteps=100000 ckpt_every=2500 min_free_mb=36500 p0_manifest_sha256=72607b922177208d56055d604b292d697b643ef3b7ab48261ab2e23a0cc2b53b
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C32_1786531887062468286-c65636ea.txt:9:sbatch sbatch --job-name=exp11-C32-train --gres=gpu:l40:8 --cpus-per-task=64 --mem=108G --time=160:00:00 --export=ALL,ARM=C32,EXPECT_SHA=2b75036651c1d23a095a32d48117747c633e6008,OUTPUT_ROOT=outputs_FLAC,RESUME_CKPT=outputs_FLAC/exp11_C32/FLAC_exp11_C32/exp11_C32/checkpoints/epoch=8-step=40000.ckpt,EXPECTED_STEP=40000 worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-06_18-00-52_C4L_8x8_jid3648568_manifest.txt:1:# exp_11 arm launch manifest
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-06_18-00-52_C4L_8x8_jid3648568_manifest.txt:6:p0_manifest_sha256 72607b922177208d56055d604b292d697b643ef3b7ab48261ab2e23a0cc2b53b
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-06_18-00-52_C4L_8x8_jid3648568_manifest.txt:18:wandb_entity yh4742-princeton-university wandb_project FLAC_exp11_smoke_C4L wandb_name exp11_smoke_C4L
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-06_18-00-52_C4L_8x8_jid3648568_manifest.txt:19:wandb_run_id exp11-C4L-1786053756799558763-4ae12465
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-06_18-00-52_C4L_8x8_jid3648568_manifest.txt:20:command torchrun --standalone --nnodes=1 --nproc_per_node=8 train.py --model-config /n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/FLAC_AR_BF_C4L.json --dataset-config src/configs/dataset_configs/AR/train/acousticroom_train.json --pretransform-ckpt-path weights/FLAC/VAE.safetensors --max-steps 30 --batch-size 8 --accum-batches 1 --num-workers 6 --seed 42 --num-gpus 8 --num-nodes 1 --strategy ddp_find_unused_parameters_true --sync-batchnorm true --precision bf16-mixed --val-every -1 --val-dataset-config  --gradient-clip-val 0.0 --logger wandb --checkpoint-every 10 --name FLAC_exp11_smoke_C4L --experiment-name exp11_smoke_C4L --save-dir outputs_FLAC/exp11_smoke/C4L
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_add_anchor.py:2:"""Record an arm's AUDITED final checkpoint (the anchor) in arm_launch_registry.json.
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_add_anchor.py:11:written with the same rigor as a leg (fa_orbit_record_restart.py), never by hand:
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_add_anchor.py:13:  * the arm's INITIAL launch manifest must still hash to the value the registry
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_add_anchor.py:14:    audited, and every identity field in it must equal the registry row -- so the
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_add_anchor.py:16:    the mutable manifest under gitignored outputs_FLAC says now;
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_add_anchor.py:17:  * the checkpoint is located by the registry's own save-dir (canonical run
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_add_anchor.py:39:import fa_orbit_producer_manifest as pm                   # noqa: E402
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_add_anchor.py:41:from fa_orbit_record_restart import kvs, parse_manifest, read_pins, resolve   # noqa: E402
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_add_anchor.py:45:    """The INITIAL manifest, byte-for-byte as audited, field for field as recorded."""
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_add_anchor.py:47:    man_path = resolve(repo_root, str(row.get("manifest_path", "")))
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_add_anchor.py:48:    if not row.get("manifest_path") or not os.path.isfile(man_path):
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_add_anchor.py:49:        return [f"the registered INITIAL launch manifest {man_path} does not exist — the anchor "
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_add_anchor.py:51:    raw, man = parse_manifest(man_path)
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_add_anchor.py:53:    if got != row.get("manifest_sha256"):
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_add_anchor.py:54:        problems.append(f"launch manifest sha256 {got[:12]} != audited "
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_add_anchor.py:55:                        f"{str(row.get('manifest_sha256'))[:12]} — it changed after registration")
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_add_anchor.py:66:                        ("p0_manifest_sha256", man.get("p0_manifest_sha256"),
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_add_anchor.py:67:                         row.get("p0_manifest_sha256")),
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_add_anchor.py:70:            problems.append(f"manifest {label} {a!r} != the registered {b!r}")
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_add_anchor.py:72:        problems.append(f"manifest rung {ak.get('rung')!r} != the pinned {pins.get('PINNED_RUNG')!r}")
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_add_anchor.py:74:        problems.append("manifest vae_sha256 != the launcher's PINNED_VAE_SHA256")
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_add_anchor.py:79:        problems.append(f"manifest model_config {cfg!r} does not exist")
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_add_anchor.py:130:    ap.add_argument("--registry", default=os.path.join(HERE, "arm_launch_registry.json"))
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_add_anchor.py:140:    store = os.path.dirname(os.path.abspath(args.registry)) or "."
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_add_anchor.py:151:    reg = json.load(open(args.registry))
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_add_anchor.py:154:        raise SystemExit(f"{arm} has no INITIAL registry entry — an anchor belongs to a launch")
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_add_anchor.py:159:        raise SystemExit(f"{arm}'s registry row has no integer max_steps to anchor at")
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_add_anchor.py:196:        pm.write_atomic(args.registry, reg)
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-12_18-21-40_C16_8x8_jid3684151_manifest.txt:1:# exp_11 arm launch manifest
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-12_18-21-40_C16_8x8_jid3684151_manifest.txt:3:job 3684151 host neu306.neuronic.cs.princeton.edu mode RESTART launch_uuid 52d8c00c-c1bb-41e9-b8e5-4d5ce1f680b9
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-12_18-21-40_C16_8x8_jid3684151_manifest.txt:6:p0_manifest_sha256 72607b922177208d56055d604b292d697b643ef3b7ab48261ab2e23a0cc2b53b
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-12_18-21-40_C16_8x8_jid3684151_manifest.txt:19:wandb_entity yh4742-princeton-university wandb_project FLAC_exp11_C16 wandb_name exp11_C16
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-12_18-21-40_C16_8x8_jid3684151_manifest.txt:20:wandb_run_id exp11-C16-1786064168022803862-f44c29b2
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-12_18-21-40_C16_8x8_jid3684151_manifest.txt:21:command torchrun --standalone --nnodes=1 --nproc_per_node=8 train.py --model-config /n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/FLAC_AR_BF_C16.json --dataset-config src/configs/dataset_configs/AR/train/acousticroom_train.json --pretransform-ckpt-path weights/FLAC/VAE.safetensors --max-steps 100000 --batch-size 8 --accum-batches 1 --num-workers 6 --seed 42 --num-gpus 8 --num-nodes 1 --strategy ddp_find_unused_parameters_true --sync-batchnorm true --precision bf16-mixed --val-every -1 --val-dataset-config  --gradient-clip-val 0.0 --logger wandb --checkpoint-every 2500 --name FLAC_exp11_C16 --experiment-name exp11_C16 --save-dir outputs_FLAC/exp11_C16 --ckpt-path outputs_FLAC/exp11_C16/FLAC_exp11_C16/exp11_C16/checkpoints/epoch=8-step=40000.ckpt
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C32_1786056695251126283-7cb3aeb0.txt:7:pins rung=8x8 maxsteps=40000 ckpt_every=2500 min_free_mb=36500 p0_manifest_sha256=72607b922177208d56055d604b292d697b643ef3b7ab48261ab2e23a0cc2b53b
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C32_1786476227212448813-095d0a36.txt:7:pins rung=8x8 maxsteps=100000 ckpt_every=2500 min_free_mb=36500 p0_manifest_sha256=72607b922177208d56055d604b292d697b643ef3b7ab48261ab2e23a0cc2b53b
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C32_1786476227212448813-095d0a36.txt:9:sbatch sbatch --job-name=exp11-C32-train --gres=gpu:l40:8 --cpus-per-task=64 --mem=108G --time=160:00:00 --export=ALL,ARM=C32,EXPECT_SHA=da7ee7f3556aee541344dea6ce76479b4495e529,OUTPUT_ROOT=outputs_FLAC,RESUME_CKPT=outputs_FLAC/exp11_C32/FLAC_exp11_C32/exp11_C32/checkpoints/epoch=8-step=40000.ckpt,EXPECTED_STEP=40000 worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_equiv_probe.sbatch:117:# --- the probe config must be the committed C32 arm manifest ------------------
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-09_12-06-39_VANL_8x8_jid3661520_manifest.txt:1:# exp_11 arm launch manifest
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-09_12-06-39_VANL_8x8_jid3661520_manifest.txt:6:p0_manifest_sha256 72607b922177208d56055d604b292d697b643ef3b7ab48261ab2e23a0cc2b53b
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-09_12-06-39_VANL_8x8_jid3661520_manifest.txt:19:wandb_entity yh4742-princeton-university wandb_project FLAC_exp11_VANL wandb_name exp11_VANL
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-09_12-06-39_VANL_8x8_jid3661520_manifest.txt:20:wandb_run_id exp11-VANL-1786291671381616649-772b3272
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-09_12-06-39_VANL_8x8_jid3661520_manifest.txt:21:command torchrun --standalone --nnodes=1 --nproc_per_node=8 train.py --model-config /n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/FLAC_AR_VANCKPT.json --dataset-config src/configs/dataset_configs/AR/train/acousticroom_train.json --pretransform-ckpt-path weights/FLAC/VAE.safetensors --max-steps 40000 --batch-size 8 --accum-batches 1 --num-workers 6 --seed 42 --num-gpus 8 --num-nodes 1 --strategy ddp_find_unused_parameters_true --sync-batchnorm true --precision bf16-mixed --val-every -1 --val-dataset-config  --gradient-clip-val 0.0 --logger wandb --checkpoint-every 2500 --name FLAC_exp11_VANL --experiment-name exp11_VANL --save-dir outputs_FLAC/exp11_VANL
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_record_restart.py:2:"""Record a RESTART leg in arm_launch_registry.json from its PUBLISHED manifest.
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_record_restart.py:4:    python3 fa_orbit_record_restart.py C4L outputs_FLAC/exp11_C4L/<manifest>.txt
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_record_restart.py:5:    python3 fa_orbit_record_restart.py C4L <manifest> --extend   # later, as the leg saves more
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_record_restart.py:9:trusted from the manifest -- equals that arm's recorded final_ckpt_sha256.
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_record_restart.py:12:only `if os.path.isfile(resume_path)`, so a manifest naming a file that could not
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_record_restart.py:14:else in the manifest was checked at all. Now:
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_record_restart.py:18:  * every identity field is validated against the INITIAL registry row (arm, job,
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_record_restart.py:19:    uuid, commit, rung, config sha, VAE and P0 manifest shas, save-dir, seed) and
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_record_restart.py:21:    step = the audited final step, and the arm's RESTART wall pin), so recorder
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_record_restart.py:40:import fa_orbit_producer_manifest as pm            # noqa: E402
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_record_restart.py:57:def parse_manifest(path):
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_record_restart.py:74:    """Every field of the RESTART manifest, against the audited INITIAL row + Q10 pins."""
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_record_restart.py:82:    if jk.get("mode") != "RESTART":
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_record_restart.py:83:        problems.append(f"manifest mode is {jk.get('mode')!r}, not RESTART")
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_record_restart.py:87:            problems.append(f"manifest records no {field} — a leg with no identity is not a record")
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_record_restart.py:89:        problems.append(f"manifest job {jk.get('job')} IS the INITIAL job — that is the launch "
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_record_restart.py:97:                             ("p0_manifest_sha256", man.get("p0_manifest_sha256"),
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_record_restart.py:98:                              initial.get("p0_manifest_sha256")),
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_record_restart.py:101:            problems.append(f"manifest {label} {got!r} != the audited INITIAL run's {want!r}")
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_record_restart.py:103:        problems.append(f"manifest rung {ak.get('rung')!r} != the pinned {pins.get('PINNED_RUNG')!r}")
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_record_restart.py:105:        problems.append(f"manifest max_steps {ak.get('max_steps')!r} != the Q10 budget pin "
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_record_restart.py:108:        problems.append(f"manifest expected_step {rk.get('expected_step')!r} != the audited final "
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_record_restart.py:110:    want_time = pins.get(f"PINNED_TIME_LIMIT_RESTART_{arm}")
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_record_restart.py:112:        problems.append(f"manifest time_limit {tk.get('time_limit')!r} != the arm's RESTART wall "
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_record_restart.py:119:        problems.append(f"manifest model_config {cfg_path!r} does not exist")
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_record_restart.py:126:        problems.append("manifest records no resume_ckpt — a RESTART that resumed nothing is not "
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_record_restart.py:130:                        "NOT accept the manifest's claimed hash in its place")
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_record_restart.py:142:            problems.append(f"manifest resume_ckpt_sha256 {str(rk.get('resume_ckpt_sha256'))[:12]} "
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_record_restart.py:152:    ap = argparse.ArgumentParser(description="record an exp_11 RESTART leg")
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_record_restart.py:154:    ap.add_argument("manifest")
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_record_restart.py:155:    ap.add_argument("--registry", default=os.path.join(HERE, "arm_launch_registry.json"))
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_record_restart.py:159:                    help="where the per-leg producer manifests are published")
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_record_restart.py:162:                    help="root the manifest's relative paths resolve against")
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_record_restart.py:164:                    help="this leg is already recorded: extend its producer manifest only")
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_record_restart.py:175:    # One writer at a time, and the lock is the registry's own DIRECTORY: no lock
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_record_restart.py:177:    store = os.path.dirname(os.path.abspath(args.registry)) or "."
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_record_restart.py:187:    reg = json.load(open(args.registry))
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_record_restart.py:190:        raise SystemExit(f"{arm} has no INITIAL registry entry")
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_record_restart.py:192:    raw, man = parse_manifest(args.manifest)
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_record_restart.py:200:            or l.get("manifest_sha256") == man_sha]
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_record_restart.py:203:                         "one leg, one row; use --extend to extend its producer manifest")
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_record_restart.py:205:        problems.append(f"{len(same)} registry rows already claim this leg — the registry is "
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_record_restart.py:216:    producer = pm.manifest_name(arm, job)
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_record_restart.py:218:        "manifest_path": args.manifest, "manifest_sha256": man_sha,
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_record_restart.py:219:        "job": job, "mode": "RESTART", "launch_uuid": jk.get("launch_uuid"),
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_record_restart.py:225:        "producer_manifest": producer, "chains_to": anchor,
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_record_restart.py:228:    header = {"arm": arm, "job": job, "launch_uuid": jk.get("launch_uuid"), "mode": "RESTART",
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_record_restart.py:232:              "chains_to": anchor, "leg_manifest_sha256": man_sha}
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_record_restart.py:250:                legs[i] = {**leg, "producer_manifest": producer}
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_record_restart.py:254:        pm.write_atomic(args.registry, reg)
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_record_restart.py:256:    print(f"{verb} {arm} RESTART job {job} chaining to {anchor[:12]} "
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_record_restart.py:258:    print(f"  producer manifest {producer}: {len(added)} checkpoint(s) added, "
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C16_1786056695148252065-920f395f.txt:7:pins rung=8x8 maxsteps=40000 ckpt_every=2500 min_free_mb=36500 p0_manifest_sha256=72607b922177208d56055d604b292d697b643ef3b7ab48261ab2e23a0cc2b53b
worklog/worklog_yixun/exp_11_fa_orbit_claude/exp11_validate_rows.py:73:EXPECTED_STEPS = 1
worklog/worklog_yixun/exp_11_fa_orbit_claude/exp11_validate_rows.py:110:    ("steps", int, EXPECTED_STEPS),
worklog/worklog_yixun/exp_11_fa_orbit_claude/exp11_validate_rows.py:387:    if side.get("steps") != EXPECTED_STEPS:
worklog/worklog_yixun/exp_11_fa_orbit_claude/exp11_validate_rows.py:388:        problems.append(f"{tag}: steps={side.get('steps')!r} != {EXPECTED_STEPS}")
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-06_20-36-00_C4L_8x8_jid3648694_manifest.txt:1:# exp_11 arm launch manifest
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-06_20-36-00_C4L_8x8_jid3648694_manifest.txt:6:p0_manifest_sha256 72607b922177208d56055d604b292d697b643ef3b7ab48261ab2e23a0cc2b53b
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-06_20-36-00_C4L_8x8_jid3648694_manifest.txt:18:wandb_entity yh4742-princeton-university wandb_project FLAC_exp11_C4L wandb_name exp11_C4L
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-06_20-36-00_C4L_8x8_jid3648694_manifest.txt:19:wandb_run_id exp11-C4L-1786063010468957329-bc46fb0a
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-06_20-36-00_C4L_8x8_jid3648694_manifest.txt:20:command torchrun --standalone --nnodes=1 --nproc_per_node=8 train.py --model-config /n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/FLAC_AR_BF_C4L.json --dataset-config src/configs/dataset_configs/AR/train/acousticroom_train.json --pretransform-ckpt-path weights/FLAC/VAE.safetensors --max-steps 40000 --batch-size 8 --accum-batches 1 --num-workers 6 --seed 42 --num-gpus 8 --num-nodes 1 --strategy ddp_find_unused_parameters_true --sync-batchnorm true --precision bf16-mixed --val-every -1 --val-dataset-config  --gradient-clip-val 0.0 --logger wandb --checkpoint-every 2500 --name FLAC_exp11_C4L --experiment-name exp11_C4L --save-dir outputs_FLAC/exp11_C4L
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_VANL_1786473966640260607-09fab791.txt:7:pins rung=8x8 maxsteps=100000 ckpt_every=2500 min_free_mb=36500 p0_manifest_sha256=72607b922177208d56055d604b292d697b643ef3b7ab48261ab2e23a0cc2b53b
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_VANL_1786473966640260607-09fab791.txt:9:sbatch sbatch --job-name=exp11-VANL-train --gres=gpu:l40:8 --cpus-per-task=64 --mem=108G --time=19:00:00 --export=ALL,ARM=VANL,EXPECT_SHA=135cb4beb3569bb41f325459f0785bf970297de8,OUTPUT_ROOT=outputs_FLAC,RESUME_CKPT=outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=8-step=40000.ckpt,EXPECTED_STEP=40000 worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_producer_manifest.py:2:"""Per-leg PRODUCER manifests: which checkpoints a RESTART leg actually produced.
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_producer_manifest.py:5:registry leg for an arm carried `mode=RESTART` and the right 40k resume hash,
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_producer_manifest.py:19:  file next to the audited registry. The screen re-hashes the checkpoint it is
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_producer_manifest.py:23:directory, exactly like arm_launch_registry.json, and screens read it from the
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_producer_manifest.py:43:                 "chains_to", "leg_manifest_sha256")
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_producer_manifest.py:54:def manifest_name(arm, job):
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_producer_manifest.py:55:    """The per-leg file name. Flat in the experiment directory, like the registry
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_producer_manifest.py:56:    and the backfill manifest, so the launcher/submitter drift gates (`$EXPDIR/*.json`)
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_producer_manifest.py:83:    run and is already anchored in the registry. Steps already published are not
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_producer_manifest.py:123:                                "producer manifest is immutable")
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_producer_manifest.py:143:        "APPEND-ONLY producer manifest for one exp_11 RESTART leg (re-pin fix 2).",
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_producer_manifest.py:145:        "fa_orbit_record_restart.py. The screen admits a >40k checkpoint only when",
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_producer_manifest.py:179:    """Every field of a registry RESTART row, against the audited INITIAL row.
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_producer_manifest.py:185:                             ("mode", leg.get("mode"), "RESTART"),
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_producer_manifest.py:192:    for field in ("job", "launch_uuid", "commit", "manifest_sha256", "producer_manifest"):
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_producer_manifest.py:215:    ``base_dir`` is the directory the registry was read from, so the per-leg files
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_producer_manifest.py:216:    travel with it (pinned worktree, or a synthetic registry under test)."""
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_producer_manifest.py:219:        return [f"{arm} is not in the audited launch registry"], ""
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_producer_manifest.py:225:        return [f"checkpoint at step {step} is above the INITIAL budget but {arm} has no RESTART "
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_producer_manifest.py:226:                "entry in the audited registry — record the leg with fa_orbit_record_restart.py "
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_producer_manifest.py:234:        # The leg's OWN restart manifest is mutable evidence under gitignored
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_producer_manifest.py:237:        leg_man = resolve(repo_root, str(leg.get("manifest_path")))
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_producer_manifest.py:239:            why.append(f"leg {leg.get('job')}: the registered RESTART manifest {leg_man} is gone")
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_producer_manifest.py:242:        if got != leg.get("manifest_sha256"):
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_producer_manifest.py:243:            why.append(f"leg {leg.get('job')}: RESTART manifest {leg_man} now hashes {got[:12]}, "
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_producer_manifest.py:244:                       f"not the registered {str(leg.get('manifest_sha256'))[:12]} — it changed "
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_producer_manifest.py:247:        man_path = resolve(base_dir, str(leg.get("producer_manifest")))
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_producer_manifest.py:250:            why.append(f"leg {leg.get('job')}: producer manifest {man_path} is missing")
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_producer_manifest.py:252:        head_bad = [f"producer manifest {f}={man.get(f)!r} != the registry leg's {leg.get(f)!r}"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_producer_manifest.py:255:        if str(man.get("leg_manifest_sha256")) != str(leg.get("manifest_sha256")):
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_producer_manifest.py:256:            head_bad.append("producer manifest is not the one this registry row published")
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_producer_manifest.py:274:        return [], (f"producer binding OK: step {step} ({ckpt_sha[:12]}) was produced by RESTART "
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_producer_manifest.py:278:    return [f"no validated RESTART leg for {arm} published step {step} with sha256 "
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C32_1786054560670066214-c4a97ed7.txt:7:pins rung=8x8 maxsteps=40000 ckpt_every=2500 min_free_mb=36500 p0_manifest_sha256=72607b922177208d56055d604b292d697b643ef3b7ab48261ab2e23a0cc2b53b
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-12_18-11-38_C4L_8x8_jid3684149_manifest.txt:1:# exp_11 arm launch manifest
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-12_18-11-38_C4L_8x8_jid3684149_manifest.txt:3:job 3684149 host neu306.neuronic.cs.princeton.edu mode RESTART launch_uuid a079ae86-ae39-4b98-abf6-fb2104e2af39
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-12_18-11-38_C4L_8x8_jid3684149_manifest.txt:6:p0_manifest_sha256 72607b922177208d56055d604b292d697b643ef3b7ab48261ab2e23a0cc2b53b
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-12_18-11-38_C4L_8x8_jid3684149_manifest.txt:19:wandb_entity yh4742-princeton-university wandb_project FLAC_exp11_C4L wandb_name exp11_C4L
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-12_18-11-38_C4L_8x8_jid3684149_manifest.txt:20:wandb_run_id exp11-C4L-1786063010468957329-bc46fb0a
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-12_18-11-38_C4L_8x8_jid3684149_manifest.txt:21:command torchrun --standalone --nnodes=1 --nproc_per_node=8 train.py --model-config /n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/FLAC_AR_BF_C4L.json --dataset-config src/configs/dataset_configs/AR/train/acousticroom_train.json --pretransform-ckpt-path weights/FLAC/VAE.safetensors --max-steps 100000 --batch-size 8 --accum-batches 1 --num-workers 6 --seed 42 --num-gpus 8 --num-nodes 1 --strategy ddp_find_unused_parameters_true --sync-batchnorm true --precision bf16-mixed --val-every -1 --val-dataset-config  --gradient-clip-val 0.0 --logger wandb --checkpoint-every 2500 --name FLAC_exp11_C4L --experiment-name exp11_C4L --save-dir outputs_FLAC/exp11_C4L --ckpt-path outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=8-step=40000.ckpt
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_VANL_1786531887194560967-be12b53a.txt:7:pins rung=8x8 maxsteps=100000 ckpt_every=2500 min_free_mb=36500 p0_manifest_sha256=72607b922177208d56055d604b292d697b643ef3b7ab48261ab2e23a0cc2b53b
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_VANL_1786531887194560967-be12b53a.txt:9:sbatch sbatch --job-name=exp11-VANL-train --gres=gpu:l40:8 --cpus-per-task=64 --mem=108G --time=19:00:00 --export=ALL,ARM=VANL,EXPECT_SHA=2b75036651c1d23a095a32d48117747c633e6008,OUTPUT_ROOT=outputs_FLAC,RESUME_CKPT=outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=8-step=40000.ckpt,EXPECTED_STEP=40000 worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch
worklog/worklog_yixun/exp_11_fa_orbit_claude/p0_manifest_1334933-1786032532843128131-8f21c960.txt:1:# exp_11 P0 submission manifest (consumed by p0_collect.py --manifest)
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_wandb_readback.py:5:the run was written and that it carries the manifest's identity.
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_wandb_readback.py:10:``wandb.init``, and that argument OVERRIDES the exported ``WANDB_DIR``. In the
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_wandb_readback.py:12:``$REPO/wandb/run-20260806_164917-exp11-C4L-<run id>`` while the readback looked
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_wandb_readback.py:13:under ``$WANDB_DIR/wandb`` and found nothing — training was green but the job
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_wandb_readback.py:14:classified 7. The launcher still exports ``WANDB_DIR`` (other wandb artifacts do
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_wandb_readback.py:15:respect it), but the readback locates the run by the id WE generated, which wandb
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_wandb_readback.py:28:def locate_run_dir(roots, run_id):
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_wandb_readback.py:29:    """Find ``<root>/wandb/run-*-<run_id>`` across ``roots``.
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_wandb_readback.py:34:    if not run_id:
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_wandb_readback.py:40:        matches.extend(sorted(glob.glob(os.path.join(root, "wandb", f"run-*-{run_id}"))))
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_wandb_readback.py:43:        return None, [f"no run directory for id {run_id} under any of {list(roots)}"]
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_wandb_readback.py:45:        return None, [f"ambiguous run id {run_id}: {matches}"]
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_wandb_readback.py:49:def verify_identity(run_dir, run_id, entity=None, project=None, name=None):
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_wandb_readback.py:50:    """Check the run directory's embedded id and its wandb-metadata identity."""
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_wandb_readback.py:54:    if not os.path.basename(run_dir).endswith(f"-{run_id}"):
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_wandb_readback.py:55:        problems.append(f"run directory {os.path.basename(run_dir)} does not carry id {run_id}")
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_wandb_readback.py:56:    meta_path = os.path.join(run_dir, "files", "wandb-metadata.json")
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_wandb_readback.py:66:        # wandb-metadata does not always carry every field; only a CONTRADICTION
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_wandb_readback.py:69:            problems.append(f"{key}={got!r} != manifest {want!r}")
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_wandb_readback.py:75:    ap.add_argument("--run-id", required=True)
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_wandb_readback.py:77:                    help="candidate root; repeat. Searched as <root>/wandb/run-*-<id>")
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_wandb_readback.py:83:    run_dir, problems = locate_run_dir(args.root, args.run_id)
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_wandb_readback.py:88:    problems = verify_identity(run_dir, args.run_id, args.entity, args.project, args.name)
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_wandb_readback.py:92:    print(f"wandb run identity OK: id {args.run_id} at {run_dir} "
worklog/worklog_yixun/exp_11_fa_orbit_claude/p0_collect.py:2:"""exp_11 P0 profiling collector — manifest-bound ``P0RESULT`` -> ``p0_report_<runid>.md``.
worklog/worklog_yixun/exp_11_fa_orbit_claude/p0_collect.py:14:match the manifest row; the poller CSV must exist, hash-match, and show complete
worklog/worklog_yixun/exp_11_fa_orbit_claude/p0_collect.py:34:The manifest declares a MODE: only ``matrix`` promises an attribution fit;
worklog/worklog_yixun/exp_11_fa_orbit_claude/p0_collect.py:37:    python p0_collect.py --manifest p0_manifest_<runid>.txt [--dir .] [--out ...]
worklog/worklog_yixun/exp_11_fa_orbit_claude/p0_collect.py:123:def parse_manifest(text):
worklog/worklog_yixun/exp_11_fa_orbit_claude/p0_collect.py:124:    """Parse the submitter's manifest.
worklog/worklog_yixun/exp_11_fa_orbit_claude/p0_collect.py:147:                raise ValueError(f"manifest lists {key} more than once")
worklog/worklog_yixun/exp_11_fa_orbit_claude/p0_collect.py:156:            raise ValueError(f"manifest cell row has {len(parts)} fields, expected 9: {line!r}")
worklog/worklog_yixun/exp_11_fa_orbit_claude/p0_collect.py:158:        raise ValueError("manifest must declare both 'runid' and 'sha'")
worklog/worklog_yixun/exp_11_fa_orbit_claude/p0_collect.py:160:        raise ValueError(f"manifest mode {mode!r} must be one of {MODES}")
worklog/worklog_yixun/exp_11_fa_orbit_claude/p0_collect.py:162:        raise ValueError("manifest declares no expected cells")
worklog/worklog_yixun/exp_11_fa_orbit_claude/p0_collect.py:166:def load_manifest(path):
worklog/worklog_yixun/exp_11_fa_orbit_claude/p0_collect.py:168:        return parse_manifest(fh.read())
worklog/worklog_yixun/exp_11_fa_orbit_claude/p0_collect.py:197:# manifest binding
worklog/worklog_yixun/exp_11_fa_orbit_claude/p0_collect.py:199:def admit_rows(rows, manifest):
worklog/worklog_yixun/exp_11_fa_orbit_claude/p0_collect.py:200:    """Keep only rows matching the manifest exactly (identity AND shape).
worklog/worklog_yixun/exp_11_fa_orbit_claude/p0_collect.py:204:    by_key = {e["key"]: e for e in manifest["expected"]}
worklog/worklog_yixun/exp_11_fa_orbit_claude/p0_collect.py:205:    jobids = {e["jobid"] for e in manifest["expected"]}
worklog/worklog_yixun/exp_11_fa_orbit_claude/p0_collect.py:209:        if row["runid"] != manifest["runid"]:
worklog/worklog_yixun/exp_11_fa_orbit_claude/p0_collect.py:210:            rejected.append((cell, f"runid {row['runid']} is not this run ({manifest['runid']})"))
worklog/worklog_yixun/exp_11_fa_orbit_claude/p0_collect.py:212:        if row["sha"] != manifest["sha"]:
worklog/worklog_yixun/exp_11_fa_orbit_claude/p0_collect.py:213:            rejected.append((cell, f"sha {row['sha'][:12]} != manifest sha {manifest['sha'][:12]}"))
worklog/worklog_yixun/exp_11_fa_orbit_claude/p0_collect.py:216:            rejected.append((cell, f"jobid {row['jobid']} is not in the manifest"))
worklog/worklog_yixun/exp_11_fa_orbit_claude/p0_collect.py:220:            other = [e for e in manifest["expected"] if e["cell"] == cell]
worklog/worklog_yixun/exp_11_fa_orbit_claude/p0_collect.py:222:                      if other else f"cell {cell} is not expected by this manifest")
worklog/worklog_yixun/exp_11_fa_orbit_claude/p0_collect.py:234:            rejected.append((cell, "execution shape differs from the manifest: " + ", ".join(
worklog/worklog_yixun/exp_11_fa_orbit_claude/p0_collect.py:269:def summarize(manifest, admitted, malformed=()):
worklog/worklog_yixun/exp_11_fa_orbit_claude/p0_collect.py:276:    for expect in manifest["expected"]:
worklog/worklog_yixun/exp_11_fa_orbit_claude/p0_collect.py:557:def render_markdown(summaries, mode="matrix", complete=False, manifest=None, poller=None,
worklog/worklog_yixun/exp_11_fa_orbit_claude/p0_collect.py:560:    The orbit fit is rendered for ``matrix`` manifests — the only mode that
worklog/worklog_yixun/exp_11_fa_orbit_claude/p0_collect.py:564:    if manifest:
worklog/worklog_yixun/exp_11_fa_orbit_claude/p0_collect.py:565:        lines += [f"Run `{manifest['runid']}` · mode `{manifest['mode']}` · commit "
worklog/worklog_yixun/exp_11_fa_orbit_claude/p0_collect.py:566:                  f"`{manifest['sha'][:12]}` — {len(manifest['expected'])} expected row(s).", ""]
worklog/worklog_yixun/exp_11_fa_orbit_claude/p0_collect.py:615:                  f"A `{mode}` manifest does not carry the FA1+C4L+C8 set; its cells are "
worklog/worklog_yixun/exp_11_fa_orbit_claude/p0_collect.py:712:def _attach_cells(malformed, manifest):
worklog/worklog_yixun/exp_11_fa_orbit_claude/p0_collect.py:716:        cell = next((e["cell"] for e in manifest["expected"] if e["cell"] in where), where)
worklog/worklog_yixun/exp_11_fa_orbit_claude/p0_collect.py:724:    ap.add_argument("--manifest", required=True,
worklog/worklog_yixun/exp_11_fa_orbit_claude/p0_collect.py:725:                    help="submission manifest written by p0_submit_matrix.sh")
worklog/worklog_yixun/exp_11_fa_orbit_claude/p0_collect.py:732:    manifest = load_manifest(args.manifest)
worklog/worklog_yixun/exp_11_fa_orbit_claude/p0_collect.py:734:    admitted, rejected = admit_rows(rows, manifest)
worklog/worklog_yixun/exp_11_fa_orbit_claude/p0_collect.py:735:    summaries = summarize(manifest, admitted, malformed=_attach_cells(malformed, manifest))
worklog/worklog_yixun/exp_11_fa_orbit_claude/p0_collect.py:739:    mode = manifest["mode"]
worklog/worklog_yixun/exp_11_fa_orbit_claude/p0_collect.py:744:    md = render_markdown(summaries, mode=mode, complete=complete, manifest=manifest,
worklog/worklog_yixun/exp_11_fa_orbit_claude/p0_collect.py:746:    out = args.out or os.path.join(args.dir, f"p0_report_{manifest['runid']}.md")
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen.sbatch:360:# subdirectory helpers, manifests and decision-relevant markdown remain covered)
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen.sbatch:383:# lineage the plan requires an audited seed-to-file manifest, so the backfill
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen.sbatch:386:  BACKFILL_MANIFEST="$EXPDIR/c4_backfill_manifest.json"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen.sbatch:387:  [ -f "$BACKFILL_MANIFEST" ] || die "backfill manifest missing: ${BACKFILL_MANIFEST} - abort"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen.sbatch:393:    sys.exit(f"step {step} is not in the audited backfill manifest "
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen.sbatch:404:print(f"backfill lineage OK: step {step} matches the audited manifest "
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen.sbatch:412:# own launch manifest (written by fa_orbit_train.sbatch at INITIAL launch)
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen.sbatch:414:# must sit in exactly that run directory and the config must be that manifest's
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen.sbatch:417:  ARM_LAUNCH_MANIFEST="${OUTPUT_ROOT}/exp11_${ARM}/launch_manifest.txt"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen.sbatch:418:  # The COMMITTED registry is the only one a real run can use: FA_ORBIT_ARM_REGISTRY
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen.sbatch:422:  ARM_REGISTRY="$EXPDIR/arm_launch_registry.json"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen.sbatch:425:      || die "FA_ORBIT_ARM_REGISTRY may not override the audited registry for the production output root - abort"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen.sbatch:428:  [ -f "$ARM_LAUNCH_MANIFEST" ] || die "arm launch manifest missing: ${ARM_LAUNCH_MANIFEST} — a screen may only evaluate a checkpoint from a recorded launch - abort"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen.sbatch:429:  [ -f "$ARM_REGISTRY" ] || die "audited arm launch registry missing: ${ARM_REGISTRY} - abort"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen.sbatch:430:  python3 - "$ARM_LAUNCH_MANIFEST" "$ARM" "$CKPT" "$CONFIG_SHA" "$CKPT_DIR" "$MAIN_REPO" "$ARM_REGISTRY" "$EXPDIR" <<'PY' || die "arm launch-manifest lineage gate FAILED - abort"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen.sbatch:444:# The manifest lives under gitignored outputs_FLAC and is MUTABLE: binding to
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen.sbatch:446:# checkpoint could forge its manifest too. Bind it to the COMMITTED, audited
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen.sbatch:447:# registry — same bytes, and every launch fact re-checked field by field.
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen.sbatch:450:    sys.exit(f"ARM LINEAGE GATE: {arm} is not in the audited launch registry {reg_path}")
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen.sbatch:452:if got_sha != reg["manifest_sha256"]:
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen.sbatch:453:    bad.append(f"launch manifest sha256 {got_sha[:12]} != audited {reg['manifest_sha256'][:12]} — "
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen.sbatch:454:               "the manifest changed after it was registered")
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen.sbatch:463:                             ("p0_manifest_sha256", man.get("p0_manifest_sha256"), reg["p0_manifest_sha256"]),
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen.sbatch:472:    bad.append(f"manifest arm {kv.get('arm')!r} != {arm!r}")
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen.sbatch:474:    bad.append(f"manifest config_sha256 {man.get('config_sha256', '')[:12]} != the config we would "
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen.sbatch:478:    bad.append("manifest records no save_dir")
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen.sbatch:486:        bad.append(f"checkpoint dir {want_dir} is not the manifest's canonical {canon}")
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen.sbatch:489:# --- Q10: a checkpoint ABOVE 40k came from a RESTART leg, not the INITIAL run --
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen.sbatch:490:# The INITIAL manifest cannot vouch for it. The first version of this gate was
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen.sbatch:491:# EXISTENTIAL (re-pin review, finding 2): once ONE leg carried mode=RESTART and
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen.sbatch:496:# step, in the append-only PRODUCER MANIFEST of a leg whose every registry field
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen.sbatch:497:# re-validates against the audited INITIAL row (see fa_orbit_producer_manifest.py).
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen.sbatch:501:    import fa_orbit_producer_manifest as pm
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen.sbatch:516:      f"max_steps {reg['max_steps']}, training seed {reg['training_seed']}); manifest "
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-12_18-19-40_C8_8x8_jid3684150_manifest.txt:1:# exp_11 arm launch manifest
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-12_18-19-40_C8_8x8_jid3684150_manifest.txt:3:job 3684150 host neu306.neuronic.cs.princeton.edu mode RESTART launch_uuid 9e128e69-f5ce-4c7c-9a18-f91326a0c51a
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-12_18-19-40_C8_8x8_jid3684150_manifest.txt:6:p0_manifest_sha256 72607b922177208d56055d604b292d697b643ef3b7ab48261ab2e23a0cc2b53b
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-12_18-19-40_C8_8x8_jid3684150_manifest.txt:19:wandb_entity yh4742-princeton-university wandb_project FLAC_exp11_C8 wandb_name exp11_C8
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-12_18-19-40_C8_8x8_jid3684150_manifest.txt:20:wandb_run_id exp11-C8-1786064131292302937-6d92e299
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-12_18-19-40_C8_8x8_jid3684150_manifest.txt:21:command torchrun --standalone --nnodes=1 --nproc_per_node=8 train.py --model-config /n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/FLAC_AR_BF_C8.json --dataset-config src/configs/dataset_configs/AR/train/acousticroom_train.json --pretransform-ckpt-path weights/FLAC/VAE.safetensors --max-steps 100000 --batch-size 8 --accum-batches 1 --num-workers 6 --seed 42 --num-gpus 8 --num-nodes 1 --strategy ddp_find_unused_parameters_true --sync-batchnorm true --precision bf16-mixed --val-every -1 --val-dataset-config  --gradient-clip-val 0.0 --logger wandb --checkpoint-every 2500 --name FLAC_exp11_C8 --experiment-name exp11_C8 --save-dir outputs_FLAC/exp11_C8 --ckpt-path outputs_FLAC/exp11_C8/FLAC_exp11_C8/exp11_C8/checkpoints/epoch=8-step=40000.ckpt

exec
/usr/bin/bash -lc "nl -ba worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch | sed -n '1,220p;300,460p;500,730p'" in /n/fs/gatrdp/codespace/FLAC
 succeeded in 6143ms:
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
    26	#   SMOKE=1            the reviewed multi-GPU smoke (see SMOKE MODE)
    27	# RUNG / MAXSTEPS / MIN_FREE_MB / time limit are NOT operator inputs any more.
    28	#
    29	# LINEAGE (fail-closed, exactly two stories):
    30	#   INITIAL  no RESUME_CKPT, EXPECTED_STEP unset/0, run directory absent.
    31	#   RESTART  EXPECTED_STEP > 0 AND RESUME_CKPT inside this arm's OWN
    32	#            <RUNDIR>/checkpoints/ AND the checkpoint passes
    33	#            fa_orbit_ckpt_preflight.py (embedded step/config/optimizer/
    34	#            scheduler/EMA + binding to the original launch manifest).
    35	#
    36	# WORLD SIZE: no absence timer (round-3 B4 — a cold start with W&B has no
    37	# measured bound, and `scancel` bypassed classification). Instead: a watcher that
    38	# terminates the torchrun process group the moment Lightning reports the WRONG
    39	# rank count, plus the post-hoc classification in fa_orbit_classify.py.
    40	#
    41	# torchrun: PL 2.1.0 elects TorchElastic before SLURMEnvironment, so the ranks
    42	# torchrun starts are used as-is; the SLURM rank variables are unset so
    43	# SLURMEnvironment cannot claim the job. train.py is unmodified and rank-safe:
    44	# WandbLogger.experiment is @rank_zero_experiment, and ModelCheckpoint.setup
    45	# broadcasts rank 0's dirpath to every rank.
    46	#
    47	# SMOKE MODE (SMOKE=1): the reviewed pre-launch smoke. Bypasses ONLY the "pins
    48	# must be pinned" gate; every other gate still runs. Uses SMOKE_RUNG,
    49	# SMOKE_MAXSTEPS (small), SMOKE_MIN_FREE_MB, its own identity
    50	# (FLAC_exp11_smoke_<ARM> / exp11_smoke_<ARM>) and its own save-dir prefix, so a
    51	# smoke can never touch or resume an arm's real lineage.
    52	#
    53	# TEST HOOK: OUTPUT_ROOT (default outputs_FLAC) relocates the output namespace so
    54	# the guard tests never write under a production prefix. It changes no gate.
    55	# ============================================================================
    56	#SBATCH --partition=all
    57	#SBATCH --nodes=1
    58	#SBATCH --ntasks=1
    59	#SBATCH --output=/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_train_%x_%j.out
    60	# TRANSCRIPT POLICY. This file is written by Slurm for the whole life of the run.
    61	# During the run it is deliberately UNTRACKED (the job removes it from the index
    62	# at launch, see the untrack block below): a tracked file that a running job
    63	# appends to is one a git checkout/stash can unlink out from under the job's file
    64	# descriptor, freezing the visible transcript while the run continues. Completed
    65	# transcripts are committed by the OPERATOR at run closure with `git add -f`.
    66	
    67	set -uo pipefail
    68	
    69	# ============================ PINNED RECIPE =================================
    70	# Filled from the reviewed P0 report; until then every value is the literal
    71	# placeholder and the launcher refuses to run (except under SMOKE=1).
    72	PIN_PLACEHOLDER="TO-PIN-AFTER-P0"
    73	PINNED_RUNG="8x8"                          # P0 run 1334933 + spot 9bf1936: fastest uniform rung where ALL arms fit (C32 peak 30,817 MiB)
    74	PINNED_MB="8"                              # micro-batch per GPU (8 x 8 = 64 = eff = BN batch)
    75	PINNED_NGPU="8"                            # ranks
    76	PINNED_MAXSTEPS=100000                     # Q10: extended budget (was 40000, the
    77	                                           # plan §2 primary matched step, which
    78	                                           # remains the TABLE step — the extension
    79	                                           # adds trajectory, it does not move the
    80	                                           # registered comparison point)
    81	PINNED_CHECKPOINT_EVERY=2500               # exp_07 cadence
    82	PINNED_MIN_FREE_MB="36500"                 # batched C32 peak 32,063 MiB + ~4.4 GB margin (max-across-arms floor)
    83	PINNED_TIME_LIMIT_C4L="24:00:00"           # batched 40k/0.6598 = 16.8 h x1.3 + startup
    84	PINNED_TIME_LIMIT_C8="35:00:00"            # batched 40k/0.4351 = 25.5 h x1.3 + startup
    85	PINNED_TIME_LIMIT_C16="60:00:00"           # batched 40k/0.2454 = 45.3 h x1.3 + startup
    86	PINNED_TIME_LIMIT_C32="112:00:00"          # batched 40k/0.1308 = 84.9 h x1.3 + startup — SINGLE segment (no wall-split needed)
    87	# VANL is the vanilla-conditioning arm of the SAME recipe (Q9): its cost comes
    88	# from the official P0 VAN_8x8 rate, not from an orbit slope, because it makes no
    89	# orbit passes at all — 40k/1.07 steps/s = 10.4 h x1.3 + startup.
    90	PINNED_TIME_LIMIT_VANL="14:00:00"
    91	# Q10 RESTART legs: 40k -> 100k is 60,000 further steps at the batched rates,
    92	# x1.3 + startup. Each must sit under the 168 h partition cap, and each does.
    93	PINNED_TIME_LIMIT_RESTART_C4L="34:00:00"    # 60k/0.6598 = 25.3 h
    94	PINNED_TIME_LIMIT_RESTART_C8="51:00:00"     # 60k/0.4351 = 38.3 h
    95	PINNED_TIME_LIMIT_RESTART_C16="89:00:00"    # 60k/0.2454 = 67.9 h
    96	PINNED_TIME_LIMIT_RESTART_C32="160:00:00"   # 60k/0.1308 = 127.4 h (cap 168 h)
    97	PINNED_TIME_LIMIT_RESTART_VANL="19:00:00"   # 60k/1.0722 = 15.5 h
    98	PINNED_P0_MANIFEST_SHA256="72607b922177208d56055d604b292d697b643ef3b7ab48261ab2e23a0cc2b53b"  # batched matrix manifest bd96575-…-a3ed28eb; spot manifest sha in the commit message
    99	# Environment pins (round-3 B6) — measured on the reviewed environment:
   100	PINNED_PYTHON="/n/fs/gatrdp/envs/flac/bin/python"
   101	PINNED_PL_VERSION="2.1.0"
   102	PINNED_TORCH_VERSION="2.7.0+cu126"
   103	PINNED_VAE_SHA256="8d82159eec35210198246f449bec6561fc19b514922f340a17515050daf7f0b9"
   104	# ============================================================================
   105	
   106	REPO=/n/fs/gatrdp/codespace/FLAC
   107	# TEST HOOK (guard tests only): sbatch copies this script to a spool dir, so the
   108	# repo path must be absolute; FA_ORBIT_REPO_OVERRIDE lets the guard suite point a
   109	# dry run at a worktree. It is honoured ONLY outside a Slurm job and scrubbed
   110	# immediately, so it can never influence a real launch.
   111	if [ -n "${FA_ORBIT_REPO_OVERRIDE:-}" ] && [ -z "${SLURM_JOB_ID:-}" ]; then
   112	  REPO="$FA_ORBIT_REPO_OVERRIDE"
   113	fi
   114	unset FA_ORBIT_REPO_OVERRIDE
   115	EXPDIR="$REPO/worklog/worklog_yixun/exp_11_fa_orbit_claude"
   116	EXP07="$REPO/worklog/worklog_yixun/exp_07_fa_scratch_claude"
   117	cd "$REPO" || exit 3
   118	unset PYTHONPATH PYTHONOPTIMIZE
   119	export PATH=/n/fs/gatrdp/envs/flac/bin:$PATH
   120	export PYTHONNOUSERSITE=1
   121	export HF_HOME=/n/fs/gatrdp/hf_cache
   122	export HF_HUB_OFFLINE=1
   123	
   124	DRYRUN="${DRYRUN:-0}"
   125	SMOKE="${SMOKE:-0}"
   126	# NEW-2: the production output namespace is not operator state. Inside a Slurm
   127	# job it is the literal below; an ambient value that disagrees aborts. The
   128	# override exists only for non-Slurm guard dry runs.
   129	PRODUCTION_OUTPUT_ROOT="outputs_FLAC"
   130	if [ -n "${SLURM_JOB_ID:-}" ]; then
   131	  if [ -n "${OUTPUT_ROOT:-}" ] && [ "$OUTPUT_ROOT" != "$PRODUCTION_OUTPUT_ROOT" ]; then
   132	    echo "ambient OUTPUT_ROOT='${OUTPUT_ROOT}' != the production literal '${PRODUCTION_OUTPUT_ROOT}' - abort"; exit 2
   133	  fi
   134	  OUTPUT_ROOT="$PRODUCTION_OUTPUT_ROOT"
   135	else
   136	  OUTPUT_ROOT="${OUTPUT_ROOT:-$PRODUCTION_OUTPUT_ROOT}"
   137	fi
   138	RESUME_CKPT="${RESUME_CKPT:-}"
   139	EXPECTED_STEP="${EXPECTED_STEP:-0}"
   140	TS="$(date '+%Y-%m-%d_%H-%M-%S')"
   141	
   142	die() { echo "$1"; exit "${2:-2}"; }
   143	
   144	# --- A. parameters ------------------------------------------------------------
   145	[ -n "${ARM:-}" ] || die "ARM must be exported (C4L|C8|C16|C32|VANL) - abort"
   146	[ -n "${EXPECT_SHA:-}" ] || die "EXPECT_SHA (full reviewed commit sha) must be exported - abort"
   147	case "$ARM" in
   148	  C4L|C8|C16|C32|VANL) ;;
   149	  *) die "ARM '${ARM}' is not a legal exp_11 arm — C4L|C8|C16|C32 only (FA1/VAN/CKPT4 are P0 profiling cells, never arms) - abort" ;;
   150	esac
   151	case "$EXPECTED_STEP" in ''|*[!0-9]*) die "EXPECTED_STEP '${EXPECTED_STEP}' must be a non-negative integer - abort";; esac
   152	
   153	# --- B. the pins decide the recipe (round-3 B1) -------------------------------
   154	if [ "$SMOKE" = "1" ]; then
   155	  RUNG="${SMOKE_RUNG:-}"; MAXSTEPS="${SMOKE_MAXSTEPS:-30}"; MIN_FREE_MB="${SMOKE_MIN_FREE_MB:-}"
   156	  CHECKPOINT_EVERY="${SMOKE_CHECKPOINT_EVERY:-10}"
   157	  [ -n "$RUNG" ] || die "SMOKE=1 requires SMOKE_RUNG (32x2|16x4|8x8) - abort"
   158	  [ -n "$MIN_FREE_MB" ] || die "SMOKE=1 requires SMOKE_MIN_FREE_MB (per-GPU floor) - abort"
   159	  TIME_LIMIT="${SMOKE_TIME:-00:30:00}"; TIME_PIN_NAME="SMOKE_TIME"
   160	  NAME="FLAC_exp11_smoke_${ARM}"; EXPNAME="exp11_smoke_${ARM}"
   161	  SAVEDIR="${OUTPUT_ROOT}/exp11_smoke/${ARM}"
   162	  echo "=== SMOKE MODE: pins bypassed, EVERY other gate active; identity ${EXPNAME} ==="
   163	else
   164	  # Q10 / re-pin fix 1: the wall pin follows the LEG, not the arm. A restart leg
   165	  # is 60,000 further steps, not 40,000 from scratch, so the submitter allocates
   166	  # PINNED_TIME_LIMIT_RESTART_<ARM>. The job selected PINNED_TIME_LIMIT_<ARM>
   167	  # regardless and then rejected its own (correct) allocation in gate H — the
   168	  # third hard-abort path the re-pin review found on jobs 3662828-30. The JOB now
   169	  # selects the same pin the submitter did and enforces THAT one.
   170	  if [ "$EXPECTED_STEP" -gt 0 ]; then
   171	    TIME_PIN_NAME="PINNED_TIME_LIMIT_RESTART_${ARM}"
   172	  else
   173	    TIME_PIN_NAME="PINNED_TIME_LIMIT_${ARM}"
   174	  fi
   175	  for PIN_NAME in PINNED_RUNG PINNED_MB PINNED_NGPU PINNED_MIN_FREE_MB PINNED_P0_MANIFEST_SHA256 \
   176	                  "$TIME_PIN_NAME"; do
   177	    eval "PIN_VAL=\${$PIN_NAME}"
   178	    [ "$PIN_VAL" != "$PIN_PLACEHOLDER" ] || die "${PIN_NAME} is still '${PIN_PLACEHOLDER}': the P0 report has not been pinned into this launcher yet — no arm may launch (use SMOKE=1 for the pre-launch smoke) - abort"
   179	  done
   180	  RUNG="$PINNED_RUNG"; MAXSTEPS="$PINNED_MAXSTEPS"; MIN_FREE_MB="$PINNED_MIN_FREE_MB"
   181	  CHECKPOINT_EVERY="$PINNED_CHECKPOINT_EVERY"
   182	  eval "TIME_LIMIT=\${${TIME_PIN_NAME}}"
   183	  NAME="FLAC_exp11_${ARM}"; EXPNAME="exp11_${ARM}"; SAVEDIR="${OUTPUT_ROOT}/exp11_${ARM}"
   184	fi
   185	
   186	case "$RUNG" in
   187	  32x2|16x4|8x8) ;;
   188	  *) die "rung '${RUNG}' must be 32x2, 16x4 or 8x8 - abort" ;;
   189	esac
   190	MB="${RUNG%x*}"; NGPU="${RUNG#*x}"
   191	[ "$((MB * NGPU))" -eq 64 ] || die "rung ${RUNG}: MB*NGPU = $((MB*NGPU)) != 64 (micro x N pin, plan §10) - abort"
   192	if [ "$SMOKE" != "1" ]; then
   193	  [ "$MB" = "$PINNED_MB" ] && [ "$NGPU" = "$PINNED_NGPU" ] || die "pin inconsistency: rung ${RUNG} vs PINNED_MB=${PINNED_MB}/PINNED_NGPU=${PINNED_NGPU} - abort"
   194	  [ "$MAXSTEPS" = "100000" ] || die "PINNED_MAXSTEPS is ${MAXSTEPS}, the registered budget is 100000 - abort"
   195	fi
   196	RUNDIR="${SAVEDIR}/${NAME}/${EXPNAME}"
   197	echo "=== exp_11 arm ${ARM} @ rung ${RUNG} (MB ${MB} x ${NGPU} GPU, grad-ckpt ON) — ${TS} — host $(hostname) ==="
   198	
   199	# --- C. commit binding + tracked-surface drift --------------------------------
   200	HEAD_SHA="$(git rev-parse HEAD 2>/dev/null)" || HEAD_SHA=""
   201	EXPREL="${EXPDIR#"$REPO"/}"; EXP07REL="${EXP07#"$REPO"/}"
   202	# The drift gate is scoped to CODE surfaces, not the whole exp folder: the four
   203	# arms are running and Slurm appends to their tracked *.out logs continuously, so
   204	# a folder-wide check would abort every screen on a live-log write. Configs,
   205	# drivers and validators are still fully covered. The patterns are QUOTED so
   206	# git, not the shell, expands them — a tracked file deleted from the worktree
   207	# still matches (content-gate review B2) — data/AR (the split JSONs the
   208	# dataloader opens) is covered, and a failing git status is fail-closed.
   209	DRIFT="$(git status --porcelain --untracked-files=no -- train.py defaults.ini src ":(exclude)src/tests" data/AR \
   210	          "$EXPREL/*.json" "$EXPREL/*.py" "$EXPREL/*.sbatch" "$EXPREL/*.sh" \
   211	          "$EXP07REL/FLAC_AR_BF.json" 2>&1)" \
   212	  || die "git status for the drift gate failed: ${DRIFT} - abort"
   213	# Commit binding is CONTENT-scoped: HEAD identity is sufficient but not
   214	# necessary. Two sessions commit to this checkout, so a pending leg must
   215	# survive commits that leave the training closure untouched — and abort on
   216	# any commit that changes it. The closure is what the job actually loads:
   217	# train.py, defaults.ini, src/, the data/AR split JSONs, the five arm
   218	# configs (enumerated — a shell glob would silently drop a config deleted
   219	# since EXPECT_SHA), this launcher, the four runtime helper scripts it
   220	# invokes, and exp_07's FLAC_AR_BF.json (C4L parity baseline).
   300	else:
   301	    want = {"C4L": 4, "C8": 8, "C16": 16, "C32": 32}[arm]
   302	    angles = t.get("frame_avg_angles")
   303	    if t.get("cond_method") != "fa_invariant":
   304	        bad.append(f"cond_method={t.get('cond_method')!r} (want fa_invariant)")
   305	    if not isinstance(angles, list) or len(angles) != want:
   306	        bad.append(f"frame_avg_angles has {angles and len(angles)} entries (want {want})")
   307	    elif angles != [k * 360.0 / want for k in range(want)]:
   308	        bad.append(f"frame_avg_angles are not the uniform C{want} orbit")
   309	if t.get("use_ema") is not True:
   310	    bad.append(f"use_ema={t.get('use_ema')!r} (want True)")
   311	vits = [c for c in cfg["model"]["conditioning"]["configs"] if c["type"] == "ViTCoordinates"]
   312	if sorted(c["id"] for c in vits) != ["context_poses_vit", "source_vit"]:
   313	    bad.append(f"ViT conditioner ids {sorted(c['id'] for c in vits)} != the expected two")
   314	# Post-P0: grad-ckpt ON for every arm; the KEY must exist and be literally True
   315	for c in vits:
   316	    if "gradient_checkpointing" not in c["config"]:
   317	        bad.append(f"{c['id']}: gradient_checkpointing key absent (want literal true)")
   318	    elif c["config"]["gradient_checkpointing"] is not True:
   319	        bad.append(f"{c['id']}: gradient_checkpointing={c['config']['gradient_checkpointing']!r} (want True)")
   320	if bad:
   321	    sys.exit("ARM/CONFIG GATE: " + "; ".join(bad))
   322	if arm == "VANL":
   323	    print(f"gate OK: {arm} is vanilla (no cond_method, no orbit), grad-ckpt True, EMA on")
   324	else:
   325	    print(f"gate OK: {arm} carries the uniform C{want} orbit, grad-ckpt True, EMA on")
   326	PY
   327	
   328	# --- E. lineage: INITIAL vs RESTART -------------------------------------------
   329	SAVEDIR_REAL="$(realpath -m "$SAVEDIR")"
   330	CKPT_DIR_REAL="$(realpath -m "${RUNDIR}/checkpoints")"
   331	LAUNCH_MANIFEST_LINK="${SAVEDIR}/launch_manifest.txt"     # written by the INITIAL launch
   332	if [ "$EXPECTED_STEP" -eq 0 ]; then
   333	  MODE="INITIAL"
   334	  [ -z "$RESUME_CKPT" ] || die "INITIAL launch must not carry RESUME_CKPT (set EXPECTED_STEP > 0 to declare a RESTART) - abort"
   335	  [ ! -e "$RUNDIR" ] || die "run directory ${RUNDIR} already exists — an INITIAL launch never clobbers a previous run - abort"
   336	else
   337	  MODE="RESTART"
   338	  [ -n "$RESUME_CKPT" ] || die "EXPECTED_STEP ${EXPECTED_STEP} declares a RESTART, but RESTART requires RESUME_CKPT - abort"
   339	  [ -f "$RESUME_CKPT" ] || die "RESUME_CKPT not found: ${RESUME_CKPT} - abort"
   340	  RESUME_REAL="$(realpath -m "$RESUME_CKPT")"
   341	  # exactly this arm's own checkpoints directory — not merely somewhere below the save root
   342	  case "$RESUME_REAL" in
   343	    "${CKPT_DIR_REAL}"/*.ckpt) ;;
   344	    *) die "a RESTART may only resume a checkpoint from ${CKPT_DIR_REAL}/ (got ${RESUME_REAL}) - abort" ;;
   345	  esac
   346	  [ "$MAXSTEPS" -gt "$EXPECTED_STEP" ] || die "MAXSTEPS ${MAXSTEPS} must exceed the resume step ${EXPECTED_STEP} - abort"
   347	fi
   348	echo "lineage: ${MODE} (expected_step ${EXPECTED_STEP}, max_steps ${MAXSTEPS}, ckpt every ${CHECKPOINT_EVERY}, time pin ${TIME_PIN_NAME}=${TIME_LIMIT})"
   349	
   350	# --- F. the exact train.py argv ----------------------------------------------
   351	ARGV=(
   352	  --model-config "$MODEL_CONFIG_ABS"
   353	  --dataset-config src/configs/dataset_configs/AR/train/acousticroom_train.json
   354	  --pretransform-ckpt-path weights/FLAC/VAE.safetensors
   355	  --max-steps "$MAXSTEPS" --batch-size "$MB" --accum-batches 1 --num-workers 6 --seed 42
   356	  --num-gpus "$NGPU" --num-nodes 1
   357	  --strategy ddp_find_unused_parameters_true --sync-batchnorm true --precision bf16-mixed
   358	  --val-every -1 --val-dataset-config ''
   359	  --gradient-clip-val 0.0
   360	  --logger wandb --checkpoint-every "$CHECKPOINT_EVERY"
   361	  --name "$NAME" --experiment-name "$EXPNAME" --save-dir "$SAVEDIR"
   362	)
   363	[ "$MODE" = "RESTART" ] && ARGV+=(--ckpt-path "$RESUME_CKPT")
   364	
   365	# --- G. argv-parity dry run (plan N13; round-3 N9 tightened) ------------------
   366	ARGV_FILE="$(mktemp)" || die "mktemp failed - abort" 3
   367	printf '%s\n' "${ARGV[@]}" > "$ARGV_FILE" || die "could not write the argv file - abort" 3
   368	python3 - "$ARGV_FILE" "$MODE" <<'PY'
   369	import sys
   370	# The exp_07 B-F reference argv (bf_scratch_launch.sh) — the lineage this sweep continues.
   371	REF = """--model-config worklog/worklog_yixun/exp_07_fa_scratch_claude/FLAC_AR_BF.json
   372	--dataset-config src/configs/dataset_configs/AR/train/acousticroom_train.json
   373	--pretransform-ckpt-path weights/FLAC/VAE.safetensors
   374	--max-steps 67500 --batch-size 32 --accum-batches 1 --num-workers 6 --seed 42
   375	--num-gpus 2 --strategy ddp_find_unused_parameters_true --sync-batchnorm true
   376	--logger wandb --checkpoint-every 2500
   377	--name FLAC_exp07_BF --experiment-name exp07_BF --save-dir outputs_FLAC/exp07_BF""".split()
   378	# Flags whose VALUE may differ from exp_07 (identity, budget, rung, resume):
   379	ALLOWED_DIFF = {"--model-config", "--name", "--experiment-name", "--save-dir", "--max-steps",
   380	                "--num-gpus", "--batch-size", "--logger", "--checkpoint-every", "--ckpt-path"}
   381	# Flags exp_07 left to defaults.ini and we state explicitly — whitelisted with their
   382	# EXACT expected values (round-3 N9: no "equals the mutable ini" escape hatch):
   383	ALLOWED_ADD = {"--num-nodes": "1", "--precision": "bf16-mixed", "--val-every": "-1",
   384	               "--val-dataset-config": "", "--gradient-clip-val": "0.0", "--ckpt-path": None}
   385	tokens = [t for t in open(sys.argv[1]).read().split("\n")]
   386	if tokens and tokens[-1] == "":
   387	    tokens.pop()
   388	mode = sys.argv[2]
   389	
   390	def as_map(toks):
   391	    out, i = {}, 0
   392	    while i < len(toks):
   393	        flag = toks[i]
   394	        if not flag.startswith("--"):
   395	            raise SystemExit(f"ARGV PARITY: stray token {flag!r}")
   396	        val = toks[i + 1] if i + 1 < len(toks) and not toks[i + 1].startswith("--") else ""
   397	        if flag in out:
   398	            raise SystemExit(f"ARGV PARITY: duplicate flag {flag}")
   399	        out[flag] = val
   400	        i += 2 if (i + 1 < len(toks) and not toks[i + 1].startswith("--")) else 1
   401	    return out
   402	
   403	ref, new = as_map(REF), as_map(tokens)
   404	violations, allowed, explicit = [], [], []
   405	for flag in sorted(set(ref) | set(new)):
   406	    if flag in ref and flag in new:
   407	        if ref[flag] != new[flag]:
   408	            (allowed if flag in ALLOWED_DIFF else violations).append(
   409	                f"{flag}: exp_07 {ref[flag]!r} -> exp_11 {new[flag]!r}")
   410	    elif flag in new:
   411	        if flag == "--ckpt-path":
   412	            (allowed if mode == "RESTART" else violations).append(
   413	                f"--ckpt-path: {new[flag]!r} (RESTART only)")
   414	        elif flag in ALLOWED_ADD and ALLOWED_ADD[flag] == new[flag]:
   415	            explicit.append(f"{flag}={new[flag]!r} (whitelisted explicit default)")
   416	        else:
   417	            violations.append(f"{flag}: added with {new[flag]!r}, not a whitelisted addition "
   418	                              f"(expected {ALLOWED_ADD.get(flag, '<not allowed>')!r})")
   419	    else:
   420	        violations.append(f"{flag}: present in exp_07 ({ref[flag]!r}), MISSING here")
   421	
   422	print("--- train.py argv ---")
   423	print(" ".join(f"{k} {v!r}" if v == "" else f"{k} {v}" for k, v in new.items()))
   424	print("--- argv parity vs exp_07 B-F ---")
   425	for d in allowed:
   426	    print(f"  allowed  {d}")
   427	for d in explicit:
   428	    print(f"  explicit {d}")
   429	if violations:
   430	    print("ARGV PARITY VIOLATIONS:")
   431	    for v in violations:
   432	        print(f"  !! {v}")
   433	    raise SystemExit(2)
   434	print(f"ARGV PARITY OK ({mode}): only whitelisted differences and additions")
   435	PY
   436	parity=$?
   437	rm -f "$ARGV_FILE"
   438	[ "$parity" -eq 0 ] || die "argv parity check FAILED - abort"
   439	if [ "$DRYRUN" = "1" ]; then
   440	  echo "DRY RUN complete: gates A–G passed for ARM=${ARM} RUNG=${RUNG} MODE=${MODE} SMOKE=${SMOKE}"
   441	  echo "  (Slurm/GPU/VRAM/env/wandb/ViT/lock gates and training are skipped in DRYRUN)"
   442	  exit 0
   443	fi
   444	
   445	# --- H. Slurm allocation must match the pins (round-3 B1) ---------------------
   446	[ "${SLURM_JOB_NUM_NODES:-1}" = "1" ] || die "expected 1 node, got ${SLURM_JOB_NUM_NODES} - abort"
   447	[ "${SLURM_NTASKS:-1}" = "1" ] || die "expected 1 task, got ${SLURM_NTASKS} - abort"
   448	WANT_CPUS="$((8 + 7 * NGPU))"; WANT_MEM_MB="$(((12 * NGPU + 12) * 1024))"
   449	GOT_CPUS="${SLURM_CPUS_PER_TASK:-${SLURM_CPUS_ON_NODE:-0}}"
   450	GOT_MEM_MB="${SLURM_MEM_PER_NODE:-0}"
   451	[ "$GOT_CPUS" = "$WANT_CPUS" ] || die "allocated ${GOT_CPUS} CPUs, the pinned rung needs ${WANT_CPUS} — submit via fa_orbit_submit.sh - abort"
   452	[ "$GOT_MEM_MB" = "$WANT_MEM_MB" ] || die "allocated ${GOT_MEM_MB} MB RAM, the pinned rung needs ${WANT_MEM_MB} — submit via fa_orbit_submit.sh - abort"
   453	GOT_TIME="$(squeue -h -j "$SLURM_JOB_ID" -o %l 2>/dev/null | tr -d ' ')"
   454	norm_minutes() { awk -v t="$1" 'BEGIN{d=0; if (t ~ /-/) {split(t,p,"-"); d=p[1]; t=p[2]} n=split(t,c,":");
   455	  if (n==3) m=c[1]*60+c[2]+c[3]/60; else if (n==2) m=c[1]+c[2]/60; else m=t; printf "%d", d*1440+m}'; }
   456	# The pin this ${MODE} leg is entitled to — an INITIAL allocation handed to a
   457	# RESTART leg (or the reverse) is refused here, in the job, not merely intended
   458	# by the submitter.
   459	[ "$(norm_minutes "$GOT_TIME")" = "$(norm_minutes "$TIME_LIMIT")" ] || die "allocated time ${GOT_TIME} != the ${TIME_PIN_NAME} pin ${TIME_LIMIT} this ${MODE} leg requires — submit via fa_orbit_submit.sh - abort"
   460	echo "allocation matches the pins: ${GOT_CPUS} cpus, ${GOT_MEM_MB} MB, ${GOT_TIME} (${TIME_PIN_NAME})"
   500	echo "--- co-tenancy disclosure at launch ---"
   501	nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true
   502	
   503	# --- K. exclusive run ownership via flock (round-3 B3 residual) --------------
   504	# mkdir + stale recovery had two races: a contender could arrive between mkdir
   505	# and the owner write, and release removed the directory without checking whose
   506	# it was. flock has neither: the kernel holds the lock while the fd is open and
   507	# releases it on close (including on kill), so there is no stale state to
   508	# recover and no recovery path to get wrong.
   509	mkdir -p "$OUTPUT_ROOT" || die "could not create ${OUTPUT_ROOT} - abort" 3
   510	LOCKFILE="${OUTPUT_ROOT}/exp11_${ARM}.lock"
   511	exec 9>"$LOCKFILE" || die "could not open the lock file ${LOCKFILE} - abort" 3
   512	if ! flock -n 9; then
   513	  OWNER="$(tr '\n' ' ' < "$LOCKFILE" 2>/dev/null)"
   514	  die "arm ${ARM} is locked by another live job (${OWNER:-<no metadata>}) - refusing a concurrent writer - abort"
   515	fi
   516	LAUNCH_UUID="$(cat /proc/sys/kernel/random/uuid)"
   517	{ echo "job ${SLURM_JOB_ID}"; echo "uuid ${LAUNCH_UUID}"; echo "arm ${ARM}"; echo "mode ${MODE}"; echo "acquired ${TS}"; } >&9 \
   518	  || die "could not write the lock owner metadata - abort" 3
   519	echo "lock acquired: ${LOCKFILE} (flock on fd 9, released on exit)"
   520	mkdir -p "$SAVEDIR" || die "could not create ${SAVEDIR} - abort" 3
   521	
   522	# --- L. RESTART preflight (round-3 B2) ---------------------------------------
   523	CKPT_SHA=""
   524	if [ "$MODE" = "RESTART" ]; then
   525	  PRE_ARGS=(--ckpt "$RESUME_CKPT" --expected-step "$EXPECTED_STEP" --config "$MODEL_CONFIG_ABS"
   526	            --max-steps "$MAXSTEPS" --arm "$ARM" --rung "$RUNG" --commit "$HEAD_SHA")
   527	  [ -n "$LAUNCH_MANIFEST_LINK" ] && PRE_ARGS+=(--launch-manifest "$LAUNCH_MANIFEST_LINK")
   528	  # Q10 / re-pin fix 1: a real arm's restart is the 40k -> 100k EXTENSION, whose
   529	  # contract binds the ORIGINAL launch identity (audited manifest bytes, job,
   530	  # uuid, launch commit, config, save-dir, seed, and the 40k anchor itself)
   531	  # without demanding that the INITIAL budget/commit equal this leg's. SMOKE
   532	  # restarts have no registered launch and keep the ordinary contract.
   533	  [ "$SMOKE" != "1" ] && PRE_ARGS+=(--extension --launch-registry "$EXPDIR/arm_launch_registry.json"
   534	                                    --repo-root "$REPO")
   535	  PRE_OUT="$(python3 "$EXPDIR/fa_orbit_ckpt_preflight.py" "${PRE_ARGS[@]}" 2>&1)"
   536	  echo "$PRE_OUT"
   537	  echo "$PRE_OUT" | grep -q "^CKPT_SHA256 " || die "restart preflight FAILED - abort"
   538	  CKPT_SHA="$(echo "$PRE_OUT" | awk '/^CKPT_SHA256 /{print $2}')"
   539	fi
   540	
   541	# --- M. wandb: scrub, pin the destination, fix the run id (round-3 B7) --------
   542	unset WANDB_MODE WANDB_DISABLED WANDB_ENTITY WANDB_RUN_ID WANDB_RESUME WANDB_DIR WANDB_PROJECT WANDB_NAME
   543	export WANDB_DIR="$REPO/$SAVEDIR"
   544	export WANDB_MODE=online
   545	eval "$(grep -E '^[[:space:]]*export[[:space:]]+WANDB_API_KEY=' ~/.bashrc 2>/dev/null | tail -1)"
   546	WANDB_ENTITY_SEEN="$(python3 - <<'PY'
   547	import sys
   548	try:
   549	    import wandb
   550	    v = wandb.Api().viewer
   551	except Exception as e:
   552	    sys.exit(f"wandb identity check FAILED: {e}")
   553	if v.email != "yh4742@princeton.edu":
   554	    sys.exit(f"wandb identity {v.email} != yh4742@princeton.edu")
   555	print(v.entity)
   556	PY
   557	)" || die "wandb identity gate FAILED (no logger fallback: the arms train with wandb) - abort"
   558	[ -n "$WANDB_ENTITY_SEEN" ] || die "wandb returned an empty entity - abort"
   559	# B7 residual: pin the destination account explicitly instead of leaving it implicit
   560	export WANDB_ENTITY="$WANDB_ENTITY_SEEN"
   561	echo "wandb identity OK: entity ${WANDB_ENTITY} (exported), mode ${WANDB_MODE}"
   562	if [ "$MODE" = "RESTART" ]; then
   563	  ORIG_WANDB_RUN_ID="$(awk '/^wandb_run_id /{print $2}' "$LAUNCH_MANIFEST_LINK" 2>/dev/null)"
   564	  [ -n "$ORIG_WANDB_RUN_ID" ] || die "RESTART needs the original wandb run id from ${LAUNCH_MANIFEST_LINK} - abort"
   565	  # A RESUMED wandb run refuses config changes: prefigure's push_wandb_config
   566	  # calls config.update() without allow_val_change, and a restart legitimately
   567	  # changes max_steps (40000->100000) and ckpt_path — legs 3684149/3684150
   568	  # died at exactly train.py:193 on this. So a restart runs as a FRESH run
   569	  # carrying the lineage in its name; curves split across runs but panels
   570	  # align on the step axis. train.py stays byte-unchanged (exp_15 admission
   571	  # pins it).
   572	  WANDB_RUN_ID="exp11-${ARM}-r${EXPECTED_STEP}-$(date +%s%N)-$(cut -c1-8 /proc/sys/kernel/random/uuid)"
   573	  export WANDB_RUN_ID
   574	  echo "wandb: new RESTART run id ${WANDB_RUN_ID} (continues ${ORIG_WANDB_RUN_ID}; not resumed — a resumed run refuses the changed max_steps/ckpt_path config)"
   575	else
   576	  WANDB_RUN_ID="exp11-${ARM}-$(date +%s%N)-$(cut -c1-8 /proc/sys/kernel/random/uuid)"
   577	  export WANDB_RUN_ID
   578	  echo "wandb: new run id ${WANDB_RUN_ID}"
   579	fi
   580	
   581	# --- N. DINOv3 pin + init-identity gate (inside the allocation) ---------------
   582	HF_HUB_OFFLINE=1 python3 "$EXPDIR/assert_arm_configs_exp11.py" "$ARM" || die "ViT-pin / init-identity GATE FAILED - abort" 1
   583	
   584	# --- O. atomic manifest, duplicated to the save-dir (round-3 B5) --------------
   585	# --- the live transcript must not be a TRACKED file while it is being written --
   586	# A tracked file that a running job appends to is a file git will happily
   587	# replace. `git rebase --autostash` (remote coordination between machines) stashes
   588	# the dirty working tree, checks out, and restores — and each of those steps
   589	# UNLINKS the path and writes a new inode. The job's stdout descriptor still
   590	# points at the old, now-nameless inode, so every subsequent line is written to a
   591	# file with no name and the visible transcript freezes forever. That is exactly
   592	# what happened to C16 (job 3648696): its transcript stops at Epoch 5 while the
   593	# run went on to step 40000, and all four arm transcripts froze at the same
   594	# instant, 02:04:07, during a rebase cycle.
   595	#
   596	# So: at launch, the job removes its OWN Slurm transcript from the index. The
   597	# file stays on disk untouched; it is simply no longer something git will move.
   598	# The operator commits completed transcripts at closure (git add -f).
   599	SLURM_OUT_AT_LAUNCH="$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null \
   600	                        | tr ' ' '\n' | awk -F= '$1=="StdOut"{print $2; exit}')"
   601	UNTRACK_STATE="not-attempted"
   602	if [ -n "$SLURM_OUT_AT_LAUNCH" ]; then
   603	  if git -C "$REPO" ls-files --error-unmatch "$SLURM_OUT_AT_LAUNCH" >/dev/null 2>&1; then
   604	    if git -C "$REPO" rm --cached --quiet -- "$SLURM_OUT_AT_LAUNCH" 2>/dev/null; then
   605	      UNTRACK_STATE="untracked-at-launch"
   606	      echo "live transcript untracked for the duration of this run: ${SLURM_OUT_AT_LAUNCH}"
   607	      echo "  (the file is untouched on disk; commit it at closure with git add -f)"
   608	    else
   609	      UNTRACK_STATE="untrack-FAILED"
   610	      echo "WARNING: could not untrack ${SLURM_OUT_AT_LAUNCH}; a git working-tree"
   611	      echo "         operation during this run can still detach the transcript"
   612	    fi
   613	  else
   614	    UNTRACK_STATE="already-untracked"      # the normal steady state
   615	  fi
   616	else
   617	  UNTRACK_STATE="stdout-path-unknown"
   618	fi
   619	
   620	TRAINLOG="${EXPDIR}/fa_orbit_${TS}_${ARM}_${RUNG}_jid${SLURM_JOB_ID}_train.log"
   621	SAVEDIR_LOG="${SAVEDIR}/fa_orbit_${TS}_${ARM}_${RUNG}_jid${SLURM_JOB_ID}_train.log"
   622	MANIFEST="${EXPDIR}/fa_orbit_${TS}_${ARM}_${RUNG}_jid${SLURM_JOB_ID}_manifest.txt"
   623	# B5 residual: a failed environment dump must not be silently hashed into the record
   624	PIPFREEZE_FILE="$(mktemp)" || die "mktemp failed - abort" 3
   625	pip freeze > "$PIPFREEZE_FILE" 2>/dev/null || die "pip freeze FAILED - the environment digest would be a lie - abort" 7
   626	ENV_SHA="$(sha256sum "$PIPFREEZE_FILE" | awk '{print $1}')"
   627	rm -f "$PIPFREEZE_FILE"
   628	{
   629	  echo "# exp_11 arm launch manifest"
   630	  echo "timestamp ${TS}"
   631	  echo "job ${SLURM_JOB_ID} host $(hostname) mode ${MODE} launch_uuid ${LAUNCH_UUID}"
   632	  echo "arm ${ARM} rung ${RUNG} micro ${MB} ngpu ${NGPU} max_steps ${MAXSTEPS} ckpt_every ${CHECKPOINT_EVERY}"
   633	  echo "commit ${HEAD_SHA}"
   634	  echo "p0_manifest_sha256 ${PINNED_P0_MANIFEST_SHA256}"
   635	  echo "model_config ${MODEL_CONFIG_ABS}"
   636	  echo "config_sha256 ${CONFIG_SHA}"
   637	  echo "vae_sha256 ${VAE_SHA}"
   638	  echo "env_pip_freeze_sha256 ${ENV_SHA}"
   639	  echo "torch_version $(python3 -c 'import torch; print(torch.__version__)') cuda ${CUDA_VER} driver ${DRIVER}"
   640	  echo "gpu_uuids ${UUID_CSV}"
   641	  echo "time_limit ${TIME_LIMIT} min_free_mb ${MIN_FREE_MB}"
   642	  echo "resume_ckpt ${RESUME_CKPT:-<none>} expected_step ${EXPECTED_STEP} resume_ckpt_sha256 ${CKPT_SHA:-<none>}"
   643	  echo "save_dir ${SAVEDIR}"
   644	  echo "slurm_transcript ${SLURM_OUT_AT_LAUNCH:-<unknown>} untrack ${UNTRACK_STATE}"
   645	  echo "train_log ${TRAINLOG}"
   646	  echo "train_log_copy ${SAVEDIR_LOG}"
   647	  echo "wandb_entity ${WANDB_ENTITY_SEEN} wandb_project ${NAME} wandb_name ${EXPNAME}"
   648	  echo "wandb_run_id ${WANDB_RUN_ID}"
   649	  echo "command torchrun --standalone --nnodes=1 --nproc_per_node=${NGPU} train.py ${ARGV[*]}"
   650	} > "${MANIFEST}.tmp" || die "manifest write FAILED - abort" 3
   651	mv "${MANIFEST}.tmp" "$MANIFEST" || die "manifest publication FAILED - abort" 3
   652	cp "$MANIFEST" "${SAVEDIR}/$(basename "$MANIFEST")" || die "manifest copy to the save-dir FAILED - abort" 3
   653	[ "$MODE" = "INITIAL" ] && { cp "$MANIFEST" "$LAUNCH_MANIFEST_LINK" || die "launch-manifest link write FAILED - abort" 3; }
   654	echo "manifest: ${MANIFEST} (copied to ${SAVEDIR})"
   655	
   656	# --- P. training: torchrun + FIFO tee (both statuses captured) ----------------
   657	unset SLURM_NTASKS SLURM_JOB_NAME SLURM_PROCID SLURM_LOCALID SLURM_NODEID
   658	: > "$TRAINLOG" || die "could not create ${TRAINLOG} - abort" 3
   659	: > "$SAVEDIR_LOG" || die "could not create ${SAVEDIR_LOG} - abort" 3
   660	FIFO="$(mktemp)" || die "mktemp failed - abort" 3
   661	rm -f "$FIFO" && mkfifo "$FIFO" || die "mkfifo failed - abort" 3
   662	trap 'rm -f "$FIFO"' EXIT
   663	tee -a "$TRAINLOG" "$SAVEDIR_LOG" < "$FIFO" &
   664	TEE_PID=$!
   665	echo "=== launching ${ARM} ${RUNG}: torchrun --standalone --nproc_per_node=${NGPU} train.py (${MODE}) ==="
   666	torchrun --standalone --nnodes=1 --nproc_per_node="$NGPU" train.py "${ARGV[@]}" > "$FIFO" 2>&1 &
   667	TR_PID=$!
   668	
   669	# world-size watcher: terminate the moment Lightning reports the WRONG rank count
   670	WORLD_RE="All distributed processes registered\. Starting with [0-9]+ processes"
   671	(
   672	  while kill -0 "$TR_PID" 2>/dev/null; do
   673	    if grep -Eq "$WORLD_RE" "$TRAINLOG" 2>/dev/null; then
   674	      GOT="$(grep -Eo "$WORLD_RE" "$TRAINLOG" | head -1 | grep -Eo '[0-9]+')"
   675	      if [ "$GOT" != "$NGPU" ]; then
   676	        echo "WORLD-SIZE WATCHER: Lightning started with ${GOT} processes, expected ${NGPU} — terminating torchrun"
   677	        pkill -TERM -P "$TR_PID" 2>/dev/null; kill -TERM "$TR_PID" 2>/dev/null
   678	      fi
   679	      exit 0
   680	    fi
   681	    sleep 5
   682	  done
   683	) &
   684	WATCHER=$!
   685	
   686	wait "$TR_PID"; rc=$?
   687	kill "$WATCHER" 2>/dev/null; wait "$WATCHER" 2>/dev/null
   688	wait "$TEE_PID"; tee_rc=$?
   689	rm -f "$FIFO"
   690	
   691	# --- Q. W&B run identity verification (round-3 B7 residual) ------------------
   692	# The gate above proves WHO we are; this proves WHERE the run actually landed.
   693	# train.py:165 builds WandbLogger(project=, name=) with NO save_dir, so PL passes
   694	# its default save_dir='.' into wandb.init and that OVERRIDES the exported
   695	# WANDB_DIR: in job 3646734 the run went to $REPO/wandb/run-<ts>-<id> while this
   696	# check looked under $WANDB_DIR/wandb and found nothing (training was green, the
   697	# job still classified 7). We keep exporting WANDB_DIR — other wandb artifacts do
   698	# respect it — but locate the run by the collision-proof id WE generated, which
   699	# wandb embeds in the directory name, across both candidate roots. Exactly one
   700	# match is required; a mismatch is a provenance failure (class 7), not a footnote.
   701	WANDB_CHECK_RC=0
   702	python3 "$EXPDIR/fa_orbit_wandb_readback.py" --run-id "$WANDB_RUN_ID" \
   703	  --root "$REPO" --root "$WANDB_DIR" \
   704	  --entity "$WANDB_ENTITY" --project "$NAME" --name "$EXPNAME" || WANDB_CHECK_RC=$?
   705	if [ "$WANDB_CHECK_RC" -ne 0 ]; then
   706	  echo "W&B run identity could not be verified against the manifest - provenance failure"
   707	fi
   708	
   709	# --- R. classification (tested taxonomy; see fa_orbit_classify.py) ------------
   710	CLASSIFY_OUT="$(python3 "$EXPDIR/fa_orbit_classify.py" --rc "$rc" --tee-rc "$tee_rc" --ngpu "$NGPU" \
   711	  --maxsteps "$MAXSTEPS" --log "$TRAINLOG" --log-copy "$SAVEDIR_LOG" 2>&1)"
   712	final_rc=$?
   713	echo "$CLASSIFY_OUT"
   714	LAST_CKPT="$(ls -1t "${RUNDIR}/checkpoints"/*.ckpt 2>/dev/null | head -1)"
   715	
   716	# --- S. durable final record (round-3 B5 residual) ---------------------------
   717	# The classifier verdict and this summary must survive in BOTH copies; the tee
   718	# status is captured (it used to be discarded) and a failure is class 7. The
   719	# whole preflight transcript (this job's Slurm output so far) is copied into the
   720	# save-dir too, so the durable record is not just torchrun's stdout.
   721	FINAL_RECORD="$(
   722	  echo "=== exp_11 ${ARM} ${RUNG} finished at $(date '+%Y-%m-%d %H:%M:%S') ==="
   723	  echo "torchrun rc=${rc} tee rc=${tee_rc} wandb_identity_rc=${WANDB_CHECK_RC} classified rc=${final_rc}"
   724	  echo "$CLASSIFY_OUT"
   725	  echo "last checkpoint: ${LAST_CKPT:-<none>}"
   726	  echo "restart: fa_orbit_submit.sh ${ARM} --resume ${LAST_CKPT:-<path>} --expected-step <step>"
   727	)"
   728	printf '%s\n' "$FINAL_RECORD" | tee -a "$TRAINLOG" "$SAVEDIR_LOG" > /dev/null
   729	final_tee_rc="${PIPESTATUS[1]}"
   730	# The transcript path is NOT guessed from the #SBATCH pattern (job 3646734 logged

codex
The delta removes the direct `WANDB_RESUME=must` export and preserves the original launch manifest as the restart anchor. I’m now tracing whether any later consumer incorrectly assumes every leg shares that original W&B ID, and whether the readback matcher treats the new ID literally.
exec
/usr/bin/bash -lc "grep -RInE 'WANDB_RESUME|resume=[\"'\"']?(must|allow|auto|never)|wandb_run_id|wandb_run|run-id|run_id' . --exclude-dir=.git --exclude='*.log' --exclude='*.out' --exclude='*.ckpt' --exclude='*.safetensors' --exclude='*.pt' --exclude='*.pth' --exclude='*.bin'" in /n/fs/gatrdp/codespace/FLAC
 exited 2 in 227ms:
/usr/bin/bash: -c: line 1: unexpected EOF while looking for matching `"'
/usr/bin/bash: -c: line 2: syntax error: unexpected end of file

exec
/usr/bin/bash -lc "grep -RInE 'WANDB_RESUME|wandb_run_id|wandb_run|run-id|run_id' . --exclude-dir=.git --exclude='*.log' --exclude='*.out' --exclude='*.ckpt' --exclude='*.safetensors' --exclude='*.pt' --exclude='*.pth' --exclude='*.bin'" in /n/fs/gatrdp/codespace/FLAC
exec
/usr/bin/bash -lc "grep -RInE 'WANDB_RESUME|wandb_run_id|wandb_run|run-id|run_id' worklog/worklog_yixun/exp_11_fa_orbit_claude src train.py defaults.ini 2>/dev/null | head -200" in /n/fs/gatrdp/codespace/FLAC
exec
/usr/bin/bash -lc "grep -n WANDB_RESUME worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch; grep -n wandb_run_id worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch; grep -nE 'wandb|Wandb' train.py | head -80" in /n/fs/gatrdp/codespace/FLAC
 succeeded in 353ms:
542:unset WANDB_MODE WANDB_DISABLED WANDB_ENTITY WANDB_RUN_ID WANDB_RESUME WANDB_DIR WANDB_PROJECT WANDB_NAME
563:  ORIG_WANDB_RUN_ID="$(awk '/^wandb_run_id /{print $2}' "$LAUNCH_MANIFEST_LINK" 2>/dev/null)"
648:  echo "wandb_run_id ${WANDB_RUN_ID}"
6:from prefigure.prefigure import get_all_args, push_wandb_config
164:    if args.logger == 'wandb':
165:        logger = pl.loggers.WandbLogger(project=args.name, name=args.experiment_name)
192:    if args.logger == 'wandb':
193:        push_wandb_config(logger, args_dict)

exec
/usr/bin/bash -lc "git grep -n -E 'WANDB_RESUME|wandb_run_id|wandb_run|run-id|run_id' -- ':"'!*.log'"' ':"'!*.out'"' ':"'!*.jsonl'"' || true" in /n/fs/gatrdp/codespace/FLAC
 succeeded in 10001ms:
src/tests/test_exp11_restart_record.py:442:        "wandb_run_id exp11-C8-initial", ""]))
src/tests/test_exp11_wandb_readback.py:1:"""Tests for the exp_11 launcher's W&B run-identity readback.
src/tests/test_exp11_wandb_readback.py:44:def _make_run(root, run_id=RUN_ID, ts="20260806_164917", meta=None):
src/tests/test_exp11_wandb_readback.py:46:    run_dir = os.path.join(root, "wandb", f"run-{ts}-{run_id}")
src/tests/test_exp11_wandb_readback.py:89:    _make_run(root, run_id="exp11-C8-someotherrun-aaaaaaaa")
src/tests/test_exp11_wandb_readback.py:95:def test_empty_run_id_is_refused(tmp_path):
src/tests/test_exp11_wandb_readback.py:129:    run = _make_run(str(tmp_path), run_id="a-different-id", meta={})
src/tests/test_exp11_wandb_readback.py:145:    rc = R.main(["--run-id", RUN_ID, "--root", str(repo), "--root", str(wandb_dir),
src/tests/test_exp11_wandb_readback.py:151:    rc = R.main(["--run-id", RUN_ID, "--root", str(tmp_path)])
worklog/worklog_yixun/exp_11_fa_orbit_claude/commits_fa_orbit.md:25:| 16 | `983a7ff` | launcher residuals — flock ownership (B3), fail-closed manifest-commit binding (B2), pip-freeze/final-tee/preflight-transcript durability (B5), exported W&B entity + post-run run-identity verification (B7), intent manifest before sbatch (NEW-3), safe FIFO (NEW-4) |
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-06_16-47-14_C4L_8x8_jid3646734_manifest.txt:19:wandb_run_id exp11-C4L-1786049318048844980-bd40da20
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-06_18-00-52_C4L_8x8_jid3648568_manifest.txt:19:wandb_run_id exp11-C4L-1786053756799558763-4ae12465
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-06_20-36-00_C4L_8x8_jid3648694_manifest.txt:19:wandb_run_id exp11-C4L-1786063010468957329-bc46fb0a
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-06_20-55-01_C8_8x8_jid3648695_manifest.txt:19:wandb_run_id exp11-C8-1786064131292302937-6d92e299
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-06_20-55-50_C16_8x8_jid3648696_manifest.txt:19:wandb_run_id exp11-C16-1786064168022803862-f44c29b2
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-07_00-16-30_C32_8x8_jid3648697_manifest.txt:19:wandb_run_id exp11-C32-1786076295103433762-98dd1f9b
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-09_12-06-39_VANL_8x8_jid3661520_manifest.txt:20:wandb_run_id exp11-VANL-1786291671381616649-772b3272
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_code_r3_review.md:85:   The parent gate verifies only `wandb.Api().viewer.email` ([fa_orbit_train.sbatch:312](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:312)). Because submission uses `--export=ALL`, ambient `WANDB_MODE`, `WANDB_DISABLED`, `WANDB_ENTITY`, `WANDB_RUN_ID` or `WANDB_RESUME` can redirect, disable, reuse or offline the actual logger while the account check still succeeds. The manifest records only project/display name, not the actual entity and run ID ([fa_orbit_train.sbatch:350](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:350)).
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_content_gate_review.md:1488:   521	unset WANDB_MODE WANDB_DISABLED WANDB_ENTITY WANDB_RUN_ID WANDB_RESUME WANDB_DIR WANDB_PROJECT WANDB_NAME
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_content_gate_review.md:1509:   542	  WANDB_RUN_ID="$(awk '/^wandb_run_id /{print $2}' "$LAUNCH_MANIFEST_LINK" 2>/dev/null)"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_content_gate_review.md:1511:   544	  export WANDB_RUN_ID WANDB_RESUME=must
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_content_gate_review.md:1586:   619	  echo "wandb_run_id ${WANDB_RUN_ID}"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_content_gate_review.md:1654:   673	python3 "$EXPDIR/fa_orbit_wandb_readback.py" --run-id "$WANDB_RUN_ID" \
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_content_gate_review.md:2871:worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_wandb_readback.py:40:        matches.extend(sorted(glob.glob(os.path.join(root, "wandb", f"run-*-{run_id}"))))
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_content_gate_review.md:2873:worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_wandb_readback.py:54:    if not os.path.basename(run_dir).endswith(f"-{run_id}"):
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_content_gate_review.md:2874:worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_wandb_readback.py:55:        problems.append(f"run directory {os.path.basename(run_dir)} does not carry id {run_id}")
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_content_gate_review_r2.md:1233:   273	wandb_run_id exp11-C8-test
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_content_gate_review_r2.md:1312:   352	    fh.write("wandb_run_id exp11-C8-ext\n")
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_content_gate_review_r2.md:2747:690:python3 "$EXPDIR/fa_orbit_wandb_readback.py" --run-id "$WANDB_RUN_ID" \
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:542:unset WANDB_MODE WANDB_DISABLED WANDB_ENTITY WANDB_RUN_ID WANDB_RESUME WANDB_DIR WANDB_PROJECT WANDB_NAME
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:563:  ORIG_WANDB_RUN_ID="$(awk '/^wandb_run_id /{print $2}' "$LAUNCH_MANIFEST_LINK" 2>/dev/null)"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:648:  echo "wandb_run_id ${WANDB_RUN_ID}"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:702:python3 "$EXPDIR/fa_orbit_wandb_readback.py" --run-id "$WANDB_RUN_ID" \
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh:287:wandb_run_id exp11-C8-test
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh:366:    fh.write("wandb_run_id exp11-C8-ext\n")
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_wandb_readback.py:28:def locate_run_dir(roots, run_id):
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_wandb_readback.py:29:    """Find ``<root>/wandb/run-*-<run_id>`` across ``roots``.
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_wandb_readback.py:34:    if not run_id:
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_wandb_readback.py:40:        matches.extend(sorted(glob.glob(os.path.join(root, "wandb", f"run-*-{run_id}"))))
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_wandb_readback.py:43:        return None, [f"no run directory for id {run_id} under any of {list(roots)}"]
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_wandb_readback.py:45:        return None, [f"ambiguous run id {run_id}: {matches}"]
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_wandb_readback.py:49:def verify_identity(run_dir, run_id, entity=None, project=None, name=None):
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_wandb_readback.py:54:    if not os.path.basename(run_dir).endswith(f"-{run_id}"):
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_wandb_readback.py:55:        problems.append(f"run directory {os.path.basename(run_dir)} does not carry id {run_id}")
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_wandb_readback.py:75:    ap.add_argument("--run-id", required=True)
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_wandb_readback.py:83:    run_dir, problems = locate_run_dir(args.root, args.run_id)
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_wandb_readback.py:88:    problems = verify_identity(run_dir, args.run_id, args.entity, args.project, args.name)
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_wandb_readback.py:92:    print(f"wandb run identity OK: id {args.run_id} at {run_dir} "
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_worklog.md:100:- **Change** — NEW-1: `FLAC_AR_VANCKPT.json` (canonical + gc:true ×2, parsed-delta-tested); CKPT4 family fully retired; matrix = 12 all-ckpt cells. NEW-2: OUTPUT_ROOT pinned to production literal under Slurm (both scripts + submitters). B2/B3/B5/B7 residuals closed (fail-closed commit binding, flock ownership, checked pip-freeze/dual-copy/transcript with class-7, WANDB_ENTITY export + post-run run-identity verification). NEW-3 intent-before-sbatch with scancel-on-failure; NEW-4 FIFO hygiene; NEW-5 comment corrected + 2 s liveness bound restored.
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_worklog.md:136:- **Re-pin** — `ea94995` (floors/limits/manifest sha from batched evidence). **Launcher fixes** `71054cf` (wandb readback by run-id glob — PL save_dir overrides WANDB_DIR, train.py:165; scontrol-derived transcript path fail-closed; separator hygiene) + 15 readback unit tests + 6 guard cases (suite: 208 pytest + 16 guards green).
worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_train.sbatch:754:unset WANDB_MODE WANDB_DISABLED WANDB_ENTITY WANDB_RUN_ID WANDB_RESUME WANDB_DIR WANDB_PROJECT WANDB_NAME
worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_train.sbatch:775:  WANDB_RUN_ID="$(awk '/^wandb_run_id /{print $2}' "$LAUNCH_MANIFEST_LINK" 2>/dev/null)"
worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_train.sbatch:777:  export WANDB_RUN_ID WANDB_RESUME=must
worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_train.sbatch:864:  echo "wandb_run_id ${WANDB_RUN_ID}"
worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_train.sbatch:895: vae_sha, save_dir, wandb_run_id) = sys.argv[1:13]
worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_train.sbatch:907:    "wandb_run_id": wandb_run_id,
worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_train.sbatch:1089:python3 "$(snap fa_orbit_wandb_readback.py)" --run-id "$WANDB_RUN_ID" \

exec
/usr/bin/bash -lc "nl -ba worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_wandb_readback.py; nl -ba src/tests/test_exp11_wandb_readback.py | sed -n '1,220p'; nl -ba worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_record_restart.py | sed -n '1,310p'" in /n/fs/gatrdp/codespace/FLAC
 succeeded in 5240ms:
     1	#!/usr/bin/env python3
     2	"""exp_11 — verify WHERE the W&B run actually landed, after training.
     3	
     4	The launcher's identity gate proves *who* we authenticate as; this proves *where*
     5	the run was written and that it carries the manifest's identity.
     6	
     7	Why the run directory must be SEARCHED FOR rather than assumed: ``train.py``
     8	constructs ``pl.loggers.WandbLogger(project=..., name=...)`` (train.py:165)
     9	without a ``save_dir``, so PL passes its own default ``save_dir='.'`` into
    10	``wandb.init``, and that argument OVERRIDES the exported ``WANDB_DIR``. In the
    11	pinned-path smoke (job 3646734) the run therefore landed in
    12	``$REPO/wandb/run-20260806_164917-exp11-C4L-<run id>`` while the readback looked
    13	under ``$WANDB_DIR/wandb`` and found nothing — training was green but the job
    14	classified 7. The launcher still exports ``WANDB_DIR`` (other wandb artifacts do
    15	respect it), but the readback locates the run by the id WE generated, which wandb
    16	embeds in the directory name, across every candidate root.
    17	
    18	Exactly one match is required: zero means the run is not where we think it is,
    19	and more than one means the id is ambiguous — both are provenance failures.
    20	"""
    21	import argparse
    22	import glob
    23	import json
    24	import os
    25	import sys
    26	
    27	
    28	def locate_run_dir(roots, run_id):
    29	    """Find ``<root>/wandb/run-*-<run_id>`` across ``roots``.
    30	
    31	    Returns ``(path, [])`` on exactly one match, else ``(None, [problems])``.
    32	    Roots are searched in order but ALL are collected first, so an id that
    33	    somehow exists under two roots is reported rather than silently preferred."""
    34	    if not run_id:
    35	        return None, ["no run id supplied"]
    36	    matches = []
    37	    for root in roots:
    38	        if not root:
    39	            continue
    40	        matches.extend(sorted(glob.glob(os.path.join(root, "wandb", f"run-*-{run_id}"))))
    41	    matches = sorted(set(matches))
    42	    if not matches:
    43	        return None, [f"no run directory for id {run_id} under any of {list(roots)}"]
    44	    if len(matches) > 1:
    45	        return None, [f"ambiguous run id {run_id}: {matches}"]
    46	    return matches[0], []
    47	
    48	
    49	def verify_identity(run_dir, run_id, entity=None, project=None, name=None):
    50	    """Check the run directory's embedded id and its wandb-metadata identity."""
    51	    problems = []
    52	    if not run_dir or not os.path.isdir(run_dir):
    53	        return [f"run directory {run_dir!r} does not exist"]
    54	    if not os.path.basename(run_dir).endswith(f"-{run_id}"):
    55	        problems.append(f"run directory {os.path.basename(run_dir)} does not carry id {run_id}")
    56	    meta_path = os.path.join(run_dir, "files", "wandb-metadata.json")
    57	    meta = {}
    58	    if os.path.isfile(meta_path):
    59	        try:
    60	            meta = json.load(open(meta_path))
    61	        except Exception as exc:
    62	            problems.append(f"unreadable {meta_path}: {exc}")
    63	            return problems
    64	    for key, want in (("entity", entity), ("project", project), ("name", name)):
    65	        got = meta.get(key)
    66	        # wandb-metadata does not always carry every field; only a CONTRADICTION
    67	        # is a failure, an absent field is not.
    68	        if want and got is not None and got != want:
    69	            problems.append(f"{key}={got!r} != manifest {want!r}")
    70	    return problems
    71	
    72	
    73	def main(argv=None):
    74	    ap = argparse.ArgumentParser(description="verify the created W&B run identity")
    75	    ap.add_argument("--run-id", required=True)
    76	    ap.add_argument("--root", action="append", default=[],
    77	                    help="candidate root; repeat. Searched as <root>/wandb/run-*-<id>")
    78	    ap.add_argument("--entity", default=None)
    79	    ap.add_argument("--project", default=None)
    80	    ap.add_argument("--name", default=None)
    81	    args = ap.parse_args(argv)
    82	
    83	    run_dir, problems = locate_run_dir(args.root, args.run_id)
    84	    if problems:
    85	        for p in problems:
    86	            print(f"WANDB IDENTITY: {p}")
    87	        return 1
    88	    problems = verify_identity(run_dir, args.run_id, args.entity, args.project, args.name)
    89	    if problems:
    90	        print("WANDB IDENTITY MISMATCH: " + "; ".join(problems))
    91	        return 1
    92	    print(f"wandb run identity OK: id {args.run_id} at {run_dir} "
    93	          f"(entity {args.entity}, project {args.project}, name {args.name})")
    94	    return 0
    95	
    96	
    97	if __name__ == "__main__":
    98	    sys.exit(main())
     1	"""Tests for the exp_11 launcher's W&B run-identity readback.
     2	
     3	Job 3646734 trained perfectly — 8 ranks, 30 steps, checkpoint written, dual logs
     4	byte-identical — and still classified 7, because the readback looked for the run
     5	under ``$WANDB_DIR/wandb`` while wandb had written it under ``$REPO/wandb``:
     6	``train.py:165`` builds ``WandbLogger(project=, name=)`` with no ``save_dir``, so
     7	PL passes its default ``save_dir='.'`` into ``wandb.init`` and that argument wins
     8	over the exported ``WANDB_DIR``.
     9	
    10	The fix locates the run by the collision-proof id the launcher generated, which
    11	wandb embeds in the run directory name, across every candidate root. These tests
    12	pin that contract: the real layout is found, both roots are searched, exactly one
    13	match is required, and a directory whose id does not match is rejected.
    14	"""
    15	import importlib.util
    16	import json
    17	import os
    18	
    19	import pytest
    20	
    21	
    22	_REPO_ROOT = os.path.dirname(
    23	    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    24	)  # src/tests/ -> src/ -> repo root
    25	_READBACK_PY = os.path.join(
    26	    _REPO_ROOT, "worklog", "worklog_yixun", "exp_11_fa_orbit_claude",
    27	    "fa_orbit_wandb_readback.py",
    28	)
    29	
    30	RUN_ID = "exp11-C4L-1786049318048844980-bd40da20"
    31	
    32	
    33	def _load_module():
    34	    spec = importlib.util.spec_from_file_location("exp11_wandb_readback", _READBACK_PY)
    35	    assert spec is not None and spec.loader is not None, f"cannot load {_READBACK_PY}"
    36	    mod = importlib.util.module_from_spec(spec)
    37	    spec.loader.exec_module(mod)
    38	    return mod
    39	
    40	
    41	R = _load_module()
    42	
    43	
    44	def _make_run(root, run_id=RUN_ID, ts="20260806_164917", meta=None):
    45	    """Create the on-disk layout wandb actually produced in job 3646734."""
    46	    run_dir = os.path.join(root, "wandb", f"run-{ts}-{run_id}")
    47	    os.makedirs(os.path.join(run_dir, "files"), exist_ok=True)
    48	    if meta is not None:
    49	        with open(os.path.join(run_dir, "files", "wandb-metadata.json"), "w") as fh:
    50	            json.dump(meta, fh)
    51	    return run_dir
    52	
    53	
    54	# --------------------------------------------------------------------------- #
    55	# 1. locating the run
    56	# --------------------------------------------------------------------------- #
    57	def test_finds_the_run_under_the_repo_root_where_pl_actually_writes_it(tmp_path):
    58	    repo, wandb_dir = tmp_path / "repo", tmp_path / "savedir"
    59	    (wandb_dir / "wandb").mkdir(parents=True)          # exported WANDB_DIR: empty
    60	    want = _make_run(str(repo))                        # PL's save_dir='.' wins
    61	    got, problems = R.locate_run_dir([str(repo), str(wandb_dir)], RUN_ID)
    62	    assert problems == [] and got == want
    63	
    64	
    65	def test_finds_the_run_under_wandb_dir_when_that_is_where_it_landed(tmp_path):
    66	    repo, wandb_dir = tmp_path / "repo", tmp_path / "savedir"
    67	    (repo / "wandb").mkdir(parents=True)
    68	    want = _make_run(str(wandb_dir))
    69	    got, problems = R.locate_run_dir([str(repo), str(wandb_dir)], RUN_ID)
    70	    assert problems == [] and got == want
    71	
    72	
    73	def test_missing_run_is_a_failure_not_a_shrug(tmp_path):
    74	    got, problems = R.locate_run_dir([str(tmp_path)], RUN_ID)
    75	    assert got is None
    76	    assert problems and RUN_ID in problems[0]
    77	
    78	
    79	def test_ambiguous_id_across_roots_is_rejected(tmp_path):
    80	    repo, wandb_dir = tmp_path / "repo", tmp_path / "savedir"
    81	    _make_run(str(repo), ts="20260806_164917")
    82	    _make_run(str(wandb_dir), ts="20260806_170000")
    83	    got, problems = R.locate_run_dir([str(repo), str(wandb_dir)], RUN_ID)
    84	    assert got is None and any("ambiguous" in p for p in problems)
    85	
    86	
    87	def test_other_runs_in_the_same_root_are_ignored(tmp_path):
    88	    root = str(tmp_path)
    89	    _make_run(root, run_id="exp11-C8-someotherrun-aaaaaaaa")
    90	    want = _make_run(root)
    91	    got, problems = R.locate_run_dir([root], RUN_ID)
    92	    assert problems == [] and got == want
    93	
    94	
    95	def test_empty_run_id_is_refused(tmp_path):
    96	    got, problems = R.locate_run_dir([str(tmp_path)], "")
    97	    assert got is None and problems
    98	
    99	
   100	# --------------------------------------------------------------------------- #
   101	# 2. verifying the identity
   102	# --------------------------------------------------------------------------- #
   103	def test_identity_matches(tmp_path):
   104	    run = _make_run(str(tmp_path), meta={"entity": "ent", "project": "FLAC_exp11_C4L",
   105	                                         "name": "exp11_C4L"})
   106	    assert R.verify_identity(run, RUN_ID, "ent", "FLAC_exp11_C4L", "exp11_C4L") == []
   107	
   108	
   109	@pytest.mark.parametrize("field,bad", [("entity", "someone-else"),
   110	                                       ("project", "FLAC_exp11_C8"),
   111	                                       ("name", "exp11_C8")])
   112	def test_identity_contradiction_is_a_failure(tmp_path, field, bad):
   113	    meta = {"entity": "ent", "project": "FLAC_exp11_C4L", "name": "exp11_C4L"}
   114	    meta[field] = bad
   115	    run = _make_run(str(tmp_path), meta=meta)
   116	    problems = R.verify_identity(run, RUN_ID, "ent", "FLAC_exp11_C4L", "exp11_C4L")
   117	    assert problems and field in problems[0]
   118	
   119	
   120	def test_absent_metadata_field_is_not_a_contradiction(tmp_path):
   121	    """wandb-metadata does not always carry every field; only a CONTRADICTION fails."""
   122	    run = _make_run(str(tmp_path), meta={"entity": "ent"})
   123	    assert R.verify_identity(run, RUN_ID, "ent", "FLAC_exp11_C4L", "exp11_C4L") == []
   124	    run2 = _make_run(str(tmp_path / "b"))                  # no metadata file at all
   125	    assert R.verify_identity(run2, RUN_ID, "ent", "P", "N") == []
   126	
   127	
   128	def test_run_dir_not_carrying_the_id_is_rejected(tmp_path):
   129	    run = _make_run(str(tmp_path), run_id="a-different-id", meta={})
   130	    problems = R.verify_identity(run, RUN_ID, None, None, None)
   131	    assert problems and "does not carry id" in problems[0]
   132	
   133	
   134	def test_missing_directory_is_rejected(tmp_path):
   135	    assert R.verify_identity(str(tmp_path / "nope"), RUN_ID, None, None, None)
   136	
   137	
   138	# --------------------------------------------------------------------------- #
   139	# 3. the CLI the launcher actually calls
   140	# --------------------------------------------------------------------------- #
   141	def test_cli_returns_zero_on_the_real_layout(tmp_path, capsys):
   142	    repo, wandb_dir = tmp_path / "repo", tmp_path / "savedir"
   143	    (wandb_dir / "wandb").mkdir(parents=True)
   144	    _make_run(str(repo), meta={"entity": "ent", "project": "P", "name": "N"})
   145	    rc = R.main(["--run-id", RUN_ID, "--root", str(repo), "--root", str(wandb_dir),
   146	                 "--entity", "ent", "--project", "P", "--name", "N"])
   147	    assert rc == 0 and "wandb run identity OK" in capsys.readouterr().out
   148	
   149	
   150	def test_cli_returns_nonzero_when_the_run_is_absent(tmp_path, capsys):
   151	    rc = R.main(["--run-id", RUN_ID, "--root", str(tmp_path)])
   152	    assert rc == 1 and "WANDB IDENTITY" in capsys.readouterr().out
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
    29	"""
    30	import argparse
    31	import fcntl
    32	import hashlib
    33	import json
    34	import os
    35	import re
    36	import sys
    37	
    38	HERE = os.path.dirname(os.path.abspath(__file__))
    39	sys.path.insert(0, HERE)
    40	import fa_orbit_producer_manifest as pm            # noqa: E402
    41	from fa_orbit_ckpt_preflight import canonical_ckpt_dir    # noqa: E402
    42	
    43	PIN_RE = re.compile(r'^(PINNED_[A-Z0-9_]+)=(?:"([^"]*)"|(\S+))')
    44	
    45	
    46	def read_pins(launcher):
    47	    """The launcher's own PINNED_* values, so the recorder cannot drift from them."""
    48	    pins = {}
    49	    with open(launcher) as fh:
    50	        for line in fh:
    51	            m = PIN_RE.match(line)
    52	            if m:
    53	                pins[m.group(1)] = m.group(2) if m.group(2) is not None else m.group(3)
    54	    return pins
    55	
    56	
    57	def parse_manifest(path):
    58	    raw = open(path, "rb").read()
    59	    man = {}
    60	    for line in raw.decode().splitlines():
    61	        line = line.strip()
    62	        if line and not line.startswith("#"):
    63	            k, _, rest = line.partition(" ")
    64	            man[k] = rest.strip()
    65	    return raw, man
    66	
    67	
    68	def kvs(man, key):
    69	    f = (f"{key} " + man.get(key, "")).split()
    70	    return {f[i]: f[i + 1] for i in range(0, len(f) - 1, 2)}
    71	
    72	
    73	def check_identity(arm, man, initial, pins, repo_root):
    74	    """Every field of the RESTART manifest, against the audited INITIAL row + Q10 pins."""
    75	    jk, ak, rk = kvs(man, "job"), kvs(man, "arm"), kvs(man, "resume_ckpt")
    76	    tk = kvs(man, "time_limit")
    77	    problems = []
    78	    anchor, final_step = initial.get("final_ckpt_sha256"), initial.get("final_step")
    79	    if not anchor:
    80	        problems.append(f"{arm} has no audited final_ckpt_sha256 to chain from — audit the "
    81	                        "INITIAL run's final checkpoint before recording a leg")
    82	    if jk.get("mode") != "RESTART":
    83	        problems.append(f"manifest mode is {jk.get('mode')!r}, not RESTART")
    84	    for field, got in (("job", jk.get("job")), ("launch_uuid", jk.get("launch_uuid")),
    85	                       ("commit", man.get("commit"))):
    86	        if not got:
    87	            problems.append(f"manifest records no {field} — a leg with no identity is not a record")
    88	    if jk.get("job") and initial.get("job") == jk.get("job"):
    89	        problems.append(f"manifest job {jk.get('job')} IS the INITIAL job — that is the launch "
    90	                        "already registered, not a restart leg")
    91	    for label, got, want in (("arm", ak.get("arm"), arm),
    92	                             ("rung", ak.get("rung"), initial.get("rung")),
    93	                             ("micro", ak.get("micro"), pins.get("PINNED_MB")),
    94	                             ("ngpu", ak.get("ngpu"), pins.get("PINNED_NGPU")),
    95	                             ("config_sha256", man.get("config_sha256"), initial.get("config_sha256")),
    96	                             ("vae_sha256", man.get("vae_sha256"), initial.get("vae_sha256")),
    97	                             ("p0_manifest_sha256", man.get("p0_manifest_sha256"),
    98	                              initial.get("p0_manifest_sha256")),
    99	                             ("save_dir", man.get("save_dir"), initial.get("save_dir"))):
   100	        if got != want:
   101	            problems.append(f"manifest {label} {got!r} != the audited INITIAL run's {want!r}")
   102	    if ak.get("rung") != pins.get("PINNED_RUNG"):
   103	        problems.append(f"manifest rung {ak.get('rung')!r} != the pinned {pins.get('PINNED_RUNG')!r}")
   104	    if ak.get("max_steps") != pins.get("PINNED_MAXSTEPS"):
   105	        problems.append(f"manifest max_steps {ak.get('max_steps')!r} != the Q10 budget pin "
   106	                        f"{pins.get('PINNED_MAXSTEPS')!r}")
   107	    if final_step is not None and str(rk.get("expected_step")) != str(final_step):
   108	        problems.append(f"manifest expected_step {rk.get('expected_step')!r} != the audited final "
   109	                        f"step {final_step!r} — a leg resumes where the INITIAL run ended")
   110	    want_time = pins.get(f"PINNED_TIME_LIMIT_RESTART_{arm}")
   111	    if tk.get("time_limit") != want_time:
   112	        problems.append(f"manifest time_limit {tk.get('time_limit')!r} != the arm's RESTART wall "
   113	                        f"pin {want_time!r}")
   114	    if int(initial.get("training_seed", -1)) != 42:
   115	        problems.append(f"registered training seed {initial.get('training_seed')!r} != 42")
   116	    # the config the leg names must still hash to the audited value
   117	    cfg_path = man.get("model_config", "")
   118	    if not cfg_path or not os.path.isfile(cfg_path):
   119	        problems.append(f"manifest model_config {cfg_path!r} does not exist")
   120	    elif hashlib.sha256(open(cfg_path, "rb").read()).hexdigest() != initial.get("config_sha256"):
   121	        problems.append(f"{cfg_path} no longer hashes to the audited config_sha256")
   122	    # --- the resume file itself: MUST exist, MUST be canonical, ALWAYS re-hashed --
   123	    resume_path = (man.get("resume_ckpt", "").split() or [""])[0]
   124	    resume_real = ""
   125	    if not resume_path or resume_path == "<none>":
   126	        problems.append("manifest records no resume_ckpt — a RESTART that resumed nothing is not "
   127	                        "a continuation of the audited run")
   128	    elif not os.path.isfile(resolve(repo_root, resume_path)):
   129	        problems.append(f"the resume checkpoint {resume_path} does not exist — the recorder does "
   130	                        "NOT accept the manifest's claimed hash in its place")
   131	    else:
   132	        resume_real = resolve(repo_root, resume_path)
   133	        canon = canonical_ckpt_dir(initial.get("save_dir", ""), arm, repo_root)
   134	        if os.path.realpath(os.path.dirname(resume_real)) != canon:
   135	            problems.append(f"the resume checkpoint is not in the audited launch's canonical "
   136	                            f"directory {canon}")
   137	        got = pm.sha256_file(resume_real)
   138	        if anchor and got != anchor:
   139	            problems.append(f"the resume file on disk hashes {got[:12]}, not the audited "
   140	                            f"{anchor[:12]} — this leg does not continue that run")
   141	        if rk.get("resume_ckpt_sha256") != got:
   142	            problems.append(f"manifest resume_ckpt_sha256 {str(rk.get('resume_ckpt_sha256'))[:12]} "
   143	                            f"!= the file's actual {got[:12]}")
   144	    return problems, resume_real
   145	
   146	
   147	def resolve(root, path):
   148	    return path if os.path.isabs(path) else os.path.join(root, path)
   149	
   150	
   151	def main(argv=None):
   152	    ap = argparse.ArgumentParser(description="record an exp_11 RESTART leg")
   153	    ap.add_argument("arm")
   154	    ap.add_argument("manifest")
   155	    ap.add_argument("--registry", default=os.path.join(HERE, "arm_launch_registry.json"))
   156	    ap.add_argument("--launcher", default=os.path.join(HERE, "fa_orbit_train.sbatch"),
   157	                    help="where the Q10 pins are read from")
   158	    ap.add_argument("--producer-dir", default=HERE,
   159	                    help="where the per-leg producer manifests are published")
   160	    # HERE = <repo>/worklog/worklog_<user>/exp_11_fa_orbit_claude
   161	    ap.add_argument("--repo-root", default=os.path.dirname(os.path.dirname(os.path.dirname(HERE))),
   162	                    help="root the manifest's relative paths resolve against")
   163	    ap.add_argument("--extend", action="store_true",
   164	                    help="this leg is already recorded: extend its producer manifest only")
   165	    ap.add_argument("--rehash-all", action="store_true",
   166	                    help="re-hash published checkpoints too (full audit, expensive)")
   167	    ap.add_argument("--dry-run", action="store_true", help="validate and report, publish nothing")
   168	    args = ap.parse_args(argv)
   169	
   170	    arm = args.arm
   171	    pins = read_pins(args.launcher)
   172	    if not pins.get("PINNED_MAXSTEPS"):
   173	        raise SystemExit(f"no PINNED_* values found in {args.launcher}")
   174	
   175	    # One writer at a time, and the lock is the registry's own DIRECTORY: no lock
   176	    # file to leave behind in a tracked tree, and it still covers the tmp+rename.
   177	    store = os.path.dirname(os.path.abspath(args.registry)) or "."
   178	    lock_fd = os.open(store, os.O_RDONLY)
   179	    try:
   180	        fcntl.flock(lock_fd, fcntl.LOCK_EX)
   181	        return record(args, arm, pins)
   182	    finally:
   183	        os.close(lock_fd)
   184	
   185	
   186	def record(args, arm, pins):
   187	    reg = json.load(open(args.registry))
   188	    initial = reg.get("arms", {}).get(arm)
   189	    if initial is None:
   190	        raise SystemExit(f"{arm} has no INITIAL registry entry")
   191	
   192	    raw, man = parse_manifest(args.manifest)
   193	    man_sha = hashlib.sha256(raw).hexdigest()
   194	    problems, resume_real = check_identity(arm, man, initial, pins, args.repo_root)
   195	    jk, ak, rk = kvs(man, "job"), kvs(man, "arm"), kvs(man, "resume_ckpt")
   196	    job = jk.get("job")
   197	
   198	    legs = reg.setdefault("restarts", {}).setdefault(arm, [])
   199	    same = [l for l in legs if l.get("job") == job or l.get("launch_uuid") == jk.get("launch_uuid")
   200	            or l.get("manifest_sha256") == man_sha]
   201	    if same and not args.extend:
   202	        raise SystemExit(f"{arm} job {job} is ALREADY recorded ({len(same)} matching leg(s)) — "
   203	                         "one leg, one row; use --extend to extend its producer manifest")
   204	    if len(same) > 1:
   205	        problems.append(f"{len(same)} registry rows already claim this leg — the registry is "
   206	                        "inconsistent; fix it before recording")
   207	    if args.extend and not same:
   208	        problems.append(f"--extend given but {arm} job {job} is not recorded yet")
   209	    if problems:
   210	        print("RECORD REFUSED:")
   211	        for p in problems:
   212	            print(f"  !! {p}")
   213	        return 2
   214	
   215	    anchor = initial["final_ckpt_sha256"]
   216	    producer = pm.manifest_name(arm, job)
   217	    row = {
   218	        "manifest_path": args.manifest, "manifest_sha256": man_sha,
   219	        "job": job, "mode": "RESTART", "launch_uuid": jk.get("launch_uuid"),
   220	        "arm": arm, "commit": man.get("commit"), "rung": ak.get("rung"),
   221	        "config_sha256": man.get("config_sha256"), "save_dir": man.get("save_dir"),
   222	        "resume_ckpt": resume_real, "resume_ckpt_sha256": anchor,
   223	        "expected_step": rk.get("expected_step"), "max_steps": ak.get("max_steps"),
   224	        "time_limit": kvs(man, "time_limit").get("time_limit"),
   225	        "producer_manifest": producer, "chains_to": anchor,
   226	        "recorded_at": pm._now(),
   227	    }
   228	    header = {"arm": arm, "job": job, "launch_uuid": jk.get("launch_uuid"), "mode": "RESTART",
   229	              "commit": man.get("commit"), "resume_ckpt_sha256": anchor,
   230	              "expected_step": rk.get("expected_step"), "max_steps": ak.get("max_steps"),
   231	              "save_dir": man.get("save_dir"), "config_sha256": man.get("config_sha256"),
   232	              "chains_to": anchor, "leg_manifest_sha256": man_sha}
   233	
   234	    ckpt_dir = canonical_ckpt_dir(initial["save_dir"], arm, args.repo_root)
   235	    prod_path = os.path.join(args.producer_dir, producer)
   236	    known = (pm.load(prod_path) or {}).get("checkpoints", {})
   237	    found, scan_problems = pm.scan_checkpoints(
   238	        ckpt_dir, int(rk["expected_step"]), int(ak["max_steps"]), known=known,
   239	        rehash_all=args.rehash_all, repo_root=args.repo_root)
   240	    added, kept, pub_problems = pm.publish(prod_path, header, found, dry_run=args.dry_run)
   241	    if scan_problems or pub_problems:
   242	        print("RECORD REFUSED:")
   243	        for p in scan_problems + pub_problems:
   244	            print(f"  !! {p}")
   245	        return 2
   246	
   247	    if args.extend:
   248	        for i, leg in enumerate(legs):
   249	            if leg.get("job") == job:
   250	                legs[i] = {**leg, "producer_manifest": producer}
   251	    else:
   252	        legs.append(row)
   253	    if not args.dry_run:
   254	        pm.write_atomic(args.registry, reg)
   255	    verb = "extended" if args.extend else "recorded"
   256	    print(f"{verb} {arm} RESTART job {job} chaining to {anchor[:12]} "
   257	          f"({'dry run, nothing written' if args.dry_run else 'published'})")
   258	    print(f"  producer manifest {producer}: {len(added)} checkpoint(s) added, "
   259	          f"{len(kept)} already published"
   260	          + (f" (steps {added[0]}..{added[-1]})" if added else ""))
   261	    return 0
   262	
   263	
   264	if __name__ == "__main__":
   265	    sys.exit(main())

exec
/usr/bin/bash -lc "python3 -m json.tool worklog/worklog_yixun/exp_11_fa_orbit_claude/arm_launch_registry.json | sed -n '1,260p'; nl -ba worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py | sed -n '60,225p;270,330p'; nl -ba worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_producer_manifest.py | sed -n '160,280p'" in /n/fs/gatrdp/codespace/FLAC
 succeeded in 2207ms:
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
    60	    entries are absent)."""
    61	    import torch
    62	    ck = torch.load(path, map_location="cpu", weights_only=False)
    63	    if not isinstance(ck, dict):
    64	        raise RuntimeError(f"not a Lightning checkpoint: {path}")
    65	    return list((ck.get("state_dict") or {}).keys())
    66	
    67	
    68	def parse_manifest(path):
    69	    """The launcher's own manifest format: whitespace-separated `key value...`."""
    70	    out = {}
    71	    with open(path, "r") as fh:
    72	        for line in fh:
    73	            line = line.strip()
    74	            if not line or line.startswith("#"):
    75	                continue
    76	            key, _, rest = line.partition(" ")
    77	            out[key] = rest.strip()
    78	    return out
    79	
    80	
    81	def check_manifest_binding(manifest_path, arm, rung, commit, maxsteps):
    82	    man = parse_manifest(manifest_path)
    83	    problems = []
    84	    fields = man.get("arm", "")
    85	    # `arm <ARM> rung <RUNG> micro <MB> ngpu <N> max_steps <S> ...`
    86	    tokens = ("arm " + fields).split()
    87	    kv = {tokens[i]: tokens[i + 1] for i in range(0, len(tokens) - 1, 2)}
    88	    if kv.get("arm") != arm:
    89	        problems.append(f"manifest arm {kv.get('arm')!r} != {arm!r}")
    90	    if kv.get("rung") != rung:
    91	        problems.append(f"manifest rung {kv.get('rung')!r} != {rung!r} "
    92	                        "(a restart may not change the rung: it would change rank count, "
    93	                        "sampler partitioning and worker seeding mid-lineage)")
    94	    if kv.get("max_steps") != str(maxsteps):
    95	        problems.append(f"manifest max_steps {kv.get('max_steps')!r} != {maxsteps}")
    96	    # Fail-CLOSED (round-3 B2 residual): an absent or empty manifest commit is not
    97	    # "no opinion", it is missing provenance — the restart must not proceed on it.
    98	    man_commit = man.get("commit", "").strip()
    99	    if not man_commit:
   100	        problems.append("launch manifest carries no 'commit' line — cannot bind the restart "
   101	                        "to the lineage that produced this checkpoint")
   102	    elif not commit:
   103	        problems.append("no running commit supplied to compare against the manifest commit")
   104	    elif man_commit != commit:
   105	        problems.append(f"manifest commit {man_commit[:12]} != running commit {commit[:12]}")
   106	    return problems, man
   107	
   108	
   109	def kv_line(man, key):
   110	    """One manifest line's `k v k v ...` pairs (the launcher's `arm ...`/`job ...`)."""
   111	    f = (f"{key} " + man.get(key, "")).split()
   112	    return {f[i]: f[i + 1] for i in range(0, len(f) - 1, 2)}
   113	
   114	
   115	def canonical_ckpt_dir(save_dir, arm, repo_root):
   116	    """<save_dir>/FLAC_exp11_<ARM>/exp11_<ARM>/checkpoints, as the launcher builds it.
   117	
   118	    save_dir is recorded relative to the repo root, so it is resolved against it."""
   119	    base = save_dir if os.path.isabs(save_dir) else os.path.join(repo_root, save_dir)
   120	    return os.path.realpath(os.path.join(base, f"FLAC_exp11_{arm}", f"exp11_{arm}", "checkpoints"))
   121	
   122	
   123	def check_extension_binding(manifest_path, registry_path, arm, rung, config_path, ckpt_path,
   124	                            ckpt_sha, expected_step, max_steps, repo_root="."):
   125	    """The 40k -> 100k EXTENSION contract (re-pin review, required fix 1).
   126	
   127	    A crash restart continues the SAME launch: same budget, same reviewed commit,
   128	    so `check_manifest_binding` demands both. An extension breaks both BY DESIGN
   129	    — it raises the budget from 40000 to 100000 and runs later reviewed code —
   130	    and demanding equality there is exactly what gave jobs 3662828-30 their third
   131	    hard-abort path.
   132	
   133	    What an extension must still prove is the ORIGINAL LAUNCH IDENTITY, and it
   134	    proves it against the COMMITTED registry rather than the mutable manifest
   135	    alone: the INITIAL manifest byte-for-byte as audited, the same job/uuid/
   136	    launch commit/rung/config/save-dir/training seed, and a resumed checkpoint
   137	    that IS that launch's audited final checkpoint, sitting in that launch's own
   138	    canonical run directory. Budget and running commit may move; nothing that
   139	    identifies the run may.
   140	    """
   141	    problems = []
   142	    if not os.path.isfile(registry_path):
   143	        return [f"audited launch registry not found: {registry_path}"], {}
   144	    reg = json.load(open(registry_path)).get("arms", {}).get(arm)
   145	    if reg is None:
   146	        return [f"{arm} is not in the audited launch registry {registry_path}"], {}
   147	    man = parse_manifest(manifest_path)
   148	    kv, jkv = kv_line(man, "arm"), kv_line(man, "job")
   149	
   150	    got_sha = sha256_file(manifest_path)
   151	    if got_sha != reg.get("manifest_sha256"):
   152	        problems.append(f"launch manifest sha256 {got_sha[:12]} != audited "
   153	                        f"{str(reg.get('manifest_sha256'))[:12]} — the manifest changed after it "
   154	                        "was registered")
   155	    for label, got_v, want_v in (("arm", kv.get("arm"), arm),
   156	                                 ("job", jkv.get("job"), reg.get("job")),
   157	                                 ("launch mode", jkv.get("mode"), "INITIAL"),
   158	                                 ("launch_uuid", jkv.get("launch_uuid"), reg.get("launch_uuid")),
   159	                                 ("rung", kv.get("rung"), reg.get("rung")),
   160	                                 ("rung (this run)", rung, reg.get("rung")),
   161	                                 ("config_sha256", man.get("config_sha256"), reg.get("config_sha256")),
   162	                                 ("save_dir", man.get("save_dir"), reg.get("save_dir"))):
   163	        if got_v != want_v:
   164	            problems.append(f"{label} {got_v!r} != registered {want_v!r}")
   165	    man_commit = man.get("commit", "").strip()
   166	    if not man_commit:
   167	        problems.append("launch manifest carries no 'commit' line — cannot bind the extension to "
   168	                        "the lineage that produced this checkpoint")
   169	    elif man_commit != reg.get("commit"):
   170	        problems.append(f"manifest commit {man_commit[:12]} != the registered launch commit "
   171	                        f"{str(reg.get('commit'))[:12]}")
   172	    if int(reg.get("training_seed", -1)) != 42:
   173	        problems.append(f"registered training seed {reg.get('training_seed')!r} != 42")
   174	    # The INITIAL budget is the manifest's and the registry's; the extension's is
   175	    # this run's, and it must strictly cover the resume point without shrinking.
   176	    initial_budget = reg.get("max_steps")
   177	    if kv.get("max_steps") != initial_budget:
   178	        problems.append(f"manifest max_steps {kv.get('max_steps')!r} != registered "
   179	                        f"{initial_budget!r} (the INITIAL budget, which an extension preserves)")
   180	    try:
   181	        if max_steps < int(initial_budget):
   182	            problems.append(f"extension budget {max_steps} does not extend the registered "
   183	                            f"{initial_budget} — an extension may only raise the budget")
   184	    except (TypeError, ValueError):
   185	        problems.append(f"registered max_steps {initial_budget!r} is not an integer")
   186	    if sha256_file(config_path) != reg.get("config_sha256"):
   187	        problems.append(f"{config_path} sha256 != the registered config_sha256 "
   188	                        f"{str(reg.get('config_sha256'))[:12]}")
   189	    # the resumed checkpoint IS the audited anchor, in the audited run directory
   190	    anchor, final_step = reg.get("final_ckpt_sha256"), reg.get("final_step")
   191	    if not anchor:
   192	        problems.append(f"{arm} has no audited final_ckpt_sha256 in the registry — the extension "
   193	                        "has nothing to chain to (audit the arm's final checkpoint first)")
   194	    elif ckpt_sha != anchor:
   195	        problems.append(f"resume checkpoint sha256 {ckpt_sha[:12]} != the audited final checkpoint "
   196	                        f"{anchor[:12]} — this leg does not continue that run")
   197	    if final_step is not None and int(final_step) != int(expected_step):
   198	        problems.append(f"EXPECTED_STEP {expected_step} != the registered final_step {final_step}")
   199	    save_dir = man.get("save_dir", "")
   200	    if not save_dir:
   201	        problems.append("manifest records no save_dir")
   202	    else:
   203	        canon = canonical_ckpt_dir(save_dir, arm, repo_root)
   204	        if os.path.realpath(os.path.dirname(ckpt_path)) != canon:
   205	            problems.append(f"resume checkpoint {ckpt_path} does not live in the registered "
   206	                            f"launch's canonical run directory {canon}")
   207	    return problems, man
   208	
   209	
   210	def main(argv=None):
   211	    ap = argparse.ArgumentParser(description="exp_11 restart checkpoint preflight")
   212	    ap.add_argument("--ckpt", required=True)
   213	    ap.add_argument("--expected-step", type=int, required=True)
   214	    ap.add_argument("--config", required=True, help="this arm's model config json")
   215	    ap.add_argument("--max-steps", type=int, required=True)
   216	    ap.add_argument("--arm", required=True)
   217	    ap.add_argument("--rung", required=True)
   218	    ap.add_argument("--commit", default="")
   219	    ap.add_argument("--launch-manifest", default="",
   220	                    help="the arm's original launch manifest (binds rung/commit/budget)")
   221	    ap.add_argument("--extension", action="store_true",
   222	                    help="this restart is the Q10 40k->100k EXTENSION: bind it to the audited "
   223	                         "INITIAL launch identity instead of requiring an equal budget/commit")
   224	    ap.add_argument("--launch-registry", default="",
   225	                    help="the committed arm launch registry (required with --extension)")
   270	    elif not opts[0].get("state"):
   271	        problems.append("optimizer state is CLEARED (stripped checkpoint); exp_11 restarts are "
   272	                        "WARM continuations and have no optimizer-reset lineage")
   273	    if not ck.get("lr_schedulers"):
   274	        problems.append("no 'lr_schedulers' -> PL 2.1 KeyErrors on resume")
   275	    sd = ck.get("state_dict") or {}
   276	    n_ema = sum(1 for k in sd if k.startswith("diffusion_ema."))
   277	    if not n_ema:
   278	        problems.append("no EMA weights in state_dict")
   279	
   280	    digest = sha256_file(args.ckpt)     # needed by the extension contract's anchor check
   281	    man = {}
   282	    if args.launch_manifest:
   283	        if not os.path.isfile(args.launch_manifest):
   284	            problems.append(f"launch manifest not found: {args.launch_manifest}")
   285	        elif args.extension:
   286	            more, man = check_extension_binding(
   287	                args.launch_manifest, args.launch_registry, args.arm, args.rung, args.config,
   288	                args.ckpt, digest, args.expected_step, args.max_steps, args.repo_root)
   289	            problems += more
   290	        else:
   291	            more, man = check_manifest_binding(args.launch_manifest, args.arm, args.rung,
   292	                                               args.commit, args.max_steps)
   293	            problems += more
   294	    elif args.extension:
   295	        problems.append("--extension requires --launch-manifest (the audited INITIAL manifest)")
   296	
   297	    if problems:
   298	        print("PREFLIGHT FAILED:")
   299	        for p in problems:
   300	            print(f"  !! {p}")
   301	        return 2
   302	
   303	    tr = mc.get("training", {})
   304	    print(f"restart lineage OK: {args.ckpt}")
   305	    print(f"  global_step={gs} epoch={ck.get('epoch')} cond_method={tr.get('cond_method')!r} "
   306	          f"angles={tr.get('frame_avg_angles')}")
   307	    print(f"  optimizer_state=FULL ({len(opts[0]['state'])} entries) "
   308	          f"lr={opts[0]['param_groups'][0].get('lr')} "
   309	          f"sched_last_epoch={ck['lr_schedulers'][0].get('last_epoch')} ema_entries={n_ema}")
   310	    if man and args.extension:
   311	        print(f"extension lineage OK: {args.arm} {args.expected_step} -> {args.max_steps} continues "
   312	              f"the audited launch job {kv_line(man, 'job').get('job')} "
   313	              f"(launch commit {man.get('commit', '')[:12]}, running commit {args.commit[:12] or '<none>'})")
   314	        print(f"  bound to the audited launch manifest: {args.launch_manifest}")
   315	    elif man:
   316	        print(f"  bound to launch manifest: {args.launch_manifest}")
   317	    print(f"CKPT_SHA256 {digest}")
   318	    return 0
   319	
   320	
   321	if __name__ == "__main__":
   322	    sys.exit(main())
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

web search: 
web search: site:docs.wandb.ai run id maximum 128 characters wandb ...
2026-08-12T22:30:45.245695Z ERROR codex_models_manager::manager: failed to renew cache TTL: missing field `base_instructions` at line 94 column 5
 succeeded in 69839ms:
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:542:unset WANDB_MODE WANDB_DISABLED WANDB_ENTITY WANDB_RUN_ID WANDB_RESUME WANDB_DIR WANDB_PROJECT WANDB_NAME
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:563:  ORIG_WANDB_RUN_ID="$(awk '/^wandb_run_id /{print $2}' "$LAUNCH_MANIFEST_LINK" 2>/dev/null)"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:648:  echo "wandb_run_id ${WANDB_RUN_ID}"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:702:python3 "$EXPDIR/fa_orbit_wandb_readback.py" --run-id "$WANDB_RUN_ID" \
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_code_r3_review.md:85:   The parent gate verifies only `wandb.Api().viewer.email` ([fa_orbit_train.sbatch:312](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:312)). Because submission uses `--export=ALL`, ambient `WANDB_MODE`, `WANDB_DISABLED`, `WANDB_ENTITY`, `WANDB_RUN_ID` or `WANDB_RESUME` can redirect, disable, reuse or offline the actual logger while the account check still succeeds. The manifest records only project/display name, not the actual entity and run ID ([fa_orbit_train.sbatch:350](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:350)).
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-06_20-55-01_C8_8x8_jid3648695_manifest.txt:19:wandb_run_id exp11-C8-1786064131292302937-6d92e299
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_worklog.md:100:- **Change** — NEW-1: `FLAC_AR_VANCKPT.json` (canonical + gc:true ×2, parsed-delta-tested); CKPT4 family fully retired; matrix = 12 all-ckpt cells. NEW-2: OUTPUT_ROOT pinned to production literal under Slurm (both scripts + submitters). B2/B3/B5/B7 residuals closed (fail-closed commit binding, flock ownership, checked pip-freeze/dual-copy/transcript with class-7, WANDB_ENTITY export + post-run run-identity verification). NEW-3 intent-before-sbatch with scancel-on-failure; NEW-4 FIFO hygiene; NEW-5 comment corrected + 2 s liveness bound restored.
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_worklog.md:136:- **Re-pin** — `ea94995` (floors/limits/manifest sha from batched evidence). **Launcher fixes** `71054cf` (wandb readback by run-id glob — PL save_dir overrides WANDB_DIR, train.py:165; scontrol-derived transcript path fail-closed; separator hygiene) + 15 readback unit tests + 6 guard cases (suite: 208 pytest + 16 guards green).
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-06_20-55-50_C16_8x8_jid3648696_manifest.txt:19:wandb_run_id exp11-C16-1786064168022803862-f44c29b2
worklog/worklog_yixun/exp_11_fa_orbit_claude/commits_fa_orbit.md:25:| 16 | `983a7ff` | launcher residuals — flock ownership (B3), fail-closed manifest-commit binding (B2), pip-freeze/final-tee/preflight-transcript durability (B5), exported W&B entity + post-run run-identity verification (B7), intent manifest before sbatch (NEW-3), safe FIFO (NEW-4) |
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-06_16-47-14_C4L_8x8_jid3646734_manifest.txt:19:wandb_run_id exp11-C4L-1786049318048844980-bd40da20
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh:287:wandb_run_id exp11-C8-test
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh:366:    fh.write("wandb_run_id exp11-C8-ext\n")
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-07_00-16-30_C32_8x8_jid3648697_manifest.txt:19:wandb_run_id exp11-C32-1786076295103433762-98dd1f9b
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-06_18-00-52_C4L_8x8_jid3648568_manifest.txt:19:wandb_run_id exp11-C4L-1786053756799558763-4ae12465
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-12_18-21-40_C16_8x8_jid3684151_manifest.txt:20:wandb_run_id exp11-C16-1786064168022803862-f44c29b2
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-09_12-06-39_VANL_8x8_jid3661520_manifest.txt:20:wandb_run_id exp11-VANL-1786291671381616649-772b3272
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:14:Round-4 delta review of worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch in /n/fs/gatrdp/codespace/FLAC (read-only; do NOT install anything or modify environments/files; output = review text only). Production failure: RESTART legs 3684149/3684150 died at train.py:193 — the launcher resumed the INITIAL leg's wandb run (WANDB_RESUME=must, same run id) and prefigure's push_wandb_config calls config.update() WITHOUT allow_val_change, so the restart's legitimately-changed config (max_steps 40000->100000, then ckpt_path) raises ConfigError. Full traceback in outputs_FLAC/exp11_C4L/fa_orbit_2026-08-12_18-11-38_C4L_8x8_jid3684149_train.log lines 129-151.
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:16:Delta (diff at /tmp/claude-374349/-n-fs-gatrdp-codespace-FLAC/66305eb9-88c7-41df-95da-dde30a426dc2/scratchpad/wandb_restart.diff): RESTART legs now mint a FRESH run id 'exp11-${ARM}-r${EXPECTED_STEP}-<ns>-<uuid8>' instead of resuming; the original id is still read from the launch manifest (fail-closed die retained) and echoed as lineage; WANDB_RESUME is no longer exported. Constraints honored: train.py byte-unchanged (exp_15's admission record pins it), prefigure untouched. Known cost: wandb curves split across runs per leg (accepted; figures come from eval JSONs).
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:18:Check: (1) any residual WANDB_RESUME/resume path that could still collide; (2) the run-id readback gate at ~line 670 (fa_orbit_wandb_readback.py --run-id + run-id glob over $REPO/wandb/run-*) works with the new id shape (r-token, dashes); (3) the manifest writer (line ~648 'wandb_run_id ${WANDB_RUN_ID}') now records the NEW id — does anything downstream (record_restart/registry/readback, future crash-restarts reading LAUNCH_MANIFEST_LINK) break or mislink when the manifest chain's wandb ids differ per leg; (4) WANDB_RUN_ID char limit (wandb caps run ids at 128 chars? verify the constructed id length is safe); (5) fresh holes. Verdict: APPROVED or REVISE.
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:53:-  WANDB_RUN_ID="$(awk '/^wandb_run_id /{print $2}' "$LAUNCH_MANIFEST_LINK" 2>/dev/null)"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:55:-  export WANDB_RUN_ID WANDB_RESUME=must
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:57:+  ORIG_WANDB_RUN_ID="$(awk '/^wandb_run_id /{print $2}' "$LAUNCH_MANIFEST_LINK" 2>/dev/null)"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:80:-  WANDB_RUN_ID="$(awk '/^wandb_run_id /{print $2}' "$LAUNCH_MANIFEST_LINK" 2>/dev/null)"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:82:-  export WANDB_RUN_ID WANDB_RESUME=must
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:84:+  ORIG_WANDB_RUN_ID="$(awk '/^wandb_run_id /{print $2}' "$LAUNCH_MANIFEST_LINK" 2>/dev/null)"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:101:/usr/bin/bash -lc "rg -n -i 'wandb|manifest|record_restart|registry|run-id|run_id|LAUNCH_MANIFEST|RESTART|EXPECTED_STEP' worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch worklog/worklog_yixun/exp_11_fa_orbit_claude --glob '*.py' --glob '*.sh' --glob '*.sbatch' --glob '*.json' --glob '*.txt'" in /n/fs/gatrdp/codespace/FLAC
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:106:/usr/bin/bash -lc "grep -RInE 'wandb|manifest|record_restart|registry|run-id|run_id|LAUNCH_MANIFEST|RESTART|EXPECTED_STEP' worklog/worklog_yixun/exp_11_fa_orbit_claude --include='*.py' --include='*.sh' --include='*.sbatch' --include='*.json' --include='*.txt'" in /n/fs/gatrdp/codespace/FLAC
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:218:worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:563:  ORIG_WANDB_RUN_ID="$(awk '/^wandb_run_id /{print $2}' "$LAUNCH_MANIFEST_LINK" 2>/dev/null)"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:230:worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:648:  echo "wandb_run_id ${WANDB_RUN_ID}"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:241:worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:702:python3 "$EXPDIR/fa_orbit_wandb_readback.py" --run-id "$WANDB_RUN_ID" \
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:275:worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-06_20-55-01_C8_8x8_jid3648695_manifest.txt:19:wandb_run_id exp11-C8-1786064131292302937-6d92e299
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:389:worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-06_20-55-50_C16_8x8_jid3648696_manifest.txt:19:wandb_run_id exp11-C16-1786064168022803862-f44c29b2
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:420:worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-06_16-47-14_C4L_8x8_jid3646734_manifest.txt:19:wandb_run_id exp11-C4L-1786049318048844980-bd40da20
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:436:worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh:287:wandb_run_id exp11-C8-test
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:460:worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh:366:    fh.write("wandb_run_id exp11-C8-ext\n")
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:525:worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-07_00-16-30_C32_8x8_jid3648697_manifest.txt:19:wandb_run_id exp11-C32-1786076295103433762-98dd1f9b
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:532:worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-06_18-00-52_C4L_8x8_jid3648568_manifest.txt:19:wandb_run_id exp11-C4L-1786053756799558763-4ae12465
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:566:worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-12_18-21-40_C16_8x8_jid3684151_manifest.txt:20:wandb_run_id exp11-C16-1786064168022803862-f44c29b2
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:575:worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-09_12-06-39_VANL_8x8_jid3661520_manifest.txt:20:wandb_run_id exp11-VANL-1786291671381616649-772b3272
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:637:worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-06_20-36-00_C4L_8x8_jid3648694_manifest.txt:19:wandb_run_id exp11-C4L-1786063010468957329-bc46fb0a
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:679:worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-12_18-11-38_C4L_8x8_jid3684149_manifest.txt:20:wandb_run_id exp11-C4L-1786063010468957329-bc46fb0a
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:690:worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_wandb_readback.py:28:def locate_run_dir(roots, run_id):
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:691:worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_wandb_readback.py:29:    """Find ``<root>/wandb/run-*-<run_id>`` across ``roots``.
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:692:worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_wandb_readback.py:34:    if not run_id:
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:693:worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_wandb_readback.py:40:        matches.extend(sorted(glob.glob(os.path.join(root, "wandb", f"run-*-{run_id}"))))
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:694:worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_wandb_readback.py:43:        return None, [f"no run directory for id {run_id} under any of {list(roots)}"]
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:695:worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_wandb_readback.py:45:        return None, [f"ambiguous run id {run_id}: {matches}"]
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:696:worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_wandb_readback.py:49:def verify_identity(run_dir, run_id, entity=None, project=None, name=None):
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:698:worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_wandb_readback.py:54:    if not os.path.basename(run_dir).endswith(f"-{run_id}"):
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:699:worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_wandb_readback.py:55:        problems.append(f"run directory {os.path.basename(run_dir)} does not carry id {run_id}")
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:703:worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_wandb_readback.py:75:    ap.add_argument("--run-id", required=True)
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:705:worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_wandb_readback.py:83:    run_dir, problems = locate_run_dir(args.root, args.run_id)
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:706:worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_wandb_readback.py:88:    problems = verify_identity(run_dir, args.run_id, args.entity, args.project, args.name)
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:707:worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_wandb_readback.py:92:    print(f"wandb run identity OK: id {args.run_id} at {run_dir} "
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:790:worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-12_18-19-40_C8_8x8_jid3684150_manifest.txt:20:wandb_run_id exp11-C8-1786064131292302937-6d92e299
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:1219:   542	unset WANDB_MODE WANDB_DISABLED WANDB_ENTITY WANDB_RUN_ID WANDB_RESUME WANDB_DIR WANDB_PROJECT WANDB_NAME
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:1240:   563	  ORIG_WANDB_RUN_ID="$(awk '/^wandb_run_id /{print $2}' "$LAUNCH_MANIFEST_LINK" 2>/dev/null)"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:1325:   648	  echo "wandb_run_id ${WANDB_RUN_ID}"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:1379:   702	python3 "$EXPDIR/fa_orbit_wandb_readback.py" --run-id "$WANDB_RUN_ID" \
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:1410:The delta removes the direct `WANDB_RESUME=must` export and preserves the original launch manifest as the restart anchor. I’m now tracing whether any later consumer incorrectly assumes every leg shares that original W&B ID, and whether the readback matcher treats the new ID literally.
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:1412:/usr/bin/bash -lc "grep -RInE 'WANDB_RESUME|resume=[\"'\"']?(must|allow|auto|never)|wandb_run_id|wandb_run|run-id|run_id' . --exclude-dir=.git --exclude='*.log' --exclude='*.out' --exclude='*.ckpt' --exclude='*.safetensors' --exclude='*.pt' --exclude='*.pth' --exclude='*.bin'" in /n/fs/gatrdp/codespace/FLAC
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:1418:/usr/bin/bash -lc "grep -RInE 'WANDB_RESUME|wandb_run_id|wandb_run|run-id|run_id' . --exclude-dir=.git --exclude='*.log' --exclude='*.out' --exclude='*.ckpt' --exclude='*.safetensors' --exclude='*.pt' --exclude='*.pth' --exclude='*.bin'" in /n/fs/gatrdp/codespace/FLAC
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:1420:/usr/bin/bash -lc "grep -RInE 'WANDB_RESUME|wandb_run_id|wandb_run|run-id|run_id' worklog/worklog_yixun/exp_11_fa_orbit_claude src train.py defaults.ini 2>/dev/null | head -200" in /n/fs/gatrdp/codespace/FLAC
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:1422:/usr/bin/bash -lc "grep -n WANDB_RESUME worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch; grep -n wandb_run_id worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch; grep -nE 'wandb|Wandb' train.py | head -80" in /n/fs/gatrdp/codespace/FLAC
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:1424:542:unset WANDB_MODE WANDB_DISABLED WANDB_ENTITY WANDB_RUN_ID WANDB_RESUME WANDB_DIR WANDB_PROJECT WANDB_NAME
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:1425:563:  ORIG_WANDB_RUN_ID="$(awk '/^wandb_run_id /{print $2}' "$LAUNCH_MANIFEST_LINK" 2>/dev/null)"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:1426:648:  echo "wandb_run_id ${WANDB_RUN_ID}"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:1434:/usr/bin/bash -lc "git grep -n -E 'WANDB_RESUME|wandb_run_id|wandb_run|run-id|run_id' -- ':"'!*.log'"' ':"'!*.out'"' ':"'!*.jsonl'"' || true" in /n/fs/gatrdp/codespace/FLAC
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:1436:src/tests/test_exp11_restart_record.py:442:        "wandb_run_id exp11-C8-initial", ""]))
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:1437:src/tests/test_exp11_wandb_readback.py:1:"""Tests for the exp_11 launcher's W&B run-identity readback.
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:1438:src/tests/test_exp11_wandb_readback.py:44:def _make_run(root, run_id=RUN_ID, ts="20260806_164917", meta=None):
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:1439:src/tests/test_exp11_wandb_readback.py:46:    run_dir = os.path.join(root, "wandb", f"run-{ts}-{run_id}")
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:1440:src/tests/test_exp11_wandb_readback.py:89:    _make_run(root, run_id="exp11-C8-someotherrun-aaaaaaaa")
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:1441:src/tests/test_exp11_wandb_readback.py:95:def test_empty_run_id_is_refused(tmp_path):
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:1442:src/tests/test_exp11_wandb_readback.py:129:    run = _make_run(str(tmp_path), run_id="a-different-id", meta={})
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:1443:src/tests/test_exp11_wandb_readback.py:145:    rc = R.main(["--run-id", RUN_ID, "--root", str(repo), "--root", str(wandb_dir),
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:1444:src/tests/test_exp11_wandb_readback.py:151:    rc = R.main(["--run-id", RUN_ID, "--root", str(tmp_path)])
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:1445:worklog/worklog_yixun/exp_11_fa_orbit_claude/commits_fa_orbit.md:25:| 16 | `983a7ff` | launcher residuals — flock ownership (B3), fail-closed manifest-commit binding (B2), pip-freeze/final-tee/preflight-transcript durability (B5), exported W&B entity + post-run run-identity verification (B7), intent manifest before sbatch (NEW-3), safe FIFO (NEW-4) |
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:1446:worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-06_16-47-14_C4L_8x8_jid3646734_manifest.txt:19:wandb_run_id exp11-C4L-1786049318048844980-bd40da20
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:1447:worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-06_18-00-52_C4L_8x8_jid3648568_manifest.txt:19:wandb_run_id exp11-C4L-1786053756799558763-4ae12465
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:1448:worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-06_20-36-00_C4L_8x8_jid3648694_manifest.txt:19:wandb_run_id exp11-C4L-1786063010468957329-bc46fb0a
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:1449:worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-06_20-55-01_C8_8x8_jid3648695_manifest.txt:19:wandb_run_id exp11-C8-1786064131292302937-6d92e299
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:1450:worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-06_20-55-50_C16_8x8_jid3648696_manifest.txt:19:wandb_run_id exp11-C16-1786064168022803862-f44c29b2
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:1451:worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-07_00-16-30_C32_8x8_jid3648697_manifest.txt:19:wandb_run_id exp11-C32-1786076295103433762-98dd1f9b
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:1452:worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-09_12-06-39_VANL_8x8_jid3661520_manifest.txt:20:wandb_run_id exp11-VANL-1786291671381616649-772b3272
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:1453:worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_code_r3_review.md:85:   The parent gate verifies only `wandb.Api().viewer.email` ([fa_orbit_train.sbatch:312](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:312)). Because submission uses `--export=ALL`, ambient `WANDB_MODE`, `WANDB_DISABLED`, `WANDB_ENTITY`, `WANDB_RUN_ID` or `WANDB_RESUME` can redirect, disable, reuse or offline the actual logger while the account check still succeeds. The manifest records only project/display name, not the actual entity and run ID ([fa_orbit_train.sbatch:350](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:350)).
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:1454:worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_content_gate_review.md:1488:   521	unset WANDB_MODE WANDB_DISABLED WANDB_ENTITY WANDB_RUN_ID WANDB_RESUME WANDB_DIR WANDB_PROJECT WANDB_NAME
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:1455:worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_content_gate_review.md:1509:   542	  WANDB_RUN_ID="$(awk '/^wandb_run_id /{print $2}' "$LAUNCH_MANIFEST_LINK" 2>/dev/null)"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:1456:worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_content_gate_review.md:1511:   544	  export WANDB_RUN_ID WANDB_RESUME=must
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:1457:worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_content_gate_review.md:1586:   619	  echo "wandb_run_id ${WANDB_RUN_ID}"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:1458:worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_content_gate_review.md:1654:   673	python3 "$EXPDIR/fa_orbit_wandb_readback.py" --run-id "$WANDB_RUN_ID" \
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:1459:worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_content_gate_review.md:2871:worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_wandb_readback.py:40:        matches.extend(sorted(glob.glob(os.path.join(root, "wandb", f"run-*-{run_id}"))))
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:1460:worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_content_gate_review.md:2873:worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_wandb_readback.py:54:    if not os.path.basename(run_dir).endswith(f"-{run_id}"):
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:1461:worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_content_gate_review.md:2874:worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_wandb_readback.py:55:        problems.append(f"run directory {os.path.basename(run_dir)} does not carry id {run_id}")
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:1462:worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_content_gate_review_r2.md:1233:   273	wandb_run_id exp11-C8-test
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:1463:worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_content_gate_review_r2.md:1312:   352	    fh.write("wandb_run_id exp11-C8-ext\n")
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:1464:worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_content_gate_review_r2.md:2747:690:python3 "$EXPDIR/fa_orbit_wandb_readback.py" --run-id "$WANDB_RUN_ID" \
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:1465:worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:542:unset WANDB_MODE WANDB_DISABLED WANDB_ENTITY WANDB_RUN_ID WANDB_RESUME WANDB_DIR WANDB_PROJECT WANDB_NAME
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:1466:worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:563:  ORIG_WANDB_RUN_ID="$(awk '/^wandb_run_id /{print $2}' "$LAUNCH_MANIFEST_LINK" 2>/dev/null)"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:1467:worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:648:  echo "wandb_run_id ${WANDB_RUN_ID}"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:1468:worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:702:python3 "$EXPDIR/fa_orbit_wandb_readback.py" --run-id "$WANDB_RUN_ID" \
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:1469:worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh:287:wandb_run_id exp11-C8-test
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:1470:worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh:366:    fh.write("wandb_run_id exp11-C8-ext\n")
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:1471:worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_wandb_readback.py:28:def locate_run_dir(roots, run_id):
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:1472:worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_wandb_readback.py:29:    """Find ``<root>/wandb/run-*-<run_id>`` across ``roots``.
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:1473:worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_wandb_readback.py:34:    if not run_id:
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:1474:worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_wandb_readback.py:40:        matches.extend(sorted(glob.glob(os.path.join(root, "wandb", f"run-*-{run_id}"))))
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:1475:worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_wandb_readback.py:43:        return None, [f"no run directory for id {run_id} under any of {list(roots)}"]
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:1476:worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_wandb_readback.py:45:        return None, [f"ambiguous run id {run_id}: {matches}"]
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:1477:worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_wandb_readback.py:49:def verify_identity(run_dir, run_id, entity=None, project=None, name=None):
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:1478:worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_wandb_readback.py:54:    if not os.path.basename(run_dir).endswith(f"-{run_id}"):
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:1479:worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_wandb_readback.py:55:        problems.append(f"run directory {os.path.basename(run_dir)} does not carry id {run_id}")
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:1480:worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_wandb_readback.py:75:    ap.add_argument("--run-id", required=True)
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:1481:worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_wandb_readback.py:83:    run_dir, problems = locate_run_dir(args.root, args.run_id)
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:1482:worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_wandb_readback.py:88:    problems = verify_identity(run_dir, args.run_id, args.entity, args.project, args.name)
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:1483:worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_wandb_readback.py:92:    print(f"wandb run identity OK: id {args.run_id} at {run_dir} "
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:1484:worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_worklog.md:100:- **Change** — NEW-1: `FLAC_AR_VANCKPT.json` (canonical + gc:true ×2, parsed-delta-tested); CKPT4 family fully retired; matrix = 12 all-ckpt cells. NEW-2: OUTPUT_ROOT pinned to production literal under Slurm (both scripts + submitters). B2/B3/B5/B7 residuals closed (fail-closed commit binding, flock ownership, checked pip-freeze/dual-copy/transcript with class-7, WANDB_ENTITY export + post-run run-identity verification). NEW-3 intent-before-sbatch with scancel-on-failure; NEW-4 FIFO hygiene; NEW-5 comment corrected + 2 s liveness bound restored.
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:1485:worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_worklog.md:136:- **Re-pin** — `ea94995` (floors/limits/manifest sha from batched evidence). **Launcher fixes** `71054cf` (wandb readback by run-id glob — PL save_dir overrides WANDB_DIR, train.py:165; scontrol-derived transcript path fail-closed; separator hygiene) + 15 readback unit tests + 6 guard cases (suite: 208 pytest + 16 guards green).
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:1486:worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_train.sbatch:754:unset WANDB_MODE WANDB_DISABLED WANDB_ENTITY WANDB_RUN_ID WANDB_RESUME WANDB_DIR WANDB_PROJECT WANDB_NAME
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:1487:worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_train.sbatch:775:  WANDB_RUN_ID="$(awk '/^wandb_run_id /{print $2}' "$LAUNCH_MANIFEST_LINK" 2>/dev/null)"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:1488:worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_train.sbatch:777:  export WANDB_RUN_ID WANDB_RESUME=must
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:1489:worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_train.sbatch:864:  echo "wandb_run_id ${WANDB_RUN_ID}"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:1490:worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_train.sbatch:895: vae_sha, save_dir, wandb_run_id) = sys.argv[1:13]
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:1491:worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_train.sbatch:907:    "wandb_run_id": wandb_run_id,
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:1492:worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_train.sbatch:1089:python3 "$(snap fa_orbit_wandb_readback.py)" --run-id "$WANDB_RUN_ID" \
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:1524:    28	def locate_run_dir(roots, run_id):
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:1525:    29	    """Find ``<root>/wandb/run-*-<run_id>`` across ``roots``.
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:1530:    34	    if not run_id:
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:1536:    40	        matches.extend(sorted(glob.glob(os.path.join(root, "wandb", f"run-*-{run_id}"))))
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:1539:    43	        return None, [f"no run directory for id {run_id} under any of {list(roots)}"]
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:1541:    45	        return None, [f"ambiguous run id {run_id}: {matches}"]
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:1545:    49	def verify_identity(run_dir, run_id, entity=None, project=None, name=None):
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:1550:    54	    if not os.path.basename(run_dir).endswith(f"-{run_id}"):
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:1551:    55	        problems.append(f"run directory {os.path.basename(run_dir)} does not carry id {run_id}")
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:1571:    75	    ap.add_argument("--run-id", required=True)
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:1579:    83	    run_dir, problems = locate_run_dir(args.root, args.run_id)
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:1584:    88	    problems = verify_identity(run_dir, args.run_id, args.entity, args.project, args.name)
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:1588:    92	    print(f"wandb run identity OK: id {args.run_id} at {run_dir} "
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:1595:     1	"""Tests for the exp_11 launcher's W&B run-identity readback.
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:1638:    44	def _make_run(root, run_id=RUN_ID, ts="20260806_164917", meta=None):
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:1640:    46	    run_dir = os.path.join(root, "wandb", f"run-{ts}-{run_id}")
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:1683:    89	    _make_run(root, run_id="exp11-C8-someotherrun-aaaaaaaa")
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:1689:    95	def test_empty_run_id_is_refused(tmp_path):
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:1723:   129	    run = _make_run(str(tmp_path), run_id="a-different-id", meta={})
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:1739:   145	    rc = R.main(["--run-id", RUN_ID, "--root", str(repo), "--root", str(wandb_dir),
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_wandb_restart_review.md:1745:   151	    rc = R.main(["--run-id", RUN_ID, "--root", str(tmp_path)])
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-06_20-36-00_C4L_8x8_jid3648694_manifest.txt:19:wandb_run_id exp11-C4L-1786063010468957329-bc46fb0a
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_content_gate_review_r2.md:1233:   273	wandb_run_id exp11-C8-test
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_content_gate_review_r2.md:1312:   352	    fh.write("wandb_run_id exp11-C8-ext\n")
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_content_gate_review_r2.md:2747:690:python3 "$EXPDIR/fa_orbit_wandb_readback.py" --run-id "$WANDB_RUN_ID" \
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-12_18-11-38_C4L_8x8_jid3684149_manifest.txt:20:wandb_run_id exp11-C4L-1786063010468957329-bc46fb0a
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_content_gate_review.md:1488:   521	unset WANDB_MODE WANDB_DISABLED WANDB_ENTITY WANDB_RUN_ID WANDB_RESUME WANDB_DIR WANDB_PROJECT WANDB_NAME
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_content_gate_review.md:1509:   542	  WANDB_RUN_ID="$(awk '/^wandb_run_id /{print $2}' "$LAUNCH_MANIFEST_LINK" 2>/dev/null)"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_content_gate_review.md:1511:   544	  export WANDB_RUN_ID WANDB_RESUME=must
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_content_gate_review.md:1586:   619	  echo "wandb_run_id ${WANDB_RUN_ID}"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_content_gate_review.md:1654:   673	python3 "$EXPDIR/fa_orbit_wandb_readback.py" --run-id "$WANDB_RUN_ID" \
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_content_gate_review.md:2871:worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_wandb_readback.py:40:        matches.extend(sorted(glob.glob(os.path.join(root, "wandb", f"run-*-{run_id}"))))
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_content_gate_review.md:2873:worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_wandb_readback.py:54:    if not os.path.basename(run_dir).endswith(f"-{run_id}"):
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_content_gate_review.md:2874:worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_wandb_readback.py:55:        problems.append(f"run directory {os.path.basename(run_dir)} does not carry id {run_id}")
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_wandb_readback.py:28:def locate_run_dir(roots, run_id):
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_wandb_readback.py:29:    """Find ``<root>/wandb/run-*-<run_id>`` across ``roots``.
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_wandb_readback.py:34:    if not run_id:
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_wandb_readback.py:40:        matches.extend(sorted(glob.glob(os.path.join(root, "wandb", f"run-*-{run_id}"))))
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_wandb_readback.py:43:        return None, [f"no run directory for id {run_id} under any of {list(roots)}"]
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_wandb_readback.py:45:        return None, [f"ambiguous run id {run_id}: {matches}"]
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_wandb_readback.py:49:def verify_identity(run_dir, run_id, entity=None, project=None, name=None):
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_wandb_readback.py:54:    if not os.path.basename(run_dir).endswith(f"-{run_id}"):
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_wandb_readback.py:55:        problems.append(f"run directory {os.path.basename(run_dir)} does not carry id {run_id}")
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_wandb_readback.py:75:    ap.add_argument("--run-id", required=True)
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_wandb_readback.py:83:    run_dir, problems = locate_run_dir(args.root, args.run_id)
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_wandb_readback.py:88:    problems = verify_identity(run_dir, args.run_id, args.entity, args.project, args.name)
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_wandb_readback.py:92:    print(f"wandb run identity OK: id {args.run_id} at {run_dir} "
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-12_18-19-40_C8_8x8_jid3684150_manifest.txt:20:wandb_run_id exp11-C8-1786064131292302937-6d92e299
src/tests/test_exp11_restart_record.py:442:        "wandb_run_id exp11-C8-initial", ""]))
src/tests/test_exp11_wandb_readback.py:1:"""Tests for the exp_11 launcher's W&B run-identity readback.
src/tests/test_exp11_wandb_readback.py:44:def _make_run(root, run_id=RUN_ID, ts="20260806_164917", meta=None):
src/tests/test_exp11_wandb_readback.py:46:    run_dir = os.path.join(root, "wandb", f"run-{ts}-{run_id}")
src/tests/test_exp11_wandb_readback.py:89:    _make_run(root, run_id="exp11-C8-someotherrun-aaaaaaaa")
src/tests/test_exp11_wandb_readback.py:95:def test_empty_run_id_is_refused(tmp_path):
src/tests/test_exp11_wandb_readback.py:129:    run = _make_run(str(tmp_path), run_id="a-different-id", meta={})
src/tests/test_exp11_wandb_readback.py:145:    rc = R.main(["--run-id", RUN_ID, "--root", str(repo), "--root", str(wandb_dir),
src/tests/test_exp11_wandb_readback.py:151:    rc = R.main(["--run-id", RUN_ID, "--root", str(tmp_path)])

exec
/usr/bin/bash -lc "find /n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/prefigure -maxdepth 2 -type f -print 2>/dev/null; nl -ba train.py | sed -n '145,205p'; grep -RInE 'def push_wandb_config|wandb.init|WANDB_RESUME|resume=' /n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/prefigure /n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/pytorch_lightning/loggers/wandb.py 2>/dev/null | head -160" in /n/fs/gatrdp/codespace/FLAC
 succeeded in 931ms:
/n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/prefigure/__pycache__/__init__.cpython-310.pyc
/n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/prefigure/__pycache__/ofc.cpython-310.pyc
/n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/prefigure/__pycache__/prefigure.cpython-310.pyc
/n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/prefigure/ofc.py
/n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/prefigure/prefigure.py
/n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/prefigure/__init__.py
   145	        disc_weights = {k.replace('discriminator.', ''): v for k, v in disc_weights.items()}
   146	        weights = {k: v for k, v in weights.items() if 'discriminator' not in k}
   147	        weights = {k: v for k, v in weights.items() if 'losses' not in k}
   148	        model.load_state_dict(weights, strict=True)
   149	
   150	    if args.remove_pretransform_weight_norm == "pre_load":
   151	        remove_weight_norm_from_model(model.pretransform)
   152	
   153	    if args.pretransform_ckpt_path:
   154	        model.pretransform.load_state_dict(load_ckpt_state_dict(args.pretransform_ckpt_path))
   155	
   156	    # Remove weight_norm from the pretransform if specified
   157	    if args.remove_pretransform_weight_norm == "post_load":
   158	        remove_weight_norm_from_model(model.pretransform)
   159	
   160	    training_wrapper = create_training_wrapper_from_config(model_config, model)
   161	
   162	    exc_callback = ExceptionCallback()
   163	
   164	    if args.logger == 'wandb':
   165	        logger = pl.loggers.WandbLogger(project=args.name, name=args.experiment_name)
   166	        logger.watch(training_wrapper)
   167	    
   168	        if args.save_dir and isinstance(logger.experiment.id, str):
   169	            checkpoint_dir = os.path.join(args.save_dir, logger.experiment.project, logger.experiment.name, "checkpoints") 
   170	        else:
   171	            checkpoint_dir = None
   172	    elif args.logger == 'comet':
   173	        logger = pl.loggers.CometLogger(project_name=args.name)
   174	        if args.save_dir and isinstance(logger.version, str):
   175	            checkpoint_dir = os.path.join(args.save_dir, logger.name, logger.version, "checkpoints") 
   176	        else:
   177	            checkpoint_dir = args.save_dir if args.save_dir else None
   178	    else:
   179	        logger = None
   180	        checkpoint_dir = args.save_dir if args.save_dir else None
   181	        
   182	    ckpt_callback = pl.callbacks.ModelCheckpoint(every_n_train_steps=args.checkpoint_every, dirpath=checkpoint_dir, save_top_k=-1)
   183	    save_model_config_callback = ModelConfigEmbedderCallback(model_config)
   184	        
   185	    #Combine args and config dicts
   186	    args_dict = vars(args)
   187	    args_dict.update({"model_config": model_config})
   188	    args_dict.update({"dataset_config": dataset_config})
   189	    args_dict.update({"val_dataset_config": val_dataset_config})
   190	
   191	    # Logger 
   192	    if args.logger == 'wandb':
   193	        push_wandb_config(logger, args_dict)
   194	    elif args.logger == 'comet':
   195	        logger.log_hyperparams(args_dict)
   196	
   197	    #Set multi-GPU strategy if specified
   198	    if args.strategy:
   199	        if args.strategy == "deepspeed":
   200	            from pytorch_lightning.strategies import DeepSpeedStrategy
   201	            strategy = DeepSpeedStrategy(stage=2,
   202	                                        contiguous_gradients=True,
   203	                                        overlap_comm=True,
   204	                                        reduce_scatter=True,
   205	                                        reduce_bucket_size=5e8,
/n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/prefigure/prefigure.py:101:def push_wandb_config(wandb_logger, args, omit=[]): 
/n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/pytorch_lightning/loggers/wandb.py:68:    A new W&B run will be created when training starts if you have not created one manually before with `wandb.init()`.
/n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/pytorch_lightning/loggers/wandb.py:166:        run = wandb.init(project="MNIST")
/n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/pytorch_lightning/loggers/wandb.py:273:        \**kwargs: Arguments passed to :func:`wandb.init` like `entity`, `group`, `tags`, etc.
/n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/pytorch_lightning/loggers/wandb.py:327:        # set wandb init arguments
/n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/pytorch_lightning/loggers/wandb.py:328:        self._wandb_init: Dict[str, Any] = {
/n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/pytorch_lightning/loggers/wandb.py:336:        self._wandb_init.update(**kwargs)
/n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/pytorch_lightning/loggers/wandb.py:338:        self._project = self._wandb_init.get("project")
/n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/pytorch_lightning/loggers/wandb.py:339:        self._save_dir = self._wandb_init.get("dir")
/n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/pytorch_lightning/loggers/wandb.py:340:        self._name = self._wandb_init.get("name")
/n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/pytorch_lightning/loggers/wandb.py:341:        self._id = self._wandb_init.get("id")
/n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/pytorch_lightning/loggers/wandb.py:399:                self._experiment = wandb.init(**self._wandb_init)

2026-08-12T22:31:00.570880Z ERROR codex_models_manager::manager: failed to renew cache TTL: missing field `base_instructions` at line 94 column 5
exec
/usr/bin/bash -lc "nl -ba /n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/prefigure/prefigure.py | sed -n '90,125p'; nl -ba /n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/pytorch_lightning/loggers/wandb.py | sed -n '300,420p'; grep -RInE 'resume *=|resume:' .wandb wandb/settings* outputs_FLAC/exp11_C4L/wandb/settings* outputs_FLAC/exp11_C8/wandb/settings* 2>/dev/null | head -100" in /n/fs/gatrdp/codespace/FLAC
 succeeded in 248ms:
    90	       wandb_config is the url of one of your runs"""
    91	    api = wandb.Api()  # might get prompted for api key login the first time
    92	    splits = wandb_config.split('/')
    93	    entity, project, run_id = splits[3], splits[4], splits[-1].split('?')[0]
    94	    run = api.run(f"{entity}/{project}/{run_id}")
    95	    for key, value in run.config.items():
    96	        if 'OMITTED' != value: defaults[key] = arg_eval(value)
    97	    return defaults
    98	
    99	
   100	
   101	def push_wandb_config(wandb_logger, args, omit=[]): 
   102	    """
   103	    save config to wandb (for possible retrieval later)
   104	    Omit: list of args you don't want pushed to wandb; will push an empty string for these
   105	    """
   106	    if hasattr(wandb_logger.experiment.config, 'update'): #On multi-GPU runs, only process rank 0 has this attribute!
   107	        copy_args = copy.deepcopy(args)
   108	        for var_str in omit:  # don't push certain reserved settings to wandb
   109	            if hasattr(copy_args, var_str):
   110	                setattr(copy_args, var_str, 'OMITTED')
   111	        wandb_logger.experiment.config.update(copy_args)
   112	
   113	
   114	def get_all_args(defaults_file=DEFAULTS_FILE):
   115	    " Config setup."
   116	    args = {}
   117	    #   1. Default settings are in defaults ini (or some other config) file
   118	    defaults, defaults_text = read_defaults(defaults_file=defaults_file)
   119	    args = setup_args(defaults, defaults_text=defaults_text)  
   120	
   121	    #   2. if --wandb-config is given, pull config from wandb to override defaults
   122	    if args.wandb_config is not None:
   123	        defaults = pull_wandb_config(args.wandb_config, defaults) # 2.
   124	
   125	    #   3. Any new command-line arguments override whatever was set earlier
   300	    ) -> None:
   301	        if not _WANDB_AVAILABLE:
   302	            raise ModuleNotFoundError(str(_WANDB_AVAILABLE))
   303	
   304	        if offline and log_model:
   305	            raise MisconfigurationException(
   306	                f"Providing log_model={log_model} and offline={offline} is an invalid configuration"
   307	                " since model checkpoints cannot be uploaded in offline mode.\n"
   308	                "Hint: Set `offline=False` to log your model."
   309	            )
   310	
   311	        super().__init__()
   312	        self._offline = offline
   313	        self._log_model = log_model
   314	        self._prefix = prefix
   315	        self._experiment = experiment
   316	        self._logged_model_time: Dict[str, float] = {}
   317	        self._checkpoint_callback: Optional[ModelCheckpoint] = None
   318	
   319	        # paths are processed as strings
   320	        if save_dir is not None:
   321	            save_dir = os.fspath(save_dir)
   322	        elif dir is not None:
   323	            dir = os.fspath(dir)
   324	
   325	        project = project or os.environ.get("WANDB_PROJECT", "lightning_logs")
   326	
   327	        # set wandb init arguments
   328	        self._wandb_init: Dict[str, Any] = {
   329	            "name": name,
   330	            "project": project,
   331	            "dir": save_dir or dir,
   332	            "id": version or id,
   333	            "resume": "allow",
   334	            "anonymous": ("allow" if anonymous else None),
   335	        }
   336	        self._wandb_init.update(**kwargs)
   337	        # extract parameters
   338	        self._project = self._wandb_init.get("project")
   339	        self._save_dir = self._wandb_init.get("dir")
   340	        self._name = self._wandb_init.get("name")
   341	        self._id = self._wandb_init.get("id")
   342	        self._checkpoint_name = checkpoint_name
   343	
   344	    def __getstate__(self) -> Dict[str, Any]:
   345	        import wandb
   346	
   347	        # Hack: If the 'spawn' launch method is used, the logger will get pickled and this `__getstate__` gets called.
   348	        # We create an experiment here in the main process, and attach to it in the worker process.
   349	        # Using wandb-service, we persist the same experiment even if multiple `Trainer.fit/test/validate` calls
   350	        # are made.
   351	        wandb.require("service")
   352	        _ = self.experiment
   353	
   354	        state = self.__dict__.copy()
   355	        # args needed to reload correct experiment
   356	        if self._experiment is not None:
   357	            state["_id"] = getattr(self._experiment, "id", None)
   358	            state["_attach_id"] = getattr(self._experiment, "_attach_id", None)
   359	            state["_name"] = self._experiment.name
   360	
   361	        # cannot be pickled
   362	        state["_experiment"] = None
   363	        return state
   364	
   365	    @property
   366	    @rank_zero_experiment
   367	    def experiment(self) -> Union["Run", "RunDisabled"]:
   368	        r"""Actual wandb object. To use wandb features in your :class:`~pytorch_lightning.core.LightningModule` do the
   369	        following.
   370	
   371	        Example::
   372	
   373	        .. code-block:: python
   374	
   375	            self.logger.experiment.some_wandb_function()
   376	
   377	        """
   378	        import wandb
   379	        from wandb.sdk.lib import RunDisabled
   380	        from wandb.wandb_run import Run
   381	
   382	        if self._experiment is None:
   383	            if self._offline:
   384	                os.environ["WANDB_MODE"] = "dryrun"
   385	
   386	            attach_id = getattr(self, "_attach_id", None)
   387	            if wandb.run is not None:
   388	                # wandb process already created in this instance
   389	                rank_zero_warn(
   390	                    "There is a wandb run already in progress and newly created instances of `WandbLogger` will reuse"
   391	                    " this run. If this is not desired, call `wandb.finish()` before instantiating `WandbLogger`."
   392	                )
   393	                self._experiment = wandb.run
   394	            elif attach_id is not None and hasattr(wandb, "_attach"):
   395	                # attach to wandb process referenced
   396	                self._experiment = wandb._attach(attach_id)
   397	            else:
   398	                # create new wandb process
   399	                self._experiment = wandb.init(**self._wandb_init)
   400	
   401	                # define default x-axis
   402	                if isinstance(self._experiment, (Run, RunDisabled)) and getattr(
   403	                    self._experiment, "define_metric", None
   404	                ):
   405	                    self._experiment.define_metric("trainer/global_step")
   406	                    self._experiment.define_metric("*", step_metric="trainer/global_step", step_sync=True)
   407	
   408	        return self._experiment
   409	
   410	    def watch(self, model: nn.Module, log: str = "gradients", log_freq: int = 100, log_graph: bool = True) -> None:
   411	        self.experiment.watch(model, log=log, log_freq=log_freq, log_graph=log_graph)
   412	
   413	    @rank_zero_only
   414	    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace]) -> None:  # type: ignore[override]
   415	        params = _convert_params(params)
   416	        params = _sanitize_callable_params(params)
   417	        self.experiment.config.update(params, allow_val_change=True)
   418	
   419	    @rank_zero_only
   420	    def log_metrics(self, metrics: Mapping[str, float], step: Optional[int] = None) -> None:

exec
/usr/bin/bash -lc "sed -n '110,165p' outputs_FLAC/exp11_C4L/fa_orbit_2026-08-12_18-11-38_C4L_8x8_jid3684149_train.log; grep -nE 'wandb:|resume|ConfigError|run ID|Run data' outputs_FLAC/exp11_C4L/fa_orbit_2026-08-12_18-11-38_C4L_8x8_jid3684149_train.log | head -100; grep -nE 'wandb:|resume|ConfigError|run ID|Run data' outputs_FLAC/exp11_C8/fa_orbit_2026-08-12_18-19-40_C8_8x8_jid3684150_train.log 2>/dev/null | head -100" in /n/fs/gatrdp/codespace/FLAC
 succeeded in 217ms:
[rank: 6] Seed set to 42
Initializing distributed: GLOBAL_RANK: 6, MEMBER: 7/8
[rank: 5] Seed set to 42
Initializing distributed: GLOBAL_RANK: 5, MEMBER: 6/8
wandb: [wandb.login()] Loaded credentials for https://api.wandb.ai from /u/yh4742/.netrc.
[rank: 4] Seed set to 42
Initializing distributed: GLOBAL_RANK: 4, MEMBER: 5/8
[rank: 1] Seed set to 42
Initializing distributed: GLOBAL_RANK: 1, MEMBER: 2/8
[rank: 7] Seed set to 42
Initializing distributed: GLOBAL_RANK: 7, MEMBER: 8/8
wandb: Currently logged in as: yh4742 (yh4742-princeton-university) to https://api.wandb.ai. Use `wandb login --relogin` to force relogin
[rank: 2] Seed set to 42
Initializing distributed: GLOBAL_RANK: 2, MEMBER: 3/8
[rank: 3] Seed set to 42
Initializing distributed: GLOBAL_RANK: 3, MEMBER: 4/8
wandb: Tracking run with wandb version 0.26.1
wandb: Run data is saved locally in wandb/run-20260812_181853-exp11-C4L-1786063010468957329-bc46fb0a
wandb: Run `wandb offline` to turn off syncing.
wandb: Resuming run exp11_C4L
wandb: ⭐️ View project at https://wandb.ai/yh4742-princeton-university/FLAC_exp11_C4L
wandb: 🚀 View run at https://wandb.ai/yh4742-princeton-university/FLAC_exp11_C4L/runs/exp11-C4L-1786063010468957329-bc46fb0a
wandb: logging graph, to disable use `wandb.watch(log_graph=False)`
wandb: ERROR Attempted to change value of key "max_steps" from 40000 to 100000
wandb: ERROR If you really want to do this, pass allow_val_change=True to config.update()
Traceback (most recent call last):
  File "/n/fs/gatrdp/codespace/FLAC/train.py", line 233, in <module>
    main()
  File "/n/fs/gatrdp/codespace/FLAC/train.py", line 193, in main
    push_wandb_config(logger, args_dict)
  File "/n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/prefigure/prefigure.py", line 111, in push_wandb_config
    wandb_logger.experiment.config.update(copy_args)
  File "/n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/wandb/sdk/wandb_config.py", line 188, in update
    sanitized = self._update(d, allow_val_change)
  File "/n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/wandb/sdk/wandb_config.py", line 181, in _update
    sanitized = self._sanitize_dict(
  File "/n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/wandb/sdk/wandb_config.py", line 268, in _sanitize_dict
    k, v = self._sanitize(k, v, allow_val_change)
  File "/n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/wandb/sdk/wandb_config.py", line 292, in _sanitize
    raise config_util.ConfigError(
wandb.sdk.lib.config_util.ConfigError: Attempted to change value of key "max_steps" from 40000 to 100000
If you really want to do this, pass allow_val_change=True to config.update()
[1;34mwandb[0m: 
[1;34mwandb[0m: 🚀 View run [33mexp11_C4L[0m at: [34mhttps://wandb.ai/yh4742-princeton-university/FLAC_exp11_C4L/runs/exp11-C4L-1786063010468957329-bc46fb0a[0m
W0812 18:19:10.010000 782144 site-packages/torch/distributed/elastic/multiprocessing/api.py:900] Sending process 782215 closing signal SIGTERM
W0812 18:19:10.012000 782144 site-packages/torch/distributed/elastic/multiprocessing/api.py:900] Sending process 782216 closing signal SIGTERM
W0812 18:19:10.030000 782144 site-packages/torch/distributed/elastic/multiprocessing/api.py:900] Sending process 782217 closing signal SIGTERM
W0812 18:19:10.031000 782144 site-packages/torch/distributed/elastic/multiprocessing/api.py:900] Sending process 782218 closing signal SIGTERM
W0812 18:19:10.032000 782144 site-packages/torch/distributed/elastic/multiprocessing/api.py:900] Sending process 782219 closing signal SIGTERM
W0812 18:19:10.044000 782144 site-packages/torch/distributed/elastic/multiprocessing/api.py:900] Sending process 782220 closing signal SIGTERM
W0812 18:19:10.047000 782144 site-packages/torch/distributed/elastic/multiprocessing/api.py:900] Sending process 782221 closing signal SIGTERM
E0812 18:19:10.808000 782144 site-packages/torch/distributed/elastic/multiprocessing/api.py:874] failed (exitcode: 1) local_rank: 0 (pid: 782214) of binary: /n/fs/gatrdp/envs/flac/bin/python3.10
Traceback (most recent call last):
  File "/n/fs/gatrdp/envs/flac/bin/torchrun", line 6, in <module>
    sys.exit(main())
  File "/n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/torch/distributed/elastic/multiprocessing/errors/__init__.py", line 355, in wrapper
109:wandb: WARNING The anonymous setting has no effect and will be removed in a future version.
114:wandb: [wandb.login()] Loaded credentials for https://api.wandb.ai from /u/yh4742/.netrc.
121:wandb: Currently logged in as: yh4742 (yh4742-princeton-university) to https://api.wandb.ai. Use `wandb login --relogin` to force relogin
126:wandb: Tracking run with wandb version 0.26.1
127:wandb: Run data is saved locally in wandb/run-20260812_181853-exp11-C4L-1786063010468957329-bc46fb0a
128:wandb: Run `wandb offline` to turn off syncing.
129:wandb: Resuming run exp11_C4L
130:wandb: ⭐️ View project at https://wandb.ai/yh4742-princeton-university/FLAC_exp11_C4L
131:wandb: 🚀 View run at https://wandb.ai/yh4742-princeton-university/FLAC_exp11_C4L/runs/exp11-C4L-1786063010468957329-bc46fb0a
132:wandb: logging graph, to disable use `wandb.watch(log_graph=False)`
133:wandb: ERROR Attempted to change value of key "max_steps" from 40000 to 100000
134:wandb: ERROR If you really want to do this, pass allow_val_change=True to config.update()
149:    raise config_util.ConfigError(
150:wandb.sdk.lib.config_util.ConfigError: Attempted to change value of key "max_steps" from 40000 to 100000
195:restart: fa_orbit_submit.sh C4L --resume outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=8-step=40000.ckpt --expected-step <step>
115:wandb: WARNING The anonymous setting has no effect and will be removed in a future version.
122:wandb: [wandb.login()] Loaded credentials for https://api.wandb.ai from /u/yh4742/.netrc.
123:wandb: Currently logged in as: yh4742 (yh4742-princeton-university) to https://api.wandb.ai. Use `wandb login --relogin` to force relogin
126:wandb: Tracking run with wandb version 0.26.1
127:wandb: Run data is saved locally in wandb/run-20260812_182038-exp11-C8-1786064131292302937-6d92e299
128:wandb: Run `wandb offline` to turn off syncing.
129:wandb: Resuming run exp11_C8
130:wandb: ⭐️ View project at https://wandb.ai/yh4742-princeton-university/FLAC_exp11_C8
131:wandb: 🚀 View run at https://wandb.ai/yh4742-princeton-university/FLAC_exp11_C8/runs/exp11-C8-1786064131292302937-6d92e299
132:wandb: logging graph, to disable use `wandb.watch(log_graph=False)`
133:wandb: ERROR Attempted to change value of key "max_steps" from 40000 to 100000
134:wandb: ERROR If you really want to do this, pass allow_val_change=True to config.update()
149:    raise config_util.ConfigError(
150:wandb.sdk.lib.config_util.ConfigError: Attempted to change value of key "max_steps" from 40000 to 100000
195:restart: fa_orbit_submit.sh C8 --resume outputs_FLAC/exp11_C8/FLAC_exp11_C8/exp11_C8/checkpoints/epoch=8-step=40000.ckpt --expected-step <step>

exec
/usr/bin/bash -lc "grep -RInE 'WANDB_RESUME|resume.*allow|_resume' /n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/wandb/sdk /n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/wandb/env.py 2>/dev/null | head -240" in /n/fs/gatrdp/codespace/FLAC
 succeeded in 16758ms:
/n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/wandb/sdk/data_types/utils.py:195:            val._load_incremental_table_state_from_resumed_run(run, key)
/n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/wandb/sdk/data_types/table.py:352:    def _load_incremental_table_state_from_resumed_run(self, run: LocalRun, key: str):
/n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/wandb/sdk/wandb_init.py:613:        self._interface.publish_resume()
/n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/wandb/sdk/wandb_init.py:1286:    resume: bool | Literal["allow", "never", "must", "auto"] | None = None,
/n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/wandb/sdk/interface/interface.py:810:    def publish_resume(self) -> None:
/n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/wandb/sdk/interface/interface.py:812:        self._publish_resume(resume)
/n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/wandb/sdk/interface/interface.py:815:    def _publish_resume(self, resume: pb.ResumeRequest) -> None:
/n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/wandb/sdk/interface/interface_shared.py:331:    def _publish_resume(self, resume: pb.ResumeRequest) -> None:
/n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/wandb/sdk/wandb_settings.py:367:    resume: Optional[Literal["allow", "must", "never", "auto"]] = None
/n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/wandb/sdk/wandb_settings.py:1257:    def validate_resume(cls, value):
/n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/wandb/sdk/wandb_settings.py:1270:    def validate_resume_from(cls, value, values) -> Optional[RunMoment]:
/n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/wandb/sdk/internal/sender_config.py:48:    def merge_resumed_config(self, old_config_tree: dict[str, Any]) -> None:
/n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/wandb/sdk/internal/internal_api.py:1016:    def run_resume_status(
/n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/wandb/sdk/internal/handler.py:676:    def handle_request_resume(self, record: Record) -> None:
/n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/wandb/sdk/internal/sender.py:218:    _resume_state: ResumeState
/n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/wandb/sdk/internal/sender.py:277:        self._resume_state = ResumeState()
/n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/wandb/sdk/internal/sender.py:680:    def _setup_resume(self, run: RunRecord) -> wandb_internal_pb2.ErrorInfo | None:
/n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/wandb/sdk/internal/sender.py:693:        resume_status = self._api.run_resume_status(
/n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/wandb/sdk/internal/sender.py:707:                    " If you are trying to start a new run, please omit the `resume` argument or use `resume='allow'`."
/n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/wandb/sdk/internal/sender.py:744:                self._resume_state.wandb_runtime = new_runtime
/n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/wandb/sdk/internal/sender.py:757:        self._resume_state.runtime = max(events_rt, history_rt)
/n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/wandb/sdk/internal/sender.py:760:        self._resume_state.step = last_step + 1 if history_line_count > 0 else last_step
/n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/wandb/sdk/internal/sender.py:761:        self._resume_state.history = history_line_count
/n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/wandb/sdk/internal/sender.py:762:        self._resume_state.events = resume_status["eventsLineCount"]
/n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/wandb/sdk/internal/sender.py:763:        self._resume_state.output = resume_status["logLineCount"]
/n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/wandb/sdk/internal/sender.py:764:        self._resume_state.config = config
/n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/wandb/sdk/internal/sender.py:765:        self._resume_state.summary = summary
/n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/wandb/sdk/internal/sender.py:766:        self._resume_state.tags = tags
/n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/wandb/sdk/internal/sender.py:767:        self._resume_state.resumed = True
/n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/wandb/sdk/internal/sender.py:768:        logger.info(f"configured resuming with: {self._resume_state}")
/n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/wandb/sdk/internal/sender.py:826:        self._resume_state.step = first_step
/n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/wandb/sdk/internal/sender.py:827:        self._resume_state.history = server_run.get("historyLineCount", 0)
/n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/wandb/sdk/internal/sender.py:841:        self._resume_state.history = self._rewind_response.get("historyLineCount", 0)
/n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/wandb/sdk/internal/sender.py:842:        self._resume_state.config = json.loads(
/n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/wandb/sdk/internal/sender.py:852:        self._resume_state.step = first_step
/n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/wandb/sdk/internal/sender.py:897:        do_resume = bool(self._settings.resume)
/n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/wandb/sdk/internal/sender.py:899:        num_resume_options_set = sum([do_fork, do_rewind, do_resume])
/n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/wandb/sdk/internal/sender.py:900:        if num_resume_options_set > 1:
/n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/wandb/sdk/internal/sender.py:915:            if do_resume:
/n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/wandb/sdk/internal/sender.py:916:                error = self._setup_resume(run)
/n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/wandb/sdk/internal/sender.py:926:        if self._resume_state.config is not None:
/n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/wandb/sdk/internal/sender.py:927:            self._consolidated_config.merge_resumed_config(
/n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/wandb/sdk/internal/sender.py:928:                config_util.dict_strip_value_dict(self._resume_state.config)
/n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/wandb/sdk/internal/sender.py:972:    def _update_resume_state(self, is_rewinding: bool, inserted: bool):
/n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/wandb/sdk/internal/sender.py:974:        if self._resume_state.resumed:
/n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/wandb/sdk/internal/sender.py:976:            if self._resume_state.wandb_runtime is not None:
/n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/wandb/sdk/internal/sender.py:977:                self._run.runtime = self._resume_state.wandb_runtime
/n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/wandb/sdk/internal/sender.py:979:            # because is_rewinding is mutually exclusive with self._resume_state.resumed,
/n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/wandb/sdk/internal/sender.py:1000:        ) - self._resume_state.runtime
/n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/wandb/sdk/internal/sender.py:1004:        if self._resume_state and self._resume_state.tags and not run.tags:
/n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/wandb/sdk/internal/sender.py:1005:            run.tags.extend(self._resume_state.tags)
/n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/wandb/sdk/internal/sender.py:1037:        if self._resume_state.resumed and is_rewinding:
/n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/wandb/sdk/internal/sender.py:1046:        self._update_resume_state(is_rewinding, inserted)
/n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/wandb/sdk/internal/sender.py:1047:        self._run.starting_step = self._resume_state.step
/n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/wandb/sdk/internal/sender.py:1050:        if self._resume_state.summary is not None:
/n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/wandb/sdk/internal/sender.py:1052:                self._interface._make_summary_from_dict(self._resume_state.summary)
/n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/wandb/sdk/internal/sender.py:1100:            file_stream.JsonlFilePolicy(start_chunk_id=self._resume_state.history),
/n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/wandb/sdk/internal/sender.py:1104:            file_stream.JsonlFilePolicy(start_chunk_id=self._resume_state.events),
/n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/wandb/sdk/internal/sender.py:1108:            file_stream.CRDedupeFilePolicy(start_chunk_id=self._resume_state.output),
/n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/wandb/sdk/launch/_project_spec.py:493:        if self.launch_spec.get("_resume_count", 0) > 0:
/n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/wandb/sdk/launch/_project_spec.py:494:            env_vars["WANDB_RESUME"] = "allow"
/n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/wandb/sdk/launch/sweeps/scheduler.py:264:            resume="allow",
/n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/wandb/sdk/launch/agent/agent.py:860:                config["_resume_count"] = config.get("_resume_count", 0) + 1
/n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/wandb/sdk/launch/agent/agent.py:863:                if config["_resume_count"] > MAX_RESUME_COUNT:
/n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/wandb/env.py:44:RESUME = "WANDB_RESUME"

exec
/usr/bin/bash -lc "grep -nE 'Settings|resume|_apply|environment' /n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/wandb/sdk/wandb_init.py | head -160; nl -ba /n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/wandb/sdk/wandb_init.py | sed -n '180,310p;420,540p;650,760p'; grep -nE 'update_from_env|from_env|WANDB_' /n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/wandb/sdk/wandb_settings.py | head -100" in /n/fs/gatrdp/codespace/FLAC
 succeeded in 250ms:
49:from .wandb_settings import Settings
67:def _handle_launch_config(settings: Settings) -> dict[str, Any]:
174:    def maybe_login(self, init_settings: Settings) -> None:
180:            init_settings: Settings passed to `wandb.init()` or set via
209:        """Warn if environment variables changed after `wandb.setup()`.
214:        if not self._wl.did_environment_change():
219:                "Changes to your `wandb` environment variables will be ignored "
231:        init_settings: Settings,
236:            init_settings: Settings specified in the call to `wandb.init()`.
274:        init_settings: Settings,
275:    ) -> tuple[Settings, _PrinterCallback]:
279:            init_settings: Settings passed to `wandb.init()` or set via
328:            # TODO: If executed in a known distributed environment (e.g. Ray or SLURM),
336:    def _load_autoresume_run_id(self, resume_file: pathlib.Path) -> str | None:
337:        """Returns the run_id stored in the auto-resume file, if any.
342:            resume_file: The file path to use for resume='auto' mode.
344:        if not resume_file.exists():
347:        with resume_file.open() as f:
353:                    f"could not decode {resume_file}, ignoring",
360:                    f"resume file at {resume_file} did not store a run_id"
364:    def _save_autoresume_run_id(
367:        resume_file: pathlib.Path,
370:        """Write the run ID to the auto-resume file."""
371:        resume_file.parent.mkdir(exist_ok=True)
372:        with resume_file.open("w") as f:
375:    def set_run_id(self, settings: Settings) -> None:
376:        """Set the run ID and possibly save it to the auto-resume file.
380:        If a `resume_from` is provided and `run_id` is not set, initialize
381:        `run_id` with the `resume_from` run's `run_id`.
384:            settings: The run's settings derived from the environment
387:        if settings.resume == "auto" and settings.resume_fname:
388:            resume_path = pathlib.Path(settings.resume_fname)
390:            resume_path = None
392:        if resume_path:
393:            previous_id = self._load_autoresume_run_id(resume_path)
398:                self._logger.info(f"loaded run ID from {resume_path}")
402:                    f"Ignoring ID {previous_id} loaded due to resume='auto'"
407:        # auto-resume file, then we generate a new ID.
409:            # If resume_from is provided and run_id is not already set,
410:            # initialize run_id with the value from resume_from.
411:            if settings.resume_from:
412:                settings.run_id = settings.resume_from.run
416:        if resume_path:
417:            self._save_autoresume_run_id(
418:                resume_file=resume_path,
422:    def set_sync_dir_suffix(self, settings: Settings) -> None:
443:        settings: Settings,
538:        if settings.resume_from is not None:
540:                "run_id": settings.resume_from.run,
541:                "step": settings.resume_from.value,
613:        self._interface.publish_resume()
635:    def monkeypatch_ipython(self, settings: Settings) -> None:
662:    def setup_run_log_directory(self, settings: Settings) -> Iterator[None]:
735:        by wandb.init(mode="disabled") or by setting the WANDB_MODE environment
746:            settings=Settings(
846:        settings: Settings,
879:            elif settings.resume == "must":
881:                    "Cannot resume a run while another run is active."
1002:        if settings._offline and settings.resume:
1004:                "`resume` will be ignored since W&B syncing is set to `offline`. "
1039:                + " setting: `wandb.init(settings=wandb.Settings(init_timeout=120))`."
1050:        if result.run_result.run.resumed:
1051:            self._logger.info("run resumed")
1053:                tel.feature.resumed = result.run_result.run.resumed
1213:def try_create_root_dir(settings: Settings) -> None:
1286:    resume: bool | Literal["allow", "never", "must", "auto"] | None = None,
1287:    resume_from: str | None = None,
1293:    settings: Settings | dict[str, Any] | None = None,
1342:            tags. To add tags to a resumed run without overwriting the current
1362:            is `False` in scripts and `True` in Notebook environments.
1376:        - `"offline"`: Suitable for air-gapped or offline environments; data
1393:        resume: Controls the behavior when resuming a run with the specified `id`.
1395:        - `"allow"`: If a run with the specified `id` exists, it will resume
1399:        - `"must"`: If a run with the specified `id` exists, it will resume
1401:        - `"auto"`: Automatically resumes the previous run if it crashed on
1404:        - `False`: Deprecated. Use the default behavior (leaving `resume`
1406:            If `resume` is set, `fork_from` and `resume_from` cannot be
1407:            used. When `resume` is unset, the system will always start a new run.
1408:        resume_from: Specifies a moment in a previous run to resume a run from,
1410:            the history logged to a run at an intermediate step and resume logging
1412:            If an `id` argument is also provided, the `resume_from` argument will
1414:            `resume`, `resume_from` and `fork_from` cannot be used together, only
1419:            resumes logging from the specified step in the target run’s history.
1423:            `resume`, `resume_from` and `fork_from` cannot be used together, only
1434:        monitor_gym: Enables automatic logging of videos of the environment when
1436:        settings: Specifies a dictionary or `wandb.Settings` object with advanced
1467:    init_settings = Settings()
1469:        init_settings = Settings(**settings)
1470:    elif isinstance(settings, Settings):
1496:    if resume is not None:
1497:        init_settings.resume = resume  # type: ignore
1511:    if resume_from is not None:
1512:        init_settings.resume_from = resume_from  # type: ignore
1546:        if run_settings.resume_from is not None:
   180	            init_settings: Settings passed to `wandb.init()` or set via
   181	                keyword arguments.
   182	        """
   183	        # Allow settings passed to init() to override inferred values.
   184	        #
   185	        # Calling login() may change settings on the singleton,
   186	        # so these may not be the final run settings.
   187	        run_settings = self._wl.settings.model_copy()
   188	        run_settings.update_from_settings(init_settings)
   189	
   190	        # NOTE: _noop or _offline can become true after _login().
   191	        #   _noop happens if _login hits a timeout.
   192	        #   _offline can be selected by the user at the login prompt.
   193	        if run_settings._noop or run_settings._offline:
   194	            return
   195	
   196	        # Only pass an explicit key when the key was provided directly
   197	        # to ensure correct messaging in _login().
   198	        explicit_key = init_settings.api_key
   199	
   200	        wandb_login._login(
   201	            host=run_settings.base_url,
   202	            force=run_settings.force,
   203	            _silent=run_settings.quiet or run_settings.silent,
   204	            key=explicit_key,
   205	            update_api_key=explicit_key is None,
   206	        )
   207	
   208	    def warn_env_vars_change_after_setup(self) -> _PrinterCallback:
   209	        """Warn if environment variables changed after `wandb.setup()`.
   210	
   211	        Returns:
   212	            A callback to print any generated warnings.
   213	        """
   214	        if not self._wl.did_environment_change():
   215	            return _noop_printer_callback()
   216	
   217	        def print_warning(run_printer: printer.Printer) -> None:
   218	            line = (
   219	                "Changes to your `wandb` environment variables will be ignored "
   220	                "because your `wandb` session has already started. "
   221	                "For more information on how to modify your settings with "
   222	                "`wandb.init()` arguments, please refer to "
   223	                f"{run_printer.link(url_registry.url('wandb-init'), 'the W&B docs')}."
   224	            )
   225	            run_printer.display(line, level="warn")
   226	
   227	        return print_warning
   228	
   229	    def clear_run_path_if_sweep_or_launch(
   230	        self,
   231	        init_settings: Settings,
   232	    ) -> _PrinterCallback:
   233	        """Clear project/entity/run_id keys if in a Sweep or a Launch context.
   234	
   235	        Args:
   236	            init_settings: Settings specified in the call to `wandb.init()`.
   237	
   238	        Returns:
   239	            A callback to print any generated warnings.
   240	        """
   241	        when_doing_thing = ""
   242	
   243	        if self._wl.settings.sweep_id:
   244	            when_doing_thing = "when running a sweep"
   245	        elif self._wl.settings.launch:
   246	            when_doing_thing = "when running from a wandb launch context"
   247	
   248	        if not when_doing_thing:
   249	            return _noop_printer_callback()
   250	
   251	        warnings = []
   252	
   253	        def warn(key: str, value: str) -> None:
   254	            warnings.append(f"Ignoring {key} {value!r} {when_doing_thing}.")
   255	
   256	        if init_settings.project is not None:
   257	            warn("project", init_settings.project)
   258	            init_settings.project = None
   259	        if init_settings.entity is not None:
   260	            warn("entity", init_settings.entity)
   261	            init_settings.entity = None
   262	        if init_settings.run_id is not None:
   263	            warn("run_id", init_settings.run_id)
   264	            init_settings.run_id = None
   265	
   266	        def print_warnings(run_printer: printer.Printer) -> None:
   267	            for warning in warnings:
   268	                run_printer.display(warning, level="warn")
   269	
   270	        return print_warnings
   271	
   272	    def make_run_settings(
   273	        self,
   274	        init_settings: Settings,
   275	    ) -> tuple[Settings, _PrinterCallback]:
   276	        """Returns the run's settings and any warnings.
   277	
   278	        Args:
   279	            init_settings: Settings passed to `wandb.init()` or set via
   280	                keyword arguments.
   281	        """
   282	        warning_callbacks: list[_PrinterCallback] = [
   283	            self.warn_env_vars_change_after_setup(),
   284	            self.clear_run_path_if_sweep_or_launch(init_settings),
   285	        ]
   286	
   287	        # Inherit global settings.
   288	        settings = self._wl.settings.model_copy()
   289	
   290	        # Apply settings from wandb.init() call.
   291	        settings.update_from_settings(init_settings)
   292	
   293	        # Infer the run ID from SageMaker.
   294	        if (
   295	            (not settings.sagemaker_disable)
   296	            and sagemaker.is_using_sagemaker()
   297	            and sagemaker.set_run_id(settings)
   298	        ):
   299	            self._logger.info("set run ID and group based on SageMaker")
   300	            self._telemetry.feature.sagemaker = True
   301	
   302	        # get status of code saving before applying user settings
   303	        save_code_pre_user_settings = settings.save_code
   304	        if not settings._offline and not settings._noop:
   305	            user_settings = self._wl._load_user_settings()
   306	            if user_settings is not None:
   307	                settings.update_from_dict(user_settings)
   308	
   309	        # ensure that user settings don't set saving to true
   310	        # if user explicitly set these to false in UI
   420	            )
   421	
   422	    def set_sync_dir_suffix(self, settings: Settings) -> None:
   423	        """Add a suffix to sync_dir if it already exists.
   424	
   425	        The sync_dir uses a timestamp with second-level precision which can
   426	        result in conflicts if a run with the same ID is initialized within the
   427	        same second. This is most likely to happen in tests.
   428	
   429	        This can't prevent conflicts from multiple processes attempting
   430	        to create a wandb run simultaneously.
   431	
   432	        Args:
   433	            settings: Fully initialized settings other than the
   434	                x_sync_dir_suffix setting which will be modified.
   435	        """
   436	        index = 1
   437	        while pathlib.Path(settings.sync_dir).exists():
   438	            settings.x_sync_dir_suffix = f"{index}"
   439	            index += 1
   440	
   441	    def make_run_config(
   442	        self,
   443	        settings: Settings,
   444	        config: dict | str | None = None,
   445	        config_exclude_keys: list[str] | None = None,
   446	        config_include_keys: list[str] | None = None,
   447	    ) -> _ConfigParts:
   448	        """Construct the run's config.
   449	
   450	        Args:
   451	            settings: The run's finalized settings.
   452	            config: The config passed to `init()`.
   453	            config_exclude_keys: Deprecated. Keys to filter out from `config`.
   454	            config_include_keys: Deprecated. Keys to include from `config`.
   455	
   456	        Returns:
   457	            Initial values for the run's config.
   458	        """
   459	        if config_exclude_keys:
   460	            self.deprecated_features_used.append(
   461	                (
   462	                    Deprecated(init__config_exclude_keys=True),
   463	                    "config_exclude_keys is deprecated. Use"
   464	                    " `config=wandb.helper.parse_config(config_object,"
   465	                    " exclude=('key',))` instead.",
   466	                )
   467	            )
   468	        if config_include_keys:
   469	            self.deprecated_features_used.append(
   470	                (
   471	                    Deprecated(init__config_include_keys=True),
   472	                    "config_include_keys is deprecated. Use"
   473	                    " `config=wandb.helper.parse_config(config_object,"
   474	                    " include=('key',))` instead.",
   475	                )
   476	            )
   477	        config = parse_config(
   478	            config or dict(),
   479	            include=config_include_keys,
   480	            exclude=config_exclude_keys,
   481	        )
   482	
   483	        result = _ConfigParts(
   484	            base_no_artifacts=dict(),
   485	            sweep_no_artifacts=dict(),
   486	            launch_no_artifacts=dict(),
   487	            artifacts=dict(),
   488	        )
   489	
   490	        if not settings.sagemaker_disable and sagemaker.is_using_sagemaker():
   491	            sagemaker_config = sagemaker.parse_sm_config()
   492	            self._split_artifacts_from_config(
   493	                sagemaker_config,
   494	                config_target=result.base_no_artifacts,
   495	                artifacts=result.artifacts,
   496	            )
   497	            self._telemetry.feature.sagemaker = True
   498	
   499	        if self._wl.config:
   500	            self._split_artifacts_from_config(
   501	                self._wl.config,
   502	                config_target=result.base_no_artifacts,
   503	                artifacts=result.artifacts,
   504	            )
   505	
   506	        if config and isinstance(config, dict):
   507	            self._split_artifacts_from_config(
   508	                config,
   509	                config_target=result.base_no_artifacts,
   510	                artifacts=result.artifacts,
   511	            )
   512	
   513	        if self._wl._sweep_config:
   514	            self._split_artifacts_from_config(
   515	                self._wl._sweep_config,
   516	                config_target=result.sweep_no_artifacts,
   517	                artifacts=result.artifacts,
   518	            )
   519	
   520	        if launch_config := _handle_launch_config(settings):
   521	            self._split_artifacts_from_config(
   522	                launch_config,
   523	                config_target=result.launch_no_artifacts,
   524	                artifacts=result.artifacts,
   525	            )
   526	
   527	        wandb_internal = result.base_no_artifacts.setdefault("_wandb", dict())
   528	
   529	        if settings.save_code and settings.program_relpath:
   530	            wandb_internal["code_path"] = paths.LogicalPath(
   531	                os.path.join("code", settings.program_relpath)
   532	            )
   533	        if settings.fork_from is not None:
   534	            wandb_internal["branch_point"] = {
   535	                "run_id": settings.fork_from.run,
   536	                "step": settings.fork_from.value,
   537	            }
   538	        if settings.resume_from is not None:
   539	            wandb_internal["branch_point"] = {
   540	                "run_id": settings.resume_from.run,
   650	            )
   651	
   652	        def publish(data, metadata=None, **kwargs) -> None:
   653	            ipython.display_pub._orig_publish(data, metadata=metadata, **kwargs)
   654	            assert self.notebook is not None
   655	            self.notebook.save_display(
   656	                ipython.execution_count, {"data": data, "metadata": metadata}
   657	            )
   658	
   659	        ipython.display_pub.publish = publish
   660	
   661	    @contextlib.contextmanager
   662	    def setup_run_log_directory(self, settings: Settings) -> Iterator[None]:
   663	        """Set up the run's log directory.
   664	
   665	        This is a context manager that closes and unregisters the log handler
   666	        in case of an uncaught exception, so that future logged messages do not
   667	        modify this run's log file.
   668	        """
   669	        filesystem.mkdir_exists_ok(os.path.dirname(settings.log_user))
   670	        filesystem.mkdir_exists_ok(os.path.dirname(settings.log_internal))
   671	        filesystem.mkdir_exists_ok(os.path.dirname(settings.sync_file))
   672	        filesystem.mkdir_exists_ok(settings.files_dir)
   673	        filesystem.mkdir_exists_ok(settings._tmp_code_dir)
   674	
   675	        if settings.symlink:
   676	            self._safe_symlink(
   677	                os.path.dirname(settings.sync_symlink_latest),
   678	                os.path.dirname(settings.sync_file),
   679	                os.path.basename(settings.sync_symlink_latest),
   680	                delete=True,
   681	            )
   682	            self._safe_symlink(
   683	                os.path.dirname(settings.log_symlink_user),
   684	                settings.log_user,
   685	                os.path.basename(settings.log_symlink_user),
   686	                delete=True,
   687	            )
   688	            self._safe_symlink(
   689	                os.path.dirname(settings.log_symlink_internal),
   690	                settings.log_internal,
   691	                os.path.basename(settings.log_symlink_internal),
   692	                delete=True,
   693	            )
   694	
   695	        assert settings.run_id
   696	        handler = wb_logging.add_file_handler(
   697	            settings.run_id,
   698	            pathlib.Path(settings.log_user),
   699	        )
   700	
   701	        if env.is_debug():
   702	            handler.setLevel(logging.DEBUG)
   703	
   704	        disposed = False
   705	
   706	        def dispose_handler() -> None:
   707	            nonlocal disposed
   708	
   709	            if not disposed:
   710	                disposed = True
   711	                logging.getLogger("wandb").removeHandler(handler)
   712	                handler.close()
   713	
   714	        try:
   715	            self._teardown_hooks.append(
   716	                TeardownHook(
   717	                    call=dispose_handler,
   718	                    stage=TeardownStage.LATE,
   719	                )
   720	            )
   721	
   722	            self._wl._early_logger_flush(logging.getLogger("wandb"))
   723	            self._logger.info(f"Logging user logs to {settings.log_user}")
   724	            self._logger.info(f"Logging internal logs to {settings.log_internal}")
   725	
   726	            yield
   727	        except Exception:
   728	            dispose_handler()
   729	            raise
   730	
   731	    def make_disabled_run(self, config: _ConfigParts) -> Run:
   732	        """Returns a Run-like object where all methods are no-ops.
   733	
   734	        This method is used when the `mode` setting is set to "disabled", such as
   735	        by wandb.init(mode="disabled") or by setting the WANDB_MODE environment
   736	        variable to "disabled".
   737	
   738	        It creates a Run object that mimics the behavior of a normal Run but doesn't
   739	        communicate with the W&B servers.
   740	
   741	        The returned Run object has all expected attributes and methods, but they
   742	        are no-op versions that don't perform any actual logging or communication.
   743	        """
   744	        run_id = runid.generate_id()
   745	        drun = Run(
   746	            settings=Settings(
   747	                mode="disabled",
   748	                root_dir=tempfile.gettempdir(),
   749	                run_id=run_id,
   750	                run_tags=tuple(),
   751	                run_notes=None,
   752	                run_group=None,
   753	                run_name=f"dummy-{run_id}",
   754	                project="dummy",
   755	                entity="dummy",
   756	            )
   757	        )
   758	        # config, summary, and metadata objects
   759	        drun._config = wandb.sdk.wandb_config.Config()
   760	        drun._config.update(config.sweep_no_artifacts)
73:    variables (`WANDB_ prefix`), and with configuration files.
133:    WANDB_APP_URL is the corresponding environment variable.
201:        default_factory=lambda: str(credentials.DEFAULT_WANDB_CREDENTIALS_FILE)
1888:    def update_from_env_vars(self, environ: Dict[str, Any]):
1893:        env_prefix: str = "WANDB_"
1897:            "WANDB_SERVICE_TRANSPORT": "x_service_transport",
1963:                "WANDB_NOTEBOOK_NAME should be a path to a notebook file, "
1967:        # host is populated by update_from_env_vars if the corresponding env

exec
/usr/bin/bash -lc 'for arm in C4L C8 C16 C32 VANL; do id="exp11-${arm}-r99999-1234567890123456789-12345678"; printf '"'%s %s\\n' \""'${#id}" "$id"; done; bash -n worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch; git diff --check -- worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch; for arm in C4L C8 C16; do p="outputs_FLAC/exp11_${arm}/launch_manifest.txt"; printf '"'%s: ' \""'$arm"; awk '"'/"'^job |''^wandb_run_id /{printf "%s | ",$0} END{print ""}'"' \""'$p" 2>/dev/null; done' in /n/fs/gatrdp/codespace/FLAC
 succeeded in 516ms:
45 exp11-C4L-r99999-1234567890123456789-12345678
44 exp11-C8-r99999-1234567890123456789-12345678
45 exp11-C16-r99999-1234567890123456789-12345678
45 exp11-C32-r99999-1234567890123456789-12345678
46 exp11-VANL-r99999-1234567890123456789-12345678
C4L: job 3648694 host neu304.neuronic.cs.princeton.edu mode INITIAL launch_uuid ceb40a63-6ce3-4d38-a2b8-d6c21f1b8cc7 | wandb_run_id exp11-C4L-1786063010468957329-bc46fb0a | 
C8: job 3648695 host neu315.neuronic.cs.princeton.edu mode INITIAL launch_uuid 1009ec7f-e5e2-443d-881e-57ae15525676 | wandb_run_id exp11-C8-1786064131292302937-6d92e299 | 
C16: job 3648696 host neu316.neuronic.cs.princeton.edu mode INITIAL launch_uuid 87e0d920-a64f-46f4-a085-2807ebf0f13c | wandb_run_id exp11-C16-1786064168022803862-f44c29b2 | 

exec
/usr/bin/bash -lc "grep -RInE '64 characters|no longer than 64|run_id.*64|len\\(.*run_id|validate_run_id|invalid.*run.*id' /n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/wandb 2>/dev/null | head -160" in /n/fs/gatrdp/codespace/FLAC
exec
/usr/bin/bash -lc "nl -ba src/tests/test_exp11_restart_record.py | sed -n '390,475p'; nl -ba worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh | sed -n '250,390p'; git grep -n -E 'launch_manifest\\.txt|manifest_path|restarts' -- worklog/worklog_yixun/exp_11_fa_orbit_claude src/tests | head -240" in /n/fs/gatrdp/codespace/FLAC
 succeeded in 352ms:
   390	    os.unlink(registry_of(world)["restarts"][ARM][0]["manifest_path"])
   391	    problems, _ = chain(world, 45000)
   392	    assert problems and "is gone" in problems[0]
   393	
   394	
   395	def test_a_producer_manifest_from_another_leg_is_refused(world):
   396	    """Swapping in a different leg's published file must not launder a checkpoint."""
   397	    assert run_record(world, leg_manifest(world)) == 0
   398	    man_path = os.path.join(world["expdir"], PM.manifest_name(ARM, LEG_JOB))
   399	    man = PM.load(man_path)
   400	    man["job"] = "9999999"
   401	    PM.write_atomic(man_path, man)
   402	    problems, _ = chain(world, 45000)
   403	    assert problems and "!= the registry leg's" in problems[0]
   404	
   405	
   406	# --------------------------------------------------------------------------
   407	# fix 6 — the anchor: an arm's audited final checkpoint
   408	# --------------------------------------------------------------------------
   409	def _ckpt_blob(cfg, step=40000, opt=True, sched=True, ema=True):
   410	    import torch
   411	    sd = {"diffusion.x": torch.zeros(1)}
   412	    if ema:
   413	        sd["diffusion_ema.x"] = torch.zeros(1)
   414	    return {"global_step": step, "epoch": 8, "model_config": cfg, "state_dict": sd,
   415	            "optimizer_states": [{"state": {0: {"step": 1}} if opt else {},
   416	                                  "param_groups": [{"lr": 1e-5}]}],
   417	            "lr_schedulers": [{"last_epoch": step}] if sched else []}
   418	
   419	
   420	@pytest.fixture
   421	def unanchored(world):
   422	    """``world`` before its anchor exists: a real INITIAL manifest, a real torch
   423	    checkpoint at 40k, and no final_ckpt_sha256 — C32's state tonight."""
   424	    import torch
   425	    cfg = json.load(open(world["config"]))
   426	    ckpt_path = world["steps"][40000][0]
   427	    torch.save(_ckpt_blob(cfg), ckpt_path)
   428	    world["steps"][40000] = (ckpt_path, PM.sha256_file(ckpt_path))
   429	
   430	    man_rel = os.path.join(world["save_dir"], "launch_manifest.txt")
   431	    man_abs = os.path.join(world["root"], man_rel)
   432	    write(man_abs, "\n".join([
   433	        "# exp_11 arm launch manifest",
   434	        f"job {INITIAL_JOB} host neu000 mode INITIAL launch_uuid initial-uuid",
   435	        f"arm {ARM} rung 8x8 micro 8 ngpu 8 max_steps 40000 ckpt_every 2500",
   436	        f"commit {LAUNCH_COMMIT}",
   437	        f"p0_manifest_sha256 {'a' * 64}",
   438	        f"model_config {world['config']}",
   439	        f"config_sha256 {world['config_sha']}",
   440	        f"vae_sha256 {'b' * 64}",
   441	        f"save_dir {world['save_dir']}",
   442	        "wandb_run_id exp11-C8-initial", ""]))
   443	    reg = registry_of(world)
   444	    row = reg["arms"][ARM]
   445	    row["manifest_path"] = man_rel
   446	    row["manifest_sha256"] = PM.sha256_file(man_abs)
   447	    row.pop("final_ckpt_sha256", None)
   448	    row.pop("final_step", None)
   449	    with open(world["registry"], "w") as fh:
   450	        json.dump(reg, fh, indent=2)
   451	    world["manifest"] = man_abs
   452	    return world
   453	
   454	
   455	def run_anchor(world, *extra):
   456	    return ANCH.main([ARM, "--registry", world["registry"], "--launcher", world["launcher"],
   457	                      "--repo-root", world["root"], *extra])
   458	
   459	
   460	def test_add_anchor_records_the_audited_final_checkpoint(unanchored):
   461	    assert run_anchor(unanchored) == 0
   462	    row = registry_of(unanchored)["arms"][ARM]
   463	    assert row["final_ckpt_sha256"] == unanchored["steps"][40000][1]
   464	    assert row["final_step"] == 40000
   465	    assert row["final_ckpt_path"].endswith("epoch=8-step=40000.ckpt")
   466	    assert not row["final_ckpt_path"].startswith("/"), "the path must be repo-relative"
   467	
   468	
   469	def test_add_anchor_is_idempotent(unanchored, capsys):
   470	    assert run_anchor(unanchored) == 0
   471	    assert run_anchor(unanchored) == 0
   472	    assert "already anchored" in capsys.readouterr().out
   473	
   474	
   475	def test_add_anchor_refuses_to_re_anchor_a_different_checkpoint(unanchored, capsys):
   250	tmp, cfg_path = sys.argv[1], sys.argv[2]
   251	cfg = json.load(open(cfg_path))
   252	def ck(step=5000, config=cfg, opt=True, sched=True, ema=True):
   253	    d = {"global_step": step, "epoch": 1, "model_config": config,
   254	         "state_dict": {"diffusion.x": torch.zeros(1)},
   255	         "optimizer_states": [{"state": {0: {"step": 1}} if opt else {},
   256	                               "param_groups": [{"lr": 1e-5}]}],
   257	         "lr_schedulers": [{"last_epoch": step}] if sched else []}
   258	    if ema:
   259	        d["state_dict"]["diffusion_ema.x"] = torch.zeros(1)
   260	    return d
   261	torch.save(ck(), os.path.join(tmp, "good.ckpt"))
   262	torch.save(ck(step=4999), os.path.join(tmp, "wrongstep.ckpt"))
   263	c4 = json.loads(json.dumps(cfg)); c4["training"]["frame_avg_angles"] = [0.0, 90.0, 180.0, 270.0]
   264	torch.save(ck(config=c4), os.path.join(tmp, "wrongorbit.ckpt"))
   265	torch.save(ck(opt=False), os.path.join(tmp, "stripped.ckpt"))
   266	torch.save(ck(ema=False), os.path.join(tmp, "noema.ckpt"))
   267	torch.save(ck(sched=False), os.path.join(tmp, "nosched.ckpt"))
   268	torch.save(ck(step=45000), os.path.join(tmp, "past.ckpt"))
   269	open(os.path.join(tmp, "empty.ckpt"), "wb").close()
   270	print("synthetic checkpoints written")
   271	PY
   272	PRE=($PY "$PREFLIGHT" --config "${EXPDIR}/FLAC_AR_BF_C8.json" --max-steps 40000 --arm C8 --rung 16x4)
   273	expect_cmd "preflight accepts a good ckpt" 0 "CKPT_SHA256" -- "${PRE[@]}" --ckpt "${TMP}/good.ckpt" --expected-step 5000
   274	expect_cmd "preflight rejects a step mismatch" 2 "global_step" -- "${PRE[@]}" --ckpt "${TMP}/wrongstep.ckpt" --expected-step 5000
   275	expect_cmd "preflight rejects a foreign orbit" 2 "embedded model_config" -- "${PRE[@]}" --ckpt "${TMP}/wrongorbit.ckpt" --expected-step 5000
   276	expect_cmd "preflight rejects a stripped optimizer" 2 "optimizer state is CLEARED" -- "${PRE[@]}" --ckpt "${TMP}/stripped.ckpt" --expected-step 5000
   277	expect_cmd "preflight rejects a missing EMA" 2 "no EMA weights" -- "${PRE[@]}" --ckpt "${TMP}/noema.ckpt" --expected-step 5000
   278	expect_cmd "preflight rejects a missing scheduler" 2 "lr_schedulers" -- "${PRE[@]}" --ckpt "${TMP}/nosched.ckpt" --expected-step 5000
   279	expect_cmd "preflight rejects a past-budget ckpt" 2 ">= max_steps" -- "${PRE[@]}" --ckpt "${TMP}/past.ckpt" --expected-step 45000
   280	expect_cmd "preflight rejects an empty file" 2 "PREFLIGHT" -- "${PRE[@]}" --ckpt "${TMP}/empty.ckpt" --expected-step 5000
   281	expect_cmd "preflight rejects a missing file" 2 "not found" -- "${PRE[@]}" --ckpt "${TMP}/nope.ckpt" --expected-step 5000
   282	# manifest binding: same rung passes, changed rung fails
   283	cat > "${TMP}/launch_manifest.txt" <<EOF
   284	# exp_11 arm launch manifest
   285	arm C8 rung 16x4 micro 16 ngpu 4 max_steps 40000 ckpt_every 2500
   286	commit ${HEAD_SHA}
   287	wandb_run_id exp11-C8-test
   288	EOF
   289	expect_cmd "preflight binds to the launch manifest" 0 "bound to launch manifest" -- \
   290	  "${PRE[@]}" --ckpt "${TMP}/good.ckpt" --expected-step 5000 --commit "$HEAD_SHA" --launch-manifest "${TMP}/launch_manifest.txt"
   291	expect_cmd "preflight rejects a rung change" 2 "manifest rung" -- \
   292	  $PY "$PREFLIGHT" --config "${EXPDIR}/FLAC_AR_BF_C8.json" --max-steps 40000 --arm C8 --rung 8x8 \
   293	     --ckpt "${TMP}/good.ckpt" --expected-step 5000 --launch-manifest "${TMP}/launch_manifest.txt"
   294	# B2 residual: a manifest with no commit, or a different commit, must fail CLOSED
   295	grep -v '^commit ' "${TMP}/launch_manifest.txt" > "${TMP}/manifest_nocommit.txt"
   296	expect_cmd "preflight rejects a manifest without a commit" 2 "no 'commit' line" -- \
   297	  "${PRE[@]}" --ckpt "${TMP}/good.ckpt" --expected-step 5000 --commit "$HEAD_SHA" \
   298	     --launch-manifest "${TMP}/manifest_nocommit.txt"
   299	sed 's/^commit .*/commit 0000000000000000000000000000000000000000/' "${TMP}/launch_manifest.txt" > "${TMP}/manifest_othercommit.txt"
   300	expect_cmd "preflight rejects a changed commit" 2 "!= running commit" -- \
   301	  "${PRE[@]}" --ckpt "${TMP}/good.ckpt" --expected-step 5000 --commit "$HEAD_SHA" \
   302	     --launch-manifest "${TMP}/manifest_othercommit.txt"
   303	expect_cmd "preflight rejects a missing running commit" 2 "no running commit" -- \
   304	  "${PRE[@]}" --ckpt "${TMP}/good.ckpt" --expected-step 5000 \
   305	     --launch-manifest "${TMP}/launch_manifest.txt"
   306	
   307	echo "--- G2. Q10: the JOB selects and enforces the RESTART time pin (re-pin fix 1) ---"
   308	# The submitter allocated 34/51/89 h for the restart legs, but the job selected
   309	# the INITIAL pin and then refused its own allocation. The pin the job enforces
   310	# must follow the LEG, not the arm.
   311	Q10_RUN="${OUT_ROOT}/exp11_C8/FLAC_exp11_C8/exp11_C8"
   312	mkdir -p "${Q10_RUN}/checkpoints"
   313	: > "${Q10_RUN}/checkpoints/epoch=8-step=40000.ckpt"
   314	Q10_ENV=(DRYRUN=1 "EXPECT_SHA=${HEAD_SHA}" "OUTPUT_ROOT=${OUT_ROOT}" "${REPO_ENV[@]}")
   315	case_run "a RESTART leg selects the RESTART pin" 0 "time pin PINNED_TIME_LIMIT_RESTART_C8=51:00:00" \
   316	  -- "${Q10_ENV[@]}" ARM=C8 EXPECTED_STEP=40000 \
   317	     "RESUME_CKPT=${Q10_RUN}/checkpoints/epoch=8-step=40000.ckpt"
   318	case_run "an INITIAL launch keeps the INITIAL pin" 0 "time pin PINNED_TIME_LIMIT_C16=60:00:00" \
   319	  -- "${Q10_ENV[@]}" ARM=C16
   320	if grep -q 'the \${TIME_PIN_NAME} pin' "$LAUNCHER"; then
   321	  echo "PASS  the allocation gate names the pin it enforced"; PASS=$((PASS+1))
   322	else
   323	  echo "FAIL  the allocation gate does not enforce the SELECTED time pin"; FAIL=$((FAIL+1))
   324	fi
   325	# submitter and job must pick the same pin for the same leg
   326	SUB_RESTART="$(env DRYRUN=1 bash "$SUBMITTER" C16 --resume "${Q10_RUN}/checkpoints/epoch=8-step=40000.ckpt" --expected-step 40000 2>&1)"
   327	if echo "$SUB_RESTART" | grep -q "time 89:00:00"; then
   328	  echo "PASS  submitter and job agree on the C16 RESTART pin"; PASS=$((PASS+1))
   329	else
   330	  echo "FAIL  the submitter no longer allocates the C16 RESTART pin"; FAIL=$((FAIL+1))
   331	fi
   332	
   333	echo "--- G3. Q10: the 40k -> 100k EXTENSION preflight contract (re-pin fix 1) ---"
   334	# The ordinary restart contract requires manifest max_steps == this run's budget
   335	# and manifest commit == the running commit. An extension violates both BY
   336	# DESIGN, so it gets its own contract: the original launch identity is preserved
   337	# (audited manifest bytes, job/uuid/commit/config/save-dir/seed, and the resumed
   338	# checkpoint IS the audited 40k anchor) while budget and commit may move.
   339	EXT_ROOT="${TMP}/ext"; EXT_SAVE="${EXT_ROOT}/exp11_C8"
   340	EXT_CKPT_DIR="${EXT_SAVE}/FLAC_exp11_C8/exp11_C8/checkpoints"
   341	mkdir -p "$EXT_CKPT_DIR" "${EXT_ROOT}/elsewhere"
   342	$PY - "$TMP" "${EXPDIR}/FLAC_AR_BF_C8.json" "$EXT_CKPT_DIR" "$EXT_SAVE" "${EXT_ROOT}/elsewhere" "$LAUNCHER" <<'PY'
   343	import hashlib, json, os, re, sys, torch
   344	tmp, cfg_path, ckpt_dir, save_dir, other, launcher = sys.argv[1:7]
   345	vae_sha = re.search(r'^PINNED_VAE_SHA256="([^"]*)"', open(launcher).read(), re.M).group(1)
   346	cfg = json.load(open(cfg_path))
   347	ck = {"global_step": 40000, "epoch": 8, "model_config": cfg,
   348	      "state_dict": {"diffusion.x": torch.zeros(1), "diffusion_ema.x": torch.zeros(1)},
   349	      "optimizer_states": [{"state": {0: {"step": 1}}, "param_groups": [{"lr": 1e-5}]}],
   350	      "lr_schedulers": [{"last_epoch": 40000}]}
   351	path = os.path.join(ckpt_dir, "epoch=8-step=40000.ckpt")
   352	torch.save(ck, path)
   353	torch.save(ck, os.path.join(other, "epoch=8-step=40000.ckpt"))
   354	h = hashlib.sha256(open(path, "rb").read()).hexdigest()
   355	cfg_sha = hashlib.sha256(open(cfg_path, "rb").read()).hexdigest()
   356	man = os.path.join(tmp, "ext_launch_manifest.txt")
   357	with open(man, "w") as fh:
   358	    fh.write("# exp_11 arm launch manifest\n")
   359	    fh.write("job 3648695 host neu000 mode INITIAL launch_uuid ext-uuid-c8\n")
   360	    fh.write("arm C8 rung 8x8 micro 8 ngpu 8 max_steps 40000 ckpt_every 2500\n")
   361	    fh.write("commit " + "2" * 40 + "\n")
   362	    fh.write(f"model_config {cfg_path}\n")
   363	    fh.write(f"config_sha256 {cfg_sha}\n")
   364	    fh.write(f"vae_sha256 {vae_sha}\n")
   365	    fh.write(f"save_dir {save_dir}\n")
   366	    fh.write("wandb_run_id exp11-C8-ext\n")
   367	reg = {"arms": {"C8": {
   368	    "manifest_path": man,
   369	    "manifest_sha256": hashlib.sha256(open(man, "rb").read()).hexdigest(),
   370	    "job": "3648695", "mode": "INITIAL", "launch_uuid": "ext-uuid-c8",
   371	    "commit": "2" * 40, "rung": "8x8", "max_steps": "40000",
   372	    "config_sha256": cfg_sha, "vae_sha256": vae_sha, "save_dir": save_dir,
   373	    "training_seed": 42,
   374	    "final_ckpt_sha256": h, "final_step": 40000}}, "restarts": {}}
   375	json.dump(reg, open(os.path.join(tmp, "ext_registry.json"), "w"), indent=2)
   376	print("extension fixture written")
   377	PY
   378	EXT_CKPT="${EXT_CKPT_DIR}/epoch=8-step=40000.ckpt"
   379	EXT=($PY "$PREFLIGHT" --config "${EXPDIR}/FLAC_AR_BF_C8.json" --arm C8 --rung 8x8
   380	     --ckpt "$EXT_CKPT" --expected-step 40000 --commit "$HEAD_SHA"
   381	     --launch-manifest "${TMP}/ext_launch_manifest.txt" --extension
   382	     --launch-registry "${TMP}/ext_registry.json")
   383	expect_cmd "the ORDINARY contract refuses the extension (the bug)" 2 "manifest max_steps" -- \
   384	  $PY "$PREFLIGHT" --config "${EXPDIR}/FLAC_AR_BF_C8.json" --arm C8 --rung 8x8 \
   385	     --ckpt "$EXT_CKPT" --expected-step 40000 --commit "$HEAD_SHA" --max-steps 100000 \
   386	     --launch-manifest "${TMP}/ext_launch_manifest.txt"
   387	expect_cmd "extension accepts the 40k->100k leg" 0 "extension lineage OK" -- "${EXT[@]}" --max-steps 100000
   388	expect_cmd "extension keeps the ORIGINAL launch commit" 0 "launch commit 2222222222" -- "${EXT[@]}" --max-steps 100000
   389	expect_cmd "extension refuses a shrinking budget" 2 "does not extend" -- "${EXT[@]}" --max-steps 39000
   390	expect_cmd "extension refuses a foreign resume path" 2 "canonical run directory" -- \
src/tests/test_exp11_restart_record.py:100:            "manifest_path": "outputs/exp11_C8/launch_manifest.txt",
src/tests/test_exp11_restart_record.py:106:        }}, "restarts": {}}, fh, indent=2)
src/tests/test_exp11_restart_record.py:162:    legs = registry_of(world)["restarts"][ARM]
src/tests/test_exp11_restart_record.py:178:    assert registry_of(world)["restarts"] == {}
src/tests/test_exp11_restart_record.py:218:    assert registry_of(world)["restarts"] == {}
src/tests/test_exp11_restart_record.py:235:    assert len(registry_of(world)["restarts"][ARM]) == 1
src/tests/test_exp11_restart_record.py:283:    assert len(registry_of(world)["restarts"][ARM]) == 1, "extending must not add a second row"
src/tests/test_exp11_restart_record.py:370:    reg["restarts"][ARM][0][field] = value
src/tests/test_exp11_restart_record.py:381:    recorded = registry_of(world)["restarts"][ARM][0]["manifest_path"]
src/tests/test_exp11_restart_record.py:390:    os.unlink(registry_of(world)["restarts"][ARM][0]["manifest_path"])
src/tests/test_exp11_restart_record.py:430:    man_rel = os.path.join(world["save_dir"], "launch_manifest.txt")
src/tests/test_exp11_restart_record.py:445:    row["manifest_path"] = man_rel
src/tests/test_exp11_restart_record.py:549:    assert registry_of(unanchored).get("restarts", {}) in ({}, {ARM: []})
src/tests/test_exp11_restart_record.py:554:    legs = registry_of(unanchored)["restarts"][ARM]
worklog/worklog_yixun/exp_11_fa_orbit_claude/arm_launch_registry.json:12:    "RESTART legs (Q10, 40k -> 100k) are recorded under 'restarts' as a CHAIN:",
worklog/worklog_yixun/exp_11_fa_orbit_claude/arm_launch_registry.json:30:      "manifest_path": "outputs_FLAC/exp11_C4L/launch_manifest.txt",
worklog/worklog_yixun/exp_11_fa_orbit_claude/arm_launch_registry.json:49:      "manifest_path": "outputs_FLAC/exp11_C8/launch_manifest.txt",
worklog/worklog_yixun/exp_11_fa_orbit_claude/arm_launch_registry.json:68:      "manifest_path": "outputs_FLAC/exp11_C16/launch_manifest.txt",
worklog/worklog_yixun/exp_11_fa_orbit_claude/arm_launch_registry.json:87:      "manifest_path": "outputs_FLAC/exp11_C32/launch_manifest.txt",
worklog/worklog_yixun/exp_11_fa_orbit_claude/arm_launch_registry.json:108:      "manifest_path": "outputs_FLAC/exp11_VANL/launch_manifest.txt",
worklog/worklog_yixun/exp_11_fa_orbit_claude/arm_launch_registry.json:125:  "restarts": {}
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-10_01-37-34_failopen_repro.log:6:    registry restarts now: {"C8": [{"manifest_path": "/n/fs/rfmprog/.tmp/yh4742/tmpou581k5a/old/restart_manifest.txt", "manifest_sha256": "1996be56
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-10_01-37-34_failopen_repro.log:10:    registry restarts now: {}
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_add_anchor.py:47:    man_path = resolve(repo_root, str(row.get("manifest_path", "")))
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_add_anchor.py:48:    if not row.get("manifest_path") or not os.path.isfile(man_path):
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py:81:def check_manifest_binding(manifest_path, arm, rung, commit, maxsteps):
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py:82:    man = parse_manifest(manifest_path)
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py:123:def check_extension_binding(manifest_path, registry_path, arm, rung, config_path, ckpt_path,
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py:147:    man = parse_manifest(manifest_path)
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py:150:    got_sha = sha256_file(manifest_path)
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py:271:        problems.append("optimizer state is CLEARED (stripped checkpoint); exp_11 restarts are "
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_code_r4_final.md:19:| **2 — active-arm checkpoint bound to original launch manifest and canonical run directory** | **PARTIALLY** | Canonical realpath equality and arm/config matching are enforced ([fa_orbit_screen.sbatch:228](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen.sbatch:228), [fa_orbit_screen.sbatch:243](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen.sbatch:243)). The row validator also uses realpath containment rather than substring matching ([exp11_validate_rows.py:355](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/exp11_validate_rows.py:355)). But the manifest is mutable evidence under ignored `outputs_FLAC`, and the gate does not validate its recorded launch commit, `mode INITIAL`, launch UUID/job, rung, training seed/command, P0 manifest hash, or VAE hash. The message even prints “seed 42 recipe” without parsing or checking it ([fa_orbit_screen.sbatch:263](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen.sbatch:263)); those fields exist in the manifest ([launch_manifest.txt:3](/n/fs/gatrdp/codespace/FLAC/outputs_FLAC/exp11_C8/launch_manifest.txt:3)). This binds the checkpoint to a directory described by the current manifest, not cryptographically to the original audited manifest. |
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_content_gate_review.md:770:   268	cat > "${TMP}/launch_manifest.txt" <<EOF
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_content_gate_review.md:780:     6	# recipe is now literally pinned rather than operator-supplied, restarts get
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_content_gate_review.md:1084:   310	LAUNCH_MANIFEST_LINK="${SAVEDIR}/launch_manifest.txt"     # written by the INITIAL launch
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_content_gate_review.md:1277:   310	LAUNCH_MANIFEST_LINK="${SAVEDIR}/launch_manifest.txt"     # written by the INITIAL launch
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_content_gate_review.md:1478:   511	  # restarts have no registered launch and keep the ordinary contract.
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_content_gate_review.md:2779:worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py:81:def check_manifest_binding(manifest_path, arm, rung, commit, maxsteps):
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_content_gate_review.md:2780:worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py:82:    man = parse_manifest(manifest_path)
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_content_gate_review.md:2791:worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py:123:def check_extension_binding(manifest_path, registry_path, arm, rung, config_path, ckpt_path,
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_content_gate_review.md:2800:worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py:147:    man = parse_manifest(manifest_path)
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_content_gate_review.md:2801:worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py:150:    got_sha = sha256_file(manifest_path)
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_content_gate_review.md:3346:def check_manifest_binding(manifest_path, arm, rung, commit, maxsteps):
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_content_gate_review.md:3347:    man = parse_manifest(manifest_path)
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_content_gate_review.md:3388:def check_extension_binding(manifest_path, registry_path, arm, rung, config_path, ckpt_path,
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_content_gate_review.md:3412:    man = parse_manifest(manifest_path)
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_content_gate_review.md:3415:    got_sha = sha256_file(manifest_path)
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_content_gate_review.md:3536:        problems.append("optimizer state is CLEARED (stripped checkpoint); exp_11 restarts are "
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_content_gate_review_r2.md:486:     6	# recipe is now literally pinned rather than operator-supplied, restarts get
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_content_gate_review_r2.md:807:   327	LAUNCH_MANIFEST_LINK="${SAVEDIR}/launch_manifest.txt"     # written by the INITIAL launch
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_content_gate_review_r2.md:1229:   269	cat > "${TMP}/launch_manifest.txt" <<EOF
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_content_gate_review_r2.md:1236:   276	  "${PRE[@]}" --ckpt "${TMP}/good.ckpt" --expected-step 5000 --commit "$HEAD_SHA" --launch-manifest "${TMP}/launch_manifest.txt"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_content_gate_review_r2.md:1239:   279	     --ckpt "${TMP}/good.ckpt" --expected-step 5000 --launch-manifest "${TMP}/launch_manifest.txt"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_content_gate_review_r2.md:1241:   281	grep -v '^commit ' "${TMP}/launch_manifest.txt" > "${TMP}/manifest_nocommit.txt"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_content_gate_review_r2.md:1245:   285	sed 's/^commit .*/commit 0000000000000000000000000000000000000000/' "${TMP}/launch_manifest.txt" > "${TMP}/manifest_othercommit.txt"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_content_gate_review_r2.md:1251:   291	     --launch-manifest "${TMP}/launch_manifest.txt"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_content_gate_review_r2.md:1302:   342	man = os.path.join(tmp, "ext_launch_manifest.txt")
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_content_gate_review_r2.md:1314:   354	    "manifest_path": man,
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_content_gate_review_r2.md:1320:   360	    "final_ckpt_sha256": h, "final_step": 40000}}, "restarts": {}}
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_content_gate_review_r2.md:1327:   367	     --launch-manifest "${TMP}/ext_launch_manifest.txt" --extension
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_content_gate_review_r2.md:1332:   372	     --launch-manifest "${TMP}/ext_launch_manifest.txt"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_content_gate_review_r2.md:1339:   379	     --launch-manifest "${TMP}/ext_launch_manifest.txt" --extension \
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_content_gate_review_r2.md:1354:   394	  --launch-manifest "${TMP}/ext_launch_manifest.txt" --extension --launch-registry "$1"; }
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_content_gate_review_r2.md:1375:   415	printf 'tamper\n' >> "${TMP}/ext_launch_manifest.txt"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_content_gate_review_r2.md:1725:327:LAUNCH_MANIFEST_LINK="${SAVEDIR}/launch_manifest.txt"     # written by the INITIAL launch
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_content_gate_review_r2.md:1842:   327	LAUNCH_MANIFEST_LINK="${SAVEDIR}/launch_manifest.txt"     # written by the INITIAL launch
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_content_gate_review_r3.md:86:# recipe is now literally pinned rather than operator-supplied, restarts get
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_readout_review.md:344:outputs_FLAC/exp11_C4L/launch_manifest.txt
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_readout_review.md:511:outputs_FLAC/exp11_VANL/launch_manifest.txt
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_repin_review.md:51:   The pinned registry has `restarts: {}` and C32 has no audited 40k anchor at [arm_launch_registry.json:79](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/arm_launch_registry.json:79). Because screens read the registry from the detached pinned worktree, recording restart rows later in the main checkout will not make them visible at `45b6154`. Every >40k trajectory evaluation would therefore fail closed at this pin, and C32 can never chain from it.
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_vanl_review.md:54:Do **not** pin VANL evaluations to `9ce96a3`: it lacks the launch-registry entry and the fixes above. The registry entry should be made only after the INITIAL job has started and published `outputs_FLAC/exp11_VANL/launch_manifest.txt`; `sbatch` return alone is insufficient.
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_producer_manifest.py:223:    legs = (reg.get("restarts") or {}).get(arm) or []
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_producer_manifest.py:237:        leg_man = resolve(repo_root, str(leg.get("manifest_path")))
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_record_restart.py:198:    legs = reg.setdefault("restarts", {}).setdefault(arm, [])
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_record_restart.py:218:        "manifest_path": args.manifest, "manifest_sha256": man_sha,
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen.sbatch:417:  ARM_LAUNCH_MANIFEST="${OUTPUT_ROOT}/exp11_${ARM}/launch_manifest.txt"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:102:    with open(os.path.join(d, "launch_manifest.txt"), "w") as fh:
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:111:        "manifest_path": os.path.join(d, "launch_manifest.txt"),
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:113:            open(os.path.join(d, "launch_manifest.txt"), "rb").read()).hexdigest(),
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:133:  $PY - "$1" "${OUT_ROOT}/exp11_$1/launch_manifest.txt" "${OUT_ROOT}/arm_launch_registry.json" <<'PY'
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:147:    "manifest_path": man_path, "manifest_sha256": hashlib.sha256(raw).hexdigest(),
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:215:    echo "save_dir ${OUT_ROOT}/exp11_C8"; } > "${OUT_ROOT}/exp11_C8/launch_manifest.txt"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:305:# restarts: a >40k checkpoint must therefore be refused outright.
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:330:r["restarts"] = {"C8": [{"mode": "RESTART", "job": "999", "resume_ckpt_sha256": "b" * 64,
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:359:r["restarts"] = {}
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:397:r = json.load(open(p)); r["restarts"]["C8"][0][field] = value
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:1454:  MAN="${OUT_ROOT}/exp11_C8/launch_manifest.txt"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:6:# recipe is now literally pinned rather than operator-supplied, restarts get
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:331:LAUNCH_MANIFEST_LINK="${SAVEDIR}/launch_manifest.txt"     # written by the INITIAL launch
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:532:  # restarts have no registered launch and keep the ordinary contract.
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh:283:cat > "${TMP}/launch_manifest.txt" <<EOF
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh:290:  "${PRE[@]}" --ckpt "${TMP}/good.ckpt" --expected-step 5000 --commit "$HEAD_SHA" --launch-manifest "${TMP}/launch_manifest.txt"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh:293:     --ckpt "${TMP}/good.ckpt" --expected-step 5000 --launch-manifest "${TMP}/launch_manifest.txt"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh:295:grep -v '^commit ' "${TMP}/launch_manifest.txt" > "${TMP}/manifest_nocommit.txt"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh:299:sed 's/^commit .*/commit 0000000000000000000000000000000000000000/' "${TMP}/launch_manifest.txt" > "${TMP}/manifest_othercommit.txt"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh:305:     --launch-manifest "${TMP}/launch_manifest.txt"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh:356:man = os.path.join(tmp, "ext_launch_manifest.txt")
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh:368:    "manifest_path": man,
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh:374:    "final_ckpt_sha256": h, "final_step": 40000}}, "restarts": {}}
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh:381:     --launch-manifest "${TMP}/ext_launch_manifest.txt" --extension
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh:386:     --launch-manifest "${TMP}/ext_launch_manifest.txt"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh:393:     --launch-manifest "${TMP}/ext_launch_manifest.txt" --extension \
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh:408:  --launch-manifest "${TMP}/ext_launch_manifest.txt" --extension --launch-registry "$1"; }
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh:429:printf 'tamper\n' >> "${TMP}/ext_launch_manifest.txt"

 succeeded in 34282ms:
/n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/wandb/apis/public/api.py:366:            raise ValueError("name must be less than 64 characters")
/n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/wandb/apis/public/api.py:554:            raise ValueError("name must be less than 64 characters")
/n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/wandb/proto/v5/wandb_api_pb2.py:19:DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x1bwandb/proto/wandb_api.proto\x12\x0ewandb_internal\x1a wandb/proto/wandb_internal.proto\x1a wandb/proto/wandb_settings.proto\"B\n\x14ServerApiInitRequest\x12*\n\x08settings\x18\x01 \x01(\x0b\x32\x18.wandb_internal.Settings\">\n\x15ServerApiInitResponse\x12\x15\n\rerror_message\x18\x01 \x01(\t\x12\x0e\n\x06\x61pi_id\x18\x02 \x01(\t\"\xaf\x01\n\nApiRequest\x12\x0e\n\x06\x61pi_id\x18\x01 \x01(\t\x12I\n\x18read_run_history_request\x18\x02 \x01(\x0b\x32%.wandb_internal.ReadRunHistoryRequestH\x00\x12;\n\x10\x66\x65\x61tures_request\x18\x03 \x01(\x0b\x32\x1f.wandb_internal.FeaturesRequestH\x00\x42\t\n\x07request\"\xe5\x01\n\x0b\x41piResponse\x12K\n\x19read_run_history_response\x18\x01 \x01(\x0b\x32&.wandb_internal.ReadRunHistoryResponseH\x00\x12=\n\x11\x66\x65\x61tures_response\x18\x03 \x01(\x0b\x32 .wandb_internal.FeaturesResponseH\x00\x12>\n\x12\x61pi_error_response\x18\x02 \x01(\x0b\x32 .wandb_internal.ApiErrorResponseH\x00\x42\n\n\x08response\"f\n\x10\x41piErrorResponse\x12\x0f\n\x07message\x18\x01 \x01(\t\x12\x32\n\nerror_type\x18\x02 \x01(\x0e\x32\x19.wandb_internal.ErrorTypeH\x00\x88\x01\x01\x42\r\n\x0b_error_type\")\n\x17ServerApiCleanupRequest\x12\x0e\n\x06\x61pi_id\x18\x01 \x01(\t\"B\n\x0f\x46\x65\x61turesRequest\x12/\n\x08\x66\x65\x61tures\x18\x01 \x03(\x0e\x32\x1d.wandb_internal.ServerFeature\"B\n\x10\x46\x65\x61turesResponse\x12.\n\x07\x65nabled\x18\x01 \x03(\x0e\x32\x1d.wandb_internal.ServerFeature\"\xd0\x03\n\x15ReadRunHistoryRequest\x12\x43\n\x15scan_run_history_init\x18\x01 \x01(\x0b\x32\".wandb_internal.ScanRunHistoryInitH\x00\x12:\n\x10scan_run_history\x18\x02 \x01(\x0b\x32\x1e.wandb_internal.ScanRunHistoryH\x00\x12I\n\x18scan_run_history_cleanup\x18\x03 \x01(\x0b\x32%.wandb_internal.ScanRunHistoryCleanupH\x00\x12K\n\x19\x64ownload_run_history_init\x18\x04 \x01(\x0b\x32&.wandb_internal.DownloadRunHistoryInitH\x00\x12\x42\n\x14\x64ownload_run_history\x18\x05 \x01(\x0b\x32\".wandb_internal.DownloadRunHistoryH\x00\x12O\n\x1b\x64ownload_run_history_status\x18\x06 \x01(\x0b\x32(.wandb_internal.DownloadRunHistoryStatusH\x00\x42\t\n\x07request\"\xf9\x03\n\x16ReadRunHistoryResponse\x12K\n\x15scan_run_history_init\x18\x01 \x01(\x0b\x32*.wandb_internal.ScanRunHistoryInitResponseH\x00\x12\x39\n\x0brun_history\x18\x02 \x01(\x0b\x32\".wandb_internal.RunHistoryResponseH\x00\x12Q\n\x18scan_run_history_cleanup\x18\x03 \x01(\x0b\x32-.wandb_internal.ScanRunHistoryCleanupResponseH\x00\x12S\n\x19\x64ownload_run_history_init\x18\x04 \x01(\x0b\x32..wandb_internal.DownloadRunHistoryInitResponseH\x00\x12J\n\x14\x64ownload_run_history\x18\x05 \x01(\x0b\x32*.wandb_internal.DownloadRunHistoryResponseH\x00\x12W\n\x1b\x64ownload_run_history_status\x18\x06 \x01(\x0b\x32\x30.wandb_internal.DownloadRunHistoryStatusResponseH\x00\x42\n\n\x08response\"f\n\x12ScanRunHistoryInit\x12\x0e\n\x06\x65ntity\x18\x01 \x01(\t\x12\x0f\n\x07project\x18\x02 \x01(\t\x12\x0e\n\x06run_id\x18\x03 \x01(\t\x12\x0c\n\x04keys\x18\x04 \x03(\t\x12\x11\n\tuse_cache\x18\x05 \x01(\x08\"0\n\x1aScanRunHistoryInitResponse\x12\x12\n\nrequest_id\x18\x01 \x01(\x05\"H\n\x0eScanRunHistory\x12\x10\n\x08min_step\x18\x01 \x01(\x03\x12\x10\n\x08max_step\x18\x02 \x01(\x03\x12\x12\n\nrequest_id\x18\x03 \x01(\x05\"F\n\x12RunHistoryResponse\x12\x30\n\x0chistory_rows\x18\x01 \x03(\x0b\x32\x1a.wandb_internal.HistoryRow\"G\n\nHistoryRow\x12\x39\n\rhistory_items\x18\x01 \x03(\x0b\x32\".wandb_internal.ParquetHistoryItem\"5\n\x12ParquetHistoryItem\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\x12\n\nvalue_json\x18\x10 \x01(\t\"+\n\x15ScanRunHistoryCleanup\x12\x12\n\nrequest_id\x18\x01 \x01(\x05\"\x1f\n\x1dScanRunHistoryCleanupResponse\"\x81\x01\n\x16\x44ownloadRunHistoryInit\x12\x0e\n\x06\x65ntity\x18\x01 \x01(\t\x12\x0f\n\x07project\x18\x02 \x01(\t\x12\x0e\n\x06run_id\x18\x03 \x01(\t\x12\x14\n\x0c\x64ownload_dir\x18\x04 \x01(\t\x12 \n\x18require_complete_history\x18\x05 \x01(\x08\"P\n\x1e\x44ownloadRunHistoryInitResponse\x12\x12\n\nrequest_id\x18\x01 \x01(\x05\x12\x1a\n\x12\x63ontains_live_data\x18\x02 \x01(\x08\"(\n\x12\x44ownloadRunHistory\x12\x12\n\nrequest_id\x18\x01 \x01(\x05\"\xad\x01\n\x1a\x44ownloadRunHistoryResponse\x12\x18\n\x10\x64ownloaded_files\x18\x01 \x03(\t\x12\x46\n\x06\x65rrors\x18\x02 \x03(\x0b\x32\x36.wandb_internal.DownloadRunHistoryResponse.ErrorsEntry\x1a-\n\x0b\x45rrorsEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t:\x02\x38\x01\"\x1b\n\x19IncompleteRunHistoryError\".\n\x18\x44ownloadRunHistoryStatus\x12\x12\n\nrequest_id\x18\x01 \x01(\x05\"[\n DownloadRunHistoryStatusResponse\x12\x37\n\x0foperation_stats\x18\x01 \x01(\x0b\x32\x1e.wandb_internal.OperationStats*@\n\tErrorType\x12\x11\n\rUNKNOWN_ERROR\x10\x00\x12 \n\x1cINCOMPLETE_RUN_HISTORY_ERROR\x10\x01\x42\x1bZ\x19\x63ore/pkg/service_go_protob\x06proto3')
/n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/wandb/proto/v5/wandb_settings_pb2.py:18:DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n wandb/proto/wandb_settings.proto\x12\x0ewandb_internal\x1a\x1egoogle/protobuf/wrappers.proto\" \n\x0fListStringValue\x12\r\n\x05value\x18\x01 \x03(\t\"\x1d\n\x0cListIntValue\x12\r\n\x05value\x18\x01 \x03(\x05\"\x8a\x01\n\x17MapStringKeyStringValue\x12\x41\n\x05value\x18\x01 \x03(\x0b\x32\x32.wandb_internal.MapStringKeyStringValue.ValueEntry\x1a,\n\nValueEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t:\x02\x38\x01\"\xcb\x01\n#MapStringKeyMapStringKeyStringValue\x12M\n\x05value\x18\x01 \x03(\x0b\x32>.wandb_internal.MapStringKeyMapStringKeyStringValue.ValueEntry\x1aU\n\nValueEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\x36\n\x05value\x18\x02 \x01(\x0b\x32\'.wandb_internal.MapStringKeyStringValue:\x02\x38\x01\"\x9a\x01\n\x12OpenMetricsFilters\x12\x33\n\x08sequence\x18\x01 \x01(\x0b\x32\x1f.wandb_internal.ListStringValueH\x00\x12\x46\n\x07mapping\x18\x02 \x01(\x0b\x32\x33.wandb_internal.MapStringKeyMapStringKeyStringValueH\x00\x42\x07\n\x05value\"7\n\tRunMoment\x12\x0b\n\x03run\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\x01\x12\x0e\n\x06metric\x18\x03 \x01(\t\"\xbeO\n\x08Settings\x12-\n\x07\x61pi_key\x18\x37 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12:\n\x13identity_token_file\x18\xaa\x01 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12\x37\n\x10\x63redentials_file\x18\xab\x01 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12\x39\n\x14insecure_disable_ssl\x18\xb9\x01 \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12,\n\x08_offline\x18\x1e \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12*\n\x06x_sync\x18\x1f \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12\x30\n\tsync_file\x18\x86\x01 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12,\n\x07_shared\x18\xa2\x01 \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12,\n\x06run_id\x18k \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12-\n\x07run_url\x18q \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12-\n\x07project\x18\x61 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12,\n\x06\x65ntity\x18\x45 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12\x33\n\x0corganization\x18\xbc\x01 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12\x32\n\x0cx_start_time\x18) \x01(\x0b\x32\x1c.google.protobuf.DoubleValue\x12.\n\x08root_dir\x18i \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12\x30\n\twandb_dir\x18\x8e\x01 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12-\n\x07log_dir\x18U \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12\x32\n\x0clog_internal\x18V \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12\x35\n\x0cignore_globs\x18N \x01(\x0b\x32\x1f.wandb_internal.ListStringValue\x12.\n\x07\x61pp_url\x18\xca\x01 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12.\n\x08\x62\x61se_url\x18\x39 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12=\n\x17x_file_stream_max_bytes\x18\xac\x01 \x01(\x0b\x32\x1b.google.protobuf.Int32Value\x12\x46\n\x1fx_file_stream_transmit_interval\x18\xaf\x01 \x01(\x0b\x32\x1c.google.protobuf.DoubleValue\x12\x45\n\x14x_extra_http_headers\x18\x0e \x01(\x0b\x32\'.wandb_internal.MapStringKeyStringValue\x12=\n\x17x_file_stream_retry_max\x18\x93\x01 \x01(\x0b\x32\x1b.google.protobuf.Int32Value\x12K\n$x_file_stream_retry_wait_min_seconds\x18\x94\x01 \x01(\x0b\x32\x1c.google.protobuf.DoubleValue\x12K\n$x_file_stream_retry_wait_max_seconds\x18\x95\x01 \x01(\x0b\x32\x1c.google.protobuf.DoubleValue\x12\x43\n\x1dx_file_stream_timeout_seconds\x18\x0f \x01(\x0b\x32\x1c.google.protobuf.DoubleValue\x12\x42\n\x1cx_file_stream_max_line_bytes\x18\xb2\x01 \x01(\x0b\x32\x1b.google.protobuf.Int32Value\x12?\n\x19x_file_transfer_retry_max\x18\x96\x01 \x01(\x0b\x32\x1b.google.protobuf.Int32Value\x12M\n&x_file_transfer_retry_wait_min_seconds\x18\x97\x01 \x01(\x0b\x32\x1c.google.protobuf.DoubleValue\x12M\n&x_file_transfer_retry_wait_max_seconds\x18\x98\x01 \x01(\x0b\x32\x1c.google.protobuf.DoubleValue\x12\x46\n\x1fx_file_transfer_timeout_seconds\x18\x99\x01 \x01(\x0b\x32\x1c.google.protobuf.DoubleValue\x12\x39\n\x13x_graphql_retry_max\x18\x9a\x01 \x01(\x0b\x32\x1b.google.protobuf.Int32Value\x12G\n x_graphql_retry_wait_min_seconds\x18\x9b\x01 \x01(\x0b\x32\x1c.google.protobuf.DoubleValue\x12G\n x_graphql_retry_wait_max_seconds\x18\x9c\x01 \x01(\x0b\x32\x1c.google.protobuf.DoubleValue\x12@\n\x19x_graphql_timeout_seconds\x18\x9d\x01 \x01(\x0b\x32\x1c.google.protobuf.DoubleValue\x12\x31\n\nhttp_proxy\x18\xa8\x01 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12\x32\n\x0bhttps_proxy\x18\xa9\x01 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12;\n\tx_proxies\x18\xc8\x01 \x01(\x0b\x32\'.wandb_internal.MapStringKeyStringValue\x12-\n\x07program\x18_ \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12\x35\n\x0fprogram_relpath\x18` \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12\x37\n\x10_code_path_local\x18\xa3\x01 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12\x36\n\x0fprogram_abspath\x18\x9f\x01 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12.\n\x05_args\x18\x01 \x01(\x0b\x32\x1f.wandb_internal.ListStringValue\x12)\n\x03_os\x18  \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12,\n\x06\x64ocker\x18\x43 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12\x32\n\x0cx_executable\x18\r \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12-\n\x07_python\x18\" \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12\x30\n\tcolab_url\x18\xa0\x01 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12*\n\x04host\x18M \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12/\n\x08username\x18\x8d\x01 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12+\n\x05\x65mail\x18\x44 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12,\n\x06resume\x18\x66 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12/\n\x0bresume_from\x18\xa7\x01 \x01(\x0b\x32\x19.wandb_internal.RunMoment\x12-\n\tfork_from\x18\xa4\x01 \x01(\x0b\x32\x19.wandb_internal.RunMoment\x12\x38\n\x14\x64isable_job_creation\x18\x41 \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12\x30\n\tsweep_url\x18\x83\x01 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12;\n\x16x_disable_update_check\x18\xa5\x01 \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12\x32\n\x0ex_disable_meta\x18\x07 \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12-\n\tsave_code\x18s \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12/\n\x0b\x64isable_git\x18? \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12;\n\x16\x64isable_git_fork_point\x18\xcb\x01 \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12;\n\x16x_disable_machine_info\x18\x9e\x01 \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12\x33\n\x0fx_disable_stats\x18\n \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12\x39\n\x13x_stats_buffer_size\x18\xa1\x01 \x01(\x0b\x32\x1b.google.protobuf.Int32Value\x12@\n\x19x_stats_sampling_interval\x18\xae\x01 \x01(\x0b\x32\x1c.google.protobuf.DoubleValue\x12\x30\n\x0bx_stats_pid\x18* \x01(\x0b\x32\x1b.google.protobuf.Int32Value\x12<\n\x12x_stats_disk_paths\x18\x92\x01 \x01(\x0b\x32\x1f.wandb_internal.ListStringValue\x12H\n\"x_stats_neuron_monitor_config_path\x18. \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12<\n\x15x_stats_dcgm_exporter\x18\xbb\x01 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12O\n\x1ex_stats_open_metrics_endpoints\x18/ \x01(\x0b\x32\'.wandb_internal.MapStringKeyStringValue\x12H\n\x1cx_stats_open_metrics_filters\x18\x30 \x01(\x0b\x32\".wandb_internal.OpenMetricsFilters\x12S\n!x_stats_open_metrics_http_headers\x18\xb8\x01 \x01(\x0b\x32\'.wandb_internal.MapStringKeyStringValue\x12=\n\x16x_stats_gpu_device_ids\x18\xba\x01 \x01(\x0b\x32\x1c.wandb_internal.ListIntValue\x12\x37\n\x11x_stats_cpu_count\x18\xc2\x01 \x01(\x0b\x32\x1b.google.protobuf.Int32Value\x12?\n\x19x_stats_cpu_logical_count\x18\xc3\x01 \x01(\x0b\x32\x1b.google.protobuf.Int32Value\x12\x37\n\x11x_stats_gpu_count\x18\xc4\x01 \x01(\x0b\x32\x1b.google.protobuf.Int32Value\x12\x37\n\x10x_stats_gpu_type\x18\xc5\x01 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12?\n\x1ax_stats_track_process_tree\x18\xc6\x01 \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12.\n\x07x_label\x18\xb5\x01 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12.\n\tx_primary\x18\xb6\x01 \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12:\n\x15x_update_finish_state\x18\xb7\x01 \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12<\n\x17\x61llow_offline_artifacts\x18\xb1\x01 \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12-\n\x07\x63onsole\x18< \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12\x36\n\x11\x63onsole_multipart\x18\xa6\x01 \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12=\n\x17\x63onsole_chunk_max_bytes\x18\xc7\x01 \x01(\x0b\x32\x1b.google.protobuf.Int32Value\x12?\n\x19\x63onsole_chunk_max_seconds\x18\xc9\x01 \x01(\x0b\x32\x1b.google.protobuf.Int32Value\x12\x35\n\x10sync_tensorboard\x18\xb3\x01 \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12\x42\n\x1dx_server_side_derived_summary\x18\xbd\x01 \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12\x46\n!x_server_side_expand_glob_metrics\x18\xbe\x01 \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12;\n\x16x_skip_transaction_log\x18\xbf\x01 \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12J\n#x_stats_coreweave_metadata_base_url\x18\xc0\x01 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12J\n#x_stats_coreweave_metadata_endpoint\x18\xc1\x01 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12/\n\x0b_aws_lambda\x18\x02 \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12\x33\n\x0fx_cli_only_mode\x18\x04 \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12*\n\x06_colab\x18\x05 \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12\x34\n\x10x_disable_viewer\x18\x0b \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12\x39\n\x15x_flow_control_custom\x18\x10 \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12;\n\x17x_flow_control_disabled\x18\x11 \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12>\n\x18x_internal_check_process\x18\x12 \x01(\x0b\x32\x1c.google.protobuf.DoubleValue\x12,\n\x08_ipython\x18\x14 \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12,\n\x08_jupyter\x18\x15 \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12\x34\n\x0ex_jupyter_root\x18\x16 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12+\n\x07_kaggle\x18\x17 \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12=\n\x18x_live_policy_rate_limit\x18\x18 \x01(\x0b\x32\x1b.google.protobuf.Int32Value\x12<\n\x17x_live_policy_wait_time\x18\x19 \x01(\x0b\x32\x1b.google.protobuf.Int32Value\x12\x30\n\x0bx_log_level\x18\x1a \x01(\x0b\x32\x1b.google.protobuf.Int32Value\x12\x35\n\x10x_network_buffer\x18\x1b \x01(\x0b\x32\x1b.google.protobuf.Int32Value\x12)\n\x05_noop\x18\x1c \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12-\n\t_notebook\x18\x1d \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12/\n\t_platform\x18! \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12\x38\n\x12x_runqueue_item_id\x18# \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12\x37\n\x13x_save_requirements\x18% \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12\x39\n\x13x_service_transport\x18& \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12\x34\n\x0ex_service_wait\x18\' \x01(\x0b\x32\x1c.google.protobuf.DoubleValue\x12\x35\n\x0f_start_datetime\x18( \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12\x33\n\r_tmp_code_dir\x18\x31 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12,\n\x08_windows\x18\x34 \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12\x38\n\x13\x61llow_media_symlink\x18\xcc\x01 \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12\x34\n\x10\x61llow_val_change\x18\x35 \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12P\n\x1f\x61zure_account_url_to_access_key\x18\x38 \x01(\x0b\x32\'.wandb_internal.MapStringKeyStringValue\x12.\n\x08\x63ode_dir\x18: \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12\x35\n\x0c\x63onfig_paths\x18; \x01(\x0b\x32\x1f.wandb_internal.ListStringValue\x12\x30\n\ndeployment\x18= \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12\x30\n\x0c\x64isable_code\x18> \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12\x31\n\rdisable_hints\x18@ \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12,\n\x08\x64isabled\x18\x42 \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12)\n\x05\x66orce\x18G \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12\x30\n\ngit_commit\x18H \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12\x30\n\ngit_remote\x18I \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12\x34\n\x0egit_remote_url\x18J \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12.\n\x08git_root\x18K \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12\x36\n\x11heartbeat_seconds\x18L \x01(\x0b\x32\x1b.google.protobuf.Int32Value\x12\x32\n\x0cinit_timeout\x18O \x01(\x0b\x32\x1c.google.protobuf.DoubleValue\x12,\n\x08is_local\x18P \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12\x30\n\njob_source\x18Q \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12\x31\n\rlabel_disable\x18R \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12*\n\x06launch\x18S \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12\x38\n\x12launch_config_path\x18T \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12:\n\x14log_symlink_internal\x18W \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12\x36\n\x10log_symlink_user\x18X \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12.\n\x08log_user\x18Y \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12\x33\n\rlogin_timeout\x18Z \x01(\x0b\x32\x1c.google.protobuf.DoubleValue\x12*\n\x04mode\x18\\ \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12\x33\n\rnotebook_name\x18] \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12\x31\n\x0bproject_url\x18\x62 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12)\n\x05quiet\x18\x63 \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12+\n\x07relogin\x18\x65 \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12\x32\n\x0cresume_fname\x18g \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12+\n\x07resumed\x18h \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12/\n\trun_group\x18j \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12\x32\n\x0crun_job_type\x18l \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12.\n\x08run_mode\x18m \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12.\n\x08run_name\x18n \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12/\n\trun_notes\x18o \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12\x31\n\x08run_tags\x18p \x01(\x0b\x32\x1f.wandb_internal.ListStringValue\x12\x35\n\x11sagemaker_disable\x18r \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12\x35\n\x0fsettings_system\x18t \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12\x38\n\x12settings_workspace\x18u \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12/\n\x0bshow_colors\x18v \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12.\n\nshow_emoji\x18w \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12/\n\x0bshow_errors\x18x \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12-\n\tshow_info\x18y \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12\x31\n\rshow_warnings\x18z \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12*\n\x06silent\x18{ \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12\x32\n\x0cstart_method\x18| \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12*\n\x06strict\x18} \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12\x33\n\x0esummary_errors\x18~ \x01(\x0b\x32\x1b.google.protobuf.Int32Value\x12\x34\n\x0fsummary_timeout\x18\x7f \x01(\x0b\x32\x1b.google.protobuf.Int32Value\x12\x36\n\x10summary_warnings\x18\x80\x01 \x01(\x0b\x32\x1b.google.protobuf.Int32Value\x12/\n\x08sweep_id\x18\x81\x01 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12\x37\n\x10sweep_param_path\x18\x82\x01 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12,\n\x07symlink\x18\x84\x01 \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12/\n\x08sync_dir\x18\x85\x01 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12:\n\x13sync_symlink_latest\x18\x87\x01 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12J\n%table_raise_on_max_row_limit_exceeded\x18\x8a\x01 \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12/\n\x08timespec\x18\x8b\x01 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12.\n\x07tmp_dir\x18\x8c\x01 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12\x35\n\x0ex_jupyter_name\x18\x8f\x01 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12\x35\n\x0ex_jupyter_path\x18\x90\x01 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12/\n\x08job_name\x18\x91\x01 \x01(\x0b\x32\x1c.google.protobuf.StringValueJ\x04\x08\x03\x10\x04J\x04\x08\x06\x10\x07J\x04\x08\x08\x10\tJ\x04\x08\t\x10\nJ\x04\x08\x0c\x10\rJ\x04\x08\x13\x10\x14J\x04\x08$\x10%J\x04\x08+\x10,J\x04\x08,\x10-J\x04\x08-\x10.J\x04\x08\x32\x10\x33J\x04\x08\x33\x10\x34J\x04\x08\x36\x10\x37J\x04\x08\x46\x10GJ\x04\x08[\x10\\J\x04\x08^\x10_J\x04\x08\x64\x10\x65J\x06\x08\x88\x01\x10\x89\x01J\x06\x08\x89\x01\x10\x8a\x01J\x06\x08\xad\x01\x10\xae\x01J\x06\x08\xb0\x01\x10\xb1\x01J\x06\x08\xb4\x01\x10\xb5\x01\x42\x1bZ\x19\x63ore/pkg/service_go_protob\x06proto3')
/n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/wandb/proto/v5/wandb_internal_pb2.py:21:DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n wandb/proto/wandb_internal.proto\x12\x0ewandb_internal\x1a\x1bgoogle/protobuf/empty.proto\x1a\x1fgoogle/protobuf/timestamp.proto\x1a\x1cwandb/proto/wandb_base.proto\x1a!wandb/proto/wandb_telemetry.proto\"\xcf\t\n\x06Record\x12\x0b\n\x03num\x18\x01 \x01(\x03\x12\x30\n\x07history\x18\x02 \x01(\x0b\x32\x1d.wandb_internal.HistoryRecordH\x00\x12\x30\n\x07summary\x18\x03 \x01(\x0b\x32\x1d.wandb_internal.SummaryRecordH\x00\x12.\n\x06output\x18\x04 \x01(\x0b\x32\x1c.wandb_internal.OutputRecordH\x00\x12.\n\x06\x63onfig\x18\x05 \x01(\x0b\x32\x1c.wandb_internal.ConfigRecordH\x00\x12,\n\x05\x66iles\x18\x06 \x01(\x0b\x32\x1b.wandb_internal.FilesRecordH\x00\x12,\n\x05stats\x18\x07 \x01(\x0b\x32\x1b.wandb_internal.StatsRecordH\x00\x12\x32\n\x08\x61rtifact\x18\x08 \x01(\x0b\x32\x1e.wandb_internal.ArtifactRecordH\x00\x12,\n\x08tbrecord\x18\t \x01(\x0b\x32\x18.wandb_internal.TBRecordH\x00\x12,\n\x05\x61lert\x18\n \x01(\x0b\x32\x1b.wandb_internal.AlertRecordH\x00\x12\x34\n\ttelemetry\x18\x0b \x01(\x0b\x32\x1f.wandb_internal.TelemetryRecordH\x00\x12.\n\x06metric\x18\x0c \x01(\x0b\x32\x1c.wandb_internal.MetricRecordH\x00\x12\x35\n\noutput_raw\x18\r \x01(\x0b\x32\x1f.wandb_internal.OutputRawRecordH\x00\x12(\n\x03run\x18\x11 \x01(\x0b\x32\x19.wandb_internal.RunRecordH\x00\x12-\n\x04\x65xit\x18\x12 \x01(\x0b\x32\x1d.wandb_internal.RunExitRecordH\x00\x12,\n\x05\x66inal\x18\x14 \x01(\x0b\x32\x1b.wandb_internal.FinalRecordH\x00\x12.\n\x06header\x18\x15 \x01(\x0b\x32\x1c.wandb_internal.HeaderRecordH\x00\x12.\n\x06\x66ooter\x18\x16 \x01(\x0b\x32\x1c.wandb_internal.FooterRecordH\x00\x12\x39\n\npreempting\x18\x17 \x01(\x0b\x32#.wandb_internal.RunPreemptingRecordH\x00\x12\x34\n\x12noop_link_artifact\x18\x18 \x01(\x0b\x32\x16.google.protobuf.EmptyH\x00\x12\x39\n\x0cuse_artifact\x18\x19 \x01(\x0b\x32!.wandb_internal.UseArtifactRecordH\x00\x12\x38\n\x0b\x65nvironment\x18\x1a \x01(\x0b\x32!.wandb_internal.EnvironmentRecordH\x00\x12*\n\x07request\x18\x64 \x01(\x0b\x32\x17.wandb_internal.RequestH\x00\x12(\n\x07\x63ontrol\x18\x10 \x01(\x0b\x32\x17.wandb_internal.Control\x12\x0c\n\x04uuid\x18\x13 \x01(\t\x12+\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1b.wandb_internal._RecordInfoB\r\n\x0brecord_type\"\xa8\x01\n\x07\x43ontrol\x12\x10\n\x08req_resp\x18\x01 \x01(\x08\x12\r\n\x05local\x18\x02 \x01(\x08\x12\x10\n\x08relay_id\x18\x03 \x01(\t\x12\x14\n\x0cmailbox_slot\x18\x04 \x01(\t\x12\x13\n\x0b\x61lways_send\x18\x05 \x01(\x08\x12\x14\n\x0c\x66low_control\x18\x06 \x01(\x08\x12\x12\n\nend_offset\x18\x07 \x01(\x03\x12\x15\n\rconnection_id\x18\x08 \x01(\t\"\xf3\x03\n\x06Result\x12\x35\n\nrun_result\x18\x11 \x01(\x0b\x32\x1f.wandb_internal.RunUpdateResultH\x00\x12\x34\n\x0b\x65xit_result\x18\x12 \x01(\x0b\x32\x1d.wandb_internal.RunExitResultH\x00\x12\x33\n\nlog_result\x18\x14 \x01(\x0b\x32\x1d.wandb_internal.HistoryResultH\x00\x12\x37\n\x0esummary_result\x18\x15 \x01(\x0b\x32\x1d.wandb_internal.SummaryResultH\x00\x12\x35\n\routput_result\x18\x16 \x01(\x0b\x32\x1c.wandb_internal.OutputResultH\x00\x12\x35\n\rconfig_result\x18\x17 \x01(\x0b\x32\x1c.wandb_internal.ConfigResultH\x00\x12,\n\x08response\x18\x64 \x01(\x0b\x32\x18.wandb_internal.ResponseH\x00\x12(\n\x07\x63ontrol\x18\x10 \x01(\x0b\x32\x17.wandb_internal.Control\x12\x0c\n\x04uuid\x18\x18 \x01(\t\x12+\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1b.wandb_internal._ResultInfoB\r\n\x0bresult_type\":\n\x0b\x46inalRecord\x12+\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1b.wandb_internal._RecordInfo\"b\n\x0bVersionInfo\x12\x10\n\x08producer\x18\x01 \x01(\t\x12\x14\n\x0cmin_consumer\x18\x02 \x01(\t\x12+\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1b.wandb_internal._RecordInfo\"n\n\x0cHeaderRecord\x12\x31\n\x0cversion_info\x18\x01 \x01(\x0b\x32\x1b.wandb_internal.VersionInfo\x12+\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1b.wandb_internal._RecordInfo\";\n\x0c\x46ooterRecord\x12+\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1b.wandb_internal._RecordInfo\"9\n\x0b\x42ranchPoint\x12\x0b\n\x03run\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\x01\x12\x0e\n\x06metric\x18\x03 \x01(\t\"\x91\x05\n\tRunRecord\x12\x0e\n\x06run_id\x18\x01 \x01(\t\x12\x0e\n\x06\x65ntity\x18\x02 \x01(\t\x12\x0f\n\x07project\x18\x03 \x01(\t\x12,\n\x06\x63onfig\x18\x04 \x01(\x0b\x32\x1c.wandb_internal.ConfigRecord\x12.\n\x07summary\x18\x05 \x01(\x0b\x32\x1d.wandb_internal.SummaryRecord\x12\x11\n\trun_group\x18\x06 \x01(\t\x12\x10\n\x08job_type\x18\x07 \x01(\t\x12\x14\n\x0c\x64isplay_name\x18\x08 \x01(\t\x12\r\n\x05notes\x18\t \x01(\t\x12\x0c\n\x04tags\x18\n \x03(\t\x12\x30\n\x08settings\x18\x0b \x01(\x0b\x32\x1e.wandb_internal.SettingsRecord\x12\x10\n\x08sweep_id\x18\x0c \x01(\t\x12\x0c\n\x04host\x18\r \x01(\t\x12\x15\n\rstarting_step\x18\x0e \x01(\x03\x12\x12\n\nstorage_id\x18\x10 \x01(\t\x12.\n\nstart_time\x18\x11 \x01(\x0b\x32\x1a.google.protobuf.Timestamp\x12\x0f\n\x07resumed\x18\x12 \x01(\x08\x12\x32\n\ttelemetry\x18\x13 \x01(\x0b\x32\x1f.wandb_internal.TelemetryRecord\x12\x0f\n\x07runtime\x18\x14 \x01(\x05\x12*\n\x03git\x18\x15 \x01(\x0b\x32\x1d.wandb_internal.GitRepoRecord\x12\x0e\n\x06\x66orked\x18\x16 \x01(\x08\x12\x31\n\x0c\x62ranch_point\x18\x17 \x01(\x0b\x32\x1b.wandb_internal.BranchPoint\x12+\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1b.wandb_internal._RecordInfo\";\n\rGitRepoRecord\x12\x1a\n\nremote_url\x18\x01 \x01(\tR\x06remote\x12\x0e\n\x06\x63ommit\x18\x02 \x01(\t\"c\n\x0fRunUpdateResult\x12&\n\x03run\x18\x01 \x01(\x0b\x32\x19.wandb_internal.RunRecord\x12(\n\x05\x65rror\x18\x02 \x01(\x0b\x32\x19.wandb_internal.ErrorInfo\"\xac\x01\n\tErrorInfo\x12\x0f\n\x07message\x18\x01 \x01(\t\x12\x31\n\x04\x63ode\x18\x02 \x01(\x0e\x32#.wandb_internal.ErrorInfo.ErrorCode\"[\n\tErrorCode\x12\x0b\n\x07UNKNOWN\x10\x00\x12\x11\n\rCOMMUNICATION\x10\x01\x12\x12\n\x0e\x41UTHENTICATION\x10\x02\x12\t\n\x05USAGE\x10\x03\x12\x0f\n\x0bUNSUPPORTED\x10\x04\"v\n\rRunExitRecord\x12\x11\n\texit_code\x18\x01 \x01(\x05\x12\x14\n\x0cnot_complete\x18\x03 \x01(\x08\x12\x0f\n\x07runtime\x18\x02 \x01(\x05\x12+\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1b.wandb_internal._RecordInfo\"\x0f\n\rRunExitResult\"B\n\x13RunPreemptingRecord\x12+\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1b.wandb_internal._RecordInfo\"\x15\n\x13RunPreemptingResult\"i\n\x0eSettingsRecord\x12*\n\x04item\x18\x01 \x03(\x0b\x32\x1c.wandb_internal.SettingsItem\x12+\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1b.wandb_internal._RecordInfo\"/\n\x0cSettingsItem\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\x12\n\nvalue_json\x18\x10 \x01(\t\"\x1a\n\x0bHistoryStep\x12\x0b\n\x03num\x18\x01 \x01(\x03\"\x92\x01\n\rHistoryRecord\x12)\n\x04item\x18\x01 \x03(\x0b\x32\x1b.wandb_internal.HistoryItem\x12)\n\x04step\x18\x02 \x01(\x0b\x32\x1b.wandb_internal.HistoryStep\x12+\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1b.wandb_internal._RecordInfo\"B\n\x0bHistoryItem\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\x12\n\nnested_key\x18\x02 \x03(\t\x12\x12\n\nvalue_json\x18\x10 \x01(\t\"\x0f\n\rHistoryResult\"\xdc\x01\n\x0cOutputRecord\x12<\n\x0boutput_type\x18\x01 \x01(\x0e\x32\'.wandb_internal.OutputRecord.OutputType\x12-\n\ttimestamp\x18\x02 \x01(\x0b\x32\x1a.google.protobuf.Timestamp\x12\x0c\n\x04line\x18\x03 \x01(\t\x12+\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1b.wandb_internal._RecordInfo\"$\n\nOutputType\x12\n\n\x06STDERR\x10\x00\x12\n\n\x06STDOUT\x10\x01\"\x0e\n\x0cOutputResult\"\xe2\x01\n\x0fOutputRawRecord\x12?\n\x0boutput_type\x18\x01 \x01(\x0e\x32*.wandb_internal.OutputRawRecord.OutputType\x12-\n\ttimestamp\x18\x02 \x01(\x0b\x32\x1a.google.protobuf.Timestamp\x12\x0c\n\x04line\x18\x03 \x01(\t\x12+\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1b.wandb_internal._RecordInfo\"$\n\nOutputType\x12\n\n\x06STDERR\x10\x00\x12\n\n\x06STDOUT\x10\x01\"\x11\n\x0fOutputRawResult\"\xb4\x03\n\x0cMetricRecord\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x11\n\tglob_name\x18\x02 \x01(\t\x12\x13\n\x0bstep_metric\x18\x04 \x01(\t\x12\x19\n\x11step_metric_index\x18\x05 \x01(\x05\x12.\n\x07options\x18\x06 \x01(\x0b\x32\x1d.wandb_internal.MetricOptions\x12.\n\x07summary\x18\x07 \x01(\x0b\x32\x1d.wandb_internal.MetricSummary\x12\x35\n\x04goal\x18\x08 \x01(\x0e\x32\'.wandb_internal.MetricRecord.MetricGoal\x12/\n\x08_control\x18\t \x01(\x0b\x32\x1d.wandb_internal.MetricControl\x12\x1a\n\x12\x65xpanded_from_glob\x18\n \x01(\x08\x12+\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1b.wandb_internal._RecordInfo\"B\n\nMetricGoal\x12\x0e\n\nGOAL_UNSET\x10\x00\x12\x11\n\rGOAL_MINIMIZE\x10\x01\x12\x11\n\rGOAL_MAXIMIZE\x10\x02\"\x0e\n\x0cMetricResult\"C\n\rMetricOptions\x12\x11\n\tstep_sync\x18\x01 \x01(\x08\x12\x0e\n\x06hidden\x18\x02 \x01(\x08\x12\x0f\n\x07\x64\x65\x66ined\x18\x03 \x01(\x08\"\"\n\rMetricControl\x12\x11\n\toverwrite\x18\x01 \x01(\x08\"~\n\rMetricSummary\x12\x0b\n\x03min\x18\x01 \x01(\x08\x12\x0b\n\x03max\x18\x02 \x01(\x08\x12\x0c\n\x04mean\x18\x03 \x01(\x08\x12\x0c\n\x04\x62\x65st\x18\x04 \x01(\x08\x12\x0c\n\x04last\x18\x05 \x01(\x08\x12\x0c\n\x04none\x18\x06 \x01(\x08\x12\x0c\n\x04\x63opy\x18\x07 \x01(\x08\x12\r\n\x05\x66irst\x18\x08 \x01(\x08\"\x93\x01\n\x0c\x43onfigRecord\x12*\n\x06update\x18\x01 \x03(\x0b\x32\x1a.wandb_internal.ConfigItem\x12*\n\x06remove\x18\x02 \x03(\x0b\x32\x1a.wandb_internal.ConfigItem\x12+\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1b.wandb_internal._RecordInfo\"A\n\nConfigItem\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\x12\n\nnested_key\x18\x02 \x03(\t\x12\x12\n\nvalue_json\x18\x10 \x01(\t\"\x0e\n\x0c\x43onfigResult\"\x96\x01\n\rSummaryRecord\x12+\n\x06update\x18\x01 \x03(\x0b\x32\x1b.wandb_internal.SummaryItem\x12+\n\x06remove\x18\x02 \x03(\x0b\x32\x1b.wandb_internal.SummaryItem\x12+\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1b.wandb_internal._RecordInfo\"B\n\x0bSummaryItem\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\x12\n\nnested_key\x18\x02 \x03(\t\x12\x12\n\nvalue_json\x18\x10 \x01(\t\"\x0f\n\rSummaryResult\"d\n\x0b\x46ilesRecord\x12(\n\x05\x66iles\x18\x01 \x03(\x0b\x32\x19.wandb_internal.FilesItem\x12+\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1b.wandb_internal._RecordInfo\"\xec\x01\n\tFilesItem\x12\x0c\n\x04path\x18\x01 \x01(\t\x12\x34\n\x06policy\x18\x02 \x01(\x0e\x32$.wandb_internal.FilesItem.PolicyType\x12\x30\n\x04type\x18\x03 \x01(\x0e\x32\".wandb_internal.FilesItem.FileType\"(\n\nPolicyType\x12\x07\n\x03NOW\x10\x00\x12\x07\n\x03\x45ND\x10\x01\x12\x08\n\x04LIVE\x10\x02\"9\n\x08\x46ileType\x12\t\n\x05OTHER\x10\x00\x12\t\n\x05WANDB\x10\x01\x12\t\n\x05MEDIA\x10\x02\x12\x0c\n\x08\x41RTIFACT\x10\x03J\x04\x08\x10\x10\x11\"\r\n\x0b\x46ilesResult\"\xe6\x01\n\x0bStatsRecord\x12\x39\n\nstats_type\x18\x01 \x01(\x0e\x32%.wandb_internal.StatsRecord.StatsType\x12-\n\ttimestamp\x18\x02 \x01(\x0b\x32\x1a.google.protobuf.Timestamp\x12\'\n\x04item\x18\x03 \x03(\x0b\x32\x19.wandb_internal.StatsItem\x12+\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1b.wandb_internal._RecordInfo\"\x17\n\tStatsType\x12\n\n\x06SYSTEM\x10\x00\",\n\tStatsItem\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\x12\n\nvalue_json\x18\x10 \x01(\t\"\xe7\x03\n\x0e\x41rtifactRecord\x12\x0e\n\x06run_id\x18\x01 \x01(\t\x12\x0f\n\x07project\x18\x02 \x01(\t\x12\x0e\n\x06\x65ntity\x18\x03 \x01(\t\x12\x0c\n\x04type\x18\x04 \x01(\t\x12\x0c\n\x04name\x18\x05 \x01(\t\x12\x0e\n\x06\x64igest\x18\x06 \x01(\t\x12\x13\n\x0b\x64\x65scription\x18\x07 \x01(\t\x12\x10\n\x08metadata\x18\x08 \x01(\t\x12\x14\n\x0cuser_created\x18\t \x01(\x08\x12\x18\n\x10use_after_commit\x18\n \x01(\x08\x12\x0f\n\x07\x61liases\x18\x0b \x03(\t\x12\x32\n\x08manifest\x18\x0c \x01(\x0b\x32 .wandb_internal.ArtifactManifest\x12\x16\n\x0e\x64istributed_id\x18\r \x01(\t\x12\x10\n\x08\x66inalize\x18\x0e \x01(\x08\x12\x11\n\tclient_id\x18\x0f \x01(\t\x12\x1a\n\x12sequence_client_id\x18\x10 \x01(\t\x12\x0f\n\x07\x62\x61se_id\x18\x11 \x01(\t\x12\x1c\n\x14ttl_duration_seconds\x18\x12 \x01(\x03\x12\x0c\n\x04tags\x18\x13 \x03(\t\x12\x19\n\x11incremental_beta1\x18\x64 \x01(\x08\x12+\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1b.wandb_internal._RecordInfo\"\xd8\x01\n\x10\x41rtifactManifest\x12\x0f\n\x07version\x18\x01 \x01(\x05\x12\x16\n\x0estorage_policy\x18\x02 \x01(\t\x12\x46\n\x15storage_policy_config\x18\x03 \x03(\x0b\x32\'.wandb_internal.StoragePolicyConfigItem\x12\x37\n\x08\x63ontents\x18\x04 \x03(\x0b\x32%.wandb_internal.ArtifactManifestEntry\x12\x1a\n\x12manifest_file_path\x18\x05 \x01(\t\"\xcf\x01\n\x15\x41rtifactManifestEntry\x12\x0c\n\x04path\x18\x01 \x01(\t\x12\x0e\n\x06\x64igest\x18\x02 \x01(\t\x12\x0b\n\x03ref\x18\x03 \x01(\t\x12\x0c\n\x04size\x18\x04 \x01(\x03\x12\x10\n\x08mimetype\x18\x05 \x01(\t\x12\x12\n\nlocal_path\x18\x06 \x01(\t\x12\x19\n\x11\x62irth_artifact_id\x18\x07 \x01(\t\x12\x12\n\nskip_cache\x18\x08 \x01(\x08\x12(\n\x05\x65xtra\x18\x10 \x03(\x0b\x32\x19.wandb_internal.ExtraItem\",\n\tExtraItem\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\x12\n\nvalue_json\x18\x02 \x01(\t\":\n\x17StoragePolicyConfigItem\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\x12\n\nvalue_json\x18\x02 \x01(\t\"\x10\n\x0e\x41rtifactResult\"\x14\n\x12LinkArtifactResult\"\xf0\x01\n\x13LinkArtifactRequest\x12\x11\n\tclient_id\x18\x01 \x01(\t\x12\x11\n\tserver_id\x18\x02 \x01(\t\x12\x16\n\x0eportfolio_name\x18\x03 \x01(\t\x12\x18\n\x10portfolio_entity\x18\x04 \x01(\t\x12\x19\n\x11portfolio_project\x18\x05 \x01(\t\x12\x19\n\x11portfolio_aliases\x18\x06 \x03(\t\x12\x1e\n\x16portfolio_organization\x18\x07 \x01(\t\x12+\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1b.wandb_internal._RecordInfo\"[\n\x14LinkArtifactResponse\x12\x15\n\rerror_message\x18\x01 \x01(\t\x12\x1a\n\rversion_index\x18\x02 \x01(\x05H\x00\x88\x01\x01\x42\x10\n\x0e_version_index\"h\n\x08TBRecord\x12+\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1b.wandb_internal._RecordInfo\x12\x0f\n\x07log_dir\x18\x01 \x01(\t\x12\x10\n\x08root_dir\x18\x03 \x01(\t\x12\x0c\n\x04save\x18\x02 \x01(\x08\"\n\n\x08TBResult\"}\n\x0b\x41lertRecord\x12\r\n\x05title\x18\x01 \x01(\t\x12\x0c\n\x04text\x18\x02 \x01(\t\x12\r\n\x05level\x18\x03 \x01(\t\x12\x15\n\rwait_duration\x18\x04 \x01(\x03\x12+\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1b.wandb_internal._RecordInfo\"\r\n\x0b\x41lertResult\"\xf4\x10\n\x07Request\x12\x38\n\x0bstop_status\x18\x01 \x01(\x0b\x32!.wandb_internal.StopStatusRequestH\x00\x12>\n\x0enetwork_status\x18\x02 \x01(\x0b\x32$.wandb_internal.NetworkStatusRequestH\x00\x12-\n\x05\x64\x65\x66\x65r\x18\x03 \x01(\x0b\x32\x1c.wandb_internal.DeferRequestH\x00\x12\x38\n\x0bget_summary\x18\x04 \x01(\x0b\x32!.wandb_internal.GetSummaryRequestH\x00\x12-\n\x05login\x18\x05 \x01(\x0b\x32\x1c.wandb_internal.LoginRequestH\x00\x12-\n\x05pause\x18\x06 \x01(\x0b\x32\x1c.wandb_internal.PauseRequestH\x00\x12/\n\x06resume\x18\x07 \x01(\x0b\x32\x1d.wandb_internal.ResumeRequestH\x00\x12\x34\n\tpoll_exit\x18\x08 \x01(\x0b\x32\x1f.wandb_internal.PollExitRequestH\x00\x12@\n\x0fsampled_history\x18\t \x01(\x0b\x32%.wandb_internal.SampledHistoryRequestH\x00\x12@\n\x0fpartial_history\x18\n \x01(\x0b\x32%.wandb_internal.PartialHistoryRequestH\x00\x12\x34\n\trun_start\x18\x0b \x01(\x0b\x32\x1f.wandb_internal.RunStartRequestH\x00\x12<\n\rcheck_version\x18\x0c \x01(\x0b\x32#.wandb_internal.CheckVersionRequestH\x00\x12:\n\x0clog_artifact\x18\r \x01(\x0b\x32\".wandb_internal.LogArtifactRequestH\x00\x12\x44\n\x11\x64ownload_artifact\x18\x0e \x01(\x0b\x32\'.wandb_internal.DownloadArtifactRequestH\x00\x12\x35\n\tkeepalive\x18\x11 \x01(\x0b\x32 .wandb_internal.KeepaliveRequestH\x00\x12\x36\n\nrun_status\x18\x14 \x01(\x0b\x32 .wandb_internal.RunStatusRequestH\x00\x12/\n\x06\x63\x61ncel\x18\x15 \x01(\x0b\x32\x1d.wandb_internal.CancelRequestH\x00\x12\x44\n\x11internal_messages\x18\x17 \x01(\x0b\x32\'.wandb_internal.InternalMessagesRequestH\x00\x12@\n\x0fpython_packages\x18\x18 \x01(\x0b\x32%.wandb_internal.PythonPackagesRequestH\x00\x12\x33\n\x08shutdown\x18@ \x01(\x0b\x32\x1f.wandb_internal.ShutdownRequestH\x00\x12/\n\x06\x61ttach\x18\x41 \x01(\x0b\x32\x1d.wandb_internal.AttachRequestH\x00\x12/\n\x06status\x18\x42 \x01(\x0b\x32\x1d.wandb_internal.StatusRequestH\x00\x12\x38\n\x0bserver_info\x18\x43 \x01(\x0b\x32!.wandb_internal.ServerInfoRequestH\x00\x12\x38\n\x0bsender_mark\x18\x44 \x01(\x0b\x32!.wandb_internal.SenderMarkRequestH\x00\x12\x38\n\x0bsender_read\x18\x45 \x01(\x0b\x32!.wandb_internal.SenderReadRequestH\x00\x12<\n\rstatus_report\x18\x46 \x01(\x0b\x32#.wandb_internal.StatusReportRequestH\x00\x12>\n\x0esummary_record\x18G \x01(\x0b\x32$.wandb_internal.SummaryRecordRequestH\x00\x12\x42\n\x10telemetry_record\x18H \x01(\x0b\x32&.wandb_internal.TelemetryRecordRequestH\x00\x12\x32\n\x08job_info\x18I \x01(\x0b\x32\x1e.wandb_internal.JobInfoRequestH\x00\x12\x45\n\x12get_system_metrics\x18J \x01(\x0b\x32\'.wandb_internal.GetSystemMetricsRequestH\x00\x12\x34\n\tjob_input\x18M \x01(\x0b\x32\x1f.wandb_internal.JobInputRequestH\x00\x12<\n\rlink_artifact\x18N \x01(\x0b\x32#.wandb_internal.LinkArtifactRequestH\x00\x12\x38\n\x0bsync_finish\x18Q \x01(\x0b\x32!.wandb_internal.SyncFinishRequestH\x00\x12;\n\noperations\x18R \x01(\x0b\x32%.wandb_internal.OperationStatsRequestH\x00\x12\x43\n\x11probe_system_info\x18S \x01(\x0b\x32&.wandb_internal.ProbeSystemInfoRequestH\x00\x12\x39\n\x0btest_inject\x18\xe8\x07 \x01(\x0b\x32!.wandb_internal.TestInjectRequestH\x00\x42\x0e\n\x0crequest_typeJ\x04\x08\x12\x10\x13J\x04\x08\x16\x10\x17J\x04\x08K\x10LJ\x04\x08L\x10MJ\x04\x08O\x10PJ\x04\x08P\x10Q\"\x83\r\n\x08Response\x12?\n\x12keepalive_response\x18\x12 \x01(\x0b\x32!.wandb_internal.KeepaliveResponseH\x00\x12\x42\n\x14stop_status_response\x18\x13 \x01(\x0b\x32\".wandb_internal.StopStatusResponseH\x00\x12H\n\x17network_status_response\x18\x14 \x01(\x0b\x32%.wandb_internal.NetworkStatusResponseH\x00\x12\x37\n\x0elogin_response\x18\x18 \x01(\x0b\x32\x1d.wandb_internal.LoginResponseH\x00\x12\x42\n\x14get_summary_response\x18\x19 \x01(\x0b\x32\".wandb_internal.GetSummaryResponseH\x00\x12>\n\x12poll_exit_response\x18\x1a \x01(\x0b\x32 .wandb_internal.PollExitResponseH\x00\x12J\n\x18sampled_history_response\x18\x1b \x01(\x0b\x32&.wandb_internal.SampledHistoryResponseH\x00\x12>\n\x12run_start_response\x18\x1c \x01(\x0b\x32 .wandb_internal.RunStartResponseH\x00\x12\x46\n\x16\x63heck_version_response\x18\x1d \x01(\x0b\x32$.wandb_internal.CheckVersionResponseH\x00\x12\x44\n\x15log_artifact_response\x18\x1e \x01(\x0b\x32#.wandb_internal.LogArtifactResponseH\x00\x12N\n\x1a\x64ownload_artifact_response\x18\x1f \x01(\x0b\x32(.wandb_internal.DownloadArtifactResponseH\x00\x12@\n\x13run_status_response\x18# \x01(\x0b\x32!.wandb_internal.RunStatusResponseH\x00\x12\x39\n\x0f\x63\x61ncel_response\x18$ \x01(\x0b\x32\x1e.wandb_internal.CancelResponseH\x00\x12N\n\x1ainternal_messages_response\x18% \x01(\x0b\x32(.wandb_internal.InternalMessagesResponseH\x00\x12=\n\x11shutdown_response\x18@ \x01(\x0b\x32 .wandb_internal.ShutdownResponseH\x00\x12\x39\n\x0f\x61ttach_response\x18\x41 \x01(\x0b\x32\x1e.wandb_internal.AttachResponseH\x00\x12\x39\n\x0fstatus_response\x18\x42 \x01(\x0b\x32\x1e.wandb_internal.StatusResponseH\x00\x12\x42\n\x14server_info_response\x18\x43 \x01(\x0b\x32\".wandb_internal.ServerInfoResponseH\x00\x12<\n\x11job_info_response\x18\x44 \x01(\x0b\x32\x1f.wandb_internal.JobInfoResponseH\x00\x12O\n\x1bget_system_metrics_response\x18\x45 \x01(\x0b\x32(.wandb_internal.GetSystemMetricsResponseH\x00\x12\x46\n\x16link_artifact_response\x18G \x01(\x0b\x32$.wandb_internal.LinkArtifactResponseH\x00\x12\x35\n\rsync_response\x18\x46 \x01(\x0b\x32\x1c.wandb_internal.SyncResponseH\x00\x12\x45\n\x13operations_response\x18J \x01(\x0b\x32&.wandb_internal.OperationStatsResponseH\x00\x12\x43\n\x14test_inject_response\x18\xe8\x07 \x01(\x0b\x32\".wandb_internal.TestInjectResponseH\x00\x42\x0f\n\rresponse_typeJ\x04\x08 \x10!J\x04\x08H\x10IJ\x04\x08I\x10J\"\xc0\x02\n\x0c\x44\x65\x66\x65rRequest\x12\x36\n\x05state\x18\x01 \x01(\x0e\x32\'.wandb_internal.DeferRequest.DeferState\"\xf7\x01\n\nDeferState\x12\t\n\x05\x42\x45GIN\x10\x00\x12\r\n\tFLUSH_RUN\x10\x01\x12\x0f\n\x0b\x46LUSH_STATS\x10\x02\x12\x19\n\x15\x46LUSH_PARTIAL_HISTORY\x10\x03\x12\x0c\n\x08\x46LUSH_TB\x10\x04\x12\r\n\tFLUSH_SUM\x10\x05\x12\x13\n\x0f\x46LUSH_DEBOUNCER\x10\x06\x12\x10\n\x0c\x46LUSH_OUTPUT\x10\x07\x12\r\n\tFLUSH_JOB\x10\x08\x12\r\n\tFLUSH_DIR\x10\t\x12\x0c\n\x08\x46LUSH_FP\x10\n\x12\x0b\n\x07JOIN_FP\x10\x0b\x12\x0c\n\x08\x46LUSH_FS\x10\x0c\x12\x0f\n\x0b\x46LUSH_FINAL\x10\r\x12\x07\n\x03\x45ND\x10\x0e\"<\n\x0cPauseRequest\x12,\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1c.wandb_internal._RequestInfo\"\x0f\n\rPauseResponse\"=\n\rResumeRequest\x12,\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1c.wandb_internal._RequestInfo\"\x10\n\x0eResumeResponse\"M\n\x0cLoginRequest\x12\x0f\n\x07\x61pi_key\x18\x01 \x01(\t\x12,\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1c.wandb_internal._RequestInfo\"&\n\rLoginResponse\x12\x15\n\ractive_entity\x18\x01 \x01(\t\"A\n\x11GetSummaryRequest\x12,\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1c.wandb_internal._RequestInfo\"?\n\x12GetSummaryResponse\x12)\n\x04item\x18\x01 \x03(\x0b\x32\x1b.wandb_internal.SummaryItem\"G\n\x17GetSystemMetricsRequest\x12,\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1c.wandb_internal._RequestInfo\"R\n\x12SystemMetricSample\x12-\n\ttimestamp\x18\x01 \x01(\x0b\x32\x1a.google.protobuf.Timestamp\x12\r\n\x05value\x18\x02 \x01(\x02\"I\n\x13SystemMetricsBuffer\x12\x32\n\x06record\x18\x01 \x03(\x0b\x32\".wandb_internal.SystemMetricSample\"\xca\x01\n\x18GetSystemMetricsResponse\x12S\n\x0esystem_metrics\x18\x01 \x03(\x0b\x32;.wandb_internal.GetSystemMetricsResponse.SystemMetricsEntry\x1aY\n\x12SystemMetricsEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\x32\n\x05value\x18\x02 \x01(\x0b\x32#.wandb_internal.SystemMetricsBuffer:\x02\x38\x01\"=\n\rStatusRequest\x12,\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1c.wandb_internal._RequestInfo\")\n\x0eStatusResponse\x12\x17\n\x0frun_should_stop\x18\x01 \x01(\x08\"A\n\x11StopStatusRequest\x12,\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1c.wandb_internal._RequestInfo\"-\n\x12StopStatusResponse\x12\x17\n\x0frun_should_stop\x18\x01 \x01(\x08\"D\n\x14NetworkStatusRequest\x12,\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1c.wandb_internal._RequestInfo\"P\n\x15NetworkStatusResponse\x12\x37\n\x11network_responses\x18\x01 \x03(\x0b\x32\x1c.wandb_internal.HttpResponse\"D\n\x0cHttpResponse\x12\x18\n\x10http_status_code\x18\x01 \x01(\x05\x12\x1a\n\x12http_response_text\x18\x02 \x01(\t\"G\n\x17InternalMessagesRequest\x12,\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1c.wandb_internal._RequestInfo\"N\n\x18InternalMessagesResponse\x12\x32\n\x08messages\x18\x01 \x01(\x0b\x32 .wandb_internal.InternalMessages\"#\n\x10InternalMessages\x12\x0f\n\x07warning\x18\x01 \x03(\t\"?\n\x0fPollExitRequest\x12,\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1c.wandb_internal._RequestInfo\"\xf5\x01\n\x10PollExitResponse\x12\x0c\n\x04\x64one\x18\x01 \x01(\x08\x12\x32\n\x0b\x65xit_result\x18\x02 \x01(\x0b\x32\x1d.wandb_internal.RunExitResult\x12\x35\n\x0cpusher_stats\x18\x03 \x01(\x0b\x32\x1f.wandb_internal.FilePusherStats\x12/\n\x0b\x66ile_counts\x18\x04 \x01(\x0b\x32\x1a.wandb_internal.FileCounts\x12\x37\n\x0foperation_stats\x18\x05 \x01(\x0b\x32\x1e.wandb_internal.OperationStats\"E\n\x15OperationStatsRequest\x12,\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1c.wandb_internal._RequestInfo\"Q\n\x16OperationStatsResponse\x12\x37\n\x0foperation_stats\x18\x01 \x01(\x0b\x32\x1e.wandb_internal.OperationStats\"h\n\x0eOperationStats\x12\r\n\x05label\x18\x03 \x01(\t\x12-\n\noperations\x18\x01 \x03(\x0b\x32\x19.wandb_internal.Operation\x12\x18\n\x10total_operations\x18\x02 \x01(\x03\"\x87\x01\n\tOperation\x12\x0c\n\x04\x64\x65sc\x18\x01 \x01(\t\x12\x17\n\x0fruntime_seconds\x18\x02 \x01(\x01\x12\x10\n\x08progress\x18\x03 \x01(\t\x12\x14\n\x0c\x65rror_status\x18\x04 \x01(\t\x12+\n\x08subtasks\x18\x05 \x03(\x0b\x32\x19.wandb_internal.Operation\"\x13\n\x11SenderMarkRequest\"\x13\n\x11SyncFinishRequest\"E\n\x0cSyncResponse\x12\x0b\n\x03url\x18\x01 \x01(\t\x12(\n\x05\x65rror\x18\x02 \x01(\x0b\x32\x19.wandb_internal.ErrorInfo\"?\n\x11SenderReadRequest\x12\x14\n\x0cstart_offset\x18\x01 \x01(\x03\x12\x14\n\x0c\x66inal_offset\x18\x02 \x01(\x03\"m\n\x13StatusReportRequest\x12\x12\n\nrecord_num\x18\x01 \x01(\x03\x12\x13\n\x0bsent_offset\x18\x02 \x01(\x03\x12-\n\tsync_time\x18\x03 \x01(\x0b\x32\x1a.google.protobuf.Timestamp\"F\n\x14SummaryRecordRequest\x12.\n\x07summary\x18\x01 \x01(\x0b\x32\x1d.wandb_internal.SummaryRecord\"L\n\x16TelemetryRecordRequest\x12\x32\n\ttelemetry\x18\x01 \x01(\x0b\x32\x1f.wandb_internal.TelemetryRecord\"A\n\x11ServerInfoRequest\x12,\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1c.wandb_internal._RequestInfo\"|\n\x12ServerInfoResponse\x12-\n\nlocal_info\x18\x01 \x01(\x0b\x32\x19.wandb_internal.LocalInfo\x12\x37\n\x0fserver_messages\x18\x02 \x01(\x0b\x32\x1e.wandb_internal.ServerMessages\"=\n\x0eServerMessages\x12+\n\x04item\x18\x01 \x03(\x0b\x32\x1d.wandb_internal.ServerMessage\"e\n\rServerMessage\x12\x12\n\nplain_text\x18\x01 \x01(\t\x12\x10\n\x08utf_text\x18\x02 \x01(\t\x12\x11\n\thtml_text\x18\x03 \x01(\t\x12\x0c\n\x04type\x18\x04 \x01(\t\x12\r\n\x05level\x18\x05 \x01(\x05\"c\n\nFileCounts\x12\x13\n\x0bwandb_count\x18\x01 \x01(\x05\x12\x13\n\x0bmedia_count\x18\x02 \x01(\x05\x12\x16\n\x0e\x61rtifact_count\x18\x03 \x01(\x05\x12\x13\n\x0bother_count\x18\x04 \x01(\x05\"U\n\x0f\x46ilePusherStats\x12\x16\n\x0euploaded_bytes\x18\x01 \x01(\x03\x12\x13\n\x0btotal_bytes\x18\x02 \x01(\x03\x12\x15\n\rdeduped_bytes\x18\x03 \x01(\x03\"\x1e\n\rFilesUploaded\x12\r\n\x05\x66iles\x18\x01 \x03(\t\"\xf4\x01\n\x17\x46ileTransferInfoRequest\x12\x42\n\x04type\x18\x01 \x01(\x0e\x32\x34.wandb_internal.FileTransferInfoRequest.TransferType\x12\x0c\n\x04path\x18\x02 \x01(\t\x12\x0b\n\x03url\x18\x03 \x01(\t\x12\x0c\n\x04size\x18\x04 \x01(\x03\x12\x11\n\tprocessed\x18\x05 \x01(\x03\x12/\n\x0b\x66ile_counts\x18\x06 \x01(\x0b\x32\x1a.wandb_internal.FileCounts\"(\n\x0cTransferType\x12\n\n\x06Upload\x10\x00\x12\x0c\n\x08\x44ownload\x10\x01\"1\n\tLocalInfo\x12\x0f\n\x07version\x18\x01 \x01(\t\x12\x13\n\x0bout_of_date\x18\x02 \x01(\x08\"?\n\x0fShutdownRequest\x12,\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1c.wandb_internal._RequestInfo\"\x12\n\x10ShutdownResponse\"P\n\rAttachRequest\x12\x11\n\tattach_id\x18\x14 \x01(\t\x12,\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1c.wandb_internal._RequestInfo\"b\n\x0e\x41ttachResponse\x12&\n\x03run\x18\x01 \x01(\x0b\x32\x19.wandb_internal.RunRecord\x12(\n\x05\x65rror\x18\x02 \x01(\x0b\x32\x19.wandb_internal.ErrorInfo\"\xd5\x02\n\x11TestInjectRequest\x12\x13\n\x0bhandler_exc\x18\x01 \x01(\x08\x12\x14\n\x0chandler_exit\x18\x02 \x01(\x08\x12\x15\n\rhandler_abort\x18\x03 \x01(\x08\x12\x12\n\nsender_exc\x18\x04 \x01(\x08\x12\x13\n\x0bsender_exit\x18\x05 \x01(\x08\x12\x14\n\x0csender_abort\x18\x06 \x01(\x08\x12\x0f\n\x07req_exc\x18\x07 \x01(\x08\x12\x10\n\x08req_exit\x18\x08 \x01(\x08\x12\x11\n\treq_abort\x18\t \x01(\x08\x12\x10\n\x08resp_exc\x18\n \x01(\x08\x12\x11\n\tresp_exit\x18\x0b \x01(\x08\x12\x12\n\nresp_abort\x18\x0c \x01(\x08\x12\x10\n\x08msg_drop\x18\r \x01(\x08\x12\x10\n\x08msg_hang\x18\x0e \x01(\x08\x12,\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1c.wandb_internal._RequestInfo\"\x14\n\x12TestInjectResponse\"\x1e\n\rHistoryAction\x12\r\n\x05\x66lush\x18\x01 \x01(\x08\"\xca\x01\n\x15PartialHistoryRequest\x12)\n\x04item\x18\x01 \x03(\x0b\x32\x1b.wandb_internal.HistoryItem\x12)\n\x04step\x18\x02 \x01(\x0b\x32\x1b.wandb_internal.HistoryStep\x12-\n\x06\x61\x63tion\x18\x03 \x01(\x0b\x32\x1d.wandb_internal.HistoryAction\x12,\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1c.wandb_internal._RequestInfo\"\x18\n\x16PartialHistoryResponse\"E\n\x15SampledHistoryRequest\x12,\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1c.wandb_internal._RequestInfo\"_\n\x12SampledHistoryItem\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\x12\n\nnested_key\x18\x02 \x03(\t\x12\x14\n\x0cvalues_float\x18\x03 \x03(\x02\x12\x12\n\nvalues_int\x18\x04 \x03(\x03\"J\n\x16SampledHistoryResponse\x12\x30\n\x04item\x18\x01 \x03(\x0b\x32\".wandb_internal.SampledHistoryItem\"@\n\x10RunStatusRequest\x12,\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1c.wandb_internal._RequestInfo\"x\n\x11RunStatusResponse\x12\x18\n\x10sync_items_total\x18\x01 \x01(\x03\x12\x1a\n\x12sync_items_pending\x18\x02 \x01(\x03\x12-\n\tsync_time\x18\x03 \x01(\x0b\x32\x1a.google.protobuf.Timestamp\"g\n\x0fRunStartRequest\x12&\n\x03run\x18\x01 \x01(\x0b\x32\x19.wandb_internal.RunRecord\x12,\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1c.wandb_internal._RequestInfo\"\x12\n\x10RunStartResponse\"\\\n\x13\x43heckVersionRequest\x12\x17\n\x0f\x63urrent_version\x18\x01 \x01(\t\x12,\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1c.wandb_internal._RequestInfo\"]\n\x14\x43heckVersionResponse\x12\x17\n\x0fupgrade_message\x18\x01 \x01(\t\x12\x14\n\x0cyank_message\x18\x02 \x01(\t\x12\x16\n\x0e\x64\x65lete_message\x18\x03 \x01(\t\">\n\x0eJobInfoRequest\x12,\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1c.wandb_internal._RequestInfo\"6\n\x0fJobInfoResponse\x12\x12\n\nsequenceId\x18\x01 \x01(\t\x12\x0f\n\x07version\x18\x02 \x01(\t\"\x9f\x01\n\x12LogArtifactRequest\x12\x30\n\x08\x61rtifact\x18\x01 \x01(\x0b\x32\x1e.wandb_internal.ArtifactRecord\x12\x14\n\x0chistory_step\x18\x02 \x01(\x03\x12\x13\n\x0bstaging_dir\x18\x03 \x01(\t\x12,\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1c.wandb_internal._RequestInfo\"A\n\x13LogArtifactResponse\x12\x13\n\x0b\x61rtifact_id\x18\x01 \x01(\t\x12\x15\n\rerror_message\x18\x02 \x01(\t\"\xbe\x01\n\x17\x44ownloadArtifactRequest\x12\x13\n\x0b\x61rtifact_id\x18\x01 \x01(\t\x12\x15\n\rdownload_root\x18\x02 \x01(\t\x12 \n\x18\x61llow_missing_references\x18\x04 \x01(\x08\x12\x12\n\nskip_cache\x18\x05 \x01(\x08\x12\x13\n\x0bpath_prefix\x18\x06 \x01(\t\x12,\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1c.wandb_internal._RequestInfo\"1\n\x18\x44ownloadArtifactResponse\x12\x15\n\rerror_message\x18\x01 \x01(\t\"@\n\x10KeepaliveRequest\x12,\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1c.wandb_internal._RequestInfo\"\x13\n\x11KeepaliveResponse\"q\n\x0c\x41rtifactInfo\x12\x10\n\x08\x61rtifact\x18\x01 \x01(\t\x12\x12\n\nentrypoint\x18\x02 \x03(\t\x12\x10\n\x08notebook\x18\x03 \x01(\x08\x12\x15\n\rbuild_context\x18\x04 \x01(\t\x12\x12\n\ndockerfile\x18\x05 \x01(\t\")\n\x07GitInfo\x12\x0e\n\x06remote\x18\x01 \x01(\t\x12\x0e\n\x06\x63ommit\x18\x02 \x01(\t\"\x87\x01\n\tGitSource\x12)\n\x08git_info\x18\x01 \x01(\x0b\x32\x17.wandb_internal.GitInfo\x12\x12\n\nentrypoint\x18\x02 \x03(\t\x12\x10\n\x08notebook\x18\x03 \x01(\x08\x12\x15\n\rbuild_context\x18\x04 \x01(\t\x12\x12\n\ndockerfile\x18\x05 \x01(\t\"\x1c\n\x0bImageSource\x12\r\n\x05image\x18\x01 \x01(\t\"\x8c\x01\n\x06Source\x12&\n\x03git\x18\x01 \x01(\x0b\x32\x19.wandb_internal.GitSource\x12.\n\x08\x61rtifact\x18\x02 \x01(\x0b\x32\x1c.wandb_internal.ArtifactInfo\x12*\n\x05image\x18\x03 \x01(\x0b\x32\x1b.wandb_internal.ImageSource\"k\n\tJobSource\x12\x10\n\x08_version\x18\x01 \x01(\t\x12\x13\n\x0bsource_type\x18\x02 \x01(\t\x12&\n\x06source\x18\x03 \x01(\x0b\x32\x16.wandb_internal.Source\x12\x0f\n\x07runtime\x18\x04 \x01(\t\"V\n\x12PartialJobArtifact\x12\x10\n\x08job_name\x18\x01 \x01(\t\x12.\n\x0bsource_info\x18\x02 \x01(\x0b\x32\x19.wandb_internal.JobSource\"\x9d\x01\n\x11UseArtifactRecord\x12\n\n\x02id\x18\x01 \x01(\t\x12\x0c\n\x04type\x18\x02 \x01(\t\x12\x0c\n\x04name\x18\x03 \x01(\t\x12\x33\n\x07partial\x18\x04 \x01(\x0b\x32\".wandb_internal.PartialJobArtifact\x12+\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1b.wandb_internal._RecordInfo\"\x13\n\x11UseArtifactResult\"R\n\rCancelRequest\x12\x13\n\x0b\x63\x61ncel_slot\x18\x01 \x01(\t\x12,\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1c.wandb_internal._RequestInfo\"\x10\n\x0e\x43\x61ncelResponse\"\x18\n\x16ProbeSystemInfoRequest\"\'\n\x08\x44iskInfo\x12\r\n\x05total\x18\x01 \x01(\x04\x12\x0c\n\x04used\x18\x02 \x01(\x04\"\x1b\n\nMemoryInfo\x12\r\n\x05total\x18\x01 \x01(\x04\"/\n\x07\x43puInfo\x12\r\n\x05\x63ount\x18\x01 \x01(\r\x12\x15\n\rcount_logical\x18\x02 \x01(\r\"\xad\x01\n\tAppleInfo\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x12\n\necpu_cores\x18\x02 \x01(\r\x12\x12\n\npcpu_cores\x18\x03 \x01(\r\x12\x11\n\tgpu_cores\x18\x04 \x01(\r\x12\x11\n\tmemory_gb\x18\x05 \x01(\r\x12\x18\n\x10swap_total_bytes\x18\x06 \x01(\x04\x12\x17\n\x0fram_total_bytes\x18\x07 \x01(\x04\x12\x11\n\tmac_model\x18\x08 \x01(\t\"k\n\rGpuNvidiaInfo\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x14\n\x0cmemory_total\x18\x02 \x01(\x04\x12\x12\n\ncuda_cores\x18\x03 \x01(\r\x12\x14\n\x0c\x61rchitecture\x18\x04 \x01(\t\x12\x0c\n\x04uuid\x18\x05 \x01(\t\"\x89\x02\n\nGpuAmdInfo\x12\n\n\x02id\x18\x01 \x01(\t\x12\x11\n\tunique_id\x18\x02 \x01(\t\x12\x15\n\rvbios_version\x18\x03 \x01(\t\x12\x19\n\x11performance_level\x18\x04 \x01(\t\x12\x15\n\rgpu_overdrive\x18\x05 \x01(\t\x12\x1c\n\x14gpu_memory_overdrive\x18\x06 \x01(\t\x12\x11\n\tmax_power\x18\x07 \x01(\t\x12\x0e\n\x06series\x18\x08 \x01(\t\x12\r\n\x05model\x18\t \x01(\t\x12\x0e\n\x06vendor\x18\n \x01(\t\x12\x0b\n\x03sku\x18\x0b \x01(\t\x12\x12\n\nsclk_range\x18\x0c \x01(\t\x12\x12\n\nmclk_range\x18\r \x01(\t\"n\n\x0cTrainiumInfo\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x0e\n\x06vendor\x18\x02 \x01(\t\x12\x1b\n\x13neuron_device_count\x18\x03 \x01(\r\x12#\n\x1bneuroncore_per_device_count\x18\x04 \x01(\r\"Q\n\x07TPUInfo\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x0f\n\x07hbm_gib\x18\x02 \x01(\r\x12\x18\n\x10\x64\x65vices_per_chip\x18\x03 \x01(\r\x12\r\n\x05\x63ount\x18\x04 \x01(\r\"E\n\rCoreWeaveInfo\x12\x14\n\x0c\x63luster_name\x18\x01 \x01(\t\x12\x0e\n\x06org_id\x18\x02 \x01(\t\x12\x0e\n\x06region\x18\x03 \x01(\t\"\xa8\t\n\x11\x45nvironmentRecord\x12\n\n\x02os\x18\x01 \x01(\t\x12\x0e\n\x06python\x18\x02 \x01(\t\x12\x39\n\nstarted_at\x18\x03 \x01(\x0b\x32\x1a.google.protobuf.TimestampR\tstartedAt\x12\x0e\n\x06\x64ocker\x18\x04 \x01(\t\x12\x0c\n\x04\x61rgs\x18\x05 \x03(\t\x12\x0f\n\x07program\x18\x06 \x01(\t\x12\x1b\n\tcode_path\x18\x07 \x01(\tR\x08\x63odePath\x12&\n\x0f\x63ode_path_local\x18\x08 \x01(\tR\rcodePathLocal\x12*\n\x03git\x18\t \x01(\x0b\x32\x1d.wandb_internal.GitRepoRecord\x12\r\n\x05\x65mail\x18\n \x01(\t\x12\x0c\n\x04root\x18\x0b \x01(\t\x12\x0c\n\x04host\x18\x0c \x01(\t\x12\x10\n\x08username\x18\r \x01(\t\x12\x12\n\nexecutable\x18\x0e \x01(\t\x12\r\n\x05\x63olab\x18\x0f \x01(\t\x12\x1c\n\tcpu_count\x18\x10 \x01(\rR\tcpu_count\x12,\n\x11\x63pu_count_logical\x18\x11 \x01(\rR\x11\x63pu_count_logical\x12\x15\n\x08gpu_type\x18\x12 \x01(\tR\x03gpu\x12\x1c\n\tgpu_count\x18\x13 \x01(\rR\tgpu_count\x12\x39\n\x04\x64isk\x18\x14 \x03(\x0b\x32+.wandb_internal.EnvironmentRecord.DiskEntry\x12*\n\x06memory\x18\x15 \x01(\x0b\x32\x1a.wandb_internal.MemoryInfo\x12$\n\x03\x63pu\x18\x16 \x01(\x0b\x32\x17.wandb_internal.CpuInfo\x12(\n\x05\x61pple\x18\x17 \x01(\x0b\x32\x19.wandb_internal.AppleInfo\x12=\n\ngpu_nvidia\x18\x18 \x03(\x0b\x32\x1d.wandb_internal.GpuNvidiaInfoR\ngpu_nvidia\x12\x14\n\x0c\x63uda_version\x18\x19 \x01(\t\x12\x34\n\x07gpu_amd\x18\x1a \x03(\x0b\x32\x1a.wandb_internal.GpuAmdInfoR\x07gpu_amd\x12;\n\x05slurm\x18\x1b \x03(\x0b\x32,.wandb_internal.EnvironmentRecord.SlurmEntry\x12.\n\x08trainium\x18\x1c \x01(\x0b\x32\x1c.wandb_internal.TrainiumInfo\x12$\n\x03tpu\x18\x1d \x01(\x0b\x32\x17.wandb_internal.TPUInfo\x12\x30\n\tcoreweave\x18\x1e \x01(\x0b\x32\x1d.wandb_internal.CoreWeaveInfo\x12\x12\n\twriter_id\x18\xc7\x01 \x01(\t\x12+\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1b.wandb_internal._RecordInfo\x1a\x45\n\tDiskEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\'\n\x05value\x18\x02 \x01(\x0b\x32\x18.wandb_internal.DiskInfo:\x02\x38\x01\x1a,\n\nSlurmEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t:\x02\x38\x01\"\x8d\x01\n\x15PythonPackagesRequest\x12\x44\n\x07package\x18\x01 \x03(\x0b\x32\x33.wandb_internal.PythonPackagesRequest.PythonPackage\x1a.\n\rPythonPackage\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x0f\n\x07version\x18\x02 \x01(\t\"\x1c\n\x0cJobInputPath\x12\x0c\n\x04path\x18\x01 \x03(\t\"\xd6\x01\n\x0eJobInputSource\x12\x44\n\nrun_config\x18\x01 \x01(\x0b\x32..wandb_internal.JobInputSource.RunConfigSourceH\x00\x12?\n\x04\x66ile\x18\x02 \x01(\x0b\x32/.wandb_internal.JobInputSource.ConfigFileSourceH\x00\x1a\x11\n\x0fRunConfigSource\x1a \n\x10\x43onfigFileSource\x12\x0c\n\x04path\x18\x01 \x01(\tB\x08\n\x06source\"\xc7\x01\n\x0fJobInputRequest\x12\x34\n\x0cinput_source\x18\x01 \x01(\x0b\x32\x1e.wandb_internal.JobInputSource\x12\x33\n\rinclude_paths\x18\x02 \x03(\x0b\x32\x1c.wandb_internal.JobInputPath\x12\x33\n\rexclude_paths\x18\x03 \x03(\x0b\x32\x1c.wandb_internal.JobInputPath\x12\x14\n\x0cinput_schema\x18\x04 \x01(\t*\xda\x05\n\rServerFeature\x12\x1e\n\x1aSERVER_FEATURE_UNSPECIFIED\x10\x00\x12\x13\n\x0fLARGE_FILENAMES\x10\x11\x12\x11\n\rARTIFACT_TAGS\x10\x01\x12\x0e\n\nCLIENT_IDS\x10\x02\x12\x1c\n\x18\x41RTIFACT_REGISTRY_SEARCH\x10\x03\x12\x1b\n\x17STRUCTURED_CONSOLE_LOGS\x10\x04\x12(\n$ARTIFACT_COLLECTION_MEMBERSHIP_FILES\x10\x05\x12\x38\n4ARTIFACT_COLLECTION_MEMBERSHIP_FILE_DOWNLOAD_HANDLER\x10\x06\x12\x34\n0USE_ARTIFACT_WITH_ENTITY_AND_PROJECT_INFORMATION\x10\x07\x12\x1f\n\x1b\x45XPAND_DEFINED_METRIC_GLOBS\x10\x08\x12\x1f\n\x1b\x41UTOMATION_EVENT_RUN_METRIC\x10\t\x12&\n\"AUTOMATION_EVENT_RUN_METRIC_CHANGE\x10\n\x12\x1b\n\x17\x41UTOMATION_ACTION_NO_OP\x10\x0b\x12/\n+INCLUDE_ARTIFACT_TYPES_IN_REGISTRY_CREATION\x10\x0c\x12*\n&PROJECT_ARTIFACT_COLLECTION_MEMBERSHIP\x10\r\x12\x31\n-ARTIFACT_MEMBERSHIP_IN_LINK_ARTIFACT_RESPONSE\x10\x0e\x12\"\n\x1eTOTAL_COUNT_IN_FILE_CONNECTION\x10\x0f\x12*\n&ARTIFACT_COLLECTIONS_FILTERING_SORTING\x10\x10\x12\x35\n1ARTIFACT_V2_DOWNLOAD_HANDLER_SUPPORTS_ARTIFACT_ID\x10\x12\x42\x1bZ\x19\x63ore/pkg/service_go_protob\x06proto3')
/n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/wandb/proto/v7/wandb_api_pb2.py:29:DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x1bwandb/proto/wandb_api.proto\x12\x0ewandb_internal\x1a wandb/proto/wandb_internal.proto\x1a wandb/proto/wandb_settings.proto\"B\n\x14ServerApiInitRequest\x12*\n\x08settings\x18\x01 \x01(\x0b\x32\x18.wandb_internal.Settings\">\n\x15ServerApiInitResponse\x12\x15\n\rerror_message\x18\x01 \x01(\t\x12\x0e\n\x06\x61pi_id\x18\x02 \x01(\t\"\xaf\x01\n\nApiRequest\x12\x0e\n\x06\x61pi_id\x18\x01 \x01(\t\x12I\n\x18read_run_history_request\x18\x02 \x01(\x0b\x32%.wandb_internal.ReadRunHistoryRequestH\x00\x12;\n\x10\x66\x65\x61tures_request\x18\x03 \x01(\x0b\x32\x1f.wandb_internal.FeaturesRequestH\x00\x42\t\n\x07request\"\xe5\x01\n\x0b\x41piResponse\x12K\n\x19read_run_history_response\x18\x01 \x01(\x0b\x32&.wandb_internal.ReadRunHistoryResponseH\x00\x12=\n\x11\x66\x65\x61tures_response\x18\x03 \x01(\x0b\x32 .wandb_internal.FeaturesResponseH\x00\x12>\n\x12\x61pi_error_response\x18\x02 \x01(\x0b\x32 .wandb_internal.ApiErrorResponseH\x00\x42\n\n\x08response\"f\n\x10\x41piErrorResponse\x12\x0f\n\x07message\x18\x01 \x01(\t\x12\x32\n\nerror_type\x18\x02 \x01(\x0e\x32\x19.wandb_internal.ErrorTypeH\x00\x88\x01\x01\x42\r\n\x0b_error_type\")\n\x17ServerApiCleanupRequest\x12\x0e\n\x06\x61pi_id\x18\x01 \x01(\t\"B\n\x0f\x46\x65\x61turesRequest\x12/\n\x08\x66\x65\x61tures\x18\x01 \x03(\x0e\x32\x1d.wandb_internal.ServerFeature\"B\n\x10\x46\x65\x61turesResponse\x12.\n\x07\x65nabled\x18\x01 \x03(\x0e\x32\x1d.wandb_internal.ServerFeature\"\xd0\x03\n\x15ReadRunHistoryRequest\x12\x43\n\x15scan_run_history_init\x18\x01 \x01(\x0b\x32\".wandb_internal.ScanRunHistoryInitH\x00\x12:\n\x10scan_run_history\x18\x02 \x01(\x0b\x32\x1e.wandb_internal.ScanRunHistoryH\x00\x12I\n\x18scan_run_history_cleanup\x18\x03 \x01(\x0b\x32%.wandb_internal.ScanRunHistoryCleanupH\x00\x12K\n\x19\x64ownload_run_history_init\x18\x04 \x01(\x0b\x32&.wandb_internal.DownloadRunHistoryInitH\x00\x12\x42\n\x14\x64ownload_run_history\x18\x05 \x01(\x0b\x32\".wandb_internal.DownloadRunHistoryH\x00\x12O\n\x1b\x64ownload_run_history_status\x18\x06 \x01(\x0b\x32(.wandb_internal.DownloadRunHistoryStatusH\x00\x42\t\n\x07request\"\xf9\x03\n\x16ReadRunHistoryResponse\x12K\n\x15scan_run_history_init\x18\x01 \x01(\x0b\x32*.wandb_internal.ScanRunHistoryInitResponseH\x00\x12\x39\n\x0brun_history\x18\x02 \x01(\x0b\x32\".wandb_internal.RunHistoryResponseH\x00\x12Q\n\x18scan_run_history_cleanup\x18\x03 \x01(\x0b\x32-.wandb_internal.ScanRunHistoryCleanupResponseH\x00\x12S\n\x19\x64ownload_run_history_init\x18\x04 \x01(\x0b\x32..wandb_internal.DownloadRunHistoryInitResponseH\x00\x12J\n\x14\x64ownload_run_history\x18\x05 \x01(\x0b\x32*.wandb_internal.DownloadRunHistoryResponseH\x00\x12W\n\x1b\x64ownload_run_history_status\x18\x06 \x01(\x0b\x32\x30.wandb_internal.DownloadRunHistoryStatusResponseH\x00\x42\n\n\x08response\"f\n\x12ScanRunHistoryInit\x12\x0e\n\x06\x65ntity\x18\x01 \x01(\t\x12\x0f\n\x07project\x18\x02 \x01(\t\x12\x0e\n\x06run_id\x18\x03 \x01(\t\x12\x0c\n\x04keys\x18\x04 \x03(\t\x12\x11\n\tuse_cache\x18\x05 \x01(\x08\"0\n\x1aScanRunHistoryInitResponse\x12\x12\n\nrequest_id\x18\x01 \x01(\x05\"H\n\x0eScanRunHistory\x12\x10\n\x08min_step\x18\x01 \x01(\x03\x12\x10\n\x08max_step\x18\x02 \x01(\x03\x12\x12\n\nrequest_id\x18\x03 \x01(\x05\"F\n\x12RunHistoryResponse\x12\x30\n\x0chistory_rows\x18\x01 \x03(\x0b\x32\x1a.wandb_internal.HistoryRow\"G\n\nHistoryRow\x12\x39\n\rhistory_items\x18\x01 \x03(\x0b\x32\".wandb_internal.ParquetHistoryItem\"5\n\x12ParquetHistoryItem\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\x12\n\nvalue_json\x18\x10 \x01(\t\"+\n\x15ScanRunHistoryCleanup\x12\x12\n\nrequest_id\x18\x01 \x01(\x05\"\x1f\n\x1dScanRunHistoryCleanupResponse\"\x81\x01\n\x16\x44ownloadRunHistoryInit\x12\x0e\n\x06\x65ntity\x18\x01 \x01(\t\x12\x0f\n\x07project\x18\x02 \x01(\t\x12\x0e\n\x06run_id\x18\x03 \x01(\t\x12\x14\n\x0c\x64ownload_dir\x18\x04 \x01(\t\x12 \n\x18require_complete_history\x18\x05 \x01(\x08\"P\n\x1e\x44ownloadRunHistoryInitResponse\x12\x12\n\nrequest_id\x18\x01 \x01(\x05\x12\x1a\n\x12\x63ontains_live_data\x18\x02 \x01(\x08\"(\n\x12\x44ownloadRunHistory\x12\x12\n\nrequest_id\x18\x01 \x01(\x05\"\xad\x01\n\x1a\x44ownloadRunHistoryResponse\x12\x18\n\x10\x64ownloaded_files\x18\x01 \x03(\t\x12\x46\n\x06\x65rrors\x18\x02 \x03(\x0b\x32\x36.wandb_internal.DownloadRunHistoryResponse.ErrorsEntry\x1a-\n\x0b\x45rrorsEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t:\x02\x38\x01\"\x1b\n\x19IncompleteRunHistoryError\".\n\x18\x44ownloadRunHistoryStatus\x12\x12\n\nrequest_id\x18\x01 \x01(\x05\"[\n DownloadRunHistoryStatusResponse\x12\x37\n\x0foperation_stats\x18\x01 \x01(\x0b\x32\x1e.wandb_internal.OperationStats*@\n\tErrorType\x12\x11\n\rUNKNOWN_ERROR\x10\x00\x12 \n\x1cINCOMPLETE_RUN_HISTORY_ERROR\x10\x01\x42\x1bZ\x19\x63ore/pkg/service_go_protob\x06proto3')
/n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/wandb/proto/v7/wandb_settings_pb2.py:28:DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n wandb/proto/wandb_settings.proto\x12\x0ewandb_internal\x1a\x1egoogle/protobuf/wrappers.proto\" \n\x0fListStringValue\x12\r\n\x05value\x18\x01 \x03(\t\"\x1d\n\x0cListIntValue\x12\r\n\x05value\x18\x01 \x03(\x05\"\x8a\x01\n\x17MapStringKeyStringValue\x12\x41\n\x05value\x18\x01 \x03(\x0b\x32\x32.wandb_internal.MapStringKeyStringValue.ValueEntry\x1a,\n\nValueEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t:\x02\x38\x01\"\xcb\x01\n#MapStringKeyMapStringKeyStringValue\x12M\n\x05value\x18\x01 \x03(\x0b\x32>.wandb_internal.MapStringKeyMapStringKeyStringValue.ValueEntry\x1aU\n\nValueEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\x36\n\x05value\x18\x02 \x01(\x0b\x32\'.wandb_internal.MapStringKeyStringValue:\x02\x38\x01\"\x9a\x01\n\x12OpenMetricsFilters\x12\x33\n\x08sequence\x18\x01 \x01(\x0b\x32\x1f.wandb_internal.ListStringValueH\x00\x12\x46\n\x07mapping\x18\x02 \x01(\x0b\x32\x33.wandb_internal.MapStringKeyMapStringKeyStringValueH\x00\x42\x07\n\x05value\"7\n\tRunMoment\x12\x0b\n\x03run\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\x01\x12\x0e\n\x06metric\x18\x03 \x01(\t\"\xbeO\n\x08Settings\x12-\n\x07\x61pi_key\x18\x37 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12:\n\x13identity_token_file\x18\xaa\x01 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12\x37\n\x10\x63redentials_file\x18\xab\x01 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12\x39\n\x14insecure_disable_ssl\x18\xb9\x01 \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12,\n\x08_offline\x18\x1e \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12*\n\x06x_sync\x18\x1f \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12\x30\n\tsync_file\x18\x86\x01 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12,\n\x07_shared\x18\xa2\x01 \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12,\n\x06run_id\x18k \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12-\n\x07run_url\x18q \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12-\n\x07project\x18\x61 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12,\n\x06\x65ntity\x18\x45 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12\x33\n\x0corganization\x18\xbc\x01 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12\x32\n\x0cx_start_time\x18) \x01(\x0b\x32\x1c.google.protobuf.DoubleValue\x12.\n\x08root_dir\x18i \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12\x30\n\twandb_dir\x18\x8e\x01 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12-\n\x07log_dir\x18U \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12\x32\n\x0clog_internal\x18V \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12\x35\n\x0cignore_globs\x18N \x01(\x0b\x32\x1f.wandb_internal.ListStringValue\x12.\n\x07\x61pp_url\x18\xca\x01 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12.\n\x08\x62\x61se_url\x18\x39 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12=\n\x17x_file_stream_max_bytes\x18\xac\x01 \x01(\x0b\x32\x1b.google.protobuf.Int32Value\x12\x46\n\x1fx_file_stream_transmit_interval\x18\xaf\x01 \x01(\x0b\x32\x1c.google.protobuf.DoubleValue\x12\x45\n\x14x_extra_http_headers\x18\x0e \x01(\x0b\x32\'.wandb_internal.MapStringKeyStringValue\x12=\n\x17x_file_stream_retry_max\x18\x93\x01 \x01(\x0b\x32\x1b.google.protobuf.Int32Value\x12K\n$x_file_stream_retry_wait_min_seconds\x18\x94\x01 \x01(\x0b\x32\x1c.google.protobuf.DoubleValue\x12K\n$x_file_stream_retry_wait_max_seconds\x18\x95\x01 \x01(\x0b\x32\x1c.google.protobuf.DoubleValue\x12\x43\n\x1dx_file_stream_timeout_seconds\x18\x0f \x01(\x0b\x32\x1c.google.protobuf.DoubleValue\x12\x42\n\x1cx_file_stream_max_line_bytes\x18\xb2\x01 \x01(\x0b\x32\x1b.google.protobuf.Int32Value\x12?\n\x19x_file_transfer_retry_max\x18\x96\x01 \x01(\x0b\x32\x1b.google.protobuf.Int32Value\x12M\n&x_file_transfer_retry_wait_min_seconds\x18\x97\x01 \x01(\x0b\x32\x1c.google.protobuf.DoubleValue\x12M\n&x_file_transfer_retry_wait_max_seconds\x18\x98\x01 \x01(\x0b\x32\x1c.google.protobuf.DoubleValue\x12\x46\n\x1fx_file_transfer_timeout_seconds\x18\x99\x01 \x01(\x0b\x32\x1c.google.protobuf.DoubleValue\x12\x39\n\x13x_graphql_retry_max\x18\x9a\x01 \x01(\x0b\x32\x1b.google.protobuf.Int32Value\x12G\n x_graphql_retry_wait_min_seconds\x18\x9b\x01 \x01(\x0b\x32\x1c.google.protobuf.DoubleValue\x12G\n x_graphql_retry_wait_max_seconds\x18\x9c\x01 \x01(\x0b\x32\x1c.google.protobuf.DoubleValue\x12@\n\x19x_graphql_timeout_seconds\x18\x9d\x01 \x01(\x0b\x32\x1c.google.protobuf.DoubleValue\x12\x31\n\nhttp_proxy\x18\xa8\x01 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12\x32\n\x0bhttps_proxy\x18\xa9\x01 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12;\n\tx_proxies\x18\xc8\x01 \x01(\x0b\x32\'.wandb_internal.MapStringKeyStringValue\x12-\n\x07program\x18_ \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12\x35\n\x0fprogram_relpath\x18` \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12\x37\n\x10_code_path_local\x18\xa3\x01 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12\x36\n\x0fprogram_abspath\x18\x9f\x01 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12.\n\x05_args\x18\x01 \x01(\x0b\x32\x1f.wandb_internal.ListStringValue\x12)\n\x03_os\x18  \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12,\n\x06\x64ocker\x18\x43 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12\x32\n\x0cx_executable\x18\r \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12-\n\x07_python\x18\" \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12\x30\n\tcolab_url\x18\xa0\x01 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12*\n\x04host\x18M \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12/\n\x08username\x18\x8d\x01 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12+\n\x05\x65mail\x18\x44 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12,\n\x06resume\x18\x66 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12/\n\x0bresume_from\x18\xa7\x01 \x01(\x0b\x32\x19.wandb_internal.RunMoment\x12-\n\tfork_from\x18\xa4\x01 \x01(\x0b\x32\x19.wandb_internal.RunMoment\x12\x38\n\x14\x64isable_job_creation\x18\x41 \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12\x30\n\tsweep_url\x18\x83\x01 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12;\n\x16x_disable_update_check\x18\xa5\x01 \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12\x32\n\x0ex_disable_meta\x18\x07 \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12-\n\tsave_code\x18s \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12/\n\x0b\x64isable_git\x18? \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12;\n\x16\x64isable_git_fork_point\x18\xcb\x01 \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12;\n\x16x_disable_machine_info\x18\x9e\x01 \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12\x33\n\x0fx_disable_stats\x18\n \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12\x39\n\x13x_stats_buffer_size\x18\xa1\x01 \x01(\x0b\x32\x1b.google.protobuf.Int32Value\x12@\n\x19x_stats_sampling_interval\x18\xae\x01 \x01(\x0b\x32\x1c.google.protobuf.DoubleValue\x12\x30\n\x0bx_stats_pid\x18* \x01(\x0b\x32\x1b.google.protobuf.Int32Value\x12<\n\x12x_stats_disk_paths\x18\x92\x01 \x01(\x0b\x32\x1f.wandb_internal.ListStringValue\x12H\n\"x_stats_neuron_monitor_config_path\x18. \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12<\n\x15x_stats_dcgm_exporter\x18\xbb\x01 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12O\n\x1ex_stats_open_metrics_endpoints\x18/ \x01(\x0b\x32\'.wandb_internal.MapStringKeyStringValue\x12H\n\x1cx_stats_open_metrics_filters\x18\x30 \x01(\x0b\x32\".wandb_internal.OpenMetricsFilters\x12S\n!x_stats_open_metrics_http_headers\x18\xb8\x01 \x01(\x0b\x32\'.wandb_internal.MapStringKeyStringValue\x12=\n\x16x_stats_gpu_device_ids\x18\xba\x01 \x01(\x0b\x32\x1c.wandb_internal.ListIntValue\x12\x37\n\x11x_stats_cpu_count\x18\xc2\x01 \x01(\x0b\x32\x1b.google.protobuf.Int32Value\x12?\n\x19x_stats_cpu_logical_count\x18\xc3\x01 \x01(\x0b\x32\x1b.google.protobuf.Int32Value\x12\x37\n\x11x_stats_gpu_count\x18\xc4\x01 \x01(\x0b\x32\x1b.google.protobuf.Int32Value\x12\x37\n\x10x_stats_gpu_type\x18\xc5\x01 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12?\n\x1ax_stats_track_process_tree\x18\xc6\x01 \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12.\n\x07x_label\x18\xb5\x01 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12.\n\tx_primary\x18\xb6\x01 \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12:\n\x15x_update_finish_state\x18\xb7\x01 \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12<\n\x17\x61llow_offline_artifacts\x18\xb1\x01 \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12-\n\x07\x63onsole\x18< \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12\x36\n\x11\x63onsole_multipart\x18\xa6\x01 \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12=\n\x17\x63onsole_chunk_max_bytes\x18\xc7\x01 \x01(\x0b\x32\x1b.google.protobuf.Int32Value\x12?\n\x19\x63onsole_chunk_max_seconds\x18\xc9\x01 \x01(\x0b\x32\x1b.google.protobuf.Int32Value\x12\x35\n\x10sync_tensorboard\x18\xb3\x01 \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12\x42\n\x1dx_server_side_derived_summary\x18\xbd\x01 \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12\x46\n!x_server_side_expand_glob_metrics\x18\xbe\x01 \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12;\n\x16x_skip_transaction_log\x18\xbf\x01 \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12J\n#x_stats_coreweave_metadata_base_url\x18\xc0\x01 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12J\n#x_stats_coreweave_metadata_endpoint\x18\xc1\x01 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12/\n\x0b_aws_lambda\x18\x02 \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12\x33\n\x0fx_cli_only_mode\x18\x04 \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12*\n\x06_colab\x18\x05 \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12\x34\n\x10x_disable_viewer\x18\x0b \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12\x39\n\x15x_flow_control_custom\x18\x10 \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12;\n\x17x_flow_control_disabled\x18\x11 \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12>\n\x18x_internal_check_process\x18\x12 \x01(\x0b\x32\x1c.google.protobuf.DoubleValue\x12,\n\x08_ipython\x18\x14 \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12,\n\x08_jupyter\x18\x15 \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12\x34\n\x0ex_jupyter_root\x18\x16 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12+\n\x07_kaggle\x18\x17 \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12=\n\x18x_live_policy_rate_limit\x18\x18 \x01(\x0b\x32\x1b.google.protobuf.Int32Value\x12<\n\x17x_live_policy_wait_time\x18\x19 \x01(\x0b\x32\x1b.google.protobuf.Int32Value\x12\x30\n\x0bx_log_level\x18\x1a \x01(\x0b\x32\x1b.google.protobuf.Int32Value\x12\x35\n\x10x_network_buffer\x18\x1b \x01(\x0b\x32\x1b.google.protobuf.Int32Value\x12)\n\x05_noop\x18\x1c \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12-\n\t_notebook\x18\x1d \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12/\n\t_platform\x18! \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12\x38\n\x12x_runqueue_item_id\x18# \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12\x37\n\x13x_save_requirements\x18% \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12\x39\n\x13x_service_transport\x18& \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12\x34\n\x0ex_service_wait\x18\' \x01(\x0b\x32\x1c.google.protobuf.DoubleValue\x12\x35\n\x0f_start_datetime\x18( \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12\x33\n\r_tmp_code_dir\x18\x31 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12,\n\x08_windows\x18\x34 \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12\x38\n\x13\x61llow_media_symlink\x18\xcc\x01 \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12\x34\n\x10\x61llow_val_change\x18\x35 \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12P\n\x1f\x61zure_account_url_to_access_key\x18\x38 \x01(\x0b\x32\'.wandb_internal.MapStringKeyStringValue\x12.\n\x08\x63ode_dir\x18: \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12\x35\n\x0c\x63onfig_paths\x18; \x01(\x0b\x32\x1f.wandb_internal.ListStringValue\x12\x30\n\ndeployment\x18= \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12\x30\n\x0c\x64isable_code\x18> \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12\x31\n\rdisable_hints\x18@ \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12,\n\x08\x64isabled\x18\x42 \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12)\n\x05\x66orce\x18G \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12\x30\n\ngit_commit\x18H \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12\x30\n\ngit_remote\x18I \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12\x34\n\x0egit_remote_url\x18J \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12.\n\x08git_root\x18K \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12\x36\n\x11heartbeat_seconds\x18L \x01(\x0b\x32\x1b.google.protobuf.Int32Value\x12\x32\n\x0cinit_timeout\x18O \x01(\x0b\x32\x1c.google.protobuf.DoubleValue\x12,\n\x08is_local\x18P \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12\x30\n\njob_source\x18Q \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12\x31\n\rlabel_disable\x18R \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12*\n\x06launch\x18S \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12\x38\n\x12launch_config_path\x18T \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12:\n\x14log_symlink_internal\x18W \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12\x36\n\x10log_symlink_user\x18X \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12.\n\x08log_user\x18Y \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12\x33\n\rlogin_timeout\x18Z \x01(\x0b\x32\x1c.google.protobuf.DoubleValue\x12*\n\x04mode\x18\\ \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12\x33\n\rnotebook_name\x18] \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12\x31\n\x0bproject_url\x18\x62 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12)\n\x05quiet\x18\x63 \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12+\n\x07relogin\x18\x65 \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12\x32\n\x0cresume_fname\x18g \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12+\n\x07resumed\x18h \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12/\n\trun_group\x18j \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12\x32\n\x0crun_job_type\x18l \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12.\n\x08run_mode\x18m \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12.\n\x08run_name\x18n \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12/\n\trun_notes\x18o \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12\x31\n\x08run_tags\x18p \x01(\x0b\x32\x1f.wandb_internal.ListStringValue\x12\x35\n\x11sagemaker_disable\x18r \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12\x35\n\x0fsettings_system\x18t \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12\x38\n\x12settings_workspace\x18u \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12/\n\x0bshow_colors\x18v \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12.\n\nshow_emoji\x18w \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12/\n\x0bshow_errors\x18x \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12-\n\tshow_info\x18y \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12\x31\n\rshow_warnings\x18z \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12*\n\x06silent\x18{ \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12\x32\n\x0cstart_method\x18| \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12*\n\x06strict\x18} \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12\x33\n\x0esummary_errors\x18~ \x01(\x0b\x32\x1b.google.protobuf.Int32Value\x12\x34\n\x0fsummary_timeout\x18\x7f \x01(\x0b\x32\x1b.google.protobuf.Int32Value\x12\x36\n\x10summary_warnings\x18\x80\x01 \x01(\x0b\x32\x1b.google.protobuf.Int32Value\x12/\n\x08sweep_id\x18\x81\x01 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12\x37\n\x10sweep_param_path\x18\x82\x01 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12,\n\x07symlink\x18\x84\x01 \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12/\n\x08sync_dir\x18\x85\x01 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12:\n\x13sync_symlink_latest\x18\x87\x01 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12J\n%table_raise_on_max_row_limit_exceeded\x18\x8a\x01 \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12/\n\x08timespec\x18\x8b\x01 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12.\n\x07tmp_dir\x18\x8c\x01 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12\x35\n\x0ex_jupyter_name\x18\x8f\x01 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12\x35\n\x0ex_jupyter_path\x18\x90\x01 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12/\n\x08job_name\x18\x91\x01 \x01(\x0b\x32\x1c.google.protobuf.StringValueJ\x04\x08\x03\x10\x04J\x04\x08\x06\x10\x07J\x04\x08\x08\x10\tJ\x04\x08\t\x10\nJ\x04\x08\x0c\x10\rJ\x04\x08\x13\x10\x14J\x04\x08$\x10%J\x04\x08+\x10,J\x04\x08,\x10-J\x04\x08-\x10.J\x04\x08\x32\x10\x33J\x04\x08\x33\x10\x34J\x04\x08\x36\x10\x37J\x04\x08\x46\x10GJ\x04\x08[\x10\\J\x04\x08^\x10_J\x04\x08\x64\x10\x65J\x06\x08\x88\x01\x10\x89\x01J\x06\x08\x89\x01\x10\x8a\x01J\x06\x08\xad\x01\x10\xae\x01J\x06\x08\xb0\x01\x10\xb1\x01J\x06\x08\xb4\x01\x10\xb5\x01\x42\x1bZ\x19\x63ore/pkg/service_go_protob\x06proto3')
/n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/wandb/proto/v7/wandb_internal_pb2.py:31:DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n wandb/proto/wandb_internal.proto\x12\x0ewandb_internal\x1a\x1bgoogle/protobuf/empty.proto\x1a\x1fgoogle/protobuf/timestamp.proto\x1a\x1cwandb/proto/wandb_base.proto\x1a!wandb/proto/wandb_telemetry.proto\"\xcf\t\n\x06Record\x12\x0b\n\x03num\x18\x01 \x01(\x03\x12\x30\n\x07history\x18\x02 \x01(\x0b\x32\x1d.wandb_internal.HistoryRecordH\x00\x12\x30\n\x07summary\x18\x03 \x01(\x0b\x32\x1d.wandb_internal.SummaryRecordH\x00\x12.\n\x06output\x18\x04 \x01(\x0b\x32\x1c.wandb_internal.OutputRecordH\x00\x12.\n\x06\x63onfig\x18\x05 \x01(\x0b\x32\x1c.wandb_internal.ConfigRecordH\x00\x12,\n\x05\x66iles\x18\x06 \x01(\x0b\x32\x1b.wandb_internal.FilesRecordH\x00\x12,\n\x05stats\x18\x07 \x01(\x0b\x32\x1b.wandb_internal.StatsRecordH\x00\x12\x32\n\x08\x61rtifact\x18\x08 \x01(\x0b\x32\x1e.wandb_internal.ArtifactRecordH\x00\x12,\n\x08tbrecord\x18\t \x01(\x0b\x32\x18.wandb_internal.TBRecordH\x00\x12,\n\x05\x61lert\x18\n \x01(\x0b\x32\x1b.wandb_internal.AlertRecordH\x00\x12\x34\n\ttelemetry\x18\x0b \x01(\x0b\x32\x1f.wandb_internal.TelemetryRecordH\x00\x12.\n\x06metric\x18\x0c \x01(\x0b\x32\x1c.wandb_internal.MetricRecordH\x00\x12\x35\n\noutput_raw\x18\r \x01(\x0b\x32\x1f.wandb_internal.OutputRawRecordH\x00\x12(\n\x03run\x18\x11 \x01(\x0b\x32\x19.wandb_internal.RunRecordH\x00\x12-\n\x04\x65xit\x18\x12 \x01(\x0b\x32\x1d.wandb_internal.RunExitRecordH\x00\x12,\n\x05\x66inal\x18\x14 \x01(\x0b\x32\x1b.wandb_internal.FinalRecordH\x00\x12.\n\x06header\x18\x15 \x01(\x0b\x32\x1c.wandb_internal.HeaderRecordH\x00\x12.\n\x06\x66ooter\x18\x16 \x01(\x0b\x32\x1c.wandb_internal.FooterRecordH\x00\x12\x39\n\npreempting\x18\x17 \x01(\x0b\x32#.wandb_internal.RunPreemptingRecordH\x00\x12\x34\n\x12noop_link_artifact\x18\x18 \x01(\x0b\x32\x16.google.protobuf.EmptyH\x00\x12\x39\n\x0cuse_artifact\x18\x19 \x01(\x0b\x32!.wandb_internal.UseArtifactRecordH\x00\x12\x38\n\x0b\x65nvironment\x18\x1a \x01(\x0b\x32!.wandb_internal.EnvironmentRecordH\x00\x12*\n\x07request\x18\x64 \x01(\x0b\x32\x17.wandb_internal.RequestH\x00\x12(\n\x07\x63ontrol\x18\x10 \x01(\x0b\x32\x17.wandb_internal.Control\x12\x0c\n\x04uuid\x18\x13 \x01(\t\x12+\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1b.wandb_internal._RecordInfoB\r\n\x0brecord_type\"\xa8\x01\n\x07\x43ontrol\x12\x10\n\x08req_resp\x18\x01 \x01(\x08\x12\r\n\x05local\x18\x02 \x01(\x08\x12\x10\n\x08relay_id\x18\x03 \x01(\t\x12\x14\n\x0cmailbox_slot\x18\x04 \x01(\t\x12\x13\n\x0b\x61lways_send\x18\x05 \x01(\x08\x12\x14\n\x0c\x66low_control\x18\x06 \x01(\x08\x12\x12\n\nend_offset\x18\x07 \x01(\x03\x12\x15\n\rconnection_id\x18\x08 \x01(\t\"\xf3\x03\n\x06Result\x12\x35\n\nrun_result\x18\x11 \x01(\x0b\x32\x1f.wandb_internal.RunUpdateResultH\x00\x12\x34\n\x0b\x65xit_result\x18\x12 \x01(\x0b\x32\x1d.wandb_internal.RunExitResultH\x00\x12\x33\n\nlog_result\x18\x14 \x01(\x0b\x32\x1d.wandb_internal.HistoryResultH\x00\x12\x37\n\x0esummary_result\x18\x15 \x01(\x0b\x32\x1d.wandb_internal.SummaryResultH\x00\x12\x35\n\routput_result\x18\x16 \x01(\x0b\x32\x1c.wandb_internal.OutputResultH\x00\x12\x35\n\rconfig_result\x18\x17 \x01(\x0b\x32\x1c.wandb_internal.ConfigResultH\x00\x12,\n\x08response\x18\x64 \x01(\x0b\x32\x18.wandb_internal.ResponseH\x00\x12(\n\x07\x63ontrol\x18\x10 \x01(\x0b\x32\x17.wandb_internal.Control\x12\x0c\n\x04uuid\x18\x18 \x01(\t\x12+\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1b.wandb_internal._ResultInfoB\r\n\x0bresult_type\":\n\x0b\x46inalRecord\x12+\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1b.wandb_internal._RecordInfo\"b\n\x0bVersionInfo\x12\x10\n\x08producer\x18\x01 \x01(\t\x12\x14\n\x0cmin_consumer\x18\x02 \x01(\t\x12+\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1b.wandb_internal._RecordInfo\"n\n\x0cHeaderRecord\x12\x31\n\x0cversion_info\x18\x01 \x01(\x0b\x32\x1b.wandb_internal.VersionInfo\x12+\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1b.wandb_internal._RecordInfo\";\n\x0c\x46ooterRecord\x12+\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1b.wandb_internal._RecordInfo\"9\n\x0b\x42ranchPoint\x12\x0b\n\x03run\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\x01\x12\x0e\n\x06metric\x18\x03 \x01(\t\"\x91\x05\n\tRunRecord\x12\x0e\n\x06run_id\x18\x01 \x01(\t\x12\x0e\n\x06\x65ntity\x18\x02 \x01(\t\x12\x0f\n\x07project\x18\x03 \x01(\t\x12,\n\x06\x63onfig\x18\x04 \x01(\x0b\x32\x1c.wandb_internal.ConfigRecord\x12.\n\x07summary\x18\x05 \x01(\x0b\x32\x1d.wandb_internal.SummaryRecord\x12\x11\n\trun_group\x18\x06 \x01(\t\x12\x10\n\x08job_type\x18\x07 \x01(\t\x12\x14\n\x0c\x64isplay_name\x18\x08 \x01(\t\x12\r\n\x05notes\x18\t \x01(\t\x12\x0c\n\x04tags\x18\n \x03(\t\x12\x30\n\x08settings\x18\x0b \x01(\x0b\x32\x1e.wandb_internal.SettingsRecord\x12\x10\n\x08sweep_id\x18\x0c \x01(\t\x12\x0c\n\x04host\x18\r \x01(\t\x12\x15\n\rstarting_step\x18\x0e \x01(\x03\x12\x12\n\nstorage_id\x18\x10 \x01(\t\x12.\n\nstart_time\x18\x11 \x01(\x0b\x32\x1a.google.protobuf.Timestamp\x12\x0f\n\x07resumed\x18\x12 \x01(\x08\x12\x32\n\ttelemetry\x18\x13 \x01(\x0b\x32\x1f.wandb_internal.TelemetryRecord\x12\x0f\n\x07runtime\x18\x14 \x01(\x05\x12*\n\x03git\x18\x15 \x01(\x0b\x32\x1d.wandb_internal.GitRepoRecord\x12\x0e\n\x06\x66orked\x18\x16 \x01(\x08\x12\x31\n\x0c\x62ranch_point\x18\x17 \x01(\x0b\x32\x1b.wandb_internal.BranchPoint\x12+\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1b.wandb_internal._RecordInfo\";\n\rGitRepoRecord\x12\x1a\n\nremote_url\x18\x01 \x01(\tR\x06remote\x12\x0e\n\x06\x63ommit\x18\x02 \x01(\t\"c\n\x0fRunUpdateResult\x12&\n\x03run\x18\x01 \x01(\x0b\x32\x19.wandb_internal.RunRecord\x12(\n\x05\x65rror\x18\x02 \x01(\x0b\x32\x19.wandb_internal.ErrorInfo\"\xac\x01\n\tErrorInfo\x12\x0f\n\x07message\x18\x01 \x01(\t\x12\x31\n\x04\x63ode\x18\x02 \x01(\x0e\x32#.wandb_internal.ErrorInfo.ErrorCode\"[\n\tErrorCode\x12\x0b\n\x07UNKNOWN\x10\x00\x12\x11\n\rCOMMUNICATION\x10\x01\x12\x12\n\x0e\x41UTHENTICATION\x10\x02\x12\t\n\x05USAGE\x10\x03\x12\x0f\n\x0bUNSUPPORTED\x10\x04\"v\n\rRunExitRecord\x12\x11\n\texit_code\x18\x01 \x01(\x05\x12\x14\n\x0cnot_complete\x18\x03 \x01(\x08\x12\x0f\n\x07runtime\x18\x02 \x01(\x05\x12+\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1b.wandb_internal._RecordInfo\"\x0f\n\rRunExitResult\"B\n\x13RunPreemptingRecord\x12+\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1b.wandb_internal._RecordInfo\"\x15\n\x13RunPreemptingResult\"i\n\x0eSettingsRecord\x12*\n\x04item\x18\x01 \x03(\x0b\x32\x1c.wandb_internal.SettingsItem\x12+\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1b.wandb_internal._RecordInfo\"/\n\x0cSettingsItem\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\x12\n\nvalue_json\x18\x10 \x01(\t\"\x1a\n\x0bHistoryStep\x12\x0b\n\x03num\x18\x01 \x01(\x03\"\x92\x01\n\rHistoryRecord\x12)\n\x04item\x18\x01 \x03(\x0b\x32\x1b.wandb_internal.HistoryItem\x12)\n\x04step\x18\x02 \x01(\x0b\x32\x1b.wandb_internal.HistoryStep\x12+\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1b.wandb_internal._RecordInfo\"B\n\x0bHistoryItem\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\x12\n\nnested_key\x18\x02 \x03(\t\x12\x12\n\nvalue_json\x18\x10 \x01(\t\"\x0f\n\rHistoryResult\"\xdc\x01\n\x0cOutputRecord\x12<\n\x0boutput_type\x18\x01 \x01(\x0e\x32\'.wandb_internal.OutputRecord.OutputType\x12-\n\ttimestamp\x18\x02 \x01(\x0b\x32\x1a.google.protobuf.Timestamp\x12\x0c\n\x04line\x18\x03 \x01(\t\x12+\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1b.wandb_internal._RecordInfo\"$\n\nOutputType\x12\n\n\x06STDERR\x10\x00\x12\n\n\x06STDOUT\x10\x01\"\x0e\n\x0cOutputResult\"\xe2\x01\n\x0fOutputRawRecord\x12?\n\x0boutput_type\x18\x01 \x01(\x0e\x32*.wandb_internal.OutputRawRecord.OutputType\x12-\n\ttimestamp\x18\x02 \x01(\x0b\x32\x1a.google.protobuf.Timestamp\x12\x0c\n\x04line\x18\x03 \x01(\t\x12+\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1b.wandb_internal._RecordInfo\"$\n\nOutputType\x12\n\n\x06STDERR\x10\x00\x12\n\n\x06STDOUT\x10\x01\"\x11\n\x0fOutputRawResult\"\xb4\x03\n\x0cMetricRecord\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x11\n\tglob_name\x18\x02 \x01(\t\x12\x13\n\x0bstep_metric\x18\x04 \x01(\t\x12\x19\n\x11step_metric_index\x18\x05 \x01(\x05\x12.\n\x07options\x18\x06 \x01(\x0b\x32\x1d.wandb_internal.MetricOptions\x12.\n\x07summary\x18\x07 \x01(\x0b\x32\x1d.wandb_internal.MetricSummary\x12\x35\n\x04goal\x18\x08 \x01(\x0e\x32\'.wandb_internal.MetricRecord.MetricGoal\x12/\n\x08_control\x18\t \x01(\x0b\x32\x1d.wandb_internal.MetricControl\x12\x1a\n\x12\x65xpanded_from_glob\x18\n \x01(\x08\x12+\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1b.wandb_internal._RecordInfo\"B\n\nMetricGoal\x12\x0e\n\nGOAL_UNSET\x10\x00\x12\x11\n\rGOAL_MINIMIZE\x10\x01\x12\x11\n\rGOAL_MAXIMIZE\x10\x02\"\x0e\n\x0cMetricResult\"C\n\rMetricOptions\x12\x11\n\tstep_sync\x18\x01 \x01(\x08\x12\x0e\n\x06hidden\x18\x02 \x01(\x08\x12\x0f\n\x07\x64\x65\x66ined\x18\x03 \x01(\x08\"\"\n\rMetricControl\x12\x11\n\toverwrite\x18\x01 \x01(\x08\"~\n\rMetricSummary\x12\x0b\n\x03min\x18\x01 \x01(\x08\x12\x0b\n\x03max\x18\x02 \x01(\x08\x12\x0c\n\x04mean\x18\x03 \x01(\x08\x12\x0c\n\x04\x62\x65st\x18\x04 \x01(\x08\x12\x0c\n\x04last\x18\x05 \x01(\x08\x12\x0c\n\x04none\x18\x06 \x01(\x08\x12\x0c\n\x04\x63opy\x18\x07 \x01(\x08\x12\r\n\x05\x66irst\x18\x08 \x01(\x08\"\x93\x01\n\x0c\x43onfigRecord\x12*\n\x06update\x18\x01 \x03(\x0b\x32\x1a.wandb_internal.ConfigItem\x12*\n\x06remove\x18\x02 \x03(\x0b\x32\x1a.wandb_internal.ConfigItem\x12+\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1b.wandb_internal._RecordInfo\"A\n\nConfigItem\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\x12\n\nnested_key\x18\x02 \x03(\t\x12\x12\n\nvalue_json\x18\x10 \x01(\t\"\x0e\n\x0c\x43onfigResult\"\x96\x01\n\rSummaryRecord\x12+\n\x06update\x18\x01 \x03(\x0b\x32\x1b.wandb_internal.SummaryItem\x12+\n\x06remove\x18\x02 \x03(\x0b\x32\x1b.wandb_internal.SummaryItem\x12+\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1b.wandb_internal._RecordInfo\"B\n\x0bSummaryItem\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\x12\n\nnested_key\x18\x02 \x03(\t\x12\x12\n\nvalue_json\x18\x10 \x01(\t\"\x0f\n\rSummaryResult\"d\n\x0b\x46ilesRecord\x12(\n\x05\x66iles\x18\x01 \x03(\x0b\x32\x19.wandb_internal.FilesItem\x12+\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1b.wandb_internal._RecordInfo\"\xec\x01\n\tFilesItem\x12\x0c\n\x04path\x18\x01 \x01(\t\x12\x34\n\x06policy\x18\x02 \x01(\x0e\x32$.wandb_internal.FilesItem.PolicyType\x12\x30\n\x04type\x18\x03 \x01(\x0e\x32\".wandb_internal.FilesItem.FileType\"(\n\nPolicyType\x12\x07\n\x03NOW\x10\x00\x12\x07\n\x03\x45ND\x10\x01\x12\x08\n\x04LIVE\x10\x02\"9\n\x08\x46ileType\x12\t\n\x05OTHER\x10\x00\x12\t\n\x05WANDB\x10\x01\x12\t\n\x05MEDIA\x10\x02\x12\x0c\n\x08\x41RTIFACT\x10\x03J\x04\x08\x10\x10\x11\"\r\n\x0b\x46ilesResult\"\xe6\x01\n\x0bStatsRecord\x12\x39\n\nstats_type\x18\x01 \x01(\x0e\x32%.wandb_internal.StatsRecord.StatsType\x12-\n\ttimestamp\x18\x02 \x01(\x0b\x32\x1a.google.protobuf.Timestamp\x12\'\n\x04item\x18\x03 \x03(\x0b\x32\x19.wandb_internal.StatsItem\x12+\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1b.wandb_internal._RecordInfo\"\x17\n\tStatsType\x12\n\n\x06SYSTEM\x10\x00\",\n\tStatsItem\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\x12\n\nvalue_json\x18\x10 \x01(\t\"\xe7\x03\n\x0e\x41rtifactRecord\x12\x0e\n\x06run_id\x18\x01 \x01(\t\x12\x0f\n\x07project\x18\x02 \x01(\t\x12\x0e\n\x06\x65ntity\x18\x03 \x01(\t\x12\x0c\n\x04type\x18\x04 \x01(\t\x12\x0c\n\x04name\x18\x05 \x01(\t\x12\x0e\n\x06\x64igest\x18\x06 \x01(\t\x12\x13\n\x0b\x64\x65scription\x18\x07 \x01(\t\x12\x10\n\x08metadata\x18\x08 \x01(\t\x12\x14\n\x0cuser_created\x18\t \x01(\x08\x12\x18\n\x10use_after_commit\x18\n \x01(\x08\x12\x0f\n\x07\x61liases\x18\x0b \x03(\t\x12\x32\n\x08manifest\x18\x0c \x01(\x0b\x32 .wandb_internal.ArtifactManifest\x12\x16\n\x0e\x64istributed_id\x18\r \x01(\t\x12\x10\n\x08\x66inalize\x18\x0e \x01(\x08\x12\x11\n\tclient_id\x18\x0f \x01(\t\x12\x1a\n\x12sequence_client_id\x18\x10 \x01(\t\x12\x0f\n\x07\x62\x61se_id\x18\x11 \x01(\t\x12\x1c\n\x14ttl_duration_seconds\x18\x12 \x01(\x03\x12\x0c\n\x04tags\x18\x13 \x03(\t\x12\x19\n\x11incremental_beta1\x18\x64 \x01(\x08\x12+\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1b.wandb_internal._RecordInfo\"\xd8\x01\n\x10\x41rtifactManifest\x12\x0f\n\x07version\x18\x01 \x01(\x05\x12\x16\n\x0estorage_policy\x18\x02 \x01(\t\x12\x46\n\x15storage_policy_config\x18\x03 \x03(\x0b\x32\'.wandb_internal.StoragePolicyConfigItem\x12\x37\n\x08\x63ontents\x18\x04 \x03(\x0b\x32%.wandb_internal.ArtifactManifestEntry\x12\x1a\n\x12manifest_file_path\x18\x05 \x01(\t\"\xcf\x01\n\x15\x41rtifactManifestEntry\x12\x0c\n\x04path\x18\x01 \x01(\t\x12\x0e\n\x06\x64igest\x18\x02 \x01(\t\x12\x0b\n\x03ref\x18\x03 \x01(\t\x12\x0c\n\x04size\x18\x04 \x01(\x03\x12\x10\n\x08mimetype\x18\x05 \x01(\t\x12\x12\n\nlocal_path\x18\x06 \x01(\t\x12\x19\n\x11\x62irth_artifact_id\x18\x07 \x01(\t\x12\x12\n\nskip_cache\x18\x08 \x01(\x08\x12(\n\x05\x65xtra\x18\x10 \x03(\x0b\x32\x19.wandb_internal.ExtraItem\",\n\tExtraItem\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\x12\n\nvalue_json\x18\x02 \x01(\t\":\n\x17StoragePolicyConfigItem\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\x12\n\nvalue_json\x18\x02 \x01(\t\"\x10\n\x0e\x41rtifactResult\"\x14\n\x12LinkArtifactResult\"\xf0\x01\n\x13LinkArtifactRequest\x12\x11\n\tclient_id\x18\x01 \x01(\t\x12\x11\n\tserver_id\x18\x02 \x01(\t\x12\x16\n\x0eportfolio_name\x18\x03 \x01(\t\x12\x18\n\x10portfolio_entity\x18\x04 \x01(\t\x12\x19\n\x11portfolio_project\x18\x05 \x01(\t\x12\x19\n\x11portfolio_aliases\x18\x06 \x03(\t\x12\x1e\n\x16portfolio_organization\x18\x07 \x01(\t\x12+\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1b.wandb_internal._RecordInfo\"[\n\x14LinkArtifactResponse\x12\x15\n\rerror_message\x18\x01 \x01(\t\x12\x1a\n\rversion_index\x18\x02 \x01(\x05H\x00\x88\x01\x01\x42\x10\n\x0e_version_index\"h\n\x08TBRecord\x12+\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1b.wandb_internal._RecordInfo\x12\x0f\n\x07log_dir\x18\x01 \x01(\t\x12\x10\n\x08root_dir\x18\x03 \x01(\t\x12\x0c\n\x04save\x18\x02 \x01(\x08\"\n\n\x08TBResult\"}\n\x0b\x41lertRecord\x12\r\n\x05title\x18\x01 \x01(\t\x12\x0c\n\x04text\x18\x02 \x01(\t\x12\r\n\x05level\x18\x03 \x01(\t\x12\x15\n\rwait_duration\x18\x04 \x01(\x03\x12+\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1b.wandb_internal._RecordInfo\"\r\n\x0b\x41lertResult\"\xf4\x10\n\x07Request\x12\x38\n\x0bstop_status\x18\x01 \x01(\x0b\x32!.wandb_internal.StopStatusRequestH\x00\x12>\n\x0enetwork_status\x18\x02 \x01(\x0b\x32$.wandb_internal.NetworkStatusRequestH\x00\x12-\n\x05\x64\x65\x66\x65r\x18\x03 \x01(\x0b\x32\x1c.wandb_internal.DeferRequestH\x00\x12\x38\n\x0bget_summary\x18\x04 \x01(\x0b\x32!.wandb_internal.GetSummaryRequestH\x00\x12-\n\x05login\x18\x05 \x01(\x0b\x32\x1c.wandb_internal.LoginRequestH\x00\x12-\n\x05pause\x18\x06 \x01(\x0b\x32\x1c.wandb_internal.PauseRequestH\x00\x12/\n\x06resume\x18\x07 \x01(\x0b\x32\x1d.wandb_internal.ResumeRequestH\x00\x12\x34\n\tpoll_exit\x18\x08 \x01(\x0b\x32\x1f.wandb_internal.PollExitRequestH\x00\x12@\n\x0fsampled_history\x18\t \x01(\x0b\x32%.wandb_internal.SampledHistoryRequestH\x00\x12@\n\x0fpartial_history\x18\n \x01(\x0b\x32%.wandb_internal.PartialHistoryRequestH\x00\x12\x34\n\trun_start\x18\x0b \x01(\x0b\x32\x1f.wandb_internal.RunStartRequestH\x00\x12<\n\rcheck_version\x18\x0c \x01(\x0b\x32#.wandb_internal.CheckVersionRequestH\x00\x12:\n\x0clog_artifact\x18\r \x01(\x0b\x32\".wandb_internal.LogArtifactRequestH\x00\x12\x44\n\x11\x64ownload_artifact\x18\x0e \x01(\x0b\x32\'.wandb_internal.DownloadArtifactRequestH\x00\x12\x35\n\tkeepalive\x18\x11 \x01(\x0b\x32 .wandb_internal.KeepaliveRequestH\x00\x12\x36\n\nrun_status\x18\x14 \x01(\x0b\x32 .wandb_internal.RunStatusRequestH\x00\x12/\n\x06\x63\x61ncel\x18\x15 \x01(\x0b\x32\x1d.wandb_internal.CancelRequestH\x00\x12\x44\n\x11internal_messages\x18\x17 \x01(\x0b\x32\'.wandb_internal.InternalMessagesRequestH\x00\x12@\n\x0fpython_packages\x18\x18 \x01(\x0b\x32%.wandb_internal.PythonPackagesRequestH\x00\x12\x33\n\x08shutdown\x18@ \x01(\x0b\x32\x1f.wandb_internal.ShutdownRequestH\x00\x12/\n\x06\x61ttach\x18\x41 \x01(\x0b\x32\x1d.wandb_internal.AttachRequestH\x00\x12/\n\x06status\x18\x42 \x01(\x0b\x32\x1d.wandb_internal.StatusRequestH\x00\x12\x38\n\x0bserver_info\x18\x43 \x01(\x0b\x32!.wandb_internal.ServerInfoRequestH\x00\x12\x38\n\x0bsender_mark\x18\x44 \x01(\x0b\x32!.wandb_internal.SenderMarkRequestH\x00\x12\x38\n\x0bsender_read\x18\x45 \x01(\x0b\x32!.wandb_internal.SenderReadRequestH\x00\x12<\n\rstatus_report\x18\x46 \x01(\x0b\x32#.wandb_internal.StatusReportRequestH\x00\x12>\n\x0esummary_record\x18G \x01(\x0b\x32$.wandb_internal.SummaryRecordRequestH\x00\x12\x42\n\x10telemetry_record\x18H \x01(\x0b\x32&.wandb_internal.TelemetryRecordRequestH\x00\x12\x32\n\x08job_info\x18I \x01(\x0b\x32\x1e.wandb_internal.JobInfoRequestH\x00\x12\x45\n\x12get_system_metrics\x18J \x01(\x0b\x32\'.wandb_internal.GetSystemMetricsRequestH\x00\x12\x34\n\tjob_input\x18M \x01(\x0b\x32\x1f.wandb_internal.JobInputRequestH\x00\x12<\n\rlink_artifact\x18N \x01(\x0b\x32#.wandb_internal.LinkArtifactRequestH\x00\x12\x38\n\x0bsync_finish\x18Q \x01(\x0b\x32!.wandb_internal.SyncFinishRequestH\x00\x12;\n\noperations\x18R \x01(\x0b\x32%.wandb_internal.OperationStatsRequestH\x00\x12\x43\n\x11probe_system_info\x18S \x01(\x0b\x32&.wandb_internal.ProbeSystemInfoRequestH\x00\x12\x39\n\x0btest_inject\x18\xe8\x07 \x01(\x0b\x32!.wandb_internal.TestInjectRequestH\x00\x42\x0e\n\x0crequest_typeJ\x04\x08\x12\x10\x13J\x04\x08\x16\x10\x17J\x04\x08K\x10LJ\x04\x08L\x10MJ\x04\x08O\x10PJ\x04\x08P\x10Q\"\x83\r\n\x08Response\x12?\n\x12keepalive_response\x18\x12 \x01(\x0b\x32!.wandb_internal.KeepaliveResponseH\x00\x12\x42\n\x14stop_status_response\x18\x13 \x01(\x0b\x32\".wandb_internal.StopStatusResponseH\x00\x12H\n\x17network_status_response\x18\x14 \x01(\x0b\x32%.wandb_internal.NetworkStatusResponseH\x00\x12\x37\n\x0elogin_response\x18\x18 \x01(\x0b\x32\x1d.wandb_internal.LoginResponseH\x00\x12\x42\n\x14get_summary_response\x18\x19 \x01(\x0b\x32\".wandb_internal.GetSummaryResponseH\x00\x12>\n\x12poll_exit_response\x18\x1a \x01(\x0b\x32 .wandb_internal.PollExitResponseH\x00\x12J\n\x18sampled_history_response\x18\x1b \x01(\x0b\x32&.wandb_internal.SampledHistoryResponseH\x00\x12>\n\x12run_start_response\x18\x1c \x01(\x0b\x32 .wandb_internal.RunStartResponseH\x00\x12\x46\n\x16\x63heck_version_response\x18\x1d \x01(\x0b\x32$.wandb_internal.CheckVersionResponseH\x00\x12\x44\n\x15log_artifact_response\x18\x1e \x01(\x0b\x32#.wandb_internal.LogArtifactResponseH\x00\x12N\n\x1a\x64ownload_artifact_response\x18\x1f \x01(\x0b\x32(.wandb_internal.DownloadArtifactResponseH\x00\x12@\n\x13run_status_response\x18# \x01(\x0b\x32!.wandb_internal.RunStatusResponseH\x00\x12\x39\n\x0f\x63\x61ncel_response\x18$ \x01(\x0b\x32\x1e.wandb_internal.CancelResponseH\x00\x12N\n\x1ainternal_messages_response\x18% \x01(\x0b\x32(.wandb_internal.InternalMessagesResponseH\x00\x12=\n\x11shutdown_response\x18@ \x01(\x0b\x32 .wandb_internal.ShutdownResponseH\x00\x12\x39\n\x0f\x61ttach_response\x18\x41 \x01(\x0b\x32\x1e.wandb_internal.AttachResponseH\x00\x12\x39\n\x0fstatus_response\x18\x42 \x01(\x0b\x32\x1e.wandb_internal.StatusResponseH\x00\x12\x42\n\x14server_info_response\x18\x43 \x01(\x0b\x32\".wandb_internal.ServerInfoResponseH\x00\x12<\n\x11job_info_response\x18\x44 \x01(\x0b\x32\x1f.wandb_internal.JobInfoResponseH\x00\x12O\n\x1bget_system_metrics_response\x18\x45 \x01(\x0b\x32(.wandb_internal.GetSystemMetricsResponseH\x00\x12\x46\n\x16link_artifact_response\x18G \x01(\x0b\x32$.wandb_internal.LinkArtifactResponseH\x00\x12\x35\n\rsync_response\x18\x46 \x01(\x0b\x32\x1c.wandb_internal.SyncResponseH\x00\x12\x45\n\x13operations_response\x18J \x01(\x0b\x32&.wandb_internal.OperationStatsResponseH\x00\x12\x43\n\x14test_inject_response\x18\xe8\x07 \x01(\x0b\x32\".wandb_internal.TestInjectResponseH\x00\x42\x0f\n\rresponse_typeJ\x04\x08 \x10!J\x04\x08H\x10IJ\x04\x08I\x10J\"\xc0\x02\n\x0c\x44\x65\x66\x65rRequest\x12\x36\n\x05state\x18\x01 \x01(\x0e\x32\'.wandb_internal.DeferRequest.DeferState\"\xf7\x01\n\nDeferState\x12\t\n\x05\x42\x45GIN\x10\x00\x12\r\n\tFLUSH_RUN\x10\x01\x12\x0f\n\x0b\x46LUSH_STATS\x10\x02\x12\x19\n\x15\x46LUSH_PARTIAL_HISTORY\x10\x03\x12\x0c\n\x08\x46LUSH_TB\x10\x04\x12\r\n\tFLUSH_SUM\x10\x05\x12\x13\n\x0f\x46LUSH_DEBOUNCER\x10\x06\x12\x10\n\x0c\x46LUSH_OUTPUT\x10\x07\x12\r\n\tFLUSH_JOB\x10\x08\x12\r\n\tFLUSH_DIR\x10\t\x12\x0c\n\x08\x46LUSH_FP\x10\n\x12\x0b\n\x07JOIN_FP\x10\x0b\x12\x0c\n\x08\x46LUSH_FS\x10\x0c\x12\x0f\n\x0b\x46LUSH_FINAL\x10\r\x12\x07\n\x03\x45ND\x10\x0e\"<\n\x0cPauseRequest\x12,\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1c.wandb_internal._RequestInfo\"\x0f\n\rPauseResponse\"=\n\rResumeRequest\x12,\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1c.wandb_internal._RequestInfo\"\x10\n\x0eResumeResponse\"M\n\x0cLoginRequest\x12\x0f\n\x07\x61pi_key\x18\x01 \x01(\t\x12,\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1c.wandb_internal._RequestInfo\"&\n\rLoginResponse\x12\x15\n\ractive_entity\x18\x01 \x01(\t\"A\n\x11GetSummaryRequest\x12,\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1c.wandb_internal._RequestInfo\"?\n\x12GetSummaryResponse\x12)\n\x04item\x18\x01 \x03(\x0b\x32\x1b.wandb_internal.SummaryItem\"G\n\x17GetSystemMetricsRequest\x12,\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1c.wandb_internal._RequestInfo\"R\n\x12SystemMetricSample\x12-\n\ttimestamp\x18\x01 \x01(\x0b\x32\x1a.google.protobuf.Timestamp\x12\r\n\x05value\x18\x02 \x01(\x02\"I\n\x13SystemMetricsBuffer\x12\x32\n\x06record\x18\x01 \x03(\x0b\x32\".wandb_internal.SystemMetricSample\"\xca\x01\n\x18GetSystemMetricsResponse\x12S\n\x0esystem_metrics\x18\x01 \x03(\x0b\x32;.wandb_internal.GetSystemMetricsResponse.SystemMetricsEntry\x1aY\n\x12SystemMetricsEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\x32\n\x05value\x18\x02 \x01(\x0b\x32#.wandb_internal.SystemMetricsBuffer:\x02\x38\x01\"=\n\rStatusRequest\x12,\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1c.wandb_internal._RequestInfo\")\n\x0eStatusResponse\x12\x17\n\x0frun_should_stop\x18\x01 \x01(\x08\"A\n\x11StopStatusRequest\x12,\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1c.wandb_internal._RequestInfo\"-\n\x12StopStatusResponse\x12\x17\n\x0frun_should_stop\x18\x01 \x01(\x08\"D\n\x14NetworkStatusRequest\x12,\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1c.wandb_internal._RequestInfo\"P\n\x15NetworkStatusResponse\x12\x37\n\x11network_responses\x18\x01 \x03(\x0b\x32\x1c.wandb_internal.HttpResponse\"D\n\x0cHttpResponse\x12\x18\n\x10http_status_code\x18\x01 \x01(\x05\x12\x1a\n\x12http_response_text\x18\x02 \x01(\t\"G\n\x17InternalMessagesRequest\x12,\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1c.wandb_internal._RequestInfo\"N\n\x18InternalMessagesResponse\x12\x32\n\x08messages\x18\x01 \x01(\x0b\x32 .wandb_internal.InternalMessages\"#\n\x10InternalMessages\x12\x0f\n\x07warning\x18\x01 \x03(\t\"?\n\x0fPollExitRequest\x12,\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1c.wandb_internal._RequestInfo\"\xf5\x01\n\x10PollExitResponse\x12\x0c\n\x04\x64one\x18\x01 \x01(\x08\x12\x32\n\x0b\x65xit_result\x18\x02 \x01(\x0b\x32\x1d.wandb_internal.RunExitResult\x12\x35\n\x0cpusher_stats\x18\x03 \x01(\x0b\x32\x1f.wandb_internal.FilePusherStats\x12/\n\x0b\x66ile_counts\x18\x04 \x01(\x0b\x32\x1a.wandb_internal.FileCounts\x12\x37\n\x0foperation_stats\x18\x05 \x01(\x0b\x32\x1e.wandb_internal.OperationStats\"E\n\x15OperationStatsRequest\x12,\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1c.wandb_internal._RequestInfo\"Q\n\x16OperationStatsResponse\x12\x37\n\x0foperation_stats\x18\x01 \x01(\x0b\x32\x1e.wandb_internal.OperationStats\"h\n\x0eOperationStats\x12\r\n\x05label\x18\x03 \x01(\t\x12-\n\noperations\x18\x01 \x03(\x0b\x32\x19.wandb_internal.Operation\x12\x18\n\x10total_operations\x18\x02 \x01(\x03\"\x87\x01\n\tOperation\x12\x0c\n\x04\x64\x65sc\x18\x01 \x01(\t\x12\x17\n\x0fruntime_seconds\x18\x02 \x01(\x01\x12\x10\n\x08progress\x18\x03 \x01(\t\x12\x14\n\x0c\x65rror_status\x18\x04 \x01(\t\x12+\n\x08subtasks\x18\x05 \x03(\x0b\x32\x19.wandb_internal.Operation\"\x13\n\x11SenderMarkRequest\"\x13\n\x11SyncFinishRequest\"E\n\x0cSyncResponse\x12\x0b\n\x03url\x18\x01 \x01(\t\x12(\n\x05\x65rror\x18\x02 \x01(\x0b\x32\x19.wandb_internal.ErrorInfo\"?\n\x11SenderReadRequest\x12\x14\n\x0cstart_offset\x18\x01 \x01(\x03\x12\x14\n\x0c\x66inal_offset\x18\x02 \x01(\x03\"m\n\x13StatusReportRequest\x12\x12\n\nrecord_num\x18\x01 \x01(\x03\x12\x13\n\x0bsent_offset\x18\x02 \x01(\x03\x12-\n\tsync_time\x18\x03 \x01(\x0b\x32\x1a.google.protobuf.Timestamp\"F\n\x14SummaryRecordRequest\x12.\n\x07summary\x18\x01 \x01(\x0b\x32\x1d.wandb_internal.SummaryRecord\"L\n\x16TelemetryRecordRequest\x12\x32\n\ttelemetry\x18\x01 \x01(\x0b\x32\x1f.wandb_internal.TelemetryRecord\"A\n\x11ServerInfoRequest\x12,\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1c.wandb_internal._RequestInfo\"|\n\x12ServerInfoResponse\x12-\n\nlocal_info\x18\x01 \x01(\x0b\x32\x19.wandb_internal.LocalInfo\x12\x37\n\x0fserver_messages\x18\x02 \x01(\x0b\x32\x1e.wandb_internal.ServerMessages\"=\n\x0eServerMessages\x12+\n\x04item\x18\x01 \x03(\x0b\x32\x1d.wandb_internal.ServerMessage\"e\n\rServerMessage\x12\x12\n\nplain_text\x18\x01 \x01(\t\x12\x10\n\x08utf_text\x18\x02 \x01(\t\x12\x11\n\thtml_text\x18\x03 \x01(\t\x12\x0c\n\x04type\x18\x04 \x01(\t\x12\r\n\x05level\x18\x05 \x01(\x05\"c\n\nFileCounts\x12\x13\n\x0bwandb_count\x18\x01 \x01(\x05\x12\x13\n\x0bmedia_count\x18\x02 \x01(\x05\x12\x16\n\x0e\x61rtifact_count\x18\x03 \x01(\x05\x12\x13\n\x0bother_count\x18\x04 \x01(\x05\"U\n\x0f\x46ilePusherStats\x12\x16\n\x0euploaded_bytes\x18\x01 \x01(\x03\x12\x13\n\x0btotal_bytes\x18\x02 \x01(\x03\x12\x15\n\rdeduped_bytes\x18\x03 \x01(\x03\"\x1e\n\rFilesUploaded\x12\r\n\x05\x66iles\x18\x01 \x03(\t\"\xf4\x01\n\x17\x46ileTransferInfoRequest\x12\x42\n\x04type\x18\x01 \x01(\x0e\x32\x34.wandb_internal.FileTransferInfoRequest.TransferType\x12\x0c\n\x04path\x18\x02 \x01(\t\x12\x0b\n\x03url\x18\x03 \x01(\t\x12\x0c\n\x04size\x18\x04 \x01(\x03\x12\x11\n\tprocessed\x18\x05 \x01(\x03\x12/\n\x0b\x66ile_counts\x18\x06 \x01(\x0b\x32\x1a.wandb_internal.FileCounts\"(\n\x0cTransferType\x12\n\n\x06Upload\x10\x00\x12\x0c\n\x08\x44ownload\x10\x01\"1\n\tLocalInfo\x12\x0f\n\x07version\x18\x01 \x01(\t\x12\x13\n\x0bout_of_date\x18\x02 \x01(\x08\"?\n\x0fShutdownRequest\x12,\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1c.wandb_internal._RequestInfo\"\x12\n\x10ShutdownResponse\"P\n\rAttachRequest\x12\x11\n\tattach_id\x18\x14 \x01(\t\x12,\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1c.wandb_internal._RequestInfo\"b\n\x0e\x41ttachResponse\x12&\n\x03run\x18\x01 \x01(\x0b\x32\x19.wandb_internal.RunRecord\x12(\n\x05\x65rror\x18\x02 \x01(\x0b\x32\x19.wandb_internal.ErrorInfo\"\xd5\x02\n\x11TestInjectRequest\x12\x13\n\x0bhandler_exc\x18\x01 \x01(\x08\x12\x14\n\x0chandler_exit\x18\x02 \x01(\x08\x12\x15\n\rhandler_abort\x18\x03 \x01(\x08\x12\x12\n\nsender_exc\x18\x04 \x01(\x08\x12\x13\n\x0bsender_exit\x18\x05 \x01(\x08\x12\x14\n\x0csender_abort\x18\x06 \x01(\x08\x12\x0f\n\x07req_exc\x18\x07 \x01(\x08\x12\x10\n\x08req_exit\x18\x08 \x01(\x08\x12\x11\n\treq_abort\x18\t \x01(\x08\x12\x10\n\x08resp_exc\x18\n \x01(\x08\x12\x11\n\tresp_exit\x18\x0b \x01(\x08\x12\x12\n\nresp_abort\x18\x0c \x01(\x08\x12\x10\n\x08msg_drop\x18\r \x01(\x08\x12\x10\n\x08msg_hang\x18\x0e \x01(\x08\x12,\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1c.wandb_internal._RequestInfo\"\x14\n\x12TestInjectResponse\"\x1e\n\rHistoryAction\x12\r\n\x05\x66lush\x18\x01 \x01(\x08\"\xca\x01\n\x15PartialHistoryRequest\x12)\n\x04item\x18\x01 \x03(\x0b\x32\x1b.wandb_internal.HistoryItem\x12)\n\x04step\x18\x02 \x01(\x0b\x32\x1b.wandb_internal.HistoryStep\x12-\n\x06\x61\x63tion\x18\x03 \x01(\x0b\x32\x1d.wandb_internal.HistoryAction\x12,\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1c.wandb_internal._RequestInfo\"\x18\n\x16PartialHistoryResponse\"E\n\x15SampledHistoryRequest\x12,\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1c.wandb_internal._RequestInfo\"_\n\x12SampledHistoryItem\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\x12\n\nnested_key\x18\x02 \x03(\t\x12\x14\n\x0cvalues_float\x18\x03 \x03(\x02\x12\x12\n\nvalues_int\x18\x04 \x03(\x03\"J\n\x16SampledHistoryResponse\x12\x30\n\x04item\x18\x01 \x03(\x0b\x32\".wandb_internal.SampledHistoryItem\"@\n\x10RunStatusRequest\x12,\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1c.wandb_internal._RequestInfo\"x\n\x11RunStatusResponse\x12\x18\n\x10sync_items_total\x18\x01 \x01(\x03\x12\x1a\n\x12sync_items_pending\x18\x02 \x01(\x03\x12-\n\tsync_time\x18\x03 \x01(\x0b\x32\x1a.google.protobuf.Timestamp\"g\n\x0fRunStartRequest\x12&\n\x03run\x18\x01 \x01(\x0b\x32\x19.wandb_internal.RunRecord\x12,\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1c.wandb_internal._RequestInfo\"\x12\n\x10RunStartResponse\"\\\n\x13\x43heckVersionRequest\x12\x17\n\x0f\x63urrent_version\x18\x01 \x01(\t\x12,\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1c.wandb_internal._RequestInfo\"]\n\x14\x43heckVersionResponse\x12\x17\n\x0fupgrade_message\x18\x01 \x01(\t\x12\x14\n\x0cyank_message\x18\x02 \x01(\t\x12\x16\n\x0e\x64\x65lete_message\x18\x03 \x01(\t\">\n\x0eJobInfoRequest\x12,\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1c.wandb_internal._RequestInfo\"6\n\x0fJobInfoResponse\x12\x12\n\nsequenceId\x18\x01 \x01(\t\x12\x0f\n\x07version\x18\x02 \x01(\t\"\x9f\x01\n\x12LogArtifactRequest\x12\x30\n\x08\x61rtifact\x18\x01 \x01(\x0b\x32\x1e.wandb_internal.ArtifactRecord\x12\x14\n\x0chistory_step\x18\x02 \x01(\x03\x12\x13\n\x0bstaging_dir\x18\x03 \x01(\t\x12,\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1c.wandb_internal._RequestInfo\"A\n\x13LogArtifactResponse\x12\x13\n\x0b\x61rtifact_id\x18\x01 \x01(\t\x12\x15\n\rerror_message\x18\x02 \x01(\t\"\xbe\x01\n\x17\x44ownloadArtifactRequest\x12\x13\n\x0b\x61rtifact_id\x18\x01 \x01(\t\x12\x15\n\rdownload_root\x18\x02 \x01(\t\x12 \n\x18\x61llow_missing_references\x18\x04 \x01(\x08\x12\x12\n\nskip_cache\x18\x05 \x01(\x08\x12\x13\n\x0bpath_prefix\x18\x06 \x01(\t\x12,\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1c.wandb_internal._RequestInfo\"1\n\x18\x44ownloadArtifactResponse\x12\x15\n\rerror_message\x18\x01 \x01(\t\"@\n\x10KeepaliveRequest\x12,\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1c.wandb_internal._RequestInfo\"\x13\n\x11KeepaliveResponse\"q\n\x0c\x41rtifactInfo\x12\x10\n\x08\x61rtifact\x18\x01 \x01(\t\x12\x12\n\nentrypoint\x18\x02 \x03(\t\x12\x10\n\x08notebook\x18\x03 \x01(\x08\x12\x15\n\rbuild_context\x18\x04 \x01(\t\x12\x12\n\ndockerfile\x18\x05 \x01(\t\")\n\x07GitInfo\x12\x0e\n\x06remote\x18\x01 \x01(\t\x12\x0e\n\x06\x63ommit\x18\x02 \x01(\t\"\x87\x01\n\tGitSource\x12)\n\x08git_info\x18\x01 \x01(\x0b\x32\x17.wandb_internal.GitInfo\x12\x12\n\nentrypoint\x18\x02 \x03(\t\x12\x10\n\x08notebook\x18\x03 \x01(\x08\x12\x15\n\rbuild_context\x18\x04 \x01(\t\x12\x12\n\ndockerfile\x18\x05 \x01(\t\"\x1c\n\x0bImageSource\x12\r\n\x05image\x18\x01 \x01(\t\"\x8c\x01\n\x06Source\x12&\n\x03git\x18\x01 \x01(\x0b\x32\x19.wandb_internal.GitSource\x12.\n\x08\x61rtifact\x18\x02 \x01(\x0b\x32\x1c.wandb_internal.ArtifactInfo\x12*\n\x05image\x18\x03 \x01(\x0b\x32\x1b.wandb_internal.ImageSource\"k\n\tJobSource\x12\x10\n\x08_version\x18\x01 \x01(\t\x12\x13\n\x0bsource_type\x18\x02 \x01(\t\x12&\n\x06source\x18\x03 \x01(\x0b\x32\x16.wandb_internal.Source\x12\x0f\n\x07runtime\x18\x04 \x01(\t\"V\n\x12PartialJobArtifact\x12\x10\n\x08job_name\x18\x01 \x01(\t\x12.\n\x0bsource_info\x18\x02 \x01(\x0b\x32\x19.wandb_internal.JobSource\"\x9d\x01\n\x11UseArtifactRecord\x12\n\n\x02id\x18\x01 \x01(\t\x12\x0c\n\x04type\x18\x02 \x01(\t\x12\x0c\n\x04name\x18\x03 \x01(\t\x12\x33\n\x07partial\x18\x04 \x01(\x0b\x32\".wandb_internal.PartialJobArtifact\x12+\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1b.wandb_internal._RecordInfo\"\x13\n\x11UseArtifactResult\"R\n\rCancelRequest\x12\x13\n\x0b\x63\x61ncel_slot\x18\x01 \x01(\t\x12,\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1c.wandb_internal._RequestInfo\"\x10\n\x0e\x43\x61ncelResponse\"\x18\n\x16ProbeSystemInfoRequest\"\'\n\x08\x44iskInfo\x12\r\n\x05total\x18\x01 \x01(\x04\x12\x0c\n\x04used\x18\x02 \x01(\x04\"\x1b\n\nMemoryInfo\x12\r\n\x05total\x18\x01 \x01(\x04\"/\n\x07\x43puInfo\x12\r\n\x05\x63ount\x18\x01 \x01(\r\x12\x15\n\rcount_logical\x18\x02 \x01(\r\"\xad\x01\n\tAppleInfo\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x12\n\necpu_cores\x18\x02 \x01(\r\x12\x12\n\npcpu_cores\x18\x03 \x01(\r\x12\x11\n\tgpu_cores\x18\x04 \x01(\r\x12\x11\n\tmemory_gb\x18\x05 \x01(\r\x12\x18\n\x10swap_total_bytes\x18\x06 \x01(\x04\x12\x17\n\x0fram_total_bytes\x18\x07 \x01(\x04\x12\x11\n\tmac_model\x18\x08 \x01(\t\"k\n\rGpuNvidiaInfo\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x14\n\x0cmemory_total\x18\x02 \x01(\x04\x12\x12\n\ncuda_cores\x18\x03 \x01(\r\x12\x14\n\x0c\x61rchitecture\x18\x04 \x01(\t\x12\x0c\n\x04uuid\x18\x05 \x01(\t\"\x89\x02\n\nGpuAmdInfo\x12\n\n\x02id\x18\x01 \x01(\t\x12\x11\n\tunique_id\x18\x02 \x01(\t\x12\x15\n\rvbios_version\x18\x03 \x01(\t\x12\x19\n\x11performance_level\x18\x04 \x01(\t\x12\x15\n\rgpu_overdrive\x18\x05 \x01(\t\x12\x1c\n\x14gpu_memory_overdrive\x18\x06 \x01(\t\x12\x11\n\tmax_power\x18\x07 \x01(\t\x12\x0e\n\x06series\x18\x08 \x01(\t\x12\r\n\x05model\x18\t \x01(\t\x12\x0e\n\x06vendor\x18\n \x01(\t\x12\x0b\n\x03sku\x18\x0b \x01(\t\x12\x12\n\nsclk_range\x18\x0c \x01(\t\x12\x12\n\nmclk_range\x18\r \x01(\t\"n\n\x0cTrainiumInfo\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x0e\n\x06vendor\x18\x02 \x01(\t\x12\x1b\n\x13neuron_device_count\x18\x03 \x01(\r\x12#\n\x1bneuroncore_per_device_count\x18\x04 \x01(\r\"Q\n\x07TPUInfo\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x0f\n\x07hbm_gib\x18\x02 \x01(\r\x12\x18\n\x10\x64\x65vices_per_chip\x18\x03 \x01(\r\x12\r\n\x05\x63ount\x18\x04 \x01(\r\"E\n\rCoreWeaveInfo\x12\x14\n\x0c\x63luster_name\x18\x01 \x01(\t\x12\x0e\n\x06org_id\x18\x02 \x01(\t\x12\x0e\n\x06region\x18\x03 \x01(\t\"\xa8\t\n\x11\x45nvironmentRecord\x12\n\n\x02os\x18\x01 \x01(\t\x12\x0e\n\x06python\x18\x02 \x01(\t\x12\x39\n\nstarted_at\x18\x03 \x01(\x0b\x32\x1a.google.protobuf.TimestampR\tstartedAt\x12\x0e\n\x06\x64ocker\x18\x04 \x01(\t\x12\x0c\n\x04\x61rgs\x18\x05 \x03(\t\x12\x0f\n\x07program\x18\x06 \x01(\t\x12\x1b\n\tcode_path\x18\x07 \x01(\tR\x08\x63odePath\x12&\n\x0f\x63ode_path_local\x18\x08 \x01(\tR\rcodePathLocal\x12*\n\x03git\x18\t \x01(\x0b\x32\x1d.wandb_internal.GitRepoRecord\x12\r\n\x05\x65mail\x18\n \x01(\t\x12\x0c\n\x04root\x18\x0b \x01(\t\x12\x0c\n\x04host\x18\x0c \x01(\t\x12\x10\n\x08username\x18\r \x01(\t\x12\x12\n\nexecutable\x18\x0e \x01(\t\x12\r\n\x05\x63olab\x18\x0f \x01(\t\x12\x1c\n\tcpu_count\x18\x10 \x01(\rR\tcpu_count\x12,\n\x11\x63pu_count_logical\x18\x11 \x01(\rR\x11\x63pu_count_logical\x12\x15\n\x08gpu_type\x18\x12 \x01(\tR\x03gpu\x12\x1c\n\tgpu_count\x18\x13 \x01(\rR\tgpu_count\x12\x39\n\x04\x64isk\x18\x14 \x03(\x0b\x32+.wandb_internal.EnvironmentRecord.DiskEntry\x12*\n\x06memory\x18\x15 \x01(\x0b\x32\x1a.wandb_internal.MemoryInfo\x12$\n\x03\x63pu\x18\x16 \x01(\x0b\x32\x17.wandb_internal.CpuInfo\x12(\n\x05\x61pple\x18\x17 \x01(\x0b\x32\x19.wandb_internal.AppleInfo\x12=\n\ngpu_nvidia\x18\x18 \x03(\x0b\x32\x1d.wandb_internal.GpuNvidiaInfoR\ngpu_nvidia\x12\x14\n\x0c\x63uda_version\x18\x19 \x01(\t\x12\x34\n\x07gpu_amd\x18\x1a \x03(\x0b\x32\x1a.wandb_internal.GpuAmdInfoR\x07gpu_amd\x12;\n\x05slurm\x18\x1b \x03(\x0b\x32,.wandb_internal.EnvironmentRecord.SlurmEntry\x12.\n\x08trainium\x18\x1c \x01(\x0b\x32\x1c.wandb_internal.TrainiumInfo\x12$\n\x03tpu\x18\x1d \x01(\x0b\x32\x17.wandb_internal.TPUInfo\x12\x30\n\tcoreweave\x18\x1e \x01(\x0b\x32\x1d.wandb_internal.CoreWeaveInfo\x12\x12\n\twriter_id\x18\xc7\x01 \x01(\t\x12+\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1b.wandb_internal._RecordInfo\x1a\x45\n\tDiskEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\'\n\x05value\x18\x02 \x01(\x0b\x32\x18.wandb_internal.DiskInfo:\x02\x38\x01\x1a,\n\nSlurmEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t:\x02\x38\x01\"\x8d\x01\n\x15PythonPackagesRequest\x12\x44\n\x07package\x18\x01 \x03(\x0b\x32\x33.wandb_internal.PythonPackagesRequest.PythonPackage\x1a.\n\rPythonPackage\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x0f\n\x07version\x18\x02 \x01(\t\"\x1c\n\x0cJobInputPath\x12\x0c\n\x04path\x18\x01 \x03(\t\"\xd6\x01\n\x0eJobInputSource\x12\x44\n\nrun_config\x18\x01 \x01(\x0b\x32..wandb_internal.JobInputSource.RunConfigSourceH\x00\x12?\n\x04\x66ile\x18\x02 \x01(\x0b\x32/.wandb_internal.JobInputSource.ConfigFileSourceH\x00\x1a\x11\n\x0fRunConfigSource\x1a \n\x10\x43onfigFileSource\x12\x0c\n\x04path\x18\x01 \x01(\tB\x08\n\x06source\"\xc7\x01\n\x0fJobInputRequest\x12\x34\n\x0cinput_source\x18\x01 \x01(\x0b\x32\x1e.wandb_internal.JobInputSource\x12\x33\n\rinclude_paths\x18\x02 \x03(\x0b\x32\x1c.wandb_internal.JobInputPath\x12\x33\n\rexclude_paths\x18\x03 \x03(\x0b\x32\x1c.wandb_internal.JobInputPath\x12\x14\n\x0cinput_schema\x18\x04 \x01(\t*\xda\x05\n\rServerFeature\x12\x1e\n\x1aSERVER_FEATURE_UNSPECIFIED\x10\x00\x12\x13\n\x0fLARGE_FILENAMES\x10\x11\x12\x11\n\rARTIFACT_TAGS\x10\x01\x12\x0e\n\nCLIENT_IDS\x10\x02\x12\x1c\n\x18\x41RTIFACT_REGISTRY_SEARCH\x10\x03\x12\x1b\n\x17STRUCTURED_CONSOLE_LOGS\x10\x04\x12(\n$ARTIFACT_COLLECTION_MEMBERSHIP_FILES\x10\x05\x12\x38\n4ARTIFACT_COLLECTION_MEMBERSHIP_FILE_DOWNLOAD_HANDLER\x10\x06\x12\x34\n0USE_ARTIFACT_WITH_ENTITY_AND_PROJECT_INFORMATION\x10\x07\x12\x1f\n\x1b\x45XPAND_DEFINED_METRIC_GLOBS\x10\x08\x12\x1f\n\x1b\x41UTOMATION_EVENT_RUN_METRIC\x10\t\x12&\n\"AUTOMATION_EVENT_RUN_METRIC_CHANGE\x10\n\x12\x1b\n\x17\x41UTOMATION_ACTION_NO_OP\x10\x0b\x12/\n+INCLUDE_ARTIFACT_TYPES_IN_REGISTRY_CREATION\x10\x0c\x12*\n&PROJECT_ARTIFACT_COLLECTION_MEMBERSHIP\x10\r\x12\x31\n-ARTIFACT_MEMBERSHIP_IN_LINK_ARTIFACT_RESPONSE\x10\x0e\x12\"\n\x1eTOTAL_COUNT_IN_FILE_CONNECTION\x10\x0f\x12*\n&ARTIFACT_COLLECTIONS_FILTERING_SORTING\x10\x10\x12\x35\n1ARTIFACT_V2_DOWNLOAD_HANDLER_SUPPORTS_ARTIFACT_ID\x10\x12\x42\x1bZ\x19\x63ore/pkg/service_go_protob\x06proto3')
/n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/wandb/proto/v4/wandb_api_pb2.py:18:DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x1bwandb/proto/wandb_api.proto\x12\x0ewandb_internal\x1a wandb/proto/wandb_internal.proto\x1a wandb/proto/wandb_settings.proto\"B\n\x14ServerApiInitRequest\x12*\n\x08settings\x18\x01 \x01(\x0b\x32\x18.wandb_internal.Settings\">\n\x15ServerApiInitResponse\x12\x15\n\rerror_message\x18\x01 \x01(\t\x12\x0e\n\x06\x61pi_id\x18\x02 \x01(\t\"\xaf\x01\n\nApiRequest\x12\x0e\n\x06\x61pi_id\x18\x01 \x01(\t\x12I\n\x18read_run_history_request\x18\x02 \x01(\x0b\x32%.wandb_internal.ReadRunHistoryRequestH\x00\x12;\n\x10\x66\x65\x61tures_request\x18\x03 \x01(\x0b\x32\x1f.wandb_internal.FeaturesRequestH\x00\x42\t\n\x07request\"\xe5\x01\n\x0b\x41piResponse\x12K\n\x19read_run_history_response\x18\x01 \x01(\x0b\x32&.wandb_internal.ReadRunHistoryResponseH\x00\x12=\n\x11\x66\x65\x61tures_response\x18\x03 \x01(\x0b\x32 .wandb_internal.FeaturesResponseH\x00\x12>\n\x12\x61pi_error_response\x18\x02 \x01(\x0b\x32 .wandb_internal.ApiErrorResponseH\x00\x42\n\n\x08response\"f\n\x10\x41piErrorResponse\x12\x0f\n\x07message\x18\x01 \x01(\t\x12\x32\n\nerror_type\x18\x02 \x01(\x0e\x32\x19.wandb_internal.ErrorTypeH\x00\x88\x01\x01\x42\r\n\x0b_error_type\")\n\x17ServerApiCleanupRequest\x12\x0e\n\x06\x61pi_id\x18\x01 \x01(\t\"B\n\x0f\x46\x65\x61turesRequest\x12/\n\x08\x66\x65\x61tures\x18\x01 \x03(\x0e\x32\x1d.wandb_internal.ServerFeature\"B\n\x10\x46\x65\x61turesResponse\x12.\n\x07\x65nabled\x18\x01 \x03(\x0e\x32\x1d.wandb_internal.ServerFeature\"\xd0\x03\n\x15ReadRunHistoryRequest\x12\x43\n\x15scan_run_history_init\x18\x01 \x01(\x0b\x32\".wandb_internal.ScanRunHistoryInitH\x00\x12:\n\x10scan_run_history\x18\x02 \x01(\x0b\x32\x1e.wandb_internal.ScanRunHistoryH\x00\x12I\n\x18scan_run_history_cleanup\x18\x03 \x01(\x0b\x32%.wandb_internal.ScanRunHistoryCleanupH\x00\x12K\n\x19\x64ownload_run_history_init\x18\x04 \x01(\x0b\x32&.wandb_internal.DownloadRunHistoryInitH\x00\x12\x42\n\x14\x64ownload_run_history\x18\x05 \x01(\x0b\x32\".wandb_internal.DownloadRunHistoryH\x00\x12O\n\x1b\x64ownload_run_history_status\x18\x06 \x01(\x0b\x32(.wandb_internal.DownloadRunHistoryStatusH\x00\x42\t\n\x07request\"\xf9\x03\n\x16ReadRunHistoryResponse\x12K\n\x15scan_run_history_init\x18\x01 \x01(\x0b\x32*.wandb_internal.ScanRunHistoryInitResponseH\x00\x12\x39\n\x0brun_history\x18\x02 \x01(\x0b\x32\".wandb_internal.RunHistoryResponseH\x00\x12Q\n\x18scan_run_history_cleanup\x18\x03 \x01(\x0b\x32-.wandb_internal.ScanRunHistoryCleanupResponseH\x00\x12S\n\x19\x64ownload_run_history_init\x18\x04 \x01(\x0b\x32..wandb_internal.DownloadRunHistoryInitResponseH\x00\x12J\n\x14\x64ownload_run_history\x18\x05 \x01(\x0b\x32*.wandb_internal.DownloadRunHistoryResponseH\x00\x12W\n\x1b\x64ownload_run_history_status\x18\x06 \x01(\x0b\x32\x30.wandb_internal.DownloadRunHistoryStatusResponseH\x00\x42\n\n\x08response\"f\n\x12ScanRunHistoryInit\x12\x0e\n\x06\x65ntity\x18\x01 \x01(\t\x12\x0f\n\x07project\x18\x02 \x01(\t\x12\x0e\n\x06run_id\x18\x03 \x01(\t\x12\x0c\n\x04keys\x18\x04 \x03(\t\x12\x11\n\tuse_cache\x18\x05 \x01(\x08\"0\n\x1aScanRunHistoryInitResponse\x12\x12\n\nrequest_id\x18\x01 \x01(\x05\"H\n\x0eScanRunHistory\x12\x10\n\x08min_step\x18\x01 \x01(\x03\x12\x10\n\x08max_step\x18\x02 \x01(\x03\x12\x12\n\nrequest_id\x18\x03 \x01(\x05\"F\n\x12RunHistoryResponse\x12\x30\n\x0chistory_rows\x18\x01 \x03(\x0b\x32\x1a.wandb_internal.HistoryRow\"G\n\nHistoryRow\x12\x39\n\rhistory_items\x18\x01 \x03(\x0b\x32\".wandb_internal.ParquetHistoryItem\"5\n\x12ParquetHistoryItem\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\x12\n\nvalue_json\x18\x10 \x01(\t\"+\n\x15ScanRunHistoryCleanup\x12\x12\n\nrequest_id\x18\x01 \x01(\x05\"\x1f\n\x1dScanRunHistoryCleanupResponse\"\x81\x01\n\x16\x44ownloadRunHistoryInit\x12\x0e\n\x06\x65ntity\x18\x01 \x01(\t\x12\x0f\n\x07project\x18\x02 \x01(\t\x12\x0e\n\x06run_id\x18\x03 \x01(\t\x12\x14\n\x0c\x64ownload_dir\x18\x04 \x01(\t\x12 \n\x18require_complete_history\x18\x05 \x01(\x08\"P\n\x1e\x44ownloadRunHistoryInitResponse\x12\x12\n\nrequest_id\x18\x01 \x01(\x05\x12\x1a\n\x12\x63ontains_live_data\x18\x02 \x01(\x08\"(\n\x12\x44ownloadRunHistory\x12\x12\n\nrequest_id\x18\x01 \x01(\x05\"\xad\x01\n\x1a\x44ownloadRunHistoryResponse\x12\x18\n\x10\x64ownloaded_files\x18\x01 \x03(\t\x12\x46\n\x06\x65rrors\x18\x02 \x03(\x0b\x32\x36.wandb_internal.DownloadRunHistoryResponse.ErrorsEntry\x1a-\n\x0b\x45rrorsEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t:\x02\x38\x01\"\x1b\n\x19IncompleteRunHistoryError\".\n\x18\x44ownloadRunHistoryStatus\x12\x12\n\nrequest_id\x18\x01 \x01(\x05\"[\n DownloadRunHistoryStatusResponse\x12\x37\n\x0foperation_stats\x18\x01 \x01(\x0b\x32\x1e.wandb_internal.OperationStats*@\n\tErrorType\x12\x11\n\rUNKNOWN_ERROR\x10\x00\x12 \n\x1cINCOMPLETE_RUN_HISTORY_ERROR\x10\x01\x42\x1bZ\x19\x63ore/pkg/service_go_protob\x06proto3')
/n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/wandb/proto/v4/wandb_settings_pb2.py:17:DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n wandb/proto/wandb_settings.proto\x12\x0ewandb_internal\x1a\x1egoogle/protobuf/wrappers.proto\" \n\x0fListStringValue\x12\r\n\x05value\x18\x01 \x03(\t\"\x1d\n\x0cListIntValue\x12\r\n\x05value\x18\x01 \x03(\x05\"\x8a\x01\n\x17MapStringKeyStringValue\x12\x41\n\x05value\x18\x01 \x03(\x0b\x32\x32.wandb_internal.MapStringKeyStringValue.ValueEntry\x1a,\n\nValueEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t:\x02\x38\x01\"\xcb\x01\n#MapStringKeyMapStringKeyStringValue\x12M\n\x05value\x18\x01 \x03(\x0b\x32>.wandb_internal.MapStringKeyMapStringKeyStringValue.ValueEntry\x1aU\n\nValueEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\x36\n\x05value\x18\x02 \x01(\x0b\x32\'.wandb_internal.MapStringKeyStringValue:\x02\x38\x01\"\x9a\x01\n\x12OpenMetricsFilters\x12\x33\n\x08sequence\x18\x01 \x01(\x0b\x32\x1f.wandb_internal.ListStringValueH\x00\x12\x46\n\x07mapping\x18\x02 \x01(\x0b\x32\x33.wandb_internal.MapStringKeyMapStringKeyStringValueH\x00\x42\x07\n\x05value\"7\n\tRunMoment\x12\x0b\n\x03run\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\x01\x12\x0e\n\x06metric\x18\x03 \x01(\t\"\xbeO\n\x08Settings\x12-\n\x07\x61pi_key\x18\x37 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12:\n\x13identity_token_file\x18\xaa\x01 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12\x37\n\x10\x63redentials_file\x18\xab\x01 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12\x39\n\x14insecure_disable_ssl\x18\xb9\x01 \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12,\n\x08_offline\x18\x1e \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12*\n\x06x_sync\x18\x1f \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12\x30\n\tsync_file\x18\x86\x01 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12,\n\x07_shared\x18\xa2\x01 \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12,\n\x06run_id\x18k \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12-\n\x07run_url\x18q \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12-\n\x07project\x18\x61 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12,\n\x06\x65ntity\x18\x45 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12\x33\n\x0corganization\x18\xbc\x01 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12\x32\n\x0cx_start_time\x18) \x01(\x0b\x32\x1c.google.protobuf.DoubleValue\x12.\n\x08root_dir\x18i \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12\x30\n\twandb_dir\x18\x8e\x01 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12-\n\x07log_dir\x18U \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12\x32\n\x0clog_internal\x18V \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12\x35\n\x0cignore_globs\x18N \x01(\x0b\x32\x1f.wandb_internal.ListStringValue\x12.\n\x07\x61pp_url\x18\xca\x01 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12.\n\x08\x62\x61se_url\x18\x39 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12=\n\x17x_file_stream_max_bytes\x18\xac\x01 \x01(\x0b\x32\x1b.google.protobuf.Int32Value\x12\x46\n\x1fx_file_stream_transmit_interval\x18\xaf\x01 \x01(\x0b\x32\x1c.google.protobuf.DoubleValue\x12\x45\n\x14x_extra_http_headers\x18\x0e \x01(\x0b\x32\'.wandb_internal.MapStringKeyStringValue\x12=\n\x17x_file_stream_retry_max\x18\x93\x01 \x01(\x0b\x32\x1b.google.protobuf.Int32Value\x12K\n$x_file_stream_retry_wait_min_seconds\x18\x94\x01 \x01(\x0b\x32\x1c.google.protobuf.DoubleValue\x12K\n$x_file_stream_retry_wait_max_seconds\x18\x95\x01 \x01(\x0b\x32\x1c.google.protobuf.DoubleValue\x12\x43\n\x1dx_file_stream_timeout_seconds\x18\x0f \x01(\x0b\x32\x1c.google.protobuf.DoubleValue\x12\x42\n\x1cx_file_stream_max_line_bytes\x18\xb2\x01 \x01(\x0b\x32\x1b.google.protobuf.Int32Value\x12?\n\x19x_file_transfer_retry_max\x18\x96\x01 \x01(\x0b\x32\x1b.google.protobuf.Int32Value\x12M\n&x_file_transfer_retry_wait_min_seconds\x18\x97\x01 \x01(\x0b\x32\x1c.google.protobuf.DoubleValue\x12M\n&x_file_transfer_retry_wait_max_seconds\x18\x98\x01 \x01(\x0b\x32\x1c.google.protobuf.DoubleValue\x12\x46\n\x1fx_file_transfer_timeout_seconds\x18\x99\x01 \x01(\x0b\x32\x1c.google.protobuf.DoubleValue\x12\x39\n\x13x_graphql_retry_max\x18\x9a\x01 \x01(\x0b\x32\x1b.google.protobuf.Int32Value\x12G\n x_graphql_retry_wait_min_seconds\x18\x9b\x01 \x01(\x0b\x32\x1c.google.protobuf.DoubleValue\x12G\n x_graphql_retry_wait_max_seconds\x18\x9c\x01 \x01(\x0b\x32\x1c.google.protobuf.DoubleValue\x12@\n\x19x_graphql_timeout_seconds\x18\x9d\x01 \x01(\x0b\x32\x1c.google.protobuf.DoubleValue\x12\x31\n\nhttp_proxy\x18\xa8\x01 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12\x32\n\x0bhttps_proxy\x18\xa9\x01 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12;\n\tx_proxies\x18\xc8\x01 \x01(\x0b\x32\'.wandb_internal.MapStringKeyStringValue\x12-\n\x07program\x18_ \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12\x35\n\x0fprogram_relpath\x18` \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12\x37\n\x10_code_path_local\x18\xa3\x01 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12\x36\n\x0fprogram_abspath\x18\x9f\x01 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12.\n\x05_args\x18\x01 \x01(\x0b\x32\x1f.wandb_internal.ListStringValue\x12)\n\x03_os\x18  \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12,\n\x06\x64ocker\x18\x43 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12\x32\n\x0cx_executable\x18\r \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12-\n\x07_python\x18\" \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12\x30\n\tcolab_url\x18\xa0\x01 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12*\n\x04host\x18M \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12/\n\x08username\x18\x8d\x01 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12+\n\x05\x65mail\x18\x44 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12,\n\x06resume\x18\x66 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12/\n\x0bresume_from\x18\xa7\x01 \x01(\x0b\x32\x19.wandb_internal.RunMoment\x12-\n\tfork_from\x18\xa4\x01 \x01(\x0b\x32\x19.wandb_internal.RunMoment\x12\x38\n\x14\x64isable_job_creation\x18\x41 \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12\x30\n\tsweep_url\x18\x83\x01 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12;\n\x16x_disable_update_check\x18\xa5\x01 \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12\x32\n\x0ex_disable_meta\x18\x07 \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12-\n\tsave_code\x18s \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12/\n\x0b\x64isable_git\x18? \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12;\n\x16\x64isable_git_fork_point\x18\xcb\x01 \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12;\n\x16x_disable_machine_info\x18\x9e\x01 \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12\x33\n\x0fx_disable_stats\x18\n \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12\x39\n\x13x_stats_buffer_size\x18\xa1\x01 \x01(\x0b\x32\x1b.google.protobuf.Int32Value\x12@\n\x19x_stats_sampling_interval\x18\xae\x01 \x01(\x0b\x32\x1c.google.protobuf.DoubleValue\x12\x30\n\x0bx_stats_pid\x18* \x01(\x0b\x32\x1b.google.protobuf.Int32Value\x12<\n\x12x_stats_disk_paths\x18\x92\x01 \x01(\x0b\x32\x1f.wandb_internal.ListStringValue\x12H\n\"x_stats_neuron_monitor_config_path\x18. \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12<\n\x15x_stats_dcgm_exporter\x18\xbb\x01 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12O\n\x1ex_stats_open_metrics_endpoints\x18/ \x01(\x0b\x32\'.wandb_internal.MapStringKeyStringValue\x12H\n\x1cx_stats_open_metrics_filters\x18\x30 \x01(\x0b\x32\".wandb_internal.OpenMetricsFilters\x12S\n!x_stats_open_metrics_http_headers\x18\xb8\x01 \x01(\x0b\x32\'.wandb_internal.MapStringKeyStringValue\x12=\n\x16x_stats_gpu_device_ids\x18\xba\x01 \x01(\x0b\x32\x1c.wandb_internal.ListIntValue\x12\x37\n\x11x_stats_cpu_count\x18\xc2\x01 \x01(\x0b\x32\x1b.google.protobuf.Int32Value\x12?\n\x19x_stats_cpu_logical_count\x18\xc3\x01 \x01(\x0b\x32\x1b.google.protobuf.Int32Value\x12\x37\n\x11x_stats_gpu_count\x18\xc4\x01 \x01(\x0b\x32\x1b.google.protobuf.Int32Value\x12\x37\n\x10x_stats_gpu_type\x18\xc5\x01 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12?\n\x1ax_stats_track_process_tree\x18\xc6\x01 \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12.\n\x07x_label\x18\xb5\x01 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12.\n\tx_primary\x18\xb6\x01 \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12:\n\x15x_update_finish_state\x18\xb7\x01 \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12<\n\x17\x61llow_offline_artifacts\x18\xb1\x01 \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12-\n\x07\x63onsole\x18< \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12\x36\n\x11\x63onsole_multipart\x18\xa6\x01 \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12=\n\x17\x63onsole_chunk_max_bytes\x18\xc7\x01 \x01(\x0b\x32\x1b.google.protobuf.Int32Value\x12?\n\x19\x63onsole_chunk_max_seconds\x18\xc9\x01 \x01(\x0b\x32\x1b.google.protobuf.Int32Value\x12\x35\n\x10sync_tensorboard\x18\xb3\x01 \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12\x42\n\x1dx_server_side_derived_summary\x18\xbd\x01 \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12\x46\n!x_server_side_expand_glob_metrics\x18\xbe\x01 \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12;\n\x16x_skip_transaction_log\x18\xbf\x01 \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12J\n#x_stats_coreweave_metadata_base_url\x18\xc0\x01 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12J\n#x_stats_coreweave_metadata_endpoint\x18\xc1\x01 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12/\n\x0b_aws_lambda\x18\x02 \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12\x33\n\x0fx_cli_only_mode\x18\x04 \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12*\n\x06_colab\x18\x05 \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12\x34\n\x10x_disable_viewer\x18\x0b \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12\x39\n\x15x_flow_control_custom\x18\x10 \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12;\n\x17x_flow_control_disabled\x18\x11 \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12>\n\x18x_internal_check_process\x18\x12 \x01(\x0b\x32\x1c.google.protobuf.DoubleValue\x12,\n\x08_ipython\x18\x14 \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12,\n\x08_jupyter\x18\x15 \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12\x34\n\x0ex_jupyter_root\x18\x16 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12+\n\x07_kaggle\x18\x17 \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12=\n\x18x_live_policy_rate_limit\x18\x18 \x01(\x0b\x32\x1b.google.protobuf.Int32Value\x12<\n\x17x_live_policy_wait_time\x18\x19 \x01(\x0b\x32\x1b.google.protobuf.Int32Value\x12\x30\n\x0bx_log_level\x18\x1a \x01(\x0b\x32\x1b.google.protobuf.Int32Value\x12\x35\n\x10x_network_buffer\x18\x1b \x01(\x0b\x32\x1b.google.protobuf.Int32Value\x12)\n\x05_noop\x18\x1c \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12-\n\t_notebook\x18\x1d \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12/\n\t_platform\x18! \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12\x38\n\x12x_runqueue_item_id\x18# \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12\x37\n\x13x_save_requirements\x18% \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12\x39\n\x13x_service_transport\x18& \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12\x34\n\x0ex_service_wait\x18\' \x01(\x0b\x32\x1c.google.protobuf.DoubleValue\x12\x35\n\x0f_start_datetime\x18( \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12\x33\n\r_tmp_code_dir\x18\x31 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12,\n\x08_windows\x18\x34 \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12\x38\n\x13\x61llow_media_symlink\x18\xcc\x01 \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12\x34\n\x10\x61llow_val_change\x18\x35 \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12P\n\x1f\x61zure_account_url_to_access_key\x18\x38 \x01(\x0b\x32\'.wandb_internal.MapStringKeyStringValue\x12.\n\x08\x63ode_dir\x18: \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12\x35\n\x0c\x63onfig_paths\x18; \x01(\x0b\x32\x1f.wandb_internal.ListStringValue\x12\x30\n\ndeployment\x18= \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12\x30\n\x0c\x64isable_code\x18> \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12\x31\n\rdisable_hints\x18@ \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12,\n\x08\x64isabled\x18\x42 \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12)\n\x05\x66orce\x18G \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12\x30\n\ngit_commit\x18H \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12\x30\n\ngit_remote\x18I \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12\x34\n\x0egit_remote_url\x18J \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12.\n\x08git_root\x18K \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12\x36\n\x11heartbeat_seconds\x18L \x01(\x0b\x32\x1b.google.protobuf.Int32Value\x12\x32\n\x0cinit_timeout\x18O \x01(\x0b\x32\x1c.google.protobuf.DoubleValue\x12,\n\x08is_local\x18P \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12\x30\n\njob_source\x18Q \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12\x31\n\rlabel_disable\x18R \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12*\n\x06launch\x18S \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12\x38\n\x12launch_config_path\x18T \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12:\n\x14log_symlink_internal\x18W \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12\x36\n\x10log_symlink_user\x18X \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12.\n\x08log_user\x18Y \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12\x33\n\rlogin_timeout\x18Z \x01(\x0b\x32\x1c.google.protobuf.DoubleValue\x12*\n\x04mode\x18\\ \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12\x33\n\rnotebook_name\x18] \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12\x31\n\x0bproject_url\x18\x62 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12)\n\x05quiet\x18\x63 \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12+\n\x07relogin\x18\x65 \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12\x32\n\x0cresume_fname\x18g \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12+\n\x07resumed\x18h \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12/\n\trun_group\x18j \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12\x32\n\x0crun_job_type\x18l \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12.\n\x08run_mode\x18m \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12.\n\x08run_name\x18n \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12/\n\trun_notes\x18o \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12\x31\n\x08run_tags\x18p \x01(\x0b\x32\x1f.wandb_internal.ListStringValue\x12\x35\n\x11sagemaker_disable\x18r \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12\x35\n\x0fsettings_system\x18t \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12\x38\n\x12settings_workspace\x18u \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12/\n\x0bshow_colors\x18v \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12.\n\nshow_emoji\x18w \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12/\n\x0bshow_errors\x18x \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12-\n\tshow_info\x18y \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12\x31\n\rshow_warnings\x18z \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12*\n\x06silent\x18{ \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12\x32\n\x0cstart_method\x18| \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12*\n\x06strict\x18} \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12\x33\n\x0esummary_errors\x18~ \x01(\x0b\x32\x1b.google.protobuf.Int32Value\x12\x34\n\x0fsummary_timeout\x18\x7f \x01(\x0b\x32\x1b.google.protobuf.Int32Value\x12\x36\n\x10summary_warnings\x18\x80\x01 \x01(\x0b\x32\x1b.google.protobuf.Int32Value\x12/\n\x08sweep_id\x18\x81\x01 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12\x37\n\x10sweep_param_path\x18\x82\x01 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12,\n\x07symlink\x18\x84\x01 \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12/\n\x08sync_dir\x18\x85\x01 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12:\n\x13sync_symlink_latest\x18\x87\x01 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12J\n%table_raise_on_max_row_limit_exceeded\x18\x8a\x01 \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12/\n\x08timespec\x18\x8b\x01 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12.\n\x07tmp_dir\x18\x8c\x01 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12\x35\n\x0ex_jupyter_name\x18\x8f\x01 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12\x35\n\x0ex_jupyter_path\x18\x90\x01 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12/\n\x08job_name\x18\x91\x01 \x01(\x0b\x32\x1c.google.protobuf.StringValueJ\x04\x08\x03\x10\x04J\x04\x08\x06\x10\x07J\x04\x08\x08\x10\tJ\x04\x08\t\x10\nJ\x04\x08\x0c\x10\rJ\x04\x08\x13\x10\x14J\x04\x08$\x10%J\x04\x08+\x10,J\x04\x08,\x10-J\x04\x08-\x10.J\x04\x08\x32\x10\x33J\x04\x08\x33\x10\x34J\x04\x08\x36\x10\x37J\x04\x08\x46\x10GJ\x04\x08[\x10\\J\x04\x08^\x10_J\x04\x08\x64\x10\x65J\x06\x08\x88\x01\x10\x89\x01J\x06\x08\x89\x01\x10\x8a\x01J\x06\x08\xad\x01\x10\xae\x01J\x06\x08\xb0\x01\x10\xb1\x01J\x06\x08\xb4\x01\x10\xb5\x01\x42\x1bZ\x19\x63ore/pkg/service_go_protob\x06proto3')
/n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/wandb/proto/v4/wandb_internal_pb2.py:20:DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n wandb/proto/wandb_internal.proto\x12\x0ewandb_internal\x1a\x1bgoogle/protobuf/empty.proto\x1a\x1fgoogle/protobuf/timestamp.proto\x1a\x1cwandb/proto/wandb_base.proto\x1a!wandb/proto/wandb_telemetry.proto\"\xcf\t\n\x06Record\x12\x0b\n\x03num\x18\x01 \x01(\x03\x12\x30\n\x07history\x18\x02 \x01(\x0b\x32\x1d.wandb_internal.HistoryRecordH\x00\x12\x30\n\x07summary\x18\x03 \x01(\x0b\x32\x1d.wandb_internal.SummaryRecordH\x00\x12.\n\x06output\x18\x04 \x01(\x0b\x32\x1c.wandb_internal.OutputRecordH\x00\x12.\n\x06\x63onfig\x18\x05 \x01(\x0b\x32\x1c.wandb_internal.ConfigRecordH\x00\x12,\n\x05\x66iles\x18\x06 \x01(\x0b\x32\x1b.wandb_internal.FilesRecordH\x00\x12,\n\x05stats\x18\x07 \x01(\x0b\x32\x1b.wandb_internal.StatsRecordH\x00\x12\x32\n\x08\x61rtifact\x18\x08 \x01(\x0b\x32\x1e.wandb_internal.ArtifactRecordH\x00\x12,\n\x08tbrecord\x18\t \x01(\x0b\x32\x18.wandb_internal.TBRecordH\x00\x12,\n\x05\x61lert\x18\n \x01(\x0b\x32\x1b.wandb_internal.AlertRecordH\x00\x12\x34\n\ttelemetry\x18\x0b \x01(\x0b\x32\x1f.wandb_internal.TelemetryRecordH\x00\x12.\n\x06metric\x18\x0c \x01(\x0b\x32\x1c.wandb_internal.MetricRecordH\x00\x12\x35\n\noutput_raw\x18\r \x01(\x0b\x32\x1f.wandb_internal.OutputRawRecordH\x00\x12(\n\x03run\x18\x11 \x01(\x0b\x32\x19.wandb_internal.RunRecordH\x00\x12-\n\x04\x65xit\x18\x12 \x01(\x0b\x32\x1d.wandb_internal.RunExitRecordH\x00\x12,\n\x05\x66inal\x18\x14 \x01(\x0b\x32\x1b.wandb_internal.FinalRecordH\x00\x12.\n\x06header\x18\x15 \x01(\x0b\x32\x1c.wandb_internal.HeaderRecordH\x00\x12.\n\x06\x66ooter\x18\x16 \x01(\x0b\x32\x1c.wandb_internal.FooterRecordH\x00\x12\x39\n\npreempting\x18\x17 \x01(\x0b\x32#.wandb_internal.RunPreemptingRecordH\x00\x12\x34\n\x12noop_link_artifact\x18\x18 \x01(\x0b\x32\x16.google.protobuf.EmptyH\x00\x12\x39\n\x0cuse_artifact\x18\x19 \x01(\x0b\x32!.wandb_internal.UseArtifactRecordH\x00\x12\x38\n\x0b\x65nvironment\x18\x1a \x01(\x0b\x32!.wandb_internal.EnvironmentRecordH\x00\x12*\n\x07request\x18\x64 \x01(\x0b\x32\x17.wandb_internal.RequestH\x00\x12(\n\x07\x63ontrol\x18\x10 \x01(\x0b\x32\x17.wandb_internal.Control\x12\x0c\n\x04uuid\x18\x13 \x01(\t\x12+\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1b.wandb_internal._RecordInfoB\r\n\x0brecord_type\"\xa8\x01\n\x07\x43ontrol\x12\x10\n\x08req_resp\x18\x01 \x01(\x08\x12\r\n\x05local\x18\x02 \x01(\x08\x12\x10\n\x08relay_id\x18\x03 \x01(\t\x12\x14\n\x0cmailbox_slot\x18\x04 \x01(\t\x12\x13\n\x0b\x61lways_send\x18\x05 \x01(\x08\x12\x14\n\x0c\x66low_control\x18\x06 \x01(\x08\x12\x12\n\nend_offset\x18\x07 \x01(\x03\x12\x15\n\rconnection_id\x18\x08 \x01(\t\"\xf3\x03\n\x06Result\x12\x35\n\nrun_result\x18\x11 \x01(\x0b\x32\x1f.wandb_internal.RunUpdateResultH\x00\x12\x34\n\x0b\x65xit_result\x18\x12 \x01(\x0b\x32\x1d.wandb_internal.RunExitResultH\x00\x12\x33\n\nlog_result\x18\x14 \x01(\x0b\x32\x1d.wandb_internal.HistoryResultH\x00\x12\x37\n\x0esummary_result\x18\x15 \x01(\x0b\x32\x1d.wandb_internal.SummaryResultH\x00\x12\x35\n\routput_result\x18\x16 \x01(\x0b\x32\x1c.wandb_internal.OutputResultH\x00\x12\x35\n\rconfig_result\x18\x17 \x01(\x0b\x32\x1c.wandb_internal.ConfigResultH\x00\x12,\n\x08response\x18\x64 \x01(\x0b\x32\x18.wandb_internal.ResponseH\x00\x12(\n\x07\x63ontrol\x18\x10 \x01(\x0b\x32\x17.wandb_internal.Control\x12\x0c\n\x04uuid\x18\x18 \x01(\t\x12+\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1b.wandb_internal._ResultInfoB\r\n\x0bresult_type\":\n\x0b\x46inalRecord\x12+\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1b.wandb_internal._RecordInfo\"b\n\x0bVersionInfo\x12\x10\n\x08producer\x18\x01 \x01(\t\x12\x14\n\x0cmin_consumer\x18\x02 \x01(\t\x12+\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1b.wandb_internal._RecordInfo\"n\n\x0cHeaderRecord\x12\x31\n\x0cversion_info\x18\x01 \x01(\x0b\x32\x1b.wandb_internal.VersionInfo\x12+\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1b.wandb_internal._RecordInfo\";\n\x0c\x46ooterRecord\x12+\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1b.wandb_internal._RecordInfo\"9\n\x0b\x42ranchPoint\x12\x0b\n\x03run\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\x01\x12\x0e\n\x06metric\x18\x03 \x01(\t\"\x91\x05\n\tRunRecord\x12\x0e\n\x06run_id\x18\x01 \x01(\t\x12\x0e\n\x06\x65ntity\x18\x02 \x01(\t\x12\x0f\n\x07project\x18\x03 \x01(\t\x12,\n\x06\x63onfig\x18\x04 \x01(\x0b\x32\x1c.wandb_internal.ConfigRecord\x12.\n\x07summary\x18\x05 \x01(\x0b\x32\x1d.wandb_internal.SummaryRecord\x12\x11\n\trun_group\x18\x06 \x01(\t\x12\x10\n\x08job_type\x18\x07 \x01(\t\x12\x14\n\x0c\x64isplay_name\x18\x08 \x01(\t\x12\r\n\x05notes\x18\t \x01(\t\x12\x0c\n\x04tags\x18\n \x03(\t\x12\x30\n\x08settings\x18\x0b \x01(\x0b\x32\x1e.wandb_internal.SettingsRecord\x12\x10\n\x08sweep_id\x18\x0c \x01(\t\x12\x0c\n\x04host\x18\r \x01(\t\x12\x15\n\rstarting_step\x18\x0e \x01(\x03\x12\x12\n\nstorage_id\x18\x10 \x01(\t\x12.\n\nstart_time\x18\x11 \x01(\x0b\x32\x1a.google.protobuf.Timestamp\x12\x0f\n\x07resumed\x18\x12 \x01(\x08\x12\x32\n\ttelemetry\x18\x13 \x01(\x0b\x32\x1f.wandb_internal.TelemetryRecord\x12\x0f\n\x07runtime\x18\x14 \x01(\x05\x12*\n\x03git\x18\x15 \x01(\x0b\x32\x1d.wandb_internal.GitRepoRecord\x12\x0e\n\x06\x66orked\x18\x16 \x01(\x08\x12\x31\n\x0c\x62ranch_point\x18\x17 \x01(\x0b\x32\x1b.wandb_internal.BranchPoint\x12+\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1b.wandb_internal._RecordInfo\";\n\rGitRepoRecord\x12\x1a\n\nremote_url\x18\x01 \x01(\tR\x06remote\x12\x0e\n\x06\x63ommit\x18\x02 \x01(\t\"c\n\x0fRunUpdateResult\x12&\n\x03run\x18\x01 \x01(\x0b\x32\x19.wandb_internal.RunRecord\x12(\n\x05\x65rror\x18\x02 \x01(\x0b\x32\x19.wandb_internal.ErrorInfo\"\xac\x01\n\tErrorInfo\x12\x0f\n\x07message\x18\x01 \x01(\t\x12\x31\n\x04\x63ode\x18\x02 \x01(\x0e\x32#.wandb_internal.ErrorInfo.ErrorCode\"[\n\tErrorCode\x12\x0b\n\x07UNKNOWN\x10\x00\x12\x11\n\rCOMMUNICATION\x10\x01\x12\x12\n\x0e\x41UTHENTICATION\x10\x02\x12\t\n\x05USAGE\x10\x03\x12\x0f\n\x0bUNSUPPORTED\x10\x04\"v\n\rRunExitRecord\x12\x11\n\texit_code\x18\x01 \x01(\x05\x12\x14\n\x0cnot_complete\x18\x03 \x01(\x08\x12\x0f\n\x07runtime\x18\x02 \x01(\x05\x12+\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1b.wandb_internal._RecordInfo\"\x0f\n\rRunExitResult\"B\n\x13RunPreemptingRecord\x12+\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1b.wandb_internal._RecordInfo\"\x15\n\x13RunPreemptingResult\"i\n\x0eSettingsRecord\x12*\n\x04item\x18\x01 \x03(\x0b\x32\x1c.wandb_internal.SettingsItem\x12+\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1b.wandb_internal._RecordInfo\"/\n\x0cSettingsItem\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\x12\n\nvalue_json\x18\x10 \x01(\t\"\x1a\n\x0bHistoryStep\x12\x0b\n\x03num\x18\x01 \x01(\x03\"\x92\x01\n\rHistoryRecord\x12)\n\x04item\x18\x01 \x03(\x0b\x32\x1b.wandb_internal.HistoryItem\x12)\n\x04step\x18\x02 \x01(\x0b\x32\x1b.wandb_internal.HistoryStep\x12+\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1b.wandb_internal._RecordInfo\"B\n\x0bHistoryItem\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\x12\n\nnested_key\x18\x02 \x03(\t\x12\x12\n\nvalue_json\x18\x10 \x01(\t\"\x0f\n\rHistoryResult\"\xdc\x01\n\x0cOutputRecord\x12<\n\x0boutput_type\x18\x01 \x01(\x0e\x32\'.wandb_internal.OutputRecord.OutputType\x12-\n\ttimestamp\x18\x02 \x01(\x0b\x32\x1a.google.protobuf.Timestamp\x12\x0c\n\x04line\x18\x03 \x01(\t\x12+\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1b.wandb_internal._RecordInfo\"$\n\nOutputType\x12\n\n\x06STDERR\x10\x00\x12\n\n\x06STDOUT\x10\x01\"\x0e\n\x0cOutputResult\"\xe2\x01\n\x0fOutputRawRecord\x12?\n\x0boutput_type\x18\x01 \x01(\x0e\x32*.wandb_internal.OutputRawRecord.OutputType\x12-\n\ttimestamp\x18\x02 \x01(\x0b\x32\x1a.google.protobuf.Timestamp\x12\x0c\n\x04line\x18\x03 \x01(\t\x12+\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1b.wandb_internal._RecordInfo\"$\n\nOutputType\x12\n\n\x06STDERR\x10\x00\x12\n\n\x06STDOUT\x10\x01\"\x11\n\x0fOutputRawResult\"\xb4\x03\n\x0cMetricRecord\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x11\n\tglob_name\x18\x02 \x01(\t\x12\x13\n\x0bstep_metric\x18\x04 \x01(\t\x12\x19\n\x11step_metric_index\x18\x05 \x01(\x05\x12.\n\x07options\x18\x06 \x01(\x0b\x32\x1d.wandb_internal.MetricOptions\x12.\n\x07summary\x18\x07 \x01(\x0b\x32\x1d.wandb_internal.MetricSummary\x12\x35\n\x04goal\x18\x08 \x01(\x0e\x32\'.wandb_internal.MetricRecord.MetricGoal\x12/\n\x08_control\x18\t \x01(\x0b\x32\x1d.wandb_internal.MetricControl\x12\x1a\n\x12\x65xpanded_from_glob\x18\n \x01(\x08\x12+\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1b.wandb_internal._RecordInfo\"B\n\nMetricGoal\x12\x0e\n\nGOAL_UNSET\x10\x00\x12\x11\n\rGOAL_MINIMIZE\x10\x01\x12\x11\n\rGOAL_MAXIMIZE\x10\x02\"\x0e\n\x0cMetricResult\"C\n\rMetricOptions\x12\x11\n\tstep_sync\x18\x01 \x01(\x08\x12\x0e\n\x06hidden\x18\x02 \x01(\x08\x12\x0f\n\x07\x64\x65\x66ined\x18\x03 \x01(\x08\"\"\n\rMetricControl\x12\x11\n\toverwrite\x18\x01 \x01(\x08\"~\n\rMetricSummary\x12\x0b\n\x03min\x18\x01 \x01(\x08\x12\x0b\n\x03max\x18\x02 \x01(\x08\x12\x0c\n\x04mean\x18\x03 \x01(\x08\x12\x0c\n\x04\x62\x65st\x18\x04 \x01(\x08\x12\x0c\n\x04last\x18\x05 \x01(\x08\x12\x0c\n\x04none\x18\x06 \x01(\x08\x12\x0c\n\x04\x63opy\x18\x07 \x01(\x08\x12\r\n\x05\x66irst\x18\x08 \x01(\x08\"\x93\x01\n\x0c\x43onfigRecord\x12*\n\x06update\x18\x01 \x03(\x0b\x32\x1a.wandb_internal.ConfigItem\x12*\n\x06remove\x18\x02 \x03(\x0b\x32\x1a.wandb_internal.ConfigItem\x12+\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1b.wandb_internal._RecordInfo\"A\n\nConfigItem\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\x12\n\nnested_key\x18\x02 \x03(\t\x12\x12\n\nvalue_json\x18\x10 \x01(\t\"\x0e\n\x0c\x43onfigResult\"\x96\x01\n\rSummaryRecord\x12+\n\x06update\x18\x01 \x03(\x0b\x32\x1b.wandb_internal.SummaryItem\x12+\n\x06remove\x18\x02 \x03(\x0b\x32\x1b.wandb_internal.SummaryItem\x12+\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1b.wandb_internal._RecordInfo\"B\n\x0bSummaryItem\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\x12\n\nnested_key\x18\x02 \x03(\t\x12\x12\n\nvalue_json\x18\x10 \x01(\t\"\x0f\n\rSummaryResult\"d\n\x0b\x46ilesRecord\x12(\n\x05\x66iles\x18\x01 \x03(\x0b\x32\x19.wandb_internal.FilesItem\x12+\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1b.wandb_internal._RecordInfo\"\xec\x01\n\tFilesItem\x12\x0c\n\x04path\x18\x01 \x01(\t\x12\x34\n\x06policy\x18\x02 \x01(\x0e\x32$.wandb_internal.FilesItem.PolicyType\x12\x30\n\x04type\x18\x03 \x01(\x0e\x32\".wandb_internal.FilesItem.FileType\"(\n\nPolicyType\x12\x07\n\x03NOW\x10\x00\x12\x07\n\x03\x45ND\x10\x01\x12\x08\n\x04LIVE\x10\x02\"9\n\x08\x46ileType\x12\t\n\x05OTHER\x10\x00\x12\t\n\x05WANDB\x10\x01\x12\t\n\x05MEDIA\x10\x02\x12\x0c\n\x08\x41RTIFACT\x10\x03J\x04\x08\x10\x10\x11\"\r\n\x0b\x46ilesResult\"\xe6\x01\n\x0bStatsRecord\x12\x39\n\nstats_type\x18\x01 \x01(\x0e\x32%.wandb_internal.StatsRecord.StatsType\x12-\n\ttimestamp\x18\x02 \x01(\x0b\x32\x1a.google.protobuf.Timestamp\x12\'\n\x04item\x18\x03 \x03(\x0b\x32\x19.wandb_internal.StatsItem\x12+\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1b.wandb_internal._RecordInfo\"\x17\n\tStatsType\x12\n\n\x06SYSTEM\x10\x00\",\n\tStatsItem\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\x12\n\nvalue_json\x18\x10 \x01(\t\"\xe7\x03\n\x0e\x41rtifactRecord\x12\x0e\n\x06run_id\x18\x01 \x01(\t\x12\x0f\n\x07project\x18\x02 \x01(\t\x12\x0e\n\x06\x65ntity\x18\x03 \x01(\t\x12\x0c\n\x04type\x18\x04 \x01(\t\x12\x0c\n\x04name\x18\x05 \x01(\t\x12\x0e\n\x06\x64igest\x18\x06 \x01(\t\x12\x13\n\x0b\x64\x65scription\x18\x07 \x01(\t\x12\x10\n\x08metadata\x18\x08 \x01(\t\x12\x14\n\x0cuser_created\x18\t \x01(\x08\x12\x18\n\x10use_after_commit\x18\n \x01(\x08\x12\x0f\n\x07\x61liases\x18\x0b \x03(\t\x12\x32\n\x08manifest\x18\x0c \x01(\x0b\x32 .wandb_internal.ArtifactManifest\x12\x16\n\x0e\x64istributed_id\x18\r \x01(\t\x12\x10\n\x08\x66inalize\x18\x0e \x01(\x08\x12\x11\n\tclient_id\x18\x0f \x01(\t\x12\x1a\n\x12sequence_client_id\x18\x10 \x01(\t\x12\x0f\n\x07\x62\x61se_id\x18\x11 \x01(\t\x12\x1c\n\x14ttl_duration_seconds\x18\x12 \x01(\x03\x12\x0c\n\x04tags\x18\x13 \x03(\t\x12\x19\n\x11incremental_beta1\x18\x64 \x01(\x08\x12+\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1b.wandb_internal._RecordInfo\"\xd8\x01\n\x10\x41rtifactManifest\x12\x0f\n\x07version\x18\x01 \x01(\x05\x12\x16\n\x0estorage_policy\x18\x02 \x01(\t\x12\x46\n\x15storage_policy_config\x18\x03 \x03(\x0b\x32\'.wandb_internal.StoragePolicyConfigItem\x12\x37\n\x08\x63ontents\x18\x04 \x03(\x0b\x32%.wandb_internal.ArtifactManifestEntry\x12\x1a\n\x12manifest_file_path\x18\x05 \x01(\t\"\xcf\x01\n\x15\x41rtifactManifestEntry\x12\x0c\n\x04path\x18\x01 \x01(\t\x12\x0e\n\x06\x64igest\x18\x02 \x01(\t\x12\x0b\n\x03ref\x18\x03 \x01(\t\x12\x0c\n\x04size\x18\x04 \x01(\x03\x12\x10\n\x08mimetype\x18\x05 \x01(\t\x12\x12\n\nlocal_path\x18\x06 \x01(\t\x12\x19\n\x11\x62irth_artifact_id\x18\x07 \x01(\t\x12\x12\n\nskip_cache\x18\x08 \x01(\x08\x12(\n\x05\x65xtra\x18\x10 \x03(\x0b\x32\x19.wandb_internal.ExtraItem\",\n\tExtraItem\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\x12\n\nvalue_json\x18\x02 \x01(\t\":\n\x17StoragePolicyConfigItem\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\x12\n\nvalue_json\x18\x02 \x01(\t\"\x10\n\x0e\x41rtifactResult\"\x14\n\x12LinkArtifactResult\"\xf0\x01\n\x13LinkArtifactRequest\x12\x11\n\tclient_id\x18\x01 \x01(\t\x12\x11\n\tserver_id\x18\x02 \x01(\t\x12\x16\n\x0eportfolio_name\x18\x03 \x01(\t\x12\x18\n\x10portfolio_entity\x18\x04 \x01(\t\x12\x19\n\x11portfolio_project\x18\x05 \x01(\t\x12\x19\n\x11portfolio_aliases\x18\x06 \x03(\t\x12\x1e\n\x16portfolio_organization\x18\x07 \x01(\t\x12+\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1b.wandb_internal._RecordInfo\"[\n\x14LinkArtifactResponse\x12\x15\n\rerror_message\x18\x01 \x01(\t\x12\x1a\n\rversion_index\x18\x02 \x01(\x05H\x00\x88\x01\x01\x42\x10\n\x0e_version_index\"h\n\x08TBRecord\x12+\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1b.wandb_internal._RecordInfo\x12\x0f\n\x07log_dir\x18\x01 \x01(\t\x12\x10\n\x08root_dir\x18\x03 \x01(\t\x12\x0c\n\x04save\x18\x02 \x01(\x08\"\n\n\x08TBResult\"}\n\x0b\x41lertRecord\x12\r\n\x05title\x18\x01 \x01(\t\x12\x0c\n\x04text\x18\x02 \x01(\t\x12\r\n\x05level\x18\x03 \x01(\t\x12\x15\n\rwait_duration\x18\x04 \x01(\x03\x12+\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1b.wandb_internal._RecordInfo\"\r\n\x0b\x41lertResult\"\xf4\x10\n\x07Request\x12\x38\n\x0bstop_status\x18\x01 \x01(\x0b\x32!.wandb_internal.StopStatusRequestH\x00\x12>\n\x0enetwork_status\x18\x02 \x01(\x0b\x32$.wandb_internal.NetworkStatusRequestH\x00\x12-\n\x05\x64\x65\x66\x65r\x18\x03 \x01(\x0b\x32\x1c.wandb_internal.DeferRequestH\x00\x12\x38\n\x0bget_summary\x18\x04 \x01(\x0b\x32!.wandb_internal.GetSummaryRequestH\x00\x12-\n\x05login\x18\x05 \x01(\x0b\x32\x1c.wandb_internal.LoginRequestH\x00\x12-\n\x05pause\x18\x06 \x01(\x0b\x32\x1c.wandb_internal.PauseRequestH\x00\x12/\n\x06resume\x18\x07 \x01(\x0b\x32\x1d.wandb_internal.ResumeRequestH\x00\x12\x34\n\tpoll_exit\x18\x08 \x01(\x0b\x32\x1f.wandb_internal.PollExitRequestH\x00\x12@\n\x0fsampled_history\x18\t \x01(\x0b\x32%.wandb_internal.SampledHistoryRequestH\x00\x12@\n\x0fpartial_history\x18\n \x01(\x0b\x32%.wandb_internal.PartialHistoryRequestH\x00\x12\x34\n\trun_start\x18\x0b \x01(\x0b\x32\x1f.wandb_internal.RunStartRequestH\x00\x12<\n\rcheck_version\x18\x0c \x01(\x0b\x32#.wandb_internal.CheckVersionRequestH\x00\x12:\n\x0clog_artifact\x18\r \x01(\x0b\x32\".wandb_internal.LogArtifactRequestH\x00\x12\x44\n\x11\x64ownload_artifact\x18\x0e \x01(\x0b\x32\'.wandb_internal.DownloadArtifactRequestH\x00\x12\x35\n\tkeepalive\x18\x11 \x01(\x0b\x32 .wandb_internal.KeepaliveRequestH\x00\x12\x36\n\nrun_status\x18\x14 \x01(\x0b\x32 .wandb_internal.RunStatusRequestH\x00\x12/\n\x06\x63\x61ncel\x18\x15 \x01(\x0b\x32\x1d.wandb_internal.CancelRequestH\x00\x12\x44\n\x11internal_messages\x18\x17 \x01(\x0b\x32\'.wandb_internal.InternalMessagesRequestH\x00\x12@\n\x0fpython_packages\x18\x18 \x01(\x0b\x32%.wandb_internal.PythonPackagesRequestH\x00\x12\x33\n\x08shutdown\x18@ \x01(\x0b\x32\x1f.wandb_internal.ShutdownRequestH\x00\x12/\n\x06\x61ttach\x18\x41 \x01(\x0b\x32\x1d.wandb_internal.AttachRequestH\x00\x12/\n\x06status\x18\x42 \x01(\x0b\x32\x1d.wandb_internal.StatusRequestH\x00\x12\x38\n\x0bserver_info\x18\x43 \x01(\x0b\x32!.wandb_internal.ServerInfoRequestH\x00\x12\x38\n\x0bsender_mark\x18\x44 \x01(\x0b\x32!.wandb_internal.SenderMarkRequestH\x00\x12\x38\n\x0bsender_read\x18\x45 \x01(\x0b\x32!.wandb_internal.SenderReadRequestH\x00\x12<\n\rstatus_report\x18\x46 \x01(\x0b\x32#.wandb_internal.StatusReportRequestH\x00\x12>\n\x0esummary_record\x18G \x01(\x0b\x32$.wandb_internal.SummaryRecordRequestH\x00\x12\x42\n\x10telemetry_record\x18H \x01(\x0b\x32&.wandb_internal.TelemetryRecordRequestH\x00\x12\x32\n\x08job_info\x18I \x01(\x0b\x32\x1e.wandb_internal.JobInfoRequestH\x00\x12\x45\n\x12get_system_metrics\x18J \x01(\x0b\x32\'.wandb_internal.GetSystemMetricsRequestH\x00\x12\x34\n\tjob_input\x18M \x01(\x0b\x32\x1f.wandb_internal.JobInputRequestH\x00\x12<\n\rlink_artifact\x18N \x01(\x0b\x32#.wandb_internal.LinkArtifactRequestH\x00\x12\x38\n\x0bsync_finish\x18Q \x01(\x0b\x32!.wandb_internal.SyncFinishRequestH\x00\x12;\n\noperations\x18R \x01(\x0b\x32%.wandb_internal.OperationStatsRequestH\x00\x12\x43\n\x11probe_system_info\x18S \x01(\x0b\x32&.wandb_internal.ProbeSystemInfoRequestH\x00\x12\x39\n\x0btest_inject\x18\xe8\x07 \x01(\x0b\x32!.wandb_internal.TestInjectRequestH\x00\x42\x0e\n\x0crequest_typeJ\x04\x08\x12\x10\x13J\x04\x08\x16\x10\x17J\x04\x08K\x10LJ\x04\x08L\x10MJ\x04\x08O\x10PJ\x04\x08P\x10Q\"\x83\r\n\x08Response\x12?\n\x12keepalive_response\x18\x12 \x01(\x0b\x32!.wandb_internal.KeepaliveResponseH\x00\x12\x42\n\x14stop_status_response\x18\x13 \x01(\x0b\x32\".wandb_internal.StopStatusResponseH\x00\x12H\n\x17network_status_response\x18\x14 \x01(\x0b\x32%.wandb_internal.NetworkStatusResponseH\x00\x12\x37\n\x0elogin_response\x18\x18 \x01(\x0b\x32\x1d.wandb_internal.LoginResponseH\x00\x12\x42\n\x14get_summary_response\x18\x19 \x01(\x0b\x32\".wandb_internal.GetSummaryResponseH\x00\x12>\n\x12poll_exit_response\x18\x1a \x01(\x0b\x32 .wandb_internal.PollExitResponseH\x00\x12J\n\x18sampled_history_response\x18\x1b \x01(\x0b\x32&.wandb_internal.SampledHistoryResponseH\x00\x12>\n\x12run_start_response\x18\x1c \x01(\x0b\x32 .wandb_internal.RunStartResponseH\x00\x12\x46\n\x16\x63heck_version_response\x18\x1d \x01(\x0b\x32$.wandb_internal.CheckVersionResponseH\x00\x12\x44\n\x15log_artifact_response\x18\x1e \x01(\x0b\x32#.wandb_internal.LogArtifactResponseH\x00\x12N\n\x1a\x64ownload_artifact_response\x18\x1f \x01(\x0b\x32(.wandb_internal.DownloadArtifactResponseH\x00\x12@\n\x13run_status_response\x18# \x01(\x0b\x32!.wandb_internal.RunStatusResponseH\x00\x12\x39\n\x0f\x63\x61ncel_response\x18$ \x01(\x0b\x32\x1e.wandb_internal.CancelResponseH\x00\x12N\n\x1ainternal_messages_response\x18% \x01(\x0b\x32(.wandb_internal.InternalMessagesResponseH\x00\x12=\n\x11shutdown_response\x18@ \x01(\x0b\x32 .wandb_internal.ShutdownResponseH\x00\x12\x39\n\x0f\x61ttach_response\x18\x41 \x01(\x0b\x32\x1e.wandb_internal.AttachResponseH\x00\x12\x39\n\x0fstatus_response\x18\x42 \x01(\x0b\x32\x1e.wandb_internal.StatusResponseH\x00\x12\x42\n\x14server_info_response\x18\x43 \x01(\x0b\x32\".wandb_internal.ServerInfoResponseH\x00\x12<\n\x11job_info_response\x18\x44 \x01(\x0b\x32\x1f.wandb_internal.JobInfoResponseH\x00\x12O\n\x1bget_system_metrics_response\x18\x45 \x01(\x0b\x32(.wandb_internal.GetSystemMetricsResponseH\x00\x12\x46\n\x16link_artifact_response\x18G \x01(\x0b\x32$.wandb_internal.LinkArtifactResponseH\x00\x12\x35\n\rsync_response\x18\x46 \x01(\x0b\x32\x1c.wandb_internal.SyncResponseH\x00\x12\x45\n\x13operations_response\x18J \x01(\x0b\x32&.wandb_internal.OperationStatsResponseH\x00\x12\x43\n\x14test_inject_response\x18\xe8\x07 \x01(\x0b\x32\".wandb_internal.TestInjectResponseH\x00\x42\x0f\n\rresponse_typeJ\x04\x08 \x10!J\x04\x08H\x10IJ\x04\x08I\x10J\"\xc0\x02\n\x0c\x44\x65\x66\x65rRequest\x12\x36\n\x05state\x18\x01 \x01(\x0e\x32\'.wandb_internal.DeferRequest.DeferState\"\xf7\x01\n\nDeferState\x12\t\n\x05\x42\x45GIN\x10\x00\x12\r\n\tFLUSH_RUN\x10\x01\x12\x0f\n\x0b\x46LUSH_STATS\x10\x02\x12\x19\n\x15\x46LUSH_PARTIAL_HISTORY\x10\x03\x12\x0c\n\x08\x46LUSH_TB\x10\x04\x12\r\n\tFLUSH_SUM\x10\x05\x12\x13\n\x0f\x46LUSH_DEBOUNCER\x10\x06\x12\x10\n\x0c\x46LUSH_OUTPUT\x10\x07\x12\r\n\tFLUSH_JOB\x10\x08\x12\r\n\tFLUSH_DIR\x10\t\x12\x0c\n\x08\x46LUSH_FP\x10\n\x12\x0b\n\x07JOIN_FP\x10\x0b\x12\x0c\n\x08\x46LUSH_FS\x10\x0c\x12\x0f\n\x0b\x46LUSH_FINAL\x10\r\x12\x07\n\x03\x45ND\x10\x0e\"<\n\x0cPauseRequest\x12,\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1c.wandb_internal._RequestInfo\"\x0f\n\rPauseResponse\"=\n\rResumeRequest\x12,\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1c.wandb_internal._RequestInfo\"\x10\n\x0eResumeResponse\"M\n\x0cLoginRequest\x12\x0f\n\x07\x61pi_key\x18\x01 \x01(\t\x12,\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1c.wandb_internal._RequestInfo\"&\n\rLoginResponse\x12\x15\n\ractive_entity\x18\x01 \x01(\t\"A\n\x11GetSummaryRequest\x12,\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1c.wandb_internal._RequestInfo\"?\n\x12GetSummaryResponse\x12)\n\x04item\x18\x01 \x03(\x0b\x32\x1b.wandb_internal.SummaryItem\"G\n\x17GetSystemMetricsRequest\x12,\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1c.wandb_internal._RequestInfo\"R\n\x12SystemMetricSample\x12-\n\ttimestamp\x18\x01 \x01(\x0b\x32\x1a.google.protobuf.Timestamp\x12\r\n\x05value\x18\x02 \x01(\x02\"I\n\x13SystemMetricsBuffer\x12\x32\n\x06record\x18\x01 \x03(\x0b\x32\".wandb_internal.SystemMetricSample\"\xca\x01\n\x18GetSystemMetricsResponse\x12S\n\x0esystem_metrics\x18\x01 \x03(\x0b\x32;.wandb_internal.GetSystemMetricsResponse.SystemMetricsEntry\x1aY\n\x12SystemMetricsEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\x32\n\x05value\x18\x02 \x01(\x0b\x32#.wandb_internal.SystemMetricsBuffer:\x02\x38\x01\"=\n\rStatusRequest\x12,\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1c.wandb_internal._RequestInfo\")\n\x0eStatusResponse\x12\x17\n\x0frun_should_stop\x18\x01 \x01(\x08\"A\n\x11StopStatusRequest\x12,\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1c.wandb_internal._RequestInfo\"-\n\x12StopStatusResponse\x12\x17\n\x0frun_should_stop\x18\x01 \x01(\x08\"D\n\x14NetworkStatusRequest\x12,\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1c.wandb_internal._RequestInfo\"P\n\x15NetworkStatusResponse\x12\x37\n\x11network_responses\x18\x01 \x03(\x0b\x32\x1c.wandb_internal.HttpResponse\"D\n\x0cHttpResponse\x12\x18\n\x10http_status_code\x18\x01 \x01(\x05\x12\x1a\n\x12http_response_text\x18\x02 \x01(\t\"G\n\x17InternalMessagesRequest\x12,\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1c.wandb_internal._RequestInfo\"N\n\x18InternalMessagesResponse\x12\x32\n\x08messages\x18\x01 \x01(\x0b\x32 .wandb_internal.InternalMessages\"#\n\x10InternalMessages\x12\x0f\n\x07warning\x18\x01 \x03(\t\"?\n\x0fPollExitRequest\x12,\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1c.wandb_internal._RequestInfo\"\xf5\x01\n\x10PollExitResponse\x12\x0c\n\x04\x64one\x18\x01 \x01(\x08\x12\x32\n\x0b\x65xit_result\x18\x02 \x01(\x0b\x32\x1d.wandb_internal.RunExitResult\x12\x35\n\x0cpusher_stats\x18\x03 \x01(\x0b\x32\x1f.wandb_internal.FilePusherStats\x12/\n\x0b\x66ile_counts\x18\x04 \x01(\x0b\x32\x1a.wandb_internal.FileCounts\x12\x37\n\x0foperation_stats\x18\x05 \x01(\x0b\x32\x1e.wandb_internal.OperationStats\"E\n\x15OperationStatsRequest\x12,\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1c.wandb_internal._RequestInfo\"Q\n\x16OperationStatsResponse\x12\x37\n\x0foperation_stats\x18\x01 \x01(\x0b\x32\x1e.wandb_internal.OperationStats\"h\n\x0eOperationStats\x12\r\n\x05label\x18\x03 \x01(\t\x12-\n\noperations\x18\x01 \x03(\x0b\x32\x19.wandb_internal.Operation\x12\x18\n\x10total_operations\x18\x02 \x01(\x03\"\x87\x01\n\tOperation\x12\x0c\n\x04\x64\x65sc\x18\x01 \x01(\t\x12\x17\n\x0fruntime_seconds\x18\x02 \x01(\x01\x12\x10\n\x08progress\x18\x03 \x01(\t\x12\x14\n\x0c\x65rror_status\x18\x04 \x01(\t\x12+\n\x08subtasks\x18\x05 \x03(\x0b\x32\x19.wandb_internal.Operation\"\x13\n\x11SenderMarkRequest\"\x13\n\x11SyncFinishRequest\"E\n\x0cSyncResponse\x12\x0b\n\x03url\x18\x01 \x01(\t\x12(\n\x05\x65rror\x18\x02 \x01(\x0b\x32\x19.wandb_internal.ErrorInfo\"?\n\x11SenderReadRequest\x12\x14\n\x0cstart_offset\x18\x01 \x01(\x03\x12\x14\n\x0c\x66inal_offset\x18\x02 \x01(\x03\"m\n\x13StatusReportRequest\x12\x12\n\nrecord_num\x18\x01 \x01(\x03\x12\x13\n\x0bsent_offset\x18\x02 \x01(\x03\x12-\n\tsync_time\x18\x03 \x01(\x0b\x32\x1a.google.protobuf.Timestamp\"F\n\x14SummaryRecordRequest\x12.\n\x07summary\x18\x01 \x01(\x0b\x32\x1d.wandb_internal.SummaryRecord\"L\n\x16TelemetryRecordRequest\x12\x32\n\ttelemetry\x18\x01 \x01(\x0b\x32\x1f.wandb_internal.TelemetryRecord\"A\n\x11ServerInfoRequest\x12,\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1c.wandb_internal._RequestInfo\"|\n\x12ServerInfoResponse\x12-\n\nlocal_info\x18\x01 \x01(\x0b\x32\x19.wandb_internal.LocalInfo\x12\x37\n\x0fserver_messages\x18\x02 \x01(\x0b\x32\x1e.wandb_internal.ServerMessages\"=\n\x0eServerMessages\x12+\n\x04item\x18\x01 \x03(\x0b\x32\x1d.wandb_internal.ServerMessage\"e\n\rServerMessage\x12\x12\n\nplain_text\x18\x01 \x01(\t\x12\x10\n\x08utf_text\x18\x02 \x01(\t\x12\x11\n\thtml_text\x18\x03 \x01(\t\x12\x0c\n\x04type\x18\x04 \x01(\t\x12\r\n\x05level\x18\x05 \x01(\x05\"c\n\nFileCounts\x12\x13\n\x0bwandb_count\x18\x01 \x01(\x05\x12\x13\n\x0bmedia_count\x18\x02 \x01(\x05\x12\x16\n\x0e\x61rtifact_count\x18\x03 \x01(\x05\x12\x13\n\x0bother_count\x18\x04 \x01(\x05\"U\n\x0f\x46ilePusherStats\x12\x16\n\x0euploaded_bytes\x18\x01 \x01(\x03\x12\x13\n\x0btotal_bytes\x18\x02 \x01(\x03\x12\x15\n\rdeduped_bytes\x18\x03 \x01(\x03\"\x1e\n\rFilesUploaded\x12\r\n\x05\x66iles\x18\x01 \x03(\t\"\xf4\x01\n\x17\x46ileTransferInfoRequest\x12\x42\n\x04type\x18\x01 \x01(\x0e\x32\x34.wandb_internal.FileTransferInfoRequest.TransferType\x12\x0c\n\x04path\x18\x02 \x01(\t\x12\x0b\n\x03url\x18\x03 \x01(\t\x12\x0c\n\x04size\x18\x04 \x01(\x03\x12\x11\n\tprocessed\x18\x05 \x01(\x03\x12/\n\x0b\x66ile_counts\x18\x06 \x01(\x0b\x32\x1a.wandb_internal.FileCounts\"(\n\x0cTransferType\x12\n\n\x06Upload\x10\x00\x12\x0c\n\x08\x44ownload\x10\x01\"1\n\tLocalInfo\x12\x0f\n\x07version\x18\x01 \x01(\t\x12\x13\n\x0bout_of_date\x18\x02 \x01(\x08\"?\n\x0fShutdownRequest\x12,\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1c.wandb_internal._RequestInfo\"\x12\n\x10ShutdownResponse\"P\n\rAttachRequest\x12\x11\n\tattach_id\x18\x14 \x01(\t\x12,\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1c.wandb_internal._RequestInfo\"b\n\x0e\x41ttachResponse\x12&\n\x03run\x18\x01 \x01(\x0b\x32\x19.wandb_internal.RunRecord\x12(\n\x05\x65rror\x18\x02 \x01(\x0b\x32\x19.wandb_internal.ErrorInfo\"\xd5\x02\n\x11TestInjectRequest\x12\x13\n\x0bhandler_exc\x18\x01 \x01(\x08\x12\x14\n\x0chandler_exit\x18\x02 \x01(\x08\x12\x15\n\rhandler_abort\x18\x03 \x01(\x08\x12\x12\n\nsender_exc\x18\x04 \x01(\x08\x12\x13\n\x0bsender_exit\x18\x05 \x01(\x08\x12\x14\n\x0csender_abort\x18\x06 \x01(\x08\x12\x0f\n\x07req_exc\x18\x07 \x01(\x08\x12\x10\n\x08req_exit\x18\x08 \x01(\x08\x12\x11\n\treq_abort\x18\t \x01(\x08\x12\x10\n\x08resp_exc\x18\n \x01(\x08\x12\x11\n\tresp_exit\x18\x0b \x01(\x08\x12\x12\n\nresp_abort\x18\x0c \x01(\x08\x12\x10\n\x08msg_drop\x18\r \x01(\x08\x12\x10\n\x08msg_hang\x18\x0e \x01(\x08\x12,\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1c.wandb_internal._RequestInfo\"\x14\n\x12TestInjectResponse\"\x1e\n\rHistoryAction\x12\r\n\x05\x66lush\x18\x01 \x01(\x08\"\xca\x01\n\x15PartialHistoryRequest\x12)\n\x04item\x18\x01 \x03(\x0b\x32\x1b.wandb_internal.HistoryItem\x12)\n\x04step\x18\x02 \x01(\x0b\x32\x1b.wandb_internal.HistoryStep\x12-\n\x06\x61\x63tion\x18\x03 \x01(\x0b\x32\x1d.wandb_internal.HistoryAction\x12,\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1c.wandb_internal._RequestInfo\"\x18\n\x16PartialHistoryResponse\"E\n\x15SampledHistoryRequest\x12,\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1c.wandb_internal._RequestInfo\"_\n\x12SampledHistoryItem\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\x12\n\nnested_key\x18\x02 \x03(\t\x12\x14\n\x0cvalues_float\x18\x03 \x03(\x02\x12\x12\n\nvalues_int\x18\x04 \x03(\x03\"J\n\x16SampledHistoryResponse\x12\x30\n\x04item\x18\x01 \x03(\x0b\x32\".wandb_internal.SampledHistoryItem\"@\n\x10RunStatusRequest\x12,\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1c.wandb_internal._RequestInfo\"x\n\x11RunStatusResponse\x12\x18\n\x10sync_items_total\x18\x01 \x01(\x03\x12\x1a\n\x12sync_items_pending\x18\x02 \x01(\x03\x12-\n\tsync_time\x18\x03 \x01(\x0b\x32\x1a.google.protobuf.Timestamp\"g\n\x0fRunStartRequest\x12&\n\x03run\x18\x01 \x01(\x0b\x32\x19.wandb_internal.RunRecord\x12,\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1c.wandb_internal._RequestInfo\"\x12\n\x10RunStartResponse\"\\\n\x13\x43heckVersionRequest\x12\x17\n\x0f\x63urrent_version\x18\x01 \x01(\t\x12,\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1c.wandb_internal._RequestInfo\"]\n\x14\x43heckVersionResponse\x12\x17\n\x0fupgrade_message\x18\x01 \x01(\t\x12\x14\n\x0cyank_message\x18\x02 \x01(\t\x12\x16\n\x0e\x64\x65lete_message\x18\x03 \x01(\t\">\n\x0eJobInfoRequest\x12,\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1c.wandb_internal._RequestInfo\"6\n\x0fJobInfoResponse\x12\x12\n\nsequenceId\x18\x01 \x01(\t\x12\x0f\n\x07version\x18\x02 \x01(\t\"\x9f\x01\n\x12LogArtifactRequest\x12\x30\n\x08\x61rtifact\x18\x01 \x01(\x0b\x32\x1e.wandb_internal.ArtifactRecord\x12\x14\n\x0chistory_step\x18\x02 \x01(\x03\x12\x13\n\x0bstaging_dir\x18\x03 \x01(\t\x12,\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1c.wandb_internal._RequestInfo\"A\n\x13LogArtifactResponse\x12\x13\n\x0b\x61rtifact_id\x18\x01 \x01(\t\x12\x15\n\rerror_message\x18\x02 \x01(\t\"\xbe\x01\n\x17\x44ownloadArtifactRequest\x12\x13\n\x0b\x61rtifact_id\x18\x01 \x01(\t\x12\x15\n\rdownload_root\x18\x02 \x01(\t\x12 \n\x18\x61llow_missing_references\x18\x04 \x01(\x08\x12\x12\n\nskip_cache\x18\x05 \x01(\x08\x12\x13\n\x0bpath_prefix\x18\x06 \x01(\t\x12,\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1c.wandb_internal._RequestInfo\"1\n\x18\x44ownloadArtifactResponse\x12\x15\n\rerror_message\x18\x01 \x01(\t\"@\n\x10KeepaliveRequest\x12,\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1c.wandb_internal._RequestInfo\"\x13\n\x11KeepaliveResponse\"q\n\x0c\x41rtifactInfo\x12\x10\n\x08\x61rtifact\x18\x01 \x01(\t\x12\x12\n\nentrypoint\x18\x02 \x03(\t\x12\x10\n\x08notebook\x18\x03 \x01(\x08\x12\x15\n\rbuild_context\x18\x04 \x01(\t\x12\x12\n\ndockerfile\x18\x05 \x01(\t\")\n\x07GitInfo\x12\x0e\n\x06remote\x18\x01 \x01(\t\x12\x0e\n\x06\x63ommit\x18\x02 \x01(\t\"\x87\x01\n\tGitSource\x12)\n\x08git_info\x18\x01 \x01(\x0b\x32\x17.wandb_internal.GitInfo\x12\x12\n\nentrypoint\x18\x02 \x03(\t\x12\x10\n\x08notebook\x18\x03 \x01(\x08\x12\x15\n\rbuild_context\x18\x04 \x01(\t\x12\x12\n\ndockerfile\x18\x05 \x01(\t\"\x1c\n\x0bImageSource\x12\r\n\x05image\x18\x01 \x01(\t\"\x8c\x01\n\x06Source\x12&\n\x03git\x18\x01 \x01(\x0b\x32\x19.wandb_internal.GitSource\x12.\n\x08\x61rtifact\x18\x02 \x01(\x0b\x32\x1c.wandb_internal.ArtifactInfo\x12*\n\x05image\x18\x03 \x01(\x0b\x32\x1b.wandb_internal.ImageSource\"k\n\tJobSource\x12\x10\n\x08_version\x18\x01 \x01(\t\x12\x13\n\x0bsource_type\x18\x02 \x01(\t\x12&\n\x06source\x18\x03 \x01(\x0b\x32\x16.wandb_internal.Source\x12\x0f\n\x07runtime\x18\x04 \x01(\t\"V\n\x12PartialJobArtifact\x12\x10\n\x08job_name\x18\x01 \x01(\t\x12.\n\x0bsource_info\x18\x02 \x01(\x0b\x32\x19.wandb_internal.JobSource\"\x9d\x01\n\x11UseArtifactRecord\x12\n\n\x02id\x18\x01 \x01(\t\x12\x0c\n\x04type\x18\x02 \x01(\t\x12\x0c\n\x04name\x18\x03 \x01(\t\x12\x33\n\x07partial\x18\x04 \x01(\x0b\x32\".wandb_internal.PartialJobArtifact\x12+\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1b.wandb_internal._RecordInfo\"\x13\n\x11UseArtifactResult\"R\n\rCancelRequest\x12\x13\n\x0b\x63\x61ncel_slot\x18\x01 \x01(\t\x12,\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1c.wandb_internal._RequestInfo\"\x10\n\x0e\x43\x61ncelResponse\"\x18\n\x16ProbeSystemInfoRequest\"\'\n\x08\x44iskInfo\x12\r\n\x05total\x18\x01 \x01(\x04\x12\x0c\n\x04used\x18\x02 \x01(\x04\"\x1b\n\nMemoryInfo\x12\r\n\x05total\x18\x01 \x01(\x04\"/\n\x07\x43puInfo\x12\r\n\x05\x63ount\x18\x01 \x01(\r\x12\x15\n\rcount_logical\x18\x02 \x01(\r\"\xad\x01\n\tAppleInfo\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x12\n\necpu_cores\x18\x02 \x01(\r\x12\x12\n\npcpu_cores\x18\x03 \x01(\r\x12\x11\n\tgpu_cores\x18\x04 \x01(\r\x12\x11\n\tmemory_gb\x18\x05 \x01(\r\x12\x18\n\x10swap_total_bytes\x18\x06 \x01(\x04\x12\x17\n\x0fram_total_bytes\x18\x07 \x01(\x04\x12\x11\n\tmac_model\x18\x08 \x01(\t\"k\n\rGpuNvidiaInfo\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x14\n\x0cmemory_total\x18\x02 \x01(\x04\x12\x12\n\ncuda_cores\x18\x03 \x01(\r\x12\x14\n\x0c\x61rchitecture\x18\x04 \x01(\t\x12\x0c\n\x04uuid\x18\x05 \x01(\t\"\x89\x02\n\nGpuAmdInfo\x12\n\n\x02id\x18\x01 \x01(\t\x12\x11\n\tunique_id\x18\x02 \x01(\t\x12\x15\n\rvbios_version\x18\x03 \x01(\t\x12\x19\n\x11performance_level\x18\x04 \x01(\t\x12\x15\n\rgpu_overdrive\x18\x05 \x01(\t\x12\x1c\n\x14gpu_memory_overdrive\x18\x06 \x01(\t\x12\x11\n\tmax_power\x18\x07 \x01(\t\x12\x0e\n\x06series\x18\x08 \x01(\t\x12\r\n\x05model\x18\t \x01(\t\x12\x0e\n\x06vendor\x18\n \x01(\t\x12\x0b\n\x03sku\x18\x0b \x01(\t\x12\x12\n\nsclk_range\x18\x0c \x01(\t\x12\x12\n\nmclk_range\x18\r \x01(\t\"n\n\x0cTrainiumInfo\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x0e\n\x06vendor\x18\x02 \x01(\t\x12\x1b\n\x13neuron_device_count\x18\x03 \x01(\r\x12#\n\x1bneuroncore_per_device_count\x18\x04 \x01(\r\"Q\n\x07TPUInfo\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x0f\n\x07hbm_gib\x18\x02 \x01(\r\x12\x18\n\x10\x64\x65vices_per_chip\x18\x03 \x01(\r\x12\r\n\x05\x63ount\x18\x04 \x01(\r\"E\n\rCoreWeaveInfo\x12\x14\n\x0c\x63luster_name\x18\x01 \x01(\t\x12\x0e\n\x06org_id\x18\x02 \x01(\t\x12\x0e\n\x06region\x18\x03 \x01(\t\"\xa8\t\n\x11\x45nvironmentRecord\x12\n\n\x02os\x18\x01 \x01(\t\x12\x0e\n\x06python\x18\x02 \x01(\t\x12\x39\n\nstarted_at\x18\x03 \x01(\x0b\x32\x1a.google.protobuf.TimestampR\tstartedAt\x12\x0e\n\x06\x64ocker\x18\x04 \x01(\t\x12\x0c\n\x04\x61rgs\x18\x05 \x03(\t\x12\x0f\n\x07program\x18\x06 \x01(\t\x12\x1b\n\tcode_path\x18\x07 \x01(\tR\x08\x63odePath\x12&\n\x0f\x63ode_path_local\x18\x08 \x01(\tR\rcodePathLocal\x12*\n\x03git\x18\t \x01(\x0b\x32\x1d.wandb_internal.GitRepoRecord\x12\r\n\x05\x65mail\x18\n \x01(\t\x12\x0c\n\x04root\x18\x0b \x01(\t\x12\x0c\n\x04host\x18\x0c \x01(\t\x12\x10\n\x08username\x18\r \x01(\t\x12\x12\n\nexecutable\x18\x0e \x01(\t\x12\r\n\x05\x63olab\x18\x0f \x01(\t\x12\x1c\n\tcpu_count\x18\x10 \x01(\rR\tcpu_count\x12,\n\x11\x63pu_count_logical\x18\x11 \x01(\rR\x11\x63pu_count_logical\x12\x15\n\x08gpu_type\x18\x12 \x01(\tR\x03gpu\x12\x1c\n\tgpu_count\x18\x13 \x01(\rR\tgpu_count\x12\x39\n\x04\x64isk\x18\x14 \x03(\x0b\x32+.wandb_internal.EnvironmentRecord.DiskEntry\x12*\n\x06memory\x18\x15 \x01(\x0b\x32\x1a.wandb_internal.MemoryInfo\x12$\n\x03\x63pu\x18\x16 \x01(\x0b\x32\x17.wandb_internal.CpuInfo\x12(\n\x05\x61pple\x18\x17 \x01(\x0b\x32\x19.wandb_internal.AppleInfo\x12=\n\ngpu_nvidia\x18\x18 \x03(\x0b\x32\x1d.wandb_internal.GpuNvidiaInfoR\ngpu_nvidia\x12\x14\n\x0c\x63uda_version\x18\x19 \x01(\t\x12\x34\n\x07gpu_amd\x18\x1a \x03(\x0b\x32\x1a.wandb_internal.GpuAmdInfoR\x07gpu_amd\x12;\n\x05slurm\x18\x1b \x03(\x0b\x32,.wandb_internal.EnvironmentRecord.SlurmEntry\x12.\n\x08trainium\x18\x1c \x01(\x0b\x32\x1c.wandb_internal.TrainiumInfo\x12$\n\x03tpu\x18\x1d \x01(\x0b\x32\x17.wandb_internal.TPUInfo\x12\x30\n\tcoreweave\x18\x1e \x01(\x0b\x32\x1d.wandb_internal.CoreWeaveInfo\x12\x12\n\twriter_id\x18\xc7\x01 \x01(\t\x12+\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1b.wandb_internal._RecordInfo\x1a\x45\n\tDiskEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\'\n\x05value\x18\x02 \x01(\x0b\x32\x18.wandb_internal.DiskInfo:\x02\x38\x01\x1a,\n\nSlurmEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t:\x02\x38\x01\"\x8d\x01\n\x15PythonPackagesRequest\x12\x44\n\x07package\x18\x01 \x03(\x0b\x32\x33.wandb_internal.PythonPackagesRequest.PythonPackage\x1a.\n\rPythonPackage\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x0f\n\x07version\x18\x02 \x01(\t\"\x1c\n\x0cJobInputPath\x12\x0c\n\x04path\x18\x01 \x03(\t\"\xd6\x01\n\x0eJobInputSource\x12\x44\n\nrun_config\x18\x01 \x01(\x0b\x32..wandb_internal.JobInputSource.RunConfigSourceH\x00\x12?\n\x04\x66ile\x18\x02 \x01(\x0b\x32/.wandb_internal.JobInputSource.ConfigFileSourceH\x00\x1a\x11\n\x0fRunConfigSource\x1a \n\x10\x43onfigFileSource\x12\x0c\n\x04path\x18\x01 \x01(\tB\x08\n\x06source\"\xc7\x01\n\x0fJobInputRequest\x12\x34\n\x0cinput_source\x18\x01 \x01(\x0b\x32\x1e.wandb_internal.JobInputSource\x12\x33\n\rinclude_paths\x18\x02 \x03(\x0b\x32\x1c.wandb_internal.JobInputPath\x12\x33\n\rexclude_paths\x18\x03 \x03(\x0b\x32\x1c.wandb_internal.JobInputPath\x12\x14\n\x0cinput_schema\x18\x04 \x01(\t*\xda\x05\n\rServerFeature\x12\x1e\n\x1aSERVER_FEATURE_UNSPECIFIED\x10\x00\x12\x13\n\x0fLARGE_FILENAMES\x10\x11\x12\x11\n\rARTIFACT_TAGS\x10\x01\x12\x0e\n\nCLIENT_IDS\x10\x02\x12\x1c\n\x18\x41RTIFACT_REGISTRY_SEARCH\x10\x03\x12\x1b\n\x17STRUCTURED_CONSOLE_LOGS\x10\x04\x12(\n$ARTIFACT_COLLECTION_MEMBERSHIP_FILES\x10\x05\x12\x38\n4ARTIFACT_COLLECTION_MEMBERSHIP_FILE_DOWNLOAD_HANDLER\x10\x06\x12\x34\n0USE_ARTIFACT_WITH_ENTITY_AND_PROJECT_INFORMATION\x10\x07\x12\x1f\n\x1b\x45XPAND_DEFINED_METRIC_GLOBS\x10\x08\x12\x1f\n\x1b\x41UTOMATION_EVENT_RUN_METRIC\x10\t\x12&\n\"AUTOMATION_EVENT_RUN_METRIC_CHANGE\x10\n\x12\x1b\n\x17\x41UTOMATION_ACTION_NO_OP\x10\x0b\x12/\n+INCLUDE_ARTIFACT_TYPES_IN_REGISTRY_CREATION\x10\x0c\x12*\n&PROJECT_ARTIFACT_COLLECTION_MEMBERSHIP\x10\r\x12\x31\n-ARTIFACT_MEMBERSHIP_IN_LINK_ARTIFACT_RESPONSE\x10\x0e\x12\"\n\x1eTOTAL_COUNT_IN_FILE_CONNECTION\x10\x0f\x12*\n&ARTIFACT_COLLECTIONS_FILTERING_SORTING\x10\x10\x12\x35\n1ARTIFACT_V2_DOWNLOAD_HANDLER_SUPPORTS_ARTIFACT_ID\x10\x12\x42\x1bZ\x19\x63ore/pkg/service_go_protob\x06proto3')
/n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/wandb/proto/v6/wandb_api_pb2.py:29:DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x1bwandb/proto/wandb_api.proto\x12\x0ewandb_internal\x1a wandb/proto/wandb_internal.proto\x1a wandb/proto/wandb_settings.proto\"B\n\x14ServerApiInitRequest\x12*\n\x08settings\x18\x01 \x01(\x0b\x32\x18.wandb_internal.Settings\">\n\x15ServerApiInitResponse\x12\x15\n\rerror_message\x18\x01 \x01(\t\x12\x0e\n\x06\x61pi_id\x18\x02 \x01(\t\"\xaf\x01\n\nApiRequest\x12\x0e\n\x06\x61pi_id\x18\x01 \x01(\t\x12I\n\x18read_run_history_request\x18\x02 \x01(\x0b\x32%.wandb_internal.ReadRunHistoryRequestH\x00\x12;\n\x10\x66\x65\x61tures_request\x18\x03 \x01(\x0b\x32\x1f.wandb_internal.FeaturesRequestH\x00\x42\t\n\x07request\"\xe5\x01\n\x0b\x41piResponse\x12K\n\x19read_run_history_response\x18\x01 \x01(\x0b\x32&.wandb_internal.ReadRunHistoryResponseH\x00\x12=\n\x11\x66\x65\x61tures_response\x18\x03 \x01(\x0b\x32 .wandb_internal.FeaturesResponseH\x00\x12>\n\x12\x61pi_error_response\x18\x02 \x01(\x0b\x32 .wandb_internal.ApiErrorResponseH\x00\x42\n\n\x08response\"f\n\x10\x41piErrorResponse\x12\x0f\n\x07message\x18\x01 \x01(\t\x12\x32\n\nerror_type\x18\x02 \x01(\x0e\x32\x19.wandb_internal.ErrorTypeH\x00\x88\x01\x01\x42\r\n\x0b_error_type\")\n\x17ServerApiCleanupRequest\x12\x0e\n\x06\x61pi_id\x18\x01 \x01(\t\"B\n\x0f\x46\x65\x61turesRequest\x12/\n\x08\x66\x65\x61tures\x18\x01 \x03(\x0e\x32\x1d.wandb_internal.ServerFeature\"B\n\x10\x46\x65\x61turesResponse\x12.\n\x07\x65nabled\x18\x01 \x03(\x0e\x32\x1d.wandb_internal.ServerFeature\"\xd0\x03\n\x15ReadRunHistoryRequest\x12\x43\n\x15scan_run_history_init\x18\x01 \x01(\x0b\x32\".wandb_internal.ScanRunHistoryInitH\x00\x12:\n\x10scan_run_history\x18\x02 \x01(\x0b\x32\x1e.wandb_internal.ScanRunHistoryH\x00\x12I\n\x18scan_run_history_cleanup\x18\x03 \x01(\x0b\x32%.wandb_internal.ScanRunHistoryCleanupH\x00\x12K\n\x19\x64ownload_run_history_init\x18\x04 \x01(\x0b\x32&.wandb_internal.DownloadRunHistoryInitH\x00\x12\x42\n\x14\x64ownload_run_history\x18\x05 \x01(\x0b\x32\".wandb_internal.DownloadRunHistoryH\x00\x12O\n\x1b\x64ownload_run_history_status\x18\x06 \x01(\x0b\x32(.wandb_internal.DownloadRunHistoryStatusH\x00\x42\t\n\x07request\"\xf9\x03\n\x16ReadRunHistoryResponse\x12K\n\x15scan_run_history_init\x18\x01 \x01(\x0b\x32*.wandb_internal.ScanRunHistoryInitResponseH\x00\x12\x39\n\x0brun_history\x18\x02 \x01(\x0b\x32\".wandb_internal.RunHistoryResponseH\x00\x12Q\n\x18scan_run_history_cleanup\x18\x03 \x01(\x0b\x32-.wandb_internal.ScanRunHistoryCleanupResponseH\x00\x12S\n\x19\x64ownload_run_history_init\x18\x04 \x01(\x0b\x32..wandb_internal.DownloadRunHistoryInitResponseH\x00\x12J\n\x14\x64ownload_run_history\x18\x05 \x01(\x0b\x32*.wandb_internal.DownloadRunHistoryResponseH\x00\x12W\n\x1b\x64ownload_run_history_status\x18\x06 \x01(\x0b\x32\x30.wandb_internal.DownloadRunHistoryStatusResponseH\x00\x42\n\n\x08response\"f\n\x12ScanRunHistoryInit\x12\x0e\n\x06\x65ntity\x18\x01 \x01(\t\x12\x0f\n\x07project\x18\x02 \x01(\t\x12\x0e\n\x06run_id\x18\x03 \x01(\t\x12\x0c\n\x04keys\x18\x04 \x03(\t\x12\x11\n\tuse_cache\x18\x05 \x01(\x08\"0\n\x1aScanRunHistoryInitResponse\x12\x12\n\nrequest_id\x18\x01 \x01(\x05\"H\n\x0eScanRunHistory\x12\x10\n\x08min_step\x18\x01 \x01(\x03\x12\x10\n\x08max_step\x18\x02 \x01(\x03\x12\x12\n\nrequest_id\x18\x03 \x01(\x05\"F\n\x12RunHistoryResponse\x12\x30\n\x0chistory_rows\x18\x01 \x03(\x0b\x32\x1a.wandb_internal.HistoryRow\"G\n\nHistoryRow\x12\x39\n\rhistory_items\x18\x01 \x03(\x0b\x32\".wandb_internal.ParquetHistoryItem\"5\n\x12ParquetHistoryItem\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\x12\n\nvalue_json\x18\x10 \x01(\t\"+\n\x15ScanRunHistoryCleanup\x12\x12\n\nrequest_id\x18\x01 \x01(\x05\"\x1f\n\x1dScanRunHistoryCleanupResponse\"\x81\x01\n\x16\x44ownloadRunHistoryInit\x12\x0e\n\x06\x65ntity\x18\x01 \x01(\t\x12\x0f\n\x07project\x18\x02 \x01(\t\x12\x0e\n\x06run_id\x18\x03 \x01(\t\x12\x14\n\x0c\x64ownload_dir\x18\x04 \x01(\t\x12 \n\x18require_complete_history\x18\x05 \x01(\x08\"P\n\x1e\x44ownloadRunHistoryInitResponse\x12\x12\n\nrequest_id\x18\x01 \x01(\x05\x12\x1a\n\x12\x63ontains_live_data\x18\x02 \x01(\x08\"(\n\x12\x44ownloadRunHistory\x12\x12\n\nrequest_id\x18\x01 \x01(\x05\"\xad\x01\n\x1a\x44ownloadRunHistoryResponse\x12\x18\n\x10\x64ownloaded_files\x18\x01 \x03(\t\x12\x46\n\x06\x65rrors\x18\x02 \x03(\x0b\x32\x36.wandb_internal.DownloadRunHistoryResponse.ErrorsEntry\x1a-\n\x0b\x45rrorsEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t:\x02\x38\x01\"\x1b\n\x19IncompleteRunHistoryError\".\n\x18\x44ownloadRunHistoryStatus\x12\x12\n\nrequest_id\x18\x01 \x01(\x05\"[\n DownloadRunHistoryStatusResponse\x12\x37\n\x0foperation_stats\x18\x01 \x01(\x0b\x32\x1e.wandb_internal.OperationStats*@\n\tErrorType\x12\x11\n\rUNKNOWN_ERROR\x10\x00\x12 \n\x1cINCOMPLETE_RUN_HISTORY_ERROR\x10\x01\x42\x1bZ\x19\x63ore/pkg/service_go_protob\x06proto3')
/n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/wandb/proto/v6/wandb_settings_pb2.py:28:DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n wandb/proto/wandb_settings.proto\x12\x0ewandb_internal\x1a\x1egoogle/protobuf/wrappers.proto\" \n\x0fListStringValue\x12\r\n\x05value\x18\x01 \x03(\t\"\x1d\n\x0cListIntValue\x12\r\n\x05value\x18\x01 \x03(\x05\"\x8a\x01\n\x17MapStringKeyStringValue\x12\x41\n\x05value\x18\x01 \x03(\x0b\x32\x32.wandb_internal.MapStringKeyStringValue.ValueEntry\x1a,\n\nValueEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t:\x02\x38\x01\"\xcb\x01\n#MapStringKeyMapStringKeyStringValue\x12M\n\x05value\x18\x01 \x03(\x0b\x32>.wandb_internal.MapStringKeyMapStringKeyStringValue.ValueEntry\x1aU\n\nValueEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\x36\n\x05value\x18\x02 \x01(\x0b\x32\'.wandb_internal.MapStringKeyStringValue:\x02\x38\x01\"\x9a\x01\n\x12OpenMetricsFilters\x12\x33\n\x08sequence\x18\x01 \x01(\x0b\x32\x1f.wandb_internal.ListStringValueH\x00\x12\x46\n\x07mapping\x18\x02 \x01(\x0b\x32\x33.wandb_internal.MapStringKeyMapStringKeyStringValueH\x00\x42\x07\n\x05value\"7\n\tRunMoment\x12\x0b\n\x03run\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\x01\x12\x0e\n\x06metric\x18\x03 \x01(\t\"\xbeO\n\x08Settings\x12-\n\x07\x61pi_key\x18\x37 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12:\n\x13identity_token_file\x18\xaa\x01 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12\x37\n\x10\x63redentials_file\x18\xab\x01 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12\x39\n\x14insecure_disable_ssl\x18\xb9\x01 \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12,\n\x08_offline\x18\x1e \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12*\n\x06x_sync\x18\x1f \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12\x30\n\tsync_file\x18\x86\x01 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12,\n\x07_shared\x18\xa2\x01 \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12,\n\x06run_id\x18k \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12-\n\x07run_url\x18q \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12-\n\x07project\x18\x61 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12,\n\x06\x65ntity\x18\x45 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12\x33\n\x0corganization\x18\xbc\x01 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12\x32\n\x0cx_start_time\x18) \x01(\x0b\x32\x1c.google.protobuf.DoubleValue\x12.\n\x08root_dir\x18i \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12\x30\n\twandb_dir\x18\x8e\x01 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12-\n\x07log_dir\x18U \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12\x32\n\x0clog_internal\x18V \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12\x35\n\x0cignore_globs\x18N \x01(\x0b\x32\x1f.wandb_internal.ListStringValue\x12.\n\x07\x61pp_url\x18\xca\x01 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12.\n\x08\x62\x61se_url\x18\x39 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12=\n\x17x_file_stream_max_bytes\x18\xac\x01 \x01(\x0b\x32\x1b.google.protobuf.Int32Value\x12\x46\n\x1fx_file_stream_transmit_interval\x18\xaf\x01 \x01(\x0b\x32\x1c.google.protobuf.DoubleValue\x12\x45\n\x14x_extra_http_headers\x18\x0e \x01(\x0b\x32\'.wandb_internal.MapStringKeyStringValue\x12=\n\x17x_file_stream_retry_max\x18\x93\x01 \x01(\x0b\x32\x1b.google.protobuf.Int32Value\x12K\n$x_file_stream_retry_wait_min_seconds\x18\x94\x01 \x01(\x0b\x32\x1c.google.protobuf.DoubleValue\x12K\n$x_file_stream_retry_wait_max_seconds\x18\x95\x01 \x01(\x0b\x32\x1c.google.protobuf.DoubleValue\x12\x43\n\x1dx_file_stream_timeout_seconds\x18\x0f \x01(\x0b\x32\x1c.google.protobuf.DoubleValue\x12\x42\n\x1cx_file_stream_max_line_bytes\x18\xb2\x01 \x01(\x0b\x32\x1b.google.protobuf.Int32Value\x12?\n\x19x_file_transfer_retry_max\x18\x96\x01 \x01(\x0b\x32\x1b.google.protobuf.Int32Value\x12M\n&x_file_transfer_retry_wait_min_seconds\x18\x97\x01 \x01(\x0b\x32\x1c.google.protobuf.DoubleValue\x12M\n&x_file_transfer_retry_wait_max_seconds\x18\x98\x01 \x01(\x0b\x32\x1c.google.protobuf.DoubleValue\x12\x46\n\x1fx_file_transfer_timeout_seconds\x18\x99\x01 \x01(\x0b\x32\x1c.google.protobuf.DoubleValue\x12\x39\n\x13x_graphql_retry_max\x18\x9a\x01 \x01(\x0b\x32\x1b.google.protobuf.Int32Value\x12G\n x_graphql_retry_wait_min_seconds\x18\x9b\x01 \x01(\x0b\x32\x1c.google.protobuf.DoubleValue\x12G\n x_graphql_retry_wait_max_seconds\x18\x9c\x01 \x01(\x0b\x32\x1c.google.protobuf.DoubleValue\x12@\n\x19x_graphql_timeout_seconds\x18\x9d\x01 \x01(\x0b\x32\x1c.google.protobuf.DoubleValue\x12\x31\n\nhttp_proxy\x18\xa8\x01 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12\x32\n\x0bhttps_proxy\x18\xa9\x01 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12;\n\tx_proxies\x18\xc8\x01 \x01(\x0b\x32\'.wandb_internal.MapStringKeyStringValue\x12-\n\x07program\x18_ \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12\x35\n\x0fprogram_relpath\x18` \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12\x37\n\x10_code_path_local\x18\xa3\x01 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12\x36\n\x0fprogram_abspath\x18\x9f\x01 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12.\n\x05_args\x18\x01 \x01(\x0b\x32\x1f.wandb_internal.ListStringValue\x12)\n\x03_os\x18  \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12,\n\x06\x64ocker\x18\x43 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12\x32\n\x0cx_executable\x18\r \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12-\n\x07_python\x18\" \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12\x30\n\tcolab_url\x18\xa0\x01 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12*\n\x04host\x18M \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12/\n\x08username\x18\x8d\x01 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12+\n\x05\x65mail\x18\x44 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12,\n\x06resume\x18\x66 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12/\n\x0bresume_from\x18\xa7\x01 \x01(\x0b\x32\x19.wandb_internal.RunMoment\x12-\n\tfork_from\x18\xa4\x01 \x01(\x0b\x32\x19.wandb_internal.RunMoment\x12\x38\n\x14\x64isable_job_creation\x18\x41 \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12\x30\n\tsweep_url\x18\x83\x01 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12;\n\x16x_disable_update_check\x18\xa5\x01 \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12\x32\n\x0ex_disable_meta\x18\x07 \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12-\n\tsave_code\x18s \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12/\n\x0b\x64isable_git\x18? \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12;\n\x16\x64isable_git_fork_point\x18\xcb\x01 \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12;\n\x16x_disable_machine_info\x18\x9e\x01 \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12\x33\n\x0fx_disable_stats\x18\n \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12\x39\n\x13x_stats_buffer_size\x18\xa1\x01 \x01(\x0b\x32\x1b.google.protobuf.Int32Value\x12@\n\x19x_stats_sampling_interval\x18\xae\x01 \x01(\x0b\x32\x1c.google.protobuf.DoubleValue\x12\x30\n\x0bx_stats_pid\x18* \x01(\x0b\x32\x1b.google.protobuf.Int32Value\x12<\n\x12x_stats_disk_paths\x18\x92\x01 \x01(\x0b\x32\x1f.wandb_internal.ListStringValue\x12H\n\"x_stats_neuron_monitor_config_path\x18. \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12<\n\x15x_stats_dcgm_exporter\x18\xbb\x01 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12O\n\x1ex_stats_open_metrics_endpoints\x18/ \x01(\x0b\x32\'.wandb_internal.MapStringKeyStringValue\x12H\n\x1cx_stats_open_metrics_filters\x18\x30 \x01(\x0b\x32\".wandb_internal.OpenMetricsFilters\x12S\n!x_stats_open_metrics_http_headers\x18\xb8\x01 \x01(\x0b\x32\'.wandb_internal.MapStringKeyStringValue\x12=\n\x16x_stats_gpu_device_ids\x18\xba\x01 \x01(\x0b\x32\x1c.wandb_internal.ListIntValue\x12\x37\n\x11x_stats_cpu_count\x18\xc2\x01 \x01(\x0b\x32\x1b.google.protobuf.Int32Value\x12?\n\x19x_stats_cpu_logical_count\x18\xc3\x01 \x01(\x0b\x32\x1b.google.protobuf.Int32Value\x12\x37\n\x11x_stats_gpu_count\x18\xc4\x01 \x01(\x0b\x32\x1b.google.protobuf.Int32Value\x12\x37\n\x10x_stats_gpu_type\x18\xc5\x01 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12?\n\x1ax_stats_track_process_tree\x18\xc6\x01 \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12.\n\x07x_label\x18\xb5\x01 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12.\n\tx_primary\x18\xb6\x01 \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12:\n\x15x_update_finish_state\x18\xb7\x01 \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12<\n\x17\x61llow_offline_artifacts\x18\xb1\x01 \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12-\n\x07\x63onsole\x18< \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12\x36\n\x11\x63onsole_multipart\x18\xa6\x01 \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12=\n\x17\x63onsole_chunk_max_bytes\x18\xc7\x01 \x01(\x0b\x32\x1b.google.protobuf.Int32Value\x12?\n\x19\x63onsole_chunk_max_seconds\x18\xc9\x01 \x01(\x0b\x32\x1b.google.protobuf.Int32Value\x12\x35\n\x10sync_tensorboard\x18\xb3\x01 \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12\x42\n\x1dx_server_side_derived_summary\x18\xbd\x01 \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12\x46\n!x_server_side_expand_glob_metrics\x18\xbe\x01 \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12;\n\x16x_skip_transaction_log\x18\xbf\x01 \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12J\n#x_stats_coreweave_metadata_base_url\x18\xc0\x01 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12J\n#x_stats_coreweave_metadata_endpoint\x18\xc1\x01 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12/\n\x0b_aws_lambda\x18\x02 \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12\x33\n\x0fx_cli_only_mode\x18\x04 \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12*\n\x06_colab\x18\x05 \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12\x34\n\x10x_disable_viewer\x18\x0b \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12\x39\n\x15x_flow_control_custom\x18\x10 \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12;\n\x17x_flow_control_disabled\x18\x11 \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12>\n\x18x_internal_check_process\x18\x12 \x01(\x0b\x32\x1c.google.protobuf.DoubleValue\x12,\n\x08_ipython\x18\x14 \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12,\n\x08_jupyter\x18\x15 \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12\x34\n\x0ex_jupyter_root\x18\x16 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12+\n\x07_kaggle\x18\x17 \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12=\n\x18x_live_policy_rate_limit\x18\x18 \x01(\x0b\x32\x1b.google.protobuf.Int32Value\x12<\n\x17x_live_policy_wait_time\x18\x19 \x01(\x0b\x32\x1b.google.protobuf.Int32Value\x12\x30\n\x0bx_log_level\x18\x1a \x01(\x0b\x32\x1b.google.protobuf.Int32Value\x12\x35\n\x10x_network_buffer\x18\x1b \x01(\x0b\x32\x1b.google.protobuf.Int32Value\x12)\n\x05_noop\x18\x1c \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12-\n\t_notebook\x18\x1d \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12/\n\t_platform\x18! \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12\x38\n\x12x_runqueue_item_id\x18# \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12\x37\n\x13x_save_requirements\x18% \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12\x39\n\x13x_service_transport\x18& \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12\x34\n\x0ex_service_wait\x18\' \x01(\x0b\x32\x1c.google.protobuf.DoubleValue\x12\x35\n\x0f_start_datetime\x18( \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12\x33\n\r_tmp_code_dir\x18\x31 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12,\n\x08_windows\x18\x34 \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12\x38\n\x13\x61llow_media_symlink\x18\xcc\x01 \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12\x34\n\x10\x61llow_val_change\x18\x35 \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12P\n\x1f\x61zure_account_url_to_access_key\x18\x38 \x01(\x0b\x32\'.wandb_internal.MapStringKeyStringValue\x12.\n\x08\x63ode_dir\x18: \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12\x35\n\x0c\x63onfig_paths\x18; \x01(\x0b\x32\x1f.wandb_internal.ListStringValue\x12\x30\n\ndeployment\x18= \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12\x30\n\x0c\x64isable_code\x18> \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12\x31\n\rdisable_hints\x18@ \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12,\n\x08\x64isabled\x18\x42 \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12)\n\x05\x66orce\x18G \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12\x30\n\ngit_commit\x18H \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12\x30\n\ngit_remote\x18I \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12\x34\n\x0egit_remote_url\x18J \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12.\n\x08git_root\x18K \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12\x36\n\x11heartbeat_seconds\x18L \x01(\x0b\x32\x1b.google.protobuf.Int32Value\x12\x32\n\x0cinit_timeout\x18O \x01(\x0b\x32\x1c.google.protobuf.DoubleValue\x12,\n\x08is_local\x18P \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12\x30\n\njob_source\x18Q \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12\x31\n\rlabel_disable\x18R \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12*\n\x06launch\x18S \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12\x38\n\x12launch_config_path\x18T \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12:\n\x14log_symlink_internal\x18W \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12\x36\n\x10log_symlink_user\x18X \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12.\n\x08log_user\x18Y \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12\x33\n\rlogin_timeout\x18Z \x01(\x0b\x32\x1c.google.protobuf.DoubleValue\x12*\n\x04mode\x18\\ \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12\x33\n\rnotebook_name\x18] \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12\x31\n\x0bproject_url\x18\x62 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12)\n\x05quiet\x18\x63 \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12+\n\x07relogin\x18\x65 \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12\x32\n\x0cresume_fname\x18g \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12+\n\x07resumed\x18h \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12/\n\trun_group\x18j \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12\x32\n\x0crun_job_type\x18l \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12.\n\x08run_mode\x18m \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12.\n\x08run_name\x18n \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12/\n\trun_notes\x18o \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12\x31\n\x08run_tags\x18p \x01(\x0b\x32\x1f.wandb_internal.ListStringValue\x12\x35\n\x11sagemaker_disable\x18r \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12\x35\n\x0fsettings_system\x18t \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12\x38\n\x12settings_workspace\x18u \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12/\n\x0bshow_colors\x18v \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12.\n\nshow_emoji\x18w \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12/\n\x0bshow_errors\x18x \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12-\n\tshow_info\x18y \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12\x31\n\rshow_warnings\x18z \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12*\n\x06silent\x18{ \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12\x32\n\x0cstart_method\x18| \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12*\n\x06strict\x18} \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12\x33\n\x0esummary_errors\x18~ \x01(\x0b\x32\x1b.google.protobuf.Int32Value\x12\x34\n\x0fsummary_timeout\x18\x7f \x01(\x0b\x32\x1b.google.protobuf.Int32Value\x12\x36\n\x10summary_warnings\x18\x80\x01 \x01(\x0b\x32\x1b.google.protobuf.Int32Value\x12/\n\x08sweep_id\x18\x81\x01 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12\x37\n\x10sweep_param_path\x18\x82\x01 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12,\n\x07symlink\x18\x84\x01 \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12/\n\x08sync_dir\x18\x85\x01 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12:\n\x13sync_symlink_latest\x18\x87\x01 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12J\n%table_raise_on_max_row_limit_exceeded\x18\x8a\x01 \x01(\x0b\x32\x1a.google.protobuf.BoolValue\x12/\n\x08timespec\x18\x8b\x01 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12.\n\x07tmp_dir\x18\x8c\x01 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12\x35\n\x0ex_jupyter_name\x18\x8f\x01 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12\x35\n\x0ex_jupyter_path\x18\x90\x01 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12/\n\x08job_name\x18\x91\x01 \x01(\x0b\x32\x1c.google.protobuf.StringValueJ\x04\x08\x03\x10\x04J\x04\x08\x06\x10\x07J\x04\x08\x08\x10\tJ\x04\x08\t\x10\nJ\x04\x08\x0c\x10\rJ\x04\x08\x13\x10\x14J\x04\x08$\x10%J\x04\x08+\x10,J\x04\x08,\x10-J\x04\x08-\x10.J\x04\x08\x32\x10\x33J\x04\x08\x33\x10\x34J\x04\x08\x36\x10\x37J\x04\x08\x46\x10GJ\x04\x08[\x10\\J\x04\x08^\x10_J\x04\x08\x64\x10\x65J\x06\x08\x88\x01\x10\x89\x01J\x06\x08\x89\x01\x10\x8a\x01J\x06\x08\xad\x01\x10\xae\x01J\x06\x08\xb0\x01\x10\xb1\x01J\x06\x08\xb4\x01\x10\xb5\x01\x42\x1bZ\x19\x63ore/pkg/service_go_protob\x06proto3')
/n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/wandb/proto/v6/wandb_internal_pb2.py:31:DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n wandb/proto/wandb_internal.proto\x12\x0ewandb_internal\x1a\x1bgoogle/protobuf/empty.proto\x1a\x1fgoogle/protobuf/timestamp.proto\x1a\x1cwandb/proto/wandb_base.proto\x1a!wandb/proto/wandb_telemetry.proto\"\xcf\t\n\x06Record\x12\x0b\n\x03num\x18\x01 \x01(\x03\x12\x30\n\x07history\x18\x02 \x01(\x0b\x32\x1d.wandb_internal.HistoryRecordH\x00\x12\x30\n\x07summary\x18\x03 \x01(\x0b\x32\x1d.wandb_internal.SummaryRecordH\x00\x12.\n\x06output\x18\x04 \x01(\x0b\x32\x1c.wandb_internal.OutputRecordH\x00\x12.\n\x06\x63onfig\x18\x05 \x01(\x0b\x32\x1c.wandb_internal.ConfigRecordH\x00\x12,\n\x05\x66iles\x18\x06 \x01(\x0b\x32\x1b.wandb_internal.FilesRecordH\x00\x12,\n\x05stats\x18\x07 \x01(\x0b\x32\x1b.wandb_internal.StatsRecordH\x00\x12\x32\n\x08\x61rtifact\x18\x08 \x01(\x0b\x32\x1e.wandb_internal.ArtifactRecordH\x00\x12,\n\x08tbrecord\x18\t \x01(\x0b\x32\x18.wandb_internal.TBRecordH\x00\x12,\n\x05\x61lert\x18\n \x01(\x0b\x32\x1b.wandb_internal.AlertRecordH\x00\x12\x34\n\ttelemetry\x18\x0b \x01(\x0b\x32\x1f.wandb_internal.TelemetryRecordH\x00\x12.\n\x06metric\x18\x0c \x01(\x0b\x32\x1c.wandb_internal.MetricRecordH\x00\x12\x35\n\noutput_raw\x18\r \x01(\x0b\x32\x1f.wandb_internal.OutputRawRecordH\x00\x12(\n\x03run\x18\x11 \x01(\x0b\x32\x19.wandb_internal.RunRecordH\x00\x12-\n\x04\x65xit\x18\x12 \x01(\x0b\x32\x1d.wandb_internal.RunExitRecordH\x00\x12,\n\x05\x66inal\x18\x14 \x01(\x0b\x32\x1b.wandb_internal.FinalRecordH\x00\x12.\n\x06header\x18\x15 \x01(\x0b\x32\x1c.wandb_internal.HeaderRecordH\x00\x12.\n\x06\x66ooter\x18\x16 \x01(\x0b\x32\x1c.wandb_internal.FooterRecordH\x00\x12\x39\n\npreempting\x18\x17 \x01(\x0b\x32#.wandb_internal.RunPreemptingRecordH\x00\x12\x34\n\x12noop_link_artifact\x18\x18 \x01(\x0b\x32\x16.google.protobuf.EmptyH\x00\x12\x39\n\x0cuse_artifact\x18\x19 \x01(\x0b\x32!.wandb_internal.UseArtifactRecordH\x00\x12\x38\n\x0b\x65nvironment\x18\x1a \x01(\x0b\x32!.wandb_internal.EnvironmentRecordH\x00\x12*\n\x07request\x18\x64 \x01(\x0b\x32\x17.wandb_internal.RequestH\x00\x12(\n\x07\x63ontrol\x18\x10 \x01(\x0b\x32\x17.wandb_internal.Control\x12\x0c\n\x04uuid\x18\x13 \x01(\t\x12+\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1b.wandb_internal._RecordInfoB\r\n\x0brecord_type\"\xa8\x01\n\x07\x43ontrol\x12\x10\n\x08req_resp\x18\x01 \x01(\x08\x12\r\n\x05local\x18\x02 \x01(\x08\x12\x10\n\x08relay_id\x18\x03 \x01(\t\x12\x14\n\x0cmailbox_slot\x18\x04 \x01(\t\x12\x13\n\x0b\x61lways_send\x18\x05 \x01(\x08\x12\x14\n\x0c\x66low_control\x18\x06 \x01(\x08\x12\x12\n\nend_offset\x18\x07 \x01(\x03\x12\x15\n\rconnection_id\x18\x08 \x01(\t\"\xf3\x03\n\x06Result\x12\x35\n\nrun_result\x18\x11 \x01(\x0b\x32\x1f.wandb_internal.RunUpdateResultH\x00\x12\x34\n\x0b\x65xit_result\x18\x12 \x01(\x0b\x32\x1d.wandb_internal.RunExitResultH\x00\x12\x33\n\nlog_result\x18\x14 \x01(\x0b\x32\x1d.wandb_internal.HistoryResultH\x00\x12\x37\n\x0esummary_result\x18\x15 \x01(\x0b\x32\x1d.wandb_internal.SummaryResultH\x00\x12\x35\n\routput_result\x18\x16 \x01(\x0b\x32\x1c.wandb_internal.OutputResultH\x00\x12\x35\n\rconfig_result\x18\x17 \x01(\x0b\x32\x1c.wandb_internal.ConfigResultH\x00\x12,\n\x08response\x18\x64 \x01(\x0b\x32\x18.wandb_internal.ResponseH\x00\x12(\n\x07\x63ontrol\x18\x10 \x01(\x0b\x32\x17.wandb_internal.Control\x12\x0c\n\x04uuid\x18\x18 \x01(\t\x12+\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1b.wandb_internal._ResultInfoB\r\n\x0bresult_type\":\n\x0b\x46inalRecord\x12+\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1b.wandb_internal._RecordInfo\"b\n\x0bVersionInfo\x12\x10\n\x08producer\x18\x01 \x01(\t\x12\x14\n\x0cmin_consumer\x18\x02 \x01(\t\x12+\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1b.wandb_internal._RecordInfo\"n\n\x0cHeaderRecord\x12\x31\n\x0cversion_info\x18\x01 \x01(\x0b\x32\x1b.wandb_internal.VersionInfo\x12+\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1b.wandb_internal._RecordInfo\";\n\x0c\x46ooterRecord\x12+\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1b.wandb_internal._RecordInfo\"9\n\x0b\x42ranchPoint\x12\x0b\n\x03run\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\x01\x12\x0e\n\x06metric\x18\x03 \x01(\t\"\x91\x05\n\tRunRecord\x12\x0e\n\x06run_id\x18\x01 \x01(\t\x12\x0e\n\x06\x65ntity\x18\x02 \x01(\t\x12\x0f\n\x07project\x18\x03 \x01(\t\x12,\n\x06\x63onfig\x18\x04 \x01(\x0b\x32\x1c.wandb_internal.ConfigRecord\x12.\n\x07summary\x18\x05 \x01(\x0b\x32\x1d.wandb_internal.SummaryRecord\x12\x11\n\trun_group\x18\x06 \x01(\t\x12\x10\n\x08job_type\x18\x07 \x01(\t\x12\x14\n\x0c\x64isplay_name\x18\x08 \x01(\t\x12\r\n\x05notes\x18\t \x01(\t\x12\x0c\n\x04tags\x18\n \x03(\t\x12\x30\n\x08settings\x18\x0b \x01(\x0b\x32\x1e.wandb_internal.SettingsRecord\x12\x10\n\x08sweep_id\x18\x0c \x01(\t\x12\x0c\n\x04host\x18\r \x01(\t\x12\x15\n\rstarting_step\x18\x0e \x01(\x03\x12\x12\n\nstorage_id\x18\x10 \x01(\t\x12.\n\nstart_time\x18\x11 \x01(\x0b\x32\x1a.google.protobuf.Timestamp\x12\x0f\n\x07resumed\x18\x12 \x01(\x08\x12\x32\n\ttelemetry\x18\x13 \x01(\x0b\x32\x1f.wandb_internal.TelemetryRecord\x12\x0f\n\x07runtime\x18\x14 \x01(\x05\x12*\n\x03git\x18\x15 \x01(\x0b\x32\x1d.wandb_internal.GitRepoRecord\x12\x0e\n\x06\x66orked\x18\x16 \x01(\x08\x12\x31\n\x0c\x62ranch_point\x18\x17 \x01(\x0b\x32\x1b.wandb_internal.BranchPoint\x12+\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1b.wandb_internal._RecordInfo\";\n\rGitRepoRecord\x12\x1a\n\nremote_url\x18\x01 \x01(\tR\x06remote\x12\x0e\n\x06\x63ommit\x18\x02 \x01(\t\"c\n\x0fRunUpdateResult\x12&\n\x03run\x18\x01 \x01(\x0b\x32\x19.wandb_internal.RunRecord\x12(\n\x05\x65rror\x18\x02 \x01(\x0b\x32\x19.wandb_internal.ErrorInfo\"\xac\x01\n\tErrorInfo\x12\x0f\n\x07message\x18\x01 \x01(\t\x12\x31\n\x04\x63ode\x18\x02 \x01(\x0e\x32#.wandb_internal.ErrorInfo.ErrorCode\"[\n\tErrorCode\x12\x0b\n\x07UNKNOWN\x10\x00\x12\x11\n\rCOMMUNICATION\x10\x01\x12\x12\n\x0e\x41UTHENTICATION\x10\x02\x12\t\n\x05USAGE\x10\x03\x12\x0f\n\x0bUNSUPPORTED\x10\x04\"v\n\rRunExitRecord\x12\x11\n\texit_code\x18\x01 \x01(\x05\x12\x14\n\x0cnot_complete\x18\x03 \x01(\x08\x12\x0f\n\x07runtime\x18\x02 \x01(\x05\x12+\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1b.wandb_internal._RecordInfo\"\x0f\n\rRunExitResult\"B\n\x13RunPreemptingRecord\x12+\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1b.wandb_internal._RecordInfo\"\x15\n\x13RunPreemptingResult\"i\n\x0eSettingsRecord\x12*\n\x04item\x18\x01 \x03(\x0b\x32\x1c.wandb_internal.SettingsItem\x12+\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1b.wandb_internal._RecordInfo\"/\n\x0cSettingsItem\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\x12\n\nvalue_json\x18\x10 \x01(\t\"\x1a\n\x0bHistoryStep\x12\x0b\n\x03num\x18\x01 \x01(\x03\"\x92\x01\n\rHistoryRecord\x12)\n\x04item\x18\x01 \x03(\x0b\x32\x1b.wandb_internal.HistoryItem\x12)\n\x04step\x18\x02 \x01(\x0b\x32\x1b.wandb_internal.HistoryStep\x12+\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1b.wandb_internal._RecordInfo\"B\n\x0bHistoryItem\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\x12\n\nnested_key\x18\x02 \x03(\t\x12\x12\n\nvalue_json\x18\x10 \x01(\t\"\x0f\n\rHistoryResult\"\xdc\x01\n\x0cOutputRecord\x12<\n\x0boutput_type\x18\x01 \x01(\x0e\x32\'.wandb_internal.OutputRecord.OutputType\x12-\n\ttimestamp\x18\x02 \x01(\x0b\x32\x1a.google.protobuf.Timestamp\x12\x0c\n\x04line\x18\x03 \x01(\t\x12+\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1b.wandb_internal._RecordInfo\"$\n\nOutputType\x12\n\n\x06STDERR\x10\x00\x12\n\n\x06STDOUT\x10\x01\"\x0e\n\x0cOutputResult\"\xe2\x01\n\x0fOutputRawRecord\x12?\n\x0boutput_type\x18\x01 \x01(\x0e\x32*.wandb_internal.OutputRawRecord.OutputType\x12-\n\ttimestamp\x18\x02 \x01(\x0b\x32\x1a.google.protobuf.Timestamp\x12\x0c\n\x04line\x18\x03 \x01(\t\x12+\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1b.wandb_internal._RecordInfo\"$\n\nOutputType\x12\n\n\x06STDERR\x10\x00\x12\n\n\x06STDOUT\x10\x01\"\x11\n\x0fOutputRawResult\"\xb4\x03\n\x0cMetricRecord\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x11\n\tglob_name\x18\x02 \x01(\t\x12\x13\n\x0bstep_metric\x18\x04 \x01(\t\x12\x19\n\x11step_metric_index\x18\x05 \x01(\x05\x12.\n\x07options\x18\x06 \x01(\x0b\x32\x1d.wandb_internal.MetricOptions\x12.\n\x07summary\x18\x07 \x01(\x0b\x32\x1d.wandb_internal.MetricSummary\x12\x35\n\x04goal\x18\x08 \x01(\x0e\x32\'.wandb_internal.MetricRecord.MetricGoal\x12/\n\x08_control\x18\t \x01(\x0b\x32\x1d.wandb_internal.MetricControl\x12\x1a\n\x12\x65xpanded_from_glob\x18\n \x01(\x08\x12+\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1b.wandb_internal._RecordInfo\"B\n\nMetricGoal\x12\x0e\n\nGOAL_UNSET\x10\x00\x12\x11\n\rGOAL_MINIMIZE\x10\x01\x12\x11\n\rGOAL_MAXIMIZE\x10\x02\"\x0e\n\x0cMetricResult\"C\n\rMetricOptions\x12\x11\n\tstep_sync\x18\x01 \x01(\x08\x12\x0e\n\x06hidden\x18\x02 \x01(\x08\x12\x0f\n\x07\x64\x65\x66ined\x18\x03 \x01(\x08\"\"\n\rMetricControl\x12\x11\n\toverwrite\x18\x01 \x01(\x08\"~\n\rMetricSummary\x12\x0b\n\x03min\x18\x01 \x01(\x08\x12\x0b\n\x03max\x18\x02 \x01(\x08\x12\x0c\n\x04mean\x18\x03 \x01(\x08\x12\x0c\n\x04\x62\x65st\x18\x04 \x01(\x08\x12\x0c\n\x04last\x18\x05 \x01(\x08\x12\x0c\n\x04none\x18\x06 \x01(\x08\x12\x0c\n\x04\x63opy\x18\x07 \x01(\x08\x12\r\n\x05\x66irst\x18\x08 \x01(\x08\"\x93\x01\n\x0c\x43onfigRecord\x12*\n\x06update\x18\x01 \x03(\x0b\x32\x1a.wandb_internal.ConfigItem\x12*\n\x06remove\x18\x02 \x03(\x0b\x32\x1a.wandb_internal.ConfigItem\x12+\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1b.wandb_internal._RecordInfo\"A\n\nConfigItem\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\x12\n\nnested_key\x18\x02 \x03(\t\x12\x12\n\nvalue_json\x18\x10 \x01(\t\"\x0e\n\x0c\x43onfigResult\"\x96\x01\n\rSummaryRecord\x12+\n\x06update\x18\x01 \x03(\x0b\x32\x1b.wandb_internal.SummaryItem\x12+\n\x06remove\x18\x02 \x03(\x0b\x32\x1b.wandb_internal.SummaryItem\x12+\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1b.wandb_internal._RecordInfo\"B\n\x0bSummaryItem\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\x12\n\nnested_key\x18\x02 \x03(\t\x12\x12\n\nvalue_json\x18\x10 \x01(\t\"\x0f\n\rSummaryResult\"d\n\x0b\x46ilesRecord\x12(\n\x05\x66iles\x18\x01 \x03(\x0b\x32\x19.wandb_internal.FilesItem\x12+\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1b.wandb_internal._RecordInfo\"\xec\x01\n\tFilesItem\x12\x0c\n\x04path\x18\x01 \x01(\t\x12\x34\n\x06policy\x18\x02 \x01(\x0e\x32$.wandb_internal.FilesItem.PolicyType\x12\x30\n\x04type\x18\x03 \x01(\x0e\x32\".wandb_internal.FilesItem.FileType\"(\n\nPolicyType\x12\x07\n\x03NOW\x10\x00\x12\x07\n\x03\x45ND\x10\x01\x12\x08\n\x04LIVE\x10\x02\"9\n\x08\x46ileType\x12\t\n\x05OTHER\x10\x00\x12\t\n\x05WANDB\x10\x01\x12\t\n\x05MEDIA\x10\x02\x12\x0c\n\x08\x41RTIFACT\x10\x03J\x04\x08\x10\x10\x11\"\r\n\x0b\x46ilesResult\"\xe6\x01\n\x0bStatsRecord\x12\x39\n\nstats_type\x18\x01 \x01(\x0e\x32%.wandb_internal.StatsRecord.StatsType\x12-\n\ttimestamp\x18\x02 \x01(\x0b\x32\x1a.google.protobuf.Timestamp\x12\'\n\x04item\x18\x03 \x03(\x0b\x32\x19.wandb_internal.StatsItem\x12+\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1b.wandb_internal._RecordInfo\"\x17\n\tStatsType\x12\n\n\x06SYSTEM\x10\x00\",\n\tStatsItem\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\x12\n\nvalue_json\x18\x10 \x01(\t\"\xe7\x03\n\x0e\x41rtifactRecord\x12\x0e\n\x06run_id\x18\x01 \x01(\t\x12\x0f\n\x07project\x18\x02 \x01(\t\x12\x0e\n\x06\x65ntity\x18\x03 \x01(\t\x12\x0c\n\x04type\x18\x04 \x01(\t\x12\x0c\n\x04name\x18\x05 \x01(\t\x12\x0e\n\x06\x64igest\x18\x06 \x01(\t\x12\x13\n\x0b\x64\x65scription\x18\x07 \x01(\t\x12\x10\n\x08metadata\x18\x08 \x01(\t\x12\x14\n\x0cuser_created\x18\t \x01(\x08\x12\x18\n\x10use_after_commit\x18\n \x01(\x08\x12\x0f\n\x07\x61liases\x18\x0b \x03(\t\x12\x32\n\x08manifest\x18\x0c \x01(\x0b\x32 .wandb_internal.ArtifactManifest\x12\x16\n\x0e\x64istributed_id\x18\r \x01(\t\x12\x10\n\x08\x66inalize\x18\x0e \x01(\x08\x12\x11\n\tclient_id\x18\x0f \x01(\t\x12\x1a\n\x12sequence_client_id\x18\x10 \x01(\t\x12\x0f\n\x07\x62\x61se_id\x18\x11 \x01(\t\x12\x1c\n\x14ttl_duration_seconds\x18\x12 \x01(\x03\x12\x0c\n\x04tags\x18\x13 \x03(\t\x12\x19\n\x11incremental_beta1\x18\x64 \x01(\x08\x12+\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1b.wandb_internal._RecordInfo\"\xd8\x01\n\x10\x41rtifactManifest\x12\x0f\n\x07version\x18\x01 \x01(\x05\x12\x16\n\x0estorage_policy\x18\x02 \x01(\t\x12\x46\n\x15storage_policy_config\x18\x03 \x03(\x0b\x32\'.wandb_internal.StoragePolicyConfigItem\x12\x37\n\x08\x63ontents\x18\x04 \x03(\x0b\x32%.wandb_internal.ArtifactManifestEntry\x12\x1a\n\x12manifest_file_path\x18\x05 \x01(\t\"\xcf\x01\n\x15\x41rtifactManifestEntry\x12\x0c\n\x04path\x18\x01 \x01(\t\x12\x0e\n\x06\x64igest\x18\x02 \x01(\t\x12\x0b\n\x03ref\x18\x03 \x01(\t\x12\x0c\n\x04size\x18\x04 \x01(\x03\x12\x10\n\x08mimetype\x18\x05 \x01(\t\x12\x12\n\nlocal_path\x18\x06 \x01(\t\x12\x19\n\x11\x62irth_artifact_id\x18\x07 \x01(\t\x12\x12\n\nskip_cache\x18\x08 \x01(\x08\x12(\n\x05\x65xtra\x18\x10 \x03(\x0b\x32\x19.wandb_internal.ExtraItem\",\n\tExtraItem\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\x12\n\nvalue_json\x18\x02 \x01(\t\":\n\x17StoragePolicyConfigItem\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\x12\n\nvalue_json\x18\x02 \x01(\t\"\x10\n\x0e\x41rtifactResult\"\x14\n\x12LinkArtifactResult\"\xf0\x01\n\x13LinkArtifactRequest\x12\x11\n\tclient_id\x18\x01 \x01(\t\x12\x11\n\tserver_id\x18\x02 \x01(\t\x12\x16\n\x0eportfolio_name\x18\x03 \x01(\t\x12\x18\n\x10portfolio_entity\x18\x04 \x01(\t\x12\x19\n\x11portfolio_project\x18\x05 \x01(\t\x12\x19\n\x11portfolio_aliases\x18\x06 \x03(\t\x12\x1e\n\x16portfolio_organization\x18\x07 \x01(\t\x12+\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1b.wandb_internal._RecordInfo\"[\n\x14LinkArtifactResponse\x12\x15\n\rerror_message\x18\x01 \x01(\t\x12\x1a\n\rversion_index\x18\x02 \x01(\x05H\x00\x88\x01\x01\x42\x10\n\x0e_version_index\"h\n\x08TBRecord\x12+\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1b.wandb_internal._RecordInfo\x12\x0f\n\x07log_dir\x18\x01 \x01(\t\x12\x10\n\x08root_dir\x18\x03 \x01(\t\x12\x0c\n\x04save\x18\x02 \x01(\x08\"\n\n\x08TBResult\"}\n\x0b\x41lertRecord\x12\r\n\x05title\x18\x01 \x01(\t\x12\x0c\n\x04text\x18\x02 \x01(\t\x12\r\n\x05level\x18\x03 \x01(\t\x12\x15\n\rwait_duration\x18\x04 \x01(\x03\x12+\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1b.wandb_internal._RecordInfo\"\r\n\x0b\x41lertResult\"\xf4\x10\n\x07Request\x12\x38\n\x0bstop_status\x18\x01 \x01(\x0b\x32!.wandb_internal.StopStatusRequestH\x00\x12>\n\x0enetwork_status\x18\x02 \x01(\x0b\x32$.wandb_internal.NetworkStatusRequestH\x00\x12-\n\x05\x64\x65\x66\x65r\x18\x03 \x01(\x0b\x32\x1c.wandb_internal.DeferRequestH\x00\x12\x38\n\x0bget_summary\x18\x04 \x01(\x0b\x32!.wandb_internal.GetSummaryRequestH\x00\x12-\n\x05login\x18\x05 \x01(\x0b\x32\x1c.wandb_internal.LoginRequestH\x00\x12-\n\x05pause\x18\x06 \x01(\x0b\x32\x1c.wandb_internal.PauseRequestH\x00\x12/\n\x06resume\x18\x07 \x01(\x0b\x32\x1d.wandb_internal.ResumeRequestH\x00\x12\x34\n\tpoll_exit\x18\x08 \x01(\x0b\x32\x1f.wandb_internal.PollExitRequestH\x00\x12@\n\x0fsampled_history\x18\t \x01(\x0b\x32%.wandb_internal.SampledHistoryRequestH\x00\x12@\n\x0fpartial_history\x18\n \x01(\x0b\x32%.wandb_internal.PartialHistoryRequestH\x00\x12\x34\n\trun_start\x18\x0b \x01(\x0b\x32\x1f.wandb_internal.RunStartRequestH\x00\x12<\n\rcheck_version\x18\x0c \x01(\x0b\x32#.wandb_internal.CheckVersionRequestH\x00\x12:\n\x0clog_artifact\x18\r \x01(\x0b\x32\".wandb_internal.LogArtifactRequestH\x00\x12\x44\n\x11\x64ownload_artifact\x18\x0e \x01(\x0b\x32\'.wandb_internal.DownloadArtifactRequestH\x00\x12\x35\n\tkeepalive\x18\x11 \x01(\x0b\x32 .wandb_internal.KeepaliveRequestH\x00\x12\x36\n\nrun_status\x18\x14 \x01(\x0b\x32 .wandb_internal.RunStatusRequestH\x00\x12/\n\x06\x63\x61ncel\x18\x15 \x01(\x0b\x32\x1d.wandb_internal.CancelRequestH\x00\x12\x44\n\x11internal_messages\x18\x17 \x01(\x0b\x32\'.wandb_internal.InternalMessagesRequestH\x00\x12@\n\x0fpython_packages\x18\x18 \x01(\x0b\x32%.wandb_internal.PythonPackagesRequestH\x00\x12\x33\n\x08shutdown\x18@ \x01(\x0b\x32\x1f.wandb_internal.ShutdownRequestH\x00\x12/\n\x06\x61ttach\x18\x41 \x01(\x0b\x32\x1d.wandb_internal.AttachRequestH\x00\x12/\n\x06status\x18\x42 \x01(\x0b\x32\x1d.wandb_internal.StatusRequestH\x00\x12\x38\n\x0bserver_info\x18\x43 \x01(\x0b\x32!.wandb_internal.ServerInfoRequestH\x00\x12\x38\n\x0bsender_mark\x18\x44 \x01(\x0b\x32!.wandb_internal.SenderMarkRequestH\x00\x12\x38\n\x0bsender_read\x18\x45 \x01(\x0b\x32!.wandb_internal.SenderReadRequestH\x00\x12<\n\rstatus_report\x18\x46 \x01(\x0b\x32#.wandb_internal.StatusReportRequestH\x00\x12>\n\x0esummary_record\x18G \x01(\x0b\x32$.wandb_internal.SummaryRecordRequestH\x00\x12\x42\n\x10telemetry_record\x18H \x01(\x0b\x32&.wandb_internal.TelemetryRecordRequestH\x00\x12\x32\n\x08job_info\x18I \x01(\x0b\x32\x1e.wandb_internal.JobInfoRequestH\x00\x12\x45\n\x12get_system_metrics\x18J \x01(\x0b\x32\'.wandb_internal.GetSystemMetricsRequestH\x00\x12\x34\n\tjob_input\x18M \x01(\x0b\x32\x1f.wandb_internal.JobInputRequestH\x00\x12<\n\rlink_artifact\x18N \x01(\x0b\x32#.wandb_internal.LinkArtifactRequestH\x00\x12\x38\n\x0bsync_finish\x18Q \x01(\x0b\x32!.wandb_internal.SyncFinishRequestH\x00\x12;\n\noperations\x18R \x01(\x0b\x32%.wandb_internal.OperationStatsRequestH\x00\x12\x43\n\x11probe_system_info\x18S \x01(\x0b\x32&.wandb_internal.ProbeSystemInfoRequestH\x00\x12\x39\n\x0btest_inject\x18\xe8\x07 \x01(\x0b\x32!.wandb_internal.TestInjectRequestH\x00\x42\x0e\n\x0crequest_typeJ\x04\x08\x12\x10\x13J\x04\x08\x16\x10\x17J\x04\x08K\x10LJ\x04\x08L\x10MJ\x04\x08O\x10PJ\x04\x08P\x10Q\"\x83\r\n\x08Response\x12?\n\x12keepalive_response\x18\x12 \x01(\x0b\x32!.wandb_internal.KeepaliveResponseH\x00\x12\x42\n\x14stop_status_response\x18\x13 \x01(\x0b\x32\".wandb_internal.StopStatusResponseH\x00\x12H\n\x17network_status_response\x18\x14 \x01(\x0b\x32%.wandb_internal.NetworkStatusResponseH\x00\x12\x37\n\x0elogin_response\x18\x18 \x01(\x0b\x32\x1d.wandb_internal.LoginResponseH\x00\x12\x42\n\x14get_summary_response\x18\x19 \x01(\x0b\x32\".wandb_internal.GetSummaryResponseH\x00\x12>\n\x12poll_exit_response\x18\x1a \x01(\x0b\x32 .wandb_internal.PollExitResponseH\x00\x12J\n\x18sampled_history_response\x18\x1b \x01(\x0b\x32&.wandb_internal.SampledHistoryResponseH\x00\x12>\n\x12run_start_response\x18\x1c \x01(\x0b\x32 .wandb_internal.RunStartResponseH\x00\x12\x46\n\x16\x63heck_version_response\x18\x1d \x01(\x0b\x32$.wandb_internal.CheckVersionResponseH\x00\x12\x44\n\x15log_artifact_response\x18\x1e \x01(\x0b\x32#.wandb_internal.LogArtifactResponseH\x00\x12N\n\x1a\x64ownload_artifact_response\x18\x1f \x01(\x0b\x32(.wandb_internal.DownloadArtifactResponseH\x00\x12@\n\x13run_status_response\x18# \x01(\x0b\x32!.wandb_internal.RunStatusResponseH\x00\x12\x39\n\x0f\x63\x61ncel_response\x18$ \x01(\x0b\x32\x1e.wandb_internal.CancelResponseH\x00\x12N\n\x1ainternal_messages_response\x18% \x01(\x0b\x32(.wandb_internal.InternalMessagesResponseH\x00\x12=\n\x11shutdown_response\x18@ \x01(\x0b\x32 .wandb_internal.ShutdownResponseH\x00\x12\x39\n\x0f\x61ttach_response\x18\x41 \x01(\x0b\x32\x1e.wandb_internal.AttachResponseH\x00\x12\x39\n\x0fstatus_response\x18\x42 \x01(\x0b\x32\x1e.wandb_internal.StatusResponseH\x00\x12\x42\n\x14server_info_response\x18\x43 \x01(\x0b\x32\".wandb_internal.ServerInfoResponseH\x00\x12<\n\x11job_info_response\x18\x44 \x01(\x0b\x32\x1f.wandb_internal.JobInfoResponseH\x00\x12O\n\x1bget_system_metrics_response\x18\x45 \x01(\x0b\x32(.wandb_internal.GetSystemMetricsResponseH\x00\x12\x46\n\x16link_artifact_response\x18G \x01(\x0b\x32$.wandb_internal.LinkArtifactResponseH\x00\x12\x35\n\rsync_response\x18\x46 \x01(\x0b\x32\x1c.wandb_internal.SyncResponseH\x00\x12\x45\n\x13operations_response\x18J \x01(\x0b\x32&.wandb_internal.OperationStatsResponseH\x00\x12\x43\n\x14test_inject_response\x18\xe8\x07 \x01(\x0b\x32\".wandb_internal.TestInjectResponseH\x00\x42\x0f\n\rresponse_typeJ\x04\x08 \x10!J\x04\x08H\x10IJ\x04\x08I\x10J\"\xc0\x02\n\x0c\x44\x65\x66\x65rRequest\x12\x36\n\x05state\x18\x01 \x01(\x0e\x32\'.wandb_internal.DeferRequest.DeferState\"\xf7\x01\n\nDeferState\x12\t\n\x05\x42\x45GIN\x10\x00\x12\r\n\tFLUSH_RUN\x10\x01\x12\x0f\n\x0b\x46LUSH_STATS\x10\x02\x12\x19\n\x15\x46LUSH_PARTIAL_HISTORY\x10\x03\x12\x0c\n\x08\x46LUSH_TB\x10\x04\x12\r\n\tFLUSH_SUM\x10\x05\x12\x13\n\x0f\x46LUSH_DEBOUNCER\x10\x06\x12\x10\n\x0c\x46LUSH_OUTPUT\x10\x07\x12\r\n\tFLUSH_JOB\x10\x08\x12\r\n\tFLUSH_DIR\x10\t\x12\x0c\n\x08\x46LUSH_FP\x10\n\x12\x0b\n\x07JOIN_FP\x10\x0b\x12\x0c\n\x08\x46LUSH_FS\x10\x0c\x12\x0f\n\x0b\x46LUSH_FINAL\x10\r\x12\x07\n\x03\x45ND\x10\x0e\"<\n\x0cPauseRequest\x12,\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1c.wandb_internal._RequestInfo\"\x0f\n\rPauseResponse\"=\n\rResumeRequest\x12,\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1c.wandb_internal._RequestInfo\"\x10\n\x0eResumeResponse\"M\n\x0cLoginRequest\x12\x0f\n\x07\x61pi_key\x18\x01 \x01(\t\x12,\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1c.wandb_internal._RequestInfo\"&\n\rLoginResponse\x12\x15\n\ractive_entity\x18\x01 \x01(\t\"A\n\x11GetSummaryRequest\x12,\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1c.wandb_internal._RequestInfo\"?\n\x12GetSummaryResponse\x12)\n\x04item\x18\x01 \x03(\x0b\x32\x1b.wandb_internal.SummaryItem\"G\n\x17GetSystemMetricsRequest\x12,\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1c.wandb_internal._RequestInfo\"R\n\x12SystemMetricSample\x12-\n\ttimestamp\x18\x01 \x01(\x0b\x32\x1a.google.protobuf.Timestamp\x12\r\n\x05value\x18\x02 \x01(\x02\"I\n\x13SystemMetricsBuffer\x12\x32\n\x06record\x18\x01 \x03(\x0b\x32\".wandb_internal.SystemMetricSample\"\xca\x01\n\x18GetSystemMetricsResponse\x12S\n\x0esystem_metrics\x18\x01 \x03(\x0b\x32;.wandb_internal.GetSystemMetricsResponse.SystemMetricsEntry\x1aY\n\x12SystemMetricsEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\x32\n\x05value\x18\x02 \x01(\x0b\x32#.wandb_internal.SystemMetricsBuffer:\x02\x38\x01\"=\n\rStatusRequest\x12,\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1c.wandb_internal._RequestInfo\")\n\x0eStatusResponse\x12\x17\n\x0frun_should_stop\x18\x01 \x01(\x08\"A\n\x11StopStatusRequest\x12,\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1c.wandb_internal._RequestInfo\"-\n\x12StopStatusResponse\x12\x17\n\x0frun_should_stop\x18\x01 \x01(\x08\"D\n\x14NetworkStatusRequest\x12,\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1c.wandb_internal._RequestInfo\"P\n\x15NetworkStatusResponse\x12\x37\n\x11network_responses\x18\x01 \x03(\x0b\x32\x1c.wandb_internal.HttpResponse\"D\n\x0cHttpResponse\x12\x18\n\x10http_status_code\x18\x01 \x01(\x05\x12\x1a\n\x12http_response_text\x18\x02 \x01(\t\"G\n\x17InternalMessagesRequest\x12,\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1c.wandb_internal._RequestInfo\"N\n\x18InternalMessagesResponse\x12\x32\n\x08messages\x18\x01 \x01(\x0b\x32 .wandb_internal.InternalMessages\"#\n\x10InternalMessages\x12\x0f\n\x07warning\x18\x01 \x03(\t\"?\n\x0fPollExitRequest\x12,\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1c.wandb_internal._RequestInfo\"\xf5\x01\n\x10PollExitResponse\x12\x0c\n\x04\x64one\x18\x01 \x01(\x08\x12\x32\n\x0b\x65xit_result\x18\x02 \x01(\x0b\x32\x1d.wandb_internal.RunExitResult\x12\x35\n\x0cpusher_stats\x18\x03 \x01(\x0b\x32\x1f.wandb_internal.FilePusherStats\x12/\n\x0b\x66ile_counts\x18\x04 \x01(\x0b\x32\x1a.wandb_internal.FileCounts\x12\x37\n\x0foperation_stats\x18\x05 \x01(\x0b\x32\x1e.wandb_internal.OperationStats\"E\n\x15OperationStatsRequest\x12,\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1c.wandb_internal._RequestInfo\"Q\n\x16OperationStatsResponse\x12\x37\n\x0foperation_stats\x18\x01 \x01(\x0b\x32\x1e.wandb_internal.OperationStats\"h\n\x0eOperationStats\x12\r\n\x05label\x18\x03 \x01(\t\x12-\n\noperations\x18\x01 \x03(\x0b\x32\x19.wandb_internal.Operation\x12\x18\n\x10total_operations\x18\x02 \x01(\x03\"\x87\x01\n\tOperation\x12\x0c\n\x04\x64\x65sc\x18\x01 \x01(\t\x12\x17\n\x0fruntime_seconds\x18\x02 \x01(\x01\x12\x10\n\x08progress\x18\x03 \x01(\t\x12\x14\n\x0c\x65rror_status\x18\x04 \x01(\t\x12+\n\x08subtasks\x18\x05 \x03(\x0b\x32\x19.wandb_internal.Operation\"\x13\n\x11SenderMarkRequest\"\x13\n\x11SyncFinishRequest\"E\n\x0cSyncResponse\x12\x0b\n\x03url\x18\x01 \x01(\t\x12(\n\x05\x65rror\x18\x02 \x01(\x0b\x32\x19.wandb_internal.ErrorInfo\"?\n\x11SenderReadRequest\x12\x14\n\x0cstart_offset\x18\x01 \x01(\x03\x12\x14\n\x0c\x66inal_offset\x18\x02 \x01(\x03\"m\n\x13StatusReportRequest\x12\x12\n\nrecord_num\x18\x01 \x01(\x03\x12\x13\n\x0bsent_offset\x18\x02 \x01(\x03\x12-\n\tsync_time\x18\x03 \x01(\x0b\x32\x1a.google.protobuf.Timestamp\"F\n\x14SummaryRecordRequest\x12.\n\x07summary\x18\x01 \x01(\x0b\x32\x1d.wandb_internal.SummaryRecord\"L\n\x16TelemetryRecordRequest\x12\x32\n\ttelemetry\x18\x01 \x01(\x0b\x32\x1f.wandb_internal.TelemetryRecord\"A\n\x11ServerInfoRequest\x12,\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1c.wandb_internal._RequestInfo\"|\n\x12ServerInfoResponse\x12-\n\nlocal_info\x18\x01 \x01(\x0b\x32\x19.wandb_internal.LocalInfo\x12\x37\n\x0fserver_messages\x18\x02 \x01(\x0b\x32\x1e.wandb_internal.ServerMessages\"=\n\x0eServerMessages\x12+\n\x04item\x18\x01 \x03(\x0b\x32\x1d.wandb_internal.ServerMessage\"e\n\rServerMessage\x12\x12\n\nplain_text\x18\x01 \x01(\t\x12\x10\n\x08utf_text\x18\x02 \x01(\t\x12\x11\n\thtml_text\x18\x03 \x01(\t\x12\x0c\n\x04type\x18\x04 \x01(\t\x12\r\n\x05level\x18\x05 \x01(\x05\"c\n\nFileCounts\x12\x13\n\x0bwandb_count\x18\x01 \x01(\x05\x12\x13\n\x0bmedia_count\x18\x02 \x01(\x05\x12\x16\n\x0e\x61rtifact_count\x18\x03 \x01(\x05\x12\x13\n\x0bother_count\x18\x04 \x01(\x05\"U\n\x0f\x46ilePusherStats\x12\x16\n\x0euploaded_bytes\x18\x01 \x01(\x03\x12\x13\n\x0btotal_bytes\x18\x02 \x01(\x03\x12\x15\n\rdeduped_bytes\x18\x03 \x01(\x03\"\x1e\n\rFilesUploaded\x12\r\n\x05\x66iles\x18\x01 \x03(\t\"\xf4\x01\n\x17\x46ileTransferInfoRequest\x12\x42\n\x04type\x18\x01 \x01(\x0e\x32\x34.wandb_internal.FileTransferInfoRequest.TransferType\x12\x0c\n\x04path\x18\x02 \x01(\t\x12\x0b\n\x03url\x18\x03 \x01(\t\x12\x0c\n\x04size\x18\x04 \x01(\x03\x12\x11\n\tprocessed\x18\x05 \x01(\x03\x12/\n\x0b\x66ile_counts\x18\x06 \x01(\x0b\x32\x1a.wandb_internal.FileCounts\"(\n\x0cTransferType\x12\n\n\x06Upload\x10\x00\x12\x0c\n\x08\x44ownload\x10\x01\"1\n\tLocalInfo\x12\x0f\n\x07version\x18\x01 \x01(\t\x12\x13\n\x0bout_of_date\x18\x02 \x01(\x08\"?\n\x0fShutdownRequest\x12,\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1c.wandb_internal._RequestInfo\"\x12\n\x10ShutdownResponse\"P\n\rAttachRequest\x12\x11\n\tattach_id\x18\x14 \x01(\t\x12,\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1c.wandb_internal._RequestInfo\"b\n\x0e\x41ttachResponse\x12&\n\x03run\x18\x01 \x01(\x0b\x32\x19.wandb_internal.RunRecord\x12(\n\x05\x65rror\x18\x02 \x01(\x0b\x32\x19.wandb_internal.ErrorInfo\"\xd5\x02\n\x11TestInjectRequest\x12\x13\n\x0bhandler_exc\x18\x01 \x01(\x08\x12\x14\n\x0chandler_exit\x18\x02 \x01(\x08\x12\x15\n\rhandler_abort\x18\x03 \x01(\x08\x12\x12\n\nsender_exc\x18\x04 \x01(\x08\x12\x13\n\x0bsender_exit\x18\x05 \x01(\x08\x12\x14\n\x0csender_abort\x18\x06 \x01(\x08\x12\x0f\n\x07req_exc\x18\x07 \x01(\x08\x12\x10\n\x08req_exit\x18\x08 \x01(\x08\x12\x11\n\treq_abort\x18\t \x01(\x08\x12\x10\n\x08resp_exc\x18\n \x01(\x08\x12\x11\n\tresp_exit\x18\x0b \x01(\x08\x12\x12\n\nresp_abort\x18\x0c \x01(\x08\x12\x10\n\x08msg_drop\x18\r \x01(\x08\x12\x10\n\x08msg_hang\x18\x0e \x01(\x08\x12,\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1c.wandb_internal._RequestInfo\"\x14\n\x12TestInjectResponse\"\x1e\n\rHistoryAction\x12\r\n\x05\x66lush\x18\x01 \x01(\x08\"\xca\x01\n\x15PartialHistoryRequest\x12)\n\x04item\x18\x01 \x03(\x0b\x32\x1b.wandb_internal.HistoryItem\x12)\n\x04step\x18\x02 \x01(\x0b\x32\x1b.wandb_internal.HistoryStep\x12-\n\x06\x61\x63tion\x18\x03 \x01(\x0b\x32\x1d.wandb_internal.HistoryAction\x12,\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1c.wandb_internal._RequestInfo\"\x18\n\x16PartialHistoryResponse\"E\n\x15SampledHistoryRequest\x12,\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1c.wandb_internal._RequestInfo\"_\n\x12SampledHistoryItem\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\x12\n\nnested_key\x18\x02 \x03(\t\x12\x14\n\x0cvalues_float\x18\x03 \x03(\x02\x12\x12\n\nvalues_int\x18\x04 \x03(\x03\"J\n\x16SampledHistoryResponse\x12\x30\n\x04item\x18\x01 \x03(\x0b\x32\".wandb_internal.SampledHistoryItem\"@\n\x10RunStatusRequest\x12,\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1c.wandb_internal._RequestInfo\"x\n\x11RunStatusResponse\x12\x18\n\x10sync_items_total\x18\x01 \x01(\x03\x12\x1a\n\x12sync_items_pending\x18\x02 \x01(\x03\x12-\n\tsync_time\x18\x03 \x01(\x0b\x32\x1a.google.protobuf.Timestamp\"g\n\x0fRunStartRequest\x12&\n\x03run\x18\x01 \x01(\x0b\x32\x19.wandb_internal.RunRecord\x12,\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1c.wandb_internal._RequestInfo\"\x12\n\x10RunStartResponse\"\\\n\x13\x43heckVersionRequest\x12\x17\n\x0f\x63urrent_version\x18\x01 \x01(\t\x12,\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1c.wandb_internal._RequestInfo\"]\n\x14\x43heckVersionResponse\x12\x17\n\x0fupgrade_message\x18\x01 \x01(\t\x12\x14\n\x0cyank_message\x18\x02 \x01(\t\x12\x16\n\x0e\x64\x65lete_message\x18\x03 \x01(\t\">\n\x0eJobInfoRequest\x12,\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1c.wandb_internal._RequestInfo\"6\n\x0fJobInfoResponse\x12\x12\n\nsequenceId\x18\x01 \x01(\t\x12\x0f\n\x07version\x18\x02 \x01(\t\"\x9f\x01\n\x12LogArtifactRequest\x12\x30\n\x08\x61rtifact\x18\x01 \x01(\x0b\x32\x1e.wandb_internal.ArtifactRecord\x12\x14\n\x0chistory_step\x18\x02 \x01(\x03\x12\x13\n\x0bstaging_dir\x18\x03 \x01(\t\x12,\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1c.wandb_internal._RequestInfo\"A\n\x13LogArtifactResponse\x12\x13\n\x0b\x61rtifact_id\x18\x01 \x01(\t\x12\x15\n\rerror_message\x18\x02 \x01(\t\"\xbe\x01\n\x17\x44ownloadArtifactRequest\x12\x13\n\x0b\x61rtifact_id\x18\x01 \x01(\t\x12\x15\n\rdownload_root\x18\x02 \x01(\t\x12 \n\x18\x61llow_missing_references\x18\x04 \x01(\x08\x12\x12\n\nskip_cache\x18\x05 \x01(\x08\x12\x13\n\x0bpath_prefix\x18\x06 \x01(\t\x12,\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1c.wandb_internal._RequestInfo\"1\n\x18\x44ownloadArtifactResponse\x12\x15\n\rerror_message\x18\x01 \x01(\t\"@\n\x10KeepaliveRequest\x12,\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1c.wandb_internal._RequestInfo\"\x13\n\x11KeepaliveResponse\"q\n\x0c\x41rtifactInfo\x12\x10\n\x08\x61rtifact\x18\x01 \x01(\t\x12\x12\n\nentrypoint\x18\x02 \x03(\t\x12\x10\n\x08notebook\x18\x03 \x01(\x08\x12\x15\n\rbuild_context\x18\x04 \x01(\t\x12\x12\n\ndockerfile\x18\x05 \x01(\t\")\n\x07GitInfo\x12\x0e\n\x06remote\x18\x01 \x01(\t\x12\x0e\n\x06\x63ommit\x18\x02 \x01(\t\"\x87\x01\n\tGitSource\x12)\n\x08git_info\x18\x01 \x01(\x0b\x32\x17.wandb_internal.GitInfo\x12\x12\n\nentrypoint\x18\x02 \x03(\t\x12\x10\n\x08notebook\x18\x03 \x01(\x08\x12\x15\n\rbuild_context\x18\x04 \x01(\t\x12\x12\n\ndockerfile\x18\x05 \x01(\t\"\x1c\n\x0bImageSource\x12\r\n\x05image\x18\x01 \x01(\t\"\x8c\x01\n\x06Source\x12&\n\x03git\x18\x01 \x01(\x0b\x32\x19.wandb_internal.GitSource\x12.\n\x08\x61rtifact\x18\x02 \x01(\x0b\x32\x1c.wandb_internal.ArtifactInfo\x12*\n\x05image\x18\x03 \x01(\x0b\x32\x1b.wandb_internal.ImageSource\"k\n\tJobSource\x12\x10\n\x08_version\x18\x01 \x01(\t\x12\x13\n\x0bsource_type\x18\x02 \x01(\t\x12&\n\x06source\x18\x03 \x01(\x0b\x32\x16.wandb_internal.Source\x12\x0f\n\x07runtime\x18\x04 \x01(\t\"V\n\x12PartialJobArtifact\x12\x10\n\x08job_name\x18\x01 \x01(\t\x12.\n\x0bsource_info\x18\x02 \x01(\x0b\x32\x19.wandb_internal.JobSource\"\x9d\x01\n\x11UseArtifactRecord\x12\n\n\x02id\x18\x01 \x01(\t\x12\x0c\n\x04type\x18\x02 \x01(\t\x12\x0c\n\x04name\x18\x03 \x01(\t\x12\x33\n\x07partial\x18\x04 \x01(\x0b\x32\".wandb_internal.PartialJobArtifact\x12+\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1b.wandb_internal._RecordInfo\"\x13\n\x11UseArtifactResult\"R\n\rCancelRequest\x12\x13\n\x0b\x63\x61ncel_slot\x18\x01 \x01(\t\x12,\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1c.wandb_internal._RequestInfo\"\x10\n\x0e\x43\x61ncelResponse\"\x18\n\x16ProbeSystemInfoRequest\"\'\n\x08\x44iskInfo\x12\r\n\x05total\x18\x01 \x01(\x04\x12\x0c\n\x04used\x18\x02 \x01(\x04\"\x1b\n\nMemoryInfo\x12\r\n\x05total\x18\x01 \x01(\x04\"/\n\x07\x43puInfo\x12\r\n\x05\x63ount\x18\x01 \x01(\r\x12\x15\n\rcount_logical\x18\x02 \x01(\r\"\xad\x01\n\tAppleInfo\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x12\n\necpu_cores\x18\x02 \x01(\r\x12\x12\n\npcpu_cores\x18\x03 \x01(\r\x12\x11\n\tgpu_cores\x18\x04 \x01(\r\x12\x11\n\tmemory_gb\x18\x05 \x01(\r\x12\x18\n\x10swap_total_bytes\x18\x06 \x01(\x04\x12\x17\n\x0fram_total_bytes\x18\x07 \x01(\x04\x12\x11\n\tmac_model\x18\x08 \x01(\t\"k\n\rGpuNvidiaInfo\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x14\n\x0cmemory_total\x18\x02 \x01(\x04\x12\x12\n\ncuda_cores\x18\x03 \x01(\r\x12\x14\n\x0c\x61rchitecture\x18\x04 \x01(\t\x12\x0c\n\x04uuid\x18\x05 \x01(\t\"\x89\x02\n\nGpuAmdInfo\x12\n\n\x02id\x18\x01 \x01(\t\x12\x11\n\tunique_id\x18\x02 \x01(\t\x12\x15\n\rvbios_version\x18\x03 \x01(\t\x12\x19\n\x11performance_level\x18\x04 \x01(\t\x12\x15\n\rgpu_overdrive\x18\x05 \x01(\t\x12\x1c\n\x14gpu_memory_overdrive\x18\x06 \x01(\t\x12\x11\n\tmax_power\x18\x07 \x01(\t\x12\x0e\n\x06series\x18\x08 \x01(\t\x12\r\n\x05model\x18\t \x01(\t\x12\x0e\n\x06vendor\x18\n \x01(\t\x12\x0b\n\x03sku\x18\x0b \x01(\t\x12\x12\n\nsclk_range\x18\x0c \x01(\t\x12\x12\n\nmclk_range\x18\r \x01(\t\"n\n\x0cTrainiumInfo\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x0e\n\x06vendor\x18\x02 \x01(\t\x12\x1b\n\x13neuron_device_count\x18\x03 \x01(\r\x12#\n\x1bneuroncore_per_device_count\x18\x04 \x01(\r\"Q\n\x07TPUInfo\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x0f\n\x07hbm_gib\x18\x02 \x01(\r\x12\x18\n\x10\x64\x65vices_per_chip\x18\x03 \x01(\r\x12\r\n\x05\x63ount\x18\x04 \x01(\r\"E\n\rCoreWeaveInfo\x12\x14\n\x0c\x63luster_name\x18\x01 \x01(\t\x12\x0e\n\x06org_id\x18\x02 \x01(\t\x12\x0e\n\x06region\x18\x03 \x01(\t\"\xa8\t\n\x11\x45nvironmentRecord\x12\n\n\x02os\x18\x01 \x01(\t\x12\x0e\n\x06python\x18\x02 \x01(\t\x12\x39\n\nstarted_at\x18\x03 \x01(\x0b\x32\x1a.google.protobuf.TimestampR\tstartedAt\x12\x0e\n\x06\x64ocker\x18\x04 \x01(\t\x12\x0c\n\x04\x61rgs\x18\x05 \x03(\t\x12\x0f\n\x07program\x18\x06 \x01(\t\x12\x1b\n\tcode_path\x18\x07 \x01(\tR\x08\x63odePath\x12&\n\x0f\x63ode_path_local\x18\x08 \x01(\tR\rcodePathLocal\x12*\n\x03git\x18\t \x01(\x0b\x32\x1d.wandb_internal.GitRepoRecord\x12\r\n\x05\x65mail\x18\n \x01(\t\x12\x0c\n\x04root\x18\x0b \x01(\t\x12\x0c\n\x04host\x18\x0c \x01(\t\x12\x10\n\x08username\x18\r \x01(\t\x12\x12\n\nexecutable\x18\x0e \x01(\t\x12\r\n\x05\x63olab\x18\x0f \x01(\t\x12\x1c\n\tcpu_count\x18\x10 \x01(\rR\tcpu_count\x12,\n\x11\x63pu_count_logical\x18\x11 \x01(\rR\x11\x63pu_count_logical\x12\x15\n\x08gpu_type\x18\x12 \x01(\tR\x03gpu\x12\x1c\n\tgpu_count\x18\x13 \x01(\rR\tgpu_count\x12\x39\n\x04\x64isk\x18\x14 \x03(\x0b\x32+.wandb_internal.EnvironmentRecord.DiskEntry\x12*\n\x06memory\x18\x15 \x01(\x0b\x32\x1a.wandb_internal.MemoryInfo\x12$\n\x03\x63pu\x18\x16 \x01(\x0b\x32\x17.wandb_internal.CpuInfo\x12(\n\x05\x61pple\x18\x17 \x01(\x0b\x32\x19.wandb_internal.AppleInfo\x12=\n\ngpu_nvidia\x18\x18 \x03(\x0b\x32\x1d.wandb_internal.GpuNvidiaInfoR\ngpu_nvidia\x12\x14\n\x0c\x63uda_version\x18\x19 \x01(\t\x12\x34\n\x07gpu_amd\x18\x1a \x03(\x0b\x32\x1a.wandb_internal.GpuAmdInfoR\x07gpu_amd\x12;\n\x05slurm\x18\x1b \x03(\x0b\x32,.wandb_internal.EnvironmentRecord.SlurmEntry\x12.\n\x08trainium\x18\x1c \x01(\x0b\x32\x1c.wandb_internal.TrainiumInfo\x12$\n\x03tpu\x18\x1d \x01(\x0b\x32\x17.wandb_internal.TPUInfo\x12\x30\n\tcoreweave\x18\x1e \x01(\x0b\x32\x1d.wandb_internal.CoreWeaveInfo\x12\x12\n\twriter_id\x18\xc7\x01 \x01(\t\x12+\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1b.wandb_internal._RecordInfo\x1a\x45\n\tDiskEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\'\n\x05value\x18\x02 \x01(\x0b\x32\x18.wandb_internal.DiskInfo:\x02\x38\x01\x1a,\n\nSlurmEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t:\x02\x38\x01\"\x8d\x01\n\x15PythonPackagesRequest\x12\x44\n\x07package\x18\x01 \x03(\x0b\x32\x33.wandb_internal.PythonPackagesRequest.PythonPackage\x1a.\n\rPythonPackage\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x0f\n\x07version\x18\x02 \x01(\t\"\x1c\n\x0cJobInputPath\x12\x0c\n\x04path\x18\x01 \x03(\t\"\xd6\x01\n\x0eJobInputSource\x12\x44\n\nrun_config\x18\x01 \x01(\x0b\x32..wandb_internal.JobInputSource.RunConfigSourceH\x00\x12?\n\x04\x66ile\x18\x02 \x01(\x0b\x32/.wandb_internal.JobInputSource.ConfigFileSourceH\x00\x1a\x11\n\x0fRunConfigSource\x1a \n\x10\x43onfigFileSource\x12\x0c\n\x04path\x18\x01 \x01(\tB\x08\n\x06source\"\xc7\x01\n\x0fJobInputRequest\x12\x34\n\x0cinput_source\x18\x01 \x01(\x0b\x32\x1e.wandb_internal.JobInputSource\x12\x33\n\rinclude_paths\x18\x02 \x03(\x0b\x32\x1c.wandb_internal.JobInputPath\x12\x33\n\rexclude_paths\x18\x03 \x03(\x0b\x32\x1c.wandb_internal.JobInputPath\x12\x14\n\x0cinput_schema\x18\x04 \x01(\t*\xda\x05\n\rServerFeature\x12\x1e\n\x1aSERVER_FEATURE_UNSPECIFIED\x10\x00\x12\x13\n\x0fLARGE_FILENAMES\x10\x11\x12\x11\n\rARTIFACT_TAGS\x10\x01\x12\x0e\n\nCLIENT_IDS\x10\x02\x12\x1c\n\x18\x41RTIFACT_REGISTRY_SEARCH\x10\x03\x12\x1b\n\x17STRUCTURED_CONSOLE_LOGS\x10\x04\x12(\n$ARTIFACT_COLLECTION_MEMBERSHIP_FILES\x10\x05\x12\x38\n4ARTIFACT_COLLECTION_MEMBERSHIP_FILE_DOWNLOAD_HANDLER\x10\x06\x12\x34\n0USE_ARTIFACT_WITH_ENTITY_AND_PROJECT_INFORMATION\x10\x07\x12\x1f\n\x1b\x45XPAND_DEFINED_METRIC_GLOBS\x10\x08\x12\x1f\n\x1b\x41UTOMATION_EVENT_RUN_METRIC\x10\t\x12&\n\"AUTOMATION_EVENT_RUN_METRIC_CHANGE\x10\n\x12\x1b\n\x17\x41UTOMATION_ACTION_NO_OP\x10\x0b\x12/\n+INCLUDE_ARTIFACT_TYPES_IN_REGISTRY_CREATION\x10\x0c\x12*\n&PROJECT_ARTIFACT_COLLECTION_MEMBERSHIP\x10\r\x12\x31\n-ARTIFACT_MEMBERSHIP_IN_LINK_ARTIFACT_RESPONSE\x10\x0e\x12\"\n\x1eTOTAL_COUNT_IN_FILE_CONNECTION\x10\x0f\x12*\n&ARTIFACT_COLLECTIONS_FILTERING_SORTING\x10\x10\x12\x35\n1ARTIFACT_V2_DOWNLOAD_HANDLER_SUPPORTS_ARTIFACT_ID\x10\x12\x42\x1bZ\x19\x63ore/pkg/service_go_protob\x06proto3')
/n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/wandb/sdk/wandb_settings.py:1308:    def validate_run_id(cls, value, values):
/n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/wandb/sdk/wandb_settings.py:1466:        - Is between 1 and 64 characters in length (inclusive)
/n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/wandb/sdk/wandb_settings.py:1479:            ValueError: If any tag is empty or exceeds 64 characters
/n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/wandb/sdk/wandb_settings.py:1496:                    f"Tag at index {i} is empty. Tags must be between 1 and 64 characters"
/n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/wandb/sdk/wandb_settings.py:1506:                    f"Tag '{display_tag}' is {len(tag_str)} characters. Tags must be between 1 and 64 characters"
/n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/wandb/sdk/launch/sweeps/scheduler.py:499:        encoded_run_id = base64.standard_b64encode(
/n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/wandb/sdk/wandb_run.py:3633:            title: The title of the alert, must be less than 64 characters long.

exec
/usr/bin/bash -lc "grep -nE 'update_from_env_vars|_settings|Settings\\(' /n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/wandb/sdk/wandb_setup.py | head -140; nl -ba /n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/wandb/sdk/wandb_setup.py | sed -n '80,180p;280,340p'; nl -ba /n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/wandb/sdk/wandb_init.py | sed -n '1450,1520p'" in /n/fs/gatrdp/codespace/FLAC
 succeeded in 345ms:
45:    from wandb.sdk.wandb_settings import Settings
110:        self._settings: Settings | None = None
111:        self._settings_environ: dict[str, str] | None = None
171:        if not self._settings_environ:
177:            for k, v in self._settings_environ.items()
191:    def _load_settings(
194:        system_settings_path: str | None,
201:            system_settings_path: Location of system settings file to use.
208:        from wandb.sdk.wandb_settings import Settings
210:        self._settings = Settings()
216:        self._settings.x_stats_pid = pid
218:        if system_settings_path:
219:            self._settings.settings_system = system_settings_path
222:            self._settings.settings_system = str(config_dir / "settings")
224:            self._settings.settings_system = str(
227:        self._settings.update_from_system_settings()
231:        self._settings_environ = os.environ.copy()
232:        self._settings.update_from_env_vars(self._settings_environ)
235:        self._settings.update_from_system_environment()
239:            not self._settings.sagemaker_disable
244:            sagemaker.set_global_settings(self._settings)
248:            self._settings.update_from_settings(overrides)
250:        wandb.termsetup(self._settings, None)
258:        if not self._settings:
259:            system_settings_path = settings.settings_system if settings else None
261:            self._load_settings(
262:                system_settings_path=system_settings_path,
267:        # This is 'elif' because load_settings already applies overrides.
269:            self._settings.update_from_settings(settings)
271:    def update_user_settings(self) -> None:
292:        if not self._settings:
293:            self._load_settings(
294:                system_settings_path=None,
297:            assert self._settings
299:        return self._settings
304:        return self._settings
307:        if self._settings and self._settings._offline:
313:        if self._settings and self._settings._offline:
318:        if self._settings and self._settings._offline:
332:    def _load_user_settings(self) -> dict[str, Any] | None:
338:        user_settings = dict()
340:            user_settings["save_code"] = flags["code_saving_enabled"]
344:            user_settings["email"] = email
346:        return user_settings
426:    return _setup(start_service=False, load_settings=False)
433:    load_settings: bool = True,
444:        load_settings: Whether to load settings from the environment
450:    if not load_settings and settings:
451:        raise ValueError("Cannot pass settings if load_settings is False.")
452:    if not load_settings and start_service:
453:        raise ValueError("Cannot use start_service if load_settings is False.")
462:        if load_settings:
597:    wbauth.unauthenticate_session(update_settings=False)
    80	        for level, msg, args, kwargs in self._log:
    81	            new_logger.log(level, msg, *args, **kwargs)
    82	        for msg, args, kwargs in self._exception:
    83	            new_logger.exception(msg, *args, **kwargs)
    84	
    85	
    86	Logger = Union[logging.Logger, _EarlyLogger]
    87	
    88	
    89	class _WandbSetup:
    90	    """W&B library singleton."""
    91	
    92	    def __init__(self, pid: int) -> None:
    93	        self._asyncer = asyncio_manager.AsyncioManager()
    94	        self._asyncer.start()
    95	
    96	        self._connection: ServiceConnection | None = None
    97	        self._connection_lock = threading.Lock()
    98	
    99	        self._active_runs: list[wandb_run.Run] = []
   100	        self._active_runs_lock = threading.Lock()
   101	
   102	        self._sweep_config: dict | None = None
   103	        self._server: server.Server | None = None
   104	        self._pid = pid
   105	
   106	        # TODO(jhr): defer strict checks until settings are fully initialized
   107	        #            and logging is ready
   108	        self._logger: Logger = _EarlyLogger()
   109	
   110	        self._settings: Settings | None = None
   111	        self._settings_environ: dict[str, str] | None = None
   112	
   113	    @property
   114	    def asyncer(self) -> asyncio_manager.AsyncioManager:
   115	        """The internal asyncio thread used by wandb."""
   116	        return self._asyncer
   117	
   118	    def add_active_run(self, run: wandb_run.Run) -> None:
   119	        """Append a run to the active runs list.
   120	
   121	        This must be called when a run is initialized.
   122	
   123	        Args:
   124	            run: A newly initialized run.
   125	        """
   126	        with self._active_runs_lock:
   127	            if run not in self._active_runs:
   128	                self._active_runs.append(run)
   129	
   130	    def remove_active_run(self, run: wandb_run.Run) -> None:
   131	        """Remove the run from the active runs list.
   132	
   133	        This must be called when a run is finished.
   134	
   135	        Args:
   136	            run: A run that is finished or crashed.
   137	        """
   138	        try:
   139	            with self._active_runs_lock:
   140	                self._active_runs.remove(run)
   141	        except ValueError:
   142	            pass  # Removing a run multiple times is not an error.
   143	
   144	    @property
   145	    def most_recent_active_run(self) -> wandb_run.Run | None:
   146	        """The most recently initialized run that is not yet finished."""
   147	        with self._active_runs_lock:
   148	            if not self._active_runs:
   149	                return None
   150	
   151	            return self._active_runs[-1]
   152	
   153	    def finish_all_active_runs(self) -> None:
   154	        """Finish all unfinished runs.
   155	
   156	        NOTE: This is slightly inefficient as it finishes runs one at a time.
   157	        This only exists to support using the `reinit="finish_previous"`
   158	        setting together with `reinit="create_new"` which does not seem to be a
   159	        useful pattern. Since `"create_new"` should eventually become the
   160	        default and only behavior, it does not seem worth optimizing.
   161	        """
   162	        # Take a snapshot as each call to `finish()` modifies `_active_runs`.
   163	        with self._active_runs_lock:
   164	            runs_copy = list(self._active_runs)
   165	
   166	        for run in runs_copy:
   167	            run.finish()
   168	
   169	    def did_environment_change(self) -> bool:
   170	        """Check if os.environ has changed since settings were initialized."""
   171	        if not self._settings_environ:
   172	            return False
   173	
   174	        exclude_env_vars = {"WANDB_SERVICE", "WANDB_KUBEFLOW_URL"}
   175	        singleton_env = {
   176	            k: v
   177	            for k, v in self._settings_environ.items()
   178	            if k.startswith("WANDB_") and k not in exclude_env_vars
   179	        }
   180	        os_env = {
   280	            self._logger._flush(new_logger)
   281	        self._logger = new_logger
   282	
   283	    def _get_logger(self) -> Logger:
   284	        return self._logger
   285	
   286	    @property
   287	    def settings(self) -> Settings:
   288	        """The global wandb settings.
   289	
   290	        Initializes settings if they have not yet been loaded.
   291	        """
   292	        if not self._settings:
   293	            self._load_settings(
   294	                system_settings_path=None,
   295	                disable_sagemaker=False,
   296	            )
   297	            assert self._settings
   298	
   299	        return self._settings
   300	
   301	    @property
   302	    def settings_if_loaded(self) -> Settings | None:
   303	        """The global wandb settings, or None if not yet loaded."""
   304	        return self._settings
   305	
   306	    def _get_entity(self) -> str | None:
   307	        if self._settings and self._settings._offline:
   308	            return None
   309	        entity = self.viewer.get("entity")
   310	        return entity
   311	
   312	    def _get_username(self) -> str | None:
   313	        if self._settings and self._settings._offline:
   314	            return None
   315	        return self.viewer.get("username")
   316	
   317	    def _get_teams(self) -> list[str]:
   318	        if self._settings and self._settings._offline:
   319	            return []
   320	        teams = self.viewer.get("teams")
   321	        if teams:
   322	            teams = [team["node"]["name"] for team in teams["edges"]]
   323	        return teams or []
   324	
   325	    @property
   326	    def viewer(self) -> dict[str, Any]:
   327	        if self._server is None:
   328	            self._server = server.Server(settings=self.settings)
   329	
   330	        return self._server.viewer
   331	
   332	    def _load_user_settings(self) -> dict[str, Any] | None:
   333	        # offline?
   334	        if self._server is None:
   335	            return None
   336	
   337	        flags = self._server._flags
   338	        user_settings = dict()
   339	        if "code_saving_enabled" in flags:
   340	            user_settings["save_code"] = flags["code_saving_enabled"]
  1450	    Examples:
  1451	    `wandb.init()` returns a `Run` object. Use the run object to log data,
  1452	    save artifacts, and manage the run lifecycle.
  1453	
  1454	    ```python
  1455	    import wandb
  1456	
  1457	    config = {"lr": 0.01, "batch_size": 32}
  1458	    with wandb.init(config=config) as run:
  1459	        # Log accuracy and loss to the run
  1460	        acc = 0.95  # Example accuracy
  1461	        loss = 0.05  # Example loss
  1462	        run.log({"accuracy": acc, "loss": loss})
  1463	    ```
  1464	    """
  1465	    init_telemetry = telemetry.TelemetryRecord()
  1466	
  1467	    init_settings = Settings()
  1468	    if isinstance(settings, dict):
  1469	        init_settings = Settings(**settings)
  1470	    elif isinstance(settings, Settings):
  1471	        init_settings = settings
  1472	
  1473	    # Explicit function arguments take precedence over settings
  1474	    if job_type is not None:
  1475	        init_settings.run_job_type = job_type
  1476	    if dir is not None:
  1477	        init_settings.root_dir = dir  # type: ignore
  1478	    if project is not None:
  1479	        init_settings.project = project
  1480	    if entity is not None:
  1481	        init_settings.entity = entity
  1482	    if reinit is not None:
  1483	        init_settings.reinit = reinit
  1484	    if tags is not None:
  1485	        init_settings.run_tags = tuple(tags)
  1486	    if group is not None:
  1487	        init_settings.run_group = group
  1488	    if name is not None:
  1489	        init_settings.run_name = name
  1490	    if notes is not None:
  1491	        init_settings.run_notes = notes
  1492	    if anonymous is not UNSET:
  1493	        init_settings.anonymous = anonymous
  1494	    if mode is not None:
  1495	        init_settings.mode = mode  # type: ignore
  1496	    if resume is not None:
  1497	        init_settings.resume = resume  # type: ignore
  1498	    if force is not None:
  1499	        init_settings.force = force
  1500	    # TODO: deprecate "tensorboard" in favor of "sync_tensorboard"
  1501	    if tensorboard is not None:
  1502	        init_settings.sync_tensorboard = tensorboard
  1503	    if sync_tensorboard is not None:
  1504	        init_settings.sync_tensorboard = sync_tensorboard
  1505	    if save_code is not None:
  1506	        init_settings.save_code = save_code
  1507	    if id is not None:
  1508	        init_settings.run_id = id
  1509	    if fork_from is not None:
  1510	        init_settings.fork_from = fork_from  # type: ignore
  1511	    if resume_from is not None:
  1512	        init_settings.resume_from = resume_from  # type: ignore
  1513	
  1514	    if config is not None:
  1515	        init_telemetry.feature.set_init_config = True
  1516	
  1517	    wl: wandb_setup._WandbSetup | None = None
  1518	
  1519	    try:
  1520	        wl = wandb_setup.singleton()

Completed the read-only delta trace, including the pinned Lightning/W&B implementation and official W&B ID constraints.
  ✓ Inspect the supplied delta and current sbatch restart/W&B logic
  ✓ Trace manifest, registry, readback, and future crash-restart consumers
  ✓ Verify W&B run-ID constraints and issue a focused verdict
codex
Verdict: APPROVED.

No blocking defects in the delta.

- Resume collision: the launcher scrubs inherited `WANDB_RESUME`, assigns a new ID, and does not re-export resume state. One caveat: pinned Lightning internally passes `resume="allow"` ([wandb.py](/n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/pytorch_lightning/loggers/wandb.py:328)), so an exact generated-ID collision would still resume. The nanosecond timestamp plus UUID suffix makes that operationally negligible.
- Readback: the new ID contains only letters, digits, and dashes, so the glob and exact suffix check work unchanged ([fa_orbit_wandb_readback.py](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_wandb_readback.py:28)). The `r40000` token and dashes have no special glob meaning.
- Manifests: `WANDB_RUN_ID` is reassigned before the writer, so restart manifests record the new ID ([fa_orbit_train.sbatch](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:648)). Only INITIAL overwrites `launch_manifest.txt`, preserving the original ID as the restart anchor. `record_restart`, registry validation, and producer-chain validation do not compare W&B IDs; they bind the restart manifest by hash and checkpoint lineage, so differing per-leg IDs do not break or mislink them.
- Future relaunches continue reading the INITIAL ID only for the lineage message, then mint another fresh ID. The current pre-existing production contract still only admits the audited 40k anchor—not arbitrary >40k crash checkpoints—but this delta neither introduces nor worsens that limitation.
- Length: W&B documents a 64-character limit, not 128. The longest production form is 46 characters (`VANL`, five-digit step), safely below the limit. The characters also require no sanitization. [W&B environment-variable documentation](https://docs.wandb.ai/models/track/environment-variables).
- Static verification: `bash -n` and `git diff --check` passed. No files or environments were modified.
tokens used
139,969
Verdict: APPROVED.

No blocking defects in the delta.

- Resume collision: the launcher scrubs inherited `WANDB_RESUME`, assigns a new ID, and does not re-export resume state. One caveat: pinned Lightning internally passes `resume="allow"` ([wandb.py](/n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/pytorch_lightning/loggers/wandb.py:328)), so an exact generated-ID collision would still resume. The nanosecond timestamp plus UUID suffix makes that operationally negligible.
- Readback: the new ID contains only letters, digits, and dashes, so the glob and exact suffix check work unchanged ([fa_orbit_wandb_readback.py](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_wandb_readback.py:28)). The `r40000` token and dashes have no special glob meaning.
- Manifests: `WANDB_RUN_ID` is reassigned before the writer, so restart manifests record the new ID ([fa_orbit_train.sbatch](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:648)). Only INITIAL overwrites `launch_manifest.txt`, preserving the original ID as the restart anchor. `record_restart`, registry validation, and producer-chain validation do not compare W&B IDs; they bind the restart manifest by hash and checkpoint lineage, so differing per-leg IDs do not break or mislink them.
- Future relaunches continue reading the INITIAL ID only for the lineage message, then mint another fresh ID. The current pre-existing production contract still only admits the audited 40k anchor—not arbitrary >40k crash checkpoints—but this delta neither introduces nor worsens that limitation.
- Length: W&B documents a 64-character limit, not 128. The longest production form is 46 characters (`VANL`, five-digit step), safely below the limit. The characters also require no sanitization. [W&B environment-variable documentation](https://docs.wandb.ai/models/track/environment-variables).
- Static verification: `bash -n` and `git diff --check` passed. No files or environments were modified.
