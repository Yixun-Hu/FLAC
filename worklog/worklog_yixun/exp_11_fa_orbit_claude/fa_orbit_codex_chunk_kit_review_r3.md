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
session id: 019ffc25-6b41-7f62-b1c2-98d35a53ba0b
--------
user
Round-5-fix-2 re-review (read-only; NO installs, NO file/env modifications; output = review text only). Repo /n/fs/gatrdp/codespace/FLAC, EXPDIR=worklog/worklog_yixun/exp_11_fa_orbit_claude. Your previous re-review (EXPDIR/fa_orbit_codex_chunk_kit_review_r2.md, final verdict section) returned REVISE with 4 blockings: (1) submission race not closed at the submitter, (2) endpoint attestation written before final classification, (3) versioned checkpoint filenames (-v1) invisible to attestation/recorder/watchdog, (4) forged-link laundering via the mutable registry. B5 remains DEFERRED by scope ruling (eval-side verify_chain, next round).

Fixes now in the tree (cumulative diff + watchdog: /tmp/claude-374349/-n-fs-gatrdp-codespace-FLAC/66305eb9-88c7-41df-95da-dde30a426dc2/scratchpad/chunk_kit_r3.diff): (1) submitter-side per-arm flock reservation with in-lock squeue check (nonzero squeue = refuse; live name = refuse; placed before the drift gate so duplicates refuse as duplicates; skipped under DRYRUN/SMOKE), watchdog persists lastjob immediately post-submit, unreachable by-ID branch removed; (2) attestation moved after the settled final classification (positive only on final_rc==0, else '<none>'), recorder requires sacct COMPLETED (SACCT_BIN test hook; --skip-sacct manual-recovery flag, watchdog never passes it); (3) both name shapes globbed with newest-mtime attestation of the actual path, recorder verifies the ATTESTED path (existence, canonical dir, step in name, re-hash), link gains final_ckpt_path, watchdog parses both shapes and resumes the recorded tip's path (unique-on-disk fallback only at the 40k anchor); (4) check_link_manifest for EVERY chain link (manifest bytes hash, job/uuid match, chunk_end==final_step, attestation==link) with an explicit cooperative-integrity threat-model paragraph. Non-blockings: deep startup anchor validation (step==40000, hex shape, unique file, full hash-once, freeze on mismatch), per-arm max-chunk table (C4L/C8/C16/C32:2500, VANL:5000), done_<ARM>=step@target staleness guard, fd-inheritance caveat documented. Guardtests 147 pass / 5 pre-existing dirty-tree fails (verified byte-identical class; new reservation cases run with a fake sbatch on PATH).

Verify the four blockings are resolved, assess the coder's disclosed deviations (reservation before drift gate; PATH-shim for squeue instead of a hook; anchor located via registry save_dir; sacct scoped to chunk links only), and hunt fresh defects strictly within the changed code. The deployment plan after your verdict: commit everything, cancel the five obsolete long legs (3687569-73), start the watchdog (ARMS=all five, CHUNK defaults). Verdict: APPROVED or REVISE. Bear in mind the declared threat model (cooperative integrity, accident-class) — findings requiring cryptographic provenance of uncommitted records are out of scope by standing ruling.
codex
I’ll audit the prior four blockers against the current implementation, then trace the disclosed deviations and changed-code edge cases. I’ll keep the repository and environment untouched.
exec
/usr/bin/bash -lc "pwd && git status --short && rg --files worklog/worklog_yixun/exp_11_fa_orbit_claude | sort | sed -n '1,240p' && sed -n '/Final [Vv]erdict/,"'$p'"' worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_chunk_kit_review_r2.md" in /n/fs/gatrdp/codespace/FLAC
 succeeded in 0ms:
/n/fs/gatrdp/codespace/FLAC
 M worklog/worklog_yixun/exp_11_fa_orbit_claude/arm_launch_registry.json
 M worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py
 M worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_record_restart.py
 M worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submit.sh
 M worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch
 M worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh
 M worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_submit.sh
 M worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_train.sbatch
?? AGENTS.md
?? AcousticRooms
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/.chunk_watchdog.lock
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/.submit_C8.lock
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-12_18-23-10_guardtests.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-13_12-06-48_guardtests.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-13_12-08-41_guardtests.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-13_12-10-57_guardtests.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-13_12-39-58_guardtests.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-13_12-42-29_guardtests.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-13_13-14-34_guardtests.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-13_13-18-38_guardtests.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_chunk_watchdog.sh
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_chunk_kit_review.md
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_chunk_kit_review_r2.md
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_chunk_kit_review_r3.md
?? worklog/worklog_yixun/exp_15_yaw_aug_claude/slurm_train_exp15-smoke-YAWAUG_3685989.out
?? worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-12_11-59-14_readback.log
?? worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-12_11-59-42_readback.log
?? worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-12_12-05-35_guardtests_fullfix.log
?? worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-12_12-35-37_guardtests_f3fix.log
?? worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-12_12-40-22_guardtests_f3fix.log
?? worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-12_17-49-31_YAWAUG_8x8_jid3685989_manifest.txt
?? worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-13_12-19-23_guardtests_chain.log
?? worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-13_12-23-35_guardtests_chain.log
?? worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-13_12-28-53_guardtests_chain.log
?? worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_chain_preflight.py
?? worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_chain_state.py
?? worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_rate_gate.py
?? worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_smoke_registry.json
/usr/bin/bash: line 1: rg: command not found

exec
/usr/bin/bash -lc "find worklog/worklog_yixun/exp_11_fa_orbit_claude -maxdepth 1 -type f -printf '%f\\n' | sort" in /n/fs/gatrdp/codespace/FLAC
 succeeded in 0ms:
.chunk_watchdog.lock
.submit_C8.lock
FLAC_AR_BF_C16.json
FLAC_AR_BF_C32.json
FLAC_AR_BF_C4L.json
FLAC_AR_BF_C8.json
FLAC_AR_BF_FA1.json
FLAC_AR_VANCKPT.json
arm_launch_registry.json
assert_arm_configs_exp11.py
c4_backfill_manifest.json
commits_fa_orbit.md
exp11_mechanism_readout.py
exp11_validate_rows.py
fa_orbit_02_trajectories.html
fa_orbit_2026-08-05_21-45-11_guardtests.log
fa_orbit_2026-08-05_21-45-57_pytest.log
fa_orbit_2026-08-06_14-34-04_pytest_batched.log
fa_orbit_2026-08-06_14-57-53_pytest_reverify.log
fa_orbit_2026-08-06_14-57-53_wrappertest.log
fa_orbit_2026-08-06_15-04-07_jid3646616_equivprobe.log
fa_orbit_2026-08-06_15-12-51_pytest_probefix.log
fa_orbit_2026-08-06_15-15-03_jid3646626_equivprobe.log
fa_orbit_2026-08-06_15-26-05_pytest_b1rootcause.log
fa_orbit_2026-08-06_15-27-42_jid3646634_equivprobe.log
fa_orbit_2026-08-06_15-35-38_pytest_boundadjust.log
fa_orbit_2026-08-06_15-37-36_jid3646653_equivprobe.log
fa_orbit_2026-08-06_16-47-14_C4L_8x8_jid3646734_manifest.txt
fa_orbit_2026-08-06_16-47-14_C4L_8x8_jid3646734_train.log
fa_orbit_2026-08-06_16-58-39_pytest_wandbfix.log
fa_orbit_2026-08-06_18-00-52_C4L_8x8_jid3648568_manifest.txt
fa_orbit_2026-08-06_18-00-52_C4L_8x8_jid3648568_train.log
fa_orbit_2026-08-06_20-36-00_C4L_8x8_jid3648694_manifest.txt
fa_orbit_2026-08-06_20-36-00_C4L_8x8_jid3648694_train.log
fa_orbit_2026-08-06_20-55-01_C8_8x8_jid3648695_manifest.txt
fa_orbit_2026-08-06_20-55-01_C8_8x8_jid3648695_train.log
fa_orbit_2026-08-06_20-55-50_C16_8x8_jid3648696_manifest.txt
fa_orbit_2026-08-06_20-55-50_C16_8x8_jid3648696_train.log
fa_orbit_2026-08-07_00-16-30_C32_8x8_jid3648697_manifest.txt
fa_orbit_2026-08-07_00-16-30_C32_8x8_jid3648697_train.log
fa_orbit_2026-08-07_00-31-51_pytest_screens.log
fa_orbit_2026-08-07_01-06-59_pytest_r4fixes.log
fa_orbit_2026-08-07_01-19-04_C4BACKFILL_S20000_s42_K8_screen.log
fa_orbit_2026-08-07_01-19-04_C4BACKFILL_S30000_s42_K8_screen.log
fa_orbit_2026-08-07_01-34-08_pytest_r4reverify.log
fa_orbit_2026-08-07_01-45-22_pytest_worktree.log
fa_orbit_2026-08-07_01-50-27_pytest_worktree.log
fa_orbit_2026-08-07_02-12-58_screen_guardtests.log
fa_orbit_2026-08-07_02-27-19_screen_guardtests.log
fa_orbit_2026-08-07_02-29-51_screen_guardtests.log
fa_orbit_2026-08-07_02-31-06_screen_guardtests.log
fa_orbit_2026-08-07_02-31-58_screen_guardtests.log
fa_orbit_2026-08-07_02-34-17_pytest_gocheck.log
fa_orbit_2026-08-07_02-49-18_screen_guardtests.log
fa_orbit_2026-08-07_02-50-55_screen_guardtests.log
fa_orbit_2026-08-07_02-52-41_screen_guardtests.log
fa_orbit_2026-08-07_02-53-42_screen_guardtests.log
fa_orbit_2026-08-07_03-02-52_pytest_gocheck2.log
fa_orbit_2026-08-07_03-19-51_screen_guardtests.log
fa_orbit_2026-08-07_03-21-40_screen_guardtests.log
fa_orbit_2026-08-07_03-23-01_screen_guardtests.log
fa_orbit_2026-08-07_03-24-16_screen_guardtests.log
fa_orbit_2026-08-07_03-25-18_pytest_gocheck3.log
fa_orbit_2026-08-07_03-39-44_screen_guardtests.log
fa_orbit_2026-08-07_03-41-03_screen_guardtests.log
fa_orbit_2026-08-07_03-42-23_screen_guardtests.log
fa_orbit_2026-08-07_03-43-45_screen_guardtests.log
fa_orbit_2026-08-07_03-45-10_screen_guardtests.log
fa_orbit_2026-08-07_03-46-13_pytest_gocheck4.log
fa_orbit_2026-08-07_04-04-35_screen_guardtests.log
fa_orbit_2026-08-07_04-06-19_screen_guardtests.log
fa_orbit_2026-08-07_04-07-53_screen_guardtests.log
fa_orbit_2026-08-07_04-09-40_screen_guardtests.log
fa_orbit_2026-08-07_04-10-55_pytest_condgo.log
fa_orbit_2026-08-07_04-20-30_C4L_S2500_s42_K8_screen.log
fa_orbit_2026-08-07_04-20-31_C4BACKFILL_S20000_s42_K8_screen.log
fa_orbit_2026-08-07_04-20-31_C4BACKFILL_S30000_s42_K8_screen.log
fa_orbit_2026-08-07_04-20-31_C4L_S10000_s42_K8_screen.log
fa_orbit_2026-08-07_04-20-31_C4L_S5000_s42_K8_screen.log
fa_orbit_2026-08-07_04-20-31_C4L_S7500_s42_K8_screen.log
fa_orbit_2026-08-07_04-21-31_C16_S5000_s42_K8_screen.log
fa_orbit_2026-08-07_04-21-31_C4L_S12500_s42_K8_screen.log
fa_orbit_2026-08-07_04-21-31_C4L_S15000_s42_K8_screen.log
fa_orbit_2026-08-07_04-21-31_C8_S10000_s42_K8_screen.log
fa_orbit_2026-08-07_04-21-31_C8_S2500_s42_K8_screen.log
fa_orbit_2026-08-07_04-21-31_C8_S5000_s42_K8_screen.log
fa_orbit_2026-08-07_04-21-32_C8_S7500_s42_K8_screen.log
fa_orbit_2026-08-07_15-22-29_C4L_S40000_s42_K8_screen.log
fa_orbit_2026-08-07_15-23-29_C4L_S40000_s42_K1_screen.log
fa_orbit_2026-08-07_15-23-29_C4L_S40000_s43_K8_screen.log
fa_orbit_2026-08-07_15-24-29_C4L_S40000_s43_K1_screen.log
fa_orbit_2026-08-07_15-24-29_C4L_S40000_s44_K8_screen.log
fa_orbit_2026-08-07_15-25-30_C4L_S40000_s44_K1_screen.log
fa_orbit_2026-08-07_15-26-30_C4L_S40000_s45_K8_screen.log
fa_orbit_2026-08-07_15-27-30_C4L_S40000_s45_K1_screen.log
fa_orbit_2026-08-07_15-28-30_C4L_S40000_s46_K8_screen.log
fa_orbit_2026-08-07_15-32-31_C4L_S40000_s46_K1_screen.log
fa_orbit_2026-08-07_15-33-31_C4L_S17500_s42_K8_screen.log
fa_orbit_2026-08-07_15-34-31_C4L_S30000_s42_K8_screen.log
fa_orbit_2026-08-07_15-34-32_C4L_S20000_s42_K8_screen.log
fa_orbit_2026-08-07_15-34-32_C4L_S22500_s42_K8_screen.log
fa_orbit_2026-08-07_15-34-32_C4L_S25000_s42_K8_screen.log
fa_orbit_2026-08-07_15-34-32_C4L_S27500_s42_K8_screen.log
fa_orbit_2026-08-07_15-34-32_C4L_S32500_s42_K8_screen.log
fa_orbit_2026-08-07_15-35-31_C4L_S35000_s42_K8_screen.log
fa_orbit_2026-08-07_15-36-32_C4L_S37500_s42_K8_screen.log
fa_orbit_2026-08-07_15-37-32_C8_S15000_s42_K8_screen.log
fa_orbit_2026-08-07_15-37-33_C8_S12500_s42_K8_screen.log
fa_orbit_2026-08-07_15-38-33_C16_S10000_s42_K8_screen.log
fa_orbit_2026-08-07_15-38-33_C16_S15000_s42_K8_screen.log
fa_orbit_2026-08-07_15-38-33_C32_S5000_s42_K8_screen.log
fa_orbit_2026-08-07_15-38-33_C8_S22500_s42_K8_screen.log
fa_orbit_2026-08-07_15-38-34_C8_S20000_s42_K8_screen.log
fa_orbit_2026-08-07_15-38-34_C8_S25000_s42_K8_screen.log
fa_orbit_2026-08-07_15-38-35_C8_S17500_s42_K8_screen.log
fa_orbit_2026-08-08_00-34-31_C8_S27500_s42_K8_screen.log
fa_orbit_2026-08-08_00-34-32_C8_S30000_s42_K8_screen.log
fa_orbit_2026-08-08_00-35-31_C8_S37500_s42_K8_screen.log
fa_orbit_2026-08-08_00-35-32_C16_S20000_s42_K8_screen.log
fa_orbit_2026-08-08_00-35-32_C32_S10000_s42_K8_screen.log
fa_orbit_2026-08-08_00-35-32_C8_S32500_s42_K8_screen.log
fa_orbit_2026-08-08_00-35-32_C8_S35000_s42_K8_screen.log
fa_orbit_2026-08-08_00-44-26_screen_guardtests.log
fa_orbit_2026-08-08_00-53-56_screen_guardtests.log
fa_orbit_2026-08-08_01-02-01_screen_guardtests.log
fa_orbit_2026-08-08_01-08-34_pytest_cells.log
fa_orbit_2026-08-08_01-14-44_screen_guardtests.log
fa_orbit_2026-08-08_01-39-37_C8_S40000_s42_K1_screen.log
fa_orbit_2026-08-08_01-39-37_C8_S40000_s42_K8_screen.log
fa_orbit_2026-08-08_01-39-37_C8_S40000_s43_K8_screen.log
fa_orbit_2026-08-08_01-40-36_C8_S40000_s43_K1_screen.log
fa_orbit_2026-08-08_01-40-36_C8_S40000_s45_K1_screen.log
fa_orbit_2026-08-08_01-40-37_C8_S40000_s44_K1_screen.log
fa_orbit_2026-08-08_01-40-37_C8_S40000_s44_K8_screen.log
fa_orbit_2026-08-08_01-40-37_C8_S40000_s45_K8_screen.log
fa_orbit_2026-08-08_01-41-36_C8_S40000_s46_K8_screen.log
fa_orbit_2026-08-08_01-41-38_C8_S40000_s46_K1_screen.log
fa_orbit_2026-08-08_01-43-36_C4BACKFILL_S40000_s42_K8_screen.log
fa_orbit_2026-08-08_01-52-38_C8_S40000_s42_K8_screen.log
fa_orbit_2026-08-08_01-53-38_C8_S40000_s42_K1_screen.log
fa_orbit_2026-08-08_01-53-38_C8_S40000_s43_K8_screen.log
fa_orbit_2026-08-08_01-54-38_C8_S40000_s44_K8_screen.log
fa_orbit_2026-08-08_01-54-39_C8_S40000_s43_K1_screen.log
fa_orbit_2026-08-08_01-54-39_C8_S40000_s44_K1_screen.log
fa_orbit_2026-08-08_01-55-39_C8_S40000_s45_K1_screen.log
fa_orbit_2026-08-08_01-55-39_C8_S40000_s45_K8_screen.log
fa_orbit_2026-08-08_01-56-38_C8_S40000_s46_K1_screen.log
fa_orbit_2026-08-08_01-56-39_C8_S40000_s46_K8_screen.log
fa_orbit_2026-08-08_01-59-38_C4BACKFILL_S40000_s42_K8_screen.log
fa_orbit_2026-08-08_01-59-39_C32_S10000_s42_K8_screen.log
fa_orbit_2026-08-08_19-32-42_screen_guardtests.log
fa_orbit_2026-08-08_19-41-23_pytest_fixround.log
fa_orbit_2026-08-08_19-45-41_C8_conf_S40000_s43_K8_jid3657712_screen.log
fa_orbit_2026-08-08_19-45-42_C8_conf_S40000_s42_K1_jid3657711_screen.log
fa_orbit_2026-08-08_19-45-42_C8_conf_S40000_s42_K8_jid3657710_screen.log
fa_orbit_2026-08-08_19-45-42_C8_conf_S40000_s43_K1_jid3657713_screen.log
fa_orbit_2026-08-08_19-45-42_C8_conf_S40000_s44_K1_jid3657715_screen.log
fa_orbit_2026-08-08_19-45-42_C8_conf_S40000_s44_K8_jid3657714_screen.log
fa_orbit_2026-08-08_19-45-42_C8_conf_S40000_s45_K1_jid3657717_screen.log
fa_orbit_2026-08-08_19-45-42_C8_conf_S40000_s45_K8_jid3657716_screen.log
fa_orbit_2026-08-08_19-45-42_C8_conf_S40000_s46_K8_jid3657718_screen.log
fa_orbit_2026-08-08_19-46-41_C16_conf_S40000_s42_K1_jid3657721_screen.log
fa_orbit_2026-08-08_19-46-41_C16_conf_S40000_s45_K1_jid3657727_screen.log
fa_orbit_2026-08-08_19-46-41_C16_conf_S40000_s45_K8_jid3657726_screen.log
fa_orbit_2026-08-08_19-46-41_C8_r3_rot22p5_S40000_s42_K8_jid3657738_screen.log
fa_orbit_2026-08-08_19-46-42_C16_conf_S40000_s42_K8_jid3657720_screen.log
fa_orbit_2026-08-08_19-46-42_C16_conf_S40000_s43_K1_jid3657723_screen.log
fa_orbit_2026-08-08_19-46-42_C16_conf_S40000_s44_K1_jid3657725_screen.log
fa_orbit_2026-08-08_19-46-42_C16_conf_S40000_s44_K8_jid3657724_screen.log
fa_orbit_2026-08-08_19-46-42_C16_conf_S40000_s46_K8_jid3657728_screen.log
fa_orbit_2026-08-08_19-46-42_C4L_r3_rot0_S40000_s42_K8_jid3657730_screen.log
fa_orbit_2026-08-08_19-46-42_C4L_r3_rot11p25_S40000_s42_K8_jid3657732_screen.log
fa_orbit_2026-08-08_19-46-42_C4L_r3_rot45_S40000_s42_K8_jid3657734_screen.log
fa_orbit_2026-08-08_19-46-42_C4L_r3_rot5p625_S40000_s42_K8_jid3657731_screen.log
fa_orbit_2026-08-08_19-46-42_C8_conf_S40000_s46_K1_jid3657719_screen.log
fa_orbit_2026-08-08_19-46-42_C8_r3_rot11p25_S40000_s42_K8_jid3657737_screen.log
fa_orbit_2026-08-08_19-46-42_C8_r3_rot5p625_S40000_s42_K8_jid3657736_screen.log
fa_orbit_2026-08-08_19-46-43_C16_conf_S40000_s43_K8_jid3657722_screen.log
fa_orbit_2026-08-08_19-46-43_C16_conf_S40000_s46_K1_jid3657729_screen.log
fa_orbit_2026-08-08_19-46-43_C4L_r3_rot22p5_S40000_s42_K8_jid3657733_screen.log
fa_orbit_2026-08-08_19-46-43_C8_r3_rot0_S40000_s42_K8_jid3657735_screen.log
fa_orbit_2026-08-08_19-47-41_C16_cross_a32_S40000_s42_K8_jid3657753_screen.log
fa_orbit_2026-08-08_19-47-41_C16_cross_a8_S40000_s42_K8_jid3657752_screen.log
fa_orbit_2026-08-08_19-47-41_C16_r3_rot0_S40000_s42_K8_jid3657740_screen.log
fa_orbit_2026-08-08_19-47-41_C16_r3_rot45_S40000_s42_K8_jid3657744_screen.log
fa_orbit_2026-08-08_19-47-41_C16_r3_rot5p625_S40000_s42_K8_jid3657741_screen.log
fa_orbit_2026-08-08_19-47-41_C32_screen_S15000_s42_K8_jid3657759_screen.log
fa_orbit_2026-08-08_19-47-41_C32_screen_S20000_s42_K8_jid3657760_screen.log
fa_orbit_2026-08-08_19-47-41_C4BACKFILL_cross_a32_S40000_s42_K8_jid3657757_screen.log
fa_orbit_2026-08-08_19-47-41_C4L_cross_a32_S40000_s42_K8_jid3657747_screen.log
fa_orbit_2026-08-08_19-47-41_C4L_cross_a8_S40000_s42_K8_jid3657745_screen.log
fa_orbit_2026-08-08_19-47-41_C8_r3_rot45_S40000_s42_K8_jid3657739_screen.log
fa_orbit_2026-08-08_19-47-42_C16_r3_rot11p25_S40000_s42_K8_jid3657742_screen.log
fa_orbit_2026-08-08_19-47-42_C16_r3_rot22p5_S40000_s42_K8_jid3657743_screen.log
fa_orbit_2026-08-08_19-47-42_C32_screen_S10000_s42_K8_jid3657758_screen.log
fa_orbit_2026-08-08_19-47-42_C4BACKFILL_cross_a16_S40000_s42_K8_jid3657756_screen.log
fa_orbit_2026-08-08_19-47-42_C4BACKFILL_cross_a8_S40000_s42_K8_jid3657755_screen.log
fa_orbit_2026-08-08_19-47-42_C4BACKFILL_screen_S40000_s42_K8_jid3657754_screen.log
fa_orbit_2026-08-08_19-47-42_C8_cross_a16_S40000_s42_K8_jid3657749_screen.log
fa_orbit_2026-08-08_19-47-42_C8_cross_a32_S40000_s42_K8_jid3657750_screen.log
fa_orbit_2026-08-08_19-47-43_C16_cross_a4_S40000_s42_K8_jid3657751_screen.log
fa_orbit_2026-08-08_19-47-43_C4L_cross_a16_S40000_s42_K8_jid3657746_screen.log
fa_orbit_2026-08-08_19-47-43_C8_cross_a4_S40000_s42_K8_jid3657748_screen.log
fa_orbit_2026-08-08_23-15-39_screen_guardtests.log
fa_orbit_2026-08-08_23-24-00_screen_guardtests.log
fa_orbit_2026-08-08_23-26-50_pytest_pin.log
fa_orbit_2026-08-08_23-49-15_screen_guardtests.log
fa_orbit_2026-08-08_23-52-34_screen_guardtests.log
fa_orbit_2026-08-08_23-55-56_pytest_pinround.log
fa_orbit_2026-08-08_23-58-55_C8_conf_S40000_s42_K1_jid3658825_screen.log
fa_orbit_2026-08-08_23-58-55_C8_conf_S40000_s42_K8_jid3658824_screen.log
fa_orbit_2026-08-08_23-58-55_C8_conf_S40000_s44_K1_jid3658829_screen.log
fa_orbit_2026-08-08_23-58-56_C8_conf_S40000_s43_K8_jid3658826_screen.log
fa_orbit_2026-08-08_23-58-56_C8_conf_S40000_s44_K8_jid3658828_screen.log
fa_orbit_2026-08-08_23-58-56_C8_conf_S40000_s45_K8_jid3658830_screen.log
fa_orbit_2026-08-08_23-58-56_C8_conf_S40000_s46_K8_jid3658832_screen.log
fa_orbit_2026-08-08_23-58-57_C8_conf_S40000_s43_K1_jid3658827_screen.log
fa_orbit_2026-08-08_23-58-57_C8_conf_S40000_s45_K1_jid3658831_screen.log
fa_orbit_2026-08-08_23-59-55_C8_conf_S40000_s46_K1_jid3658833_screen.log
fa_orbit_2026-08-08_23-59-56_C16_conf_S40000_s42_K8_jid3658834_screen.log
fa_orbit_2026-08-09_00-02-58_C16_conf_S40000_s42_K1_jid3658835_screen.log
fa_orbit_2026-08-09_00-02-58_C16_conf_S40000_s43_K8_jid3658836_screen.log
fa_orbit_2026-08-09_00-05-57_C16_conf_S40000_s43_K1_jid3658837_screen.log
fa_orbit_2026-08-09_00-05-57_C16_conf_S40000_s44_K8_jid3658838_screen.log
fa_orbit_2026-08-09_00-07-58_C16_conf_S40000_s44_K1_jid3658839_screen.log
fa_orbit_2026-08-09_00-09-56_C16_conf_S40000_s45_K1_jid3658841_screen.log
fa_orbit_2026-08-09_00-09-57_C16_conf_S40000_s45_K8_jid3658840_screen.log
fa_orbit_2026-08-09_00-09-57_C16_conf_S40000_s46_K1_jid3658843_screen.log
fa_orbit_2026-08-09_00-09-57_C16_conf_S40000_s46_K8_jid3658842_screen.log
fa_orbit_2026-08-09_10-54-10_guardtests.log
fa_orbit_2026-08-09_10-54-43_guardtests.log
fa_orbit_2026-08-09_10-55-30_guardtests.log
fa_orbit_2026-08-09_10-56-14_guardtests.log
fa_orbit_2026-08-09_10-56-34_C16_screen_S25000_s42_K8_jid3661451_screen.log
fa_orbit_2026-08-09_10-56-34_C32_screen_S10000_s42_K8_jid3661452_screen.log
fa_orbit_2026-08-09_10-57-34_C32_screen_S15000_s42_K8_jid3661453_screen.log
fa_orbit_2026-08-09_10-57-34_C32_screen_S20000_s42_K8_jid3661454_screen.log
fa_orbit_2026-08-09_10-58-34_C32_screen_S25000_s42_K8_jid3661455_screen.log
fa_orbit_2026-08-09_11-03-22_screen_guardtests.log
fa_orbit_2026-08-09_11-08-08_screen_guardtests.log
fa_orbit_2026-08-09_11-11-44_pytest_vanl.log
fa_orbit_2026-08-09_11-13-41_guardtests.log
fa_orbit_2026-08-09_12-06-39_VANL_8x8_jid3661520_manifest.txt
fa_orbit_2026-08-09_12-06-39_VANL_8x8_jid3661520_train.log
fa_orbit_2026-08-09_12-12-31_guardtests.log
fa_orbit_2026-08-09_12-12-37_guardtests.log
fa_orbit_2026-08-09_12-13-04_guardtests.log
fa_orbit_2026-08-09_12-18-08_screen_guardtests.log
fa_orbit_2026-08-09_12-22-37_screen_guardtests.log
fa_orbit_2026-08-09_12-26-47_guardtests.log
fa_orbit_2026-08-09_12-44-10_pytest_qpin.log
fa_orbit_2026-08-09_12-45-19_screen_guardtests.log
fa_orbit_2026-08-09_12-48-51_guardtests.log
fa_orbit_2026-08-09_12-50-00_guardtests.log
fa_orbit_2026-08-09_14-11-45_C16_screen_S30000_s42_K8_jid3662050_screen.log
fa_orbit_2026-08-09_14-11-45_C16_screen_S35000_s42_K8_jid3662051_screen.log
fa_orbit_2026-08-09_14-23-17_screen_guardtests.log
fa_orbit_2026-08-09_14-27-33_screen_guardtests.log
fa_orbit_2026-08-09_14-31-22_pytest_k1screens.log
fa_orbit_2026-08-09_14-36-46_C4L_screen_S2500_s42_K1_jid3662179_screen.log
fa_orbit_2026-08-09_14-36-46_C4L_screen_S7500_s42_K1_jid3662183_screen.log
fa_orbit_2026-08-09_14-36-47_C4L_screen_S5000_s42_K1_jid3662181_screen.log
fa_orbit_2026-08-09_14-36-47_C8_screen_S2500_s42_K1_jid3662180_screen.log
fa_orbit_2026-08-09_14-36-47_C8_screen_S5000_s42_K1_jid3662182_screen.log
fa_orbit_2026-08-09_14-37-46_C8_screen_S12500_s42_K1_jid3662188_screen.log
fa_orbit_2026-08-09_14-37-47_C4L_screen_S10000_s42_K1_jid3662185_screen.log
fa_orbit_2026-08-09_14-37-47_C4L_screen_S12500_s42_K1_jid3662187_screen.log
fa_orbit_2026-08-09_14-37-47_C4L_screen_S15000_s42_K1_jid3662189_screen.log
fa_orbit_2026-08-09_14-37-47_C4L_screen_S17500_s42_K1_jid3662191_screen.log
fa_orbit_2026-08-09_14-37-47_C8_screen_S10000_s42_K1_jid3662186_screen.log
fa_orbit_2026-08-09_14-37-47_C8_screen_S15000_s42_K1_jid3662190_screen.log
fa_orbit_2026-08-09_14-37-47_C8_screen_S7500_s42_K1_jid3662184_screen.log
fa_orbit_2026-08-09_14-38-47_C4L_screen_S20000_s42_K1_jid3662193_screen.log
fa_orbit_2026-08-09_14-38-47_C8_screen_S17500_s42_K1_jid3662192_screen.log
fa_orbit_2026-08-09_14-38-47_C8_screen_S20000_s42_K1_jid3662194_screen.log
fa_orbit_2026-08-09_14-39-46_C4L_screen_S22500_s42_K1_jid3662195_screen.log
fa_orbit_2026-08-09_14-39-46_C4L_screen_S25000_s42_K1_jid3662197_screen.log
fa_orbit_2026-08-09_14-39-46_C8_screen_S22500_s42_K1_jid3662196_screen.log
fa_orbit_2026-08-09_14-40-46_C8_screen_S27500_s42_K1_jid3662200_screen.log
fa_orbit_2026-08-09_14-40-47_C4L_screen_S27500_s42_K1_jid3662199_screen.log
fa_orbit_2026-08-09_14-40-47_C4L_screen_S30000_s42_K1_jid3662201_screen.log
fa_orbit_2026-08-09_14-40-47_C8_screen_S30000_s42_K1_jid3662202_screen.log
fa_orbit_2026-08-09_14-40-48_C8_screen_S25000_s42_K1_jid3662198_screen.log
fa_orbit_2026-08-09_14-41-47_C4L_screen_S32500_s42_K1_jid3662203_screen.log
fa_orbit_2026-08-09_14-41-47_C8_screen_S32500_s42_K1_jid3662204_screen.log
fa_orbit_2026-08-09_14-44-47_C4L_screen_S35000_s42_K1_jid3662205_screen.log
fa_orbit_2026-08-09_14-44-47_C8_screen_S35000_s42_K1_jid3662206_screen.log
fa_orbit_2026-08-09_14-45-47_C4L_screen_S37500_s42_K1_jid3662207_screen.log
fa_orbit_2026-08-09_14-45-47_C8_screen_S37500_s42_K1_jid3662208_screen.log
fa_orbit_2026-08-09_14-46-47_C16_screen_S15000_s42_K1_jid3662211_screen.log
fa_orbit_2026-08-09_14-46-47_C16_screen_S20000_s42_K1_jid3662212_screen.log
fa_orbit_2026-08-09_14-46-47_C16_screen_S5000_s42_K1_jid3662209_screen.log
fa_orbit_2026-08-09_14-46-48_C16_screen_S10000_s42_K1_jid3662210_screen.log
fa_orbit_2026-08-09_14-47-47_C16_screen_S25000_s42_K1_jid3662213_screen.log
fa_orbit_2026-08-09_14-47-47_C16_screen_S30000_s42_K1_jid3662214_screen.log
fa_orbit_2026-08-09_14-47-47_C16_screen_S35000_s42_K1_jid3662215_screen.log
fa_orbit_2026-08-09_14-47-47_C32_screen_S5000_s42_K1_jid3662216_screen.log
fa_orbit_2026-08-09_14-48-47_C32_screen_S20000_s42_K1_jid3662219_screen.log
fa_orbit_2026-08-09_14-48-47_C32_screen_S25000_s42_K1_jid3662220_screen.log
fa_orbit_2026-08-09_14-48-48_C32_screen_S10000_s42_K1_jid3662217_screen.log
fa_orbit_2026-08-09_14-48-48_C32_screen_S15000_s42_K1_jid3662218_screen.log
fa_orbit_2026-08-09_14-49-47_C4BACKFILL_screen_S20000_s42_K1_jid3662223_screen.log
fa_orbit_2026-08-09_14-49-47_C4BACKFILL_screen_S30000_s42_K1_jid3662224_screen.log
fa_orbit_2026-08-09_15-18-49_C32_screen_S2500_s42_K1_jid3662408_screen.log
fa_orbit_2026-08-09_15-18-49_C32_screen_S2500_s42_K8_jid3662407_screen.log
fa_orbit_2026-08-09_15-19-48_C32_screen_S22500_s42_K8_jid3662416_screen.log
fa_orbit_2026-08-09_15-19-49_C32_screen_S12500_s42_K1_jid3662413_screen.log
fa_orbit_2026-08-09_15-19-49_C32_screen_S12500_s42_K8_jid3662412_screen.log
fa_orbit_2026-08-09_15-19-49_C32_screen_S17500_s42_K1_jid3662415_screen.log
fa_orbit_2026-08-09_15-19-49_C32_screen_S17500_s42_K8_jid3662414_screen.log
fa_orbit_2026-08-09_15-19-49_C32_screen_S7500_s42_K1_jid3662411_screen.log
fa_orbit_2026-08-09_15-19-49_C32_screen_S7500_s42_K8_jid3662410_screen.log
fa_orbit_2026-08-09_15-20-48_C32_screen_S22500_s42_K1_jid3662417_screen.log
fa_orbit_2026-08-09_15-20-49_C32_screen_S27500_s42_K1_jid3662419_screen.log
fa_orbit_2026-08-09_15-20-49_C32_screen_S27500_s42_K8_jid3662418_screen.log
fa_orbit_2026-08-09_15-20-49_C32_screen_S30000_s42_K1_jid3662421_screen.log
fa_orbit_2026-08-09_15-20-49_C32_screen_S30000_s42_K8_jid3662420_screen.log
fa_orbit_2026-08-09_17-06-16_screen_guardtests.log
fa_orbit_2026-08-09_17-10-17_guardtests.log
fa_orbit_2026-08-09_17-11-12_screen_guardtests.log
fa_orbit_2026-08-09_17-15-36_pytest_q10.log
fa_orbit_2026-08-09_17-16-44_guardtests.log
fa_orbit_2026-08-09_17-24-24_screen_guardtests.log
fa_orbit_2026-08-09_17-28-58_screen_guardtests.log
fa_orbit_2026-08-09_20-20-54_C32_screen_S32500_s42_K1_jid3665920_screen.log
fa_orbit_2026-08-09_20-20-54_C32_screen_S32500_s42_K8_jid3665919_screen.log
fa_orbit_2026-08-10_01-11-19_screen_guardtests.log
fa_orbit_2026-08-10_01-24-26_guardtests.log
fa_orbit_2026-08-10_01-27-05_guardtests.log
fa_orbit_2026-08-10_01-30-00_guardtests.log
fa_orbit_2026-08-10_01-31-51_C32_screen_S35000_s42_K8_jid3668010_screen.log
fa_orbit_2026-08-10_01-37-34_failopen_repro.log
fa_orbit_2026-08-10_01-37-43_pytest_restartrecord.log
fa_orbit_2026-08-10_01-39-54_C32_screen_S35000_s42_K1_jid3668011_screen.log
fa_orbit_2026-08-10_01-40-02_screen_guardtests.log
fa_orbit_2026-08-10_01-44-34_screen_guardtests.log
fa_orbit_2026-08-10_01-49-16_screen_guardtests.log
fa_orbit_2026-08-10_01-53-27_pytest_fix2gate.log
fa_orbit_2026-08-10_01-57-49_guardtests.log
fa_orbit_2026-08-10_02-00-01_screen_guardtests.log
fa_orbit_2026-08-10_02-04-24_pytest_legmanifest.log
fa_orbit_2026-08-10_02-15-07_bandharvest_repro.log
fa_orbit_2026-08-10_02-16-31_pytest_figharvest.log
fa_orbit_2026-08-10_02-21-53_guardtests.log
fa_orbit_2026-08-10_02-24-09_guardtests.log
fa_orbit_2026-08-10_02-24-09_pytest_roundB.log
fa_orbit_2026-08-10_02-28-11_guardtests.log
fa_orbit_2026-08-10_02-29-13_screen_guardtests.log
fa_orbit_2026-08-10_06-47-34_C32_screen_S37500_s42_K8_jid3668648_screen.log
fa_orbit_2026-08-10_06-48-34_C32_screen_S37500_s42_K1_jid3668649_screen.log
fa_orbit_2026-08-10_11-57-38_C32_screen_S40000_s42_K1_jid3670799_screen.log
fa_orbit_2026-08-10_11-57-38_C32_screen_S40000_s42_K8_jid3670798_screen.log
fa_orbit_2026-08-10_18-20-00_C32_conf_S40000_s42_K8_jid3672838_screen.log
fa_orbit_2026-08-10_18-23-02_C32_conf_S40000_s42_K1_jid3672839_screen.log
fa_orbit_2026-08-10_18-23-02_C32_conf_S40000_s43_K8_jid3672840_screen.log
fa_orbit_2026-08-10_18-25-01_C32_conf_S40000_s43_K1_jid3672841_screen.log
fa_orbit_2026-08-10_18-26-02_C32_conf_S40000_s44_K1_jid3672843_screen.log
fa_orbit_2026-08-10_18-26-02_C32_conf_S40000_s44_K8_jid3672842_screen.log
fa_orbit_2026-08-10_18-29-03_C32_conf_S40000_s45_K8_jid3672844_screen.log
fa_orbit_2026-08-10_18-37-06_C32_conf_S40000_s45_K1_jid3672845_screen.log
fa_orbit_2026-08-10_18-41-06_C32_conf_S40000_s46_K8_jid3672846_screen.log
fa_orbit_2026-08-10_18-41-07_C32_conf_S40000_s46_K1_jid3672847_screen.log
fa_orbit_2026-08-11_00-46-17_VANL_screen_S2500_s42_K8_jid3674679_screen.log
fa_orbit_2026-08-11_00-50-18_VANL_q9_S40000_s42_K8_jid3674658_screen.log
fa_orbit_2026-08-11_00-56-20_VANL_q9_S40000_s42_K1_jid3674659_screen.log
fa_orbit_2026-08-11_00-59-20_VANL_q9_S40000_s44_K8_jid3674662_screen.log
fa_orbit_2026-08-11_00-59-21_VANL_q9_S40000_s43_K1_jid3674661_screen.log
fa_orbit_2026-08-11_00-59-21_VANL_q9_S40000_s43_K8_jid3674660_screen.log
fa_orbit_2026-08-11_01-03-23_VANL_q9_S40000_s44_K1_jid3674663_screen.log
fa_orbit_2026-08-11_01-03-23_VANL_q9_S40000_s45_K8_jid3674664_screen.log
fa_orbit_2026-08-11_01-04-23_VANL_screen_S2500_s42_K1_jid3674680_screen.log
fa_orbit_2026-08-11_01-05-23_VANL_q9_S40000_s45_K1_jid3674665_screen.log
fa_orbit_2026-08-11_01-05-23_VANL_q9_S40000_s46_K8_jid3674666_screen.log
fa_orbit_2026-08-11_01-06-24_VANL_screen_S5000_s42_K8_jid3674681_screen.log
fa_orbit_2026-08-11_01-07-23_C4L_q9_S40000_s43_K1_jid3674671_screen.log
fa_orbit_2026-08-11_01-07-23_C4L_q9_S40000_s43_K8_jid3674670_screen.log
fa_orbit_2026-08-11_01-07-23_C4L_q9_S40000_s44_K1_jid3674673_screen.log
fa_orbit_2026-08-11_01-07-24_C4L_q9_S40000_s42_K8_jid3674668_screen.log
fa_orbit_2026-08-11_01-07-24_C4L_q9_S40000_s44_K8_jid3674672_screen.log
fa_orbit_2026-08-11_01-07-24_VANL_q9_S40000_s46_K1_jid3674667_screen.log
fa_orbit_2026-08-11_01-07-25_C4L_q9_S40000_s42_K1_jid3674669_screen.log
fa_orbit_2026-08-11_01-08-23_VANL_screen_S10000_s42_K8_jid3674685_screen.log
fa_orbit_2026-08-11_01-08-23_VANL_screen_S7500_s42_K1_jid3674684_screen.log
fa_orbit_2026-08-11_01-08-24_C4L_q9_S40000_s45_K1_jid3674675_screen.log
fa_orbit_2026-08-11_01-08-24_C4L_q9_S40000_s45_K8_jid3674674_screen.log
fa_orbit_2026-08-11_01-08-24_C4L_q9_S40000_s46_K1_jid3674677_screen.log
fa_orbit_2026-08-11_01-08-24_VANL_screen_S5000_s42_K1_jid3674682_screen.log
fa_orbit_2026-08-11_01-08-24_VANL_screen_S7500_s42_K8_jid3674683_screen.log
fa_orbit_2026-08-11_01-08-25_C4L_q9_S40000_s46_K8_jid3674676_screen.log
fa_orbit_2026-08-11_01-09-24_VANL_screen_S10000_s42_K1_jid3674686_screen.log
fa_orbit_2026-08-11_01-10-25_VANL_screen_S12500_s42_K1_jid3674688_screen.log
fa_orbit_2026-08-11_01-10-25_VANL_screen_S12500_s42_K8_jid3674687_screen.log
fa_orbit_2026-08-11_01-10-25_VANL_screen_S15000_s42_K8_jid3674689_screen.log
fa_orbit_2026-08-11_01-11-25_VANL_screen_S15000_s42_K1_jid3674690_screen.log
fa_orbit_2026-08-11_01-12-24_VANL_screen_S17500_s42_K1_jid3674692_screen.log
fa_orbit_2026-08-11_01-12-25_VANL_screen_S17500_s42_K8_jid3674691_screen.log
fa_orbit_2026-08-11_01-13-25_VANL_screen_S20000_s42_K8_jid3674693_screen.log
fa_orbit_2026-08-11_01-14-26_VANL_screen_S20000_s42_K1_jid3674694_screen.log
fa_orbit_2026-08-11_01-14-26_VANL_screen_S22500_s42_K1_jid3674696_screen.log
fa_orbit_2026-08-11_01-14-26_VANL_screen_S22500_s42_K8_jid3674695_screen.log
fa_orbit_2026-08-11_01-14-26_VANL_screen_S25000_s42_K8_jid3674697_screen.log
fa_orbit_2026-08-11_01-16-27_VANL_screen_S25000_s42_K1_jid3674698_screen.log
fa_orbit_2026-08-11_01-16-27_VANL_screen_S27500_s42_K1_jid3674700_screen.log
fa_orbit_2026-08-11_01-16-27_VANL_screen_S27500_s42_K8_jid3674699_screen.log
fa_orbit_2026-08-11_01-17-27_VANL_screen_S30000_s42_K1_jid3674702_screen.log
fa_orbit_2026-08-11_01-17-27_VANL_screen_S30000_s42_K8_jid3674701_screen.log
fa_orbit_2026-08-11_01-17-27_VANL_screen_S32500_s42_K1_jid3674704_screen.log
fa_orbit_2026-08-11_01-17-27_VANL_screen_S32500_s42_K8_jid3674703_screen.log
fa_orbit_2026-08-11_01-17-27_VANL_screen_S35000_s42_K1_jid3674706_screen.log
fa_orbit_2026-08-11_01-17-27_VANL_screen_S35000_s42_K8_jid3674705_screen.log
fa_orbit_2026-08-11_01-17-27_VANL_screen_S37500_s42_K8_jid3674707_screen.log
fa_orbit_2026-08-11_01-18-26_VANL_screen_S40000_s42_K8_jid3674709_screen.log
fa_orbit_2026-08-11_01-18-27_VANL_screen_S37500_s42_K1_jid3674708_screen.log
fa_orbit_2026-08-11_01-18-27_VANL_screen_S40000_s42_K1_jid3674710_screen.log
fa_orbit_2026-08-11_14-55-12_guardtests.log
fa_orbit_2026-08-11_14-57-42_guardtests.log
fa_orbit_2026-08-11_15-15-45_guardtests.log
fa_orbit_2026-08-11_18-31-38_C4L_r3_rot0_S40000_s42_K8_jid3680738_screen.log
fa_orbit_2026-08-11_18-31-38_C4L_r3_rot5p625_S40000_s42_K8_jid3680739_screen.log
fa_orbit_2026-08-11_18-32-38_C4L_r3_rot11p25_S40000_s42_K8_jid3680740_screen.log
fa_orbit_2026-08-11_18-33-38_C4L_r3_rot22p5_S40000_s42_K8_jid3680741_screen.log
fa_orbit_2026-08-11_18-33-38_C4L_r3_rot45_S40000_s42_K8_jid3680742_screen.log
fa_orbit_2026-08-11_18-34-38_C8_r3_rot0_S40000_s42_K8_jid3680743_screen.log
fa_orbit_2026-08-11_18-35-38_C8_r3_rot5p625_S40000_s42_K8_jid3680744_screen.log
fa_orbit_2026-08-11_18-35-39_C8_r3_rot11p25_S40000_s42_K8_jid3680745_screen.log
fa_orbit_2026-08-11_18-36-40_C16_r3_rot0_S40000_s42_K8_jid3680748_screen.log
fa_orbit_2026-08-11_18-36-40_C16_r3_rot5p625_S40000_s42_K8_jid3680749_screen.log
fa_orbit_2026-08-11_18-36-40_C8_r3_rot22p5_S40000_s42_K8_jid3680746_screen.log
fa_orbit_2026-08-11_18-36-41_C8_r3_rot45_S40000_s42_K8_jid3680747_screen.log
fa_orbit_2026-08-11_18-37-40_C16_r3_rot11p25_S40000_s42_K8_jid3680750_screen.log
fa_orbit_2026-08-11_18-39-41_C16_r3_rot22p5_S40000_s42_K8_jid3680751_screen.log
fa_orbit_2026-08-11_18-43-43_C16_r3_rot45_S40000_s42_K8_jid3680752_screen.log
fa_orbit_2026-08-11_18-43-43_C32_r3_rot0_S40000_s42_K8_jid3680753_screen.log
fa_orbit_2026-08-11_18-43-43_C32_r3_rot5p625_S40000_s42_K8_jid3680754_screen.log
fa_orbit_2026-08-11_18-44-42_C32_r3_rot11p25_S40000_s42_K8_jid3680755_screen.log
fa_orbit_2026-08-11_18-44-42_C32_r3_rot22p5_S40000_s42_K8_jid3680756_screen.log
fa_orbit_2026-08-11_18-44-43_C32_r3_rot45_S40000_s42_K8_jid3680757_screen.log
fa_orbit_2026-08-11_18-45-42_C4L_cross_a32_S40000_s42_K8_jid3680760_screen.log
fa_orbit_2026-08-11_18-45-42_C8_cross_a4_S40000_s42_K8_jid3680761_screen.log
fa_orbit_2026-08-11_18-45-43_C4L_cross_a16_S40000_s42_K8_jid3680759_screen.log
fa_orbit_2026-08-11_18-45-43_C4L_cross_a8_S40000_s42_K8_jid3680758_screen.log
fa_orbit_2026-08-11_18-46-44_C16_cross_a4_S40000_s42_K8_jid3680762_screen.log
fa_orbit_2026-08-11_18-47-44_C32_cross_a4_S40000_s42_K8_jid3680763_screen.log
fa_orbit_2026-08-11_18-47-44_C4BACKFILL_cross_a8_S40000_s42_K8_jid3680764_screen.log
fa_orbit_2026-08-11_18-49-44_C4BACKFILL_cross_a16_S40000_s42_K8_jid3680765_screen.log
fa_orbit_2026-08-11_18-49-45_C4BACKFILL_cross_a32_S40000_s42_K8_jid3680766_screen.log
fa_orbit_2026-08-12_06-45-52_guardtests.log
fa_orbit_2026-08-12_07-49-41_C8_cross_a16_S40000_s42_K8_jid3684137_screen.log
fa_orbit_2026-08-12_07-53-42_C8_cross_a32_S40000_s42_K8_jid3684138_screen.log
fa_orbit_2026-08-12_07-55-42_C16_cross_a8_S40000_s42_K8_jid3684139_screen.log
fa_orbit_2026-08-12_07-55-43_C16_cross_a32_S40000_s42_K8_jid3684140_screen.log
fa_orbit_2026-08-12_07-59-45_C32_cross_a8_S40000_s42_K8_jid3684143_screen.log
fa_orbit_2026-08-12_08-01-45_C4BACKFILL_screen_S40000_s42_K8_jid3684155_screen.log
fa_orbit_2026-08-12_08-01-46_C32_cross_a16_S40000_s42_K8_jid3684144_screen.log
fa_orbit_2026-08-12_18-11-38_C4L_8x8_jid3684149_manifest.txt
fa_orbit_2026-08-12_18-11-38_C4L_8x8_jid3684149_train.log
fa_orbit_2026-08-12_18-19-40_C8_8x8_jid3684150_manifest.txt
fa_orbit_2026-08-12_18-19-40_C8_8x8_jid3684150_train.log
fa_orbit_2026-08-12_18-21-40_C16_8x8_jid3684151_manifest.txt
fa_orbit_2026-08-12_18-21-40_C16_8x8_jid3684151_train.log
fa_orbit_2026-08-12_18-23-10_guardtests.log
fa_orbit_2026-08-13_12-06-48_guardtests.log
fa_orbit_2026-08-13_12-08-41_guardtests.log
fa_orbit_2026-08-13_12-10-57_guardtests.log
fa_orbit_2026-08-13_12-39-58_guardtests.log
fa_orbit_2026-08-13_12-42-29_guardtests.log
fa_orbit_2026-08-13_13-14-34_guardtests.log
fa_orbit_2026-08-13_13-18-38_guardtests.log
fa_orbit_add_anchor.py
fa_orbit_analysis.md
fa_orbit_chunk_watchdog.sh
fa_orbit_ckpt_preflight.py
fa_orbit_classify.py
fa_orbit_codex_chunk_kit_review.md
fa_orbit_codex_chunk_kit_review_r2.md
fa_orbit_codex_chunk_kit_review_r3.md
fa_orbit_codex_code_batched_reverify.md
fa_orbit_codex_code_batched_review.md
fa_orbit_codex_code_consolidated_review.md
fa_orbit_codex_code_r1_review.md
fa_orbit_codex_code_r2_reverify.md
fa_orbit_codex_code_r2_review.md
fa_orbit_codex_code_r3_review.md
fa_orbit_codex_code_r4_final.md
fa_orbit_codex_code_r4_reverify.md
fa_orbit_codex_code_r4_review.md
fa_orbit_codex_condgo.md
fa_orbit_codex_content_gate_review.md
fa_orbit_codex_content_gate_review_r2.md
fa_orbit_codex_content_gate_review_r3.md
fa_orbit_codex_gocheck2.md
fa_orbit_codex_gocheck3.md
fa_orbit_codex_gocheck4.md
fa_orbit_codex_gocheck5.md
fa_orbit_codex_measurement_GO.md
fa_orbit_codex_plan_review.md
fa_orbit_codex_qpin_review.md
fa_orbit_codex_qprime_approval.md
fa_orbit_codex_readout_review.md
fa_orbit_codex_repin_review.md
fa_orbit_codex_signoff.md
fa_orbit_codex_unbundle_ruling.md
fa_orbit_codex_vanl_review.md
fa_orbit_codex_wandb_restart_review.md
fa_orbit_command.md
fa_orbit_equiv_probe.py
fa_orbit_equiv_probe.sbatch
fa_orbit_equivprobe_wrapper_test.sh
fa_orbit_measure_worktree.sh
fa_orbit_producer_manifest.py
fa_orbit_record_restart.py
fa_orbit_results.md
fa_orbit_screen.sbatch
fa_orbit_screen_guardtests.sh
fa_orbit_screen_submit.sh
fa_orbit_submission_C16_1786054560564868965-5fd4c1e1.txt
fa_orbit_submission_C16_1786056695148252065-920f395f.txt
fa_orbit_submission_C16_1786310422371467848-a776b47c.txt
fa_orbit_submission_C16_1786465302622561406-f725e951.txt
fa_orbit_submission_C16_1786476227007058126-933e8fc4.txt
fa_orbit_submission_C16_1786531886943324938-89cca289.txt
fa_orbit_submission_C16_1786574118551451993-8b506cb0.txt
fa_orbit_submission_C16_conf_S40000_s42_K1_jid3658835.txt
fa_orbit_submission_C16_conf_S40000_s42_K8_jid3658834.txt
fa_orbit_submission_C16_conf_S40000_s43_K1_jid3658837.txt
fa_orbit_submission_C16_conf_S40000_s43_K8_jid3658836.txt
fa_orbit_submission_C16_conf_S40000_s44_K1_jid3658839.txt
fa_orbit_submission_C16_conf_S40000_s44_K8_jid3658838.txt
fa_orbit_submission_C16_conf_S40000_s45_K1_jid3658841.txt
fa_orbit_submission_C16_conf_S40000_s45_K8_jid3658840.txt
fa_orbit_submission_C16_conf_S40000_s46_K1_jid3658843.txt
fa_orbit_submission_C16_conf_S40000_s46_K8_jid3658842.txt
fa_orbit_submission_C16_cross_S40000_s42_K8_jid3680762.txt
fa_orbit_submission_C16_cross_S40000_s42_K8_jid3684139.txt
fa_orbit_submission_C16_cross_S40000_s42_K8_jid3684140.txt
fa_orbit_submission_C16_r3_S40000_s42_K8_jid3680748.txt
fa_orbit_submission_C16_r3_S40000_s42_K8_jid3680749.txt
fa_orbit_submission_C16_r3_S40000_s42_K8_jid3680750.txt
fa_orbit_submission_C16_r3_S40000_s42_K8_jid3680751.txt
fa_orbit_submission_C16_r3_S40000_s42_K8_jid3680752.txt
fa_orbit_submission_C16_screen_S10000_s42_K1_jid3662210.txt
fa_orbit_submission_C16_screen_S15000_s42_K1_jid3662211.txt
fa_orbit_submission_C16_screen_S20000_s42_K1_jid3662212.txt
fa_orbit_submission_C16_screen_S25000_s42_K1_jid3662213.txt
fa_orbit_submission_C16_screen_S25000_s42_K8_jid3661451.txt
fa_orbit_submission_C16_screen_S30000_s42_K1_jid3662214.txt
fa_orbit_submission_C16_screen_S30000_s42_K8_jid3662050.txt
fa_orbit_submission_C16_screen_S35000_s42_K1_jid3662215.txt
fa_orbit_submission_C16_screen_S35000_s42_K8_jid3662051.txt
fa_orbit_submission_C16_screen_S5000_s42_K1_jid3662209.txt
fa_orbit_submission_C32_1786054560670066214-c4a97ed7.txt
fa_orbit_submission_C32_1786056695251126283-7cb3aeb0.txt
fa_orbit_submission_C32_1786465302839671032-280f410d.txt
fa_orbit_submission_C32_1786476227212448813-095d0a36.txt
fa_orbit_submission_C32_1786531887062468286-c65636ea.txt
fa_orbit_submission_C32_1786574118697104781-37837482.txt
fa_orbit_submission_C32_conf_S40000_s42_K1_jid3672839.txt
fa_orbit_submission_C32_conf_S40000_s42_K8_jid3672838.txt
fa_orbit_submission_C32_conf_S40000_s43_K1_jid3672841.txt
fa_orbit_submission_C32_conf_S40000_s43_K8_jid3672840.txt
fa_orbit_submission_C32_conf_S40000_s44_K1_jid3672843.txt
fa_orbit_submission_C32_conf_S40000_s44_K8_jid3672842.txt
fa_orbit_submission_C32_conf_S40000_s45_K1_jid3672845.txt
fa_orbit_submission_C32_conf_S40000_s45_K8_jid3672844.txt
fa_orbit_submission_C32_conf_S40000_s46_K1_jid3672847.txt
fa_orbit_submission_C32_conf_S40000_s46_K8_jid3672846.txt
fa_orbit_submission_C32_cross_S40000_s42_K8_jid3680763.txt
fa_orbit_submission_C32_cross_S40000_s42_K8_jid3684143.txt
fa_orbit_submission_C32_cross_S40000_s42_K8_jid3684144.txt
fa_orbit_submission_C32_r3_S40000_s42_K8_jid3680753.txt
fa_orbit_submission_C32_r3_S40000_s42_K8_jid3680754.txt
fa_orbit_submission_C32_r3_S40000_s42_K8_jid3680755.txt
fa_orbit_submission_C32_r3_S40000_s42_K8_jid3680756.txt
fa_orbit_submission_C32_r3_S40000_s42_K8_jid3680757.txt
fa_orbit_submission_C32_screen_S10000_s42_K1_jid3662217.txt
fa_orbit_submission_C32_screen_S10000_s42_K8_jid3661452.txt
fa_orbit_submission_C32_screen_S12500_s42_K1_jid3662413.txt
fa_orbit_submission_C32_screen_S12500_s42_K8_jid3662412.txt
fa_orbit_submission_C32_screen_S15000_s42_K1_jid3662218.txt
fa_orbit_submission_C32_screen_S15000_s42_K8_jid3661453.txt
fa_orbit_submission_C32_screen_S17500_s42_K1_jid3662415.txt
fa_orbit_submission_C32_screen_S17500_s42_K8_jid3662414.txt
fa_orbit_submission_C32_screen_S20000_s42_K1_jid3662219.txt
fa_orbit_submission_C32_screen_S20000_s42_K8_jid3661454.txt
fa_orbit_submission_C32_screen_S22500_s42_K1_jid3662417.txt
fa_orbit_submission_C32_screen_S22500_s42_K8_jid3662416.txt
fa_orbit_submission_C32_screen_S25000_s42_K1_jid3662220.txt
fa_orbit_submission_C32_screen_S25000_s42_K8_jid3661455.txt
fa_orbit_submission_C32_screen_S2500_s42_K1_jid3662408.txt
fa_orbit_submission_C32_screen_S2500_s42_K8_jid3662407.txt
fa_orbit_submission_C32_screen_S27500_s42_K1_jid3662419.txt
fa_orbit_submission_C32_screen_S27500_s42_K8_jid3662418.txt
fa_orbit_submission_C32_screen_S30000_s42_K1_jid3662421.txt
fa_orbit_submission_C32_screen_S30000_s42_K8_jid3662420.txt
fa_orbit_submission_C32_screen_S32500_s42_K1_jid3665920.txt
fa_orbit_submission_C32_screen_S32500_s42_K8_jid3665919.txt
fa_orbit_submission_C32_screen_S35000_s42_K1_jid3668011.txt
fa_orbit_submission_C32_screen_S35000_s42_K8_jid3668010.txt
fa_orbit_submission_C32_screen_S37500_s42_K1_jid3668649.txt
fa_orbit_submission_C32_screen_S37500_s42_K8_jid3668648.txt
fa_orbit_submission_C32_screen_S40000_s42_K1_jid3670799.txt
fa_orbit_submission_C32_screen_S40000_s42_K8_jid3670798.txt
fa_orbit_submission_C32_screen_S5000_s42_K1_jid3662216.txt
fa_orbit_submission_C32_screen_S7500_s42_K1_jid3662411.txt
fa_orbit_submission_C32_screen_S7500_s42_K8_jid3662410.txt
fa_orbit_submission_C4BACKFILL_cross_S40000_s42_K8_jid3680764.txt
fa_orbit_submission_C4BACKFILL_cross_S40000_s42_K8_jid3680765.txt
fa_orbit_submission_C4BACKFILL_cross_S40000_s42_K8_jid3680766.txt
fa_orbit_submission_C4BACKFILL_screen_S20000_s42_K1_jid3662223.txt
fa_orbit_submission_C4BACKFILL_screen_S30000_s42_K1_jid3662224.txt
fa_orbit_submission_C4BACKFILL_screen_S40000_s42_K1_jid3662225.txt
fa_orbit_submission_C4BACKFILL_screen_S40000_s42_K8_jid3684155.txt
fa_orbit_submission_C4L_1786038172477244742-627d96c6.txt
fa_orbit_submission_C4L_1786047832741741064-d5f916b0.txt
fa_orbit_submission_C4L_1786050012428592254-16b4b108.txt
fa_orbit_submission_C4L_1786052743947331089-816fe670.txt
fa_orbit_submission_C4L_1786054560338820300-09f373e3.txt
fa_orbit_submission_C4L_1786056694890618883-d777d6ef.txt
fa_orbit_submission_C4L_1786310422143759413-7d512809.txt
fa_orbit_submission_C4L_1786465302140788027-ae06a985.txt
fa_orbit_submission_C4L_1786476226703002708-005bbf54.txt
fa_orbit_submission_C4L_1786531886688280327-a2124841.txt
fa_orbit_submission_C4L_1786574118057925283-bbb6dd0e.txt
fa_orbit_submission_C4L_cross_S40000_s42_K8_jid3680758.txt
fa_orbit_submission_C4L_cross_S40000_s42_K8_jid3680759.txt
fa_orbit_submission_C4L_cross_S40000_s42_K8_jid3680760.txt
fa_orbit_submission_C4L_q9_S40000_s42_K1_jid3674669.txt
fa_orbit_submission_C4L_q9_S40000_s42_K8_jid3674668.txt
fa_orbit_submission_C4L_q9_S40000_s43_K1_jid3674671.txt
fa_orbit_submission_C4L_q9_S40000_s43_K8_jid3674670.txt
fa_orbit_submission_C4L_q9_S40000_s44_K1_jid3674673.txt
fa_orbit_submission_C4L_q9_S40000_s44_K8_jid3674672.txt
fa_orbit_submission_C4L_q9_S40000_s45_K1_jid3674675.txt
fa_orbit_submission_C4L_q9_S40000_s45_K8_jid3674674.txt
fa_orbit_submission_C4L_q9_S40000_s46_K1_jid3674677.txt
fa_orbit_submission_C4L_q9_S40000_s46_K8_jid3674676.txt
fa_orbit_submission_C4L_r3_S40000_s42_K8_jid3680738.txt
fa_orbit_submission_C4L_r3_S40000_s42_K8_jid3680739.txt
fa_orbit_submission_C4L_r3_S40000_s42_K8_jid3680740.txt
fa_orbit_submission_C4L_r3_S40000_s42_K8_jid3680741.txt
fa_orbit_submission_C4L_r3_S40000_s42_K8_jid3680742.txt
fa_orbit_submission_C4L_screen_S10000_s42_K1_jid3662185.txt
fa_orbit_submission_C4L_screen_S12500_s42_K1_jid3662187.txt
fa_orbit_submission_C4L_screen_S15000_s42_K1_jid3662189.txt
fa_orbit_submission_C4L_screen_S17500_s42_K1_jid3662191.txt
fa_orbit_submission_C4L_screen_S20000_s42_K1_jid3662193.txt
fa_orbit_submission_C4L_screen_S22500_s42_K1_jid3662195.txt
fa_orbit_submission_C4L_screen_S25000_s42_K1_jid3662197.txt
fa_orbit_submission_C4L_screen_S2500_s42_K1_jid3662179.txt
fa_orbit_submission_C4L_screen_S27500_s42_K1_jid3662199.txt
fa_orbit_submission_C4L_screen_S30000_s42_K1_jid3662201.txt
fa_orbit_submission_C4L_screen_S32500_s42_K1_jid3662203.txt
fa_orbit_submission_C4L_screen_S35000_s42_K1_jid3662205.txt
fa_orbit_submission_C4L_screen_S37500_s42_K1_jid3662207.txt
fa_orbit_submission_C4L_screen_S5000_s42_K1_jid3662181.txt
fa_orbit_submission_C4L_screen_S7500_s42_K1_jid3662183.txt
fa_orbit_submission_C8_1786054560465501451-9ffdd4d5.txt
fa_orbit_submission_C8_1786056695036465420-2fc0ef50.txt
fa_orbit_submission_C8_1786310422260085470-2e58ce21.txt
fa_orbit_submission_C8_1786465302361183738-e9235c57.txt
fa_orbit_submission_C8_1786476226855877847-ed87fb45.txt
fa_orbit_submission_C8_1786531886816349959-e2b269ee.txt
fa_orbit_submission_C8_1786574118385972969-63636a6b.txt
fa_orbit_submission_C8_conf_S40000_s42_K1_jid3658825.txt
fa_orbit_submission_C8_conf_S40000_s42_K8_jid3658824.txt
fa_orbit_submission_C8_conf_S40000_s43_K1_jid3658827.txt
fa_orbit_submission_C8_conf_S40000_s43_K8_jid3658826.txt
fa_orbit_submission_C8_conf_S40000_s44_K1_jid3658829.txt
fa_orbit_submission_C8_conf_S40000_s44_K8_jid3658828.txt
fa_orbit_submission_C8_conf_S40000_s45_K1_jid3658831.txt
fa_orbit_submission_C8_conf_S40000_s45_K8_jid3658830.txt
fa_orbit_submission_C8_conf_S40000_s46_K1_jid3658833.txt
fa_orbit_submission_C8_conf_S40000_s46_K8_jid3658832.txt
fa_orbit_submission_C8_cross_S40000_s42_K8_jid3680761.txt
fa_orbit_submission_C8_cross_S40000_s42_K8_jid3684137.txt
fa_orbit_submission_C8_cross_S40000_s42_K8_jid3684138.txt
fa_orbit_submission_C8_r3_S40000_s42_K8_jid3680743.txt
fa_orbit_submission_C8_r3_S40000_s42_K8_jid3680744.txt
fa_orbit_submission_C8_r3_S40000_s42_K8_jid3680745.txt
fa_orbit_submission_C8_r3_S40000_s42_K8_jid3680746.txt
fa_orbit_submission_C8_r3_S40000_s42_K8_jid3680747.txt
fa_orbit_submission_C8_r3_S40000_s42_K8_jid7654321.txt
fa_orbit_submission_C8_screen_S10000_s42_K1_jid3662186.txt
fa_orbit_submission_C8_screen_S12500_s42_K1_jid3662188.txt
fa_orbit_submission_C8_screen_S15000_s42_K1_jid3662190.txt
fa_orbit_submission_C8_screen_S17500_s42_K1_jid3662192.txt
fa_orbit_submission_C8_screen_S20000_s42_K1_jid3662194.txt
fa_orbit_submission_C8_screen_S22500_s42_K1_jid3662196.txt
fa_orbit_submission_C8_screen_S25000_s42_K1_jid3662198.txt
fa_orbit_submission_C8_screen_S2500_s42_K1_jid3662180.txt
fa_orbit_submission_C8_screen_S27500_s42_K1_jid3662200.txt
fa_orbit_submission_C8_screen_S30000_s42_K1_jid3662202.txt
fa_orbit_submission_C8_screen_S32500_s42_K1_jid3662204.txt
fa_orbit_submission_C8_screen_S35000_s42_K1_jid3662206.txt
fa_orbit_submission_C8_screen_S37500_s42_K1_jid3662208.txt
fa_orbit_submission_C8_screen_S5000_s42_K1_jid3662182.txt
fa_orbit_submission_C8_screen_S7500_s42_K1_jid3662184.txt
fa_orbit_submission_VANL_1786289214348203957-1be1edd5.txt
fa_orbit_submission_VANL_1786473966640260607-09fab791.txt
fa_orbit_submission_VANL_1786476227361783047-3dca770a.txt
fa_orbit_submission_VANL_1786531887194560967-be12b53a.txt
fa_orbit_submission_VANL_1786574118822694150-1c46ac53.txt
fa_orbit_submission_VANL_q9_S40000_s42_K1_jid3674659.txt
fa_orbit_submission_VANL_q9_S40000_s42_K8_jid3674658.txt
fa_orbit_submission_VANL_q9_S40000_s43_K1_jid3674661.txt
fa_orbit_submission_VANL_q9_S40000_s43_K8_jid3674660.txt
fa_orbit_submission_VANL_q9_S40000_s44_K1_jid3674663.txt
fa_orbit_submission_VANL_q9_S40000_s44_K8_jid3674662.txt
fa_orbit_submission_VANL_q9_S40000_s45_K1_jid3674665.txt
fa_orbit_submission_VANL_q9_S40000_s45_K8_jid3674664.txt
fa_orbit_submission_VANL_q9_S40000_s46_K1_jid3674667.txt
fa_orbit_submission_VANL_q9_S40000_s46_K8_jid3674666.txt
fa_orbit_submission_VANL_screen_S10000_s42_K1_jid3662406.txt
fa_orbit_submission_VANL_screen_S10000_s42_K1_jid3662812.txt
fa_orbit_submission_VANL_screen_S10000_s42_K1_jid3674686.txt
fa_orbit_submission_VANL_screen_S10000_s42_K8_jid3662405.txt
fa_orbit_submission_VANL_screen_S10000_s42_K8_jid3662811.txt
fa_orbit_submission_VANL_screen_S10000_s42_K8_jid3674685.txt
fa_orbit_submission_VANL_screen_S12500_s42_K1_jid3662814.txt
fa_orbit_submission_VANL_screen_S12500_s42_K1_jid3674688.txt
fa_orbit_submission_VANL_screen_S12500_s42_K8_jid3662813.txt
fa_orbit_submission_VANL_screen_S12500_s42_K8_jid3674687.txt
fa_orbit_submission_VANL_screen_S15000_s42_K1_jid3662816.txt
fa_orbit_submission_VANL_screen_S15000_s42_K1_jid3674690.txt
fa_orbit_submission_VANL_screen_S15000_s42_K8_jid3662815.txt
fa_orbit_submission_VANL_screen_S15000_s42_K8_jid3674689.txt
fa_orbit_submission_VANL_screen_S17500_s42_K1_jid3662818.txt
fa_orbit_submission_VANL_screen_S17500_s42_K1_jid3674692.txt
fa_orbit_submission_VANL_screen_S17500_s42_K8_jid3662817.txt
fa_orbit_submission_VANL_screen_S17500_s42_K8_jid3674691.txt
fa_orbit_submission_VANL_screen_S20000_s42_K1_jid3674694.txt
fa_orbit_submission_VANL_screen_S20000_s42_K8_jid3674693.txt
fa_orbit_submission_VANL_screen_S22500_s42_K1_jid3674696.txt
fa_orbit_submission_VANL_screen_S22500_s42_K8_jid3674695.txt
fa_orbit_submission_VANL_screen_S25000_s42_K1_jid3674698.txt
fa_orbit_submission_VANL_screen_S25000_s42_K8_jid3674697.txt
fa_orbit_submission_VANL_screen_S2500_s42_K1_jid3662400.txt
fa_orbit_submission_VANL_screen_S2500_s42_K1_jid3662806.txt
fa_orbit_submission_VANL_screen_S2500_s42_K1_jid3674680.txt
fa_orbit_submission_VANL_screen_S2500_s42_K8_jid3662399.txt
fa_orbit_submission_VANL_screen_S2500_s42_K8_jid3662805.txt
fa_orbit_submission_VANL_screen_S2500_s42_K8_jid3674679.txt
fa_orbit_submission_VANL_screen_S27500_s42_K1_jid3674700.txt
fa_orbit_submission_VANL_screen_S27500_s42_K8_jid3674699.txt
fa_orbit_submission_VANL_screen_S30000_s42_K1_jid3674702.txt
fa_orbit_submission_VANL_screen_S30000_s42_K8_jid3674701.txt
fa_orbit_submission_VANL_screen_S32500_s42_K1_jid3674704.txt
fa_orbit_submission_VANL_screen_S32500_s42_K8_jid3674703.txt
fa_orbit_submission_VANL_screen_S35000_s42_K1_jid3674706.txt
fa_orbit_submission_VANL_screen_S35000_s42_K8_jid3674705.txt
fa_orbit_submission_VANL_screen_S37500_s42_K1_jid3674708.txt
fa_orbit_submission_VANL_screen_S37500_s42_K8_jid3674707.txt
fa_orbit_submission_VANL_screen_S40000_s42_K1_jid3674710.txt
fa_orbit_submission_VANL_screen_S40000_s42_K8_jid3674709.txt
fa_orbit_submission_VANL_screen_S5000_s42_K1_jid3662402.txt
fa_orbit_submission_VANL_screen_S5000_s42_K1_jid3662808.txt
fa_orbit_submission_VANL_screen_S5000_s42_K1_jid3674682.txt
fa_orbit_submission_VANL_screen_S5000_s42_K8_jid3662401.txt
fa_orbit_submission_VANL_screen_S5000_s42_K8_jid3662807.txt
fa_orbit_submission_VANL_screen_S5000_s42_K8_jid3674681.txt
fa_orbit_submission_VANL_screen_S7500_s42_K1_jid3662404.txt
fa_orbit_submission_VANL_screen_S7500_s42_K1_jid3662810.txt
fa_orbit_submission_VANL_screen_S7500_s42_K1_jid3674684.txt
fa_orbit_submission_VANL_screen_S7500_s42_K8_jid3662403.txt
fa_orbit_submission_VANL_screen_S7500_s42_K8_jid3662809.txt
fa_orbit_submission_VANL_screen_S7500_s42_K8_jid3674683.txt
fa_orbit_submit.sh
fa_orbit_train.sbatch
fa_orbit_train_guardtests.sh
fa_orbit_traj_submit.sh
fa_orbit_wandb_readback.py
fa_orbit_worklog.md
fa_orbit_yixun_query.md
gen_trajectory_figures.py
gen_trajectory_pngs.py
p0_C16_8x8_2026-08-06_13-23-29_jid3646146_train.log
p0_C16_8x8_2026-08-06_13-23-29_jid3646146_vram.csv
p0_C16_8x8_2026-08-06_16-14-30_jid3646677_train.log
p0_C16_8x8_2026-08-06_16-14-30_jid3646677_vram.csv
p0_C32_8x8_2026-08-06_13-29-29_jid3646147_train.log
p0_C32_8x8_2026-08-06_13-29-29_jid3646147_vram.csv
p0_C32_8x8_2026-08-06_16-17-30_jid3646678_train.log
p0_C32_8x8_2026-08-06_16-17-30_jid3646678_vram.csv
p0_C4L_16x4_2026-08-05_18-34-48_jid3638698_train.log
p0_C4L_16x4_2026-08-05_18-34-48_jid3638698_vram.csv
p0_C4L_16x4_2026-08-05_21-56-16_jid3639675_train.log
p0_C4L_16x4_2026-08-05_21-56-16_jid3639675_vram.csv
p0_C4L_16x4_2026-08-06_12-14-24_jid3646048_train.log
p0_C4L_16x4_2026-08-06_12-14-24_jid3646048_vram.csv
p0_C4L_16x4_2026-08-06_15-46-43_jid3646671_train.log
p0_C4L_16x4_2026-08-06_15-46-43_jid3646671_vram.csv
p0_C4L_32x2_2026-08-05_18-16-39_jid3638618_train.log
p0_C4L_32x2_2026-08-05_18-16-39_jid3638618_vram.csv
p0_C4L_32x2_2026-08-05_18-21-48_jid3638639_train.log
p0_C4L_32x2_2026-08-05_18-21-48_jid3638639_vram.csv
p0_C4L_32x2_2026-08-05_18-34-49_jid3638694_train.log
p0_C4L_32x2_2026-08-05_18-34-49_jid3638694_vram.csv
p0_C4L_32x2_2026-08-05_21-52-15_jid3639671_train.log
p0_C4L_32x2_2026-08-05_21-52-15_jid3639671_vram.csv
p0_C4L_32x2_2026-08-06_12-08-53_jid3646044_train.log
p0_C4L_32x2_2026-08-06_12-08-53_jid3646044_vram.csv
p0_C4L_32x2_2026-08-06_15-42-41_jid3646667_train.log
p0_C4L_32x2_2026-08-06_15-42-41_jid3646667_vram.csv
p0_C4L_8x8_2026-08-05_20-01-57_jid3638702_train.log
p0_C4L_8x8_2026-08-05_20-01-57_jid3638702_vram.csv
p0_C4L_8x8_2026-08-05_23-46-39_jid3639679_train.log
p0_C4L_8x8_2026-08-05_23-46-39_jid3639679_vram.csv
p0_C4L_8x8_2026-08-06_12-17-30_jid3646052_train.log
p0_C4L_8x8_2026-08-06_12-17-30_jid3646052_vram.csv
p0_C4L_8x8_2026-08-06_16-10-46_jid3646675_train.log
p0_C4L_8x8_2026-08-06_16-10-46_jid3646675_vram.csv
p0_C8_16x4_2026-08-05_18-35-50_jid3638699_train.log
p0_C8_16x4_2026-08-05_18-35-50_jid3638699_vram.csv
p0_C8_16x4_2026-08-05_21-57-17_jid3639676_train.log
p0_C8_16x4_2026-08-05_21-57-17_jid3639676_vram.csv
p0_C8_16x4_2026-08-06_12-14-24_jid3646049_train.log
p0_C8_16x4_2026-08-06_12-14-24_jid3646049_vram.csv
p0_C8_16x4_2026-08-06_15-48-43_jid3646672_train.log
p0_C8_16x4_2026-08-06_15-48-43_jid3646672_vram.csv
p0_C8_32x2_2026-08-05_18-34-48_jid3638695_train.log
p0_C8_32x2_2026-08-05_18-34-48_jid3638695_vram.csv
p0_C8_32x2_2026-08-05_21-53-15_jid3639672_train.log
p0_C8_32x2_2026-08-05_21-53-15_jid3639672_vram.csv
p0_C8_32x2_2026-08-06_12-14-24_jid3646045_train.log
p0_C8_32x2_2026-08-06_12-14-24_jid3646045_vram.csv
p0_C8_32x2_2026-08-06_15-42-42_jid3646668_train.log
p0_C8_32x2_2026-08-06_15-42-42_jid3646668_vram.csv
p0_C8_8x8_2026-08-05_20-18-19_jid3638703_train.log
p0_C8_8x8_2026-08-05_20-18-19_jid3638703_vram.csv
p0_C8_8x8_2026-08-05_23-48-38_jid3639680_train.log
p0_C8_8x8_2026-08-05_23-48-38_jid3639680_vram.csv
p0_C8_8x8_2026-08-06_12-19-30_jid3646053_train.log
p0_C8_8x8_2026-08-06_12-19-30_jid3646053_vram.csv
p0_C8_8x8_2026-08-06_16-12-30_jid3646676_train.log
p0_C8_8x8_2026-08-06_16-12-30_jid3646676_vram.csv
p0_CKPT4_16x4_2026-08-05_20-28-00_jid3639145_train.log
p0_CKPT4_16x4_2026-08-05_20-28-00_jid3639145_vram.csv
p0_CKPT4_32x2_2026-08-05_18-34-49_jid3638704_train.log
p0_CKPT4_32x2_2026-08-05_18-34-49_jid3638704_vram.csv
p0_FA1_16x4_2026-08-05_18-34-49_jid3638697_train.log
p0_FA1_16x4_2026-08-05_18-34-49_jid3638697_vram.csv
p0_FA1_16x4_2026-08-05_21-55-16_jid3639674_train.log
p0_FA1_16x4_2026-08-05_21-55-16_jid3639674_vram.csv
p0_FA1_16x4_2026-08-06_12-14-24_jid3646047_train.log
p0_FA1_16x4_2026-08-06_12-14-24_jid3646047_vram.csv
p0_FA1_16x4_2026-08-06_15-45-43_jid3646670_train.log
p0_FA1_16x4_2026-08-06_15-45-43_jid3646670_vram.csv
p0_FA1_32x2_2026-08-05_18-18-16_jid3638630_train.log
p0_FA1_32x2_2026-08-05_18-18-16_jid3638630_vram.csv
p0_FA1_32x2_2026-08-05_18-21-47_jid3638638_train.log
p0_FA1_32x2_2026-08-05_18-21-47_jid3638638_vram.csv
p0_FA1_32x2_2026-08-05_18-33-48_jid3638693_train.log
p0_FA1_32x2_2026-08-05_18-33-48_jid3638693_vram.csv
p0_FA1_32x2_2026-08-05_21-51-15_jid3639670_train.log
p0_FA1_32x2_2026-08-05_21-51-15_jid3639670_vram.csv
p0_FA1_32x2_2026-08-06_12-08-55_jid3646043_train.log
p0_FA1_32x2_2026-08-06_12-08-55_jid3646043_vram.csv
p0_FA1_32x2_2026-08-06_15-42-42_jid3646666_train.log
p0_FA1_32x2_2026-08-06_15-42-42_jid3646666_vram.csv
p0_FA1_8x8_2026-08-05_18-21-47_jid3638646_train.log
p0_FA1_8x8_2026-08-05_18-21-47_jid3638646_vram.csv
p0_FA1_8x8_2026-08-05_20-00-57_jid3638701_train.log
p0_FA1_8x8_2026-08-05_20-00-57_jid3638701_vram.csv
p0_FA1_8x8_2026-08-05_23-40-25_jid3639678_train.log
p0_FA1_8x8_2026-08-05_23-40-25_jid3639678_vram.csv
p0_FA1_8x8_2026-08-06_12-16-24_jid3646051_train.log
p0_FA1_8x8_2026-08-06_12-16-24_jid3646051_vram.csv
p0_FA1_8x8_2026-08-06_16-09-46_jid3646674_train.log
p0_FA1_8x8_2026-08-06_16-09-46_jid3646674_vram.csv
p0_VAN_16x4_2026-08-05_18-21-47_jid3638641_train.log
p0_VAN_16x4_2026-08-05_18-21-47_jid3638641_vram.csv
p0_VAN_16x4_2026-08-05_18-33-49_jid3638696_train.log
p0_VAN_16x4_2026-08-05_18-33-49_jid3638696_vram.csv
p0_VAN_16x4_2026-08-05_21-54-16_jid3639673_train.log
p0_VAN_16x4_2026-08-05_21-54-16_jid3639673_vram.csv
p0_VAN_16x4_2026-08-06_12-12-23_jid3646046_train.log
p0_VAN_16x4_2026-08-06_12-12-23_jid3646046_vram.csv
p0_VAN_16x4_2026-08-06_15-43-43_jid3646669_train.log
p0_VAN_16x4_2026-08-06_15-43-43_jid3646669_vram.csv
p0_VAN_32x2_2026-08-05_18-20-48_jid3638637_train.log
p0_VAN_32x2_2026-08-05_18-20-48_jid3638637_vram.csv
p0_VAN_32x2_2026-08-05_18-33-49_jid3638692_train.log
p0_VAN_32x2_2026-08-05_18-33-49_jid3638692_vram.csv
p0_VAN_32x2_2026-08-05_21-50-15_jid3639669_train.log
p0_VAN_32x2_2026-08-05_21-50-15_jid3639669_vram.csv
p0_VAN_32x2_2026-08-06_12-08-54_jid3646042_train.log
p0_VAN_32x2_2026-08-06_12-08-54_jid3646042_vram.csv
p0_VAN_32x2_2026-08-06_15-42-42_jid3646665_train.log
p0_VAN_32x2_2026-08-06_15-42-42_jid3646665_vram.csv
p0_VAN_8x8_2026-08-05_18-20-33_jid3638645_train.log
p0_VAN_8x8_2026-08-05_18-20-33_jid3638645_vram.csv
p0_VAN_8x8_2026-08-05_19-59-57_jid3638700_train.log
p0_VAN_8x8_2026-08-05_19-59-57_jid3638700_vram.csv
p0_VAN_8x8_2026-08-05_23-19-22_jid3639677_train.log
p0_VAN_8x8_2026-08-05_23-19-22_jid3639677_vram.csv
p0_VAN_8x8_2026-08-06_12-14-24_jid3646050_train.log
p0_VAN_8x8_2026-08-06_12-14-24_jid3646050_vram.csv
p0_VAN_8x8_2026-08-06_16-08-46_jid3646673_train.log
p0_VAN_8x8_2026-08-06_16-08-46_jid3646673_vram.csv
p0_collect.py
p0_manifest_1334933-1786032532843128131-8f21c960.txt
p0_manifest_72a8114-1785969226421855487-c8d5b51f.txt
p0_manifest_86a752b-1785980874148140138-06d348d6.txt
p0_manifest_9bf1936-1786033425104073952-d8d84328.txt
p0_manifest_aa4bc18-1785968431124626318-df9602ea.txt
p0_manifest_bd96575-1786045321510462046-a3ed28eb.txt
p0_manifest_bd96575-1786045321895684456-ae4c2f92.txt
p0_profile.sbatch
p0_report_1334933.md
p0_report_86a752b.md
p0_report_batched_matrix.md
p0_report_batched_spot.md
p0_report_spot_9bf1936.md
p0_runner.py
p0_submit_matrix.sh
plan_fa_orbit.md
slurm_equivprobe_exp11-equivprobe_3646626.out
slurm_equivprobe_exp11-equivprobe_3646634.out
slurm_equivprobe_exp11-equivprobe_3646653.out
slurm_equivprobe_fa_orbit_equiv_probe.sbatch_3646612.out
slurm_equivprobe_fa_orbit_equiv_probe.sbatch_3646615.out
slurm_equivprobe_fa_orbit_equiv_probe.sbatch_3646616.out
slurm_p0_p0-C16_8x8-w6_3646146.out
slurm_p0_p0-C16_8x8-w6_3646677.out
slurm_p0_p0-C32_8x8-w6_3646147.out
slurm_p0_p0-C32_8x8-w6_3646678.out
slurm_p0_p0-C4L_16x4-w6_3638643.out
slurm_p0_p0-C4L_16x4-w6_3638698.out
slurm_p0_p0-C4L_16x4-w6_3639675.out
slurm_p0_p0-C4L_16x4-w6_3646048.out
slurm_p0_p0-C4L_16x4-w6_3646671.out
slurm_p0_p0-C4L_32x2-w6_3638639.out
slurm_p0_p0-C4L_32x2-w6_3638694.out
slurm_p0_p0-C4L_32x2-w6_3639671.out
slurm_p0_p0-C4L_32x2-w6_3646044.out
slurm_p0_p0-C4L_32x2-w6_3646667.out
slurm_p0_p0-C4L_8x8-w6_3638647.out
slurm_p0_p0-C4L_8x8-w6_3638702.out
slurm_p0_p0-C4L_8x8-w6_3639679.out
slurm_p0_p0-C4L_8x8-w6_3646052.out
slurm_p0_p0-C4L_8x8-w6_3646675.out
slurm_p0_p0-C8_16x4-w6_3638644.out
slurm_p0_p0-C8_16x4-w6_3638699.out
slurm_p0_p0-C8_16x4-w6_3639676.out
slurm_p0_p0-C8_16x4-w6_3646049.out
slurm_p0_p0-C8_16x4-w6_3646672.out
slurm_p0_p0-C8_32x2-w6_3638640.out
slurm_p0_p0-C8_32x2-w6_3638695.out
slurm_p0_p0-C8_32x2-w6_3639672.out
slurm_p0_p0-C8_32x2-w6_3646045.out
slurm_p0_p0-C8_32x2-w6_3646668.out
slurm_p0_p0-C8_8x8-w6_3638648.out
slurm_p0_p0-C8_8x8-w6_3638703.out
slurm_p0_p0-C8_8x8-w6_3639680.out
slurm_p0_p0-C8_8x8-w6_3646053.out
slurm_p0_p0-C8_8x8-w6_3646676.out
slurm_p0_p0-CKPT4_32x2-w6_3638649.out
slurm_p0_p0-CKPT4_32x2-w6_3638704.out
slurm_p0_p0-FA1_16x4-w6_3638642.out
slurm_p0_p0-FA1_16x4-w6_3638697.out
slurm_p0_p0-FA1_16x4-w6_3639674.out
slurm_p0_p0-FA1_16x4-w6_3646047.out
slurm_p0_p0-FA1_16x4-w6_3646670.out
slurm_p0_p0-FA1_32x2-w6_3638638.out
slurm_p0_p0-FA1_32x2-w6_3638693.out
slurm_p0_p0-FA1_32x2-w6_3639670.out
slurm_p0_p0-FA1_32x2-w6_3646043.out
slurm_p0_p0-FA1_32x2-w6_3646666.out
slurm_p0_p0-FA1_8x8-w6_3638646.out
slurm_p0_p0-FA1_8x8-w6_3638701.out
slurm_p0_p0-FA1_8x8-w6_3639678.out
slurm_p0_p0-FA1_8x8-w6_3646051.out
slurm_p0_p0-FA1_8x8-w6_3646674.out
slurm_p0_p0-VAN_16x4-w6_3638641.out
slurm_p0_p0-VAN_16x4-w6_3638696.out
slurm_p0_p0-VAN_16x4-w6_3639673.out
slurm_p0_p0-VAN_16x4-w6_3646046.out
slurm_p0_p0-VAN_16x4-w6_3646669.out
slurm_p0_p0-VAN_32x2-w6_3638637.out
slurm_p0_p0-VAN_32x2-w6_3638692.out
slurm_p0_p0-VAN_32x2-w6_3639669.out
slurm_p0_p0-VAN_32x2-w6_3646042.out
slurm_p0_p0-VAN_32x2-w6_3646665.out
slurm_p0_p0-VAN_8x8-w6_3638645.out
slurm_p0_p0-VAN_8x8-w6_3638700.out
slurm_p0_p0-VAN_8x8-w6_3639677.out
slurm_p0_p0-VAN_8x8-w6_3646050.out
slurm_p0_p0-VAN_8x8-w6_3646673.out
slurm_p0_p0-smoke-C4L_32x2_3638618.out
slurm_p0_p0-smoke2-FA1_32x2_3638630.out
slurm_p0_p0-supp-CKPT4_16x4_3639145.out
slurm_screen_exp11-screen-C16-10000-s42-K8_3651419.out
slurm_screen_exp11-screen-C16-15000-s42-K8_3651420.out
slurm_screen_exp11-screen-C16-20000-s42-K8_3653040.out
slurm_screen_exp11-screen-C16-5000-s42-K8_3649927.out
slurm_screen_exp11-screen-C16-conf-40000-s42-K1_3657721.out
slurm_screen_exp11-screen-C16-conf-40000-s42-K1_3658835.out
slurm_screen_exp11-screen-C16-conf-40000-s42-K8_3657720.out
slurm_screen_exp11-screen-C16-conf-40000-s42-K8_3658834.out
slurm_screen_exp11-screen-C16-conf-40000-s43-K1_3657723.out
slurm_screen_exp11-screen-C16-conf-40000-s43-K1_3658837.out
slurm_screen_exp11-screen-C16-conf-40000-s43-K8_3657722.out
slurm_screen_exp11-screen-C16-conf-40000-s43-K8_3658836.out
slurm_screen_exp11-screen-C16-conf-40000-s44-K1_3657725.out
slurm_screen_exp11-screen-C16-conf-40000-s44-K1_3658839.out
slurm_screen_exp11-screen-C16-conf-40000-s44-K8_3657724.out
slurm_screen_exp11-screen-C16-conf-40000-s44-K8_3658838.out
slurm_screen_exp11-screen-C16-conf-40000-s45-K1_3657727.out
slurm_screen_exp11-screen-C16-conf-40000-s45-K1_3658841.out
slurm_screen_exp11-screen-C16-conf-40000-s45-K8_3657726.out
slurm_screen_exp11-screen-C16-conf-40000-s45-K8_3658840.out
slurm_screen_exp11-screen-C16-conf-40000-s46-K1_3657729.out
slurm_screen_exp11-screen-C16-conf-40000-s46-K1_3658843.out
slurm_screen_exp11-screen-C16-conf-40000-s46-K8_3657728.out
slurm_screen_exp11-screen-C16-conf-40000-s46-K8_3658842.out
slurm_screen_exp11-screen-C16-cross-40000-s42-K8_3657751.out
slurm_screen_exp11-screen-C16-cross-40000-s42-K8_3657752.out
slurm_screen_exp11-screen-C16-cross-40000-s42-K8_3657753.out
slurm_screen_exp11-screen-C16-cross-40000-s42-K8_3680762.out
slurm_screen_exp11-screen-C16-cross-40000-s42-K8_3684139.out
slurm_screen_exp11-screen-C16-cross-40000-s42-K8_3684140.out
slurm_screen_exp11-screen-C16-r3-40000-s42-K8_3657740.out
slurm_screen_exp11-screen-C16-r3-40000-s42-K8_3657741.out
slurm_screen_exp11-screen-C16-r3-40000-s42-K8_3657742.out
slurm_screen_exp11-screen-C16-r3-40000-s42-K8_3657743.out
slurm_screen_exp11-screen-C16-r3-40000-s42-K8_3657744.out
slurm_screen_exp11-screen-C16-r3-40000-s42-K8_3680748.out
slurm_screen_exp11-screen-C16-r3-40000-s42-K8_3680749.out
slurm_screen_exp11-screen-C16-r3-40000-s42-K8_3680750.out
slurm_screen_exp11-screen-C16-r3-40000-s42-K8_3680751.out
slurm_screen_exp11-screen-C16-r3-40000-s42-K8_3680752.out
slurm_screen_exp11-screen-C16-screen-10000-s42-K1_3662210.out
slurm_screen_exp11-screen-C16-screen-15000-s42-K1_3662211.out
slurm_screen_exp11-screen-C16-screen-20000-s42-K1_3662212.out
slurm_screen_exp11-screen-C16-screen-25000-s42-K1_3662213.out
slurm_screen_exp11-screen-C16-screen-25000-s42-K8_3661451.out
slurm_screen_exp11-screen-C16-screen-30000-s42-K1_3662214.out
slurm_screen_exp11-screen-C16-screen-30000-s42-K8_3662050.out
slurm_screen_exp11-screen-C16-screen-35000-s42-K1_3662215.out
slurm_screen_exp11-screen-C16-screen-35000-s42-K8_3662051.out
slurm_screen_exp11-screen-C16-screen-5000-s42-K1_3662209.out
slurm_screen_exp11-screen-C32-10000-s42-K8_3653041.out
slurm_screen_exp11-screen-C32-10000-s42-K8_3653435.out
slurm_screen_exp11-screen-C32-5000-s42-K8_3651421.out
slurm_screen_exp11-screen-C32-conf-40000-s42-K1_3672839.out
slurm_screen_exp11-screen-C32-conf-40000-s42-K8_3672838.out
slurm_screen_exp11-screen-C32-conf-40000-s43-K1_3672841.out
slurm_screen_exp11-screen-C32-conf-40000-s43-K8_3672840.out
slurm_screen_exp11-screen-C32-conf-40000-s44-K1_3672843.out
slurm_screen_exp11-screen-C32-conf-40000-s44-K8_3672842.out
slurm_screen_exp11-screen-C32-conf-40000-s45-K1_3672845.out
slurm_screen_exp11-screen-C32-conf-40000-s45-K8_3672844.out
slurm_screen_exp11-screen-C32-conf-40000-s46-K1_3672847.out
slurm_screen_exp11-screen-C32-conf-40000-s46-K8_3672846.out
slurm_screen_exp11-screen-C32-cross-40000-s42-K8_3680763.out
slurm_screen_exp11-screen-C32-cross-40000-s42-K8_3684143.out
slurm_screen_exp11-screen-C32-cross-40000-s42-K8_3684144.out
slurm_screen_exp11-screen-C32-r3-40000-s42-K8_3680753.out
slurm_screen_exp11-screen-C32-r3-40000-s42-K8_3680754.out
slurm_screen_exp11-screen-C32-r3-40000-s42-K8_3680755.out
slurm_screen_exp11-screen-C32-r3-40000-s42-K8_3680756.out
slurm_screen_exp11-screen-C32-r3-40000-s42-K8_3680757.out
slurm_screen_exp11-screen-C32-screen-10000-s42-K1_3662217.out
slurm_screen_exp11-screen-C32-screen-10000-s42-K8_3657758.out
slurm_screen_exp11-screen-C32-screen-10000-s42-K8_3661452.out
slurm_screen_exp11-screen-C32-screen-12500-s42-K1_3662413.out
slurm_screen_exp11-screen-C32-screen-12500-s42-K8_3662412.out
slurm_screen_exp11-screen-C32-screen-15000-s42-K1_3662218.out
slurm_screen_exp11-screen-C32-screen-15000-s42-K8_3657759.out
slurm_screen_exp11-screen-C32-screen-15000-s42-K8_3661453.out
slurm_screen_exp11-screen-C32-screen-17500-s42-K1_3662415.out
slurm_screen_exp11-screen-C32-screen-17500-s42-K8_3662414.out
slurm_screen_exp11-screen-C32-screen-20000-s42-K1_3662219.out
slurm_screen_exp11-screen-C32-screen-20000-s42-K8_3657760.out
slurm_screen_exp11-screen-C32-screen-20000-s42-K8_3661454.out
slurm_screen_exp11-screen-C32-screen-22500-s42-K1_3662417.out
slurm_screen_exp11-screen-C32-screen-22500-s42-K8_3662416.out
slurm_screen_exp11-screen-C32-screen-2500-s42-K1_3662408.out
slurm_screen_exp11-screen-C32-screen-2500-s42-K8_3662407.out
slurm_screen_exp11-screen-C32-screen-25000-s42-K1_3662220.out
slurm_screen_exp11-screen-C32-screen-25000-s42-K8_3661455.out
slurm_screen_exp11-screen-C32-screen-27500-s42-K1_3662419.out
slurm_screen_exp11-screen-C32-screen-27500-s42-K8_3662418.out
slurm_screen_exp11-screen-C32-screen-30000-s42-K1_3662421.out
slurm_screen_exp11-screen-C32-screen-30000-s42-K8_3662420.out
slurm_screen_exp11-screen-C32-screen-32500-s42-K1_3665920.out
slurm_screen_exp11-screen-C32-screen-32500-s42-K8_3665919.out
slurm_screen_exp11-screen-C32-screen-35000-s42-K1_3668011.out
slurm_screen_exp11-screen-C32-screen-35000-s42-K8_3668010.out
slurm_screen_exp11-screen-C32-screen-37500-s42-K1_3668649.out
slurm_screen_exp11-screen-C32-screen-37500-s42-K8_3668648.out
slurm_screen_exp11-screen-C32-screen-40000-s42-K1_3670799.out
slurm_screen_exp11-screen-C32-screen-40000-s42-K8_3670798.out
slurm_screen_exp11-screen-C32-screen-5000-s42-K1_3662216.out
slurm_screen_exp11-screen-C32-screen-7500-s42-K1_3662411.out
slurm_screen_exp11-screen-C32-screen-7500-s42-K8_3662410.out
slurm_screen_exp11-screen-C4BACKFILL-20000-s42-K8_3649915.out
slurm_screen_exp11-screen-C4BACKFILL-30000-s42-K8_3649916.out
slurm_screen_exp11-screen-C4BACKFILL-40000-s42-K8_3653361.out
slurm_screen_exp11-screen-C4BACKFILL-40000-s42-K8_3653362.out
slurm_screen_exp11-screen-C4BACKFILL-40000-s42-K8_3653363.out
slurm_screen_exp11-screen-C4BACKFILL-40000-s42-K8_3653364.out
slurm_screen_exp11-screen-C4BACKFILL-40000-s42-K8_3653431.out
slurm_screen_exp11-screen-C4BACKFILL-40000-s42-K8_3653432.out
slurm_screen_exp11-screen-C4BACKFILL-40000-s42-K8_3653433.out
slurm_screen_exp11-screen-C4BACKFILL-40000-s42-K8_3653434.out
slurm_screen_exp11-screen-C4BACKFILL-cross-40000-s42-K8_3657755.out
slurm_screen_exp11-screen-C4BACKFILL-cross-40000-s42-K8_3657756.out
slurm_screen_exp11-screen-C4BACKFILL-cross-40000-s42-K8_3657757.out
slurm_screen_exp11-screen-C4BACKFILL-cross-40000-s42-K8_3680764.out
slurm_screen_exp11-screen-C4BACKFILL-cross-40000-s42-K8_3680765.out
slurm_screen_exp11-screen-C4BACKFILL-cross-40000-s42-K8_3680766.out
slurm_screen_exp11-screen-C4BACKFILL-screen-20000-s42-K1_3662223.out
slurm_screen_exp11-screen-C4BACKFILL-screen-30000-s42-K1_3662224.out
slurm_screen_exp11-screen-C4BACKFILL-screen-40000-s42-K1_3662225.out
slurm_screen_exp11-screen-C4BACKFILL-screen-40000-s42-K8_3657754.out
slurm_screen_exp11-screen-C4BACKFILL-screen-40000-s42-K8_3684155.out
slurm_screen_exp11-screen-C4L-10000-s42-K8_3649920.out
slurm_screen_exp11-screen-C4L-12500-s42-K8_3649921.out
slurm_screen_exp11-screen-C4L-15000-s42-K8_3649922.out
slurm_screen_exp11-screen-C4L-17500-s42-K8_3651404.out
slurm_screen_exp11-screen-C4L-20000-s42-K8_3651405.out
slurm_screen_exp11-screen-C4L-22500-s42-K8_3651406.out
slurm_screen_exp11-screen-C4L-2500-s42-K8_3649917.out
slurm_screen_exp11-screen-C4L-25000-s42-K8_3651407.out
slurm_screen_exp11-screen-C4L-27500-s42-K8_3651408.out
slurm_screen_exp11-screen-C4L-30000-s42-K8_3651409.out
slurm_screen_exp11-screen-C4L-32500-s42-K8_3651410.out
slurm_screen_exp11-screen-C4L-35000-s42-K8_3651411.out
slurm_screen_exp11-screen-C4L-37500-s42-K8_3651412.out
slurm_screen_exp11-screen-C4L-40000-s42-K1_3651384.out
slurm_screen_exp11-screen-C4L-40000-s42-K8_3651383.out
slurm_screen_exp11-screen-C4L-40000-s42-K8_3653345.out
slurm_screen_exp11-screen-C4L-40000-s42-K8_3653346.out
slurm_screen_exp11-screen-C4L-40000-s42-K8_3653347.out
slurm_screen_exp11-screen-C4L-40000-s42-K8_3653348.out
slurm_screen_exp11-screen-C4L-40000-s42-K8_3653349.out
slurm_screen_exp11-screen-C4L-40000-s42-K8_3653355.out
slurm_screen_exp11-screen-C4L-40000-s42-K8_3653356.out
slurm_screen_exp11-screen-C4L-40000-s42-K8_3653357.out
slurm_screen_exp11-screen-C4L-40000-s42-K8_3653415.out
slurm_screen_exp11-screen-C4L-40000-s42-K8_3653416.out
slurm_screen_exp11-screen-C4L-40000-s42-K8_3653417.out
slurm_screen_exp11-screen-C4L-40000-s42-K8_3653418.out
slurm_screen_exp11-screen-C4L-40000-s42-K8_3653419.out
slurm_screen_exp11-screen-C4L-40000-s42-K8_3653425.out
slurm_screen_exp11-screen-C4L-40000-s42-K8_3653426.out
slurm_screen_exp11-screen-C4L-40000-s42-K8_3653427.out
slurm_screen_exp11-screen-C4L-40000-s43-K1_3651386.out
slurm_screen_exp11-screen-C4L-40000-s43-K8_3651385.out
slurm_screen_exp11-screen-C4L-40000-s44-K1_3651388.out
slurm_screen_exp11-screen-C4L-40000-s44-K8_3651387.out
slurm_screen_exp11-screen-C4L-40000-s45-K1_3651390.out
slurm_screen_exp11-screen-C4L-40000-s45-K8_3651389.out
slurm_screen_exp11-screen-C4L-40000-s46-K1_3651392.out
slurm_screen_exp11-screen-C4L-40000-s46-K8_3651391.out
slurm_screen_exp11-screen-C4L-5000-s42-K8_3649918.out
slurm_screen_exp11-screen-C4L-7500-s42-K8_3649919.out
slurm_screen_exp11-screen-C4L-cross-40000-s42-K8_3657745.out
slurm_screen_exp11-screen-C4L-cross-40000-s42-K8_3657746.out
slurm_screen_exp11-screen-C4L-cross-40000-s42-K8_3657747.out
slurm_screen_exp11-screen-C4L-cross-40000-s42-K8_3680758.out
slurm_screen_exp11-screen-C4L-cross-40000-s42-K8_3680759.out
slurm_screen_exp11-screen-C4L-cross-40000-s42-K8_3680760.out
slurm_screen_exp11-screen-C4L-q9-40000-s42-K1_3674669.out
slurm_screen_exp11-screen-C4L-q9-40000-s42-K8_3674668.out
slurm_screen_exp11-screen-C4L-q9-40000-s43-K1_3674671.out
slurm_screen_exp11-screen-C4L-q9-40000-s43-K8_3674670.out
slurm_screen_exp11-screen-C4L-q9-40000-s44-K1_3674673.out
slurm_screen_exp11-screen-C4L-q9-40000-s44-K8_3674672.out
slurm_screen_exp11-screen-C4L-q9-40000-s45-K1_3674675.out
slurm_screen_exp11-screen-C4L-q9-40000-s45-K8_3674674.out
slurm_screen_exp11-screen-C4L-q9-40000-s46-K1_3674677.out
slurm_screen_exp11-screen-C4L-q9-40000-s46-K8_3674676.out
slurm_screen_exp11-screen-C4L-r3-40000-s42-K8_3657730.out
slurm_screen_exp11-screen-C4L-r3-40000-s42-K8_3657731.out
slurm_screen_exp11-screen-C4L-r3-40000-s42-K8_3657732.out
slurm_screen_exp11-screen-C4L-r3-40000-s42-K8_3657733.out
slurm_screen_exp11-screen-C4L-r3-40000-s42-K8_3657734.out
slurm_screen_exp11-screen-C4L-r3-40000-s42-K8_3680738.out
slurm_screen_exp11-screen-C4L-r3-40000-s42-K8_3680739.out
slurm_screen_exp11-screen-C4L-r3-40000-s42-K8_3680740.out
slurm_screen_exp11-screen-C4L-r3-40000-s42-K8_3680741.out
slurm_screen_exp11-screen-C4L-r3-40000-s42-K8_3680742.out
slurm_screen_exp11-screen-C4L-screen-10000-s42-K1_3662185.out
slurm_screen_exp11-screen-C4L-screen-12500-s42-K1_3662187.out
slurm_screen_exp11-screen-C4L-screen-15000-s42-K1_3662189.out
slurm_screen_exp11-screen-C4L-screen-17500-s42-K1_3662191.out
slurm_screen_exp11-screen-C4L-screen-20000-s42-K1_3662193.out
slurm_screen_exp11-screen-C4L-screen-22500-s42-K1_3662195.out
slurm_screen_exp11-screen-C4L-screen-2500-s42-K1_3662179.out
slurm_screen_exp11-screen-C4L-screen-25000-s42-K1_3662197.out
slurm_screen_exp11-screen-C4L-screen-27500-s42-K1_3662199.out
slurm_screen_exp11-screen-C4L-screen-30000-s42-K1_3662201.out
slurm_screen_exp11-screen-C4L-screen-32500-s42-K1_3662203.out
slurm_screen_exp11-screen-C4L-screen-35000-s42-K1_3662205.out
slurm_screen_exp11-screen-C4L-screen-37500-s42-K1_3662207.out
slurm_screen_exp11-screen-C4L-screen-5000-s42-K1_3662181.out
slurm_screen_exp11-screen-C4L-screen-7500-s42-K1_3662183.out
slurm_screen_exp11-screen-C4backfill-20000_3649599.out
slurm_screen_exp11-screen-C4backfill-30000_3649600.out
slurm_screen_exp11-screen-C8-10000-s42-K8_3649926.out
slurm_screen_exp11-screen-C8-12500-s42-K8_3651413.out
slurm_screen_exp11-screen-C8-15000-s42-K8_3651414.out
slurm_screen_exp11-screen-C8-17500-s42-K8_3651415.out
slurm_screen_exp11-screen-C8-20000-s42-K8_3651416.out
slurm_screen_exp11-screen-C8-22500-s42-K8_3651417.out
slurm_screen_exp11-screen-C8-2500-s42-K8_3649923.out
slurm_screen_exp11-screen-C8-25000-s42-K8_3651418.out
slurm_screen_exp11-screen-C8-27500-s42-K8_3653035.out
slurm_screen_exp11-screen-C8-30000-s42-K8_3653036.out
slurm_screen_exp11-screen-C8-32500-s42-K8_3653037.out
slurm_screen_exp11-screen-C8-35000-s42-K8_3653038.out
slurm_screen_exp11-screen-C8-37500-s42-K8_3653039.out
slurm_screen_exp11-screen-C8-40000-s42-K1_3653120.out
slurm_screen_exp11-screen-C8-40000-s42-K1_3653336.out
slurm_screen_exp11-screen-C8-40000-s42-K1_3653406.out
slurm_screen_exp11-screen-C8-40000-s42-K8_3653119.out
slurm_screen_exp11-screen-C8-40000-s42-K8_3653335.out
slurm_screen_exp11-screen-C8-40000-s42-K8_3653350.out
slurm_screen_exp11-screen-C8-40000-s42-K8_3653351.out
slurm_screen_exp11-screen-C8-40000-s42-K8_3653352.out
slurm_screen_exp11-screen-C8-40000-s42-K8_3653353.out
slurm_screen_exp11-screen-C8-40000-s42-K8_3653354.out
slurm_screen_exp11-screen-C8-40000-s42-K8_3653358.out
slurm_screen_exp11-screen-C8-40000-s42-K8_3653359.out
slurm_screen_exp11-screen-C8-40000-s42-K8_3653360.out
slurm_screen_exp11-screen-C8-40000-s42-K8_3653405.out
slurm_screen_exp11-screen-C8-40000-s42-K8_3653420.out
slurm_screen_exp11-screen-C8-40000-s42-K8_3653421.out
slurm_screen_exp11-screen-C8-40000-s42-K8_3653422.out
slurm_screen_exp11-screen-C8-40000-s42-K8_3653423.out
slurm_screen_exp11-screen-C8-40000-s42-K8_3653424.out
slurm_screen_exp11-screen-C8-40000-s42-K8_3653428.out
slurm_screen_exp11-screen-C8-40000-s42-K8_3653429.out
slurm_screen_exp11-screen-C8-40000-s42-K8_3653430.out
slurm_screen_exp11-screen-C8-40000-s43-K1_3653122.out
slurm_screen_exp11-screen-C8-40000-s43-K1_3653338.out
slurm_screen_exp11-screen-C8-40000-s43-K1_3653408.out
slurm_screen_exp11-screen-C8-40000-s43-K8_3653121.out
slurm_screen_exp11-screen-C8-40000-s43-K8_3653337.out
slurm_screen_exp11-screen-C8-40000-s43-K8_3653407.out
slurm_screen_exp11-screen-C8-40000-s44-K1_3653124.out
slurm_screen_exp11-screen-C8-40000-s44-K1_3653340.out
slurm_screen_exp11-screen-C8-40000-s44-K1_3653410.out
slurm_screen_exp11-screen-C8-40000-s44-K8_3653123.out
slurm_screen_exp11-screen-C8-40000-s44-K8_3653339.out
slurm_screen_exp11-screen-C8-40000-s44-K8_3653409.out
slurm_screen_exp11-screen-C8-40000-s45-K1_3653126.out
slurm_screen_exp11-screen-C8-40000-s45-K1_3653342.out
slurm_screen_exp11-screen-C8-40000-s45-K1_3653412.out
slurm_screen_exp11-screen-C8-40000-s45-K8_3653125.out
slurm_screen_exp11-screen-C8-40000-s45-K8_3653341.out
slurm_screen_exp11-screen-C8-40000-s45-K8_3653411.out
slurm_screen_exp11-screen-C8-40000-s46-K1_3653128.out
slurm_screen_exp11-screen-C8-40000-s46-K1_3653344.out
slurm_screen_exp11-screen-C8-40000-s46-K1_3653414.out
slurm_screen_exp11-screen-C8-40000-s46-K8_3653127.out
slurm_screen_exp11-screen-C8-40000-s46-K8_3653343.out
slurm_screen_exp11-screen-C8-40000-s46-K8_3653413.out
slurm_screen_exp11-screen-C8-5000-s42-K8_3649924.out
slurm_screen_exp11-screen-C8-7500-s42-K8_3649925.out
slurm_screen_exp11-screen-C8-conf-40000-s42-K1_3657711.out
slurm_screen_exp11-screen-C8-conf-40000-s42-K1_3658825.out
slurm_screen_exp11-screen-C8-conf-40000-s42-K8_3657710.out
slurm_screen_exp11-screen-C8-conf-40000-s42-K8_3658824.out
slurm_screen_exp11-screen-C8-conf-40000-s43-K1_3657713.out
slurm_screen_exp11-screen-C8-conf-40000-s43-K1_3658827.out
slurm_screen_exp11-screen-C8-conf-40000-s43-K8_3657712.out
slurm_screen_exp11-screen-C8-conf-40000-s43-K8_3658826.out
slurm_screen_exp11-screen-C8-conf-40000-s44-K1_3657715.out
slurm_screen_exp11-screen-C8-conf-40000-s44-K1_3658829.out
slurm_screen_exp11-screen-C8-conf-40000-s44-K8_3657714.out
slurm_screen_exp11-screen-C8-conf-40000-s44-K8_3658828.out
slurm_screen_exp11-screen-C8-conf-40000-s45-K1_3657717.out
slurm_screen_exp11-screen-C8-conf-40000-s45-K1_3658831.out
slurm_screen_exp11-screen-C8-conf-40000-s45-K8_3657716.out
slurm_screen_exp11-screen-C8-conf-40000-s45-K8_3658830.out
slurm_screen_exp11-screen-C8-conf-40000-s46-K1_3657719.out
slurm_screen_exp11-screen-C8-conf-40000-s46-K1_3658833.out
slurm_screen_exp11-screen-C8-conf-40000-s46-K8_3657718.out
slurm_screen_exp11-screen-C8-conf-40000-s46-K8_3658832.out
slurm_screen_exp11-screen-C8-cross-40000-s42-K8_3657748.out
slurm_screen_exp11-screen-C8-cross-40000-s42-K8_3657749.out
slurm_screen_exp11-screen-C8-cross-40000-s42-K8_3657750.out
slurm_screen_exp11-screen-C8-cross-40000-s42-K8_3680761.out
slurm_screen_exp11-screen-C8-cross-40000-s42-K8_3684137.out
slurm_screen_exp11-screen-C8-cross-40000-s42-K8_3684138.out
slurm_screen_exp11-screen-C8-r3-40000-s42-K8_3657735.out
slurm_screen_exp11-screen-C8-r3-40000-s42-K8_3657736.out
slurm_screen_exp11-screen-C8-r3-40000-s42-K8_3657737.out
slurm_screen_exp11-screen-C8-r3-40000-s42-K8_3657738.out
slurm_screen_exp11-screen-C8-r3-40000-s42-K8_3657739.out
slurm_screen_exp11-screen-C8-r3-40000-s42-K8_3680743.out
slurm_screen_exp11-screen-C8-r3-40000-s42-K8_3680744.out
slurm_screen_exp11-screen-C8-r3-40000-s42-K8_3680745.out
slurm_screen_exp11-screen-C8-r3-40000-s42-K8_3680746.out
slurm_screen_exp11-screen-C8-r3-40000-s42-K8_3680747.out
slurm_screen_exp11-screen-C8-screen-10000-s42-K1_3662186.out
slurm_screen_exp11-screen-C8-screen-12500-s42-K1_3662188.out
slurm_screen_exp11-screen-C8-screen-15000-s42-K1_3662190.out
slurm_screen_exp11-screen-C8-screen-17500-s42-K1_3662192.out
slurm_screen_exp11-screen-C8-screen-20000-s42-K1_3662194.out
slurm_screen_exp11-screen-C8-screen-22500-s42-K1_3662196.out
slurm_screen_exp11-screen-C8-screen-2500-s42-K1_3662180.out
slurm_screen_exp11-screen-C8-screen-25000-s42-K1_3662198.out
slurm_screen_exp11-screen-C8-screen-27500-s42-K1_3662200.out
slurm_screen_exp11-screen-C8-screen-30000-s42-K1_3662202.out
slurm_screen_exp11-screen-C8-screen-32500-s42-K1_3662204.out
slurm_screen_exp11-screen-C8-screen-35000-s42-K1_3662206.out
slurm_screen_exp11-screen-C8-screen-37500-s42-K1_3662208.out
slurm_screen_exp11-screen-C8-screen-5000-s42-K1_3662182.out
slurm_screen_exp11-screen-C8-screen-7500-s42-K1_3662184.out
slurm_screen_exp11-screen-VANL-q9-40000-s42-K1_3674659.out
slurm_screen_exp11-screen-VANL-q9-40000-s42-K8_3674658.out
slurm_screen_exp11-screen-VANL-q9-40000-s43-K1_3674661.out
slurm_screen_exp11-screen-VANL-q9-40000-s43-K8_3674660.out
slurm_screen_exp11-screen-VANL-q9-40000-s44-K1_3674663.out
slurm_screen_exp11-screen-VANL-q9-40000-s44-K8_3674662.out
slurm_screen_exp11-screen-VANL-q9-40000-s45-K1_3674665.out
slurm_screen_exp11-screen-VANL-q9-40000-s45-K8_3674664.out
slurm_screen_exp11-screen-VANL-q9-40000-s46-K1_3674667.out
slurm_screen_exp11-screen-VANL-q9-40000-s46-K8_3674666.out
slurm_screen_exp11-screen-VANL-screen-10000-s42-K1_3662406.out
slurm_screen_exp11-screen-VANL-screen-10000-s42-K1_3674686.out
slurm_screen_exp11-screen-VANL-screen-10000-s42-K8_3662405.out
slurm_screen_exp11-screen-VANL-screen-10000-s42-K8_3674685.out
slurm_screen_exp11-screen-VANL-screen-12500-s42-K1_3674688.out
slurm_screen_exp11-screen-VANL-screen-12500-s42-K8_3674687.out
slurm_screen_exp11-screen-VANL-screen-15000-s42-K1_3674690.out
slurm_screen_exp11-screen-VANL-screen-15000-s42-K8_3674689.out
slurm_screen_exp11-screen-VANL-screen-17500-s42-K1_3674692.out
slurm_screen_exp11-screen-VANL-screen-17500-s42-K8_3674691.out
slurm_screen_exp11-screen-VANL-screen-20000-s42-K1_3674694.out
slurm_screen_exp11-screen-VANL-screen-20000-s42-K8_3674693.out
slurm_screen_exp11-screen-VANL-screen-22500-s42-K1_3674696.out
slurm_screen_exp11-screen-VANL-screen-22500-s42-K8_3674695.out
slurm_screen_exp11-screen-VANL-screen-2500-s42-K1_3662400.out
slurm_screen_exp11-screen-VANL-screen-2500-s42-K1_3662806.out
slurm_screen_exp11-screen-VANL-screen-2500-s42-K1_3674680.out
slurm_screen_exp11-screen-VANL-screen-2500-s42-K8_3662399.out
slurm_screen_exp11-screen-VANL-screen-2500-s42-K8_3662805.out
slurm_screen_exp11-screen-VANL-screen-2500-s42-K8_3674679.out
slurm_screen_exp11-screen-VANL-screen-25000-s42-K1_3674698.out
slurm_screen_exp11-screen-VANL-screen-25000-s42-K8_3674697.out
slurm_screen_exp11-screen-VANL-screen-27500-s42-K1_3674700.out
slurm_screen_exp11-screen-VANL-screen-27500-s42-K8_3674699.out
slurm_screen_exp11-screen-VANL-screen-30000-s42-K1_3674702.out
slurm_screen_exp11-screen-VANL-screen-30000-s42-K8_3674701.out
slurm_screen_exp11-screen-VANL-screen-32500-s42-K1_3674704.out
slurm_screen_exp11-screen-VANL-screen-32500-s42-K8_3674703.out
slurm_screen_exp11-screen-VANL-screen-35000-s42-K1_3674706.out
slurm_screen_exp11-screen-VANL-screen-35000-s42-K8_3674705.out
slurm_screen_exp11-screen-VANL-screen-37500-s42-K1_3674708.out
slurm_screen_exp11-screen-VANL-screen-37500-s42-K8_3674707.out
slurm_screen_exp11-screen-VANL-screen-40000-s42-K1_3674710.out
slurm_screen_exp11-screen-VANL-screen-40000-s42-K8_3674709.out
slurm_screen_exp11-screen-VANL-screen-5000-s42-K1_3662402.out
slurm_screen_exp11-screen-VANL-screen-5000-s42-K1_3662808.out
slurm_screen_exp11-screen-VANL-screen-5000-s42-K1_3674682.out
slurm_screen_exp11-screen-VANL-screen-5000-s42-K8_3662401.out
slurm_screen_exp11-screen-VANL-screen-5000-s42-K8_3662807.out
slurm_screen_exp11-screen-VANL-screen-5000-s42-K8_3674681.out
slurm_screen_exp11-screen-VANL-screen-7500-s42-K1_3662404.out
slurm_screen_exp11-screen-VANL-screen-7500-s42-K1_3674684.out
slurm_screen_exp11-screen-VANL-screen-7500-s42-K8_3662403.out
slurm_screen_exp11-screen-VANL-screen-7500-s42-K8_3662809.out
slurm_screen_exp11-screen-VANL-screen-7500-s42-K8_3674683.out
slurm_train_exp11-C16-train_3648667.out
slurm_train_exp11-C16-train_3648696.out
slurm_train_exp11-C16-train_3680877.out
slurm_train_exp11-C16-train_3684151.out
slurm_train_exp11-C32-train_3648668.out
slurm_train_exp11-C32-train_3648697.out.frozen_at_0204
slurm_train_exp11-C32-train_3680878.out
slurm_train_exp11-C32-train_3684152.out
slurm_train_exp11-C4L-train_3648665.out
slurm_train_exp11-C4L-train_3648694.out
slurm_train_exp11-C4L-train_3680875.out
slurm_train_exp11-C4L-train_3684149.out
slurm_train_exp11-C8-train_3648666.out
slurm_train_exp11-C8-train_3648695.out
slurm_train_exp11-C8-train_3680876.out
slurm_train_exp11-C8-train_3684150.out
slurm_train_exp11-VANL-train_3680879.out
slurm_train_exp11-VANL-train_3684153.out
slurm_train_exp11-smoke-C4L_3646734.out
slurm_train_exp11-smoke-C4L_3646773.out
slurm_train_exp11-smoke-C4L_3648568.out

exec
/usr/bin/bash -lc "sed -n '1,320p' worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_chunk_kit_review_r2.md" in /n/fs/gatrdp/codespace/FLAC
 succeeded in 0ms:
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

exec
/usr/bin/bash -lc "sed -n '1,360p' /tmp/claude-374349/-n-fs-gatrdp-codespace-FLAC/66305eb9-88c7-41df-95da-dde30a426dc2/scratchpad/chunk_kit_r3.diff" in /n/fs/gatrdp/codespace/FLAC
 succeeded in 0ms:
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
index fc72f83..1d9473c 100644
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
@@ -196,14 +237,178 @@ def check_extension_binding(manifest_path, registry_path, arm, rung, config_path
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
+def check_link_manifest(i, link):
+    """Link ``i`` must agree with the MANIFEST it cites (round-5 r2, blocking 4).
+
+    Continuity alone proves only that the numbers in the registry line up, and
+    the registry is a mutable, uncommitted record (round-5 B1 deliberately took
+    it out of both drift gates). A single structurally valid forged link —
+    resume hash/step copied from the audited anchor, an increasing final step,
+    and an arbitrary final hash — therefore passed every ancestry check. So each
+    link is now checked against the artefact it names:
+
+      * its manifest_path exists and its BYTES hash to the recorded manifest_sha256;
+      * the manifest's job / launch_uuid are the link's;
+      * the manifest's endpoint attestation (step, sha256) is the link's
+        final_step / final_ckpt_sha256;
+      * the manifest's chunk_end is the link's final_step.
+
+    THREAT MODEL, honestly stated: this is COOPERATIVE INTEGRITY, not
+    cryptographic provenance. It defends against accidents (a stale glob, a
+    mis-recorded link, a manifest that drifted after recording) and casual
+    tampering (editing the registry by hand). It cannot defend against an author
+    who edits registry and manifest together: both are uncommitted files under
+    the same user between publication and commit, and nothing here is signed.
+    The audit trail is the git history of the committed records, not this check.
+    """
+    problems = []
+    job, uuid = link.get("job"), link.get("launch_uuid")
+    man_path, man_sha = link.get("manifest_path"), link.get("manifest_sha256")
+    if not man_path:
+        return [f"chain link {i} (job {job!r}) cites no manifest_path — a link with no manifest "
+                "is a bare assertion, not a record"]
+    if not os.path.isfile(man_path):
+        return [f"chain link {i} (job {job!r}) cites manifest {man_path}, which does not exist — "
+                "the record cannot be checked against the artefact it names"]
+    got = sha256_file(man_path)
+    if got != man_sha:
+        problems.append(f"chain link {i} (job {job!r}): its manifest {man_path} now hashes "
+                        f"{got[:12]}, not the recorded {str(man_sha)[:12]} — the manifest changed "
+                        "after the link was recorded")
+    man = parse_manifest(man_path)
+    jkv = kv_line(man, "job")
+    if jkv.get("job") != str(job):
+        problems.append(f"chain link {i}: its manifest records job {jkv.get('job')!r}, not the "
+                        f"link's {job!r}")
+    if jkv.get("launch_uuid") != uuid:
+        problems.append(f"chain link {i} (job {job!r}): its manifest records launch_uuid "
+                        f"{jkv.get('launch_uuid')!r}, not the link's {uuid!r}")
+    ckv = kv_line(man, "chunk_end")
+    if str(ckv.get("chunk_end")) != str(link.get("final_step")):
+        problems.append(f"chain link {i} (job {job!r}): its manifest declares chunk_end "
+                        f"{ckv.get('chunk_end')!r}, not the link's final_step "
+                        f"{link.get('final_step')!r}")
+    akv = kv_line(man, "endpoint_ckpt")
+    if "endpoint_ckpt" not in man:
+        problems.append(f"chain link {i} (job {job!r}): its manifest carries no endpoint "
+                        "attestation, so nothing in it says that job produced this checkpoint")
+        return problems
+    if str(akv.get("endpoint_step")) != str(link.get("final_step")):
+        problems.append(f"chain link {i} (job {job!r}): its manifest attests endpoint_step "
+                        f"{akv.get('endpoint_step')!r} != the link's final_step "
+                        f"{link.get('final_step')!r}")
+    if akv.get("endpoint_sha256") != link.get("final_ckpt_sha256"):
+        problems.append(f"chain link {i} (job {job!r}): its manifest attests endpoint_sha256 "
+                        f"{str(akv.get('endpoint_sha256'))[:12]} != the link's final_ckpt_sha256 "
+                        f"{str(link.get('final_ckpt_sha256'))[:12]} — the record does not match "
+                        "the manifest it cites")
+    return problems
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
+
+    Round-5 r2 (blocking 4): continuity is necessary but not sufficient — every
+    link is ALSO checked against the manifest it cites (check_link_manifest),
+    because a lone structurally valid forged link satisfied continuity by
+    construction.
+    """
+    problems = []
+    prev_sha, prev_step = reg.get("final_ckpt_sha256"), reg.get("final_step")
+    prev_label = "the audited INITIAL anchor"
+    for i, link in enumerate(chain):
+        job = link.get("job")
+        r_sha, r_step = link.get("resume_ckpt_sha256"), link.get("resume_step")
+        f_sha, f_step = link.get("final_ckpt_sha256"), link.get("final_step")
+        problems += check_link_manifest(i, link)
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
+
+    Round-5 review B6: the WHOLE ancestry is validated (check_chain_ancestry),
+    not merely ``chain[-1]`` — otherwise a crafted registry could append a tip
+    with an arbitrary hash and no link back to the audited 40k anchor.
+
+    Round-5 r2 (blocking 3(d)): nothing here assumes a checkpoint FILENAME shape.
+    The resume file arrives as an explicit ``--ckpt`` path and is identified by
+    its sha256 and its directory, so a Lightning-versioned endpoint
+    (``epoch=E-step=N-v1.ckpt``, written when a retry finds the unversioned name
+    taken) is admissible exactly like an unversioned one.
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
+        # B6: the FULL ancestry, not just the tip — a tip alone can be crafted.
+        problems += check_chain_ancestry(reg, chain)
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
 
 
@@ -221,13 +426,25 @@ def main(argv=None):
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
@@ -277,7 +494,18 @@ def main(argv=None):
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

exec
/usr/bin/bash -lc "grep -n -E 'Verdict|VERDICT|Blocking|blocker|B5|submission|attestation|versioned|launder' worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_chunk_kit_review_r2.md | tail -80" in /n/fs/gatrdp/codespace/FLAC
 succeeded in 0ms:
214:6. The lineage can still attribute a stale/foreign compatible checkpoint to the wrong job. A launcher manifest is published before training, and the recorder associates it with whichever exactly-named endpoint is present in the shared directory. It does not require `sacct=COMPLETED`, the classifier completion record, or a job-generated endpoint attestation. Thus a failed job’s manifest plus a pre-existing same-arm/config checkpoint can become a link; rehashing proves current bytes, not which job produced them.
216:   Separately, `--chain` validates only `chain[-1]`, not ancestry from the INITIAL anchor. A crafted registry can insert an arbitrary tip hash without valid link-to-link continuity. The recorder also treats a missing predecessor hash/step as `None`, which falls back to the INITIAL anchor. Require full-chain validation and a post-classification producer attestation binding job, UUID, resume hash, endpoint path/step/hash, and successful completion.
225:- The watchdog guard case deliberately exercises no orchestration path and can pass merely by seeing a live job or empty fixture directory. Add deterministic scheduler fakes covering settlement lag, concurrent watchdogs, failed retry, dirty registry, terminal recording, cancellation, and submission failure.
228:## VERDICT
965:    48	is now recorded only on the producing job's own post-classification attestation
1122:   205	def check_endpoint_attestation(man, chunk_end, endpoint, endpoint_sha, repo_root):
1134:   217	    this function makes that attestation MANDATORY and fail-closed:
1136:   219	      * no attestation at all           -> refuse (name the launcher append);
1144:   227	        return ["the manifest carries no endpoint attestation — a chunk is recorded only on the "
1275:   358	    problems += check_endpoint_attestation(man, chunk_end, endpoint, endpoint_sha, args.repo_root)
2085:   643	# --- O. atomic manifest, duplicated to the save-dir (round-3 B5) --------------
2124:   682	# B5 residual: a failed environment dump must not be silently hashed into the record
2230:   788	# any chunk manifest without this attestation, and refuses one whose attested
2233:   791	# laundered into the chain by a checkpoint it did not write.
2246:   804	    echo "endpoint attestation: this leg did NOT classify as complete (rc=${final_rc}) — attesting <none>"
2249:   807	    echo "endpoint attestation: no single *-step=${CHUNK_END}.ckpt under ${RUNDIR}/checkpoints (found '${ENDPOINT_CKPT:-none}') — attesting <none>"
2254:   812	      echo "endpoint attestation: ${ENDPOINT_CKPT} step ${CHUNK_END} sha256 ${ENDPOINT_SHA}"
2257:   815	      echo "endpoint attestation: sha256sum of ${ENDPOINT_CKPT} FAILED — attesting <none>"
2266:   824	    || echo "WARNING: could not append the endpoint attestation to the save-dir manifest copy"
2269:   827	# --- S. durable final record (round-3 B5 residual) ---------------------------
2279:   837	  [ -n "$CHUNK_END" ] && echo "endpoint attestation: ${ATTEST_LINE:-<not attempted>}"
2342:    27	# bug cannot launder an unrecorded checkpoint into the lineage.
2348:    33	#      and the last submitted job is re-checked BY ID before a new submission.
2450:   135	# Two watchdogs are a double-submission engine: both see "no live job", both
2512:   197	    log "${arm}: !!! FROZEN after ${n} consecutive failures (${reason}). No further submission for this arm until its frozen_${arm} line is removed from ${STATE}."
2519:   204	  log "${arm}: !!! FROZEN immediately (${reason}). No further submission for this arm until its frozen_${arm} line is removed from ${STATE}."
2719:   404	      log "${arm}: squeue -j ${ST[lastjob_$arm]} failed (rc=${brc}) — UNKNOWN, skipping this poll rather than risk a duplicate submission"
2839:    24	# --mem=(12N+12)G, --time=<the arm's pinned limit>. Each submission is recorded
2931:   116	# recorder dirties it at 42500, so the very next submission (and every later
2965:   150	MANIFEST="${EXPDIR}/fa_orbit_submission_${ARM}_${INTENT_ID}.txt"
2966:   151	[ ! -e "$MANIFEST" ] || { echo "submission manifest ${MANIFEST} already exists - abort"; exit 2; }
2969:   154	  echo "# exp_11 arm submission (intent published BEFORE sbatch)"
2996:   181	echo "submission recorded: ${MANIFEST} (job ${JID})"
3385:# B6), one with no attestation at all (a pre-B6 launcher, or a leg that never
3417:# authorship. Without the producing job's post-classification attestation, a
3420:# attestation gate — not the immutability gate — is what refuses them.
3421:expect_cmd "recorder refuses a manifest with NO endpoint attestation" 2 "no endpoint attestation" -- \
3501:# B2(a): a SECOND watchdog is a double-submission engine (both see "no live job",
3543:echo "--- H. the submitter refuses un-pinned submission ---"
3579: # --mem=(12N+12)G, --time=<the arm's pinned limit>. Each submission is recorded
3647:+# recorder dirties it at 42500, so the very next submission (and every later
3843:+# any chunk manifest without this attestation, and refuses one whose attested
3846:+# laundered into the chain by a checkpoint it did not write.
3859:+    echo "endpoint attestation: this leg did NOT classify as complete (rc=${final_rc}) — attesting <none>"
3862:+    echo "endpoint attestation: no single *-step=${CHUNK_END}.ckpt under ${RUNDIR}/checkpoints (found '${ENDPOINT_CKPT:-none}') — attesting <none>"
3867:+      echo "endpoint attestation: ${ENDPOINT_CKPT} step ${CHUNK_END} sha256 ${ENDPOINT_SHA}"
3870:+      echo "endpoint attestation: sha256sum of ${ENDPOINT_CKPT} FAILED — attesting <none>"
3879:+    || echo "WARNING: could not append the endpoint attestation to the save-dir manifest copy"
3882: # --- S. durable final record (round-3 B5 residual) ---------------------------
3889:+  [ -n "$CHUNK_END" ] && echo "endpoint attestation: ${ATTEST_LINE:-<not attempted>}"
4088:     2	"""exp_11 arm-launch outcome classification (round-3 review B5).
4246:--- H. the submitter refuses un-pinned submission ---
4278:FAIL  dry run publishes no submission manifest: want rc=0 + 'DRYRUN sbatch', got rc=2
4284:PASS  a dry run leaves no submission manifest behind
4287:--- L. FIFO and pip-freeze plumbing (NEW-4, B5 residual) ---
4314:152:FAIL  dry run publishes no submission manifest: want rc=0 + 'DRYRUN sbatch', got rc=2
4316:158:PASS  a dry run leaves no submission manifest behind
4329:Two material residuals have emerged: the submission singleton is not shared with manual use of the sanctioned submitter, and the endpoint attestation can still bind the wrong bytes on same-boundary retries because Lightning versions colliding filenames while every kit glob ignores those versioned files. I’m checking the live VANL anchor and the exact ancestry trust boundary before finalizing severity.
4507:1. **B2 remains unresolved for manual/concurrent submission.** The singleton lock is held only by the watchdog ([fa_orbit_chunk_watchdog.sh:140](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_chunk_watchdog.sh:140)); the sanctioned submitter does not acquire it before `sbatch` ([fa_orbit_submit.sh:169](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submit.sh:169)). A manual invocation can still race the watchdog’s `squeue`→submit interval and queue the same boundary twice. The job-side flock only serializes execution; a duplicate can start immediately after the first releases it.
4509:   The advertised by-ID recheck is effectively unreachable: when `lastjob` exists and the name query is empty, settlement either unsets `lastjob` or returns; therefore [the later by-ID branch](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_chunk_watchdog.sh:400) cannot normally observe it. Also, a new `lastjob` is persisted only after all arms finish processing ([line 481](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_chunk_watchdog.sh:481)), leaving a crash window after successful submission. Submission needs a reservation/lock shared by every submitter, including manual use.
4511:2. **The positive endpoint attestation is written before the job’s final classification is known.** A chunk with `final_rc=0` appends a positive endpoint at [fa_orbit_train.sbatch:793](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:793), but final-record tee, preflight-copy, or W&B failure can subsequently change `final_rc` to 7 at [line 863](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:863). The recorder checks only non-`<none>` path/step/hash—not a success class or settled `COMPLETED` state ([fa_orbit_record_restart.py:225](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_record_restart.py:225)). Consequently a Slurm `FAILED` job can carry an admissible positive attestation; with a fresh failure streak, the watchdog bumps once, records it, clears the streak, and proceeds. This violates B6’s successful-completion requirement.
4515:   - job attestation searches only `*-step=42500.ckpt` ([fa_orbit_train.sbatch:796](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:796));
4517:   - the watchdog’s checkpoint parser also ignores versioned names ([fa_orbit_chunk_watchdog.sh:214](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_chunk_watchdog.sh:214)).
4519:   A successful retry therefore positively attests the stale unversioned file, and the recorder’s rehash agrees—binding another job’s bytes to the retry. This defeats both B3 retry recovery and B6 attribution.
4521:4. **Full-chain validation proves continuity, not link authenticity.** Empty chains are correctly refused by the wrapper, and broken/non-increasing ancestry is rejected. But a single structurally valid forged link—resume hash/step equal to the INITIAL anchor, increasing final step, arbitrary final hash—produces no ancestry error. [check_chain_ancestry](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py:244) never verifies each link’s manifest bytes/SHA, endpoint attestation, job identity, UUID uniqueness, or manifest-SHA uniqueness. The latter checks exist only while appending a new link ([fa_orbit_record_restart.py:306](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_record_restart.py:306)).
4523:   Because B1 deliberately makes the registry mutable and ignored by both drift gates, a crafted single link whose final hash matches a compatible checkpoint in the canonical directory is accepted by `--chain`. B6 therefore still has a crafted-registry laundering path.
4531:- **B5:** Deferred as ruled.
4541:- The recorded guard result is indeed 123 pass / 5 dirty-tree failures, but it does not exercise the manual-submit race, final-status promotion after attestation, versioned retry filenames, or a structurally valid forged chain link.
4543:## VERDICT
4550:1. **B2 remains unresolved for manual/concurrent submission.** The singleton lock is held only by the watchdog ([fa_orbit_chunk_watchdog.sh:140](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_chunk_watchdog.sh:140)); the sanctioned submitter does not acquire it before `sbatch` ([fa_orbit_submit.sh:169](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submit.sh:169)). A manual invocation can still race the watchdog’s `squeue`→submit interval and queue the same boundary twice. The job-side flock only serializes execution; a duplicate can start immediately after the first releases it.
4552:   The advertised by-ID recheck is effectively unreachable: when `lastjob` exists and the name query is empty, settlement either unsets `lastjob` or returns; therefore [the later by-ID branch](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_chunk_watchdog.sh:400) cannot normally observe it. Also, a new `lastjob` is persisted only after all arms finish processing ([line 481](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_chunk_watchdog.sh:481)), leaving a crash window after successful submission. Submission needs a reservation/lock shared by every submitter, including manual use.
4554:2. **The positive endpoint attestation is written before the job’s final classification is known.** A chunk with `final_rc=0` appends a positive endpoint at [fa_orbit_train.sbatch:793](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:793), but final-record tee, preflight-copy, or W&B failure can subsequently change `final_rc` to 7 at [line 863](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:863). The recorder checks only non-`<none>` path/step/hash—not a success class or settled `COMPLETED` state ([fa_orbit_record_restart.py:225](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_record_restart.py:225)). Consequently a Slurm `FAILED` job can carry an admissible positive attestation; with a fresh failure streak, the watchdog bumps once, records it, clears the streak, and proceeds. This violates B6’s successful-completion requirement.
4558:   - job attestation searches only `*-step=42500.ckpt` ([fa_orbit_train.sbatch:796](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:796));
4560:   - the watchdog’s checkpoint parser also ignores versioned names ([fa_orbit_chunk_watchdog.sh:214](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_chunk_watchdog.sh:214)).
4562:   A successful retry therefore positively attests the stale unversioned file, and the recorder’s rehash agrees—binding another job’s bytes to the retry. This defeats both B3 retry recovery and B6 attribution.
4564:4. **Full-chain validation proves continuity, not link authenticity.** Empty chains are correctly refused by the wrapper, and broken/non-increasing ancestry is rejected. But a single structurally valid forged link—resume hash/step equal to the INITIAL anchor, increasing final step, arbitrary final hash—produces no ancestry error. [check_chain_ancestry](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py:244) never verifies each link’s manifest bytes/SHA, endpoint attestation, job identity, UUID uniqueness, or manifest-SHA uniqueness. The latter checks exist only while appending a new link ([fa_orbit_record_restart.py:306](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_record_restart.py:306)).
4566:   Because B1 deliberately makes the registry mutable and ignored by both drift gates, a crafted single link whose final hash matches a compatible checkpoint in the canonical directory is accepted by `--chain`. B6 therefore still has a crafted-registry laundering path.
4574:- **B5:** Deferred as ruled.
4584:- The recorded guard result is indeed 123 pass / 5 dirty-tree failures, but it does not exercise the manual-submit race, final-status promotion after attestation, versioned retry filenames, or a structurally valid forged chain link.
4586:## VERDICT

exec
/usr/bin/bash -lc 'tail -220 worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_chunk_kit_review_r2.md' in /n/fs/gatrdp/codespace/FLAC
 succeeded in 0ms:
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

exec
/usr/bin/bash -lc 'git diff --stat && git diff -- worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submit.sh worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_record_restart.py' in /n/fs/gatrdp/codespace/FLAC
 succeeded in 150ms:
 .../arm_launch_registry.json                       |   6 +-
 .../fa_orbit_ckpt_preflight.py                     | 309 +++++++++++++--
 .../fa_orbit_record_restart.py                     | 378 +++++++++++++++++-
 .../exp_11_fa_orbit_claude/fa_orbit_submit.sh      |  99 ++++-
 .../exp_11_fa_orbit_claude/fa_orbit_train.sbatch   | 143 ++++++-
 .../fa_orbit_train_guardtests.sh                   | 434 +++++++++++++++++++++
 6 files changed, 1319 insertions(+), 50 deletions(-)
diff --git a/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py b/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py
index fc72f83..1d9473c 100644
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
@@ -196,14 +237,178 @@ def check_extension_binding(manifest_path, registry_path, arm, rung, config_path
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
+def check_link_manifest(i, link):
+    """Link ``i`` must agree with the MANIFEST it cites (round-5 r2, blocking 4).
+
+    Continuity alone proves only that the numbers in the registry line up, and
+    the registry is a mutable, uncommitted record (round-5 B1 deliberately took
+    it out of both drift gates). A single structurally valid forged link —
+    resume hash/step copied from the audited anchor, an increasing final step,
+    and an arbitrary final hash — therefore passed every ancestry check. So each
+    link is now checked against the artefact it names:
+
+      * its manifest_path exists and its BYTES hash to the recorded manifest_sha256;
+      * the manifest's job / launch_uuid are the link's;
+      * the manifest's endpoint attestation (step, sha256) is the link's
+        final_step / final_ckpt_sha256;
+      * the manifest's chunk_end is the link's final_step.
+
+    THREAT MODEL, honestly stated: this is COOPERATIVE INTEGRITY, not
+    cryptographic provenance. It defends against accidents (a stale glob, a
+    mis-recorded link, a manifest that drifted after recording) and casual
+    tampering (editing the registry by hand). It cannot defend against an author
+    who edits registry and manifest together: both are uncommitted files under
+    the same user between publication and commit, and nothing here is signed.
+    The audit trail is the git history of the committed records, not this check.
+    """
+    problems = []
+    job, uuid = link.get("job"), link.get("launch_uuid")
+    man_path, man_sha = link.get("manifest_path"), link.get("manifest_sha256")
+    if not man_path:
+        return [f"chain link {i} (job {job!r}) cites no manifest_path — a link with no manifest "
+                "is a bare assertion, not a record"]
+    if not os.path.isfile(man_path):
+        return [f"chain link {i} (job {job!r}) cites manifest {man_path}, which does not exist — "
+                "the record cannot be checked against the artefact it names"]
+    got = sha256_file(man_path)
+    if got != man_sha:
+        problems.append(f"chain link {i} (job {job!r}): its manifest {man_path} now hashes "
+                        f"{got[:12]}, not the recorded {str(man_sha)[:12]} — the manifest changed "
+                        "after the link was recorded")
+    man = parse_manifest(man_path)
+    jkv = kv_line(man, "job")
+    if jkv.get("job") != str(job):
+        problems.append(f"chain link {i}: its manifest records job {jkv.get('job')!r}, not the "
+                        f"link's {job!r}")
+    if jkv.get("launch_uuid") != uuid:
+        problems.append(f"chain link {i} (job {job!r}): its manifest records launch_uuid "
+                        f"{jkv.get('launch_uuid')!r}, not the link's {uuid!r}")
+    ckv = kv_line(man, "chunk_end")
+    if str(ckv.get("chunk_end")) != str(link.get("final_step")):
+        problems.append(f"chain link {i} (job {job!r}): its manifest declares chunk_end "
+                        f"{ckv.get('chunk_end')!r}, not the link's final_step "
+                        f"{link.get('final_step')!r}")
+    akv = kv_line(man, "endpoint_ckpt")
+    if "endpoint_ckpt" not in man:
+        problems.append(f"chain link {i} (job {job!r}): its manifest carries no endpoint "
+                        "attestation, so nothing in it says that job produced this checkpoint")
+        return problems
+    if str(akv.get("endpoint_step")) != str(link.get("final_step")):
+        problems.append(f"chain link {i} (job {job!r}): its manifest attests endpoint_step "
+                        f"{akv.get('endpoint_step')!r} != the link's final_step "
+                        f"{link.get('final_step')!r}")
+    if akv.get("endpoint_sha256") != link.get("final_ckpt_sha256"):
+        problems.append(f"chain link {i} (job {job!r}): its manifest attests endpoint_sha256 "
+                        f"{str(akv.get('endpoint_sha256'))[:12]} != the link's final_ckpt_sha256 "
+                        f"{str(link.get('final_ckpt_sha256'))[:12]} — the record does not match "
+                        "the manifest it cites")
+    return problems
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
+
+    Round-5 r2 (blocking 4): continuity is necessary but not sufficient — every
+    link is ALSO checked against the manifest it cites (check_link_manifest),
+    because a lone structurally valid forged link satisfied continuity by
+    construction.
+    """
+    problems = []
+    prev_sha, prev_step = reg.get("final_ckpt_sha256"), reg.get("final_step")
+    prev_label = "the audited INITIAL anchor"
+    for i, link in enumerate(chain):
+        job = link.get("job")
+        r_sha, r_step = link.get("resume_ckpt_sha256"), link.get("resume_step")
+        f_sha, f_step = link.get("final_ckpt_sha256"), link.get("final_step")
+        problems += check_link_manifest(i, link)
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
+
+    Round-5 review B6: the WHOLE ancestry is validated (check_chain_ancestry),
+    not merely ``chain[-1]`` — otherwise a crafted registry could append a tip
+    with an arbitrary hash and no link back to the audited 40k anchor.
+
+    Round-5 r2 (blocking 3(d)): nothing here assumes a checkpoint FILENAME shape.
+    The resume file arrives as an explicit ``--ckpt`` path and is identified by
+    its sha256 and its directory, so a Lightning-versioned endpoint
+    (``epoch=E-step=N-v1.ckpt``, written when a retry finds the unversioned name
+    taken) is admissible exactly like an unversioned one.
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
+        # B6: the FULL ancestry, not just the tip — a tip alone can be crafted.
+        problems += check_chain_ancestry(reg, chain)
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
 
 
@@ -221,13 +426,25 @@ def main(argv=None):
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
@@ -277,7 +494,18 @@ def main(argv=None):
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
@@ -287,12 +515,19 @@ def main(argv=None):
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
@@ -308,10 +543,22 @@ def main(argv=None):
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
index c20019c..e318253 100644
--- a/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_record_restart.py
+++ b/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_record_restart.py
@@ -26,6 +26,44 @@ else in the manifest was checked at all. Now:
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
+
+ROUND-5 REVIEW B6 — ATTRIBUTION. Location + re-hashing prove which bytes are on
+disk, not WHICH JOB WROTE THEM: a failed leg's (pre-published) manifest plus a
+pre-existing checkpoint of the right name was enough to mint a link. So a chunk
+is now recorded only on the producing job's own post-classification attestation
+(`endpoint_ckpt … endpoint_step … endpoint_sha256 …`, appended by
+fa_orbit_train.sbatch to its own manifest), checked against this recorder's
+independent re-hash. And a chunk that HAS a predecessor is never re-parented
+onto the INITIAL anchor when that predecessor is incomplete — it is refused.
+
+ROUND-5 r2 REVIEW. Two further requirements on a chunk link:
+
+  * THE SCHEDULER MUST AGREE (blocking 2). The attestation is the job's own word.
+    `sacct -X -n -P -j <job> -o State` must report exactly COMPLETED; an empty,
+    failing or unavailable sacct refuses. `--skip-sacct` (off by default, never
+    used by the watchdog) exists for documented manual recovery.
+  * THE ATTESTED PATH IS THE RECORD (blocking 3). Lightning's ModelCheckpoint
+    version counter means a retry at an already-written boundary saves
+    `epoch=E-step=N-v1.ckpt`. Globbing `*-step=N.ckpt` would bind the failed
+    attempt's stale bytes to the retry (or find two files and refuse forever), so
+    the recorder follows the attested path — verified to exist, to sit in the
+    canonical directory, and to carry this leg's step in its name — and stores it
+    in the link as `final_ckpt_path` for the next chunk to resume from.
 """
 import argparse
 import fcntl
@@ -33,7 +71,9 @@ import hashlib
 import json
 import os
 import re
+import subprocess
 import sys
+from datetime import datetime, timezone
 
 HERE = os.path.dirname(os.path.abspath(__file__))
 sys.path.insert(0, HERE)
@@ -41,6 +81,45 @@ import fa_orbit_producer_manifest as pm            # noqa: E402
 from fa_orbit_ckpt_preflight import canonical_ckpt_dir    # noqa: E402
 
 PIN_RE = re.compile(r'^(PINNED_[A-Z0-9_]+)=(?:"([^"]*)"|(\S+))')
+# The pinned checkpoint cadence: a chunk may only end ON a saved checkpoint,
+# otherwise the next chunk has nothing to resume.
+CHUNK_STEP = 2500
+# Lightning's ModelCheckpoint version counter: a retry at a boundary whose
+# unversioned name already exists writes `epoch=E-step=N-v1.ckpt` (then -v2...).
+# An attested endpoint may therefore carry EITHER name shape (round-5 r2 B3).
+ENDPOINT_NAME_RE = re.compile(r"-step=(\d+)(?:-v\d+)?\.ckpt$")
+
+
+def sacct_state(job, sacct_bin="sacct"):
+    """(state, problem): what the SCHEDULER says about ``job``.
+
+    Round-5 r2 review, blocking 2. The producing job's own attestation says what
+    it wrote; it cannot say whether Slurm agrees the job finished — a leg killed
+    after it appended its attestation, or one whose node died during epilogue,
+    still leaves a positive attestation on disk. So a chunk link additionally
+    requires ``sacct`` to report exactly COMPLETED for the manifest's job.
+
+    Fail-CLOSED in every ambiguous direction: a missing/unrunnable ``sacct``, a
+    nonzero exit, and an EMPTY answer (the job is unknown to the accounting
+    database, or accounting is lagging) are all refusals, never "probably fine".
+    """
+    argv = [sacct_bin, "-X", "-n", "-P", "-j", str(job), "-o", "State"]
+    try:
+        proc = subprocess.run(argv, capture_output=True, text=True, timeout=120)
+    except (OSError, subprocess.SubprocessError) as exc:
+        return None, (f"could not ask the scheduler about job {job} "
+                      f"({' '.join(argv)}): {type(exc).__name__}: {exc} — a chunk link "
+                      "requires scheduler confirmation that the job COMPLETED")
+    if proc.returncode != 0:
+        return None, (f"`{' '.join(argv)}` exited {proc.returncode} "
+                      f"({proc.stderr.strip()[:200]}) — the scheduler's verdict on job {job} is "
+                      "UNKNOWN, and an unknown verdict is never read as a successful one")
+    lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
+    if not lines:
+        return None, (f"sacct reports nothing for job {job} — the scheduler cannot confirm it "
+                      "COMPLETED (accounting may be lagging, or this job never ran); a chunk "
+                      "link is never recorded on an unconfirmed job")
+    return lines[0], None
 
 
 def read_pins(launcher):
@@ -70,12 +149,19 @@ def kvs(man, key):
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
@@ -104,13 +190,18 @@ def check_identity(arm, man, initial, pins, repo_root):
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
@@ -148,6 +239,265 @@ def resolve(root, path):
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
+def resolve_attested_endpoint(man, chunk_end, ckpt_dir, repo_root):
+    """(endpoint_path, endpoint_sha, problems) for the ATTESTED endpoint file.
+
+    Round-5 r2 review, blocking 3(b). This used to glob
+    ``<canonical>/*-step=<chunk_end>.ckpt`` and require exactly one hit. That is
+    wrong twice over once Lightning's version counter is in play: a failed
+    attempt that saved, followed by a successful retry at the same boundary,
+    leaves BOTH ``epoch=E-step=N.ckpt`` (the stale attempt) and
+    ``epoch=E-step=N-v1.ckpt`` (the retry) — the glob then either binds the
+    predecessor's bytes to the retry, or finds two hits and refuses forever.
+
+    The producing job already told us which file it wrote. So the recorder now
+    follows the ATTESTED PATH and verifies it, rather than re-deriving a path
+    from a name pattern:
+
+      * the attested file must EXIST;
+      * it must sit in the audited launch's canonical checkpoint directory;
+      * its NAME must encode this leg's chunk_end (either name shape);
+      * it is re-hashed here, from disk, by this recorder.
+    """
+    attested = kvs(man, "endpoint_ckpt").get("endpoint_ckpt", "")
+    if not attested or attested == "<none>":
+        return None, None, []           # check_endpoint_attestation reports this
+    cand = resolve(repo_root, attested)
+    if not os.path.isfile(cand):
+        return None, None, [f"the attested endpoint checkpoint {attested} does not exist — the "
+                            "recorder does NOT accept the manifest's claimed hash in its place"]
+    if os.path.realpath(os.path.dirname(cand)) != ckpt_dir:
+        return None, None, [f"the attested endpoint checkpoint {attested} is not in the audited "
+                            f"launch's canonical directory {ckpt_dir}"]
+    m = ENDPOINT_NAME_RE.search(os.path.basename(cand))
+    if not m or (chunk_end is not None and int(m.group(1)) != int(chunk_end)):
+        return None, None, [f"the attested endpoint checkpoint {os.path.basename(cand)} does not "
+                            f"carry this leg's chunk end step {chunk_end} in its name (expected "
+                            f"`*-step={chunk_end}.ckpt` or `*-step={chunk_end}-v<N>.ckpt`)"]
+    return cand, pm.sha256_file(cand), []
+
+
+def check_endpoint_attestation(man, chunk_end, endpoint, endpoint_sha):
+    """The producing job's own statement of what it wrote (round-5 review B6).
+
+    The launcher publishes its manifest BEFORE training, so the manifest alone
+    proves intent, never authorship: a FAILED leg's manifest plus a pre-existing
+    checkpoint of the right name would otherwise become a chain link, and
+    re-hashing proves only which bytes are on disk now. After its exit class is
+    SETTLED the job appends to its own manifest
+
+        endpoint_ckpt <path> endpoint_step <n> endpoint_sha256 <sha>
+
+    (or `endpoint_ckpt <none> ...` for every non-success class), and this
+    function makes that attestation MANDATORY and fail-closed:
+
+      * no attestation at all           -> refuse (name the launcher append);
+      * `<none>`                        -> refuse (the leg says it produced none);
+      * endpoint_step != chunk_end      -> refuse (it attests another boundary);
+      * attested sha != our own re-hash -> refuse (it is not that file).
+
+    The attested PATH is resolved and located by resolve_attested_endpoint; the
+    ``endpoint``/``endpoint_sha`` passed here are that function's findings.
+    """
+    ak = kvs(man, "endpoint_ckpt")
+    if "endpoint_ckpt" not in man:
+        return ["the manifest carries no endpoint attestation — a chunk is recorded only on the "
+                "PRODUCING JOB's own statement of what it wrote. fa_orbit_train.sbatch appends "
+                "`endpoint_ckpt <path> endpoint_step <n> endpoint_sha256 <sha>` to this manifest "
+                "once its exit class is settled; a manifest without that line was either written "
+                "by an older launcher or belongs to a leg that never finished"]
+    attested = ak.get("endpoint_ckpt", "")
+    if not attested or attested == "<none>":
+        return [f"the producing job attested `endpoint_ckpt <none>` (class "
+                f"{ak.get('endpoint_class', '?')}) — it did not reach the success class, or wrote no "
+                "checkpoint at its chunk end; a leg that attests nothing is never a chain link"]
+    problems = []
+    if str(ak.get("endpoint_step")) != str(chunk_end):
+        problems.append(f"the attested endpoint_step {ak.get('endpoint_step')!r} != this leg's "
+                        f"chunk_end {chunk_end!r} — the job attests a different boundary")
+    if endpoint is None:
+        return problems      # the unresolvable endpoint is already a problem upstream
+    if ak.get("endpoint_sha256") != endpoint_sha:
+        problems.append(f"the attested endpoint_sha256 "
+                        f"{str(ak.get('endpoint_sha256'))[:12]} != this recorder's own re-hash of "
+                        f"{os.path.basename(endpoint)} ({str(endpoint_sha)[:12]}) — the file on disk "
+                        "is not the file that job attested writing")
+    return problems
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
+    # --- the SCHEDULER's verdict (round-5 r2 review, blocking 2) -------------
+    # The attestation is the job's own word; this is Slurm's. Both are required:
+    # the attestation says WHICH bytes, sacct says the job actually COMPLETED.
+    # --skip-sacct exists for documented MANUAL recovery only (accounting purged,
+    # a hand-audited leg) and is off by default — it is never used by the watchdog.
+    if job and not args.skip_sacct:
+        state, sacct_problem = sacct_state(job, args.sacct_bin)
+        if sacct_problem:
+            problems.append(sacct_problem)
+        elif state != "COMPLETED":
+            problems.append(f"the scheduler reports job {job} as {state!r}, not COMPLETED — a chunk "
+                            "link records a leg that the SCHEDULER agrees finished successfully; "
+                            "if this is a documented manual recovery, re-run with --skip-sacct and "
+                            "say so in the worklog")
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
+    if len(mine) > 1:
+        raise SystemExit(f"{arm} has {len(mine)} chain links claiming job {job} — the registry is "
+                         "inconsistent; fix it before recording")
+    at = mine[0] if mine else len(chain)
+    # Round-5 review NON-BLOCKING: one leg, one link — by job AND by identity. A
+    # different job id carrying an already-recorded launch uuid or manifest is an
+    # inconsistent registry, not a second chunk.
+    if not mine:
+        for i, link in enumerate(chain):
+            if jk.get("launch_uuid") and link.get("launch_uuid") == jk.get("launch_uuid"):
+                problems.append(f"chain link {i} (job {link.get('job')!r}) already carries launch_uuid "
+                                f"{jk.get('launch_uuid')!r} — a different job with the same launch uuid "
+                                "means the registry is inconsistent; fix it before recording")
+            if link.get("manifest_sha256") == man_sha:
+                problems.append(f"chain link {i} (job {link.get('job')!r}) was recorded from a manifest "
+                                f"with this exact sha256 {man_sha[:12]} — the same manifest cannot be "
+                                "two chunks")
+    prev = chain[at - 1] if at > 0 else None
+    if prev is None:
+        want_anchor = want_step = None                 # the audited INITIAL anchor
+        prev_desc = f"the audited INITIAL anchor at step {initial.get('final_step')}"
+    else:
+        want_anchor, want_step = prev.get("final_ckpt_sha256"), prev.get("final_step")
+        prev_desc = f"chain link job {prev.get('job')} ending at step {prev.get('final_step')}"
+        # Round-5 review B6: a predecessor that records no endpoint is NOT
+        # evidence, and falling back to the INITIAL 40k anchor here would silently
+        # re-parent this chunk onto the anchor — the chain's whole point is that
+        # every link continues the one before it. Refuse; the fallback exists
+        # only for the FIRST link (at == 0).
+        if not want_anchor or want_step is None:
+            problems.append(f"the predecessor chain link (index {at - 1}, job {prev.get('job')!r}) "
+                            "carries no final_ckpt_sha256/final_step — it is not evidence of a "
+                            "checkpoint, and a chunk with a predecessor is NEVER re-parented onto the "
+                            "audited INITIAL anchor; repair or re-record that link first")
+            print("RECORD REFUSED:")
+            for p in problems:
+                print(f"  !! {p}")
+            return 2
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
+    # Located by the job's OWN attestation, not by a name glob (blocking 3(b)):
+    # with Lightning's version counter a same-boundary retry writes `-v1`, so a
+    # glob would either bind the failed attempt's stale bytes or refuse forever.
+    ckpt_dir = canonical_ckpt_dir(initial.get("save_dir", ""), arm, args.repo_root)
+    endpoint, endpoint_sha, endpoint_problems = resolve_attested_endpoint(
+        man, chunk_end, ckpt_dir, args.repo_root)
+    problems += endpoint_problems
+    problems += check_endpoint_attestation(man, chunk_end, endpoint, endpoint_sha)
+    if problems:
+        print("RECORD REFUSED:")
+        for p in problems:
+            print(f"  !! {p}")
+        return 2
+
+    # check_identity already re-hashed the resume file and proved it IS this hash
+    resume_sha = want_anchor if want_anchor is not None else initial.get("final_ckpt_sha256")
+    # final_ckpt_path (blocking 3(b)): the ACTUAL endpoint file, versioned name
+    # and all, so the next chunk resumes the recorded tip by path instead of
+    # re-deriving one from a glob that cannot tell `-v1` from its stale twin.
+    link = {"job": job, "launch_uuid": jk.get("launch_uuid"),
+            "manifest_path": os.path.abspath(args.manifest), "manifest_sha256": man_sha,
+            "resume_step": resume_step, "resume_ckpt_sha256": resume_sha,
+            "final_step": chunk_end, "final_ckpt_sha256": endpoint_sha,
+            "final_ckpt_path": pm.rel_to(args.repo_root, endpoint),
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
@@ -165,6 +515,13 @@ def main(argv=None):
     ap.add_argument("--rehash-all", action="store_true",
                     help="re-hash published checkpoints too (full audit, expensive)")
     ap.add_argument("--dry-run", action="store_true", help="validate and report, publish nothing")
+    ap.add_argument("--skip-sacct", action="store_true",
+                    help="DOCUMENTED MANUAL RECOVERY ONLY: skip the scheduler's COMPLETED "
+                         "confirmation for a chunk link. Off by default; the watchdog never "
+                         "passes it. Use only when accounting cannot answer for a leg you have "
+                         "audited by hand, and record why in the worklog.")
+    ap.add_argument("--sacct-bin", default=os.environ.get("SACCT_BIN", "sacct"),
+                    help="the sacct executable (test hook; changes no decision)")
     args = ap.parse_args(argv)
 
     arm = args.arm
@@ -191,6 +548,11 @@ def record(args, arm, pins):
 
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
index 011f5bc..81a3363 100644
--- a/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submit.sh
+++ b/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submit.sh
@@ -9,12 +9,26 @@
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
+#
+# ANTI-DUPLICATE RESERVATION (round-5 r2 review, blocking 1). Every real
+# submission takes an exclusive flock on .submit_<ARM>.lock and re-checks the
+# queue for a live exp11-<ARM>-train INSIDE that lock before calling sbatch, so a
+# manual invocation and the watchdog cannot both queue the same boundary. See the
+# block below for the ordering argument.
 # ============================================================================
 set -uo pipefail
 cd "$(git -C "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" rev-parse --show-toplevel)" || exit 3
@@ -28,19 +42,35 @@ PLACEHOLDER="TO-PIN-AFTER-P0"
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
@@ -59,11 +89,17 @@ else
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
@@ -73,13 +109,66 @@ case "$RUNG" in 32x2|16x4|8x8) ;; *) echo "rung '${RUNG}' must be 32x2|16x4|8x8
 MB="${RUNG%x*}"; NGPU="${RUNG#*x}"
 [ "$((MB * NGPU))" -eq 64 ] || { echo "rung ${RUNG}: MB*NGPU != 64 - abort"; exit 2; }
 
+# --- SUBMISSION RESERVATION (round-5 r2 review, blocking 1) -------------------
+# The chunk watchdog's singleton lock only prevents a second WATCHDOG. A human
+# running this script by hand could still slip between the watchdog's queue check
+# and its sbatch and queue the same boundary twice, and the job-side run-directory
+# flock cannot help: it is taken long after scheduling, so it serialises execution
+# while still burning a second allocation. The reservation therefore lives HERE,
+# in the ONE sanctioned submitter that every path — watchdog or human — goes
+# through:
+#
+#     flock .submit_<ARM>.lock  ->  squeue -n exp11-<ARM>-train  ->  sbatch
+#
+# The queue check is INSIDE the lock, so no other submitter can observe an empty
+# queue and sbatch between our check and ours. Both directions are fail-CLOSED: a
+# held lock refuses, and a squeue that exits NONZERO refuses too — an unreadable
+# queue is never read as an empty one. The lock is released by the kernel when
+# this process exits, on every path (refusal, sbatch failure, success).
+#
+# It is placed BEFORE the code-drift gate deliberately: a duplicate submission
+# must be refused as a duplicate, not masked by whichever gate happens to fire
+# first, and holding the reservation across the (cheap, local) drift check costs
+# nothing. SMOKE and DRYRUN are excluded by design — a smoke leg carries its own
+# job name and identity, and a dry run submits nothing at all.
+if [ "$SMOKE" != "1" ] && [ "$DRYRUN" != "1" ]; then
+  SUBMIT_LOCK="${EXPDIR}/.submit_${ARM}.lock"
+  WHO="${USER:-$(id -un)}"
+  exec 9>"$SUBMIT_LOCK" || { echo "could not open the ${ARM} submission lock ${SUBMIT_LOCK} - abort"; exit 3; }
+  if ! flock -n 9; then
+    echo "another submission for ${ARM} already holds ${SUBMIT_LOCK} — a leg for this arm is already being submitted - abort"
+    exit 2
+  fi
+  # stderr is captured SEPARATELY on purpose: folded into stdout, a harmless
+  # scheduler warning would read as a live job and refuse a legitimate leg.
+  SQ_ERR="$(mktemp "${TMPDIR:-/tmp}/exp11_squeue.XXXXXX")" || { echo "could not create a temp file for the queue check - abort"; exit 3; }
+  LIVE="$(squeue -h -u "$WHO" -n "$JOBNAME" -o '%i %T' 2>"$SQ_ERR")"; QRC=$?
+  SQ_MSG="$(head -3 "$SQ_ERR" 2>/dev/null | tr '\n' ' ')"; rm -f "$SQ_ERR"
+  if [ "$QRC" -ne 0 ]; then
+    echo "squeue exited ${QRC} (${SQ_MSG:-no message}) — the queue state is UNKNOWN and an unreadable queue is never read as an empty one - abort"
+    exit 2
+  fi
+  if [ -n "$LIVE" ]; then
+    echo "a leg for this arm is already queued/running as ${JOBNAME} (${LIVE//$'\n'/; }) - abort"
+    exit 2
+  fi
+fi
+
 # --- drift gate: a queued job must run reviewed, committed code --------------
 # The drift gate is scoped to CODE surfaces, not the whole exp folder: the four
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
@@ -94,9 +183,10 @@ ARGS=(
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
@@ -120,6 +210,7 @@ TMP="$(mktemp "${MANIFEST}.XXXXXX")" || exit 3
   echo "commit ${SHA}"
   echo "pins rung=${RUNG} maxsteps=$(pin PINNED_MAXSTEPS) ckpt_every=$(pin PINNED_CHECKPOINT_EVERY) min_free_mb=$(pin PINNED_MIN_FREE_MB) p0_manifest_sha256=$(pin PINNED_P0_MANIFEST_SHA256)"
   echo "resume ${RESUME_CKPT:-<none>} expected_step ${EXPECTED_STEP}"
+  echo "chunk_end ${CHUNK_END:-<none>}"
   echo "sbatch sbatch ${ARGS[*]}"
 } >> "$TMP" || { echo "intent manifest write failed - abort"; exit 3; }
 mv -n "$TMP" "$MANIFEST" || { echo "intent manifest publication failed - abort"; exit 2; }
diff --git a/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch b/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch
index 3bfa2e7..2488562 100644
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
@@ -707,8 +767,12 @@ if [ "$WANDB_CHECK_RC" -ne 0 ]; then
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
@@ -723,7 +787,9 @@ FINAL_RECORD="$(
   echo "torchrun rc=${rc} tee rc=${tee_rc} wandb_identity_rc=${WANDB_CHECK_RC} classified rc=${final_rc}"
   echo "$CLASSIFY_OUT"
   echo "last checkpoint: ${LAST_CKPT:-<none>}"
+  [ -n "$CHUNK_END" ] && echo "endpoint attestation: appended BELOW this record, once the exit class is settled (round-5 r2 blocking 2)"
   echo "restart: fa_orbit_submit.sh ${ARM} --resume ${LAST_CKPT:-<path>} --expected-step <step>"
+  [ -n "$CHUNK_END" ] && echo "next chunk: record this leg (fa_orbit_record_restart.py ${ARM} ${MANIFEST}), then fa_orbit_submit.sh ${ARM} --resume <ckpt at ${CHUNK_END}> --expected-step ${CHUNK_END} --chunk-end <${CHUNK_END}+${PINNED_CHECKPOINT_EVERY}>"
 )"
 printf '%s\n' "$FINAL_RECORD" | tee -a "$TRAINLOG" "$SAVEDIR_LOG" > /dev/null
 final_tee_rc="${PIPESTATUS[1]}"
@@ -751,5 +817,70 @@ if [ "$final_tee_rc" -ne 0 ] || [ "$PREFLIGHT_COPY_RC" -ne 0 ] || [ "$WANDB_CHEC
   echo "LOG-PROVENANCE: final record tee rc=${final_tee_rc}, preflight copy rc=${PREFLIGHT_COPY_RC}, wandb identity rc=${WANDB_CHECK_RC}"
   [ "$final_rc" -eq 0 ] && final_rc=7
 fi
+# ^^^ THE EXIT CLASS IS SETTLED HERE. Nothing below may change final_rc. ^^^
+
+# --- R2. ENDPOINT ATTESTATION: the job attests what it produced --------------
+# (round-5 B6, moved here by the round-5 r2 review, blockings 2 and 3.)
+#
+# The manifest is published BEFORE training, so on its own it proves only what
+# this job intended. The recorder would otherwise have to associate it with
+# whichever exactly-named checkpoint happened to sit in the shared directory — a
+# failed job's manifest plus a pre-existing same-arm checkpoint could become a
+# chain link, and re-hashing proves current bytes, not authorship. So the job
+# itself appends what it produced: the endpoint path, its step, and its sha256 as
+# this job sees it on disk. fa_orbit_record_restart.py REFUSES any chunk manifest
+# without this attestation, one attesting `<none>`, and one whose attested sha
+# does not match its own re-hash.
+#
+# BLOCKING 2 — WHERE this runs is load-bearing. It used to run right after the
+# classifier, where final_rc=0 was only PROVISIONAL: the final-record tee, the
+# preflight-copy and the W&B identity check can each still promote the class to 7
+# afterwards, so a job that ended up FAILED could carry a positive attestation.
+# It now runs after the settle point above, so a positive attestation is emitted
+# ONLY for the success class (final_rc == 0); every other class attests <none>.
+#
+# BLOCKING 3 — WHICH FILE is load-bearing too. train.py builds ModelCheckpoint
+# with Lightning's version counter at its default (enable_version_counter=True),
+# so a retry at the same boundary writes `epoch=E-step=N-v1.ckpt` beside the
+# unversioned file a previous failed attempt left behind. Globbing only the
+# unversioned name would attest the STALE bytes — another job's checkpoint. Both
+# name shapes are therefore searched and the NEWEST-mtime match wins: that is the
+# file THIS job just wrote. Its actual path is what is attested and recorded.
+ATTEST_LINE=""
+if [ -n "$CHUNK_END" ]; then
+  ENDPOINT_CKPT=""
+  if [ "$final_rc" -eq 0 ]; then
+    ENDPOINT_CKPT="$(ls -1t "${RUNDIR}/checkpoints"/*-step="${CHUNK_END}".ckpt \
+                              "${RUNDIR}/checkpoints"/*-step="${CHUNK_END}"-v*.ckpt 2>/dev/null | head -1)"
+  fi
+  if [ "$final_rc" -ne 0 ]; then
+    ATTEST_LINE="endpoint_ckpt <none> endpoint_step ${CHUNK_END} endpoint_class ${final_rc}"
+    echo "endpoint attestation: this leg's SETTLED class is ${final_rc}, not the success class — attesting <none>"
+  elif [ -z "$ENDPOINT_CKPT" ] || [ ! -f "$ENDPOINT_CKPT" ]; then
+    ATTEST_LINE="endpoint_ckpt <none> endpoint_step ${CHUNK_END} endpoint_class ${final_rc}"
+    echo "endpoint attestation: no *-step=${CHUNK_END}.ckpt or *-step=${CHUNK_END}-v*.ckpt under ${RUNDIR}/checkpoints — attesting <none>"
+  else
+    ENDPOINT_SHA="$(sha256sum "$ENDPOINT_CKPT" | awk '{print $1}')"
+    if [ -n "$ENDPOINT_SHA" ]; then
+      ATTEST_LINE="endpoint_ckpt ${ENDPOINT_CKPT} endpoint_step ${CHUNK_END} endpoint_sha256 ${ENDPOINT_SHA}"
+      echo "endpoint attestation: ${ENDPOINT_CKPT} step ${CHUNK_END} sha256 ${ENDPOINT_SHA} (newest match; settled class ${final_rc})"
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
+  # ...and into BOTH durable logs, so the transcript carries what the manifest
+  # carries. The classifier's log-equality check already ran; appending the same
+  # bytes to both copies keeps them identical.
+  printf 'endpoint attestation: %s\n' "$ATTEST_LINE" | tee -a "$TRAINLOG" "$SAVEDIR_LOG" > /dev/null \
+    || echo "WARNING: could not append the endpoint attestation to the durable logs"
+fi
 printf '%s\n' "$FINAL_RECORD"
 exit "$final_rc"

exec
/usr/bin/bash -lc 'wc -l worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py && sha256sum worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py' in /n/fs/gatrdp/codespace/FLAC
exec
/usr/bin/bash -lc 'wc -l worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submit.sh && sha256sum worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submit.sh' in /n/fs/gatrdp/codespace/FLAC
exec
/usr/bin/bash -lc 'wc -l worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch && sha256sum worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch' in /n/fs/gatrdp/codespace/FLAC
exec
/usr/bin/bash -lc 'wc -l worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_record_restart.py && sha256sum worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_record_restart.py' in /n/fs/gatrdp/codespace/FLAC
exec
/usr/bin/bash -lc 'wc -l worklog/worklog_yixun/exp_11_fa_orbit_claude/arm_launch_registry.json && sha256sum worklog/worklog_yixun/exp_11_fa_orbit_claude/arm_launch_registry.json' in /n/fs/gatrdp/codespace/FLAC
 succeeded in 0ms:
569 worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py
c31de63d3fde9876d25023b88ccf38b39425c801ccd265e527a25ca2268bc382  worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py

 succeeded in 0ms:
232 worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submit.sh
dee17105600a22968a460d7d18420f04afc8ebddac38f2a96bb39d0de8b96a7a  worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submit.sh

 succeeded in 0ms:
627 worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_record_restart.py
95ca814e0195e0b5603039006936e6ce3db5996d6afaf2f324fa089dff1bc29e  worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_record_restart.py

 succeeded in 0ms:
886 worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch
0da5e2d8355817fe847ab95597d82f01c042c2f09508e9ffcdda19868312fe46  worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch

 succeeded in 0ms:
130 worklog/worklog_yixun/exp_11_fa_orbit_claude/arm_launch_registry.json
4f6bff7097967ff577b84fde90df3fe210bad567b99b4cc84e1fb3c8210610af  worklog/worklog_yixun/exp_11_fa_orbit_claude/arm_launch_registry.json

exec
/usr/bin/bash -lc 'wc -l worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_chunk_watchdog.sh && sha256sum worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_chunk_watchdog.sh' in /n/fs/gatrdp/codespace/FLAC
exec
/usr/bin/bash -lc 'wc -l worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh && sha256sum worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh' in /n/fs/gatrdp/codespace/FLAC
 succeeded in 0ms:
636 worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_chunk_watchdog.sh
bb64902abf8817f4341e04bd192cc75a3e05e18bbc533b9e01853aa46180471a  worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_chunk_watchdog.sh

 succeeded in 0ms:
960 worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh
4c8e777ebaebd8c693f2b3b21e3b85d1a5e5fcfb47e995ddcaaa5c1fccf74b08  worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh

exec
/usr/bin/bash -lc "nl -ba worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_chunk_watchdog.sh | sed -n '361,700p'" in /n/fs/gatrdp/codespace/FLAC
 succeeded in 0ms:
   361	  [ "${#hits[@]}" -eq 1 ] || return 1
   362	  printf '%s\n' "${hits[0]}"
   363	}
   364	
   365	leg_manifest_for() {   # <arm> <chunk_end> -> the ONE launcher manifest for that chunk
   366	  local arm="$1" want="$2" f ce
   367	  local -a hits=()
   368	  for f in "$EXPDIR"/fa_orbit_*_"${arm}"_8x8_jid*_manifest.txt; do
   369	    [ -f "$f" ] || continue
   370	    ce="$(awk '/^chunk_end /{print $2; exit}' "$f" 2>/dev/null)"
   371	    [ "$ce" = "$want" ] && hits+=("$f")
   372	  done
   373	  [ "${#hits[@]}" -eq 1 ] || return 1
   374	  printf '%s\n' "${hits[0]}"
   375	}
   376	
   377	manifest_chunk_end() { # <manifest> -> its chunk_end line
   378	  awk '/^chunk_end /{print $2; exit}' "$1" 2>/dev/null
   379	}
   380	
   381	process_arm() {
   382	  local arm="$1"
   383	  if [ -n "${ST[frozen_$arm]:-}" ]; then
   384	    log "${arm}: FROZEN (${ST[frozen_$arm]}) — skipping; clear it in ${STATE} to resume"
   385	    return 0
   386	  fi
   387	
   388	  # B2(b): a FAILING squeue is UNKNOWN, never "the queue is empty". Reading a
   389	  # scheduler hiccup as "no live job" is how a running leg gets a twin.
   390	  #
   391	  # Round-5 r2 (blocking 1): this query is a FAST-PATH SKIP and a settlement
   392	  # trigger, not the anti-duplicate guard. The guard is inside fa_orbit_submit.sh,
   393	  # which holds a per-arm flock and repeats this query INSIDE it before sbatch —
   394	  # the only place where "no live job" and "submit" are atomic with respect to
   395	  # every other submitter, including a human at a shell.
   396	  local queued qrc
   397	  queued="$(squeue -h -u "$USER" -n "exp11-${arm}-train" -o '%i %T' 2>/dev/null)"; qrc=$?
   398	  if [ "$qrc" -ne 0 ]; then
   399	    log "${arm}: squeue failed (rc=${qrc}) — the queue state is UNKNOWN, skipping this poll (an unknown queue is never read as an empty one)"
   400	    return 0
   401	  fi
   402	
   403	  # B3: settle the leg we last submitted — but only once the SCHEDULER says it
   404	  # is really over. An empty or lagging sacct is not a verdict: the arm is left
   405	  # alone (lastjob_<ARM> retained) so the same boundary cannot be resubmitted
   406	  # while the previous attempt is still settling.
   407	  local last="${ST[lastjob_$arm]:-}" st settled_job=""
   408	  if [ -n "$last" ] && [ -z "$queued" ]; then
   409	    st="$(sacct -X -n -P -j "$last" -o State 2>/dev/null | head -1)"
   410	    st="${st%%$'\n'*}"
   411	    case "$st" in
   412	      COMPLETED)
   413	        clear_fail "$arm" "job ${last} COMPLETED"
   414	        settled_job="$last"; unset "ST[lastjob_$arm]" ;;
   415	      CANCELLED*)
   416	        # A cancellation is a HUMAN acting on this run. Replacing the job the
   417	        # operator just killed is the one thing the watchdog must never do.
   418	        settled_job="$last"; unset "ST[lastjob_$arm]"
   419	        freeze_now "$arm" "job ${last} was cancelled by operator — human intervention signal, not a retryable failure"
   420	        return 0 ;;
   421	      FAILED|TIMEOUT|NODE_FAIL|OUT_OF_MEMORY|BOOT_FAIL|DEADLINE|PREEMPTED)
   422	        settled_job="$last"; unset "ST[lastjob_$arm]"
   423	        bump_fail "$arm" "job ${last} ended ${st}" ;;
   424	      "")
   425	        log "${arm}: sacct reports nothing yet for job ${last} — NOT settled, skipping this poll (lastjob_${arm} retained)"
   426	        return 0 ;;
   427	      *)
   428	        log "${arm}: job ${last} is ${st}, which is not a terminal state — NOT settled, skipping this poll (lastjob_${arm} retained)"
   429	        return 0 ;;
   430	    esac
   431	    [ -n "${ST[frozen_$arm]:-}" ] && return 0
   432	  fi
   433	
   434	  if [ -n "$queued" ]; then
   435	    log "${arm}: live job (${queued//$'\n'/; }) — nothing to do"
   436	    return 0
   437	  fi
   438	
   439	  local ckdir="${OUTPUT_ROOT}/exp11_${arm}/FLAC_exp11_${arm}/exp11_${arm}/checkpoints"
   440	  local S; S="$(newest_ckpt_step "$ckdir")"
   441	  if [ -z "$S" ]; then
   442	    log "${arm}: no epoch=*-step=*.ckpt under ${ckdir} — nothing to resume, skipping"
   443	    return 0
   444	  fi
   445	  # B4: a checkpoint PAST the target is an impossible state for a chunk chain
   446	  # (every leg stops ON a boundary <= TARGET). It is not success — freeze.
   447	  if [ "$S" -gt "$TARGET" ]; then
   448	    freeze_now "$arm" "newest checkpoint is step ${S}, PAST the target ${TARGET} — impossible for a chunk chain; a human must explain this checkpoint before the chain continues"
   449	    return 0
   450	  fi
   451	
   452	  # B4: a chunk that finished but was never recorded must be recorded FIRST —
   453	  # BEFORE the DONE test, or the terminal chunk (e.g. 97500 -> 100000) would be
   454	  # declared done and never recorded. The next chunk's preflight binds to the
   455	  # recorded chain, not to whatever is on disk.
   456	  local tip
   457	  if [ "$S" -gt "$ANCHOR_STEP" ]; then
   458	    tip="$(chain_tip_step "$arm")"
   459	    is_num "$tip" || { log "${arm}: could not read the chain tip from ${REGISTRY}"; bump_fail "$arm" "registry unreadable"; return 0; }
   460	    if [ "$tip" -lt "$S" ]; then
   461	      local man ce
   462	      # B3(c): the producer manifest is identified by the JOB THAT SETTLED, not
   463	      # by its chunk_end — a failed attempt and its successful retry publish two
   464	      # manifests with the SAME chunk_end, and chunk_end matching then refuses
   465	      # forever. chunk_end is kept as a cross-check on the job we picked.
   466	      if [ -n "$settled_job" ]; then
   467	        if ! man="$(leg_manifest_by_job "$arm" "$settled_job")"; then
   468	          log "${arm}: job ${settled_job} published no single manifest in ${EXPDIR} (zero or ambiguous) — cannot record the finished chunk"
   469	          bump_fail "$arm" "no unique manifest for settled job ${settled_job}"
   470	          return 0
   471	        fi
   472	        ce="$(manifest_chunk_end "$man")"
   473	        if [ "$ce" != "$S" ]; then
   474	          log "${arm}: job ${settled_job}'s manifest declares chunk_end ${ce:-<none>}, but the newest checkpoint on disk is step ${S} — refusing to attribute that checkpoint to this job"
   475	          bump_fail "$arm" "manifest chunk_end ${ce:-<none>} != on-disk step ${S}"
   476	          return 0
   477	        fi
   478	      elif ! man="$(leg_manifest_for "$arm" "$S")"; then
   479	        # no settled job id in state (e.g. the watchdog was restarted with a
   480	        # clean state file): fall back to UNIQUE chunk_end matching, and refuse
   481	        # the moment it is ambiguous.
   482	        log "${arm}: no single launcher manifest with chunk_end ${S} in ${EXPDIR} (zero or ambiguous) and no settled job id in the state — cannot record the finished chunk"
   483	        bump_fail "$arm" "manifest for chunk_end ${S} not uniquely identifiable"
   484	        return 0
   485	      fi
   486	      local -a rec=("$RECORDER" "$arm" "$man" --registry "$REGISTRY")
   487	      [ "$DRYRUN" = "1" ] && rec+=(--dry-run)
   488	      log "${arm}: recording finished chunk ${tip} -> ${S} from $(basename "$man")"
   489	      local out rc
   490	      out="$("$PY" "${rec[@]}" 2>&1)"; rc=$?
   491	      if [ "$rc" -ne 0 ]; then
   492	        log "${arm}: RECORDER REFUSED (rc=${rc}): $(printf '%s' "$out" | head -4 | tr '\n' ' ')"
   493	        bump_fail "$arm" "recorder rc=${rc}"
   494	        return 0
   495	      fi
   496	      clear_fail "$arm" "chunk ${tip} -> ${S} recorded"
   497	      if [ "$DRYRUN" = "1" ]; then
   498	        log "${arm}: DRYRUN — the chunk link was validated but not written, so the next chunk is not yet admissible"
   499	      else
   500	        tip="$(chain_tip_step "$arm")"
   501	      fi
   502	    fi
   503	  fi
   504	
   505	  # B4: DONE means BOTH the disk and the RECORD reached the target. A target
   506	  # reached on disk but missing from the chain is an unfinished lineage, not a
   507	  # finished campaign.
   508	  # NON-BLOCKING adoption (round-5 r2): done_<ARM> is qualified BY TARGET. A
   509	  # state file reused with a larger TARGET used to keep its old `done` entry, so
   510	  # the arm was counted as finished and the watchdog exited one leg later.
   511	  if [ "$S" -eq "$TARGET" ]; then
   512	    tip="$(chain_tip_step "$arm")"
   513	    if [ "$tip" = "$TARGET" ]; then
   514	      if [ "${ST[done_$arm]:-}" != "${S}@${TARGET}" ]; then
   515	        ST["done_$arm"]="${S}@${TARGET}"
   516	        log "${arm}: DONE — newest checkpoint is step ${S} == target ${TARGET}, and the recorded chain tip is ${tip}"
   517	      fi
   518	      return 0
   519	    fi
   520	    log "${arm}: newest checkpoint is the target ${TARGET} but the recorded chain tip is ${tip} — NOT declaring DONE until the terminal chunk is recorded"
   521	    [ "$DRYRUN" = "1" ] || bump_fail "$arm" "terminal chunk at ${TARGET} is not recorded"
   522	    return 0
   523	  fi
   524	
   525	  # --- which file the next chunk resumes ------------------------------------
   526	  # B3 (round-5 r2): above the anchor the resume file is the RECORDED tip's
   527	  # final_ckpt_path, never a glob. Lightning's version counter means a boundary
   528	  # can hold both `...-step=N.ckpt` (a failed attempt that saved) and
   529	  # `...-step=N-v1.ckpt` (the successful retry); only the record knows which one
   530	  # the recorded chunk actually produced. The unique-file fallback survives for
   531	  # exactly one case — resuming the audited 40k anchor, which has no chain link.
   532	  local step; step="$(chunk_for "$arm")"
   533	  local next=$(( S + step ))
   534	  [ "$next" -gt "$TARGET" ] && next="$TARGET"
   535	  local resume=""
   536	  if [ "$S" -gt "$ANCHOR_STEP" ]; then
   537	    tip="$(chain_tip_step "$arm")"
   538	    if [ "$tip" != "$S" ]; then
   539	      log "${arm}: the newest checkpoint is step ${S} but the recorded chain tip is ${tip} — refusing to submit from an unrecorded checkpoint"
   540	      bump_fail "$arm" "chain tip ${tip} != on-disk step ${S}"
   541	      return 0
   542	    fi
   543	    resume="$(chain_tip_path "$arm")"
   544	    if [ -z "$resume" ] || [ ! -f "$resume" ]; then
   545	      log "${arm}: the recorded chain tip carries no usable final_ckpt_path (got '${resume:-<none>}') — the record, not the directory listing, names the file a chunk resumes"
   546	      bump_fail "$arm" "recorded tip has no resumable final_ckpt_path"
   547	      return 0
   548	    fi
   549	  else
   550	    local -a ck=()
   551	    mapfile -t ck < <(ckpt_files_at "$ckdir" "$S")
   552	    if [ "${#ck[@]}" -ne 1 ]; then
   553	      log "${arm}: expected exactly one checkpoint file at the anchor step ${S} in ${ckdir}, found ${#ck[@]}"
   554	      bump_fail "$arm" "ambiguous checkpoint at step ${S}"
   555	      return 0
   556	    fi
   557	    resume="${ck[0]}"
   558	  fi
   559	
   560	  # submit the next chunk through the ONLY sanctioned submitter, which owns the
   561	  # anti-duplicate reservation (per-arm flock + in-lock queue re-check).
   562	  log "${arm}: submitting chunk ${S} -> ${next} (chunk size ${step}): bash ${SUBMITTER} ${arm} --resume ${resume} --expected-step ${S} --chunk-end ${next} (DRYRUN=${DRYRUN})"
   563	  local out rc jid
   564	  out="$(DRYRUN="$DRYRUN" bash "$SUBMITTER" "$arm" --resume "$resume" \
   565	          --expected-step "$S" --chunk-end "$next" 2>&1)"; rc=$?
   566	  if [ "$rc" -ne 0 ]; then
   567	    log "${arm}: SUBMIT REFUSED (rc=${rc}): $(printf '%s' "$out" | head -4 | tr '\n' ' ')"
   568	    bump_fail "$arm" "submit rc=${rc}"
   569	    return 0
   570	  fi
   571	  jid="$(printf '%s' "$out" | awk '/^submitted /{print $NF}')"
   572	  if [ -n "$jid" ]; then
   573	    ST["lastjob_$arm"]="$jid"
   574	    # Round-5 r2 (blocking 1(b)): persist NOW, not at end-of-pass. Between a
   575	    # successful sbatch and the end of a poll the watchdog could die (or another
   576	    # arm could abort the pass), and a lost lastjob_<ARM> means the next poll
   577	    # settles nothing and re-submits this boundary.
   578	    save_state || log "${arm}: WARNING — job ${jid} was submitted but the state file could not be written; record it by hand in ${STATE} as lastjob_${arm} ${jid}"
   579	    log "${arm}: submitted chunk ${S} -> ${next} as job ${jid} (lastjob_${arm} persisted)"
   580	  else
   581	    log "${arm}: submitter returned 0 without a job id (DRYRUN=${DRYRUN}) — nothing queued"
   582	  fi
   583	  return 0
   584	}
   585	
   586	log "=== chunk watchdog start: arms ${ARM_LIST[*]} chunk ${CHUNK} (VANL ${CHUNK_VANL}) target ${TARGET} poll ${POLL}s max_consec_fail ${MAX_CONSEC_FAIL} oneshot ${ONESHOT} dryrun ${DRYRUN} pid $$ ==="
   587	
   588	# --- B7: every selected arm must already carry an AUDITED anchor -------------
   589	# VANL's registry row has neither final_ckpt_sha256 nor final_step, so its first
   590	# chunk would be submitted, burn a queue slot, and be refused by the preflight.
   591	# Anchoring is an OPERATOR action (fa_orbit_add_anchor.py audits the 40k
   592	# checkpoint into the lineage) — the watchdog must never do it, so it freezes
   593	# the arm here, before anything is submitted. Round-5 r2: the check now also
   594	# proves step==40000, a well-formed digest, a unique anchor file in the
   595	# registry's canonical directory, and that the file's bytes still hash to the
   596	# audited value (one hash per arm, at startup only).
   597	load_state
   598	for ARM in "${ARM_LIST[@]}"; do
   599	  [ -n "${ST[frozen_$ARM]:-}" ] && continue
   600	  ANCHOR="$(anchor_state "$ARM")"
   601	  case "$ANCHOR" in
   602	    OK)         log "${ARM}: audited anchor verified at step ${ANCHOR_STEP} (registry sha matches the file on disk)" ;;
   603	    MISSING)    freeze_now "$ARM" "no audited anchor — run fa_orbit_add_anchor.py ${ARM} first" ;;
   604	    UNREADABLE) freeze_now "$ARM" "the registry row for ${ARM} could not be read from ${REGISTRY} — no audited anchor could be confirmed" ;;
   605	    *)          freeze_now "$ARM" "the audited anchor did not verify: ${ANCHOR}" ;;
   606	  esac
   607	done
   608	save_state || { echo "the state file ${STATE} could not be written — exiting rather than run with failure counters that do not persist" >&2; exit 3; }
   609	
   610	while :; do
   611	  load_state
   612	  for ARM in "${ARM_LIST[@]}"; do
   613	    process_arm "$ARM"
   614	  done
   615	  # NON-BLOCKING adoption: an unwritable state file means the next poll would
   616	  # re-decide from stale state (re-submitting a boundary, losing a freeze). Stop.
   617	  save_state || { log "FATAL: the state file ${STATE} could not be written — exiting rather than continue with unpersisted state"; exit 3; }
   618	
   619	  REMAINING=0
   620	  for ARM in "${ARM_LIST[@]}"; do
   621	    # A done_<ARM> entry counts only for the TARGET it was recorded against;
   622	    # `<step>@<target>` from an earlier, smaller campaign is ignored (and will be
   623	    # recomputed by process_arm), so raising TARGET restarts the arm properly.
   624	    DONE_AT="${ST[done_$ARM]:-}"
   625	    [ -n "$DONE_AT" ] && [ "${DONE_AT#*@}" = "$TARGET" ] && continue
   626	    [ -n "${ST[frozen_$ARM]:-}" ] && continue
   627	    REMAINING=$((REMAINING + 1))
   628	  done
   629	  if [ "$REMAINING" -eq 0 ]; then
   630	    log "=== every arm is DONE or FROZEN — watchdog exiting ==="
   631	    break
   632	  fi
   633	  [ "$ONESHOT" = "1" ] && { log "=== ONESHOT: one pass complete, exiting ==="; break; }
   634	  sleep "$POLL"
   635	done
   636	exit 0

exec
/usr/bin/bash -lc "nl -ba worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_chunk_watchdog.sh | sed -n '1,360p'" in /n/fs/gatrdp/codespace/FLAC
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
    46	# ROUND-5 r2 REVIEW FIXES.
    47	#   B1 the anti-duplicate reservation MOVED INTO fa_orbit_submit.sh, where it
    48	#      also covers manual invocations: the submitter takes an exclusive flock on
    49	#      .submit_<ARM>.lock, re-checks the queue INSIDE it, and only then sbatches.
    50	#      The submitter is therefore the authority on duplicates; this script's own
    51	#      name-scoped squeue is a FAST-PATH SKIP, nothing more (the unreachable
    52	#      by-ID recheck it used to carry has been removed). lastjob_<ARM> is
    53	#      persisted IMMEDIATELY after a successful submit, not at end-of-pass, so a
    54	#      crash in the same poll cannot lose the job we just queued.
    55	#   B3 versioned checkpoints: Lightning writes `epoch=E-step=N-v1.ckpt` when the
    56	#      unversioned name is taken, so BOTH name shapes are parsed for the newest
    57	#      step, and the next chunk resumes the recorded tip's final_ckpt_path (the
    58	#      unique on-disk file is used only for the 40k anchor, which has no link).
    59	#   B7+ the startup anchor check is DEEP: final_step == 40000 exactly, a 64-hex
    60	#      lowercase sha, the file present in the registry's canonical directory, and
    61	#      the file HASHED ONCE at startup and compared. Five ~700 MB hashes cost
    62	#      about a minute; a mismatch freezes that arm.
    63	#   Per-arm MAX chunk (C4L/C8/C16/C32 2500, VANL 5000): a chunk longer than the
    64	#      arm's PINNED_TIME_LIMIT_CHUNK_<ARM> was sized for cannot finish inside its
    65	#      allocation, so it is rejected with the pin named.
    66	#
    67	# LOCK LIFETIME CAVEAT (round-5 r2 non-blocking). The singleton lock is held on
    68	# fd 8, which CHILD PROCESSES INHERIT — including the `sleep` between polls. If
    69	# the watchdog is killed while a child still runs, the kernel keeps the lock
    70	# until that child also exits, so "released when the watchdog dies" is really
    71	# "released when the watchdog and its current child are both gone" (at most one
    72	# POLL interval). This is safe in the direction that matters — a second watchdog
    73	# still cannot start while the first is alive — but the release can be delayed.
    74	#
    75	# USAGE (KEY=VALUE arguments only; every key whitelisted, no value is ever eval'd)
    76	#   bash fa_orbit_chunk_watchdog.sh                       # the pinned defaults
    77	#   bash fa_orbit_chunk_watchdog.sh ARMS=C4L,C8 POLL=600
    78	#   bash fa_orbit_chunk_watchdog.sh ONESHOT=1 DRYRUN=1    # one pass, submits nothing
    79	#
    80	#   ARMS=C4L,C8,C16,C32,VANL  CHUNK=2500  TARGET=100000  POLL=300
    81	#   MAX_CONSEC_FAIL=2  ONESHOT=0  DRYRUN=0
    82	#   PER-ARM CHUNK: CHUNK_C4L= CHUNK_C8= CHUNK_C16= CHUNK_C32= CHUNK_VANL=5000
    83	#     A leg pays 10-15 min of startup whatever its length, so a 2500-step chunk
    84	#     is 20-28% overhead for the fast arm (VANL) and ~4% for the slow one (C32).
    85	#     VANL therefore DEFAULTS to 5000 (it still fits its 02:30:00 chunk pin);
    86	#     every other arm defaults to the global CHUNK. An explicit CHUNK= sets the
    87	#     default for the other four arms only — to change VANL, pass CHUNK_VANL=.
    88	#     The submitted chunk end is capped at TARGET either way.
    89	#     MAXIMUM per arm (time-pin compatibility, refused above it):
    90	#       C4L 2500  C8 2500  C16 2500  C32 2500  VANL 5000
    91	#   TEST HOOKS (change no decision, only where the script reads/writes):
    92	#   OUTPUT_ROOT=outputs_FLAC  REGISTRY=<expdir>/arm_launch_registry.json
    93	#   STATE=<expdir>/.chunk_watchdog_state  LOG=<expdir>/fa_orbit_chunk_watchdog.log
    94	#
    95	# COST. One `squeue` and one `ls` per arm per poll (plus, once an arm is above
    96	# 40000, one short python read of the registry). Nothing else runs on the login
    97	# node; the training itself is entirely inside Slurm.
    98	# ============================================================================
    99	set -uo pipefail
   100	cd "$(git -C "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" rev-parse --show-toplevel)" || exit 3
   101	
   102	EXPDIR="worklog/worklog_yixun/exp_11_fa_orbit_claude"
   103	SUBMITTER="${EXPDIR}/fa_orbit_submit.sh"
   104	RECORDER="${EXPDIR}/fa_orbit_record_restart.py"
   105	PY=/n/fs/gatrdp/envs/flac/bin/python
   106	ANCHOR_STEP=40000                 # where every arm's INITIAL run ended
   107	
   108	ARMS="C4L,C8,C16,C32,VANL"; CHUNK=2500; TARGET=100000; POLL=300
   109	MAX_CONSEC_FAIL=2; ONESHOT=0; DRYRUN=0
   110	# Per-arm chunk size. Empty = "use the global CHUNK"; VANL is 5000 by default
   111	# (its 2500-step leg would be ~a quarter startup overhead — round-5 NON-BLOCKING).
   112	CHUNK_C4L=""; CHUNK_C8=""; CHUNK_C16=""; CHUNK_C32=""; CHUNK_VANL=5000
   113	OUTPUT_ROOT="outputs_FLAC"
   114	REGISTRY="${EXPDIR}/arm_launch_registry.json"
   115	STATE="${EXPDIR}/.chunk_watchdog_state"
   116	LOG="${EXPDIR}/fa_orbit_chunk_watchdog.log"
   117	# B2: the singleton lock is FIXED to the experiment folder — the resource being
   118	# protected is "the arms of exp_11", not any particular state file, so it must
   119	# not be relocatable by an argument.
   120	LOCKFILE="${EXPDIR}/.chunk_watchdog.lock"
   121	
   122	# --- argument parsing: whitelist the KEY, shape-check the VALUE, never eval ---
   123	reject()   { echo "$1" >&2; exit 2; }
   124	is_num()   { case "${1:-}" in ''|*[!0-9]*) return 1 ;; esac; }
   125	is_armset() {
   126	  local v="$1" a
   127	  [ -n "$v" ] || return 1
   128	  local IFS=,
   129	  for a in $v; do
   130	    case "$a" in C4L|C8|C16|C32|VANL) ;; *) return 1 ;; esac
   131	  done
   132	}
   133	for kv in "$@"; do
   134	  case "$kv" in *=*) ;; *) reject "argument '${kv}' is not KEY=VALUE" ;; esac
   135	  key="${kv%%=*}"; val="${kv#*=}"
   136	  case "$key" in
   137	    ARMS)            is_armset "$val" || reject "ARMS='${val}' is not a comma-separated list of C4L|C8|C16|C32|VANL" ;;
   138	    CHUNK|TARGET|POLL|MAX_CONSEC_FAIL|CHUNK_C4L|CHUNK_C8|CHUNK_C16|CHUNK_C32|CHUNK_VANL)
   139	                     is_num "$val" || reject "${key}='${val}' is not a non-negative integer" ;;
   140	    ONESHOT|DRYRUN)  case "$val" in 0|1) ;; *) reject "${key}='${val}' must be 0 or 1" ;; esac ;;
   141	    OUTPUT_ROOT|REGISTRY|STATE|LOG)
   142	                     case "$val" in
   143	                       ''|*[!A-Za-z0-9/._-]*) reject "${key}='${val}' has unsafe characters" ;;
   144	                     esac ;;
   145	    *)               reject "unknown argument '${kv}' (expected ARMS=/CHUNK=/CHUNK_<ARM>=/TARGET=/POLL=/MAX_CONSEC_FAIL=/ONESHOT=/DRYRUN=/OUTPUT_ROOT=/REGISTRY=/STATE=/LOG=)" ;;
   146	  esac
   147	  printf -v "$key" '%s' "$val"      # name whitelisted above; value never parsed
   148	done
   149	[ "$CHUNK" -gt 0 ] || reject "CHUNK must be positive"
   150	[ "$((CHUNK % 2500))" -eq 0 ] || reject "CHUNK=${CHUNK} is not a multiple of the 2500-step checkpoint cadence"
   151	# every per-arm override gets the SAME shape checks as the global one
   152	for A in C4L C8 C16 C32 VANL; do
   153	  eval "PERARM=\${CHUNK_${A}}"                       # name from a literal whitelist
   154	  [ -n "$PERARM" ] || continue
   155	  [ "$PERARM" -gt 0 ] || reject "CHUNK_${A} must be positive"
   156	  [ "$((PERARM % 2500))" -eq 0 ] || reject "CHUNK_${A}=${PERARM} is not a multiple of the 2500-step checkpoint cadence"
   157	done
   158	[ "$TARGET" -gt "$ANCHOR_STEP" ] || reject "TARGET=${TARGET} must exceed the ${ANCHOR_STEP} anchor"
   159	[ "$((TARGET % 2500))" -eq 0 ] || reject "TARGET=${TARGET} is not a multiple of 2500"
   160	[ "$MAX_CONSEC_FAIL" -ge 1 ] || reject "MAX_CONSEC_FAIL must be at least 1"
   161	[ -f "$SUBMITTER" ] || reject "missing ${SUBMITTER}"
   162	[ -f "$RECORDER" ] || reject "missing ${RECORDER}"
   163	[ -f "$REGISTRY" ] || reject "missing ${REGISTRY}"
   164	
   165	# --- B2(c): the arm list is DE-DUPLICATED ------------------------------------
   166	# ARMS=C8,C8 is two passes over one arm in one poll: the first submits, the
   167	# second sees the job it just queued only if the scheduler is already listing it.
   168	# Built HERE (before the singleton lock) so the chunk-size checks below can be
   169	# expressed per SELECTED arm and still refuse before touching any shared state.
   170	IFS=',' read -r -a ARM_RAW <<< "$ARMS"
   171	ARM_LIST=()
   172	for A in "${ARM_RAW[@]}"; do
   173	  SEEN=0
   174	  for B in ${ARM_LIST[@]+"${ARM_LIST[@]}"}; do [ "$A" = "$B" ] && SEEN=1; done
   175	  [ "$SEEN" -eq 0 ] && ARM_LIST+=("$A")
   176	done
   177	[ "${#ARM_LIST[@]}" -gt 0 ] || reject "ARMS='${ARMS}' selected no arm"
   178	
   179	# --- per-arm MAXIMUM chunk: the wall pin is what makes a chunk feasible -------
   180	# NON-BLOCKING adoption, round-5 r2. A chunk leg is walled by
   181	# PINNED_TIME_LIMIT_CHUNK_<ARM>, and each of those pins was sized for a specific
   182	# chunk length: 2500 steps for the four orbit arms, 5000 for the (much faster)
   183	# vanilla arm. A longer chunk than its pin was sized for cannot reach its
   184	# boundary inside the allocation — the leg is wall-killed, no boundary
   185	# checkpoint is written, and the chain stalls at that step forever. So the table
   186	# below is a compatibility constraint, not a preference, and the refusal names
   187	# the pin the request would have violated.
   188	max_chunk_for() { case "$1" in VANL) printf '5000' ;; *) printf '2500' ;; esac; }
   189	for A in "${ARM_LIST[@]}"; do
   190	  eval "EFF=\${CHUNK_${A}:-}"            # name from the is_armset whitelist
   191	  SRC="CHUNK_${A}"
   192	  [ -n "$EFF" ] || { EFF="$CHUNK"; SRC="CHUNK"; }
   193	  MAXC="$(max_chunk_for "$A")"
   194	  [ "$EFF" -le "$MAXC" ] || reject "${SRC}=${EFF} gives ${A} a ${EFF}-step chunk, above the ${MAXC}-step maximum its PINNED_TIME_LIMIT_CHUNK_${A} wall pin was sized for — a longer chunk cannot reach its boundary inside the allocation the submitter requests, and a wall-killed leg writes no boundary checkpoint"
   195	done
   196	
   197	# --- B2(a): ONE watchdog per experiment folder --------------------------------
   198	# Two watchdogs are a double-submission engine: both see "no live job", both
   199	# submit the same boundary, and the job's own run-directory flock cannot help
   200	# because it is taken long after scheduling. The lock is held on fd 8 for this
   201	# process's whole lifetime and released by the kernel when it exits, so a killed
   202	# watchdog leaves nothing stale behind.
   203	exec 8>"$LOCKFILE" || reject "could not open the watchdog lock ${LOCKFILE}"
   204	flock -n 8 || reject "another chunk watchdog already holds ${LOCKFILE} — refusing to start a second instance (two watchdogs would submit the same chunk twice)"
   205	
   206	log() { printf '%s %s\n' "$(date -Is)" "$1" >> "$LOG"; echo "$1"; }
   207	
   208	chunk_for() {   # <arm> -> that arm's chunk size (per-arm override, else global)
   209	  local v; eval "v=\${CHUNK_$1:-}"         # $1 is whitelisted by is_armset
   210	  [ -n "$v" ] && { printf '%s' "$v"; return 0; }
   211	  printf '%s' "$CHUNK"
   212	}
   213	
   214	# --- state: a plain KV file a human can read and edit ------------------------
   215	# keys: fail_<ARM> <n> | frozen_<ARM> <when>|<reason> | lastjob_<ARM> <jid> |
   216	#       done_<ARM> <step>@<target>.  Removing a frozen_<ARM> line un-freezes that
   217	#       arm AND clears its failure streak (see load_state). done_<ARM> carries the
   218	#       TARGET it was reached against, so reusing a state file with a larger
   219	#       TARGET does not read the old completion as this campaign's.
   220	declare -A ST=()
   221	load_state() {
   222	  ST=()
   223	  [ -f "$STATE" ] || return 0
   224	  local k v a
   225	  while read -r k v; do
   226	    case "$k" in ''|\#*) continue ;; esac
   227	    ST["$k"]="$v"
   228	  done < "$STATE"
   229	  # NON-BLOCKING adoption: a human who deletes a frozen_<ARM> line means "try
   230	  # this arm again", but the fail counter that caused the freeze survived and
   231	  # the very next failure re-froze the arm immediately. Clearing a freeze now
   232	  # clears the streak too. Deliberately NARROWER than "reset whenever frozen is
   233	  # absent": a sub-threshold streak (fail < MAX_CONSEC_FAIL, never frozen) must
   234	  # survive across polls or MAX_CONSEC_FAIL could never be reached at all.
   235	  for a in C4L C8 C16 C32 VANL; do
   236	    if [ -z "${ST[frozen_$a]:-}" ] && [ "${ST[fail_$a]:-0}" -ge "$MAX_CONSEC_FAIL" ] 2>/dev/null; then
   237	      ST["fail_$a"]="0"
   238	      log "${a}: frozen_${a} was cleared by hand — its ${MAX_CONSEC_FAIL}-failure streak is reset too"
   239	    fi
   240	  done
   241	}
   242	save_state() {
   243	  local tmp="${STATE}.tmp.$$" k
   244	  {
   245	    echo "# exp_11 chunk watchdog state (plain KV, rewritten atomically)."
   246	    echo "# Delete a frozen_<ARM> line to let the watchdog submit that arm again;"
   247	    echo "# doing so also resets that arm's fail_<ARM> streak to 0 on the next poll."
   248	    if [ "${#ST[@]}" -gt 0 ]; then
   249	      for k in "${!ST[@]}"; do printf '%s %s\n' "$k" "${ST[$k]}"; done | sort
   250	    fi
   251	  } > "$tmp" || { echo "could not write ${tmp}" >&2; return 1; }
   252	  mv -f "$tmp" "$STATE" || { echo "could not publish ${STATE}" >&2; return 1; }
   253	}
   254	
   255	bump_fail() {   # <arm> <reason>
   256	  local arm="$1" reason="$2" n
   257	  n=$(( ${ST[fail_$arm]:-0} + 1 ))
   258	  ST["fail_$arm"]="$n"
   259	  log "${arm}: FAILURE (${reason}) — ${n}/${MAX_CONSEC_FAIL} consecutive"
   260	  if [ "$n" -ge "$MAX_CONSEC_FAIL" ]; then
   261	    ST["frozen_$arm"]="$(date -Is)|${reason}"
   262	    log "${arm}: !!! FROZEN after ${n} consecutive failures (${reason}). No further submission for this arm until its frozen_${arm} line is removed from ${STATE}."
   263	  fi
   264	}
   265	
   266	freeze_now() {  # <arm> <reason> — a freeze that is NOT a retryable failure
   267	  local arm="$1" reason="$2"
   268	  ST["frozen_$arm"]="$(date -Is)|${reason}"
   269	  log "${arm}: !!! FROZEN immediately (${reason}). No further submission for this arm until its frozen_${arm} line is removed from ${STATE}."
   270	}
   271	
   272	clear_fail() {  # <arm> <why>
   273	  if [ "${ST[fail_$1]:-0}" != "0" ]; then
   274	    log "${1}: failure streak cleared (${2})"
   275	  fi
   276	  ST["fail_$1"]="0"
   277	}
   278	
   279	# B3 (round-5 r2): BOTH Lightning name shapes count. A retry at a boundary whose
   280	# unversioned name already exists saves `epoch=E-step=N-v1.ckpt`, and a parser
   281	# that ignored those would read the chain as stuck one boundary back.
   282	newest_ckpt_step() {   # <ckpt dir> -> the largest step with a checkpoint file
   283	  ls -1 "$1" 2>/dev/null \
   284	    | sed -n -e 's/^epoch=[0-9]\{1,\}-step=\([0-9]\{1,\}\)\.ckpt$/\1/p' \
   285	             -e 's/^epoch=[0-9]\{1,\}-step=\([0-9]\{1,\}\)-v[0-9]\{1,\}\.ckpt$/\1/p' \
   286	    | sort -n | tail -1
   287	}
   288	
   289	ckpt_files_at() {      # <ckpt dir> <step> -> every checkpoint file at that step
   290	  local d="$1" s="$2" f
   291	  for f in "$d"/*-step="${s}".ckpt "$d"/*-step="${s}"-v*.ckpt; do
   292	    [ -f "$f" ] && printf '%s\n' "$f"
   293	  done
   294	  return 0
   295	}
   296	
   297	chain_tip_step() {     # <arm> -> the arm's recorded chain tip (or its anchor, or 0)
   298	  "$PY" - "$REGISTRY" "$1" <<'PY' 2>/dev/null
   299	import json, sys
   300	row = (json.load(open(sys.argv[1])).get("arms") or {}).get(sys.argv[2]) or {}
   301	chain = row.get("chain") or []
   302	print(int((chain[-1].get("final_step") if chain else row.get("final_step")) or 0))
   303	PY
   304	}
   305	
   306	chain_tip_path() {     # <arm> -> the recorded tip's final_ckpt_path ('' if no chain)
   307	  "$PY" - "$REGISTRY" "$1" <<'PY' 2>/dev/null
   308	import json, sys
   309	row = (json.load(open(sys.argv[1])).get("arms") or {}).get(sys.argv[2]) or {}
   310	chain = row.get("chain") or []
   311	print((chain[-1].get("final_ckpt_path") if chain else "") or "")
   312	PY
   313	}
   314	
   315	# B7+ (round-5 r2): the startup anchor check is DEEP, not a presence test. It
   316	# proves the arm's audited anchor is a real, unambiguous 40k checkpoint sitting
   317	# where the recorder and preflight will look for it, and that its bytes still
   318	# hash to the audited value. The canonical directory is derived from the
   319	# REGISTRY's own save_dir (what the recorder and preflight use), which in
   320	# production is exactly OUTPUT_ROOT/exp11_<ARM>/... — the same directory this
   321	# script polls. One ~700 MB hash per arm, once, at startup.
   322	anchor_state() {       # <arm> -> OK | <a one-line reason> | UNREADABLE
   323	  "$PY" - "$REGISTRY" "$1" "$EXPDIR" <<'PY' 2>/dev/null || echo UNREADABLE
   324	import hashlib, json, os, re, sys
   325	reg_path, arm, expdir = sys.argv[1:4]
   326	sys.path.insert(0, expdir)
   327	from fa_orbit_ckpt_preflight import canonical_ckpt_dir      # noqa: E402
   328	row = (json.load(open(reg_path)).get("arms") or {}).get(arm) or {}
   329	sha, step = row.get("final_ckpt_sha256"), row.get("final_step")
   330	if not sha or step is None:
   331	    print("MISSING"); raise SystemExit(0)
   332	if str(step) != "40000":
   333	    print(f"the audited anchor is step {step!r}, not the 40000 every INITIAL run ended at")
   334	    raise SystemExit(0)
   335	if not re.fullmatch(r"[0-9a-f]{64}", str(sha)):
   336	    print(f"the audited final_ckpt_sha256 {str(sha)[:20]!r} is not a 64-char lowercase hex digest")
   337	    raise SystemExit(0)
   338	ckdir = canonical_ckpt_dir(row.get("save_dir", ""), arm, os.getcwd())
   339	hits = [f for f in sorted(os.listdir(ckdir)) if re.search(r"-step=40000(-v\d+)?\.ckpt$", f)] \
   340	    if os.path.isdir(ckdir) else []
   341	if len(hits) != 1:
   342	    print(f"expected exactly one step=40000 checkpoint in {ckdir}, found {len(hits)}"
   343	          + (": " + ", ".join(hits) if hits else ""))
   344	    raise SystemExit(0)
   345	h = hashlib.sha256()
   346	with open(os.path.join(ckdir, hits[0]), "rb") as fh:
   347	    for blk in iter(lambda: fh.read(1 << 22), b""):
   348	        h.update(blk)
   349	got = h.hexdigest()
   350	print("OK" if got == sha
   351	      else f"{hits[0]} hashes {got[:12]}, not the audited anchor {str(sha)[:12]}")
   352	PY
   353	}
   354	
   355	leg_manifest_by_job() {  # <arm> <jobid> -> the manifest THAT job published
   356	  local arm="$1" jid="$2" f
   357	  local -a hits=()
   358	  for f in "$EXPDIR"/fa_orbit_*_"${arm}"_*_jid"${jid}"_manifest.txt; do
   359	    [ -f "$f" ] && hits+=("$f")
   360	  done

exec
/usr/bin/bash -lc "nl -ba worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submit.sh | sed -n '1,280p'" in /n/fs/gatrdp/codespace/FLAC
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
    26	#
    27	# ANTI-DUPLICATE RESERVATION (round-5 r2 review, blocking 1). Every real
    28	# submission takes an exclusive flock on .submit_<ARM>.lock and re-checks the
    29	# queue for a live exp11-<ARM>-train INSIDE that lock before calling sbatch, so a
    30	# manual invocation and the watchdog cannot both queue the same boundary. See the
    31	# block below for the ordering argument.
    32	# ============================================================================
    33	set -uo pipefail
    34	cd "$(git -C "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" rev-parse --show-toplevel)" || exit 3
    35	
    36	EXPDIR="worklog/worklog_yixun/exp_11_fa_orbit_claude"
    37	SBATCH_FILE="${EXPDIR}/fa_orbit_train.sbatch"
    38	DRYRUN="${DRYRUN:-0}"
    39	SMOKE="${SMOKE:-0}"
    40	PLACEHOLDER="TO-PIN-AFTER-P0"
    41	
    42	[ -f "$SBATCH_FILE" ] || { echo "missing ${SBATCH_FILE} - abort"; exit 3; }
    43	
    44	ARM="${1:-}"
    45	[ -n "$ARM" ] || { echo "usage: $0 <C4L|C8|C16|C32|VANL> [--resume <ckpt> --expected-step <n> [--chunk-end <n>]] - abort"; exit 2; }
    46	shift
    47	case "$ARM" in C4L|C8|C16|C32|VANL) ;; *) echo "ARM '${ARM}' must be C4L|C8|C16|C32|VANL - abort"; exit 2;; esac
    48	
    49	RESUME_CKPT=""; EXPECTED_STEP=0; CHUNK_END=""
    50	while [ $# -gt 0 ]; do
    51	  case "$1" in
    52	    --resume) RESUME_CKPT="${2:?--resume needs a path}"; shift 2 ;;
    53	    --expected-step) EXPECTED_STEP="${2:?--expected-step needs a number}"; shift 2 ;;
    54	    --chunk-end) CHUNK_END="${2:?--chunk-end needs a number}"; shift 2 ;;
    55	    *) echo "unknown argument '$1' - abort"; exit 2 ;;
    56	  esac
    57	done
    58	case "$EXPECTED_STEP" in ''|*[!0-9]*) echo "--expected-step must be a non-negative integer - abort"; exit 2;; esac
    59	# --- --chunk-end: shape-checked HERE, and again inside the job ---------------
    60	# A chunk is meaningful only for a leg that resumes: it narrows where THIS job
    61	# stops, never what the campaign may reach (the budget pin stays 100000). It must
    62	# land on a saved checkpoint, or the next chunk has nothing to resume from.
    63	CHUNK_BUDGET="$(awk -F= '/^PINNED_MAXSTEPS=/{split($2,a," "); print a[1]; exit}' "$SBATCH_FILE")"
    64	case "$CHUNK_BUDGET" in ''|*[!0-9]*) echo "could not read PINNED_MAXSTEPS from ${SBATCH_FILE} - abort"; exit 3;; esac
    65	if [ -n "$CHUNK_END" ]; then
    66	  case "$CHUNK_END" in ''|*[!0-9]*) echo "--chunk-end must be a positive integer - abort"; exit 2;; esac
    67	  [ "$SMOKE" != "1" ] || { echo "--chunk-end is a production chunk-chain input and has no meaning under SMOKE=1 - abort"; exit 2; }
    68	  { [ -n "$RESUME_CKPT" ] && [ "$EXPECTED_STEP" -gt 0 ]; } \
    69	    || { echo "--chunk-end is valid only together with --resume/--expected-step - abort"; exit 2; }
    70	  [ "$((CHUNK_END % 2500))" -eq 0 ] || { echo "--chunk-end ${CHUNK_END} is not a multiple of 2500 (the pinned checkpoint cadence) - abort"; exit 2; }
    71	  [ "$CHUNK_END" -gt "$EXPECTED_STEP" ] || { echo "--chunk-end ${CHUNK_END} must exceed --expected-step ${EXPECTED_STEP} - abort"; exit 2; }
    72	  [ "$CHUNK_END" -le "$CHUNK_BUDGET" ] || { echo "--chunk-end ${CHUNK_END} exceeds the pinned budget ${CHUNK_BUDGET} - abort"; exit 2; }
    73	fi
    74	
    75	# --- pins are read FROM the launcher, so submitter and job cannot disagree ----
    76	pin() {  # read one PINNED_* value out of the launcher (quoted or bare)
    77	  awk -v k="$1" '$0 ~ "^"k"=" {
    78	        if (match($0, /"[^"]*"/)) { print substr($0, RSTART + 1, RLENGTH - 2) }
    79	        else { split($0, a, "="); split(a[2], b, " "); print b[1] }
    80	        exit }' "$SBATCH_FILE"
    81	}
    82	if [ "$SMOKE" = "1" ]; then
    83	  RUNG="${SMOKE_RUNG:?SMOKE=1 requires SMOKE_RUNG (32x2|16x4|8x8)}"
    84	  TIME_LIMIT="${SMOKE_TIME:-00:30:00}"
    85	  [ -n "${SMOKE_MIN_FREE_MB:-}" ] || { echo "SMOKE=1 requires SMOKE_MIN_FREE_MB - abort"; exit 2; }
    86	  JOBNAME="exp11-smoke-${ARM}"
    87	else
    88	  RUNG="$(pin PINNED_RUNG)"
    89	  # A RESTART leg is a different budget from the INITIAL one: 60k further steps,
    90	  # not 40k from scratch. Selecting the INITIAL limit for a restart would wall-kill
    91	  # every arm partway through the extension.
    92	  # A CHUNK leg is shorter still: it stops at the next boundary, so it is walled
    93	  # by the arm's CHUNK pin (hours) rather than its whole-extension RESTART pin
    94	  # (days) — the whole point of chunking is an allocation the scheduler backfills.
    95	  if [ -n "$CHUNK_END" ]; then
    96	    TIME_LIMIT="$(pin "PINNED_TIME_LIMIT_CHUNK_${ARM}")"
    97	  elif [ -n "${EXPECTED_STEP:-}" ] && [ "${EXPECTED_STEP:-0}" -gt 0 ]; then
    98	    TIME_LIMIT="$(pin "PINNED_TIME_LIMIT_RESTART_${ARM}")"
    99	  else
   100	    TIME_LIMIT="$(pin "PINNED_TIME_LIMIT_${ARM}")"
   101	  fi
   102	  [ -n "$TIME_LIMIT" ] || { echo "the launcher carries no wall pin for this ${ARM} leg - abort"; exit 2; }
   103	  for V in "$RUNG" "$TIME_LIMIT" "$(pin PINNED_MIN_FREE_MB)" "$(pin PINNED_P0_MANIFEST_SHA256)"; do
   104	    [ "$V" != "$PLACEHOLDER" ] || { echo "the launcher still carries ${PLACEHOLDER} pins: the P0 report has not been pinned yet — no arm may be submitted (use SMOKE=1 for the smoke) - abort"; exit 2; }
   105	  done
   106	  JOBNAME="exp11-${ARM}-train"
   107	fi
   108	case "$RUNG" in 32x2|16x4|8x8) ;; *) echo "rung '${RUNG}' must be 32x2|16x4|8x8 - abort"; exit 2;; esac
   109	MB="${RUNG%x*}"; NGPU="${RUNG#*x}"
   110	[ "$((MB * NGPU))" -eq 64 ] || { echo "rung ${RUNG}: MB*NGPU != 64 - abort"; exit 2; }
   111	
   112	# --- SUBMISSION RESERVATION (round-5 r2 review, blocking 1) -------------------
   113	# The chunk watchdog's singleton lock only prevents a second WATCHDOG. A human
   114	# running this script by hand could still slip between the watchdog's queue check
   115	# and its sbatch and queue the same boundary twice, and the job-side run-directory
   116	# flock cannot help: it is taken long after scheduling, so it serialises execution
   117	# while still burning a second allocation. The reservation therefore lives HERE,
   118	# in the ONE sanctioned submitter that every path — watchdog or human — goes
   119	# through:
   120	#
   121	#     flock .submit_<ARM>.lock  ->  squeue -n exp11-<ARM>-train  ->  sbatch
   122	#
   123	# The queue check is INSIDE the lock, so no other submitter can observe an empty
   124	# queue and sbatch between our check and ours. Both directions are fail-CLOSED: a
   125	# held lock refuses, and a squeue that exits NONZERO refuses too — an unreadable
   126	# queue is never read as an empty one. The lock is released by the kernel when
   127	# this process exits, on every path (refusal, sbatch failure, success).
   128	#
   129	# It is placed BEFORE the code-drift gate deliberately: a duplicate submission
   130	# must be refused as a duplicate, not masked by whichever gate happens to fire
   131	# first, and holding the reservation across the (cheap, local) drift check costs
   132	# nothing. SMOKE and DRYRUN are excluded by design — a smoke leg carries its own
   133	# job name and identity, and a dry run submits nothing at all.
   134	if [ "$SMOKE" != "1" ] && [ "$DRYRUN" != "1" ]; then
   135	  SUBMIT_LOCK="${EXPDIR}/.submit_${ARM}.lock"
   136	  WHO="${USER:-$(id -un)}"
   137	  exec 9>"$SUBMIT_LOCK" || { echo "could not open the ${ARM} submission lock ${SUBMIT_LOCK} - abort"; exit 3; }
   138	  if ! flock -n 9; then
   139	    echo "another submission for ${ARM} already holds ${SUBMIT_LOCK} — a leg for this arm is already being submitted - abort"
   140	    exit 2
   141	  fi
   142	  # stderr is captured SEPARATELY on purpose: folded into stdout, a harmless
   143	  # scheduler warning would read as a live job and refuse a legitimate leg.
   144	  SQ_ERR="$(mktemp "${TMPDIR:-/tmp}/exp11_squeue.XXXXXX")" || { echo "could not create a temp file for the queue check - abort"; exit 3; }
   145	  LIVE="$(squeue -h -u "$WHO" -n "$JOBNAME" -o '%i %T' 2>"$SQ_ERR")"; QRC=$?
   146	  SQ_MSG="$(head -3 "$SQ_ERR" 2>/dev/null | tr '\n' ' ')"; rm -f "$SQ_ERR"
   147	  if [ "$QRC" -ne 0 ]; then
   148	    echo "squeue exited ${QRC} (${SQ_MSG:-no message}) — the queue state is UNKNOWN and an unreadable queue is never read as an empty one - abort"
   149	    exit 2
   150	  fi
   151	  if [ -n "$LIVE" ]; then
   152	    echo "a leg for this arm is already queued/running as ${JOBNAME} (${LIVE//$'\n'/; }) - abort"
   153	    exit 2
   154	  fi
   155	fi
   156	
   157	# --- drift gate: a queued job must run reviewed, committed code --------------
   158	# The drift gate is scoped to CODE surfaces, not the whole exp folder: the four
   159	# arms are running and Slurm appends to their tracked *.out logs continuously, so
   160	# a folder-wide check would abort every screen on a live-log write. Configs,
   161	# drivers and validators are still fully covered.
   162	#
   163	# arm_launch_registry.json is EXCLUDED (round-5 review B1). It is a lineage
   164	# RECORD written by the reviewed recorder as each chunk finishes — a measurement
   165	# *product*, not a measurement *surface* — and it is already outside the
   166	# launcher's commit-binding closure. Gating it deadlocks the chunk chain: the
   167	# recorder dirties it at 42500, so the very next submission (and every later
   168	# one, for every arm) is refused until a human commits. Keep it outside.
   169	DRIFT="$(git status --porcelain --untracked-files=no -- train.py defaults.ini src \
   170	         "$EXPDIR"/*.json "$EXPDIR"/*.py "$EXPDIR"/*.sbatch "$EXPDIR"/*.sh \
   171	         ":(exclude)${EXPDIR}/arm_launch_registry.json" \
   172	         worklog/worklog_yixun/exp_07_fa_scratch_claude/FLAC_AR_BF.json 2>/dev/null)"
   173	[ -z "$DRIFT" ] || { echo "tracked measurement surfaces have uncommitted changes - commit first, abort:"; echo "$DRIFT"; exit 2; }
   174	SHA="$(git rev-parse HEAD)"
   175	
   176	ARGS=(
   177	  --job-name="$JOBNAME"
   178	  --gres="gpu:l40:${NGPU}"
   179	  --cpus-per-task="$((8 + 7 * NGPU))"
   180	  --mem="$(((12 * NGPU + 12)))G"
   181	  --time="$TIME_LIMIT"
   182	  --export="ALL,ARM=${ARM},EXPECT_SHA=${SHA},OUTPUT_ROOT=outputs_FLAC"
   183	)
   184	[ "$SMOKE" = "1" ] && ARGS[5]="${ARGS[5]},SMOKE=1,SMOKE_RUNG=${SMOKE_RUNG},SMOKE_MIN_FREE_MB=${SMOKE_MIN_FREE_MB},SMOKE_MAXSTEPS=${SMOKE_MAXSTEPS:-30},SMOKE_TIME=${TIME_LIMIT}"
   185	[ -n "$RESUME_CKPT" ] && ARGS[5]="${ARGS[5]},RESUME_CKPT=${RESUME_CKPT},EXPECTED_STEP=${EXPECTED_STEP}"
   186	[ -n "$CHUNK_END" ] && ARGS[5]="${ARGS[5]},CHUNK_END=${CHUNK_END}"
   187	ARGS+=("$SBATCH_FILE")
   188	
   189	echo "arm ${ARM} | rung ${RUNG} (${MB}x${NGPU}) | time ${TIME_LIMIT} | commit ${SHA} | smoke ${SMOKE} | chunk_end ${CHUNK_END:-<none>}"
   190	if [ "$DRYRUN" = "1" ]; then
   191	  echo "DRYRUN sbatch ${ARGS[*]}"
   192	  exit 0
   193	fi
   194	
   195	# --- NEW-3: publish the INTENT before submitting -----------------------------
   196	# The provenance record must exist before the job can exist, otherwise a local
   197	# write failure leaves a queued job nobody recorded. The intent manifest carries
   198	# the exact command and pins; the job id is appended afterwards, and if that
   199	# append fails the exact job we just created is cancelled.
   200	INTENT_ID="$(date +%s%N)-$(cut -c1-8 /proc/sys/kernel/random/uuid)"
   201	MANIFEST="${EXPDIR}/fa_orbit_submission_${ARM}_${INTENT_ID}.txt"
   202	[ ! -e "$MANIFEST" ] || { echo "submission manifest ${MANIFEST} already exists - abort"; exit 2; }
   203	TMP="$(mktemp "${MANIFEST}.XXXXXX")" || exit 3
   204	{
   205	  echo "# exp_11 arm submission (intent published BEFORE sbatch)"
   206	  echo "intent_id ${INTENT_ID}"
   207	  echo "submitted_at $(date -Is)"
   208	  echo "arm ${ARM} rung ${RUNG} micro ${MB} ngpu ${NGPU}"
   209	  echo "jobname ${JOBNAME} time ${TIME_LIMIT} smoke ${SMOKE}"
   210	  echo "commit ${SHA}"
   211	  echo "pins rung=${RUNG} maxsteps=$(pin PINNED_MAXSTEPS) ckpt_every=$(pin PINNED_CHECKPOINT_EVERY) min_free_mb=$(pin PINNED_MIN_FREE_MB) p0_manifest_sha256=$(pin PINNED_P0_MANIFEST_SHA256)"
   212	  echo "resume ${RESUME_CKPT:-<none>} expected_step ${EXPECTED_STEP}"
   213	  echo "chunk_end ${CHUNK_END:-<none>}"
   214	  echo "sbatch sbatch ${ARGS[*]}"
   215	} >> "$TMP" || { echo "intent manifest write failed - abort"; exit 3; }
   216	mv -n "$TMP" "$MANIFEST" || { echo "intent manifest publication failed - abort"; exit 2; }
   217	[ -e "$MANIFEST" ] || { echo "intent manifest ${MANIFEST} did not appear - abort"; exit 2; }
   218	echo "intent manifest: ${MANIFEST}"
   219	
   220	OUT="$(sbatch "${ARGS[@]}" 2>&1)"; JID="$(echo "$OUT" | awk '/Submitted batch job/ {print $NF}')"
   221	if [ -z "$JID" ]; then
   222	  echo "SUBMIT FAILED: ${OUT}"
   223	  echo "submit_failed $(date -Is)" >> "$MANIFEST"
   224	  exit 1
   225	fi
   226	echo "submitted ${ARM} -> job ${JID}"
   227	if ! echo "jobid ${JID}" >> "$MANIFEST"; then
   228	  echo "could not append job id ${JID} to ${MANIFEST} — cancelling the job rather than leave it unrecorded"
   229	  scancel "$JID" || echo "scancel ${JID} FAILED — cancel it by hand NOW"
   230	  exit 2
   231	fi
   232	echo "submission recorded: ${MANIFEST} (job ${JID})"

exec
/usr/bin/bash -lc "nl -ba worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_chunk_watchdog.sh | sed -n '60,280p'" in /n/fs/gatrdp/codespace/FLAC
 succeeded in 0ms:
    60	#      lowercase sha, the file present in the registry's canonical directory, and
    61	#      the file HASHED ONCE at startup and compared. Five ~700 MB hashes cost
    62	#      about a minute; a mismatch freezes that arm.
    63	#   Per-arm MAX chunk (C4L/C8/C16/C32 2500, VANL 5000): a chunk longer than the
    64	#      arm's PINNED_TIME_LIMIT_CHUNK_<ARM> was sized for cannot finish inside its
    65	#      allocation, so it is rejected with the pin named.
    66	#
    67	# LOCK LIFETIME CAVEAT (round-5 r2 non-blocking). The singleton lock is held on
    68	# fd 8, which CHILD PROCESSES INHERIT — including the `sleep` between polls. If
    69	# the watchdog is killed while a child still runs, the kernel keeps the lock
    70	# until that child also exits, so "released when the watchdog dies" is really
    71	# "released when the watchdog and its current child are both gone" (at most one
    72	# POLL interval). This is safe in the direction that matters — a second watchdog
    73	# still cannot start while the first is alive — but the release can be delayed.
    74	#
    75	# USAGE (KEY=VALUE arguments only; every key whitelisted, no value is ever eval'd)
    76	#   bash fa_orbit_chunk_watchdog.sh                       # the pinned defaults
    77	#   bash fa_orbit_chunk_watchdog.sh ARMS=C4L,C8 POLL=600
    78	#   bash fa_orbit_chunk_watchdog.sh ONESHOT=1 DRYRUN=1    # one pass, submits nothing
    79	#
    80	#   ARMS=C4L,C8,C16,C32,VANL  CHUNK=2500  TARGET=100000  POLL=300
    81	#   MAX_CONSEC_FAIL=2  ONESHOT=0  DRYRUN=0
    82	#   PER-ARM CHUNK: CHUNK_C4L= CHUNK_C8= CHUNK_C16= CHUNK_C32= CHUNK_VANL=5000
    83	#     A leg pays 10-15 min of startup whatever its length, so a 2500-step chunk
    84	#     is 20-28% overhead for the fast arm (VANL) and ~4% for the slow one (C32).
    85	#     VANL therefore DEFAULTS to 5000 (it still fits its 02:30:00 chunk pin);
    86	#     every other arm defaults to the global CHUNK. An explicit CHUNK= sets the
    87	#     default for the other four arms only — to change VANL, pass CHUNK_VANL=.
    88	#     The submitted chunk end is capped at TARGET either way.
    89	#     MAXIMUM per arm (time-pin compatibility, refused above it):
    90	#       C4L 2500  C8 2500  C16 2500  C32 2500  VANL 5000
    91	#   TEST HOOKS (change no decision, only where the script reads/writes):
    92	#   OUTPUT_ROOT=outputs_FLAC  REGISTRY=<expdir>/arm_launch_registry.json
    93	#   STATE=<expdir>/.chunk_watchdog_state  LOG=<expdir>/fa_orbit_chunk_watchdog.log
    94	#
    95	# COST. One `squeue` and one `ls` per arm per poll (plus, once an arm is above
    96	# 40000, one short python read of the registry). Nothing else runs on the login
    97	# node; the training itself is entirely inside Slurm.
    98	# ============================================================================
    99	set -uo pipefail
   100	cd "$(git -C "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" rev-parse --show-toplevel)" || exit 3
   101	
   102	EXPDIR="worklog/worklog_yixun/exp_11_fa_orbit_claude"
   103	SUBMITTER="${EXPDIR}/fa_orbit_submit.sh"
   104	RECORDER="${EXPDIR}/fa_orbit_record_restart.py"
   105	PY=/n/fs/gatrdp/envs/flac/bin/python
   106	ANCHOR_STEP=40000                 # where every arm's INITIAL run ended
   107	
   108	ARMS="C4L,C8,C16,C32,VANL"; CHUNK=2500; TARGET=100000; POLL=300
   109	MAX_CONSEC_FAIL=2; ONESHOT=0; DRYRUN=0
   110	# Per-arm chunk size. Empty = "use the global CHUNK"; VANL is 5000 by default
   111	# (its 2500-step leg would be ~a quarter startup overhead — round-5 NON-BLOCKING).
   112	CHUNK_C4L=""; CHUNK_C8=""; CHUNK_C16=""; CHUNK_C32=""; CHUNK_VANL=5000
   113	OUTPUT_ROOT="outputs_FLAC"
   114	REGISTRY="${EXPDIR}/arm_launch_registry.json"
   115	STATE="${EXPDIR}/.chunk_watchdog_state"
   116	LOG="${EXPDIR}/fa_orbit_chunk_watchdog.log"
   117	# B2: the singleton lock is FIXED to the experiment folder — the resource being
   118	# protected is "the arms of exp_11", not any particular state file, so it must
   119	# not be relocatable by an argument.
   120	LOCKFILE="${EXPDIR}/.chunk_watchdog.lock"
   121	
   122	# --- argument parsing: whitelist the KEY, shape-check the VALUE, never eval ---
   123	reject()   { echo "$1" >&2; exit 2; }
   124	is_num()   { case "${1:-}" in ''|*[!0-9]*) return 1 ;; esac; }
   125	is_armset() {
   126	  local v="$1" a
   127	  [ -n "$v" ] || return 1
   128	  local IFS=,
   129	  for a in $v; do
   130	    case "$a" in C4L|C8|C16|C32|VANL) ;; *) return 1 ;; esac
   131	  done
   132	}
   133	for kv in "$@"; do
   134	  case "$kv" in *=*) ;; *) reject "argument '${kv}' is not KEY=VALUE" ;; esac
   135	  key="${kv%%=*}"; val="${kv#*=}"
   136	  case "$key" in
   137	    ARMS)            is_armset "$val" || reject "ARMS='${val}' is not a comma-separated list of C4L|C8|C16|C32|VANL" ;;
   138	    CHUNK|TARGET|POLL|MAX_CONSEC_FAIL|CHUNK_C4L|CHUNK_C8|CHUNK_C16|CHUNK_C32|CHUNK_VANL)
   139	                     is_num "$val" || reject "${key}='${val}' is not a non-negative integer" ;;
   140	    ONESHOT|DRYRUN)  case "$val" in 0|1) ;; *) reject "${key}='${val}' must be 0 or 1" ;; esac ;;
   141	    OUTPUT_ROOT|REGISTRY|STATE|LOG)
   142	                     case "$val" in
   143	                       ''|*[!A-Za-z0-9/._-]*) reject "${key}='${val}' has unsafe characters" ;;
   144	                     esac ;;
   145	    *)               reject "unknown argument '${kv}' (expected ARMS=/CHUNK=/CHUNK_<ARM>=/TARGET=/POLL=/MAX_CONSEC_FAIL=/ONESHOT=/DRYRUN=/OUTPUT_ROOT=/REGISTRY=/STATE=/LOG=)" ;;
   146	  esac
   147	  printf -v "$key" '%s' "$val"      # name whitelisted above; value never parsed
   148	done
   149	[ "$CHUNK" -gt 0 ] || reject "CHUNK must be positive"
   150	[ "$((CHUNK % 2500))" -eq 0 ] || reject "CHUNK=${CHUNK} is not a multiple of the 2500-step checkpoint cadence"
   151	# every per-arm override gets the SAME shape checks as the global one
   152	for A in C4L C8 C16 C32 VANL; do
   153	  eval "PERARM=\${CHUNK_${A}}"                       # name from a literal whitelist
   154	  [ -n "$PERARM" ] || continue
   155	  [ "$PERARM" -gt 0 ] || reject "CHUNK_${A} must be positive"
   156	  [ "$((PERARM % 2500))" -eq 0 ] || reject "CHUNK_${A}=${PERARM} is not a multiple of the 2500-step checkpoint cadence"
   157	done
   158	[ "$TARGET" -gt "$ANCHOR_STEP" ] || reject "TARGET=${TARGET} must exceed the ${ANCHOR_STEP} anchor"
   159	[ "$((TARGET % 2500))" -eq 0 ] || reject "TARGET=${TARGET} is not a multiple of 2500"
   160	[ "$MAX_CONSEC_FAIL" -ge 1 ] || reject "MAX_CONSEC_FAIL must be at least 1"
   161	[ -f "$SUBMITTER" ] || reject "missing ${SUBMITTER}"
   162	[ -f "$RECORDER" ] || reject "missing ${RECORDER}"
   163	[ -f "$REGISTRY" ] || reject "missing ${REGISTRY}"
   164	
   165	# --- B2(c): the arm list is DE-DUPLICATED ------------------------------------
   166	# ARMS=C8,C8 is two passes over one arm in one poll: the first submits, the
   167	# second sees the job it just queued only if the scheduler is already listing it.
   168	# Built HERE (before the singleton lock) so the chunk-size checks below can be
   169	# expressed per SELECTED arm and still refuse before touching any shared state.
   170	IFS=',' read -r -a ARM_RAW <<< "$ARMS"
   171	ARM_LIST=()
   172	for A in "${ARM_RAW[@]}"; do
   173	  SEEN=0
   174	  for B in ${ARM_LIST[@]+"${ARM_LIST[@]}"}; do [ "$A" = "$B" ] && SEEN=1; done
   175	  [ "$SEEN" -eq 0 ] && ARM_LIST+=("$A")
   176	done
   177	[ "${#ARM_LIST[@]}" -gt 0 ] || reject "ARMS='${ARMS}' selected no arm"
   178	
   179	# --- per-arm MAXIMUM chunk: the wall pin is what makes a chunk feasible -------
   180	# NON-BLOCKING adoption, round-5 r2. A chunk leg is walled by
   181	# PINNED_TIME_LIMIT_CHUNK_<ARM>, and each of those pins was sized for a specific
   182	# chunk length: 2500 steps for the four orbit arms, 5000 for the (much faster)
   183	# vanilla arm. A longer chunk than its pin was sized for cannot reach its
   184	# boundary inside the allocation — the leg is wall-killed, no boundary
   185	# checkpoint is written, and the chain stalls at that step forever. So the table
   186	# below is a compatibility constraint, not a preference, and the refusal names
   187	# the pin the request would have violated.
   188	max_chunk_for() { case "$1" in VANL) printf '5000' ;; *) printf '2500' ;; esac; }
   189	for A in "${ARM_LIST[@]}"; do
   190	  eval "EFF=\${CHUNK_${A}:-}"            # name from the is_armset whitelist
   191	  SRC="CHUNK_${A}"
   192	  [ -n "$EFF" ] || { EFF="$CHUNK"; SRC="CHUNK"; }
   193	  MAXC="$(max_chunk_for "$A")"
   194	  [ "$EFF" -le "$MAXC" ] || reject "${SRC}=${EFF} gives ${A} a ${EFF}-step chunk, above the ${MAXC}-step maximum its PINNED_TIME_LIMIT_CHUNK_${A} wall pin was sized for — a longer chunk cannot reach its boundary inside the allocation the submitter requests, and a wall-killed leg writes no boundary checkpoint"
   195	done
   196	
   197	# --- B2(a): ONE watchdog per experiment folder --------------------------------
   198	# Two watchdogs are a double-submission engine: both see "no live job", both
   199	# submit the same boundary, and the job's own run-directory flock cannot help
   200	# because it is taken long after scheduling. The lock is held on fd 8 for this
   201	# process's whole lifetime and released by the kernel when it exits, so a killed
   202	# watchdog leaves nothing stale behind.
   203	exec 8>"$LOCKFILE" || reject "could not open the watchdog lock ${LOCKFILE}"
   204	flock -n 8 || reject "another chunk watchdog already holds ${LOCKFILE} — refusing to start a second instance (two watchdogs would submit the same chunk twice)"
   205	
   206	log() { printf '%s %s\n' "$(date -Is)" "$1" >> "$LOG"; echo "$1"; }
   207	
   208	chunk_for() {   # <arm> -> that arm's chunk size (per-arm override, else global)
   209	  local v; eval "v=\${CHUNK_$1:-}"         # $1 is whitelisted by is_armset
   210	  [ -n "$v" ] && { printf '%s' "$v"; return 0; }
   211	  printf '%s' "$CHUNK"
   212	}
   213	
   214	# --- state: a plain KV file a human can read and edit ------------------------
   215	# keys: fail_<ARM> <n> | frozen_<ARM> <when>|<reason> | lastjob_<ARM> <jid> |
   216	#       done_<ARM> <step>@<target>.  Removing a frozen_<ARM> line un-freezes that
   217	#       arm AND clears its failure streak (see load_state). done_<ARM> carries the
   218	#       TARGET it was reached against, so reusing a state file with a larger
   219	#       TARGET does not read the old completion as this campaign's.
   220	declare -A ST=()
   221	load_state() {
   222	  ST=()
   223	  [ -f "$STATE" ] || return 0
   224	  local k v a
   225	  while read -r k v; do
   226	    case "$k" in ''|\#*) continue ;; esac
   227	    ST["$k"]="$v"
   228	  done < "$STATE"
   229	  # NON-BLOCKING adoption: a human who deletes a frozen_<ARM> line means "try
   230	  # this arm again", but the fail counter that caused the freeze survived and
   231	  # the very next failure re-froze the arm immediately. Clearing a freeze now
   232	  # clears the streak too. Deliberately NARROWER than "reset whenever frozen is
   233	  # absent": a sub-threshold streak (fail < MAX_CONSEC_FAIL, never frozen) must
   234	  # survive across polls or MAX_CONSEC_FAIL could never be reached at all.
   235	  for a in C4L C8 C16 C32 VANL; do
   236	    if [ -z "${ST[frozen_$a]:-}" ] && [ "${ST[fail_$a]:-0}" -ge "$MAX_CONSEC_FAIL" ] 2>/dev/null; then
   237	      ST["fail_$a"]="0"
   238	      log "${a}: frozen_${a} was cleared by hand — its ${MAX_CONSEC_FAIL}-failure streak is reset too"
   239	    fi
   240	  done
   241	}
   242	save_state() {
   243	  local tmp="${STATE}.tmp.$$" k
   244	  {
   245	    echo "# exp_11 chunk watchdog state (plain KV, rewritten atomically)."
   246	    echo "# Delete a frozen_<ARM> line to let the watchdog submit that arm again;"
   247	    echo "# doing so also resets that arm's fail_<ARM> streak to 0 on the next poll."
   248	    if [ "${#ST[@]}" -gt 0 ]; then
   249	      for k in "${!ST[@]}"; do printf '%s %s\n' "$k" "${ST[$k]}"; done | sort
   250	    fi
   251	  } > "$tmp" || { echo "could not write ${tmp}" >&2; return 1; }
   252	  mv -f "$tmp" "$STATE" || { echo "could not publish ${STATE}" >&2; return 1; }
   253	}
   254	
   255	bump_fail() {   # <arm> <reason>
   256	  local arm="$1" reason="$2" n
   257	  n=$(( ${ST[fail_$arm]:-0} + 1 ))
   258	  ST["fail_$arm"]="$n"
   259	  log "${arm}: FAILURE (${reason}) — ${n}/${MAX_CONSEC_FAIL} consecutive"
   260	  if [ "$n" -ge "$MAX_CONSEC_FAIL" ]; then
   261	    ST["frozen_$arm"]="$(date -Is)|${reason}"
   262	    log "${arm}: !!! FROZEN after ${n} consecutive failures (${reason}). No further submission for this arm until its frozen_${arm} line is removed from ${STATE}."
   263	  fi
   264	}
   265	
   266	freeze_now() {  # <arm> <reason> — a freeze that is NOT a retryable failure
   267	  local arm="$1" reason="$2"
   268	  ST["frozen_$arm"]="$(date -Is)|${reason}"
   269	  log "${arm}: !!! FROZEN immediately (${reason}). No further submission for this arm until its frozen_${arm} line is removed from ${STATE}."
   270	}
   271	
   272	clear_fail() {  # <arm> <why>
   273	  if [ "${ST[fail_$1]:-0}" != "0" ]; then
   274	    log "${1}: failure streak cleared (${2})"
   275	  fi
   276	  ST["fail_$1"]="0"
   277	}
   278	
   279	# B3 (round-5 r2): BOTH Lightning name shapes count. A retry at a boundary whose
   280	# unversioned name already exists saves `epoch=E-step=N-v1.ckpt`, and a parser

exec
/usr/bin/bash -lc "nl -ba worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch | sed -n '441,920p'" in /n/fs/gatrdp/codespace/FLAC
 succeeded in 0ms:
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
   780	# --- S. durable final record (round-3 B5 residual) ---------------------------
   781	# The classifier verdict and this summary must survive in BOTH copies; the tee
   782	# status is captured (it used to be discarded) and a failure is class 7. The
   783	# whole preflight transcript (this job's Slurm output so far) is copied into the
   784	# save-dir too, so the durable record is not just torchrun's stdout.
   785	FINAL_RECORD="$(
   786	  echo "=== exp_11 ${ARM} ${RUNG} finished at $(date '+%Y-%m-%d %H:%M:%S') ==="
   787	  echo "torchrun rc=${rc} tee rc=${tee_rc} wandb_identity_rc=${WANDB_CHECK_RC} classified rc=${final_rc}"
   788	  echo "$CLASSIFY_OUT"
   789	  echo "last checkpoint: ${LAST_CKPT:-<none>}"
   790	  [ -n "$CHUNK_END" ] && echo "endpoint attestation: appended BELOW this record, once the exit class is settled (round-5 r2 blocking 2)"
   791	  echo "restart: fa_orbit_submit.sh ${ARM} --resume ${LAST_CKPT:-<path>} --expected-step <step>"
   792	  [ -n "$CHUNK_END" ] && echo "next chunk: record this leg (fa_orbit_record_restart.py ${ARM} ${MANIFEST}), then fa_orbit_submit.sh ${ARM} --resume <ckpt at ${CHUNK_END}> --expected-step ${CHUNK_END} --chunk-end <${CHUNK_END}+${PINNED_CHECKPOINT_EVERY}>"
   793	)"
   794	printf '%s\n' "$FINAL_RECORD" | tee -a "$TRAINLOG" "$SAVEDIR_LOG" > /dev/null
   795	final_tee_rc="${PIPESTATUS[1]}"
   796	# The transcript path is NOT guessed from the #SBATCH pattern (job 3646734 logged
   797	# "slurm_train_job_3646734.out not found" because SLURM_JOB_NAME is unset inside
   798	# the job): ask Slurm itself.
   799	# --- BEGIN slurm-stdout-helper (guard-tested by fa_orbit_equivprobe_wrapper_test.sh) --
   800	parse_slurm_stdout() {   # `scontrol show job` text on stdin -> the StdOut path
   801	  tr ' ' '\n' | awk -F= '$1=="StdOut"{print $2; exit}'
   802	}
   803	slurm_stdout_path() {    # $1 = job id
   804	  scontrol show job "$1" 2>/dev/null | parse_slurm_stdout
   805	}
   806	# --- END slurm-stdout-helper --------------------------------------------------
   807	SLURM_OUT="$(slurm_stdout_path "$SLURM_JOB_ID")"
   808	PREFLIGHT_COPY_RC=0
   809	if [ -n "$SLURM_OUT" ] && [ -f "$SLURM_OUT" ]; then
   810	  cp "$SLURM_OUT" "${SAVEDIR}/preflight_${TS}_jid${SLURM_JOB_ID}.out" || PREFLIGHT_COPY_RC=$?
   811	  [ "$PREFLIGHT_COPY_RC" -eq 0 ] && echo "preflight transcript duplicated from ${SLURM_OUT}"
   812	else
   813	  echo "PREFLIGHT TRANSCRIPT: scontrol reported StdOut='${SLURM_OUT}' which is not a readable file"
   814	  PREFLIGHT_COPY_RC=1
   815	fi
   816	if [ "$final_tee_rc" -ne 0 ] || [ "$PREFLIGHT_COPY_RC" -ne 0 ] || [ "$WANDB_CHECK_RC" -ne 0 ]; then
   817	  echo "LOG-PROVENANCE: final record tee rc=${final_tee_rc}, preflight copy rc=${PREFLIGHT_COPY_RC}, wandb identity rc=${WANDB_CHECK_RC}"
   818	  [ "$final_rc" -eq 0 ] && final_rc=7
   819	fi
   820	# ^^^ THE EXIT CLASS IS SETTLED HERE. Nothing below may change final_rc. ^^^
   821	
   822	# --- R2. ENDPOINT ATTESTATION: the job attests what it produced --------------
   823	# (round-5 B6, moved here by the round-5 r2 review, blockings 2 and 3.)
   824	#
   825	# The manifest is published BEFORE training, so on its own it proves only what
   826	# this job intended. The recorder would otherwise have to associate it with
   827	# whichever exactly-named checkpoint happened to sit in the shared directory — a
   828	# failed job's manifest plus a pre-existing same-arm checkpoint could become a
   829	# chain link, and re-hashing proves current bytes, not authorship. So the job
   830	# itself appends what it produced: the endpoint path, its step, and its sha256 as
   831	# this job sees it on disk. fa_orbit_record_restart.py REFUSES any chunk manifest
   832	# without this attestation, one attesting `<none>`, and one whose attested sha
   833	# does not match its own re-hash.
   834	#
   835	# BLOCKING 2 — WHERE this runs is load-bearing. It used to run right after the
   836	# classifier, where final_rc=0 was only PROVISIONAL: the final-record tee, the
   837	# preflight-copy and the W&B identity check can each still promote the class to 7
   838	# afterwards, so a job that ended up FAILED could carry a positive attestation.
   839	# It now runs after the settle point above, so a positive attestation is emitted
   840	# ONLY for the success class (final_rc == 0); every other class attests <none>.
   841	#
   842	# BLOCKING 3 — WHICH FILE is load-bearing too. train.py builds ModelCheckpoint
   843	# with Lightning's version counter at its default (enable_version_counter=True),
   844	# so a retry at the same boundary writes `epoch=E-step=N-v1.ckpt` beside the
   845	# unversioned file a previous failed attempt left behind. Globbing only the
   846	# unversioned name would attest the STALE bytes — another job's checkpoint. Both
   847	# name shapes are therefore searched and the NEWEST-mtime match wins: that is the
   848	# file THIS job just wrote. Its actual path is what is attested and recorded.
   849	ATTEST_LINE=""
   850	if [ -n "$CHUNK_END" ]; then
   851	  ENDPOINT_CKPT=""
   852	  if [ "$final_rc" -eq 0 ]; then
   853	    ENDPOINT_CKPT="$(ls -1t "${RUNDIR}/checkpoints"/*-step="${CHUNK_END}".ckpt \
   854	                              "${RUNDIR}/checkpoints"/*-step="${CHUNK_END}"-v*.ckpt 2>/dev/null | head -1)"
   855	  fi
   856	  if [ "$final_rc" -ne 0 ]; then
   857	    ATTEST_LINE="endpoint_ckpt <none> endpoint_step ${CHUNK_END} endpoint_class ${final_rc}"
   858	    echo "endpoint attestation: this leg's SETTLED class is ${final_rc}, not the success class — attesting <none>"
   859	  elif [ -z "$ENDPOINT_CKPT" ] || [ ! -f "$ENDPOINT_CKPT" ]; then
   860	    ATTEST_LINE="endpoint_ckpt <none> endpoint_step ${CHUNK_END} endpoint_class ${final_rc}"
   861	    echo "endpoint attestation: no *-step=${CHUNK_END}.ckpt or *-step=${CHUNK_END}-v*.ckpt under ${RUNDIR}/checkpoints — attesting <none>"
   862	  else
   863	    ENDPOINT_SHA="$(sha256sum "$ENDPOINT_CKPT" | awk '{print $1}')"
   864	    if [ -n "$ENDPOINT_SHA" ]; then
   865	      ATTEST_LINE="endpoint_ckpt ${ENDPOINT_CKPT} endpoint_step ${CHUNK_END} endpoint_sha256 ${ENDPOINT_SHA}"
   866	      echo "endpoint attestation: ${ENDPOINT_CKPT} step ${CHUNK_END} sha256 ${ENDPOINT_SHA} (newest match; settled class ${final_rc})"
   867	    else
   868	      ATTEST_LINE="endpoint_ckpt <none> endpoint_step ${CHUNK_END} endpoint_class ${final_rc}"
   869	      echo "endpoint attestation: sha256sum of ${ENDPOINT_CKPT} FAILED — attesting <none>"
   870	    fi
   871	  fi
   872	  # The manifest was written pre-run by THIS job; this is the same job appending
   873	  # to its own record. Both copies are kept identical.
   874	  if ! printf '%s\n' "$ATTEST_LINE" >> "$MANIFEST"; then
   875	    echo "ENDPOINT ATTESTATION COULD NOT BE APPENDED to ${MANIFEST} — the recorder will refuse this chunk (fail-closed); record it by hand only after auditing the checkpoint"
   876	  fi
   877	  printf '%s\n' "$ATTEST_LINE" >> "${SAVEDIR}/$(basename "$MANIFEST")" \
   878	    || echo "WARNING: could not append the endpoint attestation to the save-dir manifest copy"
   879	  # ...and into BOTH durable logs, so the transcript carries what the manifest
   880	  # carries. The classifier's log-equality check already ran; appending the same
   881	  # bytes to both copies keeps them identical.
   882	  printf 'endpoint attestation: %s\n' "$ATTEST_LINE" | tee -a "$TRAINLOG" "$SAVEDIR_LOG" > /dev/null \
   883	    || echo "WARNING: could not append the endpoint attestation to the durable logs"
   884	fi
   885	printf '%s\n' "$FINAL_RECORD"
   886	exit "$final_rc"

exec
/usr/bin/bash -lc "nl -ba worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch | sed -n '1,440p'" in /n/fs/gatrdp/codespace/FLAC
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

codex
The four named fixes are present in the main paths. I’m now stress-tracing the recovery path—especially failed jobs that leave a checkpoint—because that is where the versioned-name support and scheduler attestation interact.
exec
/usr/bin/bash -lc "nl -ba worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py | sed -n '1,330p'" in /n/fs/gatrdp/codespace/FLAC
exec
/usr/bin/bash -lc "nl -ba worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_record_restart.py | sed -n '341,700p'" in /n/fs/gatrdp/codespace/FLAC
 succeeded in 0ms:
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
   244	def check_link_manifest(i, link):
   245	    """Link ``i`` must agree with the MANIFEST it cites (round-5 r2, blocking 4).
   246	
   247	    Continuity alone proves only that the numbers in the registry line up, and
   248	    the registry is a mutable, uncommitted record (round-5 B1 deliberately took
   249	    it out of both drift gates). A single structurally valid forged link —
   250	    resume hash/step copied from the audited anchor, an increasing final step,
   251	    and an arbitrary final hash — therefore passed every ancestry check. So each
   252	    link is now checked against the artefact it names:
   253	
   254	      * its manifest_path exists and its BYTES hash to the recorded manifest_sha256;
   255	      * the manifest's job / launch_uuid are the link's;
   256	      * the manifest's endpoint attestation (step, sha256) is the link's
   257	        final_step / final_ckpt_sha256;
   258	      * the manifest's chunk_end is the link's final_step.
   259	
   260	    THREAT MODEL, honestly stated: this is COOPERATIVE INTEGRITY, not
   261	    cryptographic provenance. It defends against accidents (a stale glob, a
   262	    mis-recorded link, a manifest that drifted after recording) and casual
   263	    tampering (editing the registry by hand). It cannot defend against an author
   264	    who edits registry and manifest together: both are uncommitted files under
   265	    the same user between publication and commit, and nothing here is signed.
   266	    The audit trail is the git history of the committed records, not this check.
   267	    """
   268	    problems = []
   269	    job, uuid = link.get("job"), link.get("launch_uuid")
   270	    man_path, man_sha = link.get("manifest_path"), link.get("manifest_sha256")
   271	    if not man_path:
   272	        return [f"chain link {i} (job {job!r}) cites no manifest_path — a link with no manifest "
   273	                "is a bare assertion, not a record"]
   274	    if not os.path.isfile(man_path):
   275	        return [f"chain link {i} (job {job!r}) cites manifest {man_path}, which does not exist — "
   276	                "the record cannot be checked against the artefact it names"]
   277	    got = sha256_file(man_path)
   278	    if got != man_sha:
   279	        problems.append(f"chain link {i} (job {job!r}): its manifest {man_path} now hashes "
   280	                        f"{got[:12]}, not the recorded {str(man_sha)[:12]} — the manifest changed "
   281	                        "after the link was recorded")
   282	    man = parse_manifest(man_path)
   283	    jkv = kv_line(man, "job")
   284	    if jkv.get("job") != str(job):
   285	        problems.append(f"chain link {i}: its manifest records job {jkv.get('job')!r}, not the "
   286	                        f"link's {job!r}")
   287	    if jkv.get("launch_uuid") != uuid:
   288	        problems.append(f"chain link {i} (job {job!r}): its manifest records launch_uuid "
   289	                        f"{jkv.get('launch_uuid')!r}, not the link's {uuid!r}")
   290	    ckv = kv_line(man, "chunk_end")
   291	    if str(ckv.get("chunk_end")) != str(link.get("final_step")):
   292	        problems.append(f"chain link {i} (job {job!r}): its manifest declares chunk_end "
   293	                        f"{ckv.get('chunk_end')!r}, not the link's final_step "
   294	                        f"{link.get('final_step')!r}")
   295	    akv = kv_line(man, "endpoint_ckpt")
   296	    if "endpoint_ckpt" not in man:
   297	        problems.append(f"chain link {i} (job {job!r}): its manifest carries no endpoint "
   298	                        "attestation, so nothing in it says that job produced this checkpoint")
   299	        return problems
   300	    if str(akv.get("endpoint_step")) != str(link.get("final_step")):
   301	        problems.append(f"chain link {i} (job {job!r}): its manifest attests endpoint_step "
   302	                        f"{akv.get('endpoint_step')!r} != the link's final_step "
   303	                        f"{link.get('final_step')!r}")
   304	    if akv.get("endpoint_sha256") != link.get("final_ckpt_sha256"):
   305	        problems.append(f"chain link {i} (job {job!r}): its manifest attests endpoint_sha256 "
   306	                        f"{str(akv.get('endpoint_sha256'))[:12]} != the link's final_ckpt_sha256 "
   307	                        f"{str(link.get('final_ckpt_sha256'))[:12]} — the record does not match "
   308	                        "the manifest it cites")
   309	    return problems
   310	
   311	
   312	def check_chain_ancestry(reg, chain):
   313	    """EVERY link, from the audited INITIAL anchor to the tip (round-5 B6).
   314	
   315	    Checking only ``chain[-1]`` accepted a crafted registry: append a link whose
   316	    ``final_ckpt_sha256`` is whatever file you want to run, and the tip check
   317	    passes while nothing connects it to the audited 40k anchor. The chain is a
   318	    lineage, so it is verified as one — link 0 must resume the INITIAL anchor,
   319	    every later link must resume its predecessor's endpoint, and the steps must
   320	    strictly increase. Any break names the offending link INDEX.
   321	
   322	    Round-5 r2 (blocking 4): continuity is necessary but not sufficient — every
   323	    link is ALSO checked against the manifest it cites (check_link_manifest),
   324	    because a lone structurally valid forged link satisfied continuity by
   325	    construction.
   326	    """
   327	    problems = []
   328	    prev_sha, prev_step = reg.get("final_ckpt_sha256"), reg.get("final_step")
   329	    prev_label = "the audited INITIAL anchor"
   330	    for i, link in enumerate(chain):

 succeeded in 0ms:
   341	def record_chunk(args, arm, reg, initial, man, man_sha, pins, chunk_raw):
   342	    """Append ONE chain link for a chunked leg: <resume_step> -> <chunk_end>.
   343	
   344	    The link is what makes the NEXT chunk admissible (preflight --chain), so it
   345	    is written only when this leg provably resumed the tip of the chain and its
   346	    endpoint checkpoint is on disk, unambiguous, and hashed here from that disk.
   347	    """
   348	    jk, rk = kvs(man, "job"), kvs(man, "resume_ckpt")
   349	    job = jk.get("job")
   350	    problems = []
   351	    if args.extend:
   352	        problems.append("--extend extends a producer manifest and has no meaning for a CHUNK leg; "
   353	                        "each chunk is recorded once, as one immutable chain link")
   354	    if not job:
   355	        problems.append("manifest records no job — a leg with no identity is not a record")
   356	
   357	    # --- the SCHEDULER's verdict (round-5 r2 review, blocking 2) -------------
   358	    # The attestation is the job's own word; this is Slurm's. Both are required:
   359	    # the attestation says WHICH bytes, sacct says the job actually COMPLETED.
   360	    # --skip-sacct exists for documented MANUAL recovery only (accounting purged,
   361	    # a hand-audited leg) and is off by default — it is never used by the watchdog.
   362	    if job and not args.skip_sacct:
   363	        state, sacct_problem = sacct_state(job, args.sacct_bin)
   364	        if sacct_problem:
   365	            problems.append(sacct_problem)
   366	        elif state != "COMPLETED":
   367	            problems.append(f"the scheduler reports job {job} as {state!r}, not COMPLETED — a chunk "
   368	                            "link records a leg that the SCHEDULER agrees finished successfully; "
   369	                            "if this is a documented manual recovery, re-run with --skip-sacct and "
   370	                            "say so in the worklog")
   371	
   372	    # --- the chunk boundary itself ------------------------------------------
   373	    chunk_end = resume_step = None
   374	    try:
   375	        chunk_end = int(chunk_raw)
   376	    except (TypeError, ValueError):
   377	        problems.append(f"manifest chunk_end {chunk_raw!r} is not an integer")
   378	    try:
   379	        resume_step = int(rk.get("expected_step"))
   380	    except (TypeError, ValueError):
   381	        problems.append(f"manifest expected_step {rk.get('expected_step')!r} is not an integer")
   382	    budget = None
   383	    try:
   384	        budget = int(pins.get("PINNED_MAXSTEPS"))
   385	    except (TypeError, ValueError):
   386	        problems.append(f"the launcher's PINNED_MAXSTEPS {pins.get('PINNED_MAXSTEPS')!r} is not an "
   387	                        "integer — the chunk boundary cannot be checked against the budget")
   388	    if chunk_end is not None:
   389	        if chunk_end % CHUNK_STEP:
   390	            problems.append(f"manifest chunk_end {chunk_end} is not a multiple of {CHUNK_STEP} (the "
   391	                            "pinned checkpoint cadence: a chunk must end ON a checkpoint)")
   392	        if resume_step is not None and chunk_end <= resume_step:
   393	            problems.append(f"manifest chunk_end {chunk_end} does not exceed the resume step "
   394	                            f"{resume_step} — the leg would produce no new checkpoint")
   395	        if budget is not None and chunk_end > budget:
   396	            problems.append(f"manifest chunk_end {chunk_end} exceeds the pinned budget {budget}")
   397	
   398	    # --- which link (or the audited anchor) this chunk continues ------------
   399	    chain = initial.get("chain") or []
   400	    mine = [i for i, link in enumerate(chain) if link.get("job") == job]
   401	    if len(mine) > 1:
   402	        raise SystemExit(f"{arm} has {len(mine)} chain links claiming job {job} — the registry is "
   403	                         "inconsistent; fix it before recording")
   404	    at = mine[0] if mine else len(chain)
   405	    # Round-5 review NON-BLOCKING: one leg, one link — by job AND by identity. A
   406	    # different job id carrying an already-recorded launch uuid or manifest is an
   407	    # inconsistent registry, not a second chunk.
   408	    if not mine:
   409	        for i, link in enumerate(chain):
   410	            if jk.get("launch_uuid") and link.get("launch_uuid") == jk.get("launch_uuid"):
   411	                problems.append(f"chain link {i} (job {link.get('job')!r}) already carries launch_uuid "
   412	                                f"{jk.get('launch_uuid')!r} — a different job with the same launch uuid "
   413	                                "means the registry is inconsistent; fix it before recording")
   414	            if link.get("manifest_sha256") == man_sha:
   415	                problems.append(f"chain link {i} (job {link.get('job')!r}) was recorded from a manifest "
   416	                                f"with this exact sha256 {man_sha[:12]} — the same manifest cannot be "
   417	                                "two chunks")
   418	    prev = chain[at - 1] if at > 0 else None
   419	    if prev is None:
   420	        want_anchor = want_step = None                 # the audited INITIAL anchor
   421	        prev_desc = f"the audited INITIAL anchor at step {initial.get('final_step')}"
   422	    else:
   423	        want_anchor, want_step = prev.get("final_ckpt_sha256"), prev.get("final_step")
   424	        prev_desc = f"chain link job {prev.get('job')} ending at step {prev.get('final_step')}"
   425	        # Round-5 review B6: a predecessor that records no endpoint is NOT
   426	        # evidence, and falling back to the INITIAL 40k anchor here would silently
   427	        # re-parent this chunk onto the anchor — the chain's whole point is that
   428	        # every link continues the one before it. Refuse; the fallback exists
   429	        # only for the FIRST link (at == 0).
   430	        if not want_anchor or want_step is None:
   431	            problems.append(f"the predecessor chain link (index {at - 1}, job {prev.get('job')!r}) "
   432	                            "carries no final_ckpt_sha256/final_step — it is not evidence of a "
   433	                            "checkpoint, and a chunk with a predecessor is NEVER re-parented onto the "
   434	                            "audited INITIAL anchor; repair or re-record that link first")
   435	            print("RECORD REFUSED:")
   436	            for p in problems:
   437	                print(f"  !! {p}")
   438	            return 2
   439	    want_time = pins.get(f"PINNED_TIME_LIMIT_CHUNK_{arm}")
   440	    if not want_time:
   441	        problems.append(f"the launcher carries no PINNED_TIME_LIMIT_CHUNK_{arm} pin, so this "
   442	                        "chunk leg's wall time cannot be bound to a pin")
   443	    ident, resume_real = check_identity(arm, man, initial, pins, args.repo_root,
   444	                                        want_anchor=want_anchor, want_step=want_step,
   445	                                        want_time=want_time)
   446	    problems += ident
   447	
   448	    # --- the endpoint checkpoint this leg produced --------------------------
   449	    # Located by the job's OWN attestation, not by a name glob (blocking 3(b)):
   450	    # with Lightning's version counter a same-boundary retry writes `-v1`, so a
   451	    # glob would either bind the failed attempt's stale bytes or refuse forever.
   452	    ckpt_dir = canonical_ckpt_dir(initial.get("save_dir", ""), arm, args.repo_root)
   453	    endpoint, endpoint_sha, endpoint_problems = resolve_attested_endpoint(
   454	        man, chunk_end, ckpt_dir, args.repo_root)
   455	    problems += endpoint_problems
   456	    problems += check_endpoint_attestation(man, chunk_end, endpoint, endpoint_sha)
   457	    if problems:
   458	        print("RECORD REFUSED:")
   459	        for p in problems:
   460	            print(f"  !! {p}")
   461	        return 2
   462	
   463	    # check_identity already re-hashed the resume file and proved it IS this hash
   464	    resume_sha = want_anchor if want_anchor is not None else initial.get("final_ckpt_sha256")
   465	    # final_ckpt_path (blocking 3(b)): the ACTUAL endpoint file, versioned name
   466	    # and all, so the next chunk resumes the recorded tip by path instead of
   467	    # re-deriving one from a glob that cannot tell `-v1` from its stale twin.
   468	    link = {"job": job, "launch_uuid": jk.get("launch_uuid"),
   469	            "manifest_path": os.path.abspath(args.manifest), "manifest_sha256": man_sha,
   470	            "resume_step": resume_step, "resume_ckpt_sha256": resume_sha,
   471	            "final_step": chunk_end, "final_ckpt_sha256": endpoint_sha,
   472	            "final_ckpt_path": pm.rel_to(args.repo_root, endpoint),
   473	            "recorded_utc": utc_now()}
   474	
   475	    if mine:
   476	        old = chain[at]
   477	        differing = sorted(k for k in link if k != "recorded_utc"
   478	                           and str(old.get(k)) != str(link[k]))
   479	        if differing:
   480	            print("RECORD REFUSED:")
   481	            print(f"  !! {arm} job {job} is ALREADY a chain link whose content differs "
   482	                  f"({', '.join(differing)}) — a recorded link is immutable, and rewriting one "
   483	                  "would re-parent every chunk recorded after it")
   484	            return 2
   485	        print(f"{arm} chunk link job {job} ({resume_step} -> {chunk_end}) is already recorded, "
   486	              "byte-identical — no-op")
   487	        return 0
   488	
   489	    initial.setdefault("chain", []).append(link)
   490	    if not args.dry_run:
   491	        pm.write_atomic(args.registry, reg)
   492	    print(f"recorded {arm} chunk link job {job}: {resume_step} -> {chunk_end}, continuing "
   493	          f"{prev_desc} " + ("(dry run, nothing written)" if args.dry_run else "(published)"))
   494	    print(f"  resume {str(resume_sha)[:12]} -> endpoint {link['final_ckpt_sha256'][:12]} "
   495	          f"({pm.rel_to(args.repo_root, endpoint)})")
   496	    print(f"  {arm} chain is now {len(initial['chain'])} link(s), tip at step {chunk_end}"
   497	          f" of the {budget} budget")
   498	    return 0
   499	
   500	
   501	def main(argv=None):
   502	    ap = argparse.ArgumentParser(description="record an exp_11 RESTART leg")
   503	    ap.add_argument("arm")
   504	    ap.add_argument("manifest")
   505	    ap.add_argument("--registry", default=os.path.join(HERE, "arm_launch_registry.json"))
   506	    ap.add_argument("--launcher", default=os.path.join(HERE, "fa_orbit_train.sbatch"),
   507	                    help="where the Q10 pins are read from")
   508	    ap.add_argument("--producer-dir", default=HERE,
   509	                    help="where the per-leg producer manifests are published")
   510	    # HERE = <repo>/worklog/worklog_<user>/exp_11_fa_orbit_claude
   511	    ap.add_argument("--repo-root", default=os.path.dirname(os.path.dirname(os.path.dirname(HERE))),
   512	                    help="root the manifest's relative paths resolve against")
   513	    ap.add_argument("--extend", action="store_true",
   514	                    help="this leg is already recorded: extend its producer manifest only")
   515	    ap.add_argument("--rehash-all", action="store_true",
   516	                    help="re-hash published checkpoints too (full audit, expensive)")
   517	    ap.add_argument("--dry-run", action="store_true", help="validate and report, publish nothing")
   518	    ap.add_argument("--skip-sacct", action="store_true",
   519	                    help="DOCUMENTED MANUAL RECOVERY ONLY: skip the scheduler's COMPLETED "
   520	                         "confirmation for a chunk link. Off by default; the watchdog never "
   521	                         "passes it. Use only when accounting cannot answer for a leg you have "
   522	                         "audited by hand, and record why in the worklog.")
   523	    ap.add_argument("--sacct-bin", default=os.environ.get("SACCT_BIN", "sacct"),
   524	                    help="the sacct executable (test hook; changes no decision)")
   525	    args = ap.parse_args(argv)
   526	
   527	    arm = args.arm
   528	    pins = read_pins(args.launcher)
   529	    if not pins.get("PINNED_MAXSTEPS"):
   530	        raise SystemExit(f"no PINNED_* values found in {args.launcher}")
   531	
   532	    # One writer at a time, and the lock is the registry's own DIRECTORY: no lock
   533	    # file to leave behind in a tracked tree, and it still covers the tmp+rename.
   534	    store = os.path.dirname(os.path.abspath(args.registry)) or "."
   535	    lock_fd = os.open(store, os.O_RDONLY)
   536	    try:
   537	        fcntl.flock(lock_fd, fcntl.LOCK_EX)
   538	        return record(args, arm, pins)
   539	    finally:
   540	        os.close(lock_fd)
   541	
   542	
   543	def record(args, arm, pins):
   544	    reg = json.load(open(args.registry))
   545	    initial = reg.get("arms", {}).get(arm)
   546	    if initial is None:
   547	        raise SystemExit(f"{arm} has no INITIAL registry entry")
   548	
   549	    raw, man = parse_manifest(args.manifest)
   550	    man_sha = hashlib.sha256(raw).hexdigest()
   551	    # Round 5: a leg that carries a `chunk_end` line stopped at a chunk boundary
   552	    # and is recorded as a CHAIN LINK, not as a whole-budget RESTART row.
   553	    chunk_raw = chunk_end_of(man)
   554	    if chunk_raw is not None:
   555	        return record_chunk(args, arm, reg, initial, man, man_sha, pins, chunk_raw)
   556	    problems, resume_real = check_identity(arm, man, initial, pins, args.repo_root)
   557	    jk, ak, rk = kvs(man, "job"), kvs(man, "arm"), kvs(man, "resume_ckpt")
   558	    job = jk.get("job")
   559	
   560	    legs = reg.setdefault("restarts", {}).setdefault(arm, [])
   561	    same = [l for l in legs if l.get("job") == job or l.get("launch_uuid") == jk.get("launch_uuid")
   562	            or l.get("manifest_sha256") == man_sha]
   563	    if same and not args.extend:
   564	        raise SystemExit(f"{arm} job {job} is ALREADY recorded ({len(same)} matching leg(s)) — "
   565	                         "one leg, one row; use --extend to extend its producer manifest")
   566	    if len(same) > 1:
   567	        problems.append(f"{len(same)} registry rows already claim this leg — the registry is "
   568	                        "inconsistent; fix it before recording")
   569	    if args.extend and not same:
   570	        problems.append(f"--extend given but {arm} job {job} is not recorded yet")
   571	    if problems:
   572	        print("RECORD REFUSED:")
   573	        for p in problems:
   574	            print(f"  !! {p}")
   575	        return 2
   576	
   577	    anchor = initial["final_ckpt_sha256"]
   578	    producer = pm.manifest_name(arm, job)
   579	    row = {
   580	        "manifest_path": args.manifest, "manifest_sha256": man_sha,
   581	        "job": job, "mode": "RESTART", "launch_uuid": jk.get("launch_uuid"),
   582	        "arm": arm, "commit": man.get("commit"), "rung": ak.get("rung"),
   583	        "config_sha256": man.get("config_sha256"), "save_dir": man.get("save_dir"),
   584	        "resume_ckpt": resume_real, "resume_ckpt_sha256": anchor,
   585	        "expected_step": rk.get("expected_step"), "max_steps": ak.get("max_steps"),
   586	        "time_limit": kvs(man, "time_limit").get("time_limit"),
   587	        "producer_manifest": producer, "chains_to": anchor,
   588	        "recorded_at": pm._now(),
   589	    }
   590	    header = {"arm": arm, "job": job, "launch_uuid": jk.get("launch_uuid"), "mode": "RESTART",
   591	              "commit": man.get("commit"), "resume_ckpt_sha256": anchor,
   592	              "expected_step": rk.get("expected_step"), "max_steps": ak.get("max_steps"),
   593	              "save_dir": man.get("save_dir"), "config_sha256": man.get("config_sha256"),
   594	              "chains_to": anchor, "leg_manifest_sha256": man_sha}
   595	
   596	    ckpt_dir = canonical_ckpt_dir(initial["save_dir"], arm, args.repo_root)
   597	    prod_path = os.path.join(args.producer_dir, producer)
   598	    known = (pm.load(prod_path) or {}).get("checkpoints", {})
   599	    found, scan_problems = pm.scan_checkpoints(
   600	        ckpt_dir, int(rk["expected_step"]), int(ak["max_steps"]), known=known,
   601	        rehash_all=args.rehash_all, repo_root=args.repo_root)
   602	    added, kept, pub_problems = pm.publish(prod_path, header, found, dry_run=args.dry_run)
   603	    if scan_problems or pub_problems:
   604	        print("RECORD REFUSED:")
   605	        for p in scan_problems + pub_problems:
   606	            print(f"  !! {p}")
   607	        return 2
   608	
   609	    if args.extend:
   610	        for i, leg in enumerate(legs):
   611	            if leg.get("job") == job:
   612	                legs[i] = {**leg, "producer_manifest": producer}
   613	    else:
   614	        legs.append(row)
   615	    if not args.dry_run:
   616	        pm.write_atomic(args.registry, reg)
   617	    verb = "extended" if args.extend else "recorded"
   618	    print(f"{verb} {arm} RESTART job {job} chaining to {anchor[:12]} "
   619	          f"({'dry run, nothing written' if args.dry_run else 'published'})")
   620	    print(f"  producer manifest {producer}: {len(added)} checkpoint(s) added, "
   621	          f"{len(kept)} already published"
   622	          + (f" (steps {added[0]}..{added[-1]})" if added else ""))
   623	    return 0
   624	
   625	
   626	if __name__ == "__main__":
   627	    sys.exit(main())

exec
/usr/bin/bash -lc "nl -ba worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_record_restart.py | sed -n '1,340p'" in /n/fs/gatrdp/codespace/FLAC
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
    53	
    54	ROUND-5 r2 REVIEW. Two further requirements on a chunk link:
    55	
    56	  * THE SCHEDULER MUST AGREE (blocking 2). The attestation is the job's own word.
    57	    `sacct -X -n -P -j <job> -o State` must report exactly COMPLETED; an empty,
    58	    failing or unavailable sacct refuses. `--skip-sacct` (off by default, never
    59	    used by the watchdog) exists for documented manual recovery.
    60	  * THE ATTESTED PATH IS THE RECORD (blocking 3). Lightning's ModelCheckpoint
    61	    version counter means a retry at an already-written boundary saves
    62	    `epoch=E-step=N-v1.ckpt`. Globbing `*-step=N.ckpt` would bind the failed
    63	    attempt's stale bytes to the retry (or find two files and refuse forever), so
    64	    the recorder follows the attested path — verified to exist, to sit in the
    65	    canonical directory, and to carry this leg's step in its name — and stores it
    66	    in the link as `final_ckpt_path` for the next chunk to resume from.
    67	"""
    68	import argparse
    69	import fcntl
    70	import hashlib
    71	import json
    72	import os
    73	import re
    74	import subprocess
    75	import sys
    76	from datetime import datetime, timezone
    77	
    78	HERE = os.path.dirname(os.path.abspath(__file__))
    79	sys.path.insert(0, HERE)
    80	import fa_orbit_producer_manifest as pm            # noqa: E402
    81	from fa_orbit_ckpt_preflight import canonical_ckpt_dir    # noqa: E402
    82	
    83	PIN_RE = re.compile(r'^(PINNED_[A-Z0-9_]+)=(?:"([^"]*)"|(\S+))')
    84	# The pinned checkpoint cadence: a chunk may only end ON a saved checkpoint,
    85	# otherwise the next chunk has nothing to resume.
    86	CHUNK_STEP = 2500
    87	# Lightning's ModelCheckpoint version counter: a retry at a boundary whose
    88	# unversioned name already exists writes `epoch=E-step=N-v1.ckpt` (then -v2...).
    89	# An attested endpoint may therefore carry EITHER name shape (round-5 r2 B3).
    90	ENDPOINT_NAME_RE = re.compile(r"-step=(\d+)(?:-v\d+)?\.ckpt$")
    91	
    92	
    93	def sacct_state(job, sacct_bin="sacct"):
    94	    """(state, problem): what the SCHEDULER says about ``job``.
    95	
    96	    Round-5 r2 review, blocking 2. The producing job's own attestation says what
    97	    it wrote; it cannot say whether Slurm agrees the job finished — a leg killed
    98	    after it appended its attestation, or one whose node died during epilogue,
    99	    still leaves a positive attestation on disk. So a chunk link additionally
   100	    requires ``sacct`` to report exactly COMPLETED for the manifest's job.
   101	
   102	    Fail-CLOSED in every ambiguous direction: a missing/unrunnable ``sacct``, a
   103	    nonzero exit, and an EMPTY answer (the job is unknown to the accounting
   104	    database, or accounting is lagging) are all refusals, never "probably fine".
   105	    """
   106	    argv = [sacct_bin, "-X", "-n", "-P", "-j", str(job), "-o", "State"]
   107	    try:
   108	        proc = subprocess.run(argv, capture_output=True, text=True, timeout=120)
   109	    except (OSError, subprocess.SubprocessError) as exc:
   110	        return None, (f"could not ask the scheduler about job {job} "
   111	                      f"({' '.join(argv)}): {type(exc).__name__}: {exc} — a chunk link "
   112	                      "requires scheduler confirmation that the job COMPLETED")
   113	    if proc.returncode != 0:
   114	        return None, (f"`{' '.join(argv)}` exited {proc.returncode} "
   115	                      f"({proc.stderr.strip()[:200]}) — the scheduler's verdict on job {job} is "
   116	                      "UNKNOWN, and an unknown verdict is never read as a successful one")
   117	    lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
   118	    if not lines:
   119	        return None, (f"sacct reports nothing for job {job} — the scheduler cannot confirm it "
   120	                      "COMPLETED (accounting may be lagging, or this job never ran); a chunk "
   121	                      "link is never recorded on an unconfirmed job")
   122	    return lines[0], None
   123	
   124	
   125	def read_pins(launcher):
   126	    """The launcher's own PINNED_* values, so the recorder cannot drift from them."""
   127	    pins = {}
   128	    with open(launcher) as fh:
   129	        for line in fh:
   130	            m = PIN_RE.match(line)
   131	            if m:
   132	                pins[m.group(1)] = m.group(2) if m.group(2) is not None else m.group(3)
   133	    return pins
   134	
   135	
   136	def parse_manifest(path):
   137	    raw = open(path, "rb").read()
   138	    man = {}
   139	    for line in raw.decode().splitlines():
   140	        line = line.strip()
   141	        if line and not line.startswith("#"):
   142	            k, _, rest = line.partition(" ")
   143	            man[k] = rest.strip()
   144	    return raw, man
   145	
   146	
   147	def kvs(man, key):
   148	    f = (f"{key} " + man.get(key, "")).split()
   149	    return {f[i]: f[i + 1] for i in range(0, len(f) - 1, 2)}
   150	
   151	
   152	def check_identity(arm, man, initial, pins, repo_root,
   153	                   want_anchor=None, want_step=None, want_time=None):
   154	    """Every field of the RESTART manifest, against the audited INITIAL row + Q10 pins.
   155	
   156	    A CHUNK leg proves the same identity but resumes the tip of the recorded
   157	    chain rather than the audited 40k anchor, and is walled by the arm's CHUNK
   158	    pin rather than its RESTART pin, so the caller may override those three
   159	    expectations. Everything else is identical for both leg kinds."""
   160	    jk, ak, rk = kvs(man, "job"), kvs(man, "arm"), kvs(man, "resume_ckpt")
   161	    tk = kvs(man, "time_limit")
   162	    problems = []
   163	    anchor = want_anchor if want_anchor is not None else initial.get("final_ckpt_sha256")
   164	    final_step = want_step if want_step is not None else initial.get("final_step")
   165	    if not anchor:
   166	        problems.append(f"{arm} has no audited final_ckpt_sha256 to chain from — audit the "
   167	                        "INITIAL run's final checkpoint before recording a leg")
   168	    if jk.get("mode") != "RESTART":
   169	        problems.append(f"manifest mode is {jk.get('mode')!r}, not RESTART")
   170	    for field, got in (("job", jk.get("job")), ("launch_uuid", jk.get("launch_uuid")),
   171	                       ("commit", man.get("commit"))):
   172	        if not got:
   173	            problems.append(f"manifest records no {field} — a leg with no identity is not a record")
   174	    if jk.get("job") and initial.get("job") == jk.get("job"):
   175	        problems.append(f"manifest job {jk.get('job')} IS the INITIAL job — that is the launch "
   176	                        "already registered, not a restart leg")
   177	    for label, got, want in (("arm", ak.get("arm"), arm),
   178	                             ("rung", ak.get("rung"), initial.get("rung")),
   179	                             ("micro", ak.get("micro"), pins.get("PINNED_MB")),
   180	                             ("ngpu", ak.get("ngpu"), pins.get("PINNED_NGPU")),
   181	                             ("config_sha256", man.get("config_sha256"), initial.get("config_sha256")),
   182	                             ("vae_sha256", man.get("vae_sha256"), initial.get("vae_sha256")),
   183	                             ("p0_manifest_sha256", man.get("p0_manifest_sha256"),
   184	                              initial.get("p0_manifest_sha256")),
   185	                             ("save_dir", man.get("save_dir"), initial.get("save_dir"))):
   186	        if got != want:
   187	            problems.append(f"manifest {label} {got!r} != the audited INITIAL run's {want!r}")
   188	    if ak.get("rung") != pins.get("PINNED_RUNG"):
   189	        problems.append(f"manifest rung {ak.get('rung')!r} != the pinned {pins.get('PINNED_RUNG')!r}")
   190	    if ak.get("max_steps") != pins.get("PINNED_MAXSTEPS"):
   191	        problems.append(f"manifest max_steps {ak.get('max_steps')!r} != the Q10 budget pin "
   192	                        f"{pins.get('PINNED_MAXSTEPS')!r}")
   193	    step_label = "audited final step" if want_step is None else "recorded chain tip's final step"
   194	    if final_step is not None and str(rk.get("expected_step")) != str(final_step):
   195	        problems.append(f"manifest expected_step {rk.get('expected_step')!r} != the {step_label} "
   196	                        f"{final_step!r} — a leg resumes where the run it continues ended")
   197	    pin_label = "RESTART wall pin"
   198	    if want_time is None:
   199	        want_time = pins.get(f"PINNED_TIME_LIMIT_RESTART_{arm}")
   200	    else:
   201	        pin_label = "CHUNK wall pin"
   202	    if tk.get("time_limit") != want_time:
   203	        problems.append(f"manifest time_limit {tk.get('time_limit')!r} != the arm's {pin_label} "
   204	                        f"{want_time!r}")
   205	    if int(initial.get("training_seed", -1)) != 42:
   206	        problems.append(f"registered training seed {initial.get('training_seed')!r} != 42")
   207	    # the config the leg names must still hash to the audited value
   208	    cfg_path = man.get("model_config", "")
   209	    if not cfg_path or not os.path.isfile(cfg_path):
   210	        problems.append(f"manifest model_config {cfg_path!r} does not exist")
   211	    elif hashlib.sha256(open(cfg_path, "rb").read()).hexdigest() != initial.get("config_sha256"):
   212	        problems.append(f"{cfg_path} no longer hashes to the audited config_sha256")
   213	    # --- the resume file itself: MUST exist, MUST be canonical, ALWAYS re-hashed --
   214	    resume_path = (man.get("resume_ckpt", "").split() or [""])[0]
   215	    resume_real = ""
   216	    if not resume_path or resume_path == "<none>":
   217	        problems.append("manifest records no resume_ckpt — a RESTART that resumed nothing is not "
   218	                        "a continuation of the audited run")
   219	    elif not os.path.isfile(resolve(repo_root, resume_path)):
   220	        problems.append(f"the resume checkpoint {resume_path} does not exist — the recorder does "
   221	                        "NOT accept the manifest's claimed hash in its place")
   222	    else:
   223	        resume_real = resolve(repo_root, resume_path)
   224	        canon = canonical_ckpt_dir(initial.get("save_dir", ""), arm, repo_root)
   225	        if os.path.realpath(os.path.dirname(resume_real)) != canon:
   226	            problems.append(f"the resume checkpoint is not in the audited launch's canonical "
   227	                            f"directory {canon}")
   228	        got = pm.sha256_file(resume_real)
   229	        if anchor and got != anchor:
   230	            problems.append(f"the resume file on disk hashes {got[:12]}, not the audited "
   231	                            f"{anchor[:12]} — this leg does not continue that run")
   232	        if rk.get("resume_ckpt_sha256") != got:
   233	            problems.append(f"manifest resume_ckpt_sha256 {str(rk.get('resume_ckpt_sha256'))[:12]} "
   234	                            f"!= the file's actual {got[:12]}")
   235	    return problems, resume_real
   236	
   237	
   238	def resolve(root, path):
   239	    return path if os.path.isabs(path) else os.path.join(root, path)
   240	
   241	
   242	def chunk_end_of(man):
   243	    """The leg's per-job stop step as written by the launcher, or None.
   244	
   245	    A manifest with no `chunk_end` line (or the literal `<none>`) is a
   246	    whole-budget RESTART leg and takes the original recording path untouched."""
   247	    v = (man.get("chunk_end", "").split() or [""])[0]
   248	    return None if not v or v == "<none>" else v
   249	
   250	
   251	def utc_now():
   252	    return datetime.now(timezone.utc).isoformat(timespec="seconds")
   253	
   254	
   255	def resolve_attested_endpoint(man, chunk_end, ckpt_dir, repo_root):
   256	    """(endpoint_path, endpoint_sha, problems) for the ATTESTED endpoint file.
   257	
   258	    Round-5 r2 review, blocking 3(b). This used to glob
   259	    ``<canonical>/*-step=<chunk_end>.ckpt`` and require exactly one hit. That is
   260	    wrong twice over once Lightning's version counter is in play: a failed
   261	    attempt that saved, followed by a successful retry at the same boundary,
   262	    leaves BOTH ``epoch=E-step=N.ckpt`` (the stale attempt) and
   263	    ``epoch=E-step=N-v1.ckpt`` (the retry) — the glob then either binds the
   264	    predecessor's bytes to the retry, or finds two hits and refuses forever.
   265	
   266	    The producing job already told us which file it wrote. So the recorder now
   267	    follows the ATTESTED PATH and verifies it, rather than re-deriving a path
   268	    from a name pattern:
   269	
   270	      * the attested file must EXIST;
   271	      * it must sit in the audited launch's canonical checkpoint directory;
   272	      * its NAME must encode this leg's chunk_end (either name shape);
   273	      * it is re-hashed here, from disk, by this recorder.
   274	    """
   275	    attested = kvs(man, "endpoint_ckpt").get("endpoint_ckpt", "")
   276	    if not attested or attested == "<none>":
   277	        return None, None, []           # check_endpoint_attestation reports this
   278	    cand = resolve(repo_root, attested)
   279	    if not os.path.isfile(cand):
   280	        return None, None, [f"the attested endpoint checkpoint {attested} does not exist — the "
   281	                            "recorder does NOT accept the manifest's claimed hash in its place"]
   282	    if os.path.realpath(os.path.dirname(cand)) != ckpt_dir:
   283	        return None, None, [f"the attested endpoint checkpoint {attested} is not in the audited "
   284	                            f"launch's canonical directory {ckpt_dir}"]
   285	    m = ENDPOINT_NAME_RE.search(os.path.basename(cand))
   286	    if not m or (chunk_end is not None and int(m.group(1)) != int(chunk_end)):
   287	        return None, None, [f"the attested endpoint checkpoint {os.path.basename(cand)} does not "
   288	                            f"carry this leg's chunk end step {chunk_end} in its name (expected "
   289	                            f"`*-step={chunk_end}.ckpt` or `*-step={chunk_end}-v<N>.ckpt`)"]
   290	    return cand, pm.sha256_file(cand), []
   291	
   292	
   293	def check_endpoint_attestation(man, chunk_end, endpoint, endpoint_sha):
   294	    """The producing job's own statement of what it wrote (round-5 review B6).
   295	
   296	    The launcher publishes its manifest BEFORE training, so the manifest alone
   297	    proves intent, never authorship: a FAILED leg's manifest plus a pre-existing
   298	    checkpoint of the right name would otherwise become a chain link, and
   299	    re-hashing proves only which bytes are on disk now. After its exit class is
   300	    SETTLED the job appends to its own manifest
   301	
   302	        endpoint_ckpt <path> endpoint_step <n> endpoint_sha256 <sha>
   303	
   304	    (or `endpoint_ckpt <none> ...` for every non-success class), and this
   305	    function makes that attestation MANDATORY and fail-closed:
   306	
   307	      * no attestation at all           -> refuse (name the launcher append);
   308	      * `<none>`                        -> refuse (the leg says it produced none);
   309	      * endpoint_step != chunk_end      -> refuse (it attests another boundary);
   310	      * attested sha != our own re-hash -> refuse (it is not that file).
   311	
   312	    The attested PATH is resolved and located by resolve_attested_endpoint; the
   313	    ``endpoint``/``endpoint_sha`` passed here are that function's findings.
   314	    """
   315	    ak = kvs(man, "endpoint_ckpt")
   316	    if "endpoint_ckpt" not in man:
   317	        return ["the manifest carries no endpoint attestation — a chunk is recorded only on the "
   318	                "PRODUCING JOB's own statement of what it wrote. fa_orbit_train.sbatch appends "
   319	                "`endpoint_ckpt <path> endpoint_step <n> endpoint_sha256 <sha>` to this manifest "
   320	                "once its exit class is settled; a manifest without that line was either written "
   321	                "by an older launcher or belongs to a leg that never finished"]
   322	    attested = ak.get("endpoint_ckpt", "")
   323	    if not attested or attested == "<none>":
   324	        return [f"the producing job attested `endpoint_ckpt <none>` (class "
   325	                f"{ak.get('endpoint_class', '?')}) — it did not reach the success class, or wrote no "
   326	                "checkpoint at its chunk end; a leg that attests nothing is never a chain link"]
   327	    problems = []
   328	    if str(ak.get("endpoint_step")) != str(chunk_end):
   329	        problems.append(f"the attested endpoint_step {ak.get('endpoint_step')!r} != this leg's "
   330	                        f"chunk_end {chunk_end!r} — the job attests a different boundary")
   331	    if endpoint is None:
   332	        return problems      # the unresolvable endpoint is already a problem upstream
   333	    if ak.get("endpoint_sha256") != endpoint_sha:
   334	        problems.append(f"the attested endpoint_sha256 "
   335	                        f"{str(ak.get('endpoint_sha256'))[:12]} != this recorder's own re-hash of "
   336	                        f"{os.path.basename(endpoint)} ({str(endpoint_sha)[:12]}) — the file on disk "
   337	                        "is not the file that job attested writing")
   338	    return problems
   339	
   340	

exec
/usr/bin/bash -lc "nl -ba worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py | sed -n '331,620p'" in /n/fs/gatrdp/codespace/FLAC
 succeeded in 0ms:
   331	        job = link.get("job")
   332	        r_sha, r_step = link.get("resume_ckpt_sha256"), link.get("resume_step")
   333	        f_sha, f_step = link.get("final_ckpt_sha256"), link.get("final_step")
   334	        problems += check_link_manifest(i, link)
   335	        if not f_sha or f_step is None:
   336	            problems.append(f"chain link {i} (job {job!r}) carries no final_ckpt_sha256/final_step — "
   337	                            "it is not evidence of a checkpoint")
   338	        if prev_sha and r_sha != prev_sha:
   339	            problems.append(f"chain link {i} (job {job!r}) resume_ckpt_sha256 {str(r_sha)[:12]} != "
   340	                            f"{prev_label}'s final_ckpt_sha256 {str(prev_sha)[:12]} — the chain is "
   341	                            f"BROKEN at link {i}: it does not continue what precedes it")
   342	        if prev_step is not None and str(r_step) != str(prev_step):
   343	            problems.append(f"chain link {i} (job {job!r}) resume_step {r_step!r} != {prev_label}'s "
   344	                            f"final_step {prev_step!r} — the chain is BROKEN at link {i}")
   345	        try:
   346	            if prev_step is not None and f_step is not None and int(f_step) <= int(prev_step):
   347	                problems.append(f"chain link {i} (job {job!r}) final_step {f_step!r} does not exceed "
   348	                                f"{prev_label}'s {prev_step!r} — chunk steps only ever increase")
   349	        except (TypeError, ValueError):
   350	            problems.append(f"chain link {i} (job {job!r}) has a non-integer step "
   351	                            f"(resume {r_step!r}, final {f_step!r})")
   352	        prev_sha, prev_step, prev_label = f_sha, f_step, f"chain link {i}"
   353	    return problems
   354	
   355	
   356	def check_chain_binding(manifest_path, registry_path, arm, rung, config_path, ckpt_path,
   357	                        ckpt_sha, expected_step, max_steps, repo_root="."):
   358	    """The CHUNKED-LEG contract: resume the TIP of this arm's recorded chain.
   359	
   360	    A chunk leg is an extension leg that stops early, so it must prove exactly
   361	    the same INITIAL launch identity — but its resume point is no longer the
   362	    audited 40k anchor: it is the endpoint of the previous chunk, recorded as the
   363	    last link of ``arms.<ARM>.chain`` by fa_orbit_record_restart.py.
   364	
   365	    Fail-closed by construction: an EMPTY or ABSENT chain refuses, so chunk N+1
   366	    cannot run until chunk N has been recorded, and every link's final_step /
   367	    final_ckpt_sha256 was written by the recorder from the file on disk. The
   368	    resume file is re-hashed by the caller (``ckpt_sha``); a manifest's claimed
   369	    hash is never trusted here or anywhere else in this chain.
   370	
   371	    Round-5 review B6: the WHOLE ancestry is validated (check_chain_ancestry),
   372	    not merely ``chain[-1]`` — otherwise a crafted registry could append a tip
   373	    with an arbitrary hash and no link back to the audited 40k anchor.
   374	
   375	    Round-5 r2 (blocking 3(d)): nothing here assumes a checkpoint FILENAME shape.
   376	    The resume file arrives as an explicit ``--ckpt`` path and is identified by
   377	    its sha256 and its directory, so a Lightning-versioned endpoint
   378	    (``epoch=E-step=N-v1.ckpt``, written when a retry finds the unversioned name
   379	    taken) is admissible exactly like an unversioned one.
   380	    """
   381	    reg, problems = _load_registry_row(registry_path, arm)
   382	    if problems:
   383	        return problems, {}
   384	    man = parse_manifest(manifest_path)
   385	    problems = check_initial_identity(man, manifest_path, reg, arm, rung, config_path, max_steps)
   386	    if not reg.get("final_ckpt_sha256"):
   387	        problems.append(f"{arm} has no audited final_ckpt_sha256 in the registry — a chain of "
   388	                        "chunks must still descend from the audited INITIAL run")
   389	    chain = reg.get("chain") or []
   390	    if not chain:
   391	        problems.append(f"{arm} has no recorded chain link in {registry_path}: a chunk may only "
   392	                        "resume the tip of a RECORDED chain, so chunk N+1 is inadmissible until "
   393	                        "chunk N is recorded — run fa_orbit_record_restart.py on the previous "
   394	                        "chunk's launcher manifest first")
   395	    else:
   396	        # B6: the FULL ancestry, not just the tip — a tip alone can be crafted.
   397	        problems += check_chain_ancestry(reg, chain)
   398	        tip = chain[-1]
   399	        tip_step, tip_sha = tip.get("final_step"), tip.get("final_ckpt_sha256")
   400	        if str(tip_step) != str(expected_step):
   401	            problems.append(f"EXPECTED_STEP {expected_step} != the last recorded chain link's "
   402	                            f"final_step {tip_step!r} (link job {tip.get('job')!r}) — a chunk "
   403	                            "resumes the TIP of the chain, never an earlier link")
   404	        if not tip_sha:
   405	            problems.append(f"the last recorded chain link (job {tip.get('job')!r}) carries no "
   406	                            "final_ckpt_sha256 — it is not evidence of a checkpoint")
   407	        elif ckpt_sha != tip_sha:
   408	            problems.append(f"resume checkpoint sha256 {ckpt_sha[:12]} != the last recorded chain "
   409	                            f"link's final_ckpt_sha256 {str(tip_sha)[:12]} — this file is not the "
   410	                            "checkpoint that chunk produced")
   411	    problems += check_canonical_dir(man, arm, ckpt_path, repo_root)
   412	    return problems, man
   413	
   414	
   415	def main(argv=None):
   416	    ap = argparse.ArgumentParser(description="exp_11 restart checkpoint preflight")
   417	    ap.add_argument("--ckpt", required=True)
   418	    ap.add_argument("--expected-step", type=int, required=True)
   419	    ap.add_argument("--config", required=True, help="this arm's model config json")
   420	    ap.add_argument("--max-steps", type=int, required=True)
   421	    ap.add_argument("--arm", required=True)
   422	    ap.add_argument("--rung", required=True)
   423	    ap.add_argument("--commit", default="")
   424	    ap.add_argument("--launch-manifest", default="",
   425	                    help="the arm's original launch manifest (binds rung/commit/budget)")
   426	    ap.add_argument("--extension", action="store_true",
   427	                    help="this restart is the Q10 40k->100k EXTENSION: bind it to the audited "
   428	                         "INITIAL launch identity instead of requiring an equal budget/commit")
   429	    ap.add_argument("--chain", action="store_true",
   430	                    help="this restart is a CHUNK continuing an already-recorded chain: same "
   431	                         "INITIAL identity as --extension, but the resume anchor is the LAST "
   432	                         "recorded arms.<ARM>.chain link instead of the audited 40k checkpoint")
   433	    ap.add_argument("--chunk-end", type=int, default=None,
   434	                    help="this leg's stop step (a chunk boundary): EXPECTED_STEP < N <= max-steps "
   435	                         "and a multiple of 2500")
   436	    ap.add_argument("--launch-registry", default="",
   437	                    help="the committed arm launch registry (required with --extension/--chain)")
   438	    ap.add_argument("--repo-root", default=".",
   439	                    help="root the registry's relative save_dir is resolved against")
   440	    args = ap.parse_args(argv)
   441	    if args.extension and args.chain:
   442	        ap.error("--extension and --chain are mutually exclusive: a leg either resumes the audited "
   443	                 "40k anchor or the tip of the recorded chunk chain, never both")
   444	    if args.extension and not args.launch_registry:
   445	        ap.error("--extension requires --launch-registry (the audited INITIAL launch row)")
   446	    if args.chain and not args.launch_registry:
   447	        ap.error("--chain requires --launch-registry (the recorded chunk chain lives in it)")
   448	
   449	    if not os.path.isfile(args.ckpt):
   450	        print(f"PREFLIGHT: checkpoint not found: {args.ckpt}")
   451	        return 2
   452	
   453	    import torch  # deferred: keeps --help and unit imports cheap
   454	
   455	    try:
   456	        ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
   457	    except Exception as exc:                      # truncated, empty or not a torch file
   458	        print(f"PREFLIGHT: {args.ckpt} is not loadable as a checkpoint: "
   459	              f"{type(exc).__name__}: {exc}")
   460	        return 2
   461	    if not isinstance(ck, dict):
   462	        print(f"PREFLIGHT: not a Lightning checkpoint: {args.ckpt}")
   463	        return 2
   464	
   465	    problems = []
   466	    gs = ck.get("global_step")
   467	    if gs != args.expected_step:
   468	        problems.append(f"global_step {gs} != EXPECTED_STEP {args.expected_step}")
   469	    if isinstance(gs, int) and gs >= args.max_steps:
   470	        problems.append(f"global_step {gs} >= max_steps {args.max_steps}: Lightning would stop "
   471	                        "immediately and still print the completion literal")
   472	
   473	    mc = ck.get("model_config")
   474	    if not isinstance(mc, dict):
   475	        problems.append("checkpoint carries no embedded 'model_config' dict")
   476	    else:
   477	        want = json.load(open(args.config))
   478	        if mc != want:
   479	            tr = mc.get("training", {}) if isinstance(mc.get("training"), dict) else {}
   480	            problems.append(
   481	                f"embedded model_config != {args.config} (parsed-object mismatch; embedded "
   482	                f"cond_method={tr.get('cond_method')!r} angles={tr.get('frame_avg_angles')!r})")
   483	
   484	    opts = ck.get("optimizer_states") or []
   485	    if len(opts) != 1:
   486	        problems.append(f"expected exactly 1 optimizer entry, found {len(opts)}")
   487	    elif not opts[0].get("state"):
   488	        problems.append("optimizer state is CLEARED (stripped checkpoint); exp_11 restarts are "
   489	                        "WARM continuations and have no optimizer-reset lineage")
   490	    if not ck.get("lr_schedulers"):
   491	        problems.append("no 'lr_schedulers' -> PL 2.1 KeyErrors on resume")
   492	    sd = ck.get("state_dict") or {}
   493	    n_ema = sum(1 for k in sd if k.startswith("diffusion_ema."))
   494	    if not n_ema:
   495	        problems.append("no EMA weights in state_dict")
   496	
   497	    # The per-job stop step is checked against the BUDGET (--max-steps), which the
   498	    # launcher still pins at 100000: a chunk narrows where this leg stops, never
   499	    # what the campaign is allowed to reach.
   500	    if args.chunk_end is not None:
   501	        if args.chunk_end % 2500 != 0:
   502	            problems.append(f"--chunk-end {args.chunk_end} is not a multiple of 2500 (the pinned "
   503	                            "checkpoint cadence: a chunk must end ON a checkpoint)")
   504	        if not args.expected_step < args.chunk_end <= args.max_steps:
   505	            problems.append(f"--chunk-end {args.chunk_end} must satisfy EXPECTED_STEP "
   506	                            f"{args.expected_step} < chunk_end <= max_steps {args.max_steps}")
   507	
   508	    digest = sha256_file(args.ckpt)     # needed by the extension/chain anchor checks
   509	    man = {}
   510	    if args.launch_manifest:
   511	        if not os.path.isfile(args.launch_manifest):
   512	            problems.append(f"launch manifest not found: {args.launch_manifest}")
   513	        elif args.extension:
   514	            more, man = check_extension_binding(
   515	                args.launch_manifest, args.launch_registry, args.arm, args.rung, args.config,
   516	                args.ckpt, digest, args.expected_step, args.max_steps, args.repo_root)
   517	            problems += more
   518	        elif args.chain:
   519	            more, man = check_chain_binding(
   520	                args.launch_manifest, args.launch_registry, args.arm, args.rung, args.config,
   521	                args.ckpt, digest, args.expected_step, args.max_steps, args.repo_root)
   522	            problems += more
   523	        else:
   524	            more, man = check_manifest_binding(args.launch_manifest, args.arm, args.rung,
   525	                                               args.commit, args.max_steps)
   526	            problems += more
   527	    elif args.extension:
   528	        problems.append("--extension requires --launch-manifest (the audited INITIAL manifest)")
   529	    elif args.chain:
   530	        problems.append("--chain requires --launch-manifest (the audited INITIAL manifest)")
   531	
   532	    if problems:
   533	        print("PREFLIGHT FAILED:")
   534	        for p in problems:
   535	            print(f"  !! {p}")
   536	        return 2
   537	
   538	    tr = mc.get("training", {})
   539	    print(f"restart lineage OK: {args.ckpt}")
   540	    print(f"  global_step={gs} epoch={ck.get('epoch')} cond_method={tr.get('cond_method')!r} "
   541	          f"angles={tr.get('frame_avg_angles')}")
   542	    print(f"  optimizer_state=FULL ({len(opts[0]['state'])} entries) "
   543	          f"lr={opts[0]['param_groups'][0].get('lr')} "
   544	          f"sched_last_epoch={ck['lr_schedulers'][0].get('last_epoch')} ema_entries={n_ema}")
   545	    if man and args.extension:
   546	        print(f"extension lineage OK: {args.arm} {args.expected_step} -> "
   547	              f"{args.chunk_end if args.chunk_end is not None else args.max_steps} continues "
   548	              f"the audited launch job {kv_line(man, 'job').get('job')} "
   549	              f"(launch commit {man.get('commit', '')[:12]}, running commit {args.commit[:12] or '<none>'})")
   550	        print(f"  bound to the audited launch manifest: {args.launch_manifest}")
   551	        if args.chunk_end is not None:
   552	            print(f"  chunk leg: stops at {args.chunk_end} of the {args.max_steps} budget")
   553	    elif man and args.chain:
   554	        print(f"chain lineage OK: {args.arm} {args.expected_step} -> "
   555	              f"{args.chunk_end if args.chunk_end is not None else args.max_steps} resumes the tip "
   556	              f"of the recorded chunk chain, under the audited launch job "
   557	              f"{kv_line(man, 'job').get('job')} "
   558	              f"(launch commit {man.get('commit', '')[:12]}, running commit {args.commit[:12] or '<none>'})")
   559	        print(f"  bound to the audited launch manifest: {args.launch_manifest}")
   560	        if args.chunk_end is not None:
   561	            print(f"  chunk leg: stops at {args.chunk_end} of the {args.max_steps} budget")
   562	    elif man:
   563	        print(f"  bound to launch manifest: {args.launch_manifest}")
   564	    print(f"CKPT_SHA256 {digest}")
   565	    return 0
   566	
   567	
   568	if __name__ == "__main__":
   569	    sys.exit(main())

2026-08-13T17:24:00.647665Z ERROR codex_models_manager::manager: failed to renew cache TTL: missing field `base_instructions` at line 94 column 5
exec
/usr/bin/bash -lc "nl -ba worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_record_restart.py | sed -n '250,470p'" in /n/fs/gatrdp/codespace/FLAC
exec
/usr/bin/bash -lc "nl -ba worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py | sed -n '241,420p'" in /n/fs/gatrdp/codespace/FLAC
 succeeded in 0ms:
   250	
   251	def utc_now():
   252	    return datetime.now(timezone.utc).isoformat(timespec="seconds")
   253	
   254	
   255	def resolve_attested_endpoint(man, chunk_end, ckpt_dir, repo_root):
   256	    """(endpoint_path, endpoint_sha, problems) for the ATTESTED endpoint file.
   257	
   258	    Round-5 r2 review, blocking 3(b). This used to glob
   259	    ``<canonical>/*-step=<chunk_end>.ckpt`` and require exactly one hit. That is
   260	    wrong twice over once Lightning's version counter is in play: a failed
   261	    attempt that saved, followed by a successful retry at the same boundary,
   262	    leaves BOTH ``epoch=E-step=N.ckpt`` (the stale attempt) and
   263	    ``epoch=E-step=N-v1.ckpt`` (the retry) — the glob then either binds the
   264	    predecessor's bytes to the retry, or finds two hits and refuses forever.
   265	
   266	    The producing job already told us which file it wrote. So the recorder now
   267	    follows the ATTESTED PATH and verifies it, rather than re-deriving a path
   268	    from a name pattern:
   269	
   270	      * the attested file must EXIST;
   271	      * it must sit in the audited launch's canonical checkpoint directory;
   272	      * its NAME must encode this leg's chunk_end (either name shape);
   273	      * it is re-hashed here, from disk, by this recorder.
   274	    """
   275	    attested = kvs(man, "endpoint_ckpt").get("endpoint_ckpt", "")
   276	    if not attested or attested == "<none>":
   277	        return None, None, []           # check_endpoint_attestation reports this
   278	    cand = resolve(repo_root, attested)
   279	    if not os.path.isfile(cand):
   280	        return None, None, [f"the attested endpoint checkpoint {attested} does not exist — the "
   281	                            "recorder does NOT accept the manifest's claimed hash in its place"]
   282	    if os.path.realpath(os.path.dirname(cand)) != ckpt_dir:
   283	        return None, None, [f"the attested endpoint checkpoint {attested} is not in the audited "
   284	                            f"launch's canonical directory {ckpt_dir}"]
   285	    m = ENDPOINT_NAME_RE.search(os.path.basename(cand))
   286	    if not m or (chunk_end is not None and int(m.group(1)) != int(chunk_end)):
   287	        return None, None, [f"the attested endpoint checkpoint {os.path.basename(cand)} does not "
   288	                            f"carry this leg's chunk end step {chunk_end} in its name (expected "
   289	                            f"`*-step={chunk_end}.ckpt` or `*-step={chunk_end}-v<N>.ckpt`)"]
   290	    return cand, pm.sha256_file(cand), []
   291	
   292	
   293	def check_endpoint_attestation(man, chunk_end, endpoint, endpoint_sha):
   294	    """The producing job's own statement of what it wrote (round-5 review B6).
   295	
   296	    The launcher publishes its manifest BEFORE training, so the manifest alone
   297	    proves intent, never authorship: a FAILED leg's manifest plus a pre-existing
   298	    checkpoint of the right name would otherwise become a chain link, and
   299	    re-hashing proves only which bytes are on disk now. After its exit class is
   300	    SETTLED the job appends to its own manifest
   301	
   302	        endpoint_ckpt <path> endpoint_step <n> endpoint_sha256 <sha>
   303	
   304	    (or `endpoint_ckpt <none> ...` for every non-success class), and this
   305	    function makes that attestation MANDATORY and fail-closed:
   306	
   307	      * no attestation at all           -> refuse (name the launcher append);
   308	      * `<none>`                        -> refuse (the leg says it produced none);
   309	      * endpoint_step != chunk_end      -> refuse (it attests another boundary);
   310	      * attested sha != our own re-hash -> refuse (it is not that file).
   311	
   312	    The attested PATH is resolved and located by resolve_attested_endpoint; the
   313	    ``endpoint``/``endpoint_sha`` passed here are that function's findings.
   314	    """
   315	    ak = kvs(man, "endpoint_ckpt")
   316	    if "endpoint_ckpt" not in man:
   317	        return ["the manifest carries no endpoint attestation — a chunk is recorded only on the "
   318	                "PRODUCING JOB's own statement of what it wrote. fa_orbit_train.sbatch appends "
   319	                "`endpoint_ckpt <path> endpoint_step <n> endpoint_sha256 <sha>` to this manifest "
   320	                "once its exit class is settled; a manifest without that line was either written "
   321	                "by an older launcher or belongs to a leg that never finished"]
   322	    attested = ak.get("endpoint_ckpt", "")
   323	    if not attested or attested == "<none>":
   324	        return [f"the producing job attested `endpoint_ckpt <none>` (class "
   325	                f"{ak.get('endpoint_class', '?')}) — it did not reach the success class, or wrote no "
   326	                "checkpoint at its chunk end; a leg that attests nothing is never a chain link"]
   327	    problems = []
   328	    if str(ak.get("endpoint_step")) != str(chunk_end):
   329	        problems.append(f"the attested endpoint_step {ak.get('endpoint_step')!r} != this leg's "
   330	                        f"chunk_end {chunk_end!r} — the job attests a different boundary")
   331	    if endpoint is None:
   332	        return problems      # the unresolvable endpoint is already a problem upstream
   333	    if ak.get("endpoint_sha256") != endpoint_sha:
   334	        problems.append(f"the attested endpoint_sha256 "
   335	                        f"{str(ak.get('endpoint_sha256'))[:12]} != this recorder's own re-hash of "
   336	                        f"{os.path.basename(endpoint)} ({str(endpoint_sha)[:12]}) — the file on disk "
   337	                        "is not the file that job attested writing")
   338	    return problems
   339	
   340	
   341	def record_chunk(args, arm, reg, initial, man, man_sha, pins, chunk_raw):
   342	    """Append ONE chain link for a chunked leg: <resume_step> -> <chunk_end>.
   343	
   344	    The link is what makes the NEXT chunk admissible (preflight --chain), so it
   345	    is written only when this leg provably resumed the tip of the chain and its
   346	    endpoint checkpoint is on disk, unambiguous, and hashed here from that disk.
   347	    """
   348	    jk, rk = kvs(man, "job"), kvs(man, "resume_ckpt")
   349	    job = jk.get("job")
   350	    problems = []
   351	    if args.extend:
   352	        problems.append("--extend extends a producer manifest and has no meaning for a CHUNK leg; "
   353	                        "each chunk is recorded once, as one immutable chain link")
   354	    if not job:
   355	        problems.append("manifest records no job — a leg with no identity is not a record")
   356	
   357	    # --- the SCHEDULER's verdict (round-5 r2 review, blocking 2) -------------
   358	    # The attestation is the job's own word; this is Slurm's. Both are required:
   359	    # the attestation says WHICH bytes, sacct says the job actually COMPLETED.
   360	    # --skip-sacct exists for documented MANUAL recovery only (accounting purged,
   361	    # a hand-audited leg) and is off by default — it is never used by the watchdog.
   362	    if job and not args.skip_sacct:
   363	        state, sacct_problem = sacct_state(job, args.sacct_bin)
   364	        if sacct_problem:
   365	            problems.append(sacct_problem)
   366	        elif state != "COMPLETED":
   367	            problems.append(f"the scheduler reports job {job} as {state!r}, not COMPLETED — a chunk "
   368	                            "link records a leg that the SCHEDULER agrees finished successfully; "
   369	                            "if this is a documented manual recovery, re-run with --skip-sacct and "
   370	                            "say so in the worklog")
   371	
   372	    # --- the chunk boundary itself ------------------------------------------
   373	    chunk_end = resume_step = None
   374	    try:
   375	        chunk_end = int(chunk_raw)
   376	    except (TypeError, ValueError):
   377	        problems.append(f"manifest chunk_end {chunk_raw!r} is not an integer")
   378	    try:
   379	        resume_step = int(rk.get("expected_step"))
   380	    except (TypeError, ValueError):
   381	        problems.append(f"manifest expected_step {rk.get('expected_step')!r} is not an integer")
   382	    budget = None
   383	    try:
   384	        budget = int(pins.get("PINNED_MAXSTEPS"))
   385	    except (TypeError, ValueError):
   386	        problems.append(f"the launcher's PINNED_MAXSTEPS {pins.get('PINNED_MAXSTEPS')!r} is not an "
   387	                        "integer — the chunk boundary cannot be checked against the budget")
   388	    if chunk_end is not None:
   389	        if chunk_end % CHUNK_STEP:
   390	            problems.append(f"manifest chunk_end {chunk_end} is not a multiple of {CHUNK_STEP} (the "
   391	                            "pinned checkpoint cadence: a chunk must end ON a checkpoint)")
   392	        if resume_step is not None and chunk_end <= resume_step:
   393	            problems.append(f"manifest chunk_end {chunk_end} does not exceed the resume step "
   394	                            f"{resume_step} — the leg would produce no new checkpoint")
   395	        if budget is not None and chunk_end > budget:
   396	            problems.append(f"manifest chunk_end {chunk_end} exceeds the pinned budget {budget}")
   397	
   398	    # --- which link (or the audited anchor) this chunk continues ------------
   399	    chain = initial.get("chain") or []
   400	    mine = [i for i, link in enumerate(chain) if link.get("job") == job]
   401	    if len(mine) > 1:
   402	        raise SystemExit(f"{arm} has {len(mine)} chain links claiming job {job} — the registry is "
   403	                         "inconsistent; fix it before recording")
   404	    at = mine[0] if mine else len(chain)
   405	    # Round-5 review NON-BLOCKING: one leg, one link — by job AND by identity. A
   406	    # different job id carrying an already-recorded launch uuid or manifest is an
   407	    # inconsistent registry, not a second chunk.
   408	    if not mine:
   409	        for i, link in enumerate(chain):
   410	            if jk.get("launch_uuid") and link.get("launch_uuid") == jk.get("launch_uuid"):
   411	                problems.append(f"chain link {i} (job {link.get('job')!r}) already carries launch_uuid "
   412	                                f"{jk.get('launch_uuid')!r} — a different job with the same launch uuid "
   413	                                "means the registry is inconsistent; fix it before recording")
   414	            if link.get("manifest_sha256") == man_sha:
   415	                problems.append(f"chain link {i} (job {link.get('job')!r}) was recorded from a manifest "
   416	                                f"with this exact sha256 {man_sha[:12]} — the same manifest cannot be "
   417	                                "two chunks")
   418	    prev = chain[at - 1] if at > 0 else None
   419	    if prev is None:
   420	        want_anchor = want_step = None                 # the audited INITIAL anchor
   421	        prev_desc = f"the audited INITIAL anchor at step {initial.get('final_step')}"
   422	    else:
   423	        want_anchor, want_step = prev.get("final_ckpt_sha256"), prev.get("final_step")
   424	        prev_desc = f"chain link job {prev.get('job')} ending at step {prev.get('final_step')}"
   425	        # Round-5 review B6: a predecessor that records no endpoint is NOT
   426	        # evidence, and falling back to the INITIAL 40k anchor here would silently
   427	        # re-parent this chunk onto the anchor — the chain's whole point is that
   428	        # every link continues the one before it. Refuse; the fallback exists
   429	        # only for the FIRST link (at == 0).
   430	        if not want_anchor or want_step is None:
   431	            problems.append(f"the predecessor chain link (index {at - 1}, job {prev.get('job')!r}) "
   432	                            "carries no final_ckpt_sha256/final_step — it is not evidence of a "
   433	                            "checkpoint, and a chunk with a predecessor is NEVER re-parented onto the "
   434	                            "audited INITIAL anchor; repair or re-record that link first")
   435	            print("RECORD REFUSED:")
   436	            for p in problems:
   437	                print(f"  !! {p}")
   438	            return 2
   439	    want_time = pins.get(f"PINNED_TIME_LIMIT_CHUNK_{arm}")
   440	    if not want_time:
   441	        problems.append(f"the launcher carries no PINNED_TIME_LIMIT_CHUNK_{arm} pin, so this "
   442	                        "chunk leg's wall time cannot be bound to a pin")
   443	    ident, resume_real = check_identity(arm, man, initial, pins, args.repo_root,
   444	                                        want_anchor=want_anchor, want_step=want_step,
   445	                                        want_time=want_time)
   446	    problems += ident
   447	
   448	    # --- the endpoint checkpoint this leg produced --------------------------
   449	    # Located by the job's OWN attestation, not by a name glob (blocking 3(b)):
   450	    # with Lightning's version counter a same-boundary retry writes `-v1`, so a
   451	    # glob would either bind the failed attempt's stale bytes or refuse forever.
   452	    ckpt_dir = canonical_ckpt_dir(initial.get("save_dir", ""), arm, args.repo_root)
   453	    endpoint, endpoint_sha, endpoint_problems = resolve_attested_endpoint(
   454	        man, chunk_end, ckpt_dir, args.repo_root)
   455	    problems += endpoint_problems
   456	    problems += check_endpoint_attestation(man, chunk_end, endpoint, endpoint_sha)
   457	    if problems:
   458	        print("RECORD REFUSED:")
   459	        for p in problems:
   460	            print(f"  !! {p}")
   461	        return 2
   462	
   463	    # check_identity already re-hashed the resume file and proved it IS this hash
   464	    resume_sha = want_anchor if want_anchor is not None else initial.get("final_ckpt_sha256")
   465	    # final_ckpt_path (blocking 3(b)): the ACTUAL endpoint file, versioned name
   466	    # and all, so the next chunk resumes the recorded tip by path instead of
   467	    # re-deriving one from a glob that cannot tell `-v1` from its stale twin.
   468	    link = {"job": job, "launch_uuid": jk.get("launch_uuid"),
   469	            "manifest_path": os.path.abspath(args.manifest), "manifest_sha256": man_sha,
   470	            "resume_step": resume_step, "resume_ckpt_sha256": resume_sha,

 succeeded in 0ms:
   241	    return problems, man
   242	
   243	
   244	def check_link_manifest(i, link):
   245	    """Link ``i`` must agree with the MANIFEST it cites (round-5 r2, blocking 4).
   246	
   247	    Continuity alone proves only that the numbers in the registry line up, and
   248	    the registry is a mutable, uncommitted record (round-5 B1 deliberately took
   249	    it out of both drift gates). A single structurally valid forged link —
   250	    resume hash/step copied from the audited anchor, an increasing final step,
   251	    and an arbitrary final hash — therefore passed every ancestry check. So each
   252	    link is now checked against the artefact it names:
   253	
   254	      * its manifest_path exists and its BYTES hash to the recorded manifest_sha256;
   255	      * the manifest's job / launch_uuid are the link's;
   256	      * the manifest's endpoint attestation (step, sha256) is the link's
   257	        final_step / final_ckpt_sha256;
   258	      * the manifest's chunk_end is the link's final_step.
   259	
   260	    THREAT MODEL, honestly stated: this is COOPERATIVE INTEGRITY, not
   261	    cryptographic provenance. It defends against accidents (a stale glob, a
   262	    mis-recorded link, a manifest that drifted after recording) and casual
   263	    tampering (editing the registry by hand). It cannot defend against an author
   264	    who edits registry and manifest together: both are uncommitted files under
   265	    the same user between publication and commit, and nothing here is signed.
   266	    The audit trail is the git history of the committed records, not this check.
   267	    """
   268	    problems = []
   269	    job, uuid = link.get("job"), link.get("launch_uuid")
   270	    man_path, man_sha = link.get("manifest_path"), link.get("manifest_sha256")
   271	    if not man_path:
   272	        return [f"chain link {i} (job {job!r}) cites no manifest_path — a link with no manifest "
   273	                "is a bare assertion, not a record"]
   274	    if not os.path.isfile(man_path):
   275	        return [f"chain link {i} (job {job!r}) cites manifest {man_path}, which does not exist — "
   276	                "the record cannot be checked against the artefact it names"]
   277	    got = sha256_file(man_path)
   278	    if got != man_sha:
   279	        problems.append(f"chain link {i} (job {job!r}): its manifest {man_path} now hashes "
   280	                        f"{got[:12]}, not the recorded {str(man_sha)[:12]} — the manifest changed "
   281	                        "after the link was recorded")
   282	    man = parse_manifest(man_path)
   283	    jkv = kv_line(man, "job")
   284	    if jkv.get("job") != str(job):
   285	        problems.append(f"chain link {i}: its manifest records job {jkv.get('job')!r}, not the "
   286	                        f"link's {job!r}")
   287	    if jkv.get("launch_uuid") != uuid:
   288	        problems.append(f"chain link {i} (job {job!r}): its manifest records launch_uuid "
   289	                        f"{jkv.get('launch_uuid')!r}, not the link's {uuid!r}")
   290	    ckv = kv_line(man, "chunk_end")
   291	    if str(ckv.get("chunk_end")) != str(link.get("final_step")):
   292	        problems.append(f"chain link {i} (job {job!r}): its manifest declares chunk_end "
   293	                        f"{ckv.get('chunk_end')!r}, not the link's final_step "
   294	                        f"{link.get('final_step')!r}")
   295	    akv = kv_line(man, "endpoint_ckpt")
   296	    if "endpoint_ckpt" not in man:
   297	        problems.append(f"chain link {i} (job {job!r}): its manifest carries no endpoint "
   298	                        "attestation, so nothing in it says that job produced this checkpoint")
   299	        return problems
   300	    if str(akv.get("endpoint_step")) != str(link.get("final_step")):
   301	        problems.append(f"chain link {i} (job {job!r}): its manifest attests endpoint_step "
   302	                        f"{akv.get('endpoint_step')!r} != the link's final_step "
   303	                        f"{link.get('final_step')!r}")
   304	    if akv.get("endpoint_sha256") != link.get("final_ckpt_sha256"):
   305	        problems.append(f"chain link {i} (job {job!r}): its manifest attests endpoint_sha256 "
   306	                        f"{str(akv.get('endpoint_sha256'))[:12]} != the link's final_ckpt_sha256 "
   307	                        f"{str(link.get('final_ckpt_sha256'))[:12]} — the record does not match "
   308	                        "the manifest it cites")
   309	    return problems
   310	
   311	
   312	def check_chain_ancestry(reg, chain):
   313	    """EVERY link, from the audited INITIAL anchor to the tip (round-5 B6).
   314	
   315	    Checking only ``chain[-1]`` accepted a crafted registry: append a link whose
   316	    ``final_ckpt_sha256`` is whatever file you want to run, and the tip check
   317	    passes while nothing connects it to the audited 40k anchor. The chain is a
   318	    lineage, so it is verified as one — link 0 must resume the INITIAL anchor,
   319	    every later link must resume its predecessor's endpoint, and the steps must
   320	    strictly increase. Any break names the offending link INDEX.
   321	
   322	    Round-5 r2 (blocking 4): continuity is necessary but not sufficient — every
   323	    link is ALSO checked against the manifest it cites (check_link_manifest),
   324	    because a lone structurally valid forged link satisfied continuity by
   325	    construction.
   326	    """
   327	    problems = []
   328	    prev_sha, prev_step = reg.get("final_ckpt_sha256"), reg.get("final_step")
   329	    prev_label = "the audited INITIAL anchor"
   330	    for i, link in enumerate(chain):
   331	        job = link.get("job")
   332	        r_sha, r_step = link.get("resume_ckpt_sha256"), link.get("resume_step")
   333	        f_sha, f_step = link.get("final_ckpt_sha256"), link.get("final_step")
   334	        problems += check_link_manifest(i, link)
   335	        if not f_sha or f_step is None:
   336	            problems.append(f"chain link {i} (job {job!r}) carries no final_ckpt_sha256/final_step — "
   337	                            "it is not evidence of a checkpoint")
   338	        if prev_sha and r_sha != prev_sha:
   339	            problems.append(f"chain link {i} (job {job!r}) resume_ckpt_sha256 {str(r_sha)[:12]} != "
   340	                            f"{prev_label}'s final_ckpt_sha256 {str(prev_sha)[:12]} — the chain is "
   341	                            f"BROKEN at link {i}: it does not continue what precedes it")
   342	        if prev_step is not None and str(r_step) != str(prev_step):
   343	            problems.append(f"chain link {i} (job {job!r}) resume_step {r_step!r} != {prev_label}'s "
   344	                            f"final_step {prev_step!r} — the chain is BROKEN at link {i}")
   345	        try:
   346	            if prev_step is not None and f_step is not None and int(f_step) <= int(prev_step):
   347	                problems.append(f"chain link {i} (job {job!r}) final_step {f_step!r} does not exceed "
   348	                                f"{prev_label}'s {prev_step!r} — chunk steps only ever increase")
   349	        except (TypeError, ValueError):
   350	            problems.append(f"chain link {i} (job {job!r}) has a non-integer step "
   351	                            f"(resume {r_step!r}, final {f_step!r})")
   352	        prev_sha, prev_step, prev_label = f_sha, f_step, f"chain link {i}"
   353	    return problems
   354	
   355	
   356	def check_chain_binding(manifest_path, registry_path, arm, rung, config_path, ckpt_path,
   357	                        ckpt_sha, expected_step, max_steps, repo_root="."):
   358	    """The CHUNKED-LEG contract: resume the TIP of this arm's recorded chain.
   359	
   360	    A chunk leg is an extension leg that stops early, so it must prove exactly
   361	    the same INITIAL launch identity — but its resume point is no longer the
   362	    audited 40k anchor: it is the endpoint of the previous chunk, recorded as the
   363	    last link of ``arms.<ARM>.chain`` by fa_orbit_record_restart.py.
   364	
   365	    Fail-closed by construction: an EMPTY or ABSENT chain refuses, so chunk N+1
   366	    cannot run until chunk N has been recorded, and every link's final_step /
   367	    final_ckpt_sha256 was written by the recorder from the file on disk. The
   368	    resume file is re-hashed by the caller (``ckpt_sha``); a manifest's claimed
   369	    hash is never trusted here or anywhere else in this chain.
   370	
   371	    Round-5 review B6: the WHOLE ancestry is validated (check_chain_ancestry),
   372	    not merely ``chain[-1]`` — otherwise a crafted registry could append a tip
   373	    with an arbitrary hash and no link back to the audited 40k anchor.
   374	
   375	    Round-5 r2 (blocking 3(d)): nothing here assumes a checkpoint FILENAME shape.
   376	    The resume file arrives as an explicit ``--ckpt`` path and is identified by
   377	    its sha256 and its directory, so a Lightning-versioned endpoint
   378	    (``epoch=E-step=N-v1.ckpt``, written when a retry finds the unversioned name
   379	    taken) is admissible exactly like an unversioned one.
   380	    """
   381	    reg, problems = _load_registry_row(registry_path, arm)
   382	    if problems:
   383	        return problems, {}
   384	    man = parse_manifest(manifest_path)
   385	    problems = check_initial_identity(man, manifest_path, reg, arm, rung, config_path, max_steps)
   386	    if not reg.get("final_ckpt_sha256"):
   387	        problems.append(f"{arm} has no audited final_ckpt_sha256 in the registry — a chain of "
   388	                        "chunks must still descend from the audited INITIAL run")
   389	    chain = reg.get("chain") or []
   390	    if not chain:
   391	        problems.append(f"{arm} has no recorded chain link in {registry_path}: a chunk may only "
   392	                        "resume the tip of a RECORDED chain, so chunk N+1 is inadmissible until "
   393	                        "chunk N is recorded — run fa_orbit_record_restart.py on the previous "
   394	                        "chunk's launcher manifest first")
   395	    else:
   396	        # B6: the FULL ancestry, not just the tip — a tip alone can be crafted.
   397	        problems += check_chain_ancestry(reg, chain)
   398	        tip = chain[-1]
   399	        tip_step, tip_sha = tip.get("final_step"), tip.get("final_ckpt_sha256")
   400	        if str(tip_step) != str(expected_step):
   401	            problems.append(f"EXPECTED_STEP {expected_step} != the last recorded chain link's "
   402	                            f"final_step {tip_step!r} (link job {tip.get('job')!r}) — a chunk "
   403	                            "resumes the TIP of the chain, never an earlier link")
   404	        if not tip_sha:
   405	            problems.append(f"the last recorded chain link (job {tip.get('job')!r}) carries no "
   406	                            "final_ckpt_sha256 — it is not evidence of a checkpoint")
   407	        elif ckpt_sha != tip_sha:
   408	            problems.append(f"resume checkpoint sha256 {ckpt_sha[:12]} != the last recorded chain "
   409	                            f"link's final_ckpt_sha256 {str(tip_sha)[:12]} — this file is not the "
   410	                            "checkpoint that chunk produced")
   411	    problems += check_canonical_dir(man, arm, ckpt_path, repo_root)
   412	    return problems, man
   413	
   414	
   415	def main(argv=None):
   416	    ap = argparse.ArgumentParser(description="exp_11 restart checkpoint preflight")
   417	    ap.add_argument("--ckpt", required=True)
   418	    ap.add_argument("--expected-step", type=int, required=True)
   419	    ap.add_argument("--config", required=True, help="this arm's model config json")
   420	    ap.add_argument("--max-steps", type=int, required=True)

exec
/usr/bin/bash -lc "nl -ba worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py | sed -n '1,240p'" in /n/fs/gatrdp/codespace/FLAC
 succeeded in 0ms:
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

exec
/usr/bin/bash -lc "nl -ba worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_record_restart.py | sed -n '471,680p'" in /n/fs/gatrdp/codespace/FLAC
 succeeded in 0ms:
   471	            "final_step": chunk_end, "final_ckpt_sha256": endpoint_sha,
   472	            "final_ckpt_path": pm.rel_to(args.repo_root, endpoint),
   473	            "recorded_utc": utc_now()}
   474	
   475	    if mine:
   476	        old = chain[at]
   477	        differing = sorted(k for k in link if k != "recorded_utc"
   478	                           and str(old.get(k)) != str(link[k]))
   479	        if differing:
   480	            print("RECORD REFUSED:")
   481	            print(f"  !! {arm} job {job} is ALREADY a chain link whose content differs "
   482	                  f"({', '.join(differing)}) — a recorded link is immutable, and rewriting one "
   483	                  "would re-parent every chunk recorded after it")
   484	            return 2
   485	        print(f"{arm} chunk link job {job} ({resume_step} -> {chunk_end}) is already recorded, "
   486	              "byte-identical — no-op")
   487	        return 0
   488	
   489	    initial.setdefault("chain", []).append(link)
   490	    if not args.dry_run:
   491	        pm.write_atomic(args.registry, reg)
   492	    print(f"recorded {arm} chunk link job {job}: {resume_step} -> {chunk_end}, continuing "
   493	          f"{prev_desc} " + ("(dry run, nothing written)" if args.dry_run else "(published)"))
   494	    print(f"  resume {str(resume_sha)[:12]} -> endpoint {link['final_ckpt_sha256'][:12]} "
   495	          f"({pm.rel_to(args.repo_root, endpoint)})")
   496	    print(f"  {arm} chain is now {len(initial['chain'])} link(s), tip at step {chunk_end}"
   497	          f" of the {budget} budget")
   498	    return 0
   499	
   500	
   501	def main(argv=None):
   502	    ap = argparse.ArgumentParser(description="record an exp_11 RESTART leg")
   503	    ap.add_argument("arm")
   504	    ap.add_argument("manifest")
   505	    ap.add_argument("--registry", default=os.path.join(HERE, "arm_launch_registry.json"))
   506	    ap.add_argument("--launcher", default=os.path.join(HERE, "fa_orbit_train.sbatch"),
   507	                    help="where the Q10 pins are read from")
   508	    ap.add_argument("--producer-dir", default=HERE,
   509	                    help="where the per-leg producer manifests are published")
   510	    # HERE = <repo>/worklog/worklog_<user>/exp_11_fa_orbit_claude
   511	    ap.add_argument("--repo-root", default=os.path.dirname(os.path.dirname(os.path.dirname(HERE))),
   512	                    help="root the manifest's relative paths resolve against")
   513	    ap.add_argument("--extend", action="store_true",
   514	                    help="this leg is already recorded: extend its producer manifest only")
   515	    ap.add_argument("--rehash-all", action="store_true",
   516	                    help="re-hash published checkpoints too (full audit, expensive)")
   517	    ap.add_argument("--dry-run", action="store_true", help="validate and report, publish nothing")
   518	    ap.add_argument("--skip-sacct", action="store_true",
   519	                    help="DOCUMENTED MANUAL RECOVERY ONLY: skip the scheduler's COMPLETED "
   520	                         "confirmation for a chunk link. Off by default; the watchdog never "
   521	                         "passes it. Use only when accounting cannot answer for a leg you have "
   522	                         "audited by hand, and record why in the worklog.")
   523	    ap.add_argument("--sacct-bin", default=os.environ.get("SACCT_BIN", "sacct"),
   524	                    help="the sacct executable (test hook; changes no decision)")
   525	    args = ap.parse_args(argv)
   526	
   527	    arm = args.arm
   528	    pins = read_pins(args.launcher)
   529	    if not pins.get("PINNED_MAXSTEPS"):
   530	        raise SystemExit(f"no PINNED_* values found in {args.launcher}")
   531	
   532	    # One writer at a time, and the lock is the registry's own DIRECTORY: no lock
   533	    # file to leave behind in a tracked tree, and it still covers the tmp+rename.
   534	    store = os.path.dirname(os.path.abspath(args.registry)) or "."
   535	    lock_fd = os.open(store, os.O_RDONLY)
   536	    try:
   537	        fcntl.flock(lock_fd, fcntl.LOCK_EX)
   538	        return record(args, arm, pins)
   539	    finally:
   540	        os.close(lock_fd)
   541	
   542	
   543	def record(args, arm, pins):
   544	    reg = json.load(open(args.registry))
   545	    initial = reg.get("arms", {}).get(arm)
   546	    if initial is None:
   547	        raise SystemExit(f"{arm} has no INITIAL registry entry")
   548	
   549	    raw, man = parse_manifest(args.manifest)
   550	    man_sha = hashlib.sha256(raw).hexdigest()
   551	    # Round 5: a leg that carries a `chunk_end` line stopped at a chunk boundary
   552	    # and is recorded as a CHAIN LINK, not as a whole-budget RESTART row.
   553	    chunk_raw = chunk_end_of(man)
   554	    if chunk_raw is not None:
   555	        return record_chunk(args, arm, reg, initial, man, man_sha, pins, chunk_raw)
   556	    problems, resume_real = check_identity(arm, man, initial, pins, args.repo_root)
   557	    jk, ak, rk = kvs(man, "job"), kvs(man, "arm"), kvs(man, "resume_ckpt")
   558	    job = jk.get("job")
   559	
   560	    legs = reg.setdefault("restarts", {}).setdefault(arm, [])
   561	    same = [l for l in legs if l.get("job") == job or l.get("launch_uuid") == jk.get("launch_uuid")
   562	            or l.get("manifest_sha256") == man_sha]
   563	    if same and not args.extend:
   564	        raise SystemExit(f"{arm} job {job} is ALREADY recorded ({len(same)} matching leg(s)) — "
   565	                         "one leg, one row; use --extend to extend its producer manifest")
   566	    if len(same) > 1:
   567	        problems.append(f"{len(same)} registry rows already claim this leg — the registry is "
   568	                        "inconsistent; fix it before recording")
   569	    if args.extend and not same:
   570	        problems.append(f"--extend given but {arm} job {job} is not recorded yet")
   571	    if problems:
   572	        print("RECORD REFUSED:")
   573	        for p in problems:
   574	            print(f"  !! {p}")
   575	        return 2
   576	
   577	    anchor = initial["final_ckpt_sha256"]
   578	    producer = pm.manifest_name(arm, job)
   579	    row = {
   580	        "manifest_path": args.manifest, "manifest_sha256": man_sha,
   581	        "job": job, "mode": "RESTART", "launch_uuid": jk.get("launch_uuid"),
   582	        "arm": arm, "commit": man.get("commit"), "rung": ak.get("rung"),
   583	        "config_sha256": man.get("config_sha256"), "save_dir": man.get("save_dir"),
   584	        "resume_ckpt": resume_real, "resume_ckpt_sha256": anchor,
   585	        "expected_step": rk.get("expected_step"), "max_steps": ak.get("max_steps"),
   586	        "time_limit": kvs(man, "time_limit").get("time_limit"),
   587	        "producer_manifest": producer, "chains_to": anchor,
   588	        "recorded_at": pm._now(),
   589	    }
   590	    header = {"arm": arm, "job": job, "launch_uuid": jk.get("launch_uuid"), "mode": "RESTART",
   591	              "commit": man.get("commit"), "resume_ckpt_sha256": anchor,
   592	              "expected_step": rk.get("expected_step"), "max_steps": ak.get("max_steps"),
   593	              "save_dir": man.get("save_dir"), "config_sha256": man.get("config_sha256"),
   594	              "chains_to": anchor, "leg_manifest_sha256": man_sha}
   595	
   596	    ckpt_dir = canonical_ckpt_dir(initial["save_dir"], arm, args.repo_root)
   597	    prod_path = os.path.join(args.producer_dir, producer)
   598	    known = (pm.load(prod_path) or {}).get("checkpoints", {})
   599	    found, scan_problems = pm.scan_checkpoints(
   600	        ckpt_dir, int(rk["expected_step"]), int(ak["max_steps"]), known=known,
   601	        rehash_all=args.rehash_all, repo_root=args.repo_root)
   602	    added, kept, pub_problems = pm.publish(prod_path, header, found, dry_run=args.dry_run)
   603	    if scan_problems or pub_problems:
   604	        print("RECORD REFUSED:")
   605	        for p in scan_problems + pub_problems:
   606	            print(f"  !! {p}")
   607	        return 2
   608	
   609	    if args.extend:
   610	        for i, leg in enumerate(legs):
   611	            if leg.get("job") == job:
   612	                legs[i] = {**leg, "producer_manifest": producer}
   613	    else:
   614	        legs.append(row)
   615	    if not args.dry_run:
   616	        pm.write_atomic(args.registry, reg)
   617	    verb = "extended" if args.extend else "recorded"
   618	    print(f"{verb} {arm} RESTART job {job} chaining to {anchor[:12]} "
   619	          f"({'dry run, nothing written' if args.dry_run else 'published'})")
   620	    print(f"  producer manifest {producer}: {len(added)} checkpoint(s) added, "
   621	          f"{len(kept)} already published"
   622	          + (f" (steps {added[0]}..{added[-1]})" if added else ""))
   623	    return 0
   624	
   625	
   626	if __name__ == "__main__":
   627	    sys.exit(main())

2026-08-13T17:24:09.696657Z ERROR codex_models_manager::manager: failed to renew cache TTL: missing field `base_instructions` at line 94 column 5
exec
/usr/bin/bash -lc 'tail -80 worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-13_13-18-38_guardtests.log' in /n/fs/gatrdp/codespace/FLAC
exec
/usr/bin/bash -lc "grep -n -E 'reservation|squeue|sacct|versioned|forged|anchor|lastjob|endpoint|retry|MAXIMUM|chunk.*failed|FAILED.*checkpoint|stale' worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh | sed -n '1,260p'" in /n/fs/gatrdp/codespace/FLAC
 succeeded in 0ms:
PASS  chain refuses a link whose manifest changed after it was recorded  (rc=2)
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
PASS  submitter refuses while another submission holds the arm's lock  (rc=2)
PASS  submitter refuses when squeue itself fails  (rc=2)
PASS  submitter refuses when a leg for the arm is already queued/running  (rc=2)
PASS  the submitter checks the queue INSIDE the reservation lock, before sbatch
PASS  watchdog rejects an off-cadence CHUNK  (rc=2)
PASS  watchdog rejects an off-cadence per-arm CHUNK_VANL  (rc=2)
PASS  watchdog rejects an unknown arm  (rc=2)
PASS  watchdog rejects an unknown key  (rc=2)
PASS  watchdog rejects a chunk above the arm's time-pin maximum  (rc=2)
PASS  watchdog rejects a global CHUNK above the orbit arms' maximum  (rc=2)
PASS  watchdog rejects a chunk above VANL's own (larger) maximum  (rc=2)
PASS  watchdog refuses a second concurrent instance  (rc=2)
PASS  the watchdog lock is free once the holder exits
PASS  watchdog freezes an arm with no audited anchor before submitting  (rc=0)
PASS  watchdog freezes an arm whose anchor file does not hash to the audited sha  (rc=0)
PASS  watchdog freezes an anchor that is not step 40000  (rc=0)
PASS  watchdog freezes a malformed anchor digest  (rc=0)
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

=== guard tests: 147 passed, 5 failed ===

 succeeded in 0ms:
338:# checkpoint IS the audited 40k anchor) while budget and commit may move.
395:$PY - "${TMP}/ext_registry.json" "${TMP}/reg_noanchor.json" "${TMP}/reg_wronganchor.json" \
398:src, noanchor, wronganchor, wrongcommit = sys.argv[1:5]
400:json.dump(r, open(noanchor, "w"), indent=2)
402:json.dump(r, open(wronganchor, "w"), indent=2)
409:expect_cmd "extension refuses an arm with no audited anchor" 2 "no audited final_ckpt_sha256" -- \
410:  ext_with_reg "${TMP}/reg_noanchor.json"
411:# ...and fa_orbit_add_anchor.py is how that arm becomes extendable (fix 6): the
412:# SAME registry that just refused is anchored and then accepted. This is C32's
413:# sequence — audit the 40k checkpoint, write the anchor, then the leg may run.
414:add_anchor() { $PY "${EXPDIR}/fa_orbit_add_anchor.py" C8 --registry "$1" \
416:expect_cmd "add_anchor dry run writes nothing" 0 "dry run, nothing written" -- \
417:  add_anchor "${TMP}/reg_noanchor.json" --dry-run
418:expect_cmd "add_anchor audits and writes the anchor" 0 "anchored C8 at step 40000" -- \
419:  add_anchor "${TMP}/reg_noanchor.json"
420:expect_cmd "the extension preflight accepts the freshly anchored arm" 0 "extension lineage OK" -- \
421:  ext_with_reg "${TMP}/reg_noanchor.json"
422:expect_cmd "add_anchor is idempotent" 0 "already anchored" -- add_anchor "${TMP}/reg_noanchor.json"
423:expect_cmd "add_anchor refuses a manifest that disagrees with the registry" 2 "!= the registered" -- \
424:  add_anchor "${TMP}/reg_wrongcommit.json"
425:expect_cmd "extension refuses a resume that is not the anchor" 2 "audited final checkpoint" -- \
426:  ext_with_reg "${TMP}/reg_wronganchor.json"
435:# INITIAL identity, plus the 42500 endpoint checkpoint a first chunk produces
481:attest = f"endpoint_ckpt {p42} endpoint_step 42500 endpoint_sha256 {sha(p42)}\n"
485:    body + "endpoint_ckpt <none> endpoint_step 42500 endpoint_class 4\n")
487:    body + f"endpoint_ckpt {p42} endpoint_step 42500 endpoint_sha256 {'e' * 64}\n")
493:# The new gates talk to the SCHEDULER: the submitter re-checks `squeue` INSIDE
494:# its per-arm reservation lock, and the recorder requires `sacct` to confirm the
505:cat > "${SHIM}/squeue" <<'EOF'
508:[ -s "${D}/squeue_out" ] && cat "${D}/squeue_out"
509:exit "$(cat "${D}/squeue_rc" 2>/dev/null || echo 0)"
511:printf '#!/usr/bin/env bash\necho COMPLETED\n'          > "${SHIM}/sacct_completed"
512:printf '#!/usr/bin/env bash\necho FAILED\n'             > "${SHIM}/sacct_failed"
513:printf '#!/usr/bin/env bash\nexit 0\n'                  > "${SHIM}/sacct_empty"
514:printf '#!/usr/bin/env bash\necho "sacct: connection refused" >&2\nexit 1\n' > "${SHIM}/sacct_rc1"
515:chmod +x "${SHIM}/sbatch" "${SHIM}/squeue" "${SHIM}"/sacct_*
516:: > "${SHIM}/squeue_out"; echo 0 > "${SHIM}/squeue_rc"
527:# NOT about that gate supplies a sacct that says COMPLETED; the sacct cases below
528:# override it (argparse takes the last --sacct-bin).
531:  --sacct-bin "${SHIM}/sacct_completed" "${@:2}"; }
535:# FAILED leg's manifest plus a pre-existing 42500 checkpoint would mint a link.
538:expect_cmd "recorder refuses a manifest with NO endpoint attestation" 2 "no endpoint attestation" -- \
540:expect_cmd "recorder refuses a leg that attested it produced nothing" 2 "endpoint_ckpt <none>" -- \
542:expect_cmd "recorder refuses an attested sha that is not the file's" 2 "attested endpoint_sha256" -- \
548:  recorder --sacct-bin "${SHIM}/sacct_failed"
549:expect_cmd "recorder refuses a chunk sacct reports nothing about" 2 "sacct reports nothing" -- \
550:  recorder --sacct-bin "${SHIM}/sacct_empty"
551:expect_cmd "recorder refuses when sacct itself fails" 2 "is UNKNOWN" -- \
552:  recorder --sacct-bin "${SHIM}/sacct_rc1"
553:expect_cmd "recorder refuses when sacct cannot be run at all" 2 "could not ask the scheduler" -- \
554:  recorder --sacct-bin "${SHIM}/sacct_does_not_exist"
569:# a retry at a boundary whose unversioned name already exists saves
571:# retry, therefore leaves TWO files at the same step — and the old glob-by-name
572:# flow either bound the STALE bytes to the retry or found two hits and refused
591:stale = os.path.join(ckpt_dir, "epoch=9-step=42500.ckpt"); torch.save(ck(42500, 9, 1.0), stale)
593:# ...and the successful retry Lightning versioned, written SECOND (newest mtime)
594:retry = os.path.join(ckpt_dir, "epoch=9-step=42500-v1.ckpt"); torch.save(ck(42500, 9, 2.0), retry)
622:        body + f"endpoint_ckpt {attested} endpoint_step 42500 endpoint_sha256 {s}\n")
623:w("vchunk_manifest.txt", retry, sha(retry))            # the retry, as the job attested it
627:json.dump({"retry": retry, "retry_sha": sha(retry), "stale_sha": sha(stale)},
629:print("versioned-retry fixture written")
634:  --sacct-bin "${SHIM}/sacct_completed" "${@:2}"; }
635:expect_cmd "recorder refuses an attested endpoint that does not exist" 2 "does not exist" -- \
637:expect_cmd "recorder refuses an attested endpoint outside the canonical directory" 2 "canonical directory" -- \
639:expect_cmd "recorder refuses an attested endpoint named for another boundary" 2 "chunk end step" -- \
641:expect_cmd "recorder records the VERSIONED retry the job attested, beside its stale twin" 0 "recorded C8 chunk link" -- \
648:print("OK" if (os.path.basename(path).endswith("-v1.ckpt") and sha == facts["retry_sha"]
649:               and sha != facts["stale_sha"]) else "NOT-OK")
652:  echo "PASS  the chain link records the retry's path and hash, not the stale twin's"; PASS=$((PASS+1))
654:  echo "FAIL  the chain link does not point at the versioned retry checkpoint"; FAIL=$((FAIL+1))
657:# and directory, so a versioned endpoint resumes exactly like an unversioned one.
658:expect_cmd "chain accepts a VERSIONED endpoint as the resume file" 0 "restart lineage OK" -- \
663:# --skip-sacct is the documented manual-recovery escape hatch: it bypasses the
664:# scheduler gate (here with an sacct that cannot even run) and nothing else.
665:expect_cmd "recorder --skip-sacct proceeds past an unusable sacct" 0 "already recorded" -- \
666:  vrecorder "${TMP}/vchunk_manifest.txt" --sacct-bin "${SHIM}/sacct_rc1" --skip-sacct
667:expect_cmd "recorder without --skip-sacct still refuses that unusable sacct" 2 "is UNKNOWN" -- \
668:  vrecorder "${TMP}/vchunk_manifest.txt" --sacct-bin "${SHIM}/sacct_rc1"
673:# audited anchor at the audited step, its final step increases, and its final
675:# cites attests a different endpoint. Before this round it was accepted.
676:$PY - "${TMP}/chain_registry.json" "${TMP}/chain_reg_forged.json" \
693:expect_cmd "chain refuses a forged link whose manifest attests a different endpoint" 2 "!= the link's final_ckpt_sha256" -- \
694:  chain_pf "${TMP}/chain_reg_forged.json"
714:# anchor. The WHOLE ancestry is checked now, and a break names its link index.
715:$PY - "${TMP}/chain_registry.json" "${TMP}/chain_reg_badanchor.json" \
718:src, badanchor, badancestry = sys.argv[1:4]
719:r = json.load(open(src))                      # link 0 does not resume the INITIAL anchor
721:json.dump(r, open(badanchor, "w"), indent=2)
722:r = json.load(open(src))                      # link 1 does not resume link 0's endpoint
731:expect_cmd "chain refuses a first link that does not descend from the audited anchor" 2 "BROKEN at link 0" -- \
732:  chain_pf "${TMP}/chain_reg_badanchor.json"
751:# The reservation therefore sits in the one sanctioned submitter: an exclusive
764:echo 1 > "${SHIM}/squeue_rc"; : > "${SHIM}/squeue_out"
765:expect_cmd "submitter refuses when squeue itself fails" 2 "the queue state is UNKNOWN" -- \
768:echo 0 > "${SHIM}/squeue_rc"; echo "4242424 RUNNING" > "${SHIM}/squeue_out"
771:echo 0 > "${SHIM}/squeue_rc"; : > "${SHIM}/squeue_out"
772:awk '/flock -n 9/{if (!f) f=NR} /^  LIVE=.*squeue/{q=NR} /^OUT="\$\(sbatch/{s=NR}
774:  && { echo "PASS  the submitter checks the queue INSIDE the reservation lock, before sbatch"; PASS=$((PASS+1)); } \
775:  || { echo "FAIL  the submitter's queue check is not sequenced flock -> squeue -> sbatch"; FAIL=$((FAIL+1)); }
815:# B7: an arm whose registry row carries no audited anchor (VANL, live) must be
818:$PY - "${TMP}/chain_registry.json" "${TMP}/chain_reg_noanchor.json" <<'PY'
824:expect_cmd "watchdog freezes an arm with no audited anchor before submitting" 0 "no audited anchor" -- \
826:    "REGISTRY=${TMP}/chain_reg_noanchor.json" "STATE=${WD_TMP}/state_noanchor" \
827:    "LOG=${WD_TMP}/log_noanchor"
828:# round-5 r2: the startup anchor check is DEEP — the registry's digest must be
829:# well formed, name step 40000, and still be the hash of the one anchor file on
832:# anchor file itself is found and hashed).
844:expect_cmd "watchdog freezes an arm whose anchor file does not hash to the audited sha" 0 "did not verify" -- \
848:expect_cmd "watchdog freezes an anchor that is not step 40000" 0 "not the 40000" -- \
852:expect_cmd "watchdog freezes a malformed anchor digest" 0 "lowercase hex digest" -- \
856:# NOTE: this case runs against the REAL queue (no squeue fake): with a live
901:# holder died (kill -9 equivalent) -> the lock must be free immediately, no stale dir
903:  echo "PASS  lock free after the holder exits (no stale-recovery path needed)"; PASS=$((PASS+1))
907:grep -q 'flock -n 9' "$LAUNCHER" && { echo "PASS  launcher uses flock, not mkdir+stale recovery"; PASS=$((PASS+1)); } \

exec
/usr/bin/bash -lc "grep -n '"'^FAIL'"\\|"'^PASS'"' worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-13_13-18-38_guardtests.log | tail -180" in /n/fs/gatrdp/codespace/FLAC
 succeeded in 0ms:
3:PASS  every launcher pin holds a concrete value
4:PASS  the launcher still refuses a placeholder pin if one returns
5:PASS  SMOKE bypasses the pins  (rc=0)
6:PASS  SMOKE needs a rung  (rc=2)
7:PASS  SMOKE needs a VRAM floor  (rc=2)
8:PASS  SMOKE identity is separate  (rc=0)
10:PASS  missing ARM  (rc=2)
11:PASS  missing EXPECT_SHA  (rc=2)
12:PASS  arm C7 rejected  (rc=2)
13:PASS  arm FA1 rejected  (rc=2)
14:PASS  arm VAN rejected  (rc=2)
15:PASS  arm CKPT4 rejected  (rc=2)
16:PASS  bogus rung rejected  (rc=2)
17:PASS  rung 32x2 accepted  (rc=0)
18:PASS  rung 16x4 accepted  (rc=0)
19:PASS  rung 8x8 accepted  (rc=0)
21:PASS  initial + RESUME_CKPT  (rc=2)
22:PASS  restart w/o ckpt  (rc=2)
23:PASS  restart ckpt missing  (rc=2)
24:PASS  restart foreign ckpt  (rc=2)
25:PASS  restart from the arm's own ckpt dir  (rc=0)
26:PASS  restart from a non-ckpt sibling  (rc=2)
27:PASS  restart MAXSTEPS<=step  (rc=2)
28:PASS  initial refuses an existing run dir  (rc=2)
30:PASS  wrong EXPECT_SHA aborts  (rc=2)
31:PASS  real mode needs sbatch  (rc=2)
32:PASS  moved HEAD, surfaces identical -> gate passes  (rc=2)
33:PASS  moved HEAD, surfaces changed -> aborts  (rc=2)
34:PASS  symbolic EXPECT_SHA refused  (rc=2)
35:PASS  src/tests-only change -> gate passes  (rc=2)
37:PASS  orbit mismatch rejected  (rc=1)
38:PASS  tracked tree unchanged by the suite (snapshot before == after)
40:PASS  class 0 complete  (rc=0)
41:PASS  class 6 world-size absent  (rc=6)
42:PASS  class 6 wrong world-size  (rc=6)
43:PASS  class 3 OOM on nonzero rc  (rc=3)
44:PASS  class 4 missing marker  (rc=4)
45:PASS  class 7 logs differ  (rc=7)
46:PASS  class 7 copy missing  (rc=7)
47:PASS  class 7 tee failed  (rc=7)
48:PASS  raw rc preserved  (rc=9)
51:PASS  preflight accepts a good ckpt  (rc=0)
52:PASS  preflight rejects a step mismatch  (rc=2)
53:PASS  preflight rejects a foreign orbit  (rc=2)
54:PASS  preflight rejects a stripped optimizer  (rc=2)
55:PASS  preflight rejects a missing EMA  (rc=2)
56:PASS  preflight rejects a missing scheduler  (rc=2)
57:PASS  preflight rejects a past-budget ckpt  (rc=2)
58:PASS  preflight rejects an empty file  (rc=2)
59:PASS  preflight rejects a missing file  (rc=2)
60:PASS  preflight binds to the launch manifest  (rc=0)
61:PASS  preflight rejects a rung change  (rc=2)
62:PASS  preflight rejects a manifest without a commit  (rc=2)
63:PASS  preflight rejects a changed commit  (rc=2)
64:PASS  preflight rejects a missing running commit  (rc=2)
66:PASS  a RESTART leg selects the RESTART pin  (rc=0)
67:PASS  an INITIAL launch keeps the INITIAL pin  (rc=0)
68:PASS  the allocation gate names the pin it enforced
69:FAIL  the submitter no longer allocates the C16 RESTART pin
72:PASS  the ORDINARY contract refuses the extension (the bug)  (rc=2)
73:PASS  extension accepts the 40k->100k leg  (rc=0)
74:PASS  extension keeps the ORIGINAL launch commit  (rc=0)
75:PASS  extension refuses a shrinking budget  (rc=2)
76:PASS  extension refuses a foreign resume path  (rc=2)
77:PASS  extension refuses an arm with no audited anchor  (rc=2)
78:PASS  add_anchor dry run writes nothing  (rc=0)
79:PASS  add_anchor audits and writes the anchor  (rc=0)
80:PASS  the extension preflight accepts the freshly anchored arm  (rc=0)
81:PASS  add_anchor is idempotent  (rc=0)
82:PASS  add_anchor refuses a manifest that disagrees with the registry  (rc=2)
83:PASS  extension refuses a resume that is not the anchor  (rc=2)
84:PASS  extension refuses a manifest commit that is not the registered one  (rc=2)
85:PASS  extension refuses a manifest that drifted after registration  (rc=2)
88:PASS  chain refuses an UNRECORDED predecessor (the fail-closed core)  (rc=2)
89:PASS  recorder refuses a manifest with NO endpoint attestation  (rc=2)
90:PASS  recorder refuses a leg that attested it produced nothing  (rc=2)
91:PASS  recorder refuses an attested sha that is not the file's  (rc=2)
92:PASS  recorder refuses a chunk whose job the scheduler reports FAILED  (rc=2)
93:PASS  recorder refuses a chunk sacct reports nothing about  (rc=2)
94:PASS  recorder refuses when sacct itself fails  (rc=2)
95:PASS  recorder refuses when sacct cannot be run at all  (rc=2)
96:PASS  recorder dry run validates but writes nothing  (rc=0)
97:PASS  chain still refuses after a dry run  (rc=2)
98:PASS  recorder records the finished chunk as a chain link  (rc=0)
99:PASS  recorder is idempotent per job  (rc=0)
100:PASS  chain accepts the recorded tip  (rc=0)
101:PASS  chain + valid --chunk-end accepted  (rc=0)
102:PASS  chain refuses a --chunk-end off the checkpoint cadence  (rc=2)
103:PASS  chain refuses a --chunk-end at or below the resume step  (rc=2)
105:PASS  recorder refuses an attested endpoint that does not exist  (rc=2)
106:PASS  recorder refuses an attested endpoint outside the canonical directory  (rc=2)
107:PASS  recorder refuses an attested endpoint named for another boundary  (rc=2)
108:PASS  recorder records the VERSIONED retry the job attested, beside its stale twin  (rc=0)
109:PASS  the chain link records the retry's path and hash, not the stale twin's
110:PASS  chain accepts a VERSIONED endpoint as the resume file  (rc=0)
111:PASS  recorder --skip-sacct proceeds past an unusable sacct  (rc=0)
112:PASS  recorder without --skip-sacct still refuses that unusable sacct  (rc=2)
113:PASS  chain refuses a forged link whose manifest attests a different endpoint  (rc=2)
114:PASS  recorder refuses to rewrite a recorded link  (rc=2)
115:PASS  chain refuses a link whose manifest changed after it was recorded  (rc=2)
116:PASS  chain refuses a resume that is not the recorded tip's checkpoint  (rc=2)
117:PASS  chain refuses a non-tip resume step  (rc=2)
118:PASS  chain refuses a first link that does not descend from the audited anchor  (rc=2)
119:PASS  chain refuses a BROKEN ancestry (link 1 does not continue link 0)  (rc=2)
120:PASS  CHUNK_END on an INITIAL launch refused  (rc=2)
121:PASS  CHUNK_END under SMOKE refused  (rc=2)
122:PASS  submitter: --chunk-end without --resume refused  (rc=2)
123:PASS  submitter: --chunk-end off cadence refused  (rc=2)
124:PASS  submitter: --chunk-end must exceed the resume step  (rc=2)
125:PASS  submitter: --chunk-end above the budget refused  (rc=2)
126:PASS  submitter refuses while another submission holds the arm's lock  (rc=2)
127:PASS  submitter refuses when squeue itself fails  (rc=2)
128:PASS  submitter refuses when a leg for the arm is already queued/running  (rc=2)
129:PASS  the submitter checks the queue INSIDE the reservation lock, before sbatch
130:PASS  watchdog rejects an off-cadence CHUNK  (rc=2)
131:PASS  watchdog rejects an off-cadence per-arm CHUNK_VANL  (rc=2)
132:PASS  watchdog rejects an unknown arm  (rc=2)
133:PASS  watchdog rejects an unknown key  (rc=2)
134:PASS  watchdog rejects a chunk above the arm's time-pin maximum  (rc=2)
135:PASS  watchdog rejects a global CHUNK above the orbit arms' maximum  (rc=2)
136:PASS  watchdog rejects a chunk above VANL's own (larger) maximum  (rc=2)
137:PASS  watchdog refuses a second concurrent instance  (rc=2)
138:PASS  the watchdog lock is free once the holder exits
139:PASS  watchdog freezes an arm with no audited anchor before submitting  (rc=0)
140:PASS  watchdog freezes an arm whose anchor file does not hash to the audited sha  (rc=0)
141:PASS  watchdog freezes an anchor that is not step 40000  (rc=0)
142:PASS  watchdog freezes a malformed anchor digest  (rc=0)
143:PASS  watchdog completes a ONESHOT pass without submitting  (rc=0)
144:PASS  watchdog defaults VANL to a 5000-step chunk (startup overhead)  (rc=0)
146:PASS  every submitter pin holds a concrete value
147:PASS  the submitter still refuses a placeholder pin if one returns
148:PASS  submitter rejects a bad arm  (rc=2)
149:FAIL  submitter derives smoke flags: want rc=0 + '--gres=gpu:l40:4', got rc=2
155:FAIL  submitter derives cpus/mem from the rung: want rc=0 + '--cpus-per-task=36', got rc=2
161:FAIL  submitter derives 8x8 resources: want rc=0 + '--mem=108G', got rc=2
168:PASS  contender refused while the lock is held
169:PASS  lock free after the holder exits (no stale-recovery path needed)
170:PASS  launcher uses flock, not mkdir+stale recovery
171:PASS  no rmdir-based lock release remains
173:PASS  ambient OUTPUT_ROOT rejected under Slurm  (rc=2)
174:PASS  launcher pins the production root literally
175:PASS  submitter exports the fixed root, not ambient state
177:FAIL  dry run publishes no submission manifest: want rc=0 + 'DRYRUN sbatch', got rc=2
183:PASS  a dry run leaves no submission manifest behind
184:PASS  intent manifest is published before the sbatch call
185:PASS  an unrecordable job is cancelled
187:PASS  FIFO no longer uses mktemp -u
188:PASS  FIFO removal is in the exit trap
189:PASS  pip freeze status is checked before hashing
190:PASS  the final record's tee status is captured
191:PASS  the approved wandb entity is exported
192:PASS  the launcher runs the wandb readback and gates on its result

exec
/usr/bin/bash -lc 'git diff -- worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh | tail -800' in /n/fs/gatrdp/codespace/FLAC
 succeeded in 0ms:
diff --git a/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh b/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh
index b487055..dceee52 100755
--- a/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh
+++ b/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh
@@ -430,6 +430,440 @@ printf 'tamper\n' >> "${TMP}/ext_launch_manifest.txt"
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
+# the launcher manifest a finished 40000->42500 chunk leg leaves behind, in four
+# variants: the real one (with the producing job's ENDPOINT ATTESTATION, round-5
+# B6), one with no attestation at all (a pre-B6 launcher, or a leg that never
+# finished), one attesting `<none>` (the leg did not reach the success class),
+# and one whose attested sha is not the file's.
+body = ("job 3999001 host neu001 mode RESTART launch_uuid chunk-uuid-1\n"
+        "arm C8 rung 8x8 micro 8 ngpu 8 max_steps 100000 ckpt_every 2500\n"
+        "commit " + "3" * 40 + "\n"
+        f"model_config {cfg_path}\nconfig_sha256 {cfg_sha}\nvae_sha256 {vae_sha}\n"
+        f"save_dir {save_dir}\n"
+        f"resume_ckpt {p40} expected_step 40000 resume_ckpt_sha256 {sha(p40)}\n"
+        f"time_limit {chunk_pin}\nchunk_end 42500\n")
+attest = f"endpoint_ckpt {p42} endpoint_step 42500 endpoint_sha256 {sha(p42)}\n"
+open(os.path.join(tmp, "chunk_leg_manifest.txt"), "w").write(body + attest)
+open(os.path.join(tmp, "chunk_leg_manifest_noattest.txt"), "w").write(body)
+open(os.path.join(tmp, "chunk_leg_manifest_noneattest.txt"), "w").write(
+    body + "endpoint_ckpt <none> endpoint_step 42500 endpoint_class 4\n")
+open(os.path.join(tmp, "chunk_leg_manifest_badattest.txt"), "w").write(
+    body + f"endpoint_ckpt {p42} endpoint_step 42500 endpoint_sha256 {'e' * 64}\n")
+print("chain fixture written")
+PY
+CH_CKPT42="${CH_CKPT_DIR}/epoch=9-step=42500.ckpt"
+
+# --- fake scheduler binaries (round-5 r2 review) -----------------------------
+# The new gates talk to the SCHEDULER: the submitter re-checks `squeue` INSIDE
+# its per-arm reservation lock, and the recorder requires `sacct` to confirm the
+# producing job COMPLETED. Both are exercised with fakes rather than the live
+# queue, so the cases are deterministic. Every case that could conceivably reach
+# `sbatch` also runs with a FAKE sbatch first on PATH: a guard test must not be
+# able to queue a real job even if its assertion fails.
+SHIM="${TMP}/shim"; mkdir -p "$SHIM"
+cat > "${SHIM}/sbatch" <<'EOF'
+#!/usr/bin/env bash
+echo "FAKE-SBATCH: guard test — nothing was submitted"
+exit 1
+EOF
+cat > "${SHIM}/squeue" <<'EOF'
+#!/usr/bin/env bash
+D="$(dirname "$(readlink -f "$0")")"
+[ -s "${D}/squeue_out" ] && cat "${D}/squeue_out"
+exit "$(cat "${D}/squeue_rc" 2>/dev/null || echo 0)"
+EOF
+printf '#!/usr/bin/env bash\necho COMPLETED\n'          > "${SHIM}/sacct_completed"
+printf '#!/usr/bin/env bash\necho FAILED\n'             > "${SHIM}/sacct_failed"
+printf '#!/usr/bin/env bash\nexit 0\n'                  > "${SHIM}/sacct_empty"
+printf '#!/usr/bin/env bash\necho "sacct: connection refused" >&2\nexit 1\n' > "${SHIM}/sacct_rc1"
+chmod +x "${SHIM}/sbatch" "${SHIM}/squeue" "${SHIM}"/sacct_*
+: > "${SHIM}/squeue_out"; echo 0 > "${SHIM}/squeue_rc"
+
+chain_pf() {  # <registry> [extra preflight args...]
+  local reg="$1"; shift
+  $PY "$PREFLIGHT" --config "${EXPDIR}/FLAC_AR_BF_C8.json" --arm C8 --rung 8x8 \
+    --max-steps 100000 --ckpt "$CH_CKPT42" --expected-step 42500 --commit "$HEAD_SHA" \
+    --launch-manifest "${TMP}/chain_launch_manifest.txt" --chain --launch-registry "$reg" "$@"
+}
+expect_cmd "chain refuses an UNRECORDED predecessor (the fail-closed core)" 2 "no recorded chain link" -- \
+  chain_pf "${TMP}/chain_registry.json"
+# The recorder now demands the SCHEDULER's confirmation, so every case that is
+# NOT about that gate supplies a sacct that says COMPLETED; the sacct cases below
+# override it (argparse takes the last --sacct-bin).
+recorder_with() { $PY "${EXPDIR}/fa_orbit_record_restart.py" C8 "$1" \
+  --registry "${TMP}/chain_registry.json" --launcher "$LAUNCHER" --repo-root "$PWD" \
+  --sacct-bin "${SHIM}/sacct_completed" "${@:2}"; }
+recorder() { recorder_with "${TMP}/chunk_leg_manifest.txt" "$@"; }
+# B6: the manifest is published BEFORE training, so it proves intent, not
+# authorship. Without the producing job's post-classification attestation, a
+# FAILED leg's manifest plus a pre-existing 42500 checkpoint would mint a link.
+# These three run BEFORE the successful record, so `mine` is empty and the
+# attestation gate — not the immutability gate — is what refuses them.
+expect_cmd "recorder refuses a manifest with NO endpoint attestation" 2 "no endpoint attestation" -- \
+  recorder_with "${TMP}/chunk_leg_manifest_noattest.txt"
+expect_cmd "recorder refuses a leg that attested it produced nothing" 2 "endpoint_ckpt <none>" -- \
+  recorder_with "${TMP}/chunk_leg_manifest_noneattest.txt"
+expect_cmd "recorder refuses an attested sha that is not the file's" 2 "attested endpoint_sha256" -- \
+  recorder_with "${TMP}/chunk_leg_manifest_badattest.txt"
+# round-5 r2 blocking 2: the attestation is the JOB's word; the link also needs
+# the SCHEDULER's. Every non-COMPLETED answer — and every answer that is not an
+# answer at all — is a refusal.
+expect_cmd "recorder refuses a chunk whose job the scheduler reports FAILED" 2 "not COMPLETED" -- \
+  recorder --sacct-bin "${SHIM}/sacct_failed"
+expect_cmd "recorder refuses a chunk sacct reports nothing about" 2 "sacct reports nothing" -- \
+  recorder --sacct-bin "${SHIM}/sacct_empty"
+expect_cmd "recorder refuses when sacct itself fails" 2 "is UNKNOWN" -- \
+  recorder --sacct-bin "${SHIM}/sacct_rc1"
+expect_cmd "recorder refuses when sacct cannot be run at all" 2 "could not ask the scheduler" -- \
+  recorder --sacct-bin "${SHIM}/sacct_does_not_exist"
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
+# --- round-5 r2 blocking 3: Lightning's VERSION COUNTER ----------------------
+# train.py builds ModelCheckpoint with enable_version_counter at its default, so
+# a retry at a boundary whose unversioned name already exists saves
+# `epoch=E-step=N-v1.ckpt`. A failed attempt that saved, followed by a successful
+# retry, therefore leaves TWO files at the same step — and the old glob-by-name
+# flow either bound the STALE bytes to the retry or found two hits and refused
+# forever. The recorder now follows the path the producing job ATTESTED.
+V_SAVE="${TMP}/vchain/exp11_C8"
+V_CKPT_DIR="${V_SAVE}/FLAC_exp11_C8/exp11_C8/checkpoints"
+mkdir -p "$V_CKPT_DIR" "${TMP}/vchain/elsewhere"
+$PY - "$TMP" "${EXPDIR}/FLAC_AR_BF_C8.json" "$V_CKPT_DIR" "$V_SAVE" "$LAUNCHER" "$CHUNK_PIN_C8" \
+     "${TMP}/vchain/elsewhere" <<'PY'
+import hashlib, json, os, re, sys, time, torch
+tmp, cfg_path, ckpt_dir, save_dir, launcher, chunk_pin, other = sys.argv[1:8]
+vae_sha = re.search(r'^PINNED_VAE_SHA256="([^"]*)"', open(launcher).read(), re.M).group(1)
+cfg = json.load(open(cfg_path))
+def ck(step, epoch, tag=0.0):
+    return {"global_step": step, "epoch": epoch, "model_config": cfg,
+            "state_dict": {"diffusion.x": torch.full((1,), tag),
+                           "diffusion_ema.x": torch.zeros(1)},
+            "optimizer_states": [{"state": {0: {"step": 1}}, "param_groups": [{"lr": 1e-5}]}],
+            "lr_schedulers": [{"last_epoch": step}]}
+p40 = os.path.join(ckpt_dir, "epoch=8-step=40000.ckpt"); torch.save(ck(40000, 8), p40)
+# the STALE twin the failed attempt left behind, written FIRST (older mtime)...
+stale = os.path.join(ckpt_dir, "epoch=9-step=42500.ckpt"); torch.save(ck(42500, 9, 1.0), stale)
+time.sleep(0.05)
+# ...and the successful retry Lightning versioned, written SECOND (newest mtime)
+retry = os.path.join(ckpt_dir, "epoch=9-step=42500-v1.ckpt"); torch.save(ck(42500, 9, 2.0), retry)
+far = os.path.join(ckpt_dir, "epoch=10-step=45000.ckpt"); torch.save(ck(45000, 10), far)
+outside = os.path.join(other, "epoch=9-step=42500-v1.ckpt"); torch.save(ck(42500, 9, 2.0), outside)
+sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
+cfg_sha = hashlib.sha256(open(cfg_path, "rb").read()).hexdigest()
+man = os.path.join(tmp, "vchain_launch_manifest.txt")
+with open(man, "w") as fh:
+    fh.write("job 3648695 host neu000 mode INITIAL launch_uuid ext-uuid-c8\n")
+    fh.write("arm C8 rung 8x8 micro 8 ngpu 8 max_steps 40000 ckpt_every 2500\n")
+    fh.write("commit " + "2" * 40 + "\n")
+    fh.write(f"model_config {cfg_path}\nconfig_sha256 {cfg_sha}\nvae_sha256 {vae_sha}\n")
+    fh.write(f"save_dir {save_dir}\nwandb_run_id exp11-C8-vchain\n")
+reg = {"arms": {"C8": {
+    "manifest_path": man, "manifest_sha256": sha(man),
+    "job": "3648695", "mode": "INITIAL", "launch_uuid": "ext-uuid-c8",
+    "commit": "2" * 40, "rung": "8x8", "max_steps": "40000",
+    "config_sha256": cfg_sha, "vae_sha256": vae_sha, "save_dir": save_dir,
+    "training_seed": 42, "final_ckpt_sha256": sha(p40), "final_step": 40000}}, "restarts": {}}
+json.dump(reg, open(os.path.join(tmp, "vchain_registry.json"), "w"), indent=2)
+body = ("job 3999011 host neu001 mode RESTART launch_uuid vchunk-uuid-1\n"
+        "arm C8 rung 8x8 micro 8 ngpu 8 max_steps 100000 ckpt_every 2500\n"
+        "commit " + "3" * 40 + "\n"
+        f"model_config {cfg_path}\nconfig_sha256 {cfg_sha}\nvae_sha256 {vae_sha}\n"
+        f"save_dir {save_dir}\n"
+        f"resume_ckpt {p40} expected_step 40000 resume_ckpt_sha256 {sha(p40)}\n"
+        f"time_limit {chunk_pin}\nchunk_end 42500\n")
+def w(name, attested, s):
+    open(os.path.join(tmp, name), "w").write(
+        body + f"endpoint_ckpt {attested} endpoint_step 42500 endpoint_sha256 {s}\n")
+w("vchunk_manifest.txt", retry, sha(retry))            # the retry, as the job attested it
+w("vchunk_manifest_missing.txt", os.path.join(ckpt_dir, "epoch=9-step=42500-v9.ckpt"), "0" * 64)
+w("vchunk_manifest_outside.txt", outside, sha(outside))
+w("vchunk_manifest_wrongname.txt", far, sha(far))
+json.dump({"retry": retry, "retry_sha": sha(retry), "stale_sha": sha(stale)},
+          open(os.path.join(tmp, "vchain_facts.json"), "w"))
+print("versioned-retry fixture written")
+PY
+V_RETRY="${V_CKPT_DIR}/epoch=9-step=42500-v1.ckpt"
+vrecorder() { $PY "${EXPDIR}/fa_orbit_record_restart.py" C8 "$1" \
+  --registry "${TMP}/vchain_registry.json" --launcher "$LAUNCHER" --repo-root "$PWD" \
+  --sacct-bin "${SHIM}/sacct_completed" "${@:2}"; }
+expect_cmd "recorder refuses an attested endpoint that does not exist" 2 "does not exist" -- \
+  vrecorder "${TMP}/vchunk_manifest_missing.txt"
+expect_cmd "recorder refuses an attested endpoint outside the canonical directory" 2 "canonical directory" -- \
+  vrecorder "${TMP}/vchunk_manifest_outside.txt"
+expect_cmd "recorder refuses an attested endpoint named for another boundary" 2 "chunk end step" -- \
+  vrecorder "${TMP}/vchunk_manifest_wrongname.txt"
+expect_cmd "recorder records the VERSIONED retry the job attested, beside its stale twin" 0 "recorded C8 chunk link" -- \
+  vrecorder "${TMP}/vchunk_manifest.txt"
+if $PY - "${TMP}/vchain_registry.json" "${TMP}/vchain_facts.json" <<'PY' | grep -q '^OK$'
+import json, os, sys
+reg, facts = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
+link = reg["arms"]["C8"]["chain"][-1]
+path, sha = link.get("final_ckpt_path", ""), link.get("final_ckpt_sha256")
+print("OK" if (os.path.basename(path).endswith("-v1.ckpt") and sha == facts["retry_sha"]
+               and sha != facts["stale_sha"]) else "NOT-OK")
+PY
+then
+  echo "PASS  the chain link records the retry's path and hash, not the stale twin's"; PASS=$((PASS+1))
+else
+  echo "FAIL  the chain link does not point at the versioned retry checkpoint"; FAIL=$((FAIL+1))
+fi
+# blocking 3(d): the preflight takes an explicit --ckpt and identifies it by hash
+# and directory, so a versioned endpoint resumes exactly like an unversioned one.
+expect_cmd "chain accepts a VERSIONED endpoint as the resume file" 0 "restart lineage OK" -- \
+  $PY "$PREFLIGHT" --config "${EXPDIR}/FLAC_AR_BF_C8.json" --arm C8 --rung 8x8 \
+    --max-steps 100000 --ckpt "$V_RETRY" --expected-step 42500 --commit "$HEAD_SHA" \
+    --launch-manifest "${TMP}/vchain_launch_manifest.txt" --chain \
+    --launch-registry "${TMP}/vchain_registry.json"
+# --skip-sacct is the documented manual-recovery escape hatch: it bypasses the
+# scheduler gate (here with an sacct that cannot even run) and nothing else.
+expect_cmd "recorder --skip-sacct proceeds past an unusable sacct" 0 "already recorded" -- \
+  vrecorder "${TMP}/vchunk_manifest.txt" --sacct-bin "${SHIM}/sacct_rc1" --skip-sacct
+expect_cmd "recorder without --skip-sacct still refuses that unusable sacct" 2 "is UNKNOWN" -- \
+  vrecorder "${TMP}/vchunk_manifest.txt" --sacct-bin "${SHIM}/sacct_rc1"
+
+# --- round-5 r2 blocking 4: a structurally valid FORGED link -----------------
+# Continuity proves the numbers line up; it cannot prove a link was ever earned.
+# This chain has ONE link that satisfies every ancestry rule — it resumes the
+# audited anchor at the audited step, its final step increases, and its final
+# hash is the very file we then present as the resume file — but the manifest it
+# cites attests a different endpoint. Before this round it was accepted.
+$PY - "${TMP}/chain_registry.json" "${TMP}/chain_reg_forged.json" \
+     "${TMP}/chunk_leg_manifest_badattest.txt" <<'PY'
+import hashlib, json, sys
+src, dest, badman = sys.argv[1:4]
+r = json.load(open(src))
+row = r["arms"]["C8"]
+real = row["chain"][-1]                       # the honestly recorded link
+row["chain"] = [{
+    "job": "3999001", "launch_uuid": "chunk-uuid-1",
+    "manifest_path": badman,
+    "manifest_sha256": hashlib.sha256(open(badman, "rb").read()).hexdigest(),
+    "resume_step": row["final_step"], "resume_ckpt_sha256": row["final_ckpt_sha256"],
+    "final_step": 42500, "final_ckpt_sha256": real["final_ckpt_sha256"],
+    "final_ckpt_path": real.get("final_ckpt_path"),
+    "recorded_utc": "2026-08-13T00:00:00+00:00"}]
+json.dump(r, open(dest, "w"), indent=2)
+PY
+expect_cmd "chain refuses a forged link whose manifest attests a different endpoint" 2 "!= the link's final_ckpt_sha256" -- \
+  chain_pf "${TMP}/chain_reg_forged.json"
+
+printf '# drift\n' >> "${TMP}/chunk_leg_manifest.txt"
+expect_cmd "recorder refuses to rewrite a recorded link" 2 "immutable" -- recorder
+expect_cmd "chain refuses a link whose manifest changed after it was recorded" 2 "changed after the link was recorded" -- \
+  chain_pf "${TMP}/chain_registry.json"
+$PY - "${TMP}/chain_registry.json" "${TMP}/chain_reg_badsha.json" <<'PY'
+import json, sys
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
+# B6: validating only the TIP accepted a crafted registry — append a link whose
+# final hash is whatever file you want to run and nothing ties it to the audited
+# anchor. The WHOLE ancestry is checked now, and a break names its link index.
+$PY - "${TMP}/chain_registry.json" "${TMP}/chain_reg_badanchor.json" \
+     "${TMP}/chain_reg_badancestry.json" <<'PY'
+import json, sys
+src, badanchor, badancestry = sys.argv[1:4]
+r = json.load(open(src))                      # link 0 does not resume the INITIAL anchor
+r["arms"]["C8"]["chain"][0]["resume_ckpt_sha256"] = "e" * 64
+json.dump(r, open(badanchor, "w"), indent=2)
+r = json.load(open(src))                      # link 1 does not resume link 0's endpoint
+link0 = r["arms"]["C8"]["chain"][0]
+r["arms"]["C8"]["chain"].append({
+    "job": "3999002", "launch_uuid": "chunk-uuid-2", "manifest_path": "/dev/null",
+    "manifest_sha256": "0" * 64, "resume_step": link0["final_step"],
+    "resume_ckpt_sha256": "e" * 64, "final_step": 45000, "final_ckpt_sha256": "f" * 64,
+    "recorded_utc": "2026-08-13T00:00:00+00:00"})
+json.dump(r, open(badancestry, "w"), indent=2)
+PY
+expect_cmd "chain refuses a first link that does not descend from the audited anchor" 2 "BROKEN at link 0" -- \
+  chain_pf "${TMP}/chain_reg_badanchor.json"
+expect_cmd "chain refuses a BROKEN ancestry (link 1 does not continue link 0)" 2 "BROKEN at link 1" -- \
+  chain_pf "${TMP}/chain_reg_badancestry.json"
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
+# --- round-5 r2 blocking 1: the SUBMISSION RESERVATION lives in the submitter -
+# The watchdog's singleton lock stops a second watchdog, not a human at a shell.
+# The reservation therefore sits in the one sanctioned submitter: an exclusive
+# per-arm flock, and a queue re-check INSIDE it. These cases run with a FAKE
+# sbatch first on PATH, so even a failing assertion cannot queue anything.
+SUB_LOCK="${EXPDIR}/.submit_C8.lock"
+SUB_RELEASE="${TMP}/release_the_submit_lock"
+( flock -n 9 || exit 1; while [ ! -e "$SUB_RELEASE" ]; do sleep 0.2; done ) 9>"$SUB_LOCK" &
+SUB_HOLDER=$!
+sleep 0.5
+expect_cmd "submitter refuses while another submission holds the arm's lock" 2 "already holds" -- \
+  env "PATH=${SHIM}:${PATH}" bash "$SUBMITTER" C8
+: > "$SUB_RELEASE"
+wait "$SUB_HOLDER" 2>/dev/null
+# a queue we cannot read is never read as an empty queue
+echo 1 > "${SHIM}/squeue_rc"; : > "${SHIM}/squeue_out"
+expect_cmd "submitter refuses when squeue itself fails" 2 "the queue state is UNKNOWN" -- \
+  env "PATH=${SHIM}:${PATH}" bash "$SUBMITTER" C8
+# ...and a live job with this arm's name is a duplicate, whoever queued it
+echo 0 > "${SHIM}/squeue_rc"; echo "4242424 RUNNING" > "${SHIM}/squeue_out"
+expect_cmd "submitter refuses when a leg for the arm is already queued/running" 2 "already queued/running" -- \
+  env "PATH=${SHIM}:${PATH}" bash "$SUBMITTER" C8
+echo 0 > "${SHIM}/squeue_rc"; : > "${SHIM}/squeue_out"
+awk '/flock -n 9/{if (!f) f=NR} /^  LIVE=.*squeue/{q=NR} /^OUT="\$\(sbatch/{s=NR}
+     END{exit !(f && q && s && f < q && q < s)}' "$SUBMITTER" \
+  && { echo "PASS  the submitter checks the queue INSIDE the reservation lock, before sbatch"; PASS=$((PASS+1)); } \
+  || { echo "FAIL  the submitter's queue check is not sequenced flock -> squeue -> sbatch"; FAIL=$((FAIL+1)); }
+# watchdog argument safety + the no-checkpoint skip (ONESHOT, everything in TMP)
+WD="${EXPDIR}/fa_orbit_chunk_watchdog.sh"
+WD_TMP="${TMP}/wd"; mkdir -p "${WD_TMP}/outputs"
+expect_cmd "watchdog rejects an off-cadence CHUNK" 2 "not a multiple" -- \
+  bash "$WD" ONESHOT=1 CHUNK=2600
+expect_cmd "watchdog rejects an off-cadence per-arm CHUNK_VANL" 2 "not a multiple" -- \
+  bash "$WD" ONESHOT=1 CHUNK_VANL=2600
+expect_cmd "watchdog rejects an unknown arm" 2 "not a comma-separated list" -- \
+  bash "$WD" ONESHOT=1 ARMS=C4L,BOGUS
+expect_cmd "watchdog rejects an unknown key" 2 "unknown argument" -- \
+  bash "$WD" ONESHOT=1 SBATCH_EXTRA=x
+# round-5 r2 non-blocking: a chunk longer than its arm's wall pin was sized for
+# cannot reach its boundary inside the allocation, so it is refused BY NAME.
+expect_cmd "watchdog rejects a chunk above the arm's time-pin maximum" 2 "PINNED_TIME_LIMIT_CHUNK_C32" -- \
+  bash "$WD" ONESHOT=1 CHUNK_C32=5000
+expect_cmd "watchdog rejects a global CHUNK above the orbit arms' maximum" 2 "PINNED_TIME_LIMIT_CHUNK_C4L" -- \
+  bash "$WD" ONESHOT=1 CHUNK=5000
+# the table is PER ARM, not a blanket 2500: VANL's own maximum is 5000 (the
+# sanctioned default, exercised by the ONESHOT case below), and 7500 is above it.
+expect_cmd "watchdog rejects a chunk above VANL's own (larger) maximum" 2 "PINNED_TIME_LIMIT_CHUNK_VANL" -- \
+  bash "$WD" ONESHOT=1 CHUNK_VANL=7500
+# B2(a): a SECOND watchdog is a double-submission engine (both see "no live job",
+# both submit the same boundary). Rather than race two watchdogs, hold the real
+# lock with a background flock holder and prove the watchdog refuses to start.
+WD_LOCK="${EXPDIR}/.chunk_watchdog.lock"
+WD_RELEASE="${WD_TMP}/release_the_lock"      # the holder exits NORMALLY on this
+( flock -n 9 || exit 1; while [ ! -e "$WD_RELEASE" ]; do sleep 0.2; done ) 9>"$WD_LOCK" &
+WD_HOLDER=$!
+sleep 0.5
+expect_cmd "watchdog refuses a second concurrent instance" 2 "refusing to start a second instance" -- \
+  bash "$WD" ONESHOT=1 DRYRUN=1 ARMS=C8 "OUTPUT_ROOT=${WD_TMP}/outputs" \
+    "REGISTRY=${TMP}/chain_registry.json" "STATE=${WD_TMP}/state" "LOG=${WD_TMP}/log"
+: > "$WD_RELEASE"
+wait "$WD_HOLDER" 2>/dev/null
+if flock -n 9 9>"$WD_LOCK" 2>/dev/null; then
+  echo "PASS  the watchdog lock is free once the holder exits"; PASS=$((PASS+1))
+else
+  echo "FAIL  the watchdog lock is still held after the holder exited"; FAIL=$((FAIL+1))
+fi
+# B7: an arm whose registry row carries no audited anchor (VANL, live) must be
+# FROZEN at startup — not submitted and then refused by the preflight. Anchoring
+# is an operator action, so the watchdog names the tool instead of running it.
+$PY - "${TMP}/chain_registry.json" "${TMP}/chain_reg_noanchor.json" <<'PY'
+import json, sys
+r = json.load(open(sys.argv[1]))
+r["arms"]["C8"].pop("final_ckpt_sha256", None); r["arms"]["C8"].pop("final_step", None)
+json.dump(r, open(sys.argv[2], "w"), indent=2)
+PY
+expect_cmd "watchdog freezes an arm with no audited anchor before submitting" 0 "no audited anchor" -- \
+  bash "$WD" ONESHOT=1 DRYRUN=1 ARMS=C8 "OUTPUT_ROOT=${WD_TMP}/outputs" \
+    "REGISTRY=${TMP}/chain_reg_noanchor.json" "STATE=${WD_TMP}/state_noanchor" \
+    "LOG=${WD_TMP}/log_noanchor"
+# round-5 r2: the startup anchor check is DEEP — the registry's digest must be
+# well formed, name step 40000, and still be the hash of the one anchor file on
+# disk. A registry that says the right shape but the wrong bytes freezes the arm
+# before a single chunk is queued (these fixtures keep the real save_dir, so the
+# anchor file itself is found and hashed).
+$PY - "${TMP}/chain_registry.json" "${TMP}/chain_reg_wrongsha.json" \
+     "${TMP}/chain_reg_wrongstep.json" "${TMP}/chain_reg_shortsha.json" <<'PY'
+import json, sys
+src, wrongsha, wrongstep, shortsha = sys.argv[1:5]
+r = json.load(open(src)); r["arms"]["C8"]["final_ckpt_sha256"] = "a" * 64
+json.dump(r, open(wrongsha, "w"), indent=2)
+r = json.load(open(src)); r["arms"]["C8"]["final_step"] = 42500
+json.dump(r, open(wrongstep, "w"), indent=2)
+r = json.load(open(src)); r["arms"]["C8"]["final_ckpt_sha256"] = "ABC123"
+json.dump(r, open(shortsha, "w"), indent=2)
+PY
+expect_cmd "watchdog freezes an arm whose anchor file does not hash to the audited sha" 0 "did not verify" -- \
+  bash "$WD" ONESHOT=1 DRYRUN=1 ARMS=C8 "OUTPUT_ROOT=${WD_TMP}/outputs" \
+    "REGISTRY=${TMP}/chain_reg_wrongsha.json" "STATE=${WD_TMP}/state_wrongsha" \
+    "LOG=${WD_TMP}/log_wrongsha"
+expect_cmd "watchdog freezes an anchor that is not step 40000" 0 "not the 40000" -- \
+  bash "$WD" ONESHOT=1 DRYRUN=1 ARMS=C8 "OUTPUT_ROOT=${WD_TMP}/outputs" \
+    "REGISTRY=${TMP}/chain_reg_wrongstep.json" "STATE=${WD_TMP}/state_wrongstep" \
+    "LOG=${WD_TMP}/log_wrongstep"
+expect_cmd "watchdog freezes a malformed anchor digest" 0 "lowercase hex digest" -- \
+  bash "$WD" ONESHOT=1 DRYRUN=1 ARMS=C8 "OUTPUT_ROOT=${WD_TMP}/outputs" \
+    "REGISTRY=${TMP}/chain_reg_shortsha.json" "STATE=${WD_TMP}/state_shortsha" \
+    "LOG=${WD_TMP}/log_shortsha"
+# NOTE: this case runs against the REAL queue (no squeue fake): with a live
+# exp11-C8-train job it logs "live job — nothing to do", without one it logs
+# "nothing to resume, skipping" (the fixture OUTPUT_ROOT is empty). Either way
+# a full ONESHOT pass must complete cleanly without submitting anything.
+expect_cmd "watchdog completes a ONESHOT pass without submitting" 0 "ONESHOT: one pass complete" -- \
+  bash "$WD" ONESHOT=1 DRYRUN=1 ARMS=C8 "OUTPUT_ROOT=${WD_TMP}/outputs" \
+    "REGISTRY=${TMP}/chain_registry.json" "STATE=${WD_TMP}/state" "LOG=${WD_TMP}/log"
+expect_cmd "watchdog defaults VANL to a 5000-step chunk (startup overhead)" 0 "chunk 2500 (VANL 5000)" -- \
+  bash "$WD" ONESHOT=1 DRYRUN=1 ARMS=C8 "OUTPUT_ROOT=${WD_TMP}/outputs" \
+    "REGISTRY=${TMP}/chain_registry.json" "STATE=${WD_TMP}/state" "LOG=${WD_TMP}/log"
+
 echo "--- H. the submitter refuses un-pinned submission ---"
 # RETIRED for the same reason as the launcher case above: all pins are concrete,
 # so the submitter's placeholder refusal is unreachable on the real file.

2026-08-13T17:24:40.466598Z ERROR codex_models_manager::manager: failed to renew cache TTL: missing field `base_instructions` at line 94 column 5
exec
/usr/bin/bash -lc "wc -l worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_chunk_kit_review_r3.md && sed -n '1,260p' worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_chunk_kit_review_r3.md" in /n/fs/gatrdp/codespace/FLAC
 succeeded in 0ms:
8591 worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_chunk_kit_review_r3.md
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
session id: 019ffc25-6b41-7f62-b1c2-98d35a53ba0b
--------
user
Round-5-fix-2 re-review (read-only; NO installs, NO file/env modifications; output = review text only). Repo /n/fs/gatrdp/codespace/FLAC, EXPDIR=worklog/worklog_yixun/exp_11_fa_orbit_claude. Your previous re-review (EXPDIR/fa_orbit_codex_chunk_kit_review_r2.md, final verdict section) returned REVISE with 4 blockings: (1) submission race not closed at the submitter, (2) endpoint attestation written before final classification, (3) versioned checkpoint filenames (-v1) invisible to attestation/recorder/watchdog, (4) forged-link laundering via the mutable registry. B5 remains DEFERRED by scope ruling (eval-side verify_chain, next round).

Fixes now in the tree (cumulative diff + watchdog: /tmp/claude-374349/-n-fs-gatrdp-codespace-FLAC/66305eb9-88c7-41df-95da-dde30a426dc2/scratchpad/chunk_kit_r3.diff): (1) submitter-side per-arm flock reservation with in-lock squeue check (nonzero squeue = refuse; live name = refuse; placed before the drift gate so duplicates refuse as duplicates; skipped under DRYRUN/SMOKE), watchdog persists lastjob immediately post-submit, unreachable by-ID branch removed; (2) attestation moved after the settled final classification (positive only on final_rc==0, else '<none>'), recorder requires sacct COMPLETED (SACCT_BIN test hook; --skip-sacct manual-recovery flag, watchdog never passes it); (3) both name shapes globbed with newest-mtime attestation of the actual path, recorder verifies the ATTESTED path (existence, canonical dir, step in name, re-hash), link gains final_ckpt_path, watchdog parses both shapes and resumes the recorded tip's path (unique-on-disk fallback only at the 40k anchor); (4) check_link_manifest for EVERY chain link (manifest bytes hash, job/uuid match, chunk_end==final_step, attestation==link) with an explicit cooperative-integrity threat-model paragraph. Non-blockings: deep startup anchor validation (step==40000, hex shape, unique file, full hash-once, freeze on mismatch), per-arm max-chunk table (C4L/C8/C16/C32:2500, VANL:5000), done_<ARM>=step@target staleness guard, fd-inheritance caveat documented. Guardtests 147 pass / 5 pre-existing dirty-tree fails (verified byte-identical class; new reservation cases run with a fake sbatch on PATH).

Verify the four blockings are resolved, assess the coder's disclosed deviations (reservation before drift gate; PATH-shim for squeue instead of a hook; anchor located via registry save_dir; sacct scoped to chunk links only), and hunt fresh defects strictly within the changed code. The deployment plan after your verdict: commit everything, cancel the five obsolete long legs (3687569-73), start the watchdog (ARMS=all five, CHUNK defaults). Verdict: APPROVED or REVISE. Bear in mind the declared threat model (cooperative integrity, accident-class) — findings requiring cryptographic provenance of uncommitted records are out of scope by standing ruling.
codex
I’ll audit the prior four blockers against the current implementation, then trace the disclosed deviations and changed-code edge cases. I’ll keep the repository and environment untouched.
exec
/usr/bin/bash -lc "pwd && git status --short && rg --files worklog/worklog_yixun/exp_11_fa_orbit_claude | sort | sed -n '1,240p' && sed -n '/Final [Vv]erdict/,"'$p'"' worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_chunk_kit_review_r2.md" in /n/fs/gatrdp/codespace/FLAC
 succeeded in 0ms:
/n/fs/gatrdp/codespace/FLAC
 M worklog/worklog_yixun/exp_11_fa_orbit_claude/arm_launch_registry.json
 M worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py
 M worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_record_restart.py
 M worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submit.sh
 M worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch
 M worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh
 M worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_submit.sh
 M worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_train.sbatch
?? AGENTS.md
?? AcousticRooms
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/.chunk_watchdog.lock
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/.submit_C8.lock
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-12_18-23-10_guardtests.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-13_12-06-48_guardtests.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-13_12-08-41_guardtests.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-13_12-10-57_guardtests.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-13_12-39-58_guardtests.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-13_12-42-29_guardtests.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-13_13-14-34_guardtests.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-13_13-18-38_guardtests.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_chunk_watchdog.sh
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_chunk_kit_review.md
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_chunk_kit_review_r2.md
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_chunk_kit_review_r3.md
?? worklog/worklog_yixun/exp_15_yaw_aug_claude/slurm_train_exp15-smoke-YAWAUG_3685989.out
?? worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-12_11-59-14_readback.log
?? worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-12_11-59-42_readback.log
?? worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-12_12-05-35_guardtests_fullfix.log
?? worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-12_12-35-37_guardtests_f3fix.log
?? worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-12_12-40-22_guardtests_f3fix.log
?? worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-12_17-49-31_YAWAUG_8x8_jid3685989_manifest.txt
?? worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-13_12-19-23_guardtests_chain.log
?? worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-13_12-23-35_guardtests_chain.log
?? worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-13_12-28-53_guardtests_chain.log
?? worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_chain_preflight.py
?? worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_chain_state.py
?? worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_rate_gate.py
?? worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_smoke_registry.json
/usr/bin/bash: line 1: rg: command not found

exec
/usr/bin/bash -lc "find worklog/worklog_yixun/exp_11_fa_orbit_claude -maxdepth 1 -type f -printf '%f\\n' | sort" in /n/fs/gatrdp/codespace/FLAC
 succeeded in 0ms:
.chunk_watchdog.lock
.submit_C8.lock
FLAC_AR_BF_C16.json
FLAC_AR_BF_C32.json
FLAC_AR_BF_C4L.json
FLAC_AR_BF_C8.json
FLAC_AR_BF_FA1.json
FLAC_AR_VANCKPT.json
arm_launch_registry.json
assert_arm_configs_exp11.py
c4_backfill_manifest.json
commits_fa_orbit.md
exp11_mechanism_readout.py
exp11_validate_rows.py
fa_orbit_02_trajectories.html
fa_orbit_2026-08-05_21-45-11_guardtests.log
fa_orbit_2026-08-05_21-45-57_pytest.log
fa_orbit_2026-08-06_14-34-04_pytest_batched.log
fa_orbit_2026-08-06_14-57-53_pytest_reverify.log
fa_orbit_2026-08-06_14-57-53_wrappertest.log
fa_orbit_2026-08-06_15-04-07_jid3646616_equivprobe.log
fa_orbit_2026-08-06_15-12-51_pytest_probefix.log
fa_orbit_2026-08-06_15-15-03_jid3646626_equivprobe.log
fa_orbit_2026-08-06_15-26-05_pytest_b1rootcause.log
fa_orbit_2026-08-06_15-27-42_jid3646634_equivprobe.log
fa_orbit_2026-08-06_15-35-38_pytest_boundadjust.log
fa_orbit_2026-08-06_15-37-36_jid3646653_equivprobe.log
fa_orbit_2026-08-06_16-47-14_C4L_8x8_jid3646734_manifest.txt
fa_orbit_2026-08-06_16-47-14_C4L_8x8_jid3646734_train.log
fa_orbit_2026-08-06_16-58-39_pytest_wandbfix.log
fa_orbit_2026-08-06_18-00-52_C4L_8x8_jid3648568_manifest.txt
fa_orbit_2026-08-06_18-00-52_C4L_8x8_jid3648568_train.log
fa_orbit_2026-08-06_20-36-00_C4L_8x8_jid3648694_manifest.txt
fa_orbit_2026-08-06_20-36-00_C4L_8x8_jid3648694_train.log
fa_orbit_2026-08-06_20-55-01_C8_8x8_jid3648695_manifest.txt
fa_orbit_2026-08-06_20-55-01_C8_8x8_jid3648695_train.log
fa_orbit_2026-08-06_20-55-50_C16_8x8_jid3648696_manifest.txt
fa_orbit_2026-08-06_20-55-50_C16_8x8_jid3648696_train.log
fa_orbit_2026-08-07_00-16-30_C32_8x8_jid3648697_manifest.txt
fa_orbit_2026-08-07_00-16-30_C32_8x8_jid3648697_train.log
fa_orbit_2026-08-07_00-31-51_pytest_screens.log
fa_orbit_2026-08-07_01-06-59_pytest_r4fixes.log
fa_orbit_2026-08-07_01-19-04_C4BACKFILL_S20000_s42_K8_screen.log
fa_orbit_2026-08-07_01-19-04_C4BACKFILL_S30000_s42_K8_screen.log
fa_orbit_2026-08-07_01-34-08_pytest_r4reverify.log
fa_orbit_2026-08-07_01-45-22_pytest_worktree.log
fa_orbit_2026-08-07_01-50-27_pytest_worktree.log
fa_orbit_2026-08-07_02-12-58_screen_guardtests.log
fa_orbit_2026-08-07_02-27-19_screen_guardtests.log
fa_orbit_2026-08-07_02-29-51_screen_guardtests.log
fa_orbit_2026-08-07_02-31-06_screen_guardtests.log
fa_orbit_2026-08-07_02-31-58_screen_guardtests.log
fa_orbit_2026-08-07_02-34-17_pytest_gocheck.log
fa_orbit_2026-08-07_02-49-18_screen_guardtests.log
fa_orbit_2026-08-07_02-50-55_screen_guardtests.log
fa_orbit_2026-08-07_02-52-41_screen_guardtests.log
fa_orbit_2026-08-07_02-53-42_screen_guardtests.log
fa_orbit_2026-08-07_03-02-52_pytest_gocheck2.log
fa_orbit_2026-08-07_03-19-51_screen_guardtests.log
fa_orbit_2026-08-07_03-21-40_screen_guardtests.log
fa_orbit_2026-08-07_03-23-01_screen_guardtests.log
fa_orbit_2026-08-07_03-24-16_screen_guardtests.log
fa_orbit_2026-08-07_03-25-18_pytest_gocheck3.log
fa_orbit_2026-08-07_03-39-44_screen_guardtests.log
fa_orbit_2026-08-07_03-41-03_screen_guardtests.log
fa_orbit_2026-08-07_03-42-23_screen_guardtests.log
fa_orbit_2026-08-07_03-43-45_screen_guardtests.log
fa_orbit_2026-08-07_03-45-10_screen_guardtests.log
fa_orbit_2026-08-07_03-46-13_pytest_gocheck4.log
fa_orbit_2026-08-07_04-04-35_screen_guardtests.log
fa_orbit_2026-08-07_04-06-19_screen_guardtests.log
fa_orbit_2026-08-07_04-07-53_screen_guardtests.log
fa_orbit_2026-08-07_04-09-40_screen_guardtests.log
fa_orbit_2026-08-07_04-10-55_pytest_condgo.log
fa_orbit_2026-08-07_04-20-30_C4L_S2500_s42_K8_screen.log
fa_orbit_2026-08-07_04-20-31_C4BACKFILL_S20000_s42_K8_screen.log
fa_orbit_2026-08-07_04-20-31_C4BACKFILL_S30000_s42_K8_screen.log
fa_orbit_2026-08-07_04-20-31_C4L_S10000_s42_K8_screen.log
fa_orbit_2026-08-07_04-20-31_C4L_S5000_s42_K8_screen.log
fa_orbit_2026-08-07_04-20-31_C4L_S7500_s42_K8_screen.log
fa_orbit_2026-08-07_04-21-31_C16_S5000_s42_K8_screen.log
fa_orbit_2026-08-07_04-21-31_C4L_S12500_s42_K8_screen.log
fa_orbit_2026-08-07_04-21-31_C4L_S15000_s42_K8_screen.log
fa_orbit_2026-08-07_04-21-31_C8_S10000_s42_K8_screen.log
fa_orbit_2026-08-07_04-21-31_C8_S2500_s42_K8_screen.log
fa_orbit_2026-08-07_04-21-31_C8_S5000_s42_K8_screen.log
fa_orbit_2026-08-07_04-21-32_C8_S7500_s42_K8_screen.log
fa_orbit_2026-08-07_15-22-29_C4L_S40000_s42_K8_screen.log
fa_orbit_2026-08-07_15-23-29_C4L_S40000_s42_K1_screen.log
fa_orbit_2026-08-07_15-23-29_C4L_S40000_s43_K8_screen.log
fa_orbit_2026-08-07_15-24-29_C4L_S40000_s43_K1_screen.log
fa_orbit_2026-08-07_15-24-29_C4L_S40000_s44_K8_screen.log
fa_orbit_2026-08-07_15-25-30_C4L_S40000_s44_K1_screen.log
fa_orbit_2026-08-07_15-26-30_C4L_S40000_s45_K8_screen.log
fa_orbit_2026-08-07_15-27-30_C4L_S40000_s45_K1_screen.log
fa_orbit_2026-08-07_15-28-30_C4L_S40000_s46_K8_screen.log
fa_orbit_2026-08-07_15-32-31_C4L_S40000_s46_K1_screen.log
fa_orbit_2026-08-07_15-33-31_C4L_S17500_s42_K8_screen.log
fa_orbit_2026-08-07_15-34-31_C4L_S30000_s42_K8_screen.log
fa_orbit_2026-08-07_15-34-32_C4L_S20000_s42_K8_screen.log
fa_orbit_2026-08-07_15-34-32_C4L_S22500_s42_K8_screen.log
fa_orbit_2026-08-07_15-34-32_C4L_S25000_s42_K8_screen.log
fa_orbit_2026-08-07_15-34-32_C4L_S27500_s42_K8_screen.log
fa_orbit_2026-08-07_15-34-32_C4L_S32500_s42_K8_screen.log
fa_orbit_2026-08-07_15-35-31_C4L_S35000_s42_K8_screen.log
fa_orbit_2026-08-07_15-36-32_C4L_S37500_s42_K8_screen.log
fa_orbit_2026-08-07_15-37-32_C8_S15000_s42_K8_screen.log
fa_orbit_2026-08-07_15-37-33_C8_S12500_s42_K8_screen.log
fa_orbit_2026-08-07_15-38-33_C16_S10000_s42_K8_screen.log
fa_orbit_2026-08-07_15-38-33_C16_S15000_s42_K8_screen.log
fa_orbit_2026-08-07_15-38-33_C32_S5000_s42_K8_screen.log
fa_orbit_2026-08-07_15-38-33_C8_S22500_s42_K8_screen.log
fa_orbit_2026-08-07_15-38-34_C8_S20000_s42_K8_screen.log
fa_orbit_2026-08-07_15-38-34_C8_S25000_s42_K8_screen.log
fa_orbit_2026-08-07_15-38-35_C8_S17500_s42_K8_screen.log
fa_orbit_2026-08-08_00-34-31_C8_S27500_s42_K8_screen.log
fa_orbit_2026-08-08_00-34-32_C8_S30000_s42_K8_screen.log
fa_orbit_2026-08-08_00-35-31_C8_S37500_s42_K8_screen.log
fa_orbit_2026-08-08_00-35-32_C16_S20000_s42_K8_screen.log
fa_orbit_2026-08-08_00-35-32_C32_S10000_s42_K8_screen.log
fa_orbit_2026-08-08_00-35-32_C8_S32500_s42_K8_screen.log
fa_orbit_2026-08-08_00-35-32_C8_S35000_s42_K8_screen.log
fa_orbit_2026-08-08_00-44-26_screen_guardtests.log
fa_orbit_2026-08-08_00-53-56_screen_guardtests.log
fa_orbit_2026-08-08_01-02-01_screen_guardtests.log
fa_orbit_2026-08-08_01-08-34_pytest_cells.log
fa_orbit_2026-08-08_01-14-44_screen_guardtests.log
fa_orbit_2026-08-08_01-39-37_C8_S40000_s42_K1_screen.log
fa_orbit_2026-08-08_01-39-37_C8_S40000_s42_K8_screen.log
fa_orbit_2026-08-08_01-39-37_C8_S40000_s43_K8_screen.log
fa_orbit_2026-08-08_01-40-36_C8_S40000_s43_K1_screen.log
fa_orbit_2026-08-08_01-40-36_C8_S40000_s45_K1_screen.log
fa_orbit_2026-08-08_01-40-37_C8_S40000_s44_K1_screen.log
fa_orbit_2026-08-08_01-40-37_C8_S40000_s44_K8_screen.log
fa_orbit_2026-08-08_01-40-37_C8_S40000_s45_K8_screen.log
fa_orbit_2026-08-08_01-41-36_C8_S40000_s46_K8_screen.log
fa_orbit_2026-08-08_01-41-38_C8_S40000_s46_K1_screen.log
fa_orbit_2026-08-08_01-43-36_C4BACKFILL_S40000_s42_K8_screen.log
fa_orbit_2026-08-08_01-52-38_C8_S40000_s42_K8_screen.log
fa_orbit_2026-08-08_01-53-38_C8_S40000_s42_K1_screen.log
fa_orbit_2026-08-08_01-53-38_C8_S40000_s43_K8_screen.log
fa_orbit_2026-08-08_01-54-38_C8_S40000_s44_K8_screen.log
fa_orbit_2026-08-08_01-54-39_C8_S40000_s43_K1_screen.log
fa_orbit_2026-08-08_01-54-39_C8_S40000_s44_K1_screen.log
fa_orbit_2026-08-08_01-55-39_C8_S40000_s45_K1_screen.log
fa_orbit_2026-08-08_01-55-39_C8_S40000_s45_K8_screen.log
fa_orbit_2026-08-08_01-56-38_C8_S40000_s46_K1_screen.log
fa_orbit_2026-08-08_01-56-39_C8_S40000_s46_K8_screen.log
fa_orbit_2026-08-08_01-59-38_C4BACKFILL_S40000_s42_K8_screen.log
fa_orbit_2026-08-08_01-59-39_C32_S10000_s42_K8_screen.log
fa_orbit_2026-08-08_19-32-42_screen_guardtests.log
fa_orbit_2026-08-08_19-41-23_pytest_fixround.log
fa_orbit_2026-08-08_19-45-41_C8_conf_S40000_s43_K8_jid3657712_screen.log
fa_orbit_2026-08-08_19-45-42_C8_conf_S40000_s42_K1_jid3657711_screen.log
fa_orbit_2026-08-08_19-45-42_C8_conf_S40000_s42_K8_jid3657710_screen.log
fa_orbit_2026-08-08_19-45-42_C8_conf_S40000_s43_K1_jid3657713_screen.log
fa_orbit_2026-08-08_19-45-42_C8_conf_S40000_s44_K1_jid3657715_screen.log
fa_orbit_2026-08-08_19-45-42_C8_conf_S40000_s44_K8_jid3657714_screen.log
fa_orbit_2026-08-08_19-45-42_C8_conf_S40000_s45_K1_jid3657717_screen.log
fa_orbit_2026-08-08_19-45-42_C8_conf_S40000_s45_K8_jid3657716_screen.log
fa_orbit_2026-08-08_19-45-42_C8_conf_S40000_s46_K8_jid3657718_screen.log
fa_orbit_2026-08-08_19-46-41_C16_conf_S40000_s42_K1_jid3657721_screen.log
fa_orbit_2026-08-08_19-46-41_C16_conf_S40000_s45_K1_jid3657727_screen.log
fa_orbit_2026-08-08_19-46-41_C16_conf_S40000_s45_K8_jid3657726_screen.log
fa_orbit_2026-08-08_19-46-41_C8_r3_rot22p5_S40000_s42_K8_jid3657738_screen.log
fa_orbit_2026-08-08_19-46-42_C16_conf_S40000_s42_K8_jid3657720_screen.log
fa_orbit_2026-08-08_19-46-42_C16_conf_S40000_s43_K1_jid3657723_screen.log
fa_orbit_2026-08-08_19-46-42_C16_conf_S40000_s44_K1_jid3657725_screen.log
fa_orbit_2026-08-08_19-46-42_C16_conf_S40000_s44_K8_jid3657724_screen.log
fa_orbit_2026-08-08_19-46-42_C16_conf_S40000_s46_K8_jid3657728_screen.log
fa_orbit_2026-08-08_19-46-42_C4L_r3_rot0_S40000_s42_K8_jid3657730_screen.log
fa_orbit_2026-08-08_19-46-42_C4L_r3_rot11p25_S40000_s42_K8_jid3657732_screen.log
fa_orbit_2026-08-08_19-46-42_C4L_r3_rot45_S40000_s42_K8_jid3657734_screen.log
fa_orbit_2026-08-08_19-46-42_C4L_r3_rot5p625_S40000_s42_K8_jid3657731_screen.log
fa_orbit_2026-08-08_19-46-42_C8_conf_S40000_s46_K1_jid3657719_screen.log
fa_orbit_2026-08-08_19-46-42_C8_r3_rot11p25_S40000_s42_K8_jid3657737_screen.log
fa_orbit_2026-08-08_19-46-42_C8_r3_rot5p625_S40000_s42_K8_jid3657736_screen.log
fa_orbit_2026-08-08_19-46-43_C16_conf_S40000_s43_K8_jid3657722_screen.log
fa_orbit_2026-08-08_19-46-43_C16_conf_S40000_s46_K1_jid3657729_screen.log
fa_orbit_2026-08-08_19-46-43_C4L_r3_rot22p5_S40000_s42_K8_jid3657733_screen.log
fa_orbit_2026-08-08_19-46-43_C8_r3_rot0_S40000_s42_K8_jid3657735_screen.log
fa_orbit_2026-08-08_19-47-41_C16_cross_a32_S40000_s42_K8_jid3657753_screen.log
fa_orbit_2026-08-08_19-47-41_C16_cross_a8_S40000_s42_K8_jid3657752_screen.log
fa_orbit_2026-08-08_19-47-41_C16_r3_rot0_S40000_s42_K8_jid3657740_screen.log
fa_orbit_2026-08-08_19-47-41_C16_r3_rot45_S40000_s42_K8_jid3657744_screen.log
fa_orbit_2026-08-08_19-47-41_C16_r3_rot5p625_S40000_s42_K8_jid3657741_screen.log
fa_orbit_2026-08-08_19-47-41_C32_screen_S15000_s42_K8_jid3657759_screen.log
fa_orbit_2026-08-08_19-47-41_C32_screen_S20000_s42_K8_jid3657760_screen.log
fa_orbit_2026-08-08_19-47-41_C4BACKFILL_cross_a32_S40000_s42_K8_jid3657757_screen.log
fa_orbit_2026-08-08_19-47-41_C4L_cross_a32_S40000_s42_K8_jid3657747_screen.log
fa_orbit_2026-08-08_19-47-41_C4L_cross_a8_S40000_s42_K8_jid3657745_screen.log
fa_orbit_2026-08-08_19-47-41_C8_r3_rot45_S40000_s42_K8_jid3657739_screen.log
fa_orbit_2026-08-08_19-47-42_C16_r3_rot11p25_S40000_s42_K8_jid3657742_screen.log

2026-08-13T17:24:48.842488Z ERROR codex_models_manager::manager: failed to renew cache TTL: missing field `base_instructions` at line 94 column 5
exec
/usr/bin/bash -lc "if [ -f worklog/worklog_yixun/exp_11_fa_orbit_claude/.chunk_watchdog_state ]; then sed -n '1,120p' worklog/worklog_yixun/exp_11_fa_orbit_claude/.chunk_watchdog_state; else echo '<no state>'; fi" in /n/fs/gatrdp/codespace/FLAC
exec
/usr/bin/bash -lc "sed -n '1,180p' worklog/worklog_yixun/exp_11_fa_orbit_claude/arm_launch_registry.json" in /n/fs/gatrdp/codespace/FLAC
 succeeded in 0ms:
<no state>

 succeeded in 0ms:
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
      "training_seed": 42,
      "final_ckpt_sha256": "1095f49330b4e7b9c469d69fdbaab1772586055236964b5e347604e712988507",
      "final_step": 40000,
      "final_ckpt_path": "outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=8-step=40000.ckpt",
      "anchored_at": "2026-08-13T12:30:38-04:00"
    }
  },
  "restarts": {}
}

exec
/usr/bin/bash -lc 'git diff -- worklog/worklog_yixun/exp_11_fa_orbit_claude/arm_launch_registry.json' in /n/fs/gatrdp/codespace/FLAC
 succeeded in 0ms:
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

exec
/usr/bin/bash -lc 'for a in C4L C8 C16 C32 VANL; do d=outputs_FLAC/exp11_${a}/FLAC_exp11_${a}/exp11_${a}/checkpoints; printf '"'%s ' \""'$a"; if [ -d "$d" ]; then find "$d" -maxdepth 1 -type f -name '"'*.ckpt' -printf '%f\\n' | sed -n -e 's/"'^epoch=[0-9][0-9]*-step='"\\([0-9][0-9]*\\)\\.ckpt"'$/'"\\1/p' -e 's/"'^epoch=[0-9][0-9]*-step='"\\([0-9][0-9]*\\)-v[0-9][0-9]*\\.ckpt"'$/'"\\1/p' | sort -n | tail -1; else echo MISSING_DIR; fi; done" in /n/fs/gatrdp/codespace/FLAC
 succeeded in 0ms:
C4L 40000
C8 40000
C16 40000
C32 40000
VANL 40000

exec
/usr/bin/bash -lc 'sha256sum outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=8-step=40000.ckpt' in /n/fs/gatrdp/codespace/FLAC
exec
/usr/bin/bash -lc 'sha256sum outputs_FLAC/exp11_C8/FLAC_exp11_C8/exp11_C8/checkpoints/epoch=8-step=40000.ckpt' in /n/fs/gatrdp/codespace/FLAC
exec
/usr/bin/bash -lc 'sha256sum outputs_FLAC/exp11_C32/FLAC_exp11_C32/exp11_C32/checkpoints/epoch=8-step=40000.ckpt' in /n/fs/gatrdp/codespace/FLAC
exec
/usr/bin/bash -lc 'sha256sum outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=8-step=40000.ckpt' in /n/fs/gatrdp/codespace/FLAC
 succeeded in 1202ms:
1095f49330b4e7b9c469d69fdbaab1772586055236964b5e347604e712988507  outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=8-step=40000.ckpt

exec
/usr/bin/bash -lc "squeue -h -j 3687569,3687570,3687571,3687572,3687573 -o '%i|%j|%T|%M|%l|%R'" in /n/fs/gatrdp/codespace/FLAC
 succeeded in 0ms:
3687572|exp11-C32-train|PENDING|0:00|6-16:00:00|(Priority)
3687573|exp11-VANL-train|PENDING|0:00|19:00:00|(Priority)
3687570|exp11-C8-train|PENDING|0:00|2-03:00:00|(Priority)
3687571|exp11-C16-train|PENDING|0:00|3-17:00:00|(Priority)
3687569|exp11-C4L-train|PENDING|0:00|1-10:00:00|(Priority)

exec
/usr/bin/bash -lc 'sha256sum outputs_FLAC/exp11_C16/FLAC_exp11_C16/exp11_C16/checkpoints/epoch=8-step=40000.ckpt' in /n/fs/gatrdp/codespace/FLAC
exec
/usr/bin/bash -lc 'for a in C4L C8 C16 C32 VANL; do d=outputs_FLAC/exp11_${a}/FLAC_exp11_${a}/exp11_${a}/checkpoints; find "$d" -maxdepth 1 -type f '"\\( -name '*-step=40000.ckpt' -o -name '*-step=40000-v*.ckpt' \\) -printf \""'$a %p'"\\n\"; done" in /n/fs/gatrdp/codespace/FLAC
 succeeded in 0ms:
C4L outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=8-step=40000.ckpt
C8 outputs_FLAC/exp11_C8/FLAC_exp11_C8/exp11_C8/checkpoints/epoch=8-step=40000.ckpt
C16 outputs_FLAC/exp11_C16/FLAC_exp11_C16/exp11_C16/checkpoints/epoch=8-step=40000.ckpt
C32 outputs_FLAC/exp11_C32/FLAC_exp11_C32/exp11_C32/checkpoints/epoch=8-step=40000.ckpt
VANL outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=8-step=40000.ckpt

 succeeded in 55686ms:
ed9d7a869ecded98cab78ecc4cef83e579df6643c8ffe564912a9e8ec5c88de8  outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=8-step=40000.ckpt

 succeeded in 56888ms:
4b58787774b5f15852f2603295d57bbe7a1e2d4e40ba217e548f55d643a52c4c  outputs_FLAC/exp11_C8/FLAC_exp11_C8/exp11_C8/checkpoints/epoch=8-step=40000.ckpt

 succeeded in 56970ms:
1beca05a02ffc7b72884e54671ff2d6836789fadcfe2aa6f0ce27ca2f739c134  outputs_FLAC/exp11_C32/FLAC_exp11_C32/exp11_C32/checkpoints/epoch=8-step=40000.ckpt

 succeeded in 57315ms:
a8a82309fe4a654efa8cbb00c6ffeeaa2fc765c5916b46cf12aaf2ec2b2d01f0  outputs_FLAC/exp11_C16/FLAC_exp11_C16/exp11_C16/checkpoints/epoch=8-step=40000.ckpt

exec
/usr/bin/bash -lc "grep -n -E 'RUNDIR=|SAVEDIR=|LAUNCH_MANIFEST|flock|MODE=' worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch | sed -n '1,180p'" in /n/fs/gatrdp/codespace/FLAC
 succeeded in 0ms:
181:  SAVEDIR="${OUTPUT_ROOT}/exp11_smoke/${ARM}"
208:  NAME="FLAC_exp11_${ARM}"; EXPNAME="exp11_${ARM}"; SAVEDIR="${OUTPUT_ROOT}/exp11_${ARM}"
237:RUNDIR="${SAVEDIR}/${NAME}/${EXPNAME}"
378:LAUNCH_MANIFEST_LINK="${SAVEDIR}/launch_manifest.txt"     # written by the INITIAL launch
380:  MODE="INITIAL"
384:  MODE="RESTART"
487:  echo "DRY RUN complete: gates A–G passed for ARM=${ARM} RUNG=${RUNG} MODE=${MODE} SMOKE=${SMOKE}"
550:# --- K. exclusive run ownership via flock (round-3 B3 residual) --------------
553:# it was. flock has neither: the kernel holds the lock while the fd is open and
559:if ! flock -n 9; then
566:echo "lock acquired: ${LOCKFILE} (flock on fd 9, released on exit)"
574:  [ -n "$LAUNCH_MANIFEST_LINK" ] && PRE_ARGS+=(--launch-manifest "$LAUNCH_MANIFEST_LINK")
603:export WANDB_MODE=online
622:  ORIG_WANDB_RUN_ID="$(awk '/^wandb_run_id /{print $2}' "$LAUNCH_MANIFEST_LINK" 2>/dev/null)"
623:  [ -n "$ORIG_WANDB_RUN_ID" ] || die "RESTART needs the original wandb run id from ${LAUNCH_MANIFEST_LINK} - abort"
713:[ "$MODE" = "INITIAL" ] && { cp "$MANIFEST" "$LAUNCH_MANIFEST_LINK" || die "launch-manifest link write FAILED - abort" 3; }

exec
/usr/bin/bash -lc "grep -R -n --exclude='*.log' --exclude='*review*.md' 'final_ckpt_path\\|chain_tip_path\\|endpoint_ckpt' worklog/worklog_yixun/exp_11_fa_orbit_claude | sed -n '1,300p'" in /n/fs/gatrdp/codespace/FLAC
exec
/usr/bin/bash -lc "grep -n -A18 -B5 'def rel_to' worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_producer_manifest.py" in /n/fs/gatrdp/codespace/FLAC
 succeeded in 0ms:
63-        return None
64-    with open(path) as fh:
65-        return json.load(fh)
66-
67-
68:def rel_to(root, path):
69-    """Repo-relative when possible (portable across the pinned worktrees), else absolute."""
70-    real, root_real = os.path.realpath(path), os.path.realpath(root)
71-    return os.path.relpath(real, root_real) if real.startswith(root_real + os.sep) else real
72-
73-
74-def resolve(root, path):
75-    return path if os.path.isabs(path) else os.path.join(root, path)
76-
77-
78-def scan_checkpoints(ckpt_dir, after_step, max_step, known=None, rehash_all=False, repo_root="."):
79-    """Re-hash the leg's checkpoints from DISK: {str(step): {path, sha256, bytes}}.
80-
81-    Only steps strictly after the resume point and no further than the budget are
82-    the leg's own output -- the resume checkpoint itself belongs to the INITIAL
83-    run and is already anchored in the registry. Steps already published are not
84-    re-read by default (they are immutable evidence, and each is ~724 MB on a
85-    shared filesystem); ``rehash_all`` forces a full audit."""
86-    known = known or {}

exec
/usr/bin/bash -lc "nl -ba worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch | sed -n '120,430p'" in /n/fs/gatrdp/codespace/FLAC
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

2026-08-13T17:26:58.266390Z ERROR codex_models_manager::manager: failed to renew cache TTL: missing field `base_instructions` at line 94 column 5
 succeeded in 15813ms:
grep: worklog/worklog_yixun/exp_11_fa_orbit_claude/__pycache__/fa_orbit_ckpt_preflight.cpython-310.pyc: binary file matches
grep: worklog/worklog_yixun/exp_11_fa_orbit_claude/__pycache__/fa_orbit_record_restart.cpython-310.pyc: binary file matches
grep: worklog/worklog_yixun/exp_11_fa_orbit_claude/__pycache__/fa_orbit_add_anchor.cpython-310.pyc: binary file matches
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py:295:    akv = kv_line(man, "endpoint_ckpt")
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py:296:    if "endpoint_ckpt" not in man:
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:857:    ATTEST_LINE="endpoint_ckpt <none> endpoint_step ${CHUNK_END} endpoint_class ${final_rc}"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:860:    ATTEST_LINE="endpoint_ckpt <none> endpoint_step ${CHUNK_END} endpoint_class ${final_rc}"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:865:      ATTEST_LINE="endpoint_ckpt ${ENDPOINT_CKPT} endpoint_step ${CHUNK_END} endpoint_sha256 ${ENDPOINT_SHA}"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:868:      ATTEST_LINE="endpoint_ckpt <none> endpoint_step ${CHUNK_END} endpoint_class ${final_rc}"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_worklog.md:309:    `final_step` / `final_ckpt_path` atomically under the recorder's lock. Refuses to re-anchor a
worklog/worklog_yixun/exp_11_fa_orbit_claude/arm_launch_registry.json:104:      "final_ckpt_path": "outputs_FLAC/exp11_C32/FLAC_exp11_C32/exp11_C32/checkpoints/epoch=8-step=40000.ckpt",
worklog/worklog_yixun/exp_11_fa_orbit_claude/arm_launch_registry.json:125:      "final_ckpt_path": "outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=8-step=40000.ckpt",
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_chunk_watchdog.sh:57:#      step, and the next chunk resumes the recorded tip's final_ckpt_path (the
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_chunk_watchdog.sh:306:chain_tip_path() {     # <arm> -> the recorded tip's final_ckpt_path ('' if no chain)
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_chunk_watchdog.sh:311:print((chain[-1].get("final_ckpt_path") if chain else "") or "")
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_chunk_watchdog.sh:527:  # final_ckpt_path, never a glob. Lightning's version counter means a boundary
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_chunk_watchdog.sh:543:    resume="$(chain_tip_path "$arm")"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_chunk_watchdog.sh:545:      log "${arm}: the recorded chain tip carries no usable final_ckpt_path (got '${resume:-<none>}') — the record, not the directory listing, names the file a chunk resumes"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_chunk_watchdog.sh:546:      bump_fail "$arm" "recorded tip has no resumable final_ckpt_path"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh:481:attest = f"endpoint_ckpt {p42} endpoint_step 42500 endpoint_sha256 {sha(p42)}\n"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh:485:    body + "endpoint_ckpt <none> endpoint_step 42500 endpoint_class 4\n")
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh:487:    body + f"endpoint_ckpt {p42} endpoint_step 42500 endpoint_sha256 {'e' * 64}\n")
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh:540:expect_cmd "recorder refuses a leg that attested it produced nothing" 2 "endpoint_ckpt <none>" -- \
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh:622:        body + f"endpoint_ckpt {attested} endpoint_step 42500 endpoint_sha256 {s}\n")
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh:647:path, sha = link.get("final_ckpt_path", ""), link.get("final_ckpt_sha256")
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh:689:    "final_ckpt_path": real.get("final_ckpt_path"),
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_add_anchor.py:193:    row["final_ckpt_path"] = pm.rel_to(args.repo_root, ckpt)
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_add_anchor.py:198:    print(f"  {row['final_ckpt_path']} (launch job {row.get('job')}, commit "
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_record_restart.py:49:(`endpoint_ckpt … endpoint_step … endpoint_sha256 …`, appended by
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_record_restart.py:66:    in the link as `final_ckpt_path` for the next chunk to resume from.
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_record_restart.py:275:    attested = kvs(man, "endpoint_ckpt").get("endpoint_ckpt", "")
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_record_restart.py:302:        endpoint_ckpt <path> endpoint_step <n> endpoint_sha256 <sha>
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_record_restart.py:304:    (or `endpoint_ckpt <none> ...` for every non-success class), and this
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_record_restart.py:315:    ak = kvs(man, "endpoint_ckpt")
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_record_restart.py:316:    if "endpoint_ckpt" not in man:
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_record_restart.py:319:                "`endpoint_ckpt <path> endpoint_step <n> endpoint_sha256 <sha>` to this manifest "
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_record_restart.py:322:    attested = ak.get("endpoint_ckpt", "")
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_record_restart.py:324:        return [f"the producing job attested `endpoint_ckpt <none>` (class "
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_record_restart.py:465:    # final_ckpt_path (blocking 3(b)): the ACTUAL endpoint file, versioned name
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_record_restart.py:472:            "final_ckpt_path": pm.rel_to(args.repo_root, endpoint),

exec
/usr/bin/bash -lc "sed -n '4490,4605p' worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_chunk_kit_review_r2.md" in /n/fs/gatrdp/codespace/FLAC
 succeeded in 0ms:
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

exec
/usr/bin/bash -lc "bash -n worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submit.sh && bash -n worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_chunk_watchdog.sh && bash -n worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch && echo 'shell syntax OK'" in /n/fs/gatrdp/codespace/FLAC
 succeeded in 0ms:
shell syntax OK

exec
/usr/bin/bash -lc "/n/fs/gatrdp/envs/flac/bin/python -B -c \"import ast, pathlib; [ast.parse(pathlib.Path(p).read_text(), filename=p) for p in ['worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py','worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_record_restart.py']]; print('python syntax OK')\"" in /n/fs/gatrdp/codespace/FLAC
 succeeded in 71ms:
python syntax OK

exec
/usr/bin/bash -lc 'git diff --check -- worklog/worklog_yixun/exp_11_fa_orbit_claude/arm_launch_registry.json worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_record_restart.py worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submit.sh worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh; rc=$?; echo "diff-check rc=$rc"' in /n/fs/gatrdp/codespace/FLAC
 succeeded in 0ms:
diff-check rc=0

exec
/usr/bin/bash -lc "if env | grep -q '"'^SACCT_BIN='"'; then env | grep '"'^SACCT_BIN='"'; else echo 'SACCT_BIN is unset'; fi" in /n/fs/gatrdp/codespace/FLAC
 succeeded in 0ms:
SACCT_BIN is unset

2026-08-13T17:30:12.629170Z ERROR codex_models_manager::manager: failed to renew cache TTL: missing field `base_instructions` at line 94 column 5
exec
/usr/bin/bash -lc 'ps -u "$(id -u)" -o pid=,etime=,stat=,cmd= | grep '"'[s]ha256sum outputs_FLAC/exp11_' || true" in /n/fs/gatrdp/codespace/FLAC
 succeeded in 0ms:
2026-08-13T17:30:22.505109Z ERROR codex_models_manager::manager: failed to renew cache TTL: missing field `base_instructions` at line 94 column 5
codex
## Verdict

**REVISE.**

The submission, final-classification, and versioned-path fixes are materially correct. Two changed-code defects still block the planned watchdog deployment.

## Blocking findings

1. **The mutable-registry laundering fix validates only the endpoint half of each link.** [check_link_manifest](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py:244) checks manifest bytes, job/UUID, `chunk_end`, and endpoint step/hash, but never compares the link’s `resume_step` and `resume_ckpt_sha256` with the cited manifest’s `expected_step` and `resume_ckpt_sha256`. [check_chain_ancestry](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py:312) then trusts those registry resume fields.

   Consequently, a genuine later-chunk manifest can be re-parented by a registry-only edit: cite a manifest that actually resumed at 42500 and ended at 45000, but make its link claim it resumed the audited 40000 anchor. Its manifest hash, job, UUID, chunk end, and endpoint all match, while ancestry accepts the fabricated resume fields. This is cooperative structural validation—not a request for cryptographic provenance—and remains within the declared threat model.

   Historical link validation also does not recheck the scheduler verdict. Thus a positive manifest from a job that suffered `NODE_FAIL` after appending its attestation could be inserted directly into the mutable registry, bypassing the recorder’s `COMPLETED` gate. Link verification must cover the recorder’s resume/identity contract and the scheduler-success requirement, not only endpoint attribution.

2. **The watchdog cannot execute the same-boundary retry that the `-v1` support is meant to handle.** For a retryable terminal failure, it first increments the failure streak at [fa_orbit_chunk_watchdog.sh:421](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_chunk_watchdog.sh:421). If that failed job left its boundary checkpoint, `S > tip`, so the watchdog attempts to record that failed job at [line 460](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_chunk_watchdog.sh:460). The recorder necessarily refuses because `sacct != COMPLETED`, after which the watchdog increments the same attempt again at [line 491](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_chunk_watchdog.sh:491). With the default threshold of two, one failed job freezes the arm.

   No branch ignores that unrecorded stale checkpoint and resubmits from the recorded tip, so the watchdog never creates the successful `-v1` retry autonomously. The guard test exercises a prepared versioned retry directly through the recorder, not this orchestration path. A retryable failed leg must count once and resubmit the same boundary from the recorded tip; only the successful retry should be reconciled by settled job ID.

## Prior blockers

- Submission reservation: resolved. The submitter holds the per-arm flock across its in-lock `squeue` check and `sbatch`; the watchdog immediately attempts to persist the returned job ID.
- Final classification: resolved. Positive attestation occurs after the final provenance promotion point and only for `final_rc == 0`; recorder-side `sacct COMPLETED` is fail-closed.
- Versioned checkpoint visibility: resolved at the attestation, recorder, link-path, parser, and resume-path layers. The remaining problem is watchdog retry control flow, not filename recognition.
- Forged-link laundering: the former arbitrary-final-hash example is rejected, but the broader blocker remains because resume lineage and scheduler success are not revalidated.

## Disclosed deviations

- Reservation before the drift gate is acceptable and preserves the intended duplicate-first refusal.
- The PATH-based `squeue` shim is acceptable with the fake `sbatch`; it does not itself weaken production behavior.
- Deriving the anchor directory from the registry’s pinned `save_dir` is correct.
- Limiting the new `sacct` requirement to chunk links rather than legacy whole-budget restart rows is acceptable. Limiting it to append-time recording while trusting historical mutable links is not.
- The deep anchor checks, per-arm chunk maxima, target-qualified `done` state, and documented FD-inheritance caveat are sound.

The recorded 147-pass/5-fail result is consistent with the disclosed dirty-tree failures. Shell syntax, Python AST parsing, and `git diff --check` pass. B5 remains deferred as ruled.
tokens used
209,910
## Verdict

**REVISE.**

The submission, final-classification, and versioned-path fixes are materially correct. Two changed-code defects still block the planned watchdog deployment.

## Blocking findings

1. **The mutable-registry laundering fix validates only the endpoint half of each link.** [check_link_manifest](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py:244) checks manifest bytes, job/UUID, `chunk_end`, and endpoint step/hash, but never compares the link’s `resume_step` and `resume_ckpt_sha256` with the cited manifest’s `expected_step` and `resume_ckpt_sha256`. [check_chain_ancestry](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py:312) then trusts those registry resume fields.

   Consequently, a genuine later-chunk manifest can be re-parented by a registry-only edit: cite a manifest that actually resumed at 42500 and ended at 45000, but make its link claim it resumed the audited 40000 anchor. Its manifest hash, job, UUID, chunk end, and endpoint all match, while ancestry accepts the fabricated resume fields. This is cooperative structural validation—not a request for cryptographic provenance—and remains within the declared threat model.

   Historical link validation also does not recheck the scheduler verdict. Thus a positive manifest from a job that suffered `NODE_FAIL` after appending its attestation could be inserted directly into the mutable registry, bypassing the recorder’s `COMPLETED` gate. Link verification must cover the recorder’s resume/identity contract and the scheduler-success requirement, not only endpoint attribution.

2. **The watchdog cannot execute the same-boundary retry that the `-v1` support is meant to handle.** For a retryable terminal failure, it first increments the failure streak at [fa_orbit_chunk_watchdog.sh:421](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_chunk_watchdog.sh:421). If that failed job left its boundary checkpoint, `S > tip`, so the watchdog attempts to record that failed job at [line 460](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_chunk_watchdog.sh:460). The recorder necessarily refuses because `sacct != COMPLETED`, after which the watchdog increments the same attempt again at [line 491](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_chunk_watchdog.sh:491). With the default threshold of two, one failed job freezes the arm.

   No branch ignores that unrecorded stale checkpoint and resubmits from the recorded tip, so the watchdog never creates the successful `-v1` retry autonomously. The guard test exercises a prepared versioned retry directly through the recorder, not this orchestration path. A retryable failed leg must count once and resubmit the same boundary from the recorded tip; only the successful retry should be reconciled by settled job ID.

## Prior blockers

- Submission reservation: resolved. The submitter holds the per-arm flock across its in-lock `squeue` check and `sbatch`; the watchdog immediately attempts to persist the returned job ID.
- Final classification: resolved. Positive attestation occurs after the final provenance promotion point and only for `final_rc == 0`; recorder-side `sacct COMPLETED` is fail-closed.
- Versioned checkpoint visibility: resolved at the attestation, recorder, link-path, parser, and resume-path layers. The remaining problem is watchdog retry control flow, not filename recognition.
- Forged-link laundering: the former arbitrary-final-hash example is rejected, but the broader blocker remains because resume lineage and scheduler success are not revalidated.

## Disclosed deviations

- Reservation before the drift gate is acceptable and preserves the intended duplicate-first refusal.
- The PATH-based `squeue` shim is acceptable with the fake `sbatch`; it does not itself weaken production behavior.
- Deriving the anchor directory from the registry’s pinned `save_dir` is correct.
- Limiting the new `sacct` requirement to chunk links rather than legacy whole-budget restart rows is acceptable. Limiting it to append-time recording while trusting historical mutable links is not.
- The deep anchor checks, per-arm chunk maxima, target-qualified `done` state, and documented FD-inheritance caveat are sound.

The recorded 147-pass/5-fail result is consistent with the disclosed dirty-tree failures. Shell syntax, Python AST parsing, and `git diff --check` pass. B5 remains deferred as ruled.
