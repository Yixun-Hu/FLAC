WARNING: failed to clean up stale arg0 temp dirs: Directory not empty (os error 39)
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
session id: 019ff231-9094-7732-872a-a688e5c624de
--------
user
You are reviewing a small change to a Slurm training launcher in the FLAC repo at /n/fs/gatrdp/codespace/FLAC. DO NOT install anything, DO NOT modify any environment, DO NOT write or delete any file: read-only review; your only output is the review text.

CONTEXT: exp_11 trains 5 arms (C4L/C8/C16/C32/VANL) via worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch, submitted by fa_orbit_submit.sh which records EXPECT_SHA=HEAD at submit. The old start gate required HEAD == EXPECT_SHA at job start. Problem: TWO agent sessions commit to this same checkout; jobs pend for hours-days, so unrelated commits (worklog records, other experiments' files) moved HEAD and killed queued legs dead-on-start (this happened today: 4 legs submitted at 0f0acb2, HEAD now 135cb4b + uncommitted fix).

CHANGE under review (unified diff at /tmp/claude-374349/-n-fs-gatrdp-codespace-FLAC/66305eb9-88c7-41df-95da-dde30a426dc2/scratchpad/content_gate.diff; live files: worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch lines ~194-240 and fa_orbit_train_guardtests.sh section D): the gate now accepts HEAD != EXPECT_SHA IFF 'git diff --name-only EXPECT_SHA HEAD -- <training closure>' is empty, where the closure = train.py, defaults.ini, src/, EXPDIR/FLAC_AR_*.json, the sbatch itself, fa_orbit_launch.sh, the four runtime helpers (fa_orbit_ckpt_preflight.py, assert_arm_configs_exp11.py, fa_orbit_wandb_readback.py, fa_orbit_classify.py), and exp_07/FLAC_AR_BF.json. Unknown EXPECT_SHA or failing git diff = fail-closed abort. arm_launch_registry.json and submission manifests are DELIBERATELY excluded (lineage records, append-only, preflight should read them fresh; gating them would recreate the kill on every record commit). The uncommitted-drift gate (git status --porcelain on the wider surface set) is unchanged. Two new guard cases scan real history for a surfaces-identical and a surfaces-changed commit (SKIP if none in 40).

REVIEW QUESTIONS: (1) is the closure complete — does the sbatch/launch path load anything else at runtime that a commit could change semantically (check fa_orbit_train.sbatch and fa_orbit_launch.sh yourself)? (2) any way a surface change slips through (glob expansion vs git pathspec subtleties, renames, mode changes, symlinks, submodules)? (3) fail-closed holes (set -uo pipefail interactions, command substitution rc, git cat-file on ambiguous short SHA)? (4) are the guard tests sound (the 2>&1 capture in SD, rc expectations)? (5) anything about the dry-run advisory branch. Verdict format: BLOCKING / NON-BLOCKING findings, then APPROVED or REVISE.
codex
I’ll do a read-only inspection of the supplied diff, both launcher scripts, and the relevant guard-test section, then trace every runtime-loaded repository path before giving the requested verdict.
exec
/usr/bin/bash -lc "pwd && git status --short && sed -n '1,280p' /tmp/claude-374349/-n-fs-gatrdp-codespace-FLAC/66305eb9-88c7-41df-95da-dde30a426dc2/scratchpad/content_gate.diff" in /n/fs/gatrdp/codespace/FLAC
 succeeded in 0ms:
/n/fs/gatrdp/codespace/FLAC
 M worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch
 M worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh
 M worklog/worklog_yixun/exp_14_yaw_gen_claude/commits_yaw_gen.md
 M worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_codex_code_r2_review.md
 M worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_screen_guardtests.sh
 M worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_screen_submit.sh
 M worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_submit_grid.sh
 M worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_worklog.md
 M worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_worklog.md
 M worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_yixun_query.md
?? AGENTS.md
?? AcousticRooms
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-09_15-18-49_C32_screen_S2500_s42_K1_jid3662408_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-09_15-18-49_C32_screen_S2500_s42_K8_jid3662407_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-09_15-19-48_C32_screen_S22500_s42_K8_jid3662416_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-09_15-19-49_C32_screen_S12500_s42_K1_jid3662413_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-09_15-19-49_C32_screen_S12500_s42_K8_jid3662412_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-09_15-19-49_C32_screen_S17500_s42_K1_jid3662415_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-09_15-19-49_C32_screen_S17500_s42_K8_jid3662414_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-09_15-19-49_C32_screen_S7500_s42_K1_jid3662411_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-09_15-19-49_C32_screen_S7500_s42_K8_jid3662410_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-09_15-20-48_C32_screen_S22500_s42_K1_jid3662417_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-09_15-20-49_C32_screen_S27500_s42_K1_jid3662419_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-09_15-20-49_C32_screen_S27500_s42_K8_jid3662418_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-09_15-20-49_C32_screen_S30000_s42_K1_jid3662421_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-09_15-20-49_C32_screen_S30000_s42_K8_jid3662420_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-09_17-06-16_screen_guardtests.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-09_17-10-17_guardtests.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-09_17-16-44_guardtests.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-09_17-24-24_screen_guardtests.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-09_20-20-54_C32_screen_S32500_s42_K1_jid3665920_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-09_20-20-54_C32_screen_S32500_s42_K8_jid3665919_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-10_01-27-05_guardtests.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-10_01-31-51_C32_screen_S35000_s42_K8_jid3668010_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-10_01-39-54_C32_screen_S35000_s42_K1_jid3668011_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-10_01-40-02_screen_guardtests.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-10_02-00-01_screen_guardtests.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-10_06-47-34_C32_screen_S37500_s42_K8_jid3668648_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-10_06-48-34_C32_screen_S37500_s42_K1_jid3668649_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-10_11-57-38_C32_screen_S40000_s42_K1_jid3670799_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-10_11-57-38_C32_screen_S40000_s42_K8_jid3670798_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-10_18-20-00_C32_conf_S40000_s42_K8_jid3672838_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-10_18-23-02_C32_conf_S40000_s42_K1_jid3672839_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-10_18-23-02_C32_conf_S40000_s43_K8_jid3672840_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-10_18-25-01_C32_conf_S40000_s43_K1_jid3672841_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-10_18-26-02_C32_conf_S40000_s44_K1_jid3672843_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-10_18-26-02_C32_conf_S40000_s44_K8_jid3672842_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-10_18-29-03_C32_conf_S40000_s45_K8_jid3672844_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-10_18-37-06_C32_conf_S40000_s45_K1_jid3672845_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-10_18-41-06_C32_conf_S40000_s46_K8_jid3672846_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-10_18-41-07_C32_conf_S40000_s46_K1_jid3672847_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-11_00-46-17_VANL_screen_S2500_s42_K8_jid3674679_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-11_00-50-18_VANL_q9_S40000_s42_K8_jid3674658_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-11_00-56-20_VANL_q9_S40000_s42_K1_jid3674659_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-11_00-59-20_VANL_q9_S40000_s44_K8_jid3674662_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-11_00-59-21_VANL_q9_S40000_s43_K1_jid3674661_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-11_00-59-21_VANL_q9_S40000_s43_K8_jid3674660_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-11_01-03-23_VANL_q9_S40000_s44_K1_jid3674663_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-11_01-03-23_VANL_q9_S40000_s45_K8_jid3674664_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-11_01-04-23_VANL_screen_S2500_s42_K1_jid3674680_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-11_01-05-23_VANL_q9_S40000_s45_K1_jid3674665_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-11_01-05-23_VANL_q9_S40000_s46_K8_jid3674666_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-11_01-06-24_VANL_screen_S5000_s42_K8_jid3674681_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-11_01-07-23_C4L_q9_S40000_s43_K1_jid3674671_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-11_01-07-23_C4L_q9_S40000_s43_K8_jid3674670_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-11_01-07-23_C4L_q9_S40000_s44_K1_jid3674673_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-11_01-07-24_C4L_q9_S40000_s42_K8_jid3674668_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-11_01-07-24_C4L_q9_S40000_s44_K8_jid3674672_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-11_01-07-24_VANL_q9_S40000_s46_K1_jid3674667_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-11_01-07-25_C4L_q9_S40000_s42_K1_jid3674669_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-11_01-08-23_VANL_screen_S10000_s42_K8_jid3674685_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-11_01-08-23_VANL_screen_S7500_s42_K1_jid3674684_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-11_01-08-24_C4L_q9_S40000_s45_K1_jid3674675_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-11_01-08-24_C4L_q9_S40000_s45_K8_jid3674674_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-11_01-08-24_C4L_q9_S40000_s46_K1_jid3674677_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-11_01-08-24_VANL_screen_S5000_s42_K1_jid3674682_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-11_01-08-24_VANL_screen_S7500_s42_K8_jid3674683_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-11_01-08-25_C4L_q9_S40000_s46_K8_jid3674676_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-11_01-09-24_VANL_screen_S10000_s42_K1_jid3674686_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-11_01-10-25_VANL_screen_S12500_s42_K1_jid3674688_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-11_01-10-25_VANL_screen_S12500_s42_K8_jid3674687_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-11_01-10-25_VANL_screen_S15000_s42_K8_jid3674689_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-11_01-11-25_VANL_screen_S15000_s42_K1_jid3674690_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-11_01-12-24_VANL_screen_S17500_s42_K1_jid3674692_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-11_01-12-25_VANL_screen_S17500_s42_K8_jid3674691_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-11_01-13-25_VANL_screen_S20000_s42_K8_jid3674693_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-11_01-14-26_VANL_screen_S20000_s42_K1_jid3674694_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-11_01-14-26_VANL_screen_S22500_s42_K1_jid3674696_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-11_01-14-26_VANL_screen_S22500_s42_K8_jid3674695_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-11_01-14-26_VANL_screen_S25000_s42_K8_jid3674697_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-11_01-16-27_VANL_screen_S25000_s42_K1_jid3674698_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-11_01-16-27_VANL_screen_S27500_s42_K1_jid3674700_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-11_01-16-27_VANL_screen_S27500_s42_K8_jid3674699_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-11_01-17-27_VANL_screen_S30000_s42_K1_jid3674702_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-11_01-17-27_VANL_screen_S30000_s42_K8_jid3674701_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-11_01-17-27_VANL_screen_S32500_s42_K1_jid3674704_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-11_01-17-27_VANL_screen_S32500_s42_K8_jid3674703_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-11_01-17-27_VANL_screen_S35000_s42_K1_jid3674706_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-11_01-17-27_VANL_screen_S35000_s42_K8_jid3674705_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-11_01-17-27_VANL_screen_S37500_s42_K8_jid3674707_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-11_01-18-26_VANL_screen_S40000_s42_K8_jid3674709_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-11_01-18-27_VANL_screen_S37500_s42_K1_jid3674708_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-11_01-18-27_VANL_screen_S40000_s42_K1_jid3674710_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-11_14-55-12_guardtests.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-11_14-57-42_guardtests.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_content_gate_review.md
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C16_1786310422371467848-a776b47c.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C16_cross_S40000_s42_K8_jid3680762.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C16_r3_S40000_s42_K8_jid3680748.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C16_r3_S40000_s42_K8_jid3680749.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C16_r3_S40000_s42_K8_jid3680750.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C16_r3_S40000_s42_K8_jid3680751.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C16_r3_S40000_s42_K8_jid3680752.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C32_conf_S40000_s42_K1_jid3672839.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C32_conf_S40000_s42_K8_jid3672838.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C32_conf_S40000_s43_K1_jid3672841.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C32_conf_S40000_s43_K8_jid3672840.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C32_conf_S40000_s44_K1_jid3672843.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C32_conf_S40000_s44_K8_jid3672842.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C32_conf_S40000_s45_K1_jid3672845.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C32_conf_S40000_s45_K8_jid3672844.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C32_conf_S40000_s46_K1_jid3672847.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C32_conf_S40000_s46_K8_jid3672846.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C32_cross_S40000_s42_K8_jid3680763.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C32_r3_S40000_s42_K8_jid3680753.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C32_r3_S40000_s42_K8_jid3680754.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C32_r3_S40000_s42_K8_jid3680755.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C32_r3_S40000_s42_K8_jid3680756.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C32_r3_S40000_s42_K8_jid3680757.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C32_screen_S12500_s42_K1_jid3662413.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C32_screen_S12500_s42_K8_jid3662412.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C32_screen_S17500_s42_K1_jid3662415.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C32_screen_S17500_s42_K8_jid3662414.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C32_screen_S22500_s42_K1_jid3662417.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C32_screen_S22500_s42_K8_jid3662416.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C32_screen_S2500_s42_K1_jid3662408.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C32_screen_S2500_s42_K8_jid3662407.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C32_screen_S27500_s42_K1_jid3662419.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C32_screen_S27500_s42_K8_jid3662418.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C32_screen_S30000_s42_K1_jid3662421.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C32_screen_S30000_s42_K8_jid3662420.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C32_screen_S32500_s42_K1_jid3665920.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C32_screen_S32500_s42_K8_jid3665919.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C32_screen_S35000_s42_K1_jid3668011.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C32_screen_S35000_s42_K8_jid3668010.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C32_screen_S37500_s42_K1_jid3668649.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C32_screen_S37500_s42_K8_jid3668648.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C32_screen_S40000_s42_K1_jid3670799.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C32_screen_S40000_s42_K8_jid3670798.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C32_screen_S7500_s42_K1_jid3662411.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C32_screen_S7500_s42_K8_jid3662410.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C4BACKFILL_cross_S40000_s42_K8_jid3680764.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C4BACKFILL_cross_S40000_s42_K8_jid3680765.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C4BACKFILL_cross_S40000_s42_K8_jid3680766.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C4L_1786310422143759413-7d512809.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C4L_cross_S40000_s42_K8_jid3680758.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C4L_cross_S40000_s42_K8_jid3680759.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C4L_cross_S40000_s42_K8_jid3680760.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C4L_q9_S40000_s42_K1_jid3674669.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C4L_q9_S40000_s42_K8_jid3674668.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C4L_q9_S40000_s43_K1_jid3674671.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C4L_q9_S40000_s43_K8_jid3674670.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C4L_q9_S40000_s44_K1_jid3674673.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C4L_q9_S40000_s44_K8_jid3674672.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C4L_q9_S40000_s45_K1_jid3674675.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C4L_q9_S40000_s45_K8_jid3674674.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C4L_q9_S40000_s46_K1_jid3674677.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C4L_q9_S40000_s46_K8_jid3674676.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C4L_r3_S40000_s42_K8_jid3680738.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C4L_r3_S40000_s42_K8_jid3680739.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C4L_r3_S40000_s42_K8_jid3680740.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C4L_r3_S40000_s42_K8_jid3680741.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C4L_r3_S40000_s42_K8_jid3680742.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C8_1786310422260085470-2e58ce21.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C8_cross_S40000_s42_K8_jid3680761.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C8_r3_S40000_s42_K8_jid3680743.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C8_r3_S40000_s42_K8_jid3680744.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C8_r3_S40000_s42_K8_jid3680745.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C8_r3_S40000_s42_K8_jid3680746.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C8_r3_S40000_s42_K8_jid3680747.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_VANL_1786473966640260607-09fab791.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_VANL_q9_S40000_s42_K1_jid3674659.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_VANL_q9_S40000_s42_K8_jid3674658.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_VANL_q9_S40000_s43_K1_jid3674661.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_VANL_q9_S40000_s43_K8_jid3674660.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_VANL_q9_S40000_s44_K1_jid3674663.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_VANL_q9_S40000_s44_K8_jid3674662.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_VANL_q9_S40000_s45_K1_jid3674665.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_VANL_q9_S40000_s45_K8_jid3674664.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_VANL_q9_S40000_s46_K1_jid3674667.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_VANL_q9_S40000_s46_K8_jid3674666.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_VANL_screen_S10000_s42_K1_jid3662406.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_VANL_screen_S10000_s42_K1_jid3662812.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_VANL_screen_S10000_s42_K1_jid3674686.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_VANL_screen_S10000_s42_K8_jid3662405.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_VANL_screen_S10000_s42_K8_jid3662811.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_VANL_screen_S10000_s42_K8_jid3674685.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_VANL_screen_S12500_s42_K1_jid3662814.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_VANL_screen_S12500_s42_K1_jid3674688.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_VANL_screen_S12500_s42_K8_jid3662813.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_VANL_screen_S12500_s42_K8_jid3674687.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_VANL_screen_S15000_s42_K1_jid3662816.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_VANL_screen_S15000_s42_K1_jid3674690.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_VANL_screen_S15000_s42_K8_jid3662815.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_VANL_screen_S15000_s42_K8_jid3674689.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_VANL_screen_S17500_s42_K1_jid3662818.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_VANL_screen_S17500_s42_K1_jid3674692.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_VANL_screen_S17500_s42_K8_jid3662817.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_VANL_screen_S17500_s42_K8_jid3674691.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_VANL_screen_S20000_s42_K1_jid3674694.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_VANL_screen_S20000_s42_K8_jid3674693.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_VANL_screen_S22500_s42_K1_jid3674696.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_VANL_screen_S22500_s42_K8_jid3674695.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_VANL_screen_S25000_s42_K1_jid3674698.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_VANL_screen_S25000_s42_K8_jid3674697.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_VANL_screen_S2500_s42_K1_jid3662400.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_VANL_screen_S2500_s42_K1_jid3662806.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_VANL_screen_S2500_s42_K1_jid3674680.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_VANL_screen_S2500_s42_K8_jid3662399.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_VANL_screen_S2500_s42_K8_jid3662805.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_VANL_screen_S2500_s42_K8_jid3674679.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_VANL_screen_S27500_s42_K1_jid3674700.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_VANL_screen_S27500_s42_K8_jid3674699.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_VANL_screen_S30000_s42_K1_jid3674702.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_VANL_screen_S30000_s42_K8_jid3674701.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_VANL_screen_S32500_s42_K1_jid3674704.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_VANL_screen_S32500_s42_K8_jid3674703.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_VANL_screen_S35000_s42_K1_jid3674706.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_VANL_screen_S35000_s42_K8_jid3674705.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_VANL_screen_S37500_s42_K1_jid3674708.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_VANL_screen_S37500_s42_K8_jid3674707.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_VANL_screen_S40000_s42_K1_jid3674710.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_VANL_screen_S40000_s42_K8_jid3674709.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_VANL_screen_S5000_s42_K1_jid3662402.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_VANL_screen_S5000_s42_K1_jid3662808.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_VANL_screen_S5000_s42_K1_jid3674682.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_VANL_screen_S5000_s42_K8_jid3662401.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_VANL_screen_S5000_s42_K8_jid3662807.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_VANL_screen_S5000_s42_K8_jid3674681.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_VANL_screen_S7500_s42_K1_jid3662404.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_VANL_screen_S7500_s42_K1_jid3662810.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_VANL_screen_S7500_s42_K1_jid3674684.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_VANL_screen_S7500_s42_K8_jid3662403.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_VANL_screen_S7500_s42_K8_jid3662809.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_VANL_screen_S7500_s42_K8_jid3674683.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-C32-conf-40000-s42-K1_3672839.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-C32-conf-40000-s42-K8_3672838.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-C32-conf-40000-s43-K1_3672841.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-C32-conf-40000-s43-K8_3672840.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-C32-conf-40000-s44-K1_3672843.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-C32-conf-40000-s44-K8_3672842.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-C32-conf-40000-s45-K1_3672845.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-C32-conf-40000-s45-K8_3672844.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-C32-conf-40000-s46-K1_3672847.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-C32-conf-40000-s46-K8_3672846.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-C32-screen-12500-s42-K1_3662413.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-C32-screen-12500-s42-K8_3662412.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-C32-screen-17500-s42-K1_3662415.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-C32-screen-17500-s42-K8_3662414.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-C32-screen-22500-s42-K1_3662417.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-C32-screen-22500-s42-K8_3662416.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-C32-screen-2500-s42-K1_3662408.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-C32-screen-2500-s42-K8_3662407.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-C32-screen-27500-s42-K1_3662419.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-C32-screen-27500-s42-K8_3662418.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-C32-screen-30000-s42-K1_3662421.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-C32-screen-30000-s42-K8_3662420.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-C32-screen-32500-s42-K1_3665920.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-C32-screen-32500-s42-K8_3665919.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-C32-screen-35000-s42-K1_3668011.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-C32-screen-35000-s42-K8_3668010.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-C32-screen-37500-s42-K1_3668649.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-C32-screen-37500-s42-K8_3668648.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-C32-screen-40000-s42-K1_3670799.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-C32-screen-40000-s42-K8_3670798.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-C32-screen-7500-s42-K1_3662411.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-C32-screen-7500-s42-K8_3662410.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-C4L-q9-40000-s42-K1_3674669.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-C4L-q9-40000-s42-K8_3674668.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-C4L-q9-40000-s43-K1_3674671.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-C4L-q9-40000-s43-K8_3674670.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-C4L-q9-40000-s44-K1_3674673.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-C4L-q9-40000-s44-K8_3674672.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-C4L-q9-40000-s45-K1_3674675.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-C4L-q9-40000-s45-K8_3674674.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-C4L-q9-40000-s46-K1_3674677.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-C4L-q9-40000-s46-K8_3674676.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-VANL-q9-40000-s42-K1_3674659.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-VANL-q9-40000-s42-K8_3674658.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-VANL-q9-40000-s43-K1_3674661.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-VANL-q9-40000-s43-K8_3674660.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-VANL-q9-40000-s44-K1_3674663.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-VANL-q9-40000-s44-K8_3674662.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-VANL-q9-40000-s45-K1_3674665.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-VANL-q9-40000-s45-K8_3674664.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-VANL-q9-40000-s46-K1_3674667.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-VANL-q9-40000-s46-K8_3674666.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-VANL-screen-10000-s42-K1_3662406.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-VANL-screen-10000-s42-K1_3674686.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-VANL-screen-10000-s42-K8_3662405.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-VANL-screen-10000-s42-K8_3674685.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-VANL-screen-12500-s42-K1_3674688.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-VANL-screen-12500-s42-K8_3674687.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-VANL-screen-15000-s42-K1_3674690.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-VANL-screen-15000-s42-K8_3674689.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-VANL-screen-17500-s42-K1_3674692.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-VANL-screen-17500-s42-K8_3674691.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-VANL-screen-20000-s42-K1_3674694.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-VANL-screen-20000-s42-K8_3674693.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-VANL-screen-22500-s42-K1_3674696.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-VANL-screen-22500-s42-K8_3674695.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-VANL-screen-2500-s42-K1_3662400.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-VANL-screen-2500-s42-K1_3662806.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-VANL-screen-2500-s42-K1_3674680.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-VANL-screen-2500-s42-K8_3662399.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-VANL-screen-2500-s42-K8_3662805.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-VANL-screen-2500-s42-K8_3674679.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-VANL-screen-25000-s42-K1_3674698.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-VANL-screen-25000-s42-K8_3674697.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-VANL-screen-27500-s42-K1_3674700.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-VANL-screen-27500-s42-K8_3674699.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-VANL-screen-30000-s42-K1_3674702.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-VANL-screen-30000-s42-K8_3674701.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-VANL-screen-32500-s42-K1_3674704.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-VANL-screen-32500-s42-K8_3674703.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-VANL-screen-35000-s42-K1_3674706.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-VANL-screen-35000-s42-K8_3674705.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-VANL-screen-37500-s42-K1_3674708.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-VANL-screen-37500-s42-K8_3674707.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-VANL-screen-40000-s42-K1_3674710.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-VANL-screen-40000-s42-K8_3674709.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-VANL-screen-5000-s42-K1_3662402.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-VANL-screen-5000-s42-K1_3662808.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-VANL-screen-5000-s42-K1_3674682.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-VANL-screen-5000-s42-K8_3662401.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-VANL-screen-5000-s42-K8_3662807.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-VANL-screen-5000-s42-K8_3674681.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-VANL-screen-7500-s42-K1_3662404.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-VANL-screen-7500-s42-K1_3674684.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-VANL-screen-7500-s42-K8_3662403.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-VANL-screen-7500-s42-K8_3662809.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-VANL-screen-7500-s42-K8_3674683.out
?? worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_2026-08-11_13-52-58_screen_guardtests.log
?? worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_2026-08-11_13-55-32_screen_guardtests.log
?? worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_2026-08-11_13-59-56_screen_guardtests.log
?? worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_2026-08-11_14-02-59_screen_guardtests.log
?? worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_2026-08-11_14-12-10_screen_guardtests.log
?? worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_2026-08-11_14-15-09_screen_guardtests.log
?? worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_2026-08-11_14-17-36_screen_guardtests.log
?? worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_2026-08-11_14-20-42_screen_guardtests.log
?? worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_2026-08-11_14-49-04_screen_guardtests.log
?? worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_2026-08-11_14-53-46_screen_guardtests.log
?? worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_2026-08-11_14-57-09_screen_guardtests.log
?? worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_command.md
?? worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_submission_C16_vctl_S40000_s42_K8_jid3679847.txt
?? worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_submission_C32_vctl_S40000_s42_K8_jid3679858.txt
?? worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_submission_C4L_vctl_S40000_s42_K8_jid3679861.txt
?? worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_submission_VANL_vctl_S40000_s42_K8_jid3679860.txt
diff --git a/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch b/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch
index e7f6d54..2216bb4 100644
--- a/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch
+++ b/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch
@@ -16,7 +16,12 @@
 # SUBMIT: never by hand — `fa_orbit_submit.sh <ARM>` derives every Slurm flag
 # from the pins below, so an operator cannot mis-enter --gres/--mem/--time.
 #   ARM                C4L | C8 | C16 | C32
-#   EXPECT_SHA         full reviewed commit sha (required)
+#   EXPECT_SHA         full reviewed commit sha (required). Binding is by
+#                      CONTENT of the training surfaces, not HEAD identity:
+#                      a launch is accepted when HEAD == EXPECT_SHA, or when
+#                      the surfaces below are byte-identical between the two
+#                      (two writers commit to this checkout; worklog/record
+#                      commits must not kill a queued leg).
 #   RESUME_CKPT/EXPECTED_STEP   crash restart only (see LINEAGE)
 #   SMOKE=1            the reviewed multi-GPU smoke (see SMOKE MODE)
 # RUNG / MAXSTEPS / MIN_FREE_MB / time limit are NOT operator inputs any more.
@@ -200,15 +205,46 @@ HEAD_SHA="$(git rev-parse HEAD 2>/dev/null)"
 DRIFT="$(git status --porcelain --untracked-files=no -- train.py defaults.ini src \
           "$EXPDIR"/*.json "$EXPDIR"/*.py "$EXPDIR"/*.sbatch "$EXPDIR"/*.sh \
           "$EXP07/FLAC_AR_BF.json" 2>/dev/null)"
+# Commit binding is CONTENT-scoped: HEAD identity is sufficient but not
+# necessary. Two sessions commit to this checkout, so a pending leg must
+# survive commits that leave the training closure untouched — and abort on
+# any commit that changes it. The closure is what the job actually loads:
+# train.py, defaults.ini, src/, the arm configs, this launcher, the launch
+# wrapper and the four runtime helper scripts it invokes, plus exp_07's
+# FLAC_AR_BF.json (C4L parity baseline).
+# Record/analysis files (registry, manifests, gen_*/validators, worklog)
+# are deliberately OUTSIDE the closure. Fail-closed: unknown EXPECT_SHA or
+# a failing git diff aborts.
+surface_diff_vs_expect() {
+  git diff --name-only "${EXPECT_SHA}" HEAD -- train.py defaults.ini src \
+      "$EXPDIR"/FLAC_AR_*.json "$EXPDIR"/fa_orbit_train.sbatch "$EXPDIR"/fa_orbit_launch.sh \
+      "$EXPDIR"/fa_orbit_ckpt_preflight.py "$EXPDIR"/assert_arm_configs_exp11.py \
+      "$EXPDIR"/fa_orbit_wandb_readback.py "$EXPDIR"/fa_orbit_classify.py \
+      "$EXP07/FLAC_AR_BF.json"
+}
 if [ "$DRYRUN" = "1" ]; then
-  [ "$HEAD_SHA" = "$EXPECT_SHA" ] && echo "commit binding OK (dry run): ${HEAD_SHA}" \
-    || echo "DRY-RUN ADVISORY: HEAD ${HEAD_SHA} != EXPECT_SHA ${EXPECT_SHA} (a real launch aborts here)"
+  if [ "$HEAD_SHA" = "$EXPECT_SHA" ]; then
+    echo "commit binding OK (dry run): ${HEAD_SHA}"
+  elif ! git cat-file -e "${EXPECT_SHA}^{commit}" 2>/dev/null; then
+    echo "DRY-RUN ADVISORY: EXPECT_SHA ${EXPECT_SHA} is not a commit known to this repo (a real launch aborts here)"
+  elif SD="$(surface_diff_vs_expect 2>&1)" && [ -z "$SD" ]; then
+    echo "commit binding OK (dry run, content): training surfaces identical, EXPECT_SHA ${EXPECT_SHA} HEAD ${HEAD_SHA}"
+  else
+    echo "DRY-RUN ADVISORY: training surfaces changed since EXPECT_SHA ${EXPECT_SHA} (a real launch aborts here):"
+    echo "$SD"
+  fi
   [ -z "$DRIFT" ] || echo "DRY-RUN ADVISORY: tracked measurement surfaces are modified (a real launch aborts here)"
 else
   [ -n "${SLURM_JOB_ID:-}" ] || die "a real launch must run under sbatch (no SLURM_JOB_ID) - abort"
-  [ "$HEAD_SHA" = "$EXPECT_SHA" ] || die "HEAD ${HEAD_SHA} != EXPECT_SHA ${EXPECT_SHA} - abort"
+  if [ "$HEAD_SHA" = "$EXPECT_SHA" ]; then
+    echo "commit binding OK: ${HEAD_SHA}"
+  else
+    git cat-file -e "${EXPECT_SHA}^{commit}" 2>/dev/null || die "EXPECT_SHA ${EXPECT_SHA} is not a commit known to this repo - abort"
+    SD="$(surface_diff_vs_expect 2>&1)" || die "surface diff vs EXPECT_SHA failed: ${SD} - abort"
+    [ -z "$SD" ] || { echo "training surfaces changed since EXPECT_SHA ${EXPECT_SHA} (HEAD ${HEAD_SHA}) - abort:"; echo "$SD"; exit 2; }
+    echo "commit binding OK (content): training surfaces identical, EXPECT_SHA ${EXPECT_SHA} HEAD ${HEAD_SHA}"
+  fi
   [ -z "$DRIFT" ] || { echo "tracked measurement surfaces modified since review - abort:"; echo "$DRIFT"; exit 2; }
-  echo "commit binding OK: ${HEAD_SHA}"
 fi
 
 # --- D. arm -> config (single source) + semantic gate -------------------------
diff --git a/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh b/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh
index 6aed577..36021c1 100755
--- a/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh
+++ b/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh
@@ -147,6 +147,37 @@ case_run "wrong EXPECT_SHA aborts" 2 "EXPECT_SHA" \
 case_run "real mode needs sbatch" 2 "must run under sbatch" \
   -- ARM=C4L SMOKE=1 SMOKE_RUNG=16x4 SMOKE_MIN_FREE_MB=99000000 "EXPECT_SHA=${HEAD_SHA}" "OUTPUT_ROOT=${OUT_ROOT}"
 
+# Content-scoped binding: a queued leg must survive commits that leave the
+# training closure untouched, and abort on ones that change it. Both cases
+# scan real history for a suitable EXPECT_SHA; if none exists in 40 commits
+# the case is SKIPped (counted as neither PASS nor FAIL, but reported).
+surfaces_diff() { git diff --name-only "$1" HEAD -- train.py defaults.ini src \
+  "$EXPDIR"/FLAC_AR_*.json "$EXPDIR"/fa_orbit_train.sbatch "$EXPDIR"/fa_orbit_launch.sh \
+  "$EXPDIR"/fa_orbit_ckpt_preflight.py "$EXPDIR"/assert_arm_configs_exp11.py \
+  "$EXPDIR"/fa_orbit_wandb_readback.py "$EXPDIR"/fa_orbit_classify.py \
+  worklog/worklog_yixun/exp_07_fa_scratch_claude/FLAC_AR_BF.json; }
+SAME_SHA=""; CHANGED_SHA=""
+for i in $(seq 1 40); do
+  S="$(git rev-parse --verify -q "HEAD~${i}")" || break
+  D="$(surfaces_diff "$S")" || continue
+  if [ -z "$D" ] && [ -z "$SAME_SHA" ]; then SAME_SHA="$S"; fi
+  if [ -n "$D" ] && [ -z "$CHANGED_SHA" ]; then CHANGED_SHA="$S"; fi
+  [ -n "$SAME_SHA" ] && [ -n "$CHANGED_SHA" ] && break
+done
+if [ -n "$SAME_SHA" ]; then
+  # gate passes (message proves it), then the 99-TB VRAM floor aborts downstream
+  case_run "moved HEAD, surfaces identical -> gate passes" 2 "commit binding OK (content)" \
+    -- ARM=C4L SMOKE=1 SMOKE_RUNG=16x4 SMOKE_MIN_FREE_MB=99000000 "EXPECT_SHA=${SAME_SHA}" SLURM_JOB_ID=999999
+else
+  echo "SKIP  moved HEAD, surfaces identical: no such commit in 40-deep history"
+fi
+if [ -n "$CHANGED_SHA" ]; then
+  case_run "moved HEAD, surfaces changed -> aborts" 2 "training surfaces changed since EXPECT_SHA" \
+    -- ARM=C4L SMOKE=1 SMOKE_RUNG=16x4 SMOKE_MIN_FREE_MB=99000000 "EXPECT_SHA=${CHANGED_SHA}" SLURM_JOB_ID=999999
+else
+  echo "SKIP  moved HEAD, surfaces changed: no such commit in 40-deep history"
+fi
+
 echo "--- E. semantic gate on a mislabelled config (temp copy; tracked tree untouched) ---"
 FAKE_EXP="${TMP}/fakeexp"; mkdir -p "$FAKE_EXP"
 cp "${EXPDIR}/FLAC_AR_BF_C4L.json" "${FAKE_EXP}/FLAC_AR_BF_C32.json"      # C4 orbit under the C32 name

exec
/usr/bin/bash -lc "nl -ba worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh | sed -n '1,270p'" in /n/fs/gatrdp/codespace/FLAC
exec
/usr/bin/bash -lc "nl -ba worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch | sed -n '1,420p'" in /n/fs/gatrdp/codespace/FLAC
 succeeded in 0ms:
     1	#!/usr/bin/env bash
     2	# ============================================================================
     3	# fa_orbit_train_guardtests.sh — guard-branch exercise for the exp_11 arm
     4	# launcher (round-3 review B8 rebuilt this suite).
     5	#
     6	# SAFETY (the old suite violated all three):
     7	#   * it never writes under a production output prefix — every case runs with
     8	#     OUTPUT_ROOT pointed at a mktemp directory;
     9	#   * it never mutates a tracked config — the mislabel case copies the tree into
    10	#     the temp root and points the launcher at the copy via OUTPUT_ROOT-style
    11	#     isolation, and any file it does touch is restored by an EXIT trap;
    12	#   * it submits nothing and touches no GPU.
    13	#
    14	# Vehicles:
    15	#   DRYRUN=1        every cheap gate (pins, arm, rung, config map, semantic
    16	#                   gate, lineage, argv parity), then exit before Slurm/GPU.
    17	#   real mode       with a fake SLURM_JOB_ID: proves the commit/drift and
    18	#                   sbatch-only gates are fail-closed.
    19	#   mocked logs     fa_orbit_classify.py is driven directly over synthetic logs
    20	#                   to prove every exit class (0/3/4/6/7).
    21	#   synthetic ckpt  fa_orbit_ckpt_preflight.py is driven over torch.save'd
    22	#                   Lightning-shaped checkpoints to prove the restart depth.
    23	#
    24	# Usage:  bash worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh
    25	# Exit 0 = every case behaved as specified.
    26	# ============================================================================
    27	set -uo pipefail
    28	cd "$(git -C "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" rev-parse --show-toplevel)" || exit 3
    29	
    30	EXPDIR="worklog/worklog_yixun/exp_11_fa_orbit_claude"
    31	LAUNCHER="${EXPDIR}/fa_orbit_train.sbatch"
    32	SUBMITTER="${EXPDIR}/fa_orbit_submit.sh"
    33	CLASSIFY="${EXPDIR}/fa_orbit_classify.py"
    34	PREFLIGHT="${EXPDIR}/fa_orbit_ckpt_preflight.py"
    35	PY=/n/fs/gatrdp/envs/flac/bin/python
    36	TS="$(date '+%Y-%m-%d_%H-%M-%S')"
    37	LOG="${EXPDIR}/fa_orbit_${TS}_guardtests.log"
    38	HEAD_SHA="$(git rev-parse HEAD)"
    39	
    40	exec > >(tee -a "$LOG") 2>&1
    41	echo "=== fa_orbit_train guard exercise — ${TS} — $(git rev-parse --short HEAD) ==="
    42	for f in "$LAUNCHER" "$SUBMITTER" "$CLASSIFY" "$PREFLIGHT"; do
    43	  [ -f "$f" ] || { echo "missing ${f} - abort"; exit 3; }
    44	done
    45	
    46	TRACKED_BEFORE="$(git status --porcelain -- "$EXPDIR" src | sort)"
    47	TMP="$(mktemp -d)"
    48	OUT_ROOT="${TMP}/outputs"            # never a production prefix
    49	mkdir -p "$OUT_ROOT"
    50	trap 'rm -rf "$TMP"' EXIT
    51	PASS=0; FAIL=0
    52	
    53	case_run() {  # <name> <want-rc> <want-substring> -- <env...>   (runs the launcher)
    54	  local name="$1" want_rc="$2" want_txt="$3"; shift 3; [ "$1" = "--" ] && shift
    55	  local out rc
    56	  out="$(env "$@" bash "$LAUNCHER" 2>&1)"; rc=$?
    57	  if [ "$rc" -eq "$want_rc" ] && echo "$out" | grep -qF -- "$want_txt"; then
    58	    echo "PASS  ${name}  (rc=${rc})"; PASS=$((PASS + 1))
    59	  else
    60	    echo "FAIL  ${name}: want rc=${want_rc} + '${want_txt}', got rc=${rc}"
    61	    echo "$out" | tail -5 | sed 's/^/        | /'; FAIL=$((FAIL + 1))
    62	  fi
    63	}
    64	
    65	expect_cmd() {  # <name> <want-rc> <want-substring> -- <command...>
    66	  local name="$1" want_rc="$2" want_txt="$3"; shift 3; [ "$1" = "--" ] && shift
    67	  local out rc
    68	  out="$("$@" 2>&1)"; rc=$?
    69	  if [ "$rc" -eq "$want_rc" ] && echo "$out" | grep -qF -- "$want_txt"; then
    70	    echo "PASS  ${name}  (rc=${rc})"; PASS=$((PASS + 1))
    71	  else
    72	    echo "FAIL  ${name}: want rc=${want_rc} + '${want_txt}', got rc=${rc}"
    73	    echo "$out" | tail -5 | sed 's/^/        | /'; FAIL=$((FAIL + 1))
    74	  fi
    75	}
    76	
    77	REPO_ENV=("FA_ORBIT_REPO_OVERRIDE=$PWD")   # dry runs read THIS tree, not the production checkout
    78	SMOKE_ENV=(DRYRUN=1 "EXPECT_SHA=${HEAD_SHA}" "OUTPUT_ROOT=${OUT_ROOT}" "${REPO_ENV[@]}" SMOKE=1 SMOKE_RUNG=16x4 SMOKE_MIN_FREE_MB=14000)
    79	
    80	echo "--- A. the pin mechanism refuses to launch un-pinned (round-3 B1) ---"
    81	# RETIRED: this asserted that an UNPINNED arm refuses, but every pin landed in
    82	# ea94995, so the placeholder no longer appears in any value and the case could
    83	# never fire. Replaced by the end state it was protecting, plus proof that the
    84	# refusal mechanism itself is still present to catch a future unpinned value.
    85	if grep -qE '^PINNED_[A-Z_]+="TO-PIN-AFTER-P0"' "$LAUNCHER"; then
    86	  echo "FAIL  a launcher pin is still the placeholder"; FAIL=$((FAIL+1))
    87	else
    88	  echo "PASS  every launcher pin holds a concrete value"; PASS=$((PASS+1))
    89	fi
    90	if grep -q 'PIN_PLACEHOLDER="TO-PIN-AFTER-P0"' "$LAUNCHER" \
    91	   && grep -q 'PIN_PLACEHOLDER' "$LAUNCHER"; then
    92	  echo "PASS  the launcher still refuses a placeholder pin if one returns"; PASS=$((PASS+1))
    93	else
    94	  echo "FAIL  the placeholder refusal mechanism is gone"; FAIL=$((FAIL+1))
    95	fi
    96	case_run "SMOKE bypasses the pins" 0 "ARGV PARITY OK" -- "${SMOKE_ENV[@]}" ARM=C8
    97	case_run "SMOKE needs a rung" 2 "SMOKE_RUNG" \
    98	  -- DRYRUN=1 SMOKE=1 ARM=C8 "EXPECT_SHA=${HEAD_SHA}" "OUTPUT_ROOT=${OUT_ROOT}" "${REPO_ENV[@]}"
    99	case_run "SMOKE needs a VRAM floor" 2 "SMOKE_MIN_FREE_MB" \
   100	  -- DRYRUN=1 SMOKE=1 SMOKE_RUNG=16x4 ARM=C8 "EXPECT_SHA=${HEAD_SHA}" "OUTPUT_ROOT=${OUT_ROOT}" "${REPO_ENV[@]}"
   101	case_run "SMOKE identity is separate" 0 "exp11_smoke_C8" -- "${SMOKE_ENV[@]}" ARM=C8
   102	
   103	echo "--- B. parameter / arm / rung gates ---"
   104	case_run "missing ARM" 2 "ARM" -- DRYRUN=1 "EXPECT_SHA=${HEAD_SHA}" "OUTPUT_ROOT=${OUT_ROOT}" "${REPO_ENV[@]}"
   105	case_run "missing EXPECT_SHA" 2 "EXPECT_SHA" -- DRYRUN=1 ARM=C8 "OUTPUT_ROOT=${OUT_ROOT}" "${REPO_ENV[@]}"
   106	for BAD in C7 FA1 VAN CKPT4; do
   107	  case_run "arm ${BAD} rejected" 2 "not a legal exp_11 arm" -- "${SMOKE_ENV[@]}" ARM=$BAD
   108	done
   109	case_run "bogus rung rejected" 2 "must be 32x2, 16x4 or 8x8" \
   110	  -- DRYRUN=1 SMOKE=1 SMOKE_RUNG=64x1 SMOKE_MIN_FREE_MB=1 ARM=C8 "EXPECT_SHA=${HEAD_SHA}" "OUTPUT_ROOT=${OUT_ROOT}" "${REPO_ENV[@]}"
   111	for R in 32x2 16x4 8x8; do   # all three rungs are feasible now that grad-ckpt is on
   112	  case_run "rung ${R} accepted" 0 "ARGV PARITY OK" \
   113	    -- DRYRUN=1 SMOKE=1 "SMOKE_RUNG=${R}" SMOKE_MIN_FREE_MB=14000 ARM=C4L "EXPECT_SHA=${HEAD_SHA}" "OUTPUT_ROOT=${OUT_ROOT}" "${REPO_ENV[@]}"
   114	done
   115	
   116	echo "--- C. lineage gates ---"
   117	: > "${TMP}/foreign.ckpt"
   118	case_run "initial + RESUME_CKPT" 2 "INITIAL launch must not carry" \
   119	  -- "${SMOKE_ENV[@]}" ARM=C8 "RESUME_CKPT=${TMP}/foreign.ckpt"
   120	case_run "restart w/o ckpt" 2 "RESTART requires RESUME_CKPT" \
   121	  -- "${SMOKE_ENV[@]}" ARM=C8 EXPECTED_STEP=5000
   122	case_run "restart ckpt missing" 2 "not found" \
   123	  -- "${SMOKE_ENV[@]}" ARM=C8 EXPECTED_STEP=5000 "RESUME_CKPT=${TMP}/nope.ckpt"
   124	case_run "restart foreign ckpt" 2 "may only resume a checkpoint from" \
   125	  -- "${SMOKE_ENV[@]}" ARM=C8 EXPECTED_STEP=5000 "RESUME_CKPT=${TMP}/foreign.ckpt"
   126	# a ckpt in the arm's own checkpoints dir but NOT named .ckpt / one level up
   127	SMOKE_RUN="${OUT_ROOT}/exp11_smoke/C8/FLAC_exp11_smoke_C8/exp11_smoke_C8"
   128	mkdir -p "${SMOKE_RUN}/checkpoints"
   129	: > "${SMOKE_RUN}/checkpoints/epoch=1-step=5000.ckpt"
   130	: > "${SMOKE_RUN}/notes.txt"
   131	case_run "restart from the arm's own ckpt dir" 0 "ARGV PARITY OK" \
   132	  -- "${SMOKE_ENV[@]}" ARM=C8 EXPECTED_STEP=5000 SMOKE_MAXSTEPS=6000 \
   133	     "RESUME_CKPT=${SMOKE_RUN}/checkpoints/epoch=1-step=5000.ckpt"
   134	case_run "restart from a non-ckpt sibling" 2 "may only resume a checkpoint from" \
   135	  -- "${SMOKE_ENV[@]}" ARM=C8 EXPECTED_STEP=5000 SMOKE_MAXSTEPS=6000 "RESUME_CKPT=${SMOKE_RUN}/notes.txt"
   136	case_run "restart MAXSTEPS<=step" 2 "must exceed the resume step" \
   137	  -- "${SMOKE_ENV[@]}" ARM=C8 EXPECTED_STEP=5000 SMOKE_MAXSTEPS=30 \
   138	     "RESUME_CKPT=${SMOKE_RUN}/checkpoints/epoch=1-step=5000.ckpt"
   139	case_run "initial refuses an existing run dir" 2 "already exists" -- "${SMOKE_ENV[@]}" ARM=C8
   140	
   141	echo "--- D. commit-binding / sbatch-only gates (REAL mode) ---"
   142	# NOTE: no OUTPUT_ROOT here — under a (fake) SLURM_JOB_ID the launcher forces the
   143	# production literal, and the commit gate aborts long before anything is written.
   144	case_run "wrong EXPECT_SHA aborts" 2 "EXPECT_SHA" \
   145	  -- ARM=C4L SMOKE=1 SMOKE_RUNG=16x4 SMOKE_MIN_FREE_MB=99000000 \
   146	     EXPECT_SHA=0000000000000000000000000000000000000000 SLURM_JOB_ID=999999
   147	case_run "real mode needs sbatch" 2 "must run under sbatch" \
   148	  -- ARM=C4L SMOKE=1 SMOKE_RUNG=16x4 SMOKE_MIN_FREE_MB=99000000 "EXPECT_SHA=${HEAD_SHA}" "OUTPUT_ROOT=${OUT_ROOT}"
   149	
   150	# Content-scoped binding: a queued leg must survive commits that leave the
   151	# training closure untouched, and abort on ones that change it. Both cases
   152	# scan real history for a suitable EXPECT_SHA; if none exists in 40 commits
   153	# the case is SKIPped (counted as neither PASS nor FAIL, but reported).
   154	surfaces_diff() { git diff --name-only "$1" HEAD -- train.py defaults.ini src \
   155	  "$EXPDIR"/FLAC_AR_*.json "$EXPDIR"/fa_orbit_train.sbatch "$EXPDIR"/fa_orbit_launch.sh \
   156	  "$EXPDIR"/fa_orbit_ckpt_preflight.py "$EXPDIR"/assert_arm_configs_exp11.py \
   157	  "$EXPDIR"/fa_orbit_wandb_readback.py "$EXPDIR"/fa_orbit_classify.py \
   158	  worklog/worklog_yixun/exp_07_fa_scratch_claude/FLAC_AR_BF.json; }
   159	SAME_SHA=""; CHANGED_SHA=""
   160	for i in $(seq 1 40); do
   161	  S="$(git rev-parse --verify -q "HEAD~${i}")" || break
   162	  D="$(surfaces_diff "$S")" || continue
   163	  if [ -z "$D" ] && [ -z "$SAME_SHA" ]; then SAME_SHA="$S"; fi
   164	  if [ -n "$D" ] && [ -z "$CHANGED_SHA" ]; then CHANGED_SHA="$S"; fi
   165	  [ -n "$SAME_SHA" ] && [ -n "$CHANGED_SHA" ] && break
   166	done
   167	if [ -n "$SAME_SHA" ]; then
   168	  # gate passes (message proves it), then the 99-TB VRAM floor aborts downstream
   169	  case_run "moved HEAD, surfaces identical -> gate passes" 2 "commit binding OK (content)" \
   170	    -- ARM=C4L SMOKE=1 SMOKE_RUNG=16x4 SMOKE_MIN_FREE_MB=99000000 "EXPECT_SHA=${SAME_SHA}" SLURM_JOB_ID=999999
   171	else
   172	  echo "SKIP  moved HEAD, surfaces identical: no such commit in 40-deep history"
   173	fi
   174	if [ -n "$CHANGED_SHA" ]; then
   175	  case_run "moved HEAD, surfaces changed -> aborts" 2 "training surfaces changed since EXPECT_SHA" \
   176	    -- ARM=C4L SMOKE=1 SMOKE_RUNG=16x4 SMOKE_MIN_FREE_MB=99000000 "EXPECT_SHA=${CHANGED_SHA}" SLURM_JOB_ID=999999
   177	else
   178	  echo "SKIP  moved HEAD, surfaces changed: no such commit in 40-deep history"
   179	fi
   180	
   181	echo "--- E. semantic gate on a mislabelled config (temp copy; tracked tree untouched) ---"
   182	FAKE_EXP="${TMP}/fakeexp"; mkdir -p "$FAKE_EXP"
   183	cp "${EXPDIR}/FLAC_AR_BF_C4L.json" "${FAKE_EXP}/FLAC_AR_BF_C32.json"      # C4 orbit under the C32 name
   184	expect_cmd "orbit mismatch rejected" 1 "ARM/CONFIG GATE" -- \
   185	  $PY - "${FAKE_EXP}/FLAC_AR_BF_C32.json" C32 <<'PY'
   186	import json, sys
   187	cfg = json.load(open(sys.argv[1])); arm = sys.argv[2]
   188	t = cfg.get("training", {}); bad = []
   189	want = {"C4L": 4, "C8": 8, "C16": 16, "C32": 32}[arm]
   190	angles = t.get("frame_avg_angles")
   191	if not isinstance(angles, list) or len(angles) != want:
   192	    bad.append(f"frame_avg_angles has {angles and len(angles)} entries (want {want})")
   193	if bad:
   194	    sys.exit("ARM/CONFIG GATE: " + "; ".join(bad))
   195	PY
   196	TRACKED_AFTER="$(git status --porcelain -- "$EXPDIR" src | sort)"
   197	if [ "$TRACKED_BEFORE" = "$TRACKED_AFTER" ]; then
   198	  echo "PASS  tracked tree unchanged by the suite (snapshot before == after)"; PASS=$((PASS+1))
   199	else
   200	  echo "FAIL  the suite changed tracked state:"; diff <(echo "$TRACKED_BEFORE") <(echo "$TRACKED_AFTER") | sed 's/^/        | /'
   201	  FAIL=$((FAIL+1))
   202	fi
   203	
   204	echo "--- F. exit taxonomy, mocked (round-3 B5) ---"
   205	mk_log() {  # $1 dest, $2 world size (0 = absent), $3 marker?, $4 oom?
   206	  : > "$1"
   207	  [ "$2" != "0" ] && echo "All distributed processes registered. Starting with $2 processes" >> "$1"
   208	  [ "$3" = "yes" ] && echo '`Trainer.fit` stopped: `max_steps=40000` reached.' >> "$1"
   209	  [ "$4" = "yes" ] && echo "torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 98.00 MiB" >> "$1"
   210	  return 0
   211	}
   212	A="${TMP}/a.log"; B="${TMP}/b.log"
   213	mk_log "$A" 4 yes no; cp "$A" "$B"
   214	expect_cmd "class 0 complete" 0 "COMPLETE" -- $PY "$CLASSIFY" --rc 0 --tee-rc 0 --ngpu 4 --maxsteps 40000 --log "$A" --log-copy "$B"
   215	mk_log "$A" 0 no no; cp "$A" "$B"
   216	expect_cmd "class 6 world-size absent" 6 "WORLD-SIZE" -- $PY "$CLASSIFY" --rc 0 --tee-rc 0 --ngpu 4 --maxsteps 40000 --log "$A" --log-copy "$B"
   217	mk_log "$A" 1 yes no; cp "$A" "$B"
   218	expect_cmd "class 6 wrong world-size" 6 "reported [1]" -- $PY "$CLASSIFY" --rc 0 --tee-rc 0 --ngpu 4 --maxsteps 40000 --log "$A" --log-copy "$B"
   219	mk_log "$A" 4 no yes; cp "$A" "$B"
   220	expect_cmd "class 3 OOM on nonzero rc" 3 "OOM" -- $PY "$CLASSIFY" --rc 1 --tee-rc 0 --ngpu 4 --maxsteps 40000 --log "$A" --log-copy "$B"
   221	mk_log "$A" 4 no no; cp "$A" "$B"
   222	expect_cmd "class 4 missing marker" 4 "NO-MARKER" -- $PY "$CLASSIFY" --rc 0 --tee-rc 0 --ngpu 4 --maxsteps 40000 --log "$A" --log-copy "$B"
   223	mk_log "$A" 4 yes no; cp "$A" "$B"; echo "divergent tail" >> "$B"
   224	expect_cmd "class 7 logs differ" 7 "LOG-PROVENANCE" -- $PY "$CLASSIFY" --rc 0 --tee-rc 0 --ngpu 4 --maxsteps 40000 --log "$A" --log-copy "$B"
   225	mk_log "$A" 4 yes no; cp "$A" "$B"; rm -f "$B"
   226	expect_cmd "class 7 copy missing" 7 "missing log copy" -- $PY "$CLASSIFY" --rc 0 --tee-rc 0 --ngpu 4 --maxsteps 40000 --log "$A" --log-copy "$B"
   227	mk_log "$A" 4 yes no; cp "$A" "$B"
   228	expect_cmd "class 7 tee failed" 7 "tee exited" -- $PY "$CLASSIFY" --rc 0 --tee-rc 1 --ngpu 4 --maxsteps 40000 --log "$A" --log-copy "$B"
   229	mk_log "$A" 4 no no; cp "$A" "$B"
   230	expect_cmd "raw rc preserved" 9 "RUNTIME" -- $PY "$CLASSIFY" --rc 9 --tee-rc 0 --ngpu 4 --maxsteps 40000 --log "$A" --log-copy "$B"
   231	
   232	echo "--- G. restart preflight depth, mocked checkpoints (round-3 B2) ---"
   233	$PY - "$TMP" "${EXPDIR}/FLAC_AR_BF_C8.json" <<'PY'
   234	import json, os, sys, torch
   235	tmp, cfg_path = sys.argv[1], sys.argv[2]
   236	cfg = json.load(open(cfg_path))
   237	def ck(step=5000, config=cfg, opt=True, sched=True, ema=True):
   238	    d = {"global_step": step, "epoch": 1, "model_config": config,
   239	         "state_dict": {"diffusion.x": torch.zeros(1)},
   240	         "optimizer_states": [{"state": {0: {"step": 1}} if opt else {},
   241	                               "param_groups": [{"lr": 1e-5}]}],
   242	         "lr_schedulers": [{"last_epoch": step}] if sched else []}
   243	    if ema:
   244	        d["state_dict"]["diffusion_ema.x"] = torch.zeros(1)
   245	    return d
   246	torch.save(ck(), os.path.join(tmp, "good.ckpt"))
   247	torch.save(ck(step=4999), os.path.join(tmp, "wrongstep.ckpt"))
   248	c4 = json.loads(json.dumps(cfg)); c4["training"]["frame_avg_angles"] = [0.0, 90.0, 180.0, 270.0]
   249	torch.save(ck(config=c4), os.path.join(tmp, "wrongorbit.ckpt"))
   250	torch.save(ck(opt=False), os.path.join(tmp, "stripped.ckpt"))
   251	torch.save(ck(ema=False), os.path.join(tmp, "noema.ckpt"))
   252	torch.save(ck(sched=False), os.path.join(tmp, "nosched.ckpt"))
   253	torch.save(ck(step=45000), os.path.join(tmp, "past.ckpt"))
   254	open(os.path.join(tmp, "empty.ckpt"), "wb").close()
   255	print("synthetic checkpoints written")
   256	PY
   257	PRE=($PY "$PREFLIGHT" --config "${EXPDIR}/FLAC_AR_BF_C8.json" --max-steps 40000 --arm C8 --rung 16x4)
   258	expect_cmd "preflight accepts a good ckpt" 0 "CKPT_SHA256" -- "${PRE[@]}" --ckpt "${TMP}/good.ckpt" --expected-step 5000
   259	expect_cmd "preflight rejects a step mismatch" 2 "global_step" -- "${PRE[@]}" --ckpt "${TMP}/wrongstep.ckpt" --expected-step 5000
   260	expect_cmd "preflight rejects a foreign orbit" 2 "embedded model_config" -- "${PRE[@]}" --ckpt "${TMP}/wrongorbit.ckpt" --expected-step 5000
   261	expect_cmd "preflight rejects a stripped optimizer" 2 "optimizer state is CLEARED" -- "${PRE[@]}" --ckpt "${TMP}/stripped.ckpt" --expected-step 5000
   262	expect_cmd "preflight rejects a missing EMA" 2 "no EMA weights" -- "${PRE[@]}" --ckpt "${TMP}/noema.ckpt" --expected-step 5000
   263	expect_cmd "preflight rejects a missing scheduler" 2 "lr_schedulers" -- "${PRE[@]}" --ckpt "${TMP}/nosched.ckpt" --expected-step 5000
   264	expect_cmd "preflight rejects a past-budget ckpt" 2 ">= max_steps" -- "${PRE[@]}" --ckpt "${TMP}/past.ckpt" --expected-step 45000
   265	expect_cmd "preflight rejects an empty file" 2 "PREFLIGHT" -- "${PRE[@]}" --ckpt "${TMP}/empty.ckpt" --expected-step 5000
   266	expect_cmd "preflight rejects a missing file" 2 "not found" -- "${PRE[@]}" --ckpt "${TMP}/nope.ckpt" --expected-step 5000
   267	# manifest binding: same rung passes, changed rung fails
   268	cat > "${TMP}/launch_manifest.txt" <<EOF
   269	# exp_11 arm launch manifest
   270	arm C8 rung 16x4 micro 16 ngpu 4 max_steps 40000 ckpt_every 2500

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
    19	#   EXPECT_SHA         full reviewed commit sha (required). Binding is by
    20	#                      CONTENT of the training surfaces, not HEAD identity:
    21	#                      a launch is accepted when HEAD == EXPECT_SHA, or when
    22	#                      the surfaces below are byte-identical between the two
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
   200	HEAD_SHA="$(git rev-parse HEAD 2>/dev/null)"
   201	# The drift gate is scoped to CODE surfaces, not the whole exp folder: the four
   202	# arms are running and Slurm appends to their tracked *.out logs continuously, so
   203	# a folder-wide check would abort every screen on a live-log write. Configs,
   204	# drivers and validators are still fully covered.
   205	DRIFT="$(git status --porcelain --untracked-files=no -- train.py defaults.ini src \
   206	          "$EXPDIR"/*.json "$EXPDIR"/*.py "$EXPDIR"/*.sbatch "$EXPDIR"/*.sh \
   207	          "$EXP07/FLAC_AR_BF.json" 2>/dev/null)"
   208	# Commit binding is CONTENT-scoped: HEAD identity is sufficient but not
   209	# necessary. Two sessions commit to this checkout, so a pending leg must
   210	# survive commits that leave the training closure untouched — and abort on
   211	# any commit that changes it. The closure is what the job actually loads:
   212	# train.py, defaults.ini, src/, the arm configs, this launcher, the launch
   213	# wrapper and the four runtime helper scripts it invokes, plus exp_07's
   214	# FLAC_AR_BF.json (C4L parity baseline).
   215	# Record/analysis files (registry, manifests, gen_*/validators, worklog)
   216	# are deliberately OUTSIDE the closure. Fail-closed: unknown EXPECT_SHA or
   217	# a failing git diff aborts.
   218	surface_diff_vs_expect() {
   219	  git diff --name-only "${EXPECT_SHA}" HEAD -- train.py defaults.ini src \
   220	      "$EXPDIR"/FLAC_AR_*.json "$EXPDIR"/fa_orbit_train.sbatch "$EXPDIR"/fa_orbit_launch.sh \
   221	      "$EXPDIR"/fa_orbit_ckpt_preflight.py "$EXPDIR"/assert_arm_configs_exp11.py \
   222	      "$EXPDIR"/fa_orbit_wandb_readback.py "$EXPDIR"/fa_orbit_classify.py \
   223	      "$EXP07/FLAC_AR_BF.json"
   224	}
   225	if [ "$DRYRUN" = "1" ]; then
   226	  if [ "$HEAD_SHA" = "$EXPECT_SHA" ]; then
   227	    echo "commit binding OK (dry run): ${HEAD_SHA}"
   228	  elif ! git cat-file -e "${EXPECT_SHA}^{commit}" 2>/dev/null; then
   229	    echo "DRY-RUN ADVISORY: EXPECT_SHA ${EXPECT_SHA} is not a commit known to this repo (a real launch aborts here)"
   230	  elif SD="$(surface_diff_vs_expect 2>&1)" && [ -z "$SD" ]; then
   231	    echo "commit binding OK (dry run, content): training surfaces identical, EXPECT_SHA ${EXPECT_SHA} HEAD ${HEAD_SHA}"
   232	  else
   233	    echo "DRY-RUN ADVISORY: training surfaces changed since EXPECT_SHA ${EXPECT_SHA} (a real launch aborts here):"
   234	    echo "$SD"
   235	  fi
   236	  [ -z "$DRIFT" ] || echo "DRY-RUN ADVISORY: tracked measurement surfaces are modified (a real launch aborts here)"
   237	else
   238	  [ -n "${SLURM_JOB_ID:-}" ] || die "a real launch must run under sbatch (no SLURM_JOB_ID) - abort"
   239	  if [ "$HEAD_SHA" = "$EXPECT_SHA" ]; then
   240	    echo "commit binding OK: ${HEAD_SHA}"
   241	  else
   242	    git cat-file -e "${EXPECT_SHA}^{commit}" 2>/dev/null || die "EXPECT_SHA ${EXPECT_SHA} is not a commit known to this repo - abort"
   243	    SD="$(surface_diff_vs_expect 2>&1)" || die "surface diff vs EXPECT_SHA failed: ${SD} - abort"
   244	    [ -z "$SD" ] || { echo "training surfaces changed since EXPECT_SHA ${EXPECT_SHA} (HEAD ${HEAD_SHA}) - abort:"; echo "$SD"; exit 2; }
   245	    echo "commit binding OK (content): training surfaces identical, EXPECT_SHA ${EXPECT_SHA} HEAD ${HEAD_SHA}"
   246	  fi
   247	  [ -z "$DRIFT" ] || { echo "tracked measurement surfaces modified since review - abort:"; echo "$DRIFT"; exit 2; }
   248	fi
   249	
   250	# --- D. arm -> config (single source) + semantic gate -------------------------
   251	arm_config_for() {
   252	  case "$1" in
   253	    C4L|C8|C16|C32) echo "$EXPDIR/FLAC_AR_BF_$1.json" ;;
   254	    VANL)           echo "$EXPDIR/FLAC_AR_VANCKPT.json" ;;
   255	    *) return 1 ;;
   256	  esac
   257	}
   258	MODEL_CONFIG="$(arm_config_for "$ARM")" || die "no config mapped for arm '${ARM}' - abort"
   259	MODEL_CONFIG_ABS="$(readlink -f "$MODEL_CONFIG" 2>/dev/null)"
   260	[ -n "$MODEL_CONFIG_ABS" ] && [ -f "$MODEL_CONFIG_ABS" ] || die "arm config '${MODEL_CONFIG}' does not exist - abort"
   261	CONFIG_SHA="$(sha256sum "$MODEL_CONFIG_ABS" | awk '{print $1}')"
   262	echo "config for ${ARM}: ${MODEL_CONFIG_ABS} sha256 ${CONFIG_SHA}"
   263	
   264	python3 - "$MODEL_CONFIG_ABS" "$ARM" <<'PY' || die "arm/config semantic gate FAILED - abort"
   265	import json, sys
   266	cfg = json.load(open(sys.argv[1])); arm = sys.argv[2]
   267	t = cfg.get("training", {}); bad = []
   268	# VANL is the same recipe with the conditioning removed, so its gate is the
   269	# MIRROR IMAGE of the orbit arms': the orbit keys must be ABSENT, not merely
   270	# different. A vanilla config that carried a stray frame_avg_angles would be a
   271	# silently fa-flavoured baseline, which would destroy the single-delta claim.
   272	if arm == "VANL":
   273	    cm = t.get("cond_method")
   274	    if cm not in (None, "vanilla"):
   275	        bad.append(f"cond_method={cm!r} (want absent or 'vanilla')")
   276	    if "frame_avg_angles" in t:
   277	        bad.append(f"frame_avg_angles is present ({t['frame_avg_angles']!r}) — a vanilla arm has no orbit")
   278	    want = None
   279	else:
   280	    want = {"C4L": 4, "C8": 8, "C16": 16, "C32": 32}[arm]
   281	    angles = t.get("frame_avg_angles")
   282	    if t.get("cond_method") != "fa_invariant":
   283	        bad.append(f"cond_method={t.get('cond_method')!r} (want fa_invariant)")
   284	    if not isinstance(angles, list) or len(angles) != want:
   285	        bad.append(f"frame_avg_angles has {angles and len(angles)} entries (want {want})")
   286	    elif angles != [k * 360.0 / want for k in range(want)]:
   287	        bad.append(f"frame_avg_angles are not the uniform C{want} orbit")
   288	if t.get("use_ema") is not True:
   289	    bad.append(f"use_ema={t.get('use_ema')!r} (want True)")
   290	vits = [c for c in cfg["model"]["conditioning"]["configs"] if c["type"] == "ViTCoordinates"]
   291	if sorted(c["id"] for c in vits) != ["context_poses_vit", "source_vit"]:
   292	    bad.append(f"ViT conditioner ids {sorted(c['id'] for c in vits)} != the expected two")
   293	# Post-P0: grad-ckpt ON for every arm; the KEY must exist and be literally True
   294	for c in vits:
   295	    if "gradient_checkpointing" not in c["config"]:
   296	        bad.append(f"{c['id']}: gradient_checkpointing key absent (want literal true)")
   297	    elif c["config"]["gradient_checkpointing"] is not True:
   298	        bad.append(f"{c['id']}: gradient_checkpointing={c['config']['gradient_checkpointing']!r} (want True)")
   299	if bad:
   300	    sys.exit("ARM/CONFIG GATE: " + "; ".join(bad))
   301	if arm == "VANL":
   302	    print(f"gate OK: {arm} is vanilla (no cond_method, no orbit), grad-ckpt True, EMA on")
   303	else:
   304	    print(f"gate OK: {arm} carries the uniform C{want} orbit, grad-ckpt True, EMA on")
   305	PY
   306	
   307	# --- E. lineage: INITIAL vs RESTART -------------------------------------------
   308	SAVEDIR_REAL="$(realpath -m "$SAVEDIR")"
   309	CKPT_DIR_REAL="$(realpath -m "${RUNDIR}/checkpoints")"
   310	LAUNCH_MANIFEST_LINK="${SAVEDIR}/launch_manifest.txt"     # written by the INITIAL launch
   311	if [ "$EXPECTED_STEP" -eq 0 ]; then
   312	  MODE="INITIAL"
   313	  [ -z "$RESUME_CKPT" ] || die "INITIAL launch must not carry RESUME_CKPT (set EXPECTED_STEP > 0 to declare a RESTART) - abort"
   314	  [ ! -e "$RUNDIR" ] || die "run directory ${RUNDIR} already exists — an INITIAL launch never clobbers a previous run - abort"
   315	else
   316	  MODE="RESTART"
   317	  [ -n "$RESUME_CKPT" ] || die "EXPECTED_STEP ${EXPECTED_STEP} declares a RESTART, but RESTART requires RESUME_CKPT - abort"
   318	  [ -f "$RESUME_CKPT" ] || die "RESUME_CKPT not found: ${RESUME_CKPT} - abort"
   319	  RESUME_REAL="$(realpath -m "$RESUME_CKPT")"
   320	  # exactly this arm's own checkpoints directory — not merely somewhere below the save root
   321	  case "$RESUME_REAL" in
   322	    "${CKPT_DIR_REAL}"/*.ckpt) ;;
   323	    *) die "a RESTART may only resume a checkpoint from ${CKPT_DIR_REAL}/ (got ${RESUME_REAL}) - abort" ;;
   324	  esac
   325	  [ "$MAXSTEPS" -gt "$EXPECTED_STEP" ] || die "MAXSTEPS ${MAXSTEPS} must exceed the resume step ${EXPECTED_STEP} - abort"
   326	fi
   327	echo "lineage: ${MODE} (expected_step ${EXPECTED_STEP}, max_steps ${MAXSTEPS}, ckpt every ${CHECKPOINT_EVERY}, time pin ${TIME_PIN_NAME}=${TIME_LIMIT})"
   328	
   329	# --- F. the exact train.py argv ----------------------------------------------
   330	ARGV=(
   331	  --model-config "$MODEL_CONFIG_ABS"
   332	  --dataset-config src/configs/dataset_configs/AR/train/acousticroom_train.json
   333	  --pretransform-ckpt-path weights/FLAC/VAE.safetensors
   334	  --max-steps "$MAXSTEPS" --batch-size "$MB" --accum-batches 1 --num-workers 6 --seed 42
   335	  --num-gpus "$NGPU" --num-nodes 1
   336	  --strategy ddp_find_unused_parameters_true --sync-batchnorm true --precision bf16-mixed
   337	  --val-every -1 --val-dataset-config ''
   338	  --gradient-clip-val 0.0
   339	  --logger wandb --checkpoint-every "$CHECKPOINT_EVERY"
   340	  --name "$NAME" --experiment-name "$EXPNAME" --save-dir "$SAVEDIR"
   341	)
   342	[ "$MODE" = "RESTART" ] && ARGV+=(--ckpt-path "$RESUME_CKPT")
   343	
   344	# --- G. argv-parity dry run (plan N13; round-3 N9 tightened) ------------------
   345	ARGV_FILE="$(mktemp)" || die "mktemp failed - abort" 3
   346	printf '%s\n' "${ARGV[@]}" > "$ARGV_FILE" || die "could not write the argv file - abort" 3
   347	python3 - "$ARGV_FILE" "$MODE" <<'PY'
   348	import sys
   349	# The exp_07 B-F reference argv (bf_scratch_launch.sh) — the lineage this sweep continues.
   350	REF = """--model-config worklog/worklog_yixun/exp_07_fa_scratch_claude/FLAC_AR_BF.json
   351	--dataset-config src/configs/dataset_configs/AR/train/acousticroom_train.json
   352	--pretransform-ckpt-path weights/FLAC/VAE.safetensors
   353	--max-steps 67500 --batch-size 32 --accum-batches 1 --num-workers 6 --seed 42
   354	--num-gpus 2 --strategy ddp_find_unused_parameters_true --sync-batchnorm true
   355	--logger wandb --checkpoint-every 2500
   356	--name FLAC_exp07_BF --experiment-name exp07_BF --save-dir outputs_FLAC/exp07_BF""".split()
   357	# Flags whose VALUE may differ from exp_07 (identity, budget, rung, resume):
   358	ALLOWED_DIFF = {"--model-config", "--name", "--experiment-name", "--save-dir", "--max-steps",
   359	                "--num-gpus", "--batch-size", "--logger", "--checkpoint-every", "--ckpt-path"}
   360	# Flags exp_07 left to defaults.ini and we state explicitly — whitelisted with their
   361	# EXACT expected values (round-3 N9: no "equals the mutable ini" escape hatch):
   362	ALLOWED_ADD = {"--num-nodes": "1", "--precision": "bf16-mixed", "--val-every": "-1",
   363	               "--val-dataset-config": "", "--gradient-clip-val": "0.0", "--ckpt-path": None}
   364	tokens = [t for t in open(sys.argv[1]).read().split("\n")]
   365	if tokens and tokens[-1] == "":
   366	    tokens.pop()
   367	mode = sys.argv[2]
   368	
   369	def as_map(toks):
   370	    out, i = {}, 0
   371	    while i < len(toks):
   372	        flag = toks[i]
   373	        if not flag.startswith("--"):
   374	            raise SystemExit(f"ARGV PARITY: stray token {flag!r}")
   375	        val = toks[i + 1] if i + 1 < len(toks) and not toks[i + 1].startswith("--") else ""
   376	        if flag in out:
   377	            raise SystemExit(f"ARGV PARITY: duplicate flag {flag}")
   378	        out[flag] = val
   379	        i += 2 if (i + 1 < len(toks) and not toks[i + 1].startswith("--")) else 1
   380	    return out
   381	
   382	ref, new = as_map(REF), as_map(tokens)
   383	violations, allowed, explicit = [], [], []
   384	for flag in sorted(set(ref) | set(new)):
   385	    if flag in ref and flag in new:
   386	        if ref[flag] != new[flag]:
   387	            (allowed if flag in ALLOWED_DIFF else violations).append(
   388	                f"{flag}: exp_07 {ref[flag]!r} -> exp_11 {new[flag]!r}")
   389	    elif flag in new:
   390	        if flag == "--ckpt-path":
   391	            (allowed if mode == "RESTART" else violations).append(
   392	                f"--ckpt-path: {new[flag]!r} (RESTART only)")
   393	        elif flag in ALLOWED_ADD and ALLOWED_ADD[flag] == new[flag]:
   394	            explicit.append(f"{flag}={new[flag]!r} (whitelisted explicit default)")
   395	        else:
   396	            violations.append(f"{flag}: added with {new[flag]!r}, not a whitelisted addition "
   397	                              f"(expected {ALLOWED_ADD.get(flag, '<not allowed>')!r})")
   398	    else:
   399	        violations.append(f"{flag}: present in exp_07 ({ref[flag]!r}), MISSING here")
   400	
   401	print("--- train.py argv ---")
   402	print(" ".join(f"{k} {v!r}" if v == "" else f"{k} {v}" for k, v in new.items()))
   403	print("--- argv parity vs exp_07 B-F ---")
   404	for d in allowed:
   405	    print(f"  allowed  {d}")
   406	for d in explicit:
   407	    print(f"  explicit {d}")
   408	if violations:
   409	    print("ARGV PARITY VIOLATIONS:")
   410	    for v in violations:
   411	        print(f"  !! {v}")
   412	    raise SystemExit(2)
   413	print(f"ARGV PARITY OK ({mode}): only whitelisted differences and additions")
   414	PY
   415	parity=$?
   416	rm -f "$ARGV_FILE"
   417	[ "$parity" -eq 0 ] || die "argv parity check FAILED - abort"
   418	if [ "$DRYRUN" = "1" ]; then
   419	  echo "DRY RUN complete: gates A–G passed for ARM=${ARM} RUNG=${RUNG} MODE=${MODE} SMOKE=${SMOKE}"
   420	  echo "  (Slurm/GPU/VRAM/env/wandb/ViT/lock gates and training are skipped in DRYRUN)"

exec
/usr/bin/bash -lc "nl -ba worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_launch.sh | sed -n '1,360p'" in /n/fs/gatrdp/codespace/FLAC
 succeeded in 0ms:
nl: worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_launch.sh: No such file or directory

codex
One immediate discrepancy is that the live checkout has no `fa_orbit_launch.sh` at the referenced path. I’m checking whether it existed at the submitted SHA, was renamed, or is merely a stale closure entry, and separately tracing the actual execution path and manifest/registry reads.
exec
/usr/bin/bash -lc "rg --files worklog/worklog_yixun/exp_11_fa_orbit_claude | rg '(fa_orbit.*\\.(sh|py|sbatch)|FLAC_AR_.*\\.json|manifest|registry)' | sort" in /n/fs/gatrdp/codespace/FLAC
 succeeded in 0ms:
/usr/bin/bash: line 1: rg: command not found
/usr/bin/bash: line 1: rg: command not found

exec
/usr/bin/bash -lc "nl -ba worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch | sed -n '245,620p'" in /n/fs/gatrdp/codespace/FLAC
 succeeded in 0ms:
   245	    echo "commit binding OK (content): training surfaces identical, EXPECT_SHA ${EXPECT_SHA} HEAD ${HEAD_SHA}"
   246	  fi
   247	  [ -z "$DRIFT" ] || { echo "tracked measurement surfaces modified since review - abort:"; echo "$DRIFT"; exit 2; }
   248	fi
   249	
   250	# --- D. arm -> config (single source) + semantic gate -------------------------
   251	arm_config_for() {
   252	  case "$1" in
   253	    C4L|C8|C16|C32) echo "$EXPDIR/FLAC_AR_BF_$1.json" ;;
   254	    VANL)           echo "$EXPDIR/FLAC_AR_VANCKPT.json" ;;
   255	    *) return 1 ;;
   256	  esac
   257	}
   258	MODEL_CONFIG="$(arm_config_for "$ARM")" || die "no config mapped for arm '${ARM}' - abort"
   259	MODEL_CONFIG_ABS="$(readlink -f "$MODEL_CONFIG" 2>/dev/null)"
   260	[ -n "$MODEL_CONFIG_ABS" ] && [ -f "$MODEL_CONFIG_ABS" ] || die "arm config '${MODEL_CONFIG}' does not exist - abort"
   261	CONFIG_SHA="$(sha256sum "$MODEL_CONFIG_ABS" | awk '{print $1}')"
   262	echo "config for ${ARM}: ${MODEL_CONFIG_ABS} sha256 ${CONFIG_SHA}"
   263	
   264	python3 - "$MODEL_CONFIG_ABS" "$ARM" <<'PY' || die "arm/config semantic gate FAILED - abort"
   265	import json, sys
   266	cfg = json.load(open(sys.argv[1])); arm = sys.argv[2]
   267	t = cfg.get("training", {}); bad = []
   268	# VANL is the same recipe with the conditioning removed, so its gate is the
   269	# MIRROR IMAGE of the orbit arms': the orbit keys must be ABSENT, not merely
   270	# different. A vanilla config that carried a stray frame_avg_angles would be a
   271	# silently fa-flavoured baseline, which would destroy the single-delta claim.
   272	if arm == "VANL":
   273	    cm = t.get("cond_method")
   274	    if cm not in (None, "vanilla"):
   275	        bad.append(f"cond_method={cm!r} (want absent or 'vanilla')")
   276	    if "frame_avg_angles" in t:
   277	        bad.append(f"frame_avg_angles is present ({t['frame_avg_angles']!r}) — a vanilla arm has no orbit")
   278	    want = None
   279	else:
   280	    want = {"C4L": 4, "C8": 8, "C16": 16, "C32": 32}[arm]
   281	    angles = t.get("frame_avg_angles")
   282	    if t.get("cond_method") != "fa_invariant":
   283	        bad.append(f"cond_method={t.get('cond_method')!r} (want fa_invariant)")
   284	    if not isinstance(angles, list) or len(angles) != want:
   285	        bad.append(f"frame_avg_angles has {angles and len(angles)} entries (want {want})")
   286	    elif angles != [k * 360.0 / want for k in range(want)]:
   287	        bad.append(f"frame_avg_angles are not the uniform C{want} orbit")
   288	if t.get("use_ema") is not True:
   289	    bad.append(f"use_ema={t.get('use_ema')!r} (want True)")
   290	vits = [c for c in cfg["model"]["conditioning"]["configs"] if c["type"] == "ViTCoordinates"]
   291	if sorted(c["id"] for c in vits) != ["context_poses_vit", "source_vit"]:
   292	    bad.append(f"ViT conditioner ids {sorted(c['id'] for c in vits)} != the expected two")
   293	# Post-P0: grad-ckpt ON for every arm; the KEY must exist and be literally True
   294	for c in vits:
   295	    if "gradient_checkpointing" not in c["config"]:
   296	        bad.append(f"{c['id']}: gradient_checkpointing key absent (want literal true)")
   297	    elif c["config"]["gradient_checkpointing"] is not True:
   298	        bad.append(f"{c['id']}: gradient_checkpointing={c['config']['gradient_checkpointing']!r} (want True)")
   299	if bad:
   300	    sys.exit("ARM/CONFIG GATE: " + "; ".join(bad))
   301	if arm == "VANL":
   302	    print(f"gate OK: {arm} is vanilla (no cond_method, no orbit), grad-ckpt True, EMA on")
   303	else:
   304	    print(f"gate OK: {arm} carries the uniform C{want} orbit, grad-ckpt True, EMA on")
   305	PY
   306	
   307	# --- E. lineage: INITIAL vs RESTART -------------------------------------------
   308	SAVEDIR_REAL="$(realpath -m "$SAVEDIR")"
   309	CKPT_DIR_REAL="$(realpath -m "${RUNDIR}/checkpoints")"
   310	LAUNCH_MANIFEST_LINK="${SAVEDIR}/launch_manifest.txt"     # written by the INITIAL launch
   311	if [ "$EXPECTED_STEP" -eq 0 ]; then
   312	  MODE="INITIAL"
   313	  [ -z "$RESUME_CKPT" ] || die "INITIAL launch must not carry RESUME_CKPT (set EXPECTED_STEP > 0 to declare a RESTART) - abort"
   314	  [ ! -e "$RUNDIR" ] || die "run directory ${RUNDIR} already exists — an INITIAL launch never clobbers a previous run - abort"
   315	else
   316	  MODE="RESTART"
   317	  [ -n "$RESUME_CKPT" ] || die "EXPECTED_STEP ${EXPECTED_STEP} declares a RESTART, but RESTART requires RESUME_CKPT - abort"
   318	  [ -f "$RESUME_CKPT" ] || die "RESUME_CKPT not found: ${RESUME_CKPT} - abort"
   319	  RESUME_REAL="$(realpath -m "$RESUME_CKPT")"
   320	  # exactly this arm's own checkpoints directory — not merely somewhere below the save root
   321	  case "$RESUME_REAL" in
   322	    "${CKPT_DIR_REAL}"/*.ckpt) ;;
   323	    *) die "a RESTART may only resume a checkpoint from ${CKPT_DIR_REAL}/ (got ${RESUME_REAL}) - abort" ;;
   324	  esac
   325	  [ "$MAXSTEPS" -gt "$EXPECTED_STEP" ] || die "MAXSTEPS ${MAXSTEPS} must exceed the resume step ${EXPECTED_STEP} - abort"
   326	fi
   327	echo "lineage: ${MODE} (expected_step ${EXPECTED_STEP}, max_steps ${MAXSTEPS}, ckpt every ${CHECKPOINT_EVERY}, time pin ${TIME_PIN_NAME}=${TIME_LIMIT})"
   328	
   329	# --- F. the exact train.py argv ----------------------------------------------
   330	ARGV=(
   331	  --model-config "$MODEL_CONFIG_ABS"
   332	  --dataset-config src/configs/dataset_configs/AR/train/acousticroom_train.json
   333	  --pretransform-ckpt-path weights/FLAC/VAE.safetensors
   334	  --max-steps "$MAXSTEPS" --batch-size "$MB" --accum-batches 1 --num-workers 6 --seed 42
   335	  --num-gpus "$NGPU" --num-nodes 1
   336	  --strategy ddp_find_unused_parameters_true --sync-batchnorm true --precision bf16-mixed
   337	  --val-every -1 --val-dataset-config ''
   338	  --gradient-clip-val 0.0
   339	  --logger wandb --checkpoint-every "$CHECKPOINT_EVERY"
   340	  --name "$NAME" --experiment-name "$EXPNAME" --save-dir "$SAVEDIR"
   341	)
   342	[ "$MODE" = "RESTART" ] && ARGV+=(--ckpt-path "$RESUME_CKPT")
   343	
   344	# --- G. argv-parity dry run (plan N13; round-3 N9 tightened) ------------------
   345	ARGV_FILE="$(mktemp)" || die "mktemp failed - abort" 3
   346	printf '%s\n' "${ARGV[@]}" > "$ARGV_FILE" || die "could not write the argv file - abort" 3
   347	python3 - "$ARGV_FILE" "$MODE" <<'PY'
   348	import sys
   349	# The exp_07 B-F reference argv (bf_scratch_launch.sh) — the lineage this sweep continues.
   350	REF = """--model-config worklog/worklog_yixun/exp_07_fa_scratch_claude/FLAC_AR_BF.json
   351	--dataset-config src/configs/dataset_configs/AR/train/acousticroom_train.json
   352	--pretransform-ckpt-path weights/FLAC/VAE.safetensors
   353	--max-steps 67500 --batch-size 32 --accum-batches 1 --num-workers 6 --seed 42
   354	--num-gpus 2 --strategy ddp_find_unused_parameters_true --sync-batchnorm true
   355	--logger wandb --checkpoint-every 2500
   356	--name FLAC_exp07_BF --experiment-name exp07_BF --save-dir outputs_FLAC/exp07_BF""".split()
   357	# Flags whose VALUE may differ from exp_07 (identity, budget, rung, resume):
   358	ALLOWED_DIFF = {"--model-config", "--name", "--experiment-name", "--save-dir", "--max-steps",
   359	                "--num-gpus", "--batch-size", "--logger", "--checkpoint-every", "--ckpt-path"}
   360	# Flags exp_07 left to defaults.ini and we state explicitly — whitelisted with their
   361	# EXACT expected values (round-3 N9: no "equals the mutable ini" escape hatch):
   362	ALLOWED_ADD = {"--num-nodes": "1", "--precision": "bf16-mixed", "--val-every": "-1",
   363	               "--val-dataset-config": "", "--gradient-clip-val": "0.0", "--ckpt-path": None}
   364	tokens = [t for t in open(sys.argv[1]).read().split("\n")]
   365	if tokens and tokens[-1] == "":
   366	    tokens.pop()
   367	mode = sys.argv[2]
   368	
   369	def as_map(toks):
   370	    out, i = {}, 0
   371	    while i < len(toks):
   372	        flag = toks[i]
   373	        if not flag.startswith("--"):
   374	            raise SystemExit(f"ARGV PARITY: stray token {flag!r}")
   375	        val = toks[i + 1] if i + 1 < len(toks) and not toks[i + 1].startswith("--") else ""
   376	        if flag in out:
   377	            raise SystemExit(f"ARGV PARITY: duplicate flag {flag}")
   378	        out[flag] = val
   379	        i += 2 if (i + 1 < len(toks) and not toks[i + 1].startswith("--")) else 1
   380	    return out
   381	
   382	ref, new = as_map(REF), as_map(tokens)
   383	violations, allowed, explicit = [], [], []
   384	for flag in sorted(set(ref) | set(new)):
   385	    if flag in ref and flag in new:
   386	        if ref[flag] != new[flag]:
   387	            (allowed if flag in ALLOWED_DIFF else violations).append(
   388	                f"{flag}: exp_07 {ref[flag]!r} -> exp_11 {new[flag]!r}")
   389	    elif flag in new:
   390	        if flag == "--ckpt-path":
   391	            (allowed if mode == "RESTART" else violations).append(
   392	                f"--ckpt-path: {new[flag]!r} (RESTART only)")
   393	        elif flag in ALLOWED_ADD and ALLOWED_ADD[flag] == new[flag]:
   394	            explicit.append(f"{flag}={new[flag]!r} (whitelisted explicit default)")
   395	        else:
   396	            violations.append(f"{flag}: added with {new[flag]!r}, not a whitelisted addition "
   397	                              f"(expected {ALLOWED_ADD.get(flag, '<not allowed>')!r})")
   398	    else:
   399	        violations.append(f"{flag}: present in exp_07 ({ref[flag]!r}), MISSING here")
   400	
   401	print("--- train.py argv ---")
   402	print(" ".join(f"{k} {v!r}" if v == "" else f"{k} {v}" for k, v in new.items()))
   403	print("--- argv parity vs exp_07 B-F ---")
   404	for d in allowed:
   405	    print(f"  allowed  {d}")
   406	for d in explicit:
   407	    print(f"  explicit {d}")
   408	if violations:
   409	    print("ARGV PARITY VIOLATIONS:")
   410	    for v in violations:
   411	        print(f"  !! {v}")
   412	    raise SystemExit(2)
   413	print(f"ARGV PARITY OK ({mode}): only whitelisted differences and additions")
   414	PY
   415	parity=$?
   416	rm -f "$ARGV_FILE"
   417	[ "$parity" -eq 0 ] || die "argv parity check FAILED - abort"
   418	if [ "$DRYRUN" = "1" ]; then
   419	  echo "DRY RUN complete: gates A–G passed for ARM=${ARM} RUNG=${RUNG} MODE=${MODE} SMOKE=${SMOKE}"
   420	  echo "  (Slurm/GPU/VRAM/env/wandb/ViT/lock gates and training are skipped in DRYRUN)"
   421	  exit 0
   422	fi
   423	
   424	# --- H. Slurm allocation must match the pins (round-3 B1) ---------------------
   425	[ "${SLURM_JOB_NUM_NODES:-1}" = "1" ] || die "expected 1 node, got ${SLURM_JOB_NUM_NODES} - abort"
   426	[ "${SLURM_NTASKS:-1}" = "1" ] || die "expected 1 task, got ${SLURM_NTASKS} - abort"
   427	WANT_CPUS="$((8 + 7 * NGPU))"; WANT_MEM_MB="$(((12 * NGPU + 12) * 1024))"
   428	GOT_CPUS="${SLURM_CPUS_PER_TASK:-${SLURM_CPUS_ON_NODE:-0}}"
   429	GOT_MEM_MB="${SLURM_MEM_PER_NODE:-0}"
   430	[ "$GOT_CPUS" = "$WANT_CPUS" ] || die "allocated ${GOT_CPUS} CPUs, the pinned rung needs ${WANT_CPUS} — submit via fa_orbit_submit.sh - abort"
   431	[ "$GOT_MEM_MB" = "$WANT_MEM_MB" ] || die "allocated ${GOT_MEM_MB} MB RAM, the pinned rung needs ${WANT_MEM_MB} — submit via fa_orbit_submit.sh - abort"
   432	GOT_TIME="$(squeue -h -j "$SLURM_JOB_ID" -o %l 2>/dev/null | tr -d ' ')"
   433	norm_minutes() { awk -v t="$1" 'BEGIN{d=0; if (t ~ /-/) {split(t,p,"-"); d=p[1]; t=p[2]} n=split(t,c,":");
   434	  if (n==3) m=c[1]*60+c[2]+c[3]/60; else if (n==2) m=c[1]+c[2]/60; else m=t; printf "%d", d*1440+m}'; }
   435	# The pin this ${MODE} leg is entitled to — an INITIAL allocation handed to a
   436	# RESTART leg (or the reverse) is refused here, in the job, not merely intended
   437	# by the submitter.
   438	[ "$(norm_minutes "$GOT_TIME")" = "$(norm_minutes "$TIME_LIMIT")" ] || die "allocated time ${GOT_TIME} != the ${TIME_PIN_NAME} pin ${TIME_LIMIT} this ${MODE} leg requires — submit via fa_orbit_submit.sh - abort"
   439	echo "allocation matches the pins: ${GOT_CPUS} cpus, ${GOT_MEM_MB} MB, ${GOT_TIME} (${TIME_PIN_NAME})"
   440	
   441	mapfile -t GPU_ROWS < <(nvidia-smi --query-gpu=uuid,name --format=csv,noheader,nounits)
   442	[ "${#GPU_ROWS[@]}" -eq "$NGPU" ] || die "rung ${RUNG} needs exactly ${NGPU} visible GPU(s), got ${#GPU_ROWS[@]} - abort"
   443	UUIDS=()
   444	for ROW in "${GPU_ROWS[@]}"; do
   445	  U="$(echo "$ROW" | cut -d, -f1 | tr -d ' ')"; N="$(echo "$ROW" | cut -d, -f2 | sed 's/^ *//')"
   446	  case "$N" in *L40*) ;; *) die "allocated GPU '${N}' is not an L40 - abort";; esac
   447	  UUIDS+=("$U"); echo "GPU: ${N} ${U}"
   448	done
   449	UUID_CSV="$(IFS=,; echo "${UUIDS[*]}")"
   450	
   451	# --- I. environment identity gates (round-3 B6) -------------------------------
   452	REAL_PY="$(command -v python)"
   453	[ "$(readlink -f "$REAL_PY")" = "$(readlink -f "$PINNED_PYTHON")" ] || die "python resolves to ${REAL_PY}, pinned ${PINNED_PYTHON} - abort"
   454	python3 - "$PINNED_PL_VERSION" "$PINNED_TORCH_VERSION" <<'PY' || die "runtime version gate FAILED - abort"
   455	import sys
   456	import pytorch_lightning as pl, torch
   457	want_pl, want_torch = sys.argv[1], sys.argv[2]
   458	bad = []
   459	if pl.__version__ != want_pl:
   460	    bad.append(f"pytorch_lightning {pl.__version__} != pinned {want_pl}")
   461	if torch.__version__ != want_torch:
   462	    bad.append(f"torch {torch.__version__} != pinned {want_torch}")
   463	if bad:
   464	    sys.exit("; ".join(bad))
   465	print(f"runtime gate OK: pl {pl.__version__}, torch {torch.__version__}, cuda {torch.version.cuda}")
   466	PY
   467	VAE_SHA="$(sha256sum weights/FLAC/VAE.safetensors | awk '{print $1}')"
   468	[ "$VAE_SHA" = "$PINNED_VAE_SHA256" ] || die "weights/FLAC/VAE.safetensors sha256 ${VAE_SHA} != pinned ${PINNED_VAE_SHA256} - abort"
   469	echo "VAE pin OK: ${VAE_SHA}"
   470	DRIVER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
   471	CUDA_VER="$(python3 -c 'import torch; print(torch.version.cuda)')"
   472	
   473	# --- J. per-GPU free-VRAM gate (floor from the pins) --------------------------
   474	for U in "${UUIDS[@]}"; do
   475	  FREE="$(nvidia-smi --id="$U" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | tr -dc '0-9')"
   476	  [ -n "$FREE" ] || die "nvidia-smi free-mem query failed on ${U} - refusing to launch blind"
   477	  [ "$FREE" -ge "$MIN_FREE_MB" ] || die "GPU ${U} free ${FREE} MiB < required ${MIN_FREE_MB} MiB - refusing to launch"
   478	done
   479	echo "--- co-tenancy disclosure at launch ---"
   480	nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true
   481	
   482	# --- K. exclusive run ownership via flock (round-3 B3 residual) --------------
   483	# mkdir + stale recovery had two races: a contender could arrive between mkdir
   484	# and the owner write, and release removed the directory without checking whose
   485	# it was. flock has neither: the kernel holds the lock while the fd is open and
   486	# releases it on close (including on kill), so there is no stale state to
   487	# recover and no recovery path to get wrong.
   488	mkdir -p "$OUTPUT_ROOT" || die "could not create ${OUTPUT_ROOT} - abort" 3
   489	LOCKFILE="${OUTPUT_ROOT}/exp11_${ARM}.lock"
   490	exec 9>"$LOCKFILE" || die "could not open the lock file ${LOCKFILE} - abort" 3
   491	if ! flock -n 9; then
   492	  OWNER="$(tr '\n' ' ' < "$LOCKFILE" 2>/dev/null)"
   493	  die "arm ${ARM} is locked by another live job (${OWNER:-<no metadata>}) - refusing a concurrent writer - abort"
   494	fi
   495	LAUNCH_UUID="$(cat /proc/sys/kernel/random/uuid)"
   496	{ echo "job ${SLURM_JOB_ID}"; echo "uuid ${LAUNCH_UUID}"; echo "arm ${ARM}"; echo "mode ${MODE}"; echo "acquired ${TS}"; } >&9 \
   497	  || die "could not write the lock owner metadata - abort" 3
   498	echo "lock acquired: ${LOCKFILE} (flock on fd 9, released on exit)"
   499	mkdir -p "$SAVEDIR" || die "could not create ${SAVEDIR} - abort" 3
   500	
   501	# --- L. RESTART preflight (round-3 B2) ---------------------------------------
   502	CKPT_SHA=""
   503	if [ "$MODE" = "RESTART" ]; then
   504	  PRE_ARGS=(--ckpt "$RESUME_CKPT" --expected-step "$EXPECTED_STEP" --config "$MODEL_CONFIG_ABS"
   505	            --max-steps "$MAXSTEPS" --arm "$ARM" --rung "$RUNG" --commit "$HEAD_SHA")
   506	  [ -n "$LAUNCH_MANIFEST_LINK" ] && PRE_ARGS+=(--launch-manifest "$LAUNCH_MANIFEST_LINK")
   507	  # Q10 / re-pin fix 1: a real arm's restart is the 40k -> 100k EXTENSION, whose
   508	  # contract binds the ORIGINAL launch identity (audited manifest bytes, job,
   509	  # uuid, launch commit, config, save-dir, seed, and the 40k anchor itself)
   510	  # without demanding that the INITIAL budget/commit equal this leg's. SMOKE
   511	  # restarts have no registered launch and keep the ordinary contract.
   512	  [ "$SMOKE" != "1" ] && PRE_ARGS+=(--extension --launch-registry "$EXPDIR/arm_launch_registry.json"
   513	                                    --repo-root "$REPO")
   514	  PRE_OUT="$(python3 "$EXPDIR/fa_orbit_ckpt_preflight.py" "${PRE_ARGS[@]}" 2>&1)"
   515	  echo "$PRE_OUT"
   516	  echo "$PRE_OUT" | grep -q "^CKPT_SHA256 " || die "restart preflight FAILED - abort"
   517	  CKPT_SHA="$(echo "$PRE_OUT" | awk '/^CKPT_SHA256 /{print $2}')"
   518	fi
   519	
   520	# --- M. wandb: scrub, pin the destination, fix the run id (round-3 B7) --------
   521	unset WANDB_MODE WANDB_DISABLED WANDB_ENTITY WANDB_RUN_ID WANDB_RESUME WANDB_DIR WANDB_PROJECT WANDB_NAME
   522	export WANDB_DIR="$REPO/$SAVEDIR"
   523	export WANDB_MODE=online
   524	eval "$(grep -E '^[[:space:]]*export[[:space:]]+WANDB_API_KEY=' ~/.bashrc 2>/dev/null | tail -1)"
   525	WANDB_ENTITY_SEEN="$(python3 - <<'PY'
   526	import sys
   527	try:
   528	    import wandb
   529	    v = wandb.Api().viewer
   530	except Exception as e:
   531	    sys.exit(f"wandb identity check FAILED: {e}")
   532	if v.email != "yh4742@princeton.edu":
   533	    sys.exit(f"wandb identity {v.email} != yh4742@princeton.edu")
   534	print(v.entity)
   535	PY
   536	)" || die "wandb identity gate FAILED (no logger fallback: the arms train with wandb) - abort"
   537	[ -n "$WANDB_ENTITY_SEEN" ] || die "wandb returned an empty entity - abort"
   538	# B7 residual: pin the destination account explicitly instead of leaving it implicit
   539	export WANDB_ENTITY="$WANDB_ENTITY_SEEN"
   540	echo "wandb identity OK: entity ${WANDB_ENTITY} (exported), mode ${WANDB_MODE}"
   541	if [ "$MODE" = "RESTART" ]; then
   542	  WANDB_RUN_ID="$(awk '/^wandb_run_id /{print $2}' "$LAUNCH_MANIFEST_LINK" 2>/dev/null)"
   543	  [ -n "$WANDB_RUN_ID" ] || die "RESTART needs the original wandb run id from ${LAUNCH_MANIFEST_LINK} - abort"
   544	  export WANDB_RUN_ID WANDB_RESUME=must
   545	  echo "wandb: resuming run ${WANDB_RUN_ID} (resume=must)"
   546	else
   547	  WANDB_RUN_ID="exp11-${ARM}-$(date +%s%N)-$(cut -c1-8 /proc/sys/kernel/random/uuid)"
   548	  export WANDB_RUN_ID
   549	  echo "wandb: new run id ${WANDB_RUN_ID}"
   550	fi
   551	
   552	# --- N. DINOv3 pin + init-identity gate (inside the allocation) ---------------
   553	HF_HUB_OFFLINE=1 python3 "$EXPDIR/assert_arm_configs_exp11.py" "$ARM" || die "ViT-pin / init-identity GATE FAILED - abort" 1
   554	
   555	# --- O. atomic manifest, duplicated to the save-dir (round-3 B5) --------------
   556	# --- the live transcript must not be a TRACKED file while it is being written --
   557	# A tracked file that a running job appends to is a file git will happily
   558	# replace. `git rebase --autostash` (remote coordination between machines) stashes
   559	# the dirty working tree, checks out, and restores — and each of those steps
   560	# UNLINKS the path and writes a new inode. The job's stdout descriptor still
   561	# points at the old, now-nameless inode, so every subsequent line is written to a
   562	# file with no name and the visible transcript freezes forever. That is exactly
   563	# what happened to C16 (job 3648696): its transcript stops at Epoch 5 while the
   564	# run went on to step 40000, and all four arm transcripts froze at the same
   565	# instant, 02:04:07, during a rebase cycle.
   566	#
   567	# So: at launch, the job removes its OWN Slurm transcript from the index. The
   568	# file stays on disk untouched; it is simply no longer something git will move.
   569	# The operator commits completed transcripts at closure (git add -f).
   570	SLURM_OUT_AT_LAUNCH="$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null \
   571	                        | tr ' ' '\n' | awk -F= '$1=="StdOut"{print $2; exit}')"
   572	UNTRACK_STATE="not-attempted"
   573	if [ -n "$SLURM_OUT_AT_LAUNCH" ]; then
   574	  if git -C "$REPO" ls-files --error-unmatch "$SLURM_OUT_AT_LAUNCH" >/dev/null 2>&1; then
   575	    if git -C "$REPO" rm --cached --quiet -- "$SLURM_OUT_AT_LAUNCH" 2>/dev/null; then
   576	      UNTRACK_STATE="untracked-at-launch"
   577	      echo "live transcript untracked for the duration of this run: ${SLURM_OUT_AT_LAUNCH}"
   578	      echo "  (the file is untouched on disk; commit it at closure with git add -f)"
   579	    else
   580	      UNTRACK_STATE="untrack-FAILED"
   581	      echo "WARNING: could not untrack ${SLURM_OUT_AT_LAUNCH}; a git working-tree"
   582	      echo "         operation during this run can still detach the transcript"
   583	    fi
   584	  else
   585	    UNTRACK_STATE="already-untracked"      # the normal steady state
   586	  fi
   587	else
   588	  UNTRACK_STATE="stdout-path-unknown"
   589	fi
   590	
   591	TRAINLOG="${EXPDIR}/fa_orbit_${TS}_${ARM}_${RUNG}_jid${SLURM_JOB_ID}_train.log"
   592	SAVEDIR_LOG="${SAVEDIR}/fa_orbit_${TS}_${ARM}_${RUNG}_jid${SLURM_JOB_ID}_train.log"
   593	MANIFEST="${EXPDIR}/fa_orbit_${TS}_${ARM}_${RUNG}_jid${SLURM_JOB_ID}_manifest.txt"
   594	# B5 residual: a failed environment dump must not be silently hashed into the record
   595	PIPFREEZE_FILE="$(mktemp)" || die "mktemp failed - abort" 3
   596	pip freeze > "$PIPFREEZE_FILE" 2>/dev/null || die "pip freeze FAILED - the environment digest would be a lie - abort" 7
   597	ENV_SHA="$(sha256sum "$PIPFREEZE_FILE" | awk '{print $1}')"
   598	rm -f "$PIPFREEZE_FILE"
   599	{
   600	  echo "# exp_11 arm launch manifest"
   601	  echo "timestamp ${TS}"
   602	  echo "job ${SLURM_JOB_ID} host $(hostname) mode ${MODE} launch_uuid ${LAUNCH_UUID}"
   603	  echo "arm ${ARM} rung ${RUNG} micro ${MB} ngpu ${NGPU} max_steps ${MAXSTEPS} ckpt_every ${CHECKPOINT_EVERY}"
   604	  echo "commit ${HEAD_SHA}"
   605	  echo "p0_manifest_sha256 ${PINNED_P0_MANIFEST_SHA256}"
   606	  echo "model_config ${MODEL_CONFIG_ABS}"
   607	  echo "config_sha256 ${CONFIG_SHA}"
   608	  echo "vae_sha256 ${VAE_SHA}"
   609	  echo "env_pip_freeze_sha256 ${ENV_SHA}"
   610	  echo "torch_version $(python3 -c 'import torch; print(torch.__version__)') cuda ${CUDA_VER} driver ${DRIVER}"
   611	  echo "gpu_uuids ${UUID_CSV}"
   612	  echo "time_limit ${TIME_LIMIT} min_free_mb ${MIN_FREE_MB}"
   613	  echo "resume_ckpt ${RESUME_CKPT:-<none>} expected_step ${EXPECTED_STEP} resume_ckpt_sha256 ${CKPT_SHA:-<none>}"
   614	  echo "save_dir ${SAVEDIR}"
   615	  echo "slurm_transcript ${SLURM_OUT_AT_LAUNCH:-<unknown>} untrack ${UNTRACK_STATE}"
   616	  echo "train_log ${TRAINLOG}"
   617	  echo "train_log_copy ${SAVEDIR_LOG}"
   618	  echo "wandb_entity ${WANDB_ENTITY_SEEN} wandb_project ${NAME} wandb_name ${EXPNAME}"
   619	  echo "wandb_run_id ${WANDB_RUN_ID}"
   620	  echo "command torchrun --standalone --nnodes=1 --nproc_per_node=${NGPU} train.py ${ARGV[*]}"

exec
/usr/bin/bash -lc "rg -n \"fa_orbit_launch|source |bash |python|python3|registry|manifest|\\.json|defaults\\.ini|train\\.py|torchrun|readlink|sha256sum|cp |tee |srun\" worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submit.sh" in /n/fs/gatrdp/codespace/FLAC
 exited 127 in 0ms:
/usr/bin/bash: line 1: rg: command not found

exec
/usr/bin/bash -lc "git log --all --name-status --format='COMMIT %H %s' -- worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_launch.sh | sed -n '1,200p'" in /n/fs/gatrdp/codespace/FLAC
 succeeded in 3098ms:
2026-08-11T19:00:06.806855Z ERROR codex_models_manager::manager: failed to renew cache TTL: missing field `base_instructions` at line 94 column 5
exec
/usr/bin/bash -lc "nl -ba worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch | sed -n '620,860p'" in /n/fs/gatrdp/codespace/FLAC
 succeeded in 0ms:
   620	  echo "command torchrun --standalone --nnodes=1 --nproc_per_node=${NGPU} train.py ${ARGV[*]}"
   621	} > "${MANIFEST}.tmp" || die "manifest write FAILED - abort" 3
   622	mv "${MANIFEST}.tmp" "$MANIFEST" || die "manifest publication FAILED - abort" 3
   623	cp "$MANIFEST" "${SAVEDIR}/$(basename "$MANIFEST")" || die "manifest copy to the save-dir FAILED - abort" 3
   624	[ "$MODE" = "INITIAL" ] && { cp "$MANIFEST" "$LAUNCH_MANIFEST_LINK" || die "launch-manifest link write FAILED - abort" 3; }
   625	echo "manifest: ${MANIFEST} (copied to ${SAVEDIR})"
   626	
   627	# --- P. training: torchrun + FIFO tee (both statuses captured) ----------------
   628	unset SLURM_NTASKS SLURM_JOB_NAME SLURM_PROCID SLURM_LOCALID SLURM_NODEID
   629	: > "$TRAINLOG" || die "could not create ${TRAINLOG} - abort" 3
   630	: > "$SAVEDIR_LOG" || die "could not create ${SAVEDIR_LOG} - abort" 3
   631	FIFO="$(mktemp)" || die "mktemp failed - abort" 3
   632	rm -f "$FIFO" && mkfifo "$FIFO" || die "mkfifo failed - abort" 3
   633	trap 'rm -f "$FIFO"' EXIT
   634	tee -a "$TRAINLOG" "$SAVEDIR_LOG" < "$FIFO" &
   635	TEE_PID=$!
   636	echo "=== launching ${ARM} ${RUNG}: torchrun --standalone --nproc_per_node=${NGPU} train.py (${MODE}) ==="
   637	torchrun --standalone --nnodes=1 --nproc_per_node="$NGPU" train.py "${ARGV[@]}" > "$FIFO" 2>&1 &
   638	TR_PID=$!
   639	
   640	# world-size watcher: terminate the moment Lightning reports the WRONG rank count
   641	WORLD_RE="All distributed processes registered\. Starting with [0-9]+ processes"
   642	(
   643	  while kill -0 "$TR_PID" 2>/dev/null; do
   644	    if grep -Eq "$WORLD_RE" "$TRAINLOG" 2>/dev/null; then
   645	      GOT="$(grep -Eo "$WORLD_RE" "$TRAINLOG" | head -1 | grep -Eo '[0-9]+')"
   646	      if [ "$GOT" != "$NGPU" ]; then
   647	        echo "WORLD-SIZE WATCHER: Lightning started with ${GOT} processes, expected ${NGPU} — terminating torchrun"
   648	        pkill -TERM -P "$TR_PID" 2>/dev/null; kill -TERM "$TR_PID" 2>/dev/null
   649	      fi
   650	      exit 0
   651	    fi
   652	    sleep 5
   653	  done
   654	) &
   655	WATCHER=$!
   656	
   657	wait "$TR_PID"; rc=$?
   658	kill "$WATCHER" 2>/dev/null; wait "$WATCHER" 2>/dev/null
   659	wait "$TEE_PID"; tee_rc=$?
   660	rm -f "$FIFO"
   661	
   662	# --- Q. W&B run identity verification (round-3 B7 residual) ------------------
   663	# The gate above proves WHO we are; this proves WHERE the run actually landed.
   664	# train.py:165 builds WandbLogger(project=, name=) with NO save_dir, so PL passes
   665	# its default save_dir='.' into wandb.init and that OVERRIDES the exported
   666	# WANDB_DIR: in job 3646734 the run went to $REPO/wandb/run-<ts>-<id> while this
   667	# check looked under $WANDB_DIR/wandb and found nothing (training was green, the
   668	# job still classified 7). We keep exporting WANDB_DIR — other wandb artifacts do
   669	# respect it — but locate the run by the collision-proof id WE generated, which
   670	# wandb embeds in the directory name, across both candidate roots. Exactly one
   671	# match is required; a mismatch is a provenance failure (class 7), not a footnote.
   672	WANDB_CHECK_RC=0
   673	python3 "$EXPDIR/fa_orbit_wandb_readback.py" --run-id "$WANDB_RUN_ID" \
   674	  --root "$REPO" --root "$WANDB_DIR" \
   675	  --entity "$WANDB_ENTITY" --project "$NAME" --name "$EXPNAME" || WANDB_CHECK_RC=$?
   676	if [ "$WANDB_CHECK_RC" -ne 0 ]; then
   677	  echo "W&B run identity could not be verified against the manifest - provenance failure"
   678	fi
   679	
   680	# --- R. classification (tested taxonomy; see fa_orbit_classify.py) ------------
   681	CLASSIFY_OUT="$(python3 "$EXPDIR/fa_orbit_classify.py" --rc "$rc" --tee-rc "$tee_rc" --ngpu "$NGPU" \
   682	  --maxsteps "$MAXSTEPS" --log "$TRAINLOG" --log-copy "$SAVEDIR_LOG" 2>&1)"
   683	final_rc=$?
   684	echo "$CLASSIFY_OUT"
   685	LAST_CKPT="$(ls -1t "${RUNDIR}/checkpoints"/*.ckpt 2>/dev/null | head -1)"
   686	
   687	# --- S. durable final record (round-3 B5 residual) ---------------------------
   688	# The classifier verdict and this summary must survive in BOTH copies; the tee
   689	# status is captured (it used to be discarded) and a failure is class 7. The
   690	# whole preflight transcript (this job's Slurm output so far) is copied into the
   691	# save-dir too, so the durable record is not just torchrun's stdout.
   692	FINAL_RECORD="$(
   693	  echo "=== exp_11 ${ARM} ${RUNG} finished at $(date '+%Y-%m-%d %H:%M:%S') ==="
   694	  echo "torchrun rc=${rc} tee rc=${tee_rc} wandb_identity_rc=${WANDB_CHECK_RC} classified rc=${final_rc}"
   695	  echo "$CLASSIFY_OUT"
   696	  echo "last checkpoint: ${LAST_CKPT:-<none>}"
   697	  echo "restart: fa_orbit_submit.sh ${ARM} --resume ${LAST_CKPT:-<path>} --expected-step <step>"
   698	)"
   699	printf '%s\n' "$FINAL_RECORD" | tee -a "$TRAINLOG" "$SAVEDIR_LOG" > /dev/null
   700	final_tee_rc="${PIPESTATUS[1]}"
   701	# The transcript path is NOT guessed from the #SBATCH pattern (job 3646734 logged
   702	# "slurm_train_job_3646734.out not found" because SLURM_JOB_NAME is unset inside
   703	# the job): ask Slurm itself.
   704	# --- BEGIN slurm-stdout-helper (guard-tested by fa_orbit_equivprobe_wrapper_test.sh) --
   705	parse_slurm_stdout() {   # `scontrol show job` text on stdin -> the StdOut path
   706	  tr ' ' '\n' | awk -F= '$1=="StdOut"{print $2; exit}'
   707	}
   708	slurm_stdout_path() {    # $1 = job id
   709	  scontrol show job "$1" 2>/dev/null | parse_slurm_stdout
   710	}
   711	# --- END slurm-stdout-helper --------------------------------------------------
   712	SLURM_OUT="$(slurm_stdout_path "$SLURM_JOB_ID")"
   713	PREFLIGHT_COPY_RC=0
   714	if [ -n "$SLURM_OUT" ] && [ -f "$SLURM_OUT" ]; then
   715	  cp "$SLURM_OUT" "${SAVEDIR}/preflight_${TS}_jid${SLURM_JOB_ID}.out" || PREFLIGHT_COPY_RC=$?
   716	  [ "$PREFLIGHT_COPY_RC" -eq 0 ] && echo "preflight transcript duplicated from ${SLURM_OUT}"
   717	else
   718	  echo "PREFLIGHT TRANSCRIPT: scontrol reported StdOut='${SLURM_OUT}' which is not a readable file"
   719	  PREFLIGHT_COPY_RC=1
   720	fi
   721	if [ "$final_tee_rc" -ne 0 ] || [ "$PREFLIGHT_COPY_RC" -ne 0 ] || [ "$WANDB_CHECK_RC" -ne 0 ]; then
   722	  echo "LOG-PROVENANCE: final record tee rc=${final_tee_rc}, preflight copy rc=${PREFLIGHT_COPY_RC}, wandb identity rc=${WANDB_CHECK_RC}"
   723	  [ "$final_rc" -eq 0 ] && final_rc=7
   724	fi
   725	printf '%s\n' "$FINAL_RECORD"
   726	exit "$final_rc"

exec
/usr/bin/bash -lc "find worklog/worklog_yixun/exp_11_fa_orbit_claude -maxdepth 1 -type f \\( -name 'fa_orbit*.sh' -o -name 'fa_orbit*.py' -o -name 'fa_orbit*.sbatch' -o -name 'FLAC_AR_*.json' -o -name '*manifest*' -o -name '*registry*' \\) -printf '%f\\n' | sort" in /n/fs/gatrdp/codespace/FLAC
exec
/usr/bin/bash -lc "nl -ba worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submit.sh | sed -n '1,360p'" in /n/fs/gatrdp/codespace/FLAC
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
    12	#   SMOKE=1 SMOKE_RUNG=16x4 SMOKE_MIN_FREE_MB=14000 ./fa_orbit_submit.sh C4L
    13	#   DRYRUN=1 ./fa_orbit_submit.sh C8        # print the sbatch line, submit nothing
    14	#
    15	# Resources per rung (micro x N = 64): --gres=gpu:l40:N, --cpus-per-task=8+7N,
    16	# --mem=(12N+12)G, --time=<the arm's pinned limit>. Each submission is recorded
    17	# in an atomic, no-clobber manifest next to the launcher.
    18	# ============================================================================
    19	set -uo pipefail
    20	cd "$(git -C "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" rev-parse --show-toplevel)" || exit 3
    21	
    22	EXPDIR="worklog/worklog_yixun/exp_11_fa_orbit_claude"
    23	SBATCH_FILE="${EXPDIR}/fa_orbit_train.sbatch"
    24	DRYRUN="${DRYRUN:-0}"
    25	SMOKE="${SMOKE:-0}"
    26	PLACEHOLDER="TO-PIN-AFTER-P0"
    27	
    28	[ -f "$SBATCH_FILE" ] || { echo "missing ${SBATCH_FILE} - abort"; exit 3; }
    29	
    30	ARM="${1:-}"
    31	[ -n "$ARM" ] || { echo "usage: $0 <C4L|C8|C16|C32|VANL> [--resume <ckpt> --expected-step <n>] - abort"; exit 2; }
    32	shift
    33	case "$ARM" in C4L|C8|C16|C32|VANL) ;; *) echo "ARM '${ARM}' must be C4L|C8|C16|C32|VANL - abort"; exit 2;; esac
    34	
    35	RESUME_CKPT=""; EXPECTED_STEP=0
    36	while [ $# -gt 0 ]; do
    37	  case "$1" in
    38	    --resume) RESUME_CKPT="${2:?--resume needs a path}"; shift 2 ;;
    39	    --expected-step) EXPECTED_STEP="${2:?--expected-step needs a number}"; shift 2 ;;
    40	    *) echo "unknown argument '$1' - abort"; exit 2 ;;
    41	  esac
    42	done
    43	case "$EXPECTED_STEP" in ''|*[!0-9]*) echo "--expected-step must be a non-negative integer - abort"; exit 2;; esac
    44	
    45	# --- pins are read FROM the launcher, so submitter and job cannot disagree ----
    46	pin() {  # read one PINNED_* value out of the launcher (quoted or bare)
    47	  awk -v k="$1" '$0 ~ "^"k"=" {
    48	        if (match($0, /"[^"]*"/)) { print substr($0, RSTART + 1, RLENGTH - 2) }
    49	        else { split($0, a, "="); split(a[2], b, " "); print b[1] }
    50	        exit }' "$SBATCH_FILE"
    51	}
    52	if [ "$SMOKE" = "1" ]; then
    53	  RUNG="${SMOKE_RUNG:?SMOKE=1 requires SMOKE_RUNG (32x2|16x4|8x8)}"
    54	  TIME_LIMIT="${SMOKE_TIME:-00:30:00}"
    55	  [ -n "${SMOKE_MIN_FREE_MB:-}" ] || { echo "SMOKE=1 requires SMOKE_MIN_FREE_MB - abort"; exit 2; }
    56	  JOBNAME="exp11-smoke-${ARM}"
    57	else
    58	  RUNG="$(pin PINNED_RUNG)"
    59	  # A RESTART leg is a different budget from the INITIAL one: 60k further steps,
    60	  # not 40k from scratch. Selecting the INITIAL limit for a restart would wall-kill
    61	  # every arm partway through the extension.
    62	  if [ -n "${EXPECTED_STEP:-}" ] && [ "${EXPECTED_STEP:-0}" -gt 0 ]; then
    63	    TIME_LIMIT="$(pin "PINNED_TIME_LIMIT_RESTART_${ARM}")"
    64	  else
    65	    TIME_LIMIT="$(pin "PINNED_TIME_LIMIT_${ARM}")"
    66	  fi
    67	  for V in "$RUNG" "$TIME_LIMIT" "$(pin PINNED_MIN_FREE_MB)" "$(pin PINNED_P0_MANIFEST_SHA256)"; do
    68	    [ "$V" != "$PLACEHOLDER" ] || { echo "the launcher still carries ${PLACEHOLDER} pins: the P0 report has not been pinned yet — no arm may be submitted (use SMOKE=1 for the smoke) - abort"; exit 2; }
    69	  done
    70	  JOBNAME="exp11-${ARM}-train"
    71	fi
    72	case "$RUNG" in 32x2|16x4|8x8) ;; *) echo "rung '${RUNG}' must be 32x2|16x4|8x8 - abort"; exit 2;; esac
    73	MB="${RUNG%x*}"; NGPU="${RUNG#*x}"
    74	[ "$((MB * NGPU))" -eq 64 ] || { echo "rung ${RUNG}: MB*NGPU != 64 - abort"; exit 2; }
    75	
    76	# --- drift gate: a queued job must run reviewed, committed code --------------
    77	# The drift gate is scoped to CODE surfaces, not the whole exp folder: the four
    78	# arms are running and Slurm appends to their tracked *.out logs continuously, so
    79	# a folder-wide check would abort every screen on a live-log write. Configs,
    80	# drivers and validators are still fully covered.
    81	DRIFT="$(git status --porcelain --untracked-files=no -- train.py defaults.ini src \
    82	         "$EXPDIR"/*.json "$EXPDIR"/*.py "$EXPDIR"/*.sbatch "$EXPDIR"/*.sh \
    83	         worklog/worklog_yixun/exp_07_fa_scratch_claude/FLAC_AR_BF.json 2>/dev/null)"
    84	[ -z "$DRIFT" ] || { echo "tracked measurement surfaces have uncommitted changes - commit first, abort:"; echo "$DRIFT"; exit 2; }
    85	SHA="$(git rev-parse HEAD)"
    86	
    87	ARGS=(
    88	  --job-name="$JOBNAME"
    89	  --gres="gpu:l40:${NGPU}"
    90	  --cpus-per-task="$((8 + 7 * NGPU))"
    91	  --mem="$(((12 * NGPU + 12)))G"
    92	  --time="$TIME_LIMIT"
    93	  --export="ALL,ARM=${ARM},EXPECT_SHA=${SHA},OUTPUT_ROOT=outputs_FLAC"
    94	)
    95	[ "$SMOKE" = "1" ] && ARGS[5]="${ARGS[5]},SMOKE=1,SMOKE_RUNG=${SMOKE_RUNG},SMOKE_MIN_FREE_MB=${SMOKE_MIN_FREE_MB},SMOKE_MAXSTEPS=${SMOKE_MAXSTEPS:-30},SMOKE_TIME=${TIME_LIMIT}"
    96	[ -n "$RESUME_CKPT" ] && ARGS[5]="${ARGS[5]},RESUME_CKPT=${RESUME_CKPT},EXPECTED_STEP=${EXPECTED_STEP}"
    97	ARGS+=("$SBATCH_FILE")
    98	
    99	echo "arm ${ARM} | rung ${RUNG} (${MB}x${NGPU}) | time ${TIME_LIMIT} | commit ${SHA} | smoke ${SMOKE}"
   100	if [ "$DRYRUN" = "1" ]; then
   101	  echo "DRYRUN sbatch ${ARGS[*]}"
   102	  exit 0
   103	fi
   104	
   105	# --- NEW-3: publish the INTENT before submitting -----------------------------
   106	# The provenance record must exist before the job can exist, otherwise a local
   107	# write failure leaves a queued job nobody recorded. The intent manifest carries
   108	# the exact command and pins; the job id is appended afterwards, and if that
   109	# append fails the exact job we just created is cancelled.
   110	INTENT_ID="$(date +%s%N)-$(cut -c1-8 /proc/sys/kernel/random/uuid)"
   111	MANIFEST="${EXPDIR}/fa_orbit_submission_${ARM}_${INTENT_ID}.txt"
   112	[ ! -e "$MANIFEST" ] || { echo "submission manifest ${MANIFEST} already exists - abort"; exit 2; }
   113	TMP="$(mktemp "${MANIFEST}.XXXXXX")" || exit 3
   114	{
   115	  echo "# exp_11 arm submission (intent published BEFORE sbatch)"
   116	  echo "intent_id ${INTENT_ID}"
   117	  echo "submitted_at $(date -Is)"
   118	  echo "arm ${ARM} rung ${RUNG} micro ${MB} ngpu ${NGPU}"
   119	  echo "jobname ${JOBNAME} time ${TIME_LIMIT} smoke ${SMOKE}"
   120	  echo "commit ${SHA}"
   121	  echo "pins rung=${RUNG} maxsteps=$(pin PINNED_MAXSTEPS) ckpt_every=$(pin PINNED_CHECKPOINT_EVERY) min_free_mb=$(pin PINNED_MIN_FREE_MB) p0_manifest_sha256=$(pin PINNED_P0_MANIFEST_SHA256)"
   122	  echo "resume ${RESUME_CKPT:-<none>} expected_step ${EXPECTED_STEP}"
   123	  echo "sbatch sbatch ${ARGS[*]}"
   124	} >> "$TMP" || { echo "intent manifest write failed - abort"; exit 3; }
   125	mv -n "$TMP" "$MANIFEST" || { echo "intent manifest publication failed - abort"; exit 2; }
   126	[ -e "$MANIFEST" ] || { echo "intent manifest ${MANIFEST} did not appear - abort"; exit 2; }
   127	echo "intent manifest: ${MANIFEST}"
   128	
   129	OUT="$(sbatch "${ARGS[@]}" 2>&1)"; JID="$(echo "$OUT" | awk '/Submitted batch job/ {print $NF}')"
   130	if [ -z "$JID" ]; then
   131	  echo "SUBMIT FAILED: ${OUT}"
   132	  echo "submit_failed $(date -Is)" >> "$MANIFEST"
   133	  exit 1
   134	fi
   135	echo "submitted ${ARM} -> job ${JID}"
   136	if ! echo "jobid ${JID}" >> "$MANIFEST"; then
   137	  echo "could not append job id ${JID} to ${MANIFEST} — cancelling the job rather than leave it unrecorded"
   138	  scancel "$JID" || echo "scancel ${JID} FAILED — cancel it by hand NOW"
   139	  exit 2
   140	fi
   141	echo "submission recorded: ${MANIFEST} (job ${JID})"

 succeeded in 0ms:
FLAC_AR_BF_C16.json
FLAC_AR_BF_C32.json
FLAC_AR_BF_C4L.json
FLAC_AR_BF_C8.json
FLAC_AR_BF_FA1.json
FLAC_AR_VANCKPT.json
arm_launch_registry.json
c4_backfill_manifest.json
fa_orbit_2026-08-06_16-47-14_C4L_8x8_jid3646734_manifest.txt
fa_orbit_2026-08-06_18-00-52_C4L_8x8_jid3648568_manifest.txt
fa_orbit_2026-08-06_20-36-00_C4L_8x8_jid3648694_manifest.txt
fa_orbit_2026-08-06_20-55-01_C8_8x8_jid3648695_manifest.txt
fa_orbit_2026-08-06_20-55-50_C16_8x8_jid3648696_manifest.txt
fa_orbit_2026-08-07_00-16-30_C32_8x8_jid3648697_manifest.txt
fa_orbit_2026-08-09_12-06-39_VANL_8x8_jid3661520_manifest.txt
fa_orbit_2026-08-10_02-04-24_pytest_legmanifest.log
fa_orbit_add_anchor.py
fa_orbit_ckpt_preflight.py
fa_orbit_classify.py
fa_orbit_equiv_probe.py
fa_orbit_equiv_probe.sbatch
fa_orbit_equivprobe_wrapper_test.sh
fa_orbit_measure_worktree.sh
fa_orbit_producer_manifest.py
fa_orbit_record_restart.py
fa_orbit_screen.sbatch
fa_orbit_screen_guardtests.sh
fa_orbit_screen_submit.sh
fa_orbit_submit.sh
fa_orbit_train.sbatch
fa_orbit_train_guardtests.sh
fa_orbit_traj_submit.sh
fa_orbit_wandb_readback.py
p0_manifest_1334933-1786032532843128131-8f21c960.txt
p0_manifest_72a8114-1785969226421855487-c8d5b51f.txt
p0_manifest_86a752b-1785980874148140138-06d348d6.txt
p0_manifest_9bf1936-1786033425104073952-d8d84328.txt
p0_manifest_aa4bc18-1785968431124626318-df9602ea.txt
p0_manifest_bd96575-1786045321510462046-a3ed28eb.txt
p0_manifest_bd96575-1786045321895684456-ae4c2f92.txt

exec
/usr/bin/bash -lc "git log --all --name-status --format='COMMIT %H %s' -- worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_launch.sh | sed -n '1,200p'; git ls-tree -r HEAD --name-only | grep -F 'fa_orbit_launch.sh' || true" in /n/fs/gatrdp/codespace/FLAC
 succeeded in 179ms:
exec
/usr/bin/bash -lc 'git diff -- worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh' in /n/fs/gatrdp/codespace/FLAC
 succeeded in 0ms:
diff --git a/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch b/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch
index e7f6d54..2216bb4 100644
--- a/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch
+++ b/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch
@@ -16,7 +16,12 @@
 # SUBMIT: never by hand — `fa_orbit_submit.sh <ARM>` derives every Slurm flag
 # from the pins below, so an operator cannot mis-enter --gres/--mem/--time.
 #   ARM                C4L | C8 | C16 | C32
-#   EXPECT_SHA         full reviewed commit sha (required)
+#   EXPECT_SHA         full reviewed commit sha (required). Binding is by
+#                      CONTENT of the training surfaces, not HEAD identity:
+#                      a launch is accepted when HEAD == EXPECT_SHA, or when
+#                      the surfaces below are byte-identical between the two
+#                      (two writers commit to this checkout; worklog/record
+#                      commits must not kill a queued leg).
 #   RESUME_CKPT/EXPECTED_STEP   crash restart only (see LINEAGE)
 #   SMOKE=1            the reviewed multi-GPU smoke (see SMOKE MODE)
 # RUNG / MAXSTEPS / MIN_FREE_MB / time limit are NOT operator inputs any more.
@@ -200,15 +205,46 @@ HEAD_SHA="$(git rev-parse HEAD 2>/dev/null)"
 DRIFT="$(git status --porcelain --untracked-files=no -- train.py defaults.ini src \
           "$EXPDIR"/*.json "$EXPDIR"/*.py "$EXPDIR"/*.sbatch "$EXPDIR"/*.sh \
           "$EXP07/FLAC_AR_BF.json" 2>/dev/null)"
+# Commit binding is CONTENT-scoped: HEAD identity is sufficient but not
+# necessary. Two sessions commit to this checkout, so a pending leg must
+# survive commits that leave the training closure untouched — and abort on
+# any commit that changes it. The closure is what the job actually loads:
+# train.py, defaults.ini, src/, the arm configs, this launcher, the launch
+# wrapper and the four runtime helper scripts it invokes, plus exp_07's
+# FLAC_AR_BF.json (C4L parity baseline).
+# Record/analysis files (registry, manifests, gen_*/validators, worklog)
+# are deliberately OUTSIDE the closure. Fail-closed: unknown EXPECT_SHA or
+# a failing git diff aborts.
+surface_diff_vs_expect() {
+  git diff --name-only "${EXPECT_SHA}" HEAD -- train.py defaults.ini src \
+      "$EXPDIR"/FLAC_AR_*.json "$EXPDIR"/fa_orbit_train.sbatch "$EXPDIR"/fa_orbit_launch.sh \
+      "$EXPDIR"/fa_orbit_ckpt_preflight.py "$EXPDIR"/assert_arm_configs_exp11.py \
+      "$EXPDIR"/fa_orbit_wandb_readback.py "$EXPDIR"/fa_orbit_classify.py \
+      "$EXP07/FLAC_AR_BF.json"
+}
 if [ "$DRYRUN" = "1" ]; then
-  [ "$HEAD_SHA" = "$EXPECT_SHA" ] && echo "commit binding OK (dry run): ${HEAD_SHA}" \
-    || echo "DRY-RUN ADVISORY: HEAD ${HEAD_SHA} != EXPECT_SHA ${EXPECT_SHA} (a real launch aborts here)"
+  if [ "$HEAD_SHA" = "$EXPECT_SHA" ]; then
+    echo "commit binding OK (dry run): ${HEAD_SHA}"
+  elif ! git cat-file -e "${EXPECT_SHA}^{commit}" 2>/dev/null; then
+    echo "DRY-RUN ADVISORY: EXPECT_SHA ${EXPECT_SHA} is not a commit known to this repo (a real launch aborts here)"
+  elif SD="$(surface_diff_vs_expect 2>&1)" && [ -z "$SD" ]; then
+    echo "commit binding OK (dry run, content): training surfaces identical, EXPECT_SHA ${EXPECT_SHA} HEAD ${HEAD_SHA}"
+  else
+    echo "DRY-RUN ADVISORY: training surfaces changed since EXPECT_SHA ${EXPECT_SHA} (a real launch aborts here):"
+    echo "$SD"
+  fi
   [ -z "$DRIFT" ] || echo "DRY-RUN ADVISORY: tracked measurement surfaces are modified (a real launch aborts here)"
 else
   [ -n "${SLURM_JOB_ID:-}" ] || die "a real launch must run under sbatch (no SLURM_JOB_ID) - abort"
-  [ "$HEAD_SHA" = "$EXPECT_SHA" ] || die "HEAD ${HEAD_SHA} != EXPECT_SHA ${EXPECT_SHA} - abort"
+  if [ "$HEAD_SHA" = "$EXPECT_SHA" ]; then
+    echo "commit binding OK: ${HEAD_SHA}"
+  else
+    git cat-file -e "${EXPECT_SHA}^{commit}" 2>/dev/null || die "EXPECT_SHA ${EXPECT_SHA} is not a commit known to this repo - abort"
+    SD="$(surface_diff_vs_expect 2>&1)" || die "surface diff vs EXPECT_SHA failed: ${SD} - abort"
+    [ -z "$SD" ] || { echo "training surfaces changed since EXPECT_SHA ${EXPECT_SHA} (HEAD ${HEAD_SHA}) - abort:"; echo "$SD"; exit 2; }
+    echo "commit binding OK (content): training surfaces identical, EXPECT_SHA ${EXPECT_SHA} HEAD ${HEAD_SHA}"
+  fi
   [ -z "$DRIFT" ] || { echo "tracked measurement surfaces modified since review - abort:"; echo "$DRIFT"; exit 2; }
-  echo "commit binding OK: ${HEAD_SHA}"
 fi
 
 # --- D. arm -> config (single source) + semantic gate -------------------------
diff --git a/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh b/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh
index 6aed577..36021c1 100755
--- a/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh
+++ b/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh
@@ -147,6 +147,37 @@ case_run "wrong EXPECT_SHA aborts" 2 "EXPECT_SHA" \
 case_run "real mode needs sbatch" 2 "must run under sbatch" \
   -- ARM=C4L SMOKE=1 SMOKE_RUNG=16x4 SMOKE_MIN_FREE_MB=99000000 "EXPECT_SHA=${HEAD_SHA}" "OUTPUT_ROOT=${OUT_ROOT}"
 
+# Content-scoped binding: a queued leg must survive commits that leave the
+# training closure untouched, and abort on ones that change it. Both cases
+# scan real history for a suitable EXPECT_SHA; if none exists in 40 commits
+# the case is SKIPped (counted as neither PASS nor FAIL, but reported).
+surfaces_diff() { git diff --name-only "$1" HEAD -- train.py defaults.ini src \
+  "$EXPDIR"/FLAC_AR_*.json "$EXPDIR"/fa_orbit_train.sbatch "$EXPDIR"/fa_orbit_launch.sh \
+  "$EXPDIR"/fa_orbit_ckpt_preflight.py "$EXPDIR"/assert_arm_configs_exp11.py \
+  "$EXPDIR"/fa_orbit_wandb_readback.py "$EXPDIR"/fa_orbit_classify.py \
+  worklog/worklog_yixun/exp_07_fa_scratch_claude/FLAC_AR_BF.json; }
+SAME_SHA=""; CHANGED_SHA=""
+for i in $(seq 1 40); do
+  S="$(git rev-parse --verify -q "HEAD~${i}")" || break
+  D="$(surfaces_diff "$S")" || continue
+  if [ -z "$D" ] && [ -z "$SAME_SHA" ]; then SAME_SHA="$S"; fi
+  if [ -n "$D" ] && [ -z "$CHANGED_SHA" ]; then CHANGED_SHA="$S"; fi
+  [ -n "$SAME_SHA" ] && [ -n "$CHANGED_SHA" ] && break
+done
+if [ -n "$SAME_SHA" ]; then
+  # gate passes (message proves it), then the 99-TB VRAM floor aborts downstream
+  case_run "moved HEAD, surfaces identical -> gate passes" 2 "commit binding OK (content)" \
+    -- ARM=C4L SMOKE=1 SMOKE_RUNG=16x4 SMOKE_MIN_FREE_MB=99000000 "EXPECT_SHA=${SAME_SHA}" SLURM_JOB_ID=999999
+else
+  echo "SKIP  moved HEAD, surfaces identical: no such commit in 40-deep history"
+fi
+if [ -n "$CHANGED_SHA" ]; then
+  case_run "moved HEAD, surfaces changed -> aborts" 2 "training surfaces changed since EXPECT_SHA" \
+    -- ARM=C4L SMOKE=1 SMOKE_RUNG=16x4 SMOKE_MIN_FREE_MB=99000000 "EXPECT_SHA=${CHANGED_SHA}" SLURM_JOB_ID=999999
+else
+  echo "SKIP  moved HEAD, surfaces changed: no such commit in 40-deep history"
+fi
+
 echo "--- E. semantic gate on a mislabelled config (temp copy; tracked tree untouched) ---"
 FAKE_EXP="${TMP}/fakeexp"; mkdir -p "$FAKE_EXP"
 cp "${EXPDIR}/FLAC_AR_BF_C4L.json" "${FAKE_EXP}/FLAC_AR_BF_C32.json"      # C4 orbit under the C32 name

exec
/usr/bin/bash -lc 'git diff --name-status 0f0acb2 HEAD; git log --oneline --decorate -12' in /n/fs/gatrdp/codespace/FLAC
 succeeded in 0ms:
A	outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=0-step=2500_metrics_1_1.0_exp11_VANL_screen_S2500_s42_K1.json
A	outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=0-step=2500_metrics_1_1.0_exp11_VANL_screen_S2500_s42_K1.json.screenmeta.json
A	outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=0-step=2500_metrics_1_1.0_exp11_VANL_screen_S2500_s42_K8.json
A	outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=0-step=2500_metrics_1_1.0_exp11_VANL_screen_S2500_s42_K8.json.screenmeta.json
A	outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=1-step=5000_metrics_1_1.0_exp11_VANL_screen_S5000_s42_K1.json
A	outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=1-step=5000_metrics_1_1.0_exp11_VANL_screen_S5000_s42_K1.json.screenmeta.json
A	outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=1-step=5000_metrics_1_1.0_exp11_VANL_screen_S5000_s42_K8.json
A	outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=1-step=5000_metrics_1_1.0_exp11_VANL_screen_S5000_s42_K8.json.screenmeta.json
A	outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=1-step=7500_metrics_1_1.0_exp11_VANL_screen_S7500_s42_K1.json
A	outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=1-step=7500_metrics_1_1.0_exp11_VANL_screen_S7500_s42_K1.json.screenmeta.json
A	outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=1-step=7500_metrics_1_1.0_exp11_VANL_screen_S7500_s42_K8.json
A	outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=1-step=7500_metrics_1_1.0_exp11_VANL_screen_S7500_s42_K8.json.screenmeta.json
A	outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=2-step=10000_metrics_1_1.0_exp11_VANL_screen_S10000_s42_K1.json
A	outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=2-step=10000_metrics_1_1.0_exp11_VANL_screen_S10000_s42_K1.json.screenmeta.json
A	outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=2-step=10000_metrics_1_1.0_exp11_VANL_screen_S10000_s42_K8.json
A	outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=2-step=10000_metrics_1_1.0_exp11_VANL_screen_S10000_s42_K8.json.screenmeta.json
A	outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=2-step=12500_metrics_1_1.0_exp11_VANL_screen_S12500_s42_K1.json
A	outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=2-step=12500_metrics_1_1.0_exp11_VANL_screen_S12500_s42_K1.json.screenmeta.json
A	outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=2-step=12500_metrics_1_1.0_exp11_VANL_screen_S12500_s42_K8.json
A	outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=2-step=12500_metrics_1_1.0_exp11_VANL_screen_S12500_s42_K8.json.screenmeta.json
A	outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=3-step=15000_metrics_1_1.0_exp11_VANL_screen_S15000_s42_K1.json
A	outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=3-step=15000_metrics_1_1.0_exp11_VANL_screen_S15000_s42_K1.json.screenmeta.json
A	outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=3-step=15000_metrics_1_1.0_exp11_VANL_screen_S15000_s42_K8.json
A	outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=3-step=15000_metrics_1_1.0_exp11_VANL_screen_S15000_s42_K8.json.screenmeta.json
A	outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=3-step=17500_metrics_1_1.0_exp11_VANL_screen_S17500_s42_K1.json
A	outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=3-step=17500_metrics_1_1.0_exp11_VANL_screen_S17500_s42_K1.json.screenmeta.json
A	outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=3-step=17500_metrics_1_1.0_exp11_VANL_screen_S17500_s42_K8.json
A	outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=3-step=17500_metrics_1_1.0_exp11_VANL_screen_S17500_s42_K8.json.screenmeta.json
A	outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=4-step=20000_metrics_1_1.0_exp11_VANL_screen_S20000_s42_K1.json
A	outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=4-step=20000_metrics_1_1.0_exp11_VANL_screen_S20000_s42_K1.json.screenmeta.json
A	outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=4-step=20000_metrics_1_1.0_exp11_VANL_screen_S20000_s42_K8.json
A	outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=4-step=20000_metrics_1_1.0_exp11_VANL_screen_S20000_s42_K8.json.screenmeta.json
A	outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=4-step=22500_metrics_1_1.0_exp11_VANL_screen_S22500_s42_K1.json
A	outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=4-step=22500_metrics_1_1.0_exp11_VANL_screen_S22500_s42_K1.json.screenmeta.json
A	outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=4-step=22500_metrics_1_1.0_exp11_VANL_screen_S22500_s42_K8.json
A	outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=4-step=22500_metrics_1_1.0_exp11_VANL_screen_S22500_s42_K8.json.screenmeta.json
A	outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=5-step=25000_metrics_1_1.0_exp11_VANL_screen_S25000_s42_K1.json
A	outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=5-step=25000_metrics_1_1.0_exp11_VANL_screen_S25000_s42_K1.json.screenmeta.json
A	outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=5-step=25000_metrics_1_1.0_exp11_VANL_screen_S25000_s42_K8.json
A	outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=5-step=25000_metrics_1_1.0_exp11_VANL_screen_S25000_s42_K8.json.screenmeta.json
A	outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=6-step=27500_metrics_1_1.0_exp11_VANL_screen_S27500_s42_K1.json
A	outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=6-step=27500_metrics_1_1.0_exp11_VANL_screen_S27500_s42_K1.json.screenmeta.json
A	outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=6-step=27500_metrics_1_1.0_exp11_VANL_screen_S27500_s42_K8.json
A	outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=6-step=27500_metrics_1_1.0_exp11_VANL_screen_S27500_s42_K8.json.screenmeta.json
A	outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=6-step=30000_metrics_1_1.0_exp11_VANL_screen_S30000_s42_K1.json
A	outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=6-step=30000_metrics_1_1.0_exp11_VANL_screen_S30000_s42_K1.json.screenmeta.json
A	outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=6-step=30000_metrics_1_1.0_exp11_VANL_screen_S30000_s42_K8.json
A	outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=6-step=30000_metrics_1_1.0_exp11_VANL_screen_S30000_s42_K8.json.screenmeta.json
A	outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=7-step=32500_metrics_1_1.0_exp11_VANL_screen_S32500_s42_K1.json
A	outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=7-step=32500_metrics_1_1.0_exp11_VANL_screen_S32500_s42_K1.json.screenmeta.json
A	outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=7-step=32500_metrics_1_1.0_exp11_VANL_screen_S32500_s42_K8.json
A	outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=7-step=32500_metrics_1_1.0_exp11_VANL_screen_S32500_s42_K8.json.screenmeta.json
A	outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=7-step=35000_metrics_1_1.0_exp11_VANL_screen_S35000_s42_K1.json
A	outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=7-step=35000_metrics_1_1.0_exp11_VANL_screen_S35000_s42_K1.json.screenmeta.json
A	outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=7-step=35000_metrics_1_1.0_exp11_VANL_screen_S35000_s42_K8.json
A	outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=7-step=35000_metrics_1_1.0_exp11_VANL_screen_S35000_s42_K8.json.screenmeta.json
A	outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=8-step=37500_metrics_1_1.0_exp11_VANL_screen_S37500_s42_K1.json
A	outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=8-step=37500_metrics_1_1.0_exp11_VANL_screen_S37500_s42_K1.json.screenmeta.json
A	outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=8-step=37500_metrics_1_1.0_exp11_VANL_screen_S37500_s42_K8.json
A	outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=8-step=37500_metrics_1_1.0_exp11_VANL_screen_S37500_s42_K8.json.screenmeta.json
A	outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp11_VANL_screen_S40000_s42_K1.json
A	outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp11_VANL_screen_S40000_s42_K1.json.screenmeta.json
A	outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp11_VANL_screen_S40000_s42_K8.json
A	outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp11_VANL_screen_S40000_s42_K8.json.screenmeta.json
A	src/tests/fixtures/exp15_yaw_aug_disabled_golden.json
M	src/tests/test_exp14_validate_cell.py
A	src/tests/test_yaw_aug_arm_config.py
A	src/tests/test_yaw_aug_record_control.py
A	src/tests/test_yaw_aug_training.py
M	src/training/diffusion.py
M	src/training/factory.py
M	worklog/worklog_yixun/NEURONIC_EXP11_METRICS_SHA256SUMS.txt
A	worklog/worklog_yixun/exp_10_fa_scratch_resume_claude/decay_A4_fa_2026-08-10_23-07-53.log
A	worklog/worklog_yixun/exp_10_fa_scratch_resume_claude/decay_A4_van_2026-08-10_23-07-53.log
M	worklog/worklog_yixun/exp_10_fa_scratch_resume_claude/fa_scratch_resume_worklog.md
M	worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_02_trajectories.html
M	worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_results_assets/fa_orbit_traj_K1_C50.png
M	worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_results_assets/fa_orbit_traj_K1_EDT.png
M	worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_results_assets/fa_orbit_traj_K1_R1.png
M	worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_results_assets/fa_orbit_traj_K1_R10.png
M	worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_results_assets/fa_orbit_traj_K1_R5.png
M	worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_results_assets/fa_orbit_traj_K1_T60.png
M	worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_results_assets/fa_orbit_traj_K8_C50.png
M	worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_results_assets/fa_orbit_traj_K8_EDT.png
M	worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_results_assets/fa_orbit_traj_K8_R1.png
M	worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_results_assets/fa_orbit_traj_K8_R10.png
M	worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_results_assets/fa_orbit_traj_K8_R5.png
M	worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_results_assets/fa_orbit_traj_K8_T60.png
M	worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_results_assets/fa_orbit_trajectories_K1.png
M	worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_results_assets/fa_orbit_trajectories_K8.png
A	worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C16_1786465302622561406-f725e951.txt
A	worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C32_1786465302839671032-280f410d.txt
A	worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C4L_1786465302140788027-ae06a985.txt
A	worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C8_1786465302361183738-e9235c57.txt
A	worklog/worklog_yixun/exp_14_yaw_gen_claude/commits_yaw_gen.md
A	worklog/worklog_yixun/exp_14_yaw_gen_claude/exp14_ckpt_expect.json
A	worklog/worklog_yixun/exp_14_yaw_gen_claude/exp14_hash_ckpts.py
M	worklog/worklog_yixun/exp_14_yaw_gen_claude/exp14_validate_cell.py
A	worklog/worklog_yixun/exp_14_yaw_gen_claude/plan_yaw_gen.md
A	worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_2026-08-11_12-29-01_screen_guardtests.log
A	worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_2026-08-11_12-32-28_screen_guardtests.log
A	worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_2026-08-11_13-21-19_screen_guardtests.log
A	worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_2026-08-11_13-24-20_screen_guardtests.log
A	worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_2026-08-11_13-48-43_screen_guardtests.log
A	worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_2026-08-11_14-23-51_screen_guardtests.log
A	worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_codex_code_r1_reverify.md
A	worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_codex_code_r1_review.md
A	worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_codex_code_r2_review.md
A	worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_codex_plan_review.md
M	worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_screen.sbatch
M	worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_screen_guardtests.sh
M	worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_screen_submit.sh
M	worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_submit_grid.sh
A	worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_worklog.md
A	worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_yixun_query.md
A	worklog/worklog_yixun/exp_15_yaw_aug_claude/FLAC_AR_YAWAUG.json
A	worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-11_12-32-24_pytest_r1.log
A	worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-11_12-33-32_pytest_r1.log
A	worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-11_12-34-20_pytest_r1.log
A	worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-11_12-35-33_pytest_r1.log
A	worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-11_12-36-31_pytest_r1.log
A	worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-11_12-38-25_pytest_r1.log
A	worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-11_12-39-38_pytest_r1.log
A	worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-11_12-40-24_pytest_r1.log
A	worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-11_12-41-32_pytest_r1.log
A	worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-11_12-42-08_pytest_r1.log
A	worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-11_12-44-13_pytest_r1.log
A	worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-11_13-05-38_pytest_r1fix.log
A	worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-11_13-06-41_pytest_r1fix.log
A	worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-11_13-07-53_pytest_r1fix.log
A	worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-11_13-08-43_pytest_r1fix.log
A	worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-11_13-10-00_pytest_r1fix.log
A	worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-11_13-11-16_pytest_r1fix.log
A	worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-11_13-12-55_pytest_r1fix.log
A	worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-11_13-13-48_pytest_r1fix.log
A	worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-11_13-20-39_pytest_r2.log
A	worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-11_13-21-13_pytest_r2.log
A	worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-11_13-22-50_pytest_r2.log
A	worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-11_13-23-50_pytest_r2.log
A	worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-11_13-24-23_record_control.log
A	worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-11_13-26-15_pytest_r2.log
A	worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-11_13-27-04_pytest_r2.log
A	worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-11_13-27-27_record_control.log
A	worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-11_13-28-13_pytest_r2.log
A	worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-11_13-47-17_pytest_r2fix.log
A	worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-11_13-48-19_pytest_r2fix.log
A	worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-11_13-48-42_pytest_r2fix.log
A	worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-11_13-49-46_pytest_r2fix.log
A	worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-11_13-50-40_pytest_r2fix.log
A	worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-11_13-51-23_pytest_r2fix.log
A	worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-11_13-52-21_pytest_r2fix.log
A	worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-11_13-53-02_pytest_r2fix.log
A	worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-11_13-53-32_pytest_r2fix.log
A	worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-11_13-54-10_pytest_r2fix.log
A	worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-11_13-54-50_pytest_r2fix.log
A	worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-11_13-55-35_record_control.log
A	worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-11_13-56-28_pytest_r2fix.log
A	worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-11_13-57-04_pytest_r2fix.log
A	worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-11_14-16-44_guardtests_r3.log
A	worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-11_14-18-57_guardtests_r3.log
A	worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-11_14-19-41_guardtests_r3.log
A	worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-11_14-20-26_guardtests_r3.log
A	worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-11_14-21-08_guardtests_r3.log
A	worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-11_14-21-46_dryrun_r3.log
A	worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_codex_code_r1_review.md
A	worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_codex_code_r2_review.md
A	worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_control_admission.json
A	worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_pin_allowlist.txt
A	worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_record_control.py
A	worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_submit.sh
A	worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_train.sbatch
A	worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_train_guardtests.sh
M	worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_worklog.md
A	worklog/worklog_yixun/fa_vs_vanilla_40k_budget.pdf
A	worklog/worklog_yixun/fa_vs_vanilla_40k_budget.png
135cb4b (HEAD -> check-equivariance-necessity) exp_14 r2-fix2: X1 + X2 + X3 — the two refutations, and the harm one of them did
548d5db exp_15 r3: guardtest + DRYRUN transcripts — 98 passed, 0 failed, nothing submitted
9fa7067 exp_15 r3: guardtest fix — bash expands every word of `local` before assigning
b71ca45 exp_15 r3: guardtest suite for the exp_15 gates — local only, nothing submitted
90f2c3c exp_15 r3: the enumerated kit deltas — identity, 40k re-pin, gates (plan §6.6)
a4bbe86 exp_15 r3: VERBATIM copies of exp_11's training kit (zero edits)
6e9d344 (origin/check-equivariance-necessity) exp_15 r2: Codex review (REVISE→fixed, record regenerated) + worklog entries — round 2 closed
3bbd2bc exp_15 r2-fix: final verification log — 485 passed across the affected suites
f51578c exp_15 r2-fix: pin the REAL admission record against drift (finding 6, MINOR)
ae7a924 exp_15 r2-fix: regenerate the admission record under the corrected recorder
1e7ef90 exp_15 r2-fix: exclusive record creation (F5) + a truly byte-level arm diff (F4)
a3719f9 exp_15 r2-fix: type-strict config and step comparison (finding 3, MAJOR)

exec
/usr/bin/bash -lc "git ls-tree -r HEAD --name-only | grep -E '("'^train'"\\.py"'$|''^defaults'"\\.ini"'$|''^src/|exp_11_fa_orbit_claude/(FLAC_AR_.*'"\\.json|fa_orbit_(train\\.sbatch|submit\\.sh|.*\\.py))|exp_07_fa_scratch_claude/FLAC_AR_BF\\.json)' | sed -n '1,260p'" in /n/fs/gatrdp/codespace/FLAC
 succeeded in 0ms:
defaults.ini
src/__init__.py
src/configs/dataset_configs/AR/eval/acousticroom_seeneval.json
src/configs/dataset_configs/AR/eval/acousticroom_seeneval_1.json
src/configs/dataset_configs/AR/eval/acousticroom_seeneval_4.json
src/configs/dataset_configs/AR/eval/acousticroom_unseeneval.json
src/configs/dataset_configs/AR/eval/acousticroom_unseeneval_1.json
src/configs/dataset_configs/AR/eval/acousticroom_unseeneval_4.json
src/configs/dataset_configs/AR/train/acousticroom_train.json
src/configs/dataset_configs/AR/train/acousticroom_train_VAECtxt.json
src/configs/dataset_configs/HAA/eval/haa_test.json
src/configs/dataset_configs/HAA/eval/haa_test_1.json
src/configs/dataset_configs/HAA/eval/haa_val.json
src/configs/dataset_configs/HAA/train/haa_train.json
src/configs/dataset_configs/custom_metadata/AR_md.py
src/configs/dataset_configs/custom_metadata/HAA_md.py
src/configs/model_configs/FLAC/AR/FLAC_AR.json
src/configs/model_configs/FLAC/AR/FLAC_AR_AllCA.json
src/configs/model_configs/FLAC/AR/FLAC_AR_CylViT.json
src/configs/model_configs/FLAC/AR/FLAC_AR_CylViT_PE_CNN.json
src/configs/model_configs/FLAC/AR/FLAC_AR_CylViT_PE_Linear.json
src/configs/model_configs/FLAC/AR/FLAC_AR_InContext.json
src/configs/model_configs/FLAC/AR/FLAC_AR_S.json
src/configs/model_configs/FLAC/AR/FLAC_AR_SimpleViT.json
src/configs/model_configs/FLAC/AR/FLAC_AR_VAECtxt.json
src/configs/model_configs/FLAC/AR/FLAC_AR_noGeom.json
src/configs/model_configs/FLAC/HAA/FLAC_HAA_finetune.json
src/configs/model_configs/VAE/VAE.json
src/data/__init__.py
src/data/dataset.py
src/data/utils.py
src/data/yaw_rotation.py
src/inference/__init__.py
src/inference/generation.py
src/inference/sampling.py
src/inference/utils.py
src/interface/aeiou.py
src/metrics/metric_callback.py
src/metrics/modules/C50.py
src/metrics/modules/EDT.py
src/metrics/modules/Env.py
src/metrics/modules/FD.py
src/metrics/modules/RT60.py
src/metrics/modules/Retrieval.py
src/metrics/modules/l1_stft.py
src/metrics/modules/l1_stft_multires.py
src/models/__init__.py
src/models/autoencoders.py
src/models/blocks.py
src/models/bottleneck.py
src/models/conditioners.py
src/models/cyl_vit.py
src/models/diffusion.py
src/models/discriminators.py
src/models/dit.py
src/models/encodec.py
src/models/factory.py
src/models/pretrained.py
src/models/pretransforms.py
src/models/simplevit.py
src/models/transformer.py
src/models/utils.py
src/tests/conftest.py
src/tests/exp14_fixed_mode_golden.json
src/tests/fixtures/exp15_yaw_aug_disabled_golden.json
src/tests/test_bn_drift_probe.py
src/tests/test_cond_dispatch.py
src/tests/test_cyl_vit_patch_embedding.py
src/tests/test_eval_paths.py
src/tests/test_exp11_equiv_probe.py
src/tests/test_exp11_orbit_configs.py
src/tests/test_exp11_p0_collect.py
src/tests/test_exp11_p0_runner.py
src/tests/test_exp11_restart_record.py
src/tests/test_exp11_trajectory_figures.py
src/tests/test_exp11_validate_rows.py
src/tests/test_exp11_wandb_readback.py
src/tests/test_exp14_fixed_mode_snapshot.py
src/tests/test_exp14_validate_cell.py
src/tests/test_finetune_cond.py
src/tests/test_gen_model_comparison_gate.py
src/tests/test_invariant_conditioning.py
src/tests/test_phase3_vit_training.py
src/tests/test_retune_lr_state.py
src/tests/test_strip_optimizer_state.py
src/tests/test_train_max_steps.py
src/tests/test_train_sync_batchnorm.py
src/tests/test_vit_gradient_checkpointing.py
src/tests/test_yaw_aug_arm_config.py
src/tests/test_yaw_aug_record_control.py
src/tests/test_yaw_aug_training.py
src/tests/test_yaw_random_eval.py
src/tests/test_yaw_symmetry.py
src/tools/__init__.py
src/tools/retune_lr_state.py
src/tools/strip_optimizer_state.py
src/training/__init__.py
src/training/autoencoders.py
src/training/diffusion.py
src/training/factory.py
src/training/losses/__init__.py
src/training/losses/auraloss.py
src/training/losses/losses.py
src/training/utils.py
train.py
worklog/worklog_yixun/exp_07_fa_scratch_claude/FLAC_AR_BF.json
worklog/worklog_yixun/exp_11_fa_orbit_claude/FLAC_AR_BF_C16.json
worklog/worklog_yixun/exp_11_fa_orbit_claude/FLAC_AR_BF_C32.json
worklog/worklog_yixun/exp_11_fa_orbit_claude/FLAC_AR_BF_C4L.json
worklog/worklog_yixun/exp_11_fa_orbit_claude/FLAC_AR_BF_C8.json
worklog/worklog_yixun/exp_11_fa_orbit_claude/FLAC_AR_BF_FA1.json
worklog/worklog_yixun/exp_11_fa_orbit_claude/FLAC_AR_VANCKPT.json
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_add_anchor.py
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_classify.py
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_equiv_probe.py
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_producer_manifest.py
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_record_restart.py
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submit.sh
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_wandb_readback.py

exec
/usr/bin/bash -lc "find . -type f -name 'fa_orbit_launch.sh' -print; find /tmp/claude-374349/-n-fs-gatrdp-codespace-FLAC/66305eb9-88c7-41df-95da-dde30a426dc2 -type f -name 'fa_orbit_launch.sh' -print 2>/dev/null" in /n/fs/gatrdp/codespace/FLAC
 succeeded in 980ms:
exec
/usr/bin/bash -lc 'set +e
for spec in "$PWD/train.py" "$PWD/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch"; do
  git diff --name-only HEAD~1 HEAD -- "$spec" >/tmp/flac-review-git-out 2>/tmp/flac-review-git-err
  rc=$?
  echo "rc=$rc spec=$spec"
  sed -n '"'1,5p' /tmp/flac-review-git-out
  sed -n '1,5p' /tmp/flac-review-git-err
done" in /n/fs/gatrdp/codespace/FLAC
 succeeded in 0ms:
rc=0 spec=/n/fs/gatrdp/codespace/FLAC/train.py
rc=0 spec=/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch

exec
/usr/bin/bash -lc "git ls-files -s -- train.py defaults.ini src worklog/worklog_yixun/exp_11_fa_orbit_claude/FLAC_AR_*.json worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py worklog/worklog_yixun/exp_11_fa_orbit_claude/assert_arm_configs_exp11.py worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_wandb_readback.py worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_classify.py worklog/worklog_yixun/exp_07_fa_scratch_claude/FLAC_AR_BF.json weights/FLAC/VAE.safetensors pyproject.toml | sed -n '1,320p'" in /n/fs/gatrdp/codespace/FLAC
 succeeded in 0ms:
100644 b209106162a61abce83de2a0b8709e8c8ac4ee8b 0	defaults.ini
100644 3852e8dc546b1991506c3c887b32d51a25464f52 0	pyproject.toml
100644 22446be50eb6617222c50b007a38d06490cbab41 0	src/__init__.py
100644 5988bf5d7f3d97270d0a636705a0dd1b028bdc04 0	src/configs/dataset_configs/AR/eval/acousticroom_seeneval.json
100644 3ebb900b63a570d853e94d53488ca5f4dad523ed 0	src/configs/dataset_configs/AR/eval/acousticroom_seeneval_1.json
100644 e81768eb04f1f309aed4e8f0a21e76e756f656d1 0	src/configs/dataset_configs/AR/eval/acousticroom_seeneval_4.json
100644 9379bf910e7213bd1d55e97b32d47269fea0b092 0	src/configs/dataset_configs/AR/eval/acousticroom_unseeneval.json
100644 f296d2b6b573bf43b3b3383b65e08f990c752aec 0	src/configs/dataset_configs/AR/eval/acousticroom_unseeneval_1.json
100644 0622a1b83f4c35cb9b006a44ca37d6da3f9f8797 0	src/configs/dataset_configs/AR/eval/acousticroom_unseeneval_4.json
100644 b2390ef56cc76829c357b57094d10bba8d8ac8db 0	src/configs/dataset_configs/AR/train/acousticroom_train.json
100644 383dfe9483e56f8174a4c053bb674a6d301fe744 0	src/configs/dataset_configs/AR/train/acousticroom_train_VAECtxt.json
100644 68173eb06de97d34f4b96e897469c0270abaa2d4 0	src/configs/dataset_configs/HAA/eval/haa_test.json
100644 999cabd3420b7fb8c959d93ec531bc498f3e762b 0	src/configs/dataset_configs/HAA/eval/haa_test_1.json
100644 c6dea463b5c3b21082e9ed71b855402bdf5de86c 0	src/configs/dataset_configs/HAA/eval/haa_val.json
100644 2bbea906cdd34e56f130e90affd421d5e910535a 0	src/configs/dataset_configs/HAA/train/haa_train.json
100644 8130bb7fdfb7dcda0b522a7ce67448f729296eec 0	src/configs/dataset_configs/custom_metadata/AR_md.py
100644 e22d8d3b024d43fef0997905374bc37b81428646 0	src/configs/dataset_configs/custom_metadata/HAA_md.py
100644 c61533e167a5ada6b76e9f06bf59264097c76d4e 0	src/configs/model_configs/FLAC/AR/FLAC_AR.json
100644 88c120ba701bbf95e6d2829918e5519827a18300 0	src/configs/model_configs/FLAC/AR/FLAC_AR_AllCA.json
100644 1f50eff2cee53125fcbb9f751047bf87dc1f04eb 0	src/configs/model_configs/FLAC/AR/FLAC_AR_CylViT.json
100644 32a21c50d85a35755bc4e6ad2eb0001799e708b5 0	src/configs/model_configs/FLAC/AR/FLAC_AR_CylViT_PE_CNN.json
100644 e70cfc43b9b6cfeeab4d0e679c52080a47cfbca6 0	src/configs/model_configs/FLAC/AR/FLAC_AR_CylViT_PE_Linear.json
100644 6fc95b758c25684581282062b7731c98cedca643 0	src/configs/model_configs/FLAC/AR/FLAC_AR_InContext.json
100644 3baf1106165a3aac1d7b18425c327db32c3f89ba 0	src/configs/model_configs/FLAC/AR/FLAC_AR_S.json
100644 2dfc5365ab1dd886a9cfa6180b59b0dbe58b94a5 0	src/configs/model_configs/FLAC/AR/FLAC_AR_SimpleViT.json
100644 9d64cb76bd1a8749394c1d833ac61d5fe89d9cd9 0	src/configs/model_configs/FLAC/AR/FLAC_AR_VAECtxt.json
100644 ee044ad6631664225ef6c5d9eb6002399484e2e2 0	src/configs/model_configs/FLAC/AR/FLAC_AR_noGeom.json
100644 20e71831098df5cd0253ef0480128dbca61c73bc 0	src/configs/model_configs/FLAC/HAA/FLAC_HAA_finetune.json
100644 5aedbe43fc0995522ab63f1de2e05a287281c736 0	src/configs/model_configs/VAE/VAE.json
100644 e69de29bb2d1d6434b8b29ae775ad8c2e48c5391 0	src/data/__init__.py
100644 b6da23e3c209074e055f754658cbaf1cf12715bf 0	src/data/dataset.py
100644 1a5cd0267455d809cdffe10bbfff1f1ced0344df 0	src/data/utils.py
100644 175a9fbb98d1c1fcaf47984e7d5c8d7475c75aff 0	src/data/yaw_rotation.py
100644 e69de29bb2d1d6434b8b29ae775ad8c2e48c5391 0	src/inference/__init__.py
100644 f85ad29f019b039d6f13896032e6b6870bebffd2 0	src/inference/generation.py
100644 2bd24b134239165dc2953ad926ed08e9bb44d1f2 0	src/inference/sampling.py
100644 6977afd778d0f2993834f6d17fc57ff70f09bd5f 0	src/inference/utils.py
100644 59d94b6e02807a289f01d473b60ebcbdc4b5da53 0	src/interface/aeiou.py
100644 d087b1a320833ba52980772ea0e41ec38e1a93ee 0	src/metrics/metric_callback.py
100644 e5b7dc2ed67f150c58811496f876576e33f48bec 0	src/metrics/modules/C50.py
100644 cfcec2f14d3c4236507f47c1cc0e8ef6c9841a76 0	src/metrics/modules/EDT.py
100644 65db0cec1e95311382638938e2efc0977d6f6d86 0	src/metrics/modules/Env.py
100644 1adb69c54ff741358c1048f8ca40987d05b86311 0	src/metrics/modules/FD.py
100644 20bc904f86ccb71cd3dccbf1e2e55c143940d575 0	src/metrics/modules/RT60.py
100644 853b51eaed01f5f40495edb5a6bf1dd0ae3ff42f 0	src/metrics/modules/Retrieval.py
100644 c651ab2148e32f3f2f93a0e6e987d0dbf406c4ca 0	src/metrics/modules/l1_stft.py
100644 ab82dafbaeef9b8a9a6303e222d9afee348c1b19 0	src/metrics/modules/l1_stft_multires.py
100644 7e27bbcb19a00a93e05ed6cf2a3a38895f26975d 0	src/models/__init__.py
100644 f0997d74e8b0d756ad653ab733d2240160d2a476 0	src/models/autoencoders.py
100644 ed021156baa3310cd109837be639680d67732971 0	src/models/blocks.py
100644 1ff462136eb8a8abb30ced80769cb16195278aef 0	src/models/bottleneck.py
100644 1dae3c53ac0c57b8be6de596c80082bdf204a5ba 0	src/models/conditioners.py
100644 7c0cd0bf489ad52f280b5095d7a6e48ad63ec9a5 0	src/models/cyl_vit.py
100644 b1910b7e5794e5ff4260970ffe6fedadcf261dff 0	src/models/diffusion.py
100644 e0d4c89b02bf1200f54314bd391e2b95fc6837f5 0	src/models/discriminators.py
100644 2000814fdcd97c4a1e0f6e3f3c267b9cb077e976 0	src/models/dit.py
100644 9b80f25a75426b3a57c180307f6cec0491be66b9 0	src/models/encodec.py
100644 07f0e70ca5ff8f2f0387cc5392138ea4b970ab58 0	src/models/factory.py
100644 e83af343587da91af92218f309c969c5a975b5ed 0	src/models/pretrained.py
100644 c119a967b594fc21b0ad6eb1671a1b3a24901758 0	src/models/pretransforms.py
100644 cddd9a33cef557b90a7bb422f024981f07621b6a 0	src/models/simplevit.py
100644 8e09291ad1f2727c899c0b930bb480a295be8742 0	src/models/transformer.py
100644 44688df45a0576a0712b153cbbd24c50e0171fc0 0	src/models/utils.py
100644 e5828e1c2477462be83146f8a2661135626158b7 0	src/tests/conftest.py
100644 b8b0d1add1ce09ae73d6a31b7d5fb96cb4559163 0	src/tests/exp14_fixed_mode_golden.json
100644 1f669d26f07ba71f45e8f6c863b636d44bd56902 0	src/tests/fixtures/exp15_yaw_aug_disabled_golden.json
100644 c6b34b3436bdb481eebedbcaf56e943aee9f810e 0	src/tests/test_bn_drift_probe.py
100644 5958cf7334cdf632233156f7630147aa204a5703 0	src/tests/test_cond_dispatch.py
100644 dd0beb9589019c65dc9089ee79bc84794c0d37bd 0	src/tests/test_cyl_vit_patch_embedding.py
100644 0f260a76f21b6ed9df8cb81aefcc2fa8ed304779 0	src/tests/test_eval_paths.py
100644 98833fe598c5014c2b520e4b39904434a0fddb91 0	src/tests/test_exp11_equiv_probe.py
100644 fb27d48017db1b18ef6e2e12f2f4aef559cf3317 0	src/tests/test_exp11_orbit_configs.py
100644 784f926a7801f765e82fbc27b9722e5f3b4673f5 0	src/tests/test_exp11_p0_collect.py
100644 b3521bb58f1c259490fb0b3de02c8e90b7c4b12a 0	src/tests/test_exp11_p0_runner.py
100644 9b6c3eb4efd3c8e2cf954d0436058b661383008c 0	src/tests/test_exp11_restart_record.py
100644 2b8d8b2872ce2ba51f5a1c5a5795470fd2fe71e6 0	src/tests/test_exp11_trajectory_figures.py
100644 aa22dc6e4a0d9f222b27054d6d29e3b604ca4fae 0	src/tests/test_exp11_validate_rows.py
100644 33ef165641b689008a9d57acd9e225711a78a192 0	src/tests/test_exp11_wandb_readback.py
100644 563e1b8740f7960eb39f4c25903b819877f6f06b 0	src/tests/test_exp14_fixed_mode_snapshot.py
100644 ce887f6e5c94a561e69e20a46dc896f5e779c808 0	src/tests/test_exp14_validate_cell.py
100644 d24f9d8a7bf39a1b68b18e1ddcadebb8f9b2e27a 0	src/tests/test_finetune_cond.py
100644 c00392366e7b98228dcfebe5518027799bd411ad 0	src/tests/test_gen_model_comparison_gate.py
100644 d668484e1a84c6352ae6a85264016ebf19b003b3 0	src/tests/test_invariant_conditioning.py
100644 b6baa74e4a2869a6eb786a94a286280c4ace0b6d 0	src/tests/test_phase3_vit_training.py
100644 d907019d51f02b26a9fcf68c861679ab8310ec2a 0	src/tests/test_retune_lr_state.py
100644 45eb4007032d25333b2198bf0e805fea36ecbc14 0	src/tests/test_strip_optimizer_state.py
100644 e355efdc432a474c0535681a99db7f54989d3597 0	src/tests/test_train_max_steps.py
100644 379fd47dba9120cbf4217ee4d5f92f8cc1a90a42 0	src/tests/test_train_sync_batchnorm.py
100644 1ffbeba73cc895f1a837a2d2b06a7ecaf1e20434 0	src/tests/test_vit_gradient_checkpointing.py
100644 37169c1f6b7ac3e063b8a98aaf6c6c8a4cbdd401 0	src/tests/test_yaw_aug_arm_config.py
100644 e4e72ad48c9a068f1c63f2c9630226eaac2366ed 0	src/tests/test_yaw_aug_record_control.py
100644 21bda7610ab0b5729f667cd58f19330a3edc2d30 0	src/tests/test_yaw_aug_training.py
100644 61435d37b5b894145d27db6f5e38a3ab3083693b 0	src/tests/test_yaw_random_eval.py
100644 86480df584ca357633b2b01b909c4483fbe4bc09 0	src/tests/test_yaw_symmetry.py
100644 e69de29bb2d1d6434b8b29ae775ad8c2e48c5391 0	src/tools/__init__.py
100644 10548107dfb2d8c75ae6136deb1dafb42c399cb8 0	src/tools/retune_lr_state.py
100644 b5dd5a1748dad7dd0666a2fc8080f136f1d4ef89 0	src/tools/strip_optimizer_state.py
100644 bdfebc6fc3a474150549915da5dd5c73a934084f 0	src/training/__init__.py
100644 eaac17ebaa374cc8b9f9b9a0badde35c78a43f4a 0	src/training/autoencoders.py
100644 c919e40bd7324629d443bf7a39e481db1c158dad 0	src/training/diffusion.py
100644 45acb63bf2bc3c7185c454045a08399d78ff477a 0	src/training/factory.py
100644 a927b2a6e4640589ea628e0adb3c8d56f345e72a 0	src/training/losses/__init__.py
100644 0aa30fdd7939cb58428c9ae0c18e36d486f10c2b 0	src/training/losses/auraloss.py
100644 25e7e9cbaf0b733768e452d2a5ff498afc428f62 0	src/training/losses/losses.py
100644 d095f7905dfc27c92f0166f8894bec12ae7d0759 0	src/training/utils.py
100644 66b0750f9e2081d2c7af7183753024877e75f89b 0	train.py
100644 62ef5b0c8adb283a4c8b33066fb36bbb7eec9c00 0	worklog/worklog_yixun/exp_07_fa_scratch_claude/FLAC_AR_BF.json
100644 3ac1676d643981c3cdbdd4f35f39aa25e9151f20 0	worklog/worklog_yixun/exp_11_fa_orbit_claude/FLAC_AR_BF_C16.json
100644 afd180f108e1a8d55c199192b5f5875ba5bcd8c8 0	worklog/worklog_yixun/exp_11_fa_orbit_claude/FLAC_AR_BF_C32.json
100644 62ef5b0c8adb283a4c8b33066fb36bbb7eec9c00 0	worklog/worklog_yixun/exp_11_fa_orbit_claude/FLAC_AR_BF_C4L.json
100644 952d3740e580341ff52f3fc946a9a6c6584808e2 0	worklog/worklog_yixun/exp_11_fa_orbit_claude/FLAC_AR_BF_C8.json
100644 8ac823e1bbc73befa841e7af47fd322456d379f0 0	worklog/worklog_yixun/exp_11_fa_orbit_claude/FLAC_AR_BF_FA1.json
100644 e41dfa82d51ae7a161c1496c7e68ee4b57497ad7 0	worklog/worklog_yixun/exp_11_fa_orbit_claude/FLAC_AR_VANCKPT.json
100755 3a3c067b29e9f08d2a0ee8c24afbc15cbb37bcb4 0	worklog/worklog_yixun/exp_11_fa_orbit_claude/assert_arm_configs_exp11.py
100644 fc72f836cb2753919051847f6c8db99b4e7f9149 0	worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py
100644 82a27a5cc18e3a79a3dbdd9d3dee9d8e5e88fc76 0	worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_classify.py
100644 e7f6d547fdc62c066e4352fa7abd55b21b121b4c 0	worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch
100644 201f382677ea142f790e41d38cb1a489ebe0fc3f 0	worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_wandb_readback.py

exec
/usr/bin/bash -lc "grep -nE \"open\\(|Path\\(|json|config|defaults|weights|load|import|metadata|\\.ini|\\.json|yaml|toml\" train.py | sed -n '1,280p'; nl -ba train.py | sed -n '1,250p'" in /n/fs/gatrdp/codespace/FLAC
 succeeded in 0ms:
1:import torch
2:import json
3:import os
4:import pytorch_lightning as pl
6:from prefigure.prefigure import get_all_args, push_wandb_config
7:from src.data.dataset import create_dataloader_from_config
8:from src.models import create_model_from_config
9:from src.models.utils import load_ckpt_state_dict, remove_weight_norm_from_model
10:from src.training import create_training_wrapper_from_config
17:    def __init__(self, model_config):
18:        self.model_config = model_config
21:        checkpoint["model_config"] = self.model_config
47:    (default 1000000 via defaults.ini) instead of a hard-coded literal, so a training
81:        "reload_dataloaders_every_n_epochs": 0,
105:    #Get JSON config from args.model_config
106:    with open(args.model_config) as f:
107:        model_config = json.load(f)
109:    with open(args.dataset_config) as f:
110:        dataset_config = json.load(f)
112:    train_dl = create_dataloader_from_config(
113:        dataset_config,
116:        sample_rate=model_config["sample_rate"],
117:        sample_size=model_config["sample_size"],
118:        audio_channels=model_config.get("audio_channels", 1),
122:    val_dataset_config = None
123:    if args.val_dataset_config:
124:        with open(args.val_dataset_config) as f:
125:            val_dataset_config = json.load(f)
127:        val_dl = create_dataloader_from_config(
128:            val_dataset_config,
131:            sample_rate=model_config["sample_rate"],
132:            sample_size=model_config["sample_size"],
133:            audio_channels=model_config.get("audio_channels", 1),
137:    model = create_model_from_config(model_config)
141:        weights = load_ckpt_state_dict(args.pretrained_ckpt_path)
142:        weights = {k.replace('diffusion.', ''): v for k, v in weights.items()} # For diffusion
143:        weights = {k.replace('autoencoder.', ''): v for k, v in weights.items()} # For VAE
144:        disc_weights = {k: v for k, v in weights.items() if 'discriminator' in k}
145:        disc_weights = {k.replace('discriminator.', ''): v for k, v in disc_weights.items()}
146:        weights = {k: v for k, v in weights.items() if 'discriminator' not in k}
147:        weights = {k: v for k, v in weights.items() if 'losses' not in k}
148:        model.load_state_dict(weights, strict=True)
150:    if args.remove_pretransform_weight_norm == "pre_load":
154:        model.pretransform.load_state_dict(load_ckpt_state_dict(args.pretransform_ckpt_path))
157:    if args.remove_pretransform_weight_norm == "post_load":
160:    training_wrapper = create_training_wrapper_from_config(model_config, model)
183:    save_model_config_callback = ModelConfigEmbedderCallback(model_config)
185:    #Combine args and config dicts
187:    args_dict.update({"model_config": model_config})
188:    args_dict.update({"dataset_config": dataset_config})
189:    args_dict.update({"val_dataset_config": val_dataset_config})
193:        push_wandb_config(logger, args_dict)
200:            from pytorch_lightning.strategies import DeepSpeedStrategy
207:                                        load_full_weights=True)
224:        callbacks=[ckpt_callback, exc_callback, save_model_config_callback],
     1	import torch
     2	import json
     3	import os
     4	import pytorch_lightning as pl
     5	
     6	from prefigure.prefigure import get_all_args, push_wandb_config
     7	from src.data.dataset import create_dataloader_from_config
     8	from src.models import create_model_from_config
     9	from src.models.utils import load_ckpt_state_dict, remove_weight_norm_from_model
    10	from src.training import create_training_wrapper_from_config
    11	
    12	class ExceptionCallback(pl.Callback):
    13	    def on_exception(self, trainer, module, err):
    14	        print(f'{type(err).__name__}: {err}')
    15	
    16	class ModelConfigEmbedderCallback(pl.Callback):
    17	    def __init__(self, model_config):
    18	        self.model_config = model_config
    19	
    20	    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
    21	        checkpoint["model_config"] = self.model_config
    22	
    23	def _as_bool(value):
    24	    """Coerce a prefigure-parsed flag to a genuine ``bool``.
    25	
    26	    prefigure parses the lowercase ini literal ``false``/``true`` as the *string*
    27	    "false"/"true" (ast.literal_eval rejects lowercase, so the flag is registered
    28	    ``type=str``); only the capitalized ``False``/``True`` yield a real bool. Accept
    29	    str or bool so a forwarded Trainer kwarg is always a genuine boolean.
    30	    """
    31	    if isinstance(value, bool):
    32	        return value
    33	    if isinstance(value, str):
    34	        s = value.strip().lower()
    35	        if s in ("true", "1", "yes", "on"):
    36	            return True
    37	        if s in ("false", "0", "no", "off", ""):
    38	            return False
    39	        raise ValueError(f"cannot interpret sync_batchnorm={value!r} as a boolean")
    40	    raise TypeError(f"sync_batchnorm must be bool or str, got {type(value).__name__}")
    41	
    42	def build_trainer_kwargs(args, strategy, callbacks, logger, checkpoint_dir, val_args):
    43	    """Assemble the pl.Trainer keyword arguments (side-effect free; unit-testable).
    44	
    45	    Reproduces the kwargs that were previously inlined into pl.Trainer(...) exactly;
    46	    the only behavioral change is that max_steps is now sourced from args.max_steps
    47	    (default 1000000 via defaults.ini) instead of a hard-coded literal, so a training
    48	    budget can be set without editing code. strategy / callbacks / logger /
    49	    checkpoint_dir / val_args are the values main() derives and passes straight through.
    50	
    51	    sync_batchnorm (default off -> key ABSENT, so the kwargs are byte-identical to the
    52	    pre-change dict and PL's own default False applies) forwards to
    53	    Trainer(sync_batchnorm=True) only when enabled. It is a multi-GPU-only feature, so
    54	    enabling it with num_gpus < 2 is a fail-closed ValueError (Yixun mandate) rather than
    55	    a silently-ignored no-op. The guard lives here so both construct_trainer/main() and
    56	    any direct caller hit it; val_args may NOT smuggle the key past the guard.
    57	    """
    58	    if "sync_batchnorm" in val_args:
    59	        raise ValueError("sync_batchnorm must come from args (guarded), not val_args")
    60	    sync_batchnorm = _as_bool(getattr(args, "sync_batchnorm", False))
    61	    if sync_batchnorm and args.num_gpus < 2:
    62	        raise ValueError(
    63	            "sync_batchnorm=True requires multi-GPU training (num_gpus >= 2); got "
    64	            f"num_gpus={args.num_gpus}. SyncBatchNorm synchronises BatchNorm statistics "
    65	            "across ranks and is a no-op / unsupported on a single device -- set "
    66	            "--num-gpus >= 2 or drop --sync-batchnorm."
    67	        )
    68	    kwargs = {
    69	        "devices": args.num_gpus,
    70	        "accelerator": "gpu",
    71	        "num_nodes": args.num_nodes,
    72	        "strategy": strategy,
    73	        "precision": args.precision,
    74	        "accumulate_grad_batches": args.accum_batches,
    75	        "callbacks": callbacks,
    76	        "logger": logger,
    77	        "log_every_n_steps": 100,
    78	        "max_steps": args.max_steps, # HAA finetune recipe: --max-steps 1000
    79	        "default_root_dir": checkpoint_dir,
    80	        "gradient_clip_val": args.gradient_clip_val,
    81	        "reload_dataloaders_every_n_epochs": 0,
    82	        "num_sanity_val_steps": 0, # If you need to debug validation, change this line
    83	        **val_args,
    84	    }
    85	    if sync_batchnorm:
    86	        kwargs["sync_batchnorm"] = True  # multi-GPU only; guarded above (fail-closed)
    87	    return kwargs
    88	
    89	def construct_trainer(args, strategy, callbacks, logger, checkpoint_dir, val_args):
    90	    """Construct the pl.Trainer from the assembled kwargs (the tested Trainer boundary)."""
    91	    return pl.Trainer(**build_trainer_kwargs(args, strategy, callbacks, logger, checkpoint_dir, val_args))
    92	
    93	def main():
    94	    torch.set_float32_matmul_precision('medium') 
    95	    torch.multiprocessing.set_sharing_strategy('file_system')
    96	    args = get_all_args()
    97	    seed = args.seed
    98	
    99	    # Set a different seed for each process if using SLURM
   100	    if os.environ.get("SLURM_PROCID") is not None:
   101	        seed += int(os.environ.get("SLURM_PROCID"))
   102	
   103	    pl.seed_everything(seed, workers=True)
   104	
   105	    #Get JSON config from args.model_config
   106	    with open(args.model_config) as f:
   107	        model_config = json.load(f)
   108	
   109	    with open(args.dataset_config) as f:
   110	        dataset_config = json.load(f)
   111	
   112	    train_dl = create_dataloader_from_config(
   113	        dataset_config,
   114	        batch_size=args.batch_size,
   115	        num_workers=args.num_workers,
   116	        sample_rate=model_config["sample_rate"],
   117	        sample_size=model_config["sample_size"],
   118	        audio_channels=model_config.get("audio_channels", 1),
   119	    )
   120	
   121	    val_dl = None
   122	    val_dataset_config = None
   123	    if args.val_dataset_config:
   124	        with open(args.val_dataset_config) as f:
   125	            val_dataset_config = json.load(f)
   126	
   127	        val_dl = create_dataloader_from_config(
   128	            val_dataset_config,
   129	            batch_size=args.batch_size,
   130	            num_workers=args.num_workers,
   131	            sample_rate=model_config["sample_rate"],
   132	            sample_size=model_config["sample_size"],
   133	            audio_channels=model_config.get("audio_channels", 1),
   134	            shuffle=False
   135	        )
   136	
   137	    model = create_model_from_config(model_config)
   138	
   139	    if args.pretrained_ckpt_path:
   140	        print('Loading pretrained model...')
   141	        weights = load_ckpt_state_dict(args.pretrained_ckpt_path)
   142	        weights = {k.replace('diffusion.', ''): v for k, v in weights.items()} # For diffusion
   143	        weights = {k.replace('autoencoder.', ''): v for k, v in weights.items()} # For VAE
   144	        disc_weights = {k: v for k, v in weights.items() if 'discriminator' in k}
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
   206	                                        allgather_bucket_size=5e8,
   207	                                        load_full_weights=True)
   208	        else:
   209	            strategy = args.strategy
   210	    else:
   211	        strategy = 'ddp_find_unused_parameters_true' if args.num_gpus > 1 else "auto"
   212	
   213	    val_args = {}
   214	    
   215	    if args.val_every > 0:
   216	        val_args.update({
   217	            "check_val_every_n_epoch": None,
   218	            "val_check_interval": args.val_every,
   219	        })
   220	
   221	    trainer = construct_trainer(
   222	        args,
   223	        strategy=strategy,
   224	        callbacks=[ckpt_callback, exc_callback, save_model_config_callback],
   225	        logger=logger,
   226	        checkpoint_dir=checkpoint_dir,
   227	        val_args=val_args,
   228	    )
   229	
   230	    trainer.fit(training_wrapper, train_dl, val_dl, ckpt_path=args.ckpt_path if args.ckpt_path else None)
   231	
   232	if __name__ == '__main__':
   233	    main()

exec
/usr/bin/bash -lc "grep -RInE \"open\\(|Path\\(|read_text|read_bytes|json\\.load|torch\\.load|load_file|from_pretrained|os\\.path|__file__|sys\\.path|subprocess|git |registry|manifest|defaults\\.ini|config\" worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py worklog/worklog_yixun/exp_11_fa_orbit_claude/assert_arm_configs_exp11.py worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_wandb_readback.py worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_classify.py | sed -n '1,320p'" in /n/fs/gatrdp/codespace/FLAC
 succeeded in 0ms:
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py:4:exp_10's `bf_resume_launch.sh` proved a restart with ONE CPU-side ``torch.load``
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py:11:  - its embedded ``model_config`` deep-equals this arm's config file, so the
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py:20:  - optionally, the arm's ORIGINAL launch manifest is re-read and the restart is
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py:23:Prints the checkpoint sha256 (for the restart manifest) and a lineage summary.
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py:35:    with open(path, "rb") as fh:
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py:41:def load_ckpt_config(path):
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py:42:    """The model_config embedded in a Lightning checkpoint, on CPU.
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py:48:    ck = torch.load(path, map_location="cpu", weights_only=False)
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py:51:    cfg = ck.get("model_config")
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py:53:        raise RuntimeError(f"checkpoint carries no embedded model_config: {path}")
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py:62:    ck = torch.load(path, map_location="cpu", weights_only=False)
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py:68:def parse_manifest(path):
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py:69:    """The launcher's own manifest format: whitespace-separated `key value...`."""
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py:71:    with open(path, "r") as fh:
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
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py:119:    base = save_dir if os.path.isabs(save_dir) else os.path.join(repo_root, save_dir)
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py:120:    return os.path.realpath(os.path.join(base, f"FLAC_exp11_{arm}", f"exp11_{arm}", "checkpoints"))
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py:123:def check_extension_binding(manifest_path, registry_path, arm, rung, config_path, ckpt_path,
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py:128:    so `check_manifest_binding` demands both. An extension breaks both BY DESIGN
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py:134:    proves it against the COMMITTED registry rather than the mutable manifest
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py:135:    alone: the INITIAL manifest byte-for-byte as audited, the same job/uuid/
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py:136:    launch commit/rung/config/save-dir/training seed, and a resumed checkpoint
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py:142:    if not os.path.isfile(registry_path):
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py:143:        return [f"audited launch registry not found: {registry_path}"], {}
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py:144:    reg = json.load(open(registry_path)).get("arms", {}).get(arm)
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py:146:        return [f"{arm} is not in the audited launch registry {registry_path}"], {}
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py:147:    man = parse_manifest(manifest_path)
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py:150:    got_sha = sha256_file(manifest_path)
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py:151:    if got_sha != reg.get("manifest_sha256"):
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py:152:        problems.append(f"launch manifest sha256 {got_sha[:12]} != audited "
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py:153:                        f"{str(reg.get('manifest_sha256'))[:12]} — the manifest changed after it "
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py:161:                                 ("config_sha256", man.get("config_sha256"), reg.get("config_sha256")),
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py:167:        problems.append("launch manifest carries no 'commit' line — cannot bind the extension to "
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py:170:        problems.append(f"manifest commit {man_commit[:12]} != the registered launch commit "
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py:174:    # The INITIAL budget is the manifest's and the registry's; the extension's is
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py:178:        problems.append(f"manifest max_steps {kv.get('max_steps')!r} != registered "
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py:186:    if sha256_file(config_path) != reg.get("config_sha256"):
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py:187:        problems.append(f"{config_path} sha256 != the registered config_sha256 "
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py:188:                        f"{str(reg.get('config_sha256'))[:12]}")
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py:192:        problems.append(f"{arm} has no audited final_ckpt_sha256 in the registry — the extension "
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py:201:        problems.append("manifest records no save_dir")
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py:204:        if os.path.realpath(os.path.dirname(ckpt_path)) != canon:
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py:214:    ap.add_argument("--config", required=True, help="this arm's model config json")
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py:219:    ap.add_argument("--launch-manifest", default="",
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py:220:                    help="the arm's original launch manifest (binds rung/commit/budget)")
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py:224:    ap.add_argument("--launch-registry", default="",
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py:225:                    help="the committed arm launch registry (required with --extension)")
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py:227:                    help="root the registry's relative save_dir is resolved against")
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py:229:    if args.extension and not args.launch_registry:
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py:230:        ap.error("--extension requires --launch-registry (the audited INITIAL launch row)")
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py:232:    if not os.path.isfile(args.ckpt):
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py:239:        ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py:256:    mc = ck.get("model_config")
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py:258:        problems.append("checkpoint carries no embedded 'model_config' dict")
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py:260:        want = json.load(open(args.config))
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py:264:                f"embedded model_config != {args.config} (parsed-object mismatch; embedded "
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py:282:    if args.launch_manifest:
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py:283:        if not os.path.isfile(args.launch_manifest):
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py:284:            problems.append(f"launch manifest not found: {args.launch_manifest}")
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py:287:                args.launch_manifest, args.launch_registry, args.arm, args.rung, args.config,
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py:291:            more, man = check_manifest_binding(args.launch_manifest, args.arm, args.rung,
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py:295:        problems.append("--extension requires --launch-manifest (the audited INITIAL manifest)")
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py:314:        print(f"  bound to the audited launch manifest: {args.launch_manifest}")
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py:316:        print(f"  bound to launch manifest: {args.launch_manifest}")
worklog/worklog_yixun/exp_11_fa_orbit_claude/assert_arm_configs_exp11.py:4:Replicated from exp_07's ``assert_arm_configs.py`` (which does NOT generalize:
worklog/worklog_yixun/exp_11_fa_orbit_claude/assert_arm_configs_exp11.py:6:vanilla-vs-fa pairing). Same checks, re-pointed at the exp_11 arm configs and
worklog/worklog_yixun/exp_11_fa_orbit_claude/assert_arm_configs_exp11.py:14:    configure_optimizers() whose AdamW/InverseLR objects are field-checked.
worklog/worklog_yixun/exp_11_fa_orbit_claude/assert_arm_configs_exp11.py:25:    HF_HUB_OFFLINE=1 python worklog/worklog_yixun/exp_11_fa_orbit_claude/assert_arm_configs_exp11.py C8
worklog/worklog_yixun/exp_11_fa_orbit_claude/assert_arm_configs_exp11.py:35:    p = os.path.abspath(p)
worklog/worklog_yixun/exp_11_fa_orbit_claude/assert_arm_configs_exp11.py:38:    while not os.path.exists(os.path.join(p, ".git")):
worklog/worklog_yixun/exp_11_fa_orbit_claude/assert_arm_configs_exp11.py:39:        parent = os.path.dirname(p)
worklog/worklog_yixun/exp_11_fa_orbit_claude/assert_arm_configs_exp11.py:46:REPO = _repo_root(os.path.dirname(os.path.abspath(__file__)))
worklog/worklog_yixun/exp_11_fa_orbit_claude/assert_arm_configs_exp11.py:47:sys.path.insert(0, REPO)  # guard against a stale pip-installed src copy
worklog/worklog_yixun/exp_11_fa_orbit_claude/assert_arm_configs_exp11.py:52:from src.models.factory import create_model_from_config  # noqa: E402
worklog/worklog_yixun/exp_11_fa_orbit_claude/assert_arm_configs_exp11.py:53:from src.training.factory import create_training_wrapper_from_config  # noqa: E402
worklog/worklog_yixun/exp_11_fa_orbit_claude/assert_arm_configs_exp11.py:55:HERE = os.path.dirname(os.path.abspath(__file__))
worklog/worklog_yixun/exp_11_fa_orbit_claude/assert_arm_configs_exp11.py:65:# Verbatim from worklog/worklog_yixun/exp_07_fa_scratch_claude/assert_arm_configs.py
worklog/worklog_yixun/exp_11_fa_orbit_claude/assert_arm_configs_exp11.py:72:    hub = os.path.join(HF_HUB_CACHE, "models--facebook--dinov3-vits16-pretrain-lvd1689m")
worklog/worklog_yixun/exp_11_fa_orbit_claude/assert_arm_configs_exp11.py:73:    snap_dir = os.path.join(hub, "snapshots")
worklog/worklog_yixun/exp_11_fa_orbit_claude/assert_arm_configs_exp11.py:74:    if not os.path.isdir(snap_dir):
worklog/worklog_yixun/exp_11_fa_orbit_claude/assert_arm_configs_exp11.py:80:    st = os.path.join(snap_dir, VIT_REV, "model.safetensors")
worklog/worklog_yixun/exp_11_fa_orbit_claude/assert_arm_configs_exp11.py:82:    with open(st, "rb") as f:
worklog/worklog_yixun/exp_11_fa_orbit_claude/assert_arm_configs_exp11.py:96:    cfg = json.load(open(os.path.join(HERE, ARM_CONFIG[arm])))
worklog/worklog_yixun/exp_11_fa_orbit_claude/assert_arm_configs_exp11.py:97:    model = create_model_from_config(cfg)
worklog/worklog_yixun/exp_11_fa_orbit_claude/assert_arm_configs_exp11.py:98:    wrapper = create_training_wrapper_from_config(cfg, model)
worklog/worklog_yixun/exp_11_fa_orbit_claude/assert_arm_configs_exp11.py:120:        # falls back to DEFAULT_FRAME_ANGLES when the config omits the key — but the
worklog/worklog_yixun/exp_11_fa_orbit_claude/assert_arm_configs_exp11.py:124:        # the presence of an inert default. (The config itself is separately required
worklog/worklog_yixun/exp_11_fa_orbit_claude/assert_arm_configs_exp11.py:144:    if w.optimizer_configs["diffusion"]["optimizer"]["config"] != {
worklog/worklog_yixun/exp_11_fa_orbit_claude/assert_arm_configs_exp11.py:146:        raise RuntimeError(f"{tag}: optimizer config drifted")
worklog/worklog_yixun/exp_11_fa_orbit_claude/assert_arm_configs_exp11.py:148:    opts, scheds = w.configure_optimizers()
worklog/worklog_yixun/exp_11_fa_orbit_claude/assert_arm_configs_exp11.py:170:        raise SystemExit(f"usage: {os.path.basename(argv[0])} <{'|'.join(ARM_ORBIT)}>")
worklog/worklog_yixun/exp_11_fa_orbit_claude/assert_arm_configs_exp11.py:179:        raw = json.load(open(os.path.join(HERE, ARM_CONFIG[arm]))).get("training", {})
worklog/worklog_yixun/exp_11_fa_orbit_claude/assert_arm_configs_exp11.py:181:            raise RuntimeError(f"{arm} config declares cond_method={raw.get('cond_method')!r}")
worklog/worklog_yixun/exp_11_fa_orbit_claude/assert_arm_configs_exp11.py:183:            raise RuntimeError(f"{arm} config carries frame_avg_angles={raw['frame_avg_angles']!r} — "
worklog/worklog_yixun/exp_11_fa_orbit_claude/assert_arm_configs_exp11.py:184:                               "a vanilla arm's config has no orbit")
worklog/worklog_yixun/exp_11_fa_orbit_claude/assert_arm_configs_exp11.py:185:        print(f"{arm} config is orbit-free (no cond_method, no frame_avg_angles)")
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_wandb_readback.py:5:the run was written and that it carries the manifest's identity.
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_wandb_readback.py:40:        matches.extend(sorted(glob.glob(os.path.join(root, "wandb", f"run-*-{run_id}"))))
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_wandb_readback.py:52:    if not run_dir or not os.path.isdir(run_dir):
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_wandb_readback.py:54:    if not os.path.basename(run_dir).endswith(f"-{run_id}"):
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_wandb_readback.py:55:        problems.append(f"run directory {os.path.basename(run_dir)} does not carry id {run_id}")
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_wandb_readback.py:56:    meta_path = os.path.join(run_dir, "files", "wandb-metadata.json")
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_wandb_readback.py:58:    if os.path.isfile(meta_path):
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_wandb_readback.py:60:            meta = json.load(open(meta_path))
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_wandb_readback.py:69:            problems.append(f"{key}={got!r} != manifest {want!r}")
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_classify.py:40:        with open(path, "r", errors="replace") as fh:
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_classify.py:61:        if not os.path.isfile(p):

exec
/usr/bin/bash -lc "grep -RInE \"open\\(|Path\\(|read_text|read_bytes|json\\.load|load_dataset_config|metadata|importlib|exec\\(|eval\\(|torch\\.load|load_file|from_pretrained\" src --include='*.py' | sed -n '1,420p'" in /n/fs/gatrdp/codespace/FLAC
 succeeded in 1848ms:
est_exp11_restart_record.py:247:    before = open(world["registry"]).read()
src/tests/test_exp11_restart_record.py:249:    assert open(world["registry"]).read() == before
src/tests/test_exp11_restart_record.py:254:    before = open(world["registry"]).read()
src/tests/test_exp11_restart_record.py:256:    assert open(world["registry"]).read() == before
src/tests/test_exp11_restart_record.py:288:    with open(world["steps"][42500][0], "w") as fh:
src/tests/test_exp11_restart_record.py:321:    with open(world["steps"][45000][0], "w") as fh:
src/tests/test_exp11_restart_record.py:337:                 open(world["steps"][45000][0]).read())
src/tests/test_exp11_restart_record.py:371:    with open(world["registry"], "w") as fh:
src/tests/test_exp11_restart_record.py:382:    with open(recorded, "a") as fh:
src/tests/test_exp11_restart_record.py:425:    cfg = json.load(open(world["config"]))
src/tests/test_exp11_restart_record.py:449:    with open(world["registry"], "w") as fh:
src/tests/test_exp11_restart_record.py:478:    cfg = json.load(open(unanchored["config"]))
src/tests/test_exp11_restart_record.py:487:    with open(unanchored["manifest"], "a") as fh:
src/tests/test_exp11_restart_record.py:502:    cfg = json.load(open(unanchored["config"]))
src/tests/test_exp11_restart_record.py:517:    cfg = json.load(open(unanchored["config"]))
src/tests/test_exp11_restart_record.py:524:    before = open(unanchored["registry"]).read()
src/tests/test_exp11_restart_record.py:526:    assert open(unanchored["registry"]).read() == before
src/tests/test_invariant_conditioning.py:14:- the caller's metadata is never mutated (finding 4).
src/tests/test_invariant_conditioning.py:40:    ``rotate_scene_metadata`` is a *fixed point* of the depth map, hiding
src/tests/test_invariant_conditioning.py:222:        rot = [yr.rotate_scene_metadata(m, math.radians(deg), 512) for m in md]
src/tests/test_invariant_conditioning.py:238:    rot = [yr.rotate_scene_metadata(m, math.radians(37.3), 512) for m in md]
src/tests/test_invariant_conditioning.py:260:                yr.rotate_scene_metadata(m, math.radians(g), 512, pose_keys=tuple(present))
src/tests/test_invariant_conditioning.py:309:# 6. deep non-mutation of the caller's metadata
src/tests/test_invariant_conditioning.py:363:def _stale_depth_conditioning(cond, metadata, device, angles=yr.DEFAULT_FRAME_ANGLES,
src/tests/test_invariant_conditioning.py:369:    md_inv = [yr.cylindrical_pose_features(md) for md in metadata]
src/tests/test_invariant_conditioning.py:372:    img_w = int(metadata[0]["depth"].shape[-1])
src/tests/test_invariant_conditioning.py:377:            v = yr.rotate_scene_metadata(m, math.radians(g), img_w,
src/tests/test_invariant_conditioning.py:394:    rot = [yr.rotate_scene_metadata(m, math.radians(90.0), 512) for m in md]
src/tests/test_invariant_conditioning.py:467:        rot = [yr.rotate_scene_metadata(m, math.radians(deg), 512) for m in md]
src/tests/test_invariant_conditioning.py:491:                yr.rotate_scene_metadata(m, math.radians(g), 512, pose_keys=tuple(present))
src/tests/test_invariant_conditioning.py:516:def _reference_orbit_average(cond, metadata, device, angles):
src/tests/test_invariant_conditioning.py:522:    md_inv = [yr.cylindrical_pose_features(m) for m in metadata]
src/tests/test_invariant_conditioning.py:525:    img_w = int(metadata[0]["depth"].shape[-1])
src/tests/test_invariant_conditioning.py:529:            yr.rotate_scene_metadata(m, math.radians(g), img_w, pose_keys=tuple(present))
src/tests/test_invariant_conditioning.py:811:    ``rotate_scene_metadata`` rolls the panorama by ``round(deg * W / 360)``
src/tests/test_invariant_conditioning.py:819:            # Use the per-column RADIUS: rotate_scene_metadata rolls the panorama
src/tests/test_yaw_symmetry.py:7:- ``rotate_scene_metadata``'s new ``pose_keys`` parameter (default preserves the
src/tests/test_yaw_symmetry.py:101:        rot = yr.rotate_scene_metadata(md, math.radians(deg), 512)
src/tests/test_yaw_symmetry.py:132:        rot = yr.rotate_scene_metadata(md, math.radians(deg), 512)
src/tests/test_yaw_symmetry.py:180:        rot = yr.rotate_scene_metadata(md, math.radians(deg), 512)
src/tests/test_yaw_symmetry.py:206:# rotate_scene_metadata pose_keys param
src/tests/test_yaw_symmetry.py:211:    out = yr.rotate_scene_metadata(md, alpha, 512)
src/tests/test_yaw_symmetry.py:223:    out = yr.rotate_scene_metadata(
src/tests/test_train_sync_batchnorm.py:15:  ``bool``. ``ast.literal_eval("false")`` raises (lowercase ``false``/``true`` are not
src/tests/test_train_sync_batchnorm.py:75:    with open(_DEFAULTS_INI) as f:
src/tests/test_train_sync_batchnorm.py:269:    with open(_TRAIN_PY) as f:
src/tests/test_yaw_aug_record_control.py:19:import importlib.util
src/tests/test_yaw_aug_record_control.py:28:_REPO = Path(__file__).resolve().parents[2]
src/tests/test_yaw_aug_record_control.py:38:    spec = importlib.util.spec_from_file_location("yaw_aug_record_control", RECORDER_PATH)
src/tests/test_yaw_aug_record_control.py:39:    module = importlib.util.module_from_spec(spec)
src/tests/test_yaw_aug_record_control.py:79:            payload["model_config"] = json.loads(CONTROL_CONFIG.read_text())
src/tests/test_yaw_aug_record_control.py:95:    assert rc.sha256_file(blob) == hashlib.sha256(blob.read_bytes()).hexdigest()
src/tests/test_yaw_aug_record_control.py:102:    """Every torch.load this tool performs must be the SAFE one.
src/tests/test_yaw_aug_record_control.py:109:    real_load = torch.load
src/tests/test_yaw_aug_record_control.py:130:    real_load = torch.load
src/tests/test_yaw_aug_record_control.py:164:        with open(path, "ab") as handle:
src/tests/test_yaw_aug_record_control.py:176:    real_read_bytes = Path.read_bytes
src/tests/test_yaw_aug_record_control.py:181:        return real_read_bytes(self)
src/tests/test_yaw_aug_record_control.py:183:    monkeypatch.setattr(Path, "read_bytes", _counting_read)
src/tests/test_yaw_aug_record_control.py:196:    assert ck["sha256"] == hashlib.sha256(ckpt.read_bytes()).hexdigest()
src/tests/test_yaw_aug_record_control.py:238:        json.dumps(json.loads(CONTROL_CONFIG.read_text()), sort_keys=True,
src/tests/test_yaw_aug_record_control.py:247:    other = json.loads(CONTROL_CONFIG.read_text())
src/tests/test_yaw_aug_record_control.py:264:    config = json.loads(CONTROL_CONFIG.read_text())
src/tests/test_yaw_aug_record_control.py:310:    payload = torch.load(ckpt, map_location="cpu", weights_only=True)
src/tests/test_yaw_aug_record_control.py:319:    before = (hashlib.sha256(ckpt.read_bytes()).hexdigest(),
src/tests/test_yaw_aug_record_control.py:320:              hashlib.sha256(CONTROL_CONFIG.read_bytes()).hexdigest())
src/tests/test_yaw_aug_record_control.py:324:    after = (hashlib.sha256(ckpt.read_bytes()).hexdigest(),
src/tests/test_yaw_aug_record_control.py:325:             hashlib.sha256(CONTROL_CONFIG.read_bytes()).hexdigest())
src/tests/test_yaw_aug_record_control.py:336:    original = out.read_bytes()
src/tests/test_yaw_aug_record_control.py:340:    assert out.read_bytes() == original, "the existing record was modified"
src/tests/test_yaw_aug_record_control.py:371:    assert out.read_text() == "someone else got here first"
src/tests/test_yaw_aug_record_control.py:463:    written = json.loads(out.read_text())
src/tests/test_yaw_aug_record_control.py:480:    return json.loads(COMMITTED_RECORD.read_text())
src/tests/test_yaw_aug_record_control.py:531:    vanl = json.loads(EXP11_REGISTRY.read_text())["arms"]["VANL"]
src/tests/test_yaw_aug_record_control.py:542:    body = RECORD_TRANSCRIPT.read_text().split("\nwrote ")[0]
src/tests/test_yaw_aug_record_control.py:543:    assert json.loads(body) == committed_record
src/tests/test_yaw_aug_record_control.py:557:    assert out.read_text() == "{}"
src/tests/test_vit_gradient_checkpointing.py:19:invocations during forward/backward — metadata flags are never trusted.
src/tests/test_vit_gradient_checkpointing.py:35:  invocations under ``eval()`` or ``torch.no_grad()``.
src/tests/test_vit_gradient_checkpointing.py:61:# any transformers-backed module so from_pretrained never reaches the network.
src/tests/test_vit_gradient_checkpointing.py:111:    with open(_BV_JSON) as f:
src/tests/test_vit_gradient_checkpointing.py:112:        cfg = json.load(f)
src/tests/test_vit_gradient_checkpointing.py:254:# enabled -> EXECUTION proof (checkpoint really runs; metadata is never trusted)
src/tests/test_vit_gradient_checkpointing.py:290:        vit.eval()
src/tests/test_vit_gradient_checkpointing.py:295:    """eval() (grad enabled) and train()+torch.no_grad() both bypass checkpointing
src/tests/test_vit_gradient_checkpointing.py:299:        vit.eval()
src/tests/test_vit_gradient_checkpointing.py:310:        vit.eval()
src/tests/test_vit_gradient_checkpointing.py:333:        vit.eval()
src/tests/test_vit_gradient_checkpointing.py:425:    stub.eval()
src/tests/test_phase3_vit_training.py:1:import importlib.util
src/tests/test_phase3_vit_training.py:10:SCRIPT = Path(__file__).resolve().parents[2] / "worklog/worklog_zhixuan/exp_05_cylvit_yaw_ablation_claude/train_vit_ablation.py"
src/tests/test_phase3_vit_training.py:11:SPEC = importlib.util.spec_from_file_location("train_vit_ablation", SCRIPT)
src/tests/test_phase3_vit_training.py:12:train_vit_ablation = importlib.util.module_from_spec(SPEC)
src/tests/test_gen_model_comparison_gate.py:13:import importlib.util
src/tests/test_gen_model_comparison_gate.py:27:    spec = importlib.util.spec_from_file_location("gen_model_comparison", _GEN_PY)
src/tests/test_gen_model_comparison_gate.py:29:    mod = importlib.util.module_from_spec(spec)
src/tests/test_gen_model_comparison_gate.py:49:    before = open(table, "rb").read()
src/tests/test_gen_model_comparison_gate.py:51:    assert open(table, "rb").read() == before, "importing the generator rewrote the table"
src/tests/test_gen_model_comparison_gate.py:99:    spec = importlib.util.spec_from_file_location(
src/tests/test_gen_model_comparison_gate.py:102:    V = importlib.util.module_from_spec(spec)
src/tests/test_gen_model_comparison_gate.py:118:    ck_sha = hashlib.sha256(ck_file.read_bytes()).hexdigest()
src/tests/test_gen_model_comparison_gate.py:119:    cfg_sha = hashlib.sha256(cfg_file.read_bytes()).hexdigest()
src/tests/test_gen_model_comparison_gate.py:167:    rec = json.load(open(paths[0]))
src/tests/test_gen_model_comparison_gate.py:169:    open(paths[0], "w").write(json.dumps(rec))
src/tests/test_gen_model_comparison_gate.py:196:    rec = json.load(open(paths[0]))
src/tests/test_gen_model_comparison_gate.py:198:    open(paths[0], "w").write(json.dumps(rec))
src/tests/test_gen_model_comparison_gate.py:218:    side = json.load(open(paths[0] + ".screenmeta.json"))
src/tests/test_gen_model_comparison_gate.py:220:    json.dump(side, open(paths[0] + ".screenmeta.json", "w"))
src/tests/test_gen_model_comparison_gate.py:289:    written = (root / "worklog" / "worklog_yixun" / "model_comparison.md").read_text()
src/tests/test_gen_model_comparison_gate.py:303:    written = (root / "worklog" / "worklog_yixun" / "model_comparison.md").read_text()
src/tests/test_gen_model_comparison_gate.py:314:    before = out.read_text()
src/tests/test_gen_model_comparison_gate.py:327:    assert out.read_text() == before, "the table was rewritten during a failed transaction"
src/tests/test_gen_model_comparison_gate.py:330:    written = out.read_text()
src/tests/test_gen_model_comparison_gate.py:346:    written = (root / "worklog" / "worklog_yixun" / "model_comparison.md").read_text()
src/tests/test_gen_model_comparison_gate.py:384:    before = out_file.read_text()
src/tests/test_gen_model_comparison_gate.py:388:    written = out_file.read_text()
src/tests/test_gen_model_comparison_gate.py:407:import importlib.util, sys, os
src/tests/test_gen_model_comparison_gate.py:408:spec = importlib.util.spec_from_file_location("gen", {_GEN_PY!r})
src/tests/test_gen_model_comparison_gate.py:409:G = importlib.util.module_from_spec(spec); spec.loader.exec_module(G)
src/tests/test_gen_model_comparison_gate.py:422:    written = (root / "worklog" / "worklog_yixun" / "model_comparison.md").read_text()
src/tests/test_gen_model_comparison_gate.py:469:    before = out.read_text()
src/tests/test_gen_model_comparison_gate.py:475:    assert out.read_text() == before, "a regressing regeneration still wrote the table"
src/tests/test_gen_model_comparison_gate.py:483:    before = out.read_text()
src/tests/test_gen_model_comparison_gate.py:487:    assert out.read_text() == before
src/tests/test_gen_model_comparison_gate.py:494:    before = out.read_text()
src/tests/test_gen_model_comparison_gate.py:498:    assert out.read_text() == before
src/tests/test_gen_model_comparison_gate.py:508:    written = out.read_text()
src/tests/test_gen_model_comparison_gate.py:526:    written = out.read_text()
src/tests/test_gen_model_comparison_gate.py:660:    written = (root / "worklog" / "worklog_yixun" / "model_comparison.md").read_text()
src/tests/test_gen_model_comparison_gate.py:675:    written = (root / "worklog" / "worklog_yixun" / "model_comparison.md").read_text()
src/tests/test_gen_model_comparison_gate.py:689:    written = (root / "worklog" / "worklog_yixun" / "model_comparison.md").read_text()
src/tests/test_exp14_fixed_mode_snapshot.py:36:_GOLDEN_PATH = Path(__file__).resolve().parent / "exp14_fixed_mode_golden.json"
src/tests/test_exp14_fixed_mode_snapshot.py:37:GOLDEN = json.loads(_GOLDEN_PATH.read_text())
src/tests/test_eval_paths.py:36:_EXP02_DIR = (Path(__file__).resolve().parents[2]
src/tests/test_eval_paths.py:152:    json.loads(json.dumps(rec))  # must be JSON-dumpable like the real call
src/tests/test_eval_paths.py:164:    json.loads(json.dumps(rec))
src/tests/test_eval_paths.py:314:    eval()/requires_grad_()/to(), a .diffusion with no pretransform, and a
src/tests/test_eval_paths.py:324:    def eval(self):
src/tests/test_eval_paths.py:394:    metrics_path = Path(calls[0]["result"]["metrics"])
src/tests/test_eval_paths.py:396:    saved = json.loads(metrics_path.read_text())
src/tests/test_eval_paths.py:438:    root = str(Path(__file__).resolve().parents[2])
src/tests/test_eval_paths.py:482:    json.loads(json.dumps(rec))
src/tests/test_eval_paths.py:528:    json.loads(json.dumps(rec))
src/tests/test_eval_paths.py:538:    json.loads(json.dumps(meta))
src/tests/test_eval_paths.py:556:    json.loads(json.dumps([rec, meta]))
src/tests/test_yaw_aug_training.py:13:one whole ``training_step`` on a seeded synthetic batch, recording the metadata
src/tests/test_yaw_aug_training.py:55:GOLDEN_PATH = Path(__file__).resolve().parent / "fixtures" / "exp15_yaw_aug_disabled_golden.json"
src/tests/test_yaw_aug_training.py:192:def _metadata_digests(metadata):
src/tests/test_yaw_aug_training.py:195:        for md in metadata
src/tests/test_yaw_aug_training.py:206:    batch -> one ``training_step``. Returns metadata-as-seen-by-the-conditioner,
src/tests/test_yaw_aug_training.py:219:    def _spy(metadata):
src/tests/test_yaw_aug_training.py:220:        seen["metadata"] = _metadata_digests(metadata)
src/tests/test_yaw_aug_training.py:221:        out = original(metadata)
src/tests/test_yaw_aug_training.py:229:    reals, metadata = _batch(2)
src/tests/test_yaw_aug_training.py:231:    loss = wrapper.training_step((reals, metadata), 0)
src/tests/test_yaw_aug_training.py:235:        "metadata_into_conditioner": seen["metadata"],
src/tests/test_yaw_aug_training.py:296:    golden = json.loads(GOLDEN_PATH.read_text())["record"]
src/tests/test_yaw_aug_training.py:299:    assert got["metadata_into_conditioner"] == golden["metadata_into_conditioner"], (
src/tests/test_yaw_aug_training.py:300:        "metadata reaching the conditioner changed on the yaw_aug-absent path"
src/tests/test_yaw_aug_training.py:591:    """Record the metadata list each ``_compute_conditioning`` call receives."""
src/tests/test_yaw_aug_training.py:595:    def _spy(metadata):
src/tests/test_yaw_aug_training.py:596:        seen.append(metadata)
src/tests/test_yaw_aug_training.py:597:        return original(metadata)
src/tests/test_yaw_aug_training.py:611:    ``rotate_scene_metadata`` — otherwise the test would only prove the code
src/tests/test_yaw_aug_training.py:723:    before = GOLDEN_PATH.read_bytes()
src/tests/test_yaw_aug_training.py:726:    assert GOLDEN_PATH.read_bytes() == before, "the golden fixture was overwritten"
src/tests/test_yaw_aug_training.py:741:    _, metadata = _batch(4, img_w=512)
src/tests/test_yaw_aug_training.py:744:    wrapper._apply_yaw_aug(metadata)
src/tests/test_yaw_aug_training.py:759:        reals, metadata = _batch(2)
src/tests/test_yaw_aug_training.py:760:        wrapper.training_step((reals, metadata), 0)
src/tests/test_yaw_aug_training.py:770:    reals, metadata = _batch(3, img_w=img_w)
src/tests/test_yaw_aug_training.py:772:                for md in metadata]
src/tests/test_yaw_aug_training.py:774:    wrapper.training_step((reals, metadata), 0)
src/tests/test_yaw_aug_training.py:791:    for md_in, md_now in zip(pristine, metadata):
src/tests/test_yaw_aug_training.py:794:                assert torch.equal(md_now[key], value), f"input metadata {key} mutated"
src/tests/test_yaw_aug_training.py:801:    reals, metadata = _batch(2, img_w=512)
src/tests/test_yaw_aug_training.py:803:    wrapper.training_step((reals, metadata), 0)
src/tests/test_yaw_aug_training.py:812:        reals, metadata = _batch(2, img_w=512)
src/tests/test_yaw_aug_training.py:813:        wrapper.training_step((reals, metadata), 0)
src/tests/test_yaw_aug_training.py:832:    reals, metadata = _batch(2, img_w=img_w)
src/tests/test_yaw_aug_training.py:834:                for md in metadata]
src/tests/test_yaw_aug_training.py:836:    wrapper.training_step((reals, metadata), 0)
src/tests/test_yaw_aug_training.py:854:def _run_enabled_step(metadata, img_w=512):
src/tests/test_yaw_aug_training.py:856:    n = len(metadata) if isinstance(metadata, (list, tuple)) else 1
src/tests/test_yaw_aug_training.py:858:    wrapper.training_step((reals, metadata), 0)
src/tests/test_yaw_aug_training.py:861:def test_guard_empty_metadata():
src/tests/test_yaw_aug_training.py:866:@pytest.mark.parametrize("bad", [None, {}, "metadata"])
src/tests/test_yaw_aug_training.py:867:def test_guard_metadata_not_a_list(bad):
src/tests/test_yaw_aug_training.py:913:    ``rotate_scene_metadata`` skips absent keys, so a sample missing one would be
src/tests/test_yaw_aug_training.py:952:    reals, metadata = _batch(2, img_w=512)
src/tests/test_yaw_aug_training.py:954:                for md in metadata]
src/tests/test_yaw_aug_training.py:956:    wrapper.validation_step((reals, metadata), 0)
src/tests/test_train_max_steps.py:201:    with open(_TRAIN_PY) as f:
src/tests/test_strip_optimizer_state.py:111:    out = torch.load(out_p, map_location="cpu", weights_only=False)
src/tests/test_strip_optimizer_state.py:152:    out = torch.load(out_p, map_location="cpu", weights_only=False)
src/tests/test_strip_optimizer_state.py:165:    reread = torch.load(src_p, map_location="cpu", weights_only=False)
src/tests/test_strip_optimizer_state.py:199:    stripped = torch.load(out_p, map_location="cpu", weights_only=False)
src/tests/test_strip_optimizer_state.py:263:    assert torch.load(out_p, map_location="cpu", weights_only=False)["optimizer_states"] != []
src/tests/test_strip_optimizer_state.py:293:    reread = torch.load(p, map_location="cpu", weights_only=False)
src/tests/test_strip_optimizer_state.py:359:    a = torch.load(once, map_location="cpu", weights_only=False)
src/tests/test_strip_optimizer_state.py:360:    b = torch.load(twice, map_location="cpu", weights_only=False)
src/tests/test_strip_optimizer_state.py:380:    out = torch.load(str(out_p), map_location="cpu", weights_only=False)
src/tests/test_strip_optimizer_state.py:396:    assert out_p.read_text() == "pre-existing"
src/tests/test_strip_optimizer_state.py:400:    out = torch.load(str(out_p), map_location="cpu", weights_only=False)
src/tests/test_exp11_equiv_probe.py:15:import importlib.util
src/tests/test_exp11_equiv_probe.py:33:    spec = importlib.util.spec_from_file_location("exp11_equiv_probe", _PROBE_PY)
src/tests/test_exp11_equiv_probe.py:35:    mod = importlib.util.module_from_spec(spec)
src/tests/test_exp11_wandb_readback.py:15:import importlib.util
src/tests/test_exp11_wandb_readback.py:34:    spec = importlib.util.spec_from_file_location("exp11_wandb_readback", _READBACK_PY)
src/tests/test_exp11_wandb_readback.py:36:    mod = importlib.util.module_from_spec(spec)
src/tests/test_exp11_wandb_readback.py:49:        with open(os.path.join(run_dir, "files", "wandb-metadata.json"), "w") as fh:
src/tests/test_exp11_wandb_readback.py:120:def test_absent_metadata_field_is_not_a_contradiction(tmp_path):
src/tests/test_exp11_wandb_readback.py:121:    """wandb-metadata does not always carry every field; only a CONTRADICTION fails."""
src/tests/test_exp11_wandb_readback.py:124:    run2 = _make_run(str(tmp_path / "b"))                  # no metadata file at all
src/tests/test_exp11_trajectory_figures.py:21:import importlib.util
src/tests/test_exp11_trajectory_figures.py:37:    spec = importlib.util.spec_from_file_location(name, os.path.join(_EXPDIR, f"{name}.py"))
src/tests/test_exp11_trajectory_figures.py:39:    mod = importlib.util.module_from_spec(spec)
src/tests/test_exp11_trajectory_figures.py:89:        with open(V.sidecar_path_for(str(p)), "w") as fh:
src/tests/test_exp11_trajectory_figures.py:226:    src = open(os.path.join(_EXPDIR, "gen_trajectory_pngs.py")).read()
src/tests/test_cyl_vit_patch_embedding.py:15:CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs/model_configs/FLAC/AR"
src/tests/test_cyl_vit_patch_embedding.py:65:        ).eval()
src/tests/test_cyl_vit_patch_embedding.py:98:        with open(CONFIG_DIR / "FLAC_AR_CylViT.json") as handle:
src/tests/test_cyl_vit_patch_embedding.py:99:            baseline = json.load(handle)
src/tests/test_cyl_vit_patch_embedding.py:106:                with open(CONFIG_DIR / filename) as handle:
src/tests/test_cyl_vit_patch_embedding.py:107:                    variant = json.load(handle)
src/tests/test_exp11_orbit_configs.py:54:    """``object_pairs_hook`` rejecting duplicate keys (plain ``json.load`` is
src/tests/test_exp11_orbit_configs.py:68:    with open(path, "r") as fh:
src/tests/test_exp11_orbit_configs.py:69:        return json.load(
src/tests/test_exp11_orbit_configs.py:232:    with open(_BF_CONFIG, "rb") as a, open(_arm_path("C4L"), "rb") as b:
src/tests/test_exp11_orbit_configs.py:298:    with open(path) as fh:
src/tests/test_yaw_aug_arm_config.py:23:_REPO = Path(__file__).resolve().parents[2]
src/tests/test_yaw_aug_arm_config.py:29:# BYTES, not text: read_text() applies universal-newline decoding, under which a
src/tests/test_yaw_aug_arm_config.py:50:    return ARM_CONFIG.read_bytes()
src/tests/test_yaw_aug_arm_config.py:55:    return CONTROL_CONFIG.read_bytes()
src/tests/test_yaw_aug_arm_config.py:109:    arm, control = json.loads(arm_text), json.loads(control_text)
src/tests/test_yaw_aug_arm_config.py:112:    stripped = json.loads(arm_text)
src/tests/test_yaw_aug_arm_config.py:123:    arm = json.loads(ARM_CONFIG.read_text())
src/tests/test_yaw_aug_arm_config.py:135:    arm = json.loads(ARM_CONFIG.read_text())
src/tests/test_yaw_aug_arm_config.py:153:    create_training_wrapper_from_config(json.loads(ARM_CONFIG.read_text()), object())
src/tests/test_yaw_aug_arm_config.py:165:    assert hashlib.sha256(CONTROL_CONFIG.read_bytes()).hexdigest() == (
src/tests/test_yaw_random_eval.py:151:# (4) exactness: every drawn angle round-trips through rotate_scene_metadata
src/tests/test_yaw_random_eval.py:153:def test_every_offset_round_trips_through_rotate_scene_metadata_quantisation():
src/tests/test_yaw_random_eval.py:154:    """``rotate_scene_metadata`` re-quantises its angle to ``dj`` columns. For the
src/tests/test_yaw_random_eval.py:214:    per_item = [yr.rotate_scene_metadata(md, a, img_w) for md, a in zip(batch, angles)]
src/tests/test_yaw_random_eval.py:216:        want = yr.rotate_scene_metadata(md, a, img_w)
src/tests/test_yaw_random_eval.py:229:    rot_a = yr.rotate_scene_metadata(md_a, yr.offsets_to_radians([10], img_w)[0], img_w)
src/tests/test_yaw_random_eval.py:230:    rot_b = yr.rotate_scene_metadata(md_b, yr.offsets_to_radians([11], img_w)[0], img_w)
src/tests/test_yaw_random_eval.py:477:    rotated = yr.rotate_scene_metadata(md, math.pi / 2, 512)
src/tests/test_yaw_random_eval.py:582:    """rotate_deg 0 must not even touch the metadata (and must not require a
src/tests/test_yaw_random_eval.py:593:    want = [yr.rotate_scene_metadata(md, math.radians(45.0), IMG_W) for md in batch]
src/tests/test_yaw_random_eval.py:625:        want = yr.rotate_scene_metadata(md, yr.offsets_to_radians([d], IMG_W)[0], IMG_W)
src/tests/test_yaw_random_eval.py:703:def test_drawn_offsets_reach_rotate_scene_metadata_over_a_two_batch_stream(monkeypatch):
src/tests/test_yaw_random_eval.py:707:    real = eval_FLAC.rotate_scene_metadata
src/tests/test_yaw_random_eval.py:713:    monkeypatch.setattr(eval_FLAC, "rotate_scene_metadata", spy)
src/tests/test_yaw_random_eval.py:755:    assert json.loads(json.dumps(rec))["rotate_deg"] is None
src/tests/test_yaw_random_eval.py:782:    json.loads(json.dumps(meta))
src/tests/test_yaw_random_eval.py:837:        def eval(self):
src/tests/test_yaw_random_eval.py:873:    saved = json.loads(out.read_text())
src/tests/test_yaw_random_eval.py:890:    saved = json.loads(out.read_text())
src/tests/test_yaw_random_eval.py:918:    root = str(Path(__file__).resolve().parents[2])
src/tests/test_yaw_random_eval.py:1089:    root = str(Path(__file__).resolve().parents[2])
src/tests/test_yaw_random_eval.py:1108:def test_yaw_column_shift_is_the_quantisation_rotate_scene_metadata_applies():
src/tests/test_yaw_random_eval.py:1118:    # ...and it is EXACTLY what rotate_scene_metadata rolls by: reconstructing the
src/tests/test_yaw_random_eval.py:1128:        got = yr.rotate_scene_metadata({"depth": depth}, alpha, IMG_W)["depth"]
src/tests/test_yaw_random_eval.py:1158:    json.loads(json.dumps(rec))
src/tests/test_yaw_random_eval.py:1181:    want = [yr.rotate_scene_metadata(md, math.radians(90.0), IMG_W) for md in batch]
src/tests/test_yaw_random_eval.py:1218:    payload = json.loads(side.read_text())
src/tests/test_yaw_random_eval.py:1235:    plain = (tmp_path / "toy_metrics_1_1.0_exp14_zref.json").read_text()
src/tests/test_yaw_random_eval.py:1240:    with_flag = (tmp_path / "toy_metrics_1_1.0_exp14_zref.json").read_text()
src/tests/test_yaw_random_eval.py:1245:    payload = json.loads(side.read_text())
src/tests/test_yaw_random_eval.py:1278:    root = str(Path(__file__).resolve().parents[2])
src/tests/test_yaw_random_eval.py:1332:    for reals, metadata in loader:
src/tests/test_yaw_random_eval.py:1333:        batch_sizes.append(len(metadata))
src/tests/test_yaw_random_eval.py:1334:        eval_FLAC.apply_rotation_plan(metadata, plan, g, stream)
src/tests/test_exp11_p0_collect.py:27:import importlib.util
src/tests/test_exp11_p0_collect.py:47:    spec = importlib.util.spec_from_file_location("exp11_p0_collect", _COLLECT_PY)
src/tests/test_exp11_p0_collect.py:49:    mod = importlib.util.module_from_spec(spec)
src/tests/test_exp11_p0_collect.py:465:    body = report.read_text()
src/tests/test_exp11_p0_collect.py:485:    body = (tmp_path / f"p0_report_{RUNID}.md").read_text()
src/tests/test_exp11_p0_collect.py:496:    body = out.read_text()
src/tests/test_cond_dispatch.py:131:    def __call__(self, conditioner, metadata, device, *args, **kwargs):
src/tests/test_cond_dispatch.py:133:        return conditioner(metadata, device)
src/tests/test_cond_dispatch.py:168:    reals, metadata = _batch(2)
src/tests/test_cond_dispatch.py:171:    wrapper.training_step((reals, metadata), 0)
src/tests/test_cond_dispatch.py:175:    wrapper.validation_step((reals, metadata), 0)
src/tests/test_cond_dispatch.py:179:    wrapper.test_step((reals, metadata), 0)
src/tests/test_cond_dispatch.py:192:    reals, metadata = _batch(2)
src/tests/test_cond_dispatch.py:193:    wrapper.training_step((reals, metadata), 0)
src/tests/test_finetune_cond.py:43:    with open(_FLAC_AR_CONFIG) as f:
src/tests/test_finetune_cond.py:44:        return json.load(f)
src/tests/test_exp11_p0_runner.py:15:import importlib.util
src/tests/test_exp11_p0_runner.py:32:    spec = importlib.util.spec_from_file_location("exp11_p0_runner", _RUNNER_PY)
src/tests/test_exp11_p0_runner.py:34:    mod = importlib.util.module_from_spec(spec)
src/tests/test_exp11_p0_runner.py:153:    with open(_RUNNER_PY) as fh:
src/tests/test_exp14_validate_cell.py:21:import importlib.util
src/tests/test_exp14_validate_cell.py:37:    spec = importlib.util.spec_from_file_location(name, os.path.join(_EXPDIR, f"{name}.py"))
src/tests/test_exp14_validate_cell.py:39:    mod = importlib.util.module_from_spec(spec)
src/tests/test_exp14_validate_cell.py:807:    open(p.replace(".json", ".stream.json"), "w").write("[]")
src/tests/test_exp14_validate_cell.py:808:    open(p + ".screenmeta.json", "w").write("[]")
src/tests/test_exp14_validate_cell.py:851:    raw = json.loads(open(_expect_file()).read())
src/tests/test_exp14_validate_cell.py:873:    open(p, "w").write(json.dumps(m))
src/tests/test_exp14_validate_cell.py:874:    open(p + ".screenmeta.json", "w").write(json.dumps(sm))
src/tests/test_exp14_validate_cell.py:875:    open(p.replace(".json", ".stream.json"), "w").write(json.dumps(st))
src/tests/test_bn_drift_probe.py:93:    bn = nn.BatchNorm2d(3).eval()
src/tests/test_bn_drift_probe.py:143:    stem = nn.Sequential(conv, bn).eval()
src/tests/test_bn_drift_probe.py:175:    bn = nn.BatchNorm2d(3).cuda().eval()
src/tests/test_bn_drift_probe.py:206:    mod.eval()
src/tests/test_bn_drift_probe.py:244:    """md_variant callable is applied once per metadata dict, in order, before
src/tests/test_bn_drift_probe.py:283:    mod = _Stack(20, c=4).eval()
src/training/autoencoders.py:638:        module.eval()
src/training/factory.py:107:            teacher_model = teacher_model.eval().requires_grad_(False)
src/training/factory.py:111:                teacher_model.load_state_dict(torch.load(teacher_model_ckpt)["state_dict"])
src/training/diffusion.py:22:    rotate_scene_metadata,
src/training/diffusion.py:370:    def _check_yaw_aug_metadata(self, metadata):
src/training/diffusion.py:380:        if not isinstance(metadata, (list, tuple)):
src/training/diffusion.py:382:                f"yaw_aug: metadata must be a list of per-sample dicts, got "
src/training/diffusion.py:383:                f"{type(metadata).__name__}"
src/training/diffusion.py:385:        if len(metadata) == 0:
src/training/diffusion.py:386:            raise ValueError("yaw_aug: metadata is empty; nothing to augment")
src/training/diffusion.py:388:        for i, md in enumerate(metadata):
src/training/diffusion.py:391:                    f"yaw_aug: metadata[{i}] must be a dict, got {type(md).__name__}"
src/training/diffusion.py:394:                raise ValueError(f"yaw_aug: metadata[{i}] has no 'depth' field")
src/training/diffusion.py:398:                    f"yaw_aug: metadata[{i}]['depth'] must be a tensor, got "
src/training/diffusion.py:403:                    f"yaw_aug: metadata[{i}]['depth'] must have shape [3, H, W], got "
src/training/diffusion.py:408:                    f"yaw_aug: metadata[{i}]['depth'] is {depth.shape[2]} columns "
src/training/diffusion.py:412:                # REQUIRED, not optional: rotate_scene_metadata skips absent keys,
src/training/diffusion.py:417:                    raise ValueError(f"yaw_aug: metadata[{i}] has no {key!r} pose field")
src/training/diffusion.py:421:                        f"yaw_aug: metadata[{i}][{key!r}] must be a tensor, got "
src/training/diffusion.py:427:                        f"yaw_aug: metadata[{i}][{key!r}] must have trailing "
src/training/diffusion.py:431:    def _apply_yaw_aug(self, metadata):
src/training/diffusion.py:445:        self._check_yaw_aug_metadata(metadata)
src/training/diffusion.py:451:        offsets = draw_yaw_offsets(len(metadata), self.yaw_aug_img_w, generator)
src/training/diffusion.py:454:            rotate_scene_metadata(md, alpha, self.yaw_aug_img_w)
src/training/diffusion.py:455:            for md, alpha in zip(metadata, angles)
src/training/diffusion.py:458:    def _compute_conditioning(self, metadata):
src/training/diffusion.py:470:                self.diffusion.conditioner, metadata, self.device, self.frame_avg_angles
src/training/diffusion.py:473:            return self.diffusion.conditioner(metadata, self.device)
src/training/diffusion.py:477:        reals, metadata = batch
src/training/diffusion.py:481:            metadata = self._apply_yaw_aug(metadata)
src/training/diffusion.py:496:        conditioning = self._compute_conditioning(metadata)
src/training/diffusion.py:502:        first_padding_mask = metadata[0]["padding_mask"]
src/training/diffusion.py:504:            padding_masks = torch.stack([md["padding_mask"][0] for md in metadata], dim=0).to(self.device) # Shape (batch_size, sequence_length)
src/training/diffusion.py:506:            padding_masks = torch.stack([md["padding_mask"] for md in metadata], dim=0).to(self.device) # Shape (batch_size, sequence_length)
src/training/diffusion.py:626:        reals, metadata = batch
src/training/diffusion.py:634:            conditioning = self._compute_conditioning(metadata)
src/training/diffusion.py:639:        # # Create batch tensor of attention masks from the "mask" field of the metadata array
src/training/diffusion.py:641:        #     padding_masks = torch.stack([md["padding_mask"][0] for md in metadata], dim=0).to(self.device) # Shape (batch_size, sequence_length)
src/training/diffusion.py:716:        reals, metadata = batch
src/training/diffusion.py:728:            conditioning = self._compute_conditioning(metadata)
src/training/diffusion.py:760:            scene_list = [md["scene"] for md in metadata]
src/training/diffusion.py:761:            depth_list = [md["depth"] if 'depth' in md else None for md in metadata]
src/training/diffusion.py:762:            query_list = [md["source"] if 'source' in md else None for md in metadata]
src/models/conditioners.py:105:        self.pretransform.eval()
src/models/conditioners.py:367:    def forward(self, batch_metadata: tp.List[tp.Dict[str, tp.Any]], device: tp.Union[torch.device, str], only_ids: tp.Optional[tp.Iterable[str]] = None) -> tp.Dict[str, tp.Any]:
src/models/conditioners.py:381:            for x in batch_metadata:
src/models/conditioners.py:386:                        raise ValueError(f"Conditioner key {condition_key} not found in batch metadata")
src/models/conditioners.py:391:                        raise ValueError(f"Conditioner {key} requires depth input, but it is not present in the batch metadata")
src/models/conditioners.py:455:                        vit_model = AutoModel.from_config(AutoConfig.from_pretrained(model_name_or_path))
src/models/conditioners.py:458:                        vit_model = AutoModel.from_pretrained(model_name_or_path)
src/models/pretransforms.py:33:        self.model.requires_grad_(False).eval()
src/models/factory.py:18:    with open(model_config_path) as f:
src/models/factory.py:19:        model_config = json.load(f)
src/models/factory.py:50:    pretransform.eval().requires_grad_(pretransform.enable_grad)
src/models/utils.py:2:from safetensors.torch import load_file
src/models/utils.py:25:        state_dict = load_file(ckpt_path)
src/models/utils.py:27:        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)["state_dict"]
src/models/pretrained.py:12:    with open(model_config_path) as f:
src/models/pretrained.py:13:        model_config = json.load(f)
src/data/yaw_rotation.py:5:This module applies that rotation to the *conditioning* metadata only: the
src/data/yaw_rotation.py:78:    integer column shift is applied by :func:`rotate_scene_metadata` with no
src/data/yaw_rotation.py:130:    Exact by construction on the column grid — :func:`rotate_scene_metadata`
src/data/yaw_rotation.py:183:        One per-sample metadata dict. ``'source'`` (shape ``[3]``) and, optionally,
src/data/yaw_rotation.py:252:    This is the rule :func:`rotate_scene_metadata` applies, extracted so that code
src/data/yaw_rotation.py:294:def rotate_scene_metadata(
src/data/yaw_rotation.py:301:    Apply a physically-consistent yaw rotation to a single sample's metadata.
src/data/yaw_rotation.py:313:        One per-sample metadata dict as produced by the AR/HAA metadata module.
src/data/yaw_rotation.py:329:        A shallow-copied metadata dict with ``depth`` and the selected pose fields

exec
/usr/bin/bash -lc "sed -n '1,340p' worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py" in /n/fs/gatrdp/codespace/FLAC
 succeeded in 0ms:
#!/usr/bin/env python3
"""exp_11 RESTART checkpoint preflight (round-3 review B2).

exp_10's `bf_resume_launch.sh` proved a restart with ONE CPU-side ``torch.load``
before spending an allocation; round 3 shipped only a path check, which would
have accepted a zero-byte file or a renamed C4 checkpoint under the C16 root
(orbit size does not change the module tree, so it would even have loaded). This
restores the exp_10 depth for the sweep:

  - the checkpoint's embedded ``global_step`` equals EXPECTED_STEP exactly;
  - its embedded ``model_config`` deep-equals this arm's config file, so the
    orbit, conditioning method and architecture it was trained under are the
    ones this run would continue;
  - full warm optimizer state (non-empty ``state``), ``lr_schedulers``, and EMA
    weights are present — a stripped/weights-only file is the wrong file, since
    exp_11 has no optimizer-reset lineage;
  - the target budget still lies ahead (``global_step < max_steps``), so a
    "restart" cannot terminate immediately on Lightning's ``>=`` stop rule while
    printing the completion literal;
  - optionally, the arm's ORIGINAL launch manifest is re-read and the restart is
    bound to the same rung, commit and budget.

Prints the checkpoint sha256 (for the restart manifest) and a lineage summary.
Exit 0 = admissible; nonzero = refuse to launch.
"""
import argparse
import hashlib
import json
import os
import sys


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_ckpt_config(path):
    """The model_config embedded in a Lightning checkpoint, on CPU.

    Shared with the screen driver (fa_orbit_screen.sbatch), which asserts the
    checkpoint's own orbit before spending an evaluation on it: a screen that
    silently evaluated the wrong arm's checkpoint would poison a futility gate."""
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(ck, dict):
        raise RuntimeError(f"not a Lightning checkpoint: {path}")
    cfg = ck.get("model_config")
    if not isinstance(cfg, dict):
        raise RuntimeError(f"checkpoint carries no embedded model_config: {path}")
    return cfg, ck.get("global_step")


def load_ckpt_state_keys(path):
    """The checkpoint's state_dict KEYS (used to prove EMA weights exist before a
    screen spends a GPU: eval_FLAC silently evaluates online weights when the EMA
    entries are absent)."""
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(ck, dict):
        raise RuntimeError(f"not a Lightning checkpoint: {path}")
    return list((ck.get("state_dict") or {}).keys())


def parse_manifest(path):
    """The launcher's own manifest format: whitespace-separated `key value...`."""
    out = {}
    with open(path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, _, rest = line.partition(" ")
            out[key] = rest.strip()
    return out


def check_manifest_binding(manifest_path, arm, rung, commit, maxsteps):
    man = parse_manifest(manifest_path)
    problems = []
    fields = man.get("arm", "")
    # `arm <ARM> rung <RUNG> micro <MB> ngpu <N> max_steps <S> ...`
    tokens = ("arm " + fields).split()
    kv = {tokens[i]: tokens[i + 1] for i in range(0, len(tokens) - 1, 2)}
    if kv.get("arm") != arm:
        problems.append(f"manifest arm {kv.get('arm')!r} != {arm!r}")
    if kv.get("rung") != rung:
        problems.append(f"manifest rung {kv.get('rung')!r} != {rung!r} "
                        "(a restart may not change the rung: it would change rank count, "
                        "sampler partitioning and worker seeding mid-lineage)")
    if kv.get("max_steps") != str(maxsteps):
        problems.append(f"manifest max_steps {kv.get('max_steps')!r} != {maxsteps}")
    # Fail-CLOSED (round-3 B2 residual): an absent or empty manifest commit is not
    # "no opinion", it is missing provenance — the restart must not proceed on it.
    man_commit = man.get("commit", "").strip()
    if not man_commit:
        problems.append("launch manifest carries no 'commit' line — cannot bind the restart "
                        "to the lineage that produced this checkpoint")
    elif not commit:
        problems.append("no running commit supplied to compare against the manifest commit")
    elif man_commit != commit:
        problems.append(f"manifest commit {man_commit[:12]} != running commit {commit[:12]}")
    return problems, man


def kv_line(man, key):
    """One manifest line's `k v k v ...` pairs (the launcher's `arm ...`/`job ...`)."""
    f = (f"{key} " + man.get(key, "")).split()
    return {f[i]: f[i + 1] for i in range(0, len(f) - 1, 2)}


def canonical_ckpt_dir(save_dir, arm, repo_root):
    """<save_dir>/FLAC_exp11_<ARM>/exp11_<ARM>/checkpoints, as the launcher builds it.

    save_dir is recorded relative to the repo root, so it is resolved against it."""
    base = save_dir if os.path.isabs(save_dir) else os.path.join(repo_root, save_dir)
    return os.path.realpath(os.path.join(base, f"FLAC_exp11_{arm}", f"exp11_{arm}", "checkpoints"))


def check_extension_binding(manifest_path, registry_path, arm, rung, config_path, ckpt_path,
                            ckpt_sha, expected_step, max_steps, repo_root="."):
    """The 40k -> 100k EXTENSION contract (re-pin review, required fix 1).

    A crash restart continues the SAME launch: same budget, same reviewed commit,
    so `check_manifest_binding` demands both. An extension breaks both BY DESIGN
    — it raises the budget from 40000 to 100000 and runs later reviewed code —
    and demanding equality there is exactly what gave jobs 3662828-30 their third
    hard-abort path.

    What an extension must still prove is the ORIGINAL LAUNCH IDENTITY, and it
    proves it against the COMMITTED registry rather than the mutable manifest
    alone: the INITIAL manifest byte-for-byte as audited, the same job/uuid/
    launch commit/rung/config/save-dir/training seed, and a resumed checkpoint
    that IS that launch's audited final checkpoint, sitting in that launch's own
    canonical run directory. Budget and running commit may move; nothing that
    identifies the run may.
    """
    problems = []
    if not os.path.isfile(registry_path):
        return [f"audited launch registry not found: {registry_path}"], {}
    reg = json.load(open(registry_path)).get("arms", {}).get(arm)
    if reg is None:
        return [f"{arm} is not in the audited launch registry {registry_path}"], {}
    man = parse_manifest(manifest_path)
    kv, jkv = kv_line(man, "arm"), kv_line(man, "job")

    got_sha = sha256_file(manifest_path)
    if got_sha != reg.get("manifest_sha256"):
        problems.append(f"launch manifest sha256 {got_sha[:12]} != audited "
                        f"{str(reg.get('manifest_sha256'))[:12]} — the manifest changed after it "
                        "was registered")
    for label, got_v, want_v in (("arm", kv.get("arm"), arm),
                                 ("job", jkv.get("job"), reg.get("job")),
                                 ("launch mode", jkv.get("mode"), "INITIAL"),
                                 ("launch_uuid", jkv.get("launch_uuid"), reg.get("launch_uuid")),
                                 ("rung", kv.get("rung"), reg.get("rung")),
                                 ("rung (this run)", rung, reg.get("rung")),
                                 ("config_sha256", man.get("config_sha256"), reg.get("config_sha256")),
                                 ("save_dir", man.get("save_dir"), reg.get("save_dir"))):
        if got_v != want_v:
            problems.append(f"{label} {got_v!r} != registered {want_v!r}")
    man_commit = man.get("commit", "").strip()
    if not man_commit:
        problems.append("launch manifest carries no 'commit' line — cannot bind the extension to "
                        "the lineage that produced this checkpoint")
    elif man_commit != reg.get("commit"):
        problems.append(f"manifest commit {man_commit[:12]} != the registered launch commit "
                        f"{str(reg.get('commit'))[:12]}")
    if int(reg.get("training_seed", -1)) != 42:
        problems.append(f"registered training seed {reg.get('training_seed')!r} != 42")
    # The INITIAL budget is the manifest's and the registry's; the extension's is
    # this run's, and it must strictly cover the resume point without shrinking.
    initial_budget = reg.get("max_steps")
    if kv.get("max_steps") != initial_budget:
        problems.append(f"manifest max_steps {kv.get('max_steps')!r} != registered "
                        f"{initial_budget!r} (the INITIAL budget, which an extension preserves)")
    try:
        if max_steps < int(initial_budget):
            problems.append(f"extension budget {max_steps} does not extend the registered "
                            f"{initial_budget} — an extension may only raise the budget")
    except (TypeError, ValueError):
        problems.append(f"registered max_steps {initial_budget!r} is not an integer")
    if sha256_file(config_path) != reg.get("config_sha256"):
        problems.append(f"{config_path} sha256 != the registered config_sha256 "
                        f"{str(reg.get('config_sha256'))[:12]}")
    # the resumed checkpoint IS the audited anchor, in the audited run directory
    anchor, final_step = reg.get("final_ckpt_sha256"), reg.get("final_step")
    if not anchor:
        problems.append(f"{arm} has no audited final_ckpt_sha256 in the registry — the extension "
                        "has nothing to chain to (audit the arm's final checkpoint first)")
    elif ckpt_sha != anchor:
        problems.append(f"resume checkpoint sha256 {ckpt_sha[:12]} != the audited final checkpoint "
                        f"{anchor[:12]} — this leg does not continue that run")
    if final_step is not None and int(final_step) != int(expected_step):
        problems.append(f"EXPECTED_STEP {expected_step} != the registered final_step {final_step}")
    save_dir = man.get("save_dir", "")
    if not save_dir:
        problems.append("manifest records no save_dir")
    else:
        canon = canonical_ckpt_dir(save_dir, arm, repo_root)
        if os.path.realpath(os.path.dirname(ckpt_path)) != canon:
            problems.append(f"resume checkpoint {ckpt_path} does not live in the registered "
                            f"launch's canonical run directory {canon}")
    return problems, man


def main(argv=None):
    ap = argparse.ArgumentParser(description="exp_11 restart checkpoint preflight")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--expected-step", type=int, required=True)
    ap.add_argument("--config", required=True, help="this arm's model config json")
    ap.add_argument("--max-steps", type=int, required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--rung", required=True)
    ap.add_argument("--commit", default="")
    ap.add_argument("--launch-manifest", default="",
                    help="the arm's original launch manifest (binds rung/commit/budget)")
    ap.add_argument("--extension", action="store_true",
                    help="this restart is the Q10 40k->100k EXTENSION: bind it to the audited "
                         "INITIAL launch identity instead of requiring an equal budget/commit")
    ap.add_argument("--launch-registry", default="",
                    help="the committed arm launch registry (required with --extension)")
    ap.add_argument("--repo-root", default=".",
                    help="root the registry's relative save_dir is resolved against")
    args = ap.parse_args(argv)
    if args.extension and not args.launch_registry:
        ap.error("--extension requires --launch-registry (the audited INITIAL launch row)")

    if not os.path.isfile(args.ckpt):
        print(f"PREFLIGHT: checkpoint not found: {args.ckpt}")
        return 2

    import torch  # deferred: keeps --help and unit imports cheap

    try:
        ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    except Exception as exc:                      # truncated, empty or not a torch file
        print(f"PREFLIGHT: {args.ckpt} is not loadable as a checkpoint: "
              f"{type(exc).__name__}: {exc}")
        return 2
    if not isinstance(ck, dict):
        print(f"PREFLIGHT: not a Lightning checkpoint: {args.ckpt}")
        return 2

    problems = []
    gs = ck.get("global_step")
    if gs != args.expected_step:
        problems.append(f"global_step {gs} != EXPECTED_STEP {args.expected_step}")
    if isinstance(gs, int) and gs >= args.max_steps:
        problems.append(f"global_step {gs} >= max_steps {args.max_steps}: Lightning would stop "
                        "immediately and still print the completion literal")

    mc = ck.get("model_config")
    if not isinstance(mc, dict):
        problems.append("checkpoint carries no embedded 'model_config' dict")
    else:
        want = json.load(open(args.config))
        if mc != want:
            tr = mc.get("training", {}) if isinstance(mc.get("training"), dict) else {}
            problems.append(
                f"embedded model_config != {args.config} (parsed-object mismatch; embedded "
                f"cond_method={tr.get('cond_method')!r} angles={tr.get('frame_avg_angles')!r})")

    opts = ck.get("optimizer_states") or []
    if len(opts) != 1:
        problems.append(f"expected exactly 1 optimizer entry, found {len(opts)}")
    elif not opts[0].get("state"):
        problems.append("optimizer state is CLEARED (stripped checkpoint); exp_11 restarts are "
                        "WARM continuations and have no optimizer-reset lineage")
    if not ck.get("lr_schedulers"):
        problems.append("no 'lr_schedulers' -> PL 2.1 KeyErrors on resume")
    sd = ck.get("state_dict") or {}
    n_ema = sum(1 for k in sd if k.startswith("diffusion_ema."))
    if not n_ema:
        problems.append("no EMA weights in state_dict")

    digest = sha256_file(args.ckpt)     # needed by the extension contract's anchor check
    man = {}
    if args.launch_manifest:
        if not os.path.isfile(args.launch_manifest):
            problems.append(f"launch manifest not found: {args.launch_manifest}")
        elif args.extension:
            more, man = check_extension_binding(
                args.launch_manifest, args.launch_registry, args.arm, args.rung, args.config,
                args.ckpt, digest, args.expected_step, args.max_steps, args.repo_root)
            problems += more
        else:
            more, man = check_manifest_binding(args.launch_manifest, args.arm, args.rung,
                                               args.commit, args.max_steps)
            problems += more
    elif args.extension:
        problems.append("--extension requires --launch-manifest (the audited INITIAL manifest)")

    if problems:
        print("PREFLIGHT FAILED:")
        for p in problems:
            print(f"  !! {p}")
        return 2

    tr = mc.get("training", {})
    print(f"restart lineage OK: {args.ckpt}")
    print(f"  global_step={gs} epoch={ck.get('epoch')} cond_method={tr.get('cond_method')!r} "
          f"angles={tr.get('frame_avg_angles')}")
    print(f"  optimizer_state=FULL ({len(opts[0]['state'])} entries) "
          f"lr={opts[0]['param_groups'][0].get('lr')} "
          f"sched_last_epoch={ck['lr_schedulers'][0].get('last_epoch')} ema_entries={n_ema}")
    if man and args.extension:
        print(f"extension lineage OK: {args.arm} {args.expected_step} -> {args.max_steps} continues "
              f"the audited launch job {kv_line(man, 'job').get('job')} "
              f"(launch commit {man.get('commit', '')[:12]}, running commit {args.commit[:12] or '<none>'})")
        print(f"  bound to the audited launch manifest: {args.launch_manifest}")
    elif man:
        print(f"  bound to launch manifest: {args.launch_manifest}")
    print(f"CKPT_SHA256 {digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

exec
/usr/bin/bash -lc "sed -n '1,230p' worklog/worklog_yixun/exp_11_fa_orbit_claude/assert_arm_configs_exp11.py" in /n/fs/gatrdp/codespace/FLAC
 succeeded in 0ms:
#!/usr/bin/env python3
"""exp_11 pre-launch audit — DINOv3 pin + per-arm wiring + init-identity.

Replicated from exp_07's ``assert_arm_configs.py`` (which does NOT generalize:
it hardcodes ``FLAC_AR_BV.json``/``FLAC_AR_BF.json``, the C4 orbit and the
vanilla-vs-fa pairing). Same checks, re-pointed at the exp_11 arm configs and
extended to the orbit sweep:

  - DINOv3 initializer pin: the ViT is trainable, so its init weights are
    lineage-relevant; the HF cache must hold EXACTLY the pinned snapshot
    (constants copied verbatim from exp_07's gate — the single upstream source).
  - FACTORY wiring for the requested arm: cond_method 'fa_invariant', the arm's
    uniform orbit, EMA on, cfg_dropout 0.1, log_snr(-1.2, 2.0), and a real
    configure_optimizers() whose AdamW/InverseLR objects are field-checked.
  - Architecture identity against the C4L bridge arm: identical parameter names
    and count (the orbit changes forward-time conditioning only).
  - Seeded init-identity: built under the same seed, this arm's state_dict must
    hash identically to C4L's — so every arm starts from the same weights and
    the sweep's only delta is the averaging orbit.

Explicit raises (not ``assert``) everywhere the outcome gates a launch, so the
gate survives an inherited ``PYTHONOPTIMIZE``.

Run from the repo root:
    HF_HUB_OFFLINE=1 python worklog/worklog_yixun/exp_11_fa_orbit_claude/assert_arm_configs_exp11.py C8
"""
import hashlib
import json
import os
import random
import sys


def _repo_root(p):  # marker-walk (same helper as exp_07's gate)
    p = os.path.abspath(p)
    # `.git` is a DIRECTORY in a normal checkout and a FILE in a linked worktree —
    # measurements run from a pinned worktree, so both must count as the root.
    while not os.path.exists(os.path.join(p, ".git")):
        parent = os.path.dirname(p)
        if parent == p:
            raise RuntimeError("repo root (.git) not found")
        p = parent
    return p


REPO = _repo_root(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)  # guard against a stale pip-installed src copy

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.models.factory import create_model_from_config  # noqa: E402
from src.training.factory import create_training_wrapper_from_config  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 42
REFERENCE_ARM = "C4L"                       # the bridge arm every arm must match
# VANL is the vanilla-conditioning arm of the SAME recipe (Q9). Its orbit is
# None, not 1: it performs no frame averaging at all, which is exactly the single
# delta that makes VANL-vs-C4L a clean fa-vs-vanilla comparison in this lineage.
ARM_ORBIT = {"C4L": 4, "C8": 8, "C16": 16, "C32": 32, "VANL": None}
ARM_CONFIG = {a: f"FLAC_AR_BF_{a}.json" for a in ("C4L", "C8", "C16", "C32")}
ARM_CONFIG["VANL"] = "FLAC_AR_VANCKPT.json"

# Verbatim from worklog/worklog_yixun/exp_07_fa_scratch_claude/assert_arm_configs.py
VIT_REV = "114c1379950215c8b35dfcd4e90a5c251dde0d32"
VIT_SHA256 = "4610ad75edef83e75afdebf162d148dc628045ea6cbb83d67d4708c709c4f91d"


def assert_vit_pin():
    from huggingface_hub.constants import HF_HUB_CACHE
    hub = os.path.join(HF_HUB_CACHE, "models--facebook--dinov3-vits16-pretrain-lvd1689m")
    snap_dir = os.path.join(hub, "snapshots")
    if not os.path.isdir(snap_dir):
        raise RuntimeError(f"DINOv3 cache missing at {snap_dir} — refuse to launch")
    snaps = sorted(os.listdir(snap_dir))
    if snaps != [VIT_REV]:
        raise RuntimeError(
            f"DINOv3 cache snapshots {snaps} != pinned [{VIT_REV!r}] — refuse to launch")
    st = os.path.join(snap_dir, VIT_REV, "model.safetensors")
    h = hashlib.sha256()
    with open(st, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    if h.hexdigest() != VIT_SHA256:
        raise RuntimeError(
            f"DINOv3 model.safetensors sha256 {h.hexdigest()} != pinned {VIT_SHA256}")
    print(f"ViT pin OK: cache {HF_HUB_CACHE}, single snapshot {VIT_REV[:12]}…, "
          f"sha256 {VIT_SHA256[:12]}… (launch with HF_HUB_OFFLINE=1)")


def build(arm):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    cfg = json.load(open(os.path.join(HERE, ARM_CONFIG[arm])))
    model = create_model_from_config(cfg)
    wrapper = create_training_wrapper_from_config(cfg, model)
    return cfg, model, wrapper


def state_hash(model):
    h = hashlib.sha256()
    sd = model.state_dict()
    for k in sorted(sd):
        h.update(k.encode())
        h.update(sd[k].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def check_recipe(tag, w, n_angles):
    """Everything except the conditioning must be identical across arms.

    ``n_angles is None`` selects the VANILLA variant: the assertion is then that
    the wrapper does NOT frame-average — the mirror image of the orbit check, and
    the thing that makes VANL a baseline rather than a differently-flavoured fa
    run. Every other line below is shared, which is the point: one delta."""
    if n_angles is None:
        # The wrapper ALWAYS holds a frame_avg_angles tuple — src/training/diffusion.py
        # falls back to DEFAULT_FRAME_ANGLES when the config omits the key — but the
        # conditioning dispatch reads it only on the fa_invariant branch (diffusion.py
        # ~215-221); under "vanilla" it goes straight to the vanilla path and the
        # tuple is never consumed. So the assertion that matters is the DISPATCH, not
        # the presence of an inert default. (The config itself is separately required
        # to carry no orbit keys — checked below and in the launcher's semantic gate.)
        if w.cond_method != "vanilla":
            raise RuntimeError(f"{tag}: cond_method {w.cond_method!r} != 'vanilla' — this arm "
                               "would frame-average, which is the very thing it exists not to do")
    else:
        if w.cond_method != "fa_invariant":
            raise RuntimeError(f"{tag}: cond_method {w.cond_method!r} != 'fa_invariant'")
        want = tuple(k * 360.0 / n_angles for k in range(n_angles))
        got = tuple(w.frame_avg_angles)
        if got != want:
            raise RuntimeError(f"{tag}: frame_avg_angles {got} != the uniform C{n_angles} orbit {want}")
    if w.diffusion_ema is None:
        raise RuntimeError(f"{tag}: EMA missing")
    if w.cfg_dropout_prob != 0.1:
        raise RuntimeError(f"{tag}: cfg_dropout_prob {w.cfg_dropout_prob} != 0.1")
    if w.timestep_sampler != "log_snr":
        raise RuntimeError(f"{tag}: timestep_sampler {w.timestep_sampler!r} != 'log_snr'")
    if (w.mean_logsnr, w.std_logsnr) != (-1.2, 2.0):
        raise RuntimeError(f"{tag}: log_snr ({w.mean_logsnr}, {w.std_logsnr}) != (-1.2, 2.0)")
    if w.optimizer_configs["diffusion"]["optimizer"]["config"] != {
            "lr": 5e-5, "betas": [0.9, 0.999], "weight_decay": 1e-3}:
        raise RuntimeError(f"{tag}: optimizer config drifted")

    opts, scheds = w.configure_optimizers()
    opt, pg = opts[0], opts[0].param_groups[0]
    if type(opt).__name__ != "AdamW":
        raise RuntimeError(f"{tag}: optimizer {type(opt).__name__} != AdamW")
    step0_lr = (1 - 0.99 ** 1) * 5e-5          # InverseLR applies its step-0 multiplier at build
    if abs(pg["lr"] - step0_lr) > 1e-18 or pg["initial_lr"] != 5e-5:
        raise RuntimeError(f"{tag}: lr {pg['lr']} / initial_lr {pg.get('initial_lr')} unexpected")
    if tuple(pg["betas"]) != (0.9, 0.999) or pg["weight_decay"] != 1e-3:
        raise RuntimeError(f"{tag}: betas/wd drifted: {pg['betas']}, {pg['weight_decay']}")
    sc = scheds[0]
    sched = sc["scheduler"]
    if sc["interval"] != "step" or type(sched).__name__ != "InverseLR":
        raise RuntimeError(f"{tag}: scheduler {type(sched).__name__} @ {sc['interval']}")
    if (sched.inv_gamma, sched.power, sched.warmup) != (1000000, 0.5, 0.99):
        raise RuntimeError(f"{tag}: InverseLR fields drifted")
    cond = "vanilla (no orbit)" if n_angles is None else f"fa_invariant C{n_angles} orbit"
    print(f"{tag}: wiring OK — {cond}, EMA on, cfg_dropout 0.1, "
          f"log_snr(-1.2,2.0), AdamW(initial_lr 5e-5, step-0 {pg['lr']:.3e}) + InverseLR@step")


def main(argv):
    if len(argv) != 2 or argv[1] not in ARM_ORBIT:
        raise SystemExit(f"usage: {os.path.basename(argv[0])} <{'|'.join(ARM_ORBIT)}>")
    arm = argv[1]

    assert_vit_pin()

    what = "vanilla" if ARM_ORBIT[arm] is None else f"fa_invariant C{ARM_ORBIT[arm]}"
    print(f"building {arm} ({what}) ...")
    if ARM_ORBIT[arm] is None:
        # mirror of the launcher's semantic gate: the CONFIG must be orbit-free
        raw = json.load(open(os.path.join(HERE, ARM_CONFIG[arm]))).get("training", {})
        if raw.get("cond_method") not in (None, "vanilla"):
            raise RuntimeError(f"{arm} config declares cond_method={raw.get('cond_method')!r}")
        if "frame_avg_angles" in raw:
            raise RuntimeError(f"{arm} config carries frame_avg_angles={raw['frame_avg_angles']!r} — "
                               "a vanilla arm's config has no orbit")
        print(f"{arm} config is orbit-free (no cond_method, no frame_avg_angles)")
    _, model_a, wrap_a = build(arm)
    check_recipe(arm, wrap_a, ARM_ORBIT[arm])

    if arm == REFERENCE_ARM:
        print(f"init identity: {arm} IS the reference arm; state_dict sha256 "
              f"{state_hash(model_a)[:16]}… under seed {SEED}")
        print(f"\nALL ASSERTS PASSED — {arm} is launch-ready.")
        return 0

    print(f"building the {REFERENCE_ARM} reference for init-identity ...")
    _, model_r, wrap_r = build(REFERENCE_ARM)
    check_recipe(REFERENCE_ARM, wrap_r, ARM_ORBIT[REFERENCE_ARM])

    names_a = [n for n, _ in model_a.named_parameters()]
    names_r = [n for n, _ in model_r.named_parameters()]
    count_a = sum(p.numel() for p in model_a.parameters())
    count_r = sum(p.numel() for p in model_r.parameters())
    if names_a != names_r:
        raise RuntimeError("parameter-name sets differ between the arms")
    if count_a != count_r:
        raise RuntimeError(f"parameter counts differ: {arm} {count_a} vs {REFERENCE_ARM} {count_r}")
    print(f"architecture: identical param names ({len(names_a)} tensors) and count "
          f"({count_a / 1e6:.2f}M) in {arm} and {REFERENCE_ARM}")

    ha, hr = state_hash(model_a), state_hash(model_r)
    if ha != hr:
        raise RuntimeError(f"init state hashes differ:\n  {arm} {ha}\n  {REFERENCE_ARM} {hr}")
    print(f"init identity: state_dict sha256 match under seed {SEED}: {ha[:16]}…")
    delta = ("only in whether the conditioning is frame-averaged at all — this is the "
             "fa-vs-vanilla single delta (Q9)") if ARM_ORBIT[arm] is None else \
            "only in the averaging orbit"
    print(f"\nALL ASSERTS PASSED — {arm} is init-identical to {REFERENCE_ARM} and differs {delta}.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

exec
/usr/bin/bash -lc "git ls-files -s -- weights/FLAC/VAE.safetensors weights src/configs/dataset_configs/AR/train/acousticroom_train.json pyproject.toml setup.py setup.cfg; stat -c '%F %a %n' weights/FLAC/VAE.safetensors src/configs/dataset_configs/AR/train/acousticroom_train.json worklog/worklog_yixun/exp_11_fa_orbit_claude/FLAC_AR_BF_C4L.json 2>&1" in /n/fs/gatrdp/codespace/FLAC
 succeeded in 0ms:
100644 3852e8dc546b1991506c3c887b32d51a25464f52 0	pyproject.toml
100644 b2390ef56cc76829c357b57094d10bba8d8ac8db 0	src/configs/dataset_configs/AR/train/acousticroom_train.json
100644 c911f0250f29e641e9d28a5ecf958955cf497594 0	weights/FLAC/FLAC_EMA_metrics_1_1.0_exp01_unseen_K1_seed42.json
100644 4c9c35340eeab63be4ec4736058cd03a51f56763 0	weights/FLAC/FLAC_EMA_metrics_1_1.0_exp01_unseen_K1_seed43.json
100644 bcd2c2d430f9bfe9b6584f23551f122fc72c1287 0	weights/FLAC/FLAC_EMA_metrics_1_1.0_exp01_unseen_K1_seed44.json
100644 c88e87f816a6bd70866bdcc634c2d87494f25a28 0	weights/FLAC/FLAC_EMA_metrics_1_1.0_exp01_unseen_K1_seed45.json
100644 2a734506f90ade55fcb9810b303dc9244c753a7e 0	weights/FLAC/FLAC_EMA_metrics_1_1.0_exp01_unseen_K1_seed46.json
100644 bfee871ae02d410a9fd71065fd591ffa75b19f54 0	weights/FLAC/FLAC_EMA_metrics_1_1.0_exp01_unseen_K8_seed42.json
100644 1aadc5969874bf28586041e34ffc74d19f6c9ded 0	weights/FLAC/FLAC_EMA_metrics_1_1.0_exp01_unseen_K8_seed43.json
100644 a14d6cc74959473c1bcedb54568af6fbbcb9115a 0	weights/FLAC/FLAC_EMA_metrics_1_1.0_exp01_unseen_K8_seed44.json
100644 e6a083be30ef1f36eb75567d08def438d4db382c 0	weights/FLAC/FLAC_EMA_metrics_1_1.0_exp01_unseen_K8_seed45.json
100644 68cc8f012b44e47094658f599e91fa9e66e59f11 0	weights/FLAC/FLAC_EMA_metrics_1_1.0_exp01_unseen_K8_seed46.json
regular file 664 weights/FLAC/VAE.safetensors
regular file 644 src/configs/dataset_configs/AR/train/acousticroom_train.json
regular file 644 worklog/worklog_yixun/exp_11_fa_orbit_claude/FLAC_AR_BF_C4L.json

exec
/usr/bin/bash -lc "sed -n '1,260p' src/configs/dataset_configs/AR/train/acousticroom_train.json; sed -n '1,320p' worklog/worklog_yixun/exp_11_fa_orbit_claude/FLAC_AR_BF_C4L.json" in /n/fs/gatrdp/codespace/FLAC
 succeeded in 0ms:
{
    "dataset_type": "audio_dir",
    "datasets": [
        {
            "id": "AcousticRooms",
            "path": "AcousticRooms",
            "json_file_path": "data/AR/train.json",
            "custom_metadata_module": "src/configs/dataset_configs/custom_metadata/AR_md.py",
            "folder_name": "single_channel_ir_1"
        }

    ],
    "random_crop": false,
    "augs": true,
    "force_channels": "mono", 

    "modalities": {
        "acoustic_context": {
            "load": true,
            "max_context": 8, 
            "max_len": 9600 
        }, 
        "depth": {
            "load": true
        }, 
        "poses": {
            "load": true
        }
    }  
}{
    "model_type": "diffusion_cond",
    "sample_size": 10240,
    "sample_rate": 22050,
    "audio_channels": 1,
    "model": {
        "pretransform": {
            "type": "autoencoder",
            "iterate_batch": true,
            "config": {
                "encoder": {
                    "type": "oobleck",
                    "config": {
                        "in_channels": 1,
                        "channels": 64,
                        "c_mults": [
                            1,
                            2,
                            4,
                            8
                        ],
                        "strides": [
                            2,
                            4,
                            8,
                            16
                        ],
                        "latent_dim": 64,
                        "use_snake": true
                    }
                },
                "decoder": {
                    "type": "oobleck",
                    "config": {
                        "out_channels": 1,
                        "channels": 64,
                        "c_mults": [
                            1,
                            2,
                            4,
                            8
                        ],
                        "strides": [
                            2,
                            4,
                            8,
                            16
                        ],
                        "latent_dim": 32,
                        "use_snake": true,
                        "final_tanh": true
                    }
                },
                "bottleneck": {
                    "type": "vae"
                },
                "latent_dim": 32,
                "downsampling_ratio": 1024,
                "io_channels": 1
            }
        },
        "conditioning": {
            "configs": [
                {
                    "id": "source",
                    "type": "dist_embedder",
                    "config": {
                        "num_freqs": 20,
                        "max_freq": 10,
                        "ch_dim": 1,
                        "include_in": true
                    }
                },
                {
                    "id": "source_vit",
                    "type": "ViTCoordinates",
                    "config": {
                        "ViT": {
                            "hf_model_name_or_path": "facebook/dinov3-vits16-pretrain-lvd1689m",
                            "ch_dim": 3,
                            "freeze": false,
                            "from_scratch": false,
                            "img_h": 256,
                            "img_w": 512
                        },
                        "max_value": 1,
                        "gradient_checkpointing": true
                    }
                },
                {
                    "id": "context_poses_vit",
                    "type": "ViTCoordinates",
                    "config": {
                        "ViT": {
                            "hf_model_name_or_path": "facebook/dinov3-vits16-pretrain-lvd1689m",
                            "ch_dim": 3,
                            "freeze": false,
                            "from_scratch": false,
                            "img_h": 256,
                            "img_w": 512
                        },
                        "max_value": 1,
                        "gradient_checkpointing": true
                    }
                },
                {
                    "id": "context_poses",
                    "type": "dist_embedder",
                    "config": {
                        "num_freqs": 20,
                        "max_freq": 10,
                        "ch_dim": 1,
                        "include_in": true
                    }
                },
                {
                    "id": "context_audio",
                    "type": "rir",
                    "config": {
                        "in_channels": 1,
                        "n_fft": 124,
                        "win_length": 31,
                        "hop_length": 62,
                        "project_out": true
                    }
                }
            ],
            "cond_dim": 256
        },
        "diffusion": {
            "cross_attention_cond_ids": [
                "context_poses_vit",
                "context_poses",
                "context_audio"
            ],
            "global_cond_ids": [
                "source",
                "source_vit"
            ],
            "type": "dit",
            "diffusion_objective": "rectified_flow",
            "config": {
                "io_channels": 32,
                "embed_dim": 256,
                "depth": 12,
                "num_heads": 8,
                "cond_token_dim": 256,
                "global_cond_dim": 512,
                "project_cond_tokens": false,
                "transformer_type": "continuous_transformer",
                "global_cond_type": "adaLN"
            }
        },
        "io_channels": 32
    },
    "training": {
        "timestep_sampler": "log_snr",
        "mask_padding": true,
        "mask_padding_dropout": 0.0,
        "cfg_dropout_prob": 0.1,
        "use_ema": true,
        "log_loss_info": false,
        "optimizer_configs": {
            "diffusion": {
                "optimizer": {
                    "type": "AdamW",
                    "config": {
                        "lr": 5e-05,
                        "betas": [
                            0.9,
                            0.999
                        ],
                        "weight_decay": 0.001
                    }
                },
                "scheduler": {
                    "type": "InverseLR",
                    "config": {
                        "inv_gamma": 1000000,
                        "power": 0.5,
                        "warmup": 0.99
                    }
                }
            }
        },
        "metrics": {
            "dataset_name": "AcousticRooms",
            "eval_T60": true,
            "eval_C50": true,
            "eval_EDT": true,
            "eval_FD": true,
            "eval_retrieval": true,
            "AGREE_ckpt": "weights/AGREE/AGREE_fullAR.pt"
        },
        "cond_method": "fa_invariant",
        "frame_avg_angles": [
            0.0,
            90.0,
            180.0,
            270.0
        ]
    }
}
2026-08-11T19:02:58.435919Z ERROR codex_models_manager::manager: failed to renew cache TTL: missing field `base_instructions` at line 94 column 5
codex
The runtime trace has found a substantive closure omission: the selected dataset config (covered under `src/`) points to the tracked `data/AR/train.json`, which the data loader opens during training, but `data/AR/train.json` is outside the proposed commit comparison. I’m finishing the edge-case and test-control-flow audit now.
exec
/usr/bin/bash -lc "sed -n '1,300p' src/data/dataset.py; sed -n '1,300p' src/configs/dataset_configs/custom_metadata/AR_md.py" in /n/fs/gatrdp/codespace/FLAC
 succeeded in 0ms:
import importlib
import numpy as np
import json
import os
import random
import time
import torch
import torchaudio

from os import path
from torch import nn
from torchaudio import transforms as T
from typing import Optional, Callable, List

from .utils import Stereo, PseudoStereo, Mono, PadCrop_Normalized_T, AddNoise, RandomTimeShift

AUDIO_KEYS = ("flac", "wav", "mp3", "m4a", "ogg", "opus")


def json_scandir( 
    dir: str,  # top-level directory at which to begin scanning
    json_file_path: str,  # json file to read
    folder_name: str = "binaural_rirs",  # folder name to search for
    scenes: list=None,  # list of scenes to search for
):
    "Retrieve files when they are specified in a json file"
    subfolders, files = [], []
    if scenes is None:
        with open(json_file_path, 'r') as f:
            split_dict = json.load(f)
        for scene in split_dict.keys():
            if isinstance(split_dict[scene], dict):
                assert 'AcousticRooms' in dir, "AcousticRooms should be in the directory name"
                for sub_scene in split_dict[scene].keys():
                    subfolders.append(os.path.join(dir, folder_name, scene, sub_scene))
                    files.extend([os.path.join(dir, folder_name, scene, sub_scene, split_dict[scene][sub_scene][i]) for i in range(len(split_dict[scene][sub_scene]))])
            else:
                subfolders.append(os.path.join(dir, scene))
                files.extend([os.path.join(dir, scene, folder_name, split_dict[scene][i]) for i in range(len(split_dict[scene]))])
    else:
        raise NotImplementedError("Scene filtering not implemented")
    print(f"Found {len(files)} files in {len(subfolders)} subfolders")
    return subfolders, files

# fast_scandir implementation by Scott Hawley originally in https://github.com/zqevans/audio-diffusion/blob/main/dataset/dataset.py
def fast_scandir(
    dir:str,  # top-level directory at which to begin scanning
    ext:list,  # list of allowed file extensions,
    #max_size = 1 * 1000 * 1000 * 1000 # Only files < 1 GB
    ):
    "very fast `glob` alternative. from https://stackoverflow.com/a/59803793/4259243"
    subfolders, files = [], []
    ext = ['.'+x if x[0]!='.' else x for x in ext]  # add starting period to extensions if needed
    try: # hope to avoid 'permission denied' by this try
        for f in os.scandir(dir):
            try: # 'hope to avoid too many levels of symbolic links' error
                if f.is_dir():
                    subfolders.append(f.path)
                elif f.is_file():
                    file_ext = os.path.splitext(f.name)[1].lower()
                    is_hidden = os.path.basename(f.path).startswith(".")

                    if file_ext in ext and not is_hidden:
                        files.append(f.path)
            except:
                pass 
    except:
        pass

    for dir in list(subfolders):
        sf, f = fast_scandir(dir, ext)
        subfolders.extend(sf)
        files.extend(f)
    return subfolders, files

def keyword_scandir(
    dir: str,  # top-level directory at which to begin scanning
    ext: list,  # list of allowed file extensions
    keywords: list,  # list of keywords to search for in the file name
):
    "very fast `glob` alternative. from https://stackoverflow.com/a/59803793/4259243"
    subfolders, files = [], []
    # make keywords case insensitive
    keywords = [keyword.lower() for keyword in keywords]
    # add starting period to extensions if needed
    ext = ['.'+x if x[0] != '.' else x for x in ext]
    banned_words = ["paxheader", "__macosx"]
    try:  # hope to avoid 'permission denied' by this try
        for f in os.scandir(dir):
            try:  # 'hope to avoid too many levels of symbolic links' error
                if f.is_dir():
                    subfolders.append(f.path)
                elif f.is_file():
                    is_hidden = f.name.split("/")[-1][0] == '.'
                    has_ext = os.path.splitext(f.name)[1].lower() in ext
                    name_lower = f.name.lower()
                    has_keyword = any(
                        [keyword in name_lower for keyword in keywords])
                    has_banned = any(
                        [banned_word in name_lower for banned_word in banned_words])
                    if has_ext and has_keyword and not has_banned and not is_hidden and not os.path.basename(f.path).startswith("._"):
                        files.append(f.path)
            except:
                pass
    except:
        pass

    for dir in list(subfolders):
        sf, f = keyword_scandir(dir, ext, keywords)
        subfolders.extend(sf)
        files.extend(f)
    return subfolders, files

def get_audio_filenames(
    paths: list,  # directories in which to search
    keywords=None,
    json_file_path=None,
    folder_name=None,
    exts=['.wav', '.mp3', '.flac', '.ogg', '.aif', '.opus']
):
    "recursively get a list of audio filenames"
    # check extension of json_file_path if not none
    if json_file_path is not None:
        json_ext = os.path.splitext(json_file_path)[1].lower()
    filenames = []
    if type(paths) is str:
        paths = [paths]
    for path in paths:               # get a list of relevant filenames
        if json_file_path is not None and folder_name is not None and json_ext == '.json':
            print('Running json scandir...')
            subfolders, files = json_scandir(dir=path, json_file_path=json_file_path, folder_name=folder_name)
        else:
            print('Running fast scandir...')
            if keywords is not None:
                subfolders, files = keyword_scandir(path, exts, keywords)
            else:
                subfolders, files = fast_scandir(path, exts)
        filenames.extend(files)
    return filenames

class LocalDatasetConfig:
    def __init__(
        self,
        id: str,
        path: str,
        custom_metadata_fn: Optional[Callable[[str], str]] = None,
        json_file_path: Optional[str] = None,
        folder_name: Optional[str] = None,  
        scenes: Optional[List[str]] = None, 
        is_eval: Optional[bool] = False, 
        unseeneval: Optional[bool] = False, 
        seeneval: Optional[bool] = False, 
        conditioning: Optional[dict] = None,  
    ):
        self.id = id
        self.path = path
        self.custom_metadata_fn = custom_metadata_fn
        self.json_file_path = json_file_path
        self.folder_name = folder_name  

        self.scenes = scenes
        self.is_eval = is_eval
        self.unseeneval = unseeneval
        self.seeneval = seeneval

        # Conditioning modalities
        self.modalities = conditioning


class SampleDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        configs,
        sample_size=10240, 
        sample_rate=22050, 
        keywords=None, 
        random_crop=True,
        force_channels="mono",
        augs=True,
    ):
        super().__init__()
        self.filenames = []

        self.eval = configs[0].is_eval if hasattr(configs[0], 'is_eval') else False
        self.unseeneval = configs[0].unseeneval if hasattr(configs[0], 'unseeneval') else False
        self.seeneval = configs[0].seeneval if hasattr(configs[0], 'seeneval') else False
        if (self.unseeneval or self.seeneval) and self.eval==False:
            self.eval = True 
        self.json_file_path = configs[0].json_file_path if hasattr(configs[0], 'json_file_path') else None

        if augs:
            print('Using Augmentations: Random Time Shift, Add Noise')
            self.augs = torch.nn.Sequential(
                RandomTimeShift(max_shift=10, p=0.5),
                AddNoise(snr_db_range=(40, 60), noise_type='pink', p=0.5), 
            )
        else:
            self.augs = None

        self.root_paths = []

        self.pad_crop = PadCrop_Normalized_T(sample_size, sample_rate, randomize=random_crop)

        self.force_channels = force_channels

        self.encoding = torch.nn.Sequential(
            Stereo() if self.force_channels == "stereo" else torch.nn.Identity(),
            PseudoStereo(sample_rate=sample_rate) if self.force_channels == "pseudostereo" else torch.nn.Identity(),
            Mono() if self.force_channels == "mono" else torch.nn.Identity(),
        )

        self.sr = sample_rate

        self.custom_metadata_fns = {}
        self.modalities_fns = {}

        for config in configs:
            self.root_paths.append(config.path)
            self.filenames.extend(get_audio_filenames(paths=config.path, keywords=keywords, json_file_path=config.json_file_path, folder_name=config.folder_name))
            if config.custom_metadata_fn is not None:
                self.custom_metadata_fns[config.path] = config.custom_metadata_fn
            
            self.modalities_fns[config.path] = config.modalities

    def load_file(self, filename):
        ext = filename.split(".")[-1]

        audio, in_sr = torchaudio.load(filename, format=ext)

        if in_sr != self.sr:
            resample_tf = T.Resample(in_sr, self.sr, lowpass_filter_width=128)
            audio = resample_tf(audio)

        return audio

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        audio_filename = self.filenames[idx]
        try:
            start_time = time.time()
            audio = self.load_file(audio_filename)

            audio, t_start, t_end, seconds_start, seconds_total, padding_mask = self.pad_crop(audio)

            # Check for silence
            if is_silence(audio):
                return self[random.randrange(len(self))]

            # Run augmentations on this sample
            if self.augs is not None:
                audio = self.augs(audio)

            audio = audio.clamp(-1, 1)

            # Encode the file to assist in prediction
            if self.encoding is not None:
                audio = self.encoding(audio)

            info = {}

            info['eval'] = self.eval

            info['unseeneval'] = self.unseeneval
            info['seeneval'] = self.seeneval
            info['json_file_path'] = self.json_file_path 

            info['idx'] = idx
            info["path"] = audio_filename
            info['sample_rate'] = self.sr
            info['sample_size'] = audio.shape[-1]

            for root_path in self.root_paths:
                if root_path in audio_filename:
                    info["relpath"] = path.relpath(audio_filename, root_path)

            info["padding_mask"] = padding_mask

            end_time = time.time()

            info["load_time"] = end_time - start_time

            for custom_md_path in self.custom_metadata_fns.keys():
                if custom_md_path in audio_filename:
                    info['modalities'] = self.modalities_fns[custom_md_path]
                    custom_metadata_fn = self.custom_metadata_fns[custom_md_path]
                    custom_metadata = custom_metadata_fn(info, audio)
                    info.update(custom_metadata)

                if "__reject__" in info and info["__reject__"]:
                    return self[random.randrange(len(self))]

                # Provide audio inputs as their own dictionary to be merged into info, each audio element will be normalized in the same way as the main audio
                if "__audio__" in info:
                    for audio_key, audio_value in info["__audio__"].items():
                        # Process the audio_value tensor, which should be a torch tensor
                        audio_value, _, _, _, _, _ = self.pad_crop(audio_value)
                        audio_value = audio_value.clamp(-1, 1)
                        if self.encoding is not None:
import os
import numpy as np
import json
import torch 
import torchaudio


def get_custom_metadata(info, audio): 
    md = {}
    full_audio_path = info["path"]
    rel_path = info["relpath"]
    common_suffix = os.path.commonpath([full_audio_path[::-1], rel_path[::-1]])[::-1]
    dataset_folder = full_audio_path[: -len(common_suffix)]
    metadata_path = os.path.join(dataset_folder, 'metadata')

    # Get Config Info
    modalities = info['modalities'] 
    acoustic_context_config = modalities.get('acoustic_context', None)
    depth_config = modalities.get('depth', None)
    pose_config = modalities.get('poses', None)

    # Get Instance Information
    scene_name = rel_path.split("/")[-3]
    scene_id = rel_path.split("/")[-2]
    filename = rel_path.split("/")[-1].split(".")[0]
    receiver_idx, source_idx = int(filename.split("_")[1][1:]), int(filename.split("_")[0][1:])
    md['scene'] = scene_name

    # Load Positions
    if pose_config.get('load', False):
        source_pos, listener_pos = get_receiver_source_location(rel_path, metadata_path)
        proj_source_pos = get_3d_point_camera_coord(listener_pos, source_pos)
        proj_source_pos = torch.Tensor(proj_source_pos).float()
        proj_listener_pos = torch.Tensor([0., 0., 0.])
        source_listener_pos = torch.cat([proj_source_pos.unsqueeze(0), proj_listener_pos.unsqueeze(0)], dim=0) # [2, 3]
        md['source'] = proj_source_pos 
        md['source_vit'] = proj_source_pos.unsqueeze(0) # [1, 3]

    # Load Acoustic Context
    if acoustic_context_config.get('load', False):
        max_len_cond = acoustic_context_config.get('max_len', 9600)
        all_ref_irs, all_ref_src_pos = get_ir_and_location_for_other_sources(full_audio_path, num_ref_sources=acoustic_context_config.get('max_context', 8), metadata_path=metadata_path, max_len=max_len_cond)
        md['context_poses'] = all_ref_src_pos # [N, 3]  
        md['context_poses_vit'] = all_ref_src_pos
        md['context_audio'] = all_ref_irs # [N, max_len_cond]

    # Load Depth
    if depth_config.get('load', False):
        pano_depth_path = dataset_folder + 'depth_map'
        pano_depth = np.load(os.path.join(pano_depth_path, scene_name, scene_id, f"{receiver_idx}.npy"))
        depth_coord = convert_equirect_to_camera_coord(torch.from_numpy(pano_depth), 256, 512) # [H, W, 3]
        md['depth'] = depth_coord.permute(2, 0, 1) # [3, H, W]
    
    return md


############# UTILS #############
def convert_equirect_to_camera_coord(depth_map, img_h, img_w): # 3D point cloud per pixel
    phi, theta = torch.meshgrid(torch.arange(img_h), torch.arange(img_w), indexing='ij')
    theta_map = (theta + 0.5) * 2.0 * np.pi / img_w - np.pi
    phi_map = (phi + 0.5) * np.pi / img_h - np.pi / 2
    sin_theta = torch.sin(theta_map)
    cos_theta = torch.cos(theta_map)
    sin_phi = torch.sin(phi_map)
    cos_phi = torch.cos(phi_map)
    return torch.stack([depth_map * cos_phi * cos_theta, depth_map * cos_phi * sin_theta, -depth_map * sin_phi], dim=-1)

def get_3d_point_camera_coord(source_pose, point_3d):
    camera_matrix = None
    lis_x, lis_y, lis_z = source_pose[0], source_pose[1], source_pose[2]
    camera_matrix = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]])
    camera_matrix[:3, 3] = np.array([-lis_x, -lis_y, -lis_z])
    point_4d = np.append(point_3d, 1.0)
    camera_coord_point = camera_matrix @ point_4d
    return camera_coord_point[:3]

def get_receiver_source_location(ir_file_path, metadata_path):
    scene_name = ir_file_path.split("/")[-3]
    scene_id = ir_file_path.split("/")[-2]
    ir_file_name = ir_file_path.split("/")[-1]
    src_node, rec_node = int(ir_file_name.split("_")[0][1:]), int(ir_file_name.split("_")[1][1:])
    json_file_name = "S00" + str(src_node) + "_R00" + str(rec_node) + ".json"
    metadata_file_path = os.path.join(metadata_path, scene_name, scene_id, json_file_name)
    with open(metadata_file_path, "r") as fin:
        meta_info = json.load(fin)
    src_loc = meta_info["src_loc"]
    rec_loc = meta_info["rec_loc"]
    return src_loc, rec_loc

def get_ir_and_location_for_other_sources(ir_file_path, num_ref_sources, metadata_path, max_len=9600):
    dir_name = os.path.dirname(ir_file_path)
    ir_file_name = ir_file_path.split("/")[-1]
    src_node, rec_node = int(ir_file_name.split("_")[0][1:]), int(ir_file_name.split("_")[1][1:])
    all_src_node = set([int(fn.split("_")[0][1:]) for fn in os.listdir(dir_name)])
    remain_src_node = list(all_src_node.difference(set([src_node])))
    valid_other_src_ir_paths = []
    for node in remain_src_node:
        rec_n = ir_file_name.split("_")[1]
        src_n = f"S00{node}"
        other_src_ir_path = os.path.join(dir_name, f"{src_n}_{rec_n}_hybrid_IR.wav")
        if os.path.exists(other_src_ir_path):
            valid_other_src_ir_paths.append(other_src_ir_path)
    try:
        select_other_src_ir_paths = np.random.choice(valid_other_src_ir_paths, num_ref_sources, replace=False)
    except Exception as e:
        select_other_src_ir_paths = np.random.choice(valid_other_src_ir_paths, num_ref_sources, replace=True)
    all_ref_irs = []
    all_ref_src_pos = []
    
    for fp in select_other_src_ir_paths:
        ref_wav, rate = torchaudio.load(fp)
        assert rate == 22050, "IR sampling rate must be 22050!"
        if ref_wav.shape[1] < max_len:
            ref_wav = torch.cat([ref_wav, torch.zeros(ref_wav.shape[0], max_len - ref_wav.shape[1])], dim=1)
        else:
            ref_wav = ref_wav[:, :max_len]
        ref_wav = ref_wav.unsqueeze(0) # C=1
        all_ref_irs.append(ref_wav)

        src_loc, rec_loc = get_receiver_source_location(fp, metadata_path=metadata_path)
        
        proj_src_loc = get_3d_point_camera_coord(rec_loc, src_loc)
        
        all_ref_src_pos.append(torch.Tensor(proj_src_loc).float())
    all_ref_irs = torch.cat(all_ref_irs, dim=0)
    all_ref_src_pos = torch.vstack(all_ref_src_pos)
    return all_ref_irs, all_ref_src_pos
exec
/usr/bin/bash -lc "grep -RInE \"AGREE_ckpt|metrics\\]|metrics\\)|create.*metric|Metric|AGREE\" src/train.py src train.py --include='*.py' 2>/dev/null | sed -n '1,320p'" in /n/fs/gatrdp/codespace/FLAC
 succeeded in 0ms:
src/tests/test_exp11_validate_rows.py:325:    path.write_text(json.dumps(_record(metrics=metrics)))
src/tests/test_exp11_validate_rows.py:521:    extra = dict(REQUIRED_METRICS); extra["NewMetric"] = 3.3
src/tests/test_exp11_validate_rows.py:532:    _row, problems = V.validate_row(_write_row(tmp_path, rec=_record(metrics=metrics)))
src/tests/test_eval_paths.py:66:    Metrics stay byte-identical to the committed exp_02 artifact (the doubled
src/tests/test_eval_paths.py:364:        eval_FLAC, "create_metric_callback_from_config",
src/tests/test_yaw_random_eval.py:854:        eval_FLAC, "create_metric_callback_from_config",
src/tests/test_yaw_random_eval.py:1134:    assert eval_FLAC.stream_sidecar_path(metrics) == (
src/tests/test_cond_dispatch.py:113:    # AcousticMetricsCallback) + a no-op sampler and metric sink.
src/training/autoencoders.py:15:from .utils import create_optimizer_from_config, create_scheduler_from_config, log_audio, log_image, log_metric, log_point_cloud, logger_project_name
src/training/autoencoders.py:269:                from ..metrics.metric_callback import AcousticMetricsCallback
src/training/autoencoders.py:270:                self.metric_callback = AcousticMetricsCallback(
src/training/factory.py:160:def create_metric_callback_from_config(model_config, dataset_id=None, per_scene=False):
src/training/factory.py:174:    from ..metrics.metric_callback import AcousticMetricsCallback
src/training/factory.py:175:    return AcousticMetricsCallback(
src/training/factory.py:193:        AGREE_ckpt=metrics_config.get("AGREE_ckpt", None),
src/training/diffusion.py:27:from .utils import create_optimizer_from_config, create_scheduler_from_config, log_metric
src/training/diffusion.py:314:        from ..metrics.metric_callback import AcousticMetricsCallback
src/training/diffusion.py:315:        self.metric_callback = AcousticMetricsCallback(
src/training/diffusion.py:330:            AGREE_ckpt=metrics.get("AGREE_ckpt", None), 
src/training/__init__.py:1:from .factory import create_training_wrapper_from_config, create_metric_callback_from_config
src/models/cyl_vit.py:79:    """Pre-norm MLP block (matches AGREE's SimpleViT feed-forward)."""
src/metrics/metric_callback.py:30:# Use the in-tree AGREE package (a fork of open_clip with audio support) instead of
src/metrics/metric_callback.py:31:# the public `open_clip_torch`, since AGREE model configs use `audio_cfg` rather than
src/metrics/metric_callback.py:33:from AGREE.AGREE.factory import get_model_config
src/metrics/metric_callback.py:34:from AGREE.AGREE.model import AGREE as CLIP
src/metrics/metric_callback.py:37:class AcousticMetricsCallback:
src/metrics/metric_callback.py:40:            name: str = "AcousticMetricsCallback",
src/metrics/metric_callback.py:59:            AGREE_ckpt: Any=None,
src/metrics/metric_callback.py:62:        self._validate_parameters(dataset_name, sample_rate, audio_channels, eval_retrieval, eval_FD, AGREE_ckpt)
src/metrics/metric_callback.py:89:        self._setup(AGREE_ckpt)
src/metrics/metric_callback.py:94:                           eval_retrieval: bool, eval_FD: bool, AGREE_ckpt: Any):
src/metrics/metric_callback.py:106:        if (eval_retrieval or eval_FD) and AGREE_ckpt is None:
src/metrics/metric_callback.py:107:            raise ValueError("AGREE_ckpt must be provided when eval_retrieval or eval_FD is True")
src/metrics/metric_callback.py:109:    def _setup(self, AGREE_ckpt=None):
src/metrics/metric_callback.py:120:        # Setup AGREE model if needed
src/metrics/metric_callback.py:121:        AGREE_model, encoder = None, None
src/metrics/metric_callback.py:122:        if AGREE_ckpt is not None and (self.eval_retrieval or self.eval_FD):
src/metrics/metric_callback.py:123:            AGREE_model, encoder = loading_AGREE_model(AGREE_ckpt, self.device)
src/metrics/metric_callback.py:124:            print('AGREE model loaded successfully')
src/metrics/metric_callback.py:125:        self.AGREE_model = AGREE_model
src/metrics/metric_callback.py:129:        self._initialize_metrics(AGREE_model, encoder)
src/metrics/metric_callback.py:131:    def _initialize_metrics(self, AGREE_model=None, encoder=None):
src/metrics/metric_callback.py:136:            self.RT60 = self._create_metric_dict(
src/metrics/metric_callback.py:142:            self.C50 = self._create_metric_dict(
src/metrics/metric_callback.py:148:            self.EDT = self._create_metric_dict(
src/metrics/metric_callback.py:154:            self.l1_stft = self._create_metric_dict(
src/metrics/metric_callback.py:160:            self.l1_stft_multires = self._create_metric_dict(
src/metrics/metric_callback.py:166:            self.FD = self._create_metric_dict(
src/metrics/metric_callback.py:172:            self.retrieval = self._create_metric_dict(
src/metrics/metric_callback.py:173:                lambda: Retrieval(AGREE=AGREE_model),
src/metrics/metric_callback.py:178:            self.Env = self._create_metric_dict(
src/metrics/metric_callback.py:183:    def _create_metric_dict(self, metric_factory, stages):
src/metrics/metric_callback.py:196:    def get_create_scene_metrics(self, stage: str, scene: str):
src/metrics/metric_callback.py:240:                    AGREE=self.AGREE_model
src/metrics/metric_callback.py:318:                scene_metrics = self.get_create_scene_metrics(stage, current_scene)
src/metrics/metric_callback.py:372:            metrics.update(retrieval_metrics)
src/metrics/metric_callback.py:420:                    scene_results[scene_name].update(retrieval_metrics)
src/metrics/metric_callback.py:432:def loading_AGREE_model(ckpt, device):
src/metrics/metric_callback.py:433:    print('Loading AGREE model from checkpoint: ', ckpt)
src/metrics/metric_callback.py:434:    AGREE_config = get_model_config('dinoV3')
src/metrics/metric_callback.py:435:    AGREE_model = CLIP(**AGREE_config)
src/metrics/metric_callback.py:437:    AGREE_ckpt = torch.load(ckpt, map_location=device)
src/metrics/metric_callback.py:439:    AGREE_state_dict = AGREE_ckpt['state_dict']
src/metrics/metric_callback.py:440:    AGREE_model.load_state_dict(AGREE_state_dict, strict=True)
src/metrics/metric_callback.py:442:    AGREE_audio_encoder = AGREE_model.audio
src/metrics/metric_callback.py:443:    print('Done loading AGREE model')
src/metrics/metric_callback.py:445:    return AGREE_model, AGREE_audio_encoder
src/metrics/modules/l1_stft.py:2:from torchmetrics import Metric
src/metrics/modules/l1_stft.py:4:class L1_STFT(Metric):
src/metrics/modules/Retrieval.py:3:from torchmetrics import Metric
src/metrics/modules/Retrieval.py:5:class Retrieval(Metric):
src/metrics/modules/Retrieval.py:7:                 AGREE=None,
src/metrics/modules/Retrieval.py:16:        self.AGREE = AGREE
src/metrics/modules/Retrieval.py:17:        self.logit_scale = self.AGREE.logit_scale.exp()
src/metrics/modules/Retrieval.py:48:            feats = self.AGREE.encode_audio(h, normalize=True)
src/metrics/modules/Retrieval.py:54:            feats = self.AGREE.encode_image(h, normalize=True)
src/metrics/modules/Retrieval.py:72:            # Sanity check: should be the same as AGREE eval 
src/metrics/modules/RT60.py:3:from torchmetrics import Metric
src/metrics/modules/RT60.py:5:class RT60Error(Metric):
src/metrics/modules/FD.py:3:from torchmetrics import Metric
src/metrics/modules/FD.py:6:class FD(Metric):
src/metrics/modules/C50.py:3:from torchmetrics import Metric
src/metrics/modules/C50.py:5:class C50(Metric):
src/metrics/modules/EDT.py:2:from torchmetrics import Metric
src/metrics/modules/EDT.py:5:class EDT(Metric):
src/metrics/modules/Env.py:4:from torchmetrics import Metric
src/metrics/modules/Env.py:9:class Env(Metric):
src/metrics/modules/Env.py:44:    diffRIR implementation of Envelope Evaluation Metric.
src/metrics/modules/l1_stft_multires.py:4:from torchmetrics import Metric
src/metrics/modules/l1_stft_multires.py:7:class L1_STFT_MultiRes(Metric):
src/metrics/modules/l1_stft_multires.py:74:    """Spectral Evaluation Metric"""

exec
/usr/bin/bash -lc "git ls-files -s -- data/AR/train.json data/AR 'weights/AGREE/AGREE_fullAR.pt'; stat -c '%F %a %n' data/AR/train.json AcousticRooms weights/AGREE/AGREE_fullAR.pt 2>&1; sed -n '1,120p' data/AR/train.json" in /n/fs/gatrdp/codespace/FLAC
exec
/usr/bin/bash -lc "git log --format='%H %s' -40 | while read s rest; do d="'$(git diff --name-only "$s" HEAD -- train.py defaults.ini src worklog/worklog_yixun/exp_11_fa_orbit_claude/FLAC_AR_*.json worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_launch.sh worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py worklog/worklog_yixun/exp_11_fa_orbit_claude/assert_arm_configs_exp11.py worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_wandb_readback.py worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_classify.py worklog/worklog_yixun/exp_07_fa_scratch_claude/FLAC_AR_BF.json); if [ -z "$d" ]; then k=same; else k=changed; fi; printf '"'%s %s %s\\n' \""'$k" "$s" "$rest"; done | sed -n '"'1,40p'" in /n/fs/gatrdp/codespace/FLAC
 succeeded in 878ms:
same 135cb4beb3569bb41f325459f0785bf970297de8 exp_14 r2-fix2: X1 + X2 + X3 — the two refutations, and the harm one of them did
same 548d5db36c43acd307c39e003d9c50d18c550867 exp_15 r3: guardtest + DRYRUN transcripts — 98 passed, 0 failed, nothing submitted
same 9fa7067d04007a2a351860249fe5069bc9d832c3 exp_15 r3: guardtest fix — bash expands every word of `local` before assigning
same b71ca45215fb42b16df62aaa9e7a49db91da5f80 exp_15 r3: guardtest suite for the exp_15 gates — local only, nothing submitted
same 90f2c3c74e59ef221e220435a40ab43bee1a8b38 exp_15 r3: the enumerated kit deltas — identity, 40k re-pin, gates (plan §6.6)
same a4bbe86876ea05684d223ec9483ae8f344f1b9ae exp_15 r3: VERBATIM copies of exp_11's training kit (zero edits)
same 6e9d344922672d03da1e50c5155a0131a46bfb27 exp_15 r2: Codex review (REVISE→fixed, record regenerated) + worklog entries — round 2 closed
same 3bbd2bc0b0e0f3ddc17e91fc76b40a72a1d249f0 exp_15 r2-fix: final verification log — 485 passed across the affected suites
same f51578c2a30d6b9e1c98396c4124252c24af8a54 exp_15 r2-fix: pin the REAL admission record against drift (finding 6, MINOR)
changed ae7a924dff5116de5e6d1f0663a70d61d39af33d exp_15 r2-fix: regenerate the admission record under the corrected recorder
changed 1e7ef90dadb9cc9553ee1a86d07556a79fd78966 exp_15 r2-fix: exclusive record creation (F5) + a truly byte-level arm diff (F4)
changed a3719f921c0a9d6d3ed6f115208657f90fcc93c9 exp_15 r2-fix: type-strict config and step comparison (finding 3, MAJOR)
changed 7e7c512c2c9eaba9ad31bf35f160141dd5671cd6 exp_15 r2-fix: the EMA family must MIRROR the online DiT (finding 2, MAJOR)
changed 30d5065774bf1bbc18d5d764cc3ab8eb3cd8f0b8 exp_15 r2-fix: one stable, safely loaded snapshot (review finding 1, BLOCKING)
changed 54b5a0c3ebfdea2bec40a8164a1af5bb3174fb71 exp_14 docs: worklog through r2 fix batch, r2 review artifact, ledger
changed b9de6c47e740ba4c3ca30804870f82bc9c9ef1eb exp_15 r2: final verification log — 456 passed across the affected suites
changed 0f056b40e0ed037326019473e24401228486979b exp_14 r2-fix: FB8c — the live-wave guard cases the review asked for (180 cases)
changed 970110f01fa6cf45e0f3202b7aca9f8389de27b9 exp_15 r2: the VANL control-admission record — all four checks passed
changed 389d0d8d76a65cadd1d5c1ebdf32597b4205820d exp_15 r2: bind the checkpoint to the config EMBEDDED inside it (plan §3.3-1)
changed d2c11db3f09f79ba5b26a94755721ac4ed94d325 exp_15 r2: yaw_aug_record_control.py — the control-admission recorder (TDD)
changed 075b7871f605b61dab4d8d522899c44b799a08a3 exp_15 r2: FLAC_AR_YAWAUG.json — the control's config plus exactly one block
changed a201e3c39bea2ae33c03f43530a2604b5fdaec5d exp_14 r2-fix: FB3 + FB5 + FB7 + FB8a/b — the submitters' safety rails
changed 58d0d631b45d4510fd1e450389299c005724cb32 exp_15 r1: Codex review (REVISE→fixed) + worklog round entries — round 1 closed
changed 2e9b8878002c94bdd4c6d3e255ba350c20bd9ddb exp_15 r1-fix: final verification log — 437 passed across the affected suites
changed 1ab22a094b16e0cfe38b24f42531128711dddd2b exp_14 r2-fix: FB2 — the Slurm job name must identify the CELL, not the arm
changed aa3a66fb0e0290cb6b7a69f5b0b251ff40f6b012 exp_15 r1-fix: make the golden-writer guard test unable to corrupt the fixture
changed 06b66b0a7c5f330f78d3611af1e4375ceaca55c3 exp_14 r2-fix: FB4 — a dedup SKIP now rests on checkpoint identity
changed a10e40971ed74071ed37f29e0572d3f489226643 exp_15 r1-fix: flush the banner, stop coercing ctor args, demand a capture SHA
changed 8a332340498bc66250afb8a1a72605451f44c3b6 exp_15 r1-fix: require all four pose fields in the schema guard (finding 2, MAJOR)
changed 076e11acdf6173a074b13e556ee2d5e6b813afb9 exp_15 r1-fix: keyed 32-bit bijection for the yaw seed (review finding 1, MAJOR)
changed 98e276ce732cdd7c8ae0c23629e3bdc819f80053 exp_14 r2-fix: FB6 — six ways a VALID verdict meant "not checked"
changed 5d6e349ce9e276da092f029acdd78dfd77eb0d5a exp_14 r2-fix: FB1 — the --rotate-deg 0 that would have failed 100 of 106 cells
changed 28f66d571bbcaa5a8daba016e7caf23f492dd47d exp_15 r1: round-1 verification logs — 397 passed across the affected suites
changed f8983deb6e96ff609d2353463faec98668f59151 exp_15 r1: training_step random-yaw augmentation hook (guards + rank-0 banner)
changed 05e6c6d240f99090403f36c47cdc10cd51088587 exp_14 r2: guard suite — the exp_14 contract, end to end (162 cases)
changed 8c4ac4bd88e4927c73ad6a4c6579290212806305 exp_15 r1: factory parses training.yaw_aug, fail-closed, no-op when absent
changed 72bab565f3a562ec28fb0b1d72aa38317b3c4d6d exp_15 r1: counter-based yaw_aug step seed (_yaw_aug_step_seed, SplitMix64)
changed 7062c00937fedc4ce2a7bac7967594e69106e48c exp_15 r1: golden disabled-path fixture — one whole training_step captured PRE-CHANGE
changed d3a03129f4b070881ae1a8c8d37e6fb8321cb8ba exp_11: VANL screen curve (2.5k-40k, K1+K8) in trajectory figures; 100k restart legs submitted (3678599-602); screen raws+manifests committed
changed 1960792f58c90364f6685d8fff40ba761cf2fb55 exp_14 docs: plan Rev 2 (approved), query, worklog through round-1 closure, all review artifacts, commit ledger

 succeeded in 4183ms:
_hybrid_IR.wav",
            "S0045_R0060_hybrid_IR.wav",
            "S0046_R0033_hybrid_IR.wav",
            "S0069_R0082_hybrid_IR.wav",
            "S0040_R0028_hybrid_IR.wav",
            "S0078_R0063_hybrid_IR.wav",
            "S0061_R0057_hybrid_IR.wav",
            "S0078_R0046_hybrid_IR.wav",
            "S0056_R0049_hybrid_IR.wav",
            "S0079_R0078_hybrid_IR.wav",
            "S0078_R0027_hybrid_IR.wav",
            "S009_R0050_hybrid_IR.wav",
            "S0085_R0036_hybrid_IR.wav",
            "S0056_R0055_hybrid_IR.wav",
            "S0022_R003_hybrid_IR.wav",
            "S0022_R005_hybrid_IR.wav",
            "S0055_R0024_hybrid_IR.wav",
            "S0030_R0079_hybrid_IR.wav",

exec
/usr/bin/bash -lc "nl -ba src/training/diffusion.py | sed -n '250,350p'; nl -ba src/metrics/metric_callback.py | sed -n '1,150p'; nl -ba src/metrics/metric_callback.py | sed -n '425,450p'" in /n/fs/gatrdp/codespace/FLAC
 succeeded in 0ms:
   250	
   251	        self.p_one_shot = p_one_shot
   252	
   253	        self.diffusion_objective = model.diffusion_objective
   254	
   255	        self.loss_modules = [
   256	            MSELoss("output",
   257	                   "targets",
   258	                   weight=1.0,
   259	                   mask_key="padding_mask" if self.mask_padding else None,
   260	                   name="mse_loss"
   261	            )
   262	        ]
   263	
   264	        self.losses = MultiLoss(self.loss_modules)
   265	
   266	        self.log_loss_info = log_loss_info
   267	
   268	        assert lr is not None or optimizer_configs is not None, "Must specify either lr or optimizer_configs in training config"
   269	
   270	        if optimizer_configs is None:
   271	            optimizer_configs = {
   272	                "diffusion": {
   273	                    "optimizer": {
   274	                        "type": "Adam",
   275	                        "config": {
   276	                            "lr": lr
   277	                        }
   278	                    }
   279	                }
   280	            }
   281	        else:
   282	            if lr is not None:
   283	                print(f"WARNING: learning_rate and optimizer_configs both specified in config. Ignoring learning_rate and using optimizer_configs.")
   284	
   285	        self.optimizer_configs = optimizer_configs
   286	
   287	        self.pre_encoded = pre_encoded
   288	
   289	        # Validation
   290	        self.validation_timesteps = validation_timesteps
   291	        self.validation_step_outputs = {}
   292	        for validation_timestep in self.validation_timesteps:
   293	            self.validation_step_outputs[f'val/loss_{validation_timestep:.1f}'] = []
   294	        
   295	        # Test
   296	        if test_param is not None:
   297	            self.set_test_config(
   298	                samples=test_param.get("samples", 10240),
   299	                cfg_scale=test_param.get("cfg_scale", 1.0),
   300	                steps=int(test_param.get("steps", 1)),
   301	                sample_rate=test_param.get("sample_rate", 22050),
   302	                audio_channels=test_param.get("audio_channels", 1),
   303	                metrics=test_param.get("metrics", {}), 
   304	                store_predictions = test_param.get("store_predictions", False),
   305	            )
   306	
   307	    def set_test_config(self, samples, cfg_scale, steps, sample_rate, audio_channels, metrics, store_predictions=False):
   308	        self.samples = samples
   309	        self.cfg_scale = cfg_scale
   310	        self.steps = steps
   311	        self.store_predictions = store_predictions
   312	        self.preds = []
   313	
   314	        from ..metrics.metric_callback import AcousticMetricsCallback
   315	        self.metric_callback = AcousticMetricsCallback(
   316	            sample_rate=sample_rate,
   317	            sample_size=self.samples,
   318	            audio_channels=audio_channels,
   319	            dataset_name= metrics.get("dataset_name", "AcousticRooms"),
   320	
   321	            eval_T60=metrics.get("eval_T60", False),
   322	            eval_C50=metrics.get("eval_C50", False),
   323	            eval_EDT=metrics.get("eval_EDT", False),
   324	            eval_l1_distance=metrics.get("eval_l1_distance", False),
   325	            eval_l1_distance_multires=metrics.get("eval_l1_distance_multires", False),
   326	            eval_FD=metrics.get("eval_FD", False),
   327	            eval_retrieval = metrics.get("eval_retrieval", False),
   328	            eval_env = metrics.get("eval_env", False),
   329	
   330	            AGREE_ckpt=metrics.get("AGREE_ckpt", None), 
   331	            dump_dir=metrics.get("dump_dir", None),
   332	            eval_per_scene=True if metrics.get("dataset_name") == "HAA" else False,
   333	        )
   334	
   335	    def configure_optimizers(self):
   336	        diffusion_opt_config = self.optimizer_configs['diffusion']
   337	        opt_diff = create_optimizer_from_config(diffusion_opt_config['optimizer'], self.diffusion.parameters())
   338	
   339	        if "scheduler" in diffusion_opt_config:
   340	            sched_diff = create_scheduler_from_config(diffusion_opt_config['scheduler'], opt_diff)
   341	            sched_diff_config = {
   342	                "scheduler": sched_diff,
   343	                "interval": "step"
   344	            }
   345	            return [opt_diff], [sched_diff_config]
   346	
   347	        return [opt_diff]
   348	
   349	    @rank_zero_only
   350	    def _print_yaw_aug_banner(self):
     1	import os
     2	import sys
     3	from pathlib import Path
     4	from typing import Any, Dict
     5	
     6	import torch
     7	import torch.nn as nn
     8	import scipy.io.wavfile as wav
     9	
    10	from .modules.C50 import C50
    11	from .modules.EDT import EDT
    12	from .modules.RT60 import RT60Error
    13	from .modules.l1_stft import L1_STFT
    14	from .modules.FD import FD
    15	from .modules.Retrieval import Retrieval
    16	from .modules.Env import Env
    17	from .modules.l1_stft_multires import L1_STFT_MultiRes
    18	
    19	# Constants
    20	SUPPORTED_DATASETS = {"AcousticRooms", "HAA"}
    21	DEFAULT_SAMPLE_RATE = 22050
    22	DEFAULT_AUDIO_CHANNELS = 1
    23	STAGES = ["train", "val", "test"]
    24	
    25	# STFT configuration constants
    26	STFT_FFT_SIZE = 124
    27	STFT_HOP_SIZE = 31
    28	STFT_WIN_LENGTH = 62
    29	
    30	# Use the in-tree AGREE package (a fork of open_clip with audio support) instead of
    31	# the public `open_clip_torch`, since AGREE model configs use `audio_cfg` rather than
    32	# `text_cfg`, which the public open_clip's config registry silently filters out.
    33	from AGREE.AGREE.factory import get_model_config
    34	from AGREE.AGREE.model import AGREE as CLIP
    35	
    36	
    37	class AcousticMetricsCallback:
    38	    def __init__(
    39	            self,
    40	            name: str = "AcousticMetricsCallback",
    41	            dataset_name: str = "AcousticRooms",
    42	            sample_rate: int=22050, 
    43	            sample_size: int=22050,
    44	            audio_channels: int=1, 
    45	            eval_per_scene: bool=False,
    46	            
    47	            device: str = "cuda",
    48	            dump_dir: Path=None,
    49	
    50	            eval_T60: bool=False,
    51	            eval_C50: bool=False,
    52	            eval_EDT: bool=False,
    53	            eval_l1_distance: bool=False,
    54	            eval_FD: bool=False,
    55	            eval_retrieval: bool=False,
    56	            eval_env: bool=False,
    57	            eval_l1_distance_multires: bool=False,
    58	
    59	            AGREE_ckpt: Any=None,
    60	        ):
    61	        # Validate parameters
    62	        self._validate_parameters(dataset_name, sample_rate, audio_channels, eval_retrieval, eval_FD, AGREE_ckpt)
    63	
    64	        self.sample_rate = sample_rate
    65	        self.audio_channels = audio_channels
    66	        self.sample_size = sample_size
    67	
    68	        self.dump_dir = dump_dir
    69	
    70	        self.eval_T60 = eval_T60
    71	        self.eval_C50 = eval_C50
    72	        self.eval_EDT = eval_EDT
    73	        self.eval_l1_distance = eval_l1_distance
    74	        self.eval_FD = eval_FD
    75	        self.eval_retrieval = eval_retrieval
    76	        self.eval_env = eval_env
    77	        self.eval_l1_distance_multires = eval_l1_distance_multires
    78	
    79	        self.dataset_name = dataset_name
    80	
    81	        if self.dataset_name == 'HAA' or eval_per_scene:
    82	            self.eval_by_scene = True
    83	            self.scene_metrics = {}
    84	            self.initialize_scene_metrics()
    85	        else:
    86	            self.eval_by_scene = False
    87	        
    88	        self.device = device
    89	        self._setup(AGREE_ckpt)
    90	
    91	        self._move_to_device(device)
    92	
    93	    def _validate_parameters(self, dataset_name: str, sample_rate: int, audio_channels: int, 
    94	                           eval_retrieval: bool, eval_FD: bool, AGREE_ckpt: Any):
    95	        """Validate parameters."""
    96	        supported_datasets = {"AcousticRooms", "HAA"}
    97	        if dataset_name not in supported_datasets:
    98	            raise ValueError(f"Dataset {dataset_name} not supported. Supported datasets: {supported_datasets}")
    99	        
   100	        if dataset_name in ["AcousticRooms", "HAA"]:
   101	            if sample_rate != 22050:
   102	                raise ValueError(f"{dataset_name} dataset requires a sample rate of 22050 Hz, got {sample_rate}")
   103	            if audio_channels != 1:
   104	                raise ValueError(f"{dataset_name} dataset requires 1 audio channel, got {audio_channels}")
   105	        
   106	        if (eval_retrieval or eval_FD) and AGREE_ckpt is None:
   107	            raise ValueError("AGREE_ckpt must be provided when eval_retrieval or eval_FD is True")
   108	
   109	    def _setup(self, AGREE_ckpt=None):
   110	        """Setup dataset-specific configuration and constants."""
   111	        if self.dataset_name in ["AcousticRooms", "HAA"]:
   112	            # max len from xRIR code
   113	            self.max_len_magenv = 9600
   114	            self.max_len = 9600 if self.dataset_name == "HAA" else 8000
   115	            self.stft = stft()
   116	            print(f'Max audio length for metric computation: {self.max_len}')
   117	        else:
   118	            raise NotImplementedError(f"Dataset {self.dataset_name} not supported yet")
   119	        
   120	        # Setup AGREE model if needed
   121	        AGREE_model, encoder = None, None
   122	        if AGREE_ckpt is not None and (self.eval_retrieval or self.eval_FD):
   123	            AGREE_model, encoder = loading_AGREE_model(AGREE_ckpt, self.device)
   124	            print('AGREE model loaded successfully')
   125	        self.AGREE_model = AGREE_model
   126	        self.encoder = encoder
   127	
   128	        # Initialize metrics using helper method
   129	        self._initialize_metrics(AGREE_model, encoder)
   130	
   131	    def _initialize_metrics(self, AGREE_model=None, encoder=None):
   132	        """Helper method to initialize metrics for all stages."""
   133	        stages = ["train", "val", "test"]
   134	        
   135	        if self.eval_T60:
   136	            self.RT60 = self._create_metric_dict(
   137	                lambda: RT60Error(fs=self.sample_rate, n_audio_ch=self.audio_channels, dataset_name=self.dataset_name),
   138	                stages
   139	            )
   140	
   141	        if self.eval_C50:
   142	            self.C50 = self._create_metric_dict(
   143	                lambda: C50(fs=self.sample_rate, n_audio_ch=self.audio_channels),
   144	                stages
   145	            )
   146	
   147	        if self.eval_EDT:
   148	            self.EDT = self._create_metric_dict(
   149	                lambda: EDT(fs=self.sample_rate, n_audio_ch=self.audio_channels),
   150	                stages
   425	                    scene_metrics["Env"].reset()
   426	
   427	            metrics["by_scene"] = scene_results
   428	
   429	        return metrics
   430	
   431	
   432	def loading_AGREE_model(ckpt, device):
   433	    print('Loading AGREE model from checkpoint: ', ckpt)
   434	    AGREE_config = get_model_config('dinoV3')
   435	    AGREE_model = CLIP(**AGREE_config)
   436	
   437	    AGREE_ckpt = torch.load(ckpt, map_location=device)
   438	
   439	    AGREE_state_dict = AGREE_ckpt['state_dict']
   440	    AGREE_model.load_state_dict(AGREE_state_dict, strict=True)
   441	
   442	    AGREE_audio_encoder = AGREE_model.audio
   443	    print('Done loading AGREE model')
   444	    
   445	    return AGREE_model, AGREE_audio_encoder
   446	
   447	
   448	class stft(nn.Module):
   449	    def __init__(self, fft_size=STFT_FFT_SIZE, hop_size=STFT_HOP_SIZE, win_length=STFT_WIN_LENGTH, window=torch.hann_window(STFT_WIN_LENGTH)):
   450	        super(stft, self).__init__()

exec
/usr/bin/bash -lc "nl -ba train.py | sed -n '90,260p'; nl -ba src/training/factory.py | sed -n '1,225p'" in /n/fs/gatrdp/codespace/FLAC
 succeeded in 0ms:
    90	    """Construct the pl.Trainer from the assembled kwargs (the tested Trainer boundary)."""
    91	    return pl.Trainer(**build_trainer_kwargs(args, strategy, callbacks, logger, checkpoint_dir, val_args))
    92	
    93	def main():
    94	    torch.set_float32_matmul_precision('medium') 
    95	    torch.multiprocessing.set_sharing_strategy('file_system')
    96	    args = get_all_args()
    97	    seed = args.seed
    98	
    99	    # Set a different seed for each process if using SLURM
   100	    if os.environ.get("SLURM_PROCID") is not None:
   101	        seed += int(os.environ.get("SLURM_PROCID"))
   102	
   103	    pl.seed_everything(seed, workers=True)
   104	
   105	    #Get JSON config from args.model_config
   106	    with open(args.model_config) as f:
   107	        model_config = json.load(f)
   108	
   109	    with open(args.dataset_config) as f:
   110	        dataset_config = json.load(f)
   111	
   112	    train_dl = create_dataloader_from_config(
   113	        dataset_config,
   114	        batch_size=args.batch_size,
   115	        num_workers=args.num_workers,
   116	        sample_rate=model_config["sample_rate"],
   117	        sample_size=model_config["sample_size"],
   118	        audio_channels=model_config.get("audio_channels", 1),
   119	    )
   120	
   121	    val_dl = None
   122	    val_dataset_config = None
   123	    if args.val_dataset_config:
   124	        with open(args.val_dataset_config) as f:
   125	            val_dataset_config = json.load(f)
   126	
   127	        val_dl = create_dataloader_from_config(
   128	            val_dataset_config,
   129	            batch_size=args.batch_size,
   130	            num_workers=args.num_workers,
   131	            sample_rate=model_config["sample_rate"],
   132	            sample_size=model_config["sample_size"],
   133	            audio_channels=model_config.get("audio_channels", 1),
   134	            shuffle=False
   135	        )
   136	
   137	    model = create_model_from_config(model_config)
   138	
   139	    if args.pretrained_ckpt_path:
   140	        print('Loading pretrained model...')
   141	        weights = load_ckpt_state_dict(args.pretrained_ckpt_path)
   142	        weights = {k.replace('diffusion.', ''): v for k, v in weights.items()} # For diffusion
   143	        weights = {k.replace('autoencoder.', ''): v for k, v in weights.items()} # For VAE
   144	        disc_weights = {k: v for k, v in weights.items() if 'discriminator' in k}
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
   206	                                        allgather_bucket_size=5e8,
   207	                                        load_full_weights=True)
   208	        else:
   209	            strategy = args.strategy
   210	    else:
   211	        strategy = 'ddp_find_unused_parameters_true' if args.num_gpus > 1 else "auto"
   212	
   213	    val_args = {}
   214	    
   215	    if args.val_every > 0:
   216	        val_args.update({
   217	            "check_val_every_n_epoch": None,
   218	            "val_check_interval": args.val_every,
   219	        })
   220	
   221	    trainer = construct_trainer(
   222	        args,
   223	        strategy=strategy,
   224	        callbacks=[ckpt_callback, exc_callback, save_model_config_callback],
   225	        logger=logger,
   226	        checkpoint_dir=checkpoint_dir,
   227	        val_args=val_args,
   228	    )
   229	
   230	    trainer.fit(training_wrapper, train_dl, val_dl, ckpt_path=args.ckpt_path if args.ckpt_path else None)
   231	
   232	if __name__ == '__main__':
   233	    main()
     1	import torch
     2	from torch.nn import Parameter
     3	from ..models.factory import create_model_from_config
     4	
     5	YAW_AUG_KEYS = ("enabled", "img_w", "seed")
     6	
     7	
     8	def _parse_yaw_aug_config(training_config):
     9	    """Validate ``training.yaw_aug`` and return the wrapper kwargs it implies.
    10	
    11	    exp_15's training-side random-yaw augmentation (plan §§3.1, 6.2). Every
    12	    failure mode here is fail-closed: a malformed block must stop the launch, not
    13	    quietly train the wrong arm — the whole experiment is one treatment against
    14	    one historical control, so a silently-off (or silently-on) augmentation is
    15	    unrecoverable after the fact.
    16	
    17	    Returns ``{}`` unless the block is present *and* enabled, so the disabled
    18	    path's construction call stays literally the pre-change call (plan §3.3-4:
    19	    the control was trained through that call).
    20	    """
    21	    if "yaw_aug" not in training_config:
    22	        return {}
    23	
    24	    block = training_config["yaw_aug"]
    25	    if not isinstance(block, dict):
    26	        raise ValueError(
    27	            f"training.yaw_aug must be an object with keys {list(YAW_AUG_KEYS)}, "
    28	            f"got {type(block).__name__}"
    29	        )
    30	
    31	    unknown = [k for k in block if k not in YAW_AUG_KEYS]
    32	    if unknown:
    33	        raise ValueError(
    34	            f"training.yaw_aug has unknown key(s) {sorted(unknown)}; "
    35	            f"allowed keys are {list(YAW_AUG_KEYS)}"
    36	        )
    37	
    38	    enabled = block.get("enabled", None)
    39	    if not isinstance(enabled, bool):
    40	        raise ValueError(
    41	            "training.yaw_aug.enabled must be a literal boolean (true/false), got "
    42	            f"{enabled!r}"
    43	        )
    44	
    45	    if not enabled:
    46	        return {}
    47	
    48	    if training_config.get("cond_method", "vanilla") == "fa_invariant":
    49	        raise ValueError(
    50	            "training.yaw_aug.enabled=true with cond_method='fa_invariant' is an "
    51	            "untested combination and out of scope for exp_15: frame averaging "
    52	            "already symmetrises over the yaw subgroup."
    53	        )
    54	
    55	    for key in ("img_w", "seed"):
    56	        if key not in block:
    57	            raise ValueError(
    58	                f"training.yaw_aug.enabled=true requires '{key}' (no default is "
    59	                "assumed: the applied rotation must be stated by the config)"
    60	            )
    61	        if isinstance(block[key], bool) or not isinstance(block[key], int):
    62	            raise ValueError(
    63	                f"training.yaw_aug.{key} must be an int, got {block[key]!r}"
    64	            )
    65	
    66	    if block["img_w"] <= 0:
    67	        raise ValueError(
    68	            f"training.yaw_aug.img_w must be > 0, got {block['img_w']}"
    69	        )
    70	
    71	    return {
    72	        "yaw_aug_enabled": True,
    73	        "yaw_aug_img_w": int(block["img_w"]),
    74	        "yaw_aug_seed": int(block["seed"]),
    75	    }
    76	
    77	
    78	def create_training_wrapper_from_config(model_config, model):
    79	    model_type = model_config.get('model_type', None)
    80	    assert model_type is not None, 'model_type must be specified in model config'
    81	
    82	    training_config = model_config.get('training', None)
    83	    assert training_config is not None, 'training config must be specified in model config'
    84	
    85	    if model_type == 'autoencoder':
    86	        from .autoencoders import AutoencoderTrainingWrapper
    87	        
    88	        ema_copy = None
    89	
    90	        if training_config.get("use_ema", False):
    91	            ema_copy = create_model_from_config(model_config)
    92	            ema_copy = create_model_from_config(model_config) # I don't know why this needs to be called twice but it broke when I called it once
    93	            # Copy each weight to the ema copy
    94	            for name, param in model.state_dict().items():
    95	                if isinstance(param, Parameter):
    96	                    # backwards compatibility for serialized parameters
    97	                    param = param.data
    98	                ema_copy.state_dict()[name].copy_(param)
    99	
   100	        use_ema = training_config.get("use_ema", False)
   101	
   102	        latent_mask_ratio = training_config.get("latent_mask_ratio", 0.0)
   103	
   104	        teacher_model = training_config.get("teacher_model", None)
   105	        if teacher_model is not None:
   106	            teacher_model = create_model_from_config(teacher_model)
   107	            teacher_model = teacher_model.eval().requires_grad_(False)
   108	
   109	            teacher_model_ckpt = training_config.get("teacher_model_ckpt", None)
   110	            if teacher_model_ckpt is not None:
   111	                teacher_model.load_state_dict(torch.load(teacher_model_ckpt)["state_dict"])
   112	            else:
   113	                raise ValueError("teacher_model_ckpt must be specified if teacher_model is specified")
   114	
   115	        return AutoencoderTrainingWrapper(
   116	            model, 
   117	            lr=training_config.get("learning_rate", None),
   118	            warmup_steps=training_config.get("warmup_steps", 0), 
   119	            encoder_freeze_on_warmup=training_config.get("encoder_freeze_on_warmup", False),
   120	            sample_rate=model_config["sample_rate"],
   121	            loss_config=training_config.get("loss_configs", None),
   122	            eval_loss_config=training_config.get("eval_loss_configs", None),
   123	            optimizer_configs=training_config.get("optimizer_configs", None),
   124	            use_ema=use_ema,
   125	            ema_copy=ema_copy if use_ema else None,
   126	            force_input_mono=training_config.get("force_input_mono", False),
   127	            latent_mask_ratio=latent_mask_ratio,
   128	            teacher_model=teacher_model
   129	        )
   130	    
   131	    elif model_type == 'diffusion_cond':
   132	       
   133	        from .diffusion import DiffusionCondTrainingWrapper
   134	
   135	        # exp_15: absent/disabled block -> {} -> the pre-change call verbatim.
   136	        yaw_aug_kwargs = _parse_yaw_aug_config(training_config)
   137	
   138	        return DiffusionCondTrainingWrapper(
   139	            model, 
   140	            lr=training_config.get("learning_rate", None),
   141	            mask_padding=training_config.get("mask_padding", False),
   142	            mask_padding_dropout=training_config.get("mask_padding_dropout", 0.0),
   143	            use_ema = training_config.get("use_ema", True),
   144	            log_loss_info=training_config.get("log_loss_info", False),
   145	            optimizer_configs=training_config.get("optimizer_configs", None),
   146	            pre_encoded=training_config.get("pre_encoded", False),
   147	            cfg_dropout_prob = training_config.get("cfg_dropout_prob", 0.1),
   148	            timestep_sampler = training_config.get("timestep_sampler", "uniform"),
   149	            timestep_sampler_options = training_config.get("timestep_sampler_options", {}),
   150	            p_one_shot=training_config.get("p_one_shot", 0.0),
   151	            test_param = model_config.get("test_setup", None),
   152	            cond_method = training_config.get("cond_method", "vanilla"),
   153	            frame_avg_angles = training_config.get("frame_avg_angles", None),
   154	            **yaw_aug_kwargs,
   155	        )
   156	    
   157	    else:
   158	        raise NotImplementedError(f'Unknown model type: {model_type}')
   159	
   160	def create_metric_callback_from_config(model_config, dataset_id=None, per_scene=False):
   161	    model_type = model_config.get('model_type', None)
   162	    assert model_type is not None, 'model_type must be specified in model config'
   163	
   164	    training_config = model_config.get('training', None)
   165	    assert training_config is not None, 'training config must be specified in model config'
   166	
   167	    metrics_config = training_config.get('metrics', None)
   168	    assert metrics_config is not None, 'metrics config must be specified in training config'
   169	
   170	    sample_rate = model_config["sample_rate"]
   171	    sample_size = model_config["sample_size"]
   172	    audio_channels = model_config.get("audio_channels", 1)
   173	
   174	    from ..metrics.metric_callback import AcousticMetricsCallback
   175	    return AcousticMetricsCallback(
   176	        sample_rate=sample_rate,
   177	        sample_size=sample_size,
   178	        audio_channels=audio_channels,
   179	        dataset_name=dataset_id,
   180	        eval_per_scene=per_scene,
   181	        
   182	        dump_dir=metrics_config.get("dump_dir", None),
   183	
   184	        eval_T60=metrics_config.get("eval_T60", False),
   185	        eval_C50=metrics_config.get("eval_C50", False),
   186	        eval_EDT=metrics_config.get("eval_EDT", False),
   187	        eval_l1_distance=metrics_config.get("eval_l1_distance", False),
   188	        eval_l1_distance_multires=metrics_config.get("eval_l1_distance_multires", False),
   189	        eval_FD=metrics_config.get("eval_FD", False),
   190	        eval_retrieval = metrics_config.get("eval_retrieval", False),
   191	        eval_env = metrics_config.get("eval_env", False),
   192	
   193	        AGREE_ckpt=metrics_config.get("AGREE_ckpt", None),
   194	    )

exec
/usr/bin/bash -lc "git ls-files -s -- AGREE | sed -n '1,80p'; git status --short -- AGREE weights/AGREE/AGREE_fullAR.pt data/AR/train.json" in /n/fs/gatrdp/codespace/FLAC
 succeeded in 0ms:
100644 f163bbe76a34ecc5738fe703ddde564c09a8ea61 0	AGREE/AGREE/__init__.py
100644 55de64db345cb47c2d844a40ee57221a214942c3 0	AGREE/AGREE/audio_model.py
100644 de6611e945a0148d60598b8bf8aea9c0ba3d04b8 0	AGREE/AGREE/factory.py
100644 898ad2110273af5ecd90ce7eb77e22a058be82d1 0	AGREE/AGREE/hf_model.py
100644 6b5c4bce89da5c47aa7e0e92b8a4a898bb7eb996 0	AGREE/AGREE/loss.py
100644 aed7c147bed05faa88552b82a97a6fcbbc75e73b 0	AGREE/AGREE/model.py
100644 bcc0e980c6e67e0aa4ceec5c565541e8ae8bf83e 0	AGREE/AGREE/model_configs/ViT-S-16.json
100644 d3d3d0d0de65bb3b19e7d49d151bfbbc9dedd344 0	AGREE/AGREE/model_configs/dinoV3.json
100644 d1f4d3e0af5aeb68d0ba228da6fef4aac954eae4 0	AGREE/AGREE/model_configs/xRIRViT.json
100644 5c8082b34df2318dd25a4ec8346b3f9a888f38de 0	AGREE/AGREE/pos_embed.py
100644 0ec4a32f3cbce1427b0425d5da88460d76fbcabf 0	AGREE/AGREE/transformer.py
100644 dea3baca1cedb3c73b4af5aff058e2153ff34176 0	AGREE/AGREE/utils.py
100644 09d9fe62a732a2a047553b417f77298383c46d89 0	AGREE/AGREE/xRIR_vit.py
100644 e69de29bb2d1d6434b8b29ae775ad8c2e48c5391 0	AGREE/AGREE_train/__init__.py
100644 9367d41a468cedf1d5fb3ae6cc5f9967ca6d261f 0	AGREE/AGREE_train/data.py
100644 6efa0b75106b43b9a91e6c296f7f984252dd2870 0	AGREE/AGREE_train/data_utils.py
100644 2fad34575f0965f1082752d9df66ceeb2f109344 0	AGREE/AGREE_train/distributed.py
100644 da9ddc96ce63af20ab1a14f94f1645239aea5f04 0	AGREE/AGREE_train/file_utils.py
100644 6d9abed92568d459cbc8d6094ae3901935d89621 0	AGREE/AGREE_train/logger.py
100644 9153523dcd69f3f1d239a771acb0d9c618c1e7d3 0	AGREE/AGREE_train/main.py
100644 562a78f4179439c5251f9440f9a685697461470e 0	AGREE/AGREE_train/params.py
100644 5af494892d1c2c0c26fc878f2e1fa69b585194cb 0	AGREE/AGREE_train/precision.py
100644 f76ba110f2b115b5c888c870d913e22640732f54 0	AGREE/AGREE_train/scheduler.py
100644 206153b9a72f232cb13b80012b912e58b6871e76 0	AGREE/AGREE_train/train.py

exec
/usr/bin/bash -lc 'git ls-tree HEAD AGREE AcousticRooms weights/AGREE/AGREE_fullAR.pt weights/FLAC/VAE.safetensors data/AR/train.json; git check-ignore -v AGREE AcousticRooms weights/AGREE/AGREE_fullAR.pt weights/FLAC/VAE.safetensors 2>&1 || true' in /n/fs/gatrdp/codespace/FLAC
 succeeded in 0ms:
040000 tree 5a21e1027ff6ee140ccf974896f03f6036c18573	AGREE
100644 blob 41649c8a8fc465d99e4d19f3e6fc71fd459a4174	data/AR/train.json
.gitignore:176:weights/	weights/AGREE/AGREE_fullAR.pt
.gitignore:176:weights/	weights/FLAC/VAE.safetensors

exec
/usr/bin/bash -lc "git config --get diff.ignoreSubmodules || true; git config --get status.submoduleSummary || true; if [ -f .gitmodules ]; then sed -n '1,200p' .gitmodules; else echo 'no .gitmodules'; fi; git ls-files -s | awk '"'$1==160000 {print}'"' | sed -n '1,120p'" in /n/fs/gatrdp/codespace/FLAC
 succeeded in 0ms:
no .gitmodules

exec
/usr/bin/bash -lc "git log --all --summary --format= | grep -E 'mode change|create mode 120000|create mode 160000|delete mode 120000|delete mode 160000' | sed -n '1,120p'" in /n/fs/gatrdp/codespace/FLAC
exec
/usr/bin/bash -lc "git log --oneline --follow -- data/AR/train.json | sed -n '1,40p'; git log --oneline --all -- AGREE | sed -n '1,40p'" in /n/fs/gatrdp/codespace/FLAC
exec
/usr/bin/bash -lc "grep -RInE \"data/AR/train\\.json|json_file_path|custom_metadata_module\" train.py src worklog/worklog_yixun/exp_11_fa_orbit_claude/FLAC_AR_*.json | sed -n '1,220p'" in /n/fs/gatrdp/codespace/FLAC
 succeeded in 211ms:
2e3f847 Adding FLAC and AGREE code
15de475 feat: add equivariant ViT CNN ablation
6bc44a8 finish ablation for nearest RIR as latent
849a847 init
2e3f847 Adding FLAC and AGREE code

 succeeded in 335ms:
src/data/dataset.py:22:    json_file_path: str,  # json file to read
src/data/dataset.py:29:        with open(json_file_path, 'r') as f:
src/data/dataset.py:117:    json_file_path=None,
src/data/dataset.py:122:    # check extension of json_file_path if not none
src/data/dataset.py:123:    if json_file_path is not None:
src/data/dataset.py:124:        json_ext = os.path.splitext(json_file_path)[1].lower()
src/data/dataset.py:129:        if json_file_path is not None and folder_name is not None and json_ext == '.json':
src/data/dataset.py:131:            subfolders, files = json_scandir(dir=path, json_file_path=json_file_path, folder_name=folder_name)
src/data/dataset.py:147:        json_file_path: Optional[str] = None,
src/data/dataset.py:158:        self.json_file_path = json_file_path
src/data/dataset.py:189:        self.json_file_path = configs[0].json_file_path if hasattr(configs[0], 'json_file_path') else None
src/data/dataset.py:219:            self.filenames.extend(get_audio_filenames(paths=config.path, keywords=keywords, json_file_path=config.json_file_path, folder_name=config.folder_name))
src/data/dataset.py:267:            info['json_file_path'] = self.json_file_path 
src/data/dataset.py:367:        custom_metadata_module_path = audio_dir_config.get("custom_metadata_module", None)
src/data/dataset.py:369:        if custom_metadata_module_path is not None:
src/data/dataset.py:370:            spec = importlib.util.spec_from_file_location("metadata_module", custom_metadata_module_path)
src/data/dataset.py:378:        json_file_path = audio_dir_config.get("json_file_path", None)
src/data/dataset.py:386:                json_file_path=json_file_path, 
src/configs/dataset_configs/AR/eval/acousticroom_seeneval_1.json:7:            "json_file_path": "data/AR/seen_eval.json",
src/configs/dataset_configs/AR/eval/acousticroom_seeneval_1.json:8:            "custom_metadata_module": "src/configs/dataset_configs/custom_metadata/AR_md.py",
src/configs/dataset_configs/AR/eval/acousticroom_seeneval_4.json:7:            "json_file_path": "data/AR/seen_eval.json",
src/configs/dataset_configs/AR/eval/acousticroom_seeneval_4.json:8:            "custom_metadata_module": "src/configs/dataset_configs/custom_metadata/AR_md.py",
src/configs/dataset_configs/AR/eval/acousticroom_unseeneval.json:7:            "json_file_path": "data/AR/unseen_eval.json",
src/configs/dataset_configs/AR/eval/acousticroom_unseeneval.json:8:            "custom_metadata_module": "src/configs/dataset_configs/custom_metadata/AR_md.py",
src/configs/dataset_configs/AR/eval/acousticroom_unseeneval_1.json:7:            "json_file_path": "data/AR/unseen_eval.json",
src/configs/dataset_configs/AR/eval/acousticroom_unseeneval_1.json:8:            "custom_metadata_module": "src/configs/dataset_configs/custom_metadata/AR_md.py",
src/configs/dataset_configs/AR/eval/acousticroom_seeneval.json:7:            "json_file_path": "data/AR/seen_eval.json",
src/configs/dataset_configs/AR/eval/acousticroom_seeneval.json:8:            "custom_metadata_module": "src/configs/dataset_configs/custom_metadata/AR_md.py",
src/configs/dataset_configs/AR/eval/acousticroom_unseeneval_4.json:7:            "json_file_path": "data/AR/unseen_eval.json",
src/configs/dataset_configs/AR/eval/acousticroom_unseeneval_4.json:8:            "custom_metadata_module": "src/configs/dataset_configs/custom_metadata/AR_md.py",
src/configs/dataset_configs/AR/train/acousticroom_train.json:7:            "json_file_path": "data/AR/train.json",
src/configs/dataset_configs/AR/train/acousticroom_train.json:8:            "custom_metadata_module": "src/configs/dataset_configs/custom_metadata/AR_md.py",
src/configs/dataset_configs/AR/train/acousticroom_train_VAECtxt.json:7:            "json_file_path": "data/AR/train.json",
src/configs/dataset_configs/AR/train/acousticroom_train_VAECtxt.json:8:            "custom_metadata_module": "src/configs/dataset_configs/custom_metadata/AR_md.py",
src/configs/dataset_configs/HAA/eval/haa_test_1.json:7:            "json_file_path": "data/HAA/test_base.json",
src/configs/dataset_configs/HAA/eval/haa_test_1.json:8:            "custom_metadata_module": "src/configs/dataset_configs/custom_metadata/HAA_md.py",
src/configs/dataset_configs/HAA/eval/haa_test.json:7:            "json_file_path": "data/HAA/test_base.json",
src/configs/dataset_configs/HAA/eval/haa_test.json:8:            "custom_metadata_module": "src/configs/dataset_configs/custom_metadata/HAA_md.py",
src/configs/dataset_configs/HAA/eval/haa_val.json:7:            "json_file_path": "data/HAA/val_base.json",
src/configs/dataset_configs/HAA/eval/haa_val.json:8:            "custom_metadata_module": "src/configs/dataset_configs/custom_metadata/HAA_md.py",
src/configs/dataset_configs/HAA/train/haa_train.json:7:            "json_file_path": "data/HAA/train_base.json",
src/configs/dataset_configs/HAA/train/haa_train.json:8:            "custom_metadata_module": "src/configs/dataset_configs/custom_metadata/HAA_md.py",

 succeeded in 3885ms:
 mode change 100644 => 100755 worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_equiv_probe.py

2026-08-11T19:05:58.191111Z ERROR codex_models_manager::manager: failed to renew cache TTL: missing field `base_instructions` at line 94 column 5
exec
/usr/bin/bash -lc 'for f in $(ls -1t worklog/worklog_yixun/exp_11_fa_orbit_claude/*guardtests.log 2>/dev/null | head -4); do echo "FILE $f"; grep -A18 -B2 -- '"'--- D\\.' \""'$f" | sed -n '"'1,40p'; tail -8 \""'$f"; done' in /n/fs/gatrdp/codespace/FLAC
 succeeded in 0ms:
FILE worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-11_14-57-42_guardtests.log
PASS  restart MAXSTEPS<=step  (rc=2)
PASS  initial refuses an existing run dir  (rc=2)
--- D. commit-binding / sbatch-only gates (REAL mode) ---
PASS  wrong EXPECT_SHA aborts  (rc=2)
PASS  real mode needs sbatch  (rc=2)
PASS  moved HEAD, surfaces identical -> gate passes  (rc=2)
PASS  moved HEAD, surfaces changed -> aborts  (rc=2)
--- E. semantic gate on a mislabelled config (temp copy; tracked tree untouched) ---
PASS  orbit mismatch rejected  (rc=1)
PASS  tracked tree unchanged by the suite (snapshot before == after)
--- F. exit taxonomy, mocked (round-3 B5) ---
PASS  class 0 complete  (rc=0)
PASS  class 6 world-size absent  (rc=6)
PASS  class 6 wrong world-size  (rc=6)
PASS  class 3 OOM on nonzero rc  (rc=3)
PASS  class 4 missing marker  (rc=4)
PASS  class 7 logs differ  (rc=7)
PASS  class 7 copy missing  (rc=7)
PASS  class 7 tee failed  (rc=7)
PASS  raw rc preserved  (rc=9)
--- G. restart preflight depth, mocked checkpoints (round-3 B2) ---
PASS  FIFO no longer uses mktemp -u
PASS  FIFO removal is in the exit trap
PASS  pip freeze status is checked before hashing
PASS  the final record's tee status is captured
PASS  the approved wandb entity is exported
PASS  the launcher runs the wandb readback and gates on its result

=== guard tests: 89 passed, 5 failed ===
FILE worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-11_14-55-12_guardtests.log
PASS  restart MAXSTEPS<=step  (rc=2)
PASS  initial refuses an existing run dir  (rc=2)
--- D. commit-binding / sbatch-only gates (REAL mode) ---
PASS  wrong EXPECT_SHA aborts  (rc=2)
PASS  real mode needs sbatch  (rc=2)
PASS  moved HEAD, surfaces identical -> gate passes  (rc=2)
PASS  moved HEAD, surfaces changed -> aborts  (rc=2)
--- E. semantic gate on a mislabelled config (temp copy; tracked tree untouched) ---
PASS  orbit mismatch rejected  (rc=1)
PASS  tracked tree unchanged by the suite (snapshot before == after)
--- F. exit taxonomy, mocked (round-3 B5) ---
PASS  class 0 complete  (rc=0)
PASS  class 6 world-size absent  (rc=6)
PASS  class 6 wrong world-size  (rc=6)
PASS  class 3 OOM on nonzero rc  (rc=3)
PASS  class 4 missing marker  (rc=4)
PASS  class 7 logs differ  (rc=7)
PASS  class 7 copy missing  (rc=7)
PASS  class 7 tee failed  (rc=7)
PASS  raw rc preserved  (rc=9)
--- G. restart preflight depth, mocked checkpoints (round-3 B2) ---
PASS  FIFO no longer uses mktemp -u
PASS  FIFO removal is in the exit trap
PASS  pip freeze status is checked before hashing
PASS  the final record's tee status is captured
PASS  the approved wandb entity is exported
PASS  the launcher runs the wandb readback and gates on its result

=== guard tests: 89 passed, 5 failed ===
FILE worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-10_02-29-13_screen_guardtests.log
PASS  an arm ckpt with no launch manifest is refused  (rc=2)
PASS  a launch manifest for another config is refused  (rc=2)
--- D. valid screens reach the eval argv ---
PASS  VANL runs a VANILLA evaluation  (rc=0)
PASS  VANL builds the vanilla eval name  (rc=0)
PASS  VANL uses the VANCKPT config  (rc=0)
PASS  VANL announces itself as orbit-free  (rc=0)
PASS  VANL screen cells exist too  (rc=0)
PASS  VANL refuses r3  (rc=2)
PASS  VANL refuses cross  (rc=2)
PASS  an ORBIT ckpt is refused as VANL  (rc=2)
PASS  VANL q9 cell is registered  (rc=0)
PASS  C4L q9 cell is registered  (rc=0)
PASS  q9 is the VANL/C4L pair only  (rc=2)
PASS  q9 is the 40k endpoint only  (rc=2)
PASS  q9 uses the confirmatory seeds  (rc=2)
PASS  r3 needs a rotation  (rc=2)
PASS  r3 rejects an unregistered offset  (rc=2)
PASS  r3 rejects 5.62 (near-miss)  (rc=2)
PASS  r3 is the 40k endpoint only  (rc=2)
PASS  r3 is seed 42 by contract  (rc=2)
PASS  the launcher untracks its own Slurm transcript at launch
PASS  the untrack outcome is recorded in the launch manifest
PASS  the closure procedure is documented in the launcher header
PASS  untracking tolerates absence and reports failure without aborting
PASS  no transcript of a RUNNING arm is tracked

=== screen guard tests: 172 passed, 0 failed ===
log: worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-10_02-29-13_screen_guardtests.log
FILE worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-10_02-28-11_guardtests.log
PASS  restart MAXSTEPS<=step  (rc=2)
PASS  initial refuses an existing run dir  (rc=2)
--- D. commit-binding / sbatch-only gates (REAL mode) ---
PASS  wrong EXPECT_SHA aborts  (rc=2)
PASS  real mode needs sbatch  (rc=2)
--- E. semantic gate on a mislabelled config (temp copy; tracked tree untouched) ---
PASS  orbit mismatch rejected  (rc=1)
PASS  tracked tree unchanged by the suite (snapshot before == after)
--- F. exit taxonomy, mocked (round-3 B5) ---
PASS  class 0 complete  (rc=0)
PASS  class 6 world-size absent  (rc=6)
PASS  class 6 wrong world-size  (rc=6)
PASS  class 3 OOM on nonzero rc  (rc=3)
PASS  class 4 missing marker  (rc=4)
PASS  class 7 logs differ  (rc=7)
PASS  class 7 copy missing  (rc=7)
PASS  class 7 tee failed  (rc=7)
PASS  raw rc preserved  (rc=9)
--- G. restart preflight depth, mocked checkpoints (round-3 B2) ---
synthetic checkpoints written
PASS  preflight accepts a good ckpt  (rc=0)
PASS  FIFO removal is in the exit trap
PASS  pip freeze status is checked before hashing
PASS  the final record's tee status is captured
PASS  the approved wandb entity is exported
PASS  the launcher runs the wandb readback and gates on its result

=== guard tests: 92 passed, 0 failed ===
log: worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-10_02-28-11_guardtests.log

exec
/usr/bin/bash -lc "grep -n '"'^FAIL'"' worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-11_14-57-42_guardtests.log | sed -n '1,80p'; grep -n -A6 -B4 '"'^FAIL'"' worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-11_14-57-42_guardtests.log | sed -n '1,180p'" in /n/fs/gatrdp/codespace/FLAC
 succeeded in 0ms:
67:FAIL  the submitter no longer allocates the C16 RESTART pin
88:FAIL  submitter derives smoke flags: want rc=0 + '--gres=gpu:l40:4', got rc=2
92:FAIL  submitter derives cpus/mem from the rung: want rc=0 + '--cpus-per-task=36', got rc=2
96:FAIL  submitter derives 8x8 resources: want rc=0 + '--mem=108G', got rc=2
110:FAIL  dry run publishes no submission manifest: want rc=0 + 'DRYRUN sbatch', got rc=2
63---- G2. Q10: the JOB selects and enforces the RESTART time pin (re-pin fix 1) ---
64-PASS  a RESTART leg selects the RESTART pin  (rc=0)
65-PASS  an INITIAL launch keeps the INITIAL pin  (rc=0)
66-PASS  the allocation gate names the pin it enforced
67:FAIL  the submitter no longer allocates the C16 RESTART pin
68---- G3. Q10: the 40k -> 100k EXTENSION preflight contract (re-pin fix 1) ---
69-extension fixture written
70-PASS  the ORDINARY contract refuses the extension (the bug)  (rc=2)
71-PASS  extension accepts the 40k->100k leg  (rc=0)
72-PASS  extension keeps the ORIGINAL launch commit  (rc=0)
73-PASS  extension refuses a shrinking budget  (rc=2)
--
84---- H. the submitter refuses un-pinned submission ---
85-PASS  every submitter pin holds a concrete value
86-PASS  the submitter still refuses a placeholder pin if one returns
87-PASS  submitter rejects a bad arm  (rc=2)
88:FAIL  submitter derives smoke flags: want rc=0 + '--gres=gpu:l40:4', got rc=2
89-        | tracked measurement surfaces have uncommitted changes - commit first, abort:
90-        |  M worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch
91-        |  M worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh
92:FAIL  submitter derives cpus/mem from the rung: want rc=0 + '--cpus-per-task=36', got rc=2
93-        | tracked measurement surfaces have uncommitted changes - commit first, abort:
94-        |  M worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch
95-        |  M worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh
96:FAIL  submitter derives 8x8 resources: want rc=0 + '--mem=108G', got rc=2
97-        | tracked measurement surfaces have uncommitted changes - commit first, abort:
98-        |  M worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch
99-        |  M worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh
100---- I. flock run ownership, both contention directions (round-3 B3 residual) ---
101-PASS  contender refused while the lock is held
102-PASS  lock free after the holder exits (no stale-recovery path needed)
--
106-PASS  ambient OUTPUT_ROOT rejected under Slurm  (rc=2)
107-PASS  launcher pins the production root literally
108-PASS  submitter exports the fixed root, not ambient state
109---- K. the submitter publishes intent BEFORE sbatch (NEW-3) ---
110:FAIL  dry run publishes no submission manifest: want rc=0 + 'DRYRUN sbatch', got rc=2
111-        | tracked measurement surfaces have uncommitted changes - commit first, abort:
112-        |  M worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch
113-        |  M worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh
114-PASS  a dry run leaves no submission manifest behind
115-PASS  intent manifest is published before the sbatch call
116-PASS  an unrecordable job is cancelled

exec
/usr/bin/bash -lc "find src worklog/worklog_yixun/exp_11_fa_orbit_claude worklog/worklog_yixun/exp_07_fa_scratch_claude -type l -print | sed -n '1,160p'; stat -c '%F %n' train.py defaults.ini worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py worklog/worklog_yixun/exp_11_fa_orbit_claude/assert_arm_configs_exp11.py worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_wandb_readback.py worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_classify.py worklog/worklog_yixun/exp_07_fa_scratch_claude/FLAC_AR_BF.json" in /n/fs/gatrdp/codespace/FLAC
 succeeded in 0ms:
regular file train.py
regular file defaults.ini
regular file worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch
regular file worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py
regular file worklog/worklog_yixun/exp_11_fa_orbit_claude/assert_arm_configs_exp11.py
regular file worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_wandb_readback.py
regular file worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_classify.py
regular file worklog/worklog_yixun/exp_07_fa_scratch_claude/FLAC_AR_BF.json

codex
BLOCKING

1. The closure omits a runtime-loaded tracked file. The launcher selects `src/configs/dataset_configs/AR/train/acousticroom_train.json`, which names `data/AR/train.json`; `src/data/dataset.py` opens that JSON to construct the training population. A commit or uncommitted edit to `data/AR/train.json` can therefore change training while this gate passes. Add it to the commit comparison, guard-test helper, and drift gate.

2. `"$EXPDIR"/FLAC_AR_*.json` is expanded by Bash using the current worktree before Git sees it. If a matching config existed only at `EXPECT_SHA` and was deleted or renamed outside the pattern, its old path is omitted because other matching configs keep the glob from remaining literal. Quote the complete Git pathspec or, preferably, enumerate the five runtime configs explicitly. The unchanged drift gate has the same deletion hole.

3. `EXPECT_SHA` is no longer enforced as a full immutable object ID. `git cat-file -e "${EXPECT_SHA}^{commit}"` rejects an ambiguous abbreviation, but accepts a unique short SHA, tag, revision expression, or `HEAD`. In particular, `EXPECT_SHA=HEAD` makes the subsequent `git diff HEAD HEAD` empty and defeats binding. Validate the expected value as the full canonical commit object ID before comparing content.

4. The gate races with the two writers it is intended to tolerate. `HEAD_SHA` is captured before `git status`, while the content comparison later uses symbolic `HEAD`. A commit between those operations can leave `HEAD_SHA == EXPECT_SHA`, take the fast path, and run the newer tree without any committed-content comparison. At minimum, compare against the captured OID and re-read `HEAD` before acceptance, aborting if it moved. A dedicated immutable worktree is the stronger solution.

5. The surfaces-identical guard is not reliable after this patch is committed. Because `fa_orbit_train.sbatch` is itself in the closure, if the launcher and guard test land together every ancestor predating that commit is surfaces-changed, so the identical case can silently SKIP. The existing PASS log was produced while the fix was uncommitted and does not establish post-commit coverage. Use deterministic synthetic history or make absence of the required fixture a test failure.

NON-BLOCKING

- The real-mode `SD="$(surface_diff_vs_expect 2>&1)" || die ...` handling is sound: the assignment preserves `git diff`’s status, `SD` remains defined, and `set -uo pipefail` introduces no hole. Unknown commits and actual diff failures abort. The existing `DRIFT="$(git status ...)"` still ignores `git status` failure, however, so that older gate remains fail-open on command failure.

- For pathspecs actually passed to Git, additions/deletions, renames, mode changes, and tracked symlink changes produce nonempty output. There are currently no submodules or symlinks in the closure. Git does not dereference symlink targets; future submodules should also use `--ignore-submodules=none` explicitly.

- `fa_orbit_launch.sh` does not exist in the live tree or repository history and is never invoked by the sbatch file. That closure entry is inert and its explanatory comment is inaccurate. The actual `fa_orbit_submit.sh` is submission-time code, so later changes to it cannot alter an already queued job.

- The dry-run branch safely remains advisory, but a failing `git diff` is mislabeled as “training surfaces changed.” Give diff failure its own advisory message. The guard’s `rc=2` expectation is workable because the acceptance text proves the gate passed, but the comment claiming the 99-TB VRAM check causes the failure is inaccurate; dirty-tree or allocation checks normally fail first.

- Already-submitted jobs will not acquire this fix because Slurm spools the batch script at submission. They require resubmission. Moreover, `0f0acb2..135cb4b` includes changes under `src/training/`, so those revisions are not identical under the proposed closure anyway.

REVISE
tokens used
158,830
BLOCKING

1. The closure omits a runtime-loaded tracked file. The launcher selects `src/configs/dataset_configs/AR/train/acousticroom_train.json`, which names `data/AR/train.json`; `src/data/dataset.py` opens that JSON to construct the training population. A commit or uncommitted edit to `data/AR/train.json` can therefore change training while this gate passes. Add it to the commit comparison, guard-test helper, and drift gate.

2. `"$EXPDIR"/FLAC_AR_*.json` is expanded by Bash using the current worktree before Git sees it. If a matching config existed only at `EXPECT_SHA` and was deleted or renamed outside the pattern, its old path is omitted because other matching configs keep the glob from remaining literal. Quote the complete Git pathspec or, preferably, enumerate the five runtime configs explicitly. The unchanged drift gate has the same deletion hole.

3. `EXPECT_SHA` is no longer enforced as a full immutable object ID. `git cat-file -e "${EXPECT_SHA}^{commit}"` rejects an ambiguous abbreviation, but accepts a unique short SHA, tag, revision expression, or `HEAD`. In particular, `EXPECT_SHA=HEAD` makes the subsequent `git diff HEAD HEAD` empty and defeats binding. Validate the expected value as the full canonical commit object ID before comparing content.

4. The gate races with the two writers it is intended to tolerate. `HEAD_SHA` is captured before `git status`, while the content comparison later uses symbolic `HEAD`. A commit between those operations can leave `HEAD_SHA == EXPECT_SHA`, take the fast path, and run the newer tree without any committed-content comparison. At minimum, compare against the captured OID and re-read `HEAD` before acceptance, aborting if it moved. A dedicated immutable worktree is the stronger solution.

5. The surfaces-identical guard is not reliable after this patch is committed. Because `fa_orbit_train.sbatch` is itself in the closure, if the launcher and guard test land together every ancestor predating that commit is surfaces-changed, so the identical case can silently SKIP. The existing PASS log was produced while the fix was uncommitted and does not establish post-commit coverage. Use deterministic synthetic history or make absence of the required fixture a test failure.

NON-BLOCKING

- The real-mode `SD="$(surface_diff_vs_expect 2>&1)" || die ...` handling is sound: the assignment preserves `git diff`’s status, `SD` remains defined, and `set -uo pipefail` introduces no hole. Unknown commits and actual diff failures abort. The existing `DRIFT="$(git status ...)"` still ignores `git status` failure, however, so that older gate remains fail-open on command failure.

- For pathspecs actually passed to Git, additions/deletions, renames, mode changes, and tracked symlink changes produce nonempty output. There are currently no submodules or symlinks in the closure. Git does not dereference symlink targets; future submodules should also use `--ignore-submodules=none` explicitly.

- `fa_orbit_launch.sh` does not exist in the live tree or repository history and is never invoked by the sbatch file. That closure entry is inert and its explanatory comment is inaccurate. The actual `fa_orbit_submit.sh` is submission-time code, so later changes to it cannot alter an already queued job.

- The dry-run branch safely remains advisory, but a failing `git diff` is mislabeled as “training surfaces changed.” Give diff failure its own advisory message. The guard’s `rc=2` expectation is workable because the acceptance text proves the gate passed, but the comment claiming the 99-TB VRAM check causes the failure is inaccurate; dirty-tree or allocation checks normally fail first.

- Already-submitted jobs will not acquire this fix because Slurm spools the batch script at submission. They require resubmission. Moreover, `0f0acb2..135cb4b` includes changes under `src/training/`, so those revisions are not identical under the proposed closure anyway.

REVISE
