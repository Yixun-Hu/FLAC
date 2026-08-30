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
session id: 01a05152-a5fc-7700-b3aa-00bdf56f11af
--------
user
Micro delta review (read-only; no installs/modifications; output = review text only). Repo /n/fs/gatrdp/codespace/FLAC. Production incident: 4 exp_11 chunk legs burned in 3 days on known ECC-flaky nodes (neu306 x2 on Aug 27, neu301 x2 NCCL-watchdog-hang on Aug 30), freezing all 4 remaining arms (2-consecutive-failure rule). The training submitter worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submit.sh had no node-exclusion path (only the screen submitter did). Delta (diff at /tmp/claude-374349/-n-fs-gatrdp-codespace-FLAC/66305eb9-88c7-41df-95da-dde30a426dc2/scratchpad/exclude_delta.diff): EXCLUDE env with DEFAULT = the 8-node sick list from CLAUDE.md, validated early (shape [a-z0-9,] only, before the drift gate), appended as --exclude=... to the sbatch argv when non-empty; EXCLUDE="" disables; intent manifest gains an 'exclude' line. 4 guard cases added (2 pass dirty, 2 are the known post-commit dirty-tree class). Check: (a) the ${EXCLUDE-default} vs ${EXCLUDE:-default} distinction is used correctly for the explicit-empty-disables semantics; (b) validation placement/shape; (c) --exclude interaction with the reservation flock/squeue path and DRYRUN; (d) any hole. Verdict: APPROVED or REVISE.
codex
I’ll inspect the supplied delta and the surrounding submitter/reservation code read-only, then return only the review verdict and findings.
exec
/usr/bin/bash -lc "sed -n '1,260p' /tmp/claude-374349/-n-fs-gatrdp-codespace-FLAC/66305eb9-88c7-41df-95da-dde30a426dc2/scratchpad/exclude_delta.diff && sed -n '1,320p' worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submit.sh && rg -n \"EXCLUDE|exclude|flock|squeue|DRYRUN|intent\" worklog/worklog_yixun/exp_11_fa_orbit_claude -g '*.sh' -g '*.bats' -g '*.md' -g '*.txt'" in /n/fs/gatrdp/codespace/FLAC
 exited 2 in 0ms:
sed: can't read /tmp/claude-374349/-n-fs-gatrdp-codespace-FLAC/66305eb9-88c7-41df-95da-dde30a426dc2/scratchpad/exclude_delta.diff: No such file or directory

codex
The exact temporary diff path is absent, so I’m locating the delta and checking whether it is already reflected in the working tree without changing anything.
exec
/usr/bin/bash -lc "find /tmp/claude-374349 -type f -name 'exclude_delta.diff' -print 2>/dev/null; git status --short; git diff -- worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submit.sh; sed -n '1,360p' worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submit.sh; rg -n \"EXCLUDE|exclude|flock|squeue|DRYRUN|intent\" worklog/worklog_yixun/exp_11_fa_orbit_claude CLAUDE.md -g '*.sh' -g '*.bats' -g '*.md' -g '*.txt'" in /n/fs/gatrdp/codespace/FLAC
 exited 127 in 0ms:
 M worklog/worklog_yixun/exp_11_fa_orbit_claude/arm_launch_registry.json
 M worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submit.sh
 M worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh
 M worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_01_results.html
 M worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_results_assets/make_figures.py
 M worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_results_assets/yaw_gen_absolute_mr.png
 M worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_results_assets/yaw_gen_absolute_mr_dark.png
 M worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_results_assets/yaw_gen_dose_response.png
 M worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_results_assets/yaw_gen_dose_response_dark.png
 M worklog/worklog_yixun/exp_15_yaw_aug_claude/slurm_screen_exp15-screen-VANL-rrob-rotrand42-40000-s42-K1_3718004.out
 M worklog/worklog_yixun/exp_15_yaw_aug_claude/slurm_screen_exp15-screen-YAWAUG-rrob-rotrand44-40000-s44-K8_3718001.out
 M worklog/worklog_yixun/exp_15_yaw_aug_claude/slurm_screen_exp15-screen-YAWAUG-rrob-rotrand45-40000-s45-K8_3718002.out
 M worklog/worklog_yixun/exp_15_yaw_aug_claude/slurm_screen_exp15-screen-YAWAUG-rrob-rotrand46-40000-s46-K8_3718003.out
 M worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-18_09-59-13_YAWAUG_rrob_rotrand44_S40000_s44_K8_jid3718001_screen.log
 M worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-18_10-04-13_YAWAUG_rrob_rotrand45_S40000_s45_K8_jid3718002_screen.log
 M worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-18_10-04-14_YAWAUG_rrob_rotrand46_S40000_s46_K8_jid3718003_screen.log
 M worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-18_10-05-14_VANL_rrob_rotrand42_S40000_s42_K1_jid3718004_screen.log
?? AGENTS.md
?? AcousticRooms
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/.chunk_watchdog.lock
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/.chunk_watchdog_state
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/.submit_C16.lock
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/.submit_C32.lock
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/.submit_C4L.lock
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/.submit_C8.lock
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/.submit_VANL.lock
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-12_18-23-10_guardtests.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-13_12-06-48_guardtests.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-13_12-08-41_guardtests.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-13_12-10-57_guardtests.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-13_12-39-58_guardtests.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-13_12-42-29_guardtests.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-13_13-14-34_guardtests.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-13_13-18-38_guardtests.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-13_13-36-00_guardtests.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-13_13-39-25_guardtests.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-13_13-46-48_guardtests.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-20_00-06-46_C8_8x8_jid3729417_manifest.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-20_12-04-27_C8_8x8_jid3738922_manifest.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-21_05-46-00_C8_8x8_jid3739783_manifest.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-22_06-52-18_C8_8x8_jid3744282_manifest.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-22_18-43-14_C8_8x8_jid3746193_manifest.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-26_00-49-16_C8_8x8_jid3746729_manifest.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-26_09-31-18_C8_8x8_jid3755918_manifest.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-27_06-28-34_C8_8x8_jid3756748_manifest.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-27_08-30-04_C4L_8x8_jid3759509_manifest.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-27_08-51-27_C8_8x8_jid3759511_manifest.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-27_09-20-15_C16_8x8_jid3759512_manifest.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-27_09-21-29_C32_8x8_jid3759513_manifest.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-27_09-25-28_C8_8x8_jid3759739_manifest.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-27_09-51-34_C4L_8x8_jid3759877_manifest.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-27_17-10-31_C4L_8x8_jid3760004_manifest.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-27_18-36-32_C16_8x8_jid3760566_manifest.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-28_13-58-24_C4L_8x8_jid3762225_manifest.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-28_16-07-41_C16_8x8_jid3763055_manifest.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-28_19-07-10_C4L_8x8_jid3766850_manifest.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-28_20-06-46_C16_8x8_jid3767211_manifest.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-28_23-02-28_C4L_8x8_jid3767339_manifest.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-29_02-03-39_C16_8x8_jid3767672_manifest.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-29_02-14-40_C4L_8x8_jid3767923_manifest.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-29_03-29-53_C4L_8x8_jid3768463_manifest.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-29_11-26-47_C4L_8x8_jid3768717_manifest.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-29_11-42-57_C16_8x8_jid3768887_manifest.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-29_12-54-00_C4L_8x8_jid3770348_manifest.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-29_13-26-01_C4L_8x8_jid3770468_manifest.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-29_14-44-31_C16_8x8_jid3770829_manifest.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-29_15-19-32_C4L_8x8_jid3770899_manifest.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-29_17-38-57_C4L_8x8_jid3771524_manifest.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-29_20-46-56_C16_8x8_jid3771761_manifest.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-30_00-19-25_C4L_8x8_jid3772057_manifest.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-30_00-50-59_C16_8x8_jid3772839_manifest.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-30_01-16-26_C4L_8x8_jid3772907_manifest.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-30_01-41-27_C16_8x8_jid3772936_manifest.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-30_02-09-59_guardtests.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-30_02-13-09_guardtests.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-30_02-14-58_guardtests.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-30_02-16-48_guardtests.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_chunk_watchdog.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_exclude_review.md
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C16_1786827863745525788-28bc54ec.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C16_1786880760750725058-30e8e545.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C16_1787833704473406742-debcc623.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C16_1787847873579363919-9707e895.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C16_1787880924099662889-49d33466.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C16_1787958694845045830-54a720f1.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C16_1787972838112693339-f5eecdf4.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C16_1787994492150658728-321b2b74.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C16_1788028934146715351-17bb9ee1.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C16_1788039750549795171-51f004ce.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C16_1788063183060898440-ecceac2c.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C16_1788067085207760940-7f4d2f69.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C32_1786867229669695302-38fe9bed.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C32_1787833704709811683-68f1da85.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C4L_1786811884092316069-e8acd396.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C4L_1786824227714286395-5c5b204f.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C4L_1786836582874260462-8fa9ab22.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C4L_1786881369650820262-7915bbe9.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C4L_1786896084024248183-812dd755.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C4L_1787833684112299278-66698562.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C4L_1787838542392229646-106efa10.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C4L_1787839743084138435-d16c284c.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C4L_1787870412980938588-af886b91.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C4L_1787944564324605348-b3d049d1.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C4L_1787962900516065326-cc4e1599.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C4L_1787977345227042823-3f15ead7.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C4L_1787988468966846588-c0501572.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C4L_1787992979584879564-19ec1a56.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C4L_1788022126866537111-e7426e70.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C4L_1788024227915375667-2183abee.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C4L_1788030139102336394-24898657.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C4L_1788036445396505854-8a1cf728.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C4L_1788043956331253226-ddf9f2c2.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C4L_1788065584215897471-2bce5326.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C8_1786817007645863367-cbf2d2ee.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C8_1786859416626206535-b103a3de.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C8_1787205563691186217-36b8af59.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C8_1787248499509931423-cfeec915.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C8_1787312205797380084-5a78683b.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C8_1787402574400771569-0f134aca.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C8_1787445512544433086-c87a8a1c.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C8_1787726825469673513-beddfe72.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C8_1787758345565431042-cd08a189.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C8_1787833704241222992-e0e3f15f.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C8_1787837006441323303-0e91ba37.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_VANL_1786815790075262537-6b2a5f4d.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_VANL_1786825748403923329-8b711ea3.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_VANL_1786870847094858474-1a1a9e8a.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_VANL_1786881373942942586-3a3e5471.txt
?? worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_01_results.html.orig
?? worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_png_figures/
?? worklog/worklog_yixun/exp_15_yaw_aug_claude/slurm_screen_exp15-screen-VANL-rrob-rotrand42-40000-s42-K8_3718010.out
?? worklog/worklog_yixun/exp_15_yaw_aug_claude/slurm_screen_exp15-screen-VANL-rrob-rotrand43-40000-s43-K1_3718006.out
?? worklog/worklog_yixun/exp_15_yaw_aug_claude/slurm_screen_exp15-screen-VANL-rrob-rotrand43-40000-s43-K8_3718011.out
?? worklog/worklog_yixun/exp_15_yaw_aug_claude/slurm_screen_exp15-screen-VANL-rrob-rotrand44-40000-s44-K1_3718007.out
?? worklog/worklog_yixun/exp_15_yaw_aug_claude/slurm_screen_exp15-screen-VANL-rrob-rotrand44-40000-s44-K8_3718064.out
?? worklog/worklog_yixun/exp_15_yaw_aug_claude/slurm_screen_exp15-screen-VANL-rrob-rotrand45-40000-s45-K1_3718008.out
?? worklog/worklog_yixun/exp_15_yaw_aug_claude/slurm_screen_exp15-screen-VANL-rrob-rotrand45-40000-s45-K8_3718065.out
?? worklog/worklog_yixun/exp_15_yaw_aug_claude/slurm_screen_exp15-screen-VANL-rrob-rotrand46-40000-s46-K1_3718009.out
?? worklog/worklog_yixun/exp_15_yaw_aug_claude/slurm_screen_exp15-screen-VANL-rrob-rotrand46-40000-s46-K8_3718066.out
?? worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-18_10-08-15_VANL_rrob_rotrand43_S40000_s43_K1_jid3718006_screen.log
?? worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-18_10-12-15_VANL_rrob_rotrand44_S40000_s44_K1_jid3718007_screen.log
?? worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-18_10-12-15_VANL_rrob_rotrand45_S40000_s45_K1_jid3718008_screen.log
?? worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-18_10-17-15_VANL_rrob_rotrand44_S40000_s44_K8_jid3718064_screen.log
?? worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-18_10-17-16_VANL_rrob_rotrand42_S40000_s42_K8_jid3718010_screen.log
?? worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-18_10-17-16_VANL_rrob_rotrand43_S40000_s43_K8_jid3718011_screen.log
?? worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-18_10-17-16_VANL_rrob_rotrand46_S40000_s46_K1_jid3718009_screen.log
?? worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-18_10-18-16_VANL_rrob_rotrand45_S40000_s45_K8_jid3718065_screen.log
?? worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-18_10-24-16_VANL_rrob_rotrand46_S40000_s46_K8_jid3718066_screen.log
?? worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_collect_stderr.log
?? worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_results.md
diff --git a/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submit.sh b/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submit.sh
index 81a3363..efb7072 100644
--- a/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submit.sh
+++ b/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submit.sh
@@ -60,6 +60,19 @@ case "$EXPECTED_STEP" in ''|*[!0-9]*) echo "--expected-step must be a non-negati
 # A chunk is meaningful only for a leg that resumes: it narrows where THIS job
 # stops, never what the campaign may reach (the budget pin stays 100000). It must
 # land on a saved checkpoint, or the next chunk has nothing to resume from.
+# --- node exclusion (round 6, 2026-08-30) -------------------------------------
+# Chunked legs re-roll the node lottery ~24x per arm, and the ECC-flaky nodes
+# (CLAUDE.md, Aug 2026) burned 4 legs in 3 days: two on neu306, two on neu301
+# (NCCL watchdog hang), freezing every remaining arm. The sick list is the
+# DEFAULT; EXCLUDE= overrides it (empty EXCLUDE="" disables exclusion, e.g. if
+# the pool shrinks). Same nodelist shape check as the screen submitter.
+EXCLUDE="${EXCLUDE-neu301,neu303,neu305,neu306,neu317,neu319,neu322,neu332}"
+if [ -n "$EXCLUDE" ]; then
+  case "$EXCLUDE" in
+    *[!a-z0-9,]*) echo "EXCLUDE='${EXCLUDE}' is not a comma-separated node list - abort"; exit 2 ;;
+  esac
+fi
+
 CHUNK_BUDGET="$(awk -F= '/^PINNED_MAXSTEPS=/{split($2,a," "); print a[1]; exit}' "$SBATCH_FILE")"
 case "$CHUNK_BUDGET" in ''|*[!0-9]*) echo "could not read PINNED_MAXSTEPS from ${SBATCH_FILE} - abort"; exit 3;; esac
 if [ -n "$CHUNK_END" ]; then
@@ -181,6 +194,7 @@ ARGS=(
   --time="$TIME_LIMIT"
   --export="ALL,ARM=${ARM},EXPECT_SHA=${SHA},OUTPUT_ROOT=outputs_FLAC"
 )
+[ -n "$EXCLUDE" ] && ARGS+=(--exclude="$EXCLUDE")
 [ "$SMOKE" = "1" ] && ARGS[5]="${ARGS[5]},SMOKE=1,SMOKE_RUNG=${SMOKE_RUNG},SMOKE_MIN_FREE_MB=${SMOKE_MIN_FREE_MB},SMOKE_MAXSTEPS=${SMOKE_MAXSTEPS:-30},SMOKE_TIME=${TIME_LIMIT}"
 [ -n "$RESUME_CKPT" ] && ARGS[5]="${ARGS[5]},RESUME_CKPT=${RESUME_CKPT},EXPECTED_STEP=${EXPECTED_STEP}"
 [ -n "$CHUNK_END" ] && ARGS[5]="${ARGS[5]},CHUNK_END=${CHUNK_END}"
@@ -211,6 +225,7 @@ TMP="$(mktemp "${MANIFEST}.XXXXXX")" || exit 3
   echo "pins rung=${RUNG} maxsteps=$(pin PINNED_MAXSTEPS) ckpt_every=$(pin PINNED_CHECKPOINT_EVERY) min_free_mb=$(pin PINNED_MIN_FREE_MB) p0_manifest_sha256=$(pin PINNED_P0_MANIFEST_SHA256)"
   echo "resume ${RESUME_CKPT:-<none>} expected_step ${EXPECTED_STEP}"
   echo "chunk_end ${CHUNK_END:-<none>}"
+  echo "exclude ${EXCLUDE:-<none>}"
   echo "sbatch sbatch ${ARGS[*]}"
 } >> "$TMP" || { echo "intent manifest write failed - abort"; exit 3; }
 mv -n "$TMP" "$MANIFEST" || { echo "intent manifest publication failed - abort"; exit 2; }
#!/usr/bin/env bash
# ============================================================================
# fa_orbit_submit.sh — the ONLY sanctioned way to submit an exp_11 arm.
#
# Round-3 review B1: an operator must never hand-assemble --gres/--cpus/--mem/
# --time. Every resource flag is derived here from the pins inside
# fa_orbit_train.sbatch (read out of the script itself, so the two can never
# disagree), and the job is refused unless the tracked tree is clean.
#
#   ./fa_orbit_submit.sh C8
#   ./fa_orbit_submit.sh C8 --resume <ckpt> --expected-step 12500
#   ./fa_orbit_submit.sh C8 --resume <ckpt> --expected-step 40000 --chunk-end 42500
#   SMOKE=1 SMOKE_RUNG=16x4 SMOKE_MIN_FREE_MB=14000 ./fa_orbit_submit.sh C4L
#   DRYRUN=1 ./fa_orbit_submit.sh C8        # print the sbatch line, submit nothing
#
# CHUNKED legs (round 5). The partition never backfills a 34-160 h allocation, so
# a leg may declare --chunk-end <n>: it trains to that boundary, saves, and exits,
# and the next leg resumes from there. A chunk leg is walled by
# PINNED_TIME_LIMIT_CHUNK_<ARM> (hours, not days), which is what makes it
# schedulable. The chain is driven by fa_orbit_chunk_watchdog.sh, which submits
# every chunk through THIS script — never through sbatch directly.
#
# Resources per rung (micro x N = 64): --gres=gpu:l40:N, --cpus-per-task=8+7N,
# --mem=(12N+12)G, --time=<the arm's pinned limit>. Each submission is recorded
# in an atomic, no-clobber manifest next to the launcher.
#
# ANTI-DUPLICATE RESERVATION (round-5 r2 review, blocking 1). Every real
# submission takes an exclusive flock on .submit_<ARM>.lock and re-checks the
# queue for a live exp11-<ARM>-train INSIDE that lock before calling sbatch, so a
# manual invocation and the watchdog cannot both queue the same boundary. See the
# block below for the ordering argument.
# ============================================================================
set -uo pipefail
cd "$(git -C "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" rev-parse --show-toplevel)" || exit 3

EXPDIR="worklog/worklog_yixun/exp_11_fa_orbit_claude"
SBATCH_FILE="${EXPDIR}/fa_orbit_train.sbatch"
DRYRUN="${DRYRUN:-0}"
SMOKE="${SMOKE:-0}"
PLACEHOLDER="TO-PIN-AFTER-P0"

[ -f "$SBATCH_FILE" ] || { echo "missing ${SBATCH_FILE} - abort"; exit 3; }

ARM="${1:-}"
[ -n "$ARM" ] || { echo "usage: $0 <C4L|C8|C16|C32|VANL> [--resume <ckpt> --expected-step <n> [--chunk-end <n>]] - abort"; exit 2; }
shift
case "$ARM" in C4L|C8|C16|C32|VANL) ;; *) echo "ARM '${ARM}' must be C4L|C8|C16|C32|VANL - abort"; exit 2;; esac

RESUME_CKPT=""; EXPECTED_STEP=0; CHUNK_END=""
while [ $# -gt 0 ]; do
  case "$1" in
    --resume) RESUME_CKPT="${2:?--resume needs a path}"; shift 2 ;;
    --expected-step) EXPECTED_STEP="${2:?--expected-step needs a number}"; shift 2 ;;
    --chunk-end) CHUNK_END="${2:?--chunk-end needs a number}"; shift 2 ;;
    *) echo "unknown argument '$1' - abort"; exit 2 ;;
  esac
done
case "$EXPECTED_STEP" in ''|*[!0-9]*) echo "--expected-step must be a non-negative integer - abort"; exit 2;; esac
# --- --chunk-end: shape-checked HERE, and again inside the job ---------------
# A chunk is meaningful only for a leg that resumes: it narrows where THIS job
# stops, never what the campaign may reach (the budget pin stays 100000). It must
# land on a saved checkpoint, or the next chunk has nothing to resume from.
# --- node exclusion (round 6, 2026-08-30) -------------------------------------
# Chunked legs re-roll the node lottery ~24x per arm, and the ECC-flaky nodes
# (CLAUDE.md, Aug 2026) burned 4 legs in 3 days: two on neu306, two on neu301
# (NCCL watchdog hang), freezing every remaining arm. The sick list is the
# DEFAULT; EXCLUDE= overrides it (empty EXCLUDE="" disables exclusion, e.g. if
# the pool shrinks). Same nodelist shape check as the screen submitter.
EXCLUDE="${EXCLUDE-neu301,neu303,neu305,neu306,neu317,neu319,neu322,neu332}"
if [ -n "$EXCLUDE" ]; then
  case "$EXCLUDE" in
    *[!a-z0-9,]*) echo "EXCLUDE='${EXCLUDE}' is not a comma-separated node list - abort"; exit 2 ;;
  esac
fi

CHUNK_BUDGET="$(awk -F= '/^PINNED_MAXSTEPS=/{split($2,a," "); print a[1]; exit}' "$SBATCH_FILE")"
case "$CHUNK_BUDGET" in ''|*[!0-9]*) echo "could not read PINNED_MAXSTEPS from ${SBATCH_FILE} - abort"; exit 3;; esac
if [ -n "$CHUNK_END" ]; then
  case "$CHUNK_END" in ''|*[!0-9]*) echo "--chunk-end must be a positive integer - abort"; exit 2;; esac
  [ "$SMOKE" != "1" ] || { echo "--chunk-end is a production chunk-chain input and has no meaning under SMOKE=1 - abort"; exit 2; }
  { [ -n "$RESUME_CKPT" ] && [ "$EXPECTED_STEP" -gt 0 ]; } \
    || { echo "--chunk-end is valid only together with --resume/--expected-step - abort"; exit 2; }
  [ "$((CHUNK_END % 2500))" -eq 0 ] || { echo "--chunk-end ${CHUNK_END} is not a multiple of 2500 (the pinned checkpoint cadence) - abort"; exit 2; }
  [ "$CHUNK_END" -gt "$EXPECTED_STEP" ] || { echo "--chunk-end ${CHUNK_END} must exceed --expected-step ${EXPECTED_STEP} - abort"; exit 2; }
  [ "$CHUNK_END" -le "$CHUNK_BUDGET" ] || { echo "--chunk-end ${CHUNK_END} exceeds the pinned budget ${CHUNK_BUDGET} - abort"; exit 2; }
fi

# --- pins are read FROM the launcher, so submitter and job cannot disagree ----
pin() {  # read one PINNED_* value out of the launcher (quoted or bare)
  awk -v k="$1" '$0 ~ "^"k"=" {
        if (match($0, /"[^"]*"/)) { print substr($0, RSTART + 1, RLENGTH - 2) }
        else { split($0, a, "="); split(a[2], b, " "); print b[1] }
        exit }' "$SBATCH_FILE"
}
if [ "$SMOKE" = "1" ]; then
  RUNG="${SMOKE_RUNG:?SMOKE=1 requires SMOKE_RUNG (32x2|16x4|8x8)}"
  TIME_LIMIT="${SMOKE_TIME:-00:30:00}"
  [ -n "${SMOKE_MIN_FREE_MB:-}" ] || { echo "SMOKE=1 requires SMOKE_MIN_FREE_MB - abort"; exit 2; }
  JOBNAME="exp11-smoke-${ARM}"
else
  RUNG="$(pin PINNED_RUNG)"
  # A RESTART leg is a different budget from the INITIAL one: 60k further steps,
  # not 40k from scratch. Selecting the INITIAL limit for a restart would wall-kill
  # every arm partway through the extension.
  # A CHUNK leg is shorter still: it stops at the next boundary, so it is walled
  # by the arm's CHUNK pin (hours) rather than its whole-extension RESTART pin
  # (days) — the whole point of chunking is an allocation the scheduler backfills.
  if [ -n "$CHUNK_END" ]; then
    TIME_LIMIT="$(pin "PINNED_TIME_LIMIT_CHUNK_${ARM}")"
  elif [ -n "${EXPECTED_STEP:-}" ] && [ "${EXPECTED_STEP:-0}" -gt 0 ]; then
    TIME_LIMIT="$(pin "PINNED_TIME_LIMIT_RESTART_${ARM}")"
  else
    TIME_LIMIT="$(pin "PINNED_TIME_LIMIT_${ARM}")"
  fi
  [ -n "$TIME_LIMIT" ] || { echo "the launcher carries no wall pin for this ${ARM} leg - abort"; exit 2; }
  for V in "$RUNG" "$TIME_LIMIT" "$(pin PINNED_MIN_FREE_MB)" "$(pin PINNED_P0_MANIFEST_SHA256)"; do
    [ "$V" != "$PLACEHOLDER" ] || { echo "the launcher still carries ${PLACEHOLDER} pins: the P0 report has not been pinned yet — no arm may be submitted (use SMOKE=1 for the smoke) - abort"; exit 2; }
  done
  JOBNAME="exp11-${ARM}-train"
fi
case "$RUNG" in 32x2|16x4|8x8) ;; *) echo "rung '${RUNG}' must be 32x2|16x4|8x8 - abort"; exit 2;; esac
MB="${RUNG%x*}"; NGPU="${RUNG#*x}"
[ "$((MB * NGPU))" -eq 64 ] || { echo "rung ${RUNG}: MB*NGPU != 64 - abort"; exit 2; }

# --- SUBMISSION RESERVATION (round-5 r2 review, blocking 1) -------------------
# The chunk watchdog's singleton lock only prevents a second WATCHDOG. A human
# running this script by hand could still slip between the watchdog's queue check
# and its sbatch and queue the same boundary twice, and the job-side run-directory
# flock cannot help: it is taken long after scheduling, so it serialises execution
# while still burning a second allocation. The reservation therefore lives HERE,
# in the ONE sanctioned submitter that every path — watchdog or human — goes
# through:
#
#     flock .submit_<ARM>.lock  ->  squeue -n exp11-<ARM>-train  ->  sbatch
#
# The queue check is INSIDE the lock, so no other submitter can observe an empty
# queue and sbatch between our check and ours. Both directions are fail-CLOSED: a
# held lock refuses, and a squeue that exits NONZERO refuses too — an unreadable
# queue is never read as an empty one. The lock is released by the kernel when
# this process exits, on every path (refusal, sbatch failure, success).
#
# It is placed BEFORE the code-drift gate deliberately: a duplicate submission
# must be refused as a duplicate, not masked by whichever gate happens to fire
# first, and holding the reservation across the (cheap, local) drift check costs
# nothing. SMOKE and DRYRUN are excluded by design — a smoke leg carries its own
# job name and identity, and a dry run submits nothing at all.
if [ "$SMOKE" != "1" ] && [ "$DRYRUN" != "1" ]; then
  SUBMIT_LOCK="${EXPDIR}/.submit_${ARM}.lock"
  WHO="${USER:-$(id -un)}"
  exec 9>"$SUBMIT_LOCK" || { echo "could not open the ${ARM} submission lock ${SUBMIT_LOCK} - abort"; exit 3; }
  if ! flock -n 9; then
    echo "another submission for ${ARM} already holds ${SUBMIT_LOCK} — a leg for this arm is already being submitted - abort"
    exit 2
  fi
  # stderr is captured SEPARATELY on purpose: folded into stdout, a harmless
  # scheduler warning would read as a live job and refuse a legitimate leg.
  SQ_ERR="$(mktemp "${TMPDIR:-/tmp}/exp11_squeue.XXXXXX")" || { echo "could not create a temp file for the queue check - abort"; exit 3; }
  LIVE="$(squeue -h -u "$WHO" -n "$JOBNAME" -o '%i %T' 2>"$SQ_ERR")"; QRC=$?
  SQ_MSG="$(head -3 "$SQ_ERR" 2>/dev/null | tr '\n' ' ')"; rm -f "$SQ_ERR"
  if [ "$QRC" -ne 0 ]; then
    echo "squeue exited ${QRC} (${SQ_MSG:-no message}) — the queue state is UNKNOWN and an unreadable queue is never read as an empty one - abort"
    exit 2
  fi
  if [ -n "$LIVE" ]; then
    echo "a leg for this arm is already queued/running as ${JOBNAME} (${LIVE//$'\n'/; }) - abort"
    exit 2
  fi
fi

# --- drift gate: a queued job must run reviewed, committed code --------------
# The drift gate is scoped to CODE surfaces, not the whole exp folder: the four
# arms are running and Slurm appends to their tracked *.out logs continuously, so
# a folder-wide check would abort every screen on a live-log write. Configs,
# drivers and validators are still fully covered.
#
# arm_launch_registry.json is EXCLUDED (round-5 review B1). It is a lineage
# RECORD written by the reviewed recorder as each chunk finishes — a measurement
# *product*, not a measurement *surface* — and it is already outside the
# launcher's commit-binding closure. Gating it deadlocks the chunk chain: the
# recorder dirties it at 42500, so the very next submission (and every later
# one, for every arm) is refused until a human commits. Keep it outside.
DRIFT="$(git status --porcelain --untracked-files=no -- train.py defaults.ini src \
         "$EXPDIR"/*.json "$EXPDIR"/*.py "$EXPDIR"/*.sbatch "$EXPDIR"/*.sh \
         ":(exclude)${EXPDIR}/arm_launch_registry.json" \
         worklog/worklog_yixun/exp_07_fa_scratch_claude/FLAC_AR_BF.json 2>/dev/null)"
[ -z "$DRIFT" ] || { echo "tracked measurement surfaces have uncommitted changes - commit first, abort:"; echo "$DRIFT"; exit 2; }
SHA="$(git rev-parse HEAD)"

ARGS=(
  --job-name="$JOBNAME"
  --gres="gpu:l40:${NGPU}"
  --cpus-per-task="$((8 + 7 * NGPU))"
  --mem="$(((12 * NGPU + 12)))G"
  --time="$TIME_LIMIT"
  --export="ALL,ARM=${ARM},EXPECT_SHA=${SHA},OUTPUT_ROOT=outputs_FLAC"
)
[ -n "$EXCLUDE" ] && ARGS+=(--exclude="$EXCLUDE")
[ "$SMOKE" = "1" ] && ARGS[5]="${ARGS[5]},SMOKE=1,SMOKE_RUNG=${SMOKE_RUNG},SMOKE_MIN_FREE_MB=${SMOKE_MIN_FREE_MB},SMOKE_MAXSTEPS=${SMOKE_MAXSTEPS:-30},SMOKE_TIME=${TIME_LIMIT}"
[ -n "$RESUME_CKPT" ] && ARGS[5]="${ARGS[5]},RESUME_CKPT=${RESUME_CKPT},EXPECTED_STEP=${EXPECTED_STEP}"
[ -n "$CHUNK_END" ] && ARGS[5]="${ARGS[5]},CHUNK_END=${CHUNK_END}"
ARGS+=("$SBATCH_FILE")

echo "arm ${ARM} | rung ${RUNG} (${MB}x${NGPU}) | time ${TIME_LIMIT} | commit ${SHA} | smoke ${SMOKE} | chunk_end ${CHUNK_END:-<none>}"
if [ "$DRYRUN" = "1" ]; then
  echo "DRYRUN sbatch ${ARGS[*]}"
  exit 0
fi

# --- NEW-3: publish the INTENT before submitting -----------------------------
# The provenance record must exist before the job can exist, otherwise a local
# write failure leaves a queued job nobody recorded. The intent manifest carries
# the exact command and pins; the job id is appended afterwards, and if that
# append fails the exact job we just created is cancelled.
INTENT_ID="$(date +%s%N)-$(cut -c1-8 /proc/sys/kernel/random/uuid)"
MANIFEST="${EXPDIR}/fa_orbit_submission_${ARM}_${INTENT_ID}.txt"
[ ! -e "$MANIFEST" ] || { echo "submission manifest ${MANIFEST} already exists - abort"; exit 2; }
TMP="$(mktemp "${MANIFEST}.XXXXXX")" || exit 3
{
  echo "# exp_11 arm submission (intent published BEFORE sbatch)"
  echo "intent_id ${INTENT_ID}"
  echo "submitted_at $(date -Is)"
  echo "arm ${ARM} rung ${RUNG} micro ${MB} ngpu ${NGPU}"
  echo "jobname ${JOBNAME} time ${TIME_LIMIT} smoke ${SMOKE}"
  echo "commit ${SHA}"
  echo "pins rung=${RUNG} maxsteps=$(pin PINNED_MAXSTEPS) ckpt_every=$(pin PINNED_CHECKPOINT_EVERY) min_free_mb=$(pin PINNED_MIN_FREE_MB) p0_manifest_sha256=$(pin PINNED_P0_MANIFEST_SHA256)"
  echo "resume ${RESUME_CKPT:-<none>} expected_step ${EXPECTED_STEP}"
  echo "chunk_end ${CHUNK_END:-<none>}"
  echo "exclude ${EXCLUDE:-<none>}"
  echo "sbatch sbatch ${ARGS[*]}"
} >> "$TMP" || { echo "intent manifest write failed - abort"; exit 3; }
mv -n "$TMP" "$MANIFEST" || { echo "intent manifest publication failed - abort"; exit 2; }
[ -e "$MANIFEST" ] || { echo "intent manifest ${MANIFEST} did not appear - abort"; exit 2; }
echo "intent manifest: ${MANIFEST}"

OUT="$(sbatch "${ARGS[@]}" 2>&1)"; JID="$(echo "$OUT" | awk '/Submitted batch job/ {print $NF}')"
if [ -z "$JID" ]; then
  echo "SUBMIT FAILED: ${OUT}"
  echo "submit_failed $(date -Is)" >> "$MANIFEST"
  exit 1
fi
echo "submitted ${ARM} -> job ${JID}"
if ! echo "jobid ${JID}" >> "$MANIFEST"; then
  echo "could not append job id ${JID} to ${MANIFEST} — cancelling the job rather than leave it unrecorded"
  scancel "$JID" || echo "scancel ${JID} FAILED — cancel it by hand NOW"
  exit 2
fi
echo "submission recorded: ${MANIFEST} (job ${JID})"
/usr/bin/bash: line 1: rg: command not found

exec
/usr/bin/bash -lc "git diff --no-ext-diff -- worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submit.sh worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh; grep -nE 'neu301|neu303|neu305|neu306|neu317|neu319|neu322|neu332|sick|exclude' CLAUDE.md 2>/dev/null; grep -nE 'submit|EXCLUDE|exclude|dirty|DRYRUN|squeue|flock' worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh" in /n/fs/gatrdp/codespace/FLAC
 succeeded in 0ms:
diff --git a/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submit.sh b/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submit.sh
index 81a3363..efb7072 100644
--- a/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submit.sh
+++ b/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submit.sh
@@ -60,6 +60,19 @@ case "$EXPECTED_STEP" in ''|*[!0-9]*) echo "--expected-step must be a non-negati
 # A chunk is meaningful only for a leg that resumes: it narrows where THIS job
 # stops, never what the campaign may reach (the budget pin stays 100000). It must
 # land on a saved checkpoint, or the next chunk has nothing to resume from.
+# --- node exclusion (round 6, 2026-08-30) -------------------------------------
+# Chunked legs re-roll the node lottery ~24x per arm, and the ECC-flaky nodes
+# (CLAUDE.md, Aug 2026) burned 4 legs in 3 days: two on neu306, two on neu301
+# (NCCL watchdog hang), freezing every remaining arm. The sick list is the
+# DEFAULT; EXCLUDE= overrides it (empty EXCLUDE="" disables exclusion, e.g. if
+# the pool shrinks). Same nodelist shape check as the screen submitter.
+EXCLUDE="${EXCLUDE-neu301,neu303,neu305,neu306,neu317,neu319,neu322,neu332}"
+if [ -n "$EXCLUDE" ]; then
+  case "$EXCLUDE" in
+    *[!a-z0-9,]*) echo "EXCLUDE='${EXCLUDE}' is not a comma-separated node list - abort"; exit 2 ;;
+  esac
+fi
+
 CHUNK_BUDGET="$(awk -F= '/^PINNED_MAXSTEPS=/{split($2,a," "); print a[1]; exit}' "$SBATCH_FILE")"
 case "$CHUNK_BUDGET" in ''|*[!0-9]*) echo "could not read PINNED_MAXSTEPS from ${SBATCH_FILE} - abort"; exit 3;; esac
 if [ -n "$CHUNK_END" ]; then
@@ -181,6 +194,7 @@ ARGS=(
   --time="$TIME_LIMIT"
   --export="ALL,ARM=${ARM},EXPECT_SHA=${SHA},OUTPUT_ROOT=outputs_FLAC"
 )
+[ -n "$EXCLUDE" ] && ARGS+=(--exclude="$EXCLUDE")
 [ "$SMOKE" = "1" ] && ARGS[5]="${ARGS[5]},SMOKE=1,SMOKE_RUNG=${SMOKE_RUNG},SMOKE_MIN_FREE_MB=${SMOKE_MIN_FREE_MB},SMOKE_MAXSTEPS=${SMOKE_MAXSTEPS:-30},SMOKE_TIME=${TIME_LIMIT}"
 [ -n "$RESUME_CKPT" ] && ARGS[5]="${ARGS[5]},RESUME_CKPT=${RESUME_CKPT},EXPECTED_STEP=${EXPECTED_STEP}"
 [ -n "$CHUNK_END" ] && ARGS[5]="${ARGS[5]},CHUNK_END=${CHUNK_END}"
@@ -211,6 +225,7 @@ TMP="$(mktemp "${MANIFEST}.XXXXXX")" || exit 3
   echo "pins rung=${RUNG} maxsteps=$(pin PINNED_MAXSTEPS) ckpt_every=$(pin PINNED_CHECKPOINT_EVERY) min_free_mb=$(pin PINNED_MIN_FREE_MB) p0_manifest_sha256=$(pin PINNED_P0_MANIFEST_SHA256)"
   echo "resume ${RESUME_CKPT:-<none>} expected_step ${EXPECTED_STEP}"
   echo "chunk_end ${CHUNK_END:-<none>}"
+  echo "exclude ${EXCLUDE:-<none>}"
   echo "sbatch sbatch ${ARGS[*]}"
 } >> "$TMP" || { echo "intent manifest write failed - abort"; exit 3; }
 mv -n "$TMP" "$MANIFEST" || { echo "intent manifest publication failed - abort"; exit 2; }
diff --git a/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh b/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh
index c64bfa6..9c1a4e0 100755
--- a/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh
+++ b/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh
@@ -984,6 +984,22 @@ else
   echo "FAIL  no post-run wandb identity verification"; FAIL=$((FAIL+1))
 fi
 
+echo "--- M. round 6: sick-node exclusion in the training submitter ---"
+# 4 chunk legs burned on neu301/neu306 (ECC list) in 3 days; the training
+# submitter now defaults --exclude to the sick list. DRYRUN prints the line.
+expect_cmd "submitter defaults --exclude to the sick list" 0 "--exclude=neu301,neu303,neu305,neu306,neu317,neu319,neu322,neu332" -- \
+  env DRYRUN=1 bash "$SUBMITTER" C8 --resume x.ckpt --expected-step 40000 --chunk-end 42500
+expect_cmd "EXCLUDE override replaces the default" 0 "--exclude=neu399" -- \
+  env DRYRUN=1 EXCLUDE=neu399 bash "$SUBMITTER" C8 --resume x.ckpt --expected-step 40000 --chunk-end 42500
+out="$(env DRYRUN=1 EXCLUDE= bash "$SUBMITTER" C8 --resume x.ckpt --expected-step 40000 --chunk-end 42500 2>&1)"
+if echo "$out" | grep -q -- "--exclude"; then
+  echo "FAIL  EXCLUDE=\"\" should disable exclusion"; FAIL=$((FAIL+1))
+else
+  echo "PASS  EXCLUDE=\"\" disables exclusion"; PASS=$((PASS+1))
+fi
+expect_cmd "submitter rejects an unsafe EXCLUDE" 2 "not a comma-separated node list" -- \
+  env DRYRUN=1 "EXCLUDE=neu301;rm -rf x" bash "$SUBMITTER" C8 --resume x.ckpt --expected-step 40000 --chunk-end 42500
+
 echo
 echo "=== guard tests: ${PASS} passed, ${FAIL} failed ==="
 [ "$FAIL" -eq 0 ] || exit 1
49:**Shared machine, concurrent writers.** A second Claude session works this same repo and branch (`check-equivariance-necessity`) from another machine and pushes frequently — **always `git pull --rebase` before committing**, and never rewrite files another session owns. ⚠️ **Queued Slurm jobs bind to commits**: exp_11 training legs verify their `EXPECT_SHA` at start via the *content-scoped* gate in `fa_orbit_train.sbatch` (Codex-approved, `da7ee7f`) — record/worklog commits are safe while legs pend, but commits touching the training closure (`train.py`, `defaults.ini`, `src/`, `data/AR`, arm configs, the launcher + its helpers) still abort them fail-closed. Never revert that gate to HEAD-identity. On neuronic, sbatch has **no `SBATCH_EXCLUDE` env var** — node exclusion works only as an explicit `--exclude`/`EXCLUDE=` argument (screen submitter only; the training submitter has no exclusion path). ECC-flaky nodes (Aug 2026): neu301/303/305/306/317/319/322/332. On the A6000 box, Yixun also runs experiments from *sibling checkouts* (`~/codespace/exp-12-arms`, `exp-08-cylvit-pe-cnn`, `exp-09-cyl-dinov3-no-ssl`, `exp-10-cyl-distill`) whose jobs share those GPUs and use their own experiment numbering. **Before assuming any `train.py` process belongs to this worktree, check `readlink /proc/<pid>/cwd`; never kill or edit a run you did not launch.**
146:Cross-experiment results live in `worklog/worklog_yixun/model_comparison.md`, regenerated **only** by `worklog/worklog_yixun/gen_model_comparison.py` (rows are glob specs aggregated from raw per-seed metric JSONs; single-seed screens are structurally excluded). Per announcement 04, regenerate + commit + push on every model-results update.
12:#   * it submits nothing and touches no GPU.
15:#   DRYRUN=1        every cheap gate (pins, arm, rung, config map, semantic
32:SUBMITTER="${EXPDIR}/fa_orbit_submit.sh"
78:SMOKE_ENV=(DRYRUN=1 "EXPECT_SHA=${HEAD_SHA}" "OUTPUT_ROOT=${OUT_ROOT}" "${REPO_ENV[@]}" SMOKE=1 SMOKE_RUNG=16x4 SMOKE_MIN_FREE_MB=14000)
98:  -- DRYRUN=1 SMOKE=1 ARM=C8 "EXPECT_SHA=${HEAD_SHA}" "OUTPUT_ROOT=${OUT_ROOT}" "${REPO_ENV[@]}"
100:  -- DRYRUN=1 SMOKE=1 SMOKE_RUNG=16x4 ARM=C8 "EXPECT_SHA=${HEAD_SHA}" "OUTPUT_ROOT=${OUT_ROOT}" "${REPO_ENV[@]}"
104:case_run "missing ARM" 2 "ARM" -- DRYRUN=1 "EXPECT_SHA=${HEAD_SHA}" "OUTPUT_ROOT=${OUT_ROOT}" "${REPO_ENV[@]}"
105:case_run "missing EXPECT_SHA" 2 "EXPECT_SHA" -- DRYRUN=1 ARM=C8 "OUTPUT_ROOT=${OUT_ROOT}" "${REPO_ENV[@]}"
110:  -- DRYRUN=1 SMOKE=1 SMOKE_RUNG=64x1 SMOKE_MIN_FREE_MB=1 ARM=C8 "EXPECT_SHA=${HEAD_SHA}" "OUTPUT_ROOT=${OUT_ROOT}" "${REPO_ENV[@]}"
113:    -- DRYRUN=1 SMOKE=1 "SMOKE_RUNG=${R}" SMOKE_MIN_FREE_MB=14000 ARM=C4L "EXPECT_SHA=${HEAD_SHA}" "OUTPUT_ROOT=${OUT_ROOT}" "${REPO_ENV[@]}"
156:# (dirty-tree drift today, run-dir/allocation gates on a clean tree) with
181:# src/tests is excluded from the closure (pytest-only; TDD sessions land tests
308:# The submitter allocated 34/51/89 h for the restart legs, but the job selected
314:Q10_ENV=(DRYRUN=1 "EXPECT_SHA=${HEAD_SHA}" "OUTPUT_ROOT=${OUT_ROOT}" "${REPO_ENV[@]}")
325:# submitter and job must pick the same pin for the same leg
326:SUB_RESTART="$(env DRYRUN=1 bash "$SUBMITTER" C16 --resume "${Q10_RUN}/checkpoints/epoch=8-step=40000.ckpt" --expected-step 40000 2>&1)"
328:  echo "PASS  submitter and job agree on the C16 RESTART pin"; PASS=$((PASS+1))
330:  echo "FAIL  the submitter no longer allocates the C16 RESTART pin"; FAIL=$((FAIL+1))
493:# The new gates talk to the SCHEDULER: the submitter re-checks `squeue` INSIDE
502:echo "FAKE-SBATCH: guard test — nothing was submitted"
505:cat > "${SHIM}/squeue" <<'EOF'
508:[ -s "${D}/squeue_out" ] && cat "${D}/squeue_out"
509:exit "$(cat "${D}/squeue_rc" 2>/dev/null || echo 0)"
515:chmod +x "${SHIM}/sbatch" "${SHIM}/squeue" "${SHIM}"/sacct_*
516:: > "${SHIM}/squeue_out"; echo 0 > "${SHIM}/squeue_rc"
735:# launcher-side CHUNK_END gates (parameter gates run under DRYRUN)
737:  -- DRYRUN=1 ARM=C4L "EXPECT_SHA=${HEAD_SHA}" "OUTPUT_ROOT=${OUT_ROOT}" "${REPO_ENV[@]}" CHUNK_END=42500
740:# submitter-side --chunk-end shape gates (refused before any pin/sbatch work)
741:expect_cmd "submitter: --chunk-end without --resume refused" 2 "valid only together" -- \
742:  env DRYRUN=1 bash "$SUBMITTER" C8 --chunk-end 42500
743:expect_cmd "submitter: --chunk-end off cadence refused" 2 "not a multiple of 2500" -- \
744:  env DRYRUN=1 bash "$SUBMITTER" C8 --resume x.ckpt --expected-step 40000 --chunk-end 42600
745:expect_cmd "submitter: --chunk-end must exceed the resume step" 2 "must exceed" -- \
746:  env DRYRUN=1 bash "$SUBMITTER" C8 --resume x.ckpt --expected-step 42500 --chunk-end 42500
747:expect_cmd "submitter: --chunk-end above the budget refused" 2 "exceeds the pinned budget" -- \
748:  env DRYRUN=1 bash "$SUBMITTER" C8 --resume x.ckpt --expected-step 40000 --chunk-end 102500
749:# --- round-5 r2 blocking 1: the SUBMISSION RESERVATION lives in the submitter -
751:# The reservation therefore sits in the one sanctioned submitter: an exclusive
752:# per-arm flock, and a queue re-check INSIDE it. These cases run with a FAKE
754:SUB_LOCK="${EXPDIR}/.submit_C8.lock"
755:SUB_RELEASE="${TMP}/release_the_submit_lock"
756:( flock -n 9 || exit 1; while [ ! -e "$SUB_RELEASE" ]; do sleep 0.2; done ) 9>"$SUB_LOCK" &
759:expect_cmd "submitter refuses while another submission holds the arm's lock" 2 "already holds" -- \
764:echo 1 > "${SHIM}/squeue_rc"; : > "${SHIM}/squeue_out"
765:expect_cmd "submitter refuses when squeue itself fails" 2 "the queue state is UNKNOWN" -- \
768:echo 0 > "${SHIM}/squeue_rc"; echo "4242424 RUNNING" > "${SHIM}/squeue_out"
769:expect_cmd "submitter refuses when a leg for the arm is already queued/running" 2 "already queued/running" -- \
771:echo 0 > "${SHIM}/squeue_rc"; : > "${SHIM}/squeue_out"
772:awk '/flock -n 9/{if (!f) f=NR} /^  LIVE=.*squeue/{q=NR} /^OUT="\$\(sbatch/{s=NR}
774:  && { echo "PASS  the submitter checks the queue INSIDE the reservation lock, before sbatch"; PASS=$((PASS+1)); } \
775:  || { echo "FAIL  the submitter's queue check is not sequenced flock -> squeue -> sbatch"; FAIL=$((FAIL+1)); }
798:# both submit the same boundary). Rather than race two watchdogs, hold the real
799:# lock with a background flock holder and prove the watchdog refuses to start.
802:( flock -n 9 || exit 1; while [ ! -e "$WD_RELEASE" ]; do sleep 0.2; done ) 9>"$WD_LOCK" &
806:  bash "$WD" ONESHOT=1 DRYRUN=1 ARMS=C8 "OUTPUT_ROOT=${WD_TMP}/outputs" \
810:if flock -n 9 9>"$WD_LOCK" 2>/dev/null; then
816:# FROZEN at startup — not submitted and then refused by the preflight. Anchoring
824:expect_cmd "watchdog freezes an arm with no audited anchor before submitting" 0 "no audited anchor" -- \
825:  bash "$WD" ONESHOT=1 DRYRUN=1 ARMS=C8 "OUTPUT_ROOT=${WD_TMP}/outputs" \
845:  bash "$WD" ONESHOT=1 DRYRUN=1 ARMS=C8 "OUTPUT_ROOT=${WD_TMP}/outputs" \
849:  bash "$WD" ONESHOT=1 DRYRUN=1 ARMS=C8 "OUTPUT_ROOT=${WD_TMP}/outputs" \
853:  bash "$WD" ONESHOT=1 DRYRUN=1 ARMS=C8 "OUTPUT_ROOT=${WD_TMP}/outputs" \
856:# NOTE: this case runs against the REAL queue (no squeue fake): with a live
859:# a full ONESHOT pass must complete cleanly without submitting anything.
860:expect_cmd "watchdog completes a ONESHOT pass without submitting" 0 "ONESHOT: one pass complete" -- \
861:  bash "$WD" ONESHOT=1 DRYRUN=1 ARMS=C8 "OUTPUT_ROOT=${WD_TMP}/outputs" \
864:  bash "$WD" ONESHOT=1 DRYRUN=1 ARMS=C8 "OUTPUT_ROOT=${WD_TMP}/outputs" \
870:# PATH (squeue: empty queue; sacct: FAILED for the remembered job); DRYRUN keeps
871:# the submitter from queueing anything real.
873:printf '#!/bin/sh\nexit 0\n' > "${WD_TMP}/bin/squeue"
875:chmod +x "${WD_TMP}/bin/squeue" "${WD_TMP}/bin/sacct"
884:  env PATH="${WD_TMP}/bin:$PATH" bash "$WD" ONESHOT=1 DRYRUN=1 ARMS=C8 \
889:# environment-dependent here: in a dirty tree the DRYRUN submit itself is
890:# refused by the submitter's clean-tree guard and adds one legitimate bump.
897:echo "--- H. the submitter refuses un-pinned submission ---"
899:# so the submitter's placeholder refusal is unreachable on the real file.
901:  echo "FAIL  a submitter pin is still the placeholder"; FAIL=$((FAIL+1))
903:  echo "PASS  every submitter pin holds a concrete value"; PASS=$((PASS+1))
906:  && { echo "PASS  the submitter still refuses a placeholder pin if one returns"; PASS=$((PASS+1)); } \
907:  || { echo "FAIL  the submitter placeholder refusal is gone"; FAIL=$((FAIL+1)); }
908:expect_cmd "submitter rejects a bad arm" 2 "must be C4L" -- env DRYRUN=1 bash "$SUBMITTER" FA1
909:expect_cmd "submitter derives smoke flags" 0 "--gres=gpu:l40:4" -- \
910:  env DRYRUN=1 SMOKE=1 SMOKE_RUNG=16x4 SMOKE_MIN_FREE_MB=14000 bash "$SUBMITTER" C4L
911:expect_cmd "submitter derives cpus/mem from the rung" 0 "--cpus-per-task=36" -- \
912:  env DRYRUN=1 SMOKE=1 SMOKE_RUNG=16x4 SMOKE_MIN_FREE_MB=14000 bash "$SUBMITTER" C4L
913:expect_cmd "submitter derives 8x8 resources" 0 "--mem=108G" -- \
914:  env DRYRUN=1 SMOKE=1 SMOKE_RUNG=8x8 SMOKE_MIN_FREE_MB=14000 bash "$SUBMITTER" C4L
916:echo "--- I. flock run ownership, both contention directions (round-3 B3 residual) ---"
921:( flock -n 9 || exit 1; read -r _ < "$FIFO_HOLD" ) 9>"$LOCKF" &
924:if flock -n 9 9>"$LOCKF" 2>/dev/null; then
925:  echo "FAIL  a second holder acquired a held flock"; FAIL=$((FAIL+1))
932:if flock -n 9 9>"$LOCKF" 2>/dev/null; then
937:grep -q 'flock -n 9' "$LAUNCHER" && { echo "PASS  launcher uses flock, not mkdir+stale recovery"; PASS=$((PASS+1)); } \
938:  || { echo "FAIL  launcher does not use flock"; FAIL=$((FAIL+1)); }
948:grep -q 'OUTPUT_ROOT=outputs_FLAC' "$SUBMITTER" && { echo "PASS  submitter exports the fixed root, not ambient state"; PASS=$((PASS+1)); } \
949:  || { echo "FAIL  submitter still forwards an ambient OUTPUT_ROOT"; FAIL=$((FAIL+1)); }
951:echo "--- K. the submitter publishes intent BEFORE sbatch (NEW-3) ---"
953:expect_cmd "dry run publishes no submission manifest" 0 "DRYRUN sbatch" -- \
954:  env DRYRUN=1 SMOKE=1 SMOKE_RUNG=16x4 SMOKE_MIN_FREE_MB=14000 bash "$SUBMITTER" C4L
987:echo "--- M. round 6: sick-node exclusion in the training submitter ---"
989:# submitter now defaults --exclude to the sick list. DRYRUN prints the line.
990:expect_cmd "submitter defaults --exclude to the sick list" 0 "--exclude=neu301,neu303,neu305,neu306,neu317,neu319,neu322,neu332" -- \
991:  env DRYRUN=1 bash "$SUBMITTER" C8 --resume x.ckpt --expected-step 40000 --chunk-end 42500
992:expect_cmd "EXCLUDE override replaces the default" 0 "--exclude=neu399" -- \
993:  env DRYRUN=1 EXCLUDE=neu399 bash "$SUBMITTER" C8 --resume x.ckpt --expected-step 40000 --chunk-end 42500
994:out="$(env DRYRUN=1 EXCLUDE= bash "$SUBMITTER" C8 --resume x.ckpt --expected-step 40000 --chunk-end 42500 2>&1)"
995:if echo "$out" | grep -q -- "--exclude"; then
996:  echo "FAIL  EXCLUDE=\"\" should disable exclusion"; FAIL=$((FAIL+1))
998:  echo "PASS  EXCLUDE=\"\" disables exclusion"; PASS=$((PASS+1))
1000:expect_cmd "submitter rejects an unsafe EXCLUDE" 2 "not a comma-separated node list" -- \
1001:  env DRYRUN=1 "EXCLUDE=neu301;rm -rf x" bash "$SUBMITTER" C8 --resume x.ckpt --expected-step 40000 --chunk-end 42500

exec
/usr/bin/bash -lc "grep -RIn --include='*.sh' --include='*.sbatch' 'EXCLUDE=' worklog/worklog_yixun | head -80; sed -n '1,130p' worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh; sed -n '470,535p' worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh; sed -n '730,790p' worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh; git diff --check -- worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submit.sh worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh; bash -n worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submit.sh; bash -n worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh" in /n/fs/gatrdp/codespace/FLAC
 succeeded in 1098ms:
worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_screen_guardtests.sh:955:  # --- EXCLUDE= reaches sbatch as an explicit FLAG ---------------------------
worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_screen_guardtests.sh:961:             YAW_GEN_TEST_MODE=1 YAW_GEN_TEST_RECORD="$TRACE" bash "$SUB" ARM=C4L CELL=zref STEP=40000 EXCLUDE=neu303,neu332 2>&1)"; rc=$?
worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_screen_guardtests.sh:963:    echo "PASS  EXCLUDE= is passed to sbatch as an explicit --exclude flag"; PASS=$((PASS + 1))
worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_screen_guardtests.sh:965:    echo "FAIL  EXCLUDE= did not reach sbatch (rc=${rc})"; sed 's/^/        | /' "$TRACE"
worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_screen_guardtests.sh:975:    echo "PASS  no EXCLUDE= means no --exclude flag"; PASS=$((PASS + 1))
worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_screen_guardtests.sh:977:    echo "FAIL  an --exclude flag appeared without EXCLUDE="; FAIL=$((FAIL + 1))
worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_screen_guardtests.sh:981:  out="$(env SBATCH_EXCLUDE=neu303 \
worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_screen_guardtests.sh:1026:             "EXCLUDE=neu1;id" "ROTATE_DEG=1.2.3"; do
worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_screen_guardtests.sh:1826:for BADARG in "WAVE=conf" "WAVE=screen" "FOO=1" "PIN_SHA=deadbeef" "EXCLUDE=neu1;id"; do
worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_submit_grid.sh:34:#   bash yaw_gen_submit_grid.sh WAVE=vctl EXCLUDE=neu303,neu332
worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_submit_grid.sh:165:WAVE=""; EXCLUDE=""; PIN_SHA=""
worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_submit_grid.sh:185:    EXCLUDE)  is_nodelist "$val" || reject "EXCLUDE='${val}' is not a comma-separated node list" ;;
worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_submit_grid.sh:187:    *)        reject "unknown argument '${kv}' (expected WAVE=/EXCLUDE=/PIN_SHA=)" ;;
worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_submit_grid.sh:191:[ -n "$WAVE" ] || { echo "usage: [DRYRUN=1] bash $0 WAVE={vctl|zref|rgen|all} [EXCLUDE=n1,n2] [PIN_SHA=<40hex>]" >&2; exit 2; }
worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_submit_grid.sh:293:  [ -z "$EXCLUDE" ] || argv="${argv} EXCLUDE=${EXCLUDE}"
worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_screen_submit.sh:195:ARM=""; STEP=40000; SEED=42; K=8; CELL=""; EXCLUDE=""; ROTATE_DEG=""
worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_screen_submit.sh:232:    EXCLUDE)    is_nodelist "$val" || reject "EXCLUDE='${val}' is not a comma-separated node list" ;;
worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_screen_submit.sh:241:    *)          reject "unknown argument '${kv}' (expected ARM=/STEP=/SEED=/K=/CELL=/EXCLUDE=/ROTATE_DEG=/PIN_SHA=/LOG=)" ;;
worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_screen_submit.sh:329:[ -n "$ARM" ] && [ -n "$CELL" ] && [ -n "$STEP" ] || { echo "usage: bash $0 ARM=C4L CELL=rgen [STEP=40000] [SEED=42] [K=8] [EXCLUDE=node[,node]] [ROTATE_DEG=90] [PIN_SHA=<40hex>] [LOG=...]" >&2; exit 2; }
worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_screen_submit.sh:447:  echo "NOTE: SBATCH_EXCLUDE='${SBATCH_EXCLUDE}' is set but sbatch does not honour it;" >&2
worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_screen_submit.sh:448:  echo "      pass EXCLUDE=${SBATCH_EXCLUDE} to this script instead - abort" >&2
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_submit.sh:62:ARM=""; STEP=""; SEED=42; K=8; CELL=screen; EXCLUDE=""; ROTATE_DEG=""; EVAL_ORBIT=""
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_submit.sh:100:    EXCLUDE)    is_nodelist "$val" || reject "EXCLUDE='${val}' is not a comma-separated node list" ;;
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_submit.sh:109:    *)          reject "unknown argument '${kv}' (expected ARM=/STEP=/SEED=/K=/CELL=/EXCLUDE=/ROTATE_DEG=/EVAL_ORBIT=/PIN_SHA=/LOG=)" ;;
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_submit.sh:133:[ -n "$ARM" ] && [ -n "$STEP" ] || { echo "usage: bash $0 ARM=C4L STEP=10000 [SEED=42] [K=8] [CELL=screen] [EXCLUDE=node[,node]] [ROTATE_DEG=..] [EVAL_ORBIT=..] [PIN_SHA=<40hex>] [LOG=...]" >&2; exit 2; }
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_submit.sh:177:  echo "NOTE: SBATCH_EXCLUDE='${SBATCH_EXCLUDE}' is set but sbatch does not honour it;" >&2
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_submit.sh:178:  echo "      pass EXCLUDE=${SBATCH_EXCLUDE} to this script instead - abort" >&2
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_traj_submit.sh:13:#   bash fa_orbit_traj_submit.sh ARM=C8 STEP=42500 [EXCLUDE=node,node] [DRYRUN=1]
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_traj_submit.sh:23:ARM=""; STEP=""; EXCLUDE=""; DRYRUN=0
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_traj_submit.sh:26:    ARM=*|STEP=*|EXCLUDE=*|DRYRUN=*) eval_free_key="${kv%%=*}"; val="${kv#*=}" ;;
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_traj_submit.sh:27:    *) echo "unknown argument '${kv}' (expected ARM=/STEP=/EXCLUDE=/DRYRUN=)" >&2; exit 2 ;;
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_traj_submit.sh:33:    EXCLUDE) EXCLUDE="$val" ;;
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_traj_submit.sh:37:[ -n "$ARM" ] && [ -n "$STEP" ] || { echo "usage: bash $0 ARM=C8 STEP=42500 [EXCLUDE=n1,n2] [DRYRUN=1]" >&2; exit 2; }
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_traj_submit.sh:47:    [ -n "$EXCLUDE" ] && ARGS+=(EXCLUDE="$EXCLUDE")
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:857:  # --- EXCLUDE= reaches sbatch as an explicit FLAG ---------------------------
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:864:             bash "$SUB" ARM=C4L STEP=10000 EXCLUDE=neu303,neu332 2>&1)"; rc=$?
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:866:    echo "PASS  EXCLUDE= is passed to sbatch as an explicit --exclude flag"; PASS=$((PASS + 1))
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:868:    echo "FAIL  EXCLUDE= did not reach sbatch (rc=${rc})"; sed 's/^/        | /' "$TRACE"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:878:    echo "PASS  no EXCLUDE= means no --exclude flag"; PASS=$((PASS + 1))
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:880:    echo "FAIL  an --exclude flag appeared without EXCLUDE="; FAIL=$((FAIL + 1))
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:884:  out="$(env MOCK_TRACE="$TRACE" MOCK_WT="$WT" SBATCH_EXCLUDE=neu303 \
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:927:  for bad in "ARM=C4L;rm" "CELL=../etc" "STEP=1e4" "K=3" "EVAL_ORBIT=6" "EXCLUDE=neu1;id" "ROTATE_DEG=1.2.3"; do
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh:993:  env DRYRUN=1 EXCLUDE=neu399 bash "$SUBMITTER" C8 --resume x.ckpt --expected-step 40000 --chunk-end 42500
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh:994:out="$(env DRYRUN=1 EXCLUDE= bash "$SUBMITTER" C8 --resume x.ckpt --expected-step 40000 --chunk-end 42500 2>&1)"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh:996:  echo "FAIL  EXCLUDE=\"\" should disable exclusion"; FAIL=$((FAIL+1))
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh:998:  echo "PASS  EXCLUDE=\"\" disables exclusion"; PASS=$((PASS+1))
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh:1001:  env DRYRUN=1 "EXCLUDE=neu301;rm -rf x" bash "$SUBMITTER" C8 --resume x.ckpt --expected-step 40000 --chunk-end 42500
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submit.sh:67:# DEFAULT; EXCLUDE= overrides it (empty EXCLUDE="" disables exclusion, e.g. if
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submit.sh:69:EXCLUDE="${EXCLUDE-neu301,neu303,neu305,neu306,neu317,neu319,neu322,neu332}"
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submit.sh:72:    *[!a-z0-9,]*) echo "EXCLUDE='${EXCLUDE}' is not a comma-separated node list - abort"; exit 2 ;;
worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_submit_grid.sh:42:#   bash yaw_aug_submit_grid.sh WAVE=vctl EXCLUDE=neu303,neu332
worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_submit_grid.sh:171:WAVE=""; EXCLUDE=""; PIN_SHA=""
worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_submit_grid.sh:191:    EXCLUDE)  is_nodelist "$val" || reject "EXCLUDE='${val}' is not a comma-separated node list" ;;
worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_submit_grid.sh:193:    *)        reject "unknown argument '${kv}' (expected WAVE=/EXCLUDE=/PIN_SHA=)" ;;
worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_submit_grid.sh:197:[ -n "$WAVE" ] || { echo "usage: [DRYRUN=1] bash $0 WAVE={vctl|tbl|rrob|all} [EXCLUDE=n1,n2] [PIN_SHA=<40hex>]" >&2; exit 2; }
worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_submit_grid.sh:391:  [ -z "$EXCLUDE" ] || argv="${argv} EXCLUDE=${EXCLUDE}"
worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_screen_submit.sh:266:ARM=""; STEP=40000; SEED=42; K=8; CELL=""; EXCLUDE=""; ROTATE_DEG=""
worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_screen_submit.sh:303:    EXCLUDE)    is_nodelist "$val" || reject "EXCLUDE='${val}' is not a comma-separated node list" ;;
worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_screen_submit.sh:312:    *)          reject "unknown argument '${kv}' (expected ARM=/STEP=/SEED=/K=/CELL=/EXCLUDE=/ROTATE_DEG=/PIN_SHA=/LOG=)" ;;
worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_screen_submit.sh:489:[ -n "$ARM" ] && [ -n "$CELL" ] && [ -n "$STEP" ] || { echo "usage: bash $0 ARM=YAWAUG CELL=tbl [STEP=40000] [SEED=42] [K=8] [EXCLUDE=node[,node]] [ROTATE_DEG=90] [PIN_SHA=<40hex>] [LOG=...]" >&2; exit 2; }
worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_screen_submit.sh:694:  echo "NOTE: SBATCH_EXCLUDE='${SBATCH_EXCLUDE}' is set but sbatch does not honour it;" >&2
worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_screen_submit.sh:695:  echo "      pass EXCLUDE=${SBATCH_EXCLUDE} to this script instead - abort" >&2
worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_screen_guardtests.sh:1435:grep -q 'SBATCH_EXCLUDE' "$SUB" && grep -q 'pass EXCLUDE=' "$SUB"
worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_screen_guardtests.sh:1439:DRYX="$(env DRYRUN=1 "YAW_EVAL_MAIN_REPO=${FAKE_REPO}" bash "$GRID" WAVE=vctl EXCLUDE=neu303,neu332 2>/dev/null)"
worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_screen_guardtests.sh:1440:echo "$DRYX" | grep -q "EXCLUDE=neu303,neu332"
worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_screen_guardtests.sh:1441:check "EXCLUDE= is passed through to every cell of a wave" $?
worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_screen_guardtests.sh:1443:  env DRYRUN=1 "YAW_EVAL_MAIN_REPO=${FAKE_REPO}" bash "$GRID" WAVE=vctl "EXCLUDE=neu303;rm"
#!/usr/bin/env bash
# ============================================================================
# fa_orbit_train_guardtests.sh — guard-branch exercise for the exp_11 arm
# launcher (round-3 review B8 rebuilt this suite).
#
# SAFETY (the old suite violated all three):
#   * it never writes under a production output prefix — every case runs with
#     OUTPUT_ROOT pointed at a mktemp directory;
#   * it never mutates a tracked config — the mislabel case copies the tree into
#     the temp root and points the launcher at the copy via OUTPUT_ROOT-style
#     isolation, and any file it does touch is restored by an EXIT trap;
#   * it submits nothing and touches no GPU.
#
# Vehicles:
#   DRYRUN=1        every cheap gate (pins, arm, rung, config map, semantic
#                   gate, lineage, argv parity), then exit before Slurm/GPU.
#   real mode       with a fake SLURM_JOB_ID: proves the commit/drift and
#                   sbatch-only gates are fail-closed.
#   mocked logs     fa_orbit_classify.py is driven directly over synthetic logs
#                   to prove every exit class (0/3/4/6/7).
#   synthetic ckpt  fa_orbit_ckpt_preflight.py is driven over torch.save'd
#                   Lightning-shaped checkpoints to prove the restart depth.
#
# Usage:  bash worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh
# Exit 0 = every case behaved as specified.
# ============================================================================
set -uo pipefail
cd "$(git -C "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" rev-parse --show-toplevel)" || exit 3

EXPDIR="worklog/worklog_yixun/exp_11_fa_orbit_claude"
LAUNCHER="${EXPDIR}/fa_orbit_train.sbatch"
SUBMITTER="${EXPDIR}/fa_orbit_submit.sh"
CLASSIFY="${EXPDIR}/fa_orbit_classify.py"
PREFLIGHT="${EXPDIR}/fa_orbit_ckpt_preflight.py"
PY=/n/fs/gatrdp/envs/flac/bin/python
TS="$(date '+%Y-%m-%d_%H-%M-%S')"
LOG="${EXPDIR}/fa_orbit_${TS}_guardtests.log"
HEAD_SHA="$(git rev-parse HEAD)"

exec > >(tee -a "$LOG") 2>&1
echo "=== fa_orbit_train guard exercise — ${TS} — $(git rev-parse --short HEAD) ==="
for f in "$LAUNCHER" "$SUBMITTER" "$CLASSIFY" "$PREFLIGHT"; do
  [ -f "$f" ] || { echo "missing ${f} - abort"; exit 3; }
done

TRACKED_BEFORE="$(git status --porcelain -- "$EXPDIR" src | sort)"
TMP="$(mktemp -d)"
OUT_ROOT="${TMP}/outputs"            # never a production prefix
mkdir -p "$OUT_ROOT"
trap 'rm -rf "$TMP"' EXIT
PASS=0; FAIL=0

case_run() {  # <name> <want-rc> <want-substring> -- <env...>   (runs the launcher)
  local name="$1" want_rc="$2" want_txt="$3"; shift 3; [ "$1" = "--" ] && shift
  local out rc
  out="$(env "$@" bash "$LAUNCHER" 2>&1)"; rc=$?
  if [ "$rc" -eq "$want_rc" ] && echo "$out" | grep -qF -- "$want_txt"; then
    echo "PASS  ${name}  (rc=${rc})"; PASS=$((PASS + 1))
  else
    echo "FAIL  ${name}: want rc=${want_rc} + '${want_txt}', got rc=${rc}"
    echo "$out" | tail -5 | sed 's/^/        | /'; FAIL=$((FAIL + 1))
  fi
}

expect_cmd() {  # <name> <want-rc> <want-substring> -- <command...>
  local name="$1" want_rc="$2" want_txt="$3"; shift 3; [ "$1" = "--" ] && shift
  local out rc
  out="$("$@" 2>&1)"; rc=$?
  if [ "$rc" -eq "$want_rc" ] && echo "$out" | grep -qF -- "$want_txt"; then
    echo "PASS  ${name}  (rc=${rc})"; PASS=$((PASS + 1))
  else
    echo "FAIL  ${name}: want rc=${want_rc} + '${want_txt}', got rc=${rc}"
    echo "$out" | tail -5 | sed 's/^/        | /'; FAIL=$((FAIL + 1))
  fi
}

REPO_ENV=("FA_ORBIT_REPO_OVERRIDE=$PWD")   # dry runs read THIS tree, not the production checkout
SMOKE_ENV=(DRYRUN=1 "EXPECT_SHA=${HEAD_SHA}" "OUTPUT_ROOT=${OUT_ROOT}" "${REPO_ENV[@]}" SMOKE=1 SMOKE_RUNG=16x4 SMOKE_MIN_FREE_MB=14000)

echo "--- A. the pin mechanism refuses to launch un-pinned (round-3 B1) ---"
# RETIRED: this asserted that an UNPINNED arm refuses, but every pin landed in
# ea94995, so the placeholder no longer appears in any value and the case could
# never fire. Replaced by the end state it was protecting, plus proof that the
# refusal mechanism itself is still present to catch a future unpinned value.
if grep -qE '^PINNED_[A-Z_]+="TO-PIN-AFTER-P0"' "$LAUNCHER"; then
  echo "FAIL  a launcher pin is still the placeholder"; FAIL=$((FAIL+1))
else
  echo "PASS  every launcher pin holds a concrete value"; PASS=$((PASS+1))
fi
if grep -q 'PIN_PLACEHOLDER="TO-PIN-AFTER-P0"' "$LAUNCHER" \
   && grep -q 'PIN_PLACEHOLDER' "$LAUNCHER"; then
  echo "PASS  the launcher still refuses a placeholder pin if one returns"; PASS=$((PASS+1))
else
  echo "FAIL  the placeholder refusal mechanism is gone"; FAIL=$((FAIL+1))
fi
case_run "SMOKE bypasses the pins" 0 "ARGV PARITY OK" -- "${SMOKE_ENV[@]}" ARM=C8
case_run "SMOKE needs a rung" 2 "SMOKE_RUNG" \
  -- DRYRUN=1 SMOKE=1 ARM=C8 "EXPECT_SHA=${HEAD_SHA}" "OUTPUT_ROOT=${OUT_ROOT}" "${REPO_ENV[@]}"
case_run "SMOKE needs a VRAM floor" 2 "SMOKE_MIN_FREE_MB" \
  -- DRYRUN=1 SMOKE=1 SMOKE_RUNG=16x4 ARM=C8 "EXPECT_SHA=${HEAD_SHA}" "OUTPUT_ROOT=${OUT_ROOT}" "${REPO_ENV[@]}"
case_run "SMOKE identity is separate" 0 "exp11_smoke_C8" -- "${SMOKE_ENV[@]}" ARM=C8

echo "--- B. parameter / arm / rung gates ---"
case_run "missing ARM" 2 "ARM" -- DRYRUN=1 "EXPECT_SHA=${HEAD_SHA}" "OUTPUT_ROOT=${OUT_ROOT}" "${REPO_ENV[@]}"
case_run "missing EXPECT_SHA" 2 "EXPECT_SHA" -- DRYRUN=1 ARM=C8 "OUTPUT_ROOT=${OUT_ROOT}" "${REPO_ENV[@]}"
for BAD in C7 FA1 VAN CKPT4; do
  case_run "arm ${BAD} rejected" 2 "not a legal exp_11 arm" -- "${SMOKE_ENV[@]}" ARM=$BAD
done
case_run "bogus rung rejected" 2 "must be 32x2, 16x4 or 8x8" \
  -- DRYRUN=1 SMOKE=1 SMOKE_RUNG=64x1 SMOKE_MIN_FREE_MB=1 ARM=C8 "EXPECT_SHA=${HEAD_SHA}" "OUTPUT_ROOT=${OUT_ROOT}" "${REPO_ENV[@]}"
for R in 32x2 16x4 8x8; do   # all three rungs are feasible now that grad-ckpt is on
  case_run "rung ${R} accepted" 0 "ARGV PARITY OK" \
    -- DRYRUN=1 SMOKE=1 "SMOKE_RUNG=${R}" SMOKE_MIN_FREE_MB=14000 ARM=C4L "EXPECT_SHA=${HEAD_SHA}" "OUTPUT_ROOT=${OUT_ROOT}" "${REPO_ENV[@]}"
done

echo "--- C. lineage gates ---"
: > "${TMP}/foreign.ckpt"
case_run "initial + RESUME_CKPT" 2 "INITIAL launch must not carry" \
  -- "${SMOKE_ENV[@]}" ARM=C8 "RESUME_CKPT=${TMP}/foreign.ckpt"
case_run "restart w/o ckpt" 2 "RESTART requires RESUME_CKPT" \
  -- "${SMOKE_ENV[@]}" ARM=C8 EXPECTED_STEP=5000
case_run "restart ckpt missing" 2 "not found" \
  -- "${SMOKE_ENV[@]}" ARM=C8 EXPECTED_STEP=5000 "RESUME_CKPT=${TMP}/nope.ckpt"
case_run "restart foreign ckpt" 2 "may only resume a checkpoint from" \
  -- "${SMOKE_ENV[@]}" ARM=C8 EXPECTED_STEP=5000 "RESUME_CKPT=${TMP}/foreign.ckpt"
# a ckpt in the arm's own checkpoints dir but NOT named .ckpt / one level up
SMOKE_RUN="${OUT_ROOT}/exp11_smoke/C8/FLAC_exp11_smoke_C8/exp11_smoke_C8"
mkdir -p "${SMOKE_RUN}/checkpoints"
: > "${SMOKE_RUN}/checkpoints/epoch=1-step=5000.ckpt"
: > "${SMOKE_RUN}/notes.txt"
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

# --- fake scheduler binaries (round-5 r2 review) -----------------------------
# The new gates talk to the SCHEDULER: the submitter re-checks `squeue` INSIDE
# its per-arm reservation lock, and the recorder requires `sacct` to confirm the
# producing job COMPLETED. Both are exercised with fakes rather than the live
# queue, so the cases are deterministic. Every case that could conceivably reach
# `sbatch` also runs with a FAKE sbatch first on PATH: a guard test must not be
# able to queue a real job even if its assertion fails.
SHIM="${TMP}/shim"; mkdir -p "$SHIM"
cat > "${SHIM}/sbatch" <<'EOF'
#!/usr/bin/env bash
echo "FAKE-SBATCH: guard test — nothing was submitted"
exit 1
EOF
cat > "${SHIM}/squeue" <<'EOF'
#!/usr/bin/env bash
D="$(dirname "$(readlink -f "$0")")"
[ -s "${D}/squeue_out" ] && cat "${D}/squeue_out"
exit "$(cat "${D}/squeue_rc" 2>/dev/null || echo 0)"
EOF
printf '#!/usr/bin/env bash\necho COMPLETED\n'          > "${SHIM}/sacct_completed"
printf '#!/usr/bin/env bash\necho FAILED\n'             > "${SHIM}/sacct_failed"
printf '#!/usr/bin/env bash\nexit 0\n'                  > "${SHIM}/sacct_empty"
printf '#!/usr/bin/env bash\necho "sacct: connection refused" >&2\nexit 1\n' > "${SHIM}/sacct_rc1"
chmod +x "${SHIM}/sbatch" "${SHIM}/squeue" "${SHIM}"/sacct_*
: > "${SHIM}/squeue_out"; echo 0 > "${SHIM}/squeue_rc"

chain_pf() {  # <registry> [extra preflight args...]
  local reg="$1"; shift
  $PY "$PREFLIGHT" --config "${EXPDIR}/FLAC_AR_BF_C8.json" --arm C8 --rung 8x8 \
    --max-steps 100000 --ckpt "$CH_CKPT42" --expected-step 42500 --commit "$HEAD_SHA" \
    --launch-manifest "${TMP}/chain_launch_manifest.txt" --chain --launch-registry "$reg" "$@"
}
expect_cmd "chain refuses an UNRECORDED predecessor (the fail-closed core)" 2 "no recorded chain link" -- \
  chain_pf "${TMP}/chain_registry.json"
# The recorder now demands the SCHEDULER's confirmation, so every case that is
# NOT about that gate supplies a sacct that says COMPLETED; the sacct cases below
# override it (argparse takes the last --sacct-bin).
recorder_with() { $PY "${EXPDIR}/fa_orbit_record_restart.py" C8 "$1" \
  --registry "${TMP}/chain_registry.json" --launcher "$LAUNCHER" --repo-root "$PWD" \
  --sacct-bin "${SHIM}/sacct_completed" "${@:2}"; }
recorder() { recorder_with "${TMP}/chunk_leg_manifest.txt" "$@"; }
# B6: the manifest is published BEFORE training, so it proves intent, not
# authorship. Without the producing job's post-classification attestation, a
# FAILED leg's manifest plus a pre-existing 42500 checkpoint would mint a link.
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
# --- round-5 r2 blocking 1: the SUBMISSION RESERVATION lives in the submitter -
# The watchdog's singleton lock stops a second watchdog, not a human at a shell.
# The reservation therefore sits in the one sanctioned submitter: an exclusive
# per-arm flock, and a queue re-check INSIDE it. These cases run with a FAKE
# sbatch first on PATH, so even a failing assertion cannot queue anything.
SUB_LOCK="${EXPDIR}/.submit_C8.lock"
SUB_RELEASE="${TMP}/release_the_submit_lock"
( flock -n 9 || exit 1; while [ ! -e "$SUB_RELEASE" ]; do sleep 0.2; done ) 9>"$SUB_LOCK" &
SUB_HOLDER=$!
sleep 0.5
expect_cmd "submitter refuses while another submission holds the arm's lock" 2 "already holds" -- \
  env "PATH=${SHIM}:${PATH}" bash "$SUBMITTER" C8
: > "$SUB_RELEASE"
wait "$SUB_HOLDER" 2>/dev/null
# a queue we cannot read is never read as an empty queue
echo 1 > "${SHIM}/squeue_rc"; : > "${SHIM}/squeue_out"
expect_cmd "submitter refuses when squeue itself fails" 2 "the queue state is UNKNOWN" -- \
  env "PATH=${SHIM}:${PATH}" bash "$SUBMITTER" C8
# ...and a live job with this arm's name is a duplicate, whoever queued it
echo 0 > "${SHIM}/squeue_rc"; echo "4242424 RUNNING" > "${SHIM}/squeue_out"
expect_cmd "submitter refuses when a leg for the arm is already queued/running" 2 "already queued/running" -- \
  env "PATH=${SHIM}:${PATH}" bash "$SUBMITTER" C8
echo 0 > "${SHIM}/squeue_rc"; : > "${SHIM}/squeue_out"
awk '/flock -n 9/{if (!f) f=NR} /^  LIVE=.*squeue/{q=NR} /^OUT="\$\(sbatch/{s=NR}
     END{exit !(f && q && s && f < q && q < s)}' "$SUBMITTER" \
  && { echo "PASS  the submitter checks the queue INSIDE the reservation lock, before sbatch"; PASS=$((PASS+1)); } \
  || { echo "FAIL  the submitter's queue check is not sequenced flock -> squeue -> sbatch"; FAIL=$((FAIL+1)); }
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
# round-5 r2 non-blocking: a chunk longer than its arm's wall pin was sized for
# cannot reach its boundary inside the allocation, so it is refused BY NAME.
expect_cmd "watchdog rejects a chunk above the arm's time-pin maximum" 2 "PINNED_TIME_LIMIT_CHUNK_C32" -- \
  bash "$WD" ONESHOT=1 CHUNK_C32=5000

codex
The production path’s expansion, validation, argv construction, and lock/queue ordering are sound so far. I’m checking the new guards themselves for whether they genuinely prove those behaviors under the current dirty-tree gate.
exec
/usr/bin/bash -lc 'set +e
for mode in default override empty unsafe; do
  case "$mode" in
    default) env DRYRUN=1 bash worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submit.sh C8 --resume x.ckpt --expected-step 40000 --chunk-end 42500 ;;
    override) env DRYRUN=1 EXCLUDE=neu399 bash worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submit.sh C8 --resume x.ckpt --expected-step 40000 --chunk-end 42500 ;;
    empty) env DRYRUN=1 EXCLUDE= bash worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submit.sh C8 --resume x.ckpt --expected-step 40000 --chunk-end 42500 ;;
    unsafe) env DRYRUN=1 '"'EXCLUDE=neu301;rm -rf x' bash worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submit.sh C8 --resume x.ckpt --expected-step 40000 --chunk-end 42500 ;;
  esac
  rc="'$?
  echo "CASE_RC ${mode} ${rc}"
done' in /n/fs/gatrdp/codespace/FLAC
 succeeded in 0ms:
tracked measurement surfaces have uncommitted changes - commit first, abort:
 M worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submit.sh
 M worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh
CASE_RC default 2
tracked measurement surfaces have uncommitted changes - commit first, abort:
 M worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submit.sh
 M worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh
CASE_RC override 2
tracked measurement surfaces have uncommitted changes - commit first, abort:
 M worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submit.sh
 M worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh
CASE_RC empty 2
EXCLUDE='neu301;rm -rf x' is not a comma-separated node list - abort
CASE_RC unsafe 2

exec
/usr/bin/bash -lc "grep -nE 'SUBMITTER|fa_orbit_submit|env -i|EXCLUDE' worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_chunk_watchdog.sh; sed -n '250,390p' worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_chunk_watchdog.sh; sed -n '150,195p' worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_submit.sh; sed -n '45,115p' worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_submit.sh" in /n/fs/gatrdp/codespace/FLAC
 succeeded in 0ms:
13:# fa_orbit_record_restart.py and fa_orbit_submit.sh. It NEVER calls sbatch
47:#   B1 the anti-duplicate reservation MOVED INTO fa_orbit_submit.sh, where it
103:SUBMITTER="${EXPDIR}/fa_orbit_submit.sh"
161:[ -f "$SUBMITTER" ] || reject "missing ${SUBMITTER}"
392:  # trigger, not the anti-duplicate guard. The guard is inside fa_orbit_submit.sh,
572:  log "${arm}: submitting chunk ${S} -> ${next} (chunk size ${step}): bash ${SUBMITTER} ${arm} --resume ${resume} --expected-step ${S} --chunk-end ${next} (DRYRUN=${DRYRUN})"
574:  out="$(DRYRUN="$DRYRUN" bash "$SUBMITTER" "$arm" --resume "$resume" \
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

freeze_now() {  # <arm> <reason> — a freeze that is NOT a retryable failure
  local arm="$1" reason="$2"
  ST["frozen_$arm"]="$(date -Is)|${reason}"
  log "${arm}: !!! FROZEN immediately (${reason}). No further submission for this arm until its frozen_${arm} line is removed from ${STATE}."
}

clear_fail() {  # <arm> <why>
  if [ "${ST[fail_$1]:-0}" != "0" ]; then
    log "${1}: failure streak cleared (${2})"
  fi
  ST["fail_$1"]="0"
}

# B3 (round-5 r2): BOTH Lightning name shapes count. A retry at a boundary whose
# unversioned name already exists saves `epoch=E-step=N-v1.ckpt`, and a parser
# that ignored those would read the chain as stuck one boundary back.
newest_ckpt_step() {   # <ckpt dir> -> the largest step with a checkpoint file
  ls -1 "$1" 2>/dev/null \
    | sed -n -e 's/^epoch=[0-9]\{1,\}-step=\([0-9]\{1,\}\)\.ckpt$/\1/p' \
             -e 's/^epoch=[0-9]\{1,\}-step=\([0-9]\{1,\}\)-v[0-9]\{1,\}\.ckpt$/\1/p' \
    | sort -n | tail -1
}

ckpt_files_at() {      # <ckpt dir> <step> -> every checkpoint file at that step
  local d="$1" s="$2" f
  for f in "$d"/*-step="${s}".ckpt "$d"/*-step="${s}"-v*.ckpt; do
    [ -f "$f" ] && printf '%s\n' "$f"
  done
  return 0
}

chain_tip_step() {     # <arm> -> the arm's recorded chain tip (or its anchor, or 0)
  "$PY" - "$REGISTRY" "$1" <<'PY' 2>/dev/null
import json, sys
row = (json.load(open(sys.argv[1])).get("arms") or {}).get(sys.argv[2]) or {}
chain = row.get("chain") or []
print(int((chain[-1].get("final_step") if chain else row.get("final_step")) or 0))
PY
}

chain_tip_path() {     # <arm> -> the recorded tip's final_ckpt_path ('' if no chain)
  "$PY" - "$REGISTRY" "$1" <<'PY' 2>/dev/null
import json, sys
row = (json.load(open(sys.argv[1])).get("arms") or {}).get(sys.argv[2]) or {}
chain = row.get("chain") or []
print((chain[-1].get("final_ckpt_path") if chain else "") or "")
PY
}

# B7+ (round-5 r2): the startup anchor check is DEEP, not a presence test. It
# proves the arm's audited anchor is a real, unambiguous 40k checkpoint sitting
# where the recorder and preflight will look for it, and that its bytes still
# hash to the audited value. The canonical directory is derived from the
# REGISTRY's own save_dir (what the recorder and preflight use), which in
# production is exactly OUTPUT_ROOT/exp11_<ARM>/... — the same directory this
# script polls. One ~700 MB hash per arm, once, at startup.
anchor_state() {       # <arm> -> OK | <a one-line reason> | UNREADABLE
  "$PY" - "$REGISTRY" "$1" "$EXPDIR" <<'PY' 2>/dev/null || echo UNREADABLE
import hashlib, json, os, re, sys
reg_path, arm, expdir = sys.argv[1:4]
sys.path.insert(0, expdir)
from fa_orbit_ckpt_preflight import canonical_ckpt_dir      # noqa: E402
row = (json.load(open(reg_path)).get("arms") or {}).get(arm) or {}
sha, step = row.get("final_ckpt_sha256"), row.get("final_step")
if not sha or step is None:
    print("MISSING"); raise SystemExit(0)
if str(step) != "40000":
    print(f"the audited anchor is step {step!r}, not the 40000 every INITIAL run ended at")
    raise SystemExit(0)
if not re.fullmatch(r"[0-9a-f]{64}", str(sha)):
    print(f"the audited final_ckpt_sha256 {str(sha)[:20]!r} is not a 64-char lowercase hex digest")
    raise SystemExit(0)
ckdir = canonical_ckpt_dir(row.get("save_dir", ""), arm, os.getcwd())
hits = [f for f in sorted(os.listdir(ckdir)) if re.search(r"-step=40000(-v\d+)?\.ckpt$", f)] \
    if os.path.isdir(ckdir) else []
if len(hits) != 1:
    print(f"expected exactly one step=40000 checkpoint in {ckdir}, found {len(hits)}"
          + (": " + ", ".join(hits) if hits else ""))
    raise SystemExit(0)
h = hashlib.sha256()
with open(os.path.join(ckdir, hits[0]), "rb") as fh:
    for blk in iter(lambda: fh.read(1 << 22), b""):
        h.update(blk)
got = h.hexdigest()
print("OK" if got == sha
      else f"{hits[0]} hashes {got[:12]}, not the audited anchor {str(sha)[:12]}")
PY
}

leg_manifest_by_job() {  # <arm> <jobid> -> the manifest THAT job published
  local arm="$1" jid="$2" f
  local -a hits=()
  for f in "$EXPDIR"/fa_orbit_*_"${arm}"_*_jid"${jid}"_manifest.txt; do
    [ -f "$f" ] && hits+=("$f")
  done
  [ "${#hits[@]}" -eq 1 ] || return 1
  printf '%s\n' "${hits[0]}"
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

manifest_chunk_end() { # <manifest> -> its chunk_end line
  awk '/^chunk_end /{print $2; exit}' "$1" 2>/dev/null
}

process_arm() {
  local arm="$1"
  if [ -n "${ST[frozen_$arm]:-}" ]; then
    log "${arm}: FROZEN (${ST[frozen_$arm]}) — skipping; clear it in ${STATE} to resume"
    return 0
  fi

  # B2(b): a FAILING squeue is UNKNOWN, never "the queue is empty". Reading a
  # scheduler hiccup as "no live job" is how a running leg gets a twin.
  #
# 1. pin + assets (the helper refuses a dirty or mismatched tree). With PIN_SHA
#    the tree is prepared at THAT commit and EXPECT_SHA follows it, so the job's
#    commit binding checks the pin, not wherever the branch has drifted to.
WT="$("$HELPER" ${PIN_SHA:+"$PIN_SHA"} | tail -1)"
[ -d "$WT" ] || { echo "could not prepare a measurement worktree" >&2; exit 3; }
EXPECT_SHA="$(git -C "$WT" rev-parse HEAD)"
if [ -n "$PIN_SHA" ] && [ "$EXPECT_SHA" != "$PIN_SHA" ]; then
  echo "the prepared tree is at ${EXPECT_SHA}, not the requested PIN_SHA ${PIN_SHA}" >&2
  exit 3
fi
[ -n "$PIN_SHA" ] && echo "campaign pin: ${PIN_SHA}"

# 2. submit HELD: the id exists before the lease, the job runs after it
JOB_NAME="exp11-screen-${ARM}-${CELL}-${STEP}-s${SEED}-K${K}"
SBATCH="${FA_ORBIT_SBATCH:-sbatch}"          # guard-suite seam; a real run uses sbatch
SCONTROL="${FA_ORBIT_SCONTROL:-scontrol}"
SCANCEL="${FA_ORBIT_SCANCEL:-scancel}"
# Node exclusion is passed as an EXPLICIT FLAG, never through the environment.
# SBATCH_EXCLUDE does not exist: of the 58 input environment variables sbatch
# documents there is no --exclude equivalent (the lookalike SBATCH_EXCLUSIVE is
# --exclusive, a different option entirely). sbatch therefore ignored it in
# silence, and every batch that believed it was excluding sick nodes was not.
EXCLUDE_ARGV=()
if [ -n "$EXCLUDE" ]; then
  EXCLUDE_ARGV=(--exclude="$EXCLUDE")
  echo "excluding nodes: ${EXCLUDE}"
elif [ -n "${SBATCH_EXCLUDE:-}" ]; then
  echo "NOTE: SBATCH_EXCLUDE='${SBATCH_EXCLUDE}' is set but sbatch does not honour it;" >&2
  echo "      pass EXCLUDE=${SBATCH_EXCLUDE} to this script instead - abort" >&2
  exit 2
fi
# CELL-specific parameters travel with the job, not as ambient state.
CELL_EXPORT=""
[ -n "$ROTATE_DEG" ] && CELL_EXPORT="${CELL_EXPORT},ROTATE_DEG=${ROTATE_DEG}"
[ -n "$EVAL_ORBIT" ] && CELL_EXPORT="${CELL_EXPORT},EVAL_ORBIT=${EVAL_ORBIT}"
[ -n "$LOG" ] && CELL_EXPORT="${CELL_EXPORT},LOG=${LOG}"
JOBID="$("$SBATCH" --hold --parsable \
  --job-name="$JOB_NAME" \
  --output="${EXPDIR}/slurm_screen_%x_%j.out" \
  "${EXCLUDE_ARGV[@]}" \
  --export=ALL,MEASURE_ROOT="$WT",EXPECT_SHA="$EXPECT_SHA",ARM="$ARM",STEP="$STEP",SEED="$SEED",K="$K",CELL="$CELL""$CELL_EXPORT" \
  "$EXPDIR/fa_orbit_screen.sbatch")" || { echo "sbatch FAILED - nothing submitted" >&2; exit 4; }
JOBID="${JOBID%%;*}"
case "$JOBID" in ''|*[!0-9]*) echo "sbatch returned '${JOBID}', not a job id - abort" >&2; exit 4 ;; esac
echo "submitted HELD as ${JOBID}"

  have="$(readlink -f /proc/self/fd/8 2>/dev/null)" || return 1
  want="$(readlink -f "$LOCKFILE" 2>/dev/null)" || return 1
  # An unresolvable path yields the EMPTY string, and two empty strings compare
  # equal — which would turn a double failure into a "match". Require both.
  [ -n "$have" ] && [ -n "$want" ] && [ "$have" = "$want" ]
}

if [ "${FA_ORBIT_STORE_LOCK_HELD:-0}" = "1" ]; then
  fd8_is_the_store_lock || {
    echo "FA_ORBIT_STORE_LOCK_HELD is set but fd 8 is not ${LOCKFILE} —" >&2
    echo "refusing to run a submission that only CLAIMS to hold the store lock" >&2
    exit 2
  }
else
  exec bash "$HELPER" --with-lock bash "$0" "$@"
fi

ARM=""; STEP=""; SEED=42; K=8; CELL=screen; EXCLUDE=""; ROTATE_DEG=""; EVAL_ORBIT=""
PIN_SHA=""; LOG=""

# --- argument parsing: NEVER eval a value ------------------------------------
# The old parser ran `eval "KEY='VALUE'"`, so a value carrying a quote executed
# whatever followed it. Every key is whitelisted, every VALUE is shape-checked
# BEFORE it is assigned, and assignment goes through printf -v into a name that
# came from the whitelist -- the value is never parsed as shell at any point.
is_num()      { case "${1:-}" in ''|*[!0-9]*) return 1 ;; esac; }
is_decimal()  { case "${1:-}" in ''|*[!0-9.]*|.*|*.) return 1 ;; *.*.*) return 1 ;; esac; }
is_hex40()    {
  case "${1:-}" in
    *[!0-9a-f]*) return 1 ;;
    ????????????????????????????????????????) return 0 ;;
    *) return 1 ;;
  esac
}
is_nodelist() { case "${1:-}" in ''|*[!a-z0-9,]*) return 1 ;; esac; }
in_set()      { local v="$1"; shift; local t; for t in "$@"; do [ "$v" = "$t" ] && return 0; done; return 1; }
reject()      { echo "$1" >&2; exit 2; }

for kv in "$@"; do
  case "$kv" in
    *=*) ;;
    *) reject "argument '${kv}' is not KEY=VALUE" ;;
  esac
  key="${kv%%=*}"; val="${kv#*=}"
  case "$key" in
    ARM)        in_set "$val" C4L C8 C16 C32 VANL C4BACKFILL \
                  || reject "ARM='${val}' is not a registered arm" ;;
    CELL)       in_set "$val" screen conf r3 cross q9 traj \
                  || reject "CELL='${val}' is not a registered cell type" ;;
    STEP)       is_num "$val" || reject "STEP='${val}' is not numeric" ;;
    SEED)       is_num "$val" || reject "SEED='${val}' is not numeric" ;;
    K)          in_set "$val" 1 8 || reject "K='${val}' is not 1 or 8" ;;
    EVAL_ORBIT) in_set "$val" 4 8 16 32 || reject "EVAL_ORBIT='${val}' is not 4|8|16|32" ;;
    ROTATE_DEG) is_decimal "$val" || reject "ROTATE_DEG='${val}' is not a decimal number" ;;
    PIN_SHA)    is_hex40 "$val" || reject "PIN_SHA='${val}' is not 40 hex characters" ;;
    EXCLUDE)    is_nodelist "$val" || reject "EXCLUDE='${val}' is not a comma-separated node list" ;;
    LOG)        # kept to a durable, provenance-preserving location on purpose:
                # the overnight incidents were a tee into the pinned worktree and
                # node-local /tmp paths, both of which this shape excludes
                case "$val" in
                  "${EXPDIR}"/*_screen.log) ;;
                  *) reject "LOG='${val}' must be an absolute ${EXPDIR}/..._screen.log path" ;;
                esac
                case "$val" in *[!A-Za-z0-9/._-]*) reject "LOG='${val}' has unsafe characters" ;; esac ;;
    *)          reject "unknown argument '${kv}' (expected ARM=/STEP=/SEED=/K=/CELL=/EXCLUDE=/ROTATE_DEG=/EVAL_ORBIT=/PIN_SHA=/LOG=)" ;;
  esac
  printf -v "$key" '%s' "$val"      # name whitelisted above; value never parsed
done

# --- the CAMPAIGN PIN is the default, and disagreement is refused ------------
CAMPAIGN_PIN="$(bash "$HELPER" --pinned 2>/dev/null)" || CAMPAIGN_PIN=""

exec
/usr/bin/bash -lc "nl -ba worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submit.sh | sed -n '60,78p;185,205p;218,232p'; nl -ba worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh | sed -n '984,1005p'" in /n/fs/gatrdp/codespace/FLAC
 succeeded in 0ms:
    60	# A chunk is meaningful only for a leg that resumes: it narrows where THIS job
    61	# stops, never what the campaign may reach (the budget pin stays 100000). It must
    62	# land on a saved checkpoint, or the next chunk has nothing to resume from.
    63	# --- node exclusion (round 6, 2026-08-30) -------------------------------------
    64	# Chunked legs re-roll the node lottery ~24x per arm, and the ECC-flaky nodes
    65	# (CLAUDE.md, Aug 2026) burned 4 legs in 3 days: two on neu306, two on neu301
    66	# (NCCL watchdog hang), freezing every remaining arm. The sick list is the
    67	# DEFAULT; EXCLUDE= overrides it (empty EXCLUDE="" disables exclusion, e.g. if
    68	# the pool shrinks). Same nodelist shape check as the screen submitter.
    69	EXCLUDE="${EXCLUDE-neu301,neu303,neu305,neu306,neu317,neu319,neu322,neu332}"
    70	if [ -n "$EXCLUDE" ]; then
    71	  case "$EXCLUDE" in
    72	    *[!a-z0-9,]*) echo "EXCLUDE='${EXCLUDE}' is not a comma-separated node list - abort"; exit 2 ;;
    73	  esac
    74	fi
    75	
    76	CHUNK_BUDGET="$(awk -F= '/^PINNED_MAXSTEPS=/{split($2,a," "); print a[1]; exit}' "$SBATCH_FILE")"
    77	case "$CHUNK_BUDGET" in ''|*[!0-9]*) echo "could not read PINNED_MAXSTEPS from ${SBATCH_FILE} - abort"; exit 3;; esac
    78	if [ -n "$CHUNK_END" ]; then
   185	         worklog/worklog_yixun/exp_07_fa_scratch_claude/FLAC_AR_BF.json 2>/dev/null)"
   186	[ -z "$DRIFT" ] || { echo "tracked measurement surfaces have uncommitted changes - commit first, abort:"; echo "$DRIFT"; exit 2; }
   187	SHA="$(git rev-parse HEAD)"
   188	
   189	ARGS=(
   190	  --job-name="$JOBNAME"
   191	  --gres="gpu:l40:${NGPU}"
   192	  --cpus-per-task="$((8 + 7 * NGPU))"
   193	  --mem="$(((12 * NGPU + 12)))G"
   194	  --time="$TIME_LIMIT"
   195	  --export="ALL,ARM=${ARM},EXPECT_SHA=${SHA},OUTPUT_ROOT=outputs_FLAC"
   196	)
   197	[ -n "$EXCLUDE" ] && ARGS+=(--exclude="$EXCLUDE")
   198	[ "$SMOKE" = "1" ] && ARGS[5]="${ARGS[5]},SMOKE=1,SMOKE_RUNG=${SMOKE_RUNG},SMOKE_MIN_FREE_MB=${SMOKE_MIN_FREE_MB},SMOKE_MAXSTEPS=${SMOKE_MAXSTEPS:-30},SMOKE_TIME=${TIME_LIMIT}"
   199	[ -n "$RESUME_CKPT" ] && ARGS[5]="${ARGS[5]},RESUME_CKPT=${RESUME_CKPT},EXPECTED_STEP=${EXPECTED_STEP}"
   200	[ -n "$CHUNK_END" ] && ARGS[5]="${ARGS[5]},CHUNK_END=${CHUNK_END}"
   201	ARGS+=("$SBATCH_FILE")
   202	
   203	echo "arm ${ARM} | rung ${RUNG} (${MB}x${NGPU}) | time ${TIME_LIMIT} | commit ${SHA} | smoke ${SMOKE} | chunk_end ${CHUNK_END:-<none>}"
   204	if [ "$DRYRUN" = "1" ]; then
   205	  echo "DRYRUN sbatch ${ARGS[*]}"
   218	{
   219	  echo "# exp_11 arm submission (intent published BEFORE sbatch)"
   220	  echo "intent_id ${INTENT_ID}"
   221	  echo "submitted_at $(date -Is)"
   222	  echo "arm ${ARM} rung ${RUNG} micro ${MB} ngpu ${NGPU}"
   223	  echo "jobname ${JOBNAME} time ${TIME_LIMIT} smoke ${SMOKE}"
   224	  echo "commit ${SHA}"
   225	  echo "pins rung=${RUNG} maxsteps=$(pin PINNED_MAXSTEPS) ckpt_every=$(pin PINNED_CHECKPOINT_EVERY) min_free_mb=$(pin PINNED_MIN_FREE_MB) p0_manifest_sha256=$(pin PINNED_P0_MANIFEST_SHA256)"
   226	  echo "resume ${RESUME_CKPT:-<none>} expected_step ${EXPECTED_STEP}"
   227	  echo "chunk_end ${CHUNK_END:-<none>}"
   228	  echo "exclude ${EXCLUDE:-<none>}"
   229	  echo "sbatch sbatch ${ARGS[*]}"
   230	} >> "$TMP" || { echo "intent manifest write failed - abort"; exit 3; }
   231	mv -n "$TMP" "$MANIFEST" || { echo "intent manifest publication failed - abort"; exit 2; }
   232	[ -e "$MANIFEST" ] || { echo "intent manifest ${MANIFEST} did not appear - abort"; exit 2; }
   984	  echo "FAIL  no post-run wandb identity verification"; FAIL=$((FAIL+1))
   985	fi
   986	
   987	echo "--- M. round 6: sick-node exclusion in the training submitter ---"
   988	# 4 chunk legs burned on neu301/neu306 (ECC list) in 3 days; the training
   989	# submitter now defaults --exclude to the sick list. DRYRUN prints the line.
   990	expect_cmd "submitter defaults --exclude to the sick list" 0 "--exclude=neu301,neu303,neu305,neu306,neu317,neu319,neu322,neu332" -- \
   991	  env DRYRUN=1 bash "$SUBMITTER" C8 --resume x.ckpt --expected-step 40000 --chunk-end 42500
   992	expect_cmd "EXCLUDE override replaces the default" 0 "--exclude=neu399" -- \
   993	  env DRYRUN=1 EXCLUDE=neu399 bash "$SUBMITTER" C8 --resume x.ckpt --expected-step 40000 --chunk-end 42500
   994	out="$(env DRYRUN=1 EXCLUDE= bash "$SUBMITTER" C8 --resume x.ckpt --expected-step 40000 --chunk-end 42500 2>&1)"
   995	if echo "$out" | grep -q -- "--exclude"; then
   996	  echo "FAIL  EXCLUDE=\"\" should disable exclusion"; FAIL=$((FAIL+1))
   997	else
   998	  echo "PASS  EXCLUDE=\"\" disables exclusion"; PASS=$((PASS+1))
   999	fi
  1000	expect_cmd "submitter rejects an unsafe EXCLUDE" 2 "not a comma-separated node list" -- \
  1001	  env DRYRUN=1 "EXCLUDE=neu301;rm -rf x" bash "$SUBMITTER" C8 --resume x.ckpt --expected-step 40000 --chunk-end 42500
  1002	
  1003	echo
  1004	echo "=== guard tests: ${PASS} passed, ${FAIL} failed ==="
  1005	[ "$FAIL" -eq 0 ] || exit 1

codex
REVISE

Blocking guard-test hole: `fa_orbit_train_guardtests.sh:994-999` ignores the submitter’s exit code. With the current dirty tree, `EXCLUDE=""` exits 2 at the drift gate without printing `--exclude`, yet the test falsely reports PASS. Require exit 0 and a `DRYRUN sbatch` line, in addition to asserting that `--exclude` is absent.

The production delta itself is sound:

- `${EXCLUDE-default}` correctly preserves explicit empty while defaulting only when unset.
- Validation permits only `[a-z0-9,]` and runs before the drift and reservation gates.
- `--exclude` is safely appended as one array argument.
- Reservation/flock/squeue behavior is unchanged; differing exclusions cannot bypass duplicate detection.
- DRYRUN bypasses reservation and manifests while displaying the effective argv.
- The default matches CLAUDE.md, and the intent manifest records the effective exclusion.
tokens used
58,197
REVISE

Blocking guard-test hole: `fa_orbit_train_guardtests.sh:994-999` ignores the submitter’s exit code. With the current dirty tree, `EXCLUDE=""` exits 2 at the drift gate without printing `--exclude`, yet the test falsely reports PASS. Require exit 0 and a `DRYRUN sbatch` line, in addition to asserting that `--exclude` is absent.

The production delta itself is sound:

- `${EXCLUDE-default}` correctly preserves explicit empty while defaulting only when unset.
- Validation permits only `[a-z0-9,]` and runs before the drift and reservation gates.
- `--exclude` is safely appended as one array argument.
- Reservation/flock/squeue behavior is unchanged; differing exclusions cannot bypass duplicate detection.
- DRYRUN bypasses reservation and manifests while displaying the effective argv.
- The default matches CLAUDE.md, and the intent manifest records the effective exclusion.
