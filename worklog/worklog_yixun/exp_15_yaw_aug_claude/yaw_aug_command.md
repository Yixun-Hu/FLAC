# exp_15 yaw_aug — reproduction commands (appended at launch time per SOP)

## 2026-08-12 — SMOKE (pre-production validation, rung 8×8)

```bash
cd /n/fs/gatrdp/codespace/FLAC
SMOKE=1 ARM=YAWAUG bash worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_submit.sh
# pin 5368108; acceptance record: worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_smoke_acceptance.json
```
(job id appended below at submission)

Submitted 2026-08-12 15:31 EDT as **job 3685989** (EXPECT_SHA `bd6d8b9`, time 00:30:00, intent manifest `yaw_aug_submission_YAWAUG_1786553496612380476-9ec25e2d.txt`). NOTE: file updated at launch; commit deferred until after the production submission (acceptance-record commit binding).

## 2026-08-12 — PRODUCTION INITIAL (40k, rung 8×8)

```bash
cd /n/fs/gatrdp/codespace/FLAC
SMOKE_WAIVER="<reason recorded in manifest + worklog 16:05 entry>" \
  bash worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_submit.sh YAWAUG
```
Submitted 2026-08-12 16:07 EDT as **job 3687499** (EXPECT_SHA `bd6d8b9`, time 24:00:00, intent manifest `yaw_aug_submission_YAWAUG_1786573598909833356-835d7790.txt`). Smoke 3685989 preceded it: FAIL on the rate check only — waived by Yixun (measurement-design artifact, VANL-log evidence); post-hoc windowed floors 0.849 (steps 100→300) / 0.843 (300→1000) with a pre-registered abort criterion.

## 2026-08-14 — CHAIN INITIAL (leg 1: steps 0→2500, self-chaining to 40000)

```bash
cd /n/fs/gatrdp/codespace/FLAC
CHAIN=1 SMOKE_WAIVER="<standing 2026-08-12 waiver; rate gate enforced in-chain at leg 1>" \
  bash worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_submit.sh YAWAUG
```
