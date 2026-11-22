# exp_46 — brief for the A6000 session: L7 (NAS publication) and L8 (40 evaluation cells)

Written 2026-11-22 11:48 EST by the neuronic coordinating session, on Yixun's word ("写 L7 + L8 的 brief"). Yixun
pastes this into the A6000 session; pasting it is his go for L7 and L8 as scoped below, and for nothing else.

## What happened on neuronic

The four exp_46 FULL trainings ended COMPLETED (exit 0:0) and SEALED on 2026-11-22, at Y46.

| Run | Job | End (11-22 EST) | Final checkpoint |
|---|---|---|---|
| `exp46_C4Ff025_l40_2x32` | 4290216 | 04:04 | `epoch=34-step=40000.ckpt` |
| `exp46_C4Ff050_l40_2x32` | 4290217 | 04:14 | `epoch=17-step=40000.ckpt` |
| `exp46_C4Ff075_l40_2x32` | 4290218 | 04:43 | `epoch=11-step=40000.ckpt` |
| `exp46_C4Ff100_l40_2x32` | 4290219 | 04:37 | `epoch=8-step=40000.ckpt` |

- Source of each run on neuronic: `/n/fs/gatrdp/codespace/FLAC/outputs_FLAC/exp46_c4flip_data_curve/<run>/`.
- Each run dir holds 16 checkpoints (11.6 GB per run, about 46 GB in all), `TRAINING_COMPLETE.json`,
  `TRAINING_COMMITTED.json`, `CHECKPOINTS.sha256`, `evidence/` and `receipts/`.
- X46 = `553fc89f523d6fe68f80db7539ee67460fa8472e`; Y46 = `77ecdaa28cff5a71c75896bb99d655ccaafc4598`.
- closure(Y46) = closure(X46) = `356a621ec16e8bfaf6e0d56cb258e4985f557cc88d7a73adf2d57a5a3ab05ad0`.
- The branch `exp-46-c4flip-data-curve` on origin is now at `1ea2897f` (records only: worklog and logs). Neither
  command sheet and none of the exp_46 tools changed after X46.

## Before anything else

1. In `~/codespace/exp-46-publish`: `git pull --rebase` (origin moved to `1ea2897f`). Never commit in tree 46
   (`~/codespace/exp-46-campaign`); it stays at X46.
2. Check that `ssh neuronic true` works without a prompt. If it asks for a password or a Duo code, STOP and tell
   Yixun; he types it himself. Never ask him for it in chat.
3. Check that the NAS is mounted and has room for about 46 GB under
   `/media/diskstation/yixunhu/FLAC/checkpoints/exp46_c4flip_data_curve`.

## L7 — staged NAS publication (neuronic sheet, step 4)

- Run the lines of step 4 of `$E/c4flip_data_curve_neuronic_commands.md` exactly as written, from tree 46, in the
  A6000 sheet's environment (step 0 of `$E/c4flip_data_curve_command.md`: `conda activate flac`,
  `PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1`, `set -e`, one command per line).
- Do it once per run, the four runs one after another: `provision` → `stage` → `rsync` → `sha256sum -c
  CHECKPOINTS.sha256` → `exp46_registry.py training-check` → `publish`.
- `training-check` must be given the closure above and the sha256 of `$E/c4flip_data_curve_e0_receipt.json`, as the
  sheet computes them. For f100 it also re-validates the delta verification record.
- Any refusal, checksum mismatch or failed check: STOP that run, change nothing, and report the message verbatim to
  Yixun. Never remove or rename a `.incoming` directory or a published directory by hand.
- The neuronic copies stay where they are. Do not delete or modify anything on neuronic.

## L8 — the 40 evaluation cells (A6000 sheet, step 2)

- Only after all four runs are published. Per run: `exp46_eval.py prepare $RUN`, then the two lanes exactly as the
  sheet writes them (K=8 on card 0, K=1 on card 1; seeds 42–46; each lane's PID waited on explicitly).
- Before each launch, look at both cards. If a card carries someone else's compute process, do not share or pre-empt
  it without Yixun's word: report it and ask. Never touch a process this session did not launch
  (`readlink /proc/<pid>/cwd`).
- After the fourth run: `exp46_collect.py`, then `exp46_collect.py verify`. All 40 cells and the E0 receipt must be
  admitted; any other exit is a STOP and a verbatim report.
- Record every launch in the A6000 sheet at launch time, and keep each run's terminal output in a timestamped log,
  in the publication worktree.

## Where to stop

- Stop after `exp46_collect.py verify` and report to Yixun: per run, the published path and the `training-check`
  result; per cell, done or failed; the headline numbers of the results file; wall-clock times.
- Step 3 of the A6000 sheet (the results commit R46) and its push are NOT part of this go. Each needs Yixun's word.
- Rough time, from E0's rate (8 cells in 1 h 45 min on one card): L8 is about 65 min per run, about 4.5 h for the
  four runs on two free cards. L7 depends on the neuronic → NAS transfer rate for 46 GB.
