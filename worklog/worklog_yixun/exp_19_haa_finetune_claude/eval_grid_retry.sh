#!/bin/bash
# exp_19: run the CYL+CYLSSL eval grid, retrying when the driver's train.py guard
# false-positives on transient processes whose COMMAND LINE merely mentions train.py
# (observed: another session's codex review prompt). Bounded retries; never overrides
# the guard when a real training holds a GPU -- it just tries again later.
set -uo pipefail
cd /home/yixunhu/codespace/FLAC
export PATH=/home/yixunhu/miniconda3/envs/flac/bin:$PATH
EXPDIR=worklog/worklog_yixun/exp_19_haa_finetune_claude
LOG=$EXPDIR/eval_grid_retry.log
say () { echo "[eval-retry] $* | $(date -Is)" >> "$LOG"; }
for i in $(seq 1 60); do
  say "attempt $i"
  ARMS="CYL CYLSSL" EXPECT_SHA=$(git rev-parse HEAD) bash $EXPDIR/haa_ft_eval.sh >> "$LOG" 2>&1
  rc=$?
  if [ $rc -eq 0 ]; then say "eval grid COMPLETE rc=0"; exit 0; fi
  if tail -20 "$LOG" | grep -q "REFUSING: train.py"; then
    say "guard refusal (rc=$rc) -- retrying in 5 min"
    sleep 300
  else
    say "non-guard failure rc=$rc -- stopping for a human look"
    exit $rc
  fi
done
say "gave up after 60 attempts (~5 h of refusals) -- needs a look"
exit 4
