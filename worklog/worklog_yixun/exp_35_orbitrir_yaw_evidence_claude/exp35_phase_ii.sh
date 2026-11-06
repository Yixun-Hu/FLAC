#!/usr/bin/env bash
# exp_35 phase (ii): the Planner-run launch kit (plan_orbitrir_yaw_evidence.md §15 + its r1 clarifications (a)-(c);
# Codex plan §15 r1 "S6 -- binding kit and test contract"). Every check lives in exp35_phase_ii.py beside this file;
# this CLI runs it in the phase-(i) launch environment (env -i), pins the reviewed kit commit, and tees every
# mutating mode to orbitrir_yaw_evidence_<ts>_phase_ii_<mode>.log here.
#
#   EXPECT_KIT_COMMIT=<reviewed commit> bash exp35_phase_ii.sh reserve <H35> --lanes 0 --report <path> --report-sha256 <hex>
#   EXPECT_KIT_COMMIT=<reviewed commit> bash exp35_phase_ii.sh dry-run <H35> --orch <orchestration directory>
#   EXPECT_KIT_COMMIT=<reviewed commit> bash exp35_phase_ii.sh launch  <H35> --orch <orchestration directory>
#   [EXPECT_KIT_COMMIT=<reviewed commit>] bash exp35_phase_ii.sh status [--orch <orchestration directory>]
#   EXPECT_KIT_COMMIT=<reviewed commit> bash exp35_phase_ii.sh release <H35> --orch <orchestration directory>
#
# status is read-only and writes no log. reserve prints the orchestration directory
# (<run worktree>/<exp_35 folder>/exp35_phase_ii_<RUN>) the other modes take. The report is a file of THIS checkout,
# committed at HEAD and clean, holding these six lines (anywhere, e.g. in a code block; each exactly once):
#   exp35-phase-ii h35 <H35>
#   exp35-phase-ii phase-i-state-sha256 <the final phase-(i) state's sha256, as status prints it>
#   exp35-phase-ii lanes <the frozen lane list: 0 | 1 | 0,1 | 1,0 -- equal to --lanes>
#   exp35-phase-ii second-lane-decision <none for one lane | the recorded Planner decision for two>
#   exp35-phase-ii eta <the phase-(ii) ETA as reported to Yixun>
#   exp35-phase-ii prior-runs <none | the comma-separated run ids of every earlier launched orchestration>
# Exit: 0 done | 1 refused or failed (the reasons printed; every failure state retained for review) | 2 bad arguments.
# Never signals, kills or attaches to any process; never touches another holder's reservation; never relaunches.
# Written by the exp_35 phase-(ii) kit Coder (Claude Opus 5.5, max effort).
set -uo pipefail
# --- BEGIN REGISTERED CONFIG (the sandbox tests replace this block, and only this block) ---
PY=/home/yixunhu/miniconda3/envs/flac/bin/python
# --- END REGISTERED CONFIG ---
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HELPER="$HERE/exp35_phase_ii.py"
ENV=(HOME=/home/yixunhu USER=yixunhu LOGNAME=yixunhu SHELL=/bin/bash LANG=en_US.UTF-8 PYTHONDONTWRITEBYTECODE=1
     PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin)
MODE="${1:-}"
USAGE="usage: EXPECT_KIT_COMMIT=<sha> bash $0 reserve|dry-run|launch|release <H35> ... | status [--orch DIR]"
case "$MODE" in
  status) shift; exec env -i "${ENV[@]}" "$PY" -B "$HELPER" status --kit-commit "${EXPECT_KIT_COMMIT:-}" "$@" ;;
  reserve|dry-run|launch|release) [ "$#" -ge 2 ] || { echo "$USAGE"; exit 2; } ;;
  *) echo "$USAGE"; exit 2 ;;
esac
LOG="$HERE/orbitrir_yaw_evidence_$(date +%Y-%m-%d_%H-%M-%S)_phase_ii_${MODE}.log"
( set -o noclobber; : > "$LOG" ) 2>/dev/null || { echo "the log ${LOG} exists - refused"; exit 2; }
env -i "${ENV[@]}" "$PY" -B "$HELPER" "$MODE" --kit-commit "${EXPECT_KIT_COMMIT:-}" --h35 "$2" "${@:3}" 2>&1 \
  | tee -a "$LOG"
rc=${PIPESTATUS[0]}
echo "rc ${rc}; log ${LOG} sha256 $(sha256sum < "$LOG" | cut -c1-64)" | tee -a "$LOG"
exit "$rc"
