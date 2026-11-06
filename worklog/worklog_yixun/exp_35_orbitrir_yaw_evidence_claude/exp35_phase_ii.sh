#!/usr/bin/env bash
# exp_35 phase (ii): the Planner-run launch kit (plan_orbitrir_yaw_evidence.md §15 + its r1 clarifications (a)-(c);
# Codex plan §15 r1 "S6 -- binding kit and test contract"). Every check lives in exp35_phase_ii.py beside this file;
# this CLI runs it in the phase-(i) launch environment (env -i), pins the reviewed kit commit, and tees every
# mutating mode to orbitrir_yaw_evidence_<ts>_phase_ii_<mode>.log here (its final sha256 printed after it, outside it).
#
# RETIRED (exp_36 code review III HIGH 2): Yixun superseded phase (ii) on 2026-11-06 15:30 EST ("Seeds 43-46 at these
# angles are NOT wanted"); its 24-cell seed-42 off-orbit replacement runs ONLY through the exp_36 chain
# (worklog/worklog_yixun/exp_36_yaw_c4_spread_claude/yaw_c4_spread_chain.sh on branch exp-36-yaw-c4-spread). The
# generic exp_25 runner this kit reaches would quarantine (rename) the H35 OFF dumps that chain shares by name, so
# reserve, dry-run and launch are refused FIRST, with rc 1 -- before the config, any log, the helper, an
# orchestration, a reservation, the runner or nvidia-smi. Two modes remain, for inspection and cleanup:
#   [EXPECT_KIT_COMMIT=<commit>] bash exp35_phase_ii.sh status [--orch <orchestration directory>]
#   EXPECT_KIT_COMMIT=<commit> bash exp35_phase_ii.sh release <H35> --orch <orchestration directory>
# <commit>: one holding exactly these kit bytes (the retirement commit or a later one; the exp_35 notebook names it).
# release keeps its rule that only the kit commit which reserved an orchestration releases it (the exp_35 notebook's
# 2026-11-06T15:33:03 entry records that nothing was ever reserved). History -- the retired forms were
#   reserve <H35> --lanes <0|1|0,1|1,0> --report <path> --report-sha256 <hex>   (its committed six-line report file:
#     exp35-phase-ii h35 / phase-i-state-sha256 / lanes / second-lane-decision / eta / prior-runs)
#   dry-run <H35> --orch <orchestration directory>   and   launch <H35> --orch <orchestration directory>
#
# status is read-only and writes no log; release tees its log as above.
# Exit: 0 done | 1 refused (a retired mode included) or failed (the reasons printed; every failure state retained for
# review) | 2 bad arguments.
# Never signals, kills or attaches to any process; never touches another holder's reservation; never relaunches.
# Written by the exp_35 phase-(ii) kit Coder (Claude Opus 5.5, max effort); retired by the exp_35 kit-retirement
# Coder (Claude Opus 5.5, max effort).
set -uo pipefail
MODE="${1:-}"
RETIRED_BODY="the exp_35 phase-(ii) kit is RETIRED -- Yixun superseded phase (ii) on 2026-11-06 15:30 EST ('Seeds"
RETIRED_BODY+=" 43-46 at these angles are NOT wanted'); its 24-cell seed-42 off-orbit replacement runs ONLY through the"
RETIRED_BODY+=" exp_36 chain (worklog/worklog_yixun/exp_36_yaw_c4_spread_claude/yaw_c4_spread_chain.sh on branch"
RETIRED_BODY+=" exp-36-yaw-c4-spread, as its yaw_c4_spread_command.md records), because the generic exp_25 runner this"
RETIRED_BODY+=" kit reaches would quarantine the H35 OFF dumps that chain shares by name (exp_36 code review III HIGH"
RETIRED_BODY+=" 2); nothing was logged, reserved, spawned or queried; status and release remain for inspection and"
RETIRED_BODY+=" cleanup."
case "$MODE" in reserve|dry-run|launch) echo "REFUSED (RETIRED): exp35_phase_ii.sh ${MODE} -- ${RETIRED_BODY}"; exit 1 ;; esac
# --- BEGIN REGISTERED CONFIG (the sandbox tests replace this block, and only this block) ---
PY=/home/yixunhu/miniconda3/envs/flac/bin/python
# --- END REGISTERED CONFIG ---
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HELPER="$HERE/exp35_phase_ii.py"
ENV=(HOME=/home/yixunhu USER=yixunhu LOGNAME=yixunhu SHELL=/bin/bash LANG=en_US.UTF-8 PYTHONDONTWRITEBYTECODE=1
     PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin)
USAGE="usage: [EXPECT_KIT_COMMIT=<sha>] bash $0 status [--orch DIR] | EXPECT_KIT_COMMIT=<sha> bash $0 release <H35>"
USAGE+=" --orch DIR (reserve, dry-run and launch are RETIRED)"
case "$MODE" in
  status) shift; exec env -i "${ENV[@]}" "$PY" -B "$HELPER" status --kit-commit "${EXPECT_KIT_COMMIT:-}" "$@" ;;
  release) [ "$#" -ge 2 ] || { echo "$USAGE"; exit 2; } ;;
  *) echo "$USAGE"; exit 2 ;;
esac
LOG="$HERE/orbitrir_yaw_evidence_$(date +%Y-%m-%d_%H-%M-%S)_phase_ii_${MODE}.log"
( set -o noclobber; : > "$LOG" ) 2>/dev/null || { echo "the log ${LOG} exists - refused"; exit 2; }
env -i "${ENV[@]}" "$PY" -B "$HELPER" "$MODE" --kit-commit "${EXPECT_KIT_COMMIT:-}" --h35 "$2" "${@:3}" 2>&1 \
  | tee -a "$LOG"
rc=${PIPESTATUS[0]}
echo "rc ${rc}; log ${LOG}" | tee -a "$LOG"
# the FINAL file's digest, footer included, printed outside the log (Codex P2K-1 NIT 1)
echo "log sha256 $(sha256sum < "$LOG" | cut -c1-64)  ${LOG} (the final file, printed outside it)"
exit "$rc"
