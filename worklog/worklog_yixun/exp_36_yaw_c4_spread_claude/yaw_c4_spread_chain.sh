#!/usr/bin/env bash
# ============================================================================
# yaw_c4_spread_chain.sh -- exp_36 yaw_c4_spread: the ONE fail-closed launch chain (plan_yaw_c4_spread.md r0 §2 "Run"
# as amended by its Codex plan review B4 / H5 / H6):
#   (1) the replay exp36_YAW_C4_S40000_K8_s42_th90 through depth_canon_eval_run.sh (--lanes 1, dumps off) ->
#   (2) depth_canon_collect.py --exp36-replay-check: the replay vs the archived exp_25 cell; it writes the SHA-bound
#       verdict -> (3) the 24 exp25_YAW_C4_S40000_K{1,8}_s{43..46}_th{90,180,270} cells (--lanes 1, dumps off),
#   ONLY if (2) exited exactly 0. Anything else is a HARD STOP: nothing further runs, nothing is retried.
#   Publication (depth_canon_collect.py --exp36-publish) is a SEPARATE command: this chain never publishes.
#
# Usage:  EXPECT_HEAD=<H36, the 40-hex commit of this worktree> bash yaw_c4_spread_chain.sh
#   DRY_RUN=1  steps (1) and (3) as runner DRY_RUNs (every gate, no evaluation); step (2) is not run.
# GPU (Codex H5): lane 1 only; ALLOW_SHARED_GPU=0 and no co-tenant tuple are passed to the runner, whose lane gate
#   waits for a card with no foreign compute process; nothing is ever signalled.
# Records (Codex H6): this chain tees everything to <exp_36 folder>/yaw_c4_spread_<TS>_chain.log and writes the
#   verdict to <exp_36 folder>/yaw_c4_spread_<TS>_replay_verdict.json; the runner keeps its run log, plan JSON,
#   no-flash pre-flight log and per-cell evaluator logs in its hard-coded worklog/worklog_yixun/
#   exp_25_depth_canon_claude/ (depth_canon_<runner TS>_*) -- both locations are kept; this log names the runner's.
# Resume: re-run the same command: the runner SKIPs every complete-and-verified cell, the replay is re-checked under a
#   new verdict, then the remaining cells run.
# Exit: 0 the 24 complete | 2 a gate refused | 3 the replay run failed or gave no plan | 4 REPLAY MISMATCH (rc 1 with
#   a mismatch verdict) | 5 REPLAY CHECK REFUSED (anything else but rc 0 with an equal verdict) | 6 the 24-cell run
#   did not complete (re-run to resume).
# EXP36_CHAIN_HARNESS=<dir under /tmp holding a .exp36_chain_harness marker>: tests only -- EXP36_CHAIN_RUNNER /
#   EXP36_CHAIN_COLLECT (stubs in that dir) are honoured and the log / lock / verdict move into it.
#
# Written by the exp_36 Coder (Claude Opus 5.5, max effort) per plan_yaw_c4_spread.md r0 + its Codex plan review.
# ============================================================================
set -uo pipefail

REPO="$(git -C "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" rev-parse --show-toplevel)"
cd "$REPO" || exit 2
EXP36="worklog/worklog_yixun/exp_36_yaw_c4_spread_claude"
SELF="$EXP36/yaw_c4_spread_chain.sh"
EXP25="worklog/worklog_yixun/exp_25_depth_canon_claude"
RUNNER="$EXP25/depth_canon_eval_run.sh"
COLLECT="$EXP25/depth_canon_collect.py"
PY="/home/yixunhu/miniconda3/envs/flac/bin/python"
LANE=1
REPLAY="exp36_YAW_C4_S40000_K8_s42_th90"
CELLS24=""
for k in 1 8; do for s in 43 44 45 46; do for d in 90 180 270; do
  CELLS24+="${CELLS24:+,}exp25_YAW_C4_S40000_K${k}_s${s}_th${d}"
done; done; done
DRY_RUN="${DRY_RUN:-0}"; EXPECT_HEAD="${EXPECT_HEAD:-}"; HARNESS="${EXP36_CHAIN_HARNESS:-}"
TS="$(date '+%Y-%m-%d_%H-%M-%S')"
LOGDIR="$EXP36"; PLAN_RE="^${EXP25}/depth_canon_[0-9_-]+_eval_plan\.json\$"

die() { echo "$*"; exit 2; }

[ $# -eq 0 ] || die "this chain takes no argument (EXPECT_HEAD / DRY_RUN come from the environment) - abort"
case "$DRY_RUN" in 0|1) ;; *) die "DRY_RUN must be 0 or 1 (got '${DRY_RUN}') - abort" ;; esac
case "${ALLOW_SHARED_GPU:-0}" in 0) ;; *) die "ALLOW_SHARED_GPU must be unset or 0: exp_36 runs on an exclusive GPU 1 - abort" ;; esac
[ -z "${EXP35_APPROVED_COTENANTS:-}${EXP35_APPROVED_COTENANTS_SHA256:-}" ] || die "no co-tenant tuple for exp_36 - abort"
if [ -n "$HARNESS" ]; then
  HREAL="$(readlink -f -- "$HARNESS" 2>/dev/null || true)"
  case "$HREAL" in /tmp/?*) ;; *) die "EXP36_CHAIN_HARNESS must be a directory under /tmp (got '${HARNESS}') - abort" ;; esac
  [ -d "$HREAL" ] && [ -f "$HREAL/.exp36_chain_harness" ] || die "EXP36_CHAIN_HARNESS '${HARNESS}' lacks its marker - abort"
  RUNNER="${EXP36_CHAIN_RUNNER:-}"; COLLECT="${EXP36_CHAIN_COLLECT:-}"
  for f in "$RUNNER" "$COLLECT"; do
    case "$(readlink -f -- "$f" 2>/dev/null)" in "$HREAL"/?*) [ -f "$f" ] ;; *) false ;; esac \
      || die "HARNESS: EXP36_CHAIN_RUNNER / EXP36_CHAIN_COLLECT must be files in ${HREAL} - abort"
  done
  LOGDIR="$HREAL"; PLAN_RE="^${HREAL}/[^/]+\.json\$"
elif [ -n "${DEPTH_CANON_EVAL_HARNESS:-}${DEPTH_CANON_EVAL_PY:-}${DEPTH_CANON_CELLS:-}${EXP36_CHAIN_RUNNER:-}${EXP36_CHAIN_COLLECT:-}" ]; then
  die "a test-harness variable is set outside the chain harness - abort"
fi

exec 9>"$LOGDIR/.yaw_c4_spread_chain.lock" || die "cannot open ${LOGDIR}/.yaw_c4_spread_chain.lock - abort"
flock -n 9 || die "another exp_36 chain holds ${LOGDIR}/.yaw_c4_spread_chain.lock - abort"
LOG="$LOGDIR/yaw_c4_spread_${TS}_chain.log"
exec > >(tee -a "$LOG") 2>&1
echo "=== exp_36 yaw_c4_spread chain -- ${TS} -- HEAD $(git rev-parse HEAD) -- dry_run=${DRY_RUN} -- harness=$([ -n "$HARNESS" ] && echo true || echo false) ==="
[[ "$EXPECT_HEAD" =~ ^[0-9a-f]{40}$ ]] || die "EXPECT_HEAD must be the full 40-hex commit H36 (got '${EXPECT_HEAD}') - abort"
[ "$(git rev-parse HEAD)" = "$EXPECT_HEAD" ] || die "HEAD PIN FAILED: HEAD $(git rev-parse HEAD) != EXPECT_HEAD ${EXPECT_HEAD} - abort"
[ -z "$(git status --porcelain --untracked-files=all -- "$SELF")" ] && git ls-files --error-unmatch -- "$SELF" >/dev/null 2>&1 \
  || die "the chain ${SELF} is not committed at HEAD (the runner binds it into its campaign digest) - abort"
[ -x "$PY" ] || die "the flac interpreter ${PY} is missing - abort"
TMPD="$(mktemp -d)" || die "mktemp failed - abort"
trap 'rm -rf -- "$TMPD"' EXIT

runner() {   # <capture> <runner args...>: the runner with the chain's fixed environment; its exit status in RC
  local out="$1"; shift
  echo "+ ALLOW_SHARED_GPU=0 DRY_RUN=${DRY_RUN} EXPECT_HEAD=${EXPECT_HEAD} bash ${RUNNER} $*"
  env -u EXP35_APPROVED_COTENANTS -u EXP35_APPROVED_COTENANTS_SHA256 ALLOW_SHARED_GPU=0 DRY_RUN="$DRY_RUN" \
    EXPECT_HEAD="$EXPECT_HEAD" bash "$RUNNER" "$@" 2>&1 | tee "$out"
  RC="${PIPESTATUS[0]}"
}

echo "--- step 1/3: the replay ${REPLAY} on GPU ${LANE}, dumps off ---"
runner "$TMPD/step1" --only "$REPLAY" --no-dump-blocks YAW-C4S --lanes "$LANE"
[ "$RC" -eq 0 ] || { echo "HARD STOP: the replay run exited ${RC}; no replay check, no 24 cells"; exit 3; }
mapfile -t PLANS < <(sed -n 's/^plan: [0-9][0-9]* cell(s) -> \(.*\)$/\1/p' "$TMPD/step1")
{ [ "${#PLANS[@]}" -eq 1 ] && [[ "${PLANS[0]}" =~ $PLAN_RE ]] && [ -f "${PLANS[0]}" ]; } \
  || { echo "HARD STOP: the replay run named no single plan JSON (${PLANS[*]:-none}); no replay check, no 24 cells"; exit 3; }
PLAN="${PLANS[0]}"
RTS="$(sed -n 's/^=== exp_25 depth_canon eval campaign -- \([0-9_-]*\) -- .*/\1/p' "$TMPD/step1" | head -n 1)"
echo "replay run plan ${PLAN} (sha256 $(sha256sum -- "$PLAN" | awk '{print $1}')); runner records: ${EXP25}/depth_canon_${RTS}_{eval_run.log,no_flash_preflight.log,eval_${REPLAY}.log}"

VERDICT="$LOGDIR/yaw_c4_spread_${TS}_replay_verdict.json"
CHECK=(--exp36-replay-check --pin "$EXPECT_HEAD" --plan-json "$PLAN" --verdict-out "$VERDICT")
if [ "$DRY_RUN" = 1 ]; then
  echo "--- step 2/3 (DRY_RUN: not run): CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 ${PY} ${COLLECT} ${CHECK[*]} ---"
else
  echo "--- step 2/3: the replay check (the hard stop) ---"
  CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 "$PY" "$COLLECT" "${CHECK[@]}"; RC=$?
  RESULT="$(jq -r '.result' "$VERDICT" 2>/dev/null || echo none)"   # the exit code AND the verdict must agree
  case "${RC}:${RESULT}" in
    0:equal) echo "replay check EQUAL: verdict ${VERDICT} (sha256 $(sha256sum -- "$VERDICT" | awk '{print $1}'))" ;;
    1:mismatch) echo "HARD STOP: REPLAY MISMATCH (verdict ${VERDICT}); the 24 cells are NOT run -- report to Yixun"; exit 4 ;;
    *) echo "HARD STOP: REPLAY CHECK REFUSED (rc ${RC}, verdict '${RESULT}'); the 24 cells are NOT run -- report to Yixun"
       exit 5 ;;
  esac
fi

echo "--- step 3/3: the 24 cells on GPU ${LANE}, dumps off ---"
runner "$TMPD/step3" --only "$CELLS24" --no-dump-blocks YAW-C4S --lanes "$LANE"
[ "$RC" -eq 0 ] || { echo "the 24-cell run exited ${RC}: not complete -- re-run this chain to resume"; exit 6; }
if [ "$DRY_RUN" = 1 ]; then echo "=== DRY_RUN complete: nothing evaluated, no replay check, nothing published ==="; exit 0; fi
echo "=== CHAIN COMPLETE: the replay EQUAL, the 24 cells complete and verified. Publication is a SEPARATE command:"
echo "    CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 ${PY} ${COLLECT} --exp36-publish --pin ${EXPECT_HEAD} --exp36-verdict ${VERDICT}"
exit 0
