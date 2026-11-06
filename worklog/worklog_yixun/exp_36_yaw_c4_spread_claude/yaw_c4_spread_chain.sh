#!/usr/bin/env bash
# ============================================================================
# yaw_c4_spread_chain.sh -- exp_36 yaw_c4_spread: the fail-closed launch chains (plan_yaw_c4_spread.md r0 §2 "Run" as
# amended by its Codex plan review B4 / H5 / H6 and by amendment A1 of 2026-11-06, Yixun 15:30 / 15:32). One mode:
#   offorbit (chain I, A1 -- FIRST): the replays exp36_{YAW,P1,BF,CANON}_C4_S40000_K8_s42_th90 ->
#            depth_canon_collect.py --exp36-replay-check for EACH (in that order; each must exit exactly 0 with an equal
#            verdict) -> the 24 off-orbit s42 cells: exp35_{P1,BF,CANON}_OFF_S40000_K{1,8}_s42_th{135,225,315} and
#            exp25_YAW_C4_S40000_K{1,8}_s42_th{135,225,315};
#   A        (chain II): the replay exp36_YAW_C4_S40000_K8_s42_th90 -> its check -> the 24
#            exp25_YAW_C4_S40000_K{1,8}_s{43..46}_th{90,180,270} cells. A YAW replay already complete and verified at
#            this HEAD under this relevant tree is SKIPPED by the runner (its result consumed) and re-checked; else it
#            is run again first.
#   Every cell dump-free (--no-dump-blocks), no-flash, on --lanes 0,1 (A1 §4: both cards). Anything but an equal
#   check is a HARD STOP: nothing further runs, nothing is retried. Publication (--exp36-publish --exp36-set OFF|A)
#   is a SEPARATE command: this chain never publishes; P1 / B-F / CANON outputs stay in this worktree's farm.
#
# Usage:  EXPECT_HEAD=<H36, the 40-hex commit of this worktree> bash yaw_c4_spread_chain.sh {offorbit|A}
#   DRY_RUN=1  both runner steps as DRY_RUNs (every gate, no evaluation); no check is run.
# GPU: ALLOW_SHARED_GPU=0 and no co-tenant tuple are passed to the runner, whose lane gate waits for a card with no
#   foreign compute process; nothing is ever signalled.
# Records (Codex H6): this chain tees everything to <exp_36 folder>/yaw_c4_spread_<TS>_<mode>_chain.log and writes
#   one verdict per replay to <exp_36 folder>/yaw_c4_spread_<TS>_<mode>_verdict_<replay>.json; the runner keeps its run
#   log, plan JSON, no-flash pre-flight log and per-cell evaluator logs in its hard-coded worklog/worklog_yixun/
#   exp_25_depth_canon_claude/ (depth_canon_<runner TS>_*) -- both locations are kept; this log names the runner's.
# Resume: re-run the same command: the runner SKIPs every complete-and-verified cell, the replays are re-checked
#   under new verdicts, then the remaining cells run.
# Exit: 0 complete | 2 a gate refused | 3 the replay run failed or gave no plan | 4 REPLAY MISMATCH (rc 1 with a
#   mismatch verdict) | 5 REPLAY CHECK REFUSED (anything but rc 0 with an equal verdict) | 6 the production run did not
#   complete (re-run to resume).
# EXP36_CHAIN_HARNESS=<dir under /tmp holding a .exp36_chain_harness marker>: tests only -- EXP36_CHAIN_RUNNER /
#   EXP36_CHAIN_COLLECT (stubs in that dir) are honoured and the log / lock / verdicts move into it.
#
# Written by the exp_36 Coder (Claude Opus 5.5, max effort) per plan_yaw_c4_spread.md r0 + its Codex plan review + A1.
# ============================================================================
set -uo pipefail

REPO="$(git -C "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" rev-parse --show-toplevel)"
cd "$REPO" || exit 2
EXP36="worklog/worklog_yixun/exp_36_yaw_c4_spread_claude"
SELF="$EXP36/yaw_c4_spread_chain.sh"
EXP25="worklog/worklog_yixun/exp_25_depth_canon_claude"
RUNNER="$EXP25/depth_canon_eval_run.sh"
COLLECT="$EXP25/depth_canon_collect.py"
CANON_CKPTS="$EXP25/depth_canon_2026-10-26_07-44-41_exp25_CANON_checkpoints.sha256"   # A1 §3, tracked at H35
CANON_CKPTS_SHA256="1e4814472b72375d6f3f42b3773f7bdb5d137719f882d00aef0387b770218cba"
PY="/home/yixunhu/miniconda3/envs/flac/bin/python"
LANES="0,1"
C4="C4_S40000_K8_s42_th90"
DRY_RUN="${DRY_RUN:-0}"; EXPECT_HEAD="${EXPECT_HEAD:-}"; HARNESS="${EXP36_CHAIN_HARNESS:-}"
TS="$(date '+%Y-%m-%d_%H-%M-%S')"
LOGDIR="$EXP36"; PLAN_RE="^${EXP25}/depth_canon_[0-9_-]+_eval_plan\.json\$"

die() { echo "$*"; exit 2; }

[ $# -eq 1 ] || die "this chain takes exactly one mode argument (offorbit | A); EXPECT_HEAD / DRY_RUN come from the environment - abort"
MODE="$1"; REPLAYS=(); CELLS=""
case "$MODE" in
  offorbit)
    REPLAYS=("exp36_YAW_${C4}" "exp36_P1_${C4}" "exp36_BF_${C4}" "exp36_CANON_${C4}"); RPL_NODUMP="YAW-C4S,RPL-S42"
    for a in P1 BF CANON; do for k in 1 8; do for d in 135 225; do
      CELLS+="${CELLS:+,}exp35_${a}_OFF_S40000_K${k}_s42_th${d}"; done; done; done
    for a in P1 BF CANON; do for k in 1 8; do CELLS+=",exp35_${a}_OFF_S40000_K${k}_s42_th315"; done; done
    for k in 1 8; do for d in 135 225 315; do CELLS+=",exp25_YAW_C4_S40000_K${k}_s42_th${d}"; done; done
    CELLS_NODUMP="OFF-P1,OFF-BF,OFF-CANON,OFF-S42"; SET="OFF" ;;
  A)
    REPLAYS=("exp36_YAW_${C4}"); RPL_NODUMP="YAW-C4S"
    for k in 1 8; do for s in 43 44 45 46; do for d in 90 180 270; do
      CELLS+="${CELLS:+,}exp25_YAW_C4_S40000_K${k}_s${s}_th${d}"; done; done; done
    CELLS_NODUMP="YAW-C4S"; SET="A" ;;
  *) die "unknown mode '${MODE}' (offorbit | A) - abort" ;;
esac
case "$DRY_RUN" in 0|1) ;; *) die "DRY_RUN must be 0 or 1 (got '${DRY_RUN}') - abort" ;; esac
case "${ALLOW_SHARED_GPU:-0}" in 0) ;; *) die "ALLOW_SHARED_GPU must be unset or 0: exp_36 runs on exclusive cards - abort" ;; esac
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
LOG="$LOGDIR/yaw_c4_spread_${TS}_${MODE}_chain.log"
exec > >(tee -a "$LOG") 2>&1
echo "=== exp_36 yaw_c4_spread chain ${MODE} -- ${TS} -- HEAD $(git rev-parse HEAD) -- dry_run=${DRY_RUN} -- harness=$([ -n "$HARNESS" ] && echo true || echo false) ==="
[[ "$EXPECT_HEAD" =~ ^[0-9a-f]{40}$ ]] || die "EXPECT_HEAD must be the full 40-hex commit H36 (got '${EXPECT_HEAD}') - abort"
[ "$(git rev-parse HEAD)" = "$EXPECT_HEAD" ] || die "HEAD PIN FAILED: HEAD $(git rev-parse HEAD) != EXPECT_HEAD ${EXPECT_HEAD} - abort"
[ -z "$(git status --porcelain --untracked-files=all -- "$SELF")" ] && git ls-files --error-unmatch -- "$SELF" >/dev/null 2>&1 \
  || die "the chain ${SELF} is not committed at HEAD (the runner binds it into its campaign digest) - abort"
got="$(sha256sum -- "$CANON_CKPTS" 2>/dev/null | awk '{print $1}')"
[ "$got" = "$CANON_CKPTS_SHA256" ] || die "the CANON checkpoint manifest ${CANON_CKPTS} hashes to '${got}', not ${CANON_CKPTS_SHA256} - abort"
[ -x "$PY" ] || die "the flac interpreter ${PY} is missing - abort"
TMPD="$(mktemp -d)" || die "mktemp failed - abort"
trap 'rm -rf -- "$TMPD"' EXIT

runner() {   # <capture> <runner args...>: the runner with the chain's fixed environment; its exit status in RC
  local out="$1"; shift
  echo "+ ALLOW_SHARED_GPU=0 DRY_RUN=${DRY_RUN} EXPECT_HEAD=${EXPECT_HEAD} bash ${RUNNER} $*"
  env -u EXP35_APPROVED_COTENANTS -u EXP35_APPROVED_COTENANTS_SHA256 ALLOW_SHARED_GPU=0 DRY_RUN="$DRY_RUN" \
    EXPECT_HEAD="$EXPECT_HEAD" bash "$RUNNER" "$@" --lanes "$LANES" --canon-ckpt-manifest "$CANON_CKPTS" 2>&1 | tee "$out"
  RC="${PIPESTATUS[0]}"
}

RPL_CSV="$(IFS=,; echo "${REPLAYS[*]}")"
echo "--- step 1/3: the replay(s) ${RPL_CSV} on GPUs ${LANES}, dumps off ---"
runner "$TMPD/step1" --only "$RPL_CSV" --no-dump-blocks "$RPL_NODUMP"
[ "$RC" -eq 0 ] || { echo "HARD STOP: the replay run exited ${RC}; no replay check, no production cell"; exit 3; }
mapfile -t PLANS < <(sed -n 's/^plan: [0-9][0-9]* cell(s) -> \(.*\)$/\1/p' "$TMPD/step1")
{ [ "${#PLANS[@]}" -eq 1 ] && [[ "${PLANS[0]}" =~ $PLAN_RE ]] && [ -f "${PLANS[0]}" ]; } \
  || { echo "HARD STOP: the replay run named no single plan JSON (${PLANS[*]:-none}); no replay check, no production cell"; exit 3; }
PLAN="${PLANS[0]}"
RTS="$(sed -n 's/^=== exp_25 depth_canon eval campaign -- \([0-9_-]*\) -- .*/\1/p' "$TMPD/step1" | head -n 1)"
echo "replay run plan ${PLAN} (sha256 $(sha256sum -- "$PLAN" | awk '{print $1}')); runner records: ${EXP25}/depth_canon_${RTS}_{eval_run.log,no_flash_preflight.log,eval_<cell>.log}"

YAW_VERDICT=""
for r in "${REPLAYS[@]}"; do
  VERDICT="$LOGDIR/yaw_c4_spread_${TS}_${MODE}_verdict_${r}.json"
  CHECK=(--exp36-replay-check --exp36-replay "$r" --pin "$EXPECT_HEAD" --plan-json "$PLAN" --verdict-out "$VERDICT"
         --canon-ckpt-manifest "$CANON_CKPTS")
  if [ "$DRY_RUN" = 1 ]; then
    echo "--- step 2/3 (DRY_RUN: not run): CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 ${PY} ${COLLECT} ${CHECK[*]} ---"
    continue
  fi
  echo "--- step 2/3: the replay check of ${r} (a hard stop) ---"
  CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 "$PY" "$COLLECT" "${CHECK[@]}"; RC=$?
  RESULT="$(jq -r '.result' "$VERDICT" 2>/dev/null || echo none)"   # the exit code AND the verdict must agree
  case "${RC}:${RESULT}" in
    0:equal) echo "replay check EQUAL: ${r}, verdict ${VERDICT} (sha256 $(sha256sum -- "$VERDICT" | awk '{print $1}'))" ;;
    1:mismatch) echo "HARD STOP: REPLAY MISMATCH of ${r} (verdict ${VERDICT}); nothing further runs -- report to Yixun"; exit 4 ;;
    *) echo "HARD STOP: REPLAY CHECK REFUSED for ${r} (rc ${RC}, verdict '${RESULT}'); nothing further runs -- report to Yixun"
       exit 5 ;;
  esac
  [ "$r" = "exp36_YAW_${C4}" ] && YAW_VERDICT="$VERDICT"
done

echo "--- step 3/3: the production cells (${MODE}) on GPUs ${LANES}, dumps off ---"
runner "$TMPD/step3" --only "$CELLS" --no-dump-blocks "$CELLS_NODUMP"
[ "$RC" -eq 0 ] || { echo "the production run exited ${RC}: not complete -- re-run this chain to resume"; exit 6; }
if [ "$DRY_RUN" = 1 ]; then echo "=== DRY_RUN complete: nothing evaluated, no replay check, nothing published ==="; exit 0; fi
echo "=== CHAIN COMPLETE (${MODE}): every replay EQUAL, the production cells complete and verified. Publication of the YAW"
echo "    cells into the exp-25 worktree is a SEPARATE command:"
echo "    CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 ${PY} ${COLLECT} --exp36-publish --exp36-set ${SET} --pin ${EXPECT_HEAD} --exp36-verdict ${YAW_VERDICT}"
exit 0
