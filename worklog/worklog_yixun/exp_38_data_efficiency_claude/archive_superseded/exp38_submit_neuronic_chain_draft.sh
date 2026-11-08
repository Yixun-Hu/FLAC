#!/usr/bin/env bash
# ============================================================================
# exp38_submit_neuronic.sh — submit one exp_38 run on neuronic as a chain of legs
# (plan_data_efficiency.md §10 item 7). Run on the neuronic login node from the
# exp-38-data-efficiency checkout:
#
#   RUN=de_bf_f050 EXPECT_SHA=$(git rev-parse HEAD) TIME=24:00:00 LEG_STEPS=10000 \
#     bash worklog/worklog_yixun/exp_38_data_efficiency_claude/exp38_submit_neuronic.sh
#
#   RUN          exp_38 run id (an _r8x8 id also needs ALLOW_NONMATCHED_RUNG=1)
#   EXPECT_SHA   must equal HEAD; HEAD must be pushed (origin/exp-38-data-efficiency contains it)
#   TIME         Slurm --time per leg (Yixun's partition limit)
#   LEG_STEPS    steps per leg, a multiple of 2500 (default 40000 = one monolithic job)
#   START        first leg start (default 0); recovery after a failed leg = the same
#                command with START=<last audited boundary>
#   ATTN_BACKEND flash (default) | sdpa
#   DRY_RUN=1    print the chain, submit nothing (no squeue / sbatch call)
#
# The legs are submitted at once as an afterok chain; each leg resumes from the
# deterministic boundary checkpoint the previous leg wrote and audits it first.
# A same-name job already queued/running refuses the submission; a scheduler
# query failure is fatal (never "no job found, submit").
# ============================================================================
set -uo pipefail

main() {
  local REPO; REPO="$(git rev-parse --show-toplevel)" || exit 3
  cd "$REPO" || exit 3
  local EXP="worklog/worklog_yixun/exp_38_data_efficiency_claude"
  die() { echo "EXP38 SUBMIT REFUSED: $1" >&2; exit "${2:-2}"; }
  local RUN="${RUN:-}" EXPECT_SHA="${EXPECT_SHA:-}" TIME="${TIME:-}" LEG_STEPS="${LEG_STEPS:-40000}"
  local START="${START:-0}" ATTN_BACKEND="${ATTN_BACKEND:-flash}" ALLOW="${ALLOW_NONMATCHED_RUNG:-0}"
  local DRY="${DRY_RUN:-0}"
  [[ "$RUN" =~ ^de_(yaw|bf|p1)_f(025|050|075)(_r8x8)?$ ]] || die "RUN '${RUN}' is not an exp_38 run id"
  [[ "$TIME" =~ ^([0-9]+-)?[0-9]{1,3}:[0-9]{2}:[0-9]{2}$ ]] || die "TIME must be a Slurm time (e.g. 24:00:00 or 3-00:00:00)"
  [[ "$LEG_STEPS" =~ ^[0-9]+$ ]] && [[ "$START" =~ ^[0-9]+$ ]] || die "LEG_STEPS / START must be integers"
  case "$ATTN_BACKEND" in flash|sdpa) ;; *) die "ATTN_BACKEND must be flash|sdpa";; esac
  local HEAD; HEAD="$(git rev-parse HEAD)"
  [ "$EXPECT_SHA" = "$HEAD" ] || die "EXPECT_SHA ${EXPECT_SHA:-unset} != HEAD ${HEAD}"
  [ -z "$(git status --porcelain --untracked-files=no)" ] || die "tracked tree is not clean"
  if [ "$DRY" != 1 ]; then
    git fetch -q origin exp-38-data-efficiency || die "git fetch failed" 3
    git merge-base --is-ancestor "$HEAD" FETCH_HEAD || die "HEAD ${HEAD} is not pushed to origin/exp-38-data-efficiency"
  fi

  local NGPU=2 CPUS=22 MEM=64G
  if [[ "$RUN" == *_r8x8 ]]; then
    [ "$ALLOW" = 1 ] || die "${RUN} is the cross-recipe 8x8 rung: set ALLOW_NONMATCHED_RUNG=1 deliberately"
    NGPU=8 CPUS=64 MEM=108G
  fi
  local PY="${EXP38_PYTHON:-python3}"
  local LEGS; LEGS="$("$PY" "$EXP/exp38_registry.py" legs --start "$START" --leg-steps "$LEG_STEPS")" || die "bad leg plan"

  local NAME_PREFIX="exp38-${RUN}"
  if [ "$DRY" != 1 ]; then
    local Q; Q="$(squeue -h -u "$USER" -o '%j' 2>&1)" || die "squeue failed: ${Q}" 3
    if grep -q "^${NAME_PREFIX}-leg" <<< "$Q"; then die "a ${NAME_PREFIX} leg is already queued or running"; fi
  fi
  local DEP="" s e JID SUBLOG="$EXP/exp38_neuronic_submissions.log"
  while read -r s e; do
    local ARGS=(--parsable --job-name="${NAME_PREFIX}-leg${e}" --gres="gpu:l40:${NGPU}" --cpus-per-task="$CPUS"
                --mem="$MEM" --time="$TIME" --chdir="$REPO"
                --output="$REPO/$EXP/slurm_train_%x_%j.out"
                --export="ALL,RUN=${RUN},EXPECT_SHA=${HEAD},LEG_START=${s},LEG_END=${e},ATTN_BACKEND=${ATTN_BACKEND},ALLOW_NONMATCHED_RUNG=${ALLOW}")
    [ -n "$DEP" ] && ARGS+=("--dependency=afterok:${DEP}" "--kill-on-invalid-dep=yes")
    if [ "$DRY" = 1 ]; then
      echo "DRYRUN sbatch ${ARGS[*]} $EXP/exp38_train.sbatch"
      DEP="<jid of leg ${e}>"
      continue
    fi
    JID="$(sbatch "${ARGS[@]}" "$EXP/exp38_train.sbatch")" || die "sbatch failed for leg [${s}, ${e}] (earlier legs of this chain may be queued: check squeue)" 3
    JID="${JID%%;*}"
    [[ "$JID" =~ ^[0-9]+$ ]] || die "sbatch returned '${JID}'" 3
    echo "$(date -Iseconds) ${RUN} leg [${s}, ${e}] job ${JID} dep ${DEP:-none} head ${HEAD} time ${TIME} backend ${ATTN_BACKEND}: sbatch ${ARGS[*]} $EXP/exp38_train.sbatch" | tee -a "$SUBLOG"
    DEP="$JID"
  done <<< "$LEGS"
  echo "submitted ${RUN} (${NGPU} GPUs per leg); submissions log ${SUBLOG}"
}

main "$@"
exit $?
