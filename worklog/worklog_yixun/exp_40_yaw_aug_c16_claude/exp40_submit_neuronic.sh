#!/usr/bin/env bash
# ============================================================================
# exp40_submit_neuronic.sh -- submit exp_40 C16 on neuronic, exactly once (plan r2 §4.4).
# NON-PRIMARY path (plan D1 b/c). Run by Yixun on a neuronic login node from a clean
# checkout of branch exp-40-yaw-aug-c16 at EXPECT_SHA (pushed).
#
#   MODE=PILOT|FULL RUNG=2x32|8x8 RUN_ID=exp40_C16_neuronic_<rung>[_r<n>] EXPECT_SHA=<40-hex>
#   [DRY_RUN=1] [EXP40_NAS_RELAXATION=post-run-publish  (FULL only; plan D4)]
#   bash worklog/worklog_yixun/exp_40_yaw_aug_c16_claude/exp40_submit_neuronic.sh
#
# Gates at submission: HEAD == EXPECT_SHA, reachable from origin/exp-40-yaw-aug-c16, clean
# closure; the AR manifest receipt for this cluster's dataset; FULL: a PASS PILOT receipt
# at the same SHA/rung, --time from its measured rate, refused above the partition MaxTime.
# The run tree is built here (a queued job is immune to later checkout changes); the
# submission is one locked transaction through exp40_kit.py neuronic-submit.
# Written by the exp_40 Coder seat (Claude Opus 5.5) per plan r2 §4.4.
# ============================================================================
set -uo pipefail
die() { echo "REFUSED: $*"; exit 2; }
SELF_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
REPO="$(git -C "$SELF_DIR" rev-parse --show-toplevel)" || die "not inside a git checkout"
EXPREL="worklog/worklog_yixun/exp_40_yaw_aug_c16_claude"
PY="/n/fs/gatrdp/envs/flac/bin/python"
MODE="${MODE:-}"; RUNG="${RUNG:-}"; RUN_ID="${RUN_ID:-}"; EXPECT_SHA="${EXPECT_SHA:-}"
DRY_RUN="${DRY_RUN:-0}"; PARTITION="${PARTITION:-all}"
case "$MODE" in PILOT|FULL) ;; *) die "MODE must be PILOT or FULL" ;; esac
case "$RUNG" in 2x32) NGPU=2; CPUS=22; MEM=64G ;; 8x8) NGPU=8; CPUS=64; MEM=108G ;; *) die "RUNG must be 2x32 or 8x8" ;; esac
[[ "$EXPECT_SHA" =~ ^[0-9a-f]{40}$ ]] || die "EXPECT_SHA must be the full 40-hex commit"
case "$DRY_RUN" in 0|1) ;; *) die "DRY_RUN must be 0 or 1" ;; esac
[ -x "$PY" ] || die "pinned interpreter $PY missing"
cd "$REPO" || die "cd $REPO"

[ "$(git rev-parse HEAD)" = "$EXPECT_SHA" ] || die "HEAD $(git rev-parse HEAD) != EXPECT_SHA"
git fetch -q origin exp-40-yaw-aug-c16 || die "git fetch origin exp-40-yaw-aug-c16 failed"
git merge-base --is-ancestor "$EXPECT_SHA" FETCH_HEAD || die "EXPECT_SHA is not pushed to origin/exp-40-yaw-aug-c16"
DIRTY="$(git status --porcelain --untracked-files=all -- train.py defaults.ini src data/AR "$EXPREL"/*.py "$EXPREL"/*.sh "$EXPREL"/*.sbatch "$EXPREL"/*.json worklog/worklog_yixun/exp_15_yaw_aug_claude/FLAC_AR_YAWAUG.json)"
[ -z "$DIRTY" ] || die "the closure is not clean: $DIRTY"

SAVE_ROOT="$REPO/outputs_FLAC/exp40_yaw_aug_c16"; RECEIPTS="$REPO/$EXPREL/receipts"
mkdir -p "$SAVE_ROOT/runtrees" "$RECEIPTS" || die "dirs"
exec 9>"$SAVE_ROOT/.submit.lock" || die "lock"; flock -n 9 || die "another submission holds $SAVE_ROOT/.submit.lock"

# --- this cluster's dataset must reproduce the committed full AR manifest (plan §14 H2) ---- #
ARRCPT="$RECEIPTS/ar_verify_neuronic.json"
[ -f "$ARRCPT" ] || die "no $ARRCPT: first run (CPU node, ~1 h): $PY $EXPREL/exp40_kit.py ar-verify --ar-root \$(readlink -f AcousticRooms) --manifest $EXPREL/exp40_ar_manifest.json --out $ARRCPT"
"$PY" -c "import json,sys; r=json.load(open(sys.argv[1])); m=json.load(open(sys.argv[2])); sys.exit(0 if r['verdict']=='PASS' and r['got_digest']==m['digest'] else 1)" \
  "$ARRCPT" "$EXPREL/exp40_ar_manifest.json" || die "$ARRCPT is not a PASS of the committed manifest"

# --- the sealed HF hub (plan §14 H3), built once from this cluster's cache ------------------ #
[ -d "$SAVE_ROOT/hf_sealed" ] || "$PY" "$EXPREL/exp40_kit.py" build-hf-sealed --src-hub /n/fs/gatrdp/hf_cache/hub \
  --dest "$SAVE_ROOT/hf_sealed" || die "sealed HF hub (the pinned DINOv3 snapshot must be in /n/fs/gatrdp/hf_cache/hub)"

TS="$(date '+%Y-%m-%d_%H-%M-%S')"
EXEC_ID="$RUN_ID"; [ "$MODE" = PILOT ] && EXEC_ID="${RUN_ID}_pilot_${TS}"
NS="$EXEC_ID"
RUNTREE="$SAVE_ROOT/runtrees/${NS}__${EXPECT_SHA:0:12}"
AR_REAL="$(readlink -f AcousticRooms)" || die "AcousticRooms link missing"
VAE_REAL="$(readlink -f weights/FLAC/VAE.safetensors)" || die "VAE missing"
MAXTIME="$(sinfo -h -p "$PARTITION" -o %l | head -1)" || die "sinfo failed"
[ -n "$MAXTIME" ] || die "sinfo returned no MaxTime for partition $PARTITION"
if [ "$MODE" = FULL ]; then
  PILOT_RCPT="$RECEIPTS/pilot_neuronic_${RUNG}_${EXPECT_SHA}.json"
  [ -f "$PILOT_RCPT" ] || die "no PASS PILOT receipt $PILOT_RCPT (run MODE=PILOT first)"
  TMIN="$("$PY" "$EXPREL/exp40_kit.py" neuronic-time --receipt "$PILOT_RCPT" --maxtime "$MAXTIME")" || die "time request refused"
else
  TMIN=120
fi
echo "partition $PARTITION MaxTime $MAXTIME; requesting ${TMIN} min"
if [ "$DRY_RUN" = 1 ]; then RUNTREE="${RUNTREE}__dryrun_${TS}"; fi
[ -e "$RUNTREE" ] || "$PY" "$EXPREL/exp40_kit.py" build-runtree --repo "$REPO" --sha "$EXPECT_SHA" \
  --dest "$RUNTREE" --link "AcousticRooms=$AR_REAL" --link "weights/FLAC/VAE.safetensors=$VAE_REAL" || die "run tree"
"$PY" "$EXPREL/exp40_kit.py" verify-runtree --repo "$REPO" --sha "$EXPECT_SHA" --tree "$RUNTREE" \
  --link "AcousticRooms=$AR_REAL" --link "weights/FLAC/VAE.safetensors=$VAE_REAL" || die "run tree verification"

SPEC="$("$PY" - "$PARTITION" "$NGPU" "$CPUS" "$MEM" "$TMIN" "$REPO" "$EXPREL" "$RUNTREE" "$MODE" "$RUNG" "$RUN_ID" "$EXPECT_SHA" "$SAVE_ROOT" "$TS" <<'PYEOF'
import json, sys
part, ngpu, cpus, mem, tmin, repo, exprel, tree, mode, rung, run_id, sha, save, ts = sys.argv[1:]
exports = {"EXP40_RUNTREE": tree, "EXP40_REPO": repo, "EXP40_MODE": mode, "EXP40_RUNG": rung,
           "EXP40_EXECUTION_ID": (run_id if mode == "FULL" else f"{run_id}_pilot_{ts}"),
           "EXP40_RUN_ID": run_id, "EXP40_EXPECT_SHA": sha, "EXP40_SAVE_ROOT": save, "EXP40_STAMP": ts}
print(json.dumps({"script": f"{tree}/{exprel}/exp40_train.sbatch",
                  "options": [f"--partition={part}", "--nodes=1", "--ntasks=1",
                              f"--gres=gpu:l40:{ngpu}", f"--cpus-per-task={cpus}", f"--mem={mem}",
                              f"--time={tmin}", f"--output={repo}/{exprel}/slurm_train_%x_%j.out",
                              "--export=ALL," + ",".join(f"{k}={v}" for k, v in exports.items())]}))
PYEOF
)" || die "sbatch spec"
EXTRA=(); [ "$DRY_RUN" = 1 ] && EXTRA+=(--dry-run)
"$PY" "$EXPREL/exp40_kit.py" neuronic-submit --states-dir "$RECEIPTS/neuronic_states" --run-id "$RUN_ID" \
  --mode "$MODE" --execution-id "$EXEC_ID" --probe-receipt "$RECEIPTS/claim_probe_neuronic.json" \
  --sbatch-json "$SPEC" "${EXTRA[@]}" | tee -a "$REPO/$EXPREL/yaw_aug_c16_neuronic_submissions.log"
exit "${PIPESTATUS[0]}"
