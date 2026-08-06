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
case_run "un-pinned arm refuses" 2 "TO-PIN-AFTER-P0" \
  -- DRYRUN=1 ARM=C8 "EXPECT_SHA=${HEAD_SHA}" "OUTPUT_ROOT=${OUT_ROOT}" "${REPO_ENV[@]}"
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
case_run "restart from the arm's own ckpt dir" 0 "ARGV PARITY OK" \
  -- "${SMOKE_ENV[@]}" ARM=C8 EXPECTED_STEP=5000 SMOKE_MAXSTEPS=6000 \
     "RESUME_CKPT=${SMOKE_RUN}/checkpoints/epoch=1-step=5000.ckpt"
case_run "restart from a non-ckpt sibling" 2 "may only resume a checkpoint from" \
  -- "${SMOKE_ENV[@]}" ARM=C8 EXPECTED_STEP=5000 SMOKE_MAXSTEPS=6000 "RESUME_CKPT=${SMOKE_RUN}/notes.txt"
case_run "restart MAXSTEPS<=step" 2 "must exceed the resume step" \
  -- "${SMOKE_ENV[@]}" ARM=C8 EXPECTED_STEP=5000 SMOKE_MAXSTEPS=30 \
     "RESUME_CKPT=${SMOKE_RUN}/checkpoints/epoch=1-step=5000.ckpt"
case_run "initial refuses an existing run dir" 2 "already exists" -- "${SMOKE_ENV[@]}" ARM=C8

echo "--- D. commit-binding / sbatch-only gates (REAL mode) ---"
# NOTE: no OUTPUT_ROOT here — under a (fake) SLURM_JOB_ID the launcher forces the
# production literal, and the commit gate aborts long before anything is written.
case_run "wrong EXPECT_SHA aborts" 2 "EXPECT_SHA" \
  -- ARM=C4L SMOKE=1 SMOKE_RUNG=16x4 SMOKE_MIN_FREE_MB=99000000 \
     EXPECT_SHA=0000000000000000000000000000000000000000 SLURM_JOB_ID=999999
case_run "real mode needs sbatch" 2 "must run under sbatch" \
  -- ARM=C4L SMOKE=1 SMOKE_RUNG=16x4 SMOKE_MIN_FREE_MB=99000000 "EXPECT_SHA=${HEAD_SHA}" "OUTPUT_ROOT=${OUT_ROOT}"

echo "--- E. semantic gate on a mislabelled config (temp copy; tracked tree untouched) ---"
FAKE_EXP="${TMP}/fakeexp"; mkdir -p "$FAKE_EXP"
cp "${EXPDIR}/FLAC_AR_BF_C4L.json" "${FAKE_EXP}/FLAC_AR_BF_C32.json"      # C4 orbit under the C32 name
expect_cmd "orbit mismatch rejected" 1 "ARM/CONFIG GATE" -- \
  $PY - "${FAKE_EXP}/FLAC_AR_BF_C32.json" C32 <<'PY'
import json, sys
cfg = json.load(open(sys.argv[1])); arm = sys.argv[2]
t = cfg.get("training", {}); bad = []
want = {"C4L": 4, "C8": 8, "C16": 16, "C32": 32}[arm]
angles = t.get("frame_avg_angles")
if not isinstance(angles, list) or len(angles) != want:
    bad.append(f"frame_avg_angles has {angles and len(angles)} entries (want {want})")
if bad:
    sys.exit("ARM/CONFIG GATE: " + "; ".join(bad))
PY
TRACKED_AFTER="$(git status --porcelain -- "$EXPDIR" src | sort)"
if [ "$TRACKED_BEFORE" = "$TRACKED_AFTER" ]; then
  echo "PASS  tracked tree unchanged by the suite (snapshot before == after)"; PASS=$((PASS+1))
else
  echo "FAIL  the suite changed tracked state:"; diff <(echo "$TRACKED_BEFORE") <(echo "$TRACKED_AFTER") | sed 's/^/        | /'
  FAIL=$((FAIL+1))
fi

echo "--- F. exit taxonomy, mocked (round-3 B5) ---"
mk_log() {  # $1 dest, $2 world size (0 = absent), $3 marker?, $4 oom?
  : > "$1"
  [ "$2" != "0" ] && echo "All distributed processes registered. Starting with $2 processes" >> "$1"
  [ "$3" = "yes" ] && echo '`Trainer.fit` stopped: `max_steps=40000` reached.' >> "$1"
  [ "$4" = "yes" ] && echo "torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 98.00 MiB" >> "$1"
  return 0
}
A="${TMP}/a.log"; B="${TMP}/b.log"
mk_log "$A" 4 yes no; cp "$A" "$B"
expect_cmd "class 0 complete" 0 "COMPLETE" -- $PY "$CLASSIFY" --rc 0 --tee-rc 0 --ngpu 4 --maxsteps 40000 --log "$A" --log-copy "$B"
mk_log "$A" 0 no no; cp "$A" "$B"
expect_cmd "class 6 world-size absent" 6 "WORLD-SIZE" -- $PY "$CLASSIFY" --rc 0 --tee-rc 0 --ngpu 4 --maxsteps 40000 --log "$A" --log-copy "$B"
mk_log "$A" 1 yes no; cp "$A" "$B"
expect_cmd "class 6 wrong world-size" 6 "reported [1]" -- $PY "$CLASSIFY" --rc 0 --tee-rc 0 --ngpu 4 --maxsteps 40000 --log "$A" --log-copy "$B"
mk_log "$A" 4 no yes; cp "$A" "$B"
expect_cmd "class 3 OOM on nonzero rc" 3 "OOM" -- $PY "$CLASSIFY" --rc 1 --tee-rc 0 --ngpu 4 --maxsteps 40000 --log "$A" --log-copy "$B"
mk_log "$A" 4 no no; cp "$A" "$B"
expect_cmd "class 4 missing marker" 4 "NO-MARKER" -- $PY "$CLASSIFY" --rc 0 --tee-rc 0 --ngpu 4 --maxsteps 40000 --log "$A" --log-copy "$B"
mk_log "$A" 4 yes no; cp "$A" "$B"; echo "divergent tail" >> "$B"
expect_cmd "class 7 logs differ" 7 "LOG-PROVENANCE" -- $PY "$CLASSIFY" --rc 0 --tee-rc 0 --ngpu 4 --maxsteps 40000 --log "$A" --log-copy "$B"
mk_log "$A" 4 yes no; cp "$A" "$B"; rm -f "$B"
expect_cmd "class 7 copy missing" 7 "missing log copy" -- $PY "$CLASSIFY" --rc 0 --tee-rc 0 --ngpu 4 --maxsteps 40000 --log "$A" --log-copy "$B"
mk_log "$A" 4 yes no; cp "$A" "$B"
expect_cmd "class 7 tee failed" 7 "tee exited" -- $PY "$CLASSIFY" --rc 0 --tee-rc 1 --ngpu 4 --maxsteps 40000 --log "$A" --log-copy "$B"
mk_log "$A" 4 no no; cp "$A" "$B"
expect_cmd "raw rc preserved" 9 "RUNTIME" -- $PY "$CLASSIFY" --rc 9 --tee-rc 0 --ngpu 4 --maxsteps 40000 --log "$A" --log-copy "$B"

echo "--- G. restart preflight depth, mocked checkpoints (round-3 B2) ---"
$PY - "$TMP" "${EXPDIR}/FLAC_AR_BF_C8.json" <<'PY'
import json, os, sys, torch
tmp, cfg_path = sys.argv[1], sys.argv[2]
cfg = json.load(open(cfg_path))
def ck(step=5000, config=cfg, opt=True, sched=True, ema=True):
    d = {"global_step": step, "epoch": 1, "model_config": config,
         "state_dict": {"diffusion.x": torch.zeros(1)},
         "optimizer_states": [{"state": {0: {"step": 1}} if opt else {},
                               "param_groups": [{"lr": 1e-5}]}],
         "lr_schedulers": [{"last_epoch": step}] if sched else []}
    if ema:
        d["state_dict"]["diffusion_ema.x"] = torch.zeros(1)
    return d
torch.save(ck(), os.path.join(tmp, "good.ckpt"))
torch.save(ck(step=4999), os.path.join(tmp, "wrongstep.ckpt"))
c4 = json.loads(json.dumps(cfg)); c4["training"]["frame_avg_angles"] = [0.0, 90.0, 180.0, 270.0]
torch.save(ck(config=c4), os.path.join(tmp, "wrongorbit.ckpt"))
torch.save(ck(opt=False), os.path.join(tmp, "stripped.ckpt"))
torch.save(ck(ema=False), os.path.join(tmp, "noema.ckpt"))
torch.save(ck(sched=False), os.path.join(tmp, "nosched.ckpt"))
torch.save(ck(step=45000), os.path.join(tmp, "past.ckpt"))
open(os.path.join(tmp, "empty.ckpt"), "wb").close()
print("synthetic checkpoints written")
PY
PRE=($PY "$PREFLIGHT" --config "${EXPDIR}/FLAC_AR_BF_C8.json" --max-steps 40000 --arm C8 --rung 16x4)
expect_cmd "preflight accepts a good ckpt" 0 "CKPT_SHA256" -- "${PRE[@]}" --ckpt "${TMP}/good.ckpt" --expected-step 5000
expect_cmd "preflight rejects a step mismatch" 2 "global_step" -- "${PRE[@]}" --ckpt "${TMP}/wrongstep.ckpt" --expected-step 5000
expect_cmd "preflight rejects a foreign orbit" 2 "embedded model_config" -- "${PRE[@]}" --ckpt "${TMP}/wrongorbit.ckpt" --expected-step 5000
expect_cmd "preflight rejects a stripped optimizer" 2 "optimizer state is CLEARED" -- "${PRE[@]}" --ckpt "${TMP}/stripped.ckpt" --expected-step 5000
expect_cmd "preflight rejects a missing EMA" 2 "no EMA weights" -- "${PRE[@]}" --ckpt "${TMP}/noema.ckpt" --expected-step 5000
expect_cmd "preflight rejects a missing scheduler" 2 "lr_schedulers" -- "${PRE[@]}" --ckpt "${TMP}/nosched.ckpt" --expected-step 5000
expect_cmd "preflight rejects a past-budget ckpt" 2 ">= max_steps" -- "${PRE[@]}" --ckpt "${TMP}/past.ckpt" --expected-step 45000
expect_cmd "preflight rejects an empty file" 2 "PREFLIGHT" -- "${PRE[@]}" --ckpt "${TMP}/empty.ckpt" --expected-step 5000
expect_cmd "preflight rejects a missing file" 2 "not found" -- "${PRE[@]}" --ckpt "${TMP}/nope.ckpt" --expected-step 5000
# manifest binding: same rung passes, changed rung fails
cat > "${TMP}/launch_manifest.txt" <<EOF
# exp_11 arm launch manifest
arm C8 rung 16x4 micro 16 ngpu 4 max_steps 40000 ckpt_every 2500
commit ${HEAD_SHA}
wandb_run_id exp11-C8-test
EOF
expect_cmd "preflight binds to the launch manifest" 0 "bound to launch manifest" -- \
  "${PRE[@]}" --ckpt "${TMP}/good.ckpt" --expected-step 5000 --commit "$HEAD_SHA" --launch-manifest "${TMP}/launch_manifest.txt"
expect_cmd "preflight rejects a rung change" 2 "manifest rung" -- \
  $PY "$PREFLIGHT" --config "${EXPDIR}/FLAC_AR_BF_C8.json" --max-steps 40000 --arm C8 --rung 8x8 \
     --ckpt "${TMP}/good.ckpt" --expected-step 5000 --launch-manifest "${TMP}/launch_manifest.txt"
# B2 residual: a manifest with no commit, or a different commit, must fail CLOSED
grep -v '^commit ' "${TMP}/launch_manifest.txt" > "${TMP}/manifest_nocommit.txt"
expect_cmd "preflight rejects a manifest without a commit" 2 "no 'commit' line" -- \
  "${PRE[@]}" --ckpt "${TMP}/good.ckpt" --expected-step 5000 --commit "$HEAD_SHA" \
     --launch-manifest "${TMP}/manifest_nocommit.txt"
sed 's/^commit .*/commit 0000000000000000000000000000000000000000/' "${TMP}/launch_manifest.txt" > "${TMP}/manifest_othercommit.txt"
expect_cmd "preflight rejects a changed commit" 2 "!= running commit" -- \
  "${PRE[@]}" --ckpt "${TMP}/good.ckpt" --expected-step 5000 --commit "$HEAD_SHA" \
     --launch-manifest "${TMP}/manifest_othercommit.txt"
expect_cmd "preflight rejects a missing running commit" 2 "no running commit" -- \
  "${PRE[@]}" --ckpt "${TMP}/good.ckpt" --expected-step 5000 \
     --launch-manifest "${TMP}/launch_manifest.txt"

echo "--- H. the submitter refuses un-pinned submission ---"
expect_cmd "submitter refuses placeholders" 2 "TO-PIN-AFTER-P0" -- env DRYRUN=1 bash "$SUBMITTER" C8
expect_cmd "submitter rejects a bad arm" 2 "must be C4L" -- env DRYRUN=1 bash "$SUBMITTER" FA1
expect_cmd "submitter derives smoke flags" 0 "--gres=gpu:l40:4" -- \
  env DRYRUN=1 SMOKE=1 SMOKE_RUNG=16x4 SMOKE_MIN_FREE_MB=14000 bash "$SUBMITTER" C4L
expect_cmd "submitter derives cpus/mem from the rung" 0 "--cpus-per-task=36" -- \
  env DRYRUN=1 SMOKE=1 SMOKE_RUNG=16x4 SMOKE_MIN_FREE_MB=14000 bash "$SUBMITTER" C4L
expect_cmd "submitter derives 8x8 resources" 0 "--mem=108G" -- \
  env DRYRUN=1 SMOKE=1 SMOKE_RUNG=8x8 SMOKE_MIN_FREE_MB=14000 bash "$SUBMITTER" C4L

echo "--- I. flock run ownership, both contention directions (round-3 B3 residual) ---"
LOCKF="${TMP}/exp11_LOCKTEST.lock"
FIFO_HOLD="${TMP}/holder.fifo"; mkfifo "$FIFO_HOLD"
# held by a live process -> a contender must fail to acquire
# the holder keeps the fd open for its whole lifetime, exactly like the launcher
( flock -n 9 || exit 1; read -r _ < "$FIFO_HOLD" ) 9>"$LOCKF" &
HOLDER=$!
sleep 0.5
if flock -n 9 9>"$LOCKF" 2>/dev/null; then
  echo "FAIL  a second holder acquired a held flock"; FAIL=$((FAIL+1))
else
  echo "PASS  contender refused while the lock is held"; PASS=$((PASS+1))
fi
echo go > "$FIFO_HOLD"        # let the holder exit, closing fd 9
wait "$HOLDER" 2>/dev/null
# holder died (kill -9 equivalent) -> the lock must be free immediately, no stale dir
if flock -n 9 9>"$LOCKF" 2>/dev/null; then
  echo "PASS  lock free after the holder exits (no stale-recovery path needed)"; PASS=$((PASS+1))
else
  echo "FAIL  lock still held after the holder exited"; FAIL=$((FAIL+1))
fi
grep -q 'flock -n 9' "$LAUNCHER" && { echo "PASS  launcher uses flock, not mkdir+stale recovery"; PASS=$((PASS+1)); } \
  || { echo "FAIL  launcher does not use flock"; FAIL=$((FAIL+1)); }
grep -q 'release_lock' "$LAUNCHER" && { echo "FAIL  the old rmdir-based release survives"; FAIL=$((FAIL+1)); } \
  || { echo "PASS  no rmdir-based lock release remains"; PASS=$((PASS+1)); }

echo "--- J. OUTPUT_ROOT is a literal inside a Slurm job (NEW-2) ---"
case_run "ambient OUTPUT_ROOT rejected under Slurm" 2 "!= the production literal" \
  -- ARM=C4L SMOKE=1 SMOKE_RUNG=16x4 SMOKE_MIN_FREE_MB=99000000 "EXPECT_SHA=${HEAD_SHA}" \
     SLURM_JOB_ID=999999 "OUTPUT_ROOT=${OUT_ROOT}"
grep -q 'PRODUCTION_OUTPUT_ROOT="outputs_FLAC"' "$LAUNCHER" && { echo "PASS  launcher pins the production root literally"; PASS=$((PASS+1)); } \
  || { echo "FAIL  launcher has no production-root literal"; FAIL=$((FAIL+1)); }
grep -q 'OUTPUT_ROOT=outputs_FLAC' "$SUBMITTER" && { echo "PASS  submitter exports the fixed root, not ambient state"; PASS=$((PASS+1)); } \
  || { echo "FAIL  submitter still forwards an ambient OUTPUT_ROOT"; FAIL=$((FAIL+1)); }

echo "--- K. the submitter publishes intent BEFORE sbatch (NEW-3) ---"
INTENT_BEFORE="$(ls "${EXPDIR}"/fa_orbit_submission_*.txt 2>/dev/null | wc -l)"
expect_cmd "dry run publishes no submission manifest" 0 "DRYRUN sbatch" -- \
  env DRYRUN=1 SMOKE=1 SMOKE_RUNG=16x4 SMOKE_MIN_FREE_MB=14000 bash "$SUBMITTER" C4L
INTENT_AFTER="$(ls "${EXPDIR}"/fa_orbit_submission_*.txt 2>/dev/null | wc -l)"
[ "$INTENT_BEFORE" = "$INTENT_AFTER" ] && { echo "PASS  a dry run leaves no submission manifest behind"; PASS=$((PASS+1)); } \
  || { echo "FAIL  a dry run created a submission manifest"; FAIL=$((FAIL+1)); }
awk '/^OUT=.*sbatch/{sb=NR} /^mv -n "\$TMP" "\$MANIFEST"/{mf=NR} END{exit !(mf && sb && mf < sb)}' "$SUBMITTER" \
  && { echo "PASS  intent manifest is published before the sbatch call"; PASS=$((PASS+1)); } \
  || { echo "FAIL  the manifest is still published after sbatch"; FAIL=$((FAIL+1)); }
grep -q 'scancel "\$JID"' "$SUBMITTER" && { echo "PASS  an unrecordable job is cancelled"; PASS=$((PASS+1)); } \
  || { echo "FAIL  no scancel path for an unrecordable job"; FAIL=$((FAIL+1)); }

echo "--- L. FIFO and pip-freeze plumbing (NEW-4, B5 residual) ---"
grep -q 'mktemp -u' "$LAUNCHER" && { echo "FAIL  race-prone 'mktemp -u' FIFO remains"; FAIL=$((FAIL+1)); } \
  || { echo "PASS  FIFO no longer uses mktemp -u"; PASS=$((PASS+1)); }
grep -q "trap 'rm -f \"\$FIFO\"' EXIT" "$LAUNCHER" && { echo "PASS  FIFO removal is in the exit trap"; PASS=$((PASS+1)); } \
  || { echo "FAIL  FIFO is not removed by the exit trap"; FAIL=$((FAIL+1)); }
grep -q 'pip freeze > "\$PIPFREEZE_FILE"' "$LAUNCHER" && { echo "PASS  pip freeze status is checked before hashing"; PASS=$((PASS+1)); } \
  || { echo "FAIL  pip freeze is still hashed blind"; FAIL=$((FAIL+1)); }
grep -q 'final_tee_rc' "$LAUNCHER" && { echo "PASS  the final record's tee status is captured"; PASS=$((PASS+1)); } \
  || { echo "FAIL  the final tee status is still discarded"; FAIL=$((FAIL+1)); }
grep -q 'WANDB_ENTITY="\$WANDB_ENTITY_SEEN"' "$LAUNCHER" && { echo "PASS  the approved wandb entity is exported"; PASS=$((PASS+1)); } \
  || { echo "FAIL  WANDB_ENTITY is not exported"; FAIL=$((FAIL+1)); }
grep -q 'wandb-metadata.json' "$LAUNCHER" && { echo "PASS  the created wandb run identity is verified post-run"; PASS=$((PASS+1)); } \
  || { echo "FAIL  no post-run wandb identity verification"; FAIL=$((FAIL+1)); }

echo
echo "=== guard tests: ${PASS} passed, ${FAIL} failed ==="
[ "$FAIL" -eq 0 ] || exit 1
echo "log: ${LOG}"
