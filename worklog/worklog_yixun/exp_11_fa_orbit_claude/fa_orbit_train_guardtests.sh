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
# RETIRED: this asserted that an UNPINNED arm refuses, but every pin landed in
# ea94995, so the placeholder no longer appears in any value and the case could
# never fire. Replaced by the end state it was protecting, plus proof that the
# refusal mechanism itself is still present to catch a future unpinned value.
if grep -qE '^PINNED_[A-Z_]+="TO-PIN-AFTER-P0"' "$LAUNCHER"; then
  echo "FAIL  a launcher pin is still the placeholder"; FAIL=$((FAIL+1))
else
  echo "PASS  every launcher pin holds a concrete value"; PASS=$((PASS+1))
fi
if grep -q 'PIN_PLACEHOLDER="TO-PIN-AFTER-P0"' "$LAUNCHER" \
   && grep -q 'PIN_PLACEHOLDER' "$LAUNCHER"; then
  echo "PASS  the launcher still refuses a placeholder pin if one returns"; PASS=$((PASS+1))
else
  echo "FAIL  the placeholder refusal mechanism is gone"; FAIL=$((FAIL+1))
fi
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

# Content-scoped binding (content-gate review B5): deterministic SYNTHETIC
# fixtures — dangling commits built with git plumbing. No ref moves, the
# tracked tree is untouched (only unreferenced objects are written; gc prunes
# them), and a missing fixture is a FAILURE, never a SKIP: the identical-tree
# case is the proof that record-only commits cannot kill a queued leg.
# The gate acceptance text is asserted; the run then aborts at a later gate
# (dirty-tree drift today, run-dir/allocation gates on a clean tree) with
# rc=2 and nothing written.
SYN_SAME="$(git commit-tree "$(git rev-parse 'HEAD^{tree}')" -p HEAD -m 'guardtest synthetic: identical tree' 2>/dev/null)"
SYN_IDX="${TMP}/synidx"; SYN_CHG=""
if GIT_INDEX_FILE="$SYN_IDX" git read-tree HEAD 2>/dev/null; then
  SYN_BLOB="$(printf 'guardtest synthetic drift\n' | git hash-object -w --stdin 2>/dev/null)"
  if [ -n "$SYN_BLOB" ] && GIT_INDEX_FILE="$SYN_IDX" git update-index --cacheinfo 100644 "$SYN_BLOB" train.py 2>/dev/null; then
    SYN_TREE="$(GIT_INDEX_FILE="$SYN_IDX" git write-tree 2>/dev/null)"
    [ -n "$SYN_TREE" ] && SYN_CHG="$(git commit-tree "$SYN_TREE" -p HEAD -m 'guardtest synthetic: train.py changed' 2>/dev/null)"
  fi
fi
if [ -n "$SYN_SAME" ]; then
  case_run "moved HEAD, surfaces identical -> gate passes" 2 "commit binding OK (content)" \
    -- ARM=C4L SMOKE=1 SMOKE_RUNG=16x4 SMOKE_MIN_FREE_MB=99000000 "EXPECT_SHA=${SYN_SAME}" SLURM_JOB_ID=999999
else
  echo "FAIL  could not synthesize the identical-tree fixture"; FAIL=$((FAIL+1))
fi
if [ -n "$SYN_CHG" ]; then
  case_run "moved HEAD, surfaces changed -> aborts" 2 "training surfaces changed since EXPECT_SHA" \
    -- ARM=C4L SMOKE=1 SMOKE_RUNG=16x4 SMOKE_MIN_FREE_MB=99000000 "EXPECT_SHA=${SYN_CHG}" SLURM_JOB_ID=999999
else
  echo "FAIL  could not synthesize the changed-surface fixture"; FAIL=$((FAIL+1))
fi
case_run "symbolic EXPECT_SHA refused" 2 "not a full lowercase 40-hex" \
  -- ARM=C4L SMOKE=1 SMOKE_RUNG=16x4 SMOKE_MIN_FREE_MB=99000000 EXPECT_SHA=HEAD SLURM_JOB_ID=999999
# src/tests is excluded from the closure (pytest-only; TDD sessions land tests
# continuously — a test-only commit must NOT kill a queued leg: 3680875-78).
SYN_TIDX="${TMP}/syntidx"; SYN_TESTS=""
if GIT_INDEX_FILE="$SYN_TIDX" git read-tree HEAD 2>/dev/null && [ -n "${SYN_BLOB:-}" ] \
   && GIT_INDEX_FILE="$SYN_TIDX" git update-index --add --cacheinfo 100644 "$SYN_BLOB" src/tests/test_guardtest_synthetic.py 2>/dev/null; then
  SYN_TTREE="$(GIT_INDEX_FILE="$SYN_TIDX" git write-tree 2>/dev/null)"
  [ -n "$SYN_TTREE" ] && SYN_TESTS="$(git commit-tree "$SYN_TTREE" -p HEAD -m 'guardtest synthetic: src/tests only' 2>/dev/null)"
fi
if [ -n "$SYN_TESTS" ]; then
  case_run "src/tests-only change -> gate passes" 2 "commit binding OK (content)" \
    -- ARM=C4L SMOKE=1 SMOKE_RUNG=16x4 SMOKE_MIN_FREE_MB=99000000 "EXPECT_SHA=${SYN_TESTS}" SLURM_JOB_ID=999999
else
  echo "FAIL  could not synthesize the src/tests-only fixture"; FAIL=$((FAIL+1))
fi

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

echo "--- G2. Q10: the JOB selects and enforces the RESTART time pin (re-pin fix 1) ---"
# The submitter allocated 34/51/89 h for the restart legs, but the job selected
# the INITIAL pin and then refused its own allocation. The pin the job enforces
# must follow the LEG, not the arm.
Q10_RUN="${OUT_ROOT}/exp11_C8/FLAC_exp11_C8/exp11_C8"
mkdir -p "${Q10_RUN}/checkpoints"
: > "${Q10_RUN}/checkpoints/epoch=8-step=40000.ckpt"
Q10_ENV=(DRYRUN=1 "EXPECT_SHA=${HEAD_SHA}" "OUTPUT_ROOT=${OUT_ROOT}" "${REPO_ENV[@]}")
case_run "a RESTART leg selects the RESTART pin" 0 "time pin PINNED_TIME_LIMIT_RESTART_C8=51:00:00" \
  -- "${Q10_ENV[@]}" ARM=C8 EXPECTED_STEP=40000 \
     "RESUME_CKPT=${Q10_RUN}/checkpoints/epoch=8-step=40000.ckpt"
case_run "an INITIAL launch keeps the INITIAL pin" 0 "time pin PINNED_TIME_LIMIT_C16=60:00:00" \
  -- "${Q10_ENV[@]}" ARM=C16
if grep -q 'the \${TIME_PIN_NAME} pin' "$LAUNCHER"; then
  echo "PASS  the allocation gate names the pin it enforced"; PASS=$((PASS+1))
else
  echo "FAIL  the allocation gate does not enforce the SELECTED time pin"; FAIL=$((FAIL+1))
fi
# submitter and job must pick the same pin for the same leg
SUB_RESTART="$(env DRYRUN=1 bash "$SUBMITTER" C16 --resume "${Q10_RUN}/checkpoints/epoch=8-step=40000.ckpt" --expected-step 40000 2>&1)"
if echo "$SUB_RESTART" | grep -q "time 89:00:00"; then
  echo "PASS  submitter and job agree on the C16 RESTART pin"; PASS=$((PASS+1))
else
  echo "FAIL  the submitter no longer allocates the C16 RESTART pin"; FAIL=$((FAIL+1))
fi

echo "--- G3. Q10: the 40k -> 100k EXTENSION preflight contract (re-pin fix 1) ---"
# The ordinary restart contract requires manifest max_steps == this run's budget
# and manifest commit == the running commit. An extension violates both BY
# DESIGN, so it gets its own contract: the original launch identity is preserved
# (audited manifest bytes, job/uuid/commit/config/save-dir/seed, and the resumed
# checkpoint IS the audited 40k anchor) while budget and commit may move.
EXT_ROOT="${TMP}/ext"; EXT_SAVE="${EXT_ROOT}/exp11_C8"
EXT_CKPT_DIR="${EXT_SAVE}/FLAC_exp11_C8/exp11_C8/checkpoints"
mkdir -p "$EXT_CKPT_DIR" "${EXT_ROOT}/elsewhere"
$PY - "$TMP" "${EXPDIR}/FLAC_AR_BF_C8.json" "$EXT_CKPT_DIR" "$EXT_SAVE" "${EXT_ROOT}/elsewhere" "$LAUNCHER" <<'PY'
import hashlib, json, os, re, sys, torch
tmp, cfg_path, ckpt_dir, save_dir, other, launcher = sys.argv[1:7]
vae_sha = re.search(r'^PINNED_VAE_SHA256="([^"]*)"', open(launcher).read(), re.M).group(1)
cfg = json.load(open(cfg_path))
ck = {"global_step": 40000, "epoch": 8, "model_config": cfg,
      "state_dict": {"diffusion.x": torch.zeros(1), "diffusion_ema.x": torch.zeros(1)},
      "optimizer_states": [{"state": {0: {"step": 1}}, "param_groups": [{"lr": 1e-5}]}],
      "lr_schedulers": [{"last_epoch": 40000}]}
path = os.path.join(ckpt_dir, "epoch=8-step=40000.ckpt")
torch.save(ck, path)
torch.save(ck, os.path.join(other, "epoch=8-step=40000.ckpt"))
h = hashlib.sha256(open(path, "rb").read()).hexdigest()
cfg_sha = hashlib.sha256(open(cfg_path, "rb").read()).hexdigest()
man = os.path.join(tmp, "ext_launch_manifest.txt")
with open(man, "w") as fh:
    fh.write("# exp_11 arm launch manifest\n")
    fh.write("job 3648695 host neu000 mode INITIAL launch_uuid ext-uuid-c8\n")
    fh.write("arm C8 rung 8x8 micro 8 ngpu 8 max_steps 40000 ckpt_every 2500\n")
    fh.write("commit " + "2" * 40 + "\n")
    fh.write(f"model_config {cfg_path}\n")
    fh.write(f"config_sha256 {cfg_sha}\n")
    fh.write(f"vae_sha256 {vae_sha}\n")
    fh.write(f"save_dir {save_dir}\n")
    fh.write("wandb_run_id exp11-C8-ext\n")
reg = {"arms": {"C8": {
    "manifest_path": man,
    "manifest_sha256": hashlib.sha256(open(man, "rb").read()).hexdigest(),
    "job": "3648695", "mode": "INITIAL", "launch_uuid": "ext-uuid-c8",
    "commit": "2" * 40, "rung": "8x8", "max_steps": "40000",
    "config_sha256": cfg_sha, "vae_sha256": vae_sha, "save_dir": save_dir,
    "training_seed": 42,
    "final_ckpt_sha256": h, "final_step": 40000}}, "restarts": {}}
json.dump(reg, open(os.path.join(tmp, "ext_registry.json"), "w"), indent=2)
print("extension fixture written")
PY
EXT_CKPT="${EXT_CKPT_DIR}/epoch=8-step=40000.ckpt"
EXT=($PY "$PREFLIGHT" --config "${EXPDIR}/FLAC_AR_BF_C8.json" --arm C8 --rung 8x8
     --ckpt "$EXT_CKPT" --expected-step 40000 --commit "$HEAD_SHA"
     --launch-manifest "${TMP}/ext_launch_manifest.txt" --extension
     --launch-registry "${TMP}/ext_registry.json")
expect_cmd "the ORDINARY contract refuses the extension (the bug)" 2 "manifest max_steps" -- \
  $PY "$PREFLIGHT" --config "${EXPDIR}/FLAC_AR_BF_C8.json" --arm C8 --rung 8x8 \
     --ckpt "$EXT_CKPT" --expected-step 40000 --commit "$HEAD_SHA" --max-steps 100000 \
     --launch-manifest "${TMP}/ext_launch_manifest.txt"
expect_cmd "extension accepts the 40k->100k leg" 0 "extension lineage OK" -- "${EXT[@]}" --max-steps 100000
expect_cmd "extension keeps the ORIGINAL launch commit" 0 "launch commit 2222222222" -- "${EXT[@]}" --max-steps 100000
expect_cmd "extension refuses a shrinking budget" 2 "does not extend" -- "${EXT[@]}" --max-steps 39000
expect_cmd "extension refuses a foreign resume path" 2 "canonical run directory" -- \
  $PY "$PREFLIGHT" --config "${EXPDIR}/FLAC_AR_BF_C8.json" --arm C8 --rung 8x8 --max-steps 100000 \
     --ckpt "${EXT_ROOT}/elsewhere/epoch=8-step=40000.ckpt" --expected-step 40000 --commit "$HEAD_SHA" \
     --launch-manifest "${TMP}/ext_launch_manifest.txt" --extension \
     --launch-registry "${TMP}/ext_registry.json"
$PY - "${TMP}/ext_registry.json" "${TMP}/reg_noanchor.json" "${TMP}/reg_wronganchor.json" \
     "${TMP}/reg_wrongcommit.json" <<'PY'
import json, sys
src, noanchor, wronganchor, wrongcommit = sys.argv[1:5]
r = json.load(open(src)); del r["arms"]["C8"]["final_ckpt_sha256"]
json.dump(r, open(noanchor, "w"), indent=2)
r = json.load(open(src)); r["arms"]["C8"]["final_ckpt_sha256"] = "c" * 64
json.dump(r, open(wronganchor, "w"), indent=2)
r = json.load(open(src)); r["arms"]["C8"]["commit"] = "9" * 40
json.dump(r, open(wrongcommit, "w"), indent=2)
PY
ext_with_reg() { $PY "$PREFLIGHT" --config "${EXPDIR}/FLAC_AR_BF_C8.json" --arm C8 --rung 8x8 \
  --max-steps 100000 --ckpt "$EXT_CKPT" --expected-step 40000 --commit "$HEAD_SHA" \
  --launch-manifest "${TMP}/ext_launch_manifest.txt" --extension --launch-registry "$1"; }
expect_cmd "extension refuses an arm with no audited anchor" 2 "no audited final_ckpt_sha256" -- \
  ext_with_reg "${TMP}/reg_noanchor.json"
# ...and fa_orbit_add_anchor.py is how that arm becomes extendable (fix 6): the
# SAME registry that just refused is anchored and then accepted. This is C32's
# sequence — audit the 40k checkpoint, write the anchor, then the leg may run.
add_anchor() { $PY "${EXPDIR}/fa_orbit_add_anchor.py" C8 --registry "$1" \
  --launcher "$LAUNCHER" --repo-root "$PWD" "${@:2}"; }
expect_cmd "add_anchor dry run writes nothing" 0 "dry run, nothing written" -- \
  add_anchor "${TMP}/reg_noanchor.json" --dry-run
expect_cmd "add_anchor audits and writes the anchor" 0 "anchored C8 at step 40000" -- \
  add_anchor "${TMP}/reg_noanchor.json"
expect_cmd "the extension preflight accepts the freshly anchored arm" 0 "extension lineage OK" -- \
  ext_with_reg "${TMP}/reg_noanchor.json"
expect_cmd "add_anchor is idempotent" 0 "already anchored" -- add_anchor "${TMP}/reg_noanchor.json"
expect_cmd "add_anchor refuses a manifest that disagrees with the registry" 2 "!= the registered" -- \
  add_anchor "${TMP}/reg_wrongcommit.json"
expect_cmd "extension refuses a resume that is not the anchor" 2 "audited final checkpoint" -- \
  ext_with_reg "${TMP}/reg_wronganchor.json"
expect_cmd "extension refuses a manifest commit that is not the registered one" 2 "registered launch commit" -- \
  ext_with_reg "${TMP}/reg_wrongcommit.json"
printf 'tamper\n' >> "${TMP}/ext_launch_manifest.txt"
expect_cmd "extension refuses a manifest that drifted after registration" 2 "changed after it was registered" -- \
  "${EXT[@]}" --max-steps 100000

echo "--- G4. Round 5: CHUNKED legs — chain preflight, chunk recorder, gates ---"
# A fresh fixture set (G3's manifest was deliberately tampered above): the same
# INITIAL identity, plus the 42500 endpoint checkpoint a first chunk produces
# and the launcher manifest that chunk leg would leave behind.
CH_ROOT="${TMP}/chain"; CH_SAVE="${CH_ROOT}/exp11_C8"
CH_CKPT_DIR="${CH_SAVE}/FLAC_exp11_C8/exp11_C8/checkpoints"
mkdir -p "$CH_CKPT_DIR"
CHUNK_PIN_C8="$(awk -F'"' '/^PINNED_TIME_LIMIT_CHUNK_C8=/{print $2; exit}' "$LAUNCHER")"
$PY - "$TMP" "${EXPDIR}/FLAC_AR_BF_C8.json" "$CH_CKPT_DIR" "$CH_SAVE" "$LAUNCHER" "$CHUNK_PIN_C8" <<'PY'
import hashlib, json, os, re, sys, torch
tmp, cfg_path, ckpt_dir, save_dir, launcher, chunk_pin = sys.argv[1:7]
vae_sha = re.search(r'^PINNED_VAE_SHA256="([^"]*)"', open(launcher).read(), re.M).group(1)
cfg = json.load(open(cfg_path))
def ck(step, epoch):
    return {"global_step": step, "epoch": epoch, "model_config": cfg,
            "state_dict": {"diffusion.x": torch.zeros(1), "diffusion_ema.x": torch.zeros(1)},
            "optimizer_states": [{"state": {0: {"step": 1}}, "param_groups": [{"lr": 1e-5}]}],
            "lr_schedulers": [{"last_epoch": step}]}
p40 = os.path.join(ckpt_dir, "epoch=8-step=40000.ckpt"); torch.save(ck(40000, 8), p40)
p42 = os.path.join(ckpt_dir, "epoch=9-step=42500.ckpt"); torch.save(ck(42500, 9), p42)
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
cfg_sha = hashlib.sha256(open(cfg_path, "rb").read()).hexdigest()
man = os.path.join(tmp, "chain_launch_manifest.txt")
with open(man, "w") as fh:
    fh.write("job 3648695 host neu000 mode INITIAL launch_uuid ext-uuid-c8\n")
    fh.write("arm C8 rung 8x8 micro 8 ngpu 8 max_steps 40000 ckpt_every 2500\n")
    fh.write("commit " + "2" * 40 + "\n")
    fh.write(f"model_config {cfg_path}\nconfig_sha256 {cfg_sha}\nvae_sha256 {vae_sha}\n")
    fh.write(f"save_dir {save_dir}\nwandb_run_id exp11-C8-chain\n")
reg = {"arms": {"C8": {
    "manifest_path": man, "manifest_sha256": sha(man),
    "job": "3648695", "mode": "INITIAL", "launch_uuid": "ext-uuid-c8",
    "commit": "2" * 40, "rung": "8x8", "max_steps": "40000",
    "config_sha256": cfg_sha, "vae_sha256": vae_sha, "save_dir": save_dir,
    "training_seed": 42, "final_ckpt_sha256": sha(p40), "final_step": 40000}}, "restarts": {}}
json.dump(reg, open(os.path.join(tmp, "chain_registry.json"), "w"), indent=2)
# the launcher manifest a finished 40000->42500 chunk leg leaves behind, in four
# variants: the real one (with the producing job's ENDPOINT ATTESTATION, round-5
# B6), one with no attestation at all (a pre-B6 launcher, or a leg that never
# finished), one attesting `<none>` (the leg did not reach the success class),
# and one whose attested sha is not the file's.
body = ("job 3999001 host neu001 mode RESTART launch_uuid chunk-uuid-1\n"
        "arm C8 rung 8x8 micro 8 ngpu 8 max_steps 100000 ckpt_every 2500\n"
        "commit " + "3" * 40 + "\n"
        f"model_config {cfg_path}\nconfig_sha256 {cfg_sha}\nvae_sha256 {vae_sha}\n"
        f"save_dir {save_dir}\n"
        f"resume_ckpt {p40} expected_step 40000 resume_ckpt_sha256 {sha(p40)}\n"
        f"time_limit {chunk_pin}\nchunk_end 42500\n")
attest = f"endpoint_ckpt {p42} endpoint_step 42500 endpoint_sha256 {sha(p42)}\n"
open(os.path.join(tmp, "chunk_leg_manifest.txt"), "w").write(body + attest)
open(os.path.join(tmp, "chunk_leg_manifest_noattest.txt"), "w").write(body)
open(os.path.join(tmp, "chunk_leg_manifest_noneattest.txt"), "w").write(
    body + "endpoint_ckpt <none> endpoint_step 42500 endpoint_class 4\n")
open(os.path.join(tmp, "chunk_leg_manifest_badattest.txt"), "w").write(
    body + f"endpoint_ckpt {p42} endpoint_step 42500 endpoint_sha256 {'e' * 64}\n")
print("chain fixture written")
PY
CH_CKPT42="${CH_CKPT_DIR}/epoch=9-step=42500.ckpt"

# --- fake scheduler binaries (round-5 r2 review) -----------------------------
# The new gates talk to the SCHEDULER: the submitter re-checks `squeue` INSIDE
# its per-arm reservation lock, and the recorder requires `sacct` to confirm the
# producing job COMPLETED. Both are exercised with fakes rather than the live
# queue, so the cases are deterministic. Every case that could conceivably reach
# `sbatch` also runs with a FAKE sbatch first on PATH: a guard test must not be
# able to queue a real job even if its assertion fails.
SHIM="${TMP}/shim"; mkdir -p "$SHIM"
cat > "${SHIM}/sbatch" <<'EOF'
#!/usr/bin/env bash
echo "FAKE-SBATCH: guard test — nothing was submitted"
exit 1
EOF
cat > "${SHIM}/squeue" <<'EOF'
#!/usr/bin/env bash
D="$(dirname "$(readlink -f "$0")")"
[ -s "${D}/squeue_out" ] && cat "${D}/squeue_out"
exit "$(cat "${D}/squeue_rc" 2>/dev/null || echo 0)"
EOF
printf '#!/usr/bin/env bash\necho COMPLETED\n'          > "${SHIM}/sacct_completed"
printf '#!/usr/bin/env bash\necho FAILED\n'             > "${SHIM}/sacct_failed"
printf '#!/usr/bin/env bash\nexit 0\n'                  > "${SHIM}/sacct_empty"
printf '#!/usr/bin/env bash\necho "sacct: connection refused" >&2\nexit 1\n' > "${SHIM}/sacct_rc1"
chmod +x "${SHIM}/sbatch" "${SHIM}/squeue" "${SHIM}"/sacct_*
: > "${SHIM}/squeue_out"; echo 0 > "${SHIM}/squeue_rc"

chain_pf() {  # <registry> [extra preflight args...]
  local reg="$1"; shift
  $PY "$PREFLIGHT" --config "${EXPDIR}/FLAC_AR_BF_C8.json" --arm C8 --rung 8x8 \
    --max-steps 100000 --ckpt "$CH_CKPT42" --expected-step 42500 --commit "$HEAD_SHA" \
    --launch-manifest "${TMP}/chain_launch_manifest.txt" --chain --launch-registry "$reg" "$@"
}
expect_cmd "chain refuses an UNRECORDED predecessor (the fail-closed core)" 2 "no recorded chain link" -- \
  chain_pf "${TMP}/chain_registry.json"
# The recorder now demands the SCHEDULER's confirmation, so every case that is
# NOT about that gate supplies a sacct that says COMPLETED; the sacct cases below
# override it (argparse takes the last --sacct-bin).
recorder_with() { $PY "${EXPDIR}/fa_orbit_record_restart.py" C8 "$1" \
  --registry "${TMP}/chain_registry.json" --launcher "$LAUNCHER" --repo-root "$PWD" \
  --sacct-bin "${SHIM}/sacct_completed" "${@:2}"; }
recorder() { recorder_with "${TMP}/chunk_leg_manifest.txt" "$@"; }
# B6: the manifest is published BEFORE training, so it proves intent, not
# authorship. Without the producing job's post-classification attestation, a
# FAILED leg's manifest plus a pre-existing 42500 checkpoint would mint a link.
# These three run BEFORE the successful record, so `mine` is empty and the
# attestation gate — not the immutability gate — is what refuses them.
expect_cmd "recorder refuses a manifest with NO endpoint attestation" 2 "no endpoint attestation" -- \
  recorder_with "${TMP}/chunk_leg_manifest_noattest.txt"
expect_cmd "recorder refuses a leg that attested it produced nothing" 2 "endpoint_ckpt <none>" -- \
  recorder_with "${TMP}/chunk_leg_manifest_noneattest.txt"
expect_cmd "recorder refuses an attested sha that is not the file's" 2 "attested endpoint_sha256" -- \
  recorder_with "${TMP}/chunk_leg_manifest_badattest.txt"
# round-5 r2 blocking 2: the attestation is the JOB's word; the link also needs
# the SCHEDULER's. Every non-COMPLETED answer — and every answer that is not an
# answer at all — is a refusal.
expect_cmd "recorder refuses a chunk whose job the scheduler reports FAILED" 2 "not COMPLETED" -- \
  recorder --sacct-bin "${SHIM}/sacct_failed"
expect_cmd "recorder refuses a chunk sacct reports nothing about" 2 "sacct reports nothing" -- \
  recorder --sacct-bin "${SHIM}/sacct_empty"
expect_cmd "recorder refuses when sacct itself fails" 2 "is UNKNOWN" -- \
  recorder --sacct-bin "${SHIM}/sacct_rc1"
expect_cmd "recorder refuses when sacct cannot be run at all" 2 "could not ask the scheduler" -- \
  recorder --sacct-bin "${SHIM}/sacct_does_not_exist"
expect_cmd "recorder dry run validates but writes nothing" 0 "dry run, nothing written" -- recorder --dry-run
expect_cmd "chain still refuses after a dry run" 2 "no recorded chain link" -- \
  chain_pf "${TMP}/chain_registry.json"
expect_cmd "recorder records the finished chunk as a chain link" 0 "recorded C8 chunk link" -- recorder
expect_cmd "recorder is idempotent per job" 0 "already recorded" -- recorder
expect_cmd "chain accepts the recorded tip" 0 "restart lineage OK" -- chain_pf "${TMP}/chain_registry.json"
expect_cmd "chain + valid --chunk-end accepted" 0 "restart lineage OK" -- \
  chain_pf "${TMP}/chain_registry.json" --chunk-end 45000
expect_cmd "chain refuses a --chunk-end off the checkpoint cadence" 2 "not a multiple of 2500" -- \
  chain_pf "${TMP}/chain_registry.json" --chunk-end 43000
expect_cmd "chain refuses a --chunk-end at or below the resume step" 2 "chunk_end <= max_steps" -- \
  chain_pf "${TMP}/chain_registry.json" --chunk-end 42500
# --- round-5 r2 blocking 3: Lightning's VERSION COUNTER ----------------------
# train.py builds ModelCheckpoint with enable_version_counter at its default, so
# a retry at a boundary whose unversioned name already exists saves
# `epoch=E-step=N-v1.ckpt`. A failed attempt that saved, followed by a successful
# retry, therefore leaves TWO files at the same step — and the old glob-by-name
# flow either bound the STALE bytes to the retry or found two hits and refused
# forever. The recorder now follows the path the producing job ATTESTED.
V_SAVE="${TMP}/vchain/exp11_C8"
V_CKPT_DIR="${V_SAVE}/FLAC_exp11_C8/exp11_C8/checkpoints"
mkdir -p "$V_CKPT_DIR" "${TMP}/vchain/elsewhere"
$PY - "$TMP" "${EXPDIR}/FLAC_AR_BF_C8.json" "$V_CKPT_DIR" "$V_SAVE" "$LAUNCHER" "$CHUNK_PIN_C8" \
     "${TMP}/vchain/elsewhere" <<'PY'
import hashlib, json, os, re, sys, time, torch
tmp, cfg_path, ckpt_dir, save_dir, launcher, chunk_pin, other = sys.argv[1:8]
vae_sha = re.search(r'^PINNED_VAE_SHA256="([^"]*)"', open(launcher).read(), re.M).group(1)
cfg = json.load(open(cfg_path))
def ck(step, epoch, tag=0.0):
    return {"global_step": step, "epoch": epoch, "model_config": cfg,
            "state_dict": {"diffusion.x": torch.full((1,), tag),
                           "diffusion_ema.x": torch.zeros(1)},
            "optimizer_states": [{"state": {0: {"step": 1}}, "param_groups": [{"lr": 1e-5}]}],
            "lr_schedulers": [{"last_epoch": step}]}
p40 = os.path.join(ckpt_dir, "epoch=8-step=40000.ckpt"); torch.save(ck(40000, 8), p40)
# the STALE twin the failed attempt left behind, written FIRST (older mtime)...
stale = os.path.join(ckpt_dir, "epoch=9-step=42500.ckpt"); torch.save(ck(42500, 9, 1.0), stale)
time.sleep(0.05)
# ...and the successful retry Lightning versioned, written SECOND (newest mtime)
retry = os.path.join(ckpt_dir, "epoch=9-step=42500-v1.ckpt"); torch.save(ck(42500, 9, 2.0), retry)
far = os.path.join(ckpt_dir, "epoch=10-step=45000.ckpt"); torch.save(ck(45000, 10), far)
outside = os.path.join(other, "epoch=9-step=42500-v1.ckpt"); torch.save(ck(42500, 9, 2.0), outside)
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
cfg_sha = hashlib.sha256(open(cfg_path, "rb").read()).hexdigest()
man = os.path.join(tmp, "vchain_launch_manifest.txt")
with open(man, "w") as fh:
    fh.write("job 3648695 host neu000 mode INITIAL launch_uuid ext-uuid-c8\n")
    fh.write("arm C8 rung 8x8 micro 8 ngpu 8 max_steps 40000 ckpt_every 2500\n")
    fh.write("commit " + "2" * 40 + "\n")
    fh.write(f"model_config {cfg_path}\nconfig_sha256 {cfg_sha}\nvae_sha256 {vae_sha}\n")
    fh.write(f"save_dir {save_dir}\nwandb_run_id exp11-C8-vchain\n")
reg = {"arms": {"C8": {
    "manifest_path": man, "manifest_sha256": sha(man),
    "job": "3648695", "mode": "INITIAL", "launch_uuid": "ext-uuid-c8",
    "commit": "2" * 40, "rung": "8x8", "max_steps": "40000",
    "config_sha256": cfg_sha, "vae_sha256": vae_sha, "save_dir": save_dir,
    "training_seed": 42, "final_ckpt_sha256": sha(p40), "final_step": 40000}}, "restarts": {}}
json.dump(reg, open(os.path.join(tmp, "vchain_registry.json"), "w"), indent=2)
body = ("job 3999011 host neu001 mode RESTART launch_uuid vchunk-uuid-1\n"
        "arm C8 rung 8x8 micro 8 ngpu 8 max_steps 100000 ckpt_every 2500\n"
        "commit " + "3" * 40 + "\n"
        f"model_config {cfg_path}\nconfig_sha256 {cfg_sha}\nvae_sha256 {vae_sha}\n"
        f"save_dir {save_dir}\n"
        f"resume_ckpt {p40} expected_step 40000 resume_ckpt_sha256 {sha(p40)}\n"
        f"time_limit {chunk_pin}\nchunk_end 42500\n")
def w(name, attested, s):
    open(os.path.join(tmp, name), "w").write(
        body + f"endpoint_ckpt {attested} endpoint_step 42500 endpoint_sha256 {s}\n")
w("vchunk_manifest.txt", retry, sha(retry))            # the retry, as the job attested it
w("vchunk_manifest_missing.txt", os.path.join(ckpt_dir, "epoch=9-step=42500-v9.ckpt"), "0" * 64)
w("vchunk_manifest_outside.txt", outside, sha(outside))
w("vchunk_manifest_wrongname.txt", far, sha(far))
json.dump({"retry": retry, "retry_sha": sha(retry), "stale_sha": sha(stale)},
          open(os.path.join(tmp, "vchain_facts.json"), "w"))
print("versioned-retry fixture written")
PY
V_RETRY="${V_CKPT_DIR}/epoch=9-step=42500-v1.ckpt"
vrecorder() { $PY "${EXPDIR}/fa_orbit_record_restart.py" C8 "$1" \
  --registry "${TMP}/vchain_registry.json" --launcher "$LAUNCHER" --repo-root "$PWD" \
  --sacct-bin "${SHIM}/sacct_completed" "${@:2}"; }
expect_cmd "recorder refuses an attested endpoint that does not exist" 2 "does not exist" -- \
  vrecorder "${TMP}/vchunk_manifest_missing.txt"
expect_cmd "recorder refuses an attested endpoint outside the canonical directory" 2 "canonical directory" -- \
  vrecorder "${TMP}/vchunk_manifest_outside.txt"
expect_cmd "recorder refuses an attested endpoint named for another boundary" 2 "chunk end step" -- \
  vrecorder "${TMP}/vchunk_manifest_wrongname.txt"
expect_cmd "recorder records the VERSIONED retry the job attested, beside its stale twin" 0 "recorded C8 chunk link" -- \
  vrecorder "${TMP}/vchunk_manifest.txt"
if $PY - "${TMP}/vchain_registry.json" "${TMP}/vchain_facts.json" <<'PY' | grep -q '^OK$'
import json, os, sys
reg, facts = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
link = reg["arms"]["C8"]["chain"][-1]
path, sha = link.get("final_ckpt_path", ""), link.get("final_ckpt_sha256")
print("OK" if (os.path.basename(path).endswith("-v1.ckpt") and sha == facts["retry_sha"]
               and sha != facts["stale_sha"]) else "NOT-OK")
PY
then
  echo "PASS  the chain link records the retry's path and hash, not the stale twin's"; PASS=$((PASS+1))
else
  echo "FAIL  the chain link does not point at the versioned retry checkpoint"; FAIL=$((FAIL+1))
fi
# blocking 3(d): the preflight takes an explicit --ckpt and identifies it by hash
# and directory, so a versioned endpoint resumes exactly like an unversioned one.
expect_cmd "chain accepts a VERSIONED endpoint as the resume file" 0 "restart lineage OK" -- \
  $PY "$PREFLIGHT" --config "${EXPDIR}/FLAC_AR_BF_C8.json" --arm C8 --rung 8x8 \
    --max-steps 100000 --ckpt "$V_RETRY" --expected-step 42500 --commit "$HEAD_SHA" \
    --launch-manifest "${TMP}/vchain_launch_manifest.txt" --chain \
    --launch-registry "${TMP}/vchain_registry.json"
# --skip-sacct is the documented manual-recovery escape hatch: it bypasses the
# scheduler gate (here with an sacct that cannot even run) and nothing else.
expect_cmd "recorder --skip-sacct proceeds past an unusable sacct" 0 "already recorded" -- \
  vrecorder "${TMP}/vchunk_manifest.txt" --sacct-bin "${SHIM}/sacct_rc1" --skip-sacct
expect_cmd "recorder without --skip-sacct still refuses that unusable sacct" 2 "is UNKNOWN" -- \
  vrecorder "${TMP}/vchunk_manifest.txt" --sacct-bin "${SHIM}/sacct_rc1"

# --- round-5 r2 blocking 4: a structurally valid FORGED link -----------------
# Continuity proves the numbers line up; it cannot prove a link was ever earned.
# This chain has ONE link that satisfies every ancestry rule — it resumes the
# audited anchor at the audited step, its final step increases, and its final
# hash is the very file we then present as the resume file — but the manifest it
# cites attests a different endpoint. Before this round it was accepted.
$PY - "${TMP}/chain_registry.json" "${TMP}/chain_reg_forged.json" \
     "${TMP}/chunk_leg_manifest_badattest.txt" <<'PY'
import hashlib, json, sys
src, dest, badman = sys.argv[1:4]
r = json.load(open(src))
row = r["arms"]["C8"]
real = row["chain"][-1]                       # the honestly recorded link
row["chain"] = [{
    "job": "3999001", "launch_uuid": "chunk-uuid-1",
    "manifest_path": badman,
    "manifest_sha256": hashlib.sha256(open(badman, "rb").read()).hexdigest(),
    "resume_step": row["final_step"], "resume_ckpt_sha256": row["final_ckpt_sha256"],
    "final_step": 42500, "final_ckpt_sha256": real["final_ckpt_sha256"],
    "final_ckpt_path": real.get("final_ckpt_path"),
    "recorded_utc": "2026-08-13T00:00:00+00:00"}]
json.dump(r, open(dest, "w"), indent=2)
PY
expect_cmd "chain refuses a forged link whose manifest attests a different endpoint" 2 "!= the link's final_ckpt_sha256" -- \
  chain_pf "${TMP}/chain_reg_forged.json"

printf '# drift\n' >> "${TMP}/chunk_leg_manifest.txt"
expect_cmd "recorder refuses to rewrite a recorded link" 2 "immutable" -- recorder
expect_cmd "chain refuses a link whose manifest changed after it was recorded" 2 "changed after the link was recorded" -- \
  chain_pf "${TMP}/chain_registry.json"
$PY - "${TMP}/chain_registry.json" "${TMP}/chain_reg_badsha.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1])); r["arms"]["C8"]["chain"][-1]["final_ckpt_sha256"] = "d" * 64
json.dump(r, open(sys.argv[2], "w"), indent=2)
PY
expect_cmd "chain refuses a resume that is not the recorded tip's checkpoint" 2 "not the checkpoint that chunk produced" -- \
  chain_pf "${TMP}/chain_reg_badsha.json"
expect_cmd "chain refuses a non-tip resume step" 2 "resumes the TIP" -- \
  $PY "$PREFLIGHT" --config "${EXPDIR}/FLAC_AR_BF_C8.json" --arm C8 --rung 8x8 \
    --max-steps 100000 --ckpt "${CH_CKPT_DIR}/epoch=8-step=40000.ckpt" --expected-step 40000 \
    --commit "$HEAD_SHA" --launch-manifest "${TMP}/chain_launch_manifest.txt" --chain \
    --launch-registry "${TMP}/chain_registry.json"
# B6: validating only the TIP accepted a crafted registry — append a link whose
# final hash is whatever file you want to run and nothing ties it to the audited
# anchor. The WHOLE ancestry is checked now, and a break names its link index.
$PY - "${TMP}/chain_registry.json" "${TMP}/chain_reg_badanchor.json" \
     "${TMP}/chain_reg_badancestry.json" <<'PY'
import json, sys
src, badanchor, badancestry = sys.argv[1:4]
r = json.load(open(src))                      # link 0 does not resume the INITIAL anchor
r["arms"]["C8"]["chain"][0]["resume_ckpt_sha256"] = "e" * 64
json.dump(r, open(badanchor, "w"), indent=2)
r = json.load(open(src))                      # link 1 does not resume link 0's endpoint
link0 = r["arms"]["C8"]["chain"][0]
r["arms"]["C8"]["chain"].append({
    "job": "3999002", "launch_uuid": "chunk-uuid-2", "manifest_path": "/dev/null",
    "manifest_sha256": "0" * 64, "resume_step": link0["final_step"],
    "resume_ckpt_sha256": "e" * 64, "final_step": 45000, "final_ckpt_sha256": "f" * 64,
    "recorded_utc": "2026-08-13T00:00:00+00:00"})
json.dump(r, open(badancestry, "w"), indent=2)
PY
expect_cmd "chain refuses a first link that does not descend from the audited anchor" 2 "BROKEN at link 0" -- \
  chain_pf "${TMP}/chain_reg_badanchor.json"
expect_cmd "chain refuses a BROKEN ancestry (link 1 does not continue link 0)" 2 "BROKEN at link 1" -- \
  chain_pf "${TMP}/chain_reg_badancestry.json"
# launcher-side CHUNK_END gates (parameter gates run under DRYRUN)
case_run "CHUNK_END on an INITIAL launch refused" 2 "only a RESTART leg may be chunked" \
  -- DRYRUN=1 ARM=C4L "EXPECT_SHA=${HEAD_SHA}" "OUTPUT_ROOT=${OUT_ROOT}" "${REPO_ENV[@]}" CHUNK_END=42500
case_run "CHUNK_END under SMOKE refused" 2 "no meaning under SMOKE" \
  -- "${SMOKE_ENV[@]}" ARM=C4L CHUNK_END=42500 EXPECTED_STEP=40000
# submitter-side --chunk-end shape gates (refused before any pin/sbatch work)
expect_cmd "submitter: --chunk-end without --resume refused" 2 "valid only together" -- \
  env DRYRUN=1 bash "$SUBMITTER" C8 --chunk-end 42500
expect_cmd "submitter: --chunk-end off cadence refused" 2 "not a multiple of 2500" -- \
  env DRYRUN=1 bash "$SUBMITTER" C8 --resume x.ckpt --expected-step 40000 --chunk-end 42600
expect_cmd "submitter: --chunk-end must exceed the resume step" 2 "must exceed" -- \
  env DRYRUN=1 bash "$SUBMITTER" C8 --resume x.ckpt --expected-step 42500 --chunk-end 42500
expect_cmd "submitter: --chunk-end above the budget refused" 2 "exceeds the pinned budget" -- \
  env DRYRUN=1 bash "$SUBMITTER" C8 --resume x.ckpt --expected-step 40000 --chunk-end 102500
# --- round-5 r2 blocking 1: the SUBMISSION RESERVATION lives in the submitter -
# The watchdog's singleton lock stops a second watchdog, not a human at a shell.
# The reservation therefore sits in the one sanctioned submitter: an exclusive
# per-arm flock, and a queue re-check INSIDE it. These cases run with a FAKE
# sbatch first on PATH, so even a failing assertion cannot queue anything.
SUB_LOCK="${EXPDIR}/.submit_C8.lock"
SUB_RELEASE="${TMP}/release_the_submit_lock"
( flock -n 9 || exit 1; while [ ! -e "$SUB_RELEASE" ]; do sleep 0.2; done ) 9>"$SUB_LOCK" &
SUB_HOLDER=$!
sleep 0.5
expect_cmd "submitter refuses while another submission holds the arm's lock" 2 "already holds" -- \
  env "PATH=${SHIM}:${PATH}" bash "$SUBMITTER" C8
: > "$SUB_RELEASE"
wait "$SUB_HOLDER" 2>/dev/null
# a queue we cannot read is never read as an empty queue
echo 1 > "${SHIM}/squeue_rc"; : > "${SHIM}/squeue_out"
expect_cmd "submitter refuses when squeue itself fails" 2 "the queue state is UNKNOWN" -- \
  env "PATH=${SHIM}:${PATH}" bash "$SUBMITTER" C8
# ...and a live job with this arm's name is a duplicate, whoever queued it
echo 0 > "${SHIM}/squeue_rc"; echo "4242424 RUNNING" > "${SHIM}/squeue_out"
expect_cmd "submitter refuses when a leg for the arm is already queued/running" 2 "already queued/running" -- \
  env "PATH=${SHIM}:${PATH}" bash "$SUBMITTER" C8
echo 0 > "${SHIM}/squeue_rc"; : > "${SHIM}/squeue_out"
awk '/flock -n 9/{if (!f) f=NR} /^  LIVE=.*squeue/{q=NR} /^OUT="\$\(sbatch/{s=NR}
     END{exit !(f && q && s && f < q && q < s)}' "$SUBMITTER" \
  && { echo "PASS  the submitter checks the queue INSIDE the reservation lock, before sbatch"; PASS=$((PASS+1)); } \
  || { echo "FAIL  the submitter's queue check is not sequenced flock -> squeue -> sbatch"; FAIL=$((FAIL+1)); }
# watchdog argument safety + the no-checkpoint skip (ONESHOT, everything in TMP)
WD="${EXPDIR}/fa_orbit_chunk_watchdog.sh"
WD_TMP="${TMP}/wd"; mkdir -p "${WD_TMP}/outputs"
expect_cmd "watchdog rejects an off-cadence CHUNK" 2 "not a multiple" -- \
  bash "$WD" ONESHOT=1 CHUNK=2600
expect_cmd "watchdog rejects an off-cadence per-arm CHUNK_VANL" 2 "not a multiple" -- \
  bash "$WD" ONESHOT=1 CHUNK_VANL=2600
expect_cmd "watchdog rejects an unknown arm" 2 "not a comma-separated list" -- \
  bash "$WD" ONESHOT=1 ARMS=C4L,BOGUS
expect_cmd "watchdog rejects an unknown key" 2 "unknown argument" -- \
  bash "$WD" ONESHOT=1 SBATCH_EXTRA=x
# round-5 r2 non-blocking: a chunk longer than its arm's wall pin was sized for
# cannot reach its boundary inside the allocation, so it is refused BY NAME.
expect_cmd "watchdog rejects a chunk above the arm's time-pin maximum" 2 "PINNED_TIME_LIMIT_CHUNK_C32" -- \
  bash "$WD" ONESHOT=1 CHUNK_C32=5000
expect_cmd "watchdog rejects a global CHUNK above the orbit arms' maximum" 2 "PINNED_TIME_LIMIT_CHUNK_C4L" -- \
  bash "$WD" ONESHOT=1 CHUNK=5000
# the table is PER ARM, not a blanket 2500: VANL's own maximum is 5000 (the
# sanctioned default, exercised by the ONESHOT case below), and 7500 is above it.
expect_cmd "watchdog rejects a chunk above VANL's own (larger) maximum" 2 "PINNED_TIME_LIMIT_CHUNK_VANL" -- \
  bash "$WD" ONESHOT=1 CHUNK_VANL=7500
# B2(a): a SECOND watchdog is a double-submission engine (both see "no live job",
# both submit the same boundary). Rather than race two watchdogs, hold the real
# lock with a background flock holder and prove the watchdog refuses to start.
WD_LOCK="${EXPDIR}/.chunk_watchdog.lock"
WD_RELEASE="${WD_TMP}/release_the_lock"      # the holder exits NORMALLY on this
( flock -n 9 || exit 1; while [ ! -e "$WD_RELEASE" ]; do sleep 0.2; done ) 9>"$WD_LOCK" &
WD_HOLDER=$!
sleep 0.5
expect_cmd "watchdog refuses a second concurrent instance" 2 "refusing to start a second instance" -- \
  bash "$WD" ONESHOT=1 DRYRUN=1 ARMS=C8 "OUTPUT_ROOT=${WD_TMP}/outputs" \
    "REGISTRY=${TMP}/chain_registry.json" "STATE=${WD_TMP}/state" "LOG=${WD_TMP}/log"
: > "$WD_RELEASE"
wait "$WD_HOLDER" 2>/dev/null
if flock -n 9 9>"$WD_LOCK" 2>/dev/null; then
  echo "PASS  the watchdog lock is free once the holder exits"; PASS=$((PASS+1))
else
  echo "FAIL  the watchdog lock is still held after the holder exited"; FAIL=$((FAIL+1))
fi
# B7: an arm whose registry row carries no audited anchor (VANL, live) must be
# FROZEN at startup — not submitted and then refused by the preflight. Anchoring
# is an operator action, so the watchdog names the tool instead of running it.
$PY - "${TMP}/chain_registry.json" "${TMP}/chain_reg_noanchor.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
r["arms"]["C8"].pop("final_ckpt_sha256", None); r["arms"]["C8"].pop("final_step", None)
json.dump(r, open(sys.argv[2], "w"), indent=2)
PY
expect_cmd "watchdog freezes an arm with no audited anchor before submitting" 0 "no audited anchor" -- \
  bash "$WD" ONESHOT=1 DRYRUN=1 ARMS=C8 "OUTPUT_ROOT=${WD_TMP}/outputs" \
    "REGISTRY=${TMP}/chain_reg_noanchor.json" "STATE=${WD_TMP}/state_noanchor" \
    "LOG=${WD_TMP}/log_noanchor"
# round-5 r2: the startup anchor check is DEEP — the registry's digest must be
# well formed, name step 40000, and still be the hash of the one anchor file on
# disk. A registry that says the right shape but the wrong bytes freezes the arm
# before a single chunk is queued (these fixtures keep the real save_dir, so the
# anchor file itself is found and hashed).
$PY - "${TMP}/chain_registry.json" "${TMP}/chain_reg_wrongsha.json" \
     "${TMP}/chain_reg_wrongstep.json" "${TMP}/chain_reg_shortsha.json" <<'PY'
import json, sys
src, wrongsha, wrongstep, shortsha = sys.argv[1:5]
r = json.load(open(src)); r["arms"]["C8"]["final_ckpt_sha256"] = "a" * 64
json.dump(r, open(wrongsha, "w"), indent=2)
r = json.load(open(src)); r["arms"]["C8"]["final_step"] = 42500
json.dump(r, open(wrongstep, "w"), indent=2)
r = json.load(open(src)); r["arms"]["C8"]["final_ckpt_sha256"] = "ABC123"
json.dump(r, open(shortsha, "w"), indent=2)
PY
expect_cmd "watchdog freezes an arm whose anchor file does not hash to the audited sha" 0 "did not verify" -- \
  bash "$WD" ONESHOT=1 DRYRUN=1 ARMS=C8 "OUTPUT_ROOT=${WD_TMP}/outputs" \
    "REGISTRY=${TMP}/chain_reg_wrongsha.json" "STATE=${WD_TMP}/state_wrongsha" \
    "LOG=${WD_TMP}/log_wrongsha"
expect_cmd "watchdog freezes an anchor that is not step 40000" 0 "not the 40000" -- \
  bash "$WD" ONESHOT=1 DRYRUN=1 ARMS=C8 "OUTPUT_ROOT=${WD_TMP}/outputs" \
    "REGISTRY=${TMP}/chain_reg_wrongstep.json" "STATE=${WD_TMP}/state_wrongstep" \
    "LOG=${WD_TMP}/log_wrongstep"
expect_cmd "watchdog freezes a malformed anchor digest" 0 "lowercase hex digest" -- \
  bash "$WD" ONESHOT=1 DRYRUN=1 ARMS=C8 "OUTPUT_ROOT=${WD_TMP}/outputs" \
    "REGISTRY=${TMP}/chain_reg_shortsha.json" "STATE=${WD_TMP}/state_shortsha" \
    "LOG=${WD_TMP}/log_shortsha"
# NOTE: this case runs against the REAL queue (no squeue fake): with a live
# exp11-C8-train job it logs "live job — nothing to do", without one it logs
# "nothing to resume, skipping" (the fixture OUTPUT_ROOT is empty). Either way
# a full ONESHOT pass must complete cleanly without submitting anything.
expect_cmd "watchdog completes a ONESHOT pass without submitting" 0 "ONESHOT: one pass complete" -- \
  bash "$WD" ONESHOT=1 DRYRUN=1 ARMS=C8 "OUTPUT_ROOT=${WD_TMP}/outputs" \
    "REGISTRY=${TMP}/chain_registry.json" "STATE=${WD_TMP}/state" "LOG=${WD_TMP}/log"
expect_cmd "watchdog defaults VANL to a 5000-step chunk (startup overhead)" 0 "chunk 2500 (VANL 5000)" -- \
  bash "$WD" ONESHOT=1 DRYRUN=1 ARMS=C8 "OUTPUT_ROOT=${WD_TMP}/outputs" \
    "REGISTRY=${TMP}/chain_registry.json" "STATE=${WD_TMP}/state" "LOG=${WD_TMP}/log"
# Round-5 r3 blocking 2: the same-boundary retry ORCHESTRATION. A settled-FAILED
# leg that left a stale boundary checkpoint must cost exactly one failure, be
# neither recorded nor resumed, and the retry must go out from the RECORDED tip
# (here the 40k anchor) at the same boundary. Scheduler answers are faked on
# PATH (squeue: empty queue; sacct: FAILED for the remembered job); DRYRUN keeps
# the submitter from queueing anything real.
mkdir -p "${WD_TMP}/bin"
printf '#!/bin/sh\nexit 0\n' > "${WD_TMP}/bin/squeue"
printf '#!/bin/sh\necho FAILED\n' > "${WD_TMP}/bin/sacct"
chmod +x "${WD_TMP}/bin/squeue" "${WD_TMP}/bin/sacct"
$PY - "${TMP}/chain_registry.json" "${TMP}/wd_reg_retry.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1])); r["arms"]["C8"].pop("chain", None)
json.dump(r, open(sys.argv[2], "w"), indent=2)
PY
printf 'lastjob_C8 999123\n' > "${WD_TMP}/state_retry"
expect_cmd "watchdog retries a failed boundary from the recorded tip (stale ckpt ignored)" 0 \
  "retrying the boundary from the recorded tip (40000)" -- \
  env PATH="${WD_TMP}/bin:$PATH" bash "$WD" ONESHOT=1 DRYRUN=1 ARMS=C8 \
    "OUTPUT_ROOT=${TMP}/chain" "REGISTRY=${TMP}/wd_reg_retry.json" \
    "STATE=${WD_TMP}/state_retry" "LOG=${WD_TMP}/log_retry"
# The invariant is that the FAILED job's stale checkpoint never reaches the
# recorder (no second bump from a recorder refusal). The absolute count is
# environment-dependent here: in a dirty tree the DRYRUN submit itself is
# refused by the submitter's clean-tree guard and adds one legitimate bump.
if grep -q "RECORDER REFUSED\|recording finished chunk" "${WD_TMP}/log_retry"; then
  echo "FAIL  the stale checkpoint from the failed job reached the recorder"; FAIL=$((FAIL+1))
else
  echo "PASS  the stale checkpoint never reached the recorder"; PASS=$((PASS+1))
fi

echo "--- H. the submitter refuses un-pinned submission ---"
# RETIRED for the same reason as the launcher case above: all pins are concrete,
# so the submitter's placeholder refusal is unreachable on the real file.
if grep -qE '^PINNED_[A-Z_]+="TO-PIN-AFTER-P0"' "$SUBMITTER"; then
  echo "FAIL  a submitter pin is still the placeholder"; FAIL=$((FAIL+1))
else
  echo "PASS  every submitter pin holds a concrete value"; PASS=$((PASS+1))
fi
grep -q 'PLACEHOLDER="TO-PIN-AFTER-P0"' "$SUBMITTER" \
  && { echo "PASS  the submitter still refuses a placeholder pin if one returns"; PASS=$((PASS+1)); } \
  || { echo "FAIL  the submitter placeholder refusal is gone"; FAIL=$((FAIL+1)); }
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
# RETIRED: the readback moved out of the launcher into fa_orbit_wandb_readback.py
# (71054cf, 15 unit tests) because PL's save_dir overrides WANDB_DIR; grepping the
# launcher for wandb-metadata.json tested the superseded shape. Assert the real
# contract instead: the launcher runs the readback and GATES its exit class on it.
if grep -q 'fa_orbit_wandb_readback.py' "$LAUNCHER" \
   && grep -q 'WANDB_CHECK_RC' "$LAUNCHER" \
   && grep -q 'WANDB_CHECK_RC" -ne 0' "$LAUNCHER"; then
  echo "PASS  the launcher runs the wandb readback and gates on its result"; PASS=$((PASS+1))
else
  echo "FAIL  no post-run wandb identity verification"; FAIL=$((FAIL+1))
fi

echo "--- M. round 6: sick-node exclusion in the training submitter ---"
# 4 chunk legs burned on neu301/neu306 (ECC list) in 3 days; the training
# submitter now defaults --exclude to the sick list. DRYRUN prints the line.
expect_cmd "submitter defaults --exclude to the sick list" 0 "--exclude=neu301,neu303,neu305,neu306,neu317,neu319,neu322,neu332" -- \
  env DRYRUN=1 bash "$SUBMITTER" C8 --resume x.ckpt --expected-step 40000 --chunk-end 42500
expect_cmd "EXCLUDE override replaces the default" 0 "--exclude=neu399" -- \
  env DRYRUN=1 EXCLUDE=neu399 bash "$SUBMITTER" C8 --resume x.ckpt --expected-step 40000 --chunk-end 42500
# Review fix: rc and the DRYRUN sbatch line are REQUIRED, or a dirty-tree
# refusal (rc=2, nothing printed) would false-PASS the absence check.
out="$(env DRYRUN=1 EXCLUDE= bash "$SUBMITTER" C8 --resume x.ckpt --expected-step 40000 --chunk-end 42500 2>&1)"; rc=$?
if [ "$rc" -eq 0 ] && echo "$out" | grep -q "DRYRUN sbatch" && ! echo "$out" | grep -q -- "--exclude"; then
  echo "PASS  EXCLUDE=\"\" disables exclusion"; PASS=$((PASS+1))
else
  echo "FAIL  EXCLUDE=\"\" should disable exclusion (rc=${rc})"; FAIL=$((FAIL+1))
fi
expect_cmd "submitter rejects an unsafe EXCLUDE" 2 "not a comma-separated node list" -- \
  env DRYRUN=1 "EXCLUDE=neu301;rm -rf x" bash "$SUBMITTER" C8 --resume x.ckpt --expected-step 40000 --chunk-end 42500

echo
echo "=== guard tests: ${PASS} passed, ${FAIL} failed ==="
[ "$FAIL" -eq 0 ] || exit 1
echo "log: ${LOG}"
