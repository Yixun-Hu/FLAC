#!/usr/bin/env bash
# ============================================================================
# fa_orbit_screen_guardtests.sh — guard-branch exercise for fa_orbit_screen.sbatch.
#
# DRYRUN=1 runs every cheap gate (parameters, arm->config/orbit mapping, commit
# binding, the exactly-one-checkpoint rule and the ckpt/arm identity gate) and
# then prints the eval argv instead of spending a GPU. Synthetic Lightning-shaped
# checkpoints are torch.save'd into a mktemp OUTPUT_ROOT, so the real arms'
# outputs are never read or written and no job is submitted.
#
# Usage: bash worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh
# Exit 0 = every case behaved as specified.
# ============================================================================
set -uo pipefail
cd "$(git -C "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" rev-parse --show-toplevel)" || exit 3

EXPDIR="worklog/worklog_yixun/exp_11_fa_orbit_claude"
SCREEN="${EXPDIR}/fa_orbit_screen.sbatch"
PY=/n/fs/gatrdp/envs/flac/bin/python
TS="$(date '+%Y-%m-%d_%H-%M-%S')"
LOG="${EXPDIR}/fa_orbit_${TS}_screen_guardtests.log"
HEAD_SHA="$(git rev-parse HEAD)"

exec > >(tee -a "$LOG") 2>&1
echo "=== fa_orbit_screen guard exercise — ${TS} — $(git rev-parse --short HEAD) ==="
[ -f "$SCREEN" ] || { echo "missing ${SCREEN} - abort"; exit 3; }

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
OUT_ROOT="${TMP}/outputs"
PASS=0; FAIL=0

# synthetic checkpoints: C8 @10000 (good), C8 @12500 (good), C4L @10000 with the
# WRONG (C4-labelled but C8-angled) config, a duplicate pair at 20000, and the
# exp_07 backfill lineage @20000.
$PY - "$OUT_ROOT" "$EXPDIR" <<'PY'
import json, os, sys, torch
out, expdir = sys.argv[1], sys.argv[2]
def ckpt(cfg, step):
    return {"global_step": step, "epoch": step // 4550, "model_config": cfg,
            "state_dict": {"diffusion_ema.x": torch.zeros(1)},
            "optimizer_states": [{"state": {0: {"step": 1}}, "param_groups": [{"lr": 1e-5}]}],
            "lr_schedulers": [{"last_epoch": step}]}
def write(root, name, exp, cfg, step, epoch=2):
    d = os.path.join(out, root, f"FLAC_{exp}", exp, "checkpoints")
    os.makedirs(d, exist_ok=True)
    torch.save(ckpt(cfg, step), os.path.join(d, f"epoch={epoch}-step={step}.ckpt"))
c8 = json.load(open(os.path.join(expdir, "FLAC_AR_BF_C8.json")))
c4l = json.load(open(os.path.join(expdir, "FLAC_AR_BF_C4L.json")))
bf = json.load(open("worklog/worklog_yixun/exp_07_fa_scratch_claude/FLAC_AR_BF.json"))
write("exp11_C8", "C8", "exp11_C8", c8, 10000)
write("exp11_C8", "C8", "exp11_C8", c8, 12500)
write("exp11_C8", "C8", "exp11_C8", c8, 20000)            # duplicate pair below
write("exp11_C8", "C8", "exp11_C8", c8, 20000, epoch=3)
write("exp11_C4L", "C4L", "exp11_C4L", c8, 10000)          # WRONG: C8 config under C4L
write("exp07_BF", "BF", "exp07_BF", bf, 20000)
print("synthetic checkpoints written")
PY

BASE=(DRYRUN=1 "EXPECT_SHA=${HEAD_SHA}" "OUTPUT_ROOT=${OUT_ROOT}" "FA_ORBIT_REPO_OVERRIDE=$PWD")

case_run() {  # <name> <want-rc> <want-substring> -- <env...>
  local name="$1" want_rc="$2" want_txt="$3"; shift 3; [ "$1" = "--" ] && shift
  local out rc
  out="$(env "$@" bash "$SCREEN" 2>&1)"; rc=$?
  if [ "$rc" -eq "$want_rc" ] && echo "$out" | grep -qF -- "$want_txt"; then
    echo "PASS  ${name}  (rc=${rc})"; PASS=$((PASS + 1))
  else
    echo "FAIL  ${name}: want rc=${want_rc} + '${want_txt}', got rc=${rc}"
    echo "$out" | tail -5 | sed 's/^/        | /'; FAIL=$((FAIL + 1))
  fi
}

echo "--- A. parameters ---"
case_run "missing ARM"          2 "ARM"          -- "${BASE[@]}" STEP=10000
case_run "missing STEP"         2 "STEP"         -- "${BASE[@]}" ARM=C8
case_run "missing EXPECT_SHA"   2 "EXPECT_SHA"   -- DRYRUN=1 ARM=C8 STEP=10000 "OUTPUT_ROOT=${OUT_ROOT}" "FA_ORBIT_REPO_OVERRIDE=$PWD"
case_run "unknown arm"          2 "not screenable" -- "${BASE[@]}" ARM=VAN STEP=10000
case_run "FA1 is not screenable" 2 "not screenable" -- "${BASE[@]}" ARM=FA1 STEP=10000
case_run "non-numeric STEP"     2 "STEP"         -- "${BASE[@]}" ARM=C8 STEP=lots
case_run "bad K"                2 "K"            -- "${BASE[@]}" ARM=C8 STEP=10000 K=4
case_run "backfill step must be 10k/20k/30k" 2 "backfill covers steps" -- "${BASE[@]}" ARM=C4BACKFILL STEP=12500

echo "--- B. checkpoint discovery ---"
case_run "no ckpt at that step"  2 "exactly 1 checkpoint" -- "${BASE[@]}" ARM=C8 STEP=99000
case_run "two ckpts at one step" 2 "exactly 1 checkpoint" -- "${BASE[@]}" ARM=C8 STEP=20000
case_run "missing arm tree"      2 "exactly 1 checkpoint" -- "${BASE[@]}" ARM=C32 STEP=10000

echo "--- C. the ckpt/arm identity gate ---"
case_run "C4L tree holding a C8 ckpt is rejected" 2 "CKPT/ARM GATE" -- "${BASE[@]}" ARM=C4L STEP=10000

echo "--- D. valid screens reach the eval argv ---"
case_run "C8 S10000 K8 default seed" 0 "exp11_C8_screen_S10000_s42_K8" -- "${BASE[@]}" ARM=C8 STEP=10000
case_run "C8 S12500 seed 43"         0 "exp11_C8_screen_S12500_s43_K8" -- "${BASE[@]}" ARM=C8 STEP=12500 SEED=43
case_run "K=1 uses the _1 split"     0 "acousticroom_unseeneval_1.json" -- "${BASE[@]}" ARM=C8 STEP=10000 K=1
case_run "K=8 uses the full split"   0 "acousticroom_unseeneval.json"   -- "${BASE[@]}" ARM=C8 STEP=10000
case_run "C8 carries its 8-angle orbit" 0 "0.0,45.0,90.0,135.0,180.0,225.0,270.0,315.0" -- "${BASE[@]}" ARM=C8 STEP=10000
case_run "protocol flags are pinned" 0 "--cond-autocast bf16 --cfg-scale 1.0 --steps 1" -- "${BASE[@]}" ARM=C8 STEP=10000
case_run "backfill maps to exp_07"   0 "exp11_C4backfill_S20000_s42_K8" -- "${BASE[@]}" ARM=C4BACKFILL STEP=20000
case_run "backfill uses the exp_07 config" 0 "exp_07_fa_scratch_claude/FLAC_AR_BF.json" -- "${BASE[@]}" ARM=C4BACKFILL STEP=20000

echo "--- E. real-mode gates ---"
case_run "wrong EXPECT_SHA aborts" 2 "EXPECT_SHA" \
  -- ARM=C8 STEP=10000 EXPECT_SHA=0000000000000000000000000000000000000000 SLURM_JOB_ID=999999
case_run "real mode needs sbatch"  2 "must run under sbatch" \
  -- ARM=C8 STEP=10000 "EXPECT_SHA=${HEAD_SHA}" "FA_ORBIT_REPO_OVERRIDE=$PWD"

echo "--- F. the emitted eval name parses under the validator's schema ---"
for NAME in "exp11_C8_screen_S10000_s42_K8" "exp11_C4backfill_S20000_s42_K8"; do
  if $PY -c "
import importlib.util,sys
s=importlib.util.spec_from_file_location('v','${EXPDIR}/exp11_validate_rows.py')
m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
print(m.parse_eval_name('${NAME}'))" >/dev/null 2>&1; then
    echo "PASS  ${NAME} parses under the validator schema"; PASS=$((PASS + 1))
  else
    echo "FAIL  ${NAME} does not parse under the validator schema"; FAIL=$((FAIL + 1))
  fi
done

echo
echo "=== screen guard tests: ${PASS} passed, ${FAIL} failed ==="
[ "$FAIL" -eq 0 ] || exit 1
echo "log: ${LOG}"
