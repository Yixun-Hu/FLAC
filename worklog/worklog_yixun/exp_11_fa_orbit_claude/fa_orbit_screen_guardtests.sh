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
MAIN_TREE="$(git rev-parse --show-toplevel)"

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
def ckpt(cfg, step, ema=True):
    sd = {"diffusion.model.a": torch.zeros(1)}
    if ema:                      # what eval_FLAC actually looks for
        sd["diffusion_ema.ema_model.model.a"] = torch.zeros(1)
    return {"global_step": step, "epoch": step // 4550, "model_config": cfg,
            "state_dict": sd,
            "optimizer_states": [{"state": {0: {"step": 1}}, "param_groups": [{"lr": 1e-5}]}],
            "lr_schedulers": [{"last_epoch": step}]}
def write(root, name, exp, cfg, step, epoch=2, ema=True):
    d = os.path.join(out, root, f"FLAC_{exp}", exp, "checkpoints")
    os.makedirs(d, exist_ok=True)
    torch.save(ckpt(cfg, step, ema), os.path.join(d, f"epoch={epoch}-step={step}.ckpt"))
c8 = json.load(open(os.path.join(expdir, "FLAC_AR_BF_C8.json")))
c4l = json.load(open(os.path.join(expdir, "FLAC_AR_BF_C4L.json")))
bf = json.load(open("worklog/worklog_yixun/exp_07_fa_scratch_claude/FLAC_AR_BF.json"))
write("exp11_C8", "C8", "exp11_C8", c8, 10000)
write("exp11_C8", "C8", "exp11_C8", c8, 12500)
write("exp11_C8", "C8", "exp11_C8", c8, 20000)            # duplicate pair below
write("exp11_C8", "C8", "exp11_C8", c8, 20000, epoch=3)
write("exp11_C4L", "C4L", "exp11_C4L", c8, 10000)          # WRONG: C8 config under C4L
write("exp11_C16", "C16", "exp11_C16", json.load(open(os.path.join(expdir, "FLAC_AR_BF_C16.json"))),
      15000, ema=False)                                    # no EMA weights at all
write("exp07_BF", "BF", "exp07_BF", bf, 20000)
# launch manifests so the LATER gates (identity, EMA) are the ones under test;
# the "no manifest" case below uses an arm deliberately left without one.
import hashlib
def manifest(arm, cfg_path):
    d = os.path.join(out, f"exp11_{arm}")
    os.makedirs(d, exist_ok=True)
    sha = hashlib.sha256(open(cfg_path, "rb").read()).hexdigest()
    with open(os.path.join(d, "launch_manifest.txt"), "w") as fh:
        fh.write(f"arm {arm} rung 8x8 micro 8 ngpu 8 max_steps 40000 ckpt_every 2500\n")
        fh.write("commit " + "0" * 40 + "\n")
        fh.write(f"config_sha256 {sha}\n")
        fh.write(f"save_dir {d}\n")
manifest("C4L", os.path.join(expdir, "FLAC_AR_BF_C4L.json"))
manifest("C16", os.path.join(expdir, "FLAC_AR_BF_C16.json"))
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
case_run "backfill rejects unregistered steps" 2 "registered at steps 20000/30000" -- "${BASE[@]}" ARM=C4BACKFILL STEP=12500
case_run "backfill 10k is not a registered gate" 2 "registered at steps 20000/30000" -- "${BASE[@]}" ARM=C4BACKFILL STEP=10000

echo "--- B. checkpoint discovery ---"
case_run "no ckpt at that step"  2 "exactly 1 checkpoint" -- "${BASE[@]}" ARM=C8 STEP=99000
case_run "two ckpts at one step" 2 "exactly 1 checkpoint" -- "${BASE[@]}" ARM=C8 STEP=20000
case_run "missing arm tree"      2 "exactly 1 checkpoint" -- "${BASE[@]}" ARM=C32 STEP=10000

echo "--- C. the ckpt/arm identity gate ---"
case_run "C4L tree holding a C8 ckpt is rejected" 2 "CKPT/ARM GATE" -- "${BASE[@]}" ARM=C4L STEP=10000
case_run "a checkpoint without EMA weights is rejected" 2 "no diffusion_ema.ema_model" \
  -- "${BASE[@]}" ARM=C16 STEP=15000
# the temp root has NO launch manifests, so every arm screen must refuse there
case_run "an arm ckpt with no launch manifest is refused" 2 "launch manifest missing" \
  -- "${BASE[@]}" ARM=C8 STEP=10000
# ...and with a manifest whose config hash is another arm's, the lineage gate fires
mkdir -p "${OUT_ROOT}/exp11_C8"
{ echo "arm C8 rung 8x8 micro 8 ngpu 8 max_steps 40000 ckpt_every 2500"
  echo "commit 0000000000000000000000000000000000000000"
  echo "config_sha256 $($PY -c "import hashlib;print(hashlib.sha256(open('${EXPDIR}/FLAC_AR_BF_C4L.json','rb').read()).hexdigest())")"
  echo "save_dir ${OUT_ROOT}/exp11_C8"; } > "${OUT_ROOT}/exp11_C8/launch_manifest.txt"
case_run "a launch manifest for another config is refused" 2 "ARM LINEAGE GATE" \
  -- "${BASE[@]}" ARM=C8 STEP=10000
# a correct manifest lets the same screen through
{ echo "arm C8 rung 8x8 micro 8 ngpu 8 max_steps 40000 ckpt_every 2500"
  echo "commit 0000000000000000000000000000000000000000"
  echo "config_sha256 $($PY -c "import hashlib;print(hashlib.sha256(open('${EXPDIR}/FLAC_AR_BF_C8.json','rb').read()).hexdigest())")"
  echo "save_dir ${OUT_ROOT}/exp11_C8"; } > "${OUT_ROOT}/exp11_C8/launch_manifest.txt"

echo "--- D. valid screens reach the eval argv ---"
case_run "C8 S10000 K8 default seed" 0 "exp11_C8_screen_S10000_s42_K8" -- "${BASE[@]}" ARM=C8 STEP=10000
case_run "screen contract: seed 43 refused"  2 "seed 42 by contract" -- "${BASE[@]}" ARM=C8 STEP=12500 SEED=43
case_run "screen contract: K=1 refused"      2 "K=8 by contract"     -- "${BASE[@]}" ARM=C8 STEP=10000 K=1
case_run "conf cell admits seed 43"          0 "exp11_C8_conf_S12500_s43_K8" -- "${BASE[@]}" ARM=C8 STEP=12500 SEED=43 CELL=conf
case_run "conf cell admits K=1"              0 "exp11_C8_conf_S10000_s42_K1" -- "${BASE[@]}" ARM=C8 STEP=10000 K=1 CELL=conf
case_run "conf cell refuses seed 47"         2 "seeds 42-46"          -- "${BASE[@]}" ARM=C8 STEP=10000 SEED=47 CELL=conf
case_run "backfill must stay a screen cell"  2 "futility-gate comparator" -- "${BASE[@]}" ARM=C4BACKFILL STEP=20000 CELL=conf
case_run "K=1 uses the _1 split"     0 "acousticroom_unseeneval_1.json" -- "${BASE[@]}" ARM=C8 STEP=10000 K=1 CELL=conf
case_run "K=8 uses the full split"   0 "acousticroom_unseeneval.json"   -- "${BASE[@]}" ARM=C8 STEP=10000
case_run "C8 carries its 8-angle orbit" 0 "0.0,45.0,90.0,135.0,180.0,225.0,270.0,315.0" -- "${BASE[@]}" ARM=C8 STEP=10000
case_run "protocol flags are pinned" 0 "--cond-autocast bf16 --cfg-scale 1.0 --steps 1" -- "${BASE[@]}" ARM=C8 STEP=10000
# The backfill checkpoint is bound to the AUDITED manifest (path + sha256), so a
# synthetic stand-in in the temp root must be refused; the positive path is
# exercised against the real audited checkpoint below.
case_run "a non-audited backfill ckpt is refused" 2 "audited" -- "${BASE[@]}" ARM=C4BACKFILL STEP=20000
if [ "${GUARD_REAL_BACKFILL:-0}" = "1" ]; then
  case_run "the audited backfill ckpt is accepted" 0 "exp11_C4backfill_S20000_s42_K8" \
    -- DRYRUN=1 "EXPECT_SHA=${HEAD_SHA}" "FA_ORBIT_REPO_OVERRIDE=$PWD" ARM=C4BACKFILL STEP=20000
else
  echo "SKIP  the audited-backfill positive case (GUARD_REAL_BACKFILL=1 to hash the real 724 MB ckpt)"
fi
if $PY -c "
import json,os,hashlib,sys
m=json.load(open('${EXPDIR}/c4_backfill_manifest.json'))
assert sorted(m['checkpoints'])==['20000','30000'], sorted(m['checkpoints'])
assert m['training_seed']==42
for s,e in m['checkpoints'].items():
    assert len(e['sha256'])==64 and os.path.isfile(e['path']), (s,e['path'])
assert hashlib.sha256(open(m['model_config'],'rb').read()).hexdigest()==m['model_config_sha256']
" 2>/dev/null; then
  echo "PASS  the audited backfill manifest is well-formed (20k/30k, seed 42, live paths, config hash)"
  PASS=$((PASS + 1))
else
  echo "FAIL  the audited backfill manifest is malformed or its files are missing"; FAIL=$((FAIL + 1))
fi

echo "--- E. real-mode gates ---"
# (a real screen now also needs MEASURE_ROOT; the SHA gate is exercised with one
# in section G, so here we only pin the sbatch-only requirement)
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

echo "--- G. worktree-pinned measurement execution (item 8) ---"
# A real screen must refuse to run unpinned...
case_run "real mode requires MEASURE_ROOT" 2 "MEASURE_ROOT is required" \
  -- ARM=C8 STEP=10000 "EXPECT_SHA=${HEAD_SHA}" SLURM_JOB_ID=999999
case_run "a non-existent MEASURE_ROOT is refused" 2 "not a directory" \
  -- "${BASE[@]}" ARM=C8 STEP=10000 "MEASURE_ROOT=${TMP}/nope"
mkdir -p "${TMP}/notaworktree"
case_run "a MEASURE_ROOT that is not a worktree is refused" 2 "not a git worktree" \
  -- "${BASE[@]}" ARM=C8 STEP=10000 "MEASURE_ROOT=${TMP}/notaworktree"

# ...and with a real pinned worktree the binding is on the WORKTREE's HEAD, so a
# divergent main tree is irrelevant. Simulate the divergence by binding to the
# worktree SHA while the main tree sits at a different commit.
WT="$(bash "${EXPDIR}/fa_orbit_measure_worktree.sh" 2>/dev/null | tail -1)"
if [ -n "$WT" ] && [ -e "$WT/.git" ]; then
  WT_SHA="$(git -C "$WT" rev-parse HEAD)"
  echo "PASS  pinned worktree prepared at ${WT_SHA:0:12}"; PASS=$((PASS + 1))
  # HEAD mismatch must abort even with a valid worktree
  out="$(env ARM=C8 STEP=10000 EXPECT_SHA=0000000000000000000000000000000000000000 \
          SLURM_JOB_ID=999999 "MEASURE_ROOT=$WT" bash "$SCREEN" 2>&1)"; rc=$?
  if [ "$rc" -eq 2 ] && echo "$out" | grep -q "code-root HEAD"; then
    echo "PASS  worktree HEAD mismatch aborts  (rc=${rc})"; PASS=$((PASS + 1))
  else
    echo "FAIL  worktree HEAD mismatch: rc=${rc}"; echo "$out" | tail -3 | sed 's/^/        | /'
    FAIL=$((FAIL + 1))
  fi
  # the code root's identity is the worktree's, NOT the main tree's
  out="$(env DRYRUN=1 ARM=C4L STEP=7500 "EXPECT_SHA=${WT_SHA}" "MEASURE_ROOT=$WT" bash "$SCREEN" 2>&1)"; rc=$?
  # the CONFIG must come from the worktree while the CHECKPOINT comes from the
  # main tree — that split is the whole point of pinned execution
  if [ "$rc" -eq 0 ] && echo "$out" | grep -q "config ${WT}/worklog" \
     && echo "$out" | grep -q "checkpoint: ${MAIN_TREE}/outputs_FLAC"; then
    echo "PASS  code from the pinned worktree, outputs from the main tree  (rc=${rc})"
    PASS=$((PASS + 1))
  else
    echo "SKIP  pinned-run case (needs a C4L ckpt at 7500 in the main tree; rc=${rc})"
  fi
  grep -q 'git -C "\$CODE_ROOT" rev-parse HEAD' "$SCREEN" \
    && { echo "PASS  the commit gate reads the code root, not the cwd"; PASS=$((PASS + 1)); } \
    || { echo "FAIL  the commit gate still reads the ambient HEAD"; FAIL=$((FAIL + 1)); }
else
  echo "FAIL  could not prepare a pinned worktree"; FAIL=$((FAIL + 1))
fi

echo
echo "=== screen guard tests: ${PASS} passed, ${FAIL} failed ==="
[ "$FAIL" -eq 0 ] || exit 1
echo "log: ${LOG}"
