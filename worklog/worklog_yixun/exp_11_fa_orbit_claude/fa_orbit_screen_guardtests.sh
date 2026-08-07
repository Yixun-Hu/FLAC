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
import hashlib, json, os, sys, torch
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
        fh.write(f"job 90000{JOBN[arm]} host synthetic mode INITIAL launch_uuid uuid-{arm}\n")
        fh.write(f"arm {arm} rung 8x8 micro 8 ngpu 8 max_steps 40000 ckpt_every 2500\n")
        fh.write("commit " + "0" * 40 + "\n")
        fh.write(f"p0_manifest_sha256 {'a' * 64}\n")
        fh.write(f"config_sha256 {sha}\n")
        fh.write(f"vae_sha256 {'b' * 64}\n")
        fh.write(f"save_dir {d}\n")
    REG["arms"][arm] = {
        "manifest_path": os.path.join(d, "launch_manifest.txt"),
        "manifest_sha256": hashlib.sha256(
            open(os.path.join(d, "launch_manifest.txt"), "rb").read()).hexdigest(),
        "job": f"90000{JOBN[arm]}", "mode": "INITIAL", "launch_uuid": f"uuid-{arm}",
        "commit": "0" * 40, "rung": "8x8", "micro": "8", "ngpu": "8",
        "max_steps": "40000", "config_sha256": sha, "vae_sha256": "b" * 64,
        "p0_manifest_sha256": "a" * 64, "save_dir": d, "training_seed": 42,
    }
JOBN = {"C4L": 1, "C8": 2, "C16": 3, "C32": 4}
REG = {"arms": {}}
manifest("C4L", os.path.join(expdir, "FLAC_AR_BF_C4L.json"))
manifest("C16", os.path.join(expdir, "FLAC_AR_BF_C16.json"))
with open(os.path.join(out, "arm_launch_registry.json"), "w") as fh:
    json.dump(REG, fh, indent=2)
print("synthetic checkpoints written")
PY

BASE=(DRYRUN=1 "EXPECT_SHA=${HEAD_SHA}" "OUTPUT_ROOT=${OUT_ROOT}" "FA_ORBIT_REPO_OVERRIDE=$PWD"
      "FA_ORBIT_ARM_REGISTRY=${OUT_ROOT}/arm_launch_registry.json")

register_manifest() {  # <arm> — record the manifest as it stands, faithfully
  $PY - "$1" "${OUT_ROOT}/exp11_$1/launch_manifest.txt" "${OUT_ROOT}/arm_launch_registry.json" <<'PY'
import hashlib, json, sys
arm, man_path, reg_path = sys.argv[1:4]
raw = open(man_path, "rb").read()
man = {}
for line in raw.decode().splitlines():
    line = line.strip()
    if line and not line.startswith("#"):
        k, _, rest = line.partition(" ")
        man[k] = rest.strip()
f = ("arm " + man.get("arm", "")).split(); kv = {f[i]: f[i+1] for i in range(0, len(f)-1, 2)}
j = ("job " + man.get("job", "")).split(); jkv = {j[i]: j[i+1] for i in range(0, len(j)-1, 2)}
reg = json.load(open(reg_path))
reg["arms"][arm] = {
    "manifest_path": man_path, "manifest_sha256": hashlib.sha256(raw).hexdigest(),
    "job": jkv.get("job"), "mode": jkv.get("mode"), "launch_uuid": jkv.get("launch_uuid"),
    "commit": man.get("commit"), "rung": kv.get("rung"), "micro": kv.get("micro"),
    "ngpu": kv.get("ngpu"), "max_steps": kv.get("max_steps"),
    "config_sha256": man.get("config_sha256"), "vae_sha256": man.get("vae_sha256"),
    "p0_manifest_sha256": man.get("p0_manifest_sha256"), "save_dir": man.get("save_dir"),
    "training_seed": 42,
}
json.dump(reg, open(reg_path, "w"), indent=2)
PY
}

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
write_c8_manifest() {  # $1 = which arm's config hash to record
  { echo "job 900002 host synthetic mode INITIAL launch_uuid uuid-C8"
    echo "arm C8 rung 8x8 micro 8 ngpu 8 max_steps 40000 ckpt_every 2500"
    echo "commit 0000000000000000000000000000000000000000"
    echo "p0_manifest_sha256 aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    echo "config_sha256 $($PY -c "import hashlib,sys;print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" "${EXPDIR}/FLAC_AR_BF_$1.json")"
    echo "vae_sha256 bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
    echo "save_dir ${OUT_ROOT}/exp11_C8"; } > "${OUT_ROOT}/exp11_C8/launch_manifest.txt"
  register_manifest C8            # audited AS WRITTEN: the field checks are the test
}
write_c8_manifest C4L
case_run "a launch manifest for another config is refused" 2 "ARM LINEAGE GATE" \
  -- "${BASE[@]}" ARM=C8 STEP=10000
# a correct manifest lets the same screen through
write_c8_manifest C8

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
  # HEAD mismatch must abort even with a valid worktree. This case is about the
  # COMMIT gate, so give the simulated job a genuine lease first — otherwise the
  # lease gate (which runs earlier, by design) is what we would be testing.
  bash "${EXPDIR}/fa_orbit_measure_worktree.sh" --lease 999999 "$WT" >/dev/null 2>&1
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


# --- ASSETS: every untracked runtime input the eval resolves relatively ------
# This is the crasher the GO check found: a fresh worktree has only TRACKED
# files, so a pinned screen died at startup on the dataset symlink and weights.
echo
echo "--- assets + lease lifecycle (fa_orbit_measure_worktree.sh) ---"
HELPER="${EXPDIR}/fa_orbit_measure_worktree.sh"
if [ -n "${WT:-}" ] && [ -d "$WT" ]; then
  MISSING=""
  for ASSET in AcousticRooms weights weights/AGREE/AGREE_fullAR.pt \
               data/AR/unseen_eval.json \
               src/configs/dataset_configs/AR/eval/acousticroom_unseeneval.json; do
    [ -e "$WT/$ASSET" ] || MISSING="${MISSING} ${ASSET}"
  done
  if [ -z "$MISSING" ]; then
    echo "PASS  every required runtime asset stats inside the worktree"; PASS=$((PASS + 1))
  else
    echo "FAIL  missing from the worktree:${MISSING}"; FAIL=$((FAIL + 1))
  fi
  # and they must point where the MAIN tree points, not somewhere plausible
  if [ "$(readlink -f "$WT/AcousticRooms")" = "$(readlink -f "${MAIN_TREE}/AcousticRooms")" ] \
     && [ "$(readlink -f "$WT/weights")" = "$(readlink -f "${MAIN_TREE}/weights")" ]; then
    echo "PASS  worktree assets resolve to the same targets as the main tree"; PASS=$((PASS + 1))
  else
    echo "FAIL  worktree assets resolve elsewhere than the main tree"; FAIL=$((FAIL + 1))
  fi
  # the screen must REFUSE a code root without them rather than crash in eval
  FAKE="${TMP}/fakeroot"; mkdir -p "$FAKE"
  out="$(env DRYRUN=1 ARM=C8 STEP=10000 "EXPECT_SHA=${HEAD_SHA}" "OUTPUT_ROOT=$OUT_ROOT" \
         "FA_ORBIT_REPO_OVERRIDE=$FAKE" bash "$SCREEN" 2>&1)"; rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "PASS  a code root without the runtime assets is refused (rc=${rc})"; PASS=$((PASS + 1))
  else
    echo "FAIL  a code root without the runtime assets was accepted"; FAIL=$((FAIL + 1))
  fi
  grep -q 'required runtime asset missing from the code root' "$SCREEN" \
    && { echo "PASS  the screen carries an explicit asset gate"; PASS=$((PASS + 1)); } \
    || { echo "FAIL  the screen has no asset gate"; FAIL=$((FAIL + 1)); }

  # --- LEASES ---------------------------------------------------------------
  # (a) a leased worktree is never pruned
  bash "$HELPER" --lease pending-guard "$WT" >/dev/null 2>&1
  bash "$HELPER" --prune >/dev/null 2>&1
  if [ -d "$WT" ] && [ -f "$WT/.leases/pending-guard" ]; then
    echo "PASS  pruning skips a LEASED worktree"; PASS=$((PASS + 1))
  else
    echo "FAIL  pruning removed a leased worktree"; FAIL=$((FAIL + 1))
  fi
  # (b) a lease naming a job Slurm no longer knows is stale -> collectable
  DEADJOB=999999999
  bash "$HELPER" --lease "$DEADJOB" "$WT" >/dev/null 2>&1
  bash "$HELPER" --prune >/dev/null 2>&1
  if [ ! -f "$WT/.leases/$DEADJOB" ]; then
    echo "PASS  a stale lease (job gone from squeue) is reaped"; PASS=$((PASS + 1))
  else
    echo "FAIL  a stale lease survived a prune sweep"; FAIL=$((FAIL + 1))
  fi
  # (c) concurrent helper calls for the SAME sha are idempotent (one worktree).
  # Wait on the helper PIDs EXPLICITLY: a bare `wait` here also waits on this
  # script's own `tee` process substitution, which never exits.
  RACE_PIDS=()
  for _ in 1 2 3; do bash "$HELPER" >/dev/null 2>&1 & RACE_PIDS+=("$!"); done
  for pid in "${RACE_PIDS[@]}"; do wait "$pid"; done
  N="$(ls -1d "${MAIN_TREE}/.measure_worktrees/${WT_SHA}" 2>/dev/null | wc -l)"
  SAME="$(bash "$HELPER" 2>/dev/null | tail -1)"
  if [ "$N" -eq 1 ] && [ "$SAME" = "$WT" ]; then
    echo "PASS  concurrent same-SHA calls are idempotent (one tree, same path)"; PASS=$((PASS + 1))
  else
    echo "FAIL  concurrent same-SHA calls raced (n=${N}, path='${SAME}')"; FAIL=$((FAIL + 1))
  fi
  # (d) the tree currently being handed out is never swept
  bash "$HELPER" --release pending-guard "$WT" >/dev/null 2>&1
  KEPT="$(bash "$HELPER" 2>/dev/null | tail -1)"
  if [ -d "$KEPT" ]; then
    echo "PASS  the helper never prunes the worktree it is returning"; PASS=$((PASS + 1))
  else
    echo "FAIL  the helper pruned the tree it just returned"; FAIL=$((FAIL + 1))
  fi
  # (e) a real (SLURM) run without its own lease must abort
  bash "$HELPER" --release pending-guard "$WT" >/dev/null 2>&1
  out="$(env ARM=C8 STEP=10000 "EXPECT_SHA=${WT_SHA}" "MEASURE_ROOT=$WT" SLURM_JOB_ID=424242 \
         bash "$SCREEN" 2>&1)"; rc=$?
  if [ "$rc" -ne 0 ] && echo "$out" | grep -q "no lease"; then
    echo "PASS  a job without its own lease refuses to run"; PASS=$((PASS + 1))
  else
    echo "FAIL  a job ran in a tree nobody leased for it (rc=${rc})"; FAIL=$((FAIL + 1))
  fi
  # (f) a lease naming ANOTHER job is not this job's lease
  mkdir -p "$WT/.leases"; printf 'jobid 111\n' > "$WT/.leases/424242"
  out="$(env ARM=C8 STEP=10000 "EXPECT_SHA=${WT_SHA}" "MEASURE_ROOT=$WT" SLURM_JOB_ID=424242 \
         bash "$SCREEN" 2>&1)"; rc=$?
  if [ "$rc" -ne 0 ] && echo "$out" | grep -q "does not name job 424242"; then
    echo "PASS  a lease naming another job is rejected"; PASS=$((PASS + 1))
  else
    echo "FAIL  a mismatched lease was accepted (rc=${rc})"; FAIL=$((FAIL + 1))
  fi
  rm -f "$WT/.leases/424242"
  grep -q "trap 'rm -f \"\$LEASE\"' EXIT" "$SCREEN" \
    && { echo "PASS  the screen releases its lease on exit (trap)"; PASS=$((PASS + 1)); } \
    || { echo "FAIL  the screen never gives its lease back"; FAIL=$((FAIL + 1)); }
  grep -q "KEEP=3\|newest" "$HELPER" \
    && { echo "FAIL  fixed-count pruning is still present"; FAIL=$((FAIL + 1)); } \
    || { echo "PASS  fixed-count pruning is gone (leases only)"; PASS=$((PASS + 1)); }
  # the submitter must lease BEFORE sbatch, and promote after
  SUB="${EXPDIR}/fa_orbit_screen_submit.sh"
  if [ -f "$SUB" ] && [ "$(grep -n -- '--lease' "$SUB" | head -1 | cut -d: -f1)" -lt \
                        "$(grep -n 'sbatch --parsable' "$SUB" | head -1 | cut -d: -f1)" ] \
     && grep -q -- '--promote' "$SUB"; then
    echo "PASS  the submitter leases before sbatch and promotes after"; PASS=$((PASS + 1))
  else
    echo "FAIL  the submitter does not lease before sbatch"; FAIL=$((FAIL + 1))
  fi
else
  echo "FAIL  no worktree available for the asset/lease cases"; FAIL=$((FAIL + 1))
fi

# --- MEASURE_ROOT identity ---------------------------------------------------
echo
echo "--- MEASURE_ROOT identity ---"
out="$(env DRYRUN=1 ARM=C8 STEP=10000 "EXPECT_SHA=${HEAD_SHA}" "MEASURE_ROOT=$MAIN_TREE" \
       "OUTPUT_ROOT=$OUT_ROOT" bash "$SCREEN" 2>&1)"; rc=$?
if [ "$rc" -ne 0 ] && echo "$out" | grep -q "outside the managed .measure_worktrees/ area\|on a BRANCH"; then
  echo "PASS  the mutable MAIN tree is refused as a MEASURE_ROOT (rc=${rc})"; PASS=$((PASS + 1))
else
  echo "FAIL  the main checkout was accepted as a pinned measurement root (rc=${rc})"; FAIL=$((FAIL + 1))
fi

# --- ARM LAUNCH REGISTRY (immutable binding) ---------------------------------
echo
echo "--- arm launch registry binding ---"
REG="${EXPDIR}/arm_launch_registry.json"
if [ -f "$REG" ]; then
  echo "PASS  the audited arm launch registry is committed"; PASS=$((PASS + 1))
  if $PY - "$REG" <<'PY'
import json, sys
reg = json.load(open(sys.argv[1]))
arms = reg["arms"]
assert set(arms) == {"C4L", "C8", "C16", "C32"}, sorted(arms)
for a, v in arms.items():
    for f in ("manifest_sha256", "job", "mode", "launch_uuid", "commit", "rung",
              "max_steps", "config_sha256", "vae_sha256", "p0_manifest_sha256", "save_dir"):
        assert v.get(f), f"{a}.{f} is empty"
    assert v["mode"] == "INITIAL", (a, v["mode"])
    assert len(v["manifest_sha256"]) == 64
    assert int(v["training_seed"]) == 42
PY
  then echo "PASS  every arm is registered INITIAL, seed 42, with hashes"; PASS=$((PASS + 1))
  else echo "FAIL  the registry is incomplete"; FAIL=$((FAIL + 1))
  fi
  # a TAMPERED manifest must be caught: same fields, different bytes
  # tamper with the (synthetic) manifest AFTER it was registered: same fields,
  # different bytes — only the sha256 binding can catch this
  MAN="${OUT_ROOT}/exp11_C8/launch_manifest.txt"
  cp "$MAN" "${TMP}/manifest.bak"
  printf '# appended after registration by a guard test\n' >> "$MAN"
  out="$(env "${BASE[@]}" ARM=C8 STEP=10000 bash "$SCREEN" 2>&1)"; rc=$?
  if [ "$rc" -ne 0 ] && echo "$out" | grep -q "changed after it was registered"; then
    echo "PASS  a launch manifest edited after registration is rejected"; PASS=$((PASS + 1))
  else
    echo "FAIL  a tampered launch manifest passed the gate (rc=${rc})"
    echo "$out" | tail -3 | sed 's/^/        | /'; FAIL=$((FAIL + 1))
  fi
  cp "${TMP}/manifest.bak" "$MAN"
  # ...and a RESTART launch (mode != INITIAL) is not a registered launch
  sed -i 's/mode INITIAL/mode RESTART/' "$MAN"
  $PY - "$MAN" "${OUT_ROOT}/arm_launch_registry.json" <<'PY'
import hashlib, json, sys                      # re-register the tampered bytes so
reg = json.load(open(sys.argv[2]))             # ONLY the mode differs
reg["arms"]["C8"]["manifest_sha256"] = hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest()
json.dump(reg, open(sys.argv[2], "w"), indent=2)
PY
  out="$(env "${BASE[@]}" ARM=C8 STEP=10000 bash "$SCREEN" 2>&1)"; rc=$?
  if [ "$rc" -ne 0 ] && echo "$out" | grep -q "is not INITIAL"; then
    echo "PASS  a RESTART launch is refused as a screen lineage"; PASS=$((PASS + 1))
  else
    echo "FAIL  a non-INITIAL launch was accepted (rc=${rc})"; FAIL=$((FAIL + 1))
  fi
  cp "${TMP}/manifest.bak" "$MAN"
  grep -q 'reg\["manifest_sha256"\]' "$SCREEN" \
    && { echo "PASS  the screen binds the manifest by sha256, not by content alone"; PASS=$((PASS + 1)); } \
    || { echo "FAIL  the screen still trusts the mutable manifest"; FAIL=$((FAIL + 1)); }
  grep -q 'launch mode {jkv.get' "$SCREEN" \
    && { echo "PASS  the screen parses and enforces mode INITIAL"; PASS=$((PASS + 1)); } \
    || { echo "FAIL  INITIAL mode is not enforced"; FAIL=$((FAIL + 1)); }
  grep -q 'seed 42 recipe' "$SCREEN" \
    && { echo "FAIL  the seed is still only PRINTED, not checked"; FAIL=$((FAIL + 1)); } \
    || { echo "PASS  the seed claim is verified against the registry, not printed"; PASS=$((PASS + 1)); }
else
  echo "FAIL  no arm launch registry"; FAIL=$((FAIL + 1))
fi

echo
echo "=== screen guard tests: ${PASS} passed, ${FAIL} failed ==="
[ "$FAIL" -eq 0 ] || exit 1
echo "log: ${LOG}"
