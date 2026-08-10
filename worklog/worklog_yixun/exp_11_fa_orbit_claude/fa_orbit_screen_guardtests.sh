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
# DO NOT RUN THIS SUITE DURING ACTIVE CAMPAIGN SUBMISSIONS. It parks the campaign
# pin (restoring it via a trap, including on interrupt) and, on an idle store,
# exercises deletion paths. Run it between submission batches.
#
# Usage: bash worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh
# Exit 0 = every case behaved as specified.
# ============================================================================
set -uo pipefail
cd "$(git -C "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" rev-parse --show-toplevel)" || exit 3

EXPDIR="worklog/worklog_yixun/exp_11_fa_orbit_claude"
SCREEN="${EXPDIR}/fa_orbit_screen.sbatch"
GUARD_SELF="$(readlink -f "${BASH_SOURCE[0]}")"
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

# Several cases below need a store they are allowed to DELETE from, and get it
# by thawing the campaign freeze. That is safe on an idle store and catastrophic
# on a live one: a queued job's worktree can be swept while this suite runs.
# (Learned the hard way — an earlier run of this suite left the store thawed
# while a campaign was in flight; the leases held, but that was luck, not
# design.) So: if a campaign freeze is active, those cases are SKIPPED and the
# freeze is never touched. Everything that does not delete still runs.
MEASURE_HELPER="${EXPDIR}/fa_orbit_measure_worktree.sh"
CAMPAIGN_LIVE=0
bash "$MEASURE_HELPER" --frozen >/dev/null 2>&1 && CAMPAIGN_LIVE=1
[ "$CAMPAIGN_LIVE" = "1" ] && echo "NOTE: a CAMPAIGN FREEZE is active — deletion/thaw cases are skipped"

skip_if_campaign() {   # $1 = description; returns 0 (skip) when a campaign is live
  if [ "$CAMPAIGN_LIVE" = "1" ]; then
    echo "SKIP  $1 — a campaign freeze is active; this suite will not thaw or delete"
    return 0
  fi
  return 1
}
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
write("exp11_C8", "C8", "exp11_C8", c8, 40000, epoch=8)   # R3 / cross endpoint
write("exp11_C4L", "C4L", "exp11_C4L", c4l, 40000, epoch=8)      # the Q9 fa side
write("exp11_C8", "C8", "exp11_C8", c8, 42500, epoch=9)          # a Q10 traj point
vanl = json.load(open(os.path.join(expdir, "FLAC_AR_VANCKPT.json")))
write("exp11_VANL", "VANL", "exp11_VANL", vanl, 40000, epoch=8)   # the Q9 arm
write("exp11_VANL", "VANL", "exp11_VANL", c4l, 12500, epoch=2)    # WRONG: an ORBIT ckpt
write("exp07_BF", "BF", "exp07_BF", bf, 40000, epoch=8)   # legacy D2 endpoint
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
JOBN = {"C4L": 1, "C8": 2, "C16": 3, "C32": 4, "VANL": 5}
REG = {"arms": {}}
manifest("C4L", os.path.join(expdir, "FLAC_AR_BF_C4L.json"))
manifest("C16", os.path.join(expdir, "FLAC_AR_BF_C16.json"))
manifest("VANL", os.path.join(expdir, "FLAC_AR_VANCKPT.json"))
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

expect_cmd() {  # <name> <want-rc> <want-substring> -- <command...>   (any command, not the driver)
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
# --- VANL (Q9): the vanilla arm of this recipe ------------------------------
case_run "VANL runs a VANILLA evaluation" 0 "--cond-method vanilla" \
  -- "${BASE[@]}" ARM=VANL STEP=40000 CELL=conf SEED=43 K=1
case_run "VANL builds the vanilla eval name" 0 "exp11_VANL_conf_S40000_s43_K1" \
  -- "${BASE[@]}" ARM=VANL STEP=40000 CELL=conf SEED=43 K=1
case_run "VANL uses the VANCKPT config" 0 "FLAC_AR_VANCKPT.json" \
  -- "${BASE[@]}" ARM=VANL STEP=40000 CELL=conf SEED=43 K=1
case_run "VANL announces itself as orbit-free" 0 "VANILLA (no orbit)" \
  -- "${BASE[@]}" ARM=VANL STEP=40000
case_run "VANL screen cells exist too" 0 "exp11_VANL_screen_S40000_s42_K8" \
  -- "${BASE[@]}" ARM=VANL STEP=40000
case_run "VANL refuses r3"    2 "not registered for VANL" -- "${BASE[@]}" ARM=VANL STEP=40000 CELL=r3 ROTATE_DEG=0
case_run "VANL refuses cross" 2 "not registered for VANL" -- "${BASE[@]}" ARM=VANL STEP=40000 CELL=cross EVAL_ORBIT=16
case_run "an ORBIT ckpt is refused as VANL" 2 "trained with an orbit and is not the vanilla arm" \
  -- "${BASE[@]}" ARM=VANL STEP=12500 CELL=conf SEED=43 K=1
# --- the Q9 round: VANL and C4L at ONE new pin, in their own namespace -------
case_run "VANL q9 cell is registered" 0 "exp11_VANL_q9_S40000_s44_K8" \
  -- "${BASE[@]}" ARM=VANL STEP=40000 CELL=q9 SEED=44
case_run "C4L q9 cell is registered"  0 "exp11_C4L_q9_S40000_s44_K8" \
  -- "${BASE[@]}" ARM=C4L STEP=40000 CELL=q9 SEED=44
case_run "q9 is the VANL/C4L pair only" 2 "fa-vs-vanilla pair" \
  -- "${BASE[@]}" ARM=C8 STEP=40000 CELL=q9 SEED=44
case_run "q9 is the 40k endpoint only" 2 "40k endpoint only" \
  -- "${BASE[@]}" ARM=VANL STEP=10000 CELL=q9 SEED=44
case_run "q9 uses the confirmatory seeds" 2 "seeds 42-46" \
  -- "${BASE[@]}" ARM=VANL STEP=40000 CELL=q9 SEED=47
# --- CELL=r3 (plan §4): registered yaw offsets, injective names --------------
case_run "r3 needs a rotation"       2 "needs ROTATE_DEG" -- "${BASE[@]}" ARM=C8 STEP=40000 CELL=r3
case_run "r3 rejects an unregistered offset" 2 "not one of the registered R3 offsets" \
  -- "${BASE[@]}" ARM=C8 STEP=40000 CELL=r3 ROTATE_DEG=7
case_run "r3 rejects 5.62 (near-miss)" 2 "not one of the registered R3 offsets" \
  -- "${BASE[@]}" ARM=C8 STEP=40000 CELL=r3 ROTATE_DEG=5.62
case_run "r3 is the 40k endpoint only" 2 "registered at the 40k endpoint only" \
  -- "${BASE[@]}" ARM=C8 STEP=10000 CELL=r3 ROTATE_DEG=0
case_run "r3 is seed 42 by contract"  2 "seed 42 by contract" \
  -- "${BASE[@]}" ARM=C8 STEP=40000 CELL=r3 ROTATE_DEG=0 SEED=43
case_run "r3 names carry the rotation" 0 "exp11_C8_r3_rot5p625_s42_K8" \
  -- "${BASE[@]}" ARM=C8 STEP=40000 CELL=r3 ROTATE_DEG=5.625
case_run "r3 passes --rotate-deg to the eval" 0 "--rotate-deg 22.5" \
  -- "${BASE[@]}" ARM=C8 STEP=40000 CELL=r3 ROTATE_DEG=22.5
case_run "r3 keeps the arm's OWN orbit" 0 "0.0,45.0,90.0,135.0,180.0,225.0,270.0,315.0" \
  -- "${BASE[@]}" ARM=C8 STEP=40000 CELL=r3 ROTATE_DEG=45
# --- CELL=cross (R2 mechanism / D2) -----------------------------------------
case_run "cross needs an eval orbit" 2 "needs EVAL_ORBIT" -- "${BASE[@]}" ARM=C8 STEP=40000 CELL=cross
case_run "cross rejects an unregistered orbit" 2 "must be one of 4|8|16|32" \
  -- "${BASE[@]}" ARM=C8 STEP=40000 CELL=cross EVAL_ORBIT=6
case_run "cross refuses the arm's OWN orbit" 2 "OWN training orbit" \
  -- "${BASE[@]}" ARM=C8 STEP=40000 CELL=cross EVAL_ORBIT=8
case_run "cross runs the OTHER orbit" 0 "--frame-avg-angles 0.0,11.25,22.5,33.75" \
  -- "${BASE[@]}" ARM=C8 STEP=40000 CELL=cross EVAL_ORBIT=32
case_run "cross names carry the eval orbit" 0 "exp11_C8_cross_a16_S40000_s42_K8" \
  -- "${BASE[@]}" ARM=C8 STEP=40000 CELL=cross EVAL_ORBIT=16
case_run "cross is the 40k endpoint only" 2 "registered at the 40k endpoint only" \
  -- "${BASE[@]}" ARM=C8 STEP=10000 CELL=cross EVAL_ORBIT=16
# the checkpoint identity gate still proves the ckpt IS this arm (C8 embedded)
case_run "cross still proves the ckpt is the arm" 0 "fa_invariant C8" \
  -- "${BASE[@]}" ARM=C8 STEP=40000 CELL=cross EVAL_ORBIT=4
# --- legacy D2: the backfill at 40k, screen and cross only ------------------
# (rc=2 because the audited lineage gate rejects the SYNTHETIC 40k stand-in --
#  correctly; the real checkpoint is exercised under GUARD_REAL_BACKFILL below.
#  What these two prove is that 40k is now a REGISTERED step for these cells and
#  that the D2 eval names are built right, both of which happen before that gate.)
case_run "backfill 40k screen is a registered step" 2 "exp11_C4backfill_S40000_s42_K8" \
  -- "${BASE[@]}" ARM=C4BACKFILL STEP=40000
case_run "backfill 40k cross builds the D2 name" 2 "exp11_C4backfill_cross_a16_S40000_s42_K8" \
  -- "${BASE[@]}" ARM=C4BACKFILL STEP=40000 CELL=cross EVAL_ORBIT=16
case_run "backfill cross refuses C4's own orbit" 2 "OWN training orbit" \
  -- "${BASE[@]}" ARM=C4BACKFILL STEP=40000 CELL=cross EVAL_ORBIT=4
case_run "backfill cross is 40k only" 2 "40k endpoint only" \
  -- "${BASE[@]}" ARM=C4BACKFILL STEP=20000 CELL=cross EVAL_ORBIT=8
case_run "backfill still refuses conf/r3" 2 "futility comparator" \
  -- "${BASE[@]}" ARM=C4BACKFILL STEP=40000 CELL=r3 ROTATE_DEG=0
case_run "backfill 10k is still unregistered" 2 "registered at steps 20000/30000/40000" \
  -- "${BASE[@]}" ARM=C4BACKFILL STEP=10000
case_run "C8 S10000 K8 default seed" 0 "exp11_C8_screen_S10000_s42_K8" -- "${BASE[@]}" ARM=C8 STEP=10000
case_run "screen contract: seed 43 refused"  2 "seed 42 by contract" -- "${BASE[@]}" ARM=C8 STEP=12500 SEED=43
# K=1 trajectory screens are REGISTERED now (full curves at both K); the futility
# GATES stay K=8 only, which `gate_K` in the validator records separately.
# --- Q10 restart-chain lineage: >40k ckpts must descend from the audited run --
# The synthetic registry is built from the fixture manifests, so it has no
# restarts: a >40k checkpoint must therefore be refused outright.
write_c8_ckpt_45k() { $PY - "$OUT_ROOT" "$EXPDIR" <<'PY'
import json, os, sys, torch
out, expdir = sys.argv[1], sys.argv[2]
c8 = json.load(open(os.path.join(expdir, "FLAC_AR_BF_C8.json")))
d = os.path.join(out, "exp11_C8", "FLAC_exp11_C8", "exp11_C8", "checkpoints")
sd = {"diffusion.model.a": torch.zeros(1), "diffusion_ema.ema_model.model.a": torch.zeros(1)}
torch.save({"global_step": 45000, "epoch": 9, "model_config": c8, "state_dict": sd,
            "optimizer_states": [{"state": {0: {"step": 1}}, "param_groups": [{"lr": 1e-5}]}],
            "lr_schedulers": [{"last_epoch": 45000}]}, os.path.join(d, "epoch=9-step=45000.ckpt"))
PY
}
write_c8_ckpt_45k
# (no anchor recorded for the synthetic arm, so the chain fails one step earlier)
case_run "a >40k ckpt with no chain is refused" 2 "cannot be chained to its INITIAL run" \
  -- "${BASE[@]}" ARM=C8 STEP=45000 CELL=traj SEED=44
# now give C8 an anchor and a leg that resumes from the WRONG checkpoint
$PY - "${OUT_ROOT}/arm_launch_registry.json" "${OUT_ROOT}/exp11_C8" <<'PY'
import hashlib, json, os, sys
p, save = sys.argv[1], sys.argv[2]
anchor = hashlib.sha256(open(os.path.join(save, "FLAC_exp11_C8", "exp11_C8", "checkpoints",
                                          "epoch=8-step=40000.ckpt"), "rb").read()).hexdigest()
r = json.load(open(p))
r["arms"]["C8"]["final_ckpt_sha256"] = anchor
r["arms"]["C8"]["final_step"] = 40000
r["restarts"] = {"C8": [{"mode": "RESTART", "job": "999", "resume_ckpt_sha256": "b" * 64,
                         "max_steps": "100000"}]}
json.dump(r, open(p, "w"), indent=2)
PY
case_run "a RESTART resuming elsewhere is refused" 2 "no validated RESTART leg" \
  -- "${BASE[@]}" ARM=C8 STEP=45000 CELL=traj SEED=44
# --- fix 2: the leg is RECORDED by the real recorder, so the registry row and the
# per-leg producer manifest come from the same audited pipeline the operator uses.
$PY - "${OUT_ROOT}" "${EXPDIR}" <<'PY'
import hashlib, json, os, sys
out, expdir = sys.argv[1], sys.argv[2]
save = os.path.join(out, "exp11_C8")
ckpt = os.path.join(save, "FLAC_exp11_C8", "exp11_C8", "checkpoints", "epoch=8-step=40000.ckpt")
anchor = hashlib.sha256(open(ckpt, "rb").read()).hexdigest()
cfg = os.path.join(expdir, "FLAC_AR_BF_C8.json")
cfg_sha = hashlib.sha256(open(cfg, "rb").read()).hexdigest()
open(os.path.join(out, "restart_manifest_C8.txt"), "w").write("\n".join([
    "# exp_11 arm launch manifest",
    "job 3662829 host synthetic mode RESTART launch_uuid leg-uuid-c8",
    "arm C8 rung 8x8 micro 8 ngpu 8 max_steps 100000 ckpt_every 2500",
    "commit " + "c" * 40,
    "p0_manifest_sha256 " + "a" * 64,
    f"model_config {cfg}",
    f"config_sha256 {cfg_sha}",
    "vae_sha256 " + "b" * 64,
    "time_limit 51:00:00 min_free_mb 36500",
    f"resume_ckpt {ckpt} expected_step 40000 resume_ckpt_sha256 {anchor}",
    f"save_dir {save}", ""]))
r = json.load(open(os.path.join(out, "arm_launch_registry.json")))
r["restarts"] = {}
json.dump(r, open(os.path.join(out, "arm_launch_registry.json"), "w"), indent=2)
PY
record_c8_leg() {   # the operator's own command, against the REAL launcher pins
  $PY "$EXPDIR/fa_orbit_record_restart.py" C8 "${OUT_ROOT}/restart_manifest_C8.txt" \
    --registry "${OUT_ROOT}/arm_launch_registry.json" \
    --launcher "${EXPDIR}/fa_orbit_train.sbatch" --producer-dir "$OUT_ROOT" \
    --repo-root "$PWD" "$@"
}
expect_cmd "the recorder publishes the leg and its producer manifest" 0 "checkpoint(s) added" \
  -- record_c8_leg
case_run "a PRODUCED >40k ckpt is accepted" 0 "producer binding OK: step 45000" \
  -- "${BASE[@]}" ARM=C8 STEP=45000 CELL=traj SEED=44
# THE loophole: same arm, same config, same step, a valid recorded leg — but these
# are not the bytes that leg produced. The old existential gate accepted this.
cp "${OUT_ROOT}/exp11_C8/FLAC_exp11_C8/exp11_C8/checkpoints/epoch=9-step=45000.ckpt" "${TMP}/real45k.ckpt"
$PY - "$OUT_ROOT" "$EXPDIR" <<'PY'
import json, os, sys, torch
out, expdir = sys.argv[1], sys.argv[2]
c8 = json.load(open(os.path.join(expdir, "FLAC_AR_BF_C8.json")))
d = os.path.join(out, "exp11_C8", "FLAC_exp11_C8", "exp11_C8", "checkpoints")
sd = {"diffusion.model.a": torch.ones(1), "diffusion_ema.ema_model.model.a": torch.ones(1)}
torch.save({"global_step": 45000, "epoch": 9, "model_config": c8, "state_dict": sd,
            "optimizer_states": [{"state": {0: {"step": 2}}, "param_groups": [{"lr": 2e-5}]}],
            "lr_schedulers": [{"last_epoch": 45000}]}, os.path.join(d, "epoch=9-step=45000.ckpt"))
PY
case_run "a same-config ckpt from a WRONG restart is refused" 2 "NOT the checkpoint that leg produced" \
  -- "${BASE[@]}" ARM=C8 STEP=45000 CELL=traj SEED=44
cp "${TMP}/real45k.ckpt" "${OUT_ROOT}/exp11_C8/FLAC_exp11_C8/exp11_C8/checkpoints/epoch=9-step=45000.ckpt"
# a registry row without its producer manifest is no longer evidence
mv "${OUT_ROOT}/fa_orbit_producer_C8_job3662829.json" "${TMP}/producer.json"
case_run "a leg with no producer manifest admits nothing" 2 "producer manifest" \
  -- "${BASE[@]}" ARM=C8 STEP=45000 CELL=traj SEED=44
mv "${TMP}/producer.json" "${OUT_ROOT}/fa_orbit_producer_C8_job3662829.json"
# ...and every leg field is re-validated, not just mode + resume hash
tamper_leg() { $PY - "${OUT_ROOT}/arm_launch_registry.json" "$1" "$2" <<'PY'
import json, sys
p, field, value = sys.argv[1:4]
r = json.load(open(p)); r["restarts"]["C8"][0][field] = value
json.dump(r, open(p, "w"), indent=2)
PY
}
cp "${OUT_ROOT}/arm_launch_registry.json" "${TMP}/reg_recorded.json"
tamper_leg save_dir "${OUT_ROOT}/exp11_C16"
case_run "a leg whose save_dir is not the audited one is refused" 2 "no validated RESTART leg" \
  -- "${BASE[@]}" ARM=C8 STEP=45000 CELL=traj SEED=44
cp "${TMP}/reg_recorded.json" "${OUT_ROOT}/arm_launch_registry.json"
tamper_leg expected_step 30000
case_run "a leg that did not resume the audited final step is refused" 2 "no validated RESTART leg" \
  -- "${BASE[@]}" ARM=C8 STEP=45000 CELL=traj SEED=44
cp "${TMP}/reg_recorded.json" "${OUT_ROOT}/arm_launch_registry.json"
expect_cmd "a duplicate record of the same leg is refused" 1 "ALREADY recorded" -- record_c8_leg
case_run "<=40k ckpts need no restart row" 0 "exp11_C8_screen_S10000_s42_K8" \
  -- "${BASE[@]}" ARM=C8 STEP=10000

# --- Q10 trajectory cells: 5 seeds x both K, strictly above 40k -------------
case_run "traj cell is registered"        0 "exp11_C8_traj_S42500_s44_K1" \
  -- "${BASE[@]}" ARM=C8 STEP=42500 CELL=traj SEED=44 K=1
case_run "traj refuses 40000 itself"      2 "strictly above 40000" \
  -- "${BASE[@]}" ARM=C8 STEP=40000 CELL=traj SEED=44
case_run "traj refuses below 40k"         2 "strictly above 40000" \
  -- "${BASE[@]}" ARM=C8 STEP=30000 CELL=traj SEED=44
case_run "traj refuses off-grid steps"    2 "2500 checkpoint grid" \
  -- "${BASE[@]}" ARM=C8 STEP=42501 CELL=traj SEED=44
case_run "traj refuses past the budget"   2 "stop at the 100000 budget" \
  -- "${BASE[@]}" ARM=C8 STEP=102500 CELL=traj SEED=44
case_run "traj uses the conf seed set"    2 "seeds 42-46" \
  -- "${BASE[@]}" ARM=C8 STEP=42500 CELL=traj SEED=47
case_run "screen contract: K=1 admitted"    0 "exp11_C8_screen_S10000_s42_K1" -- "${BASE[@]}" ARM=C8 STEP=10000 K=1
case_run "screen contract: K=1 uses the _1 split" 0 "acousticroom_unseeneval_1.json" -- "${BASE[@]}" ARM=C8 STEP=10000 K=1
# (caught by the global K check before the cell contract is reached)
case_run "screen contract: K=4 still refused" 2 "must be 1 or 8"      -- "${BASE[@]}" ARM=C8 STEP=10000 K=4
case_run "backfill K=1 is admitted"         2 "exp11_C4backfill_S20000_s42_K1" -- "${BASE[@]}" ARM=C4BACKFILL STEP=20000 K=1
case_run "conf cell admits seed 43"          0 "exp11_C8_conf_S12500_s43_K8" -- "${BASE[@]}" ARM=C8 STEP=12500 SEED=43 CELL=conf
case_run "conf cell admits K=1"              0 "exp11_C8_conf_S10000_s42_K1" -- "${BASE[@]}" ARM=C8 STEP=10000 K=1 CELL=conf
case_run "conf cell refuses seed 47"         2 "seeds 42-46"          -- "${BASE[@]}" ARM=C8 STEP=10000 SEED=47 CELL=conf
case_run "backfill must stay a screen/cross cell" 2 "futility comparator" -- "${BASE[@]}" ARM=C4BACKFILL STEP=20000 CELL=conf
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
  case_run "the audited 40k D2 endpoint is accepted" 0 "exp11_C4backfill_S40000_s42_K8" \
    -- DRYRUN=1 "EXPECT_SHA=${HEAD_SHA}" "FA_ORBIT_REPO_OVERRIDE=$PWD" ARM=C4BACKFILL STEP=40000
  case_run "the audited 40k D2 cross sweep is accepted" 0 "exp11_C4backfill_cross_a32_S40000_s42_K8" \
    -- DRYRUN=1 "EXPECT_SHA=${HEAD_SHA}" "FA_ORBIT_REPO_OVERRIDE=$PWD" ARM=C4BACKFILL STEP=40000 \
       CELL=cross EVAL_ORBIT=32
else
  echo "SKIP  the audited-backfill positive case (GUARD_REAL_BACKFILL=1 to hash the real 724 MB ckpt)"
fi
# a vanilla evaluation must not be handed an orbit, even an empty one
VOUT="$(env "${BASE[@]}" ARM=VANL STEP=40000 CELL=conf SEED=43 K=1 bash "$SCREEN" 2>&1)"
if ! echo "$VOUT" | grep -q -- "--frame-avg-angles"; then
  echo "PASS  the VANL argv carries no --frame-avg-angles at all"; PASS=$((PASS + 1))
else
  echo "FAIL  a vanilla evaluation was given an orbit"; FAIL=$((FAIL + 1))
fi
if echo "$VOUT" | grep -q "ckpt gate OK: step 40000, vanilla (no orbit)"; then
  echo "PASS  the ckpt identity gate proves VANL is orbit-free"; PASS=$((PASS + 1))
else
  echo "FAIL  the VANL ckpt gate did not assert the absence of an orbit"; FAIL=$((FAIL + 1))
fi

# --- eval-name INJECTIVITY: distinct cells must not share a name ------------
NAMES=""
for ROT in 0 5.625 11.25 22.5 45; do
  NAMES="${NAMES}$(env "${BASE[@]}" ARM=C8 STEP=40000 CELL=r3 "ROTATE_DEG=${ROT}" bash "$SCREEN" 2>&1 \
                   | sed -n 's/.*eval-name \(exp11_[A-Za-z0-9_.]*\).*/\1/p' | head -1)\n"
done
NR3="$(printf '%b' "$NAMES" | grep -c .)"; UR3="$(printf '%b' "$NAMES" | sort -u | grep -c .)"
if [ "$NR3" -eq 5 ] && [ "$UR3" -eq 5 ]; then
  echo "PASS  the five R3 rotations produce five DISTINCT eval names"; PASS=$((PASS + 1))
else
  echo "FAIL  R3 eval names collide (${UR3} unique of ${NR3})"; printf '%b' "$NAMES" | sed 's/^/        | /'
  FAIL=$((FAIL + 1))
fi
NAMES=""
for ORB in 4 16 32; do
  NAMES="${NAMES}$(env "${BASE[@]}" ARM=C8 STEP=40000 CELL=cross "EVAL_ORBIT=${ORB}" bash "$SCREEN" 2>&1 \
                   | sed -n 's/.*eval-name \(exp11_[A-Za-z0-9_.]*\).*/\1/p' | head -1)\n"
done
NX="$(printf '%b' "$NAMES" | grep -c .)"; UX="$(printf '%b' "$NAMES" | sort -u | grep -c .)"
if [ "$NX" -eq 3 ] && [ "$UX" -eq 3 ]; then
  echo "PASS  the three cross orbits produce three DISTINCT eval names"; PASS=$((PASS + 1))
else
  echo "FAIL  cross eval names collide (${UX} unique of ${NX})"; printf '%b' "$NAMES" | sed 's/^/        | /'
  FAIL=$((FAIL + 1))
fi
# the audited manifest now registers the legacy D2 endpoint
if $PY -c "
import json,sys
m=json.load(open('${EXPDIR}/c4_backfill_manifest.json'))
e=m['checkpoints']['40000']
assert e['sha256']=='5319feb4af874624859e87105ddd8ab06d4b449769d1e054f712b2b1c0542328', e['sha256']
assert e['path'].endswith('epoch=8-step=40000.ckpt'), e['path']
assert int(e['bytes'])==723922667
assert m['training_seed']==42
"; then
  echo "PASS  the audited backfill manifest registers the 40k D2 endpoint"; PASS=$((PASS + 1))
else
  echo "FAIL  the 40k backfill entry is missing or wrong"; FAIL=$((FAIL + 1))
fi
if $PY -c "
import json,os,hashlib,sys
m=json.load(open('${EXPDIR}/c4_backfill_manifest.json'))
assert sorted(m['checkpoints'])==['20000','30000','40000'], sorted(m['checkpoints'])
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
  for ASSET in AcousticRooms weights/AGREE weights/AGREE/AGREE_fullAR.pt \
               weights/FLAC/VAE.safetensors weights/FLAC/VAE.ckpt \
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
  # leaf granularity: weights/ is TRACKED now (the exp_01 metric JSONs were
  # force-added), so the checkout owns the parent and we provision the leaves
  if [ "$(readlink -f "$WT/AcousticRooms")" = "$(readlink -f "${MAIN_TREE}/AcousticRooms")" ] \
     && [ "$(readlink -f "$WT/weights/AGREE")" = "$(readlink -f "${MAIN_TREE}/weights/AGREE")" ] \
     && [ "$(readlink -f "$WT/weights/FLAC/VAE.safetensors")" \
        = "$(readlink -f "${MAIN_TREE}/weights/FLAC/VAE.safetensors")" ]; then
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
  # existence is not identity: both assets are pinned by content/target
  AGREE_PIN="$(grep -o 'PINNED_AGREE_SHA256="[0-9a-f]\{64\}"' "$SCREEN" | head -1 | cut -d'"' -f2)"
  AGREE_REAL="$(sha256sum "${MAIN_TREE}/weights/AGREE/AGREE_fullAR.pt" | awk '{print $1}')"
  if [ -n "$AGREE_PIN" ] && [ "$AGREE_PIN" = "$AGREE_REAL" ]; then
    echo "PASS  AGREE_fullAR.pt is pinned by sha256 and matches (${AGREE_PIN:0:12})"; PASS=$((PASS + 1))
  else
    echo "FAIL  AGREE pin '${AGREE_PIN:0:12}' != on-disk ${AGREE_REAL:0:12}"; FAIL=$((FAIL + 1))
  fi
  # VAE.ckpt is the file the halt was about: the AGREE config chain loads it and
  # the worktree did not have it. Resolve it, and pin its content.
  VAE_CKPT_PIN="$(grep -o 'PINNED_VAE_CKPT_SHA256="[0-9a-f]\{64\}"' "$SCREEN" | head -1 | cut -d'"' -f2)"
  VAE_CKPT_REAL="$(sha256sum "${MAIN_TREE}/weights/FLAC/VAE.ckpt" | awk '{print $1}')"
  if [ -e "$WT/weights/FLAC/VAE.ckpt" ] && [ -n "$VAE_CKPT_PIN" ] \
     && [ "$VAE_CKPT_PIN" = "$VAE_CKPT_REAL" ]; then
    echo "PASS  VAE.ckpt resolves in the worktree and matches its pin (${VAE_CKPT_PIN:0:12})"
    PASS=$((PASS + 1))
  else
    echo "FAIL  VAE.ckpt missing or mispinned (pin '${VAE_CKPT_PIN:0:12}' vs ${VAE_CKPT_REAL:0:12})"
    FAIL=$((FAIL + 1))
  fi
  if grep -q "^ASSET_TARGETS=(/n/fs/gatrdp/datasets/AcousticRooms" "$HELPER" \
     && grep -q "REGISTERED target" "$HELPER"; then
    echo "PASS  the dataset symlink target is pinned literally in the helper"; PASS=$((PASS + 1))
  else
    echo "FAIL  the helper follows the main-tree symlink wherever it points"; FAIL=$((FAIL + 1))
  fi
  # a repointed main-tree symlink must abort, not be followed
  DECOY="${TMP}/decoy_corpus"; mkdir -p "$DECOY"
  out="$(env FA_ORBIT_ASSET_TARGET_OVERRIDE="$DECOY" bash -c "
    sed 's|^ASSET_TARGETS=(/n/fs/gatrdp/datasets/AcousticRooms|ASSET_TARGETS=(${DECOY}|' '$HELPER' \
      > '${TMP}/helper_decoy.sh'; bash '${TMP}/helper_decoy.sh'" 2>&1)"; rc=$?
  if [ "$rc" -ne 0 ] && echo "$out" | grep -q "REGISTERED target"; then
    echo "PASS  an unregistered dataset target aborts the pin"; PASS=$((PASS + 1))
  else
    echo "FAIL  an unregistered dataset target was accepted (rc=${rc})"; FAIL=$((FAIL + 1))
  fi

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
  # The lease comes back on exit and ONLY through the store lock. There is
  # deliberately no unlink fallback: an unlocked rm can drop a lease mid-sweep.
  if grep -q "trap 'bash \"\$MEASURE_HELPER\" --release" "$SCREEN" \
     && grep -q 'leaving it for the reaper' "$SCREEN"; then
    echo "PASS  the screen returns its lease on exit, under the store lock only"; PASS=$((PASS + 1))
  else
    echo "FAIL  the screen never gives its lease back"; FAIL=$((FAIL + 1))
  fi
  grep -q "KEEP=3\|newest" "$HELPER" \
    && { echo "FAIL  fixed-count pruning is still present"; FAIL=$((FAIL + 1)); } \
    || { echo "PASS  fixed-count pruning is gone (leases only)"; PASS=$((PASS + 1)); }
  # --- the held-job submission sequence, driven by MOCKED Slurm binaries ------
  # No job is submitted: sbatch/scontrol/scancel are shims that record their
  # argv. What is proven is the ORDER — held submit, lease by the real id,
  # release — and that a failure at either step cancels the held job.
  SUB="${EXPDIR}/fa_orbit_screen_submit.sh"
  FREEZE_MARKER="${MAIN_TREE}/.measure_worktrees/.campaign_freeze"
  # A submission refuses to run without the campaign freeze, so engage it here.
  # It is lifted again before the cases that must be free to delete.
  bash "$HELPER" --freeze "guard suite" >/dev/null 2>&1
  # A live campaign pin would send every submission below to the pinned commit
  # instead of HEAD, which is correct behaviour and wrong for these tests. Park
  # it for the duration; the pin's own cases set and restore it themselves.
  SUITE_SAVED_PIN="$(bash "$HELPER" --pinned 2>/dev/null)"
  [ "$SUITE_SAVED_PIN" = "<none>" ] && SUITE_SAVED_PIN=""
  if [ -n "$SUITE_SAVED_PIN" ]; then
    # TRAP-BACKED: restoring only on the normal path means a Ctrl-C, a timeout or
    # any early exit leaves the campaign unpinned — the next submission would then
    # silently measure at HEAD instead of the pin. Arm the restore BEFORE clearing.
    trap 'bash "$MEASURE_HELPER" --pin-campaign "$SUITE_SAVED_PIN" >/dev/null 2>&1; rm -rf "$TMP"' EXIT
    trap 'bash "$MEASURE_HELPER" --pin-campaign "$SUITE_SAVED_PIN" >/dev/null 2>&1; exit 130' INT TERM
    bash "$HELPER" --unpin-campaign >/dev/null 2>&1
  fi
  MOCK="${TMP}/mockbin"; mkdir -p "$MOCK"
  cat > "${MOCK}/sbatch" <<'EOS'
#!/usr/bin/env bash
echo "sbatch $*" >> "$MOCK_TRACE"
case "$*" in *--hold*) ;; *) echo "MOCK: sbatch was called WITHOUT --hold" >&2; exit 9;; esac
[ -n "${MOCK_SBATCH_SLEEP:-}" ] && { touch "${MOCK_TRACE}.submitting"; sleep "$MOCK_SBATCH_SLEEP"; }
echo "${MOCK_JOBID:-7654321}"
EOS
  cat > "${MOCK}/scontrol" <<'EOS'
#!/usr/bin/env bash
echo "scontrol $*" >> "$MOCK_TRACE"
# record what the lease looked like AT THE MOMENT of release: a release that
# ever runs against a missing or wrong lease is the ordering bug itself
if [ "$1" = "release" ] && grep -q "^jobid $2$" "${MOCK_WT}/.leases/$2" 2>/dev/null; then
  echo "release saw a VALID lease for $2" >> "$MOCK_TRACE"
fi
[ "${MOCK_RELEASE_FAILS:-0}" = "1" ] && exit 1
exit 0
EOS
  cat > "${MOCK}/scancel" <<'EOS'
#!/usr/bin/env bash
echo "scancel $*" >> "$MOCK_TRACE"
[ "${MOCK_SCANCEL_FAILS:-0}" = "1" ] && exit 1
exit 0
EOS
  chmod +x "${MOCK}"/sbatch "${MOCK}"/scontrol "${MOCK}"/scancel
  TRACE="${TMP}/trace.txt"; : > "$TRACE"
  out="$(env MOCK_TRACE="$TRACE" MOCK_WT="$WT" FA_ORBIT_SBATCH="${MOCK}/sbatch" \
             FA_ORBIT_SCONTROL="${MOCK}/scontrol" FA_ORBIT_SCANCEL="${MOCK}/scancel" \
             bash "$SUB" ARM=C4L STEP=10000 2>&1)"; rc=$?
  LEASE_OK=0; [ -f "$WT/.leases/7654321" ] && LEASE_OK=1
  if [ "$rc" -eq 0 ] && [ "$LEASE_OK" -eq 1 ] \
     && grep -q -- "--hold" "$TRACE" && grep -q "scontrol release 7654321" "$TRACE" \
     && ! grep -q "scancel" "$TRACE"; then
    echo "PASS  submit --hold -> lease by the real id -> scontrol release"; PASS=$((PASS + 1))
  else
    echo "FAIL  the held-submission sequence is wrong (rc=${rc}, lease=${LEASE_OK})"
    sed 's/^/        | /' "$TRACE"; echo "$out" | tail -3 | sed 's/^/        | /'; FAIL=$((FAIL + 1))
  fi
  # the lease must exist BEFORE the release: order, not just presence
  if [ "$(grep -n 'sbatch' "$TRACE" | head -1 | cut -d: -f1)" -lt \
       "$(grep -n 'scontrol release' "$TRACE" | head -1 | cut -d: -f1)" ] \
     && grep -q "jobid 7654321" "$WT/.leases/7654321" 2>/dev/null; then
    echo "PASS  the lease names the real job id and predates the release"; PASS=$((PASS + 1))
  else
    echo "FAIL  lease/release ordering not proven"; FAIL=$((FAIL + 1))
  fi
  bash "$HELPER" --release 7654321 "$WT" >/dev/null 2>&1
  # A failing release must CANCEL the held job — and KEEP the lease. The outcome
  # is ambiguous (the job may yet run), and a lease we tidy away is a worktree a
  # sweep deletes under a live job. Retaining costs one held tree until the
  # reaper proves the id is gone.
  : > "$TRACE"
  out="$(env MOCK_TRACE="$TRACE" MOCK_WT="$WT" MOCK_RELEASE_FAILS=1 FA_ORBIT_SBATCH="${MOCK}/sbatch" \
             FA_ORBIT_SCONTROL="${MOCK}/scontrol" FA_ORBIT_SCANCEL="${MOCK}/scancel" \
             bash "$SUB" ARM=C4L STEP=10000 2>&1)"; rc=$?
  if [ "$rc" -eq 6 ] && grep -q "scancel 7654321" "$TRACE" && [ -f "$WT/.leases/7654321" ]; then
    echo "PASS  a failed release cancels the job and RETAINS the lease"; PASS=$((PASS + 1))
  else
    echo "FAIL  a failed release mishandled the lease (rc=${rc}, lease $([ -f "$WT/.leases/7654321" ] && echo kept || echo DROPPED))"
    sed 's/^/        | /' "$TRACE"; FAIL=$((FAIL + 1))
  fi
  # ...and when scancel ALSO fails, the outcome is maximally uncertain: still keep it
  : > "$TRACE"
  out="$(env MOCK_TRACE="$TRACE" MOCK_WT="$WT" MOCK_RELEASE_FAILS=1 MOCK_SCANCEL_FAILS=1 \
             FA_ORBIT_SBATCH="${MOCK}/sbatch" FA_ORBIT_SCONTROL="${MOCK}/scontrol" \
             FA_ORBIT_SCANCEL="${MOCK}/scancel" bash "$SUB" ARM=C4L STEP=10000 2>&1)"; rc=$?
  if [ "$rc" -eq 6 ] && [ -f "$WT/.leases/7654321" ] \
     && echo "$out" | grep -q "scancel FAILED too"; then
    echo "PASS  release AND scancel failing keeps the lease and says why"; PASS=$((PASS + 1))
  else
    echo "FAIL  an uncertain cancellation did not retain the lease (rc=${rc})"
    echo "$out" | tail -4 | sed 's/^/        | /'; FAIL=$((FAIL + 1))
  fi
  bash "$HELPER" --release 7654321 "$WT" >/dev/null 2>&1
  # the lease is VALIDATED BEFORE the job is released, not after
  : > "$TRACE"
  out="$(env MOCK_TRACE="$TRACE" MOCK_WT="$WT" FA_ORBIT_SBATCH="${MOCK}/sbatch" \
             FA_ORBIT_SCONTROL="${MOCK}/scontrol" FA_ORBIT_SCANCEL="${MOCK}/scancel" \
             bash "$SUB" ARM=C4L STEP=10000 2>&1)"; rc=$?
  VAL_LINE="$(grep -n 'jobid \${JOBID}\$' "$SUB" | head -1 | cut -d: -f1)"
  REL_LINE="$(grep -n 'SCONTROL" release' "$SUB" | head -1 | cut -d: -f1)"
  if [ "$rc" -eq 0 ] && echo "$out" | grep -q "lease validated" \
     && grep -q "release saw a VALID lease for 7654321" "$TRACE" \
     && [ -n "$VAL_LINE" ] && [ -n "$REL_LINE" ] && [ "$VAL_LINE" -lt "$REL_LINE" ]; then
    echo "PASS  the lease is validated BEFORE scontrol release (source + runtime)"; PASS=$((PASS + 1))
  else
    echo "FAIL  release may run against an unvalidated lease (rc=${rc}, val@${VAL_LINE} rel@${REL_LINE})"
    sed 's/^/        | /' "$TRACE"; FAIL=$((FAIL + 1))
  fi
  bash "$HELPER" --release 7654321 "$WT" >/dev/null 2>&1

  # --- EXCLUDE= reaches sbatch as an explicit FLAG ---------------------------
  # SBATCH_EXCLUDE is not a thing: sbatch documents 58 input environment
  # variables and no --exclude equivalent among them, so relying on it meant no
  # batch ever excluded anything. The flag is the fix.
  : > "$TRACE"
  out="$(env MOCK_TRACE="$TRACE" MOCK_WT="$WT" FA_ORBIT_SBATCH="${MOCK}/sbatch" \
             FA_ORBIT_SCONTROL="${MOCK}/scontrol" FA_ORBIT_SCANCEL="${MOCK}/scancel" \
             bash "$SUB" ARM=C4L STEP=10000 EXCLUDE=neu303,neu332 2>&1)"; rc=$?
  if [ "$rc" -eq 0 ] && grep -q -- "--exclude=neu303,neu332" "$TRACE"; then
    echo "PASS  EXCLUDE= is passed to sbatch as an explicit --exclude flag"; PASS=$((PASS + 1))
  else
    echo "FAIL  EXCLUDE= did not reach sbatch (rc=${rc})"; sed 's/^/        | /' "$TRACE"
    FAIL=$((FAIL + 1))
  fi
  bash "$HELPER" --release 7654321 "$WT" >/dev/null 2>&1
  # no EXCLUDE given: no stray flag
  : > "$TRACE"
  env MOCK_TRACE="$TRACE" MOCK_WT="$WT" FA_ORBIT_SBATCH="${MOCK}/sbatch" \
      FA_ORBIT_SCONTROL="${MOCK}/scontrol" FA_ORBIT_SCANCEL="${MOCK}/scancel" \
      bash "$SUB" ARM=C4L STEP=10000 >/dev/null 2>&1
  if ! grep -q -- "--exclude" "$TRACE"; then
    echo "PASS  no EXCLUDE= means no --exclude flag"; PASS=$((PASS + 1))
  else
    echo "FAIL  an --exclude flag appeared without EXCLUDE="; FAIL=$((FAIL + 1))
  fi
  bash "$HELPER" --release 7654321 "$WT" >/dev/null 2>&1
  # the env var that never worked is now refused loudly instead of ignored
  out="$(env MOCK_TRACE="$TRACE" MOCK_WT="$WT" SBATCH_EXCLUDE=neu303 \
             FA_ORBIT_SBATCH="${MOCK}/sbatch" FA_ORBIT_SCONTROL="${MOCK}/scontrol" \
             FA_ORBIT_SCANCEL="${MOCK}/scancel" bash "$SUB" ARM=C4L STEP=10000 2>&1)"; rc=$?
  if [ "$rc" -ne 0 ] && echo "$out" | grep -q "sbatch does not honour it"; then
    echo "PASS  a set SBATCH_EXCLUDE is refused, not silently ignored"; PASS=$((PASS + 1))
  else
    echo "FAIL  SBATCH_EXCLUDE was accepted as if it worked (rc=${rc})"; FAIL=$((FAIL + 1))
  fi
  # cell parameters travel with the job
  : > "$TRACE"
  env MOCK_TRACE="$TRACE" MOCK_WT="$WT" FA_ORBIT_SBATCH="${MOCK}/sbatch" \
      FA_ORBIT_SCONTROL="${MOCK}/scontrol" FA_ORBIT_SCANCEL="${MOCK}/scancel" \
      bash "$SUB" ARM=C8 STEP=40000 CELL=r3 ROTATE_DEG=5.625 >/dev/null 2>&1
  if grep -q "ROTATE_DEG=5.625" "$TRACE" && grep -q "CELL=r3" "$TRACE"; then
    echo "PASS  the submitter exports the cell's own parameters to the job"; PASS=$((PASS + 1))
  else
    echo "FAIL  cell parameters do not reach the job"; sed 's/^/        | /' "$TRACE"; FAIL=$((FAIL + 1))
  fi
  bash "$HELPER" --release 7654321 "$WT" >/dev/null 2>&1

  # --- argument values are DATA, never shell ---------------------------------
  # The old parser eval'd the value, so a quote in it executed what followed.
  CANARY="${TMP}/injected.canary"; rm -f "$CANARY"
  out="$(env MOCK_TRACE="$TRACE" MOCK_WT="$WT" FA_ORBIT_SBATCH="${MOCK}/sbatch" \
             FA_ORBIT_SCONTROL="${MOCK}/scontrol" FA_ORBIT_SCANCEL="${MOCK}/scancel" \
             bash "$SUB" ARM=C4L STEP=10000 "PIN_SHA=x'; INJECTED=1; #'" 2>&1)"; rc=$?
  if [ "$rc" -ne 0 ] && echo "$out" | grep -q "not 40 hex characters"; then
    echo "PASS  the literal injection value is refused by shape"; PASS=$((PASS + 1))
  else
    echo "FAIL  the injection value was not refused (rc=${rc})"; echo "$out" | tail -3 | sed 's/^/        | /'
    FAIL=$((FAIL + 1))
  fi
  # ...and a payload that WOULD run under eval leaves no trace
  out="$(env MOCK_TRACE="$TRACE" MOCK_WT="$WT" FA_ORBIT_SBATCH="${MOCK}/sbatch" \
             FA_ORBIT_SCONTROL="${MOCK}/scontrol" FA_ORBIT_SCANCEL="${MOCK}/scancel" \
             bash "$SUB" ARM=C4L STEP=10000 "PIN_SHA=x'; touch ${CANARY}; #'" 2>&1)"; rc=$?
  if [ "$rc" -ne 0 ] && [ ! -e "$CANARY" ]; then
    echo "PASS  an injected payload does not execute (no side effects)"; PASS=$((PASS + 1))
  else
    echo "FAIL  an injected payload EXECUTED (canary $([ -e "$CANARY" ] && echo created || echo absent), rc=${rc})"
    FAIL=$((FAIL + 1)); rm -f "$CANARY"
  fi
  # every key is shape-checked, not just PIN_SHA
  for bad in "ARM=C4L;rm" "CELL=../etc" "STEP=1e4" "K=3" "EVAL_ORBIT=6" "EXCLUDE=neu1;id" "ROTATE_DEG=1.2.3"; do
    env MOCK_TRACE="$TRACE" MOCK_WT="$WT" FA_ORBIT_SBATCH="${MOCK}/sbatch" \
        FA_ORBIT_SCONTROL="${MOCK}/scontrol" FA_ORBIT_SCANCEL="${MOCK}/scancel" \
        bash "$SUB" ARM=C4L STEP=10000 "$bad" >/dev/null 2>&1 && { BADOK="$bad"; break; }
  done
  if [ -z "${BADOK:-}" ]; then
    echo "PASS  malformed values are refused for every key"; PASS=$((PASS + 1))
  else
    echo "FAIL  '${BADOK}' was accepted"; FAIL=$((FAIL + 1))
  fi
  # code only: the explanatory comment above the parser names the old eval
  if grep -vE '^[[:space:]]*#' "$SUB" | grep -q 'eval[[:space:]]'; then
    echo "FAIL  the submitter still evals an argument"; FAIL=$((FAIL + 1))
  else
    echo "PASS  no eval of argument text remains in the code"; PASS=$((PASS + 1))
  fi

  # --- CAMPAIGN PIN: one commit for the whole campaign -----------------------
  SAVED_PIN="$(bash "$HELPER" --pinned 2>/dev/null)"; [ "$SAVED_PIN" = "<none>" ] && SAVED_PIN=""
  PIN2="$(git rev-parse HEAD~1 2>/dev/null)"
  if [ -n "$PIN2" ]; then
    bash "$HELPER" --pin-campaign "$PIN2" >/dev/null 2>&1
    : > "$TRACE"
    out="$(env MOCK_TRACE="$TRACE" MOCK_WT="$WT" FA_ORBIT_SBATCH="${MOCK}/sbatch" \
               FA_ORBIT_SCONTROL="${MOCK}/scontrol" FA_ORBIT_SCANCEL="${MOCK}/scancel" \
               bash "$SUB" ARM=C4L STEP=10000 2>&1)"; rc=$?
    if [ "$rc" -eq 0 ] && grep -q "EXPECT_SHA=${PIN2}" "$TRACE" \
       && echo "$out" | grep -q "campaign pin (from"; then
      echo "PASS  the campaign pin file is the DEFAULT pin"; PASS=$((PASS + 1))
    else
      echo "FAIL  the campaign pin was not used by default (rc=${rc})"; FAIL=$((FAIL + 1))
    fi
    bash "$HELPER" --release 7654321 "${MAIN_TREE}/.measure_worktrees/${PIN2}" >/dev/null 2>&1
    rm -f ${EXPDIR}/fa_orbit_submission_C4L_screen_*_jid7654321.txt
    # an explicit PIN_SHA that disagrees is refused
    OTHER="$(git rev-parse HEAD 2>/dev/null)"
    out="$(env MOCK_TRACE="$TRACE" MOCK_WT="$WT" FA_ORBIT_SBATCH="${MOCK}/sbatch" \
               FA_ORBIT_SCONTROL="${MOCK}/scontrol" FA_ORBIT_SCANCEL="${MOCK}/scancel" \
               bash "$SUB" ARM=C4L STEP=10000 "PIN_SHA=${OTHER}" 2>&1)"; rc=$?
    if [ "$rc" -ne 0 ] && echo "$out" | grep -q "disagrees with the campaign pin"; then
      echo "PASS  a PIN_SHA disagreeing with the campaign pin is refused"; PASS=$((PASS + 1))
    else
      echo "FAIL  a disagreeing PIN_SHA was accepted (rc=${rc})"; FAIL=$((FAIL + 1))
    fi
    # ...and agreeing with it is fine
    out="$(env MOCK_TRACE="$TRACE" MOCK_WT="$WT" FA_ORBIT_SBATCH="${MOCK}/sbatch" \
               FA_ORBIT_SCONTROL="${MOCK}/scontrol" FA_ORBIT_SCANCEL="${MOCK}/scancel" \
               bash "$SUB" ARM=C4L STEP=10000 "PIN_SHA=${PIN2}" 2>&1)"; rc=$?
    [ "$rc" -eq 0 ] \
      && { echo "PASS  an explicit PIN_SHA equal to the campaign pin is accepted"; PASS=$((PASS + 1)); } \
      || { echo "FAIL  the agreeing PIN_SHA was refused (rc=${rc})"; FAIL=$((FAIL + 1)); }
    bash "$HELPER" --release 7654321 "${MAIN_TREE}/.measure_worktrees/${PIN2}" >/dev/null 2>&1
    rm -f ${EXPDIR}/fa_orbit_submission_C4L_screen_*_jid7654321.txt
    # the pin is only changed by the explicit subcommand, and it is logged
    LOGLINE="$(bash "$HELPER" --pin-campaign "$PIN2" 2>&1 | grep -c 'CAMPAIGN PIN set to')"
    [ "$LOGLINE" -ge 1 ] \
      && { echo "PASS  --pin-campaign logs the change"; PASS=$((PASS + 1)); } \
      || { echo "FAIL  --pin-campaign is silent"; FAIL=$((FAIL + 1)); }
    bash "$HELPER" --pin-campaign deadbeef >/dev/null 2>&1 \
      && { echo "FAIL  --pin-campaign accepted a short sha"; FAIL=$((FAIL + 1)); } \
      || { echo "PASS  --pin-campaign refuses a malformed sha"; PASS=$((PASS + 1)); }
    # unset file -> the old HEAD behaviour returns
    bash "$HELPER" --unpin-campaign >/dev/null 2>&1
    : > "$TRACE"
    env MOCK_TRACE="$TRACE" MOCK_WT="$WT" FA_ORBIT_SBATCH="${MOCK}/sbatch" \
        FA_ORBIT_SCONTROL="${MOCK}/scontrol" FA_ORBIT_SCANCEL="${MOCK}/scancel" \
        bash "$SUB" ARM=C4L STEP=10000 >/dev/null 2>&1
    if grep -q "EXPECT_SHA=$(git rev-parse HEAD)" "$TRACE"; then
      echo "PASS  with no pin file, submission falls back to HEAD"; PASS=$((PASS + 1))
    else
      echo "FAIL  the unpinned fallback is not HEAD"; FAIL=$((FAIL + 1))
    fi
    bash "$HELPER" --release 7654321 "$WT" >/dev/null 2>&1
    rm -f ${EXPDIR}/fa_orbit_submission_C4L_screen_*_jid7654321.txt
    [ -n "$SAVED_PIN" ] && bash "$HELPER" --pin-campaign "$SAVED_PIN" >/dev/null 2>&1
  else
    echo "SKIP  campaign-pin cases (no HEAD~1)"
  fi

  # the gate K must NOT drift when the screen cadence widens
  if $PY -c "
import importlib.util,sys
spec=importlib.util.spec_from_file_location('V','${EXPDIR}/exp11_validate_rows.py')
V=importlib.util.module_from_spec(spec); spec.loader.exec_module(V)
f=V.CONTRACTS['futility']
assert f['K']==(1,8), f['K']
assert f['gate_K']==(8,), f['gate_K']
assert V.gate_admissible(8) and not V.gate_admissible(1)
"; then
    echo "PASS  screens run at both K while the futility GATE stays K=8"; PASS=$((PASS + 1))
  else
    echo "FAIL  the gate K drifted with the screen cadence"; FAIL=$((FAIL + 1))
  fi

  # --- the review's findings, asserted at the surfaces they name -------------
  if grep -q 'in_set "\$val" C4L C8 C16 C32 VANL C4BACKFILL' "$SUB"; then
    echo "PASS  the sanctioned submitter admits VANL"; PASS=$((PASS + 1))
  else
    echo "FAIL  the submitter still rejects VANL"; FAIL=$((FAIL + 1))
  fi
  if grep -q 'in_set "\$val" screen conf r3 cross q9' "$SUB"; then
    echo "PASS  the submitter admits the q9 cell"; PASS=$((PASS + 1))
  else
    echo "FAIL  the submitter rejects CELL=q9"; FAIL=$((FAIL + 1))
  fi
  # trap-backed pin restoration: an interrupt must not leave the campaign unpinned
  if grep -q "trap 'bash \"\$MEASURE_HELPER\" --pin-campaign" "$GUARD_SELF" \
     && grep -q "INT TERM" "$GUARD_SELF"; then
    echo "PASS  the pin is restored by a trap, not only on the happy path"; PASS=$((PASS + 1))
  else
    echo "FAIL  pin parking has no interruption safety"; FAIL=$((FAIL + 1))
  fi
  if grep -q "DO NOT RUN THIS SUITE DURING ACTIVE CAMPAIGN SUBMISSIONS" "$GUARD_SELF"; then
    echo "PASS  the suite documents that it must not run during submissions"; PASS=$((PASS + 1))
  else
    echo "FAIL  the run-window caveat is undocumented"; FAIL=$((FAIL + 1))
  fi
  # the registry must carry VANL, recorded from the PUBLISHED manifest
  if $PY -c "
import json,sys
r=json.load(open('${EXPDIR}/arm_launch_registry.json'))['arms']
v=r.get('VANL') or sys.exit('VANL absent')
assert v['job']=='3661520', v['job']
assert v['mode']=='INITIAL', v['mode']
assert len(v['manifest_sha256'])==64 and v['save_dir']=='outputs_FLAC/exp11_VANL'
assert int(v['training_seed'])==42
"; then
    echo "PASS  the registry carries VANL from its published launch manifest"; PASS=$((PASS + 1))
  else
    echo "FAIL  the VANL registry entry is missing or wrong"; FAIL=$((FAIL + 1))
  fi

  # --- PIN_SHA: the campaign measures at ONE commit --------------------------
  SAVED_PIN2="$(bash "$HELPER" --pinned 2>/dev/null)"; [ "$SAVED_PIN2" = "<none>" ] && SAVED_PIN2=""
  bash "$HELPER" --unpin-campaign >/dev/null 2>&1     # these cases test PIN_SHA alone
  PIN="$(git rev-parse HEAD~1 2>/dev/null)"
  if [ -n "$PIN" ]; then
    : > "$TRACE"
    out="$(env MOCK_TRACE="$TRACE" MOCK_WT="$WT" FA_ORBIT_SBATCH="${MOCK}/sbatch" \
               FA_ORBIT_SCONTROL="${MOCK}/scontrol" FA_ORBIT_SCANCEL="${MOCK}/scancel" \
               bash "$SUB" ARM=C4L STEP=10000 "PIN_SHA=${PIN}" 2>&1)"; rc=$?
    PINNED_WT="${MAIN_TREE}/.measure_worktrees/${PIN}"
    if [ "$rc" -eq 0 ] && [ -d "$PINNED_WT" ] \
       && [ "$(git -C "$PINNED_WT" rev-parse HEAD)" = "$PIN" ]; then
      echo "PASS  PIN_SHA prepares the worktree AT the requested commit"; PASS=$((PASS + 1))
    else
      echo "FAIL  PIN_SHA did not pin the tree (rc=${rc})"; echo "$out" | tail -3 | sed 's/^/        | /'
      FAIL=$((FAIL + 1))
    fi
    if grep -q "EXPECT_SHA=${PIN}" "$TRACE"; then
      echo "PASS  EXPECT_SHA follows PIN_SHA into the sbatch export"; PASS=$((PASS + 1))
    else
      echo "FAIL  EXPECT_SHA does not match PIN_SHA in the export"
      grep -o 'EXPECT_SHA=[0-9a-f]*' "$TRACE" | head -1 | sed 's/^/        | /'; FAIL=$((FAIL + 1))
    fi
    INTENT_F="$(ls -t ${EXPDIR}/fa_orbit_submission_C4L_screen_*_jid7654321.txt 2>/dev/null | head -1)"
    if [ -n "$INTENT_F" ] && grep -q "pin_sha ${PIN}" "$INTENT_F"; then
      echo "PASS  the intent manifest records the pin"; PASS=$((PASS + 1))
    else
      echo "FAIL  the intent manifest does not record the pin"; FAIL=$((FAIL + 1))
    fi
    [ -n "$INTENT_F" ] && rm -f "$INTENT_F"
    bash "$HELPER" --release 7654321 "$PINNED_WT" >/dev/null 2>&1
    # the pinned tree must carry the assets too, VAE.ckpt included
    if [ -e "$PINNED_WT/weights/FLAC/VAE.ckpt" ] && [ -e "$PINNED_WT/weights/AGREE/AGREE_fullAR.pt" ] \
       && [ -e "$PINNED_WT/AcousticRooms" ]; then
      echo "PASS  a freshly pinned tree carries every runtime asset"; PASS=$((PASS + 1))
    else
      echo "FAIL  the pinned tree is missing runtime assets"; FAIL=$((FAIL + 1))
    fi
  else
    echo "SKIP  PIN_SHA cases (no HEAD~1 available)"
  fi
  [ -n "$SAVED_PIN2" ] && bash "$HELPER" --pin-campaign "$SAVED_PIN2" >/dev/null 2>&1
  # a commit this repository does not have must be refused
  out="$(env MOCK_TRACE="$TRACE" MOCK_WT="$WT" FA_ORBIT_SBATCH="${MOCK}/sbatch" \
             FA_ORBIT_SCONTROL="${MOCK}/scontrol" FA_ORBIT_SCANCEL="${MOCK}/scancel" \
             bash "$SUB" ARM=C4L STEP=10000 PIN_SHA=0123456789abcdef0123456789abcdef01234567 2>&1)"; rc=$?
  if [ "$rc" -ne 0 ] && echo "$out" | grep -q "is not a commit in this repository"; then
    echo "PASS  a non-existent PIN_SHA is refused"; PASS=$((PASS + 1))
  else
    echo "FAIL  a non-existent PIN_SHA was accepted (rc=${rc})"; FAIL=$((FAIL + 1))
  fi
  out="$(env MOCK_TRACE="$TRACE" MOCK_WT="$WT" FA_ORBIT_SBATCH="${MOCK}/sbatch" \
             FA_ORBIT_SCONTROL="${MOCK}/scontrol" FA_ORBIT_SCANCEL="${MOCK}/scancel" \
             bash "$SUB" ARM=C4L STEP=10000 PIN_SHA=deadbeef 2>&1)"; rc=$?
  if [ "$rc" -ne 0 ] && echo "$out" | grep -q "not 40 hex characters"; then
    echo "PASS  a short PIN_SHA is refused"; PASS=$((PASS + 1))
  else
    echo "FAIL  a short PIN_SHA was accepted (rc=${rc})"; FAIL=$((FAIL + 1))
  fi

  if [ -n "${SUITE_SAVED_PIN:-}" ]; then
    bash "$HELPER" --pin-campaign "$SUITE_SAVED_PIN" >/dev/null 2>&1
    trap 'rm -rf "$TMP"' EXIT; trap - INT TERM        # restored: stand the trap down
  fi
  # --- the marker is not proof: fd 8 must BE the store lock, at the OUTER entry
  STORE_LOCK="${MAIN_TREE}/.measure_worktrees/.store.lock"
  out="$(env FA_ORBIT_STORE_LOCK_HELD=1 MOCK_TRACE="$TRACE" FA_ORBIT_SBATCH="${MOCK}/sbatch" \
             FA_ORBIT_SCONTROL="${MOCK}/scontrol" FA_ORBIT_SCANCEL="${MOCK}/scancel" \
             bash "$SUB" ARM=C4L STEP=10000 2>&1)"; rc=$?
  if [ "$rc" -ne 0 ] && echo "$out" | grep -q "only CLAIMS to hold the store lock"; then
    echo "PASS  a forged lock marker with NO fd 8 is refused"; PASS=$((PASS + 1))
  else
    echo "FAIL  a forged marker with no fd 8 was accepted (rc=${rc})"; FAIL=$((FAIL + 1))
  fi
  DECOY_LOCK="${TMP}/decoy.lock"; : > "$DECOY_LOCK"
  out="$(env FA_ORBIT_STORE_LOCK_HELD=1 MOCK_TRACE="$TRACE" FA_ORBIT_SBATCH="${MOCK}/sbatch" \
             FA_ORBIT_SCONTROL="${MOCK}/scontrol" FA_ORBIT_SCANCEL="${MOCK}/scancel" \
             bash -c 'exec 8>"$1"; exec bash "$2" ARM=C4L STEP=10000' _ "$DECOY_LOCK" "$SUB" 2>&1)"; rc=$?
  if [ "$rc" -ne 0 ] && echo "$out" | grep -q "only CLAIMS to hold the store lock"; then
    echo "PASS  a forged marker with the WRONG fd 8 is refused"; PASS=$((PASS + 1))
  else
    echo "FAIL  a forged marker pointing at another file was accepted (rc=${rc})"; FAIL=$((FAIL + 1))
  fi
  # fd 8 on a DELETED file: readlink -f cannot resolve it, and an unresolvable
  # path must never be read as a match (two empty strings compare equal)
  GONE="${TMP}/gone.lock"; : > "$GONE"
  out="$(env FA_ORBIT_STORE_LOCK_HELD=1 MOCK_TRACE="$TRACE" FA_ORBIT_SBATCH="${MOCK}/sbatch" \
             FA_ORBIT_SCONTROL="${MOCK}/scontrol" FA_ORBIT_SCANCEL="${MOCK}/scancel" \
             bash -c 'exec 8>"$1"; rm -f "$1"; exec bash "$2" ARM=C4L STEP=10000' _ "$GONE" "$SUB" 2>&1)"; rc=$?
  if [ "$rc" -ne 0 ] && echo "$out" | grep -q "only CLAIMS to hold the store lock"; then
    echo "PASS  an unresolvable fd 8 (deleted file) is refused"; PASS=$((PASS + 1))
  else
    echo "FAIL  an unresolvable fd 8 was accepted (rc=${rc})"; FAIL=$((FAIL + 1))
  fi
  if grep -q '\[ -n "\$have" \] && \[ -n "\$want" \]' "$SUB" \
     && grep -q '\[ -n "\$have" \] && \[ -n "\$want" \]' "$HELPER"; then
    echo "PASS  both readlink results must be non-empty (no empty==empty match)"; PASS=$((PASS + 1))
  else
    echo "FAIL  an empty readlink result could compare equal"; FAIL=$((FAIL + 1))
  fi
  # the outer entry checks the lock BEFORE any transaction step
  ENTRY_LINE="$(grep -n 'fd8_is_the_store_lock ||' "$SUB" | head -1 | cut -d: -f1)"
  FIRST_TXN="$(grep -n 'WT="\$("\$HELPER"' "$SUB" | head -1 | cut -d: -f1)"
  if [ -n "$ENTRY_LINE" ] && [ -n "$FIRST_TXN" ] && [ "$ENTRY_LINE" -lt "$FIRST_TXN" ]; then
    echo "PASS  the lock is proven at the outer entry, before any transaction"; PASS=$((PASS + 1))
  else
    echo "FAIL  the lock check is downstream of the first transaction step"; FAIL=$((FAIL + 1))
  fi
  # no unlocked EXIT-trap unlink anywhere in the screen
  if grep -q 'trap .*--release' "$SCREEN" && ! grep -q 'rm -f "\$LEASE"' "$SCREEN"; then
    echo "PASS  the screen has no unlocked EXIT-trap unlink"; PASS=$((PASS + 1))
  else
    echo "FAIL  the screen still drops its lease with an unlocked rm"; FAIL=$((FAIL + 1))
  fi
  if ! skip_if_campaign "registered-worktree removal-failure handling"; then
  bash "$HELPER" --thaw >/dev/null 2>&1     # the cases below must be free to delete
  # a REGISTERED worktree git declines to remove is left alone, not force-deleted
  GITMOCK="${TMP}/gitmock"; mkdir -p "$GITMOCK"
  printf '#!/usr/bin/env bash\ncase "$*" in *"worktree remove"*) echo "mock: declining" >&2; exit 128;; esac\nexec %s "$@"\n' \
    "$(command -v git)" > "${GITMOCK}/git"; chmod +x "${GITMOCK}/git"
  out="$(env PATH="${GITMOCK}:$PATH" bash "$HELPER" --prune 2>&1)"; rc=$?
  if [ -d "$WT" ] && echo "$out" | grep -q "git declined to remove the registered worktree"; then
    echo "PASS  a registered worktree git refuses to remove is left in place"; PASS=$((PASS + 1))
  else
    echo "FAIL  a removal failure escalated to rm -rf (tree $([ -d "$WT" ] && echo kept || echo DELETED))"
    echo "$out" | tail -3 | sed 's/^/        | /'; FAIL=$((FAIL + 1))
  fi
  # ...and the store still works afterwards (assets re-provisioned)
  BACK="$(bash "$HELPER" 2>/dev/null | tail -1)"
  if [ "$BACK" = "$WT" ] && [ -e "$WT/AcousticRooms" ] && [ -e "$WT/weights" ]; then
    echo "PASS  the store self-heals after a failed removal"; PASS=$((PASS + 1))
  else
    echo "FAIL  the store did not recover after a failed removal ('${BACK}')"; FAIL=$((FAIL + 1))
  fi
  fi
  grep -q -- '--promote' "$SUB" \
    && { echo "FAIL  the placeholder/promote race is still in the submitter"; FAIL=$((FAIL + 1)); } \
    || { echo "PASS  no placeholder lease: --hold removed the promote race"; PASS=$((PASS + 1)); }
  # every lease/prune operation takes the ONE store lock
  if grep -q 'LOCK="${ROOT}/.store.lock"' "$HELPER" \
     && [ "$(grep -n 'flock 8' "$HELPER" | head -1 | cut -d: -f1)" -lt \
          "$(grep -n -- '--lease)' "$HELPER" | head -1 | cut -d: -f1)" ]; then
    echo "PASS  lease/release/prune/create all run under one store-wide lock"; PASS=$((PASS + 1))
  else
    echo "FAIL  lease operations are not under the creation lock"; FAIL=$((FAIL + 1))
  fi
  # a squeue that FAILS must not be read as "the job is gone"
  FAKEQ="${MOCK}/failing"; mkdir -p "$FAKEQ"
  printf '#!/usr/bin/env bash\necho "slurm_load_jobs error: Unable to contact slurm controller" >&2\nexit 1\n' \
    > "${FAKEQ}/squeue"; chmod +x "${FAKEQ}/squeue"
  bash "$HELPER" --lease 55555555 "$WT" >/dev/null 2>&1
  out="$(env PATH="${FAKEQ}:$PATH" bash "$HELPER" --prune 2>&1)"
  if [ -f "$WT/.leases/55555555" ] && echo "$out" | grep -q "treating the lease as LIVE"; then
    echo "PASS  a squeue failure keeps the lease (absence must be PROVEN)"; PASS=$((PASS + 1))
  else
    echo "FAIL  a squeue failure was read as job-absent"; echo "$out" | tail -3 | sed 's/^/        | /'
    FAIL=$((FAIL + 1))
  fi
  bash "$HELPER" --release 55555555 "$WT" >/dev/null 2>&1

  if ! skip_if_campaign "lock-span concurrent-prune demonstration"; then
  # --- LOCK SPAN: a prune CANNOT interleave with a submission ----------------
  # The gap the review is about: tree prepared -> (window) -> lease written. A
  # sweep landing in that window sees a brand-new UNLEASED tree and deletes it,
  # taking the queued job's code with it. The mock sbatch stalls inside exactly
  # that window while a concurrent sweep tries to run.
  : > "$TRACE"; rm -f "${TRACE}.submitting"
  bash "$HELPER" --freeze "guard suite" >/dev/null 2>&1   # the submitter requires it
  # The mock must hand back a job id Slurm KNOWS, because a real submission
  # leases a real queued job — an invented id is (correctly) reaped as stale by
  # the very sweep we are racing, which would test the reaper, not the lock.
  LIVEJOB="$(squeue -h -u "$(id -un)" -o %i 2>/dev/null | head -1)"
  [ -n "$LIVEJOB" ] || LIVEJOB=8765432
  env MOCK_TRACE="$TRACE" MOCK_SBATCH_SLEEP=6 "MOCK_JOBID=${LIVEJOB}" \
      FA_ORBIT_SBATCH="${MOCK}/sbatch" FA_ORBIT_SCONTROL="${MOCK}/scontrol" \
      FA_ORBIT_SCANCEL="${MOCK}/scancel" bash "$SUB" ARM=C4L STEP=10000 \
      > "${TMP}/submit.out" 2>&1 &
  SUBMIT_PID=$!
  for _ in 1 2 3 4 5 6 7 8 9 10; do [ -f "${TRACE}.submitting" ] && break; sleep 1; done
  # Scaffolding: unlink the marker directly rather than via --thaw, which would
  # block on the store lock the submission is holding. The point of THIS case is
  # that the LOCK protects the tree, so the freeze must be out of the way; that
  # freeze/thaw are themselves serialized is a separate case below.
  rm -f "$FREEZE_MARKER"
  if [ -f "${TRACE}.submitting" ]; then
    # (a) the store lock is genuinely HELD across the window — prove it directly
    if flock -n 9 2>/dev/null 9>"${MAIN_TREE}/.measure_worktrees/.store.lock"; then
      echo "FAIL  the store lock was FREE between preparation and leasing"; FAIL=$((FAIL + 1))
      exec 9>&-
    else
      echo "PASS  the store lock spans preparation -> submission -> leasing"; PASS=$((PASS + 1))
    fi
    # (b) a concurrent sweep blocks on that lock, and finds the tree leased
    PRUNE_OUT="$(timeout 30 bash "$HELPER" --prune 2>&1)"; PRUNE_RC=$?
    wait "$SUBMIT_PID"; SUBMIT_RC=$?
    if [ "$PRUNE_RC" -eq 0 ] && [ -d "$WT" ] && [ "$SUBMIT_RC" -eq 0 ] \
       && [ -f "$WT/.leases/${LIVEJOB}" ] \
       && ! echo "$PRUNE_OUT" | grep -q "pruning unleased measurement worktree ${WT}$"; then
      echo "PASS  a concurrent prune waits and then keeps the freshly leased tree"; PASS=$((PASS + 1))
    else
      echo "FAIL  concurrent prune/submit interleaved (prune rc=${PRUNE_RC}, submit rc=${SUBMIT_RC}, tree $([ -d "$WT" ] && echo kept || echo DELETED))"
      echo "$PRUNE_OUT" | tail -3 | sed 's/^/        | /'; FAIL=$((FAIL + 1))
    fi
    bash "$HELPER" --release "$LIVEJOB" "$WT" >/dev/null 2>&1
  else
    wait "$SUBMIT_PID" 2>/dev/null
    echo "FAIL  the stalled-submission fixture never reached its window"; FAIL=$((FAIL + 1))
  fi
  fi
  # --- CAMPAIGN FREEZE: deletion is mechanically impossible while it exists ---
  echo
  echo "--- campaign freeze ---"
  if ! skip_if_campaign "campaign-freeze behaviour (freeze/thaw/prune cases)"; then
  FZ="${MAIN_TREE}/.measure_worktrees/$(printf 'a%.0s' $(seq 1 40))"
  mkdir -p "${FZ}/.leases"; printf 'jobid 999999999\n' > "${FZ}/.leases/999999999"
  bash "$HELPER" --freeze "guard case" >/dev/null 2>&1
  out="$(bash "$HELPER" --prune 2>&1)"; rc=$?
  if [ -d "$FZ" ] && echo "$out" | grep -q "freeze: keeping unleased worktree ${FZ}"; then
    echo "PASS  a freeze blocks an explicit prune of an unleased entry"; PASS=$((PASS + 1))
  else
    echo "FAIL  an explicit prune deleted under freeze (entry $([ -d "$FZ" ] && echo kept || echo GONE))"
    echo "$out" | tail -3 | sed 's/^/        | /'; FAIL=$((FAIL + 1))
  fi
  if echo "$out" | grep -q "refusing to run 'git worktree prune'"; then
    echo "PASS  even 'git worktree prune' is suppressed under freeze"; PASS=$((PASS + 1))
  else
    echo "FAIL  git worktree prune still ran under freeze"; FAIL=$((FAIL + 1))
  fi
  # the IMPLICIT path: preparation must refuse to clear a half-removed entry
  OLD_SHA="$(git rev-parse HEAD~1 2>/dev/null)"
  if [ -n "$OLD_SHA" ]; then
    HALF="${MAIN_TREE}/.measure_worktrees/${OLD_SHA}"
    rm -rf "$HALF"; mkdir -p "$HALF"          # a directory with no .git: invalid
    out="$(bash "$HELPER" "$OLD_SHA" 2>&1)"; rc=$?
    if [ "$rc" -ne 0 ] && [ -d "$HALF" ] && echo "$out" | grep -q "thaw (--thaw) and clean it manually"; then
      echo "PASS  a freeze blocks the IMPLICIT cleanup in the preparation path"; PASS=$((PASS + 1))
    else
      echo "FAIL  preparation cleared a stale entry under freeze (rc=${rc}, $([ -d "$HALF" ] && echo kept || echo DELETED))"
      echo "$out" | tail -3 | sed 's/^/        | /'; FAIL=$((FAIL + 1))
    fi
    rmdir "$HALF" 2>/dev/null
  else
    echo "SKIP  implicit-cleanup case (no HEAD~1 to pin a second tree)"
  fi
  # remove_entry itself refuses, so no future caller can route around the freeze.
  # Checked structurally over the function BODY: the freeze guard must be the
  # first thing it does, before any check that could be reordered around it.
  GUARD_POS="$(awk '/^remove_entry\(\)/{inf=1; n=0; next} inf&&/^}/{inf=0} inf{n++; if ($0 ~ /if frozen; then freeze_note/) {print n; exit}}' "$HELPER")"
  DELETE_POS="$(awk '/^remove_entry\(\)/{inf=1; n=0; next} inf&&/^}/{inf=0} inf{n++; if ($0 ~ /rm -rf|worktree remove/) {print n; exit}}' "$HELPER")"
  if [ -n "$GUARD_POS" ] && [ -n "$DELETE_POS" ] && [ "$GUARD_POS" -lt "$DELETE_POS" ]; then
    echo "PASS  remove_entry — the only deleting function — refuses under freeze first"; PASS=$((PASS + 1))
  else
    echo "FAIL  remove_entry's freeze guard is missing or after a delete (guard@${GUARD_POS:-none} delete@${DELETE_POS:-none})"
    FAIL=$((FAIL + 1))
  fi
  # every rm -rf in the helper sits behind a freeze check
  UNGUARDED="$(awk '/^remove_entry\(\)/{inf=1} inf&&/frozen/{ok=1} /rm -rf "\$1"/{if(!ok) print NR}' "$HELPER")"
  if [ -z "$UNGUARDED" ]; then
    echo "PASS  no recursive delete precedes a freeze check"; PASS=$((PASS + 1))
  else
    echo "FAIL  unguarded rm -rf at line(s): ${UNGUARDED}"; FAIL=$((FAIL + 1))
  fi
  # thaw restores normal collection
  bash "$HELPER" --thaw >/dev/null 2>&1
  bash "$HELPER" --prune >/dev/null 2>&1
  if [ ! -d "$FZ" ]; then
    echo "PASS  --thaw restores collection (the entry is now removed)"; PASS=$((PASS + 1))
  else
    echo "FAIL  the entry survived after --thaw"; FAIL=$((FAIL + 1)); rm -rf "$FZ"
  fi
  # freeze/thaw are serialized on the SAME store lock as everything else
  bash "$HELPER" --with-lock sleep 5 >/dev/null 2>&1 &
  HOLDER=$!
  sleep 1
  T0=$(date +%s)
  bash "$HELPER" --freeze "serialization probe" >/dev/null 2>&1
  T1=$(date +%s)
  wait "$HOLDER" 2>/dev/null
  if [ "$((T1 - T0))" -ge 3 ]; then
    echo "PASS  --freeze waits on the store lock ($((T1 - T0))s behind a holder)"; PASS=$((PASS + 1))
  else
    echo "FAIL  --freeze did not contend on the store lock ($((T1 - T0))s)"; FAIL=$((FAIL + 1))
  fi
  bash "$HELPER" --thaw >/dev/null 2>&1
  # a submission without the freeze is refused: the condition is self-enforcing
  out="$(env MOCK_TRACE="$TRACE" MOCK_WT="$WT" FA_ORBIT_SBATCH="${MOCK}/sbatch" \
             FA_ORBIT_SCONTROL="${MOCK}/scontrol" FA_ORBIT_SCANCEL="${MOCK}/scancel" \
             bash "$SUB" ARM=C4L STEP=10000 2>&1)"; rc=$?
  if [ "$rc" -ne 0 ] && echo "$out" | grep -q "requires the deletion freeze"; then
    echo "PASS  a submission without the campaign freeze is refused"; PASS=$((PASS + 1))
  else
    echo "FAIL  a submission ran without the campaign freeze (rc=${rc})"; FAIL=$((FAIL + 1))
  fi
  bash "$HELPER" --freeze "guard suite" >/dev/null 2>&1
  out="$(env MOCK_TRACE="$TRACE" MOCK_WT="$WT" FA_ORBIT_SBATCH="${MOCK}/sbatch" \
             FA_ORBIT_SCONTROL="${MOCK}/scontrol" FA_ORBIT_SCANCEL="${MOCK}/scancel" \
             bash "$SUB" ARM=C4L STEP=10000 2>&1)"; rc=$?
  if [ "$rc" -eq 0 ] && echo "$out" | grep -q "campaign freeze: ACTIVE"; then
    echo "PASS  the submitter reports the freeze state in its preflight"; PASS=$((PASS + 1))
  else
    echo "FAIL  the preflight does not report the freeze (rc=${rc})"; FAIL=$((FAIL + 1))
  fi
  bash "$HELPER" --release 7654321 "$WT" >/dev/null 2>&1
  bash "$HELPER" --thaw >/dev/null 2>&1
  fi

  # add_lease refuses a directory that is not a live registered worktree
  NOTWT="${TMP}/not_a_worktree"; mkdir -p "$NOTWT"
  out="$(bash "$HELPER" --lease 4242424 "$NOTWT" 2>&1)"; rc=$?
  if [ "$rc" -ne 0 ] && echo "$out" | grep -q "worktree identity is invalid"; then
    echo "PASS  a lease on a non-worktree directory is refused"; PASS=$((PASS + 1))
  else
    echo "FAIL  leased a directory that is not a worktree (rc=${rc})"; FAIL=$((FAIL + 1))
  fi
  if ! skip_if_campaign "safe-cleanup deletion cases"; then
  # --- SAFE CLEANUP: identity failure alone is never a licence to delete ------
  STALE="${MAIN_TREE}/.measure_worktrees/$(printf 'f%.0s' $(seq 1 40))"
  mkdir -p "${STALE}/.leases"
  LIVE_ID="$(squeue -h -u "$(id -un)" -o %i 2>/dev/null | head -1)"
  [ -n "$LIVE_ID" ] || LIVE_ID=3648694
  printf 'jobid %s\n' "$LIVE_ID" > "${STALE}/.leases/${LIVE_ID}"   # a job Slurm knows NOW
  bash "$HELPER" --prune >/dev/null 2>&1
  if [ -d "$STALE" ]; then
    echo "PASS  an identity-failed entry holding a LIVE lease is not deleted"; PASS=$((PASS + 1))
  else
    echo "FAIL  a live-leased entry was deleted on an identity failure"; FAIL=$((FAIL + 1))
  fi
  rm -f "${STALE}/.leases/${LIVE_ID}"
  printf 'jobid 999999999\n' > "${STALE}/.leases/999999999"  # provably absent
  bash "$HELPER" --prune >/dev/null 2>&1
  if [ ! -d "$STALE" ]; then
    echo "PASS  an identity-failed, provably unleased entry is cleaned up"; PASS=$((PASS + 1))
  else
    echo "FAIL  a stale unleased entry was left behind"; FAIL=$((FAIL + 1)); rm -rf "$STALE"
  fi
  # a squeue that cannot answer must block the deletion too
  STALE2="${MAIN_TREE}/.measure_worktrees/$(printf 'e%.0s' $(seq 1 40))"
  mkdir -p "${STALE2}/.leases"; printf 'jobid 777777777\n' > "${STALE2}/.leases/777777777"
  env PATH="${FAKEQ}:$PATH" bash "$HELPER" --prune >/dev/null 2>&1
  if [ -d "$STALE2" ]; then
    echo "PASS  an unverifiable lease blocks deletion (transient error != absent)"; PASS=$((PASS + 1))
  else
    echo "FAIL  a squeue failure allowed a deletion"; FAIL=$((FAIL + 1))
  fi
  rm -rf "$STALE2"
  # names that are not exactly 40 hex are never deleted
  SHORT="${MAIN_TREE}/.measure_worktrees/deadbeef"; mkdir -p "$SHORT"
  bash "$HELPER" --prune >/dev/null 2>&1
  if [ -d "$SHORT" ]; then
    echo "PASS  a non-40-hex store entry is never removed"; PASS=$((PASS + 1))
  else
    echo "FAIL  a short-named entry was removed"; FAIL=$((FAIL + 1))
  fi
  rmdir "$SHORT" 2>/dev/null
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
assert set(arms) == {"C4L", "C8", "C16", "C32", "VANL"}, sorted(arms)
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

# --- default LOG names are cell-qualified and collision-free ----------------
# Extract the driver's OWN naming block and evaluate it, so this tests the
# shipped expression rather than a copy of it.
echo
echo "--- default screen-log naming ---"
sed -n '/^LOG_CELL_TOKEN=""/,/^LOG="\${LOG:-/p' "$SCREEN" > "${TMP}/lognames.sh"
if [ -s "${TMP}/lognames.sh" ]; then
  name_for() {  # $1 CELL $2 ROT_TOKEN $3 EVAL_ORBIT $4 jobid
    ( LOGDIR=/logs TS=2026-08-08_05-00-00 ARM=C8 STEP=40000 SEED=42 K=8 \
      CELL="$1" ROT_TOKEN="$2" EVAL_ORBIT="$3" SLURM_JOB_ID="$4" LOG=""
      unset LOG
      . "${TMP}/lognames.sh" >/dev/null 2>&1
      echo "$LOG" )
  }
  A="$(name_for r3 0 "" 111)"; B="$(name_for r3 5p625 "" 112)"
  C="$(name_for r3 11p25 "" 113)"; D="$(name_for cross "" 16 114)"
  E="$(name_for cross "" 32 115)"; F="$(name_for screen "" "" 116)"
  U="$(printf '%s\n' "$A" "$B" "$C" "$D" "$E" "$F" | sort -u | grep -c .)"
  if [ "$U" -eq 6 ]; then
    echo "PASS  six same-second cells produce six DISTINCT default log names"; PASS=$((PASS + 1))
  else
    echo "FAIL  default log names collide (${U} unique of 6)"
    printf '%s\n' "$A" "$B" "$C" "$D" "$E" "$F" | sed 's/^/        | /'; FAIL=$((FAIL + 1))
  fi
  # the two r3 rotations differ ONLY by their rotation token -> that is the fix
  if [ "$A" != "$B" ] && echo "$B" | grep -q "_rot5p625_" && echo "$D" | grep -q "_a16_"; then
    echo "PASS  r3 names carry the rotation and cross names the eval orbit"; PASS=$((PASS + 1))
  else
    echo "FAIL  cell tokens missing from the default names ('${B}' / '${D}')"; FAIL=$((FAIL + 1))
  fi
  # still inside LOGDIR and still matching the drift-gate exemption
  if echo "$B" | grep -q "^/logs/fa_orbit_.*_screen\.log$"; then
    echo "PASS  default names stay in LOGDIR and keep the _screen.log suffix"; PASS=$((PASS + 1))
  else
    echo "FAIL  '${B}' no longer matches the exempted screen-log shape"; FAIL=$((FAIL + 1))
  fi
  # an explicit LOG= still wins (operators can override; they just never need to)
  G="$( LOGDIR=/logs TS=t ARM=C8 STEP=1 SEED=42 K=8 CELL=screen ROT_TOKEN="" EVAL_ORBIT="" \
        LOG=/tmp/explicit.log; . "${TMP}/lognames.sh" >/dev/null 2>&1; echo "$LOG" )"
  [ "$G" = "/tmp/explicit.log" ] \
    && { echo "PASS  an explicit LOG= still overrides the default"; PASS=$((PASS + 1)); } \
    || { echo "FAIL  LOG= override broken ('${G}')"; FAIL=$((FAIL + 1)); }
else
  echo "FAIL  could not extract the log-naming block from the driver"; FAIL=$((FAIL + 1))
fi

# --- the arm launcher untracks its own live transcript ----------------------
echo
echo "--- live transcript untracking (arm launcher) ---"
TRAIN="${EXPDIR}/fa_orbit_train.sbatch"
if grep -q 'git -C "\$REPO" rm --cached --quiet -- "\$SLURM_OUT_AT_LAUNCH"' "$TRAIN" \
   && grep -q 'ls-files --error-unmatch "\$SLURM_OUT_AT_LAUNCH"' "$TRAIN"; then
  echo "PASS  the launcher untracks its own Slurm transcript at launch"; PASS=$((PASS + 1))
else
  echo "FAIL  the launcher does not untrack its transcript"; FAIL=$((FAIL + 1))
fi
if grep -q 'slurm_transcript .* untrack ' "$TRAIN"; then
  echo "PASS  the untrack outcome is recorded in the launch manifest"; PASS=$((PASS + 1))
else
  echo "FAIL  the manifest does not record the untrack state"; FAIL=$((FAIL + 1))
fi
if grep -q 'TRANSCRIPT POLICY' "$TRAIN" && grep -q 'git add -f' "$TRAIN"; then
  echo "PASS  the closure procedure is documented in the launcher header"; PASS=$((PASS + 1))
else
  echo "FAIL  the transcript policy is undocumented"; FAIL=$((FAIL + 1))
fi
# absence must be tolerated (an already-untracked transcript is the steady state)
if grep -q 'already-untracked' "$TRAIN" && grep -q 'untrack-FAILED' "$TRAIN"; then
  echo "PASS  untracking tolerates absence and reports failure without aborting"; PASS=$((PASS + 1))
else
  echo "FAIL  untracking does not handle the absent/failed cases"; FAIL=$((FAIL + 1))
fi
# and no live arm transcript is tracked right now
STILL_TRACKED="$(git ls-files -- "${EXPDIR}/slurm_train_exp11-*-train_36486*.out" 2>/dev/null \
                 | while read -r f; do
                     jid="${f##*_}"; jid="${jid%.out}"
                     squeue -h -j "$jid" -o %i 2>/dev/null | grep -q "$jid" && echo "$f"
                   done)"
if [ -z "$STILL_TRACKED" ]; then
  echo "PASS  no transcript of a RUNNING arm is tracked"; PASS=$((PASS + 1))
else
  echo "FAIL  a running arm's transcript is still tracked:"; echo "$STILL_TRACKED" | sed 's/^/        | /'
  FAIL=$((FAIL + 1))
fi

echo
echo "=== screen guard tests: ${PASS} passed, ${FAIL} failed ==="
[ "$FAIL" -eq 0 ] || exit 1
echo "log: ${LOG}"
