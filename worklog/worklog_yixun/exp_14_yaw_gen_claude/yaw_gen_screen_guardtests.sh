#!/usr/bin/env bash
# ============================================================================
# yaw_gen_screen_guardtests.sh — guard-branch exercise for the exp_14 screen kit
# (yaw_gen_screen.sbatch, yaw_gen_screen_submit.sh, yaw_gen_submit_grid.sh).
#
# DRYRUN=1 runs every cheap gate (parameters, the three-cell contract, the
# arm->config/orbit mapping, commit binding, the exactly-one-checkpoint rule and
# the ckpt/arm identity gate) and then prints the eval argv instead of spending a
# GPU. Synthetic Lightning-shaped checkpoints are torch.save'd into mktemp output
# roots, so the real arms' outputs are never read or written and no job is
# submitted: sbatch/scontrol/scancel are mocked shims throughout.
#
# THREAT MODEL (Planner, 2026-08-11): this suite and the kit it exercises defend
# against ACCIDENTS and stray environment state — a seam left set, a case that
# forgets isolation, a rerun in a dirty shell. They are NOT hardened against an
# adversary with arbitrary control of the calling shell or of this file; the
# unshadowable preamble in both scripts is the cheap part of that boundary, not a
# claim to have crossed it. Deliberately obfuscated cases are out of scope.
#
# ISOLATION IS THE DEFAULT (review V3): YAW_GEN_MAIN_REPO is exported for the
# whole suite, so every invocation of the kit reads and writes under a TEMPORARY
# root unless it explicitly opts out. A case that forgets isolation therefore
# lands in the temp root, and one that tries to opt out in live mode is refused
# by the kit's own allowlist. The complete opt-out list is:
#
#   1. the lease-lifecycle / mocked-submission block — exercises the SHARED
#      measure-worktree store on purpose (its leases and lock are the subject);
#   2. the live-refusal probes — run the $GRID_FAKE/$SUB_FAKE copies, which are
#      already retargeted at a temp MAIN_REPO by sed, and must not carry the
#      test-only seam in live mode (they pass `env -u`);
#   3. the sbatch DRIVER cases (yaw_gen_screen.sbatch) — read-only DRYRUNs that
#      never submit and need the real configs/checkpoint fixtures.
#
# DO NOT RUN THIS SUITE DURING ACTIVE CAMPAIGN SUBMISSIONS. On an IDLE store it
# exercises deletion paths; while any campaign freeze is active those cases skip
# themselves and the freeze is never lifted. exp_14 keeps its own campaign-pin
# file, so unlike exp_11's suite this one never parks the store-wide pin marker —
# it parks only its own, and restores it via a trap including on interrupt.
#
# Usage: bash worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_screen_guardtests.sh
# Exit 0 = every case behaved as specified.
# ============================================================================
set -uo pipefail
cd "$(git -C "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" rev-parse --show-toplevel)" || exit 3

EXPDIR="worklog/worklog_yixun/exp_14_yaw_gen_claude"
EXP11="worklog/worklog_yixun/exp_11_fa_orbit_claude"   # READ-ONLY: configs, registry, store helper
SCREEN="${EXPDIR}/yaw_gen_screen.sbatch"
SUB="${EXPDIR}/yaw_gen_screen_submit.sh"
GRID="${EXPDIR}/yaw_gen_submit_grid.sh"
VALIDATOR="${EXPDIR}/exp14_validate_cell.py"
# THE SUITE NEVER TOUCHES CAMPAIGN STATE. It drives the kit through the test
# seams (honoured only against a mocked submission path), so the live
# yaw_gen_campaign_pin and yaw_gen_command.md are read-only to it — a RED run of
# an earlier version wrote both and submitted four real jobs (all cancelled).
LIVE_PIN_FILE="${EXPDIR}/yaw_gen_campaign_pin"
LIVE_CMDLOG="${EXPDIR}/yaw_gen_command.md"
LIVE_CMDLOG_SUM_AT_START="$(sha256sum "$LIVE_CMDLOG" 2>/dev/null | cut -d' ' -f1)"
# sha256-or-ABSENT, so mutation, deletion and create-then-delete all fail
LIVE_PIN_AT_START="ABSENT"
[ -f "$LIVE_PIN_FILE" ] && LIVE_PIN_AT_START="$(sha256sum "$LIVE_PIN_FILE" | cut -d' ' -f1)"
GUARD_SELF="$(readlink -f "${BASH_SOURCE[0]}")"
PY=/n/fs/gatrdp/envs/flac/bin/python
TS="$(date '+%Y-%m-%d_%H-%M-%S')"
LOG="${EXPDIR}/yaw_gen_${TS}_screen_guardtests.log"
HEAD_SHA="$(git rev-parse HEAD)"
MAIN_TREE="$(git rev-parse --show-toplevel)"

exec > >(tee -a "$LOG") 2>&1
echo "=== yaw_gen screen kit guard exercise — ${TS} — $(git rev-parse --short HEAD) ==="
for F in "$SCREEN" "$SUB" "$GRID" "$VALIDATOR"; do
  [ -f "$F" ] || { echo "missing ${F} - abort"; exit 3; }
done

TMP="$(mktemp -d)"
# Every LIVE-mode probe in this suite runs against COPIES of the kit whose
# MAIN_REPO is a temporary directory (review W2). A live-mode run against the
# real repository could, with a real campaign pin present and a cell missing,
# reach the submit branch — a guard suite must be incapable of that, not merely
# unlikely to do it. The invariant is asserted structurally below.
FAKE_REPO="${TMP}/fake_main_repo"
FAKE_EXP="${FAKE_REPO}/worklog/worklog_yixun/exp_14_yaw_gen_claude"
mkdir -p "$FAKE_EXP"
GRID_FAKE="${FAKE_EXP}/yaw_gen_submit_grid.sh"
SUB_FAKE="${FAKE_EXP}/yaw_gen_screen_submit.sh"
git -c init.defaultBranch=main init -q "$FAKE_REPO"
git -C "$FAKE_REPO" -c user.email=guard@local -c user.name=guard commit -q --allow-empty -m "isolation root"
git -C "$FAKE_REPO" -c user.email=guard@local -c user.name=guard commit -q --allow-empty -m "isolation head"
FAKE_PIN="$(git -C "$FAKE_REPO" rev-parse HEAD)"
sed "s|^MAIN_REPO=/n/fs/gatrdp/codespace/FLAC$|MAIN_REPO=${FAKE_REPO}|" "$GRID" > "$GRID_FAKE"
sed "s|^MAIN_REPO=/n/fs/gatrdp/codespace/FLAC$|MAIN_REPO=${FAKE_REPO}|" "$SUB" > "$SUB_FAKE"
cp "$VALIDATOR" "${EXPDIR}/exp14_ckpt_expect.json" "$FAKE_EXP/"
# ISOLATION BY DEFAULT: every case inherits a temporary MAIN_REPO (opt-outs are
# enumerated in the header and are the only places this is unset).
export YAW_GEN_MAIN_REPO="$FAKE_REPO"
# PUSH/POP around every opt-out (review U2): the first `unset` used to leak
# isolation-off through hundreds of unrelated lines, so a case inserted between
# blocks would have inherited it. iso_off/iso_on are always paired, and
# assert_isolated lets a section state its expectation out loud.
iso_off() {   # $1 = why this block legitimately needs the real MAIN_REPO
  ISO_SAVED="${YAW_GEN_MAIN_REPO:-}"
  unset YAW_GEN_MAIN_REPO
  echo "  (isolation OFF for: $1)"
}
iso_on() {
  export YAW_GEN_MAIN_REPO="${ISO_SAVED:-$FAKE_REPO}"
  ISO_SAVED=""
}
assert_isolated() {   # $1 = section name
  if [ -n "${YAW_GEN_MAIN_REPO:-}" ]; then
    echo "PASS  isolation armed entering: $1"; PASS=$((PASS + 1))
  else
    echo "FAIL  isolation is OFF entering: $1 — an opt-out leaked"; FAIL=$((FAIL + 1))
  fi
}
PIN_FILE="${TMP}/campaign_pin"          # the SEAM the kit is pointed at
TEST_CMDLOG="${TMP}/command.md"
LIVE_TRACE="${TMP}/live_submit.txt"     # where a simulated wave records its argv
LIVE_QUEUE="${TMP}/live_queue.txt"      # ...and the queue it is shown
TEST_INTENT_DIR="${TMP}/intents"; mkdir -p "$TEST_INTENT_DIR"
trap 'suite_exit_trap' EXIT

# Several cases below need a store they are allowed to DELETE from, and get it
# by thawing the campaign freeze. That is safe on an idle store and catastrophic
# on a live one: a queued job's worktree can be swept while this suite runs.
# (Learned the hard way — an earlier run of this suite left the store thawed
# while a campaign was in flight; the leases held, but that was luck, not
# design.) So: if a campaign freeze is active, those cases are SKIPPED and the
# freeze is never touched. Everything that does not delete still runs.
MEASURE_HELPER="${EXP11}/fa_orbit_measure_worktree.sh"
# The STORE-WIDE pin marker belongs to whichever campaign set it (exp_11's screens
# use it). This suite must leave it untouched; the value is captured here and
# compared again at the end, which is a behavioural check rather than a grep.
STORE_PIN_MARKER="$(git rev-parse --show-toplevel)/.measure_worktrees/.campaign_pin"
STORE_PIN_AT_START="$(head -1 "$STORE_PIN_MARKER" 2>/dev/null)"
CAMPAIGN_LIVE=0
bash "$MEASURE_HELPER" --frozen >/dev/null 2>&1 && CAMPAIGN_LIVE=1
[ "$CAMPAIGN_LIVE" = "1" ] && echo "NOTE: a CAMPAIGN FREEZE is active — deletion/thaw cases are skipped"

suite_exit_trap() {   # PRESERVES the script's own exit status; only worsens it
  local rc=$?
  restore_suite_state
  if ! assert_campaign_untouched; then rc=1; fi
  rm -rf "$TMP"
  exit "$rc"
}

assert_campaign_untouched() {   # runs at suite EXIT, after every case
  local now
  now="$(sha256sum "$LIVE_CMDLOG" 2>/dev/null | cut -d' ' -f1)"
  if [ "$now" != "${LIVE_CMDLOG_SUM_AT_START:-}" ]; then
    echo "FAIL  (suite exit) this suite MODIFIED the campaign command log"
    return 1
  fi
  local pin_now="ABSENT"
  [ -f "$LIVE_PIN_FILE" ] && pin_now="$(sha256sum "$LIVE_PIN_FILE" | cut -d' ' -f1)"
  if [ "$pin_now" != "${LIVE_PIN_AT_START:-ABSENT}" ]; then
    echo "FAIL  (suite exit) the campaign pin file changed (${LIVE_PIN_AT_START:0:12} -> ${pin_now:0:12})"
    return 1
  fi
  echo "PASS  (suite exit) campaign command log and pin file are exactly as found"
  return 0
}

restore_suite_state() {  # put back anything this suite parked (trap-backed)
  if [ "${SUITE_PIN_PARKED:-0}" = "1" ] && [ -f "${TMP}/campaign_pin.saved" ]; then
    cp "${TMP}/campaign_pin.saved" "$PIN_FILE" 2>/dev/null || true
  fi
  if [ "${SUITE_ENGAGED_FREEZE:-0}" = "1" ]; then
    bash "$MEASURE_HELPER" --thaw >/dev/null 2>&1 || true
  fi
}

skip_if_campaign() {   # $1 = description; returns 0 (skip) when a campaign is live
  if [ "$CAMPAIGN_LIVE" = "1" ]; then
    echo "SKIP  $1 — a campaign freeze is active; this suite will not thaw or delete"
    return 0
  fi
  return 1
}
PASS=0; FAIL=0

# --- synthetic output roots --------------------------------------------------
# exp_14 pins STEP=40000, so the negative checkpoint cases cannot share one root
# with the positive ones (a duplicate or EMA-less 40k file in the good root would
# break every good case for that arm). Each pathology therefore gets its OWN root,
# and a case selects it by overriding OUTPUT_ROOT/FA_ORBIT_ARM_REGISTRY after
# BASE — env applies assignments in order, so the last one wins.
#
#   GOOD     all five arms at 40000, each with its own config and EMA weights
#   NOEMA    C16 at 40000 with no diffusion_ema.* weights at all
#   WRONGARM C4L's tree holding a C8-config checkpoint; VANL's holding an ORBIT one
#   DUP      two epoch=*-step=40000.ckpt files for C8
#   EMPTY    no arm trees at all
GOOD="${TMP}/good"; NOEMA="${TMP}/noema"; WRONGARM="${TMP}/wrongarm"
DUP="${TMP}/dup"; EMPTY="${TMP}/empty"
mkdir -p "$EMPTY"
$PY - "$EXP11" "$GOOD" "$NOEMA" "$WRONGARM" "$DUP" <<'PY'
import hashlib, json, os, sys, torch
exp11, good, noema, wrongarm, dup = sys.argv[1:6]

CFG = {a: os.path.join(exp11, f"FLAC_AR_BF_{a}.json") for a in ("C4L", "C8", "C16", "C32")}
CFG["VANL"] = os.path.join(exp11, "FLAC_AR_VANCKPT.json")
JOBN = {"C4L": 1, "C8": 2, "C16": 3, "C32": 4, "VANL": 5}


def write_ckpt(root, arm, cfg_arm, step=40000, epoch=8, ema=True, tag=""):
    cfg = json.load(open(CFG[cfg_arm]))
    sd = {"diffusion.model.a": torch.zeros(1)}
    if ema:                                  # what eval_FLAC actually looks for
        sd["diffusion_ema.ema_model.model.a"] = torch.zeros(1)
    d = os.path.join(root, f"exp11_{arm}", f"FLAC_exp11_{arm}", f"exp11_{arm}", "checkpoints")
    os.makedirs(d, exist_ok=True)
    torch.save({"global_step": step, "epoch": epoch, "model_config": cfg, "state_dict": sd,
                "optimizer_states": [{"state": {0: {"step": 1}}, "param_groups": [{"lr": 1e-5}]}],
                "lr_schedulers": [{"last_epoch": step}]},
               os.path.join(d, f"epoch={epoch}-step={step}.ckpt"))


def manifest(root, arm, reg):
    """The arm's launch manifest, recorded in the root's own audited registry.

    The config hash is always the arm's OWN config — the one the driver will
    hash — so the D3 lineage gate passes and the LATER gates (EMA, embedded
    config identity) are the ones under test.
    """
    d = os.path.join(root, f"exp11_{arm}")
    os.makedirs(d, exist_ok=True)
    sha = hashlib.sha256(open(CFG[arm], "rb").read()).hexdigest()
    path = os.path.join(d, "launch_manifest.txt")
    with open(path, "w") as fh:
        fh.write(f"job 90000{JOBN[arm]} host synthetic mode INITIAL launch_uuid uuid-{arm}\n")
        fh.write(f"arm {arm} rung 8x8 micro 8 ngpu 8 max_steps 40000 ckpt_every 2500\n")
        fh.write("commit " + "0" * 40 + "\n")
        fh.write(f"p0_manifest_sha256 {'a' * 64}\n")
        fh.write(f"config_sha256 {sha}\n")
        fh.write(f"vae_sha256 {'b' * 64}\n")
        fh.write(f"save_dir {d}\n")
    reg["arms"][arm] = {
        "manifest_path": path,
        "manifest_sha256": hashlib.sha256(open(path, "rb").read()).hexdigest(),
        "job": f"90000{JOBN[arm]}", "mode": "INITIAL", "launch_uuid": f"uuid-{arm}",
        "commit": "0" * 40, "rung": "8x8", "micro": "8", "ngpu": "8",
        "max_steps": "40000", "config_sha256": sha, "vae_sha256": "b" * 64,
        "p0_manifest_sha256": "a" * 64, "save_dir": d, "training_seed": 42,
    }


def registry(root, arms):
    reg = {"arms": {}}
    for arm in arms:
        manifest(root, arm, reg)
    with open(os.path.join(root, "arm_launch_registry.json"), "w") as fh:
        json.dump(reg, fh, indent=2)


ALL = ("VANL", "C4L", "C8", "C16", "C32")
for arm in ALL:
    write_ckpt(good, arm, arm)
registry(good, ALL)

write_ckpt(noema, "C16", "C16", ema=False)
registry(noema, ("C16",))

write_ckpt(wrongarm, "C4L", "C8")          # C4L's tree, a C8-config checkpoint
write_ckpt(wrongarm, "VANL", "C4L")        # VANL's tree, an ORBIT checkpoint
registry(wrongarm, ("C4L", "VANL"))

write_ckpt(dup, "C8", "C8", epoch=8)
write_ckpt(dup, "C8", "C8", epoch=9)       # two files at the SAME step
registry(dup, ("C8",))
print("synthetic output roots written")
PY

# The NO-MANIFEST case needs an arm with a checkpoint and no launch manifest at
# all; strip C32's from a copy of the good root rather than from the good root.
NOMAN="${TMP}/noman"
cp -r "$GOOD" "$NOMAN" && rm -f "${NOMAN}/exp11_C32/launch_manifest.txt"

BASE=(DRYRUN=1 "EXPECT_SHA=${HEAD_SHA}" "OUTPUT_ROOT=${GOOD}" "FA_ORBIT_REPO_OVERRIDE=$PWD"
      "FA_ORBIT_ARM_REGISTRY=${GOOD}/arm_launch_registry.json")
root_env() {   # <root> — select another synthetic output root (last env wins)
  printf '%s\n' "OUTPUT_ROOT=$1" "FA_ORBIT_ARM_REGISTRY=$1/arm_launch_registry.json"
}

register_manifest() {  # <arm> [root] — record the manifest as it stands, faithfully
  local _root="${2:-$GOOD}"
  $PY - "$1" "${_root}/exp11_$1/launch_manifest.txt" "${_root}/arm_launch_registry.json" <<'PY'
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

# argv_of <env...> — the eval argv line one DRYRUN cell would run (or "").
argv_of() { env "$@" bash "$SCREEN" 2>&1 | sed -n 's/^python eval_FLAC.py //p' | head -1; }
argv_has() {  # <name> <argv> <needle>
  case "$2" in
    *"$3"*) echo "PASS  $1"; PASS=$((PASS + 1)) ;;
    *) echo "FAIL  $1: '$3' absent from"; echo "        | $2"; FAIL=$((FAIL + 1)) ;;
  esac
}
argv_lacks() {  # <name> <argv> <needle>  — an ABSENCE is a contract too
  case "$2" in
    *"$3"*) echo "FAIL  $1: '$3' present in"; echo "        | $2"; FAIL=$((FAIL + 1)) ;;
    *) echo "PASS  $1"; PASS=$((PASS + 1)) ;;
  esac
}

# --- BEST-EFFORT LINT over this suite's own source (review V3) --------------
# NOT an invariant: it is a substring scan, and a case written in another shape
# would slip past it. The SOUND mechanism is the exported YAW_GEN_MAIN_REPO
# above, which isolates every case by default. This lint is kept because it
# catches the obvious shape cheaply and names the offending line.
if $PY - "$GUARD_SELF" <<'PY'
import sys
src = open(sys.argv[1]).read().split("\n")
bad = []
for i, line in enumerate(src):
    if 'bash "$GRID"' in line or 'bash "$SUB"' in line:
        window = "\n".join(src[max(0, i - 6):i + 1])
        if "--verify-manifest" in line:
            continue          # read-only reader: no mode, no submission path
        if "if 'bash" in line:
            continue          # this checker's own source
        if "YAW_GEN_TEST_MODE=1" not in window and "DRYRUN=1" not in window:
            bad.append(f"{i + 1}: {line.strip()[:78]}")
if bad:
    print("live-capable invocations of the REAL kit:")
    print("\n".join("  " + b for b in bad))
    sys.exit(1)
sys.exit(0)
PY
then
  echo "PASS  lint: no guard case invokes the real kit in live mode (best-effort scan)"
  PASS=$((PASS + 1))
else
  echo "FAIL  lint: a guard case looks able to reach the real kit's live submit path"; FAIL=$((FAIL + 1))
fi
if grep -q 'MAIN_REPO=\${FAKE_REPO}' "$GUARD_SELF" \
   && grep -q 'sed "s|\^MAIN_REPO=' "${EXPDIR}/yaw_gen_redproof_r2fix4.sh"; then
  echo "PASS  the red-proof artifact is under the same temporary-MAIN_REPO rule"
  PASS=$((PASS + 1))
else
  echo "FAIL  the red proof can still run the kit against the real repository"; FAIL=$((FAIL + 1))
fi

# ...and the mechanism itself: with only the suite's default environment, an
# invocation of the REAL wave submitter reads the TEMP root, not the repository.
ISO="$(env YAW_GEN_TEST_MODE=1 bash "$GRID" WAVE=vctl 2>&1 | head -1)"
case "$ISO" in
  *"${FAKE_REPO}"*) echo "PASS  isolation is the DEFAULT: the kit reads the temp root, not the repo"
                    PASS=$((PASS + 1)) ;;
  *) echo "FAIL  a default-environment invocation did not land in the temp root: ${ISO}"
     FAIL=$((FAIL + 1)) ;;
esac

echo "--- A. parameters ---"
case_run "missing ARM"        2 "ARM must be exported" -- "${BASE[@]}" CELL=zref
case_run "missing STEP"       2 "STEP must be exported" -- "${BASE[@]}" ARM=C8 CELL=zref STEP=
case_run "missing CELL"       2 "CELL must be exported" -- "${BASE[@]}" ARM=C8 STEP=40000
case_run "missing EXPECT_SHA" 2 "EXPECT_SHA" \
  -- DRYRUN=1 ARM=C8 CELL=zref STEP=40000 "OUTPUT_ROOT=${GOOD}" "FA_ORBIT_REPO_OVERRIDE=$PWD"
case_run "unknown arm"        2 "not registered for exp_14" -- "${BASE[@]}" ARM=VAN CELL=zref STEP=40000
case_run "FA1 is not an exp_14 arm" 2 "not registered for exp_14" -- "${BASE[@]}" ARM=FA1 CELL=zref STEP=40000
# exp_11's comparator arm is NOT part of this campaign: a recycled command line
# must not produce an exp_14-labelled row.
case_run "C4BACKFILL is refused"  2 "not registered for exp_14" \
  -- "${BASE[@]}" ARM=C4BACKFILL CELL=zref STEP=40000
case_run "non-numeric STEP"   2 "STEP" -- "${BASE[@]}" ARM=C8 CELL=zref STEP=lots
case_run "bad K"              2 "must be 1 or 8" -- "${BASE[@]}" ARM=C8 CELL=zref STEP=40000 K=4
case_run "STEP 20000 is unregistered" 2 "STEP=40000 endpoint only" \
  -- "${BASE[@]}" ARM=C8 CELL=zref STEP=20000
case_run "STEP 42500 is unregistered" 2 "STEP=40000 endpoint only" \
  -- "${BASE[@]}" ARM=C8 CELL=rgen STEP=42500
# every exp_11 cell type is gone; naming one is an error, not a silent fallback
for OLDCELL in screen conf traj q9 r3 cross; do
  case_run "CELL=${OLDCELL} is not an exp_14 cell" 2 "must be rgen" \
    -- "${BASE[@]}" ARM=C8 "CELL=${OLDCELL}" STEP=40000
done
case_run "rgen refuses seed 47"  2 "eval seeds 42-46" -- "${BASE[@]}" ARM=C8 CELL=rgen STEP=40000 SEED=47
case_run "zref refuses seed 41"  2 "eval seeds 42-46" -- "${BASE[@]}" ARM=C8 CELL=zref STEP=40000 SEED=41
# rgen draws its own angles and zref is the theta=0 reference: neither takes one
case_run "rgen refuses ROTATE_DEG" 2 "takes no ROTATE_DEG" \
  -- "${BASE[@]}" ARM=C8 CELL=rgen STEP=40000 ROTATE_DEG=45
case_run "zref refuses ROTATE_DEG" 2 "takes no ROTATE_DEG" \
  -- "${BASE[@]}" ARM=C8 CELL=zref STEP=40000 ROTATE_DEG=0
case_run "an EVAL_ORBIT leftover is refused (rgen)" 2 "not an exp_14 parameter" \
  -- "${BASE[@]}" ARM=C8 CELL=rgen STEP=40000 EVAL_ORBIT=16
case_run "an EVAL_ORBIT leftover is refused (zref)" 2 "not an exp_14 parameter" \
  -- "${BASE[@]}" ARM=C8 CELL=zref STEP=40000 EVAL_ORBIT=8
case_run "vctl needs an angle"   2 "needs ROTATE_DEG" -- "${BASE[@]}" ARM=C4L CELL=vctl STEP=40000
case_run "vctl is seed 42 only"  2 "seed 42 by contract" \
  -- "${BASE[@]}" ARM=C4L CELL=vctl STEP=40000 SEED=43 ROTATE_DEG=90
case_run "vctl is K=8 only"      2 "K=8 by contract" \
  -- "${BASE[@]}" ARM=C4L CELL=vctl STEP=40000 K=1 ROTATE_DEG=90
case_run "vctl refuses an unregistered angle" 2 "not a registered vctl angle" \
  -- "${BASE[@]}" ARM=C4L CELL=vctl STEP=40000 ROTATE_DEG=22.5
# THE tuple list: five arms at 90, C4L at 45, nothing else. VANL@45 is the one a
# reader would expect to exist by symmetry and the plan deliberately does not.
case_run "VANL@45 is UNREGISTERED" 2 "not one of the six registered controls" \
  -- "${BASE[@]}" ARM=VANL CELL=vctl STEP=40000 ROTATE_DEG=45
case_run "C8@45 is UNREGISTERED"   2 "not one of the six registered controls" \
  -- "${BASE[@]}" ARM=C8 CELL=vctl STEP=40000 ROTATE_DEG=45
case_run "C16@45 is UNREGISTERED"  2 "not one of the six registered controls" \
  -- "${BASE[@]}" ARM=C16 CELL=vctl STEP=40000 ROTATE_DEG=45
case_run "C32@45 is UNREGISTERED"  2 "not one of the six registered controls" \
  -- "${BASE[@]}" ARM=C32 CELL=vctl STEP=40000 ROTATE_DEG=45

echo "--- B. checkpoint discovery ---"
case_run "no arm tree at all"     2 "exactly 1 checkpoint" \
  -- "${BASE[@]}" $(root_env "$EMPTY") ARM=C8 CELL=zref STEP=40000
case_run "two ckpts at one step"  2 "exactly 1 checkpoint" \
  -- "${BASE[@]}" $(root_env "$DUP") ARM=C8 CELL=zref STEP=40000

echo "--- C. the ckpt/arm identity gate ---"
case_run "C4L tree holding a C8 ckpt is rejected" 2 "CKPT/ARM GATE" \
  -- "${BASE[@]}" $(root_env "$WRONGARM") ARM=C4L CELL=zref STEP=40000
case_run "an ORBIT ckpt is refused as VANL" 2 "trained with an orbit and is not the vanilla arm" \
  -- "${BASE[@]}" $(root_env "$WRONGARM") ARM=VANL CELL=zref STEP=40000
case_run "a checkpoint without EMA weights is rejected" 2 "no diffusion_ema.ema_model" \
  -- "${BASE[@]}" $(root_env "$NOEMA") ARM=C16 CELL=zref STEP=40000
case_run "an arm ckpt with no launch manifest is refused" 2 "launch manifest missing" \
  -- "${BASE[@]}" $(root_env "$NOMAN") ARM=C32 CELL=zref STEP=40000
# ...and with a manifest whose config hash is another arm's, the lineage gate fires
write_c8_manifest() {  # $1 = which arm's config hash to record
  { echo "job 900002 host synthetic mode INITIAL launch_uuid uuid-C8"
    echo "arm C8 rung 8x8 micro 8 ngpu 8 max_steps 40000 ckpt_every 2500"
    echo "commit 0000000000000000000000000000000000000000"
    echo "p0_manifest_sha256 aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    echo "config_sha256 $($PY -c "import hashlib,sys;print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" "${EXP11}/FLAC_AR_BF_$1.json")"
    echo "vae_sha256 bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
    echo "save_dir ${GOOD}/exp11_C8"; } > "${GOOD}/exp11_C8/launch_manifest.txt"
  register_manifest C8            # audited AS WRITTEN: the field checks are the test
}
write_c8_manifest C4L
case_run "a launch manifest for another config is refused" 2 "ARM LINEAGE GATE" \
  -- "${BASE[@]}" ARM=C8 CELL=zref STEP=40000
write_c8_manifest C8              # a correct manifest lets the same cell through

echo "--- D. the three registered cells reach the eval argv ---"
# --- rgen: the headline cell -------------------------------------------------
RGEN_ARGV="$(argv_of "${BASE[@]}" ARM=C32 CELL=rgen STEP=40000 SEED=44 K=8)"
argv_has  "rgen passes --rotate-mode random with the eval seed" "$RGEN_ARGV" "--rotate-mode random --rotate-seed 44"
argv_lacks "rgen passes NO fixed angle"                         "$RGEN_ARGV" "--rotate-deg"
argv_has  "rgen names itself by its rotation seed"              "$RGEN_ARGV" "exp14_C32_rgen_S40000_s44_K8_rotrand44"
argv_has  "rgen pins the campaign batch/worker settings"        "$RGEN_ARGV" "--batch-size 64 --num-workers 4"
argv_has  "rgen pins the split size and records the stream"     "$RGEN_ARGV" "--expected-stream-count 6337 --record-stream"
argv_has  "rgen records the PER-SCENE block (plan §4 estimand)" "$RGEN_ARGV" "--record-per-scene"
argv_has  "rgen pins the protocol flags"                        "$RGEN_ARGV" "--cond-autocast bf16 --cfg-scale 1.0 --steps 1"
argv_has  "rgen keeps the arm's OWN orbit (C32)"                "$RGEN_ARGV" "--cond-method fa_invariant"
# --- zref: the theta=0 reference is the BYTE-DEFAULT fixed path --------------
ZREF_ARGV="$(argv_of "${BASE[@]}" ARM=C8 CELL=zref STEP=40000 SEED=42 K=8)"
argv_lacks "zref passes no rotation flag of any kind"           "$ZREF_ARGV" "--rotate"
argv_has  "zref names itself without a rotation token"          "$ZREF_ARGV" "exp14_C8_zref_S40000_s42_K8 "
argv_has  "zref still records its assignment stream"            "$ZREF_ARGV" "--expected-stream-count 6337 --record-stream"
argv_has  "zref records the PER-SCENE block too (Delta is paired)" "$ZREF_ARGV" "--record-per-scene"
argv_has  "zref keeps the arm's OWN orbit"                      "$ZREF_ARGV" "0.0,45.0,90.0,135.0,180.0,225.0,270.0,315.0"
# --- vctl: the fixed-angle validity controls --------------------------------
VCTL45="$(argv_of "${BASE[@]}" ARM=C4L CELL=vctl STEP=40000 ROTATE_DEG=45)"
argv_has  "vctl passes its fixed angle"                         "$VCTL45" "--rotate-deg 45"
argv_lacks "vctl passes no random-mode flag"                    "$VCTL45" "--rotate-mode"
argv_has  "vctl names itself by its angle"                      "$VCTL45" "exp14_C4L_vctl_S40000_s42_K8_rot45"
VCTL90="$(argv_of "${BASE[@]}" ARM=VANL CELL=vctl STEP=40000 ROTATE_DEG=90)"
argv_has  "the VANL positive control is registered at 90"       "$VCTL90" "exp14_VANL_vctl_S40000_s42_K8_rot90"
argv_has  "VANL runs a VANILLA evaluation"                      "$VCTL90" "--cond-method vanilla"
argv_lacks "a vanilla evaluation is given no orbit at all"      "$VCTL90" "--frame-avg-angles"
VCTL90F="$(argv_of "${BASE[@]}" ARM=C16 CELL=vctl STEP=40000 ROTATE_DEG=90.0)"
argv_has  "90.0 canonicalises to the 90 token"                  "$VCTL90F" "exp14_C16_vctl_S40000_s42_K8_rot90"
argv_has  "...and reaches the eval as 90"                       "$VCTL90F" "--rotate-deg 90"
# --- splits ------------------------------------------------------------------
K1_ARGV="$(argv_of "${BASE[@]}" ARM=C8 CELL=rgen STEP=40000 K=1 SEED=46)"
argv_has "K=1 uses the _1 split"   "$K1_ARGV"   "acousticroom_unseeneval_1.json"
argv_has "K=8 uses the full split" "$ZREF_ARGV" "acousticroom_unseeneval.json"
argv_has "K=1 still names its K"   "$K1_ARGV"   "exp14_C8_rgen_S40000_s46_K1_rotrand46"

# --- the CELL-VALIDATION argv the driver renders is the validator's rule -----
# Review B1 was exactly this rendering: a --rotate-deg 0 that the validator then
# refused, which would have failed 100 of 106 cells AFTER their GPU time. The
# driver's line must equal exp14_validate_cell's own, flag for flag.
VAL_OK=1
for SPEC in "C32 rgen 44 8 -" "C8 zref 42 8 -" "C4L vctl 42 8 45" "VANL vctl 42 8 90"; do
  # shellcheck disable=SC2086
  set -- $SPEC
  V_ENV=("${BASE[@]}" "ARM=$1" "CELL=$2" STEP=40000 "SEED=$3" "K=$4")
  [ "$5" = "-" ] || V_ENV+=("ROTATE_DEG=$5")
  GOT="$(env "${V_ENV[@]}" bash "$SCREEN" 2>&1 \
         | sed -n 's/^python3 exp14_validate_cell.py //p' | head -1)"
  # The driver binds to the CODE ROOT's HEAD, which a concurrent session in this
  # shared checkout can move mid-suite; the parity being tested is the FLAG SET,
  # so the expectation is built with whatever pin the driver just reported.
  DRIVER_PIN="$(printf '%s\n' "$GOT" | sed -n 's/.*--pin \([0-9a-f]\{40\}\).*/\1/p')"
  A_ARGS=(argv --metrics "<metrics>" --arm "$1" --cell "$2" --step 40000 --seed "$3" --k "$4")
  [ "$5" = "-" ] || A_ARGS+=(--rotate-deg "$5")
  A_ARGS+=(--pin "${DRIVER_PIN:-$HEAD_SHA}" --ckpt-sha "<ckpt-sha256>" --expected-count 6337
           --expected-scenes 17)
  WANT="$($PY "$VALIDATOR" "${A_ARGS[@]}")"
  if [ "$GOT" != "$WANT" ]; then
    echo "FAIL  validation argv mismatch for ${SPEC}"
    echo "        | driver:    ${GOT}"
    echo "        | validator: ${WANT}"
    VAL_OK=0
  fi
  case "$2" in
    vctl) case "$GOT" in *--rotate-deg*) ;; *) echo "FAIL  vctl validation argv lost its angle"; VAL_OK=0 ;; esac ;;
    *)    case "$GOT" in *--rotate-deg*) echo "FAIL  ${2} passes --rotate-deg to the validator (review B1)"; VAL_OK=0 ;; esac ;;
  esac
done
if [ "$VAL_OK" = "1" ]; then
  echo "PASS  the driver's cell-validation argv equals exp14_validate_cell.check_argv"
  echo "PASS  only vctl passes --rotate-deg to the validator (review B1)"
  PASS=$((PASS + 2))
else
  FAIL=$((FAIL + 1))
fi

# --- the eval NAME the driver renders is the validator's rule ----------------
# The driver renders the rotation token in shell (importing eval_FLAC would cost
# ~9 s of torch per DRYRUN cell); this is what stops the two rules from drifting.
NAME_OK=1
for SPEC in "C32 rgen 44 8 -:${RGEN_ARGV}" "C8 zref 42 8 -:${ZREF_ARGV}" \
            "C4L vctl 42 8 45:${VCTL45}" "VANL vctl 42 8 90:${VCTL90}" \
            "C8 rgen 46 1 -:${K1_ARGV}"; do
  SPEC_CELL="${SPEC%%:*}"; SPEC_ARGV="${SPEC#*:}"
  # shellcheck disable=SC2086
  set -- $SPEC_CELL
  WANT="$($PY - "$VALIDATOR" "$1" "$2" "$3" "$4" "$5" <<'PY'
import importlib.util, sys
spec = importlib.util.spec_from_file_location("v", sys.argv[1])
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
arm, cell, seed, k, rot = sys.argv[2:7]
deg = None if rot == "-" else float(rot)
print(m.eval_name(m.Cell(arm, cell, 40000, int(seed), int(k), deg)))
PY
)"
  case "$SPEC_ARGV" in
    *"--eval-name ${WANT}"*) ;;
    *) echo "FAIL  driver eval-name != validator's for ${SPEC_CELL} (want ${WANT})"; NAME_OK=0 ;;
  esac
done
if [ "$NAME_OK" = "1" ]; then
  echo "PASS  the driver's shell-rendered eval names equal exp14_validate_cell.eval_name"
  PASS=$((PASS + 1))
else
  FAIL=$((FAIL + 1))
fi

echo "--- E. real-mode gates ---"
# (a real screen now also needs MEASURE_ROOT; the SHA gate is exercised with one
# in section G, so here we only pin the sbatch-only requirement)
case_run "real mode needs sbatch"  2 "must run under sbatch" \
  -- ARM=C8 CELL=zref STEP=40000 "EXPECT_SHA=${HEAD_SHA}" "FA_ORBIT_REPO_OVERRIDE=$PWD"

echo "--- F. the emitted eval names parse under the validator's schema ---"
# Every name the driver can emit must round-trip through the validator that the
# collector will read, and a name from another campaign (or an unregistered cell)
# must NOT parse: an artifact nobody registered may not be read as evidence.
for NAME in "exp14_C32_rgen_S40000_s44_K8_rotrand44" "exp14_C8_zref_S40000_s42_K8" \
            "exp14_C4L_vctl_S40000_s42_K8_rot45" "exp14_VANL_vctl_S40000_s42_K8_rot90"; do
  if $PY -c "
import importlib.util,sys
s=importlib.util.spec_from_file_location('v','${VALIDATOR}')
m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
print(m.parse_eval_name('${NAME}'))" >/dev/null 2>&1; then
    echo "PASS  ${NAME} parses under the validator schema"; PASS=$((PASS + 1))
  else
    echo "FAIL  ${NAME} does not parse under the validator schema"; FAIL=$((FAIL + 1))
  fi
done
for NAME in "exp11_C8_screen_S10000_s42_K8" "exp14_VANL_vctl_S40000_s42_K8_rot45" \
            "exp14_C8_rgen_S40000_s47_K8_rotrand47" "exp14_C8_conf_S40000_s42_K8"; do
  if $PY -c "
import importlib.util,sys
s=importlib.util.spec_from_file_location('v','${VALIDATOR}')
m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
m.parse_eval_name('${NAME}')" >/dev/null 2>&1; then
    echo "FAIL  ${NAME} parsed as a registered exp_14 cell"; FAIL=$((FAIL + 1))
  else
    echo "PASS  ${NAME} is refused as unregistered"; PASS=$((PASS + 1))
  fi
done

echo "--- G. worktree-pinned measurement execution (item 8) ---"
# A real screen must refuse to run unpinned...
case_run "real mode requires MEASURE_ROOT" 2 "MEASURE_ROOT is required" \
  -- ARM=C8 CELL=zref STEP=40000 "EXPECT_SHA=${HEAD_SHA}" SLURM_JOB_ID=999999
case_run "a non-existent MEASURE_ROOT is refused" 2 "not a directory" \
  -- "${BASE[@]}" ARM=C8 CELL=zref STEP=40000 "MEASURE_ROOT=${TMP}/nope"
mkdir -p "${TMP}/notaworktree"
case_run "a MEASURE_ROOT that is not a worktree is refused" 2 "not a git worktree" \
  -- "${BASE[@]}" ARM=C8 CELL=zref STEP=40000 "MEASURE_ROOT=${TMP}/notaworktree"

# ...and with a real pinned worktree the binding is on the WORKTREE's HEAD, so a
# divergent main tree is irrelevant. Simulate the divergence by binding to the
# worktree SHA while the main tree sits at a different commit.
WT="$(bash "${MEASURE_HELPER}" 2>/dev/null | tail -1)"
if [ -n "$WT" ] && [ -e "$WT/.git" ]; then
  WT_SHA="$(git -C "$WT" rev-parse HEAD)"
  echo "PASS  pinned worktree prepared at ${WT_SHA:0:12}"; PASS=$((PASS + 1))
  # HEAD mismatch must abort even with a valid worktree. This case is about the
  # COMMIT gate, so give the simulated job a genuine lease first — otherwise the
  # lease gate (which runs earlier, by design) is what we would be testing.
  bash "${MEASURE_HELPER}" --lease 999999 "$WT" >/dev/null 2>&1
  out="$(env ARM=C8 CELL=zref STEP=40000 EXPECT_SHA=0000000000000000000000000000000000000000 \
          SLURM_JOB_ID=999999 "MEASURE_ROOT=$WT" bash "$SCREEN" 2>&1)"; rc=$?
  if [ "$rc" -eq 2 ] && echo "$out" | grep -q "code-root HEAD"; then
    echo "PASS  worktree HEAD mismatch aborts  (rc=${rc})"; PASS=$((PASS + 1))
  else
    echo "FAIL  worktree HEAD mismatch: rc=${rc}"; echo "$out" | tail -3 | sed 's/^/        | /'
    FAIL=$((FAIL + 1))
  fi
  # the code root's identity is the worktree's, NOT the main tree's
  # ...and the REAL campaign cell: code from the pinned worktree, checkpoint from
  # the main tree. That split is the whole point of pinned execution, and this is
  # the ONLY case that touches a real arm checkpoint — the identity gate loads it
  # twice at ~724 MB, so on a shared login node it is opt-in (GUARD_REAL_CKPT=1)
  # exactly like exp_11's GUARD_REAL_BACKFILL case.
  if [ "${GUARD_REAL_CKPT:-0}" = "1" ]; then
    out="$(env DRYRUN=1 ARM=C4L CELL=zref STEP=40000 "EXPECT_SHA=${WT_SHA}" \
            "MEASURE_ROOT=$WT" bash "$SCREEN" 2>&1)"; rc=$?
    if [ "$rc" -eq 0 ] && echo "$out" | grep -q "config ${WT}/worklog" \
       && echo "$out" | grep -q "checkpoint: ${MAIN_TREE}/outputs_FLAC"; then
      echo "PASS  code from the pinned worktree, outputs from the main tree  (rc=${rc})"
      PASS=$((PASS + 1))
    else
      echo "FAIL  pinned-run case failed against the real C4L 40k ckpt (rc=${rc})"
      echo "$out" | tail -3 | sed 's/^/        | /'; FAIL=$((FAIL + 1))
    fi
    # the registry that run re-verified is exp_11's, read IN PLACE
    if echo "$out" | grep -q "arm lineage OK: C4L bound to AUDITED launch job 3648694"; then
      echo "PASS  the pinned cell re-verified exp_11's audited launch registry"; PASS=$((PASS + 1))
    else
      echo "FAIL  the pinned cell did not bind to the audited registry"; FAIL=$((FAIL + 1))
    fi
  else
    echo "SKIP  the real-checkpoint pinned-run case (GUARD_REAL_CKPT=1 to load the 724 MB ckpt)"
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
HELPER="${MEASURE_HELPER}"
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
  out="$(env DRYRUN=1 ARM=C8 CELL=zref STEP=40000 "EXPECT_SHA=${HEAD_SHA}" "OUTPUT_ROOT=$GOOD" \
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
  # Judged at whatever HEAD is now: a concurrent session in this shared checkout
  # commits while the suite runs, so the tree the helper hands out may legitimately
  # differ from the one captured at suite start. What must hold is IDEMPOTENCE —
  # repeated calls return one and the same tree, and it exists exactly once.
  SAME="$(bash "$HELPER" 2>/dev/null | tail -1)"
  SAME2="$(bash "$HELPER" 2>/dev/null | tail -1)"
  N="$(ls -1d "$SAME" 2>/dev/null | wc -l)"
  if [ "$N" -eq 1 ] && [ -n "$SAME" ] && [ "$SAME" = "$SAME2" ]; then
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
  out="$(env ARM=C8 CELL=zref STEP=40000 "EXPECT_SHA=${WT_SHA}" "MEASURE_ROOT=$WT" SLURM_JOB_ID=424242 \
         bash "$SCREEN" 2>&1)"; rc=$?
  if [ "$rc" -ne 0 ] && echo "$out" | grep -q "no lease"; then
    echo "PASS  a job without its own lease refuses to run"; PASS=$((PASS + 1))
  else
    echo "FAIL  a job ran in a tree nobody leased for it (rc=${rc})"; FAIL=$((FAIL + 1))
  fi
  # (f) a lease naming ANOTHER job is not this job's lease
  mkdir -p "$WT/.leases"; printf 'jobid 111\n' > "$WT/.leases/424242"
  out="$(env ARM=C8 CELL=zref STEP=40000 "EXPECT_SHA=${WT_SHA}" "MEASURE_ROOT=$WT" SLURM_JOB_ID=424242 \
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
  FREEZE_MARKER="${MAIN_TREE}/.measure_worktrees/.campaign_freeze"
  # A submission refuses to run without the campaign freeze. The freeze is
  # STORE-WIDE and protects every campaign at once, so this suite engages it only
  # if nobody else has, and restores whatever it found — a suite that thawed a
  # neighbour's freeze would expose their queued jobs' worktrees to a sweep.
  SUITE_ENGAGED_FREEZE=0
  if ! bash "$HELPER" --frozen >/dev/null 2>&1; then
    bash "$HELPER" --freeze "yaw_gen guard suite" >/dev/null 2>&1
    SUITE_ENGAGED_FREEZE=1
  fi
  # exp_14's campaign pin is its OWN file, so parking it cannot disturb any other
  # campaign (exp_11's store-wide .campaign_pin marker is never touched by this
  # suite). It is parked because a live pin would send every mocked submission
  # below to the pinned commit instead of HEAD; the pin's own cases set and
  # restore it themselves.
  #
  # TRAP-BACKED: restoring only on the normal path means a Ctrl-C, a timeout or
  # any early exit leaves the campaign unpinned — the next submission would then
  # silently measure at HEAD instead of the pin. Arm the restore BEFORE clearing.
  SUITE_PIN_PARKED=0
  trap 'suite_exit_trap' EXIT
  trap 'restore_suite_state; exit 130' INT TERM
  rm -f "$PIN_FILE"
  # NO MOCK BINARIES. "Is this executable a mock?" is not a decidable question —
  # a wrapper, a copy, a hard link or an absolute path to the real sbatch all
  # differ from the string "sbatch" — so the kit no longer asks it: in TEST MODE
  # it runs no submit command at all and records the argv internally. The suite
  # therefore drives it with YAW_GEN_TEST_MODE=1 + YAW_GEN_TEST_RECORD and reads
  # the recorded argv, which is strictly more evidence than a mock produced.
  MOCK="${TMP}/mockbin"; mkdir -p "$MOCK"
  # PIN THE MOCKED SUBMISSIONS TO *THIS* WORKTREE. Without a pin they follow HEAD,
  # and HEAD moves: the concurrent session in this shared checkout commits while
  # the suite runs, so a submission would prepare and lease a DIFFERENT tree than
  # the one these cases inspect ($WT) and every lease assertion would fail for a
  # reason that has nothing to do with the kit.
  printf '%s\n' "$WT_SHA" > "$PIN_FILE"
  # Exported ONCE for the whole block, so no invocation can accidentally fall back
  # to the campaign's own pin file (and thus to a moving HEAD) or drop an intent
  # manifest into the campaign folder. Both are honoured by the kit only because
  # every invocation below runs in TEST MODE, which starts no submit command.
  export YAW_GEN_PIN_FILE="$PIN_FILE" YAW_GEN_INTENT_DIR="$TEST_INTENT_DIR"
  # OPT-OUT 1 (see header): these cases exercise the SHARED measure-worktree
  # store deliberately — its lock, its leases and this suite's real worktree are
  # the subject — so they run against the real MAIN_REPO. They are still test
  # mode, so they start no submit process. POPPED at the end of this block.
  iso_off "the shared-store lease/lock cases"
  TRACE="${TMP}/trace.txt"; : > "$TRACE"
  out="$(env YAW_GEN_TEST_MODE=1 YAW_GEN_TEST_RECORD="$TRACE" \
             YAW_GEN_TEST_MODE=1 YAW_GEN_TEST_RECORD="$TRACE" bash "$SUB" ARM=C4L CELL=zref STEP=40000 2>&1)"; rc=$?
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
  out="$(env YAW_GEN_TEST_RELEASE_FAILS=1 YAW_GEN_TEST_MODE=1 YAW_GEN_TEST_RECORD="$TRACE" \
             YAW_GEN_TEST_MODE=1 YAW_GEN_TEST_RECORD="$TRACE" bash "$SUB" ARM=C4L CELL=zref STEP=40000 2>&1)"; rc=$?
  if [ "$rc" -eq 6 ] && grep -q "scancel 7654321" "$TRACE" && [ -f "$WT/.leases/7654321" ]; then
    echo "PASS  a failed release cancels the job and RETAINS the lease"; PASS=$((PASS + 1))
  else
    echo "FAIL  a failed release mishandled the lease (rc=${rc}, lease $([ -f "$WT/.leases/7654321" ] && echo kept || echo DROPPED))"
    sed 's/^/        | /' "$TRACE"; FAIL=$((FAIL + 1))
  fi
  # ...and when scancel ALSO fails, the outcome is maximally uncertain: still keep it
  : > "$TRACE"
  out="$(env YAW_GEN_TEST_RELEASE_FAILS=1 YAW_GEN_TEST_SCANCEL_FAILS=1 \
             \
             YAW_GEN_TEST_MODE=1 YAW_GEN_TEST_RECORD="$TRACE" bash "$SUB" ARM=C4L CELL=zref STEP=40000 2>&1)"; rc=$?
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
  out="$(env YAW_GEN_TEST_MODE=1 YAW_GEN_TEST_RECORD="$TRACE" \
             YAW_GEN_TEST_MODE=1 YAW_GEN_TEST_RECORD="$TRACE" bash "$SUB" ARM=C4L CELL=zref STEP=40000 2>&1)"; rc=$?
  VAL_LINE="$(grep -n 'jobid \${JOBID}\$' "$SUB" | head -1 | cut -d: -f1)"
  REL_LINE="$(grep -n 'if ! slurm_release' "$SUB" | head -1 | cut -d: -f1)"
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
  out="$(env YAW_GEN_TEST_MODE=1 YAW_GEN_TEST_RECORD="$TRACE" \
             YAW_GEN_TEST_MODE=1 YAW_GEN_TEST_RECORD="$TRACE" bash "$SUB" ARM=C4L CELL=zref STEP=40000 EXCLUDE=neu303,neu332 2>&1)"; rc=$?
  if [ "$rc" -eq 0 ] && grep -q -- "--exclude=neu303,neu332" "$TRACE"; then
    echo "PASS  EXCLUDE= is passed to sbatch as an explicit --exclude flag"; PASS=$((PASS + 1))
  else
    echo "FAIL  EXCLUDE= did not reach sbatch (rc=${rc})"; sed 's/^/        | /' "$TRACE"
    FAIL=$((FAIL + 1))
  fi
  bash "$HELPER" --release 7654321 "$WT" >/dev/null 2>&1
  # no EXCLUDE given: no stray flag
  : > "$TRACE"
  env   \
             YAW_GEN_PIN_FILE="$PIN_FILE" YAW_GEN_INTENT_DIR="$TEST_INTENT_DIR" \
      YAW_GEN_TEST_MODE=1 YAW_GEN_TEST_RECORD="$TRACE" bash "$SUB" ARM=C4L CELL=zref STEP=40000 >/dev/null 2>&1
  if ! grep -q -- "--exclude" "$TRACE"; then
    echo "PASS  no EXCLUDE= means no --exclude flag"; PASS=$((PASS + 1))
  else
    echo "FAIL  an --exclude flag appeared without EXCLUDE="; FAIL=$((FAIL + 1))
  fi
  bash "$HELPER" --release 7654321 "$WT" >/dev/null 2>&1
  # the env var that never worked is now refused loudly instead of ignored
  out="$(env SBATCH_EXCLUDE=neu303 \
             \
             YAW_GEN_TEST_MODE=1 YAW_GEN_TEST_RECORD="$TRACE" bash "$SUB" ARM=C4L CELL=zref STEP=40000 2>&1)"; rc=$?
  if [ "$rc" -ne 0 ] && echo "$out" | grep -q "sbatch does not honour it"; then
    echo "PASS  a set SBATCH_EXCLUDE is refused, not silently ignored"; PASS=$((PASS + 1))
  else
    echo "FAIL  SBATCH_EXCLUDE was accepted as if it worked (rc=${rc})"; FAIL=$((FAIL + 1))
  fi
  # cell parameters travel with the job
  : > "$TRACE"
  env   \
             YAW_GEN_PIN_FILE="$PIN_FILE" YAW_GEN_INTENT_DIR="$TEST_INTENT_DIR" \
      YAW_GEN_TEST_MODE=1 YAW_GEN_TEST_RECORD="$TRACE" bash "$SUB" ARM=C4L CELL=vctl STEP=40000 ROTATE_DEG=45 >/dev/null 2>&1
  if grep -q "ROTATE_DEG=45" "$TRACE" && grep -q "CELL=vctl" "$TRACE"; then
    echo "PASS  the submitter exports the cell's own parameters to the job"; PASS=$((PASS + 1))
  else
    echo "FAIL  cell parameters do not reach the job"; sed 's/^/        | /' "$TRACE"; FAIL=$((FAIL + 1))
  fi
  bash "$HELPER" --release 7654321 "$WT" >/dev/null 2>&1
  # the mocked submission wrote a real intent manifest; it names a job that never
  # existed, so leaving it behind would put a fictional launch in the record
  rm -f ${TEST_INTENT_DIR}/yaw_gen_submission_C4L_vctl_*_jid7654321.txt

  # --- argument values are DATA, never shell ---------------------------------
  # The old parser eval'd the value, so a quote in it executed what followed.
  CANARY="${TMP}/injected.canary"; rm -f "$CANARY"
  out="$(env YAW_GEN_TEST_MODE=1 YAW_GEN_TEST_RECORD="$TRACE" \
             YAW_GEN_TEST_MODE=1 YAW_GEN_TEST_RECORD="$TRACE" bash "$SUB" ARM=C4L CELL=zref STEP=40000 "PIN_SHA=x'; INJECTED=1; #'" 2>&1)"; rc=$?
  if [ "$rc" -ne 0 ] && echo "$out" | grep -q "not 40 hex characters"; then
    echo "PASS  the literal injection value is refused by shape"; PASS=$((PASS + 1))
  else
    echo "FAIL  the injection value was not refused (rc=${rc})"; echo "$out" | tail -3 | sed 's/^/        | /'
    FAIL=$((FAIL + 1))
  fi
  # ...and a payload that WOULD run under eval leaves no trace
  out="$(env YAW_GEN_TEST_MODE=1 YAW_GEN_TEST_RECORD="$TRACE" \
             YAW_GEN_TEST_MODE=1 YAW_GEN_TEST_RECORD="$TRACE" bash "$SUB" ARM=C4L CELL=zref STEP=40000 "PIN_SHA=x'; touch ${CANARY}; #'" 2>&1)"; rc=$?
  if [ "$rc" -ne 0 ] && [ ! -e "$CANARY" ]; then
    echo "PASS  an injected payload does not execute (no side effects)"; PASS=$((PASS + 1))
  else
    echo "FAIL  an injected payload EXECUTED (canary $([ -e "$CANARY" ] && echo created || echo absent), rc=${rc})"
    FAIL=$((FAIL + 1)); rm -f "$CANARY"
  fi
  # every key is shape-checked, not just PIN_SHA
  for bad in "ARM=C4L;rm" "CELL=../etc" "CELL=screen" "STEP=1e4" "K=3" "EVAL_ORBIT=8" \
             "EXCLUDE=neu1;id" "ROTATE_DEG=1.2.3"; do
    env     \
             YAW_GEN_PIN_FILE="$PIN_FILE" YAW_GEN_INTENT_DIR="$TEST_INTENT_DIR" \
        YAW_GEN_TEST_MODE=1 YAW_GEN_TEST_RECORD="$TRACE" bash "$SUB" ARM=C4L CELL=zref STEP=40000 "$bad" >/dev/null 2>&1 && { BADOK="$bad"; break; }
  done
  if [ -z "${BADOK:-}" ]; then
    echo "PASS  malformed values are refused for every key"; PASS=$((PASS + 1))
  else
    echo "FAIL  '${BADOK}' was accepted"; FAIL=$((FAIL + 1))
  fi
  # code only: the explanatory comment above the parser names the old eval
  # `eval` in COMMAND POSITION only: an indented eval must still be caught (U3),
  # while the preamble's `unset -f ... readonly eval declare` — where the word is
  # an ARGUMENT — must not be. Matching the position, not the indentation, does
  # both; the old `grep -v "^ "` discarded every indented line.
  if grep -vE '^[[:space:]]*#' "$SUB" | grep -qE '(^|[;&|]|\bthen|\bdo|\{)[[:space:]]*eval[[:space:]]'; then
    echo "FAIL  the submitter still evals an argument"; FAIL=$((FAIL + 1))
  else
    echo "PASS  no eval of argument text remains in the code"; PASS=$((PASS + 1))
  fi

  # --- CAMPAIGN PIN: one commit for the whole campaign -----------------------
  # exp_14 reads its own ${EXPDIR}/yaw_gen_campaign_pin, so these cases never
  # touch the store-wide marker another campaign may be using right now.
  PIN2="$(git rev-parse HEAD~1 2>/dev/null)"
  if [ -n "$PIN2" ]; then
    printf '%s\n' "$PIN2" > "$PIN_FILE"
    : > "$TRACE"
    out="$(env YAW_GEN_TEST_MODE=1 YAW_GEN_TEST_RECORD="$TRACE" \
               YAW_GEN_TEST_MODE=1 YAW_GEN_TEST_RECORD="$TRACE" bash "$SUB" ARM=C4L CELL=zref STEP=40000 2>&1)"; rc=$?
    if [ "$rc" -eq 0 ] && grep -q "EXPECT_SHA=${PIN2}" "$TRACE" \
       && echo "$out" | grep -q "campaign pin (from"; then
      echo "PASS  the campaign pin file is the DEFAULT pin"; PASS=$((PASS + 1))
    else
      echo "FAIL  the campaign pin was not used by default (rc=${rc})"; FAIL=$((FAIL + 1))
    fi
    bash "$HELPER" --release 7654321 "${MAIN_TREE}/.measure_worktrees/${PIN2}" >/dev/null 2>&1
    rm -f ${TEST_INTENT_DIR}/yaw_gen_submission_C4L_zref_*_jid7654321.txt
    # an explicit PIN_SHA that disagrees is refused
    OTHER="$(git rev-parse HEAD 2>/dev/null)"
    out="$(env YAW_GEN_TEST_MODE=1 YAW_GEN_TEST_RECORD="$TRACE" \
               YAW_GEN_TEST_MODE=1 YAW_GEN_TEST_RECORD="$TRACE" bash "$SUB" ARM=C4L CELL=zref STEP=40000 "PIN_SHA=${OTHER}" 2>&1)"; rc=$?
    if [ "$rc" -ne 0 ] && echo "$out" | grep -q "disagrees with the campaign pin"; then
      echo "PASS  a PIN_SHA disagreeing with the campaign pin is refused"; PASS=$((PASS + 1))
    else
      echo "FAIL  a disagreeing PIN_SHA was accepted (rc=${rc})"; FAIL=$((FAIL + 1))
    fi
    # ...and agreeing with it is fine
    out="$(env YAW_GEN_TEST_MODE=1 YAW_GEN_TEST_RECORD="$TRACE" \
               YAW_GEN_TEST_MODE=1 YAW_GEN_TEST_RECORD="$TRACE" bash "$SUB" ARM=C4L CELL=zref STEP=40000 "PIN_SHA=${PIN2}" 2>&1)"; rc=$?
    [ "$rc" -eq 0 ] \
      && { echo "PASS  an explicit PIN_SHA equal to the campaign pin is accepted"; PASS=$((PASS + 1)); } \
      || { echo "FAIL  the agreeing PIN_SHA was refused (rc=${rc})"; FAIL=$((FAIL + 1)); }
    bash "$HELPER" --release 7654321 "${MAIN_TREE}/.measure_worktrees/${PIN2}" >/dev/null 2>&1
    rm -f ${TEST_INTENT_DIR}/yaw_gen_submission_C4L_zref_*_jid7654321.txt
    # a malformed pin FILE is refused rather than silently ignored
    printf 'not-a-sha\n' > "$PIN_FILE"
    out="$(env YAW_GEN_TEST_MODE=1 YAW_GEN_TEST_RECORD="$TRACE" \
               YAW_GEN_TEST_MODE=1 YAW_GEN_TEST_RECORD="$TRACE" bash "$SUB" ARM=C4L CELL=zref STEP=40000 2>&1)"; rc=$?
    if [ "$rc" -ne 0 ] && echo "$out" | grep -q "does not hold a 40-hex commit sha"; then
      echo "PASS  a malformed campaign pin file is refused"; PASS=$((PASS + 1))
    else
      echo "FAIL  a malformed campaign pin file was accepted (rc=${rc})"; FAIL=$((FAIL + 1))
    fi
    # no pin file -> the old HEAD behaviour returns
    rm -f "$PIN_FILE"
    : > "$TRACE"
    env     \
             YAW_GEN_PIN_FILE="$PIN_FILE" YAW_GEN_INTENT_DIR="$TEST_INTENT_DIR" \
        YAW_GEN_TEST_MODE=1 YAW_GEN_TEST_RECORD="$TRACE" bash "$SUB" ARM=C4L CELL=zref STEP=40000 >/dev/null 2>&1
    UNPINNED_SHA="$(grep -o 'EXPECT_SHA=[0-9a-f]\{40\}' "$TRACE" | head -1 | cut -d= -f2)"
    if [ -n "$UNPINNED_SHA" ] && git merge-base --is-ancestor "$UNPINNED_SHA" HEAD 2>/dev/null; then
      echo "PASS  with no pin file, submission falls back to HEAD"; PASS=$((PASS + 1))
    else
      echo "FAIL  the unpinned fallback is not HEAD ('${UNPINNED_SHA}')"; FAIL=$((FAIL + 1))
    fi
    bash "$HELPER" --release 7654321 "${MAIN_TREE}/.measure_worktrees/${UNPINNED_SHA}" >/dev/null 2>&1
    printf '%s\n' "$WT_SHA" > "$PIN_FILE"     # back to the deterministic tree
    rm -f ${TEST_INTENT_DIR}/yaw_gen_submission_C4L_zref_*_jid7654321.txt
    # a live WAVE, unlike a single cell, REFUSES to run without the pin: 106
    # cells are comparable only if they ran at one commit.
    cp "$PIN_FILE" "${TMP}/pin_for_mocks" 2>/dev/null || true
    rm -f "$PIN_FILE"                       # THE point of this case: no pin file
    out="$(env YAW_GEN_SQUEUE_FAILS=1 YAW_GEN_TEST_MODE=1 YAW_GEN_TEST_RECORD="$LIVE_TRACE" \
               YAW_GEN_PIN_FILE="${TMP}/no_such_pin" YAW_GEN_COMMAND_LOG="$TEST_CMDLOG" \
               bash "$GRID" WAVE=vctl 2>&1)"; rc=$?
    cp "${TMP}/pin_for_mocks" "$PIN_FILE" 2>/dev/null || true
    if [ "$rc" -ne 0 ] && echo "$out" | grep -q "refusing to submit a wave with no campaign pin"; then
      echo "PASS  a live wave without a campaign pin is refused"; PASS=$((PASS + 1))
    else
      echo "FAIL  a wave ran with no campaign pin (rc=${rc})"; FAIL=$((FAIL + 1))
    fi
  else
    echo "SKIP  campaign-pin cases (no HEAD~1)"
  fi

  # --- the REGISTERED GRID, asserted at the surfaces that enforce it ---------
  # The submitter's whitelist and the driver's contract must admit exactly the
  # campaign's arms and cells: a submitter that still admitted an exp_11 cell
  # type would queue jobs the driver then refuses, one wasted slot at a time.
  if grep -q 'in_set "\$val" C4L C8 C16 C32 VANL' "$SUB" && ! grep -q 'C4BACKFILL' "$SUB"; then
    echo "PASS  the submitter admits the five exp_14 arms and no comparator arm"; PASS=$((PASS + 1))
  else
    echo "FAIL  the submitter's arm whitelist is not exp_14's five arms"; FAIL=$((FAIL + 1))
  fi
  if grep -q 'in_set "\$val" rgen zref vctl' "$SUB"; then
    echo "PASS  the submitter admits exactly rgen/zref/vctl"; PASS=$((PASS + 1))
  else
    echo "FAIL  the submitter's cell whitelist is not exp_14's three cells"; FAIL=$((FAIL + 1))
  fi
  if grep -q 'EVAL_ORBIT' "$SUB"; then
    echo "FAIL  the submitter still carries exp_11's EVAL_ORBIT parameter"; FAIL=$((FAIL + 1))
  else
    echo "PASS  EVAL_ORBIT is gone from the submitter"; PASS=$((PASS + 1))
  fi
  # trap-backed pin restoration: an interrupt must not leave the campaign unpinned
  if grep -q "trap 'restore_suite_state" "$GUARD_SELF" \
     && grep -q "INT TERM" "$GUARD_SELF"; then
    echo "PASS  the pin is restored by a trap, not only on the happy path"; PASS=$((PASS + 1))
  else
    echo "FAIL  pin parking has no interruption safety"; FAIL=$((FAIL + 1))
  fi
  # ...and this suite must never park the STORE-WIDE marker another campaign uses:
  # exp_11's suite unpinned it for the duration, which would silently send a
  # neighbour's concurrent submission to HEAD instead of their pin.
  if [ "$(head -1 "$STORE_PIN_MARKER" 2>/dev/null)" = "$STORE_PIN_AT_START" ]; then
    echo "PASS  the store-wide campaign pin is untouched by this suite"; PASS=$((PASS + 1))
  else
    echo "FAIL  the store-wide campaign pin CHANGED during the suite"; FAIL=$((FAIL + 1))
  fi
  if grep -q "DO NOT RUN THIS SUITE DURING ACTIVE CAMPAIGN SUBMISSIONS" "$GUARD_SELF"; then
    echo "PASS  the suite documents that it must not run during submissions"; PASS=$((PASS + 1))
  else
    echo "FAIL  the run-window caveat is undocumented"; FAIL=$((FAIL + 1))
  fi
  # the registry must carry VANL, recorded from the PUBLISHED manifest
  if $PY -c "
import json,sys
r=json.load(open('${EXP11}/arm_launch_registry.json'))['arms']
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
  rm -f "$PIN_FILE"                                   # these cases test PIN_SHA alone
  PIN="$(git rev-parse HEAD~1 2>/dev/null)"
  if [ -n "$PIN" ]; then
    : > "$TRACE"
    out="$(env YAW_GEN_TEST_MODE=1 YAW_GEN_TEST_RECORD="$TRACE" \
               YAW_GEN_TEST_MODE=1 YAW_GEN_TEST_RECORD="$TRACE" bash "$SUB" ARM=C4L CELL=zref STEP=40000 "PIN_SHA=${PIN}" 2>&1)"; rc=$?
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
    INTENT_F="$(ls -t ${TEST_INTENT_DIR}/yaw_gen_submission_C4L_zref_*_jid7654321.txt 2>/dev/null | head -1)"
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
  # a commit this repository does not have must be refused
  out="$(env YAW_GEN_TEST_MODE=1 YAW_GEN_TEST_RECORD="$TRACE" \
             YAW_GEN_TEST_MODE=1 YAW_GEN_TEST_RECORD="$TRACE" bash "$SUB" ARM=C4L CELL=zref STEP=40000 PIN_SHA=0123456789abcdef0123456789abcdef01234567 2>&1)"; rc=$?
  if [ "$rc" -ne 0 ] && echo "$out" | grep -q "is not a commit in this repository"; then
    echo "PASS  a non-existent PIN_SHA is refused"; PASS=$((PASS + 1))
  else
    echo "FAIL  a non-existent PIN_SHA was accepted (rc=${rc})"; FAIL=$((FAIL + 1))
  fi
  out="$(env YAW_GEN_TEST_MODE=1 YAW_GEN_TEST_RECORD="$TRACE" \
             YAW_GEN_TEST_MODE=1 YAW_GEN_TEST_RECORD="$TRACE" bash "$SUB" ARM=C4L CELL=zref STEP=40000 PIN_SHA=deadbeef 2>&1)"; rc=$?
  if [ "$rc" -ne 0 ] && echo "$out" | grep -q "not 40 hex characters"; then
    echo "PASS  a short PIN_SHA is refused"; PASS=$((PASS + 1))
  else
    echo "FAIL  a short PIN_SHA was accepted (rc=${rc})"; FAIL=$((FAIL + 1))
  fi

  # (nothing to restore: the pin under test is the seam file in $TMP)
  # --- the marker is not proof: fd 8 must BE the store lock, at the OUTER entry
  STORE_LOCK="${MAIN_TREE}/.measure_worktrees/.store.lock"
  out="$(env FA_ORBIT_STORE_LOCK_HELD=1 YAW_GEN_TEST_MODE=1 YAW_GEN_TEST_RECORD="$TRACE" \
             YAW_GEN_TEST_MODE=1 YAW_GEN_TEST_RECORD="$TRACE" bash "$SUB" ARM=C4L CELL=zref STEP=40000 2>&1)"; rc=$?
  if [ "$rc" -ne 0 ] && echo "$out" | grep -q "only CLAIMS to hold the store lock"; then
    echo "PASS  a forged lock marker with NO fd 8 is refused"; PASS=$((PASS + 1))
  else
    echo "FAIL  a forged marker with no fd 8 was accepted (rc=${rc})"; FAIL=$((FAIL + 1))
  fi
  DECOY_LOCK="${TMP}/decoy.lock"; : > "$DECOY_LOCK"
  out="$(env FA_ORBIT_STORE_LOCK_HELD=1 YAW_GEN_TEST_MODE=1 YAW_GEN_TEST_RECORD="$TRACE" \
             bash -c 'exec 8>"$1"; exec bash "$2" ARM=C4L CELL=zref STEP=40000' _ "$DECOY_LOCK" "$SUB" 2>&1)"; rc=$?
  if [ "$rc" -ne 0 ] && echo "$out" | grep -q "only CLAIMS to hold the store lock"; then
    echo "PASS  a forged marker with the WRONG fd 8 is refused"; PASS=$((PASS + 1))
  else
    echo "FAIL  a forged marker pointing at another file was accepted (rc=${rc})"; FAIL=$((FAIL + 1))
  fi
  # fd 8 on a DELETED file: readlink -f cannot resolve it, and an unresolvable
  # path must never be read as a match (two empty strings compare equal)
  GONE="${TMP}/gone.lock"; : > "$GONE"
  out="$(env FA_ORBIT_STORE_LOCK_HELD=1 YAW_GEN_TEST_MODE=1 YAW_GEN_TEST_RECORD="$TRACE" \
             bash -c 'exec 8>"$1"; rm -f "$1"; exec bash "$2" ARM=C4L CELL=zref STEP=40000' _ "$GONE" "$SUB" 2>&1)"; rc=$?
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
  env YAW_GEN_TEST_SUBMIT_SLEEP=6 "YAW_GEN_TEST_JOBID=${LIVEJOB}" \
      \
      YAW_GEN_TEST_MODE=1 YAW_GEN_TEST_RECORD="$TRACE" bash "$SUB" ARM=C4L CELL=zref STEP=40000 \
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
  out="$(env YAW_GEN_TEST_MODE=1 YAW_GEN_TEST_RECORD="$TRACE" \
             YAW_GEN_TEST_MODE=1 YAW_GEN_TEST_RECORD="$TRACE" bash "$SUB" ARM=C4L CELL=zref STEP=40000 2>&1)"; rc=$?
  if [ "$rc" -ne 0 ] && echo "$out" | grep -q "requires the deletion freeze"; then
    echo "PASS  a submission without the campaign freeze is refused"; PASS=$((PASS + 1))
  else
    echo "FAIL  a submission ran without the campaign freeze (rc=${rc})"; FAIL=$((FAIL + 1))
  fi
  bash "$HELPER" --freeze "guard suite" >/dev/null 2>&1
  out="$(env YAW_GEN_TEST_MODE=1 YAW_GEN_TEST_RECORD="$TRACE" \
             YAW_GEN_TEST_MODE=1 YAW_GEN_TEST_RECORD="$TRACE" bash "$SUB" ARM=C4L CELL=zref STEP=40000 2>&1)"; rc=$?
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
iso_on                      # POP opt-out 1: isolation is armed again from here

# --- MEASURE_ROOT identity ---------------------------------------------------
echo
echo "--- MEASURE_ROOT identity ---"
assert_isolated "MEASURE_ROOT identity / registry / log naming"
out="$(env DRYRUN=1 ARM=C8 CELL=zref STEP=40000 "EXPECT_SHA=${HEAD_SHA}" "MEASURE_ROOT=$MAIN_TREE" \
       "OUTPUT_ROOT=$GOOD" bash "$SCREEN" 2>&1)"; rc=$?
if [ "$rc" -ne 0 ] && echo "$out" | grep -q "outside the managed .measure_worktrees/ area\|on a BRANCH"; then
  echo "PASS  the mutable MAIN tree is refused as a MEASURE_ROOT (rc=${rc})"; PASS=$((PASS + 1))
else
  echo "FAIL  the main checkout was accepted as a pinned measurement root (rc=${rc})"; FAIL=$((FAIL + 1))
fi

# --- ARM LAUNCH REGISTRY (immutable binding) ---------------------------------
echo
echo "--- arm launch registry binding ---"
REG="${EXP11}/arm_launch_registry.json"
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
  MAN="${GOOD}/exp11_C8/launch_manifest.txt"
  cp "$MAN" "${TMP}/manifest.bak"
  printf '# appended after registration by a guard test\n' >> "$MAN"
  out="$(env "${BASE[@]}" ARM=C8 CELL=zref STEP=40000 bash "$SCREEN" 2>&1)"; rc=$?
  if [ "$rc" -ne 0 ] && echo "$out" | grep -q "changed after it was registered"; then
    echo "PASS  a launch manifest edited after registration is rejected"; PASS=$((PASS + 1))
  else
    echo "FAIL  a tampered launch manifest passed the gate (rc=${rc})"
    echo "$out" | tail -3 | sed 's/^/        | /'; FAIL=$((FAIL + 1))
  fi
  cp "${TMP}/manifest.bak" "$MAN"
  # ...and a RESTART launch (mode != INITIAL) is not a registered launch
  sed -i 's/mode INITIAL/mode RESTART/' "$MAN"
  $PY - "$MAN" "${GOOD}/arm_launch_registry.json" <<'PY'
import hashlib, json, sys                      # re-register the tampered bytes so
reg = json.load(open(sys.argv[2]))             # ONLY the mode differs
reg["arms"]["C8"]["manifest_sha256"] = hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest()
json.dump(reg, open(sys.argv[2], "w"), indent=2)
PY
  out="$(env "${BASE[@]}" ARM=C8 CELL=zref STEP=40000 bash "$SCREEN" 2>&1)"; rc=$?
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
sed -n '/^LOG_CELL_TOKEN="\$CELL_TOKEN"/,/^LOG="\${LOG:-/p' "$SCREEN" > "${TMP}/lognames.sh"
if [ -s "${TMP}/lognames.sh" ]; then
  name_for() {  # $1 CELL $2 CELL_TOKEN $3 SEED $4 jobid
    ( LOGDIR=/logs TS=2026-08-11_05-00-00 ARM=C8 STEP=40000 K=8 \
      CELL="$1" CELL_TOKEN="$2" SEED="$3" SLURM_JOB_ID="$4" LOG=""
      unset LOG
      . "${TMP}/lognames.sh" >/dev/null 2>&1
      echo "$LOG" )
  }
  A="$(name_for rgen _rotrand42 42 111)"; B="$(name_for rgen _rotrand43 43 112)"
  C="$(name_for zref "" 42 113)";        D="$(name_for vctl _rot90 42 114)"
  E="$(name_for vctl _rot45 42 115)";    F="$(name_for zref "" 43 116)"
  U="$(printf '%s\n' "$A" "$B" "$C" "$D" "$E" "$F" | sort -u | grep -c .)"
  if [ "$U" -eq 6 ]; then
    echo "PASS  six same-second cells produce six DISTINCT default log names"; PASS=$((PASS + 1))
  else
    echo "FAIL  default log names collide (${U} unique of 6)"
    printf '%s\n' "$A" "$B" "$C" "$D" "$E" "$F" | sed 's/^/        | /'; FAIL=$((FAIL + 1))
  fi
  # the two rgen seeds differ ONLY by their rotation token — that is the fix
  if [ "$A" != "$B" ] && echo "$B" | grep -q "_rotrand43_" && echo "$D" | grep -q "_rot90_"; then
    echo "PASS  rgen names carry the rotation seed and vctl names the angle"; PASS=$((PASS + 1))
  else
    echo "FAIL  cell tokens missing from the default names ('${B}' / '${D}')"; FAIL=$((FAIL + 1))
  fi
  # still inside LOGDIR and still matching the drift-gate exemption
  if echo "$B" | grep -q "^/logs/yaw_gen_.*_screen\.log$"; then
    echo "PASS  default names stay in LOGDIR and keep the _screen.log suffix"; PASS=$((PASS + 1))
  else
    echo "FAIL  '${B}' no longer matches the exempted screen-log shape"; FAIL=$((FAIL + 1))
  fi
  # an explicit LOG= still wins (operators can override; they just never need to)
  G="$( LOGDIR=/logs TS=t ARM=C8 STEP=40000 SEED=42 K=8 CELL=zref CELL_TOKEN="" \
        LOG=/tmp/explicit.log; . "${TMP}/lognames.sh" >/dev/null 2>&1; echo "$LOG" )"
  [ "$G" = "/tmp/explicit.log" ] \
    && { echo "PASS  an explicit LOG= still overrides the default"; PASS=$((PASS + 1)); } \
    || { echo "FAIL  LOG= override broken ('${G}')"; FAIL=$((FAIL + 1)); }
else
  echo "FAIL  could not extract the log-naming block from the driver"; FAIL=$((FAIL + 1))
fi

# --- THE REGISTERED GRID, end to end (yaw_gen_submit_grid.sh) ----------------
# DRYRUN prints the exact submission argv for every cell and submits nothing.
# Parsing those lines back into cells proves two things at once: that the printed
# set IS the 106-cell grid, and that the argv the submitter would receive names
# the right cell.
echo
echo "--- wave submitter: the 106-cell DRYRUN grid ---"
assert_isolated "the DRYRUN grid and dedup cases"
GRID_OUT="${TMP}/grid_all.txt"
DRYRUN=1 bash "$GRID" WAVE=all > "$GRID_OUT" 2>"${TMP}/grid_all.err"; rc=$?
if [ "$rc" -ne 0 ]; then
  echo "FAIL  DRYRUN WAVE=all exited ${rc}"; sed 's/^/        | /' "${TMP}/grid_all.err"
  FAIL=$((FAIL + 1))
fi
# every printed line back to "ARM CELL STEP SEED K ROT"
$PY - "$GRID_OUT" > "${TMP}/grid_parsed.txt" <<'PY'
import re, sys
for line in open(sys.argv[1]):
    line = line.strip()
    if not line.startswith("bash "):
        continue
    kv = dict(p.split("=", 1) for p in line.split() if "=" in p)
    print(" ".join([kv["ARM"], kv["CELL"], kv["STEP"], kv["SEED"], kv["K"],
                    kv.get("ROTATE_DEG", "-")]))
PY
$PY "$VALIDATOR" grid --wave all | sort > "${TMP}/grid_expected.txt"
sort "${TMP}/grid_parsed.txt" > "${TMP}/grid_got.txt"
if diff -u "${TMP}/grid_expected.txt" "${TMP}/grid_got.txt" > "${TMP}/grid_diff.txt"; then
  echo "PASS  the DRYRUN grid is EXACTLY the registered 106-cell set (sorted diff empty)"
  PASS=$((PASS + 1))
else
  echo "FAIL  the DRYRUN grid differs from the registered set:"
  head -20 "${TMP}/grid_diff.txt" | sed 's/^/        | /'; FAIL=$((FAIL + 1))
fi
# ...and the block structure, read from the PRINTED lines alone
N_ALL="$(grep -c . "${TMP}/grid_got.txt")"
N_RGEN="$(awk '$2=="rgen"' "${TMP}/grid_got.txt" | wc -l)"
N_ZREF="$(awk '$2=="zref"' "${TMP}/grid_got.txt" | wc -l)"
N_VCTL="$(awk '$2=="vctl"' "${TMP}/grid_got.txt" | wc -l)"
N_UNIQ="$(sort -u "${TMP}/grid_got.txt" | grep -c .)"
if [ "$N_ALL" = "106" ] && [ "$N_UNIQ" = "106" ] && [ "$N_RGEN" = "50" ] \
   && [ "$N_ZREF" = "50" ] && [ "$N_VCTL" = "6" ]; then
  echo "PASS  106 unique cells: 50 rgen + 50 zref + 6 vctl"; PASS=$((PASS + 1))
else
  echo "FAIL  block sizes wrong (all=${N_ALL} uniq=${N_UNIQ} rgen=${N_RGEN} zref=${N_ZREF} vctl=${N_VCTL})"
  FAIL=$((FAIL + 1))
fi
VCTL_SET="$(awk '$2=="vctl" {print $1"@"$6}' "${TMP}/grid_got.txt" | sort | tr '\n' ' ')"
if [ "$VCTL_SET" = "C16@90 C32@90 C4L@45 C4L@90 C8@90 VANL@90 " ]; then
  echo "PASS  the six vctl tuples are exactly the registered ones (no VANL@45)"; PASS=$((PASS + 1))
else
  echo "FAIL  vctl tuples are '${VCTL_SET}'"; FAIL=$((FAIL + 1))
fi
# rgen/zref cover 5 arms x 2 K x 5 seeds, and only seeds 42-46
BAD_SEED="$(awk '$2!="vctl" && ($4<42 || $4>46)' "${TMP}/grid_got.txt" | wc -l)"
BAD_K="$(awk '$5!=1 && $5!=8' "${TMP}/grid_got.txt" | wc -l)"
BAD_STEP="$(awk '$3!=40000' "${TMP}/grid_got.txt" | wc -l)"
BAD_ROT="$(awk '$2!="vctl" && $6!="-"' "${TMP}/grid_got.txt" | wc -l)"
if [ "$BAD_SEED" = "0" ] && [ "$BAD_K" = "0" ] && [ "$BAD_STEP" = "0" ] && [ "$BAD_ROT" = "0" ]; then
  echo "PASS  every printed cell is s42-46, K in {1,8}, STEP=40000, no angle outside vctl"
  PASS=$((PASS + 1))
else
  echo "FAIL  grid contains unregistered parameters (seed=${BAD_SEED} K=${BAD_K} step=${BAD_STEP} rot=${BAD_ROT})"
  FAIL=$((FAIL + 1))
fi
# waves partition the grid, and each prints only its own cells
for W in vctl zref rgen; do
  DRYRUN=1 bash "$GRID" "WAVE=${W}" 2>/dev/null | grep -c "CELL=${W}" > "${TMP}/w_${W}.txt"
  DRYRUN=1 bash "$GRID" "WAVE=${W}" 2>/dev/null | grep -vc "CELL=${W}" >> "${TMP}/w_${W}.txt"
done
if [ "$(head -1 "${TMP}/w_vctl.txt")" = "6" ] && [ "$(head -1 "${TMP}/w_zref.txt")" = "50" ] \
   && [ "$(head -1 "${TMP}/w_rgen.txt")" = "50" ] \
   && [ "$(sed -n 2p "${TMP}/w_vctl.txt")" = "0" ] \
   && [ "$(sed -n 2p "${TMP}/w_zref.txt")" = "0" ] \
   && [ "$(sed -n 2p "${TMP}/w_rgen.txt")" = "0" ]; then
  echo "PASS  the three waves partition the grid (6 / 50 / 50, nothing foreign)"; PASS=$((PASS + 1))
else
  echo "FAIL  wave selection is wrong"; FAIL=$((FAIL + 1))
fi
# a DRYRUN must not submit, classify or query the queue: point every seam at a
# command that FAILS, and require success anyway.
if DRYRUN=1 env YAW_GEN_SQUEUE_FAILS=1 \
     bash "$GRID" WAVE=vctl >/dev/null 2>&1; then
  echo "PASS  DRYRUN neither submits nor queries the queue"; PASS=$((PASS + 1))
else
  echo "FAIL  DRYRUN touched sbatch or squeue"; FAIL=$((FAIL + 1))
fi
# ...and the wave submitter never calls sbatch itself: that is the single-cell
# submitter's job, under the store lock.
# (the entry prelude names sbatch only to `unset -f` it, which is the opposite of
#  calling it, so that line is excluded from the search)
if grep -vE '^[[:space:]]*#' "$GRID" | grep -qE '(^|[;&|]|\bthen|\bdo|\{|"\$\()[[:space:]]*sbatch[[:space:]]'; then
  echo "FAIL  the wave submitter calls sbatch directly"; FAIL=$((FAIL + 1))
else
  echo "PASS  the wave submitter never calls sbatch (it delegates, under the lock)"
  PASS=$((PASS + 1))
fi
for BADARG in "WAVE=conf" "WAVE=screen" "FOO=1" "PIN_SHA=deadbeef" "EXCLUDE=neu1;id"; do
  if DRYRUN=1 bash "$GRID" "$BADARG" >/dev/null 2>&1; then
    echo "FAIL  the wave submitter accepted '${BADARG}'"; FAIL=$((FAIL + 1))
  else
    echo "PASS  the wave submitter refuses '${BADARG}'"; PASS=$((PASS + 1))
  fi
done
if DRYRUN=1 MAX_INFLIGHT=32 bash "$GRID" WAVE=vctl >/dev/null 2>&1; then
  echo "FAIL  MAX_INFLIGHT above the 16-slot cap was accepted"; FAIL=$((FAIL + 1))
else
  echo "PASS  MAX_INFLIGHT is capped at the plan's 16 slots"; PASS=$((PASS + 1))
fi

# --- dedup is VALIDATE-before-skip (review B6) -------------------------------
echo
echo "--- wave submitter: validate-before-skip dedup ---"
# A cell with NO artifact classifies MISSING; one with a broken artifact must
# classify INVALID, and the wave must then halt without submitting anything.
DEDUP="${TMP}/dedup_root"
$PY - "$VALIDATOR" "$DEDUP" <<'PY'
import importlib.util, json, os, sys
spec = importlib.util.spec_from_file_location("v", sys.argv[1])
V = importlib.util.module_from_spec(spec); spec.loader.exec_module(V)
root = sys.argv[2]
cell = V.Cell("C4L", "vctl", 40000, 42, 8, 90.0)
d = os.path.join(root, "exp11_C4L", "FLAC_exp11_C4L", "exp11_C4L", "checkpoints")
os.makedirs(d, exist_ok=True)
ckpt = os.path.join(d, "epoch=8-step=40000.ckpt")
open(ckpt, "wb").write(b"")                       # only its NAME matters here
p = V.metrics_path(ckpt, cell)
json.dump({"metrics": {}, "eval_name": "nonsense"}, open(p, "w"))
print(p)
PY
printf '%s\n' "$FAKE_PIN" > "$PIN_FILE"      # a wave requires the pin FILE
OUT="$(env YAW_GEN_SQUEUE_FAILS=1 YAW_GEN_TEST_MODE=1 YAW_GEN_TEST_RECORD="$LIVE_TRACE" \
        YAW_GEN_PIN_FILE="$PIN_FILE" YAW_GEN_COMMAND_LOG="$TEST_CMDLOG" OUTPUT_ROOT="$DEDUP" \
        bash "$GRID" WAVE=vctl "PIN_SHA=${FAKE_PIN}" 2>&1)"; rc=$?
if [ "$rc" -eq 3 ] && echo "$OUT" | grep -q "HALT:" \
   && echo "$OUT" | grep -q "Nothing was submitted"; then
  echo "PASS  an artifact that exists but does not validate HALTS the wave"; PASS=$((PASS + 1))
else
  echo "FAIL  a broken artifact did not halt the wave (rc=${rc})"
  echo "$OUT" | tail -4 | sed 's/^/        | /'; FAIL=$((FAIL + 1))
fi
if echo "$OUT" | grep -q "eval_name"; then
  echo "PASS  the halt names the failing check, so it can be triaged"; PASS=$((PASS + 1))
else
  echo "FAIL  the halt does not say what was wrong"; FAIL=$((FAIL + 1))
fi
# with the broken artifact removed the same wave classifies every cell MISSING
rm -f "${DEDUP}/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/"*_metrics_*.json
OUT="$(env YAW_GEN_SQUEUE_FAILS=1 YAW_GEN_TEST_MODE=1 YAW_GEN_TEST_RECORD="$LIVE_TRACE" \
        YAW_GEN_PIN_FILE="$PIN_FILE" YAW_GEN_COMMAND_LOG="$TEST_CMDLOG" OUTPUT_ROOT="$DEDUP" \
        bash "$GRID" WAVE=vctl "PIN_SHA=${FAKE_PIN}" 2>&1)"; rc=$?
rm -f "$PIN_FILE"
if echo "$OUT" | grep -q "squeue failed - refusing to submit"; then
  echo "PASS  a squeue that fails stops the wave (absence is never assumed)"; PASS=$((PASS + 1))
else
  echo "FAIL  a squeue failure did not stop the wave (rc=${rc})"
  echo "$OUT" | tail -3 | sed 's/^/        | /'; FAIL=$((FAIL + 1))
fi

# --- LIVE WAVE (mocked): the rails the review found missing -------------------
# Nothing is submitted: in TEST MODE the wave starts no submission process at
# all and records the argv it would have used; YAW_GEN_SQUEUE prints a scripted queue. What is proven is what the wave DECIDES
# — which cells it skips, which it halts on, and which argv it would launch.
echo
echo "--- wave submitter: live-wave decisions (mocked submit + squeue) ---"
assert_isolated "the live-wave decision cases"
LIVE="${TMP}/live"; mkdir -p "$LIVE"
LIVE_PIN="$FAKE_PIN"      # a commit of the ISOLATED root, not the campaign's
LIVE_WT="${TMP}/live_wt"; mkdir -p "${LIVE_WT}/.leases"
: > "$LIVE_QUEUE"

# one landed C4L vctl@90 cell, valid under the AUDITED digest
$PY - "$VALIDATOR" "$LIVE" "$LIVE_PIN" <<'PY'
import importlib.util, json, os, sys
spec = importlib.util.spec_from_file_location("v", sys.argv[1])
V = importlib.util.module_from_spec(spec); spec.loader.exec_module(V)
root, pin = sys.argv[2], sys.argv[3]
sha = V.load_ckpt_expect()["C4L"]
cell = V.Cell("C4L", "vctl", 40000, 42, 8, 90.0)
d = os.path.join(root, "exp11_C4L", "FLAC_exp11_C4L", "exp11_C4L", "checkpoints")
os.makedirs(d, exist_ok=True)
ckpt = os.path.join(d, "epoch=8-step=40000.ckpt")
open(ckpt, "wb").write(b"")
p = V.metrics_path(ckpt, cell)
name = V.eval_name(cell)
tuples = [[i, f"{i}|r/{i}.wav", [f"c{i}"], 512] for i in range(V.EXPECTED_COUNT)]
asg = [[i, t[1], 128] for i, t in enumerate(tuples)]
json.dump({"metrics": {"T60_error": 1.0}, "ckpt_path": ckpt, "rotate_deg": 90.0,
           "cond_method": "fa_invariant", "frame_avg_angles": [0.0, 90.0, 180.0, 270.0],
           "cond_autocast": "bf16", "orbit_execution": "batched", "source_sha": pin,
           "batch_size": 64, "n_samples": V.EXPECTED_COUNT,
           "dataset_config": "src/configs/dataset_configs/AR/eval/acousticroom_unseeneval.json",
           "seed": 42, "cfg_scale": 1.0, "steps": 1, "eval_name": name,
           "weights_source": "ema", "device": "cuda",
           # the per-scene estimand (plan §4): a landed cell carries it or it is
           # not a cell of this campaign (round-3 review B1)
           "by_scene": {f"Room{i}/Room{i}_idx_{i}": {"T60": 9.0 + i}
                        for i in range(V.EXPECTED_SCENES)},
           "per_scene_schema": V.PER_SCENE_SCHEMA,
           "scene_count": V.EXPECTED_SCENES}, open(p, "w"))
json.dump({"arm": "C4L", "step": 40000, "seed": 42, "K": 8, "eval_name": name,
           "cfg_scale": 1.0, "steps": 1, "model_config": "x", "model_config_sha256": "c" * 64,
           "dataset_config": "x", "ckpt_path": ckpt, "ckpt_sha256": sha, "use_ema": True,
           "frame_avg_angles": [0.0, 90.0, 180.0, 270.0], "cond_method": "fa_invariant",
           "cond_autocast": "bf16", "commit": pin, "cell": "vctl",
           "training_orbit": 4, "eval_orbit": 4, "rotate_mode": "fixed",
           "rotate_deg": 90.0, "rotate_seed": None, "expected_stream_count": V.EXPECTED_COUNT,
           "record_stream": True, "record_per_scene": True,
           "stream_sidecar": "s", "batch_size": 64,
           "num_workers": 4}, open(p + ".screenmeta.json", "w"))
json.dump({"schema_version": 1, "fingerprint_schema": 1, "rotate_mode": "fixed",
           "rotate_seed": None, "rotate_deg": 90.0, "img_w": 512,
           "stream_count": V.EXPECTED_COUNT, "input_tuples": tuples,
           "offsets": [128] * V.EXPECTED_COUNT, "assignment_tuples": asg,
           "input_hash": V.canonical_stream_hash(tuples),
           "assignment_hash": V.canonical_stream_hash(asg)},
          open(p.replace(".json", ".stream.json"), "w"))
print(p)
PY

live_wave() {   # <extra env...> — run the vctl wave with everything mocked
  : > "$LIVE_TRACE"
  printf '%s\n' "$LIVE_PIN" > "$PIN_FILE"
  env LIVE_TRACE="$LIVE_TRACE" \
      COMMAND_LOG_UNDER_TEST="$TEST_CMDLOG" \
      YAW_GEN_SQUEUE_FIXTURE="$LIVE_QUEUE" YAW_GEN_TEST_MODE=1 YAW_GEN_TEST_RECORD="$LIVE_TRACE" \
      YAW_GEN_TEST_RECORD="$LIVE_TRACE" YAW_GEN_WT_DIR="$LIVE_WT" YAW_GEN_PIN_FILE="$PIN_FILE" \
      YAW_GEN_COMMAND_LOG="$TEST_CMDLOG" OUTPUT_ROOT="$LIVE" "$@" \
      bash "$GRID" WAVE=vctl 2>&1
}
CMDLOG_BAK="${TMP}/command_md.bak"      # (the live log is never written any more)

# (a) the landed, audited-digest cell is SKIPPED; every other vctl cell is submitted
: > "$LIVE_QUEUE"
OUT="$(live_wave)"; rc=$?
if [ "$rc" -eq 0 ] && echo "$OUT" | grep -q "SKIP  exp14-screen-C4L-vctl-rot90-40000-s42-K8: already measured"; then
  echo "PASS  a landed cell that validates against the AUDITED digest is skipped"; PASS=$((PASS + 1))
else
  echo "FAIL  the valid cell was not skipped (rc=${rc})"; echo "$OUT" | tail -4 | sed 's/^/        | /'
  FAIL=$((FAIL + 1))
fi
# (b) ...and C4L@45 is a DIFFERENT cell, submitted in the same wave (review B2)
if grep -q 'CELL=vctl' "$LIVE_TRACE" && grep -q 'ROTATE_DEG=45' "$LIVE_TRACE" \
   && echo "$OUT" | grep -q "SUBMIT exp14-screen-C4L-vctl-rot45-40000-s42-K8"; then
  echo "PASS  C4L vctl@45 and vctl@90 are separate cells with separate job names"; PASS=$((PASS + 1))
else
  echo "FAIL  the two C4L vctl angles were not distinguished"; sed 's/^/        | /' "$LIVE_TRACE"
  FAIL=$((FAIL + 1))
fi
# (c) the command log records the launch BEFORE the submitter runs (review B7)
if grep -q "launching-line-present" "$LIVE_TRACE" && ! grep -q "launching-line-MISSING" "$LIVE_TRACE"; then
  echo "PASS  the command log is written BEFORE the submission, not after"; PASS=$((PASS + 1))
else
  echo "FAIL  a job could be submitted before its command was recorded"; FAIL=$((FAIL + 1))
fi
# (d) every submitted job name equals the validator's canonical one
NAME_MISMATCH=0
for SPEC in "C4L vctl 42 8 45" "C8 vctl 42 8 90" "VANL vctl 42 8 90"; do
  # shellcheck disable=SC2086
  set -- $SPEC
  WANT="$($PY "$VALIDATOR" jobname --arm "$1" --cell "$2" --step 40000 --seed "$3" \
            --k "$4" --rotate-deg "$5")"
  echo "$OUT" | grep -q "SUBMIT ${WANT}$" || { echo "FAIL  wave did not submit ${WANT}"; NAME_MISMATCH=1; }
done
if [ "$NAME_MISMATCH" = "0" ]; then
  echo "PASS  every job name the wave submits is exp14_validate_cell.job_name's"; PASS=$((PASS + 1))
else
  FAIL=$((FAIL + 1))
fi
# (e) a wrong-checkpoint artifact HALTS the wave (review B4)
SM="$(ls "${LIVE}"/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/*.screenmeta.json)"
cp "$SM" "${TMP}/sm.bak"
$PY - "$SM" <<'PY'
import json, sys
d = json.load(open(sys.argv[1])); d["ckpt_sha256"] = "9" * 64
json.dump(d, open(sys.argv[1], "w"))
PY
OUT="$(live_wave)"; rc=$?
if [ "$rc" -eq 3 ] && echo "$OUT" | grep -q "HALT:" && echo "$OUT" | grep -q "ckpt_sha256"; then
  echo "PASS  an artifact from ANOTHER checkpoint halts the wave"; PASS=$((PASS + 1))
else
  echo "FAIL  a wrong-checkpoint artifact did not halt the wave (rc=${rc})"
  echo "$OUT" | tail -4 | sed 's/^/        | /'; FAIL=$((FAIL + 1))
fi
if [ ! -s "$LIVE_TRACE" ]; then
  echo "PASS  the halted wave submitted nothing at all"; PASS=$((PASS + 1))
else
  echo "FAIL  the halted wave still submitted"; sed 's/^/        | /' "$LIVE_TRACE"; FAIL=$((FAIL + 1))
fi
cp "${TMP}/sm.bak" "$SM"
# (f) an in-flight job WITH its lease is skipped; without it the wave HALTS (B5)
JOBID_INFLIGHT=5559999
printf '%s exp14-screen-C8-vctl-rot90-40000-s42-K8\n' "$JOBID_INFLIGHT" > "$LIVE_QUEUE"
printf 'jobid %s\n' "$JOBID_INFLIGHT" > "${LIVE_WT}/.leases/${JOBID_INFLIGHT}"
OUT="$(live_wave)"; rc=$?
if [ "$rc" -eq 0 ] && echo "$OUT" | grep -q "SKIP  exp14-screen-C8-vctl-rot90-40000-s42-K8: job ${JOBID_INFLIGHT} is in flight and holds its lease"; then
  echo "PASS  an in-flight job that holds its lease is skipped as in flight"; PASS=$((PASS + 1))
else
  echo "FAIL  a leased in-flight job was not recognised (rc=${rc})"
  echo "$OUT" | tail -4 | sed 's/^/        | /'; FAIL=$((FAIL + 1))
fi
rm -f "${LIVE_WT}/.leases/${JOBID_INFLIGHT}"
OUT="$(live_wave)"; rc=$?
if [ "$rc" -eq 5 ] && echo "$OUT" | grep -q "holds NO lease under"; then
  echo "PASS  a name-matching job with NO lease halts the wave"; PASS=$((PASS + 1))
else
  echo "FAIL  an unleased in-flight job was treated as ours (rc=${rc})"
  echo "$OUT" | tail -4 | sed 's/^/        | /'; FAIL=$((FAIL + 1))
fi
: > "$LIVE_QUEUE"
# (g) the pin FILE is required even when PIN_SHA is supplied (review B3)
rm -f "$PIN_FILE"
OUT="$(env YAW_GEN_SQUEUE_FIXTURE="$LIVE_QUEUE" YAW_GEN_TEST_MODE=1 YAW_GEN_TEST_RECORD="$LIVE_TRACE" \
       YAW_GEN_TEST_MODE=1 YAW_GEN_PIN_FILE="$PIN_FILE" YAW_GEN_COMMAND_LOG="$TEST_CMDLOG" \
       OUTPUT_ROOT="$LIVE" bash "$GRID" WAVE=vctl \
       "PIN_SHA=${LIVE_PIN}" 2>&1)"; rc=$?
if [ "$rc" -ne 0 ] && echo "$OUT" | grep -q "no campaign pin FILE"; then
  echo "PASS  PIN_SHA cannot substitute for the campaign pin file"; PASS=$((PASS + 1))
else
  echo "FAIL  PIN_SHA bypassed the pin file (rc=${rc})"; FAIL=$((FAIL + 1))
fi
# (h) ...and a PIN_SHA that disagrees with the file is refused
printf '%s\n' "$LIVE_PIN" > "$PIN_FILE"
OTHER_PIN="$(git -C "$FAKE_REPO" rev-parse "${LIVE_PIN}^" 2>/dev/null)"  # the pin's PARENT
if [ -n "$OTHER_PIN" ]; then
  OUT="$(env YAW_GEN_SQUEUE_FIXTURE="$LIVE_QUEUE" YAW_GEN_TEST_MODE=1 YAW_GEN_TEST_RECORD="$LIVE_TRACE" \
         YAW_GEN_TEST_MODE=1 YAW_GEN_PIN_FILE="$PIN_FILE" YAW_GEN_COMMAND_LOG="$TEST_CMDLOG" \
         OUTPUT_ROOT="$LIVE" bash "$GRID" WAVE=vctl \
         "PIN_SHA=${OTHER_PIN}" 2>&1)"; rc=$?
  if [ "$rc" -ne 0 ] && echo "$OUT" | grep -q "disagrees with the campaign pin"; then
    echo "PASS  a PIN_SHA disagreeing with the pin file is refused"; PASS=$((PASS + 1))
  else
    echo "FAIL  a disagreeing PIN_SHA was accepted (rc=${rc})"; FAIL=$((FAIL + 1))
  fi
fi
# (i) a pin file naming a commit this repository does not have is refused
printf '%s\n' "0123456789abcdef0123456789abcdef01234567" > "$PIN_FILE"
OUT="$(env YAW_GEN_SQUEUE_FIXTURE="$LIVE_QUEUE" YAW_GEN_TEST_MODE=1 YAW_GEN_TEST_RECORD="$LIVE_TRACE" \
       YAW_GEN_TEST_MODE=1 YAW_GEN_PIN_FILE="$PIN_FILE" YAW_GEN_COMMAND_LOG="$TEST_CMDLOG" \
       OUTPUT_ROOT="$LIVE" bash "$GRID" WAVE=vctl 2>&1)"; rc=$?
if [ "$rc" -ne 0 ] && echo "$OUT" | grep -q "is not a commit in this repository"; then
  echo "PASS  a pin file naming an unknown commit is refused"; PASS=$((PASS + 1))
else
  echo "FAIL  an unknown campaign pin was accepted (rc=${rc})"; FAIL=$((FAIL + 1))
fi
rm -f "$PIN_FILE"
# The live command log must be exactly as this suite found it: untouched.
if [ "$(sha256sum "$LIVE_CMDLOG" 2>/dev/null | cut -d' ' -f1)" = "${LIVE_CMDLOG_SUM_AT_START}" ]; then
  echo "PASS  the campaign command log is BYTE-IDENTICAL to what this suite found"
  PASS=$((PASS + 1))
else
  echo "FAIL  the suite modified the CAMPAIGN command log"; FAIL=$((FAIL + 1))
  diff <(echo) <(tail -3 "$LIVE_CMDLOG") | sed 's/^/        | /'
fi

# --- X1: the lease seam may not weaken a LIVE wave (re-verify B5) ------------
# YAW_GEN_WT_DIR made the pinned-worktree lease invariant overridable: any
# directory holding .leases/<jid> would let a wave skip a job that holds NO lease
# in the campaign's own worktree. In a live wave the directory is now DERIVED
# from the pin and the variable is ignored; honouring it requires an explicit
# test mode, which in turn refuses to run against the real submitter.
echo
echo "--- wave submitter: the lease directory is derived from the pin ---"
printf '%s\n' "$LIVE_PIN" > "$PIN_FILE"
JOBID_SEAM=5557777
printf '%s exp14-screen-C8-vctl-rot90-40000-s42-K8\n' "$JOBID_SEAM" > "$LIVE_QUEUE"
printf 'jobid %s\n' "$JOBID_SEAM" > "${LIVE_WT}/.leases/${JOBID_SEAM}"   # lease in the SEAM dir only
: > "$LIVE_TRACE"
# NOT in test mode. This case must be INCAPABLE of submitting even if a real
# campaign pin exists, so it runs a COPY of the wave submitter whose MAIN_REPO is
# a temporary directory: the real repo's pin file is invisible to it, the copy
# finds no pin of its own, and it therefore stops at the pin gate — structurally,
# not by luck of the campaign's current state.
grep -q "^MAIN_REPO=${FAKE_REPO}$" "$GRID_FAKE" && grep -q "^MAIN_REPO=${FAKE_REPO}$" "$SUB_FAKE" \
  && { echo "PASS  every live-mode probe runs against a temporary MAIN_REPO"; PASS=$((PASS + 1)); } \
  || { echo "FAIL  could not retarget the kit's MAIN_REPO"; FAIL=$((FAIL + 1)); }
OUT="$(env -u YAW_GEN_PIN_FILE -u YAW_GEN_INTENT_DIR -u YAW_GEN_MAIN_REPO \
        OUTPUT_ROOT="$LIVE" bash "$GRID_FAKE" WAVE=vctl 2>&1)"; rc=$?
if [ "$rc" -eq 2 ] && echo "$OUT" | grep -q "no campaign pin FILE" \
   && echo "$OUT" | grep -q "${FAKE_REPO}"; then
  echo "PASS  a live wave stops AT THE PIN GATE, with the real repo's pin invisible"
  PASS=$((PASS + 1))
else
  echo "FAIL  the live wave did not stop at the pin gate (rc=${rc})"
  echo "$OUT" | tail -3 | sed 's/^/        | /'; FAIL=$((FAIL + 1))
fi
# ...and a live wave that so much as CARRIES the lease seam is refused outright —
# stronger than ignoring it, and it happens at the ENTRY, before any gate runs.
OUT="$(env -u YAW_GEN_PIN_FILE -u YAW_GEN_INTENT_DIR -u YAW_GEN_MAIN_REPO \
        YAW_GEN_WT_DIR="$LIVE_WT" OUTPUT_ROOT="$LIVE" bash "$GRID_FAKE" WAVE=vctl 2>&1)"; rc=$?
if [ "$rc" -ne 0 ] && echo "$OUT" | grep -q "YAW_GEN_WT_DIR is not on this mode's allowlist"; then
  echo "PASS  a live wave carrying the lease seam is REFUSED at the entry"; PASS=$((PASS + 1))
else
  echo "FAIL  a live wave tolerated the lease seam (rc=${rc})"; FAIL=$((FAIL + 1))
fi
OUT="$(env \
        YAW_GEN_SQUEUE_FIXTURE="$LIVE_QUEUE" YAW_GEN_TEST_MODE=1 YAW_GEN_TEST_RECORD="$LIVE_TRACE" \
        YAW_GEN_PIN_FILE="$PIN_FILE" YAW_GEN_COMMAND_LOG="$TEST_CMDLOG" \
        OUTPUT_ROOT="$LIVE" bash "$GRID" WAVE=vctl 2>&1)"; rc=$?
if [ "$rc" -eq 5 ] && echo "$OUT" | grep -q "holds NO lease under"; then
  echo "PASS  the unleased in-flight job halts the wave even with the seam set"
  PASS=$((PASS + 1))
else
  echo "FAIL  the lease check did not halt (rc=${rc})"
  echo "$OUT" | tail -4 | sed 's/^/        | /'; FAIL=$((FAIL + 1))
fi
# ...and TEST MODE never starts a submission process: the record shows simulated
# submissions and the queue never grows.
N_EXP14_BEFORE="$(squeue -h -u "$(id -un)" -o "%j" 2>/dev/null | grep -c '^exp14-' || true)"
if [ "$N_EXP14_BEFORE" = "0" ]; then
  echo "PASS  no exp14- job exists after a simulated wave"; PASS=$((PASS + 1))
else
  echo "FAIL  a simulated wave left ${N_EXP14_BEFORE} exp14- jobs in the queue"; FAIL=$((FAIL + 1))
fi
rm -f "${LIVE_WT}/.leases/${JOBID_SEAM}"; : > "$LIVE_QUEUE"

# --- X2: durability failures are FATAL, not warnings (re-verify B7) ----------
echo
echo "--- durability: a record that cannot be made durable stops the launch ---"
: > "$LIVE_TRACE"
OUT="$(live_wave YAW_GEN_SYNC_FAILS=1)"; rc=$?
if [ "$rc" -ne 0 ] && echo "$OUT" | grep -q "cannot durably record"; then
  echo "PASS  a wave whose command log cannot be flushed refuses to submit"; PASS=$((PASS + 1))
else
  echo "FAIL  an unflushable command log did not stop the wave (rc=${rc})"
  echo "$OUT" | tail -3 | sed 's/^/        | /'; FAIL=$((FAIL + 1))
fi
if ! grep -q '^submit ' "$LIVE_TRACE" 2>/dev/null; then
  echo "PASS  ...and it submitted nothing (the record shows the flush, never a submit)"
  PASS=$((PASS + 1))
else
  echo "FAIL  it submitted despite the durability failure"; sed 's/^/        | /' "$LIVE_TRACE"
  FAIL=$((FAIL + 1))
fi
rm -f "$PIN_FILE"

# OPT-OUT 1 again (see header): the cases up to the trap check drive the
# SINGLE-CELL submitter, whose store lock, lease and worktree are the real shared
# ones — that machinery IS the subject. Still test mode: none can submit.
iso_off "the held-job cancellation cases"

# a HELD job whose intent manifest cannot be published is CANCELLED
: > "$TRACE"
BLOCK="${TEST_INTENT_DIR}/yaw_gen_submission_C4L_zref_S40000_s42_K8_jid7654321.txt"
rm -rf "$BLOCK"; mkdir -p "$BLOCK"          # publishing onto a directory must fail
out="$(env        \
             YAW_GEN_PIN_FILE="$PIN_FILE" YAW_GEN_INTENT_DIR="$TEST_INTENT_DIR" \
           YAW_GEN_TEST_MODE=1 YAW_GEN_TEST_RECORD="$TRACE" bash "$SUB" ARM=C4L CELL=zref STEP=40000 2>&1)"; rc=$?
if [ "$rc" -ne 0 ] && grep -q "scancel 7654321" "$TRACE" \
   && ! grep -q "scontrol release" "$TRACE"; then
  echo "PASS  a held job whose intent manifest cannot be published is CANCELLED"
  PASS=$((PASS + 1))
else
  echo "FAIL  an unpublishable intent left the job alive (rc=${rc})"
  sed 's/^/        | /' "$TRACE"; FAIL=$((FAIL + 1))
fi
rm -rf "$BLOCK"
bash "$HELPER" --release 7654321 "$WT" >/dev/null 2>&1

# an interrupted submission (SIGTERM between hold and release) cancels too
: > "$TRACE"
rm -f "${TRACE}.releasing"
env YAW_GEN_TEST_MODE=1 YAW_GEN_TEST_RECORD="$TRACE" YAW_GEN_TEST_RELEASE_SLEEP=10 \
    YAW_GEN_TEST_MODE=1 YAW_GEN_TEST_RECORD="$TRACE" bash "$SUB" ARM=C4L CELL=zref STEP=40000 >/dev/null 2>&1 &
KILL_PID=$!
for _ in 1 2 3 4 5 6 7 8 9 10; do [ -f "${TRACE}.releasing" ] && break; sleep 1; done
if [ -f "${TRACE}.releasing" ]; then
  kill -TERM "$KILL_PID" 2>/dev/null
  wait "$KILL_PID" 2>/dev/null; KILL_RC=$?
  if grep -q "scancel 7654321" "$TRACE"; then
    echo "PASS  a submission killed between hold and release cancels the held job"
    PASS=$((PASS + 1))
  else
    echo "FAIL  an interrupted submission left a held job behind"; sed 's/^/        | /' "$TRACE"
    FAIL=$((FAIL + 1))
  fi
  # ...and it must TERMINATE nonzero, so nothing downstream can run afterwards
  if [ "$KILL_RC" -ne 0 ]; then
    echo "PASS  the interrupted submission exits nonzero (rc=${KILL_RC})"; PASS=$((PASS + 1))
  else
    echo "FAIL  an interrupted submission exited 0"; FAIL=$((FAIL + 1))
  fi
else
  wait "$KILL_PID" 2>/dev/null
  echo "FAIL  the interrupt fixture never reached the release window"; FAIL=$((FAIL + 1))
fi
bash "$HELPER" --release 7654321 "$WT" >/dev/null 2>&1
rm -f ${TEST_INTENT_DIR}/yaw_gen_submission_C4L_zref_*_jid7654321.txt

iso_on                      # POP opt-out 2: isolation is armed again from here

# --- Y1: a test may not reach ANY external submit executable ----------------
# "Is this executable a mock?" is undecidable: a wrapper, a copy, a hard link or
# an absolute path to the real sbatch all differ from the string "sbatch". The
# kit therefore no longer asks — test mode starts no submit command at all — and
# every override is refused. These cases use harmless stand-ins (/bin/echo and a
# wrapper that only records) so the CLASS is demonstrated without any risk of a
# submission: what is asserted is the REFUSAL, not what the executable does.
echo
# U1: the preamble runs BEFORE `set -euo pipefail`, so a shadowed set() cannot
# turn the shell options off either. Extracted from the shipped file and sourced,
# the way the log-naming case tests the shipped naming block.
sed -n '1,/^set -euo pipefail$/p' "$SUB" > "${TMP}/preamble.sh"
PRE_FLAGS="$(bash -c 'set() { echo "SET-HIJACKED"; }; export -f set
                      . "$1" >/dev/null 2>&1
                      printf "%s|" "$-"; set -o | grep -E "^(errexit|nounset|pipefail)" | tr -d " \t" | tr "\n" "|"' \
             _ "${TMP}/preamble.sh" 2>/dev/null)"
case "$PRE_FLAGS" in
  *errexiton*nounseton*pipefailon*) echo "PASS  a shadowed set() cannot disable errexit/nounset/pipefail (${PRE_FLAGS})"
                                    PASS=$((PASS + 1)) ;;
  *) echo "FAIL  the shell options did not survive a shadowed set(): '${PRE_FLAGS}'"
     FAIL=$((FAIL + 1)) ;;
esac

echo "--- submit-executable overrides are refused, not trusted ---"
assert_isolated "the submit-executable override probes"
for SPELLING in "/bin/echo" "$(command -v sbatch 2>/dev/null || echo /usr/bin/sbatch)"; do
  out="$(env -u YAW_GEN_PIN_FILE -u YAW_GEN_INTENT_DIR -u YAW_GEN_MAIN_REPO "FA_ORBIT_SBATCH=${SPELLING}" \
         bash "$SUB_FAKE" ARM=C4L CELL=zref STEP=40000 2>&1)"; rc=$?
  if [ "$rc" -ne 0 ] && echo "$out" | grep -q "not on this mode's allowlist"; then
    echo "PASS  FA_ORBIT_SBATCH='${SPELLING}' is refused in live mode"; PASS=$((PASS + 1))
  else
    echo "FAIL  an absolute-path 'mock' was accepted (rc=${rc}) — it could be the real sbatch"
    echo "$out" | tail -3 | sed 's/^/        | /'; FAIL=$((FAIL + 1))
  fi
done
WRAP="${TMP}/sbatch_wrapper.sh"
printf '#!/usr/bin/env bash\necho "WRAPPER-WOULD-HAVE-SUBMITTED $*" >> "%s"\necho 9999999\n' \
  "${TMP}/wrapper_calls.txt" > "$WRAP"; chmod +x "$WRAP"
rm -f "${TMP}/wrapper_calls.txt"
out="$(env -u YAW_GEN_PIN_FILE -u YAW_GEN_INTENT_DIR -u YAW_GEN_MAIN_REPO "FA_ORBIT_SBATCH=${WRAP}" \
       bash "$SUB_FAKE" ARM=C4L CELL=zref STEP=40000 2>&1)"; rc=$?
if [ "$rc" -ne 0 ] && [ ! -f "${TMP}/wrapper_calls.txt" ]; then
  echo "PASS  a wrapper script is refused and never executed"; PASS=$((PASS + 1))
else
  echo "FAIL  a wrapper 'mock' ran (rc=${rc}) — a wrapper can call the real sbatch"
  FAIL=$((FAIL + 1))
fi
out="$(env YAW_GEN_TEST_MODE=1 "FA_ORBIT_SBATCH=${WRAP}" bash "$SUB" ARM=C4L CELL=zref STEP=40000 2>&1)"; rc=$?
if [ "$rc" -ne 0 ] && echo "$out" | grep -q "not on this mode's allowlist"; then
  echo "PASS  an override is refused in TEST MODE too (test mode runs no submit command)"
  PASS=$((PASS + 1))
else
  echo "FAIL  test mode accepted a submit-executable override (rc=${rc})"; FAIL=$((FAIL + 1))
fi
out="$(env YAW_GEN_SUBMIT="$WRAP" YAW_GEN_TEST_MODE=1 YAW_GEN_PIN_FILE="$PIN_FILE" \
       bash "$GRID" WAVE=vctl 2>&1)"; rc=$?
if [ "$rc" -ne 0 ] && echo "$out" | grep -q "not on this mode's allowlist"; then
  echo "PASS  the wave refuses an overridden submitter outright"; PASS=$((PASS + 1))
else
  echo "FAIL  the wave accepted an overridden submitter (rc=${rc})"; FAIL=$((FAIL + 1))
fi
# a LIVE wave may not carry the failure-injection seams either
out="$(env -u YAW_GEN_PIN_FILE -u YAW_GEN_INTENT_DIR -u YAW_GEN_MAIN_REPO YAW_GEN_SQUEUE_FAILS=1 \
       bash "$GRID_FAKE" WAVE=vctl 2>&1)"; rc=$?
if [ "$rc" -ne 0 ] && echo "$out" | grep -q "not on this mode's allowlist"; then
  echo "PASS  a live wave refuses the failure-injection seams"; PASS=$((PASS + 1))
else
  echo "FAIL  a live wave accepted an injection seam (rc=${rc})"; FAIL=$((FAIL + 1))
fi
# an UNKNOWN seam name is refused in both modes: the allowlist is the doctrine
for MODEENV in "YAW_GEN_TEST_MODE=1" "DRYRUN=0"; do
  out="$(env -u YAW_GEN_PIN_FILE -u YAW_GEN_INTENT_DIR -u YAW_GEN_MAIN_REPO $MODEENV YAW_GEN_FOO=1 \
         bash "$GRID_FAKE" WAVE=vctl 2>&1)"; rc=$?
  if [ "$rc" -ne 0 ] && echo "$out" | grep -q "YAW_GEN_FOO is not on this mode's allowlist"; then
    echo "PASS  the wave refuses an unknown YAW_GEN_* seam (${MODEENV})"; PASS=$((PASS + 1))
  else
    echo "FAIL  an unknown seam was tolerated (${MODEENV}, rc=${rc})"; FAIL=$((FAIL + 1))
  fi
  out="$(env -u YAW_GEN_PIN_FILE -u YAW_GEN_INTENT_DIR -u YAW_GEN_MAIN_REPO $MODEENV YAW_GEN_FOO=1 \
         bash "$SUB_FAKE" ARM=C4L CELL=zref STEP=40000 2>&1)"; rc=$?
  if [ "$rc" -ne 0 ] && echo "$out" | grep -q "YAW_GEN_FOO is not on this mode's allowlist"; then
    echo "PASS  the submitter refuses an unknown YAW_GEN_* seam (${MODEENV})"; PASS=$((PASS + 1))
  else
    echo "FAIL  the submitter tolerated an unknown seam (${MODEENV}, rc=${rc})"; FAIL=$((FAIL + 1))
  fi
done
# ...and no executable-bearing name survives anywhere in either script
if grep -vE '^[[:space:]]*#' "$SUB" "$GRID" | grep -qE 'YAW_GEN_(SQUEUE|SYNC|SUBMIT|PY)[^_]'; then
  echo "FAIL  an executable-bearing env seam is still read"; FAIL=$((FAIL + 1))
else
  echo "PASS  no environment variable names a command in either script"; PASS=$((PASS + 1))
fi

# --- Z2: neither PATH nor an exported function can swap a Slurm binary -------
echo
echo "--- live binaries resolve to absolute paths from a sanitized PATH ---"
FAKEBIN="${TMP}/fakebin"; mkdir -p "$FAKEBIN"
printf '#!/usr/bin/env bash
echo "FAKE-SQUEUE-RAN" >> "%s"
' "${TMP}/fake_calls.txt" > "${FAKEBIN}/squeue"
printf '#!/usr/bin/env bash
echo "FAKE-SBATCH-RAN" >> "%s"
' "${TMP}/fake_calls.txt" > "${FAKEBIN}/sbatch"
chmod +x "${FAKEBIN}/squeue" "${FAKEBIN}/sbatch"
rm -f "${TMP}/fake_calls.txt"
RESOLVED="$(env -u YAW_GEN_PIN_FILE -u YAW_GEN_INTENT_DIR -u YAW_GEN_MAIN_REPO PATH="${FAKEBIN}:$PATH" bash -c '
  sbatch() { echo "EXPORTED-FUNCTION-RAN"; }; export -f sbatch
  squeue() { echo "EXPORTED-FUNCTION-RAN"; }; export -f squeue
  bash "$1" WAVE=vctl 2>&1' _ "$GRID_FAKE" | sed -n 's/^resolved: //p')"
if [ "$RESOLVED" = "squeue=/usr/bin/squeue sync=/usr/bin/sync" ]    && [ ! -f "${TMP}/fake_calls.txt" ]; then
  echo "PASS  a poisoned PATH and exported functions do not change the resolved binaries"
  PASS=$((PASS + 1))
else
  echo "FAIL  resolution was influenced ('${RESOLVED}', fake ran: $([ -f "${TMP}/fake_calls.txt" ] && echo YES || echo no))"
  FAIL=$((FAIL + 1))
fi

# --- Y2: an unparseable job id must still cancel, BY NAME --------------------
# sbatch can succeed while its output is unreadable; the id is then unknown but
# the JOB IS REAL. The abort guard is armed before the submission and falls back
# to the cell's own (injective) job name.
echo
echo "--- an unreadable job id cancels by NAME ---"
iso_off "the malformed-job-id cancellation case"   # drives the real store lock
: > "$TRACE"
out="$(env YAW_GEN_TEST_MODE=1 YAW_GEN_TEST_RECORD="$TRACE" YAW_GEN_TEST_JOBID="not-an-id" \
       bash "$SUB" ARM=C4L CELL=zref STEP=40000 2>&1)"; rc=$?
if [ "$rc" -ne 0 ] \
   && grep -q -- "scancel --name=exp14-screen-C4L-zref-40000-s42-K8" "$TRACE" \
   && ! grep -q "scontrol release" "$TRACE"; then
  echo "PASS  a malformed job id cancels by name and never releases"; PASS=$((PASS + 1))
else
  echo "FAIL  a malformed job id left a possible job uncancelled (rc=${rc})"
  sed 's/^/        | /' "$TRACE"; FAIL=$((FAIL + 1))
fi
iso_on                      # POP: isolation armed again
# structural: the guard is armed BEFORE the submission, not after it returns
TRAP_LINE="$(grep -n '^trap cancel_held_job EXIT$' "$SUB" | head -1 | cut -d: -f1)"
SUBMIT_LINE="$(grep -n 'JOBID="\$(slurm_submit_hold' "$SUB" | head -1 | cut -d: -f1)"
if [ -n "$TRAP_LINE" ] && [ -n "$SUBMIT_LINE" ] && [ "$TRAP_LINE" -lt "$SUBMIT_LINE" ]; then
  echo "PASS  the abort guard is armed BEFORE the submission (trap@${TRAP_LINE} < submit@${SUBMIT_LINE})"
  PASS=$((PASS + 1))
else
  echo "FAIL  the abort guard is armed after the submission (trap@${TRAP_LINE:-none} submit@${SUBMIT_LINE:-none})"
  FAIL=$((FAIL + 1))
fi
if grep -q "trap 'cancel_held_job; exit 130' INT" "$SUB" \
   && grep -q "trap 'cancel_held_job; exit 143' TERM" "$SUB"; then
  echo "PASS  INT/TERM cancel AND terminate nonzero (release can never follow)"; PASS=$((PASS + 1))
else
  echo "FAIL  a signal handler cancels but falls through"; FAIL=$((FAIL + 1))
fi
# and no external submit command survives anywhere in the kit's live path
if grep -vE '^[[:space:]]*#' "$SUB" | grep -qE '"\$SBATCH"|"\$SCONTROL"|"\$SCANCEL"'; then
  echo "FAIL  the submitter still executes an overridable Slurm binary variable"; FAIL=$((FAIL + 1))
else
  echo "PASS  the submitter calls slurm_* wrappers only (no overridable binary)"; PASS=$((PASS + 1))
fi

# a manifest without its completion sentinel is rejected BY THE READER
PART="${TMP}/partial_manifest.txt"
printf 'job 1 name x submitted_at now by a@b\narm C4L cell zref step 40000 seed 42 K 8\n' > "$PART"
if env -u YAW_GEN_PIN_FILE -u YAW_GEN_INTENT_DIR -u YAW_GEN_MAIN_REPO bash "$SUB" --verify-manifest "$PART" >/dev/null 2>&1; then
  echo "FAIL  a truncated intent manifest was accepted as complete"; FAIL=$((FAIL + 1))
else
  echo "PASS  a truncated intent manifest is rejected by the reader"; PASS=$((PASS + 1))
fi
printf 'manifest_complete yes\n' >> "$PART"
if env -u YAW_GEN_PIN_FILE -u YAW_GEN_INTENT_DIR -u YAW_GEN_MAIN_REPO bash "$SUB" --verify-manifest "$PART" >/dev/null 2>&1; then
  echo "PASS  ...and a sentinel-terminated one is accepted"; PASS=$((PASS + 1))
else
  echo "FAIL  a complete manifest was rejected"; FAIL=$((FAIL + 1))
fi
# structural: the cancel-on-abort trap is armed right after the hold and stood
# down only after a successful release
ARM_LINE="$(grep -n '^trap cancel_held_job EXIT$' "$SUB" | head -1 | cut -d: -f1)"
SB_LINE="$(grep -n 'JOBID="\$(slurm_submit_hold' "$SUB" | head -1 | cut -d: -f1)"
DISARM_LINE="$(grep -n 'ABORT_ACTIVE=0; HELD_JOBID=""; trap - EXIT INT TERM' "$SUB" | head -1 | cut -d: -f1)"
REL_LINE3="$(grep -n 'if ! slurm_release' "$SUB" | head -1 | cut -d: -f1)"
if [ -n "$ARM_LINE" ] && [ -n "$SB_LINE" ] && [ -n "$DISARM_LINE" ] && [ -n "$REL_LINE3" ] \
   && [ "$ARM_LINE" -lt "$SB_LINE" ] && [ "$REL_LINE3" -lt "$DISARM_LINE" ]; then
  echo "PASS  the cancel trap is armed BEFORE the submission and disarmed after the release"
  PASS=$((PASS + 1))
else
  echo "FAIL  the cancel trap is missing or misordered (arm@${ARM_LINE:-none} submit@${SB_LINE:-none} rel@${REL_LINE3:-none} disarm@${DISARM_LINE:-none})"
  FAIL=$((FAIL + 1))
fi

# --- the single-cell submitter's DRYRUN branch (review NIT 8) -----------------
echo
echo "--- single-cell submitter: DRYRUN ---"
WT_BEFORE="$(ls -1d "${MAIN_TREE}"/.measure_worktrees/*/ 2>/dev/null | wc -l)"
DOUT="$(DRYRUN=1 YAW_GEN_TEST_MODE=1 YAW_GEN_TEST_RECORD="$TRACE" bash "$SUB" ARM=C4L CELL=vctl STEP=40000 ROTATE_DEG=45 2>&1)"; rc=$?
WT_AFTER="$(ls -1d "${MAIN_TREE}"/.measure_worktrees/*/ 2>/dev/null | wc -l)"
if [ "$rc" -eq 0 ] && echo "$DOUT" | grep -q "nothing submitted" \
   && [ "$WT_BEFORE" = "$WT_AFTER" ]; then
  echo "PASS  DRYRUN prints the submission and prepares no worktree"; PASS=$((PASS + 1))
else
  echo "FAIL  the submitter's DRYRUN did work it should not (rc=${rc}, trees ${WT_BEFORE}->${WT_AFTER})"
  FAIL=$((FAIL + 1))
fi
DRY_NAME="$(echo "$DOUT" | sed -n 's/^DRYRUN job-name //p')"
WANT_NAME="$($PY "$VALIDATOR" jobname --arm C4L --cell vctl --step 40000 --seed 42 --k 8 --rotate-deg 45)"
if [ "$DRY_NAME" = "$WANT_NAME" ]; then
  echo "PASS  the single-cell submitter's job name is the validator's canonical one"; PASS=$((PASS + 1))
else
  echo "FAIL  submitter job name '${DRY_NAME}' != '${WANT_NAME}'"; FAIL=$((FAIL + 1))
fi
DRY_NAME90="$(DRYRUN=1 YAW_GEN_TEST_MODE=1 YAW_GEN_TEST_RECORD="$TRACE" bash "$SUB" ARM=C4L CELL=vctl STEP=40000 ROTATE_DEG=90 2>&1 \
              | sed -n 's/^DRYRUN job-name //p')"
if [ -n "$DRY_NAME90" ] && [ "$DRY_NAME" != "$DRY_NAME90" ]; then
  echo "PASS  the submitter gives the two C4L vctl angles distinct job names"; PASS=$((PASS + 1))
else
  echo "FAIL  the submitter still collides the two C4L vctl angles"; FAIL=$((FAIL + 1))
fi
# the intent manifest is published while the job is HELD, before the release
INT_LINE="$(grep -n 'if ! publish_intent; then' "$SUB" | head -1 | cut -d: -f1)"
REL_LINE2="$(grep -n 'if ! slurm_release' "$SUB" | head -1 | cut -d: -f1)"
if [ -n "$INT_LINE" ] && [ -n "$REL_LINE2" ] && [ "$INT_LINE" -lt "$REL_LINE2" ] \
   && grep -q 'mv -f "\$tmp" "\$INTENT"' "$SUB" \
   && grep -q 'verify_manifest_complete "\$tmp"' "$SUB"; then
  echo "PASS  the intent manifest is published (verified tmp -> atomic rename) BEFORE the release"
  PASS=$((PASS + 1))
else
  echo "FAIL  the intent manifest is still written after the job is released"; FAIL=$((FAIL + 1))
fi
if grep -q 'os.replace(tmp, out)' "$SCREEN"; then
  echo "PASS  the screenmeta sidecar is published by atomic replace"; PASS=$((PASS + 1))
else
  echo "FAIL  the screenmeta sidecar is written directly to its final path"; FAIL=$((FAIL + 1))
fi

echo
echo "=== yaw_gen screen kit guard tests: ${PASS} passed, ${FAIL} failed ==="
[ "$FAIL" -eq 0 ] || exit 1
echo "log: ${LOG}"
