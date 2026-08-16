#!/usr/bin/env bash
# ============================================================================
# yaw_aug_screen_guardtests.sh — guard-branch exercise for the exp_15 EVAL kit
# (yaw_aug_screen.sbatch, yaw_aug_screen_submit.sh, yaw_aug_submit_grid.sh,
# exp15_validate_cell.py, exp15_admit_ckpt.py).
#
# Derived from exp_14's yaw_gen_screen_guardtests.sh, under exp_15's own
# STRICT + per-case-ledger + union-coverage regime (yaw_aug_train_guardtests.sh).
#
# SAFETY — non-negotiable, and asserted rather than promised:
#   * NOTHING IS SUBMITTED. sbatch/scancel/scontrol are never executed: the kit's
#     test mode SIMULATES the Slurm interaction in-process, and section Z proves
#     this file contains no sbatch/scancel invocation at all.
#   * ⚠️ exp_15's TRAINING CHAIN IS LIVE while this suite runs. It touches no
#     Slurm state, no training file, and no exp_11/exp_14 folder.
#   * Every case runs against a mktemp output root; the real outputs_FLAC trees
#     are read only where a case says so out loud, and never written.
#   * No tracked file is mutated; a before/after `git status` snapshot asserts it.
#   * The store-wide campaign FREEZE is never touched and no worktree is ever
#     deleted. exp_14's suite exercised the store's deletion paths; those belong
#     to exp_11's helper, were reviewed there, and running them while a chain is
#     in flight would be reckless — so exp_15's suite has no thaw/delete case at
#     all. That is a deliberate de-scope, not an omission.
#
# ISOLATION IS THE DEFAULT: YAW_EVAL_MAIN_REPO is exported for the whole suite, so
# every invocation of the SUBMITTERS reads and writes under a temporary root
# unless a block explicitly opts out. The sbatch DRIVER cases are the documented
# opt-out — they are read-only DRYRUNs that need the real configs and pinned
# assets, and they submit nothing by construction.
#
# STRICT=1 turns a SKIP into a FAILURE. Cases that genuinely cannot run in a given
# environment use skip_env and are reconciled across transcripts by
# yaw_aug_union_coverage.py, which demands a PASS somewhere for every case name.
#
# Usage:  bash worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_screen_guardtests.sh
# Exit 0 = every case behaved as specified.
# ============================================================================
set -uo pipefail
cd "$(git -C "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" rev-parse --show-toplevel)" || exit 3

EXPDIR="worklog/worklog_yixun/exp_15_yaw_aug_claude"
EXP11="worklog/worklog_yixun/exp_11_fa_orbit_claude"   # READ-ONLY: store helper, VANL config
EXP14="worklog/worklog_yixun/exp_14_yaw_gen_claude"    # READ-ONLY: never touched
SCREEN="${EXPDIR}/yaw_aug_screen.sbatch"
SUB="${EXPDIR}/yaw_aug_screen_submit.sh"
GRID="${EXPDIR}/yaw_aug_submit_grid.sh"
VALIDATOR="${EXPDIR}/exp15_validate_cell.py"
ADMIT="${EXPDIR}/exp15_admit_ckpt.py"
UNION="${EXPDIR}/yaw_aug_union_coverage.py"
YAWAUG_CONFIG="${EXPDIR}/FLAC_AR_YAWAUG.json"
VANL_CONFIG="${EXP11}/FLAC_AR_VANCKPT.json"
LIVE_REGISTRY="${EXPDIR}/yaw_aug_launch_registry.json"
LIVE_CONTROL="${EXPDIR}/yaw_aug_control_admission.json"
PY=/n/fs/gatrdp/envs/flac/bin/python
# The driver imports exp_11's DINOv3 pin probe, which would otherwise drop a .pyc
# into that READ-ONLY folder. A real screen legitimately does (exp_14's did too);
# a guard suite must not, so bytecode writing is off for the whole run.
export PYTHONDONTWRITEBYTECODE=1
TS="$(date '+%Y-%m-%d_%H-%M-%S')"
LOG="${EXPDIR}/yaw_aug_${TS}_guardtests_${GUARD_TAG:-evalr1}.log"
LEDGER="${LOG%.log}.ledger"
HEAD_SHA="$(git rev-parse HEAD)"
FAKE_PIN="deadbeef00000000000000000000000000000042"     # 40 hex, not a commit

# THE SUITE NEVER TOUCHES CAMPAIGN STATE. It drives the kit through the test
# seams (honoured only in test mode, where no submission can happen), so the live
# command log and the live eval campaign pin are read-only to it — a RED run of
# exp_14's earlier suite wrote both and submitted four real jobs (all cancelled).
LIVE_PIN_FILE="${EXPDIR}/yaw_aug_screen_campaign_pin"
LIVE_CMDLOG="${EXPDIR}/yaw_aug_command.md"
LIVE_CMDLOG_SUM_AT_START="$(sha256sum "$LIVE_CMDLOG" 2>/dev/null | cut -d' ' -f1)"
LIVE_PIN_AT_START="ABSENT"
[ -f "$LIVE_PIN_FILE" ] && LIVE_PIN_AT_START="$(sha256sum "$LIVE_PIN_FILE" | cut -d' ' -f1)"
GUARD_SELF="$(readlink -f "${BASH_SOURCE[0]}")"

: > "$LEDGER"
exec > >(tee -a "$LOG") 2>&1
echo "=== exp_15 EVAL kit guard exercise — ${TS} — $(git rev-parse --short HEAD) ==="
for F in "$SCREEN" "$SUB" "$GRID" "$VALIDATOR" "$ADMIT" "$UNION" "$YAWAUG_CONFIG" "$VANL_CONFIG"; do
  [ -f "$F" ] || { echo "missing ${F} - abort"; exit 3; }
done

# Snapshot BEFORE the temp tree exists; the ledger/log above are this suite's own
# evidence files and are excluded by being untracked in EXPDIR (compared as a set,
# so their creation shows up identically in both snapshots).
# WORKTREE-vs-INDEX, not status-vs-HEAD (learned twice): `git status --porcelain`
# changes the moment ANY session commits, so it answered "did anyone commit
# during the run" rather than "did this suite modify a tracked file". Two
# sessions commit to this checkout, so that was a false alarm waiting to happen —
# and it fired, on my own mid-run commit. `git diff` (worktree vs index) is
# invariant under commits and moves only if a file's CONTENT changes, which is
# the question this case is actually asking.
TRACKED_PATHS=("$EXPDIR" "$EXP11" "$EXP14" src data/AR eval_FLAC.py)
tracked_snapshot() {
  git diff --name-only -- "${TRACKED_PATHS[@]}" | sort
  # ...plus the content digests of the files this suite actually drives, so a
  # modify-then-stage sequence could not hide in the diff above either.
  sha256sum "$SCREEN" "$SUB" "$GRID" "$VALIDATOR" "$ADMIT" "$GUARD_SELF" 2>/dev/null
}
TRACKED_BEFORE="$(tracked_snapshot)"
# ...plus a name/size/mtime listing of the two READ-ONLY neighbour folders. The
# git snapshot cannot see an untracked file appearing there, and those folders
# belong to other experiments (one of them to a session that is writing to it
# right now), so "did WE move anything" is the only question this suite may ask.
NEIGHBOURS_BEFORE="$(ls -li --time-style=+%s "$EXP11" "$EXP14" | sort)"

TMP="$(mktemp -d)"
trap 'suite_exit_trap' EXIT
PASS=0; FAIL=0
STRICT="${STRICT:-0}"
# SKIP_HEAVY=1 omits every case that spends real CPU: the driver's own DRYRUN
# deliberately runs the full checkpoint-admission gate before it exits (the E1
# reviewer's warning), and the admission cases import torch per invocation. On a
# loaded login node those dominate the runtime.
#
# This is NOT a way to get a green run cheaply. Skipped cases are recorded as
# skip_env, so yaw_aug_union_coverage.py still demands a PASS for each of them in
# some other transcript — a SKIP_HEAVY run alone can never satisfy coverage.
SKIP_HEAVY="${SKIP_HEAVY:-0}"
ledger() { printf '%s\t%s\n' "$1" "$2" >> "$LEDGER"; }

suite_exit_trap() {   # PRESERVES the script's own exit status; only worsens it
  local rc=$?
  if ! assert_campaign_untouched; then rc=1; fi
  rm -rf "$TMP"
  exit "$rc"
}
assert_campaign_untouched() {   # runs at suite EXIT, after every case
  local now pin_now="ABSENT"
  now="$(sha256sum "$LIVE_CMDLOG" 2>/dev/null | cut -d' ' -f1)"
  if [ "$now" != "${LIVE_CMDLOG_SUM_AT_START:-}" ]; then
    echo "FAIL  (suite exit) this suite MODIFIED the campaign command log"
    return 1
  fi
  [ -f "$LIVE_PIN_FILE" ] && pin_now="$(sha256sum "$LIVE_PIN_FILE" | cut -d' ' -f1)"
  if [ "$pin_now" != "${LIVE_PIN_AT_START:-ABSENT}" ]; then
    echo "FAIL  (suite exit) the eval campaign pin file changed"
    return 1
  fi
  echo "PASS  (suite exit) campaign command log and pin file are exactly as found"
  return 0
}

skip_case() {   # <name> <reason> — a SKIP is a FAILURE under STRICT=1
  if [ "$STRICT" = "1" ]; then
    echo "FAIL  ${1} (STRICT: ${2})"; ledger FAIL "$1"; FAIL=$((FAIL + 1))
  else
    echo "SKIP  ${1} (${2})"; ledger SKIP "$1"
  fi
}
skip_env() {    # <name> <reason> — the case CANNOT run in this environment.
  # STRICT does not fail these; yaw_aug_union_coverage.py still demands a PASS in
  # the OTHER environment, so they cannot quietly go uncovered either.
  echo "SKIP  ${1} (environment: ${2})"; ledger SKIP "$1"
}
case_run() {  # <name> <want-rc> <want-substring> -- <env...>   (runs the DRIVER)
  local name="$1" want_rc="$2" want_txt="$3"; shift 3; [ "$1" = "--" ] && shift
  local out rc
  if [ "$SKIP_HEAVY" = "1" ]; then
    skip_env "$name" "SKIP_HEAVY: the driver's DRYRUN runs the checkpoint gate"
    return 0
  fi
  out="$(env "$@" bash "$SCREEN" 2>&1)"; rc=$?
  if [ "$rc" -eq "$want_rc" ] && echo "$out" | grep -qF -- "$want_txt"; then
    echo "PASS  ${name}  (rc=${rc})"; ledger PASS "$name"; PASS=$((PASS + 1))
  else
    echo "FAIL  ${name}: want rc=${want_rc} + '${want_txt}', got rc=${rc}"
    echo "$out" | tail -6 | sed 's/^/        | /'; ledger FAIL "$name"; FAIL=$((FAIL + 1))
  fi
}
expect_cmd() {  # <name> <want-rc> <want-substring> -- <command...>
  local name="$1" want_rc="$2" want_txt="$3"; shift 3; [ "$1" = "--" ] && shift
  local out rc
  out="$("$@" 2>&1)"; rc=$?
  if [ "$rc" -eq "$want_rc" ] && echo "$out" | grep -qF -- "$want_txt"; then
    echo "PASS  ${name}  (rc=${rc})"; ledger PASS "$name"; PASS=$((PASS + 1))
  else
    echo "FAIL  ${name}: want rc=${want_rc} + '${want_txt}', got rc=${rc}"
    echo "$out" | tail -6 | sed 's/^/        | /'; ledger FAIL "$name"; FAIL=$((FAIL + 1))
  fi
}
heavy_cmd() {  # like expect_cmd, for cases that import torch per invocation
  if [ "$SKIP_HEAVY" = "1" ]; then
    skip_env "$1" "SKIP_HEAVY: imports torch per invocation"
    return 0
  fi
  expect_cmd "$@"
}

check() {  # <name> <condition-rc> — for grep-style structural assertions
  if [ "$2" -eq 0 ]; then echo "PASS  $1"; ledger PASS "$1"; PASS=$((PASS + 1))
  else echo "FAIL  $1"; ledger FAIL "$1"; FAIL=$((FAIL + 1)); fi
}
eq_check() {  # <name> <got> <want>
  if [ "$2" = "$3" ]; then echo "PASS  $1"; ledger PASS "$1"; PASS=$((PASS + 1))
  else echo "FAIL  $1: got '$2' want '$3'"; ledger FAIL "$1"; FAIL=$((FAIL + 1)); fi
}
argv_of() {
  [ "$SKIP_HEAVY" = "1" ] && { printf ''; return 0; }
  env "$@" bash "$SCREEN" 2>&1 | sed -n 's/^python eval_FLAC.py //p' | head -1
}
argv_has() {  # <name> <argv> <needle>
  [ "$SKIP_HEAVY" = "1" ] && { skip_env "$1" "SKIP_HEAVY: needs a driver DRYRUN"; return 0; }
  case "$2" in
    *"$3"*) echo "PASS  $1"; ledger PASS "$1"; PASS=$((PASS + 1)) ;;
    *) echo "FAIL  $1: '$3' absent from"; echo "        | $2"; ledger FAIL "$1"; FAIL=$((FAIL + 1)) ;;
  esac
}
argv_lacks() {  # <name> <argv> <needle>  — an ABSENCE is a contract too
  [ "$SKIP_HEAVY" = "1" ] && { skip_env "$1" "SKIP_HEAVY: needs a driver DRYRUN"; return 0; }
  case "$2" in
    *"$3"*) echo "FAIL  $1: '$3' present in"; echo "        | $2"; ledger FAIL "$1"; FAIL=$((FAIL + 1)) ;;
    *) echo "PASS  $1"; ledger PASS "$1"; PASS=$((PASS + 1)) ;;
  esac
}

# --- synthetic checkpoints, output roots and admission records ---------------
# Each pathology gets its OWN output root: exp_15 pins STEP=40000, so a duplicate
# or EMA-less 40k file in the good root would break every good case for that arm.
#
#   GOOD      both arms at 40000, each with its own config, EMA mirror, launch
#             manifest, and a matching synthetic admission record
#   WRONGSTEP YAWAUG's tree holding a step-37500-inside / 40000-named checkpoint
#   BROKENEMA YAWAUG at 40000 whose EMA family does not mirror the online DiT
#   WRONGCFG  YAWAUG's tree holding a checkpoint trained WITHOUT yaw_aug
#   DUP       two epoch=*-step=40000.ckpt files for YAWAUG
GOOD="${TMP}/good"; WRONGSTEP="${TMP}/wrongstep"; BROKENEMA="${TMP}/brokenema"
WRONGCFG="${TMP}/wrongcfg"; DUP="${TMP}/dup"; EMPTY="${TMP}/empty"
mkdir -p "$EMPTY"
$PY - "$EXPDIR" "$YAWAUG_CONFIG" "$VANL_CONFIG" "$GOOD" "$WRONGSTEP" "$BROKENEMA" \
     "$WRONGCFG" "$DUP" <<'PY'
import hashlib, json, os, sys, torch
expdir, yaw_cfg_p, vanl_cfg_p, good, wrongstep, brokenema, wrongcfg, dup = sys.argv[1:9]
sys.path.insert(0, expdir)
import yaw_aug_record_control as rc

CFG_PATH = {"YAWAUG": yaw_cfg_p, "VANL": vanl_cfg_p}
TREE = {"YAWAUG": ("exp15_YAWAUG", "FLAC_exp15_YAWAUG", "exp15_YAWAUG"),
        "VANL": ("exp11_VANL", "FLAC_exp11_VANL", "exp11_VANL")}


def write_ckpt(root, arm, cfg_obj, step=40000, epoch=8, ema="mirror", tag=""):
    sd = {"diffusion.model.a": torch.zeros(2), "diffusion.model.b": torch.zeros(3),
          "diffusion.pretransform.z": torch.zeros(1)}
    if ema == "mirror":
        sd["diffusion_ema.ema_model.a"] = torch.zeros(2)
        sd["diffusion_ema.ema_model.b"] = torch.zeros(3)
    elif ema == "broken":            # same suffixes, wrong shape
        sd["diffusion_ema.ema_model.a"] = torch.zeros(2)
        sd["diffusion_ema.ema_model.b"] = torch.zeros(9)
    sd["diffusion_ema.initted"] = torch.zeros(1)
    d = os.path.join(root, *TREE[arm], "checkpoints")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"epoch={epoch}-step=40000.ckpt")
    torch.save({"global_step": step, "epoch": epoch, "model_config": cfg_obj,
                "state_dict": sd, "optimizer_states": [{"state": {}}],
                "lr_schedulers": [{"last_epoch": step}]}, path)
    return path


def manifest(root, arm):
    d = os.path.join(root, TREE[arm][0])
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, "launch_manifest.txt")
    with open(p, "w") as fh:
        fh.write(f"job 9000001 host synthetic mode INITIAL launch_uuid uuid-{arm}\n")
        fh.write(f"arm {arm} rung 8x8 micro 8 ngpu 8 max_steps 40000 ckpt_every 2500\n")
    return p


def facts(ckpt, cfg_bytes, cfg_obj, man):
    obj, sha, ident = rc.snapshot_checkpoint(ckpt)
    ema = rc.summarize_ema(obj["state_dict"])
    return {
        "sha256": sha, "bytes": ident["bytes"],
        "cfg_sha": hashlib.sha256(cfg_bytes).hexdigest(),
        "cfg_canon": rc.canonical_sha256(cfg_obj),
        "man_sha": hashlib.sha256(open(man, "rb").read()).hexdigest(),
        **ema,
    }


yaw_bytes = open(yaw_cfg_p, "rb").read(); yaw_cfg = json.loads(yaw_bytes)
vanl_bytes = open(vanl_cfg_p, "rb").read(); vanl_cfg = json.loads(vanl_bytes)

# --- the GOOD root: both arms, plus the two admission records that describe it -
ck_y = write_ckpt(good, "YAWAUG", yaw_cfg)
ck_v = write_ckpt(good, "VANL", vanl_cfg)
man_y = manifest(good, "YAWAUG"); man_v = manifest(good, "VANL")
fy = facts(ck_y, yaw_bytes, yaw_cfg, man_y)
fv = facts(ck_v, vanl_bytes, vanl_cfg, man_v)

registry = {"arms": {"YAWAUG": {
                "config_sha256": fy["cfg_sha"], "final_ckpt_sha256": fy["sha256"],
                "final_step": 40000, "training_seed": 42, "rung": "8x8",
                "mode": "INITIAL", "manifest_sha256": fy["man_sha"],
                "manifest_path": man_y, "save_dir": "outputs_FLAC/exp15_YAWAUG",
                "yaw_aug": {"enabled": True, "img_w": 512, "seed": 42}}},
            "legs": {"YAWAUG": [{
                "step": 40000, "ckpt_sha256": fy["sha256"], "ckpt_bytes": fy["bytes"],
                "ckpt_path": ck_y, "chain": True,
                "audit": {"embedded_config_canonical_sha256": fy["cfg_canon"],
                          "ema_prefix": fy["ema_prefix"],
                          "ema_inventory_sha256": fy["ema_inventory_sha256"],
                          "ema_key_count": fy["ema_key_count"],
                          "online_model_key_count": fy["online_model_key_count"]}}]}}
control = {"_meta": {"expect_step": 40000, "experiment": "exp_15"},
           "checkpoint": {"path": ck_v, "sha256": fv["sha256"], "bytes": fv["bytes"],
                          "global_step": 40000, "epoch": 8,
                          "embedded_config_canonical_sha256": fv["cfg_canon"],
                          "ema_prefix": fv["ema_prefix"],
                          "ema_inventory_sha256": fv["ema_inventory_sha256"],
                          "ema_key_count": fv["ema_key_count"],
                          "online_model_key_count": fv["online_model_key_count"]},
           "config": {"path": vanl_cfg_p, "sha256": fv["cfg_sha"],
                      "canonical_sha256": fv["cfg_canon"]},
           "exp_11_cross_references": {"manifest_sha256": fv["man_sha"]},
           "checks": {"global_step_equals_expected": True}}
json.dump(registry, open(os.path.join(good, "registry.json"), "w"), indent=2)
json.dump(control, open(os.path.join(good, "control.json"), "w"), indent=2)

# --- the pathology roots. Each carries a COPY of the good admission records, so
# --- the failing check is the pathology and never a missing expectation.
for root, kwargs in ((wrongstep, dict(step=37500)),
                     (brokenema, dict(ema="broken")),
                     (wrongcfg, dict(cfg_obj=vanl_cfg))):
    kw = dict(kwargs)
    cfg_obj = kw.pop("cfg_obj", yaw_cfg)
    write_ckpt(root, "YAWAUG", cfg_obj, **kw)
    manifest(root, "YAWAUG")
    json.dump(registry, open(os.path.join(root, "registry.json"), "w"), indent=2)
    json.dump(control, open(os.path.join(root, "control.json"), "w"), indent=2)

write_ckpt(dup, "YAWAUG", yaw_cfg, epoch=8)
os.link(os.path.join(dup, *TREE["YAWAUG"], "checkpoints", "epoch=8-step=40000.ckpt"),
        os.path.join(dup, *TREE["YAWAUG"], "checkpoints", "epoch=9-step=40000.ckpt"))
manifest(dup, "YAWAUG")
json.dump(registry, open(os.path.join(dup, "registry.json"), "w"), indent=2)
json.dump(control, open(os.path.join(dup, "control.json"), "w"), indent=2)
print("synthetic roots written")
PY
[ -f "${GOOD}/registry.json" ] || { echo "fixture build FAILED - abort"; exit 3; }

# A registry with the arm entry present but NO final checkpoint — the state the
# LIVE registry is in today, reproduced synthetically so the case is exercised
# even after the chain finishes and the live file changes.
$PY - "${GOOD}/registry.json" "${TMP}/registry_unfinished.json" <<'PY'
import json, sys
reg = json.load(open(sys.argv[1]))
reg["arms"]["YAWAUG"]["final_ckpt_sha256"] = None
reg["arms"]["YAWAUG"]["final_step"] = None
reg["legs"]["YAWAUG"] = [dict(reg["legs"]["YAWAUG"][0], step=12500)]
json.dump(reg, open(sys.argv[2], "w"), indent=2)
PY
# ...and one whose final sha simply does not match the file on disk.
$PY - "${GOOD}/registry.json" "${TMP}/registry_wrongsha.json" <<'PY'
import json, sys
reg = json.load(open(sys.argv[1]))
bad = "0" * 64
reg["arms"]["YAWAUG"]["final_ckpt_sha256"] = bad
reg["legs"]["YAWAUG"][0]["ckpt_sha256"] = bad
json.dump(reg, open(sys.argv[2], "w"), indent=2)
PY
# ...and one whose arm entry and leg entry disagree about which file is final.
$PY - "${GOOD}/registry.json" "${TMP}/registry_split.json" <<'PY'
import json, sys
reg = json.load(open(sys.argv[1]))
reg["arms"]["YAWAUG"]["final_ckpt_sha256"] = "1" * 64
json.dump(reg, open(sys.argv[2], "w"), indent=2)
PY
# ...and one that records the right bytes at the WRONG location: identity of the
# content is not provenance of the run.
$PY - "${GOOD}/registry.json" "${TMP}/registry_wrongpath.json" <<'PY'
import json, sys
reg = json.load(open(sys.argv[1]))
reg["legs"]["YAWAUG"][0]["ckpt_path"] = "/somewhere/else/epoch=8-step=40000.ckpt"
json.dump(reg, open(sys.argv[2], "w"), indent=2)
PY

GOODREG="${GOOD}/registry.json"; GOODCTL="${GOOD}/control.json"
BASE=(DRYRUN=1 "EXPECT_SHA=${HEAD_SHA}" "OUTPUT_ROOT=${GOOD}"
      "YAW_EVAL_REGISTRY=${GOODREG}" "YAW_EVAL_CONTROL=${GOODCTL}")
root_env() {   # <root> — select another synthetic output root (last env wins)
  printf '%s\n' "OUTPUT_ROOT=$1" "YAW_EVAL_REGISTRY=$1/registry.json" \
                "YAW_EVAL_CONTROL=$1/control.json"
}

# ISOLATION BY DEFAULT for the SUBMITTERS (the driver cases below are the
# documented opt-out and set OUTPUT_ROOT explicitly instead).
FAKE_REPO="${TMP}/fake_main_repo"
FAKE_EXP="${FAKE_REPO}/worklog/worklog_yixun/exp_15_yaw_aug_claude"
mkdir -p "$FAKE_EXP"
git -c init.defaultBranch=main init -q "$FAKE_REPO"
git -C "$FAKE_REPO" -c user.email=guard@local -c user.name=guard commit -q --allow-empty -m "isolation root"
FAKE_HEAD="$(git -C "$FAKE_REPO" rev-parse HEAD)"
cp "$VALIDATOR" "$ADMIT" "$GRID" "$SUB" "$SCREEN" "$YAWAUG_CONFIG" "$FAKE_EXP/"
cp "$LIVE_REGISTRY" "$LIVE_CONTROL" "$FAKE_EXP/"
export YAW_EVAL_MAIN_REPO="$FAKE_REPO"
TEST_PIN_FILE="${TMP}/campaign_pin"
TEST_CMDLOG="${TMP}/command.md"
TEST_TRACE="${TMP}/trace.txt"
TEST_QUEUE="${TMP}/queue.txt"
TEST_INTENT_DIR="${TMP}/intents"; mkdir -p "$TEST_INTENT_DIR"

echo
echo "--- 0. static checks + the suite's own safety rails ---"
bash -n "$SCREEN"; check "yaw_aug_screen.sbatch parses (bash -n)" $?
bash -n "$SUB";    check "yaw_aug_screen_submit.sh parses (bash -n)" $?
bash -n "$GRID";   check "yaw_aug_submit_grid.sh parses (bash -n)" $?
$PY -m py_compile "$VALIDATOR"; check "exp15_validate_cell.py compiles" $?
$PY -m py_compile "$ADMIT";     check "exp15_admit_ckpt.py compiles" $?
# The eval kit must not be able to disturb the LIVE TRAINING CHAIN: no EXECUTABLE
# line of it may name a file in the training closure that the chain's start gate
# diffs. Comments may (yaw_aug_screen_submit.sh says out loud that it is not
# yaw_aug_submit.sh), so comment and blank lines are stripped first — a check that
# forbade the mention would push the disambiguation out of the file that needs it.
! grep -hvE '^[[:space:]]*(#|$)' "$SCREEN" "$SUB" "$GRID" "$VALIDATOR" "$ADMIT" \
  | grep -qE '(yaw_aug_train\.sbatch|yaw_aug_submit\.sh|yaw_aug_chain_state\.py|yaw_aug_chain_preflight\.py|yaw_aug_rate_gate\.py|yaw_aug_pin_allowlist\.txt)'
check "the eval kit never EXECUTES against the training kit's live files" $?
# ...and the one training-closure file it does touch, it only IMPORTS.
grep -q 'import yaw_aug_record_control' "$ADMIT"
check "  ... and touches yaw_aug_record_control.py by import only" $?
grep -q 'yaw_aug_record_control' "$ADMIT"
check "the admission gate IMPORTS the recorder's primitives (never re-implements)" $?
! grep -qE '^[[:space:]]*(sbatch|scancel|scontrol) ' "$GUARD_SELF"
check "this suite never invokes sbatch/scancel/scontrol" $?
# Best-effort lint: no case may invoke the REAL submitters in live mode. NOT an
# invariant — it is a substring scan and a case written in another shape would
# slip past it. The SOUND mechanism is the exported YAW_EVAL_MAIN_REPO, which
# isolates every submitter case by default; this lint catches the obvious shape
# cheaply and names the offending line.
$PY - "$GUARD_SELF" <<'PY'
import sys
src = open(sys.argv[1]).read().split("\n")
# An env ARRAY counts as a mode declaration for every line that expands it, so a
# block that builds its environment once is not flagged line by line.
safe_arrays = {name for name in ("GRID_ENV",)
               if any(f"{name}=(" in l and "YAW_EVAL_TEST_MODE=1" in "\n".join(src) for l in src)}
bad = []
for i, line in enumerate(src):
    if 'bash "$GRID"' in line or 'bash "$SUB"' in line:
        if "--verify-manifest" in line or "if 'bash" in line:
            continue
        if any(f'"${{{a}[@]}}"' in line for a in safe_arrays):
            continue
        # A case whose SUBJECT is the live-mode refusal has to invoke live mode.
        # It is exempt only with this marker, and only when the surrounding case
        # asserts a REFUSAL (rc 2) — so the exemption cannot be used to smuggle in
        # a case that expects a live run to succeed.
        if "LINT-OK-live-refusal" in "\n".join(src[max(0, i - 3):i + 2]):
            window = "\n".join(src[max(0, i - 3):i + 2])
            if 'expect_cmd' in window and '" 2 ' in window:
                continue
        window = "\n".join(src[max(0, i - 8):i + 1])
        if "YAW_EVAL_TEST_MODE=1" not in window and "DRYRUN=1" not in window:
            bad.append(f"{i + 1}: {line.strip()[:78]}")
if bad:
    print("live-capable invocations of the REAL kit:")
    print("\n".join("  " + b for b in bad))
    sys.exit(1)
PY
check "lint: no guard case can reach the submitters' live path (best-effort scan)" $?
# ...and the array the lint trusts really does declare test mode.
grep -q 'GRID_ENV=(YAW_EVAL_TEST_MODE=1' "$GUARD_SELF"
check "  ... and the env array it exempts really does set YAW_EVAL_TEST_MODE=1" $?
# The MECHANISM itself: with only the suite's default environment, an invocation
# of the REAL wave submitter reads the TEMP root, not the repository. TEST_MODE
# without DRYRUN is used deliberately — it reaches the campaign-pin refusal, which
# is the first message that NAMES the tree the wave would have used.
ISO="$(env YAW_EVAL_TEST_MODE=1 bash "$GRID" WAVE=vctl 2>&1 | grep -m1 'campaign pin FILE')"
case "$ISO" in
  *"${FAKE_REPO}"*) echo "PASS  isolation is the DEFAULT: the submitters read the temp root"
                    ledger PASS "isolation is the DEFAULT: the submitters read the temp root"
                    PASS=$((PASS + 1)) ;;
  *) echo "FAIL  a default-environment invocation did not land in the temp root: ${ISO}"
     ledger FAIL "isolation is the DEFAULT: the submitters read the temp root"; FAIL=$((FAIL + 1)) ;;
esac

echo
echo "--- A. THE REGISTERED 42-CELL GRID (one definition, exhaustively) ---"
eq_check "the grid is exactly 42 cells" "$($PY "$VALIDATOR" grid | wc -l)" "42"
eq_check "  ... 20 tbl" "$($PY "$VALIDATOR" grid --wave tbl | wc -l)" "20"
eq_check "  ... 20 rrob" "$($PY "$VALIDATOR" grid --wave rrob | wc -l)" "20"
eq_check "  ... 2 vctl" "$($PY "$VALIDATOR" grid --wave vctl | wc -l)" "2"
eq_check "the vctl block is exactly VANL@90 and YAWAUG@90" \
  "$($PY "$VALIDATOR" grid --wave vctl | awk '{print $1"@"$6}' | sort | tr '\n' ' ')" \
  "VANL@90 YAWAUG@90 "
eq_check "eval names are injective over the grid" \
  "$($PY - <<'PY'
import sys; sys.path.insert(0, "worklog/worklog_yixun/exp_15_yaw_aug_claude")
import exp15_validate_cell as V
g = V.expected_grid()
print(len({V.eval_name(c) for c in g}))
PY
)" "42"
eq_check "job names are injective over the grid" \
  "$($PY - <<'PY'
import sys; sys.path.insert(0, "worklog/worklog_yixun/exp_15_yaw_aug_claude")
import exp15_validate_cell as V
print(len({V.job_name(c) for c in V.expected_grid()}))
PY
)" "42"
expect_cmd "every eval name round-trips through parse_eval_name" 0 "ROUNDTRIP OK" -- \
  $PY - <<'PY'
import sys; sys.path.insert(0, "worklog/worklog_yixun/exp_15_yaw_aug_claude")
import exp15_validate_cell as V
for c in V.expected_grid():
    assert V.parse_eval_name(V.eval_name(c)) == c, c
print("ROUNDTRIP OK")
PY
expect_cmd "an UNREGISTERED cell name is refused, not parsed" 0 "REFUSED" -- \
  $PY - <<'PY'
import sys; sys.path.insert(0, "worklog/worklog_yixun/exp_15_yaw_aug_claude")
import exp15_validate_cell as V
for bad in ("exp15_C4L_tbl_S40000_s42_K8",          # not an exp_15 arm
            "exp15_YAWAUG_zref_S40000_s42_K8",      # exp_14's cell type
            "exp15_YAWAUG_tbl_S40000_s47_K8",       # seed outside 42-46
            "exp15_YAWAUG_tbl_S40000_s42_K4",       # K outside {1,8}
            "exp15_YAWAUG_tbl_S37500_s42_K8",       # not the endpoint
            "exp15_VANL_vctl_rot45_S40000_s42_K8",  # unregistered angle
            "exp15_YAWAUG_vctl_rot90_S40000_s43_K8"):   # vctl is s42 only
    try:
        V.parse_eval_name(bad)
    except ValueError:
        continue
    raise SystemExit(f"ACCEPTED an unregistered name: {bad}")
print("REFUSED")
PY
expect_cmd "validate_cell() refuses an unregistered cell before touching disk" 0 "REFUSED" -- \
  $PY - <<'PY'
import sys; sys.path.insert(0, "worklog/worklog_yixun/exp_15_yaw_aug_claude")
import exp15_validate_cell as V
try:
    V.validate_cell("/nonexistent", V.Cell("YAWAUG", "tbl", 40000, 47, 8, None))
except ValueError as e:
    assert "not registered" in str(e), e
    print("REFUSED")
else:
    raise SystemExit("an unregistered cell was validated")
PY

echo
echo "--- B. PER-CLASS PROTOCOL ARGV (plan §4.2, checked on ALL 42 cells) ---"
expect_cmd "every cell's protocol argv is the plan's literal §4.2 string" 0 "ARGV OK 42" -- \
  $PY - <<'PY'
import sys; sys.path.insert(0, "worklog/worklog_yixun/exp_15_yaw_aug_claude")
import exp15_validate_cell as V
COMMON = "--cond-method vanilla --frame-avg-angles 0,90,180,270 --cond-autocast bf16"
n = 0
for c in V.expected_grid():
    got = " ".join(V.eval_argv_tail(c))
    if c.cell == "tbl":
        want = f"{COMMON} --rotate-mode fixed --rotate-deg 0"
    elif c.cell == "rrob":
        want = f"{COMMON} --rotate-mode random --rotate-seed {c.seed} --rotate-deg 0"
    else:
        want = f"{COMMON} --rotate-mode fixed --rotate-deg 90"
    assert got == want, f"{c}: {got!r} != {want!r}"
    n += 1
print("ARGV OK", n)
PY
expect_cmd "a T cell can NEVER carry --rotate-seed" 0 "NO SEED ON T" -- \
  $PY - <<'PY'
import sys; sys.path.insert(0, "worklog/worklog_yixun/exp_15_yaw_aug_claude")
import exp15_validate_cell as V
for c in V.expected_grid():
    if c.cell in ("tbl", "vctl"):
        assert "--rotate-seed" not in V.eval_argv_tail(c), c
print("NO SEED ON T")
PY
expect_cmd "an R cell can NEVER carry a non-zero --rotate-deg" 0 "NO ANGLE ON R" -- \
  $PY - <<'PY'
import sys; sys.path.insert(0, "worklog/worklog_yixun/exp_15_yaw_aug_claude")
import exp15_validate_cell as V
for c in V.expected_grid():
    if c.cell == "rrob":
        a = V.eval_argv_tail(c)
        assert a[a.index("--rotate-deg") + 1] == "0", c
print("NO ANGLE ON R")
PY
expect_cmd "the rotation seed of an R cell IS its eval seed" 0 "SEED==SEED" -- \
  $PY - <<'PY'
import sys; sys.path.insert(0, "worklog/worklog_yixun/exp_15_yaw_aug_claude")
import exp15_validate_cell as V
for c in V.expected_grid():
    if c.cell == "rrob":
        a = V.eval_argv_tail(c)
        assert int(a[a.index("--rotate-seed") + 1]) == c.seed, c
print("SEED==SEED")
PY
heavy_cmd "the metrics path mirrors eval_FLAC.build_output_paths on all 42 cells" 0 "MIRROR OK 42" -- \
  $PY - <<'PY'
import sys, os
sys.path.insert(0, os.getcwd())
sys.path.insert(0, "worklog/worklog_yixun/exp_15_yaw_aug_claude")
import exp15_validate_cell as V
from eval_FLAC import build_output_paths, rotation_suffix, canonical_stream_hash
ck = "/o/x/epoch=8-step=40000.ckpt"
n = 0
for c in V.expected_grid():
    mode, deg, rseed = V.rotation_expectation(c)
    want = build_output_paths(ck, V.STEPS, V.CFG_SCALE, V.eval_name(c),
                              cond_method="vanilla",
                              rotate_deg=0.0 if deg is None else float(deg),
                              n_angles=4, rotate_mode=mode, rotate_seed=rseed)["metrics"]
    assert V.metrics_path(ck, c) == want, (c, want, V.metrics_path(ck, c))
    assert V.rotation_suffix(c) == rotation_suffix(
        mode, 0.0 if deg is None else float(deg), rseed), c
    n += 1
t = [[0, "a/b.wav", ["x"], 512], [1, "c/d.wav", [], 512]]
assert canonical_stream_hash(t) == V.canonical_stream_hash(t)
print("MIRROR OK", n)
PY
eq_check "campaign constants are exp_14's (comparability of the external check)" \
  "$($PY -c "import sys; sys.path.insert(0,'worklog/worklog_yixun/exp_15_yaw_aug_claude');
import exp15_validate_cell as V
print(V.BATCH_SIZE, V.NUM_WORKERS, V.EXPECTED_COUNT, V.EXPECTED_SCENES, V.COND_AUTOCAST, V.IMG_W)")" \
  "64 4 6337 10 bf16 512"

echo
echo "--- C. the DRIVER's parameter gates (real driver, DRYRUN, synthetic roots) ---"
case_run "missing ARM"          2 "ARM must be exported"   -- "${BASE[@]}" CELL=tbl
case_run "missing CELL"         2 "CELL must be exported"  -- "${BASE[@]}" ARM=YAWAUG STEP=40000
case_run "missing STEP"         2 "STEP must be exported"  -- "${BASE[@]}" ARM=YAWAUG CELL=tbl STEP=
case_run "missing EXPECT_SHA"   2 "EXPECT_SHA" -- DRYRUN=1 "OUTPUT_ROOT=${GOOD}" ARM=YAWAUG CELL=tbl STEP=40000
case_run "an exp_14 arm is refused"   2 "not registered for exp_15" -- "${BASE[@]}" ARM=C4L CELL=tbl STEP=40000
case_run "an exp_14 cell type is refused" 2 "must be tbl" -- "${BASE[@]}" ARM=YAWAUG CELL=zref STEP=40000
case_run "STEP 12500 is unregistered" 2 "STEP=40000 endpoint only" -- "${BASE[@]}" ARM=YAWAUG CELL=tbl STEP=12500
case_run "STEP 42500 is unregistered" 2 "STEP=40000 endpoint only" -- "${BASE[@]}" ARM=YAWAUG CELL=tbl STEP=42500
case_run "bad K"                2 "must be 1 or 8" -- "${BASE[@]}" ARM=YAWAUG CELL=tbl STEP=40000 K=4
case_run "tbl refuses seed 47"  2 "eval seeds 42-46" -- "${BASE[@]}" ARM=YAWAUG CELL=tbl STEP=40000 SEED=47
case_run "rrob refuses seed 41" 2 "eval seeds 42-46" -- "${BASE[@]}" ARM=VANL CELL=rrob STEP=40000 SEED=41
case_run "tbl refuses ROTATE_DEG"  2 "takes no ROTATE_DEG" -- "${BASE[@]}" ARM=YAWAUG CELL=tbl STEP=40000 ROTATE_DEG=90
case_run "rrob refuses ROTATE_DEG" 2 "takes no ROTATE_DEG" -- "${BASE[@]}" ARM=YAWAUG CELL=rrob STEP=40000 ROTATE_DEG=90
case_run "an EVAL_ORBIT leftover is refused" 2 "not an exp_15 parameter" \
  -- "${BASE[@]}" ARM=YAWAUG CELL=tbl STEP=40000 EVAL_ORBIT=4
case_run "vctl needs an angle"  2 "needs ROTATE_DEG" -- "${BASE[@]}" ARM=VANL CELL=vctl STEP=40000
case_run "vctl is seed 42 only" 2 "seed 42 by contract" -- "${BASE[@]}" ARM=VANL CELL=vctl STEP=40000 SEED=43 ROTATE_DEG=90
case_run "vctl is K=8 only"     2 "K=8 by contract" -- "${BASE[@]}" ARM=VANL CELL=vctl STEP=40000 K=1 ROTATE_DEG=90
case_run "vctl refuses 45 degrees (exp_14's angle, not ours)" 2 "not a registered vctl angle" \
  -- "${BASE[@]}" ARM=VANL CELL=vctl STEP=40000 ROTATE_DEG=45
case_run "no checkpoint tree at all" 2 "exactly 1 checkpoint" \
  -- "${BASE[@]}" $(root_env "$EMPTY") ARM=YAWAUG CELL=tbl STEP=40000
case_run "two checkpoints at one step" 2 "exactly 1 checkpoint" \
  -- "${BASE[@]}" $(root_env "$DUP") ARM=YAWAUG CELL=tbl STEP=40000
case_run "real mode needs sbatch" 2 "must run under sbatch" \
  -- "EXPECT_SHA=${HEAD_SHA}" "OUTPUT_ROOT=${GOOD}" "YAW_EVAL_REGISTRY=${GOODREG}" \
     "YAW_EVAL_CONTROL=${GOODCTL}" ARM=YAWAUG CELL=tbl STEP=40000
case_run "a synthetic admission source is refused for the PRODUCTION root" 2 \
  "may not override the committed admission records" \
  -- DRYRUN=1 "EXPECT_SHA=${HEAD_SHA}" "YAW_EVAL_REGISTRY=${GOODREG}" ARM=YAWAUG CELL=tbl STEP=40000

echo
echo "--- D. CHECKPOINT ADMISSION (gate G4): every pathology, by name ---"
admit() {  # <root> <arm> [extra args...]
  local root="$1" arm="$2"; shift 2
  local ck; ck="$(ls "${root}"/exp1*_"${arm}"/FLAC_*/*/checkpoints/epoch=*-step=40000.ckpt 2>/dev/null | head -1)"
  local cfg="$YAWAUG_CONFIG"; [ "$arm" = "VANL" ] && cfg="$VANL_CONFIG"
  local man; man="$(ls "${root}"/exp1*_"${arm}"/launch_manifest.txt 2>/dev/null | head -1)"
  $PY "$ADMIT" verify --arm "$arm" --ckpt "$ck" --config "$cfg" \
      --launch-manifest "$man" --main-repo / \
      --registry "${root}/registry.json" --control "${root}/control.json" "$@"
}
heavy_cmd "a well-formed YAWAUG 40k checkpoint is ADMITTED" 0 "ADMITTED YAWAUG" -- admit "$GOOD" YAWAUG
heavy_cmd "a well-formed VANL 40k checkpoint is ADMITTED" 0 "ADMITTED VANL" -- admit "$GOOD" VANL
heavy_cmd "wrong embedded step is REFUSED" 1 "not the pre-registered endpoint 40000" -- admit "$WRONGSTEP" YAWAUG
heavy_cmd "a broken EMA<->online mirror is REFUSED" 1 "EMA/online mirror check FAILED" -- admit "$BROKENEMA" YAWAUG
expect_cmd "a checkpoint trained WITHOUT yaw_aug is refused as YAWAUG" 1 \
  "embedded training.yaw_aug=None" -- admit "$WRONGCFG" YAWAUG
expect_cmd "  ... and its embedded config mismatch is named too" 1 \
  "canonical bytes" -- admit "$WRONGCFG" YAWAUG
heavy_cmd "a recorded sha that does not match the file is REFUSED" 1 "checkpoint sha256: recomputed" -- \
  admit "$GOOD" YAWAUG --registry "${TMP}/registry_wrongsha.json"
# A self-contradictory registry is an "expectation unavailable" condition (rc 2),
# not a checkpoint refusal: nothing is knowable about the file until the record
# agrees with itself, and reporting it as a bad checkpoint would misname the fault.
heavy_cmd "a registry whose arm and leg disagree is REFUSED" 2 "describes two different checkpoints" -- \
  admit "$GOOD" YAWAUG --registry "${TMP}/registry_split.json"
heavy_cmd "a tampered launch manifest is REFUSED" 1 "manifest sha256" -- \
  admit "$GOOD" YAWAUG --launch-manifest "$YAWAUG_CONFIG"
heavy_cmd "a checkpoint outside the recorded location is REFUSED" 1 "is not the registered" -- \
  admit "$GOOD" YAWAUG --registry "${TMP}/registry_wrongpath.json"
# THE STATE THE LIVE REGISTRY IS IN TODAY, reproduced synthetically so the case
# survives the chain finishing.
expect_cmd "an UNFINISHED chain registry refuses admission, by name" 2 \
  "has NOT recorded its final checkpoint yet" -- admit "$GOOD" YAWAUG --registry "${TMP}/registry_unfinished.json"
heavy_cmd "  ... and names the legs recorded so far" 2 "legs recorded so far: [12500]" -- \
  admit "$GOOD" YAWAUG --registry "${TMP}/registry_unfinished.json"
heavy_cmd "  ... and the driver dies on it rather than evaluating" 2 "ADMISSION UNAVAILABLE" -- \
  env "${BASE[@]}" "YAW_EVAL_REGISTRY=${TMP}/registry_unfinished.json" ARM=YAWAUG CELL=tbl STEP=40000 \
      bash "$SCREEN"
# ...and the LIVE record, as it actually stands right now. The two outcomes are
# mutually exclusive BY DESIGN rather than by environment, so they share ONE case
# name: registering them separately would leave whichever branch is currently
# impossible permanently UNCOVERED in the union check, which is a coverage claim
# nobody could ever satisfy.
LIVE_EXPECT="$($PY "$VALIDATOR" expect --arm YAWAUG 2>&1)"; LIVE_RC=$?
if [ "$LIVE_RC" -eq 0 ]; then
  echo "  (live registry state: chain COMPLETE — YAWAUG has a final checkpoint)"
  printf '%s\n' "$LIVE_EXPECT" | grep -q '"sha256"'
else
  echo "  (live registry state: chain RUNNING — YAWAUG has no final checkpoint yet)"
  printf '%s\n' "$LIVE_EXPECT" | grep -q "has NOT recorded its final checkpoint yet"
fi
check "the LIVE registry either admits YAWAUG or refuses it BY NAME — never silently" $?
eq_check "  ... and a refusal is rc 1, never a pass" \
  "$([ "$LIVE_RC" -eq 0 ] && echo admitted || echo "refused-rc${LIVE_RC}")" \
  "$([ "$LIVE_RC" -eq 0 ] && echo admitted || echo refused-rc1)"
expect_cmd "the LIVE control record admits VANL at 40000" 0 '"sha256": "1095f493' -- \
  $PY "$VALIDATOR" expect --arm VANL
expect_cmd "  ... and the record's own booleans are NOT what admits it" 0 "NO CHECKS READ" -- \
  $PY - <<'PY'
import re, sys
src = open("worklog/worklog_yixun/exp_15_yaw_aug_claude/exp15_validate_cell.py").read()
src += open("worklog/worklog_yixun/exp_15_yaw_aug_claude/exp15_admit_ckpt.py").read()
# the string "checks" may appear in prose; what must not appear is a READ of it
assert not re.search(r'\[\s*["\']checks["\']\s*\]', src), "a record's checks block is indexed"
assert not re.search(r'\.get\(\s*["\']checks["\']', src), "a record's checks block is read"
print("NO CHECKS READ")
PY

echo
echo "--- E. the DRIVER emits the plan's argv end to end (full-path DRYRUNs) ---"
A_TBL="$(argv_of "${BASE[@]}" ARM=YAWAUG CELL=tbl STEP=40000 SEED=42 K=8)"
argv_has  "T argv: vanilla conditioning" "$A_TBL" "--cond-method vanilla"
argv_has  "T argv: the pinned frame angles" "$A_TBL" "--frame-avg-angles 0,90,180,270"
argv_has  "T argv: bf16 conditioning autocast" "$A_TBL" "--cond-autocast bf16"
argv_has  "T argv: fixed mode, angle 0" "$A_TBL" "--rotate-mode fixed --rotate-deg 0"
argv_lacks "T argv: NO --rotate-seed" "$A_TBL" "--rotate-seed"
argv_has  "T argv: the campaign batch/worker pins" "$A_TBL" "--batch-size 64 --num-workers 4"
argv_has  "T argv: the assignment audit is mandatory" "$A_TBL" "--expected-stream-count 6337 --record-stream"
argv_has  "T argv: per-scene recording is mandatory" "$A_TBL" "--record-per-scene"
argv_has  "T argv: the K=8 split" "$A_TBL" "acousticroom_unseeneval.json"
argv_has  "T argv: the registered eval name" "$A_TBL" "--eval-name exp15_YAWAUG_tbl_S40000_s42_K8"
A_K1="$(argv_of "${BASE[@]}" ARM=VANL CELL=tbl STEP=40000 SEED=45 K=1)"
argv_has  "K=1 uses the _1 split (same 6337 items)" "$A_K1" "acousticroom_unseeneval_1.json"
argv_has  "K=1 still expects 6337 items" "$A_K1" "--expected-stream-count 6337"
A_RROB="$(argv_of "${BASE[@]}" ARM=YAWAUG CELL=rrob STEP=40000 SEED=44 K=8)"
argv_has  "R argv: random mode with the eval seed" "$A_RROB" "--rotate-mode random --rotate-seed 44"
argv_has  "R argv: angle stays zero" "$A_RROB" "--rotate-deg 0"
argv_has  "R argv: the _rotrand token is in the eval name" "$A_RROB" "--eval-name exp15_YAWAUG_rrob_rotrand44_S40000_s44_K8"
A_VCTL="$(argv_of "${BASE[@]}" ARM=VANL CELL=vctl STEP=40000 SEED=42 K=8 ROTATE_DEG=90)"
argv_has  "V argv: fixed 90 degrees" "$A_VCTL" "--rotate-mode fixed --rotate-deg 90"
argv_lacks "V argv: NO --rotate-seed" "$A_VCTL" "--rotate-seed"
argv_has  "V argv: the _rot90 token is in the eval name" "$A_VCTL" "--eval-name exp15_VANL_vctl_rot90_S40000_s42_K8"
# the cell-validation argv the driver renders IS the validator's own definition
if [ "$SKIP_HEAVY" = "1" ]; then
  VLINE=""
  skip_env "the driver's cell-validation argv is check_argv's own rendering" \
           "SKIP_HEAVY: needs a driver DRYRUN"
else
# The driver reads git HEAD LIVE, while this suite captured HEAD_SHA at startup.
# A commit landing mid-run (this repo has two sessions committing to it) made the
# two disagree and failed the case for a reason that had nothing to do with the
# rendering under test. Re-read HEAD next to the invocation, and compare against
# THAT.
PIN_AT_DRIVER="$(git rev-parse HEAD)"
VLINE="$(env "${BASE[@]}" ARM=YAWAUG CELL=rrob STEP=40000 SEED=44 K=8 bash "$SCREEN" 2>&1 \
         | sed -n 's/^python3 exp15_validate_cell.py //p' | head -1)"
eq_check "the driver's cell-validation argv is check_argv's own rendering" \
  "$VLINE" \
  "$($PY - "$PIN_AT_DRIVER" <<'PY'
import sys; sys.path.insert(0, "worklog/worklog_yixun/exp_15_yaw_aug_claude")
import exp15_validate_cell as V
c = V.Cell("YAWAUG", "rrob", 40000, 44, 8, None)
print(" ".join(V.check_argv(c, "<metrics>", pin=sys.argv[1], ckpt_sha="<ckpt-sha256>",
                            expected_count=V.EXPECTED_COUNT,
                            expected_scenes=V.EXPECTED_SCENES)))
PY
)"
fi

echo
echo "--- F. the per-cell VALIDATOR: artifacts that lie are named, not skipped ---"
ART="${TMP}/artifacts"; mkdir -p "$ART"
mkart() {  # <outdir> <arm> <cell> <seed> <k> <deg|-> <n> [break]
  $PY - "$ART" "$@" <<'PY'
import json, os, sys
sys.path.insert(0, "worklog/worklog_yixun/exp_15_yaw_aug_claude")
import exp15_validate_cell as V
out, arm, cell, seed, k, deg, n = sys.argv[1:8]
brk = sys.argv[8] if len(sys.argv) > 8 else ""
n = int(n)
c = V.Cell(arm, cell, 40000, int(seed), int(k), None if deg == "-" else float(deg))
mode, d, rseed = V.rotation_expectation(c)
ck = os.path.join(out, "epoch=8-step=40000.ckpt")
open(ck, "w").close()
metrics = V.metrics_path(ck, c)
PIN = "deadbeef00000000000000000000000000000042"
inp = [[i, f"Cafe/Cafe_idx_0/{i}.wav", ["ctx0"], 512] for i in range(n)]
if mode == "random":
    offs = [(i * 37 + 5) % 512 for i in range(n)]
else:
    offs = [V.expected_column_shift(d) for _ in range(n)]
asg = [[i, inp[i][1], offs[i]] for i in range(n)]
ih, ah = V.canonical_stream_hash(inp), V.canonical_stream_hash(asg)
if brk == "offsets":            # a random cell whose draws never varied
    offs = [0] * n
    asg = [[i, inp[i][1], 0] for i in range(n)]
    ih, ah = V.canonical_stream_hash(inp), V.canonical_stream_hash(asg)
stream = {"schema_version": 1, "fingerprint_schema": 1, "rotate_mode": mode,
          "rotate_seed": rseed, "rotate_deg": d, "stream_count": n, "img_w": 512,
          "input_tuples": inp, "assignment_tuples": asg, "offsets": offs,
          "input_hash": ih, "assignment_hash": ah}
if brk == "streamhash":
    stream["input_hash"] = "0" * 64
# WELL-FORMED fixtures (eval-r1 review finding 1): the previous
# {"t60": 1.0, "c50": 2.0} stub codified the very weakness the review found —
# it was accepted as a complete cell. These are the metric names a REAL exp_14
# artifact carries, read back from the committed campaign.
SPLIT = {m: 1.0 for m in V.REQUIRED_SPLIT_METRICS}
SCENE = {m: 1.0 for m in V.REQUIRED_SCENE_METRICS}
if brk == "metricmissing":
    SPLIT.pop("FD")
if brk == "metriccase":
    SPLIT["t60"] = SPLIT.pop("T60")
if brk == "metricnan":
    SPLIT["FD"] = float("nan")
if brk == "metricinf":
    SPLIT["FD"] = float("inf")
if brk == "metricbool":
    SPLIT["FD"] = True
if brk == "scenemissing":
    SCENE = {m: 1.0 for m in V.REQUIRED_SCENE_METRICS if m != "EDT"}
if brk == "scenenan":
    SCENE = dict(SCENE, C50=float("nan"))
rec = {"metrics": SPLIT, "ckpt_path": ck, "rotate_deg": d,
       "cond_method": "vanilla", "frame_avg_angles": None, "cond_autocast": "bf16",
       "source_sha": PIN, "batch_size": 64, "n_samples": n,
       "dataset_config": "src/configs/dataset_configs/AR/eval/"
                         + (V.SPLIT_K8 if int(k) == 8 else V.SPLIT_K1),
       "seed": int(seed), "cfg_scale": 1.0, "steps": 1, "eval_name": V.eval_name(c),
       "weights_source": "ema", "device": "cuda",
       "by_scene": {s: dict(SCENE) for s in V.EXPECTED_SCENE_KEYS},
       "per_scene_schema": 1, "scene_count": 10}
if brk == "extrascene":
    rec["by_scene"]["Hallways"] = dict(SCENE); rec["scene_count"] = 11
if brk == "dropscene":
    rec["by_scene"].pop("Office"); rec["scene_count"] = 9
if mode == "random":
    rec.update({"rotate_mode": "random", "rotate_deg": None, "rotate_seed": rseed,
                "input_hash": ih, "assignment_hash": ah, "stream_count": n, "img_w": 512})
if brk == "online":             # the fa-eval trap: non-EMA weights
    rec["weights_source"] = "online"
if brk == "flagonly":           # a record claiming the fa angle list under vanilla
    rec["frame_avg_angles"] = [0.0, 90.0, 180.0, 270.0]
if brk == "randomkeys":         # a FIXED record carrying random-mode provenance
    rec["rotate_mode"] = "fixed"
if brk == "noscene":
    rec.pop("by_scene"); rec.pop("scene_count")
if brk == "short":
    rec["n_samples"] = n - 1
meta = {"arm": arm, "cell": cell, "step": 40000, "seed": int(seed), "K": int(k),
        "eval_name": V.eval_name(c), "cond_method": "vanilla", "cond_autocast": "bf16",
        "frame_avg_angles": None, "frame_avg_angles_flag": "0,90,180,270",
        "rotate_mode": mode, "rotate_seed": rseed, "rotate_deg": d,
        "batch_size": 64, "num_workers": 4, "expected_stream_count": n,
        "record_stream": True, "record_per_scene": True, "use_ema": True,
        "train_yaw_aug": V.TRAIN_YAW_AUG[arm], "commit": PIN,
        "ckpt_sha256": "a" * 64}
if brk == "noflag":             # a manifest that cannot witness the pinned flag
    meta.pop("frame_avg_angles_flag")
if brk == "workers":
    meta["num_workers"] = 6
if brk == "armflip":
    meta["train_yaw_aug"] = not meta["train_yaw_aug"]
json.dump(rec, open(metrics, "w"))
json.dump(meta, open(metrics + ".screenmeta.json", "w"))
json.dump(stream, open(metrics[:-len(".json")] + ".stream.json", "w"))
print(metrics)
PY
}
vcheck() {  # <name> <want-rc> <want-substring> <arm> <cell> <seed> <k> <deg> <n> [break]
  local name="$1" rc="$2" txt="$3"; shift 3
  local m; m="$(mkart "$@")" || { echo "FAIL  ${name}: fixture build failed"; ledger FAIL "$name"; FAIL=$((FAIL+1)); return; }
  local args=(check --metrics "$m" --arm "$1" --cell "$2" --step 40000 --seed "$3" --k "$4"
              --pin "$FAKE_PIN" --ckpt-sha "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
              --expected-count "$6" --expected-scenes 10)
  [ "$5" = "-" ] || args+=(--rotate-deg "$5")
  expect_cmd "$name" "$rc" "$txt" -- $PY "$VALIDATOR" "${args[@]}"
}
vcheck "a well-formed T cell VALIDATES"        0 "VALID"   YAWAUG tbl  42 8 - 12
vcheck "a well-formed R cell VALIDATES"        0 "VALID"   YAWAUG rrob 43 8 - 12
vcheck "a well-formed V cell VALIDATES"        0 "VALID"   VANL   vctl 42 8 90 12
vcheck "ONLINE weights are refused (EMA is the contract)" 1 "weights_source" YAWAUG tbl 42 8 - 12 online
vcheck "a record claiming the fa angle list under vanilla is refused" 1 "frame_avg_angles" \
  YAWAUG tbl 42 8 - 12 flagonly
vcheck "a manifest that never witnessed the pinned flag is refused" 1 "frame_avg_angles_flag" \
  YAWAUG tbl 42 8 - 12 noflag
vcheck "a manifest with the wrong worker count is refused" 1 "num_workers" YAWAUG tbl 42 8 - 12 workers
vcheck "a manifest that mislabels the ARM's augmentation is refused" 1 "train_yaw_aug" \
  YAWAUG tbl 42 8 - 12 armflip
vcheck "a missing by_scene block is refused (wrong estimand)" 1 "PER-SCENE" YAWAUG tbl 42 8 - 12 noscene
vcheck "a short split is refused" 1 "n_samples" YAWAUG tbl 42 8 - 12 short
vcheck "a stream hash that does not recompute is refused" 1 "recomputed input_hash" \
  YAWAUG rrob 44 8 - 12 streamhash
vcheck "a random cell whose offsets never varied is refused" 1 "the random path did not run" \
  YAWAUG rrob 45 8 - 12 offsets
# --- the metric SCHEMA E2 will consume (eval-r1 review finding 1) -------------
# Each of these used to classify VALID and be SKIPPED by the wave's dedup as
# "already measured", which is the fail-open case validate-before-skip exists to
# prevent, one level down from where it was being prevented.
vcheck "a metrics block missing FD is refused" 1 "missing required metric(s) ['FD']" \
  YAWAUG tbl 42 8 - 12 metricmissing
vcheck "a wrong-cased 't60' is refused as a MISSING 'T60'" 1 "missing required metric(s) ['T60']" \
  YAWAUG tbl 42 8 - 12 metriccase
vcheck "a NaN metric is refused" 1 "non-finite/non-numeric" YAWAUG tbl 42 8 - 12 metricnan
vcheck "an Inf metric is refused" 1 "non-finite/non-numeric" YAWAUG tbl 42 8 - 12 metricinf
vcheck "a bool metric is refused (True is not 1.0)" 1 "non-finite/non-numeric" \
  YAWAUG tbl 42 8 - 12 metricbool
vcheck "a per-scene payload missing EDT is refused" 1 "missing required metric(s) ['EDT']" \
  YAWAUG tbl 42 8 - 12 scenemissing
vcheck "a per-scene NaN is refused" 1 "non-finite/non-numeric" YAWAUG tbl 42 8 - 12 scenenan
vcheck "an ELEVENTH scene group is refused" 1 "not the release grouping" \
  YAWAUG tbl 42 8 - 12 extrascene
vcheck "a NINTH scene group is refused" 1 "not the release grouping" \
  YAWAUG tbl 42 8 - 12 dropscene
# ...and the schema is a READBACK, not a guess: the real committed exp_14 cell
# must satisfy it, or we have invented a contract the eval code cannot meet.
REAL14="outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp14_VANL_rgen_S40000_s42_K8_rotrand42_rotrand42.json"
if [ -f "$REAL14" ]; then
  expect_cmd "a REAL exp_14 metrics artifact satisfies the new metric schema" 0 "SCHEMA OK" -- \
    $PY - "$REAL14" <<'PY'
import json, sys
sys.path.insert(0, "worklog/worklog_yixun/exp_15_yaw_aug_claude")
import exp15_validate_cell as V
rec = json.load(open(sys.argv[1]))
bad = V._metric_block_reasons(rec["metrics"], "metrics", V.REQUIRED_SPLIT_METRICS)
bad += V._per_scene_reasons(rec)
if bad:
    raise SystemExit("real artifact REJECTED by our own schema: " + "; ".join(bad))
print("SCHEMA OK")
PY
else
  skip_env "a REAL exp_14 metrics artifact satisfies the new metric schema" \
           "exp_14 campaign artifacts are not present in this tree"
fi
# ...and the two omissions that used to be silently survivable
M_OK="$(mkart YAWAUG tbl 46 8 - 12)"
expect_cmd "a cell cannot be VALID without the campaign pin" 1 "campaign pin not supplied" -- \
  $PY "$VALIDATOR" check --metrics "$M_OK" --arm YAWAUG --cell tbl --step 40000 --seed 46 --k 8 \
     --ckpt-sha "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa" --expected-count 12
expect_cmd "a cell cannot be VALID without the checkpoint digest" 1 "expected ckpt sha256 not supplied" -- \
  $PY "$VALIDATOR" check --metrics "$M_OK" --arm YAWAUG --cell tbl --step 40000 --seed 46 --k 8 \
     --pin "$FAKE_PIN" --expected-count 12
expect_cmd "an ABSENT artifact is MISSING (rc 3), not invalid" 3 "MISSING" -- \
  $PY "$VALIDATOR" check --metrics "${TMP}/nope.json" --arm YAWAUG --cell tbl --step 40000 \
     --seed 42 --k 8 --pin "$FAKE_PIN" --ckpt-sha "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"

echo
echo "--- G. dedup is VALIDATE-BEFORE-SKIP, and classification is fail-closed ---"
CLS="${TMP}/cls"; mkdir -p "$CLS"
# A classification root holding the good YAWAUG checkpoint plus ONE valid vctl
# artifact and ONE corrupt vctl artifact, at the campaign's real 6337 count.
cp -r "$GOOD" "$CLS/root"
$PY - "$CLS/root" "$FAKE_PIN" <<'PY'
import json, os, sys
sys.path.insert(0, "worklog/worklog_yixun/exp_15_yaw_aug_claude")
import exp15_validate_cell as V
root, pin = sys.argv[1:3]
N = V.EXPECTED_COUNT
for arm, broken in (("VANL", False), ("YAWAUG", True)):
    c = V.Cell(arm, "vctl", 40000, 42, 8, 90.0)
    ck = V.checkpoint_path(root, arm)
    m = V.metrics_path(ck, c)
    inp = [[i, f"Cafe/Cafe_idx_0/{i}.wav", ["ctx0"], 512] for i in range(N)]
    offs = [V.expected_column_shift(90.0)] * N
    asg = [[i, inp[i][1], offs[i]] for i in range(N)]
    stream = {"schema_version": 1, "fingerprint_schema": 1, "rotate_mode": "fixed",
              "rotate_seed": None, "rotate_deg": 90.0, "stream_count": N, "img_w": 512,
              "input_tuples": inp, "assignment_tuples": asg, "offsets": offs,
              "input_hash": V.canonical_stream_hash(inp),
              "assignment_hash": V.canonical_stream_hash(asg)}
    SPLIT = {m: 1.0 for m in V.REQUIRED_SPLIT_METRICS}
    SCENE = {m: 1.0 for m in V.REQUIRED_SCENE_METRICS}
    rec = {"metrics": SPLIT, "ckpt_path": ck, "rotate_deg": 90.0,
           "cond_method": "vanilla", "frame_avg_angles": None, "cond_autocast": "bf16",
           "source_sha": pin, "batch_size": 64, "n_samples": N,
           "dataset_config": "src/configs/dataset_configs/AR/eval/" + V.SPLIT_K8,
           "seed": 42, "cfg_scale": 1.0, "steps": 1, "eval_name": V.eval_name(c),
           "weights_source": "online" if broken else "ema", "device": "cuda",
           "by_scene": {s: dict(SCENE) for s in V.EXPECTED_SCENE_KEYS},
           "per_scene_schema": 1, "scene_count": 10}
    sha = json.load(open(os.path.join(root, "registry.json")))["arms"]["YAWAUG"]["final_ckpt_sha256"] \
        if arm == "YAWAUG" else json.load(open(os.path.join(root, "control.json")))["checkpoint"]["sha256"]
    meta = {"arm": arm, "cell": "vctl", "step": 40000, "seed": 42, "K": 8,
            "eval_name": V.eval_name(c), "cond_method": "vanilla", "cond_autocast": "bf16",
            "frame_avg_angles": None, "frame_avg_angles_flag": "0,90,180,270",
            "rotate_mode": "fixed", "rotate_seed": None, "rotate_deg": 90.0,
            "batch_size": 64, "num_workers": 4, "expected_stream_count": N,
            "record_stream": True, "record_per_scene": True, "use_ema": True,
            "train_yaw_aug": V.TRAIN_YAW_AUG[arm], "commit": pin, "ckpt_sha256": sha}
    json.dump(rec, open(m, "w"))
    json.dump(meta, open(m + ".screenmeta.json", "w"))
    json.dump(stream, open(m[:-len(".json")] + ".stream.json", "w"))
print("classification fixtures written")
PY
CLSOUT="$(env $PY "$VALIDATOR" classify --wave vctl --output-root "$CLS/root" --pin "$FAKE_PIN" \
            --registry "$CLS/root/registry.json" --control "$CLS/root/control.json" 2>&1)"
echo "$CLSOUT" | grep -q "^VANL vctl 40000 42 8 90 VALID "
check "a complete, protocol-conformant cell classifies VALID" $?
echo "$CLSOUT" | grep -q "^YAWAUG vctl 40000 42 8 90 INVALID .*weights_source"
check "a cell whose artifacts exist but LIE classifies INVALID, with the reason" $?
echo "$CLSOUT" | grep -qv "MISSING"
check "  ... and neither is reported as MISSING" $?
expect_cmd "an unfinished chain refuses to CLASSIFY at all (not 'MISSING')" 2 \
  "has NOT recorded its final checkpoint yet" -- \
  $PY "$VALIDATOR" classify --wave tbl --output-root "$CLS/root" --pin "$FAKE_PIN" \
     --registry "${TMP}/registry_unfinished.json" --control "$CLS/root/control.json"
# ...and the WAVE submitter halts on that INVALID cell rather than skipping it.
printf '%s\n' "$FAKE_PIN" > "$TEST_PIN_FILE"
: > "$TEST_CMDLOG"; : > "$TEST_TRACE"; : > "$TEST_QUEUE"
git -C "$FAKE_REPO" tag -f "guardpin" >/dev/null 2>&1 || true
GRID_ENV=(YAW_EVAL_TEST_MODE=1 "YAW_EVAL_MAIN_REPO=${FAKE_REPO}"
          "YAW_EVAL_PIN_FILE=${TEST_PIN_FILE}" "YAW_EVAL_COMMAND_LOG=${TEST_CMDLOG}"
          "YAW_EVAL_TEST_RECORD=${TEST_TRACE}" "YAW_EVAL_SQUEUE_FIXTURE=${TEST_QUEUE}"
          "YAW_EVAL_WT_DIR=${TMP}/wt" "OUTPUT_ROOT=${CLS}/root"
          "YAW_EVAL_REGISTRY=${CLS}/root/registry.json" "YAW_EVAL_CONTROL=${CLS}/root/control.json")
# The pin must be a real commit in the FAKE repo for the wave to proceed.
printf '%s\n' "$FAKE_HEAD" > "$TEST_PIN_FILE"
$PY - "$CLS/root" "$FAKE_HEAD" <<'PY'
# re-stamp the artifacts' source_sha/commit to the fake repo's HEAD so the only
# thing separating the two cells is the pathology under test
import glob, json, sys
root, pin = sys.argv[1:3]
for p in glob.glob(root + "/**/*_metrics_*.json", recursive=True):
    if p.endswith(".screenmeta.json"):
        continue
    d = json.load(open(p)); d["source_sha"] = pin; json.dump(d, open(p, "w"))
for p in glob.glob(root + "/**/*.screenmeta.json", recursive=True):
    d = json.load(open(p)); d["commit"] = pin; json.dump(d, open(p, "w"))
print("re-stamped")
PY
OUT="$(env "${GRID_ENV[@]}" bash "$GRID" WAVE=vctl 2>&1)"; RC=$?
eq_check "the wave HALTS (rc 3) on an artifact that does not validate" "$RC" "3"
echo "$OUT" | grep -q "must not be skipped"
check "  ... and says so in the words of the validate-before-skip rule" $?
[ ! -s "$TEST_TRACE" ]
check "  ... having submitted NOTHING" $?
# ...and with the corrupt artifact removed, the valid one is SKIPPED and the
# missing one is SUBMITTED (simulated).
rm -f "$(ls "$CLS"/root/exp15_YAWAUG/FLAC_*/*/checkpoints/*_metrics_*vctl*.json | head -1)"
: > "$TEST_TRACE"
OUT="$(env "${GRID_ENV[@]}" bash "$GRID" WAVE=vctl 2>&1)"; RC=$?
eq_check "with the corrupt cell gone the wave runs" "$RC" "0"
echo "$OUT" | grep -q "SKIP  exp15-screen-VANL-vctl-rot90-40000-s42-K8: already measured and valid"
check "  ... the VALID cell is skipped as already measured" $?
echo "$OUT" | grep -q "SUBMIT exp15-screen-YAWAUG-vctl-rot90-40000-s42-K8"
check "  ... and only the missing cell is submitted" $?
eq_check "  ... exactly one simulated submission" "$(grep -c '^submit ' "$TEST_TRACE")" "1"
grep -q 'launching-line-present' "$TEST_TRACE"
check "  ... whose command log line was written BEFORE the submission" $?
grep -q 'LAUNCHING' "$TEST_CMDLOG"
check "  ... into the (redirected) command log" $?

echo
echo "--- H. DRYRUN enumerates the grid and submits nothing ---"
: > "$TEST_TRACE"
DRY="$(env DRYRUN=1 "YAW_EVAL_MAIN_REPO=${FAKE_REPO}" "YAW_EVAL_TEST_RECORD=${TEST_TRACE}" \
        bash "$GRID" WAVE=all 2>/dev/null)"
eq_check "DRYRUN WAVE=all prints exactly 42 cells" "$(printf '%s\n' "$DRY" | grep -c .)" "42"
eq_check "DRYRUN:   ... 20 tbl" "$(printf '%s\n' "$DRY" | grep -c 'CELL=tbl ')" "20"
eq_check "DRYRUN:   ... 20 rrob" "$(printf '%s\n' "$DRY" | grep -c 'CELL=rrob ')" "20"
eq_check "DRYRUN:   ... 2 vctl, both at 90 degrees" "$(printf '%s\n' "$DRY" | grep -c 'CELL=vctl .*ROTATE_DEG=90')" "2"
eq_check "DRYRUN:   ... 21 per arm" "$(printf '%s\n' "$DRY" | grep -c 'ARM=YAWAUG')" "21"
eq_check "DRYRUN:   ... and no cell carries a rotation angle outside vctl" \
  "$(printf '%s\n' "$DRY" | grep -c 'CELL=\(tbl\|rrob\).*ROTATE_DEG')" "0"
[ ! -s "$TEST_TRACE" ]
check "DRYRUN submitted nothing at all" $?
DRYSET="$(printf '%s\n' "$DRY" | sed 's/.*yaw_aug_screen_submit.sh //' | sort)"
GRIDSET="$($PY "$VALIDATOR" grid | awk '{printf "ARM=%s CELL=%s STEP=%s SEED=%s K=%s", $1,$2,$3,$4,$5; if ($6!="-") printf " ROTATE_DEG=%s", $6; print ""}' | sort)"
[ "$DRYSET" = "$GRIDSET" ]
check "the DRYRUN list IS the registered grid, cell for cell" $?
expect_cmd "an unregistered WAVE is refused" 2 "is not vctl|tbl|rrob|all" -- \
  env DRYRUN=1 "YAW_EVAL_MAIN_REPO=${FAKE_REPO}" bash "$GRID" WAVE=zref
expect_cmd "MAX_INFLIGHT above the campaign cap is refused" 2 "must be 1..16" -- \
  env DRYRUN=1 MAX_INFLIGHT=32 "YAW_EVAL_MAIN_REPO=${FAKE_REPO}" bash "$GRID" WAVE=all
grep -q 'grep -c .\^exp15-screen-' "$GRID"
check "concurrency counts only THIS campaign's screen jobs" $?

echo
echo "--- I. the single-cell submitter: identity, contract, and refusals ---"
sub_dry() { env DRYRUN=1 "YAW_EVAL_MAIN_REPO=${FAKE_REPO}" bash "$SUB" "$@" 2>&1; }
expect_cmd "submitter: an exp_14 arm is refused" 2 "not a registered exp_15 arm" -- sub_dry ARM=C4L CELL=tbl
expect_cmd "submitter: an exp_14 cell type is refused" 2 "not a registered cell type" -- sub_dry ARM=YAWAUG CELL=zref
expect_cmd "an unknown KEY is refused" 2 "unknown argument" -- sub_dry ARM=YAWAUG CELL=tbl ORBIT=4
expect_cmd "a LOG outside the experiment folder is refused" 2 "_screen.log path" -- \
  sub_dry ARM=YAWAUG CELL=tbl LOG=/tmp/x_screen.log
expect_cmd "the job name comes from the validator" 0 "exp15-screen-YAWAUG-rrob-rotrand46-40000-s46-K1" -- \
  sub_dry ARM=YAWAUG CELL=rrob SEED=46 K=1
expect_cmd "the pre-launch intent records the whole protocol, not just the rotation" 0 \
  "DRYRUN intent eval_argv_protocol --cond-method vanilla --frame-avg-angles 0,90,180,270" -- \
  sub_dry ARM=VANL CELL=tbl SEED=42 K=8
expect_cmd "  ... including the split and the stream/per-scene contract" 0 \
  "DRYRUN intent split acousticroom_unseeneval_1.json expected_stream_count 6337 record_stream yes" -- \
  sub_dry ARM=VANL CELL=tbl SEED=42 K=1
expect_cmd "  ... and the arm's augmentation status" 0 "DRYRUN intent train_yaw_aug yes" -- \
  sub_dry ARM=YAWAUG CELL=tbl SEED=42 K=8
expect_cmd "a dry run submits nothing and prepares no worktree" 0 "no worktree prepared, no lease written" -- \
  sub_dry ARM=YAWAUG CELL=tbl SEED=42 K=8
# --- a LIVE single submission REQUIRES the pin file (eval-r1 review finding 2) --
# The wave path was already blocked by an absent pin; the single-cell path fell
# back to HEAD, and the planned V/probe launches go through the single-cell path.
# These run in LIVE mode on purpose (that is the subject), against a temp
# MAIN_REPO, and every one exits 2 long before the freeze/worktree/lock steps.
NOPIN_REPO="${TMP}/nopin_repo"
mkdir -p "${NOPIN_REPO}/worklog/worklog_yixun/exp_15_yaw_aug_claude"
git -c init.defaultBranch=main init -q "$NOPIN_REPO"
git -C "$NOPIN_REPO" -c user.email=g@l -c user.name=g commit -q --allow-empty -m root
NOPIN_HEAD="$(git -C "$NOPIN_REPO" rev-parse HEAD)"
cp "$VALIDATOR" "${NOPIN_REPO}/worklog/worklog_yixun/exp_15_yaw_aug_claude/"
NOPIN_SUB="${NOPIN_REPO}/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_screen_submit.sh"
sed "s|^MAIN_REPO=/n/fs/gatrdp/codespace/FLAC$|MAIN_REPO=${NOPIN_REPO}|" "$SUB" > "$NOPIN_SUB"
NOPIN_PIN="${NOPIN_REPO}/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_screen_campaign_pin"
# LINT-OK-live-refusal: the SUBJECT is the live pin requirement, so live mode is
# required; it exits 2 at the pin check, before any transaction step.
expect_cmd "a LIVE single submission with NO pin file refuses" 2 "no campaign pin FILE" -- \
  env -u YAW_EVAL_MAIN_REPO bash "$NOPIN_SUB" ARM=YAWAUG CELL=tbl SEED=42 K=8
printf '%s\n' "$NOPIN_HEAD" > "$NOPIN_PIN"
# The MISMATCH branch runs before the DRYRUN block, so it is reachable
# hermetically — no store-lock re-exec and no stub helper needed. PIN_SHA is only
# ever an assertion about the file's content.
expect_cmd "  ... and a PIN_SHA that contradicts the pin file refuses" 2 "disagrees with the campaign pin" -- \
  env -u YAW_EVAL_MAIN_REPO DRYRUN=1 bash "$NOPIN_SUB" ARM=YAWAUG CELL=tbl SEED=42 K=8 \
      PIN_SHA=deadbeef00000000000000000000000000000042
expect_cmd "  ... while a PIN_SHA that AGREES is accepted" 0 "DRYRUN pin ${NOPIN_HEAD}" -- \
  env -u YAW_EVAL_MAIN_REPO DRYRUN=1 bash "$NOPIN_SUB" ARM=YAWAUG CELL=tbl SEED=42 K=8 \
      PIN_SHA="$NOPIN_HEAD"
printf 'not-a-sha\n' > "$NOPIN_PIN"
# LINT-OK-live-refusal: a malformed pin file must not read as "pinned".
expect_cmd "  ... and a MALFORMED pin file is not a pin" 2 "no campaign pin FILE" -- \
  env -u YAW_EVAL_MAIN_REPO bash "$NOPIN_SUB" ARM=YAWAUG CELL=tbl SEED=42 K=8
rm -f "$NOPIN_PIN"
expect_cmd "  ... while DRYRUN still runs unpinned" 0 "DRY RUN complete" -- \
  env -u YAW_EVAL_MAIN_REPO DRYRUN=1 bash "$NOPIN_SUB" ARM=YAWAUG CELL=tbl SEED=42 K=8
expect_cmd "  ... and says out loud that a live run would refuse" 0 \
  "a LIVE submission would REFUSE here" -- \
  env -u YAW_EVAL_MAIN_REPO DRYRUN=1 bash "$NOPIN_SUB" ARM=YAWAUG CELL=tbl SEED=42 K=8
grep -q 'no campaign pin FILE' "$SUB" && grep -q 'no campaign pin FILE' "$GRID"
check "both submission paths refuse an unpinned live launch in the same words" $?
grep -q 'refused BEFORE taking the shared store lock' "$SUB"
check "  ... and the single-cell path refuses BEFORE taking the shared store lock" $?

# --- F1 CANARY: only PINNED control-plane content is used and attributed -------
# The blocking integrative finding: the pin bound the code the JOB read, but the
# submitter, the validator that rendered identity/contract, and the driver handed
# to sbatch all came from the MOVING main checkout. A post-pin edit could change
# what executed while the artifact still recorded the pinned commit.
#
# This diverges the two trees deliberately and proves the pinned one wins.
CANARY_WT="${TMP}/canary_wt"
CANARY_EXP="${CANARY_WT}/worklog/worklog_yixun/exp_15_yaw_aug_claude"
mkdir -p "$CANARY_EXP"
cp "$SUB" "$GRID" "$CANARY_EXP/"
# the PINNED validator renders a canary job name; the MAIN one is untouched
sed 's|^    return (f"exp15-screen-{cell.arm}|    return ("PINNED-CANARY-" + f"exp15-screen-{cell.arm}|' \
    "$VALIDATOR" > "${CANARY_EXP}/exp15_validate_cell.py"
printf '#!/bin/bash\n# PINNED-DRIVER-CANARY\nexit 0\n' > "${CANARY_EXP}/yaw_aug_screen.sbatch"
$PY -c "
import sys
src = open(sys.argv[1]).read()
sys.exit(0 if 'PINNED-CANARY-' in src else 1)" "${CANARY_EXP}/exp15_validate_cell.py"
check "canary: the pinned validator really does differ from the main one" $?
CANARY_OUT="$(env -u YAW_EVAL_MAIN_REPO DRYRUN=1 "YAW_EVAL_PINNED_EXEC=${CANARY_WT}" \
              bash "${CANARY_EXP}/yaw_aug_screen_submit.sh" ARM=YAWAUG CELL=tbl SEED=42 K=8 2>&1)"
case "$CANARY_OUT" in
  *"PINNED-CANARY-exp15-screen-YAWAUG-tbl"*)
    echo "PASS  F1: identity is rendered by the PINNED validator, not the main one"
    ledger PASS "F1: identity is rendered by the PINNED validator, not the main one"
    PASS=$((PASS + 1)) ;;
  *) echo "FAIL  F1: the main-tree validator rendered the identity"
     printf '%s\n' "$CANARY_OUT" | head -4 | sed 's/^/        | /'
     ledger FAIL "F1: identity is rendered by the PINNED validator, not the main one"
     FAIL=$((FAIL + 1)) ;;
esac
case "$CANARY_OUT" in
  *"DRYRUN driver ${CANARY_EXP}/yaw_aug_screen.sbatch"*)
    echo "PASS  F1: the driver handed to sbatch comes from the PINNED tree"
    ledger PASS "F1: the driver handed to sbatch comes from the PINNED tree"
    PASS=$((PASS + 1)) ;;
  *) echo "FAIL  F1: the driver path is not the pinned one"
     printf '%s\n' "$CANARY_OUT" | grep -i driver | sed 's/^/        | /'
     ledger FAIL "F1: the driver handed to sbatch comes from the PINNED tree"
     FAIL=$((FAIL + 1)) ;;
esac
case "$CANARY_OUT" in
  *"DRYRUN control-plane PINNED"*)
    echo "PASS  F1: a pinned run SAYS it is pinned"
    ledger PASS "F1: a pinned run SAYS it is pinned"; PASS=$((PASS + 1)) ;;
  *) echo "FAIL  F1: a pinned run does not announce itself"
     ledger FAIL "F1: a pinned run SAYS it is pinned"; FAIL=$((FAIL + 1)) ;;
esac
MAIN_OUT="$(env -u YAW_EVAL_MAIN_REPO DRYRUN=1 bash "$SUB" ARM=YAWAUG CELL=tbl SEED=42 K=8 2>&1)"
case "$MAIN_OUT" in
  *"PINNED-CANARY-"*) echo "FAIL  F1: the main tree leaked pinned content"
                      ledger FAIL "F1: an unpinned dry run announces itself as MAIN-TREE"
                      FAIL=$((FAIL + 1)) ;;
  *"DRYRUN control-plane MAIN-TREE (unpinned)"*)
                      echo "PASS  F1: an unpinned dry run announces itself as MAIN-TREE"
                      ledger PASS "F1: an unpinned dry run announces itself as MAIN-TREE"
                      PASS=$((PASS + 1)) ;;
  *) echo "FAIL  F1: an unpinned dry run does not announce its tree"
     ledger FAIL "F1: an unpinned dry run announces itself as MAIN-TREE"; FAIL=$((FAIL + 1)) ;;
esac
grep -q 'exec bash "$PINNED_SELF"' "$SUB" && grep -q 'exec bash "$PINNED_SELF"' "$GRID"
check "F1: both entry points re-exec themselves from the pinned worktree" $?
grep -q 'CODE_EXPDIR/yaw_aug_screen.sbatch' "$SUB"
check "F1: sbatch receives the pinned driver, never the main copy" $?
grep -q 'VALIDATE="$CODE_EXPDIR/exp15_validate_cell.py"' "$GRID"
check "F1: wave enumeration/classification use the pinned validator" $?
INTENTS_BEFORE="$(ls "${EXPDIR}"/yaw_aug_screen_submission_*.txt 2>/dev/null | wc -l)"
sub_dry ARM=YAWAUG CELL=tbl SEED=42 K=8 >/dev/null 2>&1
INTENTS_AFTER="$(ls "${EXPDIR}"/yaw_aug_screen_submission_*.txt 2>/dev/null | wc -l)"
eq_check "a dry run leaves no intent manifest behind" "$INTENTS_BEFORE" "$INTENTS_AFTER"
awk '/^publish_intent\(\)/{pi=NR} /^if ! slurm_release/{rel=NR} END{exit !(pi && rel && pi < rel)}' "$SUB"
check "the intent is published while the job is still HELD" $?
grep -q 'slurm_cancel "--name=\${JOB_NAME}"' "$SUB"
check "an unparseable job id is still cancelled, by NAME" $?
grep -q 'yaw_aug_screen_submission_' "$SUB"
check "eval intents cannot be confused with the training kit's submission manifests" $?
grep -q 'yaw_aug_screen_campaign_pin' "$SUB" && grep -q 'yaw_aug_screen_campaign_pin' "$GRID"
check "the eval campaign keeps its OWN pin file" $?
# The read-only reader entry point takes no lock and reads no seam — so it is run
# with the suite's isolation variable UNSET, which is also what proves that a
# live-mode invocation refuses that variable rather than honouring it.
printf 'job 1 name x\narm YAWAUG cell tbl\n' > "${TMP}/truncated_intent.txt"
expect_cmd "--verify-manifest rejects a manifest with no completion sentinel" 1 "manifest INCOMPLETE" -- \
  env -u YAW_EVAL_MAIN_REPO bash "$SUB" --verify-manifest "${TMP}/truncated_intent.txt"
printf 'job 1 name x\nmanifest_complete yes\n' > "${TMP}/complete_intent.txt"
expect_cmd "  ... and accepts one that carries it" 0 "manifest complete" -- \
  env -u YAW_EVAL_MAIN_REPO bash "$SUB" --verify-manifest "${TMP}/complete_intent.txt"
# LINT-OK-live-refusal: this case's SUBJECT is the live-mode allowlist, so it must
# reach live mode; it exits 2 at the gate, before any transaction step exists.
expect_cmd "a LIVE-mode run refuses the isolation seam itself" 2 "is not on this mode's allowlist" -- \
  env "YAW_EVAL_MAIN_REPO=${FAKE_REPO}" bash "$SUB" ARM=YAWAUG CELL=tbl

echo
echo "--- J. environment hygiene: neighbouring namespaces are REFUSED ---"
expect_cmd "a leftover YAW_AUG_* (training kit) is refused by the wave submitter" 2 \
  "is not on this mode's allowlist" -- \
  env DRYRUN=1 "YAW_EVAL_MAIN_REPO=${FAKE_REPO}" YAW_AUG_SQUEUE_CMD=/bin/true bash "$GRID" WAVE=all
expect_cmd "a leftover YAW_GEN_* (exp_14) is refused by the wave submitter" 2 \
  "is not on this mode's allowlist" -- \
  env DRYRUN=1 "YAW_EVAL_MAIN_REPO=${FAKE_REPO}" YAW_GEN_TEST_MODE=1 bash "$GRID" WAVE=all
expect_cmd "a leftover YAW_AUG_* is refused by the single-cell submitter" 2 \
  "is not on this mode's allowlist" -- \
  env DRYRUN=1 "YAW_EVAL_MAIN_REPO=${FAKE_REPO}" YAW_AUG_REPO_OVERRIDE=/tmp bash "$SUB" ARM=YAWAUG CELL=tbl
# A seam that could redirect the campaign's own state is refused outright by a run
# that is not in test mode — the allowlist names it before anything else can.
expect_cmd "a test seam without TEST_MODE is refused in a run that could submit" 2 \
  "is not on this mode's allowlist" -- \
  env -u YAW_EVAL_MAIN_REPO "YAW_EVAL_PIN_FILE=${TEST_PIN_FILE}" bash "$SUB" ARM=YAWAUG CELL=tbl
# SBATCH_EXCLUDE sits deep in the LIVE path (after the freeze and the worktree),
# which no hermetic case can reach, so it is asserted STRUCTURALLY and labelled as
# such rather than dressed up as a behavioural test.
grep -q 'SBATCH_EXCLUDE' "$SUB" && grep -q 'pass EXCLUDE=' "$SUB"
check "SBATCH_EXCLUDE is refused rather than silently ignored (structural)" $?
grep -q 'EXCLUDE_ARGV=(--exclude=' "$SUB"
check "node exclusion travels as an explicit --exclude flag (structural)" $?
DRYX="$(env DRYRUN=1 "YAW_EVAL_MAIN_REPO=${FAKE_REPO}" bash "$GRID" WAVE=vctl EXCLUDE=neu303,neu332 2>/dev/null)"
echo "$DRYX" | grep -q "EXCLUDE=neu303,neu332"
check "EXCLUDE= is passed through to every cell of a wave" $?
expect_cmd "a malformed EXCLUDE is refused" 2 "not a comma-separated node list" -- \
  env DRYRUN=1 "YAW_EVAL_MAIN_REPO=${FAKE_REPO}" bash "$GRID" WAVE=vctl "EXCLUDE=neu303;rm"

echo
echo "--- Z. the suite touched nothing tracked, and submitted nothing ---"
TRACKED_AFTER="$(tracked_snapshot)"
if [ "$TRACKED_BEFORE" = "$TRACKED_AFTER" ]; then
  echo "PASS  tracked tree unchanged by the suite (worktree content before == after)"
  ledger PASS "tracked tree unchanged by the suite (worktree content before == after)"; PASS=$((PASS+1))
else
  echo "FAIL  the suite changed tracked state:"
  diff <(echo "$TRACKED_BEFORE") <(echo "$TRACKED_AFTER") | sed 's/^/        | /'
  ledger FAIL "tracked tree unchanged by the suite (worktree content before == after)"; FAIL=$((FAIL+1))
fi
NEIGHBOURS_AFTER="$(ls -li --time-style=+%s "$EXP11" "$EXP14" | sort)"
NEIGH_DIFF="$(diff <(printf '%s\n' "$NEIGHBOURS_BEFORE") <(printf '%s\n' "$NEIGHBOURS_AFTER") \
              | grep '^[<>]' || true)"
# WHAT THIS CAN AND CANNOT ESTABLISH (eval-r1 review finding 3).
#
# The previous version called a delta "ours" when its listing text contained
# exp15/yaw_aug_screen/guardtests/__pycache__, and PASSed otherwise. That is not
# attribution: a write to an EXISTING exp_11 or exp_14 filename would have been
# reported as somebody else's and still passed. The causal claim is withdrawn.
#
# What is asserted now is only what the observation supports:
#   * no delta  -> nothing in either folder changed while this suite ran. That
#                  is a fact about the folders, and it is the outcome we want.
#   * a delta   -> INCONCLUSIVE. Another session writes into exp_11's folder
#                  continuously (its live registry is uncommitted right now), so
#                  a delta is expected and this snapshot cannot say whose it is.
#                  It is NOT reported as a pass.
# The guarantees that do hold are structural and live in the case below it: the
# suite exports PYTHONDONTWRITEBYTECODE=1 so no import can drop a .pyc there, and
# no line of the kit or of this suite can write into either folder at all.
if [ -z "$NEIGH_DIFF" ]; then
  echo "PASS  nothing in the exp_11 / exp_14 folders changed while this suite ran"
  ledger PASS "nothing in the exp_11 / exp_14 folders changed while this suite ran"
  PASS=$((PASS + 1))
else
  skip_env "nothing in the exp_11 / exp_14 folders changed while this suite ran" \
           "INCONCLUSIVE — entries changed during the run and this snapshot cannot attribute them; another session writes into exp_11 continuously"
  printf '%s\n' "$NEIGH_DIFF" | sed 's/^/        | /'
fi
# ...and the SOUND half: neither the kit nor this suite contains any statement
# that writes into those folders. A static property of the code, not a race with
# whoever else is running.
NEIGH_WRITES="$(grep -nE '(>|>>|cp |mv |rm |mkdir |touch |tee )[^|]*\$(EXP11|EXP14)\b' \
                  "$GUARD_SELF" "$SCREEN" "$SUB" "$GRID" "$VALIDATOR" "$ADMIT" \
                | grep -v 'grep -nE' || true)"
[ -z "$NEIGH_WRITES" ]
check "no line of the kit or this suite writes into the exp_11 / exp_14 folders" $?
[ -n "$NEIGH_WRITES" ] && printf '%s\n' "$NEIGH_WRITES" | sed 's/^/        | /'
grep -q 'export PYTHONDONTWRITEBYTECODE=1' "$GUARD_SELF"
check "  ... and bytecode writing is disabled so an import cannot drop a .pyc there" $?
! pgrep -u "$(id -un)" -f 'yaw_aug_screen.sbatch' >/dev/null 2>&1
check "no screen driver was left running" $?

# The ledger is keyed on case NAMES, so two cases sharing one would be a single
# row in the union check and a name a reviewer cannot map back to a case. The
# regime that catches "skipped in every environment" has to catch this too.
DUPES="$(cut -f2 "$LEDGER" | sort | uniq -d)"
if [ -z "$DUPES" ]; then
  echo "PASS  every case name in the ledger is unique"
  ledger PASS "every case name in the ledger is unique"; PASS=$((PASS + 1))
else
  echo "FAIL  duplicate case names in the ledger (union coverage keys on them):"
  printf '%s\n' "$DUPES" | sed 's/^/        | /'
  ledger FAIL "every case name in the ledger is unique"; FAIL=$((FAIL + 1))
fi

echo
echo "=== guard tests: ${PASS} passed, ${FAIL} failed ($(grep -c '^SKIP' "$LEDGER" 2>/dev/null || echo 0) skipped, STRICT=${STRICT}) ==="
echo "ledger: ${LEDGER}"
echo "union coverage across transcripts:  ${PY} ${UNION} ${EXPDIR}/*_guardtests_evalr1.ledger"
[ "$FAIL" -eq 0 ] || exit 1
echo "log: ${LOG}"
