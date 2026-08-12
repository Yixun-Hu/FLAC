#!/bin/bash
# ============================================================================
# exp_15 YAWAUG TRAINING LAUNCHER — the single augmented arm at exp_11's rung.
#
# Derived VERBATIM from exp_11's fa_orbit_train.sbatch (commit 50a526d, whose
# blobs match exp_11's at parent de7c41d) and then re-pinned; every gate exp_11's
# three review rounds installed is retained, plus its later content-scoped
# binding fix (2b75036). The deltas are enumerated in plan §6.6.
#
# EXIT CLASSES: 0/3/4/6/7 are exp_11's taxonomy (fa_orbit_classify.py). exp_15
# adds two, both about whether this run is the EXPERIMENT it claims to be:
#   8  the yaw_aug banner never appeared before the first step (a green run that
#      was not actually augmented), or a green run whose banner never appeared;
#   9  the 40,000-step checkpoint could not be produced, audited or recorded.
#
# WHAT THIS ARM IS: vanilla-conditioned FLAC trained with a random horizontal
# yaw applied to every training sample's CONDITIONING (plan §3.1). It is compared
# against exp_11's VANL arm at step 40,000 — a historical, recipe-matched control
# (plan §3.3) — so the recipe here is VANL's, literally, and the only intended
# difference is training.yaw_aug in the model config.
#
# SUBMIT: never by hand — `yaw_aug_submit.sh YAWAUG` derives every Slurm flag
# from the pins below, so an operator cannot mis-enter --gres/--mem/--time.
#   ARM                YAWAUG (the only legal arm)
#   EXPECT_SHA         full reviewed commit sha (required)
#   RESUME_CKPT/EXPECTED_STEP   crash restart only (see LINEAGE)
#   SMOKE=1            the reviewed multi-GPU smoke (see SMOKE MODE)
# RUNG / MAXSTEPS / MIN_FREE_MB / time limit are NOT operator inputs any more.
#
# LINEAGE (fail-closed, exactly two stories):
#   INITIAL  no RESUME_CKPT, EXPECTED_STEP unset/0, run directory absent.
#   RESTART  a CRASH restart only, and only up to the pre-registered 40,000-step
#            endpoint: EXPECTED_STEP > 0 AND RESUME_CKPT inside this arm's OWN
#            <RUNDIR>/checkpoints/ AND the checkpoint passes
#            fa_orbit_ckpt_preflight.py (embedded step/config/optimizer/
#            scheduler/EMA + binding to the original launch manifest).
#            exp_11's Q10 "extension" mode (40k -> 100k) is NOT carried over:
#            exp_15's endpoint is pre-registered at 40k and a leg that targets
#            beyond it is refused, so there is nothing to extend.
#
# EXP_11 HELPERS, READ-ONLY: three of exp_11's reviewed, arm-agnostic tools are
# invoked from their own folder rather than copied — fa_orbit_ckpt_preflight.py,
# fa_orbit_classify.py, fa_orbit_wandb_readback.py. They take every experiment
# fact as an argument. exp_11's folder is never written by this kit; the drift
# gate below covers those three files so a silent edit cannot reach a launch.
#
# WORLD SIZE: no absence timer (round-3 B4 — a cold start with W&B has no
# measured bound, and `scancel` bypassed classification). Instead: a watcher that
# terminates the torchrun process group the moment Lightning reports the WRONG
# rank count, plus the post-hoc classification in fa_orbit_classify.py.
#
# torchrun: PL 2.1.0 elects TorchElastic before SLURMEnvironment, so the ranks
# torchrun starts are used as-is; the SLURM rank variables are unset so
# SLURMEnvironment cannot claim the job. train.py is unmodified and rank-safe:
# WandbLogger.experiment is @rank_zero_experiment, and ModelCheckpoint.setup
# broadcasts rank 0's dirpath to every rank.
#
# SMOKE MODE (SMOKE=1): the reviewed pre-launch smoke. Bypasses ONLY the "pins
# must be pinned" gate; every other gate still runs. Uses SMOKE_RUNG,
# SMOKE_MAXSTEPS (small), SMOKE_MIN_FREE_MB, its own identity
# (FLAC_exp15_smoke_<ARM> / exp15_smoke_<ARM>) and its own save-dir prefix, so a
# smoke can never touch or resume an arm's real lineage.
#
# TEST HOOK: OUTPUT_ROOT (default outputs_FLAC) relocates the output namespace so
# the guard tests never write under a production prefix. It changes no gate.
# ============================================================================
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/slurm_train_%x_%j.out
# TRANSCRIPT POLICY. This file is written by Slurm for the whole life of the run.
# During the run it is deliberately UNTRACKED (the job removes it from the index
# at launch, see the untrack block below): a tracked file that a running job
# appends to is one a git checkout/stash can unlink out from under the job's file
# descriptor, freezing the visible transcript while the run continues. Completed
# transcripts are committed by the OPERATOR at run closure with `git add -f`.

set -uo pipefail

# ============================ PINNED RECIPE =================================
# Filled from the reviewed P0 report; until then every value is the literal
# placeholder and the launcher refuses to run (except under SMOKE=1).
PIN_PLACEHOLDER="TO-PIN-AFTER-P0"
PINNED_RUNG="8x8"                          # exp_11 P0 run 1334933 + spot 9bf1936; VANL trained at this rung
PINNED_MB="8"                              # micro-batch per GPU (8 x 8 = 64 = eff = BN batch)
PINNED_NGPU="8"                            # ranks
# exp_15's endpoint is PRE-REGISTERED at 40,000 (plan §3.2, §5: no checkpoint
# selection). exp_11's kit arrived carrying its Q10 extension budget of 100000;
# re-pinned here, and the production assertion below re-pinned with it.
PINNED_MAXSTEPS=40000
PINNED_CHECKPOINT_EVERY=2500               # exp_07 cadence, as VANL ran it
PINNED_MIN_FREE_MB="36500"                 # exp_11's reviewed max-across-arms floor; VANL launched under it
# YAWAUG is VANL's recipe plus the augmentation. VANL measured 40k at ~1.07
# steps/s = 10.6 h; the augmentation adds one panorama roll + four small pose
# rotations per sample on the CPU side, benchmarked before launch (plan §7-6,
# acceptance >= 0.9x VANL's rate). 24 h is that x1.3 plus startup, with margin
# for the slower bound the acceptance test still admits.
PINNED_TIME_LIMIT_YAWAUG="24:00:00"
# A RESTART leg is a CRASH restart inside the same 40k budget — never longer than
# a fresh run, so it carries the same wall pin.
PINNED_TIME_LIMIT_RESTART_YAWAUG="24:00:00"
PINNED_P0_MANIFEST_SHA256="72607b922177208d56055d604b292d697b643ef3b7ab48261ab2e23a0cc2b53b"  # exp_11's batched P0 matrix manifest — provenance of the rung this arm inherits
# Environment pins (exp_11 round-3 B6), cross-checked against VANL's own launch
# manifest at gate I2 so "the same environment" is verified, not asserted:
PINNED_PYTHON="/n/fs/gatrdp/envs/flac/bin/python"
PINNED_PL_VERSION="2.1.0"
PINNED_TORCH_VERSION="2.7.0+cu126"
PINNED_VAE_SHA256="8d82159eec35210198246f449bec6561fc19b514922f340a17515050daf7f0b9"
# The control's launch manifest (plan §3.3-5) and exp_11's registry pin for it.
PINNED_CONTROL_MANIFEST="outputs_FLAC/exp11_VANL/launch_manifest.txt"
PINNED_CONTROL_MANIFEST_SHA256="113d06a284c6198cf9487e99a2efb7ccde94ae13e656a403fe2af0281d3de8b1"
# The commit VANL was trained at: the base of the launch-pin diff allowlist gate.
PINNED_CONTROL_COMMIT="81ddac372076ea92751ae09cbaf371df70f396e5"
# The registered augmentation block (plan §3.1). ONE source of truth: the config
# gate asserts training.yaw_aug equals exactly these values, and the post-launch
# log gate waits for the banner they imply, so a config and a gate cannot drift.
PINNED_YAW_AUG_IMG_W="512"
PINNED_YAW_AUG_SEED="42"
# ============================================================================

REPO=/n/fs/gatrdp/codespace/FLAC
# TEST HOOK (guard tests only): sbatch copies this script to a spool dir, so the
# repo path must be absolute; YAW_AUG_REPO_OVERRIDE lets the guard suite point a
# dry run at a worktree. It is honoured ONLY outside a Slurm job and scrubbed
# immediately, so it can never influence a real launch.
if [ -n "${YAW_AUG_REPO_OVERRIDE:-}" ] && [ -z "${SLURM_JOB_ID:-}" ]; then
  REPO="$YAW_AUG_REPO_OVERRIDE"
fi
unset YAW_AUG_REPO_OVERRIDE
EXPDIR="$REPO/worklog/worklog_yixun/exp_15_yaw_aug_claude"
EXP11DIR="$REPO/worklog/worklog_yixun/exp_11_fa_orbit_claude"   # READ-ONLY: arm-agnostic helpers + the control's manifest
EXP07="$REPO/worklog/worklog_yixun/exp_07_fa_scratch_claude"
cd "$REPO" || exit 3
unset PYTHONPATH PYTHONOPTIMIZE
export PATH=/n/fs/gatrdp/envs/flac/bin:$PATH
export PYTHONNOUSERSITE=1
export HF_HOME=/n/fs/gatrdp/hf_cache
export HF_HUB_OFFLINE=1
# Belt and braces for the launch-gate banner (round-1 review, finding 3): the
# banner already prints with flush=True, but torchrun's stdout reaches the log
# through a FIFO, where a block-buffered child could otherwise hold back the one
# line the post-launch gate below waits for.
export PYTHONUNBUFFERED=1

DRYRUN="${DRYRUN:-0}"
SMOKE="${SMOKE:-0}"
# NEW-2: the production output namespace is not operator state. Inside a Slurm
# job it is the literal below; an ambient value that disagrees aborts. The
# override exists only for non-Slurm guard dry runs.
PRODUCTION_OUTPUT_ROOT="outputs_FLAC"
if [ -n "${SLURM_JOB_ID:-}" ]; then
  if [ -n "${OUTPUT_ROOT:-}" ] && [ "$OUTPUT_ROOT" != "$PRODUCTION_OUTPUT_ROOT" ]; then
    echo "ambient OUTPUT_ROOT='${OUTPUT_ROOT}' != the production literal '${PRODUCTION_OUTPUT_ROOT}' - abort"; exit 2
  fi
  OUTPUT_ROOT="$PRODUCTION_OUTPUT_ROOT"
else
  OUTPUT_ROOT="${OUTPUT_ROOT:-$PRODUCTION_OUTPUT_ROOT}"
fi
RESUME_CKPT="${RESUME_CKPT:-}"
EXPECTED_STEP="${EXPECTED_STEP:-0}"
TS="$(date '+%Y-%m-%d_%H-%M-%S')"

die() { echo "$1"; exit "${2:-2}"; }

# --- A. parameters ------------------------------------------------------------
[ -n "${ARM:-}" ] || die "ARM must be exported (YAWAUG) - abort"
[ -n "${EXPECT_SHA:-}" ] || die "EXPECT_SHA (full reviewed commit sha) must be exported - abort"
case "$ARM" in
  YAWAUG) ;;
  *) die "ARM '${ARM}' is not a legal exp_15 arm — YAWAUG is the only one (VANL is exp_11's historical control and is never retrained here) - abort" ;;
esac
case "$EXPECTED_STEP" in ''|*[!0-9]*) die "EXPECTED_STEP '${EXPECTED_STEP}' must be a non-negative integer - abort";; esac

# --- B. the pins decide the recipe (round-3 B1) -------------------------------
if [ "$SMOKE" = "1" ]; then
  RUNG="${SMOKE_RUNG:-}"; MAXSTEPS="${SMOKE_MAXSTEPS:-30}"; MIN_FREE_MB="${SMOKE_MIN_FREE_MB:-}"
  CHECKPOINT_EVERY="${SMOKE_CHECKPOINT_EVERY:-10}"
  [ -n "$RUNG" ] || die "SMOKE=1 requires SMOKE_RUNG (${PINNED_RUNG}) - abort"
  [ -n "$MIN_FREE_MB" ] || die "SMOKE=1 requires SMOKE_MIN_FREE_MB (per-GPU floor) - abort"
  # The approved smoke is the REAL rung (plan §7-6: the point is to measure the
  # augmented step rate on the production topology and prove the banner fires
  # there). A smoke at 32x2 or 16x4 would measure a recipe nobody will run, and
  # exp_11's kit accepted all three.
  [ "$RUNG" = "$PINNED_RUNG" ] || die "SMOKE_RUNG '${RUNG}' != the pinned production rung ${PINNED_RUNG}: exp_15's smoke runs the real topology or not at all - abort"
  TIME_LIMIT="${SMOKE_TIME:-00:30:00}"; TIME_PIN_NAME="SMOKE_TIME"
  NAME="FLAC_exp15_smoke_${ARM}"; EXPNAME="exp15_smoke_${ARM}"
  SAVEDIR="${OUTPUT_ROOT}/exp15_smoke/${ARM}"
  echo "=== SMOKE MODE: pins bypassed, EVERY other gate active; identity ${EXPNAME} ==="
else
  # The wall pin follows the LEG, not just the arm (exp_11 re-pin fix 1: the job
  # used to select the INITIAL pin and then reject its own correct allocation).
  # For exp_15 both pins are 24:00:00 — a crash restart finishes the SAME 40k
  # budget — but the selection machinery is kept so the job and the submitter
  # provably agree on which pin they are enforcing.
  if [ "$EXPECTED_STEP" -gt 0 ]; then
    TIME_PIN_NAME="PINNED_TIME_LIMIT_RESTART_${ARM}"
  else
    TIME_PIN_NAME="PINNED_TIME_LIMIT_${ARM}"
  fi
  for PIN_NAME in PINNED_RUNG PINNED_MB PINNED_NGPU PINNED_MIN_FREE_MB PINNED_P0_MANIFEST_SHA256 \
                  "$TIME_PIN_NAME"; do
    eval "PIN_VAL=\${$PIN_NAME}"
    [ "$PIN_VAL" != "$PIN_PLACEHOLDER" ] || die "${PIN_NAME} is still '${PIN_PLACEHOLDER}': the P0 report has not been pinned into this launcher yet — no arm may launch (use SMOKE=1 for the pre-launch smoke) - abort"
  done
  RUNG="$PINNED_RUNG"; MAXSTEPS="$PINNED_MAXSTEPS"; MIN_FREE_MB="$PINNED_MIN_FREE_MB"
  CHECKPOINT_EVERY="$PINNED_CHECKPOINT_EVERY"
  eval "TIME_LIMIT=\${${TIME_PIN_NAME}}"
  NAME="FLAC_exp15_${ARM}"; EXPNAME="exp15_${ARM}"; SAVEDIR="${OUTPUT_ROOT}/exp15_${ARM}"
fi

case "$RUNG" in
  8x8) ;;
  *) die "rung '${RUNG}' must be 8x8 — exp_15 has ONE topology, the one VANL trained at - abort" ;;
esac
MB="${RUNG%x*}"; NGPU="${RUNG#*x}"
[ "$((MB * NGPU))" -eq 64 ] || die "rung ${RUNG}: MB*NGPU = $((MB*NGPU)) != 64 (micro x N pin, plan §10) - abort"
# micro/NGPU are pinned for SMOKE too (review finding 1): a smoke that measured a
# different topology would answer a question nobody asked.
[ "$MB" = "$PINNED_MB" ] && [ "$NGPU" = "$PINNED_NGPU" ] || die "pin inconsistency: rung ${RUNG} vs PINNED_MB=${PINNED_MB}/PINNED_NGPU=${PINNED_NGPU} - abort"
if [ "$SMOKE" != "1" ]; then
  # The pre-registered endpoint (plan §3.2/§5). A production launch at any other
  # budget is not the experiment that was registered.
  [ "$MAXSTEPS" = "40000" ] || die "PINNED_MAXSTEPS is ${MAXSTEPS}, the registered budget is 40000 - abort"
fi
RUNDIR="${SAVEDIR}/${NAME}/${EXPNAME}"
echo "=== exp_15 arm ${ARM} @ rung ${RUNG} (MB ${MB} x ${NGPU} GPU, grad-ckpt ON, yaw_aug ON) — ${TS} — host $(hostname) ==="

# --- C. commit binding + tracked-surface drift --------------------------------
# Ported from exp_11's content-scoped gate (2b75036 and its predecessor): the
# closure is what the JOB ACTUALLY LOADS, the pathspecs are QUOTED so git — not
# the shell — expands them (a tracked file DELETED from the worktree still
# matches), a failing git invocation is fatal rather than an empty "clean"
# answer, and src/tests is excluded because pytest-only code is never imported by
# train.py and the concurrent TDD sessions land test files continuously (that
# exact over-broad closure killed exp_11 legs 3680875-78 on 2026-08-12).
HEAD_SHA="$(git rev-parse HEAD 2>/dev/null)" || HEAD_SHA=""
EXPREL="${EXPDIR#"$REPO"/}"; EXP11REL="${EXP11DIR#"$REPO"/}"; EXP07REL="${EXP07#"$REPO"/}"
# The runtime closure, enumerated once and used by BOTH gates below. data/AR
# holds the split JSONs the dataloader opens (train.json is the one this run
# reads); an uncommitted edit there changes which samples are trained while
# leaving every code hash intact, which is exactly the hole this closes.
TRAINING_CLOSURE=(
  train.py defaults.ini src ":(exclude)src/tests" data/AR
  "$EXPREL/FLAC_AR_YAWAUG.json"
  "$EXPREL/yaw_aug_train.sbatch" "$EXPREL/yaw_aug_submit.sh"
  "$EXPREL/yaw_aug_record_control.py"            # imported by the completion gate
  "$EXPREL/yaw_aug_pin_allowlist.txt" "$EXPREL/yaw_aug_control_admission.json"
  "$EXP11REL/fa_orbit_ckpt_preflight.py" "$EXP11REL/fa_orbit_classify.py"
  "$EXP11REL/fa_orbit_wandb_readback.py" "$EXP11REL/FLAC_AR_VANCKPT.json"
  "$EXP07REL/FLAC_AR_BF.json"
)
DRIFT="$(git status --porcelain --untracked-files=no -- "${TRAINING_CLOSURE[@]}" 2>&1)" \
  || die "git status for the drift gate failed: ${DRIFT} - abort"
# Commit binding is CONTENT-scoped: HEAD identity is sufficient but not
# necessary. Two sessions commit to this checkout, so a pending leg must survive
# commits that leave the training closure untouched — and abort on any commit
# that changes it. Fail-closed on every edge: EXPECT_SHA must be a full 40-hex
# commit OID (a symbolic ref would defeat the binding), the diff runs against the
# CAPTURED HEAD OID, and HEAD is re-read afterwards to close the
# commit-during-the-gate race.
surface_diff_vs_expect() {
  git diff --name-only "${EXPECT_SHA}" "${HEAD_SHA}" -- "${TRAINING_CLOSURE[@]}"
}
GATE_FAIL=""; GATE_OK_MSG=""
if [ -z "$HEAD_SHA" ]; then
  GATE_FAIL="cannot resolve HEAD"
elif ! printf '%s\n' "$EXPECT_SHA" | grep -qE '^[0-9a-f]{40}$'; then
  GATE_FAIL="EXPECT_SHA '${EXPECT_SHA}' is not a full lowercase 40-hex commit id"
elif [ "$(git rev-parse --verify -q "${EXPECT_SHA}^{commit}" 2>/dev/null)" != "$EXPECT_SHA" ]; then
  GATE_FAIL="EXPECT_SHA ${EXPECT_SHA} is not a commit known to this repo"
elif [ "$HEAD_SHA" = "$EXPECT_SHA" ]; then
  GATE_OK_MSG="commit binding OK: ${HEAD_SHA}"
elif SD="$(surface_diff_vs_expect 2>&1)"; then
  if [ -z "$SD" ]; then
    GATE_OK_MSG="commit binding OK (content): training surfaces identical, EXPECT_SHA ${EXPECT_SHA} HEAD ${HEAD_SHA}"
  else
    GATE_FAIL="training surfaces changed since EXPECT_SHA ${EXPECT_SHA} (HEAD ${HEAD_SHA}): ${SD}"
  fi
else
  GATE_FAIL="surface diff vs EXPECT_SHA failed: ${SD}"
fi
if [ -z "$GATE_FAIL" ] && [ "$(git rev-parse HEAD 2>/dev/null)" != "$HEAD_SHA" ]; then
  GATE_FAIL="HEAD moved during the commit-binding check (was ${HEAD_SHA})"
fi
if [ "$DRYRUN" = "1" ]; then
  [ -z "$GATE_FAIL" ] && echo "${GATE_OK_MSG} (dry run)" \
    || echo "DRY-RUN ADVISORY: ${GATE_FAIL} (a real launch aborts here)"
  [ -z "$DRIFT" ] || { echo "DRY-RUN ADVISORY: tracked measurement surfaces are modified "\
                            "(a real launch aborts here):"; echo "$DRIFT"; }
else
  [ -n "${SLURM_JOB_ID:-}" ] || die "a real launch must run under sbatch (no SLURM_JOB_ID) - abort"
  [ -z "$GATE_FAIL" ] || die "${GATE_FAIL} - abort"
  [ -z "$DRIFT" ] || { echo "tracked measurement surfaces modified since review - abort:"; echo "$DRIFT"; exit 2; }
  echo "$GATE_OK_MSG"
fi

# --- C2. launch-pin diff allowlist (plan §3.3-2) ------------------------------
# The control is HISTORICAL: VANL trained at ${PINNED_CONTROL_COMMIT}, this arm
# trains at HEAD. Everything that changed on a production surface in between is a
# candidate second factor, so the set of changed files must be one a human
# reviewed and committed — not whatever happens to be in the tree today. Anything
# outside the allowlist aborts the launch rather than quietly joining the recipe.
# Scope is train.py, defaults.ini and src/ per the plan; eval_FLAC.py is not a
# training surface and cannot influence this run.
ALLOWLIST_FILE="${EXPDIR}/yaw_aug_pin_allowlist.txt"
[ -f "$ALLOWLIST_FILE" ] || die "the launch-pin allowlist ${ALLOWLIST_FILE} is missing - abort"
PIN_DIFF="$(git diff --name-only "${PINNED_CONTROL_COMMIT}..HEAD" -- train.py defaults.ini src 2>&1)" \
  || die "could not diff ${PINNED_CONTROL_COMMIT}..HEAD (${PIN_DIFF}) — the allowlist gate cannot be evaluated - abort"
mapfile -t ALLOW_PATTERNS < <(sed -e 's/#.*//' -e 's/[[:space:]]*$//' "$ALLOWLIST_FILE" | grep -v '^$')
[ "${#ALLOW_PATTERNS[@]}" -gt 0 ] || die "the launch-pin allowlist is empty - abort"
PIN_SURPRISES=""
while IFS= read -r CHANGED; do
  [ -n "$CHANGED" ] || continue
  MATCHED=0
  for PAT in "${ALLOW_PATTERNS[@]}"; do
    case "$CHANGED" in $PAT) MATCHED=1; break ;; esac
  done
  [ "$MATCHED" = "1" ] || PIN_SURPRISES="${PIN_SURPRISES}${CHANGED}"$'\n'
done <<< "$PIN_DIFF"
if [ -n "$PIN_SURPRISES" ]; then
  echo "LAUNCH-PIN ALLOWLIST: production surfaces changed since the control's commit"
  echo "  ${PINNED_CONTROL_COMMIT} that are NOT on ${ALLOWLIST_FILE}:"
  printf '%s' "$PIN_SURPRISES" | sed 's/^/    /'
  if [ "$DRYRUN" = "1" ]; then
    echo "DRY-RUN ADVISORY: a real launch aborts here"
  else
    die "unreviewed production-surface changes since the control's commit - abort"
  fi
else
  echo "launch-pin allowlist OK: $(printf '%s' "$PIN_DIFF" | grep -c . 2>/dev/null || echo 0) changed production file(s) since ${PINNED_CONTROL_COMMIT}, all allowlisted"
fi

# --- D. arm -> config (single source) + semantic gate -------------------------
arm_config_for() {
  case "$1" in
    YAWAUG) echo "$EXPDIR/FLAC_AR_YAWAUG.json" ;;
    *) return 1 ;;
  esac
}
MODEL_CONFIG="$(arm_config_for "$ARM")" || die "no config mapped for arm '${ARM}' - abort"
MODEL_CONFIG_ABS="$(readlink -f "$MODEL_CONFIG" 2>/dev/null)"
[ -n "$MODEL_CONFIG_ABS" ] && [ -f "$MODEL_CONFIG_ABS" ] || die "arm config '${MODEL_CONFIG}' does not exist - abort"
CONFIG_SHA="$(sha256sum "$MODEL_CONFIG_ABS" | awk '{print $1}')"
echo "config for ${ARM}: ${MODEL_CONFIG_ABS} sha256 ${CONFIG_SHA}"

python3 - "$MODEL_CONFIG_ABS" "$ARM" "$PINNED_YAW_AUG_IMG_W" "$PINNED_YAW_AUG_SEED" \
  <<'PY' || die "arm/config semantic gate FAILED - abort"
import json, sys
cfg = json.load(open(sys.argv[1])); arm = sys.argv[2]
pin_img_w, pin_seed = int(sys.argv[3]), int(sys.argv[4])
t = cfg.get("training", {}); bad = []
# YAWAUG is VANL's recipe plus the augmentation, so the gate is the mirror image
# of exp_11's orbit gate: the FA keys must be ABSENT (a stray cond_method or
# frame_avg_angles would make this a silently fa-flavoured arm and destroy the
# single-delta claim, and announcement 05's protocol trap starts exactly there),
# and the yaw_aug block must be the registered one LITERALLY. Equality of the
# whole block is the right test because the factory rejects unknown keys, so
# there is no legal extra key a looser check would have to tolerate.
WANT_YAW_AUG = {"enabled": True, "img_w": pin_img_w, "seed": pin_seed}
if "cond_method" in t:
    bad.append(f"cond_method is present ({t['cond_method']!r}) — the arm must be vanilla-conditioned")
if "frame_avg_angles" in t:
    bad.append(f"frame_avg_angles is present ({t['frame_avg_angles']!r}) — a vanilla arm has no orbit")
ya = t.get("yaw_aug")
if ya != WANT_YAW_AUG:
    bad.append(f"training.yaw_aug={ya!r} != the registered block {WANT_YAW_AUG!r}")
elif type(ya.get("enabled")) is not bool or type(ya.get("img_w")) is not int \
        or type(ya.get("seed")) is not int:
    bad.append(f"training.yaw_aug has non-literal types: {ya!r}")
if t.get("use_ema") is not True:
    bad.append(f"use_ema={t.get('use_ema')!r} (want True)")
vits = [c for c in cfg["model"]["conditioning"]["configs"] if c["type"] == "ViTCoordinates"]
if sorted(c["id"] for c in vits) != ["context_poses_vit", "source_vit"]:
    bad.append(f"ViT conditioner ids {sorted(c['id'] for c in vits)} != the expected two")
for c in vits:
    if "gradient_checkpointing" not in c["config"]:
        bad.append(f"{c['id']}: gradient_checkpointing key absent (want literal true)")
    elif c["config"]["gradient_checkpointing"] is not True:
        bad.append(f"{c['id']}: gradient_checkpointing={c['config']['gradient_checkpointing']!r} (want True)")
    # The ViT pin (exp_11 gate N, kept as a config assertion) AND the coupling
    # that matters here: the augmentation rolls a panorama of width img_w, so a
    # ViT expecting a different width would be fed a roll it cannot interpret.
    vit = c["config"].get("ViT", {})
    if vit.get("hf_model_name_or_path") != "facebook/dinov3-vits16-pretrain-lvd1689m":
        bad.append(f"{c['id']}: ViT {vit.get('hf_model_name_or_path')!r} != the pinned DINOv3-S/16")
    if (vit.get("img_h"), vit.get("img_w")) != (256, 512):
        bad.append(f"{c['id']}: ViT img {(vit.get('img_h'), vit.get('img_w'))} != (256, 512)")
    if isinstance(ya, dict) and vit.get("img_w") != ya.get("img_w"):
        bad.append(f"{c['id']}: ViT img_w {vit.get('img_w')!r} != yaw_aug.img_w {ya.get('img_w')!r}")
if bad:
    sys.exit("ARM/CONFIG GATE: " + "; ".join(bad))
print(f"gate OK: {arm} is vanilla (no cond_method, no orbit), yaw_aug {WANT_YAW_AUG}, "
      "grad-ckpt True, EMA on, DINOv3-S/16 at 256x512")
PY

# --- E. lineage: INITIAL vs RESTART -------------------------------------------
SAVEDIR_REAL="$(realpath -m "$SAVEDIR")"
CKPT_DIR_REAL="$(realpath -m "${RUNDIR}/checkpoints")"
LAUNCH_MANIFEST_LINK="${SAVEDIR}/launch_manifest.txt"     # written by the INITIAL launch
if [ "$EXPECTED_STEP" -eq 0 ]; then
  MODE="INITIAL"
  [ -z "$RESUME_CKPT" ] || die "INITIAL launch must not carry RESUME_CKPT (set EXPECTED_STEP > 0 to declare a RESTART) - abort"
  [ ! -e "$RUNDIR" ] || die "run directory ${RUNDIR} already exists — an INITIAL launch never clobbers a previous run - abort"
else
  MODE="RESTART"
  [ -n "$RESUME_CKPT" ] || die "EXPECTED_STEP ${EXPECTED_STEP} declares a RESTART, but RESTART requires RESUME_CKPT - abort"
  [ -f "$RESUME_CKPT" ] || die "RESUME_CKPT not found: ${RESUME_CKPT} - abort"
  RESUME_REAL="$(realpath -m "$RESUME_CKPT")"
  # exactly this arm's own checkpoints directory — not merely somewhere below the save root
  case "$RESUME_REAL" in
    "${CKPT_DIR_REAL}"/*.ckpt) ;;
    *) die "a RESTART may only resume a checkpoint from ${CKPT_DIR_REAL}/ (got ${RESUME_REAL}) - abort" ;;
  esac
  # The pre-registered endpoint caps every leg (plan §3.2/§6.6). exp_11's kit
  # arrived with the Q10 EXTENSION semantics, where a restart deliberately raised
  # the budget past the registered comparison point; exp_15 has no such mode, so a
  # leg that resumes at or beyond 40,000 — or targets beyond it — is refused here.
  # Checked BEFORE the generic budget arithmetic below so an operator reaching for
  # an extension is told what is actually wrong, not that 40000 fails to exceed
  # 40000.
  if [ "$SMOKE" != "1" ]; then
    [ "$MAXSTEPS" -le 40000 ] || die "a RESTART leg may not target beyond the pre-registered 40000-step endpoint (MAXSTEPS ${MAXSTEPS}) - abort"
    [ "$EXPECTED_STEP" -lt 40000 ] || die "resume step ${EXPECTED_STEP} is already at/past the pre-registered 40000-step endpoint — there is nothing left to run and exp_15 has no extension mode - abort"
  fi
  [ "$MAXSTEPS" -gt "$EXPECTED_STEP" ] || die "MAXSTEPS ${MAXSTEPS} must exceed the resume step ${EXPECTED_STEP} - abort"
fi
echo "lineage: ${MODE} (expected_step ${EXPECTED_STEP}, max_steps ${MAXSTEPS}, ckpt every ${CHECKPOINT_EVERY}, time pin ${TIME_PIN_NAME}=${TIME_LIMIT})"

# --- F. the exact train.py argv ----------------------------------------------
# ACCUMULATION IS LOAD-BEARING (round-1 code review, accumulation finding).
# PL's global_step counts OPTIMIZER steps, and the augmentation seeds its yaw
# draws from (yaw_aug.seed, global_step, global_rank). With --accum-batches > 1
# every micro-batch inside one accumulation window shares a global_step and would
# therefore receive the IDENTICAL yaw vector — the augmentation would silently
# collapse to one draw per window. The pin is asserted below, after the argv is
# built, so the assertion reads the value that will actually be passed.
PINNED_ACCUM_BATCHES=1
ARGV=(
  --model-config "$MODEL_CONFIG_ABS"
  --dataset-config src/configs/dataset_configs/AR/train/acousticroom_train.json
  --pretransform-ckpt-path weights/FLAC/VAE.safetensors
  --max-steps "$MAXSTEPS" --batch-size "$MB" --accum-batches "$PINNED_ACCUM_BATCHES" --num-workers 6 --seed 42
  --num-gpus "$NGPU" --num-nodes 1
  --strategy ddp_find_unused_parameters_true --sync-batchnorm true --precision bf16-mixed
  --val-every -1 --val-dataset-config ''
  --gradient-clip-val 0.0
  --logger wandb --checkpoint-every "$CHECKPOINT_EVERY"
  --name "$NAME" --experiment-name "$EXPNAME" --save-dir "$SAVEDIR"
)
[ "$MODE" = "RESTART" ] && ARGV+=(--ckpt-path "$RESUME_CKPT")

# --- F2. accumulation gate: read the value OUT of the constructed argv --------
# Not a re-statement of the pin: the argv array is scanned so that any future
# edit which changes what is actually passed is caught here rather than at the
# post-mortem of a run whose yaw draws were duplicated 1-in-N.
ACCUM_IN_ARGV=""
for _i in "${!ARGV[@]}"; do
  [ "${ARGV[$_i]}" = "--accum-batches" ] && ACCUM_IN_ARGV="${ARGV[$((_i + 1))]:-}"
done
[ -n "$ACCUM_IN_ARGV" ] || die "the constructed argv carries no --accum-batches — yaw draws would follow whatever defaults.ini says - abort"
[ "$ACCUM_IN_ARGV" = "1" ] || die "the constructed argv sets --accum-batches ${ACCUM_IN_ARGV}: PL's global_step counts optimizer steps, so every micro-batch in an accumulation window would draw the SAME yaws - abort"
echo "accumulation gate OK: --accum-batches ${ACCUM_IN_ARGV} (one optimizer step per micro-batch, so one independent yaw draw per sample per visit)"

# --- G. argv-parity dry run (exp_11 plan N13; round-3 N9 tightened) -----------
ARGV_FILE="$(mktemp)" || die "mktemp failed - abort" 3
printf '%s\n' "${ARGV[@]}" > "$ARGV_FILE" || die "could not write the argv file - abort" 3
python3 - "$ARGV_FILE" "$MODE" <<'PY'
import sys
# The exp_07 B-F reference argv (bf_scratch_launch.sh) — the lineage this arm continues.
REF = """--model-config worklog/worklog_yixun/exp_07_fa_scratch_claude/FLAC_AR_BF.json
--dataset-config src/configs/dataset_configs/AR/train/acousticroom_train.json
--pretransform-ckpt-path weights/FLAC/VAE.safetensors
--max-steps 67500 --batch-size 32 --accum-batches 1 --num-workers 6 --seed 42
--num-gpus 2 --strategy ddp_find_unused_parameters_true --sync-batchnorm true
--logger wandb --checkpoint-every 2500
--name FLAC_exp07_BF --experiment-name exp07_BF --save-dir outputs_FLAC/exp07_BF""".split()
# Flags whose VALUE may differ from exp_07 (identity, budget, rung, resume):
ALLOWED_DIFF = {"--model-config", "--name", "--experiment-name", "--save-dir", "--max-steps",
                "--num-gpus", "--batch-size", "--logger", "--checkpoint-every", "--ckpt-path"}
# Flags exp_07 left to defaults.ini and we state explicitly — whitelisted with their
# EXACT expected values (round-3 N9: no "equals the mutable ini" escape hatch):
ALLOWED_ADD = {"--num-nodes": "1", "--precision": "bf16-mixed", "--val-every": "-1",
               "--val-dataset-config": "", "--gradient-clip-val": "0.0", "--ckpt-path": None}
tokens = [t for t in open(sys.argv[1]).read().split("\n")]
if tokens and tokens[-1] == "":
    tokens.pop()
mode = sys.argv[2]

def as_map(toks):
    out, i = {}, 0
    while i < len(toks):
        flag = toks[i]
        if not flag.startswith("--"):
            raise SystemExit(f"ARGV PARITY: stray token {flag!r}")
        val = toks[i + 1] if i + 1 < len(toks) and not toks[i + 1].startswith("--") else ""
        if flag in out:
            raise SystemExit(f"ARGV PARITY: duplicate flag {flag}")
        out[flag] = val
        i += 2 if (i + 1 < len(toks) and not toks[i + 1].startswith("--")) else 1
    return out

ref, new = as_map(REF), as_map(tokens)
violations, allowed, explicit = [], [], []
for flag in sorted(set(ref) | set(new)):
    if flag in ref and flag in new:
        if ref[flag] != new[flag]:
            (allowed if flag in ALLOWED_DIFF else violations).append(
                f"{flag}: exp_07 {ref[flag]!r} -> exp_15 {new[flag]!r}")
    elif flag in new:
        if flag == "--ckpt-path":
            (allowed if mode == "RESTART" else violations).append(
                f"--ckpt-path: {new[flag]!r} (RESTART only)")
        elif flag in ALLOWED_ADD and ALLOWED_ADD[flag] == new[flag]:
            explicit.append(f"{flag}={new[flag]!r} (whitelisted explicit default)")
        else:
            violations.append(f"{flag}: added with {new[flag]!r}, not a whitelisted addition "
                              f"(expected {ALLOWED_ADD.get(flag, '<not allowed>')!r})")
    else:
        violations.append(f"{flag}: present in exp_07 ({ref[flag]!r}), MISSING here")

print("--- train.py argv ---")
print(" ".join(f"{k} {v!r}" if v == "" else f"{k} {v}" for k, v in new.items()))
print("--- argv parity vs exp_07 B-F ---")
for d in allowed:
    print(f"  allowed  {d}")
for d in explicit:
    print(f"  explicit {d}")
if violations:
    print("ARGV PARITY VIOLATIONS:")
    for v in violations:
        print(f"  !! {v}")
    raise SystemExit(2)
print(f"ARGV PARITY OK ({mode}): only whitelisted differences and additions")
PY
parity=$?
rm -f "$ARGV_FILE"
[ "$parity" -eq 0 ] || die "argv parity check FAILED - abort"
if [ "$DRYRUN" = "1" ]; then
  echo "DRY RUN complete: gates A–G passed for ARM=${ARM} RUNG=${RUNG} MODE=${MODE} SMOKE=${SMOKE}"
  echo "  (the allowlist gate ran above, advisory-only here; Slurm/GPU/VRAM/env/wandb/lock"
  echo "   gates and training are skipped in DRYRUN)"
  exit 0
fi

# --- H. Slurm allocation must match the pins (round-3 B1) ---------------------
[ "${SLURM_JOB_NUM_NODES:-1}" = "1" ] || die "expected 1 node, got ${SLURM_JOB_NUM_NODES} - abort"
[ "${SLURM_NTASKS:-1}" = "1" ] || die "expected 1 task, got ${SLURM_NTASKS} - abort"
WANT_CPUS="$((8 + 7 * NGPU))"; WANT_MEM_MB="$(((12 * NGPU + 12) * 1024))"
GOT_CPUS="${SLURM_CPUS_PER_TASK:-${SLURM_CPUS_ON_NODE:-0}}"
GOT_MEM_MB="${SLURM_MEM_PER_NODE:-0}"
[ "$GOT_CPUS" = "$WANT_CPUS" ] || die "allocated ${GOT_CPUS} CPUs, the pinned rung needs ${WANT_CPUS} — submit via yaw_aug_submit.sh - abort"
[ "$GOT_MEM_MB" = "$WANT_MEM_MB" ] || die "allocated ${GOT_MEM_MB} MB RAM, the pinned rung needs ${WANT_MEM_MB} — submit via yaw_aug_submit.sh - abort"
GOT_TIME="$(squeue -h -j "$SLURM_JOB_ID" -o %l 2>/dev/null | tr -d ' ')"
norm_minutes() { awk -v t="$1" 'BEGIN{d=0; if (t ~ /-/) {split(t,p,"-"); d=p[1]; t=p[2]} n=split(t,c,":");
  if (n==3) m=c[1]*60+c[2]+c[3]/60; else if (n==2) m=c[1]+c[2]/60; else m=t; printf "%d", d*1440+m}'; }
# The pin this ${MODE} leg is entitled to — an INITIAL allocation handed to a
# RESTART leg (or the reverse) is refused here, in the job, not merely intended
# by the submitter.
[ "$(norm_minutes "$GOT_TIME")" = "$(norm_minutes "$TIME_LIMIT")" ] || die "allocated time ${GOT_TIME} != the ${TIME_PIN_NAME} pin ${TIME_LIMIT} this ${MODE} leg requires — submit via yaw_aug_submit.sh - abort"
echo "allocation matches the pins: ${GOT_CPUS} cpus, ${GOT_MEM_MB} MB, ${GOT_TIME} (${TIME_PIN_NAME})"

mapfile -t GPU_ROWS < <(nvidia-smi --query-gpu=uuid,name --format=csv,noheader,nounits)
[ "${#GPU_ROWS[@]}" -eq "$NGPU" ] || die "rung ${RUNG} needs exactly ${NGPU} visible GPU(s), got ${#GPU_ROWS[@]} - abort"
UUIDS=()
for ROW in "${GPU_ROWS[@]}"; do
  U="$(echo "$ROW" | cut -d, -f1 | tr -d ' ')"; N="$(echo "$ROW" | cut -d, -f2 | sed 's/^ *//')"
  case "$N" in *L40*) ;; *) die "allocated GPU '${N}' is not an L40 - abort";; esac
  UUIDS+=("$U"); echo "GPU: ${N} ${U}"
done
UUID_CSV="$(IFS=,; echo "${UUIDS[*]}")"

# --- I. environment identity gates (round-3 B6) -------------------------------
REAL_PY="$(command -v python)"
[ "$(readlink -f "$REAL_PY")" = "$(readlink -f "$PINNED_PYTHON")" ] || die "python resolves to ${REAL_PY}, pinned ${PINNED_PYTHON} - abort"
python3 - "$PINNED_PL_VERSION" "$PINNED_TORCH_VERSION" <<'PY' || die "runtime version gate FAILED - abort"
import sys
import pytorch_lightning as pl, torch
want_pl, want_torch = sys.argv[1], sys.argv[2]
bad = []
if pl.__version__ != want_pl:
    bad.append(f"pytorch_lightning {pl.__version__} != pinned {want_pl}")
if torch.__version__ != want_torch:
    bad.append(f"torch {torch.__version__} != pinned {want_torch}")
if bad:
    sys.exit("; ".join(bad))
print(f"runtime gate OK: pl {pl.__version__}, torch {torch.__version__}, cuda {torch.version.cuda}")
PY
VAE_SHA="$(sha256sum weights/FLAC/VAE.safetensors | awk '{print $1}')"
[ "$VAE_SHA" = "$PINNED_VAE_SHA256" ] || die "weights/FLAC/VAE.safetensors sha256 ${VAE_SHA} != pinned ${PINNED_VAE_SHA256} - abort"
echo "VAE pin OK: ${VAE_SHA}"
DRIVER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
CUDA_VER="$(python3 -c 'import torch; print(torch.version.cuda)')"

# --- I2. env cross-check against the CONTROL's own launch manifest (§3.3-5) ----
# The pins above say what this run's environment must be. This says the control
# ran in that same environment — read out of VANL's launch manifest rather than
# taken on trust, with the manifest itself sha-gated against exp_11's registry so
# a drifted or rewritten manifest cannot supply the agreement.
[ -f "$PINNED_CONTROL_MANIFEST" ] || die "the control's launch manifest ${PINNED_CONTROL_MANIFEST} is missing — the §3.3-5 env cross-check cannot be made - abort"
# ONE invocation, ONE read (round-2 finding 1, applied here): hashing with
# sha256sum and then reopening the file in python are two lookups, and a
# replacement in between would let unverified values be trusted. The bytes that
# are hashed are the bytes that are parsed.
python3 - "$PINNED_CONTROL_MANIFEST" "$PINNED_CONTROL_MANIFEST_SHA256" \
         "$EXP11DIR/arm_launch_registry.json" "$PINNED_TORCH_VERSION" \
         "$PINNED_PL_VERSION" "$PINNED_VAE_SHA256" \
  <<'PY' || die "control env cross-check FAILED - abort"
# --- BEGIN control-env-python (guard-tested by yaw_aug_train_guardtests.sh) ---
import hashlib, json, re, sys
path, pinned_sha, registry, want_torch, want_pl, want_vae = sys.argv[1:7]

# The reviewed constant is the anchor; exp_11's live registry must AGREE with it
# before it is allowed to vouch for anything.
with open(registry, "rb") as fh:
    reg_sha = json.loads(fh.read())["arms"]["VANL"]["manifest_sha256"]
if reg_sha != pinned_sha:
    sys.exit(f"exp_11's registry now records manifest_sha256 {reg_sha} for VANL, "
             f"but the reviewed pin is {pinned_sha}: the control's identity moved")

with open(path, "rb") as fh:
    raw = fh.read()                      # the ONLY read
got = hashlib.sha256(raw).hexdigest()
if got != pinned_sha:
    sys.exit(f"control manifest sha256 {got} != the pin {pinned_sha}")

text = raw.decode()                      # ...parsed from those same bytes
bad = []
m = re.search(r'^torch_version (\S+)', text, re.M)
if not m:
    bad.append("no 'torch_version' line in the control manifest")
elif m.group(1) != want_torch:
    bad.append(f"control trained on torch {m.group(1)}, we pin {want_torch}")
m = re.search(r'^vae_sha256 (\S+)', text, re.M)
if not m:
    bad.append("no 'vae_sha256' line in the control manifest")
elif m.group(1) != want_vae:
    bad.append(f"control used VAE {m.group(1)}, we pin {want_vae}")
# exp_11's manifest records torch/cuda/driver but not the PL version; the PL pin
# is enforced against the live import above. Say so rather than imply coverage.
if bad:
    sys.exit("CONTROL ENV CROSS-CHECK: " + "; ".join(bad))
print(f"control env cross-check OK: manifest sha {got} matches the reviewed pin AND "
      f"exp_11's live registry; torch and VAE agree (PL {want_pl} is gated against "
      "the live import; the control manifest does not record it)")
# --- END control-env-python ---
PY

# --- J. per-GPU free-VRAM gate (floor from the pins) --------------------------
for U in "${UUIDS[@]}"; do
  FREE="$(nvidia-smi --id="$U" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | tr -dc '0-9')"
  [ -n "$FREE" ] || die "nvidia-smi free-mem query failed on ${U} - refusing to launch blind"
  [ "$FREE" -ge "$MIN_FREE_MB" ] || die "GPU ${U} free ${FREE} MiB < required ${MIN_FREE_MB} MiB - refusing to launch"
done
echo "--- co-tenancy disclosure at launch ---"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true

# --- K. exclusive run ownership via flock (round-3 B3 residual) --------------
# mkdir + stale recovery had two races: a contender could arrive between mkdir
# and the owner write, and release removed the directory without checking whose
# it was. flock has neither: the kernel holds the lock while the fd is open and
# releases it on close (including on kill), so there is no stale state to
# recover and no recovery path to get wrong.
mkdir -p "$OUTPUT_ROOT" || die "could not create ${OUTPUT_ROOT} - abort" 3
LOCKFILE="${OUTPUT_ROOT}/exp15_${ARM}.lock"
exec 9>"$LOCKFILE" || die "could not open the lock file ${LOCKFILE} - abort" 3
if ! flock -n 9; then
  OWNER="$(tr '\n' ' ' < "$LOCKFILE" 2>/dev/null)"
  die "arm ${ARM} is locked by another live job (${OWNER:-<no metadata>}) - refusing a concurrent writer - abort"
fi
LAUNCH_UUID="$(cat /proc/sys/kernel/random/uuid)"
{ echo "job ${SLURM_JOB_ID}"; echo "uuid ${LAUNCH_UUID}"; echo "arm ${ARM}"; echo "mode ${MODE}"; echo "acquired ${TS}"; } >&9 \
  || die "could not write the lock owner metadata - abort" 3
echo "lock acquired: ${LOCKFILE} (flock on fd 9, released on exit)"
mkdir -p "$SAVEDIR" || die "could not create ${SAVEDIR} - abort" 3

# --- L. RESTART preflight (round-3 B2) ---------------------------------------
CKPT_SHA=""
if [ "$MODE" = "RESTART" ]; then
  PRE_ARGS=(--ckpt "$RESUME_CKPT" --expected-step "$EXPECTED_STEP" --config "$MODEL_CONFIG_ABS"
            --max-steps "$MAXSTEPS" --arm "$ARM" --rung "$RUNG" --commit "$HEAD_SHA")
  [ -n "$LAUNCH_MANIFEST_LINK" ] && PRE_ARGS+=(--launch-manifest "$LAUNCH_MANIFEST_LINK")
  # exp_11's --extension contract is deliberately NOT used: it exists to let a
  # restart raise the budget past the registered comparison point, which exp_15
  # forbids (the endpoint is pre-registered at 40k). A crash restart here keeps
  # the ORDINARY contract — same budget, same commit, same config, bound to this
  # arm's own launch manifest — which is the stricter of the two.
  PRE_OUT="$(python3 "$EXP11DIR/fa_orbit_ckpt_preflight.py" "${PRE_ARGS[@]}" 2>&1)"
  echo "$PRE_OUT"
  echo "$PRE_OUT" | grep -q "^CKPT_SHA256 " || die "restart preflight FAILED - abort"
  CKPT_SHA="$(echo "$PRE_OUT" | awk '/^CKPT_SHA256 /{print $2}')"
fi

# --- M. wandb: scrub, pin the destination, fix the run id (round-3 B7) --------
unset WANDB_MODE WANDB_DISABLED WANDB_ENTITY WANDB_RUN_ID WANDB_RESUME WANDB_DIR WANDB_PROJECT WANDB_NAME
export WANDB_DIR="$REPO/$SAVEDIR"
export WANDB_MODE=online
eval "$(grep -E '^[[:space:]]*export[[:space:]]+WANDB_API_KEY=' ~/.bashrc 2>/dev/null | tail -1)"
WANDB_ENTITY_SEEN="$(python3 - <<'PY'
import sys
try:
    import wandb
    v = wandb.Api().viewer
except Exception as e:
    sys.exit(f"wandb identity check FAILED: {e}")
if v.email != "yh4742@princeton.edu":
    sys.exit(f"wandb identity {v.email} != yh4742@princeton.edu")
print(v.entity)
PY
)" || die "wandb identity gate FAILED (no logger fallback: the arms train with wandb) - abort"
[ -n "$WANDB_ENTITY_SEEN" ] || die "wandb returned an empty entity - abort"
# B7 residual: pin the destination account explicitly instead of leaving it implicit
export WANDB_ENTITY="$WANDB_ENTITY_SEEN"
echo "wandb identity OK: entity ${WANDB_ENTITY} (exported), mode ${WANDB_MODE}"
if [ "$MODE" = "RESTART" ]; then
  WANDB_RUN_ID="$(awk '/^wandb_run_id /{print $2}' "$LAUNCH_MANIFEST_LINK" 2>/dev/null)"
  [ -n "$WANDB_RUN_ID" ] || die "RESTART needs the original wandb run id from ${LAUNCH_MANIFEST_LINK} - abort"
  export WANDB_RUN_ID WANDB_RESUME=must
  echo "wandb: resuming run ${WANDB_RUN_ID} (resume=must)"
else
  WANDB_RUN_ID="exp15-${ARM}-$(date +%s%N)-$(cut -c1-8 /proc/sys/kernel/random/uuid)"
  export WANDB_RUN_ID
  echo "wandb: new run id ${WANDB_RUN_ID}"
fi

# --- N. DINOv3 pin + architecture identity ------------------------------------
# exp_11 ran assert_arm_configs_exp11.py here: it built every arm and compared
# each one's initial state hash to the C4L bridge, proving the four arms differed
# only in their orbit. exp_15 has ONE arm, so there is no cross-arm hash to take
# — and the property that gate existed to establish is already proved more
# strongly upstream: FLAC_AR_YAWAUG.json is byte-for-byte FLAC_AR_VANCKPT.json
# plus the training.yaw_aug block (src/tests/test_yaw_aug_arm_config.py asserts
# exactly that, at byte level), so the architecture the control trained and the
# architecture this arm trains are the same object by construction. The ViT pin
# half of the gate survives as a literal config assertion in gate D above.
echo "ViT pin + architecture identity: asserted in gate D (config) and by the byte-level"
echo "  single-delta test against ${EXP11DIR#$REPO/}/FLAC_AR_VANCKPT.json"

# --- O. atomic manifest, duplicated to the save-dir (round-3 B5) --------------
# --- the live transcript must not be a TRACKED file while it is being written --
# A tracked file that a running job appends to is a file git will happily
# replace. `git rebase --autostash` (remote coordination between machines) stashes
# the dirty working tree, checks out, and restores — and each of those steps
# UNLINKS the path and writes a new inode. The job's stdout descriptor still
# points at the old, now-nameless inode, so every subsequent line is written to a
# file with no name and the visible transcript freezes forever. That is exactly
# what happened to C16 (job 3648696): its transcript stops at Epoch 5 while the
# run went on to step 40000, and all four arm transcripts froze at the same
# instant, 02:04:07, during a rebase cycle.
#
# So: at launch, the job removes its OWN Slurm transcript from the index. The
# file stays on disk untouched; it is simply no longer something git will move.
# The operator commits completed transcripts at closure (git add -f).
SLURM_OUT_AT_LAUNCH="$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null \
                        | tr ' ' '\n' | awk -F= '$1=="StdOut"{print $2; exit}')"
UNTRACK_STATE="not-attempted"
if [ -n "$SLURM_OUT_AT_LAUNCH" ]; then
  if git -C "$REPO" ls-files --error-unmatch "$SLURM_OUT_AT_LAUNCH" >/dev/null 2>&1; then
    if git -C "$REPO" rm --cached --quiet -- "$SLURM_OUT_AT_LAUNCH" 2>/dev/null; then
      UNTRACK_STATE="untracked-at-launch"
      echo "live transcript untracked for the duration of this run: ${SLURM_OUT_AT_LAUNCH}"
      echo "  (the file is untouched on disk; commit it at closure with git add -f)"
    else
      UNTRACK_STATE="untrack-FAILED"
      echo "WARNING: could not untrack ${SLURM_OUT_AT_LAUNCH}; a git working-tree"
      echo "         operation during this run can still detach the transcript"
    fi
  else
    UNTRACK_STATE="already-untracked"      # the normal steady state
  fi
else
  UNTRACK_STATE="stdout-path-unknown"
fi

TRAINLOG="${EXPDIR}/yaw_aug_${TS}_${ARM}_${RUNG}_jid${SLURM_JOB_ID}_train.log"
SAVEDIR_LOG="${SAVEDIR}/yaw_aug_${TS}_${ARM}_${RUNG}_jid${SLURM_JOB_ID}_train.log"
MANIFEST="${EXPDIR}/yaw_aug_${TS}_${ARM}_${RUNG}_jid${SLURM_JOB_ID}_manifest.txt"
# B5 residual: a failed environment dump must not be silently hashed into the record
PIPFREEZE_FILE="$(mktemp)" || die "mktemp failed - abort" 3
pip freeze > "$PIPFREEZE_FILE" 2>/dev/null || die "pip freeze FAILED - the environment digest would be a lie - abort" 7
ENV_SHA="$(sha256sum "$PIPFREEZE_FILE" | awk '{print $1}')"
rm -f "$PIPFREEZE_FILE"
{
  echo "# exp_15 arm launch manifest"
  echo "timestamp ${TS}"
  echo "job ${SLURM_JOB_ID} host $(hostname) mode ${MODE} launch_uuid ${LAUNCH_UUID}"
  echo "arm ${ARM} rung ${RUNG} micro ${MB} ngpu ${NGPU} max_steps ${MAXSTEPS} ckpt_every ${CHECKPOINT_EVERY}"
  echo "commit ${HEAD_SHA}"
  echo "p0_manifest_sha256 ${PINNED_P0_MANIFEST_SHA256}"
  echo "model_config ${MODEL_CONFIG_ABS}"
  echo "config_sha256 ${CONFIG_SHA}"
  echo "vae_sha256 ${VAE_SHA}"
  echo "env_pip_freeze_sha256 ${ENV_SHA}"
  echo "torch_version $(python3 -c 'import torch; print(torch.__version__)') cuda ${CUDA_VER} driver ${DRIVER}"
  echo "gpu_uuids ${UUID_CSV}"
  echo "time_limit ${TIME_LIMIT} min_free_mb ${MIN_FREE_MB}"
  echo "resume_ckpt ${RESUME_CKPT:-<none>} expected_step ${EXPECTED_STEP} resume_ckpt_sha256 ${CKPT_SHA:-<none>}"
  echo "save_dir ${SAVEDIR}"
  echo "slurm_transcript ${SLURM_OUT_AT_LAUNCH:-<unknown>} untrack ${UNTRACK_STATE}"
  echo "train_log ${TRAINLOG}"
  echo "train_log_copy ${SAVEDIR_LOG}"
  echo "wandb_entity ${WANDB_ENTITY_SEEN} wandb_project ${NAME} wandb_name ${EXPNAME}"
  echo "wandb_run_id ${WANDB_RUN_ID}"
  echo "command torchrun --standalone --nnodes=1 --nproc_per_node=${NGPU} train.py ${ARGV[*]}"
} > "${MANIFEST}.tmp" || die "manifest write FAILED - abort" 3
mv "${MANIFEST}.tmp" "$MANIFEST" || die "manifest publication FAILED - abort" 3
cp "$MANIFEST" "${SAVEDIR}/$(basename "$MANIFEST")" || die "manifest copy to the save-dir FAILED - abort" 3
[ "$MODE" = "INITIAL" ] && { cp "$MANIFEST" "$LAUNCH_MANIFEST_LINK" || die "launch-manifest link write FAILED - abort" 3; }
echo "manifest: ${MANIFEST} (copied to ${SAVEDIR})"

# --- O2. launch registry (plan §6.6) ------------------------------------------
# exp_11's registry pinned each arm's launch but never backfilled
# final_ckpt_sha256 for VANL — which is precisely why exp_15 had to write its own
# control-admission record in round 2. That omission is not repeated: the INITIAL
# launch registers here, fail-closed (a job that cannot be recorded does not
# run), and the completion block below writes the audited final checkpoint hash.
# A SMOKE registers in its OWN file (review finding 1). The plan's sequence is
# smoke-then-launch, and a smoke that claimed arms.YAWAUG in the production
# registry would make the real INITIAL abort as "already registered" — the gate
# meant to protect the launch would be the thing that blocked it. The smoke still
# exercises this code path, in its own namespace, which is the point of a smoke.
if [ "$SMOKE" = "1" ]; then
  LAUNCH_REGISTRY="${EXPDIR}/yaw_aug_smoke_registry.json"
else
  LAUNCH_REGISTRY="${EXPDIR}/yaw_aug_launch_registry.json"
fi
if [ "$MODE" = "INITIAL" ]; then
  python3 - "$LAUNCH_REGISTRY" "$ARM" "$MANIFEST" "$SLURM_JOB_ID" "$LAUNCH_UUID" "$HEAD_SHA" \
           "$RUNG" "$MAXSTEPS" "$CONFIG_SHA" "$VAE_SHA" "$SAVEDIR" "$WANDB_RUN_ID" \
    <<'PY' || die "launch registry write FAILED — refusing to run an unregistered job - abort" 3
# --- BEGIN registry-init-python (guard-tested by yaw_aug_train_guardtests.sh) ---
import hashlib, json, os, sys
(path, arm, manifest, job, uuid, commit, rung, max_steps, config_sha,
 vae_sha, save_dir, wandb_run_id) = sys.argv[1:13]
reg = json.load(open(path)) if os.path.exists(path) else {"arms": {}, "restarts": {}}
reg.setdefault("arms", {}); reg.setdefault("restarts", {})
if arm in reg["arms"]:
    sys.exit(f"{arm} is already registered in {path}: an INITIAL launch never "
             "overwrites a registered launch")
reg["arms"][arm] = {
    "manifest_path": manifest,
    "manifest_sha256": hashlib.sha256(open(manifest, "rb").read()).hexdigest(),
    "job": job, "mode": "INITIAL", "launch_uuid": uuid, "commit": commit,
    "rung": rung, "max_steps": max_steps, "config_sha256": config_sha,
    "vae_sha256": vae_sha, "save_dir": save_dir, "training_seed": 42,
    "wandb_run_id": wandb_run_id,
    "yaw_aug": {"enabled": True, "img_w": 512, "seed": 42},
    "final_ckpt_sha256": None, "final_step": None,
}
tmp = path + ".tmp"
json.dump(reg, open(tmp, "w"), indent=2, sort_keys=True)
os.replace(tmp, path)
print(f"registered {arm} in {path} (job {job}, uuid {uuid})")
# --- END registry-init-python ---
PY
fi

# --- P. training: torchrun + FIFO tee (both statuses captured) ----------------
unset SLURM_NTASKS SLURM_JOB_NAME SLURM_PROCID SLURM_LOCALID SLURM_NODEID
: > "$TRAINLOG" || die "could not create ${TRAINLOG} - abort" 3
: > "$SAVEDIR_LOG" || die "could not create ${SAVEDIR_LOG} - abort" 3
FIFO="$(mktemp)" || die "mktemp failed - abort" 3
rm -f "$FIFO" && mkfifo "$FIFO" || die "mkfifo failed - abort" 3
trap 'rm -f "$FIFO"' EXIT
tee -a "$TRAINLOG" "$SAVEDIR_LOG" < "$FIFO" &
TEE_PID=$!
echo "=== launching ${ARM} ${RUNG}: torchrun --standalone --nproc_per_node=${NGPU} train.py (${MODE}) ==="
torchrun --standalone --nnodes=1 --nproc_per_node="$NGPU" train.py "${ARGV[@]}" > "$FIFO" 2>&1 &
TR_PID=$!

# world-size watcher: terminate the moment Lightning reports the WRONG rank count
WORLD_RE="All distributed processes registered\. Starting with [0-9]+ processes"
(
  while kill -0 "$TR_PID" 2>/dev/null; do
    if grep -Eq "$WORLD_RE" "$TRAINLOG" 2>/dev/null; then
      GOT="$(grep -Eo "$WORLD_RE" "$TRAINLOG" | head -1 | grep -Eo '[0-9]+')"
      if [ "$GOT" != "$NGPU" ]; then
        echo "WORLD-SIZE WATCHER: Lightning started with ${GOT} processes, expected ${NGPU} — terminating torchrun"
        pkill -TERM -P "$TR_PID" 2>/dev/null; kill -TERM "$TR_PID" 2>/dev/null
      fi
      exit 0
    fi
    sleep 5
  done
) &
WATCHER=$!

# --- P2. post-launch yaw_aug banner gate (plan §6.6) --------------------------
# The treatment is invisible in the argv: whether this run is augmented at all is
# decided inside the training wrapper. The wrapper announces it once on rank 0 at
# fit start, and this watcher requires that line to appear BEFORE the first
# training step. If steps begin without it, the run is a vanilla run wearing the
# YAWAUG name — the single worst outcome available, because it would look
# perfectly healthy and produce a full set of numbers — so the watcher terminates
# torchrun and the job exits class 8.
BANNER="yaw_aug ENABLED img_w=${PINNED_YAW_AUG_IMG_W} seed=${PINNED_YAW_AUG_SEED}"
BANNER_STEP_RE="Epoch [0-9]+:"     # PL's progress bar: training batches have begun
BANNER_STATUS="$(mktemp)" || die "mktemp failed - abort" 3
printf 'PENDING\n' > "$BANNER_STATUS"
# --- BEGIN banner-watcher-helper (guard-tested by yaw_aug_train_guardtests.sh) --
yaw_aug_banner_verdict() {   # <log> <banner> <step-regex> -> OK | MISSING | PENDING
  local log="$1" banner="$2" step_re="$3"
  local b s
  # -F -x: the banner must be a WHOLE LINE. Substring matching would accept
  # "yaw_aug ENABLED img_w=512 seed=420" for a seed-42 run, or any line that
  # merely quotes the banner (this log also carries our own echoes of it).
  b="$(grep -nFx -m1 -- "$banner" "$log" 2>/dev/null | cut -d: -f1)"
  s="$(grep -nE -m1 -- "$step_re" "$log" 2>/dev/null | cut -d: -f1)"
  # ORDER matters, not mere presence: the wrapper prints at on_fit_start, before
  # any step. A banner appearing AFTER step evidence did not gate anything.
  if [ -n "$b" ] && { [ -z "$s" ] || [ "$b" -lt "$s" ]; }; then echo "OK"; return 0; fi
  if [ -n "$s" ]; then echo "MISSING"; return 1; fi
  echo "PENDING"; return 2
}
# --- END banner-watcher-helper -------------------------------------------------
(
  while kill -0 "$TR_PID" 2>/dev/null; do
    VERDICT="$(yaw_aug_banner_verdict "$TRAINLOG" "$BANNER" "$BANNER_STEP_RE")"
    case "$VERDICT" in
      OK)
        printf 'OK\n' > "$BANNER_STATUS"
        echo "YAW_AUG BANNER GATE: '${BANNER}' observed before the first step"
        exit 0 ;;
      MISSING)
        printf 'MISSING\n' > "$BANNER_STATUS"
        echo "YAW_AUG BANNER GATE: training steps began WITHOUT '${BANNER}' — this run is NOT augmented — terminating torchrun"
        pkill -TERM -P "$TR_PID" 2>/dev/null; kill -TERM "$TR_PID" 2>/dev/null
        exit 1 ;;
    esac
    sleep 5
  done
) &
BANNER_WATCHER=$!

wait "$TR_PID"; rc=$?
kill "$WATCHER" 2>/dev/null; wait "$WATCHER" 2>/dev/null
kill "$BANNER_WATCHER" 2>/dev/null; wait "$BANNER_WATCHER" 2>/dev/null
wait "$TEE_PID"; tee_rc=$?
rm -f "$FIFO"
# Re-read the log once the run is over: a run that died before its first step
# leaves the watcher PENDING, and "we never got far enough to tell" must not be
# reported as "the treatment was on".
BANNER_VERDICT="$(cat "$BANNER_STATUS" 2>/dev/null || echo PENDING)"
if [ "$BANNER_VERDICT" != "OK" ]; then
  BANNER_VERDICT="$(yaw_aug_banner_verdict "$TRAINLOG" "$BANNER" "$BANNER_STEP_RE")"
fi
rm -f "$BANNER_STATUS"
echo "yaw_aug banner verdict: ${BANNER_VERDICT}"

# --- Q. W&B run identity verification (round-3 B7 residual) ------------------
# The gate above proves WHO we are; this proves WHERE the run actually landed.
# train.py:165 builds WandbLogger(project=, name=) with NO save_dir, so PL passes
# its default save_dir='.' into wandb.init and that OVERRIDES the exported
# WANDB_DIR: in job 3646734 the run went to $REPO/wandb/run-<ts>-<id> while this
# check looked under $WANDB_DIR/wandb and found nothing (training was green, the
# job still classified 7). We keep exporting WANDB_DIR — other wandb artifacts do
# respect it — but locate the run by the collision-proof id WE generated, which
# wandb embeds in the directory name, across both candidate roots. Exactly one
# match is required; a mismatch is a provenance failure (class 7), not a footnote.
WANDB_CHECK_RC=0
python3 "$EXP11DIR/fa_orbit_wandb_readback.py" --run-id "$WANDB_RUN_ID" \
  --root "$REPO" --root "$WANDB_DIR" \
  --entity "$WANDB_ENTITY" --project "$NAME" --name "$EXPNAME" || WANDB_CHECK_RC=$?
if [ "$WANDB_CHECK_RC" -ne 0 ]; then
  echo "W&B run identity could not be verified against the manifest - provenance failure"
fi

# --- R. classification (tested taxonomy; see fa_orbit_classify.py) ------------
CLASSIFY_OUT="$(python3 "$EXP11DIR/fa_orbit_classify.py" --rc "$rc" --tee-rc "$tee_rc" --ngpu "$NGPU" \
  --maxsteps "$MAXSTEPS" --log "$TRAINLOG" --log-copy "$SAVEDIR_LOG" 2>&1)"
final_rc=$?
echo "$CLASSIFY_OUT"
LAST_CKPT="$(ls -1t "${RUNDIR}/checkpoints"/*.ckpt 2>/dev/null | head -1)"

# --- R2. AUDIT and record the final checkpoint (round-2 rigour, fail-closed) ---
# The gap exp_11 left for VANL (no final_ckpt_sha256, which cost exp_15 a whole
# round-2 admission recorder) is closed here — to the SAME standard. Round 2
# established what it takes to bind a checkpoint as evidence, and this block does
# not re-implement any of it: it IMPORTS yaw_aug_record_control.py and calls the
# reviewed primitives (snapshot_checkpoint's fd-pinned hash + safe mmap load +
# before/after inode identity, canonical_bytes' type-strict config comparison,
# summarize_ema's exact suffix/shape/dtype mirror and inventory digest).
#
# Fail-closed, class 9: a run whose 40k checkpoint cannot be produced AND audited
# has not met the launch acceptance criteria, and saying so at exit is the whole
# point — a green job with an unverified artifact is how exp_11's VANL became a
# checkpoint identified only by a path.
if [ "$SMOKE" = "1" ]; then
  echo "completion audit: skipped (SMOKE has no 40k endpoint; its registry is ${LAUNCH_REGISTRY})"
elif [ "$final_rc" -ne 0 ]; then
  echo "completion audit: skipped (the run did not finish cleanly: class ${final_rc})"
else
  # The checkpoint is selected by the endpoint it must BE, not by mtime: PL names
  # it epoch=<N>-step=<STEP>.ckpt, so exactly one file may match the pinned step.
  COMPLETION_RC=0
  python3 - "$EXPDIR/yaw_aug_record_control.py" "$LAUNCH_REGISTRY" "$ARM" \
           "${RUNDIR}/checkpoints" "$MAXSTEPS" "$MODEL_CONFIG_ABS" <<'PY' || COMPLETION_RC=$?
# --- BEGIN completion-audit-python (guard-tested by yaw_aug_train_guardtests.sh) ---
import glob, importlib.util, json, os, sys

recorder_path, registry, arm, ckpt_dir, max_steps, config_path = sys.argv[1:7]
max_steps = int(max_steps)

spec = importlib.util.spec_from_file_location("yaw_aug_record_control", recorder_path)
rc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rc)          # the round-2 recorder, imported not copied

matches = sorted(glob.glob(os.path.join(ckpt_dir, f"*-step={max_steps}.ckpt")))
if len(matches) != 1:
    sys.exit(f"expected exactly ONE *-step={max_steps}.ckpt in {ckpt_dir}, found "
             f"{len(matches)}: {matches}")
ckpt = matches[0]

checkpoint, digest, identity = rc.snapshot_checkpoint(ckpt)

step = checkpoint.get("global_step")
if isinstance(step, bool) or not isinstance(step, int):
    sys.exit(f"{ckpt}: global_step is {step!r} ({type(step).__name__}), expected a plain int")
if step != max_steps:
    sys.exit(f"{ckpt}: embedded global_step {step} != the pinned endpoint {max_steps}")

if "model_config" not in checkpoint:
    sys.exit(f"{ckpt}: no embedded model_config — the checkpoint cannot be bound to "
             "the config it was trained with")
with open(config_path, "rb") as fh:
    want_cfg = json.loads(fh.read())
if rc.canonical_bytes(checkpoint["model_config"]) != rc.canonical_bytes(want_cfg):
    sys.exit(f"{ckpt}: the embedded model_config is not {config_path} — this "
             "checkpoint was not trained with the arm config")
if checkpoint["model_config"].get("training", {}).get("yaw_aug") != \
        {"enabled": True, "img_w": 512, "seed": 42}:
    sys.exit(f"{ckpt}: the embedded config does not carry the registered yaw_aug block")

ema = rc.summarize_ema(checkpoint.get("state_dict") or {})

reg = json.load(open(registry))
entry = reg["arms"][arm]
if entry.get("final_ckpt_sha256") not in (None, digest):
    sys.exit(f"{arm} already carries a different final_ckpt_sha256 — refusing to overwrite")
entry["final_ckpt_sha256"] = digest
entry["final_step"] = step                    # from the CHECKPOINT, not the request
entry["final_ckpt_path"] = ckpt
entry["final_ckpt_bytes"] = identity["bytes"]
entry["final_ckpt_audit"] = {
    "loaded_with": {"mmap": True, "map_location": "cpu", "weights_only": True},
    "embedded_config_canonical_sha256":
        rc.canonical_sha256(checkpoint["model_config"]),
    **ema,
}
tmp = registry + ".tmp"
with open(tmp, "w") as fh:
    json.dump(reg, fh, indent=2, sort_keys=True)
os.replace(tmp, registry)
print(f"completion audit OK: {os.path.basename(ckpt)} step {step} sha256 {digest}")
print(f"  EMA {ema['ema_key_count']} keys mirror {ema['online_model_key_count']} "
      f"online DiT keys; inventory {ema['ema_inventory_sha256']}")
print(f"registry: {arm} final_ckpt_sha256 recorded in {registry}")
# --- END completion-audit-python ---
PY
  if [ "$COMPLETION_RC" -ne 0 ]; then
    echo "COMPLETION AUDIT FAILED (rc ${COMPLETION_RC}): the 40k checkpoint was not"
    echo "  produced, not verifiable, or not recordable — this run does not meet the"
    echo "  launch acceptance criteria (class 9)"
    final_rc=9
  fi
fi

# --- S. durable final record (round-3 B5 residual) ---------------------------
# The classifier verdict and this summary must survive in BOTH copies; the tee
# status is captured (it used to be discarded) and a failure is class 7. The
# whole preflight transcript (this job's Slurm output so far) is copied into the
# save-dir too, so the durable record is not just torchrun's stdout.
FINAL_RECORD="$(
  echo "=== exp_15 ${ARM} ${RUNG} finished at $(date '+%Y-%m-%d %H:%M:%S') ==="
  echo "torchrun rc=${rc} tee rc=${tee_rc} wandb_identity_rc=${WANDB_CHECK_RC} classified rc=${final_rc}"
  echo "$CLASSIFY_OUT"
  echo "yaw_aug banner: ${BANNER_VERDICT} (class 8 if not OK)"
  echo "last checkpoint: ${LAST_CKPT:-<none>}"
  echo "restart: yaw_aug_submit.sh ${ARM} --resume ${LAST_CKPT:-<path>} --expected-step <step>"
)"
printf '%s\n' "$FINAL_RECORD" | tee -a "$TRAINLOG" "$SAVEDIR_LOG" > /dev/null
final_tee_rc="${PIPESTATUS[1]}"
# The transcript path is NOT guessed from the #SBATCH pattern (job 3646734 logged
# "slurm_train_job_3646734.out not found" because SLURM_JOB_NAME is unset inside
# the job): ask Slurm itself.
# --- BEGIN slurm-stdout-helper (guard-tested by exp_11) --
parse_slurm_stdout() {   # `scontrol show job` text on stdin -> the StdOut path
  tr ' ' '\n' | awk -F= '$1=="StdOut"{print $2; exit}'
}
slurm_stdout_path() {    # $1 = job id
  scontrol show job "$1" 2>/dev/null | parse_slurm_stdout
}
# --- END slurm-stdout-helper --------------------------------------------------
SLURM_OUT="$(slurm_stdout_path "$SLURM_JOB_ID")"
PREFLIGHT_COPY_RC=0
if [ -n "$SLURM_OUT" ] && [ -f "$SLURM_OUT" ]; then
  cp "$SLURM_OUT" "${SAVEDIR}/preflight_${TS}_jid${SLURM_JOB_ID}.out" || PREFLIGHT_COPY_RC=$?
  [ "$PREFLIGHT_COPY_RC" -eq 0 ] && echo "preflight transcript duplicated from ${SLURM_OUT}"
else
  echo "PREFLIGHT TRANSCRIPT: scontrol reported StdOut='${SLURM_OUT}' which is not a readable file"
  PREFLIGHT_COPY_RC=1
fi
if [ "$final_tee_rc" -ne 0 ] || [ "$PREFLIGHT_COPY_RC" -ne 0 ] || [ "$WANDB_CHECK_RC" -ne 0 ]; then
  echo "LOG-PROVENANCE: final record tee rc=${final_tee_rc}, preflight copy rc=${PREFLIGHT_COPY_RC}, wandb identity rc=${WANDB_CHECK_RC}"
  [ "$final_rc" -eq 0 ] && final_rc=7
fi
# exp_15 class 8: the run trained, but not as the treatment it claims to be. A
# complete, healthy, UNAUGMENTED run is the failure this experiment cannot afford
# to publish, so class 8 outranks a green classification.
#
# PENDING is NOT that (review finding 5). A job that died before on_fit_start —
# OOM, a bad allocation, an import error — has a real diagnosis already, and
# relabelling it "banner missing" would send the next reader hunting the
# augmentation for a fault that lives somewhere else. So: MISSING is always class
# 8; PENDING only turns an otherwise GREEN run into class 8 (a green run must
# have been augmented) and otherwise leaves the existing failure class intact.
if [ "$BANNER_VERDICT" = "MISSING" ]; then
  echo "YAW_AUG BANNER: training steps ran WITHOUT the banner — this run may not be"
  echo "  treated as YAWAUG (class 8)"
  final_rc=8
elif [ "$BANNER_VERDICT" != "OK" ]; then
  if [ "$final_rc" -eq 0 ]; then
    echo "YAW_AUG BANNER: verdict ${BANNER_VERDICT} on an otherwise COMPLETE run — the"
    echo "  augmentation was never announced by rank 0; this run may not be treated as YAWAUG (class 8)"
    final_rc=8
  else
    echo "YAW_AUG BANNER: verdict ${BANNER_VERDICT}; the run had already failed (class ${final_rc}),"
    echo "  which keeps its own taxonomy — the banner never had a chance to print"
  fi
fi
printf '%s\n' "$FINAL_RECORD"
exit "$final_rc"
