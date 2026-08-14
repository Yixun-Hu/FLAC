#!/usr/bin/env bash
# ============================================================================
# are_launch.sh - exp_16 (are_port) Phase 1: training launcher for the ARE-V
# arm, 2-GPU DDP + SyncBN. Mirrors exp_14's dsarm_launch.sh gate-for-gate.
#
# Seat: Opus 5 Coder (SOP §Roles); main session plans, codex MCP reviews.
#
# THE TREATMENT IS THE OBJECTIVE, AND IT IS A CONFIG KEY.
# FLAC's rectified flow normally learns  noise -> z.  ARE-V learns
#     noise -> (z - lambda * A(p))
# where A(p) is an analytic direct-sound anchor: a Hann-windowed sinc placed at
# t* = r/343*fs + delta_hat with amplitude A_g/r, pushed through the FROZEN VAE
# encoder's MEAN, minus Enc(0), truncated to latent frames 0-2 and gated to zero
# where the listener's depth panorama says the source bearing is occluded. At
# inference lambda*A_query is added back to the latent before the single decode.
#
# lambda is declared as `training.are_lambda` and the calibrated constants as
# `training.are_anchor`, both of which train.py embeds into every checkpoint - so
# the arm's objective is auditable after the fact instead of being an ambient
# property of whatever the code happened to be that week (announcement 06's rule,
# applied to the objective rather than to the chunk plan).
#
#   ARM=AREV   are_lambda 1.0, vanilla conditioning  (partner control: P1@40k,
#              already on record - its lambda is absent, i.e. today's objective)
#
# Everything else is pinned to the exp_07 B-Vp1 recipe and asserted below as a
# parsed-object identity: the arm config must be FLAC_AR_BVp1.json PLUS EXACTLY
# `training.are_lambda` and `training.are_anchor`. 32/GPU x 2 GPUs x accum 1
# (eff 64), SyncBN (BN=64), ViT grad-ckpt via config, seed 42, bf16-mixed (ini
# default), wandb, env flac.
#
# ---------------------------------------------------------------------------
# MODE IS EXPLICIT. It is NEVER inferred from MAXSTEPS (exp_14 r1 finding 2:
# step-count inference made a 16-step run an "ordinary" run whose artifact gate
# degraded to a warning, and re-labelled a nearly-finished restart a probe).
#
#   MODE=PROBE    short fit / plumbing probe. Its OWN save-dir and W&B identity
#                 (FLAC_exp16_AREV_probe / exp16_AREV_probe /
#                 outputs_FLAC/exp16_AREV_probe) - a probe NEVER writes into the
#                 production namespace. Defaults 15 steps, ckpt every 5, and its
#                 schedule is deliberately FREE. Requires no admission evidence:
#                 the probe is how evidence is EARNED.
#   MODE=FULL     the production scratch run, PINNED to exactly 40,000 steps on a
#                 2,500-step cadence. Production identity. Refuses to start if
#                 that namespace already holds checkpoints.
#   MODE=RESTART  crash restart of a FULL run: RESUME_CKPT + EXPECTED_STEP>0,
#                 inside this arm's own production save-dir, finishing at the
#                 same pinned endpoint on the same pinned cadence.
#
# THE SCHEDULE IS PINNED, NOT DEFAULTED (r1 review finding 1). Round 1 expressed
# 40,000/2,500 as overridable shell defaults, so `MODE=FULL MAXSTEPS=1000` passed
# every gate and produced a truncated arm with production provenance. The
# endpoint, the cadence and the post-run verdict now live in readback.py, which
# this script imports and src/tests/test_are_launch_schedule.py exercises.
#
# ADMISSION IS GATED, NOT ADVISORY. FULL and RESTART require stamped evidence,
# verified against the current source SHA, the current ARE-code fingerprint and
# the current arm-config hash by stamp_evidence.py::require_evidence:
#   * are_fit_AREV.json   (plan §4 fit probe)
# A missing, stale, mis-hashed or FAIL-verdict record is a hard abort. FULL and
# RESTART additionally refuse to run with a DIRTY treatment path, so the recorded
# SHA describes what actually runs. There is NO env bypass for that refusal
# (r1 review finding 3): the only thing that tolerates a dirty tree is
# ARE_GUARD_DRYRUN=1, which ALSO makes this script exit before train.py, so
# setting it can never produce a training run.
# exp_14 also required a cross-arm sequencing record; exp_16 Phase 1 is a SINGLE
# arm, so there is deliberately no second requirement to discharge.
#
# ARTIFACT READBACK, BOUND TO THIS INVOCATION. After any rc=0 run the newest
# checkpoint is reloaded and readback.py::readback_problems is applied to it: for
# FULL/RESTART it must prove global_step == 40,000, and in every mode the
# embedded are_lambda / are_anchor must equal this arm's. Only a checkpoint that
# did not exist before this launch and whose mtime is at or after this launch
# counts. A missing checkpoint is a hard abort in every mode.
#
# NOT the fit probe itself. MODE=PROBE is how the probe is run and
# stamp_evidence.py is how its verdict is recorded. The VRAM floor here is the
# campaign's standing co-tenancy number (21,900 MiB), measured on the FA arm and
# NOT requalified for ARE-V; a loud NOTE says so.
#
# Knobs (env): ARM MODE MAXSTEPS CHECKPOINT_EVERY EXPECTED_STEP RESUME_CKPT
#              LOGGER MB ACC MIN_FREE_MB MIN_FREE_DISK_MB
#              ARE_GUARD_DRYRUN (guard-suite only; see above - it cannot launch)
#
# Usage (fit/plumbing probe - earns the are_fit evidence):
#   ARM=AREV MODE=PROBE bash worklog/worklog_yixun/exp_16_are_port_claude/are_launch.sh
#   # then, only if it really passed:
#   python worklog/worklog_yixun/exp_16_are_port_claude/stamp_evidence.py \
#     --kind are_fit --arm AREV --verdict PASS --log <that run's log> --notes "..."
#
# Usage (production arm):
#   ARM=AREV MODE=FULL bash worklog/worklog_yixun/exp_16_are_port_claude/are_launch.sh
#
# Usage (crash restart):
#   ARM=AREV MODE=RESTART EXPECTED_STEP=20000 \
#   RESUME_CKPT=outputs_FLAC/exp16_AREV/FLAC_exp16_AREV/exp16_AREV/checkpoints/epoch=4-step=20000.ckpt \
#     bash worklog/worklog_yixun/exp_16_are_port_claude/are_launch.sh
# ============================================================================
set -uo pipefail
cd "$(git -C "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" rev-parse --show-toplevel)" || exit 3

EXPDIR07="worklog/worklog_yixun/exp_07_fa_scratch_claude"       # BVp1 reference + pin gate
EXPDIR16="worklog/worklog_yixun/exp_16_are_port_claude"         # arm config, evidence, logs
BASE_PATH="${EXPDIR07}/FLAC_AR_BVp1.json"                       # the single-delta base
LAM_KEY="are_lambda"
ANCHOR_KEY="are_anchor"
LOGGER="${LOGGER:-wandb}"
MB="${MB:-32}"; ACC="${ACC:-1}"
# GUARD SUITE ONLY. Setting this tolerates a dirty treatment tree AND forces this
# script to abort before train.py, so it can never produce a training run - which
# is what makes it safe to expose at all (r1 review finding 3 replaced
# ALLOW_DIRTY_TREATMENT, which relaxed the gate and then launched).
ARE_GUARD_DRYRUN="${ARE_GUARD_DRYRUN:-0}"

_posint() { # $1=name $2=value -> must be a positive integer
  case "$2" in ''|*[!0-9]*) echo "$1 must be a positive integer (got '$2') - abort"; return 1;; esac
  [ "$2" -gt 0 ] || { echo "$1 must be > 0 (got '$2') - abort"; return 1; }
}

# --- arm selection (fail-closed: there is no default arm) ---
ARM="${ARM:-}"
case "$ARM" in
  AREV) MODEL_CONFIG_PATH="${EXPDIR16}/FLAC_AR_ARE.json"; WANT_LAM="1.0";;
  *) echo "ARM must be exactly AREV (vanilla conditioning, are_lambda 1.0 - exp_16 Phase 1)"
     echo "  got ARM='${ARM}' - abort"
     echo "  (ARE-FA and ARE-CYL are Phases 2/3 and have no config in this experiment yet)"
     exit 2;;
esac

# --- mode selection (explicit; never inferred from the step count) ---
MODE="${MODE:-}"
case "$MODE" in
  PROBE)
    NAME="FLAC_exp16_${ARM}_probe"; EXPNAME="exp16_${ARM}_probe"
    SAVEDIR="outputs_FLAC/exp16_${ARM}_probe"
    DEFAULT_MAXSTEPS=15; DEFAULT_CKPT_EVERY=5; NEEDS_EVIDENCE=0;;
  FULL|RESTART)
    NAME="FLAC_exp16_${ARM}"; EXPNAME="exp16_${ARM}"
    SAVEDIR="outputs_FLAC/exp16_${ARM}"
    DEFAULT_MAXSTEPS=40000; DEFAULT_CKPT_EVERY=2500; NEEDS_EVIDENCE=1;;
  *) echo "MODE must be exactly one of PROBE (short fit/plumbing probe, its own namespace),"
     echo "  FULL (the 40,000-step production scratch run) or RESTART (crash restart of a FULL run)."
     echo "  got MODE='${MODE}' - abort"
     echo "  (the mode is never inferred from MAXSTEPS: that made a 16-step run an 'ordinary' run"
     echo "   whose artifact gate degraded to a warning, and re-labelled a nearly-finished restart"
     echo "   a probe - exp_14 r1 review finding 2)"
     exit 2;;
esac

# --- environment gate (plain `python` must not resolve to another env, and the
# --- PL version decides checkpoint/restore semantics) ---
[ "${CONDA_DEFAULT_ENV:-}" = "flac" ] || { echo "CONDA_DEFAULT_ENV must be 'flac' (got '${CONDA_DEFAULT_ENV:-<unset>}') - run 'conda activate flac' - abort"; exit 2; }
python - <<'PY' || { echo "environment gate FAILED (need pytorch_lightning 2.1.0) - abort"; exit 2; }
import sys
import pytorch_lightning as pl, torch
print(f"env gate: python {sys.version.split()[0]} | pytorch_lightning {pl.__version__} | torch {torch.__version__}")
sys.exit(0 if pl.__version__ == "2.1.0" else 2)
PY

# --- knob validation ---
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-$DEFAULT_CKPT_EVERY}"; _posint CHECKPOINT_EVERY "$CHECKPOINT_EVERY" || exit 2
MAXSTEPS="${MAXSTEPS:-$DEFAULT_MAXSTEPS}";                   _posint MAXSTEPS "$MAXSTEPS" || exit 2

# --- THE PINNED SCHEDULE (r1 review finding 1). FULL/RESTART may not run to a
# --- different endpoint or on a different cadence, whatever the caller asks for.
# --- The rule lives in readback.py so a test can exercise it; this is the only
# --- place it is applied to the invocation. ---
MODE="$MODE" MAXSTEPS="$MAXSTEPS" CHECKPOINT_EVERY="$CHECKPOINT_EVERY" EXPDIR16="$EXPDIR16" \
PYTHONDONTWRITEBYTECODE=1 python3 - <<'PY' || { echo "schedule gate FAILED - abort"; exit 2; }
import os, sys
sys.path.insert(0, os.path.join(os.getcwd(), os.environ["EXPDIR16"]))
from readback import pinned_schedule, schedule_problems

mode = os.environ["MODE"]
steps = int(os.environ["MAXSTEPS"]); every = int(os.environ["CHECKPOINT_EVERY"])
problems = schedule_problems(mode, steps, every)
for p in problems:
    print(f"  {p}")
pinned = pinned_schedule(mode)
if pinned is None:
    print(f"schedule OK ({mode}): FREE (probe) -> {steps} steps, ckpt every {every}")
else:
    print(f"schedule OK ({mode}): PINNED {pinned[0]} steps, ckpt every {pinned[1]}")
sys.exit(2 if problems else 0)
PY

# invariant (exp_07 review): accumulation never feeds BN statistics, so the BN=64
# mandate leaves exactly ONE legal rung - pinned literally (string equality;
# arithmetic like MB*2*ACC==64 is bypassable via bash integer overflow).
[ "$MB" = "32" ] && [ "$ACC" = "1" ] || { echo "only the BN-compliant rung MB=32 ACC=1 is allowed (got MB='${MB}' ACC='${ACC}') - abort"; exit 2; }

[ -f "$MODEL_CONFIG_PATH" ] || { echo "arm config not found: ${MODEL_CONFIG_PATH} - abort"; exit 2; }
[ -f "$BASE_PATH" ]         || { echo "BVp1 reference config not found: ${BASE_PATH} - abort"; exit 2; }

# --- resume arguments, per mode ---
RESUME_CKPT="${RESUME_CKPT:-}"
EXPECTED_STEP="${EXPECTED_STEP:-0}"
case "$EXPECTED_STEP" in ''|*[!0-9]*) echo "EXPECTED_STEP must be a non-negative integer (got '${EXPECTED_STEP}') - abort"; exit 2;; esac
SAVEDIR_REAL="$(realpath -m "$SAVEDIR")"

if [ "$MODE" = "RESTART" ]; then
  [ -n "$RESUME_CKPT" ] || { echo "MODE=RESTART requires RESUME_CKPT - abort"; exit 2; }
  [ "$EXPECTED_STEP" -gt 0 ] || { echo "MODE=RESTART requires EXPECTED_STEP > 0: a resume must state the step it claims to be at - abort"; exit 2; }
  [ -f "$RESUME_CKPT" ] || { echo "RESUME_CKPT not found: ${RESUME_CKPT} - abort"; exit 2; }
  RESUME_REAL="$(realpath -m "$RESUME_CKPT")"
  case "$RESUME_REAL" in
    "${SAVEDIR_REAL}"/*) ;;
    *) echo "an ${ARM} RESTART may only resume a checkpoint written by THIS arm's FULL run, i.e. one inside"
       echo "  ${SAVEDIR_REAL}/"
       echo "  got: ${RESUME_REAL}"
       echo "  -> a PROBE has its own namespace and any other arm has a different objective;"
       echo "     crossing either would train a third method. abort"
       exit 2;;
  esac
  [ "$MAXSTEPS" -gt "$EXPECTED_STEP" ] || { echo "MAXSTEPS ${MAXSTEPS} must exceed the resume step ${EXPECTED_STEP} - abort"; exit 2; }
else
  [ -z "$RESUME_CKPT" ] || { echo "MODE=${MODE} is a from-scratch launch; RESUME_CKPT must be unset (use MODE=RESTART) - abort"; exit 2; }
  [ "$EXPECTED_STEP" -eq 0 ] || { echo "MODE=${MODE} is a from-scratch launch; EXPECTED_STEP must be unset (use MODE=RESTART) - abort"; exit 2; }
  # A scratch launch into a namespace that already holds checkpoints would mix two
  # lineages under one identity - and, for FULL, would silently restart the arm.
  EXISTING="$(find "$SAVEDIR" -name '*.ckpt' 2>/dev/null | head -3)"
  [ -z "$EXISTING" ] || {
    echo "MODE=${MODE} refuses to launch into a namespace that already holds checkpoints:"
    echo "$EXISTING" | sed 's/^/    /'
    echo "  -> to continue that run use MODE=RESTART; to redo it, archive ${SAVEDIR} first. abort"
    exit 2; }
fi

# --- checkpoint-cadence sanity: the post-run artifact readback is MANDATORY in
# --- every mode, so the run must be able to save at least once ---
FIRST_SAVE=$(( (EXPECTED_STEP / CHECKPOINT_EVERY + 1) * CHECKPOINT_EVERY ))
[ "$FIRST_SAVE" -le "$MAXSTEPS" ] || {
  echo "CHECKPOINT_EVERY=${CHECKPOINT_EVERY} would not save before MAXSTEPS=${MAXSTEPS}"
  echo "  (first save at global_step ${FIRST_SAVE}); the post-run embedded-lambda gate reads a"
  echo "  checkpoint written BY THIS INVOCATION, so it would have nothing to read. Lower"
  echo "  CHECKPOINT_EVERY. abort"
  exit 2; }

TS="$(date '+%Y-%m-%d_%H-%M-%S')"
LOG="${EXPDIR16}/are_port_${TS}_${EXPNAME}_train.log"

exec > >(tee -a "$LOG") 2>&1
echo "=== exp_16 are_port ${ARM} (${MODE}) DDP+SyncBN - ${TS} - $(git rev-parse --short HEAD 2>/dev/null) ==="
echo "identity: --name ${NAME} --experiment-name ${EXPNAME} --save-dir ${SAVEDIR}"
echo "recipe: ${MB}x2x${ACC} eff64 seed42 -> ${MAXSTEPS} | ckpt-every ${CHECKPOINT_EVERY} | logger=${LOGGER}"
if [ "$MODE" = "PROBE" ]; then
  echo "schedule: FREE (PROBE) - ${MAXSTEPS} steps, ckpt every ${CHECKPOINT_EVERY}"
else
  echo "schedule: PINNED (${MODE}) - endpoint ${MAXSTEPS} steps, ckpt every ${CHECKPOINT_EVERY}; the post-run readback requires global_step == ${MAXSTEPS}"
fi
echo "arm config: ${MODEL_CONFIG_PATH} (delta vs ${BASE_PATH}: training.${LAM_KEY}=${WANT_LAM} + training.${ANCHOR_KEY})"
echo "mode=${MODE} | resume: '${RESUME_CKPT}' | expected_step=${EXPECTED_STEP} | evidence_required=${NEEDS_EVIDENCE}"

# --- config contract (fail-closed, parsed objects, cheap so it runs FIRST):
# --- (1) the arm config IS FLAC_AR_BVp1.json plus EXACTLY the two ARE keys;
# --- (2) the rest of the pinned arm identity (vanilla conditioning, EMA, ckpt);
# --- (3) the factory really parses the keys into the wrapper kwargs;
# --- (4) the anchor's kept-frame window really holds AR's longest path. ---
are_config_gate() {
  ARM_CFG="$MODEL_CONFIG_PATH" BASE_CFG="$BASE_PATH" WANT_LAM="$WANT_LAM" \
  ARM_LABEL="$ARM" python3 - <<'PY'
import copy, json, os, sys
sys.path.insert(0, os.getcwd())          # beat any stale pip-installed src copy
from src.data import are_anchor as ar
from src.training.factory import _parse_are_config

arm_p, base_p = os.environ["ARM_CFG"], os.environ["BASE_CFG"]
want_lam = float(os.environ["WANT_LAM"]); label = os.environ["ARM_LABEL"]
LAM, ANCH = "are_lambda", "are_anchor"

arm, base = json.load(open(arm_p)), json.load(open(base_p))

# (1) POSITIVE: the declared lambda is this arm's lambda, typed float, and the
#     calibrated constants are typed floats. `1 == 1.0 == True` in Python, so the
#     type is checked before the value.
got = arm["training"].get(LAM)
if isinstance(got, bool) or not isinstance(got, float) or got != want_lam:
    sys.exit(f"{arm_p}: training.{LAM} is {got!r}, expected {want_lam!r} (a typed float)")
block = arm["training"].get(ANCH)
if not isinstance(block, dict):
    sys.exit(f"{arm_p}: training.{ANCH} is {block!r}, expected an object with the "
             "calibrated constants")
if sorted(block) != sorted(ar.ANCHOR_REQUIRED_KEYS):
    sys.exit(f"{arm_p}: training.{ANCH} keys {sorted(block)} != "
             f"{sorted(ar.ANCHOR_REQUIRED_KEYS)}")
for k, v in block.items():
    if isinstance(v, bool) or not isinstance(v, float):
        sys.exit(f"{arm_p}: training.{ANCH}.{k} is {v!r}, expected a typed float")

stripped = copy.deepcopy(arm)
del stripped["training"][LAM]
del stripped["training"][ANCH]
if stripped != base:
    sys.exit(f"{arm_p} differs from {base_p} somewhere OTHER than training.{LAM} / "
             f"training.{ANCH} (parsed-object mismatch) -> this is no longer a "
             "single-delta arm")

# (2) the rest of the pinned arm identity (announcement 05: the eval flag must
#     match how the checkpoint was trained, so the training method is asserted).
cond = arm["training"].get("cond_method", "vanilla")
if cond != "vanilla":
    sys.exit(f"{arm_p}: cond_method {cond!r} != 'vanilla' (ARE-FA is Phase 2)")
if arm["training"].get("use_ema") is not True:
    sys.exit(f"{arm_p}: use_ema must be true")
n_gc = sum(1 for c in arm["model"]["conditioning"]["configs"]
           if c.get("type") == "ViTCoordinates" and c["config"].get("gradient_checkpointing") is True)
if n_gc != 2:
    sys.exit(f"{arm_p}: expected 2 grad-checkpointed ViT conditioners, found {n_gc}")
if arm["model"]["diffusion"].get("diffusion_objective") != "rectified_flow":
    sys.exit(f"{arm_p}: ARE is a reparameterisation of the RECTIFIED-FLOW target; "
             f"got objective {arm['model']['diffusion'].get('diffusion_objective')!r}")

# (3) the keys really reach the wrapper kwargs (the config path, not a re-read).
kwargs = _parse_are_config(arm, arm["training"])
if kwargs.get(LAM) != want_lam:
    sys.exit(f"factory parse produced {kwargs.get(LAM)!r} for {LAM}, expected {want_lam}")
resolved = kwargs.get(ANCH) or {}
if resolved.get("sample_rate") != arm["sample_rate"] or resolved.get("sample_size") != arm["sample_size"]:
    sys.exit(f"factory did not thread the model time base into the anchor block: {resolved!r}")

# (4) THE KEPT-FRAME WINDOW, checked against AR's MEASURED worst case.
hop = arm["model"]["pretransform"]["config"]["downsampling_ratio"]
cfg = ar.anchor_config_from_dict(resolved, hop)
rep = ar.assert_worst_case_frames(cfg)
print(f"config contract OK ({label}): {arm_p} == {base_p} + training.{LAM}={want_lam} "
      f"+ training.{ANCH}={block}")
print(f"  ANCHOR: delta_hat={cfg.delta_hat} A_g={cfg.a_g:.6f} p_exp={cfg.p_exp} "
      f"H={cfg.kernel_half_width} LOS>={cfg.los_threshold}*r guard={cfg.dist_guard} m")
print(f"  TIME BASE: fs={cfg.sample_rate} sample_size={cfg.sample_size} hop={cfg.hop} "
      f"-> {cfg.latent_len} latent frames, keeping 0..{cfg.early_frames - 1}")
print(f"  AUGMENTATION: the AR train loader's RandomTimeShift moves the target by "
      f"0..{cfg.max_time_shift} samples; the per-sample draw is published as "
      "metadata['time_shift'] and ADDED to t* (fail-closed if absent)")
print(f"  WORST CASE (AR max {rep['max_distance']:.4f} m + max shift "
      f"{rep['max_time_shift']}): direct peak at sample "
      f"{rep['t_star']:.1f} (frame {rep['peak_frame']}), kernel tail to "
      f"{rep['last_sample']:.1f} -> needs {rep['required_frames']} frame(s) <= "
      f"{rep['early_frames']} kept  [headroom to "
      f"{rep['max_representable_distance']:.2f} m]")
PY
}
are_config_gate || { echo "config contract FAILED - abort"; exit 2; }

# --- ADMISSION EVIDENCE (FULL/RESTART only). Plan §4's fit probe is discharged
# --- by a stamped, hash-bound file, not by prose. A PROBE requires none: the
# --- probe is how evidence is EARNED. ---
if [ "$NEEDS_EVIDENCE" -eq 1 ]; then
  echo "--- admission evidence for ${ARM} (${MODE}) ---"
  # PYTHONDONTWRITEBYTECODE: importing stamp_evidence from the worklog directory
  # must not litter it with a __pycache__ that then shows up as an untracked
  # artifact of a launch.
  ARM_LABEL="$ARM" EXPDIR16="$EXPDIR16" GUARD_DRYRUN="$ARE_GUARD_DRYRUN" \
  PYTHONDONTWRITEBYTECODE=1 python3 - <<'PY' || { echo "ADMISSION EVIDENCE GATE FAILED - abort"; exit 2; }
import os, sys
sys.path.insert(0, os.path.join(os.getcwd(), os.environ["EXPDIR16"]))
from stamp_evidence import (REQUIRED, dirty_treatment_paths, require_evidence,
                            source_sha, treatment_fingerprint)

arm = os.environ["ARM_LABEL"]
root = os.getcwd()
print(f"  source_sha        : {source_sha(root)}")
print(f"  treatment_sha256  : {treatment_fingerprint(root)}")
print(f"  required          : {[f'{k}({s})' for k, s in REQUIRED[arm]]}")
ok, lines = require_evidence(arm, root)
for line in lines:
    print(line)

dirty = dirty_treatment_paths(root)
if dirty:
    print("  treatment paths are DIRTY: " + ", ".join(dirty))
    print("  -> the recorded source_sha does not describe what would run. Commit them "
          "(and re-stamp the evidence) before a production launch.")
    # The ONLY tolerance, and it is not a bypass: ARE_GUARD_DRYRUN also forces
    # this script to abort before train.py (see the dry-run stop below), so it
    # cannot be used to launch a run from a dirty tree - it can only prevent one.
    if os.environ.get("GUARD_DRYRUN") == "1":
        print("  ARE_GUARD_DRYRUN=1 -> tolerated for THIS GATE ONLY; this invocation is "
              "already committed to aborting before train.py, so no run can result")
    else:
        ok = False
else:
    print("  treatment paths clean (working tree == HEAD for the ARE files)")

sys.exit(0 if ok else 2)
PY
  echo "admission evidence OK"
else
  echo "--- admission evidence: NOT required in MODE=PROBE (a probe is how evidence is earned) ---"
fi

# --- disk floor on the outputs volume (each 2,500-step checkpoint is ~690 MB;
# --- 16 of them per arm). Bypass for guard-testing only: MIN_FREE_DISK_MB=1. ---
MIN_FREE_DISK_MB="${MIN_FREE_DISK_MB:-20480}"; _posint MIN_FREE_DISK_MB "$MIN_FREE_DISK_MB" || exit 2
DF_TARGET="outputs_FLAC"; [ -d "$DF_TARGET" ] || DF_TARGET="."
DISK_FREE_MB="$(df -P -B1M "$DF_TARGET" 2>/dev/null | awk 'NR==2{print $4}' | tr -dc '0-9')"
[ -n "$DISK_FREE_MB" ] || { echo "df query failed on ${DF_TARGET} - refusing to launch blind - abort"; exit 2; }
echo "disk: ${DISK_FREE_MB} MiB free on the volume holding ${DF_TARGET} (floor ${MIN_FREE_DISK_MB} MiB)"
[ "$DISK_FREE_MB" -ge "$MIN_FREE_DISK_MB" ] || { echo "free disk ${DISK_FREE_MB} MiB < required ${MIN_FREE_DISK_MB} MiB - refusing to launch"; exit 2; }

# --- RESTART only: the resume-safety gate, at exp_10/13/14 full-state strength.
# --- train.py rebuilds the wrapper from the CURRENT json BEFORE PL loads the
# --- checkpoint, so a lambda (or a re-calibrated delta_hat/A_g) edited between
# --- the crash and the restart would silently take effect. The embedded
# --- model_config must therefore equal this arm's current JSON under a
# --- TYPE-STRICT comparison - plain Python equality reads 1 == 1.0 == True, so a
# --- lambda serialised as 1 (int) or `true`, or an A_g that became an int, would
# --- pass a naive check and then take a different code path in _parse_are_config
# --- (which rejects bools and requires numbers). ---
if [ "$MODE" = "RESTART" ]; then
  RESUME_CKPT="$RESUME_CKPT" EXPECTED_STEP="$EXPECTED_STEP" ARM_CFG="$MODEL_CONFIG_PATH" \
  WANT_LAM="$WANT_LAM" EXPDIR16="$EXPDIR16" PYTHONDONTWRITEBYTECODE=1 \
  python3 - <<'PY' || { echo "resume-lineage check FAILED - abort"; exit 2; }
import json, os, sys, torch
sys.path.insert(0, os.path.join(os.getcwd(), os.environ["EXPDIR16"]))
from readback import type_strict_diff

p = os.environ["RESUME_CKPT"]; want = int(os.environ["EXPECTED_STEP"])
cfg_path = os.environ["ARM_CFG"]; want_lam = float(os.environ["WANT_LAM"])
LAM, ANCH = "are_lambda", "are_anchor"


def strict_diff(a, b):
    """The shared rule, labelled for THIS call site (checkpoint vs config)."""
    return type_strict_diff(a, b, label_a="the checkpoint", label_b="the config")


ck = torch.load(p, map_location="cpu", weights_only=False)
if not isinstance(ck, dict):
    sys.exit(f"not a Lightning checkpoint: {p}")
gs = ck.get("global_step")
if gs != want:
    sys.exit(f"global_step {gs} != expected {want} (set EXPECTED_STEP to resume a different step)")

mc = ck.get("model_config")
if not isinstance(mc, dict):
    sys.exit("checkpoint carries no embedded 'model_config' dict -> cannot prove which "
             "objective trained it; refusing to resume it under an assumed one")
want_cfg = json.load(open(cfg_path))
diff = strict_diff(mc, want_cfg)
if diff is not None:
    emb_lam = (mc.get("training") or {}).get(LAM, "<absent>")
    emb_anch = (mc.get("training") or {}).get(ANCH, "<absent>")
    sys.exit(f"embedded model_config != {cfg_path} (type-strict mismatch)\n  first difference: {diff}\n"
             f"  embedded training.{LAM}={emb_lam!r} training.{ANCH}={emb_anch!r}, current "
             f"{LAM}={want_lam!r}. train.py rebuilds the wrapper from the CURRENT json before "
             "loading the checkpoint, so resuming under a different config would silently "
             "change the objective mid-run")

# --- full-state shape checks (exp_10/13/14 level). exp_16's arm has no reset
# --- lineage: a cleared optimizer state, an absent param_group or a missing
# --- scheduler means the wrong file, not a recoverable run.
opts = ck.get("optimizer_states")
if not opts:
    sys.exit("no 'optimizer_states' -> weights-only ckpt; PL 2.1 KeyErrors on resume")
if len(opts) != 1:
    sys.exit(f"expected exactly 1 optimizer entry, found {len(opts)}")
groups = opts[0].get("param_groups")
if not groups:
    sys.exit("optimizer state has no 'param_groups' -> not a resumable optimizer state")
if len(groups) != 1:
    sys.exit(f"expected exactly 1 param_group, found {len(groups)}")
n_state = len(opts[0].get("state", {}))
if not n_state:
    sys.exit("optimizer state is CLEARED (stripped checkpoint) -> exp_16's arm is a WARM "
             "continuation; there is no optimizer-reset lineage in this experiment")
scheds = ck.get("lr_schedulers")
if not scheds:
    sys.exit("no 'lr_schedulers' -> PL 2.1 KeyErrors on resume")
s0 = scheds[0]
sd = ck.get("state_dict") or {}
n_ema = sum(1 for k in sd if k.startswith("diffusion_ema."))
if not n_ema:
    sys.exit("no EMA weights in state_dict (the arm config has use_ema true)")

lr = groups[0].get("lr")
print(f"resume lineage OK: {p}\n  global_step={gs} epoch={ck.get('epoch')} "
      f"optimizer_state=FULL ({n_state} entries) lr={lr} ema_entries={n_ema}")
print(f"  scheduler state: inv_gamma={s0.get('inv_gamma')} power={s0.get('power')} "
      f"warmup={s0.get('warmup')} last_epoch={s0.get('last_epoch')}")
print(f"  embedded model_config == {cfg_path} (type-strict), training.{LAM}={want_lam}")
PY
fi

# --- per-GPU FREE-VRAM gate (Yixun co-tenancy policy) ---
MIN_FREE_MB="${MIN_FREE_MB:-21900}"
cat <<'NOTE'
################################################################################
## NOTE - the VRAM floor below (21,900 MiB) is the campaign's STANDING         ##
## co-tenancy number, measured on the FA arm at cap 64. It is NOT requalified  ##
## for ARE-V, which has a different (vanilla, no-orbit) activation profile AND ##
## adds one B=1 frozen-VAE encode per sample inside training_step. Plan §4     ##
## gates this arm on a REAL 15-step DDP fit probe with VRAM sampled - run it   ##
## as MODE=PROBE and record its verdict with stamp_evidence.py. If it does not ##
## fit, STOP and report; do not lower the rung, which would break BN=64.       ##
################################################################################
NOTE
for G in 0 1; do
  FREE="$(nvidia-smi -i "$G" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | tr -dc '0-9')"
  rc_q=$?
  [ "$rc_q" -eq 0 ] && [ -n "$FREE" ] || { echo "nvidia-smi free-mem query failed on GPU ${G} (rc=${rc_q}) - refusing to launch blind"; exit 2; }
  [ "$FREE" -ge "$MIN_FREE_MB" ] || { echo "GPU ${G} free ${FREE} MiB < required ${MIN_FREE_MB} MiB - refusing to launch"; exit 2; }
done
echo "--- co-tenancy disclosure: compute apps at launch ---"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader 2>/dev/null || true

# --- THE DRY-RUN STOP (guard suite only; r1 review finding 3). Everything above
# --- has run for real. Nothing below ever will under this flag: the flag exists
# --- so the guard exercise can drive the gate chain to completion WITHOUT the
# --- launcher ever being able to start training, which is what lets the
# --- dirty-tree tolerance above be safe. A production caller who sets it gets no
# --- run at all - that is the point, not a side effect. ---
if [ "$ARE_GUARD_DRYRUN" = "1" ]; then
  echo "ARE_GUARD_DRYRUN=1 -> all pre-launch gates executed; refusing to launch training."
  echo "  (this flag exists for are_launch_guardtests.sh; it can only PREVENT a run)"
  exit 2
fi

# --- fail-closed wandb identity gate (only when wandb is requested) ---
if [ "$LOGGER" = "wandb" ]; then
  # ~/.bashrc's interactive guard blocks non-interactive sourcing, so extract the
  # newest exported key directly; the gate below re-verifies at every launch.
  eval "$(grep -E '^[[:space:]]*export[[:space:]]+WANDB_API_KEY=' ~/.bashrc 2>/dev/null | tail -1)"
  python - <<'PY'
import sys
try:
    import wandb
    email = wandb.Api().viewer.email
except Exception as e:
    print("wandb identity check FAILED:", e); sys.exit(1)
print("wandb identity:", email)
sys.exit(0 if email == "yh4742@princeton.edu" else 2)
PY
  gate=$?
  [ "$gate" -eq 0 ] || { echo "wandb identity != yh4742@princeton.edu (exit ${gate}) - set the right WANDB_API_KEY or use LOGGER=none - abort"; exit 2; }
fi

# --- DINOv3 pin + seeded init-identity (offline, fail-closed) ---
HF_HUB_OFFLINE=1 python "${EXPDIR07}/assert_arm_configs.py" || { echo "GATE FAILED - abort"; exit 1; }

cat <<'NOTE'
################################################################################
## NOTE - READ THE LINES ABOVE CAREFULLY.                                     ##
## assert_arm_configs.py is an exp_07 artifact. It builds FLAC_AR_BV.json and  ##
## FLAC_AR_BF.json - NOT this arm's config. It is kept for its DINOv3          ##
## initializer PIN + seeded init-identity check, which do apply: exp_16's arm  ##
## is FLAC_AR_BVp1.json plus two TRAINING keys, and BVp1 is FLAC_AR_BV.json    ##
## plus ViT gradient checkpointing, so the module tree and its initialization  ##
## are B-V's exactly. The authoritative exp_16 assertion is the config         ##
## contract re-run below.                                                     ##
################################################################################
NOTE
are_config_gate || { echo "config contract FAILED - abort"; exit 2; }

echo "--- env manifest ---"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader
pip freeze 2>/dev/null | sha256sum | awk '{print "pip-freeze sha256:", $1}'
echo "sync_batchnorm: true (fail-closed in train.py below num_gpus 2) | strategy: ddp_find_unused_parameters_true | rung: ${MB}x2x${ACC} | grad-ckpt: config"
echo "arm: ${ARM} | mode: ${MODE} | model-config: ${MODEL_CONFIG_PATH} | are_lambda: ${WANT_LAM}"

# --- pre-run checkpoint inventory: the readback below only accepts a file that
# --- is NOT in this list and whose mtime is at/after this launch, so a stale
# --- checkpoint can never discharge the artifact gate. ---
RUN_START_EPOCH="$(date '+%s')"
PRE_INVENTORY="$(mktemp)"
find "$SAVEDIR" -name '*.ckpt' 2>/dev/null | sort > "$PRE_INVENTORY"
echo "pre-run checkpoint inventory: $(wc -l < "$PRE_INVENTORY") file(s) under ${SAVEDIR} at epoch ${RUN_START_EPOCH}"

RESUME_ARGS=()
[ "$MODE" = "RESTART" ] && RESUME_ARGS=(--ckpt-path "$RESUME_CKPT")

HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0,1 python train.py \
  --model-config "$MODEL_CONFIG_PATH" \
  --dataset-config src/configs/dataset_configs/AR/train/acousticroom_train.json \
  --pretransform-ckpt-path weights/FLAC/VAE.safetensors \
  "${RESUME_ARGS[@]}" \
  --max-steps "$MAXSTEPS" --batch-size "$MB" --accum-batches "$ACC" --num-workers 6 --seed 42 \
  --num-gpus 2 --strategy ddp_find_unused_parameters_true --sync-batchnorm true \
  --logger "$LOGGER" --checkpoint-every "$CHECKPOINT_EVERY" \
  --name "$NAME" --experiment-name "$EXPNAME" \
  --save-dir "$SAVEDIR"
rc=$?
echo "=== exp_16 ${ARM} (${MODE}) exited rc=${rc} at $(date '+%Y-%m-%d %H:%M:%S') ==="

# ============================================================================
# Embedded-lambda readback - MANDATORY in every mode, bound to THIS invocation.
#
# The treatment is only real if it is IN THE ARTIFACT: train.py's
# ModelConfigEmbedderCallback writes model_config into every checkpoint, so a
# checkpoint written by this run is reloaded and its embedded are_lambda /
# are_anchor compared with this arm's. Candidates are restricted to files absent
# from the pre-run inventory AND with mtime >= the launch timestamp; a stale
# checkpoint from an earlier run in the same directory therefore cannot discharge
# the gate, and no checkpoint at all is a hard abort (not a warning) in every mode.
# ============================================================================
if [ "$rc" -eq 0 ]; then
  echo "--- embedded-lambda readback (mode=${MODE}, bound to this invocation) ---"
  SAVEDIR="$SAVEDIR" WANT_LAM="$WANT_LAM" ARM_CFG="$MODEL_CONFIG_PATH" \
  EXPECTED_STEP="$EXPECTED_STEP" MAXSTEPS="$MAXSTEPS" ARM_LABEL="$ARM" MODE="$MODE" \
  PRE_INVENTORY="$PRE_INVENTORY" RUN_START_EPOCH="$RUN_START_EPOCH" EXPDIR16="$EXPDIR16" \
  PYTHONDONTWRITEBYTECODE=1 \
  python3 - <<'PY' || { rm -f "$PRE_INVENTORY"; echo "embedded-lambda readback FAILED - abort"; exit 2; }
import glob, os, sys, torch
sys.path.insert(0, os.path.join(os.getcwd(), os.environ["EXPDIR16"]))
from readback import ENDPOINT_STEPS, load_json, readback_problems

savedir = os.environ["SAVEDIR"]; want_lam = float(os.environ["WANT_LAM"])
start = int(os.environ["EXPECTED_STEP"]); maxsteps = int(os.environ["MAXSTEPS"])
label = os.environ["ARM_LABEL"]; cfg_path = os.environ["ARM_CFG"]; mode = os.environ["MODE"]
run_start = float(os.environ["RUN_START_EPOCH"])
LAM, ANCH = "are_lambda", "are_anchor"

with open(os.environ["PRE_INVENTORY"]) as f:
    pre = {line.strip() for line in f if line.strip()}

allc = glob.glob(os.path.join(savedir, "**", "*.ckpt"), recursive=True)
fresh = [p for p in allc if p not in pre and os.path.getmtime(p) >= run_start - 1]
if not fresh:
    sys.exit(f"no checkpoint written by THIS invocation under {savedir} "
             f"({len(allc)} file(s) present, {len(pre)} of them pre-existing). The "
             "embedded-lambda gate only accepts an artifact this run produced, so there "
             "is nothing to verify -> the run cannot be admitted")
p = max(fresh, key=os.path.getmtime)
ck = torch.load(p, map_location="cpu", weights_only=False)
gs = ck.get("global_step")
mc = ck.get("model_config")
print(f"checkpoint written by this run: {p}\n  global_step={gs} "
      f"({len(fresh)} fresh of {len(allc)} present)")
if isinstance(mc, dict):
    training = mc.get("training") or {}
    print(f"  embedded training.{LAM} = {training.get(LAM, '<absent>')!r}  "
          f"(arm {label} declares {want_lam})")
    print(f"  embedded training.{ANCH} = {training.get(ANCH, '<absent>')!r}")
if mode != "PROBE":
    print(f"  endpoint requirement: global_step must be exactly {ENDPOINT_STEPS}")

# ONE implementation of the verdict, shared with src/tests/test_are_launch_schedule.py
fails = readback_problems(mode, mc, gs, load_json(cfg_path), want_lam,
                          expected_step=start, max_steps=maxsteps)
if fails:
    sys.exit("embedded-lambda readback FAILED:\n  - " + "\n  - ".join(fails))
print(f"embedded-lambda readback PASSED: the checkpoint this {mode} run wrote declares "
      f"training.{LAM}={want_lam} for arm {label} at global_step {gs}.")
if mode == "PROBE":
    print("PROBE complete. If - and only if - the fit/plumbing verdict is genuinely PASS, record "
          "it with:\n  python worklog/worklog_yixun/exp_16_are_port_claude/stamp_evidence.py "
          f"--kind are_fit --arm {label} --verdict PASS --log <this log> --notes '<VRAM, throughput>'")
PY
  echo "embedded-lambda readback OK"
fi
rm -f "$PRE_INVENTORY"

exit $rc
