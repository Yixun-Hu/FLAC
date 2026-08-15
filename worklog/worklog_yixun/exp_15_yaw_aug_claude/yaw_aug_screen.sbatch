#!/bin/bash
# ============================================================================
# exp_15 SCREEN DRIVER — one checkpoint, one arm, one yaw protocol (plan §4).
#
# Adapted from exp_14's yaw_gen_screen.sbatch (the verbatim copy is commit 1 of
# this round; exp_14's own kit stays read-only to this experiment). Yixun approved
# that base over the plan's §6.7 wording — exp_14's kit is review-closed, proven
# over a 106-cell campaign, and already implements the assignment-integrity
# contract exp_15 adopted verbatim (worklog 2026-08-15T14:45).
#
# TWO arms, both VANILLA-conditioned models, distinguished by what their TRAINING
# did:
#   ARM=YAWAUG   trained with per-visit random yaw augmentation (exp_15's chain)
#   ARM=VANL     exp_11's matched no-augmentation control at the same rung/seed
# Because both are vanilla, EVERY cell runs --cond-method vanilla. There is no
# orbit axis here; exp_14's TRAIN_ANGLES/EVAL_ORBIT machinery is deliberately gone
# rather than renamed, and an exported EVAL_ORBIT is refused outright.
#
# Three cell types, all at the ONE 40k endpoint:
#   CELL=tbl    the HEADLINE Table-1 cell: theta = 0, fixed mode. 2 arms x
#               K in {1,8} x seeds 42-46 = 20 cells. No eval-name suffix.
#   CELL=rrob   the ROBUSTNESS cell: every eval sample independently draws a
#               panorama-column yaw offset (--rotate-mode random --rotate-seed
#               $SEED; rotation seed = eval seed). Same 20-cell shape, so
#               Delta = m_R - m_T is seed-paired. Suffix _rotrand<SEED>.
#   CELL=vctl   the two pre-registered fixed-90-degree controls, s42 / K=8 only:
#               VANL@90 is gate G1's POSITIVE control and YAWAUG@90 is
#               DESCRIPTIVE ONLY (plan §5, review F3 — gating on it would
#               presuppose the augmentation succeeded). Every other tuple is
#               refused.
#
# Protocol constants, pinned identically for every cell (plan §4.2): the full
# published unseen split (6,337 items / 17 rooms, announcement 01), EMA weights,
# cfg 1.0, 1 diffusion step, bf16 conditioning autocast, --batch-size 64 and
# --num-workers 4 (exp_14's campaign values, kept identical so exp_15's VANL T
# rows stay comparable to exp_14's VANL Z rows as the pre-declared external
# check), plus --expected-stream-count 6337 and --record-stream so the §4.3
# assignment audit (.stream.json) lands beside every metrics JSON — that sidecar
# is what makes T<->R pairing and the cross-arm rotation match falsifiable rather
# than asserted. --record-per-scene is passed for EVERY cell too: the campaign's
# per-seed observation is the PER-SCENE mean over the split's 10 ROOM FAMILIES,
# so a cell without a `by_scene` block did not measure the estimand.
#
# The per-class PROTOCOL FLAGS are not written here. They are rendered by
# exp15_validate_cell.eval_argv_tail — the same function the pre-launch intent and
# the guard suite read — so the plan's literal argv exists in exactly one place
# instead of three that agree today (announcement 05).
#
# Fail-closed before spending the GPU: commit binding + tracked drift, exactly one
# visible L40, EXACTLY ONE checkpoint matching the step, and then the full
# CHECKPOINT ADMISSION gate (exp15_admit_ckpt.py, plan §5 G4) which re-derives the
# file's sha256, its embedded global_step, canonical embedded-config equality, the
# EMA<->online mirror and the arm's identity, and binds all of it to the committed
# admission record. Afterwards it writes the .screenmeta.json sidecar, VALIDATES
# the produced artifacts with exp15_validate_cell.py and emits one SCREENRESULT.
#
# WORKTREE-PINNED EXECUTION: code comes from a worktree pinned to the submission
# SHA, outputs go to the main tree. Development and measurement are therefore
# DECOUPLED — the main checkout may keep committing while jobs run, and source_sha
# still equals the pinned SHA. MEASURE_ROOT is REQUIRED for a real run.
#
# Submitted through yaw_aug_screen_submit.sh (which owns the pin/lease/hold
# transaction) or, for a whole wave, yaw_aug_submit_grid.sh.
#
# DRYRUN=1 (outside Slurm) runs every gate, prints the eval argv and exits.
# ============================================================================
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/slurm_screen_%x_%j.out

set -uo pipefail
MAIN_REPO=/n/fs/gatrdp/codespace/FLAC        # where OUTPUTS and logs always live
REPO="$MAIN_REPO"
# TEST HOOK (guard tests only; ignored inside a Slurm job, like the launcher's).
# The eval kit's seams are namespaced YAW_EVAL_* precisely so a leftover from the
# TRAINING kit (YAW_AUG_*) or from exp_14's (YAW_GEN_*) cannot be mistaken for one
# of ours — the submitters' allowlist refuses those outright.
if [ -n "${YAW_EVAL_REPO_OVERRIDE:-}" ] && [ -z "${SLURM_JOB_ID:-}" ]; then
  REPO="$YAW_EVAL_REPO_OVERRIDE"; MAIN_REPO="$YAW_EVAL_REPO_OVERRIDE"
fi
unset YAW_EVAL_REPO_OVERRIDE

# --- WORKTREE-PINNED MEASUREMENT ---------------------------------------------
# Development and measurement are DECOUPLED: this job reads all code, configs and
# validators from a git worktree pinned to the submission SHA, while writing every
# output into the MAIN tree. Nothing the main checkout does during the run can
# change what this job executed, so source_sha always equals the pinned SHA and
# there is no reason to freeze development while measurements are in flight.
# Prepare one with fa_orbit_measure_worktree.sh and export MEASURE_ROOT.
CODE_ROOT="$REPO"
if [ -n "${MEASURE_ROOT:-}" ]; then
  [ -d "$MEASURE_ROOT" ] || { echo "MEASURE_ROOT '${MEASURE_ROOT}' is not a directory - abort"; exit 2; }
  [ -e "$MEASURE_ROOT/.git" ] || { echo "MEASURE_ROOT '${MEASURE_ROOT}' is not a git worktree - abort"; exit 2; }
  # It must be a MANAGED, DETACHED worktree of THIS repository at the canonical
  # location — not, say, the mutable main checkout passed in by hand.
  WT_TOP="$(git -C "$MEASURE_ROOT" rev-parse --show-toplevel 2>/dev/null)"
  WT_COMMON="$(git -C "$MEASURE_ROOT" rev-parse --git-common-dir 2>/dev/null)"
  MAIN_COMMON="$(git -C "$MAIN_REPO" rev-parse --git-common-dir 2>/dev/null)"
  [ -n "$WT_TOP" ] && [ "$(readlink -f "$WT_TOP")" = "$(readlink -f "$MEASURE_ROOT")" ] \
    || { echo "MEASURE_ROOT is not the top level of its worktree ('${WT_TOP}') - abort"; exit 2; }
  [ -n "$WT_COMMON" ] && [ "$(readlink -f "$WT_COMMON")" = "$(readlink -f "$MAIN_COMMON")" ] \
    || { echo "MEASURE_ROOT belongs to another repository ('${WT_COMMON}') - abort"; exit 2; }
  case "$(readlink -f "$MEASURE_ROOT")/" in
    "$(readlink -f "$MAIN_REPO")/.measure_worktrees/"*) ;;
    *) echo "MEASURE_ROOT is outside the managed .measure_worktrees/ area - abort"; exit 2 ;;
  esac
  if git -C "$MEASURE_ROOT" symbolic-ref -q HEAD >/dev/null 2>&1; then
    echo "MEASURE_ROOT is on a BRANCH; a pinned measurement tree must be detached - abort"; exit 2
  fi
  CODE_ROOT="$MEASURE_ROOT"
elif [ -n "${SLURM_JOB_ID:-}" ]; then
  echo "MEASURE_ROOT is required for a real screen: prepare it with"
  echo "  MEASURE_ROOT=\$(bash ${MAIN_REPO}/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_measure_worktree.sh)"
  echo "so the job's code identity cannot move while it runs - abort"
  exit 2
fi
EXPDIR="$CODE_ROOT/worklog/worklog_yixun/exp_15_yaw_aug_claude"      # CODE (pinned)
# exp_11's folder is READ-ONLY PROVENANCE for this campaign: the shared
# measure-worktree store helper, the DINOv3 pin probe and the VANL control's model
# config live there and are referenced in place. Copying them would create a second
# copy of an audited artifact — i.e. a second thing that can drift out of agreement
# with the checkpoint it vouches for.
EXP11="$CODE_ROOT/worklog/worklog_yixun/exp_11_fa_orbit_claude"      # CODE (pinned, read-only)
LOGDIR="$MAIN_REPO/worklog/worklog_yixun/exp_15_yaw_aug_claude"      # LOGS (main tree)
cd "$CODE_ROOT" || exit 3
unset PYTHONPATH PYTHONOPTIMIZE
export PATH=/n/fs/gatrdp/envs/flac/bin:$PATH
export PYTHONNOUSERSITE=1
export HF_HOME=/n/fs/gatrdp/hf_cache
export HF_HUB_OFFLINE=1

DRYRUN="${DRYRUN:-0}"
SEED="${SEED:-42}"
K="${K:-8}"
# CELL selects the purpose-specific contract. exp_15 has exactly three, and the
# driver refuses every combination the plan does not pre-register — the point of
# a registered grid is that an unregistered cell cannot be produced by accident
# and then read as evidence. There is deliberately NO default: a cell type is a
# protocol choice (announcement 05), and defaulting one would let an omitted
# export decide the experiment.
CELL="${CELL:-}"
[ -n "$CELL" ] || { echo "CELL must be exported (tbl|rrob|vctl) - abort"; exit 2; }
OUTPUT_ROOT="${OUTPUT_ROOT:-$MAIN_REPO/outputs_FLAC}"
PRODUCTION_OUTPUT_ROOT="$MAIN_REPO/outputs_FLAC"
if [ -n "${SLURM_JOB_ID:-}" ]; then
  if [ -n "${OUTPUT_ROOT:-}" ] && [ "$OUTPUT_ROOT" != "$PRODUCTION_OUTPUT_ROOT" ]; then
    echo "ambient OUTPUT_ROOT='${OUTPUT_ROOT}' != '${PRODUCTION_OUTPUT_ROOT}' - abort"; exit 2
  fi
  OUTPUT_ROOT="$PRODUCTION_OUTPUT_ROOT"
fi
# TEST SEAM for the admission sources, under the same lock exp_14 put on its
# registry override: a synthetic admission record is honoured ONLY against a
# synthetic output root, and a Slurm run has already been forced onto the
# production root above. The check runs HERE, before any gate, so an attempt to
# aim it at the real tree is refused immediately rather than after the driver has
# hashed 350 MB of pinned weights — and so the refusal cannot be masked by an
# unrelated earlier failure.
ADMIT_SOURCE_ARGS=()
if [ -n "${YAW_EVAL_CONTROL:-}${YAW_EVAL_REGISTRY:-}" ]; then
  if [ "$OUTPUT_ROOT" = "$PRODUCTION_OUTPUT_ROOT" ]; then
    echo "YAW_EVAL_CONTROL / YAW_EVAL_REGISTRY may not override the committed admission records for the production output root - abort"
    exit 2
  fi
  [ -z "${YAW_EVAL_CONTROL:-}" ]  || ADMIT_SOURCE_ARGS+=(--control "$YAW_EVAL_CONTROL")
  [ -z "${YAW_EVAL_REGISTRY:-}" ] || ADMIT_SOURCE_ARGS+=(--registry "$YAW_EVAL_REGISTRY")
  echo "TEST MODE: admission sources overridden (synthetic output root ${OUTPUT_ROOT})"
fi
TS="$(date '+%Y-%m-%d_%H-%M-%S')"

# --- CAMPAIGN CONSTANTS (plan §4.2) -------------------------------------------
# Not defaults and not overridable: the batch size regroups the split's tail
# batch and the worker count decides which process serves which position, so a
# cell that differs in either is not paired with the rest of its block. They are
# LITERALS here and are recorded in every manifest; the validator rejects a cell
# whose manifest deviates.
#
# They are exp_14's campaign values, which were in turn exp_11's conf-cell values
# (eval_FLAC.py's own argparse defaults, passed EXPLICITLY here so the recipe no
# longer depends on a default). Keeping them identical is what lets exp_15's VANL
# T rows be checked against exp_14's VANL Z rows as an external reproduction
# check (plan §5) — a different batch size would make that comparison meaningless.
BATCH_SIZE=64
NUM_WORKERS=4
# The published unseen split: 6,337 items / 17 rooms (announcement 01), the same
# item set at K=1 and K=8 (the _1 config changes max_context only). Passed as
# --expected-stream-count so a subsampled or partly-failing split FAILS the cell
# instead of reporting a plausible number over fewer items.
EXPECTED_STREAM_COUNT=6337
# ...and its per-scene twin: 10 ROOM FAMILIES. The released metric callback groups
# on md['scene'], and AR_md.py sets that to the family (rel_path[-3], e.g. "Cafe")
# — the per-room id never reaches it. So the release convention's per-scene mean
# is over the split's 10 families; "6,337 items / 17 rooms" describes the split's
# CONTENT, not the grouping. The per-seed observation is that mean, so the
# grouping is a campaign constant: pinned here, recorded in the manifest, and
# checked per cell as an exact KEY SET.
EXPECTED_SCENES=10

die() { echo "$1"; exit "${2:-2}"; }

# --- A. parameters ------------------------------------------------------------
[ -n "${ARM:-}" ] || die "ARM must be exported (YAWAUG|VANL) - abort"
[ -n "${STEP:-}" ] || die "STEP must be exported - abort"
[ -n "${EXPECT_SHA:-}" ] || die "EXPECT_SHA (full reviewed commit sha) must be exported - abort"
case "$ARM" in
  YAWAUG|VANL) ;;
  *) die "ARM '${ARM}' is not registered for exp_15 (YAWAUG|VANL) - abort" ;;
esac
case "$STEP" in ''|*[!0-9]*) die "STEP '${STEP}' must be a positive integer - abort";; esac
# ONE endpoint for the whole campaign (plan §4.1): both arms' 40,000-step
# checkpoint, at one campaign pin. A cell at any other step has no T reference,
# no seed block and no gate — it would look like registered evidence and is not.
# It is also the PRE-REGISTERED endpoint: exp_15 does no checkpoint selection, and
# admitting a nearby step would quietly turn the band caveat into a choice.
[ "$STEP" = "40000" ] || die "exp_15 is registered at the STEP=40000 endpoint only (got ${STEP}) - abort"
case "$SEED" in ''|*[!0-9]*) die "SEED '${SEED}' must be an integer - abort";; esac
case "$K" in 1|8) ;; *) die "K '${K}' must be 1 or 8 (the registered eval schedules) - abort";; esac
# EVAL_ORBIT / N_ANGLES belonged to exp_11's and exp_14's orbit campaigns and mean
# nothing here: both exp_15 arms are vanilla models. An exported leftover is
# refused rather than ignored — a recycled command line must not silently produce
# an exp_15-labelled row (announcement 05).
[ -z "${EVAL_ORBIT:-}" ] \
  || die "EVAL_ORBIT='${EVAL_ORBIT}' is not an exp_15 parameter (both arms are vanilla models) - abort"
ROTATE_MODE=fixed          # only CELL=rrob changes this
ROTATE_SEED=""             # ...and only rrob resolves one
case "$CELL" in
  tbl|rrob)
    case "$SEED" in 42|43|44|45|46) ;; *) die "a ${CELL} cell uses eval seeds 42-46 (got ${SEED}) - abort";; esac
    # Neither cell takes a fixed angle: rrob draws one per sample and tbl is the
    # theta=0 reference. Silently ignoring an exported ROTATE_DEG is exactly the
    # trap exp_14's review B4 closed on the CLI side, so it is a hard error here.
    [ -z "${ROTATE_DEG:-}" ] \
      || die "CELL=${CELL} takes no ROTATE_DEG (got '${ROTATE_DEG}'): rrob draws a per-sample yaw and tbl is the theta=0 reference; use CELL=vctl for a fixed angle - abort"
    if [ "$CELL" = "rrob" ]; then
      ROTATE_MODE=random
      ROTATE_SEED="$SEED"    # rotation seed = eval seed (plan §4.1)
    fi ;;
  vctl)
    # The two pre-registered validity controls (plan §4.1 block V). They are QA
    # readouts, not headline rows, and the tuple list is exhaustive: there is no
    # 45-degree control because exp_15 has no group structure for an off-group
    # angle to probe (that was exp_14's C4L mechanism control).
    [ "$SEED" = "42" ] || die "a vctl cell is seed 42 by contract (got ${SEED}) - abort"
    [ "$K" = "8" ] || die "a vctl cell is K=8 by contract (got K=${K}) - abort"
    [ -n "${ROTATE_DEG:-}" ] || die "CELL=vctl needs ROTATE_DEG (one of the two registered tuples) - abort"
    case "$ROTATE_DEG" in
      90|90.0) ;;
      *) die "ROTATE_DEG '${ROTATE_DEG}' is not a registered vctl angle (90) - abort" ;;
    esac
    ROTATE_DEG="${ROTATE_DEG%.0}"          # canonical integer form: 90.0 -> 90
    case "${ARM}@${ROTATE_DEG}" in
      VANL@90|YAWAUG@90) ;;
      *) die "vctl tuple ${ARM}@${ROTATE_DEG} is not one of the two registered controls: VANL@90 (gate G1), YAWAUG@90 (descriptive) (plan §4.1) - abort" ;;
    esac ;;
  *) die "CELL '${CELL}' must be tbl (theta=0 headline), rrob (random per-sample yaw) or vctl (fixed-angle validity control) - abort" ;;
esac

# --- B. arm -> config, checkpoint tree, identity ------------------------------
# The arm's identity is what its TRAINING did — augmentation on or off — and that
# is asserted inside gate D3 against the checkpoint's OWN embedded config, never
# inferred from the tree it was found in.
#
# YAWAUG's config is exp_15's own; VANL's is exp_11's, referenced in place because
# exp_15 re-measures that arm rather than retraining it.
case "$ARM" in
  YAWAUG)
    MODEL_CONFIG="$EXPDIR/FLAC_AR_YAWAUG.json"
    CKPT_DIR="${OUTPUT_ROOT}/exp15_YAWAUG/FLAC_exp15_YAWAUG/exp15_YAWAUG/checkpoints"
    LAUNCH_MANIFEST="${OUTPUT_ROOT}/exp15_YAWAUG/launch_manifest.txt"
    TRAIN_YAW_AUG=true
    ;;
  VANL)
    MODEL_CONFIG="$EXP11/FLAC_AR_VANCKPT.json"
    CKPT_DIR="${OUTPUT_ROOT}/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints"
    LAUNCH_MANIFEST="${OUTPUT_ROOT}/exp11_VANL/launch_manifest.txt"
    TRAIN_YAW_AUG=false
    ;;
esac
# Both arms are vanilla models, so this is a constant — spelled out rather than
# implied so a log can be read for protocol compliance on its own.
COND_METHOD=vanilla
ROTATE_DEG="${ROTATE_DEG:-0}"
[ -f "$MODEL_CONFIG" ] || die "model config not found: ${MODEL_CONFIG} - abort"
CONFIG_SHA="$(sha256sum "$MODEL_CONFIG" | awk '{print $1}')" || die "sha256sum failed - abort"

# The EVAL NAME and the per-class PROTOCOL FLAGS are rendered by
# exp15_validate_cell.py — the ONE definition of both — rather than duplicated in
# shell. exp_14 rendered them here and pinned them with a guard case because
# importing eval_FLAC costs ~9 s of torch per invocation; exp15_validate_cell is
# deliberately torch-free, so there is no reason to keep a second copy that merely
# agrees today. A guard case still asserts the rendered flags equal the plan's
# literal §4.2 strings.
CELL_ARGS=(--arm "$ARM" --cell "$CELL" --step "$STEP" --seed "$SEED" --k "$K")
[ "$CELL" = "vctl" ] && CELL_ARGS+=(--rotate-deg "$ROTATE_DEG")
EVAL_NAME="$(python3 "$EXPDIR/exp15_validate_cell.py" evalname "${CELL_ARGS[@]}")" \
  || die "could not render the eval name for ${ARM}/${CELL} - abort"
PROTOCOL_ARGV_LINE="$(python3 "$EXPDIR/exp15_validate_cell.py" evalargv "${CELL_ARGS[@]}")" \
  || die "could not render the cell's eval-protocol flags - abort"
read -r -a PROTOCOL_ARGV <<< "$PROTOCOL_ARGV_LINE"
[ "${#PROTOCOL_ARGV[@]}" -ge 8 ] \
  || die "the rendered protocol argv is implausibly short: '${PROTOCOL_ARGV_LINE}' - abort"

case "$K" in
  8) DATASET_CONFIG="src/configs/dataset_configs/AR/eval/acousticroom_unseeneval.json" ;;
  1) DATASET_CONFIG="src/configs/dataset_configs/AR/eval/acousticroom_unseeneval_1.json" ;;
esac
[ -f "$DATASET_CONFIG" ] || die "eval dataset config not found: ${DATASET_CONFIG} - abort"
# The yaw protocol is printed on its own line, in the same words the manifest and
# the metrics record use, so a log can be read for protocol compliance without
# reconstructing it from flags (announcement 05).
case "$CELL" in
  rrob) ROT_DESC="random per-sample yaw, rotation seed ${ROTATE_SEED}" ;;
  vctl) ROT_DESC="fixed yaw ${ROTATE_DEG} deg" ;;
  *)    ROT_DESC="theta=0 (no rotation)" ;;
esac
echo "=== exp_15 ${CELL} ${ARM} S${STEP} s${SEED} K${K} — ${TS} — host $(hostname) ==="
echo "config ${MODEL_CONFIG} sha256 ${CONFIG_SHA} | VANILLA (both arms) | train_yaw_aug ${TRAIN_YAW_AUG}" \
     "| eval-name ${EVAL_NAME}"
echo "yaw protocol: ${ROT_DESC} | split ${DATASET_CONFIG} (expect ${EXPECTED_STREAM_COUNT} items / ${EXPECTED_SCENES} scenes)"
echo "protocol argv (from exp15_validate_cell.eval_argv_tail): ${PROTOCOL_ARGV_LINE}"

# --- B2. lease: this job must hold one, and must give it back -----------------
# The submitter writes .leases/<jobid> before sbatch returns and the pruner never
# removes a leased tree (fa_orbit_measure_worktree.sh). A job that cannot find
# its own lease is running in a tree nobody reserved for it: refuse, rather than
# race a pruner that believes the tree is free.
if [ -n "${MEASURE_ROOT:-}" ] && [ -n "${SLURM_JOB_ID:-}" ]; then
  LEASE="${MEASURE_ROOT}/.leases/${SLURM_JOB_ID}"
  [ -f "$LEASE" ] || die "no lease ${LEASE} for this job — the worktree was not reserved for it - abort"
  grep -q "^jobid ${SLURM_JOB_ID}$" "$LEASE" || die "lease ${LEASE} does not name job ${SLURM_JOB_ID} - abort"
  # Release ONLY through the helper, so it always happens under the store-wide
  # lock. There is deliberately no direct-unlink fallback: an unlocked rm can
  # drop a lease while a sweep is mid-enumeration, and the sweep then deletes a
  # worktree it believes is free. If the helper cannot run, the lease simply
  # stays until the reaper proves — via a successful squeue — that this job id
  # is gone. Keeping a lease too long costs one worktree; dropping one early
  # costs a running job its code.
  MEASURE_HELPER="$EXP11/fa_orbit_measure_worktree.sh"
  trap 'bash "$MEASURE_HELPER" --release "$SLURM_JOB_ID" "$MEASURE_ROOT" \
          || echo "could not release ${LEASE}; leaving it for the reaper" >&2' EXIT
  echo "lease OK: ${LEASE}"
fi

# --- B3. the untracked runtime assets the eval resolves RELATIVELY ------------
# A fresh worktree holds only TRACKED files, but the eval dataset config names
# "AcousticRooms" and the arm configs name "weights/AGREE/AGREE_fullAR.pt", both
# relative to the code root. Without them a pinned screen dies at startup, so
# stat every one of them here rather than 20 minutes into a queue slot.
# weights/FLAC/VAE.ckpt is on this list because of exp_11's R3 incident: the
# AGREE config chain loads it, and its absence killed a whole wave.
for ASSET in AcousticRooms weights/AGREE/AGREE_fullAR.pt weights/FLAC/VAE.ckpt "$DATASET_CONFIG"; do
  [ -e "$CODE_ROOT/$ASSET" ] \
    || die "required runtime asset missing from the code root: ${CODE_ROOT}/${ASSET} — prepare the tree with fa_orbit_measure_worktree.sh - abort"
done
echo "runtime assets OK under ${CODE_ROOT} (AcousticRooms, weights/AGREE, weights/FLAC/VAE.ckpt, ${DATASET_CONFIG})"
# Existence is not identity. The AGREE conditioner is part of the measured stack,
# so it is pinned by content — a swapped-but-present file would otherwise change
# every metric silently.
PINNED_AGREE_SHA256="3a13243d6c6a11082697592c2c5db84790d37859451df2963eb51d655b23c787"
AGREE_SHA="$(sha256sum "$CODE_ROOT/weights/AGREE/AGREE_fullAR.pt" | awk '{print $1}')" \
  || die "could not hash the AGREE checkpoint - abort"
[ "$AGREE_SHA" = "$PINNED_AGREE_SHA256" ] \
  || die "weights/AGREE/AGREE_fullAR.pt sha256 ${AGREE_SHA} != pinned ${PINNED_AGREE_SHA256} - abort"
echo "AGREE pin OK: ${AGREE_SHA}"
PINNED_VAE_CKPT_SHA256="6198ff0d781fa05f546d19a0137a830c3f265437193c3581d8043713e7271baf"
VAE_CKPT_SHA="$(sha256sum "$CODE_ROOT/weights/FLAC/VAE.ckpt" | awk '{print $1}')" \
  || die "could not hash weights/FLAC/VAE.ckpt - abort"
[ "$VAE_CKPT_SHA" = "$PINNED_VAE_CKPT_SHA256" ] \
  || die "weights/FLAC/VAE.ckpt sha256 ${VAE_CKPT_SHA} != pinned ${PINNED_VAE_CKPT_SHA256} - abort"
echo "VAE.ckpt pin OK: ${VAE_CKPT_SHA}"

# The ViT init is lineage-relevant at eval time too: same pin the probe gates on.
python3 -c "
import sys; sys.path.insert(0, '$EXP11')
from fa_orbit_equiv_probe import assert_vit_pin
assert_vit_pin(); print('DINOv3 cache pin OK')" || die "DINOv3 cache pin gate FAILED - abort"

# --- C. commit binding + tracked drift ---------------------------------------
# The binding is on the CODE ROOT (the pinned worktree when one is used), so a
# main tree that keeps committing during the run is irrelevant.
HEAD_SHA="$(git -C "$CODE_ROOT" rev-parse HEAD 2>/dev/null)" || die "git rev-parse FAILED - abort"
DRIFT="$(git -C "$CODE_ROOT" status --porcelain --untracked-files=no)" \
  || die "git status FAILED - refusing to treat an error as a clean tree - abort"
if [ "$DRYRUN" = "1" ]; then
  [ "$HEAD_SHA" = "$EXPECT_SHA" ] && echo "commit binding OK (dry run): ${HEAD_SHA}" \
    || echo "DRY-RUN ADVISORY: HEAD != EXPECT_SHA (a real screen aborts here)"
else
  [ -n "${SLURM_JOB_ID:-}" ] || die "a real screen must run under sbatch - abort"
  [ "$HEAD_SHA" = "$EXPECT_SHA" ] || die "code-root HEAD ${HEAD_SHA} != EXPECT_SHA ${EXPECT_SHA} - abort"
  [ -z "$DRIFT" ] || { echo "the code root is not pristine - abort:"; echo "$DRIFT"; exit 2; }
  echo "commit binding OK: ${HEAD_SHA} (code root ${CODE_ROOT})"
fi

# --- D. exactly ONE checkpoint at this step ----------------------------------
mapfile -t CKPTS < <(ls -1 "${CKPT_DIR}"/epoch=*-step=${STEP}.ckpt 2>/dev/null)
[ "${#CKPTS[@]}" -eq 1 ] || die "expected exactly 1 checkpoint at step ${STEP} under ${CKPT_DIR}, found ${#CKPTS[@]} - abort"
CKPT="${CKPTS[0]}"
echo "checkpoint: ${CKPT}"

# --- D3. CHECKPOINT ADMISSION (plan §5, gate G4) ------------------------------
# One gate replaces exp_14's pair (its arm-lineage gate and its embedded-config
# gate), because exp_15's admission question is different in kind: exp_14 asked
# "was this trained on the C_n orbit the registry records?", exp_15 asks "is this
# the exact 40k file the chain (or the control record) attests to, and was it
# trained with — or without — the augmentation?". exp15_admit_ckpt.py RECOMPUTES
# all of it from the file (sha256 through the descriptor it loads from, embedded
# global_step, canonical embedded-config equality, the EMA<->diffusion.model.*
# suffix/shape/dtype mirror and its inventory digest, the arm's yaw_aug identity)
# and only then compares against the COMMITTED record. It also binds provenance:
# the file must sit where the record says and the run's launch manifest must still
# hash to the committed pin.
#
# Until the training chain finishes 40,000 steps, arms.YAWAUG.final_ckpt_sha256 is
# null and every YAWAUG cell is REFUSED here, by name, with the legs recorded so
# far — the pre-registered fail-closed state, not an error to work around.
# ADMIT_SOURCE_ARGS was resolved (and its production-root lock enforced) at entry.
ADMIT_ARGS=(verify --arm "$ARM" --ckpt "$CKPT" --config "$MODEL_CONFIG"
            --launch-manifest "$LAUNCH_MANIFEST" --main-repo "$MAIN_REPO"
            --expect-step "$STEP" "${ADMIT_SOURCE_ARGS[@]+"${ADMIT_SOURCE_ARGS[@]}"}")
ADMIT_OUT="$(python3 "$EXPDIR/exp15_admit_ckpt.py" "${ADMIT_ARGS[@]}" 2>&1)"; ADMIT_RC=$?
echo "$ADMIT_OUT"
[ "$ADMIT_RC" -eq 0 ] || die "CHECKPOINT ADMISSION gate FAILED (rc=${ADMIT_RC}) - abort"
# Reuse the digest the gate already computed rather than streaming 724 MB twice.
CKPT_SHA="$(printf '%s\n' "$ADMIT_OUT" | sed -n "s/^ADMITTED ${ARM} ckpt_sha256=\([0-9a-f]\{64\}\)$/\1/p" | tail -1)"
[ -n "$CKPT_SHA" ] || die "the admission gate returned rc=0 without an ADMITTED sha256 line - abort"

# --- F. the eval argv ---------------------------------------------------------
# Every protocol flag is passed EXPLICITLY, including the ones whose value equals
# an argparse default: announcement 05's rule is that eval-protocol flags are part
# of the experiment, and a recipe that relies on a default silently changes when
# the default does. The per-class half comes from PROTOCOL_ARGV (rendered above by
# the ONE definition); this block adds only the identity half, which is what the
# driver knows and the validator does not.
EVAL_ARGV=(
  --model-config "$MODEL_CONFIG"
  --dataset-config "$DATASET_CONFIG"
  --ckpt-path "$CKPT"
  --cfg-scale 1.0 --steps 1
  --seed "$SEED" --eval-name "$EVAL_NAME"
  --batch-size "$BATCH_SIZE" --num-workers "$NUM_WORKERS"
  --expected-stream-count "$EXPECTED_STREAM_COUNT" --record-stream
  --record-per-scene
  "${PROTOCOL_ARGV[@]}"
)
# The CELL-VALIDATION argv (gate I) is built by one function and printed here, so
# a DRYRUN shows both halves of what a real cell runs and a guard case can pin
# this rendering against exp15_validate_cell.check_argv — the flags that decide
# whether a finished cell is admissible must not be a second, drifting copy.
# ROTATE_DEG is passed unconditionally because the validator reads "0" as "no
# angle" (parse_deg) and omits the flag itself.
build_validate_argv() {   # <metrics path> <ckpt sha256> -> fills VALIDATE_ARGV
  local line
  line="$(python3 "$EXPDIR/exp15_validate_cell.py" argv --metrics "$1" \
            --arm "$ARM" --cell "$CELL" --step "$STEP" --seed "$SEED" --k "$K" \
            --rotate-deg "$ROTATE_DEG" --pin "$HEAD_SHA" --ckpt-sha "$2" \
            --expected-count "$EXPECTED_STREAM_COUNT" \
            --expected-scenes "$EXPECTED_SCENES")" \
    || die "could not render the cell-validation argv - abort"
  read -r -a VALIDATE_ARGV <<< "$line"
}
echo "--- eval argv ---"
echo "python eval_FLAC.py ${EVAL_ARGV[*]}"
build_validate_argv "<metrics>" "<ckpt-sha256>"
echo "--- cell-validation argv ---"
echo "python3 exp15_validate_cell.py ${VALIDATE_ARGV[*]}"
if [ "$DRYRUN" = "1" ]; then
  echo "DRY RUN complete: gates A-D3 passed for ARM=${ARM} CELL=${CELL} STEP=${STEP} SEED=${SEED} K=${K}"
  exit 0
fi

# --- G. GPU + run -------------------------------------------------------------
mapfile -t GPU_ROWS < <(nvidia-smi --query-gpu=uuid,name --format=csv,noheader,nounits) \
  || die "nvidia-smi FAILED - abort"
[ "${#GPU_ROWS[@]}" -eq 1 ] || die "expected exactly 1 visible GPU, got ${#GPU_ROWS[@]} - abort"
case "$(echo "${GPU_ROWS[0]}" | cut -d, -f2)" in *L40*) ;; *) die "not an L40 - abort";; esac
echo "GPU: ${GPU_ROWS[0]}"

# --- BEGIN screen-run-and-record (guard-tested by yaw_aug_screen_guardtests.sh) --
# The DEFAULT name is fully qualified so it never needs overriding: three of
# exp_11's overnight incidents came from a hand-supplied LOG= (a tee into the
# pinned worktree, then node-local /tmp paths that broke log provenance). It
# carries the cell, whatever distinguishes that cell (the fixed angle for vctl,
# the rotation seed for rrob) and the job id, so cells started in the same second
# — a whole submitted wave — cannot collide. It stays inside LOGDIR and keeps the
# _screen.log suffix.
CELL_TOKEN=""
case "$CELL" in
  rrob) CELL_TOKEN="_rotrand${ROTATE_SEED}" ;;
  vctl) CELL_TOKEN="_rot${ROTATE_DEG}" ;;
esac
LOG_JID="${SLURM_JOB_ID:+_jid${SLURM_JOB_ID}}"
LOG="${LOG:-${LOGDIR}/yaw_aug_${TS}_${ARM}_${CELL}${CELL_TOKEN}_S${STEP}_s${SEED}_K${K}${LOG_JID}_screen.log}"
python eval_FLAC.py "${EVAL_ARGV[@]}" 2>&1 | tee "$LOG"
PIPE_RC=("${PIPESTATUS[@]}")
EVAL_RC="${PIPE_RC[0]}"; TEE_RC="${PIPE_RC[1]}"
[ "$TEE_RC" -eq 0 ] || die "tee exited ${TEE_RC} - the screen log is not durable - abort" 7
[ "$EVAL_RC" -eq 0 ] || die "eval_FLAC.py exited ${EVAL_RC} (see ${LOG})" "$EVAL_RC"

# --- H. sidecar + result line -------------------------------------------------
# The path is DERIVED from eval_FLAC itself (never reconstructed by hand): the
# rotation-suffix rule lives there, and a cell that looked for the wrong file
# would report "metrics JSON not written" for a run that in fact succeeded.
METRICS="$(python3 - "$CKPT" "$EVAL_NAME" "$ROTATE_DEG" "$COND_METHOD" \
                    "$ROTATE_MODE" "$ROTATE_SEED" <<'PY'
import os, sys
sys.path.insert(0, os.getcwd())
from eval_FLAC import build_output_paths
mode, rseed = sys.argv[5], sys.argv[6]
print(build_output_paths(sys.argv[1], 1, 1.0, sys.argv[2], cond_method=sys.argv[4],
                         rotate_deg=float(sys.argv[3]),
                         n_angles=4, rotate_mode=mode,
                         rotate_seed=int(rseed) if rseed else None)["metrics"])
PY
)" || die "could not derive the metrics path - abort"
[ -f "$METRICS" ] || die "metrics JSON not written: ${METRICS} - abort" 5
# Recorded repo-RELATIVE: an absolute path into a pinned worktree becomes a
# dangling reference the moment that (prunable) tree is removed.
MODEL_CONFIG_REL="$(python3 -c "import os,sys; print(os.path.relpath(os.path.realpath(sys.argv[1]), sys.argv[2]))" "$MODEL_CONFIG" "$CODE_ROOT")" \
  || die "could not make the model-config path repo-relative - abort"
# The literal --frame-avg-angles value this cell carried. eval_FLAC records the
# angle list only under fa_invariant and None under vanilla, so the RECORD cannot
# witness that the flag was pinned at all — the manifest is the only artifact that
# can, and announcement 05 requires it to (plan §4.2).
FRAME_AVG_ANGLES_FLAG="$(printf '%s\n' "$PROTOCOL_ARGV_LINE" \
  | sed -n 's/.*--frame-avg-angles \([^ ]*\).*/\1/p')"
[ -n "$FRAME_AVG_ANGLES_FLAG" ] \
  || die "the rendered protocol argv carries no --frame-avg-angles - abort"
python3 - "$METRICS" "$ARM" "$STEP" "$SEED" "$K" "$EVAL_NAME" "$MODEL_CONFIG_REL" "$CONFIG_SHA" \
         "$ROTATE_DEG" "$CELL" "$COND_METHOD" "$FRAME_AVG_ANGLES_FLAG" "$TRAIN_YAW_AUG" \
         "$DATASET_CONFIG" "$CKPT" "$CKPT_SHA" "$HEAD_SHA" \
         "$ROTATE_MODE" "$ROTATE_SEED" "$EXPECTED_STREAM_COUNT" "$BATCH_SIZE" "$NUM_WORKERS" \
         <<'PY' || die "sidecar write FAILED - abort" 7
import json, os, sys
(metrics, arm, step, seed, k, eval_name, cfg, cfg_sha,
 rotate_deg, cell, cond_method, frame_flag, train_yaw_aug,
 ds, ckpt, ckpt_sha, commit,
 rotate_mode, rotate_seed, expected_count, batch_size, workers) = sys.argv[1:23]
side = {
    "arm": arm, "step": int(step), "seed": int(seed), "K": int(k), "eval_name": eval_name,
    "cfg_scale": 1.0, "steps": 1, "model_config": cfg, "model_config_sha256": cfg_sha,
    "dataset_config": ds, "ckpt_path": ckpt, "ckpt_sha256": ckpt_sha, "use_ema": True,
    # BOTH spellings, from their two different sources. `frame_avg_angles` is the
    # EFFECTIVE value, which mirrors what eval_FLAC wrote into the metrics record
    # (None under vanilla, whatever the flag said); `frame_avg_angles_flag` is the
    # LITERAL command-line value this cell pinned. A manifest carrying only the
    # first could not witness that the flag was passed at all.
    "frame_avg_angles": None,
    "frame_avg_angles_flag": frame_flag,
    "cond_method": cond_method, "cond_autocast": "bf16", "commit": commit,
    # exp_15's arm axis: what the TRAINING did. exp_14 recorded a training/eval
    # ORBIT here; exp_15 has none, and recording one anyway would invite a reader
    # to compare two campaigns' rows on a field that means different things.
    "cell": cell, "train_yaw_aug": train_yaw_aug == "true",
    # THE YAW PROTOCOL (announcement 05). The two angle fields are deliberately
    # exclusive and mirror eval_FLAC's own record: a fixed cell has an angle and no
    # seed, a random cell has a seed and rotate_deg null. An unrotated cell and a
    # randomly-rotated one must never be readable as the same protocol.
    "rotate_mode": rotate_mode,
    "rotate_deg": None if rotate_mode == "random" else float(rotate_deg),
    "rotate_seed": int(rotate_seed) if rotate_seed else None,
    # ...and the runtime pins the validator re-checks per cell (plan §4.2).
    "expected_stream_count": int(expected_count),
    "record_stream": True,
    "record_per_scene": True,
    "stream_sidecar": os.path.basename(metrics)[:-len(".json")] + ".stream.json",
    "batch_size": int(batch_size), "num_workers": int(workers),
}
# ATOMIC PUBLISH: tmp + os.replace in the SAME directory, so a reader — the wave
# submitter's dedup, the collector — never sees a half-written manifest and
# mistakes a truncated file for a malformed cell.
out = metrics + ".screenmeta.json"
tmp = out + ".tmp"
with open(tmp, "w") as fh:
    json.dump(side, fh, indent=2)
    fh.flush()
    os.fsync(fh.fileno())
os.replace(tmp, out)
print("sidecar:", out)
PY

build_validate_argv "$METRICS" "$CKPT_SHA"

# --- I. the validator is a GATE, not advice ----------------------------------
# A cell that cannot be validated must not announce a result: SCREENRESULT is what
# the wave submitter's dedup and the collector consume, and a cell that "looks
# done" but fails validation is exactly the fail-open case exp_14's review B6
# closed.
#
# ONE exp_15 cell is a complete self-contained artifact — the metrics record plus
# its .stream.json assignment audit — so it can and must be validated the moment
# it lands. Cross-cell equalities (T<->R pairing, cross-arm rotation matching,
# 5/5 seed blocks) belong to the collector and are deliberately NOT claimed here.
python3 "$EXPDIR/exp15_validate_cell.py" "${VALIDATE_ARGV[@]}" \
  || die "cell validation FAILED - refusing to emit SCREENRESULT for an unprovable cell" 6
VALIDATED="cell-valid"

echo "SCREENRESULT arm=${ARM} cell=${CELL} step=${STEP} seed=${SEED} k=${K} rc=${EVAL_RC} validated=${VALIDATED} train_yaw_aug=${TRAIN_YAW_AUG} rotate_mode=${ROTATE_MODE} rotate_seed=${ROTATE_SEED:-<n/a>} rotate_deg=${ROTATE_DEG} eval_name=${EVAL_NAME} metrics=${METRICS} sidecar=${METRICS}.screenmeta.json stream=${METRICS%.json}.stream.json config_sha256=${CONFIG_SHA} ckpt_sha256=${CKPT_SHA} commit=${HEAD_SHA} log=${LOG}"
# --- END screen-run-and-record -------------------------------------------------
exit 0
